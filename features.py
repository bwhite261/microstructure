"""Build 1-second feature bars from recorded book and trade events.

Bar t aggregates events with timestamps in [t, t+1) and carries the book state
as of the last update in the bar, so every feature is known by the bar close.
Targets are forward log mid-price returns at several horizons.

    python features.py data/*.jsonl.gz
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from book import replay

HORIZONS = (1, 5, 30)
DEPTHS = (1, 5, 10)


def ofi_term(bid, ask, prev_bid, prev_ask) -> float:
    """One event's order-flow imbalance contribution (Cont, Kukanov & Stoikov).

    Best-level quantity counts as arriving flow when its price improves or
    holds, and as departing flow when it retreats.
    """
    (bp, bq), (ap, aq) = bid, ask
    (pbp, pbq), (pap, paq) = prev_bid, prev_ask
    e = 0.0
    if bp >= pbp:
        e += float(bq)
    if bp <= pbp:
        e -= float(pbq)
    if ap <= pap:
        e -= float(aq)
    if ap >= pap:
        e += float(paq)
    return e


class BarBuilder:
    """Accumulates one symbol's events into 1-second bars."""

    def __init__(self):
        self.rows: list[dict] = []
        self.bar_ts = None
        self.close_state: dict | None = None
        self.prev_bid = self.prev_ask = None
        self._reset()

    def _reset(self):
        self.ofi = 0.0
        self.buy_vol = 0.0
        self.sell_vol = 0.0
        self.n_updates = 0

    def _roll(self, ts: float):
        second = int(ts)
        if self.bar_ts is None:
            self.bar_ts = second
        elif second > self.bar_ts:
            self._flush()
            self.bar_ts = second

    def _flush(self):
        if self.close_state is not None:
            self.rows.append({
                "ts": self.bar_ts, **self.close_state,
                "ofi": self.ofi, "trade_flow": self.buy_vol - self.sell_vol,
                "trade_vol": self.buy_vol + self.sell_vol, "n_updates": self.n_updates,
            })
        self._reset()

    def on_book(self, ts: float, book):
        self._roll(ts)
        bid, ask = book.best_bid(), book.best_ask()
        if self.prev_bid is not None:
            self.ofi += ofi_term(bid, ask, self.prev_bid, self.prev_ask)
        self.prev_bid, self.prev_ask = bid, ask
        self.n_updates += 1
        mid = float(bid[0] + ask[0]) / 2
        state = {"mid": mid, "best_bid": float(bid[0]), "best_ask": float(ask[0]),
                 "spread_bps": float(ask[0] - bid[0]) / mid * 1e4}
        for d in DEPTHS:
            b, a = float(book.depth_qty("bids", d)), float(book.depth_qty("asks", d))
            state[f"imb{d}"] = (b - a) / (b + a)
        self.close_state = state

    def on_trade(self, ts: float, trade: dict):
        self._roll(ts)
        if trade["side"] == "buy":
            self.buy_vol += float(trade["qty"])
        else:
            self.sell_vol += float(trade["qty"])

    def to_frame(self) -> pd.DataFrame:
        self._flush()
        df = pd.DataFrame(self.rows).set_index("ts")
        df["ret_1s"] = np.log(df["mid"]).diff()
        df["vol_60s"] = df["ret_1s"].rolling(60, min_periods=30).std()
        for h in HORIZONS:
            fwd_mid = df["mid"].reindex(df.index + h)
            df[f"fwd_ret_{h}s"] = np.log(fwd_mid.values) - np.log(df["mid"].values)
        return df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recordings", nargs="*", type=Path)
    parser.add_argument("--out-dir", type=Path, default=Path("data"))
    args = parser.parse_args()
    paths = sorted(args.recordings) or sorted(Path("data").glob("*.jsonl.gz"))

    builders: dict[str, BarBuilder] = defaultdict(BarBuilder)
    for ts, kind, payload in replay(paths):
        if kind == "book":
            builders[payload.symbol].on_book(ts, payload)
        else:
            builders[payload["symbol"]].on_trade(ts, payload)

    for symbol, builder in builders.items():
        df = builder.to_frame()
        out = args.out_dir / f"features-{symbol.replace('/', '-')}.csv"
        df.to_csv(out)
        hours = (df.index[-1] - df.index[0]) / 3600
        up = [f"{(df[f'fwd_ret_{h}s'] > 0).mean():.1%} up at {h}s" for h in HORIZONS]
        print(f"{symbol}: {len(df):,} bars over {hours:.1f}h -> {out} ({', '.join(up)})")


if __name__ == "__main__":
    main()
