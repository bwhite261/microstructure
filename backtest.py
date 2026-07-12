"""Does the signal survive trading costs?

Trains on the first 80% of bars, trades the last 20%: go long when the model's
p(up) clears a threshold, short when p(down) does, hold for the horizon, no
overlapping positions. Each trade is priced three ways:

  mid    - mid-to-mid, no costs (is there any signal at all?)
  spread - buy at the ask, unwind at the bid (can it cross the spread?)
  fees   - spread plus taker fees both ways (can a retail taker profit?)

    python backtest.py data/features-BTC-USD.csv --horizon 5
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from models import FEATURES, direction_dataset

def run(df: pd.DataFrame, symbol: str, horizon: int, threshold: float, fee_bps: float,
        out_dir: Path) -> None:
    target = f"fwd_ret_{horizon}s"
    df = df.dropna(subset=FEATURES + [target]).copy()
    for col in ("best_bid", "best_ask", "mid"):
        df[f"fwd_{col}"] = df[col].reindex(df.index + horizon).to_numpy()
    df = df.dropna(subset=["fwd_best_bid", "fwd_best_ask"])

    split = int(len(df) * 0.8)
    train, test = df.iloc[:split], df.iloc[split:]
    X_train, y_train = direction_dataset(train, horizon)
    model = HistGradientBoostingClassifier(random_state=0).fit(X_train, y_train)
    prob = model.predict_proba(test[FEATURES].to_numpy())[:, 1]

    trades, busy_until = [], 0
    for (ts, row), p in zip(test.iterrows(), prob):
        if ts < busy_until:
            continue
        if p > threshold:
            side = 1
        elif p < 1 - threshold:
            side = -1
        else:
            continue
        busy_until = ts + horizon
        mid_bps = side * (row["fwd_mid"] / row["mid"] - 1) * 1e4
        if side == 1:  # buy the ask now, sell the bid at exit
            exec_bps = (row["fwd_best_bid"] / row["best_ask"] - 1) * 1e4
        else:  # sell the bid now, buy the ask back at exit
            exec_bps = (row["best_bid"] / row["fwd_best_ask"] - 1) * 1e4
        trades.append({"ts": ts, "side": side, "mid": mid_bps,
                       "spread": exec_bps, "fees": exec_bps - 2 * fee_bps})

    hours = (test.index[-1] - test.index[0]) / 3600
    print(f"\n{symbol} {horizon}s horizon, threshold {threshold}, "
          f"{len(trades)} trades in {hours:.1f}h of test data")
    if not trades:
        return
    t = pd.DataFrame(trades)
    for scenario in ("mid", "spread", "fees"):
        r = t[scenario]
        print(f"  {scenario:7s} hit rate {(r > 0).mean():.1%},  "
              f"avg {r.mean():+.2f} bps/trade,  total {r.sum():+.0f} bps")

    fig, ax = plt.subplots(figsize=(8, 4))
    for scenario, label in (("mid", "mid-to-mid (no costs)"),
                            ("spread", "crossing the spread"),
                            ("fees", f"spread + {fee_bps:.0f} bps taker fee/side")):
        ax.plot((t["ts"] - t["ts"].iloc[0]) / 60, t[scenario].cumsum(), label=label)
    ax.set_xlabel("test period (minutes)")
    ax.set_ylabel("cumulative return (bps)")
    ax.set_title(f"{symbol}, {horizon}s horizon")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = out_dir / f"backtest-{symbol.replace('/', '-')}-{horizon}s.png"
    fig.savefig(out, dpi=120)
    print(f"  equity curves -> {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("features", nargs="*", type=Path)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--threshold", type=float, default=0.6)
    parser.add_argument("--fee-bps", type=float, default=26.0, help="Kraken base-tier taker fee")
    parser.add_argument("--out-dir", type=Path, default=Path("results"))
    args = parser.parse_args()
    paths = args.features or sorted(Path("data").glob("features-*.csv"))
    args.out_dir.mkdir(exist_ok=True)

    for path in paths:
        symbol = path.stem.removeprefix("features-").replace("-", "/", 1)
        df = pd.read_csv(path, index_col="ts")
        run(df, symbol, args.horizon, args.threshold, args.fee_bps, args.out_dir)


if __name__ == "__main__":
    main()
