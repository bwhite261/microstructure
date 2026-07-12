from decimal import Decimal

import numpy as np

from features import BarBuilder, ofi_term

D = Decimal


def test_ofi_bid_qty_increase_is_buying_pressure():
    # same prices, more qty at the best bid
    e = ofi_term(bid=(D(100), D(5)), ask=(D(101), D(2)),
                 prev_bid=(D(100), D(3)), prev_ask=(D(101), D(2)))
    assert e == 5 - 3  # bid qty in, prev bid qty out, ask side cancels


def test_ofi_bid_price_improvement():
    # bid steps up: all new bid qty counts in, old level's qty doesn't count out
    e = ofi_term(bid=(D(101), D(4)), ask=(D(102), D(2)),
                 prev_bid=(D(100), D(3)), prev_ask=(D(102), D(2)))
    assert e == 4


def test_ofi_ask_price_drop_is_selling_pressure():
    e = ofi_term(bid=(D(100), D(3)), ask=(D(101), D(6)),
                 prev_bid=(D(100), D(3)), prev_ask=(D(102), D(2)))
    assert e == -6


class FakeBook:
    def __init__(self, bid, ask):
        self._bid, self._ask = bid, ask
        self.symbol = "TEST/USD"

    def best_bid(self):
        return self._bid

    def best_ask(self):
        return self._ask

    def depth_qty(self, side, n):
        return (self._bid if side == "bids" else self._ask)[1]


def test_bars_and_forward_returns():
    builder = BarBuilder()
    # one book update per second, mid drifting up 100 -> 103
    for t in range(4):
        mid = 100 + t
        builder.on_book(float(t), FakeBook((D(mid) - D("0.5"), D(1)), (D(mid) + D("0.5"), D(1))))
    df = builder.to_frame()
    assert list(df.index) == [0, 1, 2, 3]
    assert df.loc[0, "mid"] == 100.0  # bar 0 closes on its own state, not bar 1's
    expected = np.log(df.loc[1, "mid"] / df.loc[0, "mid"])
    assert abs(df.loc[0, "fwd_ret_1s"] - expected) < 1e-12
    assert df["fwd_ret_1s"].isna().iloc[-1]  # no future bar to label the last one
