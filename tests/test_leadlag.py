import numpy as np
import pandas as pd

from leadlag import lead_lag


def test_detects_known_lead():
    """If b is a is delayed by 2 seconds, a leads b and the peak is at +2."""
    rng = np.random.default_rng(0)
    a = pd.Series(rng.standard_normal(3000), index=range(3000))
    b = a.shift(2)
    ll = lead_lag(a, b, max_lag=5)
    assert max(ll, key=lambda k: ll[k]) == 2


def test_symmetric_when_independent():
    rng = np.random.default_rng(1)
    a = pd.Series(rng.standard_normal(3000), index=range(3000))
    b = pd.Series(rng.standard_normal(3000), index=range(3000))
    ll = lead_lag(a, b, max_lag=5)
    assert abs(ll[0]) < 0.1  # unrelated series: no meaningful correlation
