"""Lead-lag analysis between two assets' 1-second return series.

Three questions, answered on the recorded feature bars:
  1. Co-movement  - how correlated are the two assets' returns in the same second?
  2. Lead-lag     - does one asset's move predict the other's next move?
  3. Cross-asset  - does adding the leader's features improve a direction model
                    for the follower?

Reads the CSVs written by features.py, so re-running after features.py folds in
newly recorded data.

    python leadlag.py BTC/USD ETH/USD
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from models import FEATURES, cv_auc, direction_dataset

# what the leader lends the follower: its recent return, order flow, and imbalance
CROSS_FEATURES = ["ret_1s", "ofi", "imb1"]


def load(symbol: str, data_dir: Path) -> pd.DataFrame:
    return pd.read_csv(data_dir / f"features-{symbol.replace('/', '-')}.csv", index_col="ts")


def lead_lag(a: pd.Series, b: pd.Series, max_lag: int) -> dict[int, float]:
    """corr(a_t, b_{t+k}) for k in [-max_lag, max_lag].

    A peak at k > 0 means a's current return tracks b's future return, i.e. a
    leads b by k seconds. Both series are reindexed onto a gap-free second grid
    first, so shifting by k rows is exactly a k-second shift.
    """
    grid = range(min(a.index.min(), b.index.min()), max(a.index.max(), b.index.max()) + 1)
    a, b = a.reindex(grid), b.reindex(grid)
    return {k: a.corr(b.shift(-k)) for k in range(-max_lag, max_lag + 1)}


def logistic():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))


def cross_asset_gain(follower: pd.DataFrame, leader: pd.DataFrame, horizon: int, embargo: int):
    """Follower-direction AUC with its own features vs. those plus the leader's,
    evaluated on identical rows. The leader's values at time t are known at t, so
    this is a fair (no-lookahead) test of whether cross-asset information helps.
    Returns (own_auc, augmented_auc)."""
    lead = leader[CROSS_FEATURES].add_prefix("lead_")
    both = FEATURES + list(lead.columns)
    X, y = direction_dataset(follower.join(lead), horizon, both)
    own = cv_auc(logistic(), X[:, :len(FEATURES)], y, 5, embargo)  # leading cols are FEATURES
    augmented = cv_auc(logistic(), X, y, 5, embargo)
    return own[0], augmented[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("leader", nargs="?", default="BTC/USD")
    parser.add_argument("follower", nargs="?", default="ETH/USD")
    parser.add_argument("--max-lag", type=int, default=5)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--embargo", type=int, default=30)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    args = parser.parse_args()

    a, b = load(args.leader, args.data_dir), load(args.follower, args.data_dir)
    ll = lead_lag(a["ret_1s"], b["ret_1s"], args.max_lag)

    print(f"{args.leader} vs {args.follower}")
    print(f"contemporaneous return correlation (lag 0): {ll[0]:.3f}\n")
    print("cross-correlation by lag (k>0 = leader leads):")
    for k in sorted(ll):
        print(f"  {k:+d}s: {ll[k]:+.3f}")
    fwd = np.mean([ll[k] for k in range(1, args.max_lag + 1)])
    back = np.mean([ll[k] for k in range(-args.max_lag, 0)])
    verdict = args.leader if fwd > back else args.follower if back > fwd else "neither"
    print(f"positive-lag mean {fwd:+.3f} vs negative-lag mean {back:+.3f} -> {verdict} leads\n")

    own, augmented = cross_asset_gain(b, a, args.horizon, args.embargo)
    print(f"{args.follower} {args.horizon}s direction AUC:")
    print(f"  own book only            {own:.3f}")
    print(f"  + {args.leader} features  {augmented:.3f}  ({augmented - own:+.3f})")


if __name__ == "__main__":
    main()
