"""Walk-forward evaluation of short-horizon direction models.

The label is the sign of the forward mid return (flat bars dropped). Splits
are chronological (expanding train window, test on the future) — shuffling
would leak information both ways in an autocorrelated series.

    python models.py data/features-BTC-USD.csv
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

FEATURES = ["spread_bps", "imb1", "imb5", "imb10", "ofi",
            "trade_flow", "trade_vol", "n_updates", "ret_1s", "vol_60s"]
HORIZONS = (1, 5, 30)


def make_models():
    return {
        "logistic": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
        "gbdt": HistGradientBoostingClassifier(random_state=0),
    }


FEATURE_GROUPS = {
    "book (imbalance+spread+ofi)": ["spread_bps", "imb1", "imb5", "imb10", "ofi"],
    "imb1 alone": ["imb1"],
    "ret_1s alone (momentum)": ["ret_1s"],
}


def direction_dataset(df: pd.DataFrame, horizon: int):
    """X, y for 'given the mid moves in the next h seconds, which way?'"""
    target = f"fwd_ret_{horizon}s"
    data = df.dropna(subset=FEATURES + [target])
    data = data[data[target] != 0]
    return data[FEATURES].to_numpy(), (data[target] > 0).to_numpy()


def attribution(df: pd.DataFrame, symbol: str, horizon: int, embargo: int) -> None:
    """Which features carry the signal? Guards against a momentum artifact:
    if book imbalance drives AUC and the last return alone does not, the signal
    is genuine microstructure, not autocorrelation."""
    X, y = direction_dataset(df, horizon)
    splitter = TimeSeriesSplit(5, gap=embargo)
    print(f"{symbol} {horizon}s feature attribution:")
    for label, feats in FEATURE_GROUPS.items():
        cols = [FEATURES.index(f) for f in feats]
        aucs = []
        for tr, te in splitter.split(X):
            if len(np.unique(y[te])) < 2:
                continue
            model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
            model.fit(X[tr][:, cols], y[tr])
            aucs.append(roc_auc_score(y[te], model.predict_proba(X[te][:, cols])[:, 1]))
        if aucs:
            print(f"  {label:30s} AUC {np.mean(aucs):.3f}")


def evaluate(df: pd.DataFrame, symbol: str, n_splits: int, embargo: int) -> list[dict]:
    results = []
    for horizon in HORIZONS:
        X, y = direction_dataset(df, horizon)
        if len(y) < 10 * n_splits:
            print(f"{symbol} {horizon}s: only {len(y)} samples, skipping")
            continue
        print(f"{symbol} {horizon}s: n={len(y):,}, base rate {y.mean():.1%} up")
        splitter = TimeSeriesSplit(n_splits, gap=embargo)
        for name, model in make_models().items():
            aucs, accs = [], []
            for train_idx, test_idx in splitter.split(X):
                if len(np.unique(y[test_idx])) < 2:
                    continue  # degenerate fold, AUC undefined
                model.fit(X[train_idx], y[train_idx])
                prob = model.predict_proba(X[test_idx])[:, 1]
                aucs.append(roc_auc_score(y[test_idx], prob))
                accs.append(accuracy_score(y[test_idx], prob > 0.5))
            if not aucs:
                continue
            print(f"  {name:9s} AUC {np.mean(aucs):.3f} ± {np.std(aucs):.3f},  "
                  f"acc {np.mean(accs):.1%}")
            results.append({"symbol": symbol, "horizon_s": horizon, "model": name,
                            "n": len(y), "auc_mean": np.mean(aucs), "auc_std": np.std(aucs),
                            "acc_mean": np.mean(accs)})
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("features", nargs="*", type=Path)
    parser.add_argument("--splits", type=int, default=5)
    parser.add_argument("--embargo", type=int, default=30,
                        help="bars purged between train and test to prevent adjacency leakage")
    parser.add_argument("--out-dir", type=Path, default=Path("results"))
    args = parser.parse_args()
    paths = args.features or sorted(Path("data").glob("features-*.csv"))

    all_results = []
    for path in paths:
        symbol = path.stem.removeprefix("features-").replace("-", "/", 1)
        df = pd.read_csv(path, index_col="ts")
        all_results += evaluate(df, symbol, args.splits, args.embargo)
        attribution(df, symbol, 5, args.embargo)

    args.out_dir.mkdir(exist_ok=True)
    pd.DataFrame(all_results).to_csv(args.out_dir / "metrics.csv", index=False)
    print(f"wrote {args.out_dir / 'metrics.csv'}")


if __name__ == "__main__":
    main()
