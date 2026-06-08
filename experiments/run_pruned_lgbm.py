"""
experiments/run_pruned_lgbm.py

Simple script to retrain LGBM with a pruned feature list.
Used for Experiment 1: Feature Pruning on cascade_v2.5_hybrid.

Usage:
    python experiments/run_pruned_lgbm.py --run-id cascade_v2.5_hybrid_pruned --pruned-json experiments/cascade_v2.5_hybrid_pruned_features.json
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix

from core.utils import setup_logger

logger = setup_logger("run_pruned_lgbm")

from config import (
    ALL_COINS, PROC_DIR, LABEL_DIR, TRAIN_CUTOFF_DATE,
    N_FOLDS, PURGE_GAP_BARS, MODEL_DIR,
    LGBM_PARAMS, LGBM_CLASS_WEIGHTS, LGBM_EARLY_STOPPING,
    LABEL_MAP
)
from pipeline.shared import build_purged_folds

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--pruned-json", required=True, help="Path to pruned_features.json")
    args = parser.parse_args()

    run_dir = MODEL_DIR / "runs" / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # Load pruned feature list
    with open(args.pruned_json) as f:
        prune_info = json.load(f)

    kept_features = prune_info["kept_features"]
    print(f"Using {len(kept_features)} features (removed {prune_info['n_removed']})")

    # Load data - prioritize fresh _features_v3.parquet from LABEL_DIR
    print("Loading data...")
    all_dfs = []
    for coin in ALL_COINS:
        # Try fresh engineered data first (from 03_engineer)
        path_v3 = LABEL_DIR / f"{coin}_features_v3.parquet"
        path_old = PROC_DIR / f"{coin}_engineered.parquet"

        if path_v3.exists():
            path = path_v3
        elif path_old.exists():
            path = path_old
        else:
            print(f"  WARNING: No engineered file found for {coin}")
            continue

        df = pd.read_parquet(path)
        df = df[df.index < TRAIN_CUTOFF_DATE]
        if "label" not in df.columns:
            print(f"  WARNING: No label column in {coin}")
            continue

        keep = [c for c in kept_features if c in df.columns] + ["label"]
        df = df[keep].dropna()
        if len(df) > 0:
            all_dfs.append(df)
            print(f"  Loaded {coin}: {len(df):,} rows from {path.name}")
        else:
            print(f"  WARNING: No valid rows for {coin} after filtering")

    if not all_dfs:
        raise ValueError("No data could be loaded for any coin. Check if 03_engineer.py has been run.")

    combined = pd.concat(all_dfs).sort_index()

    # Encode labels to integers (same as original 04_train_lgbm.py)
    mask = combined["label"].isin(LABEL_MAP)
    if (~mask).sum():
        logger.warning(f"Dropping {(~mask).sum():,} rows with unknown labels.")
        combined = combined[mask].copy()

    X = combined[kept_features]
    y = combined["label"].map(LABEL_MAP).astype(np.int32).values

    print(f"\nData loaded: {len(X):,} samples, {len(kept_features)} features")

    # Run purged CV
    print("Running purged CV...")
    folds = build_purged_folds(X.index, n_folds=N_FOLDS, purge=PURGE_GAP_BARS)

    all_metrics = []
    for i, (tr_idx, val_idx) in enumerate(folds):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]

        sample_w = np.array([LGBM_CLASS_WEIGHTS[int(label)] for label in y_tr])

        params = LGBM_PARAMS.copy()
        model = lgb.LGBMClassifier(**params)
        model.fit(
            X_tr, y_tr,
            sample_weight=sample_w,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(LGBM_EARLY_STOPPING, verbose=False)]
        )

        y_pred = model.predict(X_val)
        f1 = f1_score(y_val, y_pred, average="macro", zero_division=0)

        metrics = {
            "fold": i + 1,
            "n_train": len(X_tr),
            "n_val": len(X_val),
            "best_iteration": model.best_iteration_,
            "f1_macro": round(f1, 4),
            "accuracy": round(accuracy_score(y_val, y_pred), 4),
        }
        all_metrics.append(metrics)
        print(f"  Fold {i+1}: F1={f1:.4f} | best_iter={model.best_iteration_}")

    # Save results
    with open(run_dir / "lgbm_pruned_cv_results.json", "w") as f:
        json.dump({
            "run_id": args.run_id,
            "pruned_from": prune_info,
            "folds": all_metrics
        }, f, indent=2)

    print(f"\nPruned CV results saved to {run_dir / 'lgbm_pruned_cv_results.json'}")

if __name__ == "__main__":
    main()
