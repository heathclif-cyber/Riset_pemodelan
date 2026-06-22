"""
pipeline/09_train_meta_lgbm_tb.py — Train binary meta-model.

Input : models/runs/tb_meta_v1/oof_meta_dataset.parquet
Target: win=1 (trade akan profit) / win=0 (akan loss)
Model : LGBM binary, purged CV

Output: models/runs/tb_meta_v1/meta_lgbm.pkl
"""
import json, sys, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import joblib, lightgbm as lgb
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

from pipeline.shared import build_purged_folds
from core.utils import setup_logger
from config import *

logger = setup_logger("09_meta_lgbm")

RUN_META = "tb_meta_v1"
META_DIR = MODEL_DIR / "runs" / RUN_META

# Semua feature yang tersedia di meta-dataset
META_FEATURES = [
    "p_short", "p_flat", "p_long", "confidence", "direction",
    "atr_percentile_h1", "funding_rate", "wyckoff_phase",
    "stochrsi_k", "ofi_h4_delta", "Sell_Liq",
]

# Binary LGBM params
META_PARAMS = {
    "objective": "binary",
    "n_estimators": 400, "learning_rate": 0.03,
    "max_depth": 4, "num_leaves": 15, "min_child_samples": 50,
    "subsample": 0.8, "colsample_bytree": 0.7,
    "verbose": -1, "n_jobs": -1, "random_state": 42,
}
EARLY_STOPPING = 50


def main():
    print(f"\n{'='*60}")
    print(f"  META-MODEL TRAINING — {RUN_META}")
    print(f"  Target: win=1/0 (binary)")
    print(f"  Features: {len(META_FEATURES)}")
    print(f"{'='*60}\n")

    # Load OOF meta-dataset
    meta_path = META_DIR / "oof_meta_dataset.parquet"
    if not meta_path.exists():
        raise FileNotFoundError(f"Run 08_generate_meta_labels_tb.py first: {meta_path}")

    meta_df = pd.read_parquet(meta_path)
    logger.info(f"Loaded {len(meta_df):,} OOF trades | "
                f"WIN={meta_df['win'].sum():,} ({meta_df['win'].mean()*100:.1f}%)")

    avail = [c for c in META_FEATURES if c in meta_df.columns]
    missing = [c for c in META_FEATURES if c not in meta_df.columns]
    if missing:
        logger.warning(f"Missing features: {missing}")

    X = meta_df[avail].ffill().fillna(0)
    y = meta_df["win"].values.astype(np.int32)

    logger.info(f"Features used: {avail}")
    logger.info(f"Class balance: WIN={y.mean()*100:.1f}% LOSS={(1-y.mean())*100:.1f}%")

    # Purged CV
    folds = build_purged_folds(X.index, N_FOLDS, PURGE_GAP_BARS)
    cv_metrics = []
    best_auc, best_model, best_fold = -1.0, None, -1

    for fold, (tr_idx, val_idx) in enumerate(folds, 1):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]

        if len(np.unique(y_tr)) < 2 or len(np.unique(y_val)) < 2:
            logger.warning(f"  Fold {fold}: skipped (single class)")
            continue

        model = lgb.LGBMClassifier(**META_PARAMS)
        model.fit(X_tr, y_tr,
                  eval_set=[(X_val, y_val)],
                  callbacks=[lgb.early_stopping(EARLY_STOPPING, verbose=False),
                             lgb.log_evaluation(period=-1)])

        p_val  = model.predict_proba(X_val)[:, 1]
        y_pred = (p_val >= 0.5).astype(int)
        auc    = float(roc_auc_score(y_val, p_val))
        acc    = float(accuracy_score(y_val, y_pred))
        # WR quando meta-model aceita (p_win >= 0.5)
        sel_mask = p_val >= 0.5
        wr_selected = y_val[sel_mask].mean() if sel_mask.sum() > 0 else 0.0
        wr_rejected = y_val[~sel_mask].mean() if (~sel_mask).sum() > 0 else 0.0

        cv_metrics.append({
            "fold": fold, "n_train": len(X_tr), "n_val": len(X_val),
            "best_iteration": model.best_iteration_,
            "auc": round(auc, 4), "accuracy": round(acc, 4),
            "wr_selected": round(float(wr_selected), 4),
            "wr_rejected": round(float(wr_rejected), 4),
            "n_selected": int(sel_mask.sum()),
        })

        if auc > best_auc:
            best_auc, best_model, best_fold = auc, model, fold

        logger.info(f"  Fold {fold}: AUC={auc:.4f} Acc={acc:.4f} | "
                    f"Selected={sel_mask.sum():,} WR={wr_selected*100:.1f}% | "
                    f"Rejected WR={wr_rejected*100:.1f}% | Iter={model.best_iteration_}")
        del model

    if not cv_metrics:
        raise RuntimeError("No valid folds")

    # Full retrain
    avg_iter = max(int(np.mean([m["best_iteration"] for m in cv_metrics])), 50)
    logger.info(f"CV done. Avg iter: {avg_iter} | Best Fold: {best_fold} (AUC={best_auc:.4f})")

    final_params = META_PARAMS.copy()
    final_params["n_estimators"] = avg_iter
    final_model = lgb.LGBMClassifier(**final_params)
    final_model.fit(X, y)

    # Save
    model_path = META_DIR / "meta_lgbm.pkl"
    joblib.dump(final_model, model_path)

    with open(META_DIR / f"{RUN_META}_features.json", "w") as f:
        json.dump(avail, f, indent=2)

    mean_auc  = float(np.mean([m["auc"] for m in cv_metrics]))
    mean_wr_sel = float(np.mean([m["wr_selected"] for m in cv_metrics]))
    mean_wr_rej = float(np.mean([m["wr_rejected"] for m in cv_metrics]))

    summary = {
        "run_name": RUN_META, "model_type": "binary_meta_lgbm",
        "n_samples": len(X), "base_win_rate": round(float(y.mean()), 4),
        "n_features": len(avail), "features": avail,
        "cv_mean_auc": round(mean_auc, 4),
        "cv_mean_wr_selected": round(mean_wr_sel, 4),
        "cv_mean_wr_rejected": round(mean_wr_rej, 4),
        "best_fold": best_fold, "best_auc": round(best_auc, 4),
        "final_n_estimators": avg_iter,
        "folds": cv_metrics,
        "trained_at": datetime.now().isoformat(),
    }
    with open(META_DIR / f"{RUN_META}_train_meta.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Feature importance
    imp = sorted(zip(avail, final_model.feature_importances_),
                 key=lambda x: x[1], reverse=True)

    print(f"\n{'='*60}")
    print(f"  META-MODEL TRAINING COMPLETE")
    print(f"  CV AUC: {mean_auc:.4f} | WR selected: {mean_wr_sel*100:.1f}% | "
          f"WR rejected: {mean_wr_rej*100:.1f}%")
    print(f"  Model: {model_path}")
    print(f"\n  Top features:")
    for feat, score in imp[:8]:
        print(f"    {feat:<30} {score:>8.1f}")
    print(f"\n  {'Fold':>4}  {'AUC':>7}  {'WR sel':>8}  {'WR rej':>8}  {'n_sel':>7}  {'Iter':>6}")
    print("  " + "-"*50)
    for m in cv_metrics:
        print(f"  {m['fold']:>4}  {m['auc']:>7.4f}  {m['wr_selected']*100:>7.1f}%  "
              f"{m['wr_rejected']*100:>7.1f}%  {m['n_selected']:>7,}  {m['best_iteration']:>6}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
