"""
pipeline/09_train_meta_lgbm_ic32.py — Train binary meta-model on ic32 OOF trades.

Prerequisite: python pipeline/08_generate_ic32_oof_trades.py
Spec: pipeline/meta_label_spec.json

Output: models/runs/ic32_meta_v1/meta_lgbm.pkl
"""
import json
import sys
import warnings
from datetime import datetime
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from config import MODEL_DIR, N_FOLDS, PURGE_GAP_BARS
from core.meta_labeling import load_spec
from core.utils import setup_logger
from pipeline.shared import build_purged_folds

logger = setup_logger("09_meta_ic32")

RUN_NAME = "ic32_meta_v1"
RUN_DIR = MODEL_DIR / "runs" / RUN_NAME
OOF_PATH = ROOT / "data" / "meta_labels" / "ic32_oof_trades.parquet"
SPEC = load_spec()
META_FEATURES = SPEC["meta"]["features"]

META_PARAMS = {
    "objective": "binary",
    "n_estimators": 400,
    "learning_rate": 0.03,
    "max_depth": 4,
    "num_leaves": 15,
    "min_child_samples": 50,
    "subsample": 0.8,
    "colsample_bytree": 0.7,
    "verbose": -1,
    "n_jobs": -1,
    "random_state": 42,
}
EARLY_STOPPING = 50


def main():
    print(f"\n{'='*60}")
    print(f"  META-MODEL TRAINING — {RUN_NAME}")
    print(f"  Input : {OOF_PATH.name}")
    print(f"  Target: win=1/0 (OOF trade outcome)")
    print(f"{'='*60}\n")

    if not OOF_PATH.exists():
        raise FileNotFoundError(f"Run 08_generate_ic32_oof_trades.py first: {OOF_PATH}")

    RUN_DIR.mkdir(parents=True, exist_ok=True)
    meta_df = pd.read_parquet(OOF_PATH)
    if "timestamp" in meta_df.columns:
        meta_df = meta_df.set_index("timestamp")
    meta_df.index = pd.to_datetime(meta_df.index, utc=True)
    meta_df = meta_df.sort_index()

    logger.info(
        f"Loaded {len(meta_df):,} OOF trades | "
        f"WIN={meta_df['win'].sum():,} ({meta_df['win'].mean()*100:.1f}%)"
    )

    avail = [c for c in META_FEATURES if c in meta_df.columns]
    missing = [c for c in META_FEATURES if c not in meta_df.columns]
    if missing:
        logger.warning(f"Missing features (filled 0): {missing}")

    X = meta_df[avail].ffill().fillna(0)
    y = meta_df["win"].values.astype(np.int32)

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
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            callbacks=[
                lgb.early_stopping(EARLY_STOPPING, verbose=False),
                lgb.log_evaluation(period=-1),
            ],
        )

        p_val = model.predict_proba(X_val)[:, 1]
        y_pred = (p_val >= 0.5).astype(int)
        auc = float(roc_auc_score(y_val, p_val))
        acc = float(accuracy_score(y_val, y_pred))
        f1 = float(f1_score(y_val, y_pred, zero_division=0))

        sel_wr = float(y_val[p_val >= 0.5].mean()) if (p_val >= 0.5).any() else 0.0
        rej_wr = float(y_val[p_val < 0.5].mean()) if (p_val < 0.5).any() else 0.0

        cv_metrics.append({
            "fold": fold, "auc": round(auc, 4), "acc": round(acc, 4),
            "f1": round(f1, 4), "wr_selected": round(sel_wr, 4),
            "wr_rejected": round(rej_wr, 4),
        })
        logger.info(
            f"  Fold {fold}: AUC={auc:.4f} | sel_WR={sel_wr*100:.1f}% "
            f"| rej_WR={rej_wr*100:.1f}%"
        )

        if auc > best_auc:
            best_auc, best_model, best_fold = auc, model, fold

    if best_model is None:
        raise RuntimeError("No valid CV fold — check OOF dataset class balance")

    final_model = lgb.LGBMClassifier(**META_PARAMS)
    final_model.set_params(n_estimators=best_model.best_iteration_ or 100)
    final_model.fit(X, y)

    out_pkl = RUN_DIR / "meta_lgbm.pkl"
    joblib.dump(final_model, out_pkl)

    feat_path = RUN_DIR / f"{RUN_NAME}_features.json"
    with open(feat_path, "w", encoding="utf-8") as f:
        json.dump(avail, f, indent=2)

    summary = {
        "run_name": RUN_NAME,
        "spec_version": SPEC["version"],
        "trained_at": datetime.now().isoformat(),
        "oof_source": str(OOF_PATH),
        "n_oof_trades": len(meta_df),
        "base_win_rate": round(float(y.mean()), 4),
        "features": avail,
        "cv_mean_auc": round(float(np.mean([m["auc"] for m in cv_metrics])), 4),
        "best_fold_auc": round(best_auc, 4),
        "best_fold": best_fold,
        "cv_metrics": cv_metrics,
        "n_estimators_final": int(final_model.n_estimators),
        "leak_check": "Labels from walk-forward OOF only — see ic32_oof_trades_meta.json",
        "next_step": "python pipeline/14_eval_meta_entry_ablation.py",
    }
    with open(RUN_DIR / f"{RUN_NAME}_train_meta.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\n  CV mean AUC : {summary['cv_mean_auc']:.4f}")
    print(f"  Best fold   : {best_fold} (AUC {best_auc:.4f})")
    print(f"  Saved model : {out_pkl}")
    print(f"  Features    : {avail}")


if __name__ == "__main__":
    main()