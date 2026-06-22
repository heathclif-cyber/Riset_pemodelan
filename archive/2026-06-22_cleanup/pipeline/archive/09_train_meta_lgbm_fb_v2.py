"""
pipeline/09_train_meta_lgbm_fb_v2.py — Train meta LGBM for flatboost_v2 stack.

Prerequisite: python pipeline/08_generate_meta_labels_fb_v2.py
Output: models/runs/tb_meta_fb_v2/meta_lgbm.pkl
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

from config import MODEL_DIR, N_FOLDS
from core.utils import setup_logger
from pipeline.shared import build_purged_folds

logger = setup_logger("09_meta_fb_v2")

RUN_NAME = "tb_meta_fb_v2"
RUN_DIR = MODEL_DIR / "runs" / RUN_NAME
OOF_PATH = ROOT / "data" / "meta_labels" / "fb_v2_oof_trades.parquet"
PURGE_GAP = 36

META_FEATURES = [
    "p_short", "p_flat", "p_long", "confidence", "direction",
    "hmm_regime_enc", "atr_percentile_h1", "funding_rate",
    "vol_spike_zscore", "ofi_h4_delta", "cvd_slope_h4",
]

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
    if not OOF_PATH.exists():
        raise FileNotFoundError(f"Run 08_generate_meta_labels_fb_v2.py first: {OOF_PATH}")

    RUN_DIR.mkdir(parents=True, exist_ok=True)
    meta_df = pd.read_parquet(OOF_PATH)
    if "timestamp" in meta_df.columns:
        meta_df = meta_df.set_index("timestamp")
    meta_df.index = pd.to_datetime(meta_df.index, utc=True)
    meta_df = meta_df.sort_index()

    avail = [c for c in META_FEATURES if c in meta_df.columns]
    X = meta_df[avail].ffill().fillna(0)
    y = meta_df["win"].values.astype(np.int32)

    print(f"\n{'='*60}")
    print(f"  META TRAIN — {RUN_NAME} | {len(meta_df):,} OOF trades")
    print(f"  Base WR: {y.mean()*100:.1f}% | features: {len(avail)}")
    print(f"{'='*60}\n")

    folds = build_purged_folds(X.index, N_FOLDS, PURGE_GAP)
    cv_metrics = []
    best_auc, best_model, best_fold = -1.0, None, -1

    for fold, (tr_idx, val_idx) in enumerate(folds, 1):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]
        if len(np.unique(y_tr)) < 2 or len(np.unique(y_val)) < 2:
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
        auc = float(roc_auc_score(y_val, p_val))
        sel_wr = float(y_val[p_val >= 0.5].mean()) if (p_val >= 0.5).any() else 0.0
        rej_wr = float(y_val[p_val < 0.5].mean()) if (p_val < 0.5).any() else 0.0
        cv_metrics.append({"fold": fold, "auc": round(auc, 4), "wr_sel": round(sel_wr, 4)})
        logger.info(f"  Fold {fold}: AUC={auc:.4f} sel_WR={sel_wr*100:.1f}% rej_WR={rej_wr*100:.1f}%")
        if auc > best_auc:
            best_auc, best_model, best_fold = auc, model, fold

    final_model = lgb.LGBMClassifier(**META_PARAMS)
    final_model.set_params(n_estimators=best_model.best_iteration_ or 100)
    final_model.fit(X, y)
    joblib.dump(final_model, RUN_DIR / "meta_lgbm.pkl")
    with open(RUN_DIR / f"{RUN_NAME}_features.json", "w", encoding="utf-8") as f:
        json.dump(avail, f, indent=2)

    summary = {
        "run_name": RUN_NAME,
        "base_model": "tb_lgbm_flatboost_v2",
        "trained_at": datetime.now().isoformat(),
        "n_oof_trades": len(meta_df),
        "base_win_rate": round(float(y.mean()), 4),
        "cv_mean_auc": round(float(np.mean([m["auc"] for m in cv_metrics])), 4),
        "best_fold_auc": round(best_auc, 4),
        "features": avail,
        "cv_metrics": cv_metrics,
    }
    with open(RUN_DIR / f"{RUN_NAME}_train_meta.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"  CV mean AUC: {summary['cv_mean_auc']:.4f}")
    print(f"  Saved: {RUN_DIR / 'meta_lgbm.pkl'}")


if __name__ == "__main__":
    main()