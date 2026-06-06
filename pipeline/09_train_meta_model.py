"""
pipeline/09_train_meta_model.py — Train Meta-Labeling Model

Input : data/meta_labels/meta_labels_training.csv
Output: models/meta_model.pkl + models/meta_model_features.json

Meta-model: binary LGBM classifier
  Target: win (1) = trade profitable, loss (0) = trade rugi
  Features: LGBM probabilities + confidence + HMM regime + market context

Usage:
    python pipeline/09_train_meta_model.py
    python pipeline/09_train_meta_model.py --threshold 0.55
    python pipeline/09_train_meta_model.py --input data/meta_labels/meta_labels_training.csv
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
from sklearn.calibration import CalibratedClassifierCV
import lightgbm as lgb

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from config import MODEL_DIR, TRAIN_CUTOFF_DATE, PURGE_GAP_BARS, N_FOLDS
from pipeline.shared import build_purged_folds
from core.utils import setup_logger

logger = setup_logger("09_train_meta_model")

# Features untuk meta-model
META_FEATURES = [
    # LGBM raw signal quality
    "lgbm_prob_long", "lgbm_prob_short", "lgbm_prob_flat",
    "prob_margin",       # |prob_long - prob_short| → conviction
    "confidence",        # post-LSTM adjusted confidence

    # Regime context
    "hmm_regime_enc",

    # Direction (LONG=1, SHORT=0)
    "direction_enc",

    # Momentum context at entry
    "rsi_6", "stochrsi_k", "stochrsi_d",
    "rsi_h4", "rsi_slope_h4",

    # Flow
    "cvd_slope_h4", "ofi_h4_delta", "cvd_momentum_adv",

    # Structure
    "swing_momentum", "dist_from_8h_high", "price_in_range",
    "long_short_ratio", "dist_liq_50x_long", "dist_liq_50x_short",

    # Volatility context
    "atr_14_h1",

    # Coin identity (different coins have different reliability)
    "coin_enc",
]

META_LGBM_PARAMS = {
    "objective":        "binary",
    "metric":           "auc",
    "n_estimators":     500,
    "learning_rate":    0.05,
    "num_leaves":       31,
    "max_depth":        6,
    "min_child_samples": 50,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "reg_alpha":        0.1,
    "reg_lambda":       1.0,
    "random_state":     42,
    "verbose":          -1,
    "n_jobs":           -1,
}


def load_and_prepare(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    # Direction encoding
    df["direction_enc"] = (df["direction"] == "LONG").astype(int)

    # Coin encoding
    coins = sorted(df["coin"].unique())
    coin_map = {c: i for i, c in enumerate(coins)}
    df["coin_enc"] = df["coin"].map(coin_map)

    logger.info(f"Loaded {len(df):,} trades | WR={df['win'].mean()*100:.1f}%")
    logger.info(f"Coins: {len(coins)} | Date range: {df['timestamp'].min().date()} → {df['timestamp'].max().date()}")
    logger.info(f"Direction: LONG={( df['direction']=='LONG').sum():,}, SHORT={(df['direction']=='SHORT').sum():,}")

    return df, coin_map


def walk_forward_cv(df: pd.DataFrame, feat_cols: list) -> list:
    logger.info(f"Walk-Forward CV (n_folds={N_FOLDS}, purge={PURGE_GAP_BARS})...")

    df_sorted = df.sort_values("timestamp").reset_index(drop=True)
    X = df_sorted[feat_cols].fillna(0).values
    y = df_sorted["win"].values
    ts = df_sorted["timestamp"].values

    folds = build_purged_folds(
        pd.DatetimeIndex(ts),
        n_folds=N_FOLDS,
        purge=PURGE_GAP_BARS,
    )

    results = []
    for fold, (train_idx, val_idx) in enumerate(folds, 1):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        if len(np.unique(y_tr)) < 2 or len(np.unique(y_val)) < 2:
            logger.warning(f"  Fold {fold}: insufficient class diversity, skip")
            continue

        model = lgb.LGBMClassifier(**META_LGBM_PARAMS)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            callbacks=[
                lgb.early_stopping(50, verbose=False),
                lgb.log_evaluation(period=-1),
            ],
        )

        prob_val = model.predict_proba(X_val)[:, 1]
        pred_val = (prob_val >= 0.5).astype(int)

        acc  = accuracy_score(y_val, pred_val)
        f1   = f1_score(y_val, pred_val, zero_division=0)
        auc  = roc_auc_score(y_val, prob_val)
        wr_pred_1 = y_val[pred_val == 1].mean() if (pred_val == 1).sum() > 0 else 0
        coverage  = (pred_val == 1).mean()

        logger.info(
            f"  Fold {fold}: AUC={auc:.4f} | Acc={acc:.4f} | F1={f1:.4f} | "
            f"WR(pred=1)={wr_pred_1*100:.1f}% | Coverage={coverage*100:.1f}%"
        )

        results.append({
            "fold": fold, "auc": auc, "accuracy": acc, "f1": f1,
            "wr_when_pred_win": wr_pred_1,
            "coverage": coverage,
            "n_val": len(y_val),
            "best_iteration": model.best_iteration_,
        })

    return results


def train_final_model(df: pd.DataFrame, feat_cols: list, avg_iter: int) -> lgb.LGBMClassifier:
    X = df[feat_cols].fillna(0).values
    y = df["win"].values

    params = META_LGBM_PARAMS.copy()
    params["n_estimators"] = avg_iter

    model = lgb.LGBMClassifier(**params)
    model.fit(X, y)
    logger.info(f"Final meta-model trained: {avg_iter} estimators, {len(feat_cols)} features")
    return model


def evaluate_threshold(model, df: pd.DataFrame, feat_cols: list):
    X = df[feat_cols].fillna(0).values
    y = df["win"].values
    prob = model.predict_proba(X)[:, 1]

    print(f"\n{'Threshold':>10} {'Coverage':>10} {'WR(kept)':>10} {'Delta WR':>10}")
    print("-" * 44)
    base_wr = y.mean() * 100
    for thr in [0.40, 0.45, 0.50, 0.52, 0.55, 0.58, 0.60, 0.65]:
        mask = prob >= thr
        if mask.sum() < 10:
            continue
        wr = y[mask].mean() * 100
        cov = mask.mean() * 100
        print(f"{thr:>10.2f} {cov:>9.1f}% {wr:>9.1f}% {wr-base_wr:>+9.1f}pp")
    print(f"  Baseline WR: {base_wr:.1f}%")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/meta_labels/meta_labels_training.csv")
    parser.add_argument("--output-model", default="models/meta_model.pkl")
    parser.add_argument("--output-features", default="models/meta_model_features.json")
    parser.add_argument("--threshold", type=float, default=0.50)
    args = parser.parse_args()

    print(f"\n{'='*65}")
    print(f" TRAIN META-MODEL | input={args.input}")
    print(f"{'='*65}\n")

    df, coin_map = load_and_prepare(args.input)

    # Pilih fitur yang tersedia
    avail_feat = [f for f in META_FEATURES if f in df.columns]
    missing = [f for f in META_FEATURES if f not in df.columns]
    if missing:
        logger.warning(f"Missing features: {missing}")
    logger.info(f"Meta features: {len(avail_feat)}")

    # Walk-forward CV
    cv_results = walk_forward_cv(df, avail_feat)

    if not cv_results:
        logger.error("CV gagal — tidak ada hasil")
        return

    mean_auc = np.mean([r["auc"] for r in cv_results])
    mean_wr  = np.mean([r["wr_when_pred_win"] for r in cv_results]) * 100
    mean_cov = np.mean([r["coverage"] for r in cv_results]) * 100
    avg_iter = int(np.mean([r["best_iteration"] for r in cv_results]))

    print(f"\n{'='*65}")
    print(f"  CV SUMMARY")
    print(f"  Mean AUC         : {mean_auc:.4f}")
    print(f"  Mean WR (pred=1) : {mean_wr:.1f}%  (baseline: {df['win'].mean()*100:.1f}%)")
    print(f"  Mean Coverage    : {mean_cov:.1f}%")
    print(f"  Avg Best Iter    : {avg_iter}")
    print(f"{'='*65}")

    # Train final model
    final_model = train_final_model(df, avail_feat, avg_iter)

    # Threshold analysis
    evaluate_threshold(final_model, df, avail_feat)

    # Save
    joblib.dump(final_model, args.output_model)
    meta_config = {
        "features": avail_feat,
        "coin_map": coin_map,
        "default_threshold": args.threshold,
        "cv_mean_auc": round(mean_auc, 4),
        "cv_mean_wr_pred1": round(mean_wr / 100, 4),
        "cv_mean_coverage": round(mean_cov / 100, 4),
        "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
    }
    with open(args.output_features, "w") as f:
        json.dump(meta_config, f, indent=2)

    logger.info(f"Saved: {args.output_model}")
    logger.info(f"Saved: {args.output_features}")

    print(f"\n  Feature importance (top 10):")
    fi = dict(zip(avail_feat, final_model.feature_importances_))
    for feat, imp in sorted(fi.items(), key=lambda x: -x[1])[:10]:
        print(f"    {feat:<35} {imp:.0f}")


if __name__ == "__main__":
    main()
