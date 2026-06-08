"""
pipeline/04_train_momentum_lgbm.py — Train Momentum LGBM (flow-based labels)

Dual-Model Architecture — Phase 3.
Model kedua untuk trending market: fitur momentum + momentum labels.
Dipasangkan dengan swing LGBM (lgbm_baseline.pkl) via HMM selector.

Usage:
  python pipeline/04_train_momentum_lgbm.py --all
  python pipeline/04_train_momentum_lgbm.py --run-id momentum_v1
"""

import argparse, json, sys, warnings, numpy as np, pandas as pd
from pathlib import Path

ROOT = Path(__file__).parent.parent; sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

from config import (
    TRAINING_COINS, LABEL_DIR, MODEL_DIR, TRAIN_CUTOFF_DATE,
    N_FOLDS, PURGE_GAP_BARS, LGBM_PARAMS, LGBM_EARLY_STOPPING,
)
from core.utils import setup_logger
from pipeline.shared import build_purged_folds
import lightgbm as lgb
from sklearn.metrics import f1_score

logger = setup_logger("04_momentum_lgbm")

# Momentum features (flow-based, no swing levels)
MOMENTUM_FEATS = [
    # Order flow
    "ofi_z_score", "ofi_acceleration", "cvd_momentum_adv", "absorption_z",
    # Volume
    "volume_delta", "vol_ratio_20",
    # Price trajectory
    "log_ret_1", "log_ret_5", "log_ret_20",
    # Oscillator
    "rsi_6",
    # Cross-market
    "btc_h1_return",
    # HMM regime probabilities (context)
    "hmm_prob_0", "hmm_prob_1", "hmm_prob_2", "hmm_prob_3",
    # Trend context
    "h4_trend", "trend_strength",
]

MOMENTUM_LABEL_MAP = {"BEARISH": 0, "NEUTRAL": 1, "BULLISH": 2}


def load_data(coins):
    X_list, y_list, ts_list = [], [], []
    actual_feats = []

    for coin in coins:
        feat_path = LABEL_DIR / f"{coin}_features_v3.parquet"
        label_path = LABEL_DIR / f"{coin}_momentum_v2_labels.parquet"
        prob_path = LABEL_DIR / f"{coin}_hmm_probs.parquet"
        reg_path = LABEL_DIR / f"{coin}_regime_h1.parquet"

        if not feat_path.exists() or not label_path.exists():
            continue

        df = pd.read_parquet(feat_path).sort_index()
        df = df[df.index < TRAIN_CUTOFF_DATE]

        # Merge momentum labels
        lbl = pd.read_parquet(label_path).sort_index()
        df = df.join(lbl["momentum_v2_label"], how="inner")

        # Merge HMM probs
        if prob_path.exists():
            probs = pd.read_parquet(prob_path).sort_index()
            for i in range(4):
                c = f"hmm_prob_{i}"
                if c in probs.columns:
                    df[c] = probs[c]
        for i in range(4):
            c = f"hmm_prob_{i}"
            if c not in df.columns:
                df[c] = 0.25

        # Merge HMM regime
        if reg_path.exists():
            reg = pd.read_parquet(reg_path)
            if "hmm_regime_enc" in df.columns:
                df = df.drop(columns=["hmm_regime_enc"])
            df = df.join(reg[["hmm_regime_enc"]], how="left")
            df["hmm_regime_enc"] = df["hmm_regime_enc"].fillna(1).astype("int32")

        # Fill missing features
        for c in MOMENTUM_FEATS:
            if c not in df.columns:
                df[c] = 0.0

        df = df.dropna(subset=["momentum_v2_label"])
        if len(df) < 100:
            continue

        # First coin determines feature list
        feat_cols = [c for c in MOMENTUM_FEATS if c in df.columns]
        if not actual_feats:
            actual_feats = feat_cols

        X = df[actual_feats].ffill().fillna(0)
        y = df["momentum_v2_label"].values.astype(np.int64)
        ts = df.index

        X_list.append(X); y_list.append(y); ts_list.append(ts)
        logger.info(f"{coin}: {len(df):,} bars | BULL={sum(y==2)/len(y)*100:.0f}% NEU={sum(y==1)/len(y)*100:.0f}% BEAR={sum(y==0)/len(y)*100:.0f}%")

    X_all = pd.concat(X_list, axis=0)
    y_all = np.concatenate(y_list)
    ts_all = np.concatenate(ts_list)
    return X_all, y_all, ts_all, actual_feats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--run-id", default="momentum_v1")
    args = parser.parse_args()

    coins = TRAINING_COINS if args.all else TRAINING_COINS[:5]

    print(f"\n{'='*60}")
    print(f"  MOMENTUM LGBM TRAINING | run={args.run_id}")
    print(f"  Features: flow-based | Labels: momentum_v2")
    print(f"  Coins: {len(coins)} | Cutoff: {TRAIN_CUTOFF_DATE}")
    print(f"{'='*60}\n")

    X, y, ts, feat_cols = load_data(coins)
    logger.info(f"Total: {len(X):,} samples, {len(feat_cols)} features")
    for i, lbl in enumerate(["BEARISH", "NEUTRAL", "BULLISH"]):
        logger.info(f"  {lbl}: {(y==i).sum():,} ({(y==i).mean()*100:.1f}%)")

    ts_idx = pd.DatetimeIndex(ts)
    folds = build_purged_folds(ts_idx, N_FOLDS, PURGE_GAP_BARS)

    cv_results = []
    best_logloss = float("inf")
    best_model = None

    for fi, (tr_idx, te_idx) in enumerate(folds):
        X_tr = X.iloc[tr_idx]; y_tr = y[tr_idx]
        X_te = X.iloc[te_idx]; y_te = y[te_idx]

        params = {**LGBM_PARAMS}
        model = lgb.LGBMClassifier(**params)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_te, y_te)],
            eval_metric="multi_logloss",
            callbacks=[lgb.early_stopping(LGBM_EARLY_STOPPING), lgb.log_evaluation(0)],
        )

        y_prob = model.predict_proba(X_te)
        y_pred = np.argmax(y_prob, axis=1)
        f1 = f1_score(y_te, y_pred, average="macro")
        logloss = -np.mean(np.log(y_prob[np.arange(len(y_te)), y_te] + 1e-10))

        cv_results.append({"fold": fi+1, "logloss": round(logloss, 4), "f1": round(f1, 4)})
        logger.info(f"Fold {fi+1}: logloss={logloss:.4f} f1={f1:.4f}")

        if logloss < best_logloss:
            best_logloss = logloss
            best_model = model

    # Final retrain
    logger.info(f"Best CV logloss: {best_logloss:.4f}")
    final = lgb.LGBMClassifier(**LGBM_PARAMS)
    final.fit(X, y)

    run_dir = MODEL_DIR / "runs" / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    import joblib
    joblib.dump(final, run_dir / "lgbm.pkl")
    with open(run_dir / "feature_cols.json", "w") as f:
        json.dump(feat_cols, f, indent=2)
    with open(run_dir / "cv_results.json", "w") as f:
        json.dump({"run_id": args.run_id, "n_features": len(feat_cols), "folds": cv_results}, f, indent=2)

    # Feature importance
    imp = list(zip(feat_cols, final.feature_importances_))
    imp.sort(key=lambda x: x[1], reverse=True)

    print(f"\n{'='*60}")
    print(f"  MOMENTUM LGBM COMPLETE — {args.run_id}")
    print(f"  Top 10 features:")
    for i, (f, v) in enumerate(imp[:10]):
        print(f"  {i+1:>2}. {f:<30} {v:>8.1f}")

    f1s = [m["f1"] for m in cv_results]
    print(f"\n  CV F1: {np.mean(f1s):.4f} ± {np.std(f1s):.4f}  (random baseline: 0.333)")
    print(f"  Model: {run_dir / 'lgbm.pkl'}")


if __name__ == "__main__":
    main()
