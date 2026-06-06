"""
pipeline/04_train_lgbm_hmm_probs.py — Retrain LGBM with HMM probabilities

Version B: 32 KEEP + 4 hmm_prob_* (replace hmm_regime_enc argmax)
Total: 36 features

Usage:
  python pipeline/04_train_lgbm_hmm_probs.py --all
  python pipeline/04_train_lgbm_hmm_probs.py --run-id lgbm_hmm_probs_v1
"""

import argparse, json, sys, warnings, joblib, numpy as np, pandas as pd
from pathlib import Path

ROOT = Path(__file__).parent.parent; sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

from config import (
    TRAINING_COINS, LABEL_DIR, MODEL_DIR, TRAIN_CUTOFF_DATE,
    LABEL_MAP, NUM_CLASSES, N_FOLDS, PURGE_GAP_BARS,
    LGBM_PARAMS, LGBM_EARLY_STOPPING, LGBM_CLASS_WEIGHTS,
)
from core.utils import setup_logger
from pipeline.shared import build_purged_folds
import lightgbm as lgb
from sklearn.metrics import f1_score

logger = setup_logger("04_lgbm_hmm_probs")

# Load KEEP features (ic32)
KEEP_FEATS = json.load(open(MODEL_DIR / "feature_cols_ic32.json"))
HMM_PROB_FEATS = ["hmm_prob_0", "hmm_prob_1", "hmm_prob_2", "hmm_prob_3"]
VERSION_B_FEATS = [f for f in KEEP_FEATS if f != "hmm_regime_enc"] + HMM_PROB_FEATS


def load_data(coins):
    X_list, y_list, ts_list = [], [], []
    for coin in coins:
        feat_path = LABEL_DIR / f"{coin}_features_v3.parquet"
        prob_path = LABEL_DIR / f"{coin}_hmm_probs.parquet"
        if not feat_path.exists(): continue
        df = pd.read_parquet(feat_path).sort_index()
        df = df[df.index < TRAIN_CUTOFF_DATE]

        # Merge HMM probabilities
        if prob_path.exists():
            probs = pd.read_parquet(prob_path).sort_index()
            for c in HMM_PROB_FEATS:
                if c in probs.columns:
                    df[c] = probs[c]
            # Fill any missing prob columns
            for c in HMM_PROB_FEATS:
                if c not in df.columns:
                    df[c] = 0.25

        # Merge HMM regime (for Version A comparison)
        reg_path = LABEL_DIR / f"{coin}_regime_h1.parquet"
        if reg_path.exists():
            reg = pd.read_parquet(reg_path)
            if "hmm_regime_enc" in df.columns:
                df = df.drop(columns=["hmm_regime_enc"])
            df = df.join(reg[["hmm_regime_enc"]], how="left")
            df["hmm_regime_enc"] = df["hmm_regime_enc"].fillna(1).astype("int32")

        mask = df["label"].astype(str).isin(LABEL_MAP)
        df = df[mask].copy()
        if len(df) < 100: continue

        # Build feature matrix as DataFrame (preserve column names for LGBM)
        feat_cols = [c for c in VERSION_B_FEATS if c in df.columns]
        X = df[feat_cols].ffill().fillna(0)
        y = df["label"].map(LABEL_MAP).values.astype(np.int64)
        ts = df.index

        X_list.append(X); y_list.append(y); ts_list.append(ts)
        logger.info(f"{coin}: {len(df):,} bars")

    X_all = pd.concat(X_list, axis=0); y_all = np.concatenate(y_list)
    ts_all = np.concatenate(ts_list)
    return X_all, y_all, ts_all, feat_cols


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--run-id", default="lgbm_hmm_probs_v1")
    args = parser.parse_args()

    coins = TRAINING_COINS if args.all else TRAINING_COINS[:5]

    print(f"\n{'='*60}")
    print(f"  LGBM HMM PROBS RETRAIN | run={args.run_id}")
    print(f"  Features: {len(VERSION_B_FEATS)} (32 KEEP + 4 hmm_probs)")
    print(f"  Coins: {len(coins)} | Cutoff: {TRAIN_CUTOFF_DATE}")
    print(f"{'='*60}\n")

    X, y, ts, feat_cols = load_data(coins)
    logger.info(f"Total: {len(X):,} samples, {len(feat_cols)} features")
    for i, lbl in enumerate(["SHORT", "FLAT", "LONG"]):
        logger.info(f"  {lbl}: {(y==i).sum():,} ({(y==i).mean()*100:.1f}%)")

    # Purged CV
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
    logger.info("Retraining final on all data...")

    final = lgb.LGBMClassifier(**{**LGBM_PARAMS})
    final.fit(X, y)

    # Save
    run_dir = MODEL_DIR / "runs" / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(final, run_dir / "lgbm.pkl")
    with open(run_dir / "feature_cols.json", "w") as f:
        json.dump(feat_cols, f, indent=2)
    with open(run_dir / "cv_results.json", "w") as f:
        json.dump({"run_id": args.run_id, "n_features": len(feat_cols), "folds": cv_results}, f, indent=2)

    # Feature importance
    imp = list(zip(feat_cols, final.feature_importances_))
    imp.sort(key=lambda x: x[1], reverse=True)

    print(f"\n{'='*60}")
    print(f"  TRAINING COMPLETE — {args.run_id}")
    print(f"{'='*60}")
    print(f"  Features: {len(feat_cols)}")
    print(f"\n  Top 20 Feature Importance:")
    for i, (f, v) in enumerate(imp[:20]):
        tag = "[HMM]" if f.startswith("hmm_prob") else ""
        print(f"  {i+1:>2}. {f:<35} {v:>8.1f} {tag}")

    # CV summary
    print(f"\n  CV Results:")
    for r in cv_results:
        print(f"  Fold {r['fold']}: logloss={r['logloss']:.4f} f1={r['f1']:.4f}")

    print(f"\n  Model: {run_dir / 'lgbm.pkl'}")
    print(f"  Feats: {run_dir / 'feature_cols.json'}")
    print(f"\n  Next: Compare with Version A (argmax) via holdout backtest")

if __name__ == "__main__":
    main()
