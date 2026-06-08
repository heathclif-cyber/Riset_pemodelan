"""
pipeline/04_train_triple_barrier_lgbm.py — Triple Barrier LGBM Training

Momentum model untuk trending market: TB labels + momentum features.
Dipakai sebagai Model 3 di dual-model ensemble.

Usage:
  python pipeline/04_train_triple_barrier_lgbm.py --all
  python pipeline/04_train_triple_barrier_lgbm.py --run-id tb_momentum_v1
"""

import argparse, json, sys, warnings, numpy as np, pandas as pd
from pathlib import Path

ROOT = Path(__file__).parent.parent; sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

from config import (
    TRAINING_COINS, LABEL_DIR, HOLDOUT_DIR, MODEL_DIR, TRAIN_CUTOFF_DATE,
    N_FOLDS, PURGE_GAP_BARS, LGBM_PARAMS, LGBM_EARLY_STOPPING,
    MAX_HOLDING_BARS,
)
from core.utils import setup_logger
from core.features import triple_barrier_labeling
from pipeline.shared import build_purged_folds
import lightgbm as lgb
from sklearn.metrics import f1_score

logger = setup_logger("04_tb_lgbm")

# Momentum features for TB (no swing levels!)
TB_FEATS = [
    "ofi_z_score", "ofi_acceleration", "cvd_momentum_adv", "absorption_z",
    "volume_delta", "vol_ratio_20",
    "log_ret_1", "log_ret_5", "log_ret_20",
    "rsi_6", "stochrsi_k", "stochrsi_d",
    "rsi_h4", "rsi_slope_h4", "ema_21_slope_h4",
    "cvd_slope_h4", "ofi_h4_delta",
    "atr_14_h1", "atr_percentile_h1", "vol_spike_zscore",
    "hmm_prob_0", "hmm_prob_1", "hmm_prob_2", "hmm_prob_3",
    "h4_trend", "trend_strength",
    "long_short_ratio", "price_accel_1h",
]

TB_LABEL_MAP = {"SHORT": 0, "FLAT": 1, "LONG": 2}
TP_ATR = 3.0
SL_ATR = 1.5


def load_data(coins, is_holdout=False):
    base_dir = HOLDOUT_DIR / "labeled" if is_holdout else LABEL_DIR
    X_list, y_list, ts_list = [], [], []
    actual_feats = []

    for coin in coins:
        feat_path = base_dir / f"{coin}_features_v3.parquet"
        prob_path = base_dir / f"{coin}_hmm_probs.parquet"
        reg_path = base_dir / f"{coin}_regime_h1.parquet"

        if not feat_path.exists():
            continue

        df = pd.read_parquet(feat_path).sort_index()
        if not is_holdout:
            df = df[df.index < TRAIN_CUTOFF_DATE]

        # Generate TB labels
        if "close" not in df.columns or "atr_14_h1" not in df.columns:
            continue

        tb_labels = triple_barrier_labeling(
            df["close"], df["high"] if "high" in df.columns else df["close"],
            df["low"] if "low" in df.columns else df["close"],
            df["atr_14_h1"], tp_atr_mult=TP_ATR, sl_atr_mult=SL_ATR, max_hold=MAX_HOLDING_BARS,
        )
        df["tb_label"] = tb_labels

        # Merge HMM probs
        if prob_path.exists():
            probs = pd.read_parquet(prob_path).sort_index()
            for i in range(4):
                c = f"hmm_prob_{i}"
                if c in probs.columns:
                    df[c] = probs[c]
        for i in range(4):
            if f"hmm_prob_{i}" not in df.columns:
                df[f"hmm_prob_{i}"] = 0.25

        # Merge HMM regime
        if reg_path.exists():
            reg = pd.read_parquet(reg_path)
            if "hmm_regime_enc" in df.columns:
                df = df.drop(columns=["hmm_regime_enc"])
            df = df.join(reg[["hmm_regime_enc"]], how="left")
            df["hmm_regime_enc"] = df["hmm_regime_enc"].fillna(1).astype("int32")

        for c in TB_FEATS:
            if c not in df.columns:
                df[c] = 0.0

        df = df.dropna(subset=["tb_label"])
        mask = df["tb_label"].astype(str).isin(TB_LABEL_MAP)
        df = df[mask].copy()
        if len(df) < 100:
            continue

        if not actual_feats:
            actual_feats = [c for c in TB_FEATS if c in df.columns]

        X = df[actual_feats].ffill().fillna(0)
        y = df["tb_label"].map(TB_LABEL_MAP).values.astype(np.int64)
        ts = df.index

        X_list.append(X); y_list.append(y); ts_list.append(ts)
        logger.info(f"{coin}: {len(df):,} bars | LONG={sum(y==2)/len(y)*100:.0f}% SHORT={sum(y==0)/len(y)*100:.0f}% FLAT={sum(y==1)/len(y)*100:.0f}%")

    X_all = pd.concat(X_list, axis=0)
    y_all = np.concatenate(y_list)
    ts_all = np.concatenate(ts_list)
    return X_all, y_all, ts_all, actual_feats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--run-id", default="tb_momentum_v1")
    args = parser.parse_args()

    coins = TRAINING_COINS if args.all else TRAINING_COINS[:5]

    print(f"\n{'='*60}")
    print(f"  TRIPLE BARRIER LGBM | run={args.run_id}")
    print(f"  TP={TP_ATR}xATR SL={SL_ATR}xATR Hold={MAX_HOLDING_BARS}")
    print(f"  Features: {len(TB_FEATS)} (momentum, no swing)")
    print(f"  Coins: {len(coins)} | Cutoff: {TRAIN_CUTOFF_DATE}")
    print(f"{'='*60}\n")

    X, y, ts, feat_cols = load_data(coins)
    logger.info(f"Total: {len(X):,} samples, {len(feat_cols)} features")
    for i, lbl in enumerate(["SHORT", "FLAT", "LONG"]):
        logger.info(f"  {lbl}: {(y==i).sum():,} ({(y==i).mean()*100:.1f}%)")

    ts_idx = pd.DatetimeIndex(ts)
    folds = build_purged_folds(ts_idx, N_FOLDS, PURGE_GAP_BARS)

    cv_results = []
    best_f1, best_model = 0, None

    for fi, (tr_idx, te_idx) in enumerate(folds):
        X_tr = X.iloc[tr_idx]; y_tr = y[tr_idx]
        X_te = X.iloc[te_idx]; y_te = y[te_idx]

        params = {**LGBM_PARAMS}
        params["class_weight"] = {0: 2.5, 1: 1.0, 2: 2.5}  # balance SHORT/LONG
        model = lgb.LGBMClassifier(**params)
        model.fit(X_tr, y_tr,
                  eval_set=[(X_te, y_te)], eval_metric="multi_logloss",
                  callbacks=[lgb.early_stopping(LGBM_EARLY_STOPPING), lgb.log_evaluation(0)])

        y_pred = np.argmax(model.predict_proba(X_te), axis=1)
        f1 = f1_score(y_te, y_pred, average="macro")
        cv_results.append({"fold": fi+1, "f1": round(f1, 4)})
        logger.info(f"Fold {fi+1}: f1={f1:.4f}")
        if f1 > best_f1: best_f1, best_model = f1, model

    final = lgb.LGBMClassifier(**{**LGBM_PARAMS, "class_weight": {0: 2.5, 1: 1.0, 2: 2.5}})
    final.fit(X, y)

    run_dir = MODEL_DIR / "runs" / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    import joblib
    joblib.dump(final, run_dir / "lgbm.pkl")
    with open(run_dir / "feature_cols.json", "w") as f:
        json.dump(feat_cols, f, indent=2)

    imp = list(zip(feat_cols, final.feature_importances_))
    imp.sort(key=lambda x: x[1], reverse=True)

    f1s = [m["f1"] for m in cv_results]
    print(f"\n{'='*60}")
    print(f"  TB LGBM COMPLETE — {args.run_id}")
    print(f"  CV F1: {np.mean(f1s):.4f} +/- {np.std(f1s):.4f} (random: 0.333)")
    print(f"  Top 10 features:")
    for i, (f, v) in enumerate(imp[:10]):
        print(f"  {i+1:>2}. {f:<30} {v:>8.1f}")


if __name__ == "__main__":
    main()
