"""
pipeline/04_train_trend_momentum.py — Trend-Following Binary LGBM

Model momentum untuk trending market.
Binary: "Apakah ini saat yang tepat untuk entry WITH-TREND?"

Hanya dilatih pada TRENDING bars (HMM regime 0,3).
Label: 1 jika harga lanjut searah tren, 0 jika tidak.

Usage:
  python pipeline/04_train_trend_momentum.py --all
"""

import argparse, json, sys, warnings, numpy as np, pandas as pd
from pathlib import Path

ROOT = Path(__file__).parent.parent; sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

from config import (
    TRAINING_COINS, LABEL_DIR, MODEL_DIR, TRAIN_CUTOFF_DATE,
    N_FOLDS, PURGE_GAP_BARS,
)
from core.utils import setup_logger
from pipeline.shared import build_purged_folds
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

logger = setup_logger("04_trend_momentum")

# Features optimized for trend-following
TREND_FEATS = [
    # Price trajectory (momentum strength)
    "log_ret_1", "log_ret_5", "log_ret_20",
    # Oscillators
    "rsi_6", "stochrsi_k", "stochrsi_d",
    "rsi_h4", "rsi_slope_h4",
    # Trend strength
    "h4_trend", "trend_strength", "ema_21_slope_h4",
    # Flow
    "cvd_slope_h4", "cvd_momentum_adv", "ofi_h4_delta",
    "ofi_z_score", "volume_delta", "vol_ratio_20",
    # Volatility / regime
    "atr_14_h1", "atr_percentile_h1", "vol_spike_zscore",
    # HMM context
    "hmm_prob_0", "hmm_prob_1", "hmm_prob_2", "hmm_prob_3",
    # Acceleration
    "price_accel_1h", "absorption_z",
]


def generate_trend_labels(df, lookforward=8, min_move=0.003):
    """
    Binary trend-following labels.
    Uses HMM regime to determine trend direction, then labels whether
    price continues in that direction.

    TRENDING_UP (regime 3): label=1 if close[t+N] > close[t] * (1 + min_move)
    TRENDING_DOWN (regime 0): label=1 if close[t+N] < close[t] * (1 - min_move)

    Only labels bars where HMM regime is TRENDING (0 or 3).
    """
    n = len(df)
    labels = np.full(n, -1, dtype=np.int8)  # -1 = not trending, skip
    close = df["close"].values
    regime = df["hmm_regime_enc"].values if "hmm_regime_enc" in df.columns else np.ones(n)

    for i in range(n - lookforward - 1):
        r = int(regime[i])
        if r not in [0, 3]:  # Only trending regimes
            continue

        future_close = close[i + lookforward]
        current_close = close[i]

        if np.isnan(future_close) or np.isnan(current_close) or current_close <= 0:
            continue

        ret = (future_close - current_close) / current_close

        if r == 3:  # TRENDING_UP: continuation = up
            labels[i] = 1 if ret > min_move else 0
        elif r == 0:  # TRENDING_DOWN: continuation = down
            labels[i] = 1 if ret < -min_move else 0

    return labels


def load_data(coins):
    X_list, y_list, ts_list = [], [], []
    actual_feats = []

    for coin in coins:
        feat_path = LABEL_DIR / f"{coin}_features_v3.parquet"
        prob_path = LABEL_DIR / f"{coin}_hmm_probs.parquet"
        reg_path = LABEL_DIR / f"{coin}_regime_h1.parquet"

        if not feat_path.exists(): continue

        df = pd.read_parquet(feat_path).sort_index()
        df = df[df.index < TRAIN_CUTOFF_DATE]

        # Merge HMM
        if reg_path.exists():
            reg = pd.read_parquet(reg_path)
            if "hmm_regime_enc" in df.columns: df = df.drop(columns=["hmm_regime_enc"])
            df = df.join(reg[["hmm_regime_enc"]], how="left")
            df["hmm_regime_enc"] = df["hmm_regime_enc"].fillna(1).astype("int32")

        # Merge HMM probs
        if prob_path.exists():
            probs = pd.read_parquet(prob_path).sort_index()
            for i in range(4):
                c = f"hmm_prob_{i}"
                if c in probs.columns: df[c] = probs[c]
        for i in range(4):
            if f"hmm_prob_{i}" not in df.columns:
                df[f"hmm_prob_{i}"] = 0.25

        for c in TREND_FEATS:
            if c not in df.columns: df[c] = 0.0

        # Generate binary trend labels
        df["trend_label"] = generate_trend_labels(df)
        df = df[df["trend_label"] >= 0].copy()  # Only trending bars
        if len(df) < 100: continue

        if not actual_feats:
            actual_feats = [c for c in TREND_FEATS if c in df.columns]

        X = df[actual_feats].ffill().fillna(0)
        y = df["trend_label"].values.astype(np.int64)
        ts = df.index

        X_list.append(X); y_list.append(y); ts_list.append(ts)
        n_trend = len(df)
        n_cont = int(sum(y == 1))
        logger.info(f"{coin}: {n_trend:,} trending bars | continuation={n_cont/n_trend*100:.0f}%")

    X_all = pd.concat(X_list, axis=0)
    y_all = np.concatenate(y_list)
    ts_all = np.concatenate(ts_list)
    return X_all, y_all, ts_all, actual_feats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--run-id", default="trend_momentum_v1")
    args = parser.parse_args()

    coins = TRAINING_COINS if args.all else TRAINING_COINS[:5]

    print(f"\n{'='*60}")
    print(f"  TREND-FOLLOWING BINARY LGBM | {args.run_id}")
    print(f"  Label: continuation in TRENDING regime (0/1)")
    print(f"  Features: 25 (momentum + HMM + trend)")
    print(f"{'='*60}\n")

    X, y, ts, feat_cols = load_data(coins)
    logger.info(f"Total: {len(X):,} trending bars | Positive: {y.sum():,} ({y.mean()*100:.1f}%)")

    ts_idx = pd.DatetimeIndex(ts)
    folds = build_purged_folds(ts_idx, N_FOLDS, PURGE_GAP_BARS)

    cv_aucs = []; best_auc, best_model = 0, None

    for fi, (tr_idx, te_idx) in enumerate(folds):
        X_tr = X.iloc[tr_idx]; y_tr = y[tr_idx]
        X_te = X.iloc[te_idx]; y_te = y[te_idx]
        if len(np.unique(y_tr)) < 2: continue

        pos_weight = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)
        model = lgb.LGBMClassifier(
            objective="binary", n_estimators=500, learning_rate=0.03,
            max_depth=5, num_leaves=31, min_child_samples=50,
            subsample=0.8, colsample_bytree=0.7,
            scale_pos_weight=pos_weight,
            verbose=-1, n_jobs=-1, random_state=42,
        )
        model.fit(X_tr, y_tr,
                  eval_set=[(X_te, y_te)], eval_metric="auc",
                  callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])

        auc = roc_auc_score(y_te, model.predict_proba(X_te)[:, 1])
        cv_aucs.append(auc)
        logger.info(f"Fold {fi+1}: AUC={auc:.4f}")
        if auc > best_auc: best_auc, best_model = auc, model

    pos_w = (y == 0).sum() / max((y == 1).sum(), 1)
    final = lgb.LGBMClassifier(
        objective="binary", n_estimators=500, learning_rate=0.03,
        max_depth=5, num_leaves=31, min_child_samples=50,
        subsample=0.8, colsample_bytree=0.7,
        scale_pos_weight=pos_w, verbose=-1, n_jobs=-1, random_state=42,
    )
    final.fit(X, y)

    run_dir = MODEL_DIR / "runs" / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    import joblib
    joblib.dump(final, run_dir / "lgbm.pkl")
    with open(run_dir / "feature_cols.json", "w") as f:
        json.dump(feat_cols, f, indent=2)

    imp = list(zip(feat_cols, final.feature_importances_))
    imp.sort(key=lambda x: x[1], reverse=True)

    print(f"\n{'='*60}")
    print(f"  TREND-FOLLOWING COMPLETE — {args.run_id}")
    print(f"  CV AUC: {np.mean(cv_aucs):.4f} +/- {np.std(cv_aucs):.4f} (baseline: 0.500)")
    print(f"  Gain vs random: {np.mean(cv_aucs)-0.5:+.4f}")
    print(f"  Top 10 features:")
    for i, (f, v) in enumerate(imp[:10]):
        print(f"  {i+1:>2}. {f:<30} {v:>8.1f}")
    print(f"\n  Model: {run_dir / 'lgbm.pkl'}")


if __name__ == "__main__":
    main()
