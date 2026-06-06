"""Train Momentum LGBM with 38 IC-validated KEEP features on 21 coins."""
import sys, json, warnings, numpy as np, pandas as pd
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

logger = setup_logger("04_momentum_ic38")

# 38 KEEP features from IC test vs momentum_v2 labels
IC38_FEATS = [
    # Top tier (IC > 0.10)
    "cvd_momentum_adv", "rsi_h4", "ema_21_slope_h4",
    "whale_retail_divergence", "rsi_6", "price_vs_ema_50_h4",
    "log_ret_5", "Sell_Liq", "log_ret_20", "Buy_Liq",
    "ema_50_slope_h4", "h4_trend", "vol_price_confirm",
    "dist_liq_50x_long", "trend_strength", "stochrsi_d",
    "stochrsi_k", "trend_accel_4h", "dist_liq_50x_short",
    "ofi_h4_delta",
    # Mid tier (IC 0.05-0.10)
    "dist_liq_20x_long", "cvd_slope_h4", "volume_delta",
    "long_short_ratio", "dist_liq_20x_short", "log_ret_1",
    "swing_momentum", "rsi_slope_h4", "ofi_raw",
    # Lower tier (IC 0.02-0.05)
    "open_interest", "cvd_div_h4", "funding_rate", "cvd",
    "atr_zscore_20d", "atr_percentile_h1", "atr_percent_h4",
    "ofi_z_score", "rsi_divergence",
]

LABEL_MAP_MOM = {"BEARISH": 0, "NEUTRAL": 1, "BULLISH": 2}


def load_data(coins):
    X_list, y_list, ts_list = [], [], []
    actual_feats = []

    for coin in coins:
        fp = LABEL_DIR / f"{coin}_features_v3.parquet"
        lp = LABEL_DIR / f"{coin}_momentum_v2_labels.parquet"
        pp = LABEL_DIR / f"{coin}_hmm_probs.parquet"

        if not fp.exists() or not lp.exists():
            continue

        df = pd.read_parquet(fp).sort_index()
        df = df[df.index < TRAIN_CUTOFF_DATE]
        lbl = pd.read_parquet(lp).sort_index()
        df["label"] = lbl["momentum_v2_label"]

        # HMM probs (not used as features in IC38, but keep for reference)
        if pp.exists():
            probs = pd.read_parquet(pp).sort_index()
            for i in range(4):
                c = f"hmm_prob_{i}"
                if c in probs.columns: df[c] = probs[c]

        for c in IC38_FEATS:
            if c not in df.columns: df[c] = 0.0

        df = df.dropna(subset=["label"])
        df = df[df["label"].isin([0, 1, 2])]
        if len(df) < 100: continue

        if not actual_feats:
            actual_feats = [c for c in IC38_FEATS if c in df.columns]

        X = df[actual_feats].ffill().fillna(0)
        y = df["label"].values.astype(np.int64)
        ts = df.index

        X_list.append(X); y_list.append(y); ts_list.append(ts)
        n = len(df)
        logger.info(f"{coin}: {n:,} bars | BULL={sum(y==2)/n*100:.0f}% NEU={sum(y==1)/n*100:.0f}% BEAR={sum(y==0)/n*100:.0f}%")

    X_all = pd.concat(X_list, axis=0)
    y_all = np.concatenate(y_list)
    ts_all = np.concatenate(ts_list)
    return X_all, y_all, ts_all, actual_feats


def main():
    coins = TRAINING_COINS
    print(f"\n{'='*60}")
    print(f"  MOMENTUM IC38 LGBM | 21 coins | 38 KEEP features")
    print(f"  Labels: momentum_v2 | Cutoff: {TRAIN_CUTOFF_DATE}")
    print(f"{'='*60}\n")

    X, y, ts, feat_cols = load_data(coins)
    logger.info(f"Total: {len(X):,} samples, {len(feat_cols)} features")
    for i, lbl in enumerate(["BEARISH", "NEUTRAL", "BULLISH"]):
        logger.info(f"  {lbl}: {(y==i).sum():,} ({(y==i).mean()*100:.1f}%)")

    ts_idx = pd.DatetimeIndex(ts)
    folds = build_purged_folds(ts_idx, N_FOLDS, PURGE_GAP_BARS)

    cv_results = []
    best_f1, best_model = 0, None

    for fi, (tr_idx, te_idx) in enumerate(folds):
        X_tr = X.iloc[tr_idx]; y_tr = y[tr_idx]
        X_te = X.iloc[te_idx]; y_te = y[te_idx]

        model = lgb.LGBMClassifier(**LGBM_PARAMS)
        model.fit(X_tr, y_tr,
                  eval_set=[(X_te, y_te)], eval_metric="multi_logloss",
                  callbacks=[lgb.early_stopping(LGBM_EARLY_STOPPING), lgb.log_evaluation(0)])

        y_pred = np.argmax(model.predict_proba(X_te), axis=1)
        f1 = f1_score(y_te, y_pred, average="macro")
        cv_results.append({"fold": fi+1, "f1": round(f1, 4)})
        logger.info(f"Fold {fi+1}: F1={f1:.4f}")
        if f1 > best_f1: best_f1, best_model = f1, model

    # Final retrain
    final = lgb.LGBMClassifier(**LGBM_PARAMS)
    final.fit(X, y)

    run_dir = MODEL_DIR / "runs" / "momentum_ic38"
    run_dir.mkdir(parents=True, exist_ok=True)

    import joblib
    joblib.dump(final, run_dir / "lgbm.pkl")
    with open(run_dir / "feature_cols.json", "w") as f:
        json.dump(feat_cols, f, indent=2)

    f1s = [m["f1"] for m in cv_results]
    imp = list(zip(feat_cols, final.feature_importances_))
    imp.sort(key=lambda x: x[1], reverse=True)

    print(f"\n{'='*60}")
    print(f"  MOMENTUM IC38 COMPLETE")
    print(f"  CV F1: {np.mean(f1s):.4f} +/- {np.std(f1s):.4f}")
    print(f"  Random baseline: 0.333")
    print(f"  Gain vs random: {np.mean(f1s)-0.333:+.4f}")
    print(f"\n  Top 15 features:")
    for i, (f, v) in enumerate(imp[:15]):
        print(f"  {i+1:>2}. {f:<35} {v:>8.1f}")
    print(f"\n  Model: {run_dir / 'lgbm.pkl'}")
    print(f"  Feats: {run_dir / 'feature_cols.json'}")


if __name__ == "__main__":
    main()
