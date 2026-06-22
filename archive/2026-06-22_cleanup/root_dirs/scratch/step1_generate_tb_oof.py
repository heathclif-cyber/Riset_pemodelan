"""
scratch/step1_generate_tb_oof.py
Generate TB OOF confidence scores dari tb_lgbm_v3.

Sama dengan 04_train_lgbm_tb_v3.py tapi menyimpan OOF predictions
per-bar (bukan hanya aggregate metrics).

Output: data/meta_labels/tb_lgbm_v3_oof.parquet
Kolom: timestamp, coin, conf_short, conf_flat, conf_long,
        pred_label, true_label, tb_win (LONG→win jika label=LONG, SHORT→win jika label=SHORT)
"""
import sys, warnings
import numpy as np, pandas as pd
from pathlib import Path
import joblib, lightgbm as lgb
from sklearn.metrics import f1_score

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from pipeline.shared import build_purged_folds
from core.features import triple_barrier_labeling
from config import (ALL_COINS, LABEL_DIR, TRAIN_CUTOFF_DATE,
                    N_FOLDS, PURGE_GAP_BARS,
                    TP_SL_FALLBACK_TP, TP_SL_FALLBACK_SL, MAX_HOLDING_BARS)

OUT_PATH = ROOT / "data" / "meta_labels" / "tb_lgbm_v3_oof.parquet"

TB_V3_FEATURES = [
    "etf_gbtc_change_usd", "etf_total_change_usd", "cvd_slope_h4",
    "ofi_h4_delta", "wyckoff_phase", "Sell_Liq", "atr_percentile_h1",
    "stochrsi_k", "dist_liq_50x_short", "funding_rate", "ema_7_h1",
    "dow_cos", "cvd_div_h4", "dist_swing_low", "VAH",
    "cvd_momentum_adv", "dist_from_8h_high", "ema_200_h1",
]

TB_V3_PARAMS = {
    "objective": "multiclass", "num_class": 3,
    "n_estimators": 500, "learning_rate": 0.03,
    "max_depth": 5, "num_leaves": 31, "min_child_samples": 50,
    "subsample": 0.8, "colsample_bytree": 0.7,
    "class_weight": "balanced",
    "verbose": -1, "n_jobs": -1, "random_state": 42,
}


def banner(t, w=65):
    print(f"\n{'='*w}\n  {t}\n{'='*w}")


banner("Step 1: Generate TB OOF Predictions")

# Load + label
print("\nLoading data + TB labeling...")
frames = []
for sym in ALL_COINS:
    path = LABEL_DIR / f"{sym}_features_v3.parquet"
    if not path.exists():
        continue
    df = pd.read_parquet(path)
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, utc=True)
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    df = df[df.index < TRAIN_CUTOFF_DATE]
    if df.empty:
        continue
    tb = triple_barrier_labeling(df["close"], df["high"], df["low"],
                                  df["atr_14_h1"],
                                  TP_SL_FALLBACK_TP, TP_SL_FALLBACK_SL, MAX_HOLDING_BARS)
    df["tb_label"] = tb.map({"SHORT": 0, "FLAT": 1, "LONG": 2})
    df["coin"] = sym
    frames.append(df.dropna(subset=["tb_label"]))

df_all = pd.concat(frames).sort_index()
print(f"  Total bars: {len(df_all):,} | Coins: {df_all['coin'].nunique()}")

avail = [c for c in TB_V3_FEATURES if c in df_all.columns]
X = df_all[avail].ffill().fillna(0)
y = df_all["tb_label"].astype(np.int32).values

# Purged CV — save OOF predictions
print("\nRunning 8-fold purged CV (saving OOF)...")
folds    = build_purged_folds(X.index, N_FOLDS, PURGE_GAP_BARS)
oof_rows = []

for fold, (tr_idx, val_idx) in enumerate(folds, 1):
    X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
    y_tr, y_val = y[tr_idx],      y[val_idx]

    model = lgb.LGBMClassifier(**TB_V3_PARAMS)
    model.fit(X_tr, y_tr,
              eval_set=[(X_val, y_val)],
              callbacks=[lgb.early_stopping(50, verbose=False),
                         lgb.log_evaluation(period=-1)])

    proba = model.predict_proba(X_val)   # shape (N, 3): [SHORT, FLAT, LONG]
    pred  = model.predict(X_val)

    f1 = f1_score(y_val, pred, average="macro", zero_division=0)
    print(f"  Fold {fold}: F1={f1:.4f} | best_iter={model.best_iteration_} | n_val={len(X_val):,}")

    val_df = df_all.iloc[val_idx][["coin", "tb_label"]].copy()
    val_df["conf_short"] = proba[:, 0]
    val_df["conf_flat"]  = proba[:, 1]
    val_df["conf_long"]  = proba[:, 2]
    val_df["pred_label"] = pred
    val_df["fold"]       = fold
    oof_rows.append(val_df)

oof = pd.concat(oof_rows).sort_index()

# Derived columns for IC test
# "win" for the entry direction:
#   LONG entry (pred=2) → win jika true_label=2 (price hit TP going up)
#   SHORT entry (pred=0) → win jika true_label=0 (price hit TP going down)
#   FLAT / wrong direction → 0
oof["entry_direction"] = oof["pred_label"].map({0: "SHORT", 1: "FLAT", 2: "LONG"})
oof["conf_entry"] = np.where(
    oof["pred_label"] == 2, oof["conf_long"],
    np.where(oof["pred_label"] == 0, oof["conf_short"], oof["conf_flat"])
)
# win = predicted correctly (excluding FLAT)
oof["win"] = ((oof["pred_label"] == oof["tb_label"]) &
              (oof["pred_label"] != 1)).astype(int)

# Save
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
oof.to_parquet(OUT_PATH)

banner("Step 1 Complete")
total      = len(oof)
directional = oof[oof["pred_label"] != 1]
print(f"\n  OOF bars total         : {total:,}")
print(f"  Directional entries    : {len(directional):,}")
print(f"    LONG                 : {(oof['pred_label']==2).sum():,}")
print(f"    SHORT                : {(oof['pred_label']==0).sum():,}")
print(f"    FLAT                 : {(oof['pred_label']==1).sum():,}")
print(f"  Entry WR               : {oof['win'].mean():.3f}")
print(f"  Avg conf_long          : {oof['conf_long'].mean():.4f}")
print(f"  Avg conf_short         : {oof['conf_short'].mean():.4f}")
print(f"\n  Saved -> {OUT_PATH}")
print(f"\n  Next: python scratch/step2_temporal_features.py")
