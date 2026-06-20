"""
pipeline/04b_feature_sample_train.py -- Sample Training 5 Koin untuk Feature Selection

Tujuan:
  Resolusi redundansi fitur dengan membiarkan LGBM memutuskan via feature importance.
  Throw in semua kandidat (current 27 + new), lihat mana yang model pakai.

5 Koin representatif:
  BTCUSDT   -- large cap reference
  SOLUSDT   -- high-beta alt, trend-following
  DOGEUSDT  -- meme coin, volatility-driven
  XRPUSDT   -- alt layer-1, regulatory/macro sensitive
  ARBUSDT   -- small cap, newer listing

Metodologi: sama persis dengan genuine_v1 (purged CV, TB labels, same LGBM params).
Hanya berbeda di: jumlah koin (5 vs 21) dan feature set (diperluas).

Output:
  - feature_importance.csv : gain + split score per fitur
  - Rekomendasi final feature set untuk genuine_v2
"""

import json, sys, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import joblib, lightgbm as lgb
from sklearn.metrics import f1_score

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from pipeline.shared import build_purged_folds
from core.features import triple_barrier_labeling
from core.utils import setup_logger, ensure_utc_index
from config import (
    TRAIN_CUTOFF_DATE, N_FOLDS, TB_PURGE_GAP_BARS,
    MAX_HOLDING_BARS, MODEL_DIR, LABEL_DIR,
    TP_SL_FALLBACK_TP, TP_SL_FALLBACK_SL,
)

logger = setup_logger("04b_feature_sample_train")

SAMPLE_COINS = ["BTCUSDT", "SOLUSDT", "DOGEUSDT", "XRPUSDT", "ARBUSDT"]

TP_ATR   = TP_SL_FALLBACK_TP   # 2.0
SL_ATR   = TP_SL_FALLBACK_SL   # 1.5
MAX_HOLD = MAX_HOLDING_BARS     # 36

LGBM_PARAMS = {
    "objective":         "multiclass",
    "num_class":         3,
    "n_estimators":      600,
    "learning_rate":     0.03,
    "max_depth":         5,
    "num_leaves":        31,
    "min_child_samples": 50,
    "subsample":         0.8,
    "colsample_bytree":  0.7,
    "class_weight":      "balanced",
    "verbose":           -1,
    "n_jobs":            -1,
    "random_state":      42,
}
EARLY_STOP = 50

# ─── Feature set ─────────────────────────────────────────────────────────────
# ic32_regime (33) -- buang 6 yang bermasalah, tambah 8 macro baru
# Dibuang dari ic32: cvd (raw), ema_50_h1 (abs price), Fib_618/786 (abs price),
#                   MSB_BOS (0.05% gain), long_short_ratio (IC~0)

IC32_BASE = [
    # Liquidity distances
    "dist_from_8h_high", "dist_liq_50x_long", "dist_liq_50x_short",
    "dist_liq_20x_long", "dist_liq_20x_short",
    # Momentum / oscillator
    "rsi_6", "rsi_h4", "rsi_slope_h4", "stochrsi_d", "stochrsi_k",
    "swing_momentum", "trend_accel_4h",
    # CVD / Order flow
    "cvd_slope_h4", "cvd_momentum_adv", "cvd_div_h4", "ofi_h4_delta",
    "ofi_acceleration",
    # Liquidity / structure
    "Buy_Liq", "Sell_Liq",
    # EMA slope / trend
    "ema_21_slope_h4", "ema_50_slope_h4", "h4_trend",
    # Others
    "relative_strength_z", "whale_retail_divergence",
    "log_ret_20", "vol_price_confirm",
    # Regime
    "hmm_regime_enc",
]

# Tambahan macro yang ic32 tidak punya -- ini yang menjawab concern bull/bear
MACRO_NEW = [
    "ret_7d",           # computed: log return 7 hari
    "ret_14d",          # computed: log return 14 hari
    "ret_30d",          # computed: log return 30 hari
    "atr_zscore_20d",   # ATR normalized -- rank #2 di sample sebelumnya
    "atr_percentile_h1",# ATR rank H1 -- rank #7
    "funding_rate",     # sentiment/cost -- rank #9
    "dist_pdh",         # computed: jarak dari Previous Day High
    "dist_pdl",         # computed: jarak dari Previous Day Low
]

ALL_CANDIDATES = IC32_BASE + MACRO_NEW
CURRENT_27 = set(IC32_BASE)  # untuk marking di output

# Kolom computed yang perlu dihitung dari raw data
COMPUTED_FEATS = {
    "ret_30d": lambda d: np.log(d["close"] / d["close"].shift(2880).replace(0, np.nan)),
    "ret_14d": lambda d: np.log(d["close"] / d["close"].shift(1344).replace(0, np.nan)),
    "ret_7d":  lambda d: np.log(d["close"] / d["close"].shift(672).replace(0, np.nan)),
    "dist_pdl": lambda d: d["close"] / d["PDL"].replace(0, np.nan) - 1,
    "dist_pdh": lambda d: d["close"] / d["PDH"].replace(0, np.nan) - 1,
}


def load_data() -> pd.DataFrame:
    frames = []
    for sym in SAMPLE_COINS:
        path = LABEL_DIR / f"{sym}_features_v3.parquet"
        if not path.exists():
            logger.warning(f"[{sym}] File tidak ditemukan")
            continue
        df = pd.read_parquet(path)
        df = ensure_utc_index(df).sort_index()
        df = df[df.index < TRAIN_CUTOFF_DATE]

        # Compute tambahan features
        for feat, fn in COMPUTED_FEATS.items():
            try:
                df[feat] = fn(df)
            except Exception as e:
                logger.warning(f"[{sym}] Computed feat {feat} gagal: {e}")

        # TB labels
        tb = triple_barrier_labeling(
            df["close"], df["high"], df["low"],
            df["atr_14_h1"], TP_ATR, SL_ATR, MAX_HOLD,
        )
        df["tb_label"] = tb.map({"SHORT": 0, "FLAT": 1, "LONG": 2})
        df = df.dropna(subset=["tb_label"])
        df["coin"] = sym

        dist = df["tb_label"].value_counts()
        n = len(df)
        logger.info(f"[{sym}] {n:,} bars | "
                    f"LONG={dist.get(2,0)/n*100:.1f}% "
                    f"SHORT={dist.get(0,0)/n*100:.1f}% "
                    f"FLAT={dist.get(1,0)/n*100:.1f}%")
        frames.append(df)

    if not frames:
        raise RuntimeError("Tidak ada data!")
    return pd.concat(frames).sort_index()


def main():
    print("=" * 65)
    print("  SAMPLE TRAINING 5 KOIN -- Feature Importance Analysis")
    print(f"  Koin    : {', '.join(SAMPLE_COINS)}")
    print(f"  Fitur   : {len(ALL_CANDIDATES)} ({len(IC32_BASE)} ic32-base + {len(MACRO_NEW)} macro-new)")
    print(f"  Method  : Purged Walk-Forward CV ({N_FOLDS} folds)")
    print("=" * 65)

    # Load data
    df = load_data()

    # Verifikasi ketersediaan fitur
    available = [f for f in ALL_CANDIDATES if f in df.columns]
    missing   = [f for f in ALL_CANDIDATES if f not in df.columns]
    if missing:
        logger.warning(f"Missing features (akan di-skip): {missing}")
    print(f"\n  Fitur tersedia: {len(available)}/{len(ALL_CANDIDATES)}")
    if missing:
        print(f"  Missing       : {missing}")

    # Prepare X, y
    X_all = df[available].ffill().fillna(0).values.astype(np.float32)
    y_all = df["tb_label"].values.astype(np.int32)
    idx   = np.arange(len(df))

    # Purged walk-forward CV
    folds = build_purged_folds(df.index, N_FOLDS, TB_PURGE_GAP_BARS)
    print(f"\n[CV] {len(folds)} folds, purge={TB_PURGE_GAP_BARS} bars\n")

    fold_f1s    = []
    importance_gain  = np.zeros(len(available))
    importance_split = np.zeros(len(available))

    for k, (tr_idx, val_idx) in enumerate(folds, 1):
        X_tr, y_tr = X_all[tr_idx], y_all[tr_idx]
        X_val, y_val = X_all[val_idx], y_all[val_idx]

        model = lgb.LGBMClassifier(**LGBM_PARAMS)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            callbacks=[
                lgb.early_stopping(EARLY_STOP, verbose=False),
                lgb.log_evaluation(period=-1),
            ],
        )
        y_pred = model.predict(X_val)
        f1 = f1_score(y_val, y_pred, average="macro")
        fold_f1s.append(f1)
        logger.info(f"  Fold {k}: F1={f1:.4f} | iter={model.best_iteration_}")

        importance_gain  += model.booster_.feature_importance(importance_type="gain")
        importance_split += model.booster_.feature_importance(importance_type="split")

    mean_f1 = np.mean(fold_f1s)
    print(f"\n  OOF Mean F1 macro: {mean_f1:.4f} (+/- {np.std(fold_f1s):.4f})")

    # ─── Feature Importance Table ─────────────────────────────────────────────
    importance_df = pd.DataFrame({
        "feature":    available,
        "gain":       importance_gain / len(folds),
        "split":      importance_split / len(folds),
        "in_ic32":    [f in IC32_BASE for f in available],
        "in_current": [f in IC32_BASE for f in available],
    }).sort_values("gain", ascending=False).reset_index(drop=True)

    # Normalize gain to %
    total_gain = importance_df["gain"].sum()
    importance_df["gain_pct"] = importance_df["gain"] / total_gain * 100

    print(f"\n{'='*75}")
    print(f"  FEATURE IMPORTANCE (sorted by gain%)")
    print(f"  [I] = dari ic32_base  [N] = macro baru")
    print(f"{'='*75}")
    print(f"  {'#':>2} {'Feature':<28} {'gain%':>7} {'split':>7}  status")
    print(f"  {'-'*65}")

    cumulative = 0.0
    cutoff_80  = None
    for i, row in importance_df.iterrows():
        marker  = "[I]" if row["in_ic32"] else "[N]"
        cumulative += row["gain_pct"]
        if cumulative >= 80 and cutoff_80 is None:
            cutoff_80 = i + 1
        # Highlight low-importance current features
        note = ""
        if row["gain_pct"] < 0.5:
            note = " << LEMAH"
        elif row["gain_pct"] > 5.0:
            note = " <<"
        print(f"  {i+1:>2} {marker} {row['feature']:<25} {row['gain_pct']:>6.2f}% "
              f"{row['split']:>7.0f}  cum={cumulative:.0f}%{note}")

    print(f"\n  Top-{cutoff_80} fitur = 80% total gain")

    # ─── Summary: keep / add / drop ───────────────────────────────────────────
    KEEP_THRESHOLD   = 0.3   # % gain -- di bawah ini lemah
    strong = importance_df[importance_df["gain_pct"] >= KEEP_THRESHOLD]
    weak   = importance_df[importance_df["gain_pct"] <  KEEP_THRESHOLD]

    macro_weak  = weak[~weak["in_ic32"]]
    ic32_weak   = weak[weak["in_ic32"]]

    print(f"\n{'='*65}")
    print(f"  HASIL EVALUASI FEATURE SET")
    print(f"{'='*65}")

    print(f"\n  [N] MACRO BARU yang kuat (gain >= {KEEP_THRESHOLD}%):")
    macro_strong = strong[~strong["in_ic32"]]
    if len(macro_strong) > 0:
        for _, r in macro_strong.iterrows():
            print(f"    + {r['feature']:<28}  gain={r['gain_pct']:.2f}%")
    else:
        print("    (tidak ada)")

    print(f"\n  [I] IC32 yang LEMAH (gain < {KEEP_THRESHOLD}% -- kandidat drop):")
    if len(ic32_weak) > 0:
        for _, r in ic32_weak.iterrows():
            print(f"    - {r['feature']:<28}  gain={r['gain_pct']:.2f}%")
    else:
        print("    (semua ic32 features masih berkontribusi)")

    print(f"\n  [N] MACRO BARU yang lemah (gain < {KEEP_THRESHOLD}% -- skip):")
    if len(macro_weak) > 0:
        for _, r in macro_weak.iterrows():
            print(f"    - {r['feature']:<28}  gain={r['gain_pct']:.2f}%")
    else:
        print("    (semua macro additions berkontribusi)")

    # Recommended final feature set
    final_set = list(strong["feature"])
    print(f"\n  Final feature set yang direkomendasikan: {len(final_set)} fitur")

    # ─── Save ────────────────────────────────────────────────────────────────
    out_dir = MODEL_DIR / "runs" / "tb_lgbm_genuine_v1"
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "sample_feature_importance.csv"
    importance_df.to_csv(csv_path, index=False)
    print(f"\n  Saved: {csv_path}")

    json_path = out_dir / "sample_recommended_features.json"
    with open(json_path, "w") as f:
        json.dump({
            "created":       datetime.now().isoformat(),
            "coins":         SAMPLE_COINS,
            "oof_f1":        round(mean_f1, 4),
            "n_candidates":  len(available),
            "recommended":   final_set,
            "n_recommended": len(final_set),
        }, f, indent=2)
    print(f"  Saved: {json_path}")
    print()


if __name__ == "__main__":
    main()
