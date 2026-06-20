"""
pipeline/05_feature_ic_test.py ? IC Test Kandidat Fitur Macro-Horizon

Tujuan: Ukur Information Coefficient (IC = Spearman rank corr) kandidat fitur
macro terhadap arah Triple Barrier label ? SEBELUM diputuskan masuk ke LGBM retrain.

Tidak ada model yang dilatih. Tidak menyentuh holdout.
Hanya membaca training data (< 2026-04-01) + OOF labels dari genuine_v1.

IC interpretation:
  IC > 0  : fitur naik -> lebih banyak LONG outcome (momentum)
  IC < 0  : fitur naik -> lebih banyak SHORT outcome (mean-reversion)
  |IC| > 0.02 : biasanya sudah meaningful untuk tree model
  |IC| > 0.05 : kuat, layak dipertimbangkan
  t-stat > 2   : statistik signifikan di level 95%

Kandidat yang ditest:
  Kelompok A: Macro time-horizon (return multi-hari, jarak dari EMA200)
  Kelompok B: Existing features yg belum masuk LGBM (ETF flow, fear_greed, HMM, dll)
  Kelompok C: Weekly structure (Previous Week High/Low distance)
"""

import sys, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from config import ALL_COINS, LABEL_DIR, MODEL_DIR

# --- Kandidat fitur -----------------------------------------------------------

# Fitur yang SUDAH ada di parquet (tidak perlu dihitung)
EXISTING_COLS = [
    "h4_trend",             # A: H4 trend direction (existing ? tapi tidak di 27-feat LGBM)
    "hmm_regime_enc",       # A: HMM regime 0-3
    "fear_greed",           # B: Fear & Greed index (0-100)
    "btc_dominance",        # B: BTC dominance %
    "etf_total_change_usd", # B: Total ETF inflow/outflow (daily)
    "etf_gbtc_change_usd",  # B: GBTC inflow/outflow
    "log_ret_1",            # C: 1-bar return (15m)
    "log_ret_5",            # C: 5-bar return (~75m)
    "long_short_ratio",     # B: Global L/S ratio
    "price_in_range",       # B: Price position dalam H4 range
    "trend_strength",       # A: Trend strength indicator
]

# Fitur yang DIHITUNG dari kolom existing
COMPUTED_COLS = [
    "dist_ema200_h4",   # close/ema_200_h4 - 1  (% di atas/bawah EMA200 H4)
    "dist_ema200_h1",   # close/ema_200_h1 - 1  (% di atas/bawah EMA200 H1)
    "ret_1d",           # log(close / close.shift(96))    ? 24 jam (96 ? 15m)
    "ret_3d",           # log(close / close.shift(288))   ? 3 hari
    "ret_7d",           # log(close / close.shift(672))   ? 7 hari
    "ret_30d",          # log(close / close.shift(2880))  ? 30 hari
    "dist_pwh",         # close/PWH - 1  (jarak dari Previous Week High)
    "dist_pwl",         # close/PWL - 1  (jarak dari Previous Week Low)
    "above_ema200_h4",  # binary: 1 jika close > ema_200_h4, else 0
]

ALL_CANDIDATES = EXISTING_COLS + COMPUTED_COLS

# --- Load OOF labels ----------------------------------------------------------

oof_path = MODEL_DIR / "runs" / "tb_lgbm_genuine_v1" / "oof_predictions.parquet"
oof = pd.read_parquet(oof_path)
oof = oof[oof["has_oof"] == True][["coin", "tb_label"]].copy()

# Direction: LONG=+1, SHORT=-1, FLAT=exclude
oof["direction"] = oof["tb_label"].map({2: 1.0, 0: -1.0, 1: np.nan})

print("=" * 65)
print("  IC TEST ? Kandidat Fitur Macro-Horizon")
print(f"  OOF bars (has_oof): {len(oof):,}")
print(f"  Training period   : 2020-01-01 -> 2026-04-01")
print(f"  Candidates        : {len(ALL_CANDIDATES)} fitur")
print("=" * 65)

# --- IC computation per coin -------------------------------------------------

results = []

for coin in ALL_COINS:
    feat_path = LABEL_DIR / f"{coin}_features_v3.parquet"
    if not feat_path.exists():
        continue

    df = pd.read_parquet(feat_path)
    df = df[df.index < pd.Timestamp("2026-04-01", tz="UTC")]

    # Get OOF direction for this coin
    coin_dir = oof[oof["coin"] == coin][["direction"]].copy()
    if len(coin_dir) == 0:
        continue

    # Join on timestamp index
    merged = df.join(coin_dir, how="inner")
    merged = merged.dropna(subset=["direction"])  # remove FLAT + missing
    if len(merged) < 200:
        continue

    # Compute new features
    merged = merged.copy()
    merged["dist_ema200_h4"]  = merged["close"] / merged["ema_200_h4"].replace(0, np.nan) - 1
    merged["dist_ema200_h1"]  = merged["close"] / merged["ema_200_h1"].replace(0, np.nan) - 1
    merged["above_ema200_h4"] = (merged["close"] > merged["ema_200_h4"]).astype(float)
    merged["ret_1d"]  = np.log(merged["close"] / merged["close"].shift(96).replace(0, np.nan))
    merged["ret_3d"]  = np.log(merged["close"] / merged["close"].shift(288).replace(0, np.nan))
    merged["ret_7d"]  = np.log(merged["close"] / merged["close"].shift(672).replace(0, np.nan))
    merged["ret_30d"] = np.log(merged["close"] / merged["close"].shift(2880).replace(0, np.nan))
    merged["dist_pwh"] = merged["close"] / merged["PWH"].replace(0, np.nan) - 1
    merged["dist_pwl"] = merged["close"] / merged["PWL"].replace(0, np.nan) - 1

    # IC per candidate
    for feat in ALL_CANDIDATES:
        if feat not in merged.columns:
            continue
        sub = merged[["direction", feat]].dropna()
        if len(sub) < 100:
            continue
        ic, pval = spearmanr(sub[feat], sub["direction"])
        results.append({
            "coin":    coin,
            "feature": feat,
            "IC":      ic,
            "pval":    pval,
            "n":       len(sub),
        })

# --- Aggregate ----------------------------------------------------------------

df_res = pd.DataFrame(results)

summary = (
    df_res.groupby("feature")
    .agg(
        mean_IC    = ("IC", "mean"),
        std_IC     = ("IC", "std"),
        median_IC  = ("IC", "median"),
        n_coins    = ("IC", "count"),
        hit_rate   = ("IC", lambda x: (x > 0).mean()),
        pval_05_pct= ("pval", lambda x: (x < 0.05).mean()),
    )
    .assign(t_stat=lambda d: d["mean_IC"] / (d["std_IC"] / d["n_coins"].pow(0.5)))
    .sort_values("mean_IC", key=abs, ascending=False)
    .round(4)
)

print("\n-- Semua Kandidat (sorted by |IC|) --\n")
print(f"{'Feature':<25} {'mean_IC':>8} {'median':>8} {'t_stat':>7} {'hit%':>6} {'sig%':>6}")
print("-" * 65)
for feat, row in summary.iterrows():
    flag = ""
    if abs(row["mean_IC"]) > 0.05:
        flag = " **"
    elif abs(row["mean_IC"]) > 0.02:
        flag = " *"
    print(f"{feat:<25} {row['mean_IC']:>8.4f} {row['median_IC']:>8.4f} "
          f"{row['t_stat']:>7.2f} {row['hit_rate']*100:>5.0f}% {row['pval_05_pct']*100:>5.0f}%{flag}")

print("\nLegenda:")
print("  mean_IC  : rata-rata Spearman IC lintas 21 koin")
print("  t_stat   : signifikansi statistik (>2 = significant)")
print("  hit%     : % koin dengan IC > 0 (> 60% = konsisten arah)")
print("  sig%     : % koin dengan p-value < 0.05")
print("  ** |IC|>0.05 sangat kuat  |  * |IC|>0.02 layak dipertimbangkan")

print("\n-- Rekomendasi untuk LGBM --")
strong = summary[abs(summary["mean_IC"]) > 0.02]
print(f"  Kandidat dengan |IC| > 0.02 ({len(strong)} fitur):")
for feat, row in strong.iterrows():
    direction = "momentum" if row["mean_IC"] > 0 else "mean-reversion"
    print(f"    {feat:<25}  IC={row['mean_IC']:+.4f}  ({direction})")

print()
