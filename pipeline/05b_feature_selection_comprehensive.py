"""
pipeline/05b_feature_selection_comprehensive.py
Comprehensive Walk-Forward IC Feature Selection untuk LGBM retrain.

Metodologi:
  1. Test semua fitur yang tersedia di features_v3.parquet (120 col)
  2. Tambah kandidat baru (dist_ema200, ret multi-hari, dll)
  3. Walk-forward IC: hitung IC per tahun x per koin -> statistik temporal
  4. Korelasi antar fitur top-30 (redundancy check)
  5. Bandingkan dengan 27 fitur flatboost_v2 yang ada sekarang

IC Gate criteria:
  mean_IC  : |IC| > 0.01 minimum, > 0.02 layak, > 0.05 kuat
  t_stat   : > 2.0 (95% significance)
  hit_rate : > 55% koin dengan IC searah
  ic_ir    : IC Information Ratio (mean/std) > 0.3 artinya stabil lintas waktu

Output:
  reports/experiments/YYYY-MM-DD_feature_selection.csv
  reports/experiments/YYYY-MM-DD_feature_selection_report.md
"""

import sys, warnings, json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from config import ALL_COINS, LABEL_DIR, MODEL_DIR, REPORT_DIR

TODAY = datetime.now().strftime("%Y-%m-%d")

# ─── Kolom yang DIKECUALIKAN ─────────────────────────────────────────────────
# Raw OHLCV -- bukan fitur signal
RAW_OHLCV = {"open", "high", "low", "close", "volume", "buy_volume", "sell_volume", "cvd"}

# Absolute price levels -- pakai versi relatif (dist_, ratio) bukan harga absolut
ABS_PRICE = {
    "ema_7_h1", "ema_21_h1", "ema_50_h1", "ema_200_h1",
    "ema_7_h4", "ema_21_h4", "ema_50_h4", "ema_200_h4",
    "PDH", "PDL", "PWH", "PWL", "Fib_618", "Fib_786",
    "h4_swing_high", "h4_swing_low",
}

# LOOK-AHEAD LEAKAGE -- mengandung informasi dari bar masa depan
LOOKAHEAD = {"mfe_8", "mfe_12", "mae_8", "mae_12",
             "time_above_entry_8", "time_below_entry_8", "label"}

# ATR raw -- pakai yg sudah dinormalisasi (atr_percent, atr_percentile)
ATR_RAW = {"atr_14_h1", "atr_14_h4"}

EXCLUDE = RAW_OHLCV | ABS_PRICE | LOOKAHEAD | ATR_RAW

# ─── 27 Fitur flatboost_v2 (LGBM aktif sekarang) ────────────────────────────
CURRENT_27 = {
    "dist_liq_50x_long", "dist_liq_50x_short", "dist_liq_20x_short",
    "dist_from_8h_high", "dist_swing_high", "VAH",
    "cvd_slope_h4", "ofi_h4_delta", "cvd_momentum_adv",
    "trend_accel_4h", "stochrsi_d", "log_ret_20", "atr_percent_h4", "atr_percentile_h1",
    "whale_retail_divergence", "Buy_Liq", "vol_spike_zscore", "range_expansion_h4",
    "ultra_high_vol", "absorption_z", "vol_accel_3h", "vol_ratio_20",
    "no_supply", "no_demand", "effort_vs_result",
    "dow_cos", "funding_rate",
}

# ─── Fitur tambahan yang dihitung (tidak ada di parquet) ─────────────────────
COMPUTED = {
    "dist_ema200_h4":  lambda d: d["close"] / d["ema_200_h4"].replace(0, np.nan) - 1,
    "dist_ema200_h1":  lambda d: d["close"] / d["ema_200_h1"].replace(0, np.nan) - 1,
    "dist_ema50_h4":   lambda d: d["close"] / d["ema_50_h4"].replace(0, np.nan) - 1,
    "above_ema200_h4": lambda d: (d["close"] > d["ema_200_h4"]).astype(float),
    "ret_1d":          lambda d: np.log(d["close"] / d["close"].shift(96).replace(0, np.nan)),
    "ret_3d":          lambda d: np.log(d["close"] / d["close"].shift(288).replace(0, np.nan)),
    "ret_7d":          lambda d: np.log(d["close"] / d["close"].shift(672).replace(0, np.nan)),
    "ret_14d":         lambda d: np.log(d["close"] / d["close"].shift(1344).replace(0, np.nan)),
    "ret_30d":         lambda d: np.log(d["close"] / d["close"].shift(2880).replace(0, np.nan)),
    "dist_pwh":        lambda d: d["close"] / d["PWH"].replace(0, np.nan) - 1,
    "dist_pwl":        lambda d: d["close"] / d["PWL"].replace(0, np.nan) - 1,
    "dist_pdh":        lambda d: d["close"] / d["PDH"].replace(0, np.nan) - 1,
    "dist_pdl":        lambda d: d["close"] / d["PDL"].replace(0, np.nan) - 1,
}

# Walk-forward time windows (per tahun)
TIME_WINDOWS = [
    ("2020", pd.Timestamp("2020-01-01", tz="UTC"), pd.Timestamp("2021-01-01", tz="UTC")),
    ("2021", pd.Timestamp("2021-01-01", tz="UTC"), pd.Timestamp("2022-01-01", tz="UTC")),
    ("2022", pd.Timestamp("2022-01-01", tz="UTC"), pd.Timestamp("2023-01-01", tz="UTC")),
    ("2023", pd.Timestamp("2023-01-01", tz="UTC"), pd.Timestamp("2024-01-01", tz="UTC")),
    ("2024", pd.Timestamp("2024-01-01", tz="UTC"), pd.Timestamp("2025-01-01", tz="UTC")),
    ("2025", pd.Timestamp("2025-01-01", tz="UTC"), pd.Timestamp("2026-04-01", tz="UTC")),
]

# ─── Load OOF labels ──────────────────────────────────────────────────────────
oof_path = MODEL_DIR / "runs" / "tb_lgbm_genuine_v1" / "oof_predictions.parquet"
oof = pd.read_parquet(oof_path)
oof = oof[oof["has_oof"] == True][["coin", "tb_label"]].copy()
oof["direction"] = oof["tb_label"].map({2: 1.0, 0: -1.0, 1: np.nan})

print("=" * 70)
print("  COMPREHENSIVE WALK-FORWARD FEATURE SELECTION")
print(f"  OOF bars     : {len(oof):,}  |  Coins: {len(ALL_COINS)}")
print(f"  Time windows : {len(TIME_WINDOWS)} (per tahun)")
print("=" * 70)

# ─── Main IC computation ──────────────────────────────────────────────────────
records = []   # one row per (feature, coin, time_window)
feat_values_by_coin = {}  # for correlation matrix later

for coin in ALL_COINS:
    feat_path = LABEL_DIR / f"{coin}_features_v3.parquet"
    if not feat_path.exists():
        continue

    df = pd.read_parquet(feat_path)
    df = df[df.index < pd.Timestamp("2026-04-01", tz="UTC")]

    # OOF direction for this coin
    coin_dir = oof[oof["coin"] == coin][["direction"]].copy()
    merged = df.join(coin_dir, how="inner")
    if len(merged) < 200:
        continue

    # Compute engineered candidates
    for feat_name, fn in COMPUTED.items():
        try:
            merged[feat_name] = fn(df).reindex(merged.index)
        except Exception:
            pass

    # Determine feature columns to test
    candidates = [
        c for c in merged.columns
        if c not in EXCLUDE
        and c != "direction"
        and c != "tb_label"
        and pd.api.types.is_numeric_dtype(merged[c])
    ]

    # Store data for correlation matrix (full period)
    active_all = merged[merged["direction"].notna()].copy()
    if coin not in feat_values_by_coin:
        feat_values_by_coin[coin] = active_all[candidates + ["direction"]].copy()

    # IC per time window
    for wname, wstart, wend in TIME_WINDOWS:
        window_data = merged[(merged.index >= wstart) & (merged.index < wend)]
        active = window_data[window_data["direction"].notna()].dropna(subset=["direction"])
        if len(active) < 100:
            continue

        for feat in candidates:
            if feat not in active.columns:
                continue
            sub = active[["direction", feat]].dropna()
            if len(sub) < 50:
                continue
            ic, pval = spearmanr(sub[feat], sub["direction"])
            if np.isnan(ic):
                continue
            records.append({
                "feature": feat,
                "coin":    coin,
                "window":  wname,
                "IC":      ic,
                "pval":    pval,
                "n":       len(sub),
            })

    if ALL_COINS.index(coin) % 5 == 0:
        print(f"  Processed {ALL_COINS.index(coin)+1}/{len(ALL_COINS)} coins...")

df_rec = pd.DataFrame(records)
print(f"\n  Total IC records: {len(df_rec):,}")
print(f"  Features tested : {df_rec['feature'].nunique()}")

# ─── Aggregate: per feature ───────────────────────────────────────────────────
agg = (
    df_rec.groupby("feature")
    .agg(
        mean_IC    = ("IC", "mean"),
        std_IC     = ("IC", "std"),
        median_IC  = ("IC", "median"),
        n_obs      = ("IC", "count"),
        hit_rate   = ("IC", lambda x: (x > 0).mean()),
        sig_pct    = ("pval", lambda x: (x < 0.05).mean()),
    )
    .assign(
        t_stat = lambda d: d["mean_IC"] / (d["std_IC"] / d["n_obs"].pow(0.5)),
        ic_ir  = lambda d: d["mean_IC"].abs() / d["std_IC"],   # IC Information Ratio
    )
    .sort_values("mean_IC", key=abs, ascending=False)
    .round(4)
)

# Mark current-27 membership
agg["in_current_27"] = agg.index.isin(CURRENT_27)

# ─── Print results ────────────────────────────────────────────────────────────
print("\n" + "=" * 90)
print("  HASIL WALK-FORWARD IC (sorted by |mean_IC|)")
print("  [C] = fitur yg sudah ada di 27-feat flatboost_v2")
print("=" * 90)
print(f"  {'Feature':<28} {'IC':>7} {'t_stat':>7} {'IC_IR':>6} {'hit%':>5} {'sig%':>5} {'flag'}")
print("  " + "-" * 80)

for feat, row in agg.iterrows():
    marker = "[C]" if row["in_current_27"] else "   "
    qual = ""
    if abs(row["mean_IC"]) > 0.05 and row["t_stat"] > 3:
        qual = "STRONG"
    elif abs(row["mean_IC"]) > 0.02 and row["t_stat"] > 2:
        qual = "good  "
    elif abs(row["mean_IC"]) > 0.01 and row["t_stat"] > 2:
        qual = "weak  "
    direction_note = "momentum" if row["mean_IC"] > 0 else "mean-rev"
    print(f"  {marker} {feat:<25} {row['mean_IC']:>7.4f} {row['t_stat']:>7.2f} "
          f"{row['ic_ir']:>6.2f} {row['hit_rate']*100:>4.0f}% {row['sig_pct']*100:>4.0f}% "
          f"  {qual} {direction_note if qual else ''}")

# ─── Tabel summary per gate ───────────────────────────────────────────────────
gate_pass = agg[(abs(agg["mean_IC"]) > 0.01) & (agg["t_stat"].abs() > 2.0)]
strong    = agg[(abs(agg["mean_IC"]) > 0.02) & (agg["t_stat"].abs() > 2.0)]
new_cands = strong[~strong["in_current_27"]]
current_weak = agg[agg["in_current_27"] & ((abs(agg["mean_IC"]) <= 0.01) | (agg["t_stat"].abs() <= 2.0))]

print(f"\n{'='*70}")
print(f"  Lulus IC gate (|IC|>0.01, t>2)    : {len(gate_pass)} fitur")
print(f"  Strong (|IC|>0.02, t>2)           : {len(strong)} fitur")
print(f"  NEW kuat (tidak ada di current 27) : {len(new_cands)} fitur")
print(f"  CURRENT 27 yang LEMAH (IC<=0.01)  : {len(current_weak)} fitur")

if len(new_cands) > 0:
    print(f"\n  --- NEW kandidat kuat untuk ditambah ---")
    for feat, row in new_cands.iterrows():
        print(f"  + {feat:<28}  IC={row['mean_IC']:+.4f}  t={row['t_stat']:.1f}  "
              f"hit={row['hit_rate']*100:.0f}%")

if len(current_weak) > 0:
    print(f"\n  --- Fitur CURRENT 27 yang lemah (kandidat drop) ---")
    for feat, row in current_weak.iterrows():
        print(f"  - {feat:<28}  IC={row['mean_IC']:+.4f}  t={row['t_stat']:.1f}")

# ─── Correlation matrix (top 30 fitur) ───────────────────────────────────────
print(f"\n{'='*70}")
print("  KORELASI ANTAR FITUR (top 30 by |IC|) -- redundancy check")
print(f"{'='*70}")

top30 = agg.head(30).index.tolist()

# Concat data across coins for correlation
all_dfs = []
for coin, cdf in feat_values_by_coin.items():
    avail = [f for f in top30 if f in cdf.columns]
    all_dfs.append(cdf[avail].dropna(how="all"))

if all_dfs:
    combined = pd.concat(all_dfs, ignore_index=True)
    avail_top30 = [f for f in top30 if f in combined.columns]
    corr = combined[avail_top30].corr(method="spearman").round(2)

    # Print only high-corr pairs (|r| > 0.7)
    high_corr_pairs = []
    for i, f1 in enumerate(avail_top30):
        for j, f2 in enumerate(avail_top30):
            if j <= i:
                continue
            if f1 not in corr.index or f2 not in corr.columns:
                continue
            r = corr.loc[f1, f2]
            if abs(r) > 0.70:
                high_corr_pairs.append((f1, f2, r))

    if high_corr_pairs:
        print(f"  Pasangan dengan |corr| > 0.70 (kandidat untuk dipilih salah satu):")
        for f1, f2, r in sorted(high_corr_pairs, key=lambda x: -abs(x[2])):
            c1 = "[C]" if f1 in CURRENT_27 else "[ ]"
            c2 = "[C]" if f2 in CURRENT_27 else "[ ]"
            print(f"    {c1} {f1:<28} vs {c2} {f2:<28}  r={r:+.2f}")
    else:
        print("  Tidak ada pasangan dengan korelasi > 0.70 di top-30.")

# ─── Save outputs ─────────────────────────────────────────────────────────────
out_dir = REPORT_DIR / "experiments"
out_dir.mkdir(parents=True, exist_ok=True)

csv_path = out_dir / f"{TODAY}_feature_selection.csv"
agg.to_csv(csv_path)
print(f"\n  CSV saved: {csv_path}")

# Markdown report
md_lines = [
    f"# Feature Selection Report — {TODAY}",
    "",
    "## Metodologi",
    "- Walk-forward IC (Spearman rank) per tahun per koin",
    "- IC gate: |mean_IC| > 0.02 dan t_stat > 2.0",
    f"- Fitur ditest: {df_rec['feature'].nunique()} total",
    f"- Koin: {len(feat_values_by_coin)}  |  Periode: 2020-2026-04",
    "",
    "## Rekomendasi",
    "",
    "### Tambah ke LGBM (NEW, tidak ada di current 27)",
    "",
]
if len(new_cands) > 0:
    md_lines.append("| Fitur | IC | t-stat | hit% | Arah |")
    md_lines.append("|-------|-----|--------|------|------|")
    for feat, row in new_cands.iterrows():
        direction = "momentum" if row["mean_IC"] > 0 else "mean-rev"
        md_lines.append(f"| `{feat}` | {row['mean_IC']:+.4f} | {row['t_stat']:.1f} | {row['hit_rate']*100:.0f}% | {direction} |")
else:
    md_lines.append("*(tidak ada kandidat baru yang lolos gate)*")

md_lines += ["", "### Drop dari LGBM (current 27 yang lemah)", ""]
if len(current_weak) > 0:
    md_lines.append("| Fitur | IC | t-stat |")
    md_lines.append("|-------|-----|--------|")
    for feat, row in current_weak.iterrows():
        md_lines.append(f"| `{feat}` | {row['mean_IC']:+.4f} | {row['t_stat']:.1f} |")
else:
    md_lines.append("*(semua fitur current 27 lulus gate)*")

md_lines += [
    "",
    "## Tabel Lengkap (top-50 by |IC|)",
    "",
    "| # | Fitur | IC | t-stat | IC_IR | hit% | sig% | current? |",
    "|---|-------|----|--------|-------|------|------|----------|",
]
for rank, (feat, row) in enumerate(agg.head(50).iterrows(), 1):
    c = "YES" if row["in_current_27"] else ""
    md_lines.append(
        f"| {rank} | `{feat}` | {row['mean_IC']:+.4f} | {row['t_stat']:.2f} | "
        f"{row['ic_ir']:.2f} | {row['hit_rate']*100:.0f}% | {row['sig_pct']*100:.0f}% | {c} |"
    )

md_path = out_dir / f"{TODAY}_feature_selection_report.md"
with open(md_path, "w", encoding="utf-8") as f:
    f.write("\n".join(md_lines))
print(f"  Report saved: {md_path}")
print()
