"""
Analyze: apakah LGBM confidence sudah memberikan lift WR yang cukup?
Input: ic32_oof_trades.parquet (16,093 trades)

Pertanyaan: sebelum invest di binary meta v2,
apakah confidence LGBM sendiri sudah bisa jadi combiner?
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

OOF_PATH = ROOT / "data" / "meta_labels" / "ic32_oof_trades.parquet"

df = pd.read_parquet(OOF_PATH)

# Confidence = max(conf_long, conf_short) per direction
df["conf"] = np.where(df["direction"] == 1, df["conf_long"], df["conf_short"])
df["log_ret_20"] = df.get("log_ret_20", np.nan)

base_wr = df["win"].mean()
print(f"\nBase WR: {base_wr*100:.1f}%  |  N={len(df):,}")
print(f"LONG: {(df['direction']==1).sum():,}  |  SHORT: {(df['direction']==-1).sum():,}")

# ── Lift WR by LGBM confidence quintile ──────────────────────────────────────
print(f"\n{'='*60}")
print("  Lift WR by LGBM confidence (quintile)")
print(f"{'='*60}")
print(f"  {'Quintile':>10} {'Min Conf':>10} {'Max Conf':>10} {'N':>7} {'WR%':>8}")

quintiles = np.quantile(df["conf"], np.arange(0.2, 1.01, 0.2))
prev = df["conf"].min() - 0.001
for q_pct, q_val in zip(range(20, 101, 20), quintiles):
    mask = (df["conf"] > prev) & (df["conf"] <= q_val)
    n = mask.sum()
    wr = df.loc[mask, "win"].mean() * 100 if n > 0 else 0.0
    print(f"  {q_pct:>9}%  {prev:>10.4f}  {q_val:>10.4f}  {n:>7,}  {wr:>7.1f}%")
    prev = q_val

# Top vs bottom half spread
top = df["conf"] >= np.median(df["conf"])
bot = ~top
top_wr = df.loc[top, "win"].mean() * 100
bot_wr = df.loc[bot, "win"].mean() * 100
spread = top_wr - bot_wr
print(f"\n  Top-half WR    : {top_wr:.1f}%")
print(f"  Bottom-half WR : {bot_wr:.1f}%")
print(f"  Spread         : {spread:+.1f} pp  (>15pp = deployable)")

# ── Lift WR by confidence tertile per direction ──────────────────────────────
print(f"\n{'='*60}")
print("  By direction + confidence tertile")
print(f"{'='*60}")
for direction, label in [(1, "LONG"), (-1, "SHORT")]:
    sub = df[df["direction"] == direction]
    t33, t67 = np.quantile(sub["conf"], [0.333, 0.667])
    low  = sub["conf"] <= t33
    mid  = (sub["conf"] > t33) & (sub["conf"] <= t67)
    high = sub["conf"] > t67
    print(f"\n  {label} (n={len(sub):,}, base WR={sub['win'].mean()*100:.1f}%)")
    for lbl, mask in [("Low  (<P33)", low), ("Mid  (P33-P67)", mid), ("High (>P67)", high)]:
        n = mask.sum()
        wr = sub.loc[mask, "win"].mean() * 100 if n > 0 else 0.0
        t_ = sub.loc[mask, "conf"].mean()
        print(f"    {lbl}: conf_avg={t_:.3f}  n={n:,}  WR={wr:.1f}%")

# ── Threshold sweep: if we filter by conf >= X, what's WR? ──────────────────
print(f"\n{'='*60}")
print("  Threshold sweep: conf >= X")
print(f"{'='*60}")
print(f"  {'Threshold':>12} {'N kept':>8} {'Keep%':>7} {'WR%':>8}")
for thr in [0.60, 0.62, 0.64, 0.66, 0.69, 0.71, 0.73, 0.75, 0.80]:
    mask = df["conf"] >= thr
    n = mask.sum()
    keep_pct = n / len(df) * 100
    wr = df.loc[mask, "win"].mean() * 100 if n > 0 else 0.0
    print(f"  {thr:>12.2f} {n:>8,} {keep_pct:>6.1f}%  {wr:>7.1f}%")

# ── Feature correlation check ─────────────────────────────────────────────────
print(f"\n{'='*60}")
print("  Binary meta v1 feature overlap with ic32 (from OOF parquet columns)")
print(f"{'='*60}")
oof_cols = set(df.columns)
ic32_feats = [
    'dist_from_8h_high', 'rsi_6', 'swing_momentum', 'rsi_h4', 'stochrsi_k',
    'dist_liq_50x_long', 'trend_accel_4h', 'rsi_slope_h4', 'Fib_786', 'Fib_618',
    'stochrsi_d', 'ofi_h4_delta', 'dist_liq_20x_long', 'cvd_momentum_adv', 'Sell_Liq',
    'long_short_ratio', 'cvd_slope_h4', 'ema_21_slope_h4', 'ema_50_h1', 'h4_trend',
    'log_ret_20', 'whale_retail_divergence', 'dist_liq_20x_short', 'vol_price_confirm',
    'ema_50_slope_h4', 'MSB_BOS', 'cvd', 'ofi_acceleration', 'cvd_div_h4',
    'hmm_regime_enc', 'dist_liq_50x_short', 'Buy_Liq', 'relative_strength_z'
]
meta_v1_feats = [
    "atr_zscore_20d", "atr_percent_h4", "atr_percentile_h1",
    "funding_rate", "ofi_h4_delta", "ofi_raw",
    "cvd_slope_h4", "cvd_momentum_adv",
    "rsi_6", "rsi_h4", "log_ret_1", "log_ret_5",
    "swing_momentum", "trend_strength", "ema_50_slope_h4",
]
ic32_set = set(ic32_feats)
print(f"\n  Binary meta v1 features ({len(meta_v1_feats)}):")
for f in meta_v1_feats:
    in_ic32 = "IN ic32" if f in ic32_set else "NOT in ic32"
    in_oof  = "(in OOF parquet)" if f in oof_cols else ""
    print(f"    {f:<30} {in_ic32:<14} {in_oof}")

# ── Rank IC of conf vs win ───────────────────────────────────────────────────
from scipy import stats
ic, _ = stats.spearmanr(df["conf"], df["win"])
n = len(df)
t = ic * np.sqrt((n - 2) / (1 - ic**2 + 1e-9))
print(f"\n{'='*60}")
print(f"  Rank IC (LGBM conf vs WIN): IC={ic:+.4f}  t={t:.1f}")
print(f"  Compared to binary meta v1: IC=+0.0682, t=9.77")
print(f"{'='*60}\n")
