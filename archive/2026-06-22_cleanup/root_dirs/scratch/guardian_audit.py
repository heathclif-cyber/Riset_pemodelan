"""
scratch/guardian_audit.py
Guardian deep audit — tiga pertanyaan Simon:
  1. Fitur apa yang mendominasi Guardian decisions?
  2. Di kondisi apa Guardian gagal (20.4% miss)?
  3. Apakah 79.6% WR robust di walk-forward?

Input:
  models/runs/ic32_guardian_clean_v2/guardian.pkl
  models/runs/ic32_guardian_clean_v2/holdout_trade_history.csv
  data/meta_labels/ic32_oof_trades.parquet  (untuk walk-forward proxy)
"""
import json, sys, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from scipy import stats

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

GD_DIR   = ROOT / "models" / "runs" / "ic32_guardian_clean_v2"
OOF_PATH = ROOT / "data" / "meta_labels" / "ic32_oof_trades.parquet"
OUT_PATH = GD_DIR / "guardian_audit.json"

STATIC_FEAT_COUNT = 33   # market snapshot features (same as LGBM ic32)
DYNAMIC_FEATS = [
    "bars_held_norm", "current_pnl_pct", "current_pnl_atr",
    "max_favorable_pnl_pct", "drawdown_from_peak_pct",
    "direction", "entry_price_ratio",
]


def banner(title, width=65):
    print(f"\n{'='*width}")
    print(f"  {title}")
    print(f"{'='*width}")


# ── Load model ────────────────────────────────────────────────────────────────
model = joblib.load(GD_DIR / "guardian.pkl")
with open(GD_DIR / "guardian_feature_cols.json") as f:
    feat_cols = json.load(f)

static_feats  = feat_cols[:STATIC_FEAT_COUNT]
dynamic_feats = feat_cols[STATIC_FEAT_COUNT:]

banner("GUARDIAN AUDIT — ic32_guardian_clean_v2")
print(f"\n  Total features : {len(feat_cols)}")
print(f"  Static (market): {len(static_feats)}  (same as LGBM ic32)")
print(f"  Dynamic (trade): {len(dynamic_feats)}  {dynamic_feats}")

# ── 1. FEATURE IMPORTANCE ─────────────────────────────────────────────────────
banner("1. FEATURE IMPORTANCE — What drives Guardian decisions?")

importances = model.feature_importances_
feat_imp = sorted(zip(feat_cols, importances), key=lambda x: -x[1])
total_imp = sum(importances)

print(f"\n  {'Rank':>5} {'Feature':<30} {'Importance':>12} {'Cumul%':>8} {'Type':>8}")
cumul = 0.0
for i, (feat, imp) in enumerate(feat_imp[:20], 1):
    cumul += imp / total_imp * 100
    ftype = "DYNAMIC" if feat in dynamic_feats else "static"
    flag  = " <--" if feat in dynamic_feats else ""
    print(f"  {i:>5} {feat:<30} {imp/total_imp*100:>11.2f}% {cumul:>7.1f}% {ftype:>8}{flag}")

# Concentration check
top5_pct  = sum(imp for _, imp in feat_imp[:5]) / total_imp * 100
top10_pct = sum(imp for _, imp in feat_imp[:10]) / total_imp * 100
dyn_pct   = sum(imp for f, imp in feat_imp if f in dynamic_feats) / total_imp * 100
sta_pct   = 100 - dyn_pct

print(f"\n  Concentration:")
print(f"    Top-5 features  : {top5_pct:.1f}% of total importance")
print(f"    Top-10 features : {top10_pct:.1f}% of total importance")
print(f"    Dynamic features: {dyn_pct:.1f}% (trade state)")
print(f"    Static features : {sta_pct:.1f}% (market snapshot)")

if dyn_pct > 60:
    print(f"\n  [FINDING] Guardian is primarily a TRADE-STATE model")
    print(f"  Edge comes from: how long held, unrealized PnL, drawdown from peak")
    print(f"  Implication: robust to market regime shift (not market-structure dependent)")
elif sta_pct > 60:
    print(f"\n  [FINDING] Guardian is primarily a MARKET-SNAPSHOT model")
    print(f"  Edge comes from: price position, momentum, liquidity features")
    print(f"  Implication: fragile if market structure regime shifts")
else:
    print(f"\n  [FINDING] Guardian uses balanced mix of trade-state + market features")

audit = {
    "feature_importance": [{"feature": f, "importance": round(float(i/total_imp), 5)}
                             for f, i in feat_imp],
    "concentration": {
        "top5_pct": round(top5_pct, 2),
        "top10_pct": round(top10_pct, 2),
        "dynamic_pct": round(dyn_pct, 2),
        "static_pct": round(sta_pct, 2),
    },
}

# ── 2. FAILURE ANALYSIS ───────────────────────────────────────────────────────
banner("2. FAILURE ANALYSIS — When does Guardian miss?")

df = pd.read_csv(GD_DIR / "holdout_trade_history.csv")
df["win"]      = (df["PnL ($)"] > 0).astype(int)
df["gd_exit"]  = df["Exit Reason"].str.contains("Guardian|guardian|GUARDIAN", na=False)
df["sl_exit"]  = df["Exit Reason"].str.contains("SL|stop", na=False, regex=False)
df["Direction_num"] = df["Direction"].map({"LONG": 1, "SHORT": -1}).fillna(0)

n_total   = len(df)
n_gd      = df["gd_exit"].sum()
n_sl      = df["sl_exit"].sum()
n_time    = n_total - n_gd - n_sl
wr_total  = df["win"].mean() * 100
wr_gd     = df.loc[df["gd_exit"], "win"].mean() * 100
wr_sl     = df.loc[df["sl_exit"], "win"].mean() * 100
wr_time   = df.loc[~df["gd_exit"] & ~df["sl_exit"], "win"].mean() * 100

print(f"\n  Exit breakdown:")
print(f"  {'Type':<18} {'N':>7} {'WR%':>8} {'AvgPnL':>10}")
for label, mask, wr in [
    ("Guardian exit", df["gd_exit"],             wr_gd),
    ("SL hit",        df["sl_exit"],             wr_sl),
    ("Time exit",     ~df["gd_exit"]&~df["sl_exit"], wr_time),
]:
    sub = df[mask]
    avg_pnl = sub["PnL ($)"].mean()
    print(f"  {label:<18} {mask.sum():>7,} {wr:>7.1f}% {avg_pnl:>+10.3f}")

# Guardian MISS = Guardian exited but trade was a LOSS
gd_trades = df[df["gd_exit"]].copy()
gd_win    = gd_trades[gd_trades["win"] == 1]
gd_loss   = gd_trades[gd_trades["win"] == 0]

print(f"\n  Guardian exits breakdown (total {n_gd:,}):")
print(f"    WIN  ({wr_gd:.1f}%): {len(gd_win):,}  avg PnL = {gd_win['PnL ($)'].mean():+.3f}")
print(f"    LOSS ({100-wr_gd:.1f}%): {len(gd_loss):,}  avg PnL = {gd_loss['PnL ($)'].mean():+.3f}")

# EV of Guardian exits
ev_gd = gd_trades["PnL ($)"].mean()
print(f"    EV per Guardian exit: {ev_gd:+.3f}")

# ── 2A: Failure by Hold Duration ─────────────────────────────────────────────
print(f"\n  Guardian MISS by hold duration:")
hold_bins  = [0, 3, 6, 10, 18, 100]
hold_labs  = ["1-3", "4-6", "7-10", "11-18", "19+"]
gd_trades["hold_bin"] = pd.cut(gd_trades["Hold Bars"], bins=hold_bins, labels=hold_labs)
print(f"  {'Hold bars':<12} {'N':>7} {'WR%':>8} {'AvgPnL':>10}")
for label, grp in gd_trades.groupby("hold_bin", observed=True):
    wr  = grp["win"].mean() * 100
    avg = grp["PnL ($)"].mean()
    print(f"  {str(label):<12} {len(grp):>7,} {wr:>7.1f}% {avg:>+10.3f}")

# ── 2B: Failure by Confidence ────────────────────────────────────────────────
print(f"\n  Guardian MISS by entry confidence:")
conf_bins = [0, 0.62, 0.66, 0.70, 0.74, 0.78, 1.0]
conf_labs = ["<0.62", "0.62-0.66", "0.66-0.70", "0.70-0.74", "0.74-0.78", "0.78+"]
gd_trades["conf_bin"] = pd.cut(gd_trades["Conf"], bins=conf_bins, labels=conf_labs)
print(f"  {'Conf range':<14} {'N':>7} {'WR%':>8} {'AvgPnL':>10}")
for label, grp in gd_trades.groupby("conf_bin", observed=True):
    if len(grp) < 5:
        continue
    wr  = grp["win"].mean() * 100
    avg = grp["PnL ($)"].mean()
    print(f"  {str(label):<14} {len(grp):>7,} {wr:>7.1f}% {avg:>+10.3f}")

# ── 2C: Failure by Direction ─────────────────────────────────────────────────
print(f"\n  Guardian MISS by direction:")
for dir_, label in [("LONG", "LONG"), ("SHORT", "SHORT")]:
    sub = gd_trades[gd_trades["Direction"] == dir_]
    wr  = sub["win"].mean() * 100
    avg = sub["PnL ($)"].mean()
    print(f"  {label:<8} n={len(sub):,}  WR={wr:.1f}%  AvgPnL={avg:+.3f}")

# ── 2D: Failure by H4 Trend ──────────────────────────────────────────────────
if "H4 Trend" in gd_trades.columns:
    print(f"\n  Guardian MISS by H4 Trend:")
    for tr, grp in gd_trades.groupby("H4 Trend"):
        if len(grp) < 10:
            continue
        wr  = grp["win"].mean() * 100
        avg = grp["PnL ($)"].mean()
        print(f"  {str(tr):<12} n={len(grp):,}  WR={wr:.1f}%  AvgPnL={avg:+.3f}")

# ── 2E: Failure by Vol Regime ────────────────────────────────────────────────
if "Vol Regime" in gd_trades.columns and gd_trades["Vol Regime"].notna().sum() > 50:
    print(f"\n  Guardian MISS by Volatility Regime (ATR percentile):")
    vol_bins = [0, 25, 50, 75, 100]
    vol_labs = ["low(0-25)", "mid(25-50)", "high(50-75)", "vhigh(75+)"]
    gd_trades["vol_bin"] = pd.cut(gd_trades["Vol Regime"], bins=vol_bins, labels=vol_labs)
    for label, grp in gd_trades.groupby("vol_bin", observed=True):
        if len(grp) < 10:
            continue
        wr  = grp["win"].mean() * 100
        avg = grp["PnL ($)"].mean()
        print(f"  {str(label):<14} n={len(grp):,}  WR={wr:.1f}%  AvgPnL={avg:+.3f}")

# ── 3. WALK-FORWARD STABILITY ────────────────────────────────────────────────
banner("3. WALK-FORWARD STABILITY — Is 79.6% robust across time?")

df["Opened_dt"] = pd.to_datetime(df["Opened"], errors="coerce", utc=True)
df["month"]     = df["Opened_dt"].dt.to_period("M")

print(f"\n  Monthly Guardian WR (all trades):")
print(f"  {'Month':<10} {'Trades':>8} {'WR%':>8} {'AvgPnL':>10} {'GdExit%':>10}")
monthly_stats = []
for month, grp in df.groupby("month"):
    n      = len(grp)
    wr     = grp["win"].mean() * 100
    avg    = grp["PnL ($)"].mean()
    gd_pct = grp["gd_exit"].mean() * 100
    monthly_stats.append({"month": str(month), "n": n, "wr": round(wr, 1),
                           "avg_pnl": round(avg, 3), "gd_pct": round(gd_pct, 1)})
    print(f"  {str(month):<10} {n:>8,} {wr:>7.1f}% {avg:>+10.3f} {gd_pct:>9.1f}%")

# WR stability: std of monthly WR
monthly_wrs = [s["wr"] for s in monthly_stats]
wr_std = np.std(monthly_wrs)
wr_min = min(monthly_wrs)
wr_max = max(monthly_wrs)
print(f"\n  Monthly WR: mean={np.mean(monthly_wrs):.1f}%  std={wr_std:.1f}pp  "
      f"range=[{wr_min:.1f}%, {wr_max:.1f}%]")

if wr_std < 5:
    stability = "STABLE (std < 5pp)"
elif wr_std < 10:
    stability = "MODERATE (std 5-10pp)"
else:
    stability = "UNSTABLE (std > 10pp)"
print(f"  Stability: {stability}")

# ── Guardian-specific walk-forward: WR of Guardian exits only ────────────────
print(f"\n  Monthly Guardian EXIT WR only:")
for month, grp in df[df["gd_exit"]].groupby("month"):
    n   = len(grp)
    wr  = grp["win"].mean() * 100
    avg = grp["PnL ($)"].mean()
    print(f"  {str(month):<10} {n:>8,} {wr:>7.1f}%  avg={avg:+.3f}")

# ── 4. EDGE CONCENTRATION ─────────────────────────────────────────────────────
banner("4. EDGE CONCENTRATION — Fragile or Distributed?")

print(f"\n  Edge source analysis:")

# Per-coin Guardian WR
print(f"\n  Per-coin Guardian exit WR:")
print(f"  {'Coin':<18} {'N':>7} {'WR%':>8} {'AvgPnL':>10}")
coin_stats = []
for coin, grp in df[df["gd_exit"]].groupby("Coin"):
    n   = len(grp)
    wr  = grp["win"].mean() * 100
    avg = grp["PnL ($)"].mean()
    coin_stats.append({"coin": coin, "n": n, "wr": round(wr, 1)})
    flag = " <--" if wr < 65 or wr > 90 else ""
    print(f"  {coin:<18} {n:>7,} {wr:>7.1f}% {avg:>+10.3f}{flag}")

wr_spread = max(s["wr"] for s in coin_stats) - min(s["wr"] for s in coin_stats)
print(f"\n  Per-coin WR spread: {wr_spread:.1f}pp")
if wr_spread > 30:
    print(f"  [FINDING] High coin-level variance — Guardian edge is concentrated in specific coins")
elif wr_spread > 15:
    print(f"  [FINDING] Moderate coin-level variance — Guardian works better on some coins")
else:
    print(f"  [FINDING] Low coin-level variance — Guardian edge is distributed across coins")

# ── 5. SYNTHESIS ──────────────────────────────────────────────────────────────
banner("5. SYNTHESIS — Guardian: Robust or Fragile?")

top_feat = feat_imp[0][0]
top_pct  = feat_imp[0][1] / total_imp * 100
print(f"""
  Key findings:

  [1] Feature dominance: {top_feat} ({top_pct:.1f}% importance)
      Dynamic vs Static split: {dyn_pct:.1f}% dynamic / {sta_pct:.1f}% static
      {'Trade-state driven (robust)' if dyn_pct > 50 else 'Market-snapshot driven (fragile)'}

  [2] Exit WR stability: {stability}
      Monthly range: {wr_min:.1f}% - {wr_max:.1f}%

  [3] Edge concentration: per-coin spread {wr_spread:.1f}pp
      {'Concentrated (fragile)' if wr_spread > 30 else 'Distributed (robust)'}

  [4] EV per Guardian exit: {ev_gd:+.3f} USD
      This is the dollar amount Guardian adds per exit decision

  Robustness verdict:
""")

robustness_score = 0
if dyn_pct > 50: robustness_score += 1
if wr_std < 8: robustness_score += 1
if wr_spread < 25: robustness_score += 1
if top_pct < 40: robustness_score += 1

verdict = {4: "ROBUST — edge is distributed, stable, trade-state driven",
           3: "PROBABLY ROBUST — minor fragility in one dimension",
           2: "MIXED — some fragility, monitor closely",
           1: "FRAGILE — concentrated edge, likely market-regime dependent",
           0: "HIGH RISK — multiple fragility indicators"}

print(f"  Score: {robustness_score}/4")
print(f"  Verdict: {verdict.get(robustness_score, 'UNKNOWN')}")

audit["walk_forward"] = {"monthly_stats": monthly_stats, "wr_std": round(wr_std, 2),
                          "wr_min": round(wr_min, 1), "wr_max": round(wr_max, 1),
                          "stability": stability}
audit["edge_concentration"] = {"per_coin_wr_spread": round(wr_spread, 1),
                                 "coin_stats": coin_stats}
audit["ev_per_guardian_exit"] = round(ev_gd, 4)
audit["robustness_score"] = f"{robustness_score}/4"
audit["robustness_verdict"] = verdict.get(robustness_score, "UNKNOWN")

with open(OUT_PATH, "w") as f:
    json.dump(audit, f, indent=2, default=str)
print(f"\n  Saved: {OUT_PATH}")
