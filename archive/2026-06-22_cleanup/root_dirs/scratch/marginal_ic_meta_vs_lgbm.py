"""
Test Simon marginal IC: apakah binary meta bisa menambah informasi
DI ATAS LGBM confidence yang sudah ada?

Cara Simon: partial correlation / Gram-Schmidt residual.
Kalau marginal_IC(meta | LGBM_conf) < threshold -> meta tidak layak digunakan.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

OOF_PATH = ROOT / "data" / "meta_labels" / "ic32_oof_trades.parquet"
META_V1_DIR = ROOT / "models" / "runs" / "tb_lstm_binary_meta_v1"

df = pd.read_parquet(OOF_PATH)
df["conf"] = np.where(df["direction"] == 1, df["conf_long"], df["conf_short"])
y = df["win"].values.astype(float)
conf = df["conf"].values.astype(float)


def rank_ic(x, y_):
    mask = ~(np.isnan(x) | np.isnan(y_))
    x, y_ = x[mask], y_[mask]
    ic, _ = stats.spearmanr(x, y_)
    n = len(x)
    t = ic * np.sqrt((n - 2) / (1 - ic**2 + 1e-9))
    return float(ic), float(t), n


def marginal_ic_gram_schmidt(x_new, x_existing, y_):
    """
    Marginal IC of x_new given x_existing (Gram-Schmidt residual).
    Remove component of x_new explained by x_existing, then test IC vs y_.
    """
    mask = ~(np.isnan(x_new) | np.isnan(x_existing) | np.isnan(y_))
    xn, xe, y_m = x_new[mask], x_existing[mask], y_[mask]

    # Residual: x_new after projecting out x_existing
    corr = np.corrcoef(xn, xe)[0, 1]
    xn_resid = xn - corr * xe

    ic, _ = stats.spearmanr(xn_resid, y_m)
    n = mask.sum()
    t = ic * np.sqrt((n - 2) / (1 - ic**2 + 1e-9))
    return float(ic), float(t), float(corr), n


print("\n" + "="*65)
print("  Simon Marginal IC Analysis")
print("  Question: does binary meta add IC above LGBM confidence?")
print("="*65)

# Baseline: standalone IC of LGBM confidence
ic_conf, t_conf, n = rank_ic(conf, y)
print(f"\n  [1] Standalone IC(LGBM_conf, ic32_WIN)")
print(f"      IC={ic_conf:+.4f}  t={t_conf:.1f}  n={n:,}")

# Simulate: what marginal IC would binary meta need to be "worth it"?
# Using formula: marginal_IC = standalone_IC(meta) - corr(meta, conf) × IC(conf)
print(f"\n  [2] Simulated marginal IC at different overlap levels")
print(f"      (IC_meta=0.068 standalone, IC_conf={ic_conf:.4f})")
print(f"\n      {'corr(meta,conf)':>18} {'marginal_IC':>14} {'Pass(>0.03)?':>14}")
for corr_assumed in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
    marg = 0.068 - corr_assumed * ic_conf
    passes = "YES" if abs(marg) >= 0.03 else "NO"
    print(f"      {corr_assumed:>18.1f} {marg:>+14.4f} {passes:>14}")

# Empirical: test if any ic32 OOF features have marginal IC above conf
# (These are proxies for what binary meta might capture)
ic32_feat_cols = [c for c in df.columns
                  if c not in ("timestamp", "coin", "fold", "direction", "win",
                               "net_pnl", "conf_long", "conf_short", "hold_bars", "conf")]

print(f"\n  [3] Empirical marginal IC of ic32 features GIVEN LGBM conf")
print(f"      (These are the 33 features; binary meta uses a subset)")
print(f"\n      {'Feature':<30} {'Standalone IC':>14} {'corr w/conf':>12} {'Marginal IC':>12} {'t':>8}")

results = []
for feat in ic32_feat_cols:
    x = df[feat].values.astype(float)
    ic_sa, t_sa, _ = rank_ic(x, y)
    ic_marg, t_marg, corr_xc, _ = marginal_ic_gram_schmidt(x, conf, y)
    results.append((feat, ic_sa, ic_marg, t_marg, corr_xc))

# Sort by marginal IC
results.sort(key=lambda r: abs(r[2]), reverse=True)
for feat, ic_sa, ic_marg, t_marg, corr_xc in results[:15]:
    flag = "<-- marginal PASS" if abs(ic_marg) >= 0.03 and abs(t_marg) >= 2.0 else ""
    print(f"      {feat:<30} {ic_sa:>+13.4f}  {corr_xc:>11.3f}  {ic_marg:>+11.4f}  {t_marg:>+7.2f}  {flag}")

# Count how many pass marginal gate
pass_marg = [(f, m, t) for f, _, m, t, _ in results if abs(m) >= 0.03 and abs(t) >= 2.0]
print(f"\n      Features passing marginal IC gate: {len(pass_marg)}/{len(results)}")
if pass_marg:
    print(f"      Passing: {[f for f, _, _ in pass_marg]}")
else:
    print(f"      NONE — LGBM confidence already captures all predictive information")
    print(f"      in the ic32 feature set. Binary meta adds zero marginal value.")

print(f"\n{'='*65}")
print(f"  CONCLUSION")
print(f"{'='*65}")
print(f"  If corr(binary_meta_proba, LGBM_conf) >= 0.4:")
print(f"    marginal_IC < 0.03 → binary meta v2 tidak akan lolos gate")
print(f"  The structural fix needed is NOT better labels —")
print(f"  it is genuinely orthogonal features (not a subset of ic32).")
print(f"{'='*65}\n")
