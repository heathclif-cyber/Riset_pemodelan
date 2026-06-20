"""
scratch/step3_marginal_ic_test.py
Marginal IC Test — Simon's Fundamental Law.

Pertanyaan: apakah temporal trajectory features punya informasi yang ORTHOGONAL
terhadap LGBM tb_lgbm_v3 confidence?

Jika marginal_IC(traj | lgbm_conf) >= 0.02 DAN t-stat >= 2.0 pada setidaknya
2 features dari kategori berbeda → design LSTM architecture.
Jika tidak → LSTM tidak justified, stop di LGBM+Guardian.

Methodology:
  1. IC standalone   : spearman(traj_feat, tb_win_outcome)
  2. Correlation     : corr(traj_feat, lgbm_conf)
  3. Marginal IC     : IC(traj) - corr(traj, conf) * IC(conf)
     (Gram-Schmidt orthogonalization)
  4. t-stat dengan autocorrelation correction (effective N)
  5. IC temporal stability — 6 temporal windows

Target: tb_win = 1 jika pred_label == true_label AND pred != FLAT
"""
import sys, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

FEAT_PATH = ROOT / "data" / "meta_labels" / "tb_v3_temporal_feats.parquet"
OOF_PATH  = ROOT / "data" / "meta_labels" / "tb_lgbm_v3_oof.parquet"

IC_THRESHOLD       = 0.02
TSTAT_THRESHOLD    = 2.0
MARGINAL_THRESHOLD = 0.015   # lebih longgar untuk marginal
WINDOWS_STABILITY  = 6
SIGN_CONS_MIN      = 4


def banner(t, w=68):
    print(f"\n{'='*w}\n  {t}\n{'='*w}")


def ic_with_tstat(x: np.ndarray, y: np.ndarray, lag: int = 24) -> tuple:
    """Spearman IC + autocorrelation-corrected t-stat."""
    mask = ~(np.isnan(x) | np.isnan(y))
    x, y = x[mask], y[mask]
    n = len(x)
    if n < 100:
        return np.nan, np.nan, n
    ic, _ = spearmanr(x, y)
    if np.isnan(ic):
        return np.nan, np.nan, n
    # Effective N via lag-1 autocorrelation of IC time-series proxy
    ic_ts = pd.Series(x * y).rolling(lag).mean().dropna()
    eff_n = float(n)
    if len(ic_ts) > lag:
        rho = ic_ts.autocorr(1)
        if not np.isnan(rho) and abs(rho) < 1.0:
            eff_n = max(n * (1 - rho) / (1 + rho + 1e-8), 30.0)
    se = np.sqrt((1 - ic**2) / max(eff_n - 2, 1))
    t  = ic / (se + 1e-10)
    return ic, t, int(eff_n)


banner("Step 3: Marginal IC Test (Simon's Fundamental Law)")

# Load data
print("\nLoading trajectory features + OOF...")
df = pd.read_parquet(FEAT_PATH)
df.index = pd.to_datetime(df.index, utc=True)
print(f"  Rows: {len(df):,} | Columns: {len(df.columns)}")

# Target: tb_win (binary outcome)
target = df["win"].values.astype(float)

# LGBM confidence = conf_entry (direction-aware)
lgbm_conf = df["conf_entry"].values.astype(float)

# IC of LGBM conf alone (baseline)
ic_conf, t_conf, n_conf = ic_with_tstat(lgbm_conf, target)
print(f"\n  LGBM conf baseline IC : {ic_conf:+.4f} | t={t_conf:.2f} | eff_N={n_conf:,}")

# Trajectory feature columns
traj_cols = [c for c in df.columns
             if any(c.endswith(suf) for suf in
                    ["_delta4","_delta8","_delta16","_delta32",
                     "_slope4","_slope8","_slope16","_slope32",
                     "_z4","_z8","_z16","_z32","_accel"])]
print(f"  Trajectory features   : {len(traj_cols)}")

# ── IC test semua trajectory features ─────────────────────────────────────────
banner("Standalone IC + Marginal IC per trajectory feature")

results = {}
for col in traj_cols:
    x = df[col].values.astype(float)
    if np.isnan(x).mean() > 0.5:
        continue

    ic_s, t_s, eff_n = ic_with_tstat(x, target)
    if np.isnan(ic_s):
        continue

    # Correlation dengan LGBM conf (for marginal IC calculation)
    mask = ~(np.isnan(x) | np.isnan(lgbm_conf))
    corr_with_conf = spearmanr(x[mask], lgbm_conf[mask])[0] if mask.sum() > 100 else 0.0

    # Gram-Schmidt marginal IC
    marginal_ic = ic_s - corr_with_conf * ic_conf

    results[col] = {
        "ic_standalone": ic_s,
        "t_standalone":  t_s,
        "corr_conf":     corr_with_conf,
        "marginal_ic":   marginal_ic,
        "eff_n":         eff_n,
    }

# Sort by marginal IC absolute value
sorted_results = sorted(results.items(), key=lambda x: abs(x[1]["marginal_ic"]), reverse=True)

print(f"\n  {'Feature':<38} {'IC_s':>7} {'t':>6} {'corr_c':>7} {'marg_IC':>8} {'verdict'}")
print(f"  {'-'*38} {'-'*7} {'-'*6} {'-'*7} {'-'*8} {'-'*10}")

keep_features  = []
for col, r in sorted_results:
    verdict = "KEEP" if (abs(r["marginal_ic"]) >= MARGINAL_THRESHOLD
                         and abs(r["t_standalone"]) >= TSTAT_THRESHOLD) else "."
    if verdict == "KEEP":
        keep_features.append(col)
        # Print all KEEP features
        print(f"  {col:<38} {r['ic_standalone']:>+7.4f} {r['t_standalone']:>6.1f} "
              f"{r['corr_conf']:>+7.3f} {r['marginal_ic']:>+8.4f} KEEP <--")

# Print top-20 WEAK by marginal IC (for reference)
weak_results = [(c, r) for c, r in sorted_results if c not in keep_features]
print(f"\n  --- Top 20 weak (tidak memenuhi threshold) ---")
for col, r in weak_results[:20]:
    print(f"  {col:<38} {r['ic_standalone']:>+7.4f} {r['t_standalone']:>6.1f} "
          f"{r['corr_conf']:>+7.3f} {r['marginal_ic']:>+8.4f}")

# ── Top results ringkasan ──────────────────────────────────────────────────────
banner("Top 10 Features by |Marginal IC|")
for col, r in sorted_results[:10]:
    verdict = "KEEP" if (abs(r["marginal_ic"]) >= MARGINAL_THRESHOLD
                         and abs(r["t_standalone"]) >= TSTAT_THRESHOLD) else "WEAK"
    print(f"  {col:<38} marg_IC={r['marginal_ic']:>+7.4f}  {verdict}")

# ── Temporal stability untuk KEEP features ────────────────────────────────────
banner("IC Temporal Stability (6 windows)")
df_sorted = df.sort_index()
total_ts  = len(df_sorted)
win_size  = total_ts // WINDOWS_STABILITY

stable_keeps = []
if keep_features:
    for col in keep_features:  # test semua KEEP features
        x_all = df_sorted[col].values.astype(float)
        y_all = df_sorted["win"].values.astype(float)
        ic_windows = []
        for i in range(WINDOWS_STABILITY):
            sl = slice(i * win_size, (i+1) * win_size)
            xw, yw = x_all[sl], y_all[sl]
            mask = ~(np.isnan(xw) | np.isnan(yw))
            if mask.sum() < 200:
                continue
            ic_w, _ = spearmanr(xw[mask], yw[mask])
            ic_windows.append(ic_w)

        if not ic_windows:
            continue
        sign_cons = sum(1 for ic in ic_windows if ic * results[col]["ic_standalone"] > 0)
        ic_ir     = np.mean(ic_windows) / (np.std(ic_windows) + 1e-8)
        stable    = (abs(ic_ir) >= 0.5) and (sign_cons >= SIGN_CONS_MIN)
        label     = "STABLE" if stable else "UNSTABLE"
        if stable:
            stable_keeps.append(col)
        per_w = "  ".join(f"{ic:+.3f}" for ic in ic_windows)
        print(f"  {col:<38} IC_IR={ic_ir:+.2f} sign={sign_cons}/{len(ic_windows)} {label}")
        print(f"    windows: {per_w}")
else:
    print("  Tidak ada KEEP features untuk stability test")

# ── KEPUTUSAN FINAL ───────────────────────────────────────────────────────────
banner("KEPUTUSAN: Apakah LSTM Layak Dibangun?")

n_keep         = len(keep_features)
n_stable       = len(stable_keeps)
# Deduplicate by source feature (strip the _delta4 / _slope8 / etc suffix)
unique_sources = set()
all_source_feats = set(results.keys())   # all trajectory feat names we tested
for col in stable_keeps:
    # Source = longest matching prefix that is itself a known base feature
    # Convention: feat_name_{suffix}{window} — source is everything before last '_'
    # e.g. atr_percent_h4_z32 -> atr_percent_h4
    parts = col.rsplit("_", 1)  # split on last underscore
    if len(parts) == 2:
        candidate = parts[0]
        # Strip window digits from candidate if present at end
        candidate2 = candidate.rstrip("0123456789").rstrip("_")
        unique_sources.add(candidate2 if candidate2 else candidate)
    else:
        unique_sources.add(col)

print(f"""
  LGBM conf baseline IC : {ic_conf:+.4f}  (t={t_conf:.2f})
  KEEP features (marg_IC >= {MARGINAL_THRESHOLD}, t >= {TSTAT_THRESHOLD}): {n_keep}
  STABLE KEEP features  : {n_stable}
  Unique signal sources : {len(unique_sources)} ({', '.join(sorted(unique_sources)) or 'none'})

  Threshold deploy LSTM : stable_keeps >= 2 AND unique_sources >= 2
""")

if n_stable >= 2 and len(unique_sources) >= 2:
    print("  [YES] LSTM JUSTIFIED")
    print(f"  Rekomendasi: bangun LSTM dengan trajectory dari: {', '.join(sorted(unique_sources))}")
    print(f"  LSTM input: {min(len(stable_keeps), 10)} trajectory features + seq_len=32")
    print(f"  Target    : tb_win (binary: 1=entry wins, 0=entry loses/flat)")
    print(f"\n  Stable KEEP features untuk LSTM input:")
    for col in stable_keeps[:10]:
        r = results[col]
        print(f"    {col:<38} marg_IC={r['marginal_ic']:>+7.4f}")
else:
    print("  [NO] LSTM TIDAK JUSTIFIED dengan data ini")
    print(f"  stable_keeps={n_stable} | unique_sources={len(unique_sources)}")
    print(f"  Perlu >= 2 stable KEEP dari >= 2 sumber berbeda")
    print(f"\n  Opsi lain:")
    print(f"  1. Tambah positioning data (OI delta, Taker ratio) — 6 bulan lagi")
    print(f"  2. Perbesar seq_len (64 bar) atau feature window (W=48/64)")
    print(f"  3. Tetap LGBM+Guardian tanpa LSTM — sudah PF 3.23, WR 55.5%")
