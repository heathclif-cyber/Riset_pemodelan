"""
Multistage Feature Selection untuk TB LSTM.
Target: Triple Barrier 3-class label (WIN/LOSS/NEUTRAL)

Stage 1: Standalone IC — |IC| >= 0.02, |t-stat| >= 2.0
Stage 2: Marginal IC (Gram-Schmidt) — buang redundant
Stage 3: IC Decay (temporal windows) — stabilitas
Stage 4: Verdict: KEEP / REDUNDANT / WEAK / UNSTABLE

Starting pool: FEATURE_COLS_V3 (~95 features)
Period: 2020-01-01 – 2025-11-01 (training only)
Output: models/runs/tb_lstm_v1/tb_lstm_v1_keep_features.json
"""
import json, sys, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from config import *

# ── Setup ─────────────────────────────────────────────────────────────────────
OUT_DIR = MODEL_DIR / "runs" / "tb_lstm_v1"
OUT_DIR.mkdir(parents=True, exist_ok=True)

LM = {"SHORT": 0, "FLAT": 1, "LONG": 2}
TP_MULT  = TP_SL_FALLBACK_TP       # 2.0
SL_MULT  = TP_SL_FALLBACK_SL       # 1.5
MAX_HOLD = MAX_HOLDING_BARS        # 36

# Starting pool — only features actually available in ALL training coins
def get_available_features():
    # Read schema only (fast) from first coin, then verify on others
    import pyarrow.parquet as pq
    first_coin = TRAINING_COINS[0]
    path = LABEL_DIR / f"{first_coin}_features_v3.parquet"
    common = set(pq.read_schema(path).names)
    for sym in TRAINING_COINS[1:]:
        path = LABEL_DIR / f"{sym}_features_v3.parquet"
        if not path.exists(): continue
        cols = set(pq.read_schema(path).names)
        common &= cols
    avail = [f for f in FEATURE_COLS_V3 if f in common]
    missing = [f for f in FEATURE_COLS_V3 if f not in common]
    return avail, missing

FEATS, MISSING_FEATS = get_available_features()
if MISSING_FEATS:
    print(f"  Missing from training data: {MISSING_FEATS}")

# IC thresholds
MIN_ABS_IC       = 0.02
MIN_TSTAT        = 2.0
MIN_MARGINAL_IC  = 0.01       # setelah dikontrol fitur lain
N_TEMPORAL_WIN   = 5          # jumlah temporal windows
MIN_CONSISTENCY  = 0.60       # 3/5 windows sign sama
MIN_IC_IR        = 0.50       # IC / std(IC) across windows

print(f"\n{'='*65}")
print(f"  MULTISTAGE FEATURE SELECTION — TB LSTM")
print(f"  Target: Triple Barrier (TP={TP_MULT}xATR, SL={SL_MULT}xATR, "
      f"max_hold={MAX_HOLD})")
print(f"  Starting pool: {len(FEATS)} features from FEATURE_COLS_V3")
print(f"  Training: 2020-01-01 – {TRAIN_CUTOFF_DATE}")
print(f"{'='*65}")


# ── Helper functions ──────────────────────────────────────────────────────────
def compute_tb_label(close, high, low, atr, i, n):
    """Triple Barrier label. 0=SHORT, 1=FLAT, 2=LONG"""
    entry   = close[i]
    tp_long  = entry + TP_MULT * atr[i]
    sl_long  = entry - SL_MULT * atr[i]
    tp_short = entry - TP_MULT * atr[i]
    sl_short = entry + SL_MULT * atr[i]
    end = min(i + MAX_HOLD, n)

    # LONG scenario
    label_long = 1
    for j in range(i + 1, end + 1):
        if high[j] >= tp_long:  label_long = 2; break
        if low[j]  <= sl_long:  label_long = 0; break

    # SHORT scenario
    label_short = 1
    for j in range(i + 1, end + 1):
        if low[j]  <= tp_short: label_short = 0; break
        if high[j] >= sl_short: label_short = 2; break

    if label_long != 1 and label_short != 1: return label_long
    elif label_long != 1: return label_long
    elif label_short != 1: return label_short
    else: return 1


def ordinal_label(raw_label):
    """Convert 0/1/2 to ordinal: SHORT=-1, FLAT=0, LONG=+1"""
    if raw_label == 0: return -1
    elif raw_label == 2: return 1
    else: return 0


def standalone_ic(X, y, effective_n=None):
    """Spearman IC + t-stat per feature."""
    n, f = X.shape
    results = []
    for i in range(f):
        col = X[:, i]
        valid = ~np.isnan(col) & ~np.isnan(y)
        if valid.sum() < 100:
            results.append({"idx": i, "ic": 0.0, "tstat": 0.0, "pval": 1.0})
            continue

        from scipy.stats import spearmanr
        rho, pval = spearmanr(col[valid], y[valid])
        n_eff = effective_n if effective_n else valid.sum()
        tstat = rho * np.sqrt((n_eff - 2) / max(1 - rho**2, 1e-9))
        results.append({"idx": i, "ic": float(rho), "tstat": float(tstat),
                        "pval": float(pval)})
    return results


def gram_schmidt_marginal(X, y, results):
    """
    Gram-Schmidt orthogonalization untuk marginal IC.
    Fitur di-ortogonalisasi relatif terhadap fitur yang sudah KEEP.
    """
    n, f = X.shape
    keep_mask = np.zeros(f, dtype=bool)
    # First pass: identify KEEP based on standalone
    for r in results:
        if abs(r["ic"]) >= MIN_ABS_IC and abs(r["tstat"]) >= MIN_TSTAT:
            keep_mask[r["idx"]] = True

    residual = np.zeros(n, dtype=np.float64)
    used_any = False
    for i in range(f):
        if not keep_mask[i]: continue
        col = X[:, i].copy()
        col = np.where(np.isnan(col), 0.0, col)
        col = (col - np.mean(col)) / max(np.std(col), 1e-9)
        if not used_any:
            residual = col
            used_any = True
        else:
            # Orthogonalize
            proj = np.sum(residual * col) / max(np.sum(residual**2), 1e-9) * residual
            col_residual = col - proj
            residual += col_residual / max(np.linalg.norm(col_residual), 1e-9)

    # Normalize residual
    residual = residual / max(np.std(residual), 1e-6)

    # Marginal IC: correlation of each non-keep feature with residual
    from scipy.stats import spearmanr
    all_marginal = []
    for i in range(f):
        if keep_mask[i]:
            all_marginal.append({"idx": i, "marginal_ic": 0.0, "verdict": "KEEP"})
            continue

        col = X[:, i].copy()
        col = np.where(np.isnan(col), 0.0, col)
        valid = ~np.isnan(residual)
        if valid.sum() < 100:
            all_marginal.append({"idx": i, "marginal_ic": 0.0, "verdict": "WEAK"})
            continue

        rho, _ = spearmanr(col[valid], residual[valid])
        all_marginal.append({"idx": i, "marginal_ic": float(rho),
                             "verdict": "KEEP" if abs(rho) >= MIN_MARGINAL_IC else "REDUNDANT"})
    return all_marginal


def ic_decay_test(X, y_ord, ts_idx, keep_indices, n_windows=5):
    """IC stability across temporal windows."""
    t_min = ts_idx.min()
    t_max = ts_idx.max()
    boundaries = np.linspace(0, len(X), n_windows + 1).astype(int)

    results = []
    for feat_i in keep_indices:
        col = X[:, feat_i]
        window_ics = []
        for w in range(n_windows):
            s = boundaries[w]
            e = boundaries[w + 1]
            valid = ~np.isnan(col[s:e]) & ~np.isnan(y_ord[s:e])
            if valid.sum() < 50:
                window_ics.append(0.0); continue
            from scipy.stats import spearmanr
            rho, _ = spearmanr(col[s:e][valid], y_ord[s:e][valid])
            window_ics.append(float(rho))

        ic_arr = np.array(window_ics)
        ic_mean = np.mean(ic_arr)
        ic_std  = np.std(ic_arr)
        ic_ir   = ic_mean / max(ic_std, 1e-9)
        sign_consistency = np.mean(np.sign(ic_arr) == np.sign(ic_mean))

        verdict = "STABLE" if (ic_ir >= MIN_IC_IR and \
                                sign_consistency >= MIN_CONSISTENCY) else "UNSTABLE"
        results.append({
            "idx": feat_i, "window_ics": [round(v, 5) for v in window_ics],
            "ic_mean": round(ic_mean, 5), "ic_std": round(ic_std, 5),
            "ic_ir": round(ic_ir, 3), "sign_cons": round(sign_consistency, 2),
            "verdict": verdict,
        })
    return results


# ── Load all training data ──────────────────────────────────────────────────────
print("\n[Loading training data...]")

all_X_parts, all_y_parts, all_ts_parts = [], [], []

for sym in TRAINING_COINS:
    path = LABEL_DIR / f"{sym}_features_v3.parquet"
    if not path.exists(): continue

    df = pd.read_parquet(path).sort_index()
    df = df[df.index < TRAIN_CUTOFF_DATE]

    if "label" not in df.columns: continue

    # Fill features
    avail = [c for c in FEATS if c in df.columns]
    df_feat = df[avail].ffill().fillna(0.0)

    close = df["close"].values.astype(np.float64)
    high  = df["high"].values.astype(np.float64)
    low   = df["low"].values.astype(np.float64)
    atr   = df["atr_14_h1"].values.astype(np.float64)
    n     = len(df)

    high = np.where(np.isnan(high), close, high)
    low  = np.where(np.isnan(low), close, low)

    X_c = df_feat.values.astype(np.float64)
    labels_c = np.full(n, 1, dtype=np.int32)

    for i in range(n - MAX_HOLD):
        labels_c[i] = compute_tb_label(close, high, low, atr, i, n)

    all_X_parts.append(X_c[:n - MAX_HOLD])
    all_y_parts.append(labels_c[:n - MAX_HOLD])
    all_ts_parts.append(df.index[:n - MAX_HOLD])

X_raw = np.concatenate(all_X_parts, axis=0)
y_raw = np.concatenate(all_y_parts, axis=0)
ts_all = pd.DatetimeIndex(np.concatenate([[t for t in ts] for ts in all_ts_parts]))

n_total, n_feat = X_raw.shape
dist = {l: int((y_raw == l).sum()) for l in [0, 1, 2]}
print(f"  Samples: {n_total:,} | Features: {n_feat}")
print(f"  Labels: SHORT={dist[0]:,} FLAT={dist[1]:,} LONG={dist[2]:,}")
print(f"  Ratio: {dist[0]/n_total*100:.0f}%/{dist[1]/n_total*100:.0f}%"
      f"/{dist[2]/n_total*100:.0f}%")

# ── Stage 1: Standalone IC ─────────────────────────────────────────────────────
print(f"\n[Stage 1] Standalone IC (|IC| >= {MIN_ABS_IC}, |t| >= {MIN_TSTAT})...")

from scipy.stats import spearmanr
y_ord = np.array([ordinal_label(v) for v in y_raw], dtype=np.float64)

s1_results = standalone_ic(X_raw, y_ord)

# Map feature names
for r in s1_results:
    r["feature"] = FEATS[r["idx"]] if r["idx"] < len(FEATS) else f"col_{r['idx']}"

keep_s1 = [r for r in s1_results if abs(r["ic"]) >= MIN_ABS_IC and abs(r["tstat"]) >= MIN_TSTAT]
weak_s1 = [r for r in s1_results if abs(r["ic"]) < MIN_ABS_IC]
noisy_s1 = [r for r in s1_results if abs(r["ic"]) >= MIN_ABS_IC and abs(r["tstat"]) < MIN_TSTAT]

print(f"  KEEP: {len(keep_s1)} | WEAK: {len(weak_s1)} | NOISY (IC ok but low t): {len(noisy_s1)}")
print(f"  Top 10 by |IC|:")
for r in sorted(keep_s1, key=lambda x: -abs(x["ic"]))[:10]:
    print(f"    {r['feature']:<35} IC={r['ic']:+.4f}   t={r['tstat']:+.1f}")

# ── Stage 2: Marginal IC (Gram-Schmidt) ────────────────────────────────────────
print(f"\n[Stage 2] Marginal IC (post Gram-Schmidt, |marginal_IC| >= {MIN_MARGINAL_IC})...")

s2_results = gram_schmidt_marginal(X_raw, y_ord, s1_results)
for r in s2_results:
    r["feature"] = FEATS[r["idx"]] if r["idx"] < len(FEATS) else f"col_{r['idx']}"
    # Merge standalone info
    s1 = next((x for x in s1_results if x["idx"] == r["idx"]), None)
    if s1:
        r["ic"] = s1["ic"]; r["tstat"] = s1["tstat"]

keep_s2  = [r for r in s2_results if r["verdict"] == "KEEP"]
redun_s2 = [r for r in s2_results if r["verdict"] == "REDUNDANT"]
weak_s2  = [r for r in s2_results if r["verdict"] == "WEAK"]

print(f"  KEEP: {len(keep_s2)} | REDUNDANT: {len(redun_s2)} | WEAK: {len(weak_s2)}")
if redun_s2:
    print(f"  Dropped (redundant): {[r['feature'] for r in redun_s2[:10]]}...")
if weak_s2:
    print(f"  Dropped (weak): {[r['feature'] for r in weak_s2[:10]]}...")

# ── Stage 3: IC Decay ──────────────────────────────────────────────────────────
print(f"\n[Stage 3] IC Decay ({N_TEMPORAL_WIN} temporal windows, "
      f"IR >= {MIN_IC_IR}, sign_cons >= {MIN_CONSISTENCY})...")

keep_indices = [r["idx"] for r in keep_s2]
s3_results = ic_decay_test(X_raw, y_ord, ts_all, keep_indices, N_TEMPORAL_WIN)
for r in s3_results:
    r["feature"] = FEATS[r["idx"]] if r["idx"] < len(FEATS) else f"col_{r['idx']}"
    # Merge
    s1 = next((x for x in s1_results if x["idx"] == r["idx"]), None)
    s2 = next((x for x in s2_results if x["idx"] == r["idx"]), None)
    r["ic"] = s1["ic"] if s1 else 0.0
    r["tstat"] = s1["tstat"] if s1 else 0.0
    r["marginal_ic"] = s2["marginal_ic"] if s2 else 0.0

stable   = [r for r in s3_results if r["verdict"] == "STABLE"]
unstable = [r for r in s3_results if r["verdict"] == "UNSTABLE"]

print(f"  STABLE: {len(stable)} | UNSTABLE: {len(unstable)}")
if unstable:
    print(f"  Unstable: {[r['feature'] for r in unstable]}")

# ── Stage 4: Final Verdict ─────────────────────────────────────────────────────
print(f"\n[Stage 4] Final Verdict...")

final_keep = [r for r in s3_results if r["verdict"] == "STABLE"]
final_drop = [r for r in s3_results if r["verdict"] != "STABLE"]
final_weak = [r for r in s1_results if r["idx"] not in set(k["idx"] for k in keep_s2)]

# Add features from FEATURE_COLS_V3 that weren't in training data
available_cols = set()
for sym in TRAINING_COINS[:1]:
    path = LABEL_DIR / f"{sym}_features_v3.parquet"
    if path.exists():
        df = pd.read_parquet(path)
        available_cols = set(df.columns)
        break

missing = [f for f in FEATS if f not in available_cols]
if missing:
    print(f"  Features not in parquet (dropped): {missing}")

print(f"\n{'='*65}")
print(f"  FINAL SELECTION SUMMARY")
print(f"{'='*65}")
print(f"  Starting pool    : {len(FEATS)}")
print(f"  Stage 1 (IC)     : {len(keep_s1)} KEEP / {len(weak_s1)} WEAK / "
      f"{len(noisy_s1)} NOISY")
print(f"  Stage 2 (MargIC) : {len(keep_s2)} KEEP / {len(redun_s2)} REDUNDANT / "
      f"{len(weak_s2)} WEAK")
print(f"  Stage 3 (Decay)  : {len(stable)} STABLE / {len(unstable)} UNSTABLE")
print(f"  ----------------------------------")
print(f"  FINAL KEEP       : {len(final_keep)} features")

final_names = [r["feature"] for r in final_keep]
print(f"\n  KEPT FEATURES ({len(final_names)}):")
for f in final_names:
    print(f"    {f}")

# Save
out = {
    "run": "tb_lstm_v1",
    "target": "Triple Barrier (TP=2xATR, SL=1.5xATR, max_hold=36)",
    "starting_pool": len(FEATS),
    "stage1_keep": len(keep_s1),
    "stage1_weak": len(weak_s1),
    "stage1_noisy": len(noisy_s1),
    "stage2_keep": len(keep_s2),
    "stage2_redundant": len(redun_s2),
    "stage2_weak": len(weak_s2),
    "stage3_stable": len(stable),
    "stage3_unstable": len(unstable),
    "final_keep": len(final_keep),
    "final_features": final_names,
    "dropped_unstable": [r["feature"] for r in unstable],
    "dropped_redundant": [r["feature"] for r in redun_s2],
    "dropped_weak_ic": [r["feature"] for r in weak_s1],
    "details": {
        "keep": [{**r, "feature": r["feature"]} for r in final_keep],
        "unstable": unstable,
        "redundant": redun_s2,
        "weak": weak_s1,
    },
    "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
}
out_path = OUT_DIR / "tb_lstm_v1_feature_selection.json"
with open(out_path, "w") as f:
    json.dump(out, f, indent=2, default=float)

# Also save just the feature list for easy loading
with open(OUT_DIR / "tb_lstm_v1_keep_features.json", "w") as f:
    json.dump(final_names, f, indent=2)

print(f"\n  Saved -> {out_path}")
print(f"  Feature list -> {OUT_DIR / 'tb_lstm_v1_keep_features.json'}")
print(f"{'='*65}")
