"""Leakage + overfitting audit for active stack (widyawardhana v2 + continuation_v1)."""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from config import TRAIN_CUTOFF_DATE, LABEL_DIR, HOLDOUT_DIR, MODEL_DIR, PURGE_GAP_BARS, GUARDIAN_PURGE_GAP_BARS

COINS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "ADAUSDT", "DOGEUSDT"]
issues = []
warnings = []
passes = []


def ok(msg):
    passes.append(msg)


def warn(msg):
    warnings.append(msg)


def fail(msg):
    issues.append(msg)


print("=" * 70)
print("  LEAKAGE & OVERFITTING AUDIT — tb_widyawardhana_v2_continuation")
print("=" * 70)
print(f"TRAIN_CUTOFF_DATE = {TRAIN_CUTOFF_DATE}")
print()

# ── 1. Train vs holdout temporal separation ──────────────────────────────
train_max = {}
train_min = {}
hold_min = {}
hold_max = {}
overlap_bars = 0

for sym in COINS:
    tp = LABEL_DIR / f"{sym}_features_v3.parquet"
    hp = HOLDOUT_DIR / "labeled" / f"{sym}_features_v3.parquet"
    if not tp.exists() or not hp.exists():
        continue
    tdf = pd.read_parquet(tp)
    hdf = pd.read_parquet(hp)
    tdf.index = pd.to_datetime(tdf.index, utc=True)
    hdf.index = pd.to_datetime(hdf.index, utc=True)

    train_min[sym] = tdf.index.min()
    train_max[sym] = tdf.index.max()
    hold_min[sym] = hdf.index.min()
    hold_max[sym] = hdf.index.max()

    tset = set(tdf.index)
    hset = set(hdf.index)
    ov = len(tset & hset)
    overlap_bars += ov
    if ov > 0:
        fail(f"{sym}: {ov} overlapping timestamps between LABEL_DIR and HOLDOUT_DIR")
    if train_max[sym] >= TRAIN_CUTOFF_DATE:
        fail(f"{sym}: training data extends to {train_max[sym]} >= cutoff {TRAIN_CUTOFF_DATE}")
    if hold_min[sym] < TRAIN_CUTOFF_DATE:
        fail(f"{sym}: holdout starts {hold_min[sym]} < cutoff {TRAIN_CUTOFF_DATE}")
    else:
        ok(f"{sym}: train [{train_min[sym].date()}..{train_max[sym].date()}] | holdout [{hold_min[sym].date()}..{hold_max[sym].date()}]")

if overlap_bars == 0:
    ok(f"Zero timestamp overlap across {len(COINS)} sample coins")

# ── 2. Model training uses cutoff filter ─────────────────────────────────
for script, must_have in [
    ("pipeline/06p_train_guardian_continuation_v1.py", "df.index < TRAIN_CUTOFF_DATE"),
    ("pipeline/06l_train_guardian_profit_v1.py", "df.index < TRAIN_CUTOFF_DATE"),
    ("pipeline/04_train_lgbm_tb_fs28.py", "TRAIN_CUTOFF_DATE"),
]:
    p = ROOT / script
    if p.exists():
        txt = p.read_text(encoding="utf-8", errors="ignore")
        if must_have in txt:
            ok(f"{script}: cutoff filter present")
        else:
            warn(f"{script}: cutoff filter not found in source")

# flatboost v2 trainer
fb_scripts = list((ROOT / "pipeline").glob("*flatboost*"))
if not fb_scripts:
    fb_scripts = [ROOT / "pipeline/04_train_lgbm_tb_fs28.py"]
for p in fb_scripts[:3]:
    if p.exists():
        txt = p.read_text(encoding="utf-8", errors="ignore")
        if "TRAIN_CUTOFF" in txt or "train_cutoff" in txt.lower():
            ok(f"{p.name}: uses TRAIN_CUTOFF")
        else:
            warn(f"{p.name}: TRAIN_CUTOFF not explicit")

# ── 3. Guardian label lookahead (training-only, not holdout feature leak) ─
warn(
    "Guardian labels use best_future_pnl look-ahead (Rules 4,7 in 06p) — "
    "VALID for supervised training labels, but inflates CV F1 vs live; not feature leakage at inference"
)

# ── 4. Guardian circular entry dependency ────────────────────────────────
warn(
    "Guardian training samples generated from flatboost_v2 predictions ON SAME training period — "
    "entry model sees in-sample trades; mild circularity; holdout uses frozen models (OK for OOS eval)"
)

# ── 5. CV vs holdout gap (overfitting signal) ────────────────────────────
gmeta = json.load(open(MODEL_DIR / "runs/tb_guardian_continuation_v1/tb_guardian_continuation_v1_meta.json"))
f1s = [r["f1_macro"] for r in gmeta["cv_results"]]
cv_f1 = float(np.mean(f1s))
hold = json.load(open(MODEL_DIR / "runs/tb_guardian_continuation_v1/holdout_guardian_compare.json"))
cont = [v for v in hold["variants"] if "continuation" in v["name"]][0]
fb = json.load(open(MODEL_DIR / "runs/tb_lgbm_flatboost_v2/tb_lgbm_flatboost_v2_meta.json"))

print()
print("--- Overfitting signals ---")
print(f"LGBM flatboost_v2 CV F1 macro : {fb.get('cv_mean_f1_macro', '?'):.4f}")
print(f"Guardian continuation CV F1   : {cv_f1:.4f}")
print(f"Holdout WR (stack)            : {cont['win_rate']}%")
print(f"Holdout PF                    : {cont['profit_factor']}")

if cv_f1 > 0.80:
    warn(f"Guardian CV F1={cv_f1:.3f} very high — labels use future PnL; expect optimistic training metric")
if fb.get("cv_mean_f1_macro", 0) < 0.45 and cont["win_rate"] > 65:
    warn(
        f"LGBM CV F1={fb['cv_mean_f1_macro']:.3f} low but holdout WR={cont['win_rate']}% — "
        "possible threshold/HMM tuning on holdout lifts live-like metrics"
    )

# ── 6. Holdout tuning bias ───────────────────────────────────────────────
sweep_files = [
    MODEL_DIR / "runs/tb_lgbm_flatboost_v2/threshold_sweep.json",
    MODEL_DIR / "runs/tb_lgbm_flatboost_v2/hmm_adaptive_sweep.json",
]
for sf in sweep_files:
    if sf.exists():
        fail(f"Test-set selection bias: {sf.name} tuned on holdout Apr-Jun 2026")
    else:
        ok(f"No {sf.name} in run dir (or not checked)")

# Check active config T50_R55 from inference
ic = json.load(open(MODEL_DIR / "inference_config.json"))
if ic.get("hmm", {}).get("trending_threshold") == 0.50:
    fail(
        "HMM T50_R55 + cascade thresholds likely selected via holdout sweeps "
        "(EXPERIMENTS.md 2026-06-13: 288 combo sweep on Apr-Jun holdout)"
    )

# ── 7. Fixed-model holdout (in-sample leakage if used) ───────────────────
holdout_script = (ROOT / "pipeline/07_holdout_guardian_continuation_compare.py").read_text()
if "joblib.load" in holdout_script and "full_trading_report" in holdout_script:
    if "purged" not in holdout_script.lower() and "retrain" not in holdout_script.lower():
        warn(
            "Holdout backtest uses FIXED pretrained models (no per-fold retrain) — "
            "CORRECT for deployment eval, but NOT purged CV OOF; scorecard is single-period OOS"
        )

# ── 8. Production feature gaps ───────────────────────────────────────────
if "etf" not in " ".join(json.load(open(MODEL_DIR / "guardian_feature_cols.json"))):
    ok("Guardian continuation_v1: no ETF features (avoids production zero-fill leak)")
else:
    warn("Guardian still has ETF features — zero-filled in production")

# ── 9. Purge gap config ──────────────────────────────────────────────────
print()
print(f"PURGE_GAP_BARS (LGBM)     = {PURGE_GAP_BARS}")
print(f"GUARDIAN_PURGE_GAP_BARS   = {GUARDIAN_PURGE_GAP_BARS}")

# ── Summary ──────────────────────────────────────────────────────────────
print()
print("=" * 70)
print(f"PASS  ({len(passes)})")
for p in passes[:12]:
    print(f"  [OK]   {p}")
if len(passes) > 12:
    print(f"  ... +{len(passes)-12} more")

print(f"\nWARN  ({len(warnings)})")
for w in warnings:
    print(f"  [WARN] {w}")

print(f"\nFAIL  ({len(issues)})")
for f in issues:
    print(f"  [FAIL] {f}")

verdict = "CONDITIONAL PASS"
if issues:
    if any("overlap" in i or "extends to" in i for i in issues):
        verdict = "FAIL — temporal leakage"
    else:
        verdict = "CONDITIONAL PASS — methodology bias, not raw data leak"
elif len(warnings) >= 4:
    verdict = "CONDITIONAL PASS — monitor live degradation"

print()
print(f"VERDICT: {verdict}")
print("=" * 70)