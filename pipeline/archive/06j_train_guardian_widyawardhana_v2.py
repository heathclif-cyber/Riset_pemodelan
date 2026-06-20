"""
pipeline/06j_train_guardian_widyawardhana_v2.py
Guardian untuk tb_widyawardhana_v3 — arsitektur clean_v2 + feature selection.

Perbaikan dari v1:
  1. Trade pool: Triple Barrier simulation (TP=2×ATR, SL=1.5×ATR, max_hold=36)
     → best_future_pnl bounded → label EXIT di zona profit terbentuk
  2. Feature selection: buang fitur importance < 1% total
  3. Label: HOLD=0, PARTIAL=1, FULL_EXIT=2 (identik dengan clean_v2)

Training : 2020-01-01 – 2025-11-01, purged CV 8 fold
Output   : models/runs/tb_guardian_widyawardhana_v2/
"""
import json, sys, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import joblib
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score

from core.utils import setup_logger, ensure_utc_index
from pipeline.shared import build_purged_folds
from config import *

logger = setup_logger("06j_guardian_wdyw_v2")

RUN_NAME  = "tb_guardian_widyawardhana_v2"
OUT_DIR   = MODEL_DIR / "runs" / RUN_NAME
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Barrier params (sama dengan training tb_widyawardhana_v3)
TP_MULT  = TP_SL_FALLBACK_TP       # 2.0
SL_MULT  = TP_SL_FALLBACK_SL       # 1.5
MAX_HOLD = MAX_HOLDING_BARS        # 36
REGIME_THRESH = {0: 0.45, 1: 0.50, 2: 0.50, 3: 0.45}
LM = {"SHORT": 0, "FLAT": 1, "LONG": 2}

DYNAMIC_FEATS = [
    "bars_held_norm", "current_pnl_pct", "current_pnl_atr",
    "max_favorable_pnl_pct", "drawdown_from_peak_pct",
    "direction", "entry_price_ratio",
]

# Feature selection: minimum importance fraction
MIN_IMP_FRAC = 0.01   # drop features with <1% total importance

# ── Load entry model ───────────────────────────────────────────────────────────
tb_model = joblib.load(MODEL_DIR / "runs" / "tb_lgbm_widyawardhana_v3" / "lgbm.pkl")
with open(MODEL_DIR / "runs" / "tb_lgbm_widyawardhana_v3" / "tb_lgbm_widyawardhana_v3_features.json") as f:
    TB_FEATS = json.load(f)

logger.info(f"Entry model: tb_lgbm_widyawardhana_v3 ({len(TB_FEATS)} features)")
logger.info(f"Barriers: TP={TP_MULT}xATR  SL={SL_MULT}xATR  max_hold={MAX_HOLD}")


def generate_samples_coin(sym):
    """Generate in-trade labeled samples using Triple Barrier trade pool."""
    path = LABEL_DIR / f"{sym}_features_v3.parquet"
    if not path.exists():
        return []

    df = pd.read_parquet(path)
    df = ensure_utc_index(df).sort_index()
    df = df[df.index < TRAIN_CUTOFF_DATE]

    rp  = LABEL_DIR / f"{sym}_regime_h1.parquet"
    hmm = np.full(len(df), 1, np.int32)
    if rp.exists():
        reg = pd.read_parquet(rp)
        if "hmm_regime_enc" in reg.columns:
            hmm = reg["hmm_regime_enc"].reindex(df.index, fill_value=1).values.astype(np.int32)

    mask = df["label"].isin(LM)
    df   = df[mask].copy()
    hmm  = hmm[mask.values]
    if len(df) < 200:
        return []

    n     = len(df)
    close = df["close"].values.astype(np.float64)
    high  = df["high"].values.astype(np.float64)
    low   = df["low"].values.astype(np.float64)
    atr   = df["atr_14_h1"].values.astype(np.float64)

    # Entry model predictions
    X = np.zeros((n, len(TB_FEATS)), dtype=np.float64)
    for idx, c in enumerate(TB_FEATS):
        if c in df.columns:
            X[:, idx] = df[c].ffill().fillna(0).values.astype(np.float64)

    proba = tb_model.predict_proba(X)
    conf  = np.max(proba, axis=1)
    yp    = np.argmax(proba, axis=1).astype(np.int32)
    for r, th in REGIME_THRESH.items():
        yp[(hmm == r) & (yp != 1) & (conf < th)] = 1

    # Triple Barrier simulation: generate trade pool
    trades = []
    i = 0
    while i < n:
        if yp[i] == 1:
            i += 1; continue

        direction = 1 if yp[i] == 2 else -1
        entry     = close[i]
        tp_price  = entry + direction * TP_MULT * atr[i]
        sl_price  = entry - direction * SL_MULT * atr[i]
        bar_out   = min(i + MAX_HOLD, n - 1)
        outcome   = "TIMEOUT"

        for j in range(i + 1, min(i + MAX_HOLD + 1, n)):
            if direction == 1:
                if high[j] >= tp_price:
                    bar_out, outcome = j, "TP"; break
                if low[j]  <= sl_price:
                    bar_out, outcome = j, "SL"; break
            else:
                if low[j]  <= tp_price:
                    bar_out, outcome = j, "TP"; break
                if high[j] >= sl_price:
                    bar_out, outcome = j, "SL"; break

        if bar_out > i + GUARDIAN_MIN_HOLD_BARS:
            trades.append({
                "bar_in": i, "bar_out": bar_out, "direction": direction,
                "entry": entry, "atr_entry": atr[i], "outcome": outcome,
            })
        i = bar_out + 1

    if not trades:
        return []

    # Build labeled samples from in-trade bars
    samples = []
    for t in trades:
        bar_in    = t["bar_in"]
        bar_out   = t["bar_out"]
        dir_sign  = t["direction"]
        entry     = t["entry"]
        atr_entry = t["atr_entry"]
        atr_pct   = atr_entry / entry if entry > 0 else 0.01
        mfe_sofar = 0.0
        dmult     = 1.0 if dir_sign == 1 else -1.0

        for j in range(bar_in + 1, bar_out):
            bars_held    = j - bar_in
            current_pnl  = (close[j] - entry) / entry * dir_sign
            mfe_sofar    = max(mfe_sofar, current_pnl)

            # Best future PnL within this trade (bounded by Triple Barrier)
            best_future = 0.0
            for k in range(j + 1, bar_out + 1):
                best_future = max(best_future, (close[k] - entry) / entry * dir_sign)

            # ── Labeling (identik clean_v2) ──────────────────────────────
            upside_ratio = (best_future - current_pnl) / best_future \
                           if best_future > 0.001 else 0.0

            if bars_held < GUARDIAN_MIN_HOLD_BARS:
                label = 0
            elif current_pnl < -1.0 * atr_pct:
                label = 2
            elif mfe_sofar > 0.015 and current_pnl < mfe_sofar * 0.25:
                label = 2
            elif current_pnl >= best_future * 0.95:
                label = 2
            elif mfe_sofar > 0.015 and current_pnl < mfe_sofar * 0.55:
                label = 1
            elif current_pnl > 0.008 and upside_ratio < 0.03:
                label = 1
            elif best_future > current_pnl * 1.05:
                label = 0
            else:
                continue

            # Dynamic features
            bars_held_norm    = bars_held / MAX_HOLD
            current_pnl_atr   = current_pnl / atr_pct if atr_pct > 0 else 0.0
            dd_from_peak      = (mfe_sofar - current_pnl) / mfe_sofar \
                                if mfe_sofar > 0.001 else 0.0
            entry_price_ratio = entry / close[j] if close[j] > 0 else 1.0

            row = {c: float(X[j, idx]) for idx, c in enumerate(TB_FEATS)}
            row.update({
                "bars_held_norm"       : bars_held_norm,
                "current_pnl_pct"      : current_pnl,
                "current_pnl_atr"      : current_pnl_atr,
                "max_favorable_pnl_pct": mfe_sofar,
                "drawdown_from_peak_pct": dd_from_peak,
                "direction"            : float(dir_sign),
                "entry_price_ratio"    : entry_price_ratio,
                "timestamp"            : df.index[j],
                "label"                : label,
            })
            samples.append(row)

    return samples


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 1: Generate samples
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*65}")
print(f"  Guardian v2: tb_widyawardhana_v3 — Triple Barrier pool + FS")
print(f"  Barriers: TP={TP_MULT}xATR  SL={SL_MULT}xATR  max_hold={MAX_HOLD}")
print(f"  Features: {len(TB_FEATS)} static + {len(DYNAMIC_FEATS)} dynamic")
print(f"  Period: 2020-01-01 – {TRAIN_CUTOFF_DATE} | {len(TRAINING_COINS)} coins")
print(f"{'='*65}\n")

print("[1/4] Generating in-trade samples (Triple Barrier pool)...")
all_samples = []
for sym in TRAINING_COINS:
    samples = generate_samples_coin(sym)
    if samples:
        all_samples.extend(samples)
        logger.info(f"  {sym}: {len(samples)} samples")

print(f"  Total: {len(all_samples):,} samples")

if len(all_samples) < 1000:
    print("ERROR: Not enough samples"); sys.exit(1)

df_s = pd.DataFrame(all_samples)
df_s["timestamp"] = pd.to_datetime(df_s["timestamp"])
df_s = df_s.set_index("timestamp").sort_index()

n0 = int((df_s["label"] == 0).sum())
n1 = int((df_s["label"] == 1).sum())
n2 = int((df_s["label"] == 2).sum())
print(f"  Labels: HOLD={n0:,}  PARTIAL={n1:,}  FULL_EXIT={n2:,}")

# ═══════════════════════════════════════════════════════════════════════════════
# Phase 2: Initial training (all features) + feature selection
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[2/4] Initial training + feature importance analysis...")

all_feat_cols = [c for c in df_s.columns if c not in ("label",)]
X_all = df_s[all_feat_cols].values.astype(np.float64)
y_all = df_s["label"].values.astype(np.int64)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_all)

folds = build_purged_folds(df_s.index, GUARDIAN_N_FOLDS, GUARDIAN_PURGE_GAP_BARS)

# Collect feature importance across folds
imp_accum = np.zeros(len(all_feat_cols), dtype=np.float64)
best_logloss = float("inf")
best_iters_acc = []

for fold_idx, (train_idx, test_idx) in enumerate(folds):
    if len(test_idx) < 20: continue
    X_tr, y_tr = X_scaled[train_idx], y_all[train_idx]
    X_te, y_te = X_scaled[test_idx],  y_all[test_idx]

    model = lgb.LGBMClassifier(**GUARDIAN_LGBM_PARAMS)
    model.fit(X_tr, y_tr,
              eval_set=[(X_te, y_te)],
              eval_metric="multi_logloss",
              callbacks=[lgb.early_stopping(GUARDIAN_EARLY_STOPPING),
                         lgb.log_evaluation(0)])

    imp_accum += model.feature_importances_
    best_iters_acc.append(model.best_iteration_ or 100)

    prob = model.predict_proba(X_te)
    pred = np.argmax(prob, axis=1)
    loss = -np.mean(np.log(prob[np.arange(len(y_te)), y_te] + 1e-10))
    f1   = f1_score(y_te, pred, average="macro", zero_division=0)

    if loss < best_logloss:
        best_logloss = loss
    logger.info(f"  Fold {fold_idx+1}: logloss={loss:.4f}  f1={f1:.4f}")

# Feature importance analysis
imp_avg = imp_accum / len([f for f in folds if len(f[1]) >= 20])
imp_total = imp_avg.sum()
imp_frac = imp_avg / imp_total

# Select features above threshold
selected = []
dropped  = []
for i, c in enumerate(all_feat_cols):
    dyn_tag = " [DYN]" if c in set(DYNAMIC_FEATS) else ""
    if imp_frac[i] >= MIN_IMP_FRAC:
        selected.append(c)
    else:
        dropped.append(c)
    logger.info(f"  {c:<40} imp={imp_frac[i]:.4f} ({imp_frac[i]*100:.1f}%)"
                f"{' DROP' if imp_frac[i] < MIN_IMP_FRAC else ''}{dyn_tag}")

print(f"\n  Selected: {len(selected)} features ({len(selected)-len(DYNAMIC_FEATS)} static + {len(DYNAMIC_FEATS)} dynamic)")
print(f"  Dropped : {len(dropped)} low-importance features")
if dropped:
    print(f"  Dropped list: {dropped}")

# ═══════════════════════════════════════════════════════════════════════════════
# Phase 3: Retrain with selected features only
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n[3/4] Retraining with {len(selected)} selected features...")

X_sel = df_s[selected].values.astype(np.float64)
scaler_sel = StandardScaler()
X_sel_scaled = scaler_sel.fit_transform(X_sel)

cv_results = []
best_score = float("inf")
best_iters_final = 100

for fold_idx, (train_idx, test_idx) in enumerate(folds):
    if len(test_idx) < 20: continue
    X_tr, y_tr = X_sel_scaled[train_idx], y_all[train_idx]
    X_te, y_te = X_sel_scaled[test_idx],  y_all[test_idx]

    model = lgb.LGBMClassifier(**GUARDIAN_LGBM_PARAMS)
    model.fit(X_tr, y_tr,
              eval_set=[(X_te, y_te)],
              eval_metric="multi_logloss",
              callbacks=[lgb.early_stopping(GUARDIAN_EARLY_STOPPING),
                         lgb.log_evaluation(0)])

    prob = model.predict_proba(X_te)
    pred = np.argmax(prob, axis=1)
    f1   = f1_score(y_te, pred, average="macro", zero_division=0)
    loss = -np.mean(np.log(prob[np.arange(len(y_te)), y_te] + 1e-10))
    cv_results.append({"fold": fold_idx+1, "logloss": round(loss,4), "f1": round(f1,4)})
    logger.info(f"  Fold {fold_idx+1}: logloss={loss:.4f}  f1={f1:.4f}")

    if loss < best_score:
        best_score = loss
        best_iters_final = model.best_iteration_ or 100

print(f"  Best CV logloss: {best_score:.4f}")

# Final retrain
n_est = int(np.median(best_iters_acc))
params_final = {**GUARDIAN_LGBM_PARAMS, "n_estimators": n_est}
final_model = lgb.LGBMClassifier(**params_final)
final_model.fit(X_sel_scaled, y_all)

# ═══════════════════════════════════════════════════════════════════════════════
# Phase 4: Save
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[4/4] Saving...")
joblib.dump(final_model,  OUT_DIR / "guardian.pkl")
joblib.dump(scaler_sel,   OUT_DIR / "guardian_scaler.pkl")
with open(OUT_DIR / f"{RUN_NAME}_feature_cols.json", "w") as f:
    json.dump(selected, f, indent=2)

meta = {
    "run_name"      : RUN_NAME,
    "entry_model"   : "tb_lgbm_widyawardhana_v3",
    "trade_pool"    : "Triple Barrier (TP=2xATR, SL=1.5xATR, max_hold=36)",
    "n_features_total" : len(all_feat_cols),
    "n_features_sel"  : len(selected),
    "n_dropped"       : len(dropped),
    "dropped_features": dropped,
    "n_samples"       : len(df_s),
    "label_dist"      : {"HOLD": n0, "PARTIAL": n1, "FULL_EXIT": n2},
    "cv_results"      : cv_results,
    "best_cv_logloss" : round(best_score, 4),
    "min_imp_frac"    : MIN_IMP_FRAC,
    "train_period"    : f"2020-01-01 – {TRAIN_CUTOFF_DATE}",
    "trained_at"      : datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
}
with open(OUT_DIR / f"{RUN_NAME}_meta.json", "w") as f:
    json.dump(meta, f, indent=2)

# Importance
imp_final = sorted(zip(selected, final_model.feature_importances_),
                   key=lambda x: x[1], reverse=True)
dyn_set = set(DYNAMIC_FEATS)
dyn_share = sum(v for n,v in imp_final if n in dyn_set) / max(sum(v for _,v in imp_final), 1) * 100

print(f"\n  Final feature importance ({len(selected)} features):")
print(f"    Dynamic : {dyn_share:.1f}%")
print(f"    Static  : {100-dyn_share:.1f}%")
print(f"  Top 10:")
for name, val in imp_final[:10]:
    tag = "[DYN]" if name in dyn_set else ""
    print(f"    {name:<40} {val:>8.0f}  {tag}")

print(f"\n  CV Summary ({len(cv_results)} folds):")
for r in cv_results:
    print(f"    Fold {r['fold']}: logloss={r['logloss']}  f1={r['f1']}")

print(f"\n{'='*65}")
print(f"  Guardian v2 saved -> {OUT_DIR}")
print(f"  Files: guardian.pkl | guardian_scaler.pkl | {RUN_NAME}_feature_cols.json")
print(f"{'='*65}")
