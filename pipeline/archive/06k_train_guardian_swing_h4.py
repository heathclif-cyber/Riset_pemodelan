"""
pipeline/06k_train_guardian_swing_h4.py
Guardian TB widyawardhana dengan swing H4-based exit labels.

Perbedaan dari 06i (MFE-heuristic):
  Label lama : EXIT ketika mfe > 0.015 & pnl < mfe*0.25 (oracle MFE)
  Label baru : EXIT ketika price mencapai H4 swing HIGH (LONG) / swing LOW (SHORT)
               PARTIAL ketika approaching dalam 1.5 ATR
               + SL protection (pnl_atr < -1.0) → EXIT

Tambahan feature dynamic:
  dist_to_swing_target_atr : jarak ke swing target (ATR units, positif = belum sampai)

Output: models/runs/tb_guardian_swing_h4_v1/
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
# Guardian TB pakai purge = MAX_HOLD agar tidak ada trade yang straddling fold boundary
_PURGE = TB_PURGE_GAP_BARS  # 36

logger = setup_logger("06k_guardian_swing_h4")

RUN_NAME  = "tb_guardian_swing_h4_v1"
OUT_DIR   = MODEL_DIR / "runs" / RUN_NAME
OUT_DIR.mkdir(parents=True, exist_ok=True)

SL_MULT  = TP_SL_FALLBACK_SL
MAX_HOLD = MAX_HOLDING_BARS
LEVER    = LEVERAGE_SIM[0] if isinstance(LEVERAGE_SIM, list) else LEVERAGE_SIM
REGIME_THRESH = {0: 0.42, 1: 0.52, 2: 0.52, 3: 0.42}  # T042_R052 optimal
LM = {"SHORT": 0, "FLAT": 1, "LONG": 2}

# Thresholds untuk swing-based labeling
APPROACHING_ATR   = 1.5   # dalam X ATR dari swing target → PARTIAL
AT_TARGET_ATR     = 0.3   # dalam X ATR → FULL_EXIT (dianggap sudah di target)
LOSS_EXIT_ATR     = -1.0  # pnl_atr < ini → FULL_EXIT (stop loss zone)

DYNAMIC_FEATS = [
    "bars_held_norm", "current_pnl_pct", "current_pnl_atr",
    "max_favorable_pnl_pct", "drawdown_from_peak_pct",
    "direction", "entry_price_ratio",
    # Note: dist_to_swing_target_atr sengaja TIDAK dimasukkan — agar model
    # belajar dari label (swing-based) tanpa feature yang identik dengan label trigger.
    # Swing proximity terinferensikan dari dist_swing_high/low di TB_FEATS static.
]

# ── Load entry model ───────────────────────────────────────────────────────────
tb_model = joblib.load(MODEL_DIR / "runs" / "tb_lgbm_widyawardhana_v3" / "lgbm.pkl")
with open(MODEL_DIR / "runs" / "tb_lgbm_widyawardhana_v3" / "tb_lgbm_widyawardhana_v3_features.json") as f:
    TB_FEATS = json.load(f)

logger.info(f"Entry model: tb_lgbm_widyawardhana_v3 ({len(TB_FEATS)} features)")
logger.info(f"Static: {len(TB_FEATS)} | Dynamic: {len(DYNAMIC_FEATS)} | Total: {len(TB_FEATS)+len(DYNAMIC_FEATS)}")


def generate_samples_coin(sym):
    path = LABEL_DIR / f"{sym}_features_v3.parquet"
    if not path.exists():
        return []

    df = pd.read_parquet(path)
    df = ensure_utc_index(df).sort_index()
    df = df[df.index < TRAIN_CUTOFF_DATE]

    # HMM regime
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

    # H4 swing levels (nearest, ffill'd, shift-3 applied in engineer)
    sh_arr = df["h4_swing_high"].values.astype(np.float64) if "h4_swing_high" in df.columns \
             else np.full(n, np.nan)
    sl_arr = df["h4_swing_low"].values.astype(np.float64)  if "h4_swing_low"  in df.columns \
             else np.full(n, np.nan)

    # Fallback: estimate from ATR if missing
    for i in range(n):
        if np.isnan(sh_arr[i]):
            sh_arr[i] = close[i] * 1.02
        if np.isnan(sl_arr[i]):
            sl_arr[i] = close[i] * 0.98

    # LGBM predictions
    X = np.zeros((n, len(TB_FEATS)), dtype=np.float64)
    for idx, c in enumerate(TB_FEATS):
        if c in df.columns:
            X[:, idx] = df[c].ffill().fillna(0).values.astype(np.float64)

    proba = tb_model.predict_proba(X)
    conf  = np.max(proba, axis=1)
    yp    = np.argmax(proba, axis=1).astype(np.int32)
    for r, th in REGIME_THRESH.items():
        yp[(hmm == r) & (yp != 1) & (conf < th)] = 1

    X_static = X  # reuse TB features as static Guardian features

    # Generate trades via live-like simulation
    trades = []
    i = 0
    while i < n:
        if yp[i] == 1:
            i += 1
            continue
        direction = 1 if yp[i] == 2 else -1
        entry     = close[i]
        sl_price  = entry - direction * SL_MULT * atr[i]
        bar_out   = min(i + MAX_HOLD, n - 1)
        outcome   = "TIME_EXIT"

        for j in range(i + 1, min(i + MAX_HOLD + 1, n)):
            if (direction == 1 and low[j] <= sl_price) or \
               (direction == -1 and high[j] >= sl_price):
                bar_out = j
                outcome = "SL"
                break

        if bar_out > i + GUARDIAN_MIN_HOLD_BARS:
            trades.append({
                "bar_in": i, "bar_out": bar_out,
                "direction": direction, "entry": entry,
                "atr_entry": atr[i], "outcome": outcome,
            })
        i = bar_out + 1

    if not trades:
        return []

    # ── Build samples with swing H4-based labels ──────────────────────────────
    samples = []
    for t in trades:
        bar_in    = t["bar_in"]
        bar_out   = t["bar_out"]
        direction = t["direction"]
        entry     = t["entry"]
        atr_entry = t["atr_entry"]
        atr_pct   = atr_entry / entry if entry > 0 else 0.01

        mfe_sofar = 0.0

        for j in range(bar_in + 1, bar_out):
            bars_held = j - bar_in
            cp        = close[j]
            atr_j     = max(atr[j], 1e-8)

            if direction == 1:
                pnl = (cp - entry) / entry
                # Swing HIGH = resistance target untuk LONG
                # Valid target: hanya jika di atas entry (upside still available)
                swing_target = sh_arr[j]
                if swing_target > entry:
                    dist_to_target = (swing_target - cp) / atr_j
                else:
                    # Swing high sudah di bawah entry → target tidak valid, pakai ATR fallback
                    dist_to_target = 99.0
            else:
                pnl = (entry - cp) / entry
                # Swing LOW = support target untuk SHORT
                swing_target = sl_arr[j]
                if swing_target < entry:
                    dist_to_target = (cp - swing_target) / atr_j
                else:
                    dist_to_target = 99.0

            mfe_sofar = max(mfe_sofar, pnl)
            pnl_atr   = pnl / atr_pct if atr_pct > 0 else 0.0

            # ── Swing H4-based label ───────────────────────────────────────────
            if bars_held < GUARDIAN_MIN_HOLD_BARS:
                label = 0
            elif pnl_atr < LOSS_EXIT_ATR:
                label = 2  # SL zone — rugi > 1 ATR
            elif dist_to_target <= AT_TARGET_ATR:
                label = 2  # AT atau PAST swing target
            elif dist_to_target <= APPROACHING_ATR:
                label = 1  # APPROACHING swing target (dalam 1.5 ATR)
            elif bars_held >= MAX_HOLD - 2:
                label = 2  # near time limit
            else:
                label = 0  # HOLD

            # Dynamic features
            bars_held_norm  = bars_held / MAX_HOLD
            current_pnl_atr = pnl / atr_pct if atr_pct > 0 else 0.0
            dd_from_peak    = (mfe_sofar - pnl) / mfe_sofar if mfe_sofar > 0.001 else 0.0
            entry_price_ratio = entry / cp if cp > 0 else 1.0

            row = {c: float(X_static[j, idx]) for idx, c in enumerate(TB_FEATS)}
            row.update({
                "bars_held_norm"           : bars_held_norm,
                "current_pnl_pct"          : pnl,
                "current_pnl_atr"          : current_pnl_atr,
                "max_favorable_pnl_pct"    : mfe_sofar,
                "drawdown_from_peak_pct"   : dd_from_peak,
                "direction"                : float(direction),
                "entry_price_ratio"        : entry_price_ratio,
                "timestamp"                : df.index[j],
                "label"                    : label,
            })
            samples.append(row)

    return samples


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--all",   action="store_true")
    parser.add_argument("--coins", nargs="+", default=None)
    args  = parser.parse_args()
    coins = args.coins or (ALL_COINS if args.all else TRAINING_COINS[:5])

    print(f"\n{'='*70}")
    print(f"  TB GUARDIAN (Swing H4 Labels) — {RUN_NAME}")
    print(f"  {len(TB_FEATS)} static + {len(DYNAMIC_FEATS)} dynamic = "
          f"{len(TB_FEATS)+len(DYNAMIC_FEATS)} total features")
    print(f"  Label: EXIT at H4 swing target | APPROACHING <{APPROACHING_ATR} ATR | SL <{LOSS_EXIT_ATR} ATR")
    print(f"  Period: 2020-01-01 – {TRAIN_CUTOFF_DATE} | {len(coins)} coins")
    print(f"{'='*70}\n")

    print("[1/3] Generating in-trade samples...")
    all_samples = []
    for sym in coins:
        samples = generate_samples_coin(sym)
        if samples:
            all_samples.extend(samples)
            logger.info(f"  {sym}: {len(samples)} samples")
        else:
            logger.warning(f"  {sym}: no samples")

    print(f"  Total: {len(all_samples):,} samples")
    if not all_samples:
        print("ERROR: No samples"); sys.exit(1)

    df_s = pd.DataFrame(all_samples)
    df_s["timestamp"] = pd.to_datetime(df_s["timestamp"])
    df_s = df_s.set_index("timestamp").sort_index()

    n0 = (df_s["label"] == 0).sum()
    n1 = (df_s["label"] == 1).sum()
    n2 = (df_s["label"] == 2).sum()
    total = len(df_s)
    print(f"  Labels: HOLD={n0:,} ({n0/total*100:.1f}%)  "
          f"PARTIAL={n1:,} ({n1/total*100:.1f}%)  "
          f"FULL_EXIT={n2:,} ({n2/total*100:.1f}%)")

    # ── Train ──────────────────────────────────────────────────────────────────
    print("\n[2/3] Training Guardian (purged CV)...")

    feat_cols = [c for c in df_s.columns if c != "label"]
    X_all  = df_s[feat_cols].values.astype(np.float64)
    y_all  = df_s["label"].values.astype(np.int64)

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X_all)

    folds = build_purged_folds(df_s.index, GUARDIAN_N_FOLDS, _PURGE)
    logger.info(f"Purged CV: {len(folds)} folds")

    best_loss  = float("inf")
    best_iters = 100
    cv_results = []

    for fold_idx, (train_idx, test_idx) in enumerate(folds):
        if len(test_idx) < 20:
            continue
        X_tr, y_tr = X_scaled[train_idx], y_all[train_idx]
        X_te, y_te = X_scaled[test_idx],  y_all[test_idx]

        model = lgb.LGBMClassifier(**GUARDIAN_LGBM_PARAMS)
        model.fit(X_tr, y_tr,
                  eval_set=[(X_te, y_te)],
                  eval_metric="multi_logloss",
                  callbacks=[lgb.early_stopping(GUARDIAN_EARLY_STOPPING),
                             lgb.log_evaluation(0)])

        prob  = model.predict_proba(X_te)
        pred  = np.argmax(prob, axis=1)
        f1    = f1_score(y_te, pred, average="macro", zero_division=0)
        loss  = -np.mean(np.log(prob[np.arange(len(y_te)), y_te] + 1e-10))
        cv_results.append({"fold": fold_idx+1, "logloss": round(loss, 4), "f1": round(f1, 4)})
        logger.info(f"  Fold {fold_idx+1}: logloss={loss:.4f}  f1={f1:.4f}")

        if loss < best_loss:
            best_loss  = loss
            best_iters = model.best_iteration_ or 100

    print(f"  Best CV logloss: {best_loss:.4f}")

    # Final retrain
    print("  Retraining final model on all data...")
    params_final = {**GUARDIAN_LGBM_PARAMS, "n_estimators": best_iters}
    final_model  = lgb.LGBMClassifier(**params_final)
    final_model.fit(X_scaled, y_all)

    # ── Save ───────────────────────────────────────────────────────────────────
    print("\n[3/3] Saving...")
    joblib.dump(final_model, OUT_DIR / "guardian.pkl")
    joblib.dump(scaler,      OUT_DIR / "guardian_scaler.pkl")
    with open(OUT_DIR / f"{RUN_NAME}_feature_cols.json", "w") as f:
        json.dump(feat_cols, f, indent=2)

    meta = {
        "run_name"     : RUN_NAME,
        "entry_model"  : "tb_lgbm_widyawardhana_v3",
        "label_method" : "swing_h4",
        "label_params" : {
            "approaching_atr": APPROACHING_ATR,
            "at_target_atr"  : AT_TARGET_ATR,
            "loss_exit_atr"  : LOSS_EXIT_ATR,
        },
        "n_features"   : len(feat_cols),
        "n_static"     : len(TB_FEATS),
        "n_dynamic"    : len(DYNAMIC_FEATS),
        "n_samples"    : len(df_s),
        "label_dist"   : {"HOLD": int(n0), "PARTIAL": int(n1), "FULL_EXIT": int(n2)},
        "cv_results"   : cv_results,
        "best_cv_logloss": round(best_loss, 4),
        "purge_gap_bars": _PURGE,
        "train_period" : f"2020-01-01 – {TRAIN_CUTOFF_DATE}",
        "trained_at"   : datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(OUT_DIR / f"{RUN_NAME}_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    # Feature importance
    imp      = sorted(zip(feat_cols, final_model.feature_importances_),
                      key=lambda x: x[1], reverse=True)
    dyn_set  = set(DYNAMIC_FEATS)
    dyn_share = sum(v for n, v in imp if n in dyn_set) / sum(v for _, v in imp) * 100

    print(f"\n  Feature importance:  Dynamic={dyn_share:.1f}%  Static={100-dyn_share:.1f}%")
    print(f"  Top 10:")
    for name, val in imp[:10]:
        tag = "[DYN]" if name in dyn_set else ""
        print(f"    {name:<40} {val:>8.0f}  {tag}")

    print(f"\n  Saved: {OUT_DIR}")
    print(f"  Label dist: HOLD={n0/total*100:.1f}%  PARTIAL={n1/total*100:.1f}%  EXIT={n2/total*100:.1f}%")
