"""
pipeline/04e_train_lgbm_genuine_v2_dow.py -- LGBM Genuine OOF v2 + dow (cyclic day-of-week)

Ablation: genuine_v2 (34 feat) + dow_cos, dow_sin (+2 = 36 feat)

KONTROL GENUINE (wajib, sama dengan 04c):
  1. Data training ONLY: index < TRAIN_CUTOFF_DATE (audit runtime)
  2. Walk-forward purged CV (N_FOLDS=8, purge=TB_PURGE_GAP_BARS=36)
  3. OOF predictions: setiap bar diprediksi model yang tidak pernah melihatnya
  4. Threshold sweep HANYA via OOF simulation (bukan holdout)
  5. Final model = retrain seluruh training period (untuk inference)
  6. Holdout Apr-Jun 2026 TIDAK dievaluasi / TIDAK disentuh
  7. dow_cos/dow_sin = fungsi timestamp bar saja (causal, no lookahead)
  8. Promote ke artefak aktif HANYA jika --promote DAN lolos OOF gate vs baseline

Output: models/runs/tb_lgbm_genuine_v2_dow/
  - oof_predictions.parquet, best_thresholds.json, lgbm.pkl, features.json
  - cv_results.json, genuine_audit.json
"""
import argparse
import json
import shutil
import sys
import warnings
import itertools
from datetime import datetime
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from pipeline.shared import build_purged_folds
from core.features import triple_barrier_labeling
from core.evaluator import simulate_trades_swing
from core.utils import setup_logger, ensure_utc_index
from config import (
    ALL_COINS, TRAIN_CUTOFF_DATE, N_FOLDS, TB_PURGE_GAP_BARS,
    MAX_HOLDING_BARS, MODAL_PER_TRADE, LEVERAGE_SIM, FEE_PER_SIDE,
    SLIPPAGE_PER_SIDE, SWING_LABEL_MIN_RR, SWING_LABEL_MIN_TP,
    SWING_LABEL_MAX_SL, TP_SL_FALLBACK_TP, TP_SL_FALLBACK_SL,
    MODEL_DIR, LABEL_DIR,
)

logger = setup_logger("04e_train_lgbm_genuine_v2_dow")

RUN_NAME        = "tb_lgbm_genuine_v2_dow"
FEAT_SOURCE_RUN = "tb_lgbm_genuine_v1"
BASELINE_RUN    = "tb_lgbm_genuine_v2"
DOW_FEATS       = ["dow_cos", "dow_sin"]

LGBM_PARAMS = {
    "objective":         "multiclass",
    "num_class":         3,
    "n_estimators":      600,
    "learning_rate":     0.03,
    "max_depth":         5,
    "num_leaves":        31,
    "min_child_samples": 50,
    "subsample":         0.8,
    "colsample_bytree":  0.7,
    "class_weight":      "balanced",
    "verbose":           -1,
    "n_jobs":            -1,
    "random_state":      42,
}
EARLY_STOP = 50

TP_ATR   = TP_SL_FALLBACK_TP
SL_ATR   = TP_SL_FALLBACK_SL
MAX_HOLD = MAX_HOLDING_BARS

THR_LONGS  = [0.45, 0.50, 0.55, 0.60, 0.65]
THR_SHORTS = [0.45, 0.50, 0.55, 0.60]

LM = {"SHORT": 0, "FLAT": 1, "LONG": 2}

COMPUTED_FEATS = {
    "ret_7d":  lambda d: np.log(d["close"] / d["close"].shift(672).replace(0, np.nan)),
    "ret_14d": lambda d: np.log(d["close"] / d["close"].shift(1344).replace(0, np.nan)),
    "ret_30d": lambda d: np.log(d["close"] / d["close"].shift(2880).replace(0, np.nan)),
    "dist_pdh": lambda d: d["close"] / d["PDH"].replace(0, np.nan) - 1,
    "dist_pdl": lambda d: d["close"] / d["PDL"].replace(0, np.nan) - 1,
}


def assert_genuine_data_bounds(df: pd.DataFrame) -> dict:
    """Runtime audit: tidak ada bar post-cutoff."""
    max_ts = df.index.max()
    if max_ts >= TRAIN_CUTOFF_DATE:
        raise RuntimeError(
            f"LEAKAGE: data max={max_ts} >= TRAIN_CUTOFF_DATE={TRAIN_CUTOFF_DATE}"
        )
    return {
        "train_cutoff": str(TRAIN_CUTOFF_DATE),
        "data_max_timestamp": str(max_ts),
        "data_min_timestamp": str(df.index.min()),
        "n_bars": len(df),
        "cutoff_ok": True,
    }


def load_training_data(coins: list, features: list) -> pd.DataFrame:
    frames = []
    for sym in coins:
        path = LABEL_DIR / f"{sym}_features_v3.parquet"
        if not path.exists():
            logger.warning(f"[{sym}] Not found: {path.name}")
            continue
        df = pd.read_parquet(path)
        df = ensure_utc_index(df).sort_index()
        df = df[df.index < TRAIN_CUTOFF_DATE]
        if df.empty:
            continue

        for req in ["close", "high", "low", "atr_14_h1"]:
            if req not in df.columns:
                logger.warning(f"[{sym}] Missing required column: {req}")
                break
        else:
            for feat, fn in COMPUTED_FEATS.items():
                if feat in features and feat not in df.columns:
                    try:
                        df[feat] = fn(df)
                    except Exception as e:
                        logger.warning(f"[{sym}] Computed {feat} gagal: {e}")

            tb = triple_barrier_labeling(
                df["close"], df["high"], df["low"],
                df["atr_14_h1"], TP_ATR, SL_ATR, MAX_HOLD,
            )
            df["tb_label"] = tb.map({"SHORT": 0, "FLAT": 1, "LONG": 2})
            df = df.dropna(subset=["tb_label"])
            if len(df) < 100:
                continue
            df["coin"] = sym
            frames.append(df)

    if not frames:
        raise RuntimeError("Tidak ada data training!")
    return pd.concat(frames).sort_index()


def oof_threshold_sweep(df: pd.DataFrame, oof_probas: np.ndarray) -> list:
    results = []
    for thr_long, thr_short in itertools.product(THR_LONGS, THR_SHORTS):
        agg_trades = agg_wins = 0
        agg_pnl = 0.0

        for sym in df["coin"].unique():
            sym_mask   = (df["coin"].values == sym)
            sym_probas = oof_probas[sym_mask]
            sym_df     = df[sym_mask]

            has_oof = ~np.isnan(sym_probas[:, 0])
            if has_oof.sum() < 30:
                continue

            sym_probas = sym_probas[has_oof]
            sym_df     = sym_df[has_oof]
            n          = len(sym_df)

            y_pred = np.full(n, LM["FLAT"], np.int32)
            y_pred[sym_probas[:, 2] >= thr_long] = LM["LONG"]
            short_m = (sym_probas[:, 0] >= thr_short) & (y_pred != LM["LONG"])
            y_pred[short_m] = LM["SHORT"]

            if (y_pred != LM["FLAT"]).sum() == 0:
                continue

            h4_sh = sym_df["h4_swing_high"].values if "h4_swing_high" in sym_df.columns \
                else np.full(n, np.nan)
            h4_sl = sym_df["h4_swing_low"].values if "h4_swing_low" in sym_df.columns \
                else np.full(n, np.nan)

            result = simulate_trades_swing(
                y_pred=y_pred,
                close=sym_df["close"].values,
                high=sym_df["high"].values,
                low=sym_df["low"].values,
                atr=sym_df["atr_14_h1"].values,
                h4_swing_highs=h4_sh,
                h4_swing_lows=h4_sl,
                modal=MODAL_PER_TRADE,
                leverage=LEVERAGE_SIM[0],
                fee_per_side=FEE_PER_SIDE,
                slippage=SLIPPAGE_PER_SIDE,
                max_hold=MAX_HOLD,
                min_rr=SWING_LABEL_MIN_RR,
                min_tp_atr=SWING_LABEL_MIN_TP,
                max_sl_atr=SWING_LABEL_MAX_SL,
                tp_fallback_atr=TP_ATR,
                sl_fallback_atr=SL_ATR,
                guardian_enabled=False,
            )
            agg_trades += result.get("total_trades", 0)
            agg_wins   += result.get("wins", 0)
            agg_pnl    += result.get("total_pnl", 0.0)

        if agg_trades < 200:
            continue

        wr  = agg_wins / agg_trades if agg_trades > 0 else 0.0
        ppt = agg_pnl / agg_trades  if agg_trades > 0 else 0.0
        results.append({
            "thr_long":      round(thr_long, 2),
            "thr_short":     round(thr_short, 2),
            "trades":        agg_trades,
            "wr":            round(wr, 4),
            "pnl":           round(agg_pnl, 2),
            "pnl_per_trade": round(ppt, 4),
        })

    results.sort(key=lambda x: x["pnl"], reverse=True)
    return results


def check_oof_promotion_gate(mean_f1: float, oof_pnl: float,
                             baseline_f1: float, baseline_pnl: float) -> dict:
    """Promote hanya jika OOF F1 dan OOF PnL >= baseline (apples-to-apples)."""
    f1_ok  = mean_f1 >= baseline_f1
    pnl_ok = oof_pnl >= baseline_pnl
    passed = bool(f1_ok and pnl_ok)
    return {
        "f1_ok": bool(f1_ok),
        "pnl_ok": bool(pnl_ok),
        "passed": passed,
        "candidate_f1": round(mean_f1, 4),
        "baseline_f1": baseline_f1,
        "candidate_oof_pnl": oof_pnl,
        "baseline_oof_pnl": baseline_pnl,
        "delta_f1": round(mean_f1 - baseline_f1, 4),
        "delta_oof_pnl": round(oof_pnl - baseline_pnl, 2),
    }


def maybe_promote(run_dir: Path, avail: list, final_model, gate: dict) -> bool:
    if not gate["passed"]:
        logger.warning(
            "OOF gate FAIL -- tidak promote. "
            f"F1 {gate['candidate_f1']} vs {gate['baseline_f1']}, "
            f"PnL ${gate['candidate_oof_pnl']:.2f} vs ${gate['baseline_oof_pnl']:.2f}"
        )
        return False

    active_feat = MODEL_DIR / "feature_cols_v2.json"
    with open(active_feat, "w", encoding="utf-8") as f:
        json.dump(avail, f, indent=2)
    joblib.dump(final_model, MODEL_DIR / "lgbm_baseline.pkl")

    baseline_dir = MODEL_DIR / "runs" / BASELINE_RUN
    joblib.dump(final_model, baseline_dir / "lgbm.pkl")
    with open(baseline_dir / "features.json", "w") as f:
        json.dump(avail, f, indent=2)
    for fname in ["cv_results.json", "best_thresholds.json", "oof_predictions.parquet"]:
        shutil.copy2(run_dir / fname, baseline_dir / fname)

    logger.info("OOF gate PASS -- promoted to feature_cols_v2.json + lgbm_baseline.pkl")
    return True


def main():
    parser = argparse.ArgumentParser(description="Genuine OOF LGBM v2 + dow features")
    parser.add_argument(
        "--promote", action="store_true",
        help="Promote ke artefak aktif HANYA jika lolos OOF gate vs baseline genuine_v2",
    )
    args = parser.parse_args()

    run_dir = MODEL_DIR / "runs" / RUN_NAME
    run_dir.mkdir(parents=True, exist_ok=True)

    feat_json = MODEL_DIR / "runs" / FEAT_SOURCE_RUN / "sample_recommended_features.json"
    if not feat_json.exists():
        raise FileNotFoundError(f"Feature list tidak ditemukan: {feat_json}")
    with open(feat_json, encoding="utf-8") as f:
        rec = json.load(f)
    FEATURES = rec["recommended"] + DOW_FEATS

    sep = "=" * 70
    print(f"\n{sep}")
    print(f"  TB LGBM GENUINE OOF v2 + DOW -- {RUN_NAME}")
    print(f"  Training period  : 2020-01-01 -> {TRAIN_CUTOFF_DATE.date()}")
    print(f"  Base features    : 34 (genuine_v2)")
    print(f"  Added dow feats  : {DOW_FEATS}")
    print(f"  Total features   : {len(FEATURES)}")
    print(f"  CV folds         : {N_FOLDS}, purge={TB_PURGE_GAP_BARS} bars")
    print(f"  Holdout          : TIDAK DIJALANKAN (sealed)")
    print(f"  Promote flag     : {args.promote}")
    print(f"  Output           : {run_dir}")
    print(f"{sep}\n")

    print("STAGE 1: Loading + Triple Barrier Labeling...")
    print("-" * 55)
    df = load_training_data(ALL_COINS, FEATURES)
    bounds_audit = assert_genuine_data_bounds(df)

    avail   = [c for c in FEATURES if c in df.columns]
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise RuntimeError(f"Fitur dow tidak tersedia di parquet: {missing}")
    print(f"  Features: {len(avail)}/{len(FEATURES)} tersedia")
    print(f"  Data bounds OK : max={bounds_audit['data_max_timestamp']} < cutoff")
    print(f"  Total bars     : {bounds_audit['n_bars']:,}")

    X_all   = df[avail].ffill().fillna(0).values.astype(np.float32)
    y_all   = df["tb_label"].values.astype(np.int32)
    n_total = len(df)

    print(f"\nSTAGE 2: Purged Walk-Forward CV ({N_FOLDS} folds, purge={TB_PURGE_GAP_BARS})...")
    print("-" * 55)
    oof_probas   = np.full((n_total, 3), np.nan, dtype=np.float32)
    folds        = build_purged_folds(df.index, N_FOLDS, TB_PURGE_GAP_BARS)
    fold_metrics = []
    fold_iters   = []

    for fold_idx, (train_idx, val_idx) in enumerate(folds, 1):
        X_tr, y_tr   = X_all[train_idx], y_all[train_idx]
        X_val, y_val = X_all[val_idx],   y_all[val_idx]

        model = lgb.LGBMClassifier(**LGBM_PARAMS)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            callbacks=[
                lgb.early_stopping(EARLY_STOP, verbose=False),
                lgb.log_evaluation(period=-1),
            ],
        )

        fold_probas = model.predict_proba(X_val).astype(np.float32)
        oof_probas[val_idx] = fold_probas

        y_pred_val = fold_probas.argmax(axis=1)
        f1_macro   = float(f1_score(y_val, y_pred_val, average="macro", zero_division=0))
        f1_per     = f1_score(y_val, y_pred_val, average=None, labels=[0, 1, 2], zero_division=0)

        fold_iters.append(model.best_iteration_ or LGBM_PARAMS["n_estimators"])
        fold_metrics.append({
            "fold": fold_idx, "n_train": len(train_idx), "n_val": len(val_idx),
            "best_iter": model.best_iteration_,
            "f1_macro": round(f1_macro, 4),
            "f1_SHORT": round(float(f1_per[0]), 4),
            "f1_FLAT":  round(float(f1_per[1]), 4),
            "f1_LONG":  round(float(f1_per[2]), 4),
        })
        logger.info(
            f"  Fold {fold_idx}: F1={f1_macro:.4f} "
            f"S={f1_per[0]:.4f} F={f1_per[1]:.4f} L={f1_per[2]:.4f} "
            f"| iter={model.best_iteration_}"
        )

    has_oof  = ~np.isnan(oof_probas[:, 0])
    mean_f1  = float(np.mean([m["f1_macro"] for m in fold_metrics]))
    std_f1   = float(np.std([m["f1_macro"] for m in fold_metrics]))
    avg_iter = int(np.mean(fold_iters))

    print(f"  OOF coverage   : {has_oof.sum():,}/{n_total:,} ({has_oof.mean()*100:.1f}%)")
    print(f"  Mean F1 macro  : {mean_f1:.4f} +/- {std_f1:.4f}")

    print(f"\nSTAGE 3: Threshold Sweep via OOF ({len(THR_LONGS)*len(THR_SHORTS)} kombinasi)...")
    print("-" * 55)
    sweep_results = oof_threshold_sweep(df, oof_probas)
    if not sweep_results:
        logger.warning("Tidak ada kombinasi >= 200 trades, pakai default 0.50/0.55")
        best = {"thr_long": 0.50, "thr_short": 0.55, "trades": 0, "wr": 0, "pnl": 0, "pnl_per_trade": 0}
    else:
        best = sweep_results[0]

    print(f"  BEST (OOF): thr={best['thr_long']}/{best['thr_short']} "
          f"PnL=${best['pnl']:.2f} WR={best['wr']*100:.1f}% trades={best['trades']:,}")

    print(f"\nSTAGE 4: Final Retrain on full training period (n_estimators={avg_iter})...")
    final_params = {**LGBM_PARAMS, "n_estimators": max(avg_iter, 100)}
    final_model  = lgb.LGBMClassifier(**final_params)
    final_model.fit(X_all, y_all)

    gain = final_model.booster_.feature_importance(importance_type="gain")
    dow_imp = {f: round(float(g), 1) for f, g in zip(avail, gain) if f in DOW_FEATS}
    print(f"  Dow gain importance: {dow_imp}")

    baseline_cv = MODEL_DIR / "runs" / BASELINE_RUN / "cv_results.json"
    if not baseline_cv.exists():
        raise FileNotFoundError(f"Baseline cv_results tidak ditemukan: {baseline_cv}")
    with open(baseline_cv, encoding="utf-8") as f:
        bl = json.load(f)
    baseline_f1  = bl["mean_f1_macro"]
    baseline_pnl = bl["oof_pnl"]

    gate = check_oof_promotion_gate(mean_f1, best["pnl"], baseline_f1, baseline_pnl)
    print(f"\n  OOF gate vs {BASELINE_RUN}:")
    print(f"    F1  : {gate['candidate_f1']} vs {gate['baseline_f1']} "
          f"({'PASS' if gate['f1_ok'] else 'FAIL'})")
    print(f"    PnL : ${gate['candidate_oof_pnl']:.2f} vs ${gate['baseline_oof_pnl']:.2f} "
          f"({'PASS' if gate['pnl_ok'] else 'FAIL'})")

    genuine_audit = {
        "methodology": "genuine_oof_v1",
        "holdout_evaluated": False,
        "holdout_period": "2026-04-01 to 2026-06-13 (SEALED -- not touched)",
        "train_cutoff_enforced": True,
        "data_bounds": bounds_audit,
        "cv": {
            "n_folds": N_FOLDS,
            "purge_bars": TB_PURGE_GAP_BARS,
            "purge_equals_max_hold": TB_PURGE_GAP_BARS == MAX_HOLDING_BARS,
            "oof_coverage_pct": round(has_oof.mean() * 100, 2),
        },
        "threshold_selection": {
            "method": "OOF_simulation_only",
            "holdout_used": False,
            "best_thr_long": best["thr_long"],
            "best_thr_short": best["thr_short"],
        },
        "dow_features": {
            "added": DOW_FEATS,
            "causal": True,
            "source": "bar timestamp UTC (dayofweek sin/cos)",
            "no_future_data": True,
        },
        "oof_promotion_gate": gate,
        "promoted": False,
    }

    print(f"\nSTAGE 5: Saving to {run_dir}...")
    oof_df = pd.DataFrame({
        "coin": df["coin"].values,
        "p0": oof_probas[:, 0], "p1": oof_probas[:, 1], "p2": oof_probas[:, 2],
        "has_oof": has_oof, "tb_label": y_all.astype(np.int8),
    }, index=df.index)
    oof_df.to_parquet(run_dir / "oof_predictions.parquet")

    with open(run_dir / "best_thresholds.json", "w") as f:
        json.dump({
            "thr_long": best["thr_long"], "thr_short": best["thr_short"],
            "oof_pnl": best["pnl"], "oof_wr": best["wr"], "oof_trades": best["trades"],
            "pnl_per_trade": best["pnl_per_trade"],
            "sweep_method": "OOF_simulation",
            "holdout_used": False,
            "sweep_all": sweep_results,
            "created": datetime.now().isoformat(),
        }, f, indent=2)

    joblib.dump(final_model, run_dir / "lgbm.pkl")
    with open(run_dir / "features.json", "w") as f:
        json.dump(avail, f, indent=2)

    cv_results = {
        "run_name": RUN_NAME,
        "created": datetime.now().isoformat(),
        "train_cutoff": str(TRAIN_CUTOFF_DATE.date()),
        "n_features": len(avail),
        "features": avail,
        "dow_features_added": DOW_FEATS,
        "dow_feature_importance_gain": dow_imp,
        "baseline_run": BASELINE_RUN,
        "baseline_f1_macro": baseline_f1,
        "baseline_oof_pnl": baseline_pnl,
        "delta_f1_macro": gate["delta_f1"],
        "delta_oof_pnl": gate["delta_oof_pnl"],
        "oof_gate_passed": bool(gate["passed"]),
        "mean_f1_macro": round(mean_f1, 4),
        "std_f1_macro": round(std_f1, 4),
        "avg_iterations": avg_iter,
        "best_thr_long": best["thr_long"],
        "best_thr_short": best["thr_short"],
        "oof_pnl": best["pnl"],
        "oof_wr": round(best["wr"], 4),
        "oof_trades": best["trades"],
        "folds": fold_metrics,
        "notes": "genuine_v2 + dow_cos/dow_sin. Threshold via OOF only. Holdout sealed.",
    }
    with open(run_dir / "cv_results.json", "w") as f:
        json.dump(cv_results, f, indent=2)

    promoted = False
    if args.promote:
        promoted = maybe_promote(run_dir, avail, final_model, gate)
        genuine_audit["promoted"] = promoted
    else:
        print("  Promote: SKIP (--promote tidak diberikan)")

    with open(run_dir / "genuine_audit.json", "w") as f:
        json.dump(genuine_audit, f, indent=2)

    print(f"\n{sep}")
    print(f"  DONE -- {RUN_NAME}")
    print(f"  OOF F1    : {mean_f1:.4f} +/- {std_f1:.4f}")
    print(f"  OOF PnL   : ${best['pnl']:.2f} ({best['trades']:,} trades)")
    print(f"  OOF gate  : {'PASS' if gate['passed'] else 'FAIL'}")
    print(f"  Promoted  : {promoted}")
    print(f"  Audit     : {run_dir / 'genuine_audit.json'}")
    if gate["passed"] and not args.promote:
        print(f"  Untuk promote: python pipeline/04e_train_lgbm_genuine_v2_dow.py --promote")
    print(f"{sep}\n")


if __name__ == "__main__":
    main()