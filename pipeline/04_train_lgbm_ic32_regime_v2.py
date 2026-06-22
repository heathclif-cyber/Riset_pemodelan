"""
pipeline/04_train_lgbm_ic32_regime_v2.py — LGBM Genuine OOF Training

METODOLOGI GENUINE (tidak ada leakage ke holdout):
  1. Rolling walk-forward purged CV (N_FOLDS=8, purge=PURGE_GAP_BARS=20, window_splits=3)
  2. OOF predictions dikumpulkan per bar selama CV
     - Setiap bar diprediksi oleh model yang TIDAK pernah melihatnya
  3. Threshold sweep via OOF simulation (BUKAN holdout, BUKAN final model)
  4. Final model dilatih pada SELURUH training period

Output:
  - oof_predictions.parquet  : OOF probas + coin + has_oof flag
  - best_thresholds.json     : {thr_long, thr_short} hasil sweep OOF
  - lgbm.pkl                 : final model
  - features.json            : feature list yang dipakai
  - cv_results.json          : CV metrics per fold

Jalankan: python pipeline/04_train_lgbm_ic32_regime_v2.py
"""
import json, sys, warnings, itertools
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
import joblib, lightgbm as lgb
from sklearn.metrics import f1_score

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from pipeline.shared import build_rolling_folds
from core.evaluator import simulate_trades_swing
from core.utils import setup_logger, ensure_utc_index
from config import (
    ALL_COINS, TRAIN_CUTOFF_DATE, N_FOLDS, PURGE_GAP_BARS,
    MAX_HOLDING_BARS, MODAL_PER_TRADE, LEVERAGE_SIM, FEE_PER_SIDE,
    SLIPPAGE_PER_SIDE, SWING_LABEL_MIN_RR, SWING_LABEL_MIN_TP,
    SWING_LABEL_MAX_SL, TP_SL_FALLBACK_TP, TP_SL_FALLBACK_SL,
    MODEL_DIR, LABEL_DIR, LABEL_MAP as CONFIG_LABEL_MAP,
    LGBM_PARAMS as CONFIG_LGBM_PARAMS, LGBM_EARLY_STOPPING,
    LGBM_CLASS_WEIGHTS,
)

logger = setup_logger("04_train_lgbm_ic32_regime_v2")

RUN_NAME = "ic32_regime_v2_parity"

# LGBM params — dari config.py (ic32 genuine), n_estimators ditingkatkan ke 1500
LGBM_PARAMS = {**CONFIG_LGBM_PARAMS, "n_estimators": 1500}
EARLY_STOP = LGBM_EARLY_STOPPING  # 50 dari config

# Swing label params
MAX_HOLD = MAX_HOLDING_BARS     # 36
TP_ATR   = TP_SL_FALLBACK_TP   # 2.0  (untuk simulasi threshold sweep)
SL_ATR   = TP_SL_FALLBACK_SL   # 1.5

# Threshold sweep grid — mulai dari 0.55 (sesuai ic32 conf_thr range 0.59+)
THR_LONGS  = [0.55, 0.60, 0.65, 0.70, 0.75]
THR_SHORTS = [0.55, 0.60, 0.65, 0.70]

LM = {"SHORT": 0, "FLAT": 1, "LONG": 2}


LABEL_MAP = {"SHORT": 0, "FLAT": 1, "LONG": 2}


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
        if "label" not in df.columns:
            logger.warning(f"[{sym}] Kolom 'label' tidak ada — skip")
            continue
        # Swing label dari 03_engineer.py
        mask = df["label"].astype(str).isin(LABEL_MAP)
        df = df[mask].copy()
        df["y"] = df["label"].map(LABEL_MAP).astype(np.int32)
        if len(df) < 100:
            continue
        df["coin"] = sym
        dist = df["label"].value_counts()
        n = len(df)
        logger.info(
            f"[{sym}] {n:,} bars | "
            f"LONG={dist.get('LONG',0)/n*100:.1f}% "
            f"SHORT={dist.get('SHORT',0)/n*100:.1f}% "
            f"FLAT={dist.get('FLAT',0)/n*100:.1f}%"
        )
        frames.append(df)
    if not frames:
        raise RuntimeError("Tidak ada data training! Pastikan LABEL_DIR berisi features_v3.parquet")
    return pd.concat(frames).sort_index()


def oof_threshold_sweep(df: pd.DataFrame, oof_probas: np.ndarray) -> list:
    """
    Sweep threshold pada OOF predictions (BUKAN holdout).
    Return list of result dicts, sorted by pnl descending.
    """
    results = []
    for thr_long, thr_short in itertools.product(THR_LONGS, THR_SHORTS):
        agg_trades = agg_wins = 0
        agg_pnl = 0.0

        for sym in df["coin"].unique():
            sym_mask = (df["coin"].values == sym)
            sym_probas = oof_probas[sym_mask]
            sym_df = df[sym_mask]

            # Hanya gunakan bar dengan OOF predictions
            has_oof = ~np.isnan(sym_probas[:, 0])
            if has_oof.sum() < 30:
                continue

            sym_probas = sym_probas[has_oof]
            sym_df = sym_df[has_oof]
            n = len(sym_df)

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

        if agg_trades < 200:   # minimum trades untuk threshold yang bermakna
            continue

        wr  = agg_wins / agg_trades if agg_trades > 0 else 0.0
        ppt = agg_pnl / agg_trades if agg_trades > 0 else 0.0
        results.append({
            "thr_long":      round(thr_long, 2),
            "thr_short":     round(thr_short, 2),
            "trades":        agg_trades,
            "wr":            round(wr, 4),
            "pnl":           round(agg_pnl, 2),
            "pnl_per_trade": round(ppt, 4),
        })

    results.sort(key=lambda x: x["pnl_per_trade"], reverse=True)
    return results


def main():
    run_dir = MODEL_DIR / "runs" / RUN_NAME
    run_dir.mkdir(parents=True, exist_ok=True)

    # Load feature list dari feature_cols_v2.json (33 fitur ic32)
    feat_json = MODEL_DIR / "feature_cols_v2.json"
    if not feat_json.exists():
        raise FileNotFoundError(f"Feature list tidak ditemukan: {feat_json}")
    with open(feat_json, encoding="utf-8") as f:
        FEATURES = json.load(f)

    sep = "=" * 70
    print(f"\n{sep}")
    print(f"  LGBM GENUINE OOF — {RUN_NAME}")
    print(f"  Training period : 2020-01-01 -> {TRAIN_CUTOFF_DATE.date()}")
    print(f"  Features        : {len(FEATURES)} (from feature_cols_v2.json)")
    print(f"  CV folds        : {N_FOLDS}, purge={PURGE_GAP_BARS} bars")
    print(f"  TB params       : TP={TP_ATR}x  SL={SL_ATR}x  MaxHold={MAX_HOLD}")
    print(f"  Output          : {run_dir}")
    print(f"{sep}\n")

    # ── STAGE 1: Load + Label ──────────────────────────────────────────────────
    print("STAGE 1: Loading + Swing Labels...")
    print("-" * 55)
    df = load_training_data(ALL_COINS, FEATURES)

    avail   = [c for c in FEATURES if c in df.columns]
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        logger.warning(f"{len(missing)} features missing: {missing}")
    print(f"  Features: {len(avail)}/{len(FEATURES)} tersedia")

    X_all   = df[avail].ffill().fillna(0).values.astype(np.float32)
    y_all   = df["y"].values.astype(np.int32)
    n_total = len(df)

    print(f"  Total bars      : {n_total:,}")
    print(f"  Koin            : {df['coin'].nunique()}")
    for i, name in enumerate(["SHORT", "FLAT", "LONG"]):
        print(f"  {name}          : {(y_all==i).sum():,} ({(y_all==i).mean()*100:.1f}%)")

    # ── STAGE 2: Walk-Forward CV + OOF Collection ──────────────────────────────
    print(f"\nSTAGE 2: Purged Walk-Forward CV ({N_FOLDS} folds, purge={PURGE_GAP_BARS} bar)...")
    print("-" * 55)

    oof_probas  = np.full((n_total, 3), np.nan, dtype=np.float32)
    folds       = build_rolling_folds(df.index, N_FOLDS, PURGE_GAP_BARS)
    fold_metrics = []
    fold_iters   = []

    for fold_idx, (train_idx, val_idx) in enumerate(folds, 1):
        X_tr, y_tr   = X_all[train_idx], y_all[train_idx]
        X_val, y_val = X_all[val_idx],   y_all[val_idx]

        # Class weights per sample (SHORT=3x, FLAT=1.5x, LONG=3x)
        sample_w = np.array([LGBM_CLASS_WEIGHTS[int(lbl)] for lbl in y_tr], dtype=np.float32)

        model = lgb.LGBMClassifier(**LGBM_PARAMS)
        model.fit(
            X_tr, y_tr,
            sample_weight=sample_w,
            eval_set=[(X_val, y_val)],
            callbacks=[
                lgb.early_stopping(EARLY_STOP, verbose=False),
                lgb.log_evaluation(period=-1),
            ],
        )

        # Kumpulkan OOF predictions
        fold_probas = model.predict_proba(X_val).astype(np.float32)
        oof_probas[val_idx] = fold_probas

        y_pred_val = fold_probas.argmax(axis=1)
        f1_macro   = float(f1_score(y_val, y_pred_val, average="macro", zero_division=0))
        f1_per     = f1_score(y_val, y_pred_val, average=None, labels=[0,1,2], zero_division=0)

        fold_iters.append(model.best_iteration_ or LGBM_PARAMS["n_estimators"])
        fold_metrics.append({
            "fold":      fold_idx,
            "n_train":   len(train_idx),
            "n_val":     len(val_idx),
            "best_iter": model.best_iteration_,
            "f1_macro":  round(f1_macro, 4),
            "f1_SHORT":  round(float(f1_per[0]), 4),
            "f1_FLAT":   round(float(f1_per[1]), 4),
            "f1_LONG":   round(float(f1_per[2]), 4),
        })
        logger.info(
            f"  Fold {fold_idx}: "
            f"F1={f1_macro:.4f} "
            f"S={f1_per[0]:.4f} F={f1_per[1]:.4f} L={f1_per[2]:.4f} "
            f"| iter={model.best_iteration_}"
        )

    has_oof = ~np.isnan(oof_probas[:, 0])
    mean_f1 = float(np.mean([m["f1_macro"] for m in fold_metrics]))
    std_f1  = float(np.std([m["f1_macro"]  for m in fold_metrics]))
    avg_iter = int(np.mean(fold_iters))

    print(f"\n  OOF coverage: {has_oof.sum():,}/{n_total:,} bars ({has_oof.mean()*100:.1f}%)")
    print(f"  Mean F1 macro: {mean_f1:.4f} +/- {std_f1:.4f}")
    print(f"  Avg iterations: {avg_iter}")

    # ── STAGE 3: Threshold Sweep on OOF (BUKAN holdout) ───────────────────────
    print(f"\nSTAGE 3: Threshold Sweep via OOF ({len(THR_LONGS)*len(THR_SHORTS)} combinations)...")
    print("-" * 55)
    sweep_results = oof_threshold_sweep(df, oof_probas)

    if not sweep_results:
        logger.warning("Tidak ada kombinasi dengan >= 200 trades! Pakai default 0.50/0.55")
        best = {"thr_long": 0.50, "thr_short": 0.55, "trades": 0, "wr": 0, "pnl": 0, "pnl_per_trade": 0}
    else:
        best = sweep_results[0]

    print(f"\n  Top 5 threshold combinations (by OOF PnL):")
    print(f"  {'thr_L':>6} {'thr_S':>6} {'trades':>7} {'WR%':>6} {'PnL':>8} {'PPT':>7}")
    for r in sweep_results[:5]:
        print(
            f"  {r['thr_long']:>6.2f} {r['thr_short']:>6.2f} "
            f"{r['trades']:>7,} {r['wr']*100:>6.1f}% "
            f"{r['pnl']:>8.2f} {r['pnl_per_trade']:>7.4f}"
        )
    print(f"\n  BEST: thr_long={best['thr_long']} thr_short={best['thr_short']} "
          f"PnL=${best['pnl']:.2f} WR={best['wr']*100:.1f}% trades={best['trades']:,}")

    # ── STAGE 4: Final Retrain on All Data ────────────────────────────────────
    print(f"\nSTAGE 4: Final Retrain (seluruh training period, n_estimators={avg_iter})...")
    final_params = {**LGBM_PARAMS, "n_estimators": max(avg_iter, 100)}
    final_model  = lgb.LGBMClassifier(**final_params)
    final_sample_w = np.array([LGBM_CLASS_WEIGHTS[int(lbl)] for lbl in y_all], dtype=np.float32)
    final_model.fit(X_all, y_all, sample_weight=final_sample_w)
    logger.info(f"Final model trained: {final_params['n_estimators']} estimators, {len(avail)} features")

    # ── STAGE 5: Save Outputs ─────────────────────────────────────────────────
    print(f"\nSTAGE 5: Saving to {run_dir}...")

    # OOF predictions parquet — dibutuhkan oleh 06_train_guardian_genuine_v1.py
    oof_df = pd.DataFrame({
        "coin":    df["coin"].values,
        "p0":      oof_probas[:, 0],
        "p1":      oof_probas[:, 1],
        "p2":      oof_probas[:, 2],
        "has_oof": has_oof,
        "swing_label": y_all.astype(np.int8),
    }, index=df.index)
    oof_df.to_parquet(run_dir / "oof_predictions.parquet")
    print(f"  oof_predictions.parquet saved ({has_oof.sum():,} bars dengan OOF)")

    # Best thresholds
    with open(run_dir / "best_thresholds.json", "w") as f:
        json.dump({
            "thr_long":      best["thr_long"],
            "thr_short":     best["thr_short"],
            "oof_pnl":       best["pnl"],
            "oof_wr":        best["wr"],
            "oof_trades":    best["trades"],
            "pnl_per_trade": best["pnl_per_trade"],
            "sweep_method":  "OOF_simulation",
            "sweep_all":     sweep_results,
            "created":       datetime.now().isoformat(),
        }, f, indent=2)
    print(f"  best_thresholds.json saved (thr_long={best['thr_long']}, thr_short={best['thr_short']})")

    # Final model
    joblib.dump(final_model, run_dir / "lgbm.pkl")
    print(f"  lgbm.pkl saved")

    # Feature list
    with open(run_dir / "features.json", "w") as f:
        json.dump(avail, f, indent=2)
    print(f"  features.json saved ({len(avail)} features)")

    # CV results
    cv_results = {
        "run_name":      RUN_NAME,
        "created":       datetime.now().isoformat(),
        "train_cutoff":  str(TRAIN_CUTOFF_DATE.date()),
        "n_features":    len(avail),
        "features":      avail,
        "n_samples":     n_total,
        "n_folds":       N_FOLDS,
        "purge_bars":    PURGE_GAP_BARS,
        "tp_atr":        TP_ATR,
        "sl_atr":        SL_ATR,
        "max_hold":      MAX_HOLD,
        "mean_f1_macro": round(mean_f1, 4),
        "std_f1_macro":  round(std_f1, 4),
        "avg_iterations": avg_iter,
        "best_thr_long":  best["thr_long"],
        "best_thr_short": best["thr_short"],
        "oof_pnl":        best["pnl"],
        "oof_wr":         round(best["wr"], 4),
        "oof_trades":     best["trades"],
        "folds":          fold_metrics,
    }
    with open(run_dir / "cv_results.json", "w") as f:
        json.dump(cv_results, f, indent=2)
    print(f"  cv_results.json saved")

    print(f"\n{'='*70}")
    print(f"  DONE — {RUN_NAME}")
    print(f"  OOF F1    : {mean_f1:.4f} +/- {std_f1:.4f}")
    print(f"  OOF PnL   : ${best['pnl']:.2f} ({best['trades']:,} trades, WR {best['wr']*100:.1f}%)")
    print(f"  Thresholds: LONG={best['thr_long']} SHORT={best['thr_short']}")
    print(f"")
    print(f"  LANGKAH BERIKUTNYA:")
    print(f"  python pipeline/06_train_guardian_genuine_v1.py")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
