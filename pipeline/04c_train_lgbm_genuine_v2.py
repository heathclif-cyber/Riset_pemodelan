"""
pipeline/04c_train_lgbm_genuine_v2.py -- LGBM Genuine OOF Training v2

PERBEDAAN DARI genuine_v1:
  - Feature set: 34 fitur (dari sample_recommended_features.json)
    vs 27 fitur flatboost_v2 di genuine_v1
  - Tambahan macro-horizon: ret_7d, ret_14d, ret_30d, dist_pdh, dist_pdl
    (menjawab concern model tidak tau kondisi bull/bear multi-hari)
  - Fitur ic32_base yang dibuang: h4_trend (0.06% gain di sample training)
  - Training cutoff: SAMA -- 2026-04-01 (apples-to-apples vs genuine_v1)

KONTROL GENUINE (tidak berubah dari v1):
  1. Walk-forward purged CV (N_FOLDS=8, purge=36 bars)
  2. OOF predictions dikumpulkan per bar -- setiap bar diprediksi tanpa model
     pernah melihatnya saat training
  3. Threshold sweep via OOF simulation BUKAN holdout
  4. Final model dilatih pada seluruh training period
  5. Holdout Apr-Jun 2026 TIDAK dievaluasi (sudah sealed di genuine_v1)

Output: models/runs/tb_lgbm_genuine_v2/
  - oof_predictions.parquet  : OOF probas (input 06b_train_guardian_genuine_v2.py)
  - best_thresholds.json     : {thr_long, thr_short} dari OOF sweep
  - lgbm.pkl                 : final model
  - features.json            : 34 feature list
  - cv_results.json          : CV metrics per fold
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

logger = setup_logger("04c_train_lgbm_genuine_v2")

RUN_NAME        = "tb_lgbm_genuine_v2"
FEAT_SOURCE_RUN = "tb_lgbm_genuine_v1"  # sample_recommended_features.json di sini

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

TP_ATR   = TP_SL_FALLBACK_TP   # 2.0
SL_ATR   = TP_SL_FALLBACK_SL   # 1.5
MAX_HOLD = MAX_HOLDING_BARS     # 36

THR_LONGS  = [0.45, 0.50, 0.55, 0.60, 0.65]
THR_SHORTS = [0.45, 0.50, 0.55, 0.60]

LM = {"SHORT": 0, "FLAT": 1, "LONG": 2}

# Fitur yang perlu dihitung dari kolom raw (tidak ada di parquet)
COMPUTED_FEATS = {
    "ret_7d":  lambda d: np.log(d["close"] / d["close"].shift(672).replace(0, np.nan)),
    "ret_14d": lambda d: np.log(d["close"] / d["close"].shift(1344).replace(0, np.nan)),
    "ret_30d": lambda d: np.log(d["close"] / d["close"].shift(2880).replace(0, np.nan)),
    "dist_pdh": lambda d: d["close"] / d["PDH"].replace(0, np.nan) - 1,
    "dist_pdl": lambda d: d["close"] / d["PDL"].replace(0, np.nan) - 1,
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
            # Compute extra features SEBELUM CV (tidak ada leakage)
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

            dist = df["tb_label"].value_counts()
            n = len(df)
            logger.info(
                f"[{sym}] {n:,} bars | "
                f"LONG={dist.get(2,0)/n*100:.1f}% "
                f"SHORT={dist.get(0,0)/n*100:.1f}% "
                f"FLAT={dist.get(1,0)/n*100:.1f}%"
            )
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


def main():
    run_dir = MODEL_DIR / "runs" / RUN_NAME
    run_dir.mkdir(parents=True, exist_ok=True)

    # Load 34-feature list dari sample training
    feat_json = MODEL_DIR / "runs" / FEAT_SOURCE_RUN / "sample_recommended_features.json"
    if not feat_json.exists():
        raise FileNotFoundError(
            f"Feature list tidak ditemukan: {feat_json}\n"
            "Jalankan dahulu: python pipeline/04b_feature_sample_train.py"
        )
    with open(feat_json, encoding="utf-8") as f:
        rec = json.load(f)
    FEATURES = rec["recommended"]

    sep = "=" * 70
    print(f"\n{sep}")
    print(f"  TB LGBM GENUINE OOF v2 -- {RUN_NAME}")
    print(f"  Training period  : 2020-01-01 -> {TRAIN_CUTOFF_DATE.date()}")
    print(f"  Features         : {len(FEATURES)} (dari sample_recommended_features.json)")
    print(f"  CV folds         : {N_FOLDS}, purge={TB_PURGE_GAP_BARS} bars")
    print(f"  TB params        : TP={TP_ATR}x  SL={SL_ATR}x  MaxHold={MAX_HOLD}")
    print(f"  Holdout          : TIDAK DIJALANKAN (Apr-Jun 2026 sealed di genuine_v1)")
    print(f"  Output           : {run_dir}")
    print(f"{sep}\n")

    # ── STAGE 1: Load + Label ──────────────────────────────────────────────────
    print("STAGE 1: Loading + Triple Barrier Labeling...")
    print("-" * 55)
    df = load_training_data(ALL_COINS, FEATURES)

    avail   = [c for c in FEATURES if c in df.columns]
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        logger.warning(f"{len(missing)} features missing (skip): {missing}")
    print(f"  Features: {len(avail)}/{len(FEATURES)} tersedia")
    if missing:
        print(f"  Missing : {missing}")

    X_all   = df[avail].ffill().fillna(0).values.astype(np.float32)
    y_all   = df["tb_label"].values.astype(np.int32)
    n_total = len(df)

    print(f"  Total bars      : {n_total:,}")
    print(f"  Koin            : {df['coin'].nunique()}")
    for i, name in enumerate(["SHORT", "FLAT", "LONG"]):
        print(f"  {name:<6}         : {(y_all==i).sum():,} ({(y_all==i).mean()*100:.1f}%)")

    # ── STAGE 2: Walk-Forward CV + OOF Collection ──────────────────────────────
    print(f"\nSTAGE 2: Purged Walk-Forward CV ({N_FOLDS} folds, purge={TB_PURGE_GAP_BARS} bar)...")
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

    has_oof  = ~np.isnan(oof_probas[:, 0])
    mean_f1  = float(np.mean([m["f1_macro"] for m in fold_metrics]))
    std_f1   = float(np.std([m["f1_macro"]  for m in fold_metrics]))
    avg_iter = int(np.mean(fold_iters))

    print(f"\n  OOF coverage: {has_oof.sum():,}/{n_total:,} bars ({has_oof.mean()*100:.1f}%)")
    print(f"  Mean F1 macro: {mean_f1:.4f} +/- {std_f1:.4f}")
    print(f"  Avg iterations: {avg_iter}")

    # ── STAGE 3: Threshold Sweep on OOF (BUKAN holdout) ───────────────────────
    print(f"\nSTAGE 3: Threshold Sweep via OOF ({len(THR_LONGS)*len(THR_SHORTS)} kombinasi)...")
    print("-" * 55)
    sweep_results = oof_threshold_sweep(df, oof_probas)

    if not sweep_results:
        logger.warning("Tidak ada kombinasi >= 200 trades, pakai default 0.50/0.55")
        best = {"thr_long": 0.50, "thr_short": 0.55, "trades": 0, "wr": 0, "pnl": 0, "pnl_per_trade": 0}
    else:
        best = sweep_results[0]

    print(f"\n  Top 5 (by OOF PnL):")
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
    print(f"\nSTAGE 4: Final Retrain (n_estimators={avg_iter})...")
    final_params = {**LGBM_PARAMS, "n_estimators": max(avg_iter, 100)}
    final_model  = lgb.LGBMClassifier(**final_params)
    final_model.fit(X_all, y_all)
    logger.info(f"Final model: {final_params['n_estimators']} iter, {len(avail)} features")

    # ── STAGE 5: Save ─────────────────────────────────────────────────────────
    print(f"\nSTAGE 5: Saving to {run_dir}...")

    oof_df = pd.DataFrame({
        "coin":     df["coin"].values,
        "p0":       oof_probas[:, 0],
        "p1":       oof_probas[:, 1],
        "p2":       oof_probas[:, 2],
        "has_oof":  has_oof,
        "tb_label": y_all.astype(np.int8),
    }, index=df.index)
    oof_df.to_parquet(run_dir / "oof_predictions.parquet")
    print(f"  oof_predictions.parquet saved ({has_oof.sum():,} bars)")

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
    print(f"  best_thresholds.json saved")

    joblib.dump(final_model, run_dir / "lgbm.pkl")
    print(f"  lgbm.pkl saved")

    with open(run_dir / "features.json", "w") as f:
        json.dump(avail, f, indent=2)
    print(f"  features.json saved ({len(avail)} features)")

    cv_results = {
        "run_name":       RUN_NAME,
        "created":        datetime.now().isoformat(),
        "train_cutoff":   str(TRAIN_CUTOFF_DATE.date()),
        "n_features":     len(avail),
        "features":       avail,
        "n_samples":      n_total,
        "n_folds":        N_FOLDS,
        "purge_bars":     TB_PURGE_GAP_BARS,
        "tp_atr":         TP_ATR,
        "sl_atr":         SL_ATR,
        "max_hold":       MAX_HOLD,
        "mean_f1_macro":  round(mean_f1, 4),
        "std_f1_macro":   round(std_f1, 4),
        "avg_iterations": avg_iter,
        "best_thr_long":  best["thr_long"],
        "best_thr_short": best["thr_short"],
        "oof_pnl":        best["pnl"],
        "oof_wr":         round(best["wr"], 4),
        "oof_trades":     best["trades"],
        "folds":          fold_metrics,
        "notes":          "34 features (ic32_base minus h4_trend + 8 macro-horizon). "
                          "Holdout sealed at Apr-Jun 2026. Threshold via OOF only.",
    }
    with open(run_dir / "cv_results.json", "w") as f:
        json.dump(cv_results, f, indent=2)
    print(f"  cv_results.json saved")

    print(f"\n{sep}")
    print(f"  DONE -- {RUN_NAME}")
    print(f"  OOF F1    : {mean_f1:.4f} +/- {std_f1:.4f}  (genuine_v1: 0.3913)")
    print(f"  OOF PnL   : ${best['pnl']:.2f} ({best['trades']:,} trades, WR {best['wr']*100:.1f}%)")
    print(f"  Thresholds: LONG={best['thr_long']} SHORT={best['thr_short']}")
    print(f"")
    print(f"  LANGKAH BERIKUTNYA:")
    print(f"  python pipeline/06b_train_guardian_genuine_v2.py")
    print(f"{sep}\n")


if __name__ == "__main__":
    main()
