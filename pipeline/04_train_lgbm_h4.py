"""
pipeline/04_train_lgbm_h4.py — Fase 4b: LightGBM H4 Regime Filter Training

H4 LGBM berfungsi sebagai regime filter (bias gate) dalam hierarchical cascade:
  STEP 1: H4 LGBM → bias direction (LONG/SHORT/FLAT)  ← file ini
  STEP 2: H1 LGBM → entry signal (04_train_lgbm.py)
  STEP 3: LSTM    → momentum confirmation (05_train_lstm.py)

Key differences dari H1 LGBM:
  - Data diresample dari H1 → H4 sebelum training
  - Fitur: subset H4_FEATURE_COLS (30 fitur H4-specific)
  - Label: dihitung ulang dengan H4_SWING_LABEL_MIN_RR=0.6 (ATR-based; RR = min_tp/max_sl ≈ 0.667)
  - Model: LGBM_H4_PARAMS (lebih konservatif karena ~1/4 sample H1)
  - Purge gap: H4_PURGE_GAP_BARS=6 bars H4 ≈ 24 jam

Guard: verifikasi proporsi LONG/SHORT di H4 labels >= 2-3% total bars.

Jalankan:
  python pipeline/04_train_lgbm_h4.py               # training coins (default)
  python pipeline/04_train_lgbm_h4.py --all         # semua 20 koin
  python pipeline/04_train_lgbm_h4.py --run-id my_run

Output:
  models/lgbm_h4.pkl            ← H4 regime filter model
  models/h4_feature_cols.json   ← H4 feature list
  models/runs/{run_id}/lgbm_h4_cv_results.json
"""

import argparse
import json
import sys
import warnings
from datetime import datetime
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from config import (
    TRAINING_COINS, ALL_COINS, SYMBOL_MAP,
    LABEL_DIR, MODEL_DIR,
    LGBM_H4_PARAMS, LGBM_EARLY_STOPPING,
    H4_N_FOLDS, H4_PURGE_GAP_BARS,
    H4_FEATURE_COLS, H4_BINARY_THRESHOLD_LONG,
    H4_SWING_LABEL_MIN_RR, H4_SWING_LABEL_MIN_TP,
    H4_SWING_LABEL_MAX_SL, H4_SWING_LABEL_MAX_HOLD,
    LABEL_MAP,
)
from core.models import ProbabilityCalibrator
from core.utils import setup_logger

logger = setup_logger("04_train_lgbm_h4")

NON_FEATURE_COLS = {"label", "h4_swing_high", "h4_swing_low"}


# ─── H4 Label Generation ──────────────────────────────────────────────────────

def compute_h4_swing_labels(
    df_h4: pd.DataFrame,
    min_rr: float,
    min_tp_atr: float,
    max_sl_atr: float,
    max_hold: int,
) -> pd.Series:
    """
    Hitung swing labels (LONG/FLAT/SHORT) pada H4 timeframe.
    Menggunakan ATR H4 (atr_14_h4) sebagai basis TP/SL.

    Logic:
      - LONG  : dalam max_hold bar ke depan, price mencapai TP (min_tp_atr × atr)
                sebelum mencapai SL (max_sl_atr × atr) dengan RR >= min_rr
      - SHORT : kebalikannya
      - FLAT  : tidak memenuhi syarat LONG atau SHORT
    """
    n = len(df_h4)
    labels = np.full(n, "FLAT", dtype=object)

    close = df_h4["close"].values
    atr   = df_h4["atr_14_h4"].values if "atr_14_h4" in df_h4.columns else \
            df_h4["atr_14_h1"].values if "atr_14_h1" in df_h4.columns else \
            np.abs(np.diff(close, prepend=close[0])) * 5  # rough fallback

    for i in range(n - 1):
        c   = close[i]
        a   = atr[i]
        if a <= 0 or np.isnan(a):
            continue

        tp_long  = c + min_tp_atr * a
        sl_long  = c - max_sl_atr * a
        tp_short = c - min_tp_atr * a
        sl_short = c + max_sl_atr * a

        rr_long  = (tp_long - c) / (c - sl_long)   if (c - sl_long)  > 0 else 0
        rr_short = (c - tp_short) / (sl_short - c)  if (sl_short - c) > 0 else 0

        # Scan forward bars
        end = min(i + 1 + max_hold, n)
        future_close = close[i + 1:end]

        long_hit  = False
        short_hit = False

        for fc in future_close:
            if fc >= tp_long and not long_hit and not short_hit:
                long_hit = True
                break
            if fc <= sl_long:
                break

        if not long_hit:
            for fc in future_close:
                if fc <= tp_short and not short_hit:
                    short_hit = True
                    break
                if fc >= sl_short:
                    break

        if long_hit and rr_long >= min_rr:
            labels[i] = "LONG"
        elif short_hit and rr_short >= min_rr:
            labels[i] = "SHORT"

    return pd.Series(labels, index=df_h4.index, name="label")


# ─── Data Loading & H4 Resampling ────────────────────────────────────────────

def load_and_resample_to_h4(symbol: str) -> pd.DataFrame | None:
    """
    Load H1 features parquet, resample ke H4 OHLCV,
    lalu merge kembali H4-native features yang sudah ada.

    Strategy:
      1. Load {symbol}_features_v3.parquet (H1 data dengan fitur H4)
      2. Resample OHLCV ke H4 (open=first, high=max, low=min, close=last, vol=sum)
      3. Ambil H4-aligned features (ema_*_h4, rsi_h4, dll.) dengan last() per H4 bar
      4. Hitung label H4 menggunakan compute_h4_swing_labels()
    """
    path = LABEL_DIR / f"{symbol}_features_v3.parquet"
    if not path.exists():
        logger.warning(f"[{symbol}] File tidak ditemukan: {path}")
        return None

    df = pd.read_parquet(path)
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, utc=True)
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    df = df.sort_index()

    # ── Resample OHLCV ke H4 ──────────────────────────────────────────────────
    ohlcv_cols = {
        "open":   "first",
        "high":   "max",
        "low":    "min",
        "close":  "last",
        "volume": "sum",
    }
    available_ohlcv = {k: v for k, v in ohlcv_cols.items() if k in df.columns}
    df_h4_ohlcv = df[list(available_ohlcv.keys())].resample("4h").agg(available_ohlcv)

    # ── Ambil H4-native features dengan last() per H4 bar ────────────────────
    h4_specific_cols = [
        c for c in H4_FEATURE_COLS
        if c in df.columns and c not in available_ohlcv
    ]
    if h4_specific_cols:
        df_h4_feat = df[h4_specific_cols].resample("4h").last()
        df_h4 = pd.concat([df_h4_ohlcv, df_h4_feat], axis=1)
    else:
        df_h4 = df_h4_ohlcv

    # ── Hapus bar dengan NaN critical ────────────────────────────────────────
    df_h4 = df_h4.dropna(subset=["open", "high", "low", "close"])

    if len(df_h4) < 100:
        logger.warning(f"[{symbol}] H4 data terlalu sedikit: {len(df_h4)} bars — skip")
        return None

    # ── Hitung H4 swing labels ────────────────────────────────────────────────
    labels = compute_h4_swing_labels(
        df_h4    = df_h4,
        min_rr   = H4_SWING_LABEL_MIN_RR,
        min_tp_atr = H4_SWING_LABEL_MIN_TP,
        max_sl_atr = H4_SWING_LABEL_MAX_SL,
        max_hold = H4_SWING_LABEL_MAX_HOLD,
    )
    df_h4["label"] = labels

    # ── Guard: cek proporsi LONG/SHORT ────────────────────────────────────────
    label_counts = df_h4["label"].value_counts(normalize=True)
    long_pct  = label_counts.get("LONG",  0.0)
    short_pct = label_counts.get("SHORT", 0.0)
    logger.info(
        f"[{symbol}] H4 labels: LONG={long_pct:.1%} SHORT={short_pct:.1%} "
        f"FLAT={label_counts.get('FLAT', 0):.1%} | total={len(df_h4):,} bars"
    )

    if (long_pct + short_pct) < 0.02:
        logger.warning(
            f"[{symbol}] ⚠ LONG+SHORT hanya {(long_pct+short_pct):.1%} — "
            f"H4 label imbalance ekstrem, pertimbangkan longgarkan H4_SWING params"
        )

    # ── Forward-fill H4 features ──────────────────────────────────────────────
    feat_cols = [c for c in H4_FEATURE_COLS if c in df_h4.columns]
    df_h4[feat_cols] = df_h4[feat_cols].ffill().fillna(0)

    # Tambahkan symbol ID
    if "symbol" in H4_FEATURE_COLS and "symbol" not in df_h4.columns:
        df_h4["symbol"] = SYMBOL_MAP.get(symbol, -1)

    logger.info(f"[{symbol}] H4 resampled: {len(df_h4):,} bars × {len(feat_cols)} features")
    return df_h4


def load_all_symbols_h4(coins: list[str]) -> pd.DataFrame:
    frames = []
    for sym in coins:
        df_h4 = load_and_resample_to_h4(sym)
        if df_h4 is not None:
            frames.append(df_h4)
    if not frames:
        raise FileNotFoundError("Tidak ada H4 data ditemukan!")
    combined = pd.concat(frames).sort_index()
    logger.info(f"Total H4: {len(combined):,} rows × {len(combined.columns)} cols")
    return combined


# ─── Walk-Forward CV (H4) ─────────────────────────────────────────────────────

def walk_forward_cv_h4(
    X: pd.DataFrame,
    y: pd.Series,
    params: dict,
    n_splits: int = 8,
    gap_bars: int = 6,
) -> list[dict]:
    """
    Walk-forward CV untuk H4 LGBM (binary: SHORT=0, LONG=1).
    gap_bars=6 bar H4 ≈ 24 jam.
    Binary class weights: SHORT=3x, LONG=3x (seimbang).
    """
    logger.info(f"H4 Binary Walk-Forward CV (n_splits={n_splits}, gap={gap_bars} bar H4 ≈ {gap_bars*4}h)")
    logger.info("Binary class weights: SHORT=3x, LONG=3x")

    tscv = TimeSeriesSplit(n_splits=n_splits, gap=gap_bars)
    # Binary sample weights: kedua kelas diperlakukan setara (3x vs class balancing)
    BINARY_WEIGHTS = {0: 3.0, 1: 3.0}  # SHORT=0, LONG=1

    results = []
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        sample_w = np.array([BINARY_WEIGHTS[int(lbl)] for lbl in y_tr], dtype=np.float32)

        model = lgb.LGBMClassifier(**params)
        model.fit(
            X_tr, y_tr,
            sample_weight = sample_w,
            eval_set      = [(X_val, y_val)],
            callbacks     = [
                lgb.early_stopping(LGBM_EARLY_STOPPING, verbose=False),
                lgb.log_evaluation(period=-1),
            ],
        )

        y_pred  = model.predict(X_val)
        y_proba = model.predict_proba(X_val)[:, 1]  # prob_LONG
        acc     = float(accuracy_score(y_val, y_pred))

        try:
            auc = float(roc_auc_score(y_val, y_proba))
        except Exception:
            auc = 0.0

        # Confusion matrix 2x2: SHORT=0, LONG=1
        cm = confusion_matrix(y_val, y_pred, labels=[0, 1]).tolist()
        tn, fp, fn, tp_ = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
        precision_long = tp_ / (tp_ + fp + 1e-9)
        recall_long    = tp_ / (tp_ + fn + 1e-9)

        metrics = {
            "fold":           fold,
            "n_train":        len(X_tr),
            "n_val":          len(X_val),
            "best_iteration": model.best_iteration_,
            "accuracy":       round(acc, 4),
            "auc":            round(auc, 4),
            "precision_LONG": round(precision_long, 4),
            "recall_LONG":    round(recall_long, 4),
            "confusion_matrix": cm,
        }

        logger.info(
            f"  Fold {fold}: AUC={auc:.4f} | Acc={acc:.4f} | "
            f"Prec-LONG={precision_long:.4f} Rec-LONG={recall_long:.4f} | "
            f"val={len(y_val):,} (SHORT={int((y_val==0).sum())} LONG={int((y_val==1).sum())})"
        )

        results.append({
            "metrics": metrics, "model": model, "auc": auc,
            "val_proba": y_proba, "val_labels_binary": y_val.values,
        })

    return results


# ─── Main ─────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="H4 LGBM Regime Filter Training")
    parser.add_argument("--all",    action="store_true", help="Pakai semua koin")
    parser.add_argument("--run-id", default=None, help="Custom run ID")
    return parser.parse_args()


def main():
    args   = parse_args()
    coins  = ALL_COINS if args.all else TRAINING_COINS
    run_id = args.run_id or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    run_dir = MODEL_DIR / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"[H4 LGBM] Run ID: {run_id} | Output: {run_dir}")

    # ── Load & resample semua koin ke H4 ─────────────────────────────────────
    df_h4 = load_all_symbols_h4(coins)

    # ── Encode labels (3-class dulu, lalu drop FLAT) ──────────────────────────
    LABEL_MAP_H4 = {"SHORT": 0, "FLAT": 1, "LONG": 2}
    mask = df_h4["label"].isin(LABEL_MAP_H4)
    if (~mask).sum():
        logger.warning(f"Drop {(~mask).sum():,} baris label tidak dikenal")
    df_h4 = df_h4[mask].copy()
    y_3cls = df_h4["label"].map(LABEL_MAP_H4).astype(np.int32)

    # Label distribution sebelum drop FLAT
    label_dist = y_3cls.value_counts(normalize=True)
    logger.info(
        f"H4 label distribution (raw): "
        f"SHORT={label_dist.get(0, 0):.1%} "
        f"FLAT={label_dist.get(1, 0):.1%} "
        f"LONG={label_dist.get(2, 0):.1%} | "
        f"Total: {len(df_h4):,} bars"
    )

    # ── Drop FLAT — binary classification hanya LONG vs SHORT ─────────────────
    binary_mask = y_3cls != 1  # 1 = FLAT
    n_dropped   = (~binary_mask).sum()
    df_h4 = df_h4[binary_mask].copy()
    y_3cls = y_3cls[binary_mask]
    logger.info(f"Dropped {n_dropped:,} FLAT samples → binary training: "
                f"SHORT={int((y_3cls==0).sum()):,} LONG={int((y_3cls==2).sum()):,}")

    # Remap: SHORT=0 → 0, LONG=2 → 1 (binary)
    y = y_3cls.map({0: 0, 2: 1}).astype(np.int32)

    # ── Feature matrix ────────────────────────────────────────────────────────
    feat_cols = [c for c in H4_FEATURE_COLS if c in df_h4.columns]
    df_X      = df_h4[feat_cols]
    logger.info(f"H4 Binary Features: {len(feat_cols)} | Coins: {coins}")

    # ── Walk-Forward CV ───────────────────────────────────────────────────────
    cv_results = walk_forward_cv_h4(
        df_X, y, LGBM_H4_PARAMS,
        n_splits  = H4_N_FOLDS,
        gap_bars  = H4_PURGE_GAP_BARS,
    )

    best_model, best_auc, best_fold = None, -1.0, -1
    all_val_proba   = []   # ← untuk concatenated calibration
    all_val_labels  = []   # ←
    all_metrics = []

    for res in cv_results:
        metrics = res["metrics"]
        model   = res["model"]
        all_metrics.append(metrics)
        # Kumpulkan semua validation fold untuk kalibrasi global
        all_val_proba.append(res["val_proba"])
        all_val_labels.append(res["val_labels_binary"])
        if res["auc"] > best_auc:
            best_auc, best_model, best_fold = res["auc"], model, metrics["fold"]

    # ── Simpan model ──────────────────────────────────────────────────────────
    model_path_run  = run_dir / "lgbm_h4.pkl"
    model_path_root = MODEL_DIR / "lgbm_h4.pkl"
    joblib.dump(best_model, model_path_run)
    joblib.dump(best_model, model_path_root)
    logger.info(f"[H4 LGBM] Best model (fold {best_fold}, AUC={best_auc:.4f}) → {model_path_root}")

    # ── Fit & simpan isotonic calibrator (concatenated all folds) ─────────────
    if all_val_proba:
        val_proba_all  = np.concatenate(all_val_proba)
        val_labels_all = np.concatenate(all_val_labels)
        logger.info(
            f"[H4 LGBM] Fitting calibrator on {len(val_proba_all):,} samples "
            f"(concatenated {len(all_val_proba)} folds)"
        )

        # Log percentile BEFORE calibration
        p_before = np.percentile(val_proba_all, [10, 25, 50, 75, 90])
        logger.info(
            f"[H4 LGBM] Val proba BEFORE calibration — "
            f"P10={p_before[0]:.3f} P25={p_before[1]:.3f} "
            f"P50={p_before[2]:.3f} P75={p_before[3]:.3f} P90={p_before[4]:.3f}"
        )

        calibrator = ProbabilityCalibrator()
        calibrator.fit(val_proba_all.reshape(-1, 1), val_labels_all)

        # Log percentile AFTER calibration
        cal_proba = calibrator.transform(val_proba_all.reshape(-1, 1))[:, 0]
        p_after = np.percentile(cal_proba, [10, 25, 50, 75, 90])
        logger.info(
            f"[H4 LGBM] Val proba AFTER  calibration — "
            f"P10={p_after[0]:.3f} P25={p_after[1]:.3f} "
            f"P50={p_after[2]:.3f} P75={p_after[3]:.3f} P90={p_after[4]:.3f}"
        )
        logger.info(
            f"[H4 LGBM] Calibration shift: "
            f"P50 {p_before[2]:.3f}→{p_after[2]:.3f} "
            f"({(p_after[2]-p_before[2])*100:+.1f}pp) | "
            f"threshold={H4_BINARY_THRESHOLD_LONG} → "
            f"pass_rate_before={(val_proba_all >= H4_BINARY_THRESHOLD_LONG).mean():.1%} "
            f"pass_rate_after={(cal_proba >= H4_BINARY_THRESHOLD_LONG).mean():.1%}"
        )

        cal_path_run  = run_dir / "h4_calibrator.pkl"
        cal_path_root = MODEL_DIR / "h4_calibrator.pkl"
        calibrator.save(cal_path_run)
        calibrator.save(cal_path_root)
        logger.info(f"[H4 LGBM] Isotonic calibrator (concatenated {len(all_val_proba)} folds) → {cal_path_root}")
    else:
        logger.warning("[H4 LGBM] Tidak ada calibration data — skip isotonic calibrator")

    # ── Simpan H4 feature cols ────────────────────────────────────────────────
    h4_feat_path = MODEL_DIR / "h4_feature_cols.json"
    with open(h4_feat_path, "w") as f:
        json.dump(feat_cols, f, indent=2)
    logger.info(f"H4 feature cols ({len(feat_cols)}) → {h4_feat_path}")

    # ── CV summary ────────────────────────────────────────────────────────────
    aucs = [m["auc"]      for m in all_metrics]
    accs = [m["accuracy"] for m in all_metrics]

    cv_summary = {
        "run_id":          run_id,
        "coins":           coins,
        "n_folds":         H4_N_FOLDS,
        "gap_bars_h4":     H4_PURGE_GAP_BARS,
        "model_type":      "binary (SHORT=0, LONG=1)",
        "best_fold":       best_fold,
        "best_auc":        round(best_auc, 4),
        "mean_auc":        round(float(np.mean(aucs)), 4),
        "std_auc":         round(float(np.std(aucs)), 4),
        "mean_accuracy":   round(float(np.mean(accs)), 4),
        "lgbm_h4_params":  LGBM_H4_PARAMS,
        "h4_feature_cols": feat_cols,
        "h4_label_params": {
            "min_rr":     H4_SWING_LABEL_MIN_RR,
            "min_tp_atr": H4_SWING_LABEL_MIN_TP,
            "max_sl_atr": H4_SWING_LABEL_MAX_SL,
            "max_hold":   H4_SWING_LABEL_MAX_HOLD,
        },
        "folds": all_metrics,
    }

    cv_path = run_dir / "lgbm_h4_cv_results.json"
    with open(cv_path, "w") as f:
        json.dump(cv_summary, f, indent=2, default=str)

    # ── Print summary ─────────────────────────────────────────────────────────
    sep = "=" * 60
    print(f"\n{sep}")
    print(f"  H4 LGBM BINARY TRAINING SELESAI — {run_id}")
    print(f"{sep}")
    print(f"  Model type : binary (SHORT=0, LONG=1)")
    print(f"  Best fold  : {best_fold} (AUC={best_auc:.4f})")
    print(f"  Mean AUC   : {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")
    print(f"  Mean Acc   : {np.mean(accs):.4f}")
    print(f"  Model      : {model_path_root}")
    print(f"  CV results : {cv_path}")
    print(f"{sep}\n")

    print(f"  {'Fold':>4}  {'Acc':>7}  {'AUC':>7}  {'Prec-L':>7}  {'Rec-L':>7}  {'Iter':>6}")
    print("  " + "-" * 48)
    for m in all_metrics:
        print(
            f"  {m['fold']:>4}  {m['accuracy']:>7.4f}  {m['auc']:>7.4f}  "
            f"{m['precision_LONG']:>7.4f}  {m['recall_LONG']:>7.4f}  "
            f"{m['best_iteration']:>6}"
        )


if __name__ == "__main__":
    main()
