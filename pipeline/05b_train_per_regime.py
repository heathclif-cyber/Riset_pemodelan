"""
pipeline/05b_train_per_regime.py — Fase 05b: Per-Regime LGBM Training (B+)

Train LGBM terpisah untuk setiap regime yang terdeteksi HMM.
Ini adalah inti dari arsitektur B+ (meta-model regime switch).

Arsitektur:
  Inference time:
    1. Deteksi regime saat ini (dari hmm_regime_enc di features)
    2. Pilih model LGBM yang sesuai: lgbm_regime_{REGIME}.pkl
    3. Generate entry signal dari model tersebut

Output:
  models/lgbm_regime_{REGIME_NAME}.pkl   — 1 model per regime
  models/regime_model_meta.json          — metadata (feature_cols, regime names, dll)

Jalankan SETELAH 03b_regime_hmm.py + 04_engineer.py + 05_train_lgbm.py:
  python pipeline/05b_train_per_regime.py
  python pipeline/05b_train_per_regime.py --all
  python pipeline/05b_train_per_regime.py --min-samples 500
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
from sklearn.metrics import f1_score, accuracy_score

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from config import (
    TRAINING_COINS, ALL_COINS,
    LABEL_DIR, MODEL_DIR,
    LGBM_PARAMS, LGBM_EARLY_STOPPING,
    N_FOLDS, PURGE_GAP_BARS,
    LABEL_MAP, LABEL_MAP_INV, NUM_CLASSES,
    LGBM_CLASS_WEIGHTS,
    HMM_N_STATES, REGIME_NAMES,
    FEATURE_COLS_V3,
)
from core.utils import setup_logger
from pipeline.shared import build_purged_folds

logger = setup_logger("05b_train_per_regime")

NON_FEATURE_COLS = {"label", "h4_swing_high", "h4_swing_low", "hmm_regime", "hmm_regime_enc"}
MIN_SAMPLES_DEFAULT = 300   # minimum samples per regime untuk training


# ─── Data Loading ─────────────────────────────────────────────────────────────

def load_symbols(coins: list[str]) -> pd.DataFrame:
    frames = []
    for sym in coins:
        path = LABEL_DIR / f"{sym}_features_v3.parquet"
        if not path.exists():
            logger.warning(f"Skip {sym}: file tidak ditemukan")
            continue
        df = pd.read_parquet(path)
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index, utc=True)
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        frames.append(df)
        logger.info(f"Loaded {sym}: {len(df):,} rows")

    if not frames:
        raise FileNotFoundError("Tidak ada file parquet ditemukan!")

    combined = pd.concat(frames).sort_index()
    logger.info(f"Total: {len(combined):,} rows")
    return combined


# ─── Feature Column Resolver ─────────────────────────────────────────────────

def get_feature_cols(df: pd.DataFrame) -> list[str]:
    """Semua kolom kecuali NON_FEATURE_COLS."""
    return [c for c in df.columns if c not in NON_FEATURE_COLS]


# ─── Per-Regime Training ─────────────────────────────────────────────────────

def train_regime_model(
    df_regime: pd.DataFrame,
    feat_cols: list[str],
    regime_name: str,
    min_samples: int = MIN_SAMPLES_DEFAULT,
) -> tuple:
    """
    Train LGBM pada subset data untuk satu regime.

    Returns (model, metrics_dict) atau (None, None) jika data tidak cukup.
    """
    # Filter label valid
    mask = df_regime["label"].isin(LABEL_MAP)
    df_regime = df_regime[mask].copy()
    y = df_regime["label"].map(LABEL_MAP).astype(np.int32)

    n = len(df_regime)
    if n < min_samples:
        logger.warning(
            f"  [{regime_name}] Hanya {n} samples (min={min_samples}) — skip"
        )
        return None, None

    # Class distribution
    dist = y.value_counts().to_dict()
    logger.info(f"  [{regime_name}] Samples: {n:,} | Classes: {dist}")

    valid_cols = [c for c in feat_cols if c in df_regime.columns]
    X = df_regime[valid_cols]

    # Walk-forward folds (sama dengan 05_train_lgbm.py)
    folds = build_purged_folds(n, n_folds=N_FOLDS, purge=PURGE_GAP_BARS)

    best_model, best_f1, best_fold = None, -1.0, -1
    all_metrics = []

    for fold_num, (train_idx, val_idx) in enumerate(folds, 1):
        if len(train_idx) < 50 or len(val_idx) < 20:
            continue

        X_tr = X.iloc[train_idx]
        X_val = X.iloc[val_idx]
        y_tr = y.iloc[train_idx]
        y_val = y.iloc[val_idx]

        # Check ada semua class di training
        if len(y_tr.unique()) < 2:
            continue

        sample_w = np.array(
            [LGBM_CLASS_WEIGHTS[int(l)] for l in y_tr], dtype=np.float32
        )

        model = lgb.LGBMClassifier(**LGBM_PARAMS)
        try:
            model.fit(
                X_tr, y_tr,
                sample_weight=sample_w,
                eval_set=[(X_val, y_val)],
                callbacks=[
                    lgb.early_stopping(LGBM_EARLY_STOPPING, verbose=False),
                    lgb.log_evaluation(period=-1),
                ],
            )
        except Exception as e:
            logger.warning(f"  [{regime_name}] Fold {fold_num} fit failed: {e}")
            continue

        y_pred = model.predict(X_val)
        f1_mac = float(f1_score(y_val, y_pred, average="macro", zero_division=0))
        acc    = float(accuracy_score(y_val, y_pred))

        f1_per = f1_score(y_val, y_pred, average=None, zero_division=0, labels=[0, 1, 2])
        metrics = {
            "fold": fold_num,
            "n_train": len(X_tr), "n_val": len(X_val),
            "f1_macro": round(f1_mac, 4),
            "accuracy": round(acc, 4),
            "f1_SHORT": round(float(f1_per[0]), 4),
            "f1_FLAT":  round(float(f1_per[1]), 4),
            "f1_LONG":  round(float(f1_per[2]), 4),
            "best_iteration": getattr(model, "best_iteration_", -1),
        }
        all_metrics.append(metrics)
        logger.info(
            f"  [{regime_name}] Fold {fold_num}: F1={f1_mac:.4f} Acc={acc:.4f} "
            f"LONG={f1_per[2]:.4f} SHORT={f1_per[0]:.4f}"
        )

        if f1_mac > best_f1:
            best_f1, best_model, best_fold = f1_mac, model, fold_num

    if best_model is None:
        logger.warning(f"  [{regime_name}] Tidak ada model valid — semua fold gagal")
        return None, None

    summary = {
        "regime":         regime_name,
        "n_samples":      n,
        "class_dist":     dist,
        "best_fold":      best_fold,
        "best_f1_macro":  round(best_f1, 4),
        "mean_f1_macro":  round(float(np.mean([m["f1_macro"] for m in all_metrics])), 4),
        "n_folds_valid":  len(all_metrics),
        "folds":          all_metrics,
    }
    return best_model, summary


# ─── Main ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Per-Regime LGBM Training (B+)")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--all", action="store_true")
    g.add_argument("--coins", nargs="+", metavar="SYMBOL")
    p.add_argument("--min-samples", type=int, default=MIN_SAMPLES_DEFAULT,
                   help="Min samples per regime untuk training")
    p.add_argument("--run-id", default=None)
    return p.parse_args()


def main():
    args    = parse_args()
    run_id  = args.run_id or f"regime_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = MODEL_DIR / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    coins = ALL_COINS if args.all else (
        [c.upper() for c in args.coins] if args.coins else TRAINING_COINS
    )

    logger.info(f"Run ID: {run_id}")
    logger.info(f"Coins: {coins}")

    # Load data
    df = load_symbols(coins)

    # Check kolom regime
    if "hmm_regime" not in df.columns:
        logger.error(
            "Kolom hmm_regime tidak ditemukan! "
            "Jalankan 03b_regime_hmm.py + 04_engineer.py terlebih dahulu."
        )
        return

    feat_cols = get_feature_cols(df)
    logger.info(f"Feature cols: {len(feat_cols)}")

    # Encode label
    mask = df["label"].isin(LABEL_MAP)
    df   = df[mask].copy()

    # Cek distribusi regime
    regime_dist = df["hmm_regime"].value_counts()
    logger.info(f"\nRegime distribution (all coins):\n{regime_dist}")

    # ── Training per regime ────────────────────────────────────────────────────
    regime_models  = {}
    regime_meta    = {}
    regime_feat_cols = {}  # per-regime bisa berbeda jika di masa depan

    for regime_name in REGIME_NAMES:
        logger.info(f"\n{'─'*50}")
        logger.info(f"Training regime: {regime_name}")

        df_reg = df[df["hmm_regime"] == regime_name].copy()
        if len(df_reg) == 0:
            logger.warning(f"  [{regime_name}] Tidak ada data, skip")
            continue

        model, summary = train_regime_model(
            df_reg, feat_cols, regime_name, min_samples=args.min_samples
        )

        if model is None:
            continue

        # Simpan model per regime
        safe_name   = regime_name.replace(" ", "_")
        model_path  = MODEL_DIR / f"lgbm_regime_{safe_name}.pkl"
        joblib.dump(model, model_path)
        logger.info(f"  [{regime_name}] Model → {model_path.name}")

        # Juga simpan ke run_dir
        joblib.dump(model, run_dir / f"lgbm_regime_{safe_name}.pkl")

        regime_models[regime_name]    = model_path.name
        regime_meta[regime_name]      = summary
        regime_feat_cols[regime_name] = feat_cols

    if not regime_models:
        logger.error("Tidak ada model regime yang berhasil ditraining!")
        return

    # ── Save feature cols ──────────────────────────────────────────────────────
    feat_cols_path = MODEL_DIR / "feature_cols_v2.json"
    with open(feat_cols_path, "w") as f:
        json.dump(feat_cols, f, indent=2)

    # ── Save metadata ──────────────────────────────────────────────────────────
    meta = {
        "run_id":          run_id,
        "created_at":      datetime.now().isoformat(),
        "coins":           coins,
        "regime_names":    REGIME_NAMES,
        "n_states":        HMM_N_STATES,
        "models":          regime_models,
        "feature_cols":    feat_cols,
        "n_features":      len(feat_cols),
        "per_regime_meta": regime_meta,
    }

    meta_path = MODEL_DIR / "regime_model_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, default=str)
    logger.info(f"Metadata → {meta_path}")

    # ── Summary print ─────────────────────────────────────────────────────────
    sep = "=" * 60
    print(f"\n{sep}")
    print(f"  PER-REGIME LGBM TRAINING — {run_id}")
    print(f"{sep}")
    print(f"  {'Regime':<22}  {'Samples':>8}  {'F1-macro':>8}  {'Best Fold':>9}")
    print(f"  {'-'*52}")
    for rname, rmeta in regime_meta.items():
        print(
            f"  {rname:<22}  {rmeta['n_samples']:>8,}  "
            f"{rmeta['best_f1_macro']:>8.4f}  {rmeta['best_fold']:>9}"
        )
    print(f"{sep}")
    print(f"  Models → models/lgbm_regime_*.pkl")
    print(f"  Meta   → {meta_path.name}")
    print(f"{sep}\n")


if __name__ == "__main__":
    main()
