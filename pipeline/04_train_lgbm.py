"""
pipeline/04_train_lgbm.py — Fase 4: LightGBM Baseline Training
Purged Walk-Forward Cross-Validation (8 fold, purge gap 5 bar)

Jalankan:
  python pipeline/04_train_lgbm.py               # training coins (default)
  python pipeline/04_train_lgbm.py --all         # semua 20 koin
  python pipeline/04_train_lgbm.py --run-id my_run  # custom run ID

Output disimpan di models/runs/{run_id}/
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
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from config import (
    TRAINING_COINS, ALL_COINS, SYMBOL_MAP,
    LABEL_DIR, MODEL_DIR,
    LGBM_PARAMS, LGBM_EARLY_STOPPING,
    N_FOLDS, PURGE_GAP_BARS,
    LABEL_MAP, LABEL_MAP_INV, NUM_CLASSES,
)
from core.utils import setup_logger

logger = setup_logger("04_train_lgbm")

NON_FEATURE_COLS = {"label"}


# ─── Data Loading ─────────────────────────────────────────────────────────────

def load_symbols(coins: list[str]) -> pd.DataFrame:
    frames = []
    for sym in coins:
        path = LABEL_DIR / f"{sym}_features.parquet"
        if not path.exists():
            logger.warning(f"File tidak ditemukan, skip: {path}")
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
    logger.info(f"Total: {len(combined):,} rows × {len(combined.columns)} cols")
    return combined


def encode_labels(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    mask = df["label"].isin(LABEL_MAP)
    if (~mask).sum():
        logger.warning(f"Drop {(~mask).sum():,} baris label tidak dikenal.")
        df = df[mask].copy()
    y = df["label"].map(LABEL_MAP).values.astype(np.int32)
    return df, y


def compute_sample_weights(y: np.ndarray) -> np.ndarray:
    counts = np.bincount(y, minlength=NUM_CLASSES)
    counts = np.where(counts == 0, 1, counts)
    cw = len(y) / (NUM_CLASSES * counts)
    for idx, name in LABEL_MAP_INV.items():
        logger.info(f"  '{name}': count={counts[idx]:,}, weight={cw[idx]:.4f}")
    return cw[y].astype(np.float32)


def get_feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in NON_FEATURE_COLS]


# ─── Walk-Forward CV ─────────────────────────────────────────────────────────

def build_purged_folds(n: int) -> list[tuple[np.ndarray, np.ndarray]]:
    splits = np.array_split(np.arange(n), N_FOLDS + 1)
    folds  = []
    for k in range(1, N_FOLDS + 1):
        train_raw = np.concatenate(splits[:k])
        test_idx  = splits[k]
        train_idx = train_raw[:-PURGE_GAP_BARS] if len(train_raw) > PURGE_GAP_BARS else train_raw
        folds.append((train_idx, test_idx))
    return folds


def train_fold(X_tr, y_tr, sw_tr, X_val, y_val, fold_num, feat_cols):
    logger.info(f"[Fold {fold_num}] Train={len(X_tr):,} | Val={len(X_val):,}")
    model = lgb.LGBMClassifier(**LGBM_PARAMS)
    model.fit(
        X_tr[feat_cols], y_tr,
        sample_weight = sw_tr,
        eval_set      = [(X_val[feat_cols], y_val)],
        callbacks     = [
            lgb.early_stopping(stopping_rounds=LGBM_EARLY_STOPPING, verbose=False),
            lgb.log_evaluation(period=-1),
        ],
    )
    y_pred   = model.predict(X_val[feat_cols])
    f1_macro = float(f1_score(y_val, y_pred, average="macro", zero_division=0))
    f1_per   = f1_score(y_val, y_pred, average=None, zero_division=0, labels=[0, 1, 2])
    metrics  = {
        "fold": fold_num,
        "n_train": len(X_tr), "n_val": len(X_val),
        "best_iteration": model.best_iteration_,
        "accuracy":    round(float(accuracy_score(y_val, y_pred)), 4),
        "f1_macro":    round(f1_macro, 4),
        "f1_weighted": round(float(f1_score(y_val, y_pred, average="weighted", zero_division=0)), 4),
        "f1_SHORT": round(float(f1_per[0]), 4),
        "f1_FLAT":  round(float(f1_per[1]), 4),
        "f1_LONG":  round(float(f1_per[2]), 4),
        "confusion_matrix": confusion_matrix(y_val, y_pred, labels=[0, 1, 2]).tolist(),
    }
    logger.info(
        f"[Fold {fold_num}] Acc={metrics['accuracy']:.4f} | "
        f"F1-macro={f1_macro:.4f} | LONG={f1_per[2]:.4f} "
        f"SHORT={f1_per[0]:.4f} FLAT={f1_per[1]:.4f}"
    )
    return model, metrics


# ─── Main ─────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true", help="Pakai semua 20 koin")
    parser.add_argument("--run-id", default=None, help="Custom run ID (default: tanggal)")
    return parser.parse_args()


def main():
    args   = parse_args()
    coins  = ALL_COINS if args.all else TRAINING_COINS
    run_id = args.run_id or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    run_dir = MODEL_DIR / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Run ID: {run_id} | Output: {run_dir}")

    df = load_symbols(coins)
    df, y = encode_labels(df)
    sw    = compute_sample_weights(y)
    feat_cols = get_feature_cols(df)
    logger.info(f"Features: {len(feat_cols)} | Samples: {len(df):,}")

    folds        = build_purged_folds(len(df))
    all_metrics  = []
    best_model, best_f1, best_fold = None, -1.0, -1

    for fold_num, (tr_pos, te_pos) in enumerate(folds, 1):
        model, metrics = train_fold(
            df.iloc[tr_pos], y[tr_pos], sw[tr_pos],
            df.iloc[te_pos], y[te_pos],
            fold_num, feat_cols,
        )
        all_metrics.append(metrics)
        if metrics["f1_macro"] > best_f1:
            best_f1, best_model, best_fold = metrics["f1_macro"], model, fold_num

    # Simpan model
    model_path = run_dir / "lgbm.pkl"
    joblib.dump(best_model, model_path)
    # Juga update symlink ke models/ root untuk inference
    root_model = MODEL_DIR / "lgbm_baseline.pkl"
    joblib.dump(best_model, root_model)
    logger.info(f"Best model (fold {best_fold}, F1={best_f1:.4f}) → {model_path}")

    f1s  = [m["f1_macro"] for m in all_metrics]
    accs = [m["accuracy"] for m in all_metrics]
    cv_summary = {
        "run_id": run_id, "coins": coins,
        "n_folds": N_FOLDS, "purge_gap_bars": PURGE_GAP_BARS,
        "best_fold": best_fold, "best_f1_macro": round(best_f1, 4),
        "mean_f1_macro": round(float(np.mean(f1s)), 4),
        "std_f1_macro":  round(float(np.std(f1s)), 4),
        "mean_accuracy": round(float(np.mean(accs)), 4),
        "lgbm_params": LGBM_PARAMS, "feature_cols": feat_cols,
        "folds": all_metrics,
    }

    cv_path = run_dir / "lgbm_cv_results.json"
    with open(cv_path, "w") as f:
        json.dump(cv_summary, f, indent=2, default=str)
    # Juga simpan di root models/
    with open(MODEL_DIR / "cv_results.json", "w") as f:
        json.dump(cv_summary, f, indent=2, default=str)

    sep = "=" * 60
    print(f"\n{sep}")
    print(f"  LGBM TRAINING SELESAI — {run_id}")
    print(f"{sep}")
    print(f"  Best fold  : {best_fold} (F1-macro={best_f1:.4f})")
    print(f"  Mean F1    : {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")
    print(f"  Mean Acc   : {np.mean(accs):.4f}")
    print(f"  Model      : {model_path}")
    print(f"  CV results : {cv_path}")
    print(f"{sep}\n")

    print(f"  {'Fold':>4}  {'Acc':>7}  {'F1-mac':>7}  {'LONG':>7}  {'SHORT':>7}  {'FLAT':>7}  {'Iter':>6}")
    print("  " + "-" * 52)
    for m in all_metrics:
        print(f"  {m['fold']:>4}  {m['accuracy']:>7.4f}  {m['f1_macro']:>7.4f}  "
              f"{m['f1_LONG']:>7.4f}  {m['f1_SHORT']:>7.4f}  {m['f1_FLAT']:>7.4f}  "
              f"{m['best_iteration']:>6}")


if __name__ == "__main__":
    main()
