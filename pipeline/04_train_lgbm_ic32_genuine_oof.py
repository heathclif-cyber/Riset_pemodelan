"""
pipeline/04_train_lgbm_ic32_genuine_oof.py — ic32_regime_v1 Genuine OOF LGBM

Walk-forward purged CV (8 folds, gap=20) pada swing labels + 33 fitur ic32.
Menyimpan oof_predictions.parquet untuk full-stack OOF sim (08_oof_ic32_full_stack.py).

Tidak menimpa lgbm.pkl produksi — hanya menambah artefak OOF di run dir.

Jalankan:
  python pipeline/04_train_lgbm_ic32_genuine_oof.py
"""
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

from pipeline.shared import build_purged_folds
from core.utils import setup_logger, ensure_utc_index
from config import (
    ALL_COINS, LABEL_DIR, MODEL_DIR, TRAIN_CUTOFF_DATE,
    LGBM_PARAMS, LGBM_EARLY_STOPPING, LGBM_CLASS_WEIGHTS,
    N_FOLDS, PURGE_GAP_BARS, LABEL_MAP,
)

logger = setup_logger("04_ic32_oof")
RUN_ID = "ic32_regime_v1"
RUN_DIR = MODEL_DIR / "runs" / RUN_ID


def load_ic32_features() -> list[str]:
    for path in (
        MODEL_DIR / "feature_cols_ic32_regime.json",
        RUN_DIR / "lgbm_cv_results.json",
    ):
        if not path.exists():
            continue
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        if isinstance(data, dict) and "feature_cols" in data:
            return data["feature_cols"]
    raise FileNotFoundError("ic32 feature list not found")


def load_training_data(coins: list[str], features: list[str]) -> pd.DataFrame:
    frames = []
    for sym in coins:
        path = LABEL_DIR / f"{sym}_features_v3.parquet"
        if not path.exists():
            logger.warning(f"[{sym}] skip — {path.name} missing")
            continue
        df = pd.read_parquet(path)
        df = ensure_utc_index(df).sort_index()
        df = df[df.index < TRAIN_CUTOFF_DATE]

        regime_path = LABEL_DIR / f"{sym}_regime_h1.parquet"
        if regime_path.exists():
            try:
                reg = pd.read_parquet(regime_path)
                if "hmm_regime_enc" in df.columns:
                    df = df.drop(columns=["hmm_regime_enc"])
                df = df.join(reg[["hmm_regime_enc"]], how="left")
                df["hmm_regime_enc"] = df["hmm_regime_enc"].fillna(1).astype("int32")
            except Exception as exc:
                logger.warning(f"[{sym}] regime merge failed: {exc}")

        mask = df["label"].astype(str).isin(LABEL_MAP)
        df = df[mask].copy()
        if len(df) < 100:
            continue
        df["coin"] = sym
        frames.append(df)
        logger.info(f"[{sym}] {len(df):,} bars")

    if not frames:
        raise RuntimeError("No training data — run pipeline/03_engineer.py first")
    return pd.concat(frames).sort_index()


def main():
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    features = load_ic32_features()

    sep = "=" * 70
    print(f"\n{sep}")
    print("  ic32_regime_v1 — GENUINE OOF LGBM")
    print(f"  Period     : 2020-01-01 -> {TRAIN_CUTOFF_DATE.date()}")
    print(f"  Features   : {len(features)} (ic32 swing)")
    print(f"  CV         : {N_FOLDS} folds, purge={PURGE_GAP_BARS} bars")
    print(f"  Output     : {RUN_DIR}")
    print(f"{sep}\n")

    df = load_training_data(ALL_COINS, features)
    avail = [c for c in features if c in df.columns]
    missing = [c for c in features if c not in df.columns]
    if missing:
        logger.warning(f"Missing features ({len(missing)}): {missing[:5]}...")
    if len(avail) != len(features):
        raise RuntimeError(f"Only {len(avail)}/{len(features)} features available")

    y_all = df["label"].map(LABEL_MAP).values.astype(np.int32)
    X_all = df[avail].ffill().fillna(0).values.astype(np.float32)
    n_total = len(df)

    print(f"  Bars   : {n_total:,}")
    print(f"  Coins  : {df['coin'].nunique()}")
    for i, name in enumerate(["SHORT", "FLAT", "LONG"]):
        print(f"  {name:<6}: {(y_all == i).sum():,} ({(y_all == i).mean() * 100:.1f}%)")

    print(f"\nSTAGE 2: Purged walk-forward CV...")
    print("-" * 55)

    oof_probas = np.full((n_total, 3), np.nan, dtype=np.float32)
    folds = build_purged_folds(df.index, N_FOLDS, PURGE_GAP_BARS)
    fold_metrics = []

    for fold_idx, (train_idx, val_idx) in enumerate(folds, 1):
        X_tr, y_tr = X_all[train_idx], y_all[train_idx]
        X_val, y_val = X_all[val_idx], y_all[val_idx]
        sample_w = np.array([LGBM_CLASS_WEIGHTS[int(lbl)] for lbl in y_tr], dtype=np.float32)

        model = lgb.LGBMClassifier(**LGBM_PARAMS)
        model.fit(
            X_tr, y_tr,
            sample_weight=sample_w,
            eval_set=[(X_val, y_val)],
            callbacks=[
                lgb.early_stopping(LGBM_EARLY_STOPPING, verbose=False),
                lgb.log_evaluation(period=-1),
            ],
        )

        fold_probas = model.predict_proba(X_val).astype(np.float32)
        oof_probas[val_idx] = fold_probas

        y_pred = fold_probas.argmax(axis=1)
        f1_macro = float(f1_score(y_val, y_pred, average="macro", zero_division=0))
        f1_per = f1_score(y_val, y_pred, average=None, labels=[0, 1, 2], zero_division=0)
        acc = float(accuracy_score(y_val, y_pred))

        y_train_pred = model.predict(X_tr)
        train_f1 = float(f1_score(y_tr, y_train_pred, average="macro", zero_division=0))

        metrics = {
            "fold": fold_idx,
            "n_train": len(train_idx),
            "n_val": len(val_idx),
            "best_iteration": model.best_iteration_,
            "train_f1_macro": round(train_f1, 4),
            "accuracy": round(acc, 4),
            "f1_macro": round(f1_macro, 4),
            "f1_SHORT": round(float(f1_per[0]), 4),
            "f1_FLAT": round(float(f1_per[1]), 4),
            "f1_LONG": round(float(f1_per[2]), 4),
            "confusion_matrix": confusion_matrix(y_val, y_pred, labels=[0, 1, 2]).tolist(),
        }
        fold_metrics.append(metrics)
        logger.info(
            f"  Fold {fold_idx}: F1={f1_macro:.4f} "
            f"S={f1_per[0]:.4f} F={f1_per[1]:.4f} L={f1_per[2]:.4f} "
            f"| iter={model.best_iteration_}"
        )

    has_oof = ~np.isnan(oof_probas[:, 0])
    f1s = [m["f1_macro"] for m in fold_metrics]
    mean_f1 = float(np.mean(f1s))
    std_f1 = float(np.std(f1s))

    print(f"\n  OOF coverage : {has_oof.sum():,}/{n_total:,} ({has_oof.mean() * 100:.1f}%)")
    print(f"  Mean F1 macro: {mean_f1:.4f} +/- {std_f1:.4f}")

    oof_df = pd.DataFrame({
        "coin": df["coin"].values,
        "p0": oof_probas[:, 0],
        "p1": oof_probas[:, 1],
        "p2": oof_probas[:, 2],
        "has_oof": has_oof,
        "label": y_all.astype(np.int8),
    }, index=df.index)
    oof_path = RUN_DIR / "oof_predictions.parquet"
    oof_df.to_parquet(oof_path)
    print(f"\n  Saved -> {oof_path}")

    oof_meta = {
        "run_id": RUN_ID,
        "created": datetime.now().isoformat(),
        "methodology": "genuine_oof_purged_cv",
        "train_cutoff": str(TRAIN_CUTOFF_DATE.date()),
        "n_folds": N_FOLDS,
        "gap_bars": PURGE_GAP_BARS,
        "feature_cols": avail,
        "n_samples": n_total,
        "oof_coverage": int(has_oof.sum()),
        "mean_f1_macro": round(mean_f1, 4),
        "std_f1_macro": round(std_f1, 4),
        "folds": fold_metrics,
        "note": "Swing labels (not TB). Does not overwrite lgbm.pkl.",
    }
    meta_path = RUN_DIR / "oof_cv_results.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(oof_meta, f, indent=2)
    print(f"  Saved -> {meta_path}")

    print(f"\n{sep}")
    print("  DONE — LGBM OOF ready for full-stack sim")
    print("  Next: python pipeline/08_oof_ic32_full_stack.py")
    print(f"{sep}\n")


if __name__ == "__main__":
    main()