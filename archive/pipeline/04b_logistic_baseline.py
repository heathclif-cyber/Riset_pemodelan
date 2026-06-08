"""
pipeline/04b_logistic_baseline.py — Logistic Regression Baseline (Simon Step 4)
Validasi: apakah sinyal bisa ditangkap model linear?

IC dari model linear = batas atas teoritis sinyal.
  F1 >> 0.333 (random) → ada sinyal yang bisa dipelajari → lanjut ke LGBM
  F1 ≈ 0.333           → sinyal tidak ada → arsitektur kompleks tidak menolong

Setup identik dengan LGBM: walk-forward CV 8 folds, purge=20, class weights sama.
Perbedaan: StandardScaler per fold (LogReg sensitif skala, LGBM tidak).

Usage:
    python pipeline/04b_logistic_baseline.py
    python pipeline/04b_logistic_baseline.py --feature-cols models/feature_cols_ic32.json
    python pipeline/04b_logistic_baseline.py --compare-with ic32_lgbm_v1
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent.parent))
from pipeline.shared import build_purged_folds
from config import (
    TRAINING_COINS, LABEL_DIR, MODEL_DIR,
    LABEL_MAP, LGBM_CLASS_WEIGHTS,
    N_FOLDS, PURGE_GAP_BARS, TRAIN_CUTOFF_DATE,
)
from core.utils import setup_logger

logger = setup_logger("04b_logistic_baseline")

NON_FEATURE_COLS = {"label", "h4_swing_high", "h4_swing_low", "hmm_regime", "hmm_regime_enc"}


def load_data(feature_cols: list) -> tuple:
    frames = []
    for sym in TRAINING_COINS:
        path = LABEL_DIR / f"{sym}_features_v3.parquet"
        if not path.exists():
            continue
        df = pd.read_parquet(path)
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index, utc=True)
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        df = df[df.index < TRAIN_CUTOFF_DATE]
        frames.append(df)

    combined = pd.concat(frames).sort_index()
    mask = combined["label"].isin(LABEL_MAP)
    combined = combined[mask].copy()
    y = combined["label"].map(LABEL_MAP).astype(np.int32)

    avail = [c for c in feature_cols if c in combined.columns]
    missing = [c for c in feature_cols if c not in combined.columns]
    if missing:
        logger.warning(f"Missing features: {missing}")

    X = combined[avail].copy()
    # fill NaN dengan median kolom (LogReg tidak toleran NaN)
    for col in X.columns:
        if X[col].isnull().any():
            X[col] = X[col].fillna(X[col].median())

    logger.info(f"Loaded {len(X):,} rows x {len(avail)} features")
    return X, y


def walk_forward_cv(X: pd.DataFrame, y: pd.Series) -> list:
    logger.info(f"Walk-Forward CV (n_splits={N_FOLDS}, purge={PURGE_GAP_BARS} bars)...")

    # class weights dict untuk LogReg
    class_weight = {0: LGBM_CLASS_WEIGHTS[0], 1: LGBM_CLASS_WEIGHTS[1], 2: LGBM_CLASS_WEIGHTS[2]}

    folds = build_purged_folds(X.index, n_folds=N_FOLDS, purge=PURGE_GAP_BARS)
    results = []

    for fold, (train_idx, val_idx) in enumerate(folds, 1):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        # scale per fold — fit pada train saja, transform keduanya
        scaler = StandardScaler()
        X_tr_s  = scaler.fit_transform(X_tr)
        X_val_s = scaler.transform(X_val)

        model = LogisticRegression(
            solver="lbfgs",
            max_iter=1000,
            C=1.0,
            class_weight=class_weight,
            random_state=42,
            n_jobs=-1,
        )
        model.fit(X_tr_s, y_tr)

        y_pred      = model.predict(X_val_s)
        y_pred_tr   = model.predict(X_tr_s)

        f1_val   = float(f1_score(y_val, y_pred, average="macro", zero_division=0))
        f1_train = float(f1_score(y_tr, y_pred_tr, average="macro", zero_division=0))
        f1_per   = f1_score(y_val, y_pred, average=None, zero_division=0, labels=[0, 1, 2])
        acc      = float(accuracy_score(y_val, y_pred))

        logger.info(
            f"  Fold {fold}: Train F1={f1_train:.4f} | Val F1={f1_val:.4f} | Gap={f1_train-f1_val:+.4f} | "
            f"Acc={acc:.4f} | LONG={f1_per[2]:.4f} SHORT={f1_per[0]:.4f} FLAT={f1_per[1]:.4f}"
        )

        results.append({
            "fold":       fold,
            "n_train":    len(X_tr),
            "n_val":      len(X_val),
            "f1_macro":   round(f1_val, 4),
            "f1_train":   round(f1_train, 4),
            "accuracy":   round(acc, 4),
            "f1_LONG":    round(float(f1_per[2]), 4),
            "f1_SHORT":   round(float(f1_per[0]), 4),
            "f1_FLAT":    round(float(f1_per[1]), 4),
        })

    return results


def compare_with_lgbm(lgbm_run_id: str, logreg_results: list):
    lgbm_path = MODEL_DIR / "runs" / lgbm_run_id / "lgbm_cv_results.json"
    if not lgbm_path.exists():
        logger.warning(f"LGBM results tidak ditemukan: {lgbm_path}")
        return

    lgbm = json.load(open(lgbm_path))
    lgbm_folds = lgbm["folds"]

    print(f"\n{'='*70}")
    print(f"  PERBANDINGAN: Logistic Regression vs LGBM ({lgbm_run_id})")
    print(f"{'='*70}")
    print(f"  {'Fold':>4}  {'LogReg F1':>10}  {'LGBM F1':>9}  {'Delta':>8}  {'LogReg Acc':>11}  {'LGBM Acc':>9}")
    print("  " + "-" * 60)

    deltas = []
    for lr, lgbm_f in zip(logreg_results, lgbm_folds):
        delta = lgbm_f["f1_macro"] - lr["f1_macro"]
        deltas.append(delta)
        print(f"  {lr['fold']:>4}  {lr['f1_macro']:>10.4f}  {lgbm_f['f1_macro']:>9.4f}  {delta:>+8.4f}  "
              f"{lr['accuracy']:>11.4f}  {lgbm_f['accuracy']:>9.4f}")

    lr_mean  = np.mean([r["f1_macro"] for r in logreg_results])
    lgbm_mean = np.mean([r["f1_macro"] for r in lgbm_folds])
    print("  " + "-" * 60)
    print(f"  {'Mean':>4}  {lr_mean:>10.4f}  {lgbm_mean:>9.4f}  {lgbm_mean-lr_mean:>+8.4f}")

    print(f"\n  Interpretasi:")
    if lr_mean > 0.40:
        print(f"  [OK] LogReg F1={lr_mean:.4f} >> random (0.333) — sinyal nyata dan linear")
    elif lr_mean > 0.35:
        print(f"  [OK] LogReg F1={lr_mean:.4f} > random — sinyal ada tapi lemah secara linear")
    else:
        print(f"  [!] LogReg F1={lr_mean:.4f} ~ random — sinyal tidak ada!")

    gain = lgbm_mean - lr_mean
    if gain > 0.05:
        print(f"  [OK] LGBM gain +{gain:.4f} — non-linearitas LGBM memberikan nilai tambah signifikan")
    elif gain > 0.01:
        print(f"  [OK] LGBM gain +{gain:.4f} — ada nilai tambah non-linear tapi moderat")
    else:
        print(f"  [?] LGBM gain +{gain:.4f} — hampir sama dengan LogReg, fitur mungkin sudah sangat linear")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-cols", default="models/feature_cols_ic32.json")
    parser.add_argument("--compare-with",  default="ic32_lgbm_v1",
                        help="Run ID LGBM untuk perbandingan (default: ic32_lgbm_v1)")
    parser.add_argument("--run-id", default=None)
    args = parser.parse_args()

    run_id = args.run_id or f"logreg_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    with open(args.feature_cols) as f:
        feature_cols = json.load(f)
    logger.info(f"Feature cols: {len(feature_cols)} fitur ({args.feature_cols})")

    X, y = load_data(feature_cols)
    results = walk_forward_cv(X, y)

    f1s  = [r["f1_macro"] for r in results]
    accs = [r["accuracy"] for r in results]

    print(f"\n{'='*60}")
    print(f"  LOGISTIC REGRESSION BASELINE — {run_id}")
    print(f"{'='*60}")
    print(f"  Random baseline (3-class): F1 = 0.333")
    print(f"  Mean F1    : {np.mean(f1s):.4f} +- {np.std(f1s):.4f}")
    print(f"  Mean Acc   : {np.mean(accs):.4f}")
    print(f"  Best fold  : {max(results, key=lambda r: r['f1_macro'])['fold']} "
          f"(F1={max(f1s):.4f})")
    print(f"\n  {'Fold':>4}  {'Acc':>7}  {'F1-mac':>7}  {'LONG':>7}  {'SHORT':>7}  {'FLAT':>7}")
    print("  " + "-" * 48)
    for r in results:
        print(f"  {r['fold']:>4}  {r['accuracy']:>7.4f}  {r['f1_macro']:>7.4f}  "
              f"{r['f1_LONG']:>7.4f}  {r['f1_SHORT']:>7.4f}  {r['f1_FLAT']:>7.4f}")

    # per-class mean
    print(f"\n  Mean per class:")
    print(f"    LONG  : {np.mean([r['f1_LONG']  for r in results]):.4f}")
    print(f"    SHORT : {np.mean([r['f1_SHORT'] for r in results]):.4f}")
    print(f"    FLAT  : {np.mean([r['f1_FLAT']  for r in results]):.4f}")

    if args.compare_with:
        compare_with_lgbm(args.compare_with, results)

    # simpan hasil
    output = {
        "run_id": run_id,
        "feature_cols_path": args.feature_cols,
        "n_features": len(feature_cols),
        "random_baseline_f1": 0.333,
        "mean_f1_macro": round(float(np.mean(f1s)), 4),
        "std_f1_macro":  round(float(np.std(f1s)), 4),
        "mean_accuracy": round(float(np.mean(accs)), 4),
        "mean_f1_LONG":  round(float(np.mean([r["f1_LONG"]  for r in results])), 4),
        "mean_f1_SHORT": round(float(np.mean([r["f1_SHORT"] for r in results])), 4),
        "mean_f1_FLAT":  round(float(np.mean([r["f1_FLAT"]  for r in results])), 4),
        "folds": results,
    }
    out_path = MODEL_DIR / "runs" / "ic32_lgbm_v1" / "logreg_baseline.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    logger.info(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
