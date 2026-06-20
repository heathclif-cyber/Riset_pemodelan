"""
pipeline/04_train_lgbm_rolling_v1.py — Expanding vs Rolling Walk-Forward CV (apple-to-apple)

Membandingkan dua metode CV dengan SEMUA parameter, fitur, dan labeling yang SAMA:
  1. Expanding (existing) — baseline
  2. Rolling (window=3 splits, ~25 bulan)

Setup:
  - Fitur  : 33 dari feature_cols_v2.json (ic32 swing features + hmm_regime_enc)
  - Labels : swing H4 (`label` column dari 03_engineer.py)
  - Params : LGBM_PARAMS dari config.py
  - Purge  : PURGE_GAP_BARS=20 (ic32 standard)
  - Folds  : N_FOLDS=8

Jalankan: python pipeline/04_train_lgbm_rolling_v1.py
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

from pipeline.shared import build_purged_folds, build_rolling_folds
from core.utils import setup_logger, ensure_utc_index
from config import (
    ALL_COINS, LABEL_DIR, MODEL_DIR, TRAIN_CUTOFF_DATE,
    LGBM_PARAMS, LGBM_EARLY_STOPPING,
    N_FOLDS, PURGE_GAP_BARS, LABEL_MAP,
)

logger = setup_logger("04_train_lgbm_rolling")

RUN_NAME        = "tb_lgbm_cv_comparison"
FEAT_SOURCE     = MODEL_DIR / "feature_cols_v2.json"
ROLLING_WINDOW  = 3  # jumlah split untuk rolling window (~25 bulan)

EARLY_STOP = LGBM_EARLY_STOPPING  # 50


def load_training_data(coins: list[str], features: list[str]) -> pd.DataFrame:
    """Load labeled data dari 03_engineer.py output, merge hmm_regime_enc jika ada."""
    frames = []
    for sym in coins:
        path = LABEL_DIR / f"{sym}_features_v3.parquet"
        if not path.exists():
            logger.warning(f"[{sym}] skip — {path.name} missing")
            continue
        df = pd.read_parquet(path)
        df = ensure_utc_index(df).sort_index()
        df = df[df.index < TRAIN_CUTOFF_DATE]

        # Merge regime labels jika tersedia
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

        # Filter hanya swing labels (SHORT/FLAT/LONG)
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


def run_cv(df: pd.DataFrame, features: list[str], cv_mode: str, run_dir: Path) -> dict:
    """
    Run walk-forward CV dengan mode `expanding` atau `rolling`.
    Returns dict dengan cv_results.
    """
    avail = [c for c in features if c in df.columns]
    missing = [c for c in features if c not in df.columns]
    if missing:
        logger.warning(f"Missing features ({len(missing)}): {missing}")

    y_all   = df["label"].map(LABEL_MAP).values.astype(np.int32)
    X_all   = df[avail].ffill().fillna(0).values.astype(np.float32)
    n_total = len(df)

    # Build folds
    if cv_mode == "expanding":
        folds = build_purged_folds(df.index, N_FOLDS, PURGE_GAP_BARS)
        cv_label = "Expanding"
    else:
        folds = build_rolling_folds(df.index, N_FOLDS, PURGE_GAP_BARS, ROLLING_WINDOW)
        cv_label = f"Rolling (w={ROLLING_WINDOW})"

    print(f"\n  [{cv_label}] {N_FOLDS} folds, purge={PURGE_GAP_BARS} bars, {n_total:,} bars, {len(avail)} features")

    oof_probas  = np.full((n_total, 3), np.nan, dtype=np.float32)
    fold_metrics = []
    fold_iters   = []

    for fold_idx, (train_idx, val_idx) in enumerate(folds, 1):
        X_tr, y_tr = X_all[train_idx], y_all[train_idx]
        X_val, y_val = X_all[val_idx], y_all[val_idx]

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
            "n_train":   int(len(train_idx)),
            "n_val":     int(len(val_idx)),
            "best_iter": model.best_iteration_,
            "f1_macro":  round(f1_macro, 4),
            "f1_SHORT":  round(float(f1_per[0]), 4),
            "f1_FLAT":   round(float(f1_per[1]), 4),
            "f1_LONG":   round(float(f1_per[2]), 4),
        })
        logger.info(
            f"  [{cv_label}] Fold {fold_idx}: "
            f"F1={f1_macro:.4f} "
            f"S={f1_per[0]:.4f} F={f1_per[1]:.4f} L={f1_per[2]:.4f} "
            f"| train={len(train_idx):,} val={len(val_idx):,} iter={model.best_iteration_}"
        )

    has_oof = ~np.isnan(oof_probas[:, 0])
    mean_f1 = float(np.mean([m["f1_macro"] for m in fold_metrics]))
    std_f1  = float(np.std([m["f1_macro"]  for m in fold_metrics]))
    avg_iter = int(np.mean(fold_iters))

    print(f"  [{cv_label}] OOF coverage: {has_oof.sum():,}/{n_total:,} ({has_oof.mean()*100:.1f}%)")
    print(f"  [{cv_label}] Mean F1: {mean_f1:.4f} +/- {std_f1:.4f}")

    # Save OOF predictions
    oof_df = pd.DataFrame({
        "coin":      df["coin"].values,
        "p0":        oof_probas[:, 0],
        "p1":        oof_probas[:, 1],
        "p2":        oof_probas[:, 2],
        "has_oof":   has_oof,
        "label":     y_all.astype(np.int8),
    }, index=df.index)
    oof_path = run_dir / f"oof_predictions_{cv_mode}.parquet"
    oof_df.to_parquet(oof_path)
    print(f"  [{cv_label}] Saved: {oof_path.name}")

    return {
        "cv_mode":       cv_mode,
        "window_splits": ROLLING_WINDOW if cv_mode == "rolling" else None,
        "n_features":    len(avail),
        "features":      avail,
        "n_samples":     n_total,
        "n_folds":       N_FOLDS,
        "purge_bars":    PURGE_GAP_BARS,
        "mean_f1_macro": round(mean_f1, 4),
        "std_f1_macro":  round(std_f1, 4),
        "avg_iterations": avg_iter,
        "folds":         fold_metrics,
    }


def main():
    run_dir = MODEL_DIR / "runs" / RUN_NAME
    run_dir.mkdir(parents=True, exist_ok=True)

    # Load feature list
    if not FEAT_SOURCE.exists():
        raise FileNotFoundError(f"Feature list not found: {FEAT_SOURCE}")
    with open(FEAT_SOURCE, encoding="utf-8") as f:
        FEATURES = json.load(f)

    sep = "=" * 70
    print(f"\n{sep}")
    print(f"  CV COMPARISON: Expanding vs Rolling (apple-to-apple)")
    print(f"  Features   : {len(FEATURES)} (feature_cols_v2.json)")
    print(f"  Labels     : swing H4 (`label` column)")
    print(f"  Params     : LGBM_PARAMS from config.py")
    print(f"  Train up to: {TRAIN_CUTOFF_DATE.date()}")
    print(f"  Purge      : {PURGE_GAP_BARS} bars")
    print(f"  Output     : {run_dir}")
    print(f"{sep}")

    # ── Load Data ──────────────────────────────────────────────────────────
    print("\nSTAGE 1: Loading labeled data (swing H4)...")
    print("-" * 55)
    df = load_training_data(ALL_COINS, FEATURES)

    dist = df["label"].value_counts()
    n = len(df)
    print(f"  Total bars : {n:,}")
    print(f"  Koin       : {df['coin'].nunique()}")
    for label_name in ["SHORT", "FLAT", "LONG"]:
        idx = LABEL_MAP[label_name]
        print(f"  {label_name:<6}: {dist.get(label_name, 0):,} ({dist.get(label_name, 0)/n*100:.1f}%)")

    # ── Run Both CV Methods ────────────────────────────────────────────────
    results = {}

    print(f"\n{'='*70}")
    print("  RUN 1: EXPANDING CV (baseline)")
    print(f"{'='*70}")
    results["expanding"] = run_cv(df, FEATURES, "expanding", run_dir)

    print(f"\n{'='*70}")
    print("  RUN 2: ROLLING CV")
    print(f"{'='*70}")
    results["rolling"]   = run_cv(df, FEATURES, "rolling", run_dir)

    # ── Comparison ─────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("  COMPARISON")
    print(f"{'='*70}")

    exp = results["expanding"]
    rol = results["rolling"]

    print(f"\n  {'Method':<12} {'Mean F1':>8} {'Std F1':>8} {'F1 Range':>10} {'Train Range':>16}")
    print(f"  {'-'*12} {'-'*8} {'-'*8} {'-'*10} {'-'*16}")
    e_min = min(f["f1_macro"] for f in exp["folds"])
    e_max = max(f["f1_macro"] for f in exp["folds"])
    r_min = min(f["f1_macro"] for f in rol["folds"])
    r_max = max(f["f1_macro"] for f in rol["folds"])
    e_tmin = min(f["n_train"] for f in exp["folds"])
    e_tmax = max(f["n_train"] for f in exp["folds"])
    r_tmin = min(f["n_train"] for f in rol["folds"])
    r_tmax = max(f["n_train"] for f in rol["folds"])

    print(f"  {'Expanding':<12} {exp['mean_f1_macro']:>8.4f} {exp['std_f1_macro']:>8.4f} "
          f"{e_min:.4f}-{e_max:.4f} {f'{e_tmin:,}-{e_tmax:,}':>16}")
    print(f"  {'Rolling':<12} {rol['mean_f1_macro']:>8.4f} {rol['std_f1_macro']:>8.4f} "
          f"{r_min:.4f}-{r_max:.4f} {f'{r_tmin:,}-{r_tmax:,}':>16}")
    print(f"  {'Delta':<12} {rol['mean_f1_macro']-exp['mean_f1_macro']:>+8.4f} "
          f"{rol['std_f1_macro']-exp['std_f1_macro']:>+8.4f}")

    print(f"\n  Per-Fold Comparison:")
    print(f"  {'Fold':>5} {'Exp F1':>8} {'Rol F1':>8} {'Delta':>8}  "
          f"{'Exp Train':>10} {'Rol Train':>10}")
    print(f"  {'-'*5} {'-'*8} {'-'*8} {'-'*8}  {'-'*10} {'-'*10}")
    for i in range(len(exp["folds"])):
        ef = exp["folds"][i]
        rf = rol["folds"][i]
        delta = rf["f1_macro"] - ef["f1_macro"]
        print(f"  {ef['fold']:>5} {ef['f1_macro']:>8.4f} {rf['f1_macro']:>8.4f} "
              f"{delta:>+8.4f}  {ef['n_train']:>10,} {rf['n_train']:>10,}")

    # ── Save ───────────────────────────────────────────────────────────────
    comparison = {
        "run_name":     RUN_NAME,
        "created":      datetime.now().isoformat(),
        "train_cutoff": str(TRAIN_CUTOFF_DATE.date()),
        "features_source": str(FEAT_SOURCE),
        "features":     FEATURES,
        "n_features_available": exp["n_features"],
        "lgbm_params":  {k: v for k, v in LGBM_PARAMS.items()
                         if k not in ("verbose", "n_jobs", "random_state")},
        "early_stop":   EARLY_STOP,
        "expanding":    exp,
        "rolling":      rol,
    }
    out_path = run_dir / "cv_comparison.json"
    with open(out_path, "w") as f:
        json.dump(comparison, f, indent=2)
    print(f"\n  Saved: {out_path}")

    print(f"\n{'='*70}")
    print(f"  DONE — Expanding vs Rolling comparison")
    print(f"  Kesimpulan awal: lihat std F1 dan F1 range")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
