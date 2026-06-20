"""
pipeline/04_train_lgbm_regimes.py — Train Regime-Specific LGBM Models + Global Fallback Model
Menggunakan subset fitur hasil Simons 3-stage feature selection.

Jalankan:
  python pipeline/04_train_lgbm_regimes.py --run-id simons_hybrid_v1
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
from sklearn.metrics import accuracy_score, f1_score

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from pipeline.shared import build_purged_folds
from core.utils import setup_logger
from config import (
    ALL_COINS, LABEL_DIR, MODEL_DIR,
    LGBM_PARAMS, LGBM_EARLY_STOPPING,
    N_FOLDS, PURGE_GAP_BARS,
    LABEL_MAP, NUM_CLASSES,
    LGBM_CLASS_WEIGHTS, TRAIN_CUTOFF_DATE,
)

logger = setup_logger("04_train_lgbm_regimes")

# Feature lists are loaded dynamically from multistage_selected_feats.json in main()
# This avoids hardcoding and ensures features stay in sync with latest feature selection.

LABEL_ORDINAL = {"SHORT": 0, "FLAT": 1, "LONG": 2}

def load_data(coins: list[str]) -> pd.DataFrame:
    """Load data features dan HMM regime dari parquet files.
    
    Menggunakan kolom 'label' bawaan dari parquet (swing labeling oleh 03_engineer.py),
    BUKAN re-generate triple_barrier_labeling() on-the-fly.
    Ini konsisten dengan ic32_regime_v1 (04_train_lgbm.py).
    """
    frames = []
    for sym in coins:
        fpath = LABEL_DIR / f"{sym}_features_v3.parquet"
        regpath = LABEL_DIR / f"{sym}_regime_h1.parquet"
        if not fpath.exists() or not regpath.exists():
            continue
            
        df = pd.read_parquet(fpath)
        reg = pd.read_parquet(regpath)
        
        # Drop columns if they exist to prevent suffix issues
        for col in ["hmm_regime_enc", "hmm_regime"]:
            if col in df.columns:
                df = df.drop(columns=[col])
                
        df = df.join(reg[["hmm_regime_enc", "hmm_regime"]], how="left")
        df["hmm_regime_enc"] = df["hmm_regime_enc"].fillna(1).astype("int32")
        
        df = df[df.index < TRAIN_CUTOFF_DATE]
        if df.empty:
            continue
        
        if "label" not in df.columns:
            logger.warning(f"{sym}: no 'label' column, skip")
            continue
            
        df["coin"] = sym
        frames.append(df)
        
    if not frames:
        raise FileNotFoundError("Tidak ada data ditemukan untuk training!")
        
    combined = pd.concat(frames).sort_index()
    combined = combined[combined["label"].isin(LABEL_ORDINAL)].copy()
    combined["label_ord"] = combined["label"].map(LABEL_ORDINAL).astype(np.int32)
    return combined

def train_sp_model(df: pd.DataFrame, regime_enc: int | None, regime_name: str, feature_cols: list, run_dir: Path) -> lgb.LGBMClassifier:
    """Melatih LightGBM untuk regime tertentu atau global.
    
    Catatan: Melatih model spesifik regime hanya pada bar data regime yang sesuai.
    """
    X_df = df.copy()
    if regime_enc is not None:
        X_df = X_df[X_df["hmm_regime_enc"] == regime_enc].copy()
        
    if len(X_df) < 100:
        logger.warning(f"SKIP {regime_name} — Sampel terlalu sedikit ({len(X_df)})")
        return None

    # Gunakan integer index unik untuk CV purging agar tidak ada duplikasi index datetime
    X_df = X_df.reset_index(drop=True)
    X = X_df[feature_cols].copy()
    y = X_df["label_ord"].astype(np.int32)

    # Isi missing values
    for col in X.columns:
        X[col] = X[col].ffill().fillna(X[col].median()).fillna(0.0)

    logger.info(f"Training Model: {regime_name} (ALL TIMES) — Samples: {len(X):,} × Features: {len(X.columns)}")
    
    folds = build_purged_folds(X_df.index, n_folds=N_FOLDS, purge=PURGE_GAP_BARS)
    
    all_metrics = []
    for fold, (train_idx, val_idx) in enumerate(folds, 1):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # Cost-sensitive weights
        sample_w = np.array([LGBM_CLASS_WEIGHTS[int(label)] for label in y_tr], dtype=np.float32)
        
        model = lgb.LGBMClassifier(**LGBM_PARAMS)
        model.fit(
            X_tr, y_tr,
            sample_weight=sample_w,
            eval_set=[(X_val, y_val)],
            callbacks=[
                lgb.early_stopping(LGBM_EARLY_STOPPING, verbose=False),
                lgb.log_evaluation(period=-1)
            ]
        )
        
        y_pred = model.predict(X_val)
        f1_macro = float(f1_score(y_val, y_pred, average="macro", zero_division=0))
        acc = float(accuracy_score(y_val, y_pred))
        
        metrics = {
            "fold": fold,
            "best_iteration": model.best_iteration_,
            "f1_macro": round(f1_macro, 4),
            "accuracy": round(acc, 4)
        }
        all_metrics.append(metrics)
        logger.info(f"  Fold {fold}: Best Iter={model.best_iteration_} | Val F1={f1_macro:.4f} | Acc={acc:.4f}")
        
    avg_best_iter = int(np.mean([m["best_iteration"] for m in all_metrics]))
    logger.info(f"CV Complete. Avg best_iteration: {avg_best_iter}")
    
    # Retrain final pada 100% training data
    final_params = LGBM_PARAMS.copy()
    final_params["n_estimators"] = avg_best_iter
    
    final_model = lgb.LGBMClassifier(**final_params)
    full_sample_w = np.array([LGBM_CLASS_WEIGHTS[int(label)] for label in y], dtype=np.float32)
    final_model.fit(X, y, sample_weight=full_sample_w)
    
    # Simpan model
    model_fname = f"lgbm.pkl" if regime_name == "GLOBAL" else f"lgbm_regime_{regime_name}.pkl"
    joblib.dump(final_model, run_dir / model_fname)
    
    # Copy ke root models/ directory
    root_fname = "lgbm_baseline.pkl" if regime_name == "GLOBAL" else model_fname
    joblib.dump(final_model, MODEL_DIR / root_fname)
    
    logger.info(f"Model saved -> {run_dir / model_fname} and copied to root -> {MODEL_DIR / root_fname}")
    return final_model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    
    run_dir = MODEL_DIR / "runs" / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Load feature lists from multi-stage feature selection results
    feats_path = run_dir / "multistage_selected_feats.json"
    if not feats_path.exists():
        raise FileNotFoundError(
            f"Feature selection belum dijalankan! File tidak ditemukan: {feats_path}\n"
            f"Jalankan dulu: python pipeline/03c_feature_selection_multistage.py"
        )
    
    with open(feats_path) as f:
        feats = json.load(f)
    
    UP_FEATS = feats["up_feats"]
    DOWN_FEATS = feats["down_feats"]
    RANGING_LOW_FEATS = feats["ranging_low_feats"]
    RANGING_HIGH_FEATS = feats["ranging_high_feats"]
    GLOBAL_FEATS = feats.get("global_feats", [])
    UNION_FEATS = list(dict.fromkeys(UP_FEATS + DOWN_FEATS + RANGING_LOW_FEATS + RANGING_HIGH_FEATS))
    
    # Use global_feats if available, otherwise use union
    fallback_feats = GLOBAL_FEATS if GLOBAL_FEATS else UNION_FEATS
    
    logger.info(f"Start Regime-Specific LGBM Training. Run ID: {args.run_id} | Dir: {run_dir}")
    logger.info(f"Feature counts: UP={len(UP_FEATS)}, DOWN={len(DOWN_FEATS)}, "
                f"RANGE_LOW={len(RANGING_LOW_FEATS)}, RANGE_HIGH={len(RANGING_HIGH_FEATS)}, "
                f"GLOBAL={len(fallback_feats)}")
    
    df = load_data(ALL_COINS)
    logger.info(f"Loaded training dataset: {len(df):,} total samples.")
    
    # 1. Train TRENDING_UP (HMM State 3)
    logger.info("\n=== STEP 1: TRENDING_UP MODEL ===")
    train_sp_model(df, 3, "TRENDING_UP", UP_FEATS, run_dir)
    
    # 2. Train TRENDING_DOWN (HMM State 0)
    logger.info("\n=== STEP 2: TRENDING_DOWN MODEL ===")
    train_sp_model(df, 0, "TRENDING_DOWN", DOWN_FEATS, run_dir)
    
    # 3. Train RANGING_LOW_VOL (HMM State 1)
    logger.info("\n=== STEP 3: RANGING_LOW_VOL MODEL ===")
    train_sp_model(df, 1, "RANGING_LOW_VOL", RANGING_LOW_FEATS, run_dir)
    
    # 4. Train RANGING_HIGH_VOL (HMM State 2)
    logger.info("\n=== STEP 4: RANGING_HIGH_VOL MODEL ===")
    train_sp_model(df, 2, "RANGING_HIGH_VOL", RANGING_HIGH_FEATS, run_dir)
    
    # 5. Train GLOBAL Fallback Model
    logger.info("\n=== STEP 5: GLOBAL FALLBACK MODEL ===")
    train_sp_model(df, None, "GLOBAL", fallback_feats, run_dir)
    
    # Simpan metadata training
    meta = {
        "run_id": args.run_id,
        "timestamp": str(datetime.now()),
        "up_feats": UP_FEATS,
        "down_feats": DOWN_FEATS,
        "ranging_low_feats": RANGING_LOW_FEATS,
        "ranging_high_feats": RANGING_HIGH_FEATS,
        "global_feats": fallback_feats,
        "union_feats": UNION_FEATS,
        "lgbm_params": LGBM_PARAMS,
    }
    with open(run_dir / "lgbm_regimes_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)
        
    print(f"\n{'='*70}")
    print(f"  LGBM REGIME-SPECIFIC TRAINING COMPLETE — {args.run_id}")
    print(f"  Models saved to {run_dir} and copied to root models/")
    print(f"{'='*70}\n")

if __name__ == "__main__":
    main()
