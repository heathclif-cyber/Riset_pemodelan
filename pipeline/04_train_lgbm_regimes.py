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
from core.features import triple_barrier_labeling
from core.utils import setup_logger
from config import (
    ALL_COINS, LABEL_DIR, MODEL_DIR,
    LGBM_PARAMS, LGBM_EARLY_STOPPING,
    N_FOLDS, PURGE_GAP_BARS,
    LABEL_MAP, NUM_CLASSES,
    LGBM_CLASS_WEIGHTS, TRAIN_CUTOFF_DATE,
)

logger = setup_logger("04_train_lgbm_regimes")

# Simons 3-stage selected features
UP_FEATS = [
    "ofi_h4_delta", "cvd_slope_h4", "cvd_div_h4", "ema_50_h1", "stochrsi_d", 
    "VAH", "atr_zscore_20d", "whale_retail_divergence", "ofi_acceleration", 
    "vol_accel_3h", "vol_spike_zscore", "dist_from_8h_high", "log_ret_5", 
    "ema_7_h1", "ema_21_h1", "atr_14_h4", "cvd", "open_interest", 
    "vol_price_confirm", "Sell_Liq"
]

DOWN_FEATS = [
    "ofi_h4_delta", "cvd_slope_h4", "log_ret_20", "cvd_div_h4", "cvd", 
    "ema_21_slope_h4", "cvd_momentum_adv", "stochrsi_d", "price_in_range", 
    "ema_21_h1", "btc_dominance", "ofi_acceleration", "log_ret_5", 
    "vol_price_confirm", "trend_accel_4h", "dist_swing_low", "VAL", 
    "wyckoff_phase"
]

RANGING_FEATS = [
    "cvd_slope_h4", "ofi_h4_delta", "log_ret_20", "cvd_div_h4", "cvd", 
    "ofi_z_score", "atr_percentile_h1", "stochrsi_d", "whale_retail_divergence", 
    "log_ret_5", "cvd_momentum_adv", "atr_zscore_20d", "ema_200_h1", 
    "buy_volume", "VAL", "dist_swing_high", "atr_percent_h4"
]

# Union of all selected features
UNION_FEATS = list(dict.fromkeys(UP_FEATS + DOWN_FEATS + RANGING_FEATS))

LABEL_ORDINAL = {"SHORT": 0, "FLAT": 1, "LONG": 2}

def load_data(coins: list[str]) -> pd.DataFrame:
    """Load data features dan HMM regime dari parquet files."""
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
            
        # Generate triple barrier labels on the fly
        tb_labels = triple_barrier_labeling(
            close=df["close"],
            high=df["high"],
            low=df["low"],
            atr_base=df["atr_14_h1"],
            tp_atr_mult=2.0,
            sl_atr_mult=1.5,
            max_hold=36
        )
        df["trend_label"] = tb_labels
        df["coin"] = sym
        frames.append(df)
        
    if not frames:
        raise FileNotFoundError("Tidak ada data ditemukan untuk training!")
        
    combined = pd.concat(frames).sort_index()
    combined = combined[combined["trend_label"].isin(LABEL_ORDINAL)].copy()
    combined["label_ord"] = combined["trend_label"].map(LABEL_ORDINAL).astype(np.int32)
    return combined

def train_sp_model(df: pd.DataFrame, regime_enc: int | None, regime_name: str, feature_cols: list, run_dir: Path) -> lgb.LGBMClassifier:
    """Melatih LightGBM untuk regime tertentu atau global."""
    if regime_enc is not None:
        X_df = df[df["hmm_regime_enc"] == regime_enc].copy()
    else:
        X_df = df.copy()
        
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

    logger.info(f"Training Model: {regime_name} — Samples: {len(X):,} × Features: {len(X.columns)}")
    
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
    
    logger.info(f"Start Regime-Specific LGBM Training. Run ID: {args.run_id} | Dir: {run_dir}")
    
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
    train_sp_model(df, 1, "RANGING_LOW_VOL", RANGING_FEATS, run_dir)
    
    # 4. Train RANGING_HIGH_VOL (HMM State 2)
    logger.info("\n=== STEP 4: RANGING_HIGH_VOL MODEL ===")
    train_sp_model(df, 2, "RANGING_HIGH_VOL", RANGING_FEATS, run_dir)
    
    # 5. Train GLOBAL Fallback Model
    logger.info("\n=== STEP 5: GLOBAL FALLBACK MODEL ===")
    train_sp_model(df, None, "GLOBAL", UNION_FEATS, run_dir)
    
    # Simpan metadata training
    meta = {
        "run_id": args.run_id,
        "timestamp": str(datetime.now()),
        "up_feats": UP_FEATS,
        "down_feats": DOWN_FEATS,
        "ranging_feats": RANGING_FEATS,
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
