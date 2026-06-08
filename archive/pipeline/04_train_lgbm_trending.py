"""
pipeline/04_train_lgbm_trending.py — Train Regime-Specific LGBM Trending Models
Menggunakan Opsi A (KEEP + WEAK dengan Marginal IC >= 0.015).

Jalankan:
  python pipeline/04_train_lgbm_trending.py --run-id lgbm_trending_v1
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

from pipeline.shared import build_purged_folds
from core.features import triple_barrier_labeling
from core.utils import setup_logger
from config import (
    TRAINING_COINS, ALL_COINS, LABEL_DIR, MODEL_DIR,
    LGBM_PARAMS, LGBM_EARLY_STOPPING,
    N_FOLDS, PURGE_GAP_BARS,
    LABEL_MAP, NUM_CLASSES,
    LGBM_CLASS_WEIGHTS, TRAIN_CUTOFF_DATE,
)

logger = setup_logger("04_train_lgbm_trending")

# Opsi A selected features
UP_FEATS = [
    # KEEP
    "ofi_h4_delta", "cvd_slope_h4", "funding_rate", "wyckoff_phase", "atr_zscore_20d",
    # WEAK with abs(marginal_ic) >= 0.015
    "stochrsi_d", "dist_liq_20x_short", "cvd_div_h4", "vol_spike_zscore",
    "price_in_range", "ema_7_h1", "VAL", "atr_14_h1", "dow_cos", "VAH",
    "cvd_momentum_adv", "sell_volume", "whale_retail_divergence", "h4_trend",
    "log_ret_20", "price_accel_1h", "rsi_slope_h4", "ofi_acceleration"
]

DOWN_FEATS = [
    # KEEP
    "cvd_slope_h4", "ofi_h4_delta", "wyckoff_phase", "stochrsi_d",
    "ema_21_slope_h4", "trend_accel_4h", "PDH", "ema_50_h1",
    # WEAK with abs(marginal_ic) >= 0.015
    "cvd_div_h4", "price_in_range", "h4_trend", "cvd_momentum_adv",
    "swing_momentum", "PWH", "cvd", "dow_sin", "whale_retail_divergence",
    "log_ret_20", "atr_percentile_h1", "ofi_acceleration", "PWL",
    "ema_200_h1", "long_short_ratio"
]

LABEL_ORDINAL = {"SHORT": 0, "FLAT": 1, "LONG": 2}

def load_regime_data(coins: list[str], regime_enc: int, feature_cols: list) -> tuple[pd.DataFrame, pd.Series]:
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
        
        # Filter to only the specific regime
        df_reg = df[df["hmm_regime_enc"] == regime_enc].copy()
        if df_reg.empty:
            continue
            
        frames.append(df_reg)
        
    if not frames:
        raise FileNotFoundError(f"Tidak ada data ditemukan untuk regime {regime_enc}")
        
    combined = pd.concat(frames).sort_index()
    combined = combined[combined["trend_label"].isin(LABEL_ORDINAL)].copy()
    
    X = combined[feature_cols].copy()
    y = combined["trend_label"].map(LABEL_ORDINAL).astype(np.int32)
    
    return X, y

def train_sp_model(X: pd.DataFrame, y: pd.Series, regime_name: str, run_dir: Path) -> lgb.LGBMClassifier:
    logger.info(f"Training Specialist LGBM for {regime_name} — Samples: {len(X):,} × Features: {len(X.columns)}")
    
    folds = build_purged_folds(X.index, n_folds=N_FOLDS, purge=PURGE_GAP_BARS)
    
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
    
    # Retrain on 100% data
    final_params = LGBM_PARAMS.copy()
    final_params["n_estimators"] = avg_best_iter
    
    final_model = lgb.LGBMClassifier(**final_params)
    full_sample_w = np.array([LGBM_CLASS_WEIGHTS[int(label)] for label in y], dtype=np.float32)
    final_model.fit(X, y, sample_weight=full_sample_w)
    
    # Save model
    model_fname = f"lgbm_regime_{regime_name}.pkl"
    joblib.dump(final_model, run_dir / model_fname)
    joblib.dump(final_model, MODEL_DIR / model_fname)
    logger.info(f"Saved: {run_dir / model_fname} and root baseline {MODEL_DIR / model_fname}")
    
    return final_model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default=None)
    args = parser.parse_args()
    
    run_id = args.run_id or f"trending_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = MODEL_DIR / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Run ID: {run_id} | Dir: {run_dir}")
    
    # Train TRENDING_UP (HMM State 3)
    logger.info("=== STEP 1: TRENDING_UP MODEL ===")
    X_up, y_up = load_regime_data(ALL_COINS, 3, UP_FEATS)
    train_sp_model(X_up, y_up, "TRENDING_UP", run_dir)
    
    # Train TRENDING_DOWN (HMM State 0)
    logger.info("\n=== STEP 2: TRENDING_DOWN MODEL ===")
    X_down, y_down = load_regime_data(ALL_COINS, 0, DOWN_FEATS)
    train_sp_model(X_down, y_down, "TRENDING_DOWN", run_dir)
    
    # Save training metadata
    meta = {
        "run_id": run_id,
        "timestamp": str(datetime.now()),
        "up_feats": UP_FEATS,
        "down_feats": DOWN_FEATS,
        "params": LGBM_PARAMS,
    }
    with open(run_dir / "trending_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)
        
    print(f"\n{'='*60}")
    print(f"  LGBM TRENDING TRAINING COMPLETE — {run_id}")
    print(f"  Models saved to {run_dir}")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()
