"""
pipeline/06_ensemble.py — Fase 6: Stacking Ensemble LightGBM + LSTM → Logistic Regression

Jalankan:
  python pipeline/06_ensemble.py
  python pipeline/06_ensemble.py --run-id my_run

Output: models/runs/{run_id}/ensemble_meta.pkl, ensemble_cv_results.json
"""

import argparse
import json
import sys
import warnings
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from config import (
    TRAINING_COINS, ALL_COINS, LABEL_DIR, MODEL_DIR,
    N_FOLDS, PURGE_GAP_BARS,
    LABEL_MAP, LABEL_MAP_INV, NUM_CLASSES,
    LSTM_SEQ_LEN, LSTM_HIDDEN, LSTM_LAYERS, LSTM_DROPOUT,
)
from core.models import TradingLSTM, load_lstm, ProbabilityCalibrator
from core.utils import setup_logger
from pipeline.p05_utils import SequenceDataset  # reuse dari 05

logger = setup_logger("06_ensemble")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NON_FEATURE_COLS = {"label"}


def load_symbols(coins):
    frames = []
    for sym in coins:
        path = LABEL_DIR / f"{sym}_features_v2.parquet"
        if not path.exists():
            continue
        df = pd.read_parquet(path)
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index, utc=True)
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        if "OB_price" in df.columns:
            df.drop(columns=["OB_price"], inplace=True)
        frames.append(df)
    combined = pd.concat(frames).sort_index()
    return combined


def preprocess(df):
    label_str = df["label"].astype(str)
    mask = label_str.isin(LABEL_MAP)
    df   = df[mask].copy()
    y    = label_str[mask].map(LABEL_MAP).values.astype(np.int64)
    feat_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]
    df[feat_cols] = df[feat_cols].ffill().fillna(0)
    return df, y


def build_purged_folds(n):
    splits = np.array_split(np.arange(n), N_FOLDS + 1)
    folds  = []
    for k in range(1, N_FOLDS + 1):
        train_raw = np.concatenate(splits[:k])
        test_raw  = splits[k]
        train_idx = train_raw[:-PURGE_GAP_BARS] if len(train_raw) > PURGE_GAP_BARS else train_raw
        test_idx  = test_raw[PURGE_GAP_BARS:]   if len(test_raw) > PURGE_GAP_BARS  else test_raw
        folds.append((train_idx, test_idx))
    return folds


def get_lstm_proba(model, scaler, X: np.ndarray) -> np.ndarray:
    X_sc  = scaler.transform(X)
    dummy = np.zeros(len(X_sc), dtype=np.int64)
    ds    = SequenceDataset(X_sc, dummy)
    loader = DataLoader(ds, batch_size=1024, shuffle=False)
    proba_list = []
    model.eval()
    with torch.no_grad():
        for xb, _ in loader:
            logits = model(xb.to(DEVICE))
            proba  = torch.softmax(logits, dim=1).cpu().numpy()
            proba_list.append(proba)
    return np.vstack(proba_list)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--run-id", default=None)
    args   = parser.parse_args()
    coins  = ALL_COINS if args.all else TRAINING_COINS
    run_id = args.run_id or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = MODEL_DIR / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # Load base models
    lgbm_path  = MODEL_DIR / "lgbm_baseline.pkl"
    lstm_path  = MODEL_DIR / "lstm_best.pt"
    scaler_path = MODEL_DIR / "lstm_scaler.pkl"
    if not lgbm_path.exists() or not lstm_path.exists():
        raise FileNotFoundError("Jalankan 04_train_lgbm.py dan 05_train_lstm.py dulu.")

    lgbm_model  = joblib.load(lgbm_path)
    lstm_model  = load_lstm(lstm_path, device=str(DEVICE)).to(DEVICE)
    lstm_scaler = joblib.load(scaler_path)
    logger.info(f"Base models loaded. Device: {DEVICE}")

    df = load_symbols(coins)
    df, y = preprocess(df)
    feat_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]
    X = df[feat_cols].values.astype(np.float64)

    folds       = build_purged_folds(len(df))
    oof_proba   = np.zeros((len(y), NUM_CLASSES * 2))
    oof_valid   = np.zeros(len(y), dtype=bool)
    all_metrics = []

    for fold_num, (tr_pos, te_pos) in enumerate(folds, 1):
        logger.info(f"[Fold {fold_num}] Generating OOF probabilities...")
        X_te = df.iloc[te_pos][feat_cols]
        X_te_raw = X[te_pos]

        lgbm_proba = lgbm_model.predict_proba(X_te)
        lstm_proba = get_lstm_proba(lstm_model, lstm_scaler, X_te_raw)

        # Align LSTM (seq_len offset) dengan LGBM
        seq_offset = LSTM_SEQ_LEN - 1
        if len(lstm_proba) < len(lgbm_proba):
            pad = np.ones((len(lgbm_proba) - len(lstm_proba), NUM_CLASSES)) / NUM_CLASSES
            lstm_proba = np.vstack([pad, lstm_proba])

        meta = np.hstack([lgbm_proba, lstm_proba])
        oof_proba[te_pos] = meta
        oof_valid[te_pos] = True

        # Evaluasi ensemble sederhana (argmax dari rata-rata proba)
        avg_proba = (lgbm_proba + lstm_proba) / 2
        y_pred = avg_proba.argmax(axis=1)
        f1 = float(f1_score(y[te_pos], y_pred, average="macro", zero_division=0))
        f1_per = f1_score(y[te_pos], y_pred, average=None, zero_division=0, labels=[0, 1, 2])
        all_metrics.append({
            "fold": fold_num, "f1_macro": round(f1, 4),
            "f1_SHORT": round(float(f1_per[0]), 4),
            "f1_FLAT":  round(float(f1_per[1]), 4),
            "f1_LONG":  round(float(f1_per[2]), 4),
        })
        logger.info(f"[Fold {fold_num}] F1-macro={f1:.4f} LONG={f1_per[2]:.4f}")

    # Train final meta-learner pada semua OOF
    valid_idx = np.where(oof_valid)[0]
    meta_X = oof_proba[valid_idx]
    meta_y = y[valid_idx]

    meta_learner = LogisticRegression(
        max_iter=1000, C=1.0, multi_class="multinomial",
        solver="lbfgs", random_state=42,
    )
    meta_learner.fit(meta_X, meta_y)
    logger.info("Meta-learner Logistic Regression trained.")

    # Simpan meta-learner
    meta_path = run_dir / "ensemble_meta.pkl"
    joblib.dump(meta_learner, meta_path)
    joblib.dump(meta_learner, MODEL_DIR / "ensemble_meta.pkl")

    # ★ v2: fit dan simpan probability calibrator
    logger.info("Fitting ProbabilityCalibrator pada OOF predictions...")
    calibrator = ProbabilityCalibrator(method="isotonic")
    oof_meta_proba = meta_learner.predict_proba(meta_X)
    calibrator.fit(oof_meta_proba, meta_y)

    cal_path = run_dir / "calibrator.pkl"
    calibrator.save(cal_path)
    calibrator.save(MODEL_DIR / "calibrator.pkl")
    logger.info(f"Calibrator → {cal_path}")

    f1s = [m["f1_macro"] for m in all_metrics]
    cv_summary = {
        "run_id": run_id, "coins": coins,
        "mean_f1_macro": round(float(np.mean(f1s)), 4),
        "std_f1_macro":  round(float(np.std(f1s)), 4),
        "folds": all_metrics,
    }
    cv_path = run_dir / "ensemble_cv_results.json"
    with open(cv_path, "w") as f:
        json.dump(cv_summary, f, indent=2)
    with open(MODEL_DIR / "ensemble_cv_results.json", "w") as f:
        json.dump(cv_summary, f, indent=2)

    sep = "=" * 60
    print(f"\n{sep}")
    print(f"  ENSEMBLE TRAINING SELESAI — {run_id}")
    print(f"{sep}")
    print(f"  Mean F1 : {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")
    print(f"  Model   : {meta_path}")
    print(f"{sep}\n")


if __name__ == "__main__":
    main()
