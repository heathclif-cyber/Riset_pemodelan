"""
pipeline/05c_train_lstm_momentum.py — Training LSTM Momentum Detector (H4 Sequence)

Perbedaan fundamental dari 05_train_lstm.py:
  LAMA : Input H1 flat features (104 fitur), swing labels (81% FLAT)
  BARU : Input H4 sequence pre-built (16 bar × 8 fitur), momentum labels (~48% FLAT)

Peran LSTM baru dalam cascade:
  LGBM  → "Apakah setup struktural valid?" (snapshot 104 fitur H1)
  LSTM  → "Apakah ada momentum directional 2-3 hari terakhir?" (H4 sequence)

  Entry = weighted fusion: combined = 0.65×lgbm_probs + 0.35×lstm_probs
  Threshold diterapkan ke combined probability, bukan LGBM murni.

Prerequisite:
  python pipeline/05a_generate_momentum_labels.py --all
  python pipeline/05b_build_h4_sequences.py --all

Jalankan:
  python pipeline/05c_train_lstm_momentum.py
  python pipeline/05c_train_lstm_momentum.py --run-id cascade_v4.2

Output:
  models/runs/{run_id}/lstm_momentum.pt
  models/runs/{run_id}/lstm_momentum_scaler.pkl
  models/runs/{run_id}/lstm_momentum_cv.json
  models/lstm_momentum.pt        ← symlink inference (selalu overwrite)
  models/lstm_momentum_scaler.pkl
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
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from config import (
    TRAINING_COINS, ALL_COINS,
    TRAINING_DIR, MODEL_DIR,
    N_FOLDS, PURGE_GAP_BARS,
    LSTM_HIDDEN, LSTM_LAYERS, LSTM_DROPOUT,
    LSTM_EPOCHS, LSTM_PATIENCE, LSTM_BATCH_SIZE, LSTM_LR,
)
from core.models import TradingLSTM, save_lstm
from core.utils import setup_logger, get_lstm_device
from pipeline.shared import build_purged_folds

logger   = setup_logger("05c_lstm_momentum")
DEVICE   = get_lstm_device()
SEQ_DIR  = TRAINING_DIR / "h4_sequences"

# Konstanta dataset
SEQ_LEN    = 16   # H4 bars per sequence
N_FEATURES = 8    # h4_return, volume, rsi_h4, ema_21_slope_h4,
                  # h4_trend, trend_strength, cvd_slope_h4, atr_percent_h4
NUM_CLASSES = 3   # SHORT=0, FLAT=1, LONG=2
LABEL_NAMES = {0: "SHORT", 1: "FLAT", 2: "LONG"}


# ─── Dataset ──────────────────────────────────────────────────────────────────

class H4SequenceDataset(Dataset):
    """
    Dataset sederhana untuk H4 sequences yang sudah pre-built.
    X: (n_samples, seq_len, n_features) — sudah berbentuk sequences
    y: (n_samples,) — momentum labels (SHORT=0, FLAT=1, LONG=2)
    """
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.int64))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

    def get_labels(self) -> np.ndarray:
        return self.y.numpy()


# ─── Data Loading ─────────────────────────────────────────────────────────────

def load_dataset(coins: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load pre-built H4 sequences dari file .npz per coin atau combined.
    Filter hanya koin yang diminta.

    Returns:
        X  : (n, 16, 8) float32
        y  : (n,) int32
        ts : (n,) int64 — H1 unix timestamps untuk purged CV
    """
    combined = SEQ_DIR / "all_coins_seq.npz"
    if combined.exists():
        data    = np.load(combined)
        X_all   = data["X"]
        y_all   = data["y"]
        ts_all  = data["ts"]
        cid_all = data["coin_id"]

        # Filter koin jika bukan ALL_COINS
        if set(coins) != set(ALL_COINS):
            from config import SYMBOL_MAP
            valid_cids = {SYMBOL_MAP[c] for c in coins if c in SYMBOL_MAP}
            mask = np.isin(cid_all, list(valid_cids))
            X_all, y_all, ts_all = X_all[mask], y_all[mask], ts_all[mask]

        logger.info(f"Loaded combined: {len(X_all):,} samples dari {len(set(cid_all))} koin")
        return X_all, y_all, ts_all

    # Fallback: load per-coin
    X_list, y_list, ts_list = [], [], []
    for sym in coins:
        path = SEQ_DIR / f"{sym}_seq.npz"
        if not path.exists():
            logger.warning(f"{sym}: file seq tidak ditemukan, skip")
            continue
        d = np.load(path)
        X_list.append(d["X"])
        y_list.append(d["y"])
        ts_list.append(d["ts"])
        logger.info(f"  {sym}: {len(d['y']):,} samples")

    X  = np.concatenate(X_list,  axis=0)
    y  = np.concatenate(y_list,  axis=0)
    ts = np.concatenate(ts_list, axis=0)

    # Sort by timestamp
    order = np.argsort(ts)
    return X[order], y[order].astype(np.int32), ts[order]


# ─── Scaling ──────────────────────────────────────────────────────────────────

def fit_scaler(X: np.ndarray) -> StandardScaler:
    """
    Fit StandardScaler pada X yang berbentuk (n, seq_len, n_features).
    Reshape ke (n*seq_len, n_features) untuk fitting per-feature.
    """
    n, s, f = X.shape
    scaler = StandardScaler()
    scaler.fit(X.reshape(-1, f))
    return scaler


def scale_X(X: np.ndarray, scaler: StandardScaler) -> np.ndarray:
    n, s, f = X.shape
    return scaler.transform(X.reshape(-1, f)).reshape(n, s, f).astype(np.float32)


# ─── Training Helpers ─────────────────────────────────────────────────────────

def build_sampler(labels: np.ndarray) -> WeightedRandomSampler:
    counts  = np.bincount(labels, minlength=NUM_CLASSES).astype(np.float64)
    counts  = np.where(counts == 0, 1, counts)
    class_w = 1.0 / counts
    sample_w = class_w[labels]
    return WeightedRandomSampler(
        weights     = torch.from_numpy(sample_w).float(),
        num_samples = len(sample_w),
        replacement = True,
    )


def compute_class_weights(y: np.ndarray) -> torch.Tensor:
    counts  = np.bincount(y, minlength=NUM_CLASSES).astype(np.float64)
    counts  = np.where(counts == 0, 1, counts)
    weights = len(y) / (NUM_CLASSES * counts)
    return torch.tensor(weights, dtype=torch.float32).to(DEVICE)


def train_one_fold(
    X_tr: np.ndarray, y_tr: np.ndarray,
    X_te: np.ndarray, y_te: np.ndarray,
    fold:  int,
    scaler: StandardScaler,
) -> tuple[TradingLSTM, dict]:

    logger.info(f"[Fold {fold}] train={len(X_tr):,} | val={len(X_te):,}")

    X_tr_sc = scale_X(X_tr, scaler)
    X_te_sc = scale_X(X_te, scaler)

    train_ds = H4SequenceDataset(X_tr_sc, y_tr)
    val_ds   = H4SequenceDataset(X_te_sc, y_te)
    sampler  = build_sampler(train_ds.get_labels())

    # num_workers=0 untuk Windows compatibility
    train_loader = DataLoader(train_ds, batch_size=LSTM_BATCH_SIZE,
                              sampler=sampler, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=LSTM_BATCH_SIZE,
                              shuffle=False, num_workers=0)

    model     = TradingLSTM(N_FEATURES, LSTM_HIDDEN, LSTM_LAYERS, LSTM_DROPOUT).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=compute_class_weights(y_tr))
    optimizer = torch.optim.Adam(model.parameters(), lr=LSTM_LR)

    best_f1, best_state, patience_cnt, best_epoch = -1.0, None, 0, 1

    for epoch in range(1, LSTM_EPOCHS + 1):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        # Validation
        model.eval()
        preds_all, labels_all = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                p = model(xb.to(DEVICE)).argmax(dim=1).cpu().numpy()
                preds_all.extend(p)
                labels_all.extend(yb.numpy())

        f1 = float(f1_score(labels_all, preds_all, average="macro", zero_division=0))

        if f1 > best_f1:
            best_f1      = f1
            best_state   = {k: v.cpu() for k, v in model.state_dict().items()}
            patience_cnt = 0
            best_epoch   = epoch
        else:
            patience_cnt += 1
            if patience_cnt >= LSTM_PATIENCE:
                logger.info(f"[Fold {fold}] Early stop @ epoch {epoch} | best F1={best_f1:.4f} @ epoch {best_epoch}")
                break

        if epoch % 10 == 0 or epoch == 1:
            logger.info(f"[Fold {fold}] epoch {epoch:>3} | F1={f1:.4f} | best={best_f1:.4f}")

    model.load_state_dict(best_state)
    model.eval()

    preds_all, labels_all = [], []
    with torch.no_grad():
        for xb, yb in val_loader:
            p = model(xb.to(DEVICE)).argmax(dim=1).cpu().numpy()
            preds_all.extend(p)
            labels_all.extend(yb.numpy())

    f1_per = f1_score(labels_all, preds_all, average=None, zero_division=0, labels=[0, 1, 2])
    metrics = {
        "fold":        fold,
        "n_train":     len(X_tr),
        "n_val":       len(X_te),
        "best_epoch":  best_epoch,
        "accuracy":    round(float(accuracy_score(labels_all, preds_all)), 4),
        "f1_macro":    round(best_f1, 4),
        "f1_weighted": round(float(f1_score(labels_all, preds_all, average="weighted", zero_division=0)), 4),
        "f1_SHORT":    round(float(f1_per[0]), 4),
        "f1_FLAT":     round(float(f1_per[1]), 4),
        "f1_LONG":     round(float(f1_per[2]), 4),
        "confusion_matrix": confusion_matrix(labels_all, preds_all, labels=[0, 1, 2]).tolist(),
    }
    logger.info(
        f"[Fold {fold}] F1={best_f1:.4f} | "
        f"LONG={f1_per[2]:.4f} FLAT={f1_per[1]:.4f} SHORT={f1_per[0]:.4f}"
    )
    return model, metrics


def retrain_final(
    X_all: np.ndarray, y_all: np.ndarray,
    final_epochs: int, scaler: StandardScaler,
) -> TradingLSTM:
    logger.info(f"Final retrain: {len(X_all):,} samples | epochs={final_epochs}")
    X_sc     = scale_X(X_all, scaler)
    ds       = H4SequenceDataset(X_sc, y_all)
    sampler  = build_sampler(ds.get_labels())
    loader   = DataLoader(ds, batch_size=LSTM_BATCH_SIZE, sampler=sampler, num_workers=0)

    model     = TradingLSTM(N_FEATURES, LSTM_HIDDEN, LSTM_LAYERS, LSTM_DROPOUT).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=compute_class_weights(y_all))
    optimizer = torch.optim.Adam(model.parameters(), lr=LSTM_LR)

    for epoch in range(1, final_epochs + 1):
        model.train()
        total_loss = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        if epoch % 10 == 0 or epoch in (1, final_epochs):
            logger.info(f"[Final] epoch {epoch:>3}/{final_epochs} | loss={total_loss/len(loader):.4f}")

    model.eval()
    return model


# ─── Main ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run-id", default=None,        help="Run ID (default: tanggal)")
    p.add_argument("--all",    action="store_true",  help="Pakai ALL_COINS")
    return p.parse_args()


def main():
    args   = parse_args()
    coins  = ALL_COINS if args.all else TRAINING_COINS
    run_id = args.run_id or f"lstm_momentum_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    run_dir = MODEL_DIR / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Run ID : {run_id}")
    logger.info(f"Device : {DEVICE}")
    logger.info(f"Koin   : {len(coins)}")
    logger.info(f"Model  : TradingLSTM(input={N_FEATURES}, hidden={LSTM_HIDDEN}, layers={LSTM_LAYERS})")
    logger.info("-" * 65)

    # 1. Load dataset
    X, y, ts = load_dataset(coins)
    logger.info(f"Dataset: X={X.shape}, y={y.shape}")
    for lbl, name in LABEL_NAMES.items():
        n = (y == lbl).sum()
        logger.info(f"  {name:5}: {n:>8,} ({n/len(y)*100:.1f}%)")

    # 2. Fit scaler pada seluruh data
    logger.info("Fitting scaler...")
    final_scaler = fit_scaler(X)

    # 3. Build purged folds dari H1 timestamps
    ts_index = pd.to_datetime(ts, utc=True)
    folds    = build_purged_folds(ts_index, n_folds=N_FOLDS, purge=PURGE_GAP_BARS)
    logger.info(f"Purged CV: {N_FOLDS} folds, purge={PURGE_GAP_BARS} bars")

    # 4. Walk-forward CV
    all_metrics = []
    best_model, best_f1, best_fold = None, -1.0, -1

    for fold_num, (tr_idx, te_idx) in enumerate(folds, 1):
        model, metrics = train_one_fold(
            X[tr_idx], y[tr_idx],
            X[te_idx], y[te_idx],
            fold=fold_num,
            scaler=final_scaler,
        )
        all_metrics.append(metrics)
        if metrics["f1_macro"] > best_f1:
            best_f1, best_model, best_fold = metrics["f1_macro"], model, fold_num

    # 5. Final retrain pada 100% data
    avg_epochs  = int(np.mean([m["best_epoch"] for m in all_metrics]))
    final_model = retrain_final(X, y, avg_epochs, final_scaler)

    # 6. Simpan model & scaler
    model_path  = run_dir / "lstm_momentum.pt"
    scaler_path = run_dir / "lstm_momentum_scaler.pkl"
    save_lstm(final_model, model_path)
    joblib.dump(final_scaler, scaler_path)
    logger.info(f"Model  saved → {model_path}")
    logger.info(f"Scaler saved → {scaler_path}")

    # Overwrite root models/ untuk inference langsung
    joblib.dump(final_scaler, MODEL_DIR / "lstm_momentum_scaler.pkl")
    save_lstm(final_model, MODEL_DIR / "lstm_momentum.pt")

    # 7. Simpan metadata model — penting untuk inference tahu ini H4 momentum LSTM
    meta = {
        "model_type":   "lstm_momentum_h4",
        "seq_len":      SEQ_LEN,
        "n_features":   N_FEATURES,
        "feature_order": [
            "h4_return", "volume", "rsi_h4", "ema_21_slope_h4",
            "h4_trend", "trend_strength", "cvd_slope_h4", "atr_percent_h4",
        ],
        "label_map":    {"SHORT": 0, "FLAT": 1, "LONG": 2},
        "fusion_weight_lstm": 0.35,
        "fusion_weight_lgbm": 0.65,
        "run_id":       run_id,
        "trained_at":   datetime.now().isoformat(),
        "n_coins":      len(coins),
    }
    meta_path = run_dir / "lstm_momentum_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    with open(MODEL_DIR / "lstm_momentum_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    # 8. Simpan CV results
    f1s = [m["f1_macro"] for m in all_metrics]
    cv_summary = {
        "run_id":           run_id,
        "model_type":       "lstm_momentum_h4",
        "seq_len":          SEQ_LEN,
        "n_features":       N_FEATURES,
        "n_folds":          N_FOLDS,
        "purge_gap_bars":   PURGE_GAP_BARS,
        "best_fold":        best_fold,
        "best_f1_macro":    round(best_f1, 4),
        "mean_f1_macro":    round(float(np.mean(f1s)), 4),
        "std_f1_macro":     round(float(np.std(f1s)), 4),
        "final_epochs":     avg_epochs,
        "folds":            all_metrics,
    }
    with open(run_dir / "lstm_momentum_cv.json", "w") as f:
        json.dump(cv_summary, f, indent=2, default=str)

    # 9. Print summary
    sep = "=" * 65
    print(f"\n{sep}")
    print(f"  LSTM MOMENTUM TRAINING SELESAI — {run_id}")
    print(f"{sep}")
    print(f"  Device      : {DEVICE}")
    print(f"  Best fold   : {best_fold} (F1={best_f1:.4f})")
    print(f"  Mean F1     : {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")
    print(f"  Final epochs: {avg_epochs}")
    print(f"  Model       : {model_path}")
    print(f"  Inference   : {MODEL_DIR}/lstm_momentum.pt")
    print(f"{sep}")
    print(f"\n  {'Fold':>4}  {'F1-mac':>7}  {'LONG':>7}  {'FLAT':>7}  {'SHORT':>7}  {'Epoch':>6}")
    print(f"  {'-'*50}")
    for m in all_metrics:
        print(f"  {m['fold']:>4}  {m['f1_macro']:>7.4f}  {m['f1_LONG']:>7.4f}  "
              f"{m['f1_FLAT']:>7.4f}  {m['f1_SHORT']:>7.4f}  {m['best_epoch']:>6}")
    print()


if __name__ == "__main__":
    main()
