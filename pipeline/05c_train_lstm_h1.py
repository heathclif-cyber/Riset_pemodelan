"""
pipeline/05c_train_lstm_h1.py — Training LSTM Momentum Detector (H1 Sequence)

Perbedaan dari 05c (H4):
  INPUT : 32 H1 bars × 12 fitur      (vs 16 H4 bars × 8 fitur)
  TARGET: momentum label N=8 H1       (sama)
  FIX 1 : Hapus WeightedRandomSampler — pakai class weights di loss saja
  FIX 2 : PATIENCE = 10              (vs 5 sebelumnya)

Skala alignment terjaga:
  32 jam input → prediksi 8 jam ke depan (rasio 4:1, masuk akal)

Prerequisite:
  python pipeline/05a_generate_momentum_labels.py --all
  python pipeline/05b_build_h1_sequences.py --all

Jalankan:
  python pipeline/05c_train_lstm_h1.py --all
  python pipeline/05c_train_lstm_h1.py --all --run-id cascade_v4.3
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
from torch.utils.data import DataLoader, Dataset

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from config import (
    TRAINING_COINS, ALL_COINS,
    TRAINING_DIR, MODEL_DIR,
    N_FOLDS, PURGE_GAP_BARS,
    LSTM_HIDDEN, LSTM_LAYERS, LSTM_DROPOUT,
    LSTM_EPOCHS, LSTM_BATCH_SIZE, LSTM_LR,
)
from core.models import TradingLSTM, save_lstm
from core.utils import setup_logger, get_lstm_device
from pipeline.shared import build_purged_folds

logger     = setup_logger("05c_lstm_h1")
DEVICE     = get_lstm_device()
SEQ_DIR_H1 = TRAINING_DIR / "h1_sequences"

SEQ_LEN     = 32   # H1 bars
N_FEATURES  = 12   # fitur per bar
NUM_CLASSES = 3
PATIENCE    = 15   # batch 1024 → 2x fewer updates/epoch, perlu lebih banyak epoch
LABEL_NAMES = {0: "SHORT", 1: "FLAT", 2: "LONG"}


class H1SequenceDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.int64))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

    def get_labels(self) -> np.ndarray:
        return self.y.numpy()


def load_dataset(coins: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    combined = SEQ_DIR_H1 / "all_coins_seq.npz"
    if combined.exists():
        data = np.load(combined)
        logger.info(f"Loaded combined H1 sequences: {len(data['y']):,} samples")
        return data["X"], data["y"], data["ts"]

    X_list, y_list, ts_list = [], [], []
    for sym in coins:
        path = SEQ_DIR_H1 / f"{sym}_seq.npz"
        if not path.exists():
            logger.warning(f"{sym}: skip")
            continue
        d = np.load(path)
        X_list.append(d["X"]); y_list.append(d["y"]); ts_list.append(d["ts"])

    X  = np.concatenate(X_list,  axis=0)
    y  = np.concatenate(y_list,  axis=0)
    ts = np.concatenate(ts_list, axis=0)
    order = np.argsort(ts)
    return X[order], y[order].astype(np.int32), ts[order]


def fit_scaler(X: np.ndarray) -> StandardScaler:
    n, s, f = X.shape
    scaler = StandardScaler()
    scaler.fit(X.reshape(-1, f))
    return scaler


def scale_X(X: np.ndarray, scaler: StandardScaler) -> np.ndarray:
    n, s, f = X.shape
    return scaler.transform(X.reshape(-1, f)).reshape(n, s, f).astype(np.float32)


def compute_class_weights(y: np.ndarray) -> torch.Tensor:
    counts  = np.bincount(y, minlength=NUM_CLASSES).astype(np.float64)
    counts  = np.where(counts == 0, 1, counts)
    weights = len(y) / (NUM_CLASSES * counts)
    return torch.tensor(weights, dtype=torch.float32).to(DEVICE)


def train_one_fold(
    X_tr: np.ndarray, y_tr: np.ndarray,
    X_te: np.ndarray, y_te: np.ndarray,
    fold: int,
) -> tuple[TradingLSTM, dict]:

    logger.info(f"[Fold {fold}] train={len(X_tr):,} | val={len(X_te):,}")

    # Fit scaler hanya dari training fold — cegah val statistics bocor ke scaler
    fold_scaler = fit_scaler(X_tr)
    X_tr_sc = scale_X(X_tr, fold_scaler)
    X_te_sc = scale_X(X_te, fold_scaler)

    train_ds = H1SequenceDataset(X_tr_sc, y_tr)
    val_ds   = H1SequenceDataset(X_te_sc, y_te)

    # FIX: shuffle=True saja, tidak pakai WeightedRandomSampler
    train_loader = DataLoader(train_ds, batch_size=LSTM_BATCH_SIZE,
                              shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=LSTM_BATCH_SIZE,
                              shuffle=False, num_workers=0)

    model     = TradingLSTM(N_FEATURES, LSTM_HIDDEN, LSTM_LAYERS, LSTM_DROPOUT).to(DEVICE)
    # FIX: class weights di loss saja (tidak double dengan sampler)
    criterion = nn.CrossEntropyLoss(weight=compute_class_weights(y_tr))
    optimizer = torch.optim.Adam(model.parameters(), lr=LSTM_LR, weight_decay=1e-4)

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

        model.eval()
        preds_all, labels_all = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                p = model(xb.to(DEVICE)).argmax(dim=1).cpu().numpy()
                preds_all.extend(p); labels_all.extend(yb.numpy())

        f1 = float(f1_score(labels_all, preds_all, average="macro", zero_division=0))
        if f1 > best_f1:
            best_f1      = f1
            best_state   = {k: v.cpu() for k, v in model.state_dict().items()}
            patience_cnt = 0
            best_epoch   = epoch
        else:
            patience_cnt += 1
            if patience_cnt >= PATIENCE:
                logger.info(f"[Fold {fold}] Early stop @ epoch {epoch} | best F1={best_f1:.4f} @ epoch {best_epoch}")
                break

        if epoch % 5 == 0 or epoch == 1:
            logger.info(f"[Fold {fold}] epoch {epoch:>3} | F1={f1:.4f} | best={best_f1:.4f}")

    model.load_state_dict(best_state)
    model.eval()

    preds_all, labels_all = [], []
    with torch.no_grad():
        for xb, yb in val_loader:
            p = model(xb.to(DEVICE)).argmax(dim=1).cpu().numpy()
            preds_all.extend(p); labels_all.extend(yb.numpy())

    f1_per = f1_score(labels_all, preds_all, average=None, zero_division=0, labels=[0, 1, 2])
    metrics = {
        "fold": fold, "n_train": len(X_tr), "n_val": len(X_te),
        "best_epoch":  best_epoch,
        "accuracy":    round(float(accuracy_score(labels_all, preds_all)), 4),
        "f1_macro":    round(best_f1, 4),
        "f1_weighted": round(float(f1_score(labels_all, preds_all, average="weighted", zero_division=0)), 4),
        "f1_SHORT":    round(float(f1_per[0]), 4),
        "f1_FLAT":     round(float(f1_per[1]), 4),
        "f1_LONG":     round(float(f1_per[2]), 4),
        "confusion_matrix": confusion_matrix(labels_all, preds_all, labels=[0, 1, 2]).tolist(),
    }
    logger.info(f"[Fold {fold}] F1={best_f1:.4f} | LONG={f1_per[2]:.4f} FLAT={f1_per[1]:.4f} SHORT={f1_per[0]:.4f}")
    return model, metrics


def retrain_final(X_all, y_all, final_epochs, scaler):
    logger.info(f"Final retrain: {len(X_all):,} samples | epochs={final_epochs}")
    X_sc    = scale_X(X_all, scaler)
    ds      = H1SequenceDataset(X_sc, y_all)
    loader  = DataLoader(ds, batch_size=LSTM_BATCH_SIZE, shuffle=True, num_workers=0)
    model   = TradingLSTM(N_FEATURES, LSTM_HIDDEN, LSTM_LAYERS, LSTM_DROPOUT).to(DEVICE)
    crit    = nn.CrossEntropyLoss(weight=compute_class_weights(y_all))
    optim   = torch.optim.Adam(model.parameters(), lr=LSTM_LR, weight_decay=1e-4)

    for epoch in range(1, final_epochs + 1):
        model.train()
        total_loss = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optim.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            total_loss += loss.item()
        if epoch % 5 == 0 or epoch in (1, final_epochs):
            logger.info(f"[Final] epoch {epoch:>3}/{final_epochs} | loss={total_loss/len(loader):.4f}")

    model.eval()
    return model


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run-id", default=None)
    p.add_argument("--all",    action="store_true")
    return p.parse_args()


def main():
    args   = parse_args()
    coins  = ALL_COINS if args.all else TRAINING_COINS
    run_id = args.run_id or f"lstm_h1_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    run_dir = MODEL_DIR / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Run ID  : {run_id}")
    logger.info(f"Device  : {DEVICE}")
    logger.info(f"Model   : TradingLSTM(input={N_FEATURES}, hidden={LSTM_HIDDEN}, layers={LSTM_LAYERS})")
    logger.info(f"Seq len : {SEQ_LEN} H1 bars = {SEQ_LEN} jam")
    logger.info(f"Patience: {PATIENCE}  (FIX dari 5)")
    logger.info(f"Sampler : DISABLED  (FIX — class weights di loss saja)")
    logger.info("-" * 65)

    X, y, ts = load_dataset(coins)
    logger.info(f"Dataset: X={X.shape}, y={y.shape}")
    for lbl, name in LABEL_NAMES.items():
        n = (y == lbl).sum()
        logger.info(f"  {name:5}: {n:>8,} ({n/len(y)*100:.1f}%)")

    final_scaler = fit_scaler(X)
    ts_index     = pd.to_datetime(ts, unit="ns", utc=True)
    folds        = build_purged_folds(ts_index, n_folds=N_FOLDS, purge=PURGE_GAP_BARS)
    logger.info(f"Purged CV: {N_FOLDS} folds, purge={PURGE_GAP_BARS} bars")

    all_metrics, best_model, best_f1, best_fold = [], None, -1.0, -1

    for fold_num, (tr_idx, te_idx) in enumerate(folds, 1):
        model, metrics = train_one_fold(
            X[tr_idx], y[tr_idx],
            X[te_idx], y[te_idx],
            fold=fold_num,
        )
        all_metrics.append(metrics)
        if metrics["f1_macro"] > best_f1:
            best_f1, best_model, best_fold = metrics["f1_macro"], model, fold_num

    avg_epochs  = int(np.mean([m["best_epoch"] for m in all_metrics]))
    final_model = retrain_final(X, y, avg_epochs, final_scaler)

    # Simpan
    model_path  = run_dir / "lstm_momentum.pt"
    scaler_path = run_dir / "lstm_momentum_scaler.pkl"
    save_lstm(final_model, model_path)
    joblib.dump(final_scaler, scaler_path)
    save_lstm(final_model, MODEL_DIR / "lstm_momentum.pt")
    joblib.dump(final_scaler, MODEL_DIR / "lstm_momentum_scaler.pkl")

    meta = {
        "model_type":    "lstm_momentum_h1",
        "seq_len":       SEQ_LEN,
        "n_features":    N_FEATURES,
        "feature_order": [
            "h1_return", "log_ret_5", "log_ret_20",
            "volume_delta", "ofi_raw", "ofi_acceleration",
            "rsi_6", "stochrsi_k", "vwdp_smooth",
            "atr_14_h1", "vol_ratio_20", "bars_since_BOS",
        ],
        "label_map":           {"SHORT": 0, "FLAT": 1, "LONG": 2},
        "fusion_weight_lstm":  0.35,
        "fusion_weight_lgbm":  0.65,
        "fixes_applied":       ["no_weighted_sampler", "patience_10", "h1_scale_aligned", "fold_scaler", "weight_decay_1e4"],
        "run_id":              run_id,
        "trained_at":          datetime.now().isoformat(),
    }
    for p in [run_dir / "lstm_momentum_meta.json", MODEL_DIR / "lstm_momentum_meta.json"]:
        with open(p, "w") as f:
            json.dump(meta, f, indent=2)

    f1s = [m["f1_macro"] for m in all_metrics]
    with open(run_dir / "lstm_momentum_cv.json", "w") as f:
        json.dump({
            "run_id": run_id, "model_type": "lstm_momentum_h1",
            "seq_len": SEQ_LEN, "n_features": N_FEATURES,
            "patience": PATIENCE, "weighted_sampler": False,
            "mean_f1_macro": round(float(np.mean(f1s)), 4),
            "std_f1_macro":  round(float(np.std(f1s)),  4),
            "best_fold": best_fold, "best_f1_macro": round(best_f1, 4),
            "final_epochs": avg_epochs, "folds": all_metrics,
        }, f, indent=2, default=str)

    sep = "=" * 65
    print(f"\n{sep}")
    print(f"  LSTM H1 TRAINING SELESAI — {run_id}")
    print(f"{sep}")
    print(f"  Device      : {DEVICE}")
    print(f"  Best fold   : {best_fold} (F1={best_f1:.4f})")
    print(f"  Mean F1     : {np.mean(f1s):.4f} +/- {np.std(f1s):.4f}")
    print(f"  Final epochs: {avg_epochs}")
    print(f"  Model       : {model_path}")
    print(f"{sep}")
    print(f"\n  {'Fold':>4}  {'F1-mac':>7}  {'LONG':>7}  {'FLAT':>7}  {'SHORT':>7}  {'Epoch':>6}")
    print(f"  {'-'*50}")
    for m in all_metrics:
        print(f"  {m['fold']:>4}  {m['f1_macro']:>7.4f}  {m['f1_LONG']:>7.4f}  "
              f"{m['f1_FLAT']:>7.4f}  {m['f1_SHORT']:>7.4f}  {m['best_epoch']:>6}")
    print()


if __name__ == "__main__":
    main()
