"""
pipeline/05_train_lstm.py — Fase 5: LSTM Temporal Sequence Model
Purged Walk-Forward CV (8 fold, purge gap 5 bar)

Jalankan:
  python pipeline/05_train_lstm.py               # training coins
  python pipeline/05_train_lstm.py --all         # semua 20 koin
  python pipeline/05_train_lstm.py --run-id my_run

CATATAN: Lebih baik dijalankan di Google Colab dengan GPU T4.
Output: models/runs/{run_id}/lstm.pt, lstm_scaler.pkl, lstm_cv_results.json
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
    TRAINING_COINS, ALL_COINS, SYMBOL_MAP,
    LABEL_DIR, MODEL_DIR,
    N_FOLDS, PURGE_GAP_BARS,
    LABEL_MAP, LABEL_MAP_INV, NUM_CLASSES,
    LSTM_SEQ_LEN, LSTM_HIDDEN, LSTM_LAYERS, LSTM_DROPOUT,
    LSTM_EPOCHS, LSTM_PATIENCE, LSTM_BATCH_SIZE, LSTM_LR,
)
from core.models import TradingLSTM, save_lstm
from core.utils import setup_logger

logger = setup_logger("05_train_lstm")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NON_FEATURE_COLS = {"label"}


# ─── Data ────────────────────────────────────────────────────────────────────

def load_symbols(coins: list[str]) -> pd.DataFrame:
    frames = []
    for sym in coins:
        path = LABEL_DIR / f"{sym}_features_v3.parquet"
        if not path.exists():
            logger.warning(f"Skip {sym}: file tidak ada")
            continue
        df = pd.read_parquet(path)
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index, utc=True)
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        if "OB_price" in df.columns:
            df.drop(columns=["OB_price"], inplace=True)
        frames.append(df)
        logger.info(f"Loaded {sym}: {len(df):,} rows")
    combined = pd.concat(frames).sort_index()
    logger.info(f"Total: {len(combined):,} rows × {len(combined.columns)} cols")
    return combined


def preprocess(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    label_str = df["label"].astype(str)
    mask = label_str.isin(LABEL_MAP)
    if (~mask).sum():
        logger.warning(f"Drop {(~mask).sum():,} baris label tidak dikenal.")
        df = df[mask].copy()
        label_str = label_str[mask]
    y = label_str.map(LABEL_MAP).values.astype(np.int64)
    feat_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]
    df[feat_cols] = df[feat_cols].ffill().fillna(0)
    return df, y


def build_purged_folds(n: int) -> list[tuple[np.ndarray, np.ndarray]]:
    splits = np.array_split(np.arange(n), N_FOLDS + 1)
    folds  = []
    for k in range(1, N_FOLDS + 1):
        train_raw = np.concatenate(splits[:k])
        test_raw  = splits[k]
        train_idx = train_raw[:-PURGE_GAP_BARS] if len(train_raw) > PURGE_GAP_BARS else train_raw
        test_idx  = test_raw[PURGE_GAP_BARS:]   if len(test_raw) > PURGE_GAP_BARS  else test_raw
        folds.append((train_idx, test_idx))
    return folds


# ─── Dataset ─────────────────────────────────────────────────────────────────

class SequenceDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, seq_len: int = LSTM_SEQ_LEN):
        self.X       = torch.from_numpy(X.astype(np.float32))
        self.y       = torch.from_numpy(y.astype(np.int64))
        self.seq_len = seq_len
        self.indices = list(range(seq_len - 1, len(X)))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        end = self.indices[idx]
        return self.X[end - self.seq_len + 1: end + 1], self.y[end]

    def get_labels(self):
        return self.y[self.indices].numpy()


def build_sampler(labels: np.ndarray) -> WeightedRandomSampler:
    counts  = np.bincount(labels, minlength=NUM_CLASSES).astype(np.float64)
    counts  = np.where(counts == 0, 1, counts)
    class_w = 1.0 / counts
    sample_w = class_w[labels]
    return WeightedRandomSampler(
        weights=torch.from_numpy(sample_w).float(),
        num_samples=len(sample_w),
        replacement=True,
    )


def compute_class_weights(y: np.ndarray) -> torch.Tensor:
    counts  = np.bincount(y, minlength=NUM_CLASSES).astype(np.float64)
    counts  = np.where(counts == 0, 1, counts)
    weights = len(y) / (NUM_CLASSES * counts)
    return torch.tensor(weights, dtype=torch.float32).to(DEVICE)


# ─── Training ────────────────────────────────────────────────────────────────

def train_fold(
    X_tr: np.ndarray, y_tr: np.ndarray,
    X_te: np.ndarray, y_te: np.ndarray,
    fold_num: int,
) -> tuple[TradingLSTM, StandardScaler, dict]:
    logger.info(f"[Fold {fold_num}] Train={len(X_tr):,} | Test={len(X_te):,}")

    scaler   = StandardScaler()
    X_tr_sc  = scaler.fit_transform(X_tr)
    X_te_sc  = scaler.transform(X_te)

    train_ds = SequenceDataset(X_tr_sc, y_tr)
    test_ds  = SequenceDataset(X_te_sc, y_te)
    sampler  = build_sampler(train_ds.get_labels())

    train_loader = DataLoader(train_ds, batch_size=LSTM_BATCH_SIZE,
                              sampler=sampler, num_workers=0, pin_memory=False)
    test_loader  = DataLoader(test_ds,  batch_size=LSTM_BATCH_SIZE * 2,
                              shuffle=False, num_workers=0)

    n_features = X_tr.shape[1]
    model = TradingLSTM(n_features, LSTM_HIDDEN, LSTM_LAYERS, LSTM_DROPOUT).to(DEVICE)
    cw    = compute_class_weights(y_tr)
    criterion = nn.CrossEntropyLoss(weight=cw)
    optimizer = torch.optim.Adam(model.parameters(), lr=LSTM_LR)

    best_f1, best_state, patience_count = -1.0, None, 0
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
        all_preds, all_labels = [], []
        with torch.no_grad():
            for xb, yb in test_loader:
                preds = model(xb.to(DEVICE)).argmax(dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(yb.numpy())

        f1 = float(f1_score(all_labels, all_preds, average="macro", zero_division=0))
        if f1 > best_f1:
            best_f1, best_state, patience_count = f1, {k: v.cpu() for k, v in model.state_dict().items()}, 0
        else:
            patience_count += 1
            if patience_count >= LSTM_PATIENCE:
                logger.info(f"[Fold {fold_num}] Early stop epoch {epoch} (best F1={best_f1:.4f})")
                break

        if epoch % 5 == 0 or epoch == 1:
            logger.info(f"[Fold {fold_num}] Epoch {epoch:>2} | F1-macro={f1:.4f} | Best={best_f1:.4f}")

    model.load_state_dict(best_state)
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            preds = model(xb.to(DEVICE)).argmax(dim=1).cpu().numpy()
            all_preds.extend(preds); all_labels.extend(yb.numpy())

    f1_per = f1_score(all_labels, all_preds, average=None, zero_division=0, labels=[0, 1, 2])
    metrics = {
        "fold": fold_num, "n_train": len(X_tr), "n_test": len(X_te),
        "accuracy":    round(float(accuracy_score(all_labels, all_preds)), 4),
        "f1_macro":    round(best_f1, 4),
        "f1_weighted": round(float(f1_score(all_labels, all_preds, average="weighted", zero_division=0)), 4),
        "f1_SHORT":    round(float(f1_per[0]), 4),
        "f1_FLAT":     round(float(f1_per[1]), 4),
        "f1_LONG":     round(float(f1_per[2]), 4),
        "confusion_matrix": confusion_matrix(all_labels, all_preds, labels=[0, 1, 2]).tolist(),
    }
    logger.info(f"[Fold {fold_num}] F1-macro={best_f1:.4f} | LONG={f1_per[2]:.4f} SHORT={f1_per[0]:.4f}")
    return model, scaler, metrics


# ─── Main ─────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--run-id", default=None)
    return parser.parse_args()


def main():
    args   = parse_args()
    coins  = ALL_COINS if args.all else TRAINING_COINS
    run_id = args.run_id or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = MODEL_DIR / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Device: {DEVICE} | Run: {run_id}")

    df = load_symbols(coins)
    df, y = preprocess(df)
    feat_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]
    X = df[feat_cols].values.astype(np.float64)

    folds       = build_purged_folds(len(df))
    all_metrics = []
    best_model, best_scaler, best_f1, best_fold = None, None, -1.0, -1

    for fold_num, (tr_pos, te_pos) in enumerate(folds, 1):
        model, scaler, metrics = train_fold(
            X[tr_pos], y[tr_pos], X[te_pos], y[te_pos], fold_num
        )
        all_metrics.append(metrics)
        if metrics["f1_macro"] > best_f1:
            best_f1, best_model, best_scaler, best_fold = metrics["f1_macro"], model, scaler, fold_num

    # Simpan
    lstm_path   = run_dir / "lstm.pt"
    scaler_path = run_dir / "lstm_scaler.pkl"
    save_lstm(best_model, lstm_path)
    joblib.dump(best_scaler, scaler_path)
    # Copy ke root models/
    save_lstm(best_model, MODEL_DIR / "lstm_best.pt")
    joblib.dump(best_scaler, MODEL_DIR / "lstm_scaler.pkl")
    logger.info(f"Best model fold {best_fold} F1={best_f1:.4f} → {lstm_path}")

    # ★ v2: simpan feature_cols_v2.json (konsisten dengan 04_train_lgbm)
    feat_cols_path = MODEL_DIR / "feature_cols_v2.json"
    if not feat_cols_path.exists():
        with open(feat_cols_path, "w") as f:
            json.dump(feat_cols, f, indent=2)
        logger.info(f"Feature cols v2 ({len(feat_cols)}) → {feat_cols_path}")

    f1s = [m["f1_macro"] for m in all_metrics]
    cv_summary = {
        "run_id": run_id, "coins": coins, "n_features": X.shape[1],
        "n_folds": N_FOLDS, "purge_gap_bars": PURGE_GAP_BARS,
        "best_fold": best_fold, "best_f1_macro": round(best_f1, 4),
        "mean_f1_macro": round(float(np.mean(f1s)), 4),
        "std_f1_macro":  round(float(np.std(f1s)), 4),
        "feature_cols":  feat_cols, "folds": all_metrics,
    }
    cv_path = run_dir / "lstm_cv_results.json"
    with open(cv_path, "w") as f:
        json.dump(cv_summary, f, indent=2, default=str)
    with open(MODEL_DIR / "lstm_cv_results.json", "w") as f:
        json.dump(cv_summary, f, indent=2, default=str)

    sep = "=" * 60
    print(f"\n{sep}")
    print(f"  LSTM TRAINING SELESAI — {run_id}")
    print(f"{sep}")
    print(f"  Device     : {DEVICE}")
    print(f"  Best fold  : {best_fold} (F1-macro={best_f1:.4f})")
    print(f"  Mean F1    : {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")
    print(f"  Model      : {lstm_path}")
    print(f"{sep}\n")


if __name__ == "__main__":
    main()
