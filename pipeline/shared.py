"""
pipeline/shared.py — Shared utilities untuk pipeline training dan backtest
"""
import numpy as np
import torch
from torch.utils.data import Dataset

from config import LSTM_SEQ_LEN, N_FOLDS, PURGE_GAP_BARS


def build_purged_folds(n: int, n_folds: int = N_FOLDS, purge: int = PURGE_GAP_BARS) -> list:
    """
    Build expanding-window folds with purging on both sides.

    Data is split into n_folds+1 equal chunks. Fold k trains on chunks [0..k-1],
    tests on chunk k. The last `purge` bars of training and the first `purge`
    bars of testing are removed to prevent leakage through rolling features.

    This is used by LGBM, LSTM, TP/SL regressor, and backtest — guaranteeing
    consistent fold boundaries across all models.
    """
    splits = np.array_split(np.arange(n), n_folds + 1)
    folds = []
    for k in range(1, n_folds + 1):
        train_raw = np.concatenate(splits[:k])
        test_raw = splits[k]
        train_idx = train_raw[:-purge] if len(train_raw) > purge else train_raw
        test_idx = test_raw[purge:] if len(test_raw) > purge else test_raw
        folds.append((train_idx, test_idx))
    return folds


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
