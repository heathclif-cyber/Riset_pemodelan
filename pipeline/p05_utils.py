"""
pipeline/p05_utils.py — SequenceDataset shared antara 05_train_lstm dan 06_ensemble
"""
import numpy as np
import torch
from torch.utils.data import Dataset

from config import LSTM_SEQ_LEN


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
