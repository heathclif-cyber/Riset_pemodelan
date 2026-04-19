"""
core/models.py — Definisi arsitektur model dan fungsi load/save
Dipakai bersama oleh pipeline/05_train_lstm.py, pipeline/06_ensemble.py,
dan Swing_Trade9.6/ml/ml_signal.py

PENTING: Arsitektur TradingLSTM TIDAK BOLEH diubah tanpa retraining.
         n_features=58, hidden_size=128, num_layers=2, dropout=0.3
"""

import json
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn


# ─── TradingLSTM ─────────────────────────────────────────────────────────────

class TradingLSTM(nn.Module):
    """
    Unidirectional LSTM untuk prediksi sinyal trading multiclass.

    Input  : (batch, seq_len, n_features)
    Output : (batch, num_classes) — raw logits

    State dict keys:
      lstm.weight_ih_l0, lstm.weight_hh_l0, lstm.bias_ih_l0, lstm.bias_hh_l0
      lstm.weight_ih_l1, lstm.weight_hh_l1, lstm.bias_ih_l1, lstm.bias_hh_l1
      norm.weight, norm.bias
      fc.weight, fc.bias
    """

    def __init__(
        self,
        n_features:  int   = 58,
        hidden_size: int   = 128,
        num_layers:  int   = 2,
        dropout:     float = 0.3,
        num_classes: int   = 3,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size    = n_features,
            hidden_size   = hidden_size,
            num_layers    = num_layers,
            dropout       = dropout if num_layers > 1 else 0.0,
            batch_first   = True,
            bidirectional = False,
        )
        self.norm    = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _  = self.lstm(x)
        last    = out[:, -1, :]
        last    = self.norm(last)
        last    = self.dropout(last)
        return self.fc(last)


# ─── Load / Save ─────────────────────────────────────────────────────────────

def save_lstm(model: TradingLSTM, path: Path) -> None:
    """Simpan state dict saja (bukan full checkpoint)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), str(path))


def load_lstm(
    path: Path,
    n_features:  int   = 58,
    hidden_size: int   = 128,
    num_layers:  int   = 2,
    dropout:     float = 0.3,
    num_classes: int   = 3,
    device: str        = "cpu",
) -> TradingLSTM:
    """Load LSTM dari state dict."""
    model = TradingLSTM(n_features, hidden_size, num_layers, dropout, num_classes)
    state = torch.load(str(path), map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


# ─── Model Registry ──────────────────────────────────────────────────────────

DEFAULT_REGISTRY = {
    "active": "ensemble_v1",
    "models": {
        "ensemble_v1": {
            "type":       "ensemble",
            "lgbm_path":  "lgbm_baseline.pkl",
            "lstm_path":  "lstm_best.pt",
            "scaler_path":"lstm_scaler.pkl",
            "meta_path":  "ensemble_meta.pkl",
            "n_features": 58,
            "lstm_hidden": 128,
            "lstm_layers": 2,
            "description": "LightGBM + LSTM stacking ensemble",
            "f1_macro":   0.5249,
        }
    }
}


def save_registry(registry: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(registry, f, indent=2)


def load_registry(path: Path) -> dict:
    if not path.exists():
        return DEFAULT_REGISTRY
    with open(path) as f:
        return json.load(f)
