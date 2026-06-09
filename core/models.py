"""
core/models.py — Definisi arsitektur model dan fungsi load/save
Dipakai bersama oleh pipeline/05_train_lstm.py, pipeline/06_ensemble.py,
dan Swing_Trade9.6/ml/ml_signal.py

PENTING: Arsitektur TradingLSTM TIDAK BOLEH diubah tanpa retraining.
         n_features=100 (dinamis sesuai len(FEATURE_COLS_V3)), hidden_size=128, num_layers=2, dropout=0.3
"""

import pickle
from pathlib import Path

import torch
import torch.nn as nn

try:
    from config import FEATURE_COLS_V3
    DEFAULT_N_FEATURES = len(FEATURE_COLS_V3)
except ImportError:
    DEFAULT_N_FEATURES = 105  # Fallback to cascade_v4.1 feature count

# ─── TradingLSTM ─────────────────────────────────────────────────────────────

class _ManualLSTMCell(nn.Module):
    """
    LSTM cell manual — hanya pakai matmul + sigmoid + tanh.
    Menghindari _thnn_fused_lstm_cell yang tidak support DirectML/AMD GPU.
    """
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        k = hidden_size ** -0.5
        self.W_ih = nn.Parameter(torch.empty(4 * hidden_size, input_size).uniform_(-k, k))
        self.W_hh = nn.Parameter(torch.empty(4 * hidden_size, hidden_size).uniform_(-k, k))
        self.b    = nn.Parameter(torch.zeros(4 * hidden_size))

    def forward(self, x: torch.Tensor, hx: tuple) -> tuple:
        h, c = hx
        # Satu matmul gabungan: lebih efisien di GPU
        gates = x @ self.W_ih.t() + h @ self.W_hh.t() + self.b
        i, f, g, o = gates.chunk(4, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        g = torch.tanh(g)
        o = torch.sigmoid(o)
        c_new = f * c + i * g
        h_new = o * torch.tanh(c_new)
        return h_new, c_new


class _CellLSTM(nn.Module):
    """
    Multi-layer LSTM menggunakan _ManualLSTMCell.
    Kompatibel dengan DirectML (AMD RX 6600) karena hanya pakai basic ops.
    """
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers  = num_layers
        self.cells = nn.ModuleList([
            _ManualLSTMCell(input_size if i == 0 else hidden_size, hidden_size)
            for i in range(num_layers)
        ])
        self.drop = nn.Dropout(dropout) if dropout > 0 and num_layers > 1 else nn.Identity()

    def forward(self, x: torch.Tensor):
        # x: (batch, seq_len, input_size)
        B, T, _ = x.shape
        dev = x.device
        h = [torch.zeros(B, self.hidden_size).to(dev) for _ in self.cells]
        c = [torch.zeros(B, self.hidden_size).to(dev) for _ in self.cells]

        out_seq = []
        for t in range(T):
            inp = x[:, t, :]
            for i, cell in enumerate(self.cells):
                h[i], c[i] = cell(inp, (h[i], c[i]))
                inp = self.drop(h[i]) if i < self.num_layers - 1 else h[i]
            out_seq.append(h[-1])

        return torch.stack(out_seq, dim=1), None  # (batch, seq_len, hidden)


class TradingLSTM(nn.Module):
    """
    Unidirectional LSTM untuk prediksi sinyal trading multiclass.

    Input  : (batch, seq_len, n_features)
    Output : (batch, num_classes) — raw logits
    """

    def __init__(
        self,
        n_features:  int   = DEFAULT_N_FEATURES,
        hidden_size: int   = 128,
        num_layers:  int   = 2,
        dropout:     float = 0.3,
        num_classes: int   = 3,
        use_native:  bool  = False,
    ):
        super().__init__()
        self.use_native = use_native
        if use_native:
            self.lstm = nn.LSTM(
                input_size  = n_features,
                hidden_size = hidden_size,
                num_layers  = num_layers,
                dropout     = dropout if num_layers > 1 else 0.0,
                batch_first = True,
            )
        else:
            self.lstm = _CellLSTM(
                input_size  = n_features,
                hidden_size = hidden_size,
                num_layers  = num_layers,
                dropout     = dropout if num_layers > 1 else 0.0,
            )
        self.norm    = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        last   = self.norm(out[:, -1, :])
        last   = self.dropout(last)
        return self.fc(last)


# ─── Load / Save LSTM ────────────────────────────────────────────────────────

def save_lstm(model: TradingLSTM, path: Path) -> None:
    """Simpan state dict saja (bukan full checkpoint)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if getattr(model, "use_native", False):
        # Convert native state dict to manual state dict on-the-fly
        native_state = model.state_dict()
        manual_state = {}
        # Copy norm and fc weights directly
        for k in ["norm.weight", "norm.bias", "fc.weight", "fc.bias"]:
            if k in native_state:
                manual_state[k] = native_state[k]
        
        # Determine num_layers from native_state keys
        num_layers = sum(1 for k in native_state if k.startswith("lstm.weight_ih_l"))
        
        for l in range(num_layers):
            manual_state[f"lstm.cells.{l}.W_ih"] = native_state[f"lstm.weight_ih_l{l}"]
            manual_state[f"lstm.cells.{l}.W_hh"] = native_state[f"lstm.weight_hh_l{l}"]
            
            # Sum bias_ih and bias_hh for the manual cell
            b_ih = native_state[f"lstm.bias_ih_l{l}"]
            b_hh = native_state[f"lstm.bias_hh_l{l}"]
            manual_state[f"lstm.cells.{l}.b"] = b_ih + b_hh
            
        torch.save(manual_state, str(path))
    else:
        torch.save(model.state_dict(), str(path))


def load_lstm(
    path: Path,
    n_features:  int | None = None,
    hidden_size: int | None = None,
    num_layers:  int | None = None,
    dropout:     float = 0.3,
    num_classes: int   = 3,
    device: str        = "cpu",
) -> TradingLSTM:
    """Load LSTM dari state dict ke CPU. Auto-detect n_features dan hidden_size dari checkpoint."""
    state = torch.load(str(path), map_location="cpu", weights_only=False)
    # Auto-detect dari shape W_ih: [4*hidden, n_features]
    if n_features is None:
        n_features = state["lstm.cells.0.W_ih"].shape[1]
    if hidden_size is None:
        hidden_size = state["lstm.cells.0.W_ih"].shape[0] // 4
    if num_layers is None:
        num_layers = sum(1 for k in state if k.startswith("lstm.cells.") and k.endswith(".W_ih"))
    model = TradingLSTM(n_features, hidden_size, num_layers, dropout, num_classes)
    model.load_state_dict(state)
    model.eval()
    return model


# ─── Probability Calibrator ──────────────────────────────────────────────────

class ProbabilityCalibrator:
    """
    Kalibrasi probabilitas post-hoc untuk output ensemble.
    Difit pada validation set setelah ensemble selesai ditraining.

    Method "isotonic" (default) lebih fleksibel untuk distribusi
    non-Gaussian seperti trading signals. "platt" = Logistic Regression.

    Usage:
        cal = ProbabilityCalibrator()
        cal.fit(proba_val, y_val)        # proba: (n, 3), y: (n,) int
        cal_proba = cal.transform(proba) # output: (n, 3), setiap baris sum=1
        cal.save(path)
        cal = ProbabilityCalibrator.load(path)
    """

    def __init__(self, method: str = "isotonic"):
        self.method      = method
        self.calibrators = {}  # {class_idx: fitted_estimator}

    def fit(self, proba: "np.ndarray", y: "np.ndarray") -> None:
        import numpy as np
        from sklearn.isotonic import IsotonicRegression
        from sklearn.linear_model import LogisticRegression

        for c in range(proba.shape[1]):
            y_bin = (y == c).astype(int)
            if self.method == "isotonic":
                est = IsotonicRegression(out_of_bounds="clip")
                est.fit(proba[:, c], y_bin)
            else:
                est = LogisticRegression(C=1.0)
                est.fit(proba[:, c].reshape(-1, 1), y_bin)
            self.calibrators[c] = est

    def transform(self, proba: "np.ndarray") -> "np.ndarray":
        import numpy as np

        cal = np.zeros_like(proba)
        for c, est in self.calibrators.items():
            if self.method == "isotonic":
                cal[:, c] = est.predict(proba[:, c])
            else:
                cal[:, c] = est.predict_proba(proba[:, c].reshape(-1, 1))[:, 1]

        row_sum = cal.sum(axis=1, keepdims=True)
        row_sum = np.where(row_sum == 0, 1, row_sum)
        return cal / row_sum

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: Path) -> "ProbabilityCalibrator":
        with open(path, "rb") as f:
            return pickle.load(f)