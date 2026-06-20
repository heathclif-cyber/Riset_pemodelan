"""
pipeline/05_train_lstm_macro_v1.py — LSTM macro + temporal dynamics
Goal: Complement ic32_regime_v1 LGBM dengan macro regime + H4 temporal context.

Label   : Triple Barrier (TP=2xATR, SL=1.5xATR, max_hold=36)
Features: 7 IC-validated (IC >= 0.02, |t| >= 2.0, marginal IC >= 0.01)
          OHLCV temporal: cvd_slope_h4, ofi_h4_delta, ema_50_slope_h4,
                          ema_21_slope_h4, cvd_momentum_adv
          Macro (T-1 lag): tlt_ret_5d_ff, vix_z20
Arch    : VectorizedLSTM — GPU-optimized via pre-computed input projections
          hidden=64, 2 layers, dropout=0.35, seq_len=32
Training: 2020-01-01 - TRAIN_CUTOFF_DATE, purged CV 8 fold, RobustScaler per-fold
Output  : models/runs/tb_lstm_macro_v1/

Usage:
  python pipeline/05_train_lstm_macro_v1.py
"""
import gc, json, sys, warnings
from pathlib import Path
from datetime import datetime

warnings.filterwarnings("ignore", message="The operator 'aten::lerp")
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Dataset
import joblib

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from config import *
from core.utils import setup_logger
from pipeline.shared import build_purged_folds


# ── GPU-optimized LSTM for DirectML ───────────────────────────────────────────
# Pre-computes ALL input projections in one big matmul (batch×T, feat)@(feat, 4H)
# instead of T separate (batch, feat)@(feat, 4H) calls.
# Reduces GPU dispatch overhead from 2T → T+1 per layer.

class _VLSTMCell(nn.Module):
    """One LSTM layer with vectorized input projection."""
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.H = hidden_size
        k = hidden_size ** -0.5
        self.W_ih = nn.Parameter(torch.empty(4 * hidden_size, input_size).uniform_(-k, k))
        self.W_hh = nn.Parameter(torch.empty(4 * hidden_size, hidden_size).uniform_(-k, k))
        self.b    = nn.Parameter(torch.zeros(4 * hidden_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, in_size)  →  output: (B, T, H)
        Pre-computes ih = x @ W_ih.T + b for ALL timesteps at once.
        """
        B, T, _ = x.shape
        dev = x.device
        # One big matmul for input → (B, T, 4H)
        ih = (x.reshape(B * T, -1) @ self.W_ih.t() + self.b).reshape(B, T, 4 * self.H)

        h = torch.zeros(B, self.H, device=dev)
        c = torch.zeros(B, self.H, device=dev)
        out = []
        for t in range(T):
            gates = ih[:, t, :] + h @ self.W_hh.t()   # only hh is per-step
            ig, fg, gg, og = gates.chunk(4, dim=1)
            ig = torch.sigmoid(ig); fg = torch.sigmoid(fg)
            gg = torch.tanh(gg);   og = torch.sigmoid(og)
            c  = fg * c + ig * gg
            h  = og * torch.tanh(c)
            out.append(h)
        return torch.stack(out, dim=1)  # (B, T, H)


class VectorizedLSTM(nn.Module):
    """Multi-layer LSTM using vectorized input projection — GPU-optimized for DirectML."""
    def __init__(self, n_features: int, hidden: int = 64, n_layers: int = 2,
                 dropout: float = 0.35, num_classes: int = 3):
        super().__init__()
        self.layers = nn.ModuleList([
            _VLSTMCell(n_features if i == 0 else hidden, hidden)
            for i in range(n_layers)
        ])
        self.drop = nn.Dropout(dropout) if dropout > 0 and n_layers > 1 else nn.Identity()
        self.norm = nn.LayerNorm(hidden)
        self.fc   = nn.Linear(hidden, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = self.drop(x)
        last = self.norm(x[:, -1, :])
        last = self.drop(last)
        return self.fc(last)

logger = setup_logger("05_lstm_macro_v1")
try:
    import torch_directml
    DEVICE = torch_directml.device()
    logger.info("Device: DirectML (AMD GPU)")
except Exception:
    DEVICE = torch.device("cpu")
    logger.info("Device: CPU (DirectML not available)")

RUN_NAME = "tb_lstm_macro_v1"
OUT_DIR  = MODEL_DIR / "runs" / RUN_NAME
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEQ_LEN  = 32     # 32 H1 bars (~8 H4 periods)
HIDDEN   = 64
N_LAYERS = 2
DROPOUT  = 0.35
BATCH    = 128    # larger batch = fewer steps per epoch
EPOCHS   = 200
PATIENCE = 25
LR       = 0.0007
WD       = 2e-4
STEP     = 4      # skip 4 bars between sequences (reduces dataset ~50%)

TP_MULT = TP_SL_FALLBACK_TP   # 2.0
SL_MULT = TP_SL_FALLBACK_SL   # 1.5
TB_HOLD = MAX_HOLDING_BARS     # 36

MACRO_DIR = ROOT / "data" / "macro"

# 7 IC-validated features (IC>=0.02, |t|>=2.0, marginal IC>=0.01)
OHLCV_FEATS = ["cvd_slope_h4", "ofi_h4_delta", "ema_50_slope_h4",
               "ema_21_slope_h4", "cvd_momentum_adv"]
MACRO_FEATS = ["tlt_ret_5d_ff", "vix_z20"]
FEAT_COLS   = OHLCV_FEATS + MACRO_FEATS


def _norm_daily_index(df: pd.DataFrame) -> pd.DataFrame:
    idx = pd.to_datetime(df.index)
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    else:
        idx = idx.tz_convert("UTC")
    idx = idx.normalize().astype("datetime64[us, UTC]")
    df = df.copy(); df.index = idx
    return df[~df.index.duplicated(keep="last")].sort_index()


def load_macro_daily() -> pd.DataFrame:
    """Load TLT/VIX from macro_cross_asset, derive tlt_ret_5d_ff and vix_z20, T-1 lag."""
    p = MACRO_DIR / "macro_cross_asset.parquet"
    dm = pd.read_parquet(p)
    dm = _norm_daily_index(dm)

    # VIX z-score vs 20-day rolling (ffill weekends before rolling)
    vix_f = dm["vix_close"].ffill()
    dm["vix_z20"] = (
        (vix_f - vix_f.rolling(20, min_periods=10).mean()) /
        (vix_f.rolling(20, min_periods=10).std() + 1e-9)
    )

    # TLT 5-day return (ffill weekends)
    dm["tlt_ret_5d_ff"] = dm["tlt_ret_5d"].ffill()

    out = dm[["tlt_ret_5d_ff", "vix_z20"]].copy()
    # T-1 lag: macro value known at end of day T-1
    out = out.shift(1)
    # Resample to ensure daily index
    out = out.resample("1D").last().ffill()
    return out


def compute_tb_label(close, high, low, atr, i, n):
    """Triple Barrier: 0=SHORT, 1=FLAT, 2=LONG"""
    entry = close[i]
    upper = entry + TP_MULT * atr[i]
    lower = entry - SL_MULT * atr[i]
    end   = min(i + TB_HOLD, n - 1)
    for j in range(i + 1, end + 1):
        if high[j] >= upper: return 2
        if low[j]  <= lower: return 0
    return 1


class SeqDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.int64))
    def __len__(self):       return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]


def fit_scaler(X):
    n, s, f = X.shape
    sc = RobustScaler()
    sc.fit(X.reshape(-1, f))
    return sc

def scale_X(X, sc):
    n, s, f = X.shape
    return sc.transform(X.reshape(-1, f)).reshape(n, s, f).astype(np.float32)


def load_sequences(coins, macro_daily: pd.DataFrame):
    """Build sequences per-coin, merging macro daily to H1."""
    X_all, y_all, ts_all = [], [], []

    for coin in coins:
        path = LABEL_DIR / f"{coin}_features_v3.parquet"
        if not path.exists():
            logger.warning(f"{coin}: not found, skip"); continue

        df = pd.read_parquet(path).sort_index()
        cutoff = pd.Timestamp(TRAIN_CUTOFF_DATE)
        if cutoff.tzinfo is None:
            cutoff = cutoff.tz_localize("UTC")
        df = df[df.index < cutoff].copy()

        if len(df) < SEQ_LEN + TB_HOLD + 100:
            logger.warning(f"{coin}: too few rows, skip"); continue

        # Merge macro (T-1 lag already applied)
        idx_date = df.index.normalize().astype("datetime64[us, UTC]")
        macro_aligned = macro_daily.reindex(idx_date, method="ffill")
        macro_aligned.index = df.index

        for feat in MACRO_FEATS:
            df[feat] = macro_aligned[feat].values if feat in macro_aligned.columns else np.nan

        # Forward-fill + zero-fill features
        avail = [c for c in FEAT_COLS if c in df.columns]
        df[avail] = df[avail].ffill().fillna(0.0)

        close = df["close"].values.astype(np.float64)
        high  = df["high"].values.astype(np.float64)
        low   = df["low"].values.astype(np.float64)
        atr   = df["atr_14_h1"].values.astype(np.float64)
        n     = len(df)

        high = np.where(np.isnan(high), close, high)
        low  = np.where(np.isnan(low),  close, low)
        atr  = np.where(np.isnan(atr) | (atr <= 0), 1e-8, atr)

        X_c  = df[avail].values.astype(np.float32)
        ts_c = df.index.astype(np.int64).values

        for i in range(SEQ_LEN - 1, n - TB_HOLD, STEP):
            if i + TB_HOLD >= n: break
            label = compute_tb_label(close, high, low, atr, i, n)
            X_all.append(X_c[i - SEQ_LEN + 1 : i + 1])
            y_all.append(label)
            ts_all.append(ts_c[i])

        logger.info(f"  {coin}: {len(df):,} bars")

    if not X_all:
        raise ValueError("No sequences built — check FEAT_COLS vs parquet columns")

    X  = np.stack(X_all)
    y  = np.array(y_all, dtype=np.int64)
    ts = np.array(ts_all, dtype=np.int64)
    order = np.argsort(ts)
    ts_idx = pd.DatetimeIndex([pd.Timestamp(t) for t in ts[order]])
    return X[order], y[order], ts_idx, avail


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print(f"  LSTM Macro+Temporal v1 Training")
print(f"  Features ({len(FEAT_COLS)}): {FEAT_COLS}")
print(f"  Seq={SEQ_LEN} | Hidden={HIDDEN} | Layers={N_LAYERS}")
print(f"  Period: 2020-01-01 - {TRAIN_CUTOFF_DATE} | {len(TRAINING_COINS)} coins")
print(f"{'='*60}")

print("\nLoading macro daily features...")
macro_daily = load_macro_daily()
print(f"  Macro range: {macro_daily.index.min().date()} -> {macro_daily.index.max().date()}")
print(f"  Cols: {list(macro_daily.columns)}")

print(f"\nBuilding sequences ({len(TRAINING_COINS)} coins)...")
X, y, ts_idx, feat_cols_used = load_sequences(TRAINING_COINS, macro_daily)
n_samples, seq_len, n_feat = X.shape

unq, cnt = np.unique(y, return_counts=True)
dist = {int(u): int(c) for u, c in zip(unq, cnt)}
print(f"\n[1/2] Sequences built: {n_samples:,} seqs, {n_feat} features, seq_len={seq_len}")
print(f"  Label dist: SHORT={dist.get(0,0):,} FLAT={dist.get(1,0):,} LONG={dist.get(2,0):,}")
print(f"  Class ratio: SHORT={dist.get(0,0)/n_samples*100:.0f}% "
      f"FLAT={dist.get(1,0)/n_samples*100:.0f}% "
      f"LONG={dist.get(2,0)/n_samples*100:.0f}%")
print(f"  Features used: {feat_cols_used}")

# ═══════════════════════════════════════════════════════════════════════════════
# Purged CV Training
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n[2/2] Purged CV training ({N_FOLDS} folds)...")

folds = build_purged_folds(ts_idx, N_FOLDS, PURGE_GAP_BARS)
cv_results       = []
best_loss        = float("inf")
best_state       = None
all_best_epochs  = []

for fold_idx, (train_idx, val_idx) in enumerate(folds):
    if len(val_idx) < 20: continue

    X_tr, y_tr = X[train_idx], y[train_idx]
    X_te, y_te = X[val_idx],   y[val_idx]

    scaler = fit_scaler(X_tr)
    X_tr_s = scale_X(X_tr, scaler); del X_tr; gc.collect()
    X_te_s = scale_X(X_te, scaler); del X_te; gc.collect()

    train_ds = SeqDataset(X_tr_s, y_tr)
    test_ds  = SeqDataset(X_te_s, y_te)
    train_ld = DataLoader(train_ds, batch_size=BATCH, shuffle=True,  num_workers=0)
    test_ld  = DataLoader(test_ds,  batch_size=BATCH, shuffle=False, num_workers=0)

    classes, counts = np.unique(y_tr, return_counts=True)
    total = len(y_tr)
    cw_dict = {c: total / (len(classes) * cnt) for c, cnt in zip(classes, counts)}
    cw = torch.tensor([cw_dict.get(i, 1.0) for i in range(3)],
                      dtype=torch.float32).to(DEVICE)

    model     = VectorizedLSTM(n_feat, HIDDEN, N_LAYERS, DROPOUT).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=cw)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD, foreach=False)

    best_f1, best_st, patience_ct, best_ep = -1.0, None, 0, 1

    for epoch in range(1, EPOCHS + 1):
        model.train()
        for xb, yb in train_ld:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        model.eval()
        preds, labels = [], []
        with torch.no_grad():
            for xb, yb in test_ld:
                p = model(xb.to(DEVICE)).argmax(dim=1).cpu().numpy()
                preds.extend(p); labels.extend(yb.numpy())

        f1 = f1_score(labels, preds, average="macro", zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_st  = {k: v.cpu() for k, v in model.state_dict().items()}
            patience_ct = 0; best_ep = epoch
        else:
            patience_ct += 1
            if patience_ct >= PATIENCE: break

        if epoch % 10 == 0 or epoch == 1:
            acc = np.mean(np.array(preds) == np.array(labels))
            logger.info(f"[F{fold_idx+1}] E{epoch:>3} | F1={f1:.4f} "
                        f"Acc={acc:.4f} BestF1={best_f1:.4f}")

    model.load_state_dict(best_st); model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for xb, yb in test_ld:
            preds.extend(model(xb.to(DEVICE)).argmax(dim=1).cpu().numpy())
            labels.extend(yb.numpy())

    f1_val  = f1_score(labels, preds, average="macro", zero_division=0)
    acc_val = np.mean(np.array(preds) == np.array(labels))
    cv_results.append({"fold": fold_idx+1, "f1": round(float(f1_val), 4),
                       "acc": round(float(acc_val), 4), "best_epoch": best_ep})
    all_best_epochs.append(best_ep)
    logger.info(f"  Fold {fold_idx+1} FINAL: F1={f1_val:.4f} "
                f"Acc={acc_val:.4f} BestEpoch={best_ep}")

    fold_loss = 1.0 - f1_val
    if fold_loss < best_loss:
        best_loss = fold_loss; best_state = best_st

    del model, X_tr_s, X_te_s; gc.collect()

cv_f1   = np.mean([r["f1"] for r in cv_results])
cv_std  = np.std([r["f1"]  for r in cv_results])
cv_acc  = np.mean([r["acc"] for r in cv_results])
print(f"\n  CV F1: {cv_f1:.4f} +/- {cv_std:.4f}  Acc: {cv_acc:.4f}")
print(f"  Random baseline: 0.333")

# ═══════════════════════════════════════════════════════════════════════════════
# Final retrain on 100% data
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n  Retraining final model ({int(np.mean(all_best_epochs))} epochs)...")

scaler_all = fit_scaler(X)
X_s = scale_X(X, scaler_all)
del X; gc.collect()

final_model = VectorizedLSTM(n_feat, HIDDEN, N_LAYERS, DROPOUT).to(DEVICE)
n_ep_final  = int(np.mean(all_best_epochs))

classes, counts = np.unique(y, return_counts=True)
total   = len(y)
cw_dict = {c: total / (len(classes) * cnt) for c, cnt in zip(classes, counts)}
cw      = torch.tensor([cw_dict.get(i, 1.0) for i in range(3)],
                        dtype=torch.float32).to(DEVICE)
criterion = nn.CrossEntropyLoss(weight=cw)
optimizer = torch.optim.Adam(final_model.parameters(), lr=LR, weight_decay=WD, foreach=False)

full_ds = SeqDataset(X_s, y)
full_ld = DataLoader(full_ds, batch_size=BATCH, shuffle=True, num_workers=0)

for epoch in range(1, n_ep_final + 1):
    final_model.train()
    for xb, yb in full_ld:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad()
        loss = criterion(final_model(xb), yb)
        loss.backward()
        nn.utils.clip_grad_norm_(final_model.parameters(), 1.0)
        optimizer.step()
    if epoch % 20 == 0:
        logger.info(f"  Final retrain epoch {epoch}/{n_ep_final}")

final_model.eval()

# ═══════════════════════════════════════════════════════════════════════════════
# Save
# ═══════════════════════════════════════════════════════════════════════════════
print("\nSaving...")
# VectorizedLSTM — save state dict directly (custom arch, not TradingLSTM-compatible)
torch.save(final_model.state_dict(), str(OUT_DIR / "lstm.pt"))
joblib.dump(scaler_all, OUT_DIR / "lstm_scaler.pkl")

with open(OUT_DIR / f"{RUN_NAME}_features.json", "w") as f:
    json.dump(feat_cols_used, f, indent=2)

meta = {
    "run_name":    RUN_NAME,
    "label_type":  "Triple Barrier",
    "label_params": {"tp_mult": TP_MULT, "sl_mult": SL_MULT, "max_hold": TB_HOLD},
    "architecture": f"VectorizedLSTM(n_feat={n_feat}, hidden={HIDDEN}, "
                    f"layers={N_LAYERS}, dropout={DROPOUT}, seq_len={SEQ_LEN})",
    "seq_len":      SEQ_LEN,
    "batch_size":   BATCH,
    "n_samples":    n_samples,
    "n_features":   n_feat,
    "feature_cols": feat_cols_used,
    "ohlcv_feats":  OHLCV_FEATS,
    "macro_feats":  MACRO_FEATS,
    "macro_lag":    "T-1 (daily macro shift(1) to avoid look-ahead)",
    "label_dist":   dist,
    "cv_results":   cv_results,
    "cv_f1_mean":   round(float(cv_f1), 4),
    "cv_f1_std":    round(float(cv_std), 4),
    "cv_acc_mean":  round(float(cv_acc), 4),
    "training_period": f"2020-01-01 - {TRAIN_CUTOFF_DATE}",
    "trained_at":   datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "integration_note": (
        "Intended as ic32 cascade complement — LONG/SHORT signal "
        "adjusts LGBM confidence by +/-0.05 (AGREE/DISAGREE with LGBM direction)"
    ),
}
with open(OUT_DIR / f"{RUN_NAME}_meta.json", "w") as f:
    json.dump(meta, f, indent=2)

print(f"\n{'='*60}")
print(f"  SAVED: {OUT_DIR}")
print(f"  CV F1:  {cv_f1:.4f} +/- {cv_std:.4f}")
print(f"  Random: 0.333")
print(f"  Gain:   {cv_f1 - 0.333:+.4f}")
print(f"{'='*60}")
