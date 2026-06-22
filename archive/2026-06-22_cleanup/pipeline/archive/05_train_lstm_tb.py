"""
pipeline/05_train_lstm_tb.py — LSTM dengan Triple Barrier labels.
Goal: menghasilkan prediksi decorrelated dari GBM tb_widyawardhana_v3.

Label   : Triple Barrier (TP=2×ATR, SL=1.5×ATR, max_hold=36) → WIN/LOSS/NEUTRAL
Fitur   : 18 fitur tb_widyawardhana_v3 (sama dengan LGBM)
Arsitek : TradingLSTM (ManualLSTMCell), hidden=96, 2 layers, dropout=0.45
Training: 2020-2025, purged CV 8 fold, RobustScaler per-fold
Output  : models/runs/tb_lstm_v1/

Usage:
  python pipeline/05_train_lstm_tb.py
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
from core.models import TradingLSTM, save_lstm
from core.utils import setup_logger
from pipeline.shared import build_purged_folds

logger = setup_logger("05_lstm_tb")
try:
    import torch_directml
    DEVICE = torch_directml.device()
    logger.info("Device: DirectML (AMD GPU)")
except Exception:
    DEVICE = torch.device("cpu")
    logger.info("Device: CPU (DirectML not available)")

RUN_NAME = "tb_lstm_v1"
OUT_DIR  = MODEL_DIR / "runs" / RUN_NAME
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEQ_LEN   = 48
HIDDEN    = 96
N_LAYERS  = 2
DROPOUT   = 0.45
BATCH     = 64
EPOCHS    = 200
PATIENCE  = 25
LR        = 0.0007
WD        = 2e-4
STEP      = 2        # 2x faster, 50% overlap reduction

TP_MULT  = TP_SL_FALLBACK_TP      # 2.0
SL_MULT  = TP_SL_FALLBACK_SL      # 1.5
TB_HOLD  = MAX_HOLDING_BARS       # 36
LM       = {"SHORT": 0, "FLAT": 1, "LONG": 2}

# Fitur = hasil multistage selection (41 features)
with open(MODEL_DIR / "runs/tb_lstm_v1/tb_lstm_v1_keep_features.json") as f:
    FEAT_COLS = json.load(f)

logger.info(f"Features: {len(FEAT_COLS)} (multistage IC selection from 109 FEATURE_COLS_V3 pool)")


def compute_tb_label(close, high, low, atr, i, n):
    """
    Triple Barrier label untuk bar i — single direction.
    Entry at close[i]: upper=+2xATR, lower=-1.5xATR, max_hold=36.
    Cek barrier mana yang kena duluan dari perspektif netral.
    Returns 0=SHORT, 1=FLAT(timeout), 2=LONG
    """
    entry  = close[i]
    upper  = entry + TP_MULT * atr[i]   # +2×ATR
    lower  = entry - SL_MULT * atr[i]   # -1.5×ATR
    end    = min(i + TB_HOLD, n - 1)

    for j in range(i + 1, end + 1):
        if high[j] >= upper:
            return 2   # LONG — upper hit first
        if low[j] <= lower:
            return 0   # SHORT — lower hit first
    return 1           # FLAT — timeout, price tidak bergerak jauh


class SeqDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.int64))

    def __len__(self): return len(self.X)

    def __getitem__(self, idx): return self.X[idx], self.y[idx]


def fit_scaler(X):
    n, s, f = X.shape
    sc = RobustScaler()
    sc.fit(X.reshape(-1, f))
    return sc


def scale_X(X, sc):
    n, s, f = X.shape
    return sc.transform(X.reshape(-1, f)).reshape(n, s, f).astype(np.float32)


def load_sequences(coins):
    """Build sequences per-coin with Triple Barrier labels."""
    X_all, y_all, ts_all = [], [], []

    for coin in coins:
        path = LABEL_DIR / f"{coin}_features_v3.parquet"
        if not path.exists():
            logger.warning(f"{coin}: not found, skip"); continue

        df = pd.read_parquet(path).sort_index()
        df = df[df.index < TRAIN_CUTOFF_DATE]

        if "label" not in df.columns:
            logger.warning(f"{coin}: no label, skip"); continue

        # Features
        avail = [c for c in FEAT_COLS if c in df.columns]
        df_feat = df[avail].ffill().fillna(0.0)

        # OHLCV for Triple Barrier labeling
        close = df["close"].values.astype(np.float64)
        high  = df["high"].values.astype(np.float64)
        low   = df["low"].values.astype(np.float64)
        atr   = df["atr_14_h1"].values.astype(np.float64)
        n     = len(df)

        # Fill NaN in high/low
        high = np.where(np.isnan(high), close, high)
        low  = np.where(np.isnan(low), close, low)

        X_c = df_feat.values.astype(np.float32)
        ts_c = df.index.astype(np.int64).values

        for i in range(SEQ_LEN - 1, n - TB_HOLD, STEP):
            if i + TB_HOLD >= n: break
            label = compute_tb_label(close, high, low, atr, i, n)

            X_all.append(X_c[i - SEQ_LEN + 1 : i + 1])
            y_all.append(label)
            ts_all.append(ts_c[i])

    if not X_all:
        raise ValueError("No sequences built")

    X  = np.stack(X_all)
    y  = np.array(y_all, dtype=np.int64)
    ts = np.array(ts_all, dtype=np.int64)
    order = np.argsort(ts)
    # Convert to pandas DatetimeIndex for build_purged_folds
    ts_idx = pd.DatetimeIndex([pd.Timestamp(t) for t in ts[order]])
    return X[order], y[order], ts_idx, avail


# ═══════════════════════════════════════════════════════════════════════════════
# Load data
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print(f"  LSTM Triple Barrier Training")
print(f"  Features: {len(FEAT_COLS)} | Seq={SEQ_LEN} | Hidden={HIDDEN} | "
      f"Layers={N_LAYERS}")
print(f"  Period: 2020-01-01 – {TRAIN_CUTOFF_DATE} | {len(TRAINING_COINS)} coins")
print(f"{'='*60}")

X, y, ts_idx, feat_cols_used = load_sequences(TRAINING_COINS)
n_samples, seq_len, n_feat = X.shape

unq, cnt = np.unique(y, return_counts=True)
dist = {u: c for u, c in zip(unq.tolist(), cnt.tolist())}
print(f"\n[1/2] Data loaded: {n_samples:,} sequences, {n_feat} features, seq={seq_len}")
print(f"  Label distribution: SHORT={dist.get(0,0):,} FLAT={dist.get(1,0):,} "
      f"LONG={dist.get(2,0):,}")
print(f"  Class ratio: {dist.get(0,0)/n_samples*100:.0f}%/{dist.get(1,0)/n_samples*100:.0f}%/"
      f"{dist.get(2,0)/n_samples*100:.0f}%")

# ═══════════════════════════════════════════════════════════════════════════════
# Training
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n[2/2] Purged CV training ({N_FOLDS} folds)...")

folds = build_purged_folds(ts_idx, N_FOLDS, PURGE_GAP_BARS)
cv_results = []
best_loss  = float("inf")
best_state = None
all_best_epochs = []

for fold_idx, (train_idx, val_idx) in enumerate(folds):
    if len(val_idx) < 20: continue

    X_tr, y_tr = X[train_idx], y[train_idx]
    X_te, y_te = X[val_idx],  y[val_idx]

    # Fit scaler per-fold
    scaler = fit_scaler(X_tr)
    X_tr_s = scale_X(X_tr, scaler)
    del X_tr; gc.collect()
    X_te_s = scale_X(X_te, scaler)
    del X_te; gc.collect()

    train_ds = SeqDataset(X_tr_s, y_tr)
    test_ds  = SeqDataset(X_te_s, y_te)
    train_ld = DataLoader(train_ds, batch_size=BATCH, shuffle=True, num_workers=0)
    test_ld  = DataLoader(test_ds,  batch_size=BATCH, shuffle=False, num_workers=0)

    # Class weights
    classes, counts = np.unique(y_tr, return_counts=True)
    total = len(y_tr)
    cw_dict = {c: total / (len(classes) * cnt) for c, cnt in zip(classes, counts)}
    cw = torch.tensor([cw_dict.get(i, 1.0) for i in range(3)], dtype=torch.float32).to(DEVICE)

    model     = TradingLSTM(n_feat, HIDDEN, N_LAYERS, DROPOUT, use_native=True).to(DEVICE)
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
                xb = xb.to(DEVICE)
                p = model(xb).argmax(dim=1).cpu().numpy()
                preds.extend(p); labels.extend(yb.numpy())

        f1 = f1_score(labels, preds, average="macro", zero_division=0)
        if f1 > best_f1:
            best_f1 = f1; best_st = {k: v.cpu() for k, v in model.state_dict().items()}
            patience_ct = 0; best_ep = epoch
        else:
            patience_ct += 1
            if patience_ct >= PATIENCE:
                break

        if epoch % 10 == 0 or epoch == 1:
            acc = np.mean(np.array(preds) == np.array(labels))
            logger.info(f"[F{fold_idx+1}] E{epoch:>3} | F1={f1:.4f} Acc={acc:.4f} "
                        f"BestF1={best_f1:.4f}")

    model.load_state_dict(best_st); model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for xb, yb in test_ld:
            preds.extend(model(xb).argmax(dim=1).cpu().numpy())
            labels.extend(yb.numpy())

    f1_val = f1_score(labels, preds, average="macro", zero_division=0)
    acc_val = np.mean(np.array(preds) == np.array(labels))
    cv_results.append({"fold": fold_idx+1, "f1": round(float(f1_val),4),
                       "acc": round(float(acc_val),4), "best_epoch": best_ep})
    all_best_epochs.append(best_ep)
    logger.info(f"  Fold {fold_idx+1} FINAL: F1={f1_val:.4f} Acc={acc_val:.4f} "
                f"BestEpoch={best_ep}")

    fold_loss = 1.0 - f1_val
    if fold_loss < best_loss:
        best_loss = fold_loss
        best_state = best_st

# Final model
print(f"\n  CV Results: F1 avg={np.mean([r['f1'] for r in cv_results]):.4f} "
      f"+/-{np.std([r['f1'] for r in cv_results]):.4f} "
      f"Acc avg={np.mean([r['acc'] for r in cv_results]):.4f}")
print(f"  Retraining final model on 100% data...")

scaler_all = fit_scaler(X)
X_s = scale_X(X, scaler_all)
del X; gc.collect()

final_model = TradingLSTM(n_feat, HIDDEN, N_LAYERS, DROPOUT, use_native=True).to(DEVICE)
n_epochs_final = int(np.mean(all_best_epochs))
classes, counts = np.unique(y, return_counts=True)
total = len(y)
cw_dict = {c: total / (len(classes) * cnt) for c, cnt in zip(classes, counts)}
cw = torch.tensor([cw_dict.get(i, 1.0) for i in range(3)], dtype=torch.float32).to(DEVICE)
criterion = nn.CrossEntropyLoss(weight=cw)
optimizer = torch.optim.Adam(final_model.parameters(), lr=LR, weight_decay=WD, foreach=False)

full_ds = SeqDataset(X_s, y)
full_ld = DataLoader(full_ds, batch_size=BATCH, shuffle=True, num_workers=0)

for epoch in range(1, n_epochs_final + 1):
    final_model.train()
    for xb, yb in full_ld:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad()
        loss = criterion(final_model(xb), yb)
        loss.backward()
        nn.utils.clip_grad_norm_(final_model.parameters(), 1.0)
        optimizer.step()

final_model.eval()

# ═══════════════════════════════════════════════════════════════════════════════
# Save
# ═══════════════════════════════════════════════════════════════════════════════
print("\nSaving model...")
save_lstm(final_model, OUT_DIR / "lstm.pt")
joblib.dump(scaler_all, OUT_DIR / "lstm_scaler.pkl")
with open(OUT_DIR / f"{RUN_NAME}_features.json", "w") as f:
    json.dump(feat_cols_used, f, indent=2)

meta = {
    "run_name": RUN_NAME, "labels": "Triple Barrier",
    "label_params": {"tp_mult": TP_MULT, "sl_mult": SL_MULT, "max_hold": TB_HOLD},
    "architecture": f"TradingLSTM(n_feat={n_feat}, hidden={HIDDEN}, layers={N_LAYERS}, dropout={DROPOUT})",
    "seq_len": SEQ_LEN, "batch_size": BATCH,
    "n_samples": n_samples, "n_features": n_feat,
    "label_dist": dist,
    "cv_results": cv_results,
    "cv_f1_mean": round(float(np.mean([r['f1'] for r in cv_results])), 4),
    "cv_f1_std": round(float(np.std([r['f1'] for r in cv_results])), 4),
    "training_period": f"2020-01-01 – {TRAIN_CUTOFF_DATE}",
    "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
}
with open(OUT_DIR / f"{RUN_NAME}_meta.json", "w") as f:
    json.dump(meta, f, indent=2)

print(f"\n{'='*60}")
print(f"  LSTM TB saved -> {OUT_DIR}")
print(f"  CV F1: {meta['cv_f1_mean']:.4f} +/- {meta['cv_f1_std']:.4f}")
print(f"  Random baseline F1: 0.333")
print(f"{'='*60}")
