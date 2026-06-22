"""
pipeline/05_train_lstm_widyawardhana_v1.py
LSTM untuk TB widyawardhana series — IC-validated 13 features.

Perbedaan dari tb_lstm_v1:
  - 13 fitur dari marginal IC test (step1-3), bukan 41 fitur
  - SEQ_LEN=32 (vs 48)
  - Hidden=64 (lebih kecil, proporsional dengan n_features)
  - Fitur dipilih berdasarkan marg_IC >= 0.015, stable 6/6 windows,
    dari 15 unique signal sources — dijamin orthogonal terhadap LGBM conf

Integration di inference:
  LGBM says LONG, LSTM says LONG  → hard_consensus → enter
  LGBM says LONG, LSTM says FLAT  → soft signal (pakai LGBM conf × 0.85)
  LGBM says LONG, LSTM says SHORT → FLIP → skip atau reverse
  (Logic sama seperti ic32 cascade tapi domain-matched ke TB labels)

Run name: tb_lstm_widyawardhana_v1

Usage:
  python pipeline/05_train_lstm_widyawardhana_v1.py
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
from sklearn.metrics import f1_score, classification_report
from torch.utils.data import DataLoader, Dataset
import joblib

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from config import (
    LABEL_DIR, TRAIN_CUTOFF_DATE, TRAINING_COINS, MODEL_DIR,
    N_FOLDS, PURGE_GAP_BARS,
    TP_SL_FALLBACK_TP, TP_SL_FALLBACK_SL, MAX_HOLDING_BARS,
)
from core.models import TradingLSTM, save_lstm
from core.utils import setup_logger
from pipeline.shared import build_purged_folds

logger = setup_logger("05_lstm_widyawardhana_v1")
try:
    import torch_directml
    DEVICE = torch_directml.device()
    logger.info("Device: DirectML (AMD GPU)")
except Exception:
    DEVICE = torch.device("cpu")
    logger.info("Device: CPU")

RUN_NAME = "tb_lstm_widyawardhana_v1"
OUT_DIR  = MODEL_DIR / "runs" / RUN_NAME
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Hyperparameters ────────────────────────────────────────────────────────────
SEQ_LEN  = 16
HIDDEN   = 64       # proporsional dengan 13 features (vs 96 untuk 41 feat)
N_LAYERS = 2
DROPOUT  = 0.40
BATCH    = 512
EPOCHS   = 200
PATIENCE = 25
LR       = 0.0007
WD       = 2e-4
STEP     = 2        # 50% stride overlap reduction
MAX_CW   = 5.0      # cap class weight — cegah FLAT 16x over-predict

TP_MULT = TP_SL_FALLBACK_TP   # 2.0
SL_MULT = TP_SL_FALLBACK_SL   # 1.5
TB_HOLD = MAX_HOLDING_BARS    # 36
LM      = {"SHORT": 0, "FLAT": 1, "LONG": 2}

# ── IC-validated features (marginal IC test, 15 unique sources) ───────────────
# Satu representatif per sumber — IC terkuat yang tersedia di features_v3.parquet.
# Sumber: Volatility (5), Volume (6), Structure (2), Pressure/Flow (3)
IC_FEATURES = [
    # Volatility — IC negatif kuat (expanding vol = trades lose)
    "atr_14_h4",           # marg_IC -0.086, IC_IR -11.8  (strongest vol signal)
    "atr_percent_h4",      # marg_IC -0.091, IC_IR -13.0
    "atr_percentile_h1",   # marg_IC -0.084, IC_IR -7.4
    # Volume — directional volume trajectory
    "buy_volume",          # marg_IC -0.050, IC_IR -4.2  (slope32 strongly negative)
    "sell_volume",         # marg_IC -0.053, IC_IR -4.2
    "vol_ratio_20",        # marg_IC +0.049, IC_IR +5.9  (relative volume vs 20-bar avg)
    "vol_spike_zscore",    # marg_IC +0.044, IC_IR +6.9  (spike detection)
    "vol_efficiency",      # marg_IC -0.053, IC_IR -6.1  (vol vs price move)
    # Structure
    "bars_since_BOS",      # marg_IC +0.028, IC_IR +3.3  (time since break of structure)
    "dist_liq_20x_long",   # marg_IC +0.062, IC_IR +3.3  (distance to liquidation cluster)
    # Pressure / Flow
    "effort_vs_result",    # marg_IC +0.042, IC_IR +6.7  (Wyckoff absorption)
    "spread_to_volume",    # marg_IC -0.024, IC_IR -13.6 (most stable: IC_IR -13.5)
    "vwdp_smooth",         # marg_IC +0.024, IC_IR +2.5
]

assert len(IC_FEATURES) == 13, f"Expect 13 features, got {len(IC_FEATURES)}"


# ── Helpers ────────────────────────────────────────────────────────────────────
def compute_tb_label(close, high, low, atr, i, n):
    """Triple Barrier label bar i: LONG=2, FLAT=1, SHORT=0."""
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
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]


def scale(X_tr, X_val):
    n, s, f = X_tr.shape
    sc = RobustScaler()
    sc.fit(X_tr.reshape(-1, f))
    Xt = sc.transform(X_tr.reshape(-1, f)).reshape(n, s, f).astype(np.float32)
    Xv = sc.transform(X_val.reshape(-1, X_val.shape[1]*f).reshape(-1, f)
                      ).reshape(X_val.shape[0], s, f).astype(np.float32)
    return Xt, Xv, sc


def load_sequences(coins):
    """Build SEQ_LEN-bar sequences with TB labels for all coins."""
    X_all, y_all, ts_all = [], [], []

    for coin in coins:
        path = LABEL_DIR / f"{coin}_features_v3.parquet"
        if not path.exists():
            logger.warning(f"{coin}: skip (not found)"); continue

        df = pd.read_parquet(path).sort_index()
        df = df[df.index < TRAIN_CUTOFF_DATE]

        avail = [c for c in IC_FEATURES if c in df.columns]
        if len(avail) < len(IC_FEATURES):
            missing = set(IC_FEATURES) - set(avail)
            logger.warning(f"{coin}: missing features {missing}")

        df_feat = df[avail].ffill().fillna(0.0)
        close = df["close"].values.astype(np.float64)
        high  = np.where(np.isnan(df["high"].values), close, df["high"].values)
        low   = np.where(np.isnan(df["low"].values),  close, df["low"].values)
        atr   = df["atr_14_h1"].values.astype(np.float64)
        n     = len(df)
        X_c   = df_feat.values.astype(np.float32)
        ts_c  = df.index.astype(np.int64).values

        for i in range(SEQ_LEN - 1, n - TB_HOLD, STEP):
            if i + TB_HOLD >= n: break
            label = compute_tb_label(close, high, low, atr, i, n)
            X_all.append(X_c[i - SEQ_LEN + 1 : i + 1])
            y_all.append(label)
            ts_all.append(ts_c[i])

    if not X_all:
        raise ValueError("No sequences built — check LABEL_DIR and feature names")

    X  = np.stack(X_all)
    y  = np.array(y_all, dtype=np.int64)
    ts = np.array(ts_all, dtype=np.int64)
    order  = np.argsort(ts)
    ts_idx = pd.DatetimeIndex([pd.Timestamp(t) for t in ts[order]])
    return X[order], y[order], ts_idx, avail


# ── Print header ───────────────────────────────────────────────────────────────
print(f"\n{'='*65}")
print(f"  LSTM TB — widyawardhana_v1")
print(f"  Features : {len(IC_FEATURES)} IC-validated | Seq={SEQ_LEN} | "
      f"Hidden={HIDDEN} | Layers={N_LAYERS}")
print(f"  Period   : 2020-01-01 – {TRAIN_CUTOFF_DATE} | {len(TRAINING_COINS)} coins")
print(f"  Sources  : Volatility(3), Volume(5), Structure(2), Pressure(3)")
print(f"{'='*65}")

# ── Load data ──────────────────────────────────────────────────────────────────
X, y, ts_idx, feat_used = load_sequences(TRAINING_COINS)
n_samples, seq_len, n_feat = X.shape

unq, cnt = np.unique(y, return_counts=True)
dist = {int(u): int(c) for u, c in zip(unq, cnt)}
print(f"\n[1/2] Data: {n_samples:,} sequences | {n_feat} features | seq={seq_len}")
print(f"  SHORT={dist.get(0,0):,}  FLAT={dist.get(1,0):,}  LONG={dist.get(2,0):,}")
print(f"  Ratio: {dist.get(0,0)/n_samples*100:.0f}% / "
      f"{dist.get(1,0)/n_samples*100:.0f}% / "
      f"{dist.get(2,0)/n_samples*100:.0f}%")

# ── Purged CV ──────────────────────────────────────────────────────────────────
print(f"\n[2/2] Purged CV ({N_FOLDS} folds) ...")
folds = build_purged_folds(ts_idx, N_FOLDS, PURGE_GAP_BARS)
cv_results  = []
best_loss   = float("inf")
best_state  = None
best_epochs = []

for fold_idx, (tr_idx, val_idx) in enumerate(folds):
    if len(val_idx) < 20: continue

    X_tr, y_tr = X[tr_idx], y[tr_idx]
    X_val, y_val = X[val_idx], y[val_idx]

    # Per-fold RobustScaler
    n_tr, s, f = X_tr.shape
    sc = RobustScaler().fit(X_tr.reshape(-1, f))
    X_tr_s  = sc.transform(X_tr.reshape(-1, f)).reshape(n_tr, s, f).astype(np.float32)
    n_val   = len(X_val)
    X_val_s = sc.transform(X_val.reshape(-1, f)).reshape(n_val, s, f).astype(np.float32)
    del X_tr, X_val; gc.collect()

    train_ld = DataLoader(SeqDataset(X_tr_s, y_tr),  BATCH, shuffle=True,  num_workers=0)
    val_ld   = DataLoader(SeqDataset(X_val_s, y_val), BATCH, shuffle=False, num_workers=0)

    # Balanced class weights, capped at MAX_CW
    classes, counts = np.unique(y_tr, return_counts=True)
    total = len(y_tr)
    cw = torch.tensor(
        [total / (len(classes) * counts[classes == i][0])
         if i in classes else 1.0 for i in range(3)],
        dtype=torch.float32
    ).clamp(max=MAX_CW).to(DEVICE)

    model     = TradingLSTM(n_feat, HIDDEN, N_LAYERS, DROPOUT, use_native=False).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=cw)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD, foreach=False)

    best_f1, best_st, patience_ct, best_ep = -1.0, None, 0, 1

    for epoch in range(1, EPOCHS + 1):
        model.train()
        for xb, yb in train_ld:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            criterion(model(xb), yb).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        model.eval()
        preds, labels = [], []
        with torch.no_grad():
            for xb, yb in val_ld:
                preds.extend(model(xb.to(DEVICE)).argmax(1).cpu().numpy())
                labels.extend(yb.numpy())

        f1 = f1_score(labels, preds, average="macro", zero_division=0)
        if f1 > best_f1:
            best_f1 = f1; patience_ct = 0; best_ep = epoch
            best_st = {k: v.cpu() for k, v in model.state_dict().items()}
        else:
            patience_ct += 1
            if patience_ct >= PATIENCE: break

        if epoch % 10 == 0 or epoch == 1:
            # Train F1 on random subsample (5000) — cukup untuk deteksi overfitting
            n_sub = min(5000, len(X_tr_s))
            idx_sub = np.random.choice(len(X_tr_s), n_sub, replace=False)
            sub_ds  = SeqDataset(X_tr_s[idx_sub], y_tr[idx_sub])
            sub_ld  = DataLoader(sub_ds, BATCH, shuffle=False, num_workers=0)
            tr_preds, tr_labels = [], []
            with torch.no_grad():
                for xb, yb in sub_ld:
                    tr_preds.extend(model(xb.to(DEVICE)).argmax(1).cpu().numpy())
                    tr_labels.extend(yb.numpy())
            f1_tr = f1_score(tr_labels, tr_preds, average="macro", zero_division=0)
            gap   = f1_tr - f1
            gap_str = f"+{gap:.4f}" if gap >= 0 else f"{gap:.4f}"
            overfit = " [OVERFIT?]" if gap > 0.05 else ""
            logger.info(f"  [F{fold_idx+1}] E{epoch:>3} | "
                        f"tr={f1_tr:.4f} val={f1:.4f} gap={gap_str} "
                        f"best={best_f1:.4f}{overfit}")

    # Final eval for this fold
    model.load_state_dict(best_st); model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for xb, yb in val_ld:
            preds.extend(model(xb.to(DEVICE)).argmax(1).cpu().numpy())
            labels.extend(yb.numpy())

    f1_val  = f1_score(labels, preds, average="macro", zero_division=0)
    acc_val = float(np.mean(np.array(preds) == np.array(labels)))
    cv_results.append({"fold": fold_idx+1, "f1": round(f1_val,4), "acc": round(acc_val,4),
                       "best_epoch": best_ep})
    best_epochs.append(best_ep)
    logger.info(f"  Fold {fold_idx+1} DONE: F1={f1_val:.4f} Acc={acc_val:.4f} "
                f"Epoch={best_ep}")
    print(classification_report(labels, preds,
                                 target_names=["SHORT","FLAT","LONG"], zero_division=0))

    if (1 - f1_val) < best_loss:
        best_loss  = 1 - f1_val
        best_state = best_st

# ── Final retrain ──────────────────────────────────────────────────────────────
f1_mean = float(np.mean([r["f1"] for r in cv_results]))
f1_std  = float(np.std([r["f1"] for r in cv_results]))
print(f"\n  CV F1: {f1_mean:.4f} +/- {f1_std:.4f}  (random baseline: 0.333)")

print(f"  Retraining final model on 100% data ({int(np.mean(best_epochs))} epochs)...")

n_all, s_all, f_all = X.shape
sc_all = RobustScaler().fit(X.reshape(-1, f_all))
X_s    = sc_all.transform(X.reshape(-1, f_all)).reshape(n_all, s_all, f_all).astype(np.float32)
del X; gc.collect()

classes, counts = np.unique(y, return_counts=True)
total = len(y)
cw_all = torch.tensor(
    [total / (len(classes) * counts[classes == i][0])
     if i in classes else 1.0 for i in range(3)],
    dtype=torch.float32
).clamp(max=MAX_CW).to(DEVICE)

final_model = TradingLSTM(n_feat, HIDDEN, N_LAYERS, DROPOUT, use_native=False).to(DEVICE)
criterion   = nn.CrossEntropyLoss(weight=cw_all)
optimizer   = torch.optim.Adam(final_model.parameters(), lr=LR, weight_decay=WD, foreach=False)
full_ld     = DataLoader(SeqDataset(X_s, y), BATCH, shuffle=True, num_workers=0)
n_final_ep  = int(np.mean(best_epochs))

for epoch in range(1, n_final_ep + 1):
    final_model.train()
    for xb, yb in full_ld:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad()
        criterion(final_model(xb), yb).backward()
        nn.utils.clip_grad_norm_(final_model.parameters(), 1.0)
        optimizer.step()
    if epoch % 10 == 0:
        logger.info(f"  Final retrain epoch {epoch}/{n_final_ep}")

final_model.eval()

# ── Save ───────────────────────────────────────────────────────────────────────
print("\nSaving...")
save_lstm(final_model, OUT_DIR / "lstm.pt")
joblib.dump(sc_all, OUT_DIR / "lstm_scaler.pkl")

meta = {
    "run_name":    RUN_NAME,
    "labels":      "Triple Barrier (SHORT/FLAT/LONG)",
    "label_params": {"tp_mult": TP_MULT, "sl_mult": SL_MULT, "max_hold": TB_HOLD},
    "feature_source": "marginal_IC_test_step3 — 13 features from 15 unique sources",
    "features":    feat_used,
    "n_features":  n_feat,
    "seq_len":     SEQ_LEN,
    "architecture": f"TradingLSTM(n_feat={n_feat}, hidden={HIDDEN}, "
                    f"layers={N_LAYERS}, dropout={DROPOUT})",
    "n_samples":   n_samples,
    "label_dist":  dist,
    "cv_results":  cv_results,
    "cv_f1_mean":  round(f1_mean, 4),
    "cv_f1_std":   round(f1_std,  4),
    "random_baseline_f1": 0.333,
    "gain_vs_random": round(f1_mean - 0.333, 4),
    "training_period": "2020-01-01 to 2025-10-31",
    "trained_at":  datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
}
with open(OUT_DIR / f"{RUN_NAME}_meta.json", "w") as f:
    json.dump(meta, f, indent=2)
with open(OUT_DIR / f"{RUN_NAME}_features.json", "w") as f:
    json.dump(feat_used, f, indent=2)

print(f"\n{'='*65}")
print(f"  Saved -> {OUT_DIR}")
print(f"  CV F1 : {f1_mean:.4f} +/- {f1_std:.4f}")
print(f"  Gain  : +{f1_mean - 0.333:.4f} vs random baseline (0.333)")
print(f"  Next  : python pipeline/07_holdout_livelike_lstm.py --model {RUN_NAME}")
print(f"{'='*65}")
