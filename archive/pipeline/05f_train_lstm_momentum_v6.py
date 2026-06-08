"""
pipeline/05f_train_lstm_momentum_v6.py — LSTM Momentum V6 (Focal Loss + Attention)

Key upgrades vs V3/V5:
  - Focal Loss (γ=2.0, α per-class): fokus ke NEUTRAL yang sulit
  - Multi-Head Self-Attention di atas LSTM: focus ke bar penting di sequence
  - Label Smoothing: kurangi overconfidence pada NEUTRAL
  - Cosine LR scheduler + warmup: training lebih stabil
  - Stronger dropout + weight decay

Target: breakthrough F1 > 0.44 dari ceiling OHLCV 0.41

Usage:
  python pipeline/05f_train_lstm_momentum_v6.py --all
"""

import argparse, gc, json, math, sys, warnings, numpy as np, pandas as pd
from pathlib import Path

ROOT = Path(__file__).parent.parent; sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

import torch, torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import f1_score
import joblib

from config import (
    TRAINING_COINS, LABEL_DIR, TRAIN_CUTOFF_DATE, MODEL_DIR,
    N_FOLDS, PURGE_GAP_BARS,
    LSTM_BATCH_SIZE, LSTM_EPOCHS, LSTM_PATIENCE,
    LSTM_V2_HIDDEN, LSTM_V2_LAYERS, LSTM_V2_WEIGHT_DECAY,
)
from core.utils import setup_logger, get_lstm_device
from pipeline.shared import build_purged_folds

logger = setup_logger("05f_lstm_momentum_v6")
DEVICE = get_lstm_device()

# ─── V6 Features: 16 OHLCV (same as V3) ────────────────────────────────────
LSTM_MOMENTUM_V6_FEATURES = [
    "log_ret_1", "log_ret_5", "log_ret_20",
    "rsi_6", "rsi_h4",
    "h4_trend", "trend_strength",
    "ema_21_slope_h4", "ema_50_slope_h4",
    "price_vs_ema_50_h4",
    "cvd", "cvd_momentum_adv", "volume_delta",
    "Buy_Liq", "Sell_Liq",
    "whale_retail_divergence",
]
N_FEATURES = len(LSTM_MOMENTUM_V6_FEATURES)

_PERCOIN_ZSCORE_FEATS = {"cvd", "volume_delta", "Buy_Liq", "Sell_Liq"}
_ZSCORE_WINDOW = 500

MOMENTUM_LABEL_MAP = {"BEARISH": 0, "NEUTRAL": 1, "BULLISH": 2}

# ─── V6 Hyperparameters ────────────────────────────────────────────────────
SEQ_LEN       = 16
HIDDEN        = 128
NUM_LAYERS    = 3           # 2 -> 3 layers (more capacity)
DROPOUT       = 0.50        # 0.45 -> 0.50 (stronger regularization)
NUM_HEADS     = 4           # Multi-head attention heads
ATTN_DROPOUT  = 0.35
LR            = 3e-4        # lower LR for stable training
BATCH_SIZE    = 256         # larger batch
EPOCHS        = 80
PATIENCE      = 15
WEIGHT_DECAY  = 0.02        # 0.01 -> 0.02
FOCAL_GAMMA   = 2.0
LABEL_SMOOTH  = 0.08        # 8% smoothing for NEUTRAL
WARMUP_EPOCHS = 3


# ─── Focal Loss ─────────────────────────────────────────────────────────────
class FocalLoss(nn.Module):
    """
    Multi-class Focal Loss.
    FL = -alpha * (1-p_t)^gamma * log(p_t)

    gamma=2.0: fokus ke contoh sulit (prob rendah = misclassified)
    alpha: per-class weights untuk imbalance
    label_smoothing: kurangi overconfidence
    """
    def __init__(self, alpha=None, gamma=2.0, label_smoothing=0.0):
        super().__init__()
        self.alpha = alpha          # tensor [num_classes]
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, logits, targets):
        """
        logits:  (B, C) on GPU
        targets: (B,) int64 on GPU

        Focal loss: CE computed on GPU, focal weights computed on CPU
        (DirectML doesn't support clamp/pow — weight calculation on CPU).
        """
        B, C = logits.shape

        # CE on GPU (supported by DirectML)
        if self.label_smoothing > 0:
            with torch.no_grad():
                smooth_targets = torch.full((B, C), self.label_smoothing / (C - 1),
                                           device=logits.device)
                smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.label_smoothing)
            log_probs = F.log_softmax(logits, dim=1)
            ce = -(smooth_targets * log_probs).sum(dim=1)
        else:
            ce = F.cross_entropy(logits, targets, reduction='none')

        # Focal weights on CPU (DirectML-safe)
        with torch.no_grad():
            ce_cpu = ce.detach().cpu()
            pt_cpu = torch.exp(-ce_cpu)
            focal_w_cpu = (1.0 - pt_cpu).clamp(min=1e-7, max=1.0 - 1e-7).pow(self.gamma)

        # Multiply back on GPU
        focal_weight = focal_w_cpu.to(logits.device)

        if self.alpha is not None:
            alpha_t = self.alpha.to(logits.device)[targets]
            focal = alpha_t * focal_weight * ce
        else:
            focal = focal_weight * ce

        return focal.mean()


# ─── Multi-Head Self-Attention ──────────────────────────────────────────────
class MultiHeadSelfAttention(nn.Module):
    """Self-attention over the temporal dimension (seq_len)."""
    def __init__(self, hidden_size, num_heads=4, dropout=0.3):
        super().__init__()
        assert hidden_size % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = float(self.head_dim) ** -0.5  # python float, not tensor

        self.qkv = nn.Linear(hidden_size, hidden_size * 3)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        x: (B, seq_len, hidden_size)
        Returns: (B, seq_len, hidden_size) — attended sequence
        """
        B, T, H = x.shape

        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, heads, T, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, heads, T, T)
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = attn @ v  # (B, heads, T, head_dim)
        out = out.transpose(1, 2).reshape(B, T, H)
        return self.out_proj(out)


# ─── LSTM + Attention Model ─────────────────────────────────────────────────
class TradingLSTMAttention(nn.Module):
    """
    LSTM (3-layer, ManualLSTMCell) + Multi-Head Self-Attention + LayerNorm.
    Attention di atas LSTM output → fokus ke bar penting dalam sequence.
    """
    def __init__(self, n_features, hidden_size=128, num_layers=3,
                 dropout=0.5, num_heads=4, attn_dropout=0.35, num_classes=3):
        super().__init__()
        from core.models import _CellLSTM

        self.lstm = _CellLSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
        )
        self.lstm_norm = nn.LayerNorm(hidden_size)
        self.attention = MultiHeadSelfAttention(hidden_size, num_heads, attn_dropout)
        self.attn_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        """
        x: (B, seq_len, n_features)
        Returns: (B, num_classes) logits
        """
        # LSTM encoder
        out, _ = self.lstm(x)              # (B, seq_len, hidden)
        out = self.lstm_norm(out)

        # Self-attention over time
        attn_out = self.attention(out)     # (B, seq_len, hidden)
        attn_out = self.attn_norm(attn_out + out)  # residual

        # Aggregate: weighted average of last 4 positions + attended output
        last = attn_out[:, -1, :]          # (B, hidden)
        last = self.dropout(last)
        return self.fc(last)

    def save(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), str(path))


# ─── Data Loading ───────────────────────────────────────────────────────────
class PrebuiltSeqDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.int64))
    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]


def _percoin_z(series, window=_ZSCORE_WINDOW):
    s = pd.Series(series)
    mean = s.rolling(window=window, min_periods=50).mean()
    std = s.rolling(window=window, min_periods=50).std().clip(lower=1e-8)
    return ((s - mean) / std).clip(-4, 4).fillna(0).values.astype(np.float32)


def fit_scaler(X):
    n, s, f = X.shape; scl = RobustScaler(); scl.fit(X.reshape(-1, f)); return scl


def scale_X(X, scl):
    n, s, f = X.shape; return scl.transform(X.reshape(-1, f)).reshape(n, s, f).astype(np.float32)


def load_data(coins):
    feats = LSTM_MOMENTUM_V6_FEATURES
    X_seqs, y_seqs, ts_seqs = [], [], []
    skipped = []

    for coin in coins:
        fp = LABEL_DIR / f"{coin}_features_v3.parquet"
        lp = LABEL_DIR / f"{coin}_momentum_v2_labels.parquet"
        if not fp.exists() or not lp.exists():
            skipped.append(coin); continue

        df = pd.read_parquet(fp).sort_index()
        df = df[df.index < TRAIN_CUTOFF_DATE]
        lbl = pd.read_parquet(lp).sort_index()
        df = df.join(lbl["momentum_v2_label"], how="inner")
        df = df.dropna(subset=["momentum_v2_label"])

        if len(df) < SEQ_LEN + 10:
            skipped.append(coin); continue

        feat_vals = {}
        missing = []
        for c in feats:
            if c in df.columns:
                vals = df[c].ffill().fillna(0).values.astype(np.float32)
                if c in _PERCOIN_ZSCORE_FEATS:
                    vals = _percoin_z(vals.astype(np.float64)).astype(np.float32)
                feat_vals[c] = vals
            else:
                feat_vals[c] = np.zeros(len(df), dtype=np.float32)
                missing.append(c)

        if missing:
            logger.info(f"{coin}: missing (filled 0): {missing}")

        X_c = np.column_stack([feat_vals[c] for c in feats])
        y_c = df["momentum_v2_label"].values.astype(np.int64)
        ts_c = df.index.astype(np.int64).values

        for i in range(SEQ_LEN - 1, len(X_c)):
            X_seqs.append(X_c[i - SEQ_LEN + 1:i + 1])
            y_seqs.append(y_c[i])
            ts_seqs.append(ts_c[i])

        n_total = len(y_c)
        logger.info(f"{coin}: {len(df):,} bars | BULL={(y_c==2).sum()/n_total*100:.0f}% NEU={(y_c==1).sum()/n_total*100:.0f}% BEAR={(y_c==0).sum()/n_total*100:.0f}% | seqs={len(df)-SEQ_LEN+1:,}")

    if skipped:
        logger.warning(f"Skipped: {skipped}")
    if not X_seqs:
        raise ValueError("No sequences.")
    X = np.stack(X_seqs); y = np.array(y_seqs, dtype=np.int64); ts = np.array(ts_seqs, dtype=np.int64)
    order = np.argsort(ts)
    logger.info(f"Total: {len(X_seqs):,} sequences | X={X.shape}")
    return X[order], y[order], ts[order], feats


def compute_alpha_weights(y):
    """Focal loss alpha — higher weight for NEUTRAL (harder class)."""
    classes, counts = np.unique(y, return_counts=True)
    total = len(y)
    # NEUTRAL (1) gets 2x BULLISH/BEARISH weight
    base = {c: total / (len(classes) * cnt) for c, cnt in zip(classes, counts)}
    # Boost NEUTRAL
    if 1 in base:
        base[1] *= 2.2  # 2.2x higher focus on hard NEUTRAL examples
    weights = [base.get(i, 1.0) for i in range(3)]
    # Normalize
    s = sum(weights)
    return torch.tensor([w / s * 3 for w in weights], dtype=torch.float32)


# ─── Training ───────────────────────────────────────────────────────────────
def train_one_fold(X_tr, y_tr, X_te, y_te, fold_num):
    fold_scaler = fit_scaler(X_tr)
    X_tr_s = scale_X(X_tr, fold_scaler); del X_tr; gc.collect()
    X_te_s = scale_X(X_te, fold_scaler); del X_te; gc.collect()

    tr_ds = PrebuiltSeqDataset(X_tr_s, y_tr); te_ds = PrebuiltSeqDataset(X_te_s, y_te)
    tr_ld = DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    te_ld = DataLoader(te_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = TradingLSTMAttention(
        n_features=N_FEATURES, hidden_size=HIDDEN, num_layers=NUM_LAYERS,
        dropout=DROPOUT, num_heads=NUM_HEADS, attn_dropout=ATTN_DROPOUT,
    ).to(DEVICE)

    alpha = compute_alpha_weights(y_tr).to(DEVICE)
    criterion = FocalLoss(alpha=alpha, gamma=FOCAL_GAMMA, label_smoothing=LABEL_SMOOTH)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=15, T_mult=2, eta_min=1e-6)

    best_f1, best_state, patience_count = -1.0, None, 0
    for epoch in range(1, EPOCHS + 1):
        # Warmup
        if epoch <= WARMUP_EPOCHS:
            for pg in optimizer.param_groups:
                pg['lr'] = LR * epoch / WARMUP_EPOCHS

        model.train()
        for xb, yb in tr_ld:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        if epoch > WARMUP_EPOCHS:
            scheduler.step()

        model.eval()
        pv, lv = [], []
        with torch.no_grad():
            for xb, yb in te_ld:
                pv.extend(model(xb.to(DEVICE)).argmax(dim=1).cpu().numpy())
                lv.extend(yb.numpy())
        val_f1 = float(f1_score(lv, pv, average="macro", zero_division=0))

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= PATIENCE:
                break

        if epoch % 5 == 0 or epoch == 1:
            lr_now = optimizer.param_groups[0]['lr']
            logger.info(f"[F{fold_num}] E{epoch:>3} | F1={val_f1:.4f} Best={best_f1:.4f} lr={lr_now:.1e}")

    model.load_state_dict(best_state); model.eval()
    pv, lv = [], []
    with torch.no_grad():
        for xb, yb in te_ld:
            pv.extend(model(xb.to(DEVICE)).argmax(dim=1).cpu().numpy())
            lv.extend(yb.numpy())
    val_f1 = float(f1_score(lv, pv, average="macro", zero_division=0))
    val_f1_p = f1_score(lv, pv, average=None, zero_division=0, labels=[0, 1, 2])

    tr_pv, tr_lv = [], []
    with torch.no_grad():
        for xb, yb in tr_ld:
            tr_pv.extend(model(xb.to(DEVICE)).argmax(dim=1).cpu().numpy())
            tr_lv.extend(yb.numpy())
    train_f1 = float(f1_score(tr_lv, tr_pv, average="macro", zero_division=0))

    metrics = {
        "fold": fold_num,
        "train_f1": round(train_f1, 4), "val_f1": round(val_f1, 4),
        "f1_BEARISH": round(float(val_f1_p[0]), 4),
        "f1_NEUTRAL": round(float(val_f1_p[1]), 4),
        "f1_BULLISH": round(float(val_f1_p[2]), 4),
    }
    logger.info(f"[F{fold_num}] Train={train_f1:.4f} Val={val_f1:.4f} Gap={train_f1-val_f1:+.4f} | B={val_f1_p[0]:.3f} N={val_f1_p[1]:.3f} BU={val_f1_p[2]:.3f}")
    return model, fold_scaler, metrics


def retrain_final(X_all, y_all, n_epochs):
    final_scaler = fit_scaler(X_all)
    X_sc = scale_X(X_all, final_scaler); del X_all; gc.collect()

    ds = PrebuiltSeqDataset(X_sc, y_all)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    model = TradingLSTMAttention(
        n_features=N_FEATURES, hidden_size=HIDDEN, num_layers=NUM_LAYERS,
        dropout=DROPOUT, num_heads=NUM_HEADS, attn_dropout=ATTN_DROPOUT,
    ).to(DEVICE)

    alpha = compute_alpha_weights(y_all).to(DEVICE)
    criterion = FocalLoss(alpha=alpha, gamma=FOCAL_GAMMA, label_smoothing=LABEL_SMOOTH)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=15, T_mult=2, eta_min=1e-6)

    model.train()
    for epoch in range(1, n_epochs + 1):
        if epoch <= WARMUP_EPOCHS:
            for pg in optimizer.param_groups:
                pg['lr'] = LR * epoch / WARMUP_EPOCHS
        total_loss = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
        if epoch > WARMUP_EPOCHS:
            scheduler.step()
        if epoch % 10 == 0 or epoch == 1:
            logger.info(f"[Final] E{epoch:>3}/{n_epochs} | loss={total_loss/len(loader):.4f}")

    model.eval()
    return model, final_scaler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default="lstm_momentum_v6")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--coins", type=int, default=5)
    args = parser.parse_args()

    coins = TRAINING_COINS if args.all else TRAINING_COINS[:args.coins]

    print(f"\n{'='*65}")
    print(f"  LSTM MOMENTUM V6 | run_id={args.run_id}")
    print(f"  Architecture: LSTM(3x128) + 4-Head Self-Attention")
    print(f"  Loss: FocalLoss(gamma={FOCAL_GAMMA}, label_smooth={LABEL_SMOOTH}) + NEUTRAL boost 2.2x")
    print(f"  Features: {N_FEATURES} | Seq: {SEQ_LEN} | Hidden: {HIDDEN}")
    print(f"  Dropout: {DROPOUT} | Attn Drop: {ATTN_DROPOUT} | WD: {WEIGHT_DECAY}")
    print(f"  LR: {LR} | Warmup: {WARMUP_EPOCHS}ep | Scheduler: CosineAnnealing")
    print(f"  Coins: {len(coins)} | Target F1: > 0.44")
    print(f"{'='*65}\n")

    torch.manual_seed(42); np.random.seed(42)
    X, y, ts, feat_cols = load_data(coins)

    run_dir = MODEL_DIR / "runs" / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "lstm_momentum_v6_feature_cols.json", "w") as f:
        json.dump(feat_cols, f, indent=2)

    ts_index = pd.to_datetime(ts, unit="ns", utc=True)
    folds = build_purged_folds(ts_index, n_folds=N_FOLDS, purge=PURGE_GAP_BARS)

    all_metrics = []
    for fi, (tr_idx, te_idx) in enumerate(folds):
        _, _, m = train_one_fold(X[tr_idx], y[tr_idx], X[te_idx], y[te_idx], fi + 1)
        all_metrics.append(m)

    avg_best_epoch = int(np.median([m.get("best_epoch", 30) for m in all_metrics])) if all_metrics else 30
    final_epochs = max(30, min(avg_best_epoch + 5, EPOCHS))

    logger.info(f"Retraining final on 100% data for {final_epochs} epochs...")
    final_model, final_scaler = retrain_final(X, y, final_epochs)

    final_model.save(run_dir / "lstm_momentum_v6.pt")
    joblib.dump(final_scaler, run_dir / "lstm_momentum_v6_scaler.pkl")

    meta = {
        "run_id": args.run_id, "model_type": "lstm_momentum_v6",
        "architecture": "LSTM(3x128) + 4-Head Self-Attention + FocalLoss",
        "n_features": len(feat_cols), "features": feat_cols,
        "seq_len": SEQ_LEN, "hidden": HIDDEN, "num_layers": NUM_LAYERS,
        "dropout": DROPOUT, "num_heads": NUM_HEADS, "attn_dropout": ATTN_DROPOUT,
        "focal_gamma": FOCAL_GAMMA, "label_smooth": LABEL_SMOOTH,
        "lr": LR, "weight_decay": WEIGHT_DECAY,
        "cv_folds": len(all_metrics),
    }
    with open(run_dir / "lstm_momentum_v6_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    with open(run_dir / "lstm_momentum_v6_cv_results.json", "w") as f:
        json.dump({"run_id": args.run_id, "metrics": all_metrics}, f, indent=2)

    val_f1s = [m["val_f1"] for m in all_metrics]
    train_f1s = [m["train_f1"] for m in all_metrics]
    gaps = [t - v for t, v in zip(train_f1s, val_f1s)]
    n_f1s = [m["f1_NEUTRAL"] for m in all_metrics]

    print(f"\n{'='*65}")
    print(f"  LSTM MOMENTUM V6 COMPLETE — {args.run_id}")
    print(f"  Mean Val F1:     {np.mean(val_f1s):.4f} +/- {np.std(val_f1s):.4f}")
    print(f"  Mean NEUTRAL F1:  {np.mean(n_f1s):.4f}")
    print(f"  Mean Train F1:   {np.mean(train_f1s):.4f}")
    print(f"  Mean Gap:        {np.mean(gaps):+.4f}")
    print(f"  Gain vs random:  {np.mean(val_f1s) - 0.333:+.4f}")
    print(f"\n  Fold results:")
    for m in all_metrics:
        g = m["train_f1"] - m["val_f1"]
        print(f"  F{m['fold']}: Train={m['train_f1']:.4f} Val={m['val_f1']:.4f} Gap={g:+.4f} | BEAR={m['f1_BEARISH']:.3f} NEU={m['f1_NEUTRAL']:.3f} BULL={m['f1_BULLISH']:.3f}")
    print(f"\n  V3 baseline (CE Loss, LSTM only):  0.407 ± 0.007 | NEU F1 ~0.266")
    print(f"  V6 target  (Focal+Attention):      > 0.44")
    print(f"\n  Model: {run_dir / 'lstm_momentum_v6.pt'}")


if __name__ == "__main__":
    main()
