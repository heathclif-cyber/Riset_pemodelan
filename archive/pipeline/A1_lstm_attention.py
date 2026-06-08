"""
A1: V3 baseline + Multi-Head Self-Attention (minimal change)

EXACT same as V3:
  - 16 OHLCV features, seq=16, 2-layer LSTM, CE loss, class weights
  - Same LR, dropout (0.45), batch size, patience

ONLY addition:
  - 4-head self-attention over LSTM output before classification head

Target: F1 > 0.41 (V3 baseline 0.407)
"""
import argparse, gc, json, sys, warnings, numpy as np, pandas as pd
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
    LSTM_V2_HIDDEN, LSTM_V2_LAYERS, LSTM_V2_DROPOUT,
    LSTM_V2_WEIGHT_DECAY, LSTM_V2_LR,
)
from core.models import _CellLSTM
from core.utils import setup_logger, get_lstm_device
from pipeline.shared import build_purged_folds

logger = setup_logger("A1_attention")
DEVICE = get_lstm_device()

FEATURES = [
    "log_ret_1", "log_ret_5", "log_ret_20",
    "rsi_6", "rsi_h4",
    "h4_trend", "trend_strength",
    "ema_21_slope_h4", "ema_50_slope_h4",
    "price_vs_ema_50_h4",
    "cvd", "cvd_momentum_adv", "volume_delta",
    "Buy_Liq", "Sell_Liq",
    "whale_retail_divergence",
]
N_FEATURES = len(FEATURES)
_PERCOIN_ZSCORE = {"cvd", "volume_delta", "Buy_Liq", "Sell_Liq"}
_ZSCORE_WIN = 500

# V3 hyperparameters (unchanged except attention addition)
SEQ_LEN = 16; HIDDEN = 96; LAYERS = 2; DROPOUT = 0.45
LR = LSTM_V2_LR; WD = LSTM_V2_WEIGHT_DECAY
BATCH = 128; EPOCHS = 80; PATIENCE = 15
N_HEADS = 4; ATTN_DROP = 0.30


# ─── Attention (DirectML-safe, no pow/clamp on GPU) ────────────────────────
class SelfAttention(nn.Module):
    def __init__(self, hidden, heads=4, dropout=0.3):
        super().__init__()
        self.heads = heads
        self.d_k = hidden // heads
        self.qkv = nn.Linear(hidden, hidden * 3)
        self.out = nn.Linear(hidden, hidden)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        B, T, H = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.heads, self.d_k).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        scale = float(self.d_k) ** -0.5
        attn = F.softmax((q @ k.transpose(-2, -1)) * scale, dim=-1)
        attn = self.drop(attn)
        out = (attn @ v).transpose(1, 2).reshape(B, T, H)
        return self.out(out)


# ─── V3 LSTM + Attention ───────────────────────────────────────────────────
class LSTM_Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = _CellLSTM(N_FEATURES, HIDDEN, LAYERS, DROPOUT)
        self.ln1 = nn.LayerNorm(HIDDEN)
        self.attn = SelfAttention(HIDDEN, N_HEADS, ATTN_DROP)
        self.ln2 = nn.LayerNorm(HIDDEN)
        self.drop = nn.Dropout(DROPOUT)
        self.fc = nn.Linear(HIDDEN, 3)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.ln1(out)
        out = self.ln2(self.attn(out) + out)  # residual
        last = self.drop(out[:, -1, :])
        return self.fc(last)


# ─── Data (same as V3) ─────────────────────────────────────────────────────
class SeqDS(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.int64))
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]


def pz(s, w=_ZSCORE_WIN):
    s = pd.Series(s)
    m = s.rolling(w, min_periods=50).mean()
    st = s.rolling(w, min_periods=50).std().clip(lower=1e-8)
    return ((s - m) / st).clip(-4, 4).fillna(0).values.astype(np.float32)


def fit_scaler(X):
    n, s, f = X.shape; sc = RobustScaler(); sc.fit(X.reshape(-1, f)); return sc


def scale_X(X, sc):
    n, s, f = X.shape; return sc.transform(X.reshape(-1, f)).reshape(n, s, f).astype(np.float32)


def load_data(coins):
    Xs, ys, ts = [], [], []
    for coin in coins:
        fp = LABEL_DIR / f"{coin}_features_v3.parquet"
        lp = LABEL_DIR / f"{coin}_momentum_v2_labels.parquet"
        if not fp.exists() or not lp.exists(): continue
        df = pd.read_parquet(fp).sort_index()
        df = df[df.index < TRAIN_CUTOFF_DATE]
        lbl = pd.read_parquet(lp).sort_index()
        df = df.join(lbl["momentum_v2_label"], how="inner")
        df = df.dropna(subset=["momentum_v2_label"])
        if len(df) < SEQ_LEN + 10: continue

        fv = {}
        miss = []
        for c in FEATURES:
            if c in df.columns:
                v = df[c].ffill().fillna(0).values.astype(np.float32)
                fv[c] = pz(v.astype(np.float64)).astype(np.float32) if c in _PERCOIN_ZSCORE else v
            else:
                fv[c] = np.zeros(len(df), dtype=np.float32)
                miss.append(c)
        if miss: logger.info(f"{coin}: missing {miss}")

        Xc = np.column_stack([fv[c] for c in FEATURES])
        yc = df["momentum_v2_label"].values.astype(np.int64)
        tc = df.index.astype(np.int64).values
        for i in range(SEQ_LEN - 1, len(Xc)):
            Xs.append(Xc[i - SEQ_LEN + 1:i + 1])
            ys.append(yc[i]); ts.append(tc[i])

        n = len(yc)
        logger.info(f"{coin}: {len(df):,}b | BULL={(yc==2).sum()/n*100:.0f}% NEU={(yc==1).sum()/n*100:.0f}% BEAR={(yc==0).sum()/n*100:.0f}% | seqs={len(Xc)-SEQ_LEN+1:,}")

    X = np.stack(Xs); y = np.array(ys, dtype=np.int64); t = np.array(ts, dtype=np.int64)
    o = np.argsort(t)
    logger.info(f"Total {len(Xs):,} seqs X={X.shape}")
    return X[o], y[o], t[o], FEATURES


def class_weights(y):
    uc, cnt = np.unique(y, return_counts=True)
    n = len(y)
    w = {c: n / (3.0 * cnt_i) for c, cnt_i in zip(uc, cnt)}
    return torch.tensor([w.get(i, 1.0) for i in range(3)], dtype=torch.float32)


def train_fold(Xtr, ytr, Xte, yte, fi):
    sc = fit_scaler(Xtr)
    Xtr_s = scale_X(Xtr, sc); del Xtr; gc.collect()
    Xte_s = scale_X(Xte, sc); del Xte; gc.collect()

    tr_ds = SeqDS(Xtr_s, ytr); te_ds = SeqDS(Xte_s, yte)
    tr_ld = DataLoader(tr_ds, batch_size=BATCH, shuffle=True)
    te_ld = DataLoader(te_ds, batch_size=BATCH, shuffle=False)

    model = LSTM_Attention().to(DEVICE)
    cw = class_weights(ytr).to(DEVICE)
    crit = nn.CrossEntropyLoss(weight=cw)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)

    best, best_st, pc = -1.0, None, 0
    for ep in range(1, EPOCHS + 1):
        model.train()
        for xb, yb in tr_ld:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            crit(model(xb), yb).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        model.eval()
        pv, lv = [], []
        with torch.no_grad():
            for xb, yb in te_ld:
                pv.extend(model(xb.to(DEVICE)).argmax(1).cpu().numpy())
                lv.extend(yb.numpy())
        f1 = float(f1_score(lv, pv, average="macro", zero_division=0))

        if f1 > best:
            best, best_st, pc = f1, {k: v.cpu() for k, v in model.state_dict().items()}, 0
        else:
            pc += 1
            if pc >= PATIENCE: break

        if ep % 10 == 0 or ep == 1:
            logger.info(f"[F{fi}] E{ep:>3} | F1={f1:.4f} Best={best:.4f}")

    model.load_state_dict(best_st); model.eval()
    pv, lv = [], []
    with torch.no_grad():
        for xb, yb in te_ld:
            pv.extend(model(xb.to(DEVICE)).argmax(1).cpu().numpy())
            lv.extend(yb.numpy())
    vf1 = float(f1_score(lv, pv, average="macro", zero_division=0))
    vp = f1_score(lv, pv, average=None, zero_division=0, labels=[0, 1, 2])

    tp, tl = [], []
    with torch.no_grad():
        for xb, yb in tr_ld:
            tp.extend(model(xb.to(DEVICE)).argmax(1).cpu().numpy())
            tl.extend(yb.numpy())
    tf1 = float(f1_score(tl, tp, average="macro", zero_division=0))

    m = {"fold": fi, "train_f1": round(tf1, 4), "val_f1": round(vf1, 4),
         "f1_BEARISH": round(float(vp[0]), 4),
         "f1_NEUTRAL": round(float(vp[1]), 4),
         "f1_BULLISH": round(float(vp[2]), 4)}
    logger.info(f"[F{fi}] Train={tf1:.4f} Val={vf1:.4f} Gap={tf1-vf1:+.4f} | B={vp[0]:.3f} N={vp[1]:.3f} BU={vp[2]:.3f}")
    return model, sc, m


def retrain_final(X, y, eps):
    sc = fit_scaler(X); Xs = scale_X(X, sc); del X; gc.collect()
    ds = SeqDS(Xs, y); ld = DataLoader(ds, batch_size=BATCH, shuffle=True)

    model = LSTM_Attention().to(DEVICE)
    cw = class_weights(y).to(DEVICE)
    crit = nn.CrossEntropyLoss(weight=cw)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)

    model.train()
    for ep in range(1, eps + 1):
        tl = 0.0
        for xb, yb in ld:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            crit(model(xb), yb).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tl += float(crit(model(xb), yb))
        if ep % 10 == 0 or ep == 1:
            logger.info(f"[Final] E{ep:>3}/{eps} loss={tl/len(ld):.4f}")
    model.eval()
    return model, sc


def main():
    coins = TRAINING_COINS[:5]
    run = "lstm_momentum_v7_A1"

    print(f"\n{'='*55}")
    print(f"  A1: V3 + Attention Only | {run}")
    print(f"  Features:{N_FEATURES}  Seq:{SEQ_LEN}  Hidden:{HIDDEN}  Layers:{LAYERS}")
    print(f"  Dropout:{DROPOUT}  Attn:{N_HEADS}heads  Loss:CE+weights")
    print(f"  V3 baseline: 0.407 | Target: >0.41")
    print(f"{'='*55}\n")

    torch.manual_seed(42); np.random.seed(42)
    X, y, t, fc = load_data(coins)

    rd = MODEL_DIR / "runs" / run; rd.mkdir(parents=True, exist_ok=True)
    json.dump(fc, open(rd / "feature_cols.json", "w"), indent=2)

    folds = build_purged_folds(pd.to_datetime(t, unit="ns", utc=True), N_FOLDS, PURGE_GAP_BARS)
    metrics = []
    for fi, (tr, te) in enumerate(folds):
        _, _, m = train_fold(X[tr], y[tr], X[te], y[te], fi + 1)
        metrics.append(m)

    avge = int(np.median([m["val_f1"] for m in metrics]))
    final_eps = max(25, min(avge + 5, EPOCHS))
    logger.info(f"Retrain final {final_eps} epochs...")
    model, scaler = retrain_final(X, y, final_eps)

    torch.save(model.state_dict(), str(rd / "model.pt"))
    joblib.dump(scaler, rd / "scaler.pkl")

    vf = [m["val_f1"] for m in metrics]
    tf = [m["train_f1"] for m in metrics]
    gf = [t - v for t, v in zip(tf, vf)]
    nf = [m["f1_NEUTRAL"] for m in metrics]

    print(f"\n{'='*55}")
    print(f"  A1 COMPLETE")
    print(f"  Mean Val F1:    {np.mean(vf):.4f} +/- {np.std(vf):.4f}")
    print(f"  Mean NEUTRAL F1: {np.mean(nf):.4f}")
    print(f"  Mean Gap:       {np.mean(gf):+.4f}")
    print(f"  V3 baseline:    0.407 +/- 0.007")
    for m in metrics:
        g = m["train_f1"] - m["val_f1"]
        print(f"  F{m['fold']}: Train={m['train_f1']:.4f} Val={m['val_f1']:.4f} Gap={g:+.4f} | B={m['f1_BEARISH']:.3f} N={m['f1_NEUTRAL']:.3f} BU={m['f1_BULLISH']:.3f}")
    print(f"\n  Model: {rd / 'model.pt'}")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()
