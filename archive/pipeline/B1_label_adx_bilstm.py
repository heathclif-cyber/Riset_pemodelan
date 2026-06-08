"""
B1: Improved Labels + ADX + BiLSTM + Moderate Focal Loss

Changes from V3:
  1. Label: N=6 (was 8), threshold >=3/4 (was >=2/4) -> cleaner NEUTRAL
  2. ADX features: adx_14, plus_di, minus_di, adx_slope -> regime context
  3. BiLSTM: bidirectional LSTM + LayerNorm -> both directions
  4. Focal Loss: gamma=1.0, NEUTRAL alpha=1.5x -> moderate focus

Target: F1 > 0.43
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
    LSTM_V2_LR, LSTM_V2_WEIGHT_DECAY,
)
from core.models import _ManualLSTMCell
from core.utils import setup_logger, get_lstm_device
from pipeline.shared import build_purged_folds

logger = setup_logger("B1"); DEVICE = get_lstm_device()

# ─── Config ─────────────────────────────────────────────────────────────────
OHLCV_FEATS = [
    "log_ret_1", "log_ret_5", "log_ret_20",
    "rsi_6", "rsi_h4",
    "h4_trend", "trend_strength",
    "ema_21_slope_h4", "ema_50_slope_h4",
    "price_vs_ema_50_h4",
    "cvd", "cvd_momentum_adv", "volume_delta",
    "Buy_Liq", "Sell_Liq",
    "whale_retail_divergence",
]
ADX_FEATS = ["adx_14", "plus_di_14", "minus_di_14", "adx_slope_14"]
ALL_FEATS = OHLCV_FEATS + ADX_FEATS
N_F = len(ALL_FEATS)

_PERCOIN_Z = {"cvd", "volume_delta", "Buy_Liq", "Sell_Liq"}
_ZWIN = 500

# Label params (IMPROVED)
MOM_N = 6         # was 8 — shorter horizon, clearer signal
MOM_MIN_VOTES = 3  # was 2 — stricter, less noisy NEUTRAL

# Architecture
SEQ = 16; HID = 128; LAY = 2; DR = 0.40
LR = 5e-4; WD = 0.015; BATCH = 128; EPOCHS = 80; PAT = 12
GAMMA = 1.0        # focal gamma (moderate)
NEU_ALPHA = 1.5    # NEUTRAL class weight multiplier


# ─── ADX Computation ───────────────────────────────────────────────────────
def compute_adx(high, low, close, period=14):
    """Compute ADX(14), +DI, -DI. Wilder's smoothing."""
    n = len(high)
    tr = np.zeros(n); pdm = np.zeros(n); ndm = np.zeros(n)

    for i in range(1, n):
        hl = high[i] - low[i]
        hc = abs(high[i] - close[i-1])
        lc = abs(low[i] - close[i-1])
        tr[i] = max(hl, hc, lc)

        up = high[i] - high[i-1]
        dn = low[i-1] - low[i]
        pdm[i] = up if up > dn and up > 0 else 0
        ndm[i] = dn if dn > up and dn > 0 else 0

    # Wilder's smoothing (exponential with alpha=1/period)
    alpha = 1.0 / period
    atr = np.zeros(n); atr[period] = tr[1:period+1].mean()
    spdm = np.zeros(n); spdm[period] = pdm[1:period+1].mean()
    sndm = np.zeros(n); sndm[period] = ndm[1:period+1].mean()

    for i in range(period+1, n):
        atr[i] = atr[i-1] + alpha * (tr[i] - atr[i-1])
        spdm[i] = spdm[i-1] + alpha * (pdm[i] - spdm[i-1])
        sndm[i] = sndm[i-1] + alpha * (ndm[i] - sndm[i-1])

    pdi = np.zeros(n); ndi = np.zeros(n); adx = np.zeros(n)
    for i in range(period, n):
        if atr[i] > 0:
            pdi[i] = 100.0 * spdm[i] / atr[i]
            ndi[i] = 100.0 * sndm[i] / atr[i]
        if pdi[i] + ndi[i] > 0:
            adx[i] = 100.0 * abs(pdi[i] - ndi[i]) / (pdi[i] + ndi[i])

    # Smooth ADX (Wilder's)
    sadx = np.zeros(n)
    sadx[period*2-1] = adx[period:period*2].mean()
    for i in range(period*2, n):
        sadx[i] = sadx[i-1] + alpha * (adx[i] - sadx[i-1])

    adx_slope = np.zeros(n)
    adx_slope[period*2:] = sadx[period*2:] - sadx[period*2-1:-1]

    return sadx, pdi, ndi, adx_slope


# ─── Improved Labeling ─────────────────────────────────────────────────────
def compute_labels_v3(df, N=MOM_N, min_votes=MOM_MIN_VOTES):
    """Flow-based momentum labels with stricter voting."""
    n = len(df)
    labels = np.ones(n, dtype=np.int8)
    close = df["close"].values

    ofi = df["ofi_z_score"].values if "ofi_z_score" in df.columns else np.zeros(n)

    if "cvd_momentum_adv" in df.columns:
        s = pd.Series(df["cvd_momentum_adv"].values)
        m = s.rolling(500, min_periods=50).mean()
        st = s.rolling(500, min_periods=50).std().clip(lower=1e-8)
        cvd_z = ((s - m) / st).clip(-4, 4).fillna(0).values
    else:
        cvd_z = np.zeros(n)

    if "volume_delta" in df.columns:
        s = pd.Series(df["volume_delta"].values)
        m = s.rolling(500, min_periods=50).mean()
        st = s.rolling(500, min_periods=50).std().clip(lower=1e-8)
        vd_z = ((s - m) / st).clip(-4, 4).fillna(0).values
    else:
        vd_z = np.zeros(n)

    for t in range(n - N - 1):
        end = t + N + 1

        ofi_f = float(np.nanmean(ofi[t+1:end]))
        ofi_v = 1 if ofi_f > 0.25 else (-1 if ofi_f < -0.25 else 0)

        cvd_d = float(cvd_z[end-1] - cvd_z[t])
        cvd_v = 1 if cvd_d > 0 else (-1 if cvd_d < 0 else 0)

        vd_f = float(np.nanmean(vd_z[t+1:end]))
        vd_v = 1 if vd_f > 0.25 else (-1 if vd_f < -0.25 else 0)

        if close[t] > 0 and close[end-1] > 0:
            p_ret = (close[end-1] - close[t]) / close[t]
            p_v = 1 if p_ret > 0.0003 else (-1 if p_ret < -0.0003 else 0)
        else:
            p_v = 0

        vote = ofi_v + cvd_v + vd_v + p_v

        if vote >= min_votes:
            labels[t] = 2
        elif vote <= -min_votes:
            labels[t] = 0

    return labels


# ─── BiLSTM (DirectML-safe, manual cells) ──────────────────────────────────
class BiLSTM(nn.Module):
    """Bidirectional LSTM using ManualLSTMCell — two passes, concat."""
    def __init__(self, inp, hid, layers, dropout):
        super().__init__()
        self.hid = hid; self.layers = layers

        # Forward LSTM
        self.fwd = nn.ModuleList([
            _ManualLSTMCell(inp if i == 0 else hid, hid)
            for i in range(layers)
        ])
        # Backward LSTM
        self.bwd = nn.ModuleList([
            _ManualLSTMCell(inp if i == 0 else hid, hid)
            for i in range(layers)
        ])
        self.drop = nn.Dropout(dropout)
        self.ln_fwd = nn.ModuleList([nn.LayerNorm(hid) for _ in range(layers)])
        self.ln_bwd = nn.ModuleList([nn.LayerNorm(hid) for _ in range(layers)])

    def _pass(self, x, cells, lns):
        """Single direction pass. x: (B, T, inp)"""
        B, T, _ = x.shape
        dev = x.device
        h = [torch.zeros(B, self.hid, device=dev) for _ in cells]
        c = [torch.zeros(B, self.hid, device=dev) for _ in cells]
        out = []
        for t in range(T):
            inp = x[:, t, :]
            for i, cell in enumerate(cells):
                h[i], c[i] = cell(inp, (h[i], c[i]))
                inp = lns[i](h[i])
                if i < len(cells) - 1:
                    inp = self.drop(inp)
            out.append(inp)
        return torch.stack(out, dim=1)  # (B, T, hid)

    def forward(self, x):
        fwd_out = self._pass(x, self.fwd, self.ln_fwd)
        bwd_out = self._pass(torch.flip(x, [1]), self.bwd, self.ln_bwd)
        bwd_out = torch.flip(bwd_out, [1])
        return torch.cat([fwd_out, bwd_out], dim=-1)  # (B, T, 2*hid)


# ─── Full Model ─────────────────────────────────────────────────────────────
class MomentumBiLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.bilstm = BiLSTM(N_F, HID, LAY, DR)
        self.ln = nn.LayerNorm(HID * 2)
        self.drop = nn.Dropout(DR)
        self.fc = nn.Linear(HID * 2, 3)

    def forward(self, x):
        out = self.bilstm(x)
        last = self.ln(out[:, -1, :])
        return self.fc(self.drop(last))


# ─── Focal Loss (DirectML-safe: weights on CPU) ────────────────────────────
class SafeFocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=1.0):
        super().__init__()
        self.alpha = alpha  # tensor [3]
        self.gamma = gamma

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, reduction='none')
        with torch.no_grad():
            pt = torch.exp(-ce.detach().cpu())
            fw = (1.0 - pt).clamp(1e-7, 0.9999).pow(self.gamma).to(logits.device)
        if self.alpha is not None:
            at = self.alpha.to(logits.device)[targets]
            return (at * fw * ce).mean()
        return (fw * ce).mean()


# ─── Data ──────────────────────────────────────────────────────────────────
class SeqDS(Dataset):
    def __init__(self, X, y): self.X = torch.from_numpy(X.astype(np.float32)); self.y = torch.from_numpy(y.astype(np.int64))
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]


def _pz(s, w=_ZWIN):
    s = pd.Series(s); m = s.rolling(w, min_periods=50).mean()
    st = s.rolling(w, min_periods=50).std().clip(lower=1e-8)
    return ((s - m) / st).clip(-4, 4).fillna(0).values.astype(np.float32)


def fs(X): n, s, f = X.shape; sc = RobustScaler(); sc.fit(X.reshape(-1, f)); return sc
def sx(X, sc): n, s, f = X.shape; return sc.transform(X.reshape(-1, f)).reshape(n, s, f).astype(np.float32)


def load_data(coins):
    Xs, ys, ts = [], [], []
    for coin in coins:
        fp = LABEL_DIR / f"{coin}_features_v3.parquet"
        if not fp.exists(): continue
        df = pd.read_parquet(fp).sort_index()
        df = df[df.index < TRAIN_CUTOFF_DATE]
        if len(df) < SEQ + 10: continue

        # Compute ADX
        h = df["high"].values if "high" in df.columns else df["close"].values
        l = df["low"].values if "low" in df.columns else df["close"].values
        c = df["close"].values
        adx, pdi, ndi, adxs = compute_adx(h, l, c, 14)

        # Compute improved labels
        labs = compute_labels_v3(df)

        # Fill NaN ADX
        adx = np.nan_to_num(adx, nan=0.0); pdi = np.nan_to_num(pdi, nan=0.0)
        ndi = np.nan_to_num(ndi, nan=0.0); adxs = np.nan_to_num(adxs, nan=0.0)

        # Build feature matrix
        fv = {}
        for col in OHLCV_FEATS:
            if col in df.columns:
                v = df[col].ffill().fillna(0).values.astype(np.float32)
                fv[col] = _pz(v.astype(np.float64)).astype(np.float32) if col in _PERCOIN_Z else v
            else:
                fv[col] = np.zeros(len(df), dtype=np.float32)

        fv["adx_14"] = adx.astype(np.float32)
        fv["plus_di_14"] = pdi.astype(np.float32)
        fv["minus_di_14"] = ndi.astype(np.float32)
        fv["adx_slope_14"] = adxs.astype(np.float32)

        Xc = np.column_stack([fv[c] for c in ALL_FEATS])
        yc = labs

        for i in range(SEQ - 1, len(Xc)):
            Xs.append(Xc[i - SEQ + 1:i + 1])
            ys.append(yc[i]); ts.append(df.index[i].value)

        n = len(yc)
        nb = (yc == 2).sum(); nn = (yc == 1).sum(); ns = (yc == 0).sum()
        logger.info(f"{coin}: {len(df):,}b | BULL={nb/n*100:.0f}% NEU={nn/n*100:.0f}% BEAR={ns/n*100:.0f}% | N={MOM_N},vote>={MOM_MIN_VOTES} | seqs={len(df)-SEQ+1:,}")

    X = np.stack(Xs); y = np.array(ys, dtype=np.int64); t = np.array(ts, dtype=np.int64)
    o = np.argsort(t)
    logger.info(f"Total {len(Xs):,} seqs X={X.shape}")
    return X[o], y[o], t[o]


def cw(y):
    uc, cnt = np.unique(y, return_counts=True); n = len(y)
    w = {c: n / (3.0 * cnt_i) for c, cnt_i in zip(uc, cnt)}
    if 1 in w: w[1] *= NEU_ALPHA
    return torch.tensor([w.get(i, 1.0) for i in range(3)], dtype=torch.float32)


def train_fold(Xtr, ytr, Xte, yte, fi):
    sc = fs(Xtr); Xtr_s = sx(Xtr, sc); del Xtr; gc.collect()
    Xte_s = sx(Xte, sc); del Xte; gc.collect()

    tr_ds = SeqDS(Xtr_s, ytr); te_ds = SeqDS(Xte_s, yte)
    tr_ld = DataLoader(tr_ds, batch_size=BATCH, shuffle=True)
    te_ld = DataLoader(te_ds, batch_size=BATCH, shuffle=False)

    model = MomentumBiLSTM().to(DEVICE)
    alpha = cw(ytr)
    crit = SafeFocalLoss(alpha=alpha, gamma=GAMMA)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)

    best, best_st, pc = -1.0, None, 0
    for ep in range(1, EPOCHS + 1):
        model.train()
        for xb, yb in tr_ld:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad(); crit(model(xb), yb).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()

        model.eval()
        pv, lv = [], []
        with torch.no_grad():
            for xb, yb in te_ld:
                pv.extend(model(xb.to(DEVICE)).argmax(1).cpu().numpy())
                lv.extend(yb.numpy())
        f1 = float(f1_score(lv, pv, average="macro", zero_division=0))

        if f1 > best: best, best_st, pc = f1, {k: v.cpu() for k, v in model.state_dict().items()}, 0
        else:
            pc += 1
            if pc >= PAT: break

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
         "f1_BEARISH": round(float(vp[0]), 4), "f1_NEUTRAL": round(float(vp[1]), 4),
         "f1_BULLISH": round(float(vp[2]), 4)}
    logger.info(f"[F{fi}] Train={tf1:.4f} Val={vf1:.4f} Gap={tf1-vf1:+.4f} | B={vp[0]:.3f} N={vp[1]:.3f} BU={vp[2]:.3f}")
    return model, sc, m


def retrain(X, y, eps):
    sc = fs(X); Xs = sx(X, sc); del X; gc.collect()
    ds = SeqDS(Xs, y); ld = DataLoader(ds, batch_size=BATCH, shuffle=True)

    model = MomentumBiLSTM().to(DEVICE)
    alpha = cw(y)
    crit = SafeFocalLoss(alpha=alpha, gamma=GAMMA)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)

    model.train()
    for ep in range(1, eps + 1):
        tl = 0.0
        for xb, yb in ld:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad(); tl += float(crit(model(xb), yb))
            crit(model(xb), yb).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        if ep % 10 == 0 or ep == 1:
            logger.info(f"[Final] E{ep:>3}/{eps} loss={tl/len(ld):.4f}")
    model.eval()
    return model, sc


def main():
    coins = TRAINING_COINS[:5]; run = "lstm_momentum_v7_B1"

    print(f"\n{'='*55}")
    print(f"  B1: Labels(N=6,vote>=3) + ADX + BiLSTM + Focal(gamma=1.0)")
    print(f"  Features:{N_F} (16 OHLCV + 4 ADX)  Seq:{SEQ}  Hidden:{HID}")
    print(f"  BiLSTM: 2x{HID}={HID*2}  Drop:{DR}  Focal:g={GAMMA} NEU_a={NEU_ALPHA}")
    print(f"  V3 baseline: 0.407 | Target: >0.43")
    print(f"{'='*55}\n")

    torch.manual_seed(42); np.random.seed(42)
    X, y, t = load_data(coins)

    rd = MODEL_DIR / "runs" / run; rd.mkdir(parents=True, exist_ok=True)
    json.dump(ALL_FEATS, open(rd / "feature_cols.json", "w"), indent=2)
    json.dump({"N": MOM_N, "min_votes": MOM_MIN_VOTES, "label_note": "shorter horizon + stricter voting"},
              open(rd / "label_params.json", "w"), indent=2)

    folds = build_purged_folds(pd.to_datetime(t, unit="ns", utc=True), N_FOLDS, PURGE_GAP_BARS)
    metrics = []
    for fi, (tr, te) in enumerate(folds):
        _, _, m = train_fold(X[tr], y[tr], X[te], y[te], fi + 1)
        metrics.append(m)

    final_eps = max(30, min(EPOCHS, int(np.median([m["val_f1"] for m in metrics])) + 5))
    logger.info(f"Retrain final {final_eps} epochs...")
    model, scaler = retrain(X, y, final_eps)

    torch.save(model.state_dict(), str(rd / "model.pt"))
    joblib.dump(scaler, rd / "scaler.pkl")

    vf = [m["val_f1"] for m in metrics]; tf = [m["train_f1"] for m in metrics]
    gf = [t - v for t, v in zip(tf, vf)]; nf = [m["f1_NEUTRAL"] for m in metrics]

    print(f"\n{'='*55}")
    print(f"  B1 COMPLETE")
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
