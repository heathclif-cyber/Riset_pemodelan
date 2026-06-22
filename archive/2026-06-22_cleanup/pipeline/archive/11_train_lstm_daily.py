"""
pipeline/11_train_lstm_daily.py — LSTM Daily Momentum (Binary)

Features: 5 per-coin (OI+LS) + 5 OHLCV daily + 2 global + 5 masks = 17
Target:   Return 7D > 2% (BULL) vs < -2% (BEAR) — binary, NO NEUTRAL
Mask:     1=positioning data available, 0=period before Coinank (2020-2024)
Model:    BiLSTM (ManualLSTMCell, DirectML-safe), 32-bar daily sequence
Use:      Weekly directional bias → block counter-trend entries in TRENDING
"""
import gc, json, sys, warnings, numpy as np, pandas as pd
from pathlib import Path

ROOT = Path(__file__).parent.parent; sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

import torch, torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, precision_score, recall_score
import joblib

from config import TRAINING_COINS, LABEL_DIR, TRAIN_CUTOFF_DATE, MODEL_DIR, N_FOLDS, PURGE_GAP_BARS
from core.models import _ManualLSTMCell
from core.utils import setup_logger, get_lstm_device
from pipeline.shared import build_purged_folds

logger = setup_logger("11_lstm_daily"); DEVICE = get_lstm_device()

# ─── Config ──────────────────────────────────────────────────────────────
SEQ_LEN = 32; HIDDEN = 96; LAYERS = 2; DROPOUT = 0.40; N_HEADS = 4
LR = 5e-4; WD = 0.01; BATCH = 64; EPOCHS = 80; PATIENCE = 15

COINANK_DIR = ROOT / "data" / "coinank"
MACRO_DIR = ROOT / "data" / "macro"
RET_THRESHOLD = 0.02   # 2% return = BULL/BEAR threshold

# ─── BiLSTM (DirectML-safe) ──────────────────────────────────────────────
class BiLSTM(nn.Module):
    def __init__(self, inp, hid, layers, dropout):
        super().__init__()
        self.hid = hid; self.layers = layers
        self.fwd = nn.ModuleList([_ManualLSTMCell(inp if i==0 else hid, hid) for i in range(layers)])
        self.bwd = nn.ModuleList([_ManualLSTMCell(inp if i==0 else hid, hid) for i in range(layers)])
        self.drop = nn.Dropout(dropout)
        self.ln_f = nn.ModuleList([nn.LayerNorm(hid) for _ in range(layers)])
        self.ln_b = nn.ModuleList([nn.LayerNorm(hid) for _ in range(layers)])

    def _pass(self, x, cells, lns):
        B, T, _ = x.shape; dev = x.device
        h = [torch.zeros(B, self.hid, device=dev) for _ in cells]
        c = [torch.zeros(B, self.hid, device=dev) for _ in cells]
        out = []
        for t in range(T):
            inp = x[:, t, :]
            for i, cell in enumerate(cells):
                h[i], c[i] = cell(inp, (h[i], c[i]))
                inp = lns[i](h[i])
                if i < len(cells)-1: inp = self.drop(inp)
            out.append(inp)
        return torch.stack(out, dim=1)

    def forward(self, x):
        f = self._pass(x, self.fwd, self.ln_f)
        b = self._pass(torch.flip(x, [1]), self.bwd, self.ln_b)
        return torch.cat([f, torch.flip(b, [1])], dim=-1)

class DailyLSTM(nn.Module):
    def __init__(self, n_feat):
        super().__init__()
        self.bilstm = BiLSTM(n_feat, HIDDEN, LAYERS, DROPOUT)
        self.ln = nn.LayerNorm(HIDDEN*2)
        self.drop = nn.Dropout(DROPOUT)
        self.fc = nn.Linear(HIDDEN*2, 1)
    def forward(self, x): return torch.sigmoid(self.fc(self.drop(self.ln(self.bilstm(x)[:, -1, :])))).squeeze(-1)

# ─── Data ────────────────────────────────────────────────────────────────
class DS(Dataset):
    def __init__(self, X, y): self.X=torch.from_numpy(X.astype(np.float32)); self.y=torch.from_numpy(y.astype(np.float32))
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]

def load_positioning():
    """Load per-coin positioning data and global macro data."""
    pos_data, macro = {}, {}

    for coin in TRAINING_COINS:
        oi_p = COINANK_DIR / f"{coin}_oi.parquet"
        lsp_p = COINANK_DIR / f"{coin}_ls_position.parquet"
        lsa_p = COINANK_DIR / f"{coin}_ls_account.parquet"
        if not oi_p.exists(): continue

        oi = pd.read_parquet(oi_p).sort_index()
        lsp = pd.read_parquet(lsp_p).sort_index() if lsp_p.exists() else None
        lsa = pd.read_parquet(lsa_p).sort_index() if lsa_p.exists() else None

        daily = pd.DataFrame(index=oi.index)
        oi_cols = [c for c in oi.columns if c.startswith("oi_") or "oi" in c.lower()]
        oi_t = oi[oi_cols[0]].copy() if oi_cols else None
        if oi_t is None: continue

        om = oi_t.rolling(20).mean(); os = oi_t.rolling(20).std().clip(lower=1e-8)
        daily["oi_z20"] = (oi_t - om) / os
        daily["oi_d7"] = oi_t.pct_change(7)

        if lsp is not None and "top_trader_position_ls" in lsp.columns:
            ls = lsp["top_trader_position_ls"]
            lm = ls.rolling(20).mean(); ls_s = ls.rolling(20).std().clip(lower=1e-8)
            daily["ls_z20"] = (ls - lm) / ls_s
            daily["ls_d7"] = ls.diff(7)
            if lsa is not None and "top_trader_account_ls" in lsa.columns:
                daily["smart_retail"] = ls - lsa["top_trader_account_ls"]

        pos_data[coin] = daily

    # Global data
    fg_p = MACRO_DIR / "fear_greed.parquet"
    etf_p = MACRO_DIR / "etf_btc_combined.parquet"

    if fg_p.exists():
        fg = pd.read_parquet(fg_p)
        if "fear_greed_value" in fg.columns: macro["fear_greed"] = fg[["fear_greed_value"]]

    if etf_p.exists():
        etf = pd.read_parquet(etf_p)
        if "btc_etf_volume_usd" in etf.columns:
            macro["etf_btc"] = etf[["btc_etf_volume_usd"]]

    return pos_data, macro

def build_daily_dataset(coins, pos_data, macro):
    """Build daily features + binary target for all coins."""
    Xs, ys = [], []; total_seq = 0; total_bull = 0; total_bear = 0

    for coin in coins:
        fp = LABEL_DIR / f"{coin}_features_v3.parquet"
        if not fp.exists(): continue

        df = pd.read_parquet(fp).sort_index()
        df = df[df.index < TRAIN_CUTOFF_DATE]

        # Resample H1 → daily
        daily = df[["open","high","low","close","volume"]].resample("1D").agg({
            "open":"first","high":"max","low":"min","close":"last","volume":"sum"}).dropna()

        # OHLCV features
        daily["ret_1d"] = daily["close"].pct_change(1)
        daily["ret_5d"] = daily["close"].pct_change(5)
        daily["range_pct"] = (daily["high"] - daily["low"]) / daily["close"]
        daily["vol_chg"] = daily["volume"].pct_change(1)
        atr = (daily["high"] - daily["low"]).rolling(14).mean()
        daily["atr_14d"] = atr / daily["close"]

        # Positioning features
        if coin in pos_data:
            pos = pos_data[coin]
            date_idx = pd.to_datetime(daily.index.date, utc=True)
            pos_idx = pd.to_datetime(pos.index.date, utc=True)
            # Align
            for col in ["oi_z20","oi_d7","ls_z20","ls_d7","smart_retail"]:
                if col in pos.columns:
                    s = pos[col].copy()
                    s.index = pd.to_datetime(s.index.date, utc=True)
                    s = s[~s.index.duplicated(keep="last")]
                    daily[col] = s.reindex(date_idx).ffill().fillna(0).values
                    # Mask: 1 = has real data, 0 = filled (no Coinank data)
                    daily[f"{col}_avl"] = (daily[col] != 0).astype(float)
                else:
                    daily[col] = 0.0
                    daily[f"{col}_avl"] = 0.0
        else:
            for col in ["oi_z20","oi_d7","ls_z20","ls_d7","smart_retail"]:
                daily[col] = 0.0; daily[f"{col}_avl"] = 0.0

        # Global features
        date_idx = pd.to_datetime(daily.index.date, utc=True)
        if "fear_greed" in macro and macro["fear_greed"] is not None:
            fg = macro["fear_greed"].copy()
            fg.index = pd.to_datetime(fg.index.date, utc=True)
            fg = fg[~fg.index.duplicated(keep="last")]
            daily["fear_greed"] = (fg["fear_greed_value"].reindex(date_idx).ffill().fillna(50) - 50) / 25
        else:
            daily["fear_greed"] = 0.0

        if "etf_btc" in macro and macro["etf_btc"] is not None:
            etf = macro["etf_btc"].copy()
            etf.index = pd.to_datetime(etf.index.date, utc=True)
            etf = etf[~etf.index.duplicated(keep="last")]
            etf_d5 = etf["btc_etf_volume_usd"].reindex(date_idx).ffill().fillna(0).pct_change(5).fillna(0)
            daily["etf_btc_d5"] = etf_d5.values
        else:
            daily["etf_btc_d5"] = 0.0

        # Target: return 7D
        daily["fwd_ret_7d"] = daily["close"].pct_change(7).shift(-7)

        # Drop NaN
        daily = daily.dropna(subset=["fwd_ret_7d"] + ["oi_z20","ls_z20","fear_greed"])
        if len(daily) < SEQ_LEN + 10: continue

        # Binary target
        daily["target"] = 0.0  # default BEAR
        daily.loc[daily["fwd_ret_7d"] > RET_THRESHOLD, "target"] = 1.0   # BULL
        # Drop NETRAL (|ret| < 2%) — not used for binary training
        mask = daily["fwd_ret_7d"].abs() > RET_THRESHOLD
        daily = daily[mask]

        # Feature columns
        feat_cols = ["ret_1d","ret_5d","range_pct","vol_chg","atr_14d",
                     "oi_z20","oi_d7","ls_z20","ls_d7","smart_retail",
                     "fear_greed","etf_btc_d5"]
        mask_cols = [f"{c}_avl" for c in ["oi_z20","oi_d7","ls_z20","ls_d7","smart_retail"]]
        all_cols = feat_cols + mask_cols

        # Build sequences — clip extremes, replace inf
        Xc = daily[all_cols].fillna(0).clip(-10, 10).replace([np.inf, -np.inf], 0).values.astype(np.float32)
        yc = daily["target"].values.astype(np.float32)

        for i in range(SEQ_LEN - 1, len(Xc)):
            Xs.append(Xc[i - SEQ_LEN + 1:i + 1])
            ys.append(yc[i])
            total_seq += 1
            if yc[i] > 0.5: total_bull += 1
            else: total_bear += 1

        nb = int((yc > 0.5).sum()); ns = int((yc < 0.5).sum())
        logger.info(f"{coin}: {len(daily)} daily bars | BULL={nb} BEAR={ns} | seqs={len(daily)-SEQ_LEN+1}")

    X = np.stack(Xs); y = np.array(ys, dtype=np.float32)
    logger.info(f"TOTAL: {total_seq} seqs | BULL={total_bull} ({total_bull/total_seq*100:.0f}%) BEAR={total_bear}")
    return X, y, all_cols

# ─── Training ────────────────────────────────────────────────────────────
def fs(X): n,s,f=X.shape; sc=RobustScaler(); sc.fit(X.reshape(-1,f)); return sc
def sx(X,sc): n,s,f=X.shape; return sc.transform(X.reshape(-1,f)).reshape(n,s,f).astype(np.float32)

def cw(y):
    nb = (y>0.5).sum(); ns = (y<0.5).sum(); n = len(y)
    return torch.tensor([n/(2*ns) if ns>0 else 1.0, n/(2*nb) if nb>0 else 1.0], dtype=torch.float32)

def train_fold(Xtr, ytr, Xte, yte, fi):
    sc = fs(Xtr); Xtr_s = sx(Xtr, sc); del Xtr; gc.collect()
    Xte_s = sx(Xte, sc); del Xte; gc.collect()
    tr_ld = DataLoader(DS(Xtr_s, ytr), batch_size=BATCH, shuffle=True)
    te_ld = DataLoader(DS(Xte_s, yte), batch_size=BATCH, shuffle=False)

    model = DailyLSTM(Xtr_s.shape[2]).to(DEVICE)
    w = cw(ytr).to(DEVICE)
    crit = nn.BCELoss()  # Simple BCE for binary
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)

    best_auc, best_st, pc = 0.0, None, 0
    for ep in range(1, EPOCHS+1):
        model.train()
        for xb, yb in tr_ld:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad(); crit(model(xb), yb).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()

        model.eval(); pv, lv = [], []
        with torch.no_grad():
            for xb, yb in te_ld:
                pv.extend(model(xb.to(DEVICE)).cpu().numpy()); lv.extend(yb.numpy())
        auc = float(roc_auc_score(lv, pv)) if len(np.unique(lv))>1 else 0.5
        if auc > best_auc: best_auc, best_st, pc = auc, {k:v.cpu() for k,v in model.state_dict().items()}, 0
        else: pc += 1
        if pc >= PATIENCE: break
        if ep%10==0 or ep==1: logger.info(f"[F{fi}] E{ep:>3} AUC={auc:.4f} Best={best_auc:.4f}")

    model.load_state_dict(best_st); model.eval(); pv, lv = [], []
    with torch.no_grad():
        for xb, yb in te_ld: pv.extend(model(xb.to(DEVICE)).cpu().numpy()); lv.extend(yb.numpy())
    pv_a = (np.array(pv) > 0.5).astype(int)
    return {"fold":fi, "auc":float(roc_auc_score(lv,pv)), "f1":float(f1_score(lv,pv_a)),
            "acc":float(accuracy_score(lv,pv_a)), "prec":float(precision_score(lv,pv_a)),
            "rec":float(recall_score(lv,pv_a)), "n_test":len(lv)}

def retrain(X, y, eps):
    sc = fs(X); Xs = sx(X, sc); del X; gc.collect()
    ld = DataLoader(DS(Xs, y), batch_size=BATCH, shuffle=True)
    model = DailyLSTM(Xs.shape[2]).to(DEVICE)
    crit = nn.BCELoss(); opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    model.train()
    for ep in range(1, eps+1):
        tl = 0.0
        for xb, yb in ld:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad(); l=crit(model(xb), yb); l.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); tl += float(l)
        if ep%10==0 or ep==1: logger.info(f"[Final] E{ep:>3}/{eps} loss={tl/len(ld):.4f}")
    model.eval(); return model, sc

# ─── Main ────────────────────────────────────────────────────────────────
def main():
    coins = TRAINING_COINS
    print(f"\n{'='*60}")
    print(f"  LSTM DAILY MOMENTUM — Binary Classifier")
    print(f"  Features: 5 OHLCV + 5 positioning + 2 global + 5 masks = 17")
    print(f"  Target: return 7D > 2% (BULL) vs < -2% (BEAR)")
    print(f"  Seq: {SEQ_LEN} daily bars | BiLSTM {HIDDEN} hidden")
    print(f"  Baseline random: AUC 0.50 | Target: AUC > 0.60")
    print(f"{'='*60}\n")

    print("Loading positioning data...")
    pos_data, macro = load_positioning()
    print(f"  Coins with POS data: {len(pos_data)}")
    print(f"  Global data: {list(macro.keys())}")

    print("\nBuilding daily dataset...")
    X, y, feat_cols = build_daily_dataset(coins, pos_data, macro)

    ts_index = pd.date_range("2020-01-01", periods=len(X), freq="1D")
    folds = build_purged_folds(ts_index, N_FOLDS, max(1, PURGE_GAP_BARS // 24))  # daily purge

    metrics = []
    for fi, (tr, te) in enumerate(folds):
        m = train_fold(X[tr], y[tr], X[te], y[te], fi+1); metrics.append(m)

    final_eps = max(30, min(EPOCHS, int(np.median([m.get("auc",0.5)*50 for m in metrics])) + 5))
    logger.info(f"Retrain final {final_eps} epochs...")
    model, scaler = retrain(X, y, final_eps)

    run_dir = MODEL_DIR / "runs" / "lstm_daily_v1"
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), str(run_dir / "model.pt"))
    joblib.dump(scaler, run_dir / "scaler.pkl")
    json.dump(feat_cols, open(run_dir / "feature_cols.json", "w"), indent=2)
    json.dump(metrics, open(run_dir / "cv_results.json", "w"), indent=2)

    aucs = [m["auc"] for m in metrics]
    f1s = [m["f1"] for m in metrics]
    print(f"\n{'='*60}")
    print(f"  LSTM DAILY RESULTS")
    print(f"  Mean AUC: {np.mean(aucs):.4f} +/- {np.std(aucs):.4f}")
    print(f"  Mean F1:  {np.mean(f1s):.4f} +/- {np.std(f1s):.4f}")
    print(f"  Random:   AUC 0.50")
    print(f"  Folds:    {[f'{a:.4f}' for a in aucs]}")
    print(f"  Model:    {run_dir / 'model.pt'}")
    print(f"{'='*60}\n")

if __name__ == "__main__": main()
