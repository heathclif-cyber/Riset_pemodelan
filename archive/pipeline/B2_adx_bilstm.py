"""
B2: V3 baseline + 4 ADX features + BiLSTM + LayerNorm
Only change from V3: add ADX features, use bidirectional LSTM.
Labels: V2 (N=8, vote>=2) — proven balanced. Loss: CE+weights.
"""
import gc, json, sys, warnings, numpy as np, pandas as pd
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
from core.models import _ManualLSTMCell
from core.utils import setup_logger, get_lstm_device
from pipeline.shared import build_purged_folds

logger = setup_logger("B2"); DEVICE = get_lstm_device()

# ─── V3 baseline features + ADX ────────────────────────────────────────────
OHLCV_F = [
    "log_ret_1", "log_ret_5", "log_ret_20",
    "rsi_6", "rsi_h4",
    "h4_trend", "trend_strength",
    "ema_21_slope_h4", "ema_50_slope_h4",
    "price_vs_ema_50_h4",
    "cvd", "cvd_momentum_adv", "volume_delta",
    "Buy_Liq", "Sell_Liq",
    "whale_retail_divergence",
]
ADX_F = ["adx_14", "plus_di_14", "minus_di_14", "adx_slope"]
FEATS = OHLCV_F + ADX_F; NF = len(FEATS)
_PZ = {"cvd", "volume_delta", "Buy_Liq", "Sell_Liq"}; _ZW = 500

# V3 hyperparameters (unchanged)
SEQ = 16; HID = 96; LAY = 2; DR = 0.45
LR = LSTM_V2_LR; WD = LSTM_V2_WEIGHT_DECAY
BATCH = 128; EPOCHS = 80; PAT = 12


# ─── ADX ───────────────────────────────────────────────────────────────────
def adx_comp(h, l, c, p=14):
    n = len(h); tr = np.zeros(n); pdm = np.zeros(n); ndm = np.zeros(n)
    for i in range(1, n):
        hl = h[i] - l[i]; hc = abs(h[i] - c[i-1]); lc = abs(l[i] - c[i-1])
        tr[i] = max(hl, hc, lc)
        up = h[i] - h[i-1]; dn = l[i-1] - l[i]
        pdm[i] = up if up > dn and up > 0 else 0
        ndm[i] = dn if dn > up and dn > 0 else 0
    a = 1.0/p; atr=np.zeros(n); sp=np.zeros(n); sn=np.zeros(n)
    atr[p]=tr[1:p+1].mean(); sp[p]=pdm[1:p+1].mean(); sn[p]=ndm[1:p+1].mean()
    for i in range(p+1,n):
        atr[i]=atr[i-1]+a*(tr[i]-atr[i-1])
        sp[i]=sp[i-1]+a*(pdm[i]-sp[i-1]); sn[i]=sn[i-1]+a*(ndm[i]-sn[i-1])
    pdi=np.zeros(n); ndi=np.zeros(n); ax=np.zeros(n)
    for i in range(p,n):
        if atr[i]>0: pdi[i]=100*sp[i]/atr[i]; ndi[i]=100*sn[i]/atr[i]
        if pdi[i]+ndi[i]>0: ax[i]=100*abs(pdi[i]-ndi[i])/(pdi[i]+ndi[i])
    sx=np.zeros(n); sx[p*2-1]=ax[p:p*2].mean()
    for i in range(p*2,n): sx[i]=sx[i-1]+a*(ax[i]-sx[i-1])
    sl=np.zeros(n); sl[p*2:]=sx[p*2:]-sx[p*2-1:-1]
    return np.nan_to_num(sx,0).astype(np.float32), np.nan_to_num(pdi,0).astype(np.float32), np.nan_to_num(ndi,0).astype(np.float32), np.nan_to_num(sl,0).astype(np.float32)


# ─── BiLSTM (DirectML-safe) ────────────────────────────────────────────────
class BiLSTM(nn.Module):
    def __init__(self, inp, hid, layers, dropout):
        super().__init__()
        self.hid = hid; self.layers = layers
        self.fwd = nn.ModuleList([_ManualLSTMCell(inp if i==0 else hid, hid) for i in range(layers)])
        self.bwd = nn.ModuleList([_ManualLSTMCell(inp if i==0 else hid, hid) for i in range(layers)])
        self.drop = nn.Dropout(dropout)
        self.ln_f = nn.ModuleList([nn.LayerNorm(hid) for _ in range(layers)])
        self.ln_b = nn.ModuleList([nn.LayerNorm(hid) for _ in range(layers)])

    def _go(self, x, cells, lns):
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
        f = self._go(x, self.fwd, self.ln_f)
        b = self._go(torch.flip(x, [1]), self.bwd, self.ln_b)
        return torch.cat([f, torch.flip(b, [1])], dim=-1)


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.bilstm = BiLSTM(NF, HID, LAY, DR)
        self.ln = nn.LayerNorm(HID*2)
        self.drop = nn.Dropout(DR)
        self.fc = nn.Linear(HID*2, 3)
    def forward(self, x):
        out = self.bilstm(x)
        return self.fc(self.drop(self.ln(out[:, -1, :])))


# ─── Data ──────────────────────────────────────────────────────────────────
class DS(Dataset):
    def __init__(self, X, y): self.X=torch.from_numpy(X.astype(np.float32)); self.y=torch.from_numpy(y.astype(np.int64))
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]

def _pz(s, w=_ZW):
    s=pd.Series(s); m=s.rolling(w,min_periods=50).mean(); st=s.rolling(w,min_periods=50).std().clip(lower=1e-8)
    return ((s-m)/st).clip(-4,4).fillna(0).values.astype(np.float32)

def fs(X): n,s,f=X.shape; sc=RobustScaler(); sc.fit(X.reshape(-1,f)); return sc
def sx(X,sc): n,s,f=X.shape; return sc.transform(X.reshape(-1,f)).reshape(n,s,f).astype(np.float32)

def load(coins):
    Xs,ys,ts=[],[],[]
    for coin in coins:
        fp=LABEL_DIR/f"{coin}_features_v3.parquet"; lp=LABEL_DIR/f"{coin}_momentum_v2_labels.parquet"
        if not fp.exists() or not lp.exists(): continue
        df=pd.read_parquet(fp).sort_index(); df=df[df.index<TRAIN_CUTOFF_DATE]
        lbl=pd.read_parquet(lp).sort_index(); df=df.join(lbl["momentum_v2_label"],how="inner")
        df=df.dropna(subset=["momentum_v2_label"])
        if len(df)<SEQ+10: continue

        h=df["high"].values if "high" in df.columns else df["close"].values
        l=df["low"].values if "low" in df.columns else df["close"].values
        c=df["close"].values
        adx,pdi,ndi,adxs=adx_comp(h,l,c,14)

        fv={}
        for col in OHLCV_F:
            if col in df.columns:
                v=df[col].ffill().fillna(0).values.astype(np.float32)
                fv[col]=_pz(v.astype(np.float64)).astype(np.float32) if col in _PZ else v
            else: fv[col]=np.zeros(len(df),dtype=np.float32)
        fv["adx_14"]=adx; fv["plus_di_14"]=pdi; fv["minus_di_14"]=ndi; fv["adx_slope"]=adxs

        Xc=np.column_stack([fv[c] for c in FEATS])
        yc=df["momentum_v2_label"].values.astype(np.int64)
        for i in range(SEQ-1,len(Xc)): Xs.append(Xc[i-SEQ+1:i+1]); ys.append(yc[i]); ts.append(df.index[i].value)

        n=len(yc); nb=(yc==2).sum(); nn=(yc==1).sum(); ns=(yc==0).sum()
        logger.info(f"{coin}: {len(df):,}b | BULL={nb/n*100:.0f}% NEU={nn/n*100:.0f}% BEAR={ns/n*100:.0f}% | seqs={len(df)-SEQ+1:,}")

    X=np.stack(Xs); y=np.array(ys,dtype=np.int64); t=np.array(ts,dtype=np.int64); o=np.argsort(t)
    logger.info(f"Total {len(Xs):,} seqs X={X.shape}")
    return X[o],y[o],t[o]

def cw(y):
    uc,cnt=np.unique(y,return_counts=True); n=len(y)
    w={c:n/(3.0*ci) for c,ci in zip(uc,cnt)}
    return torch.tensor([w.get(i,1.0) for i in range(3)],dtype=torch.float32)

# ─── Training ──────────────────────────────────────────────────────────────
def train_fold(Xtr,ytr,Xte,yte,fi):
    sc=fs(Xtr); Xtr_s=sx(Xtr,sc); del Xtr; gc.collect()
    Xte_s=sx(Xte,sc); del Xte; gc.collect()
    tr=DataLoader(DS(Xtr_s,ytr),batch_size=BATCH,shuffle=True)
    te=DataLoader(DS(Xte_s,yte),batch_size=BATCH,shuffle=False)

    m=Model().to(DEVICE); w=cw(ytr).to(DEVICE)
    crit=nn.CrossEntropyLoss(weight=w); opt=torch.optim.Adam(m.parameters(),lr=LR,weight_decay=WD)

    best,bs,pc=-1.0,None,0
    for ep in range(1,EPOCHS+1):
        m.train()
        for xb,yb in tr: xb,yb=xb.to(DEVICE),yb.to(DEVICE); opt.zero_grad(); crit(m(xb),yb).backward(); nn.utils.clip_grad_norm_(m.parameters(),1.0); opt.step()
        m.eval(); pv,lv=[],[]
        with torch.no_grad():
            for xb,yb in te: pv.extend(m(xb.to(DEVICE)).argmax(1).cpu().numpy()); lv.extend(yb.numpy())
        f1=float(f1_score(lv,pv,average="macro",zero_division=0))
        if f1>best: best,bs,pc=f1,{k:v.cpu() for k,v in m.state_dict().items()},0
        else: pc+=1
        if pc>=PAT: break
        if ep%10==0 or ep==1: logger.info(f"[F{fi}] E{ep:>3} | F1={f1:.4f} Best={best:.4f}")

    m.load_state_dict(bs); m.eval(); pv,lv=[],[]
    with torch.no_grad():
        for xb,yb in te: pv.extend(m(xb.to(DEVICE)).argmax(1).cpu().numpy()); lv.extend(yb.numpy())
    vf1=float(f1_score(lv,pv,average="macro",zero_division=0))
    vp=f1_score(lv,pv,average=None,zero_division=0,labels=[0,1,2])
    tp,tl=[],[]
    with torch.no_grad():
        for xb,yb in tr: tp.extend(m(xb.to(DEVICE)).argmax(1).cpu().numpy()); tl.extend(yb.numpy())
    tf1=float(f1_score(tl,tp,average="macro",zero_division=0))
    met={"fold":fi,"train_f1":round(tf1,4),"val_f1":round(vf1,4),
         "f1_BEARISH":round(float(vp[0]),4),"f1_NEUTRAL":round(float(vp[1]),4),"f1_BULLISH":round(float(vp[2]),4)}
    logger.info(f"[F{fi}] Train={tf1:.4f} Val={vf1:.4f} Gap={tf1-vf1:+.4f} | B={vp[0]:.3f} N={vp[1]:.3f} BU={vp[2]:.3f}")
    return m,sc,met

def retrain(X,y,eps):
    sc=fs(X); Xs=sx(X,sc); del X; gc.collect()
    ld=DataLoader(DS(Xs,y),batch_size=BATCH,shuffle=True)
    m=Model().to(DEVICE); w=cw(y).to(DEVICE); crit=nn.CrossEntropyLoss(weight=w)
    opt=torch.optim.Adam(m.parameters(),lr=LR,weight_decay=WD)
    m.train()
    for ep in range(1,eps+1):
        tl=0.0
        for xb,yb in ld: xb,yb=xb.to(DEVICE),yb.to(DEVICE); opt.zero_grad(); l=crit(m(xb),yb); l.backward(); nn.utils.clip_grad_norm_(m.parameters(),1.0); opt.step(); tl+=float(l)
        if ep%10==0 or ep==1: logger.info(f"[Final] E{ep:>3}/{eps} loss={tl/len(ld):.4f}")
    m.eval(); return m,sc

def main():
    coins=TRAINING_COINS[:5]; run="lstm_momentum_v7_B2"
    print(f"\n{'='*55}")
    print(f"  B2: V3 labels + 4 ADX + BiLSTM + CE loss")
    print(f"  Features:{NF} (16 OHLCV + 4 ADX)  Seq:{SEQ}  Hidden:{HID}")
    print(f"  BiLSTM: 2x{HID}={HID*2}  Drop:{DR}  Loss:CE+weights")
    print(f"  V3 baseline: 0.407 | Target: >0.415")
    print(f"{'='*55}\n")

    torch.manual_seed(42); np.random.seed(42)
    X,y,t=load(coins)
    rd=MODEL_DIR/"runs"/run; rd.mkdir(parents=True,exist_ok=True)
    json.dump(FEATS,open(rd/"feature_cols.json","w"),indent=2)

    folds=build_purged_folds(pd.to_datetime(t,unit="ns",utc=True),N_FOLDS,PURGE_GAP_BARS)
    metrics=[]
    for fi,(tr,te) in enumerate(folds):
        _,_,m=train_fold(X[tr],y[tr],X[te],y[te],fi+1); metrics.append(m)

    eps=max(25,min(EPOCHS,int(np.median([m["val_f1"] for m in metrics]))+5))
    logger.info(f"Retrain final {eps} epochs...")
    model,scaler=retrain(X,y,eps)
    torch.save(model.state_dict(),str(rd/"model.pt")); joblib.dump(scaler,rd/"scaler.pkl")

    vf=[m["val_f1"] for m in metrics]; tf=[m["train_f1"] for m in metrics]
    gf=[t-v for t,v in zip(tf,vf)]; nf=[m["f1_NEUTRAL"] for m in metrics]

    print(f"\n{'='*55}")
    print(f"  B2 COMPLETE")
    print(f"  Val F1:    {np.mean(vf):.4f} +/- {np.std(vf):.4f}")
    print(f"  NEUTRAL F1:{np.mean(nf):.4f}")
    print(f"  Gap:       {np.mean(gf):+.4f}")
    print(f"  V3:        0.407 +/- 0.007")
    for m in metrics:
        g=m["train_f1"]-m["val_f1"]
        print(f"  F{m['fold']}: Train={m['train_f1']:.4f} Val={m['val_f1']:.4f} Gap={g:+.4f} | B={m['f1_BEARISH']:.3f} N={m['f1_NEUTRAL']:.3f} BU={m['f1_BULLISH']:.3f}")
    print(f"\n  Model: {rd/'model.pt'}")
    print(f"{'='*55}")

if __name__=="__main__": main()
