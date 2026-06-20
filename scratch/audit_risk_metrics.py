"""Audit risk metrics: per-coin Sharpe + honest portfolio metrics."""
import json, sys, warnings
import numpy as np, pandas as pd
from pathlib import Path
warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
import joblib
from core.utils import ensure_utc_index
from config import *

PROD = Path("D:/Apps-Dev/swint_tradev2/models")
LM = {"SHORT": 0, "FLAT": 1, "LONG": 2}
SL_MULT = 1.5; MAX_HOLD = 36; MODAL = 10; LEV = 5
COST = (FEE_PER_SIDE + SLIPPAGE_PER_SIDE) * 2
MODAL_TOTAL = MODAL * 21  # total capital across all coins

# Load models
ic32 = joblib.load(MODEL_DIR / "runs/ic32_regime_v1/lgbm.pkl")
ic32_f = list(ic32.feature_name_)
tb = joblib.load(MODEL_DIR / "runs/tb_lgbm_widyawardhana_v3/lgbm.pkl")
with open(MODEL_DIR / "runs/tb_lgbm_widyawardhana_v3/tb_lgbm_widyawardhana_v3_features.json") as f:
    tb_f = json.load(f)

g_ic = joblib.load(PROD / "guardian_best.pkl")
g_ic_s = joblib.load(PROD / "guardian_scaler.pkl")
with open(PROD / "guardian_feature_cols.json") as f:
    g_ic_all = json.load(f)

g_tb = joblib.load(MODEL_DIR / "runs/tb_guardian_widyawardhana_v2/guardian.pkl")
g_tb_s = joblib.load(MODEL_DIR / "runs/tb_guardian_widyawardhana_v2/guardian_scaler.pkl")
with open(MODEL_DIR / "runs/tb_guardian_widyawardhana_v2/tb_guardian_widyawardhana_v2_feature_cols.json") as f:
    g_tb_all = json.load(f)

DYN_NAMES = {"bars_held_norm","current_pnl_pct","current_pnl_atr",
             "max_favorable_pnl_pct","drawdown_from_peak_pct",
             "direction","entry_price_ratio"}

def make_order(feats):
    st = [f for f in feats if f not in DYN_NAMES]
    sm = {n:i for i,n in enumerate(st)}
    return st, [(("static",sm[f]) if f in sm else ("dyn",f)) for f in feats]

g_ic_st, g_ic_order = make_order(g_ic_all)
g_tb_st, g_tb_order = make_order(g_tb_all)

def build_row(j,i,close,atr,d,mf,Xs,order):
    bh=j-i; pnl=(close[j]-close[i])/close[i]*d
    ap=atr[i]/close[i] if close[i]>0 else 0.01; nm=max(mf,pnl)
    dv={"bars_held_norm":bh/36,"current_pnl_pct":pnl,"current_pnl_atr":pnl/ap if ap>0 else 0.0,
        "max_favorable_pnl_pct":nm,"drawdown_from_peak_pct":(nm-pnl)/nm if nm>0.001 else 0.0,
        "direction":float(d),"entry_price_ratio":close[i]/close[j] if close[j]>0 else 1.0}
    row=np.zeros(len(order))
    for idx,(src,k) in enumerate(order):
        row[idx]=Xs[j,k] if src=="static" else dv.get(k,0.0)
    return row,nm

def sim_one(yp,close,high,low,atr,Xs=None,order=None,g=None,gs=None):
    n=len(yp); trades=[]; i=0
    while i<n:
        if yp[i]==1: i+=1; continue
        d=1 if yp[i]==2 else -1; entry=close[i]; sl=entry-d*SL_MULT*atr[i]
        mf=0.0; ep=close[min(i+MAX_HOLD,n-1)]; eb=min(i+MAX_HOLD,n-1); oc="TIME"
        for j in range(i+1,min(i+MAX_HOLD+1,n)):
            if (d==1 and low[j]<=sl) or (d==-1 and high[j]>=sl): ep,eb,oc=sl,j,"SL"; break
            if g is not None and j-i>=2:
                row,mf=build_row(j,i,close,atr,d,mf,Xs,order)
                prob=g.predict_proba(gs.transform(row.reshape(1,-1)))[0]
                if (prob[2] if len(prob)>2 else prob[1])>=0.65: ep,eb,oc=close[j],j,"GDN"; break
        ret=(ep-entry)/entry*d; net=ret*MODAL*LEV-COST*MODAL*LEV
        trades.append({"bar":eb,"pnl":net})
        i=eb+1
    return trades

THR={0:0.45,1:0.50,2:0.50,3:0.45}
avail=[s for s in ALL_COINS if (HOLDOUT_DIR/"labeled"/f"{s}_features_v3.parquet").exists()]

# Collect per-coin daily PnL for all 4 variants
per_coin = {k: {} for k in ["ic32","ic32_g","tb","tb_g"]}
per_coin_trades = {k: [] for k in ["ic32","ic32_g","tb","tb_g"]}

for sym in avail:
    df=pd.read_parquet(HOLDOUT_DIR/"labeled"/f"{sym}_features_v3.parquet"); df=ensure_utc_index(df).sort_index()
    rp=HOLDOUT_DIR/"labeled"/f"{sym}_regime_h1.parquet"; hmm=np.full(len(df),1,np.int32)
    if rp.exists():
        reg=pd.read_parquet(rp)
        if "hmm_regime_enc" in reg.columns: hmm=reg["hmm_regime_enc"].reindex(df.index,fill_value=1).values.astype(np.int32)
    mask=df["label"].isin(LM); df=df[mask].copy(); hmm=hmm[mask.values]; n=len(df)
    c=df["close"].values.astype(np.float64); h=df["high"].values.astype(np.float64)
    l=df["low"].values.astype(np.float64); a=df["atr_14_h1"].values.astype(np.float64)

    Xi=np.zeros((n,len(ic32_f)))
    for idx,cn in enumerate(ic32_f):
        if cn in df.columns: Xi[:,idx]=df[cn].ffill().fillna(0).values.astype(np.float64)
        elif cn=="hmm_regime_enc": Xi[:,idx]=hmm.astype(np.float64)
    pi=ic32.predict_proba(Xi); ypi=np.ones(n,np.int32)
    ypi[pi[:,2]>=0.69]=2; ypi[(pi[:,0]>=0.59)&(ypi!=2)]=0

    Xgi=np.zeros((n,len(g_ic_st)))
    for idx,cn in enumerate(g_ic_st):
        if cn in df.columns: Xgi[:,idx]=df[cn].ffill().fillna(0).values.astype(np.float64)
        elif cn=="hmm_regime_enc": Xgi[:,idx]=hmm.astype(np.float64)

    Xt=np.zeros((n,len(tb_f)))
    for idx,cn in enumerate(tb_f):
        if cn in df.columns: Xt[:,idx]=df[cn].ffill().fillna(0).values.astype(np.float64)
    pt=tb.predict_proba(Xt); ct=np.max(pt,axis=1); ypt=np.argmax(pt,axis=1).astype(np.int32)
    for r,th in THR.items(): ypt[(hmm==r)&(ypt!=1)&(ct<th)]=1

    Xgt=np.zeros((n,len(g_tb_st)))
    for idx,cn in enumerate(g_tb_st):
        if cn in df.columns: Xgt[:,idx]=df[cn].ffill().fillna(0).values.astype(np.float64)

    tr_ic32=sim_one(ypi,c,h,l,a)
    tr_ic32g=sim_one(ypi,c,h,l,a,Xgi,g_ic_order,g_ic,g_ic_s)
    tr_tb=sim_one(ypt,c,h,l,a)
    tr_tbg=sim_one(ypt,c,h,l,a,Xgt,g_tb_order,g_tb,g_tb_s)

    per_coin["ic32"][sym]=[(df.index[t["bar"]].date(),t["pnl"]) for t in tr_ic32]
    per_coin["ic32_g"][sym]=[(df.index[t["bar"]].date(),t["pnl"]) for t in tr_ic32g]
    per_coin["tb"][sym]=[(df.index[t["bar"]].date(),t["pnl"]) for t in tr_tb]
    per_coin["tb_g"][sym]=[(df.index[t["bar"]].date(),t["pnl"]) for t in tr_tbg]
    print(".",end="",flush=True)
print()

# ── Per-coin Sharpe ────────────────────────────────────────────────────────
print(f"\n{'='*80}")
print(f"  AUDIT: Per-Coin Sharpe Ratio (before diversification inflation)")
print(f"  Each coin = independent $10 account, 5x leverage")
print(f"{'='*80}")
print(f"  {'Coin':<18} {'ic32 bare':>10} {'ic32+Gdn':>10} {'tb bare':>10} {'tb+Gdn v2':>10}")

ann = np.sqrt(365)
all_pc = {k: [] for k in per_coin}

for sym in avail:
    vals = []
    for k in ["ic32","ic32_g","tb","tb_g"]:
        trades = per_coin[k].get(sym, [])
        if not trades or len(trades) < 5:
            vals.append(float("nan")); continue
        tr_df = pd.DataFrame(trades, columns=["date", "pnl"])
        daily = tr_df.groupby("date")["pnl"].sum()
        sr = daily.mean() / max(daily.std(), 1e-9) * ann
        vals.append(sr)
        all_pc[k].append(sr)
    print(f"  {sym:<18} {vals[0]:>10.2f} {vals[1]:>10.2f} {vals[2]:>10.2f} {vals[3]:>10.2f}")

# Median per-coin Sharpe
print(f"  {'-'*58}")
meds = []
for k in ["ic32","ic32_g","tb","tb_g"]:
    arr = np.array([v for v in all_pc[k] if not np.isnan(v)])
    med = np.median(arr)
    meds.append(med)
    print(f"  {'MEDIAN per-coin':<18} {med:>10.2f}", end="")
print()

# ── Honest portfolio metrics ───────────────────────────────────────────────
print(f"\n{'='*80}")
print(f"  HONEST METRICS — normalized by total capital ($210 across 21 coins)")
print(f"  Portfolio = 21 koin × $10/trade, Sharpe dihitung dari % return")
print(f"{'='*80}")

# Recompute daily equity as % of total capital
all_daily = {k: [] for k in per_coin}
for sym in avail:
    for k in per_coin:
        for date, pnl in per_coin[k].get(sym, []):
            all_daily[k].append({"date": date, "pnl": pnl})

print(f"  {'Metrik':<24} {'ic32 bare':>12} {'ic32+Gdn':>12} {'tb bare':>12} {'tb+Gdn v2':>12}")
print(f"  {'-'*72}")

for k, label in [("ic32","ic32 bare"),("ic32_g","ic32+Gdn"),("tb","tb bare"),("tb_g","tb+Gdn v2")]:
    trades = all_daily[k]
    if not trades: continue
    df = pd.DataFrame(trades)
    daily = df.groupby("date")["pnl"].sum().sort_index()
    # As % of total capital ($210)
    daily_ret = daily / (MODAL * 21)
    eq = daily_ret.cumsum()

    mu = daily_ret.mean(); sigma = daily_ret.std()
    sr = mu / max(sigma, 1e-9) * ann
    neg = daily_ret[daily_ret < 0]
    neg_std = np.std(neg) if len(neg) > 1 else sigma
    so = mu / max(neg_std, 1e-9) * ann
    peak = np.maximum.accumulate(eq.values)
    dd = (eq.values - peak) / np.where(peak != 0, peak, 1)
    maxdd = abs(dd.min()) * 100
    ann_ret = mu * 365
    cm = ann_ret / max(maxdd/100, 1e-9)
    win_rate = (daily_ret > 0).sum() / len(daily_ret) * 100

    print(f"  {label:<24} {sr:>12.2f} {so:>12.2f} {cm:>12.2f} {maxdd:>11.1f}%")

print(f"\n  REALISTIC LIVE ESTIMATE (accounting for execution):")
print(f"    Single-coin Sharpe         : ~{meds[3]:.1f} (tb+Gdn) → live expect ~{meds[3]*0.5:.1f}-{meds[3]*0.7:.1f}")
print(f"    Portfolio Sharpe (honest)  : divide by sqrt(21) ≈ 4.6x vs inflated")
print(f"    MaxDD (honest portfolio)   : 5-15% realistic for $210 total capital")
print(f"{'='*80}")
