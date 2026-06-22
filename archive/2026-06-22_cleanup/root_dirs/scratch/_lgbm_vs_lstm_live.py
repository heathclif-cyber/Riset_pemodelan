import json, sys, warnings
import numpy as np, pandas as pd, joblib, torch
warnings.filterwarnings("ignore")
from pathlib import Path
ROOT=Path('.').resolve(); sys.path.insert(0,str(ROOT))
from config import LSTM_SEQ_LEN, MODEL_DIR, HOLDOUT_DIR
from core.models import load_lstm
from core.utils import ensure_utc_index
from pipeline.lstm_fusion_shared import load_hmm_cfg, apply_hmm_thr
from tools.live_db_bridge import load_trades

LSTM_DIR=MODEL_DIR/"runs"/"tb_lstm_genuine_v2"
lgbm=joblib.load(MODEL_DIR/"lgbm_baseline.pkl")
lgbm_feats=json.load(open(MODEL_DIR/"feature_cols_v2.json"))
lstm_feats=json.load(open(LSTM_DIR/"lstm_v4_selected_features.json"))
lstm=load_lstm(LSTM_DIR/"lstm_momentum.pt",device="cpu")
lstm_scaler=joblib.load(LSTM_DIR/"lstm_momentum_scaler.pkl")
hmm_cfg=load_hmm_cfg()

def lstm_proba(X,seq=LSTM_SEQ_LEN):
    n,f=X.shape; out=np.full((n,3),1/3,np.float32)
    if n<seq: return out
    Xs=lstm_scaler.transform(X).astype(np.float32)
    seqs=np.stack([Xs[i-seq+1:i+1] for i in range(seq-1,n)])
    ch=[]
    with torch.no_grad():
        for b in range(0,len(seqs),512):
            ch.append(torch.softmax(lstm(torch.from_numpy(seqs[b:b+512])),1).cpu().numpy())
    out[seq-1:]=np.concatenate(ch); return out

c=load_trades(); c=c[(c['is_live']==1)&(c['status']=='closed')].copy()
c['opened_at']=pd.to_datetime(c['opened_at'])
rows=[]
for sym,g in c.groupby('coin_symbol'):
    p=HOLDOUT_DIR/"labeled"/f"{sym}_features_v3.parquet"
    if not p.exists(): continue
    df=ensure_utc_index(pd.read_parquet(p)).sort_index()
    n=len(df)
    Xl=np.zeros((n,len(lgbm_feats)))
    for i,col in enumerate(lgbm_feats):
        if col in df: Xl[:,i]=df[col].ffill().fillna(0).values
    pr=lgbm.predict_proba(Xl); p0=pr[:,0]; p2=pr[:,2]
    Xs=np.zeros((n,len(lstm_feats)))
    for i,col in enumerate(lstm_feats):
        if col in df: Xs[:,i]=df[col].ffill().fillna(0).values
    lp=lstm_proba(Xs)
    hmm=df['hmm_regime_enc'].fillna(-1).values.astype(int) if 'hmm_regime_enc' in df else np.full(n,-1)
    ts=df.index.tz_localize(None) if df.index.tz else df.index
    sdf=pd.DataFrame({'p0':p0,'p2':p2,'l0':lp[:,0],'l1':lp[:,1],'l2':lp[:,2],'hmm':hmm}, index=ts)
    for _,r in g.iterrows():
        idx=sdf.index.get_indexer([r['opened_at']],method='nearest',tolerance=pd.Timedelta('59min'))
        if idx[0]==-1: continue
        row=sdf.iloc[idx[0]]
        rows.append({'coin':sym,'traded_dir':r['direction'],'pnl':r['pnl_net'],
            'lgbm_p0':row.p0,'lgbm_p2':row.p2,'lstm_l0':row.l0,'lstm_l1':row.l1,'lstm_l2':row.l2})

R=pd.DataFrame(rows)
print(f"matched {len(R)}/{len(c)} live trades to holdout bars")
R['lgbm_lean']=np.where(R['lgbm_p2']>R['lgbm_p0'],'LONG','SHORT')
lstm_arg=R[['lstm_l0','lstm_l1','lstm_l2']].values.argmax(1)
R['lstm_lean']=np.array(['SHORT','FLAT','LONG'])[lstm_arg]
R['agree']=np.where(R['lgbm_lean']==R['lstm_lean'],'agree',
            np.where(R['lstm_lean']=='FLAT','lstm_flat','CONTRADICT'))

def blk(g):
    pnl=g['pnl']; w=g[pnl>0]; l=g[pnl<0]; gl=abs(l['pnl'].sum())
    return pd.Series({'n':len(g),'wr%':round(len(w)/len(g)*100,1),'net':round(pnl.sum(),2),
        'pf':round(w['pnl'].sum()/gl,2) if gl>0 else np.inf,'avg':round(pnl.mean(),3)})

print("\n=== LGBM lean vs LSTM lean (count) ===")
print(pd.crosstab(R['lgbm_lean'],R['lstm_lean']))
print("\n=== Outcome by agreement ===")
print(R.groupby('agree').apply(blk))
print("\n=== CONTRADICT detail: lgbm_lean x traded_dir ===")
con=R[R['agree']=='CONTRADICT']
print(con.groupby(['lgbm_lean','lstm_lean','traded_dir']).apply(blk))
print("\n=== Specifically LGBM SHORT + LSTM LONG ===")
ss=R[(R['lgbm_lean']=='SHORT')&(R['lstm_lean']=='LONG')]
print(blk(ss) if len(ss) else 'none')
print("\n=== LGBM LONG + LSTM SHORT ===")
sl=R[(R['lgbm_lean']=='LONG')&(R['lstm_lean']=='SHORT')]
print(blk(sl) if len(sl) else 'none')

print("\n\n########## FOLLOW-UP: apakah fusi menimpa LGBM yang merusak? ##########")
R['followed_lgbm']=np.where(R['traded_dir']==R['lgbm_lean'],'followed_LGBM','OVERRODE_LGBM')
print("\n=== traded_dir vs lgbm_lean ===")
print(R.groupby('followed_lgbm').apply(blk))
print("\n=== executed LONG trades, by lgbm_lean (siapa sumber LONG) ===")
lg=R[R['traded_dir']=='LONG']
print(lg.groupby('lgbm_lean').apply(blk))
print("\n=== executed SHORT trades, by lgbm_lean ===")
sh=R[R['traded_dir']=='SHORT']
print(sh.groupby('lgbm_lean').apply(blk))
print("\n=== LSTM lean distribution (semua matched) ===")
print(R['lstm_lean'].value_counts())
print("LSTM mean probs: l0(short)=%.3f l1(flat)=%.3f l2(long)=%.3f" % (R.lstm_l0.mean(),R.lstm_l1.mean(),R.lstm_l2.mean()))
