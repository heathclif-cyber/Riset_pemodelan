"""
pipeline/11_holdout_binary_meta_tb.py
Holdout evaluation: TB LGBM + Binary LSTM Meta + Guardian

Variants:
  1. tb+Guardian              (baseline)
  2. tb+BinaryMeta(0.50)+Guardian
  3. tb+BinaryMeta(0.55)+Guardian
  4. tb+BinaryMeta(0.60)+Guardian

Binary LSTM Meta filter: skip entry jika p_win < threshold
Guardian: exit early per bar seperti biasa
"""
import json, sys, warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
import joblib
from core.utils import setup_logger, ensure_utc_index
from config import *

logger = setup_logger("11_holdout_binary_meta")

# ── Constants ──────────────────────────────────────────────────────────────────
SL_MULT  = TP_SL_FALLBACK_SL
MAX_HOLD = MAX_HOLDING_BARS
MODAL    = MODAL_PER_TRADE
LEVERAGE = LEVERAGE_SIM[0] if isinstance(LEVERAGE_SIM, list) else LEVERAGE_SIM
COST_RT  = (FEE_PER_SIDE + SLIPPAGE_PER_SIDE) * 2
REGIME_THRESH = {0: 0.45, 1: 0.50, 2: 0.50, 3: 0.45}
GDN_MIN  = 2
GDN_THR  = 0.65
SEQ_LEN  = 32
META_THRESHOLDS = [0.50, 0.55, 0.60]
DYNAMIC_NAMES = frozenset({
    "bars_held_norm", "current_pnl_pct", "current_pnl_atr",
    "max_favorable_pnl_pct", "drawdown_from_peak_pct",
    "direction", "entry_price_ratio",
})

# ── Binary LSTM Meta architecture (sama dengan training) ───────────────────────
class BinaryLSTMMeta(nn.Module):
    def __init__(self, n_feat, hidden=32, n_layers=1, dropout=0.5):
        super().__init__()
        self.lstm = nn.LSTM(n_feat, hidden, n_layers, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.fc   = nn.Linear(hidden, 1)

    def forward(self, x):
        _, (h, _) = self.lstm(x)
        return torch.sigmoid(self.fc(self.drop(h[-1])))


# ── Load models ────────────────────────────────────────────────────────────────
tb_model = joblib.load(MODEL_DIR / "runs/tb_lgbm_widyawardhana_v3/lgbm.pkl")
with open(MODEL_DIR / "runs/tb_lgbm_widyawardhana_v3/tb_lgbm_widyawardhana_v3_features.json") as f:
    tb_feats = json.load(f)

META_RUN = MODEL_DIR / "runs/tb_lstm_binary_meta_v1"
with open(META_RUN / "tb_lstm_binary_meta_v1_features.json") as f:
    meta_feats = json.load(f)
meta_scaler = joblib.load(META_RUN / "lstm_binary_meta_scaler.pkl")
n_feat_total = len(meta_feats) + 1  # +1 direction
meta_model = BinaryLSTMMeta(n_feat_total, hidden=32, n_layers=1, dropout=0.5)
meta_model.load_state_dict(torch.load(META_RUN / "lstm_binary_meta.pt", map_location="cpu"))
meta_model.eval()

tbg_model  = joblib.load(MODEL_DIR / "runs/tb_guardian_widyawardhana_v2/guardian.pkl")
tbg_scaler = joblib.load(MODEL_DIR / "runs/tb_guardian_widyawardhana_v2/guardian_scaler.pkl")
with open(MODEL_DIR / "runs/tb_guardian_widyawardhana_v2/tb_guardian_widyawardhana_v2_feature_cols.json") as f:
    tbg_all_feats = json.load(f)

tbg_static = [f for f in tbg_all_feats if f not in DYNAMIC_NAMES]
tbg_smap   = {n: i for i, n in enumerate(tbg_static)}
tbg_order  = [("static", tbg_smap[f]) if f not in DYNAMIC_NAMES else ("dyn", f)
               for f in tbg_all_feats]

available = [s for s in ALL_COINS
             if (HOLDOUT_DIR / "labeled" / f"{s}_features_v3.parquet").exists()]

print(f"\n{'='*72}")
print(f"  HOLDOUT: TB + Binary LSTM Meta + Guardian | Nov 2025–Apr 2026")
print(f"  Binary Meta thresholds: {META_THRESHOLDS}")
print(f"  Coins: {len(available)} | ${MODAL}/trade {LEVERAGE}x")
print(f"{'='*72}\n")


# ── Guardian row builder ───────────────────────────────────────────────────────
def gdn_row(j, i, close, atr, direction, max_fav, X_static):
    bh  = j - i
    pnl = (close[j] - close[i]) / close[i] * direction
    atp = atr[i] / close[i] if close[i] > 0 else 0.01
    nmx = max(max_fav, pnl)
    dyn = {
        "bars_held_norm"        : bh / MAX_HOLD,
        "current_pnl_pct"       : pnl,
        "current_pnl_atr"       : pnl / atp if atp > 0 else 0.0,
        "max_favorable_pnl_pct" : nmx,
        "drawdown_from_peak_pct": (nmx - pnl) / nmx if nmx > 0.001 else 0.0,
        "direction"             : float(direction),
        "entry_price_ratio"     : close[i] / close[j] if close[j] > 0 else 1.0,
    }
    row = np.zeros(len(tbg_order), dtype=np.float64)
    for k, (src, key) in enumerate(tbg_order):
        row[k] = X_static[j, key] if src == "static" else dyn.get(key, 0.0)
    return row, nmx


# ── Simulate ───────────────────────────────────────────────────────────────────
def simulate(yp, close, high, low, atr, X_tbg, use_guardian):
    n = len(yp); trades = []; i = 0
    while i < n:
        sig = yp[i]
        if sig == 1: i += 1; continue
        direction = 1 if sig == 2 else -1
        entry    = close[i]
        sl_price = entry - direction * SL_MULT * atr[i]
        max_fav  = 0.0
        exit_p, exit_b, outcome = close[min(i+MAX_HOLD,n-1)], min(i+MAX_HOLD,n-1), "TIME"
        for j in range(i+1, min(i+MAX_HOLD+1, n)):
            if direction==1 and low[j]<=sl_price: exit_p,exit_b,outcome=sl_price,j,"SL"; break
            if direction==-1 and high[j]>=sl_price: exit_p,exit_b,outcome=sl_price,j,"SL"; break
            if use_guardian and j-i >= GDN_MIN:
                row, max_fav = gdn_row(j, i, close, atr, direction, max_fav, X_tbg)
                sc = tbg_scaler.transform(row.reshape(1,-1))
                prob = tbg_model.predict_proba(sc)[0]
                ep   = prob[2] if len(prob) > 2 else prob[1]
                if ep >= GDN_THR: exit_p,exit_b,outcome=close[j],j,"GDN"; break
            else:
                max_fav = max(max_fav, (close[j]-entry)/entry*direction)
        ret = (exit_p-entry)/entry*direction
        net = ret*MODAL*LEVERAGE - COST_RT*MODAL*LEVERAGE
        trades.append({"net_pnl":net, "outcome":outcome,
                       "direction":"LONG" if direction==1 else "SHORT",
                       "bars_held":exit_b-i})
        i = exit_b + 1
    return trades


def new_agg():
    return {"trades":0,"wins":0,"pnl":0.0,"longs":0,"long_wins":0,
            "sl":0,"gdn":0,"bars":[]}

def upd(a, trades):
    for t in trades:
        a["trades"]+=1; a["pnl"]+=t["net_pnl"]; a["bars"].append(t["bars_held"])
        if t["net_pnl"]>0: a["wins"]+=1
        if t["outcome"]=="SL": a["sl"]+=1
        if t["outcome"]=="GDN": a["gdn"]+=1
        if t["direction"]=="LONG":
            a["longs"]+=1
            if t["net_pnl"]>0: a["long_wins"]+=1


# Keys: "tb_gdn" + "tb_meta_{thr}_gdn" for each threshold
variant_keys = ["tb_gdn"] + [f"tb_meta{int(t*100)}_gdn" for t in META_THRESHOLDS]
agg = {k: new_agg() for k in variant_keys}

for sym in available:
    df = pd.read_parquet(HOLDOUT_DIR / "labeled" / f"{sym}_features_v3.parquet")
    df = ensure_utc_index(df).sort_index()
    mask = df["label"].isin({"SHORT":0,"FLAT":1,"LONG":2})
    df   = df[mask].copy()
    n    = len(df)

    rp  = HOLDOUT_DIR / "labeled" / f"{sym}_regime_h1.parquet"
    hmm = np.full(n, 1, np.int32)
    if rp.exists():
        reg = pd.read_parquet(rp)
        if "hmm_regime_enc" in reg.columns:
            hmm = reg["hmm_regime_enc"].reindex(df.index, fill_value=1).values.astype(np.int32)

    close = df["close"].values.astype(np.float64)
    high  = df["high"].values.astype(np.float64)
    low   = df["low"].values.astype(np.float64)
    atr   = df["atr_14_h1"].values.astype(np.float64)

    # ── LGBM predictions ──────────────────────────────────────────────────────
    X_tb = np.zeros((n, len(tb_feats)))
    for idx, c in enumerate(tb_feats):
        if c in df.columns:
            X_tb[:, idx] = df[c].ffill().fillna(0).values
    p_tb   = tb_model.predict_proba(X_tb).astype(np.float32)
    conf_tb= np.max(p_tb, axis=1)
    yp_tb  = np.argmax(p_tb, axis=1).astype(np.int32)
    for r, th in REGIME_THRESH.items():
        yp_tb[(hmm==r)&(yp_tb!=1)&(conf_tb<th)] = 1

    # ── Binary LSTM Meta: build all sequences, batch infer ────────────────────
    X_meta_raw = np.zeros((n, len(meta_feats)), dtype=np.float32)
    for idx, c in enumerate(meta_feats):
        if c in df.columns:
            X_meta_raw[:, idx] = df[c].ffill().fillna(0).values.astype(np.float32)

    # Build seq array: (n - SEQ_LEN, SEQ_LEN, n_feat_total)
    n_seq = n - SEQ_LEN
    if n_seq <= 0:
        logger.warning(f"[{sym}] too few bars ({n}), skip")
        continue

    # Append direction (LGBM predicted direction) as 16th raw feature BEFORE scaling
    # (scaler was trained on 16 features including direction)
    lgbm_dir = np.where(yp_tb==2, 1.0, np.where(yp_tb==0, -1.0, 0.0)).astype(np.float32)
    X_meta_full = np.concatenate([X_meta_raw, lgbm_dir[:, np.newaxis]], axis=1)  # (n, 16)

    # Vectorized sequence build (n_seq, SEQ_LEN, 16)
    X_seq_raw = np.lib.stride_tricks.as_strided(
        X_meta_full,
        shape=(n_seq, SEQ_LEN, n_feat_total),
        strides=(X_meta_full.strides[0], X_meta_full.strides[0], X_meta_full.strides[1])
    ).copy()

    # Scale: reshape to (n_seq*SEQ_LEN, 16), transform, reshape back
    orig_shape = X_seq_raw.shape
    X_2d = X_seq_raw.reshape(-1, n_feat_total)
    X_2d_sc = meta_scaler.transform(X_2d).astype(np.float32)
    X_seq_full = X_2d_sc.reshape(orig_shape)  # (n_seq, 32, 16)

    # Batch inference
    INFER_BATCH = 512
    p_win_all = np.zeros(n, dtype=np.float32)
    for start in range(0, n_seq, INFER_BATCH):
        end  = min(start + INFER_BATCH, n_seq)
        xb   = torch.FloatTensor(X_seq_full[start:end])
        with torch.no_grad():
            scores = meta_model(xb).squeeze(1).numpy()
        # score at position (SEQ_LEN + start) to (SEQ_LEN + end)
        p_win_all[SEQ_LEN + start: SEQ_LEN + end] = scores
    # First SEQ_LEN bars: no score available, set to base rate
    p_win_all[:SEQ_LEN] = 0.412  # base WIN rate

    # ── Guardian static features ───────────────────────────────────────────
    X_tbg = np.zeros((n, len(tbg_static)))
    for idx, c in enumerate(tbg_static):
        if c in df.columns:
            X_tbg[:, idx] = df[c].ffill().fillna(0).values

    # ── Simulate: tb+Guardian (baseline) ──────────────────────────────────
    upd(agg["tb_gdn"], simulate(yp_tb, close, high, low, atr, X_tbg, use_guardian=True))

    # ── Simulate: tb+BinaryMeta(thr)+Guardian ─────────────────────────────
    for thr in META_THRESHOLDS:
        key = f"tb_meta{int(thr*100)}_gdn"
        yp_filtered = yp_tb.copy()
        # Skip directional entries where LSTM p_win < threshold
        yp_filtered[(yp_filtered != 1) & (p_win_all < thr)] = 1
        upd(agg[key], simulate(yp_filtered, close, high, low, atr, X_tbg, use_guardian=True))

    logger.info(
        f"[{sym}] baseline={agg['tb_gdn']['trades']} "
        f"meta50={agg['tb_meta50_gdn']['trades']} "
        f"meta55={agg['tb_meta55_gdn']['trades']} "
        f"meta60={agg['tb_meta60_gdn']['trades']}"
    )


# ── Scorecard ──────────────────────────────────────────────────────────────────
def sc(a):
    t=a["trades"]; w=a["wins"]; p=a["pnl"]
    l=a["longs"]; lw=a["long_wins"]; sl=a["sl"]; gd=a["gdn"]; bh=a["bars"]
    sw = w - lw; sh = t - l
    pnls = None  # compute PF from agg if needed
    return dict(
        trades=t, wr=w/max(t,1)*100,
        long_wr=lw/max(l,1)*100, short_wr=sw/max(sh,1)*100,
        long_pct=l/max(t,1)*100,
        sl_rate=sl/max(t,1)*100, gdn_rate=gd/max(t,1)*100,
        avg_hold=float(np.mean(bh)) if bh else 0,
        pnl=p, ppm=p/5, ppt=p/max(t,1),
    )

s = {k: sc(agg[k]) for k in variant_keys}

LABELS = {
    "tb_gdn"       : "tb+Guardian",
    "tb_meta50_gdn": "tb+Meta(0.50)+Gdn",
    "tb_meta55_gdn": "tb+Meta(0.55)+Gdn",
    "tb_meta60_gdn": "tb+Meta(0.60)+Gdn",
}
W = 20

print(f"\n{'='*88}")
print(f"  SCORECARD — Binary LSTM Meta + Guardian | Nov 2025–Apr 2026 | 21 koin")
print(f"{'='*88}")
hdr = f"  {'Metrik':<22}"
for k in variant_keys:
    hdr += f" {LABELS[k]:>{W}}"
print(hdr)
print(f"  {'-'*84}")

rows = [
    ("Total Trades",    lambda x: f"{x['trades']:,}"),
    ("Trades/bulan",    lambda x: f"{x['trades']//5:,}"),
    ("Win Rate",        lambda x: f"{x['wr']:.1f}%"),
    ("  LONG WR",       lambda x: f"{x['long_wr']:.1f}% ({x['long_pct']:.0f}%)"),
    ("  SHORT WR",      lambda x: f"{x['short_wr']:.1f}%"),
    ("SL hit rate",     lambda x: f"{x['sl_rate']:.1f}%"),
    ("Guardian exits",  lambda x: f"{x['gdn_rate']:.1f}%"),
    ("Avg hold (bar)",  lambda x: f"{x['avg_hold']:.1f}"),
    ("Net PnL (5bln)",  lambda x: f"${x['pnl']:+.0f}"),
    ("PnL/bulan",       lambda x: f"${x['ppm']:+.0f}"),
    ("PnL/trade",       lambda x: f"${x['ppt']:+.3f}"),
]

for label, fn in rows:
    row = f"  {label:<22}"
    for k in variant_keys:
        row += f" {fn(s[k]):>{W}}"
    if "PnL (5bln)" in label:
        row += "  <--"
    print(row)

base = s["tb_gdn"]
print(f"\n  Delta vs tb+Guardian:")
for k in variant_keys[1:]:
    v = s[k]
    print(f"  {LABELS[k]:<22}: PnL {v['pnl']-base['pnl']:>+7.0f}  "
          f"WR {v['wr']-base['wr']:>+5.1f}pp  "
          f"Trades {v['trades']-base['trades']:>+5}")

# Save
out = {k: {**s[k], "label": LABELS[k]} for k in variant_keys}
out_path = MODEL_DIR / "runs/tb_lstm_binary_meta_v1/holdout_binary_meta_results.json"
with open(out_path, "w") as f:
    json.dump(out, f, indent=2, default=float)
print(f"\n  Saved: {out_path}")
print(f"{'='*88}")
