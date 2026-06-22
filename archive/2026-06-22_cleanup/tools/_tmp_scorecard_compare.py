"""
Scorecard perbandingan threshold lama vs baru di OOF.
Tampilkan: Trades, WR, PF, Net ATR — overall + per direction.
"""

import sys, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from core.regime import _align_states
from config import TRAINING_COINS

PROC_DIR  = Path("data/training/processed")
OOF_PATH  = Path("models/runs/ic32_regime_v2/oof_predictions.parquet")
LABEL_DIR = Path("data/training/labeled")
AVG_WIN   = 0.40
AVG_LOSS  = 0.31

CURRENT_THR = {
    "TRENDING_DOWN":    (0.75, 0.70),
    "RANGING_LOW_VOL":  (0.75, 0.70),
    "RANGING_HIGH_VOL": (0.75, 0.70),
    "TRENDING_UP":      (0.75, 0.80),
}
NEW_THR = {
    "TRENDING_DOWN":    (0.85, 0.70),
    "RANGING_LOW_VOL":  (0.77, 0.72),
    "RANGING_HIGH_VOL": (0.72, 0.77),
    "TRENDING_UP":      (0.77, 0.82),
}
REGIME_NAMES_4 = ["TRENDING_DOWN","RANGING_LOW_VOL","RANGING_HIGH_VOL","TRENDING_UP"]

# ── Load ──────────────────────────────────────────────────────────────────────

def load_h4(coin):
    fp = PROC_DIR / f"{coin}_clean.parquet"
    if not fp.exists(): return None
    df = pd.read_parquet(fp).rename(columns={
        "1h_open":"open","1h_high":"high","1h_low":"low",
        "1h_close":"close","1h_volume":"volume"})
    if "close" not in df.columns: return None
    return df[["open","high","low","close","volume"]].resample("4h").agg(
        {"open":"first","high":"max","low":"min","close":"last","volume":"sum"}).dropna()

h4_cache = {c: h for c in TRAINING_COINS if (h := load_h4(c)) is not None and len(h) >= 200}
btc_h4   = h4_cache["BTCUSDT"]
btc_ret  = btc_h4["close"].pct_change().fillna(0)
btc_mom  = (btc_h4["close"] / btc_h4["close"].rolling(48, min_periods=8).mean() - 1).fillna(0)

coin_index = {}
for coin in TRAINING_COINS:
    fp = LABEL_DIR / f"{coin}_features_v3.parquet"
    if fp.exists(): coin_index[coin] = pd.read_parquet(fp, columns=[]).index

state_records = []
for coin in TRAINING_COINS:
    if coin not in h4_cache or coin not in coin_index: continue
    h4  = h4_cache[coin]
    ret = h4["close"].pct_change().fillna(0)
    vol = ret.rolling(24, min_periods=4).std().fillna(ret.std())
    mom = (h4["close"] / h4["close"].rolling(48, min_periods=8).mean() - 1).fillna(0)
    vr  = np.log((h4["volume"]/h4["volume"].rolling(48,min_periods=8).mean()).clip(0.01,20)).fillna(0)
    b_r = btc_ret.reindex(h4.index).ffill().fillna(0)
    b_m = btc_mom.reindex(h4.index).ffill().fillna(0)
    X   = np.nan_to_num(np.column_stack([ret,vol,mom,vr,b_r,b_m]), nan=0.0, posinf=0.0, neginf=0.0)
    try:
        m = GaussianHMM(n_components=4, covariance_type="diag", n_iter=100, random_state=42, tol=1e-3, verbose=False)
        m.fit(X)
        sm = _align_states(m, 4)
        labels = pd.Series([sm[int(s)] for s in m.predict(X)], index=h4.index)
        h1 = labels.reindex(coin_index[coin]).ffill().dropna()
        state_records.append(h1.rename("state"))
    except Exception: continue

all_states = pd.concat(state_records)
oof = pd.read_parquet(OOF_PATH)
oof = oof[oof["has_oof"]].copy()
oof_s = oof.join(all_states, how="left")
oof_s["hmm"] = oof_s["state"].fillna("RANGING_LOW_VOL")

# ── Scorecard helper ──────────────────────────────────────────────────────────

def get_trades(df_state, long_thr, short_thr):
    lm = df_state["p2"] >= long_thr
    sm = df_state["p0"] >= short_thr
    both = lm & sm
    longs  = df_state[lm & ~(both & (df_state["p2"] <= df_state["p0"]))]
    shorts = df_state[sm & ~(both & (df_state["p0"] <  df_state["p2"]))]
    longs  = longs.copy();  longs["dir"]  = "LONG"
    shorts = shorts.copy(); shorts["dir"] = "SHORT"
    return pd.concat([longs, shorts])

def scorecard(trades):
    if len(trades) == 0:
        return dict(n=0, wr=0, pf=0, net=0, n_l=0, n_s=0, wr_l=0, wr_s=0, pf_l=0, pf_s=0)
    longs  = trades[trades["dir"] == "LONG"]
    shorts = trades[trades["dir"] == "SHORT"]

    def sub_sc(sub, label):
        if len(sub) == 0: return 0, 0.0, 0.0, 0.0
        w = (sub["swing_label"] == label).sum()
        l = len(sub) - w
        wr = w / len(sub)
        pf = (w*AVG_WIN) / (l*AVG_LOSS) if l > 0 else 99.0
        net = w*AVG_WIN - l*AVG_LOSS
        return len(sub), wr, pf, net

    nl, wrl, pfl, evl = sub_sc(longs, 2)
    ns, wrs, pfs, evs = sub_sc(shorts, 0)
    n = nl + ns
    w_tot = wrl*nl + wrs*ns
    l_tot = n - w_tot
    pf = (w_tot*AVG_WIN)/(l_tot*AVG_LOSS) if l_tot > 0 else 99.0
    wr = w_tot / n if n > 0 else 0
    net = evl + evs
    return dict(n=n, wr=wr, pf=pf, net=net, n_l=nl, n_s=ns,
                wr_l=wrl, wr_s=wrs, pf_l=pfl, pf_s=pfs)

# ── Kumpulkan trades ──────────────────────────────────────────────────────────

old_trades_all, new_trades_all = [], []
per_state = {}

for state in REGIME_NAMES_4:
    df_st = oof_s[oof_s["hmm"] == state]
    if len(df_st) < 50: continue
    ol, os_ = CURRENT_THR[state]
    nl, ns  = NEW_THR[state]
    old_t = get_trades(df_st, ol, os_)
    new_t = get_trades(df_st, nl, ns)
    old_trades_all.append(old_t)
    new_trades_all.append(new_t)
    per_state[state] = (scorecard(old_t), scorecard(new_t))

old_all = scorecard(pd.concat(old_trades_all))
new_all = scorecard(pd.concat(new_trades_all))

# ── Print ─────────────────────────────────────────────────────────────────────

SEP  = "=" * 72
SEP2 = "-" * 72

def fmt_delta(val, ref, pct=False, higher_better=True):
    d = val - ref
    sym = "+" if d >= 0 else ""
    tag = " ^" if (d > 0 and higher_better) or (d < 0 and not higher_better) else (" v" if d != 0 else "  ")
    if pct:
        return f"{val*100:6.2f}%  ({sym}{d*100:.2f}pp){tag}"
    return f"{val:7.3f}  ({sym}{d:.3f}){tag}"

def fmt_n(val, ref):
    d = val - ref
    sym = "+" if d >= 0 else ""
    return f"{val:>8,}  ({sym}{d:,})"

def fmt_net(val, ref):
    d = val - ref
    sym = "+" if d >= 0 else ""
    return f"{val:>9,.0f}  ({sym}{d:,.0f})"

print()
print(SEP)
print("  OOF SCORECARD: THRESHOLD LAMA vs BARU")
print(f"  Proxy: avg_win={AVG_WIN} ATR, avg_loss={AVG_LOSS} ATR")
print(SEP)

# ── Per state ──
for state in REGIME_NAMES_4:
    if state not in per_state: continue
    o, n = per_state[state]
    ol, os_ = CURRENT_THR[state]
    nl, ns  = NEW_THR[state]

    print()
    print(f"  [{state}]")
    print(f"  {'':18} {'LAMA':>20}  {'BARU':>28}")
    print(f"  {'':18}  long={ol}  short={os_}    long={nl}  short={ns}")
    print(f"  {SEP2}")
    print(f"  {'Trades':18} {fmt_n(n['n'], o['n'])}")
    print(f"  {'  LONG':18} {o['n_l']:>8,}                  {n['n_l']:>8,}  ({n['n_l']-o['n_l']:+,})")
    print(f"  {'  SHORT':18} {o['n_s']:>8,}                  {n['n_s']:>8,}  ({n['n_s']-o['n_s']:+,})")
    print(f"  {'WR overall':18} {o['wr']*100:>7.2f}%              {fmt_delta(n['wr'], o['wr'], pct=True)}")
    print(f"  {'  WR LONG':18} {o['wr_l']*100:>7.2f}%              {fmt_delta(n['wr_l'], o['wr_l'], pct=True)}")
    print(f"  {'  WR SHORT':18} {o['wr_s']*100:>7.2f}%              {fmt_delta(n['wr_s'], o['wr_s'], pct=True)}")
    print(f"  {'PF overall':18} {o['pf']:>7.3f}               {fmt_delta(n['pf'], o['pf'])}")
    print(f"  {'  PF LONG':18} {o['pf_l']:>7.3f}               {fmt_delta(n['pf_l'], o['pf_l'])}")
    print(f"  {'  PF SHORT':18} {o['pf_s']:>7.3f}               {fmt_delta(n['pf_s'], o['pf_s'])}")
    print(f"  {'Net ATR':18} {fmt_net(n['net'], o['net'])}")

# ── Total ──
print()
print(SEP)
print("  TOTAL OOF")
print(SEP)
print(f"  {'':18} {'LAMA':>20}  {'BARU':>28}")
print(f"  {SEP2}")
print(f"  {'Trades':18} {fmt_n(new_all['n'], old_all['n'])}")
print(f"  {'  LONG':18} {old_all['n_l']:>8,}                  {new_all['n_l']:>8,}  ({new_all['n_l']-old_all['n_l']:+,})")
print(f"  {'  SHORT':18} {old_all['n_s']:>8,}                  {new_all['n_s']:>8,}  ({new_all['n_s']-old_all['n_s']:+,})")
print(f"  {'WR overall':18} {old_all['wr']*100:>7.2f}%              {fmt_delta(new_all['wr'], old_all['wr'], pct=True)}")
print(f"  {'  WR LONG':18} {old_all['wr_l']*100:>7.2f}%              {fmt_delta(new_all['wr_l'], old_all['wr_l'], pct=True)}")
print(f"  {'  WR SHORT':18} {old_all['wr_s']*100:>7.2f}%              {fmt_delta(new_all['wr_s'], old_all['wr_s'], pct=True)}")
print(f"  {'PF overall':18} {old_all['pf']:>7.3f}               {fmt_delta(new_all['pf'], old_all['pf'])}")
print(f"  {'  PF LONG':18} {old_all['pf_l']:>7.3f}               {fmt_delta(new_all['pf_l'], old_all['pf_l'])}")
print(f"  {'  PF SHORT':18} {old_all['pf_s']:>7.3f}               {fmt_delta(new_all['pf_s'], old_all['pf_s'])}")
print(f"  {'Net ATR':18} {fmt_net(new_all['net'], old_all['net'])}")

ev_per_trade_old = old_all['net'] / old_all['n'] if old_all['n'] > 0 else 0
ev_per_trade_new = new_all['net'] / new_all['n'] if new_all['n'] > 0 else 0
print(f"  {'EV/trade (ATR)':18} {ev_per_trade_old:>7.4f}               {ev_per_trade_new:>7.4f}  ({ev_per_trade_new-ev_per_trade_old:+.4f}) ^")
print()
