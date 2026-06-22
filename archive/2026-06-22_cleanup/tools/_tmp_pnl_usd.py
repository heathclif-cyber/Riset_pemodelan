"""
PnL aktual OOF dalam USD — menggunakan harga nyata (close, h4_swing_high, h4_swing_low).
Join dilakukan per-coin untuk hindari masalah duplicate timestamp.
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

NOTIONAL = 100.0
SEP = "=" * 68

# ── H4 + HMM V2 ───────────────────────────────────────────────────────────────

def load_h4(coin):
    fp = PROC_DIR / f"{coin}_clean.parquet"
    if not fp.exists(): return None
    df = pd.read_parquet(fp).rename(columns={
        "1h_open":"open","1h_high":"high","1h_low":"low",
        "1h_close":"close","1h_volume":"volume"})
    if "close" not in df.columns: return None
    return df[["open","high","low","close","volume"]].resample("4h").agg(
        {"open":"first","high":"max","low":"min","close":"last","volume":"sum"}).dropna()

print("Loading...")
h4_cache = {c: h for c in TRAINING_COINS if (h := load_h4(c)) is not None and len(h) >= 200}
btc_h4   = h4_cache["BTCUSDT"]
btc_ret  = btc_h4["close"].pct_change().fillna(0)
btc_mom  = (btc_h4["close"] / btc_h4["close"].rolling(48, min_periods=8).mean() - 1).fillna(0)

def get_hmm_states(coin):
    h4 = h4_cache.get(coin)
    if h4 is None: return None
    ret = h4["close"].pct_change().fillna(0)
    vol = ret.rolling(24, min_periods=4).std().fillna(ret.std())
    mom = (h4["close"] / h4["close"].rolling(48, min_periods=8).mean() - 1).fillna(0)
    vr  = np.log((h4["volume"]/h4["volume"].rolling(48,min_periods=8).mean()).clip(0.01,20)).fillna(0)
    b_r = btc_ret.reindex(h4.index).ffill().fillna(0)
    b_m = btc_mom.reindex(h4.index).ffill().fillna(0)
    X   = np.nan_to_num(np.column_stack([ret,vol,mom,vr,b_r,b_m]), nan=0.0, posinf=0.0, neginf=0.0)
    m = GaussianHMM(n_components=4, covariance_type="diag", n_iter=100, random_state=42, tol=1e-3, verbose=False)
    m.fit(X)
    sm = _align_states(m, 4)
    return pd.Series([sm[int(s)] for s in m.predict(X)], index=h4.index, name="hmm")

# ── Load OOF ──────────────────────────────────────────────────────────────────

oof_all = pd.read_parquet(OOF_PATH)
oof_all = oof_all[oof_all["has_oof"]].copy()

# ── Per-coin join + trade simulation ─────────────────────────────────────────

def simulate_coin(coin, thr_map):
    """Return DataFrame of trades for one coin with actual PnL%."""
    oof_coin = oof_all[oof_all["coin"] == coin].copy()
    if len(oof_coin) == 0: return None

    fp = LABEL_DIR / f"{coin}_features_v3.parquet"
    if not fp.exists(): return None
    prices = pd.read_parquet(fp, columns=["close","h4_swing_high","h4_swing_low"])

    df = oof_coin.join(prices, how="left")

    try:
        h4_states = get_hmm_states(coin)
        h1_states = h4_states.reindex(df.index).ffill()
        df["hmm"] = h1_states.fillna("RANGING_LOW_VOL")
    except Exception:
        df["hmm"] = "RANGING_LOW_VOL"

    trades_parts = []
    for state, (lt, st_) in thr_map.items():
        sub = df[df["hmm"] == state].copy()
        if len(sub) == 0: continue
        lm = sub["p2"] >= lt
        sm = sub["p0"] >= st_
        both = lm & sm
        longs  = sub[lm & ~(both & (sub["p2"] <= sub["p0"]))].copy()
        shorts = sub[sm & ~(both & (sub["p0"] <  sub["p2"]))].copy()
        longs["dir"]  = "LONG";  longs["hmm_state"]  = state
        shorts["dir"] = "SHORT"; shorts["hmm_state"] = state
        trades_parts.extend([longs, shorts])

    if not trades_parts: return None
    t = pd.concat(trades_parts)

    def pnl_pct(row):
        c, sh, sl = row["close"], row["h4_swing_high"], row["h4_swing_low"]
        if pd.isna(c) or pd.isna(sh) or pd.isna(sl) or c <= 0: return np.nan
        label = int(row["swing_label"])
        if row["dir"] == "LONG":
            return (sh - c)/c if label == 2 else (sl - c)/c
        else:
            return (c - sl)/c if label == 0 else (c - sh)/c

    t["pnl_pct"] = t.apply(pnl_pct, axis=1)
    t["pnl_usd"] = t["pnl_pct"] * NOTIONAL
    t["coin"]    = coin
    return t.dropna(subset=["pnl_pct"])

print("Simulating per-coin (current thresholds)...")
old_parts = [r for c in TRAINING_COINS if (r := simulate_coin(c, CURRENT_THR)) is not None]
print("Simulating per-coin (new thresholds)...")
new_parts = [r for c in TRAINING_COINS if (r := simulate_coin(c, NEW_THR)) is not None]

old_trades = pd.concat(old_parts) if old_parts else pd.DataFrame()
new_trades = pd.concat(new_parts) if new_parts else pd.DataFrame()
print(f"Trades — lama: {len(old_trades):,}  baru: {len(new_trades):,}\n")

# ── Scorecard ─────────────────────────────────────────────────────────────────

def sc(trades):
    if len(trades) == 0: return {}
    w = trades["pnl_usd"] > 0
    l = ~w
    n    = len(trades)
    wr   = w.mean()
    gw   = trades.loc[w, "pnl_usd"].sum()
    gl   = trades.loc[l, "pnl_usd"].abs().sum()
    pf   = gw / gl if gl > 0 else 99.0
    net  = trades["pnl_usd"].sum()
    aw   = trades.loc[w, "pnl_usd"].mean() if w.any() else 0
    al   = trades.loc[l, "pnl_usd"].mean() if l.any() else 0
    lt = trades[trades["dir"]=="LONG"];  st = trades[trades["dir"]=="SHORT"]
    wrl  = (lt["pnl_usd"]>0).mean() if len(lt) else 0
    wrs  = (st["pnl_usd"]>0).mean() if len(st) else 0
    gwl  = lt[lt["pnl_usd"]>0]["pnl_usd"].sum()
    gll  = lt[lt["pnl_usd"]<=0]["pnl_usd"].abs().sum()
    gws  = st[st["pnl_usd"]>0]["pnl_usd"].sum()
    gls  = st[st["pnl_usd"]<=0]["pnl_usd"].abs().sum()
    pfl  = gwl/gll if gll > 0 else 99.0
    pfs  = gws/gls if gls > 0 else 99.0
    return dict(n=n, wr=wr, pf=pf, net=net, aw=aw, al=al,
                nl=len(lt), ns=len(st), wrl=wrl, wrs=wrs, pfl=pfl, pfs=pfs)

o  = sc(old_trades)
nw = sc(new_trades)

# ── Print scorecard ───────────────────────────────────────────────────────────

print(SEP)
print(f"  OOF PnL AKTUAL — NOTIONAL ${NOTIONAL:.0f}/trade")
print(SEP)
print(f"  {'':20} {'LAMA':>18}  {'BARU':>18}  {'Delta':>10}")
print(f"  {'-'*68}")

def row(name, ov, nv, delta):
    print(f"  {name:20} {ov:>18}  {nv:>18}  {delta:>10}")

row("Trades",     f"{o['n']:,}",        f"{nw['n']:,}",       f"{nw['n']-o['n']:+,}")
row("  LONG",     f"{o['nl']:,}",       f"{nw['nl']:,}",      f"{nw['nl']-o['nl']:+,}")
row("  SHORT",    f"{o['ns']:,}",       f"{nw['ns']:,}",      f"{nw['ns']-o['ns']:+,}")
row("WR",         f"{o['wr']*100:.2f}%",f"{nw['wr']*100:.2f}%",f"{(nw['wr']-o['wr'])*100:+.2f}pp")
row("  WR LONG",  f"{o['wrl']*100:.2f}%",f"{nw['wrl']*100:.2f}%",f"{(nw['wrl']-o['wrl'])*100:+.2f}pp")
row("  WR SHORT", f"{o['wrs']*100:.2f}%",f"{nw['wrs']*100:.2f}%",f"{(nw['wrs']-o['wrs'])*100:+.2f}pp")
row("PF",         f"{o['pf']:.3f}",     f"{nw['pf']:.3f}",    f"{nw['pf']-o['pf']:+.3f}")
row("  PF LONG",  f"{o['pfl']:.3f}",    f"{nw['pfl']:.3f}",   f"{nw['pfl']-o['pfl']:+.3f}")
row("  PF SHORT", f"{o['pfs']:.3f}",    f"{nw['pfs']:.3f}",   f"{nw['pfs']-o['pfs']:+.3f}")
row("Avg Win",    f"${o['aw']:.2f}",    f"${nw['aw']:.2f}",   f"${nw['aw']-o['aw']:+.2f}")
row("Avg Loss",   f"${o['al']:.2f}",    f"${nw['al']:.2f}",   f"${nw['al']-o['al']:+.2f}")
row("Net PnL",    f"${o['net']:,.0f}",  f"${nw['net']:,.0f}", f"${nw['net']-o['net']:+,.0f}")

ev_o = o['net']/o['n']  if o['n']  else 0
ev_n = nw['net']/nw['n'] if nw['n'] else 0
row("EV/trade",   f"${ev_o:.3f}",       f"${ev_n:.3f}",       f"${ev_n-ev_o:+.3f}")

# ── Monthly PnL ───────────────────────────────────────────────────────────────

def monthly(trades):
    ts = pd.to_datetime(trades.index.get_level_values("timestamp") if "timestamp" in trades.index.names else trades.index)
    return trades.groupby(pd.PeriodIndex(ts, freq="M"))["pnl_usd"].sum()

om = monthly(old_trades)
nm = monthly(new_trades)
mdf = pd.DataFrame({"Lama": om, "Baru": nm}).fillna(0)
mdf = mdf[mdf.index >= "2021-01"]

print()
print(SEP)
print(f"  MONTHLY PnL — ${NOTIONAL:.0f}/trade (dari Jan 2021)")
print(SEP)
print(f"  {'Bulan':>10} {'Lama':>12} {'Baru':>12} {'Delta':>10}")
print(f"  {'-'*50}")

yearly = {}
for period, row_ in mdf.iterrows():
    yr = period.year
    if yr not in yearly: yearly[yr] = {"l":0,"b":0}
    yearly[yr]["l"] += row_["Lama"]
    yearly[yr]["b"] += row_["Baru"]
    d = row_["Baru"] - row_["Lama"]
    print(f"  {str(period):>10} ${row_['Lama']:>10,.0f} ${row_['Baru']:>10,.0f} ${d:>+9,.0f}")
    if period.month == 12 or period == mdf.index[-1]:
        y = yearly[yr]
        print(f"  {'['+str(yr)+']':>10} ${y['l']:>10,.0f} ${y['b']:>10,.0f} ${y['b']-y['l']:>+9,.0f}")
        print()

mean_o = om[om.index >= "2021-01"].mean()
mean_n = nm[nm.index >= "2021-01"].mean()
print(SEP)
print(f"  Avg/bulan (OOF)  ${mean_o:>10,.0f}   ${mean_n:>10,.0f}  (notional ${NOTIONAL:.0f}/trade)")
print()
print(f"  Scaling proportional:")
print(f"    $500/trade  -> lama ${mean_o*5:>8,.0f}/bln  baru ${mean_n*5:>8,.0f}/bln")
print(f"    $1000/trade -> lama ${mean_o*10:>8,.0f}/bln  baru ${mean_n*10:>8,.0f}/bln")
print(f"    $2000/trade -> lama ${mean_o*20:>8,.0f}/bln  baru ${mean_n*20:>8,.0f}/bln")
