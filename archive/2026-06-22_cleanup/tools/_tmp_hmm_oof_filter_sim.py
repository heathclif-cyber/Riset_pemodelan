"""
Simulasi dampak direction filter HMM V2 pada OOF.

Baseline : thr_long=0.75, thr_short=0.70, tanpa filter arah
Filter   : + blok LONG jika TRENDING_DOWN, blok SHORT jika TRENDING_UP

Metric: N trades, WR, PF (estimasi), net PnL ATR
PnL proxy dari holdout v2: avg_win=0.40 ATR, avg_loss=0.31 ATR
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

THR_LONG  = 0.75
THR_SHORT = 0.70

# Dari holdout v2
AVG_WIN_ATR  =  0.40
AVG_LOSS_ATR =  0.31

SEP = "=" * 65


# ─── Load H4 & cross-asset features ──────────────────────────────────────────

def load_h4(coin):
    fp = PROC_DIR / f"{coin}_clean.parquet"
    if not fp.exists(): return None
    df = pd.read_parquet(fp).rename(columns={
        "1h_open":"open","1h_high":"high","1h_low":"low",
        "1h_close":"close","1h_volume":"volume"
    })
    if "close" not in df.columns: return None
    return df[["open","high","low","close","volume"]].resample("4h").agg(
        {"open":"first","high":"max","low":"min","close":"last","volume":"sum"}
    ).dropna()

print("Loading data...")
h4_cache = {c: h for c in TRAINING_COINS if (h := load_h4(c)) is not None and len(h) >= 200}
btc_h4   = h4_cache["BTCUSDT"]
btc_ret  = btc_h4["close"].pct_change().fillna(0)
btc_mom  = (btc_h4["close"] / btc_h4["close"].rolling(48, min_periods=8).mean() - 1).fillna(0)


# ─── Fit HMM V2 per koin ─────────────────────────────────────────────────────

def build_X_v2(coin):
    h4 = h4_cache.get(coin)
    if h4 is None: return None
    ret = h4["close"].pct_change().fillna(0)
    vol = ret.rolling(24, min_periods=4).std().fillna(ret.std())
    mom = (h4["close"] / h4["close"].rolling(48, min_periods=8).mean() - 1).fillna(0)
    vr  = np.log((h4["volume"]/h4["volume"].rolling(48,min_periods=8).mean()).clip(0.01,20)).fillna(0)
    b_r = btc_ret.reindex(h4.index).ffill().fillna(0)
    b_m = btc_mom.reindex(h4.index).ffill().fillna(0)
    X = np.column_stack([ret.values, vol.values, mom.values, vr.values, b_r.values, b_m.values])
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

print("Fit HMM V2 per koin...")
coin_index = {}
state_records = []

for coin in TRAINING_COINS:
    fp = LABEL_DIR / f"{coin}_features_v3.parquet"
    if fp.exists():
        coin_index[coin] = pd.read_parquet(fp, columns=[]).index

    if coin not in h4_cache: continue
    X = build_X_v2(coin)
    if X is None or len(X) < 200: continue

    try:
        m = GaussianHMM(n_components=4, covariance_type="diag",
                        n_iter=100, random_state=42, tol=1e-3, verbose=False)
        m.fit(X)
        sm = _align_states(m, 4)
        labels = pd.Series([sm[int(s)] for s in m.predict(X)], index=h4_cache[coin].index)
        if coin in coin_index:
            h1 = labels.reindex(coin_index[coin]).ffill().dropna()
            state_records.append(pd.DataFrame({"state": h1}))
    except Exception:
        continue

all_states = pd.concat(state_records)["state"]
print(f"  State labels: {len(all_states):,} H1 bars\n")


# ─── Load OOF, apply threshold ────────────────────────────────────────────────

oof = pd.read_parquet(OOF_PATH)
oof = oof[oof["has_oof"]].copy()

# Apply production thresholds
oof["signal"] = "FLAT"
oof.loc[oof["p2"] >= THR_LONG,  "signal"] = "LONG"
oof.loc[oof["p0"] >= THR_SHORT, "signal"] = "SHORT"
# Jika keduanya di atas threshold, ambil argmax proba
both = (oof["p2"] >= THR_LONG) & (oof["p0"] >= THR_SHORT)
oof.loc[both, "signal"] = oof.loc[both, ["p0","p1","p2"]].values.argmax(axis=1)
oof.loc[both & (oof["signal"] == 2), "signal"] = "LONG"
oof.loc[both & (oof["signal"] == 0), "signal"] = "SHORT"

trades = oof[oof["signal"] != "FLAT"].copy()
print(f"OOF trades (setelah threshold {THR_LONG}/{THR_SHORT}): {len(trades):,}")

# Join HMM state
trades = trades.join(all_states.rename("hmm_state"), how="left")
no_state = trades["hmm_state"].isna().sum()
print(f"Trades tanpa HMM state: {no_state:,} ({no_state/len(trades)*100:.1f}%) — akan dianggap RANGING\n")
trades["hmm_state"] = trades["hmm_state"].fillna("RANGING_LOW_VOL")


# ─── Helper: scorecard dari trades ───────────────────────────────────────────

def scorecard(df, label):
    if len(df) == 0:
        print(f"  {label}: 0 trades")
        return

    # Cek apakah prediksi benar vs swing_label
    # signal: LONG=2, SHORT=0; swing_label: LONG=2, SHORT=0, FLAT=1
    sig_num = df["signal"].map({"LONG": 2, "SHORT": 0})
    correct = (sig_num == df["swing_label"]).astype(int)

    n      = len(df)
    wr     = correct.mean()
    wins   = correct.sum()
    losses = n - wins
    gross_profit = wins  * AVG_WIN_ATR
    gross_loss   = losses * AVG_LOSS_ATR
    pf     = gross_profit / gross_loss if gross_loss > 0 else 99.0
    net    = gross_profit - gross_loss

    n_long  = (df["signal"] == "LONG").sum()
    n_short = (df["signal"] == "SHORT").sum()

    print(f"\n  [{label}]")
    print(f"  Trades: {n:,}  (L:{n_long:,}  S:{n_short:,})")
    print(f"  WR    : {wr*100:.2f}%")
    print(f"  PF    : {pf:.3f}")
    print(f"  Net   : {net:.1f} ATR  (proxy, avg win={AVG_WIN_ATR} avg loss={AVG_LOSS_ATR})")
    return {"n": n, "wr": wr, "pf": pf, "net": net, "n_long": n_long, "n_short": n_short}


# ─── Baseline: tanpa filter ────────────────────────────────────────────────────

print(SEP)
print("BASELINE — tanpa direction filter")
print(SEP)
base = scorecard(trades, "No Filter")

# Per direction breakdown
for d in ["LONG", "SHORT"]:
    sub = trades[trades["signal"] == d]
    sig_num = sub["signal"].map({"LONG":2,"SHORT":0})
    wr = (sig_num == sub["swing_label"]).mean()
    print(f"    {d}: {len(sub):,} trades  WR={wr*100:.1f}%")


# ─── Filter A: blok counter-trend di extreme states ──────────────────────────

print()
print(SEP)
print("FILTER A — blok LONG jika TRENDING_DOWN, blok SHORT jika TRENDING_UP")
print(SEP)

mask_keep = ~(
    ((trades["signal"] == "LONG")  & (trades["hmm_state"] == "TRENDING_DOWN")) |
    ((trades["signal"] == "SHORT") & (trades["hmm_state"] == "TRENDING_UP"))
)
filtered_a = trades[mask_keep].copy()
n_blocked_a = len(trades) - len(filtered_a)

print(f"  Trade diblok: {n_blocked_a:,} ({n_blocked_a/len(trades)*100:.1f}%)")
filt_a = scorecard(filtered_a, "Filter A")

for d in ["LONG", "SHORT"]:
    sub = filtered_a[filtered_a["signal"] == d]
    sig_num = sub["signal"].map({"LONG":2,"SHORT":0})
    wr = (sig_num == sub["swing_label"]).mean()
    print(f"    {d}: {len(sub):,} trades  WR={wr*100:.1f}%")


# ─── Filter B: hanya trading di arah aligned dengan state ────────────────────

print()
print(SEP)
print("FILTER B — hanya trade jika arah ALIGNED dengan state")
print("  (LONG di TRENDING_UP, SHORT di TRENDING_DOWN, semua di RANGING)")
print(SEP)

mask_b = (
    ((trades["signal"] == "LONG")  & (trades["hmm_state"] == "TRENDING_UP"))    |
    ((trades["signal"] == "SHORT") & (trades["hmm_state"] == "TRENDING_DOWN"))  |
    (trades["hmm_state"].isin(["RANGING_LOW_VOL", "RANGING_HIGH_VOL"]))
)
filtered_b = trades[mask_b].copy()
n_blocked_b = len(trades) - len(filtered_b)

print(f"  Trade diblok: {n_blocked_b:,} ({n_blocked_b/len(trades)*100:.1f}%)")
filt_b = scorecard(filtered_b, "Filter B")

for d in ["LONG", "SHORT"]:
    sub = filtered_b[filtered_b["signal"] == d]
    sig_num = sub["signal"].map({"LONG":2,"SHORT":0})
    wr = (sig_num == sub["swing_label"]).mean()
    print(f"    {d}: {len(sub):,} trades  WR={wr*100:.1f}%")


# ─── Ringkasan perbandingan ────────────────────────────────────────────────────

print()
print(SEP)
print("PERBANDINGAN RINGKAS")
print(SEP)
print(f"  {'':30} {'Trades':>8} {'WR':>8} {'PF':>8} {'Net ATR':>10}")
print(f"  {'-'*30} {'-'*8} {'-'*8} {'-'*8} {'-'*10}")

for label, res in [("Baseline (no filter)", base),
                   ("Filter A (block counter)", filt_a),
                   ("Filter B (aligned only)", filt_b)]:
    if res:
        print(f"  {label:30} {res['n']:>8,} {res['wr']*100:>7.2f}% {res['pf']:>8.3f} {res['net']:>10.1f}")

print()
if filt_a and base:
    wr_delta  = (filt_a["wr"] - base["wr"]) * 100
    pf_delta  = filt_a["pf"]  - base["pf"]
    net_delta = filt_a["net"] - base["net"]
    trade_pct = filt_a["n"] / base["n"] * 100
    print(f"  Filter A vs Baseline:")
    print(f"    WR  : {wr_delta:+.2f}%")
    print(f"    PF  : {pf_delta:+.3f}")
    print(f"    Net : {net_delta:+.1f} ATR")
    print(f"    Trade retained: {trade_pct:.1f}%")

if filt_b and base:
    wr_delta  = (filt_b["wr"] - base["wr"]) * 100
    pf_delta  = filt_b["pf"]  - base["pf"]
    net_delta = filt_b["net"] - base["net"]
    trade_pct = filt_b["n"] / base["n"] * 100
    print(f"\n  Filter B vs Baseline:")
    print(f"    WR  : {wr_delta:+.2f}%")
    print(f"    PF  : {pf_delta:+.3f}")
    print(f"    Net : {net_delta:+.1f} ATR")
    print(f"    Trade retained: {trade_pct:.1f}%")
