"""
Per-state threshold sweep pada OOF.
Untuk tiap HMM state, cari (long_thr, short_thr) yang maksimalkan net ATR.
Constraint: trade >= 60% baseline per state (jangan terlalu sedikit).

Metric proxy: net = wins*AVG_WIN - losses*AVG_LOSS
"""

import sys, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from itertools import product
from hmmlearn.hmm import GaussianHMM
from core.regime import _align_states
from config import TRAINING_COINS

PROC_DIR  = Path("data/training/processed")
OOF_PATH  = Path("models/runs/ic32_regime_v2/oof_predictions.parquet")
LABEL_DIR = Path("data/training/labeled")

AVG_WIN  = 0.40
AVG_LOSS = 0.31
SEP = "=" * 65

# Sweep grid
THR_GRID = [0.60, 0.62, 0.65, 0.67, 0.70, 0.72, 0.75, 0.77, 0.80, 0.82, 0.85]
MIN_TRADE_FRAC = 0.70   # trades >= 70% baseline (jangan terlalu sedikit)
MAX_TRADE_FRAC = 1.30   # trades <= 130% baseline (bukan sekadar perbanyak trade)
# Metric: PF (Profit Factor) = quality metric, bukan net ATR yang reward volume

# Current config thresholds (baseline)
CURRENT_THR = {
    "TRENDING_DOWN":    (0.75, 0.70),
    "RANGING_LOW_VOL":  (0.75, 0.70),
    "RANGING_HIGH_VOL": (0.75, 0.70),
    "TRENDING_UP":      (0.75, 0.80),
}

REGIME_NAMES_4 = ["TRENDING_DOWN","RANGING_LOW_VOL","RANGING_HIGH_VOL","TRENDING_UP"]

# ─── Load data ────────────────────────────────────────────────────────────────

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

# OOF
oof = pd.read_parquet(OOF_PATH)
oof = oof[oof["has_oof"]].copy()

coin_index = {}
for coin in TRAINING_COINS:
    fp = LABEL_DIR / f"{coin}_features_v3.parquet"
    if fp.exists():
        coin_index[coin] = pd.read_parquet(fp, columns=[]).index

# Fit HMM V2
print("Fit HMM V2...")
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
        m = GaussianHMM(n_components=4, covariance_type="diag",
                        n_iter=100, random_state=42, tol=1e-3, verbose=False)
        m.fit(X)
        sm = _align_states(m, 4)
        labels = pd.Series([sm[int(s)] for s in m.predict(X)], index=h4.index)
        h1 = labels.reindex(coin_index[coin]).ffill().dropna()
        state_records.append(h1.rename("state"))
    except Exception:
        continue

all_states = pd.concat(state_records)
oof_s = oof.join(all_states, how="left")
oof_s["hmm"] = oof_s["state"].fillna("RANGING_LOW_VOL")
print(f"OOF bars: {len(oof_s):,}  states: {oof_s['hmm'].value_counts().to_dict()}\n")


# ─── Scorecard helper ─────────────────────────────────────────────────────────

def score_state(df_state, long_thr, short_thr):
    """Scorecard untuk satu state: n, PF, net_ev, WR_L, WR_S."""
    long_mask  = df_state["p2"] >= long_thr
    short_mask = df_state["p0"] >= short_thr
    # Resolve conflict: argmax
    both = long_mask & short_mask
    longs  = df_state[long_mask  & ~(both & (df_state["p2"] <= df_state["p0"]))]
    shorts = df_state[short_mask & ~(both & (df_state["p0"] <  df_state["p2"]))]

    def ev(sub, pred_label):
        if len(sub) == 0: return 0, 0.0, 0.0
        correct = (sub["swing_label"] == pred_label).sum()
        wrong   = len(sub) - correct
        wr      = correct / len(sub)
        net     = correct * AVG_WIN - wrong * AVG_LOSS
        return len(sub), net, wr

    n_l, ev_l, wr_l = ev(longs,  2)
    n_s, ev_s, wr_s = ev(shorts, 0)
    n   = n_l + n_s
    net = ev_l + ev_s
    wins   = wr_l * n_l + wr_s * n_s
    losses = n - wins
    pf = (wins * AVG_WIN) / (losses * AVG_LOSS) if losses > 0 else 99.0
    return n, net, wr_l, wr_s, pf


# ─── Sweep per state ──────────────────────────────────────────────────────────

print(SEP)
print("PER-STATE THRESHOLD SWEEP")
print(SEP)

best_per_state = {}

for state in REGIME_NAMES_4:
    df_st = oof_s[oof_s["hmm"] == state].copy()
    if len(df_st) < 100:
        print(f"\n[{state}] — tidak cukup data ({len(df_st)}), skip")
        continue

    # Baseline
    cur_l, cur_s = CURRENT_THR[state]
    n_base, ev_base, wr_l_base, wr_s_base, pf_base = score_state(df_st, cur_l, cur_s)
    min_trades = int(n_base * MIN_TRADE_FRAC)
    max_trades = int(n_base * MAX_TRADE_FRAC)

    best = {"long_thr": cur_l, "short_thr": cur_s,
            "n": n_base, "ev": ev_base, "pf": pf_base, "wr_l": wr_l_base, "wr_s": wr_s_base}

    for lt, st in product(THR_GRID, THR_GRID):
        n, ev, wr_l, wr_s, pf = score_state(df_st, lt, st)
        if n < min_trades or n > max_trades:
            continue
        if pf > best["pf"]:   # optimize PF, bukan net ATR
            best = {"long_thr": lt, "short_thr": st,
                    "n": n, "ev": ev, "pf": pf, "wr_l": wr_l, "wr_s": wr_s}

    best_per_state[state] = best

    improved = best["pf"] > pf_base
    tag = " [IMPROVED]" if improved else " [no change]"

    print(f"\n[{state}]{tag}")
    print(f"  Baseline : long={cur_l}  short={cur_s}  "
          f"n={n_base:,}  PF={pf_base:.3f}  WR_L={wr_l_base*100:.1f}%  WR_S={wr_s_base*100:.1f}%")
    print(f"  Best     : long={best['long_thr']}  short={best['short_thr']}  "
          f"n={best['n']:,}  PF={best['pf']:.3f}  WR_L={best['wr_l']*100:.1f}%  WR_S={best['wr_s']*100:.1f}%")
    if improved:
        delta_pf = best["pf"] - pf_base
        delta_ev = best["ev"] - ev_base
        print(f"  Delta    : PF={delta_pf:+.3f}  ev={delta_ev:+,.0f} ATR  trades={best['n']-n_base:+,}")


# ─── Full pipeline dengan best thresholds ─────────────────────────────────────

print()
print(SEP)
print("SIMULASI FULL PIPELINE DENGAN BEST THRESHOLDS PER STATE")
print(SEP)

# Baseline: semua pakai current thr
total_base_ev = 0
total_base_n  = 0
for state in REGIME_NAMES_4:
    df_st = oof_s[oof_s["hmm"] == state]
    if len(df_st) < 10: continue
    cur_l, cur_s = CURRENT_THR[state]
    n, ev, _, _, pf = score_state(df_st, cur_l, cur_s)
    total_base_ev += ev
    total_base_n  += n

# Best: pakai optimized thr per state
total_best_ev = 0
total_best_n  = 0
for state in REGIME_NAMES_4:
    df_st = oof_s[oof_s["hmm"] == state]
    if len(df_st) < 10 or state not in best_per_state: continue
    b = best_per_state[state]
    n, ev, _, _, pf = score_state(df_st, b["long_thr"], b["short_thr"])
    total_best_ev += ev
    total_best_n  += n

wins_base = sum(score_state(oof_s[oof_s["hmm"]==s], *CURRENT_THR[s])[2] * score_state(oof_s[oof_s["hmm"]==s], *CURRENT_THR[s])[0]
               for s in REGIME_NAMES_4 if len(oof_s[oof_s["hmm"]==s]) >= 10)
losses_base = total_base_n - wins_base
pf_base_total = (wins_base * AVG_WIN) / (losses_base * AVG_LOSS) if losses_base > 0 else 0

wins_best = 0
for state in REGIME_NAMES_4:
    df_st = oof_s[oof_s["hmm"] == state]
    if len(df_st) < 10 or state not in best_per_state: continue
    b = best_per_state[state]
    n, ev, wrl, wrs, pf = score_state(df_st, b["long_thr"], b["short_thr"])
    wins_best += (wrl + wrs) / 2 * n  # approximate
losses_best = total_best_n - wins_best
pf_best_total = (wins_best * AVG_WIN) / (losses_best * AVG_LOSS) if losses_best > 0 else 0

print(f"\n  Baseline  : n={total_base_n:,}  net_ev={total_base_ev:,.0f} ATR")
print(f"  Optimized : n={total_best_n:,}  net_ev={total_best_ev:,.0f} ATR")
delta_ev = total_best_ev - total_base_ev
delta_n  = total_best_n  - total_base_n
print(f"  Delta     : ev={delta_ev:+,.0f} ATR  trades={delta_n:+,}  ({delta_n/total_base_n*100:+.1f}%)")

print()
print(SEP)
print("REKOMENDASI THRESHOLD PER STATE")
print(SEP)
for state in REGIME_NAMES_4:
    if state not in best_per_state: continue
    b  = best_per_state[state]
    cl, cs = CURRENT_THR[state]
    same = b["long_thr"] == cl and b["short_thr"] == cs
    tag = " (tidak berubah)" if same else " << UPDATE"
    print(f"  {state:22}: long={b['long_thr']}  short={b['short_thr']}{tag}")
    if not same:
        print(f"    sebelumnya: long={cl}  short={cs}")
