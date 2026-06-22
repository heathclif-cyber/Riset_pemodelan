"""
Kalman Residual Spread — log(altcoin/BTC) mean-reversion signal.
Prinsip Simons: sinyal decorrelated dari trend-following LGBM.
Kalman estimasi fair-value trend spread; deviasi dari trend = sinyal.
"""
import json, sys, warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import joblib
from core.utils import ensure_utc_index, setup_logger
from config import *

logger = setup_logger("kalman_spread")

LM = {"SHORT": 0, "FLAT": 1, "LONG": 2}
SL_MULT = 1.5; MAX_HOLD = 36; MODAL = 10; LEV = 5
COST_RT = (FEE_PER_SIDE + SLIPPAGE_PER_SIDE) * 2
REGIME_THRESH = {0: 0.45, 1: 0.50, 2: 0.50, 3: 0.45}

tb_model = joblib.load(MODEL_DIR / "runs/tb_lgbm_widyawardhana_v3/lgbm.pkl")
with open(MODEL_DIR / "runs/tb_lgbm_widyawardhana_v3/tb_lgbm_widyawardhana_v3_features.json") as f:
    tb_feats = json.load(f)


class KalmanSpread:
    """
    Kalman Filter pada log(alt_price / btc_price).
    State: [fair_log_spread, drift_rate]
    Observasi: log(alt/btc) aktual tiap H4.
    """
    def __init__(self, q_level=1e-5, q_drift=1e-8, r=0.002):
        self.F = np.array([[1.0, 1.0], [0.0, 1.0]])
        self.Q = np.array([[q_level, 0.0], [0.0, q_drift]])
        self.H = np.array([1.0, 0.0]).reshape(1, 2)
        self.R = np.array([[r]])
        self.state = None
        self.cov   = np.eye(2) * 0.1

    def update(self, log_spread):
        if self.state is None:
            self.state = np.array([log_spread, 0.0])
            return log_spread, 0.0, 0.0, 0.0  # level, fair, dev, zscore

        # Predict
        s_pred  = self.F @ self.state
        c_pred  = self.F @ self.cov @ self.F.T + self.Q

        # Update
        innov   = log_spread - self.H @ s_pred
        S       = self.H @ c_pred @ self.H.T + self.R
        K       = c_pred @ self.H.T / S[0, 0]
        self.state = s_pred + K.flatten() * innov
        self.cov   = c_pred - K @ self.H @ c_pred

        level   = self.state[0]   # fair-value log spread
        dev     = log_spread - level
        zscore  = dev / max(np.sqrt(self.cov[0, 0]), 1e-9)
        drift   = self.state[1]

        return level, drift, dev, zscore


def run_kalman_spread(alt_close, btc_close, q_level=1e-5, q_drift=1e-8, r=0.002):
    """
    Return arrays: direction (0=SHORT,1=FLAT,2=LONG), confidence, dev, zscore.
    Signal based on zscore deviation from fair value:
      z > +2.5 → altcoin overbought vs BTC → SHORT signal (bearish alt)
      z < -2.5 → altcoin oversold vs BTC → LONG signal (bullish alt)
    """
    n = len(alt_close)
    kf = KalmanSpread(q_level=q_level, q_drift=q_drift, r=r)

    direction  = np.ones(n, dtype=np.int32)
    confidence = np.zeros(n, dtype=np.float64)
    zscore_arr = np.zeros(n, dtype=np.float64)
    dev_arr    = np.zeros(n, dtype=np.float64)

    for i in range(n):
        log_spread = np.log(max(alt_close[i], 1e-9) / max(btc_close[i], 1e-9))
        level, drift, dev, z = kf.update(log_spread)
        zscore_arr[i] = z
        dev_arr[i]    = dev

        # z-score thresholds
        if z < -2.5:
            direction[i]  = 2   # oversold → LONG alt
            confidence[i] = min(abs(z) / 5.0, 0.90)
        elif z > 2.5:
            direction[i]  = 0   # overbought → SHORT alt
            confidence[i] = min(abs(z) / 5.0, 0.90)
        elif z < -1.5:
            direction[i]  = 2   # mild oversold → weak LONG
            confidence[i] = min(abs(z) / 5.0, 0.50)
        elif z > 1.5:
            direction[i]  = 0   # mild overbought → weak SHORT
            confidence[i] = min(abs(z) / 5.0, 0.50)
        else:
            direction[i]  = 1   # in range → FLAT
            confidence[i] = 0.0

    return direction, confidence, zscore_arr, dev_arr


def simulate(yp, close, high, low, atr):
    """Live-like simulation: SL + time exit, no TP."""
    n = len(yp); trades = []; i = 0
    while i < n:
        sig = int(yp[i])
        if sig == 1: i += 1; continue
        d  = 1 if sig == 2 else -1; entry = close[i]
        sl = entry - d * SL_MULT * atr[i]
        ep = close[min(i + MAX_HOLD, n - 1)]; eb = min(i + MAX_HOLD, n - 1)
        oc = "TIME_EXIT"
        for j in range(i + 1, min(i + MAX_HOLD + 1, n)):
            if (d == 1 and low[j] <= sl) or (d == -1 and high[j] >= sl):
                ep, eb, oc = sl, j, "SL"; break
        ret = (ep - entry) / entry * d
        net = ret * MODAL * LEV - COST_RT * MODAL * LEV
        trades.append({"dir": "L" if d == 1 else "S", "out": oc, "pnl": net, "bh": eb - i})
        i = eb + 1
    return trades


def accumulate(agg, tr):
    for t in tr:
        agg["t"] += 1; agg["p"] += t["pnl"]
        if t["pnl"] > 0: agg["w"] += 1
        if t["out"] == "SL": agg["sl"] += 1
        if t["dir"] == "L":
            agg["ls"] += 1
            if t["pnl"] > 0: agg["lw"] += 1

# ═══════════════════════════════════════════════════════════════════════════════
# Evaluate
# ═══════════════════════════════════════════════════════════════════════════════
available = [s for s in ALL_COINS
             if (HOLDOUT_DIR / "labeled" / f"{s}_features_v3.parquet").exists()]

# Coins to exclude from Kalman spread analysis (BTC itself + non-comparable)
SKIP_SPREAD = {"BTCUSDT"}

agg = {k: {"t":0,"w":0,"p":0.0,"ls":0,"lw":0,"sl":0}
       for k in ["bare_lgbm", "kalman_only", "combined_agree", "combined_cascade"]}
stats = {"agree":0, "disagree":0, "lgbm_only":0, "kalman_only":0}

BTC_PARQUET = HOLDOUT_DIR / "labeled" / "BTCUSDT_features_v3.parquet"

for sym in available:
    df = pd.read_parquet(HOLDOUT_DIR / "labeled" / f"{sym}_features_v3.parquet")
    df = ensure_utc_index(df).sort_index()

    rp = HOLDOUT_DIR / "labeled" / f"{sym}_regime_h1.parquet"
    hmm = np.full(len(df), 1, np.int32)
    if rp.exists():
        reg = pd.read_parquet(rp)
        if "hmm_regime_enc" in reg.columns:
            hmm = reg["hmm_regime_enc"].reindex(df.index, fill_value=1).values.astype(np.int32)

    mask = df["label"].isin(LM)
    df = df[mask].copy(); hmm = hmm[mask.values]; n = len(df)
    close = df["close"].values.astype(np.float64)
    high  = df["high"].values.astype(np.float64)
    low   = df["low"].values.astype(np.float64)
    atr   = df["atr_14_h1"].values.astype(np.float64)

    # LGBM predictions
    X = np.zeros((n, len(tb_feats)), dtype=np.float64)
    for idx, c in enumerate(tb_feats):
        if c in df.columns:
            X[:, idx] = df[c].ffill().fillna(0).values.astype(np.float64)
    proba = tb_model.predict_proba(X)
    conf  = np.max(proba, axis=1)
    yp_lgbm = np.argmax(proba, axis=1).astype(np.int32)
    for r, th in REGIME_THRESH.items():
        yp_lgbm[(hmm == r) & (yp_lgbm != 1) & (conf < th)] = 1

    # Kalman Spread (skip for BTC)
    yp_spread = np.ones(n, np.int32)
    conf_spread = np.zeros(n, np.float64)

    if sym not in SKIP_SPREAD and BTC_PARQUET.exists():
        btc_df = pd.read_parquet(BTC_PARQUET)
        btc_df = ensure_utc_index(btc_df).sort_index()
        btc_close = btc_df["close"].reindex(df.index).ffill().bfill().values.astype(np.float64)
        yp_spread, conf_spread, _, _ = run_kalman_spread(close, btc_close)
    elif sym in SKIP_SPREAD:
        pass  # BTC: Kalman spread not applicable

    # Signal combination
    yp_agree    = yp_lgbm.copy()
    yp_cascade  = yp_lgbm.copy()
    conf_cascade = conf.copy()

    for i in range(n):
        sl = int(yp_lgbm[i]); sk = int(yp_spread[i])

        if sl != 1 and sk != 1:
            if sl == sk: stats["agree"] += 1
            else:        stats["disagree"] += 1
        elif sl != 1 and sk == 1: stats["lgbm_only"] += 1
        elif sl == 1 and sk != 1: stats["kalman_only"] += 1

        # Hard agree: only enter if BOTH agree on direction
        if sl != 1 and (sk == 1 or sk == sl):
            pass  # keep LGBM signal
        elif sl != 1 and sk != 1 and sk != sl:
            yp_agree[i] = 1  # disagree → skip
        # Kalman only but no LGBM → skip (hard agree only uses LGBM gates)

        # Cascade: Kalman adjusts LGBM confidence
        if sl == 1:
            conf_cascade[i] = 0.0
        elif sk == sl:
            conf_cascade[i] = min(float(proba[i, sl]) + 0.03, 1.0)
        elif sk != 1 and sk != sl:
            conf_cascade[i] = max(float(proba[i, sl]) - 0.25, 0.0)
            if conf_cascade[i] < 0.40:
                yp_cascade[i] = 1

    tr_bare   = simulate(yp_lgbm, close, high, low, atr)
    tr_kal    = simulate(yp_spread, close, high, low, atr)
    tr_agree  = simulate(yp_agree, close, high, low, atr)
    tr_cascade= simulate(yp_cascade, close, high, low, atr)

    accumulate(agg["bare_lgbm"],       tr_bare)
    accumulate(agg["kalman_only"],     tr_kal)
    accumulate(agg["combined_agree"],  tr_agree)
    accumulate(agg["combined_cascade"],tr_cascade)

    print(".", end="", flush=True)
print()


def sc(agg, name):
    t=agg["t"]; w=agg["w"]; p=agg["p"]
    l=agg["ls"]; lw=agg["lw"]; sl=agg["sl"]
    return {
        "name":name,"trades":t,"wr":round(w/max(t,1)*100,2),
        "lwr":round(lw/max(l,1)*100,2),
        "swr":round((w-lw)/max(t-l,1)*100,2),
        "lpct":round(l/max(t,1)*100,2),
        "slr":round(sl/max(t,1)*100,2),
        "pnl":round(p,2),"ppm":round(p/5,2),"ppt":round(p/max(t,1),3),
    }

ss = {k: sc(agg[k], k) for k in agg}

total_sig = stats["agree"] + stats["disagree"]
agree_pct = stats["agree"] / max(total_sig, 1) * 100

print(f"\n{'='*80}")
print(f"  KALMAN SPREAD — log(alt/BTC) mean-reversion | {len(available)} coins")
print(f"  Signal agreement: {agree_pct:.1f}% ({stats['agree']:,}/{total_sig:,})")
print(f"  LGBM only: {stats['lgbm_only']:,}  Kalman only: {stats['kalman_only']:,}")
print(f"  Disagree (LGBM vs Kalman): {stats['disagree']:,}")
print(f"{'='*80}")
print(f"  {'Metrik':<22} {'LGBM bare':>12} {'Kalman only':>12} "
      f"{'Agree':>12} {'Cascade':>12}")
print(f"  {'-'*72}")

rows = [
    ("Total Trades",   lambda s: f"{s['trades']:,}"),
    ("Win Rate",       lambda s: f"{s['wr']:.1f}%"),
    ("  LONG WR",      lambda s: f"{s['lwr']:.1f}% ({s['lpct']:.0f}%)"),
    ("  SHORT WR",     lambda s: f"{s['swr']:.1f}%"),
    ("SL hit rate",    lambda s: f"{s['slr']:.1f}%"),
    ("Net PnL (5 bln)",lambda s: f"${s['pnl']:+.0f}"),
    ("PnL/bulan",      lambda s: f"${s['ppm']:+.0f}"),
    ("PnL/trade",      lambda s: f"${s['ppt']:+.3f}"),
]
for label, fn in rows:
    vals = "".join(f" {fn(ss[k]):>12}" for k in agg)
    print(f"  {label:<22}{vals}")

best = max(ss, key=lambda k: ss[k]["pnl"])
print(f"\n  Best: {best}  (${ss[best]['pnl']:+.0f})")
print(f"{'='*80}")
