"""
Kalman Filter — Local Linear Trend untuk tb_widyawardhana_v3.
Estimasi hidden state (level + slope) dari H4 close price secara online.
Output: arah trend + confidence → dipakai sebagai filter/konfirmasi sinyal LGBM.

Blueprint ref: Section 6.3 — Kalman Filter, Layer 1 Ensemble Predictor
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

logger = setup_logger("kalman_dev")

LM = {"SHORT": 0, "FLAT": 1, "LONG": 2}
SL_MULT = 1.5
MAX_HOLD = 36
MODAL = 10
LEV = 5
COST_RT = (FEE_PER_SIDE + SLIPPAGE_PER_SIDE) * 2
REGIME_THRESH = {0: 0.45, 1: 0.50, 2: 0.50, 3: 0.45}

# ── Load tb_widyawardhana model ────────────────────────────────────────────────
tb_model = joblib.load(MODEL_DIR / "runs/tb_lgbm_widyawardhana_v3/lgbm.pkl")
with open(MODEL_DIR / "runs/tb_lgbm_widyawardhana_v3/tb_lgbm_widyawardhana_v3_features.json") as f:
    tb_feats = json.load(f)

# ── Kalman Filter Implementation ───────────────────────────────────────────────
class LocalLinearTrend:
    """
    Kalman Filter dengan model Local Linear Trend (level + slope).
    State: [level, slope]
    Observasi: harga close

    Update rekursif per bar — tidak butuh training.
    """
    def __init__(self, q_level=1e-4, q_slope=1e-6, r=0.01):
        # State: [level, slope]
        self.state = np.array([0.0, 0.0])         # akan diinisialisasi saat first obs
        self.cov   = np.eye(2) * 1.0              # initial uncertainty besar

        # Transition matrix: level' = level + slope, slope' = slope
        self.F = np.array([[1.0, 1.0],
                           [0.0, 1.0]])

        # Process noise: small random walk pada slope
        self.Q = np.array([[q_level, 0.0],
                           [0.0, q_slope]])

        # Observation matrix: kita hanya observasi level (harga)
        self.H = np.array([1.0, 0.0]).reshape(1, 2)
        self.R = np.array([[r]])                    # observation noise

        self.initialized = False

    def update(self, price):
        """Update filter dengan observasi harga baru. Returns (level, slope, slope_std)."""
        if not self.initialized:
            self.state = np.array([price, 0.0])
            self.initialized = True
            return price, 0.0, np.sqrt(self.cov[1, 1])

        # Predict
        state_pred = self.F @ self.state
        cov_pred   = self.F @ self.cov @ self.F.T + self.Q

        # Update
        innovation = price - self.H @ state_pred
        S          = self.H @ cov_pred @ self.H.T + self.R
        K          = cov_pred @ self.H.T / S[0, 0]

        self.state = state_pred + K.flatten() * innovation
        self.cov   = cov_pred - K @ self.H @ cov_pred

        level = self.state[0]
        slope = self.state[1]
        slope_std = np.sqrt(self.cov[1, 1])

        return level, slope, slope_std


def run_kalman_on_coin(close_prices, q_level=1e-4, q_slope=1e-7, r=0.005):
    """Run Kalman Filter on a price series. Returns dict of arrays."""
    n = len(close_prices)
    kf = LocalLinearTrend(q_level=q_level, q_slope=q_slope, r=r)

    level_arr   = np.zeros(n, dtype=np.float64)
    slope_arr   = np.zeros(n, dtype=np.float64)
    slope_std   = np.zeros(n, dtype=np.float64)
    direction   = np.ones(n, dtype=np.int32)       # 0=SHORT, 1=FLAT, 2=LONG
    confidence  = np.zeros(n, dtype=np.float64)

    for i in range(n):
        level_arr[i], slope_arr[i], slope_std[i] = kf.update(close_prices[i])

        # Signal: slope direction + significance
        sig = slope_arr[i] / max(slope_std[i], 1e-9)   # signal-to-noise ratio

        if sig > 0.5:
            direction[i] = 2        # LONG
            confidence[i] = min(abs(sig) / 3.0, 0.95)
        elif sig < -0.5:
            direction[i] = 0        # SHORT
            confidence[i] = min(abs(sig) / 3.0, 0.95)
        else:
            direction[i] = 1        # FLAT
            confidence[i] = max(0.0, 1.0 - abs(sig) / 0.5)

    return {
        "level": level_arr, "slope": slope_arr, "slope_std": slope_std,
        "direction": direction, "confidence": confidence,
    }


def simulate_kalman_filter(yp_lgbm, yp_kalman, conf_kalman,
                            close, high, low, atr):
    """
    Kalman filter as confirmation: require LGBM + Kalman agreement.
    - If Kalman agrees with LGBM → entry
    - If Kalman disagrees → skip (idle)
    """
    n = len(yp_lgbm)
    trades = []
    i = 0
    while i < n:
        sig_lgbm = yp_lgbm[i]
        if sig_lgbm == 1:
            i += 1; continue

        # Kalman confirmation
        sig_kal = yp_kalman[i]
        if sig_kal != sig_lgbm and sig_kal != 1:
            i += 1; continue   # Kalman opposes → skip

        direction = 1 if sig_lgbm == 2 else -1
        entry     = close[i]
        sl_price  = entry - direction * SL_MULT * atr[i]
        exit_price = close[min(i + MAX_HOLD, n - 1)]
        exit_bar   = min(i + MAX_HOLD, n - 1)
        outcome    = "TIME_EXIT"

        for j in range(i + 1, min(i + MAX_HOLD + 1, n)):
            if (direction == 1 and low[j] <= sl_price) or \
               (direction == -1 and high[j] >= sl_price):
                exit_price, exit_bar, outcome = sl_price, j, "SL"; break

        ret     = (exit_price - entry) / entry * direction
        net_pnl = ret * MODAL * LEV - COST_RT * MODAL * LEV

        trades.append({
            "direction": "LONG" if direction == 1 else "SHORT",
            "outcome"  : outcome,
            "net_pnl"  : net_pnl,
            "bars_held": exit_bar - i,
        })
        i = exit_bar + 1
    return trades


# ── Evaluate: 3 approaches ─────────────────────────────────────────────────────
available = [s for s in ALL_COINS
             if (HOLDOUT_DIR / "labeled" / f"{s}_features_v3.parquet").exists()]

Q_LEVEL  = 1e-3
Q_SLOPE  = 1e-7
R_NOISE  = 0.003
KALMAN_THRESH = 0.3   # lower threshold = more Kalman signals

# Re-run with lower threshold
def run_kalman_on_coin_v2(close_prices, q_level, q_slope, r, threshold):
    n = len(close_prices)
    kf = LocalLinearTrend(q_level=q_level, q_slope=q_slope, r=r)
    direction  = np.ones(n, dtype=np.int32)
    confidence = np.zeros(n, dtype=np.float64)
    slope_vals = np.zeros(n, dtype=np.float64)
    slope_std  = np.zeros(n, dtype=np.float64)

    for i in range(n):
        _, s, ss = kf.update(close_prices[i])
        slope_vals[i] = s; slope_std[i] = ss
        sig = s / max(ss, 1e-9)
        if sig > threshold:
            direction[i] = 2; confidence[i] = min(abs(sig)/3.0, 0.95)
        elif sig < -threshold:
            direction[i] = 0; confidence[i] = min(abs(sig)/3.0, 0.95)
        else:
            direction[i] = 1; confidence[i] = 0.0
    return {"direction": direction, "confidence": confidence,
            "slope": slope_vals, "slope_std": slope_std}


def simulate_cascade(yp_lgbm, conf_lgbm, proba_lgbm, yp_kal, conf_kal,
                      close, high, low, atr):
    """
    Cascade approach: Kalman adjusts LGBM confidence (not hard gate).
    Agree: +0.03 boost. Disagree (Kalman opposite): -0.20 penalty.
    Kalman FLAT: no adjustment.
    """
    n = len(yp_lgbm)
    # Adjust yp based on Kalman
    yp_adj    = yp_lgbm.copy()
    conf_adj  = np.zeros(n, dtype=np.float64)
    proba_adj = proba_lgbm.copy()

    for i in range(n):
        sig_l = int(yp_lgbm[i])
        sig_k = int(yp_kal[i])

        if sig_l == 1:
            conf_adj[i] = 0.0; continue

        base = float(proba_lgbm[i, sig_l])
        if sig_k == 1:
            adj = 0.0   # Kalman flat → no opinion
        elif sig_k == sig_l:
            adj = +0.03  # agree boost
        else:
            adj = -0.20  # disagree penalty

        new = np.clip(base + adj, 0.0, 1.0)
        proba_adj[i, sig_l] = new
        # Re-normalize other classes
        other = [j for j in range(3) if j != sig_l]
        surplus = max(0.0, base - new)
        proba_adj[i, other[0]] += surplus / 2
        proba_adj[i, other[1]] += surplus / 2

        thr = 0.45   # same regime threshold basis
        if new < thr:
            yp_adj[i] = 1  # degraded to FLAT
        conf_adj[i] = new

    # Simulate with adjusted signals
    trades = []
    i = 0
    while i < n:
        sig = int(yp_adj[i])
        if sig == 1:
            i += 1; continue

        direction = 1 if sig == 2 else -1
        entry     = close[i]
        sl_price  = entry - direction * SL_MULT * atr[i]
        exit_price = close[min(i + MAX_HOLD, n - 1)]
        exit_bar   = min(i + MAX_HOLD, n - 1)
        outcome    = "TIME_EXIT"

        for j in range(i + 1, min(i + MAX_HOLD + 1, n)):
            if (direction == 1 and low[j] <= sl_price) or \
               (direction == -1 and high[j] >= sl_price):
                exit_price, exit_bar, outcome = sl_price, j, "SL"; break

        ret     = (exit_price - entry) / entry * direction
        net_pnl = ret * MODAL * LEV - COST_RT * MODAL * LEV
        trades.append({
            "direction": "LONG" if direction == 1 else "SHORT",
            "outcome": outcome, "net_pnl": net_pnl, "bars_held": exit_bar - i,
        })
        i = exit_bar + 1
    return trades


def simulate_agree_only(yp_lgbm, yp_kal, close, high, low, atr):
    """Hard agree — Kalman must match LGBM (both non-FLAT) or Kalman is FLAT."""
    n = len(yp_lgbm)
    trades = []
    i = 0
    while i < n:
        sig_l = int(yp_lgbm[i])
        sig_k = int(yp_kal[i])
        if sig_l == 1:
            i += 1; continue
        # Skip if Kalman opposes (not FLAT and not same)
        if sig_k != 1 and sig_k != sig_l:
            i += 1; continue

        direction = 1 if sig_l == 2 else -1
        entry     = close[i]
        sl_price  = entry - direction * SL_MULT * atr[i]
        exit_price = close[min(i + MAX_HOLD, n - 1)]
        exit_bar   = min(i + MAX_HOLD, n - 1)
        outcome    = "TIME_EXIT"

        for j in range(i + 1, min(i + MAX_HOLD + 1, n)):
            if (direction == 1 and low[j] <= sl_price) or \
               (direction == -1 and high[j] >= sl_price):
                exit_price, exit_bar, outcome = sl_price, j, "SL"; break

        ret     = (exit_price - entry) / entry * direction
        net_pnl = ret * MODAL * LEV - COST_RT * MODAL * LEV
        trades.append({
            "direction": "LONG" if direction == 1 else "SHORT",
            "outcome": outcome, "net_pnl": net_pnl, "bars_held": exit_bar - i,
        })
        i = exit_bar + 1
    return trades


print(f"\n{'='*70}")
print(f"  KALMAN FILTER — 3 Approaches Comparison")
print(f"  Threshold={KALMAN_THRESH}  Q_level={Q_LEVEL}  Q_slope={Q_SLOPE}  R={R_NOISE}")
print(f"{'='*70}\n")

agg_bare   = {"trades": 0, "wins": 0, "pnl": 0.0, "longs": 0, "long_wins": 0, "sl": 0}
agg_hard   = {"trades": 0, "wins": 0, "pnl": 0.0, "longs": 0, "long_wins": 0, "sl": 0}
agg_cascade= {"trades": 0, "wins": 0, "pnl": 0.0, "longs": 0, "long_wins": 0, "sl": 0}
kalman_stats = {"agree": 0, "disagree": 0, "kal_flat": 0}

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
    df   = df[mask].copy(); hmm = hmm[mask.values]; n = len(df)
    close = df["close"].values.astype(np.float64)
    high  = df["high"].values.astype(np.float64)
    low   = df["low"].values.astype(np.float64)
    atr   = df["atr_14_h1"].values.astype(np.float64)

    X = np.zeros((n, len(tb_feats)), dtype=np.float64)
    for idx, c in enumerate(tb_feats):
        if c in df.columns:
            X[:, idx] = df[c].ffill().fillna(0).values.astype(np.float64)
    proba = tb_model.predict_proba(X)
    conf  = np.max(proba, axis=1)
    yp_lgbm = np.argmax(proba, axis=1).astype(np.int32)
    for r, th in REGIME_THRESH.items():
        yp_lgbm[(hmm == r) & (yp_lgbm != 1) & (conf < th)] = 1

    kf = run_kalman_on_coin_v2(close, Q_LEVEL, Q_SLOPE, R_NOISE, KALMAN_THRESH)
    yp_kal = kf["direction"]; conf_kal = kf["confidence"]

    sig_mask = yp_lgbm != 1
    ks = kalman_stats
    ks["agree"]    += int(((yp_lgbm == yp_kal) & sig_mask).sum())
    ks["disagree"] += int(((yp_lgbm != yp_kal) & sig_mask & (yp_kal != 1)).sum())
    ks["kal_flat"] += int((yp_kal == 1).sum())

    tr_bare    = simulate_agree_only(yp_lgbm, np.full(n, 1, np.int32), close, high, low, atr)
    tr_hard    = simulate_agree_only(yp_lgbm, yp_kal, close, high, low, atr)
    tr_cascade = simulate_cascade(yp_lgbm, conf, proba, yp_kal, conf_kal, close, high, low, atr)

    for agg, tr in [(agg_bare, tr_bare), (agg_hard, tr_hard), (agg_cascade, tr_cascade)]:
        for t in tr:
            agg["trades"] += 1; agg["pnl"] += t["net_pnl"]
            if t["net_pnl"] > 0: agg["wins"] += 1
            if t["outcome"] == "SL": agg["sl"] += 1
            if t["direction"] == "LONG":
                agg["longs"] += 1
                if t["net_pnl"] > 0: agg["long_wins"] += 1

    print(".", end="", flush=True)
print()


def sc(agg, name):
    t = agg["trades"]; w = agg["wins"]; p = agg["pnl"]
    l = agg["longs"]; lw = agg["long_wins"]; sl = agg["sl"]
    return {
        "name": name, "trades": t, "wr": round(w/max(t,1)*100,2),
        "long_wr": round(lw/max(l,1)*100,2),
        "short_wr": round((w-lw)/max(t-l,1)*100,2),
        "long_pct": round(l/max(t,1)*100,2),
        "sl_rate": round(sl/max(t,1)*100,2),
        "pnl": round(p,2), "ppm": round(p/5,2), "ppt": round(p/max(t,1),3),
    }

s_bare   = sc(agg_bare, "tb bare")
s_hard   = sc(agg_hard, "tb + Kalman hard")
s_cascade= sc(agg_cascade, "tb + Kalman cascade")

total_sig = kalman_stats["agree"] + kalman_stats["disagree"]
agree_pct = kalman_stats["agree"] / max(total_sig, 1) * 100

print(f"\n{'='*75}")
print(f"  KALMAN FILTER — {len(available)} coins | agree rate: {agree_pct:.1f}%")
print(f"  ({kalman_stats['agree']:,} agree / {kalman_stats['disagree']:,} disagree)")
print(f"{'='*75}")
print(f"  {'Metrik':<24} {'tb bare':>14} {'hard gate':>14} {'cascade':>14}")
print(f"  {'-'*68}")

rows = [
    ("Total Trades",     f"{s_bare['trades']:,}",    f"{s_hard['trades']:,}",    f"{s_cascade['trades']:,}"),
    ("Win Rate",         f"{s_bare['wr']:.1f}%",      f"{s_hard['wr']:.1f}%",      f"{s_cascade['wr']:.1f}%"),
    ("SL hit rate",      f"{s_bare['sl_rate']:.1f}%", f"{s_hard['sl_rate']:.1f}%", f"{s_cascade['sl_rate']:.1f}%"),
    ("Net PnL (5 bln)",  f"${s_bare['pnl']:+.0f}",    f"${s_hard['pnl']:+.0f}",    f"${s_cascade['pnl']:+.0f}"),
    ("PnL/bulan",        f"${s_bare['ppm']:+.0f}",     f"${s_hard['ppm']:+.0f}",     f"${s_cascade['ppm']:+.0f}"),
    ("PnL/trade",        f"${s_bare['ppt']:+.3f}",     f"${s_hard['ppt']:+.3f}",     f"${s_cascade['ppt']:+.3f}"),
]
for label, v1, v2, v3 in rows:
    print(f"  {label:<24} {v1:>14} {v2:>14} {v3:>14}")

best = max([("bare", s_bare["pnl"]), ("hard", s_hard["pnl"]), ("cascade", s_cascade["pnl"])], key=lambda x: x[1])
print(f"\n  Best: {best[0]} (${best[1]:+.0f})")
print(f"{'='*75}")
