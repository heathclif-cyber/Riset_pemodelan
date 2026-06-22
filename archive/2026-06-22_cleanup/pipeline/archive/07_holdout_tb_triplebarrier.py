"""
pipeline/07_holdout_tb_triplebarrier.py
Holdout OOS backtest TB v3 widyawardhana_v3 — Triple Barrier simulation.

Model predict LONG/SHORT/FLAT, exit dengan barrier yang sama seperti training:
  TP = 2.0 x ATR, SL = 1.5 x ATR, max_hold = 36 bar
  $10/trade, 5x leverage, fee 0.04%/side + slippage 0.02%/side
"""
import json, sys, warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import joblib
from core.utils import setup_logger, ensure_utc_index
from config import *

logger = setup_logger("07_holdout_tb_tb")

TB_RUN = "tb_lgbm_widyawardhana_v3"
LM     = {"SHORT": 0, "FLAT": 1, "LONG": 2}

TP_MULT      = TP_SL_FALLBACK_TP    # 2.0
SL_MULT      = TP_SL_FALLBACK_SL    # 1.5
MAX_HOLD     = MAX_HOLDING_BARS     # 36
MODAL        = MODAL_PER_TRADE      # $10
LEVERAGE     = LEVERAGE_SIM[0] if isinstance(LEVERAGE_SIM, list) else LEVERAGE_SIM
FEE          = FEE_PER_SIDE         # 0.0004
SLIP         = SLIPPAGE_PER_SIDE    # 0.0002
COST_RT      = (FEE + SLIP) * 2     # round-trip cost rate

# Regime-aware threshold (sama seperti training)
REGIME_THRESH = {0: 0.45, 1: 0.50, 2: 0.50, 3: 0.45}

# ── Load model ─────────────────────────────────────────────────────────────────
model = joblib.load(MODEL_DIR / "runs" / TB_RUN / "lgbm.pkl")
with open(MODEL_DIR / "runs" / TB_RUN / f"{TB_RUN}_features.json") as f:
    feats = json.load(f)

available = [s for s in ALL_COINS
             if (HOLDOUT_DIR / "labeled" / f"{s}_features_v3.parquet").exists()]

print(f"\n{'='*65}")
print(f"  HOLDOUT: TB v3 widyawardhana_v3 — Triple Barrier Simulation")
print(f"  Coins : {len(available)} | Period: Nov 2025 – Apr 2026")
print(f"  Exit  : TP={TP_MULT}xATR  SL={SL_MULT}xATR  max_hold={MAX_HOLD} bar")
print(f"  Trade : ${MODAL}/trade  {LEVERAGE}x leverage  fee+slip={COST_RT*100:.2f}% RT")
print(f"{'='*65}\n")


def simulate_triple_barrier(yp, close, high, low, atr):
    """
    Triple Barrier simulation.
    Returns list of trade dicts.
    """
    n      = len(yp)
    trades = []
    i      = 0
    while i < n:
        sig = yp[i]
        if sig == 1:   # FLAT
            i += 1
            continue

        direction = 1 if sig == 2 else -1   # LONG=1, SHORT=-1
        entry     = close[i]
        tp_price  = entry + direction * TP_MULT * atr[i]
        sl_price  = entry - direction * SL_MULT * atr[i]

        exit_price  = close[min(i + MAX_HOLD, n - 1)]
        exit_bar    = min(i + MAX_HOLD, n - 1)
        outcome     = "TIMEOUT"

        for j in range(i + 1, min(i + MAX_HOLD + 1, n)):
            if direction == 1:
                if high[j] >= tp_price:
                    exit_price = tp_price
                    exit_bar   = j
                    outcome    = "TP"
                    break
                if low[j] <= sl_price:
                    exit_price = sl_price
                    exit_bar   = j
                    outcome    = "SL"
                    break
            else:
                if low[j] <= tp_price:
                    exit_price = tp_price
                    exit_bar   = j
                    outcome    = "TP"
                    break
                if high[j] >= sl_price:
                    exit_price = sl_price
                    exit_bar   = j
                    outcome    = "SL"
                    break

        ret      = (exit_price - entry) / entry * direction
        gross    = ret * MODAL * LEVERAGE
        net_pnl  = gross - COST_RT * MODAL * LEVERAGE

        trades.append({
            "entry_bar": i, "exit_bar": exit_bar,
            "direction": "LONG" if direction == 1 else "SHORT",
            "entry": entry, "exit": exit_price,
            "outcome": outcome, "net_pnl": net_pnl,
            "bars_held": exit_bar - i,
        })

        i = exit_bar + 1   # no overlap — next trade after this exits

    return trades


# ── Per-coin loop ──────────────────────────────────────────────────────────────
agg = {
    "trades": 0, "wins": 0, "tp_hits": 0, "sl_hits": 0, "timeouts": 0,
    "pnl": 0.0, "longs": 0, "long_wins": 0, "bars_held": [],
}

for sym in available:
    df = pd.read_parquet(HOLDOUT_DIR / "labeled" / f"{sym}_features_v3.parquet")
    df = ensure_utc_index(df).sort_index()

    # HMM regime for threshold gating
    rp  = HOLDOUT_DIR / "labeled" / f"{sym}_regime_h1.parquet"
    hmm = np.full(len(df), 1, np.int32)
    if rp.exists():
        reg = pd.read_parquet(rp)
        if "hmm_regime_enc" in reg.columns:
            hmm = reg["hmm_regime_enc"].reindex(df.index, fill_value=1).values.astype(np.int32)

    mask = df["label"].isin(LM)
    df   = df[mask].copy()
    hmm  = hmm[mask.values]
    n    = len(df)

    close = df["close"].values.astype(np.float64)
    high  = df["high"].values.astype(np.float64)
    low   = df["low"].values.astype(np.float64)
    atr   = df["atr_14_h1"].values.astype(np.float64)

    # Feature matrix
    X = np.zeros((n, len(feats)), dtype=np.float64)
    for idx, c in enumerate(feats):
        if c in df.columns:
            X[:, idx] = df[c].ffill().fillna(0).values.astype(np.float64)

    # Predict
    proba = model.predict_proba(X)
    conf  = np.max(proba, axis=1)
    yp    = np.argmax(proba, axis=1).astype(np.int32)

    # Regime-aware threshold gate
    for r, th in REGIME_THRESH.items():
        yp[(hmm == r) & (yp != 1) & (conf < th)] = 1

    # Simulate
    trades = simulate_triple_barrier(yp, close, high, low, atr)

    for t in trades:
        agg["trades"]    += 1
        agg["pnl"]       += t["net_pnl"]
        agg["bars_held"].append(t["bars_held"])
        if t["net_pnl"] > 0:
            agg["wins"] += 1
        if t["outcome"] == "TP":
            agg["tp_hits"] += 1
        elif t["outcome"] == "SL":
            agg["sl_hits"] += 1
        else:
            agg["timeouts"] += 1
        if t["direction"] == "LONG":
            agg["longs"] += 1
            if t["net_pnl"] > 0:
                agg["long_wins"] += 1

    logger.info(f"[{sym}] trades={len(trades)}  "
                f"TP={sum(1 for t in trades if t['outcome']=='TP')}  "
                f"SL={sum(1 for t in trades if t['outcome']=='SL')}  "
                f"pnl={sum(t['net_pnl'] for t in trades):+.2f}")


# ── Scorecard ──────────────────────────────────────────────────────────────────
t   = agg["trades"]
w   = agg["wins"]
p   = agg["pnl"]
l   = agg["longs"]
lw  = agg["long_wins"]
tp  = agg["tp_hits"]
sl  = agg["sl_hits"]
to  = agg["timeouts"]
avg_hold = np.mean(agg["bars_held"]) if agg["bars_held"] else 0

wr   = w  / max(t, 1) * 100
lwr  = lw / max(l, 1) * 100
swr  = (w - lw) / max(t - l, 1) * 100

print(f"{'='*65}")
print(f"  SCORECARD — TB v3 widyawardhana_v3 | Triple Barrier OOS")
print(f"  Holdout: Nov 2025 – Apr 2026 | {len(available)} coins | $10/trade 5x")
print(f"{'='*65}")
rows = [
    ("Total Trades",     f"{t:,}"),
    ("Trades/bulan",     f"{t/5:.0f}"),
    ("Win Rate",         f"{wr:.1f}%"),
    ("  LONG WR",        f"{lwr:.1f}%  ({l} trades, {l/max(t,1)*100:.1f}%)"),
    ("  SHORT WR",       f"{swr:.1f}%  ({t-l} trades)"),
    ("TP hit rate",      f"{tp/max(t,1)*100:.1f}%  ({tp} trades)"),
    ("SL hit rate",      f"{sl/max(t,1)*100:.1f}%  ({sl} trades)"),
    ("Timeout rate",     f"{to/max(t,1)*100:.1f}%  ({to} trades)"),
    ("Avg hold (bars)",  f"{avg_hold:.1f}"),
    ("Net PnL (5 bln)",  f"${p:+.0f}"),
    ("PnL/bulan",        f"${p/5:+.0f}"),
    ("PnL/trade",        f"${p/max(t,1):+.3f}"),
]
for label, val in rows:
    print(f"  {label:<22} {val}")
print(f"{'='*65}")

# Save
out = {
    "model": TB_RUN,
    "simulation": "triple_barrier",
    "tp_mult": TP_MULT, "sl_mult": SL_MULT, "max_hold": MAX_HOLD,
    "trades": t, "wins": w, "wr": round(wr, 2),
    "long_wr": round(lwr, 2), "short_wr": round(swr, 2),
    "long_pct": round(l / max(t, 1) * 100, 2),
    "tp_rate": round(tp / max(t, 1) * 100, 2),
    "sl_rate": round(sl / max(t, 1) * 100, 2),
    "timeout_rate": round(to / max(t, 1) * 100, 2),
    "avg_hold_bars": round(avg_hold, 1),
    "pnl": round(p, 2),
    "pnl_per_trade": round(p / max(t, 1), 3),
    "pnl_per_month": round(p / 5, 2),
    "period": "Nov2025-Apr2026",
    "coins": len(available),
}
out_path = MODEL_DIR / "runs" / TB_RUN / "holdout_triplebarrier.json"
with open(out_path, "w") as f:
    json.dump(out, f, indent=2)
print(f"\n  Saved -> {out_path}")
