"""
pipeline/07_holdout_livelike.py
Evaluasi OOS kedua model dengan pendekatan live:
  - Entry  : close[i] saat model signal
  - SL     : 1.5 x ATR (hard stop, tidak berubah)
  - TP     : TIDAK ADA — biarkan berjalan sampai SL atau time exit
  - Exit   : SL hit ATAU max_hold=36 bar
  - Cost   : fee + slippage per sisi

Model:
  1. ic32_regime_v1       — threshold long>=0.69 / short>=0.59
  2. tb_widyawardhana_v3  — regime-aware threshold 0.45-0.50
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

logger = setup_logger("07_livelike")

LM       = {"SHORT": 0, "FLAT": 1, "LONG": 2}
SL_MULT  = TP_SL_FALLBACK_SL          # 1.5
MAX_HOLD = MAX_HOLDING_BARS            # 36
MODAL    = MODAL_PER_TRADE             # $10
LEVERAGE = LEVERAGE_SIM[0] if isinstance(LEVERAGE_SIM, list) else LEVERAGE_SIM
COST_RT  = (FEE_PER_SIDE + SLIPPAGE_PER_SIDE) * 2

# Thresholds
IC32_THR_LONG  = LGBM_THRESHOLD_LONG   # 0.69
IC32_THR_SHORT = LGBM_THRESHOLD_SHORT  # 0.59
REGIME_THRESH  = {0: 0.45, 1: 0.50, 2: 0.50, 3: 0.45}

# ── Load models ────────────────────────────────────────────────────────────────
ic32_model = joblib.load(MODEL_DIR / "runs" / "ic32_regime_v1" / "lgbm.pkl")
ic32_feats = list(ic32_model.feature_name_)

tb_model = joblib.load(MODEL_DIR / "runs" / "tb_lgbm_widyawardhana_v3" / "lgbm.pkl")
with open(MODEL_DIR / "runs" / "tb_lgbm_widyawardhana_v3" / "tb_lgbm_widyawardhana_v3_features.json") as f:
    tb_feats = json.load(f)

available = [s for s in ALL_COINS
             if (HOLDOUT_DIR / "labeled" / f"{s}_features_v3.parquet").exists()]

print(f"\n{'='*65}")
print(f"  HOLDOUT OOS — Live-like Evaluation")
print(f"  Model 1 : ic32_regime_v1  (long>={IC32_THR_LONG} / short>={IC32_THR_SHORT})")
print(f"  Model 2 : tb_widyawardhana_v3  (regime-aware {min(REGIME_THRESH.values())}-{max(REGIME_THRESH.values())})")
print(f"  Exit    : SL={SL_MULT}xATR (hard) | max_hold={MAX_HOLD} bar | NO TP")
print(f"  Coins   : {len(available)} | Nov 2025 - Apr 2026 | ${MODAL}/trade {LEVERAGE}x")
print(f"{'='*65}\n")


def simulate_livelike(yp, close, high, low, atr):
    """
    Live-like simulation: SL hard stop, no TP, time exit.
    Sequential: next trade dimulai setelah trade sebelumnya selesai.
    """
    n      = len(yp)
    trades = []
    i      = 0
    while i < n:
        sig = yp[i]
        if sig == 1:
            i += 1
            continue

        direction = 1 if sig == 2 else -1
        entry     = close[i]
        sl_price  = entry - direction * SL_MULT * atr[i]

        exit_price = close[min(i + MAX_HOLD, n - 1)]
        exit_bar   = min(i + MAX_HOLD, n - 1)
        outcome    = "TIME_EXIT"

        for j in range(i + 1, min(i + MAX_HOLD + 1, n)):
            if direction == 1 and low[j] <= sl_price:
                exit_price = sl_price
                exit_bar   = j
                outcome    = "SL"
                break
            elif direction == -1 and high[j] >= sl_price:
                exit_price = sl_price
                exit_bar   = j
                outcome    = "SL"
                break

        ret     = (exit_price - entry) / entry * direction
        net_pnl = ret * MODAL * LEVERAGE - COST_RT * MODAL * LEVERAGE

        trades.append({
            "direction": "LONG" if direction == 1 else "SHORT",
            "outcome"  : outcome,
            "net_pnl"  : net_pnl,
            "bars_held": exit_bar - i,
        })
        i = exit_bar + 1

    return trades


def new_agg():
    return {"trades": 0, "wins": 0, "pnl": 0.0,
            "longs": 0, "long_wins": 0, "sl_hits": 0, "bars_held": []}

ic32_agg = new_agg()
tb_agg   = new_agg()

for sym in available:
    df = pd.read_parquet(HOLDOUT_DIR / "labeled" / f"{sym}_features_v3.parquet")
    df = ensure_utc_index(df).sort_index()

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

    # ── ic32_regime_v1 predictions ─────────────────────────────────────────
    X_ic = np.zeros((n, len(ic32_feats)), dtype=np.float64)
    for idx, c in enumerate(ic32_feats):
        if c in df.columns:
            X_ic[:, idx] = df[c].ffill().fillna(0).values.astype(np.float64)
        elif c == "hmm_regime_enc":
            X_ic[:, idx] = hmm.astype(np.float64)

    p_ic  = ic32_model.predict_proba(X_ic)
    yp_ic = np.ones(n, dtype=np.int32)
    yp_ic[p_ic[:, 2] >= IC32_THR_LONG]  = 2
    yp_ic[(p_ic[:, 0] >= IC32_THR_SHORT) & (yp_ic != 2)] = 0

    # ── tb_widyawardhana_v3 predictions ────────────────────────────────────
    X_tb = np.zeros((n, len(tb_feats)), dtype=np.float64)
    for idx, c in enumerate(tb_feats):
        if c in df.columns:
            X_tb[:, idx] = df[c].ffill().fillna(0).values.astype(np.float64)

    p_tb   = tb_model.predict_proba(X_tb)
    conf_tb = np.max(p_tb, axis=1)
    yp_tb  = np.argmax(p_tb, axis=1).astype(np.int32)
    for r, th in REGIME_THRESH.items():
        yp_tb[(hmm == r) & (yp_tb != 1) & (conf_tb < th)] = 1

    # ── Simulate ───────────────────────────────────────────────────────────
    for agg, yp, tag in [(ic32_agg, yp_ic, "ic32"), (tb_agg, yp_tb, "tb")]:
        trades = simulate_livelike(yp, close, high, low, atr)
        for t in trades:
            agg["trades"]    += 1
            agg["pnl"]       += t["net_pnl"]
            agg["bars_held"].append(t["bars_held"])
            if t["net_pnl"] > 0:
                agg["wins"] += 1
            if t["outcome"] == "SL":
                agg["sl_hits"] += 1
            if t["direction"] == "LONG":
                agg["longs"] += 1
                if t["net_pnl"] > 0:
                    agg["long_wins"] += 1

    logger.info(f"[{sym}] ic32={len(simulate_livelike(yp_ic,close,high,low,atr))} "
                f"tb={len(simulate_livelike(yp_tb,close,high,low,atr))}")


# ── Print scorecard ────────────────────────────────────────────────────────────
def sc(agg):
    t  = agg["trades"]; w = agg["wins"]; p = agg["pnl"]
    l  = agg["longs"];  lw = agg["long_wins"]; sl = agg["sl_hits"]
    bh = agg["bars_held"]
    return {
        "trades"       : t,
        "wr"           : w / max(t, 1) * 100,
        "long_wr"      : lw / max(l, 1) * 100,
        "short_wr"     : (w - lw) / max(t - l, 1) * 100,
        "long_pct"     : l / max(t, 1) * 100,
        "sl_rate"      : sl / max(t, 1) * 100,
        "time_exit_rate": (t - sl) / max(t, 1) * 100,
        "avg_hold"     : np.mean(bh) if bh else 0,
        "pnl"          : p,
        "ppm"          : p / 5,
        "ppt"          : p / max(t, 1),
    }

s1 = sc(ic32_agg)
s2 = sc(tb_agg)

W = 22
print(f"\n{'='*65}")
print(f"  SCORECARD — Live-like OOS | Nov 2025 – Apr 2026")
print(f"  Exit: SL={SL_MULT}xATR hard stop | max_hold={MAX_HOLD} bar | NO TP")
print(f"{'='*65}")
print(f"  {'Metrik':<26} {'ic32_regime_v1':>{W}} {'tb_widyawardhana_v3':>{W}}")
print(f"  {'-'*62}")

rows = [
    ("Total Trades",      f"{s1['trades']:,}",                     f"{s2['trades']:,}"),
    ("Trades/bulan",      f"{s1['trades']/5:.0f}",                 f"{s2['trades']/5:.0f}"),
    ("Win Rate",          f"{s1['wr']:.1f}%",                      f"{s2['wr']:.1f}%"),
    ("  LONG WR",         f"{s1['long_wr']:.1f}% ({s1['long_pct']:.0f}% share)",
                          f"{s2['long_wr']:.1f}% ({s2['long_pct']:.0f}% share)"),
    ("  SHORT WR",        f"{s1['short_wr']:.1f}%",                f"{s2['short_wr']:.1f}%"),
    ("SL hit rate",       f"{s1['sl_rate']:.1f}%",                 f"{s2['sl_rate']:.1f}%"),
    ("Time exit rate",    f"{s1['time_exit_rate']:.1f}%",          f"{s2['time_exit_rate']:.1f}%"),
    ("Avg hold (bar)",    f"{s1['avg_hold']:.1f}",                  f"{s2['avg_hold']:.1f}"),
    ("Net PnL (5 bln)",   f"${s1['pnl']:+.0f}",                    f"${s2['pnl']:+.0f}"),
    ("PnL/bulan",         f"${s1['ppm']:+.0f}",                    f"${s2['ppm']:+.0f}"),
    ("PnL/trade",         f"${s1['ppt']:+.3f}",                    f"${s2['ppt']:+.3f}"),
]

for label, v1, v2 in rows:
    marker = " <--" if label == "Net PnL (5 bln)" else ""
    print(f"  {label:<26} {v1:>{W}} {v2:>{W}}{marker}")

delta = s2["pnl"] - s1["pnl"]
print(f"\n  Delta tb_widyawardhana vs ic32: {delta:+.0f} "
      f"({'tb menang' if delta > 0 else 'ic32 menang'})")

# Save
out = {
    "ic32_regime_v1"       : {**s1, "wr": round(s1["wr"], 2)},
    "tb_widyawardhana_v3"  : {**s2, "wr": round(s2["wr"], 2)},
    "exit_method"          : f"SL={SL_MULT}xATR + max_hold={MAX_HOLD}bar, no TP",
    "delta_pnl"            : round(delta, 2),
}
out_path = MODEL_DIR / "runs" / "tb_lgbm_widyawardhana_v3" / "holdout_livelike.json"
with open(out_path, "w") as f:
    json.dump(out, f, indent=2, default=float)
print(f"\n  Saved -> {out_path}")
print(f"{'='*65}")
