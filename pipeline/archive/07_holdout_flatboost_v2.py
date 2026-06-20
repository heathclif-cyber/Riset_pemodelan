"""
pipeline/07_holdout_flatboost_v2.py
Holdout backtest flatboost_v2 vs ic32_regime_v1.

Period  : Apr 2026 - Jun 2026 (OOS)
Eval    : Live-like — SL 1.5xATR hard stop | max_hold=36 | NO TP | win=net_pnl>0
          Sama metodologi dengan 07_holdout_livelike.py
"""
import json, sys, warnings, numpy as np, pandas as pd
from pathlib import Path
warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import joblib
from core.utils import setup_logger, ensure_utc_index
from config import *

logger = setup_logger("07_holdout_flatboost_v2")

FB_RUN   = "tb_lgbm_flatboost_v2"
IC32_RUN = "ic32_regime_v1"
PERIOD_MONTHS = 2.5

LM = {"SHORT": 0, "FLAT": 1, "LONG": 2}
FB_CONF_THRESHOLD = 0.40

SL_MULT  = TP_SL_FALLBACK_SL          # 1.5
MAX_HOLD = MAX_HOLDING_BARS            # 36
MODAL    = MODAL_PER_TRADE
LEVERAGE = LEVERAGE_SIM[0] if isinstance(LEVERAGE_SIM, list) else LEVERAGE_SIM
COST_RT  = (FEE_PER_SIDE + SLIPPAGE_PER_SIDE) * 2

# ── Load models ─────────────────────────────────────────────────────────────────
fb_model = joblib.load(MODEL_DIR / "runs" / FB_RUN / "lgbm.pkl")
with open(MODEL_DIR / "runs" / FB_RUN / f"{FB_RUN}_features.json") as f:
    fb_feats = json.load(f)

ic32_model = joblib.load(MODEL_DIR / "runs" / IC32_RUN / "lgbm.pkl")
ic32_feats = list(ic32_model.feature_name_)

available = sorted(
    s for s in ALL_COINS
    if (HOLDOUT_DIR / "labeled" / f"{s}_features_v3.parquet").exists()
)

print(f"\n{'='*72}")
print(f"  HOLDOUT: flatboost_v2 vs ic32_regime_v1")
print(f"  Coins: {len(available)} | Period: Apr 2026 - Jun 2026 (~{PERIOD_MONTHS} bln)")
print(f"  flatboost_v2 : {len(fb_feats)} feat | conf>={FB_CONF_THRESHOLD} | argmax")
print(f"  ic32         : {len(ic32_feats)} feat | LONG>={LGBM_THRESHOLD_LONG} | SHORT>={LGBM_THRESHOLD_SHORT}")
print(f"  Eval         : live-like (SL={SL_MULT}xATR + max_hold={MAX_HOLD}, NO TP, win=net_pnl>0)")
print(f"{'='*72}\n")


def simulate_livelike(yp, close, high, low, atr):
    """Live-like simulation: SL hard stop, no TP, time exit. Sequential."""
    n = len(yp)
    trades = []
    i = 0
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


# ── Aggregate accumulators ───────────────────────────────────────────────────────
def make_agg():
    return {"trades": 0, "wins": 0, "pnl": 0.0,
            "longs": 0, "long_wins": 0, "shorts": 0, "short_wins": 0,
            "sl_hits": 0, "bars_held": []}

fb_agg   = make_agg()
ic32_agg = make_agg()

# ── Per-coin loop ────────────────────────────────────────────────────────────────
for sym in available:
    df  = pd.read_parquet(HOLDOUT_DIR / "labeled" / f"{sym}_features_v3.parquet")
    df  = ensure_utc_index(df).sort_index()

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

    # flatboost_v2 predictions
    X_fb = np.zeros((n, len(fb_feats)), dtype=np.float64)
    for i, c in enumerate(fb_feats):
        if c in df.columns:
            X_fb[:, i] = df[c].ffill().fillna(0).values.astype(np.float64)

    proba_fb = fb_model.predict_proba(X_fb)
    conf_fb  = proba_fb.max(axis=1)
    yp_fb    = proba_fb.argmax(axis=1).astype(np.int32)
    yp_fb[conf_fb < FB_CONF_THRESHOLD] = 1

    # ic32_regime_v1 predictions
    X_ic = np.zeros((n, len(ic32_feats)), dtype=np.float64)
    for i, c in enumerate(ic32_feats):
        if c in df.columns:
            X_ic[:, i] = df[c].ffill().fillna(0).values.astype(np.float64)
        elif c == "hmm_regime_enc":
            X_ic[:, i] = hmm.astype(np.float64)

    proba_ic = ic32_model.predict_proba(X_ic)
    yp_ic    = np.ones(n, dtype=np.int32)
    yp_ic[proba_ic[:, 2] > LGBM_THRESHOLD_LONG]  = 2
    yp_ic[(proba_ic[:, 0] > LGBM_THRESHOLD_SHORT) & (yp_ic != 2)] = 0

    # Accumulate
    for agg, yp in [(fb_agg, yp_fb), (ic32_agg, yp_ic)]:
        for t in simulate_livelike(yp, close, high, low, atr):
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
            else:
                agg["shorts"] += 1
                if t["net_pnl"] > 0:
                    agg["short_wins"] += 1


# ── Scorecard ────────────────────────────────────────────────────────────────────
def scorecard(name, agg, months):
    t = agg["trades"]; w = agg["wins"]; p = agg["pnl"]
    l = agg["longs"];  lw = agg["long_wins"]
    s = agg["shorts"]; sw = agg["short_wins"]
    sl = agg["sl_hits"]; bh = agg["bars_held"]
    return {
        "name":            name,
        "trades":          t,
        "wr":              w  / max(t, 1) * 100,
        "long_wr":         lw / max(l, 1) * 100,
        "short_wr":        sw / max(s, 1) * 100,
        "long_pct":        l  / max(t, 1) * 100,
        "pnl":             p,
        "pnl_per_month":   p / months,
        "pnl_per_trade":   p / max(t, 1),
        "sl_rate":         sl / max(t, 1) * 100,
        "time_exit_rate":  (t - sl) / max(t, 1) * 100,
        "avg_hold":        float(np.mean(bh)) if bh else 0,
    }

fb_sc   = scorecard(f"flatboost_v2 ({len(fb_feats)} feat)", fb_agg,   PERIOD_MONTHS)
ic32_sc = scorecard("ic32_regime_v1 (benchmark)",           ic32_agg, PERIOD_MONTHS)

print(f"\n{'='*72}")
print(f"  SCORECARD - Standalone LGBM, Live-like Eval (SL+time exit, NO TP)")
print(f"  Holdout: Apr-Jun 2026 ({len(available)} koin, ~{PERIOD_MONTHS} bln, $10/5x)")
print(f"{'='*72}")
print(f"\n  {'Metrik':<30} {'flatboost_v2':>18} {'ic32 (bench)':>18}")
print("  " + "-" * 68)

rows = [
    ("Total Trades",        f"{fb_sc['trades']:,}",              f"{ic32_sc['trades']:,}"),
    ("Trades/bulan",        f"{fb_sc['trades']/PERIOD_MONTHS:.0f}", f"{ic32_sc['trades']/PERIOD_MONTHS:.0f}"),
    ("Win Rate",            f"{fb_sc['wr']:.1f}%",               f"{ic32_sc['wr']:.1f}%"),
    ("  LONG WR",           f"{fb_sc['long_wr']:.1f}%",          f"{ic32_sc['long_wr']:.1f}%"),
    ("  SHORT WR",          f"{fb_sc['short_wr']:.1f}%",         f"{ic32_sc['short_wr']:.1f}%"),
    ("  LONG share",        f"{fb_sc['long_pct']:.1f}%",         f"{ic32_sc['long_pct']:.1f}%"),
    ("Net PnL",             f"${fb_sc['pnl']:+.0f}",             f"${ic32_sc['pnl']:+.0f}"),
    ("PnL/bulan",           f"${fb_sc['pnl_per_month']:+.0f}",   f"${ic32_sc['pnl_per_month']:+.0f}"),
    ("PnL/trade",           f"${fb_sc['pnl_per_trade']:+.3f}",   f"${ic32_sc['pnl_per_trade']:+.3f}"),
    ("SL Hit Rate",         f"{fb_sc['sl_rate']:.1f}%",          f"{ic32_sc['sl_rate']:.1f}%"),
    ("Time Exit Rate",      f"{fb_sc['time_exit_rate']:.1f}%",   f"{ic32_sc['time_exit_rate']:.1f}%"),
    ("Avg Hold (bars)",     f"{fb_sc['avg_hold']:.1f}",          f"{ic32_sc['avg_hold']:.1f}"),
]
for label, v_fb, v_ic in rows:
    marker = " <--" if "Net PnL" in label else ""
    print(f"  {label:<30} {v_fb:>18} {v_ic:>18}{marker}")

delta  = fb_sc["pnl"] - ic32_sc["pnl"]
winner = "flatboost_v2" if delta > 0 else "ic32_regime_v1"
print(f"\n  Delta flatboost vs ic32: PnL {delta:+.0f}")
print(f"  -> {winner} lebih baik (standalone, Apr-Jun 2026)")

# ── Save ─────────────────────────────────────────────────────────────────────────
out = {
    "flatboost_v2":   fb_sc,
    "ic32_benchmark": ic32_sc,
    "delta_pnl":      delta,
    "period":         "2026-04-01 to 2026-06-13",
    "n_coins":        len(available),
    "note":           f"Standalone LGBM, no LSTM/Guardian, live-like eval: SL={SL_MULT}xATR + max_hold={MAX_HOLD} bar, NO TP, win=net_pnl>0",
}
out_path = MODEL_DIR / "runs" / FB_RUN / "holdout_apr_jun26_vs_ic32.json"
with open(out_path, "w") as f:
    json.dump(out, f, indent=2)
print(f"\n  Saved -> {out_path}")
print(f"{'='*72}")
