"""
pipeline/07_holdout_flatboost_v3.py
Holdout backtest flatboost_v3 vs flatboost_v2 vs ic32_regime_v1.

Period  : Apr 2026 - Jun 2026 (OOS)
Eval    : Live-like — SL 1.5xATR hard stop | max_hold=36 | NO TP | win=net_pnl>0
          Sama metodologi dengan 07_holdout_livelike.py
New     : market_breadth precomputed cross-coin dari holdout data
"""
import json, sys, warnings, numpy as np, pandas as pd
from pathlib import Path
warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import joblib
from core.utils import setup_logger, ensure_utc_index
from config import *

logger = setup_logger("07_holdout_flatboost_v3")

V3_RUN   = "tb_lgbm_flatboost_v3"
V2_RUN   = "tb_lgbm_flatboost_v2"
IC32_RUN = "ic32_regime_v1"
PERIOD_MONTHS = 2.5

LM = {"SHORT": 0, "FLAT": 1, "LONG": 2}
FB_CONF_THRESHOLD = 0.40

SL_MULT  = TP_SL_FALLBACK_SL
MAX_HOLD = MAX_HOLDING_BARS
MODAL    = MODAL_PER_TRADE
LEVERAGE = LEVERAGE_SIM[0] if isinstance(LEVERAGE_SIM, list) else LEVERAGE_SIM
COST_RT  = (FEE_PER_SIDE + SLIPPAGE_PER_SIDE) * 2

# ── Load models ──────────────────────────────────────────────────────────────
v3_model  = joblib.load(MODEL_DIR / "runs" / V3_RUN / "lgbm.pkl")
with open(MODEL_DIR / "runs" / V3_RUN / f"{V3_RUN}_features.json") as f:
    v3_feats = json.load(f)

v2_model  = joblib.load(MODEL_DIR / "runs" / V2_RUN / "lgbm.pkl")
with open(MODEL_DIR / "runs" / V2_RUN / f"{V2_RUN}_features.json") as f:
    v2_feats = json.load(f)

ic32_model = joblib.load(MODEL_DIR / "runs" / IC32_RUN / "lgbm.pkl")
ic32_feats = list(ic32_model.feature_name_)

available = sorted(
    s for s in ALL_COINS
    if (HOLDOUT_DIR / "labeled" / f"{s}_features_v3.parquet").exists()
)

print(f"\n{'='*72}")
print(f"  HOLDOUT: flatboost_v3 vs flatboost_v2 vs ic32_regime_v1")
print(f"  Coins: {len(available)} | Period: Apr 2026 - Jun 2026 (~{PERIOD_MONTHS} bln)")
print(f"  flatboost_v3 : {len(v3_feats)} feat | conf>={FB_CONF_THRESHOLD}")
print(f"  flatboost_v2 : {len(v2_feats)} feat | conf>={FB_CONF_THRESHOLD}")
print(f"  ic32         : {len(ic32_feats)} feat | thr_long={LGBM_THRESHOLD_LONG}")
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


# ── Precompute market_breadth dari holdout data (cross-coin) ─────────────────
print("STAGE 0: Computing market_breadth dari holdout data...")
ret_dict = {}
for sym in available:
    path = HOLDOUT_DIR / "labeled" / f"{sym}_features_v3.parquet"
    df_tmp = pd.read_parquet(path, columns=["log_ret_1"])
    df_tmp = ensure_utc_index(df_tmp)
    ret_dict[sym] = df_tmp["log_ret_1"]

ret_panel = pd.DataFrame(ret_dict)
market_breadth = (ret_panel > 0).sum(axis=1) / ret_panel.notna().sum(axis=1)
market_breadth.name = "market_breadth"
print(f"  market_breadth: {len(market_breadth):,} timestamps, "
      f"range [{market_breadth.min():.3f}, {market_breadth.max():.3f}]\n")


# ── Aggregate accumulators ────────────────────────────────────────────────────
def make_agg():
    return {"trades": 0, "wins": 0, "pnl": 0.0,
            "longs": 0, "long_wins": 0, "shorts": 0, "short_wins": 0,
            "sl_hits": 0, "bars_held": []}

v3_agg   = make_agg()
v2_agg   = make_agg()
ic32_agg = make_agg()


# ── Per-coin loop ─────────────────────────────────────────────────────────────
for sym in available:
    df  = pd.read_parquet(HOLDOUT_DIR / "labeled" / f"{sym}_features_v3.parquet")
    df  = ensure_utc_index(df).sort_index()

    df["market_breadth"] = market_breadth.reindex(df.index).ffill().fillna(0.5)

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

    def build_X(feats):
        X = np.zeros((n, len(feats)), dtype=np.float64)
        for i, c in enumerate(feats):
            if c in df.columns:
                X[:, i] = df[c].ffill().fillna(0).values.astype(np.float64)
            elif c == "hmm_regime_enc":
                X[:, i] = hmm.astype(np.float64)
        return X

    # flatboost_v3
    proba_v3 = v3_model.predict_proba(build_X(v3_feats))
    conf_v3  = proba_v3.max(axis=1)
    yp_v3    = proba_v3.argmax(axis=1).astype(np.int32)
    yp_v3[conf_v3 < FB_CONF_THRESHOLD] = 1

    # flatboost_v2
    proba_v2 = v2_model.predict_proba(build_X(v2_feats))
    conf_v2  = proba_v2.max(axis=1)
    yp_v2    = proba_v2.argmax(axis=1).astype(np.int32)
    yp_v2[conf_v2 < FB_CONF_THRESHOLD] = 1

    # ic32_regime_v1
    proba_ic = ic32_model.predict_proba(build_X(ic32_feats))
    yp_ic    = np.ones(n, dtype=np.int32)
    yp_ic[proba_ic[:, 2] > LGBM_THRESHOLD_LONG]                    = 2
    yp_ic[(proba_ic[:, 0] > LGBM_THRESHOLD_SHORT) & (yp_ic != 2)] = 0

    # Accumulate
    for agg, yp in [(v3_agg, yp_v3), (v2_agg, yp_v2), (ic32_agg, yp_ic)]:
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


# ── Scorecard ─────────────────────────────────────────────────────────────────
def scorecard(name, agg, months):
    t = agg["trades"]; w = agg["wins"]; p = agg["pnl"]
    l = agg["longs"];  lw = agg["long_wins"]
    s = agg["shorts"]; sw = agg["short_wins"]
    sl = agg["sl_hits"]; bh = agg["bars_held"]
    return {
        "name":           name,
        "trades":         t,
        "wr":             w  / max(t, 1) * 100,
        "long_wr":        lw / max(l, 1) * 100,
        "short_wr":       sw / max(s, 1) * 100,
        "long_pct":       l  / max(t, 1) * 100,
        "pnl":            p,
        "pnl_per_month":  p / months,
        "pnl_per_trade":  p / max(t, 1),
        "sl_rate":        sl / max(t, 1) * 100,
        "time_exit_rate": (t - sl) / max(t, 1) * 100,
        "avg_hold":       float(np.mean(bh)) if bh else 0,
    }

v3_sc   = scorecard(f"flatboost_v3 ({len(v3_feats)} feat)", v3_agg,   PERIOD_MONTHS)
v2_sc   = scorecard(f"flatboost_v2 ({len(v2_feats)} feat)", v2_agg,   PERIOD_MONTHS)
ic32_sc = scorecard("ic32_regime_v1 (benchmark)",           ic32_agg, PERIOD_MONTHS)

print(f"\n{'='*80}")
print(f"  SCORECARD - Standalone LGBM, Live-like Eval (SL+time exit, NO TP)")
print(f"  Holdout: Apr-Jun 2026 ({len(available)} koin, ~{PERIOD_MONTHS} bln, $10/5x)")
print(f"{'='*80}")
print(f"\n  {'Metrik':<28} {'flatboost_v3':>16} {'flatboost_v2':>16} {'ic32 (bench)':>16}")
print("  " + "-" * 78)

rows = [
    ("Total Trades",   f"{v3_sc['trades']:,}",                    f"{v2_sc['trades']:,}",                    f"{ic32_sc['trades']:,}"),
    ("Trades/bulan",   f"{v3_sc['trades']/PERIOD_MONTHS:.0f}",    f"{v2_sc['trades']/PERIOD_MONTHS:.0f}",    f"{ic32_sc['trades']/PERIOD_MONTHS:.0f}"),
    ("Win Rate",       f"{v3_sc['wr']:.1f}%",                     f"{v2_sc['wr']:.1f}%",                     f"{ic32_sc['wr']:.1f}%"),
    ("  LONG WR",      f"{v3_sc['long_wr']:.1f}%",                f"{v2_sc['long_wr']:.1f}%",                f"{ic32_sc['long_wr']:.1f}%"),
    ("  SHORT WR",     f"{v3_sc['short_wr']:.1f}%",               f"{v2_sc['short_wr']:.1f}%",               f"{ic32_sc['short_wr']:.1f}%"),
    ("  LONG share",   f"{v3_sc['long_pct']:.1f}%",               f"{v2_sc['long_pct']:.1f}%",               f"{ic32_sc['long_pct']:.1f}%"),
    ("Net PnL",        f"${v3_sc['pnl']:+.0f}",                   f"${v2_sc['pnl']:+.0f}",                   f"${ic32_sc['pnl']:+.0f}"),
    ("PnL/bulan",      f"${v3_sc['pnl_per_month']:+.0f}",         f"${v2_sc['pnl_per_month']:+.0f}",         f"${ic32_sc['pnl_per_month']:+.0f}"),
    ("PnL/trade",      f"${v3_sc['pnl_per_trade']:+.3f}",         f"${v2_sc['pnl_per_trade']:+.3f}",         f"${ic32_sc['pnl_per_trade']:+.3f}"),
    ("SL Hit Rate",    f"{v3_sc['sl_rate']:.1f}%",                f"{v2_sc['sl_rate']:.1f}%",                f"{ic32_sc['sl_rate']:.1f}%"),
    ("Time Exit Rate", f"{v3_sc['time_exit_rate']:.1f}%",         f"{v2_sc['time_exit_rate']:.1f}%",         f"{ic32_sc['time_exit_rate']:.1f}%"),
    ("Avg Hold (bars)",f"{v3_sc['avg_hold']:.1f}",                f"{v2_sc['avg_hold']:.1f}",                f"{ic32_sc['avg_hold']:.1f}"),
]
for label, v3v, v2v, icv in rows:
    marker = " <--" if label == "Net PnL" else ""
    print(f"  {label:<28} {v3v:>16} {v2v:>16} {icv:>16}{marker}")

d_v3_v2   = v3_sc["pnl"] - v2_sc["pnl"]
d_v3_ic32 = v3_sc["pnl"] - ic32_sc["pnl"]
print(f"\n  Delta v3 vs v2  : PnL {d_v3_v2:+.0f}")
print(f"  Delta v3 vs ic32: PnL {d_v3_ic32:+.0f}")
winner = "flatboost_v3" if v3_sc["pnl"] >= v2_sc["pnl"] else "flatboost_v2"
print(f"  -> {winner} lebih baik (standalone, Apr-Jun 2026)")

# ── Save ──────────────────────────────────────────────────────────────────────
run_dir = MODEL_DIR / "runs" / V3_RUN
out = {
    "flatboost_v3":      v3_sc,
    "flatboost_v2":      v2_sc,
    "ic32_benchmark":    ic32_sc,
    "delta_v3_vs_v2":    round(d_v3_v2, 4),
    "delta_v3_vs_ic32":  round(d_v3_ic32, 4),
    "period":            "2026-04-01 to 2026-06-13",
    "n_coins":           len(available),
    "note":              f"Standalone LGBM, no LSTM/Guardian, live-like eval: SL={SL_MULT}xATR + max_hold={MAX_HOLD} bar, NO TP, win=net_pnl>0",
}
out_path = run_dir / "holdout_apr_jun26_vs_v2_ic32.json"
with open(out_path, "w") as f:
    json.dump(out, f, indent=2)
print(f"\n  Saved: {out_path}")
