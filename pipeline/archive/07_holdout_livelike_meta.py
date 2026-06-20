"""
pipeline/07_holdout_livelike_meta.py — Holdout comparison: tb bare vs tb+meta.

Meta-labeling Layer 2: binary LGBM yang filter trade mana yang layak diambil.
Entry hanya kalau p_win (meta-model) >= threshold.

Model:
  1. tb_widyawardhana_v3 bare
  2. tb_widyawardhana_v3 + meta_lgbm (threshold sweep)

Exit: SL=1.5xATR hard stop | max_hold=36 bar | NO TP
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

logger = setup_logger("07_meta_holdout")

SL_MULT  = TP_SL_FALLBACK_SL
MAX_HOLD = MAX_HOLDING_BARS
MODAL    = MODAL_PER_TRADE
LEVERAGE = LEVERAGE_SIM[0] if isinstance(LEVERAGE_SIM, list) else LEVERAGE_SIM
COST_RT  = (FEE_PER_SIDE + SLIPPAGE_PER_SIDE) * 2

REGIME_THRESH = {0: 0.45, 1: 0.50, 2: 0.50, 3: 0.45}

# Meta-model threshold sweep
META_THRESHOLDS = [0.45, 0.50, 0.55, 0.60, 0.65]

META_CONTEXT = [
    "atr_percentile_h1", "funding_rate", "wyckoff_phase",
    "stochrsi_k", "ofi_h4_delta", "Sell_Liq",
]

# Load models
tb_model = joblib.load(MODEL_DIR / "runs" / "tb_lgbm_widyawardhana_v3" / "lgbm.pkl")
with open(MODEL_DIR / "runs" / "tb_lgbm_widyawardhana_v3" /
          "tb_lgbm_widyawardhana_v3_features.json") as f:
    tb_feats = json.load(f)

meta_model_path = MODEL_DIR / "runs" / "tb_meta_v1" / "meta_lgbm.pkl"
with open(MODEL_DIR / "runs" / "tb_meta_v1" / "tb_meta_v1_features.json") as f:
    meta_feats = json.load(f)
meta_model = joblib.load(meta_model_path)

available = [s for s in ALL_COINS
             if (HOLDOUT_DIR / "labeled" / f"{s}_features_v3.parquet").exists()]

print(f"\n{'='*65}")
print(f"  HOLDOUT OOS — Meta-Labeling Evaluation")
print(f"  Base  : tb_widyawardhana_v3")
print(f"  Meta  : tb_meta_v1 (threshold sweep: {META_THRESHOLDS})")
print(f"  Exit  : SL={SL_MULT}xATR hard stop | max_hold={MAX_HOLD} bar | NO TP")
print(f"  Coins : {len(available)} | Nov 2025 - Apr 2026 | ${MODAL}/trade {LEVERAGE}x")
print(f"{'='*65}\n")


def simulate_livelike(yp, close, high, low, atr):
    n = len(yp); trades = []; i = 0
    while i < n:
        sig = yp[i]
        if sig == 1:
            i += 1; continue
        direction = 1 if sig == 2 else -1
        entry     = close[i]
        sl_price  = entry - direction * SL_MULT * atr[i]
        exit_price = close[min(i + MAX_HOLD, n - 1)]
        exit_bar   = min(i + MAX_HOLD, n - 1)
        outcome    = "TIME"
        for j in range(i + 1, min(i + MAX_HOLD + 1, n)):
            if direction == 1 and low[j] <= sl_price:
                exit_price = sl_price; exit_bar = j; outcome = "SL"; break
            elif direction == -1 and high[j] >= sl_price:
                exit_price = sl_price; exit_bar = j; outcome = "SL"; break
        ret = (exit_price - entry) / entry * direction
        net_pnl = ret * MODAL * LEVERAGE - COST_RT * MODAL * LEVERAGE
        trades.append({
            "direction": "LONG" if direction == 1 else "SHORT",
            "outcome": outcome, "net_pnl": net_pnl,
            "bars_held": exit_bar - i,
        })
        i = exit_bar + 1
    return trades


def new_agg():
    return {"trades": 0, "wins": 0, "pnl": 0.0,
            "longs": 0, "long_wins": 0, "sl_hits": 0, "bars_held": []}


def update_agg(agg, trades):
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


# Aggregators: bare + one per meta threshold
tb_agg   = new_agg()
meta_aggs = {thr: new_agg() for thr in META_THRESHOLDS}

for sym in available:
    df = pd.read_parquet(HOLDOUT_DIR / "labeled" / f"{sym}_features_v3.parquet")
    df = ensure_utc_index(df).sort_index()

    rp  = HOLDOUT_DIR / "labeled" / f"{sym}_regime_h1.parquet"
    hmm = np.full(len(df), 1, np.int32)
    if rp.exists():
        reg = pd.read_parquet(rp)
        if "hmm_regime_enc" in reg.columns:
            hmm = reg["hmm_regime_enc"].reindex(df.index, fill_value=1).values.astype(np.int32)

    mask = df["label"].isin({"SHORT": 0, "FLAT": 1, "LONG": 2})
    df   = df[mask].copy()
    hmm  = hmm[mask.values]
    n    = len(df)

    close = df["close"].values.astype(np.float64)
    high  = df["high"].values.astype(np.float64)
    low   = df["low"].values.astype(np.float64)
    atr   = df["atr_14_h1"].values.astype(np.float64)

    # TB LGBM predictions
    X_tb = np.zeros((n, len(tb_feats)), dtype=np.float64)
    for idx, c in enumerate(tb_feats):
        if c in df.columns:
            X_tb[:, idx] = df[c].ffill().fillna(0).values.astype(np.float64)

    p_tb    = tb_model.predict_proba(X_tb).astype(np.float32)
    conf_tb = np.max(p_tb, axis=1)
    yp_tb   = np.argmax(p_tb, axis=1).astype(np.int32)
    for r, th in REGIME_THRESH.items():
        yp_tb[(hmm == r) & (yp_tb != 1) & (conf_tb < th)] = 1

    # tb bare
    update_agg(tb_agg, simulate_livelike(yp_tb, close, high, low, atr))

    # Meta-features for each bar
    meta_X_list = []
    for feat in meta_feats:
        if feat == "p_short":
            meta_X_list.append(p_tb[:, 0])
        elif feat == "p_flat":
            meta_X_list.append(p_tb[:, 1])
        elif feat == "p_long":
            meta_X_list.append(p_tb[:, 2])
        elif feat == "confidence":
            meta_X_list.append(conf_tb)
        elif feat == "direction":
            meta_X_list.append((yp_tb == 2).astype(np.float32))
        elif feat in df.columns:
            meta_X_list.append(df[feat].ffill().fillna(0).values.astype(np.float32))
        else:
            meta_X_list.append(np.zeros(n, dtype=np.float32))

    X_meta = np.stack(meta_X_list, axis=1)
    p_win  = meta_model.predict_proba(X_meta)[:, 1]

    # tb+meta for each threshold
    for thr in META_THRESHOLDS:
        yp_meta = yp_tb.copy()
        # Filter: skip directional signals where meta-model not confident
        yp_meta[(yp_meta != 1) & (p_win < thr)] = 1
        update_agg(meta_aggs[thr], simulate_livelike(yp_meta, close, high, low, atr))

    logger.info(f"[{sym}] tb={tb_agg['trades']} trades so far")


def sc(agg):
    t = agg["trades"]; w = agg["wins"]; p = agg["pnl"]
    l = agg["longs"];  lw = agg["long_wins"]; sl = agg["sl_hits"]
    bh = agg["bars_held"]
    return {
        "trades": t, "wr": w / max(t, 1) * 100,
        "long_wr": lw / max(l, 1) * 100,
        "short_wr": (w - lw) / max(t - l, 1) * 100,
        "long_pct": l / max(t, 1) * 100,
        "sl_rate": sl / max(t, 1) * 100,
        "avg_hold": float(np.mean(bh)) if bh else 0,
        "pnl": p, "ppm": p / 5, "ppt": p / max(t, 1),
    }


s_bare = sc(tb_agg)

print(f"\n{'='*65}")
print(f"  SCORECARD — Meta-Labeling | Nov 2025 – Apr 2026")
print(f"{'='*65}")
print(f"  {'Metrik':<22} {'tb_bare':>12} " +
      " ".join(f"{'meta'+str(t):>12}" for t in META_THRESHOLDS))
print(f"  {'-'*62}")

meta_scores = {thr: sc(meta_aggs[thr]) for thr in META_THRESHOLDS}

rows = [
    ("Total Trades",   f"{s_bare['trades']:,}",     [f"{meta_scores[t]['trades']:,}" for t in META_THRESHOLDS]),
    ("Win Rate",       f"{s_bare['wr']:.1f}%",      [f"{meta_scores[t]['wr']:.1f}%" for t in META_THRESHOLDS]),
    ("  LONG WR",      f"{s_bare['long_wr']:.1f}%", [f"{meta_scores[t]['long_wr']:.1f}%" for t in META_THRESHOLDS]),
    ("  SHORT WR",     f"{s_bare['short_wr']:.1f}%",[f"{meta_scores[t]['short_wr']:.1f}%" for t in META_THRESHOLDS]),
    ("Net PnL",        f"${s_bare['pnl']:.0f}",     [f"${meta_scores[t]['pnl']:.0f}" for t in META_THRESHOLDS]),
    ("PnL/bulan",      f"${s_bare['ppm']:.0f}",     [f"${meta_scores[t]['ppm']:.0f}" for t in META_THRESHOLDS]),
    ("PnL/trade",      f"${s_bare['ppt']:.2f}",     [f"${meta_scores[t]['ppt']:.2f}" for t in META_THRESHOLDS]),
    ("SL hit rate",    f"{s_bare['sl_rate']:.1f}%", [f"{meta_scores[t]['sl_rate']:.1f}%" for t in META_THRESHOLDS]),
    ("Avg Hold Bars",  f"{s_bare['avg_hold']:.1f}", [f"{meta_scores[t]['avg_hold']:.1f}" for t in META_THRESHOLDS]),
]

for label, v_bare, v_meta_list in rows:
    row = f"  {label:<22} {v_bare:>12} " + " ".join(f"{v:>12}" for v in v_meta_list)
    print(row)

print(f"{'='*65}\n")

# Save results
results = {
    "tb_bare": s_bare,
    "meta": {str(thr): meta_scores[thr] for thr in META_THRESHOLDS},
    "holdout_period": "Nov 2025 – Apr 2026",
    "n_coins": len(available),
}
out_path = MODEL_DIR / "runs" / "tb_meta_v1" / "holdout_meta_results.json"
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"  Results saved: {out_path}")
