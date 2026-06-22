"""
pipeline/07_holdout_guardian_momentum_compare.py
Holdout test: Guardian profit_v1 vs momentum_v1

Periode : Apr 1 2026 - Jun 13 2026 (~2.5 bulan)
Entry   : flatboost_v2 LGBM (27 feat, thr=0.50/0.55 adaptive HMM T50_R55)
Exit    : Guardian profit_v1  — 18+7=25 feat (baseline)
          Guardian momentum_v1 — 22+7=29 feat (+4 momentum change)
Eval    : live-like — SL=1.5xATR, max_hold=36 bar, no TP, Guardian or time exit

Usage   : python pipeline/07_holdout_guardian_momentum_compare.py
"""
import json, sys, warnings
from pathlib import Path

import joblib, numpy as np, pandas as pd
warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from core.evaluator import full_trading_report
from core.utils import setup_logger, ensure_utc_index
from pipeline.backtest_utils import compute_guardian_static_array
from config import (
    ALL_COINS, HOLDOUT_DIR, MODEL_DIR, LABEL_MAP,
    MODAL_PER_TRADE, LEVERAGE_SIM, FEE_PER_SIDE, SLIPPAGE_PER_SIDE,
    MAX_HOLDING_BARS,
    SWING_LABEL_MIN_RR, SWING_LABEL_MIN_TP, SWING_LABEL_MAX_SL,
    TP_SL_FALLBACK_TP, TP_SL_FALLBACK_SL,
    GUARDIAN_EXIT_THRESHOLD, GUARDIAN_MIN_HOLD_BARS,
    GUARDIAN_DYNAMIC_FEATURES as _GDYN,
)

logger = setup_logger("07_holdout_guardian_momentum_compare")

# ── Config ─────────────────────────────────────────────────────────────────────
FB_RUN        = "tb_lgbm_flatboost_v2"
GD_PROFIT_RUN = "tb_guardian_profit_v1"
GD_MOM_RUN    = "tb_guardian_momentum_v1"

THR_TRENDING_LONG  = 0.50
THR_TRENDING_SHORT = 0.55
THR_RANGING_LONG   = 0.55
THR_RANGING_SHORT  = 0.60
TRENDING_STATES    = {0, 3}
GD_EXIT_THR        = GUARDIAN_EXIT_THRESHOLD  # 0.70
GD_MIN_HOLD        = GUARDIAN_MIN_HOLD_BARS   # 3

PERIOD_LABEL = "Apr 2026-Jun 2026 (~2.5 bln)"
MONTHS       = 2.5
LM = LABEL_MAP if isinstance(LABEL_MAP, dict) else {"SHORT": 0, "FLAT": 1, "LONG": 2}

# ── Derived features for momentum_v1 ──────────────────────────────────────────
def add_momentum_feats(df):
    df = df.copy()
    df["cvd_slope_h4_delta"] = df["cvd_slope_h4"].diff(1)     if "cvd_slope_h4" in df.columns else 0.0
    df["ofi_h4_accel"]       = df["ofi_h4_delta"].diff(2)     if "ofi_h4_delta"  in df.columns else 0.0
    df["rsi_h4_slope"]       = df["rsi_h4"].diff(2)           if "rsi_h4"        in df.columns else 0.0
    if "dist_liq_50x_long" not in df.columns:
        df["dist_liq_50x_long"] = 0.0
    return df


# ── Load entry model ───────────────────────────────────────────────────────────
fb_model = joblib.load(MODEL_DIR / "runs" / FB_RUN / "lgbm.pkl")
with open(MODEL_DIR / "runs" / FB_RUN / f"{FB_RUN}_features.json") as f:
    fb_feats = json.load(f)
logger.info(f"Entry: flatboost_v2 ({len(fb_feats)} feat)")

# ── Load Guardian profit_v1 ────────────────────────────────────────────────────
gd_profit_model  = joblib.load(MODEL_DIR / "runs" / GD_PROFIT_RUN / "guardian.pkl")
gd_profit_scaler = joblib.load(MODEL_DIR / "runs" / GD_PROFIT_RUN / "guardian_scaler.pkl")
with open(MODEL_DIR / "runs" / GD_PROFIT_RUN / "guardian_feature_cols.json") as f:
    gd_profit_feats = json.load(f)
gd_profit_static = [c for c in gd_profit_feats if c not in set(_GDYN)]
logger.info(f"Guardian profit_v1: {len(gd_profit_static)} static + {len(_GDYN)} dynamic")

# ── Load Guardian momentum_v1 (if exists) ─────────────────────────────────────
gd_mom_dir   = MODEL_DIR / "runs" / GD_MOM_RUN
gd_mom_ok    = (gd_mom_dir / "guardian.pkl").exists()
gd_mom_model = gd_mom_scaler = gd_mom_static = None

if gd_mom_ok:
    gd_mom_model  = joblib.load(gd_mom_dir / "guardian.pkl")
    gd_mom_scaler = joblib.load(gd_mom_dir / "guardian_scaler.pkl")
    with open(gd_mom_dir / "guardian_feature_cols.json") as f:
        gd_mom_feats = json.load(f)
    gd_mom_static = [c for c in gd_mom_feats if c not in set(_GDYN)]
    logger.info(f"Guardian momentum_v1: {len(gd_mom_static)} static + {len(_GDYN)} dynamic")
else:
    logger.warning("momentum_v1 not found — run 06o_train_guardian_momentum_v1.py first")

# ── Holdout coins ─────────────────────────────────────────────────────────────
available = [
    s for s in ALL_COINS
    if (HOLDOUT_DIR / "labeled" / f"{s}_features_v3.parquet").exists()
]
logger.info(f"Coins: {len(available)}")

# ── Aggregators ────────────────────────────────────────────────────────────────
def new_agg():
    return {"trades": 0, "wins": 0, "pnl": 0.0, "longs": 0, "long_wins": 0,
            "sl": 0, "gd_exits": 0, "gross_win": 0.0, "gross_loss": 0.0}

agg_base    = new_agg()
agg_profit  = new_agg()
agg_mom     = new_agg() if gd_mom_ok else None

# ── Per-coin loop ──────────────────────────────────────────────────────────────
for sym in available:
    path = HOLDOUT_DIR / "labeled" / f"{sym}_features_v3.parquet"
    df   = ensure_utc_index(pd.read_parquet(path)).sort_index()

    reg_path = HOLDOUT_DIR / "labeled" / f"{sym}_regime_h1.parquet"
    hmm = np.full(len(df), 1, np.int32)
    if reg_path.exists():
        try:
            reg = pd.read_parquet(reg_path)
            if "hmm_regime_enc" in reg.columns:
                hmm = reg["hmm_regime_enc"].reindex(df.index, fill_value=1).values.astype(np.int32)
        except Exception:
            pass

    df_mom = add_momentum_feats(df)

    mask = df["label"].astype(str).isin(LM)
    df     = df[mask].copy()
    df_mom = df_mom[mask].copy()
    hmm    = hmm[mask.values]

    n = len(df)
    if n < 50:
        continue

    close = df["close"].values
    high  = df["high"].values
    low   = df["low"].values
    atr   = df["atr_14_h1"].values
    h4_sh = df["h4_swing_high"].values if "h4_swing_high" in df.columns else np.full(n, np.nan)
    h4_sl = df["h4_swing_low"].values  if "h4_swing_low"  in df.columns else np.full(n, np.nan)
    h4t   = df["h4_trend"].values      if "h4_trend"      in df.columns else None
    y_act = np.array([LM.get(str(v), 1) for v in df["label"].values], dtype=np.int32)

    # Entry predictions
    X_fb = np.zeros((n, len(fb_feats)), dtype=np.float64)
    for i, c in enumerate(fb_feats):
        if c in df.columns:
            X_fb[:, i] = df[c].ffill().fillna(0).values.astype(np.float64)
    proba_fb = fb_model.predict_proba(X_fb)

    yp = np.full(n, 1, np.int32)
    for i in range(n):
        is_trend  = hmm[i] in TRENDING_STATES
        tl = THR_TRENDING_LONG  if is_trend else THR_RANGING_LONG
        ts = THR_TRENDING_SHORT if is_trend else THR_RANGING_SHORT
        if proba_fb[i, 2] >= tl:
            yp[i] = 2
        elif proba_fb[i, 0] >= ts:
            yp[i] = 0

    conf    = proba_fb.max(axis=1)

    # Guardian static arrays
    X_gd_profit = compute_guardian_static_array(df, gd_profit_static)
    X_gd_mom    = compute_guardian_static_array(df_mom, gd_mom_static) if gd_mom_ok else None

    base_kw = dict(
        y_pred=yp, y_actual=y_act, atr=atr, close=close, high=high, low=low,
        h4_swing_highs=h4_sh, h4_swing_lows=h4_sl,
        index=df.index, modal=MODAL_PER_TRADE, leverages=LEVERAGE_SIM,
        fee_per_side=FEE_PER_SIDE, slippage=SLIPPAGE_PER_SIDE,
        min_rr=SWING_LABEL_MIN_RR, min_tp_atr=SWING_LABEL_MIN_TP,
        max_sl_atr=SWING_LABEL_MAX_SL,
        tp_fallback_atr=TP_SL_FALLBACK_TP, sl_fallback_atr=TP_SL_FALLBACK_SL,
        max_hold=MAX_HOLDING_BARS, symbol=sym, confidence=conf,
        h4_trend=h4t, trailing_stop_enabled=False,
    )

    # 1. Baseline (no Guardian)
    rep_base   = full_trading_report(**base_kw, guardian_enabled=False)

    # 2. profit_v1
    rep_profit = full_trading_report(
        **base_kw,
        guardian_enabled=True,
        guardian_model=gd_profit_model,
        guardian_scaler=gd_profit_scaler,
        X_guardian=X_gd_profit,
        guardian_exit_threshold=GD_EXIT_THR,
        guardian_min_hold_bars=GD_MIN_HOLD,
    )

    # 3. momentum_v1
    rep_mom = None
    if gd_mom_ok:
        rep_mom = full_trading_report(
            **base_kw,
            guardian_enabled=True,
            guardian_model=gd_mom_model,
            guardian_scaler=gd_mom_scaler,
            X_guardian=X_gd_mom,
            guardian_exit_threshold=GD_EXIT_THR,
            guardian_min_hold_bars=GD_MIN_HOLD,
        )

    def extract_agg(rep):
        lev = rep.get("lev5x", rep)
        trades = lev.get("trades", [])
        return {
            "trades": len(trades),
            "wins":   sum(1 for t in trades if t.get("net_pnl", 0) > 0),
            "pnl":    sum(t.get("net_pnl", 0) for t in trades),
            "longs":  sum(1 for t in trades if t.get("direction") == "LONG"),
            "long_wins": sum(1 for t in trades if t.get("direction") == "LONG" and t.get("net_pnl", 0) > 0),
            "sl":     sum(1 for t in trades if t.get("exit_reason") == "sl" or "SL" in str(t.get("outcome", ""))),
            "gd_exits": sum(1 for t in trades if "GUARDIAN" in str(t.get("outcome", ""))),
            "gross_win":  sum(t.get("net_pnl", 0) for t in trades if t.get("net_pnl", 0) > 0),
            "gross_loss": sum(t.get("net_pnl", 0) for t in trades if t.get("net_pnl", 0) < 0),
        }

    def merge(dst, src):
        for k, v in src.items():
            dst[k] += v

    merge(agg_base,   extract_agg(rep_base))
    merge(agg_profit, extract_agg(rep_profit))
    if rep_mom is not None:
        merge(agg_mom, extract_agg(rep_mom))

    n_base   = extract_agg(rep_base)["trades"]
    n_profit = extract_agg(rep_profit)["trades"]
    n_mom    = extract_agg(rep_mom)["trades"] if rep_mom else "-"
    logger.info(f"  [{sym}] base={n_base} profit={n_profit} mom={n_mom}")


# ── Scorecard builder ──────────────────────────────────────────────────────────
def scorecard(name, agg):
    t    = max(agg["trades"], 1)
    wr   = agg["wins"] / t * 100
    lwr  = agg["long_wins"] / max(agg["longs"], 1) * 100
    swr  = (agg["wins"] - agg["long_wins"]) / max(t - agg["longs"], 1) * 100
    pf   = abs(agg["gross_win"] / agg["gross_loss"]) if agg["gross_loss"] != 0 else float("inf")
    ppm  = agg["pnl"] / MONTHS
    return {
        "name": name,
        "trades": agg["trades"], "trades_per_month": round(agg["trades"] / MONTHS, 0),
        "win_rate": round(wr, 1), "long_wr": round(lwr, 1), "short_wr": round(swr, 1),
        "long_pct": round(agg["longs"] / max(agg["trades"], 1) * 100, 1),
        "pnl": round(agg["pnl"], 2), "pnl_per_month": round(ppm, 2),
        "pnl_per_trade": round(agg["pnl"] / max(agg["trades"], 1), 3),
        "profit_factor": round(pf, 2),
        "sl_rate": round(agg["sl"] / max(agg["trades"], 1) * 100, 1),
        "gd_exit_rate": round(agg["gd_exits"] / max(agg["trades"], 1) * 100, 1),
    }

sc_base   = scorecard("FB v2 Standalone",       agg_base)
sc_profit = scorecard("FB v2 + Guardian profit_v1", agg_profit)
sc_mom    = scorecard("FB v2 + Guardian momentum_v1", agg_mom) if gd_mom_ok else None

# ── Print ──────────────────────────────────────────────────────────────────────
variants = [sc_base, sc_profit]
if sc_mom:
    variants.append(sc_mom)

col_w = 22
hdr   = f"{'Metrik':<30}" + "".join(f"{v['name']:>{col_w}}" for v in variants)
sep   = "-" * (30 + col_w * len(variants))

print(f"\n{'='*80}")
print(f"  HOLDOUT — Guardian momentum_v1 vs profit_v1 vs Standalone")
print(f"  Periode : {PERIOD_LABEL}")
print(f"  Koin    : {len(available)} | Modal $10/trade 5x")
print(f"{'='*80}")
print(hdr)
print(sep)

rows = [
    ("Total Trades",    "trades",           lambda x: f"{x:,}"),
    ("Trades/bulan",    "trades_per_month", lambda x: f"{x:.0f}"),
    ("Win Rate",        "win_rate",         lambda x: f"{x:.1f}%"),
    ("  LONG WR",       "long_wr",          lambda x: f"{x:.1f}%"),
    ("  SHORT WR",      "short_wr",         lambda x: f"{x:.1f}%"),
    ("  LONG share",    "long_pct",         lambda x: f"{x:.1f}%"),
    ("Net PnL",         "pnl",              lambda x: f"${x:+.0f}"),
    ("PnL/bulan",       "pnl_per_month",    lambda x: f"${x:+.0f}"),
    ("PnL/trade",       "pnl_per_trade",    lambda x: f"${x:+.3f}"),
    ("Profit Factor",   "profit_factor",    lambda x: f"{x:.2f}"),
    ("SL Hit Rate",     "sl_rate",          lambda x: f"{x:.1f}%"),
    ("Guardian exits",  "gd_exit_rate",     lambda x: f"{x:.1f}%" if x else "—"),
]

for label, key, fmt in rows:
    row = f"  {label:<28}"
    for sc in variants:
        val = sc.get(key, 0)
        row += f"{fmt(val):>{col_w}}"
    print(row)

print(sep)
if sc_mom:
    delta_profit = sc_profit["pnl"] - sc_base["pnl"]
    delta_mom    = sc_mom["pnl"]    - sc_base["pnl"]
    delta_vs_p   = sc_mom["pnl"]    - sc_profit["pnl"]
    print(f"\n  Guardian profit_v1  vs Standalone : {delta_profit:+.0f} USD")
    print(f"  Guardian momentum_v1 vs Standalone : {delta_mom:+.0f} USD")
    print(f"  momentum_v1 vs profit_v1           : {delta_vs_p:+.0f} USD")
    winner = "momentum_v1" if sc_mom["pnl"] > sc_profit["pnl"] else "profit_v1"
    if abs(delta_vs_p) < 10:
        winner = "DRAW (delta < $10)"
    print(f"  --> {winner}")

# ── Save results ───────────────────────────────────────────────────────────────
out_dir = MODEL_DIR / "runs" / GD_MOM_RUN
out_dir.mkdir(parents=True, exist_ok=True)

results = {
    "period": PERIOD_LABEL,
    "coins":  len(available),
    "variants": [sc_base, sc_profit] + ([sc_mom] if sc_mom else []),
}
with open(out_dir / "holdout_guardian_compare.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved -> {out_dir / 'holdout_guardian_compare.json'}")

# Also save summary to profit_v1 dir for cross-reference
profit_dir = MODEL_DIR / "runs" / GD_PROFIT_RUN
with open(profit_dir / "holdout_guardian_compare.json", "w") as f:
    json.dump(results, f, indent=2)
