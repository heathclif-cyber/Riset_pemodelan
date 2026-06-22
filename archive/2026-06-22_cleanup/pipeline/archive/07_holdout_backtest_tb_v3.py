"""
pipeline/07_holdout_backtest_tb_v3.py
Holdout backtest TB v3 (widyawardhana_v3) vs ic32_regime_v1 benchmark.

Kondisi IDENTIK untuk keduanya:
  - Standalone LGBM only (no LSTM, no Guardian)
  - full_trading_report() — swing-based TP/SL dari H4 swing highs/lows
  - Fallback: TP=2.0xATR, SL=1.5xATR jika swing tidak tersedia
  - max_hold=36, $10/trade, 5x leverage
  - 21 koin, Nov 2025 – Apr 2026
"""
import json, sys, warnings, numpy as np, pandas as pd
from pathlib import Path
warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import joblib
from core.evaluator import full_trading_report
from core.utils import setup_logger, ensure_utc_index
from config import *

logger = setup_logger("07_holdout_tb_v3")

TB_RUN   = "tb_lgbm_widyawardhana_v3"
IC32_RUN = "ic32_regime_v1"

# TB v3 regime-aware thresholds (sama dengan training)
REGIME_THRESH = {0: 0.45, 1: 0.50, 2: 0.50, 3: 0.45}
LM = {"SHORT": 0, "FLAT": 1, "LONG": 2}

# ── Load models ────────────────────────────────────────────────────────────────
tb_model = joblib.load(MODEL_DIR / "runs" / TB_RUN / "lgbm.pkl")
with open(MODEL_DIR / "runs" / TB_RUN / f"{TB_RUN}_features.json") as f:
    tb_feats = json.load(f)

ic32_model = joblib.load(MODEL_DIR / "runs" / IC32_RUN / "lgbm.pkl")
ic32_feats = ic32_model.feature_name_

available = [s for s in ALL_COINS if (HOLDOUT_DIR / "labeled" / f"{s}_features_v3.parquet").exists()]
print(f"\n{'='*70}")
print(f"  HOLDOUT: TB v3 vs ic32_regime_v1")
print(f"  Coins: {len(available)} | Period: Nov 2025 - Apr 2026")
print(f"  Simulation: full_trading_report() — swing-based TP/SL")
print(f"  Thresholds: TB v3 regime-aware | ic32 long>=0.69 / short>=0.59")
print(f"{'='*70}\n")

# ── Per-coin results ───────────────────────────────────────────────────────────
tb_agg   = {'trades': 0, 'wins': 0, 'pnl': 0.0, 'longs': 0, 'long_wins': 0, 'sl_hits': 0}
ic32_agg = {'trades': 0, 'wins': 0, 'pnl': 0.0, 'longs': 0, 'long_wins': 0, 'sl_hits': 0}

for sym in available:
    df = pd.read_parquet(HOLDOUT_DIR / "labeled" / f"{sym}_features_v3.parquet")
    df = ensure_utc_index(df).sort_index()

    # HMM regime for TB v3 adaptive threshold
    rp = HOLDOUT_DIR / "labeled" / f"{sym}_regime_h1.parquet"
    hmm = np.full(len(df), 1, np.int32)
    if rp.exists():
        reg = pd.read_parquet(rp)
        if "hmm_regime_enc" in reg.columns:
            hmm = reg["hmm_regime_enc"].reindex(df.index, fill_value=1).values.astype(np.int32)

    # Filter to valid label rows
    mask = df["label"].isin(LM)
    df   = df[mask].copy()
    hmm  = hmm[mask.values]

    close = df["close"].values; high = df["high"].values; low = df["low"].values
    atr   = df["atr_14_h1"].values; n = len(df)
    h4_sh = df["h4_swing_high"].values if "h4_swing_high" in df.columns else np.full(n, np.nan)
    h4_sl = df["h4_swing_low"].values  if "h4_swing_low"  in df.columns else np.full(n, np.nan)
    h4_tr = df["h4_trend"].values      if "h4_trend"      in df.columns else None
    yt    = df["label"].map(LM).values.astype(np.int32)

    # ── TB v3 predictions ──────────────────────────────────────────────────
    X_tb = np.zeros((n, len(tb_feats)), dtype=np.float64)
    for i, c in enumerate(tb_feats):
        if c in df.columns:
            X_tb[:, i] = df[c].ffill().fillna(0).values.astype(np.float64)

    proba_tb = tb_model.predict_proba(X_tb)
    conf_tb  = np.max(proba_tb, axis=1)
    yp_tb    = np.argmax(proba_tb, axis=1)
    # Regime-aware confidence filter
    for r, th in REGIME_THRESH.items():
        yp_tb[(hmm == r) & (yp_tb != 1) & (conf_tb < th)] = 1

    rep_tb = full_trading_report(
        y_pred=yp_tb, y_actual=yt, atr=atr, close=close, high=high, low=low,
        h4_swing_highs=h4_sh, h4_swing_lows=h4_sl,
        index=df.index, modal=MODAL_PER_TRADE, leverages=LEVERAGE_SIM,
        fee_per_side=FEE_PER_SIDE, slippage=SLIPPAGE_PER_SIDE,
        min_rr=SWING_LABEL_MIN_RR, min_tp_atr=SWING_LABEL_MIN_TP,
        max_sl_atr=SWING_LABEL_MAX_SL,
        tp_fallback_atr=TP_SL_FALLBACK_TP, sl_fallback_atr=TP_SL_FALLBACK_SL,
        max_hold=MAX_HOLDING_BARS, symbol=sym, confidence=conf_tb,
        guardian_enabled=False, trailing_stop_enabled=False, h4_trend=h4_tr,
    )

    # ── ic32 predictions ───────────────────────────────────────────────────
    X_ic = np.zeros((n, len(ic32_feats)), dtype=np.float64)
    for i, c in enumerate(ic32_feats):
        if c in df.columns:
            X_ic[:, i] = df[c].ffill().fillna(0).values.astype(np.float64)
        elif c == "hmm_regime_enc":
            X_ic[:, i] = hmm.astype(np.float64)  # from regime parquet

    proba_ic = ic32_model.predict_proba(X_ic)
    yp_ic    = np.full(n, 1, np.int32)
    yp_ic[proba_ic[:, 2] > LGBM_THRESHOLD_LONG]  = 2
    short_m = (proba_ic[:, 0] > LGBM_THRESHOLD_SHORT) & (yp_ic != 2)
    yp_ic[short_m] = 0
    conf_ic = np.max(proba_ic, axis=1)

    rep_ic = full_trading_report(
        y_pred=yp_ic, y_actual=yt, atr=atr, close=close, high=high, low=low,
        h4_swing_highs=h4_sh, h4_swing_lows=h4_sl,
        index=df.index, modal=MODAL_PER_TRADE, leverages=LEVERAGE_SIM,
        fee_per_side=FEE_PER_SIDE, slippage=SLIPPAGE_PER_SIDE,
        min_rr=SWING_LABEL_MIN_RR, min_tp_atr=SWING_LABEL_MIN_TP,
        max_sl_atr=SWING_LABEL_MAX_SL,
        tp_fallback_atr=TP_SL_FALLBACK_TP, sl_fallback_atr=TP_SL_FALLBACK_SL,
        max_hold=MAX_HOLDING_BARS, symbol=sym, confidence=conf_ic,
        guardian_enabled=False, trailing_stop_enabled=False, h4_trend=h4_tr,
    )

    # Accumulate
    for agg, rep in [(tb_agg, rep_tb), (ic32_agg, rep_ic)]:
        lev = rep.get("lev5x", rep)
        tr  = lev.get("trades", [])
        agg["trades"] += len(tr)
        agg["wins"]   += sum(1 for t in tr if "WIN" in str(t.get("outcome", "")))
        agg["pnl"]    += sum(t.get("net_pnl", 0) for t in tr)
        agg["longs"]  += sum(1 for t in tr if t.get("direction") == "LONG")
        agg["long_wins"] += sum(1 for t in tr if t.get("direction") == "LONG" and "WIN" in str(t.get("outcome", "")))
        agg["sl_hits"] += sum(1 for t in tr if t.get("exit_reason") == "sl" or "SL" in str(t.get("outcome", "")))


def scorecard(name, agg):
    t  = agg["trades"]; w = agg["wins"]; p = agg["pnl"]
    lw = agg["long_wins"]; l = agg["longs"]; sl = agg["sl_hits"]
    wr = w / max(t, 1) * 100
    lwr = lw / max(l, 1) * 100
    swr = (w - lw) / max(t - l, 1) * 100
    sl_r = sl / max(t, 1) * 100
    pt   = p / max(t, 1)
    return {
        "name": name, "trades": t, "wr": wr,
        "long_wr": lwr, "short_wr": swr,
        "long_pct": l / max(t, 1) * 100,
        "pnl": p, "pnl_per_trade": pt,
        "sl_rate": sl_r,
        "pnl_per_month": p / 5,
    }

tb_sc   = scorecard("TB v3 widyawardhana_v3", tb_agg)
ic32_sc = scorecard("ic32_regime_v1 (benchmark)", ic32_agg)

# ── Print scorecard ────────────────────────────────────────────────────────────
print(f"\n{'='*70}")
print(f"  SCORECARD — Standalone LGBM Only, No LSTM, No Guardian")
print(f"  Holdout: Nov 2025 – Apr 2026 | {len(available)} coins | $10/trade 5x")
print(f"{'='*70}")
print(f"\n{'Metrik':<30} {'TB v3':>18} {'ic32 (bench)':>18}")
print("-" * 70)

rows = [
    ("Total Trades",      f"{tb_sc['trades']:,}",           f"{ic32_sc['trades']:,}"),
    ("Trades/bulan",      f"{tb_sc['trades']/5:.0f}",       f"{ic32_sc['trades']/5:.0f}"),
    ("Win Rate",          f"{tb_sc['wr']:.1f}%",            f"{ic32_sc['wr']:.1f}%"),
    ("  LONG WR",         f"{tb_sc['long_wr']:.1f}%",       f"{ic32_sc['long_wr']:.1f}%"),
    ("  SHORT WR",        f"{tb_sc['short_wr']:.1f}%",      f"{ic32_sc['short_wr']:.1f}%"),
    ("  LONG share",      f"{tb_sc['long_pct']:.1f}%",      f"{ic32_sc['long_pct']:.1f}%"),
    ("Net PnL (5 bln)",   f"${tb_sc['pnl']:+.0f}",          f"${ic32_sc['pnl']:+.0f}"),
    ("PnL/bulan",         f"${tb_sc['pnl_per_month']:+.0f}",f"${ic32_sc['pnl_per_month']:+.0f}"),
    ("PnL/trade",         f"${tb_sc['pnl_per_trade']:+.2f}",f"${ic32_sc['pnl_per_trade']:+.2f}"),
    ("SL Hit Rate",       f"{tb_sc['sl_rate']:.1f}%",       f"{ic32_sc['sl_rate']:.1f}%"),
]
for label, v_tb, v_ic in rows:
    marker = " <--" if label == "Net PnL (5 bln)" else ""
    print(f"  {label:<28} {v_tb:>18} {v_ic:>18}{marker}")

delta = tb_sc["pnl"] - ic32_sc["pnl"]
print(f"\n  Delta TB vs ic32: PnL {delta:+.0f} ({delta/max(abs(ic32_sc['pnl']),1)*100:+.1f}%)")
winner = "TB v3" if delta > 0 else "ic32"
print(f"  -> {winner} lebih baik dalam kondisi apples-to-apples")

# Save
out = {
    "tb_v3": tb_sc,
    "ic32_benchmark": ic32_sc,
    "delta_pnl": delta,
    "note": "Standalone LGBM only, no LSTM, no Guardian, swing-based TP/SL"
}
out_path = MODEL_DIR / "runs" / TB_RUN / "holdout_v3_vs_ic32.json"
with open(out_path, "w") as f:
    json.dump(out, f, indent=2)
print(f"\n  Saved -> {out_path}")
print(f"{'='*70}")
