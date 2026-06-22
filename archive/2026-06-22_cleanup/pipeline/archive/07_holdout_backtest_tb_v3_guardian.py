"""
pipeline/07_holdout_backtest_tb_v3_guardian.py
Holdout backtest TB v3 + Guardian Delta vs TB v3 standalone (benchmark).

Kondisi identik:
  - full_trading_report() — swing-based TP/SL (H4 swing highs/lows, ATR fallback)
  - $10/trade, 5x leverage, 21 koin, Nov 2025 – Apr 2026
  - Guardian Delta: mom_thresh=0.45, def_thresh=0.70, def_min_loss=-0.010
    (best dari holdout_guardian_sweep.py)
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

logger = setup_logger("07_tb_v3_guardian")

TB_RUN  = "tb_lgbm_widyawardhana_v3"
GD_RUN  = "tb_guardian_delta_v1"

# Best thresholds from holdout_guardian_sweep.py
GD_MOM_THRESH    = 0.45
GD_DEF_THRESH    = 0.70
GD_DEF_MIN_LOSS  = -0.010

REGIME_THRESH = {0: 0.45, 1: 0.50, 2: 0.50, 3: 0.45}
LM = {"SHORT": 0, "FLAT": 1, "LONG": 2}

BASE_FEATURES = [
    "ofi_z_score", "cvd_momentum_adv", "cvd_slope_h4", "ofi_h4_delta",
    "absorption_z", "price_accel_1h",
    "rsi_6", "trend_strength", "ema_21_slope_h4",
    "vol_ratio_20", "vol_spike_zscore", "atr_percentile_h1",
    "funding_rate", "h4_trend",
]

# ── Load models ────────────────────────────────────────────────────────────────
tb_model = joblib.load(MODEL_DIR / "runs" / TB_RUN / "lgbm.pkl")
with open(MODEL_DIR / "runs" / TB_RUN / f"{TB_RUN}_features.json") as f:
    tb_feats = json.load(f)

gd_model = joblib.load(MODEL_DIR / "runs" / GD_RUN / "guardian.pkl")
with open(MODEL_DIR / "runs" / GD_RUN / f"{GD_RUN}_features.json") as f:
    gd_feats = json.load(f)

available = [s for s in ALL_COINS
             if (HOLDOUT_DIR / "labeled" / f"{s}_features_v3.parquet").exists()]

print(f"\n{'='*70}")
print(f"  HOLDOUT: TB v3 + Guardian Delta vs TB v3 standalone")
print(f"  Coins: {len(available)} | Period: Nov 2025 - Apr 2026")
print(f"  Simulation: full_trading_report() — swing-based TP/SL")
print(f"  Guardian Delta: mom={GD_MOM_THRESH} def={GD_DEF_THRESH} loss>{GD_DEF_MIN_LOSS}")
print(f"{'='*70}\n")

# ── Per-coin aggregators ───────────────────────────────────────────────────────
base_agg  = {'trades': 0, 'wins': 0, 'pnl': 0.0, 'longs': 0, 'long_wins': 0, 'sl_hits': 0, 'gd_exits': 0}
gd_agg    = {'trades': 0, 'wins': 0, 'pnl': 0.0, 'longs': 0, 'long_wins': 0, 'sl_hits': 0, 'gd_exits': 0}

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
    df   = df[mask].copy()
    hmm  = hmm[mask.values]

    close = df["close"].values; high = df["high"].values; low = df["low"].values
    atr   = df["atr_14_h1"].values; n = len(df)
    h4_sh = df["h4_swing_high"].values if "h4_swing_high" in df.columns else np.full(n, np.nan)
    h4_sl = df["h4_swing_low"].values  if "h4_swing_low"  in df.columns else np.full(n, np.nan)
    h4_tr = df["h4_trend"].values      if "h4_trend"      in df.columns else None
    yt    = df["label"].map(LM).values.astype(np.int32)

    # TB v3 predictions
    X_tb = np.zeros((n, len(tb_feats)), dtype=np.float64)
    for i, c in enumerate(tb_feats):
        if c in df.columns:
            X_tb[:, i] = df[c].ffill().fillna(0).values.astype(np.float64)

    proba_tb = tb_model.predict_proba(X_tb)
    conf_tb  = np.max(proba_tb, axis=1)
    yp_tb    = np.argmax(proba_tb, axis=1)
    for r, th in REGIME_THRESH.items():
        yp_tb[(hmm == r) & (yp_tb != 1) & (conf_tb < th)] = 1

    # Guardian Delta raw feature arrays (for delta computation inside evaluator)
    gd_raw = {f: df[f].values.astype(np.float64) for f in BASE_FEATURES if f in df.columns}

    common_kwargs = dict(
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

    # Baseline (no Guardian Delta)
    rep_base = full_trading_report(**common_kwargs)

    # With Guardian Delta
    rep_gd = full_trading_report(
        **common_kwargs,
        guardian_delta_raw=gd_raw,
        guardian_delta_feats=gd_feats,
        guardian_model=gd_model,
        guardian_mom_thresh=GD_MOM_THRESH,
        guardian_def_thresh=GD_DEF_THRESH,
        guardian_def_min_loss=GD_DEF_MIN_LOSS,
    )

    def accumulate(agg, rep):
        lev = rep.get("lev5x", rep)
        tr  = lev.get("trades", [])
        agg["trades"]     += len(tr)
        agg["wins"]       += sum(1 for t in tr if t.get("net_pnl", 0) > 0)
        agg["pnl"]        += sum(t.get("net_pnl", 0) for t in tr)
        agg["longs"]      += sum(1 for t in tr if t.get("direction") == "LONG")
        agg["long_wins"]  += sum(1 for t in tr if t.get("direction") == "LONG" and t.get("net_pnl", 0) > 0)
        agg["sl_hits"]    += sum(1 for t in tr if "SL" in str(t.get("outcome", "")) or t.get("exit_reason") == "sl")
        agg["gd_exits"]   += sum(1 for t in tr if t.get("outcome") == "GUARDIAN_DELTA_EXIT")

    accumulate(base_agg, rep_base)
    accumulate(gd_agg,   rep_gd)
    logger.info(f"[{sym}] base={len(rep_base.get('lev5x', rep_base).get('trades', []))} "
                f"gd={len(rep_gd.get('lev5x', rep_gd).get('trades', []))}")


def scorecard(name, agg):
    t  = agg["trades"]; w = agg["wins"]; p = agg["pnl"]
    lw = agg["long_wins"]; l = agg["longs"]; sl = agg["sl_hits"]
    gd = agg["gd_exits"]
    wr  = w / max(t, 1) * 100
    lwr = lw / max(l, 1) * 100
    swr = (w - lw) / max(t - l, 1) * 100
    sl_r = sl / max(t, 1) * 100
    gd_r = gd / max(t, 1) * 100
    pt   = p / max(t, 1)
    return {
        "name": name, "trades": t, "wr": wr,
        "long_wr": lwr, "short_wr": swr,
        "long_pct": l / max(t, 1) * 100,
        "pnl": p, "pnl_per_trade": pt,
        "sl_rate": sl_r,
        "gd_exit_rate": gd_r,
        "pnl_per_month": p / 5,
    }

base_sc = scorecard("TB v3 standalone",           base_agg)
gd_sc   = scorecard(f"TB v3 + Guardian Delta ({GD_MOM_THRESH}/{GD_DEF_THRESH})", gd_agg)

# ── Print scorecard ────────────────────────────────────────────────────────────
print(f"\n{'='*70}")
print(f"  SCORECARD — TB v3: Guardian Delta vs Standalone")
print(f"  Holdout: Nov 2025 – Apr 2026 | {len(available)} coins | $10/trade 5x")
print(f"{'='*70}")
print(f"\n{'Metrik':<30} {'TB v3 Standalone':>18} {'+ Guard.Delta':>15}")
print("-" * 65)

rows = [
    ("Total Trades",      f"{base_sc['trades']:,}",            f"{gd_sc['trades']:,}"),
    ("Trades/bulan",      f"{base_sc['trades']/5:.0f}",        f"{gd_sc['trades']/5:.0f}"),
    ("Win Rate",          f"{base_sc['wr']:.1f}%",             f"{gd_sc['wr']:.1f}%"),
    ("  LONG WR",         f"{base_sc['long_wr']:.1f}%",        f"{gd_sc['long_wr']:.1f}%"),
    ("  SHORT WR",        f"{base_sc['short_wr']:.1f}%",       f"{gd_sc['short_wr']:.1f}%"),
    ("  LONG share",      f"{base_sc['long_pct']:.1f}%",       f"{gd_sc['long_pct']:.1f}%"),
    ("Net PnL (5 bln)",   f"${base_sc['pnl']:+.0f}",           f"${gd_sc['pnl']:+.0f}"),
    ("PnL/bulan",         f"${base_sc['pnl_per_month']:+.0f}", f"${gd_sc['pnl_per_month']:+.0f}"),
    ("PnL/trade",         f"${base_sc['pnl_per_trade']:+.2f}", f"${gd_sc['pnl_per_trade']:+.2f}"),
    ("SL Hit Rate",       f"{base_sc['sl_rate']:.1f}%",        f"{gd_sc['sl_rate']:.1f}%"),
    ("Guard.Delta exits", "—",                                  f"{gd_sc['gd_exit_rate']:.1f}%"),
]
for label, v_base, v_gd in rows:
    marker = " <--" if label == "Net PnL (5 bln)" else ""
    print(f"  {label:<28} {v_base:>18} {v_gd:>15}{marker}")

delta_pnl = gd_sc["pnl"] - base_sc["pnl"]
delta_pct  = delta_pnl / max(abs(base_sc["pnl"]), 1) * 100
print(f"\n  Guardian Delta impact: {delta_pnl:+.0f} ({delta_pct:+.1f}% vs standalone)")
winner = "Guardian Delta HELPS" if delta_pnl > 0 else "Standalone better"
print(f"  -> {winner}")

# Save results
out = {
    "standalone": base_sc,
    "guardian_delta": gd_sc,
    "delta_pnl": delta_pnl,
    "guardian_params": {
        "mom_thresh": GD_MOM_THRESH,
        "def_thresh": GD_DEF_THRESH,
        "def_min_loss": GD_DEF_MIN_LOSS,
    },
    "note": "swing-based TP/SL, full_trading_report(), 21 coins Nov2025-Apr2026",
}
out_path = MODEL_DIR / "runs" / TB_RUN / "holdout_guardian_delta_result.json"
with open(out_path, "w") as f:
    json.dump(out, f, indent=2)
print(f"\n  Saved -> {out_path}")
print(f"{'='*70}")
