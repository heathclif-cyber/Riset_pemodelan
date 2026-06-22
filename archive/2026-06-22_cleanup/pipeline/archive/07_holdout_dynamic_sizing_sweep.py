"""
pipeline/07_holdout_dynamic_sizing_sweep.py

Sweep konfigurasi threshold dynamic sizing.
Data di-load SEKALI, semua skenario dihitung dari trades yang sama.

Skenario:
  A: baseline   HIGH=0.70 LOW=0.60 (original, referensi)
  B: shift-1    HIGH=0.62 LOW=0.54
  C: shift-2    HIGH=0.60 LOW=0.52
  D: shift-3    HIGH=0.58 LOW=0.51
  E: narrow     HIGH=0.60 LOW=0.54 (spread lebih kecil)
  F: aggressive HIGH=0.58 LOW=0.51 MAX=2.5x
  G: no_streak  HIGH=0.60 LOW=0.52 streak disabled

Usage: python pipeline/07_holdout_dynamic_sizing_sweep.py
"""
import json
import sys
import warnings
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from core.evaluator import full_trading_report
from core.utils import setup_logger, ensure_utc_index
from pipeline.backtest_utils import compute_guardian_static_array
from config import (
    ALL_COINS, HOLDOUT_DIR, MODEL_DIR,
    MODAL_PER_TRADE, LEVERAGE_SIM, FEE_PER_SIDE, SLIPPAGE_PER_SIDE,
    MAX_HOLDING_BARS, SWING_LABEL_MIN_RR, SWING_LABEL_MIN_TP,
    SWING_LABEL_MAX_SL, TP_SL_FALLBACK_TP, TP_SL_FALLBACK_SL,
    GUARDIAN_EXIT_THRESHOLD, GUARDIAN_MIN_HOLD_BARS,
    GUARDIAN_DYNAMIC_FEATURES as _GDYN, LABEL_MAP,
)

logger = setup_logger("07_sweep")

TRENDING_STATES = {0, 3}
FLOW_MOM_WINDOW = 3
LM     = LABEL_MAP if isinstance(LABEL_MAP, dict) else {"SHORT": 0, "FLAT": 1, "LONG": 2}
LM_INV = {v: k for k, v in LM.items()}

FB_RUN      = "tb_lgbm_flatboost_v2"
GD_CONT_RUN = "tb_guardian_continuation_v1"

THR_TRENDING_LONG  = 0.50
THR_TRENDING_SHORT = 0.55
THR_RANGING_LONG   = 0.55
THR_RANGING_SHORT  = 0.60

BASE_MODAL = 10.0


# ── Skenario ─────────────────────────────────────────────────────────────

@dataclass
class Scenario:
    name:           str
    label:          str
    conf_high:      float
    conf_low:       float
    factor_min:     float = 0.5
    factor_max:     float = 2.0
    streak_enabled: bool  = True
    streak_win_thr: int   = 3
    streak_loss_thr:int   = -3
    streak_boost:   float = 0.2
    streak_pen:     float = 0.3
    dd_guard:       float = -0.10
    profit_guard:   float = 0.20


SCENARIOS = [
    Scenario("A_baseline",   "HIGH=0.70 LOW=0.60 (original)",         conf_high=0.70, conf_low=0.60),
    Scenario("B_shift1",     "HIGH=0.62 LOW=0.54",                    conf_high=0.62, conf_low=0.54),
    Scenario("C_shift2",     "HIGH=0.60 LOW=0.52",                    conf_high=0.60, conf_low=0.52),
    Scenario("D_shift3",     "HIGH=0.58 LOW=0.51",                    conf_high=0.58, conf_low=0.51),
    Scenario("E_narrow",     "HIGH=0.60 LOW=0.54 spread=0.06",        conf_high=0.60, conf_low=0.54),
    Scenario("F_aggressive", "HIGH=0.58 LOW=0.51 MAX=2.5x",           conf_high=0.58, conf_low=0.51, factor_max=2.5),
    Scenario("G_no_streak",  "HIGH=0.60 LOW=0.52 streak=OFF",         conf_high=0.60, conf_low=0.52, streak_enabled=False),
]


def calc_factor(sc: Scenario, conf: float, hmm_state: int,
                streak: int, monthly_pnl_pct: float) -> float:
    f = 1.0
    if conf >= sc.conf_high:
        f += 0.5
    elif conf < sc.conf_low:
        f -= 0.5
    f += 0.2 if hmm_state in TRENDING_STATES else -0.1
    if sc.streak_enabled:
        if streak >= sc.streak_win_thr:
            f += sc.streak_boost
        elif streak <= sc.streak_loss_thr:
            f -= sc.streak_pen
    if monthly_pnl_pct <= sc.dd_guard:
        f = min(f, 0.7)
    elif monthly_pnl_pct >= sc.profit_guard:
        f = min(f, 1.3)
    return round(max(sc.factor_min, min(sc.factor_max, f)), 2)


def add_momentum_feats(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["cvd_slope_h4_delta"] = df["cvd_slope_h4"].diff(1) if "cvd_slope_h4" in df.columns else 0.0
    df["ofi_h4_accel"]       = df["ofi_h4_delta"].diff(2) if "ofi_h4_delta" in df.columns else 0.0
    df["rsi_h4_slope"]       = df["rsi_h4"].diff(2)       if "rsi_h4" in df.columns else 0.0
    if "dist_liq_50x_long" not in df.columns:
        df["dist_liq_50x_long"] = 0.0
    if "ofi_z_score" in df.columns:
        df["flow_momentum_3bar"] = df["ofi_z_score"].rolling(FLOW_MOM_WINDOW, min_periods=1).mean()
    else:
        df["flow_momentum_3bar"] = 0.0
    return df


# ── Load models ───────────────────────────────────────────────────────────
logger.info("Loading models...")
fb_model = joblib.load(MODEL_DIR / "runs" / FB_RUN / "lgbm.pkl")
with open(MODEL_DIR / "runs" / FB_RUN / f"{FB_RUN}_features.json") as f:
    fb_feats = json.load(f)

cont_dir  = MODEL_DIR / "runs" / GD_CONT_RUN
gd_model  = joblib.load(cont_dir / "guardian.pkl")
gd_scaler = joblib.load(cont_dir / "guardian_scaler.pkl")
with open(cont_dir / "guardian_feature_cols.json") as fh:
    gd_feats = json.load(fh)
gd_static = [c for c in gd_feats if c not in set(_GDYN) and c not in {
    "cvd_slope_h4_delta_entry", "ofi_h4_delta_entry", "flow_momentum_3bar",
}]

available = [
    s for s in ALL_COINS
    if (HOLDOUT_DIR / "labeled" / f"{s}_features_v3.parquet").exists()
]
logger.info(f"Coins: {len(available)}")


# ── Kumpulkan trades sekali ───────────────────────────────────────────────
all_trades = []

for sym in available:
    df = ensure_utc_index(
        pd.read_parquet(HOLDOUT_DIR / "labeled" / f"{sym}_features_v3.parquet")
    ).sort_index()

    hmm = np.full(len(df), 1, np.int32)
    reg_path = HOLDOUT_DIR / "labeled" / f"{sym}_regime_h1.parquet"
    if reg_path.exists():
        try:
            reg = pd.read_parquet(reg_path)
            if "hmm_regime_enc" in reg.columns:
                hmm = reg["hmm_regime_enc"].reindex(df.index, fill_value=1).values.astype(np.int32)
        except Exception:
            pass

    df_mom = add_momentum_feats(df)
    mask   = df["label"].astype(str).isin(LM)
    df     = df[mask].copy()
    df_mom = df_mom[mask].copy()
    hmm    = hmm[mask.values]
    n      = len(df)
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
    times = df.index

    X_fb = np.zeros((n, len(fb_feats)), dtype=np.float64)
    for i, c in enumerate(fb_feats):
        if c in df.columns:
            X_fb[:, i] = df[c].ffill().fillna(0).values.astype(np.float64)
    proba_fb = fb_model.predict_proba(X_fb)
    conf     = proba_fb.max(axis=1)

    yp = np.full(n, 1, np.int32)
    for i in range(n):
        is_t = hmm[i] in TRENDING_STATES
        tl   = THR_TRENDING_LONG  if is_t else THR_RANGING_LONG
        ts   = THR_TRENDING_SHORT if is_t else THR_RANGING_SHORT
        if proba_fb[i, 2] >= tl:
            yp[i] = 2
        elif proba_fb[i, 0] >= ts:
            yp[i] = 0

    flow_arr = (
        df_mom["flow_momentum_3bar"].ffill().fillna(0).values.astype(np.float64)
        if "flow_momentum_3bar" in df_mom.columns else np.zeros(n)
    )
    X_gd = compute_guardian_static_array(df_mom, gd_static)

    rep = full_trading_report(
        y_pred=yp, y_actual=y_act, atr=atr, close=close, high=high, low=low,
        h4_swing_highs=h4_sh, h4_swing_lows=h4_sl,
        index=times, modal=MODAL_PER_TRADE, leverages=LEVERAGE_SIM,
        fee_per_side=FEE_PER_SIDE, slippage=SLIPPAGE_PER_SIDE,
        min_rr=SWING_LABEL_MIN_RR, min_tp_atr=SWING_LABEL_MIN_TP,
        max_sl_atr=SWING_LABEL_MAX_SL,
        tp_fallback_atr=TP_SL_FALLBACK_TP, sl_fallback_atr=TP_SL_FALLBACK_SL,
        max_hold=MAX_HOLDING_BARS, symbol=sym, confidence=conf,
        h4_trend=h4t, trailing_stop_enabled=False,
        guardian_enabled=True, guardian_model=gd_model, guardian_scaler=gd_scaler,
        X_guardian=X_gd, guardian_exit_threshold=GUARDIAN_EXIT_THRESHOLD,
        guardian_min_hold_bars=GUARDIAN_MIN_HOLD_BARS,
        guardian_feat_cols=gd_feats, guardian_static_names=gd_static,
        flow_momentum_arr=flow_arr,
    )

    for t in rep.get("lev5x", rep).get("trades", []):
        bi = int(min(t.get("bar_in", t.get("entry_bar", 0)), n - 1))
        all_trades.append({
            "coin":      sym,
            "opened":    times[bi],
            "net_pnl":   float(t.get("net_pnl", t.get("pnl", 0.0))),
            "conf":      float(conf[bi]),
            "hmm_state": int(hmm[bi]),
        })

    logger.info(f"  [{sym}] {len(rep.get('lev5x', rep).get('trades', []))} trades")

logger.info(f"Total trades: {len(all_trades)}")

base_df = pd.DataFrame(all_trades).sort_values("opened").reset_index(drop=True)
base_df["month"] = pd.to_datetime(base_df["opened"]).dt.to_period("M")
months      = sorted(base_df["month"].unique())
total       = len(base_df)
wins        = (base_df["net_pnl"] > 0).sum()
fixed_pnl   = base_df["net_pnl"].sum()


# ── Simulate ──────────────────────────────────────────────────────────────
def simulate(sc: Scenario) -> dict:
    streak = 0
    cum    = 0.0
    month_pnls  = []
    factors_all = []

    for month in months:
        mdf         = base_df[base_df["month"] == month]
        month_start = cum
        m_pnl       = 0.0

        for _, row in mdf.iterrows():
            monthly_pct = (cum - month_start) / (BASE_MODAL * 100)
            f           = calc_factor(sc, row["conf"], row["hmm_state"], streak, monthly_pct)
            pnl         = row["net_pnl"] * f
            m_pnl      += pnl
            cum        += pnl
            factors_all.append(f)
            streak = (max(0, streak) + 1) if row["net_pnl"] > 0 else (min(0, streak) - 1)

        month_pnls.append(round(m_pnl, 2))

    avg_f = sum(factors_all) / len(factors_all)
    fc    = Counter(round(f, 1) for f in factors_all)
    return {
        "total_pnl":    round(cum, 2),
        "pnl_per_trade":round(cum / total, 4),
        "delta_usd":    round(cum - fixed_pnl, 2),
        "delta_pct":    round((cum - fixed_pnl) / abs(fixed_pnl) * 100, 1) if fixed_pnl else 0,
        "avg_factor":   round(avg_f, 2),
        "avg_modal":    round(BASE_MODAL * avg_f, 2),
        "best_month":   max(month_pnls),
        "worst_month":  min(month_pnls),
        "pct_boost":    round(sum(1 for f in factors_all if f > 1.0) / len(factors_all) * 100, 1),
        "pct_reduced":  round(sum(1 for f in factors_all if f < 1.0) / len(factors_all) * 100, 1),
        "factor_dist":  dict(sorted(fc.items())),
        "monthly_pnls": month_pnls,
    }


# ── Run dan tampilkan ─────────────────────────────────────────────────────
print(f"\n{'='*80}")
print(f"  SWEEP DYNAMIC SIZING  —  tb_widyawardhana_v2_continuation  |  Apr-Jun 2026")
print(f"  Base $10/trade 5x  |  {total} trades  {wins}W  WR={wins/total*100:.1f}%")
print(f"  Fixed baseline PnL: ${fixed_pnl:+.2f}")
print(f"{'='*80}\n")

results_all = {}
for sc in SCENARIOS:
    r = simulate(sc)
    results_all[sc.name] = {"config": sc.label, **r}
    dist = "  ".join(f"{k:.1f}x={v}" for k, v in r["factor_dist"].items())
    month_str = "  ".join(f"{str(m)[-7:]}=${p:>+.1f}" for m, p in zip(months, r["monthly_pnls"]))
    print(f"  [{sc.name}]  {sc.label}")
    print(f"    PnL : ${r['total_pnl']:>+8.2f}  delta vs fixed: ${r['delta_usd']:>+.2f} ({r['delta_pct']:>+.1f}%)")
    print(f"    Size: avg {r['avg_factor']:.2f}x (${r['avg_modal']:.2f})  "
          f"boost {r['pct_boost']:.0f}%  reduced {r['pct_reduced']:.0f}%")
    print(f"    Month: best=${r['best_month']:>+.2f}  worst=${r['worst_month']:>+.2f}  |  {month_str}")
    print(f"    Dist: {dist}\n")


# ── Ranking ───────────────────────────────────────────────────────────────
print(f"{'='*80}")
print(f"  RANKING  (by total PnL)")
print(f"{'='*80}")
ranked = sorted(results_all.items(), key=lambda x: x[1]["total_pnl"], reverse=True)
for rank, (name, r) in enumerate(ranked, 1):
    marker = " <-- WINNER" if rank == 1 else ""
    print(f"  #{rank}  {name:<16}  ${r['total_pnl']:>+8.2f}  ({r['delta_pct']:>+.1f}%)  "
          f"avg={r['avg_factor']:.2f}x  {r['config']}{marker}")
print(f"  ---  {'fixed $10':16}  ${fixed_pnl:>+8.2f}  (0.0%)  reference")


# ── Save ──────────────────────────────────────────────────────────────────
out_dir = ROOT / "reports" / "experiments"
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / "holdout_dynamic_sizing_sweep.json"
with open(out_path, "w") as f:
    json.dump({
        "model":        "tb_widyawardhana_v2_continuation",
        "period":       "Apr-Jun 2026",
        "total_trades": total,
        "win_rate":     round(wins / total * 100, 1),
        "fixed_pnl":    round(fixed_pnl, 2),
        "base_modal":   BASE_MODAL,
        "scenarios":    results_all,
    }, f, indent=2, default=str)
print(f"\n  Saved -> {out_path}")
print(f"  Done.")
