"""
pipeline/07_holdout_dynamic_sizing.py

Holdout test: tb_widyawardhana_v2_continuation dengan Dynamic Sizing
Period: Apr-Jun 2026 (21 koin)

Output:
  - reports/experiments/holdout_dynamic_sizing_trades.csv   (per-trade detail)
  - reports/experiments/holdout_dynamic_sizing_report.json  (aggregate)

Usage: python pipeline/07_holdout_dynamic_sizing.py
"""
import json
import sys
import warnings
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

logger = setup_logger("07_holdout_dynamic_sizing")

# ── Konstanta model ───────────────────────────────────────────────────────
FB_RUN      = "tb_lgbm_flatboost_v2"
GD_CONT_RUN = "tb_guardian_continuation_v1"

THR_TRENDING_LONG  = 0.50
THR_TRENDING_SHORT = 0.55
THR_RANGING_LONG   = 0.55
THR_RANGING_SHORT  = 0.60
TRENDING_STATES    = {0, 3}
GD_EXIT_THR        = GUARDIAN_EXIT_THRESHOLD
GD_MIN_HOLD        = GUARDIAN_MIN_HOLD_BARS
FLOW_MOM_WINDOW    = 3
MONTHS             = 2.5
LM = LABEL_MAP if isinstance(LABEL_MAP, dict) else {"SHORT": 0, "FLAT": 1, "LONG": 2}
LM_INV = {v: k for k, v in LM.items()}

# ── Dynamic Sizing config ─────────────────────────────────────────────────
BASE_MODAL      = 10.0    # user set
FACTOR_MIN      = 0.5
FACTOR_MAX      = 2.0
CONF_HIGH_THR   = 0.70
CONF_LOW_THR    = 0.60
STREAK_WIN_THR  = 3
STREAK_LOSS_THR = -3
DD_GUARD_PCT    = -0.10
PROFIT_GUARD    =  0.20


def calc_factor(conf: float, hmm_state: int, streak: int, monthly_pnl_pct: float) -> float:
    factor = 1.0

    # 1. Confidence
    if conf >= CONF_HIGH_THR:
        factor += 0.5
    elif conf < CONF_LOW_THR:
        factor -= 0.5

    # 2. Regime (TRENDING=0,3 → boost; RANGING=1,2 → reduce)
    if hmm_state in TRENDING_STATES:
        factor += 0.2
    else:
        factor -= 0.1

    # 3. Streak
    if streak >= STREAK_WIN_THR:
        factor += 0.2
    elif streak <= STREAK_LOSS_THR:
        factor -= 0.3

    # 4. Monthly drawdown / profit guard
    if monthly_pnl_pct <= DD_GUARD_PCT:
        factor = min(factor, 0.7)
    elif monthly_pnl_pct >= PROFIT_GUARD:
        factor = min(factor, 1.3)

    return round(max(FACTOR_MIN, min(FACTOR_MAX, factor)), 2)


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
fb_model = joblib.load(MODEL_DIR / "runs" / FB_RUN / "lgbm.pkl")
with open(MODEL_DIR / "runs" / FB_RUN / f"{FB_RUN}_features.json") as f:
    fb_feats = json.load(f)

cont_run_dir = MODEL_DIR / "runs" / GD_CONT_RUN
gd_model  = joblib.load(cont_run_dir / "guardian.pkl")
gd_scaler = joblib.load(cont_run_dir / "guardian_scaler.pkl")
with open(cont_run_dir / "guardian_feature_cols.json") as f:
    gd_feats = json.load(f)
gd_static = [c for c in gd_feats if c not in set(_GDYN) and c not in {
    "cvd_slope_h4_delta_entry", "ofi_h4_delta_entry", "flow_momentum_3bar",
}]

available = [
    s for s in ALL_COINS
    if (HOLDOUT_DIR / "labeled" / f"{s}_features_v3.parquet").exists()
]
logger.info(f"Model loaded. Coins available: {len(available)}")

# ── Kumpulkan semua trades ────────────────────────────────────────────────
all_trades = []   # list of dict per trade

for sym in available:
    path = HOLDOUT_DIR / "labeled" / f"{sym}_features_v3.parquet"
    df   = ensure_utc_index(pd.read_parquet(path)).sort_index()

    # HMM regime
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

    close  = df["close"].values
    high   = df["high"].values
    low    = df["low"].values
    atr    = df["atr_14_h1"].values
    h4_sh  = df["h4_swing_high"].values if "h4_swing_high" in df.columns else np.full(n, np.nan)
    h4_sl  = df["h4_swing_low"].values  if "h4_swing_low"  in df.columns else np.full(n, np.nan)
    h4t    = df["h4_trend"].values       if "h4_trend"      in df.columns else None
    y_act  = np.array([LM.get(str(v), 1) for v in df["label"].values], dtype=np.int32)
    times  = df.index

    # LGBM predictions
    X_fb = np.zeros((n, len(fb_feats)), dtype=np.float64)
    for i, c in enumerate(fb_feats):
        if c in df.columns:
            X_fb[:, i] = df[c].ffill().fillna(0).values.astype(np.float64)
    proba_fb = fb_model.predict_proba(X_fb)

    yp   = np.full(n, 1, np.int32)
    conf = proba_fb.max(axis=1)
    for i in range(n):
        is_trend = hmm[i] in TRENDING_STATES
        tl = THR_TRENDING_LONG  if is_trend else THR_RANGING_LONG
        ts = THR_TRENDING_SHORT if is_trend else THR_RANGING_SHORT
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
        index=df.index, modal=MODAL_PER_TRADE, leverages=LEVERAGE_SIM,
        fee_per_side=FEE_PER_SIDE, slippage=SLIPPAGE_PER_SIDE,
        min_rr=SWING_LABEL_MIN_RR, min_tp_atr=SWING_LABEL_MIN_TP,
        max_sl_atr=SWING_LABEL_MAX_SL,
        tp_fallback_atr=TP_SL_FALLBACK_TP, sl_fallback_atr=TP_SL_FALLBACK_SL,
        max_hold=MAX_HOLDING_BARS, symbol=sym, confidence=conf,
        h4_trend=h4t, trailing_stop_enabled=False,
        guardian_enabled=True,
        guardian_model=gd_model,
        guardian_scaler=gd_scaler,
        X_guardian=X_gd,
        guardian_exit_threshold=GD_EXIT_THR,
        guardian_min_hold_bars=GD_MIN_HOLD,
        guardian_feat_cols=gd_feats,
        guardian_static_names=gd_static,
        flow_momentum_arr=flow_arr,
    )

    trades_raw = rep.get("lev5x", rep).get("trades", [])

    for t in trades_raw:
        bar_in  = t.get("bar_in",  t.get("entry_bar", 0))
        bar_out = t.get("bar_out", t.get("exit_bar",  0))
        bar_in  = int(min(bar_in,  n - 1))
        bar_out = int(min(bar_out, n - 1))

        direction_int = t.get("pred", t.get("direction_int",
                        2 if t.get("direction", "") == "LONG" else 0))

        all_trades.append({
            "coin":        sym,
            "opened":      times[bar_in]  if bar_in  < len(times) else pd.NaT,
            "closed":      times[bar_out] if bar_out < len(times) else pd.NaT,
            "direction":   LM_INV.get(int(direction_int), str(direction_int)),
            "conf":        float(conf[bar_in]),
            "hmm_state":   int(hmm[bar_in]),
            "hmm_regime":  "TRENDING" if hmm[bar_in] in TRENDING_STATES else "RANGING",
            "entry_price": t.get("entry", t.get("entry_price", 0.0)),
            "exit_price":  t.get("exit",  t.get("exit_price",  0.0)),
            "net_pnl":     t.get("net_pnl", t.get("pnl", 0.0)),
            "exit_reason": t.get("outcome", ""),
            "hold_bars":   bar_out - bar_in,
        })

    logger.info(f"  [{sym}] {len(trades_raw)} trades")

logger.info(f"Total trades collected: {len(all_trades)}")

# ── Dynamic Sizing Simulation ─────────────────────────────────────────────
trades_df = pd.DataFrame(all_trades)
trades_df  = trades_df.sort_values("opened").reset_index(drop=True)
trades_df["month"] = pd.to_datetime(trades_df["opened"]).dt.to_period("M")

print(f"\n{'='*70}")
print(f"  HOLDOUT: tb_widyawardhana_v2_continuation — Apr-Jun 2026")
print(f"  {len(trades_df)} trades | {len(available)} koin | base $10/trade 5x leverage")
print(f"{'='*70}")

# Simulate month-by-month
months = sorted(trades_df["month"].unique())
streak           = 0
month_start_pnl  = 0.0
cumulative_fixed = 0.0
cumulative_dyn   = 0.0
monthly_results  = []

dyn_pnl_col    = []
dyn_factor_col = []
dyn_modal_col  = []

for month in months:
    mdf = trades_df[trades_df["month"] == month].copy()
    m_fixed = 0.0
    m_dyn   = 0.0
    month_start_dyn = cumulative_dyn
    factors_used = []

    for idx, row in mdf.iterrows():
        monthly_pnl_pct = (
            (cumulative_dyn - month_start_dyn) / (BASE_MODAL * 100)
            if BASE_MODAL > 0 else 0.0
        )

        # Fixed
        pnl_fixed = float(row["net_pnl"])
        m_fixed  += pnl_fixed

        # Dynamic
        factor        = calc_factor(row["conf"], row["hmm_state"], streak, monthly_pnl_pct)
        modal_dyn     = BASE_MODAL * factor
        pnl_dyn       = pnl_fixed * factor   # scale dari fixed
        m_dyn        += pnl_dyn
        cumulative_dyn += pnl_dyn
        factors_used.append(factor)

        dyn_pnl_col.append(round(pnl_dyn, 4))
        dyn_factor_col.append(factor)
        dyn_modal_col.append(round(modal_dyn, 2))

        # Update streak
        streak = (max(0, streak) + 1) if pnl_fixed > 0 else (min(0, streak) - 1)

    cumulative_fixed += m_fixed
    wins_m = (mdf["net_pnl"] > 0).sum()
    n_m    = len(mdf)
    wr_m   = wins_m / n_m * 100 if n_m > 0 else 0
    avg_f  = sum(factors_used) / len(factors_used) if factors_used else 1.0
    delta  = m_dyn - m_fixed

    monthly_results.append({
        "month": str(month), "trades": n_m, "wr": round(wr_m, 1),
        "pnl_fixed": round(m_fixed, 2), "pnl_dynamic": round(m_dyn, 2),
        "avg_factor": round(avg_f, 2), "delta": round(delta, 2),
    })

    print(f"\n  {month}  |  {n_m} trades  WR={wr_m:.0f}%  avg_factor={avg_f:.2f}x")
    print(f"    Fixed   : {m_fixed:>+8.2f}  cum={cumulative_fixed:>+8.2f}")
    print(f"    Dynamic : {m_dyn:>+8.2f}  cum={cumulative_dyn:>+8.2f}  (delta: {delta:>+.2f})")
    print(f"    Factors : {min(factors_used):.1f}x - {max(factors_used):.1f}x")

# Tambah kolom ke trades_df
trades_df["dyn_factor"] = dyn_factor_col
trades_df["dyn_modal"]  = dyn_modal_col
trades_df["dyn_pnl"]    = dyn_pnl_col

# ── Ringkasan ─────────────────────────────────────────────────────────────
total_trades  = len(trades_df)
total_wins    = (trades_df["net_pnl"] > 0).sum()
total_losses  = total_trades - total_wins
gross_win     = trades_df[trades_df["net_pnl"] > 0]["net_pnl"].sum()
gross_loss    = trades_df[trades_df["net_pnl"] < 0]["net_pnl"].sum()
pf_fixed      = abs(gross_win / gross_loss) if gross_loss != 0 else float("inf")

dyn_wins_pnl  = trades_df[trades_df["dyn_pnl"] > 0]["dyn_pnl"].sum()
dyn_loss_pnl  = trades_df[trades_df["dyn_pnl"] < 0]["dyn_pnl"].sum()
pf_dyn        = abs(dyn_wins_pnl / dyn_loss_pnl) if dyn_loss_pnl != 0 else float("inf")

# Factor distribution
fc = trades_df["dyn_factor"].value_counts().sort_index()

print(f"\n{'='*70}")
print(f"  RINGKASAN FINAL")
print(f"{'='*70}")
print(f"  Trades : {total_trades} ({total_wins}W / {total_losses}L,  WR={total_wins/total_trades*100:.1f}%)")
print()
print(f"  FIXED $10/trade:")
print(f"    Total PnL    : ${cumulative_fixed:>+.2f}")
print(f"    PnL/trade    : ${cumulative_fixed/total_trades:>+.4f}")
print(f"    Profit Factor: {pf_fixed:.2f}")
print(f"    Best month   : ${max(r['pnl_fixed'] for r in monthly_results):>+.2f}")
print(f"    Worst month  : ${min(r['pnl_fixed'] for r in monthly_results):>+.2f}")
print()
print(f"  DYNAMIC (base $10):")
print(f"    Total PnL    : ${cumulative_dyn:>+.2f}")
print(f"    PnL/trade    : ${cumulative_dyn/total_trades:>+.4f}")
print(f"    Profit Factor: {pf_dyn:.2f}")
print(f"    Best month   : ${max(r['pnl_dynamic'] for r in monthly_results):>+.2f}")
print(f"    Worst month  : ${min(r['pnl_dynamic'] for r in monthly_results):>+.2f}")
print()
improvement     = cumulative_dyn - cumulative_fixed
improvement_pct = improvement / abs(cumulative_fixed) * 100 if cumulative_fixed != 0 else 0
avg_factor_all  = trades_df["dyn_factor"].mean()
print(f"  Delta PnL     : ${improvement:>+.2f}  ({improvement_pct:>+.1f}% vs fixed)")
print(f"  Avg factor    : {avg_factor_all:.2f}x  (avg modal = ${BASE_MODAL * avg_factor_all:.2f})")
print()
print(f"  Distribusi factor:")
for f_val, cnt in fc.items():
    bar = "#" * int(cnt / total_trades * 40)
    pct = cnt / total_trades * 100
    print(f"    {f_val:.1f}x : {bar:<40} {cnt:>3} trade ({pct:.0f}%)")

# Exit reason breakdown (dynamic)
print(f"\n  Exit reason breakdown:")
for reason, grp in trades_df.groupby("exit_reason"):
    n_r  = len(grp)
    wr_r = (grp["net_pnl"] > 0).mean() * 100
    pnl_r = grp["dyn_pnl"].sum()
    print(f"    {reason:<30} {n_r:>4} trades  WR={wr_r:.0f}%  PnL={pnl_r:>+.2f}")

# ── Save outputs ──────────────────────────────────────────────────────────
out_dir = ROOT / "reports" / "experiments"
out_dir.mkdir(parents=True, exist_ok=True)

csv_path = out_dir / "holdout_dynamic_sizing_trades.csv"
trades_df.to_csv(csv_path, index=False)
print(f"\n  Saved per-trade CSV -> {csv_path}")

report = {
    "model": "tb_widyawardhana_v2_continuation",
    "period": "Apr-Jun 2026",
    "coins": len(available),
    "base_modal": BASE_MODAL,
    "leverage": LEVERAGE_SIM,
    "total_trades": total_trades,
    "win_rate": round(total_wins / total_trades * 100, 1),
    "fixed": {
        "total_pnl": round(cumulative_fixed, 2),
        "pnl_per_trade": round(cumulative_fixed / total_trades, 4),
        "profit_factor": round(pf_fixed, 2),
        "monthly": monthly_results,
    },
    "dynamic": {
        "total_pnl": round(cumulative_dyn, 2),
        "pnl_per_trade": round(cumulative_dyn / total_trades, 4),
        "profit_factor": round(pf_dyn, 2),
        "avg_factor": round(avg_factor_all, 2),
        "avg_modal": round(BASE_MODAL * avg_factor_all, 2),
        "monthly": [
            {**r, "pnl_dynamic": r["pnl_dynamic"]} for r in monthly_results
        ],
    },
    "improvement_usd": round(improvement, 2),
    "improvement_pct": round(improvement_pct, 1),
    "factor_config": {
        "base_modal": BASE_MODAL,
        "factor_min": FACTOR_MIN,
        "factor_max": FACTOR_MAX,
        "conf_high_thr": CONF_HIGH_THR,
        "conf_low_thr": CONF_LOW_THR,
        "streak_win_thr": STREAK_WIN_THR,
        "streak_loss_thr": STREAK_LOSS_THR,
        "dd_guard_pct": DD_GUARD_PCT,
        "profit_guard": PROFIT_GUARD,
    },
}

json_path = out_dir / "holdout_dynamic_sizing_report.json"
with open(json_path, "w") as f:
    json.dump(report, f, indent=2)
print(f"  Saved report JSON  -> {json_path}")
print(f"\n  Done.")
