"""
pipeline/07_holdout_kelly.py

Bandingkan Dynamic Sizing (D_shift3) vs Kelly Criterion.
Implementasi Kelly:
  - p per trade = model confidence (conf)
  - b (payoff ratio) = avg_win / avg_loss dari data holdout
  - f* = (p*b - (1-p)) / b
  - Variants: Full Kelly, Half Kelly, Quarter Kelly
  - Rolling bankroll: bankroll berubah setelah tiap trade

Comparison table:
  1. Fixed $10/trade
  2. D_shift3 (HIGH=0.58 LOW=0.51) — best dari sweep
  3. Full Kelly   (f*)
  4. Half Kelly   (f*/2)
  5. Quarter Kelly (f*/4)

Usage: python pipeline/07_holdout_kelly.py
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

logger = setup_logger("07_kelly")

TRENDING_STATES = {0, 3}
FLOW_MOM_WINDOW = 3
LM = LABEL_MAP if isinstance(LABEL_MAP, dict) else {"SHORT": 0, "FLAT": 1, "LONG": 2}

FB_RUN      = "tb_lgbm_flatboost_v2"
GD_CONT_RUN = "tb_guardian_continuation_v1"
THR_TRENDING_LONG  = 0.50; THR_TRENDING_SHORT = 0.55
THR_RANGING_LONG   = 0.55; THR_RANGING_SHORT  = 0.60

BASE_MODAL        = 10.0
INITIAL_BANKROLL  = 1000.0   # assumed total kapital user
FACTOR_MIN        = 0.3      # floor = $3/trade  (lebih rendah dari D_shift3 karena Kelly bisa sangat kecil)
FACTOR_MAX        = 5.0      # cap = $50/trade   (Kelly bisa sangat besar, perlu dibatasi)
MODAL_MAX         = BASE_MODAL * FACTOR_MAX
MODAL_MIN         = BASE_MODAL * FACTOR_MIN


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

available = [s for s in ALL_COINS if (HOLDOUT_DIR / "labeled" / f"{s}_features_v3.parquet").exists()]
logger.info(f"Coins: {len(available)}")


# ── Kumpulkan trades ─────────────────────────────────────────────────────
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

    close = df["close"].values; high = df["high"].values; low = df["low"].values
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
        if proba_fb[i, 2] >= tl:   yp[i] = 2
        elif proba_fb[i, 0] >= ts: yp[i] = 0

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

logger.info(f"Total trades: {len(all_trades)}")

base_df = pd.DataFrame(all_trades).sort_values("opened").reset_index(drop=True)
base_df["month"] = pd.to_datetime(base_df["opened"]).dt.to_period("M")
months    = sorted(base_df["month"].unique())
total     = len(base_df)
wins_n    = (base_df["net_pnl"] > 0).sum()
fixed_pnl = base_df["net_pnl"].sum()


# ── Hitung RR dari data aktual ─────────────────────────────────────────────
avg_win  = base_df[base_df["net_pnl"] > 0]["net_pnl"].mean()
avg_loss = abs(base_df[base_df["net_pnl"] < 0]["net_pnl"].mean())
PAYOFF_RATIO = avg_win / avg_loss

logger.info(f"Payoff ratio (b) = {PAYOFF_RATIO:.3f}  (avg_win={avg_win:.4f}, avg_loss={avg_loss:.4f})")

# Kelly fraction menggunakan conf sebagai p estimate
def kelly_frac(p: float, b: float) -> float:
    """f* = (p*b - (1-p)) / b"""
    f = (p * b - (1 - p)) / b
    return max(0.0, f)  # tidak bisa negatif (jangan bet kalau Kelly negatif)

# Rata-rata Kelly dari dataset — untuk informasi
avg_conf = base_df["conf"].mean()
f_star_avg = kelly_frac(avg_conf, PAYOFF_RATIO)
logger.info(f"Avg conf={avg_conf:.3f}  Full Kelly avg={f_star_avg:.3f} ({f_star_avg*100:.1f}% of bankroll)")
logger.info(f"  Full Kelly = ${INITIAL_BANKROLL * f_star_avg:.1f}/trade  "
            f"Half = ${INITIAL_BANKROLL * f_star_avg * 0.5:.1f}/trade  "
            f"Quarter = ${INITIAL_BANKROLL * f_star_avg * 0.25:.1f}/trade")


# ── D_shift3 factor function (winner dari sweep) ──────────────────────────
def d_shift3_factor(conf: float, hmm_state: int, streak: int, monthly_pct: float) -> float:
    f = 1.0
    if conf >= 0.58:    f += 0.5
    elif conf < 0.51:   f -= 0.5
    f += 0.2 if hmm_state in TRENDING_STATES else -0.1
    if streak >= 3:     f += 0.2
    elif streak <= -3:  f -= 0.3
    if monthly_pct <= -0.10: f = min(f, 0.7)
    elif monthly_pct >= 0.20: f = min(f, 1.3)
    return round(max(0.5, min(2.0, f)), 2)


# ── Simulate semua metode ──────────────────────────────────────────────────
def sim_fixed() -> dict:
    """Fixed $10/trade — baseline."""
    pnls = base_df["net_pnl"].tolist()
    by_month = {m: base_df[base_df["month"] == m]["net_pnl"].sum() for m in months}
    return {
        "total_pnl": round(fixed_pnl, 2),
        "monthly":   {str(k): round(v, 2) for k, v in by_month.items()},
        "avg_factor": 1.0,
        "max_dd":    _calc_dd(base_df["net_pnl"].cumsum()),
    }


def sim_d_shift3() -> dict:
    streak = 0; cum = 0.0; month_pnls = []; factors = []
    for month in months:
        mdf = base_df[base_df["month"] == month]
        m_start = cum; m_pnl = 0.0
        for _, row in mdf.iterrows():
            monthly_pct = (cum - m_start) / (BASE_MODAL * 100)
            f    = d_shift3_factor(row["conf"], row["hmm_state"], streak, monthly_pct)
            pnl  = row["net_pnl"] * f
            m_pnl += pnl; cum += pnl; factors.append(f)
            streak = (max(0, streak) + 1) if row["net_pnl"] > 0 else (min(0, streak) - 1)
        month_pnls.append(round(m_pnl, 2))
    return {
        "total_pnl":  round(cum, 2),
        "monthly":    {str(m): p for m, p in zip(months, month_pnls)},
        "avg_factor": round(sum(factors) / len(factors), 3),
        "max_dd":     _calc_dd(pd.Series([base_df.iloc[i]["net_pnl"] * factors[i] for i in range(total)]).cumsum()),
    }


def sim_kelly(kelly_mult: float, label: str) -> dict:
    """
    Rolling bankroll Kelly.
    Tiap trade: modal_kelly = bankroll * kelly_frac(conf, PAYOFF_RATIO) * kelly_mult
    Modal di-clip ke [MODAL_MIN, MODAL_MAX].
    PnL = net_pnl_base * (modal_kelly / BASE_MODAL)
    Bankroll += pnl setelah setiap trade.
    """
    bankroll = INITIAL_BANKROLL
    cum_pnl  = 0.0
    month_pnls = []
    factors_all = []
    bankroll_series = []

    for month in months:
        mdf  = base_df[base_df["month"] == month]
        m_pnl = 0.0
        for _, row in mdf.iterrows():
            f_raw   = kelly_frac(row["conf"], PAYOFF_RATIO) * kelly_mult
            modal   = bankroll * f_raw
            modal   = max(MODAL_MIN, min(MODAL_MAX, modal))
            factor  = modal / BASE_MODAL
            pnl     = row["net_pnl"] * factor
            m_pnl  += pnl
            cum_pnl += pnl
            bankroll += pnl
            bankroll  = max(bankroll, 1.0)   # prevent bankruptcy dari clip
            factors_all.append(factor)
            bankroll_series.append(bankroll)

        month_pnls.append(round(m_pnl, 2))

    return {
        "total_pnl":    round(cum_pnl, 2),
        "monthly":      {str(m): p for m, p in zip(months, month_pnls)},
        "avg_factor":   round(sum(factors_all) / len(factors_all), 3),
        "avg_modal":    round(sum(factors_all) / len(factors_all) * BASE_MODAL, 2),
        "max_dd":       _calc_dd(pd.Series(bankroll_series) - INITIAL_BANKROLL),
        "final_bankroll": round(bankroll_series[-1], 2),
        "factor_range": (round(min(factors_all), 2), round(max(factors_all), 2)),
    }


def _calc_dd(pnl_series: pd.Series) -> float:
    """Max drawdown dari cumulative PnL."""
    cum = pnl_series.cumsum() if pnl_series.is_monotonic_increasing else pnl_series
    # pnl_series sudah cumsum di caller
    peak = pnl_series.cummax()
    dd   = (pnl_series - peak).min()
    return round(dd, 2)


# Override _calc_dd karena input sudah cumsum di beberapa caller
def _max_dd_from_pnl_list(pnls: list) -> float:
    cum = 0.0; peak = 0.0; max_dd = 0.0
    for p in pnls:
        cum += p
        if cum > peak: peak = cum
        dd = cum - peak
        if dd < max_dd: max_dd = dd
    return round(max_dd, 2)


def sim_fixed_v2() -> dict:
    pnls = base_df["net_pnl"].tolist()
    by_month = {str(m): round(base_df[base_df["month"] == m]["net_pnl"].sum(), 2) for m in months}
    return {"total_pnl": round(fixed_pnl, 2), "monthly": by_month, "avg_factor": 1.0,
            "max_dd": _max_dd_from_pnl_list(pnls), "avg_modal": BASE_MODAL}


def sim_d_v2() -> dict:
    streak = 0; cum = 0.0; m_pnls = []; factors = []; pnls_all = []
    for month in months:
        mdf = base_df[base_df["month"] == month]; m_start = cum; m_pnl = 0.0
        for _, row in mdf.iterrows():
            monthly_pct = (cum - m_start) / (BASE_MODAL * 100)
            f   = d_shift3_factor(row["conf"], row["hmm_state"], streak, monthly_pct)
            pnl = row["net_pnl"] * f
            m_pnl += pnl; cum += pnl; factors.append(f); pnls_all.append(pnl)
            streak = (max(0, streak) + 1) if row["net_pnl"] > 0 else (min(0, streak) - 1)
        m_pnls.append(round(m_pnl, 2))
    return {"total_pnl": round(cum, 2), "monthly": {str(m): p for m, p in zip(months, m_pnls)},
            "avg_factor": round(sum(factors)/len(factors), 3),
            "avg_modal": round(sum(factors)/len(factors)*BASE_MODAL, 2),
            "max_dd": _max_dd_from_pnl_list(pnls_all)}


def sim_kelly_v2(kelly_mult: float) -> dict:
    bankroll = INITIAL_BANKROLL; cum_pnl = 0.0; m_pnls = []
    factors_all = []; pnls_all = []; bankrolls = []
    for month in months:
        mdf = base_df[base_df["month"] == month]; m_pnl = 0.0
        for _, row in mdf.iterrows():
            f_raw  = kelly_frac(row["conf"], PAYOFF_RATIO) * kelly_mult
            modal  = max(MODAL_MIN, min(MODAL_MAX, bankroll * f_raw))
            factor = modal / BASE_MODAL
            pnl    = row["net_pnl"] * factor
            m_pnl += pnl; cum_pnl += pnl
            bankroll = max(1.0, bankroll + pnl)
            factors_all.append(factor); pnls_all.append(pnl); bankrolls.append(bankroll)
        m_pnls.append(round(m_pnl, 2))
    return {
        "total_pnl":      round(cum_pnl, 2),
        "monthly":        {str(m): p for m, p in zip(months, m_pnls)},
        "avg_factor":     round(sum(factors_all)/len(factors_all), 3),
        "avg_modal":      round(sum(factors_all)/len(factors_all)*BASE_MODAL, 2),
        "max_dd":         _max_dd_from_pnl_list(pnls_all),
        "final_bankroll": round(bankrolls[-1], 2),
        "factor_range":   (round(min(factors_all), 2), round(max(factors_all), 2)),
    }


# ── Run semua ─────────────────────────────────────────────────────────────
R = {
    "Fixed $10":          sim_fixed_v2(),
    "D_shift3":           sim_d_v2(),
    "Full Kelly (f*)":    sim_kelly_v2(1.0),
    "Half Kelly (f*/2)":  sim_kelly_v2(0.5),
    "Qtr Kelly (f*/4)":   sim_kelly_v2(0.25),
}


# ── Print hasil ───────────────────────────────────────────────────────────
print(f"\n{'='*82}")
print(f"  KELLY vs DYNAMIC SIZING  —  tb_widyawardhana_v2_continuation  |  Apr-Jun 2026")
print(f"  {total} trades  {wins_n}W  WR={wins_n/total*100:.1f}%  |  "
      f"avg_win=${avg_win:.3f}  avg_loss=${avg_loss:.3f}  Payoff ratio b={PAYOFF_RATIO:.3f}")
print(f"  Full Kelly fraction f* = {f_star_avg*100:.1f}% of bankroll "
      f"(at avg conf={avg_conf:.3f})  |  Bankroll awal = ${INITIAL_BANKROLL:.0f}")
print(f"{'='*82}")

print(f"\n  {'Method':<22}  {'PnL':>9}  {'Delta':>9}  {'Avg Size':>10}  {'Max DD':>8}  "
      f"{'Apr':>8}  {'May':>8}  {'Jun':>8}")
print(f"  {'-'*22}  {'-'*9}  {'-'*9}  {'-'*10}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}")

for name, r in R.items():
    delta    = r["total_pnl"] - R["Fixed $10"]["total_pnl"]
    delta_s  = f"{delta:>+.2f}"
    m_vals   = list(r["monthly"].values())
    print(f"  {name:<22}  ${r['total_pnl']:>+8.2f}  {delta_s:>9}  "
          f"${r['avg_modal']:>8.2f}  ${r['max_dd']:>7.2f}  "
          f"{m_vals[0]:>+8.2f}  {m_vals[1]:>+8.2f}  {m_vals[2]:>+8.2f}")

print()
print(f"  Catatan Kelly — Bankroll akhir:")
for name, r in R.items():
    if "final_bankroll" in r:
        growth = (r["final_bankroll"] - INITIAL_BANKROLL) / INITIAL_BANKROLL * 100
        fr     = r["factor_range"]
        print(f"    {name:<22}  bankroll ${r['final_bankroll']:.2f} "
              f"({growth:>+.1f}%)  factor range [{fr[0]:.2f}x - {fr[1]:.2f}x]")

# Analisis Kelly fraction distribution
print(f"\n  Kelly fraction stats (at avg conf={avg_conf:.3f}, b={PAYOFF_RATIO:.3f}):")
conf_vals = base_df["conf"].values
f_full  = np.array([kelly_frac(c, PAYOFF_RATIO)       for c in conf_vals])
f_half  = np.array([kelly_frac(c, PAYOFF_RATIO) * 0.5 for c in conf_vals])
f_qtr   = np.array([kelly_frac(c, PAYOFF_RATIO) * 0.25 for c in conf_vals])
print(f"    Full Kelly  f*:  min={f_full.min():.3f}  max={f_full.max():.3f}  "
      f"mean={f_full.mean():.3f}  -> modal min=${INITIAL_BANKROLL*f_full.min():.1f}  "
      f"max=${INITIAL_BANKROLL*f_full.max():.1f}")
print(f"    Half Kelly f*/2: min={f_half.min():.3f}  max={f_half.max():.3f}  "
      f"mean={f_half.mean():.3f}  -> modal min=${INITIAL_BANKROLL*f_half.min():.1f}  "
      f"max=${INITIAL_BANKROLL*f_half.max():.1f}")
print(f"    Qtr Kelly f*/4:  min={f_qtr.min():.3f}  max={f_qtr.max():.3f}  "
      f"mean={f_qtr.mean():.3f}  -> modal min=${INITIAL_BANKROLL*f_qtr.min():.1f}  "
      f"max=${INITIAL_BANKROLL*f_qtr.max():.1f}")
print(f"    (actual di-clip ke ${MODAL_MIN:.1f}-${MODAL_MAX:.1f})")

# Berapa persen trade yang kena clip?
clipped_up  = np.sum(INITIAL_BANKROLL * f_full > MODAL_MAX) / total * 100
clipped_dn  = np.sum(INITIAL_BANKROLL * f_full < MODAL_MIN) / total * 100
print(f"\n    Full Kelly trades kena cap atas (>${MODAL_MAX:.0f}): {clipped_up:.1f}%")
print(f"    Full Kelly trades kena floor (<${MODAL_MIN:.0f}): {clipped_dn:.1f}%")

# ── Save ──────────────────────────────────────────────────────────────────
out_dir = ROOT / "reports" / "experiments"
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / "holdout_kelly_compare.json"
with open(out_path, "w") as f:
    json.dump({
        "model":         "tb_widyawardhana_v2_continuation",
        "period":        "Apr-Jun 2026",
        "total_trades":  total,
        "win_rate":      round(wins_n/total*100, 1),
        "payoff_ratio":  round(PAYOFF_RATIO, 4),
        "full_kelly_avg_frac": round(float(f_star_avg), 4),
        "initial_bankroll": INITIAL_BANKROLL,
        "results":       R,
    }, f, indent=2, default=str)
print(f"\n  Saved -> {out_path}")
print(f"  Done.")
