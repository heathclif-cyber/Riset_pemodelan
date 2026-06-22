"""
pipeline/07_holdout_kelly_normalized.py

Kelly ternormalisasi ke BASE_MODAL=$10:
  factor = kelly_frac(conf) / kelly_frac(avg_conf)
  modal  = $10 * factor

Sehingga: conf rata-rata -> factor=1.0 -> $10/trade
          conf tinggi    -> factor>1   -> lebih dari $10
          conf rendah    -> factor<1   -> kurang dari $10

Komparasi lengkap:
  1. Fixed $10
  2. Kelly-normalized (conf only)
  3. Kelly-normalized + regime (conf + HMM)
  4. Kelly-normalized + regime + streak (full hybrid Kelly)
  5. D_shift3 (threshold-based, winner dari sweep)

Usage: python pipeline/07_holdout_kelly_normalized.py
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

logger = setup_logger("07_kelly_norm")

TRENDING_STATES = {0, 3}
FLOW_MOM_WINDOW = 3
LM = LABEL_MAP if isinstance(LABEL_MAP, dict) else {"SHORT": 0, "FLAT": 1, "LONG": 2}

FB_RUN      = "tb_lgbm_flatboost_v2"
GD_CONT_RUN = "tb_guardian_continuation_v1"
THR_TRENDING_LONG  = 0.50; THR_TRENDING_SHORT = 0.55
THR_RANGING_LONG   = 0.55; THR_RANGING_SHORT  = 0.60

BASE_MODAL   = 10.0
FACTOR_MIN   = 0.3   # floor: $3/trade
FACTOR_MAX   = 3.0   # cap:   $30/trade


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
    df     = df[mask].copy(); df_mom = df_mom[mask].copy()
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
        if proba_fb[i, 2] >= (THR_TRENDING_LONG  if is_t else THR_RANGING_LONG):  yp[i] = 2
        elif proba_fb[i, 0] >= (THR_TRENDING_SHORT if is_t else THR_RANGING_SHORT): yp[i] = 0

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


# ── Hitung parameter Kelly ────────────────────────────────────────────────
avg_win      = base_df[base_df["net_pnl"] > 0]["net_pnl"].mean()
avg_loss     = abs(base_df[base_df["net_pnl"] < 0]["net_pnl"].mean())
PAYOFF_RATIO = avg_win / avg_loss   # b
avg_conf     = base_df["conf"].mean()


def kelly_frac(p: float, b: float) -> float:
    """f* = (p*b - (1-p)) / b,  floor=0"""
    return max(0.0, (p * b - (1 - p)) / b)


# Kelly fraction di confidence rata-rata = "baseline" kita
F_BASELINE = kelly_frac(avg_conf, PAYOFF_RATIO)

logger.info(f"Payoff ratio b={PAYOFF_RATIO:.4f}  avg_conf={avg_conf:.4f}  "
            f"f_baseline={F_BASELINE:.4f}")
logger.info(f"Conf range: min={base_df['conf'].min():.3f}  max={base_df['conf'].max():.3f}")

# Distribusi factor Kelly-normalized sebelum clipping
conf_vals = base_df["conf"].values
f_raw = np.array([kelly_frac(c, PAYOFF_RATIO) / F_BASELINE for c in conf_vals])
logger.info(f"Kelly-norm factor (pre-clip): min={f_raw.min():.2f}  "
            f"max={f_raw.max():.2f}  mean={f_raw.mean():.2f}")


# ── Factor functions ──────────────────────────────────────────────────────
def factor_kelly_only(conf: float) -> float:
    """Kelly ternormalisasi — hanya confidence."""
    f = kelly_frac(conf, PAYOFF_RATIO) / F_BASELINE
    return round(max(FACTOR_MIN, min(FACTOR_MAX, f)), 3)


def factor_kelly_regime(conf: float, hmm_state: int) -> float:
    """Kelly-norm + koreksi regime HMM (+0.2 TRENDING, -0.1 RANGING)."""
    f = kelly_frac(conf, PAYOFF_RATIO) / F_BASELINE
    f += 0.2 if hmm_state in TRENDING_STATES else -0.1
    return round(max(FACTOR_MIN, min(FACTOR_MAX, f)), 3)


def factor_kelly_full(conf: float, hmm_state: int, streak: int,
                      monthly_pct: float) -> float:
    """Kelly-norm + regime + streak + monthly guard."""
    f = kelly_frac(conf, PAYOFF_RATIO) / F_BASELINE
    # Regime
    f += 0.2 if hmm_state in TRENDING_STATES else -0.1
    # Streak
    if streak >= 3:    f += 0.2
    elif streak <= -3: f -= 0.3
    # Monthly guard
    if monthly_pct <= -0.10:  f = min(f, 0.7)
    elif monthly_pct >= 0.20: f = min(f, 1.3)
    return round(max(FACTOR_MIN, min(FACTOR_MAX, f)), 3)


def factor_d_shift3(conf: float, hmm_state: int, streak: int,
                    monthly_pct: float) -> float:
    """D_shift3: threshold HIGH=0.58 LOW=0.51 + regime + streak + guard."""
    f = 1.0
    if conf >= 0.58:    f += 0.5
    elif conf < 0.51:   f -= 0.5
    f += 0.2 if hmm_state in TRENDING_STATES else -0.1
    if streak >= 3:     f += 0.2
    elif streak <= -3:  f -= 0.3
    if monthly_pct <= -0.10:  f = min(f, 0.7)
    elif monthly_pct >= 0.20: f = min(f, 1.3)
    return round(max(0.5, min(2.0, f)), 2)


# ── Simulator ─────────────────────────────────────────────────────────────
def _max_dd(pnls: list) -> float:
    cum = 0.0; peak = 0.0; worst = 0.0
    for p in pnls:
        cum += p
        if cum > peak: peak = cum
        if (cum - peak) < worst: worst = cum - peak
    return round(worst, 2)


def simulate(factor_fn) -> dict:
    streak = 0; cum = 0.0; m_pnls = []; factors = []; pnls_all = []
    for month in months:
        mdf = base_df[base_df["month"] == month]
        m_start = cum; m_pnl = 0.0
        for _, row in mdf.iterrows():
            monthly_pct = (cum - m_start) / (BASE_MODAL * 100)
            f    = factor_fn(row["conf"], row["hmm_state"], streak, monthly_pct)
            pnl  = row["net_pnl"] * f
            m_pnl += pnl; cum += pnl
            factors.append(f); pnls_all.append(pnl)
            streak = (max(0, streak) + 1) if row["net_pnl"] > 0 else (min(0, streak) - 1)
        m_pnls.append(round(m_pnl, 2))
    return {
        "total_pnl":  round(cum, 2),
        "monthly":    {str(m): p for m, p in zip(months, m_pnls)},
        "avg_factor": round(sum(factors) / len(factors), 3),
        "avg_modal":  round(sum(factors) / len(factors) * BASE_MODAL, 2),
        "max_dd":     _max_dd(pnls_all),
        "f_min":      round(min(factors), 2),
        "f_max":      round(max(factors), 2),
    }


# Kelly-only tidak butuh streak/monthly, wrap agar signature sama
def simulate_kelly_only():
    streak = 0; cum = 0.0; m_pnls = []; factors = []; pnls_all = []
    for month in months:
        mdf = base_df[base_df["month"] == month]; m_pnl = 0.0
        for _, row in mdf.iterrows():
            f   = factor_kelly_only(row["conf"])
            pnl = row["net_pnl"] * f
            m_pnl += pnl; cum += pnl; factors.append(f); pnls_all.append(pnl)
            streak = (max(0, streak) + 1) if row["net_pnl"] > 0 else (min(0, streak) - 1)
        m_pnls.append(round(m_pnl, 2))
    return {"total_pnl": round(cum, 2), "monthly": {str(m): p for m, p in zip(months, m_pnls)},
            "avg_factor": round(sum(factors)/len(factors), 3),
            "avg_modal": round(sum(factors)/len(factors)*BASE_MODAL, 2),
            "max_dd": _max_dd(pnls_all), "f_min": round(min(factors),2), "f_max": round(max(factors),2)}


def simulate_kelly_regime():
    streak = 0; cum = 0.0; m_pnls = []; factors = []; pnls_all = []
    for month in months:
        mdf = base_df[base_df["month"] == month]; m_pnl = 0.0
        for _, row in mdf.iterrows():
            f   = factor_kelly_regime(row["conf"], row["hmm_state"])
            pnl = row["net_pnl"] * f
            m_pnl += pnl; cum += pnl; factors.append(f); pnls_all.append(pnl)
        m_pnls.append(round(m_pnl, 2))
    return {"total_pnl": round(cum, 2), "monthly": {str(m): p for m, p in zip(months, m_pnls)},
            "avg_factor": round(sum(factors)/len(factors), 3),
            "avg_modal": round(sum(factors)/len(factors)*BASE_MODAL, 2),
            "max_dd": _max_dd(pnls_all), "f_min": round(min(factors),2), "f_max": round(max(factors),2)}


# Fixed baseline
R_fixed = {
    "total_pnl": round(fixed_pnl, 2),
    "monthly":   {str(m): round(base_df[base_df["month"]==m]["net_pnl"].sum(),2) for m in months},
    "avg_factor": 1.0, "avg_modal": BASE_MODAL,
    "max_dd": _max_dd(base_df["net_pnl"].tolist()),
    "f_min": 1.0, "f_max": 1.0,
}


# ── Run semua ─────────────────────────────────────────────────────────────
METHODS = {
    "Fixed $10":              R_fixed,
    "Kelly-norm (conf only)": simulate_kelly_only(),
    "Kelly-norm + regime":    simulate_kelly_regime(),
    "Kelly-norm + all":       simulate(factor_kelly_full),
    "D_shift3 (threshold)":   simulate(factor_d_shift3),
}


# ── Output ────────────────────────────────────────────────────────────────
print(f"\n{'='*84}")
print(f"  KELLY-NORMALIZED vs D_SHIFT3  —  base modal $10  |  Apr-Jun 2026  |  918 trades")
print(f"  Payoff ratio b={PAYOFF_RATIO:.3f}  avg_conf={avg_conf:.3f}  "
      f"f_baseline={F_BASELINE:.4f}  (faktor=1.0 di conf rata-rata)")
print(f"{'='*84}")
print(f"\n  {'Method':<27}  {'PnL':>9}  {'Delta':>8}  {'AvgSize':>8}  {'MaxDD':>8}  "
      f"{'Apr':>8}  {'May':>8}  {'Jun':>8}  {'Range'}  ")
print(f"  {'-'*27}  {'-'*9}  {'-'*8}  {'-'*8}  {'-'*8}  "
      f"{'-'*8}  {'-'*8}  {'-'*8}")

base_pnl = R_fixed["total_pnl"]
for name, r in METHODS.items():
    delta  = r["total_pnl"] - base_pnl
    m_vals = list(r["monthly"].values())
    rng    = f"[{r['f_min']:.2f}x-{r['f_max']:.2f}x]"
    print(f"  {name:<27}  ${r['total_pnl']:>+8.2f}  {delta:>+8.2f}  "
          f"${r['avg_modal']:>6.2f}  ${r['max_dd']:>7.2f}  "
          f"{m_vals[0]:>+8.2f}  {m_vals[1]:>+8.2f}  {m_vals[2]:>+8.2f}  {rng}")

# Detail faktor Kelly-norm
print(f"\n  Detail distribusi Kelly-normalized factor (sebelum clip):")
print(f"  {'Confidence':>12}  {'f_kelly':>8}  {'factor_raw':>12}  {'factor_clipped':>16}")
for c in [0.51, 0.54, 0.58, 0.60, 0.62, 0.65, 0.70]:
    fk  = kelly_frac(c, PAYOFF_RATIO)
    fr  = fk / F_BASELINE
    fc  = max(FACTOR_MIN, min(FACTOR_MAX, fr))
    print(f"  {c:>12.2f}  {fk:>8.4f}  {fr:>12.3f}x  "
          f"{'-> ' + str(round(fc, 3)) + 'x':>16}{'  (kena cap)' if fr > FACTOR_MAX or fr < FACTOR_MIN else ''}")

print(f"\n  Perbandingan Kelly-norm vs D_shift3 (threshold-based):")
print(f"  Kelly-norm: factor bervariasi MULUS mengikuti nilai confidence aktual (kontinu)")
print(f"  D_shift3  : factor melompat di ambang batas 0.58 dan 0.51 (diskrit 3 level)")
print(f"  Kelly-norm + all MENAMBAH regime dan streak di atas faktor Kelly — sama seperti D_shift3")

# Save
out_dir  = ROOT / "reports" / "experiments"
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / "holdout_kelly_normalized.json"
with open(out_path, "w") as f:
    json.dump({
        "model": "tb_widyawardhana_v2_continuation", "period": "Apr-Jun 2026",
        "total_trades": total, "win_rate": round(wins_n/total*100, 1),
        "payoff_ratio": round(PAYOFF_RATIO, 4),
        "avg_conf": round(avg_conf, 4), "f_baseline": round(F_BASELINE, 4),
        "base_modal": BASE_MODAL, "factor_min": FACTOR_MIN, "factor_max": FACTOR_MAX,
        "results": METHODS,
    }, f, indent=2, default=str)
print(f"\n  Saved -> {out_path}")
print(f"  Done.")
