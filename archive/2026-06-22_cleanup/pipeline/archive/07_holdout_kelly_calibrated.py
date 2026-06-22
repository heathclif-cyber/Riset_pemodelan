"""
pipeline/07_holdout_kelly_calibrated.py

Kelly dengan dua improvement:
  1. Confidence calibration (Platt / Isotonic) — raw softmax → true win probability
  2. Per-trade b — gunakan RR dari TP/SL aktual tiap trade, bukan rata-rata

Walk-forward calibration (honest, tanpa lookahead):
  - April  : no prior data → raw conf sebagai p estimate
  - May    : calibrate on April trades → apply to May
  - June   : calibrate on April+May trades → apply to June

Komparasi akhir:
  Fixed $10 | D_shift3 | Kelly-norm-uncal | Kelly-Platt+pertrade-b | Kelly-Isotonic+pertrade-b

Usage: python pipeline/07_holdout_kelly_calibrated.py
"""
import json
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression

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

logger = setup_logger("07_kelly_cal")

TRENDING_STATES = {0, 3}
FLOW_MOM_WINDOW = 3
LM = LABEL_MAP if isinstance(LABEL_MAP, dict) else {"SHORT": 0, "FLAT": 1, "LONG": 2}

FB_RUN      = "tb_lgbm_flatboost_v2"
GD_CONT_RUN = "tb_guardian_continuation_v1"
THR_TRENDING_LONG  = 0.50; THR_TRENDING_SHORT = 0.55
THR_RANGING_LONG   = 0.55; THR_RANGING_SHORT  = 0.60

BASE_MODAL = 10.0
FACTOR_MIN = 0.3
FACTOR_MAX = 3.0


# ── Helpers ───────────────────────────────────────────────────────────────
def add_momentum_feats(df):
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


def kelly_frac(p, b):
    return max(0.0, (p * b - (1 - p)) / b)


def _max_dd(pnls):
    cum = peak = 0.0; worst = 0.0
    for p in pnls:
        cum += p
        if cum > peak: peak = cum
        if (cum - peak) < worst: worst = cum - peak
    return round(worst, 2)


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


# ── Kumpulkan trades + rr ─────────────────────────────────────────────────
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
    hmm    = hmm[mask.values]; n = len(df)
    if n < 50: continue

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
        bi  = int(min(t.get("bar_in", t.get("entry_bar", 0)), n - 1))
        rr  = float(t.get("rr", 1.5))          # per-trade planned RR (b)
        rr  = max(0.5, min(10.0, rr))           # sanity clip
        all_trades.append({
            "coin":      sym,
            "opened":    times[bi],
            "net_pnl":   float(t.get("net_pnl", t.get("pnl", 0.0))),
            "conf":      float(conf[bi]),
            "hmm_state": int(hmm[bi]),
            "rr":        rr,                    # per-trade b
            "win":       int(t.get("net_pnl", t.get("pnl", 0.0)) > 0),
        })

logger.info(f"Total trades: {len(all_trades)}")

base_df = pd.DataFrame(all_trades).sort_values("opened").reset_index(drop=True)
base_df["month"] = pd.to_datetime(base_df["opened"]).dt.to_period("M")
months    = sorted(base_df["month"].unique())
total     = len(base_df)
wins_n    = base_df["win"].sum()
fixed_pnl = base_df["net_pnl"].sum()

avg_rr   = base_df["rr"].mean()
avg_conf = base_df["conf"].mean()


# ── Analisis per-trade RR ─────────────────────────────────────────────────
print(f"\n  Per-trade RR (b) stats:")
print(f"    min={base_df['rr'].min():.2f}  max={base_df['rr'].max():.2f}  "
      f"mean={avg_rr:.3f}  median={base_df['rr'].median():.3f}")
rr_bins = [0, 1.5, 2.0, 2.5, 3.0, 99]
rr_labels = ["<1.5", "1.5-2.0", "2.0-2.5", "2.5-3.0", "3.0+"]
base_df["rr_bin"] = pd.cut(base_df["rr"], bins=rr_bins, labels=rr_labels)
for lbl, grp in base_df.groupby("rr_bin", observed=True):
    print(f"    RR {lbl:<8}: {len(grp):>3} trades  "
          f"WR={grp['win'].mean()*100:.1f}%  "
          f"avg_pnl=${grp['net_pnl'].mean():>+.4f}")


# ── Walk-forward calibration ──────────────────────────────────────────────
# Untuk tiap trade, simpan p_cal dari Platt dan Isotonic
base_df["p_platt"]    = base_df["conf"].copy()   # default: uncalibrated
base_df["p_isotonic"] = base_df["conf"].copy()

for month_idx, month in enumerate(months):
    curr_mask  = base_df["month"] == month
    prior_mask = base_df["month"].isin(months[:month_idx])
    prior_df   = base_df[prior_mask]

    if month_idx == 0 or len(prior_df) < 30:
        # Tidak ada data prior yang cukup → pakai raw conf
        logger.info(f"  {month}: no calibration (prior_n={len(prior_df)}) → using raw conf")
        continue

    X_prior = prior_df["conf"].values.reshape(-1, 1)
    y_prior = prior_df["win"].values

    # Platt scaling (logistic regression)
    platt = LogisticRegression(C=1.0, solver="lbfgs", max_iter=500)
    platt.fit(X_prior, y_prior)
    p_platt_curr = platt.predict_proba(
        base_df.loc[curr_mask, "conf"].values.reshape(-1, 1)
    )[:, 1]
    base_df.loc[curr_mask, "p_platt"] = p_platt_curr

    # Isotonic regression
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(X_prior.ravel(), y_prior)
    p_iso_curr = iso.predict(base_df.loc[curr_mask, "conf"].values)
    base_df.loc[curr_mask, "p_isotonic"] = p_iso_curr

    n_curr = curr_mask.sum()
    logger.info(f"  {month}: calibrated with {len(prior_df)} prior trades → {n_curr} trades "
                f"| Platt p: {p_platt_curr.mean():.3f}  Iso p: {p_iso_curr.mean():.3f}")


# ── Kualitas kalibrasi ────────────────────────────────────────────────────
print(f"\n{'='*78}")
print(f"  KUALITAS KALIBRASI (conf bins vs actual win rate)")
print(f"  Perfect calibration: p_cal ≈ actual WR")
print(f"{'='*78}")
print(f"  {'Conf bin':<12}  {'N':>5}  {'ActualWR':>9}  {'RawConf':>9}  "
      f"{'Platt p':>9}  {'Isotonic':>9}  {'Error_raw':>10}  {'Error_platt':>12}")

conf_bins   = [0.50, 0.55, 0.60, 0.65, 0.70, 1.0]
conf_labels = ["0.50-0.55", "0.55-0.60", "0.60-0.65", "0.65-0.70", "0.70+"]
base_df["conf_bin"] = pd.cut(base_df["conf"], bins=conf_bins, labels=conf_labels)

total_err_raw = total_err_platt = 0.0
for lbl, grp in base_df.groupby("conf_bin", observed=True):
    if len(grp) == 0:
        continue
    actual_wr  = grp["win"].mean()
    raw_conf   = grp["conf"].mean()
    platt_p    = grp["p_platt"].mean()
    iso_p      = grp["p_isotonic"].mean()
    err_raw    = abs(raw_conf - actual_wr)
    err_platt  = abs(platt_p - actual_wr)
    total_err_raw   += err_raw * len(grp)
    total_err_platt += err_platt * len(grp)
    print(f"  {lbl:<12}  {len(grp):>5}  {actual_wr:>8.1%}  {raw_conf:>9.3f}  "
          f"{platt_p:>9.3f}  {iso_p:>9.3f}  {err_raw:>10.3f}  {err_platt:>12.3f}")

mae_raw   = total_err_raw / total
mae_platt = total_err_platt / total
print(f"\n  MAE raw conf = {mae_raw:.4f}   MAE Platt = {mae_platt:.4f}   "
      f"(lebih kecil = lebih baik, 0 = sempurna)")


# ── Normalisasi f_baseline ────────────────────────────────────────────────
# Hitung rata-rata Kelly fraction dari semua trades (untuk anchor ke $10)
# Uncalibrated: pakai raw conf + avg rr (seperti Kelly-norm sebelumnya)
f_baseline_uncal = base_df.apply(
    lambda r: kelly_frac(r["conf"], avg_rr), axis=1
).mean()
f_baseline_uncal = max(f_baseline_uncal, 1e-6)

# Calibrated Platt + per-trade rr
f_baseline_platt = base_df.apply(
    lambda r: kelly_frac(r["p_platt"], r["rr"]), axis=1
).mean()
f_baseline_platt = max(f_baseline_platt, 1e-6)

# Calibrated Isotonic + per-trade rr
f_baseline_iso = base_df.apply(
    lambda r: kelly_frac(r["p_isotonic"], r["rr"]), axis=1
).mean()
f_baseline_iso = max(f_baseline_iso, 1e-6)

logger.info(f"f_baseline: uncal={f_baseline_uncal:.4f}  platt={f_baseline_platt:.4f}  iso={f_baseline_iso:.4f}")


# ── Fungsi faktor ─────────────────────────────────────────────────────────
def factor_d_shift3(conf, hmm_state, streak, monthly_pct):
    f = 1.0
    if conf >= 0.58:    f += 0.5
    elif conf < 0.51:   f -= 0.5
    f += 0.2 if hmm_state in TRENDING_STATES else -0.1
    if streak >= 3:     f += 0.2
    elif streak <= -3:  f -= 0.3
    if monthly_pct <= -0.10: f = min(f, 0.7)
    elif monthly_pct >= 0.20: f = min(f, 1.3)
    return round(max(0.5, min(2.0, f)), 3)


# ── Simulator ─────────────────────────────────────────────────────────────
def simulate(factor_col, f_baseline, use_pertrade_rr=True, use_streak=False):
    """
    factor_col: kolom p yang dipakai ('conf', 'p_platt', 'p_isotonic')
    f_baseline: normalisasi anchor ke $10
    use_pertrade_rr: True = pakai rr per trade, False = pakai avg_rr
    """
    streak = 0; cum = 0.0; m_pnls = []; factors = []; pnls_all = []

    for month in months:
        mdf = base_df[base_df["month"] == month]
        m_start = cum; m_pnl = 0.0

        for _, row in mdf.iterrows():
            p = row[factor_col]
            b = row["rr"] if use_pertrade_rr else avg_rr
            fk = kelly_frac(p, b)
            f  = max(FACTOR_MIN, min(FACTOR_MAX, fk / f_baseline))

            pnl = row["net_pnl"] * f
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
        "f_range":    (round(min(factors), 2), round(max(factors), 2)),
        "pct_above1": round(sum(1 for f in factors if f > 1.0) / len(factors) * 100, 1),
    }


def simulate_d_shift3():
    streak = 0; cum = 0.0; m_pnls = []; factors = []; pnls_all = []
    for month in months:
        mdf = base_df[base_df["month"] == month]; m_start = cum; m_pnl = 0.0
        for _, row in mdf.iterrows():
            monthly_pct = (cum - m_start) / (BASE_MODAL * 100)
            f   = factor_d_shift3(row["conf"], row["hmm_state"], streak, monthly_pct)
            pnl = row["net_pnl"] * f
            m_pnl += pnl; cum += pnl; factors.append(f); pnls_all.append(pnl)
            streak = (max(0, streak) + 1) if row["net_pnl"] > 0 else (min(0, streak) - 1)
        m_pnls.append(round(m_pnl, 2))
    return {"total_pnl": round(cum, 2), "monthly": {str(m): p for m, p in zip(months, m_pnls)},
            "avg_factor": round(sum(factors)/len(factors), 3),
            "avg_modal": round(sum(factors)/len(factors)*BASE_MODAL, 2),
            "max_dd": _max_dd(pnls_all),
            "f_range": (round(min(factors),2), round(max(factors),2)),
            "pct_above1": round(sum(1 for f in factors if f > 1.0)/len(factors)*100, 1)}


R_fixed = {
    "total_pnl":  round(fixed_pnl, 2),
    "monthly":    {str(m): round(base_df[base_df["month"]==m]["net_pnl"].sum(),2) for m in months},
    "avg_factor": 1.0, "avg_modal": BASE_MODAL,
    "max_dd":     _max_dd(base_df["net_pnl"].tolist()),
    "f_range":    (1.0, 1.0), "pct_above1": 0.0,
}

METHODS = {
    "Fixed $10":               R_fixed,
    "D_shift3":                simulate_d_shift3(),
    "Kelly-norm (uncal, avg_b)": simulate("conf",      f_baseline_uncal, use_pertrade_rr=False),
    "Kelly-Platt + per-b":     simulate("p_platt",    f_baseline_platt,  use_pertrade_rr=True),
    "Kelly-Isotonic + per-b":  simulate("p_isotonic", f_baseline_iso,    use_pertrade_rr=True),
}


# ── Output tabel ──────────────────────────────────────────────────────────
print(f"\n{'='*92}")
print(f"  HASIL  —  base modal $10  |  Apr-Jun 2026  |  {total} trades  WR={wins_n/total*100:.1f}%")
print(f"  Walk-forward calibration: Platt & Isotonic")
print(f"  Per-trade b: min={base_df['rr'].min():.2f}  max={base_df['rr'].max():.2f}  mean={avg_rr:.3f}")
print(f"{'='*92}")
print(f"\n  {'Method':<27}  {'PnL':>9}  {'Delta':>8}  {'AvgModal':>9}  {'MaxDD':>8}  "
      f"{'Apr':>8}  {'May':>8}  {'Jun':>8}  {'>1x':>5}")
print(f"  {'-'*27}  {'-'*9}  {'-'*8}  {'-'*9}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*5}")

base_pnl = R_fixed["total_pnl"]
for name, r in METHODS.items():
    delta  = r["total_pnl"] - base_pnl
    m_vals = list(r["monthly"].values())
    winner = " <--" if r["total_pnl"] == max(v["total_pnl"] for v in METHODS.values()) else ""
    print(f"  {name:<27}  ${r['total_pnl']:>+8.2f}  {delta:>+8.2f}  "
          f"${r['avg_modal']:>7.2f}  ${r['max_dd']:>7.2f}  "
          f"{m_vals[0]:>+8.2f}  {m_vals[1]:>+8.2f}  {m_vals[2]:>+8.2f}  "
          f"{r['pct_above1']:>4.0f}%{winner}")


# ── Per-trade RR vs Kelly ─────────────────────────────────────────────────
print(f"\n  Efek per-trade b vs avg b (Kelly-Platt):")
print(f"  Kolom: RR bin | Kelly fraction rata-rata | Avg faktor | Kontribusi PnL")
base_df["f_platt_pertrade"] = base_df.apply(
    lambda r: max(FACTOR_MIN, min(FACTOR_MAX, kelly_frac(r["p_platt"], r["rr"]) / f_baseline_platt)), axis=1
)
base_df["f_platt_avgrr"] = base_df.apply(
    lambda r: max(FACTOR_MIN, min(FACTOR_MAX, kelly_frac(r["p_platt"], avg_rr) / f_baseline_platt)), axis=1
)
for lbl, grp in base_df.groupby("rr_bin", observed=True):
    avg_f_pt  = grp["f_platt_pertrade"].mean()
    avg_f_avg = grp["f_platt_avgrr"].mean()
    pnl_pt    = (grp["net_pnl"] * grp["f_platt_pertrade"]).sum()
    pnl_avg   = (grp["net_pnl"] * grp["f_platt_avgrr"]).sum()
    delta     = pnl_pt - pnl_avg
    print(f"    RR {lbl:<8}: per-trade f={avg_f_pt:.2f}x  avg-rr f={avg_f_avg:.2f}x  "
          f"delta_pnl={delta:>+.2f}")


# ── Insight kalibrasi ─────────────────────────────────────────────────────
print(f"\n  Distribusi p_cal (Platt) vs raw conf:")
print(f"  {'Metric':<15}  {'Raw conf':>10}  {'Platt p_cal':>12}  {'Diff':>8}")
print(f"  {'mean':<15}  {base_df['conf'].mean():>10.4f}  {base_df['p_platt'].mean():>12.4f}  "
      f"{base_df['p_platt'].mean()-base_df['conf'].mean():>+8.4f}")
print(f"  {'min':<15}  {base_df['conf'].min():>10.4f}  {base_df['p_platt'].min():>12.4f}  "
      f"{base_df['p_platt'].min()-base_df['conf'].min():>+8.4f}")
print(f"  {'max':<15}  {base_df['conf'].max():>10.4f}  {base_df['p_platt'].max():>12.4f}  "
      f"{base_df['p_platt'].max()-base_df['conf'].max():>+8.4f}")
print(f"  {'std':<15}  {base_df['conf'].std():>10.4f}  {base_df['p_platt'].std():>12.4f}  "
      f"{base_df['p_platt'].std()-base_df['conf'].std():>+8.4f}")


# ── Save ──────────────────────────────────────────────────────────────────
out_dir = ROOT / "reports" / "experiments"
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / "holdout_kelly_calibrated.json"
with open(out_path, "w") as fh:
    json.dump({
        "model": "tb_widyawardhana_v2_continuation", "period": "Apr-Jun 2026",
        "total_trades": total, "win_rate": round(wins_n/total*100, 1),
        "avg_rr": round(avg_rr, 4),
        "mae_raw_conf": round(mae_raw, 4),
        "mae_platt": round(mae_platt, 4),
        "f_baseline_uncal": round(f_baseline_uncal, 4),
        "f_baseline_platt": round(f_baseline_platt, 4),
        "f_baseline_iso":   round(f_baseline_iso, 4),
        "results": METHODS,
    }, fh, indent=2, default=str)
print(f"\n  Saved -> {out_path}")
print(f"  Done.")
