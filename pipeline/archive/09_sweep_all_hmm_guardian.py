"""
pipeline/09_sweep_all_hmm_guardian.py
Sweep SEMUA HMM threshold combos x Guardian profit_v1

For each (thr_T, thr_R) combo:
  - Run tanpa Guardian  -> PnL_base
  - Run dengan Guardian profit_v1 (t70_mh3) -> PnL_gd
  - Delta = PnL_gd - PnL_base

Sort by PnL_gd (total PnL with Guardian).

thr_T : 0.38 - 0.54 step 0.02  (9 nilai)
thr_R : 0.50 - 0.70 step 0.05  (5 nilai)
valid : thr_T < thr_R  (42 combos)
+ fixed_0.50 baseline (thr_T = thr_R = 0.50)
"""
import json, sys, warnings, itertools, numpy as np, pandas as pd
from pathlib import Path
warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import joblib
from core.evaluator import full_trading_report
from core.models import load_lstm
from core.utils import setup_logger, ensure_utc_index
from pipeline.backtest_utils import get_lstm_proba, compute_guardian_static_array
from config import *

logger = setup_logger("09_sweep_all_hmm_guardian")

FB_RUN     = "tb_lgbm_flatboost_v2"
GD_RUN     = "tb_guardian_profit_v1"
SHORT_OFF  = 0.05
LSTM_THR   = 0.50
LM         = {"SHORT": 0, "FLAT": 1, "LONG": 2}
TRENDING   = {0, 3}
PERIOD     = 2.5
IC32       = {"trades": 936, "wr": 62.07, "pnl": 207.22, "pnl_per_trade": 0.2214}

# Best Guardian params dari eval sebelumnya
GD_EXIT_THR = 0.70
GD_MIN_HOLD = 3

THR_T_VALS = np.round(np.arange(0.38, 0.55, 0.02), 2)
THR_R_VALS = np.round(np.arange(0.50, 0.71, 0.05), 2)

# ── Load ──────────────────────────────────────────────────────────────────────
fb_model = joblib.load(MODEL_DIR / "runs" / FB_RUN / "lgbm.pkl")
with open(MODEL_DIR / "runs" / FB_RUN / f"{FB_RUN}_features.json") as f:
    fb_feats = json.load(f)

lstm_model  = load_lstm(MODEL_DIR / "lstm_best.pt")
lstm_scaler = joblib.load(MODEL_DIR / "lstm_scaler.pkl")
with open(MODEL_DIR / "feature_cols_lstm_temporal.json") as f:
    lstm_feats = json.load(f)

gd_model  = joblib.load(MODEL_DIR / "runs" / GD_RUN / "guardian.pkl")
gd_scaler = joblib.load(MODEL_DIR / "runs" / GD_RUN / "guardian_scaler.pkl")
with open(MODEL_DIR / "runs" / GD_RUN / "guardian_feature_cols.json") as f:
    gd_feats = json.load(f)
gd_static = [c for c in gd_feats if c not in set(GUARDIAN_DYNAMIC_FEATURES)]

available = sorted(
    s for s in ALL_COINS
    if (HOLDOUT_DIR / "labeled" / f"{s}_features_v3.parquet").exists()
)

# ── Pre-load ──────────────────────────────────────────────────────────────────
print(f"Pre-loading {len(available)} koin...")
coin_data = {}
for sym in available:
    df = pd.read_parquet(HOLDOUT_DIR / "labeled" / f"{sym}_features_v3.parquet")
    df = ensure_utc_index(df).sort_index()

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

    X_fb = np.zeros((n, len(fb_feats)), dtype=np.float64)
    for i, c in enumerate(fb_feats):
        if c in df.columns:
            X_fb[:, i] = df[c].ffill().fillna(0).values.astype(np.float64)
    proba_fb = fb_model.predict_proba(X_fb)

    X_lstm = np.zeros((n, len(lstm_feats)), dtype=np.float64)
    for i, c in enumerate(lstm_feats):
        if c in df.columns:
            X_lstm[:, i] = df[c].ffill().fillna(0).values.astype(np.float64)
    proba_lstm = get_lstm_proba(lstm_model, lstm_scaler, X_lstm, n)

    X_gd = compute_guardian_static_array(df, gd_static)

    coin_data[sym] = {
        "proba_fb":   proba_fb,
        "proba_lstm": proba_lstm,
        "hmm":        hmm,
        "conf":       proba_fb.max(axis=1),
        "close":      df["close"].values,
        "high":       df["high"].values,
        "low":        df["low"].values,
        "atr":        df["atr_14_h1"].values,
        "h4_sh":      df["h4_swing_high"].values if "h4_swing_high" in df.columns else np.full(n, np.nan),
        "h4_sl":      df["h4_swing_low"].values  if "h4_swing_low"  in df.columns else np.full(n, np.nan),
        "h4_tr":      df["h4_trend"].values      if "h4_trend"      in df.columns else None,
        "yt":         df["label"].map(LM).values.astype(np.int32),
        "index":      df.index,
        "X_gd":       X_gd,
    }
print(f"Loaded. Running sweep...\n")


# ── Helpers ───────────────────────────────────────────────────────────────────
def build_yp(proba_fb, proba_lstm, hmm, thr_T, thr_R):
    n  = len(proba_fb)
    yp = np.full(n, 1, np.int32)
    for i in range(n):
        is_t  = hmm[i] in TRENDING
        thr_l = thr_T if is_t else thr_R
        thr_s = thr_l + SHORT_OFF
        lc, sc = proba_fb[i, 2], proba_fb[i, 0]
        if   lc >= thr_l: sig = 2
        elif sc >= thr_s: sig = 0
        else: continue
        if is_t:
            opp = proba_lstm[i, 0] if sig == 2 else proba_lstm[i, 2]
            if opp >= LSTM_THR:
                ec = lc if sig == 2 else sc
                if ec - opp * (ec - thr_l + 0.05) < thr_l:
                    continue
        yp[i] = sig
    return yp


def run_eval(yp_map, use_guardian):
    agg = {"trades": 0, "wins": 0, "pnl": 0.0,
           "gross_profit": 0.0, "gross_loss": 0.0,
           "longs": 0, "long_wins": 0, "shorts": 0, "short_wins": 0,
           "gd": 0, "sl": 0}
    for sym, d in coin_data.items():
        rep = full_trading_report(
            y_pred=yp_map[sym], y_actual=d["yt"],
            atr=d["atr"], close=d["close"], high=d["high"], low=d["low"],
            h4_swing_highs=d["h4_sh"], h4_swing_lows=d["h4_sl"],
            index=d["index"], modal=MODAL_PER_TRADE, leverages=LEVERAGE_SIM,
            fee_per_side=FEE_PER_SIDE, slippage=SLIPPAGE_PER_SIDE,
            min_rr=SWING_LABEL_MIN_RR, min_tp_atr=SWING_LABEL_MIN_TP,
            max_sl_atr=SWING_LABEL_MAX_SL,
            tp_fallback_atr=TP_SL_FALLBACK_TP, sl_fallback_atr=TP_SL_FALLBACK_SL,
            max_hold=MAX_HOLDING_BARS, symbol=sym, confidence=d["conf"],
            guardian_enabled=use_guardian,
            guardian_model=gd_model  if use_guardian else None,
            guardian_scaler=gd_scaler if use_guardian else None,
            X_guardian=d["X_gd"] if use_guardian else None,
            guardian_exit_threshold=GD_EXIT_THR,
            guardian_min_hold_bars=GD_MIN_HOLD,
            trailing_stop_enabled=False, h4_trend=d["h4_tr"],
        )
        lev = rep.get("lev5x", rep)
        tr  = lev.get("trades", [])
        agg["trades"] += len(tr)
        agg["wins"]   += sum(1 for t in tr if t.get("net_pnl", 0) > 0)
        agg["pnl"]    += sum(t.get("net_pnl", 0) for t in tr)
        for t in tr:
            pnl_t = t.get("net_pnl", 0)
            if pnl_t > 0: agg["gross_profit"] += pnl_t
            else:         agg["gross_loss"]   += abs(pnl_t)
            out = str(t.get("outcome", ""))
            if t.get("direction") == "LONG":
                agg["longs"] += 1
                if pnl_t > 0: agg["long_wins"] += 1
            else:
                agg["shorts"] += 1
                if pnl_t > 0: agg["short_wins"] += 1
            if "GUARDIAN" in out: agg["gd"] += 1
            if "SL" in out or "STOP" in out: agg["sl"] += 1
    t, w, p = agg["trades"], agg["wins"], agg["pnl"]
    l, lw   = agg["longs"],  agg["long_wins"]
    s, sw   = agg["shorts"], agg["short_wins"]
    pf = round(agg["gross_profit"] / max(agg["gross_loss"], 0.001), 2)
    return {
        "trades":        t,
        "wr":            round(w/max(t,1)*100, 1),
        "pf":            pf,
        "long_wr":       round(lw/max(l,1)*100, 1),
        "short_wr":      round(sw/max(s,1)*100, 1),
        "long_pct":      round(l/max(t,1)*100, 1),
        "pnl":           round(p, 1),
        "pnl_per_month": round(p/PERIOD, 1),
        "pnl_per_trade": round(p/max(t,1), 3),
        "gd_pct":        round(agg["gd"]/max(t,1)*100, 1),
        "sl_pct":        round(agg["sl"]/max(t,1)*100, 1),
    }


# ── Sweep ─────────────────────────────────────────────────────────────────────
combos = [(0.50, 0.50)]  # fixed baseline
combos += [(float(tT), float(tR))
           for tT, tR in itertools.product(THR_T_VALS, THR_R_VALS)
           if tT < tR]

print(f"Running {len(combos)} configs x 2 (no_gd + profit_v1)...\n")

results = []
for tT, tR in combos:
    label = "fixed_0.50" if tT == tR else f"T{int(round(tT*100))}_R{int(round(tR*100))}"
    print(f"  {label:<12} ", end="", flush=True)

    # Build predictions once, reuse for both runs
    yp_map = {sym: build_yp(d["proba_fb"], d["proba_lstm"], d["hmm"], tT, tR)
              for sym, d in coin_data.items()}

    sc_base = run_eval(yp_map, use_guardian=False)
    sc_gd   = run_eval(yp_map, use_guardian=True)
    delta   = sc_gd["pnl"] - sc_base["pnl"]

    row = {
        "label":         label,
        "thr_T":         round(tT, 2),
        "thr_R":         round(tR, 2),
        # no guardian
        "base_trades":   sc_base["trades"],
        "base_wr":       sc_base["wr"],
        "base_pf":       sc_base["pf"],
        "base_pnl":      sc_base["pnl"],
        "base_ppt":      sc_base["pnl_per_trade"],
        # with guardian
        "gd_trades":     sc_gd["trades"],
        "gd_wr":         sc_gd["wr"],
        "gd_pf":         sc_gd["pf"],
        "gd_pnl":        sc_gd["pnl"],
        "gd_ppt":        sc_gd["pnl_per_trade"],
        "gd_pct":        sc_gd["gd_pct"],
        "sl_pct":        sc_gd["sl_pct"],
        "long_pct":      sc_gd["long_pct"],
        # delta
        "delta_pnl":     round(delta, 1),
        "delta_ppt":     round(sc_gd["pnl_per_trade"] - sc_base["pnl_per_trade"], 3),
    }
    results.append(row)
    print(f"  no_gd: {sc_base['trades']:>5} | WR {sc_base['wr']:.1f}% | PF {sc_base['pf']:.2f} | ${sc_base['pnl']:+.0f}"
          f"   +gd: WR {sc_gd['wr']:.1f}% | PF {sc_gd['pf']:.2f} | ${sc_gd['pnl']:+.0f}"
          f"   delta ${delta:+.1f}")


# ── Scorecard ─────────────────────────────────────────────────────────────────
df = pd.DataFrame(results).sort_values("gd_pnl", ascending=False)

IC32["pf"] = 2.54  # from CLAUDE.md scorecard

print(f"\n{'='*150}")
print(f"  FULL SWEEP: HMM Adaptive Threshold x Guardian profit_v1 (t70_mh3)")
print(f"  Sorted by: Total PnL WITH Guardian")
print(f"  ic32: 936 trades | WR 62.1% | PF 2.54 | PnL $207 | PnL/trade $0.221")
print(f"{'='*150}")
print(f"  {'Label':<12} {'thr_T':>5} {'thr_R':>5}  "
      f"{'Trades':>7} {'WR_b':>6} {'PF_b':>5} {'PnL_b':>8}  "
      f"{'WR_gd':>6} {'PF_gd':>6} {'GD%':>5} {'SL%':>4} {'PnL_gd':>8} {'PPT_gd':>7}  "
      f"{'delta':>7}  {'vs_ic32':>8}")
print("  " + "-"*150)

records = df.to_dict("records")
for r in records:
    d_ic32 = r["gd_pnl"] - IC32["pnl"]
    mark   = " <--" if r == records[0] else ""
    print(f"  {r['label']:<12} {r['thr_T']:>5.2f} {r['thr_R']:>5.2f}  "
          f"{int(r['base_trades']):>7,} {r['base_wr']:>5.1f}% {r['base_pf']:>5.2f} {r['base_pnl']:>+8.1f}  "
          f"{r['gd_wr']:>5.1f}% {r['gd_pf']:>6.2f} {r['gd_pct']:>4.1f}% {r['sl_pct']:>3.1f}% "
          f"{r['gd_pnl']:>+8.1f} {r['gd_ppt']:>+7.3f}  "
          f"{r['delta_pnl']:>+7.1f}  {d_ic32:>+8.1f}{mark}")

# Heatmap total PnL with Guardian
sub = [r for r in results if r["label"] != "fixed_0.50"]
df_h = pd.DataFrame(sub)

print(f"\n  Heatmap Total PnL WITH Guardian (profit_v1 t70_mh3):")
piv_pnl = df_h.pivot_table(index="thr_T", columns="thr_R", values="gd_pnl", aggfunc="first").sort_index(ascending=False)
print(f"  {'thr_T':>6}  " + "".join(f"  thr_R={c:.2f}" for c in piv_pnl.columns))
for idx, row in piv_pnl.iterrows():
    vals = "".join(f"    {v:+7.1f}" if not pd.isna(v) else "          -" for v in row.values)
    marker = " <- baseline thr_T" if abs(idx - 0.50) < 0.01 else ""
    print(f"  {idx:.2f}  {vals}{marker}")

print(f"\n  Heatmap Profit Factor WITH Guardian:")
piv_pf = df_h.pivot_table(index="thr_T", columns="thr_R", values="gd_pf", aggfunc="first").sort_index(ascending=False)
print(f"  {'thr_T':>6}  " + "".join(f"  thr_R={c:.2f}" for c in piv_pf.columns))
for idx, row in piv_pf.iterrows():
    vals = "".join(f"      {v:5.2f}" if not pd.isna(v) else "          -" for v in row.values)
    marker = " <- baseline thr_T" if abs(idx - 0.50) < 0.01 else ""
    print(f"  {idx:.2f}  {vals}{marker}")

print(f"\n{'='*150}")
best = df.iloc[0]
print(f"  BEST CONFIG: {best['label']} | "
      f"{best['gd_trades']} trades | WR {best['gd_wr']:.1f}% | PF {best['gd_pf']:.2f} | "
      f"PnL/trade ${best['gd_ppt']:+.3f} | Total PnL ${best['gd_pnl']:+.1f} "
      f"(+${best['gd_pnl']-IC32['pnl']:+.1f} vs ic32)")

# Save
out = {
    "period":         "2026-04-01 to 2026-06-13",
    "guardian":       GD_RUN,
    "gd_exit_thr":    GD_EXIT_THR,
    "gd_min_hold":    GD_MIN_HOLD,
    "ic32_benchmark": IC32,
    "results":        df.to_dict("records"),
}
out_path = MODEL_DIR / "runs" / FB_RUN / "hmm_guardian_sweep.json"
with open(out_path, "w") as f:
    json.dump(out, f, indent=2)
print(f"  Saved -> {out_path}")
