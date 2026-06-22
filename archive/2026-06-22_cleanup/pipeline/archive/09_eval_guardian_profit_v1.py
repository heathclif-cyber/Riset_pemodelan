"""
pipeline/09_eval_guardian_profit_v1.py
Compare: No Guardian vs clean_v2 vs profit_v1

Base config: flatboost_v2 T54_R60 + LSTM soft_pen_hmm_t50
Sweep exit_thr untuk profit_v1: 0.50, 0.55, 0.60, 0.65, 0.70
min_hold_bars: 2 dan 3 (yang terbaik dari sweep sebelumnya)

Tujuan: profit_v1 harus menghasilkan PnL TOTAL lebih tinggi dari no_guardian.
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

logger = setup_logger("09_eval_guardian_profit_v1")

FB_RUN       = "tb_lgbm_flatboost_v2"
GD_OLD_RUN   = "ic32_guardian_clean_v2"
GD_NEW_RUN   = "tb_guardian_profit_v1"
THR_T_LONG   = 0.54
THR_R_LONG   = 0.60
SHORT_OFFSET = 0.05
LSTM_THR     = 0.50
LM           = {"SHORT": 0, "FLAT": 1, "LONG": 2}
TRENDING     = {0, 3}
PERIOD_MONTHS = 2.5
IC32_BENCH   = {"trades": 936, "wr": 62.07, "pnl": 207.22, "pnl_per_trade": 0.2214}

# ── Load models ───────────────────────────────────────────────────────────────
fb_model = joblib.load(MODEL_DIR / "runs" / FB_RUN / "lgbm.pkl")
with open(MODEL_DIR / "runs" / FB_RUN / f"{FB_RUN}_features.json") as f:
    fb_feats = json.load(f)

lstm_model  = load_lstm(MODEL_DIR / "lstm_best.pt")
lstm_scaler = joblib.load(MODEL_DIR / "lstm_scaler.pkl")
with open(MODEL_DIR / "feature_cols_lstm_temporal.json") as f:
    lstm_feats = json.load(f)

# old guardian (clean_v2)
gd_old_model  = joblib.load(MODEL_DIR / "runs" / GD_OLD_RUN / "guardian.pkl")
gd_old_scaler = joblib.load(MODEL_DIR / "runs" / GD_OLD_RUN / "guardian_scaler.pkl")
with open(MODEL_DIR / "runs" / GD_OLD_RUN / "guardian_feature_cols.json") as f:
    gd_old_feats = json.load(f)
gd_old_static = [c for c in gd_old_feats if c not in set(GUARDIAN_DYNAMIC_FEATURES)]

# new guardian (profit_v1)
gd_new_model  = joblib.load(MODEL_DIR / "runs" / GD_NEW_RUN / "guardian.pkl")
gd_new_scaler = joblib.load(MODEL_DIR / "runs" / GD_NEW_RUN / "guardian_scaler.pkl")
with open(MODEL_DIR / "runs" / GD_NEW_RUN / "guardian_feature_cols.json") as f:
    gd_new_feats = json.load(f)
gd_new_static = [c for c in gd_new_feats if c not in set(GUARDIAN_DYNAMIC_FEATURES)]

available = sorted(
    s for s in ALL_COINS
    if (HOLDOUT_DIR / "labeled" / f"{s}_features_v3.parquet").exists()
)

# ── Pre-load ──────────────────────────────────────────────────────────────────
print(f"\nPre-loading {len(available)} koin...")
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

    # T54_R60 + LSTM predictions
    yp = np.full(n, 1, np.int32)
    for i in range(n):
        is_trend = hmm[i] in TRENDING
        thr_l = THR_T_LONG if is_trend else THR_R_LONG
        thr_s = thr_l + SHORT_OFFSET
        lc, sc = proba_fb[i, 2], proba_fb[i, 0]
        if lc >= thr_l:     sig = 2
        elif sc >= thr_s:   sig = 0
        else:                continue
        if is_trend:
            opp = proba_lstm[i, 0] if sig == 2 else proba_lstm[i, 2]
            if opp >= LSTM_THR:
                ec = lc if sig == 2 else sc
                if ec - opp * (ec - thr_l + 0.05) < thr_l:
                    continue
        yp[i] = sig

    X_gd_old = compute_guardian_static_array(df, gd_old_static)
    X_gd_new = compute_guardian_static_array(df, gd_new_static)

    coin_data[sym] = {
        "yp":     yp,
        "conf":   proba_fb.max(axis=1),
        "close":  df["close"].values,
        "high":   df["high"].values,
        "low":    df["low"].values,
        "atr":    df["atr_14_h1"].values,
        "h4_sh":  df["h4_swing_high"].values if "h4_swing_high" in df.columns else np.full(n, np.nan),
        "h4_sl":  df["h4_swing_low"].values  if "h4_swing_low"  in df.columns else np.full(n, np.nan),
        "h4_tr":  df["h4_trend"].values      if "h4_trend"      in df.columns else None,
        "yt":     df["label"].map(LM).values.astype(np.int32),
        "index":  df.index,
        "X_gd_old": X_gd_old,
        "X_gd_new": X_gd_new,
    }
print("Data loaded.\n")


# ── Run one config ────────────────────────────────────────────────────────────
def run_config(gd_model, gd_scaler, X_gd_key, exit_thr, min_hold, use_guardian):
    agg = {"trades": 0, "wins": 0, "pnl": 0.0,
           "gd_exits": 0, "sl_exits": 0, "time_exits": 0,
           "longs": 0, "long_wins": 0, "shorts": 0, "short_wins": 0}
    for sym, d in coin_data.items():
        rep = full_trading_report(
            y_pred=d["yp"], y_actual=d["yt"],
            atr=d["atr"], close=d["close"], high=d["high"], low=d["low"],
            h4_swing_highs=d["h4_sh"], h4_swing_lows=d["h4_sl"],
            index=d["index"], modal=MODAL_PER_TRADE, leverages=LEVERAGE_SIM,
            fee_per_side=FEE_PER_SIDE, slippage=SLIPPAGE_PER_SIDE,
            min_rr=SWING_LABEL_MIN_RR, min_tp_atr=SWING_LABEL_MIN_TP,
            max_sl_atr=SWING_LABEL_MAX_SL,
            tp_fallback_atr=TP_SL_FALLBACK_TP, sl_fallback_atr=TP_SL_FALLBACK_SL,
            max_hold=MAX_HOLDING_BARS, symbol=sym, confidence=d["conf"],
            guardian_enabled=use_guardian,
            guardian_model=gd_model if use_guardian else None,
            guardian_scaler=gd_scaler if use_guardian else None,
            X_guardian=d[X_gd_key] if use_guardian else None,
            guardian_exit_threshold=exit_thr,
            guardian_min_hold_bars=min_hold,
            trailing_stop_enabled=False, h4_trend=d["h4_tr"],
        )
        lev = rep.get("lev5x", rep)
        tr  = lev.get("trades", [])
        agg["trades"] += len(tr)
        agg["wins"]   += sum(1 for t in tr if t.get("net_pnl", 0) > 0)
        agg["pnl"]    += sum(t.get("net_pnl", 0) for t in tr)
        for t in tr:
            out = str(t.get("outcome", ""))
            if t.get("direction") == "LONG":
                agg["longs"] += 1
                if t.get("net_pnl", 0) > 0: agg["long_wins"] += 1
            else:
                agg["shorts"] += 1
                if t.get("net_pnl", 0) > 0: agg["short_wins"] += 1
            if "GUARDIAN" in out:   agg["gd_exits"]   += 1
            elif "TIME"   in out:   agg["time_exits"]  += 1
            elif "SL"     in out or "STOP" in out: agg["sl_exits"] += 1

    t, w, p = agg["trades"], agg["wins"], agg["pnl"]
    l, lw   = agg["longs"],  agg["long_wins"]
    s, sw   = agg["shorts"], agg["short_wins"]
    gd      = agg["gd_exits"]
    return {
        "trades":         t,
        "wr":             round(w/max(t,1)*100, 1),
        "long_wr":        round(lw/max(l,1)*100, 1),
        "short_wr":       round(sw/max(s,1)*100, 1),
        "long_pct":       round(l/max(t,1)*100, 1),
        "pnl":            round(p, 1),
        "pnl_per_month":  round(p/PERIOD_MONTHS, 1),
        "pnl_per_trade":  round(p/max(t,1), 3),
        "gd_pct":         round(gd/max(t,1)*100, 1),
        "sl_pct":         round(agg["sl_exits"]/max(t,1)*100, 1),
        "time_pct":       round(agg["time_exits"]/max(t,1)*100, 1),
    }


# ── Run all configs ───────────────────────────────────────────────────────────
results = []

# 1. Baseline: no Guardian
print("[1] No Guardian (baseline)...", end="", flush=True)
sc = run_config(None, None, "X_gd_old", 0.65, 2, False)
sc["label"] = "no_guardian"
sc["guardian"] = "-"
sc["exit_thr"] = "-"
sc["min_hold"] = "-"
results.append(sc)
print(f" -> {sc['trades']:,} | WR {sc['wr']:.1f}% | PnL ${sc['pnl']:+.0f} | /trade ${sc['pnl_per_trade']:+.3f}")

# 2. Guardian clean_v2 (best params dari sweep sebelumnya)
for exit_thr, min_hold in [(0.60, 3), (0.65, 3), (0.65, 2)]:
    label = f"clean_v2_t{int(exit_thr*100)}_mh{min_hold}"
    print(f"[2] {label}...", end="", flush=True)
    sc = run_config(gd_old_model, gd_old_scaler, "X_gd_old", exit_thr, min_hold, True)
    sc["label"] = label
    sc["guardian"] = "clean_v2"
    sc["exit_thr"] = exit_thr
    sc["min_hold"] = min_hold
    results.append(sc)
    print(f" -> {sc['trades']:,} | WR {sc['wr']:.1f}% | GD {sc['gd_pct']:.0f}% | PnL ${sc['pnl']:+.0f} | /trade ${sc['pnl_per_trade']:+.3f}")

# 3. Guardian profit_v1 -- sweep exit_thr x min_hold
print()
for exit_thr, min_hold in itertools.product([0.50, 0.55, 0.60, 0.65, 0.70], [2, 3]):
    label = f"profit_v1_t{int(exit_thr*100)}_mh{min_hold}"
    print(f"[3] {label}...", end="", flush=True)
    sc = run_config(gd_new_model, gd_new_scaler, "X_gd_new", exit_thr, min_hold, True)
    sc["label"] = label
    sc["guardian"] = "profit_v1"
    sc["exit_thr"] = exit_thr
    sc["min_hold"] = min_hold
    results.append(sc)
    print(f" -> {sc['trades']:,} | WR {sc['wr']:.1f}% | GD {sc['gd_pct']:.0f}% | "
          f"SL {sc['sl_pct']:.0f}% | PnL ${sc['pnl']:+.0f} | /trade ${sc['pnl_per_trade']:+.3f}")


# ── Print scorecard ───────────────────────────────────────────────────────────
df_res = pd.DataFrame(results).sort_values("pnl", ascending=False)  # sort by TOTAL PnL

print(f"\n{'='*120}")
print(f"  GUARDIAN COMPARISON — base: T54_R60 + LSTM | Holdout Apr-Jun 2026 | 21 koin | $10/5x")
print(f"  ic32 benchmark: 936 trades | WR 62.1% | PnL $207 | PnL/trade $0.221")
print(f"  Sorted by: total PnL (tujuan utama Guardian > no_guardian)")
print(f"{'='*120}")
print(f"  {'Label':<28} {'GD':>8} {'thr':>5} {'mh':>3} "
      f"{'Trades':>7} {'WR':>6} {'GD%':>5} {'SL%':>5} "
      f"{'PnL/tr':>8} {'Total PnL':>10} {'PnL/bln':>8}  diff_vs_nogd")
print("  " + "-"*120)

no_gd_pnl = next(r["pnl"] for r in results if r["label"] == "no_guardian")
for r in df_res.to_dict("records"):
    diff = r["pnl"] - no_gd_pnl
    et   = f"{r['exit_thr']:.2f}" if r["exit_thr"] != "-" else "  -"
    mh   = str(r["min_hold"])
    mark = " <-- BEST" if r == df_res.to_dict("records")[0] else ""
    print(f"  {r['label']:<28} {r.get('guardian',''):>8} {et:>5} {mh:>3} "
          f"{int(r['trades']):>7,} {r['wr']:>5.1f}% {r['gd_pct']:>4.1f}% {r['sl_pct']:>4.1f}% "
          f"{r['pnl_per_trade']:>+8.3f} {r['pnl']:>+10.1f} {r['pnl_per_month']:>+8.1f}  "
          f"{diff:>+7.1f}{mark}")

# Check: apakah best profit_v1 > no_guardian?
best_p1 = df_res[df_res["guardian"] == "profit_v1"].iloc[0]
print(f"\n  profit_v1 best vs no_guardian: PnL ${best_p1['pnl']:+.1f} vs ${no_gd_pnl:+.1f} "
      f"(diff {best_p1['pnl'] - no_gd_pnl:+.1f})")
if best_p1["pnl"] > no_gd_pnl:
    print("  RESULT: Guardian profit_v1 BERHASIL meningkatkan total PnL!")
else:
    print("  RESULT: Guardian profit_v1 belum beat no_guardian -- perlu tuning lebih lanjut.")

# Save
out = {
    "period": "2026-04-01 to 2026-06-13",
    "base":   "T54_R60 + LSTM soft_pen_hmm_t50",
    "ic32_benchmark": IC32_BENCH,
    "results": df_res.to_dict("records"),
}
out_path = MODEL_DIR / "runs" / FB_RUN / "guardian_profit_v1_eval.json"
with open(out_path, "w") as f:
    json.dump(out, f, indent=2)
print(f"\n  Saved -> {out_path}")
