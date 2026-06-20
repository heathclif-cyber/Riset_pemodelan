"""
pipeline/09_tune_guardian.py
Guardian sweep di atas HMM adaptive threshold flatboost_v2

Base config (fixed):
  LGBM     : flatboost_v2 (27 feat)
  Threshold: T54_R60 — TRENDING=0.54/0.59, RANGING=0.60/0.65
  LSTM     : soft_pen_hmm_t50 (aktif di TRENDING, veto jika opp conf >= 0.50)

Sweep:
  guardian_exit_threshold : 0.55, 0.60, 0.65, 0.70, 0.75
  guardian_min_hold_bars  : 1, 2, 3
  Total                   : 15 combos + 1 no-guardian baseline

Guardian: ic32_guardian_clean_v2 (33 static + 7 dynamic = 40 feat)
ic32 benchmark: 936 trades | WR 62.1% | PnL $207 | PnL/trade $0.221
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

logger = setup_logger("09_tune_guardian")

# ── Constants ─────────────────────────────────────────────────────────────────
FB_RUN        = "tb_lgbm_flatboost_v2"
GD_RUN        = "ic32_guardian_clean_v2"
PERIOD_MONTHS = 2.5
IC32_BENCH    = {"trades": 936, "wr": 62.07, "pnl": 207.22, "pnl_per_trade": 0.2214}

# HMM adaptive threshold — T54_R60 (best dari sweep)
THR_T_LONG    = 0.54
THR_R_LONG    = 0.60
SHORT_OFFSET  = 0.05
LSTM_THR      = 0.50
LM            = {"SHORT": 0, "FLAT": 1, "LONG": 2}
TRENDING      = {0, 3}

# Sweep axes
EXIT_THRESHOLDS = [0.55, 0.60, 0.65, 0.70, 0.75]
MIN_HOLD_BARS   = [1, 2, 3]

# ── Load models ───────────────────────────────────────────────────────────────
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
    gd_feats_all = json.load(f)

# Filter out dynamic features — evaluator appends them per-bar
_GDYN        = set(GUARDIAN_DYNAMIC_FEATURES)
gd_feats_static = [c for c in gd_feats_all if c not in _GDYN]
print(f"Guardian: {len(gd_feats_all)} total feat = "
      f"{len(gd_feats_static)} static + {len(_GDYN)} dynamic")

available = sorted(
    s for s in ALL_COINS
    if (HOLDOUT_DIR / "labeled" / f"{s}_features_v3.parquet").exists()
)

# ── Pre-load all coin data ────────────────────────────────────────────────────
print(f"\nPre-loading {len(available)} koin...")
coin_data = {}
for sym in available:
    df = pd.read_parquet(HOLDOUT_DIR / "labeled" / f"{sym}_features_v3.parquet")
    df = ensure_utc_index(df).sort_index()

    # HMM regime
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

    # LGBM flatboost_v2
    X_fb = np.zeros((n, len(fb_feats)), dtype=np.float64)
    for i, c in enumerate(fb_feats):
        if c in df.columns:
            X_fb[:, i] = df[c].ffill().fillna(0).values.astype(np.float64)
    proba_fb = fb_model.predict_proba(X_fb)

    # LSTM
    X_lstm = np.zeros((n, len(lstm_feats)), dtype=np.float64)
    for i, c in enumerate(lstm_feats):
        if c in df.columns:
            X_lstm[:, i] = df[c].ffill().fillna(0).values.astype(np.float64)
    proba_lstm = get_lstm_proba(lstm_model, lstm_scaler, X_lstm, n)

    # Guardian static features
    X_gd = compute_guardian_static_array(df, gd_feats_static)

    # HMM adaptive T54_R60 predictions
    yp = np.full(n, 1, np.int32)
    for i in range(n):
        is_trend = hmm[i] in TRENDING
        thr_l = THR_T_LONG if is_trend else THR_R_LONG
        thr_s = thr_l + SHORT_OFFSET

        long_c  = proba_fb[i, 2]
        short_c = proba_fb[i, 0]

        if long_c >= thr_l:
            sig = 2
        elif short_c >= thr_s:
            sig = 0
        else:
            continue

        if is_trend:
            lstm_opp = proba_lstm[i, 0] if sig == 2 else proba_lstm[i, 2]
            if lstm_opp >= LSTM_THR:
                entry_conf = long_c if sig == 2 else short_c
                adjusted   = entry_conf - lstm_opp * (entry_conf - thr_l + 0.05)
                if adjusted < thr_l:
                    continue
        yp[i] = sig

    coin_data[sym] = {
        "yp":    yp,
        "conf":  proba_fb.max(axis=1),
        "close": df["close"].values,
        "high":  df["high"].values,
        "low":   df["low"].values,
        "atr":   df["atr_14_h1"].values,
        "h4_sh": df["h4_swing_high"].values if "h4_swing_high" in df.columns else np.full(n, np.nan),
        "h4_sl": df["h4_swing_low"].values  if "h4_swing_low"  in df.columns else np.full(n, np.nan),
        "h4_tr": df["h4_trend"].values      if "h4_trend"      in df.columns else None,
        "yt":    df["label"].map(LM).values.astype(np.int32),
        "index": df.index,
        "X_gd":  X_gd,
    }

print(f"Data loaded.\n")


# ── Run one config ────────────────────────────────────────────────────────────
def run_config(exit_thr: float, min_hold: int, use_guardian: bool) -> dict:
    agg = {"trades": 0, "wins": 0, "pnl": 0.0,
           "longs": 0, "long_wins": 0, "shorts": 0, "short_wins": 0,
           "gd_exits": 0, "time_exits": 0, "sl_exits": 0}

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
            X_guardian=d["X_gd"] if use_guardian else None,
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
            if "GUARDIAN" in out:   agg["gd_exits"] += 1
            elif "TIME" in out:     agg["time_exits"] += 1
            elif "SL" in out or "STOP" in out: agg["sl_exits"] += 1

    t, w, p = agg["trades"], agg["wins"], agg["pnl"]
    l, lw   = agg["longs"],  agg["long_wins"]
    s, sw   = agg["shorts"], agg["short_wins"]
    gd      = agg["gd_exits"]
    return {
        "trades": t, "wr": round(w/max(t,1)*100,1),
        "long_wr": round(lw/max(l,1)*100,1),
        "short_wr": round(sw/max(s,1)*100,1),
        "long_pct": round(l/max(t,1)*100,1),
        "pnl": round(p,1),
        "pnl_per_month": round(p/PERIOD_MONTHS,1),
        "pnl_per_trade": round(p/max(t,1),3),
        "gd_exit_pct": round(gd/max(t,1)*100,1),
        "sl_pct": round(agg["sl_exits"]/max(t,1)*100,1),
        "time_exit_pct": round(agg["time_exits"]/max(t,1)*100,1),
    }


# ── Sweep ─────────────────────────────────────────────────────────────────────
n_combos = len(EXIT_THRESHOLDS) * len(MIN_HOLD_BARS)
print(f"Running {n_combos} Guardian combos + 1 no-guardian baseline...\n")

# Baseline: T54_R60 + LSTM, no Guardian
print("  [no_guardian] T54_R60 + LSTM, Guardian=OFF ...", end="", flush=True)
bl = run_config(0.65, 2, use_guardian=False)
bl["label"]      = "no_guardian"
bl["exit_thr"]   = "-"
bl["min_hold"]   = "-"
print(f" -> {bl['trades']:,} trades | WR {bl['wr']:.1f}% | "
      f"PnL ${bl['pnl']:+.0f} | PnL/trade ${bl['pnl_per_trade']:+.3f}\n")

results = [bl]

for exit_thr, min_hold in itertools.product(EXIT_THRESHOLDS, MIN_HOLD_BARS):
    label = f"gd_t{int(exit_thr*100)}_mh{min_hold}"
    print(f"  [{label}] exit_thr={exit_thr:.2f} min_hold={min_hold} ...", end="", flush=True)

    sc = run_config(exit_thr, min_hold, use_guardian=True)
    sc["label"]    = label
    sc["exit_thr"] = round(exit_thr, 2)
    sc["min_hold"] = min_hold
    results.append(sc)
    print(f" -> {sc['trades']:,} | WR {sc['wr']:.1f}% | "
          f"GD_exit {sc['gd_exit_pct']:.0f}% | "
          f"PnL ${sc['pnl']:+.0f} | PnL/trade ${sc['pnl_per_trade']:+.3f}")


# ── Print scorecard ───────────────────────────────────────────────────────────
df_res = pd.DataFrame(results).sort_values("pnl_per_trade", ascending=False)

print(f"\n{'='*115}")
print(f"  GUARDIAN SWEEP — base: flatboost_v2 T54_R60 + LSTM soft_pen_hmm_t50")
print(f"  TRENDING(0,3)=0.54/0.59 | RANGING(1,2)=0.60/0.65 | LSTM thr=0.50")
print(f"  ic32 benchmark: 936 trades | WR 62.1% | PnL $207 | PnL/trade $0.221")
print(f"{'='*115}")
print(f"  {'Label':<20} {'exit_thr':>8} {'min_h':>5} "
      f"{'Trades':>7} {'WR':>6} {'LWR':>6} {'SWR':>6} "
      f"{'Long%':>6} {'GD_ex%':>7} {'SL%':>5} {'Tm%':>5} "
      f"{'PnL':>8} {'PnL/tr':>8}  {'vs_ic32':>8}")
print("  " + "-"*115)

for r in df_res.to_dict("records"):
    d_pt = r["pnl_per_trade"] - IC32_BENCH["pnl_per_trade"]
    mark = " <--" if r == df_res.to_dict("records")[0] else ""
    et   = f"{r['exit_thr']:.2f}" if r["exit_thr"] != "-" else "  —"
    mh   = str(r["min_hold"])
    print(f"  {r['label']:<20} {et:>8} {mh:>5} "
          f"{int(r['trades']):>7,} {r['wr']:>5.1f}% {r['long_wr']:>5.1f}% {r['short_wr']:>5.1f}% "
          f"{r['long_pct']:>5.1f}% {r['gd_exit_pct']:>6.1f}% {r['sl_pct']:>4.1f}% {r['time_exit_pct']:>4.1f}% "
          f"{r['pnl']:>+8.1f} {r['pnl_per_trade']:>+8.3f}  {d_pt:>+7.3f}{mark}")

# Heatmap PnL/trade: exit_thr (baris) x min_hold (kolom)
sub = [r for r in results if r["label"] != "no_guardian"]
df_h = pd.DataFrame(sub)
piv  = df_h.pivot_table(index="exit_thr", columns="min_hold",
                         values="pnl_per_trade", aggfunc="first")
piv  = piv.sort_index(ascending=False)
print(f"\n  Heatmap PnL/trade — baris=exit_thr, kolom=min_hold_bars:")
print(f"  {'exit_thr':<10}" + "".join(f"  min_hold={c}" for c in piv.columns))
for idx, row in piv.iterrows():
    vals = "".join(f"      {v:+.3f}" if not pd.isna(v) else "          -" for v in row.values)
    print(f"  {idx:.2f}{'':6}{vals}")

print(f"\n{'='*115}")

# Save
out = {
    "period":      "2026-04-01 to 2026-06-13",
    "base_config": "T54_R60 + LSTM soft_pen_hmm_t50",
    "thr_T":       THR_T_LONG, "thr_R": THR_R_LONG, "lstm_thr": LSTM_THR,
    "guardian":    GD_RUN,
    "n_coins":     len(available),
    "ic32_benchmark": IC32_BENCH,
    "results":     df_res.to_dict("records"),
}
out_path = MODEL_DIR / "runs" / FB_RUN / "guardian_sweep.json"
with open(out_path, "w") as f:
    json.dump(out, f, indent=2)
print(f"  Saved -> {out_path}")
