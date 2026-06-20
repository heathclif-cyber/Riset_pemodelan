"""
pipeline/09_tune_hmm_directional.py
LGBM + LSTM + HMM Directional Adaptive Threshold

True adaptive: threshold TURUN untuk arah WITH-TREND, naik untuk COUNTER-TREND dan RANGING.

  TRENDING_UP   (regime 3): thr_long = thr_T   (TURUN)  | thr_short = thr_R   (NAIK)
  TRENDING_DOWN (regime 0): thr_long = thr_R   (NAIK)   | thr_short = thr_T   (TURUN)
  RANGING       (regime 1,2): thr_long = thr_R, thr_short = thr_R  (NAIK semua)

  thr_T (with-trend  threshold): 0.38 - 0.52 step 0.02   (7 nilai, di bawah/sama baseline)
  thr_R (counter/ranging thr)  : 0.55 - 0.75 step 0.05   (5 nilai, di atas baseline)
  Total: 7 x 5 = 35 kombinasi

LSTM: soft_pen_hmm_t50 — penalty saat TRENDING, hanya arah berlawanan conf >= 0.50

Benchmark ic32: 936 trades | WR 62.1% | PnL $207 | PnL/trade $0.221
"""
import json, sys, warnings, itertools, numpy as np, pandas as pd
from pathlib import Path
warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import joblib, torch
from core.evaluator import full_trading_report
from core.models import load_lstm
from core.utils import setup_logger, ensure_utc_index
from pipeline.backtest_utils import get_lstm_proba
from config import *

logger = setup_logger("09_tune_hmm_directional")

# ── Constants ─────────────────────────────────────────────────────────────────
FB_RUN        = "tb_lgbm_flatboost_v2"
PERIOD_MONTHS = 2.5
IC32_BENCH    = {"trades": 936, "wr": 62.07, "pnl": 207.22, "pnl_per_trade": 0.2214}

LM             = {"SHORT": 0, "FLAT": 1, "LONG": 2}
REGIME_UP      = 3   # TRENDING_UP
REGIME_DOWN    = 0   # TRENDING_DOWN
RANGING_RGMS   = {1, 2}
LSTM_THR       = 0.50

# Sweep axes
THR_T_VALS = np.round(np.arange(0.38, 0.53, 0.02), 2)  # with-trend: 0.38..0.52
THR_R_VALS = np.round(np.arange(0.55, 0.76, 0.05), 2)  # counter/ranging: 0.55..0.75

# ── Load models ───────────────────────────────────────────────────────────────
fb_model = joblib.load(MODEL_DIR / "runs" / FB_RUN / "lgbm.pkl")
with open(MODEL_DIR / "runs" / FB_RUN / f"{FB_RUN}_features.json") as f:
    fb_feats = json.load(f)

lstm_model  = load_lstm(MODEL_DIR / "lstm_best.pt")
lstm_scaler = joblib.load(MODEL_DIR / "lstm_scaler.pkl")
with open(MODEL_DIR / "feature_cols_lstm_temporal.json") as f:
    lstm_feats = json.load(f)

available = sorted(
    s for s in ALL_COINS
    if (HOLDOUT_DIR / "labeled" / f"{s}_features_v3.parquet").exists()
)
logger.info(f"Coins: {len(available)} | thr_T sweep: {THR_T_VALS} | thr_R sweep: {THR_R_VALS}")

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

    coin_data[sym] = {
        "proba_fb":   proba_fb,
        "proba_lstm": proba_lstm,
        "hmm":        hmm,
        "close": df["close"].values,
        "high":  df["high"].values,
        "low":   df["low"].values,
        "atr":   df["atr_14_h1"].values,
        "h4_sh": df["h4_swing_high"].values if "h4_swing_high" in df.columns else np.full(n, np.nan),
        "h4_sl": df["h4_swing_low"].values  if "h4_swing_low"  in df.columns else np.full(n, np.nan),
        "h4_tr": df["h4_trend"].values      if "h4_trend"      in df.columns else None,
        "yt":    df["label"].map(LM).values.astype(np.int32),
        "index": df.index,
    }
print(f"Data loaded. Running directional adaptive sweep ({len(THR_T_VALS)*len(THR_R_VALS)} combos + 1 baseline)...\n")


# ── Prediction builder ────────────────────────────────────────────────────────
def build_predictions(proba_fb, proba_lstm, hmm, thr_T, thr_R):
    """
    Per bar:
      TRENDING_UP   (3): long threshold = thr_T (TURUN), short threshold = thr_R (NAIK)
      TRENDING_DOWN (0): long threshold = thr_R (NAIK),  short threshold = thr_T (TURUN)
      RANGING     (1,2): long = thr_R, short = thr_R

    LSTM soft_pen: hanya TRENDING bar, vetoes arah berlawanan jika conf >= LSTM_THR.
    """
    n  = len(proba_fb)
    yp = np.full(n, 1, np.int32)

    for i in range(n):
        regime = hmm[i]

        # Tentukan threshold per arah berdasarkan regime
        if regime == REGIME_UP:
            thr_l = thr_T   # with-trend (LONG) → lebih rendah
            thr_s = thr_R   # counter (SHORT)   → lebih tinggi
        elif regime == REGIME_DOWN:
            thr_l = thr_R   # counter (LONG)    → lebih tinggi
            thr_s = thr_T   # with-trend (SHORT) → lebih rendah
        else:  # RANGING
            thr_l = thr_R
            thr_s = thr_R

        long_conf  = proba_fb[i, 2]
        short_conf = proba_fb[i, 0]

        # LGBM entry signal — LONG prioritas
        if long_conf >= thr_l:
            signal = 2
        elif short_conf >= thr_s:
            signal = 0
        else:
            continue  # FLAT

        # LSTM soft_pen (hanya TRENDING bar)
        if regime in (REGIME_UP, REGIME_DOWN):
            if signal == 2:
                lstm_opp = proba_lstm[i, 0]
                if lstm_opp >= LSTM_THR:
                    adjusted = long_conf - lstm_opp * (long_conf - thr_l + 0.05)
                    if adjusted < thr_l:
                        continue  # vetoed → FLAT
            else:
                lstm_opp = proba_lstm[i, 2]
                if lstm_opp >= LSTM_THR:
                    adjusted = short_conf - lstm_opp * (short_conf - thr_s + 0.05)
                    if adjusted < thr_s:
                        continue

        yp[i] = signal

    return yp


# ── Run one config ────────────────────────────────────────────────────────────
def run_config(thr_T, thr_R):
    agg = {"trades": 0, "wins": 0, "pnl": 0.0,
           "longs": 0, "long_wins": 0, "shorts": 0, "short_wins": 0,
           "up_longs": 0, "down_shorts": 0}

    for sym, d in coin_data.items():
        yp   = build_predictions(d["proba_fb"], d["proba_lstm"], d["hmm"], thr_T, thr_R)
        conf = d["proba_fb"].max(axis=1)

        rep = full_trading_report(
            y_pred=yp, y_actual=d["yt"],
            atr=d["atr"], close=d["close"], high=d["high"], low=d["low"],
            h4_swing_highs=d["h4_sh"], h4_swing_lows=d["h4_sl"],
            index=d["index"], modal=MODAL_PER_TRADE, leverages=LEVERAGE_SIM,
            fee_per_side=FEE_PER_SIDE, slippage=SLIPPAGE_PER_SIDE,
            min_rr=SWING_LABEL_MIN_RR, min_tp_atr=SWING_LABEL_MIN_TP,
            max_sl_atr=SWING_LABEL_MAX_SL,
            tp_fallback_atr=TP_SL_FALLBACK_TP, sl_fallback_atr=TP_SL_FALLBACK_SL,
            max_hold=MAX_HOLDING_BARS, symbol=sym, confidence=conf,
            guardian_enabled=False, trailing_stop_enabled=False, h4_trend=d["h4_tr"],
        )
        lev = rep.get("lev5x", rep)
        tr  = lev.get("trades", [])
        agg["trades"] += len(tr)
        agg["wins"]   += sum(1 for t in tr if t.get("net_pnl", 0) > 0)
        agg["pnl"]    += sum(t.get("net_pnl", 0) for t in tr)

        for t in tr:
            if t.get("direction") == "LONG":
                agg["longs"] += 1
                if t.get("net_pnl", 0) > 0:
                    agg["long_wins"] += 1
            else:
                agg["shorts"] += 1
                if t.get("net_pnl", 0) > 0:
                    agg["short_wins"] += 1

        # Count with-trend trades specifically
        hmm_arr = d["hmm"]
        # TRENDING_UP LONG trades
        agg["up_longs"]   += sum(
            1 for t in tr if t.get("direction") == "LONG"
            and t.get("bar_index") is not None
            and hmm_arr[t["bar_index"]] == REGIME_UP
        ) if tr and "bar_index" in tr[0] else 0

    t, w, p = agg["trades"], agg["wins"], agg["pnl"]
    l, lw   = agg["longs"],  agg["long_wins"]
    s, sw   = agg["shorts"], agg["short_wins"]
    return {
        "trades":        t,
        "wr":            round(w  / max(t,1) * 100, 1),
        "long_wr":       round(lw / max(l,1) * 100, 1),
        "short_wr":      round(sw / max(s,1) * 100, 1),
        "long_pct":      round(l  / max(t,1) * 100, 1),
        "pnl":           round(p, 1),
        "pnl_per_month": round(p / PERIOD_MONTHS, 1),
        "pnl_per_trade": round(p / max(t,1), 3),
    }


# ── Sweep ─────────────────────────────────────────────────────────────────────

# Baseline: fixed 0.50/0.55 semua regime + LSTM (non-directional, dari sweep sebelumnya)
print("  [baseline] fixed 0.50 / 0.55 (non-adaptive) + LSTM soft_pen_hmm ...")
bl = run_config(0.50, 0.50)   # thr_T=thr_R=0.50 → fixed threshold
bl["label"] = "fixed_0.50"
bl["thr_T"] = 0.50; bl["thr_R"] = 0.50
print(f"    -> {bl['trades']:,} trades | WR {bl['wr']:.1f}% | "
      f"PnL ${bl['pnl']:+.0f} | PnL/trade ${bl['pnl_per_trade']:+.3f}\n")

results = [bl]

for thr_T, thr_R in itertools.product(THR_T_VALS, THR_R_VALS):
    label = f"T{int(round(thr_T*100))}_R{int(round(thr_R*100))}"
    print(f"  [{label}] with-trend={thr_T:.2f} | counter/ranging={thr_R:.2f}", end="", flush=True)

    sc = run_config(float(thr_T), float(thr_R))
    sc["label"] = label
    sc["thr_T"] = round(float(thr_T), 2)
    sc["thr_R"] = round(float(thr_R), 2)
    results.append(sc)
    print(f" -> {sc['trades']:,} trades | WR {sc['wr']:.1f}% | "
          f"PnL ${sc['pnl']:+.0f} | PnL/trade ${sc['pnl_per_trade']:+.3f}")


# ── Print scorecard ───────────────────────────────────────────────────────────
df_res = pd.DataFrame(results).sort_values("pnl_per_trade", ascending=False)

print(f"\n{'='*105}")
print(f"  LGBM + LSTM + HMM DIRECTIONAL ADAPTIVE — flatboost_v2")
print(f"  TRENDING_UP : thr_long=thr_T (TURUN)  | thr_short=thr_R (NAIK)")
print(f"  TRENDING_DN : thr_long=thr_R (NAIK)   | thr_short=thr_T (TURUN)")
print(f"  RANGING     : thr_long=thr_R, thr_short=thr_R (KEDUANYA NAIK)")
print(f"  LSTM soft_pen aktif di TRENDING, thr_lstm={LSTM_THR}")
print(f"  ic32 benchmark: 936 trades | WR 62.1% | PnL $207 | PnL/trade $0.221")
print(f"{'='*105}")
print(f"  {'Label':<14} {'thr_T':>6} {'thr_R':>6} {'Trades':>7} {'WR':>6} "
      f"{'LONG%':>6} {'LONG WR':>8} {'SHORT WR':>9} {'PnL':>8} {'PnL/tr':>8}  {'vs ic32':>8}")
print("  " + "-"*105)

for r in df_res.to_dict("records"):
    d_pt = r["pnl_per_trade"] - IC32_BENCH["pnl_per_trade"]
    mark = " <--" if r == df_res.to_dict("records")[0] else ""
    print(f"  {r['label']:<14} {r['thr_T']:>6.2f} {r['thr_R']:>6.2f} "
          f"{int(r['trades']):>7,} {r['wr']:>5.1f}% {r['long_pct']:>5.1f}% "
          f"{r['long_wr']:>7.1f}% {r['short_wr']:>8.1f}% "
          f"{r['pnl']:>+8.1f} {r['pnl_per_trade']:>+8.3f}  {d_pt:>+7.3f}{mark}")

print(f"{'='*105}")

# Heatmap tekstual: PnL/trade per kombinasi (exclude baseline)
sub = [r for r in results if r["label"] != "fixed_0.50"]
df_heat = pd.DataFrame(sub)
pivot = df_heat.pivot(index="thr_T", columns="thr_R", values="pnl_per_trade")
pivot = pivot.sort_index(ascending=False)

print(f"\n  Heatmap PnL/trade (baris=thr_T with-trend, kolom=thr_R counter/ranging):")
print(f"  {'thr_T \\ thr_R':<14}" + "".join(f"  {c:.2f}" for c in pivot.columns))
for idx, row in pivot.iterrows():
    vals = "".join(f"  {v:+.3f}" for v in row.values)
    print(f"  {idx:.2f}{' '*10}{vals}")

# Save
out = {
    "period":    "2026-04-01 to 2026-06-13",
    "lstm_mode": "soft_pen_hmm_t50",
    "n_coins":   len(available),
    "ic32_benchmark": IC32_BENCH,
    "design":    "TRENDING_UP: long=thr_T(down), short=thr_R(up). TRENDING_DOWN: vice versa. RANGING: both=thr_R",
    "results":   df_res.to_dict("records"),
}
out_path = MODEL_DIR / "runs" / FB_RUN / "hmm_directional_sweep.json"
with open(out_path, "w") as f:
    json.dump(out, f, indent=2)
print(f"\n  Saved -> {out_path}")
