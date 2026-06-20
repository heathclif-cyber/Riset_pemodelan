"""
pipeline/09_tune_hmm_adaptive.py
LGBM + LSTM + HMM Adaptive Threshold sweep untuk flatboost_v2.

HMM sebagai adaptive threshold:
  - TRENDING (regime 0, 3) : pakai thr_T_long / thr_T_short (lebih rendah = lebih banyak entry)
  - RANGING  (regime 1, 2) : pakai thr_R_long / thr_R_short (lebih tinggi = lebih selektif)

LSTM config terbaik dari 09_tune_fb_lstm.py:
  soft_pen_hmm_t50 — penalty aktif saat TRENDING, thr_lstm=0.50

Sweep 2D:
  thr_T (base TRENDING threshold): 0.40, 0.45, 0.50, 0.55  (SHORT = thr + 0.05)
  thr_R (base RANGING  threshold): 0.55, 0.60, 0.65, 0.70  (SHORT = thr + 0.05)
  Total: 4 x 4 = 16 kombinasi

Baseline:
  fixed_0.50_0.55 — thr_long=0.50, thr_short=0.55 untuk semua regime (no HMM adaptive)

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

logger = setup_logger("09_tune_hmm_adaptive")

# ── Constants ─────────────────────────────────────────────────────────────────
FB_RUN        = "tb_lgbm_flatboost_v2"
PERIOD_MONTHS = 2.5
IC32_BENCH    = {"trades": 936, "wr": 62.07, "pnl": 207.22, "pnl_per_trade": 0.2214}

LM            = {"SHORT": 0, "FLAT": 1, "LONG": 2}
TRENDING_RGMS = {0, 3}   # TRENDING_DOWN=0, TRENDING_UP=3
RANGING_RGMS  = {1, 2}   # RANGING_LOW_VOL=1, RANGING_HIGH_VOL=2
DEVICE        = torch.device("cpu")

# LSTM config terbaik: soft_pen_hmm_t50
LSTM_THR      = 0.50

# ── Sweep axes ────────────────────────────────────────────────────────────────
SHORT_OFFSET  = 0.05   # thr_short = thr_long + SHORT_OFFSET

# Trending: lebih agresif (threshold lebih rendah = lebih banyak entry, LSTM akan filter)
THR_T_LONGS  = [0.40, 0.45, 0.50, 0.55]

# Ranging: lebih selektif (threshold lebih tinggi = LGBM harus sangat yakin)
THR_R_LONGS  = [0.55, 0.60, 0.65, 0.70]

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
logger.info(f"Models loaded. Coins: {len(available)}")

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

    # LGBM probabilities (semua bar)
    X_fb = np.zeros((n, len(fb_feats)), dtype=np.float64)
    for i, c in enumerate(fb_feats):
        if c in df.columns:
            X_fb[:, i] = df[c].ffill().fillna(0).values.astype(np.float64)
    proba_fb = fb_model.predict_proba(X_fb)

    # LSTM probabilities
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
print(f"Data loaded. Running sweep...\n")


# ── Prediction builder ────────────────────────────────────────────────────────
def build_predictions(proba_fb: np.ndarray, proba_lstm: np.ndarray,
                      hmm: np.ndarray,
                      thr_T_long: float, thr_T_short: float,
                      thr_R_long: float, thr_R_short: float,
                      use_hmm_adaptive: bool = True) -> np.ndarray:
    """
    Step 1 — LGBM HMM-adaptive threshold:
      TRENDING bar  : thr_T_long / thr_T_short
      RANGING  bar  : thr_R_long / thr_R_short  (atau fixed jika use_hmm_adaptive=False)

    Step 2 — LSTM soft_pen (aktif hanya di TRENDING):
      Jika LGBM fires LONG dan LSTM SHORT conf >= LSTM_THR → kurangi LGBM long_conf
      Jika LGBM fires SHORT dan LSTM LONG conf >= LSTM_THR → kurangi LGBM short_conf
      Jika setelah penalty conf turun di bawah threshold → FLAT
    """
    n  = len(proba_fb)
    yp = np.full(n, 1, np.int32)  # default FLAT

    for i in range(n):
        is_trending = hmm[i] in TRENDING_RGMS

        # Pilih threshold berdasarkan regime
        if use_hmm_adaptive and not is_trending:
            thr_l = thr_R_long
            thr_s = thr_R_short
        else:
            thr_l = thr_T_long
            thr_s = thr_T_short

        long_conf  = proba_fb[i, 2]
        short_conf = proba_fb[i, 0]

        # LGBM entry signal
        if long_conf >= thr_l:
            signal = 2   # LONG
        elif short_conf >= thr_s:
            signal = 0   # SHORT
        else:
            yp[i] = 1    # FLAT — tidak masuk
            continue

        # LSTM soft_pen (hanya TRENDING)
        if is_trending:
            if signal == 2:   # LONG → check LSTM SHORT confidence
                lstm_opp = proba_lstm[i, 0]
                if lstm_opp >= LSTM_THR:
                    # Kurangi long_conf proportional to lstm_opp
                    adjusted = long_conf - lstm_opp * (long_conf - thr_l + 0.05)
                    if adjusted < thr_l:
                        yp[i] = 1
                        continue
            else:             # SHORT → check LSTM LONG confidence
                lstm_opp = proba_lstm[i, 2]
                if lstm_opp >= LSTM_THR:
                    adjusted = short_conf - lstm_opp * (short_conf - thr_s + 0.05)
                    if adjusted < thr_s:
                        yp[i] = 1
                        continue

        yp[i] = signal

    return yp


# ── Run one config ────────────────────────────────────────────────────────────
def run_config(thr_T_long, thr_T_short, thr_R_long, thr_R_short,
               use_hmm_adaptive=True) -> dict:
    agg = {"trades": 0, "wins": 0, "pnl": 0.0,
           "longs": 0, "long_wins": 0, "shorts": 0, "short_wins": 0}

    for sym, d in coin_data.items():
        yp   = build_predictions(
            d["proba_fb"], d["proba_lstm"], d["hmm"],
            thr_T_long, thr_T_short, thr_R_long, thr_R_short,
            use_hmm_adaptive=use_hmm_adaptive,
        )
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
        longs  = [t for t in tr if t.get("direction") == "LONG"]
        shorts = [t for t in tr if t.get("direction") == "SHORT"]
        agg["longs"]      += len(longs)
        agg["long_wins"]  += sum(1 for t in longs if t.get("net_pnl", 0) > 0)
        agg["shorts"]     += len(shorts)
        agg["short_wins"] += sum(1 for t in shorts if t.get("net_pnl", 0) > 0)

    t, w, p = agg["trades"], agg["wins"], agg["pnl"]
    l, lw   = agg["longs"], agg["long_wins"]
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

# Baseline 1: fixed threshold, dengan LSTM soft_pen_hmm
print("  [baseline] fixed T0.50/S0.55 + LSTM soft_pen_hmm ...")
bl_lstm = run_config(0.50, 0.55, 0.50, 0.55, use_hmm_adaptive=False)
bl_lstm["label"] = "T050_R050+lstm"
bl_lstm["thr_T_long"] = 0.50; bl_lstm["thr_T_short"] = 0.55
bl_lstm["thr_R_long"] = 0.50; bl_lstm["thr_R_short"] = 0.55
print(f"    -> {bl_lstm['trades']} trades | WR {bl_lstm['wr']:.1f}% | "
      f"PnL/trade ${bl_lstm['pnl_per_trade']:+.3f}")

results = [bl_lstm]

# 16 HMM-adaptive combos
for thr_T, thr_R in itertools.product(THR_T_LONGS, THR_R_LONGS):
    thr_T_s = round(thr_T + SHORT_OFFSET, 2)
    thr_R_s = round(thr_R + SHORT_OFFSET, 2)
    label   = f"T{int(thr_T*100)}_R{int(thr_R*100)}"
    print(f"  [{label}] thr_T_long={thr_T} thr_T_short={thr_T_s} "
          f"| thr_R_long={thr_R} thr_R_short={thr_R_s}", end="", flush=True)

    sc = run_config(thr_T, thr_T_s, thr_R, thr_R_s, use_hmm_adaptive=True)
    sc["label"]      = label
    sc["thr_T_long"] = thr_T
    sc["thr_T_short"]= thr_T_s
    sc["thr_R_long"] = thr_R
    sc["thr_R_short"]= thr_R_s
    results.append(sc)
    print(f" -> {sc['trades']} trades | WR {sc['wr']:.1f}% | "
          f"PnL ${sc['pnl']:+.0f} | PnL/trade ${sc['pnl_per_trade']:+.3f}")


# ── Print scorecard ───────────────────────────────────────────────────────────
df_res = pd.DataFrame(results).sort_values("pnl_per_trade", ascending=False)

print(f"\n{'='*100}")
print(f"  LGBM + LSTM + HMM Adaptive Threshold — flatboost_v2")
print(f"  LSTM: soft_pen_hmm_t50 (penalty hanya TRENDING, thr=0.50)")
print(f"  HMM: TRENDING(0,3)=thr_T | RANGING(1,2)=thr_R | short=long+0.05")
print(f"  Holdout Apr-Jun 2026 ({len(available)} koin, {PERIOD_MONTHS} bln, $10/5x, no Guardian)")
print(f"  ic32 benchmark: 936 trades | WR 62.1% | PnL $207 | PnL/trade $0.221")
print(f"{'='*100}")
print(f"  {'Label':<18} {'thr_TL':>6} {'thr_TS':>6} {'thr_RL':>6} {'thr_RS':>6} "
      f"{'Trades':>7} {'WR':>6} {'LONG%':>6} {'PnL':>8} {'PnL/trade':>10}  {'vs ic32':>8}")
print("  " + "-"*100)

for r in df_res.to_dict("records"):
    d_pt = r["pnl_per_trade"] - IC32_BENCH["pnl_per_trade"]
    mark = " <--" if r == df_res.to_dict("records")[0] else ""
    print(f"  {r['label']:<18} "
          f"{r['thr_T_long']:>6.2f} {r['thr_T_short']:>6.2f} "
          f"{r['thr_R_long']:>6.2f} {r['thr_R_short']:>6.2f} "
          f"{int(r['trades']):>7,} {r['wr']:>5.1f}% {r['long_pct']:>5.1f}% "
          f"{r['pnl']:>+8.1f} {r['pnl_per_trade']:>+10.3f}  {d_pt:>+7.3f}{mark}")

print(f"{'='*100}")

# Tampilkan juga per-WR segmen
best = df_res.iloc[0]
print(f"\n  Best HMM-adaptive: {best['label']} | "
      f"T_long={best['thr_T_long']:.2f}/T_short={best['thr_T_short']:.2f} "
      f"R_long={best['thr_R_long']:.2f}/R_short={best['thr_R_short']:.2f}")
print(f"    -> {int(best['trades'])} trades | WR {best['wr']:.1f}% | "
      f"LONG WR {best['long_wr']:.1f}% | SHORT WR {best['short_wr']:.1f}%")
print(f"    -> PnL ${best['pnl']:+.1f} | PnL/mo ${best['pnl_per_month']:+.1f} "
      f"| PnL/trade ${best['pnl_per_trade']:+.3f}")

# ── Save ─────────────────────────────────────────────────────────────────────
out = {
    "period":    "2026-04-01 to 2026-06-13",
    "lstm_mode": "soft_pen_hmm_t50",
    "short_offset": SHORT_OFFSET,
    "n_coins":   len(available),
    "ic32_benchmark": IC32_BENCH,
    "results":   df_res.to_dict("records"),
    "note": "HMM-adaptive threshold: TRENDING=thr_T, RANGING=thr_R, LSTM soft_pen hanya TRENDING",
}
out_path = MODEL_DIR / "runs" / FB_RUN / "hmm_adaptive_sweep.json"
with open(out_path, "w") as f:
    json.dump(out, f, indent=2)
print(f"\n  Saved -> {out_path}")
print(f"{'='*100}")
