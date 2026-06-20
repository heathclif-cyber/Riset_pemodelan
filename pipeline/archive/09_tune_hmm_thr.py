"""
pipeline/09_tune_hmm_thr.py
LGBM + LSTM + HMM Adaptive Threshold (versi benar)

HMM mendeteksi regime → threshold otomatis berubah:
  TRENDING (state 0, 3) : thr_T  — bisa TURUN dari baseline → lebih banyak entry
  RANGING  (state 1, 2) : thr_R  — NAIK dari baseline    → lebih sedikit entry

thr_short = thr_long + 0.05 (fixed offset, konsisten dengan flatboost_v2 best)

LSTM: soft_pen_hmm_t50 — aktif di TRENDING, veto jika arah berlawanan conf >= 0.50

Sweep 2D:
  thr_T_long : 0.38 - 0.54 step 0.02  (9 nilai — BISA LEBIH RENDAH dari baseline 0.50)
  thr_R_long : 0.50 - 0.70 step 0.05  (5 nilai — LEBIH TINGGI dari baseline)
  Constraint : thr_T < thr_R (jika tidak, HMM tidak ada gunanya)
  Total      : valid combinations dari 9 x 5 = 45 kombinasi

Baseline: fixed thr_long=0.50 thr_short=0.55 (best flatboost_v2 dari sweep sebelumnya)
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

logger = setup_logger("09_tune_hmm_thr")

# ── Constants ─────────────────────────────────────────────────────────────────
FB_RUN        = "tb_lgbm_flatboost_v2"
PERIOD_MONTHS = 2.5
IC32_BENCH    = {"trades": 936, "wr": 62.07, "pnl": 207.22, "pnl_per_trade": 0.2214}
SHORT_OFFSET  = 0.05
LSTM_THR      = 0.50
LM            = {"SHORT": 0, "FLAT": 1, "LONG": 2}
TRENDING      = {0, 3}
RANGING       = {1, 2}

# Sweep axes
THR_T_LONGS = np.round(np.arange(0.38, 0.55, 0.02), 2)  # 0.38..0.54 (9 val)
THR_R_LONGS = np.round(np.arange(0.50, 0.71, 0.05), 2)  # 0.50..0.70 (5 val)

# ── Load ──────────────────────────────────────────────────────────────────────
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

# Hitung distribusi HMM di holdout
all_hmm = np.concatenate([d["hmm"] for d in coin_data.values()])
total   = len(all_hmm)
print(f"\nHMM regime distribution (holdout Apr-Jun 2026):")
for state, name in enumerate(["TRENDING_DOWN", "RANGING_LOW", "RANGING_HIGH", "TRENDING_UP"]):
    cnt = (all_hmm == state).sum()
    print(f"  State {state} {name:<18}: {cnt:>7,} bars ({cnt/total*100:.1f}%)")
print()


# ── Prediction builder ────────────────────────────────────────────────────────
def build_predictions(proba_fb, proba_lstm, hmm,
                      thr_T_long, thr_R_long) -> np.ndarray:
    """
    Per bar:
      HMM TRENDING (0,3) → thr_long = thr_T_long, thr_short = thr_T_long + offset
      HMM RANGING  (1,2) → thr_long = thr_R_long, thr_short = thr_R_long + offset

    LSTM soft_pen: aktif di TRENDING — jika LSTM berlawanan conf >= LSTM_THR → veto.
    """
    n  = len(proba_fb)
    yp = np.full(n, 1, np.int32)

    for i in range(n):
        is_trending = hmm[i] in TRENDING

        # HMM pilih threshold
        if is_trending:
            thr_l = thr_T_long
            thr_s = thr_T_long + SHORT_OFFSET
        else:
            thr_l = thr_R_long
            thr_s = thr_R_long + SHORT_OFFSET

        long_conf  = proba_fb[i, 2]
        short_conf = proba_fb[i, 0]

        # LGBM entry signal
        if long_conf >= thr_l:
            signal = 2
        elif short_conf >= thr_s:
            signal = 0
        else:
            continue  # FLAT

        # LSTM soft_pen (hanya TRENDING)
        if is_trending:
            if signal == 2:
                lstm_opp = proba_lstm[i, 0]
            else:
                lstm_opp = proba_lstm[i, 2]
            if lstm_opp >= LSTM_THR:
                entry_conf = long_conf if signal == 2 else short_conf
                adjusted   = entry_conf - lstm_opp * (entry_conf - thr_l + 0.05)
                if adjusted < thr_l:
                    continue  # vetoed

        yp[i] = signal

    return yp


# ── Run one config ────────────────────────────────────────────────────────────
def run_config(thr_T_long, thr_R_long):
    agg = {"trades": 0, "wins": 0, "pnl": 0.0,
           "longs": 0, "long_wins": 0, "shorts": 0, "short_wins": 0}

    for sym, d in coin_data.items():
        yp   = build_predictions(d["proba_fb"], d["proba_lstm"], d["hmm"],
                                 thr_T_long, thr_R_long)
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
                if t.get("net_pnl", 0) > 0: agg["long_wins"] += 1
            else:
                agg["shorts"] += 1
                if t.get("net_pnl", 0) > 0: agg["short_wins"] += 1

    t, w, p = agg["trades"], agg["wins"], agg["pnl"]
    l, lw   = agg["longs"],  agg["long_wins"]
    s, sw   = agg["shorts"], agg["short_wins"]
    return {
        "trades": t, "wr": round(w/max(t,1)*100,1),
        "long_wr": round(lw/max(l,1)*100,1),
        "short_wr": round(sw/max(s,1)*100,1),
        "long_pct": round(l/max(t,1)*100,1),
        "pnl": round(p,1),
        "pnl_per_month": round(p/PERIOD_MONTHS,1),
        "pnl_per_trade": round(p/max(t,1),3),
    }


# ── Sweep ─────────────────────────────────────────────────────────────────────
n_combos = sum(1 for tT, tR in itertools.product(THR_T_LONGS, THR_R_LONGS) if tT < tR)
print(f"Running {n_combos} valid combos (thr_T < thr_R) + 1 baseline...\n")

# Baseline: fixed 0.50/0.55
print("  [baseline] thr_long=0.50 thr_short=0.55 (no HMM adaptive) ...")
bl = run_config(0.50, 0.50)  # thr_T=thr_R → effectively fixed
bl["label"]      = "fixed_0.50"
bl["thr_T_long"] = 0.50; bl["thr_R_long"] = 0.50
print(f"    -> {bl['trades']:,} trades | WR {bl['wr']:.1f}% | "
      f"PnL ${bl['pnl']:+.0f} | PnL/trade ${bl['pnl_per_trade']:+.3f}\n")

results = [bl]

for thr_T, thr_R in itertools.product(THR_T_LONGS, THR_R_LONGS):
    if thr_T >= thr_R:
        continue  # HMM tidak berguna jika trending threshold >= ranging
    label = f"T{int(round(thr_T*100))}_R{int(round(thr_R*100))}"
    print(f"  [{label}] TRENDING={thr_T:.2f}/{thr_T+SHORT_OFFSET:.2f} "
          f"RANGING={thr_R:.2f}/{thr_R+SHORT_OFFSET:.2f}", end="", flush=True)

    sc = run_config(float(thr_T), float(thr_R))
    sc["label"]      = label
    sc["thr_T_long"] = round(float(thr_T), 2)
    sc["thr_R_long"] = round(float(thr_R), 2)
    sc["gap"]        = round(float(thr_R - thr_T), 2)
    results.append(sc)
    print(f" -> {sc['trades']:,} trades | WR {sc['wr']:.1f}% | "
          f"PnL ${sc['pnl']:+.0f} | PnL/trade ${sc['pnl_per_trade']:+.3f}")


# ── Print scorecard ───────────────────────────────────────────────────────────
df_res = pd.DataFrame(results).sort_values("pnl_per_trade", ascending=False)

print(f"\n{'='*105}")
print(f"  LGBM + LSTM + HMM ADAPTIVE THRESHOLD — flatboost_v2")
print(f"  TRENDING (0,3): thr_long=thr_T, thr_short=thr_T+0.05  (threshold TURUN = lebih banyak entry)")
print(f"  RANGING  (1,2): thr_long=thr_R, thr_short=thr_R+0.05  (threshold NAIK  = lebih sedikit entry)")
print(f"  LSTM soft_pen aktif di TRENDING, thr_lstm={LSTM_THR}")
print(f"  ic32 benchmark: 936 trades | WR 62.1% | PnL $207 | PnL/trade $0.221")
print(f"{'='*105}")
print(f"  {'Label':<14} {'thr_T':>5} {'thr_R':>5} {'gap':>5} "
      f"{'Trades':>7} {'WR':>6} {'LongWR':>7} {'ShrtWR':>7} "
      f"{'Long%':>6} {'PnL':>8} {'PnL/tr':>8}  {'vs ic32':>8}")
print("  " + "-"*105)

for r in df_res.to_dict("records"):
    d_pt = r["pnl_per_trade"] - IC32_BENCH["pnl_per_trade"]
    gap  = r.get("gap", 0.0)
    mark = " <--" if r == df_res.to_dict("records")[0] else ""
    print(f"  {r['label']:<14} {r['thr_T_long']:>5.2f} {r['thr_R_long']:>5.2f} {gap:>5.2f} "
          f"{int(r['trades']):>7,} {r['wr']:>5.1f}% {r['long_wr']:>6.1f}% {r['short_wr']:>6.1f}% "
          f"{r['long_pct']:>5.1f}% {r['pnl']:>+8.1f} {r['pnl_per_trade']:>+8.3f}  {d_pt:>+7.3f}{mark}")

# Heatmap PnL/trade
print(f"\n  Heatmap PnL/trade — baris=thr_T (TRENDING), kolom=thr_R (RANGING):")
sub = [r for r in results if r["label"] != "fixed_0.50"]
df_h = pd.DataFrame(sub)
piv  = df_h.pivot(index="thr_T_long", columns="thr_R_long", values="pnl_per_trade")
piv  = piv.sort_index(ascending=False)
print(f"  {'thr_T\\thr_R':<12}" + "".join(f"  {c:.2f}" for c in piv.columns))
for idx, row in piv.iterrows():
    vals = ""
    for v in row.values:
        if pd.isna(v):
            vals += "      -"
        else:
            vals += f"  {v:+.3f}"
    marker = " <- baseline" if abs(idx - 0.50) < 0.01 else ""
    print(f"  {idx:.2f}{'':8}{vals}{marker}")

print(f"\n{'='*105}")

# Save
out = {
    "period": "2026-04-01 to 2026-06-13",
    "design": "TRENDING(0,3)=thr_T | RANGING(1,2)=thr_R | thr_short=thr_long+0.05",
    "lstm":   f"soft_pen_hmm thr={LSTM_THR}",
    "n_coins": len(available),
    "ic32_benchmark": IC32_BENCH,
    "results": df_res.to_dict("records"),
}
out_path = MODEL_DIR / "runs" / FB_RUN / "hmm_thr_sweep.json"
with open(out_path, "w") as f:
    json.dump(out, f, indent=2)
print(f"  Saved -> {out_path}")
