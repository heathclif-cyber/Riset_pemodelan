"""
pipeline/09_tune_fb_lstm.py
Sweep konfigurasi LGBM + LSTM untuk flatboost_v2 @ thr_long=0.50/thr_short=0.55

Dimensi yang di-sweep:
  A. LSTM veto threshold  : 0.40 - 0.70 step 0.05 (seberapa yakin LSTM harus untuk veto)
  B. HMM gate             : off (semua regime) vs on (TRENDING_DOWN/UP only)
  C. LSTM mode            : "veto" vs "soft_pen"
       veto     — LSTM opposite + conf>=thr_lstm → set FLAT
       soft_pen — LSTM opposite → kurangi LGBM probability proportional to lstm_conf;
                  jika turun di bawah threshold → FLAT

  Total:  7 threshold × 2 HMM × 2 mode = 28 kombinasi
  + baseline: lgbm_only (no LSTM)

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
from pipeline.backtest_utils import get_lstm_proba, compute_guardian_static_array
from config import *

logger = setup_logger("09_tune_fb_lstm")

# ── Constants ─────────────────────────────────────────────────────────────────
FB_RUN        = "tb_lgbm_flatboost_v2"
THR_LONG      = 0.50
THR_SHORT     = 0.55
PERIOD_MONTHS = 2.5
IC32_BENCH    = {"trades": 936, "wr": 62.07, "pnl": 207.22, "pnl_per_trade": 0.2214}

LM             = {"SHORT": 0, "FLAT": 1, "LONG": 2}
TRENDING_RGMS  = {0, 3}  # TRENDING_DOWN=0, TRENDING_UP=3
DEVICE         = torch.device("cpu")

# Sweep axes
LSTM_THRESHOLDS = np.round(np.arange(0.40, 0.71, 0.05), 2)
HMM_GATES       = [False]  # user request: no HMM for now, only LGBM + LSTM
LSTM_MODES      = ["veto", "soft_pen", "hard_consensus"]  # added hard_consensus for more scenarios

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

# ── Pre-load all coin data ────────────────────────────────────────────────────
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

    # LGBM flatboost_v2 probabilities
    X_fb = np.zeros((n, len(fb_feats)), dtype=np.float64)
    for i, c in enumerate(fb_feats):
        if c in df.columns:
            X_fb[:, i] = df[c].ffill().fillna(0).values.astype(np.float64)
    proba_fb = fb_model.predict_proba(X_fb)

    # Base prediction (0.50/0.55 threshold)
    yp_base = np.full(n, 1, np.int32)
    yp_base[proba_fb[:, 2] >= THR_LONG]  = 2
    yp_base[(proba_fb[:, 0] >= THR_SHORT) & (yp_base != 2)] = 0

    # LSTM probabilities
    X_lstm = np.zeros((n, len(lstm_feats)), dtype=np.float64)
    for i, c in enumerate(lstm_feats):
        if c in df.columns:
            X_lstm[:, i] = df[c].ffill().fillna(0).values.astype(np.float64)
    proba_lstm = get_lstm_proba(lstm_model, lstm_scaler, X_lstm, n)

    coin_data[sym] = {
        "df":          df,
        "proba_fb":    proba_fb,
        "yp_base":     yp_base,
        "proba_lstm":  proba_lstm,
        "hmm":         hmm,
        "close":       df["close"].values,
        "high":        df["high"].values,
        "low":         df["low"].values,
        "atr":         df["atr_14_h1"].values,
        "h4_sh": df["h4_swing_high"].values if "h4_swing_high" in df.columns else np.full(n, np.nan),
        "h4_sl": df["h4_swing_low"].values  if "h4_swing_low"  in df.columns else np.full(n, np.nan),
        "h4_tr": df["h4_trend"].values      if "h4_trend"      in df.columns else None,
        "yt":    df["label"].map(LM).values.astype(np.int32),
        "index": df.index,
    }
print(f"Data loaded.\n")


# ── LSTM filter functions ─────────────────────────────────────────────────────

def apply_veto(yp_base: np.ndarray, proba_lstm: np.ndarray,
               thr_lstm: float, hmm: np.ndarray, hmm_gate: bool) -> np.ndarray:
    """
    Hard veto: LGBM fires LONG/SHORT → jika LSTM prediksi arah berlawanan
    dengan conf >= thr_lstm → ubah ke FLAT.
    Jika hmm_gate=True: hanya veto saat regime TRENDING (0 atau 3).
    """
    yp = yp_base.copy()
    for i in range(len(yp)):
        if yp[i] == 1:
            continue
        if hmm_gate and hmm[i] not in TRENDING_RGMS:
            continue
        # Opposite direction confidence
        if yp[i] == 2:   # LGBM=LONG → LSTM SHORT confidence
            lstm_opp_conf = proba_lstm[i, 0]
        else:              # LGBM=SHORT → LSTM LONG confidence
            lstm_opp_conf = proba_lstm[i, 2]
        if lstm_opp_conf >= thr_lstm:
            yp[i] = 1  # veto
    return yp


def apply_soft_pen(yp_base: np.ndarray, proba_fb: np.ndarray, proba_lstm: np.ndarray,
                   thr_lstm: float, hmm: np.ndarray, hmm_gate: bool) -> np.ndarray:
    """
    Soft penalty: LSTM opposite → kurangi LGBM confidence sebesar lstm_opp_conf.
    Jika LGBM confidence turun di bawah entry threshold → FLAT.
    Jika hmm_gate=True: hanya apply penalty saat TRENDING.
    """
    yp = yp_base.copy()
    for i in range(len(yp)):
        if yp[i] == 1:
            continue
        if hmm_gate and hmm[i] not in TRENDING_RGMS:
            continue
        if yp[i] == 2:
            entry_conf = proba_fb[i, 2]
            lstm_opp   = proba_lstm[i, 0]
            eff_thr    = THR_LONG
        else:
            entry_conf = proba_fb[i, 0]
            lstm_opp   = proba_lstm[i, 2]
            eff_thr    = THR_SHORT
        if lstm_opp >= thr_lstm:
            # Reduce confidence by LSTM opposite confidence
            adjusted = entry_conf - lstm_opp * (entry_conf - eff_thr + 0.05)
            if adjusted < eff_thr:
                yp[i] = 1
    return yp


# ── Run one config ────────────────────────────────────────────────────────────
def run_one(yp_arr_fn) -> dict:
    agg = {"trades": 0, "wins": 0, "pnl": 0.0}
    for sym, d in coin_data.items():
        yp   = yp_arr_fn(sym, d)
        conf = d["proba_fb"].max(axis=1)
        rep  = full_trading_report(
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
        agg["wins"]   += sum(1 for t in tr if "WIN" in str(t.get("outcome", "")) or t.get("net_pnl", 0) > 0)
        agg["pnl"]    += sum(t.get("net_pnl", 0) for t in tr)
    t, w, p = agg["trades"], agg["wins"], agg["pnl"]
    return {
        "trades": t,
        "wr":     round(w / max(t, 1) * 100, 1),
        "pnl":    round(p, 1),
        "pnl_per_trade": round(p / max(t, 1), 3),
    }


# ── Sweep ─────────────────────────────────────────────────────────────────────
print("Running sweep (28 combos + 1 baseline)...\n")

# Baseline: LGBM only
print("  [baseline] lgbm_only ...")
baseline = run_one(lambda sym, d: d["yp_base"].copy())
baseline["label"] = "lgbm_only"
print(f"    -> {baseline['trades']} trades | WR {baseline['wr']:.1f}% | "
      f"PnL ${baseline['pnl']:+.0f} | PnL/trade ${baseline['pnl_per_trade']:+.3f}")

results = [baseline]

for mode, hmm_gate, thr in itertools.product(LSTM_MODES, HMM_GATES, LSTM_THRESHOLDS):
    gate_str = "hmm" if hmm_gate else "all"
    label    = f"{mode}_{gate_str}_t{int(thr*100)}"
    print(f"  [{label}]", end="", flush=True)

    if mode == "veto":
        fn = lambda sym, d, _thr=thr, _gate=hmm_gate: apply_veto(
            d["yp_base"], d["proba_lstm"], _thr, d["hmm"], _gate)
    else:
        fn = lambda sym, d, _thr=thr, _gate=hmm_gate: apply_soft_pen(
            d["yp_base"], d["proba_fb"], d["proba_lstm"], _thr, d["hmm"], _gate)

    sc = run_one(fn)
    sc["label"] = label
    sc["mode"]  = mode
    sc["hmm_gate"] = hmm_gate
    sc["thr_lstm"] = round(float(thr), 2)
    results.append(sc)
    print(f" -> {sc['trades']} trades | WR {sc['wr']:.1f}% | "
          f"PnL ${sc['pnl']:+.0f} | PnL/trade ${sc['pnl_per_trade']:+.3f}")


# ── Print scorecard ───────────────────────────────────────────────────────────
df_res = pd.DataFrame(results).fillna("-")
df_res = df_res.sort_values("pnl_per_trade", ascending=False)

print(f"\n{'='*90}")
print(f"  LGBM + LSTM SWEEP — flatboost_v2 @ thr_long={THR_LONG}/thr_short={THR_SHORT}")
print(f"  Holdout Apr-Jun 2026 ({len(available)} koin, {PERIOD_MONTHS} bln, $10/5x, no Guardian)")
print(f"  ic32 benchmark: 936 trades | WR 62.1% | PnL $207 | PnL/trade $0.221")
print(f"{'='*90}")
print(f"  {'Label':<25} {'Trades':>7} {'WR':>7} {'Net PnL':>9} {'PnL/trade':>10}  {'vs ic32':>8}")
print("  " + "-"*75)
for r in df_res.to_dict("records"):
    d_pt  = r['pnl_per_trade'] - IC32_BENCH['pnl_per_trade']
    mark  = " <-- BEST" if r == df_res.to_dict("records")[0] else ""
    print(f"  {r['label']:<25} {int(r['trades']):>7,} {r['wr']:>6.1f}% "
          f"{r['pnl']:>+9.1f} {r['pnl_per_trade']:>+10.3f}  {d_pt:>+7.3f}{mark}")

print(f"{'='*90}")

# ── Save ─────────────────────────────────────────────────────────────────────
out = {
    "period": "2026-04-01 to 2026-06-13",
    "thr_long": THR_LONG, "thr_short": THR_SHORT,
    "n_coins":  len(available),
    "ic32_benchmark": IC32_BENCH,
    "results":  df_res.to_dict("records"),
    "note": "LGBM+LSTM sweep: veto vs soft_pen, all-regime vs HMM-gated, thr 0.40-0.70",
}
out_path = MODEL_DIR / "runs" / FB_RUN / "lstm_sweep.json"
with open(out_path, "w") as f:
    json.dump(out, f, indent=2)
print(f"\n  Saved -> {out_path}")
