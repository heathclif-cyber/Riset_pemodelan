"""
pipeline/07_holdout_fb_stack.py
Stack ablation: flatboost_v2 @ thr_long=0.50/thr_short=0.55 + LSTM + HMM + Guardian

Konfigurasi yang diuji:
  1. fb_lgbm_only    : LGBM only (baseline dari threshold sweep)
  2. fb_guardian     : LGBM + guardian_clean_v2
  3. fb_lstm         : LGBM + LSTM hard_consensus (semua regime)
  4. fb_lstm_guard   : LGBM + LSTM + guardian_clean_v2
  5. fb_hmm_lstm     : LGBM + HMM-gated LSTM (LSTM aktif hanya TRENDING)
  6. fb_full         : LGBM + HMM-gated LSTM + guardian_clean_v2

Benchmark: ic32_regime_v1 Apr-Jun 2026 = 936 trades, WR 62.07%, PnL $207, PnL/trade $0.221

Period: Apr 2026 - Jun 2026, 21 koin, $10/5x, swing TP/SL
"""
import json, sys, warnings, numpy as np, pandas as pd
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
from config import GUARDIAN_DYNAMIC_FEATURES as _GDYN

logger = setup_logger("07_holdout_fb_stack")

# ── Constants ─────────────────────────────────────────────────────────────────
FB_RUN      = "tb_lgbm_flatboost_v2"
THR_LONG    = 0.50
THR_SHORT   = 0.55
PERIOD_MONTHS = 2.5

# ic32 Apr-Jun 2026 benchmark (dari holdout_apr_jun26.json)
IC32_BENCH = {"trades": 936, "wr": 62.07, "pnl": 207.22, "pnl_per_trade": 0.2214}

LM = {"SHORT": 0, "FLAT": 1, "LONG": 2}

# HMM TRENDING regimes: TRENDING_DOWN=0, TRENDING_UP=3
TRENDING_REGIMES = {0, 3}

DEVICE = torch.device("cpu")

# ── Load models ───────────────────────────────────────────────────────────────
logger.info("Loading models...")

fb_model = joblib.load(MODEL_DIR / "runs" / FB_RUN / "lgbm.pkl")
with open(MODEL_DIR / "runs" / FB_RUN / f"{FB_RUN}_features.json") as f:
    fb_feats = json.load(f)

# LSTM (ic32 temporal features)
lstm_model  = load_lstm(MODEL_DIR / "lstm_best.pt")
lstm_scaler = joblib.load(MODEL_DIR / "lstm_scaler.pkl")
with open(MODEL_DIR / "feature_cols_lstm_temporal.json") as f:
    lstm_feats = json.load(f)

# Guardian clean_v2 (40 feat)
gd_run   = "ic32_guardian_clean_v2"
gd_model  = joblib.load(MODEL_DIR / "runs" / gd_run / "guardian.pkl")
gd_scaler = joblib.load(MODEL_DIR / "runs" / gd_run / "guardian_scaler.pkl")
with open(MODEL_DIR / "runs" / gd_run / "guardian_feature_cols.json") as f:
    gd_feats = json.load(f)

available = sorted(
    s for s in ALL_COINS
    if (HOLDOUT_DIR / "labeled" / f"{s}_features_v3.parquet").exists()
)
logger.info(f"Models loaded. Coins: {len(available)}")

# ── Pre-load all coin data (1x) ───────────────────────────────────────────────
print(f"\nPre-loading {len(available)} koin...")

coin_data = {}
for sym in available:
    df = pd.read_parquet(HOLDOUT_DIR / "labeled" / f"{sym}_features_v3.parquet")
    df = ensure_utc_index(df).sort_index()

    # HMM regime
    rp  = HOLDOUT_DIR / "labeled" / f"{sym}_regime_h1.parquet"
    hmm = np.full(len(df), 1, np.int32)  # default RANGING_LOW_VOL
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
    proba_fb = fb_model.predict_proba(X_fb)  # (N, 3)

    # Prediction dengan threshold 0.50/0.55
    yp_base = np.full(n, 1, np.int32)
    yp_base[proba_fb[:, 2] >= THR_LONG]  = 2
    short_m = (proba_fb[:, 0] >= THR_SHORT) & (yp_base != 2)
    yp_base[short_m] = 0

    # LSTM probabilities
    X_lstm = np.zeros((n, len(lstm_feats)), dtype=np.float64)
    for i, c in enumerate(lstm_feats):
        if c in df.columns:
            X_lstm[:, i] = df[c].ffill().fillna(0).values.astype(np.float64)
    proba_lstm = get_lstm_proba(lstm_model, lstm_scaler, X_lstm, n)  # (N, 3)

    # Guardian static features (HANYA fitur statis — dynamic ditambah oleh evaluator per-bar)
    gd_static_feats = [c for c in gd_feats if c not in set(_GDYN)]
    X_gd = compute_guardian_static_array(df, gd_static_feats)  # (N, n_gd_static)

    coin_data[sym] = {
        "df":        df,
        "proba_fb":  proba_fb,
        "yp_base":   yp_base,
        "proba_lstm": proba_lstm,
        "hmm":       hmm,
        "X_gd":      X_gd,
        "close":     df["close"].values,
        "high":      df["high"].values,
        "low":       df["low"].values,
        "atr":       df["atr_14_h1"].values,
        "h4_sh": df["h4_swing_high"].values if "h4_swing_high" in df.columns else np.full(n, np.nan),
        "h4_sl": df["h4_swing_low"].values  if "h4_swing_low"  in df.columns else np.full(n, np.nan),
        "h4_tr": df["h4_trend"].values      if "h4_trend"      in df.columns else None,
        "yt":    df["label"].map(LM).values.astype(np.int32),
        "index": df.index,
    }

print(f"Data loaded. Running stack ablation...\n")


# ── Helper: apply LSTM hard_consensus filter ─────────────────────────────────
def apply_lstm_filter(yp: np.ndarray, proba_fb: np.ndarray, proba_lstm: np.ndarray,
                      hmm: np.ndarray = None, hmm_gate: bool = False) -> np.ndarray:
    """
    Hard-consensus LSTM veto.
    LONG dan SHORT masing-masing punya threshold terpisah di LGBM; untuk LSTM
    kita ambil argmax sebagai direktori LSTM.

    Veto: jika LGBM=LONG dan LSTM=SHORT (atau sebaliknya) → set FLAT.
    Jika hmm_gate=True: hanya veto saat regime TRENDING (hmm in {0,3}).
    """
    yp = yp.copy()
    lstm_pred = proba_lstm.argmax(axis=1).astype(np.int32)

    for i in range(len(yp)):
        if yp[i] == 1:  # LGBM already FLAT, skip
            continue
        # Check HMM gate
        if hmm_gate and hmm is not None:
            if hmm[i] not in TRENDING_REGIMES:
                continue  # RANGING: LSTM silent, keep LGBM
        # Veto if strictly opposite direction
        if yp[i] == 2 and lstm_pred[i] == 0:  # LGBM LONG, LSTM SHORT
            yp[i] = 1
        elif yp[i] == 0 and lstm_pred[i] == 2:  # LGBM SHORT, LSTM LONG
            yp[i] = 1
    return yp


# ── Helper: run one configuration ────────────────────────────────────────────
def run_config(yp_fn, guardian_enabled: bool, label: str) -> dict:
    """
    yp_fn: callable(sym, d) -> np.ndarray of int32 predictions
    Returns aggregate scorecard dict.
    """
    agg = {"trades": 0, "wins": 0, "pnl": 0.0,
           "longs": 0, "long_wins": 0, "shorts": 0, "short_wins": 0}

    for sym, d in coin_data.items():
        yp = yp_fn(sym, d)
        conf = np.abs(d["proba_fb"].max(axis=1))  # confidence proxy

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
            guardian_enabled=guardian_enabled,
            guardian_model=gd_model if guardian_enabled else None,
            guardian_scaler=gd_scaler if guardian_enabled else None,
            X_guardian=d["X_gd"] if guardian_enabled else None,
            guardian_exit_threshold=GUARDIAN_EXIT_THRESHOLD,
            trailing_stop_enabled=False, h4_trend=d["h4_tr"],
        )
        lev = rep.get("lev5x", rep)
        tr  = lev.get("trades", [])
        agg["trades"] += len(tr)
        agg["wins"]   += sum(1 for t in tr if "WIN" in str(t.get("outcome", "")))
        agg["pnl"]    += sum(t.get("net_pnl", 0) for t in tr)
        longs  = [t for t in tr if t.get("direction") == "LONG"]
        shorts = [t for t in tr if t.get("direction") == "SHORT"]
        agg["longs"]      += len(longs)
        agg["long_wins"]  += sum(1 for t in longs if "WIN" in str(t.get("outcome", "")))
        agg["shorts"]     += len(shorts)
        agg["short_wins"] += sum(1 for t in shorts if "WIN" in str(t.get("outcome", "")))

    t, w, p = agg["trades"], agg["wins"], agg["pnl"]
    l, lw   = agg["longs"], agg["long_wins"]
    s, sw   = agg["shorts"], agg["short_wins"]
    return {
        "label":     label,
        "trades":    t,
        "wr":        round(w / max(t, 1) * 100, 1),
        "long_wr":   round(lw / max(l, 1) * 100, 1),
        "short_wr":  round(sw / max(s, 1) * 100, 1),
        "long_pct":  round(l / max(t, 1) * 100, 1),
        "pnl":       round(p, 1),
        "pnl_per_month": round(p / PERIOD_MONTHS, 1),
        "pnl_per_trade": round(p / max(t, 1), 3),
    }


# ── Run all configurations ────────────────────────────────────────────────────

configs = [
    # 1. LGBM only (baseline)
    (lambda sym, d: d["yp_base"].copy(), False, "fb_lgbm_only"),
    # 2. LGBM + Guardian
    (lambda sym, d: d["yp_base"].copy(), True,  "fb_guardian"),
    # 3. LGBM + LSTM (all regimes)
    (lambda sym, d: apply_lstm_filter(d["yp_base"], d["proba_fb"], d["proba_lstm"],
                                      hmm_gate=False),
     False, "fb_lstm"),
    # 4. LGBM + LSTM + Guardian
    (lambda sym, d: apply_lstm_filter(d["yp_base"], d["proba_fb"], d["proba_lstm"],
                                      hmm_gate=False),
     True,  "fb_lstm_guard"),
    # 5. LGBM + HMM-gated LSTM
    (lambda sym, d: apply_lstm_filter(d["yp_base"], d["proba_fb"], d["proba_lstm"],
                                      hmm=d["hmm"], hmm_gate=True),
     False, "fb_hmm_lstm"),
    # 6. LGBM + HMM-gated LSTM + Guardian (FULL STACK)
    (lambda sym, d: apply_lstm_filter(d["yp_base"], d["proba_fb"], d["proba_lstm"],
                                      hmm=d["hmm"], hmm_gate=True),
     True,  "fb_full"),
]

results = []
for yp_fn, use_gd, label in configs:
    print(f"  Running {label}...")
    sc = run_config(yp_fn, use_gd, label)
    results.append(sc)
    print(f"    -> {sc['trades']} trades, WR {sc['wr']:.1f}%, PnL ${sc['pnl']:+.0f}, "
          f"PnL/trade ${sc['pnl_per_trade']:+.3f}")


# ── Print scorecard ───────────────────────────────────────────────────────────
def fmt(val, typ="s"):
    if typ == "pct":  return f"{val:.1f}%"
    if typ == "usd":  return f"${val:+.0f}"
    if typ == "pt":   return f"${val:+.3f}"
    if typ == "n":    return f"{val:,}"
    return str(val)

cols = [
    ("Trades",      "trades",        "n"),
    ("WR",          "wr",            "pct"),
    ("LONG WR",     "long_wr",       "pct"),
    ("SHORT WR",    "short_wr",      "pct"),
    ("LONG%",       "long_pct",      "pct"),
    ("Net PnL",     "pnl",           "usd"),
    ("PnL/mo",      "pnl_per_month", "usd"),
    ("PnL/trade",   "pnl_per_trade", "pt"),
]

W = 16
print(f"\n{'='*90}")
print(f"  STACK ABLATION — flatboost_v2 @ thr_long={THR_LONG}/thr_short={THR_SHORT}")
print(f"  Holdout: Apr-Jun 2026 ({len(available)} koin, ~{PERIOD_MONTHS} bln, $10/5x)")
print(f"  ic32 benchmark (Apr-Jun 2026): 936 trades | WR 62.1% | PnL $207 | PnL/trade $0.221")
print(f"{'='*90}")

header = f"  {'Metrik':<18}" + "".join(f"  {r['label']:<{W}}" for r in results)
print(header)
print("  " + "-" * (18 + (W + 2) * len(results)))

for label, key, typ in cols:
    row = f"  {label:<18}"
    for r in results:
        v = fmt(r[key], typ)
        row += f"  {v:<{W}}"
    print(row)

print(f"{'='*90}")

# vs ic32 summary
print(f"\n  vs ic32 (Apr-Jun 2026):")
for r in results:
    d_pnl = r["pnl"] - IC32_BENCH["pnl"]
    d_pt  = r["pnl_per_trade"] - IC32_BENCH["pnl_per_trade"]
    print(f"  {r['label']:<20} | trades={r['trades']:>5} | WR={r['wr']:.1f}% | "
          f"PnL/trade=${r['pnl_per_trade']:+.3f} ({d_pt:+.3f} vs ic32)")

# ── Save ─────────────────────────────────────────────────────────────────────
out_path = MODEL_DIR / "runs" / FB_RUN / "stack_ablation_apr_jun26.json"
out = {
    "period":  "2026-04-01 to 2026-06-13",
    "thr_long": THR_LONG, "thr_short": THR_SHORT,
    "n_coins":  len(available),
    "ic32_benchmark": IC32_BENCH,
    "results":  results,
    "note": "flatboost_v2 stack ablation: LSTM hard_consensus + HMM gate + guardian_clean_v2",
}
with open(out_path, "w") as f:
    json.dump(out, f, indent=2)
print(f"\n  Saved -> {out_path}")
print(f"{'='*90}")
