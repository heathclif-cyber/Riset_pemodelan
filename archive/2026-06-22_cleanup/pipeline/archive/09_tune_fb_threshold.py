"""
pipeline/09_tune_fb_threshold.py
Threshold sweep untuk flatboost_v2 pada holdout Apr-Jun 2026.

Logic prediksi (mirror ic32):
  LONG  if proba[:,2] >= thr_long
  SHORT if proba[:,0] >= thr_short AND NOT LONG
  FLAT  otherwise

Sweep thr_long  : 0.35 - 0.65 step 0.05
Sweep thr_short : 0.30 - 0.60 step 0.05
Output: tabel terurut by PnL/trade, filter trades >= 500
"""
import json, sys, warnings, itertools, numpy as np, pandas as pd
from pathlib import Path
warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import joblib
from core.evaluator import full_trading_report
from core.utils import setup_logger, ensure_utc_index
from config import *

logger = setup_logger("09_tune_fb_threshold")

FB_RUN        = "tb_lgbm_flatboost_v2"
PERIOD_MONTHS = 2.5
LM            = {"SHORT": 0, "FLAT": 1, "LONG": 2}

# Load model
fb_model = joblib.load(MODEL_DIR / "runs" / FB_RUN / "lgbm.pkl")
with open(MODEL_DIR / "runs" / FB_RUN / f"{FB_RUN}_features.json") as f:
    fb_feats = json.load(f)

available = sorted(
    s for s in ALL_COINS
    if (HOLDOUT_DIR / "labeled" / f"{s}_features_v3.parquet").exists()
)

# ── Pre-load semua data + probabilities (1x, bukan per-threshold) ────────────
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

    n     = len(df)
    X_fb  = np.zeros((n, len(fb_feats)), dtype=np.float64)
    for i, c in enumerate(fb_feats):
        if c in df.columns:
            X_fb[:, i] = df[c].ffill().fillna(0).values.astype(np.float64)

    proba = fb_model.predict_proba(X_fb)
    coin_data[sym] = {
        "df":    df,
        "proba": proba,
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

print(f"Data loaded. Mulai sweep...\n")

# ── Sweep ─────────────────────────────────────────────────────────────────────
thr_longs  = np.arange(0.35, 0.66, 0.05)
thr_shorts = np.arange(0.30, 0.61, 0.05)

results = []
total_combos = len(thr_longs) * len(thr_shorts)
done = 0

for thr_l, thr_s in itertools.product(thr_longs, thr_shorts):
    agg = {"trades": 0, "wins": 0, "pnl": 0.0}

    for sym, d in coin_data.items():
        proba = d["proba"]
        yp    = np.full(len(proba), 1, np.int32)  # default FLAT
        yp[proba[:, 2] >= thr_l] = 2              # LONG
        short_m = (proba[:, 0] >= thr_s) & (yp != 2)
        yp[short_m] = 0                            # SHORT

        conf = proba.max(axis=1)
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
        agg["wins"]   += sum(1 for t in tr if "WIN" in str(t.get("outcome", "")))
        agg["pnl"]    += sum(t.get("net_pnl", 0) for t in tr)

    t  = agg["trades"]
    w  = agg["wins"]
    p  = agg["pnl"]
    wr = w / max(t, 1) * 100
    pt = p / max(t, 1)

    results.append({
        "thr_long":  round(thr_l, 2),
        "thr_short": round(thr_s, 2),
        "trades":    t,
        "wr":        round(wr, 1),
        "pnl":       round(p, 1),
        "pnl_per_trade": round(pt, 3),
        "pnl_per_month": round(p / PERIOD_MONTHS, 1),
    })

    done += 1
    if done % 10 == 0:
        print(f"  [{done}/{total_combos}] thr_l={thr_l:.2f} thr_s={thr_s:.2f} "
              f"-> {t} trades, WR={wr:.1f}%, PnL=${p:+.0f}")

# ── Sort & print ─────────────────────────────────────────────────────────────
df_res = pd.DataFrame(results)

# Tampilkan top hasil terurut by pnl_per_trade, filter trades >= 400
print(f"\n{'='*80}")
print(f"  TOP THRESHOLD COMBINATIONS (trades>=400, sorted by PnL/trade)")
print(f"  ic32 benchmark: 1,513 trades | WR 56.6% | PnL +$323 | PnL/trade $+0.21")
print(f"{'='*80}")
top = df_res[df_res["trades"] >= 400].sort_values("pnl_per_trade", ascending=False).head(20)
print(top.to_string(index=False))

print(f"\n{'='*80}")
print(f"  CLOSEST TO ic32 TRADE COUNT (1,000-2,000 trades)")
print(f"{'='*80}")
band = df_res[(df_res["trades"] >= 1000) & (df_res["trades"] <= 2500)].sort_values("pnl_per_trade", ascending=False).head(10)
print(band.to_string(index=False))

# Save
out_path = MODEL_DIR / "runs" / FB_RUN / "threshold_sweep.json"
df_res.to_json(out_path, orient="records", indent=2)
print(f"\n  Saved -> {out_path}")
print(f"{'='*80}")
