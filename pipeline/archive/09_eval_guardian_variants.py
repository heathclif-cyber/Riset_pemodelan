"""
pipeline/09_eval_guardian_variants.py
Bandingkan 3 Guardian variants pada config T50_R55 (thr_T=0.50, thr_R=0.55)

Variants:
  profit_v1    -- 18 static + 7 dynamic = 25 feat (feature_cols_v2.json)
  fb_v1        -- 27 static + 7 dynamic = 34 feat (Opsi A: flatboost_v2)
  fb_union_v1  -- 36 static + 7 dynamic = 43 feat (Opsi B: union)

Base: T50_R55 + LSTM veto (thr_T=0.50, thr_R=0.55, short_off=0.05)
Guardian params: exit_thr=0.70, min_hold=3
"""
import json, sys, warnings, numpy as np, pandas as pd
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

logger = setup_logger("09_eval_guardian_variants")

FB_RUN    = "tb_lgbm_flatboost_v2"
THR_T     = 0.50
THR_R     = 0.55
SHORT_OFF = 0.05
LSTM_THR  = 0.50
TRENDING  = {0, 3}
LM        = {"SHORT": 0, "FLAT": 1, "LONG": 2}
PERIOD    = 2.5

GD_EXIT_THR = 0.70
GD_MIN_HOLD = 3

IC32 = {"trades": 936, "wr": 62.07, "pf": 2.54, "pnl": 207.22}

GUARDIAN_VARIANTS = [
    ("profit_v1",    "tb_guardian_profit_v1"),
    ("fb_v1",        "tb_guardian_fb_v1"),
    ("fb_union_v1",  "tb_guardian_fb_union_v1"),
]

# ── Load entry models ─────────────────────────────────────────────────────────
fb_model = joblib.load(MODEL_DIR / "runs" / FB_RUN / "lgbm.pkl")
with open(MODEL_DIR / "runs" / FB_RUN / f"{FB_RUN}_features.json") as f:
    fb_feats = json.load(f)

lstm_model  = load_lstm(MODEL_DIR / "lstm_best.pt")
lstm_scaler = joblib.load(MODEL_DIR / "lstm_scaler.pkl")
with open(MODEL_DIR / "feature_cols_lstm_temporal.json") as f:
    lstm_feats = json.load(f)

# ── Load all Guardian models ──────────────────────────────────────────────────
gd_models = {}
for vname, run_id in GUARDIAN_VARIANTS:
    gd_dir = MODEL_DIR / "runs" / run_id
    m  = joblib.load(gd_dir / "guardian.pkl")
    sc = joblib.load(gd_dir / "guardian_scaler.pkl")
    with open(gd_dir / "guardian_feature_cols.json") as f:
        fc = json.load(f)
    gd_models[vname] = {"model": m, "scaler": sc, "feat_cols": fc,
                        "static": [c for c in fc if c not in set(GUARDIAN_DYNAMIC_FEATURES)]}
    logger.info(f"Loaded {vname}: {len(fc)} features ({len(gd_models[vname]['static'])} static)")

# ── Pre-load holdout data ─────────────────────────────────────────────────────
available = sorted(
    s for s in ALL_COINS
    if (HOLDOUT_DIR / "labeled" / f"{s}_features_v3.parquet").exists()
)
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

    # Pre-compute static arrays for all Guardian variants
    X_gd_variants = {}
    for vname, info in gd_models.items():
        X_gd_variants[vname] = compute_guardian_static_array(df, info["static"])

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
        "h4_tr":      df["h4_trend"].values       if "h4_trend"      in df.columns else None,
        "yt":         df["label"].map(LM).values.astype(np.int32),
        "index":      df.index,
        "X_gd":       X_gd_variants,
    }
print("Loaded.\n")


# ── Build T50_R55 predictions ─────────────────────────────────────────────────
def build_yp(proba_fb, proba_lstm, hmm):
    n  = len(proba_fb)
    yp = np.full(n, 1, np.int32)
    for i in range(n):
        is_t  = hmm[i] in TRENDING
        thr_l = THR_T if is_t else THR_R
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


def compute_pf(trades):
    gp = sum(t.get("net_pnl", 0) for t in trades if t.get("net_pnl", 0) > 0)
    gl = abs(sum(t.get("net_pnl", 0) for t in trades if t.get("net_pnl", 0) < 0))
    return round(gp / max(gl, 0.001), 2)


def run_eval(yp_map, guardian_name=None):
    info = gd_models[guardian_name] if guardian_name else None
    agg  = {"trades": 0, "wins": 0, "pnl": 0.0,
            "gross_profit": 0.0, "gross_loss": 0.0,
            "gd": 0, "sl": 0}

    for sym, d in coin_data.items():
        use_gd = guardian_name is not None
        X_gd   = d["X_gd"][guardian_name] if use_gd else None

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
            guardian_enabled=use_gd,
            guardian_model=info["model"]   if use_gd else None,
            guardian_scaler=info["scaler"] if use_gd else None,
            X_guardian=X_gd,
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
            p = t.get("net_pnl", 0)
            if p > 0: agg["gross_profit"] += p
            else:     agg["gross_loss"]   += abs(p)
            out = str(t.get("outcome", ""))
            if "GUARDIAN" in out: agg["gd"] += 1
            if "SL" in out or "STOP" in out: agg["sl"] += 1

    t, w, p = agg["trades"], agg["wins"], agg["pnl"]
    pf = round(agg["gross_profit"] / max(agg["gross_loss"], 0.001), 2)
    return {
        "trades":    t,
        "wr":        round(w / max(t, 1) * 100, 1),
        "pf":        pf,
        "pnl":       round(p, 1),
        "ppt":       round(p / max(t, 1), 3),
        "gd_pct":    round(agg["gd"] / max(t, 1) * 100, 1),
        "sl_pct":    round(agg["sl"] / max(t, 1) * 100, 1),
    }


# ── Run ───────────────────────────────────────────────────────────────────────
yp_map = {sym: build_yp(d["proba_fb"], d["proba_lstm"], d["hmm"])
          for sym, d in coin_data.items()}
total_entries = sum((yp_map[sym] != 1).sum() for sym in yp_map)
print(f"T50_R55 entries: {total_entries}")

print("Running evaluations...")
sc_base = run_eval(yp_map, guardian_name=None)
print(f"  no_guardian done: {sc_base['trades']} trades")

results = [{"name": "no_guardian", "feats": "-", **sc_base}]
for vname, run_id in GUARDIAN_VARIANTS:
    sc = run_eval(yp_map, guardian_name=vname)
    n_feats = len(gd_models[vname]["feat_cols"])
    n_static = len(gd_models[vname]["static"])
    feat_desc = f"{n_static}s+7d={n_feats}"
    results.append({"name": vname, "feats": feat_desc, **sc})
    print(f"  {vname} done: {sc['trades']} trades | WR {sc['wr']:.1f}% | PF {sc['pf']:.2f} | ${sc['pnl']:+.1f}")


# ── Print scorecard ───────────────────────────────────────────────────────────
print(f"\n{'='*110}")
print(f"  GUARDIAN VARIANTS COMPARISON -- Config T50_R55 (thr_T=0.50, thr_R=0.55) + LSTM")
print(f"  Guardian params: exit_thr={GD_EXIT_THR}, min_hold={GD_MIN_HOLD}")
print(f"  ic32 benchmark: {IC32['trades']} trades | WR {IC32['wr']}% | PF {IC32['pf']} | ${IC32['pnl']}")
print(f"{'='*110}")
print(f"  {'Variant':<16} {'Features':<14} {'Trades':>7} {'WR':>7} {'PF':>6} "
      f"{'Total PnL':>10} {'PPT':>8} {'GD%':>6} {'SL%':>5}  {'vs ic32':>8}")
print("  " + "-"*110)

base_pnl = results[0]["pnl"]
for r in results:
    vs_base = f"{r['pnl']-base_pnl:+.1f}" if r["name"] != "no_guardian" else "   base"
    vs_ic32 = f"{r['pnl']-IC32['pnl']:+.1f}"
    print(f"  {r['name']:<16} {r['feats']:<14} {r['trades']:>7,} {r['wr']:>6.1f}% {r['pf']:>6.2f} "
          f"{r['pnl']:>+10.1f} {r['ppt']:>+8.3f} {r['gd_pct']:>5.1f}% {r['sl_pct']:>4.1f}%  {vs_ic32:>8}")

print(f"\n  Delta vs no_guardian:")
for r in results[1:]:
    delta_pnl = r["pnl"] - base_pnl
    delta_pf  = r["pf"] - results[0]["pf"]
    delta_wr  = r["wr"] - results[0]["wr"]
    print(f"    {r['name']:<16}: PnL {delta_pnl:+.1f} | PF {delta_pf:+.2f} | WR {delta_wr:+.1f}%")

print(f"\n  CV F1 (training, higher=better internal fit):")
cv_info = [
    ("profit_v1",   0.8509),
    ("fb_v1",       0.8487),
    ("fb_union_v1", 0.8507),
]
for name, f1 in cv_info:
    print(f"    {name:<16}: F1 {f1:.4f}")

# Save
out = {
    "config":     "T50_R55 + LSTM veto",
    "gd_params":  {"exit_thr": GD_EXIT_THR, "min_hold": GD_MIN_HOLD},
    "ic32":       IC32,
    "results":    results,
}
out_path = MODEL_DIR / "runs" / FB_RUN / "guardian_variants_eval.json"
with open(out_path, "w") as f:
    json.dump(out, f, indent=2)
print(f"\n  Saved -> {out_path}")
