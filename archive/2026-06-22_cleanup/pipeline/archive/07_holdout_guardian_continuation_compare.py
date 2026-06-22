"""
pipeline/07_holdout_guardian_continuation_compare.py
Holdout: profit_v1 vs momentum_v1 vs continuation_v1

Usage: python pipeline/07_holdout_guardian_continuation_compare.py
"""
import json
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from core.evaluator import full_trading_report
from core.utils import setup_logger, ensure_utc_index
from pipeline.backtest_utils import compute_guardian_static_array
from config import (
    ALL_COINS,
    HOLDOUT_DIR,
    MODEL_DIR,
    MODAL_PER_TRADE,
    LEVERAGE_SIM,
    FEE_PER_SIDE,
    SLIPPAGE_PER_SIDE,
    MAX_HOLDING_BARS,
    SWING_LABEL_MIN_RR,
    SWING_LABEL_MIN_TP,
    SWING_LABEL_MAX_SL,
    TP_SL_FALLBACK_TP,
    TP_SL_FALLBACK_SL,
    GUARDIAN_EXIT_THRESHOLD,
    GUARDIAN_MIN_HOLD_BARS,
    GUARDIAN_DYNAMIC_FEATURES as _GDYN,
    LABEL_MAP,
)

logger = setup_logger("07_holdout_guardian_continuation_compare")

FB_RUN = "tb_lgbm_flatboost_v2"
GD_PROFIT_RUN = "tb_guardian_profit_v1"
GD_MOM_RUN = "tb_guardian_momentum_v1"
GD_CONT_RUN = "tb_guardian_continuation_v1"

THR_TRENDING_LONG = 0.50
THR_TRENDING_SHORT = 0.55
THR_RANGING_LONG = 0.55
THR_RANGING_SHORT = 0.60
TRENDING_STATES = {0, 3}
GD_EXIT_THR = GUARDIAN_EXIT_THRESHOLD
GD_MIN_HOLD = GUARDIAN_MIN_HOLD_BARS
FLOW_MOM_WINDOW = 3

PERIOD_LABEL = "Apr 2026-Jun 2026 (~2.5 bln)"
MONTHS = 2.5
LM = LABEL_MAP if isinstance(LABEL_MAP, dict) else {"SHORT": 0, "FLAT": 1, "LONG": 2}


def add_momentum_feats(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "cvd_slope_h4" in df.columns:
        df["cvd_slope_h4_delta"] = df["cvd_slope_h4"].diff(1)
    else:
        df["cvd_slope_h4_delta"] = 0.0
    if "ofi_h4_delta" in df.columns:
        df["ofi_h4_accel"] = df["ofi_h4_delta"].diff(2)
    else:
        df["ofi_h4_accel"] = 0.0
    if "rsi_h4" in df.columns:
        df["rsi_h4_slope"] = df["rsi_h4"].diff(2)
    else:
        df["rsi_h4_slope"] = 0.0
    if "dist_liq_50x_long" not in df.columns:
        df["dist_liq_50x_long"] = 0.0
    if "ofi_z_score" in df.columns:
        df["flow_momentum_3bar"] = df["ofi_z_score"].rolling(FLOW_MOM_WINDOW, min_periods=1).mean()
    else:
        df["flow_momentum_3bar"] = 0.0
    return df


def load_guardian(run_name: str):
    run_dir = MODEL_DIR / "runs" / run_name
    if not (run_dir / "guardian.pkl").exists():
        return None
    model = joblib.load(run_dir / "guardian.pkl")
    scaler = joblib.load(run_dir / "guardian_scaler.pkl")
    with open(run_dir / "guardian_feature_cols.json") as f:
        feats = json.load(f)
    static = [c for c in feats if c not in set(_GDYN) and c not in {
        "cvd_slope_h4_delta_entry", "ofi_h4_delta_entry", "flow_momentum_3bar",
    }]
    return {
        "model": model,
        "scaler": scaler,
        "feats": feats,
        "static": static,
        "use_assembler": "cvd_slope_h4_delta_entry" in feats,
    }


fb_model = joblib.load(MODEL_DIR / "runs" / FB_RUN / "lgbm.pkl")
with open(MODEL_DIR / "runs" / FB_RUN / f"{FB_RUN}_features.json") as f:
    fb_feats = json.load(f)

gd_profit = load_guardian(GD_PROFIT_RUN)
gd_mom = load_guardian(GD_MOM_RUN)
gd_cont = load_guardian(GD_CONT_RUN)

available = [
    s for s in ALL_COINS
    if (HOLDOUT_DIR / "labeled" / f"{s}_features_v3.parquet").exists()
]
logger.info(f"Coins: {len(available)}")


def new_agg():
    return {
        "trades": 0, "wins": 0, "pnl": 0.0, "longs": 0, "long_wins": 0,
        "short_trades": 0, "short_wins": 0,
        "sl": 0, "gd_exits": 0, "gross_win": 0.0, "gross_loss": 0.0,
        "short_pnl": 0.0,
    }


aggs = {
    "base": new_agg(),
    "profit": new_agg(),
    "mom": new_agg(),
    "cont": new_agg(),
}


for sym in available:
    path = HOLDOUT_DIR / "labeled" / f"{sym}_features_v3.parquet"
    df = ensure_utc_index(pd.read_parquet(path)).sort_index()

    reg_path = HOLDOUT_DIR / "labeled" / f"{sym}_regime_h1.parquet"
    hmm = np.full(len(df), 1, np.int32)
    if reg_path.exists():
        try:
            reg = pd.read_parquet(reg_path)
            if "hmm_regime_enc" in reg.columns:
                hmm = reg["hmm_regime_enc"].reindex(df.index, fill_value=1).values.astype(np.int32)
        except Exception:
            pass

    df_mom = add_momentum_feats(df)

    mask = df["label"].astype(str).isin(LM)
    df = df[mask].copy()
    df_mom = df_mom[mask].copy()
    hmm = hmm[mask.values]
    n = len(df)
    if n < 50:
        continue

    close = df["close"].values
    high = df["high"].values
    low = df["low"].values
    atr = df["atr_14_h1"].values
    h4_sh = df["h4_swing_high"].values if "h4_swing_high" in df.columns else np.full(n, np.nan)
    h4_sl = df["h4_swing_low"].values if "h4_swing_low" in df.columns else np.full(n, np.nan)
    h4t = df["h4_trend"].values if "h4_trend" in df.columns else None
    y_act = np.array([LM.get(str(v), 1) for v in df["label"].values], dtype=np.int32)

    X_fb = np.zeros((n, len(fb_feats)), dtype=np.float64)
    for i, c in enumerate(fb_feats):
        if c in df.columns:
            X_fb[:, i] = df[c].ffill().fillna(0).values.astype(np.float64)
    proba_fb = fb_model.predict_proba(X_fb)

    yp = np.full(n, 1, np.int32)
    for i in range(n):
        is_trend = hmm[i] in TRENDING_STATES
        tl = THR_TRENDING_LONG if is_trend else THR_RANGING_LONG
        ts = THR_TRENDING_SHORT if is_trend else THR_RANGING_SHORT
        if proba_fb[i, 2] >= tl:
            yp[i] = 2
        elif proba_fb[i, 0] >= ts:
            yp[i] = 0
    conf = proba_fb.max(axis=1)

    flow_arr = (
        df_mom["flow_momentum_3bar"].ffill().fillna(0).values.astype(np.float64)
        if "flow_momentum_3bar" in df_mom.columns else np.zeros(n)
    )

    base_kw = dict(
        y_pred=yp, y_actual=y_act, atr=atr, close=close, high=high, low=low,
        h4_swing_highs=h4_sh, h4_swing_lows=h4_sl,
        index=df.index, modal=MODAL_PER_TRADE, leverages=LEVERAGE_SIM,
        fee_per_side=FEE_PER_SIDE, slippage=SLIPPAGE_PER_SIDE,
        min_rr=SWING_LABEL_MIN_RR, min_tp_atr=SWING_LABEL_MIN_TP,
        max_sl_atr=SWING_LABEL_MAX_SL,
        tp_fallback_atr=TP_SL_FALLBACK_TP, sl_fallback_atr=TP_SL_FALLBACK_SL,
        max_hold=MAX_HOLDING_BARS, symbol=sym, confidence=conf,
        h4_trend=h4t, trailing_stop_enabled=False,
    )

    reps = {"base": full_trading_report(**base_kw, guardian_enabled=False)}

    if gd_profit:
        X_gd = compute_guardian_static_array(df, gd_profit["static"])
        reps["profit"] = full_trading_report(
            **base_kw,
            guardian_enabled=True,
            guardian_model=gd_profit["model"],
            guardian_scaler=gd_profit["scaler"],
            X_guardian=X_gd,
            guardian_exit_threshold=GD_EXIT_THR,
            guardian_min_hold_bars=GD_MIN_HOLD,
        )

    if gd_mom:
        X_gd = compute_guardian_static_array(df_mom, gd_mom["static"])
        reps["mom"] = full_trading_report(
            **base_kw,
            guardian_enabled=True,
            guardian_model=gd_mom["model"],
            guardian_scaler=gd_mom["scaler"],
            X_guardian=X_gd,
            guardian_exit_threshold=GD_EXIT_THR,
            guardian_min_hold_bars=GD_MIN_HOLD,
        )

    if gd_cont:
        X_gd = compute_guardian_static_array(df_mom, gd_cont["static"])
        reps["cont"] = full_trading_report(
            **base_kw,
            guardian_enabled=True,
            guardian_model=gd_cont["model"],
            guardian_scaler=gd_cont["scaler"],
            X_guardian=X_gd,
            guardian_exit_threshold=GD_EXIT_THR,
            guardian_min_hold_bars=GD_MIN_HOLD,
            guardian_feat_cols=gd_cont["feats"],
            guardian_static_names=gd_cont["static"],
            flow_momentum_arr=flow_arr,
        )

    def extract_agg(rep):
        lev = rep.get("lev5x", rep)
        trades = lev.get("trades", [])
        shorts = [t for t in trades if t.get("direction") == "SHORT"]
        return {
            "trades": len(trades),
            "wins": sum(1 for t in trades if t.get("net_pnl", 0) > 0),
            "pnl": sum(t.get("net_pnl", 0) for t in trades),
            "longs": sum(1 for t in trades if t.get("direction") == "LONG"),
            "long_wins": sum(1 for t in trades if t.get("direction") == "LONG" and t.get("net_pnl", 0) > 0),
            "short_trades": len(shorts),
            "short_wins": sum(1 for t in shorts if t.get("net_pnl", 0) > 0),
            "short_pnl": sum(t.get("net_pnl", 0) for t in shorts),
            "sl": sum(1 for t in trades if "SL" in str(t.get("outcome", ""))),
            "gd_exits": sum(1 for t in trades if "GUARDIAN" in str(t.get("outcome", ""))),
            "gross_win": sum(t.get("net_pnl", 0) for t in trades if t.get("net_pnl", 0) > 0),
            "gross_loss": sum(t.get("net_pnl", 0) for t in trades if t.get("net_pnl", 0) < 0),
        }

    for key, rep in reps.items():
        src = extract_agg(rep)
        for k, v in src.items():
            aggs[key][k] += v

    logger.info(
        f"  [{sym}] base={extract_agg(reps['base'])['trades']} "
        f"profit={extract_agg(reps.get('profit', reps['base']))['trades'] if 'profit' in reps else '-'} "
        f"mom={extract_agg(reps.get('mom', reps['base']))['trades'] if 'mom' in reps else '-'} "
        f"cont={extract_agg(reps.get('cont', reps['base']))['trades'] if 'cont' in reps else '-'}"
    )


def scorecard(name, agg):
    t = max(agg["trades"], 1)
    wr = agg["wins"] / t * 100
    lwr = agg["long_wins"] / max(agg["longs"], 1) * 100
    swr = agg["short_wins"] / max(agg["short_trades"], 1) * 100
    pf = abs(agg["gross_win"] / agg["gross_loss"]) if agg["gross_loss"] != 0 else float("inf")
    return {
        "name": name,
        "trades": agg["trades"],
        "trades_per_month": round(agg["trades"] / MONTHS, 0),
        "win_rate": round(wr, 1),
        "long_wr": round(lwr, 1),
        "short_wr": round(swr, 1),
        "short_pnl": round(agg["short_pnl"], 2),
        "pnl": round(agg["pnl"], 2),
        "pnl_per_month": round(agg["pnl"] / MONTHS, 2),
        "pnl_per_trade": round(agg["pnl"] / max(agg["trades"], 1), 3),
        "profit_factor": round(pf, 2),
        "sl_rate": round(agg["sl"] / max(agg["trades"], 1) * 100, 1),
        "gd_exit_rate": round(agg["gd_exits"] / max(agg["trades"], 1) * 100, 1),
    }


variants = [
    scorecard("FB v2 Standalone", aggs["base"]),
    scorecard("FB v2 + profit_v1", aggs["profit"]),
    scorecard("FB v2 + momentum_v1", aggs["mom"]),
    scorecard("FB v2 + continuation_v1", aggs["cont"]),
]

col_w = 20
print(f"\n{'='*90}")
print(f"  HOLDOUT — Guardian continuation_v1 vs momentum_v1 vs profit_v1")
print(f"  Periode: {PERIOD_LABEL} | Koin: {len(available)} | $10/trade 5x")
print(f"{'='*90}")
hdr = f"{'Metrik':<28}" + "".join(f"{v['name']:>{col_w}}" for v in variants)
print(hdr)
print("-" * (28 + col_w * len(variants)))

rows = [
    ("Total Trades", "trades", lambda x: f"{x:,}"),
    ("Win Rate", "win_rate", lambda x: f"{x:.1f}%"),
    ("  SHORT WR", "short_wr", lambda x: f"{x:.1f}%"),
    ("  SHORT PnL", "short_pnl", lambda x: f"${x:+.0f}"),
    ("Net PnL", "pnl", lambda x: f"${x:+.0f}"),
    ("PnL/trade", "pnl_per_trade", lambda x: f"${x:+.3f}"),
    ("Profit Factor", "profit_factor", lambda x: f"{x:.2f}"),
    ("Guardian exits", "gd_exit_rate", lambda x: f"{x:.1f}%"),
]

for label, key, fmt in rows:
    row = f"  {label:<26}"
    for sc in variants:
        row += f"{fmt(sc.get(key, 0)):>{col_w}}"
    print(row)

print("-" * (28 + col_w * len(variants)))
d_cont_mom = variants[3]["pnl"] - variants[2]["pnl"]
d_cont_profit = variants[3]["pnl"] - variants[1]["pnl"]
d_short = variants[3]["short_pnl"] - variants[2]["short_pnl"]
print(f"\n  continuation_v1 vs momentum_v1 : {d_cont_mom:+.0f} USD (total) | SHORT {d_short:+.0f} USD")
print(f"  continuation_v1 vs profit_v1     : {d_cont_profit:+.0f} USD")

out_dir = MODEL_DIR / "runs" / GD_CONT_RUN
out_dir.mkdir(parents=True, exist_ok=True)
results = {"period": PERIOD_LABEL, "coins": len(available), "variants": variants}
with open(out_dir / "holdout_guardian_compare.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved -> {out_dir / 'holdout_guardian_compare.json'}")