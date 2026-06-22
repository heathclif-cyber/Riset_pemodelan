import json, sys, warnings, numpy as np, pandas as pd
from pathlib import Path
warnings.filterwarnings('ignore')
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
import joblib
from core.features import triple_barrier_labeling
from core.evaluator import full_trading_report
from core.utils import setup_logger, ensure_utc_index
from config import *

logger = setup_logger("tb_holdout_v2")

RUN_NAME = "tb_lgbm_widyawardhana_v2"
HOLDOUT_LABEL_DIR = HOLDOUT_DIR / "labeled"
TB_CONFIDENCE_THRESHOLD = 0.55

# Load model
model_path = MODEL_DIR / "runs" / RUN_NAME / "lgbm.pkl"
feat_path = MODEL_DIR / "runs" / RUN_NAME / f"{RUN_NAME}_features.json"
model = joblib.load(model_path)
with open(feat_path) as f:
    features = json.load(f)
logger.info(f"Model: {len(features)} features (multistage KEEP+STABLE)")

# Load baseline
bl_model = joblib.load(MODEL_DIR / "lgbm_baseline.pkl")
if hasattr(bl_model, 'feature_name_') and bl_model.feature_name_:
    bl_features = list(bl_model.feature_name_)
else:
    with open(MODEL_DIR / "feature_cols_v2.json") as f:
        bl_features = json.load(f)

available = [s for s in ALL_COINS if (HOLDOUT_LABEL_DIR / f"{s}_features_v3.parquet").exists()]

results_tb = []; results_bl = []
print(f"{'Coin':<15} {'TB Tr':>7} {'TB WR':>7} {'TB PnL':>9} {'BL Tr':>7} {'BL WR':>7} {'BL PnL':>9}")
print("-" * 67)

for sym in available:
    df = pd.read_parquet(HOLDOUT_LABEL_DIR / f"{sym}_features_v3.parquet")
    df = ensure_utc_index(df).sort_index()
    tb = triple_barrier_labeling(df["close"], df["high"], df["low"],
                                 df["atr_14_h1"], TP_SL_FALLBACK_TP,
                                 TP_SL_FALLBACK_SL, MAX_HOLDING_BARS)
    df["tb_label"] = tb.map({"SHORT": 0, "FLAT": 1, "LONG": 2})
    df_bin = df[df["tb_label"] != 1].copy()
    df_bin["tb_binary"] = (df_bin["tb_label"] == 2).astype(np.int32)

    atr_arr = df_bin["atr_14_h1"].values; close_arr = df_bin["close"].values
    high_arr = df_bin["high"].values; low_arr = df_bin["low"].values
    h4_sh = df_bin["h4_swing_high"].values if "h4_swing_high" in df_bin.columns else None
    h4_sl = df_bin["h4_swing_low"].values if "h4_swing_low" in df_bin.columns else None

    # TB inference
    X_tb = np.zeros((len(df_bin), len(features)), dtype=np.float64)
    for idx, col in enumerate(features):
        if col in df_bin.columns:
            X_tb[:, idx] = df_bin[col].ffill().fillna(0).values.astype(np.float64)
    proba = model.predict_proba(X_tb)
    conf_tb = np.max(proba, axis=1)
    y_tb = np.where(proba[:, 1] > 0.5, 2, 0).astype(np.int32)
    y_tb[conf_tb < TB_CONFIDENCE_THRESHOLD] = 1

    if (y_tb != 1).sum() == 0:
        results_tb.append({"symbol": sym, "total_trades": 0, "net_pnl": 0, "win_rate_pct": 0,
                          "peak_eq": 0, "dd_correct": 0})
    else:
        report = full_trading_report(
            y_pred=y_tb, y_actual=df_bin["tb_binary"].values,
            atr=atr_arr, close=close_arr, high=high_arr, low=low_arr,
            h4_swing_highs=h4_sh, h4_swing_lows=h4_sl,
            index=df_bin.index, modal=MODAL_PER_TRADE, leverages=LEVERAGE_SIM,
            fee_per_side=FEE_PER_SIDE, slippage=SLIPPAGE_PER_SIDE,
            min_rr=SWING_LABEL_MIN_RR, min_tp_atr=SWING_LABEL_MIN_TP,
            max_sl_atr=SWING_LABEL_MAX_SL, tp_fallback_atr=TP_SL_FALLBACK_TP,
            sl_fallback_atr=TP_SL_FALLBACK_SL, max_hold=MAX_HOLDING_BARS,
            symbol=sym, confidence=conf_tb,
            guardian_enabled=False, trailing_stop_enabled=False,
        )
        lev5 = report.get("lev5x", report)
        trades = lev5.get("trades", [])
        wins = sum(1 for t in trades if t.get("outcome") == "WIN")
        pnl = sum(t.get("net_pnl", 0) for t in trades)
        wr = wins / max(len(trades), 1) * 100
        eq = np.array(lev5.get("equity_curve", [0]))
        peak = np.maximum.accumulate(eq); dd_amt = peak - eq
        peak_eq = peak.max()
        dd_correct = np.where(peak > 0, dd_amt / peak * 100, 0).max()
        results_tb.append({"symbol": sym, "total_trades": len(trades), "net_pnl": round(pnl,2),
                          "win_rate_pct": round(wr,2), "peak_eq": round(float(peak_eq),0),
                          "dd_correct": round(float(dd_correct),1)})

    # Baseline inference
    X_bl = np.zeros((len(df_bin), len(bl_features)), dtype=np.float64)
    for idx, col in enumerate(bl_features):
        if col in df_bin.columns:
            X_bl[:, idx] = df_bin[col].ffill().fillna(0).values.astype(np.float64)
    bl_proba = bl_model.predict_proba(X_bl)
    bl_conf = np.max(bl_proba, axis=1); bl_y = np.argmax(bl_proba, axis=1)
    bl_y_filt = bl_y.copy()
    for i in range(len(bl_y)):
        if bl_y[i] == 2 and bl_proba[i, 2] < LGBM_THRESHOLD_LONG: bl_y_filt[i] = 1
        elif bl_y[i] == 0 and bl_proba[i, 0] < LGBM_THRESHOLD_SHORT: bl_y_filt[i] = 1
    bl_y_filt[(bl_y_filt != 1) & (bl_conf < CONFIDENCE_THRESHOLD_ENTRY)] = 1

    if (bl_y_filt != 1).sum() == 0:
        results_bl.append({"symbol": sym, "total_trades": 0, "net_pnl": 0, "win_rate_pct": 0})
    else:
        report_bl = full_trading_report(
            y_pred=bl_y_filt, y_actual=df_bin["tb_binary"].values,
            atr=atr_arr, close=close_arr, high=high_arr, low=low_arr,
            h4_swing_highs=h4_sh, h4_swing_lows=h4_sl,
            index=df_bin.index, modal=MODAL_PER_TRADE, leverages=LEVERAGE_SIM,
            fee_per_side=FEE_PER_SIDE, slippage=SLIPPAGE_PER_SIDE,
            min_rr=SWING_LABEL_MIN_RR, min_tp_atr=SWING_LABEL_MIN_TP,
            max_sl_atr=SWING_LABEL_MAX_SL, tp_fallback_atr=TP_SL_FALLBACK_TP,
            sl_fallback_atr=TP_SL_FALLBACK_SL, max_hold=MAX_HOLDING_BARS,
            symbol=sym, confidence=bl_conf,
            guardian_enabled=False, trailing_stop_enabled=False,
        )
        lev5 = report_bl.get("lev5x", report_bl)
        trades = lev5.get("trades", [])
        wins = sum(1 for t in trades if t.get("outcome") == "WIN")
        pnl = sum(t.get("net_pnl", 0) for t in trades)
        wr = wins / max(len(trades), 1) * 100
        results_bl.append({"symbol": sym, "total_trades": len(trades), "net_pnl": round(pnl,2),
                          "win_rate_pct": round(wr,2)})

    r_tb = results_tb[-1]; r_bl = results_bl[-1]
    print(f"{sym:<15} {r_tb['total_trades']:>7} {r_tb['win_rate_pct']:>6.1f}% {r_tb['net_pnl']:>+8.0f} {r_bl['total_trades']:>7} {r_bl['win_rate_pct']:>6.1f}% {r_bl['net_pnl']:>+8.0f}")

# Summary
print("-" * 67)
tb_t = sum(r["total_trades"] for r in results_tb); tb_pnl = sum(r["net_pnl"] for r in results_tb)
bl_t = sum(r["total_trades"] for r in results_bl); bl_pnl = sum(r["net_pnl"] for r in results_bl)
tb_w = sum(int(r["total_trades"]*r["win_rate_pct"]/100) for r in results_tb)
bl_w = sum(int(r["total_trades"]*r["win_rate_pct"]/100) for r in results_bl)
print(f"{'AGGREGATE':<15} {tb_t:>7} {tb_w/max(tb_t,1)*100:>6.1f}% {tb_pnl:>+8.0f} {bl_t:>7} {bl_w/max(bl_t,1)*100:>6.1f}% {bl_pnl:>+8.0f}")

# DD analysis
print(f"\n{'='*70}")
print(f"  DD ANALYSIS (correct: dd/peak_equity)")
print(f"{'='*70}")
for r in results_tb:
    if r.get("peak_eq", 0) > 0:
        print(f"  {r['symbol']:<15} trades={r['total_trades']:>5} peak_eq=${r['peak_eq']:>6.0f} DD={r['dd_correct']:>5.1f}%")
