"""
Quick holdout test for TB Binary v2 — standalone LGBM, static TP/SL.
"""
import json, sys, warnings, numpy as np, pandas as pd
from pathlib import Path
warnings.filterwarnings("ignore")
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

# TB-specific thresholds (lower than swing model)
TB_CONFIDENCE_THRESHOLD = 0.55  # model confidence ~0.50-0.85

# Load model
model_path = MODEL_DIR / "runs" / RUN_NAME / "lgbm.pkl"
feat_path = MODEL_DIR / "runs" / RUN_NAME / f"{RUN_NAME}_features.json"
model = joblib.load(model_path)
with open(feat_path) as f:
    features = json.load(f)
logger.info(f"Model: {model_path}")
logger.info(f"Features: {len(features)}")

# Load baseline for comparison
bl_model = joblib.load(MODEL_DIR / "lgbm_baseline.pkl")
if hasattr(bl_model, 'feature_name_') and bl_model.feature_name_:
    bl_features = list(bl_model.feature_name_)
else:
    with open(MODEL_DIR / "feature_cols_v2.json") as f:
        bl_features = json.load(f)
logger.info(f"Baseline: {len(bl_features)} features")

available = [s for s in ALL_COINS if (HOLDOUT_LABEL_DIR / f"{s}_features_v3.parquet").exists()]
logger.info(f"Coins: {len(available)}")

results_tb = []
results_bl = []

for sym in available:
    df = pd.read_parquet(HOLDOUT_LABEL_DIR / f"{sym}_features_v3.parquet")
    df = ensure_utc_index(df).sort_index()

    # TB labeling
    tb = triple_barrier_labeling(df["close"], df["high"], df["low"],
                                 df["atr_14_h1"], TP_SL_FALLBACK_TP,
                                 TP_SL_FALLBACK_SL, MAX_HOLDING_BARS)
    df["tb_label"] = tb.map({"SHORT": 0, "FLAT": 1, "LONG": 2})
    df_bin = df[df["tb_label"] != 1].copy()  # drop FLAT
    df_bin["tb_binary"] = (df_bin["tb_label"] == 2).astype(np.int32)

    atr_arr = df_bin["atr_14_h1"].values
    close_arr = df_bin["close"].values
    high_arr = df_bin["high"].values
    low_arr = df_bin["low"].values
    h4_sh = df_bin["h4_swing_high"].values if "h4_swing_high" in df_bin.columns else None
    h4_sl = df_bin["h4_swing_low"].values if "h4_swing_low" in df_bin.columns else None

    # ── TB model inference ─────────────────────────────────────────────────
    X_tb = np.zeros((len(df_bin), len(features)), dtype=np.float64)
    for idx, col in enumerate(features):
        if col in df_bin.columns:
            X_tb[:, idx] = df_bin[col].ffill().fillna(0).values.astype(np.float64)

    proba = model.predict_proba(X_tb)
    p_long = proba[:, 1]
    conf_tb = np.max(proba, axis=1)
    y_tb = np.where(p_long > 0.5, 2, 0).astype(np.int32)  # 0=SHORT, 2=LONG

    # Threshold filter: convert low-confidence to FLAT (1)
    y_tb_filt = y_tb.copy()
    y_tb_filt[conf_tb < TB_CONFIDENCE_THRESHOLD] = 1

    n_long = (y_tb_filt == 2).sum()
    n_short = (y_tb_filt == 0).sum()
    n_flat = (y_tb_filt == 1).sum()

    if n_long + n_short == 0:
        logger.info(f"[{sym}] TB: 0 trades (all filtered)")
        results_tb.append({"symbol": sym, "total_trades": 0, "net_pnl": 0, "win_rate_pct": 0})
    else:
        report_tb = full_trading_report(
            y_pred=y_tb_filt, y_actual=df_bin["tb_binary"].values,
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
        lev5 = report_tb.get("lev5x", report_tb)
        trades = lev5.get("trades", []) if isinstance(lev5, dict) else []
        wins = sum(1 for t in trades if t.get("outcome") == "WIN")
        pnl = sum(t.get("net_pnl", 0) for t in trades)
        wr = wins / max(len(trades), 1) * 100
        logger.info(f"[{sym}] TB: {len(trades)} trades | WR={wr:.1f}% | PnL=${pnl:+.0f}")
        results_tb.append({"symbol": sym, "total_trades": len(trades),
                           "net_pnl": round(pnl, 2), "win_rate_pct": round(wr, 2)})

    # ── Baseline inference ─────────────────────────────────────────────────
    X_bl = np.zeros((len(df_bin), len(bl_features)), dtype=np.float64)
    for idx, col in enumerate(bl_features):
        if col in df_bin.columns:
            X_bl[:, idx] = df_bin[col].ffill().fillna(0).values.astype(np.float64)

    bl_proba = bl_model.predict_proba(X_bl)
    bl_conf = np.max(bl_proba, axis=1)
    bl_y = np.argmax(bl_proba, axis=1)

    # Apply same threshold logic (swing thresholds)
    bl_y_filt = bl_y.copy()
    for i in range(len(bl_y)):
        if bl_y[i] == 2 and bl_proba[i, 2] < LGBM_THRESHOLD_LONG:
            bl_y_filt[i] = 1
        elif bl_y[i] == 0 and bl_proba[i, 0] < LGBM_THRESHOLD_SHORT:
            bl_y_filt[i] = 1
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
        trades = lev5.get("trades", []) if isinstance(lev5, dict) else []
        wins = sum(1 for t in trades if t.get("outcome") == "WIN")
        pnl = sum(t.get("net_pnl", 0) for t in trades)
        wr = wins / max(len(trades), 1) * 100
        logger.info(f"[{sym}] BL: {len(trades)} trades | WR={wr:.1f}% | PnL=${pnl:+.0f}")
        results_bl.append({"symbol": sym, "total_trades": len(trades),
                           "net_pnl": round(pnl, 2), "win_rate_pct": round(wr, 2)})

# ── Summary ──────────────────────────────────────────────────────────────────
print(f"\n{'='*70}")
print(f"  COMPARISON: TB Binary v2 vs ic32_regime_v1 (on TB-labeled holdout)")
print(f"{'='*70}")
print(f"{'Coin':<15} {'TB Trades':>10} {'TB WR%':>8} {'TB PnL':>10} {'BL Trades':>10} {'BL WR%':>8} {'BL PnL':>10}")
print("-" * 70)

tb_total_t, tb_total_pnl, tb_total_wins = 0, 0.0, 0
bl_total_t, bl_total_pnl, bl_total_wins = 0, 0.0, 0

for r_tb, r_bl in zip(results_tb, results_bl):
    sym = r_tb["symbol"]
    tb_total_t += r_tb["total_trades"]
    tb_total_pnl += r_tb["net_pnl"]
    tb_total_wins += int(r_tb["total_trades"] * r_tb["win_rate_pct"] / 100)
    bl_total_t += r_bl["total_trades"]
    bl_total_pnl += r_bl["net_pnl"]
    bl_total_wins += int(r_bl["total_trades"] * r_bl["win_rate_pct"] / 100)
    print(f"{sym:<15} {r_tb['total_trades']:>10} {r_tb['win_rate_pct']:>7.1f}% {r_tb['net_pnl']:>+9.0f} {r_bl['total_trades']:>10} {r_bl['win_rate_pct']:>7.1f}% {r_bl['net_pnl']:>+9.0f}")

print("-" * 70)
tb_wr = tb_total_wins / max(tb_total_t, 1) * 100
bl_wr = bl_total_wins / max(bl_total_t, 1) * 100
print(f"{'AGGREGATE':<15} {tb_total_t:>10} {tb_wr:>7.1f}% {tb_total_pnl:>+9.0f} {bl_total_t:>10} {bl_wr:>7.1f}% {bl_total_pnl:>+9.0f}")
print(f"{'='*70}")
print(f"\nTB threshold: {TB_CONFIDENCE_THRESHOLD} | BL thresholds: L={LGBM_THRESHOLD_LONG} S={LGBM_THRESHOLD_SHORT} C={CONFIDENCE_THRESHOLD_ENTRY}")
