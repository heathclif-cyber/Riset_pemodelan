"""Holdout test: TB 3-Class v3 vs ic32_regime_v1"""
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

logger = setup_logger("tb_holdout_v3")

RUN_NAME = "tb_lgbm_widyawardhana_v3"
HOLDOUT_LABEL_DIR = HOLDOUT_DIR / "labeled"

# Load models
tb_model = joblib.load(MODEL_DIR / "runs" / RUN_NAME / "lgbm.pkl")
with open(MODEL_DIR / "runs" / RUN_NAME / f"{RUN_NAME}_features.json") as f:
    tb_feats = json.load(f)

bl_model = joblib.load(MODEL_DIR / "lgbm_baseline.pkl")
if hasattr(bl_model, 'feature_name_') and bl_model.feature_name_:
    bl_feats = list(bl_model.feature_name_)
else:
    with open(MODEL_DIR / "feature_cols_v2.json") as f:
        bl_feats = json.load(f)

logger.info(f"TB v3: {len(tb_feats)} features | Baseline: {len(bl_feats)} features")

available = [s for s in ALL_COINS if (HOLDOUT_LABEL_DIR / f"{s}_features_v3.parquet").exists()]

# Threshold sweep for TB model
print(f"\n{'='*80}")
print(f"  THRESHOLD SWEEP — TB 3-Class v3 (BTCUSDT)")
print(f"{'='*80}")

sym = 'BTCUSDT'
df = pd.read_parquet(HOLDOUT_LABEL_DIR / f"{sym}_features_v3.parquet")
df = ensure_utc_index(df).sort_index()
tb = triple_barrier_labeling(df["close"], df["high"], df["low"],
                             df["atr_14_h1"], 2.0, 1.5, 36)
df["tb_label"] = tb.map({"SHORT": 0, "FLAT": 1, "LONG": 2})
df = df.dropna(subset=["tb_label"])

X_tb = np.zeros((len(df), len(tb_feats)), dtype=np.float64)
for idx, col in enumerate(tb_feats):
    if col in df.columns:
        X_tb[:, idx] = df[col].ffill().fillna(0).values.astype(np.float64)
proba = tb_model.predict_proba(X_tb)
conf = np.max(proba, axis=1)
y_raw = np.argmax(proba, axis=1)

y_true = df["tb_label"].values.astype(np.int32)
atr_arr = df["atr_14_h1"].values
close_arr = df["close"].values; high_arr = df["high"].values; low_arr = df["low"].values
h4_sh = df["h4_swing_high"].values if "h4_swing_high" in df.columns else None
h4_sl = df["h4_swing_low"].values if "h4_swing_low" in df.columns else None

for conf_thresh in [0.40, 0.43, 0.45, 0.48, 0.50]:
    y_filt = y_raw.copy()
    # Respect FLAT predictions + confidence filter
    for i in range(len(y_filt)):
        if y_filt[i] != 1 and conf[i] < conf_thresh:
            y_filt[i] = 1  # low confidence -> FLAT

    if (y_filt != 1).sum() == 0:
        print(f"  thresh={conf_thresh:.2f}: 0 trades (all filtered)")
        continue

    report = full_trading_report(
        y_pred=y_filt, y_actual=y_true,
        atr=atr_arr, close=close_arr, high=high_arr, low=low_arr,
        h4_swing_highs=h4_sh, h4_swing_lows=h4_sl,
        index=df.index, modal=MODAL_PER_TRADE, leverages=LEVERAGE_SIM,
        fee_per_side=FEE_PER_SIDE, slippage=SLIPPAGE_PER_SIDE,
        min_rr=SWING_LABEL_MIN_RR, min_tp_atr=SWING_LABEL_MIN_TP,
        max_sl_atr=SWING_LABEL_MAX_SL, tp_fallback_atr=TP_SL_FALLBACK_TP,
        sl_fallback_atr=TP_SL_FALLBACK_SL, max_hold=MAX_HOLDING_BARS,
        symbol=sym, confidence=conf,
        guardian_enabled=False, trailing_stop_enabled=False,
    )
    lev5 = report.get("lev5x", report)
    trades = lev5.get("trades", [])
    wins = sum(1 for t in trades if t.get("outcome") == "WIN")
    pnl = sum(t.get("net_pnl", 0) for t in trades)
    wr = wins / max(len(trades), 1) * 100
    eq = np.array(lev5.get("equity_curve", [0]))
    peak = np.maximum.accumulate(eq); dd_amt = peak - eq
    dd_correct = np.where(peak > 0, dd_amt / peak * 100, 0).max()
    n_flat = (y_filt == 1).sum()
    n_short = (y_filt == 0).sum(); n_long = (y_filt == 2).sum()
    print(f"  thresh={conf_thresh:.2f}: {len(trades)} trades | WR={wr:.1f}% | PnL=${pnl:+.0f} | DD={dd_correct:.1f}% | S={n_short} F={n_flat} L={n_long}")

# Run full comparison at best threshold
BEST_THRESH = 0.43  # balanced from sweep above
print(f"\n{'='*80}")
print(f"  FULL HOLDOUT — TB v3 (thresh={BEST_THRESH}) vs ic32_regime_v1")
print(f"{'='*80}")
print(f"{'Coin':<15} {'TBv3 Tr':>7} {'TBv3 WR':>7} {'TBv3 PnL':>9} {'BL Tr':>7} {'BL WR':>7} {'BL PnL':>9}")
print("-" * 67)

results_tb = []; results_bl = []

for sym in available:
    df = pd.read_parquet(HOLDOUT_LABEL_DIR / f"{sym}_features_v3.parquet")
    df = ensure_utc_index(df).sort_index()
    tb_labels = triple_barrier_labeling(df["close"], df["high"], df["low"],
                                        df["atr_14_h1"], 2.0, 1.5, 36)
    df["tb_label"] = tb_labels.map({"SHORT": 0, "FLAT": 1, "LONG": 2})
    df = df.dropna(subset=["tb_label"])

    atr_arr = df["atr_14_h1"].values; close_arr = df["close"].values
    high_arr = df["high"].values; low_arr = df["low"].values
    h4_sh = df["h4_swing_high"].values if "h4_swing_high" in df.columns else None
    h4_sl = df["h4_swing_low"].values if "h4_swing_low" in df.columns else None
    y_true = df["tb_label"].values.astype(np.int32)

    # TB inference
    X_tb = np.zeros((len(df), len(tb_feats)), dtype=np.float64)
    for idx, col in enumerate(tb_feats):
        if col in df.columns:
            X_tb[:, idx] = df[col].ffill().fillna(0).values.astype(np.float64)
    proba_tb = tb_model.predict_proba(X_tb)
    conf_tb = np.max(proba_tb, axis=1)
    y_tb = np.argmax(proba_tb, axis=1)
    y_tb[(y_tb != 1) & (conf_tb < BEST_THRESH)] = 1

    tb_trades = 0; tb_pnl = 0.0; tb_wr = 0.0
    if (y_tb != 1).sum() > 0:
        report = full_trading_report(
            y_pred=y_tb, y_actual=y_true, atr=atr_arr, close=close_arr,
            high=high_arr, low=low_arr, h4_swing_highs=h4_sh, h4_swing_lows=h4_sl,
            index=df.index, modal=MODAL_PER_TRADE, leverages=LEVERAGE_SIM,
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
        tb_pnl = sum(t.get("net_pnl", 0) for t in trades)
        tb_trades = len(trades)
        tb_wr = wins / max(tb_trades, 1) * 100

    results_tb.append({"symbol": sym, "trades": tb_trades, "pnl": tb_pnl, "wr": tb_wr})

    # Baseline inference
    X_bl = np.zeros((len(df), len(bl_feats)), dtype=np.float64)
    for idx, col in enumerate(bl_feats):
        if col in df.columns:
            X_bl[:, idx] = df[col].ffill().fillna(0).values.astype(np.float64)
    proba_bl = bl_model.predict_proba(X_bl)
    conf_bl = np.max(proba_bl, axis=1); y_bl = np.argmax(proba_bl, axis=1)
    for i in range(len(y_bl)):
        if y_bl[i] == 2 and proba_bl[i, 2] < LGBM_THRESHOLD_LONG: y_bl[i] = 1
        elif y_bl[i] == 0 and proba_bl[i, 0] < LGBM_THRESHOLD_SHORT: y_bl[i] = 1
    y_bl[(y_bl != 1) & (conf_bl < CONFIDENCE_THRESHOLD_ENTRY)] = 1

    bl_trades = 0; bl_pnl = 0.0; bl_wr = 0.0
    if (y_bl != 1).sum() > 0:
        report_bl = full_trading_report(
            y_pred=y_bl, y_actual=y_true, atr=atr_arr, close=close_arr,
            high=high_arr, low=low_arr, h4_swing_highs=h4_sh, h4_swing_lows=h4_sl,
            index=df.index, modal=MODAL_PER_TRADE, leverages=LEVERAGE_SIM,
            fee_per_side=FEE_PER_SIDE, slippage=SLIPPAGE_PER_SIDE,
            min_rr=SWING_LABEL_MIN_RR, min_tp_atr=SWING_LABEL_MIN_TP,
            max_sl_atr=SWING_LABEL_MAX_SL, tp_fallback_atr=TP_SL_FALLBACK_TP,
            sl_fallback_atr=TP_SL_FALLBACK_SL, max_hold=MAX_HOLDING_BARS,
            symbol=sym, confidence=conf_bl,
            guardian_enabled=False, trailing_stop_enabled=False,
        )
        lev5 = report_bl.get("lev5x", report_bl)
        trades = lev5.get("trades", [])
        wins = sum(1 for t in trades if t.get("outcome") == "WIN")
        bl_pnl = sum(t.get("net_pnl", 0) for t in trades)
        bl_trades = len(trades)
        bl_wr = wins / max(bl_trades, 1) * 100

    results_bl.append({"symbol": sym, "trades": bl_trades, "pnl": bl_pnl, "wr": bl_wr})

    r_tb = results_tb[-1]; r_bl = results_bl[-1]
    print(f"{sym:<15} {r_tb['trades']:>7} {r_tb['wr']:>6.1f}% {r_tb['pnl']:>+8.0f} {r_bl['trades']:>7} {r_bl['wr']:>6.1f}% {r_bl['pnl']:>+8.0f}")

# Summary
print("-" * 67)
tb_t = sum(r["trades"] for r in results_tb); tb_pnl = sum(r["pnl"] for r in results_tb)
bl_t = sum(r["trades"] for r in results_bl); bl_pnl = sum(r["pnl"] for r in results_bl)
tb_w = sum(int(r["trades"]*r["wr"]/100) for r in results_tb)
bl_w = sum(int(r["trades"]*r["wr"]/100) for r in results_bl)
print(f"{'AGGREGATE':<15} {tb_t:>7} {tb_w/max(tb_t,1)*100:>6.1f}% {tb_pnl:>+8.0f} {bl_t:>7} {bl_w/max(bl_t,1)*100:>6.1f}% {bl_pnl:>+8.0f}")

# Per-class accuracy
print(f"\n{'='*60}")
print(f"  TB v3 Prediction Distribution (first 5 coins)")
print(f"{'='*60}")
for sym in available[:5]:
    df = pd.read_parquet(HOLDOUT_LABEL_DIR / f"{sym}_features_v3.parquet")
    df = ensure_utc_index(df).sort_index()
    tb_labels = triple_barrier_labeling(df["close"], df["high"], df["low"],
                                        df["atr_14_h1"], 2.0, 1.5, 36)
    df["tb_label"] = tb_labels.map({"SHORT": 0, "FLAT": 1, "LONG": 2})
    df = df.dropna(subset=["tb_label"])
    X_tb = np.zeros((len(df), len(tb_feats)), dtype=np.float64)
    for idx, col in enumerate(tb_feats):
        if col in df.columns:
            X_tb[:, idx] = df[col].ffill().fillna(0).values.astype(np.float64)
    proba_tb = tb_model.predict_proba(X_tb)
    conf_tb = np.max(proba_tb, axis=1)
    y_tb = np.argmax(proba_tb, axis=1)
    y_tb[(y_tb != 1) & (conf_tb < BEST_THRESH)] = 1
    y_true = df["tb_label"].values.astype(np.int32)
    from sklearn.metrics import classification_report
    names = ['SHORT','FLAT','LONG']
    print(f"\n  {sym}:")
    print(f"  Predicted: S={(y_tb==0).sum()} F={(y_tb==1).sum()} L={(y_tb==2).sum()}")
    print(f"  Actual:    S={(y_true==0).sum()} F={(y_true==1).sum()} L={(y_true==2).sum()}")
    for i, name in enumerate(names):
        mask = y_tb == i
        if mask.sum() > 0:
            acc = (y_tb[mask] == y_true[mask]).mean()
            print(f"  {name} precision: {acc:.3f} (n={mask.sum()})")
