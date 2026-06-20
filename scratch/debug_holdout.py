"""Debug holdout PnL issue."""
import json, sys, numpy as np, pandas as pd, lightgbm as lgb
from pathlib import Path
sys.path.insert(0, ".")
from core.evaluator import simulate_trades_swing
from core.utils import ensure_utc_index
from config import (ALL_COINS, LABEL_DIR, HOLDOUT_DIR, TRAIN_CUTOFF_DATE,
                    LGBM_PARAMS, LABEL_MAP, MODAL_PER_TRADE, LEVERAGE_SIM,
                    FEE_PER_SIDE, SLIPPAGE_PER_SIDE, MAX_HOLDING_BARS,
                    SWING_LABEL_MIN_RR, SWING_LABEL_MIN_TP, SWING_LABEL_MAX_SL,
                    TP_SL_FALLBACK_TP, TP_SL_FALLBACK_SL)

FEAT_SOURCE = Path("models/feature_cols_v2.json")
with open(FEAT_SOURCE) as f: features = json.load(f)

# Quick train on 1 coin
sym = "BTCUSDT"
df = pd.read_parquet(LABEL_DIR / f"{sym}_features_v3.parquet")
df = ensure_utc_index(df).sort_index()
df = df[df.index < TRAIN_CUTOFF_DATE]
mask = df["label"].astype(str).isin(LABEL_MAP)
df = df[mask].copy()
avail = [c for c in features if c in df.columns]
X_train = df[avail].ffill().fillna(0).values.astype(np.float32)
y_train = df["label"].map(LABEL_MAP).values.astype(np.int32)
print(f"Train: {len(X_train)} bars, features: {len(avail)}")

model = lgb.LGBMClassifier(**LGBM_PARAMS)
model.fit(X_train, y_train)
print(f"Model trained: n_estimators_={model.n_estimators_}")
print(f"Class distribution train: {np.bincount(y_train)}")

# Holdout
df_hold = pd.read_parquet(HOLDOUT_DIR / "labeled" / f"{sym}_features_v3.parquet")
df_hold = ensure_utc_index(df_hold).sort_index()
df_hold = df_hold[df_hold.index >= TRAIN_CUTOFF_DATE]
mask = df_hold["label"].astype(str).isin(LABEL_MAP)
df_hold = df_hold[mask].copy()
df_hold["coin"] = sym
print(f"Holdout: {len(df_hold)} bars")

X_hold = df_hold[avail].ffill().fillna(0).values.astype(np.float32)
probas = model.predict_proba(X_hold)
print(f"Probas range: [{probas.min():.4f}, {probas.max():.4f}]")
print(f"Probas mean per class: {probas.mean(axis=0)}")

# Test with just one threshold
n = len(df_hold)
LM = {"SHORT": 0, "FLAT": 1, "LONG": 2}
y_pred = np.full(n, LM["FLAT"], np.int32)
y_pred[probas[:, 2] >= 0.50] = LM["LONG"]
short_m = (probas[:, 0] >= 0.50) & (y_pred != LM["LONG"])
y_pred[short_m] = LM["SHORT"]
print(f"Signals: LONG={(y_pred==2).sum()} SHORT={(y_pred==0).sum()} FLAT={(y_pred==1).sum()}")

# Check ATR and price
print(f"close mean: {df_hold['close'].mean():.2f}")
print(f"ATR mean: {df_hold['atr_14_h1'].mean():.4f}")
print(f"first trade would use: close={df_hold['close'].iloc[0]:.2f} atr={df_hold['atr_14_h1'].iloc[0]:.4f}")

# Simulate one coin
result = simulate_trades_swing(
    y_pred=y_pred,
    close=df_hold["close"].values,
    high=df_hold["high"].values,
    low=df_hold["low"].values,
    atr=df_hold["atr_14_h1"].values,
    h4_swing_highs=df_hold["h4_swing_high"].values,
    h4_swing_lows=df_hold["h4_swing_low"].values,
    modal=MODAL_PER_TRADE, leverage=LEVERAGE_SIM[0],
    fee_per_side=FEE_PER_SIDE, slippage=SLIPPAGE_PER_SIDE,
    max_hold=MAX_HOLDING_BARS, min_rr=SWING_LABEL_MIN_RR,
    min_tp_atr=SWING_LABEL_MIN_TP, max_sl_atr=SWING_LABEL_MAX_SL,
    tp_fallback_atr=TP_SL_FALLBACK_TP, sl_fallback_atr=TP_SL_FALLBACK_SL,
    guardian_enabled=False,
)
trades = result.get("trades", [])
print(f"\nTrades: {len(trades)}")
if trades:
    for t in trades[:3]:
        print(f"  {t['direction']} entry={t['entry']:.2f} exit={t['exit']:.2f} "
              f"SL={t['sl']:.2f} TP={t['tp']:.2f} outcome={t['outcome']} pnl={t['net_pnl']:.4f}")
    losses = [t for t in trades if t["net_pnl"] < 0]
    wins   = [t for t in trades if t["net_pnl"] > 0]
    print(f"Losses: {len(losses)}, avg loss: ${np.mean([t['net_pnl'] for t in losses]) if losses else 0:.4f}")
    print(f"Wins: {len(wins)}, avg win: ${np.mean([t['net_pnl'] for t in wins]) if wins else 0:.4f}")
    print(f"Total PnL: ${sum(t['net_pnl'] for t in trades):.2f}")
