"""Debug: per-coin holdout simulation."""
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

# Train on ALL coins
train_frames = []
for sym in ALL_COINS:
    path = LABEL_DIR / f"{sym}_features_v3.parquet"
    if not path.exists(): continue
    df = pd.read_parquet(path); df = ensure_utc_index(df).sort_index()
    df = df[df.index < TRAIN_CUTOFF_DATE]
    mask = df["label"].astype(str).isin(LABEL_MAP); df = df[mask].copy()
    df["coin"] = sym; train_frames.append(df)
df_train = pd.concat(train_frames).sort_index()
avail = [c for c in features if c in df_train.columns]
X_train = df_train[avail].ffill().fillna(0).values.astype(np.float32)
y_train = df_train["label"].map(LABEL_MAP).values.astype(np.int32)
print(f"Train: {len(X_train)} bars, {len(avail)} feat")

model = lgb.LGBMClassifier(**LGBM_PARAMS)
model.fit(X_train, y_train)
print(f"Model: {model.n_estimators_} trees")

# Test each coin separately
LM = {"SHORT": 0, "FLAT": 1, "LONG": 2}
thr_long, thr_short = 0.50, 0.50
total_trades = total_pnl = 0.0
total_wins = 0

for sym in ALL_COINS:
    path = HOLDOUT_DIR / "labeled" / f"{sym}_features_v3.parquet"
    if not path.exists(): continue
    df = pd.read_parquet(path); df = ensure_utc_index(df).sort_index()
    df = df[(df.index >= TRAIN_CUTOFF_DATE) & (df.index < pd.Timestamp("2026-06-14", tz="UTC"))]
    mask = df["label"].astype(str).isin(LABEL_MAP); df = df[mask].copy()
    df["coin"] = sym
    if len(df) < 50: continue

    avail_coin = [c for c in avail if c in df.columns]
    if len(avail_coin) != len(avail):
        print(f"  {sym}: SKIP ({len(avail_coin)}/{len(avail)} features)")
        continue

    X = df[avail].ffill().fillna(0).values.astype(np.float32)
    probas = model.predict_proba(X)
    n = len(df)
    y_pred = np.full(n, LM["FLAT"], np.int32)
    y_pred[probas[:, 2] >= thr_long] = LM["LONG"]
    short_m = (probas[:, 0] >= thr_short) & (y_pred != LM["LONG"])
    y_pred[short_m] = LM["SHORT"]
    n_sig = (y_pred != LM["FLAT"]).sum()
    if n_sig < 5:
        print(f"  {sym}: {n_sig} signals, skip")
        continue

    result = simulate_trades_swing(
        y_pred=y_pred, close=df["close"].values,
        high=df["high"].values, low=df["low"].values,
        atr=df["atr_14_h1"].values,
        h4_swing_highs=df["h4_swing_high"].values,
        h4_swing_lows=df["h4_swing_low"].values,
        modal=MODAL_PER_TRADE, leverage=LEVERAGE_SIM[0],
        fee_per_side=FEE_PER_SIDE, slippage=SLIPPAGE_PER_SIDE,
        max_hold=MAX_HOLDING_BARS, min_rr=SWING_LABEL_MIN_RR,
        min_tp_atr=SWING_LABEL_MIN_TP, max_sl_atr=SWING_LABEL_MAX_SL,
        tp_fallback_atr=TP_SL_FALLBACK_TP, sl_fallback_atr=TP_SL_FALLBACK_SL,
        guardian_enabled=False,
    )
    trades = result.get("total_trades", 0)
    pnl = result.get("total_pnl", 0.0)
    wins = result.get("wins", 0)
    total_trades += trades
    total_pnl += pnl
    total_wins += wins
    print(f"  {sym}: {trades:>4}t  WR={wins/trades*100 if trades else 0:>5.1f}%  PnL=${pnl:>8.2f}")

print(f"\nTOTAL: {int(total_trades)} trades, WR={total_wins/total_trades*100:.1f}%, PnL=${total_pnl:.2f}")
