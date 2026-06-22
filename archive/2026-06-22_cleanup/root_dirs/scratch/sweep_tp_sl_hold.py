"""Sweep TP/SL/max_hold for TB v3 — find optimal parameters"""
import json, sys, warnings, numpy as np, pandas as pd
warnings.filterwarnings('ignore')
sys.path.insert(0, '.')
from config import *
import joblib, lightgbm as lgb
from core.features import triple_barrier_labeling
from core.evaluator import simulate_trades_swing, full_trading_report
from core.utils import ensure_utc_index
from pipeline.shared import build_purged_folds
from sklearn.metrics import f1_score

TB_FEATURES = [
    "etf_gbtc_change_usd", "etf_total_change_usd", "cvd_slope_h4",
    "ofi_h4_delta", "wyckoff_phase", "Sell_Liq", "atr_percentile_h1",
    "stochrsi_k", "dist_liq_50x_short", "funding_rate", "ema_7_h1",
    "dow_cos", "cvd_div_h4", "dist_swing_low", "VAH",
    "cvd_momentum_adv", "dist_from_8h_high", "ema_200_h1",
]

TB_PARAMS = {
    "objective": "multiclass", "num_class": 3,
    "n_estimators": 500, "learning_rate": 0.03,
    "max_depth": 5, "num_leaves": 31, "min_child_samples": 50,
    "subsample": 0.8, "colsample_bytree": 0.7,
    "class_weight": "balanced",
    "verbose": -1, "n_jobs": -1, "random_state": 42,
}

# Test on 3 coins for speed
test_coins = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT']
LM = {'SHORT': 0, 'FLAT': 1, 'LONG': 2}

# Pre-load holdout data
holdout_data = {}
for sym in test_coins:
    df = pd.read_parquet(HOLDOUT_DIR / 'labeled' / f'{sym}_features_v3.parquet')
    df = ensure_utc_index(df).sort_index()
    hm = df[df['label'].isin(LM)].copy()
    holdout_data[sym] = {
        'close': hm['close'].values, 'high': hm['high'].values, 'low': hm['low'].values,
        'atr': hm['atr_14_h1'].values, 'yt': hm['label'].map(LM).values.astype(np.int32),
        'h4_sh': hm['h4_swing_high'].values if 'h4_swing_high' in hm.columns else None,
        'h4_sl': hm['h4_swing_low'].values if 'h4_swing_low' in hm.columns else None,
        'ht': hm['h4_trend'].values if 'h4_trend' in hm.columns else None,
        'df': hm, 'X': None,
    }
    # Pre-build feature matrix
    X = np.zeros((len(hm), len(TB_FEATURES)), dtype=np.float64)
    for i, f in enumerate(TB_FEATURES):
        if f in hm.columns: X[:, i] = hm[f].ffill().fillna(0).values.astype(np.float64)
    holdout_data[sym]['X'] = X

print(f"{'='*85}")
print(f"  TP/SL/MAX_HOLD SWEEP — TB v3")
print(f"  Coins: {test_coins} | Features: {len(TB_FEATURES)}")
print(f"{'='*85}")

# Sweep
combos = []
for tp in [1.5, 2.0, 2.5, 3.0]:
    for sl in [1.0, 1.5, 2.0]:
        for mh in [24, 36, 48]:
            rr = tp / sl
            if rr < 1.2: continue  # RR too low
            if rr > 3.0: continue  # RR too high
            combos.append((tp, sl, mh))

print(f"\nCombinations: {len(combos)} (TP={ {1.5,2.0,2.5,3.0} }, SL={ {1.0,1.5,2.0} }, MH={ {24,36,48} })")
print(f"\n{'TP':>5} {'SL':>5} {'MH':>5} {'RR':>6} {'Train F1':>9} {'Hold Tr':>7} {'WR%':>7} {'PnL':>8} {'P/t':>5}")
print("-" * 65)

results = []
for tp, sl, mh in combos:
    rr = tp / sl

    # Train on training data
    frames = []
    for sym in test_coins:
        path = LABEL_DIR / f'{sym}_features_v3.parquet'
        if not path.exists(): continue
        df = pd.read_parquet(path)
        if not isinstance(df.index, pd.DatetimeIndex): df.index = pd.to_datetime(df.index, utc=True)
        if df.index.tz is None: df.index = df.index.tz_localize("UTC")
        df = df[df.index < TRAIN_CUTOFF_DATE]
        if df.empty: continue
        tb = triple_barrier_labeling(df['close'], df['high'], df['low'], df['atr_14_h1'], tp, sl, mh)
        df['tb_label'] = tb.map({'SHORT': 0, 'FLAT': 1, 'LONG': 2})
        df = df.dropna(subset=['tb_label'])
        if len(df) < 100: continue
        frames.append(df)

    if len(frames) < 2: continue
    df_train = pd.concat(frames).sort_index()
    avail_f = [c for c in TB_FEATURES if c in df_train.columns]
    X_tr = df_train[avail_f].ffill().fillna(0)
    y_tr = df_train['tb_label'].values.astype(np.int32)

    # Quick CV (4 folds for speed)
    folds = build_purged_folds(X_tr.index, 4, 20)
    f1s = []
    for tr_idx, val_idx in folds:
        model = lgb.LGBMClassifier(**TB_PARAMS)
        model.fit(X_tr.iloc[tr_idx], y_tr[tr_idx],
                  eval_set=[(X_tr.iloc[val_idx], y_tr[val_idx])],
                  callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(period=-1)])
        yp = model.predict(X_tr.iloc[val_idx])
        f1s.append(f1_score(y_tr[val_idx], yp, average='macro', zero_division=0))

    cv_f1 = np.mean(f1s)

    # Full retrain
    final_model = lgb.LGBMClassifier(**TB_PARAMS)
    final_model.fit(X_tr, y_tr)

    # Holdout test
    total_t, total_w, total_pnl = 0, 0, 0.0
    for sym in test_coins:
        d = holdout_data[sym]
        proba = final_model.predict_proba(d['X'])
        conf = np.max(proba, axis=1)
        yp = np.argmax(proba, axis=1)
        yp[(yp != 1) & (conf < 0.48)] = 1  # default threshold

        if (yp != 1).sum() == 0: continue
        rep = full_trading_report(
            y_pred=yp, y_actual=d['yt'], atr=d['atr'], close=d['close'],
            high=d['high'], low=d['low'], h4_swing_highs=d['h4_sh'], h4_swing_lows=d['h4_sl'],
            index=d['df'].index, modal=MODAL_PER_TRADE, leverages=LEVERAGE_SIM,
            fee_per_side=FEE_PER_SIDE, slippage=SLIPPAGE_PER_SIDE,
            min_rr=SWING_LABEL_MIN_RR, min_tp_atr=SWING_LABEL_MIN_TP,
            max_sl_atr=SWING_LABEL_MAX_SL, tp_fallback_atr=tp, sl_fallback_atr=sl,
            max_hold=mh, symbol=sym, confidence=conf,
            guardian_enabled=False, trailing_stop_enabled=False, h4_trend=d.get('ht'),
        )
        lev = rep.get('lev5x', rep); tr = lev.get('trades', [])
        total_t += len(tr)
        total_w += sum(1 for t in tr if 'WIN' in str(t.get('outcome', '')))
        total_pnl += sum(t.get('net_pnl', 0) for t in tr)

    wr = total_w / max(total_t, 1) * 100
    pnl_t = total_pnl / max(total_t, 1)

    results.append({
        'tp': tp, 'sl': sl, 'mh': mh, 'rr': rr, 'cv_f1': round(cv_f1, 4),
        'trades': total_t, 'wr': round(wr, 1), 'pnl': round(total_pnl, 0), 'pnl_t': round(pnl_t, 2)
    })
    print(f"{tp:>5.1f} {sl:>5.1f} {mh:>5} {rr:>6.2f} {cv_f1:>9.4f} {total_t:>7} {wr:>6.1f}% {total_pnl:>+8.0f} {pnl_t:>5.2f}")

# Sort by PnL
results.sort(key=lambda x: x['pnl'], reverse=True)
print(f"\n{'='*85}")
print(f"  TOP 10 CONFIGURATIONS")
print(f"{'='*85}")
print(f"{'Rank':>4} {'TP':>5} {'SL':>5} {'MH':>5} {'RR':>6} {'CV F1':>8} {'Trades':>7} {'WR%':>7} {'PnL':>10} {'P/t':>6}")
print("-" * 75)
for i, r in enumerate(results[:10]):
    print(f"{i+1:>4} {r['tp']:>5.1f} {r['sl']:>5.1f} {r['mh']:>5} {r['rr']:>6.2f} {r['cv_f1']:>8.4f} "
          f"{r['trades']:>7} {r['wr']:>6.1f}% {r['pnl']:>+10.0f} {r['pnl_t']:>6.2f}")
