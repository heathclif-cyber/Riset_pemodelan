"""Sweep TB v3 thresholds on ORIGINAL ic32 holdout (swing labels)"""
import json, sys, warnings, numpy as np, pandas as pd
warnings.filterwarnings('ignore')
sys.path.insert(0, '.')
from config import *
import joblib
from core.evaluator import full_trading_report
from core.utils import ensure_utc_index

tb_model = joblib.load('models/runs/tb_lgbm_widyawardhana_v3/lgbm.pkl')
with open('models/runs/tb_lgbm_widyawardhana_v3/tb_lgbm_widyawardhana_v3_features.json') as f:
    tb_feats = json.load(f)

bl_model = joblib.load(MODEL_DIR / 'lgbm_baseline.pkl')
if hasattr(bl_model, 'feature_name_') and bl_model.feature_name_:
    bl_feats = list(bl_model.feature_name_)
else:
    with open(MODEL_DIR / 'feature_cols_v2.json') as f:
        bl_feats = json.load(f)

available = [s for s in ALL_COINS if (HOLDOUT_DIR / 'labeled' / f'{s}_features_v3.parquet').exists()]

THRESHOLDS = [0.45, 0.48, 0.50, 0.55, 0.60, 0.65]
LABEL_MAP_IC32 = {"SHORT": 0, "FLAT": 1, "LONG": 2}

# Precompute baseline once (same for all thresholds)
print("Computing baseline (ic32) on swing holdout...")
bl_results = {}
for sym in available:
    df = pd.read_parquet(HOLDOUT_DIR / 'labeled' / f'{sym}_features_v3.parquet')
    df = ensure_utc_index(df).sort_index()
    mask = df['label'].isin(LABEL_MAP_IC32)
    df = df[mask].copy()
    y_true = df['label'].map(LABEL_MAP_IC32).values.astype(np.int32)

    atr_arr = df['atr_14_h1'].values; close_arr = df['close'].values
    high_arr = df['high'].values; low_arr = df['low'].values
    h4_sh = df['h4_swing_high'].values if 'h4_swing_high' in df.columns else None
    h4_sl = df['h4_swing_low'].values if 'h4_swing_low' in df.columns else None
    h4_trend_arr = df['h4_trend'].values if 'h4_trend' in df.columns else None

    X_bl = np.zeros((len(df), len(bl_feats)), dtype=np.float64)
    for idx, col in enumerate(bl_feats):
        if col in df.columns:
            X_bl[:, idx] = df[col].ffill().fillna(0).values.astype(np.float64)
    proba_bl = bl_model.predict_proba(X_bl)
    conf_bl = np.max(proba_bl, axis=1)
    y_bl = np.argmax(proba_bl, axis=1)
    for i in range(len(y_bl)):
        if y_bl[i] == 2 and proba_bl[i, 2] < LGBM_THRESHOLD_LONG: y_bl[i] = 1
        elif y_bl[i] == 0 and proba_bl[i, 0] < LGBM_THRESHOLD_SHORT: y_bl[i] = 1
    y_bl[(y_bl != 1) & (conf_bl < CONFIDENCE_THRESHOLD_ENTRY)] = 1

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
            h4_trend=h4_trend_arr,
        )
        lev5 = report_bl.get('lev5x', report_bl)
        trades = lev5.get('trades', [])
        wins = sum(1 for t in trades if t.get('outcome') == 'WIN')
        bl_pnl = sum(t.get('net_pnl', 0) for t in trades)
        bl_trades = len(trades)
        bl_wr = wins / max(bl_trades, 1) * 100
    else:
        bl_trades, bl_pnl, bl_wr = 0, 0.0, 0.0

    bl_results[sym] = {'trades': bl_trades, 'pnl': bl_pnl, 'wr': bl_wr}

    # Precompute TB features
    X_tb = np.zeros((len(df), len(tb_feats)), dtype=np.float64)
    for idx, col in enumerate(tb_feats):
        if col in df.columns:
            X_tb[:, idx] = df[col].ffill().fillna(0).values.astype(np.float64)

    bl_results[sym] = {**bl_results[sym], 'X_tb': X_tb, 'y_true': y_true,
                        'atr': atr_arr, 'close': close_arr, 'high': high_arr,
                        'low': low_arr, 'h4_sh': h4_sh, 'h4_sl': h4_sl,
                        'index': df.index, 'h4_trend': h4_trend_arr}

# Sweep TB thresholds
for THRESH in THRESHOLDS:
    sep = '=' * 75
    print(f'\n{sep}')
    print(f'  TB v3 -- thresh={THRESH} (on ic32 swing-labeled holdout)')
    print(f'{sep}')
    hdr = f'{"Coin":<15} {"TB Tr":>7} {"TB WR":>7} {"TB PnL":>9} {"BL Tr":>7} {"BL WR":>7} {"BL PnL":>9}'
    print(hdr)
    print('-' * 67)

    tb_total_t, tb_total_pnl, tb_total_wins = 0, 0.0, 0
    bl_total_t, bl_total_pnl, bl_total_wins = 0, 0.0, 0

    for sym in available:
        d = bl_results[sym]
        proba_tb = tb_model.predict_proba(d['X_tb'])
        conf_tb = np.max(proba_tb, axis=1)
        y_tb = np.argmax(proba_tb, axis=1)
        y_tb[(y_tb != 1) & (conf_tb < THRESH)] = 1

        tb_trades, tb_pnl, tb_wr = 0, 0.0, 0.0
        if (y_tb != 1).sum() > 0:
            report = full_trading_report(
                y_pred=y_tb, y_actual=d['y_true'], atr=d['atr'],
                close=d['close'], high=d['high'], low=d['low'],
                h4_swing_highs=d['h4_sh'], h4_swing_lows=d['h4_sl'],
                index=d['index'], modal=MODAL_PER_TRADE, leverages=LEVERAGE_SIM,
                fee_per_side=FEE_PER_SIDE, slippage=SLIPPAGE_PER_SIDE,
                min_rr=SWING_LABEL_MIN_RR, min_tp_atr=SWING_LABEL_MIN_TP,
                max_sl_atr=SWING_LABEL_MAX_SL, tp_fallback_atr=TP_SL_FALLBACK_TP,
                sl_fallback_atr=TP_SL_FALLBACK_SL, max_hold=MAX_HOLDING_BARS,
                symbol=sym, confidence=conf_tb,
                guardian_enabled=False, trailing_stop_enabled=False,
                h4_trend=d.get('h4_trend'),
            )
            lev5 = report.get('lev5x', report)
            trades = lev5.get('trades', [])
            wins = sum(1 for t in trades if t.get('outcome') == 'WIN')
            tb_pnl = sum(t.get('net_pnl', 0) for t in trades)
            tb_trades = len(trades)
            tb_wr = wins / max(tb_trades, 1) * 100

        tb_total_t += tb_trades; tb_total_pnl += tb_pnl
        tb_total_wins += int(tb_trades * tb_wr / 100)

        bl_t = d['trades']; bl_p = d['pnl']; bl_w = d['wr']
        bl_total_t += bl_t; bl_total_pnl += bl_p; bl_total_wins += int(bl_t * bl_w / 100)

        print(f'{sym:<15} {tb_trades:>7} {tb_wr:>6.1f}% {tb_pnl:>+8.0f} {bl_t:>7} {bl_w:>6.1f}% {bl_p:>+8.0f}')

    print('-' * 67)
    tb_agg_wr = tb_total_wins / max(tb_total_t, 1) * 100
    bl_agg_wr = bl_total_wins / max(bl_total_t, 1) * 100
    print(f'{"AGGREGATE":<15} {tb_total_t:>7} {tb_agg_wr:>6.1f}% {tb_total_pnl:>+8.0f} {bl_total_t:>7} {bl_agg_wr:>6.1f}% {bl_total_pnl:>+8.0f}')
    print(f'  TB: PnL/trade=${tb_total_pnl/max(tb_total_t,1):.2f} | Trades/koin={tb_total_t//len(available)}')
