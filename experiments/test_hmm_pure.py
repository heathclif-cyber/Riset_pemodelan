"""Test pure HMM signal and LGBM+HMM ensemble."""
import sys, json, joblib, numpy as np, pandas as pd
from pathlib import Path
ROOT = Path(__file__).parent.parent; sys.path.insert(0, str(ROOT))

from config import MODEL_DIR, HOLDOUT_DIR, TRAINING_COINS, LABEL_MAP
from config import (
    MODAL_PER_TRADE, LEVERAGE_SIM, FEE_PER_SIDE, SLIPPAGE_PER_SIDE,
    MAX_HOLDING_BARS, SWING_LABEL_MIN_RR, SWING_LABEL_MIN_TP, SWING_LABEL_MAX_SL,
    TP_SL_FALLBACK_TP, TP_SL_FALLBACK_SL, GUARDIAN_EXIT_THRESHOLD, GUARDIAN_DYNAMIC_FEATURES,
)
from pipeline.backtest_utils import compute_guardian_static_array, hierarchical_predict
from core.evaluator import simulate_trades_swing
import pipeline.backtest_utils as btu

btu.SMART_ENTRY_MODE = 'disabled'; btu.LSTM_CONFIRMATION_ENABLED = False
btu.LSTM_FLAT_REVIEW_ENABLED = False
lgbm = joblib.load(MODEL_DIR / 'lgbm_baseline.pkl')
lstm_feat_cols = json.load(open(MODEL_DIR / 'feature_cols_lstm_temporal.json'))
guardian = joblib.load(MODEL_DIR / 'guardian_best.pkl')
g_scaler = joblib.load(MODEL_DIR / 'guardian_scaler.pkl')
g_feats = json.load(open(MODEL_DIR / 'guardian_feature_cols.json'))
g_static_cols = [c for c in g_feats if c not in set(GUARDIAN_DYNAMIC_FEATURES)]
NON_FEATURE_COLS = {'label', 'h4_swing_high', 'h4_swing_low'}

all_coin_data = {}
for coin in TRAINING_COINS[:5]:
    path = HOLDOUT_DIR / 'labeled' / f'{coin}_features_v3.parquet'
    prob_path = HOLDOUT_DIR / 'labeled' / f'{coin}_hmm_probs.parquet'
    if not path.exists() or not prob_path.exists(): continue
    df = pd.read_parquet(path).sort_index()
    probs = pd.read_parquet(prob_path).sort_index()
    for i in range(4): df[f'hmm_prob_{i}'] = probs[f'hmm_prob_{i}']
    reg_path = HOLDOUT_DIR / 'labeled' / f'{coin}_regime_h1.parquet'
    if reg_path.exists():
        reg = pd.read_parquet(reg_path)
        if 'hmm_regime_enc' in df.columns: df = df.drop(columns=['hmm_regime_enc'])
        df = df.join(reg[['hmm_regime_enc']], how='left')
        df['hmm_regime_enc'] = df['hmm_regime_enc'].fillna(1).astype('int32')
    mask = df['label'].astype(str).isin(LABEL_MAP); df = df[mask].copy()
    if len(df) >= 50: all_coin_data[coin] = df

def run_sim(y_pred, conf):
    all_trades = []
    for coin, df in all_coin_data.items():
        n = len(df)
        X_guardian = compute_guardian_static_array(df, g_static_cols)
        atr = df['atr_14_h1'].values if 'atr_14_h1' in df.columns else np.ones(n)
        close = df['close'].values
        high = df['high'].values if 'high' in df.columns else close
        low = df['low'].values if 'low' in df.columns else close
        sh = df['h4_swing_high'].values if 'h4_swing_high' in df.columns else np.full(n, np.nan)
        sl = df['h4_swing_low'].values if 'h4_swing_low' in df.columns else np.full(n, np.nan)
        result = simulate_trades_swing(
            y_pred=y_pred, close=close, high=high, low=low, atr=atr,
            h4_swing_highs=sh, h4_swing_lows=sl,
            modal=MODAL_PER_TRADE, leverage=LEVERAGE_SIM[0],
            fee_per_side=FEE_PER_SIDE, slippage=SLIPPAGE_PER_SIDE,
            max_hold=MAX_HOLDING_BARS, min_rr=SWING_LABEL_MIN_RR, min_tp_atr=SWING_LABEL_MIN_TP,
            max_sl_atr=SWING_LABEL_MAX_SL,
            tp_fallback_atr=TP_SL_FALLBACK_TP, sl_fallback_atr=TP_SL_FALLBACK_SL,
            confidence=conf, guardian_enabled=True,
            guardian_model=guardian, guardian_scaler=g_scaler,
            X_guardian=X_guardian, guardian_exit_threshold=GUARDIAN_EXIT_THRESHOLD,
            guardian_min_hold_bars=2,
        )
        trades = result.get('trades', [])
        for t in trades: t['coin'] = coin
        all_trades.extend(trades)
    return all_trades

print('PURE HMM & LGBM+HMM ENSEMBLE')
print(f'{"Config":<40} {"Trades":>6} {"WR%":>6} {"PnL":>8}')

# LGBM-only baseline
y_all = np.concatenate([np.ones(len(df), dtype=np.int64) for df in all_coin_data.values()])
c_all = np.full(len(y_all), 0.5)
offset = 0
for coin, df in all_coin_data.items():
    n = len(df)
    X = np.zeros((n, len(lstm_feat_cols)))
    for i, col in enumerate(lstm_feat_cols):
        if col in df.columns: X[:, i] = df[col].ffill().fillna(0).values
    yp, cf = hierarchical_predict(None, lgbm, None, None, X, lstm_feat_cols, [], df, trend_alignment_enabled=False)
    below = (yp != 1) & (cf < 0.59); yp[below] = 1
    y_all[offset:offset+n] = yp; c_all[offset:offset+n] = cf
    offset += n
trades = run_sim(y_all, c_all)
n=len(trades); wins=[t for t in trades if t.get('net_pnl',0)>0]
wr=len(wins)/n*100 if n else 0; pnl=sum(t.get('net_pnl',0) for t in trades)
print(f'LGBM-ONLY baseline                          {n:>6} {wr:>5.1f}% {pnl:>8.1f}')

# HMM-only
for thr in [0.30, 0.35, 0.40, 0.45]:
    y_all = np.ones(len(y_all), dtype=np.int64); c_all = np.full(len(y_all), 0.5)
    offset = 0
    for coin, df in all_coin_data.items():
        n = len(df)
        for i in range(n):
            pu = df['hmm_prob_3'].iloc[i]; pd = df['hmm_prob_0'].iloc[i]
            if pu > thr: y_all[offset+i] = 2; c_all[offset+i] = pu
            elif pd > thr: y_all[offset+i] = 0; c_all[offset+i] = pd
        offset += n
    below = (y_all != 1) & (c_all < 0.59); y_all[below] = 1
    trades = run_sim(y_all, c_all)
    n=len(trades); wins=[t for t in trades if t.get('net_pnl',0)>0]
    wr=len(wins)/n*100 if n else 0; pnl=sum(t.get('net_pnl',0) for t in trades)
    print(f'HMM-only thr={thr}                            {n:>6} {wr:>5.1f}% {pnl:>8.1f}')

# LGBM + HMM soft vote
for hmm_w in [0.15, 0.25, 0.35]:
    for hmm_thr in [0.35, 0.40]:
        y_all = np.ones(len(y_all), dtype=np.int64); c_all = np.full(len(y_all), 0.5)
        offset = 0
        for coin, df in all_coin_data.items():
            n = len(df)
            X = np.zeros((n, len(lstm_feat_cols)))
            for i, col in enumerate(lstm_feat_cols):
                if col in df.columns: X[:, i] = df[col].ffill().fillna(0).values
            yp_lgbm, cf_lgbm = hierarchical_predict(
                None, lgbm, None, None, X, lstm_feat_cols, [], df, trend_alignment_enabled=False)
            for i in range(n):
                ld = yp_lgbm[i]; lc = cf_lgbm[i]
                pu = df['hmm_prob_3'].iloc[i]; pd = df['hmm_prob_0'].iloc[i]
                if ld == 2:  # LGBM LONG
                    bonus = hmm_w * (pu - pd)
                    ac = float(np.clip(lc + bonus, 0, 1))
                    if ac >= 0.59: y_all[offset+i] = 2; c_all[offset+i] = ac
                elif ld == 0:  # LGBM SHORT
                    bonus = hmm_w * (pd - pu)
                    ac = float(np.clip(lc + bonus, 0, 1))
                    if ac >= 0.59: y_all[offset+i] = 0; c_all[offset+i] = ac
                elif ld == 1:  # LGBM FLAT, HMM standalone
                    if pu > hmm_thr: y_all[offset+i] = 2; c_all[offset+i] = pu
                    elif pd > hmm_thr: y_all[offset+i] = 0; c_all[offset+i] = pd
            offset += n
        below = (y_all != 1) & (c_all < 0.59); y_all[below] = 1
        trades = run_sim(y_all, c_all)
        n=len(trades); wins=[t for t in trades if t.get('net_pnl',0)>0]
        wr=len(wins)/n*100 if n else 0; pnl=sum(t.get('net_pnl',0) for t in trades)
        print(f'LGBM+HMM w={hmm_w} thr={hmm_thr}                     {n:>6} {wr:>5.1f}% {pnl:>8.1f}')
