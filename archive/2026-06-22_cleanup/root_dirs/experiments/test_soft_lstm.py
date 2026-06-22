"""Quick test: LSTM momentum v2 as soft confidence modulator (not hard gate)."""
import sys, json, joblib, numpy as np, pandas as pd
from pathlib import Path
ROOT = Path(__file__).parent.parent; sys.path.insert(0, str(ROOT))

from config import MODEL_DIR, HOLDOUT_DIR, TRAINING_COINS, LABEL_MAP
from config import (
    MODAL_PER_TRADE, LEVERAGE_SIM, FEE_PER_SIDE, SLIPPAGE_PER_SIDE,
    MAX_HOLDING_BARS, SWING_LABEL_MIN_RR, SWING_LABEL_MIN_TP, SWING_LABEL_MAX_SL,
    TP_SL_FALLBACK_TP, TP_SL_FALLBACK_SL, GUARDIAN_EXIT_THRESHOLD, GUARDIAN_DYNAMIC_FEATURES,
)
from core.models import load_lstm
from pipeline.backtest_utils import compute_guardian_static_array, hierarchical_predict
from core.evaluator import simulate_trades_swing
import pipeline.backtest_utils as btu

lgbm = joblib.load(MODEL_DIR / 'lgbm_baseline.pkl')
lstm_model = load_lstm(MODEL_DIR / 'lstm_best.pt')
lstm_scaler = joblib.load(MODEL_DIR / 'lstm_scaler.pkl')
lstm_feat_cols = json.load(open(MODEL_DIR / 'feature_cols_lstm_temporal.json'))
guardian = joblib.load(MODEL_DIR / 'guardian_best.pkl')
g_scaler = joblib.load(MODEL_DIR / 'guardian_scaler.pkl')
g_feats = json.load(open(MODEL_DIR / 'guardian_feature_cols.json'))
g_static_cols = [c for c in g_feats if c not in set(GUARDIAN_DYNAMIC_FEATURES)]
NON_FEATURE_COLS = {'label', 'h4_swing_high', 'h4_swing_low'}

all_coin_data = {}
for coin in TRAINING_COINS[:5]:
    path = HOLDOUT_DIR / 'labeled' / f'{coin}_features_v3.parquet'
    if not path.exists(): continue
    df = pd.read_parquet(path).sort_index()
    reg_path = HOLDOUT_DIR / 'labeled' / f'{coin}_regime_h1.parquet'
    if reg_path.exists():
        reg = pd.read_parquet(reg_path)
        if 'hmm_regime_enc' in df.columns: df = df.drop(columns=['hmm_regime_enc'])
        df = df.join(reg[['hmm_regime_enc']], how='left')
        df['hmm_regime_enc'] = df['hmm_regime_enc'].fillna(1).astype('int32')
    mask = df['label'].astype(str).isin(LABEL_MAP); df = df[mask].copy()
    if len(df) >= 50: all_coin_data[coin] = df

def run_config(label, lstm_on, opp_pen, neut_pen, trend_on):
    btu.SMART_ENTRY_MODE = 'disabled'
    btu.LSTM_CONFIRMATION_ENABLED = lstm_on
    btu.LSTM_FLAT_REVIEW_ENABLED = lstm_on
    btu.LSTM_DIRECTIONAL_REVIEW_THRESHOLD = 0.35
    btu.LSTM_ADJUST_OPPOSITE_PEN = opp_pen
    btu.LSTM_ADJUST_NEUTRAL_PEN = neut_pen
    btu.LSTM_ADJUST_AGREE_BOOST = 0.05

    all_trades = []
    for coin, df in all_coin_data.items():
        n = len(df)
        X = np.zeros((n, len(lstm_feat_cols)), dtype=np.float64)
        for i, col in enumerate(lstm_feat_cols):
            if col in df.columns: X[:, i] = df[col].ffill().fillna(0).values
        y_pred, confidence = hierarchical_predict(
            None, lgbm, lstm_model if lstm_on else None,
            lstm_scaler if lstm_on else None,
            X, lstm_feat_cols, [], df, trend_alignment_enabled=trend_on,
        )
        below = (y_pred != 1) & (confidence < 0.59); y_pred[below] = 1
        X_guardian = compute_guardian_static_array(df, g_static_cols)
        atr = df['atr_14_h1'].values if 'atr_14_h1' in df.columns else np.ones(n)
        close = df['close'].values; high = df['high'].values if 'high' in df.columns else close
        low = df['low'].values if 'low' in df.columns else close
        sh = df['h4_swing_high'].values if 'h4_swing_high' in df.columns else np.full(n, np.nan)
        sl = df['h4_swing_low'].values if 'h4_swing_low' in df.columns else np.full(n, np.nan)
        result = simulate_trades_swing(
            y_pred=y_pred, close=close, high=high, low=low, atr=atr,
            h4_swing_highs=sh, h4_swing_lows=sl,
            modal=MODAL_PER_TRADE, leverage=LEVERAGE_SIM[0],
            fee_per_side=FEE_PER_SIDE, slippage=SLIPPAGE_PER_SIDE,
            max_hold=MAX_HOLDING_BARS,
            min_rr=SWING_LABEL_MIN_RR, min_tp_atr=SWING_LABEL_MIN_TP,
            max_sl_atr=SWING_LABEL_MAX_SL,
            tp_fallback_atr=TP_SL_FALLBACK_TP, sl_fallback_atr=TP_SL_FALLBACK_SL,
            confidence=confidence, guardian_enabled=True,
            guardian_model=guardian, guardian_scaler=g_scaler,
            X_guardian=X_guardian, guardian_exit_threshold=GUARDIAN_EXIT_THRESHOLD,
            guardian_min_hold_bars=2,
        )
        trades = result.get('trades', [])
        for t in trades: t['coin'] = coin
        all_trades.extend(trades)

    n = len(all_trades)
    wins = [t for t in all_trades if t.get('net_pnl', 0) > 0]
    wr = len(wins) / n * 100 if n else 0
    pnl = sum(t.get('net_pnl', 0) for t in all_trades)
    long_t = [t for t in all_trades if t.get('direction') == 'LONG']
    short_t = [t for t in all_trades if t.get('direction') == 'SHORT']
    long_wr = len([t for t in long_t if t.get('net_pnl', 0) > 0]) / len(long_t) * 100 if long_t else 0
    short_wr = len([t for t in short_t if t.get('net_pnl', 0) > 0]) / len(short_t) * 100 if short_t else 0
    print(f'{label:<42} {n:>6} {wr:>5.1f}% {long_wr:>5.1f}% {short_wr:>5.1f}% {pnl:>8.1f}')
    return pnl

print(f'LSTM Momentum V2 sebagai SOFT MODULATOR (bukan hard gate)')
print(f'{"Config":<42} {"Trades":>6} {"WR%":>5} {"L_WR%":>5} {"S_WR%":>5} {"PnL":>8}')
print('-' * 75)

# BASELINES
run_config('LSTM=OFF trend=OFF', False, 0.65, 0.0, False)
run_config('LSTM=OFF trend=ON', False, 0.65, 0.0, True)

# Old hard consensus (opp=0.65)
run_config('OLD hard_gate opp=0.65 trend=OFF', True, 0.65, 0.0, False)
run_config('OLD hard_gate opp=0.65 trend=ON', True, 0.65, 0.0, True)

# Soft modulator variants
for opp in [0.03, 0.05, 0.08, 0.12, 0.15]:
    run_config(f'SOFT opp={opp:.2f} neut=0.00 trend=OFF', True, opp, 0.00, False)
    run_config(f'SOFT opp={opp:.2f} neut=0.00 trend=ON', True, opp, 0.00, True)
    run_config(f'SOFT opp={opp:.2f} neut=0.02 trend=OFF', True, opp, 0.02, False)
    run_config(f'SOFT opp={opp:.2f} neut=0.02 trend=ON', True, opp, 0.02, True)
