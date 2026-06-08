"""Robustness analysis: entry delay, regime breakdown, monthly PnL."""
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
g_static = [c for c in g_feats if c not in set(GUARDIAN_DYNAMIC_FEATURES)]
NON_FEATURE_COLS = {'label', 'h4_swing_high', 'h4_swing_low'}

all_data = {}
for coin in TRAINING_COINS:
    path = HOLDOUT_DIR / 'labeled' / f'{coin}_features_v3.parquet'
    pp = HOLDOUT_DIR / 'labeled' / f'{coin}_hmm_probs.parquet'
    rp = HOLDOUT_DIR / 'labeled' / f'{coin}_regime_h1.parquet'
    if not path.exists(): continue
    df = pd.read_parquet(path).sort_index()
    if pp.exists():
        probs = pd.read_parquet(pp).sort_index()
        for i in range(4): df[f'hmm_prob_{i}'] = probs[f'hmm_prob_{i}']
    if rp.exists():
        reg = pd.read_parquet(rp)
        if 'hmm_regime_enc' in df.columns: df = df.drop(columns=['hmm_regime_enc'])
        df = df.join(reg[['hmm_regime_enc']], how='left')
        df['hmm_regime_enc'] = df['hmm_regime_enc'].fillna(1).astype('int32')
    mask = df['label'].astype(str).isin(LABEL_MAP); df = df[mask].copy()
    if len(df) >= 50: all_data[coin] = df

def run_backtest(delay_bars=0, coin_filter=None):
    """Run backtest with optional entry delay."""
    coins = coin_filter if coin_filter else list(all_data.keys())
    all_trades = []
    for coin in coins:
        df = all_data[coin]; n = len(df)
        X = np.zeros((n, len(lstm_feat_cols)))
        for i, col in enumerate(lstm_feat_cols):
            if col in df.columns: X[:, i] = df[col].ffill().fillna(0).values
        yp, cf = hierarchical_predict(None, lgbm, None, None, X, lstm_feat_cols, [], df,
                                       trend_alignment_enabled=True)

        if delay_bars > 0:
            # Shift predictions forward: entry happens N bars AFTER signal
            yp_delayed = np.ones(n, dtype=np.int64)
            cf_delayed = np.full(n, 1.0/3)
            for i in range(n - delay_bars):
                if yp[i] != 1 and cf[i] >= 0.59:
                    # Enter at bar i+delay with adjusted entry price
                    yp_delayed[i + delay_bars] = yp[i]
                    cf_delayed[i + delay_bars] = cf[i]
            yp, cf = yp_delayed, cf_delayed

        below = (yp != 1) & (cf < 0.59); yp[below] = 1
        Xg = compute_guardian_static_array(df, g_static)
        atr = df['atr_14_h1'].values if 'atr_14_h1' in df.columns else np.ones(n)
        close = df['close'].values; high = df['high'].values if 'high' in df.columns else close
        low = df['low'].values if 'low' in df.columns else close
        sh = df['h4_swing_high'].values if 'h4_swing_high' in df.columns else np.full(n, np.nan)
        sl = df['h4_swing_low'].values if 'h4_swing_low' in df.columns else np.full(n, np.nan)
        r = simulate_trades_swing(
            y_pred=yp, close=close, high=high, low=low, atr=atr,
            h4_swing_highs=sh, h4_swing_lows=sl,
            modal=MODAL_PER_TRADE, leverage=LEVERAGE_SIM[0],
            fee_per_side=FEE_PER_SIDE, slippage=SLIPPAGE_PER_SIDE,
            max_hold=MAX_HOLDING_BARS, min_rr=SWING_LABEL_MIN_RR, min_tp_atr=SWING_LABEL_MIN_TP,
            max_sl_atr=SWING_LABEL_MAX_SL,
            tp_fallback_atr=TP_SL_FALLBACK_TP, sl_fallback_atr=TP_SL_FALLBACK_SL,
            confidence=cf, guardian_enabled=True,
            guardian_model=guardian, guardian_scaler=g_scaler,
            X_guardian=Xg, guardian_exit_threshold=GUARDIAN_EXIT_THRESHOLD,
            guardian_min_hold_bars=2,
        )
        for t in r.get('trades', []):
            t['coin'] = coin
            t['timestamp'] = df.index[t['bar_in']]
            t['regime'] = int(df['hmm_regime_enc'].iloc[t['bar_in']]) if 'hmm_regime_enc' in df.columns else -1
        all_trades.extend(r.get('trades', []))
    return all_trades

# ===== 1. ENTRY DELAY ROBUSTNESS =====
print('=' * 70)
print('  1. ROBUSTNESS: ENTRY DELAY')
print('=' * 70)
print(f'  {"Delay":<10} {"Trades":>7} {"WR%":>7} {"PnL":>9} {"PF":>7}')
print('  ' + '-' * 45)

for delay in [0, 1, 2]:
    trades = run_backtest(delay_bars=delay)
    n = len(trades); wins = [t for t in trades if t.get('net_pnl', 0) > 0]
    wr = len(wins)/n*100 if n else 0; pnl = sum(t.get('net_pnl', 0) for t in trades)
    gw = sum(t.get('net_pnl', 0) for t in wins)
    gl = abs(sum(t.get('net_pnl', 0) for t in trades if t.get('net_pnl', 0) <= 0))
    pf = gw/gl if gl > 0 else float('inf')
    impact = (pnl - 848) / 848 * 100 if delay == 0 else 0
    print(f'  {delay:<10} {n:>7} {wr:>6.1f}% {pnl:>8.1f} {pf:>6.2f}')
    if delay == 0: base_pnl = pnl

# ===== 2. REGIME BREAKDOWN =====
print()
print('=' * 70)
print('  2. BREAKDOWN PER HMM REGIME (delay=0)')
print('=' * 70)
trades = run_backtest(delay_bars=0)
regime_names = {0: 'TRENDING_DOWN', 1: 'RANGING_LOW_VOL', 2: 'RANGING_HIGH_VOL', 3: 'TRENDING_UP'}
regime_pcts = {0: 3, 1: 38, 2: 52, 3: 7}  # approximate from holdout distribution
print(f'  {"Regime":<20} {"%Bars":>7} {"Trades":>7} {"WR%":>7} {"PnL":>9} {"PF":>7}')
print('  ' + '-' * 60)

for regime in [0, 1, 2, 3]:
    rt = [t for t in trades if t.get('regime') == regime]
    n = len(rt)
    if n == 0:
        print(f'  {regime_names[regime]:<20} {regime_pcts[regime]:>6}% {0:>7} {"-":>7} {"-":>9} {"-":>7}')
        continue
    wins = [t for t in rt if t.get('net_pnl', 0) > 0]
    wr = len(wins)/n*100
    pnl = sum(t.get('net_pnl', 0) for t in rt)
    gw = sum(t.get('net_pnl', 0) for t in wins)
    gl = abs(sum(t.get('net_pnl', 0) for t in rt if t.get('net_pnl', 0) <= 0))
    pf = gw/gl if gl > 0 else float('inf')
    flag = '  <-- WATCH' if wr < 60 else ''
    print(f'  {regime_names[regime]:<20} {regime_pcts[regime]:>5.0f}% {n:>7} {wr:>6.1f}% {pnl:>8.1f} {pf:>6.2f}{flag}')

# ===== 3. MONTHLY PNL DISTRIBUTION =====
print()
print('=' * 70)
print('  3. MONTHLY PNL DISTRIBUTION (delay=0)')
print('=' * 70)
months = {}
for t in trades:
    ts = pd.Timestamp(t['timestamp'])
    m = ts.strftime('%Y-%m')
    months[m] = months.get(m, {'pnl': 0, 'trades': 0, 'wins': 0})
    months[m]['pnl'] += t.get('net_pnl', 0)
    months[m]['trades'] += 1
    if t.get('net_pnl', 0) > 0: months[m]['wins'] += 1

print(f'  {"Month":<10} {"Trades":>7} {"WR%":>7} {"PnL":>9} {"CumPnL":>9}')
print('  ' + '-' * 50)
cum = 0; neg_months = 0
for m in sorted(months.keys()):
    data = months[m]
    wr = data['wins']/data['trades']*100 if data['trades'] > 0 else 0
    cum += data['pnl']
    if data['pnl'] < 0: neg_months += 1
    flag = ' <--' if data['pnl'] < 0 else ''
    print(f'  {m:<10} {data["trades"]:>7} {wr:>6.1f}% {data["pnl"]:>8.1f} {cum:>8.1f}{flag}')

print(f'\n  Total months: {len(months)} | Negative: {neg_months} | Positive: {len(months)-neg_months}')
print(f'  Monthly PnL: mean={np.mean([m["pnl"] for m in months.values()]):.1f} std={np.std([m["pnl"] for m in months.values()]):.1f}')

# ===== 4. PER-COIN WR RANKING =====
print()
print('=' * 70)
print('  4. PER-COIN WR & PnL (delay=0)')
print('=' * 70)
coin_metrics = {}
for t in trades:
    c = t['coin']
    if c not in coin_metrics: coin_metrics[c] = {'pnl': 0, 'trades': 0, 'wins': 0}
    coin_metrics[c]['pnl'] += t.get('net_pnl', 0)
    coin_metrics[c]['trades'] += 1
    if t.get('net_pnl', 0) > 0: coin_metrics[c]['wins'] += 1

print(f'  {"Coin":<15} {"Trades":>7} {"WR%":>7} {"PnL":>9}')
print('  ' + '-' * 45)
for c in sorted(coin_metrics, key=lambda x: coin_metrics[x]['pnl'], reverse=True):
    m = coin_metrics[c]
    wr = m['wins']/m['trades']*100
    flag = ' <-- ' if wr < 60 else ''
    print(f'  {c:<15} {m["trades"]:>7} {wr:>6.1f}% {m["pnl"]:>8.1f}{flag}')
