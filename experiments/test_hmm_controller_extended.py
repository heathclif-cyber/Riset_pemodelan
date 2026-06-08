"""Test HMM Controller on 2020-2025 purged CV extended backtest."""
import sys, json, joblib, numpy as np, pandas as pd
from pathlib import Path
ROOT = Path(__file__).parent.parent; sys.path.insert(0, str(ROOT))

from config import MODEL_DIR, LABEL_DIR, TRAINING_COINS, LABEL_MAP
from config import (
    MODAL_PER_TRADE, LEVERAGE_SIM, FEE_PER_SIDE, SLIPPAGE_PER_SIDE,
    MAX_HOLDING_BARS, SWING_LABEL_MIN_RR, SWING_LABEL_MIN_TP, SWING_LABEL_MAX_SL,
    TP_SL_FALLBACK_TP, TP_SL_FALLBACK_SL, GUARDIAN_EXIT_THRESHOLD, GUARDIAN_DYNAMIC_FEATURES,
    TRAIN_CUTOFF_DATE, N_FOLDS, PURGE_GAP_BARS,
)
from pipeline.backtest_utils import compute_guardian_static_array, hierarchical_predict
from core.evaluator import simulate_trades_swing
from pipeline.shared import build_purged_folds
import pipeline.backtest_utils as btu
import lightgbm as lgb

def run_extended(use_hmm_controller):
    btu.SMART_ENTRY_MODE = 'disabled'; btu.LSTM_CONFIRMATION_ENABLED = False
    btu.LSTM_FLAT_REVIEW_ENABLED = False

    guardian = joblib.load(MODEL_DIR / 'guardian_best.pkl')
    g_scaler = joblib.load(MODEL_DIR / 'guardian_scaler.pkl')
    g_feats = json.load(open(MODEL_DIR / 'guardian_feature_cols.json'))
    g_static = [c for c in g_feats if c not in set(GUARDIAN_DYNAMIC_FEATURES)]
    lstm_feat_cols = json.load(open(MODEL_DIR / 'feature_cols_lstm_temporal.json'))
    lgbm_feats = json.load(open(MODEL_DIR / 'feature_cols_v2.json'))
    NON_FEATURE_COLS = {'label', 'h4_swing_high', 'h4_swing_low'}

    all_trades = []
    for coin in TRAINING_COINS:
        feat_path = LABEL_DIR / f'{coin}_features_v3.parquet'
        reg_path = LABEL_DIR / f'{coin}_regime_h1.parquet'
        if not feat_path.exists(): continue
        df = pd.read_parquet(feat_path).sort_index()
        df = df[df.index < TRAIN_CUTOFF_DATE]
        if reg_path.exists():
            reg = pd.read_parquet(reg_path)
            if 'hmm_regime_enc' in df.columns: df = df.drop(columns=['hmm_regime_enc'])
            df = df.join(reg[['hmm_regime_enc']], how='left')
            df['hmm_regime_enc'] = df['hmm_regime_enc'].fillna(1).astype('int32')
        mask = df['label'].astype(str).isin(LABEL_MAP); df = df[mask].copy()
        if len(df) < 500: continue

        ts_index = pd.DatetimeIndex(df.index)
        folds = build_purged_folds(ts_index, N_FOLDS, PURGE_GAP_BARS)

        for fi, (tr_idx, te_idx) in enumerate(folds):
            if len(te_idx) < 100: continue
            df_tr = df.iloc[tr_idx]; df_te = df.iloc[te_idx]
            feat_cols = [c for c in lgbm_feats if c in df_tr.columns]
            X_tr = df_tr[feat_cols].ffill().fillna(0)
            y_tr = df_tr['label'].map(LABEL_MAP).values.astype(np.int64)
            if len(np.unique(y_tr)) < 3: continue

            params = {'objective': 'multiclass', 'num_class': 3, 'n_estimators': 500,
                      'learning_rate': 0.05, 'max_depth': 6, 'num_leaves': 31,
                      'min_child_samples': 50, 'subsample': 0.8, 'colsample_bytree': 0.8,
                      'verbose': -1, 'n_jobs': -1, 'random_state': 42}
            fold_model = lgb.LGBMClassifier(**params)
            fold_model.fit(X_tr, y_tr)

            n_te = len(df_te)
            X_te = np.zeros((n_te, len(feat_cols)), dtype=np.float64)
            for i, col in enumerate(feat_cols):
                if col in df_te.columns: X_te[:, i] = df_te[col].ffill().fillna(0).values

            yp, cf = hierarchical_predict(None, fold_model, None, None, X_te, feat_cols, [], df_te,
                                           trend_alignment_enabled=True)

            # HMM Controller
            if use_hmm_controller:
                for i in range(n_te):
                    regime = int(df_te['hmm_regime_enc'].iloc[i])
                    h4_t = df_te['h4_trend'].iloc[i] if 'h4_trend' in df_te.columns else 0
                    # Block counter-trend in TRENDING
                    if regime == 3 and yp[i] == 0 and h4_t > 0: yp[i] = 1
                    if regime == 0 and yp[i] == 2 and h4_t < 0: yp[i] = 1

            below = (yp != 1) & (cf < 0.59); yp[below] = 1
            Xg = compute_guardian_static_array(df_te, g_static)
            atr = df_te['atr_14_h1'].values if 'atr_14_h1' in df_te.columns else np.ones(n_te)
            close = df_te['close'].values; high = df_te['high'].values if 'high' in df_te.columns else close
            low = df_te['low'].values if 'low' in df_te.columns else close
            sh = df_te['h4_swing_high'].values if 'h4_swing_high' in df_te.columns else np.full(n_te, np.nan)
            sl = df_te['h4_swing_low'].values if 'h4_swing_low' in df_te.columns else np.full(n_te, np.nan)

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
                t['coin'] = coin; t['timestamp'] = df_te.index[t['bar_in']]
            all_trades.extend(r.get('trades', []))

    return all_trades

# Run both
print('Running BASELINE...')
trades_base = run_extended(False)
print('Running HMM CONTROLLER...')
trades_hmm = run_extended(True)

# Compare
def monthly_stats(trades, label):
    months = {}
    for t in trades:
        m = pd.Timestamp(t['timestamp']).strftime('%Y-%m')
        months[m] = months.get(m, {'pnl': 0, 'trades': 0, 'wins': 0})
        months[m]['pnl'] += t.get('net_pnl', 0)
        months[m]['trades'] += 1
        if t.get('net_pnl', 0) > 0: months[m]['wins'] += 1

    all_pnl = [d['pnl'] for d in months.values()]
    n = sum(d['trades'] for d in months.values())
    wins = sum(d['wins'] for d in months.values())
    wr = wins/n*100 if n > 0 else 0
    total_pnl = sum(all_pnl)
    neg_months = sum(1 for p in all_pnl if p < 0)

    # Yearly
    yearly = {}
    for m, d in months.items():
        y = m[:4]; yearly[y] = yearly.get(y, {'pnl':0,'trades':0})
        yearly[y]['pnl'] += d['pnl']; yearly[y]['trades'] += d['trades']

    return months, n, wr, total_pnl, neg_months, all_pnl, yearly

m_base, n_base, wr_base, pnl_base, neg_base, mpnl_base, yr_base = monthly_stats(trades_base, 'BASELINE')
m_hmm, n_hmm, wr_hmm, pnl_hmm, neg_hmm, mpnl_hmm, yr_hmm = monthly_stats(trades_hmm, 'HMM')

print(f'\n{"="*75}')
print(f'  HMM CONTROLLER — EXTENDED BACKTEST (2020-2025, 63 months)')
print(f'{"="*75}')
print(f'  {"Metrik":<30} {"BASELINE":>15} {"HMM CONTROLLER":>15}')
print(f'  {"-"*60}')
print(f'  {"Trades":<30} {n_base:>15,} {n_hmm:>15,}')
print(f'  {"WR":<30} {wr_base:>14.1f}% {wr_hmm:>14.1f}%')
print(f'  {"Total PnL":<30} {pnl_base:>14.1f} {pnl_hmm:>14.1f}')
print(f'  {"Delta":<30} {"":>15} {pnl_hmm-pnl_base:>+14.1f}')
print(f'  {"Negative months":<30} {neg_base:>15} {neg_hmm:>15}')
print(f'  {"Monthly mean PnL":<30} {np.mean(mpnl_base):>14.1f} {np.mean(mpnl_hmm):>14.1f}')
print(f'  {"Monthly std PnL":<30} {np.std(mpnl_base):>14.1f} {np.std(mpnl_hmm):>14.1f}')

print(f'\n  Yearly:')
print(f'  {"Year":<8} {"BASELINE PnL":>15} {"HMM PnL":>15} {"Delta":>10}')
for y in sorted(yr_base):
    pb = yr_base[y]['pnl']; ph = yr_hmm.get(y, {}).get('pnl', 0)
    print(f'  {y:<8} {pb:>14.1f} {ph:>14.1f} {ph-pb:>+9.1f}')

# Key periods
print(f'\n  Key periods:')
for label, start, end in [
    ('2021 Bull peak (Apr-May)', '2021-04', '2021-05'),
    ('2021 ATH+corr (Oct-Dec)', '2021-10', '2021-12'),
    ('2022 Bear (May-Jul)', '2022-05', '2022-07'),
    ('2022 FTX (Nov-Dec)', '2022-11', '2022-12'),
    ('2023 Recovery (Jan-Mar)', '2023-01', '2023-03'),
    ('2024 Pre-election (Oct-Dec)', '2024-10', '2024-12'),
]:
    pb = sum(m_base[m]['pnl'] for m in m_base if start <= m <= end)
    ph = sum(m_hmm[m]['pnl'] for m in m_hmm if start <= m <= end)
    saved = ph - pb
    tag = 'SAVED!' if saved > 20 else ('WORSE' if saved < -20 else 'SAME')
    print(f'  {label:<30} BASE={pb:>8.1f}  HMM={ph:>8.1f}  Delta={saved:>+7.1f}  [{tag}]')
