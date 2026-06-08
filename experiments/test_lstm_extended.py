"""Test LSTM=ON on extended backtest vs BASELINE (LSTM=OFF)."""
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
from core.models import load_lstm

def run_extended(lstm_on):
    btu.SMART_ENTRY_MODE = 'disabled'
    btu.LSTM_CONFIRMATION_ENABLED = lstm_on
    btu.LSTM_FLAT_REVIEW_ENABLED = lstm_on
    btu.LSTM_DIRECTIONAL_REVIEW_THRESHOLD = 0.35

    guardian = joblib.load(MODEL_DIR / 'guardian_best.pkl')
    g_scaler = joblib.load(MODEL_DIR / 'guardian_scaler.pkl')
    g_feats = json.load(open(MODEL_DIR / 'guardian_feature_cols.json'))
    g_static = [c for c in g_feats if c not in set(GUARDIAN_DYNAMIC_FEATURES)]
    lstm_feat_cols = json.load(open(MODEL_DIR / 'feature_cols_lstm_temporal.json'))
    lgbm_feats = json.load(open(MODEL_DIR / 'feature_cols_v2.json'))

    if lstm_on:
        lstm_model = load_lstm(MODEL_DIR / 'lstm_best.pt')
        lstm_scaler = joblib.load(MODEL_DIR / 'lstm_scaler.pkl')
    else:
        lstm_model = None; lstm_scaler = None

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
            X_te = np.zeros((n_te, len(lstm_feat_cols)), dtype=np.float64)
            for i, col in enumerate(lstm_feat_cols):
                if col in df_te.columns: X_te[:, i] = df_te[col].ffill().fillna(0).values
            yp, cf = hierarchical_predict(None, fold_model, lstm_model, lstm_scaler,
                                           X_te, lstm_feat_cols, [], df_te,
                                           trend_alignment_enabled=True)
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

    months = {}
    for t in all_trades:
        m = pd.Timestamp(t['timestamp']).strftime('%Y-%m')
        months[m] = months.get(m, {'pnl': 0, 'trades': 0, 'wins': 0})
        months[m]['pnl'] += t.get('net_pnl', 0); months[m]['trades'] += 1
        if t.get('net_pnl', 0) > 0: months[m]['wins'] += 1

    n = sum(d['trades'] for d in months.values())
    wins = sum(d['wins'] for d in months.values())
    wr = wins/n*100 if n > 0 else 0; pnl = sum(d['pnl'] for d in months.values())
    neg = sum(1 for d in months.values() if d['pnl'] < 0)
    mpnl = [d['pnl'] for d in months.values()]

    yearly = {}
    for m, d in months.items():
        y = m[:4]; yearly[y] = yearly.get(y, {'pnl':0})
        yearly[y]['pnl'] += d['pnl']

    label = f'LSTM_ON' if lstm_on else 'LSTM_OFF'
    print(f'{label}: Trades={n} | WR={wr:.1f}% | PnL=${pnl:.0f} | Neg={neg}/{len(months)} | '
          f'Mean=${np.mean(mpnl):.0f} Std=${np.std(mpnl):.0f}')
    for y in sorted(yearly):
        p = yearly[y]['pnl']
        bar = '+'*max(0,int(p/20)) if p>0 else '-'*max(0,int(-p/20))
        print(f'  {y}: ${p:>7.0f} {bar}')
    return pnl

print('LSTM ON vs OFF — Extended Backtest (63 months)')
print()
pnl_off = run_extended(False)
print()
pnl_on = run_extended(True)
print()
print(f'Delta: {pnl_on-pnl_off:+.0f} ({(pnl_on-pnl_off)/abs(pnl_off)*100:+.1f}%)')
