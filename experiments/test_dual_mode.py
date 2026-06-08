"""Dual-Mode System: Swing (ranging) + Momentum (trending) based on HMM regime."""
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

def run_extended(mode):
    """mode: 'baseline', 'hmm_block', 'dual_mode'"""
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

            if mode == 'dual_mode':
                # DUAL MODE: per-bar regime-aware trend alignment
                # RANGING: counter-trend (+0.05 boost, -0.10 penalty) -- swing
                # TRENDING: with-trend (+0.10 boost, -0.05 penalty) -- momentum
                for i in range(n_te):
                    regime = int(df_te['hmm_regime_enc'].iloc[i])
                    if regime in [0, 3]:  # TRENDING
                        # Flip: with-trend gets boost, counter-trend gets penalty
                        pass  # handled below in per-trade logic

                # Use per-bar override in hierarchical_predict args
                yp = np.ones(n_te, dtype=np.int64)
                cf = np.full(n_te, 1.0/3)
                for i in range(n_te):
                    regime = int(df_te['hmm_regime_enc'].iloc[i])
                    h4_t = df_te['h4_trend'].iloc[i] if 'h4_trend' in df_te.columns else 0

                    if regime in [1, 2]:  # RANGING: counter-trend = swing mode
                        trend_on = True
                        # Default h4_trend logic: counter-trend +0.05, with-trend -0.10
                    else:  # TRENDING (0,3): with-trend = momentum mode
                        trend_on = True
                        # In trending: counter-trend gets penalty, with-trend gets boost
                        # We implement this by FLIPPING the trend alignment
                        # Normal: counter+LONG = +0.05, with+LONG = -0.10
                        # Flipped: counter+LONG = -0.10, with+LONG = +0.10
                        # Achieved by negating the h4_trend value temporarily

                # Simpler approach: run hierarchical_predict per bar with different params
                # Too complex. Let's use a simpler dual-mode:
                # In TRENDING: ONLY allow with-trend entries
                # In RANGING: normal swing logic

                # Run normal prediction
                yp_temp, cf_temp = hierarchical_predict(
                    None, fold_model, None, None, X_te, feat_cols, [], df_te,
                    trend_alignment_enabled=True)

                for i in range(n_te):
                    regime = int(df_te['hmm_regime_enc'].iloc[i])
                    h4_t = df_te['h4_trend'].iloc[i] if 'h4_trend' in df_te.columns else 0

                    if regime == 3 and h4_t > 0:  # TRENDING_UP: only LONG
                        if yp_temp[i] == 0: yp_temp[i] = 1  # block SHORT
                    elif regime == 0 and h4_t < 0:  # TRENDING_DOWN: only SHORT
                        if yp_temp[i] == 2: yp_temp[i] = 1  # block LONG
                    # RANGING: keep original

                yp, cf = yp_temp, cf_temp

            elif mode == 'hmm_block':
                yp, cf = hierarchical_predict(
                    None, fold_model, None, None, X_te, feat_cols, [], df_te,
                    trend_alignment_enabled=True)
                for i in range(n_te):
                    regime = int(df_te['hmm_regime_enc'].iloc[i])
                    h4_t = df_te['h4_trend'].iloc[i] if 'h4_trend' in df_te.columns else 0
                    if regime == 3 and yp[i] == 0 and h4_t > 0: yp[i] = 1
                    if regime == 0 and yp[i] == 2 and h4_t < 0: yp[i] = 1

            else:  # baseline
                yp, cf = hierarchical_predict(
                    None, fold_model, None, None, X_te, feat_cols, [], df_te,
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

    return all_trades

# Run all three
for mode, label in [('baseline', 'BASELINE'), ('hmm_block', 'HMM BLOCK'), ('dual_mode', 'DUAL-MODE')]:
    print(f'Running {label}...')
    trades = run_extended(mode)

    months = {}
    for t in trades:
        m = pd.Timestamp(t['timestamp']).strftime('%Y-%m')
        months[m] = months.get(m, {'pnl': 0, 'trades': 0, 'wins': 0})
        months[m]['pnl'] += t.get('net_pnl', 0); months[m]['trades'] += 1
        if t.get('net_pnl', 0) > 0: months[m]['wins'] += 1

    n = sum(d['trades'] for d in months.values())
    wins = sum(d['wins'] for d in months.values())
    wr = wins/n*100 if n > 0 else 0
    pnl = sum(d['pnl'] for d in months.values())
    neg = sum(1 for d in months.values() if d['pnl'] < 0)
    mpnl = [d['pnl'] for d in months.values()]

    # Yearly
    yearly = {}
    for m, d in months.items():
        y = m[:4]; yearly[y] = yearly.get(y, {'pnl':0,'trades':0})
        yearly[y]['pnl'] += d['pnl']; yearly[y]['trades'] += d['trades']

    print(f'  {label}: {n} trades | WR={wr:.1f}% | PnL=${pnl:.1f} | NegMonths={neg}/{len(months)}')
    print(f'    Yearly: { {y: round(d["pnl"]) for y,d in yearly.items()} }')
    print()
