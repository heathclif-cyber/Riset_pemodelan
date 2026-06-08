"""Dual-Model Ensemble: Swing LGBM + Momentum LGBM + HMM Selector."""
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

# Load both models
swing_lgbm = joblib.load(MODEL_DIR / 'lgbm_baseline.pkl')
momentum_lgbm = joblib.load(MODEL_DIR / 'runs/momentum_v1/lgbm.pkl')
swing_feats = json.load(open(MODEL_DIR / 'feature_cols_v2.json'))
momentum_feats = json.load(open(MODEL_DIR / 'runs/momentum_v1/feature_cols.json'))

guardian = joblib.load(MODEL_DIR / 'guardian_best.pkl')
g_scaler = joblib.load(MODEL_DIR / 'guardian_scaler.pkl')
g_feats = json.load(open(MODEL_DIR / 'guardian_feature_cols.json'))
g_static = [c for c in g_feats if c not in set(GUARDIAN_DYNAMIC_FEATURES)]
lstm_feat_cols = json.load(open(MODEL_DIR / 'feature_cols_lstm_temporal.json'))
NON_FEATURE_COLS = {'label', 'h4_swing_high', 'h4_swing_low'}

btu.SMART_ENTRY_MODE = 'disabled'; btu.LSTM_CONFIRMATION_ENABLED = False
btu.LSTM_FLAT_REVIEW_ENABLED = False

all_data = {}
for coin in TRAINING_COINS[:10]:
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
    for c in momentum_feats:
        if c not in df.columns: df[c] = 0.0
    mask = df['label'].astype(str).isin(LABEL_MAP); df = df[mask].copy()
    if len(df) >= 50: all_data[coin] = df

def run_backtest(mode):
    """mode: 'swing_only', 'dual_model'"""
    all_trades = []
    for coin, df in all_data.items():
        n = len(df)
        # Build X for both models
        X_swing = np.zeros((n, len(swing_feats)))
        for i, col in enumerate(swing_feats):
            if col in df.columns: X_swing[:, i] = df[col].ffill().fillna(0).values
        X_mom = np.zeros((n, len(momentum_feats)))
        for i, col in enumerate(momentum_feats):
            if col in df.columns: X_mom[:, i] = df[col].ffill().fillna(0).values

        # Swing model prediction
        yp_swing, cf_swing = hierarchical_predict(
            None, swing_lgbm, None, None, X_swing, swing_feats, [], df,
            trend_alignment_enabled=True)

        yp = np.ones(n, dtype=np.int64); cf = np.full(n, 0.5)

        if mode == 'swing_only':
            yp, cf = yp_swing, cf_swing

        elif mode == 'dual_model':
            # Momentum model prediction (separate prediction)
            gbm_mom = momentum_lgbm.feature_name_
            X_pred_mom = np.zeros((n, len(gbm_mom)), dtype=np.float64)
            for i, col in enumerate(gbm_mom):
                if col in df.columns: X_pred_mom[:, i] = df[col].ffill().fillna(0).values
            proba_mom = momentum_lgbm.predict_proba(X_pred_mom)

            for i in range(n):
                regime = int(df['hmm_regime_enc'].iloc[i])
                h4_t = df['h4_trend'].iloc[i] if 'h4_trend' in df.columns else 0

                if regime in [1, 2]:  # RANGING → swing model
                    yp[i] = yp_swing[i]; cf[i] = cf_swing[i]
                else:  # TRENDING (0,3) → momentum model
                    # Momentum says: BULLISH(2) or BEARISH(0)
                    mom_dir = int(np.argmax(proba_mom[i]))
                    mom_conf = float(np.max(proba_mom[i]))
                    mom_bull = float(proba_mom[i, 2])
                    mom_bear = float(proba_mom[i, 0])

                    if regime == 3 and h4_t > 0:  # TRENDING_UP: only BULLISH
                        if mom_bull > 0.45:
                            yp[i] = 2; cf[i] = mom_bull  # LONG
                    elif regime == 0 and h4_t < 0:  # TRENDING_DOWN: only BEARISH
                        if mom_bear > 0.45:
                            yp[i] = 0; cf[i] = mom_bear  # SHORT
                    else:
                        # TRENDING but h4_t not aligned → fallback to swing
                        yp[i] = yp_swing[i]; cf[i] = cf_swing[i]

        elif mode == 'ensemble':
            # Ensemble: soft-vote between swing and momentum
            gbm_mom = momentum_lgbm.feature_name_
            X_pred_mom = np.zeros((n, len(gbm_mom)), dtype=np.float64)
            for i, col in enumerate(gbm_mom):
                if col in df.columns: X_pred_mom[:, i] = df[col].ffill().fillna(0).values
            proba_mom = momentum_lgbm.predict_proba(X_pred_mom)

            for i in range(n):
                regime = int(df['hmm_regime_enc'].iloc[i])
                # Weight: ranging → more swing, trending → more momentum
                w_swing = 0.80 if regime in [1,2] else 0.30
                w_mom = 0.20 if regime in [1,2] else 0.70

                # Combine directions: swing LONG=2, momentum BULLISH=2
                sw_dir = yp_swing[i]; sw_conf = cf_swing[i]
                mom_dir = int(np.argmax(proba_mom[i]))
                mom_conf = float(np.max(proba_mom[i]))

                if sw_dir == 1 and mom_dir == 1:
                    continue  # both FLAT
                elif sw_dir == 1:
                    if mom_conf >= 0.45: yp[i] = mom_dir; cf[i] = mom_conf
                elif mom_dir == 1:
                    yp[i] = sw_dir; cf[i] = sw_conf
                elif sw_dir == mom_dir:
                    yp[i] = sw_dir; cf[i] = w_swing*sw_conf + w_mom*mom_conf
                else:
                    # Disagree: weighted confidence
                    combined = w_swing*sw_conf - w_mom*mom_conf
                    if combined >= 0.59:
                        yp[i] = sw_dir; cf[i] = combined

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
        for t in r.get('trades', []): t['coin'] = coin
        all_trades.extend(r.get('trades', []))

    n = len(all_trades); wins = [t for t in all_trades if t.get('net_pnl', 0) > 0]
    wr = len(wins)/n*100 if n else 0; pnl = sum(t.get('net_pnl', 0) for t in all_trades)
    gw = sum(t.get('net_pnl', 0) for t in wins)
    gl = abs(sum(t.get('net_pnl', 0) for t in all_trades if t.get('net_pnl', 0) <= 0))
    pf = gw/gl if gl > 0 else float('inf')
    lt = [t for t in all_trades if t.get('direction') == 'LONG']
    st = [t for t in all_trades if t.get('direction') == 'SHORT']
    lwr = len([t for t in lt if t.get('net_pnl', 0) > 0])/len(lt)*100 if lt else 0
    swr = len([t for t in st if t.get('net_pnl', 0) > 0])/len(st)*100 if st else 0
    print(f'{mode:<20} {n:>6} {wr:>5.1f}% {lwr:>5.1f}% {swr:>5.1f}% {pnl:>8.1f} {pf:>6.2f}')

print('DUAL-MODEL ENSEMBLE (10 coins, holdout)')
print(f'{"Mode":<20} {"Trades":>6} {"WR%":>5} {"L_WR%":>5} {"S_WR%":>5} {"PnL":>8} {"PF":>6}')
print('-' * 65)
run_backtest('swing_only')
run_backtest('dual_model')
run_backtest('ensemble')
