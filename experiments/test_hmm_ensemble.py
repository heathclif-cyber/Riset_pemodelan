"""HMM as ensemble member: soft-vote LGBM + HMM signal."""
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

btu.SMART_ENTRY_MODE = 'disabled'; btu.LSTM_CONFIRMATION_ENABLED = False; btu.LSTM_FLAT_REVIEW_ENABLED = False
lgbm = joblib.load(MODEL_DIR / 'lgbm_baseline.pkl')
lstm_feat_cols = json.load(open(MODEL_DIR / 'feature_cols_lstm_temporal.json'))
guardian = joblib.load(MODEL_DIR / 'guardian_best.pkl'); g_scaler = joblib.load(MODEL_DIR / 'guardian_scaler.pkl')
g_feats = json.load(open(MODEL_DIR / 'guardian_feature_cols.json'))
g_static = [c for c in g_feats if c not in set(GUARDIAN_DYNAMIC_FEATURES)]
NON_FEATURE_COLS = {'label', 'h4_swing_high', 'h4_swing_low'}

all_data = {}
for coin in TRAINING_COINS[:10]:
    path = HOLDOUT_DIR / 'labeled' / f'{coin}_features_v3.parquet'
    pp = HOLDOUT_DIR / 'labeled' / f'{coin}_hmm_probs.parquet'
    rp = HOLDOUT_DIR / 'labeled' / f'{coin}_regime_h1.parquet'
    if not path.exists() or not pp.exists(): continue
    df = pd.read_parquet(path).sort_index()
    probs = pd.read_parquet(pp).sort_index()
    for i in range(4): df[f'hmm_prob_{i}'] = probs[f'hmm_prob_{i}']
    if rp.exists():
        reg = pd.read_parquet(rp)
        if 'hmm_regime_enc' in df.columns: df = df.drop(columns=['hmm_regime_enc'])
        df = df.join(reg[['hmm_regime_enc']], how='left')
        df['hmm_regime_enc'] = df['hmm_regime_enc'].fillna(1).astype('int32')
    mask = df['label'].astype(str).isin(LABEL_MAP); df = df[mask].copy()
    if len(df) >= 50: all_data[coin] = df

def run(label, hmm_weight, hmm_conf_thr, hmm_standalone_thr):
    all_trades = []
    for coin, df in all_data.items():
        n = len(df); X = np.zeros((n, len(lstm_feat_cols)))
        for i, col in enumerate(lstm_feat_cols):
            if col in df.columns: X[:, i] = df[col].ffill().fillna(0).values
        yp_lgbm, cf_lgbm = hierarchical_predict(None, lgbm, None, None, X, lstm_feat_cols, [], df,
                                                  trend_alignment_enabled=True)
        yp = np.ones(n, dtype=np.int64); cf = np.full(n, 0.5)
        for i in range(n):
            l_dir = yp_lgbm[i]; l_conf = cf_lgbm[i]
            p_up = df['hmm_prob_3'].iloc[i]; p_dn = df['hmm_prob_0'].iloc[i]
            hmm_max = max(p_up, p_dn)

            # HMM signal
            hmm_dir = 1; hmm_conf = 0.0
            if p_up > hmm_conf_thr and p_up > p_dn:
                hmm_dir = 2; hmm_conf = p_up
            elif p_dn > hmm_conf_thr and p_dn > p_up:
                hmm_dir = 0; hmm_conf = p_dn

            if hmm_dir == 1:
                # HMM no signal -> pure LGBM
                if l_dir != 1: yp[i] = l_dir; cf[i] = l_conf
            elif l_dir == 1:
                # LGBM FLAT, HMM has signal -> standalone entry if confident
                if hmm_conf >= hmm_standalone_thr:
                    yp[i] = hmm_dir; cf[i] = hmm_conf
            elif l_dir == hmm_dir:
                # Both agree -> boost
                yp[i] = l_dir; cf[i] = l_conf + hmm_weight * hmm_conf
            else:
                # Disagree -> HMM reduces LGBM confidence softly
                penalty = hmm_weight * hmm_conf
                combined = l_conf - penalty
                if combined >= 0.59:
                    yp[i] = l_dir; cf[i] = combined

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
    print(f'{label:<50} {n:>5} {wr:>4.1f}% {pnl:>8.1f}')

print('HMM ENSEMBLE: LGBM + HMM soft-vote (10 coins, holdout)')
print(f'{"Config":<50} {"Trades":>5} {"WR%":>4} {"PnL":>8}')
print('-' * 70)
run('BASELINE (LGBM only)', 0, 0, 0)

for w in [0.10, 0.15, 0.20, 0.25]:
    for thr in [0.30, 0.35]:
        for sa in [0.45]:
            run(f'HMM_ENS w={w} thr={thr} standalone={sa}', w, thr, sa)
