"""Complete scorecard: BASELINE vs FLIP trend alignment."""
import sys, json, joblib, numpy as np, pandas as pd
from pathlib import Path
ROOT = Path(__file__).parent.parent; sys.path.insert(0, str(ROOT))

from config import MODEL_DIR, HOLDOUT_DIR, TRAINING_COINS, LABEL_MAP
from config import *
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

all_data = {}
for coin in TRAINING_COINS:
    path = HOLDOUT_DIR / 'labeled' / f'{coin}_features_v3.parquet'
    rp = HOLDOUT_DIR / 'labeled' / f'{coin}_regime_h1.parquet'
    if not path.exists(): continue
    df = pd.read_parquet(path).sort_index()
    if rp.exists():
        reg = pd.read_parquet(rp)
        if 'hmm_regime_enc' in df.columns: df = df.drop(columns=['hmm_regime_enc'])
        df = df.join(reg[['hmm_regime_enc']], how='left')
        df['hmm_regime_enc'] = df['hmm_regime_enc'].fillna(1).astype('int32')
    mask = df['label'].astype(str).isin(LABEL_MAP); df = df[mask].copy()
    if len(df) >= 50: all_data[coin] = df

def run_full(label, flip):
    all_trades = []
    for coin, df in all_data.items():
        n = len(df); X = np.zeros((n, len(lstm_feat_cols)))
        for i, col in enumerate(lstm_feat_cols):
            if col in df.columns: X[:, i] = df[col].ffill().fillna(0).values

        if flip:
            yp_base, cf_base = hierarchical_predict(None, lgbm, None, None, X, lstm_feat_cols, [], df,
                                                      trend_alignment_enabled=False)
            yp = np.ones(n, dtype=np.int64); cf = np.full(n, 0.5)
            for i in range(n):
                regime = int(df['hmm_regime_enc'].iloc[i])
                h4_t = df['h4_trend'].iloc[i] if 'h4_trend' in df.columns else 0
                adj_conf = cf_base[i]
                is_dir = yp_base[i] != 1
                if regime in [1, 2]:  # RANGING: counter-trend
                    if is_dir:
                        is_with = (yp_base[i]==2 and h4_t>0) or (yp_base[i]==0 and h4_t<0)
                        is_counter = (yp_base[i]==2 and h4_t<0) or (yp_base[i]==0 and h4_t>0)
                        if is_with: adj_conf -= 0.10
                        elif is_counter: adj_conf += 0.05
                else:  # TRENDING: with-trend (FLIPPED)
                    if is_dir:
                        is_with = (yp_base[i]==2 and h4_t>0) or (yp_base[i]==0 and h4_t<0)
                        is_counter = (yp_base[i]==2 and h4_t<0) or (yp_base[i]==0 and h4_t>0)
                        if is_with: adj_conf += 0.10
                        elif is_counter: adj_conf -= 0.05
                adj_conf = float(np.clip(adj_conf, 0, 1))
                if is_dir and adj_conf >= 0.59: yp[i] = yp_base[i]; cf[i] = adj_conf
            below = (yp != 1) & (cf < 0.59); yp[below] = 1
        else:
            yp, cf = hierarchical_predict(None, lgbm, None, None, X, lstm_feat_cols, [], df,
                                           trend_alignment_enabled=True)
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
            t['coin'] = coin; t['timestamp'] = df.index[t['bar_in']]
            t['regime'] = int(df['hmm_regime_enc'].iloc[t['bar_in']])
        all_trades.extend(r.get('trades', []))

    n = len(all_trades)
    wins = [t for t in all_trades if t.get('net_pnl', 0) > 0]
    losses = [t for t in all_trades if t.get('net_pnl', 0) <= 0]
    wr = len(wins)/n*100 if n else 0
    pnl = sum(t.get('net_pnl', 0) for t in all_trades)
    gw = sum(t.get('net_pnl', 0) for t in wins)
    gl = abs(sum(t.get('net_pnl', 0) for t in losses))
    pf = gw/gl if gl > 0 else float('inf')

    lt = [t for t in all_trades if t.get('direction') == 'LONG']
    st = [t for t in all_trades if t.get('direction') == 'SHORT']
    lwr = len([t for t in lt if t.get('net_pnl', 0) > 0])/len(lt)*100 if lt else 0
    swr = len([t for t in st if t.get('net_pnl', 0) > 0])/len(st)*100 if st else 0

    sl_hits = len([t for t in all_trades if str(t.get('outcome', '')).lower() == 'loss'])
    gx = len([t for t in all_trades if 'guardian' in str(t.get('outcome', '')).lower()])
    gx_wr = len([t for t in all_trades if 'guardian' in str(t.get('outcome', '')).lower()
                 and t.get('net_pnl', 0) > 0])/gx*100 if gx else 0
    holds = [t.get('bar_out', 0) - t.get('bar_in', 0) for t in all_trades if 'bar_in' in t]
    avg_hold = np.mean(holds) if holds else 0

    max_cl = cur = 0
    for t in sorted(all_trades, key=lambda x: str(x.get('timestamp', ''))):
        if t.get('net_pnl', 0) <= 0: cur += 1; max_cl = max(max_cl, cur)
        else: cur = 0

    months = {}
    for t in all_trades:
        m = pd.Timestamp(t['timestamp']).strftime('%Y-%m')
        months[m] = months.get(m, {'pnl': 0, 'trades': 0, 'wins': 0})
        months[m]['pnl'] += t.get('net_pnl', 0)
        months[m]['trades'] += 1
        if t.get('net_pnl', 0) > 0: months[m]['wins'] += 1
    neg_m = sum(1 for d in months.values() if d['pnl'] < 0)
    mpnl = [d['pnl'] for d in months.values()]

    reg_names = {0: 'TRENDING_DN', 1: 'RANGING_LOW', 2: 'RANGING_HIGH', 3: 'TRENDING_UP'}
    reg_pnl = {}; reg_wr = {}
    for r in [0, 1, 2, 3]:
        rt = [t for t in all_trades if t.get('regime') == r]
        if rt:
            reg_pnl[r] = sum(t.get('net_pnl', 0) for t in rt)
            reg_wr[r] = len([t for t in rt if t.get('net_pnl', 0) > 0])/len(rt)*100

    print(f'\n=== {label} ===')
    print(f'Trades: {n} ({n/5:.0f}/bulan) | WR: {wr:.1f}% | PnL: ${pnl:.1f} (${pnl/5:.0f}/mo)')
    print(f'LONG WR: {lwr:.1f}% ({len(lt)}) | SHORT WR: {swr:.1f}% ({len(st)}) | LONG%: {len(lt)/n*100:.0f}%')
    print(f'PF: {pf:.2f} | Max CL: {max_cl} | Avg Hold: {avg_hold:.1f} bars')
    print(f'SL Hit: {sl_hits} ({sl_hits/n*100:.1f}%) | Guardian: {gx} ({gx_wr:.1f}% WR)')
    for r in [0,1,2,3]:
        print(f'  {reg_names[r]}: PnL=${reg_pnl.get(r,0):.0f} WR={reg_wr.get(r,0):.0f}%')
    print(f'Monthly: mean=${np.mean(mpnl):.0f} std=${np.std(mpnl):.0f} neg={neg_m}/{len(months)}')
    for m in sorted(months.keys()):
        d = months[m]; mwr = d['wins']/d['trades']*100
        print(f'  {m}: {d["trades"]:>4} trades WR={mwr:>5.1f}% PnL=${d["pnl"]:>7.1f}')

run_full('BASELINE (counter-trend always)', False)
run_full('FLIP (regime-aware alignment)', True)
