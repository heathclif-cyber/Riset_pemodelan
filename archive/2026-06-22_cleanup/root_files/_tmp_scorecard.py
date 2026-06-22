import pandas as pd, numpy as np, sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from config import (LABEL_DIR, TRAIN_CUTOFF_DATE,
                    MODAL_PER_TRADE, LEVERAGE_SIM, FEE_PER_SIDE, SLIPPAGE_PER_SIDE,
                    MAX_HOLDING_BARS, TP_SL_FALLBACK_TP, TP_SL_FALLBACK_SL,
                    SWING_LABEL_MIN_RR, SWING_LABEL_MIN_TP, SWING_LABEL_MAX_SL)
from core.evaluator import simulate_trades_swing
from core.utils import ensure_utc_index

oof = pd.read_parquet('models/runs/ic32_regime_v2/oof_predictions.parquet')
oof = oof[oof.has_oof].copy()
oof.index = pd.to_datetime(oof.index, utc=True)

# Koin yang simulasinya crash karena harga micro-price era 2020-2021
EXCLUDE_SIM = {'1000SHIBUSDT', '1000PEPEUSDT'}

def run_scorecard(thr_long, thr_short):
    agg = dict(trades=0, wins=0, losses=0, time_exits=0,
               pnl=0.0, sum_win=0.0, sum_loss=0.0,
               long_t=0, long_w=0, short_t=0, short_w=0,
               coin_results=[])

    for sym in sorted(oof['coin'].unique()):
        fp = LABEL_DIR / f'{sym}_features_v3.parquet'
        if not fp.exists():
            continue

        df = pd.read_parquet(fp)
        df = ensure_utc_index(df).sort_index()
        df = df[df.index < TRAIN_CUTOFF_DATE]
        sym_oof = oof[oof['coin'] == sym]
        df = df.join(sym_oof[['p0', 'p2']], how='inner').dropna(subset=['p0', 'p2'])
        if len(df) < 30:
            continue

        n = len(df)
        y_pred = np.full(n, 1, np.int32)
        p2 = df['p2'].values; p0 = df['p0'].values
        y_pred[p2 >= thr_long] = 2
        y_pred[(p0 >= thr_short) & (y_pred != 2)] = 0

        if (y_pred != 1).sum() == 0:
            continue

        h4_sh = df['h4_swing_high'].values if 'h4_swing_high' in df.columns else np.full(n, np.nan)
        h4_sl = df['h4_swing_low'].values if 'h4_swing_low' in df.columns else np.full(n, np.nan)

        r = simulate_trades_swing(
            y_pred=y_pred, close=df['close'].values, high=df['high'].values,
            low=df['low'].values, atr=df['atr_14_h1'].values,
            h4_swing_highs=h4_sh, h4_swing_lows=h4_sl,
            modal=MODAL_PER_TRADE, leverage=LEVERAGE_SIM[0],
            fee_per_side=FEE_PER_SIDE, slippage=SLIPPAGE_PER_SIDE,
            max_hold=MAX_HOLDING_BARS, min_rr=SWING_LABEL_MIN_RR,
            min_tp_atr=SWING_LABEL_MIN_TP, max_sl_atr=SWING_LABEL_MAX_SL,
            tp_fallback_atr=TP_SL_FALLBACK_TP, sl_fallback_atr=TP_SL_FALLBACK_SL,
            guardian_enabled=False,
        )

        t = r.get('total_trades', 0)
        if t == 0:
            continue

        w = r.get('wins', 0)
        lo = r.get('losses', 0)
        te = r.get('time_exits', 0)
        pnl = r.get('net_pnl_total', 0.0)
        pf  = r.get('profit_factor', 0.0)
        wbc = r.get('win_by_class', {})  # {'LONG': wr_long, 'SHORT': wr_short}
        wr_long  = wbc.get('LONG', 0.0)
        wr_short = wbc.get('SHORT', 0.0)

        ppt_list = r.get('pnl_per_trade', [])
        if isinstance(ppt_list, (list, np.ndarray)):
            arr = np.array([float(x) for x in ppt_list])
            sw  = float(arr[arr > 0].sum()) if (arr > 0).any() else 0.0
            sl_ = float(abs(arr[arr < 0].sum())) if (arr < 0).any() else 0.0
        else:
            sw = float(r.get('avg_win', 0)) * w
            sl_ = float(abs(r.get('avg_loss', 0))) * lo

        # Hitung long/short trade count dari trade_log
        tlog = r.get('trade_log', [])
        long_t  = sum(1 for x in tlog if x.get('direction') == 'LONG')
        short_t = sum(1 for x in tlog if x.get('direction') == 'SHORT')
        long_w  = round(wr_long  * long_t)
        short_w = round(wr_short * short_t)

        tag = " [SIM-SKIP]" if sym in EXCLUDE_SIM else ""
        agg['coin_results'].append({
            'coin': sym, 'trades': t, 'wr': w/t, 'pnl': pnl, 'pf': pf,
            'wr_long': wr_long, 'wr_short': wr_short,
            'long_t': long_t, 'short_t': short_t, 'te': te,
            'excluded': sym in EXCLUDE_SIM,
        })

        if sym in EXCLUDE_SIM:
            continue  # exclude dari aggregate

        agg['trades']   += t
        agg['wins']     += w
        agg['losses']   += lo
        agg['time_exits'] += te
        agg['pnl']      += pnl
        agg['sum_win']  += sw
        agg['sum_loss'] += sl_
        agg['long_t']   += long_t
        agg['long_w']   += long_w
        agg['short_t']  += short_t
        agg['short_w']  += short_w

    return agg

with open('models/runs/ic32_regime_v2/cv_results.json') as f:
    cv = json.load(f)

print("=" * 65)
print("  SCORECARD — ic32_regime_v2 LGBM (class weights, lr=0.05)")
print("  Swing sim proper | Tanpa LSTM, Guardian, HMM")
print("  Exclude SHIB/PEPE (micro-price sim artifact)")
print("=" * 65)

print("\n--- CV PER FOLD ---")
for fold in cv['folds']:
    converged = "OK" if fold['best_iter'] < 1490 else "cap"
    print(f"  Fold {fold['fold']}: F1={fold['f1_macro']:.4f}  S={fold['f1_SHORT']:.4f}  "
          f"F={fold['f1_FLAT']:.4f}  L={fold['f1_LONG']:.4f}  "
          f"iter={fold['best_iter']} ({converged})")
print(f"  Mean: {cv['mean_f1_macro']:.4f} ± {cv['std_f1_macro']:.4f}  "
      f"avg_iter={cv['avg_iterations']}")

print("\n--- OOF SCORECARD PER THRESHOLD (19 koin, tanpa SHIB/PEPE) ---")
for thr_long, thr_short in [(0.60, 0.60), (0.65, 0.65), (0.70, 0.70), (0.75, 0.70)]:
    a = run_scorecard(thr_long, thr_short)
    t = a['trades']; w = a['wins']
    lt = a['long_t']; lw = a['long_w']
    st = a['short_t']; sw2 = a['short_w']
    te = a['time_exits']
    pnl = a['pnl']
    sw = a['sum_win']; sl_ = a['sum_loss']
    wr    = w / t if t else 0
    wr_l  = lw / lt if lt else 0
    wr_s  = sw2 / st if st else 0
    pf    = sw / sl_ if sl_ > 0 else float('inf')
    ppt   = pnl / t if t else 0

    print(f"\n  thr L={thr_long}  S={thr_short}")
    print(f"    Trades     : {t:,}  (L={lt:,}  S={st:,}  TimeExit={te:,})")
    print(f"    WR Total   : {wr*100:.1f}%")
    print(f"    WR LONG    : {wr_l*100:.1f}%  ({lw:,}/{lt:,})")
    print(f"    WR SHORT   : {wr_s*100:.1f}%  ({sw2:,}/{st:,})")
    print(f"    PnL Total  : ${pnl:,.2f}")
    print(f"    PnL/Trade  : ${ppt:.4f}")
    print(f"    Profit Fac : {pf:.3f}")
    print(f"    Gross Win  : ${sw:,.2f}  |  Gross Loss: ${sl_:,.2f}")

print("\n--- PER-KOIN (thr 0.65/0.60) ---")
a = run_scorecard(0.75, 0.70)
sorted_coins = sorted(a['coin_results'], key=lambda x: x['pnl'], reverse=True)
print(f"  {'Coin':<18} {'T':>5} {'WR%':>6} {'WR_L%':>6} {'WR_S%':>6} {'PnL':>8} {'PF':>5}")
for c in sorted_coins:
    tag = " *excl*" if c['excluded'] else ""
    print(f"  {c['coin']:<18} {c['trades']:>5,} {c['wr']*100:>6.1f} "
          f"{c['wr_long']*100:>6.1f} {c['wr_short']*100:>6.1f} "
          f"{c['pnl']:>8.0f} {c['pf']:>5.2f}{tag}")
