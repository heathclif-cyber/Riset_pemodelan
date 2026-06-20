"""
Analyze actual price outcome untuk setiap entry signal 19 Juni 2026 WITA.
Menggunakan TP/SL logic yang sama dengan evaluator (swing-based).
"""
import json, sys, warnings, numpy as np, pandas as pd
from pathlib import Path
warnings.filterwarnings('ignore')
sys.path.insert(0, str(Path(__file__).parent.parent))
from core.utils import ensure_utc_index
from config import (
    HOLDOUT_DIR, MODEL_DIR,
    SWING_LABEL_MIN_RR, SWING_LABEL_MIN_TP, SWING_LABEL_MAX_SL,
    TP_SL_FALLBACK_TP, TP_SL_FALLBACK_SL,
    MAX_HOLDING_BARS, MODAL_PER_TRADE, LEVERAGE_SIM, FEE_PER_SIDE, SLIPPAGE_PER_SIDE,
)

with open(MODEL_DIR / 'runs' / 'ic32_regime_v1' / 'holdout_diag_jun19_wita.json') as f:
    diag = json.load(f)

entries = diag['entries']
if not entries:
    print("Tidak ada entry signal.")
    sys.exit(0)

print(f"\n{'='*85}")
print("  OUTCOME ANALYSIS — Entry Signals 19 Juni 2026 WITA")
print(f"  (TP/SL fallback ATR: tp={TP_SL_FALLBACK_TP}x, sl={TP_SL_FALLBACK_SL}x | max_hold={MAX_HOLDING_BARS})")
print(f"{'='*85}\n")

results = []

for sig in entries:
    sym     = sig['symbol']
    pred    = sig['pred']     # LONG / SHORT
    ts_utc  = pd.Timestamp(sig['ts_utc'], tz='UTC')
    conf    = sig['conf']
    close   = sig['close']
    regime  = sig['hmm_reg']

    df = pd.read_parquet(HOLDOUT_DIR / 'labeled' / f'{sym}_features_v3.parquet')
    df = ensure_utc_index(df).sort_index()

    # Temukan posisi bar entry
    idx_arr = np.searchsorted(df.index, ts_utc)
    if idx_arr >= len(df):
        continue
    entry_bar = df.index[idx_arr]

    # Ambil forward bars (MAX_HOLDING_BARS ke depan)
    fwd = df.iloc[idx_arr: idx_arr + MAX_HOLDING_BARS + 1]
    if len(fwd) < 2:
        continue

    entry_close = float(fwd['close'].iloc[0])
    atr         = float(fwd['atr_14_h1'].iloc[0]) if 'atr_14_h1' in fwd.columns else entry_close * 0.01

    # Swing high/low TP/SL
    h4_sh = fwd['h4_swing_high'].iloc[0] if 'h4_swing_high' in fwd.columns else np.nan
    h4_sl = fwd['h4_swing_low'].iloc[0]  if 'h4_swing_low'  in fwd.columns else np.nan

    # Calculate TP/SL levels
    if pred == 'LONG':
        raw_tp = h4_sh if (not np.isnan(h4_sh) and h4_sh > entry_close) else entry_close + TP_SL_FALLBACK_TP * atr
        raw_sl = h4_sl if (not np.isnan(h4_sl) and h4_sl < entry_close) else entry_close - TP_SL_FALLBACK_SL * atr
        tp_price = min(raw_tp, entry_close + 10 * atr)
        sl_price = max(raw_sl, entry_close - TP_SL_FALLBACK_SL * atr)
        rr = (tp_price - entry_close) / max(entry_close - sl_price, 1e-10)
    else:  # SHORT
        raw_tp = h4_sl if (not np.isnan(h4_sl) and h4_sl < entry_close) else entry_close - TP_SL_FALLBACK_TP * atr
        raw_sl = h4_sh if (not np.isnan(h4_sh) and h4_sh > entry_close) else entry_close + TP_SL_FALLBACK_SL * atr
        tp_price = max(raw_tp, entry_close - 10 * atr)
        sl_price = min(raw_sl, entry_close + TP_SL_FALLBACK_SL * atr)
        rr = (entry_close - tp_price) / max(sl_price - entry_close, 1e-10)

    # Simulate bar by bar
    outcome   = 'TIME_EXIT'
    exit_bar  = fwd.index[-1]
    exit_px   = float(fwd['close'].iloc[-1])
    bars_held = len(fwd) - 1

    for b in range(1, len(fwd)):
        row_high  = float(fwd['high'].iloc[b])
        row_low   = float(fwd['low'].iloc[b])
        row_close = float(fwd['close'].iloc[b])

        if pred == 'LONG':
            if row_close <= sl_price:
                outcome = 'SL_HIT'; exit_bar = fwd.index[b]; exit_px = sl_price; bars_held = b; break
            if row_close >= tp_price:
                outcome = 'TP_HIT'; exit_bar = fwd.index[b]; exit_px = tp_price; bars_held = b; break
        else:
            if row_close >= sl_price:
                outcome = 'SL_HIT'; exit_bar = fwd.index[b]; exit_px = sl_price; bars_held = b; break
            if row_close <= tp_price:
                outcome = 'TP_HIT'; exit_bar = fwd.index[b]; exit_px = tp_price; bars_held = b; break

    # PnL calculation
    lev  = float(LEVERAGE_SIM[0]) if isinstance(LEVERAGE_SIM, list) else float(LEVERAGE_SIM)
    modal = float(MODAL_PER_TRADE)
    fee  = (FEE_PER_SIDE + SLIPPAGE_PER_SIDE) * 2
    if pred == 'LONG':
        raw_pnl = (exit_px - entry_close) / entry_close * lev * modal
    else:
        raw_pnl = (entry_close - exit_px) / entry_close * lev * modal
    net_pnl = raw_pnl - fee * lev * modal

    win = net_pnl > 0

    ts_wita = ts_utc + pd.Timedelta(hours=8)
    exit_wita = exit_bar + pd.Timedelta(hours=8)

    results.append({
        'symbol':    sym,
        'pred':      pred,
        'ts_wita':   ts_wita.strftime('%H:%M'),
        'ts_utc':    sig['ts_utc'],
        'conf':      conf,
        'regime':    regime,
        'entry_px':  entry_close,
        'tp_price':  round(tp_price, 4),
        'sl_price':  round(sl_price, 4),
        'rr':        round(rr, 2),
        'atr':       round(atr, 4),
        'exit_px':   round(exit_px, 4),
        'bars_held': bars_held,
        'exit_wita': exit_wita.strftime('%Y-%m-%d %H:%M'),
        'outcome':   outcome,
        'net_pnl':   round(net_pnl, 4),
        'win':       win,
    })

# Print hasil
print(f"  {'Koin':<16} {'WITA':>5} {'Dir':>5} {'Conf':>5} {'Entry':>10}  {'TP':>10}  {'SL':>10}  {'RR':>4}  {'Bars':>4}  {'Outcome':>10}  {'PnL':>7}  {'Regime'}")
print("  " + "-" * 115)
for r in results:
    wl  = 'WIN' if r['win'] else 'LOSS'
    pnl = f"${r['net_pnl']:+.3f}"
    print(f"  {r['symbol']:<16} {r['ts_wita']:>5} {r['pred']:>5} {r['conf']:>5.3f} {r['entry_px']:>10.4f}  {r['tp_price']:>10.4f}  {r['sl_price']:>10.4f}  {r['rr']:>4.2f}  {r['bars_held']:>4}  {r['outcome']:>10}  {pnl:>7}  {r['regime']}")

n_t   = len(results)
n_win = sum(1 for r in results if r['win'])
total_pnl = sum(r['net_pnl'] for r in results)
gpnl  = sum(r['net_pnl'] for r in results if r['net_pnl'] > 0)
gloss = abs(sum(r['net_pnl'] for r in results if r['net_pnl'] <= 0))
pf    = gpnl / gloss if gloss > 0 else float('inf')

longs  = [r for r in results if r['pred'] == 'LONG']
shorts = [r for r in results if r['pred'] == 'SHORT']

print(f"\n{'='*85}")
print(f"  SCORECARD 19 Juni 2026 WITA — ic32_b_dir_combined")
print(f"{'='*85}")
print(f"  Total entries   : {n_t}")
print(f"  LONG  / SHORT   : {len(longs)} / {len(shorts)}")
print(f"  Win Rate        : {n_win}/{n_t} = {n_win/n_t*100:.1f}%")
print(f"  Net PnL         : ${total_pnl:+.3f}")
print(f"  Profit Factor   : {pf:.3f}")
print(f"  PnL/trade       : ${total_pnl/n_t:+.4f}")
_lev_disp = float(LEVERAGE_SIM[0]) if isinstance(LEVERAGE_SIM, list) else float(LEVERAGE_SIM)
print(f"\n  (Modal ${MODAL_PER_TRADE}/trade, {_lev_disp}x leverage, fee+slippage {(FEE_PER_SIDE+SLIPPAGE_PER_SIDE)*2:.4f}/side)")
print(f"{'='*85}\n")

# Simpan
with open(MODEL_DIR / 'runs' / 'ic32_regime_v1' / 'holdout_diag_jun19_outcomes.json', 'w') as f:
    json.dump({'results': results, 'aggregate': {
        'n_trades': n_t, 'n_wins': n_win, 'wr': round(n_win/n_t*100,2),
        'pnl': round(total_pnl,4), 'pf': round(pf,4),
    }}, f, indent=2)
print("  Saved -> models/runs/ic32_regime_v1/holdout_diag_jun19_outcomes.json")
