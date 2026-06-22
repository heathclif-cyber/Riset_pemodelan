import pandas as pd, numpy as np, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from config import (LABEL_DIR, TRAIN_CUTOFF_DATE, MODAL_PER_TRADE, LEVERAGE_SIM,
                    FEE_PER_SIDE, SLIPPAGE_PER_SIDE, MAX_HOLDING_BARS,
                    TP_SL_FALLBACK_TP, TP_SL_FALLBACK_SL,
                    SWING_LABEL_MIN_RR, SWING_LABEL_MIN_TP, SWING_LABEL_MAX_SL)
from core.evaluator import simulate_trades_swing
from core.utils import ensure_utc_index

oof = pd.read_parquet('models/runs/ic32_regime_v2/oof_predictions.parquet')
oof = oof[oof.has_oof].copy()
oof.index = pd.to_datetime(oof.index, utc=True)

sym = '1000PEPEUSDT'
df = pd.read_parquet(LABEL_DIR / f'{sym}_features_v3.parquet')
df = ensure_utc_index(df).sort_index()
df = df[df.index < TRAIN_CUTOFF_DATE]
sym_oof = oof[oof['coin'] == sym]
df = df.join(sym_oof[['p0', 'p2']], how='inner').dropna(subset=['p0', 'p2'])

n = len(df)
y_pred = np.full(n, 1, np.int32)
y_pred[df['p2'].values >= 0.65] = 2
y_pred[(df['p0'].values >= 0.60) & (y_pred != 2)] = 0

h4_sh = df['h4_swing_high'].values if 'h4_swing_high' in df.columns else np.full(n, np.nan)
h4_sl = df['h4_swing_low'].values if 'h4_swing_low' in df.columns else np.full(n, np.nan)

r = simulate_trades_swing(
    y_pred=y_pred, close=df['close'].values, high=df['high'].values, low=df['low'].values,
    atr=df['atr_14_h1'].values, h4_swing_highs=h4_sh, h4_swing_lows=h4_sl,
    modal=MODAL_PER_TRADE, leverage=LEVERAGE_SIM[0],
    fee_per_side=FEE_PER_SIDE, slippage=SLIPPAGE_PER_SIDE,
    max_hold=MAX_HOLDING_BARS, min_rr=SWING_LABEL_MIN_RR,
    min_tp_atr=SWING_LABEL_MIN_TP, max_sl_atr=SWING_LABEL_MAX_SL,
    tp_fallback_atr=TP_SL_FALLBACK_TP, sl_fallback_atr=TP_SL_FALLBACK_SL,
    guardian_enabled=False,
)

# Dump semua keys dan sample
print("=== RETURN KEYS ===")
for k, v in r.items():
    if isinstance(v, (list, np.ndarray)):
        print(f"  {k}: list/array len={len(v)}, sample={str(v[:2]) if len(v) > 0 else '[]'}")
    else:
        print(f"  {k}: {v}")

# Sample trade_log entry
tlog = r.get('trade_log', [])
if tlog:
    print(f"\n=== SAMPLE TRADE LOG ENTRY [0] ===")
    for k, v in tlog[0].items():
        print(f"  {k}: {v}")

# PnL dari pnl_per_trade
ppt = r.get('pnl_per_trade', [])
if len(ppt) > 0:
    arr = np.array([float(x) for x in ppt])
    print(f"\n=== PNL_PER_TRADE ({len(arr)} values) ===")
    print(f"  Total : ${arr.sum():,.2f}")
    print(f"  Min   : ${arr.min():,.2f}")
    print(f"  Max   : ${arr.max():,.2f}")
    print(f"  < -10 : {(arr < -10).sum()} trades")
    print(f"  < -50 : {(arr < -50).sum()} trades")
    worst_idx = np.argsort(arr)[:10]
    print(f"\nWorst 10 PPT indices: {worst_idx.tolist()}")
    for i in worst_idx:
        print(f"  idx={i} pnl=${arr[i]:.4f}")
        if i < len(tlog):
            t = tlog[i]
            print(f"    keys: {list(t.keys())}")
            print(f"    entry_bar={t.get('entry_bar', t.get('entry_idx', '?'))} "
                  f"dir={t.get('direction','?')} "
                  f"entry_px={t.get('entry_price', t.get('entry', '?'))} "
                  f"exit_px={t.get('exit_price', t.get('exit', '?'))} "
                  f"reason={t.get('exit_reason', t.get('reason', '?'))}")
