import json, sys, warnings, numpy as np, pandas as pd
warnings.filterwarnings('ignore')
sys.path.insert(0, '.')
from config import *
import joblib
from core.features import triple_barrier_labeling
from core.evaluator import full_trading_report
from core.utils import ensure_utc_index

model = joblib.load('models/runs/tb_lgbm_widyawardhana_v2/lgbm.pkl')
with open('models/runs/tb_lgbm_widyawardhana_v2/tb_lgbm_widyawardhana_v2_features.json') as f:
    feats = json.load(f)

for sym in ['BTCUSDT', 'SOLUSDT', 'ETHUSDT', 'DOTUSDT']:
    df = pd.read_parquet(HOLDOUT_DIR / 'labeled' / f'{sym}_features_v3.parquet')
    df = ensure_utc_index(df).sort_index()
    tb = triple_barrier_labeling(df['close'], df['high'], df['low'],
                                 df['atr_14_h1'], 2.0, 1.5, 36)
    df['tb_label'] = tb.map({'SHORT': 0, 'FLAT': 1, 'LONG': 2})
    df_bin = df[df['tb_label'] != 1].copy()

    X = np.zeros((len(df_bin), len(feats)), dtype=np.float64)
    for idx, col in enumerate(feats):
        if col in df_bin.columns:
            X[:, idx] = df_bin[col].ffill().fillna(0).values.astype(np.float64)
    proba = model.predict_proba(X)
    conf = np.max(proba, axis=1)
    y_pred = np.where(proba[:, 1] > 0.5, 2, 0).astype(np.int32)
    y_pred[conf < 0.55] = 1

    report = full_trading_report(
        y_pred=y_pred, y_actual=df_bin['tb_label'].values,
        atr=df_bin['atr_14_h1'].values, close=df_bin['close'].values,
        high=df_bin['high'].values, low=df_bin['low'].values,
        index=df_bin.index, modal=MODAL_PER_TRADE, leverages=LEVERAGE_SIM,
        fee_per_side=FEE_PER_SIDE, slippage=SLIPPAGE_PER_SIDE,
        min_rr=SWING_LABEL_MIN_RR, min_tp_atr=SWING_LABEL_MIN_TP,
        max_sl_atr=SWING_LABEL_MAX_SL,
        tp_fallback_atr=TP_SL_FALLBACK_TP, sl_fallback_atr=TP_SL_FALLBACK_SL,
        max_hold=MAX_HOLDING_BARS, symbol=sym, confidence=conf,
        guardian_enabled=False, trailing_stop_enabled=False,
    )

    lev5 = report.get('lev5x', report)
    eq = np.array(lev5.get('equity_curve', [0]))
    peak = np.maximum.accumulate(eq)
    dd_amt = peak - eq
    dd_correct = np.where(peak > 0, dd_amt / peak * 100, 0)
    dd_bug = dd_amt.max() / MODAL_PER_TRADE * 100 if MODAL_PER_TRADE > 0 else 0
    n_trades = len(lev5.get('trades', []))
    total_pnl = eq[-1]

    print(f'{sym}: {n_trades} trades | PnL=${total_pnl:.0f} | '
          f'Peak=${peak.max():.0f} | DD(bug)={dd_bug:.0f}% | DD(correct)={dd_correct.max():.0f}%')
