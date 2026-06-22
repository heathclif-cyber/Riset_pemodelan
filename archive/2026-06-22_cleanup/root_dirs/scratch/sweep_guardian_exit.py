"""Sweep Guardian exit parameters for TB v3"""
import json, sys, warnings, numpy as np, pandas as pd
warnings.filterwarnings('ignore')
sys.path.insert(0, '.')
from config import *
import joblib
from core.evaluator import full_trading_report
from core.utils import ensure_utc_index
from sklearn.preprocessing import StandardScaler

# Load models
tb = joblib.load('models/runs/tb_lgbm_widyawardhana_v3/lgbm.pkl')
with open('models/runs/tb_lgbm_widyawardhana_v3/tb_lgbm_widyawardhana_v3_features.json') as f:
    tb_feats = json.load(f)
with open('models/runs/tb_guardian_widyawardhana_v1/tb_guardian_widyawardhana_v1_features.json') as f:
    gd_feats = json.load(f)
gd = joblib.load('models/runs/tb_guardian_widyawardhana_v1/guardian.pkl')
gd_static = [c for c in gd_feats if c in tb_feats]

# Dummy scaler
gd_scaler = StandardScaler()
gd_scaler.mean_ = np.zeros(len(gd_feats))
gd_scaler.scale_ = np.ones(len(gd_feats))

REGIME_THRESH = {0: 0.45, 1: 0.50, 2: 0.50, 3: 0.45}
LM = {'SHORT': 0, 'FLAT': 1, 'LONG': 2}

# Test on 3 coins for speed
coins = ['BTCUSDT', 'ETHUSDT', 'ADAUSDT']
print("Loading data...")
data = {}
for sym in coins:
    df = pd.read_parquet(HOLDOUT_DIR / 'labeled' / f'{sym}_features_v3.parquet')
    df = ensure_utc_index(df).sort_index()
    rp = HOLDOUT_DIR / 'labeled' / f'{sym}_regime_h1.parquet'
    hmm = np.full(len(df), 1, dtype=np.int32)
    if rp.exists():
        reg = pd.read_parquet(rp); hmm = reg['hmm_regime_enc'].reindex(df.index, fill_value=1).values.astype(np.int32)
    mask = df['label'].isin(LM); df = df[mask].copy(); hmm = hmm[mask.values]
    X = np.zeros((len(df), len(tb_feats)), dtype=np.float64)
    for i, f in enumerate(tb_feats):
        if f in df.columns: X[:, i] = df[f].ffill().fillna(0).values.astype(np.float64)
    proba = tb.predict_proba(X); conf = np.max(proba, axis=1)
    yp = np.argmax(proba, axis=1)
    for r, th in REGIME_THRESH.items(): yp[(hmm==r)&(yp!=1)&(conf<th)] = 1
    Xg = np.zeros((len(df), len(gd_static)), dtype=np.float64)
    for i, f in enumerate(gd_static):
        if f in df.columns: Xg[:, i] = df[f].ffill().fillna(0).values.astype(np.float64)
    data[sym] = {'df': df, 'yp': yp, 'conf': conf, 'Xg': Xg,
                 'a': df['atr_14_h1'].values, 'c': df['close'].values,
                 'hi': df['high'].values, 'lo': df['low'].values,
                 'yt': df['label'].map(LM).values.astype(np.int32),
                 'hs': df['h4_swing_high'].values if 'h4_swing_high' in df.columns else None,
                 'ls': df['h4_swing_low'].values if 'h4_swing_low' in df.columns else None,
                 'ht': df['h4_trend'].values if 'h4_trend' in df.columns else None}

# Baseline: no Guardian
print("Computing baseline (no Guardian)...")
base_t, base_w, base_pnl = 0, 0, 0.0
for sym in coins:
    d = data[sym]
    rep = full_trading_report(
        y_pred=d['yp'], y_actual=d['yt'], atr=d['a'], close=d['c'],
        high=d['hi'], low=d['lo'], h4_swing_highs=d['hs'], h4_swing_lows=d['ls'],
        index=d['df'].index, modal=MODAL_PER_TRADE, leverages=LEVERAGE_SIM,
        fee_per_side=FEE_PER_SIDE, slippage=SLIPPAGE_PER_SIDE,
        min_rr=SWING_LABEL_MIN_RR, min_tp_atr=SWING_LABEL_MIN_TP,
        max_sl_atr=SWING_LABEL_MAX_SL, tp_fallback_atr=TP_SL_FALLBACK_TP,
        sl_fallback_atr=TP_SL_FALLBACK_SL, max_hold=MAX_HOLDING_BARS,
        symbol=sym, confidence=d['conf'],
        guardian_enabled=False, trailing_stop_enabled=False, h4_trend=d['ht'])
    lev = rep.get('lev5x', rep); tr = lev.get('trades', [])
    base_t += len(tr); base_w += sum(1 for t in tr if 'WIN' in str(t.get('outcome','')))
    base_pnl += sum(t.get('net_pnl', 0) for t in tr)
base_wr = base_w/max(base_t,1)*100
print(f"Baseline: {base_t} trades, WR={base_wr:.1f}%, PnL=${base_pnl:+.0f}")

# Sweep Guardian parameters
print(f"\n{'='*80}")
print(f"  GUARDIAN PARAMETER SWEEP")
print(f"{'='*80}")

# Parameters to sweep
params = [
    # (activation_atr, exit_thresh, sl_safety_atr, min_hold, guardian_tp_atr, label)
    # Test 1: Baseline Guardian (instant activation)
    (0.0, 0.60, 1.5, 2, 2.0, "base"),
    # Test 2: Higher exit threshold (less intervention)
    (0.0, 0.70, 1.5, 2, 2.0, "thresh=0.70"),
    (0.0, 0.80, 1.5, 2, 2.0, "thresh=0.80"),
    # Test 3: Delayed activation (let profit run)
    (0.5, 0.60, 1.5, 2, 2.0, "act=0.5ATR"),
    (1.0, 0.60, 1.5, 2, 2.0, "act=1.0ATR"),
    (1.5, 0.60, 1.5, 2, 2.0, "act=1.5ATR"),
    (2.0, 0.60, 1.5, 2, 2.0, "act=2.0ATR"),
    # Test 4: Longer min hold
    (0.0, 0.60, 1.5, 3, 2.0, "minhold=3"),
    (0.0, 0.60, 1.5, 4, 2.0, "minhold=4"),
    (1.0, 0.60, 1.5, 3, 2.0, "act1.0_min3"),
    (1.0, 0.60, 1.5, 4, 2.0, "act1.0_min4"),
    # Test 5: Higher TP extension with momentum
    (1.0, 0.60, 1.5, 2, 2.5, "TPext=2.5"),
    (1.0, 0.60, 1.5, 2, 3.0, "TPext=3.0"),
    # Test 6: Tighter SL safety
    (1.0, 0.60, 1.0, 2, 2.5, "SLsafe=1.0"),
    # Test 7: Best combos
    (1.5, 0.70, 1.5, 3, 3.0, "act1.5_th0.7_mh3_tp3"),
    (2.0, 0.70, 1.5, 3, 3.0, "act2.0_th0.7_mh3_tp3"),
    (1.5, 0.80, 1.5, 4, 3.0, "act1.5_th0.8_mh4_tp3"),
    (2.0, 0.80, 1.5, 4, 3.0, "act2.0_th0.8_mh4_tp3"),
]

print(f"\n{'Config':<30} {'Tr':>6} {'WR%':>7} {'PnL':>9} {'PF':>6} {'vs Base':>8}")
print("-" * 72)

best_pnl, best_cfg = -99999, ""

for act_atr, ex_th, sl_safe, min_h, tp_ext, label in params:
    total_t, total_w, total_pnl = 0, 0, 0.0
    for sym in coins:
        d = data[sym]
        rep = full_trading_report(
            y_pred=d['yp'], y_actual=d['yt'], atr=d['a'], close=d['c'],
            high=d['hi'], low=d['lo'], h4_swing_highs=d['hs'], h4_swing_lows=d['ls'],
            index=d['df'].index, modal=MODAL_PER_TRADE, leverages=LEVERAGE_SIM,
            fee_per_side=FEE_PER_SIDE, slippage=SLIPPAGE_PER_SIDE,
            min_rr=SWING_LABEL_MIN_RR, min_tp_atr=SWING_LABEL_MIN_TP,
            max_sl_atr=SWING_LABEL_MAX_SL, tp_fallback_atr=TP_SL_FALLBACK_TP,
            sl_fallback_atr=TP_SL_FALLBACK_SL, max_hold=MAX_HOLDING_BARS,
            symbol=sym, confidence=d['conf'],
            guardian_model=gd, guardian_scaler=gd_scaler, X_guardian=d['Xg'],
            guardian_enabled=True,
            guardian_exit_threshold=ex_th,
            guardian_activation_atr=act_atr,
            guardian_sl_safety_atr=sl_safe,
            guardian_min_hold_bars=min_h,
            guardian_tp_atr=tp_ext,
            trailing_stop_enabled=False, h4_trend=d['ht'])
        lev = rep.get('lev5x', rep); tr = lev.get('trades', [])
        total_t += len(tr)
        total_w += sum(1 for t in tr if 'WIN' in str(t.get('outcome','')))
        total_pnl += sum(t.get('net_pnl', 0) for t in tr)
    wr = total_w/max(total_t,1)*100
    pf = lev.get('profit_factor', 0)
    delta = total_pnl - base_pnl
    print(f"{label:<30} {total_t:>6} {wr:>6.1f}% {total_pnl:>+9.0f} {pf:>5.2f} {delta:>+8.0f}")
    if total_pnl > best_pnl:
        best_pnl, best_cfg = total_pnl, label

print(f"\n  BEST: {best_cfg} (PnL=${best_pnl:+.0f})")
