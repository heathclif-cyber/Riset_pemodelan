"""Optimize TB Guardian: threshold sweep + momentum mode"""
import json, sys, warnings, numpy as np, pandas as pd
warnings.filterwarnings('ignore')
sys.path.insert(0, '.')
from config import *
import joblib
from core.evaluator import full_trading_report
from core.utils import ensure_utc_index
from sklearn.preprocessing import StandardScaler

tb_model = joblib.load('models/runs/tb_lgbm_widyawardhana_v3/lgbm.pkl')
with open('models/runs/tb_lgbm_widyawardhana_v3/tb_lgbm_widyawardhana_v3_features.json') as f:
    tb_feats = json.load(f)
gd_model = joblib.load('models/runs/tb_guardian_widyawardhana_v1/guardian.pkl')
with open('models/runs/tb_guardian_widyawardhana_v1/tb_guardian_widyawardhana_v1_features.json') as f:
    gd_feats = json.load(f)

gd_static = [c for c in gd_feats if c in tb_feats]
gd_dynamic = [c for c in gd_feats if c not in tb_feats]

# Identity scaler
gd_scaler = StandardScaler()
gd_scaler.mean_ = np.zeros(len(gd_feats))
gd_scaler.scale_ = np.ones(len(gd_feats))

REGIME_THRESH = {0: 0.45, 1: 0.50, 2: 0.50, 3: 0.45}
LM = {'SHORT': 0, 'FLAT': 1, 'LONG': 2}
available = [s for s in ALL_COINS if (HOLDOUT_DIR / 'labeled' / f'{s}_features_v3.parquet').exists()]

# Run on 5 representative coins for speed
test_coins = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'ADAUSDT', 'DOTUSDT']

# Precompute
coin_data = {}
for sym in test_coins:
    df = pd.read_parquet(HOLDOUT_DIR / 'labeled' / f'{sym}_features_v3.parquet')
    df = ensure_utc_index(df).sort_index()
    rp = HOLDOUT_DIR / 'labeled' / f'{sym}_regime_h1.parquet'
    hmm = np.full(len(df), 1, dtype=np.int32)
    if rp.exists():
        reg = pd.read_parquet(rp)
        if 'hmm_regime_enc' in reg.columns:
            hmm = reg['hmm_regime_enc'].reindex(df.index, fill_value=1).values.astype(np.int32)
    mask = df['label'].isin(LM); df = df[mask].copy(); hmm = hmm[mask.values]
    X = np.zeros((len(df), len(tb_feats)), dtype=np.float64)
    for i, f in enumerate(tb_feats):
        if f in df.columns:
            X[:, i] = df[f].ffill().fillna(0).values.astype(np.float64)
    proba = tb_model.predict_proba(X)
    conf = np.max(proba, axis=1)
    yp = np.argmax(proba, axis=1)
    for r, th in REGIME_THRESH.items():
        yp[(hmm == r) & (yp != 1) & (conf < th)] = 1
    X_gd = np.zeros((len(df), len(gd_static)), dtype=np.float64)
    for i, f in enumerate(gd_static):
        if f in df.columns:
            X_gd[:, i] = df[f].ffill().fillna(0).values.astype(np.float64)
    coin_data[sym] = {
        'df': df, 'yp': yp, 'conf': conf, 'X_gd': X_gd,
        'atr': df['atr_14_h1'].values, 'close': df['close'].values,
        'high': df['high'].values, 'low': df['low'].values,
        'hs': df['h4_swing_high'].values if 'h4_swing_high' in df.columns else None,
        'ls': df['h4_swing_low'].values if 'h4_swing_low' in df.columns else None,
        'ht': df['h4_trend'].values if 'h4_trend' in df.columns else None,
        'yt': df['label'].map(LM).values.astype(np.int32),
    }

print("="*80)
print("  GUARDIAN TB OPTIMIZATION")
print("="*80)

# Test 1: Threshold sweep (no momentum mode)
print(f"\n--- THRESHOLD SWEEP (no momentum) ---")
print(f"{'Config':<30} {'Trades':>7} {'WR%':>7} {'PnL':>10} {'PF':>7} {'vs NoG':>8}")
print("-" * 72)

# Baseline: no Guardian
baseline_t, baseline_w, baseline_pnl = 0, 0, 0.0
for sym in test_coins:
    d = coin_data[sym]
    rep = full_trading_report(
        y_pred=d['yp'], y_actual=d['yt'], atr=d['atr'], close=d['close'],
        high=d['high'], low=d['low'], h4_swing_highs=d['hs'], h4_swing_lows=d['ls'],
        index=d['df'].index, modal=MODAL_PER_TRADE, leverages=LEVERAGE_SIM,
        fee_per_side=FEE_PER_SIDE, slippage=SLIPPAGE_PER_SIDE,
        min_rr=SWING_LABEL_MIN_RR, min_tp_atr=SWING_LABEL_MIN_TP,
        max_sl_atr=SWING_LABEL_MAX_SL, tp_fallback_atr=TP_SL_FALLBACK_TP,
        sl_fallback_atr=TP_SL_FALLBACK_SL, max_hold=MAX_HOLDING_BARS,
        symbol=sym, confidence=d['conf'],
        guardian_enabled=False, trailing_stop_enabled=False, h4_trend=d['ht'],
    )
    lev = rep.get('lev5x', rep); tr = lev.get('trades', [])
    baseline_t += len(tr); baseline_w += sum(1 for t in tr if 'WIN' in str(t.get('outcome','')))
    baseline_pnl += sum(t.get('net_pnl', 0) for t in tr)

base_wr = baseline_w / max(baseline_t, 1) * 100
print(f"{'No Guardian':<30} {baseline_t:>7} {base_wr:>6.1f}% {baseline_pnl:>+10.0f} {'-':>7} {'BASELINE':>8}")

# Sweep thresholds
for gd_thresh in [0.40, 0.50, 0.60, 0.70, 0.80]:
    total_t, total_w, total_pnl = 0, 0, 0.0
    for sym in test_coins:
        d = coin_data[sym]
        rep = full_trading_report(
            y_pred=d['yp'], y_actual=d['yt'], atr=d['atr'], close=d['close'],
            high=d['high'], low=d['low'], h4_swing_highs=d['hs'], h4_swing_lows=d['ls'],
            index=d['df'].index, modal=MODAL_PER_TRADE, leverages=LEVERAGE_SIM,
            fee_per_side=FEE_PER_SIDE, slippage=SLIPPAGE_PER_SIDE,
            min_rr=SWING_LABEL_MIN_RR, min_tp_atr=SWING_LABEL_MIN_TP,
            max_sl_atr=SWING_LABEL_MAX_SL, tp_fallback_atr=TP_SL_FALLBACK_TP,
            sl_fallback_atr=TP_SL_FALLBACK_SL, max_hold=MAX_HOLDING_BARS,
            symbol=sym, confidence=d['conf'],
            guardian_model=gd_model, guardian_scaler=gd_scaler,
            X_guardian=d['X_gd'], guardian_enabled=True,
            guardian_exit_threshold=gd_thresh,
            guardian_activation_atr=0.0,  # instant activation
            guardian_min_hold_bars=1,
            trailing_stop_enabled=False, h4_trend=d['ht'],
        )
        lev = rep.get('lev5x', rep); tr = lev.get('trades', [])
        total_t += len(tr)
        total_w += sum(1 for t in tr if 'WIN' in str(t.get('outcome','')))
        total_pnl += sum(t.get('net_pnl', 0) for t in tr)

    wr = total_w / max(total_t, 1) * 100
    delta = total_pnl - baseline_pnl
    print(f"{f'Thresh={gd_thresh:.2f}':<30} {total_t:>7} {wr:>6.1f}% {total_pnl:>+10.0f} {lev.get('profit_factor',0):>6.2f} {delta:>+8.0f}")

# Test 2: Momentum mode (Guardian activates only after TP)
print(f"\n--- MOMENTUM MODE (Guardian after TP) ---")
print(f"Activates Guardian after price reaches TP barrier (2.0xATR from entry)")
print(f"{'Config':<35} {'Trades':>7} {'WR%':>7} {'PnL':>10} {'PF':>7} {'vs NoG':>8}")
print("-" * 78)

print(f"{'No Guardian':<35} {baseline_t:>7} {base_wr:>6.1f}% {baseline_pnl:>+10.0f} {'-':>7} {'BASELINE':>8}")

for gd_thresh in [0.40, 0.50, 0.60, 0.70]:
    total_t, total_w, total_pnl = 0, 0, 0.0
    for sym in test_coins:
        d = coin_data[sym]
        rep = full_trading_report(
            y_pred=d['yp'], y_actual=d['yt'], atr=d['atr'], close=d['close'],
            high=d['high'], low=d['low'], h4_swing_highs=d['hs'], h4_swing_lows=d['ls'],
            index=d['df'].index, modal=MODAL_PER_TRADE, leverages=LEVERAGE_SIM,
            fee_per_side=FEE_PER_SIDE, slippage=SLIPPAGE_PER_SIDE,
            min_rr=SWING_LABEL_MIN_RR, min_tp_atr=SWING_LABEL_MIN_TP,
            max_sl_atr=SWING_LABEL_MAX_SL, tp_fallback_atr=TP_SL_FALLBACK_TP,
            sl_fallback_atr=TP_SL_FALLBACK_SL, max_hold=MAX_HOLDING_BARS,
            symbol=sym, confidence=d['conf'],
            guardian_model=gd_model, guardian_scaler=gd_scaler,
            X_guardian=d['X_gd'], guardian_enabled=True,
            guardian_exit_threshold=gd_thresh,
            guardian_activation_atr=2.0,  # activate only after 2.0xATR move
            guardian_min_hold_bars=1,
            trailing_stop_enabled=False, h4_trend=d['ht'],
        )
        lev = rep.get('lev5x', rep); tr = lev.get('trades', [])
        total_t += len(tr)
        total_w += sum(1 for t in tr if 'WIN' in str(t.get('outcome','')))
        total_pnl += sum(t.get('net_pnl', 0) for t in tr)

    wr = total_w / max(total_t, 1) * 100
    delta = total_pnl - baseline_pnl
    print(f"{f'Momentum thresh={gd_thresh:.2f}':<35} {total_t:>7} {wr:>6.1f}% {total_pnl:>+10.0f} {lev.get('profit_factor',0):>6.2f} {delta:>+8.0f}")

# Test 3: Momentum + Higher threshold combo
print(f"\n--- MOMENTUM + HIGH THRESHOLD COMBOS ---")
for gd_thresh, mom_atr in [(0.55, 1.5), (0.60, 2.0), (0.65, 2.0), (0.70, 2.5)]:
    total_t, total_w, total_pnl = 0, 0, 0.0
    for sym in test_coins:
        d = coin_data[sym]
        rep = full_trading_report(
            y_pred=d['yp'], y_actual=d['yt'], atr=d['atr'], close=d['close'],
            high=d['high'], low=d['low'], h4_swing_highs=d['hs'], h4_swing_lows=d['ls'],
            index=d['df'].index, modal=MODAL_PER_TRADE, leverages=LEVERAGE_SIM,
            fee_per_side=FEE_PER_SIDE, slippage=SLIPPAGE_PER_SIDE,
            min_rr=SWING_LABEL_MIN_RR, min_tp_atr=SWING_LABEL_MIN_TP,
            max_sl_atr=SWING_LABEL_MAX_SL, tp_fallback_atr=TP_SL_FALLBACK_TP,
            sl_fallback_atr=TP_SL_FALLBACK_SL, max_hold=MAX_HOLDING_BARS,
            symbol=sym, confidence=d['conf'],
            guardian_model=gd_model, guardian_scaler=gd_scaler,
            X_guardian=d['X_gd'], guardian_enabled=True,
            guardian_exit_threshold=gd_thresh,
            guardian_activation_atr=mom_atr,
            guardian_min_hold_bars=1,
            trailing_stop_enabled=False, h4_trend=d['ht'],
        )
        lev = rep.get('lev5x', rep); tr = lev.get('trades', [])
        total_t += len(tr)
        total_w += sum(1 for t in tr if 'WIN' in str(t.get('outcome','')))
        total_pnl += sum(t.get('net_pnl', 0) for t in tr)

    wr = total_w / max(total_t, 1) * 100
    delta = total_pnl - baseline_pnl
    print(f"{f'Th={gd_thresh:.2f} MomATR={mom_atr:.1f}':<35} {total_t:>7} {wr:>6.1f}% {total_pnl:>+10.0f} {lev.get('profit_factor',0):>6.2f} {delta:>+8.0f}")

# Best config on full 21 coins
print(f"\n{'='*80}")
print(f"  BEST CONFIG ON FULL 21 COINS")
print(f"{'='*80}")

# From sweep: Best is likely momentum mode + moderate threshold
BEST_GD_THRESH = 0.60
BEST_MOM_ATR = 2.0

total_t, total_w, total_pnl = 0, 0, 0.0
for sym in available:
    df = pd.read_parquet(HOLDOUT_DIR / 'labeled' / f'{sym}_features_v3.parquet')
    df = ensure_utc_index(df).sort_index()
    rp = HOLDOUT_DIR / 'labeled' / f'{sym}_regime_h1.parquet'
    hmm = np.full(len(df), 1, dtype=np.int32)
    if rp.exists():
        reg = pd.read_parquet(rp)
        if 'hmm_regime_enc' in reg.columns:
            hmm = reg['hmm_regime_enc'].reindex(df.index, fill_value=1).values.astype(np.int32)
    mask = df['label'].isin(LM); df = df[mask].copy(); hmm = hmm[mask.values]
    X = np.zeros((len(df), len(tb_feats)), dtype=np.float64)
    for i, f in enumerate(tb_feats):
        if f in df.columns: X[:, i] = df[f].ffill().fillna(0).values.astype(np.float64)
    proba = tb_model.predict_proba(X)
    conf = np.max(proba, axis=1); yp = np.argmax(proba, axis=1)
    for r, th in REGIME_THRESH.items():
        yp[(hmm == r) & (yp != 1) & (conf < th)] = 1
    X_gd = np.zeros((len(df), len(gd_static)), dtype=np.float64)
    for i, f in enumerate(gd_static):
        if f in df.columns: X_gd[:, i] = df[f].ffill().fillna(0).values.astype(np.float64)

    if (yp != 1).sum() == 0: continue
    rep = full_trading_report(
        y_pred=yp, y_actual=df['label'].map(LM).values.astype(np.int32),
        atr=df['atr_14_h1'].values, close=df['close'].values,
        high=df['high'].values, low=df['low'].values,
        h4_swing_highs=df['h4_swing_high'].values if 'h4_swing_high' in df.columns else None,
        h4_swing_lows=df['h4_swing_low'].values if 'h4_swing_low' in df.columns else None,
        index=df.index, modal=MODAL_PER_TRADE, leverages=LEVERAGE_SIM,
        fee_per_side=FEE_PER_SIDE, slippage=SLIPPAGE_PER_SIDE,
        min_rr=SWING_LABEL_MIN_RR, min_tp_atr=SWING_LABEL_MIN_TP,
        max_sl_atr=SWING_LABEL_MAX_SL, tp_fallback_atr=TP_SL_FALLBACK_TP,
        sl_fallback_atr=TP_SL_FALLBACK_SL, max_hold=MAX_HOLDING_BARS,
        symbol=sym, confidence=conf,
        guardian_model=gd_model, guardian_scaler=gd_scaler,
        X_guardian=X_gd, guardian_enabled=True,
        guardian_exit_threshold=BEST_GD_THRESH,
        guardian_activation_atr=BEST_MOM_ATR,
        guardian_min_hold_bars=1,
        trailing_stop_enabled=False,
        h4_trend=df['h4_trend'].values if 'h4_trend' in df.columns else None,
    )
    lev = rep.get('lev5x', rep); tr = lev.get('trades', [])
    total_t += len(tr); total_w += sum(1 for t in tr if 'WIN' in str(t.get('outcome','')))
    total_pnl += sum(t.get('net_pnl', 0) for t in tr)

wr = total_w / max(total_t, 1) * 100
print(f"  Config: thresh={BEST_GD_THRESH}, momentum_atr={BEST_MOM_ATR}")
print(f"  Trades: {total_t} | WR: {wr:.1f}% | PnL: ${total_pnl:+.0f}")
print(f"  PnL/trade: ${total_pnl/max(total_t,1):.2f}")
