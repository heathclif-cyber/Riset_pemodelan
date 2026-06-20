"""TB v3 + Momentum Guardian (binary HOLD/EXIT) — custom per-bar simulation"""
import json, sys, warnings, numpy as np, pandas as pd
warnings.filterwarnings('ignore')
sys.path.insert(0, '.')
from config import *
import joblib
from core.evaluator import simulate_trades_swing
from core.utils import ensure_utc_index

tb_model = joblib.load('models/runs/tb_lgbm_widyawardhana_v3/lgbm.pkl')
with open('models/runs/tb_lgbm_widyawardhana_v3/tb_lgbm_widyawardhana_v3_features.json') as f:
    tb_feats = json.load(f)
gd_model = joblib.load('models/runs/tb_guardian_momentum_v1/guardian.pkl')
with open('models/runs/tb_guardian_momentum_v1/tb_guardian_momentum_v1_features.json') as f:
    gd_feats = json.load(f)

REGIME_THRESH = {0: 0.45, 1: 0.50, 2: 0.50, 3: 0.45}
LM = {'SHORT': 0, 'FLAT': 1, 'LONG': 2}
available = [s for s in ALL_COINS if (HOLDOUT_DIR / 'labeled' / f'{s}_features_v3.parquet').exists()]

# Test on 5 coins
test_coins = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'ADAUSDT', 'DOTUSDT']

print(f"Guardian Momentum features: {len(gd_feats)}")

# Baseline: no Guardian
print(f"\n{'='*70}")
print(f"  BASELINE: TB v3 No Guardian")
print(f"{'='*70}")
base_trades_list = []

for sym in available:
    df = pd.read_parquet(HOLDOUT_DIR / 'labeled' / f'{sym}_features_v3.parquet')
    df = ensure_utc_index(df).sort_index()
    rp = HOLDOUT_DIR / 'labeled' / f'{sym}_regime_h1.parquet'
    hmm = np.full(len(df), 1, dtype=np.int32)
    if rp.exists():
        reg = pd.read_parquet(rp); hmm = reg['hmm_regime_enc'].reindex(df.index, fill_value=1).values.astype(np.int32)
    mask = df['label'].isin(LM); df = df[mask].copy(); hmm = hmm[mask.values]
    X = np.zeros((len(df), len(tb_feats)), dtype=np.float64)
    for i, f in enumerate(tb_feats):
        if f in df.columns: X[:,i] = df[f].ffill().fillna(0).values.astype(np.float64)
    proba = tb_model.predict_proba(X); conf = np.max(proba, axis=1)
    yp = np.argmax(proba, axis=1)
    for r, th in REGIME_THRESH.items(): yp[(hmm==r)&(yp!=1)&(conf<th)] = 1
    yt = df['label'].map(LM).values.astype(np.int32)

    result = simulate_trades_swing(
        y_pred=yp, close=df['close'].values, high=df['high'].values, low=df['low'].values,
        atr=df['atr_14_h1'].values, h4_swing_highs=np.full(len(df), np.nan),
        h4_swing_lows=np.full(len(df), np.nan),
        modal=MODAL_PER_TRADE, leverage=LEVERAGE_SIM[0],
        fee_per_side=FEE_PER_SIDE, slippage=SLIPPAGE_PER_SIDE, max_hold=MAX_HOLDING_BARS,
        min_rr=SWING_LABEL_MIN_RR, min_tp_atr=SWING_LABEL_MIN_TP,
        max_sl_atr=SWING_LABEL_MAX_SL,
        tp_fallback_atr=TP_SL_FALLBACK_TP, sl_fallback_atr=TP_SL_FALLBACK_SL,
        guardian_enabled=False, trailing_stop_enabled=False,
    )
    trades = result.get('trades', [])
    wins = sum(1 for t in trades if 'WIN' in str(t.get('outcome','')))
    pnl = sum(t.get('net_pnl',0) for t in trades)
    base_trades_list.append({'sym': sym, 'trades': len(trades), 'wins': wins, 'pnl': pnl, 'wr': wins/max(len(trades),1)*100})
    print(f"  {sym}: {len(trades)} trades, WR={wins/max(len(trades),1)*100:.1f}%, PnL=${pnl:+.0f}")

base_t = sum(r['trades'] for r in base_trades_list)
base_w = sum(r['wins'] for r in base_trades_list)
base_pnl = sum(r['pnl'] for r in base_trades_list)
print(f"  TOTAL: {base_t} trades, WR={base_w/max(base_t,1)*100:.1f}%, PnL=${base_pnl:+.0f}")

# Now: custom momentum Guardian per-bar simulation
print(f"\n{'='*70}")
print(f"  MOMENTUM GUARDIAN — Custom per-bar simulation")
print(f"{'='*70}")

for gd_thresh in [0.60, 0.65, 0.70, 0.75]:
    print(f"\n--- Guardian thresh={gd_thresh:.2f} ---")

    total_trades, total_wins, total_pnl = 0, 0, 0.0
    for sym in available:
        df = pd.read_parquet(HOLDOUT_DIR / 'labeled' / f'{sym}_features_v3.parquet')
        df = ensure_utc_index(df).sort_index()
        rp = HOLDOUT_DIR / 'labeled' / f'{sym}_regime_h1.parquet'
        hmm = np.full(len(df), 1, dtype=np.int32)
        if rp.exists():
            reg = pd.read_parquet(rp); hmm = reg['hmm_regime_enc'].reindex(df.index, fill_value=1).values.astype(np.int32)
        mask = df['label'].isin(LM); df = df[mask].copy(); hmm = hmm[mask.values]

        # TB inference
        X = np.zeros((len(df), len(tb_feats)), dtype=np.float64)
        for i, f in enumerate(tb_feats):
            if f in df.columns: X[:,i] = df[f].ffill().fillna(0).values.astype(np.float64)
        proba = tb_model.predict_proba(X); conf = np.max(proba, axis=1)
        yp = np.argmax(proba, axis=1)
        for r, th in REGIME_THRESH.items(): yp[(hmm==r)&(yp!=1)&(conf<th)] = 1

        # Precompute Guardian features
        X_gd = np.zeros((len(df), len(gd_feats)), dtype=np.float64)
        for i, f in enumerate(gd_feats):
            if f in df.columns: X_gd[:,i] = df[f].ffill().fillna(0).values.astype(np.float64)

        # Custom trade simulation with momentum Guardian
        close = df['close'].values; high = df['high'].values; low = df['low'].values
        atr = df['atr_14_h1'].values; n = len(close)

        trades = []
        in_trade = False; trade_dir = 0; entry_bar = 0; entry_price = 0.0

        for i in range(n - 1):
            if yp[i] == 1:  # FLAT
                continue

            if not in_trade and yp[i] != 1:
                # Open trade
                in_trade = True; trade_dir = 1 if yp[i] == 2 else -1
                entry_bar = i; entry_price = close[i]
                continue

            if in_trade:
                bars_held = i - entry_bar
                if bars_held < 2:  # min hold
                    continue

                # Check Guardian: should we exit?
                gd_features = X_gd[i:i+1]  # momentum features at bar i
                gd_proba = gd_model.predict_proba(gd_features)[0]
                # gd_proba = [P(HOLD), P(PARTIAL), P(EXIT)]
                p_exit = gd_proba[2]  # P(FULL_EXIT)

                # Compute current PnL
                if trade_dir == 1:
                    pnl_pct = (close[i] - entry_price) / entry_price
                else:
                    pnl_pct = (entry_price - close[i]) / entry_price

                # Exit conditions:
                # 1. Guardian says EXIT with high confidence
                # 2. PnL reached extreme loss (SL territory)
                # 3. Time limit

                exit_now = False; exit_reason = ""

                if p_exit > gd_thresh and pnl_pct < 0.02:
                    # Guardian detects momentum reversal + not in strong profit
                    exit_now = True; exit_reason = "guardian_momentum"
                elif pnl_pct < -0.05:  # hard SL at -5%
                    exit_now = True; exit_reason = "hard_sl"
                elif bars_held >= MAX_HOLDING_BARS:
                    exit_now = True; exit_reason = "time_exit"
                elif pnl_pct > 0.04 and p_exit > gd_thresh:
                    # Good profit but momentum dying
                    exit_now = True; exit_reason = "guardian_tp"

                if exit_now:
                    if trade_dir == 1:
                        exit_price = close[i] * (1 - FEE_PER_SIDE)
                        trade_pnl = (exit_price - entry_price) * MODAL_PER_TRADE * LEVERAGE_SIM[0] / entry_price
                    else:
                        exit_price = close[i] * (1 + FEE_PER_SIDE)
                        trade_pnl = (entry_price - exit_price) * MODAL_PER_TRADE * LEVERAGE_SIM[0] / entry_price

                    outcome = "WIN" if trade_pnl > 0 else "LOSS"
                    trades.append({'outcome': outcome, 'net_pnl': trade_pnl,
                                   'bar_in': entry_bar, 'bar_out': i,
                                   'direction': 'LONG' if trade_dir==1 else 'SHORT',
                                   'exit_reason': exit_reason})
                    in_trade = False
                    continue

        wins = sum(1 for t in trades if t['outcome'] == 'WIN')
        pnl = sum(t['net_pnl'] for t in trades)
        wr = wins/max(len(trades),1)*100
        total_trades += len(trades); total_wins += wins; total_pnl += pnl
        print(f"  {sym}: {len(trades)} trades, WR={wr:.1f}%, PnL=${pnl:+.0f} "
              f"(exits: {sum(1 for t in trades if 'guardian' in t.get('exit_reason',''))} guardian)")

    wr = total_wins/max(total_trades,1)*100
    pnl_t = total_pnl/max(total_trades,1)
    print(f"  TOTAL: {total_trades} trades, WR={wr:.1f}%, PnL=${total_pnl:+.0f}, PnL/t=${pnl_t:+.2f}")
    print(f"  vs Baseline: PnL delta=${total_pnl - base_pnl:+.0f}")
