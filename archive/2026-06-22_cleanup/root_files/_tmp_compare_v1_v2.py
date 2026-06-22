"""
Perbandingan holdout ic32_regime_v1 (live) vs ic32_regime_v2 (new)
V1 config diambil langsung dari D:/Apps-Dev/swint_tradev2/models/inference_config.json
"""
import sys, json, numpy as np, pandas as pd, joblib, torch
sys.path.insert(0, '.')
from config import (ALL_COINS, MODEL_DIR, HOLDOUT_DIR,
    MODAL_PER_TRADE, LEVERAGE_SIM, FEE_PER_SIDE, SLIPPAGE_PER_SIDE,
    MAX_HOLDING_BARS, SWING_LABEL_MIN_RR, SWING_LABEL_MIN_TP,
    SWING_LABEL_MAX_SL, TP_SL_FALLBACK_TP, TP_SL_FALLBACK_SL)
from core.evaluator import simulate_trades_swing
from core.utils import ensure_utc_index
from core.models import TradingLSTM

SWINT = 'D:/Apps-Dev/swint_tradev2/models'
OOS_START = pd.Timestamp('2026-04-01', tz='UTC')
OOS_END   = pd.Timestamp('2026-06-20 16:00', tz='UTC')
LM = {'SHORT': 0, 'FLAT': 1, 'LONG': 2}
SEQ_LEN    = 72  # V1 LSTM seq_len — production default (_lstm_momentum_seq_len: 72)
SEQ_LEN_V2 = 72  # V2 LSTM seq_len (ic32_lstm_regime_v2)

# =========================================================
# V1 LIVE CONFIG  (dari swint_tradev2)
# =========================================================
lgbm_v1    = joblib.load(f'{SWINT}/lgbm_baseline.pkl')
feats_lgbm = json.load(open(f'{SWINT}/feature_cols_v2.json'))

lstm_sd    = torch.load(f'{SWINT}/lstm_best.pt', map_location='cpu', weights_only=False)
lstm_v1    = TradingLSTM(n_features=11, hidden_size=96, num_layers=2, num_classes=3)
lstm_v1.load_state_dict(lstm_sd)
lstm_v1.eval()
lstm_scaler = joblib.load(f'{SWINT}/lstm_scaler.pkl')
feats_lstm  = json.load(open(f'{SWINT}/feature_cols_lstm_temporal.json'))  # 11 feats

g1         = joblib.load(f'{SWINT}/guardian_best.pkl')
g1_scl     = joblib.load(f'{SWINT}/guardian_scaler.pkl')
g1_feats   = json.load(open(f'{SWINT}/guardian_feature_cols.json'))
G1_DYNAMIC = {'bars_held_norm', 'current_pnl_pct', 'current_pnl_atr', 'max_favorable_pnl_pct',
              'drawdown_from_peak_pct', 'direction', 'entry_price_ratio',
              'cvd_slope_h4_delta_entry', 'ofi_h4_delta_entry', 'flow_momentum_3bar'}
g1_static  = [f for f in g1_feats if f not in G1_DYNAMIC]

PER_STATE_THR = {0: [0.72, 0.72], 1: [0.72, 0.72], 2: [0.72, 0.72],
                 3: [0.65, 0.72], -1: [0.69, 0.59]}
LSTM_VETO_THR = 0.5  # lstm_no_veto_threshold

# =========================================================
# V2 NEW CONFIG
# =========================================================
lgbm_v2    = joblib.load(MODEL_DIR / 'runs' / 'ic32_regime_v2' / 'lgbm.pkl')
feats_v2   = json.load(open(MODEL_DIR / 'runs' / 'ic32_regime_v2' / 'features.json'))

# LSTM V2 (complement model, seq=72)
lstm_v2_sd  = torch.load(str(MODEL_DIR / 'runs' / 'ic32_lstm_regime_v2' / 'lstm_momentum.pt'), map_location='cpu', weights_only=False)
lstm_v2m    = TradingLSTM(n_features=11, hidden_size=96, num_layers=2, num_classes=3)
lstm_v2m.load_state_dict(lstm_v2_sd)
lstm_v2m.eval()
lstm_v2_scaler = joblib.load(MODEL_DIR / 'runs' / 'ic32_lstm_regime_v2' / 'lstm_momentum_scaler.pkl')
feats_lstm_v2  = json.load(open(MODEL_DIR / 'runs' / 'ic32_lstm_regime_v2' / 'lstm_v4_selected_features.json'))

g2         = joblib.load(MODEL_DIR / 'runs' / 'ic32_rv2_guard_oof' / 'guardian.pkl')
g2_scl     = joblib.load(MODEL_DIR / 'runs' / 'ic32_rv2_guard_oof' / 'guardian_scaler.pkl')
g2_feats   = json.load(open(MODEL_DIR / 'runs' / 'ic32_rv2_guard_oof' / 'guardian_features.json'))
G2_DYNAMIC = {'bars_held_norm', 'current_pnl_pct', 'current_pnl_atr', 'max_favorable_pnl_pct',
              'drawdown_from_peak_pct', 'direction', 'entry_price_ratio',
              'lgbm_entry_conf', 'mfe_atr_ratio'}
g2_static  = [f for f in g2_feats if f not in G2_DYNAMIC]


def load_coin(sym):
    path = HOLDOUT_DIR / 'labeled' / f'{sym}_features_v3.parquet'
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    df = ensure_utc_index(df).sort_index()
    df = df[(df.index >= OOS_START) & (df.index < OOS_END)]
    return df if len(df) >= 30 else None


def derive_missing(df):
    """Hitung fitur yang tidak ada di parquet tapi bisa diturunkan."""
    if 'cvd_slope_h4_delta' not in df.columns and 'cvd_slope_h4' in df.columns:
        df = df.copy()
        df['cvd_slope_h4_delta'] = df['cvd_slope_h4'].diff(1)
    if 'ofi_h4_accel' not in df.columns and 'ofi_h4_delta' in df.columns:
        df = df.copy() if 'cvd_slope_h4_delta' not in df.attrs else df
        df['ofi_h4_accel'] = df['ofi_h4_delta'].diff(1)
    if 'rsi_h4_slope' not in df.columns and 'rsi_h4' in df.columns:
        df['rsi_h4_slope'] = df['rsi_h4'].diff(1)
    return df


def lstm_infer(df_arr, n, model, scaler, seq_len):
    """Jalankan LSTM inference, return proba array shape (n, 3)."""
    scaled = scaler.transform(df_arr)
    proba = np.zeros((n, 3), np.float32)
    proba[:, 1] = 1.0  # default NEUTRAL
    with torch.no_grad():
        for i in range(seq_len, n):
            seq = torch.tensor(scaled[i - seq_len:i], dtype=torch.float32).unsqueeze(0)
            out = model(seq)
            proba[i] = torch.softmax(out, dim=-1).numpy()[0]
    return proba  # idx 0=BEARISH/SHORT, 1=NEUTRAL, 2=BULLISH/LONG


# Preload BTC untuk btc_h1_return
print("Loading BTC for btc_h1_return...")
btc_df = load_coin('BTCUSDT')
btc_ret = btc_df['log_ret_1'].rename('btc_h1_return') if btc_df is not None else None

all_v1  = []
all_v2  = []
all_v2l = []   # V2 + LSTM V1 veto
all_v2l2 = []  # V2 + LSTM V2 veto (ic32_lstm_regime_v2, seq=72)

print("Evaluating coins...")
for sym in ALL_COINS:
    df = load_coin(sym)
    if df is None:
        continue

    df = derive_missing(df)

    # Tambah btc_h1_return (cross-coin)
    if btc_ret is not None and 'btc_h1_return' not in df.columns:
        df = df.join(btc_ret, how='left')
    if 'btc_h1_return' not in df.columns:
        df['btc_h1_return'] = 0.0

    n = len(df)
    regime = df['hmm_regime_enc'].fillna(-1).astype(int).values if 'hmm_regime_enc' in df.columns else np.zeros(n, int)
    h4_sh  = df['h4_swing_high'].values if 'h4_swing_high' in df.columns else np.full(n, np.nan)
    h4_sl  = df['h4_swing_low'].values  if 'h4_swing_low'  in df.columns else np.full(n, np.nan)

    # ---- V1 LGBM ----
    X1 = np.zeros((n, len(feats_lgbm)), np.float64)
    for i, c in enumerate(feats_lgbm):
        if c in df.columns:
            X1[:, i] = df[c].ffill().fillna(0).values
    p_v1 = lgbm_v1.predict_proba(X1)

    y1 = np.full(n, LM['FLAT'], np.int32)
    for i in range(n):
        st = regime[i]
        thr = PER_STATE_THR.get(st, [0.72, 0.72])
        if p_v1[i, 2] >= thr[0]:
            y1[i] = LM['LONG']
        elif p_v1[i, 0] >= thr[1]:
            y1[i] = LM['SHORT']

    # ---- V1 LSTM hard_consensus veto ----
    Xl = np.zeros((n, len(feats_lstm)), np.float64)
    for i, c in enumerate(feats_lstm):
        if c in df.columns:
            Xl[:, i] = df[c].ffill().fillna(0).values
    lstm_p = lstm_infer(Xl, n, lstm_v1, lstm_scaler, SEQ_LEN)
    # BEARISH=0 vetoes LONG; BULLISH=2 vetoes SHORT
    for i in range(n):
        if y1[i] == LM['LONG']  and lstm_p[i, 0] > LSTM_VETO_THR:
            y1[i] = LM['FLAT']
        elif y1[i] == LM['SHORT'] and lstm_p[i, 2] > LSTM_VETO_THR:
            y1[i] = LM['FLAT']

    # ---- V1 Guardian ----
    Xg1 = np.zeros((n, len(g1_static)), np.float64)
    for i, c in enumerate(g1_static):
        if c in df.columns:
            Xg1[:, i] = df[c].ffill().fillna(0).values

    r1 = simulate_trades_swing(
        y_pred=y1, close=df['close'].values, high=df['high'].values,
        low=df['low'].values, atr=df['atr_14_h1'].values,
        h4_swing_highs=h4_sh, h4_swing_lows=h4_sl,
        modal=MODAL_PER_TRADE, leverage=LEVERAGE_SIM[0],
        fee_per_side=FEE_PER_SIDE, slippage=SLIPPAGE_PER_SIDE,
        max_hold=MAX_HOLDING_BARS, min_rr=SWING_LABEL_MIN_RR,
        min_tp_atr=SWING_LABEL_MIN_TP, max_sl_atr=SWING_LABEL_MAX_SL,
        tp_fallback_atr=TP_SL_FALLBACK_TP, sl_fallback_atr=TP_SL_FALLBACK_SL,
        guardian_enabled=True, guardian_model=g1, guardian_scaler=g1_scl,
        X_guardian=Xg1, guardian_feat_cols=g1_feats, guardian_static_names=g1_static,
        guardian_exit_threshold=0.65, guardian_min_hold_bars=4)

    for t in r1.get('trades', []):
        bi = t['bar_in']
        ts = df.index[bi] if bi < len(df.index) else None
        all_v1.append({'coin': sym, 'ts': ts, 'dir': t['direction'],
                       'entry': round(t['entry'], 6), 'exit': round(t['exit'], 6),
                       'outcome': t['outcome'], 'pnl': t['net_pnl'],
                       'bars': t['bar_out'] - t['bar_in'], 'rr': t.get('rr', 0)})

    # ---- V2 ----
    X2 = np.zeros((n, len(feats_v2)), np.float64)
    for i, c in enumerate(feats_v2):
        if c in df.columns:
            X2[:, i] = df[c].ffill().fillna(0).values
    p_v2  = lgbm_v2.predict_proba(X2)
    p0_2  = p_v2[:, 0]; p2_2 = p_v2[:, 2]

    y2 = np.full(n, LM['FLAT'], np.int32)
    y2[p2_2 >= 0.75] = LM['LONG']
    for i in range(n):
        if y2[i] == LM['LONG']:
            continue
        thr_s = 0.80 if regime[i] == 3 else 0.70
        if p0_2[i] >= thr_s:
            y2[i] = LM['SHORT']
    lc = np.where(y2 == LM['LONG'], p2_2, np.where(y2 == LM['SHORT'], p0_2, 0.0))

    Xg2 = np.zeros((n, len(g2_static)), np.float64)
    for i, c in enumerate(g2_static):
        if c in df.columns:
            Xg2[:, i] = df[c].ffill().fillna(0).values

    r2 = simulate_trades_swing(
        y_pred=y2, close=df['close'].values, high=df['high'].values,
        low=df['low'].values, atr=df['atr_14_h1'].values,
        h4_swing_highs=h4_sh, h4_swing_lows=h4_sl,
        modal=MODAL_PER_TRADE, leverage=LEVERAGE_SIM[0],
        fee_per_side=FEE_PER_SIDE, slippage=SLIPPAGE_PER_SIDE,
        max_hold=MAX_HOLDING_BARS, min_rr=SWING_LABEL_MIN_RR,
        min_tp_atr=SWING_LABEL_MIN_TP, max_sl_atr=SWING_LABEL_MAX_SL,
        tp_fallback_atr=TP_SL_FALLBACK_TP, sl_fallback_atr=TP_SL_FALLBACK_SL,
        guardian_enabled=True, guardian_model=g2, guardian_scaler=g2_scl,
        X_guardian=Xg2, guardian_feat_cols=g2_feats, guardian_static_names=g2_static,
        lgbm_conf_arr=lc, guardian_exit_threshold=0.90, guardian_min_hold_bars=2)

    for t in r2.get('trades', []):
        bi = t['bar_in']
        ts = df.index[bi] if bi < len(df.index) else None
        all_v2.append({'coin': sym, 'ts': ts, 'dir': t['direction'],
                       'entry': round(t['entry'], 6), 'exit': round(t['exit'], 6),
                       'outcome': t['outcome'], 'pnl': t['net_pnl'],
                       'bars': t['bar_out'] - t['bar_in'], 'rr': t.get('rr', 0)})

    # ---- LSTM V2 inference (seq=72) ----
    Xl2 = np.zeros((n, len(feats_lstm_v2)), np.float64)
    for i, c in enumerate(feats_lstm_v2):
        if c in df.columns:
            Xl2[:, i] = df[c].ffill().fillna(0).values
    lstm_p2 = lstm_infer(Xl2, n, lstm_v2m, lstm_v2_scaler, SEQ_LEN_V2)

    # ---- V2 + LSTM veto (same LSTM as V1) ----
    y2l = y2.copy()
    for i in range(n):
        if y2l[i] == LM['LONG']  and lstm_p[i, 0] > LSTM_VETO_THR:
            y2l[i] = LM['FLAT']
        elif y2l[i] == LM['SHORT'] and lstm_p[i, 2] > LSTM_VETO_THR:
            y2l[i] = LM['FLAT']
    lc2l = np.where(y2l == LM['LONG'], p2_2, np.where(y2l == LM['SHORT'], p0_2, 0.0))

    r2l = simulate_trades_swing(
        y_pred=y2l, close=df['close'].values, high=df['high'].values,
        low=df['low'].values, atr=df['atr_14_h1'].values,
        h4_swing_highs=h4_sh, h4_swing_lows=h4_sl,
        modal=MODAL_PER_TRADE, leverage=LEVERAGE_SIM[0],
        fee_per_side=FEE_PER_SIDE, slippage=SLIPPAGE_PER_SIDE,
        max_hold=MAX_HOLDING_BARS, min_rr=SWING_LABEL_MIN_RR,
        min_tp_atr=SWING_LABEL_MIN_TP, max_sl_atr=SWING_LABEL_MAX_SL,
        tp_fallback_atr=TP_SL_FALLBACK_TP, sl_fallback_atr=TP_SL_FALLBACK_SL,
        guardian_enabled=True, guardian_model=g2, guardian_scaler=g2_scl,
        X_guardian=Xg2, guardian_feat_cols=g2_feats, guardian_static_names=g2_static,
        lgbm_conf_arr=lc2l, guardian_exit_threshold=0.90, guardian_min_hold_bars=2)

    for t in r2l.get('trades', []):
        bi = t['bar_in']
        ts = df.index[bi] if bi < len(df.index) else None
        all_v2l.append({'coin': sym, 'ts': ts, 'dir': t['direction'],
                        'entry': round(t['entry'], 6), 'exit': round(t['exit'], 6),
                        'outcome': t['outcome'], 'pnl': t['net_pnl'],
                        'bars': t['bar_out'] - t['bar_in'], 'rr': t.get('rr', 0)})

    # ---- V2 + LSTM V2 veto (ic32_lstm_regime_v2, seq=72) ----
    y2l2 = y2.copy()
    for i in range(n):
        if y2l2[i] == LM['LONG']  and lstm_p2[i, 0] > LSTM_VETO_THR:
            y2l2[i] = LM['FLAT']
        elif y2l2[i] == LM['SHORT'] and lstm_p2[i, 2] > LSTM_VETO_THR:
            y2l2[i] = LM['FLAT']
    lc2l2 = np.where(y2l2 == LM['LONG'], p2_2, np.where(y2l2 == LM['SHORT'], p0_2, 0.0))

    r2l2 = simulate_trades_swing(
        y_pred=y2l2, close=df['close'].values, high=df['high'].values,
        low=df['low'].values, atr=df['atr_14_h1'].values,
        h4_swing_highs=h4_sh, h4_swing_lows=h4_sl,
        modal=MODAL_PER_TRADE, leverage=LEVERAGE_SIM[0],
        fee_per_side=FEE_PER_SIDE, slippage=SLIPPAGE_PER_SIDE,
        max_hold=MAX_HOLDING_BARS, min_rr=SWING_LABEL_MIN_RR,
        min_tp_atr=SWING_LABEL_MIN_TP, max_sl_atr=SWING_LABEL_MAX_SL,
        tp_fallback_atr=TP_SL_FALLBACK_TP, sl_fallback_atr=TP_SL_FALLBACK_SL,
        guardian_enabled=True, guardian_model=g2, guardian_scaler=g2_scl,
        X_guardian=Xg2, guardian_feat_cols=g2_feats, guardian_static_names=g2_static,
        lgbm_conf_arr=lc2l2, guardian_exit_threshold=0.90, guardian_min_hold_bars=2)

    for t in r2l2.get('trades', []):
        bi = t['bar_in']
        ts = df.index[bi] if bi < len(df.index) else None
        all_v2l2.append({'coin': sym, 'ts': ts, 'dir': t['direction'],
                         'entry': round(t['entry'], 6), 'exit': round(t['exit'], 6),
                         'outcome': t['outcome'], 'pnl': t['net_pnl'],
                         'bars': t['bar_out'] - t['bar_in'], 'rr': t.get('rr', 0)})

    print(f"  {sym}: v1={len(r1.get('trades',[]))} v2={len(r2.get('trades',[]))} v2+lstm1={len(r2l.get('trades',[]))} v2+lstm2={len(r2l2.get('trades',[]))}")

df1   = pd.DataFrame(all_v1)
df2   = pd.DataFrame(all_v2)
df2l  = pd.DataFrame(all_v2l)
df2l2 = pd.DataFrame(all_v2l2)


def scorecard(df, label):
    t = len(df)
    if t == 0:
        print(f"  {label}: NO TRADES")
        return
    w   = (df['pnl'] > 0).sum()
    pnl = df['pnl'].sum()
    gw  = df[df['pnl'] > 0]['pnl'].sum()
    gl  = abs(df[df['pnl'] < 0]['pnl'].sum())
    pf  = gw / gl if gl > 1e-9 else 99.0
    wr  = w / t
    months   = (OOS_END - OOS_START).days / 30.44
    g_exits  = df['outcome'].str.contains('GUARDIAN').sum()
    sl_hits  = (df['outcome'] == 'LOSS').sum()
    to_hits  = df['outcome'].str.contains('TIMEOUT').sum()
    long_df  = df[df['dir'] == 'LONG']
    short_df = df[df['dir'] == 'SHORT']
    long_wr  = (long_df['pnl'] > 0).mean() if len(long_df) > 0 else 0
    short_wr = (short_df['pnl'] > 0).mean() if len(short_df) > 0 else 0
    avg_win  = df[df['pnl'] > 0]['pnl'].mean() if w > 0 else 0
    avg_loss = df[df['pnl'] < 0]['pnl'].mean() if (t - w) > 0 else 0
    avg_bars = df['bars'].mean()
    oc = df['outcome'].value_counts().to_dict()

    print(f"  {label}")
    print(f"  Period     : {OOS_START.date()} - {OOS_END.date()} (~{(OOS_END-OOS_START).days} hari / {months:.1f} bln)")
    print(f"  Trades     : {t:,}  ({t/months:.1f}/bln  |  LONG={len(long_df)} SHORT={len(short_df)})")
    print(f"  WR         : {wr*100:.2f}%   LONG {long_wr*100:.1f}%  SHORT {short_wr*100:.1f}%")
    print(f"  PF         : {pf:.3f}")
    print(f"  PnL Total  : ${pnl:.2f}  (${pnl/months:.2f}/bln  |  ${pnl/t:.4f}/trade)")
    print(f"  Avg Win    : ${avg_win:.4f}   Avg Loss: ${avg_loss:.4f}   Ratio: {abs(avg_win/avg_loss):.2f}x")
    print(f"  Avg Hold   : {avg_bars:.1f} bar")
    print(f"  Guardian%  : {g_exits/t*100:.1f}%   SL%: {sl_hits/t*100:.1f}%   Timeout%: {to_hits/t*100:.1f}%")
    print(f"  Outcomes:")
    for k, v in sorted(oc.items(), key=lambda x: -x[1]):
        print(f"    {k:<32} {v:>4}  ({v/t*100:.1f}%)")
    print()


SEP = '=' * 68
print(f"\n{SEP}")
scorecard(df1,   'V1 LIVE  [lgbm+LSTMv1+guardian_best]')
print(SEP)
scorecard(df2,   'V2 plain [rolling WF+HMM+guard_oof]')
print(SEP)
scorecard(df2l,  'V2+LSTMv1 veto [+lstm_momentum_v2_full, seq=72]')
print(SEP)
scorecard(df2l2, 'V2+LSTMv2 veto [+ic32_lstm_regime_v2, seq=72]')
print(SEP)

# Per-coin
print("\nPER-COIN (sorted by V2+LSTM PnL)")
print(f"  {'Coin':<16} | V1: Tr   WR%    PF    PnL  | V2: Tr   WR%    PF    PnL  | V2+LSTM: Tr   WR%    PF    PnL")
print("  " + "-" * 108)
rows = []
for sym in ALL_COINS:
    d1  = df1[df1['coin'] == sym]
    d2  = df2[df2['coin'] == sym]
    d2l = df2l[df2l['coin'] == sym]
    if len(d1) == 0 and len(d2) == 0:
        continue

    def cs(d):
        t = len(d)
        if t == 0: return 0, 0, 0, 0
        gw = d[d['pnl'] > 0]['pnl'].sum()
        gl = abs(d[d['pnl'] < 0]['pnl'].sum())
        return t, (d['pnl'] > 0).mean(), gw / gl if gl > 1e-9 else 99.0, d['pnl'].sum()

    t1, wr1, pf1, p1   = cs(d1)
    t2, wr2, pf2, p2   = cs(d2)
    t2l, wr2l, pf2l, p2l = cs(d2l)
    rows.append((sym, t1, wr1, pf1, p1, t2, wr2, pf2, p2, t2l, wr2l, pf2l, p2l))

for r in sorted(rows, key=lambda x: -x[12]):
    sym,t1,wr1,pf1,p1,t2,wr2,pf2,p2,t2l,wr2l,pf2l,p2l = r
    mk = '>>' if p2l > p1+2 else ('>' if p2l > p1 else '<')
    print(f"  {sym:<16} | {t1:>3}  {wr1*100:>4.1f}%  {pf1:>5.2f}  ${p1:>6.2f} | {t2:>3}  {wr2*100:>4.1f}%  {pf2:>5.2f}  ${p2:>6.2f} | {t2l:>4}  {wr2l*100:>4.1f}%  {pf2l:>5.2f}  ${p2l:>6.2f} {mk}")

# Save
df1['model'] = 'v1'; df2['model'] = 'v2'
df2l['model'] = 'v2+lstm1'; df2l2['model'] = 'v2+lstm2'
out = str(MODEL_DIR / 'runs' / 'ic32_regime_v2' / 'holdout_v1_vs_v2_trades.csv')
pd.concat([df1, df2, df2l, df2l2], ignore_index=True).to_csv(out, index=False)
print(f"\nSaved: {out}")
print(f"  v1={len(df1)}  v2={len(df2)}  v2+lstm1={len(df2l)}  v2+lstm2={len(df2l2)}")
