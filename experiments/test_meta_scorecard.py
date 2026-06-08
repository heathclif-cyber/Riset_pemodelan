"""Scorecard: LSTM meta ensemble with/without h4_trend."""
import sys, json, joblib, torch, torch.nn as nn, numpy as np, pandas as pd
from pathlib import Path
ROOT = Path(__file__).parent.parent; sys.path.insert(0, str(ROOT))
from sklearn.preprocessing import RobustScaler

from config import MODEL_DIR, HOLDOUT_DIR, TRAINING_COINS, LABEL_MAP
from config import (
    MODAL_PER_TRADE, LEVERAGE_SIM, FEE_PER_SIDE, SLIPPAGE_PER_SIDE,
    MAX_HOLDING_BARS, SWING_LABEL_MIN_RR, SWING_LABEL_MIN_TP, SWING_LABEL_MAX_SL,
    TP_SL_FALLBACK_TP, TP_SL_FALLBACK_SL, GUARDIAN_EXIT_THRESHOLD, GUARDIAN_DYNAMIC_FEATURES,
)
from pipeline.backtest_utils import compute_guardian_static_array, hierarchical_predict
from core.evaluator import simulate_trades_swing
import pipeline.backtest_utils as btu

# Model
class AttnPool(nn.Module):
    def __init__(self, h): super().__init__(); self.attn = nn.Linear(h, 1)
    def forward(self, x): w = torch.softmax(self.attn(x), dim=1); return (x * w).sum(dim=1)

class MetaLSTM(nn.Module):
    def __init__(self, nf, h, nl, do):
        super().__init__()
        self.lstm = nn.LSTM(nf, h, nl, batch_first=True, dropout=do if nl > 1 else 0)
        self.attention = AttnPool(h); self.dropout = nn.Dropout(do); self.classifier = nn.Linear(h, 1)
    def forward(self, x):
        o, _ = self.lstm(x); return torch.sigmoid(self.classifier(self.dropout(self.attention(o)))).squeeze(-1)

def fs(X): n, s, f = X.shape; scl = RobustScaler(); scl.fit(X.reshape(-1, f)); return scl
def sx(X, scl): n, s, f = X.shape; return scl.transform(X.reshape(-1, f)).reshape(n, s, f).astype(np.float32)

# Load
btu.SMART_ENTRY_MODE = 'disabled'; btu.LSTM_CONFIRMATION_ENABLED = False; btu.LSTM_FLAT_REVIEW_ENABLED = False
lgbm = joblib.load(MODEL_DIR / 'lgbm_baseline.pkl')
lstm_feat_cols = json.load(open(MODEL_DIR / 'feature_cols_lstm_temporal.json'))
guardian = joblib.load(MODEL_DIR / 'guardian_best.pkl'); g_scaler = joblib.load(MODEL_DIR / 'guardian_scaler.pkl')
g_feats = json.load(open(MODEL_DIR / 'guardian_feature_cols.json'))
g_static = [c for c in g_feats if c not in set(GUARDIAN_DYNAMIC_FEATURES)]
with open('data/meta_labels_v2/meta_labels_v2_meta.json') as f: mi = json.load(f)
META_FEATS = mi['features']; SEQ_LEN = mi['seq_len']
meta_model = MetaLSTM(len(META_FEATS), 96, 2, 0.40)
meta_model.load_state_dict(torch.load(MODEL_DIR / 'runs/lstm_meta_v1/lstm_meta.pt', map_location='cpu'))
meta_model.eval(); meta_scaler = joblib.load(MODEL_DIR / 'runs/lstm_meta_v1/lstm_meta_scaler.pkl')

# Data
NON_FEATURE_COLS = {'label', 'h4_swing_high', 'h4_swing_low'}
all_coin_data = {}
for coin in TRAINING_COINS[:5]:
    path = HOLDOUT_DIR / 'labeled' / f'{coin}_features_v3.parquet'
    if not path.exists(): continue
    df = pd.read_parquet(path).sort_index()
    for pp in [HOLDOUT_DIR / 'labeled' / f'{coin}_hmm_probs.parquet',
               HOLDOUT_DIR / 'labeled' / f'{coin}_regime_h1.parquet']:
        if pp.exists():
            extra = pd.read_parquet(pp).sort_index()
            for c in extra.columns:
                if c in df.columns: df = df.drop(columns=[c])
            df = df.join(extra, how='left')
            if 'hmm_regime_enc' in df.columns: df['hmm_regime_enc'] = df['hmm_regime_enc'].fillna(1).astype('int32')
    for c in META_FEATS:
        if c not in df.columns: df[c] = 0.0
    mask = df['label'].astype(str).isin(LABEL_MAP); df = df[mask].copy()
    if len(df) >= SEQ_LEN + 50: all_coin_data[coin] = df

def pred_meta(df, bar_idx):
    start = bar_idx - SEQ_LEN + 1
    if start < 0: return 0.5
    seq = np.zeros((1, SEQ_LEN, len(META_FEATS)), dtype=np.float32)
    for j, col in enumerate(META_FEATS):
        seq[0, :, j] = df[col].iloc[start:bar_idx+1].ffill().fillna(0).values
    with torch.no_grad(): return meta_model(torch.from_numpy(sx(seq, meta_scaler))).item()

def run(label, use_meta, meta_s, use_trend):
    all_trades = []
    for coin, df in all_coin_data.items():
        n = len(df); X = np.zeros((n, len(lstm_feat_cols)))
        for i, col in enumerate(lstm_feat_cols):
            if col in df.columns: X[:, i] = df[col].ffill().fillna(0).values
        yp, cf = hierarchical_predict(None, lgbm, None, None, X, lstm_feat_cols, [], df,
                                       trend_alignment_enabled=use_trend)
        if use_meta:
            for i in range(n):
                if yp[i] != 1 and cf[i] >= 0.59:
                    mp = pred_meta(df, i)
                    mult = float(np.clip(0.70 + meta_s * mp, 0.7, 1.3))
                    cf[i] = float(np.clip(cf[i] * mult, 0, 1))
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
    wr = len(wins) / n * 100 if n else 0; pnl = sum(t.get('net_pnl', 0) for t in all_trades)
    lt = [t for t in all_trades if t.get('direction') == 'LONG']
    st = [t for t in all_trades if t.get('direction') == 'SHORT']
    lwr = len([t for t in lt if t.get('net_pnl', 0) > 0]) / len(lt) * 100 if lt else 0
    swr = len([t for t in st if t.get('net_pnl', 0) > 0]) / len(st) * 100 if st else 0
    gw = sum(t.get('net_pnl', 0) for t in wins)
    gl = abs(sum(t.get('net_pnl', 0) for t in all_trades if t.get('net_pnl', 0) <= 0))
    pf = gw / gl if gl > 0 else float('inf')
    print(f'{label:<42} {n:>5} {wr:>5.1f}% {lwr:>5.1f}% {swr:>5.1f}% {pnl:>8.1f} {pf:>6.2f}')

print('=== SCORECARD: LSTM META ENSEMBLE (AUC=0.594, genuine OOF) ===')
print('Config                                       Trades   WR% L_WR% S_WR%      PnL     PF')
print('-' * 82)

# No trend
run('BASELINE (no trend, no meta)          ', False, 0, False)
run('META s=0.3 (no trend)                 ', True, 0.3, False)
run('META s=0.5 (no trend)                 ', True, 0.5, False)
run('META s=0.7 (no trend)                 ', True, 0.7, False)

# With trend
run('BASELINE (h4_trend, no meta)          ', False, 0, True)
run('META s=0.3 (h4_trend)                 ', True, 0.3, True)
run('META s=0.5 (h4_trend)                 ', True, 0.5, True)
run('META s=0.7 (h4_trend)                 ', True, 0.7, True)
