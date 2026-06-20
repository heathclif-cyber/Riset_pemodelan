"""Debug: lihat prediksi per-bar June 19 WITA untuk semua koin."""
import json, sys, warnings, numpy as np, pandas as pd
from pathlib import Path
warnings.filterwarnings('ignore')
sys.path.insert(0, str(Path(__file__).parent.parent))
import pipeline.backtest_utils as btu
from pipeline.backtest_utils import hierarchical_predict
from core.models import load_lstm
from core.utils import ensure_utc_index
from config import HOLDOUT_DIR, MODEL_DIR, LABEL_MAP, ALL_COINS, GUARDIAN_DYNAMIC_FEATURES
import joblib

with open(MODEL_DIR / 'inference_config.json') as f:
    cfg = json.load(f)
cascade = cfg['cascade']
btu.CONFIDENCE_THRESHOLD_ENTRY         = float(cascade['confidence_threshold_entry'])
btu.LSTM_ADJUST_AGREE_BOOST            = float(cascade['lstm_adjust_agree_boost'])
btu.LSTM_ADJUST_NEUTRAL_PEN            = float(cascade.get('lstm_adjust_neutral_pen', 0.0))
btu.LSTM_ADJUST_OPPOSITE_PEN           = float(cascade['lstm_adjust_opposite_pen'])
btu.LSTM_DIRECTIONAL_REVIEW_THRESHOLD  = float(cascade['lstm_directional_review_threshold'])
btu.LSTM_FLAT_REVIEW_ENABLED           = True
btu.LSTM_CONFIRMATION_ENABLED          = True
btu.REGIME_AWARE_ALIGNMENT             = True
btu.SMART_ENTRY_MODE                   = 'disabled'
btu.MOMENTUM_DYNAMIC_THRESHOLD_ENABLED = False
btu.TREND_DYNAMIC_THRESHOLD_ENABLED    = False
btu.LSTM_STANDALONE_ENABLED            = False

with open(MODEL_DIR / 'runs' / 'ic32_regime_v1' / 'b_dir_combined_frozen.json') as f:
    frozen = json.load(f)
hmm_cfg = {int(k): (float(v[0]), float(v[1])) for k, v in frozen['per_state_thresholds'].items()}

with open(MODEL_DIR / 'feature_cols_ic32_regime.json') as f:
    feat_cols = json.load(f)
with open(MODEL_DIR / 'feature_cols_lstm_temporal.json') as f:
    lstm_feat_cols = json.load(f)

lgbm       = joblib.load(MODEL_DIR / 'runs' / 'ic32_regime_v1' / 'lgbm.pkl')
lstm_model = load_lstm(MODEL_DIR / 'lstm_best.pt', device='cpu')
lst_scaler = joblib.load(MODEL_DIR / 'lstm_scaler.pkl')

JUN19_START = pd.Timestamp('2026-06-18 16:00', tz='UTC')
JUN19_END   = pd.Timestamp('2026-06-19 16:00', tz='UTC')
CONF_THR    = 0.59

all_signals = []

for sym in ALL_COINS:
    p  = HOLDOUT_DIR / 'labeled' / f'{sym}_features_v3.parquet'
    rp = HOLDOUT_DIR / 'labeled' / f'{sym}_regime_h1.parquet'
    if not p.exists():
        continue
    df = pd.read_parquet(p)
    df = ensure_utc_index(df).sort_index()
    if rp.exists():
        reg = pd.read_parquet(rp)
        for col in ['hmm_regime_enc','hmm_regime']:
            if col in df.columns: df = df.drop(columns=[col])
        rc = [c for c in ['hmm_regime_enc','hmm_regime'] if c in reg.columns]
        df = df.join(reg[rc], how='left')
    df['hmm_regime_enc'] = df['hmm_regime_enc'].fillna(1).astype('int32')
    df['hmm_regime']     = df.get('hmm_regime', pd.Series('RANGING_LOW_VOL', index=df.index)).fillna('RANGING_LOW_VOL')

    mask = df['label'].astype(str).isin(LABEL_MAP)
    df   = df[mask].copy()
    n    = len(df)
    if n < 10:
        continue

    X = np.zeros((n, len(feat_cols)), dtype=np.float64)
    for idx, col in enumerate(feat_cols):
        if col in df.columns:
            X[:, idx] = df[col].ffill().fillna(0).values.astype(np.float64)

    hmm_enc   = df['hmm_regime_enc'].values.astype(np.int32)
    default_tl, default_ts = hmm_cfg[-1]
    tl_arr = np.full(n, default_tl, dtype=np.float64)
    ts_arr = np.full(n, default_ts, dtype=np.float64)
    for state, (tl, ts) in hmm_cfg.items():
        if state == -1: continue
        m = hmm_enc == state; tl_arr[m] = tl; ts_arr[m] = ts

    y_pred, confidence = hierarchical_predict(
        None, lgbm, lstm_model, lst_scaler, X, feat_cols, [], df,
        model_dir=MODEL_DIR / 'runs' / 'ic32_regime_v1',
        lstm_feat_cols=lstm_feat_cols, per_bar_thr_long=tl_arr, per_bar_thr_short=ts_arr,
    )
    below = (y_pred != 1) & (confidence < CONF_THR)
    y_pred[below] = 1

    mask_jun19 = (df.index >= JUN19_START) & (df.index < JUN19_END)
    for i in np.where(mask_jun19)[0]:
        pname = {0: 'SHORT', 1: 'FLAT', 2: 'LONG'}.get(int(y_pred[i]), '?')
        all_signals.append({
            'symbol':  sym,
            'ts_utc':  str(df.index[i])[:16],
            'pred':    pname,
            'conf':    round(float(confidence[i]), 4),
            'hmm_enc': int(hmm_enc[i]),
            'hmm_reg': str(df['hmm_regime'].iloc[i]),
            'close':   round(float(df['close'].iloc[i]), 4),
            'label':   str(df['label'].iloc[i]),
        })

entries = [s for s in all_signals if s['pred'] != 'FLAT']
print(f"\nTotal bars Jun 19 WITA (all coins): {len(all_signals)}")
print(f"Entry signals (non-FLAT): {len(entries)}")
print()

if entries:
    print("ENTRY SIGNALS — 19 Juni 2026 WITA:")
    print(f"  {'Symbol':<16} {'UTC Time':>16} {'Dir':>5} {'Conf':>5} {'Regime':<20} {'Close':>12}")
    print("  " + "-" * 80)
    for s in sorted(entries, key=lambda x: x['ts_utc']):
        ts_wita = (pd.Timestamp(s['ts_utc'], tz='UTC') + pd.Timedelta(hours=8)).strftime('%H:%M')
        print(f"  {s['symbol']:<16} {s['ts_utc']:>16}({ts_wita} WITA) {s['pred']:>5} {s['conf']:>5.3f} {s['hmm_reg']:<20} {s['close']:>12.4f}")
else:
    print("TIDAK ADA entry signal pada 19 Juni 2026 WITA.")
    print()
    print("Sample FLAT predictions (conf > 0.55) — sinyal yang hampir masuk:")
    near_miss = [s for s in all_signals if s['conf'] > 0.55]
    for s in sorted(near_miss, key=lambda x: -x['conf'])[:10]:
        ts_wita = (pd.Timestamp(s['ts_utc'], tz='UTC') + pd.Timedelta(hours=8)).strftime('%H:%M')
        print(f"  {s['symbol']:<16} {s['ts_utc']:>16}({ts_wita} WITA) FLAT {s['conf']:>5.3f} {s['hmm_reg']}")

# Simpan ke JSON
with open(MODEL_DIR / 'runs' / 'ic32_regime_v1' / 'holdout_diag_jun19_wita.json', 'w') as f:
    import datetime as dt
    json.dump({
        'meta': {
            'date_wita': '2026-06-19',
            'entry_window_utc': f'{JUN19_START} -> {JUN19_END}',
            'generated_at': dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        },
        'total_bars': len(all_signals),
        'entry_signals': len(entries),
        'signals': all_signals,
        'entries': entries,
    }, f, indent=2)
print(f"\nOutput disimpan ke models/runs/ic32_regime_v1/holdout_diag_jun19_wita.json")
