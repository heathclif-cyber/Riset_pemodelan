"""IC test: which features have temporal predictability for LSTM."""
import sys, numpy as np, pandas as pd
from pathlib import Path
ROOT = Path(__file__).parent.parent; sys.path.insert(0, str(ROOT))
from config import LABEL_DIR, TRAINING_COINS
from scipy import stats

LSTM_FEATS = ['ofi_z_score','ofi_acceleration','cvd_momentum_adv','absorption_z',
    'volume_delta','vol_ratio_20','log_ret_1','log_ret_5','log_ret_20','rsi_6']

IC38_FEATS = ['cvd_momentum_adv','rsi_h4','ema_21_slope_h4','whale_retail_divergence',
    'rsi_6','price_vs_ema_50_h4','log_ret_5','Sell_Liq','log_ret_20','Buy_Liq',
    'ema_50_slope_h4','h4_trend','vol_price_confirm','dist_liq_50x_long',
    'trend_strength','stochrsi_d','stochrsi_k','trend_accel_4h','dist_liq_50x_short',
    'ofi_h4_delta','dist_liq_20x_long','cvd_slope_h4','volume_delta','long_short_ratio',
    'dist_liq_20x_short','log_ret_1','swing_momentum','rsi_slope_h4','ofi_raw',
    'open_interest','cvd_div_h4','funding_rate','cvd','atr_zscore_20d',
    'atr_percentile_h1','atr_percent_h4','ofi_z_score','rsi_divergence']

# Features NOT yet in LSTM
NEW_FEATS = list(dict.fromkeys([f for f in IC38_FEATS if f not in LSTM_FEATS]))

def test_feature(feat, coins):
    ics, deltas, cross_vars = [], [], []
    for coin in coins:
        fp = LABEL_DIR / f'{coin}_features_v3.parquet'
        lp = LABEL_DIR / f'{coin}_momentum_v2_labels.parquet'
        if not fp.exists() or not lp.exists(): continue
        df = pd.read_parquet(fp).sort_index()
        lbl = pd.read_parquet(lp).sort_index()
        if feat not in df.columns: continue
        x = df[feat].ffill().fillna(0)
        x_delta = x.diff(16)  # Same window LSTM sees
        y = lbl['momentum_v2_label'].map({0: -1, 1: 0, 2: 1})
        mask = ~(x_delta.isna() | y.isna())
        if mask.sum() < 100: continue
        ic_val, _ = stats.spearmanr(x_delta[mask], y[mask])
        ics.append(ic_val)
        deltas.append(abs(x_delta[mask]).mean())
        x_roll_std = x.rolling(16).std()
        cross_vars.append(x_roll_std[mask].mean())
    if not ics: return None
    return (np.mean(ics), np.std(ics), np.mean(deltas), np.mean(cross_vars),
            sum(1 for v in ics if v * np.mean(ics) > 0) / len(ics) * 100)

coins = TRAINING_COINS[:5]
print('=== LSTM TEMPORAL IC TEST ===')
print('IC(feature_delta_16bar, momentum_label). High delta + cross-var = good for LSTM.')

# Test new features
print(f'\n--- NEW features ({len(NEW_FEATS)} candidates) ---')
print(f'{"Feature":<30} {"IC_delta":>8} {"|Delta|":>10} {"SeqStd":>8} {"Sign%":>7} {"Verdict":>10}')
results = []
for feat in NEW_FEATS:
    r = test_feature(feat, coins)
    if r is None: continue
    mean_ic, std_ic, delta, cv, sign = r
    abs_ic = abs(mean_ic)
    if abs_ic >= 0.02 and sign >= 60:
        verdict = 'KEEP'
    elif abs_ic >= 0.01:
        verdict = 'WEAK'
    else:
        verdict = 'DROP'
    results.append((feat, mean_ic, delta, cv, verdict))
    print(f'{feat:<30} {mean_ic:>+8.4f} {delta:>10.4f} {cv:>8.4f} {sign:>6.0f}% {verdict:>10}')

# Also check current LSTM features
print(f'\n--- Current LSTM features (baseline) ---')
for feat in LSTM_FEATS:
    r = test_feature(feat, coins)
    if r is None: continue
    print(f'  {feat:<30} IC_delta={r[0]:+.4f}  |Delta|={r[2]:.4f}  SeqStd={r[3]:.4f}')

keep = [r for r in results if r[4] == 'KEEP']
print(f'\nKEEP: {len(keep)}/{len(results)} new features')
print(f'Current LSTM: {len(LSTM_FEATS)} features')
print(f'Potential upgrade: {len(LSTM_FEATS) + len(keep)} features')
if keep:
    print(f'\nTop KEEP features (sorted by |IC_delta|):')
    keep.sort(key=lambda x: abs(x[1]), reverse=True)
    for feat, ic, delta, cv, _ in keep[:10]:
        print(f'  {feat:<30} IC={ic:+.4f}  |Delta|={delta:.4f}  SeqStd={cv:.4f}')
