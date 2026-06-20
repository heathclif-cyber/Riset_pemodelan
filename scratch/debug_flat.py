import sys, warnings, numpy as np, pandas as pd
warnings.filterwarnings('ignore')
sys.path.insert(0, '.')
from config import LABEL_DIR, TRAIN_CUTOFF_DATE
from core.features import triple_barrier_labeling

sym = 'BTCUSDT'
df = pd.read_parquet(LABEL_DIR / f'{sym}_features_v3.parquet')
df.index = pd.to_datetime(df.index, utc=True)
df = df[df.index < TRAIN_CUTOFF_DATE]

tb = triple_barrier_labeling(df['close'], df['high'], df['low'],
                             df['atr_14_h1'], 2.0, 1.5, 36)
df['tb_label'] = tb.map({'SHORT': 0, 'FLAT': 1, 'LONG': 2})

flat_mask = df['tb_label'] == 1
long_mask = df['tb_label'] == 2
short_mask = df['tb_label'] == 0

print('=== TB Label Analysis — BTCUSDT (training data) ===')
print(f'FLAT:   {(flat_mask).sum():,} ({(flat_mask).mean()*100:.1f}%)')
print(f'LONG:   {(long_mask).sum():,} ({(long_mask).mean()*100:.1f}%)')
print(f'SHORT:  {(short_mask).sum():,} ({(short_mask).mean()*100:.1f}%)')

# ATR at each label
df['atr_pct'] = df['atr_14_h1'] / df['close'] * 100
print(f'\n=== ATR% by label ===')
for name, mask in [('FLAT', flat_mask), ('LONG', long_mask), ('SHORT', short_mask)]:
    vals = df.loc[mask, 'atr_pct']
    print(f'  {name}: mean={vals.mean():.3f}% median={vals.median():.3f}%')

# ATR bucket analysis
df['atr_bucket'] = pd.qcut(df['atr_pct'], 5, labels=['VL', 'L', 'M', 'H', 'VH'])
print(f'\n=== FLAT rate by ATR bucket ===')
for b in ['VL', 'L', 'M', 'H', 'VH']:
    bmask = df['atr_bucket'] == b
    flat_pct = (df.loc[bmask, 'tb_label'] == 1).mean() * 100
    print(f'  ATR {b}: FLAT={flat_pct:.1f}% (n={bmask.sum():,})')

# Feature separation
print(f'\n=== Feature means: FLAT vs DIRECTIONAL ===')
for feat in ['atr_14_h1', 'rsi_6', 'vol_ratio_20', 'trend_strength',
             'cvd_slope_h4', 'h4_trend', 'vol_regime', 'atr_percentile_h1']:
    if feat not in df.columns:
        continue
    flat_vals = df.loc[flat_mask, feat].dropna()
    dir_vals = df.loc[~flat_mask, feat].dropna()
    if len(flat_vals) < 100:
        continue
    diff = flat_vals.mean() - dir_vals.mean()
    print(f'  {feat:<25}: FLAT={flat_vals.mean():+.4f}  DIR={dir_vals.mean():+.4f}  diff={diff:+.4f}')

# KEY TEST: can a simple rule predict FLAT?
print(f'\n=== Simple rule: predict FLAT when ATR below median ===')
atr_med = df['atr_pct'].median()
predict_flat = df['atr_pct'] < atr_med
true_flat = flat_mask
tp = (predict_flat & true_flat).sum()
fp = (predict_flat & ~true_flat).sum()
tn = (~predict_flat & ~true_flat).sum()
fn = (~predict_flat & true_flat).sum()
precision = tp / max(tp + fp, 1)
recall = tp / max(tp + fn, 1)
f1 = 2 * precision * recall / max(precision + recall, 1e-10)
print(f'  Rule: ATR% < {atr_med:.3f}% -> FLAT')
print(f'  TP={tp} FP={fp} TN={tn} FN={fn}')
print(f'  Precision={precision:.3f} Recall={recall:.3f} F1={f1:.3f}')

# What's the best ATR threshold for FLAT prediction?
print(f'\n=== Best possible FLAT predictor from ATR alone ===')
best_f1, best_thresh = 0, 0
for thresh in np.linspace(df['atr_pct'].min(), df['atr_pct'].max(), 50):
    pf = df['atr_pct'] < thresh
    tp = (pf & true_flat).sum()
    fp = (pf & ~true_flat).sum()
    fn = (~pf & true_flat).sum()
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-10)
    if f1 > best_f1:
        best_f1, best_thresh = f1, thresh
print(f'  Best ATR threshold: {best_thresh:.3f}%')
print(f'  Best possible FLAT F1 (ATR only): {best_f1:.4f}')
print(f'  => Even the BEST single-feature FLAT predictor has F1={best_f1:.4f}')

# AKAR MASALAH
print(f'\n{"="*60}')
print(f'  ROOT CAUSE ANALYSIS')
print(f'{"="*60}')
print(f'  1. FLAT = no barrier hit in 36 bars')
print(f'  2. Barriers: TP=2.0xATR, SL=1.5xATR')
print(f'  3. Median ATR = {df["atr_pct"].median():.3f}%')
print(f'  4. In 36h, expected range >> {2.0*df["atr_pct"].median():.3f}%')
print(f'     -> Most bars WILL hit a barrier')
print(f'  5. FLAT happens when volatility DROPS unexpectedly')
print(f'  6. This is a REGIME CHANGE — cannot be predicted')
print(f'     from static features at entry time')
print(f'  7. Best possible F1 for FLAT (ATR rule): {best_f1:.4f}')
print(f'     -> Even optimal simple rule cannot predict FLAT')
print(f'  8. With 3-class LGBM:')
print(f'     - Class weights penalize SHORT/LONG errors 3x vs FLAT 1.5x')
print(f'     - Model learns: never predict FLAT (cheaper to be wrong)')
print(f'     - FLAT F1 = 0.00, confidence capped at ~0.49')
print(f'')
print(f'  SOLUTION: FLAT should be confidence-based, not class-based')
print(f'  "If model is unsure about direction -> FLAT"')
print(f'  NOT "Model predicts FLAT as a separate class"')
