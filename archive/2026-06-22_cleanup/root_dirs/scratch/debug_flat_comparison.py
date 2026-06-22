import sys, warnings, numpy as np, pandas as pd
warnings.filterwarnings('ignore')
sys.path.insert(0, '.')
from config import LABEL_DIR, TRAIN_CUTOFF_DATE
from core.features import triple_barrier_labeling

sym = 'BTCUSDT'
df = pd.read_parquet(LABEL_DIR / f'{sym}_features_v3.parquet')
df.index = pd.to_datetime(df.index, utc=True)
df = df[df.index < TRAIN_CUTOFF_DATE]

# Generate TB labels
tb = triple_barrier_labeling(df['close'], df['high'], df['low'],
                             df['atr_14_h1'], 2.0, 1.5, 36)
df['tb_label'] = tb.map({'SHORT': 0, 'FLAT': 1, 'LONG': 2})

# Swing labels already exist in 'label' column
# label: SHORT/FLAT/LONG from swing-based labeling

print('=' * 70)
print('  FLAT COMPARISON: Triple Barrier vs Swing-Based Labeling')
print('=' * 70)

# Overall distribution
print(f'\n=== Label Distribution ===')
for name, col in [('SWING', 'label'), ('TB', 'tb_label')]:
    if col == 'tb_label':
        dist = df[col].value_counts()
    else:
        dist = df[col].value_counts()
    n = len(df)
    print(f'  {name}: LONG={dist.get("LONG", dist.get(2, 0))/n*100:.1f}%  '
          f'SHORT={dist.get("SHORT", dist.get(0, 0))/n*100:.1f}%  '
          f'FLAT={dist.get("FLAT", dist.get(1, 0))/n*100:.1f}%')

# KEY ANALYSIS: what features correlate with FLAT in each labeling?
swing_flat = df['label'] == 'FLAT'
tb_flat = df['tb_label'] == 1

print(f'\n=== FLAT vs DIRECTIONAL: Feature Separation ===')
print(f'{"Feature":<25} {"SWING FLAT":>10} {"SWING DIR":>10} {"DIFF":>8} | {"TB FLAT":>10} {"TB DIR":>10} {"DIFF":>8}')
print('-' * 75)

features_to_check = [
    'atr_14_h1', 'atr_percentile_h1', 'rsi_6', 'rsi_h4',
    'trend_strength', 'h4_trend', 'vol_regime', 'vol_ratio_20',
    'cvd_slope_h4', 'ofi_h4_delta', 'price_in_range',
    'stochrsi_k', 'ema_21_slope_h4', 'wyckoff_phase'
]

for feat in features_to_check:
    if feat not in df.columns:
        continue

    # Swing
    s_flat = df.loc[swing_flat, feat].dropna()
    s_dir = df.loc[~swing_flat, feat].dropna()
    s_diff = s_flat.mean() - s_dir.mean()

    # TB
    t_flat = df.loc[tb_flat, feat].dropna()
    t_dir = df.loc[~tb_flat, feat].dropna()
    t_diff = t_flat.mean() - t_dir.mean()

    print(f'{feat:<25} {s_flat.mean():>+10.4f} {s_dir.mean():>+10.4f} {s_diff:>+8.4f} | {t_flat.mean():>+10.4f} {t_dir.mean():>+10.4f} {t_diff:>+8.4f}')

# DECISIVE TEST: what's the best possible FLAT F1 for each labeling?
print(f'\n=== Best Possible FLAT F1 (optimal single-feature rule) ===')
for label_name, flat_mask in [('SWING', swing_flat), ('TB', tb_flat)]:
    best_f1, best_feat, best_thresh, best_dir = 0, '', 0, ''
    for feat in features_to_check:
        if feat not in df.columns:
            continue
        vals = df[feat].dropna()
        if len(vals) < 100:
            continue
        # Try both directions: feature < threshold -> FLAT, feature > threshold -> FLAT
        for direction in ['<', '>']:
            for thresh in np.linspace(vals.min(), vals.max(), 50):
                if direction == '<':
                    pf = vals < thresh
                else:
                    pf = vals > thresh
                pf = pf.reindex(df.index, fill_value=False)
                tp = (pf & flat_mask).sum()
                fp = (pf & ~flat_mask).sum()
                fn = (~pf & flat_mask).sum()
                prec = tp / max(tp + fp, 1)
                rec = tp / max(tp + fn, 1)
                f1 = 2 * prec * rec / max(prec + rec, 1e-10)
                if f1 > best_f1:
                    best_f1 = f1
                    best_feat = feat
                    best_thresh = thresh
                    best_dir = direction
    print(f'  {label_name}: best F1={best_f1:.4f} (feat={best_feat}, dir={best_dir}, thresh={best_thresh:.4f})')

# THE FUNDAMENTAL DIFFERENCE
print(f'\n{"="*70}')
print(f'  THE FUNDAMENTAL DIFFERENCE')
print(f'{"="*70}')
print(f'')
print(f'  SWING FLAT: "Price is ranging between H4 swing levels RIGHT NOW"')
print(f'    -> Features CAN detect ranging: h4_trend~0, trend_strength low,')
print(f'       price_in_range mid, stochrsi neutral, vol_regime low')
print(f'    -> Structural property of CURRENT market')
print(f'    -> LEARNABLE')
print(f'')
print(f'  TB FLAT: "In 36 hours, vol will drop enough that no barrier is hit"')
print(f'    -> Features measure NOW, not FUTURE vol change')
print(f'    -> FLAT happens when market goes from volatile -> calm')
print(f'    -> This is a REGIME TRANSITION, not a structural property')
print(f'    -> NOT LEARNABLE from snapshot features')
print(f'')
print(f'  SWING = structure-based (support/resistance = persistent)')
print(f'  TB    = path-based (volatility-dependent, random walk component)')
