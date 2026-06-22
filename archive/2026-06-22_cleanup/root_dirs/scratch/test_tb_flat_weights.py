"""Test: can TB 3-class learn FLAT with proper class weights?"""
import sys, warnings, numpy as np, pandas as pd
warnings.filterwarnings('ignore')
sys.path.insert(0, '.')
import lightgbm as lgb
from config import *
from core.features import triple_barrier_labeling
from pipeline.shared import build_purged_folds
from sklearn.metrics import f1_score

sym = 'BTCUSDT'
df = pd.read_parquet(LABEL_DIR / f'{sym}_features_v3.parquet')
df.index = pd.to_datetime(df.index, utc=True)
df = df[df.index < TRAIN_CUTOFF_DATE]

# TB labels
tb = triple_barrier_labeling(df['close'], df['high'], df['low'],
                             df['atr_14_h1'], 2.0, 1.5, 36)
df['tb_label'] = tb.map({'SHORT': 0, 'FLAT': 1, 'LONG': 2})
df = df[df['tb_label'].notna()]

# Use ALL features (same as v1)
META = {"label", "coin", "symbol", "h4_swing_high", "h4_swing_low",
        "hmm_regime", "hmm_regime_enc", "tb_label"}
from config import FEATURE_COLS_V3
features = [c for c in FEATURE_COLS_V3 if c in df.columns and c not in META]
print(f'Features: {len(features)}, Samples: {len(df):,}')

X = df[features].ffill().fillna(0)
y = df['tb_label'].astype(np.int32).values

# Label distribution
for i, name in enumerate(['SHORT', 'FLAT', 'LONG']):
    print(f'  {name}: {(y==i).sum():,} ({(y==i).mean()*100:.1f}%)')

folds = build_purged_folds(X.index, N_FOLDS, PURGE_GAP_BARS)

# Test 3 different class_weight strategies
weight_configs = [
    ("BALANCED", "balanced"),
    ("NONE (1:1:1)", None),
    ("SWING_WEIGHTS", {0: 3.0, 1: 1.5, 2: 3.0}),
    ("TB_PROPORTIONAL", {0: 1.0, 1: 3.0, 2: 1.0}),  # FLAT dilawan 3x karena minority
    ("AUTO_COMPUTED", None),  # will compute balanced weights manually
]

LGBM_PARAMS_3C = {
    "objective": "multiclass", "num_class": 3,
    "n_estimators": 500, "learning_rate": 0.03,
    "max_depth": 5, "num_leaves": 31, "min_child_samples": 50,
    "subsample": 0.8, "colsample_bytree": 0.7,
    "verbose": -1, "n_jobs": -1, "random_state": 42,
}

print(f'\n{"="*80}')
print(f'  TEST: TB 3-Class FLAT F1 with Different Class Weights')
print(f'{"="*80}')
print(f'{"Weight Config":<25} {"SHORT F1":>10} {"FLAT F1":>10} {"LONG F1":>10} {"MACRO F1":>10} {"Iter":>8}')
print('-' * 70)

for wname, wval in weight_configs:
    all_f1_short, all_f1_flat, all_f1_long, all_f1_macro, all_iter = [], [], [], [], []

    for fold, (tr_idx, val_idx) in enumerate(folds, 1):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]

        params = LGBM_PARAMS_3C.copy()
        if wname == "AUTO_COMPUTED":
            # sklearn-style balanced: inversely proportional to class frequency
            from sklearn.utils.class_weight import compute_class_weight
            cw = compute_class_weight('balanced', classes=np.array([0,1,2]), y=y_tr)
            params["class_weight"] = {0: cw[0], 1: cw[1], 2: cw[2]}
        elif wval is not None:
            params["class_weight"] = wval

        model = lgb.LGBMClassifier(**params)
        model.fit(X_tr, y_tr,
                  eval_set=[(X_val, y_val)],
                  callbacks=[lgb.early_stopping(50, verbose=False),
                             lgb.log_evaluation(period=-1)])

        y_pred = model.predict(X_val)
        f1_per = f1_score(y_val, y_pred, average=None, labels=[0, 1, 2], zero_division=0)
        f1_macro = f1_score(y_val, y_pred, average='macro', zero_division=0)

        all_f1_short.append(f1_per[0])
        all_f1_flat.append(f1_per[1])
        all_f1_long.append(f1_per[2])
        all_f1_macro.append(f1_macro)
        all_iter.append(model.best_iteration_)

    print(f'{wname:<25} {np.mean(all_f1_short):>10.4f} {np.mean(all_f1_flat):>10.4f} '
          f'{np.mean(all_f1_long):>10.4f} {np.mean(all_f1_macro):>10.4f} {int(np.mean(all_iter)):>8}')

# Also test: BINARY with confidence threshold as FLAT
print(f'\n{"="*80}')
print(f'  BINARY APPROACH: SHORT vs LONG, FLAT = confidence < threshold')
print(f'{"="*80}')
# Drop FLAT samples
df_bin = df[df['tb_label'] != 1].copy()
df_bin['tb_binary'] = (df_bin['tb_label'] == 2).astype(np.int32)
X_bin = df_bin[features].ffill().fillna(0)
y_bin = df_bin['tb_binary'].values
folds_bin = build_purged_folds(X_bin.index, N_FOLDS, PURGE_GAP_BARS)

bin_params = {**LGBM_PARAMS_3C, "objective": "binary", "num_class": 1}
all_bin_f1, all_bin_acc, all_bin_iter = [], [], []

for fold, (tr_idx, val_idx) in enumerate(folds_bin, 1):
    X_tr, X_val = X_bin.iloc[tr_idx], X_bin.iloc[val_idx]
    y_tr, y_val = y_bin[tr_idx], y_bin[val_idx]
    model = lgb.LGBMClassifier(**bin_params)
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
              callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(period=-1)])
    proba = model.predict_proba(X_val)
    conf = np.max(proba, axis=1)
    y_pred = (proba[:, 1] > 0.5).astype(np.int32)

    # Apply confidence threshold to create FLAT
    for thresh in [0.50, 0.53, 0.55, 0.58, 0.60]:
        y_thresh = y_pred.copy()
        y_thresh[conf < thresh] = 2  # use 2 as "FLAT proxy" — not a trade
        # Count how many get filtered
        pass

    f1 = f1_score(y_val, y_pred, average='binary', zero_division=0)
    acc = (y_pred == y_val).mean()
    all_bin_f1.append(f1)
    all_bin_acc.append(acc)
    all_bin_iter.append(model.best_iteration_)

print(f'  Binary CV F1: {np.mean(all_bin_f1):.4f} +/- {np.std(all_bin_f1):.4f}')
print(f'  Binary CV Acc: {np.mean(all_bin_acc):.4f}')
print(f'  Avg iteration: {int(np.mean(all_bin_iter))}')

# Confidence sweep for binary model
print(f'\n  Confidence Threshold Sweep (Fold 1):')
X_tr, X_val = X_bin.iloc[folds_bin[0][0]], X_bin.iloc[folds_bin[0][1]]
y_tr, y_val = y_bin[folds_bin[0][0]], y_bin[folds_bin[0][1]]
model = lgb.LGBMClassifier(**bin_params)
model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
          callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(period=-1)])
proba = model.predict_proba(X_val)
conf = np.max(proba, axis=1)
y_pred = (proba[:, 1] > 0.5).astype(np.int32)

for thresh in [0.50, 0.53, 0.55, 0.58, 0.60, 0.63, 0.65]:
    y_filt = y_pred.copy()
    passed = conf >= thresh
    n_pass = passed.sum()
    n_flat = (~passed).sum()
    if n_pass > 0:
        f1_pass = f1_score(y_val[passed], y_filt[passed], average='binary', zero_division=0)
        acc_pass = (y_filt[passed] == y_val[passed]).mean()
    else:
        f1_pass = 0; acc_pass = 0
    print(f'    thresh={thresh:.2f}: {n_pass} trades ({n_flat} flat) | F1={f1_pass:.4f} Acc={acc_pass:.4f}')
