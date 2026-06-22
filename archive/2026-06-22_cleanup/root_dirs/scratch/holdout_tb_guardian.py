"""TB v3 + Guardian vs TB v3 No Guardian vs ic32 — holdout comparison"""
import json, sys, warnings, numpy as np, pandas as pd
warnings.filterwarnings('ignore')
sys.path.insert(0, '.')
from config import *
import joblib
from core.evaluator import full_trading_report
from core.utils import ensure_utc_index

# Load models
tb_model = joblib.load('models/runs/tb_lgbm_widyawardhana_v3/lgbm.pkl')
with open('models/runs/tb_lgbm_widyawardhana_v3/tb_lgbm_widyawardhana_v3_features.json') as f:
    tb_feats = json.load(f)
gd_model = joblib.load('models/runs/tb_guardian_widyawardhana_v1/guardian.pkl')
with open('models/runs/tb_guardian_widyawardhana_v1/tb_guardian_widyawardhana_v1_features.json') as f:
    gd_feats = json.load(f)

# Guardian features split: static (from TB) + dynamic (computed by evaluator)
gd_static = [c for c in gd_feats if c in tb_feats]  # 18 static
gd_dynamic = [c for c in gd_feats if c not in tb_feats]  # 7 dynamic
print(f"Guardian: {len(gd_static)} static + {len(gd_dynamic)} dynamic = {len(gd_feats)} total")
print(f"Model expects: {gd_model.n_features_} features")

bl_model = joblib.load(MODEL_DIR / 'lgbm_baseline.pkl')
bl_feats = list(bl_model.feature_name_) if hasattr(bl_model, 'feature_name_') else json.load(open(MODEL_DIR / 'feature_cols_v2.json'))

REGIME_THRESH = {0: 0.45, 1: 0.50, 2: 0.50, 3: 0.45}
available = [s for s in ALL_COINS if (HOLDOUT_DIR / 'labeled' / f'{s}_features_v3.parquet').exists()]
LM = {'SHORT': 0, 'FLAT': 1, 'LONG': 2}

print(f"\n{'='*85}")
print(f"  TB v3 + Guardian vs TB v3 No Guardian vs ic32 (swing holdout)")
print(f"{'='*85}")

for gd_enabled in [False, True]:
    label = "TB+Guardian" if gd_enabled else "TB No Guard"
    print(f"\n--- {label} ---")
    hdr = f'{"Coin":<15} {"Trades":>7} {"WR%":>7} {"PnL":>9} {"PF":>7} {"DD%":>7}'
    print(hdr); print('-' * 58)
    total_t, total_pnl, total_w = 0, 0.0, 0

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
        yt = df['label'].map(LM).values.astype(np.int32)
        a = df['atr_14_h1'].values; c = df['close'].values
        hi = df['high'].values; lo = df['low'].values
        hs = df['h4_swing_high'].values if 'h4_swing_high' in df.columns else None
        ls = df['h4_swing_low'].values if 'h4_swing_low' in df.columns else None
        ht = df['h4_trend'].values if 'h4_trend' in df.columns else None

        # TB inference
        X = np.zeros((len(df), len(tb_feats)), dtype=np.float64)
        for i, f in enumerate(tb_feats):
            if f in df.columns: X[:, i] = df[f].ffill().fillna(0).values.astype(np.float64)
        proba = tb_model.predict_proba(X)
        conf = np.max(proba, axis=1)
        yp = np.argmax(proba, axis=1)
        for r, th in REGIME_THRESH.items():
            yp[(hmm == r) & (yp != 1) & (conf < th)] = 1

        # Guardian setup
        X_gd = None
        gd_scaler = None
        if gd_enabled:
            # Build guardian static features matching training order
            X_gd = np.zeros((len(df), len(gd_static)), dtype=np.float64)
            for i, f in enumerate(gd_static):
                if f in df.columns:
                    X_gd[:, i] = df[f].ffill().fillna(0).values.astype(np.float64)
            # Dummy scaler (model trained on raw features, no scaling)
            from sklearn.preprocessing import StandardScaler
            gd_scaler = StandardScaler()
            gd_scaler.mean_ = np.zeros(len(gd_static) + len(gd_dynamic))
            gd_scaler.scale_ = np.ones(len(gd_static) + len(gd_dynamic))

        if (yp != 1).sum() == 0:
            print(f'{sym:<15} {0:>7} {"-":>7} {0:>+9.0f} {"-":>7} {"-":>7}')
            continue

        rep = full_trading_report(
            y_pred=yp, y_actual=yt, atr=a, close=c, high=hi, low=lo,
            h4_swing_highs=hs, h4_swing_lows=ls, index=df.index,
            modal=MODAL_PER_TRADE, leverages=LEVERAGE_SIM,
            fee_per_side=FEE_PER_SIDE, slippage=SLIPPAGE_PER_SIDE,
            min_rr=SWING_LABEL_MIN_RR, min_tp_atr=SWING_LABEL_MIN_TP,
            max_sl_atr=SWING_LABEL_MAX_SL, tp_fallback_atr=TP_SL_FALLBACK_TP,
            sl_fallback_atr=TP_SL_FALLBACK_SL, max_hold=MAX_HOLDING_BARS,
            symbol=sym, confidence=conf,
            guardian_model=gd_model if gd_enabled else None,
            guardian_scaler=gd_scaler,
            X_guardian=X_gd,
            guardian_enabled=gd_enabled,
            guardian_exit_threshold=0.60,
            trailing_stop_enabled=False, h4_trend=ht,
        )
        lev = rep.get('lev5x', rep)
        tr = lev.get('trades', [])
        wins = sum(1 for t in tr if t.get('outcome') == 'WIN')
        pnl = sum(t.get('net_pnl', 0) for t in tr)
        wr = wins / max(len(tr), 1) * 100
        eq = np.array(lev.get('equity_curve', [0]))
        peak = np.maximum.accumulate(eq); dd_amt = peak - eq
        dd_correct = np.where(peak > 0, dd_amt / peak * 100, 0).max()
        total_t += len(tr); total_pnl += pnl; total_w += wins
        print(f'{sym:<15} {len(tr):>7} {wr:>6.1f}% {pnl:>+9.0f} {lev.get("profit_factor",0):>6.2f} {dd_correct:>6.1f}%')

    print('-' * 58)
    twr = total_w / max(total_t, 1) * 100
    print(f'{"AGGREGATE":<15} {total_t:>7} {twr:>6.1f}% {total_pnl:>+9.0f}')
    print(f'  PnL/t=${total_pnl/max(total_t,1):.2f} | T/koin={total_t//len(available)}')
