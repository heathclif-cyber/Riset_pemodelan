"""Custom holdout test for Guardian Delta — per-bar delta feature computation"""
import json, sys, warnings, numpy as np, pandas as pd
warnings.filterwarnings('ignore')
sys.path.insert(0, '.')
from config import *
import joblib
from core.utils import ensure_utc_index

# Load models
tb = joblib.load('models/runs/tb_lgbm_widyawardhana_v3/lgbm.pkl')
with open('models/runs/tb_lgbm_widyawardhana_v3/tb_lgbm_widyawardhana_v3_features.json') as f: tb_feats = json.load(f)
gd = joblib.load('models/runs/tb_guardian_delta_v1/guardian.pkl')
with open('models/runs/tb_guardian_delta_v1/tb_guardian_delta_v1_features.json') as f: gd_feats = json.load(f)

REGIME_THRESH = {0:0.45,1:0.50,2:0.50,3:0.45}
LM = {'SHORT':0,'FLAT':1,'LONG':2}
MOMENTUM_WINDOW = 3
BASE_FEATURES = ["ofi_z_score","cvd_momentum_adv","cvd_slope_h4","ofi_h4_delta","absorption_z",
    "price_accel_1h","rsi_6","trend_strength","ema_21_slope_h4","vol_ratio_20",
    "vol_spike_zscore","atr_percentile_h1","funding_rate","h4_trend"]

available = [s for s in ALL_COINS if (HOLDOUT_DIR/'labeled'/f'{s}_features_v3.parquet').exists()]
print(f"Testing Guardian Delta on {len(available)} coins...")

# Sweep Guardian thresholds (None = no guardian baseline)
for gd_thresh in [None, 0.50, 0.55, 0.60, 0.65, 0.70]:
    total_t, total_w, total_pnl, total_gd_exits = 0, 0, 0.0, 0
    for sym in available:
        df = pd.read_parquet(HOLDOUT_DIR/'labeled'/f'{sym}_features_v3.parquet')
        df = ensure_utc_index(df).sort_index()
        rp = HOLDOUT_DIR/'labeled'/f'{sym}_regime_h1.parquet'
        hmm = np.full(len(df),1,np.int32)
        if rp.exists():
            reg = pd.read_parquet(rp); hmm = reg['hmm_regime_enc'].reindex(df.index,fill_value=1).values.astype(np.int32)
        mask = df['label'].isin(LM); df = df[mask].copy(); hmm = hmm[mask.values]

        X = np.zeros((len(df),len(tb_feats)),dtype=np.float64)
        for i,f in enumerate(tb_feats):
            if f in df.columns: X[:,i] = df[f].ffill().fillna(0).values.astype(np.float64)
        proba = tb.predict_proba(X); conf = np.max(proba,axis=1)
        yp = np.argmax(proba,axis=1)
        for r,th in REGIME_THRESH.items(): yp[(hmm==r)&(yp!=1)&(conf<th)] = 1

        close = df['close'].values; high = df['high'].values; low = df['low'].values
        atr = df['atr_14_h1'].values; n = len(close)

        # Feature cache
        feat_cache = {}
        for f in BASE_FEATURES:
            if f in df.columns: feat_cache[f] = df[f].values

        # Custom trade simulation with Guardian
        trades = []
        tp_atr, sl_atr = 2.0, 1.5
        max_hold = MAX_HOLDING_BARS

        i = 0
        while i < n - 1:
            if yp[i] == 1:  # FLAT
                i += 1; continue

            direction = 1 if yp[i] == 2 else -1
            entry_price = close[i]; entry_bar = i
            atr_entry = atr[i]
            tp_price = entry_price + direction * tp_atr * atr_entry
            sl_price = entry_price - direction * sl_atr * atr_entry

            bar = i + 1
            exited = False
            while bar < min(n, i + max_hold + 1):
                bars_held = bar - i
                if bars_held < 2:
                    bar += 1; continue

                current = close[bar]

                # Check SL
                if (direction == 1 and low[bar] <= sl_price) or (direction == -1 and high[bar] >= sl_price):
                    if direction == 1: exit_p = sl_price * (1 - FEE_PER_SIDE)
                    else: exit_p = sl_price * (1 + FEE_PER_SIDE)
                    pnl = (exit_p - entry_price) * direction * MODAL_PER_TRADE * LEVERAGE_SIM[0] / entry_price
                    trades.append({'outcome':'LOSS','net_pnl':pnl,'direction':'LONG' if direction==1 else 'SHORT','exit_reason':'sl','bars':bars_held})
                    exited = True; break

                # Check TP
                tp_hit = (direction == 1 and high[bar] >= tp_price) or (direction == -1 and low[bar] <= tp_price)
                pnl_pct = (current - entry_price) / entry_price * direction

                # Baseline: exit at TP hit
                if gd_thresh is None:
                    if tp_hit:
                        exit_p = tp_price * (1 - FEE_PER_SIDE) if direction == 1 else tp_price * (1 + FEE_PER_SIDE)
                        pnl = (exit_p - entry_price) * direction * MODAL_PER_TRADE * LEVERAGE_SIM[0] / entry_price
                        trades.append({'outcome':'WIN','net_pnl':pnl,'direction':'LONG' if direction==1 else 'SHORT','exit_reason':'tp','bars':bars_held})
                        exited = True; break
                    bar += 1; continue

                # Guardian mode: compute delta features
                gd_features = {}
                for f in BASE_FEATURES:
                    if f in feat_cache:
                        ev = feat_cache[f][entry_bar] if entry_bar < len(feat_cache[f]) else 0
                        cv = feat_cache[f][bar] if bar < len(feat_cache[f]) else 0
                        ev = 0 if np.isnan(ev) else ev; cv = 0 if np.isnan(cv) else cv
                        gd_features[f'{f}_delta'] = cv - ev
                        gd_features[f'{f}_curr'] = cv
                    else:
                        gd_features[f'{f}_delta'] = 0.0; gd_features[f'{f}_curr'] = 0.0

                # Multi-bar momentum
                for f in ["ofi_z_score","cvd_momentum_adv","price_accel_1h","rsi_6"]:
                    if f in feat_cache:
                        start = max(entry_bar, bar - MOMENTUM_WINDOW + 1)
                        vals = [feat_cache[f][b] for b in range(start, bar+1)
                                if b < len(feat_cache[f]) and not np.isnan(feat_cache[f][b])]
                        if vals:
                            gd_features[f'{f}_mean{MOMENTUM_WINDOW}'] = np.mean(vals)
                            gd_features[f'{f}_trend{MOMENTUM_WINDOW}'] = vals[-1]-vals[0] if len(vals)>=2 else 0
                        else:
                            gd_features[f'{f}_mean{MOMENTUM_WINDOW}'] = 0; gd_features[f'{f}_trend{MOMENTUM_WINDOW}'] = 0
                    else:
                        gd_features[f'{f}_mean{MOMENTUM_WINDOW}'] = 0; gd_features[f'{f}_trend{MOMENTUM_WINDOW}'] = 0

                # Price momentum
                if bars_held >= 3:
                    pch = [(close[b] - close[b-1]) / close[b-1] * direction for b in range(entry_bar+1, bar+1)]
                    gd_features['price_momentum_3bar'] = np.mean(pch[-3:]) if len(pch)>=3 else 0
                else:
                    gd_features['price_momentum_3bar'] = 0

                gd_features['pnl_pct'] = pnl_pct
                gd_features['bars_held'] = bars_held
                gd_features['direction'] = direction

                X_gd = np.array([gd_features.get(f, 0.0) for f in gd_feats]).reshape(1, -1)
                gd_proba = gd.predict_proba(X_gd)[0]  # [P(EXIT), P(HOLD)]
                p_exit = gd_proba[0]

                guardian_exit = False
                if tp_hit:
                    # Momentum mode: hold past TP only if Guardian confident to HOLD
                    if p_exit > gd_thresh:
                        guardian_exit = True  # Guardian says exit — take TP
                elif pnl_pct < -0.01:
                    # Defense mode: early exit if Guardian confident EXIT
                    if p_exit > (gd_thresh - 0.05):
                        guardian_exit = True

                if guardian_exit:
                    exit_p = current * (1 - FEE_PER_SIDE) if direction == 1 else current * (1 + FEE_PER_SIDE)
                    pnl = (exit_p - entry_price) * direction * MODAL_PER_TRADE * LEVERAGE_SIM[0] / entry_price
                    outcome = 'WIN' if pnl > 0 else 'LOSS'
                    trades.append({'outcome':outcome,'net_pnl':pnl,'direction':'LONG' if direction==1 else 'SHORT',
                                   'exit_reason':'guardian','bars':bars_held})
                    total_gd_exits += 1
                    exited = True; break

                # Time exit
                if bars_held >= max_hold:
                    exit_p = current * (1 - FEE_PER_SIDE) if direction == 1 else current * (1 + FEE_PER_SIDE)
                    pnl = (exit_p - entry_price) * direction * MODAL_PER_TRADE * LEVERAGE_SIM[0] / entry_price
                    outcome = 'WIN' if pnl > 0 else 'LOSS'
                    trades.append({'outcome':outcome,'net_pnl':pnl,'direction':'LONG' if direction==1 else 'SHORT',
                                   'exit_reason':'time','bars':bars_held})
                    exited = True; break

                bar += 1

            if not exited:
                # Force exit at last bar
                exit_p = close[min(i+max_hold, n-1)] * (1 - FEE_PER_SIDE) if direction == 1 else close[min(i+max_hold, n-1)] * (1 + FEE_PER_SIDE)
                pnl = (exit_p - entry_price) * direction * MODAL_PER_TRADE * LEVERAGE_SIM[0] / entry_price
                outcome = 'WIN' if pnl > 0 else 'LOSS'
                trades.append({'outcome':outcome,'net_pnl':pnl,'direction':'LONG' if direction==1 else 'SHORT','exit_reason':'time'})

            i = bar  # next trade starts after exit bar (no overlap)

        total_t += len(trades)
        total_w += sum(1 for t in trades if t['outcome']=='WIN')
        total_pnl += sum(t['net_pnl'] for t in trades)

    wr = total_w/max(total_t,1)*100
    pnl_t = total_pnl/max(total_t,1)
    label = "BASELINE (no GD)" if gd_thresh is None else f"Thresh={gd_thresh:.2f}"
    gd_str = "" if gd_thresh is None else f", Guardian exits={total_gd_exits}"
    print(f"{label}: {total_t} trades, WR={wr:.1f}%, PnL=${total_pnl:+.0f}, P/t=${pnl_t:+.2f}{gd_str}")
