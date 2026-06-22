"""
scratch/holdout_guardian_sweep.py
Sweep separate momentum_thresh x defense_thresh x defense_min_loss for Guardian Delta v1.
Pre-caches TB predictions + feature arrays — per-combo only the simulation loop reruns.
"""
import json, sys, warnings, numpy as np, pandas as pd, itertools
warnings.filterwarnings('ignore')
sys.path.insert(0, '.')
from config import *
import joblib
from core.utils import ensure_utc_index

# ── Load models ────────────────────────────────────────────────────────────────
tb = joblib.load('models/runs/tb_lgbm_widyawardhana_v3/lgbm.pkl')
with open('models/runs/tb_lgbm_widyawardhana_v3/tb_lgbm_widyawardhana_v3_features.json') as f:
    tb_feats = json.load(f)
gd = joblib.load('models/runs/tb_guardian_delta_v1/guardian.pkl')
with open('models/runs/tb_guardian_delta_v1/tb_guardian_delta_v1_features.json') as f:
    gd_feats = json.load(f)

REGIME_THRESH = {0: 0.45, 1: 0.50, 2: 0.50, 3: 0.45}
LM = {'SHORT': 0, 'FLAT': 1, 'LONG': 2}
MOMENTUM_WINDOW = 3
TP_ATR, SL_ATR = 2.0, 1.5
BASE_FEATURES = [
    "ofi_z_score", "cvd_momentum_adv", "cvd_slope_h4", "ofi_h4_delta", "absorption_z",
    "price_accel_1h", "rsi_6", "trend_strength", "ema_21_slope_h4", "vol_ratio_20",
    "vol_spike_zscore", "atr_percentile_h1", "funding_rate", "h4_trend",
]
MOM_FEATURES = ["ofi_z_score", "cvd_momentum_adv", "price_accel_1h", "rsi_6"]

available = [s for s in ALL_COINS if (HOLDOUT_DIR / 'labeled' / f'{s}_features_v3.parquet').exists()]
print(f"Pre-caching {len(available)} coins...")

# ── Pre-cache per coin: predictions + feature arrays ──────────────────────────
coin_cache = {}
for sym in available:
    df = pd.read_parquet(HOLDOUT_DIR / 'labeled' / f'{sym}_features_v3.parquet')
    df = ensure_utc_index(df).sort_index()

    rp = HOLDOUT_DIR / 'labeled' / f'{sym}_regime_h1.parquet'
    hmm = np.full(len(df), 1, np.int32)
    if rp.exists():
        reg = pd.read_parquet(rp)
        hmm = reg['hmm_regime_enc'].reindex(df.index, fill_value=1).values.astype(np.int32)

    mask = df['label'].isin(LM)
    df = df[mask].copy()
    hmm = hmm[mask.values]

    X = np.zeros((len(df), len(tb_feats)), dtype=np.float64)
    for i, f in enumerate(tb_feats):
        if f in df.columns:
            X[:, i] = df[f].ffill().fillna(0).values.astype(np.float64)

    proba = tb.predict_proba(X)
    conf = np.max(proba, axis=1)
    yp = np.argmax(proba, axis=1)
    for r, th in REGIME_THRESH.items():
        yp[(hmm == r) & (yp != 1) & (conf < th)] = 1

    feat_cache = {}
    for f in BASE_FEATURES:
        if f in df.columns:
            feat_cache[f] = df[f].values.astype(np.float64)

    coin_cache[sym] = {
        'yp': yp,
        'close': df['close'].values.astype(np.float64),
        'high': df['high'].values.astype(np.float64),
        'low': df['low'].values.astype(np.float64),
        'atr': df['atr_14_h1'].values.astype(np.float64),
        'feat': feat_cache,
        'n': len(df),
    }

print(f"  Done. Total coins cached: {len(coin_cache)}\n")


# ── Simulation function ────────────────────────────────────────────────────────
def run_sim(mom_thresh, def_thresh, def_min_loss, use_guardian=True):
    """
    mom_thresh   : P(EXIT) to confirm exit AFTER TP hit (lower = hold longer = more aggressive momentum)
    def_thresh   : P(EXIT) to cut early loss (higher = more conservative = less intervention)
    def_min_loss : minimum pnl_pct before defense activates (negative, e.g. -0.010)
    """
    total_t = total_w = total_gd_exits = 0
    total_pnl = 0.0

    for sym, c in coin_cache.items():
        yp = c['yp']; close = c['close']; high = c['high']; low = c['low']
        atr = c['atr']; feat_cache = c['feat']; n = c['n']

        trades = []
        i = 0
        while i < n - 1:
            if yp[i] == 1:
                i += 1; continue

            direction = 1 if yp[i] == 2 else -1
            entry_price = close[i]; entry_bar = i
            atr_entry = atr[i]
            tp_price = entry_price + direction * TP_ATR * atr_entry
            sl_price = entry_price - direction * SL_ATR * atr_entry

            bar = i + 1
            exited = False

            while bar < min(n, i + MAX_HOLDING_BARS + 1):
                bars_held = bar - i
                if bars_held < 2:
                    bar += 1; continue

                current = close[bar]
                pnl_pct = (current - entry_price) / entry_price * direction

                # SL check (always hard stop)
                sl_hit = (direction == 1 and low[bar] <= sl_price) or (direction == -1 and high[bar] >= sl_price)
                if sl_hit:
                    exit_p = sl_price * (1 - FEE_PER_SIDE) if direction == 1 else sl_price * (1 + FEE_PER_SIDE)
                    pnl = (exit_p - entry_price) * direction * MODAL_PER_TRADE * LEVERAGE_SIM[0] / entry_price
                    trades.append({'outcome': 'LOSS', 'net_pnl': pnl, 'exit_reason': 'sl', 'bars': bars_held})
                    exited = True; break

                tp_hit = (direction == 1 and high[bar] >= tp_price) or (direction == -1 and low[bar] <= tp_price)

                # No Guardian: static TP exit
                if not use_guardian:
                    if tp_hit:
                        exit_p = tp_price * (1 - FEE_PER_SIDE) if direction == 1 else tp_price * (1 + FEE_PER_SIDE)
                        pnl = (exit_p - entry_price) * direction * MODAL_PER_TRADE * LEVERAGE_SIM[0] / entry_price
                        trades.append({'outcome': 'WIN', 'net_pnl': pnl, 'exit_reason': 'tp', 'bars': bars_held})
                        exited = True; break
                    bar += 1; continue

                # ── Guardian feature computation ────────────────────────────
                gd_features = {}
                for f in BASE_FEATURES:
                    arr = feat_cache.get(f)
                    if arr is not None:
                        ev = arr[entry_bar] if entry_bar < len(arr) else 0.0
                        cv = arr[bar] if bar < len(arr) else 0.0
                        ev = 0.0 if np.isnan(ev) else ev
                        cv = 0.0 if np.isnan(cv) else cv
                        gd_features[f'{f}_delta'] = cv - ev
                        gd_features[f'{f}_curr'] = cv
                    else:
                        gd_features[f'{f}_delta'] = 0.0
                        gd_features[f'{f}_curr'] = 0.0

                for f in MOM_FEATURES:
                    arr = feat_cache.get(f)
                    if arr is not None:
                        start = max(entry_bar, bar - MOMENTUM_WINDOW + 1)
                        vals = [arr[b] for b in range(start, bar + 1)
                                if b < len(arr) and not np.isnan(arr[b])]
                        if vals:
                            gd_features[f'{f}_mean{MOMENTUM_WINDOW}'] = float(np.mean(vals))
                            gd_features[f'{f}_trend{MOMENTUM_WINDOW}'] = float(vals[-1] - vals[0]) if len(vals) >= 2 else 0.0
                        else:
                            gd_features[f'{f}_mean{MOMENTUM_WINDOW}'] = 0.0
                            gd_features[f'{f}_trend{MOMENTUM_WINDOW}'] = 0.0
                    else:
                        gd_features[f'{f}_mean{MOMENTUM_WINDOW}'] = 0.0
                        gd_features[f'{f}_trend{MOMENTUM_WINDOW}'] = 0.0

                if bars_held >= 3:
                    pch = [(close[b] - close[b - 1]) / close[b - 1] * direction
                           for b in range(entry_bar + 1, bar + 1)]
                    gd_features['price_momentum_3bar'] = float(np.mean(pch[-3:])) if len(pch) >= 3 else 0.0
                else:
                    gd_features['price_momentum_3bar'] = 0.0

                gd_features['pnl_pct'] = pnl_pct
                gd_features['bars_held'] = bars_held
                gd_features['direction'] = direction

                X_gd = np.array([gd_features.get(f, 0.0) for f in gd_feats], dtype=np.float64).reshape(1, -1)
                p_exit = gd.predict_proba(X_gd)[0][0]  # P(EXIT)

                # ── Exit decision ───────────────────────────────────────────
                guardian_exit = False

                if tp_hit:
                    # MOMENTUM MODE: default is HOLD past TP, exit only if P(EXIT) > mom_thresh
                    if p_exit > mom_thresh:
                        guardian_exit = True
                elif pnl_pct <= def_min_loss:
                    # DEFENSE MODE: cut only if very confident (conservative threshold)
                    if p_exit > def_thresh:
                        guardian_exit = True

                if guardian_exit:
                    exit_p = current * (1 - FEE_PER_SIDE) if direction == 1 else current * (1 + FEE_PER_SIDE)
                    pnl = (exit_p - entry_price) * direction * MODAL_PER_TRADE * LEVERAGE_SIM[0] / entry_price
                    outcome = 'WIN' if pnl > 0 else 'LOSS'
                    trades.append({'outcome': outcome, 'net_pnl': pnl, 'exit_reason': 'guardian', 'bars': bars_held})
                    total_gd_exits += 1
                    exited = True; break

                if bars_held >= MAX_HOLDING_BARS:
                    exit_p = current * (1 - FEE_PER_SIDE) if direction == 1 else current * (1 + FEE_PER_SIDE)
                    pnl = (exit_p - entry_price) * direction * MODAL_PER_TRADE * LEVERAGE_SIM[0] / entry_price
                    outcome = 'WIN' if pnl > 0 else 'LOSS'
                    trades.append({'outcome': outcome, 'net_pnl': pnl, 'exit_reason': 'time', 'bars': bars_held})
                    exited = True; break

                bar += 1

            if not exited:
                eb = min(i + MAX_HOLDING_BARS, n - 1)
                exit_p = close[eb] * (1 - FEE_PER_SIDE) if direction == 1 else close[eb] * (1 + FEE_PER_SIDE)
                pnl = (exit_p - entry_price) * direction * MODAL_PER_TRADE * LEVERAGE_SIM[0] / entry_price
                outcome = 'WIN' if pnl > 0 else 'LOSS'
                trades.append({'outcome': outcome, 'net_pnl': pnl, 'exit_reason': 'time'})

            i = bar

        total_t += len(trades)
        total_w += sum(1 for t in trades if t['outcome'] == 'WIN')
        total_pnl += sum(t['net_pnl'] for t in trades)

    wr = total_w / max(total_t, 1) * 100
    pnl_t = total_pnl / max(total_t, 1)
    return total_t, wr, total_pnl, pnl_t, total_gd_exits


# ── Baseline ───────────────────────────────────────────────────────────────────
bt, bwr, bpnl, bpt, _ = run_sim(0, 0, 0, use_guardian=False)
print(f"{'Config':<42} {'Tr':>5} {'WR%':>6} {'PnL':>8} {'P/t':>5} {'vs Base':>8} {'GD%':>6}")
print("-" * 82)
print(f"{'BASELINE (no Guardian)':<42} {bt:>5} {bwr:>5.1f}% {bpnl:>+8.0f} {bpt:>+5.2f} {'':>8}")

# ── Sweep grid ─────────────────────────────────────────────────────────────────
# mom_thresh  : after TP hit, exit if P(EXIT) > this  (lower = hold more aggressively)
# def_thresh  : in loss, exit if P(EXIT) > this        (higher = cut less = conservative)
# def_min_loss: minimum loss pct to activate defense   (more negative = wait longer)
mom_thresholds  = [0.45, 0.50, 0.55]
def_thresholds  = [0.65, 0.70, 0.75, 0.80]
def_min_losses  = [-0.010, -0.015]

results = []
for mt, dt, dl in itertools.product(mom_thresholds, def_thresholds, def_min_losses):
    t, wr, pnl, pt, gd_ex = run_sim(mt, dt, dl)
    delta = pnl - bpnl
    gd_pct = gd_ex / max(t, 1) * 100
    label = f"mom={mt:.2f} def={dt:.2f} loss>{dl:.3f}"
    print(f"{label:<42} {t:>5} {wr:>5.1f}% {pnl:>+8.0f} {pt:>+5.2f} {delta:>+8.0f} {gd_pct:>5.1f}%")
    results.append({'mom': mt, 'def': dt, 'loss': dl, 'trades': t, 'wr': wr, 'pnl': pnl, 'pt': pt, 'delta': delta})

print("\n" + "=" * 82)
print("  TOP 5 by PnL:")
results.sort(key=lambda x: x['pnl'], reverse=True)
for r in results[:5]:
    print(f"  mom={r['mom']:.2f} def={r['def']:.2f} loss>{r['loss']:.3f}  "
          f"=> {r['trades']} trades, WR={r['wr']:.1f}%, PnL=${r['pnl']:+.0f} ({r['delta']:+.0f} vs base)")

print("\n  TOP 5 by P/trade:")
results.sort(key=lambda x: x['pt'], reverse=True)
for r in results[:5]:
    print(f"  mom={r['mom']:.2f} def={r['def']:.2f} loss>{r['loss']:.3f}  "
          f"=> {r['trades']} trades, WR={r['wr']:.1f}%, P/t=${r['pt']:+.2f}, PnL=${r['pnl']:+.0f}")

best = max(results, key=lambda x: x['pnl'])
print(f"\n  BEST: mom={best['mom']:.2f} | def={best['def']:.2f} | loss>{best['loss']:.3f}")
print(f"        Trades={best['trades']} | WR={best['wr']:.1f}% | PnL=${best['pnl']:+.0f} | P/t=${best['pt']:+.2f}")
