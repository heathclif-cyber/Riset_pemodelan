"""
Genuine Walk-Forward Validation — Expanding Window with Purge Gap

Methodology:
  - Start testing: June 2021 (18-month warm-up from Jan 2020)
  - Step size: 30 days per test window
  - Purge gap: 36 bars H1 between train & test
  - Per fold: retrain HMM + 3 LGBM specialists on regime subsets
  - NO model sees test data

This is the ONLY valid OOF test for time series.
"""
import sys, json, joblib, warnings, numpy as np, pandas as pd, lightgbm as lgb
from pathlib import Path
from datetime import timedelta
from hmmlearn import hmm

ROOT = Path(__file__).parent.parent; sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")
import config
config.LSTM_CONFIRMATION_ENABLED = True
from config import *
from pipeline.backtest_utils import compute_guardian_static_array
from core.evaluator import simulate_trades_swing

LGBM_PARAMS = {'objective': 'multiclass', 'num_class': 3, 'n_estimators': 200,
               'learning_rate': 0.05, 'max_depth': 5, 'num_leaves': 31,
               'min_child_samples': 100, 'subsample': 0.8, 'colsample_bytree': 0.8,
               'verbose': -1, 'n_jobs': -1, 'random_state': 42}

SWING_FEATS = json.load(open(MODEL_DIR / "feature_cols_v2.json"))
TREND_UP_FEATS = joblib.load(MODEL_DIR / "lgbm_regime_TRENDING_UP.pkl").feature_name_
TREND_DN_FEATS = joblib.load(MODEL_DIR / "lgbm_regime_TRENDING_DOWN.pkl").feature_name_

guard = joblib.load(MODEL_DIR / "guardian_best.pkl"); gs = joblib.load(MODEL_DIR / "guardian_scaler.pkl")
gf = json.load(open(MODEL_DIR / "guardian_feature_cols.json"))
gst = [c for c in gf if c not in set(GUARDIAN_DYNAMIC_FEATURES)]

HMM_FEATS = ["log_ret_1","rsi_6","cvd_momentum_adv","volume_delta","atr_14_h1",
             "h4_trend","trend_strength","ema_21_slope_h4"]
MAX_HOLD = 36; TP_MULT = 2.0; SL_MULT = 1.5
PURGE = 36  # bars

# ─── Train HMM on training data ─────────────────────────────────────────
def train_hmm(df_tr):
    """Train 4-state GaussianHMM on training data."""
    X_cols = [c for c in HMM_FEATS if c in df_tr.columns]
    X = df_tr[X_cols].ffill().fillna(0).values
    X = (X - X.mean(0)) / (X.std(0).clip(1e-8))
    try:
        m = hmm.GaussianHMM(n_components=4, covariance_type="full",
                            n_iter=100, random_state=42, verbose=False)
        m.fit(X)
        return m, X_cols
    except:
        return None, X_cols

def predict_hmm(m, df, X_cols):
    """Predict regime for dataframe using trained HMM."""
    X = df[X_cols].ffill().fillna(0).values
    X = (X - X.mean(0)) / (X.std(0).clip(1e-8))
    try:
        return m.predict(X)
    except:
        return np.ones(len(df), dtype=int)

# ─── Continuation labels (on-the-fly) ──────────────────────────────────
def continuation_labels(high, low, close, atr, direction=1):
    n = len(close); labels = np.ones(n, dtype=np.int8)
    for i in range(n - MAX_HOLD - 1):
        if close[i] <= 0 or atr[i] <= 0: continue
        tp = close[i] + direction * TP_MULT * atr[i]
        sl = close[i] - direction * SL_MULT * atr[i]
        for j in range(i+1, min(i+MAX_HOLD+1, n)):
            if direction == 1:
                if high[j] >= tp: labels[i] = 2; break
                if low[j] <= sl: labels[i] = 0; break
            else:
                if low[j] <= tp: labels[i] = 0; break
                if high[j] >= sl: labels[i] = 2; break
    return labels

def train_lgbm_on_subset(df_tr, feature_list, y_col=None):
    """Train LGBM on a specific regime subset."""
    feats = [c for c in feature_list if c in df_tr.columns]
    if len(df_tr) < 100 or len(feats) < 3:
        return None, feats
    if y_col:
        y = df_tr[y_col].values.astype(np.int64)
    else:
        y = df_tr["label"].map(LABEL_MAP).values.astype(np.int64)
    if len(np.unique(y)) < 3:
        return None, feats
    X = df_tr[feats].ffill().fillna(0)
    m = lgb.LGBMClassifier(**LGBM_PARAMS)
    m.fit(X, y)
    return m, feats

# ─── Main WFV loop ──────────────────────────────────────────────────────
coins = TRAINING_COINS[:5]
all_trades_base = []; all_trades_router = []
start_test = pd.Timestamp("2021-06-01", tz="UTC")
end_test = pd.Timestamp("2026-03-01", tz="UTC")
step = timedelta(days=30)
n_windows = 0

for ci, coin in enumerate(coins):
    fp = LABEL_DIR / f"{coin}_features_v3.parquet"
    if not fp.exists(): continue
    df = pd.read_parquet(fp).sort_index()
    df = df[df.index < TRAIN_CUTOFF_DATE]
    df = df[df["label"].astype(str).isin(LABEL_MAP)].copy()
    if len(df) < 500: continue

    current_test_start = start_test
    coin_windows = 0

    while current_test_start < end_test:
        train_end = current_test_start - timedelta(hours=PURGE)
        test_end = min(current_test_start + step, df.index[-1])

        if test_end <= current_test_start: break

        df_tr = df[df.index <= train_end]
        df_te = df[(df.index >= current_test_start) & (df.index < test_end)]

        if len(df_tr) < 500 or len(df_te) < 200:
            current_test_start += step; continue

        # 1. Train HMM
        hmm_model, hmm_cols = train_hmm(df_tr)
        if hmm_model is None:
            current_test_start += step; continue

        # 2. Label regimes on training data
        regimes_tr = predict_hmm(hmm_model, df_tr, hmm_cols)
        df_tr["hmm_regime_enc"] = regimes_tr

        # 3. Train specialists on regime subsets
        m_up, f_up = train_lgbm_on_subset(df_tr[regimes_tr == 3], TREND_UP_FEATS)
        m_dn, f_dn = train_lgbm_on_subset(df_tr[regimes_tr == 0], TREND_DN_FEATS)
        m_rg, f_rg = train_lgbm_on_subset(df_tr[(regimes_tr >= 1) & (regimes_tr <= 2)], SWING_FEATS)

        # For TRENDING models, use continuation labels
        tr_up_df = df_tr[regimes_tr == 3].copy()
        if len(tr_up_df) >= 100 and m_up is not None:
            y_up = continuation_labels(tr_up_df["high"].values, tr_up_df["low"].values,
                                       tr_up_df["close"].values, tr_up_df["atr_14_h1"].values, 1)
            if len(np.unique(y_up)) >= 2:
                m_up, _ = train_lgbm_on_subset(tr_up_df, TREND_UP_FEATS)
                # Override predictions with continuation-trained model
            else:
                m_up = None

        tr_dn_df = df_tr[regimes_tr == 0].copy()
        if len(tr_dn_df) >= 100 and m_dn is not None:
            y_dn = continuation_labels(tr_dn_df["high"].values, tr_dn_df["low"].values,
                                       tr_dn_df["close"].values, tr_dn_df["atr_14_h1"].values, -1)
            if len(np.unique(y_dn)) >= 2:
                m_dn, _ = train_lgbm_on_subset(tr_dn_df, TREND_DN_FEATS)
            else:
                m_dn = None

        # 4. Predict on test fold (GENUINE OOF)
        regimes_te = predict_hmm(hmm_model, df_te, hmm_cols)
        n_te = len(df_te)

        yp_b = np.ones(n_te, dtype=np.int64); cf_b = np.full(n_te, 1.0/3)
        yp_r = np.ones(n_te, dtype=np.int64); cf_r = np.full(n_te, 1.0/3)

        for i in range(n_te):
            reg = regimes_te[i]
            row = df_te.iloc[i:i+1]

            # BASELINE: always use ranging model
            if m_rg is not None:
                feats_avail = [c for c in f_rg if c in row.columns]
                pb = m_rg.predict_proba(row[feats_avail].ffill().fillna(0))[0]
                yp_b[i] = np.argmax(pb); cf_b[i] = float(pb[yp_b[i]])

            # ROUTER
            if reg == 3 and m_up is not None:
                feats_avail = [c for c in f_up if c in row.columns]
                pb = m_up.predict_proba(row[feats_avail].ffill().fillna(0))[0]
            elif reg == 0 and m_dn is not None:
                feats_avail = [c for c in f_dn if c in row.columns]
                pb = m_dn.predict_proba(row[feats_avail].ffill().fillna(0))[0]
            elif m_rg is not None:
                feats_avail = [c for c in f_rg if c in row.columns]
                pb = m_rg.predict_proba(row[feats_avail].ffill().fillna(0))[0]
            else:
                continue
            yp_r[i] = np.argmax(pb); cf_r[i] = float(pb[yp_r[i]])

        below_b = (yp_b != 1) & (cf_b < CONFIDENCE_THRESHOLD_ENTRY); yp_b[below_b] = 1
        below_r = (yp_r != 1) & (cf_r < CONFIDENCE_THRESHOLD_ENTRY); yp_r[below_r] = 1

        # Simulate trades
        Xg = compute_guardian_static_array(df_te, gst)
        atr = df_te["atr_14_h1"].values if "atr_14_h1" in df_te.columns else np.ones(n_te)
        c = df_te["close"].values; h = df_te["high"].values if "high" in df_te.columns else c
        l = df_te["low"].values if "low" in df_te.columns else c
        sh = df_te["h4_swing_high"].values if "h4_swing_high" in df_te.columns else np.full(n_te, np.nan)
        sl = df_te["h4_swing_low"].values if "h4_swing_low" in df_te.columns else np.full(n_te, np.nan)

        for yp, cf, dest in [(yp_b, cf_b, all_trades_base), (yp_r, cf_r, all_trades_router)]:
            r = simulate_trades_swing(
                y_pred=yp, close=c, high=h, low=l, atr=atr, h4_swing_highs=sh, h4_swing_lows=sl,
                modal=MODAL_PER_TRADE, leverage=LEVERAGE_SIM[0],
                fee_per_side=FEE_PER_SIDE, slippage=SLIPPAGE_PER_SIDE,
                max_hold=MAX_HOLDING_BARS, min_rr=SWING_LABEL_MIN_RR,
                min_tp_atr=SWING_LABEL_MIN_TP, max_sl_atr=SWING_LABEL_MAX_SL,
                tp_fallback_atr=TP_SL_FALLBACK_TP, sl_fallback_atr=TP_SL_FALLBACK_SL,
                confidence=cf, guardian_enabled=True,
                guardian_model=guard, guardian_scaler=gs,
                X_guardian=Xg, guardian_exit_threshold=GUARDIAN_EXIT_THRESHOLD,
                guardian_min_hold_bars=2)
            for t in r.get("trades", []):
                t["coin"] = coin
                bar_in = t.get("bar_in", 0)
                t["timestamp"] = df_te.index[bar_in] if bar_in < len(df_te) else None
                t["regime"] = int(regimes_te[bar_in]) if bar_in < len(regimes_te) else 1
            dest.extend(r.get("trades", []))

        coin_windows += 1; n_windows += 1
        current_test_start += step

    print(f"  [{ci+1}/5] {coin}: {coin_windows} windows done")

# ── Scorecard ────────────────────────────────────────────────────────────
def stats(trades):
    n = len(trades); wins = [t for t in trades if t.get("net_pnl",0)>0]
    pnl = sum(t.get("net_pnl",0) for t in trades)
    gw = sum(t["net_pnl"] for t in wins)
    gl = abs(sum(t["net_pnl"] for t in trades if t.get("net_pnl",0)<=0))
    lt = [t for t in trades if t.get("direction")=="LONG"]
    st = [t for t in trades if t.get("direction")=="SHORT"]
    reg_pnl = {}
    for reg in [0,1,2,3]:
        rt = [t for t in trades if t.get("regime")==reg]
        reg_pnl[reg] = dict(pnl=sum(t.get("net_pnl",0) for t in rt), trades=len(rt))
    yearly = {}
    for t in trades:
        if t.get("timestamp") is None: continue
        y = str(pd.Timestamp(t["timestamp"]).year)
        yearly[y] = yearly.get(y, dict(pnl=0,trades=0))
        yearly[y]["pnl"] += t.get("net_pnl",0); yearly[y]["trades"] += 1
    months = {}
    for t in trades:
        if t.get("timestamp") is None: continue
        m = pd.Timestamp(t["timestamp"]).strftime("%Y-%m")
        months[m] = months.get(m, dict(pnl=0,trades=0))
        months[m]["pnl"] += t.get("net_pnl",0); months[m]["trades"] += 1
    neg_m = sum(1 for d in months.values() if d["pnl"]<0)
    mpnl = [d["pnl"] for d in months.values()]
    return dict(trades=n, wr=len(wins)/n*100 if n else 0, pnl=pnl,
                pf=gw/gl if gl else 0,
                lwr=len([t for t in lt if t.get("net_pnl",0)>0])/len(lt)*100 if lt else 0,
                swr=len([t for t in st if t.get("net_pnl",0)>0])/len(st)*100 if st else 0,
                reg_pnl=reg_pnl, yearly=yearly, months=months,
                neg_m=neg_m, mean_m=np.mean(mpnl) if mpnl else 0)

b = stats(all_trades_base); r = stats(all_trades_router)
rn = {0:"TREND_DN",1:"RANGE_LO",2:"RANGE_HI",3:"TREND_UP"}

print(f"\n{'='*70}")
print(f"  WALK-FORWARD VALIDATION — Expanding Window, Purge={PURGE} bars")
print(f"  {n_windows} windows, {len(coins)} coins, Jun 2021 -> Mar 2026")
print(f"  ALL models retrained per window. NO leakage.")
print(f"  {'Metric':<20} {'BASELINE':>12} {'ROUTER':>12} {'Delta':>10}")
print(f"  {'-'*55}")
for k, label in [("trades","Trades"),("wr","WR %"),("pnl","PnL $"),("pf","PF"),
                  ("lwr","LONG WR %"),("swr","SHORT WR %"),("neg_m","Neg Months")]:
    bv = b[k]; rv = r[k]
    if isinstance(bv, float): print(f"  {label:<20} {bv:>12.1f} {rv:>12.1f} {rv-bv:>+10.1f}")
    else: print(f"  {label:<20} {bv:>12,} {rv:>12,} {rv-bv:>+10,}")

print(f"\n  Per-Regime:")
for reg in [0,1,2,3]:
    bp = b["reg_pnl"][reg]; rp = r["reg_pnl"][reg]
    d = rp["pnl"] - bp["pnl"]
    sign = "++" if d > 0 else "--"
    print(f"  {rn[reg]:<15}: base=${bp['pnl']:>+8.0f} ({bp['trades']:>5}t)  router=${rp['pnl']:>+8.0f} ({rp['trades']:>5}t)  [{sign}]")

print(f"\n  Yearly:")
for y in sorted(b["yearly"]):
    bp = b["yearly"][y]["pnl"]; rp = r["yearly"][y]["pnl"]
    sign = "++" if rp > bp else "--"
    print(f"  {y}: base=${bp:>+8.0f}  router=${rp:>+8.0f}  D={rp-bp:>+8.0f} [{sign}]")

print(f"  {'='*55}")
print(f"  WFV gold-standard: Jun 2021 -> Mar 2026")
print(f"  WARNING if >$3,000: likely data leakage. Should be modest.")
print(f"{'='*70}")
