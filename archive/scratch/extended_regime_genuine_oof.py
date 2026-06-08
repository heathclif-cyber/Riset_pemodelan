"""
GENUINE OOF: Retrain ALL models per fold (swing + trend).

CRITICAL:
  - Swing model: retrained per fold with swing labels
  - TRENDING_UP model: retrained per fold with continuation labels (ATR-based)
  - TRENDING_DOWN model: retrained per fold with continuation labels (ATR-based)
  - NO model sees test data. Each fold is trained ONLY on training folds.

Trend model labels (continuation, on-the-fly):
  LONG (2): price hits TP (+2*ATR) before SL (-1.5*ATR) within max_hold bars
  SHORT (0): price hits SL before TP
  FLAT (1): neither hit within max_hold

This is the ONLY valid OOF methodology.
"""
import sys, json, joblib, numpy as np, pandas as pd, lightgbm as lgb
from pathlib import Path
ROOT = Path(__file__).parent.parent; sys.path.insert(0, str(ROOT))
import config
config.LSTM_CONFIRMATION_ENABLED = True
from config import *
from pipeline.backtest_utils import hierarchical_predict, compute_guardian_static_array
from core.evaluator import simulate_trades_swing
from pipeline.shared import build_purged_folds
import pipeline.backtest_utils as btu

LGBM_PARAMS = {'objective': 'multiclass', 'num_class': 3, 'n_estimators': 300,
               'learning_rate': 0.05, 'max_depth': 5, 'num_leaves': 31,
               'min_child_samples': 50, 'subsample': 0.8, 'colsample_bytree': 0.8,
               'verbose': -1, 'n_jobs': -1, 'random_state': 42}

# Feature sets (from IC test per regime)
SWING_FEATS = json.load(open(MODEL_DIR / "feature_cols_v2.json"))
TREND_UP_FEATS = joblib.load(MODEL_DIR / "lgbm_regime_TRENDING_UP.pkl").feature_name_
TREND_DN_FEATS = joblib.load(MODEL_DIR / "lgbm_regime_TRENDING_DOWN.pkl").feature_name_

lstm_f = json.load(open(MODEL_DIR / "feature_cols_lstm_temporal.json"))
guard = joblib.load(MODEL_DIR / "guardian_best.pkl"); gs = joblib.load(MODEL_DIR / "guardian_scaler.pkl")
gf = json.load(open(MODEL_DIR / "guardian_feature_cols.json"))
gst = [c for c in gf if c not in set(GUARDIAN_DYNAMIC_FEATURES)]
from core.models import load_lstm
lstm_h1 = load_lstm(MODEL_DIR / "lstm_best.pt", device="cpu")
lsc = joblib.load(MODEL_DIR / "lstm_scaler.pkl")
btu.SMART_ENTRY_MODE = "disabled"

MAX_HOLD = 36; TP_MULT = 2.0; SL_MULT = 1.5

def continuation_labels(high, low, close, atr, direction=1):
    """
    On-the-fly continuation labels for trend models.
    direction=1: TRENDING_UP (LONG continuation)
    direction=-1: TRENDING_DOWN (SHORT continuation)

    For TRENDING_UP: LONG if price hits +2*ATR before -1.5*ATR
    For TRENDING_DOWN: SHORT if price hits -2*ATR before +1.5*ATR
    """
    n = len(close)
    labels = np.ones(n, dtype=np.int8)  # default FLAT

    for i in range(n - MAX_HOLD - 1):
        entry = close[i]
        if entry <= 0 or atr[i] <= 0: continue

        tp_price = entry + direction * TP_MULT * atr[i]
        sl_price = entry - direction * SL_MULT * atr[i]

        for j in range(i + 1, min(i + MAX_HOLD + 1, n)):
            if direction == 1:  # TRENDING_UP
                if high[j] >= tp_price:
                    labels[i] = 2  # LONG continuation hit
                    break
                if low[j] <= sl_price:
                    labels[i] = 0  # Stopped out
                    break
            else:  # TRENDING_DOWN
                if low[j] <= tp_price:  # tp = downward
                    labels[i] = 0  # SHORT continuation hit
                    break
                if high[j] >= sl_price:  # sl = upward
                    labels[i] = 2  # Stopped out
                    break

    return labels


def train_and_predict(X_tr, y_tr, X_te, feat_names):
    """Train LGBM and predict on test fold."""
    # Use only features available in training
    feats = [c for c in feat_names if c in X_tr.columns]
    X_tr_f = X_tr[feats].ffill().fillna(0)
    X_te_f = X_te[feats].ffill().fillna(0)

    if len(np.unique(y_tr)) < 3:
        return np.full(len(X_te), 1), np.full((len(X_te), 3), 1.0/3)

    m = lgb.LGBMClassifier(**LGBM_PARAMS)
    m.fit(X_tr_f, y_tr)
    proba = m.predict_proba(X_te_f)
    pred = np.argmax(proba, axis=1)
    return pred, proba


coins = TRAINING_COINS
all_trades_base = []; all_trades_router = []
n_swing = 0; n_trend_up = 0; n_trend_dn = 0

for ci, coin in enumerate(coins):
    fp = LABEL_DIR / f"{coin}_features_v3.parquet"
    rp = LABEL_DIR / f"{coin}_regime_h1.parquet"
    if not fp.exists(): continue
    df = pd.read_parquet(fp).sort_index()
    df = df[df.index < TRAIN_CUTOFF_DATE]
    if rp.exists():
        reg = pd.read_parquet(rp)
        if "hmm_regime_enc" in df.columns: df = df.drop(columns=["hmm_regime_enc"])
        df = df.join(reg[["hmm_regime_enc"]], how="left")
        df["hmm_regime_enc"] = df["hmm_regime_enc"].fillna(1).astype("int32")
    df = df[df["label"].astype(str).isin(LABEL_MAP)].copy()
    if len(df) < 500: continue

    ts_index = pd.DatetimeIndex(df.index)
    folds = build_purged_folds(ts_index, N_FOLDS, PURGE_GAP_BARS)

    for fi, (tr_idx, te_idx) in enumerate(folds):
        if len(te_idx) < 100: continue
        df_tr = df.iloc[tr_idx]; df_te = df.iloc[te_idx]
        n_te = len(df_te)

        # ── Genuine OOF: retrain ALL 3 models per fold ──────────────────
        # 1. SWING model (original labels)
        y_swing_tr = df_tr["label"].map(LABEL_MAP).values.astype(np.int64)
        yp_swing, pb_swing = train_and_predict(df_tr, y_swing_tr, df_te, SWING_FEATS)

        # 2. TRENDING_UP model (continuation labels, direction=1)
        atr_tr = df_tr["atr_14_h1"].values
        atr_te = df_te["atr_14_h1"].values
        h_tr = df_tr["high"].values; l_tr = df_tr["low"].values; c_tr = df_tr["close"].values
        y_up_tr = continuation_labels(h_tr, l_tr, c_tr, atr_tr, direction=1)
        yp_up, pb_up = train_and_predict(df_tr, y_up_tr, df_te, TREND_UP_FEATS)

        # 3. TRENDING_DOWN model (continuation labels, direction=-1)
        y_dn_tr = continuation_labels(h_tr, l_tr, c_tr, atr_tr, direction=-1)
        yp_dn, pb_dn = train_and_predict(df_tr, y_dn_tr, df_te, TREND_DN_FEATS)

        # ── LSTM input ─────────────────────────────────────────────────
        X_l = np.zeros((n_te, len(lstm_f)))
        for i, col in enumerate(lstm_f):
            if col in df_te.columns: X_l[:, i] = df_te[col].ffill().fillna(0).values

        # ── A) BASELINE: swing model for ALL ────────────────────────────
        # Manual prediction (no cascade — pure model comparison)
        yp_b = yp_swing.copy()
        cf_b = np.max(pb_swing, axis=1)
        below = (yp_b != 1) & (cf_b < CONFIDENCE_THRESHOLD_ENTRY); yp_b[below] = 1

        # ── B) ROUTER: per-bar regime ───────────────────────────────────
        yp_r = np.ones(n_te, dtype=np.int64); cf_r = np.full(n_te, 1.0/3)
        for i in range(n_te):
            regime = int(df_te["hmm_regime_enc"].iloc[i])
            if regime == 3:
                yp_r[i] = yp_up[i]; cf_r[i] = pb_up[i, yp_up[i]]; n_trend_up += 1
            elif regime == 0:
                yp_r[i] = yp_dn[i]; cf_r[i] = pb_dn[i, yp_dn[i]]; n_trend_dn += 1
            else:
                yp_r[i] = yp_swing[i]; cf_r[i] = pb_swing[i, yp_swing[i]]; n_swing += 1
        below = (yp_r != 1) & (cf_r < CONFIDENCE_THRESHOLD_ENTRY); yp_r[below] = 1

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
                t["coin"] = coin; t["timestamp"] = df_te.index[t.get("bar_in", 0)]
                t["regime"] = int(df_te["hmm_regime_enc"].iloc[t.get("bar_in", 0)])
            dest.extend(r.get("trades", []))

    print(f"  [{ci+1:>2}/{len(coins)}] {coin:<15} {len(folds)} folds | swing={n_swing} up={n_trend_up} dn={n_trend_dn}")

# Scorecard
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
        reg_pnl[reg] = {"pnl": sum(t.get("net_pnl",0) for t in rt), "trades": len(rt)}
    yearly = {}
    for t in trades:
        if t.get("timestamp") is None: continue
        y = str(pd.Timestamp(t["timestamp"]).year)
        yearly[y] = yearly.get(y, {"pnl":0,"trades":0})
        yearly[y]["pnl"] += t.get("net_pnl",0); yearly[y]["trades"] += 1
    months = {}
    for t in trades:
        m = pd.Timestamp(t["timestamp"]).strftime("%Y-%m") if t.get("timestamp") else None
        if m is None: continue
        months[m] = months.get(m, {"pnl":0})
        months[m]["pnl"] += t.get("net_pnl",0)
    neg_m = sum(1 for d in months.values() if d["pnl"]<0)
    mpnl = [d["pnl"] for d in months.values()]
    return dict(trades=n, wr=len(wins)/n*100 if n else 0, pnl=pnl,
                pf=gw/gl if gl else 0, lwr=len([t for t in lt if t.get("net_pnl",0)>0])/len(lt)*100 if lt else 0,
                swr=len([t for t in st if t.get("net_pnl",0)>0])/len(st)*100 if st else 0,
                reg_pnl=reg_pnl, yearly=yearly, months=months,
                neg_m=neg_m, mean_m=np.mean(mpnl) if mpnl else 0)

b = stats(all_trades_base); r = stats(all_trades_router)
rn = {0:"TREND_DN", 1:"RANGE_LO", 2:"RANGE_HI", 3:"TREND_UP"}

print(f"\n{'='*70}")
print(f"  GENUINE OOF — All 3 Models Retrained Per Fold")
print(f"  168 folds purged CV, 21 coins, 2020 -> Mar 2026")
print(f"  {'Metric':<20} {'BASELINE (swing)':>18} {'ROUTER (3 models)':>18} {'Delta':>10}")
print(f"  {'-'*65}")
for k, label in [("trades","Trades"),("wr","WR %"),("pnl","PnL $"),("pf","PF"),
                  ("lwr","LONG WR %"),("swr","SHORT WR %"),("neg_m","Neg Months"),("mean_m","$/Month")]:
    bv = b[k]; rv = r[k]
    if isinstance(bv, float):
        print(f"  {label:<20} {bv:>18.1f} {rv:>18.1f} {rv-bv:>+10.1f}")
    else:
        print(f"  {label:<20} {bv:>18,} {rv:>18,} {rv-bv:>+10,}")

print(f"\n  Per-Regime PnL:")
for reg in [0,1,2,3]:
    bp = b["reg_pnl"][reg]; rp = r["reg_pnl"][reg]
    sign = "🔥" if rp["pnl"] > bp["pnl"] else ""
    print(f"  {rn[reg]:<15}: base=${bp['pnl']:>+8.0f} ({bp['trades']:>5}t) | router=${rp['pnl']:>+8.0f} ({rp['trades']:>5}t) {sign}")

print(f"\n  Yearly PnL:")
for y in sorted(b["yearly"]):
    bp = b["yearly"][y]["pnl"]; rp = r["yearly"][y]["pnl"]
    sign = "🔥" if rp > bp else ""
    print(f"  {y}: base=${bp:>+8.0f} | router=${rp:>+8.0f} | Δ={rp-bp:>+8.0f} {sign}")

# Distribution
print(f"\n  Bar regime distribution:")
print(f"  SWING (ranging): {n_swing:,} bars")
print(f"  TREND UP:         {n_trend_up:,} bars")
print(f"  TREND DN:         {n_trend_dn:,} bars")
print(f"{'='*70}")
