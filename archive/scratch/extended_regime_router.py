"""
Extended OOF Backtest: Regime Router with 3 Specialized LGBM Models

- RANGING (1,2): lgbm_baseline.pkl (swing, 108 feat)
- TRENDING_UP (3): lgbm_regime_TRENDING_UP.pkl (momentum, 23 feat)
- TRENDING_DOWN (0): lgbm_regime_TRENDING_DOWN.pkl (momentum, 23 feat)

Purged CV: LGBM retrained per fold for SWING only.
Trend models are fixed (pre-trained with continuation labels).
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

LGBM_PARAMS = {'objective': 'multiclass', 'num_class': 3, 'n_estimators': 500,
               'learning_rate': 0.05, 'max_depth': 6, 'num_leaves': 31,
               'min_child_samples': 50, 'subsample': 0.8, 'colsample_bytree': 0.8,
               'verbose': -1, 'n_jobs': -1, 'random_state': 42}

lgbm_f = json.load(open(MODEL_DIR / "feature_cols_v2.json"))
lstm_f = json.load(open(MODEL_DIR / "feature_cols_lstm_temporal.json"))
guard = joblib.load(MODEL_DIR / "guardian_best.pkl"); gs = joblib.load(MODEL_DIR / "guardian_scaler.pkl")
gf = json.load(open(MODEL_DIR / "guardian_feature_cols.json"))
gst = [c for c in gf if c not in set(GUARDIAN_DYNAMIC_FEATURES)]
from core.models import load_lstm
lstm_h1 = load_lstm(MODEL_DIR / "lstm_best.pt", device="cpu")
lsc = joblib.load(MODEL_DIR / "lstm_scaler.pkl")
btu.SMART_ENTRY_MODE = "disabled"

# Load pre-trained trend models + their feature columns
trend_up = joblib.load(MODEL_DIR / "lgbm_regime_TRENDING_UP.pkl")
trend_dn = joblib.load(MODEL_DIR / "lgbm_regime_TRENDING_DOWN.pkl")
trend_up_feats = trend_up.feature_name_
trend_dn_feats = trend_dn.feature_name_
print(f"Trend UP: {len(trend_up_feats)} feats")
print(f"Trend DN: {len(trend_dn_feats)} feats")

coins = TRAINING_COINS
all_trades_router = []; all_trades_baseline = []
total_folds = 0

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

        # Train swing model per fold (OOF)
        feat_cols = [c for c in lgbm_f if c in df_tr.columns]
        X_tr = df_tr[feat_cols].ffill().fillna(0)
        y_tr = df_tr["label"].map(LABEL_MAP).values.astype(np.int64)
        if len(np.unique(y_tr)) < 3: continue
        fold_swing = lgb.LGBMClassifier(**LGBM_PARAMS)
        fold_swing.fit(X_tr, y_tr)

        n_te = len(df_te)
        # Build feature matrices for each model
        # Swing features
        X_swing = np.zeros((n_te, len(feat_cols)))
        for i, col in enumerate(feat_cols):
            if col in df_te.columns: X_swing[:, i] = df_te[col].ffill().fillna(0).values
        # Trend UP features
        trend_up_cols = [c for c in trend_up_feats if c in df_te.columns]
        X_up = np.zeros((n_te, len(trend_up_cols)))
        for i, col in enumerate(trend_up_cols):
            if col in df_te.columns: X_up[:, i] = df_te[col].ffill().fillna(0).values
        # Trend DN features
        trend_dn_cols = [c for c in trend_dn_feats if c in df_te.columns]
        X_dn = np.zeros((n_te, len(trend_dn_cols)))
        for i, col in enumerate(trend_dn_cols):
            if col in df_te.columns: X_dn[:, i] = df_te[col].ffill().fillna(0).values

        X_l = np.zeros((n_te, len(lstm_f)))
        for i, col in enumerate(lstm_f):
            if col in df_te.columns: X_l[:, i] = df_te[col].ffill().fillna(0).values

        # A) BASELINE: use fold_swing for ALL regimes
        yp_b, cf_b = hierarchical_predict(None, fold_swing, lstm_h1, lsc, X_l, feat_cols, [], df_te,
                                            trend_alignment_enabled=False, regime_aware_alignment=True)
        below = (yp_b != 1) & (cf_b < CONFIDENCE_THRESHOLD_ENTRY); yp_b[below] = 1

        # B) ROUTER: per-bar regime routing
        yp_r = np.ones(n_te, dtype=np.int64); cf_r = np.full(n_te, 1.0/3)

        for i in range(n_te):
            regime = int(df_te["hmm_regime_enc"].iloc[i])
            # Select model + feature matrix per regime
            if regime == 3 and trend_up is not None:
                proba = trend_up.predict_proba(X_up[i:i+1])[0]
            elif regime == 0 and trend_dn is not None:
                proba = trend_dn.predict_proba(X_dn[i:i+1])[0]
            else:
                proba = fold_swing.predict_proba(X_swing[i:i+1])[0]  # RANGING → swing

            y_pred_i = int(np.argmax(proba))
            conf_i = float(proba[y_pred_i])
            yp_r[i] = y_pred_i
            cf_r[i] = conf_i

        below_r = (yp_r != 1) & (cf_r < CONFIDENCE_THRESHOLD_ENTRY); yp_r[below_r] = 1

        Xg = compute_guardian_static_array(df_te, gst)
        atr = df_te["atr_14_h1"].values if "atr_14_h1" in df_te.columns else np.ones(n_te)
        c = df_te["close"].values
        h = df_te["high"].values if "high" in df_te.columns else c
        l = df_te["low"].values if "low" in df_te.columns else c
        sh = df_te["h4_swing_high"].values if "h4_swing_high" in df_te.columns else np.full(n_te, np.nan)
        sl = df_te["h4_swing_low"].values if "h4_swing_low" in df_te.columns else np.full(n_te, np.nan)

        for yp, cf, label in [(yp_b, cf_b, "B"), (yp_r, cf_r, "R")]:
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
            if label == "B": all_trades_baseline.extend(r.get("trades", []))
            else: all_trades_router.extend(r.get("trades", []))

        total_folds += 1
    print(f"  [{ci+1:>2}/{len(coins)}] {coin:<15} {len(folds)} folds")

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
        reg_pnl[reg] = {"pnl": sum(t.get("net_pnl",0) for t in rt), "trades": len(rt)} if rt else {"pnl":0,"trades":0}
    yearly = {}
    for t in trades:
        if t.get("timestamp") is None: continue
        y = str(pd.Timestamp(t["timestamp"]).year)
        yearly[y] = yearly.get(y, {"pnl":0,"trades":0})
        yearly[y]["pnl"] += t.get("net_pnl",0); yearly[y]["trades"] += 1
    return dict(trades=n, wr=len(wins)/n*100 if n else 0, pnl=pnl,
                pf=gw/gl if gl else 0, lwr=len([t for t in lt if t.get("net_pnl",0)>0])/len(lt)*100 if lt else 0,
                swr=len([t for t in st if t.get("net_pnl",0)>0])/len(st)*100 if st else 0,
                reg_pnl=reg_pnl, yearly=yearly)

b = stats(all_trades_baseline)
r = stats(all_trades_router)
rn = {0:"TREND_DN", 1:"RANGE_LO", 2:"RANGE_HI", 3:"TREND_UP"}

print(f"\n{'='*70}")
print(f"  REGIME ROUTER vs BASELINE — {total_folds} folds purged CV, {len(coins)} coins")
print(f"  {'Metric':<20} {'BASELINE (1 model)':>18} {'ROUTER (3 models)':>18} {'Delta':>10}")
print(f"  {'-'*65}")
for k, label in [("trades","Trades"),("wr","WR %"),("pnl","PnL $"),("pf","PF"),
                  ("lwr","LONG WR %"),("swr","SHORT WR %")]:
    bv = b[k]; rv = r[k]
    if isinstance(bv, float):
        print(f"  {label:<20} {bv:>18.1f} {rv:>18.1f} {rv-bv:>+10.1f}")
    else:
        print(f"  {label:<20} {bv:>18,} {rv:>18,} {rv-bv:>+10,}")

print(f"\n  Per-Regime PnL:")
for reg in [0,1,2,3]:
    bp = b["reg_pnl"][reg]; rp = r["reg_pnl"][reg]
    delta = rp["pnl"] - bp["pnl"]
    print(f"  {rn[reg]:<15}: base=${bp['pnl']:>+8.0f} ({bp['trades']:>5}t)  router=${rp['pnl']:>+8.0f} ({rp['trades']:>5}t)  Δ={delta:>+8.0f}")

print(f"\n  Yearly PnL:")
for y in sorted(b["yearly"]):
    bp = b["yearly"][y]["pnl"]; rp = r["yearly"][y]["pnl"]
    print(f"  {y}: base=${bp:>+8.0f}  router=${rp:>+8.0f}  Δ={rp-bp:>+8.0f}")
print(f"{'='*70}")
