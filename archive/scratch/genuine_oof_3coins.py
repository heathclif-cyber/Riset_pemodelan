"""Genuine OOF: 3 coins, all 3 LGBM retrained per fold"""
import sys, json, joblib, numpy as np, pandas as pd, lightgbm as lgb
from pathlib import Path
ROOT = Path(__file__).parent.parent; sys.path.insert(0, str(ROOT))
import config; config.LSTM_CONFIRMATION_ENABLED = True
from config import *
from pipeline.backtest_utils import compute_guardian_static_array
from core.evaluator import simulate_trades_swing
from pipeline.shared import build_purged_folds

LGBM_PARAMS = {'objective': 'multiclass', 'num_class': 3, 'n_estimators': 200,
               'learning_rate': 0.05, 'max_depth': 5, 'num_leaves': 31,
               'min_child_samples': 50, 'subsample': 0.8, 'colsample_bytree': 0.8,
               'verbose': -1, 'n_jobs': -1, 'random_state': 42}

SWING_FEATS = json.load(open(MODEL_DIR / "feature_cols_v2.json"))
up = joblib.load(MODEL_DIR / "lgbm_regime_TRENDING_UP.pkl")
dn = joblib.load(MODEL_DIR / "lgbm_regime_TRENDING_DOWN.pkl")
TREND_UP_FEATS = up.feature_name_; TREND_DN_FEATS = dn.feature_name_
del up, dn

guard = joblib.load(MODEL_DIR / "guardian_best.pkl"); gs = joblib.load(MODEL_DIR / "guardian_scaler.pkl")
gf = json.load(open(MODEL_DIR / "guardian_feature_cols.json"))
gst = [c for c in gf if c not in set(GUARDIAN_DYNAMIC_FEATURES)]

MAX_HOLD = 36; TP_MULT = 2.0; SL_MULT = 1.5
coins = TRAINING_COINS[:3]
all_trades_base = []; all_trades_router = []

for ci, coin in enumerate(coins):
    fp = LABEL_DIR / f"{coin}_features_v3.parquet"
    rp = LABEL_DIR / f"{coin}_regime_h1.parquet"
    if not fp.exists(): continue
    df = pd.read_parquet(fp).sort_index(); df = df[df.index < TRAIN_CUTOFF_DATE]
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
        df_tr = df.iloc[tr_idx]; df_te = df.iloc[te_idx]; n_te = len(df_te)
        ho, lo, co, ao = df_tr["high"].values, df_tr["low"].values, df_tr["close"].values, df_tr["atr_14_h1"].values

        # 1. SWING
        y_tr = df_tr["label"].map(LABEL_MAP).values.astype(np.int64)
        xs = [c for c in SWING_FEATS if c in df_tr.columns]; m_s = lgb.LGBMClassifier(**LGBM_PARAMS)
        m_s.fit(df_tr[xs].ffill().fillna(0), y_tr)
        xst = [c for c in SWING_FEATS if c in df_te.columns]
        ps = m_s.predict_proba(df_te[xst].ffill().fillna(0)); yp_s = np.argmax(ps, axis=1)

        # 2. TRENDING_UP (direction=1)
        yu = np.ones(len(df_tr), dtype=np.int8)
        for i in range(len(df_tr) - MAX_HOLD - 1):
            tp = co[i] + TP_MULT * ao[i]; sl = co[i] - SL_MULT * ao[i]
            for j in range(i+1, min(i+MAX_HOLD+1, len(df_tr))):
                if ho[j] >= tp: yu[i] = 2; break
                if lo[j] <= sl: yu[i] = 0; break
        xu = [c for c in TREND_UP_FEATS if c in df_tr.columns]; m_u = lgb.LGBMClassifier(**LGBM_PARAMS)
        m_u.fit(df_tr[xu].ffill().fillna(0), yu)
        xut = [c for c in TREND_UP_FEATS if c in df_te.columns]
        pu = m_u.predict_proba(df_te[xut].ffill().fillna(0))

        # 3. TRENDING_DOWN (direction=-1)
        yd = np.ones(len(df_tr), dtype=np.int8)
        for i in range(len(df_tr) - MAX_HOLD - 1):
            tp = co[i] - TP_MULT * ao[i]; sl = co[i] + SL_MULT * ao[i]
            for j in range(i+1, min(i+MAX_HOLD+1, len(df_tr))):
                if lo[j] <= tp: yd[i] = 0; break
                if ho[j] >= sl: yd[i] = 2; break
        xd = [c for c in TREND_DN_FEATS if c in df_tr.columns]; m_d = lgb.LGBMClassifier(**LGBM_PARAMS)
        m_d.fit(df_tr[xd].ffill().fillna(0), yd)
        xdt = [c for c in TREND_DN_FEATS if c in df_te.columns]
        pd_prob = m_d.predict_proba(df_te[xdt].ffill().fillna(0))

        # A) BASELINE: swing everywhere
        yp_b = yp_s.copy(); cf_b = np.max(ps, axis=1)
        below = (yp_b != 1) & (cf_b < CONFIDENCE_THRESHOLD_ENTRY); yp_b[below] = 1

        # B) ROUTER: per-regime
        yp_r = np.ones(n_te, dtype=np.int64); cf_r = np.full(n_te, 1.0/3)
        for i in range(n_te):
            rg = int(df_te["hmm_regime_enc"].iloc[i])
            if rg == 3: yp_r[i] = np.argmax(pu[i]); cf_r[i] = pu[i, yp_r[i]]
            elif rg == 0: yp_r[i] = np.argmax(pd_prob[i]); cf_r[i] = pd_prob[i, yp_r[i]]
            else: yp_r[i] = yp_s[i]; cf_r[i] = ps[i, yp_s[i]]
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

    print(f"  [{ci+1}/3] {coin} done ({len(folds)} folds)")

def stats(trades):
    n = len(trades); wins = [t for t in trades if t.get("net_pnl",0)>0]
    pnl = sum(t.get("net_pnl",0) for t in trades)
    gw = sum(t["net_pnl"] for t in wins)
    gl = abs(sum(t["net_pnl"] for t in trades if t.get("net_pnl",0)<=0))
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
    return dict(trades=n, wr=len(wins)/n*100, pnl=pnl, pf=gw/gl if gl else 0,
                reg_pnl=reg_pnl, yearly=yearly)

b = stats(all_trades_base); r = stats(all_trades_router)
rn = {0:"TREND_DN",1:"RANGE_LO",2:"RANGE_HI",3:"TREND_UP"}

print(f"\n{'='*65}")
print(f"  GENUINE OOF — 3 coins, ALL models retrained per fold")
print(f"  {'Metric':<20} {'BASELINE':>12} {'ROUTER':>12} {'Delta':>10}")
print(f"  {'-'*55}")
for k, label in [("trades","Trades"),("wr","WR %"),("pnl","PnL $"),("pf","PF")]:
    bv = b[k]; rv = r[k]
    if isinstance(bv, float): print(f"  {label:<20} {bv:>12.1f} {rv:>12.1f} {rv-bv:>+10.1f}")
    else: print(f"  {label:<20} {bv:>12,} {rv:>12,} {rv-bv:>+10,}")

print(f"\n  Per-Regime:")
for reg in [0,1,2,3]:
    bp = b["reg_pnl"][reg]; rp = r["reg_pnl"][reg]
    print(f"  {rn[reg]:<15}: base=${bp['pnl']:>+8.0f} ({bp['trades']:>5}t)  router=${rp['pnl']:>+8.0f} ({rp['trades']:>5}t)")

print(f"\n  Yearly:")
for y in sorted(b["yearly"]):
    bp = b["yearly"][y]["pnl"]; rp = r["yearly"][y]["pnl"]
    print(f"  {y}: base=${bp:>+8.0f}  router=${rp:>+8.0f}  D={rp-bp:>+8.0f}")
print(f"{'='*65}")
