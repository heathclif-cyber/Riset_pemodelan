"""
GENUINE OOF Extended Backtest — Purged CV (retrain LGBM per fold)
Compares: A=BASELINE  B=HMM GATE LSTM  C=POS SIZE MULT + HMM GATE

CRITICAL: LGBM retrained per fold. Model NEVER sees test data.
This is the ONLY valid methodology for out-of-sample evaluation.
"""
import sys, json, joblib, numpy as np, pandas as pd, lightgbm as lgb
from pathlib import Path
ROOT = Path(__file__).parent.parent; sys.path.insert(0, str(ROOT))
import config
config.LSTM_CONFIRMATION_ENABLED = True
config.LSTM_FLAT_REVIEW_ENABLED = True
from config import *
from pipeline.backtest_utils import hierarchical_predict, compute_guardian_static_array, _last_size_mult
from core.evaluator import simulate_trades_swing
from pipeline.shared import build_purged_folds
import pipeline.backtest_utils as btu

C = ROOT / "data" / "coinank"

LGBM_PARAMS = {'objective': 'multiclass', 'num_class': 3, 'n_estimators': 500,
               'learning_rate': 0.05, 'max_depth': 6, 'num_leaves': 31,
               'min_child_samples': 50, 'subsample': 0.8, 'colsample_bytree': 0.8,
               'verbose': -1, 'n_jobs': -1, 'random_state': 42}

def add_pos(df, coin):
    oi_p = C / f"{coin}_oi.parquet"; lsp_p = C / f"{coin}_ls_position.parquet"
    if not oi_p.exists(): return df
    oi = pd.read_parquet(oi_p).sort_index()
    lsp = pd.read_parquet(lsp_p).sort_index() if lsp_p.exists() else None
    daily = pd.DataFrame(index=oi.index)
    if lsp is not None and "top_trader_position_ls" in lsp.columns:
        ls = lsp["top_trader_position_ls"]
        lm = ls.rolling(20).mean(); ls_s = ls.rolling(20).std().clip(lower=1e-8)
        daily["ls_extreme"] = ((ls - lm) / ls_s).abs().gt(2.0).astype(float) if ls_s.sum() > 0 else 0.0
    df["_d"] = pd.to_datetime(df.index.date, utc=True)
    daily["_d"] = pd.to_datetime(daily.index.date, utc=True)
    daily = daily.dropna(subset=["_d"]).set_index("_d")
    daily = daily[~daily.index.duplicated(keep="last")]
    df["pos_extreme"] = daily["ls_extreme"].reindex(df["_d"]).ffill().fillna(0).values if "ls_extreme" in daily.columns else 0.0
    return df.drop(columns=["_d"])

def run_fold(df_test, fold_lgbm, lstm, lsc, lstm_f, guard, gs, gst, pos_on, hmm_on):
    """Run ONE test fold with ONE config."""
    config.POSITIONING_ENGINE_ENABLED = pos_on
    config.HMM_GATE_LSTM_ENABLED = hmm_on
    btu.SMART_ENTRY_MODE = "disabled"
    n = len(df_test)
    X = np.zeros((n, len(lstm_f)))
    for i, col in enumerate(lstm_f):
        if col in df_test.columns: X[:, i] = df_test[col].ffill().fillna(0).values
    feat_cols = [c for c in fold_lgbm.feature_name_ if c in df_test.columns]
    yp, cf = hierarchical_predict(None, fold_lgbm, lstm, lsc, X, feat_cols, [], df_test,
                                   trend_alignment_enabled=False, regime_aware_alignment=True)
    sm = btu._last_size_mult.copy() if btu._last_size_mult is not None else np.ones(n)
    below = (yp != 1) & (cf < CONFIDENCE_THRESHOLD_ENTRY); yp[below] = 1

    Xg = compute_guardian_static_array(df_test, gst)
    atr = df_test["atr_14_h1"].values if "atr_14_h1" in df_test.columns else np.ones(n)
    c = df_test["close"].values; h = df_test["high"].values if "high" in df_test.columns else c
    l = df_test["low"].values if "low" in df_test.columns else c
    sh = df_test["h4_swing_high"].values if "h4_swing_high" in df_test.columns else np.full(n, np.nan)
    sl = df_test["h4_swing_low"].values if "h4_swing_low" in df_test.columns else np.full(n, np.nan)

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

    trades = []
    for t in r.get("trades", []):
        bar_in = t.get("bar_in", 0)
        entry_sm = sm[bar_in] if bar_in < len(sm) else 1.0
        t["net_pnl"] = t.get("net_pnl", 0) * entry_sm
        t["size_mult"] = entry_sm
        trades.append(t)
    return trades

def compute_stats(trades):
    n = len(trades); wins = [t for t in trades if t.get("net_pnl",0)>0]
    pnl = sum(t.get("net_pnl",0) for t in trades)
    gw = sum(t["net_pnl"] for t in wins)
    gl = abs(sum(t["net_pnl"] for t in trades if t.get("net_pnl",0)<=0))
    lt = [t for t in trades if t.get("direction")=="LONG"]
    st = [t for t in trades if t.get("direction")=="SHORT"]
    months = {}
    for t in trades:
        if t.get("timestamp") is None: continue
        m = pd.Timestamp(t["timestamp"]).strftime("%Y-%m")
        months[m] = months.get(m, {"pnl":0,"trades":0,"wins":0})
        months[m]["pnl"] += t.get("net_pnl",0)
        months[m]["trades"] += 1
        if t.get("net_pnl",0) > 0: months[m]["wins"] += 1
    neg_m = sum(1 for d in months.values() if d["pnl"]<0)
    mpnl = [d["pnl"] for d in months.values()]
    yearly = {}
    for m, d in months.items():
        y = m[:4]; yearly[y] = yearly.get(y, {"pnl":0,"trades":0})
        yearly[y]["pnl"] += d["pnl"]; yearly[y]["trades"] += d["trades"]
    sl_hits = sum(1 for t in trades if str(t.get("outcome","")).lower()=="loss")
    gx = sum(1 for t in trades if "guardian" in str(t.get("outcome","")).lower())
    gx_wr = sum(1 for t in trades if "guardian" in str(t.get("outcome","")).lower() and t.get("net_pnl",0)>0)/gx*100 if gx else 0
    max_cl = cur = 0
    for t in sorted(trades, key=lambda x: str(x.get("timestamp",""))):
        if t.get("net_pnl",0) <= 0: cur += 1; max_cl = max(max_cl, cur)
        else: cur = 0
    return dict(trades=n, wr=len(wins)/n*100 if n else 0, pnl=pnl, pf=gw/gl if gl else 0,
                lwr=len([t for t in lt if t.get("net_pnl",0)>0])/len(lt)*100 if lt else 0,
                swr=len([t for t in st if t.get("net_pnl",0)>0])/len(st)*100 if st else 0,
                sl_hits=sl_hits, sl_pct=sl_hits/n*100 if n else 0,
                gx=gx, gx_wr=gx_wr, neg_m=neg_m, n_mo=len(months),
                max_cl=max_cl, yearly=yearly, months=months,
                mean_mpnl=np.mean(mpnl) if mpnl else 0, std_mpnl=np.std(mpnl) if mpnl else 0)

# ── Load fixed models ────────────────────────────────────────────────────
print("Loading models...")
lstm_f = json.load(open(MODEL_DIR / "feature_cols_lstm_temporal.json"))
guard = joblib.load(MODEL_DIR / "guardian_best.pkl"); gs = joblib.load(MODEL_DIR / "guardian_scaler.pkl")
gf = json.load(open(MODEL_DIR / "guardian_feature_cols.json"))
gst = [c for c in gf if c not in set(GUARDIAN_DYNAMIC_FEATURES)]
lgbm_feats = json.load(open(MODEL_DIR / "feature_cols_v2.json"))
from core.models import load_lstm
lstm = load_lstm(MODEL_DIR / "lstm_best.pt", device="cpu")
lsc = joblib.load(MODEL_DIR / "lstm_scaler.pkl")

# ── Main loop ────────────────────────────────────────────────────────────
coins = TRAINING_COINS
all_trades = {"A": [], "B": [], "C": []}
total_folds = 0; resized_total = 0

print(f"\n{'='*65}")
print(f"  GENUINE OOF EXTENDED BACKTEST — Purged CV, Retrain per Fold")
print(f"  {len(coins)} coins, {N_FOLDS}-fold purged CV")
print(f"  A=BASELINE  B=HMM GATE  C=SIZE MULT + HMM GATE")
print(f"{'='*65}\n")

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
    df = add_pos(df, coin)

    ts_index = pd.DatetimeIndex(df.index)
    folds = build_purged_folds(ts_index, N_FOLDS, PURGE_GAP_BARS)

    for fi, (tr_idx, te_idx) in enumerate(folds):
        if len(te_idx) < 100: continue
        df_tr = df.iloc[tr_idx]; df_te = df.iloc[te_idx]
        feat_cols = [c for c in lgbm_feats if c in df_tr.columns]
        X_tr = df_tr[feat_cols].ffill().fillna(0)
        y_tr = df_tr["label"].map(LABEL_MAP).values.astype(np.int64)
        if len(np.unique(y_tr)) < 3: continue

        # Retrain LGBM from scratch for THIS fold (GENUINE OOF)
        fold_model = lgb.LGBMClassifier(**LGBM_PARAMS)
        fold_model.fit(X_tr, y_tr)

        # Add timestamp for tracking
        df_te = df_te.copy()
        trades_a = run_fold(df_te, fold_model, lstm, lsc, lstm_f, guard, gs, gst, False, False)
        for t in trades_a: t["coin"] = coin; t["timestamp"] = df_te.index[t.get("bar_in",0)] if t.get("bar_in",0) < len(df_te) else None
        all_trades["A"].extend(trades_a)

        trades_b = run_fold(df_te, fold_model, lstm, lsc, lstm_f, guard, gs, gst, False, True)
        for t in trades_b: t["coin"] = coin; t["timestamp"] = df_te.index[t.get("bar_in",0)] if t.get("bar_in",0) < len(df_te) else None
        all_trades["B"].extend(trades_b)

        trades_c = run_fold(df_te, fold_model, lstm, lsc, lstm_f, guard, gs, gst, True, True)
        for t in trades_c: t["coin"] = coin; t["timestamp"] = df_te.index[t.get("bar_in",0)] if t.get("bar_in",0) < len(df_te) else None
        resized_total += sum(1 for t in trades_c if t.get("size_mult",1.0) < 1.0)
        all_trades["C"].extend(trades_c)

        total_folds += 1

    print(f"  [{ci+1:>2}/{len(coins)}] {coin:<15} {len(folds)} folds done")

# ── Scorecard ────────────────────────────────────────────────────────────
sa = compute_stats(all_trades["A"])
sb = compute_stats(all_trades["B"])
sc = compute_stats(all_trades["C"])

print(f"\n{'='*70}")
print(f"  GENUINE OOF SCORECARD — {total_folds} folds purged CV, {len(coins)} coins")
print(f"  {'Metric':<22} {'A: BASELINE':>12} {'B: HMM GATE':>12} {'C: SIZE+HMM':>12}")
print(f"  {'-'*60}")
for k, label in [("trades","Total Trades"),("wr","Win Rate %"),("pnl","Net PnL $"),
                  ("pf","Profit Factor"),("lwr","LONG WR %"),("swr","SHORT WR %"),
                  ("sl_hits","SL Hits"),("sl_pct","SL Hit Rate %"),
                  ("gx_wr","Guardian Exit WR%"),("neg_m","Negative Months"),
                  ("max_cl","Max Cons Loss"),("n_mo","Total Months")]:
    av = sa[k]; bv = sb[k]; cv = sc[k]
    if isinstance(av, float):
        print(f"  {label:<22} {av:>12.1f} {bv:>12.1f} {cv:>12.1f}")
    else:
        print(f"  {label:<22} {av:>12,} {bv:>12,} {cv:>12,}")

print(f"\n  Yearly PnL:")
print(f"  {'Year':<8} {'A: BASELINE':>12} {'B: HMM GATE':>12} {'C: SIZE+HMM':>12}")
all_years = sorted(set(list(sa["yearly"].keys()) + list(sb["yearly"].keys()) + list(sc["yearly"].keys())))
for y in all_years:
    ay = sa["yearly"].get(y, {}).get("pnl", 0)
    by_ = sb["yearly"].get(y, {}).get("pnl", 0)
    cy = sc["yearly"].get(y, {}).get("pnl", 0)
    print(f"  {y:<8} {ay:>+12.0f} {by_:>+12.0f} {cy:>+12.0f}")

print(f"\n  Size reduced trades (C): {resized_total} / {sc['trades']} ({resized_total/max(sc['trades'],1)*100:.1f}%)")
print(f"  Delta A->B (HMM gate):     PnL {sb['pnl']-sa['pnl']:+.0f}")
print(f"  Delta B->C (size mult):    PnL {sc['pnl']-sb['pnl']:+.0f}")
print(f"  Delta A->C (total):        PnL {sc['pnl']-sa['pnl']:+.0f}")
print(f"{'='*70}")
