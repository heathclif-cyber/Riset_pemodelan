"""
Extended Purged CV Backtest — Training extended to Mar 2026
Combines training (Jan 2020 - Oct 2025) + old holdout (Nov 2025 - Mar 2026)
TRAIN_CUTOFF_DATE = 2026-04-01
"""
import sys, json, joblib, numpy as np, pandas as pd, lightgbm as lgb
from pathlib import Path
ROOT = Path(__file__).parent.parent; sys.path.insert(0, str(ROOT))
import config
config.LSTM_CONFIRMATION_ENABLED = True
config.LSTM_FLAT_REVIEW_ENABLED = True
from config import *
from pipeline.backtest_utils import hierarchical_predict, compute_guardian_static_array
from core.evaluator import simulate_trades_swing
from pipeline.shared import build_purged_folds
import pipeline.backtest_utils as btu

LGBM_PARAMS = {'objective': 'multiclass', 'num_class': 3, 'n_estimators': 500,
               'learning_rate': 0.05, 'max_depth': 6, 'num_leaves': 31,
               'min_child_samples': 50, 'subsample': 0.8, 'colsample_bytree': 0.8,
               'verbose': -1, 'n_jobs': -1, 'random_state': 42}

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
btu.SMART_ENTRY_MODE = "disabled"

# ── Load extended data ───────────────────────────────────────────────────
print("Loading extended training data (2020 -> 2026-03)...")
coins = TRAINING_COINS
all_trades = []; total_folds = 0; total_months = set()

for ci, coin in enumerate(coins):
    fp_train = LABEL_DIR / f"{coin}_features_v3.parquet"
    fp_holdout = HOLDOUT_DIR / "labeled" / f"{coin}_features_v3.parquet"
    rp_train = LABEL_DIR / f"{coin}_regime_h1.parquet"
    rp_holdout = HOLDOUT_DIR / "labeled" / f"{coin}_regime_h1.parquet"

    if not fp_train.exists(): continue

    # Load training data (before old cutoff)
    df_train = pd.read_parquet(fp_train).sort_index()
    df_train = df_train[df_train.index < TRAIN_CUTOFF_DATE]  # now Mar 2026

    # Load old holdout data (Nov 2025 -> Mar 2026) — now part of training
    if fp_holdout.exists():
        df_holdout = pd.read_parquet(fp_holdout).sort_index()
        df_holdout = df_holdout[df_holdout.index < TRAIN_CUTOFF_DATE]
        # Combine: training + old holdout
        df = pd.concat([df_train, df_holdout])
        df = df[~df.index.duplicated(keep="last")].sort_index()
    else:
        df = df_train

    # Merge regime data
    regime_parts = []
    if rp_train.exists():
        reg = pd.read_parquet(rp_train)
        if "hmm_regime_enc" in reg.columns:
            regime_parts.append(reg[["hmm_regime_enc"]])
    if rp_holdout.exists():
        reg = pd.read_parquet(rp_holdout)
        if "hmm_regime_enc" in reg.columns:
            regime_parts.append(reg[["hmm_regime_enc"]])
    if regime_parts:
        reg_all = pd.concat(regime_parts)
        reg_all = reg_all[~reg_all.index.duplicated(keep="last")].sort_index()
        if "hmm_regime_enc" in df.columns: df = df.drop(columns=["hmm_regime_enc"])
        df = df.join(reg_all, how="left")
        df["hmm_regime_enc"] = df["hmm_regime_enc"].fillna(1).astype("int32")
    else:
        df["hmm_regime_enc"] = 1

    df = df[df["label"].astype(str).isin(LABEL_MAP)].copy()
    if len(df) < 500: continue

    # Track months
    for ts in df.index:
        total_months.add(ts.strftime("%Y-%m"))

    ts_index = pd.DatetimeIndex(df.index)
    folds = build_purged_folds(ts_index, N_FOLDS, PURGE_GAP_BARS)

    for fi, (tr_idx, te_idx) in enumerate(folds):
        if len(te_idx) < 100: continue
        df_tr = df.iloc[tr_idx]; df_te = df.iloc[te_idx]

        feat_cols = [c for c in lgbm_feats if c in df_tr.columns]
        X_tr = df_tr[feat_cols].ffill().fillna(0)
        y_tr = df_tr["label"].map(LABEL_MAP).values.astype(np.int64)
        if len(np.unique(y_tr)) < 3: continue

        # Retrain LGBM per fold (GENUINE OOF)
        fold_model = lgb.LGBMClassifier(**LGBM_PARAMS)
        fold_model.fit(X_tr, y_tr)

        # Predict
        n_te = len(df_te)
        feat_cols_te = [c for c in lgbm_feats if c in df_te.columns]
        X_te = np.zeros((n_te, len(feat_cols_te)))
        for i, col in enumerate(feat_cols_te):
            if col in df_te.columns: X_te[:, i] = df_te[col].ffill().fillna(0).values

        # LSTM input
        X_lstm = np.zeros((n_te, len(lstm_f)))
        for i, col in enumerate(lstm_f):
            if col in df_te.columns: X_lstm[:, i] = df_te[col].ffill().fillna(0).values

        yp, cf = hierarchical_predict(None, fold_model, lstm, lsc, X_lstm, feat_cols_te, [], df_te,
                                       trend_alignment_enabled=False, regime_aware_alignment=True)
        below = (yp != 1) & (cf < CONFIDENCE_THRESHOLD_ENTRY); yp[below] = 1

        Xg = compute_guardian_static_array(df_te, gst)
        atr = df_te["atr_14_h1"].values if "atr_14_h1" in df_te.columns else np.ones(n_te)
        c = df_te["close"].values
        h = df_te["high"].values if "high" in df_te.columns else c
        l = df_te["low"].values if "low" in df_te.columns else c
        sh = df_te["h4_swing_high"].values if "h4_swing_high" in df_te.columns else np.full(n_te, np.nan)
        sl = df_te["h4_swing_low"].values if "h4_swing_low" in df_te.columns else np.full(n_te, np.nan)

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
        all_trades.extend(r.get("trades", []))
        total_folds += 1

    print(f"  [{ci+1:>2}/{len(coins)}] {coin:<15} {len(df):,} bars, {len(folds)} folds")

# ── Scorecard ────────────────────────────────────────────────────────────
n = len(all_trades); wins = [t for t in all_trades if t.get("net_pnl",0)>0]
pnl = sum(t.get("net_pnl",0) for t in all_trades)
gw = sum(t["net_pnl"] for t in wins)
gl = abs(sum(t["net_pnl"] for t in all_trades if t.get("net_pnl",0)<=0))
lt = [t for t in all_trades if t.get("direction")=="LONG"]
st = [t for t in all_trades if t.get("direction")=="SHORT"]
sl_hits = sum(1 for t in all_trades if str(t.get("outcome","")).lower()=="loss")
gx = sum(1 for t in all_trades if "guardian" in str(t.get("outcome","")).lower())
gx_wr = sum(1 for t in all_trades if "guardian" in str(t.get("outcome","")).lower() and t.get("net_pnl",0)>0)/gx*100 if gx else 0

months = {}
for t in all_trades:
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

max_cl = cur = 0
for t in sorted(all_trades, key=lambda x: str(x.get("timestamp",""))):
    if t.get("net_pnl",0) <= 0: cur += 1; max_cl = max(max_cl, cur)
    else: cur = 0

print(f"\n{'='*70}")
print(f"  GENUINE OOF SCORECARD — {total_folds} folds, {len(coins)} coins")
print(f"  Training: Jan 2020 -> Mar 2026 ({len(total_months)} months)")
print(f"  {'Metric':<22} {'Value':>15}")
print(f"  {'-'*40}")
for label, val in [("Total Trades",n),("Win Rate %",f"{len(wins)/n*100:.1f}%"),
    ("Net PnL $",f"${pnl:.0f}"),("Profit Factor",f"{gw/gl:.2f}" if gl else "inf"),
    ("LONG WR %",f"{len([t for t in lt if t.get('net_pnl',0)>0])/len(lt)*100:.1f}%" if lt else "N/A"),
    ("SHORT WR %",f"{len([t for t in st if t.get('net_pnl',0)>0])/len(st)*100:.1f}%" if st else "N/A"),
    ("SL Hit Rate",f"{sl_hits/n*100:.1f}%"),("Guardian Exit WR",f"{gx_wr:.1f}%"),
    ("Negative Months",f"{neg_m}/{len(months)}"),("Max Cons Loss",max_cl),
    ("$ / Month",f"${np.mean(mpnl):.0f} +/- ${np.std(mpnl):.0f}")]:
    print(f"  {label:<22} {val:>15}")

print(f"\n  Yearly PnL:")
for y in sorted(yearly):
    d = yearly[y]
    print(f"  {y}: {d['trades']:>6,} trades  PnL={d['pnl']:>+10.0f}")

# Key months
print(f"\n  Key periods:")
for label, start, end in [
    ("2021 Bull (Jan-May)", "2021-01", "2021-05"),
    ("2022 Bear (May-Jul)", "2022-05", "2022-07"),
    ("2025 Ranging (Jan-Jun)", "2025-01", "2025-06"),
    ("2025 Q4 (Oct-Dec)", "2025-10", "2025-12"),
    ("2026 Q1 (Jan-Mar)", "2026-01", "2026-03"),
]:
    pnl_p = sum(months[m]["pnl"] for m in months if start <= m <= end)
    trades_p = sum(months[m]["trades"] for m in months if start <= m <= end)
    print(f"  {label:<30}: PnL={pnl_p:>+8.0f}  Trades={trades_p:>5}")

print(f"{'='*70}")
