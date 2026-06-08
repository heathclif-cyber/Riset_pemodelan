"""
A/B Test: Position Sizing (Kelly + Regime)
Compares BASELINE vs SIZED in purged CV OOF
"""
import sys, json, joblib, numpy as np, pandas as pd, lightgbm as lgb
from pathlib import Path
ROOT = Path(__file__).parent.parent; sys.path.insert(0, str(ROOT))
import config
config.LSTM_CONFIRMATION_ENABLED = True
from config import *
from pipeline.backtest_utils import hierarchical_predict, compute_guardian_static_array
from core.evaluator import simulate_trades_swing
from core.position_sizing import PositionSizer, DEFAULT_SIZER
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

def run_backtest(coins, use_sizing):
    sizer = PositionSizer(win_rate=0.514, avg_win=0.75, avg_loss=0.60,
                          base_risk_pct=0.02, max_kelly_mult=2.0)
    trades = []; equity = 100.0; equity_curve = []

    for coin in coins:
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

            feat_cols = [c for c in lgbm_f if c in df_tr.columns]
            X_tr = df_tr[feat_cols].ffill().fillna(0)
            y_tr = df_tr["label"].map(LABEL_MAP).values.astype(np.int64)
            if len(np.unique(y_tr)) < 3: continue

            fold_model = lgb.LGBMClassifier(**LGBM_PARAMS)
            fold_model.fit(X_tr, y_tr)

            n_te = len(df_te)
            X_te = np.zeros((n_te, len(feat_cols)))
            for i, col in enumerate(feat_cols):
                if col in df_te.columns: X_te[:, i] = df_te[col].ffill().fillna(0).values
            X_l = np.zeros((n_te, len(lstm_f)))
            for i, col in enumerate(lstm_f):
                if col in df_te.columns: X_l[:, i] = df_te[col].ffill().fillna(0).values

            yp, cf = hierarchical_predict(None, fold_model, lstm_h1, lsc, X_l, feat_cols, [], df_te,
                                           trend_alignment_enabled=False, regime_aware_alignment=True)
            below = (yp != 1) & (cf < CONFIDENCE_THRESHOLD_ENTRY); yp[below] = 1

            Xg = compute_guardian_static_array(df_te, gst)
            atr = df_te["atr_14_h1"].values if "atr_14_h1" in df_te.columns else np.ones(n_te)
            c = df_te["close"].values; h = df_te["high"].values if "high" in df_te.columns else c
            l = df_te["low"].values if "low" in df_te.columns else c
            sh = df_te["h4_swing_high"].values if "h4_swing_high" in df_te.columns else np.full(n_te, np.nan)
            sl = df_te["h4_swing_low"].values if "h4_swing_low" in df_te.columns else np.full(n_te, np.nan)

            effective_modal = MODAL_PER_TRADE
            if use_sizing:
                hmm = int(df_te["hmm_regime_enc"].iloc[0]) if len(df_te) > 0 else 1
                mult = sizer.get_multiplier(hmm)
                effective_modal = MODAL_PER_TRADE * mult

            r = simulate_trades_swing(
                y_pred=yp, close=c, high=h, low=l, atr=atr, h4_swing_highs=sh, h4_swing_lows=sl,
                modal=effective_modal, leverage=LEVERAGE_SIM[0],
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
            trades.extend(r.get("trades", []))

    return trades


coins = TRAINING_COINS[:5]
print("Running BASELINE...")
base_trades = run_backtest(coins, False)
print("Running KELLY+REGIME...")
sized_trades = run_backtest(coins, True)

def stats(trades):
    n = len(trades); wins = [t for t in trades if t.get("net_pnl",0)>0]
    pnl = sum(t.get("net_pnl",0) for t in trades)
    gw = sum(t["net_pnl"] for t in wins)
    gl = abs(sum(t["net_pnl"] for t in trades if t.get("net_pnl",0)<=0))
    lt = [t for t in trades if t.get("direction")=="LONG"]
    st = [t for t in trades if t.get("direction")=="SHORT"]

    # Regime breakdown
    reg_pnl = {}
    for reg in [0,1,2,3]:
        rt = [t for t in trades if t.get("regime")==reg]
        reg_pnl[reg] = sum(t.get("net_pnl",0) for t in rt) if rt else 0

    # Yearly breakdown
    yearly = {}
    for t in trades:
        if t.get("timestamp") is None: continue
        y = str(pd.Timestamp(t["timestamp"]).year)
        yearly[y] = yearly.get(y, {"pnl":0,"trades":0})
        yearly[y]["pnl"] += t.get("net_pnl",0)
        yearly[y]["trades"] += 1

    return dict(trades=n, wr=len(wins)/n*100 if n else 0, pnl=pnl,
                pf=gw/gl if gl else 0, yearly=yearly, reg_pnl=reg_pnl,
                lwr=len([t for t in lt if t.get("net_pnl",0)>0])/len(lt)*100 if lt else 0,
                swr=len([t for t in st if t.get("net_pnl",0)>0])/len(st)*100 if st else 0)

b = stats(base_trades); s = stats(sized_trades)
reg_names = {0:"TREND_DN", 1:"RANGE_LO", 2:"RANGE_HI", 3:"TREND_UP"}

print(f"\n{'='*65}")
print(f"  KELLY + REGIME SIZING — Purged CV OOF, 5 coins")
print(f"  Kelly Half-f: {DEFAULT_SIZER.half_kelly:.3f} | Mult: {DEFAULT_SIZER.kelly_mult:.2f}x")
print(f"  TRENDING size: 0.50x | RANGING size: 1.00x")
print(f"  {'Metric':<20} {'BASELINE':>12} {'KELLY+REGIME':>15} {'Delta':>10}")
print(f"  {'-'*58}")
for k, label in [("trades","Trades"),("wr","WR %"),("pnl","PnL $"),("pf","PF"),
                  ("lwr","LONG WR %"),("swr","SHORT WR %")]:
    bv = b[k]; sv = s[k]
    if isinstance(bv, float):
        print(f"  {label:<20} {bv:>12.1f} {sv:>15.1f} {sv-bv:>+10.1f}")
    else:
        print(f"  {label:<20} {bv:>12,} {sv:>15,} {sv-bv:>+10,}")

print(f"\n  Regime PnL breakdown:")
for reg in [0,1,2,3]:
    bp = b["reg_pnl"][reg]; sp = s["reg_pnl"][reg]
    print(f"  {reg_names[reg]:<15}: base=${bp:>+8.0f}  sized=${sp:>+8.0f}")

print(f"\n  Yearly PnL:")
for y in sorted(b["yearly"]):
    bp = b["yearly"][y]["pnl"]; sp = s["yearly"][y]["pnl"]
    print(f"  {y}: base=${bp:>+8.0f}  sized=${sp:>+8.0f}  ({sp/bp-1:+.0%})" if bp else f"  {y}: base=${bp:>+8.0f}  sized=${sp:>+8.0f}")
print(f"{'='*65}")
