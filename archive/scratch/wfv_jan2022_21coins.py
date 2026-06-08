"""WFV Jan 2022 — 21 coins, 1 trending window. Tests regime router on ALTCOINS."""
import sys, json, joblib, warnings, numpy as np, pandas as pd, lightgbm as lgb
from pathlib import Path; from datetime import timedelta; from hmmlearn import hmm
ROOT = Path(__file__).parent.parent; sys.path.insert(0, str(ROOT)); warnings.filterwarnings("ignore")
import config; config.LSTM_CONFIRMATION_ENABLED = True
from config import *
from pipeline.backtest_utils import compute_guardian_static_array
from core.evaluator import simulate_trades_swing

LGBM_PARAMS = {'objective':'multiclass','num_class':3,'n_estimators':100,'learning_rate':0.05,
               'max_depth':4,'num_leaves':15,'min_child_samples':100,'verbose':-1,'n_jobs':-1,'random_state':42}
SWING = json.load(open(MODEL_DIR/"feature_cols_v2.json"))
UP_F = joblib.load(MODEL_DIR/"lgbm_regime_TRENDING_UP.pkl").feature_name_
DN_F = joblib.load(MODEL_DIR/"lgbm_regime_TRENDING_DOWN.pkl").feature_name_
guard = joblib.load(MODEL_DIR/"guardian_best.pkl"); gs = joblib.load(MODEL_DIR/"guardian_scaler.pkl")
gf = json.load(open(MODEL_DIR/"guardian_feature_cols.json"))
gst = [c for c in gf if c not in set(GUARDIAN_DYNAMIC_FEATURES)]
HM = ["log_ret_1","rsi_6","cvd_momentum_adv","volume_delta","atr_14_h1",
      "h4_trend","trend_strength","ema_21_slope_h4"]
PURGE = 36; MH = 36; TP_M = 2.0; SL_M = 1.5

test_start = pd.Timestamp("2022-01-01", tz="UTC"); test_end = test_start + timedelta(days=30)
train_end = test_start - timedelta(hours=PURGE)

coins = TRAINING_COINS; all_b = []; all_r = []
print(f"WFV Jan 2022 — 21 coins. Train until {train_end.date()}, Test {test_start.date()}-{test_end.date()}")
print()

for ci, coin in enumerate(coins):
    fp = LABEL_DIR / f"{coin}_features_v3.parquet"
    if not fp.exists(): continue
    df = pd.read_parquet(fp).sort_index(); df = df[df.index < TRAIN_CUTOFF_DATE]
    df = df[df["label"].astype(str).isin(LABEL_MAP)].copy()
    if len(df) < 500: continue

    df_tr = df[df.index <= train_end]; df_te = df[(df.index >= test_start) & (df.index < test_end)]
    if len(df_tr) < 500 or len(df_te) < 100: continue

    # HMM
    Xc = [c for c in HM if c in df_tr.columns]
    Xh = df_tr[Xc].ffill().fillna(0).values
    Xh = (Xh - Xh.mean(0)) / (Xh.std(0).clip(1e-8))
    try:
        hm_m = hmm.GaussianHMM(n_components=4, covariance_type="full", n_iter=50, random_state=42, verbose=False)
        hm_m.fit(Xh); reg_tr = hm_m.predict(Xh)
    except: continue

    # Train 3 models
    models = {}
    for reg, name, feats in [(3,"UP",UP_F),(0,"DN",DN_F),(1,"RG",SWING)]:
        mask = (reg_tr == reg) if reg in (0,3) else ((reg_tr >= 1) & (reg_tr <= 2))
        subset = df_tr[mask]
        if len(subset) < 100: models[name] = None; continue
        fc = [c for c in feats if c in subset.columns]
        if reg in (0,3):
            ho = subset["high"].values; lo = subset["low"].values
            co = subset["close"].values; ao = subset["atr_14_h1"].values
            yl = np.ones(len(subset), dtype=np.int8); d = 1 if reg == 3 else -1
            for i in range(len(subset) - MH - 1):
                tp = co[i] + d * TP_M * ao[i]; sl = co[i] - d * SL_M * ao[i]
                for j in range(i+1, min(i+MH+1, len(subset))):
                    if d == 1:
                        if ho[j] >= tp: yl[i] = 2; break
                        if lo[j] <= sl: yl[i] = 0; break
                    else:
                        if lo[j] <= tp: yl[i] = 0; break
                        if ho[j] >= sl: yl[i] = 2; break
        else:
            yl = subset["label"].map(LABEL_MAP).values.astype(np.int64)
        if len(np.unique(yl)) >= 3:
            models[name] = lgb.LGBMClassifier(**LGBM_PARAMS)
            models[name].fit(subset[fc].ffill().fillna(0), yl)
        else: models[name] = None

    # Predict on test
    Xt = [c for c in HM if c in df_te.columns]
    Xth = df_te[Xt].ffill().fillna(0).values
    Xth = (Xth - Xth.mean(0)) / (Xth.std(0).clip(1e-8))
    try: reg_te = hm_m.predict(Xth)
    except: continue
    n_te = len(df_te)
    yp_b = np.ones(n_te, dtype=np.int64); cf_b = np.full(n_te, 0.5)
    yp_r = np.ones(n_te, dtype=np.int64); cf_r = np.full(n_te, 0.5)

    for i in range(n_te):
        rg = reg_te[i]; row = df_te.iloc[i:i+1]
        if models.get("RG"):
            fc = [c for c in SWING if c in row.columns]
            pb = models["RG"].predict_proba(row[fc].ffill().fillna(0))[0]
            yp_b[i] = np.argmax(pb); cf_b[i] = float(pb[yp_b[i]])

        m = models.get("RG"); fl = SWING
        if rg == 3 and models.get("UP"): m = models["UP"]; fl = UP_F
        elif rg == 0 and models.get("DN"): m = models["DN"]; fl = DN_F
        if m:
            fc = [c for c in fl if c in row.columns]
            pb = m.predict_proba(row[fc].ffill().fillna(0))[0]
            yp_r[i] = np.argmax(pb); cf_r[i] = float(pb[yp_r[i]])

    below = (yp_b != 1) & (cf_b < CONFIDENCE_THRESHOLD_ENTRY); yp_b[below] = 1
    below = (yp_r != 1) & (cf_r < CONFIDENCE_THRESHOLD_ENTRY); yp_r[below] = 1

    Xg = compute_guardian_static_array(df_te, gst)
    atr = df_te["atr_14_h1"].values if "atr_14_h1" in df_te.columns else np.ones(n_te)
    c = df_te["close"].values; h = df_te["high"].values if "high" in df_te.columns else c
    l = df_te["low"].values if "low" in df_te.columns else c
    sh = df_te["h4_swing_high"].values if "h4_swing_high" in df_te.columns else np.full(n_te, np.nan)
    sl = df_te["h4_swing_low"].values if "h4_swing_low" in df_te.columns else np.full(n_te, np.nan)

    for yp, cf, dest in [(yp_b, cf_b, all_b), (yp_r, cf_r, all_r)]:
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
            t["coin"] = coin; bar_in = t.get("bar_in", 0)
            t["timestamp"] = df_te.index[bar_in] if bar_in < len(df_te) else None
            t["regime"] = int(reg_te[bar_in]) if bar_in < len(reg_te) else 1
        dest.extend(r.get("trades", []))

    nb = sum(1 for t in all_b if t.get("coin") == coin)
    nr = sum(1 for t in all_r if t.get("coin") == coin)
    bp = sum(t.get("net_pnl", 0) for t in all_b if t.get("coin") == coin)
    rp = sum(t.get("net_pnl", 0) for t in all_r if t.get("coin") == coin)
    reg_dist = dict(zip(*np.unique(reg_te, return_counts=True)))
    print(f"  [{ci+1:>2}/{len(coins)}] {coin:<15} BASE:{nb:>4}t ${bp:>+6.1f}  ROUTER:{nr:>4}t ${rp:>+6.1f}  reg={reg_dist}")

print()
bn = len(all_b); bw = sum(1 for t in all_b if t.get("net_pnl", 0) > 0)
bp = sum(t.get("net_pnl", 0) for t in all_b)
rn = len(all_r); rw = sum(1 for t in all_r if t.get("net_pnl", 0) > 0)
rp = sum(t.get("net_pnl", 0) for t in all_r)
print(f"TOTAL BASE:   {bn}t WR={bw/bn*100:.1f}% PnL=${bp:.1f}")
print(f"TOTAL ROUTER: {rn}t WR={rw/rn*100:.1f}% PnL=${rp:.1f}")
print(f"Delta: PnL=${rp-bp:+.1f}  Trades={rn-bn:+}")
