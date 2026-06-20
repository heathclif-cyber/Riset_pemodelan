"""TB v3 + HMM — retrain + holdout apple-to-apple vs ic32"""
import json, sys, warnings, numpy as np, pandas as pd
warnings.filterwarnings('ignore')
sys.path.insert(0, '.')
from config import *
import joblib, lightgbm as lgb
from core.features import triple_barrier_labeling
from core.evaluator import full_trading_report
from core.utils import setup_logger, ensure_utc_index
from pipeline.shared import build_purged_folds
from sklearn.metrics import f1_score, accuracy_score

logger = setup_logger("tb_v3_hmm")

RUN_NAME = "tb_lgbm_widyawardhana_v3_hmm"

# 18 multistage + HMM
TB_V3_HMM_FEATURES = [
    "etf_gbtc_change_usd", "etf_total_change_usd", "cvd_slope_h4",
    "ofi_h4_delta", "wyckoff_phase", "Sell_Liq", "atr_percentile_h1",
    "stochrsi_k", "dist_liq_50x_short", "funding_rate", "ema_7_h1",
    "dow_cos", "cvd_div_h4", "dist_swing_low", "VAH",
    "cvd_momentum_adv", "dist_from_8h_high", "ema_200_h1",
    "hmm_regime_enc",  # NEW
]

TB_PARAMS = {
    "objective": "multiclass", "num_class": 3,
    "n_estimators": 500, "learning_rate": 0.03,
    "max_depth": 5, "num_leaves": 31, "min_child_samples": 50,
    "subsample": 0.8, "colsample_bytree": 0.7,
    "class_weight": "balanced",
    "verbose": -1, "n_jobs": -1, "random_state": 42,
}

# ── Load training data ─────────────────────────────────────────────────────
print("Loading training data...")
frames = []
for sym in ALL_COINS:
    path = LABEL_DIR / f"{sym}_features_v3.parquet"
    reg_path = LABEL_DIR / f"{sym}_regime_h1.parquet"
    if not path.exists(): continue
    df = pd.read_parquet(path)
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, utc=True)
    if df.index.tz is None: df.index = df.index.tz_localize("UTC")
    df = df[df.index < TRAIN_CUTOFF_DATE]
    if df.empty: continue

    # Merge HMM
    if reg_path.exists():
        reg = pd.read_parquet(reg_path)
        if "hmm_regime_enc" in df.columns: df = df.drop(columns=["hmm_regime_enc"])
        df = df.join(reg[["hmm_regime_enc"]], how="left")
        df["hmm_regime_enc"] = df["hmm_regime_enc"].fillna(1).astype("int32")

    # TB labels
    tb = triple_barrier_labeling(df["close"], df["high"], df["low"],
                                 df["atr_14_h1"], 2.0, 1.5, 36)
    df["tb_label"] = tb.map({"SHORT":0, "FLAT":1, "LONG":2})
    df = df.dropna(subset=["tb_label"])
    if len(df) < 100: continue
    frames.append(df)

df_train = pd.concat(frames).sort_index()
y_train = df_train["tb_label"].values.astype(np.int32)
print(f"Training: {len(df_train):,} bars, SHORT={(y_train==0).sum():,} FLAT={(y_train==1).sum():,} LONG={(y_train==2).sum():,}")

# Check HMM availability
avail = [c for c in TB_V3_HMM_FEATURES if c in df_train.columns]
print(f"Features: {len(avail)}/{len(TB_V3_HMM_FEATURES)} available")
if "hmm_regime_enc" in avail:
    print(f"  HMM regime enc present: {df_train['hmm_regime_enc'].value_counts().to_dict()}")

# ── Train ────────────────────────────────────────────────────────────────────
print("\nTraining 3-class with HMM...")
X_train = df_train[avail].ffill().fillna(0)
folds = build_purged_folds(X_train.index, N_FOLDS, PURGE_GAP_BARS)
all_metrics = []; best_f1, best_model = -1.0, None

for fold, (tr_idx, val_idx) in enumerate(folds, 1):
    X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
    y_tr, y_val = y_train[tr_idx], y_train[val_idx]
    model = lgb.LGBMClassifier(**TB_PARAMS)
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
              callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(period=-1)])
    y_pred = model.predict(X_val)
    f1_per = f1_score(y_val, y_pred, average=None, labels=[0,1,2], zero_division=0)
    f1_m = float(f1_score(y_val, y_pred, average="macro", zero_division=0))
    all_metrics.append({"fold": fold, "f1_macro": round(f1_m,4), "f1_SHORT": round(float(f1_per[0]),4),
                        "f1_FLAT": round(float(f1_per[1]),4), "f1_LONG": round(float(f1_per[2]),4),
                        "iter": model.best_iteration_})
    if f1_m > best_f1: best_f1, best_model = f1_m, model
    print(f"  Fold {fold}: macro={f1_m:.4f} S={f1_per[0]:.4f} F={f1_per[1]:.4f} L={f1_per[2]:.4f} iter={model.best_iteration_}")

avg_iter = int(np.mean([m["iter"] for m in all_metrics]))
f1s = [m["f1_macro"] for m in all_metrics]
print(f"CV: F1={np.mean(f1s):.4f} +/- {np.std(f1s):.4f} | avg_iter={avg_iter}")

# Full retrain
final_params = TB_PARAMS.copy(); final_params["n_estimators"] = max(avg_iter, 100)
final_model = lgb.LGBMClassifier(**final_params)
final_model.fit(X_train, y_train)

# Save
run_dir = MODEL_DIR / "runs" / RUN_NAME; run_dir.mkdir(parents=True, exist_ok=True)
joblib.dump(final_model, run_dir / "lgbm.pkl")
with open(run_dir / f"{RUN_NAME}_features.json", "w") as f: json.dump(avail, f, indent=2)
print(f"Model saved -> {run_dir / 'lgbm.pkl'}")

# ── Holdout: apple-to-apple vs ic32 ────────────────────────────────────────
print("\n" + "="*75)
print("  HOLDOUT: TB v3+HMM vs ic32_regime_v1 (swing labels)")
print("="*75)

bl_model = joblib.load(MODEL_DIR / "lgbm_baseline.pkl")
if hasattr(bl_model, 'feature_name_') and bl_model.feature_name_:
    bl_feats = list(bl_model.feature_name_)
else:
    with open(MODEL_DIR / "feature_cols_v2.json") as f: bl_feats = json.load(f)

available = [s for s in ALL_COINS if (HOLDOUT_DIR / "labeled" / f"{s}_features_v3.parquet").exists()]
LABEL_MAP_IC32 = {"SHORT": 0, "FLAT": 1, "LONG": 2}

for THRESH in [0.45, 0.48, 0.50, 0.55]:
    print(f"\n--- thresh={THRESH} ---")
    print(f"{'Coin':<15} {'TB Tr':>7} {'TB WR':>7} {'TB PnL':>9} {'BL Tr':>7} {'BL WR':>7} {'BL PnL':>9}")
    print("-" * 67)
    tb_t, tb_pnl, tb_w = 0, 0.0, 0
    bl_t, bl_pnl, bl_w = 0, 0.0, 0

    for sym in available:
        df = pd.read_parquet(HOLDOUT_DIR / "labeled" / f"{sym}_features_v3.parquet")
        df = ensure_utc_index(df).sort_index()
        # Merge HMM for holdout
        reg_path = HOLDOUT_DIR / "labeled" / f"{sym}_regime_h1.parquet"
        if reg_path.exists():
            reg = pd.read_parquet(reg_path)
            if "hmm_regime_enc" in df.columns: df = df.drop(columns=["hmm_regime_enc"])
            df = df.join(reg[["hmm_regime_enc"]], how="left")
            df["hmm_regime_enc"] = df["hmm_regime_enc"].fillna(1).astype("int32")

        mask = df["label"].isin(LABEL_MAP_IC32); df = df[mask].copy()
        y_true = df["label"].map(LABEL_MAP_IC32).values.astype(np.int32)
        atr_arr = df["atr_14_h1"].values; close_arr = df["close"].values
        high_arr = df["high"].values; low_arr = df["low"].values
        h4_sh = df["h4_swing_high"].values if "h4_swing_high" in df.columns else None
        h4_sl = df["h4_swing_low"].values if "h4_swing_low" in df.columns else None
        h4_trend_arr = df["h4_trend"].values if "h4_trend" in df.columns else None

        # TB inference
        X_tb = np.zeros((len(df), len(avail)), dtype=np.float64)
        for idx, col in enumerate(avail):
            if col in df.columns:
                X_tb[:, idx] = df[col].ffill().fillna(0).values.astype(np.float64)
        proba_tb = final_model.predict_proba(X_tb)
        conf_tb = np.max(proba_tb, axis=1)
        y_tb = np.argmax(proba_tb, axis=1)
        y_tb[(y_tb != 1) & (conf_tb < THRESH)] = 1

        r_tb = {"trades": 0, "pnl": 0.0, "wr": 0.0}
        if (y_tb != 1).sum() > 0:
            rep = full_trading_report(
                y_pred=y_tb, y_actual=y_true, atr=atr_arr, close=close_arr,
                high=high_arr, low=low_arr, h4_swing_highs=h4_sh, h4_swing_lows=h4_sl,
                index=df.index, modal=MODAL_PER_TRADE, leverages=LEVERAGE_SIM,
                fee_per_side=FEE_PER_SIDE, slippage=SLIPPAGE_PER_SIDE,
                min_rr=SWING_LABEL_MIN_RR, min_tp_atr=SWING_LABEL_MIN_TP,
                max_sl_atr=SWING_LABEL_MAX_SL, tp_fallback_atr=TP_SL_FALLBACK_TP,
                sl_fallback_atr=TP_SL_FALLBACK_SL, max_hold=MAX_HOLDING_BARS,
                symbol=sym, confidence=conf_tb,
                guardian_enabled=False, trailing_stop_enabled=False, h4_trend=h4_trend_arr,
            )
            lev5 = rep.get("lev5x", rep); trades = lev5.get("trades", [])
            wins = sum(1 for t in trades if t.get("outcome") == "WIN")
            r_tb = {"trades": len(trades), "pnl": sum(t.get("net_pnl",0) for t in trades),
                    "wr": wins/max(len(trades),1)*100}

        tb_t += r_tb["trades"]; tb_pnl += r_tb["pnl"]; tb_w += int(r_tb["trades"]*r_tb["wr"]/100)

        # Baseline
        X_bl = np.zeros((len(df), len(bl_feats)), dtype=np.float64)
        for idx, col in enumerate(bl_feats):
            if col in df.columns:
                X_bl[:, idx] = df[col].ffill().fillna(0).values.astype(np.float64)
        proba_bl = bl_model.predict_proba(X_bl)
        conf_bl = np.max(proba_bl, axis=1); y_bl = np.argmax(proba_bl, axis=1)
        for i in range(len(y_bl)):
            if y_bl[i] == 2 and proba_bl[i,2] < LGBM_THRESHOLD_LONG: y_bl[i] = 1
            elif y_bl[i] == 0 and proba_bl[i,0] < LGBM_THRESHOLD_SHORT: y_bl[i] = 1
        y_bl[(y_bl != 1) & (conf_bl < CONFIDENCE_THRESHOLD_ENTRY)] = 1

        r_bl = {"trades": 0, "pnl": 0.0, "wr": 0.0}
        if (y_bl != 1).sum() > 0:
            rep = full_trading_report(
                y_pred=y_bl, y_actual=y_true, atr=atr_arr, close=close_arr,
                high=high_arr, low=low_arr, h4_swing_highs=h4_sh, h4_swing_lows=h4_sl,
                index=df.index, modal=MODAL_PER_TRADE, leverages=LEVERAGE_SIM,
                fee_per_side=FEE_PER_SIDE, slippage=SLIPPAGE_PER_SIDE,
                min_rr=SWING_LABEL_MIN_RR, min_tp_atr=SWING_LABEL_MIN_TP,
                max_sl_atr=SWING_LABEL_MAX_SL, tp_fallback_atr=TP_SL_FALLBACK_TP,
                sl_fallback_atr=TP_SL_FALLBACK_SL, max_hold=MAX_HOLDING_BARS,
                symbol=sym, confidence=conf_bl,
                guardian_enabled=False, trailing_stop_enabled=False, h4_trend=h4_trend_arr,
            )
            lev5 = rep.get("lev5x", rep); trades = lev5.get("trades", [])
            wins = sum(1 for t in trades if t.get("outcome") == "WIN")
            r_bl = {"trades": len(trades), "pnl": sum(t.get("net_pnl",0) for t in trades),
                    "wr": wins/max(len(trades),1)*100}

        bl_t += r_bl["trades"]; bl_pnl += r_bl["pnl"]; bl_w += int(r_bl["trades"]*r_bl["wr"]/100)
        print(f"{sym:<15} {r_tb['trades']:>7} {r_tb['wr']:>6.1f}% {r_tb['pnl']:>+8.0f} {r_bl['trades']:>7} {r_bl['wr']:>6.1f}% {r_bl['pnl']:>+8.0f}")

    print("-" * 67)
    print(f"{'AGGREGATE':<15} {tb_t:>7} {tb_w/max(tb_t,1)*100:>6.1f}% {tb_pnl:>+8.0f} {bl_t:>7} {bl_w/max(bl_t,1)*100:>6.1f}% {bl_pnl:>+8.0f}")
    print(f"  PnL/trade: ${tb_pnl/max(tb_t,1):.2f} | Trades/koin: {tb_t//len(available)}")

# Save comparison
print(f"\nDone. Model: {run_dir}")
