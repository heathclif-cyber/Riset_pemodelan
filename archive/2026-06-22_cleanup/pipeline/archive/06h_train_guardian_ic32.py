"""
pipeline/06h_train_guardian_ic32.py — Guardian Delta untuk ic32_regime_v1

Sama dengan 06g tapi berbasis ic32_regime_v1 swing trades (bukan TB trades):
  - Entry model: ic32_regime_v1 LGBM (33 feat, thresh long=0.69 / short=0.59)
  - Exit barrier: TP_SL_FALLBACK (2.0 ATR TP, 1.5 ATR SL), max_hold=36
  - Label: optimal stopping — HOLD jika upside > downside * 1.5 AND > 0.3 * ATR%
  - Features: delta sejak entry + multi-bar momentum (14 base features)
  - Output: models/runs/ic32_guardian_delta_v1/
"""
import json, sys, warnings, numpy as np, pandas as pd
from pathlib import Path
import joblib, lightgbm as lgb
warnings.filterwarnings('ignore')
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from core.evaluator import simulate_trades_swing
from core.utils import setup_logger, ensure_utc_index
from sklearn.metrics import f1_score
from scipy import stats as sc_stats
from config import *

logger = setup_logger("06h_guardian_ic32")

RUN_NAME = "ic32_guardian_delta_v1"
IC32_RUN = "ic32_regime_v1"

# ic32 entry thresholds (from config / model_registry)
IC32_THRESH_LONG = LGBM_THRESHOLD_LONG    # 0.69
IC32_THRESH_SHORT = LGBM_THRESHOLD_SHORT  # 0.59

# Same exit barriers as ic32 production
TP_ATR = TP_SL_FALLBACK_TP   # 2.0
SL_ATR = TP_SL_FALLBACK_SL   # 1.5

BASE_FEATURES = [
    "ofi_z_score", "cvd_momentum_adv", "cvd_slope_h4", "ofi_h4_delta",
    "absorption_z", "price_accel_1h",
    "rsi_6", "trend_strength", "ema_21_slope_h4",
    "vol_ratio_20", "vol_spike_zscore", "atr_percentile_h1",
    "funding_rate", "h4_trend",
]
MOM_FEATURES = ["ofi_z_score", "cvd_momentum_adv", "price_accel_1h", "rsi_6"]
MOMENTUM_WINDOW = 3

GUARDIAN_PARAMS = {
    "objective": "binary",
    "n_estimators": 500, "learning_rate": 0.03,
    "max_depth": 5, "num_leaves": 31, "min_child_samples": 50,
    "subsample": 0.8, "colsample_bytree": 0.7,
    "verbose": -1, "n_jobs": -1, "random_state": 42,
}

ic32_model = joblib.load(MODEL_DIR / "runs" / IC32_RUN / "lgbm.pkl")
ic32_feats = ic32_model.feature_name_


def apply_ic32_thresholds(proba):
    """Apply ic32 dual-threshold entry logic: LONG / FLAT / SHORT."""
    y_pred = np.full(len(proba), 1, dtype=np.int32)  # default FLAT
    y_pred[proba[:, 2] > IC32_THRESH_LONG] = 2    # LONG
    # SHORT only if not already LONG
    short_mask = (proba[:, 0] > IC32_THRESH_SHORT) & (y_pred != 2)
    y_pred[short_mask] = 0
    return y_pred


def generate_samples(symbol):
    path = LABEL_DIR / f"{symbol}_features_v3.parquet"
    if not path.exists():
        return None

    df = pd.read_parquet(path)
    df = ensure_utc_index(df).sort_index()
    df = df[df.index < TRAIN_CUTOFF_DATE]
    if len(df) < 100:
        return None

    # ic32 entry predictions
    X = np.zeros((len(df), len(ic32_feats)), dtype=np.float64)
    for i, c in enumerate(ic32_feats):
        if c in df.columns:
            X[:, i] = df[c].ffill().fillna(0).values.astype(np.float64)

    proba = ic32_model.predict_proba(X)
    y_pred = apply_ic32_thresholds(proba)

    close_arr = df["close"].values
    high_arr  = df["high"].values
    low_arr   = df["low"].values
    atr_arr   = df["atr_14_h1"].values
    n = len(close_arr)

    feat_cache = {f: df[f].values for f in BASE_FEATURES if f in df.columns}

    # Simulate trades (no Guardian, no LSTM — pure entry signal)
    result = simulate_trades_swing(
        y_pred=y_pred, close=close_arr, high=high_arr, low=low_arr, atr=atr_arr,
        h4_swing_highs=np.full(n, np.nan), h4_swing_lows=np.full(n, np.nan),
        modal=MODAL_PER_TRADE, leverage=LEVERAGE_SIM[0],
        fee_per_side=FEE_PER_SIDE, slippage=SLIPPAGE_PER_SIDE,
        max_hold=MAX_HOLDING_BARS,
        min_rr=SWING_LABEL_MIN_RR, min_tp_atr=SWING_LABEL_MIN_TP,
        max_sl_atr=SWING_LABEL_MAX_SL,
        tp_fallback_atr=TP_ATR, sl_fallback_atr=SL_ATR,
        guardian_enabled=False, trailing_stop_enabled=False,
    )
    trades = result.get("trades", [])
    if not trades:
        return None

    samples = []
    for t in trades:
        bar_in, bar_out = t["bar_in"], t["bar_out"]
        direction = 1 if t["direction"] == "LONG" else -1
        entry_price = t["entry"]

        for bar in range(bar_in + 2, min(bar_out, n)):
            current_price = close_arr[bar]
            pnl_pct = (current_price - entry_price) / entry_price * direction

            future_start = bar + 1
            future_end   = min(bar_out, n - 1)
            if future_start > future_end:
                continue

            future_prices = close_arr[future_start:future_end + 1]
            if len(future_prices) < 1:
                continue

            atr_now = atr_arr[bar] if bar < n else 1.0
            atr_pct = atr_now / current_price

            if direction == 1:
                upside   = (future_prices.max() - current_price) / current_price
                downside = (current_price - future_prices.min()) / current_price
            else:
                upside   = (current_price - future_prices.min()) / current_price
                downside = (future_prices.max() - current_price) / current_price

            if upside > downside * 1.5 and upside > atr_pct * 0.3:
                label = 1  # HOLD
            elif downside > upside * 1.5 and downside > atr_pct * 0.3:
                label = 0  # EXIT
            else:
                continue   # ambiguous

            # Delta features
            feats = {"label": label, "pnl_pct": pnl_pct, "direction": direction,
                     "bars_held": bar - bar_in}

            for f in BASE_FEATURES:
                arr = feat_cache.get(f)
                if arr is not None:
                    ev = arr[bar_in] if bar_in < len(arr) else 0.0
                    cv = arr[bar]    if bar    < len(arr) else 0.0
                    ev = 0.0 if np.isnan(ev) else ev
                    cv = 0.0 if np.isnan(cv) else cv
                    feats[f"{f}_delta"] = cv - ev
                    feats[f"{f}_curr"]  = cv
                else:
                    feats[f"{f}_delta"] = 0.0
                    feats[f"{f}_curr"]  = 0.0

            for f in MOM_FEATURES:
                arr = feat_cache.get(f)
                if arr is not None:
                    start = max(bar_in, bar - MOMENTUM_WINDOW + 1)
                    vals = [arr[b] for b in range(start, bar + 1)
                            if b < len(arr) and not np.isnan(arr[b])]
                    if vals:
                        feats[f"{f}_mean{MOMENTUM_WINDOW}"]  = float(np.mean(vals))
                        feats[f"{f}_trend{MOMENTUM_WINDOW}"] = float(vals[-1] - vals[0]) if len(vals) >= 2 else 0.0
                    else:
                        feats[f"{f}_mean{MOMENTUM_WINDOW}"]  = 0.0
                        feats[f"{f}_trend{MOMENTUM_WINDOW}"] = 0.0
                else:
                    feats[f"{f}_mean{MOMENTUM_WINDOW}"]  = 0.0
                    feats[f"{f}_trend{MOMENTUM_WINDOW}"] = 0.0

            if bar - bar_in >= 3:
                pch = [(close_arr[b] - close_arr[b-1]) / close_arr[b-1] * direction
                       for b in range(bar_in + 1, bar + 1)]
                feats["price_momentum_3bar"] = float(np.mean(pch[-3:])) if len(pch) >= 3 else 0.0
            else:
                feats["price_momentum_3bar"] = 0.0

            samples.append(feats)

    n_hold = sum(1 for s in samples if s["label"] == 1)
    n_exit = sum(1 for s in samples if s["label"] == 0)
    logger.info(f"  [{symbol}] {len(trades)} trades -> {len(samples)} samples (HOLD={n_hold} EXIT={n_exit})")
    return samples


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--coins", nargs="+", default=None)
    args = parser.parse_args()
    coins = args.coins or (ALL_COINS if args.all else TRAINING_COINS[:5])

    run_dir = MODEL_DIR / "runs" / RUN_NAME
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"  GUARDIAN DELTA for IC32_REGIME_V1 — {RUN_NAME}")
    print(f"  Entry: ic32 LGBM (long>={IC32_THRESH_LONG}, short>={IC32_THRESH_SHORT})")
    print(f"  Exit barriers: TP={TP_ATR}xATR, SL={SL_ATR}xATR, max_hold={MAX_HOLDING_BARS}")
    print(f"  Features: 14 base x (delta + curr) + 4 x (mean3 + trend3) + 3 context = ~47")
    print(f"{'='*70}\n")

    # Stage 1: Generate samples
    print(f"STAGE 1: Generating samples from ic32 trades ({len(coins)} coins)...")
    all_samples = []
    for sym in coins:
        s = generate_samples(sym)
        if s:
            all_samples.extend(s)

    if not all_samples:
        print("ERROR: No samples!"); sys.exit(1)

    samples_df = pd.DataFrame(all_samples)
    n_hold = (samples_df["label"] == 1).sum()
    n_exit = (samples_df["label"] == 0).sum()
    print(f"\n  Total: {len(samples_df):,} | HOLD: {n_hold:,} ({n_hold/len(samples_df)*100:.1f}%) | EXIT: {n_exit:,} ({n_exit/len(samples_df)*100:.1f}%)")

    profit_m = samples_df["pnl_pct"] > 0
    loss_m   = samples_df["pnl_pct"] < 0
    if profit_m.sum() > 0:
        print(f"  In PROFIT: HOLD={(samples_df[profit_m]['label']==1).sum()}/{profit_m.sum()} ({(samples_df[profit_m]['label']==1).mean()*100:.1f}%)")
    if loss_m.sum() > 0:
        print(f"  In LOSS:   HOLD={(samples_df[loss_m]['label']==1).sum()}/{loss_m.sum()} ({(samples_df[loss_m]['label']==1).mean()*100:.1f}%)")

    # Stage 2: Feature selection
    print(f"\nSTAGE 2: IC Test...")
    feat_cols = [c for c in samples_df.columns if c != "label"]
    y_target  = samples_df["label"].values.astype(np.float64)

    standalone = {}
    for feat in feat_cols:
        x = samples_df[feat].values
        mask = ~(np.isnan(x) | np.isnan(y_target))
        if mask.sum() >= 100:
            c, _ = sc_stats.spearmanr(x[mask], y_target[mask])
            standalone[feat] = float(c) if not np.isnan(c) else 0.0
        else:
            standalone[feat] = 0.0

    ic_sorted = sorted(standalone.items(), key=lambda x: abs(x[1]), reverse=True)
    print(f"  Top 20 features:")
    for f, v in ic_sorted[:20]:
        print(f"    {v:+.4f}  {f}")

    selected = [f for f, v in ic_sorted if abs(v) >= 0.01]
    print(f"\n  Selected: {len(selected)} features (|IC|>=0.01)")

    # Stage 3: Forward-only CV (train on past, val on future — no reverse leakage)
    print(f"\nSTAGE 3: Training Guardian ({len(selected)} features, {GUARDIAN_N_FOLDS}-fold forward CV)...")
    X_all = samples_df[selected].ffill().fillna(0).values.astype(np.float64)
    y_all = samples_df["label"].values.astype(np.int32)
    n_tot = len(X_all)

    all_metrics = []
    for fold in range(2, GUARDIAN_N_FOLDS + 1):
        split     = int((fold - 1) / GUARDIAN_N_FOLDS * n_tot)
        v2        = int(fold / GUARDIAN_N_FOLDS * n_tot)
        train_idx = np.arange(0, split)
        val_idx   = np.arange(split, v2)

        if len(train_idx) < 1000 or len(val_idx) < 100:
            continue

        model = lgb.LGBMClassifier(**GUARDIAN_PARAMS)
        model.fit(X_all[train_idx], y_all[train_idx],
                  eval_set=[(X_all[val_idx], y_all[val_idx])],
                  callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(period=-1)])
        yp = model.predict(X_all[val_idx])
        f1 = float(f1_score(y_all[val_idx], yp, average="binary", zero_division=0))
        all_metrics.append({"fold": fold, "f1": round(f1, 4), "iter": model.best_iteration_})
        logger.info(f"  Fold {fold}: F1={f1:.4f} Iter={model.best_iteration_}")

    f1s = [m["f1"] for m in all_metrics]
    print(f"\n  CV F1: {np.mean(f1s):.4f} +/- {np.std(f1s):.4f}")

    # Full retrain
    avg_it = int(np.mean([m["iter"] for m in all_metrics]))
    fp = GUARDIAN_PARAMS.copy()
    fp["n_estimators"] = max(avg_it, 100)
    final_model = lgb.LGBMClassifier(**fp)
    final_model.fit(X_all, y_all)

    # Stage 4: Save
    print(f"\nSTAGE 4: Saving to {run_dir}...")
    joblib.dump(final_model, run_dir / "guardian.pkl")
    with open(run_dir / f"{RUN_NAME}_features.json", "w") as f:
        json.dump(selected, f, indent=2)
    with open(run_dir / f"{RUN_NAME}_meta.json", "w") as f:
        json.dump({
            "run_name": RUN_NAME,
            "entry_model": IC32_RUN,
            "n_samples": len(samples_df), "n_coins": len(coins),
            "n_features": len(selected),
            "cv_f1": round(float(np.mean(f1s)), 4),
            "hold_pct": round(float(n_hold / len(samples_df) * 100), 2),
            "thresh_long": IC32_THRESH_LONG, "thresh_short": IC32_THRESH_SHORT,
        }, f, indent=2)

    imp = sorted(zip(selected, final_model.feature_importances_), key=lambda x: x[1], reverse=True)
    print(f"\n  Top 10 features:")
    for i, (f, v) in enumerate(imp[:10]):
        print(f"  {i+1:>2}. {f:<42} {v:>8.0f}")

    print(f"\n{'='*70}")
    print(f"  COMPLETE — {RUN_NAME}")
    print(f"  CV F1: {np.mean(f1s):.4f} | Features: {len(selected)}")
    print(f"  Model -> {run_dir / 'guardian.pkl'}")
    print(f"{'='*70}")
