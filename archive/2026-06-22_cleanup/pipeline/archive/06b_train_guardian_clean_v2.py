"""
pipeline/06b_train_guardian_clean_v2.py — Guardian clean v2 Training
33 static (dari feature_cols_v2.json) + 7 dynamic (NO delta) = 40 features

Training: 2020-2025, purged CV 8 folds
Output: models/guardian_clean_v2.pkl + guardian_clean_v2_scaler.pkl + guardian_clean_v2_feature_cols.json

Jalankan:
  python pipeline/06b_train_guardian_clean_v2.py
"""

import argparse, json, sys, warnings, os
from datetime import datetime
from pathlib import Path

import joblib, numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
import lightgbm as lgb

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

from config import (
    TRAINING_COINS, LABEL_MAP,
    GUARDIAN_LGBM_PARAMS, GUARDIAN_EARLY_STOPPING,
    GUARDIAN_N_FOLDS, GUARDIAN_PURGE_GAP_BARS, GUARDIAN_MIN_SAMPLES_COIN,
    GUARDIAN_MIN_HOLD_BARS,
    TRAIN_CUTOFF_DATE,
    MAX_HOLDING_BARS, MODAL_PER_TRADE, LEVERAGE_SIM,
    FEE_PER_SIDE, SLIPPAGE_PER_SIDE,
    SWING_LABEL_MIN_RR, SWING_LABEL_MIN_TP, SWING_LABEL_MAX_SL,
    TP_SL_FALLBACK_TP, TP_SL_FALLBACK_SL,
    LGBM_THRESHOLD_LONG, LGBM_THRESHOLD_SHORT,
    MODEL_DIR, LABEL_DIR,
)
from core.models import load_lstm
from core.evaluator import simulate_trades_swing
from core.utils import setup_logger
from pipeline.backtest_utils import hierarchical_predict
from pipeline.shared import build_purged_folds

logger = setup_logger("06b_guardian_clean_v2")

TRAIN_LABEL_DIR = LABEL_DIR

# ── Clean v2 feature config ──────────────────────────────────────────────────
# Static: 33 features dari feature_cols_v2.json (= 32 KEEP ic32 + hmm_regime_enc)
with open(MODEL_DIR / "feature_cols_v2.json") as f:
    CLEAN_V2_STATIC = json.load(f)

# Dynamic: 7 original ONLY (NO delta features)
CLEAN_V2_DYNAMIC = [
    "bars_held_norm", "current_pnl_pct", "current_pnl_atr",
    "max_favorable_pnl_pct", "drawdown_from_peak_pct",
    "direction", "entry_price_ratio",
]

# No delta map
CLEAN_V2_DELTA_MAP = {}

ALL_FEATURES = CLEAN_V2_STATIC + CLEAN_V2_DYNAMIC
logger.info(f"Clean v2: {len(CLEAN_V2_STATIC)} static + {len(CLEAN_V2_DYNAMIC)} dynamic = {len(ALL_FEATURES)} total")


def load_entry_models():
    lgbm_path = MODEL_DIR / "lgbm_baseline.pkl"
    lstm_path = MODEL_DIR / "lstm_best.pt"
    scaler_path = MODEL_DIR / "lstm_scaler.pkl"
    feat_path = MODEL_DIR / "feature_cols_v2.json"

    lgbm = joblib.load(lgbm_path)
    lstm = load_lstm(lstm_path)
    scaler = joblib.load(scaler_path)
    with open(feat_path) as f:
        feat_cols = json.load(f)

    logger.info(f"Entry models loaded — LGBM feats={len(lgbm.feature_name_)}")
    return lgbm, lstm, scaler, feat_cols


def generate_labels_for_coin(symbol, lgbm_model, lstm_model, lstm_scaler, feat_cols):
    path = TRAIN_LABEL_DIR / f"{symbol}_features_v3.parquet"
    if not path.exists():
        logger.warning(f"  [{symbol}] Data not found")
        return []

    df = pd.read_parquet(path).sort_index()
    df = df[df.index < TRAIN_CUTOFF_DATE]

    if "hmm_regime_enc" in feat_cols:
        regime_path = TRAIN_LABEL_DIR / f"{symbol}_regime_h1.parquet"
        if regime_path.exists():
            try:
                reg = pd.read_parquet(regime_path)
                if "hmm_regime_enc" in df.columns:
                    df = df.drop(columns=["hmm_regime_enc"])
                df = df.join(reg[["hmm_regime_enc"]], how="left")
                df["hmm_regime_enc"] = df["hmm_regime_enc"].fillna(1).astype("int32")
            except Exception:
                df["hmm_regime_enc"] = 1

    mask = df["label"].astype(str).isin(LABEL_MAP)
    df = df[mask].copy()
    if len(df) < 100:
        logger.warning(f"  [{symbol}] SKIP — {len(df)} rows")
        return []

    n = len(df)
    X = np.zeros((n, len(feat_cols)), dtype=np.float64)
    for idx, col in enumerate(feat_cols):
        if col in df.columns:
            X[:, idx] = df[col].ffill().fillna(0).values.astype(np.float64)

    # DISABLE LSTM during label generation (DirectML issue)
    import pipeline.backtest_utils as btu
    btu.LSTM_CONFIRMATION_ENABLED = False

    y_pred, confidence = hierarchical_predict(
        None, lgbm_model, lstm_model, lstm_scaler,
        X, feat_cols, [], df,
        trend_alignment_enabled=False,
    )
    below = (y_pred != 1) & (confidence < LGBM_THRESHOLD_LONG)
    y_pred[below] = 1

    close_arr = df["close"].values
    high_arr = df["high"].values if "high" in df.columns else close_arr
    low_arr = df["low"].values if "low" in df.columns else close_arr
    atr_arr = df["atr_14_h1"].values if "atr_14_h1" in df.columns else np.ones(len(df))
    sh_arr = df["h4_swing_high"].values if "h4_swing_high" in df.columns else np.full(len(df), np.nan)
    sl_arr = df["h4_swing_low"].values if "h4_swing_low" in df.columns else np.full(len(df), np.nan)

    result = simulate_trades_swing(
        y_pred=y_pred, close=close_arr, high=high_arr, low=low_arr, atr=atr_arr,
        h4_swing_highs=sh_arr, h4_swing_lows=sl_arr,
        modal=MODAL_PER_TRADE, leverage=LEVERAGE_SIM[0],
        fee_per_side=FEE_PER_SIDE, slippage=SLIPPAGE_PER_SIDE,
        max_hold=MAX_HOLDING_BARS,
        min_rr=SWING_LABEL_MIN_RR, min_tp_atr=SWING_LABEL_MIN_TP,
        max_sl_atr=SWING_LABEL_MAX_SL,
        tp_fallback_atr=TP_SL_FALLBACK_TP, sl_fallback_atr=TP_SL_FALLBACK_SL,
        confidence=confidence,
        guardian_enabled=False,
    )

    trades = result.get("trades", [])
    if not trades:
        return []

    # Build static feature array using CLEAN_V2_STATIC (33 features)
    g_static_cols = [c for c in CLEAN_V2_STATIC if c in df.columns or c in feat_cols]
    X_static = np.zeros((len(df), len(g_static_cols)), dtype=np.float64)
    for idx, col in enumerate(g_static_cols):
        if col in df.columns:
            X_static[:, idx] = df[col].ffill().fillna(0).values.astype(np.float64)

    samples = []
    for t in trades:
        bar_in = t["bar_in"]
        bar_out = t["bar_out"]
        direction = 2 if t["direction"] == "LONG" else 0
        entry_price = t["entry"]
        atr_entry = atr_arr[bar_in] if bar_in < len(atr_arr) else 1.0

        if bar_out <= bar_in + 1:
            continue

        for j in range(bar_in + 1, min(bar_out, len(df))):
            current_price = close_arr[j]
            if np.isnan(current_price):
                continue

            # Best future PnL
            best_future_pnl = 0.0
            for k in range(j + 1, min(bar_out, len(df))):
                if np.isnan(close_arr[k]): continue
                if direction == 2:
                    best_future_pnl = max(best_future_pnl, (close_arr[k] - entry_price) / entry_price)
                else:
                    best_future_pnl = max(best_future_pnl, (entry_price - close_arr[k]) / entry_price)

            if direction == 2:
                current_pnl = (current_price - entry_price) / entry_price
            else:
                current_pnl = (entry_price - current_price) / entry_price

            bars_held = j - bar_in
            atr_pct = atr_entry / entry_price if entry_price > 0 else 0.01

            # MFE so far
            mfe_sofar = 0.0
            for k in range(bar_in + 1, j + 1):
                if np.isnan(close_arr[k]): continue
                if direction == 2:
                    mfe_sofar = max(mfe_sofar, (close_arr[k] - entry_price) / entry_price)
                else:
                    mfe_sofar = max(mfe_sofar, (entry_price - close_arr[k]) / entry_price)

            upside_ratio = (best_future_pnl - current_pnl) / best_future_pnl if best_future_pnl > 0.001 else 0.0

            # Labeling logic (same as original)
            if bars_held < GUARDIAN_MIN_HOLD_BARS:
                label = 0
            elif current_pnl < -1.0 * atr_pct:
                label = 2
            elif mfe_sofar > 0.015 and current_pnl < mfe_sofar * 0.25:
                label = 2
            elif current_pnl >= best_future_pnl * 0.95:
                label = 2
            elif mfe_sofar > 0.015 and current_pnl < mfe_sofar * 0.55:
                label = 1
            elif current_pnl > 0.008 and upside_ratio < 0.03:
                label = 1
            elif best_future_pnl > current_pnl * 1.05:
                label = 0
            else:
                continue

            # Dynamic features (7 original, no delta)
            bars_held_norm = bars_held / MAX_HOLDING_BARS
            current_pnl_atr = current_pnl / atr_pct if atr_pct > 0 else 0.0
            dd_from_peak = (mfe_sofar - current_pnl) / mfe_sofar if mfe_sofar > 0.001 else 0.0
            entry_ratio = entry_price / current_price if current_price > 0 else 1.0

            sample = {
                **{c: float(X_static[j, idx]) for idx, c in enumerate(g_static_cols)},
                "timestamp": df.index[j],
                "bars_held_norm": bars_held_norm,
                "current_pnl_pct": current_pnl,
                "current_pnl_atr": current_pnl_atr,
                "max_favorable_pnl_pct": mfe_sofar,
                "drawdown_from_peak_pct": dd_from_peak,
                "direction": 1.0 if direction == 2 else 0.0,
                "entry_price_ratio": entry_ratio,
                "label": label,
            }
            samples.append(sample)

    return samples


def train_guardian(samples_df, n_folds=GUARDIAN_N_FOLDS):
    g_static_cols = [c for c in samples_df.columns
                     if c not in set(CLEAN_V2_DYNAMIC) and c not in ("label", "timestamp")]
    all_feat_cols = g_static_cols + CLEAN_V2_DYNAMIC

    X_all = samples_df[all_feat_cols].values.astype(np.float64)
    y_all = samples_df["label"].values.astype(np.int64)

    n_hold = int((y_all == 0).sum())
    n_partial = int((y_all == 1).sum())
    n_full = int((y_all == 2).sum())
    logger.info(f"Training: {len(samples_df)} samples, {len(all_feat_cols)} features")
    logger.info(f"Labels: HOLD={n_hold} PARTIAL={n_partial} FULL={n_full}")

    if len(samples_df) < 100:
        logger.error("Not enough samples")
        return None, None, None

    folds = build_purged_folds(samples_df.index, n_folds, GUARDIAN_PURGE_GAP_BARS)
    logger.info(f"Purged CV: {len(folds)} folds")

    # Fit scaler ONCE on all data before fold loop
    scaler = StandardScaler()
    X_all_scaled = scaler.fit_transform(X_all)

    best_score = float("inf")
    best_model = None
    cv_results = []

    for fold_idx, (train_idx, test_idx) in enumerate(folds):
        if len(test_idx) < 10:
            continue

        X_train = X_all_scaled[train_idx]
        y_train = y_all[train_idx]
        X_test = X_all_scaled[test_idx]
        y_test = y_all[test_idx]

        classes_in_fold = set(np.unique(y_train))
        class_weight = {}
        for c in sorted(classes_in_fold):
            if c == 0: class_weight[c] = 1.0
            elif c == 1: class_weight[c] = 3.0
            elif c == 2: class_weight[c] = 2.0

        params = {**GUARDIAN_LGBM_PARAMS}
        model = lgb.LGBMClassifier(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            eval_metric="multi_logloss",
            callbacks=[lgb.early_stopping(GUARDIAN_EARLY_STOPPING), lgb.log_evaluation(0)],
        )

        y_prob = model.predict_proba(X_test)
        y_pred = np.argmax(y_prob, axis=1)
        f1 = f1_score(y_test, y_pred, average="macro")
        logloss = -np.mean(np.log(y_prob[np.arange(len(y_test)), y_test] + 1e-10))

        cv_results.append({"fold": fold_idx + 1, "logloss": round(logloss, 4), "f1_macro": round(f1, 4)})
        logger.info(f"  Fold {fold_idx+1}: logloss={logloss:.4f} f1_macro={f1:.4f}")

        if logloss < best_score:
            best_score = logloss
            best_model = model
            best_iters = model.best_iteration_

    # Retrain final on all data
    logger.info(f"Best CV logloss: {best_score:.4f}")
    logger.info("Retraining final model on all data...")

    params_final = {**GUARDIAN_LGBM_PARAMS}
    if best_iters and best_iters > 0:
        params_final["n_estimators"] = best_iters
    final_model = lgb.LGBMClassifier(**params_final)
    final_model.fit(X_all_scaled, y_all)

    return final_model, scaler, all_feat_cols, cv_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--coins", nargs="+", default=TRAINING_COINS)
    parser.add_argument("--n-folds", type=int, default=GUARDIAN_N_FOLDS)
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f" Guardian Clean v2 Training")
    print(f" {len(CLEAN_V2_STATIC)} static + {len(CLEAN_V2_DYNAMIC)} dynamic = {len(ALL_FEATURES)} total features")
    print(f" Coins: {len(args.coins)} | Folds: {args.n_folds}")
    print(f" Cutoff: {TRAIN_CUTOFF_DATE}")
    print(f"{'='*60}\n")

    # Load entry models
    lgbm_model, lstm_model, lstm_scaler, feat_cols = load_entry_models()

    # Generate labels
    print("[1/3] Generating in-trade labeled samples...")
    all_samples = []
    for coin in args.coins:
        samples = generate_labels_for_coin(coin, lgbm_model, lstm_model, lstm_scaler, feat_cols)
        if samples:
            all_samples.extend(samples)
            logger.info(f"  {coin}: {len(samples)} samples")
    print(f"  Total: {len(all_samples)} samples from {sum(1 for c in args.coins if c)} coins")

    if not all_samples:
        print("ERROR: No samples generated")
        return

    samples_df = pd.DataFrame(all_samples)
    samples_df["timestamp"] = pd.to_datetime(samples_df["timestamp"])
    samples_df = samples_df.set_index("timestamp").sort_index()

    # Train
    print("\n[2/3] Training multiclass LGBM...")
    model, scaler, feat_cols_out, cv_results = train_guardian(samples_df, args.n_folds)

    if model is None:
        print("ERROR: Training failed")
        return

    # Save
    print("\n[3/3] Saving model...")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    model_path = MODEL_DIR / "guardian_clean_v2.pkl"
    scaler_path = MODEL_DIR / "guardian_clean_v2_scaler.pkl"
    feat_path = MODEL_DIR / "guardian_clean_v2_feature_cols.json"

    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    with open(feat_path, "w") as f:
        json.dump(feat_cols_out, f, indent=2)

    print(f"  Model : {model_path}")
    print(f"  Scaler: {scaler_path}")
    print(f"  Feats : {feat_path} ({len(feat_cols_out)} columns)")

    # Print feature importance top 10
    imp = list(zip(feat_cols_out, model.feature_importances_))
    dyn_imp = [(n, v) for n, v in imp if n in set(CLEAN_V2_DYNAMIC)]
    sta_imp = [(n, v) for n, v in imp if n not in set(CLEAN_V2_DYNAMIC)]
    dyn_share = sum(v for _, v in dyn_imp) / sum(v for _, v in imp) * 100
    sta_share = sum(v for _, v in sta_imp) / sum(v for _, v in imp) * 100
    print(f"\n  Feature importance share:")
    print(f"    Dynamic: {dyn_share:.1f}% ({len(dyn_imp)} features)")
    print(f"    Static : {sta_share:.1f}% ({len(sta_imp)} features)")
    print(f"\n  Top 10:")
    for f, v in sorted(imp, key=lambda x: x[1], reverse=True)[:10]:
        tag = "[DYN]" if f in set(CLEAN_V2_DYNAMIC) else ""
        print(f"    {f:<35} {v:>8.1f}  {tag}")

    # CV summary
    if cv_results:
        print(f"\n  CV Results ({len(cv_results)} folds):")
        for r in cv_results:
            print(f"    Fold {r['fold']}: logloss={r['logloss']:.4f} f1={r['f1_macro']:.4f}")

    print(f"\nDone! Clean v2 saved to {model_path.name}")
    print(f"Update guardian_best.pkl? Run:")
    print(f"  copy {model_path.name} guardian_best.pkl")
    print(f"  copy {scaler_path.name} guardian_scaler.pkl")
    print(f"  copy {feat_path.name} guardian_feature_cols.json")


if __name__ == "__main__":
    main()
