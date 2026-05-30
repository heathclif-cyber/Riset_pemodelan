"""
pipeline/06_train_guardian.py — Exit Guardian v3 Training
Multiclass LGBM for per-bar decisions: 0=HOLD, 1=PARTIAL_EXIT, 2=FULL_EXIT.

Data generation:
  1. Load feature-labeled parquet for each coin
  2. Run cascade (LGBM + LSTM) to get entry signals
  3. Simulate trades via simulate_trades_swing() to get trade timelines
  4. For each in-trade bar, compute features + forward-looking HOLD/EXIT label
  5. Accumulate labeled training data across all folds

Training:
  - Purged walk-forward CV (same folds as LGBM/LSTM)
  - Binary LGBM with class weight balancing
  - Save best model, scaler, and feature column list

Jalankan:
  python pipeline/06_train_guardian.py
  python pipeline/06_train_guardian.py --all
"""

import argparse, json, sys, warnings
from datetime import datetime
from pathlib import Path

import joblib, numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, roc_auc_score
import lightgbm as lgb

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

from config import (
    TRAINING_COINS, ALL_COINS, LABEL_MAP,
    GUARDIAN_STATIC_FEATURES, GUARDIAN_DYNAMIC_FEATURES,
    GUARDIAN_LGBM_PARAMS, GUARDIAN_EARLY_STOPPING,
    GUARDIAN_N_FOLDS, GUARDIAN_PURGE_GAP_BARS, GUARDIAN_MIN_SAMPLES_COIN,
    GUARDIAN_EXIT_THRESHOLD, GUARDIAN_SL_SAFETY_ATR,
    GUARDIAN_PARTIAL_EXIT_RATIO,
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

logger = setup_logger("06_train_guardian")

TRAIN_LABEL_DIR = LABEL_DIR


def load_models():
    lgbm = joblib.load(MODEL_DIR / "lgbm_baseline.pkl")
    lstm = load_lstm(MODEL_DIR / "lstm_best.pt")
    scaler = joblib.load(MODEL_DIR / "lstm_scaler.pkl")
    with open(MODEL_DIR / "feature_cols_v2.json") as f:
        feat_cols = json.load(f)
    return lgbm, lstm, scaler, feat_cols


def generate_labels_for_coin(
    symbol: str, lgbm_model, lstm_model, lstm_scaler, feat_cols: list[str],
) -> list[dict]:
    """
    Run cascade + simulate trades on one coin, return per-bar labeled samples.
    Each sample: {static features..., dynamic features..., label: 0/1}
    """
    path = TRAIN_LABEL_DIR / f"{symbol}_features_v3.parquet"
    if not path.exists():
        logger.warning(f"  [{symbol}] Data not found: {path}")
        return []

    df = pd.read_parquet(path)
    df = df.sort_index()
    # Potong di TRAIN_CUTOFF_DATE — TIDAK BOLEH ada data testing bocor ke training
    df = df[df.index < TRAIN_CUTOFF_DATE]
    # Skip coin jika data terlalu sedikit setelah date filter
    mask = df["label"].astype(str).isin(LABEL_MAP)
    df = df[mask].copy()
    if len(df) < 100:
        logger.warning(f"  [{symbol}] SKIP — only {len(df)} rows after 2025-05-01 filter")
        return []

    # Align feature columns ke model LGBM (feat_cols dari feature_cols_v2.json)
    # Column yang hilang dari DataFrame diisi 0 (misal hmm_regime_enc)
    n = len(df)
    X = np.zeros((n, len(feat_cols)), dtype=np.float64)
    for idx, col in enumerate(feat_cols):
        if col in df.columns:
            X[:, idx] = df[col].ffill().fillna(0).values.astype(np.float64)

    # Cascade signals (pakai feat_cols asli — 93 kolom, match dengan model)
    y_pred, confidence = hierarchical_predict(
        None, lgbm_model, lstm_model, lstm_scaler,
        X, feat_cols, [], df,
        trend_alignment_enabled=False,
    )
    below = (y_pred != 1) & (confidence < LGBM_THRESHOLD_LONG)
    y_pred[below] = 1

    # Simulate trades
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
        guardian_enabled=False,  # No guardian during label generation
    )

    trades = result.get("trades", [])
    if not trades:
        return []

    # Guardian static features = same feature list as LGBM (dari feature_cols_v2.json)
    g_static_cols = feat_cols
    X_static = np.zeros((len(df), len(feat_cols)), dtype=np.float64)
    for idx, col in enumerate(feat_cols):
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
            continue  # Skip 0-bar trades

        # Scan each in-trade bar, compute label
        for j in range(bar_in + 1, min(bar_out, len(df))):
            current_price = close_arr[j]
            if np.isnan(current_price):
                continue

            # Compute best future PnL from j+1 to bar_out
            best_future_pnl = 0.0
            for k in range(j + 1, min(bar_out, len(df))):
                if np.isnan(close_arr[k]):
                    continue
                if direction == 2:  # LONG
                    pnl_k = (close_arr[k] - entry_price) / entry_price
                else:
                    pnl_k = (entry_price - close_arr[k]) / entry_price
                best_future_pnl = max(best_future_pnl, pnl_k)

            # Current PnL if closed at bar j
            if direction == 2:
                current_pnl = (current_price - entry_price) / entry_price
            else:
                current_pnl = (entry_price - current_price) / entry_price

            # Label: 0=HOLD, 1=PARTIAL_EXIT, 2=FULL_EXIT (v3 — multiclass)
            bars_held = j - bar_in
            atr_pct = atr_entry / entry_price if entry_price > 0 else 0.01

            # Track max favorable PnL seen so far (for reversal detection)
            mfe_sofar = 0.0
            for k in range(bar_in + 1, j + 1):
                if np.isnan(close_arr[k]): continue
                if direction == 2:
                    mfe_sofar = max(mfe_sofar, (close_arr[k] - entry_price) / entry_price)
                else:
                    mfe_sofar = max(mfe_sofar, (entry_price - close_arr[k]) / entry_price)

            # Upside remaining ratio (0..1, 0 = at peak, 1 = far from peak)
            upside_ratio = (best_future_pnl - current_pnl) / best_future_pnl if best_future_pnl > 0.001 else 0.0

            if bars_held < 3:
                label = 0  # HOLD — min hold period
            elif current_pnl < -1.0 * atr_pct:
                label = 2  # FULL_EXIT — deep loss beyond 1x ATR
            elif mfe_sofar > 0.015 and current_pnl < mfe_sofar * 0.25:
                label = 2  # FULL_EXIT — severe reversal (lost 75% of peak)
            elif current_pnl >= best_future_pnl * 0.95:
                label = 2  # FULL_EXIT — near optimal (within 5% of peak)
            elif mfe_sofar > 0.015 and current_pnl < mfe_sofar * 0.55:
                label = 1  # PARTIAL_EXIT — moderate pullback (lost 45% of peak)
            elif current_pnl > 0.008 and upside_ratio < 0.03:
                label = 1  # PARTIAL_EXIT — profit taking (near peak, < 3% upside)
            elif best_future_pnl > current_pnl * 1.05:
                label = 0  # HOLD — 5%+ upside still available
            else:
                continue  # ambiguous zone, skip for cleaner training

            # Dynamic features (no forward-looking leakage)
            mfe_pnl = mfe_sofar  # max favorable PnL seen up to this bar
            bars_held_norm = bars_held / MAX_HOLDING_BARS
            current_pnl_atr = current_pnl / atr_pct if atr_pct > 0 else 0.0
            dd_from_peak = (
                (mfe_pnl - current_pnl) / mfe_pnl if mfe_pnl > 0.001 else 0.0
            )
            entry_ratio = entry_price / current_price if current_price > 0 else 1.0

            sample = {
                **{c: X_static[j, idx] for idx, c in enumerate(g_static_cols)},
                "timestamp": df.index[j],
                "bars_held_norm": bars_held_norm,
                "current_pnl_pct": current_pnl,
                "current_pnl_atr": current_pnl_atr,
                "max_favorable_pnl_pct": mfe_pnl,
                "drawdown_from_peak_pct": dd_from_peak,
                "direction": 1.0 if direction == 2 else 0.0,
                "entry_price_ratio": entry_ratio,
                "label": label,
            }
            samples.append(sample)

    return samples


def train_guardian(samples_df: pd.DataFrame, coin_names: list[str]):
    """Train multiclass LGBM with purged CV, save best model + scaler."""
    # Static features = semua kolom kecuali dynamic + label (sudah difilter di generate_labels)
    g_static_cols = [c for c in samples_df.columns
                     if c not in GUARDIAN_DYNAMIC_FEATURES and c != "label"]
    all_feat_cols = g_static_cols + GUARDIAN_DYNAMIC_FEATURES

    X_all = samples_df[all_feat_cols].values.astype(np.float64)
    y_all = samples_df["label"].values.astype(np.int64)

    n_hold = int((y_all == 0).sum())
    n_partial = int((y_all == 1).sum())
    n_full = int((y_all == 2).sum())
    logger.info(f"Training data: {len(samples_df)} samples, {len(all_feat_cols)} features")
    logger.info(f"Label balance: HOLD={n_hold} PARTIAL_EXIT={n_partial} FULL_EXIT={n_full}")

    if len(samples_df) < 100:
        logger.error("Not enough training samples (< 100)")
        return None, None, None

    folds = build_purged_folds(samples_df.index, GUARDIAN_N_FOLDS, GUARDIAN_PURGE_GAP_BARS)
    logger.info(f"Purged CV: {len(folds)} folds")

    best_score = float("inf")  # multi_logloss: lower is better
    best_model = None
    scaler = StandardScaler()
    best_iters = []

    for fold_idx, (train_idx, test_idx) in enumerate(folds):
        if len(test_idx) < 10:
            continue

        X_train = X_all[train_idx]
        y_train = y_all[train_idx]
        X_test = X_all[test_idx]
        y_test = y_all[test_idx]

        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        # Class weight balancing for multiclass
        n_hold_t = int((y_train == 0).sum())
        n_partial_t = int((y_train == 1).sum())
        n_full_t = int((y_train == 2).sum())
        total = n_hold_t + n_partial_t + n_full_t
        class_weight = {
            0: total / (3 * max(n_hold_t, 1)),
            1: total / (3 * max(n_partial_t, 1)),
            2: total / (3 * max(n_full_t, 1)),
        }

        model = lgb.LGBMClassifier(**GUARDIAN_LGBM_PARAMS, class_weight=class_weight)
        model.fit(
            X_train_s, y_train,
            eval_set=[(X_test_s, y_test)],
            eval_metric="multi_logloss",
            callbacks=[lgb.early_stopping(GUARDIAN_EARLY_STOPPING),
                       lgb.log_evaluation(0)],
        )

        y_prob = model.predict_proba(X_test_s)
        y_pred = y_prob.argmax(axis=1)
        logloss = model.best_score_["valid_0"]["multi_logloss"]
        acc = (y_pred == y_test).mean()
        f1_macro = f1_score(y_test, y_pred, average="macro")

        logger.info(f"  Fold {fold_idx+1}: logloss={logloss:.4f} Acc={acc:.3f} F1_macro={f1_macro:.3f} "
                    f"n_train={len(train_idx)} n_test={len(test_idx)}")
        
        best_iters.append(model.best_iteration_)

        if logloss < best_score:
            best_score = logloss
            best_model = model

    if not best_iters:
        logger.error("No model trained successfully")
        return None, None, None

    avg_best_iter = int(np.mean(best_iters))
    logger.info(f"CV complete. Best logloss: {best_score:.4f} | Average best_iteration: {avg_best_iter}")

    # ── Final Scaler & Model Retraining ──────────────────────────────────────
    # Fit scaler pada 100% data training (X_all)
    logger.info(f"Fitting final Guardian scaler on all {len(X_all):,} samples...")
    final_scaler = StandardScaler()
    X_all_scaled = final_scaler.fit_transform(X_all)

    # Class weight balancing untuk seluruh data training (100%)
    n_hold_all = int((y_all == 0).sum())
    n_partial_all = int((y_all == 1).sum())
    n_full_all = int((y_all == 2).sum())
    total_all = n_hold_all + n_partial_all + n_full_all
    class_weight_all = {
        0: total_all / (3 * max(n_hold_all, 1)),
        1: total_all / (3 * max(n_partial_all, 1)),
        2: total_all / (3 * max(n_full_all, 1)),
    }

    # Set parameters final dengan n_estimators = avg_best_iter
    final_params = GUARDIAN_LGBM_PARAMS.copy()
    final_params["n_estimators"] = avg_best_iter

    logger.info(f"Retraining final Guardian model on 100% data with n_estimators={avg_best_iter}...")
    final_model = lgb.LGBMClassifier(**final_params, class_weight=class_weight_all)
    final_model.fit(X_all_scaled, y_all)

    return final_model, final_scaler, all_feat_cols


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true", help="Use all 20 coins")
    parser.add_argument("--coins", nargs="+", default=None)
    parser.add_argument("--min-samples", type=int, default=GUARDIAN_MIN_SAMPLES_COIN)
    parser.add_argument("--run-id", default=None, help="Custom run ID")
    args = parser.parse_args()

    coins = ALL_COINS if args.all else (args.coins or TRAINING_COINS)
    run_id = args.run_id or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = MODEL_DIR / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Coins: {len(coins)} — {coins} | Run ID: {run_id}")

    lgbm_model, lstm_model, lstm_scaler, feat_cols = load_models()
    logger.info("Models loaded")

    all_samples = []
    for idx, sym in enumerate(coins, 1):
        samples = generate_labels_for_coin(sym, lgbm_model, lstm_model, lstm_scaler, feat_cols)
        logger.info(f"  [{idx:2d}/{len(coins)}] {sym:<14} {len(samples):>5d} in-trade bars")
        if len(samples) >= args.min_samples:
            all_samples.extend(samples)
        else:
            logger.warning(f"  [{idx:2d}/{len(coins)}] {sym:<14} SKIP — below min {args.min_samples}")

    if not all_samples:
        logger.error("No training samples generated")
        return

    df = pd.DataFrame(all_samples)
    df = df.sort_values("timestamp")
    df.set_index("timestamp", inplace=True)
    logger.info(f"Total labeled samples: {len(df)}")

    model, scaler, feat_cols_out = train_guardian(df, coins)
    if model is None:
        return

    # Save to runs folder
    model_path = run_dir / "guardian.pkl"
    scaler_path = run_dir / "guardian_scaler.pkl"
    feat_path = run_dir / "guardian_feature_cols.json"

    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    with open(feat_path, "w") as f:
        json.dump(feat_cols_out, f, indent=2)

    # Copy to root models/ for active inference
    joblib.dump(model, MODEL_DIR / "guardian_best.pkl")
    joblib.dump(scaler, MODEL_DIR / "guardian_scaler.pkl")
    with open(MODEL_DIR / "guardian_feature_cols.json", "w") as f:
        json.dump(feat_cols_out, f, indent=2)

    logger.info(f"Saved: {model_path} and copied to root models/")
    logger.info(f"Saved: {scaler_path} and copied to root models/")
    logger.info(f"Saved: {feat_path} and copied to root models/")

    # Feature importance
    importance = sorted(zip(feat_cols_out, model.feature_importances_),
                        key=lambda x: -x[1])
    logger.info("Top 10 features:")
    for name, imp in importance[:10]:
        logger.info(f"  {name:30s} {imp:.4f}")


if __name__ == "__main__":
    main()
