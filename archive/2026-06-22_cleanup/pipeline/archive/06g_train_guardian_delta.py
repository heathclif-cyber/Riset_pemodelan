"""
pipeline/06g_train_guardian_delta.py — Guardian dengan Delta + Multi-bar Momentum

Perbaikan fundamental:
1. Delta features: perubahan fitur sejak entry bar (bukan snapshot)
2. Multi-bar momentum: rolling window 3-5 bar dari key flow indicators
3. Label: optimal stopping — "exit sekarang atau hold lebih menguntungkan?"
4. Hanya decision points yang dilatih (skip bar awal dan akhir trade)
"""
import json, sys, warnings, numpy as np, pandas as pd
from datetime import datetime
from pathlib import Path
import joblib, lightgbm as lgb
warnings.filterwarnings('ignore')
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from core.evaluator import simulate_trades_swing
from core.utils import setup_logger, ensure_utc_index
from pipeline.shared import build_purged_folds
from sklearn.metrics import f1_score
from scipy import stats as sc_stats
from config import *

logger = setup_logger("06g_guardian_delta")

RUN_NAME = "tb_guardian_delta_v1"
TB_MODEL_RUN = "tb_lgbm_widyawardhana_v3"
REGIME_THRESH = {0: 0.45, 1: 0.50, 2: 0.50, 3: 0.45}
TP_ATR = TP_SL_FALLBACK_TP
SL_ATR = TP_SL_FALLBACK_SL

# Core features that change meaningfully during a trade
BASE_FEATURES = [
    "ofi_z_score", "cvd_momentum_adv", "cvd_slope_h4", "ofi_h4_delta",
    "absorption_z", "price_accel_1h",
    "rsi_6", "trend_strength", "ema_21_slope_h4",
    "vol_ratio_20", "vol_spike_zscore", "atr_percentile_h1",
    "funding_rate", "h4_trend",
]

GUARDIAN_PARAMS = {
    "objective": "binary",
    "n_estimators": 500, "learning_rate": 0.03,
    "max_depth": 5, "num_leaves": 31, "min_child_samples": 50,
    "subsample": 0.8, "colsample_bytree": 0.7,
    "verbose": -1, "n_jobs": -1, "random_state": 42,
}

MOMENTUM_WINDOW = 3  # bars for rolling momentum calculation

tb_model = joblib.load(MODEL_DIR / "runs" / TB_MODEL_RUN / "lgbm.pkl")
with open(MODEL_DIR / "runs" / TB_MODEL_RUN / f"{TB_MODEL_RUN}_features.json") as f:
    tb_feats = json.load(f)


def compute_delta_and_momentum(df, bar, bar_in, feat_cache, close_arr, direction):
    """Compute delta features (change since entry) and multi-bar momentum at a bar."""
    features = {}

    # Delta from entry
    for f in BASE_FEATURES:
        if f in feat_cache:
            entry_val = feat_cache[f][bar_in] if bar_in < len(feat_cache[f]) else 0
            curr_val = feat_cache[f][bar] if bar < len(feat_cache[f]) else 0
            entry_val = 0 if np.isnan(entry_val) else entry_val
            curr_val = 0 if np.isnan(curr_val) else curr_val
            features[f"{f}_delta"] = curr_val - entry_val
            features[f"{f}_curr"] = curr_val
        else:
            features[f"{f}_delta"] = 0.0
            features[f"{f}_curr"] = 0.0

    # Multi-bar momentum: rolling mean over last MOMENTUM_WINDOW bars
    for f in ["ofi_z_score", "cvd_momentum_adv", "price_accel_1h", "rsi_6"]:
        if f in feat_cache:
            start = max(bar_in, bar - MOMENTUM_WINDOW + 1)
            vals = [feat_cache[f][b] for b in range(start, bar + 1)
                    if b < len(feat_cache[f]) and not np.isnan(feat_cache[f][b])]
            if vals:
                features[f"{f}_mean{MOMENTUM_WINDOW}"] = np.mean(vals)
                if len(vals) >= 2:
                    features[f"{f}_trend{MOMENTUM_WINDOW}"] = vals[-1] - vals[0]
                else:
                    features[f"{f}_trend{MOMENTUM_WINDOW}"] = 0.0
            else:
                features[f"{f}_mean{MOMENTUM_WINDOW}"] = 0.0
                features[f"{f}_trend{MOMENTUM_WINDOW}"] = 0.0
        else:
            features[f"{f}_mean{MOMENTUM_WINDOW}"] = 0.0
            features[f"{f}_trend{MOMENTUM_WINDOW}"] = 0.0

    # Price momentum within trade
    pnl_pct = (close_arr[bar] - close_arr[bar_in]) / close_arr[bar_in] * direction
    if bar - bar_in >= 3:
        price_changes = [(close_arr[b] - close_arr[b-1]) / close_arr[b-1] * direction
                         for b in range(bar_in + 1, bar + 1)]
        features["price_momentum_3bar"] = np.mean(price_changes[-3:]) if len(price_changes) >= 3 else 0
    else:
        features["price_momentum_3bar"] = 0.0

    features["pnl_pct"] = pnl_pct
    features["bars_held"] = bar - bar_in

    return features


def generate_trades_and_samples(symbol):
    """Generate TB trades + per-bar Guardian samples with optimal-stopping labels."""
    path = LABEL_DIR / f"{symbol}_features_v3.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    df = ensure_utc_index(df).sort_index()
    df = df[df.index < TRAIN_CUTOFF_DATE]
    if len(df) < 100:
        return None

    reg_path = LABEL_DIR / f"{symbol}_regime_h1.parquet"
    hmm = np.full(len(df), 1, dtype=np.int32)
    if reg_path.exists():
        reg = pd.read_parquet(reg_path)
        if "hmm_regime_enc" in reg.columns:
            hmm = reg["hmm_regime_enc"].reindex(df.index, fill_value=1).values.astype(np.int32)

    X = np.zeros((len(df), len(tb_feats)), dtype=np.float64)
    for i, c in enumerate(tb_feats):
        if c in df.columns: X[:, i] = df[c].ffill().fillna(0).values.astype(np.float64)
    proba = tb_model.predict_proba(X)
    conf = np.max(proba, axis=1)
    y_pred = np.argmax(proba, axis=1)
    for r, th in REGIME_THRESH.items():
        y_pred[(hmm == r) & (y_pred != 1) & (conf < th)] = 1

    close_arr = df["close"].values; high_arr = df["high"].values
    low_arr = df["low"].values; atr_arr = df["atr_14_h1"].values; n = len(close_arr)

    feat_cache = {}
    for f in BASE_FEATURES:
        if f in df.columns:
            feat_cache[f] = df[f].values

    result = simulate_trades_swing(
        y_pred=y_pred, close=close_arr, high=high_arr, low=low_arr, atr=atr_arr,
        h4_swing_highs=np.full(n, np.nan), h4_swing_lows=np.full(n, np.nan),
        modal=MODAL_PER_TRADE, leverage=LEVERAGE_SIM[0],
        fee_per_side=FEE_PER_SIDE, slippage=SLIPPAGE_PER_SIDE, max_hold=MAX_HOLDING_BARS,
        min_rr=SWING_LABEL_MIN_RR, min_tp_atr=SWING_LABEL_MIN_TP,
        max_sl_atr=SWING_LABEL_MAX_SL,
        tp_fallback_atr=TP_SL_FALLBACK_TP, sl_fallback_atr=TP_SL_FALLBACK_SL,
        guardian_enabled=False, trailing_stop_enabled=False,
    )
    trades = result.get("trades", [])

    samples = []
    for t in trades:
        bar_in, bar_out = t["bar_in"], t["bar_out"]
        direction = 1 if t["direction"] == "LONG" else -1
        entry_price = t["entry"]

        for bar in range(bar_in + 2, min(bar_out, n)):  # min_hold=2
            current_price = close_arr[bar]
            if direction == 1:
                pnl_pct = (current_price - entry_price) / entry_price
            else:
                pnl_pct = (entry_price - current_price) / entry_price

            # Future: what happens AFTER this bar? (excluding current bar)
            future_start = bar + 1
            future_end = min(bar_out, n - 1)
            if future_start > future_end:
                continue

            future_prices = close_arr[future_start:future_end + 1]
            if len(future_prices) < 1:
                continue

            atr_now = atr_arr[bar] if bar < n else 1.0

            if direction == 1:
                future_best = future_prices.max()
                future_worst = future_prices.min()
                upside = (future_best - current_price) / current_price
                downside = (current_price - future_worst) / current_price
            else:
                future_best = future_prices.min()
                future_worst = future_prices.max()
                upside = (current_price - future_best) / current_price
                downside = (future_worst - current_price) / current_price

            # Label: HOLD if there's meaningful upside after this bar (>0.3% price move)
            # EXIT if there's more downside than upside and downside is significant
            atr_pct = atr_now / current_price

            if upside > downside * 1.5 and upside > atr_pct * 0.3:
                label = 1  # HOLD — asymmetric upside, momentum likely to continue
            elif downside > upside * 1.5 and downside > atr_pct * 0.3:
                label = 0  # EXIT — asymmetric downside, likely to reverse
            else:
                continue  # ambiguous, skip

            # Compute delta + momentum features
            feats = compute_delta_and_momentum(df, bar, bar_in, feat_cache, close_arr, direction)
            feats["label"] = label
            feats["pnl_pct"] = pnl_pct
            feats["direction"] = direction
            samples.append(feats)

    if len(trades) > 0:
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
    print(f"  GUARDIAN DELTA + MULTI-BAR — {RUN_NAME}")
    print(f"  Features: delta (change since entry) + rolling momentum + price trend")
    print(f"  Label: optimal stopping (hold only if significant upside)")
    print(f"{'='*70}\n")

    # Stage 1
    print("STAGE 1: Generating delta-momentum samples...")
    all_samples = []
    for sym in coins:
        s = generate_trades_and_samples(sym)
        if s: all_samples.extend(s)

    if not all_samples:
        print("ERROR: No samples!"); sys.exit(1)

    samples_df = pd.DataFrame(all_samples)
    n_hold = (samples_df["label"] == 1).sum()
    n_exit = (samples_df["label"] == 0).sum()
    print(f"\n  Total: {len(samples_df):,} | HOLD: {n_hold:,} ({n_hold/len(samples_df)*100:.1f}%) | EXIT: {n_exit:,} ({n_exit/len(samples_df)*100:.1f}%)")

    # Separate stats by profit/loss
    profit_mask = samples_df["pnl_pct"] > 0
    loss_mask = samples_df["pnl_pct"] < 0
    if profit_mask.sum() > 0:
        print(f"  In PROFIT: HOLD={(samples_df[profit_mask]['label']==1).sum()}/{profit_mask.sum()} ({(samples_df[profit_mask]['label']==1).mean()*100:.1f}%)")
    if loss_mask.sum() > 0:
        print(f"  In LOSS:   HOLD={(samples_df[loss_mask]['label']==1).sum()}/{loss_mask.sum()} ({(samples_df[loss_mask]['label']==1).mean()*100:.1f}%)")

    # Stage 2: Feature selection
    print(f"\nSTAGE 2: IC Test...")
    feat_cols = [c for c in samples_df.columns if c not in ["label"]]
    y_target = samples_df["label"].values.astype(np.float64)

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
    print(f"  Top features:")
    for f, v in ic_sorted[:20]:
        print(f"    {v:+.4f}  {f}")

    selected = [f for f, v in ic_sorted if abs(v) >= 0.01]
    print(f"\n  Selected: {len(selected)} features (|IC|>=0.01)")

    # Stage 3: Train
    print(f"\nSTAGE 3: Training Guardian Delta ({len(selected)} features)...")
    X_all = samples_df[selected].ffill().fillna(0).values.astype(np.float64)
    y_all = samples_df["label"].values.astype(np.int32)
    n_tot = len(X_all)

    # Forward-only CV: train on past, validate on future (no reverse leakage)
    # Skip first fold (too little train data), use folds 2..N
    all_metrics = []
    for fold in range(2, GUARDIAN_N_FOLDS + 1):
        split = int((fold - 1) / GUARDIAN_N_FOLDS * n_tot)
        v2    = int(fold / GUARDIAN_N_FOLDS * n_tot)
        train_idx = np.arange(0, split)        # everything BEFORE val
        val_idx   = np.arange(split, v2)       # next time block

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
    print(f"\n  CV F1: {np.mean(f1s):.4f} +/- {np.std(f1s):.4f} (random=0.50)")

    # Full retrain
    avg_it = int(np.mean([m["iter"] for m in all_metrics]))
    fp = GUARDIAN_PARAMS.copy(); fp["n_estimators"] = max(avg_it, 100)
    final_model = lgb.LGBMClassifier(**fp); final_model.fit(X_all, y_all)

    # Save
    print(f"\nSTAGE 4: Saving...")
    joblib.dump(final_model, run_dir / "guardian.pkl")
    with open(run_dir / f"{RUN_NAME}_features.json", "w") as f:
        json.dump(selected, f, indent=2)
    with open(run_dir / f"{RUN_NAME}_meta.json", "w") as f:
        json.dump({"run_name": RUN_NAME, "n_samples": len(samples_df), "n_coins": len(coins),
                   "n_features": len(selected), "cv_f1": round(float(np.mean(f1s)), 4),
                   "hold_pct": n_hold/len(samples_df)*100}, f, indent=2)

    imp = list(zip(selected, final_model.feature_importances_))
    imp.sort(key=lambda x: x[1], reverse=True)
    print(f"\n  Top 10 features:")
    for i, (f, v) in enumerate(imp[:10]):
        print(f"  {i+1:>2}. {f:<40} {v:>8.0f}")

    print(f"\n{'='*70}")
    print(f"  GUARDIAN DELTA COMPLETE — {RUN_NAME}")
    print(f"  CV F1: {np.mean(f1s):.4f} | Features: {len(selected)}")
    print(f"  Model -> {run_dir / 'guardian.pkl'}")
    print(f"{'='*70}")
