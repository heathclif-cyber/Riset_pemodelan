"""
pipeline/06d_train_guardian_momentum.py — Guardian Momentum untuk TB v3

Label berbasis MOMENTUM (bukan PnL statis):
  HOLD    = momentum masih kuat, bisa lanjut beyond TP
  EXIT    = momentum reversal / divergence
  PARTIAL = momentum melemah, ambil profit sebagian

Fitur: OFI, CVD, price acceleration, volume — flow-based
"""
import json, sys, warnings, numpy as np, pandas as pd
from datetime import datetime
from pathlib import Path
import joblib, lightgbm as lgb
warnings.filterwarnings('ignore')
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from core.features import triple_barrier_labeling
from core.evaluator import simulate_trades_swing
from core.utils import setup_logger, ensure_utc_index
from pipeline.shared import build_purged_folds
from sklearn.metrics import f1_score
from scipy import stats as sc_stats
from config import *

logger = setup_logger("06d_guardian_mom")

RUN_NAME = "tb_guardian_momentum_v1"
TB_MODEL_RUN = "tb_lgbm_widyawardhana_v3"
REGIME_THRESH = {0: 0.45, 1: 0.50, 2: 0.50, 3: 0.45}

# Momentum-specific features (NOT PnL-based!)
MOMENTUM_FEATURES = [
    # Flow indicators (perubahan intra-trade)
    "ofi_z_score",           # Order flow imbalance
    "cvd_momentum_adv",      # CVD momentum
    "cvd_slope_h4",          # CVD slope
    "ofi_h4_delta",          # OFI delta H4
    "absorption_z",          # Absorption
    # Price dynamics (perubahan intra-trade)
    "price_accel_1h",        # Price acceleration
    "log_ret_1", "log_ret_5",  # Short-term returns
    "rsi_6",                 # RSI for overbought/oversold
    # Volume context
    "vol_ratio_20",          # Volume ratio
    "vol_spike_zscore",      # Volume spike
    # Trend strength
    "trend_strength",        # How strong is the trend
    "h4_trend",              # H4 trend direction
    "ema_21_slope_h4",       # EMA slope
    # Risk context
    "atr_percentile_h1",     # Current volatility percentile
    "funding_rate",          # Funding pressure
]

GUARDIAN_PARAMS = {
    "objective": "multiclass", "num_class": 3,
    "n_estimators": 2000, "learning_rate": 0.02,
    "max_depth": 6, "num_leaves": 63, "min_child_samples": 30,
    "subsample": 0.8, "colsample_bytree": 0.8,
    "lambda_l1": 0.1, "lambda_l2": 0.1,
    "verbose": -1, "n_jobs": -1, "random_state": 42,
    "class_weight": "balanced",
}

# Load TB model
tb_model = joblib.load(MODEL_DIR / "runs" / TB_MODEL_RUN / "lgbm.pkl")
with open(MODEL_DIR / "runs" / TB_MODEL_RUN / f"{TB_MODEL_RUN}_features.json") as f:
    tb_feats = json.load(f)


def generate_tb_trades(symbol):
    """Generate TB v3 trades for training data."""
    path = LABEL_DIR / f"{symbol}_features_v3.parquet"
    if not path.exists():
        return None, None
    df = pd.read_parquet(path)
    df = ensure_utc_index(df).sort_index()
    df = df[df.index < TRAIN_CUTOFF_DATE]
    if len(df) < 100:
        return None, None
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

    close_arr = df["close"].values
    high_arr = df["high"].values if "high" in df.columns else close_arr
    low_arr = df["low"].values if "low" in df.columns else close_arr
    atr_arr = df["atr_14_h1"].values if "atr_14_h1" in df.columns else np.ones(len(df))
    result = simulate_trades_swing(
        y_pred=y_pred, close=close_arr, high=high_arr, low=low_arr, atr=atr_arr,
        h4_swing_highs=np.full(len(df), np.nan), h4_swing_lows=np.full(len(df), np.nan),
        modal=MODAL_PER_TRADE, leverage=LEVERAGE_SIM[0],
        fee_per_side=FEE_PER_SIDE, slippage=SLIPPAGE_PER_SIDE, max_hold=MAX_HOLDING_BARS,
        min_rr=SWING_LABEL_MIN_RR, min_tp_atr=SWING_LABEL_MIN_TP,
        max_sl_atr=SWING_LABEL_MAX_SL,
        tp_fallback_atr=TP_SL_FALLBACK_TP, sl_fallback_atr=TP_SL_FALLBACK_SL,
        guardian_enabled=False, trailing_stop_enabled=False,
    )
    trades = result.get("trades", [])
    logger.info(f"  [{symbol}] {len(trades)} trades")
    return trades, df


def label_momentum_samples(trades, df):
    """
    Momentum-based labeling untuk Guardian.

    Di setiap bar trade, cek momentum flow indicators:
    - HOLD (0): OFI masih positif/naik, CVD accelerating, trend kuat → lanjut hold
    - PARTIAL (1): Momentum mulai flat/menurun, tapi belum reversal → ambil profit
    - FULL_EXIT (2): OFI reversal signifikan, divergence, atau approaching SL
    """
    samples = []
    if not trades:
        return samples

    close_arr = df["close"].values
    n = len(close_arr)

    # Cache momentum feature arrays
    feat_cache = {}
    for f in MOMENTUM_FEATURES:
        if f in df.columns:
            feat_cache[f] = df[f].values

    for t in trades:
        bar_in = t["bar_in"]
        bar_out = t["bar_out"]
        direction = 1 if t["direction"] == "LONG" else -1
        entry_price = t["entry"]

        for bar in range(bar_in, min(bar_out, n)):
            bars_held = bar - bar_in + 1
            if bars_held < 2:
                continue

            current_price = close_arr[bar]
            if direction == 1:
                pnl_pct = (current_price - entry_price) / entry_price
            else:
                pnl_pct = (entry_price - current_price) / entry_price

            # Compute momentum score at this bar
            # OFI: positive = buying pressure (good for LONG, bad for SHORT)
            ofi_val = feat_cache.get("ofi_z_score", np.zeros(n))[bar] if "ofi_z_score" in feat_cache else 0
            ofi_val = ofi_val if not np.isnan(ofi_val) else 0
            cvd_mom = feat_cache.get("cvd_momentum_adv", np.zeros(n))[bar] if "cvd_momentum_adv" in feat_cache else 0
            cvd_mom = cvd_mom if not np.isnan(cvd_mom) else 0
            price_accel = feat_cache.get("price_accel_1h", np.zeros(n))[bar] if "price_accel_1h" in feat_cache else 0
            price_accel = price_accel if not np.isnan(price_accel) else 0
            trend_str = feat_cache.get("trend_strength", np.zeros(n))[bar] if "trend_strength" in feat_cache else 0
            trend_str = trend_str if not np.isnan(trend_str) else 0
            rsi_val = feat_cache.get("rsi_6", np.zeros(n))[bar] if "rsi_6" in feat_cache else 50
            rsi_val = rsi_val if not np.isnan(rsi_val) else 50

            # Momentum alignment with trade direction
            if direction == 1:  # LONG
                flow_score = ofi_val + cvd_mom + price_accel * 10
            else:  # SHORT
                flow_score = -(ofi_val + cvd_mom + price_accel * 10)

            # Future momentum check: is there still upside beyond current bar?
            future_bars = min(bar_out, n) - bar
            future_prices = close_arr[bar:min(bar_out, n)]
            if len(future_prices) > 0:
                if direction == 1:
                    future_best = future_prices.max()
                    future_worst = future_prices.min()
                else:
                    future_best = future_prices.min()
                    future_worst = future_prices.max()
                upside_remaining = (future_best - current_price) / current_price if direction == 1 else \
                                   (current_price - future_best) / current_price
                downside_remaining = (current_price - future_worst) / current_price if direction == 1 else \
                                     (future_worst - current_price) / current_price
            else:
                upside_remaining = 0
                downside_remaining = 0

            # ── Momentum Label Logic ─────────────────────────────────────────
            label = 0  # default HOLD

            # RSI extremes
            is_overbought = (direction == 1 and rsi_val > 75) or (direction == -1 and rsi_val < 25)
            is_oversold = (direction == 1 and rsi_val < 25) or (direction == -1 and rsi_val > 75)

            # Flow reversal detection
            flow_reversing = flow_score < -0.5  # momentum berlawanan arah trade

            # Trend weakening
            trend_weak = trend_str < 0.02

            # Strong momentum = continue
            strong_momentum = flow_score > 1.0 and trend_str > 0.05

            if flow_reversing and pnl_pct < 0:
                label = 2  # FULL_EXIT — momentum reversal + loss
            elif is_overbought and upside_remaining < 0.01:
                label = 2  # FULL_EXIT — overbought, no more upside
            elif flow_reversing is False and trend_weak and upside_remaining < 0.02:
                label = 1  # PARTIAL — momentum flat, limited upside
            elif is_overbought and strong_momentum:
                label = 0  # HOLD — overbought but momentum still strong (extended run)
            elif bars_held >= MAX_HOLDING_BARS - 3:
                label = 2  # FULL_EXIT — time approaching
            elif pnl_pct > 0.03 and upside_remaining < 0.005:
                label = 1  # PARTIAL — good profit, almost no upside left

            # Build sample with momentum features at this bar
            sample = {"label": label}
            for f in MOMENTUM_FEATURES:
                if f in feat_cache:
                    val = feat_cache[f][bar] if bar < len(feat_cache[f]) else 0.0
                    sample[f] = 0.0 if np.isnan(val) else float(val)
                else:
                    sample[f] = 0.0
            samples.append(sample)

    return samples


# ── Main ──────────────────────────────────────────────────────────────────────
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
    print(f"  GUARDIAN MOMENTUM — {RUN_NAME}")
    print(f"  Features: {len(MOMENTUM_FEATURES)} momentum/flow indicators")
    print(f"  Coins: {len(coins)}")
    print(f"{'='*70}\n")

    # Stage 1: Generate trades + momentum labels
    print("STAGE 1: Generating TB trades + momentum labels...")
    print("-" * 50)
    all_samples = []
    for sym in coins:
        trades, df = generate_tb_trades(sym)
        if trades is not None:
            samples = label_momentum_samples(trades, df)
            all_samples.extend(samples)
            s_dist = pd.Series([s["label"] for s in samples]).value_counts().to_dict()
            print(f"  [{sym}] {len(trades)} trades -> {len(samples)} samples | "
                  f"HOLD={s_dist.get(0,0)} PARTIAL={s_dist.get(1,0)} EXIT={s_dist.get(2,0)}")

    if not all_samples:
        print("ERROR: No samples!"); sys.exit(1)

    samples_df = pd.DataFrame(all_samples)
    print(f"\n  Total samples: {len(samples_df):,}")
    for i, name in enumerate(["HOLD", "PARTIAL", "FULL_EXIT"]):
        n = (samples_df["label"] == i).sum()
        print(f"  {name}: {n:,} ({n/len(samples_df)*100:.1f}%)")

    # Stage 2: Feature selection
    print(f"\nSTAGE 2: IC Test on momentum features...")
    print("-" * 50)
    avail = [c for c in MOMENTUM_FEATURES if c in samples_df.columns]
    y_target = samples_df["label"].values.astype(np.float64)

    standalone = {}
    for feat in avail:
        x = samples_df[feat].values
        mask = ~(np.isnan(x) | np.isnan(y_target))
        if mask.sum() >= 100:
            c, _ = sc_stats.spearmanr(x[mask], y_target[mask])
            standalone[feat] = float(c) if not np.isnan(c) else 0.0
        else:
            standalone[feat] = 0.0

    ic_sorted = sorted(standalone.items(), key=lambda x: abs(x[1]), reverse=True)
    print(f"  Top features by |IC| vs exit urgency:")
    for f, v in ic_sorted[:12]:
        print(f"    {v:+.4f}  {f}")

    # Stage 3: Train with all momentum features
    print(f"\nSTAGE 3: Training Guardian Momentum...")
    print("-" * 50)
    X = samples_df[avail].ffill().fillna(0).values.astype(np.float64)
    y = samples_df["label"].values.astype(np.int32)
    n_tot = len(X)

    all_metrics = []
    for fold in range(1, GUARDIAN_N_FOLDS + 1):
        v1 = int((fold - 1) / GUARDIAN_N_FOLDS * n_tot)
        v2 = int(fold / GUARDIAN_N_FOLDS * n_tot)
        val_idx = np.arange(v1, v2)
        train_idx = np.setdiff1d(np.arange(n_tot), val_idx)
        model = lgb.LGBMClassifier(**GUARDIAN_PARAMS)
        model.fit(X[train_idx], y[train_idx],
                  eval_set=[(X[val_idx], y[val_idx])],
                  callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(period=-1)])
        yp = model.predict(X[val_idx])
        f1_per = f1_score(y[val_idx], yp, average=None, labels=[0, 1, 2], zero_division=0)
        f1_m = float(f1_score(y[val_idx], yp, average="macro", zero_division=0))
        all_metrics.append({"fold": fold, "f1_macro": round(f1_m, 4),
                            "f1_HOLD": round(float(f1_per[0]), 4),
                            "f1_PARTIAL": round(float(f1_per[1]), 4),
                            "f1_EXIT": round(float(f1_per[2]), 4),
                            "iter": model.best_iteration_})
        logger.info(f"  Fold {fold}: macro={f1_m:.4f} H={f1_per[0]:.4f} P={f1_per[1]:.4f} X={f1_per[2]:.4f}")

    f1s = [m["f1_macro"] for m in all_metrics]
    print(f"\n  CV Macro F1: {np.mean(f1s):.4f} +/- {np.std(f1s):.4f}")

    # Full retrain
    avg_it = int(np.mean([m["iter"] for m in all_metrics]))
    fp = GUARDIAN_PARAMS.copy(); fp["n_estimators"] = max(avg_it, 200)
    final_model = lgb.LGBMClassifier(**fp); final_model.fit(X, y)

    # Save
    print(f"\nSTAGE 4: Saving...")
    joblib.dump(final_model, run_dir / "guardian.pkl")
    with open(run_dir / f"{RUN_NAME}_features.json", "w") as f:
        json.dump(avail, f, indent=2)
    with open(run_dir / f"{RUN_NAME}_cv_results.json", "w") as f:
        json.dump({"mean_f1": round(float(np.mean(f1s)), 4), "std_f1": round(float(np.std(f1s)), 4),
                   "folds": all_metrics}, f, indent=2, default=str)
    with open(run_dir / f"{RUN_NAME}_meta.json", "w") as f:
        json.dump({"run_name": RUN_NAME, "created": datetime.now().isoformat(),
                   "n_samples": len(samples_df), "n_coins": len(coins),
                   "n_features": len(avail), "cv_f1": round(float(np.mean(f1s)), 4)}, f, indent=2)

    # Feature importance
    imp = list(zip(avail, final_model.feature_importances_))
    imp.sort(key=lambda x: x[1], reverse=True)
    print(f"\n  Top 10 feature importance:")
    for i, (f, v) in enumerate(imp[:10]):
        print(f"  {i+1:>2}. {f:<30} {v:>8.0f}")

    print(f"\n{'='*70}")
    print(f"  GUARDIAN MOMENTUM COMPLETE — {RUN_NAME}")
    print(f"  CV F1: {np.mean(f1s):.4f} | Features: {len(avail)}")
    print(f"  Model -> {run_dir / 'guardian.pkl'}")
    print(f"{'='*70}")
