"""
pipeline/06e_train_guardian_momentum_v2.py — Guardian Momentum v2

Perbaikan fundamental:
1. Guardian hanya aktif SETELAH TP tercapai (momentum mode)
2. Label berbasis: setelah TP, apakah harga lanjut atau reversal?
3. Fitur: snapshot kondisi pasar saat TP tercapai
4. Satu keputusan per trade: HOLD (lanjut beyond TP) vs EXIT (ambil profit)

Jalankan: python pipeline/06e_train_guardian_momentum_v2.py --all
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
from sklearn.metrics import f1_score, accuracy_score
from scipy import stats as sc_stats
from config import *

logger = setup_logger("06e_guardian_mom_v2")

RUN_NAME = "tb_guardian_momentum_v2"
TB_MODEL_RUN = "tb_lgbm_widyawardhana_v3"
REGIME_THRESH = {0: 0.45, 1: 0.50, 2: 0.50, 3: 0.45}
TP_ATR = TP_SL_FALLBACK_TP  # 2.0

# Features at TP bar (snapshot of market condition)
GUARDIAN_FEATURES = [
    # === TB v3 features (18) ===
    "etf_gbtc_change_usd", "etf_total_change_usd", "cvd_slope_h4",
    "ofi_h4_delta", "wyckoff_phase", "Sell_Liq", "atr_percentile_h1",
    "stochrsi_k", "dist_liq_50x_short", "funding_rate", "ema_7_h1",
    "dow_cos", "cvd_div_h4", "dist_swing_low", "VAH",
    "cvd_momentum_adv", "dist_from_8h_high", "ema_200_h1",
    # === Additional momentum context ===
    "ofi_z_score", "price_accel_1h", "log_ret_1", "log_ret_5",
    "rsi_6", "vol_ratio_20", "vol_spike_zscore", "trend_strength",
    "h4_trend", "ema_21_slope_h4", "absorption_z",
]

GUARDIAN_PARAMS = {
    "objective": "binary",
    "n_estimators": 500, "learning_rate": 0.03,
    "max_depth": 5, "num_leaves": 31, "min_child_samples": 50,
    "subsample": 0.8, "colsample_bytree": 0.7,
    "verbose": -1, "n_jobs": -1, "random_state": 42,
}

tb_model = joblib.load(MODEL_DIR / "runs" / TB_MODEL_RUN / "lgbm.pkl")
with open(MODEL_DIR / "runs" / TB_MODEL_RUN / f"{TB_MODEL_RUN}_features.json") as f:
    tb_feats = json.load(f)


def generate_trades_with_tp(symbol):
    """Generate TB v3 trades, find which ones hit TP, record TP bar."""
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
    logger.info(f"  [{symbol}] {len(trades)} trades generated")
    return trades, df


def label_tp_momentum(trades, df):
    """
    Momentum labeling untuk Guardian.
    Hanya sample trade yang MENCAPAI TP.

    Di bar TP, cek: apakah setelah TP harga lanjut atau reversal?
    - HOLD (1): best_future > TP × 1.03 → momentum lanjut, bisa dapat lebih
    - EXIT (0): best_future < TP × 0.97 → momentum reversal, harusnya exit
    - (skip trade yang tidak mencapai TP)
    """
    samples = []
    if not trades:
        return samples

    close_arr = df["close"].values
    high_arr = df["high"].values if "high" in df.columns else close_arr
    low_arr = df["low"].values if "low" in df.columns else close_arr
    n = len(close_arr)

    # Cache feature arrays
    feat_cache = {}
    for f in GUARDIAN_FEATURES:
        if f in df.columns:
            feat_cache[f] = df[f].values

    n_tp_hit = 0
    n_continued = 0
    n_reversed = 0

    for t in trades:
        bar_in = t["bar_in"]
        bar_out = t["bar_out"]
        direction = 1 if t["direction"] == "LONG" else -1
        entry_price = t["entry"]
        tp_price = t.get("tp", entry_price * (1 + direction * TP_ATR * 0.01))  # approximate

        # Compute actual TP barrier
        atr_entry = df["atr_14_h1"].values[bar_in] if "atr_14_h1" in df.columns else 1.0
        if direction == 1:
            tp_barrier = entry_price + TP_ATR * atr_entry
        else:
            tp_barrier = entry_price - TP_ATR * atr_entry

        # Find bar where TP was first hit (include bar_out)
        tp_bar = None
        for bar in range(bar_in + 1, min(bar_out + 1, n)):
            if direction == 1 and high_arr[bar] >= tp_barrier:
                tp_bar = bar
                break
            elif direction == -1 and low_arr[bar] <= tp_barrier:
                tp_bar = bar
                break

        if tp_bar is None:
            continue

        n_tp_hit += 1
        current_price = close_arr[tp_bar]

        # After TP bar: check if price continued or reversed
        end_check = min(bar_out, n - 1)

        # Use HIGH/LOW for upside (TP crossed by wicks, not just close)
        if direction == 1:
            future_highs = high_arr[tp_bar:end_check + 1]
            best_future = future_highs.max()
            upside_after_tp = (best_future - tp_barrier) / tp_barrier
        else:
            future_lows = low_arr[tp_bar:end_check + 1]
            best_future = future_lows.min()
            upside_after_tp = (tp_barrier - best_future) / tp_barrier

        if len(future_highs if direction == 1 else future_lows) < 2:
            label = 0; n_reversed += 1
        elif upside_after_tp > 0.005:  # >0.5% more beyond TP
            label = 1; n_continued += 1
        else:
            label = 0; n_reversed += 1

        # Build sample with features at TP bar
        sample = {"label": label}
        for f in GUARDIAN_FEATURES:
            if f in feat_cache:
                val = feat_cache[f][tp_bar] if tp_bar < len(feat_cache[f]) else 0.0
                sample[f] = 0.0 if np.isnan(val) else float(val)
            else:
                sample[f] = 0.0

        # Add trade context
        sample["direction"] = direction
        sample["pnl_at_tp"] = (current_price - entry_price) / entry_price * direction
        sample["bars_to_tp"] = tp_bar - bar_in
        sample["upside_after_tp"] = upside_after_tp

        samples.append(sample)

    if n_tp_hit > 0:
        pass  # logging done in main

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
    print(f"  GUARDIAN MOMENTUM v2 — {RUN_NAME}")
    print(f"  Strategy: Only evaluate at TP bar; HOLD if continued, EXIT if reversed")
    print(f"  Features: {len(GUARDIAN_FEATURES)} (market snapshot at TP)")
    print(f"  Coins: {len(coins)}")
    print(f"{'='*70}\n")

    # Stage 1: Generate TP-momentum samples
    print("STAGE 1: Generating TP-momentum samples...")
    print("-" * 50)
    all_samples = []
    total_tp = 0
    for sym in coins:
        trades, df = generate_trades_with_tp(sym)
        if trades is not None:
            samples = label_tp_momentum(trades, df)
            all_samples.extend(samples)
            n_hold = sum(1 for s in samples if s["label"] == 1)
            n_exit = sum(1 for s in samples if s["label"] == 0)
            total_tp += len(samples)
            print(f"  [{sym}] {len(trades)} trades -> {len(samples)} TP samples | HOLD={n_hold} EXIT={n_exit}")

    if not all_samples:
        print("ERROR: No TP samples!"); sys.exit(1)

    samples_df = pd.DataFrame(all_samples)
    print(f"\n  Total: {len(samples_df):,} TP samples")
    print(f"  HOLD (continue): {(samples_df['label']==1).sum():,} ({(samples_df['label']==1).mean()*100:.1f}%)")
    print(f"  EXIT (reverse):  {(samples_df['label']==0).sum():,} ({(samples_df['label']==0).mean()*100:.1f}%)")
    print(f"  Avg upside after TP (HOLD): {samples_df[samples_df['label']==1]['upside_after_tp'].mean():.4f}")
    print(f"  Avg upside after TP (EXIT): {samples_df[samples_df['label']==0]['upside_after_tp'].mean():.4f}")

    # Stage 2: Feature Selection
    print(f"\nSTAGE 2: Feature Selection (binary HOLD vs EXIT)...")
    print("-" * 50)
    avail = [c for c in GUARDIAN_FEATURES if c in samples_df.columns]
    y_target = samples_df["label"].values.astype(np.float64)

    standalone = {}
    for feat in avail:
        x = samples_df[feat].values
        mask = ~(np.isnan(x) | np.isnan(y_target))
        if mask.sum() >= 50:
            c, _ = sc_stats.spearmanr(x[mask], y_target[mask])
            standalone[feat] = float(c) if not np.isnan(c) else 0.0
        else:
            standalone[feat] = 0.0

    ic_sorted = sorted(standalone.items(), key=lambda x: abs(x[1]), reverse=True)
    print(f"  Top features by |IC| vs HOLD/EXIT:")
    for f, v in ic_sorted[:15]:
        print(f"    {v:+.4f}  {f}")

    # Select features with |IC| >= 0.01 for training
    selected = [f for f, v in ic_sorted if abs(v) >= 0.01][:30]
    print(f"\n  Selected: {len(selected)} features (|IC|>=0.01, max 30)")

    # Stage 3: Train Guardian
    print(f"\nSTAGE 3: Training Guardian v2 ({len(selected)} features)...")
    print("-" * 50)
    X = samples_df[selected].ffill().fillna(0).values.astype(np.float64)
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
                  callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(period=-1)])
        yp = model.predict(X[val_idx])
        f1 = float(f1_score(y[val_idx], yp, average="binary", zero_division=0))
        acc = float(accuracy_score(y[val_idx], yp))
        all_metrics.append({"fold": fold, "f1": round(f1, 4), "acc": round(acc, 4),
                            "iter": model.best_iteration_})
        logger.info(f"  Fold {fold}: F1={f1:.4f} Acc={acc:.4f} Iter={model.best_iteration_}")

    f1s = [m["f1"] for m in all_metrics]
    accs = [m["acc"] for m in all_metrics]
    print(f"\n  CV F1: {np.mean(f1s):.4f} +/- {np.std(f1s):.4f}")
    print(f"  CV Acc: {np.mean(accs):.4f}")
    print(f"  Random baseline: 0.50 (balanced binary)")

    # Full retrain
    avg_it = int(np.mean([m["iter"] for m in all_metrics]))
    fp = GUARDIAN_PARAMS.copy(); fp["n_estimators"] = max(avg_it, 100)
    final_model = lgb.LGBMClassifier(**fp); final_model.fit(X, y)

    # Save
    print(f"\nSTAGE 4: Saving...")
    joblib.dump(final_model, run_dir / "guardian.pkl")
    with open(run_dir / f"{RUN_NAME}_features.json", "w") as f:
        json.dump(selected, f, indent=2)
    with open(run_dir / f"{RUN_NAME}_ic.json", "w") as f:
        json.dump([{"feature": f, "ic": round(v, 4)} for f, v in ic_sorted[:30]], f, indent=2)
    with open(run_dir / f"{RUN_NAME}_cv_results.json", "w") as f:
        json.dump({"mean_f1": round(float(np.mean(f1s)), 4), "std_f1": round(float(np.std(f1s)), 4),
                   "mean_acc": round(float(np.mean(accs)), 4), "folds": all_metrics}, f, indent=2, default=str)
    with open(run_dir / f"{RUN_NAME}_meta.json", "w") as f:
        json.dump({"run_name": RUN_NAME, "created": datetime.now().isoformat(),
                   "n_samples": len(samples_df), "n_coins": len(coins),
                   "n_features": len(selected), "cv_f1": round(float(np.mean(f1s)), 4)}, f, indent=2)

    imp = list(zip(selected, final_model.feature_importances_))
    imp.sort(key=lambda x: x[1], reverse=True)
    print(f"\n  Top 10 features:")
    for i, (f, v) in enumerate(imp[:10]):
        print(f"  {i+1:>2}. {f:<30} {v:>8.0f}")

    print(f"\n{'='*70}")
    print(f"  GUARDIAN MOMENTUM v2 COMPLETE — {RUN_NAME}")
    print(f"  CV F1: {np.mean(f1s):.4f} | Features: {len(selected)}")
    print(f"  Model -> {run_dir / 'guardian.pkl'}")
    print(f"{'='*70}")
