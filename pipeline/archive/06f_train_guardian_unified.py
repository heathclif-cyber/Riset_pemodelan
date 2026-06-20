"""
pipeline/06f_train_guardian_unified.py — Guardian Unified (Momentum + Defense)

Satu model binary (HOLD vs EXIT) untuk dua mode:
  - MOMENTUM: setelah TP, apakah masih ada upside? → HOLD jika ya, EXIT jika tidak
  - DEFENSE:  saat loss, apakah akan kena SL? → HOLD jika tidak, EXIT jika ya

Label: outcome-based — "kalau hold di bar ini, hasilnya lebih baik atau lebih buruk?"
Fitur: flow (OFI/CVD) + risk (divergence/vol) + context (pnl/bars/direction)
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

logger = setup_logger("06f_guardian_unified")

RUN_NAME = "tb_guardian_unified_v1"
TB_MODEL_RUN = "tb_lgbm_widyawardhana_v3"
REGIME_THRESH = {0: 0.45, 1: 0.50, 2: 0.50, 3: 0.45}
TP_ATR = TP_SL_FALLBACK_TP  # 2.0
SL_ATR = TP_SL_FALLBACK_SL  # 1.5

# Unified features: flow + risk + context
GUARDIAN_FEATURES = [
    # Flow indicators (momentum detection)
    "ofi_z_score", "cvd_momentum_adv", "cvd_slope_h4", "ofi_h4_delta",
    "absorption_z", "price_accel_1h",
    # Risk indicators (defense detection)
    "vol_spike_zscore", "vol_ratio_20", "rsi_6", "trend_strength",
    "h4_trend", "ema_21_slope_h4",
    # Market context
    "atr_percentile_h1", "funding_rate", "wyckoff_phase",
    "etf_gbtc_change_usd", "etf_total_change_usd",
    # Note: pnl_pct, bars_held, direction added as dynamic features during inference
]

GUARDIAN_PARAMS = {
    "objective": "binary",
    "n_estimators": 500, "learning_rate": 0.03,
    "max_depth": 5, "num_leaves": 31, "min_child_samples": 30,
    "subsample": 0.8, "colsample_bytree": 0.7,
    "verbose": -1, "n_jobs": -1, "random_state": 42,
}

tb_model = joblib.load(MODEL_DIR / "runs" / TB_MODEL_RUN / "lgbm.pkl")
with open(MODEL_DIR / "runs" / TB_MODEL_RUN / f"{TB_MODEL_RUN}_features.json") as f:
    tb_feats = json.load(f)


def generate_trades_and_samples(symbol):
    """Generate TB trades + per-bar Guardian samples with outcome-based labels."""
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

    # Feature cache
    feat_cache = {}
    for f in GUARDIAN_FEATURES:
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
    n_momentum, n_defense = 0, 0

    for t in trades:
        bar_in, bar_out = t["bar_in"], t["bar_out"]
        direction = 1 if t["direction"] == "LONG" else -1
        entry_price = t["entry"]
        atr_entry = atr_arr[bar_in]
        tp_barrier = entry_price + direction * TP_ATR * atr_entry
        sl_barrier = entry_price - direction * SL_ATR * atr_entry

        tp_hit = False
        for bar in range(bar_in + 1, min(bar_out + 1, n)):
            bars_held = bar - bar_in
            if bars_held < 2:
                continue

            current_price = close_arr[bar]
            if direction == 1:
                pnl_pct = (current_price - entry_price) / entry_price
            else:
                pnl_pct = (entry_price - current_price) / entry_price

            # Check if TP was already hit
            if not tp_hit:
                if direction == 1 and high_arr[bar] >= tp_barrier:
                    tp_hit = True
                elif direction == -1 and low_arr[bar] <= tp_barrier:
                    tp_hit = True

            # Determine label: if we HOLD, will outcome be better or worse?
            # Compare current exit vs future best exit
            future_end = min(bar_out, n - 1)
            future_prices = close_arr[bar:future_end + 1]
            if len(future_prices) < 2:
                continue

            if direction == 1:
                future_best = future_prices.max()
                future_worst = future_prices.min()
            else:
                future_best = future_prices.min()
                future_worst = future_prices.max()

            future_best_pnl = (future_best - entry_price) / entry_price * direction
            future_worst_pnl = (future_worst - entry_price) / entry_price * direction
            current_pnl = pnl_pct

            # Label: HOLD(1) if holding leads to better outcome, EXIT(0) if holding leads to worse
            # Better = best future PnL is significantly higher than current PnL
            upside = future_best_pnl - current_pnl
            downside = current_pnl - future_worst_pnl

            if upside > 0.005 and upside > downside * 0.5:
                label = 1  # HOLD — still has meaningful upside
            else:
                label = 0  # EXIT — limited upside, risk of reversal

            # Context features
            sample = {"label": label, "pnl_pct": pnl_pct, "bars_held_norm": min(bars_held / MAX_HOLDING_BARS, 1.0),
                      "direction": direction, "tp_hit": int(tp_hit)}

            for f in GUARDIAN_FEATURES:
                if f in feat_cache:
                    val = feat_cache[f][bar] if bar < len(feat_cache[f]) else 0.0
                    sample[f] = 0.0 if np.isnan(val) else float(val)
                else:
                    sample[f] = 0.0

            samples.append(sample)
            if tp_hit: n_momentum += 1
            else: n_defense += 1

    logger.info(f"  [{symbol}] {len(trades)} trades -> {len(samples)} samples (mom={n_momentum} def={n_defense})")
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
    print(f"  GUARDIAN UNIFIED — {RUN_NAME}")
    print(f"  One model, two modes: HOLD if upside remains, EXIT if not")
    print(f"  Features: {len(GUARDIAN_FEATURES)} static + 4 dynamic (pnl,bars,dir,tp_hit)")
    print(f"{'='*70}\n")

    # Stage 1: Generate samples
    print("STAGE 1: Generating unified samples...")
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

    # Break down by mode
    mom = samples_df[samples_df["tp_hit"] == 1]
    def_ = samples_df[samples_df["tp_hit"] == 0]
    print(f"  Momentum (after TP): {len(mom):,} samples | HOLD: {(mom['label']==1).sum()/len(mom)*100:.1f}%")
    print(f"  Defense (before TP): {len(def_):,} samples | HOLD: {(def_['label']==1).sum()/len(def_)*100:.1f}%")

    # Stage 2: Feature selection
    print(f"\nSTAGE 2: IC Test...")
    static_only = [c for c in GUARDIAN_FEATURES if c in samples_df.columns]
    # Add dynamic context features
    all_f = static_only + ["pnl_pct", "bars_held_norm", "direction", "tp_hit"]
    y_target = samples_df["label"].values.astype(np.float64)

    standalone = {}
    for feat in all_f:
        if feat in samples_df.columns:
            x = samples_df[feat].values
            mask = ~(np.isnan(x) | np.isnan(y_target))
            if mask.sum() >= 100:
                c, _ = sc_stats.spearmanr(x[mask], y_target[mask])
                standalone[feat] = float(c) if not np.isnan(c) else 0.0
            else:
                standalone[feat] = 0.0

    ic_sorted = sorted(standalone.items(), key=lambda x: abs(x[1]), reverse=True)
    print(f"  Top IC vs HOLD/EXIT:")
    for f, v in ic_sorted[:15]:
        print(f"    {v:+.4f}  {f}")

    selected = [f for f, v in ic_sorted if abs(v) >= 0.01]
    print(f"\n  Selected: {len(selected)} features (|IC|>=0.01)")

    # Stage 3: Train
    print(f"\nSTAGE 3: Training Unified Guardian ({len(selected)} features)...")
    X_all = samples_df[selected].ffill().fillna(0).values.astype(np.float64)
    y_all = samples_df["label"].values.astype(np.int32)
    n_tot = len(X_all)

    all_metrics = []
    for fold in range(1, GUARDIAN_N_FOLDS + 1):
        v1 = int((fold - 1) / GUARDIAN_N_FOLDS * n_tot)
        v2 = int(fold / GUARDIAN_N_FOLDS * n_tot)
        val_idx = np.arange(v1, v2)
        train_idx = np.setdiff1d(np.arange(n_tot), val_idx)

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
        print(f"  {i+1:>2}. {f:<30} {v:>8.0f}")

    print(f"\n{'='*70}")
    print(f"  GUARDIAN UNIFIED COMPLETE — {RUN_NAME}")
    print(f"  CV F1: {np.mean(f1s):.4f} | Features: {len(selected)}")
    print(f"  Model -> {run_dir / 'guardian.pkl'}")
    print(f"{'='*70}")
