"""
pipeline/06c_train_guardian_tb.py — Guardian khusus TB v3 (widyawardhana)

Jalankan:
  python pipeline/06c_train_guardian_tb.py --all
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
from config import *

logger = setup_logger("06c_guardian_tb")

RUN_NAME = "tb_guardian_widyawardhana_v1"
TB_MODEL_RUN = "tb_lgbm_widyawardhana_v3"
REGIME_THRESH = {0: 0.45, 1: 0.50, 2: 0.50, 3: 0.45}

# 18 TB features + 7 dynamic
GUARDIAN_STATIC = [
    "etf_gbtc_change_usd", "etf_total_change_usd", "cvd_slope_h4",
    "ofi_h4_delta", "wyckoff_phase", "Sell_Liq", "atr_percentile_h1",
    "stochrsi_k", "dist_liq_50x_short", "funding_rate", "ema_7_h1",
    "dow_cos", "cvd_div_h4", "dist_swing_low", "VAH",
    "cvd_momentum_adv", "dist_from_8h_high", "ema_200_h1",
]
GUARDIAN_DYNAMIC = [
    "bars_held_norm", "current_pnl_pct", "current_pnl_atr",
    "max_favorable_pnl_pct", "drawdown_from_peak_pct",
    "direction", "entry_price_ratio",
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

# ── Load TB v3 model ─────────────────────────────────────────────────────────
tb_model = joblib.load(MODEL_DIR / "runs" / TB_MODEL_RUN / "lgbm.pkl")
with open(MODEL_DIR / "runs" / TB_MODEL_RUN / f"{TB_MODEL_RUN}_features.json") as f:
    tb_feats = json.load(f)
logger.info(f"TB v3 model loaded: {len(tb_feats)} features")


def generate_tb_trades(symbol):
    """Run TB v3 on training data, simulate trades, return trade trajectories."""
    path = LABEL_DIR / f"{symbol}_features_v3.parquet"
    if not path.exists():
        return None, None

    df = pd.read_parquet(path)
    df = ensure_utc_index(df).sort_index()
    df = df[df.index < TRAIN_CUTOFF_DATE]
    if len(df) < 100:
        return None, None

    # Merge HMM
    reg_path = LABEL_DIR / f"{symbol}_regime_h1.parquet"
    hmm = np.full(len(df), 1, dtype=np.int32)
    if reg_path.exists():
        reg = pd.read_parquet(reg_path)
        if "hmm_regime_enc" in reg.columns:
            hmm = reg["hmm_regime_enc"].reindex(df.index, fill_value=1).values.astype(np.int32)

    # TB inference
    X = np.zeros((len(df), len(tb_feats)), dtype=np.float64)
    for i, c in enumerate(tb_feats):
        if c in df.columns:
            X[:, i] = df[c].ffill().fillna(0).values.astype(np.float64)
    proba = tb_model.predict_proba(X)
    conf = np.max(proba, axis=1)
    y_pred = np.argmax(proba, axis=1)
    for r, th in REGIME_THRESH.items():
        y_pred[(hmm == r) & (y_pred != 1) & (conf < th)] = 1

    close_arr = df["close"].values
    high_arr = df["high"].values if "high" in df.columns else close_arr
    low_arr = df["low"].values if "low" in df.columns else close_arr
    atr_arr = df["atr_14_h1"].values if "atr_14_h1" in df.columns else np.ones(len(df))

    # Simulate trades (isolated, no Guardian)
    result = simulate_trades_swing(
        y_pred=y_pred, close=close_arr, high=high_arr, low=low_arr, atr=atr_arr,
        h4_swing_highs=np.full(len(df), np.nan),
        h4_swing_lows=np.full(len(df), np.nan),
        modal=MODAL_PER_TRADE, leverage=LEVERAGE_SIM[0],
        fee_per_side=FEE_PER_SIDE, slippage=SLIPPAGE_PER_SIDE,
        max_hold=MAX_HOLDING_BARS,
        min_rr=SWING_LABEL_MIN_RR, min_tp_atr=SWING_LABEL_MIN_TP,
        max_sl_atr=SWING_LABEL_MAX_SL,
        tp_fallback_atr=TP_SL_FALLBACK_TP, sl_fallback_atr=TP_SL_FALLBACK_SL,
        guardian_enabled=False, trailing_stop_enabled=False,
    )
    trades = result.get("trades", [])
    logger.info(f"  [{symbol}] {len(trades)} trades generated")
    return trades, df


def label_guardian_samples(trades, df):
    """For each bar in each trade, generate HOLD/PARTIAL_EXIT/FULL_EXIT labels + static features."""
    samples = []
    if not trades:
        return samples

    close_arr = df["close"].values
    atr_arr = df["atr_14_h1"].values if "atr_14_h1" in df.columns else np.ones(len(df))
    n = len(close_arr)

    # Cache static feature arrays
    static_cache = {}
    for sf in GUARDIAN_STATIC:
        if sf in df.columns:
            static_cache[sf] = df[sf].values

    for t in trades:
        bar_in = t["bar_in"]
        bar_out = t["bar_out"]
        direction = 1 if t["direction"] == "LONG" else -1
        entry_price = t["entry"]
        atr_entry = atr_arr[bar_in] if bar_in < n else 1.0

        # Walk through each bar of the trade
        for bar in range(bar_in, min(bar_out, n)):
            bars_held = bar - bar_in + 1
            if bars_held < 2:  # min hold = 2
                continue

            current_price = close_arr[bar]
            if direction == 1:
                pnl_pct = (current_price - entry_price) / entry_price
            else:
                pnl_pct = (entry_price - current_price) / entry_price

            # MFE
            future_prices = close_arr[bar:min(bar_out, n)]
            if len(future_prices) > 0:
                if direction == 1:
                    best_future = future_prices.max()
                else:
                    best_future = future_prices.min()
                mfe = (best_future - entry_price) / entry_price if direction == 1 else \
                     (entry_price - best_future) / entry_price
                mfe = max(0, mfe)
            else:
                mfe = max(0, pnl_pct)

            # DD from peak
            prices_since_entry = close_arr[bar_in:bar + 1]
            if direction == 1:
                peak_so_far = prices_since_entry.max()
                dd_pct = (peak_so_far - current_price) / entry_price
            else:
                peak_so_far = prices_since_entry.min()
                dd_pct = (current_price - peak_so_far) / entry_price

            # Future upside
            future_all = close_arr[bar:min(bar_out + 1, n)]
            if len(future_all) > 1:
                if direction == 1:
                    future_best = future_all.max()
                else:
                    future_best = future_all.min()
                upside = (future_best - current_price) / current_price if direction == 1 else \
                         (current_price - future_best) / current_price
            else:
                upside = 0

            # Guardian Label
            atr_now = atr_arr[bar] if bar < n else atr_entry
            pnl_atr = pnl_pct * entry_price / atr_now if atr_now > 0 else 0
            label = 0
            if pnl_atr < -1.0:
                label = 2
            elif mfe > 0.015 and pnl_pct < mfe * 0.25:
                label = 2
            elif upside < 0.01 and pnl_pct > 0.005:
                label = 2
            elif mfe > 0.015 and pnl_pct < mfe * 0.55:
                label = 1
            elif pnl_pct > 0.008 and upside < 0.03:
                label = 1
            elif bars_held >= MAX_HOLDING_BARS - 2:
                label = 2

            atr_now_val = atr_arr[bar] if bar < n else atr_entry
            sample = {
                "bars_held_norm": min(bars_held / MAX_HOLDING_BARS, 1.0),
                "current_pnl_pct": pnl_pct,
                "current_pnl_atr": pnl_pct * entry_price / atr_now_val if atr_now_val > 0 else 0,
                "max_favorable_pnl_pct": mfe,
                "drawdown_from_peak_pct": dd_pct,
                "direction": direction,
                "entry_price_ratio": current_price / entry_price,
                "label": label,
            }
            # Add static features at this bar
            for sf in GUARDIAN_STATIC:
                sample[sf] = static_cache[sf][bar] if sf in static_cache and bar < len(static_cache[sf]) else 0.0

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
    print(f"  TB GUARDIAN TRAINING — {RUN_NAME}")
    print(f"  Static: {len(GUARDIAN_STATIC)} features")
    print(f"  Dynamic: {len(GUARDIAN_DYNAMIC)} features")
    print(f"  Coins: {len(coins)}")
    print(f"{'='*70}\n")

    # Stage 1: Generate trade trajectories + label samples
    print("STAGE 1: Generating TB trade trajectories + Guardian labels...")
    print("-" * 50)
    all_samples = []
    for sym in coins:
        trades, df = generate_tb_trades(sym)
        if trades is not None:
            samples = label_guardian_samples(trades, df)
            all_samples.extend(samples)
            s_dist = pd.Series([s["label"] for s in samples]).value_counts().to_dict()
            print(f"  [{sym}] {len(trades)} trades -> {len(samples)} samples | "
                  f"HOLD={s_dist.get(0,0)} PARTIAL={s_dist.get(1,0)} FULL={s_dist.get(2,0)}")

    if not all_samples:
        print("ERROR: No samples generated!")
        sys.exit(1)

    samples_df = pd.DataFrame(all_samples)
    print(f"\n  Total samples: {len(samples_df):,}")
    for i, name in enumerate(["HOLD", "PARTIAL_EXIT", "FULL_EXIT"]):
        n = (samples_df["label"] == i).sum()
        print(f"  {name}: {n:,} ({n/len(samples_df)*100:.1f}%)")

    # Stage 2: Feature selection
    print(f"\nSTAGE 2: Multistage Feature Selection...")
    print("-" * 50)

    # All features: static + dynamic
    all_feats = GUARDIAN_STATIC + GUARDIAN_DYNAMIC
    avail = [c for c in all_feats if c in samples_df.columns]
    print(f"  Features: {len(avail)} ({len(GUARDIAN_STATIC)} static + {len(GUARDIAN_DYNAMIC)} dynamic)")

    # Encode target as ordinal
    from scipy import stats as sc_stats
    y_target = samples_df["label"].values.astype(np.float64)
    # HOLD=0, PARTIAL=1, FULL=2 -> these are meaningful ordinal for exit urgency
    y_ord = y_target  # already 0,1,2

    # Standalone IC
    n_total = len(samples_df)
    standalone = {}
    for feat in avail:
        x = samples_df[feat].values
        mask = ~(np.isnan(x) | np.isnan(y_ord))
        if mask.sum() >= 100:
            c, _ = sc_stats.spearmanr(x[mask], y_ord[mask])
            standalone[feat] = float(c) if not np.isnan(c) else 0.0
        else:
            standalone[feat] = 0.0

    # Sort by |IC|
    ic_sorted = sorted(standalone.items(), key=lambda x: abs(x[1]), reverse=True)
    print(f"\n  Top 15 features by |IC|:")
    for f, v in ic_sorted[:15]:
        print(f"    {v:+.4f}  {f}")

    # Marginal IC via Gram-Schmidt (50K sample)
    max_gs = 50000
    if n_total > max_gs:
        rng = np.random.RandomState(42)
        idx = rng.choice(n_total, max_gs, replace=False)
        X_gs = samples_df[avail].values[idx].astype(np.float64)
        y_gs = y_ord[idx]
        print(f"  Gram-Schmidt: downsampled to {max_gs:,} rows")
    else:
        X_gs = samples_df[avail].values.astype(np.float64)
        y_gs = y_ord

    # Rank normalize
    def _rn(x):
        x = np.where(np.isnan(x), np.nanmedian(x) if not np.isnan(np.nanmedian(x)) else 0.0, x)
        r = sc_stats.rankdata(x).astype(np.float64); r -= r.mean(); s = r.std()
        return r / s if s > 1e-10 else np.zeros_like(r)
    def _po(vec, pivot):
        nq = np.dot(pivot, pivot)
        return vec - (np.dot(vec, pivot)/nq)*pivot if nq > 1e-10 else vec.copy()

    X_r = np.column_stack([_rn(X_gs[:, j]) for j in range(len(avail))])
    y_r = _rn(y_gs)
    remaining = list(range(len(avail)))
    marginal = {}
    for _ in range(len(avail)):
        if not remaining: break
        corrs = np.zeros(len(remaining))
        for k, j in enumerate(remaining):
            xj = X_r[:, j]; nx = np.sqrt(np.dot(xj, xj)); ny = np.sqrt(np.dot(y_r, y_r))
            corrs[k] = np.dot(xj, y_r)/(nx*ny) if nx>1e-10 and ny>1e-10 else 0.0
        best_j = remaining[int(np.argmax(np.abs(corrs)))]
        marginal[avail[best_j]] = float(corrs[np.argmax(np.abs(corrs))])
        pivot = X_r[:, best_j].copy()
        for j in remaining:
            if j != best_j: X_r[:, j] = _po(X_r[:, j], pivot)
        y_r = _po(y_r, pivot); remaining.remove(best_j)

    # Auto-correlation correction
    AUTOCORR = 24
    def tstat(ic, n):
        ne = max(n//AUTOCORR, 10); denom = np.sqrt(max(1.0-ic**2, 1e-10))
        return ic*np.sqrt(ne)/denom

    # Verdict
    verdicts = {}
    for feat in avail:
        sa = standalone[feat]; ts = tstat(sa, n_total); mg = marginal.get(feat, 0.0)
        sa_ok = abs(sa) >= 0.02 and abs(ts) >= 2.0; mg_ok = abs(mg) >= 0.01
        if sa_ok and mg_ok: verdicts[feat] = "KEEP"
        elif sa_ok and not mg_ok: verdicts[feat] = "REDUNDANT"
        elif not sa_ok and mg_ok: verdicts[feat] = "WEAK"
        else: verdicts[feat] = "DROP"

    keep = [f for f, v in verdicts.items() if v == "KEEP"]
    weak = [f for f, v in verdicts.items() if v == "WEAK"]
    print(f"\n  Verdict: KEEP={len(keep)} REDUNDANT={sum(1 for v in verdicts.values() if v=='REDUNDANT')} "
          f"WEAK={len(weak)} DROP={sum(1 for v in verdicts.values() if v=='DROP')}")
    print(f"  KEEP: {keep}")
    print(f"  WEAK: {weak}")

    # Stage 3: Train with different feature sets
    print(f"\nSTAGE 3: Training Guardian with different feature sets...")
    print("-" * 50)

    configs = [
        ("KEEP only", keep),
        ("KEEP + WEAK", keep + weak),
        ("ALL features", avail),
    ]

    best_config, best_f1 = None, -1
    all_results = {}

    for cname, cfeats in configs:
        avail_c = [c for c in cfeats if c in samples_df.columns]
        X = samples_df[avail_c].ffill().fillna(0).values.astype(np.float64)
        y = samples_df["label"].values.astype(np.int32)
        n_tot = len(X)

        fold_metrics = []
        for fold in range(1, GUARDIAN_N_FOLDS + 1):
            val_start = int((fold - 1) / GUARDIAN_N_FOLDS * n_tot)
            val_end = int(fold / GUARDIAN_N_FOLDS * n_tot)
            val_idx = np.arange(val_start, val_end)
            train_idx = np.setdiff1d(np.arange(n_tot), val_idx)

            model = lgb.LGBMClassifier(**GUARDIAN_PARAMS)
            model.fit(X[train_idx], y[train_idx],
                      eval_set=[(X[val_idx], y[val_idx])],
                      callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(period=-1)])
            yp = model.predict(X[val_idx])
            f1_m = float(f1_score(y[val_idx], yp, average="macro", zero_division=0))
            fold_metrics.append({"fold": fold, "f1_macro": round(f1_m, 4), "best_iter": model.best_iteration_})

        mean_f1 = float(np.mean([m["f1_macro"] for m in fold_metrics]))
        std_f1 = float(np.std([m["f1_macro"] for m in fold_metrics]))
        all_results[cname] = {"features": len(avail_c), "f1_mean": round(mean_f1, 4), "f1_std": round(std_f1, 4)}

        # Full retrain best config
        avg_it = int(np.mean([m["best_iter"] for m in fold_metrics]))
        fp = GUARDIAN_PARAMS.copy(); fp["n_estimators"] = max(avg_it, 200)
        fm = lgb.LGBMClassifier(**fp); fm.fit(X, y)

        print(f"  {cname:<20}: {len(avail_c):>3} feats, F1={mean_f1:.4f}+/-{std_f1:.4f}, iter={avg_it}")

        if mean_f1 > best_f1:
            best_f1, best_config = mean_f1, cname
            # Save best model
            best_model = fm
            best_feats = avail_c

    print(f"\n  BEST: {best_config} (F1={best_f1:.4f})")

    # Stage 4: Save
    print(f"\nSTAGE 4: Saving best model ({best_config})...")
    joblib.dump(best_model, run_dir / "guardian.pkl")
    with open(run_dir / f"{RUN_NAME}_features.json", "w") as f:
        json.dump(best_feats, f, indent=2)
    with open(run_dir / f"{RUN_NAME}_feature_selection.json", "w") as f:
        json.dump({"verdicts": verdicts, "keep": keep, "weak": weak,
                   "best_config": best_config, "configs": all_results,
                   "ic_results": [{"feature": f, "standalone_ic": round(standalone[f],4),
                                   "marginal_ic": round(marginal.get(f,0),4), "verdict": verdicts[f]}
                                  for f in avail]}, f, indent=2, default=str)
    with open(run_dir / f"{RUN_NAME}_meta.json", "w") as f:
        json.dump({"run_name": RUN_NAME, "created": datetime.now().isoformat(),
                   "n_samples": len(samples_df), "n_coins": len(coins),
                   "best_config": best_config, "n_features": len(best_feats),
                   "cv_f1": round(best_f1, 4)}, f, indent=2)
    print(f"  Model -> {run_dir / 'guardian.pkl'}")
    print(f"  Features: {len(best_feats)} ({best_config})")
    print(f"\n{'='*70}")
    print(f"  TB GUARDIAN COMPLETE — {RUN_NAME}")
    print(f"  CV F1: {best_f1:.4f} | Config: {best_config} | Features: {len(best_feats)}")
    print(f"{'='*70}")
