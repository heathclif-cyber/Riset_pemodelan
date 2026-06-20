#!/usr/bin/env python3
"""
pipeline/06c_feature_selection_shap.py
Multi-method feature selection for TB LGBM.

Methods (union rule — masuk jika lolos >= 1 metode):
  1. IC          : |Spearman IC| >= 0.02 AND |t-stat| >= 2.0 (N_eff = N/24)
  2. Mutual Info : MI >= 0.005 bits (sklearn, non-linear)
  3. LGBM Gain   : gain >= mean(all gains) — top half
  4. SHAP        : mean |SHAP| >= mean(all mean |SHAP|) — top half

Output: models/runs/tb_fs_shap_v1/
  fs_results.json   — skor semua metode per fitur
  fs_selected.json  — final union features
  shap_summary.json — SHAP stats
"""

import sys, json, warnings, argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.feature_selection import mutual_info_classif
import lightgbm as lgb
import joblib

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from config import (
    LABEL_DIR, TRAINING_COINS, TRAIN_CUTOFF_DATE,
    LGBM_PARAMS, LGBM_EARLY_STOPPING, LGBM_CLASS_WEIGHTS,
    TP_SL_FALLBACK_TP, TP_SL_FALLBACK_SL, MAX_HOLDING_BARS,
    N_FOLDS, TB_PURGE_GAP_BARS,
)
from core.features import triple_barrier_labeling

OUT_DIR = ROOT / "models" / "runs" / "tb_fs_shap_v1"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── ETF (look-ahead bias — confirmed IC 0.016 @ shift=24)
ETF_BLACKLIST = {"etf_total_change_usd", "etf_gbtc_change_usd"}

# ── Candidate pool — FEATURE_COLS_V3 yang tersedia di parquet + derived
BASE_CANDIDATES = [
    "volume_delta", "cvd", "buy_volume", "sell_volume",
    "MSB_BOS", "CHoCH", "bars_since_BOS",
    "FVG_up", "FVG_down", "Buy_Liq", "Sell_Liq", "SFP_sweep",
    "open_interest", "dynamic_position_pressure", "funding_rate",
    "ema_7_h1", "ema_21_h1", "ema_50_h1", "ema_200_h1",
    "ema_7_h4", "ema_21_h4", "ema_50_h4", "ema_200_h4",
    "rsi_6", "stochrsi_k", "stochrsi_d",
    "atr_14_h1", "atr_14_h4",
    "PDH", "PDL", "PWH", "PWL", "Fib_618", "Fib_786",
    "POC", "VAH", "VAL",
    "market_session", "btc_dominance", "fear_greed",
    "log_ret_1", "log_ret_5", "log_ret_20", "vol_ratio_20",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
    "time_to_funding_norm",
    "long_short_ratio",
    "dist_swing_high", "dist_swing_low", "price_in_range", "swing_momentum",
    "h4_trend", "trend_strength", "vol_regime",
    "cvd_div_h4", "cvd_slope_h4",
    "vol_efficiency", "absorption_z",
    "funding_price_div",
    "rsi_h4", "rsi_divergence",
    "wyckoff_phase", "spring_upthrust",
    "ofi_raw", "ofi_acceleration", "ofi_z_score", "ofi_h4_delta",
    "vwdp", "vwdp_smooth",
    "hidden_divergence", "cvd_momentum_adv",
    "absorption_at_swing",
    "spread_to_volume", "ultra_high_vol", "no_demand", "no_supply", "effort_vs_result",
    "ema_21_slope_h4", "ema_50_slope_h4", "price_vs_ema_50_h4",
    "rsi_slope_h4", "atr_percent_h4", "range_expansion_h4",
    "trend_accel_4h", "vol_price_confirm", "dist_from_8h_high",
    "relative_strength_z", "relative_strength_momentum",
    "dist_liq_50x_long", "dist_liq_20x_long",
    "dist_liq_50x_short", "dist_liq_20x_short",
    "whale_retail_divergence",
    "atr_zscore_20d", "atr_percentile_h1", "vol_spike_zscore",
    "price_accel_1h", "ofi_momentum_ratio", "vol_accel_3h",
    # hmm sebagai fitur (ic32 pakai ini)
    "hmm_regime_enc",
]

DERIVED_CANDIDATES = [
    "candle_body_ratio",
    "upper_wick_ratio",
    "lower_wick_ratio",
    "candle_range_atr_ratio",
    "funding_rate_change_8h",
    "funding_rate_zscore_20d",
    "oi_pct_1d",
]

ALL_CANDIDATES = BASE_CANDIDATES + DERIVED_CANDIDATES


# ─────────────────────────────────────────────────────────────────────────────
def compute_derived(df: pd.DataFrame) -> pd.DataFrame:
    hl = (df["high"] - df["low"]).replace(0, np.nan)
    body = (df["close"] - df["open"]).abs()
    df["candle_body_ratio"]      = body / hl
    df["upper_wick_ratio"]       = (df["high"] - df[["open","close"]].max(axis=1)) / hl
    df["lower_wick_ratio"]       = (df[["open","close"]].min(axis=1) - df["low"]) / hl
    df["candle_range_atr_ratio"] = hl / df["atr_14_h1"].replace(0, np.nan)
    df["funding_rate_change_8h"] = df["funding_rate"].diff(8)
    fr_mean = df["funding_rate"].rolling(480, min_periods=100).mean()
    fr_std  = df["funding_rate"].rolling(480, min_periods=100).std()
    df["funding_rate_zscore_20d"] = (df["funding_rate"] - fr_mean) / fr_std.replace(0, np.nan)
    oi_prev = df["open_interest"].shift(24)
    df["oi_pct_1d"] = (df["open_interest"] - oi_prev) / oi_prev.abs().replace(0, np.nan)
    return df


def build_purged_folds(n, n_folds, purge_gap):
    fold_size = n // n_folds
    folds = []
    for k in range(n_folds):
        val_start = k * fold_size
        val_end   = val_start + fold_size if k < n_folds - 1 else n
        tr_end    = max(0, val_start - purge_gap)
        if tr_end < 50:
            continue
        train_idx = list(range(tr_end))
        val_idx   = list(range(val_start, val_end))
        folds.append((train_idx, val_idx))
    return folds


def spearman_ic(x, y, n_eff_divisor=24):
    mask = ~(np.isnan(x) | np.isnan(y))
    if mask.sum() < 50:
        return 0.0, 0.0, mask.sum()
    c, _ = stats.spearmanr(x[mask], y[mask])
    n_eff = max(mask.sum() // n_eff_divisor, 2)
    denom = max(1 - c**2, 1e-10)
    t = c * np.sqrt(n_eff) / np.sqrt(denom)
    return float(c), float(t), int(mask.sum())


# ─────────────────────────────────────────────────────────────────────────────
def load_data():
    frames = []
    print(f"Loading {len(TRAINING_COINS)} coins ...")
    for coin in TRAINING_COINS:
        p = LABEL_DIR / f"{coin}_features_v3.parquet"
        if not p.exists():
            print(f"  SKIP {coin} (no parquet)")
            continue
        df = pd.read_parquet(p)
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index, utc=True)
        df = df[df.index < TRAIN_CUTOFF_DATE].copy()
        if len(df) < 500:
            print(f"  SKIP {coin} (<500 rows)")
            continue

        # TB label
        tb = triple_barrier_labeling(
            df["close"], df["high"], df["low"], df["atr_14_h1"],
            TP_SL_FALLBACK_TP, TP_SL_FALLBACK_SL, MAX_HOLDING_BARS,
        )
        df["tb_label"] = tb.map({"SHORT": -1, "FLAT": 0, "LONG": 1})
        df["tb_cls"]   = tb.map({"SHORT": 0,  "FLAT": 1, "LONG": 2})

        df = compute_derived(df)
        frames.append(df)
        print(f"  {coin}: {len(df):,} rows")

    combined = pd.concat(frames).sort_index()
    print(f"\nTotal pooled: {len(combined):,} rows")
    return combined


# ─────────────────────────────────────────────────────────────────────────────
def method_ic(df, candidates, y_col="tb_label"):
    print("\n[Method 1] IC (Spearman) ...")
    y = df[y_col].values
    results = {}
    for feat in candidates:
        if feat not in df.columns:
            results[feat] = {"ic": 0.0, "t": 0.0, "n": 0, "pass_ic": False}
            continue
        ic, t, n = spearman_ic(df[feat].values, y)
        pass_ic = abs(ic) >= 0.02 and abs(t) >= 2.0
        results[feat] = {"ic": round(ic, 5), "t": round(t, 3), "n": n, "pass_ic": bool(pass_ic)}
        if pass_ic:
            print(f"  PASS  {feat:<35} IC={ic:+.4f} t={t:+.2f}")
    n_pass = sum(v["pass_ic"] for v in results.values())
    print(f"  IC PASS: {n_pass}/{len(candidates)}")
    return results


def method_mi(df, candidates, y_col="tb_cls"):
    print("\n[Method 2] Mutual Information ...")
    # subset ke baris yang punya y
    valid_mask = df[y_col].notna().values
    df_v = df[valid_mask].copy()
    y = df_v[y_col].values.astype(int)

    mi_vals = {}
    # Build matrix hanya dari candidates yang ada, fill NaN with median + clip inf
    cols_avail = [f for f in candidates if f in df_v.columns]
    X = df_v[cols_avail].copy()
    for c in cols_avail:
        X[c] = X[c].replace([np.inf, -np.inf], np.nan)
        med = X[c].median()
        if np.isnan(med):
            med = 0.0
        X[c] = X[c].fillna(med)
    X = X.values.astype(np.float64)

    mi = mutual_info_classif(X, y, random_state=42, n_neighbors=5)
    for feat, val in zip(cols_avail, mi):
        mi_vals[feat] = float(val)
    # Missing candidates
    for feat in candidates:
        if feat not in mi_vals:
            mi_vals[feat] = 0.0

    threshold = 0.005
    n_pass = sum(v >= threshold for v in mi_vals.values())
    print(f"  MI threshold: {threshold}  |  PASS: {n_pass}/{len(candidates)}")
    for feat, val in sorted(mi_vals.items(), key=lambda x: -x[1])[:20]:
        flag = "PASS" if val >= threshold else "    "
        print(f"  {flag}  {feat:<35} MI={val:.5f}")
    return mi_vals, threshold


# ─────────────────────────────────────────────────────────────────────────────
def method_lgbm_shap(df, candidates, y_col="tb_cls"):
    print("\n[Method 3+4] LGBM Gain + SHAP ...")

    try:
        import shap as shap_lib
        has_shap = True
        print("  shap library: OK")
    except ImportError:
        has_shap = False
        print("  shap library: NOT FOUND — skipping SHAP (gain only)")

    # Filter candidates: available + not all-NaN
    cols_use = []
    for f in candidates:
        if f not in df.columns:
            continue
        nan_frac = df[f].isna().mean()
        if nan_frac > 0.5:
            print(f"  SKIP {f} (NaN {nan_frac:.0%})")
            continue
        cols_use.append(f)

    print(f"  Training on {len(cols_use)} candidates ...")

    valid_mask = df[y_col].notna().values
    df_v = df[valid_mask].copy()
    y = df_v[y_col].values.astype(int)
    X = df_v[cols_use].copy()
    for c in cols_use:
        med = X[c].median()
        X[c] = X[c].fillna(med)
    X = X.values.astype(np.float32)

    folds = build_purged_folds(len(df_v), N_FOLDS, TB_PURGE_GAP_BARS)
    print(f"  {len(folds)} folds")

    gain_per_fold   = []
    shap_per_fold   = []
    oof_preds       = np.zeros((len(df_v), 3))

    params = dict(LGBM_PARAMS)
    params["verbose"] = -1
    params["n_estimators"] = 300  # faster for selection pass

    for i, (tr_idx, val_idx) in enumerate(folds):
        X_tr, y_tr = X[tr_idx], y[tr_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        sw = np.array([LGBM_CLASS_WEIGHTS[yi] for yi in y_tr], dtype=np.float32)

        model = lgb.LGBMClassifier(**params)
        model.fit(
            X_tr, y_tr,
            sample_weight=sw,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(20, verbose=False), lgb.log_evaluation(-1)],
        )

        oof_preds[val_idx] = model.predict_proba(X_val)
        gain = model.booster_.feature_importance(importance_type="gain")
        gain_per_fold.append(gain)

        if has_shap:
            explainer = shap_lib.TreeExplainer(model)
            sv = explainer.shap_values(X_val)
            if isinstance(sv, list):
                # list of arrays (one per class) → mean abs across classes
                sv_abs = np.mean([np.abs(s) for s in sv], axis=0)
            else:
                sv_abs = np.abs(sv).mean(axis=-1) if sv.ndim == 3 else np.abs(sv)
            shap_per_fold.append(sv_abs.mean(axis=0))

        print(f"  fold {i+1}/{len(folds)} done (n_iter={model.best_iteration_})", flush=True)

    gain_mean = np.mean(gain_per_fold, axis=0)
    results_gain = {f: float(g) for f, g in zip(cols_use, gain_mean)}
    for f in candidates:
        if f not in results_gain:
            results_gain[f] = 0.0

    results_shap = {}
    if has_shap and shap_per_fold:
        shap_mean = np.mean(shap_per_fold, axis=0)
        results_shap = {f: float(v) for f, v in zip(cols_use, shap_mean)}
        for f in candidates:
            if f not in results_shap:
                results_shap[f] = 0.0
        print("\n  Top 25 features by SHAP:")
        for feat, val in sorted(results_shap.items(), key=lambda x: -x[1])[:25]:
            print(f"    {feat:<35} SHAP={val:.5f}")

    print("\n  Top 25 features by Gain:")
    for feat, val in sorted(results_gain.items(), key=lambda x: -x[1])[:25]:
        print(f"    {feat:<35} Gain={val:.1f}")

    return results_gain, results_shap, has_shap


# ─────────────────────────────────────────────────────────────────────────────
def union_select(ic_res, mi_vals, mi_thr, gain_res, shap_res, has_shap, candidates):
    print("\n[Union Selection]")

    # Gain threshold: mean of top 50 (avoid noise floor)
    gain_vals = [v for v in gain_res.values() if v > 0]
    gain_thr  = float(np.mean(gain_vals)) if gain_vals else 1.0

    shap_vals = [v for v in shap_res.values() if v > 0] if shap_res else []
    shap_thr  = float(np.mean(shap_vals)) if shap_vals else 0.0

    print(f"  Thresholds: IC |ic|>=0.02 & |t|>=2 | MI: >={mi_thr:.4f} | "
          f"Gain: >={gain_thr:.1f} | SHAP: >={shap_thr:.5f}")

    selected = []
    all_scores = []
    for feat in candidates:
        ic_d   = ic_res.get(feat, {})
        pass_ic   = ic_d.get("pass_ic", False)
        pass_mi   = mi_vals.get(feat, 0.0) >= mi_thr
        pass_gain = gain_res.get(feat, 0.0) >= gain_thr
        pass_shap = (shap_res.get(feat, 0.0) >= shap_thr) if has_shap else False

        methods_pass = []
        if pass_ic:   methods_pass.append("IC")
        if pass_mi:   methods_pass.append("MI")
        if pass_gain: methods_pass.append("Gain")
        if pass_shap: methods_pass.append("SHAP")

        union = len(methods_pass) > 0
        if union:
            selected.append(feat)

        all_scores.append({
            "feature":     feat,
            "ic":          ic_d.get("ic", 0.0),
            "ic_t":        ic_d.get("t",  0.0),
            "mi":          round(mi_vals.get(feat, 0.0), 6),
            "gain":        round(gain_res.get(feat, 0.0), 2),
            "shap":        round(shap_res.get(feat, 0.0), 6),
            "pass_ic":     bool(pass_ic),
            "pass_mi":     bool(pass_mi),
            "pass_gain":   bool(pass_gain),
            "pass_shap":   bool(pass_shap),
            "n_methods":   len(methods_pass),
            "methods":     methods_pass,
            "selected":    bool(union),
        })

    # Sort by n_methods desc, then shap desc
    all_scores.sort(key=lambda x: (-x["n_methods"], -x["shap"]))

    print(f"\n  SELECTED: {len(selected)}/{len(candidates)} features")
    print(f"  {'Feature':<35} {'IC':>7} {'MI':>8} {'Gain':>9} {'SHAP':>9} {'Methods'}")
    print("  " + "-"*85)
    for s in all_scores:
        if s["selected"]:
            print(f"  {s['feature']:<35} {s['ic']:>+7.4f} {s['mi']:>8.5f} "
                  f"{s['gain']:>9.1f} {s['shap']:>9.5f}  {'+'.join(s['methods'])}")

    return selected, all_scores


# ─────────────────────────────────────────────────────────────────────────────
def train_final(df, selected_features, y_col="tb_cls"):
    print(f"\n[Final Training] {len(selected_features)} features ...")

    valid_mask = df[y_col].notna().values
    df_v = df[valid_mask].copy()
    y    = df_v[y_col].values.astype(int)
    X    = df_v[selected_features].copy()
    for c in selected_features:
        X[c] = X[c].fillna(X[c].median())
    X = X.values.astype(np.float32)

    folds = build_purged_folds(len(df_v), N_FOLDS, TB_PURGE_GAP_BARS)

    from sklearn.metrics import f1_score
    fold_f1 = []
    best_iters = []
    oof_all = np.zeros((len(df_v), 3))

    params = dict(LGBM_PARAMS)
    params["verbose"] = -1

    for i, (tr_idx, val_idx) in enumerate(folds):
        X_tr, y_tr = X[tr_idx], y[tr_idx]
        X_val, y_val = X[val_idx], y[val_idx]
        sw = np.array([LGBM_CLASS_WEIGHTS[yi] for yi in y_tr], dtype=np.float32)
        model = lgb.LGBMClassifier(**params)
        model.fit(
            X_tr, y_tr, sample_weight=sw,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(LGBM_EARLY_STOPPING, verbose=False),
                       lgb.log_evaluation(-1)],
        )
        preds = model.predict(X_val)
        f1 = f1_score(y_val, preds, average="macro")
        fold_f1.append(f1)
        best_iters.append(model.best_iteration_)
        oof_all[val_idx] = model.predict_proba(X_val)
        print(f"  fold {i+1}: F1={f1:.4f}  iter={model.best_iteration_}")

    mean_f1 = float(np.mean(fold_f1))
    std_f1  = float(np.std(fold_f1))
    print(f"\n  CV F1 macro: {mean_f1:.4f} +/- {std_f1:.4f}")
    print(f"  Comparison: v3_no_etf baseline = 0.3472")

    # Full retrain
    n_est = int(np.mean(best_iters))
    params_final = dict(params)
    params_final["n_estimators"] = n_est
    sw_all = np.array([LGBM_CLASS_WEIGHTS[yi] for yi in y], dtype=np.float32)
    final_model = lgb.LGBMClassifier(**params_final)
    final_model.fit(X, y, sample_weight=sw_all)
    print(f"  Full retrain done (n_estimators={n_est})")

    joblib.dump(final_model, OUT_DIR / "lgbm.pkl")
    print(f"  Model saved: {OUT_DIR}/lgbm.pkl")

    return fold_f1, mean_f1, std_f1


# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-train", action="store_true", help="only selection, skip final train")
    parser.add_argument("--coins", nargs="+", default=None, help="subset of coins (default: all)")
    args = parser.parse_args()

    print("="*70)
    print("TB Feature Selection — IC + MI + LGBM Gain + SHAP (union rule)")
    print("="*70)

    # Override TRAINING_COINS if subset requested (for fast testing)
    global TRAINING_COINS
    if args.coins:
        import config as cfg
        cfg.TRAINING_COINS = args.coins
        from config import TRAINING_COINS as TC
        globals()["TRAINING_COINS"] = args.coins

    df = load_data()

    # Filter to valid TB labels
    df = df.dropna(subset=["tb_label", "tb_cls"])
    label_dist = df["tb_cls"].value_counts().to_dict()
    print(f"Label dist: SHORT={label_dist.get(0,0)} FLAT={label_dist.get(1,0)} LONG={label_dist.get(2,0)}")

    # ── Method 1: IC
    ic_res = method_ic(df, ALL_CANDIDATES)

    # ── Method 2: MI
    mi_vals, mi_thr = method_mi(df, ALL_CANDIDATES)

    # ── Method 3+4: LGBM Gain + SHAP
    gain_res, shap_res, has_shap = method_lgbm_shap(df, ALL_CANDIDATES)

    # ── Union select
    selected, all_scores = union_select(ic_res, mi_vals, mi_thr, gain_res, shap_res, has_shap, ALL_CANDIDATES)

    # ── Save results
    with open(OUT_DIR / "fs_results.json", "w") as f:
        json.dump({
            "all_features": all_scores,
            "thresholds": {"ic": 0.02, "ic_t": 2.0, "mi": mi_thr,
                          "gain_mean": float(np.mean([v for v in gain_res.values() if v>0])) if any(v>0 for v in gain_res.values()) else 0,
                          "shap_mean": float(np.mean([v for v in shap_res.values() if v>0])) if shap_res and any(v>0 for v in shap_res.values()) else 0},
            "n_candidates": len(ALL_CANDIDATES),
            "n_selected": len(selected),
        }, f, indent=2)

    with open(OUT_DIR / "fs_selected.json", "w") as f:
        json.dump(selected, f, indent=2)

    if shap_res:
        shap_sorted = sorted(shap_res.items(), key=lambda x: -x[1])
        with open(OUT_DIR / "shap_summary.json", "w") as f:
            json.dump([{"feature": k, "mean_abs_shap": round(v, 6)} for k, v in shap_sorted], f, indent=2)

    print(f"\nResults saved to {OUT_DIR}")
    print(f"Selected {len(selected)} features (union)")

    # ── Final training
    if not args.skip_train:
        fold_f1, mean_f1, std_f1 = train_final(df, selected)
        meta = {
            "n_selected": len(selected),
            "selected_features": selected,
            "cv_f1_mean": round(mean_f1, 4),
            "cv_f1_std":  round(std_f1, 4),
            "baseline_v3_no_etf": 0.3472,
            "delta_vs_baseline": round(mean_f1 - 0.3472, 4),
            "methods": ["IC", "MI", "LGBM_Gain", "SHAP"],
            "union_rule": "pass >= 1 method",
        }
        with open(OUT_DIR / "meta.json", "w") as f:
            json.dump(meta, f, indent=2)
        print(f"\nFinal CV F1: {mean_f1:.4f} vs baseline 0.3472 (delta={mean_f1-0.3472:+.4f})")
    else:
        print("\n--skip-train: no final training")

    print("\nDone.")


if __name__ == "__main__":
    main()
