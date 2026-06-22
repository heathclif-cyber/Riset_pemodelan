"""
pipeline/06o_train_guardian_momentum_v1.py
Guardian Momentum v1 -- profit_v1 + 4 fitur momentum continuation

Tambahan vs profit_v1:
  cvd_slope_h4_delta  : cvd_slope_h4.diff(1)  — apakah selling pressure flatten?
  ofi_h4_accel        : ofi_h4_delta.diff(2)   — apakah OFI masih mempercepat?
  rsi_h4_slope        : rsi_h4.diff(2)          — arah RSI H4, LGBM learn interaksi dg direction
  dist_liq_50x_long   : jarak ke long liquidation wall — untuk SHORT, kalau jauh → tahan

Semua 4 fitur dihitung on-the-fly dari kolom yang sudah ada di parquet.
Labeling identik dengan profit_v1 (profit-only, MFE-based).
"""
import json, sys, warnings
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
    GUARDIAN_N_FOLDS, GUARDIAN_PURGE_GAP_BARS, GUARDIAN_MIN_HOLD_BARS,
    TRAIN_CUTOFF_DATE,
    MAX_HOLDING_BARS, MODAL_PER_TRADE, LEVERAGE_SIM,
    FEE_PER_SIDE, SLIPPAGE_PER_SIDE,
    SWING_LABEL_MIN_RR, SWING_LABEL_MIN_TP, SWING_LABEL_MAX_SL,
    TP_SL_FALLBACK_TP, TP_SL_FALLBACK_SL,
    MODEL_DIR, LABEL_DIR,
)
from core.evaluator import simulate_trades_swing
from core.utils import setup_logger
from pipeline.shared import build_purged_folds

logger = setup_logger("06o_guardian_momentum_v1")

RUN_NAME = "tb_guardian_momentum_v1"
OUT_DIR  = MODEL_DIR / "runs" / RUN_NAME
OUT_DIR.mkdir(parents=True, exist_ok=True)

FB_RUN    = "tb_lgbm_flatboost_v2"
THR_LONG  = 0.50
THR_SHORT = 0.55

# ── Fitur profit_v1 (baseline) ─────────────────────────────────────────────────
STATIC_BASE = [
    "etf_gbtc_change_usd", "etf_total_change_usd", "cvd_slope_h4", "ofi_h4_delta",
    "wyckoff_phase", "Sell_Liq", "atr_percentile_h1", "stochrsi_k",
    "dist_liq_50x_short", "funding_rate", "ema_7_h1", "dow_cos",
    "cvd_div_h4", "dist_swing_low", "VAH", "cvd_momentum_adv",
    "dist_from_8h_high", "ema_200_h1",
]

# ── Fitur tambahan momentum continuation ──────────────────────────────────────
STATIC_NEW = [
    "cvd_slope_h4_delta",   # cvd_slope_h4.diff(1)  — slope masih turun atau flatten?
    "ofi_h4_accel",         # ofi_h4_delta.diff(2)  — OFI masih mempercepat?
    "rsi_h4_slope",         # rsi_h4.diff(2)         — RSI mulai berbalik?
    "dist_liq_50x_long",    # jarak ke long liquidation wall (untuk SHORT: target dump)
]

STATIC_FEATS  = STATIC_BASE + STATIC_NEW
DYNAMIC_FEATS = [
    "bars_held_norm", "current_pnl_pct", "current_pnl_atr",
    "max_favorable_pnl_pct", "drawdown_from_peak_pct",
    "direction", "entry_price_ratio",
]
ALL_FEATS = STATIC_FEATS + DYNAMIC_FEATS
logger.info(
    f"Features: {len(STATIC_BASE)} base + {len(STATIC_NEW)} new + "
    f"{len(DYNAMIC_FEATS)} dynamic = {len(ALL_FEATS)} total"
)


def compute_momentum_feats(df):
    """Hitung 4 fitur momentum baru in-place dari kolom yang sudah ada."""
    df = df.copy()
    if "cvd_slope_h4" in df.columns:
        df["cvd_slope_h4_delta"] = df["cvd_slope_h4"].diff(1)
    else:
        df["cvd_slope_h4_delta"] = 0.0

    if "ofi_h4_delta" in df.columns:
        df["ofi_h4_accel"] = df["ofi_h4_delta"].diff(2)
    else:
        df["ofi_h4_accel"] = 0.0

    if "rsi_h4" in df.columns:
        df["rsi_h4_slope"] = df["rsi_h4"].diff(2)
    else:
        df["rsi_h4_slope"] = 0.0

    if "dist_liq_50x_long" not in df.columns:
        df["dist_liq_50x_long"] = 0.0

    return df


def load_entry_model():
    model = joblib.load(MODEL_DIR / "runs" / FB_RUN / "lgbm.pkl")
    with open(MODEL_DIR / "runs" / FB_RUN / f"{FB_RUN}_features.json") as f:
        feats = json.load(f)
    logger.info(f"Entry model: flatboost_v2 ({len(feats)} feat, thr={THR_LONG}/{THR_SHORT})")
    return model, feats


def generate_labels_for_coin(symbol, fb_model, fb_feats):
    path = LABEL_DIR / f"{symbol}_features_v3.parquet"
    if not path.exists():
        logger.warning(f"  [{symbol}] Not found: {path}")
        return []

    df = pd.read_parquet(path).sort_index()
    df = df[df.index < TRAIN_CUTOFF_DATE]
    df = compute_momentum_feats(df)

    regime_path = LABEL_DIR / f"{symbol}_regime_h1.parquet"
    if regime_path.exists():
        try:
            reg = pd.read_parquet(regime_path)
            if "hmm_regime_enc" in df.columns:
                df = df.drop(columns=["hmm_regime_enc"])
            df = df.join(reg[["hmm_regime_enc"]], how="left")
            df["hmm_regime_enc"] = df["hmm_regime_enc"].fillna(1).astype("int32")
        except Exception:
            df["hmm_regime_enc"] = 1
    elif "hmm_regime_enc" not in df.columns:
        df["hmm_regime_enc"] = 1

    mask = df["label"].astype(str).isin(LABEL_MAP)
    df   = df[mask].copy()
    if len(df) < 100:
        logger.warning(f"  [{symbol}] SKIP — {len(df)} rows")
        return []

    n = len(df)
    X_fb = np.zeros((n, len(fb_feats)), dtype=np.float64)
    for idx, col in enumerate(fb_feats):
        if col in df.columns:
            X_fb[:, idx] = df[col].ffill().fillna(0).values.astype(np.float64)
    proba = fb_model.predict_proba(X_fb)

    y_pred = np.full(n, 1, np.int32)
    y_pred[proba[:, 2] >= THR_LONG]                    = 2
    y_pred[(proba[:, 0] >= THR_SHORT) & (y_pred != 2)] = 0

    close_arr = df["close"].values
    high_arr  = df["high"].values if "high"  in df.columns else close_arr
    low_arr   = df["low"].values  if "low"   in df.columns else close_arr
    atr_arr   = df["atr_14_h1"].values if "atr_14_h1" in df.columns else np.ones(n)
    sh_arr    = df["h4_swing_high"].values if "h4_swing_high" in df.columns else np.full(n, np.nan)
    sl_arr    = df["h4_swing_low"].values  if "h4_swing_low"  in df.columns else np.full(n, np.nan)

    result = simulate_trades_swing(
        y_pred=y_pred, close=close_arr, high=high_arr, low=low_arr, atr=atr_arr,
        h4_swing_highs=sh_arr, h4_swing_lows=sl_arr,
        modal=MODAL_PER_TRADE, leverage=LEVERAGE_SIM[0],
        fee_per_side=FEE_PER_SIDE, slippage=SLIPPAGE_PER_SIDE,
        max_hold=MAX_HOLDING_BARS,
        min_rr=SWING_LABEL_MIN_RR, min_tp_atr=SWING_LABEL_MIN_TP,
        max_sl_atr=SWING_LABEL_MAX_SL,
        tp_fallback_atr=TP_SL_FALLBACK_TP, sl_fallback_atr=TP_SL_FALLBACK_SL,
        guardian_enabled=False,
    )
    trades = result.get("trades", [])
    if not trades:
        return []

    # Build static feature matrix (base + new)
    g_static_cols = [c for c in STATIC_FEATS if c in df.columns]
    X_static = np.zeros((n, len(g_static_cols)), dtype=np.float64)
    for idx, col in enumerate(g_static_cols):
        X_static[:, idx] = df[col].ffill().fillna(0).values.astype(np.float64)

    samples = []
    rule_counts = {2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, "profit_lock": 0}

    for t in trades:
        bar_in      = t["bar_in"]
        bar_out     = t["bar_out"]
        direction   = 2 if t["direction"] == "LONG" else 0
        entry_price = t["entry"]
        atr_entry   = atr_arr[bar_in] if bar_in < n else 0.01

        if bar_out <= bar_in + 1:
            continue

        for j in range(bar_in + 1, min(bar_out, n)):
            cp = close_arr[j]
            if np.isnan(cp):
                continue

            current_pnl = (cp - entry_price) / entry_price if direction == 2 \
                          else (entry_price - cp) / entry_price
            bars_held = j - bar_in

            mfe_sofar = 0.0
            for k in range(bar_in + 1, j + 1):
                if np.isnan(close_arr[k]): continue
                pnl_k = (close_arr[k] - entry_price) / entry_price if direction == 2 \
                        else (entry_price - close_arr[k]) / entry_price
                mfe_sofar = max(mfe_sofar, pnl_k)

            best_future_pnl = 0.0
            for k in range(j + 1, min(bar_out, n)):
                if np.isnan(close_arr[k]): continue
                pnl_k = (close_arr[k] - entry_price) / entry_price if direction == 2 \
                        else (entry_price - close_arr[k]) / entry_price
                best_future_pnl = max(best_future_pnl, pnl_k)

            upside_ratio = (best_future_pnl - current_pnl) / best_future_pnl \
                           if best_future_pnl > 0.001 else 0.0

            # ── Same labeling as profit_v1 ────────────────────────────────
            label = None
            if bars_held < GUARDIAN_MIN_HOLD_BARS:
                label = 0
            elif mfe_sofar > 0.02 and current_pnl > 0 and current_pnl < mfe_sofar * 0.40:
                label = 2; rule_counts["profit_lock"] += 1
            elif mfe_sofar > 0.015 and current_pnl < mfe_sofar * 0.25:
                label = 2; rule_counts[3] += 1
            elif current_pnl >= best_future_pnl * 0.95 and current_pnl > 0.005:
                label = 2; rule_counts[4] += 1
            elif mfe_sofar > 0.015 and current_pnl < mfe_sofar * 0.55:
                label = 1; rule_counts[5] += 1
            elif current_pnl > 0.008 and upside_ratio < 0.03:
                label = 1; rule_counts[6] += 1
            elif best_future_pnl > current_pnl * 1.05:
                label = 0; rule_counts[7] += 1
            else:
                continue

            atr_pct        = atr_entry / entry_price if entry_price > 0 else 0.01
            bars_held_norm = bars_held / MAX_HOLDING_BARS
            current_pnl_atr = current_pnl / atr_pct if atr_pct > 0 else 0.0
            dd_from_peak   = (mfe_sofar - current_pnl) / mfe_sofar \
                             if mfe_sofar > 0.001 else 0.0
            entry_ratio    = entry_price / cp if cp > 0 else 1.0

            sample = {
                **{c: float(X_static[j, idx]) for idx, c in enumerate(g_static_cols)},
                "timestamp":              df.index[j],
                "bars_held_norm":         bars_held_norm,
                "current_pnl_pct":        current_pnl,
                "current_pnl_atr":        current_pnl_atr,
                "max_favorable_pnl_pct":  mfe_sofar,
                "drawdown_from_peak_pct": dd_from_peak,
                "direction":              1.0 if direction == 2 else 0.0,
                "entry_price_ratio":      entry_ratio,
                "label":                  label,
            }
            samples.append(sample)

    n_trades = len(trades)
    n_smp    = len(samples)
    logger.info(
        f"  [{symbol}] {n_trades} trades -> {n_smp} samples | "
        f"lock={rule_counts['profit_lock']} r3={rule_counts[3]} "
        f"r4={rule_counts[4]} r5={rule_counts[5]} hold={rule_counts[7]}"
    )
    return samples


def train_guardian(samples_df):
    g_static = [c for c in samples_df.columns
                if c not in set(DYNAMIC_FEATS) and c not in ("label", "timestamp")]
    feat_cols = g_static + DYNAMIC_FEATS

    X_all = samples_df[feat_cols].values.astype(np.float64)
    y_all = samples_df["label"].values.astype(np.int64)

    n0, n1, n2 = (y_all == 0).sum(), (y_all == 1).sum(), (y_all == 2).sum()
    logger.info(f"Samples: {len(samples_df):,} | HOLD={n0:,} PARTIAL={n1:,} EXIT={n2:,}")

    pnl_col = samples_df["current_pnl_pct"].values
    exit2_mask = (y_all == 2)
    pct_neg = (pnl_col[exit2_mask] < 0).mean() * 100
    logger.info(f"EXIT-2 at negative PnL: {pct_neg:.1f}% (target: <5%)")

    folds    = build_purged_folds(samples_df.index, GUARDIAN_N_FOLDS, GUARDIAN_PURGE_GAP_BARS)
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X_all)

    best_score = float("inf")
    best_model = best_iters = None
    cv_results = []

    for fold_idx, (train_idx, test_idx) in enumerate(folds):
        if len(test_idx) < 10:
            continue
        model = lgb.LGBMClassifier(**GUARDIAN_LGBM_PARAMS)
        model.fit(
            X_scaled[train_idx], y_all[train_idx],
            eval_set=[(X_scaled[test_idx], y_all[test_idx])],
            eval_metric="multi_logloss",
            callbacks=[lgb.early_stopping(GUARDIAN_EARLY_STOPPING), lgb.log_evaluation(0)],
        )
        y_prob = model.predict_proba(X_scaled[test_idx])
        y_pred = np.argmax(y_prob, axis=1)
        f1 = f1_score(y_all[test_idx], y_pred, average="macro")
        ll = -np.mean(np.log(y_prob[np.arange(len(test_idx)), y_all[test_idx]] + 1e-10))
        cv_results.append({"fold": fold_idx + 1, "logloss": round(ll, 4), "f1_macro": round(f1, 4)})
        logger.info(f"  Fold {fold_idx+1}: logloss={ll:.4f} f1={f1:.4f}")
        if ll < best_score:
            best_score = ll; best_model = model; best_iters = model.best_iteration_

    logger.info(f"Best CV logloss: {best_score:.4f} | Retraining final...")
    params_final = {**GUARDIAN_LGBM_PARAMS}
    if best_iters and best_iters > 0:
        params_final["n_estimators"] = best_iters
    final_model = lgb.LGBMClassifier(**params_final)
    final_model.fit(X_scaled, y_all)
    return final_model, scaler, feat_cols, cv_results


def main():
    print(f"\n{'='*65}")
    print(f"  Guardian Momentum v1 Training")
    print(f"  Base : profit_v1 ({len(STATIC_BASE)} static feat)")
    print(f"  +New : {STATIC_NEW}")
    print(f"  Total: {len(ALL_FEATS)} feat")
    print(f"  Out  : {OUT_DIR}")
    print(f"{'='*65}\n")

    fb_model, fb_feats = load_entry_model()

    print("[1/3] Generating labeled samples ...")
    all_samples = []
    label_dist  = {0: 0, 1: 0, 2: 0}
    for coin in TRAINING_COINS:
        samples = generate_labels_for_coin(coin, fb_model, fb_feats)
        for s in samples:
            label_dist[s["label"]] += 1
        all_samples.extend(samples)

    print(f"  Total: {len(all_samples):,} samples | "
          f"HOLD={label_dist[0]:,} PARTIAL={label_dist[1]:,} EXIT={label_dist[2]:,}")

    samples_df = pd.DataFrame(all_samples)
    samples_df["timestamp"] = pd.to_datetime(samples_df["timestamp"])
    samples_df = samples_df.set_index("timestamp").sort_index()

    pnl_vals    = samples_df["current_pnl_pct"].values
    exit2_mask  = samples_df["label"].values == 2
    pnl_at_exit = pnl_vals[exit2_mask]
    print(f"  EXIT-2 PnL: mean={pnl_at_exit.mean()*100:+.2f}% "
          f"pos={((pnl_at_exit>0).sum()/len(pnl_at_exit)*100):.0f}% "
          f"neg={((pnl_at_exit<0).sum()/len(pnl_at_exit)*100):.0f}%")

    print("\n[2/3] Training LGBM ...")
    model, scaler, feat_cols, cv_results = train_guardian(samples_df)
    if model is None:
        print("ERROR: Training failed."); return

    print("\n[3/3] Saving ...")
    joblib.dump(model,  OUT_DIR / "guardian.pkl")
    joblib.dump(scaler, OUT_DIR / "guardian_scaler.pkl")
    with open(OUT_DIR / "guardian_feature_cols.json", "w") as f:
        json.dump(feat_cols, f, indent=2)

    imp      = sorted(zip(feat_cols, model.feature_importances_),
                      key=lambda x: x[1], reverse=True)
    dyn_imp  = sum(v for n, v in imp if n in set(DYNAMIC_FEATS))
    new_imp  = sum(v for n, v in imp if n in set(STATIC_NEW))
    total    = sum(v for _, v in imp)

    meta = {
        "run_name"       : RUN_NAME,
        "trained_at"     : datetime.now().isoformat(),
        "entry_model"    : FB_RUN,
        "n_static_base"  : len(STATIC_BASE),
        "n_static_new"   : len(STATIC_NEW),
        "n_dynamic"      : len(DYNAMIC_FEATS),
        "n_total"        : len(feat_cols),
        "n_samples"      : len(samples_df),
        "label_dist"     : {str(k): int(v) for k, v in label_dist.items()},
        "exit2_pnl_mean" : round(float(pnl_at_exit.mean() * 100), 3),
        "exit2_pct_pos"  : round(float((pnl_at_exit > 0).mean() * 100), 1),
        "new_feats_importance_pct": round(new_imp / total * 100, 1),
        "dynamic_importance_pct"  : round(dyn_imp / total * 100, 1),
        "cv_results"     : cv_results,
        "top12_features" : [{"feature": n, "importance": round(float(v), 5)}
                             for n, v in imp[:12]],
    }
    with open(OUT_DIR / f"{RUN_NAME}_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n  Feature importance (total={total:.0f}):")
    print(f"    Dynamic  : {dyn_imp/total*100:.1f}%")
    print(f"    New feats: {new_imp/total*100:.1f}%")
    print(f"\n  Top 12:")
    for name, val in imp[:12]:
        tag = "[DYN]" if name in set(DYNAMIC_FEATS) \
              else "[NEW]" if name in set(STATIC_NEW) else "     "
        print(f"    {tag} {name:<35} {val/total*100:>5.1f}%")

    if cv_results:
        ll_vals = [r["logloss"] for r in cv_results]
        f1_vals = [r["f1_macro"] for r in cv_results]
        print(f"\n  CV ({len(cv_results)} folds): "
              f"logloss={np.mean(ll_vals):.4f} f1={np.mean(f1_vals):.4f}")

    print(f"\nDone -> {OUT_DIR}")


if __name__ == "__main__":
    main()
