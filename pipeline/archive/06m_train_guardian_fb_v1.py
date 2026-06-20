"""
pipeline/06m_train_guardian_fb_v1.py
Guardian Opsi A: pakai 27 fitur flatboost_v2 + 7 dynamic = 34 total

Labeling: sama persis dengan profit_v1 (profit-only, no loss-based EXIT)
Perbedaan: STATIC_FEATS = fb_v2 features (bukan feature_cols_v2.json)

Output: models/runs/tb_guardian_fb_v1/
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

logger = setup_logger("06m_guardian_fb_v1")

RUN_NAME  = "tb_guardian_fb_v1"
OUT_DIR   = MODEL_DIR / "runs" / RUN_NAME
OUT_DIR.mkdir(parents=True, exist_ok=True)

FB_RUN    = "tb_lgbm_flatboost_v2"
THR_LONG  = 0.50
THR_SHORT = 0.55

# ── Feature config: Opsi A = flatboost_v2 features ───────────────────────────
with open(MODEL_DIR / "runs" / FB_RUN / f"{FB_RUN}_features.json") as f:
    STATIC_FEATS = json.load(f)  # 27 fitur flatboost_v2

DYNAMIC_FEATS = [
    "bars_held_norm", "current_pnl_pct", "current_pnl_atr",
    "max_favorable_pnl_pct", "drawdown_from_peak_pct",
    "direction", "entry_price_ratio",
]
ALL_FEATS = STATIC_FEATS + DYNAMIC_FEATS
logger.info(f"Opsi A: {len(STATIC_FEATS)} static (fb_v2) + {len(DYNAMIC_FEATS)} dynamic = {len(ALL_FEATS)} total")


def load_entry_model():
    model = joblib.load(MODEL_DIR / "runs" / FB_RUN / "lgbm.pkl")
    with open(MODEL_DIR / "runs" / FB_RUN / f"{FB_RUN}_features.json") as f:
        feats = json.load(f)
    return model, feats


def generate_labels_for_coin(symbol, fb_model, fb_feats):
    path = LABEL_DIR / f"{symbol}_features_v3.parquet"
    if not path.exists():
        return []

    df = pd.read_parquet(path).sort_index()
    df = df[df.index < TRAIN_CUTOFF_DATE]

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
        return []

    n = len(df)

    X_fb = np.zeros((n, len(fb_feats)), dtype=np.float64)
    for idx, col in enumerate(fb_feats):
        if col in df.columns:
            X_fb[:, idx] = df[col].ffill().fillna(0).values.astype(np.float64)
    proba = fb_model.predict_proba(X_fb)

    y_pred = np.full(n, 1, np.int32)
    y_pred[proba[:, 2] >= THR_LONG]  = 2
    y_pred[(proba[:, 0] >= THR_SHORT) & (y_pred != 2)] = 0

    close_arr = df["close"].values
    high_arr  = df["high"].values  if "high"         in df.columns else close_arr
    low_arr   = df["low"].values   if "low"          in df.columns else close_arr
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

            if direction == 2:
                current_pnl = (cp - entry_price) / entry_price
            else:
                current_pnl = (entry_price - cp) / entry_price

            bars_held = j - bar_in

            mfe_sofar = 0.0
            for k in range(bar_in + 1, j + 1):
                if np.isnan(close_arr[k]): continue
                if direction == 2:
                    mfe_sofar = max(mfe_sofar, (close_arr[k] - entry_price) / entry_price)
                else:
                    mfe_sofar = max(mfe_sofar, (entry_price - close_arr[k]) / entry_price)

            best_future_pnl = 0.0
            for k in range(j + 1, min(bar_out, n)):
                if np.isnan(close_arr[k]): continue
                if direction == 2:
                    best_future_pnl = max(best_future_pnl, (close_arr[k] - entry_price) / entry_price)
                else:
                    best_future_pnl = max(best_future_pnl, (entry_price - close_arr[k]) / entry_price)

            upside_ratio = (best_future_pnl - current_pnl) / best_future_pnl \
                           if best_future_pnl > 0.001 else 0.0

            label = None

            if bars_held < GUARDIAN_MIN_HOLD_BARS:
                label = 0
            elif mfe_sofar > 0.02 and current_pnl > 0 and current_pnl < mfe_sofar * 0.40:
                label = 2
                rule_counts["profit_lock"] += 1
            elif mfe_sofar > 0.015 and current_pnl < mfe_sofar * 0.25:
                label = 2
                rule_counts[3] += 1
            elif current_pnl >= best_future_pnl * 0.95 and current_pnl > 0.005:
                label = 2
                rule_counts[4] += 1
            elif mfe_sofar > 0.015 and current_pnl < mfe_sofar * 0.55:
                label = 1
                rule_counts[5] += 1
            elif current_pnl > 0.008 and upside_ratio < 0.03:
                label = 1
                rule_counts[6] += 1
            elif best_future_pnl > current_pnl * 1.05:
                label = 0
                rule_counts[7] += 1
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

    logger.info(
        f"  [{symbol}] {len(trades)} trades -> {len(samples)} samples | "
        f"profit_lock={rule_counts['profit_lock']} r3={rule_counts[3]} "
        f"r4={rule_counts[4]} r5={rule_counts[5]} hold={rule_counts[7]}"
    )
    return samples


def train_guardian(samples_df):
    g_static = [c for c in samples_df.columns
                if c not in set(DYNAMIC_FEATS) and c not in ("label", "timestamp")]
    feat_cols = g_static + DYNAMIC_FEATS

    X_all = samples_df[feat_cols].values.astype(np.float64)
    y_all = samples_df["label"].values.astype(np.int64)

    n0 = int((y_all == 0).sum())
    n1 = int((y_all == 1).sum())
    n2 = int((y_all == 2).sum())
    logger.info(f"Samples: {len(samples_df):,} | HOLD={n0:,} PARTIAL={n1:,} EXIT={n2:,}")

    pnl_col = samples_df["current_pnl_pct"].values
    exit2_mask = (y_all == 2)
    pct_neg = (pnl_col[exit2_mask] < 0).mean() * 100
    logger.info(f"EXIT-2 at negative PnL: {pct_neg:.1f}% (target: <5%)")

    folds = build_purged_folds(samples_df.index, GUARDIAN_N_FOLDS, GUARDIAN_PURGE_GAP_BARS)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_all)

    best_score = float("inf")
    best_model = None
    best_iters = None
    cv_results = []

    for fold_idx, (train_idx, test_idx) in enumerate(folds):
        if len(test_idx) < 10:
            continue
        X_tr, y_tr = X_scaled[train_idx], y_all[train_idx]
        X_te, y_te = X_scaled[test_idx],  y_all[test_idx]

        model = lgb.LGBMClassifier(**GUARDIAN_LGBM_PARAMS)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_te, y_te)],
            eval_metric="multi_logloss",
            callbacks=[lgb.early_stopping(GUARDIAN_EARLY_STOPPING), lgb.log_evaluation(0)],
        )

        y_prob = model.predict_proba(X_te)
        y_pred = np.argmax(y_prob, axis=1)
        f1  = f1_score(y_te, y_pred, average="macro")
        ll  = -np.mean(np.log(y_prob[np.arange(len(y_te)), y_te] + 1e-10))

        cv_results.append({"fold": fold_idx + 1, "logloss": round(ll, 4), "f1_macro": round(f1, 4)})
        logger.info(f"  Fold {fold_idx+1}: logloss={ll:.4f} f1={f1:.4f}")
        if ll < best_score:
            best_score = ll
            best_model = model
            best_iters = model.best_iteration_

    logger.info(f"Best CV logloss: {best_score:.4f}")
    params_final = {**GUARDIAN_LGBM_PARAMS}
    if best_iters and best_iters > 0:
        params_final["n_estimators"] = best_iters
    final_model = lgb.LGBMClassifier(**params_final)
    final_model.fit(X_scaled, y_all)

    return final_model, scaler, feat_cols, cv_results


def main():
    print(f"\n{'='*65}")
    print(f"  Guardian Opsi A Training  [tb_guardian_fb_v1]")
    print(f"  Static: 27 fitur flatboost_v2 (sama dengan entry LGBM)")
    print(f"  Dynamic: 7 fitur trade-progress")
    print(f"  Total : {len(ALL_FEATS)} fitur")
    print(f"  Labeling: profit-only (sama dengan profit_v1)")
    print(f"{'='*65}\n")

    fb_model, fb_feats = load_entry_model()

    print("[1/3] Generating labeled samples...")
    all_samples  = []
    rule_summary = {0: 0, 1: 0, 2: 0}
    for coin in TRAINING_COINS:
        samples = generate_labels_for_coin(coin, fb_model, fb_feats)
        for s in samples:
            rule_summary[s["label"]] += 1
        all_samples.extend(samples)

    print(f"  Total: {len(all_samples):,} samples | "
          f"HOLD={rule_summary[0]:,} PARTIAL={rule_summary[1]:,} EXIT={rule_summary[2]:,}")
    if not all_samples:
        print("ERROR: No samples.")
        return

    samples_df = pd.DataFrame(all_samples)
    samples_df["timestamp"] = pd.to_datetime(samples_df["timestamp"])
    samples_df = samples_df.set_index("timestamp").sort_index()

    pnl_vals   = samples_df["current_pnl_pct"].values
    exit2_mask = samples_df["label"].values == 2
    pnl_exit   = pnl_vals[exit2_mask]
    print(f"\n  EXIT-2 PnL: mean={pnl_exit.mean()*100:+.2f}% "
          f"pos={((pnl_exit>0).mean()*100):.0f}% neg={((pnl_exit<0).mean()*100):.0f}%")

    print("\n[2/3] Training LGBM multiclass...")
    model, scaler, feat_cols, cv_results = train_guardian(samples_df)

    print("\n[3/3] Saving...")
    joblib.dump(model, OUT_DIR / "guardian.pkl")
    joblib.dump(scaler, OUT_DIR / "guardian_scaler.pkl")
    with open(OUT_DIR / "guardian_feature_cols.json", "w") as f:
        json.dump(feat_cols, f, indent=2)

    imp = sorted(zip(feat_cols, model.feature_importances_), key=lambda x: x[1], reverse=True)
    dyn_share = sum(v for n, v in imp if n in set(DYNAMIC_FEATS)) / sum(v for _, v in imp) * 100

    meta = {
        "run_name":      RUN_NAME,
        "trained_at":    datetime.now().isoformat(),
        "variant":       "Opsi A: 27 flatboost_v2 static + 7 dynamic",
        "n_static":      len(feat_cols) - len(DYNAMIC_FEATS),
        "n_dynamic":     len(DYNAMIC_FEATS),
        "n_total_feats": len(feat_cols),
        "n_samples":     len(samples_df),
        "label_dist":    {str(k): int(v) for k, v in rule_summary.items()},
        "exit2_pnl_mean": float(round(pnl_exit.mean() * 100, 3)),
        "exit2_pct_pos":  float(round((pnl_exit > 0).mean() * 100, 1)),
        "cv_results":    cv_results,
        "dynamic_importance_pct": round(dyn_share, 1),
        "top10_features": [{"feature": n, "importance": round(float(v), 5)} for n, v in imp[:10]],
    }
    with open(OUT_DIR / f"{RUN_NAME}_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"  Saved to {OUT_DIR} ({len(feat_cols)} features)")
    print(f"  Dynamic importance: {dyn_share:.1f}%")
    if cv_results:
        ll_vals = [r["logloss"] for r in cv_results]
        f1_vals = [r["f1_macro"] for r in cv_results]
        print(f"  CV mean: logloss={np.mean(ll_vals):.4f} f1={np.mean(f1_vals):.4f}")
        for r in cv_results:
            print(f"    Fold {r['fold']}: logloss={r['logloss']:.4f} f1={r['f1_macro']:.4f}")
    print(f"\nDone! {RUN_NAME}")


if __name__ == "__main__":
    main()
