"""
pipeline/04_train_lgbm_tb_v3.py — TB LGBM 3-Class widyawardhana_v3

Perbaikan dari v1/v2:
- 3-class dengan class_weight='balanced' (FLAT bisa dipelajari)
- 18 fitur multistage KEEP+STABLE
- Threshold sweep untuk confidence calibration

Jalankan: python pipeline/04_train_lgbm_tb_v3.py --all
"""
import argparse, json, sys, warnings, numpy as np, pandas as pd
from datetime import datetime
from pathlib import Path
import joblib, lightgbm as lgb
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from scipy import stats

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from pipeline.shared import build_purged_folds
from core.features import triple_barrier_labeling
from core.utils import setup_logger
from config import *
# TB training pakai purge gap yang benar: >= max_hold=36
_PURGE = TB_PURGE_GAP_BARS

logger = setup_logger("04_train_lgbm_tb_v3")

RUN_NAME = "tb_lgbm_widyawardhana_v3"

# 18 multistage KEEP+STABLE features
TB_V3_FEATURES = [
    "etf_gbtc_change_usd", "etf_total_change_usd", "cvd_slope_h4",
    "ofi_h4_delta", "wyckoff_phase", "Sell_Liq", "atr_percentile_h1",
    "stochrsi_k", "dist_liq_50x_short", "funding_rate", "ema_7_h1",
    "dow_cos", "cvd_div_h4", "dist_swing_low", "VAH",
    "cvd_momentum_adv", "dist_from_8h_high", "ema_200_h1",
]

# 3-class LGBM params with balanced weights
TB_V3_PARAMS = {
    "objective": "multiclass", "num_class": 3,
    "n_estimators": 500, "learning_rate": 0.03,
    "max_depth": 5, "num_leaves": 31, "min_child_samples": 50,
    "subsample": 0.8, "colsample_bytree": 0.7,
    "class_weight": "balanced",
    "verbose": -1, "n_jobs": -1, "random_state": 42,
}
TB_EARLY_STOPPING = 50


def load_and_label(coins, tp_atr, sl_atr, max_hold):
    frames = []
    for sym in coins:
        path = LABEL_DIR / f"{sym}_features_v3.parquet"
        if not path.exists():
            continue
        df = pd.read_parquet(path)
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index, utc=True)
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        df = df[df.index < TRAIN_CUTOFF_DATE]
        if df.empty:
            continue
        required = ["close", "high", "low", "atr_14_h1"]
        if any(c not in df.columns for c in required):
            continue
        tb = triple_barrier_labeling(df["close"], df["high"], df["low"],
                                     df["atr_14_h1"], tp_atr, sl_atr, max_hold)
        df["tb_label"] = tb.map({"SHORT": 0, "FLAT": 1, "LONG": 2})
        df = df.dropna(subset=["tb_label"])
        if len(df) < 100:
            continue
        df["coin"] = sym
        frames.append(df)
        n = len(df)
        dist = df["tb_label"].value_counts()
        logger.info(f"  [{sym}] {n:,} bars | LONG={dist.get(2,0)/n*100:.1f}% "
                    f"SHORT={dist.get(0,0)/n*100:.1f}% FLAT={dist.get(1,0)/n*100:.1f}%")
    if not frames:
        raise RuntimeError("No training data!")
    return pd.concat(frames).sort_index()


def main():
    parser = argparse.ArgumentParser(description="TB LGBM 3-Class widyawardhana_v3")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--coins", nargs="+", default=None)
    parser.add_argument("--tp", type=float, default=TP_SL_FALLBACK_TP)
    parser.add_argument("--sl", type=float, default=TP_SL_FALLBACK_SL)
    parser.add_argument("--max-hold", type=int, default=MAX_HOLDING_BARS)
    args = parser.parse_args()

    coins = args.coins or (ALL_COINS if args.all else TRAINING_COINS)
    tp_atr, sl_atr, max_hold = args.tp, args.sl, args.max_hold

    run_dir = MODEL_DIR / "runs" / RUN_NAME
    run_dir.mkdir(parents=True, exist_ok=True)

    sep = "=" * 70
    print(f"\n{sep}")
    print(f"  TB LGBM 3-CLASS — {RUN_NAME}")
    print(f"{sep}")
    print(f"  TP={tp_atr}xATR  SL={sl_atr}xATR  MaxHold={max_hold}")
    print(f"  Features: {len(TB_V3_FEATURES)} (multistage KEEP+STABLE)")
    print(f"  Class weight: balanced")
    print(f"{sep}\n")

    # Stage 1
    print("STAGE 1: Loading + TB Labeling...")
    print("-" * 50)
    df = load_and_label(coins, tp_atr, sl_atr, max_hold)
    y = df["tb_label"].astype(np.int32).values

    # Use pre-selected features
    avail = [c for c in TB_V3_FEATURES if c in df.columns]
    missing = [c for c in TB_V3_FEATURES if c not in df.columns]
    if missing:
        print(f"  WARNING: {len(missing)} features missing: {missing}")
    print(f"  Features available: {len(avail)}/{len(TB_V3_FEATURES)}")
    print(f"  Total: {len(df):,} bars")

    for i, name in enumerate(["SHORT", "FLAT", "LONG"]):
        print(f"  {name}: {(y==i).sum():,} ({(y==i).mean()*100:.1f}%)")

    # Stage 2: CV Training
    print(f"\nSTAGE 2: 3-Class LGBM Training ({len(avail)} features)...")
    print("-" * 50)

    X_train = df[avail].ffill().fillna(0)
    y_train = df["tb_label"].values.astype(np.int32)
    assert len(X_train) == len(y_train)

    folds = build_purged_folds(X_train.index, N_FOLDS, _PURGE)
    all_metrics = []
    best_f1, best_model, best_fold = -1.0, None, -1

    for fold, (tr_idx, val_idx) in enumerate(folds, 1):
        X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train[tr_idx], y_train[val_idx]

        model = lgb.LGBMClassifier(**TB_V3_PARAMS)
        model.fit(X_tr, y_tr,
                  eval_set=[(X_val, y_val)],
                  callbacks=[lgb.early_stopping(TB_EARLY_STOPPING, verbose=False),
                             lgb.log_evaluation(period=-1)])

        y_pred = model.predict(X_val)
        f1_per = f1_score(y_val, y_pred, average=None, labels=[0, 1, 2], zero_division=0)
        f1_macro = float(f1_score(y_val, y_pred, average="macro", zero_division=0))
        acc = float(accuracy_score(y_val, y_pred))

        y_tr_pred = model.predict(X_tr)
        tr_f1_macro = float(f1_score(y_tr, y_tr_pred, average="macro", zero_division=0))

        metrics = {
            "fold": fold, "n_train": len(X_tr), "n_val": len(X_val),
            "best_iteration": model.best_iteration_,
            "train_f1_macro": round(tr_f1_macro, 4),
            "f1_macro": round(f1_macro, 4),
            "f1_SHORT": round(float(f1_per[0]), 4),
            "f1_FLAT": round(float(f1_per[1]), 4),
            "f1_LONG": round(float(f1_per[2]), 4),
            "accuracy": round(acc, 4),
        }
        all_metrics.append(metrics)
        if f1_macro > best_f1:
            best_f1, best_model, best_fold = f1_macro, model, fold

        gap = tr_f1_macro - f1_macro
        logger.info(f"  Fold {fold}: Train F1={tr_f1_macro:.4f} | Val F1={f1_macro:.4f} | "
                    f"Gap={gap:+.4f} | SHORT={f1_per[0]:.4f} FLAT={f1_per[1]:.4f} LONG={f1_per[2]:.4f} | Iter={model.best_iteration_}")

    # Full retrain
    avg_iter = int(np.mean([m["best_iteration"] for m in all_metrics]))
    logger.info(f"CV complete. Avg best_iteration: {avg_iter} | Best Fold: {best_fold} (F1={best_f1:.4f})")

    final_params = TB_V3_PARAMS.copy()
    final_params["n_estimators"] = max(avg_iter, 100)
    final_model = lgb.LGBMClassifier(**final_params)
    final_model.fit(X_train, y_train)
    logger.info(f"Final model trained with n_estimators={final_params['n_estimators']}")

    # Save
    print(f"\nSTAGE 3: Saving...")
    model_path = run_dir / "lgbm.pkl"
    joblib.dump(final_model, model_path)

    with open(run_dir / f"{RUN_NAME}_features.json", "w") as f:
        json.dump(avail, f, indent=2)

    f1s = [m["f1_macro"] for m in all_metrics]
    accs = [m["accuracy"] for m in all_metrics]
    cv_summary = {
        "run_name": RUN_NAME, "tp_atr_mult": tp_atr, "sl_atr_mult": sl_atr,
        "max_hold": max_hold, "n_features": len(avail), "n_folds": N_FOLDS,
        "purge_gap_bars": _PURGE,
        "class_weight": "balanced",
        "mean_f1_macro": round(float(np.mean(f1s)), 4),
        "std_f1_macro": round(float(np.std(f1s)), 4),
        "mean_accuracy": round(float(np.mean(accs)), 4),
        "best_fold": best_fold, "best_f1_macro": round(best_f1, 4),
        "avg_best_iteration": avg_iter, "final_n_estimators": max(avg_iter, 100),
        "folds": all_metrics,
    }
    with open(run_dir / f"{RUN_NAME}_cv_results.json", "w") as f:
        json.dump(cv_summary, f, indent=2, default=str)

    meta = {
        "run_name": RUN_NAME, "created": datetime.now().isoformat(),
        "tp_atr_mult": tp_atr, "sl_atr_mult": sl_atr, "max_hold": max_hold,
        "n_samples": len(df), "n_coins": len(coins), "n_features": len(avail),
        "class_weight": "balanced",
        "cv_mean_f1_macro": round(float(np.mean(f1s)), 4),
    }
    with open(run_dir / f"{RUN_NAME}_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    # Feature importance
    imp = list(zip(avail, final_model.feature_importances_))
    imp.sort(key=lambda x: x[1], reverse=True)
    print(f"\n  Top 10 features:")
    for i, (f, v) in enumerate(imp[:10]):
        print(f"  {i+1:>2}. {f:<35} {v:>8.1f}")

    print(f"\n{sep}")
    print(f"  TB LGBM 3-CLASS COMPLETE — {RUN_NAME}")
    print(f"  CV Macro F1: {np.mean(f1s):.4f} +/- {np.std(f1s):.4f} (random=0.333)")
    print(f"  Model: {model_path}")
    print(f"{sep}\n")
    print(f"  {'Fold':>4}  {'Acc':>7}  {'F1-mac':>7}  {'SHORT':>7}  {'FLAT':>7}  {'LONG':>7}  {'Iter':>6}")
    print("  " + "-" * 52)
    for m in all_metrics:
        print(f"  {m['fold']:>4}  {m['accuracy']:>7.4f}  {m['f1_macro']:>7.4f}  "
              f"{m['f1_SHORT']:>7.4f}  {m['f1_FLAT']:>7.4f}  {m['f1_LONG']:>7.4f}  "
              f"{m['best_iteration']:>6}")


if __name__ == "__main__":
    main()
