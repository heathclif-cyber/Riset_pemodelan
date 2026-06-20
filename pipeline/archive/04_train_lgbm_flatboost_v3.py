"""
pipeline/04_train_lgbm_flatboost_v3.py — TB LGBM 3-Class flatboost_v3

Sama dengan flatboost_v2 (27 fitur) + 2 fitur cross-market hasil IC test:
  - relative_strength_z   : IC=-0.0476, Marginal IC=-0.0135 (rank #9)
  - market_breadth        : IC=-0.0378, Marginal IC=-0.0180 (rank #8)

Keduanya fitur kontarian — sedikit coin naik → boost LONG, banyak coin naik → boost SHORT.
market_breadth dicompute cross-coin sebelum per-coin loop (tidak ada di parquet).

Jalankan: python pipeline/04_train_lgbm_flatboost_v3.py --all
"""
import argparse, json, sys, warnings
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
import joblib, lightgbm as lgb
from sklearn.metrics import accuracy_score, f1_score

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from pipeline.shared import build_purged_folds
from core.features import triple_barrier_labeling
from core.utils import setup_logger
from config import *

logger = setup_logger("04_train_lgbm_flatboost_v3")

RUN_NAME  = "tb_lgbm_flatboost_v3"
FEAT_JSON = ROOT / "models" / "feature_cols_tb_flatboost_v3.json"

# Sama dengan flatboost_v2 — tidak diubah
LGBM_PARAMS = {
    "objective":         "multiclass",
    "num_class":         3,
    "n_estimators":      600,
    "learning_rate":     0.03,
    "max_depth":         5,
    "num_leaves":        31,
    "min_child_samples": 50,
    "subsample":         0.8,
    "colsample_bytree":  0.7,
    "class_weight":      "balanced",
    "verbose":           -1,
    "n_jobs":            -1,
    "random_state":      42,
}
EARLY_STOP = 50


def compute_market_breadth(coins: list, cutoff) -> pd.Series:
    """
    Untuk setiap timestamp, hitung fraksi coin dengan log_ret_1 > 0.
    Dicompute dari semua coin sekaligus sebelum per-coin training loop.
    """
    ret_dict = {}
    for sym in coins:
        path = LABEL_DIR / f"{sym}_features_v3.parquet"
        if not path.exists():
            continue
        df = pd.read_parquet(path, columns=["log_ret_1"])
        df = df[df.index < cutoff]
        if not df.empty:
            ret_dict[sym] = df["log_ret_1"]

    if not ret_dict:
        return pd.Series(dtype=float, name="market_breadth")

    panel = pd.DataFrame(ret_dict)
    mb = (panel > 0).sum(axis=1) / panel.notna().sum(axis=1)
    mb.name = "market_breadth"
    logger.info(f"market_breadth: {len(mb):,} timestamps, range [{mb.min():.3f}, {mb.max():.3f}]")
    return mb


def load_and_label(coins, tp_atr, sl_atr, max_hold, market_breadth: pd.Series):
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

        # Inject market_breadth (cross-coin, precomputed)
        if not market_breadth.empty:
            df["market_breadth"] = market_breadth.reindex(df.index)

        tb = triple_barrier_labeling(
            df["close"], df["high"], df["low"],
            df["atr_14_h1"], tp_atr, sl_atr, max_hold,
        )
        df["tb_label"] = tb.map({"SHORT": 0, "FLAT": 1, "LONG": 2})
        df = df.dropna(subset=["tb_label"])
        if len(df) < 100:
            continue
        df["coin"] = sym
        frames.append(df)
        n = len(df)
        dist = df["tb_label"].value_counts()
        logger.info(
            f"  [{sym}] {n:,} bars | "
            f"LONG={dist.get(2,0)/n*100:.1f}% "
            f"SHORT={dist.get(0,0)/n*100:.1f}% "
            f"FLAT={dist.get(1,0)/n*100:.1f}%"
        )
    if not frames:
        raise RuntimeError("No training data!")
    return pd.concat(frames).sort_index()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--all",      action="store_true")
    parser.add_argument("--coins",    nargs="+", default=None)
    parser.add_argument("--tp",       type=float, default=TP_SL_FALLBACK_TP)
    parser.add_argument("--sl",       type=float, default=TP_SL_FALLBACK_SL)
    parser.add_argument("--max-hold", type=int,   default=MAX_HOLDING_BARS)
    parser.add_argument("--run-id",   default=None)
    args = parser.parse_args()

    coins    = args.coins or (ALL_COINS if args.all else TRAINING_COINS)
    tp_atr   = args.tp
    sl_atr   = args.sl
    max_hold = args.max_hold
    purge    = max(max_hold, TB_PURGE_GAP_BARS)
    run_name = args.run_id or RUN_NAME
    run_dir  = MODEL_DIR / "runs" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    with open(FEAT_JSON, encoding="utf-8") as f:
        FEATURES = json.load(f)

    # Baseline v2 untuk perbandingan
    v2_cv_path = MODEL_DIR / "runs" / "tb_lgbm_flatboost_v2" / "tb_lgbm_flatboost_v2_cv_results.json"
    v2_baseline = {}
    if v2_cv_path.exists():
        with open(v2_cv_path) as f:
            v2_data = json.load(f)
        v2_baseline["mean_f1_macro"] = v2_data.get("mean_f1_macro")
        v2_baseline["mean_f1_flat"]  = float(np.mean([fold.get("f1_FLAT", 0) for fold in v2_data.get("folds", [])]))

    sep = "=" * 72
    print(f"\n{sep}")
    print(f"  TB LGBM 3-CLASS FLATBOOST V3 — {run_name}")
    print(f"{sep}")
    print(f"  TP={tp_atr}xATR  SL={sl_atr}xATR  MaxHold={max_hold}  Purge={purge}")
    print(f"  Features: {len(FEATURES)} (27 flatboost_v2 + relative_strength_z + market_breadth)")
    if v2_baseline:
        print(f"  v2 Baseline: F1_macro={v2_baseline.get('mean_f1_macro')}  F1_FLAT={v2_baseline.get('mean_f1_flat','?'):.4f}")
    print(f"{sep}\n")

    # Precompute market_breadth cross-coin
    print("STAGE 0: Computing market_breadth cross-coin...")
    market_breadth = compute_market_breadth(coins, TRAIN_CUTOFF_DATE)

    print("\nSTAGE 1: Loading + TB Labeling...")
    print("-" * 55)
    df = load_and_label(coins, tp_atr, sl_atr, max_hold, market_breadth)
    y  = df["tb_label"].astype(np.int32).values

    avail   = [c for c in FEATURES if c in df.columns]
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        print(f"  WARNING: {len(missing)} features missing dari parquet: {missing}")
    print(f"  Features available: {len(avail)}/{len(FEATURES)}")
    print(f"  Total: {len(df):,} bars")
    for i, name in enumerate(["SHORT", "FLAT", "LONG"]):
        print(f"  {name}: {(y==i).sum():,} ({(y==i).mean()*100:.1f}%)")

    # Purged Walk-Forward CV
    print(f"\nSTAGE 2: Purged Walk-Forward CV ({len(avail)} features, {N_FOLDS} folds)...")
    print("-" * 55)

    X_train = df[avail].ffill().fillna(0)
    y_train = df["tb_label"].values.astype(np.int32)
    folds   = build_purged_folds(X_train.index, N_FOLDS, purge)
    all_metrics = []
    best_f1, best_model, best_fold = -1.0, None, -1

    for fold, (tr_idx, val_idx) in enumerate(folds, 1):
        X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train[tr_idx], y_train[val_idx]

        model = lgb.LGBMClassifier(**LGBM_PARAMS)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            callbacks=[
                lgb.early_stopping(EARLY_STOP, verbose=False),
                lgb.log_evaluation(period=-1),
            ],
        )

        y_pred   = model.predict(X_val)
        f1_per   = f1_score(y_val, y_pred, average=None, labels=[0, 1, 2], zero_division=0)
        f1_macro = float(f1_score(y_val, y_pred, average="macro", zero_division=0))
        acc      = float(accuracy_score(y_val, y_pred))
        y_tr_pred   = model.predict(X_tr)
        tr_f1_macro = float(f1_score(y_tr, y_tr_pred, average="macro", zero_division=0))

        metrics = {
            "fold": fold,
            "n_train": len(X_tr), "n_val": len(X_val),
            "best_iteration": model.best_iteration_,
            "train_f1_macro": round(tr_f1_macro, 4),
            "f1_macro": round(f1_macro, 4),
            "f1_SHORT": round(float(f1_per[0]), 4),
            "f1_FLAT":  round(float(f1_per[1]), 4),
            "f1_LONG":  round(float(f1_per[2]), 4),
            "accuracy": round(acc, 4),
        }
        all_metrics.append(metrics)
        if f1_macro > best_f1:
            best_f1, best_model, best_fold = f1_macro, model, fold

        gap = tr_f1_macro - f1_macro
        logger.info(
            f"  Fold {fold}: Train={tr_f1_macro:.4f} | Val F1={f1_macro:.4f} | Gap={gap:+.4f} | "
            f"SHORT={f1_per[0]:.4f} FLAT={f1_per[1]:.4f} LONG={f1_per[2]:.4f} | "
            f"Iter={model.best_iteration_}"
        )

    # Summary CV
    mean_f1    = float(np.mean([m["f1_macro"]  for m in all_metrics]))
    mean_flat  = float(np.mean([m["f1_FLAT"]   for m in all_metrics]))
    mean_short = float(np.mean([m["f1_SHORT"]  for m in all_metrics]))
    mean_long  = float(np.mean([m["f1_LONG"]   for m in all_metrics]))
    std_f1     = float(np.std( [m["f1_macro"]  for m in all_metrics]))
    avg_iter   = int(np.mean(  [m["best_iteration"] for m in all_metrics]))

    print(f"\n{'='*55}")
    print(f"  CV RESULT  (flatboost_v3, {len(avail)} feat)")
    print(f"  Mean F1 Macro : {mean_f1:.4f} ± {std_f1:.4f}")
    print(f"  Mean F1 SHORT : {mean_short:.4f}")
    print(f"  Mean F1 FLAT  : {mean_flat:.4f}")
    print(f"  Mean F1 LONG  : {mean_long:.4f}")
    print(f"  Avg iterations: {avg_iter}")
    if v2_baseline:
        delta = mean_f1 - (v2_baseline.get("mean_f1_macro") or 0)
        print(f"  vs v2 baseline: {delta:+.4f}")
    print(f"{'='*55}\n")

    # Full retrain
    final_params = LGBM_PARAMS.copy()
    final_params["n_estimators"] = max(avg_iter, 100)
    final_model = lgb.LGBMClassifier(**final_params)
    final_model.fit(X_train, y_train)
    logger.info(f"Final model trained: n_estimators={final_params['n_estimators']}")

    # Save
    print(f"STAGE 3: Saving to {run_dir}...")
    joblib.dump(final_model, run_dir / "lgbm.pkl")
    with open(run_dir / f"{run_name}_features.json", "w") as f:
        json.dump(avail, f, indent=2)

    cv_results = {
        "run_name":      run_name,
        "created":       datetime.now().isoformat(),
        "n_features":    len(avail),
        "features":      avail,
        "n_samples":     len(df),
        "tp_atr_mult":   tp_atr,
        "sl_atr_mult":   sl_atr,
        "max_hold":      max_hold,
        "mean_f1_macro": round(mean_f1, 4),
        "std_f1_macro":  round(std_f1, 4),
        "mean_f1_SHORT": round(mean_short, 4),
        "mean_f1_FLAT":  round(mean_flat, 4),
        "mean_f1_LONG":  round(mean_long, 4),
        "avg_iterations": avg_iter,
        "v2_baseline_f1": v2_baseline.get("mean_f1_macro"),
        "delta_vs_v2":    round(mean_f1 - (v2_baseline.get("mean_f1_macro") or 0), 4),
        "folds":         all_metrics,
    }
    with open(run_dir / f"{run_name}_cv_results.json", "w") as f:
        json.dump(cv_results, f, indent=2)

    print(f"  lgbm.pkl saved")
    print(f"  {run_name}_features.json saved ({len(avail)} features)")
    print(f"  {run_name}_cv_results.json saved")
    print(f"\nDone. Mean F1 Macro = {mean_f1:.4f} ± {std_f1:.4f}")


if __name__ == "__main__":
    main()
