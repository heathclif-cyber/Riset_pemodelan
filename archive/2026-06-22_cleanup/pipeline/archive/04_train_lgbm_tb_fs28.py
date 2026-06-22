"""
pipeline/04_train_lgbm_tb_fs28.py — TB LGBM 3-Class widyawardhana_fs28

Perbedaan dari v3 (18 fitur):
- 28 fitur hasil multi-method feature selection:
    IC Test (Gram-Schmidt) + SHAP + Gain + Correlation Pruning
- Prinsip Simon: Decorrelation > Individual Strength
- Feature file: models/feature_cols_tb_widyawardhana_v3_28.json

Target evaluasi:
- Macro F1 vs v3 (baseline: ~0.427 mean)
- F1 FLAT naik? (baseline: ~0.80)
- PnL & WR di holdout Apr-Jun 2026

Jalankan: python pipeline/04_train_lgbm_tb_fs28.py --all
"""
import argparse, json, sys, warnings
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
import joblib, lightgbm as lgb
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from pipeline.shared import build_purged_folds
from core.features import triple_barrier_labeling
from core.utils import setup_logger
from config import *

logger = setup_logger("04_train_lgbm_tb_fs28")

RUN_NAME     = "tb_lgbm_widyawardhana_fs28"
FEAT_JSON    = ROOT / "models" / "feature_cols_tb_widyawardhana_v3_28.json"
_PURGE       = TB_PURGE_GAP_BARS   # >= max_hold=36

# Baseline v3 untuk perbandingan (isi dari run sebelumnya jika ada)
V3_BASELINE = {
    "run_name":       "tb_lgbm_widyawardhana_v3",
    "n_features":     18,
    "mean_f1_macro":  None,   # akan di-load dari JSON jika ada
}

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


# ── Data ─────────────────────────────────────────────────────────────────────

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


# ── Training ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="TB LGBM fs28")
    parser.add_argument("--all",       action="store_true")
    parser.add_argument("--coins",     nargs="+", default=None)
    parser.add_argument("--tp",        type=float, default=TP_SL_FALLBACK_TP)
    parser.add_argument("--sl",        type=float, default=TP_SL_FALLBACK_SL)
    parser.add_argument("--max-hold",  type=int,   default=MAX_HOLDING_BARS)
    parser.add_argument("--run-id",    default=None,
                        help="Override RUN_NAME (default: tb_lgbm_widyawardhana_fs28)")
    parser.add_argument("--feat-json", default=None,
                        help="Override path ke feature JSON (default: feature_cols_tb_widyawardhana_v3_28.json)")
    args = parser.parse_args()

    coins    = args.coins or (ALL_COINS if args.all else TRAINING_COINS)
    tp_atr   = args.tp
    sl_atr   = args.sl
    max_hold = args.max_hold
    purge    = max(max_hold, TB_PURGE_GAP_BARS)  # purge gap >= max_hold agar tidak leakage

    run_name = args.run_id or RUN_NAME
    run_dir  = MODEL_DIR / "runs" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    # Load baseline v3 jika ada
    v3_cv_path = MODEL_DIR / "runs" / "tb_lgbm_widyawardhana_v3" / "tb_lgbm_widyawardhana_v3_cv_results.json"
    if v3_cv_path.exists():
        with open(v3_cv_path) as f:
            v3_data = json.load(f)
        V3_BASELINE["mean_f1_macro"]  = v3_data.get("mean_f1_macro")
        V3_BASELINE["mean_f1_flat"]   = float(np.mean([fold.get("f1_FLAT", 0) for fold in v3_data.get("folds", [])]))
        V3_BASELINE["mean_f1_short"]  = float(np.mean([fold.get("f1_SHORT", 0) for fold in v3_data.get("folds", [])]))
        V3_BASELINE["mean_f1_long"]   = float(np.mean([fold.get("f1_LONG", 0) for fold in v3_data.get("folds", [])]))

    # Load feature list
    feat_path = Path(args.feat_json) if args.feat_json else FEAT_JSON
    with open(feat_path, encoding="utf-8") as f:
        FEATURES = json.load(f)

    sep = "=" * 72
    print(f"\n{sep}")
    print(f"  TB LGBM 3-CLASS FS28 — {run_name}")
    print(f"{sep}")
    print(f"  TP={tp_atr}xATR  SL={sl_atr}xATR  MaxHold={max_hold}")
    print(f"  Features: {len(FEATURES)} (multi-method feature selection)")
    print(f"  Feat JSON: {feat_path.name}")
    print(f"  Purge gap: {purge} bars")
    print(f"  V3 Baseline: {V3_BASELINE['mean_f1_macro']} macro F1 ({V3_BASELINE['n_features']} feat)")
    print(f"{sep}\n")

    # Load + Label
    print("STAGE 1: Loading + TB Labeling...")
    print("-" * 55)
    df = load_and_label(coins, tp_atr, sl_atr, max_hold)
    y  = df["tb_label"].astype(np.int32).values

    avail   = [c for c in FEATURES if c in df.columns]
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        print(f"  WARNING: {len(missing)} features missing: {missing}")
    print(f"  Features available: {len(avail)}/{len(FEATURES)}")
    print(f"  Total: {len(df):,} bars")
    for i, name in enumerate(["SHORT", "FLAT", "LONG"]):
        print(f"  {name}: {(y==i).sum():,} ({(y==i).mean()*100:.1f}%)")

    # Purged Walk-Forward CV
    print(f"\nSTAGE 2: Purged Walk-Forward CV ({len(avail)} features, {N_FOLDS} folds)...")
    print("-" * 55)

    X_train   = df[avail].ffill().fillna(0)
    y_train   = df["tb_label"].values.astype(np.int32)
    folds     = build_purged_folds(X_train.index, N_FOLDS, purge)
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

        y_tr_pred  = model.predict(X_tr)
        tr_f1_macro = float(f1_score(y_tr, y_tr_pred, average="macro", zero_division=0))

        # FLAT prediction distribution
        pred_flat_pct = float((y_pred == 1).mean() * 100)
        true_flat_pct = float((y_val == 1).mean() * 100)

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
            "pred_flat_pct": round(pred_flat_pct, 1),
            "true_flat_pct": round(true_flat_pct, 1),
        }
        all_metrics.append(metrics)
        if f1_macro > best_f1:
            best_f1, best_model, best_fold = f1_macro, model, fold

        gap = tr_f1_macro - f1_macro
        logger.info(
            f"  Fold {fold}: Train={tr_f1_macro:.4f} | Val F1={f1_macro:.4f} | Gap={gap:+.4f} | "
            f"SHORT={f1_per[0]:.4f} FLAT={f1_per[1]:.4f} LONG={f1_per[2]:.4f} | "
            f"Pred_FLAT={pred_flat_pct:.1f}% | Iter={model.best_iteration_}"
        )

    # Full retrain
    avg_iter = int(np.mean([m["best_iteration"] for m in all_metrics]))
    logger.info(f"CV complete. Avg iter={avg_iter} | Best Fold={best_fold} (F1={best_f1:.4f})")

    final_params = LGBM_PARAMS.copy()
    final_params["n_estimators"] = max(avg_iter, 100)
    final_model = lgb.LGBMClassifier(**final_params)
    final_model.fit(X_train, y_train)
    logger.info(f"Final model: n_estimators={final_params['n_estimators']}")

    # Save
    print(f"\nSTAGE 3: Saving to {run_dir}...")
    joblib.dump(final_model, run_dir / "lgbm.pkl")

    with open(run_dir / f"{run_name}_features.json", "w") as f:
        json.dump(avail, f, indent=2)

    f1s         = [m["f1_macro"]  for m in all_metrics]
    f1s_short   = [m["f1_SHORT"]  for m in all_metrics]
    f1s_flat    = [m["f1_FLAT"]   for m in all_metrics]
    f1s_long    = [m["f1_LONG"]   for m in all_metrics]
    accs        = [m["accuracy"]  for m in all_metrics]
    flat_preds  = [m["pred_flat_pct"] for m in all_metrics]

    cv_summary = {
        "run_name":          run_name,
        "tp_atr_mult":       tp_atr, "sl_atr_mult": sl_atr, "max_hold": max_hold,
        "n_features":        len(avail), "n_folds": N_FOLDS, "purge_gap_bars": purge,
        "class_weight":      "balanced",
        "mean_f1_macro":     round(float(np.mean(f1s)),       4),
        "std_f1_macro":      round(float(np.std(f1s)),        4),
        "mean_f1_SHORT":     round(float(np.mean(f1s_short)), 4),
        "mean_f1_FLAT":      round(float(np.mean(f1s_flat)),  4),
        "mean_f1_LONG":      round(float(np.mean(f1s_long)),  4),
        "mean_accuracy":     round(float(np.mean(accs)),      4),
        "mean_pred_flat_pct": round(float(np.mean(flat_preds)), 1),
        "best_fold":         best_fold, "best_f1_macro": round(best_f1, 4),
        "avg_best_iteration": avg_iter,
        "final_n_estimators": max(avg_iter, 100),
        "folds":             all_metrics,
    }
    with open(run_dir / f"{run_name}_cv_results.json", "w") as f:
        json.dump(cv_summary, f, indent=2, default=str)

    meta = {
        "run_name":     run_name, "created": datetime.now().isoformat(),
        "tp_atr_mult":  tp_atr, "sl_atr_mult": sl_atr, "max_hold": max_hold,
        "n_samples":    len(df), "n_coins": len(coins), "n_features": len(avail),
        "class_weight": "balanced",
        "feature_source": str(feat_path),
        "cv_mean_f1_macro": round(float(np.mean(f1s)), 4),
        "cv_mean_f1_FLAT":  round(float(np.mean(f1s_flat)), 4),
    }
    with open(run_dir / f"{run_name}_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    # Feature importance
    imp = sorted(zip(avail, final_model.feature_importances_), key=lambda x: -x[1])
    print(f"\n  Top 15 features (Gain):")
    for i, (feat, val) in enumerate(imp[:15]):
        print(f"  {i+1:>2}. {feat:<35} {val:>8.1f}")

    # ── Scorecard + Comparison ────────────────────────────────────────────────
    print(f"\n{sep}")
    print(f"  HASIL CV — {run_name}")
    print(f"{sep}")
    print(f"\n  {'Fold':>4}  {'Acc':>7}  {'F1-mac':>7}  {'SHORT':>7}  {'FLAT':>7}  {'LONG':>7}  {'PredFL%':>8}  {'Iter':>5}")
    print("  " + "-" * 64)
    for m in all_metrics:
        print(
            f"  {m['fold']:>4}  {m['accuracy']:>7.4f}  {m['f1_macro']:>7.4f}  "
            f"{m['f1_SHORT']:>7.4f}  {m['f1_FLAT']:>7.4f}  {m['f1_LONG']:>7.4f}  "
            f"{m['pred_flat_pct']:>8.1f}%  {m['best_iteration']:>5}"
        )

    print(f"\n  MEAN: {'':>7}  {np.mean(f1s):>7.4f}  "
          f"{np.mean(f1s_short):>7.4f}  {np.mean(f1s_flat):>7.4f}  "
          f"{np.mean(f1s_long):>7.4f}  {np.mean(flat_preds):>8.1f}%")

    # Perbandingan dengan v3
    print(f"\n{sep}")
    print(f"  PERBANDINGAN vs BASELINE v3 ({V3_BASELINE['n_features']} fitur)")
    print(f"{sep}")
    v3_f1 = V3_BASELINE.get("mean_f1_macro")
    new_f1 = round(float(np.mean(f1s)), 4)
    if v3_f1 is not None:
        delta_f1   = new_f1 - v3_f1
        delta_flat = round(float(np.mean(f1s_flat)) - V3_BASELINE.get("mean_f1_flat", 0), 4)
        print(f"  Macro F1  : {v3_f1:.4f} (v3) -> {new_f1:.4f} (fs28)  delta={delta_f1:+.4f}")
        print(f"  F1 FLAT   : {V3_BASELINE.get('mean_f1_flat', 0):.4f} (v3) -> {np.mean(f1s_flat):.4f} (fs28)  delta={delta_flat:+.4f}")
        print(f"  F1 SHORT  : {V3_BASELINE.get('mean_f1_short', 0):.4f} (v3) -> {np.mean(f1s_short):.4f} (fs28)")
        print(f"  F1 LONG   : {V3_BASELINE.get('mean_f1_long', 0):.4f} (v3) -> {np.mean(f1s_long):.4f} (fs28)")
    else:
        print(f"  V3 baseline tidak ditemukan — jalankan v3 dulu untuk perbandingan.")
        print(f"  fs28 Macro F1: {new_f1:.4f} (random=0.333)")
    print(f"  Mean Pred FLAT%: {np.mean(flat_preds):.1f}% (lebih tinggi = model lebih konservatif)")
    print(f"\n  Model: {run_dir}/lgbm.pkl")
    print(f"{sep}\n")


if __name__ == "__main__":
    main()
