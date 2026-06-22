"""
pipeline/04_train_lgbm_tb_compare.py — Training comparison 4 feature-set options

Options:
  A  v2_strict   (15 feat) — IC v2 KEEP+STABLE, no ETF, no coinank
  B  v3_no_etf   (16 feat) — v3 original minus ETF 2 (conservative baseline)
  C  v2_extended (21 feat) — v2_strict + 6 dropped v3 features (let LGBM decide)
  D  v2_coinank  (17 feat) — v2_strict + 2 coinank (LGBM native NaN 2020-2024)

Output: models/runs/tb_compare/
  - tb_compare_results.json     -- summary perbandingan semua option
  - {option}_cv_results.json    -- detail per-fold tiap option
  - {option}_lgbm.pkl           -- model final tiap option

Jalankan:
  python pipeline/04_train_lgbm_tb_compare.py --all
  python pipeline/04_train_lgbm_tb_compare.py --options A B D
"""

import argparse
import json
import sys
import warnings
from datetime import datetime
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from pipeline.shared import build_purged_folds
from core.features import triple_barrier_labeling
from core.utils import setup_logger
from config import (
    TRAINING_COINS, ALL_COINS,
    LABEL_DIR, MODEL_DIR,
    LGBM_PARAMS, LGBM_EARLY_STOPPING,
    N_FOLDS, PURGE_GAP_BARS,
    LGBM_CLASS_WEIGHTS,
    TRAIN_CUTOFF_DATE,
    MAX_HOLDING_BARS, TP_SL_FALLBACK_TP, TP_SL_FALLBACK_SL,
)

logger = setup_logger("04_train_lgbm_tb_compare")

RUN_NAME    = "tb_compare"
COINANK_DIR = ROOT / "data" / "coinank"
TB_MAP      = {"SHORT": 0, "FLAT": 1, "LONG": 2}

# ─── Feature Set Definitions ────────────────────────────────────────────────────

# IC v2 FINAL (KEEP + STABLE, no ETF, no coinank)
FEATS_V2_STRICT = [
    "cvd_slope_h4",
    "ofi_h4_delta",
    "wyckoff_phase",
    "atr_percentile_h1",
    "stochrsi_d",
    "Sell_Liq",
    "ema_50_slope_h4",
    "Buy_Liq",
    "Fib_786",
    "ema_7_h1",
    "dow_cos",
    "cvd_div_h4",
    "dist_swing_high",
    "VAH",
    "cvd_momentum_adv",
]

# v3 original minus ETF (IC v1 minus 2 ETF yang data-nya kini tidak prediktif)
FEATS_V3_NO_ETF = [
    "cvd_slope_h4", "ofi_h4_delta", "wyckoff_phase", "Sell_Liq",
    "atr_percentile_h1", "stochrsi_k", "dist_liq_50x_short", "funding_rate",
    "ema_7_h1", "dow_cos", "cvd_div_h4", "dist_swing_low", "VAH",
    "cvd_momentum_adv", "dist_from_8h_high", "ema_200_h1",
]

# v2_strict + 6 fitur v3 yang di-drop (biarkan LGBM via feature importance)
FEATS_V2_EXTENDED = sorted(set(FEATS_V2_STRICT) | {
    "stochrsi_k", "dist_liq_50x_short", "funding_rate",
    "dist_swing_low", "dist_from_8h_high", "ema_200_h1",
}, key=lambda f: FEATS_V2_STRICT.index(f) if f in FEATS_V2_STRICT else 999)

# v2_strict + 2 coinank (LGBM handle NaN 2020-2024 natively)
FEATS_V2_COINANK_EXTRA = ["ls_pos_zscore_20d", "smart_retail_div_delta_1d"]
FEATS_V2_COINANK = FEATS_V2_STRICT + FEATS_V2_COINANK_EXTRA

OPTION_DEFS = {
    "A": {
        "name":        "v2_strict",
        "description": "IC v2 FINAL — 15 KEEP+STABLE, no ETF",
        "features":    FEATS_V2_STRICT,
        "coinank":     False,
    },
    "B": {
        "name":        "v3_no_etf",
        "description": "v3 original minus ETF — 16 feat conservative",
        "features":    FEATS_V3_NO_ETF,
        "coinank":     False,
    },
    "C": {
        "name":        "v2_extended",
        "description": "v2_strict + 6 dropped v3 feat — 21 feat, let LGBM decide",
        "features":    FEATS_V2_EXTENDED,
        "coinank":     False,
    },
    "D": {
        "name":        "v2_coinank",
        "description": "v2_strict + 2 coinank — 17 feat, NaN 2020-2024",
        "features":    FEATS_V2_COINANK,
        "coinank":     True,
    },
}


# ─── Data Loading ────────────────────────────────────────────────────────────────

def add_coinank(df: pd.DataFrame, sym: str) -> pd.DataFrame:
    idx = df.index
    lsp_p = COINANK_DIR / f"{sym}_ls_position.parquet"
    lsa_p = COINANK_DIR / f"{sym}_ls_account.parquet"

    ls_pos = None
    if lsp_p.exists():
        lsp = pd.read_parquet(lsp_p).sort_index()
        col = next((c for c in ["top_trader_position_ls", "ls_position"]
                    if c in lsp.columns), None)
        if col:
            ls_pos = lsp[col].reindex(idx, method="ffill")
            lp_m = ls_pos.rolling(24 * 20, min_periods=100).mean()
            lp_s = ls_pos.rolling(24 * 20, min_periods=100).std().clip(lower=1e-8)
            df["ls_pos_zscore_20d"] = (ls_pos - lp_m) / lp_s

    if lsa_p.exists() and ls_pos is not None:
        lsa = pd.read_parquet(lsa_p).sort_index()
        col = next((c for c in ["top_trader_account_ls", "ls_account"]
                    if c in lsa.columns), None)
        if col:
            ls_acc = lsa[col].reindex(idx, method="ffill")
            df["smart_retail_div_delta_1d"] = (ls_pos - ls_acc).diff(24)

    return df


def load_data(coins: list, need_coinank: bool = False) -> pd.DataFrame:
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
        if any(c not in df.columns for c in ["close", "high", "low", "atr_14_h1"]):
            continue

        tb = triple_barrier_labeling(
            df["close"], df["high"], df["low"], df["atr_14_h1"],
            TP_SL_FALLBACK_TP, TP_SL_FALLBACK_SL, MAX_HOLDING_BARS,
        )
        df["tb_label"] = tb.map(TB_MAP)
        df = df.dropna(subset=["tb_label"])
        if len(df) < 100:
            continue

        if need_coinank:
            df = add_coinank(df, sym)

        df["coin"] = sym
        frames.append(df)

    if not frames:
        raise RuntimeError("Tidak ada data!")
    return pd.concat(frames).sort_index()


# ─── Training ───────────────────────────────────────────────────────────────────

def run_cv(X: pd.DataFrame, y: np.ndarray, option_key: str) -> dict:
    is_coinank = OPTION_DEFS[option_key]["coinank"]
    folds = build_purged_folds(X.index, N_FOLDS, PURGE_GAP_BARS)
    all_metrics = []

    for fold, (tr_idx, val_idx) in enumerate(folds, 1):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]

        if is_coinank:
            # Base features: ffill+fillna(0), coinank: ffill only (NaN allowed)
            base_cols   = [c for c in X.columns if c not in FEATS_V2_COINANK_EXTRA]
            coin_cols   = [c for c in X.columns if c in FEATS_V2_COINANK_EXTRA]
            X_tr_base   = X_tr[base_cols].ffill().fillna(0)
            X_val_base  = X_val[base_cols].ffill().fillna(0)
            X_tr_coin   = X_tr[coin_cols].ffill() if coin_cols else pd.DataFrame(index=X_tr.index)
            X_val_coin  = X_val[coin_cols].ffill() if coin_cols else pd.DataFrame(index=X_val.index)
            X_tr_f  = pd.concat([X_tr_base, X_tr_coin], axis=1)[X.columns]
            X_val_f = pd.concat([X_val_base, X_val_coin], axis=1)[X.columns]
        else:
            X_tr_f  = X_tr.ffill().fillna(0)
            X_val_f = X_val.ffill().fillna(0)

        sw = np.array([LGBM_CLASS_WEIGHTS[int(l)] for l in y_tr], dtype=np.float32)
        model = lgb.LGBMClassifier(**LGBM_PARAMS)
        model.fit(
            X_tr_f, y_tr,
            sample_weight=sw,
            eval_set=[(X_val_f, y_val)],
            callbacks=[
                lgb.early_stopping(LGBM_EARLY_STOPPING, verbose=False),
                lgb.log_evaluation(period=-1),
            ],
        )

        y_pred  = model.predict(X_val_f)
        f1_per  = f1_score(y_val, y_pred, average=None, labels=[0, 1, 2], zero_division=0)
        f1_mac  = float(f1_score(y_val, y_pred, average="macro", zero_division=0))
        acc     = float(accuracy_score(y_val, y_pred))
        tr_f1   = float(f1_score(y_tr, model.predict(X_tr_f), average="macro", zero_division=0))

        metrics = {
            "fold": fold,
            "n_train": len(X_tr), "n_val": len(X_val),
            "best_iteration":  model.best_iteration_,
            "train_f1_macro":  round(tr_f1, 4),
            "f1_macro":        round(f1_mac, 4),
            "f1_SHORT":        round(float(f1_per[0]), 4),
            "f1_FLAT":         round(float(f1_per[1]), 4),
            "f1_LONG":         round(float(f1_per[2]), 4),
            "accuracy":        round(acc, 4),
        }
        all_metrics.append({"metrics": metrics, "model": model})

        gap = tr_f1 - f1_mac
        logger.info(
            f"    Fold {fold}: Train={tr_f1:.4f} Val={f1_mac:.4f} Gap={gap:+.4f} | "
            f"S={f1_per[0]:.3f} F={f1_per[1]:.3f} L={f1_per[2]:.3f} | "
            f"iter={model.best_iteration_}"
        )

    return all_metrics


def train_option(option_key: str, df: pd.DataFrame, run_dir: Path) -> dict:
    opt = OPTION_DEFS[option_key]
    logger.info(f"\n{'='*60}")
    logger.info(f"Option {option_key}: {opt['name']} — {opt['description']}")
    logger.info(f"Features ({len(opt['features'])}): {opt['features']}")
    logger.info(f"{'='*60}")

    avail = [c for c in opt["features"] if c in df.columns]
    missing = [c for c in opt["features"] if c not in df.columns]
    if missing:
        logger.warning(f"  Missing columns: {missing}")

    X = df[avail]
    y = df["tb_label"].values.astype(np.int32)

    cv_results = run_cv(X, y, option_key)
    mets   = [r["metrics"] for r in cv_results]
    f1s    = [m["f1_macro"] for m in mets]
    accs   = [m["accuracy"] for m in mets]
    avg_iter = int(np.mean([m["best_iteration"] for m in mets]))

    logger.info(f"  CV Mean F1: {np.mean(f1s):.4f} ± {np.std(f1s):.4f} | "
                f"Avg iter: {avg_iter}")

    # Full retrain
    is_coinank = opt["coinank"]
    if is_coinank:
        base_cols  = [c for c in avail if c not in FEATS_V2_COINANK_EXTRA]
        coin_cols  = [c for c in avail if c in FEATS_V2_COINANK_EXTRA]
        X_full = pd.concat([
            df[base_cols].ffill().fillna(0),
            df[coin_cols].ffill() if coin_cols else pd.DataFrame(index=df.index),
        ], axis=1)[avail]
    else:
        X_full = X.ffill().fillna(0)

    final_params = LGBM_PARAMS.copy()
    final_params["n_estimators"] = avg_iter
    sw = np.array([LGBM_CLASS_WEIGHTS[int(l)] for l in y], dtype=np.float32)
    final_model = lgb.LGBMClassifier(**final_params)
    final_model.fit(X_full, y, sample_weight=sw)

    # Save model
    opt_dir = run_dir / opt["name"]
    opt_dir.mkdir(parents=True, exist_ok=True)
    model_path = opt_dir / "lgbm.pkl"
    joblib.dump(final_model, model_path)

    # Save feature list
    with open(opt_dir / "features.json", "w") as f:
        json.dump(avail, f, indent=2)

    # Save CV results
    cv_out = {
        "option": option_key,
        "name": opt["name"],
        "description": opt["description"],
        "n_features": len(avail),
        "features": avail,
        "mean_f1_macro": round(float(np.mean(f1s)), 4),
        "std_f1_macro":  round(float(np.std(f1s)), 4),
        "mean_accuracy": round(float(np.mean(accs)), 4),
        "avg_best_iteration": avg_iter,
        "folds": mets,
        "created": datetime.now().isoformat(),
    }
    with open(opt_dir / "cv_results.json", "w") as f:
        json.dump(cv_out, f, indent=2)

    return {
        "option":       option_key,
        "name":         opt["name"],
        "description":  opt["description"],
        "n_features":   len(avail),
        "mean_f1":      round(float(np.mean(f1s)), 4),
        "std_f1":       round(float(np.std(f1s)), 4),
        "mean_accuracy": round(float(np.mean(accs)), 4),
        "avg_iter":     avg_iter,
        "model_path":   str(model_path),
        "fold_f1s":     [m["f1_macro"] for m in mets],
    }


# ─── Main ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--coins", nargs="+", default=None)
    parser.add_argument("--options", nargs="+", default=list(OPTION_DEFS.keys()),
                        choices=list(OPTION_DEFS.keys()),
                        help="Option mana yang dijalankan (default: semua)")
    args = parser.parse_args()

    coins    = args.coins or (ALL_COINS if args.all else TRAINING_COINS)
    run_opts = args.options
    need_coinank = "D" in run_opts

    run_dir = MODEL_DIR / "runs" / RUN_NAME
    run_dir.mkdir(parents=True, exist_ok=True)

    sep = "=" * 70
    print(f"\n{sep}")
    print(f"  TB LGBM FEATURE COMPARISON — {RUN_NAME}")
    print(f"  Options: {run_opts}")
    print(f"  TP={TP_SL_FALLBACK_TP}xATR  SL={TP_SL_FALLBACK_SL}xATR  MaxHold={MAX_HOLDING_BARS}")
    for k in run_opts:
        opt = OPTION_DEFS[k]
        print(f"  [{k}] {opt['name']} ({len(opt['features'])} feat): {opt['description']}")
    print(f"{sep}\n")

    print("Loading data + TB labeling...")
    df = load_data(coins, need_coinank=need_coinank)
    logger.info(f"Total: {len(df):,} rows | Coins: {len(df['coin'].unique())}")
    dist = df["tb_label"].value_counts().to_dict()
    n = len(df)
    print(f"  LONG={dist.get(2,0)/n*100:.1f}% SHORT={dist.get(0,0)/n*100:.1f}% FLAT={dist.get(1,0)/n*100:.1f}%")
    if need_coinank:
        for f in FEATS_V2_COINANK_EXTRA:
            if f in df.columns:
                nn = df[f].notna().sum()
                print(f"  {f}: {nn:,} non-NaN ({nn/n*100:.1f}%)")

    # Train each option
    all_results = []
    for opt_key in run_opts:
        result = train_option(opt_key, df, run_dir)
        all_results.append(result)

    # Save comparison summary
    summary = {
        "run_name":  RUN_NAME,
        "coins":     coins,
        "n_coins":   len(coins),
        "tp_atr":    TP_SL_FALLBACK_TP,
        "sl_atr":    TP_SL_FALLBACK_SL,
        "max_hold":  MAX_HOLDING_BARS,
        "n_folds":   N_FOLDS,
        "options":   all_results,
        "created":   datetime.now().isoformat(),
    }
    with open(run_dir / "tb_compare_results.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Print comparison table
    print(f"\n{sep}")
    print(f"  COMPARISON RESULTS — {RUN_NAME}")
    print(f"{sep}")
    print(f"\n  {'Opt':<4} {'Name':<15} {'Feat':>5} {'Mean F1':>8} {'Std F1':>7} {'Accuracy':>9}  Folds")
    print(f"  {'-'*4} {'-'*15} {'-----':>5} {'--------':>8} {'-------':>7} {'---------':>9}  {'------'}")
    for r in sorted(all_results, key=lambda x: x["mean_f1"], reverse=True):
        folds_str = " ".join(f"{f:.3f}" for f in r["fold_f1s"])
        print(f"  [{r['option']}]  {r['name']:<15} {r['n_features']:>5} "
              f"{r['mean_f1']:>8.4f} {r['std_f1']:>7.4f} {r['mean_accuracy']:>9.4f}  {folds_str}")

    best = max(all_results, key=lambda x: x["mean_f1"])
    print(f"\n  Best: Option [{best['option']}] {best['name']} — Mean F1 = {best['mean_f1']:.4f}")
    print(f"  (v3 baseline: 0.4173)")
    print(f"\n  Results: {run_dir}")
    print(f"{sep}\n")


if __name__ == "__main__":
    main()
