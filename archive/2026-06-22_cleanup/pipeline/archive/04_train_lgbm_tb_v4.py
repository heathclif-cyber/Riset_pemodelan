"""
pipeline/04_train_lgbm_tb_v4.py — TB LGBM 3-Class widyawardhana_v4

Perbedaan dari v3:
- +4 fitur coinank (KEEP+STABLE dari IC test):
    ls_pos_zscore_20d        IC=+0.075  t=+5.51
    smart_retail_div_delta_1d IC=+0.034  t=+2.60
    oi_pct_1d                IC=+0.029  t=+2.08
    oi_price_div_1d          IC=+0.028  t=+2.03
- Total: 22 fitur (18 v3 + 4 coinank)
- Coinank hanya tersedia Jan 2025+; LGBM handle NaN natively untuk 2020-2024

Jalankan: python pipeline/04_train_lgbm_tb_v4.py --all
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

logger = setup_logger("04_train_lgbm_tb_v4")

RUN_NAME    = "tb_lgbm_widyawardhana_v4"
COINANK_DIR = ROOT / "data" / "coinank"

# 18 fitur dari v3
TB_V3_FEATURES = [
    "etf_gbtc_change_usd", "etf_total_change_usd", "cvd_slope_h4",
    "ofi_h4_delta", "wyckoff_phase", "Sell_Liq", "atr_percentile_h1",
    "stochrsi_k", "dist_liq_50x_short", "funding_rate", "ema_7_h1",
    "dow_cos", "cvd_div_h4", "dist_swing_low", "VAH",
    "cvd_momentum_adv", "dist_from_8h_high", "ema_200_h1",
]

# 4 fitur coinank baru (KEEP+STABLE, IC test Jan-Oct 2025)
COINANK_FEATURES = [
    "ls_pos_zscore_20d",
    "smart_retail_div_delta_1d",
    "oi_pct_1d",
    "oi_price_div_1d",
]

ALL_FEATURES = TB_V3_FEATURES + COINANK_FEATURES

TB_V4_PARAMS = {
    "objective": "multiclass", "num_class": 3,
    "n_estimators": 500, "learning_rate": 0.03,
    "max_depth": 5, "num_leaves": 31, "min_child_samples": 50,
    "subsample": 0.8, "colsample_bytree": 0.7,
    "class_weight": "balanced",
    "verbose": -1, "n_jobs": -1, "random_state": 42,
}
TB_EARLY_STOPPING = 50


def add_coinank_features(df, sym):
    """
    Load coinank data dan compute derived features.
    NaN untuk 2020-2024 (sebelum coinank tersedia) — LGBM handle native.
    """
    oi_p  = COINANK_DIR / f"{sym}_oi.parquet"
    lsp_p = COINANK_DIR / f"{sym}_ls_position.parquet"
    lsa_p = COINANK_DIR / f"{sym}_ls_account.parquet"

    idx = df.index

    # ── OI features ──────────────────────────────────────────────────────────
    if oi_p.exists():
        oi = pd.read_parquet(oi_p).sort_index()
        if "oi_binance" in oi.columns:
            oi_s = oi["oi_binance"].reindex(idx, method="ffill")
            df["oi_pct_1d"]       = oi_s.pct_change(24)
            df["oi_price_div_1d"] = oi_s.pct_change(24) - df["close"].pct_change(24)

    # ── L/S Position (top trader) ─────────────────────────────────────────────
    ls_pos = None
    if lsp_p.exists():
        lsp = pd.read_parquet(lsp_p).sort_index()
        if "top_trader_position_ls" in lsp.columns:
            ls_pos = lsp["top_trader_position_ls"].reindex(idx, method="ffill")
            lp_mean = ls_pos.rolling(24 * 20, min_periods=100).mean()
            lp_std  = ls_pos.rolling(24 * 20, min_periods=100).std().clip(lower=1e-8)
            df["ls_pos_zscore_20d"] = (ls_pos - lp_mean) / lp_std

    # ── L/S Account (retail) + smart_retail_div_delta_1d ─────────────────────
    if lsa_p.exists() and ls_pos is not None:
        lsa = pd.read_parquet(lsa_p).sort_index()
        if "top_trader_account_ls" in lsa.columns:
            ls_acc = lsa["top_trader_account_ls"].reindex(idx, method="ffill")
            smart_div = ls_pos - ls_acc
            df["smart_retail_div_delta_1d"] = smart_div.diff(24)

    return df


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
        if any(c not in df.columns for c in ["close", "high", "low", "atr_14_h1"]):
            continue

        # TB labels
        tb = triple_barrier_labeling(df["close"], df["high"], df["low"],
                                     df["atr_14_h1"], tp_atr, sl_atr, max_hold)
        df["tb_label"] = tb.map({"SHORT": 0, "FLAT": 1, "LONG": 2})
        df = df.dropna(subset=["tb_label"])
        if len(df) < 100:
            continue

        # Tambahkan coinank features
        df = add_coinank_features(df, sym)

        df["coin"] = sym
        frames.append(df)
        n = len(df)
        dist = df["tb_label"].value_counts()
        # Cek coinank coverage
        cn_ok = df["ls_pos_zscore_20d"].notna().sum() if "ls_pos_zscore_20d" in df.columns else 0
        logger.info(f"  [{sym}] {n:,} bars | LONG={dist.get(2,0)/n*100:.1f}% "
                    f"SHORT={dist.get(0,0)/n*100:.1f}% FLAT={dist.get(1,0)/n*100:.1f}% "
                    f"| coinank={cn_ok:,} bars ({cn_ok/n*100:.1f}%)")

    if not frames:
        raise RuntimeError("No training data")
    return pd.concat(frames).sort_index()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--coins", nargs="+", default=None)
    parser.add_argument("--tp", type=float, default=TP_SL_FALLBACK_TP)
    parser.add_argument("--sl", type=float, default=TP_SL_FALLBACK_SL)
    parser.add_argument("--max-hold", type=int, default=MAX_HOLDING_BARS)
    args = parser.parse_args()

    coins   = args.coins or (ALL_COINS if args.all else TRAINING_COINS)
    tp_atr, sl_atr, max_hold = args.tp, args.sl, args.max_hold

    run_dir = MODEL_DIR / "runs" / RUN_NAME
    run_dir.mkdir(parents=True, exist_ok=True)

    sep = "=" * 70
    print(f"\n{sep}")
    print(f"  TB LGBM 3-CLASS — {RUN_NAME}")
    print(f"{sep}")
    print(f"  TP={tp_atr}xATR  SL={sl_atr}xATR  MaxHold={max_hold}")
    print(f"  Features: {len(ALL_FEATURES)} ({len(TB_V3_FEATURES)} v3 + {len(COINANK_FEATURES)} coinank)")
    print(f"  Coinank feats: {COINANK_FEATURES}")
    print(f"  NaN for 2020-2024 handled natively by LGBM")
    print(f"{sep}\n")

    print("STAGE 1: Loading + TB Labeling + Coinank Features...")
    print("-" * 50)
    df = load_and_label(coins, tp_atr, sl_atr, max_hold)
    y  = df["tb_label"].astype(np.int32).values

    avail = [c for c in ALL_FEATURES if c in df.columns]
    missing = [c for c in ALL_FEATURES if c not in df.columns]
    if missing:
        print(f"  WARNING missing: {missing}")
    print(f"  Features available: {len(avail)}/{len(ALL_FEATURES)}")
    print(f"  Total bars: {len(df):,}")
    for i, name in enumerate(["SHORT", "FLAT", "LONG"]):
        print(f"  {name}: {(y==i).sum():,} ({(y==i).mean()*100:.1f}%)")

    # Coinank coverage stat
    for f in COINANK_FEATURES:
        if f in df.columns:
            nn = df[f].notna().sum()
            print(f"  {f}: {nn:,} non-NaN ({nn/len(df)*100:.1f}%)")

    print(f"\nSTAGE 2: 3-Class LGBM Training ({len(avail)} features)...")
    print("-" * 50)

    # Coinank features: hanya ffill (biarkan NaN 2020-2024 untuk LGBM native handling)
    # Base features: ffill + fillna(0)
    base_feats   = [c for c in avail if c in TB_V3_FEATURES]
    coinank_cols = [c for c in avail if c in COINANK_FEATURES]

    X_base   = df[base_feats].ffill().fillna(0)
    X_coin   = df[coinank_cols].ffill() if coinank_cols else pd.DataFrame(index=df.index)
    X_train  = pd.concat([X_base, X_coin], axis=1)[avail]  # preserve feature order
    y_train  = y

    folds = build_purged_folds(X_train.index, N_FOLDS, PURGE_GAP_BARS)
    all_metrics = []
    best_f1, best_model, best_fold = -1.0, None, -1

    for fold, (tr_idx, val_idx) in enumerate(folds, 1):
        X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train[tr_idx], y_train[val_idx]

        model = lgb.LGBMClassifier(**TB_V4_PARAMS)
        model.fit(X_tr, y_tr,
                  eval_set=[(X_val, y_val)],
                  callbacks=[lgb.early_stopping(TB_EARLY_STOPPING, verbose=False),
                             lgb.log_evaluation(period=-1)])

        y_pred     = model.predict(X_val)
        f1_per     = f1_score(y_val, y_pred, average=None, labels=[0,1,2], zero_division=0)
        f1_macro   = float(f1_score(y_val, y_pred, average="macro", zero_division=0))
        acc        = float(accuracy_score(y_val, y_pred))
        tr_f1      = float(f1_score(y_tr, model.predict(X_tr), average="macro", zero_division=0))

        metrics = {
            "fold": fold, "n_train": len(X_tr), "n_val": len(X_val),
            "best_iteration": model.best_iteration_,
            "train_f1_macro": round(tr_f1, 4),
            "f1_macro": round(f1_macro, 4),
            "f1_SHORT": round(float(f1_per[0]), 4),
            "f1_FLAT":  round(float(f1_per[1]), 4),
            "f1_LONG":  round(float(f1_per[2]), 4),
            "accuracy": round(acc, 4),
        }
        all_metrics.append(metrics)
        if f1_macro > best_f1:
            best_f1, best_model, best_fold = f1_macro, model, fold

        gap = tr_f1 - f1_macro
        logger.info(f"  Fold {fold}: Train={tr_f1:.4f} Val={f1_macro:.4f} Gap={gap:+.4f} | "
                    f"SHORT={f1_per[0]:.4f} FLAT={f1_per[1]:.4f} LONG={f1_per[2]:.4f} | "
                    f"Iter={model.best_iteration_}")

    avg_iter = int(np.mean([m["best_iteration"] for m in all_metrics]))
    logger.info(f"CV done. Avg iter={avg_iter} | Best Fold {best_fold} (F1={best_f1:.4f})")

    print(f"\nSTAGE 3: Full retrain + Save...")
    final_params = TB_V4_PARAMS.copy()
    final_params["n_estimators"] = max(avg_iter, 100)
    final_model = lgb.LGBMClassifier(**final_params)
    final_model.fit(X_train, y_train)

    joblib.dump(final_model, run_dir / "lgbm.pkl")
    with open(run_dir / f"{RUN_NAME}_features.json", "w") as f:
        json.dump(avail, f, indent=2)

    f1s  = [m["f1_macro"] for m in all_metrics]
    accs = [m["accuracy"] for m in all_metrics]
    cv_summary = {
        "run_name": RUN_NAME,
        "tp_atr_mult": tp_atr, "sl_atr_mult": sl_atr, "max_hold": max_hold,
        "n_features": len(avail), "features_v3": base_feats, "features_coinank": coinank_cols,
        "n_folds": N_FOLDS, "class_weight": "balanced",
        "mean_f1_macro": round(float(np.mean(f1s)), 4),
        "std_f1_macro":  round(float(np.std(f1s)), 4),
        "mean_accuracy": round(float(np.mean(accs)), 4),
        "best_fold": best_fold, "best_f1_macro": round(best_f1, 4),
        "avg_best_iteration": avg_iter,
        "folds": all_metrics,
        "created": datetime.now().isoformat(),
    }
    with open(run_dir / f"{RUN_NAME}_cv_results.json", "w") as f:
        json.dump(cv_summary, f, indent=2, default=str)

    # Feature importance
    imp = sorted(zip(avail, final_model.feature_importances_),
                 key=lambda x: x[1], reverse=True)

    print(f"\n{sep}")
    print(f"  TB LGBM v4 COMPLETE — {RUN_NAME}")
    print(f"  CV Macro F1: {np.mean(f1s):.4f} +/- {np.std(f1s):.4f}  (v3 baseline: 0.4173)")
    print(f"  Model: {run_dir / 'lgbm.pkl'}")
    print(f"\n  Top 10 features:")
    for i, (feat, score) in enumerate(imp[:10]):
        tag = " [coinank]" if feat in COINANK_FEATURES else ""
        print(f"  {i+1:>2}. {feat:<35} {score:>8.1f}{tag}")
    print(f"\n  {'Fold':>4}  {'Acc':>7}  {'F1-mac':>7}  {'SHORT':>7}  {'FLAT':>7}  {'LONG':>7}  {'Iter':>6}")
    print("  " + "-" * 52)
    for m in all_metrics:
        print(f"  {m['fold']:>4}  {m['accuracy']:>7.4f}  {m['f1_macro']:>7.4f}  "
              f"{m['f1_SHORT']:>7.4f}  {m['f1_FLAT']:>7.4f}  {m['f1_LONG']:>7.4f}  "
              f"{m['best_iteration']:>6}")
    print(f"{sep}\n")


if __name__ == "__main__":
    main()
