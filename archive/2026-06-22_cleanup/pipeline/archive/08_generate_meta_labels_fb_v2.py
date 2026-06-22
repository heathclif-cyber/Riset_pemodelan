"""
pipeline/08_generate_meta_labels_fb_v2.py
OOF meta-labels untuk model aktif: tb_lgbm_flatboost_v2 (Widyawardhana v2).

Walk-forward OOF — label WIN dari simulasi trade kandidat primary (candidate_thr=0.40).
Output: data/meta_labels/fb_v2_oof_trades.parquet

Usage:
  python pipeline/08_generate_meta_labels_fb_v2.py
"""
import argparse
import json
import sys
import warnings
from datetime import datetime
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from config import (
    TRAINING_COINS, LABEL_DIR, MODEL_DIR,
    N_FOLDS, PURGE_GAP_BARS,
    LEVERAGE_SIM, MODAL_PER_TRADE, FEE_PER_SIDE, SLIPPAGE_PER_SIDE,
    TRAIN_CUTOFF_DATE, TP_SL_FALLBACK_TP, TP_SL_FALLBACK_SL, MAX_HOLDING_BARS,
)
from core.features import triple_barrier_labeling
from core.meta_labeling import build_meta_row, directional_candidate
from core.utils import setup_logger
from pipeline.shared import build_purged_folds

logger = setup_logger("08_meta_fb_v2")

RUN_NAME = "tb_meta_fb_v2"
FB_RUN = "tb_lgbm_flatboost_v2"
OUT_DIR = ROOT / "data" / "meta_labels"
OUT_PATH = OUT_DIR / "fb_v2_oof_trades.parquet"

LEVERAGE = LEVERAGE_SIM[0] if isinstance(LEVERAGE_SIM, list) else LEVERAGE_SIM
SL_MULT = TP_SL_FALLBACK_SL
MAX_HOLD = MAX_HOLDING_BARS
COST_RT = (FEE_PER_SIDE + SLIPPAGE_PER_SIDE) * 2
PURGE_GAP = 36  # flatboost_v2 cv

with open(MODEL_DIR / "runs" / FB_RUN / f"{FB_RUN}_features.json", encoding="utf-8") as f:
    FB_FEATS = json.load(f)

META_CONTEXT = [
    "hmm_regime_enc", "atr_percentile_h1", "funding_rate",
    "vol_spike_zscore", "ofi_h4_delta", "cvd_slope_h4",
]

LGBM_PARAMS = {
    "objective": "multiclass",
    "num_class": 3,
    "n_estimators": 600,
    "learning_rate": 0.03,
    "max_depth": 5,
    "num_leaves": 31,
    "min_child_samples": 50,
    "subsample": 0.8,
    "colsample_bytree": 0.7,
    "class_weight": "balanced",
    "verbose": -1,
    "n_jobs": -1,
    "random_state": 42,
}
EARLY_STOP = 50


def load_and_label(coins):
    frames = []
    for sym in coins:
        path = LABEL_DIR / f"{sym}_features_v3.parquet"
        if not path.exists():
            continue
        df = pd.read_parquet(path)
        df.index = pd.to_datetime(df.index, utc=True)
        df = df[df.index < TRAIN_CUTOFF_DATE]
        if df.empty:
            continue
        tb = triple_barrier_labeling(
            df["close"], df["high"], df["low"], df["atr_14_h1"],
            TP_SL_FALLBACK_TP, TP_SL_FALLBACK_SL, MAX_HOLD,
        )
        df["tb_label"] = tb.map({"SHORT": 0, "FLAT": 1, "LONG": 2})
        df = df.dropna(subset=["tb_label"])
        regime_path = LABEL_DIR / f"{sym}_regime_h1.parquet"
        if regime_path.exists():
            try:
                reg = pd.read_parquet(regime_path)
                if "hmm_regime_enc" in df.columns:
                    df = df.drop(columns=["hmm_regime_enc"])
                df = df.join(reg[["hmm_regime_enc"]], how="left")
                df["hmm_regime_enc"] = df["hmm_regime_enc"].fillna(1).astype("int32")
            except Exception as e:
                logger.warning(f"{sym}: regime merge failed ({e})")
        df["_coin"] = sym
        frames.append(df)
    if not frames:
        raise RuntimeError("No training data")
    return pd.concat(frames).sort_index()


def simulate_coin(df_coin, proba, candidate_thr):
    close = df_coin["close"].values.astype(np.float64)
    high = df_coin["high"].values.astype(np.float64)
    low = df_coin["low"].values.astype(np.float64)
    atr = df_coin["atr_14_h1"].values.astype(np.float64)
    n = len(df_coin)
    for k in range(1, n):
        if np.isnan(atr[k]):
            atr[k] = atr[k - 1]

    trades = []
    i = 0
    while i < n:
        sig, _conf = directional_candidate(proba[i], candidate_thr)
        if sig == -1:
            i += 1
            continue
        direction = 1 if sig == 2 else -1
        if np.isnan(atr[i]) or atr[i] <= 0:
            i += 1
            continue

        entry = close[i]
        sl_price = entry - direction * SL_MULT * atr[i]
        exit_i = min(i + MAX_HOLD, n - 1)
        exit_price = close[exit_i]
        for j in range(i + 1, min(i + MAX_HOLD + 1, n)):
            if direction == 1 and low[j] <= sl_price:
                exit_price = sl_price
                exit_i = j
                break
            if direction == -1 and high[j] >= sl_price:
                exit_price = sl_price
                exit_i = j
                break

        ret = (exit_price - entry) / entry * direction
        net_pnl = ret * MODAL_PER_TRADE * LEVERAGE - COST_RT * MODAL_PER_TRADE * LEVERAGE
        row_series = df_coin.iloc[i]
        meta_feats = build_meta_row(proba[i], sig, row_series, META_CONTEXT)
        trades.append({
            "timestamp": df_coin.index[i],
            "direction": direction,
            "win": int(net_pnl > 0),
            "net_pnl": round(float(net_pnl), 6),
            "hold_bars": exit_i - i,
            **meta_feats,
        })
        i = exit_i + 1
    return trades


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-thr", type=float, default=0.40)
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info(f"  OOF Meta Labels — {FB_RUN}")
    logger.info(f"  candidate_thr={args.candidate_thr} | purge={PURGE_GAP}")
    logger.info("=" * 60)

    combined = load_and_label(TRAINING_COINS)
    feat_present = [c for c in FB_FEATS if c in combined.columns]
    X_all = combined[feat_present].ffill().fillna(0.0)
    y_all = combined["tb_label"].values.astype(np.int32)

    folds = build_purged_folds(X_all.index, N_FOLDS, PURGE_GAP)
    all_trades = []
    fold_stats = []

    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        fold_num = fold_idx + 1
        X_tr, y_tr = X_all.iloc[train_idx], y_all[train_idx]
        X_val = X_all.iloc[val_idx]
        val_df = combined.iloc[val_idx]

        model = lgb.LGBMClassifier(**LGBM_PARAMS)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_all[val_idx])],
            callbacks=[
                lgb.early_stopping(EARLY_STOP, verbose=False),
                lgb.log_evaluation(period=-1),
            ],
        )
        proba_val = model.predict_proba(X_val).astype(np.float32)
        val_f1 = f1_score(y_all[val_idx], np.argmax(proba_val, axis=1), average="macro", zero_division=0)
        logger.info(f"Fold {fold_num}: val_f1={val_f1:.4f}")

        fold_trades = []
        for coin in TRAINING_COINS:
            mask = val_df["_coin"] == coin
            if mask.sum() == 0:
                continue
            df_c = val_df[mask]
            proba_c = proba_val[mask.values]
            for t in simulate_coin(df_c, proba_c, args.candidate_thr):
                t["coin"] = coin
                t["fold"] = fold_num
                fold_trades.append(t)

        n_win = sum(t["win"] == 1 for t in fold_trades)
        n_tot = len(fold_trades)
        wr = n_win / n_tot * 100 if n_tot else 0.0
        fold_stats.append({"fold": fold_num, "trades": n_tot, "wr": round(wr, 2)})
        logger.info(f"  Fold {fold_num}: {n_tot} trades | WR {wr:.1f}%")
        all_trades.extend(fold_trades)

    df_trades = pd.DataFrame(all_trades)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df_trades.to_parquet(OUT_PATH, index=False)

    total = len(df_trades)
    meta = {
        "run_name": RUN_NAME,
        "base_model": FB_RUN,
        "generated_at": datetime.now().isoformat(),
        "n_trades": total,
        "overall_wr": round(float(df_trades["win"].mean() * 100), 2) if total else 0,
        "candidate_thr": args.candidate_thr,
        "meta_features": ["p_short", "p_flat", "p_long", "confidence", "direction"] + META_CONTEXT,
        "fold_stats": fold_stats,
        "leak_check": "walk-forward OOF — retrain per fold, label from val predictions only",
    }
    with open(OUT_DIR / "fb_v2_oof_trades_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"\n  OOF trades: {total:,} | WR: {meta['overall_wr']:.1f}%")
    print(f"  Saved: {OUT_PATH}")


if __name__ == "__main__":
    main()