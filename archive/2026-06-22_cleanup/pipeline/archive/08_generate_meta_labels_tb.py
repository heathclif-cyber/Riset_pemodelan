"""
pipeline/08_generate_meta_labels_tb.py — Generate OOF meta-labels.

Walk-forward OOF: setiap fold k, LGBM ditraining di fold != k,
predict di fold k. Trades dari prediksi fold k = genuine OOF (no leakage).
Label: WIN=1 jika net_pnl > 0, LOSS=0.

Output: models/runs/tb_meta_v1/oof_meta_dataset.parquet
"""
import json, sys, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import joblib, lightgbm as lgb
from sklearn.metrics import f1_score

from pipeline.shared import build_purged_folds
from core.features import triple_barrier_labeling
from core.utils import setup_logger
from config import *

logger = setup_logger("08_meta_labels")

RUN_META  = "tb_meta_v1"
RUN_BASE  = "tb_lgbm_widyawardhana_v3"
OUT_DIR   = MODEL_DIR / "runs" / RUN_META
OUT_DIR.mkdir(parents=True, exist_ok=True)

SL_MULT  = TP_SL_FALLBACK_SL
MAX_HOLD = MAX_HOLDING_BARS
MODAL    = MODAL_PER_TRADE
LEVERAGE = LEVERAGE_SIM[0] if isinstance(LEVERAGE_SIM, list) else LEVERAGE_SIM
COST_RT  = (FEE_PER_SIDE + SLIPPAGE_PER_SIDE) * 2
CONF_THR = 0.45  # min threshold — konsisten dengan holdout REGIME_THRESH

TB_V3_FEATURES = [
    "etf_gbtc_change_usd", "etf_total_change_usd", "cvd_slope_h4",
    "ofi_h4_delta", "wyckoff_phase", "Sell_Liq", "atr_percentile_h1",
    "stochrsi_k", "dist_liq_50x_short", "funding_rate", "ema_7_h1",
    "dow_cos", "cvd_div_h4", "dist_swing_low", "VAH",
    "cvd_momentum_adv", "dist_from_8h_high", "ema_200_h1",
]

TB_V3_PARAMS = {
    "objective": "multiclass", "num_class": 3,
    "n_estimators": 500, "learning_rate": 0.03,
    "max_depth": 5, "num_leaves": 31, "min_child_samples": 50,
    "subsample": 0.8, "colsample_bytree": 0.7,
    "class_weight": "balanced",
    "verbose": -1, "n_jobs": -1, "random_state": 42,
}
TB_EARLY_STOPPING = 50

# Context features untuk meta-dataset (regime + volatility + sentiment)
META_CONTEXT = [
    "atr_percentile_h1", "funding_rate", "wyckoff_phase",
    "stochrsi_k", "ofi_h4_delta", "Sell_Liq",
]


def load_and_label(coins):
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
        tb = triple_barrier_labeling(df["close"], df["high"], df["low"],
                                     df["atr_14_h1"], TP_SL_FALLBACK_TP, TP_SL_FALLBACK_SL, MAX_HOLD)
        df["tb_label"] = tb.map({"SHORT": 0, "FLAT": 1, "LONG": 2})
        df = df.dropna(subset=["tb_label"])
        if len(df) < 100:
            continue
        df["coin"] = sym
        frames.append(df)
    if not frames:
        raise RuntimeError("No training data")
    return pd.concat(frames).sort_index()


def simulate_coin(yp, conf, close, high, low, atr, timestamps, p_all, df_c, feat_cols):
    """Simulate live-like trades per-coin, return list of meta-records."""
    n = len(yp)
    records = []
    i = 0
    while i < n:
        sig = int(yp[i])
        if sig == -1 or sig == 1 or conf[i] < CONF_THR:
            i += 1
            continue
        direction = 1 if sig == 2 else -1
        entry     = close[i]
        sl_price  = entry - direction * SL_MULT * atr[i]
        exit_price = close[min(i + MAX_HOLD, n - 1)]
        exit_bar   = min(i + MAX_HOLD, n - 1)
        outcome    = "TIME"
        for j in range(i + 1, min(i + MAX_HOLD + 1, n)):
            if direction == 1 and low[j] <= sl_price:
                exit_price = sl_price; exit_bar = j; outcome = "SL"; break
            elif direction == -1 and high[j] >= sl_price:
                exit_price = sl_price; exit_bar = j; outcome = "SL"; break
        ret     = (exit_price - entry) / entry * direction
        net_pnl = ret * MODAL * LEVERAGE - COST_RT * MODAL * LEVERAGE

        row = {
            "ts": timestamps[i],
            "direction": direction,
            "p_short":    float(p_all[i, 0]),
            "p_flat":     float(p_all[i, 1]),
            "p_long":     float(p_all[i, 2]),
            "confidence": float(conf[i]),
            "net_pnl":    net_pnl,
            "win":        int(net_pnl > 0),
        }
        for feat in feat_cols:
            row[feat] = float(df_c.iloc[i][feat]) if feat in df_c.columns else 0.0
        records.append(row)
        i = exit_bar + 1
    return records


def main():
    print(f"\n{'='*60}")
    print(f"  META-LABEL GENERATION (OOF Walk-Forward) — {RUN_META}")
    print(f"  Base model : {RUN_BASE}")
    print(f"  CV         : {N_FOLDS} folds purged, PURGE={PURGE_GAP_BARS}")
    print(f"  Conf thr   : {CONF_THR}")
    print(f"{'='*60}\n")

    df = load_and_label(TRAINING_COINS)
    avail = [c for c in TB_V3_FEATURES if c in df.columns]
    X_train = df[avail].ffill().fillna(0)
    y_train = df["tb_label"].values.astype(np.int32)

    n_total = len(df)
    folds   = build_purged_folds(X_train.index, N_FOLDS, PURGE_GAP_BARS)

    oof_proba = np.full((n_total, 3), np.nan, dtype=np.float32)
    oof_pred  = np.full(n_total, -1, dtype=np.int32)

    logger.info("Running purged CV to collect OOF predictions...")
    for fold, (tr_idx, val_idx) in enumerate(folds, 1):
        X_tr  = X_train.iloc[tr_idx]
        y_tr  = y_train[tr_idx]
        X_val = X_train.iloc[val_idx]
        y_val = y_train[val_idx]

        model = lgb.LGBMClassifier(**TB_V3_PARAMS)
        model.fit(X_tr, y_tr,
                  eval_set=[(X_val, y_val)],
                  callbacks=[lgb.early_stopping(TB_EARLY_STOPPING, verbose=False),
                             lgb.log_evaluation(period=-1)])

        p_val = model.predict_proba(X_val).astype(np.float32)
        oof_proba[val_idx] = p_val
        oof_pred[val_idx]  = np.argmax(p_val, axis=1)

        val_f1 = f1_score(y_val, oof_pred[val_idx], average="macro", zero_division=0)
        logger.info(f"  Fold {fold}: n_train={len(tr_idx):,} n_val={len(val_idx):,} "
                    f"val_F1={val_f1:.4f}")
        del model

    has_oof = ~np.isnan(oof_proba[:, 0])
    logger.info(f"OOF coverage: {has_oof.sum():,}/{n_total:,} bars "
                f"({has_oof.sum()/n_total*100:.1f}%)")

    # Per-coin simulation
    logger.info("Simulating OOF trades per coin...")
    all_records = []
    coins = df["coin"].unique()
    for sym in coins:
        coin_pos = np.where((df["coin"] == sym).values)[0]
        df_c    = df.iloc[coin_pos]
        p_c     = oof_proba[coin_pos]
        yp_c    = oof_pred[coin_pos]
        conf_c  = np.where(np.isnan(p_c[:, 0]), 0.0,
                           np.max(p_c, axis=1))
        close_c = df_c["close"].values.astype(np.float64)
        high_c  = df_c["high"].values.astype(np.float64)
        low_c   = df_c["low"].values.astype(np.float64)
        atr_c   = df_c["atr_14_h1"].values.astype(np.float64)
        ts_c    = df_c.index.values

        recs = simulate_coin(yp_c, conf_c, close_c, high_c, low_c, atr_c,
                             ts_c, p_c, df_c.reset_index(drop=True), META_CONTEXT)
        for r in recs:
            r["coin"] = sym
        all_records.extend(recs)
        logger.info(f"  [{sym}] {len(recs)} OOF trades")

    meta_df = pd.DataFrame(all_records)
    meta_df["ts"] = pd.to_datetime(meta_df["ts"])
    meta_df = meta_df.set_index("ts").sort_index()

    wins  = int(meta_df["win"].sum())
    total = len(meta_df)
    logger.info(f"OOF Trades total: {total:,} | WIN={wins} ({wins/max(total,1)*100:.1f}%) "
                f"| LOSS={total-wins} ({(total-wins)/max(total,1)*100:.1f}%)")

    out_path = OUT_DIR / "oof_meta_dataset.parquet"
    meta_df.to_parquet(out_path)

    summary = {
        "run_name": RUN_META, "base_model": RUN_BASE,
        "total_oof_trades": total, "win_rate": round(wins / max(total, 1), 4),
        "loss_rate": round((total - wins) / max(total, 1), 4),
        "oof_bar_coverage": int(has_oof.sum()),
        "meta_features": ["p_short", "p_flat", "p_long", "confidence",
                          "direction"] + META_CONTEXT,
        "conf_threshold": CONF_THR,
        "generated_at": datetime.now().isoformat(),
    }
    with open(OUT_DIR / f"{RUN_META}_gen_meta.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n  OOF Meta Dataset: {total:,} trades | WR={wins/max(total,1)*100:.1f}%")
    print(f"  Saved: {out_path}")


if __name__ == "__main__":
    main()
