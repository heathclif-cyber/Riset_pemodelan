"""
pipeline/08_generate_meta_labels.py — Generate Meta-Labels untuk Meta-Labeling Model

Alur:
  1. Load training data per koin (2020–2025)
  2. Run LGBM + LSTM predictions per bar → dapatkan prob_long, prob_short, confidence
  3. Filter: bar dengan entry signal (confidence >= threshold)
  4. Untuk setiap entry bar: simulasi trade dengan Guardian → win (1) / loss (0)
  5. Simpan: timestamp, coin, fitur, lgbm_proba, confidence, win → CSV

Output: data/meta_labels/meta_labels_training.csv

Note: LGBM dipakai pada data training-nya sendiri (slight in-sample bias).
Ini acceptable untuk proof-of-concept. Meta-model dievaluasi pada holdout murni.

Usage:
    python pipeline/08_generate_meta_labels.py
    python pipeline/08_generate_meta_labels.py --run-id ic32_lstm_multi_v1
    python pipeline/08_generate_meta_labels.py --coins SOLUSDT ETHUSDT
"""

import argparse
import json
import sys
import os
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from config import (
    TRAINING_COINS, ALL_COINS, LABEL_DIR, MODEL_DIR,
    TRAIN_CUTOFF_DATE, CONFIDENCE_THRESHOLD_ENTRY,
    LABEL_MAP, LABEL_MAP_INV, NUM_CLASSES,
    LSTM_CONFIRMATION_ENABLED,
    GUARDIAN_EXIT_THRESHOLD, GUARDIAN_MIN_HOLD_BARS,
    GUARDIAN_ACTIVATION_ATR, GUARDIAN_PARTIAL_EXIT_RATIO,
    TP_SL_MIN_RR, TP_SL_MIN_TP, TP_SL_MAX_SL,
    TP_SL_FALLBACK_TP, TP_SL_FALLBACK_SL,
    MAX_HOLDING_BARS, MODAL_PER_TRADE, LEVERAGE_SIM,
    FEE_PER_SIDE, SLIPPAGE_PER_SIDE,
)
from core.utils import setup_logger, get_lstm_device
from core.models import load_lstm
from core.evaluator import full_trading_report
from pipeline.backtest_utils import (
    hierarchical_predict, compute_guardian_static_array,
    load_regime_models,
)

logger = setup_logger("08_generate_meta_labels")
DEVICE = get_lstm_device()

# Fitur yang disimpan sebagai context untuk meta-model
META_CONTEXT_FEATURES = [
    "rsi_6", "stochrsi_k", "stochrsi_d",
    "rsi_h4", "rsi_slope_h4",
    "cvd_slope_h4", "ofi_h4_delta", "cvd_momentum_adv",
    "swing_momentum", "dist_from_8h_high", "price_in_range",
    "long_short_ratio", "dist_liq_50x_long", "dist_liq_50x_short",
    "hmm_regime_enc", "h4_trend", "atr_14_h1",
]


def resolve_model_path(run_id: str, fname: str, fallback: str) -> Path:
    if run_id:
        p = MODEL_DIR / "runs" / run_id / fname
        if p.exists():
            return p
    return MODEL_DIR / fallback


def load_models(run_id: str):
    lgbm_path  = resolve_model_path(run_id, "lgbm.pkl",              "lgbm_baseline.pkl")
    lstm_path  = resolve_model_path(run_id, "lstm_v2_style.pt",      "lstm_best.pt")
    lstm_sc_path = resolve_model_path(run_id, "lstm_v2_style_scaler.pkl", "lstm_scaler.pkl")
    feat_path  = resolve_model_path(run_id, "feature_cols_v2.json",  "feature_cols_v2.json")
    guard_path = resolve_model_path(run_id, "guardian.pkl",          "guardian_best.pkl")
    guard_sc_path = MODEL_DIR / "guardian_scaler.pkl"
    guard_feat_path = MODEL_DIR / "guardian_feature_cols.json"

    lgbm_model  = joblib.load(lgbm_path)
    feat_cols   = json.load(open(feat_path))
    guardian    = joblib.load(guard_path)
    guard_scaler = joblib.load(guard_sc_path) if guard_sc_path.exists() else None
    guard_feats  = json.load(open(guard_feat_path))

    lstm_model, lstm_scaler = None, None
    if LSTM_CONFIRMATION_ENABLED and lstm_path.exists() and lstm_sc_path.exists():
        try:
            lstm_model  = load_lstm(str(lstm_path), device=str(DEVICE)).to(DEVICE)
            lstm_scaler = joblib.load(lstm_sc_path)
        except Exception as e:
            logger.warning(f"LSTM load gagal: {e} — lanjut tanpa LSTM")

    logger.info(f"LGBM : {lgbm_path.name} ({len(lgbm_model.feature_name_)} feat)")
    logger.info(f"LSTM : {'loaded' if lstm_model else 'disabled'}")
    logger.info(f"Guardian: {guard_path.name} ({len(guard_feats)} feat)")

    return lgbm_model, lstm_model, lstm_scaler, feat_cols, guardian, guard_scaler, guard_feats


def process_coin(
    coin: str,
    lgbm_model, lstm_model, lstm_scaler, feat_cols,
    guardian, guard_scaler, guard_feats,
) -> list[dict]:
    path = LABEL_DIR / f"{coin}_features_v3.parquet"
    if not path.exists():
        logger.warning(f"[{coin}] Feature file tidak ada, skip")
        return []

    df = pd.read_parquet(path)
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, utc=True)
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    df = df[df.index < TRAIN_CUTOFF_DATE].copy()
    if len(df) < 100:
        return []

    # Merge regime
    reg_path = LABEL_DIR / f"{coin}_regime_h1.parquet"
    if reg_path.exists():
        try:
            reg = pd.read_parquet(reg_path)
            if "hmm_regime_enc" in df.columns:
                df = df.drop(columns=["hmm_regime_enc"])
            df = df.join(reg[["hmm_regime_enc"]], how="left")
            df["hmm_regime_enc"] = df["hmm_regime_enc"].fillna(1).astype("int32")
        except Exception:
            df["hmm_regime_enc"] = 1

    df = df.sort_index()

    # Build X matrix untuk LGBM feat cols
    avail_feat = [c for c in feat_cols if c in df.columns]
    X_lstm = np.zeros((len(df), len(avail_feat)), dtype=np.float64)
    for i, col in enumerate(avail_feat):
        X_lstm[:, i] = df[col].ffill().fillna(0).values

    # LGBM raw probabilities — pakai feature_name_ dari model
    gbm_feats = lgbm_model.feature_name_
    X_lgbm = np.zeros((len(df), len(gbm_feats)), dtype=np.float64)
    for i, col in enumerate(gbm_feats):
        if col in df.columns:
            X_lgbm[:, i] = df[col].ffill().fillna(0).values
    lgbm_proba = lgbm_model.predict_proba(X_lgbm)  # (N, 3): SHORT, FLAT, LONG

    # Cascade prediction — LGBM only untuk meta-label generation
    # (LSTM di-skip untuk menghindari DirectML issues pada training data)
    y_pred, confidence = hierarchical_predict(
        None, lgbm_model, None, None,
        X_lstm, avail_feat, [], df,
    )

    # Guardian static features — exclude dynamic features (ditambah per-bar saat simulasi)
    DYNAMIC_FEATS = {
        "bars_held_norm", "current_pnl_pct", "current_pnl_atr",
        "max_favorable_pnl_pct", "drawdown_from_peak_pct",
        "direction", "entry_price_ratio",
    }
    try:
        g_static_feats = [c for c in guard_feats if c not in DYNAMIC_FEATS]
        X_guardian = compute_guardian_static_array(df, g_static_feats)
    except Exception:
        X_guardian = None

    # Full trading simulation
    close_arr  = df["close"].ffill().fillna(0).values
    high_arr   = df["high"].ffill().fillna(0).values
    low_arr    = df["low"].ffill().fillna(0).values
    atr_arr    = df["atr_14_h1"].ffill().fillna(0).values if "atr_14_h1" in df.columns else np.ones(len(df)) * 0.01
    h4_sh_arr  = df["h4_swing_high"].ffill().fillna(0).values if "h4_swing_high" in df.columns else close_arr * 1.05
    h4_sl_arr  = df["h4_swing_low"].ffill().fillna(0).values  if "h4_swing_low"  in df.columns else close_arr * 0.95

    lev = LEVERAGE_SIM[0] if isinstance(LEVERAGE_SIM, list) else LEVERAGE_SIM
    report = full_trading_report(
        y_pred         = y_pred,
        y_actual       = df["label"].map(LABEL_MAP).fillna(1).astype(np.int64).values if "label" in df.columns else np.ones(len(df), dtype=np.int64),
        atr            = atr_arr,
        close          = close_arr,
        high           = high_arr,
        low            = low_arr,
        h4_swing_highs = h4_sh_arr,
        h4_swing_lows  = h4_sl_arr,
        index          = df.index,
        modal          = MODAL_PER_TRADE,
        leverages      = [lev],
        fee_per_side   = FEE_PER_SIDE,
        slippage       = SLIPPAGE_PER_SIDE,
        confidence     = confidence,
        symbol         = coin,
        guardian_model  = guardian,
        guardian_scaler = guard_scaler,
        X_guardian      = X_guardian,
    )

    # Ekstrak per-trade data
    if not report or "trades" not in report:
        return []

    trades = report.get("trades", [])
    if not trades:
        return []

    # Build lookup: entry_bar_idx → trade outcome
    rows = []
    for trade in trades:
        entry_bar = trade.get("bar_in") or trade.get("entry_bar")
        if entry_bar is None:
            continue
        if entry_bar >= len(df):
            continue

        ts  = df.index[entry_bar]
        pnl = trade.get("net_pnl") or trade.get("pnl", 0)
        win = 1 if float(pnl) > 0 else 0

        row = {
            "timestamp":   ts,
            "coin":        coin,
            "direction":   trade.get("direction", ""),
            "confidence":  float(confidence[entry_bar]),
            "lgbm_prob_short": float(lgbm_proba[entry_bar, 0]),
            "lgbm_prob_flat":  float(lgbm_proba[entry_bar, 1]),
            "lgbm_prob_long":  float(lgbm_proba[entry_bar, 2]),
            "prob_margin": float(abs(lgbm_proba[entry_bar, 2] - lgbm_proba[entry_bar, 0])),
            "win":         win,
            "pnl":         float(pnl),
        }

        # Context features
        for feat in META_CONTEXT_FEATURES:
            row[feat] = float(df[feat].iloc[entry_bar]) if feat in df.columns else 0.0

        rows.append(row)

    logger.info(f"[{coin}] {len(rows)} trades | WR={sum(r['win'] for r in rows)/max(len(rows),1)*100:.1f}%")
    return rows


def main():
    parser = argparse.ArgumentParser(description="Generate meta-labels dari training simulation")
    parser.add_argument("--run-id", default="ic32_lstm_multi_v1")
    parser.add_argument("--coins", nargs="+", default=None)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--output", default="data/meta_labels/meta_labels_training.csv")
    args = parser.parse_args()

    coins = args.coins or (ALL_COINS if args.all else TRAINING_COINS)

    print(f"\n{'='*65}")
    print(f" GENERATE META-LABELS | run_id={args.run_id}")
    print(f" Coins: {len(coins)} | Cutoff: {TRAIN_CUTOFF_DATE.date()}")
    print(f"{'='*65}\n")

    lgbm_model, lstm_model, lstm_scaler, feat_cols, guardian, guard_scaler, guard_feats = \
        load_models(args.run_id)

    all_rows = []
    for coin in coins:
        rows = process_coin(
            coin, lgbm_model, lstm_model, lstm_scaler, feat_cols,
            guardian, guard_scaler, guard_feats,
        )
        all_rows.extend(rows)

    if not all_rows:
        logger.error("Tidak ada rows terkumpul!")
        return

    df_out = pd.DataFrame(all_rows)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    df_out.to_csv(args.output, index=False)

    wr = df_out["win"].mean() * 100
    print(f"\n{'='*65}")
    print(f" META-LABELS SELESAI")
    print(f" Total trades : {len(df_out):,}")
    print(f" Win Rate     : {wr:.1f}%")
    print(f" Output       : {args.output}")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()
