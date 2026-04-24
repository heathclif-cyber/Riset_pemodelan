"""
pipeline/08_backtest.py — Fase 8: Walk-Forward Backtest

Walk-forward backtest menggunakan ensemble model (LGBM + LSTM + Calibrator).
Berbeda dengan 07_evaluate.py yang hanya pakai LGBM untuk SHAP,
08_backtest.py mensimulasikan trade nyata menggunakan full ensemble pipeline.

Jalankan:
  python pipeline/08_backtest.py                 # training coins
  python pipeline/08_backtest.py --all           # semua 20 koin
  python pipeline/08_backtest.py --coins SOLUSDT ETHUSDT
  python pipeline/08_backtest.py --run-id my_run

Output:
  models/runs/{run_id}/backtest_results.json
  models/inference_config.json   ★ BARU: semua parameter untuk deployment
"""

import argparse
import json
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from config import (
    TRAINING_COINS, ALL_COINS, LABEL_DIR, MODEL_DIR,
    TRAIN_START, TRAIN_END,
    N_FOLDS, PURGE_GAP_BARS,
    LABEL_MAP, LABEL_MAP_INV, NUM_CLASSES,
    LSTM_SEQ_LEN, LSTM_HIDDEN, LSTM_LAYERS, LSTM_DROPOUT,
    MODAL_PER_TRADE, LEVERAGE_SIM, FEE_PER_SIDE,
    TP_ATR_MULT, SL_ATR_MULT, MAX_HOLDING_BARS,
    CONFIDENCE_THRESHOLD_ENTRY, CONFIDENCE_FULL, CONFIDENCE_HALF,
    MIN_HOLD_BARS,
    SWING_LABEL_MIN_RR, SWING_LABEL_MIN_TP, SWING_LABEL_MAX_SL,
)
from core.models import load_lstm, ProbabilityCalibrator
from core.evaluator import full_trading_report
from core.utils import setup_logger, update_model_metrics
from pipeline.p05_utils import SequenceDataset

logger = setup_logger("08_backtest")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NON_FEATURE_COLS = {"label"}


# ─── Data Loading ─────────────────────────────────────────────────────────────

def load_symbol(symbol: str, feat_cols: list[str]) -> tuple[pd.DataFrame, np.ndarray] | None:
    path = LABEL_DIR / f"{symbol}_features_v3.parquet"
    if not path.exists():
        logger.warning(f"[{symbol}] File tidak ditemukan: {path}")
        return None

    df = pd.read_parquet(path)
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, utc=True)
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    df = df.sort_index()

    mask = df["label"].astype(str).isin(LABEL_MAP)
    df   = df[mask].copy()
    y    = df["label"].map(LABEL_MAP).values.astype(np.int64)

    valid_cols = [c for c in feat_cols if c in df.columns]
    df[valid_cols] = df[valid_cols].ffill().fillna(0)

    logger.info(f"[{symbol}] Loaded: {len(df):,} rows")
    return df, y


# ─── Ensemble Inference ───────────────────────────────────────────────────────

def get_ensemble_proba(
    lgbm_model,
    lstm_model,
    lstm_scaler,
    meta_learner,
    calibrator: ProbabilityCalibrator,
    X: np.ndarray,
    feat_cols: list[str],
    df_slice: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    lgbm_proba = lgbm_model.predict_proba(df_slice[feat_cols])

    X_sc  = lstm_scaler.transform(X)
    dummy = np.zeros(len(X_sc), dtype=np.int64)
    ds    = SequenceDataset(X_sc, dummy)
    loader = DataLoader(ds, batch_size=1024, shuffle=False)

    lstm_list = []
    lstm_model.eval()
    with torch.no_grad():
        for xb, _ in loader:
            logits = lstm_model(xb.to(DEVICE))
            proba  = torch.softmax(logits, dim=1).cpu().numpy()
            lstm_list.append(proba)
    lstm_proba = np.vstack(lstm_list)

    if len(lstm_proba) < len(lgbm_proba):
        pad = np.ones((len(lgbm_proba) - len(lstm_proba), NUM_CLASSES)) / NUM_CLASSES
        lstm_proba = np.vstack([pad, lstm_proba])

    meta_input = np.hstack([lgbm_proba, lstm_proba])
    meta_proba = meta_learner.predict_proba(meta_input)
    cal_proba  = calibrator.transform(meta_proba)

    return cal_proba.argmax(axis=1), cal_proba.max(axis=1)


# ─── Walk-Forward Folds ───────────────────────────────────────────────────────

def build_purged_folds(n: int) -> list[tuple[np.ndarray, np.ndarray]]:
    splits = np.array_split(np.arange(n), N_FOLDS + 1)
    folds  = []
    for k in range(1, N_FOLDS + 1):
        train_raw = np.concatenate(splits[:k])
        test_raw  = splits[k]
        train_idx = train_raw[:-PURGE_GAP_BARS] if len(train_raw) > PURGE_GAP_BARS else train_raw
        test_idx  = test_raw[PURGE_GAP_BARS:]   if len(test_raw)  > PURGE_GAP_BARS  else test_raw
        folds.append((train_idx, test_idx))
    return folds


# ─── Backtest Per Symbol ──────────────────────────────────────────────────────

def backtest_symbol(
    symbol: str,
    feat_cols: list[str],
    lgbm_model,
    lstm_model,
    lstm_scaler,
    meta_learner,
    calibrator: ProbabilityCalibrator,
) -> dict | None:
    result = load_symbol(symbol, feat_cols)
    if result is None:
        return None

    df, y = result
    X     = df[feat_cols].values.astype(np.float64)
    folds = build_purged_folds(len(df))

    oof_pred  = np.full(len(y), -1, dtype=np.int64)
    oof_conf  = np.zeros(len(y), dtype=np.float64)
    oof_valid = np.zeros(len(y), dtype=bool)

    for fold_num, (_, te_pos) in enumerate(folds, 1):
        if len(te_pos) == 0:
            continue
        df_te = df.iloc[te_pos]
        X_te  = X[te_pos]
        y_pred, confidence = get_ensemble_proba(
            lgbm_model, lstm_model, lstm_scaler,
            meta_learner, calibrator,
            X_te, feat_cols, df_te,
        )
        oof_pred[te_pos]  = y_pred
        oof_conf[te_pos]  = confidence
        oof_valid[te_pos] = True
        logger.info(f"[{symbol}] Fold {fold_num} done — test={len(te_pos):,} bars")

    valid_idx = np.where(oof_valid)[0]
    if len(valid_idx) == 0:
        logger.warning(f"[{symbol}] Tidak ada valid OOF predictions.")
        return None

    y_pred_filtered = oof_pred[valid_idx].copy()
    conf_filtered   = oof_conf[valid_idx]
    below_threshold = (y_pred_filtered != 1) & (conf_filtered < CONFIDENCE_THRESHOLD_ENTRY)
    y_pred_filtered[below_threshold] = 1
    n_filtered = int(below_threshold.sum())
    logger.info(f"[{symbol}] Confidence filter: {n_filtered} sinyal di-skip (< {CONFIDENCE_THRESHOLD_ENTRY:.0%})")

    df_valid  = df.iloc[valid_idx]
    y_valid   = y[valid_idx]
    atr_arr   = df_valid["atr_14_h1"].values if "atr_14_h1" in df_valid.columns else np.ones(len(df_valid))
    close_arr = df_valid["close"].values       if "close"      in df_valid.columns else np.ones(len(df_valid))
    h4_sh_arr = df_valid["h4_swing_high"].values if "h4_swing_high" in df_valid.columns else None
    h4_sl_arr = df_valid["h4_swing_low"].values  if "h4_swing_low"  in df_valid.columns else None
    high_arr  = df_valid["high"].values if "high" in df_valid.columns else close_arr
    low_arr   = df_valid["low"].values if "low" in df_valid.columns else close_arr

    report = full_trading_report(
        y_pred          = y_pred_filtered,
        y_actual        = y_valid,
        atr             = atr_arr,
        close           = close_arr,
        high            = high_arr,
        low             = low_arr,
        h4_swing_highs  = h4_sh_arr,
        h4_swing_lows   = h4_sl_arr,
        index           = df_valid.index,
        modal           = MODAL_PER_TRADE,
        leverages       = LEVERAGE_SIM,
        fee_per_side    = FEE_PER_SIDE,
        min_rr          = SWING_LABEL_MIN_RR,
        min_tp_atr      = SWING_LABEL_MIN_TP,
        max_sl_atr      = SWING_LABEL_MAX_SL,
        max_hold        = MAX_HOLDING_BARS,
        symbol          = symbol,
    )

    report["n_filtered_by_confidence"] = n_filtered
    report["confidence_threshold"]     = CONFIDENCE_THRESHOLD_ENTRY
    return report


# ─── Generate Inference Config ────────────────────────────────────────────────

def generate_inference_config(
    aggregate: dict,
    results: dict,
    feat_cols: list[str],
) -> dict:
    """
    Generate inference_config.json — semua parameter yang dibutuhkan
    aplikasi untuk plug-and-play deployment model.

    Aplikasi cukup baca file ini untuk tahu:
      - Parameter inference (threshold, hold bars, dll)
      - Parameter labeling (TP/SL mult)
      - Parameter risk (leverage, modal, fee)
      - Koin mana yang validated dan direkomendasikan
      - Hasil backtest per koin
      - Arsitektur model (n_features, hidden size, dll)
    """
    # Tentukan kategori koin berdasarkan winrate dan drawdown
    recommended, acceptable, caution = [], [], []
    for sym, r in results.items():
        wr  = r.get("winrate", 0)
        dd3 = r.get("max_drawdown_lev3x", 1)
        if wr >= 0.63 and dd3 <= 0.20:
            recommended.append(sym)
        elif wr >= 0.60 and dd3 <= 0.30:
            acceptable.append(sym)
        else:
            caution.append(sym)

    return {
        "model_version": "ensemble_v2",
        "created_at":    datetime.now(timezone.utc).isoformat(),
        "training_period": {
            "start": str(TRAIN_START.date()),
            "end":   str(TRAIN_END.date()),
        },

        # ── Parameter Inference ───────────────────────────────────────────────
        # Semua parameter yang dibutuhkan saat generate sinyal live
        "inference": {
            "confidence_threshold_entry": CONFIDENCE_THRESHOLD_ENTRY,
            "confidence_full_size":       CONFIDENCE_FULL,
            "confidence_half_size":       CONFIDENCE_HALF,
            "min_hold_bars":              MIN_HOLD_BARS,
            "max_hold_bars":              MAX_HOLDING_BARS,
            "timeframe":                  "1h",
            "seq_len":                    LSTM_SEQ_LEN,
            "label_map":                  LABEL_MAP,
            "label_map_inv":              {str(v): k for k, v in LABEL_MAP.items()},
        },

        # ── Parameter Labeling ────────────────────────────────────────────────
        # Dipakai untuk hitung TP/SL di aplikasi
        "labeling": {
            "tp_atr_mult":      TP_ATR_MULT,
            "sl_atr_mult":      SL_ATR_MULT,
            "max_holding_bars": MAX_HOLDING_BARS,
        },

        # ── Parameter Risk ────────────────────────────────────────────────────
        "risk": {
            "modal_per_trade":      MODAL_PER_TRADE,
            "leverage_recommended": 3.0,
            "leverage_max":         5.0,
            "fee_per_side":         FEE_PER_SIDE,
        },

        # ── Koin yang Sudah Divalidasi ────────────────────────────────────────
        # recommended : winrate ≥ 63% dan DD lev3x ≤ 20%
        # acceptable  : winrate ≥ 60% dan DD lev3x ≤ 30%
        # caution     : di bawah threshold — pakai dengan hati-hati
        "coins_validated": {
            "recommended": recommended,
            "acceptable":  acceptable,
            "caution":     caution,
        },

        # ── Hasil Backtest ────────────────────────────────────────────────────
        "backtest_summary": {
            "mean_winrate":         aggregate["mean_winrate"],
            "mean_trade_per_month": aggregate["mean_trade_per_month"],
            "mean_drawdown_lev3x":  aggregate["mean_drawdown_lev3x"],
            "max_consecutive_loss": aggregate["max_consecutive_loss"],
        },
        "backtest_per_coin": {
            sym: {
                "winrate":         r.get("winrate", 0),
                "trade_per_month": r.get("trade_per_month", 0),
                "dd_lev3x":        r.get("max_drawdown_lev3x", 0),
                "dd_lev5x":        r.get("max_drawdown_lev5x", 0),
                "pnl_lev3x":       r.get("pnl_lev3x", 0),
                "pnl_lev5x":       r.get("pnl_lev5x", 0),
                "max_consec_loss": r.get("max_consecutive_loss", 0),
                "total_trades":    r.get("total_trades", 0),
            }
            for sym, r in results.items()
        },

        # ── File Model ────────────────────────────────────────────────────────
        # Path relatif dari folder models/
        "model_files": {
            "lgbm":       "lgbm_baseline.pkl",
            "lstm":       "lstm_best.pt",
            "scaler":     "lstm_scaler.pkl",
            "meta":       "ensemble_meta.pkl",
            "calibrator": "calibrator.pkl",
            "features":   "feature_cols_v2.json",
        },

        # ── Arsitektur Model ──────────────────────────────────────────────────
        # Dipakai untuk load LSTM dengan parameter yang benar
        "model_architecture": {
            "n_features":   len(feat_cols),
            "lstm_hidden":  LSTM_HIDDEN,
            "lstm_layers":  LSTM_LAYERS,
            "lstm_dropout": LSTM_DROPOUT,
            "num_classes":  NUM_CLASSES,
        },
    }


# ─── Main ─────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Walk-Forward Backtest v2")
    group  = parser.add_mutually_exclusive_group()
    group.add_argument("--all",   action="store_true", help="Semua koin")
    group.add_argument("--coins", nargs="+", metavar="SYMBOL")
    parser.add_argument("--run-id", default=None)
    return parser.parse_args()


def main():
    args    = parse_args()
    run_id  = args.run_id or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = MODEL_DIR / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    if args.all:
        coins = ALL_COINS
    elif args.coins:
        coins = [c.upper() for c in args.coins]
    else:
        coins = TRAINING_COINS

    # ── Load models ───────────────────────────────────────────────────────────
    for path, name in [
        (MODEL_DIR / "lgbm_baseline.pkl",    "LightGBM"),
        (MODEL_DIR / "lstm_best.pt",         "LSTM"),
        (MODEL_DIR / "lstm_scaler.pkl",      "LSTM Scaler"),
        (MODEL_DIR / "ensemble_meta.pkl",    "Meta-learner"),
        (MODEL_DIR / "calibrator.pkl",       "Calibrator"),
        (MODEL_DIR / "feature_cols_v2.json", "Feature cols"),
    ]:
        if not path.exists():
            raise FileNotFoundError(
                f"{name} tidak ditemukan: {path}\n"
                f"Jalankan pipeline 04-06 terlebih dahulu."
            )

    lgbm_model   = joblib.load(MODEL_DIR / "lgbm_baseline.pkl")
    lstm_model   = load_lstm(MODEL_DIR / "lstm_best.pt", device=str(DEVICE)).to(DEVICE)
    lstm_scaler  = joblib.load(MODEL_DIR / "lstm_scaler.pkl")
    meta_learner = joblib.load(MODEL_DIR / "ensemble_meta.pkl")
    calibrator   = ProbabilityCalibrator.load(MODEL_DIR / "calibrator.pkl")

    with open(MODEL_DIR / "feature_cols_v2.json") as f:
        feat_cols = json.load(f)

    logger.info(f"Models loaded | Device: {DEVICE} | Features: {len(feat_cols)} | Coins: {coins}")

    # ── Backtest per symbol ───────────────────────────────────────────────────
    results         = {}
    success, failed = [], []

    for symbol in coins:
        try:
            report = backtest_symbol(
                symbol, feat_cols,
                lgbm_model, lstm_model, lstm_scaler,
                meta_learner, calibrator,
            )
            if report:
                results[symbol] = report
                success.append(symbol)
            else:
                failed.append(symbol)
        except Exception as e:
            logger.exception(f"[{symbol}] Backtest error: {e}")
            failed.append(symbol)

    if not results:
        logger.error("Tidak ada hasil backtest — semua koin gagal.")
        return

    # ── Aggregate metrics ─────────────────────────────────────────────────────
    all_wr   = [r["winrate"]             for r in results.values()]
    all_tpm  = [r["trade_per_month"]     for r in results.values()]
    all_mcl  = [r["max_consecutive_loss"] for r in results.values()]
    all_pnl3 = [r.get("pnl_lev3x", 0)   for r in results.values()]
    all_pnl5 = [r.get("pnl_lev5x", 0)   for r in results.values()]
    all_dd3  = [r.get("max_drawdown_lev3x", 0) for r in results.values()]

    aggregate = {
        "run_id":               run_id,
        "coins":                coins,
        "success":              success,
        "failed":               failed,
        "n_coins":              len(success),
        "confidence_threshold": CONFIDENCE_THRESHOLD_ENTRY,
        "modal_per_trade":      MODAL_PER_TRADE,
        "leverage_sim":         LEVERAGE_SIM,
        "mean_winrate":         round(float(np.mean(all_wr)),   4),
        "std_winrate":          round(float(np.std(all_wr)),    4),
        "mean_trade_per_month": round(float(np.mean(all_tpm)), 2),
        "mean_pnl_lev3x":       round(float(np.mean(all_pnl3)), 2),
        "mean_pnl_lev5x":       round(float(np.mean(all_pnl5)), 2),
        "mean_drawdown_lev3x":  round(float(np.mean(all_dd3)), 4),
        "max_consecutive_loss": int(max(all_mcl)),
        "per_symbol":           results,
    }

    # ── Simpan backtest results ───────────────────────────────────────────────
    out_path = run_dir / "backtest_results.json"
    with open(out_path, "w") as f:
        json.dump(aggregate, f, indent=2, default=str)
    logger.info(f"Backtest results → {out_path}")

    # ── Update model registry ─────────────────────────────────────────────────
    update_model_metrics(
        "ensemble_v2",
        winrate              = aggregate["mean_winrate"],
        trade_per_month      = aggregate["mean_trade_per_month"],
        pnl_lev3x            = aggregate["mean_pnl_lev3x"],
        pnl_lev5x            = aggregate["mean_pnl_lev5x"],
        max_drawdown         = aggregate["mean_drawdown_lev3x"],
        max_consecutive_loss = aggregate["max_consecutive_loss"],
        status               = "active",
    )
    logger.info("Model registry updated.")

    # ── Generate inference_config.json ★ BARU ─────────────────────────────────
    inference_cfg      = generate_inference_config(aggregate, results, feat_cols)
    inference_cfg_path = MODEL_DIR / "inference_config.json"
    with open(inference_cfg_path, "w") as f:
        json.dump(inference_cfg, f, indent=2, default=str)
    logger.info(f"Inference config → {inference_cfg_path}")

    # ── Print summary ─────────────────────────────────────────────────────────
    sep = "=" * 60
    print(f"\n{sep}")
    print(f"  BACKTEST SELESAI — {run_id}")
    print(f"{sep}")
    print(f"  Koin berhasil  : {len(success)} — {success}")
    print(f"  Koin gagal     : {len(failed)}  — {failed}")
    print(f"{sep}")
    print(f"  {'Metric':<28}  {'Value':>10}")
    print(f"  {'-'*28}  {'-'*10}")
    print(f"  {'Mean Winrate':<28}  {aggregate['mean_winrate']:>10.2%}")
    print(f"  {'Mean Trade/Bulan':<28}  {aggregate['mean_trade_per_month']:>10.1f}")
    print(f"  {'Mean PnL Lev3x (USD)':<28}  {aggregate['mean_pnl_lev3x']:>+10.2f}")
    print(f"  {'Mean PnL Lev5x (USD)':<28}  {aggregate['mean_pnl_lev5x']:>+10.2f}")
    print(f"  {'Mean Max Drawdown Lev3x':<28}  {aggregate['mean_drawdown_lev3x']:>10.2%}")
    print(f"  {'Max Consecutive Loss':<28}  {aggregate['max_consecutive_loss']:>10}")
    print(f"{sep}")
    print(f"\n  Per-symbol winrate:")
    for sym, r in results.items():
        bar = "█" * int(r["winrate"] * 20)
        print(f"  {sym:<14} {r['winrate']:.2%}  {bar}")
    print(f"\n  Backtest : {out_path}")
    print(f"  Inference config : {inference_cfg_path}")

    # Print coin categories
    cv = inference_cfg["coins_validated"]
    print(f"\n  Koin recommended : {cv['recommended']}")
    print(f"  Koin acceptable  : {cv['acceptable']}")
    print(f"  Koin caution     : {cv['caution']}")
    print(f"{sep}\n")


if __name__ == "__main__":
    main()