"""
pipeline/07_holdout_backtest.py — Hold-Out Backtest (Genuine Out-of-Sample)

Fetch data baru (default: Mei 2025 – Apr 2026), engineer fitur,
lalu backtest menggunakan 2-model cascade (LGBM - LSTM) TANPA retraining.

Output disimpan terpisah di:
  data/holdout/raw/        ← raw klines hold-out
  data/holdout/processed/  ← cleaned hold-out
  data/holdout/labeled/    ← fitur hold-out
  models/runs/{run_id}/holdout_backtest_results.json

Jalankan:
  python pipeline/07_holdout_backtest.py
  python pipeline/07_holdout_backtest.py --all
  python pipeline/07_holdout_backtest.py --start 2025-05-01 --end 2026-04-01
  python pipeline/07_holdout_backtest.py --coins SOLUSDT ETHUSDT
"""

import argparse
import json
import sys
import traceback
import warnings
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import pyarrow as pa
import pyarrow.parquet as pq
import torch

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from config import (
    TRAINING_COINS, ALL_COINS, SYMBOL_MAP,
    RAW_DIR, MODEL_DIR, REPORT_DIR,
    BINANCE_BASE_URL, SLEEP_BETWEEN_REQUESTS,
    SLEEP_ON_RATE_LIMIT, MAX_RETRIES, RETRY_BACKOFF_BASE,
    KLINE_INTERVALS, KLINE_LIMIT, FUNDING_LIMIT,
    LABEL_MAP, NUM_CLASSES, LSTM_SEQ_LEN,
    MODAL_PER_TRADE, LEVERAGE_SIM, FEE_PER_SIDE, SLIPPAGE_PER_SIDE,
    MAX_HOLDING_BARS, CONFIDENCE_THRESHOLD_ENTRY,
    SWING_LABEL_MAX_HOLD, SWING_LABEL_MIN_RR,
    SWING_LABEL_MIN_TP, SWING_LABEL_MAX_SL,
    FEATURE_COLS_V3, VP_WINDOW, VP_BINS,
    SWING_LOOKBACK, FVG_MIN_GAP_ATR,
    GUARDIAN_ENABLED,
    GUARDIAN_EXIT_THRESHOLD,
    TRAILING_STOP_ENABLED, TRAILING_STOP_ATR, TRAILING_STOP_MIN_BARS,
)
from core.binance_client import BinanceClient
from core.fetchers import fetch_coin, fetch_all_macro
from core.models import load_lstm
from core.evaluator import full_trading_report
from core.utils import setup_logger, ensure_utc_index
from core.features import engineer_features
from pipeline.backtest_utils import hierarchical_predict

logger = setup_logger("07_holdout_backtest")
from core.utils import get_lstm_device
DEVICE = torch.device("cpu")  # LSTM inference via CPU

# ── Hold-out directories (terpisah dari training data) ────────────────────────
HOLDOUT_DIR       = ROOT / "data" / "holdout"
HOLDOUT_RAW_DIR   = HOLDOUT_DIR / "raw"
HOLDOUT_PROC_DIR  = HOLDOUT_DIR / "processed"
HOLDOUT_LABEL_DIR = HOLDOUT_DIR / "labeled"

NON_FEATURE_COLS  = {"label", "h4_swing_high", "h4_swing_low"}
LONG_MAX_PIR      = 0.8
SHORT_MIN_PIR     = 0.2


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1 — FETCH
# ═══════════════════════════════════════════════════════════════════════════════

def fetch_holdout(coins: list[str], start: datetime, end: datetime) -> list[str]:
    """
    Fetch data hold-out. Karena fetch_coin tidak support raw_dir, 
    kita fetch ke RAW_DIR biasa lalu filter berdasarkan tanggal saat clean.
    Data training (sebelum start) tidak akan terpengaruh karena clean_holdout_symbol
    memfilter berdasarkan tanggal dan menyimpannya di holdout dir.
    """
    client = BinanceClient(
        base_url         = BINANCE_BASE_URL,
        sleep_between    = SLEEP_BETWEEN_REQUESTS,
        sleep_rate_limit = SLEEP_ON_RATE_LIMIT,
        max_retries      = MAX_RETRIES,
        backoff_base     = RETRY_BACKOFF_BASE,
    )
    if not client.test_connection():
        raise ConnectionError("Koneksi ke Binance gagal.")
    logger.info(f"Binance OK | Periode: {start.date()} - {end.date()}")

    # Fetch macro ke holdout raw (fear & greed, btc dominance)
    macro_holdout_dir = HOLDOUT_RAW_DIR / "macro"
    macro_holdout_dir.mkdir(parents=True, exist_ok=True)
    fetch_all_macro(start, end, progress={})

    success = []
    for i, symbol in enumerate(coins, 1):
        logger.info(f"[{i}/{len(coins)}] Fetching {symbol} hold-out...")
        try:
            result = fetch_coin(
                client        = client,
                symbol        = symbol,
                start         = start,
                end           = end,
                intervals     = KLINE_INTERVALS,
                progress      = {},
                kline_limit   = KLINE_LIMIT,
                funding_limit = FUNDING_LIMIT,
            )
            if result:
                # Simpan copy ke holdout dir dengan filter tanggal
                for tf in KLINE_INTERVALS:
                    src = RAW_DIR / "klines" / symbol / f"{tf}_all.parquet"
                    dst = HOLDOUT_RAW_DIR / "klines" / symbol / f"{tf}_all.parquet"
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    if src.exists():
                        df_raw = pd.read_parquet(src)
                        df_raw = ensure_utc_index(df_raw)
                        df_raw = df_raw[df_raw.index >= start]  # ← filter tanggal
                        _save_parquet(df_raw, dst)

                # Funding rate
                fr_src = RAW_DIR / "funding_rate" / f"{symbol}_8h.parquet"
                fr_dst = HOLDOUT_RAW_DIR / "funding_rate" / f"{symbol}_8h.parquet"
                fr_dst.parent.mkdir(parents=True, exist_ok=True)
                if fr_src.exists():
                    df_fr = pd.read_parquet(fr_src)
                    df_fr = ensure_utc_index(df_fr)
                    df_fr = df_fr[df_fr.index >= start]
                    _save_parquet(df_fr, fr_dst)

                success.append(symbol)
        except Exception as e:
            logger.error(f"[{symbol}] Fetch error: {e}")
    return success


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 2 — CLEAN
# ═══════════════════════════════════════════════════════════════════════════════

def _load(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        df = pd.read_parquet(path)
        return ensure_utc_index(df) if not df.empty else None
    except Exception as e:
        logger.warning(f"Gagal load {path}: {e}")
        return None


def _save_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(df, preserve_index=True)
    pq.write_table(table, str(path), compression="snappy")


def _fix_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    col_map = {c.lower(): c for c in df.columns}
    if not {"open", "high", "low", "close"}.issubset(col_map):
        return df
    ocols = [col_map[k] for k in ("open", "high", "low", "close")]
    mat   = df[ocols].values.astype(float)
    df    = df.copy()
    df[col_map["high"]] = np.nanmax(mat, axis=1)
    df[col_map["low"]]  = np.nanmin(mat, axis=1)
    return df


def clean_holdout_symbol(symbol: str) -> bool:
    """Clean satu koin dari HOLDOUT_RAW_DIR - HOLDOUT_PROC_DIR."""
    INTERVALS     = ["1h", "4h", "1d"]
    INTERVAL_FREQ = {"1h": "1h", "4h": "4h", "1d": "1D"}

    klines = {}
    for tf in INTERVALS:
        path = HOLDOUT_RAW_DIR / "klines" / symbol / f"{tf}_all.parquet"
        df   = _load(path)
        if df is not None:
            df = _fix_ohlc(df)
        klines[tf] = df

    base_h1 = klines.get("1h")
    if base_h1 is None:
        logger.error(f"[{symbol}] Tidak ada H1 klines hold-out — skip.")
        return False

    master = base_h1.copy()
    master.columns = [f"1h_{c}" for c in master.columns]

    for tf in ("4h", "1d"):
        df_tf = klines.get(tf)
        if df_tf is None:
            continue
        df_tf = df_tf.rename(columns={c: f"{tf}_{c}" for c in df_tf.columns})
        df_tf_h1 = df_tf.reindex(df_tf.index.union(master.index)).sort_index().ffill()
        master = master.join(df_tf_h1.reindex(master.index), how="left")

    # Funding rate
    fr_path = HOLDOUT_RAW_DIR / "funding_rate" / f"{symbol}_8h.parquet"
    df_fr   = _load(fr_path)
    if df_fr is not None:
        df_fr = df_fr.rename(columns={c: f"funding_rate_{c}" for c in df_fr.columns})
        df_fr_h1 = df_fr.reindex(df_fr.index.union(master.index)).sort_index().ffill()
        master = master.join(df_fr_h1.reindex(master.index), how="left")

    # Macro — load dari holdout raw, fallback ke training raw
    for name, fname in [
        ("btc_dominance",    "btc_dominance.parquet"),
        ("fear_greed_index", "fear_greed_index.parquet"),
    ]:
        path = HOLDOUT_RAW_DIR / "macro" / fname
        if not path.exists():
            path = RAW_DIR / "macro" / fname   # fallback ke training macro
        df_macro = _load(path)
        if df_macro is None:
            continue
        df_macro = df_macro.rename(columns={c: f"macro_{name}_{c}" for c in df_macro.columns})
        resampled = df_macro.reindex(
            df_macro.index.union(master.index)
        ).sort_index().ffill().reindex(master.index)
        master = master.join(resampled, how="left")

    out_path = HOLDOUT_PROC_DIR / f"{symbol}_clean.parquet"
    _save_parquet(master, out_path)
    logger.info(f"[{symbol}] Clean - {out_path} ({len(master):,} rows)")
    return True


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 3 — ENGINEER
# ═══════════════════════════════════════════════════════════════════════════════

def engineer_holdout_symbol(symbol: str) -> bool:
    """Engineer fitur dari HOLDOUT_PROC_DIR - HOLDOUT_LABEL_DIR."""
    in_path = HOLDOUT_PROC_DIR / f"{symbol}_clean.parquet"
    if not in_path.exists():
        logger.error(f"[{symbol}] Clean file tidak ada: {in_path}")
        return False

    df        = pd.read_parquet(in_path)
    df        = ensure_utc_index(df)
    symbol_id = SYMBOL_MAP.get(symbol, -1)

    try:
        feat_df = engineer_features(
            df                       = df,
            symbol                   = symbol,
            symbol_id                = symbol_id,
            max_hold                 = SWING_LABEL_MAX_HOLD,
            min_rr                   = SWING_LABEL_MIN_RR,
            min_tp_atr               = SWING_LABEL_MIN_TP,
            max_sl_atr               = SWING_LABEL_MAX_SL,
            vp_window                = VP_WINDOW,
            vp_bins                  = VP_BINS,
            swing_lookback           = SWING_LOOKBACK,
            fvg_min_gap              = FVG_MIN_GAP_ATR,
            long_max_price_in_range  = LONG_MAX_PIR,
            short_min_price_in_range = SHORT_MIN_PIR,
            add_label                = True,
        )

        cols_to_keep = [c for c in FEATURE_COLS_V3 if c in feat_df.columns] + \
                       ["label", "h4_swing_high", "h4_swing_low"]
        feat_df = feat_df[cols_to_keep]

        CRITICAL_COLS = ["open", "high", "low", "close", "volume", "atr_14_h1", "rsi_6", "label"]
        critical_present = [c for c in CRITICAL_COLS if c in feat_df.columns]
        feat_df = feat_df.dropna(subset=critical_present)

        for col in ["btc_dominance", "fear_greed", "long_short_ratio"]:
            if col in feat_df.columns:
                feat_df[col] = feat_df[col].ffill().fillna(0)

        out_path = HOLDOUT_LABEL_DIR / f"{symbol}_features_v3.parquet"
        _save_parquet(feat_df, out_path)
        logger.info(f"[{symbol}] Engineer - {out_path} ({len(feat_df):,} rows)")
        return True

    except Exception as e:
        logger.error(f"[{symbol}] Engineer error: {e}")
        logger.error(traceback.format_exc())
        return False


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 4 — BACKTEST (Hierarchical Cascade)
# ═══════════════════════════════════════════════════════════════════════════════
# get_ensemble_proba() dihapus — gunakan hierarchical_predict() dari 08_backtest

def backtest_holdout_symbol(
    symbol: str,
    feat_cols: list[str],
    lgbm_model,
    lstm_model,
    lstm_scaler,
    guardian_model    = None,
    guardian_scaler   = None,
    guardian_enabled  = False,
    guardian_exit_threshold = 0.60,
    trailing_stop_enabled   = False,
    trailing_stop_atr       = 2.0,
    trailing_stop_min_bars  = 2,
) -> dict | None:
    path = HOLDOUT_LABEL_DIR / f"{symbol}_features_v3.parquet"
    if not path.exists():
        logger.warning(f"[{symbol}] Hold-out features tidak ada — skip.")
        return None

    df = pd.read_parquet(path)
    df = ensure_utc_index(df)
    df = df.sort_index()

    mask = df["label"].astype(str).isin(LABEL_MAP)
    df   = df[mask].copy()
    y    = df["label"].map(LABEL_MAP).values.astype(np.int64)

    # Align kolom persis ke model LGBM — missing column di-fill 0
    X = np.zeros((len(df), len(feat_cols)), dtype=np.float64)
    for idx, col in enumerate(feat_cols):
        if col in df.columns:
            X[:, idx] = df[col].ffill().fillna(0).values.astype(np.float64)

    logger.info(f"[{symbol}] Hold-out inference: {len(df):,} bars...")

    # 2-model cascade — predict seluruh hold-out (murni out-of-sample)
    y_pred, confidence = hierarchical_predict(
        None, lgbm_model, lstm_model, lstm_scaler,
        X, feat_cols, [], df,
    )

    # Confidence filter
    below = (y_pred != 1) & (confidence < CONFIDENCE_THRESHOLD_ENTRY)
    y_pred_filtered = y_pred.copy()
    y_pred_filtered[below] = 1
    n_filtered = int(below.sum())
    logger.info(f"[{symbol}] Confidence filter: {n_filtered} sinyal di-skip")

    atr_arr   = df["atr_14_h1"].values if "atr_14_h1" in df.columns else np.ones(len(df))
    close_arr = df["close"].values    if "close"      in df.columns else np.ones(len(df))
    high_arr  = df["high"].values     if "high"       in df.columns else close_arr
    low_arr   = df["low"].values      if "low"        in df.columns else close_arr
    h4_sh_arr = df["h4_swing_high"].values if "h4_swing_high" in df.columns else None
    h4_sl_arr = df["h4_swing_low"].values  if "h4_swing_low"  in df.columns else None

    h4_trend_arr = df["h4_trend"].values if "h4_trend" in df.columns else None
    vol_ratio_arr = df["vol_ratio_20"].values if "vol_ratio_20" in df.columns else None

    # ── Guardian: pre-compute static feature array ──────────────────────────
    X_guardian = None
    if guardian_enabled and guardian_model is not None:
        from pipeline.backtest_utils import compute_guardian_static_array
        guardian_feat_path = MODEL_DIR / "guardian_feature_cols.json"
        if guardian_feat_path.exists():
            with open(guardian_feat_path) as f:
                g_feat_cols = json.load(f)
            g_static = [c for c in g_feat_cols if c not in [
                "bars_held_norm", "current_pnl_pct", "current_pnl_atr",
                "max_favorable_pnl_pct", "drawdown_from_peak_pct",
                "direction", "entry_price_ratio",
            ]]
            X_guardian = compute_guardian_static_array(df, g_static)

    report = full_trading_report(
        y_pred         = y_pred_filtered,
        y_actual       = y,
        atr            = atr_arr,
        close          = close_arr,
        high           = high_arr,
        low            = low_arr,
        h4_swing_highs = h4_sh_arr,
        h4_swing_lows  = h4_sl_arr,
        index          = df.index,
        modal          = MODAL_PER_TRADE,
        leverages      = LEVERAGE_SIM,
        fee_per_side   = FEE_PER_SIDE,
        slippage       = SLIPPAGE_PER_SIDE,
        min_rr         = SWING_LABEL_MIN_RR,
        min_tp_atr     = SWING_LABEL_MIN_TP,
        max_sl_atr     = SWING_LABEL_MAX_SL,
        max_hold       = MAX_HOLDING_BARS,
        symbol         = symbol,
        confidence     = confidence,
        guardian_model  = guardian_model,
        guardian_scaler = guardian_scaler,
        X_guardian      = X_guardian,
        guardian_enabled = guardian_enabled,
        guardian_exit_threshold = guardian_exit_threshold,
        trailing_stop_enabled   = trailing_stop_enabled,
        trailing_stop_atr       = trailing_stop_atr,
        trailing_stop_min_bars  = trailing_stop_min_bars,
        vol_ratio       = vol_ratio_arr,
        h4_trend        = h4_trend_arr,
    )
    report["n_filtered_by_confidence"] = n_filtered

    # ── Enrich trades for unified trade history ───────────────────────────────
    enriched_trades = []
    raw_trades = report.get("trades", [])
    for t in raw_trades:
        bar_in = t["bar_in"]
        bar_out = t["bar_out"]

        entry_idx = df.index[bar_in]
        exit_idx = df.index[min(bar_out, len(df) - 1)]

        opened_str = entry_idx.strftime("%Y-%m-%d %H:%M")
        closed_str = exit_idx.strftime("%Y-%m-%d %H:%M")

        # Trend
        h4_trend_val = df["h4_trend"].iloc[bar_in] if "h4_trend" in df.columns else np.nan
        if pd.isna(h4_trend_val) or np.isnan(h4_trend_val):
            h4_trend_str = "N/A"
        elif h4_trend_val > 0:
            h4_trend_str = "UP"
        elif h4_trend_val < 0:
            h4_trend_str = "DOWN"
        else:
            h4_trend_str = "RANGING"

        # Vol Regime
        vol_regime_val = df["vol_regime"].iloc[bar_in] if "vol_regime" in df.columns else np.nan
        vol_regime_str = f"{vol_regime_val:.2f}" if not (pd.isna(vol_regime_val) or np.isnan(vol_regime_val)) else "0.00"

        # H4 High / Low at entry
        h4_high = df["h4_swing_high"].iloc[bar_in] if "h4_swing_high" in df.columns else np.nan
        h4_low = df["h4_swing_low"].iloc[bar_in] if "h4_swing_low" in df.columns else np.nan

        # % to H4Hi / H4Lo
        entry_price = t["entry"]
        pct_to_hi_str = ""
        pct_to_lo_str = ""
        if not (pd.isna(h4_high) or np.isnan(h4_high)) and entry_price > 0:
            pct_to_hi = ((h4_high - entry_price) / entry_price) * 100
            pct_to_hi_str = f"{pct_to_hi:+.1f}"
        if not (pd.isna(h4_low) or np.isnan(h4_low)) and entry_price > 0:
            pct_to_lo = ((entry_price - h4_low) / entry_price) * 100
            pct_to_lo_str = f"{pct_to_lo:+.1f}"

        # Confidence at entry
        conf_val = confidence[bar_in] if confidence is not None else 1.0

        # Qty
        qty = 100.0

        # Exit reason mapping
        outcome = t["outcome"]
        exit_reason = "guardian_exit"
        if outcome == "WIN":
            exit_reason = "tp_hit"
        elif outcome == "LOSS":
            exit_reason = "sl_hit"
        elif outcome == "TIMEOUT":
            exit_reason = "time_exit"
        elif outcome == "TRAILING_STOP":
            exit_reason = "trailing_stop"
        elif "GUARDIAN" in outcome:
            exit_reason = "guardian_exit"

        # PnL %
        pnl_pct = (t["net_pnl"] / MODAL_PER_TRADE) * 100

        enriched = {
            "Opened": opened_str,
            "Closed": closed_str,
            "Coin": symbol,
            "Model": "cascade_v3.1",
            "Direction": t["direction"],
            "Conf": round(float(conf_val), 2),
            "Entry": round(float(t["entry"]), 8),
            "Exit": round(float(t["exit"]), 8),
            "TP": round(float(t["tp"]), 8),
            "SL": round(float(t["sl"]), 8),
            "ATR": round(float(df["atr_14_h1"].iloc[bar_in]), 8) if "atr_14_h1" in df.columns else 0.0,
            "% to H4Hi": pct_to_hi_str,
            "% to H4Lo": pct_to_lo_str,
            "RR": round(float(t["rr"]), 2),
            "H4 Trend": h4_trend_str,
            "Vol Regime": vol_regime_str,
            "H4 High": round(float(h4_high), 8) if not np.isnan(h4_high) else "",
            "H4 Low": round(float(h4_low), 8) if not np.isnan(h4_low) else "",
            "Qty": round(float(qty), 4),
            "Leverage": 5.0,
            "PnL ($)": round(float(t["net_pnl"]), 4),
            "PnL (%)": round(float(pnl_pct), 1),
            "Exit Reason": exit_reason,
            "Hold Bars": int(bar_out - bar_in),
            "Status": "closed"
        }
        enriched_trades.append(enriched)

    report["enriched_trades"] = enriched_trades
    return report


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(description="Hold-Out Backtest Pipeline")
    group  = parser.add_mutually_exclusive_group()
    group.add_argument("--all",   action="store_true")
    group.add_argument("--coins", nargs="+", metavar="SYMBOL")
    parser.add_argument("--start", default="2025-11-01",
                        help="Start date hold-out (default: 2025-11-01, match TRAIN_CUTOFF_DATE)")
    parser.add_argument("--end",   default="2026-04-01",
                        help="End date hold-out (default: 2026-04-01)")
    parser.add_argument("--skip-fetch",   action="store_true",
                        help="Skip fetch (gunakan data hold-out yang sudah ada)")
    parser.add_argument("--skip-clean",   action="store_true",
                        help="Skip cleaning")
    parser.add_argument("--skip-engineer", action="store_true",
                        help="Skip feature engineering")
    parser.add_argument("--run-id", default=None)
    return parser.parse_args()


def main():
    args  = parse_args()
    coins = ALL_COINS if args.all else (
        [c.upper() for c in args.coins] if args.coins else TRAINING_COINS
    )
    start = datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end   = datetime.strptime(args.end,   "%Y-%m-%d").replace(tzinfo=timezone.utc)
    run_id  = args.run_id or f"holdout_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = MODEL_DIR / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    for d in [HOLDOUT_RAW_DIR, HOLDOUT_PROC_DIR, HOLDOUT_LABEL_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    sep = "=" * 60
    print(f"\n{sep}")
    print(f"  HOLD-OUT BACKTEST — {run_id}")
    print(f"  Periode : {start.date()} - {end.date()}")
    print(f"  Koin    : {coins}")
    print(f"{sep}\n")

    # ── Step 1: Fetch ─────────────────────────────────────────────────────────
    if not args.skip_fetch:
        logger.info("=== STEP 1: FETCH HOLD-OUT DATA ===")
        fetched = fetch_holdout(coins, start, end)
        logger.info(f"Fetch selesai: {len(fetched)}/{len(coins)} koin berhasil")
    else:
        logger.info("=== STEP 1: SKIP FETCH ===")

    # ── Step 2: Clean ─────────────────────────────────────────────────────────
    if not args.skip_clean:
        logger.info("=== STEP 2: CLEAN HOLD-OUT DATA ===")
        for symbol in coins:
            try:
                clean_holdout_symbol(symbol)
            except Exception as e:
                logger.error(f"[{symbol}] Clean error: {e}")
    else:
        logger.info("=== STEP 2: SKIP CLEAN ===")

    # ── Step 3: Engineer ──────────────────────────────────────────────────────
    if not args.skip_engineer:
        logger.info("=== STEP 3: ENGINEER HOLD-OUT FEATURES ===")
        for symbol in coins:
            try:
                engineer_holdout_symbol(symbol)
            except Exception as e:
                logger.error(f"[{symbol}] Engineer error: {e}")
    else:
        logger.info("=== STEP 3: SKIP ENGINEER ===")

    # ── Step 4: Load models (2-model cascade: LGBM + LSTM) ───────────────────
    logger.info("=== STEP 4: BACKTEST (2-Model Cascade) ===")
    required_models = [
        (MODEL_DIR / "lgbm_baseline.pkl",    "LightGBM"),
        (MODEL_DIR / "lstm_best.pt",         "LSTM"),
        (MODEL_DIR / "lstm_scaler.pkl",      "LSTM Scaler"),
        (MODEL_DIR / "feature_cols_v2.json", "Feature cols"),
    ]
    for path, name in required_models:
        if not path.exists():
            raise FileNotFoundError(f"{name} tidak ditemukan: {path}")

    lgbm_model    = joblib.load(MODEL_DIR / "lgbm_baseline.pkl")
    lstm_model  = load_lstm(MODEL_DIR / "lstm_best.pt", device=str(DEVICE)).to(DEVICE)
    lstm_scaler = joblib.load(MODEL_DIR / "lstm_scaler.pkl")

    with open(MODEL_DIR / "feature_cols_v2.json") as f:
        feat_cols = json.load(f)

    # ── Guardian model (optional — graceful fallback) ────────────────────────
    guardian_model = None
    guardian_scaler = None
    guardian_enabled = GUARDIAN_ENABLED
    guardian_path = MODEL_DIR / "guardian_best.pkl"
    if guardian_path.exists() and guardian_enabled:
        guardian_model = joblib.load(guardian_path)
        guardian_scaler = joblib.load(MODEL_DIR / "guardian_scaler.pkl")
        logger.info(f"Guardian model loaded: {guardian_path.name}")
    elif guardian_enabled:
        logger.warning("GUARDIAN_ENABLED=True but guardian model not found")
        guardian_enabled = False

    logger.info(f"Models loaded | Device: {DEVICE} | Features: {len(feat_cols)}")

    # ── Step 5: Backtest per symbol ───────────────────────────────────────────
    results         = {}
    success, failed = [], []

    for symbol in coins:
        try:
            report = backtest_holdout_symbol(
                symbol, feat_cols,
                lgbm_model, lstm_model, lstm_scaler,
                guardian_model, guardian_scaler, guardian_enabled,
                guardian_exit_threshold=GUARDIAN_EXIT_THRESHOLD,
                trailing_stop_enabled=TRAILING_STOP_ENABLED,
                trailing_stop_atr=TRAILING_STOP_ATR,
                trailing_stop_min_bars=TRAILING_STOP_MIN_BARS,
            )
            if report:
                results[symbol] = report
                success.append(symbol)
                logger.info(
                    f"[{symbol}] Winrate: {report['winrate']:.2%} | "
                    f"Trades: {report['total_trades']} | "
                    f"DD lev5x: {report.get('max_drawdown_lev5x', 0):.2%}"
                )
            else:
                failed.append(symbol)
        except Exception as e:
            logger.error(f"[{symbol}] Backtest error: {e}")
            logger.error(traceback.format_exc())
            failed.append(symbol)

    if not results:
        logger.error("Tidak ada hasil — semua koin gagal.")
        return

    # ── Aggregate ─────────────────────────────────────────────────────────────
    all_wr  = [r["winrate"]                   for r in results.values()]
    all_tpm = [r["trade_per_month"]           for r in results.values()]
    all_dd5 = [r.get("max_drawdown_lev5x", 0) for r in results.values()]
    all_mcl = [r["max_consecutive_loss"]      for r in results.values()]
    all_sh  = [r.get("sharpe_ratio", 0)       for r in results.values()]
    all_so  = [r.get("sortino_ratio", 0)      for r in results.values()]
    all_ca  = [r.get("calmar_ratio", 0)       for r in results.values()]
    all_pf  = [r.get("profit_factor", 0)      for r in results.values()]

    aggregate = {
        "run_id":               run_id,
        "holdout_period":       f"{start.date()} - {end.date()}",
        "model_type":           "hierarchical_v1",
        "coins":                coins,
        "success":              success,
        "failed":               failed,
        "mean_winrate":         round(float(np.mean(all_wr)),  4),
        "std_winrate":          round(float(np.std(all_wr)),   4),
        "mean_trade_per_month": round(float(np.mean(all_tpm)), 2),
        "mean_drawdown_lev5x":  round(float(np.mean(all_dd5)), 4),
        "mean_sharpe":          round(float(np.mean(all_sh)),  4),
        "mean_sortino":         round(float(np.mean(all_so)),  4),
        "mean_calmar":          round(float(np.mean(all_ca)),  4),
        "mean_profit_factor":   round(float(np.mean(all_pf)),  4),
        "max_consecutive_loss": int(max(all_mcl)),
        "per_symbol":           results,
    }

    out_path = run_dir / "holdout_backtest_results.json"
    with open(out_path, "w") as f:
        json.dump(aggregate, f, indent=2, default=str)

    # ── Compile all enriched trades across all successfully backtested coins ──
    all_trades = []
    for symbol, r in results.items():
        if "enriched_trades" in r:
            all_trades.extend(r["enriched_trades"])

    worst_trade_pnl = 0.0
    p95_trade_loss = 0.0
    if all_trades:
        # Convert to DataFrame
        trade_df = pd.DataFrame(all_trades)
        worst_trade_pnl = float(trade_df["PnL (%)"].min())
        # Calculate 5th percentile (worst 5% threshold)
        p95_trade_loss = float(np.percentile(trade_df["PnL (%)"], 5))

        # Sort by Opened time chronological
        trade_df["Opened_dt"] = pd.to_datetime(trade_df["Opened"])
        trade_df = trade_df.sort_values(by="Opened_dt").drop(columns=["Opened_dt"])

        # Save path inside models/runs/{run_id}/
        csv_path = run_dir / "holdout_trade_history.csv"
        trade_df.to_csv(csv_path, index=False)
        logger.info(f"Unified trade history saved to: {csv_path}")

        # Also copy it to reports/experiments/ to be easily found by the user
        experiment_csv = ROOT / "reports" / "experiments" / f"{run_id}_holdout_trade_history.csv"
        experiment_csv.parent.mkdir(parents=True, exist_ok=True)
        trade_df.to_csv(experiment_csv, index=False)
        logger.info(f"Chronological database copy saved to: {experiment_csv}")

    # Add to aggregate dict
    aggregate["worst_single_trade_pnl"] = worst_trade_pnl
    aggregate["p95_single_trade_loss"] = p95_trade_loss

    # Re-save aggregate json with the new fields
    with open(out_path, "w") as f:
        json.dump(aggregate, f, indent=2, default=str)

    # ── Print summary ─────────────────────────────────────────────────────────
    print(f"\n{sep}")
    print(f"  HOLD-OUT BACKTEST SELESAI — {run_id}")
    print(f"  Periode  : {start.date()} - {end.date()}")
    print(f"{sep}")
    print(f"  {'Metric':<28}  {'Value':>10}")
    print(f"  {'-'*28}  {'-'*10}")
    print(f"  {'Mean Winrate':<28}  {aggregate['mean_winrate']:>10.2%}")
    print(f"  {'Mean Trade/Bulan':<28}  {aggregate['mean_trade_per_month']:>10.1f}")
    print(f"  {'Worst Single-Trade Loss':<28}  {worst_trade_pnl:>10.1f}%")
    print(f"  {'95% Trades Loss Under':<28}  {abs(p95_trade_loss):>10.1f}%")
    print(f"  {'Mean Max DD (Portfolio)':<28}  {aggregate['mean_drawdown_lev5x']:>10.2%}")
    print(f"  {'Mean Sharpe Ratio':<28}  {aggregate['mean_sharpe']:>10.2f}")
    print(f"  {'Mean Profit Factor':<28}  {aggregate['mean_profit_factor']:>10.2f}")
    print(f"  {'Max Consecutive Loss':<28}  {aggregate['max_consecutive_loss']:>10}")
    print(f"{sep}")
    print(f"\n  Per-symbol winrate:")
    for sym, r in results.items():
        bar = "#" * int(r["winrate"] * 20)
        print(f"  {sym:<14} {r['winrate']:.2%}  {bar}")
    print(f"\n  Output: {out_path}")
    print(f"{sep}\n")


if __name__ == "__main__":
    main()