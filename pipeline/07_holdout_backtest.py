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
    RAW_DIR, MODEL_DIR, REPORT_DIR, HOLDOUT_DIR,
    BINANCE_BASE_URL, SLEEP_BETWEEN_REQUESTS,
    SLEEP_ON_RATE_LIMIT, MAX_RETRIES, RETRY_BACKOFF_BASE,
    KLINE_INTERVALS, KLINE_LIMIT, FUNDING_LIMIT, OI_LIMIT, LONG_SHORT_LIMIT,
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

# ── Holdout-test directories (terpisah dari data/training/) ──────────────────
# HOLDOUT_DIR = data/holdout-test/ (diimport dari config)
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
    Fetch data hold-out LANGSUNG ke HOLDOUT_RAW_DIR (data/holdout/raw/).
    Tidak lagi melalui RAW_DIR — data OOS sepenuhnya terpisah dari training.

    PRASYARAT: Pastikan sudah menjalankan:
        python pipeline/01_fetch.py --all --oos
    untuk pre-fetch OOS data. Fungsi ini hanya dipakai jika belum ada data
    atau ingin re-fetch ulang dari dalam pipeline 07.
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
    logger.info(f"Output   : {HOLDOUT_RAW_DIR}")

    # Fetch macro ke holdout raw (fear & greed, btc dominance)
    macro_holdout_dir = HOLDOUT_RAW_DIR / "macro"
    macro_holdout_dir.mkdir(parents=True, exist_ok=True)
    fetch_all_macro(start, end, progress={})

    success = []
    for i, symbol in enumerate(coins, 1):
        logger.info(f"[{i}/{len(coins)}] Fetching {symbol} hold-out -> {HOLDOUT_RAW_DIR}...")
        try:
            result = fetch_coin(
                client           = client,
                symbol           = symbol,
                start            = start,
                end              = end,
                intervals        = KLINE_INTERVALS,
                progress         = {},
                kline_limit      = KLINE_LIMIT,
                funding_limit    = FUNDING_LIMIT,
                oi_limit         = OI_LIMIT,
                long_short_limit = LONG_SHORT_LIMIT,
                raw_dir          = HOLDOUT_RAW_DIR,  # <-- langsung ke holdout/raw/
            )
            if result:
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
    INTERVALS     = ["1h", "4h"]
    INTERVAL_FREQ = {"1h": "1h", "4h": "4h"}

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

    for tf in ("4h",):
        df_tf = klines.get(tf)
        if df_tf is None:
            continue
        # Geser 1 bar agar data lilin H4/D1 yang belum ditutup tidak bocor ke H1 master grid
        df_tf = df_tf.shift(1)
        df_tf = df_tf.rename(columns={c: f"{tf}_{c}" for c in df_tf.columns})
        df_tf_h1 = df_tf.reindex(df_tf.index.union(master.index)).sort_index().ffill()
        master = master.join(df_tf_h1.reindex(master.index), how="left")

    # Join BTC close price (untuk kalkulasi Relative Strength) jika bukan BTCUSDT sendiri
    if symbol != "BTCUSDT":
        btc_path = HOLDOUT_PROC_DIR / "BTCUSDT_clean.parquet"
        if btc_path.exists():
            try:
                btc_df = pd.read_parquet(btc_path)
                btc_close = btc_df["1h_close"].rename("btc_close")
                master = master.join(btc_close, how="left").ffill()
            except Exception as e:
                logger.warning(f"[{symbol}] Gagal load BTCUSDT_clean.parquet: {e}")
                master["btc_close"] = master["1h_close"]
        else:
            master["btc_close"] = master["1h_close"]
    else:
        master["btc_close"] = master["1h_close"]

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

    # Merge HMM regime labels jika tersedia
    regime_path = HOLDOUT_LABEL_DIR / f"{symbol}_regime_h1.parquet"
    if regime_path.exists():
        try:
            reg = pd.read_parquet(regime_path)
            if "hmm_regime_enc" in df.columns:
                df = df.drop(columns=["hmm_regime_enc"])
            df = df.join(reg[["hmm_regime_enc"]], how="left")
            df["hmm_regime_enc"] = df["hmm_regime_enc"].fillna(1).astype("int32")
        except Exception:
            pass

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
            from config import GUARDIAN_DYNAMIC_FEATURES as _GDYN
            g_static = [c for c in g_feat_cols if c not in set(_GDYN)]
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


def generate_markdown_report(aggregate, feat_cols, start, end, run_id, all_trades, run_dir):
    logger.info("Generating comprehensive Markdown report...")
    
    n_coins = len(aggregate.get("success", []))
    holdout_period = f"{start.date()} - {end.date()}"
    
    mean_winrate = float(aggregate.get("mean_winrate", 0.0))
    mean_tpm = float(aggregate.get("mean_trade_per_month", 0.0))
    mean_dd5x = float(aggregate.get("mean_drawdown_lev5x", 0.0))
    mean_sharpe = float(aggregate.get("mean_sharpe", 0.0))
    mean_sortino = float(aggregate.get("mean_sortino", 0.0))
    mean_calmar = float(aggregate.get("mean_calmar", 0.0))
    mean_pf = float(aggregate.get("mean_profit_factor", 0.0))
    max_consec_loss = int(aggregate.get("max_consecutive_loss", 0))
    worst_trade_pnl = float(aggregate.get("worst_single_trade_pnl", 0.0))
    p95_trade_loss = float(aggregate.get("p95_single_trade_loss", 0.0))
    
    total_trades = len(all_trades)
    long_count = 0
    short_count = 0
    long_wins = 0
    short_wins = 0
    total_pnl = 0.0
    
    avg_win_usd = 0.0
    avg_loss_usd = 0.0
    avg_win_pct = 0.0
    avg_loss_pct = 0.0
    
    wins_usd = []
    losses_usd = []
    wins_pct = []
    losses_pct = []
    
    exit_reasons = {}
    monthly_perf = {}
    
    for t in all_trades:
        direction = str(t.get("Direction", "")).upper()
        pnl_usd = float(t.get("PnL ($)", 0.0))
        pnl_pct = float(t.get("PnL (%)", 0.0))
        opened = str(t.get("Opened", ""))
        
        is_win = pnl_usd > 0
        total_pnl += pnl_usd
        
        if direction == "LONG":
            long_count += 1
            if is_win:
                long_wins += 1
        elif direction == "SHORT":
            short_count += 1
            if is_win:
                short_wins += 1
                
        if is_win:
            wins_usd.append(pnl_usd)
            wins_pct.append(pnl_pct)
        else:
            losses_usd.append(pnl_usd)
            losses_pct.append(pnl_pct)
            
        # Exit reason mapping
        exit_r = t.get("Exit Reason") or t.get("Outcome") or "unknown"
        exit_r = str(exit_r).lower()
        if "tp" in exit_r or "win" in exit_r:
            exit_r = "tp_hit"
        elif "sl" in exit_r or "loss" in exit_r:
            exit_r = "sl_hit"
        elif "guardian" in exit_r:
            if "momentum" in exit_r:
                exit_r = "guardian_momentum_exit"
            else:
                exit_r = "guardian_exit"
        elif "trailing" in exit_r:
            exit_r = "trailing_stop"
        elif "time" in exit_r or "timeout" in exit_r:
            exit_r = "time_exit"
            
        if exit_r not in exit_reasons:
            exit_reasons[exit_r] = {"count": 0, "wins": 0, "losses": 0, "pnl": 0.0}
        exit_reasons[exit_r]["count"] += 1
        exit_reasons[exit_r]["pnl"] += pnl_usd
        if is_win:
            exit_reasons[exit_r]["wins"] += 1
        else:
            exit_reasons[exit_r]["losses"] += 1
            
        # Monthly performance mapping
        if opened and len(opened) >= 7:
            month_str = opened[:7]
            if month_str not in monthly_perf:
                monthly_perf[month_str] = {"trades": 0, "wins": 0, "pnl": 0.0}
            monthly_perf[month_str]["trades"] += 1
            monthly_perf[month_str]["pnl"] += pnl_usd
            if is_win:
                monthly_perf[month_str]["wins"] += 1

    long_winrate = (long_wins / long_count) if long_count > 0 else 0.0
    short_winrate = (short_wins / short_count) if short_count > 0 else 0.0
    
    if wins_usd:
        avg_win_usd = float(np.mean(wins_usd))
        avg_win_pct = float(np.mean(wins_pct))
    if losses_usd:
        avg_loss_usd = float(np.mean(losses_usd))
        avg_loss_pct = float(np.mean(losses_pct))
        
    portfolio_roi = (total_pnl / (100.0 * n_coins)) * 100 if n_coins > 0 else 0.0
    trades_per_day = (mean_tpm * 12) / 365.25
    
    md = []
    md.append(f"# 📊 Holdout Backtest Report: `{run_id}`")
    md.append("")
    md.append(f"**Tanggal Pembuatan**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} UTC")
    md.append(f"**Model Run ID**: `{run_id}`")
    md.append(f"**Periode Pengujian (Temporal OOS)**: `{holdout_period}`")
    md.append("")
    
    md.append("> [!NOTE]")
    md.append("> **Ringkasan Portofolio Eksekutif**:")
    md.append(f"> *   **Total Net Profit**: **${total_pnl:+,.2f} USD** (ROI Portofolio: **{portfolio_roi:+.2f}%**)")
    md.append(f"> *   **Rata-rata Win Rate**: **{mean_winrate:.2%}** | Total Trades: **{total_trades:,}**")
    md.append(f"> *   **Rata-rata Max Drawdown (5x)**: **{mean_dd5x:.2%}**")
    md.append(f"> *   **Risk-Adjusted**: Sharpe: **{mean_sharpe:.2f}** | Sortino: **{mean_sortino:.2f}** | Calmar: **{mean_calmar:.2f}** | Profit Factor: **{mean_pf:.2f}**")
    md.append("")
    
    md.append("## 📈 Performa Scorecard Portofolio")
    md.append("")
    md.append("| Metrik Utama | Nilai Portofolio | Catatan |")
    md.append("|:---|:---:|:---|")
    md.append(f"| **Total Net Profit ($)** | `${total_pnl:+,.2f}` | Akumulasi keuntungan bersih 5x leverage |")
    md.append(f"| **Portfolio ROI (%)** | `{portfolio_roi:+.2f}%` | ROI berdasarkan kapital portofolio $100/koin |")
    md.append(f"| **Overall Win Rate** | `{mean_winrate:.2%}` | Rasio kemenangan rata-rata seluruh aset |")
    md.append(f"| **Total Trades** | `{total_trades:,}` | Jumlah total posisi yang dieksekusi |")
    md.append(f"| **Rata-rata Trade / Bulan** | `{mean_tpm:.1f}` | Rata-rata frekuensi trade bulanan portofolio |")
    md.append(f"| **Rata-rata Trade / Hari** | `{trades_per_day:.2f}` | Rata-rata frekuensi trade harian portofolio |")
    md.append(f"| **Max Drawdown (5x)** | `{mean_dd5x:.2%}` | Rata-rata penurunan terdalam portofolio |")
    md.append(f"| **Sharpe Ratio** | `{mean_sharpe:.2f}` | Efisiensi profit terhadap volatilitas portofolio |")
    md.append(f"| **Sortino Ratio** | `{mean_sortino:.2f}` | Efisiensi profit terhadap downside deviation |")
    md.append(f"| **Calmar Ratio** | `{mean_calmar:.2f}` | Rasio return tahunan terhadap drawdown |")
    md.append(f"| **Profit Factor** | `{mean_pf:.2f}` | Rasio gross profit dibagi gross loss |")
    md.append(f"| **Max Consecutive Loss** | `{max_consec_loss}` trades | Streak kekalahan beruntun terpanjang |")
    md.append(f"| **Worst Single Trade PnL** | `{worst_trade_pnl:+.2f}%` | Kerugian terdalam dalam satu trade tunggal |")
    md.append(f"| **95% Trades Loss Under** | `{abs(p95_trade_loss):.2f}%` | Nilai risiko (VaR P95) kerugian maksimal |")
    md.append("")
    
    md.append("## ↕️ Analisis Arah Signal (LONG vs SHORT)")
    md.append("")
    md.append("| Arah Posisi | Jumlah Trade | Distribusi | Menang | Kalah | Win Rate | PnL Bersih ($) |")
    md.append("|:---|:---:|:---:|:---:|:---:|:---:|:---:|")
    long_dist = long_count / total_trades if total_trades > 0 else 0
    short_dist = short_count / total_trades if total_trades > 0 else 0
    md.append(f"| **LONG** | {long_count:,} | {long_dist:.1%} | {long_wins:,} | {long_count - long_wins:,} | {long_winrate:.2%} | {sum(t['PnL ($)'] for t in all_trades if t['Direction'] == 'LONG'):+,.2f} |")
    md.append(f"| **SHORT** | {short_count:,} | {short_dist:.1%} | {short_wins:,} | {short_count - short_wins:,} | {short_winrate:.2%} | {sum(t['PnL ($)'] for t in all_trades if t['Direction'] == 'SHORT'):+,.2f} |")
    md.append("")
    
    md.append("### Rincian Rata-rata Profitabilitas per Trade")
    md.append("")
    md.append("| Tipe Trade | PnL Rata-rata ($) | PnL Rata-rata (%) |")
    md.append("|:---|:---:|:---:|")
    md.append(f"| **Trade Menang (Wins)** | `${avg_win_usd:+,.4f}` | `{avg_win_pct:+.2f}%` |")
    md.append(f"| **Trade Kalah (Losses)** | `${avg_loss_usd:+,.4f}` | `{avg_loss_pct:+.2f}%` |")
    md.append("")
    
    if monthly_perf:
        md.append("## 📅 Scorecard Bulanan Portofolio")
        md.append("")
        md.append("| Bulan | Total Trades | Wins | Losses | Win Rate | Net PnL ($) |")
        md.append("|:---|:---:|:---:|:---:|:---:|:---:|")
        for month_str in sorted(monthly_perf.keys()):
            p = monthly_perf[month_str]
            losses_count = p["trades"] - p["wins"]
            wr = p["wins"] / p["trades"] if p["trades"] > 0 else 0.0
            md.append(f"| {month_str} | {p['trades']} | {p['wins']} | {losses_count} | {wr:.2%} | ${p['pnl']:+,.2f} |")
        md.append("")
        
    if exit_reasons:
        md.append("## 🚪 Distribusi Alasan Keluar Posisi (Exit Reasons)")
        md.append("")
        md.append("| Alasan Exit | Jumlah | Persentase | Wins | Losses | Win Rate | PnL Bersih ($) |")
        md.append("|:---|:---:|:---:|:---:|:---:|:---:|:---:|")
        for exit_r, stats in sorted(exit_reasons.items(), key=lambda x: x[1]["count"], reverse=True):
            pct = stats["count"] / total_trades if total_trades > 0 else 0
            wr = stats["wins"] / stats["count"] if stats["count"] > 0 else 0.0
            md.append(f"| `{exit_r}` | {stats['count']:,} | {pct:.1%} | {stats['wins']:,} | {stats['losses']:,} | {wr:.2%} | ${stats['pnl']:+,.2f} |")
        md.append("")
        
    md.append("## 🪙 Scorecard Per Koin (Detailed Assets)")
    md.append("")
    md.append("| Token | Win Rate | Trades | LONG WR | SHORT WR | Net PnL ($) | Max DD | Sharpe | Sortino | Calmar | PF |")
    md.append("|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|")
    
    per_sym = aggregate.get("per_symbol", {})
    for sym in sorted(per_sym.keys()):
        s = per_sym[sym]
        s_wr = float(s.get("winrate", 0.0))
        s_tr = int(s.get("total_trades", 0))
        s_pnl = float(s.get("pnl_lev5x", 0.0))
        s_dd = float(s.get("max_drawdown_lev5x", 0.0))
        s_sh = float(s.get("sharpe_ratio", 0.0))
        s_so = float(s.get("sortino_ratio", 0.0))
        s_ca = float(s.get("calmar_ratio", 0.0))
        s_pf = float(s.get("profit_factor", 0.0))
        
        win_class = s.get("win_by_class", {})
        l_wr = float(win_class.get("LONG", 0.0))
        s_wr_class = float(win_class.get("SHORT", 0.0))
        
        md.append(f"| **{sym.replace('USDT','')}** | {s_wr:.2%} | {s_tr:,} | {l_wr:.1%} | {s_wr_class:.1%} | `${s_pnl:+,.2f}` | {s_dd:.2%} | {s_sh:.2f} | {s_so:.2f} | {s_ca:.2f} | {s_pf:.2f} |")
    md.append("")
    
    md.append("## ⛓️ Daftar Fitur Aktif dalam Model")
    md.append("")
    md.append(f"Total terdapat **{len(feat_cols)} fitur aktif** yang digunakan oleh LightGBM entry, LSTM Soft Confirmation, dan Exit Guardian v3:")
    md.append("")
    md.append("<details>")
    md.append("<summary>▶ Klik untuk melihat daftar lengkap fitur aktif</summary>")
    md.append("")
    for i, col in enumerate(feat_cols, 1):
        md.append(f"{i}. `{col}`")
    md.append("")
    md.append("</details>")
    md.append("")
    
    md_content = "\n".join(md)
    
    # 1. Save to reports/experiments/
    exp_report_dir = ROOT / "reports" / "experiments"
    exp_report_dir.mkdir(parents=True, exist_ok=True)
    report_path = exp_report_dir / f"{run_id}_holdout_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    logger.info(f"Markdown holdout report saved to: {report_path}")
    
    # 2. Save a copy to models/runs/{run_id}/
    run_report_path = run_dir / "holdout_report.md"
    with open(run_report_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    logger.info(f"Markdown holdout report copy saved to: {run_report_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

SCENARIO_PARAMS = {
    "A": {
        # cascade_v2.5_hybrid original — LGBM selektif, LSTM modifier bukan gatekeeper
        "lgbm_threshold_long":  0.69,
        "lgbm_threshold_short": 0.59,
        "lstm_neutral_pen":     0.00,
        "lstm_opposite_pen":    0.65,
        "lstm_agree_boost":     0.05,
        "confidence_entry":     0.59,
        "desc": "v2.5hybrid original (LGBM 0.69/0.59, LSTM neutral=0.0)",
    },
    "B": {
        # Relaxed LGBM threshold — LSTM sebagai filter yang lebih aktif
        "lgbm_threshold_long":  0.60,
        "lgbm_threshold_short": 0.55,
        "lstm_neutral_pen":     0.10,
        "lstm_opposite_pen":    0.55,
        "lstm_agree_boost":     0.10,
        "confidence_entry":     0.59,
        "desc": "relaxed threshold (LGBM 0.60/0.55, LSTM neutral=-0.10)",
    },
    "C": {
        # LGBM longgar, LSTM strict gatekeeper, conf rendah
        # Hipotesis: LSTM jadi penentu utama, volume tinggi, kualitas bergantung LSTM
        "lgbm_threshold_long":  0.45,
        "lgbm_threshold_short": 0.45,
        "lstm_neutral_pen":     0.99,
        "lstm_opposite_pen":    0.99,
        "lstm_agree_boost":     0.20,
        "confidence_entry":     0.59,
        "desc": "LGBM gate=0.45, LSTM strict gatekeeper (neutral=0.99), conf=0.59",
    },
    "D": {
        # Sama seperti C tapi conf_entry dinaikkan ke 0.65
        # Hipotesis: lebih sedikit trade tapi LSTM + conf bar = quality lebih tinggi
        "lgbm_threshold_long":  0.45,
        "lgbm_threshold_short": 0.45,
        "lstm_neutral_pen":     0.99,
        "lstm_opposite_pen":    0.99,
        "lstm_agree_boost":     0.20,
        "confidence_entry":     0.65,
        "desc": "LGBM gate=0.45, LSTM strict (neutral=0.99), conf=0.65",
    },
    "E": {
        # Production v3.1 exact — LGBM 0.45, boost 0.25, conf 0.65
        "lgbm_threshold_long":  0.45,
        "lgbm_threshold_short": 0.45,
        "lstm_neutral_pen":     0.99,
        "lstm_opposite_pen":    0.99,
        "lstm_agree_boost":     0.25,
        "confidence_entry":     0.65,
        "desc": "production v3.1 exact (LGBM 0.45, boost=0.25, conf=0.65)",
    },
    "F": {
        # LGBM longgar, LSTM partial filter — neutral dibiarkan, hanya opposite yang diblok
        # Hipotesis: volume lebih tinggi dari C/D tapi tetap blok sinyal berlawanan
        "lgbm_threshold_long":  0.45,
        "lgbm_threshold_short": 0.45,
        "lstm_neutral_pen":     0.30,
        "lstm_opposite_pen":    0.99,
        "lstm_agree_boost":     0.20,
        "confidence_entry":     0.59,
        "desc": "LGBM gate=0.45, LSTM blok opposite saja (neutral=0.30), conf=0.59",
    },
    "G": {
        # LGBM sedikit lebih tinggi (0.55), LSTM strict gatekeeper
        # Hipotesis: sweet spot antara A dan C — LGBM pre-filter + LSTM strict
        "lgbm_threshold_long":  0.55,
        "lgbm_threshold_short": 0.55,
        "lstm_neutral_pen":     0.99,
        "lstm_opposite_pen":    0.99,
        "lstm_agree_boost":     0.20,
        "confidence_entry":     0.59,
        "desc": "LGBM gate=0.55, LSTM strict gatekeeper (neutral=0.99), conf=0.59",
    },
    "H": {
        # Scenario A + LSTM Momentum Rescue
        # Masalah: saat pump/dump, LGBM bisa 0.55-0.68 (below 0.69 threshold)
        # tapi LSTM mendeteksi momentum kuat dari sequence 16 bar terakhir.
        # Fix: turunkan LSTM_OVERRIDE_THRESHOLD dari 0.70 ke 0.60 supaya
        # LSTM bisa rescue trade yang ditolak LGBM karena sedikit di bawah threshold.
        # LGBM directional review aktif untuk score > 0.45 (lebih tinggi dari default 0.35
        # untuk menghindari noise — hanya LGBM yang punya conviction cukup).
        "lgbm_threshold_long":        0.69,
        "lgbm_threshold_short":       0.59,
        "lstm_neutral_pen":           0.00,
        "lstm_opposite_pen":          0.65,
        "lstm_agree_boost":           0.05,
        "confidence_entry":           0.59,
        "lstm_flat_review_enabled":   True,
        "lstm_directional_threshold": 0.45,
        "lstm_override_threshold":    0.60,
        "desc": "Scenario A + momentum rescue (LSTM override thr=0.60, dir review > 0.45)",
    },
    "I": {
        # Scenario A + Dynamic Threshold berbasis vol_spike_zscore
        # vol_spike >= 2.0 → threshold turun 0.04 (misal 0.69 → 0.65)
        # vol_spike >= 3.0 → threshold turun 0.07 (misal 0.69 → 0.62)
        # Hipotesis: saat volume spike ekstrem, pasar sedang dalam momentum
        # kuat — threshold lebih rendah valid karena LGBM mungkin sedikit lag.
        "lgbm_threshold_long":              0.69,
        "lgbm_threshold_short":             0.59,
        "lstm_neutral_pen":                 0.00,
        "lstm_opposite_pen":                0.65,
        "lstm_agree_boost":                 0.05,
        "confidence_entry":                 0.59,
        "momentum_dynamic_threshold":       True,
        "momentum_spike_l1":                2.0,
        "momentum_spike_l2":                3.0,
        "momentum_reduce_l1":               0.04,
        "momentum_reduce_l2":               0.07,
        "desc": "Scenario A + dynamic threshold (vol_spike>=2→-0.04, >=3→-0.07)",
    },
    "J": {
        # Scenario I + Trend Score Asymmetric Threshold
        # Uptrend kuat  (h4_trend=1, trend_strength>2.0, ema_21_slope>0)
        #   → LONG threshold turun 0.05 (0.69→0.64), SHORT tidak berubah
        # Downtrend kuat (h4_trend=-1, trend_strength<-2.0, ema_21_slope<0)
        #   → SHORT threshold turun 0.05 (0.59→0.54), LONG tidak berubah
        # NOTE: Holdout Nov2025-Mar2026 = mixed/corrective period.
        # Hasil mungkin CONSERVATIVE — di bull market full benefit lebih besar.
        "lgbm_threshold_long":              0.69,
        "lgbm_threshold_short":             0.59,
        "lstm_neutral_pen":                 0.00,
        "lstm_opposite_pen":                0.65,
        "lstm_agree_boost":                 0.05,
        "confidence_entry":                 0.59,
        "momentum_dynamic_threshold":       True,
        "momentum_spike_l1":                2.0,
        "momentum_spike_l2":                3.0,
        "momentum_reduce_l1":               0.04,
        "momentum_reduce_l2":               0.07,
        "trend_dynamic_threshold":          True,
        "trend_strength_min":               2.0,
        "trend_reduce_amount":              0.05,
        "desc": "Scenario I + trend score asymmetric (uptrend→LONG-0.05, downtrend→SHORT-0.05)",
    },
    "K": {
        # Dual-Path: LGBM jalur structural + LSTM jalur momentum independen.
        # Saat LGBM di bawah threshold, LSTM bisa masuk sendiri jika confidence >= 0.65.
        # Murni output model — tidak ada filter teknikal tambahan.
        # Tujuan: tangkap momentum pump/dump yang LGBM belum cukup yakin,
        # tapi LSTM sudah deteksi lewat sequence 16 bar.
        "lgbm_threshold_long":       0.69,
        "lgbm_threshold_short":      0.59,
        "lstm_neutral_pen":          0.10,
        "lstm_opposite_pen":         0.85,
        "lstm_agree_boost":          0.05,
        "confidence_entry":          0.59,
        "lstm_standalone_enabled":   True,
        "lstm_standalone_threshold": 0.65,
        "desc": "Dual-Path: LGBM structural + LSTM momentum standalone (conf >= 0.65, no extra filters)",
    },

    # ── Smart Entry Experiments ───────────────────────────────────────────────
    # Kedua model berkontribusi bersamaan ke setiap bar.
    # LGBM tidak lagi gatekeeper tunggal.

    "L": {
        # Soft Vote 60/40, threshold 0.45
        # LGBM dominan tapi LSTM punya suara nyata
        "lgbm_threshold_long":  0.69, "lgbm_threshold_short": 0.59,
        "lstm_neutral_pen": 0.10, "lstm_opposite_pen": 0.85,
        "lstm_agree_boost": 0.05, "confidence_entry": 0.59,
        "smart_entry_mode": "soft_vote", "smart_entry_lgbm_w": 0.60,
        "smart_entry_threshold": 0.45,
        "desc": "Smart: Soft Vote 60/40, combined_thr=0.45",
    },
    "M": {
        # Soft Vote 50/50 — suara setara
        "lgbm_threshold_long":  0.69, "lgbm_threshold_short": 0.59,
        "lstm_neutral_pen": 0.10, "lstm_opposite_pen": 0.85,
        "lstm_agree_boost": 0.05, "confidence_entry": 0.59,
        "smart_entry_mode": "soft_vote", "smart_entry_lgbm_w": 0.50,
        "smart_entry_threshold": 0.45,
        "desc": "Smart: Soft Vote 50/50, combined_thr=0.45",
    },
    "N": {
        # Soft Vote 60/40, threshold lebih tinggi (0.52) — lebih selektif
        "lgbm_threshold_long":  0.69, "lgbm_threshold_short": 0.59,
        "lstm_neutral_pen": 0.10, "lstm_opposite_pen": 0.85,
        "lstm_agree_boost": 0.05, "confidence_entry": 0.59,
        "smart_entry_mode": "soft_vote", "smart_entry_lgbm_w": 0.60,
        "smart_entry_threshold": 0.52,
        "desc": "Smart: Soft Vote 60/40, combined_thr=0.52",
    },
    "O": {
        # Geometric Mean — strict consensus, crash protection kuat
        "lgbm_threshold_long":  0.69, "lgbm_threshold_short": 0.59,
        "lstm_neutral_pen": 0.10, "lstm_opposite_pen": 0.85,
        "lstm_agree_boost": 0.05, "confidence_entry": 0.59,
        "smart_entry_mode": "geometric", "smart_entry_lgbm_w": 0.60,
        "smart_entry_threshold": 0.38,
        "desc": "Smart: Geometric Mean, combined_thr=0.38",
    },
    "P": {
        # Dynamic Weight — model lebih yakin dapat bobot lebih
        "lgbm_threshold_long":  0.69, "lgbm_threshold_short": 0.59,
        "lstm_neutral_pen": 0.10, "lstm_opposite_pen": 0.85,
        "lstm_agree_boost": 0.05, "confidence_entry": 0.59,
        "smart_entry_mode": "dynamic_weight", "smart_entry_lgbm_w": 0.60,
        "smart_entry_threshold": 0.48,
        "desc": "Smart: Dynamic Weight (conf-proportional), combined_thr=0.48",
    },

    "Q": {
        # mutual_agree 60/40 — LGBM+LSTM bersama, LSTM punya veto
        # Saat LSTM netral (<38%): entry diblokir, exhaustion protection aktif
        # Saat LSTM agree (>=38%): combined score menentukan
        # LGBM tidak bisa masuk sendiri tanpa LSTM berkontribusi
        "lgbm_threshold_long":       0.69,
        "lgbm_threshold_short":      0.59,
        "lstm_neutral_pen":          0.10,
        "lstm_opposite_pen":         0.85,
        "lstm_agree_boost":          0.05,
        "confidence_entry":          0.59,
        "smart_entry_mode":          "mutual_agree",
        "smart_entry_lgbm_w":        0.60,
        "smart_entry_threshold":     0.50,
        "smart_entry_lstm_min":      0.38,
        "desc": "Smart mutual_agree 60/40: LSTM veto bila netral, combined>=0.50",
    },
    "R": {
        # mutual_agree 50/50 — suara benar-benar setara
        # LSTM dan LGBM punya bobot sama, LSTM tetap punya veto
        "lgbm_threshold_long":       0.69,
        "lgbm_threshold_short":      0.59,
        "lstm_neutral_pen":          0.10,
        "lstm_opposite_pen":         0.85,
        "lstm_agree_boost":          0.05,
        "confidence_entry":          0.59,
        "smart_entry_mode":          "mutual_agree",
        "smart_entry_lgbm_w":        0.50,
        "smart_entry_threshold":     0.50,
        "smart_entry_lstm_min":      0.38,
        "desc": "Smart mutual_agree 50/50: suara setara, LSTM veto bila netral",
    },

    "S": {
        # Dual Hard Gate — LGBM dan LSTM masing-masing punya threshold sendiri
        # Keduanya harus lolos secara INDEPENDEN dan arah SAMA
        # LGBM gate diturunkan ke 0.55 (lebih banyak kandidat)
        # tapi LSTM harus agree 0.42+ (di atas random 0.333)
        # True paralel — tidak ada yang bisa bypass yang lain
        "lgbm_threshold_long":     0.69, "lgbm_threshold_short": 0.59,
        "lstm_neutral_pen":        0.10, "lstm_opposite_pen":    0.85,
        "lstm_agree_boost":        0.05, "confidence_entry":     0.59,
        "smart_entry_mode":        "dual_gate",
        "smart_entry_lgbm_gate":   0.55,   # LGBM harus >= 55% untuk arahnya
        "smart_entry_lstm_gate":   0.42,   # LSTM harus >= 42% untuk arah SAMA
        "desc": "Dual Hard Gate: LGBM>=0.55 AND LSTM>=0.42, arah sama — true paralel",
    },
    "T": {
        # Dual Hard Gate — lebih ketat, gate lebih tinggi
        # LGBM 0.60 + LSTM 0.45 = lebih selektif
        "lgbm_threshold_long":     0.69, "lgbm_threshold_short": 0.59,
        "lstm_neutral_pen":        0.10, "lstm_opposite_pen":    0.85,
        "lstm_agree_boost":        0.05, "confidence_entry":     0.59,
        "smart_entry_mode":        "dual_gate",
        "smart_entry_lgbm_gate":   0.60,   # lebih ketat
        "smart_entry_lstm_gate":   0.45,   # lebih ketat
        "desc": "Dual Hard Gate: LGBM>=0.60 AND LSTM>=0.45 — lebih selektif",
    },

    # ── Tweaking Scenario T ───────────────────────────────────────────────────
    "T1": {
        # Longgarkan LSTM gate saja: 0.45 → 0.43
        # LGBM tetap ketat 0.60, LSTM lebih sering lolos
        "lgbm_threshold_long": 0.69, "lgbm_threshold_short": 0.59,
        "lstm_neutral_pen": 0.10, "lstm_opposite_pen": 0.85,
        "lstm_agree_boost": 0.05, "confidence_entry": 0.59,
        "smart_entry_mode": "dual_gate",
        "smart_entry_lgbm_gate": 0.60, "smart_entry_lstm_gate": 0.43,
        "desc": "T tweak: LGBM>=0.60 AND LSTM>=0.43",
    },
    "T2": {
        # Longgarkan LGBM gate saja: 0.60 → 0.58
        # LSTM tetap ketat 0.45
        "lgbm_threshold_long": 0.69, "lgbm_threshold_short": 0.59,
        "lstm_neutral_pen": 0.10, "lstm_opposite_pen": 0.85,
        "lstm_agree_boost": 0.05, "confidence_entry": 0.59,
        "smart_entry_mode": "dual_gate",
        "smart_entry_lgbm_gate": 0.58, "smart_entry_lstm_gate": 0.45,
        "desc": "T tweak: LGBM>=0.58 AND LSTM>=0.45",
    },
    "T3": {
        # Longgarkan keduanya sedikit: 0.60/0.45 → 0.58/0.43
        "lgbm_threshold_long": 0.69, "lgbm_threshold_short": 0.59,
        "lstm_neutral_pen": 0.10, "lstm_opposite_pen": 0.85,
        "lstm_agree_boost": 0.05, "confidence_entry": 0.59,
        "smart_entry_mode": "dual_gate",
        "smart_entry_lgbm_gate": 0.58, "smart_entry_lstm_gate": 0.43,
        "desc": "T tweak: LGBM>=0.58 AND LSTM>=0.43",
    },
    "T4": {
        # Longgarkan LSTM lebih agresif: 0.45 → 0.40
        # Test apakah PF masih terjaga di gate LSTM lebih rendah
        "lgbm_threshold_long": 0.69, "lgbm_threshold_short": 0.59,
        "lstm_neutral_pen": 0.10, "lstm_opposite_pen": 0.85,
        "lstm_agree_boost": 0.05, "confidence_entry": 0.59,
        "smart_entry_mode": "dual_gate",
        "smart_entry_lgbm_gate": 0.60, "smart_entry_lstm_gate": 0.40,
        "desc": "T tweak: LGBM>=0.60 AND LSTM>=0.40",
    },
    "T5": {
        # LSTM gate sangat longgar 0.35 — hampir semua sinyal LSTM lolos
        # LGBM tetap 0.60, test apakah LGBM strict cukup untuk jaga kualitas
        "lgbm_threshold_long": 0.69, "lgbm_threshold_short": 0.59,
        "lstm_neutral_pen": 0.10, "lstm_opposite_pen": 0.85,
        "lstm_agree_boost": 0.05, "confidence_entry": 0.59,
        "smart_entry_mode": "dual_gate",
        "smart_entry_lgbm_gate": 0.60, "smart_entry_lstm_gate": 0.35,
        "desc": "T5: LGBM>=0.60 AND LSTM>=0.35",
    },
    "T6": {
        # LGBM lebih longgar + LSTM 0.35
        "lgbm_threshold_long": 0.69, "lgbm_threshold_short": 0.59,
        "lstm_neutral_pen": 0.10, "lstm_opposite_pen": 0.85,
        "lstm_agree_boost": 0.05, "confidence_entry": 0.59,
        "smart_entry_mode": "dual_gate",
        "smart_entry_lgbm_gate": 0.58, "smart_entry_lstm_gate": 0.35,
        "desc": "T6: LGBM>=0.58 AND LSTM>=0.35",
    },
    "T7": {"lgbm_threshold_long": 0.69, "lgbm_threshold_short": 0.59,
           "lstm_neutral_pen": 0.10, "lstm_opposite_pen": 0.85,
           "lstm_agree_boost": 0.05, "confidence_entry": 0.59,
           "smart_entry_mode": "dual_gate",
           "smart_entry_lgbm_gate": 0.60, "smart_entry_lstm_gate": 0.50,
           "desc": "T7: LGBM>=0.60 AND LSTM>=0.50"},
    "T8": {"lgbm_threshold_long": 0.69, "lgbm_threshold_short": 0.59,
           "lstm_neutral_pen": 0.10, "lstm_opposite_pen": 0.85,
           "lstm_agree_boost": 0.05, "confidence_entry": 0.59,
           "smart_entry_mode": "dual_gate",
           "smart_entry_lgbm_gate": 0.58, "smart_entry_lstm_gate": 0.50,
           "desc": "T8: LGBM>=0.58 AND LSTM>=0.50"},
    "T9": {"lgbm_threshold_long": 0.69, "lgbm_threshold_short": 0.59,
           "lstm_neutral_pen": 0.10, "lstm_opposite_pen": 0.85,
           "lstm_agree_boost": 0.05, "confidence_entry": 0.59,
           "smart_entry_mode": "dual_gate",
           "smart_entry_lgbm_gate": 0.55, "smart_entry_lstm_gate": 0.50,
           "desc": "T9: LGBM>=0.55 AND LSTM>=0.50"},
    # ── Soft Gate: LGBM gates, LSTM hanya blok jika kuat berlawanan ──────────
    "U":  {"lgbm_threshold_long": 0.69, "lgbm_threshold_short": 0.59,
           "lstm_neutral_pen": 0.10, "lstm_opposite_pen": 0.85,
           "lstm_agree_boost": 0.05, "confidence_entry": 0.59,
           "smart_entry_mode": "soft_gate",
           "smart_entry_lgbm_gate": 0.60,
           "lstm_soft_gate_opp_max": 0.35,
           "desc": "U: Soft Gate — LGBM>=0.60, LSTM_opp<0.35 (tanpa argmax match)"},
    # ── Dual Gate Sweep: LGBM [0.40,0.50,0.60] x LSTM [0.37,0.39,0.41] ──────
    "V1": {"lgbm_threshold_long": 0.69, "lgbm_threshold_short": 0.59,
           "lstm_neutral_pen": 0.10, "lstm_opposite_pen": 0.85,
           "lstm_agree_boost": 0.05, "confidence_entry": 0.59,
           "smart_entry_mode": "dual_gate",
           "smart_entry_lgbm_gate": 0.40, "smart_entry_lstm_gate": 0.37,
           "desc": "V1: dual_gate LGBM>=0.40 AND LSTM>=0.37"},
    "V2": {"lgbm_threshold_long": 0.69, "lgbm_threshold_short": 0.59,
           "lstm_neutral_pen": 0.10, "lstm_opposite_pen": 0.85,
           "lstm_agree_boost": 0.05, "confidence_entry": 0.59,
           "smart_entry_mode": "dual_gate",
           "smart_entry_lgbm_gate": 0.40, "smart_entry_lstm_gate": 0.39,
           "desc": "V2: dual_gate LGBM>=0.40 AND LSTM>=0.39"},
    "V3": {"lgbm_threshold_long": 0.69, "lgbm_threshold_short": 0.59,
           "lstm_neutral_pen": 0.10, "lstm_opposite_pen": 0.85,
           "lstm_agree_boost": 0.05, "confidence_entry": 0.59,
           "smart_entry_mode": "dual_gate",
           "smart_entry_lgbm_gate": 0.40, "smart_entry_lstm_gate": 0.41,
           "desc": "V3: dual_gate LGBM>=0.40 AND LSTM>=0.41"},
    "V4": {"lgbm_threshold_long": 0.69, "lgbm_threshold_short": 0.59,
           "lstm_neutral_pen": 0.10, "lstm_opposite_pen": 0.85,
           "lstm_agree_boost": 0.05, "confidence_entry": 0.59,
           "smart_entry_mode": "dual_gate",
           "smart_entry_lgbm_gate": 0.50, "smart_entry_lstm_gate": 0.37,
           "desc": "V4: dual_gate LGBM>=0.50 AND LSTM>=0.37"},
    "V5": {"lgbm_threshold_long": 0.69, "lgbm_threshold_short": 0.59,
           "lstm_neutral_pen": 0.10, "lstm_opposite_pen": 0.85,
           "lstm_agree_boost": 0.05, "confidence_entry": 0.59,
           "smart_entry_mode": "dual_gate",
           "smart_entry_lgbm_gate": 0.50, "smart_entry_lstm_gate": 0.39,
           "desc": "V5: dual_gate LGBM>=0.50 AND LSTM>=0.39"},
    "V6": {"lgbm_threshold_long": 0.69, "lgbm_threshold_short": 0.59,
           "lstm_neutral_pen": 0.10, "lstm_opposite_pen": 0.85,
           "lstm_agree_boost": 0.05, "confidence_entry": 0.59,
           "smart_entry_mode": "dual_gate",
           "smart_entry_lgbm_gate": 0.50, "smart_entry_lstm_gate": 0.41,
           "desc": "V6: dual_gate LGBM>=0.50 AND LSTM>=0.41"},
    "V7": {"lgbm_threshold_long": 0.69, "lgbm_threshold_short": 0.59,
           "lstm_neutral_pen": 0.10, "lstm_opposite_pen": 0.85,
           "lstm_agree_boost": 0.05, "confidence_entry": 0.59,
           "smart_entry_mode": "dual_gate",
           "smart_entry_lgbm_gate": 0.60, "smart_entry_lstm_gate": 0.37,
           "desc": "V7: dual_gate LGBM>=0.60 AND LSTM>=0.37"},
    "V8": {"lgbm_threshold_long": 0.69, "lgbm_threshold_short": 0.59,
           "lstm_neutral_pen": 0.10, "lstm_opposite_pen": 0.85,
           "lstm_agree_boost": 0.05, "confidence_entry": 0.59,
           "smart_entry_mode": "dual_gate",
           "smart_entry_lgbm_gate": 0.60, "smart_entry_lstm_gate": 0.39,
           "desc": "V8: dual_gate LGBM>=0.60 AND LSTM>=0.39"},
    "V9": {"lgbm_threshold_long": 0.69, "lgbm_threshold_short": 0.59,
           "lstm_neutral_pen": 0.10, "lstm_opposite_pen": 0.85,
           "lstm_agree_boost": 0.05, "confidence_entry": 0.59,
           "smart_entry_mode": "dual_gate",
           "smart_entry_lgbm_gate": 0.60, "smart_entry_lstm_gate": 0.41,
           "desc": "V9: dual_gate LGBM>=0.60 AND LSTM>=0.41"},
    # ── Ratio Gate: direktional >= N× lawan di kedua model ───────────────────
    # W: tanpa floor — hanya arsip (terlalu banyak noise trade)
    "W":  {"lgbm_threshold_long": 0.69, "lgbm_threshold_short": 0.59,
           "lstm_neutral_pen": 0.10, "lstm_opposite_pen": 0.85,
           "lstm_agree_boost": 0.05, "confidence_entry": 0.0,
           "smart_entry_mode": "ratio_gate",
           "ratio_multiplier": 2.0,
           "desc": "W: Ratio Gate 2x tanpa floor (arsip — terlalu noise)"},
    # W2: ratio_gate + minimum LGBM floor 0.35
    "W2": {"lgbm_threshold_long": 0.69, "lgbm_threshold_short": 0.59,
           "lstm_neutral_pen": 0.10, "lstm_opposite_pen": 0.85,
           "lstm_agree_boost": 0.05, "confidence_entry": 0.35,
           "smart_entry_mode": "ratio_gate",
           "ratio_multiplier": 2.0,
           "desc": "W2: Ratio Gate 2x + conf_entry>=0.35 (filter noise)"},
    # W3: ratio_gate + floor 0.40
    "W3": {"lgbm_threshold_long": 0.69, "lgbm_threshold_short": 0.59,
           "lstm_neutral_pen": 0.10, "lstm_opposite_pen": 0.85,
           "lstm_agree_boost": 0.05, "confidence_entry": 0.40,
           "smart_entry_mode": "ratio_gate",
           "ratio_multiplier": 2.0,
           "desc": "W3: Ratio Gate 2x + conf_entry>=0.40"},
    # W4: ratio_gate + floor 0.45
    "W4": {"lgbm_threshold_long": 0.69, "lgbm_threshold_short": 0.59,
           "lstm_neutral_pen": 0.10, "lstm_opposite_pen": 0.85,
           "lstm_agree_boost": 0.05, "confidence_entry": 0.45,
           "smart_entry_mode": "ratio_gate",
           "ratio_multiplier": 2.0,
           "desc": "W4: Ratio Gate 2x + conf_entry>=0.45"},
    # W5: ratio 2x + FLAT < 0.70
    "W5": {"lgbm_threshold_long": 0.69, "lgbm_threshold_short": 0.59,
           "lstm_neutral_pen": 0.10, "lstm_opposite_pen": 0.85,
           "lstm_agree_boost": 0.05, "confidence_entry": 0.0,
           "smart_entry_mode": "ratio_gate",
           "ratio_multiplier": 2.0, "ratio_flat_max": 0.70,
           "desc": "W5: Ratio Gate 2x + FLAT<0.70 (noise filter murni)"},
    # W6: ratio 2x + FLAT < 0.60
    "W6": {"lgbm_threshold_long": 0.69, "lgbm_threshold_short": 0.59,
           "lstm_neutral_pen": 0.10, "lstm_opposite_pen": 0.85,
           "lstm_agree_boost": 0.05, "confidence_entry": 0.0,
           "smart_entry_mode": "ratio_gate",
           "ratio_multiplier": 2.0, "ratio_flat_max": 0.60,
           "desc": "W6: Ratio Gate 2x + FLAT<0.60 (lebih ketat)"},
    # ── LSTM Ratio Gate: LGBM standard threshold, LSTM pakai ratio 2x ─────────
    "X1": {"lgbm_threshold_long": 0.69, "lgbm_threshold_short": 0.59,
           "lstm_neutral_pen": 0.10, "lstm_opposite_pen": 0.85,
           "lstm_agree_boost": 0.05, "confidence_entry": 0.59,
           "smart_entry_mode": "lstm_ratio",
           "ratio_multiplier": 2.0, "ratio_flat_max": 0.70,
           "desc": "X1: LGBM std threshold + LSTM ratio 2x + FLAT<0.70"},
    "X2": {"lgbm_threshold_long": 0.69, "lgbm_threshold_short": 0.59,
           "lstm_neutral_pen": 0.10, "lstm_opposite_pen": 0.85,
           "lstm_agree_boost": 0.05, "confidence_entry": 0.59,
           "smart_entry_mode": "lstm_ratio",
           "ratio_multiplier": 2.0, "ratio_flat_max": 0.60,
           "desc": "X2: LGBM std threshold + LSTM ratio 2x + FLAT<0.60"},
    "X3": {"lgbm_threshold_long": 0.69, "lgbm_threshold_short": 0.59,
           "lstm_neutral_pen": 0.10, "lstm_opposite_pen": 0.85,
           "lstm_agree_boost": 0.05, "confidence_entry": 0.59,
           "smart_entry_mode": "lstm_ratio",
           "ratio_multiplier": 2.0, "ratio_flat_max": 0.50,
           "desc": "X3: LGBM std threshold + LSTM ratio 2x + FLAT<0.50 (LSTM direktional > FLAT)"},
    # ── LSTM Dominant Gate: LGBM standard, LSTM dominant dir >= threshold ─────
    "Y1": {"lgbm_threshold_long": 0.69, "lgbm_threshold_short": 0.59,
           "lstm_neutral_pen": 0.10, "lstm_opposite_pen": 0.85,
           "lstm_agree_boost": 0.05, "confidence_entry": 0.59,
           "smart_entry_mode": "lstm_dominant",
           "lstm_dominant_threshold": 0.33,
           "desc": "Y1: LSTM dominant >= 0.33 (sedikit di atas random 1/3)"},
    "Y2": {"lgbm_threshold_long": 0.69, "lgbm_threshold_short": 0.59,
           "lstm_neutral_pen": 0.10, "lstm_opposite_pen": 0.85,
           "lstm_agree_boost": 0.05, "confidence_entry": 0.59,
           "smart_entry_mode": "lstm_dominant",
           "lstm_dominant_threshold": 0.35,
           "desc": "Y2: LSTM dominant >= 0.35 (threshold user)"},
    "Y3": {"lgbm_threshold_long": 0.69, "lgbm_threshold_short": 0.59,
           "lstm_neutral_pen": 0.10, "lstm_opposite_pen": 0.85,
           "lstm_agree_boost": 0.05, "confidence_entry": 0.59,
           "smart_entry_mode": "lstm_dominant",
           "lstm_dominant_threshold": 0.40,
           "desc": "Y3: LSTM dominant >= 0.40 (lebih ketat)"},
    # ── Dual Dominant: dual gate struktur + LSTM dominant logic ──────────────
    "Z1": {"lgbm_threshold_long": 0.69, "lgbm_threshold_short": 0.59,
           "lstm_neutral_pen": 0.10, "lstm_opposite_pen": 0.85,
           "lstm_agree_boost": 0.05, "confidence_entry": 0.59,
           "smart_entry_mode": "dual_dominant",
           "smart_entry_lgbm_gate": 0.55,
           "lstm_dominant_threshold": 0.35,
           "desc": "Z1: dual_dominant LGBM>=0.55 + LSTM max(L,S)>=0.35"},
    "Z2": {"lgbm_threshold_long": 0.69, "lgbm_threshold_short": 0.59,
           "lstm_neutral_pen": 0.10, "lstm_opposite_pen": 0.85,
           "lstm_agree_boost": 0.05, "confidence_entry": 0.59,
           "smart_entry_mode": "dual_dominant",
           "smart_entry_lgbm_gate": 0.60,
           "lstm_dominant_threshold": 0.35,
           "desc": "Z2: dual_dominant LGBM>=0.60 + LSTM max(L,S)>=0.35"},
    "Z3": {"lgbm_threshold_long": 0.69, "lgbm_threshold_short": 0.59,
           "lstm_neutral_pen": 0.10, "lstm_opposite_pen": 0.85,
           "lstm_agree_boost": 0.05, "confidence_entry": 0.59,
           "smart_entry_mode": "dual_dominant",
           "smart_entry_lgbm_gate": 0.65,
           "lstm_dominant_threshold": 0.35,
           "desc": "Z3: dual_dominant LGBM>=0.65 + LSTM max(L,S)>=0.35"},
}


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
    parser.add_argument("--run-id", default=None,
                        help="Run ID untuk load model dari models/runs/{run_id}/")
    parser.add_argument("--scenario", choices=list(SCENARIO_PARAMS.keys()), default=None,
                        help="A=v2.5hybrid original, B=relaxed threshold. "
                             "Suffix _scenario_X ditambahkan ke output run_id.")
    return parser.parse_args()


def main():
    args  = parse_args()
    coins = ALL_COINS if args.all else (
        [c.upper() for c in args.coins] if args.coins else TRAINING_COINS
    )
    start = datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end   = datetime.strptime(args.end,   "%Y-%m-%d").replace(tzinfo=timezone.utc)
    base_id = args.run_id or f"holdout_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_id  = f"{base_id}_scenario_{args.scenario}" if args.scenario else base_id
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

    # Resolusi path model — prioritaskan run directory jika ada
    def _resolve(run_fname: str, root_fname: str) -> Path:
        if args.run_id:
            p = MODEL_DIR / "runs" / args.run_id / run_fname
            if p.exists():
                return p
        return MODEL_DIR / root_fname

    lgbm_path        = _resolve("lgbm.pkl",                    "lgbm_baseline.pkl")
    lstm_path        = _resolve("lstm_v2_style.pt",            "lstm_best.pt")
    lstm_scaler_path = _resolve("lstm_v2_style_scaler.pkl",    "lstm_scaler.pkl")
    feat_path        = _resolve("feature_cols_v2.json",        "feature_cols_v2.json")
    guardian_path    = _resolve("guardian.pkl",                "guardian_best.pkl")
    g_scaler_path    = _resolve("guardian_scaler.pkl",         "guardian_scaler.pkl")

    for path, name in [
        (lgbm_path,        "LightGBM"),
        (lstm_path,        "LSTM"),
        (lstm_scaler_path, "LSTM Scaler"),
        (feat_path,        "Feature cols"),
    ]:
        if not path.exists():
            raise FileNotFoundError(f"{name} tidak ditemukan: {path}")

    lgbm_model  = joblib.load(lgbm_path)
    lstm_model  = load_lstm(lstm_path, device=str(DEVICE)).to(DEVICE)
    lstm_scaler = joblib.load(lstm_scaler_path)

    with open(feat_path) as f:
        feat_cols = json.load(f)

    logger.info(f"LGBM   : {lgbm_path} ({len(lgbm_model.feature_name_)} fitur)")
    logger.info(f"LSTM   : {lstm_path}")
    logger.info(f"Feats  : {feat_path} ({len(feat_cols)} fitur)")

    # ── Guardian model (optional — graceful fallback) ────────────────────────
    guardian_model = None
    guardian_scaler = None
    guardian_enabled = GUARDIAN_ENABLED
    if guardian_path.exists() and guardian_enabled:
        guardian_model  = joblib.load(guardian_path)
        guardian_scaler = joblib.load(g_scaler_path)
        logger.info(f"Guardian: {guardian_path}")
    elif guardian_enabled:
        logger.warning("GUARDIAN_ENABLED=True but guardian model not found")
        guardian_enabled = False

    # ── Scenario parameter override ──────────────────────────────────────────
    import pipeline.backtest_utils as _bu
    global CONFIDENCE_THRESHOLD_ENTRY
    if args.scenario:
        sc = SCENARIO_PARAMS[args.scenario]
        _bu.LGBM_THRESHOLD_LONG      = sc["lgbm_threshold_long"]
        _bu.LGBM_THRESHOLD_SHORT     = sc["lgbm_threshold_short"]
        _bu.LSTM_ADJUST_NEUTRAL_PEN  = sc["lstm_neutral_pen"]
        _bu.LSTM_ADJUST_OPPOSITE_PEN = sc["lstm_opposite_pen"]
        _bu.LSTM_ADJUST_AGREE_BOOST  = sc["lstm_agree_boost"]
        CONFIDENCE_THRESHOLD_ENTRY   = sc["confidence_entry"]
        if "lstm_flat_review_enabled" in sc:
            _bu.LSTM_FLAT_REVIEW_ENABLED          = sc["lstm_flat_review_enabled"]
            _bu.LSTM_DIRECTIONAL_REVIEW_THRESHOLD = sc["lstm_directional_threshold"]
            _bu.LSTM_OVERRIDE_THRESHOLD           = sc["lstm_override_threshold"]
            logger.info(f"  LSTM rescue: flat_review={sc['lstm_flat_review_enabled']} "
                        f"dir_thr={sc['lstm_directional_threshold']} "
                        f"override_thr={sc['lstm_override_threshold']}")
        if sc.get("momentum_dynamic_threshold"):
            _bu.MOMENTUM_DYNAMIC_THRESHOLD_ENABLED = True
            _bu.MOMENTUM_SPIKE_L1  = sc.get("momentum_spike_l1",  2.0)
            _bu.MOMENTUM_SPIKE_L2  = sc.get("momentum_spike_l2",  3.0)
            _bu.MOMENTUM_REDUCE_L1 = sc.get("momentum_reduce_l1", 0.04)
            _bu.MOMENTUM_REDUCE_L2 = sc.get("momentum_reduce_l2", 0.07)
            logger.info(f"  Momentum dynamic threshold: "
                        f"L1={_bu.MOMENTUM_SPIKE_L1}→-{_bu.MOMENTUM_REDUCE_L1}, "
                        f"L2={_bu.MOMENTUM_SPIKE_L2}→-{_bu.MOMENTUM_REDUCE_L2}")
        else:
            _bu.MOMENTUM_DYNAMIC_THRESHOLD_ENABLED = False

        # LSTM Standalone (Dual-Path)
        if sc.get("lstm_standalone_enabled"):
            _bu.LSTM_STANDALONE_ENABLED   = True
            _bu.LSTM_STANDALONE_THRESHOLD = sc.get("lstm_standalone_threshold", 0.65)
            logger.info(f"  LSTM standalone enabled: threshold={_bu.LSTM_STANDALONE_THRESHOLD}")
        else:
            _bu.LSTM_STANDALONE_ENABLED = False

        # Smart Entry (Simultaneous Fusion)
        _bu.SMART_ENTRY_MODE      = sc.get("smart_entry_mode", "disabled")
        _bu.SMART_ENTRY_LGBM_W    = sc.get("smart_entry_lgbm_w", 0.60)
        _bu.SMART_ENTRY_THRESHOLD = sc.get("smart_entry_threshold", 0.45)
        _bu.SMART_ENTRY_LSTM_MIN  = sc.get("smart_entry_lstm_min", 0.38)
        _bu.SMART_ENTRY_LGBM_GATE = sc.get("smart_entry_lgbm_gate", 0.55)
        _bu.SMART_ENTRY_LSTM_GATE = sc.get("smart_entry_lstm_gate", 0.42)
        _bu.LSTM_SOFT_GATE_OPP_MAX = sc.get("lstm_soft_gate_opp_max", 0.35)
        _bu.RATIO_MULTIPLIER         = sc.get("ratio_multiplier", 2.0)
        _bu.RATIO_FLAT_MAX           = sc.get("ratio_flat_max", 0.70)
        _bu.LSTM_DOMINANT_THRESHOLD  = sc.get("lstm_dominant_threshold", 0.35)
        if _bu.SMART_ENTRY_MODE != "disabled":
            logger.info(
                f"  Smart Entry: mode={_bu.SMART_ENTRY_MODE} "
                f"lgbm_w={_bu.SMART_ENTRY_LGBM_W} "
                f"threshold={_bu.SMART_ENTRY_THRESHOLD}"
            )
        if sc.get("trend_dynamic_threshold"):
            _bu.TREND_DYNAMIC_THRESHOLD_ENABLED = True
            _bu.TREND_STRENGTH_MIN  = sc.get("trend_strength_min",  2.0)
            _bu.TREND_REDUCE_AMOUNT = sc.get("trend_reduce_amount", 0.05)
            logger.info(f"  Trend dynamic threshold: strength_min={_bu.TREND_STRENGTH_MIN}, "
                        f"reduce={_bu.TREND_REDUCE_AMOUNT} (LONG searah uptrend, SHORT searah downtrend)")
        else:
            _bu.TREND_DYNAMIC_THRESHOLD_ENABLED = False
        logger.info(f"Scenario {args.scenario}: {sc['desc']}")
        logger.info(f"  LGBM thr LONG={sc['lgbm_threshold_long']} SHORT={sc['lgbm_threshold_short']}")
        logger.info(f"  LSTM neutral_pen={sc['lstm_neutral_pen']} opposite_pen={sc['lstm_opposite_pen']} agree_boost={sc['lstm_agree_boost']}")
        logger.info(f"  Confidence entry={sc['confidence_entry']}")

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

    # Generate Markdown Report automatically
    generate_markdown_report(aggregate, feat_cols, start, end, run_id, all_trades, run_dir)

    # ── Print summary ─────────────────────────────────────────────────────────
    n_coins              = len(success)
    mean_tpm             = aggregate["mean_trade_per_month"]
    total_trades_per_month = mean_tpm * n_coins
    total_trades_per_day   = total_trades_per_month / 30.44
    total_pnl            = sum(float(t.get("PnL ($)", 0.0)) for t in all_trades)
    n_months             = max((end - start).days / 30.44, 1)

    print(f"\n{sep}")
    print(f"  HOLD-OUT BACKTEST SELESAI — {run_id}")
    print(f"  Periode  : {start.date()} - {end.date()}")
    print(f"{sep}")
    print(f"  {'Metric':<36}  {'Value':>12}")
    print(f"  {'-'*36}  {'-'*12}")
    print(f"  {'Win Rate (mean per koin)':<36}  {aggregate['mean_winrate']:>11.2%}")
    print(f"  {'Total PnL (semua koin, 5x)':<36}  ${total_pnl:>+10,.2f}")
    print(f"  {'Trade/Bulan (per koin, mean)':<36}  {mean_tpm:>12.1f}")
    print(f"  {'Trade/Bulan (semua koin total)':<36}  {total_trades_per_month:>12.0f}")
    print(f"  {'Trade/Hari (semua koin total)':<36}  {total_trades_per_day:>12.1f}")
    print(f"  {'Mean Sharpe Ratio':<36}  {aggregate['mean_sharpe']:>12.2f}")
    print(f"  {'Mean Profit Factor':<36}  {aggregate['mean_profit_factor']:>12.2f}")
    print(f"  {'Mean Max DD (per koin)':<36}  {aggregate['mean_drawdown_lev5x']:>11.2%}")
    print(f"  {'Max Consecutive Loss':<36}  {aggregate['max_consecutive_loss']:>12}")
    print(f"  {'Worst Single-Trade Loss':<36}  {worst_trade_pnl:>11.1f}%")
    print(f"  {'95% Trades Loss Under':<36}  {abs(p95_trade_loss):>11.1f}%")
    print(f"{sep}")
    print(f"\n  Per-koin: WR | Trade/Bulan | PnL (5x)")
    print(f"  {'-'*52}")
    for sym, r in results.items():
        coin_tpm = r["trade_per_month"]
        coin_pnl = r.get("pnl_lev5x", 0.0)
        bar = "#" * int(r["winrate"] * 20)
        print(f"  {sym:<16} {r['winrate']:>6.2%}  {coin_tpm:>6.1f}/bln  ${coin_pnl:>+8,.2f}  {bar}")
    print(f"\n  Output: {out_path}")
    print(f"{sep}\n")


if __name__ == "__main__":
    main()