"""
pipeline/09_holdout_backtest.py — Hold-Out Backtest (Genuine Out-of-Sample)

Fetch data baru (default: Mei 2025 – Apr 2026), engineer fitur,
lalu backtest menggunakan model yang sudah ada TANPA retraining.

Output disimpan terpisah di:
  data/holdout/raw/        ← raw klines hold-out
  data/holdout/processed/  ← cleaned hold-out
  data/holdout/labeled/    ← fitur hold-out
  models/runs/{run_id}/holdout_backtest_results.json

Jalankan:
  python pipeline/09_holdout_backtest.py
  python pipeline/09_holdout_backtest.py --all
  python pipeline/09_holdout_backtest.py --start 2025-05-01 --end 2026-04-01
  python pipeline/09_holdout_backtest.py --coins SOLUSDT ETHUSDT
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
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torch.utils.data import DataLoader

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
    MODAL_PER_TRADE, LEVERAGE_SIM, FEE_PER_SIDE,
    MAX_HOLDING_BARS, CONFIDENCE_THRESHOLD_ENTRY,
    SWING_LABEL_MAX_HOLD, SWING_LABEL_MIN_RR,
    SWING_LABEL_MIN_TP, SWING_LABEL_MAX_SL,
    FEATURE_COLS_V3, VP_WINDOW, VP_BINS,
    SWING_LOOKBACK, FVG_MIN_GAP_ATR,
)
from core.binance_client import BinanceClient
from core.fetchers import fetch_coin, fetch_all_macro
from core.models import load_lstm, ProbabilityCalibrator
from core.evaluator import full_trading_report
from core.utils import setup_logger, ensure_utc_index
from core.features import engineer_features
from pipeline.p05_utils import SequenceDataset

logger = setup_logger("09_holdout_backtest")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    logger.info(f"Binance OK | Periode: {start.date()} → {end.date()}")

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
    """Clean satu koin dari HOLDOUT_RAW_DIR → HOLDOUT_PROC_DIR."""
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
    logger.info(f"[{symbol}] Clean → {out_path} ({len(master):,} rows)")
    return True


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 3 — ENGINEER
# ═══════════════════════════════════════════════════════════════════════════════

def engineer_holdout_symbol(symbol: str) -> bool:
    """Engineer fitur dari HOLDOUT_PROC_DIR → HOLDOUT_LABEL_DIR."""
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
        logger.info(f"[{symbol}] Engineer → {out_path} ({len(feat_df):,} rows)")
        return True

    except Exception as e:
        logger.error(f"[{symbol}] Engineer error: {e}")
        logger.error(traceback.format_exc())
        return False


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 4 — BACKTEST
# ═══════════════════════════════════════════════════════════════════════════════

def get_ensemble_proba(
    lgbm_model, lstm_model, lstm_scaler,
    meta_learner, calibrator,
    X: np.ndarray, feat_cols: list[str], df_slice: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    lgbm_proba = lgbm_model.predict_proba(df_slice[feat_cols])

    X_sc   = lstm_scaler.transform(X)
    dummy  = np.zeros(len(X_sc), dtype=np.int64)
    ds     = SequenceDataset(X_sc, dummy)
    loader = DataLoader(ds, batch_size=1024, shuffle=False)

    lstm_list = []
    lstm_model.eval()
    with torch.no_grad():
        for xb, _ in loader:
            logits = lstm_model(xb.to(DEVICE))
            lstm_list.append(torch.softmax(logits, dim=1).cpu().numpy())
    lstm_proba = np.vstack(lstm_list)

    if len(lstm_proba) < len(lgbm_proba):
        pad = np.ones((len(lgbm_proba) - len(lstm_proba), NUM_CLASSES)) / NUM_CLASSES
        lstm_proba = np.vstack([pad, lstm_proba])

    meta_input = np.hstack([lgbm_proba, lstm_proba])
    cal_proba  = calibrator.transform(meta_learner.predict_proba(meta_input))
    return cal_proba.argmax(axis=1), cal_proba.max(axis=1)


def backtest_holdout_symbol(
    symbol: str, feat_cols: list[str],
    lgbm_model, lstm_model, lstm_scaler,
    meta_learner, calibrator,
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

    valid_cols = [c for c in feat_cols if c in df.columns]
    df[valid_cols] = df[valid_cols].ffill().fillna(0)
    X = df[valid_cols].values.astype(np.float64)

    logger.info(f"[{symbol}] Hold-out inference: {len(df):,} bars...")

    # Tidak ada fold — predict seluruh hold-out sekaligus (murni out-of-sample)
    y_pred, confidence = get_ensemble_proba(
        lgbm_model, lstm_model, lstm_scaler,
        meta_learner, calibrator,
        X, valid_cols, df[valid_cols],
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
        min_rr         = SWING_LABEL_MIN_RR,
        min_tp_atr     = SWING_LABEL_MIN_TP,
        max_sl_atr     = SWING_LABEL_MAX_SL,
        max_hold       = MAX_HOLDING_BARS,
        symbol         = symbol,
    )
    report["n_filtered_by_confidence"] = n_filtered
    return report


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(description="Hold-Out Backtest Pipeline")
    group  = parser.add_mutually_exclusive_group()
    group.add_argument("--all",   action="store_true")
    group.add_argument("--coins", nargs="+", metavar="SYMBOL")
    parser.add_argument("--start", default="2025-05-01",
                        help="Start date hold-out (default: 2025-05-01)")
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
    print(f"  Periode : {start.date()} → {end.date()}")
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

    # ── Step 4: Load models ───────────────────────────────────────────────────
    logger.info("=== STEP 4: BACKTEST ===")
    for path, name in [
        (MODEL_DIR / "lgbm_baseline.pkl",    "LightGBM"),
        (MODEL_DIR / "lstm_best.pt",         "LSTM"),
        (MODEL_DIR / "lstm_scaler.pkl",      "LSTM Scaler"),
        (MODEL_DIR / "ensemble_meta.pkl",    "Meta-learner"),
        (MODEL_DIR / "calibrator.pkl",       "Calibrator"),
        (MODEL_DIR / "feature_cols_v2.json", "Feature cols"),
    ]:
        if not path.exists():
            raise FileNotFoundError(f"{name} tidak ditemukan: {path}")

    lgbm_model   = joblib.load(MODEL_DIR / "lgbm_baseline.pkl")
    lstm_model   = load_lstm(MODEL_DIR / "lstm_best.pt",
                             device=str(DEVICE)).to(DEVICE)  # ← PERBAIKAN DI SINI
    lstm_scaler  = joblib.load(MODEL_DIR / "lstm_scaler.pkl")
    meta_learner = joblib.load(MODEL_DIR / "ensemble_meta.pkl")
    calibrator   = ProbabilityCalibrator.load(MODEL_DIR / "calibrator.pkl")

    with open(MODEL_DIR / "feature_cols_v2.json") as f:
        feat_cols = json.load(f)

    logger.info(f"Models loaded | Device: {DEVICE} | Features: {len(feat_cols)}")

    # ── Step 5: Backtest per symbol ───────────────────────────────────────────
    results         = {}
    success, failed = [], []

    for symbol in coins:
        try:
            report = backtest_holdout_symbol(
                symbol, feat_cols,
                lgbm_model, lstm_model, lstm_scaler,
                meta_learner, calibrator,
            )
            if report:
                results[symbol] = report
                success.append(symbol)
                logger.info(
                    f"[{symbol}] Winrate: {report['winrate']:.2%} | "
                    f"Trades: {report['total_trades']} | "
                    f"DD lev3x: {report.get('max_drawdown_lev3x', 0):.2%}"
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
    all_wr  = [r["winrate"]              for r in results.values()]
    all_tpm = [r["trade_per_month"]      for r in results.values()]
    all_dd3 = [r.get("max_drawdown_lev3x", 0) for r in results.values()]
    all_mcl = [r["max_consecutive_loss"] for r in results.values()]

    aggregate = {
        "run_id":               run_id,
        "holdout_period":       f"{start.date()} → {end.date()}",
        "coins":                coins,
        "success":              success,
        "failed":               failed,
        "mean_winrate":         round(float(np.mean(all_wr)),  4),
        "std_winrate":          round(float(np.std(all_wr)),   4),
        "mean_trade_per_month": round(float(np.mean(all_tpm)), 2),
        "mean_drawdown_lev3x":  round(float(np.mean(all_dd3)), 4),
        "max_consecutive_loss": int(max(all_mcl)),
        "per_symbol":           results,
    }

    out_path = run_dir / "holdout_backtest_results.json"
    with open(out_path, "w") as f:
        json.dump(aggregate, f, indent=2, default=str)

    # ── Print summary ─────────────────────────────────────────────────────────
    print(f"\n{sep}")
    print(f"  HOLD-OUT BACKTEST SELESAI — {run_id}")
    print(f"  Periode  : {start.date()} → {end.date()}")
    print(f"{sep}")
    print(f"  {'Metric':<28}  {'Value':>10}")
    print(f"  {'-'*28}  {'-'*10}")
    print(f"  {'Mean Winrate':<28}  {aggregate['mean_winrate']:>10.2%}")
    print(f"  {'Mean Trade/Bulan':<28}  {aggregate['mean_trade_per_month']:>10.1f}")
    print(f"  {'Mean Max DD Lev3x':<28}  {aggregate['mean_drawdown_lev3x']:>10.2%}")
    print(f"  {'Max Consecutive Loss':<28}  {aggregate['max_consecutive_loss']:>10}")
    print(f"{sep}")
    print(f"\n  Per-symbol winrate:")
    for sym, r in results.items():
        bar = "█" * int(r["winrate"] * 20)
        print(f"  {sym:<14} {r['winrate']:.2%}  {bar}")
    print(f"\n  Output: {out_path}")
    print(f"{sep}\n")


if __name__ == "__main__":
    main()