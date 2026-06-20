"""
pipeline/01d_fetch_binance_vision_metrics.py
Download historical futures metrics dari data.binance.vision.

Data: OI, top-trader L/S, global L/S, taker L/S ratio — daily granularity
Coverage: BTC dari 2021, 13 koin inti dari 2022, altcoin baru dari 2024-2025
Output: data/positioning_hist/{coin}_metrics.parquet

Kolom output (setelah normalisasi):
  date, symbol, oi_base, oi_usd, toptrader_ls_ratio, global_ls_ratio, taker_ls_vol_ratio

Usage:
  python pipeline/01d_fetch_binance_vision_metrics.py
"""
import io, sys, time, zipfile, urllib.request, urllib.error
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from config import TRAINING_COINS, TRAIN_CUTOFF_DATE
from core.utils import setup_logger

logger = setup_logger("01d_fetch_metrics")

BASE_URL  = "https://data.binance.vision/data/futures/um/daily/metrics"
OUT_DIR   = ROOT / "data" / "positioning_hist"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Earliest start per coin berdasarkan coverage check
COIN_START = {
    "BTCUSDT":       datetime(2021, 1, 1),
    "ETHUSDT":       datetime(2022, 1, 1),
    "SOLUSDT":       datetime(2022, 1, 1),
    "BNBUSDT":       datetime(2022, 1, 1),
    "XRPUSDT":       datetime(2022, 1, 1),
    "DOGEUSDT":      datetime(2022, 1, 1),
    "ADAUSDT":       datetime(2022, 1, 1),
    "TRXUSDT":       datetime(2022, 1, 1),
    "1000SHIBUSDT":  datetime(2022, 1, 1),
    "AVAXUSDT":      datetime(2022, 1, 1),
    "LINKUSDT":      datetime(2022, 1, 1),
    "DOTUSDT":       datetime(2022, 1, 1),
    "NEARUSDT":      datetime(2022, 1, 1),
    "HBARUSDT":      datetime(2022, 1, 1),
    "SUIUSDT":       datetime(2024, 1, 1),
    "1000PEPEUSDT":  datetime(2024, 1, 1),
    "ARBUSDT":       datetime(2024, 1, 1),
    "TONUSDT":       datetime(2025, 1, 1),
    "POLUSDT":       datetime(2025, 1, 1),
    "TAOUSDT":       datetime(2025, 1, 1),
    "ONDOUSDT":      datetime(2025, 1, 1),
}

# Download sampai 1 hari sebelum TRAIN_CUTOFF (training set)
# Holdout period juga didownload untuk inference
END_DATE = datetime(2026, 6, 10)  # kemarin


def fetch_day(symbol: str, dt: datetime) -> pd.DataFrame | None:
    date_str = dt.strftime("%Y-%m-%d")
    url = f"{BASE_URL}/{symbol}/{symbol}-metrics-{date_str}.zip"
    try:
        with urllib.request.urlopen(url, timeout=10) as r:
            data = r.read()
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            with z.open(z.namelist()[0]) as f:
                df = pd.read_csv(f)
        return df
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None
        logger.warning(f"  HTTP {e.code} for {symbol} {date_str}")
        return None
    except Exception as e:
        logger.warning(f"  Error {symbol} {date_str}: {e}")
        return None


def fetch_coin(symbol: str) -> pd.DataFrame:
    start = COIN_START.get(symbol, datetime(2022, 1, 1))
    rows = []
    current = start
    total_days = (END_DATE - start).days + 1
    fetched = 0
    missing = 0

    logger.info(f"  {symbol}: {start.date()} -> {END_DATE.date()} ({total_days} days)")

    while current <= END_DATE:
        df = fetch_day(symbol, current)
        if df is not None and len(df) > 0:
            rows.append(df)
            fetched += 1
        else:
            missing += 1
        current += timedelta(days=1)
        # Small delay to avoid rate limit
        if fetched % 100 == 0 and fetched > 0:
            time.sleep(0.2)

    if not rows:
        logger.warning(f"  {symbol}: NO DATA")
        return pd.DataFrame()

    result = pd.concat(rows, ignore_index=True)
    logger.info(f"  {symbol}: {fetched} days fetched, {missing} missing")
    return result


def normalize_df(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    if df.empty:
        return df

    col_map = {
        "create_time":                    "date",
        "sum_open_interest":              "oi_base",
        "sum_open_interest_value":        "oi_usd",
        "sum_toptrader_long_short_ratio": "toptrader_ls_ratio",
        "count_toptrader_long_short_ratio": "toptrader_ls_acct",
        "count_long_short_ratio":         "global_ls_ratio",
        "sum_taker_long_short_vol_ratio": "taker_ls_vol_ratio",
    }
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df["symbol"] = symbol
    df = df.sort_values("date").drop_duplicates("date")

    # Derived features
    df["oi_usd_delta_pct"] = df["oi_usd"].pct_change().clip(-1, 1)
    df["taker_ls_delta"]   = df["taker_ls_vol_ratio"].diff()
    df["toptrader_ls_delta"] = df["toptrader_ls_ratio"].diff()

    keep = ["date", "symbol", "oi_base", "oi_usd", "oi_usd_delta_pct",
            "toptrader_ls_ratio", "toptrader_ls_acct", "toptrader_ls_delta",
            "global_ls_ratio", "taker_ls_vol_ratio", "taker_ls_delta"]
    return df[[c for c in keep if c in df.columns]]


def main():
    logger.info(f"Fetching Binance Vision metrics for {len(TRAINING_COINS)} coins")
    logger.info(f"Output: {OUT_DIR}")

    summary = []
    for symbol in TRAINING_COINS:
        raw = fetch_coin(symbol)
        if raw.empty:
            summary.append({"symbol": symbol, "rows": 0, "start": None, "end": None})
            continue

        df = normalize_df(raw, symbol)
        out_path = OUT_DIR / f"{symbol}_metrics.parquet"
        df.to_parquet(out_path, index=False)
        summary.append({
            "symbol": symbol,
            "rows": len(df),
            "start": str(df["date"].min().date()),
            "end": str(df["date"].max().date()),
        })
        logger.info(f"  Saved: {out_path.name} ({len(df)} rows)")

    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    print(f"{'Symbol':<20} {'Rows':>6}  {'Start':<12}  {'End':<12}")
    print("-" * 60)
    for s in summary:
        rows = s["rows"]
        start = s["start"] or "N/A"
        end = s["end"] or "N/A"
        print(f"{s['symbol']:<20} {rows:>6}  {start:<12}  {end:<12}")
    print("=" * 60)


if __name__ == "__main__":
    main()
