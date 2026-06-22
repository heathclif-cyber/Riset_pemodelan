"""
pipeline/01b_fetch_binance_extra.py — Fetch Binance Futures data tambahan

Tiga fitur gratis dari Binance Futures API untuk momentum model:
1. Taker Buy/Sell Volume Ratio
2. Top Trader Long/Short Ratio
3. Open Interest (sudah ada — compute delta 1h/4h/24h)

Output: {coin}_taker_ratio.parquet, {coin}_top_trader.parquet, {coin}_oi_delta.parquet

Usage:
  python pipeline/01b_fetch_binance_extra.py --all
  python pipeline/01b_fetch_binance_extra.py --all --holdout
"""

import argparse, sys, time, numpy as np, pandas as pd
from datetime import datetime, timezone
from pathlib import Path
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

ROOT = Path(__file__).parent.parent; sys.path.insert(0, str(ROOT))
from config import (
    TRAINING_COINS, LABEL_DIR, HOLDOUT_DIR,
    TRAIN_CUTOFF_DATE, BINANCE_BASE_URL, SLEEP_BETWEEN_REQUESTS,
)
from core.utils import setup_logger
from core.binance_client import BinanceClient
import requests

logger = setup_logger("01b_binance_extra")


def fetch_binance_data(client, endpoint, symbol, start_ms, end_ms, limit=500):
    """Generic Binance Futures data fetcher with pagination."""
    all_data = []
    current = start_ms
    step = 3600000 * limit  # hourly data, limit candles per request
    max_retries = 3

    while current < end_ms:
        params = {
            "symbol": symbol,
            "period": "1h",
            "startTime": current,
            "endTime": min(current + step, end_ms),
            "limit": limit,
        }
        for attempt in range(max_retries):
            try:
                resp = requests.get(
                    f"{BINANCE_BASE_URL}{endpoint}",
                    params=params, timeout=30, verify=False,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    if data:
                        all_data.extend(data)
                    break
                elif resp.status_code == 429:
                    time.sleep(10)
                else:
                    logger.warning(f"{symbol}: {resp.status_code} at {endpoint}")
            except Exception as e:
                logger.warning(f"{symbol}: error {e}")
                time.sleep(5)
        current += step
        time.sleep(SLEEP_BETWEEN_REQUESTS)

    if not all_data:
        return None

    df = pd.DataFrame(all_data)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"].astype(np.int64), unit="ms", utc=True)
        df = df.set_index("timestamp").sort_index()
    return df


def fetch_taker_ratio(client, symbol, start, end):
    """Fetch Taker Buy/Sell Volume Ratio."""
    start_ms = int(start.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000)

    # Try Binance Vision first (CDN, better for historical)
    # takerlongshortRatio endpoint (taker buy/sell volume ratio)
    for endpoint in ["/futures/data/takerlongshortRatio"]:
        try:
            df = fetch_binance_data(client, endpoint, symbol, start_ms, end_ms)
            if df is not None and len(df) > 0:
                return df
        except:
            continue
    return None


def fetch_top_trader_ratio(client, symbol, start, end):
    """Fetch Top Trader Long/Short Ratio (positions)."""
    start_ms = int(start.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000)
    return fetch_binance_data(client, "/futures/data/topLongShortPositionRatio", symbol, start_ms, end_ms)


def compute_oi_delta(coin, is_holdout):
    """Compute OI delta from existing OI data."""
    base_dir = HOLDOUT_DIR / "raw" if is_holdout else LABEL_DIR.parent / "training" / "funding_rate"
    oi_path = base_dir / f"{coin}_open_interest.parquet"

    # OI might be in different location, check
    if not oi_path.exists():
        oi_path = LABEL_DIR.parent / "training" / "funding_rate" / f"{coin}_open_interest.parquet"
    if not oi_path.exists():
        return None

    oi = pd.read_parquet(oi_path).sort_index()
    if "open_interest" not in oi.columns and len(oi.columns) > 0:
        oi_col = oi.columns[0]
    else:
        oi_col = "open_interest"

    df = pd.DataFrame(index=oi.index)
    df["oi_delta_1h"] = oi[oi_col].diff(1) / oi[oi_col].shift(1).abs()
    df["oi_delta_4h"] = oi[oi_col].diff(4) / oi[oi_col].shift(4).abs()
    df["oi_delta_24h"] = oi[oi_col].diff(24) / oi[oi_col].shift(24).abs()
    df = df.clip(-0.5, 0.5).fillna(0)
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--holdout", action="store_true")
    parser.add_argument("--coins", nargs="+", default=None)
    args = parser.parse_args()

    coins = args.coins or (TRAINING_COINS if args.all else TRAINING_COINS[:5])
    tag = "holdout" if args.holdout else "training"

    client = BinanceClient()

    print(f"\n{'='*60}")
    print(f"  BINANCE EXTRA DATA | {len(coins)} coins | {tag}")
    print(f"  Taker Ratio + Top Trader L/S + OI Delta")
    print(f"{'='*60}\n")

    for coin in coins:
        base_dir = HOLDOUT_DIR / "labeled" if args.holdout else LABEL_DIR
        out_dir = base_dir

        symbol = coin

        # Determine date range
        if args.holdout:
            start = TRAIN_CUTOFF_DATE
            end = datetime(2026, 4, 1, tzinfo=timezone.utc)
        else:
            start = datetime(2020, 1, 1, tzinfo=timezone.utc)
            end = TRAIN_CUTOFF_DATE

        # 1. Taker Buy/Sell Ratio
        tr_path = out_dir / f"{coin}_taker_ratio.parquet"
        if not tr_path.exists():
            df_tr = fetch_taker_ratio(client, symbol, start, end)
            if df_tr is not None and len(df_tr) > 0:
                df_tr.to_parquet(tr_path)
                logger.info(f"{coin}: Taker Ratio saved ({len(df_tr)} rows)")
            else:
                logger.warning(f"{coin}: Taker Ratio failed")
        else:
            logger.info(f"{coin}: Taker Ratio already exists")

        # 2. Top Trader Long/Short
        tt_path = out_dir / f"{coin}_top_trader.parquet"
        if not tt_path.exists():
            df_tt = fetch_top_trader_ratio(client, symbol, start, end)
            if df_tt is not None and len(df_tt) > 0:
                df_tt.to_parquet(tt_path)
                logger.info(f"{coin}: Top Trader saved ({len(df_tt)} rows)")
            else:
                logger.warning(f"{coin}: Top Trader failed")
        else:
            logger.info(f"{coin}: Top Trader already exists")

        # 3. OI Delta
        oi_path = out_dir / f"{coin}_oi_delta.parquet"
        if not oi_path.exists():
            df_oi = compute_oi_delta(coin, args.holdout)
            if df_oi is not None and len(df_oi) > 0:
                df_oi.to_parquet(oi_path)
                logger.info(f"{coin}: OI Delta saved ({len(df_oi)} rows)")
            else:
                logger.warning(f"{coin}: OI Delta failed (no OI data)")
        else:
            logger.info(f"{coin}: OI Delta already exists")

        time.sleep(0.5)

    print(f"\nDone! Files saved to {out_dir}")


if __name__ == "__main__":
    main()
