"""
pipeline/fetch_extend_holdout_jun20.py
Extend holdout klines dari titik terakhir (Jun 13) hingga Jun 20 2026.

Strategi:
  - Fetch klines 1h/4h/1d per koin dari Jun 10 2026 (overlap safety) -> Jun 20 2026
  - Merge dengan parquet yang sudah ada: concat + dedup by index
  - Juga extend funding_rate per koin
  - Jalankan sekali, lalu pipeline 02/03 bisa di-re-run

Jalankan:
  python pipeline/fetch_extend_holdout_jun20.py
"""
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from config import ALL_COINS, HOLDOUT_DIR, BINANCE_BASE_URL, SLEEP_BETWEEN_REQUESTS, KLINE_LIMIT
from core.binance_client import BinanceClient
from core.utils import setup_logger, chunk_time_range, to_ms

logger = setup_logger("fetch_extend_holdout")

FETCH_FROM = datetime(2026, 6, 10, tzinfo=timezone.utc)
FETCH_TO   = datetime(2026, 6, 20, tzinfo=timezone.utc)
INTERVALS  = ["1h", "4h", "1d"]
KLINES_DIR = HOLDOUT_DIR / "raw" / "klines"
FUNDING_DIR = HOLDOUT_DIR / "raw" / "funding_rate"


def parse_klines(raw: list) -> pd.DataFrame:
    cols = [
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trades",
        "taker_buy_volume", "taker_buy_quote_volume", "_ignore"
    ]
    df = pd.DataFrame(raw, columns=cols)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df = df.set_index("open_time")
    df.index.name = "timestamp"
    float_cols = ["open", "high", "low", "close", "volume",
                  "quote_volume", "taker_buy_volume", "taker_buy_quote_volume"]
    df[float_cols] = df[float_cols].astype(float)
    df["trades"] = df["trades"].astype(int)
    df = df.drop(columns=["close_time", "_ignore"])
    return df


def merge_parquet(path: Path, new_df: pd.DataFrame) -> pd.DataFrame:
    if path.exists():
        old_df = pd.read_parquet(path)
        if not isinstance(old_df.index, pd.DatetimeIndex):
            old_df.index = pd.to_datetime(old_df.index, utc=True)
        if old_df.index.tz is None:
            old_df.index = old_df.index.tz_localize("UTC")
        # align columns
        common = old_df.columns.intersection(new_df.columns)
        merged = pd.concat([old_df[common], new_df[common]])
        merged = merged[~merged.index.duplicated(keep="last")].sort_index()
        return merged
    return new_df


def save_parquet(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(df, preserve_index=True)
    pq.write_table(table, str(path), compression="snappy")


def fetch_klines_range(client, symbol, interval, start, end):
    start_ms = to_ms(start)
    end_ms   = to_ms(end)
    frames = []
    for chunk_start, chunk_end in chunk_time_range(start_ms, end_ms, interval, KLINE_LIMIT):
        raw = client.get_klines(
            symbol=symbol, interval=interval,
            start_time_ms=chunk_start,
            end_time_ms=chunk_end - 1,
            limit=KLINE_LIMIT,
        )
        if raw:
            frames.append(parse_klines(raw))
        time.sleep(SLEEP_BETWEEN_REQUESTS)
    if not frames:
        return None
    df = pd.concat(frames)
    return df[~df.index.duplicated(keep="first")].sort_index()


def fetch_funding_range(client, symbol, start, end):
    from config import FUNDING_LIMIT
    start_ms = to_ms(start)
    end_ms   = to_ms(end)
    records = []
    cur = start_ms
    while cur < end_ms:
        chunk_end = min(cur + FUNDING_LIMIT * 8 * 3600 * 1000, end_ms)
        raw = client.get_funding_rate(
            symbol=symbol,
            start_time_ms=cur,
            end_time_ms=chunk_end,
            limit=FUNDING_LIMIT,
        )
        if not raw:
            break
        for item in raw:
            ts = int(item["fundingTime"])
            records.append({
                "timestamp":    datetime.fromtimestamp(ts / 1000, tz=timezone.utc),
                "funding_rate": float(item.get("fundingRate", 0.0)),
                "mark_price":   float(item.get("markPrice") or 0.0),
            })
        last_ts = int(raw[-1]["fundingTime"])
        if last_ts <= cur:
            break
        cur = last_ts + 1
        time.sleep(SLEEP_BETWEEN_REQUESTS)
    if not records:
        return None
    df = pd.DataFrame(records).set_index("timestamp")
    df.index = pd.DatetimeIndex(df.index, tz=timezone.utc)
    return df.sort_index()


def main():
    client = BinanceClient(
        base_url=BINANCE_BASE_URL,
        sleep_between=SLEEP_BETWEEN_REQUESTS,
        sleep_rate_limit=60.0,
        max_retries=3,
        backoff_base=2.0,
    )
    if not client.test_connection():
        logger.error("Koneksi ke Binance gagal!")
        sys.exit(1)

    logger.info(f"Extend holdout klines {FETCH_FROM.date()} -> {FETCH_TO.date()}")
    logger.info(f"Coins: {ALL_COINS}")

    ok_coins, fail_coins = [], []

    for sym in ALL_COINS:
        logger.info(f"\n[{sym}] Fetching klines + funding rate...")
        coin_ok = True

        for iv in INTERVALS:
            path = KLINES_DIR / sym / f"{iv}_all.parquet"
            try:
                new_df = fetch_klines_range(client, sym, iv, FETCH_FROM, FETCH_TO)
                if new_df is None or new_df.empty:
                    logger.warning(f"  [{sym}] {iv}: tidak ada data baru")
                    continue
                merged = merge_parquet(path, new_df)
                save_parquet(merged, path)
                logger.info(f"  [{sym}] {iv}: {len(merged):,} rows total -> {merged.index[-1]}")
            except Exception as e:
                logger.error(f"  [{sym}] {iv}: ERROR {e}")
                coin_ok = False

        # Funding rate
        try:
            fund_path = FUNDING_DIR / f"{sym}_8h.parquet"
            new_fund = fetch_funding_range(client, sym, FETCH_FROM, FETCH_TO)
            if new_fund is not None and not new_fund.empty:
                merged_fund = merge_parquet(fund_path, new_fund)
                save_parquet(merged_fund, fund_path)
                logger.info(f"  [{sym}] funding rate: {len(merged_fund)} records total")
        except Exception as e:
            logger.warning(f"  [{sym}] funding rate: {e}")

        if coin_ok:
            ok_coins.append(sym)
        else:
            fail_coins.append(sym)

    logger.info(f"\n=== SELESAI ===")
    logger.info(f"OK  : {ok_coins}")
    if fail_coins:
        logger.warning(f"FAIL: {fail_coins}")
    logger.info("Selanjutnya jalankan: 02_clean.py --all --holdout-test")


if __name__ == "__main__":
    main()
