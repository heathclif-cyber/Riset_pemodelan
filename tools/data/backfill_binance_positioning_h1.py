"""
Backfill historis OI + L/S H1 dari Binance Futures (paginated, >500 jam).

Menulis ke path yang dipakai 02_clean + positioning:
  data/training/open_interest/{coin}_1h.parquet
  data/training/long_short_ratio/{coin}_1h.parquet
  data/positioning/{coin}_binance_oi.parquet
  data/positioning/{coin}_global_ls.parquet
  data/positioning/{coin}_top_trader.parquet
  data/positioning/{coin}_top_account.parquet

Usage:
  python tools/data/backfill_binance_positioning_h1.py --coins BTCUSDT
  python tools/data/backfill_binance_positioning_h1.py --all
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from pipeline._bootstrap import setup_path_from_file

ROOT = setup_path_from_file(__file__)
from config import TRAINING_COINS, TRAIN_START, RAW_DIR
from core.binance_client import BinanceClient
from core.fetchers import _parse_long_short_ratio, _parse_open_interest
from core.utils import setup_logger, to_ms

logger = setup_logger("backfill_positioning_h1")

POSITIONING_DIR = ROOT / "data" / "positioning"
OI_DIR = RAW_DIR / "open_interest"
LS_DIR = RAW_DIR / "long_short_ratio"
HOUR_MS = 3_600_000
LIMIT = 500
SLEEP = 0.15

ENDPOINTS = {
    "oi": ("/futures/data/openInterestHist", "open_interest"),
    "global_ls": ("/futures/data/globalLongShortAccountRatio", "long_short_ratio"),
    "top_trader": ("/futures/data/topLongShortPositionRatio", "top_trader_ls_ratio"),
    "top_account": ("/futures/data/topLongShortAccountRatio", "top_account_ls_ratio"),
}


def _parse_ratio(raw: list, value_col: str) -> pd.DataFrame:
    rows = []
    for item in raw:
        ts = int(item["timestamp"])
        rows.append({
            "timestamp": datetime.fromtimestamp(ts / 1000, tz=timezone.utc),
            value_col: float(item.get("longShortRatio") or item.get("longShortRatio", 0) or 0),
        })
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).set_index("timestamp").sort_index()
    df.index = pd.DatetimeIndex(df.index, tz=timezone.utc)
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def paginate_hist(
    client: BinanceClient,
    symbol: str,
    endpoint: str,
    start: datetime,
    end: datetime,
    value_col: str,
) -> pd.DataFrame:
    start_ms = to_ms(start)
    end_ms = to_ms(end)
    current = start_ms
    frames: list[pd.DataFrame] = []
    empty_streak = 0

    while current < end_ms:
        params = {
            "symbol": symbol,
            "period": "1h",
            "limit": LIMIT,
            "startTime": current,
            "endTime": end_ms,
        }
        raw = client._get(endpoint, params)
        if not raw:
            empty_streak += 1
            if empty_streak >= 3:
                break
            current += LIMIT * HOUR_MS
            time.sleep(SLEEP)
            continue
        empty_streak = 0

        if endpoint == "/futures/data/openInterestHist":
            chunk = _parse_open_interest(raw)
        elif value_col in ("long_short_ratio",):
            chunk = _parse_long_short_ratio(raw)
        else:
            chunk = _parse_ratio(raw, value_col)

        if chunk.empty:
            current += LIMIT * HOUR_MS
            time.sleep(SLEEP)
            continue

        frames.append(chunk)
        last_ms = int(chunk.index[-1].timestamp() * 1000)
        next_ms = last_ms + HOUR_MS
        if next_ms <= current:
            current += LIMIT * HOUR_MS
        else:
            current = next_ms

        if len(raw) < LIMIT:
            break
        time.sleep(SLEEP)

    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames)
    out = out[~out.index.duplicated(keep="last")].sort_index()
    return out


def save_parquet(path: Path, df: pd.DataFrame) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        old = pd.read_parquet(path)
        df = pd.concat([old, df])
        df = df[~df.index.duplicated(keep="last")].sort_index()
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df.to_parquet(path)
    return len(df)


def backfill_coin(client: BinanceClient, coin: str, start: datetime, end: datetime) -> dict:
    logger.info(f"[{coin}] backfill {start.date()} -> {end.date()}")
    stats = {}

    oi = paginate_hist(client, coin, *ENDPOINTS["oi"][:1], start, end, ENDPOINTS["oi"][1])
    if not oi.empty:
        n = save_parquet(OI_DIR / f"{coin}_1h.parquet", oi)
        save_parquet(POSITIONING_DIR / f"{coin}_binance_oi.parquet", oi[["open_interest"]])
        stats["oi"] = n
        logger.info(f"[{coin}] OI: {len(oi):,} new chunks -> {n:,} total")

    gls = paginate_hist(client, coin, *ENDPOINTS["global_ls"][:1], start, end, ENDPOINTS["global_ls"][1])
    if not gls.empty:
        n = save_parquet(LS_DIR / f"{coin}_1h.parquet", gls)
        ren = gls.rename(columns={"long_short_ratio": "global_ls_ratio"})
        save_parquet(POSITIONING_DIR / f"{coin}_global_ls.parquet", ren)
        stats["global_ls"] = n
        logger.info(f"[{coin}] global L/S: {len(gls):,} -> {n:,} total")

    tt = paginate_hist(client, coin, *ENDPOINTS["top_trader"][:1], start, end, ENDPOINTS["top_trader"][1])
    if not tt.empty:
        n = save_parquet(POSITIONING_DIR / f"{coin}_top_trader.parquet", tt)
        stats["top_trader"] = n
        logger.info(f"[{coin}] top trader: {len(tt):,} -> {n:,} total")

    ta = paginate_hist(client, coin, *ENDPOINTS["top_account"][:1], start, end, ENDPOINTS["top_account"][1])
    if not ta.empty:
        n = save_parquet(POSITIONING_DIR / f"{coin}_top_account.parquet", ta)
        stats["top_account"] = n
        logger.info(f"[{coin}] top account: {len(ta):,} -> {n:,} total")

    return stats


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--coins", nargs="+", default=None)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--start", default=None, help="YYYY-MM-DD UTC (default TRAIN_START)")
    parser.add_argument("--end", default=None, help="YYYY-MM-DD UTC (default now)")
    args = parser.parse_args()

    coins = TRAINING_COINS if args.all or not args.coins else args.coins
    start = datetime.fromisoformat(args.start).replace(tzinfo=timezone.utc) if args.start else TRAIN_START
    end = datetime.fromisoformat(args.end).replace(tzinfo=timezone.utc) if args.end else datetime.now(timezone.utc)

    client = BinanceClient()
    if not client.test_connection():
        logger.error("Binance connection failed")
        return 1

    summary = {}
    for coin in coins:
        try:
            summary[coin] = backfill_coin(client, coin, start, end)
        except Exception as e:
            logger.error(f"[{coin}] backfill error: {e}", exc_info=True)
            summary[coin] = {"error": str(e)}

    print("\n=== BACKFILL SUMMARY ===")
    for coin, st in summary.items():
        print(f"  {coin}: {st}")
    return 0


if __name__ == "__main__":
    sys.exit(main())