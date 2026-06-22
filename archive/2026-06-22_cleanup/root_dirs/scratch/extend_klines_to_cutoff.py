# -*- coding: utf-8 -*-
"""Extend stale training klines to TRAIN_CUTOFF_DATE (incremental append)."""
from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import ALL_COINS, KLINE_LIMIT, TRAIN_CUTOFF_DATE, TRAINING_DIR
from core.binance_client import BinanceClient
from core.fetchers import _parse_klines
from core.utils import (
    chunk_time_range,
    ensure_utc_index,
    get_filepath,
    interval_to_ms,
    load_df,
    save_df,
    setup_logger,
    to_ms,
    validate_ohlcv,
)

logger = setup_logger("extend_klines")
INTERVALS = ("1h", "4h")


def _extend_one(client: BinanceClient, symbol: str, interval: str) -> dict:
    path = get_filepath("klines", symbol, interval, base_dir=TRAINING_DIR)
    existing = load_df(path, logger)
    if existing is None or existing.empty:
        return {"status": "missing", "path": str(path)}

    existing = ensure_utc_index(existing).sort_index()
    last_ts = existing.index.max()
    target_end = TRAIN_CUTOFF_DATE - timedelta(hours=1)
    if last_ts >= target_end:
        return {"status": "ok", "rows": len(existing), "end": str(last_ts), "added": 0}

    start = last_ts + timedelta(milliseconds=interval_to_ms(interval))
    start_ms = to_ms(start)
    end_ms = to_ms(TRAIN_CUTOFF_DATE)

    logger.info(
        f"[{symbol}] {interval}: extend {start.date()} -> {TRAIN_CUTOFF_DATE.date()} "
        f"(existing end {last_ts.date()})"
    )

    frames = []
    for chunk_start, chunk_end in chunk_time_range(start_ms, end_ms, interval, KLINE_LIMIT):
        raw = client.get_klines(
            symbol=symbol,
            interval=interval,
            start_time_ms=chunk_start,
            end_time_ms=chunk_end - 1,
            limit=KLINE_LIMIT,
        )
        if not raw:
            continue
        frames.append(_parse_klines(raw))

    if not frames:
        return {"status": "no_new_data", "end": str(last_ts), "added": 0}

    import pandas as pd

    new_df = pd.concat(frames)
    merged = pd.concat([existing, new_df])
    merged = merged[~merged.index.duplicated(keep="first")].sort_index()
    validate_ohlcv(merged, symbol, interval, logger)
    save_df(merged, path, logger)

    added = len(merged) - len(existing)
    return {
        "status": "extended",
        "rows": len(merged),
        "end": str(merged.index.max()),
        "added": added,
    }


def main():
    client = BinanceClient()
    if not client.test_connection():
        logger.error("Binance connection failed")
        sys.exit(1)

    summary = {}
    for symbol in ALL_COINS:
        summary[symbol] = {}
        for interval in INTERVALS:
            try:
                summary[symbol][interval] = _extend_one(client, symbol, interval)
            except Exception as exc:
                summary[symbol][interval] = {"status": "error", "error": str(exc)}
                logger.error(f"[{symbol}] {interval}: {exc}")

    extended = sum(
        1 for sym in summary.values()
        for iv in sym.values()
        if iv.get("status") == "extended"
    )
    logger.info(f"Done — {extended} interval(s) extended across {len(ALL_COINS)} coins")


if __name__ == "__main__":
    main()