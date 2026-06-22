"""
Fetch M15 klines untuk riset entry confirmation (tidak masuk pipeline utama).

Protokol ketat (METHODOLOGY.md):
  --mode oof      : 2020-08-26 -> 2025-10-31 (selaras genuine OOF + labeled training)
  --mode holdout  : OOS_START -> OOS_END (amplop tersegel, fetch terpisah)
  --mode all      : keduanya

Output:
  data/research/m15/klines/{SYMBOL}_15m.parquet         (OOF / training)
  data/research/m15/klines/{SYMBOL}_15m_holdout.parquet (holdout)

Jalankan:
  python tools/fetch_m15_research.py --mode oof --force
  python tools/fetch_m15_research.py --mode holdout
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import ALL_COINS, OOS_START, OOS_END, KLINE_LIMIT
from core.binance_client import BinanceClient
from core.fetchers import fetch_klines
from core.utils import setup_logger, save_df

logger = setup_logger("fetch_m15_research")

M15_DIR = ROOT / "data" / "research" / "m15" / "klines"

# First bar with has_oof=True in ic32 oof_predictions.parquet (purged CV start)
OOF_M15_START = datetime(2020, 8, 26, 11, 0, tzinfo=timezone.utc)
# Labeled training features end (must match LABEL_DIR parquets)
OOF_M15_END = datetime(2025, 10, 31, 23, 0, tzinfo=timezone.utc)


def parse_args():
    p = argparse.ArgumentParser(description="Fetch M15 klines — protokol genuine OOF")
    p.add_argument("--coins", nargs="+", default=None, help="Default: ALL_COINS")
    p.add_argument("--mode", choices=("oof", "holdout", "all"), default="oof")
    p.add_argument("--force", action="store_true", help="Re-fetch meski file ada")
    return p.parse_args()


def _fetch_one(client, sym: str, start: datetime, end: datetime, out: Path, force: bool) -> str:
    if out.exists() and not force:
        logger.info(f"[{sym}] skip — {out.name} ada")
        return "skip"
    df = fetch_klines(client, sym, "15m", start, end, kline_limit=KLINE_LIMIT, raw_dir=None)
    if df is None or df.empty:
        logger.error(f"[{sym}] fetch gagal {start.date()} -> {end.date()}")
        return "fail"
    save_df(df, out, logger)
    logger.info(f"[{sym}] saved {len(df)} bars -> {out}")
    return "ok"


def main():
    args = parse_args()
    coins = args.coins or ALL_COINS
    M15_DIR.mkdir(parents=True, exist_ok=True)
    client = BinanceClient()
    counts = {"ok": 0, "skip": 0, "fail": 0}

    for sym in coins:
        if args.mode in ("oof", "all"):
            out = M15_DIR / f"{sym}_15m.parquet"
            st = _fetch_one(client, sym, OOF_M15_START, OOF_M15_END, out, args.force)
            counts[st] = counts.get(st, 0) + 1
        if args.mode in ("holdout", "all"):
            out_h = M15_DIR / f"{sym}_15m_holdout.parquet"
            st = _fetch_one(client, sym, OOS_START, OOS_END, out_h, args.force)
            counts[st] = counts.get(st, 0) + 1

    print(f"Done mode={args.mode}: {counts} dir={M15_DIR}")


if __name__ == "__main__":
    main()