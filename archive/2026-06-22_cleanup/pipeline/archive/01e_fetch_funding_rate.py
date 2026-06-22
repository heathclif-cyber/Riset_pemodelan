#!/usr/bin/env python3
"""
pipeline/01e_fetch_funding_rate.py
Targeted re-fetch funding rate (8H) untuk semua koin, 2020-01-01 -> TRAIN_CUTOFF_DATE.

Root cause: raw parquet di data/training/funding_rate/ hanya punya data Apr 2026+
(holdout period) sehingga 20 dari 21 koin punya 100% NaN di features.

Usage:
  python pipeline/01e_fetch_funding_rate.py --all
  python pipeline/01e_fetch_funding_rate.py --coins BTCUSDT ETHUSDT
"""

import sys, json, time, argparse
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from config import (
    TRAINING_COINS, TRAIN_CUTOFF_DATE, TRAIN_START,
    BINANCE_BASE_URL, SLEEP_BETWEEN_REQUESTS,
    FUNDING_LIMIT,
)
from core.utils import setup_logger
from core.binance_client import BinanceClient
from core.fetchers import fetch_funding_rate

logger = setup_logger("01e_fetch_funding_rate")

FUNDING_DIR  = ROOT / "data" / "training" / "funding_rate"
PROGRESS_FILE = ROOT / "data" / "training" / ".fetch_progress.json"


def load_progress():
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE) as f:
            return json.load(f)
    return {}


def save_progress(prog):
    with open(PROGRESS_FILE, "w") as f:
        json.dump(prog, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--coins", nargs="+", default=None)
    parser.add_argument("--force", action="store_true",
                        help="Force re-fetch even if marked done in progress")
    args = parser.parse_args()

    coins = args.coins or (TRAINING_COINS if args.all else TRAINING_COINS[:3])

    print("=" * 60)
    print(f"  Funding Rate Fetch | {len(coins)} coins")
    print(f"  Period: {TRAIN_START.date()} -> {TRAIN_CUTOFF_DATE.date()}")
    print(f"  Output: {FUNDING_DIR}")
    print("=" * 60)

    FUNDING_DIR.mkdir(parents=True, exist_ok=True)

    prog = load_progress()
    client = BinanceClient()

    success, failed = [], []

    for coin in coins:
        prog_key = f"funding_rate_{coin}"

        # Force-clear progress so fetch_funding_rate won't skip
        if args.force or prog_key in prog:
            if prog_key in prog:
                print(f"  {coin}: clearing progress (was marked done)")
                del prog[prog_key]
                save_progress(prog)

        # Delete stale raw file if it only has recent data
        raw_path = FUNDING_DIR / f"{coin}_8h.parquet"
        if raw_path.exists():
            try:
                import pandas as pd
                existing = pd.read_parquet(raw_path)
                if len(existing) < 500:
                    print(f"  {coin}: deleting stale parquet ({len(existing)} rows, likely holdout-only)")
                    raw_path.unlink()
            except Exception as e:
                logger.warning(f"{coin}: could not read existing parquet: {e}")

        print(f"\n  [{coins.index(coin)+1}/{len(coins)}] {coin} fetching ...", flush=True)

        df = fetch_funding_rate(
            client=client,
            symbol=coin,
            start=TRAIN_START,
            end=TRAIN_CUTOFF_DATE,
            progress=prog,
            funding_limit=FUNDING_LIMIT,
            raw_dir=FUNDING_DIR,
        )

        if df is not None and len(df) > 0:
            print(f"  {coin}: OK — {len(df):,} rows  "
                  f"({df.index[0].date()} to {df.index[-1].date()})")
            success.append(coin)
        else:
            print(f"  {coin}: FAILED — no data returned")
            failed.append(coin)

        save_progress(prog)
        time.sleep(0.5)

    print(f"\n{'='*60}")
    print(f"  Done: {len(success)} OK, {len(failed)} failed")
    if failed:
        print(f"  Failed: {failed}")
    print(f"\nNext steps:")
    print(f"  python pipeline/02_clean.py --all")
    print(f"  python pipeline/03_engineer.py --all")
    print(f"  python pipeline/06c_feature_selection_shap.py")


if __name__ == "__main__":
    main()
