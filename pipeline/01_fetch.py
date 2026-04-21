"""
pipeline/01_fetch.py — Fase 1: Fetch semua data dari Binance + Macro

Jalankan:
  python pipeline/01_fetch.py                    # fetch training coins (5 koin)
  python pipeline/01_fetch.py --new              # fetch new coins (15 koin)
  python pipeline/01_fetch.py --all              # fetch semua 20 koin
  python pipeline/01_fetch.py --coins SOLUSDT ETHUSDT  # koin spesifik

Progress disimpan di data/raw/.fetch_progress.json (resume-capable).
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

# ── Add project root ke sys.path ─────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from config import (
    TRAINING_COINS, NEW_COINS, ALL_COINS,
    TRAIN_START, TRAIN_END,
    NEW_COINS_START, NEW_COINS_END,
    KLINE_INTERVALS, KLINE_LIMIT, FUNDING_LIMIT,
    BINANCE_BASE_URL, SLEEP_BETWEEN_REQUESTS,
    SLEEP_ON_RATE_LIMIT, MAX_RETRIES, RETRY_BACKOFF_BASE,
)
from core.binance_client import BinanceClient
from core.fetchers import fetch_coin, fetch_all_macro
from core.utils import setup_logger, load_progress, save_progress
from config import RAW_DIR

logger = setup_logger("01_fetch")

PROGRESS_FILE = RAW_DIR / ".fetch_progress.json"


def parse_args():
    parser = argparse.ArgumentParser(description="Fetch data dari Binance")
    group  = parser.add_mutually_exclusive_group()
    group.add_argument("--training", action="store_true",
                       help=f"Fetch training coins: {TRAINING_COINS}")
    group.add_argument("--new",  action="store_true",
                       help=f"Fetch new coins: {NEW_COINS}")
    group.add_argument("--all",  action="store_true",
                       help="Fetch semua 20 koin")
    group.add_argument("--coins", nargs="+", metavar="SYMBOL",
                       help="Fetch koin spesifik")
    parser.add_argument("--reset", action="store_true",
                        help="Reset progress (fetch ulang dari awal)")
    return parser.parse_args()


def main():
    args = parse_args()

    # ── Tentukan koin dan periode ─────────────────────────────────────────────
    if args.new:
        coins = NEW_COINS
        start = NEW_COINS_START
        end   = NEW_COINS_END
        label = "NEW COINS (Apr 2023 → Apr 2025)"
    elif args.all:
        coins = ALL_COINS
        start = NEW_COINS_START   # gunakan periode yang lebih pendek untuk semua
        end   = NEW_COINS_END
        label = "ALL COINS (Apr 2023 → Apr 2025)"
    elif args.coins:
        coins = [c.upper() for c in args.coins]
        start = NEW_COINS_START
        end   = NEW_COINS_END
        label = f"CUSTOM: {coins}"
    else:
        # Default: training coins dengan periode penuh
        coins = TRAINING_COINS
        start = TRAIN_START
        end   = TRAIN_END
        label = "TRAINING COINS (Jan 2022 → Apr 2025)"

    logger.info("=" * 60)
    logger.info(f"  FETCH DATA: {label}")
    logger.info(f"  Koin: {coins}")
    logger.info(f"  Periode: {start.date()} → {end.date()}")
    logger.info("=" * 60)

    # ── Load/reset progress ────────────────────────────────────────────────────
    if args.reset:
        progress = {}
        logger.info("Progress di-reset.")
    else:
        progress = load_progress(PROGRESS_FILE)
        logger.info(f"Progress loaded: {len(progress)} keys selesai.")

    # ── Init Binance client ───────────────────────────────────────────────────
    client = BinanceClient(
        base_url        = BINANCE_BASE_URL,
        sleep_between   = SLEEP_BETWEEN_REQUESTS,
        sleep_rate_limit= SLEEP_ON_RATE_LIMIT,
        max_retries     = MAX_RETRIES,
        backoff_base    = RETRY_BACKOFF_BASE,
    )

    if not client.test_connection():
        logger.error("Koneksi ke Binance gagal. Cek internet/VPN.")
        sys.exit(1)
    logger.info("Koneksi Binance OK.")

    # ── Fetch macro data (sekali saja) ────────────────────────────────────────
    logger.info("\n--- FETCH MACRO DATA ---")
    fetch_all_macro(start, end, progress=progress)
    save_progress(progress, PROGRESS_FILE)

    # ── Fetch per koin ────────────────────────────────────────────────────────
    success = []
    failed  = []

    for i, symbol in enumerate(coins, 1):
        logger.info(f"\n[{i}/{len(coins)}] Fetching {symbol}...")
        try:
            results = fetch_coin(
                client   = client,
                symbol   = symbol,
                start    = start,
                end      = end,
                intervals= KLINE_INTERVALS,
                progress = progress,
                kline_limit   = KLINE_LIMIT,
                funding_limit = FUNDING_LIMIT,
            )
            if results:
                success.append(symbol)
            else:
                failed.append(symbol)
        except Exception as e:
            logger.exception(f"[{symbol}] Error: {e}")
            failed.append(symbol)
        finally:
            save_progress(progress, PROGRESS_FILE)

    # ── Ringkasan ─────────────────────────────────────────────────────────────
    sep = "=" * 60
    print(f"\n{sep}")
    print(f"  FETCH SELESAI")
    print(f"{sep}")
    print(f"  Berhasil : {len(success)} koin — {success}")
    print(f"  Gagal    : {len(failed)}  koin — {failed}")
    print(f"  Progress : {PROGRESS_FILE}")
    print(f"{sep}\n")


if __name__ == "__main__":
    main()
