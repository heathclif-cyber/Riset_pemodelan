"""
pipeline/01_fetch.py — Fase 1: Fetch semua data dari Binance + Macro

Jalankan:
  python pipeline/01_fetch.py                          # fetch training coins (ke data/raw/)
  python pipeline/01_fetch.py --all                    # fetch semua 20 koin (ke data/raw/)
  python pipeline/01_fetch.py --all --holdout-test     # fetch holdout-test (ke data/holdout/raw/)
  python pipeline/01_fetch.py --coins SOLUSDT          # koin spesifik (ke data/raw/)
  python pipeline/01_fetch.py --coins SOLUSDT --holdout-test  # koin spesifik ke holdout-test

Periode:
  Training     : TRAIN_START (2020-01-01) -> TRAIN_END (2025-11-01) -> data/raw/
  Holdout-Test : OOS_START (2025-11-01)  -> OOS_END (2026-04-01)   -> data/holdout/raw/

Progress disimpan terpisah:
  Training     : data/raw/.fetch_progress.json
  Holdout-Test : data/holdout/raw/.fetch_progress.json
"""

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

# ── Add project root ke sys.path ──────────────────────────────────────────────
from pipeline._bootstrap import setup_path_from_file
ROOT = setup_path_from_file(__file__)

from config import (
    TRAINING_COINS, NEW_COINS, ALL_COINS,
    TRAIN_START, TRAIN_END, OOS_START, OOS_END,
    KLINE_INTERVALS, KLINE_LIMIT, FUNDING_LIMIT, OI_LIMIT, LONG_SHORT_LIMIT,
    BINANCE_BASE_URL, SLEEP_BETWEEN_REQUESTS,
    SLEEP_ON_RATE_LIMIT, MAX_RETRIES, RETRY_BACKOFF_BASE,
)
from core.binance_client import BinanceClient
from core.fetchers import fetch_coin, fetch_all_macro
from core.utils import setup_logger, load_progress, save_progress
from config import RAW_DIR, HOLDOUT_DIR

logger = setup_logger("01_fetch")

TRAINING_PROGRESS_FILE  = RAW_DIR / ".fetch_progress.json"
HOLDOUT_PROGRESS_FILE   = HOLDOUT_DIR / "raw" / ".fetch_progress.json"

# Alias backward-compat
PROGRESS_FILE = TRAINING_PROGRESS_FILE


def wita_end_to_utc(date_str: str) -> datetime:
    """Akhir kalender WITA (UTC+8) -> batas UTC eksklusif untuk fetch/filter."""
    d = datetime.strptime(date_str, "%Y-%m-%d")
    # 27 Jun 00:00 WITA = 26 Jun 16:00 UTC  =>  akhir 26 Jun WITA
    return datetime(d.year, d.month, d.day, 16, 0, tzinfo=timezone.utc)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fetch data dari Binance",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Contoh:\n"
            "  # Fetch TRAINING (2020-01-01 -> 2025-11-01) ke data/raw/\n"
            "  python pipeline/01_fetch.py --all\n\n"
            "  # Fetch HOLDOUT-TEST (2025-11-01 -> 2026-04-01) ke data/holdout/raw/\n"
            "  python pipeline/01_fetch.py --all --holdout-test\n"
        ),
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--new",   action="store_true",
                       help=f"Fetch new coins: {NEW_COINS}")
    group.add_argument("--all",   action="store_true",
                       help="Fetch semua 20 koin")
    group.add_argument("--coins", nargs="+", metavar="SYMBOL",
                       help="Fetch koin spesifik (contoh: SOLUSDT ETHUSDT)")

    parser.add_argument(
        "--holdout-test", action="store_true", dest="holdout_test",
        help=(
            "Fetch HOLDOUT-TEST data (2025-11-01 -> 2026-04-01) "
            "ke data/holdout/raw/. TERPISAH dari data training."
        ),
    )
    parser.add_argument("--reset", action="store_true",
                        help="Reset progress (fetch ulang dari awal)")
    parser.add_argument(
        "--end-date", metavar="YYYY-MM-DD", default=None,
        help="Override fetch end (WITA calendar day, e.g. 2026-06-26 = hingga akhir 26 Jun WITA)",
    )
    return parser.parse_args()


def _build_coin_schedule(args, is_holdout: bool) -> tuple[list[tuple], str]:
    """
    Kembalikan (coin_schedule, label).
    coin_schedule = [(symbol, start, end), ...]

    Training (is_holdout=False)  : TRAIN_START (2020-01-01) -> TRAIN_END (2025-11-01) -> data/raw/
    Holdout  (is_holdout=True)   : OOS_START   (2025-11-01) -> OOS_END (2026-04-01)   -> data/holdout/raw/

    Koin yang listing setelah 2020 (SUI, TON, PEPE, TAO, ARB) akan otomatis
    mendapat data lebih pendek sesuai tanggal listing mereka di Binance.
    """
    fetch_start = OOS_START   if is_holdout else TRAIN_START
    fetch_end   = OOS_END     if is_holdout else TRAIN_END
    if args.end_date:
        fetch_end = wita_end_to_utc(args.end_date)
    mode_label  = "HOLDOUT-TEST" if is_holdout else "TRAINING"

    if args.new:
        coins = NEW_COINS
        label = f"{mode_label} | NEW COINS ({fetch_start.date()} -> {fetch_end.date()})"
    elif args.all:
        coins = ALL_COINS
        label = f"{mode_label} | ALL COINS ({fetch_start.date()} -> {fetch_end.date()})"
    elif args.coins:
        coins = [c.upper() for c in args.coins]
        label = f"{mode_label} | CUSTOM: {coins} ({fetch_start.date()} -> {fetch_end.date()})"
    else:
        coins = TRAINING_COINS
        label = f"{mode_label} | TRAINING COINS ({fetch_start.date()} -> {fetch_end.date()})"

    schedule = [(sym, fetch_start, fetch_end) for sym in coins]
    return schedule, label


def main():
    args       = parse_args()
    is_holdout = args.holdout_test

    coin_schedule, label = _build_coin_schedule(args, is_holdout)

    # Pilih direktori & progress file berdasarkan mode
    if is_holdout:
        output_dir    = HOLDOUT_DIR / "raw"
        progress_file = HOLDOUT_PROGRESS_FILE
    else:
        output_dir    = RAW_DIR
        progress_file = TRAINING_PROGRESS_FILE

    output_dir.mkdir(parents=True, exist_ok=True)

    # Macro mencakup rentang terluas dari semua koin
    macro_start = min(s for _, s, _ in coin_schedule)
    macro_end   = max(e for _, _, e in coin_schedule)

    sep = "=" * 60
    logger.info(sep)
    logger.info(f"  FETCH DATA: {label}")
    logger.info(f"  Mode    : {'HOLDOUT-TEST' if is_holdout else 'TRAINING'}")
    logger.info(f"  Output  : {output_dir}")
    logger.info(f"  Koin    : {[sym for sym, _, _ in coin_schedule]}")
    logger.info(f"  Macro   : {macro_start.date()} -> {macro_end.date()}")
    logger.info(sep)

    # ── Load/reset progress ────────────────────────────────────────────────────
    if args.reset:
        progress = {}
        logger.info("Progress di-reset.")
    else:
        progress = load_progress(progress_file)
        logger.info(f"Progress loaded: {len(progress)} keys selesai.")

    # ── Init Binance client ────────────────────────────────────────────────────
    client = BinanceClient(
        base_url         = BINANCE_BASE_URL,
        sleep_between    = SLEEP_BETWEEN_REQUESTS,
        sleep_rate_limit = SLEEP_ON_RATE_LIMIT,
        max_retries      = MAX_RETRIES,
        backoff_base     = RETRY_BACKOFF_BASE,
    )

    if not client.test_connection():
        logger.error("Koneksi ke Binance gagal. Cek internet/VPN.")
        sys.exit(1)
    logger.info("Koneksi Binance OK.")

    # ── Fetch macro data ───────────────────────────────────────────────────────
    logger.info("\n--- FETCH MACRO DATA ---")
    fetch_all_macro(macro_start, macro_end, progress=progress)
    save_progress(progress, progress_file)

    # ── Fetch per koin ─────────────────────────────────────────────────────────
    success = []
    failed  = []

    for i, (symbol, start, end) in enumerate(coin_schedule, 1):
        logger.info(f"[{i}/{len(coin_schedule)}] Fetching {symbol} ({start.date()} -> {end.date()})...")
        try:
            results = fetch_coin(
                client           = client,
                symbol           = symbol,
                start            = start,
                end              = end,
                intervals        = KLINE_INTERVALS,
                progress         = progress,
                raw_dir          = output_dir,   # <-- training: data/raw/, holdout: data/holdout/raw/
                kline_limit      = KLINE_LIMIT,
                funding_limit    = FUNDING_LIMIT,
                oi_limit         = OI_LIMIT,
                long_short_limit = LONG_SHORT_LIMIT,
            )
            if results:
                success.append(symbol)
            else:
                failed.append(symbol)
        except Exception as e:
            logger.exception(f"[{symbol}] Error: {e}")
            failed.append(symbol)
        finally:
            save_progress(progress, progress_file)

    # ── Ringkasan ──────────────────────────────────────────────────────────────
    print(f"\n{sep}")
    print(f"  FETCH SELESAI — {'HOLDOUT-TEST' if is_holdout else 'TRAINING'}")
    print(sep)
    print(f"  Berhasil : {len(success)} koin — {success}")
    print(f"  Gagal    : {len(failed)}  koin — {failed}")
    print(f"  Output   : {output_dir}")
    print(f"  Progress : {progress_file}")
    print(f"{sep}\n")


if __name__ == "__main__":
    main()
