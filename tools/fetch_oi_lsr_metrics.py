"""
tools/fetch_oi_lsr_metrics.py — Fetch REAL OI & Long/Short Ratio histori multi-tahun
dari data.binance.vision futures metrics dumps (daily zip, 5-min granularity).

Menggantikan batasan REST 30-hari. Output H1 ke data/metrics/{coin}_oi_lsr_h1.parquet.

Kolom metrics:
  sum_open_interest            -> open_interest (kontrak)
  sum_open_interest_value      -> open_interest_value (USD)
  count_long_short_ratio       -> long_short_ratio (global account ratio, = REST globalLongShortAccountRatio)
  sum_taker_long_short_vol_ratio -> taker_ls_ratio

Usage:
  python tools/fetch_oi_lsr_metrics.py --all
  python tools/fetch_oi_lsr_metrics.py --coins BTCUSDT ETHUSDT
  python tools/fetch_oi_lsr_metrics.py --all --holdout-test   # rentang holdout saja
"""
import argparse
import io
import sys
import urllib.request
import urllib.error
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from config import TRAINING_COINS, ALL_COINS, PROC_DIR, HOLDOUT_DIR, TRAIN_CUTOFF_DATE
from core.utils import setup_logger, ensure_utc_index

logger = setup_logger("fetch_oi_lsr_metrics")

BASE = "https://data.binance.vision/data/futures/um/daily/metrics"
OUT_DIR = ROOT / "data" / "metrics"
OUT_DIR.mkdir(parents=True, exist_ok=True)

USE_COLS = {
    "sum_open_interest":              "open_interest",
    "sum_open_interest_value":        "open_interest_value",
    "count_long_short_ratio":         "long_short_ratio",
    "sum_taker_long_short_vol_ratio": "taker_ls_ratio",
}


def _fetch_day(sym: str, date: str, retries: int = 3) -> pd.DataFrame | None:
    """Download + parse satu hari metrics. None jika 404/gagal."""
    url = f"{BASE}/{sym}/{sym}-metrics-{date}.zip"
    for attempt in range(retries):
        try:
            data = urllib.request.urlopen(url, timeout=30).read()
            z = zipfile.ZipFile(io.BytesIO(data))
            df = pd.read_csv(z.open(z.namelist()[0]))
            return df
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None  # hari sebelum data ada — normal
            if attempt == retries - 1:
                return None
        except Exception:
            if attempt == retries - 1:
                return None
    return None


def fetch_coin(sym: str, start: datetime, end: datetime, workers: int = 16) -> pd.DataFrame:
    """Fetch seluruh rentang [start, end] untuk satu koin, resample ke H1."""
    days = []
    d = start
    while d <= end:
        days.append(d.strftime("%Y-%m-%d"))
        d += timedelta(days=1)

    frames = []
    n_ok = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_fetch_day, sym, day): day for day in days}
        for fut in as_completed(futs):
            df = fut.result()
            if df is not None and not df.empty:
                frames.append(df)
                n_ok += 1

    if not frames:
        logger.warning(f"[{sym}] tidak ada data metrics sama sekali")
        return pd.DataFrame()

    raw = pd.concat(frames, ignore_index=True)
    raw["create_time"] = pd.to_datetime(raw["create_time"], utc=True)
    raw = raw.drop_duplicates("create_time").set_index("create_time").sort_index()

    have = [c for c in USE_COLS if c in raw.columns]
    h1 = raw[have].resample("1h").last()
    h1 = h1.rename(columns={k: v for k, v in USE_COLS.items() if k in have})
    h1 = h1.apply(pd.to_numeric, errors="coerce")

    logger.info(f"[{sym}] {n_ok}/{len(days)} hari OK -> {len(h1)} H1 bars "
                f"({h1.index.min().date()} -> {h1.index.max().date()})")
    return h1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coins", nargs="+", default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--holdout-test", action="store_true", dest="holdout")
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    coins = args.coins if args.coins else (ALL_COINS if args.all else TRAINING_COINS)

    proc_dir = (HOLDOUT_DIR / "processed") if args.holdout else PROC_DIR
    suffix = "_holdout" if args.holdout else ""

    print(f"\n{'='*64}")
    print(f"  FETCH REAL OI/LSR METRICS | {len(coins)} koin | {'HOLDOUT' if args.holdout else 'TRAINING'}")
    print(f"  Sumber: data.binance.vision futures metrics (daily, 5-min)")
    print(f"{'='*64}\n")

    ok, fail = [], []
    for coin in coins:
        clean_p = proc_dir / f"{coin}_clean.parquet"
        if not clean_p.exists():
            logger.warning(f"[{coin}] clean parquet tidak ada — skip")
            fail.append(coin)
            continue
        idx = ensure_utc_index(pd.read_parquet(clean_p, columns=[])).sort_index().index
        start = idx.min().to_pydatetime()
        end   = idx.max().to_pydatetime()
        # metrics mulai ~Okt 2020; clamp start
        start = max(start, datetime(2020, 9, 1, tzinfo=timezone.utc))

        h1 = fetch_coin(coin, start, end, workers=args.workers)
        if h1.empty:
            fail.append(coin)
            continue
        out = OUT_DIR / f"{coin}_oi_lsr_h1{suffix}.parquet"
        h1.to_parquet(out)
        ok.append(coin)

    print(f"\n{'='*64}")
    print(f"  SELESAI: {len(ok)} sukses | {len(fail)} gagal")
    if fail:
        print(f"  Gagal: {fail}")
    print(f"  Output: {OUT_DIR}")
    print(f"{'='*64}")


if __name__ == "__main__":
    main()
