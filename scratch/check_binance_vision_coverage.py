"""
scratch/check_binance_vision_coverage.py
Verifikasi ketersediaan data historis di data.binance.vision
untuk futures metrics (OI, top-trader L/S, global L/S, taker ratio).

URL pattern: https://data.binance.vision/data/futures/um/daily/metrics/{symbol}/
File: {symbol}-metrics-{date}.zip

Output: tabel coverage per koin (earliest date, total days, missing %)
"""
import sys, urllib.request, urllib.error, json
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from config import TRAINING_COINS

BASE_URL = "https://data.binance.vision/data/futures/um/daily/metrics"

# Rentang cek — semua dari 2020-01-01
START = datetime(2020, 1, 1)
END   = datetime(2026, 6, 11)

PROBE_DATES = [
    datetime(2020, 1, 15),
    datetime(2020, 6, 1),
    datetime(2021, 1, 1),
    datetime(2021, 6, 1),
    datetime(2022, 1, 1),
    datetime(2022, 6, 1),
    datetime(2023, 1, 1),
    datetime(2024, 1, 1),
    datetime(2025, 1, 1),
    datetime(2025, 6, 1),
    datetime(2026, 1, 1),
    datetime(2026, 6, 1),
]

def check_date(symbol: str, dt: datetime) -> bool:
    date_str = dt.strftime("%Y-%m-%d")
    url = f"{BASE_URL}/{symbol}/{symbol}-metrics-{date_str}.zip"
    try:
        req = urllib.request.Request(url, method="HEAD")
        urllib.request.urlopen(req, timeout=8)
        return True
    except (urllib.error.HTTPError, urllib.error.URLError):
        return False

def find_earliest(symbol: str) -> str | None:
    """Binary search untuk earliest available date."""
    # Cek apakah ada di 2021-01-01 dulu
    if not check_date(symbol, datetime(2021, 1, 1)):
        # Coba lebih lambat
        for year in [2021, 2022, 2023]:
            for month in [1, 4, 7, 10]:
                dt = datetime(year, month, 1)
                if check_date(symbol, dt):
                    # Binary search backward dari sini
                    lo = START
                    hi = dt
                    while (hi - lo).days > 7:
                        mid = lo + (hi - lo) / 2
                        if check_date(symbol, mid):
                            hi = mid
                        else:
                            lo = mid
                    return hi.strftime("%Y-%m-%d")
        return None
    else:
        # Cek apakah 2020-01-15 sudah ada
        if check_date(symbol, datetime(2020, 1, 15)):
            return "2020-01-15 (atau lebih awal)"
        # Binary search 2020-01-15 to 2021-01-01
        lo = datetime(2020, 1, 15)
        hi = datetime(2021, 1, 1)
        while (hi - lo).days > 7:
            mid = lo + (hi - lo) / 2
            if check_date(symbol, mid):
                hi = mid
            else:
                lo = mid
        return hi.strftime("%Y-%m-%d")

def count_available(symbol: str) -> int:
    """Count berapa probe_dates yang tersedia."""
    return sum(1 for dt in PROBE_DATES if check_date(symbol, dt))

print("=" * 60)
print(f"  Binance Vision Metrics Coverage — {len(TRAINING_COINS)} koin")
print("=" * 60)
print(f"{'Koin':<20} {'Probe OK':>10} {'Earliest Est.':>20}")
print("-" * 60)

results = {}
for coin in TRAINING_COINS:
    n_ok = count_available(coin)
    if n_ok == 0:
        earliest = "TIDAK ADA"
    elif n_ok == len(PROBE_DATES):
        earliest = "<= 2020-01-15"
    else:
        # Cek secara kasar
        earliest = "partial"
        for dt in PROBE_DATES:
            if check_date(coin, dt):
                earliest = f">= {dt.strftime('%Y-%m-%d')}"
                break

    results[coin] = {"n_probe": n_ok, "earliest_est": earliest}
    status = "OK" if n_ok >= 8 else ("PARTIAL" if n_ok > 0 else "MISSING")
    print(f"{coin:<20} {n_ok:>5}/{len(PROBE_DATES):<4} {earliest:>20}  [{status}]")

print("=" * 60)

# Sample satu file BTCUSDT untuk lihat kolom
print("\nSample kolom dari BTC metrics 2024-01-01:")
import io, zipfile
try:
    url = f"{BASE_URL}/BTCUSDT/BTCUSDT-metrics-2024-01-01.zip"
    with urllib.request.urlopen(url, timeout=10) as r:
        data = r.read()
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        fname = z.namelist()[0]
        with z.open(fname) as f:
            header = f.readline().decode().strip()
            sample = f.readline().decode().strip()
    print(f"  Header : {header}")
    print(f"  Sample : {sample[:120]}...")
except Exception as e:
    print(f"  ERROR: {e}")
