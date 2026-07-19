"""
Backfill historis OI + L/S dari data.binance.vision (daily metrics ZIP).

Output:
  data/positioning_hist/{coin}_metrics.parquet  (daily)
  data/training/open_interest/{coin}_1h.parquet (H1 ffill dari daily)
  data/training/long_short_ratio/{coin}_1h.parquet
  data/positioning/{coin}_binance_oi.parquet, {coin}_global_ls.parquet, {coin}_top_trader.parquet

Usage:
  python tools/data/backfill_binance_vision_metrics.py --coins BTCUSDT
  python tools/data/backfill_binance_vision_metrics.py --all
"""
from __future__ import annotations

import argparse
import io
import sys
import time
import urllib.error
import urllib.request
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from pipeline._bootstrap import setup_path_from_file

ROOT = setup_path_from_file(__file__)
from config import TRAINING_COINS, RAW_DIR
from core.utils import setup_logger

logger = setup_logger("backfill_vision_metrics")

BASE_URL = "https://data.binance.vision/data/futures/um/daily/metrics"
HIST_DIR = ROOT / "data" / "positioning_hist"
POSITIONING_DIR = ROOT / "data" / "positioning"
OI_DIR = RAW_DIR / "open_interest"
LS_DIR = RAW_DIR / "long_short_ratio"

COIN_START = {
    "BTCUSDT": datetime(2021, 1, 1),
    "ETHUSDT": datetime(2022, 1, 1),
    "SOLUSDT": datetime(2022, 1, 1),
    "BNBUSDT": datetime(2022, 1, 1),
    "XRPUSDT": datetime(2022, 1, 1),
    "DOGEUSDT": datetime(2022, 1, 1),
    "ADAUSDT": datetime(2022, 1, 1),
    "TRXUSDT": datetime(2022, 1, 1),
    "1000SHIBUSDT": datetime(2022, 1, 1),
    "AVAXUSDT": datetime(2022, 1, 1),
    "LINKUSDT": datetime(2022, 1, 1),
    "DOTUSDT": datetime(2022, 1, 1),
    "NEARUSDT": datetime(2022, 1, 1),
    "HBARUSDT": datetime(2022, 1, 1),
    "SUIUSDT": datetime(2024, 1, 1),
    "1000PEPEUSDT": datetime(2024, 1, 1),
    "ARBUSDT": datetime(2024, 1, 1),
    "TONUSDT": datetime(2025, 1, 1),
    "POLUSDT": datetime(2025, 1, 1),
    "TAOUSDT": datetime(2025, 1, 1),
    "ONDOUSDT": datetime(2025, 1, 1),
}


def fetch_day(symbol: str, dt: datetime) -> pd.DataFrame | None:
    date_str = dt.strftime("%Y-%m-%d")
    url = f"{BASE_URL}/{symbol}/{symbol}-metrics-{date_str}.zip"
    try:
        with urllib.request.urlopen(url, timeout=15) as r:
            data = r.read()
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            with z.open(z.namelist()[0]) as f:
                return pd.read_csv(f)
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None
        logger.warning(f"HTTP {e.code} {symbol} {date_str}")
        return None
    except Exception as e:
        logger.warning(f"Error {symbol} {date_str}: {e}")
        return None


def normalize_daily(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    col_map = {
        "create_time": "date",
        "sum_open_interest": "oi_base",
        "sum_open_interest_value": "oi_usd",
        "sum_toptrader_long_short_ratio": "toptrader_ls_ratio",
        "count_toptrader_long_short_ratio": "toptrader_ls_acct",
        "count_long_short_ratio": "global_ls_ratio",
        "sum_taker_long_short_vol_ratio": "taker_ls_vol_ratio",
    }
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
    if "date" not in df.columns:
        return pd.DataFrame()
    df["date"] = pd.to_datetime(df["date"], utc=True).dt.normalize()
    df["symbol"] = symbol
    return df.sort_values("date").drop_duplicates("date")


def daily_to_h1(daily: pd.DataFrame, symbol: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Expand daily metrics ke grid H1 dari klines (ffill)."""
    kpath = RAW_DIR / "klines" / symbol / "1h_all.parquet"
    if not kpath.exists():
        logger.warning(f"[{symbol}] klines tidak ada — skip H1 expand")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    h1_idx = pd.read_parquet(kpath, columns=["close"]).index
    h1_idx = pd.DatetimeIndex(h1_idx, tz="UTC")

    d = daily.set_index("date").sort_index()
    d = d.reindex(d.index.union(h1_idx)).sort_index().ffill().reindex(h1_idx)

    oi = pd.DataFrame({"open_interest": d["oi_base"].astype(float)}, index=h1_idx) if "oi_base" in d else pd.DataFrame()
    ls = pd.DataFrame({"long_short_ratio": d["global_ls_ratio"].astype(float)}, index=h1_idx) if "global_ls_ratio" in d else pd.DataFrame()
    tt = pd.DataFrame({"top_trader_ls_ratio": d["toptrader_ls_ratio"].astype(float)}, index=h1_idx) if "toptrader_ls_ratio" in d else pd.DataFrame()
    return oi, ls, tt


def save_df(path: Path, df: pd.DataFrame) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and len(df) > 0:
        old = pd.read_parquet(path)
        df = pd.concat([old, df])
        df = df[~df.index.duplicated(keep="last")].sort_index()
    if len(df) == 0:
        return 0
    df.to_parquet(path)
    return len(df)


def backfill_coin(symbol: str, end: datetime) -> dict:
    start = COIN_START.get(symbol, datetime(2022, 1, 1))
    rows = []
    current = start
    fetched = missing = 0
    logger.info(f"[{symbol}] vision {start.date()} -> {end.date()}")

    while current <= end:
        raw = fetch_day(symbol, current)
        if raw is not None and len(raw) > 0:
            rows.append(raw)
            fetched += 1
        else:
            missing += 1
        current += timedelta(days=1)
        if fetched % 200 == 0 and fetched > 0:
            time.sleep(0.1)

    if not rows:
        return {"rows": 0, "error": "no data"}

    daily = normalize_daily(pd.concat(rows, ignore_index=True), symbol)
    HIST_DIR.mkdir(parents=True, exist_ok=True)
    daily.to_parquet(HIST_DIR / f"{symbol}_metrics.parquet", index=False)

    oi_h1, ls_h1, tt_h1 = daily_to_h1(daily, symbol)
    stats = {"daily_rows": len(daily), "fetched": fetched, "missing": missing,
             "start": str(daily["date"].min().date()), "end": str(daily["date"].max().date())}

    if not oi_h1.empty:
        stats["oi_h1"] = save_df(OI_DIR / f"{symbol}_1h.parquet", oi_h1)
        save_df(POSITIONING_DIR / f"{symbol}_binance_oi.parquet", oi_h1)
    if not ls_h1.empty:
        stats["ls_h1"] = save_df(LS_DIR / f"{symbol}_1h.parquet", ls_h1)
        gls = ls_h1.rename(columns={"long_short_ratio": "global_ls_ratio"})
        save_df(POSITIONING_DIR / f"{symbol}_global_ls.parquet", gls)
    if not tt_h1.empty:
        stats["top_trader_h1"] = save_df(POSITIONING_DIR / f"{symbol}_top_trader.parquet", tt_h1)

    logger.info(f"[{symbol}] done: {stats}")
    return stats


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--coins", nargs="+", default=None)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--end", default=None, help="YYYY-MM-DD")
    args = parser.parse_args()

    coins = TRAINING_COINS if args.all or not args.coins else args.coins
    if args.end:
        end = datetime.fromisoformat(args.end)
    else:
        end = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    if end.tzinfo is not None:
        end = end.replace(tzinfo=None)

    summary = {}
    for coin in coins:
        try:
            summary[coin] = backfill_coin(coin, end)
        except Exception as e:
            logger.error(f"[{coin}] {e}", exc_info=True)
            summary[coin] = {"error": str(e)}

    print("\n=== VISION BACKFILL SUMMARY ===")
    for coin, st in summary.items():
        print(f"  {coin}: {st}")
    return 0


if __name__ == "__main__":
    sys.exit(main())