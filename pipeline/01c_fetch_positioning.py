"""
pipeline/01c_fetch_positioning.py — Fetch Positioning Data (Hourly)

Empat fitur gratis dari Binance + Bybit API untuk momentum model:
1. Binance: Taker Buy/Sell Volume Ratio (aggressor flow)
2. Binance: Top Trader Long/Short Position Ratio (elite positioning)
3. Binance: Global Long/Short Account Ratio (retail positioning)
4. Bybit:   Open Interest History (total market exposure)

Simpan per koin ke: data/positioning/{coin}_{type}.parquet
Append data baru — tidak overwrite history.

Usage:
  python pipeline/01c_fetch_positioning.py                    # fetch all coins, last 200 bars
  python pipeline/01c_fetch_positioning.py --coins BTCUSDT    # single coin
  python pipeline/01c_fetch_positioning.py --schedule          # run forever, every hour

Setup cron / Task Scheduler:
  Windows: schtasks /create /tn "FetchPositioning" /tr "python pipeline/01c_fetch_positioning.py" /sc HOURLY
"""

import argparse, json, os, sys, time, numpy as np, pandas as pd
from datetime import datetime, timezone
from pathlib import Path
import urllib3, requests
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

ROOT = Path(__file__).parent.parent; sys.path.insert(0, str(ROOT))
from config import TRAINING_COINS

POSITIONING_DIR = ROOT / "data" / "positioning"
POSITIONING_DIR.mkdir(parents=True, exist_ok=True)

BINANCE_BASE = "https://fapi.binance.com"
BYBIT_BASE = "https://api.bybit.com"
SLEEP_BETWEEN = 0.3  # seconds between API calls
MAX_LIMIT = 200


def fetch_binance(endpoint, symbol, limit=200):
    """Fetch data from Binance Futures data endpoint."""
    params = {"symbol": symbol, "period": "1h", "limit": limit}
    try:
        resp = requests.get(f"{BINANCE_BASE}{endpoint}", params=params,
                            timeout=30, verify=False)
        if resp.status_code == 200:
            data = resp.json()
            if isinstance(data, list) and len(data) > 0:
                df = pd.DataFrame(data)
                if "timestamp" in df.columns:
                    df["timestamp"] = pd.to_datetime(
                        df["timestamp"].astype(np.int64), unit="ms", utc=True
                    )
                    df = df.set_index("timestamp").sort_index()
                return df
    except Exception as e:
        print(f"  Binance {endpoint}: {e}")
    return None


def fetch_bybit(endpoint, symbol, params_extra=None):
    """Fetch data from Bybit API."""
    params = {"category": "linear", "symbol": symbol, "intervalTime": "1h", "limit": MAX_LIMIT}
    if params_extra:
        params.update(params_extra)
    try:
        resp = requests.get(f"{BYBIT_BASE}{endpoint}", params=params,
                            timeout=30, verify=False)
        if resp.status_code == 200:
            data = resp.json()
            if data.get("retCode") == 0:
                items = data["result"]["list"]
                if items:
                    df = pd.DataFrame(items)
                    if "timestamp" in df.columns:
                        df["timestamp"] = pd.to_datetime(
                            df["timestamp"].astype(np.int64), unit="ms", utc=True
                        )
                        df = df.set_index("timestamp").sort_index()
                    return df
    except Exception as e:
        print(f"  Bybit {endpoint}: {e}")
    return None


def update_parquet(filepath, new_df):
    """Append new data to existing parquet, deduplicate by index."""
    if filepath.exists():
        existing = pd.read_parquet(filepath)
        combined = pd.concat([existing, new_df])
        combined = combined[~combined.index.duplicated(keep="last")]
        combined = combined.sort_index()
    else:
        combined = new_df
    combined.to_parquet(filepath)
    return len(combined)


def fetch_all_coins(coins):
    """Fetch positioning data for all specified coins."""
    now = datetime.now(timezone.utc)
    print(f"\n{'='*55}")
    print(f"  POSITIONING DATA FETCH | {now.strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"  Coins: {len(coins)}")
    print(f"{'='*55}")

    for coin in coins:
        symbol = coin
        print(f"\n[{coin}]")

        # 1. Binance Taker Buy/Sell Ratio
        df_taker = fetch_binance("/futures/data/takerlongshortRatio", symbol)
        if df_taker is not None:
            # Rename columns
            df_taker = df_taker.rename(columns={
                "buySellRatio": "taker_buy_sell_ratio",
                "buyVol": "taker_buy_vol",
                "sellVol": "taker_sell_vol",
            })
            n = update_parquet(POSITIONING_DIR / f"{coin}_taker_ratio.parquet", df_taker)
            print(f"  Taker Ratio: {len(df_taker)} new -> {n} total")
            time.sleep(SLEEP_BETWEEN)

        # 2. Binance Top Trader Long/Short
        df_top = fetch_binance("/futures/data/topLongShortPositionRatio", symbol)
        if df_top is not None:
            df_top = df_top.rename(columns={
                "longShortRatio": "top_trader_ls_ratio",
                "longAccount": "top_trader_long_pct",
                "shortAccount": "top_trader_short_pct",
            })
            n = update_parquet(POSITIONING_DIR / f"{coin}_top_trader.parquet", df_top)
            print(f"  Top Trader: {len(df_top)} new -> {n} total")
            time.sleep(SLEEP_BETWEEN)

        # 3. Binance Global Long/Short Account Ratio
        df_gls = fetch_binance("/futures/data/globalLongShortAccountRatio", symbol)
        if df_gls is not None:
            df_gls = df_gls.rename(columns={
                "longShortRatio": "global_ls_ratio",
                "longAccount": "global_long_pct",
                "shortAccount": "global_short_pct",
            })
            n = update_parquet(POSITIONING_DIR / f"{coin}_global_ls.parquet", df_gls)
            print(f"  Global L/S: {len(df_gls)} new -> {n} total")
            time.sleep(SLEEP_BETWEEN)

        # 4. Bybit Open Interest
        df_oi = fetch_bybit("/v5/market/open-interest", symbol)
        if df_oi is not None:
            if "openInterest" in df_oi.columns:
                df_oi["openInterest"] = pd.to_numeric(df_oi["openInterest"])
                df_oi = df_oi.rename(columns={"openInterest": "bybit_oi"})
            n = update_parquet(POSITIONING_DIR / f"{coin}_bybit_oi.parquet", df_oi)
            print(f"  Bybit OI: {len(df_oi)} new -> {n} total")
            time.sleep(SLEEP_BETWEEN)

    # Summary
    files = list(POSITIONING_DIR.glob("*.parquet"))
    total_coins = len(set(f.stem.split("_")[0] for f in files))
    print(f"\n{'='*55}")
    print(f"  DONE: {len(files)} files, {total_coins} coins")
    print(f"  Data dir: {POSITIONING_DIR}")
    print(f"{'='*55}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--coins", nargs="+", default=None)
    parser.add_argument("--schedule", action="store_true",
                        help="Run every hour (infinite loop)")
    args = parser.parse_args()

    coins = args.coins or TRAINING_COINS

    if args.schedule:
        print("Running in schedule mode (every hour). Ctrl+C to stop.")
        while True:
            try:
                fetch_all_coins(coins)
                print(f"\nNext fetch in 60 minutes...")
                time.sleep(3600)
            except KeyboardInterrupt:
                print("\nStopped.")
                break
    else:
        fetch_all_coins(coins)


if __name__ == "__main__":
    main()
