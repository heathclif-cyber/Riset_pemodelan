"""
pipeline/01c_fetch_positioning.py — Fetch Positioning + Macro Data

=== PER-COIN (hourly): Binance + Bybit positioning ===
1. Binance: Taker Buy/Sell Volume Ratio (aggressor flow)
2. Binance: Top Trader Long/Short Position Ratio (elite positioning)
3. Binance: Global Long/Short Account Ratio (retail positioning)
4. Bybit:   Open Interest History (total market exposure)

=== GLOBAL MACRO (daily): CoinGecko + Fear&Greed + ETF Flow ===
5. CoinGecko Global: BTC dominance, USDT market cap, total market cap
6. Alternative.me: Fear & Greed Index
7. Dune Analytics: BTC/ETH ETF netflow (needs DUNE_API_KEY env var)

Simpan ke:
  Per-coin: data/positioning/{coin}_{type}.parquet
  Macro:    data/macro/{source}.parquet

Usage:
  python pipeline/01c_fetch_positioning.py                     # fetch all coins + macro
  python pipeline/01c_fetch_positioning.py --coins BTCUSDT     # single coin
  python pipeline/01c_fetch_positioning.py --macro-only         # only macro data
  python pipeline/01c_fetch_positioning.py --schedule           # hourly per-coin, daily macro
"""

import argparse, json, os, sys, time, numpy as np, pandas as pd
from datetime import datetime, timezone, timedelta
from pathlib import Path
import urllib3, requests
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

ROOT = Path(__file__).parent.parent; sys.path.insert(0, str(ROOT))
from config import TRAINING_COINS

POSITIONING_DIR = ROOT / "data" / "positioning"
POSITIONING_DIR.mkdir(parents=True, exist_ok=True)
MACRO_DIR = ROOT / "data" / "macro"
MACRO_DIR.mkdir(parents=True, exist_ok=True)

BINANCE_BASE = "https://fapi.binance.com"
BYBIT_BASE = "https://api.bybit.com"
COINGECKO_BASE = "https://api.coingecko.com/api/v3"
FNG_BASE = "https://api.alternative.me"
SLEEP_BETWEEN = 0.3  # seconds between API calls
MAX_LIMIT = 200
LAST_MACRO_DATE_FILE = MACRO_DIR / "_last_macro_fetch.txt"


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


# ─── Global Macro Data Fetchers ──────────────────────────────────────────────

def _should_fetch_macro():
    """Return True if macro data should be fetched today (once daily)."""
    if not LAST_MACRO_DATE_FILE.exists():
        return True
    try:
        last_date = LAST_MACRO_DATE_FILE.read_text().strip()
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        return last_date != today
    except Exception:
        return True


def _mark_macro_fetched():
    LAST_MACRO_DATE_FILE.write_text(datetime.now(timezone.utc).strftime("%Y-%m-%d"))


def fetch_coingecko_global():
    """Fetch BTC dominance + total market cap + USDT volume from CoinGecko."""
    try:
        resp = requests.get(f"{COINGECKO_BASE}/global", timeout=15)
        if resp.status_code == 200:
            data = resp.json()["data"]
            now = datetime.now(timezone.utc)
            row = {
                "timestamp": now,
                "btc_dominance": float(data["market_cap_percentage"]["btc"]),
                "eth_dominance": float(data["market_cap_percentage"]["eth"]),
                "total_market_cap_usd": float(data["total_market_cap"]["usd"]),
                "total_volume_24h_usd": float(data["total_volume"]["usd"]),
                "active_cryptos": int(data["active_cryptocurrencies"]),
            }
            new_df = pd.DataFrame([row]).set_index("timestamp")
            n = update_parquet(MACRO_DIR / "coingecko_global.parquet", new_df)
            print(f"  CoinGecko Global: BTC dom={row['btc_dominance']:.1f}% | MCap=${row['total_market_cap_usd']/1e12:.2f}T | -> {n} total")
            return True
    except Exception as e:
        print(f"  CoinGecko Global FAIL: {e}")
    return False


def fetch_fear_greed():
    """Fetch Fear & Greed Index from alternative.me (daily)."""
    try:
        resp = requests.get(f"{FNG_BASE}/fng/?limit=2", timeout=10)
        if resp.status_code == 200:
            items = resp.json()["data"]
            rows = []
            for item in items:
                ts = datetime.fromtimestamp(int(item["timestamp"]), tz=timezone.utc)
                rows.append({
                    "timestamp": ts,
                    "fear_greed_value": int(item["value"]),
                    "fear_greed_class": item["value_classification"],
                })
            new_df = pd.DataFrame(rows).set_index("timestamp").sort_index()
            n = update_parquet(MACRO_DIR / "fear_greed.parquet", new_df)
            latest = rows[0]
            print(f"  Fear&Greed: {latest['fear_greed_value']} ({latest['fear_greed_class']}) | -> {n} total")
            return True
    except Exception as e:
        print(f"  Fear&Greed FAIL: {e}")
    return False


def fetch_etf_flow_dune():
    """
    Fetch BTC/ETH Spot ETF netflow from Dune Analytics API.
    Requires DUNE_API_KEY environment variable.
    Free tier: 1,000 queries/month — enough for daily fetch.

    Query: combined BTC+ETH ETF daily netflow (USD).
    """
    dune_key = os.environ.get("DUNE_API_KEY", "")
    if not dune_key:
        print("  ETF Flow: SKIP (DUNE_API_KEY not set)")
        return False

    # Dune query ID for "BTC ETF Daily Netflow"
    # Using community query: https://dune.com/queries/3802960 (BTC spot ETF flow)
    QUERY_ID = 3802960
    url = f"https://api.dune.com/api/v1/query/{QUERY_ID}/results"

    try:
        resp = requests.get(url, headers={"X-Dune-API-Key": dune_key}, timeout=30)
        if resp.status_code == 200:
            data = resp.json()
            rows = data.get("result", {}).get("rows", [])
            if rows:
                df = pd.DataFrame(rows)
                if "date" in df.columns or "day" in df.columns:
                    date_col = "date" if "date" in df.columns else "day"
                    df[date_col] = pd.to_datetime(df[date_col], utc=True)
                    df = df.set_index(date_col).sort_index()
                    n = update_parquet(MACRO_DIR / "etf_flow.parquet", df)
                    latest = df.iloc[-1]
                    print(f"  ETF Flow: {len(rows)} rows latest={latest.name.date()} | -> {n} total")
                    return True
        elif resp.status_code == 401:
            print(f"  ETF Flow: DUNE_API_KEY invalid (401)")
        else:
            print(f"  ETF Flow FAIL: HTTP {resp.status_code}")
    except Exception as e:
        print(f"  ETF Flow FAIL: {e}")
    return False


def fetch_all_macro():
    """Fetch all global macro data (once daily)."""
    if not _should_fetch_macro():
        print(f"  Macro: already fetched today, skipping")
        return

    now = datetime.now(timezone.utc)
    print(f"\n  --- GLOBAL MACRO | {now.strftime('%Y-%m-%d %H:%M UTC')} ---")
    fetch_coingecko_global()
    fetch_fear_greed()
    fetch_etf_flow_dune()
    _mark_macro_fetched()


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
    macro_files = list(MACRO_DIR.glob("*.parquet"))
    total_coins = len(set(f.stem.split("_")[0] for f in files))
    print(f"\n{'='*55}")
    print(f"  DONE: {len(files)} positioning files ({total_coins} coins) + {len(macro_files)} macro files")
    print(f"  Data dirs: {POSITIONING_DIR} | {MACRO_DIR}")
    print(f"{'='*55}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--coins", nargs="+", default=None)
    parser.add_argument("--macro-only", action="store_true",
                        help="Only fetch macro data (no per-coin)")
    parser.add_argument("--schedule", action="store_true",
                        help="Run every hour (per-coin hourly, macro daily)")
    args = parser.parse_args()

    coins = args.coins or TRAINING_COINS

    if args.macro_only:
        fetch_all_macro()
        return

    if args.schedule:
        print("Running in schedule mode (per-coin hourly, macro daily). Ctrl+C to stop.")
        while True:
            try:
                start = time.time()
                fetch_all_coins(coins)
                fetch_all_macro()
                elapsed = time.time() - start
                sleep_time = max(60, 3600 - elapsed)
                print(f"\nNext fetch in {sleep_time/60:.0f} minutes...")
                time.sleep(sleep_time)
            except KeyboardInterrupt:
                print("\nStopped.")
                break
    else:
        fetch_all_coins(coins)
        fetch_all_macro()


if __name__ == "__main__":
    main()
