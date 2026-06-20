"""
pipeline/01c_fetch_positioning.py — Fetch Positioning + Macro Data

=== PER-COIN (hourly): Binance + Bybit positioning ===
1. Binance: Taker Buy/Sell Volume Ratio (aggressor flow)
2. Binance: Top Trader Long/Short Position Ratio (elite positioning)
3. Binance: Global Long/Short Account Ratio (retail positioning)
4. Binance: Open Interest History (sumOpenInterest — parity 01_fetch)
5. Bybit:   Open Interest History (fallback cross-exchange)

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
from config import TRAINING_COINS, RAW_DIR

POSITIONING_DIR = ROOT / "data" / "positioning"
POSITIONING_DIR.mkdir(parents=True, exist_ok=True)
OI_DIR = RAW_DIR / "open_interest"
LS_DIR = RAW_DIR / "long_short_ratio"
OI_DIR.mkdir(parents=True, exist_ok=True)
LS_DIR.mkdir(parents=True, exist_ok=True)
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


# ─── Backfill Historical Macro Data ─────────────────────────────────────────

def backfill_fear_greed():
    """Fetch ALL historical Fear & Greed data (Dec 2020 -> now)."""
    print(f"\n  --- BACKFILL Fear & Greed ---")
    try:
        resp = requests.get(f"{FNG_BASE}/fng/?limit=2000", timeout=15)
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
            print(f"  Fear&Greed: {len(rows)} total points | {new_df.index[0].date()} -> {new_df.index[-1].date()} | -> {n} saved")
            return True
    except Exception as e:
        print(f"  Fear&Greed backfill FAIL: {e}")
    return False


def backfill_coingecko_btc():
    """
    Fetch BTC historical market data from CoinGecko (daily).
    Free API: up to 365 days. We fetch the full year.
    """
    print(f"\n  --- BACKFILL CoinGecko BTC History ---")
    # We can get daily data for last 365 days from free API
    # For longer history, need to call with different parameters
    ranges = [
        ("365d", 365),
    ]
    all_rows = []
    try:
        for label, days_val in ranges:
            url = f"{COINGECKO_BASE}/coins/bitcoin/market_chart?vs_currency=usd&days={days_val}&interval=daily"
            resp = requests.get(url, timeout=30)
            if resp.status_code == 200:
                data = resp.json()
                prices = data.get("prices", [])
                market_caps = data.get("market_caps", [])
                total_volumes = data.get("total_volumes", [])

                mc_map = {ts: v for ts, v in market_caps} if market_caps else {}
                vol_map = {ts: v for ts, v in total_volumes} if total_volumes else {}

                for ts_ms, price in prices:
                    dt = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)
                    all_rows.append({
                        "timestamp": dt,
                        "btc_price_usd": float(price),
                        "btc_market_cap_usd": float(mc_map.get(ts_ms, 0)),
                        "btc_total_volume_usd": float(vol_map.get(ts_ms, 0)),
                    })
                print(f"  CoinGecko {label}: {len(prices)} points")
            else:
                print(f"  CoinGecko {label}: HTTP {resp.status_code}")
            time.sleep(1.5)  # Rate limit

        if all_rows:
            new_df = pd.DataFrame(all_rows).set_index("timestamp").sort_index()
            new_df = new_df[~new_df.index.duplicated(keep="last")]
            n = update_parquet(MACRO_DIR / "coingecko_btc_history.parquet", new_df)
            print(f"  BTC History: {len(all_rows)} total points | {new_df.index[0].date()} -> {new_df.index[-1].date()} | -> {n} saved")
            return True
    except Exception as e:
        print(f"  CoinGecko backfill FAIL: {e}")
    return False


# ─── Yahoo Finance ETF Data ──────────────────────────────────────────────────

YAHOO_BASE = "https://query1.finance.yahoo.com/v8/finance/chart"

# BTC Spot ETF tickers (proxy for institutional flow)
ETF_TICKERS = {
    "IBIT": "BlackRock BTC ETF (largest, most liquid)",
    "FBTC": "Fidelity BTC ETF",
    "GBTC": "Grayscale BTC Trust (legacy, outflows)",
    "ARKB": "ARK 21Shares BTC ETF",
    "BITB": "Bitwise BTC ETF",
    "ETHW": "Bitwise ETH ETF",  # ETH ETF proxy
    "ETHA": "BlackRock ETH ETF",
}


def fetch_yahoo_etf(ticker, period="5y", interval="1d"):
    """
    Fetch ETF OHLCV data from Yahoo Finance.
    period: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max
    """
    params = {
        "period1": 0,  # auto-calc from period
        "period2": int(datetime.now(timezone.utc).timestamp()),
        "interval": interval,
        "includePrePost": "false",
        "events": "div,splits",
    }
    url = f"{YAHOO_BASE}/{ticker}"
    try:
        resp = requests.get(url, params=params, timeout=15,
                           headers={"User-Agent": "Mozilla/5.0"})
        if resp.status_code == 200:
            data = resp.json()
            result = data.get("chart", {}).get("result", [])
            if not result:
                return None
            meta = result[0]["meta"]
            quotes = result[0]["indicators"]["quote"][0]
            timestamps = result[0]["timestamp"]

            rows = []
            for i, ts in enumerate(timestamps):
                dt = datetime.fromtimestamp(ts, tz=timezone.utc)
                row = {
                    "timestamp": dt,
                    f"{ticker}_open": float(quotes["open"][i]) if quotes["open"][i] else None,
                    f"{ticker}_high": float(quotes["high"][i]) if quotes["high"][i] else None,
                    f"{ticker}_low": float(quotes["low"][i]) if quotes["low"][i] else None,
                    f"{ticker}_close": float(quotes["close"][i]) if quotes["close"][i] else None,
                    f"{ticker}_volume": int(quotes["volume"][i]) if quotes["volume"][i] else 0,
                }
                rows.append(row)

            if hasattr(meta, "get"):
                print(f"  {ticker}: {meta.get('symbol','?')} | {len(rows)} bars | period={period}")
            return rows
        else:
            print(f"  {ticker}: HTTP {resp.status_code}")
    except Exception as e:
        print(f"  {ticker} FAIL: {e}")
    return None


def backfill_etf_yahoo():
    """
    Backfill ALL historical ETF data from Yahoo Finance (free, 5y+ history).
    Saves per-ticker to data/macro/etf_{ticker}.parquet
    Also creates combined etf_btc_flow.parquet with key metrics.
    """
    print(f"\n  --- BACKFILL ETF Data (Yahoo Finance) ---")
    all_etf_data = {}

    for ticker, desc in ETF_TICKERS.items():
        rows = fetch_yahoo_etf(ticker, period="5y", interval="1d")
        if rows:
            df = pd.DataFrame(rows).set_index("timestamp").sort_index()
            df = df.dropna(how="all")
            n = update_parquet(MACRO_DIR / f"etf_{ticker}.parquet", df)
            all_etf_data[ticker] = df
            print(f"  {ticker} ({desc}): {len(df)} rows saved -> {n} total | {df.index[0].date()} to {df.index[-1].date()}")
        time.sleep(0.5)

    # Build combined BTC ETF metrics
    btc_etfs = ["IBIT", "FBTC", "GBTC", "ARKB", "BITB"]
    combined = None
    for ticker in btc_etfs:
        filepath = MACRO_DIR / f"etf_{ticker}.parquet"
        if filepath.exists():
            df = pd.read_parquet(filepath)
            close_col = f"{ticker}_close"
            vol_col = f"{ticker}_volume"
            if close_col in df.columns:
                s = df[[close_col, vol_col]].copy()
                s.columns = ["close", "volume"]
                s["ticker"] = ticker
                if combined is None:
                    combined = s
                else:
                    combined = pd.concat([combined, s])

    if combined is not None:
        # Daily: sum volume across all BTC ETFs, avg price
        daily = combined.groupby(combined.index).agg(
            btc_etf_total_volume=("volume", "sum"),
            btc_etf_avg_price=("close", "mean"),
            btc_etf_tickers=("ticker", "nunique"),
        )
        daily["btc_etf_volume_usd"] = daily["btc_etf_avg_price"] * daily["btc_etf_total_volume"]
        n = update_parquet(MACRO_DIR / "etf_btc_combined.parquet", daily)
        print(f"\n  Combined BTC ETF: {len(daily)} daily rows | {daily.index[0].date()} to {daily.index[-1].date()} | -> {n} total")
        print(f"  Avg daily volume: ${daily['btc_etf_volume_usd'].mean()/1e9:.1f}B")

    return len(all_etf_data) > 0


def backfill_all_macro():
    """Run all macro backfills to get full historical data."""
    now = datetime.now(timezone.utc)
    print(f"\n{'='*55}")
    print(f"  MACRO DATA BACKFILL | {now.strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"  Fetching ALL available historical data for training period")
    print(f"{'='*55}")
    backfill_fear_greed()
    backfill_coingecko_btc()
    backfill_etf_yahoo()
    print(f"\n  Backfill complete. Check data/macro/ for parquet files.")
    print(f"{'='*55}")


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
            # Sync ke path 02_clean.py (data asli, bukan synthetic)
            if "global_ls_ratio" in df_gls.columns:
                ls_clean = df_gls[["global_ls_ratio"]].rename(
                    columns={"global_ls_ratio": "long_short_ratio"}
                )
                n_ls = update_parquet(LS_DIR / f"{coin}_1h.parquet", ls_clean)
                print(f"  Global L/S -> 02_clean: {n_ls} rows")
            time.sleep(SLEEP_BETWEEN)

        # 4. Binance Open Interest (parity 01_fetch / 02_clean)
        df_oi_bn = fetch_binance("/futures/data/openInterestHist", symbol)
        if df_oi_bn is not None and "sumOpenInterest" in df_oi_bn.columns:
            df_oi_bn["sumOpenInterest"] = pd.to_numeric(df_oi_bn["sumOpenInterest"], errors="coerce")
            oi_clean = df_oi_bn[["sumOpenInterest"]].rename(
                columns={"sumOpenInterest": "open_interest"}
            ).dropna()
            if not oi_clean.empty:
                n_oi = update_parquet(OI_DIR / f"{coin}_1h.parquet", oi_clean)
                n_pos = update_parquet(
                    POSITIONING_DIR / f"{coin}_binance_oi.parquet", oi_clean
                )
                print(f"  Binance OI: {len(oi_clean)} new -> clean={n_oi} pos={n_pos}")
            time.sleep(SLEEP_BETWEEN)

        # 5. Bybit Open Interest (fallback)
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
    parser.add_argument("--backfill", action="store_true",
                        help="Fetch ALL available historical macro data for training")
    parser.add_argument("--schedule", action="store_true",
                        help="Run every hour (per-coin hourly, macro daily)")
    args = parser.parse_args()

    coins = args.coins or TRAINING_COINS

    if args.backfill:
        backfill_all_macro()
        return

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
