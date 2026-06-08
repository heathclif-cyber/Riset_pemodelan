"""
pipeline/02c_fetch_coinank.py — Backfill positioning data from Coinank Plan2

Plan2 capabilities (7-day trial, exp 2026-06-14):
  OI chart:            360d @ 4H
  LS position/account: 180-360d @ 4H
  LS kline:            180d
  Funding rate:        720d
  ETF inflow:          360d (daily)
  Fear & Greed:        full history
  Liquidation:         360d

Usage:
  python pipeline/02c_fetch_coinank.py                      # all coins, all data
  python pipeline/02c_fetch_coinank.py --coins BTCUSDT       # single coin
  python pipeline/02c_fetch_coinank.py --macro-only          # ETF + F&G only
"""
import argparse, json, os, sys, subprocess, time
import numpy as np, pandas as pd
from datetime import datetime, timezone
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT = Path(__file__).parent.parent; sys.path.insert(0, str(ROOT))
from config import TRAINING_COINS

COINANK_DIR = ROOT / "data" / "coinank"
COINANK_DIR.mkdir(parents=True, exist_ok=True)

API_KEY = os.environ.get("COINANK_API_KEY", "d15dc004e6de4a4c828dedf2220588e9")
COINANK_BIN = "C:\\Users\\Bagas\\AppData\\Roaming\\npm\\coinank.cmd"
BINANCE_COINS = {
    "BTCUSDT": "BTC", "ETHUSDT": "ETH", "SOLUSDT": "SOL", "BNBUSDT": "BNB",
    "XRPUSDT": "XRP", "DOGEUSDT": "DOGE", "ADAUSDT": "ADA", "TRXUSDT": "TRX",
    "LINKUSDT": "LINK", "DOTUSDT": "DOT", "AVAXUSDT": "AVAX", "NEARUSDT": "NEAR",
    "SUIUSDT": "SUI", "TONUSDT": "TON", "1000PEPEUSDT": "PEPE",
    "1000SHIBUSDT": "SHIB", "ARBUSDT": "ARB", "TAOUSDT": "TAO",
    "POLUSDT": "POL", "HBARUSDT": "HBAR", "ONDOUSDT": "ONDO",
}


def coinank_cmd(args_list, timeout=60):
    """Run coinank CLI with --json flag, return parsed JSON or None."""
    env = {**os.environ, "COINANK_API_KEY": API_KEY}
    cmd = [COINANK_BIN] + args_list + ["--json"]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=env)
        if result.returncode == 0 and result.stdout.strip():
            return json.loads(result.stdout)
        else:
            err = result.stderr[:200] if result.stderr else "no stderr"
            return None
    except subprocess.TimeoutExpired:
        return None
    except Exception as e:
        return None


def fetch_oi(coin_cc, interval="4h", size=2000):
    """Fetch OI history chart data."""
    data = coinank_cmd(["oi", "chart", "-c", coin_cc, "-i", interval, "-n", str(size)])
    if not data or "tss" not in data or "dataValues" not in data:
        return None

    rows = []
    for i, ts_ms in enumerate(data["tss"]):
        dt = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)
        row = {"timestamp": dt}
        for exchange, values in data["dataValues"].items():
            if i < len(values) and values[i] is not None:
                row[f"oi_{exchange.lower()}"] = float(values[i])
        if "prices" in data and data["prices"][i] is not None:
            row["price"] = float(data["prices"][i])
        rows.append(row)

    if not rows:
        return None
    return pd.DataFrame(rows).set_index("timestamp").sort_index()


def fetch_ls_position(coin_cc, interval="4h", size=2000):
    """Fetch Top Trader Position Ratio (smart money)."""
    data = coinank_cmd(["ls", "position", "-c", coin_cc, "-e", "Binance",
                         "-i", interval, "-n", str(size)])
    if not data or "tss" not in data:
        return None

    rows = []
    for i, ts_ms in enumerate(data["tss"]):
        dt = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)
        row = {"timestamp": dt}
        if "longShortRatio" in data and i < len(data["longShortRatio"]):
            row["top_trader_position_ls"] = float(data["longShortRatio"][i])
        rows.append(row)

    return pd.DataFrame(rows).set_index("timestamp").sort_index() if rows else None


def fetch_ls_account(coin_cc, interval="4h", size=2000):
    """Fetch Top Trader Account Ratio."""
    data = coinank_cmd(["ls", "account", "-c", coin_cc, "-e", "Binance",
                         "-i", interval, "-n", str(size)])
    if not data or "tss" not in data:
        return None

    rows = []
    for i, ts_ms in enumerate(data["tss"]):
        dt = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)
        row = {"timestamp": dt}
        if "longShortRatio" in data and i < len(data["longShortRatio"]):
            row["top_trader_account_ls"] = float(data["longShortRatio"][i])
        rows.append(row)

    return pd.DataFrame(rows).set_index("timestamp").sort_index() if rows else None


def fetch_funding_rate(coin_cc, size=2000):
    """Fetch historical funding rates."""
    data = coinank_cmd(["fr", "hist", "-c", coin_cc, "-t", "USDT", "-n", str(size)])
    if not data or "tss" not in data:
        return None

    rows = []
    for i, ts_ms in enumerate(data.get("tss", [])):
        dt = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)
        row = {"timestamp": dt}
        if "fundingRates" in data and i < len(data["fundingRates"]):
            row["funding_rate"] = float(data["fundingRates"][i])
        rows.append(row)

    return pd.DataFrame(rows).set_index("timestamp").sort_index() if rows else None


def fetch_liq(coin_cc, interval="4h", size=2000):
    """Fetch liquidation data."""
    data = coinank_cmd(["liq", "-c", coin_cc, "-i", interval, "-n", str(size)])
    if not data:
        return None

    # Liq data format may vary — try to parse from common structures
    if isinstance(data, list):
        rows = []
        for item in data:
            if "timestamp" in item or "ts" in item:
                ts_val = item.get("timestamp") or item.get("ts")
                dt = datetime.fromtimestamp(ts_val / 1000, tz=timezone.utc)
                row = {"timestamp": dt}
                for k, v in item.items():
                    if k not in ("timestamp", "ts") and isinstance(v, (int, float)):
                        row[f"liq_{k}"] = float(v)
                rows.append(row)
        if rows:
            return pd.DataFrame(rows).set_index("timestamp").sort_index()

    if isinstance(data, dict) and "tss" in data:
        rows = []
        for i, ts_ms in enumerate(data["tss"]):
            dt = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)
            row = {"timestamp": dt}
            for k, vals in data.items():
                if k != "tss" and isinstance(vals, list) and i < len(vals):
                    row[f"liq_{k}"] = float(vals[i]) if vals[i] is not None else 0.0
            rows.append(row)
        if rows:
            return pd.DataFrame(rows).set_index("timestamp").sort_index()

    return None


def fetch_etf_inflow(asset="btc"):
    """Fetch ETF daily net inflow (BTC or ETH)."""
    data = coinank_cmd(["etf", "inflow", asset])
    if not data or not isinstance(data, list):
        return None

    rows = []
    for day in data:
        if "date" not in day:
            continue
        dt = datetime.fromtimestamp(day["date"] / 1000, tz=timezone.utc)
        total_change = day.get("change", 0)
        total_change_usd = day.get("changeUsd", 0)
        # Per-ticker breakdown
        ticker_changes = {}
        for ticker_data in day.get("list", []):
            ticker = ticker_data.get("ticker", "").lower()
            ticker_changes[f"etf_{ticker}_change"] = ticker_data.get("change", 0)
            ticker_changes[f"etf_{ticker}_change_usd"] = ticker_data.get("changeUsd", 0)

        rows.append({
            "timestamp": dt,
            "etf_total_change": total_change,
            "etf_total_change_usd": total_change_usd,
            **ticker_changes,
        })

    return pd.DataFrame(rows).set_index("timestamp").sort_index() if rows else None


def fetch_fear_greed():
    """Fetch Fear & Greed index."""
    data = coinank_cmd(["indicator", "fg"])
    if not data:
        return None
    # Parse the indicator data
    if isinstance(data, list):
        rows = []
        for item in data:
            if "timestamp" in item or "ts" in item:
                ts_val = item.get("timestamp") or item.get("ts")
                dt = datetime.fromtimestamp(ts_val / 1000, tz=timezone.utc)
                rows.append({
                    "timestamp": dt,
                    "fear_greed_value": item.get("value", item.get("fg", 0)),
                })
        if rows:
            return pd.DataFrame(rows).set_index("timestamp").sort_index()
    return None


def update_parquet(filepath, new_df):
    """Append/deduplicate parquet."""
    if new_df is None or len(new_df) == 0:
        return 0
    if filepath.exists():
        existing = pd.read_parquet(filepath)
        combined = pd.concat([existing, new_df])
        combined = combined[~combined.index.duplicated(keep="last")].sort_index()
    else:
        combined = new_df
    combined.to_parquet(filepath)
    return len(combined)


def fetch_coin(coin_usdt, interval="4h", size=2000):
    """Fetch all positioning data for one coin."""
    coin_cc = BINANCE_COINS.get(coin_usdt, coin_usdt.replace("USDT", ""))
    symbol = coin_usdt

    results = {}

    # 1. Open Interest
    df = fetch_oi(coin_cc, interval, size)
    if df is not None:
        n = update_parquet(COINANK_DIR / f"{symbol}_oi.parquet", df)
        results["oi"] = n
        time.sleep(0.3)

    # 2. Top Trader Position L/S
    df = fetch_ls_position(coin_cc, interval, size)
    if df is not None:
        n = update_parquet(COINANK_DIR / f"{symbol}_ls_position.parquet", df)
        results["ls_position"] = n
        time.sleep(0.3)

    # 3. Top Trader Account L/S
    df = fetch_ls_account(coin_cc, interval, size)
    if df is not None:
        n = update_parquet(COINANK_DIR / f"{symbol}_ls_account.parquet", df)
        results["ls_account"] = n
        time.sleep(0.3)

    # 4. Funding Rate
    df = fetch_funding_rate(coin_cc, size)
    if df is not None:
        n = update_parquet(COINANK_DIR / f"{symbol}_funding.parquet", df)
        results["funding"] = n
        time.sleep(0.3)

    # 5. Liquidations
    df = fetch_liq(coin_cc, interval, size)
    if df is not None:
        n = update_parquet(COINANK_DIR / f"{symbol}_liq.parquet", df)
        results["liq"] = n
        time.sleep(0.3)

    return results


def fetch_all_macro():
    """Fetch global macro data (not per-coin)."""
    results = {}

    # ETF Inflow (BTC + ETH)
    for asset in ["btc", "eth"]:
        df = fetch_etf_inflow(asset)
        if df is not None:
            n = update_parquet(COINANK_DIR / f"etf_inflow_{asset}.parquet", df)
            results[f"etf_{asset}"] = n
        time.sleep(0.5)

    # Fear & Greed
    df = fetch_fear_greed()
    if df is not None:
        n = update_parquet(COINANK_DIR / "fear_greed.parquet", df)
        results["fear_greed"] = n

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--coins", nargs="+", default=None)
    parser.add_argument("--macro-only", action="store_true")
    parser.add_argument("--interval", default="4h")
    parser.add_argument("--size", type=int, default=2000)
    parser.add_argument("--parallel", type=int, default=3)
    args = parser.parse_args()

    coins = args.coins or TRAINING_COINS
    interval = args.interval
    size = args.size

    now = datetime.now(timezone.utc)
    print(f"\n{'='*60}")
    print(f"  COINANK FETCH | {now.strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"  Plan2 | Coins: {len(coins)} | Interval: {interval} | Size: {size}")
    print(f"{'='*60}\n")

    if not args.macro_only:
        # Per-coin fetch (sequential with small parallel to respect rate limits)
        total = len(coins)
        for ci, coin in enumerate(coins):
            try:
                res = fetch_coin(coin, interval, size)
                parts = [f"{k}={v}" for k, v in res.items() if v]
                print(f"  [{ci+1}/{total}] {coin}: {', '.join(parts) if parts else 'FAIL'}")
            except Exception as e:
                print(f"  [{ci+1}/{total}] {coin}: ERROR {e}")

    # Macro fetch
    print(f"\n  --- MACRO ---")
    macro_results = fetch_all_macro()
    parts = [f"{k}={v}" for k, v in macro_results.items() if v]
    print(f"  {', '.join(parts) if parts else 'FAIL'}")

    # Summary
    files = list(COINANK_DIR.glob("*.parquet"))
    print(f"\n{'='*60}")
    print(f"  DONE: {len(files)} files")
    for f in sorted(files):
        df = pd.read_parquet(f)
        print(f"    {f.name:<40} {len(df):>6} rows | {df.index[0]} -> {df.index[-1]}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
