"""
02f_fetch_free_features.py — Free macro/on-chain features from public APIs

Sources (ALL free, no API key needed):
  1. CoinGecko — BTC dominance, ETH dominance, total market cap (daily)
  2. Blockchain.com — BTC on-chain data (daily)
  3. Computed from existing data — 200W MA, RSI breadth, OI ranking

Usage: python pipeline/02f_fetch_free_features.py --backfill
"""
import sys, time, numpy as np, pandas as pd, requests
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).parent.parent; sys.path.insert(0, str(ROOT))
MACRO_DIR = ROOT / "data" / "macro"; MACRO_DIR.mkdir(parents=True, exist_ok=True)

def update_parquet(fp, df):
    if fp.exists():
        e = pd.read_parquet(fp); df = pd.concat([e, df])
        df = df[~df.index.duplicated(keep="last")].sort_index()
    df.to_parquet(fp)
    return len(df)

# ═══════════════════════════════════════════════════════════════════════════
# 1. COINGECKO — BTC Dominance + Total Market Cap (daily, free)
# ═══════════════════════════════════════════════════════════════════════════
def fetch_coingecko_history(days=365):
    """Pull BTC dominance + market cap history from CoinGecko (free)."""
    print(f"\n  --- COINGECKO BTC History ({days}d) ---")
    url = f"https://api.coingecko.com/api/v3/coins/bitcoin/market_chart?vs_currency=usd&days={days}&interval=daily"
    try:
        r = requests.get(url, timeout=30)
        if r.status_code != 200:
            print(f"  HTTP {r.status_code}")
            return None

        data = r.json()
        rows = []
        prices = data.get("prices", [])
        mcaps = data.get("market_caps", [])
        vols = data.get("total_volumes", [])

        for i, (ts_ms, price) in enumerate(prices):
            dt = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)
            row = {"btc_price": price}
            if i < len(mcaps): row["btc_mcap"] = mcaps[i][1]
            if i < len(vols): row["btc_vol"] = vols[i][1]
            rows.append({"timestamp": dt, **row})

        df = pd.DataFrame(rows).set_index("timestamp").sort_index()
        n = update_parquet(MACRO_DIR / "coingecko_btc.parquet", df)
        print(f"  {n} rows | {df.index[0].date()} -> {df.index[-1].date()}")
        return df
    except Exception as e:
        print(f"  FAIL: {e}")
    return None

def fetch_coingecko_global():
    """Pull current global market data (BTC dominance snapshot)."""
    print(f"\n  --- COINGECKO Global ---")
    try:
        r = requests.get("https://api.coingecko.com/api/v3/global", timeout=15)
        if r.status_code != 200: return None
        d = r.json()["data"]
        now = datetime.now(timezone.utc)
        row = {
            "btc_dom": float(d["market_cap_percentage"]["btc"]),
            "eth_dom": float(d["market_cap_percentage"]["eth"]),
            "total_mcap": float(d["total_market_cap"]["usd"]),
            "total_vol_24h": float(d["total_volume"]["usd"]),
        }
        df = pd.DataFrame([{"timestamp": now, **row}]).set_index("timestamp")
        n = update_parquet(MACRO_DIR / "coingecko_global.parquet", df)
        print(f"  BTC dom={row['btc_dom']:.1f}% MCap=${row['total_mcap']/1e12:.2f}T | {n} total")
        return df
    except Exception as e:
        print(f"  FAIL: {e}")
    return None

# ═══════════════════════════════════════════════════════════════════════════
# 2. BLOCKCHAIN.COM — On-chain data (daily, free, no key)
# ═══════════════════════════════════════════════════════════════════════════
def fetch_blockchain_chart(chart_name, filename, col_name, days=365):
    """Pull a Blockchain.com chart (free, no auth)."""
    url = f"https://api.blockchain.info/charts/{chart_name}?format=json&timespan={days}days"
    try:
        r = requests.get(url, timeout=15)
        if r.status_code != 200: return None
        data = r.json()
        values = data.get("values", [])
        rows = []
        for v in values:
            dt = datetime.fromtimestamp(v["x"], tz=timezone.utc)
            rows.append({"timestamp": dt, col_name: float(v["y"])})
        df = pd.DataFrame(rows).set_index("timestamp").sort_index()
        n = update_parquet(MACRO_DIR / filename, df)
        print(f"  {col_name}: {n} rows | {df.index[0].date()} -> {df.index[-1].date()}")
        return df
    except Exception as e:
        print(f"  {col_name}: FAIL ({e})")
    return None

def fetch_blockchain_all():
    """Pull multiple Blockchain.com charts."""
    print(f"\n  --- BLOCKCHAIN.COM On-Chain ---")
    charts = [
        # (chart_name, filename, column_name)
        ("hash-rate", "btc_hashrate.parquet", "btc_hashrate"),
        ("n-transactions", "btc_tx_count.parquet", "btc_tx_count"),
        ("estimated-transaction-volume-usd", "btc_tx_volume.parquet", "btc_tx_volume_usd"),
        ("miners-revenue", "btc_miner_revenue.parquet", "btc_miner_revenue"),
        ("market-price", "btc_market_price.parquet", "btc_blockchain_price"),
    ]
    results = {}
    for chart, fname, col in charts:
        df = fetch_blockchain_chart(chart, fname, col, 365)
        if df is not None: results[col] = df
        time.sleep(0.5)
    return results

# ═══════════════════════════════════════════════════════════════════════════
# 3. COMPUTED FROM EXISTING KLINE DATA — 200W MA, RSI Breadth, OI Ranking
# ═══════════════════════════════════════════════════════════════════════════
def compute_200w_ma():
    """Compute 200-week moving average from BTC daily kline (already have)."""
    from config import LABEL_DIR
    fp = LABEL_DIR / "BTCUSDT_features_v3.parquet"
    if not fp.exists():
        print("  200W MA: no BTC data")
        return None

    df = pd.read_parquet(fp).sort_index()
    daily = df[["close"]].resample("1W").last().dropna()
    daily["ma_200w"] = daily["close"].rolling(200).mean()
    daily["price_to_200w"] = daily["close"] / daily["ma_200w"]
    daily["above_200w"] = (daily["close"] > daily["ma_200w"]).astype(int)

    result = daily[["ma_200w", "price_to_200w", "above_200w"]].dropna()
    result.index.name = "timestamp"
    n = update_parquet(MACRO_DIR / "btc_200w_ma.parquet", result)
    print(f"  200W MA: {n} weekly rows | {result.index[0].date()} -> {result.index[-1].date()}")
    print(f"  Current: {result['price_to_200w'].iloc[-1]:.2f}x 200W MA")
    return result


def main():
    now = datetime.now(timezone.utc)
    print(f"\n{'='*55}")
    print(f"  FREE MACRO FEATURE FETCH | {now.strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"  Sources: CoinGecko + Blockchain.com + Computed")
    print(f"{'='*55}")

    # CoinGecko
    fetch_coingecko_history(365)
    fetch_coingecko_global()
    time.sleep(1)

    # Blockchain.com
    fetch_blockchain_all()

    # Computed
    print(f"\n  --- COMPUTED FROM KLINE DATA ---")
    compute_200w_ma()

    # Summary
    print(f"\n{'='*55}")
    print(f"  MACRO FEATURES:")
    for f in sorted(MACRO_DIR.glob("*.parquet")):
        df = pd.read_parquet(f)
        cols = list(df.columns)
        print(f"  {f.name:<45} {len(df):>6} rows | {str(df.index[0])[:19]} -> {str(df.index[-1])[:19]} | {cols}")
    print(f"  Total: {len(list(MACRO_DIR.glob('*.parquet')))} macro files")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()
