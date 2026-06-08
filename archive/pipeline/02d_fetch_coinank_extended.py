"""
02d_fetch_coinank_extended.py — Pull remaining Coinank data

New vs 02c:
  - Funding Rate history (720d available)
  - Liquidation data
  - Grayscale holdings
  - Altseason index
  - OI vs Market Cap ratio
"""
import json, os, subprocess, sys, time
import numpy as np, pandas as pd
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).parent.parent; sys.path.insert(0, str(ROOT))
from config import TRAINING_COINS

COINANK_DIR = ROOT / "data" / "coinank"
COINANK_DIR.mkdir(parents=True, exist_ok=True)

API_KEY = "d15dc004e6de4a4c828dedf2220588e9"
COINANK_BIN = "C:\\Users\\Bagas\\AppData\\Roaming\\npm\\coinank.cmd"
BINANCE_COINS = {
    "BTCUSDT": "BTC", "ETHUSDT": "ETH", "SOLUSDT": "SOL", "BNBUSDT": "BNB",
    "XRPUSDT": "XRP", "DOGEUSDT": "DOGE", "ADAUSDT": "ADA", "TRXUSDT": "TRX",
    "LINKUSDT": "LINK", "DOTUSDT": "DOT", "AVAXUSDT": "AVAX", "NEARUSDT": "NEAR",
    "SUIUSDT": "SUI", "TONUSDT": "TON", "1000PEPEUSDT": "PEPE",
    "1000SHIBUSDT": "SHIB", "ARBUSDT": "ARB", "TAOUSDT": "TAO",
    "POLUSDT": "POL", "HBARUSDT": "HBAR", "ONDOUSDT": "ONDO",
}


def cmd(args_list, timeout=120):
    """Run coinank CLI with --json."""
    env = {**os.environ, "COINANK_API_KEY": API_KEY, "PYTHONIOENCODING": "utf-8"}
    try:
        r = subprocess.run([COINANK_BIN] + args_list + ["--json"],
                           capture_output=True, timeout=timeout, env=env,
                           encoding="utf-8", errors="replace")
        if r.returncode == 0 and r.stdout.strip():
            return json.loads(r.stdout)
        return None
    except Exception:
        return None


def save(filepath, df):
    if df is None or len(df) == 0: return 0
    if filepath.exists():
        existing = pd.read_parquet(filepath)
        df = pd.concat([existing, df])
        df = df[~df.index.duplicated(keep="last")].sort_index()
    df.to_parquet(filepath)
    return len(df)


def ts_to_dt(ts_ms):
    return datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)


# ─── Funding Rate ────────────────────────────────────────────────────────
def fetch_funding_rate(coin_cc, size=500):
    """Pull historical funding rates."""
    data = cmd(["fr", "hist", "-c", coin_cc, "-t", "USDT", "-n", str(size)])
    if not data or "tss" not in data:
        return None

    rows = []
    rates = data.get("fundingRates", [])
    tss = data.get("tss", [])
    for i, ts_ms in enumerate(tss):
        if i < len(rates) and rates[i] is not None:
            rows.append({"timestamp": ts_to_dt(ts_ms), "funding_rate": float(rates[i])})

    return pd.DataFrame(rows).set_index("timestamp").sort_index() if rows else None


def fetch_funding_accumulated(coin_cc, size=500):
    """Pull accumulated funding rates (more useful for momentum)."""
    for period in ["day", "week"]:
        data = cmd(["fr", "accumulated", "-c", coin_cc, "-t", "USDT", "-n", str(size)])
        if data and "tss" in data:
            rows = []
            rates = data.get("fundingRates", [])
            tss = data.get("tss", [])
            for i, ts_ms in enumerate(tss):
                if i < len(rates) and rates[i] is not None:
                    rows.append({"timestamp": ts_to_dt(ts_ms), "funding_acc": float(rates[i])})
            if rows:
                return pd.DataFrame(rows).set_index("timestamp").sort_index()
    return None


# ─── Liquidation ─────────────────────────────────────────────────────────
def fetch_liquidation(coin_cc, interval="4h", size=500):
    """Pull liquidation data. Try multiple command formats."""
    # Try liq command
    data = cmd(["liq", "-c", coin_cc, "-i", interval, "-n", str(size)])
    if data:
        if isinstance(data, list) and len(data) > 0:
            rows = []
            for item in data:
                ts_val = item.get("timestamp") or item.get("ts") or item.get("t")
                if ts_val:
                    dt = ts_to_dt(ts_val) if ts_val > 1e11 else datetime.fromtimestamp(ts_val, tz=timezone.utc)
                    row = {"timestamp": dt}
                    for k, v in item.items():
                        if k not in ("timestamp", "ts", "t") and isinstance(v, (int, float)):
                            row[f"liq_{k}"] = float(v)
                    rows.append(row)
            if rows:
                return pd.DataFrame(rows).set_index("timestamp").sort_index()

        # Try dict format
        if isinstance(data, dict) and "tss" in data:
            rows = []
            tss = data["tss"]
            for key, vals in data.items():
                if key != "tss" and isinstance(vals, list):
                    for i, ts_ms in enumerate(tss):
                        if i < len(vals) and vals[i] is not None:
                            pass  # Complex — skip for now
    return None


# ─── Grayscale ────────────────────────────────────────────────────────────
def fetch_grayscale(symbol="BTC"):
    """Pull Grayscale holdings (trust/fund flow data)."""
    data = cmd(["indicator", "grayscale", symbol])
    if data:
        if isinstance(data, list) and len(data) > 0:
            rows = []
            for item in data:
                ts_val = item.get("timestamp") or item.get("ts") or item.get("t")
                if ts_val:
                    dt = ts_to_dt(ts_val) if ts_val > 1e11 else datetime.fromtimestamp(ts_val, tz=timezone.utc)
                    row = {"timestamp": dt}
                    for k, v in item.items():
                        if k not in ("timestamp", "ts", "t") and isinstance(v, (int, float)):
                            row[f"grayscale_{k}"] = float(v)
                    rows.append(row)
            if rows:
                return pd.DataFrame(rows).set_index("timestamp").sort_index()
    return None


# ─── Altseason Index ──────────────────────────────────────────────────────
def fetch_altseason():
    """Pull altcoin season index."""
    data = cmd(["indicator", "altseason"])
    if data and isinstance(data, list) and len(data) > 0:
        rows = []
        for item in data:
            ts_val = item.get("timestamp") or item.get("ts") or item.get("t")
            if ts_val:
                dt = ts_to_dt(ts_val) if ts_val > 1e11 else datetime.fromtimestamp(ts_val, tz=timezone.utc)
                row = {"timestamp": dt}
                for k, v in item.items():
                    if k not in ("timestamp", "ts", "t") and isinstance(v, (int, float)):
                        row[k] = float(v)
                rows.append(row)
        if rows:
            return pd.DataFrame(rows).set_index("timestamp").sort_index()
    return None


# ─── OI vs Market Cap ─────────────────────────────────────────────────────
def fetch_oi_vs_mc(coin_cc="BTC"):
    """Pull OI vs market cap ratio."""
    data = cmd(["oi", "vs-mc", "-c", coin_cc])
    if data and isinstance(data, list) and len(data) > 0:
        rows = []
        for item in data:
            ts_val = item.get("timestamp") or item.get("ts") or item.get("t")
            if ts_val:
                dt = ts_to_dt(ts_val) if ts_val > 1e11 else datetime.fromtimestamp(ts_val, tz=timezone.utc)
                row = {"timestamp": dt}
                for k, v in item.items():
                    if k not in ("timestamp", "ts", "t") and isinstance(v, (int, float)):
                        row[k] = float(v)
                rows.append(row)
        if rows:
            return pd.DataFrame(rows).set_index("timestamp").sort_index()
    return None


def fetch_all_remaining(coins):
    """Pull all remaining data for all coins."""
    total = len(coins)
    success = {"fr": 0, "fr_acc": 0, "liq": 0, "gs": 0, "alt": 0, "oi_vs_mc": 0}

    for ci, coin in enumerate(coins):
        coin_cc = BINANCE_COINS.get(coin, coin.replace("USDT", ""))
        print(f"  [{ci+1}/{total}] {coin} ...", end=" ", flush=True)

        parts = []

        # Funding Rate
        df = fetch_funding_rate(coin_cc)
        if df is not None and len(df) > 0:
            n = save(COINANK_DIR / f"{coin}_funding.parquet", df)
            parts.append(f"FR={n}")
            success["fr"] += 1
        time.sleep(0.4)

        # Accumulated Funding
        df = fetch_funding_accumulated(coin_cc)
        if df is not None and len(df) > 0:
            n = save(COINANK_DIR / f"{coin}_funding_acc.parquet", df)
            parts.append(f"FR_acc={n}")
            success["fr_acc"] += 1
        time.sleep(0.4)

        # Liquidation
        df = fetch_liquidation(coin_cc)
        if df is not None and len(df) > 0:
            n = save(COINANK_DIR / f"{coin}_liq.parquet", df)
            parts.append(f"LIQ={n}")
            success["liq"] += 1
        time.sleep(0.4)

        print(", ".join(parts) if parts else "no new data")

    # Macro data (not per-coin)
    print(f"\n  --- MACRO ---")

    # Grayscale
    for sym in ["BTC", "ETH"]:
        df = fetch_grayscale(sym)
        if df is not None and len(df) > 0:
            n = save(COINANK_DIR / f"grayscale_{sym.lower()}.parquet", df)
            success["gs"] += n
            print(f"  Grayscale {sym}: {n} rows")
        time.sleep(0.5)

    # Altseason
    df = fetch_altseason()
    if df is not None and len(df) > 0:
        n = save(COINANK_DIR / "altseason.parquet", df)
        success["alt"] = n
        print(f"  Altseason: {n} rows")

    # OI vs MC (BTC only)
    df = fetch_oi_vs_mc("BTC")
    if df is not None and len(df) > 0:
        n = save(COINANK_DIR / "oi_vs_mc.parquet", df)
        success["oi_vs_mc"] = n
        print(f"  OI vs MC: {n} rows")

    return success


def main():
    coins = TRAINING_COINS
    now = datetime.now(timezone.utc)
    print(f"\n{'='*60}")
    print(f"  COINANK EXTENDED FETCH | {now.strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"  Funding Rate + Liquidation + Grayscale + Altseason")
    print(f"  Coins: {len(coins)}")
    print(f"{'='*60}\n")

    results = fetch_all_remaining(coins)

    print(f"\n{'='*60}")
    print(f"  FETCH COMPLETE")
    for k, v in results.items():
        print(f"  {k}: {v}")
    print(f"\n  All files:")
    for f in sorted(COINANK_DIR.glob("*.parquet")):
        df = pd.read_parquet(f)
        print(f"    {f.name:<45} {len(df):>6} rows | {df.index[0]} -> {df.index[-1]}")
    print(f"  Total: {len(list(COINANK_DIR.glob('*.parquet')))} files")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
