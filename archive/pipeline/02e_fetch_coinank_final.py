"""
02e_fetch_coinank_final.py — Pull Funding Rate + Grayscale + Altseason
Fixed parsing for Coinank's actual JSON formats.
"""
import json, os, subprocess, sys, time, numpy as np, pandas as pd
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).parent.parent; sys.path.insert(0, str(ROOT))
COINANK_DIR = ROOT / "data" / "coinank"; COINANK_DIR.mkdir(parents=True, exist_ok=True)
API_KEY = "d15dc004e6de4a4c828dedf2220588e9"
BIN = r"C:\Users\Bagas\AppData\Roaming\npm\coinank.cmd"
CC_MAP = {"BTCUSDT":"BTC","ETHUSDT":"ETH","SOLUSDT":"SOL","BNBUSDT":"BNB","XRPUSDT":"XRP",
          "DOGEUSDT":"DOGE","ADAUSDT":"ADA","TRXUSDT":"TRX","LINKUSDT":"LINK","DOTUSDT":"DOT",
          "AVAXUSDT":"AVAX","NEARUSDT":"NEAR","SUIUSDT":"SUI","TONUSDT":"TON",
          "1000PEPEUSDT":"PEPE","1000SHIBUSDT":"SHIB","ARBUSDT":"ARB","TAOUSDT":"TAO",
          "POLUSDT":"POL","HBARUSDT":"HBAR","ONDOUSDT":"ONDO"}

def cmd(args, timeout=120):
    env = {**os.environ, "COINANK_API_KEY": API_KEY, "PYTHONIOENCODING": "utf-8"}
    r = subprocess.run([BIN] + args + ["--json"], capture_output=True,
                       timeout=timeout, encoding="utf-8", errors="replace", env=env)
    return json.loads(r.stdout) if (r.returncode == 0 and r.stdout.strip()) else None

def save(fp, df):
    if df is None or len(df) == 0: return 0
    if fp.exists():
        e = pd.read_parquet(fp); df = pd.concat([e, df])
        df = df[~df.index.duplicated(keep="last")].sort_index()
    df.to_parquet(fp); return len(df)

def ts_to_dt(ms):
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc)

# ═══════════════════════════════════════════════════════════════════════════
# FUNDING RATE — historical time series
# ═══════════════════════════════════════════════════════════════════════════
def fetch_fr_hist(coin_cc, size=1000):
    """fr hist returns [{ts, details: {exchange: {fundingRate}}}, ...]"""
    data = cmd(["fr", "hist", "-c", coin_cc, "-t", "USDT", "-n", str(size)])
    if not data or not isinstance(data, list):
        return None
    rows = []
    for item in data:
        ts = item.get("ts", 0)
        if not ts: continue
        dt = ts_to_dt(ts)
        details = item.get("details", {})
        # Binance USDT funding rate
        binance = details.get("Binance", {})
        fr = binance.get("fundingRate", None)
        if fr is not None:
            rows.append({"timestamp": dt, "funding_rate": float(fr)})
    return pd.DataFrame(rows).set_index("timestamp").sort_index() if rows else None

# ═══════════════════════════════════════════════════════════════════════════
# GRAYSCALE — historical holdings
# ═══════════════════════════════════════════════════════════════════════════
def fetch_grayscale(symbol="BTC"):
    """grayscale returns {timeList: '[...]', opValueList: '[...]', priceList: '[...]'}"""
    data = cmd(["indicator", "grayscale", symbol])
    if not data or not isinstance(data, dict):
        return None

    try:
        tss = json.loads(data.get("timeList", "[]"))
        vals = json.loads(data.get("opValueList", "[]"))
        prices = json.loads(data.get("priceList", "[]"))
    except (json.JSONDecodeError, TypeError):
        return None

    rows = []
    for i, ts_ms in enumerate(tss):
        dt = ts_to_dt(ts_ms)
        row = {"timestamp": dt}
        if i < len(vals): row["grayscale_holdings"] = float(vals[i])
        if i < len(prices): row["btc_price"] = float(prices[i])
        rows.append(row)
    return pd.DataFrame(rows).set_index("timestamp").sort_index() if rows else None

# ═══════════════════════════════════════════════════════════════════════════
# ALTSEASON INDEX
# ═══════════════════════════════════════════════════════════════════════════
def fetch_altseason():
    data = cmd(["indicator", "altseason"])
    if not data or not isinstance(data, list):
        return None
    rows = []
    for item in data:
        ts_val = item.get("timestamp") or item.get("ts") or item.get("t")
        if not ts_val: continue
        dt = ts_to_dt(ts_val) if ts_val > 1e11 else datetime.fromtimestamp(ts_val, tz=timezone.utc)
        row = {"timestamp": dt}
        for k, v in item.items():
            if k not in ("timestamp", "ts", "t") and isinstance(v, (int, float)):
                row[f"altseason_{k}"] = float(v)
        rows.append(row)
    return pd.DataFrame(rows).set_index("timestamp").sort_index() if rows else None

# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════
def main():
    from config import TRAINING_COINS
    coins = TRAINING_COINS
    now = datetime.now(timezone.utc)

    print(f"\n{'='*60}")
    print(f"  COINANK FINAL FETCH | {now.strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"  FR hist + Grayscale BTC/ETH + Altseason")
    print(f"{'='*60}\n")

    # ── Funding Rate (per-coin) ──
    fr_ok = 0
    for ci, coin in enumerate(coins):
        coin_cc = CC_MAP.get(coin, coin.replace("USDT", ""))
        df = fetch_fr_hist(coin_cc)
        if df is not None and len(df) > 0:
            n = save(COINANK_DIR / f"{coin}_funding.parquet", df)
            print(f"  [{ci+1:>2}/{len(coins)}] {coin:<15} FR={n:>5} rows | {df.index[0].date()} -> {df.index[-1].date()}")
            fr_ok += 1
        else:
            print(f"  [{ci+1:>2}/{len(coins)}] {coin:<15} FR=FAIL")
        time.sleep(0.4)

    print(f"\n  FR: {fr_ok}/{len(coins)} coins OK")

    # ── Grayscale ──
    print(f"\n  --- GRAYSCALE ---")
    for sym in ["BTC", "ETH"]:
        df = fetch_grayscale(sym)
        if df is not None and len(df) > 0:
            n = save(COINANK_DIR / f"grayscale_{sym.lower()}.parquet", df)
            print(f"  {sym}: {n} rows | {df.index[0].date()} -> {df.index[-1].date()}")
        else:
            print(f"  {sym}: FAIL")
        time.sleep(0.5)

    # ── Altseason ──
    print(f"\n  --- ALTSEASON ---")
    df = fetch_altseason()
    if df is not None and len(df) > 0:
        n = save(COINANK_DIR / "altseason.parquet", df)
        print(f"  Altseason: {n} rows | {df.index[0].date()} -> {df.index[-1].date()}")
    else:
        print(f"  Altseason: FAIL")

    # ── Summary ──
    print(f"\n{'='*60}")
    print(f"  FINAL INVENTORY")
    for f in sorted(COINANK_DIR.glob("*.parquet")):
        df = pd.read_parquet(f)
        print(f"  {f.name:<45} {len(df):>6} rows | {str(df.index[0])[:19]} -> {str(df.index[-1])[:19]}")
    print(f"  Total: {len(list(COINANK_DIR.glob('*.parquet')))} files")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
