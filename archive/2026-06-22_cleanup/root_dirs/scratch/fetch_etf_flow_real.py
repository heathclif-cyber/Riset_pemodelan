"""
scratch/fetch_etf_flow_real.py — Cari sumber data ETF flow yang benar
Coba: Dune search API + SoSoValue + Farside
"""
import os, sys, json, requests
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
OUT_DIR = ROOT / "data" / "macro"

DUNE_KEY = os.environ.get("DUNE_API_KEY", "")
HEADERS = {"X-Dune-API-Key": DUNE_KEY} if DUNE_KEY else {}

# ─── 1. Search Dune for BTC ETF queries ─────────────────────────────────────
print("="*60)
print("1. Searching Dune for BTC ETF flow queries")
print("="*60)

if DUNE_KEY:
    # Try known community query IDs from Dune
    # These are popular BTC ETF flow dashboards
    CANDIDATE_QUERIES = {
        3802960: "Production code (WRONG - Ethereum miner)",
        3615936: "btc_etf_daily_v2 (404)",
        3726336: "BTC ETF Flows (popular)",
        3835897: "BTC Spot ETF Flows",
        3951673: "Bitcoin ETF daily netflow",
        3920997: "BTC ETF combined flow",
        4058764: "BTC Spot ETF Netflow",
        4095423: "BTC ETF inflow/outflow daily",
    }

    found = []
    for qid, desc in CANDIDATE_QUERIES.items():
        url = f"https://api.dune.com/api/v1/query/{qid}/results?limit=3"
        try:
            r = requests.get(url, headers=HEADERS, timeout=10)
            if r.status_code == 200:
                rows = r.json().get("result", {}).get("rows", [])
                if rows:
                    cols = list(rows[0].keys())
                    print(f"  FOUND [{qid}] {desc}")
                    print(f"    cols: {cols}")
                    print(f"    sample: {rows[0]}")
                    found.append((qid, desc, cols, rows[0]))
            else:
                print(f"  [{qid}] HTTP {r.status_code}: {desc}")
        except Exception as e:
            print(f"  [{qid}] error: {e}")

    if found:
        print(f"\nFound {len(found)} valid queries")
else:
    print("  DUNE_API_KEY not set")

# ─── 2. SoSoValue API (free, no key needed) ──────────────────────────────────
print("\n" + "="*60)
print("2. SoSoValue BTC ETF Flow API")
print("="*60)

# SoSoValue provides BTC spot ETF flow data
SOSOVALUE_URLS = [
    "https://ssosovalue.com/api/etf/spot-btc-etf/history",
    "https://api.ssosovalue.com/v1/etf/btc-spot/history",
    "https://ssosovalue.com/api/etf/btc-spot-etf/history",
    "https://ssosovalue.com/api/etf/spot-etf/history?type=BTC",
]

for url in SOSOVALUE_URLS:
    try:
        r = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        print(f"  {url}")
        print(f"    HTTP {r.status_code}")
        if r.status_code == 200:
            try:
                data = r.json()
                print(f"    Response keys: {list(data.keys()) if isinstance(data, dict) else type(data)}")
                if isinstance(data, list) and len(data) > 0:
                    print(f"    Sample: {data[0]}")
                elif isinstance(data, dict):
                    print(f"    Data: {str(data)[:300]}")
            except Exception:
                print(f"    Non-JSON: {r.text[:200]}")
    except Exception as e:
        print(f"    Error: {e}")

# ─── 3. Alternative: ETF.com / TheBlock / BitMEX Research ────────────────────
print("\n" + "="*60)
print("3. Alternative free ETF flow sources")
print("="*60)

# TheBlock data
tb_urls = [
    "https://data.theblock.co/api/bitcoin/etf-flows",
    "https://api.theblockresearch.com/btc-etf-flows",
]
for url in tb_urls:
    try:
        r = requests.get(url, timeout=8, headers={"User-Agent": "Mozilla/5.0"})
        print(f"  TheBlock {url}: HTTP {r.status_code}")
        if r.status_code == 200:
            print(f"    {r.text[:200]}")
    except Exception as e:
        print(f"    Error: {e}")

# CoinGlass ETF flow (public API)
cg_urls = [
    "https://open-api.coinglass.com/public/v2/etf/bitcoin/flow",
    "https://api.coinglass.com/api/bitcoin-etf/flow",
]
for url in cg_urls:
    try:
        r = requests.get(url, timeout=8, headers={"User-Agent": "Mozilla/5.0",
                                                   "Accept": "application/json"})
        print(f"  CoinGlass {url}: HTTP {r.status_code}")
        if r.status_code == 200:
            print(f"    {r.text[:300]}")
    except Exception as e:
        print(f"    Error: {e}")

# ─── 4. Derive real flow from ETF shares outstanding ─────────────────────────
print("\n" + "="*60)
print("4. ETF Share Count from yfinance (.info)")
print("="*60)
import yfinance as yf

tickers = ["IBIT", "FBTC", "GBTC", "ARKB", "BITB"]
for t in tickers:
    try:
        info = yf.Ticker(t).info
        shares = info.get("sharesOutstanding", "N/A")
        implied_btc = info.get("totalAssets", 0) / info.get("navPrice", 1) if info.get("navPrice") else "N/A"
        print(f"  {t}: sharesOutstanding={shares}, navPrice={info.get('navPrice','N/A')}, "
              f"totalAssets={info.get('totalAssets','N/A')}")
    except Exception as e:
        print(f"  {t}: error {e}")

print("\nDone.")
