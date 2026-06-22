"""Check Dune API availability + test query without key."""
import os, requests
from pathlib import Path

# 1. Check API key
key = os.environ.get("DUNE_API_KEY", "")
print(f"DUNE_API_KEY set: {bool(key)}")

possible_env = [
    Path("D:/Apps-Dev/swint_tradev2/.env"),
    Path("D:/Apps-Dev/Riset_pemodelan/.env"),
    Path.home() / ".env",
]
for p in possible_env:
    if p.exists():
        txt = p.read_text()
        if "DUNE" in txt:
            for line in txt.splitlines():
                if "DUNE" in line:
                    masked = line.split("=")[0] + "=***" if "=" in line else line
                    print(f"  Found in {p}: {masked}")

# 2. Check existing ETF parquets
import pandas as pd
macro_dir = Path("data/macro")
for f in macro_dir.glob("*.parquet"):
    if "etf" in f.name.lower():
        print(f"\n{f.name}:")
        try:
            df = pd.read_parquet(f)
            print(f"  rows={len(df)}, cols={list(df.columns)[:6]}")
            print(f"  range: {df.index.min()} -> {df.index.max()}")
        except Exception as e:
            print(f"  Error: {e}")

# 3. Test Dune query 3802960 without auth (public queries accessible)
print("\n--- Testing Dune query 3802960 (no auth) ---")
QUERY_ID = 3802960
url = f"https://api.dune.com/api/v1/query/{QUERY_ID}/results"
try:
    resp = requests.get(url, headers={}, timeout=15)
    print(f"  HTTP {resp.status_code}")
    if resp.status_code == 200:
        data = resp.json()
        rows = data.get("result", {}).get("rows", [])
        print(f"  rows={len(rows)}")
        if rows:
            print(f"  columns: {list(rows[0].keys())}")
            print(f"  sample: {rows[-1]}")
    else:
        print(f"  Response: {resp.text[:300]}")
except Exception as e:
    print(f"  Error: {e}")

# 4. Try alternative queries (no API key needed for public)
print("\n--- Alternative Dune queries ---")
ALT_QUERIES = {
    "BTC ETF netflow": 3802960,
    "BTC ETF flows v2": 3615936,
}
for name, qid in ALT_QUERIES.items():
    url = f"https://api.dune.com/api/v1/query/{qid}/results?limit=3"
    try:
        r = requests.get(url, timeout=10)
        print(f"  [{name}] query={qid}: HTTP {r.status_code}")
        if r.status_code == 200:
            rows = r.json().get("result", {}).get("rows", [])
            print(f"    cols={list(rows[0].keys()) if rows else 'empty'}")
    except Exception as e:
        print(f"  [{name}]: {e}")
