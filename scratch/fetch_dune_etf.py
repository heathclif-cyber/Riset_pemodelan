"""
scratch/fetch_dune_etf.py — Fetch BTC ETF flow dari Dune Analytics
Test query + inspect columns + save ke data/macro/etf_flow_dune.parquet
"""
import os, sys, json, requests, time
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

DUNE_KEY = os.environ.get("DUNE_API_KEY", "")
if not DUNE_KEY:
    print("DUNE_API_KEY not set — set via env or pass inline")
    sys.exit(1)

HEADERS = {"X-Dune-API-Key": DUNE_KEY}
BASE    = "https://api.dune.com/api/v1"
OUT_DIR = ROOT / "data" / "macro"

def get_query_results(query_id, limit=10000):
    url = f"{BASE}/query/{query_id}/results"
    params = {"limit": limit}
    r = requests.get(url, headers=HEADERS, params=params, timeout=30)
    print(f"  Query {query_id}: HTTP {r.status_code}")
    if r.status_code == 200:
        return r.json()
    else:
        print(f"  Error: {r.text[:400]}")
        return None

def execute_and_wait(query_id):
    """Execute query and wait for results (for fresh data)."""
    url = f"{BASE}/query/{query_id}/execute"
    r = requests.post(url, headers=HEADERS, timeout=30)
    if r.status_code != 200:
        print(f"  Execute failed: {r.text[:200]}")
        return None
    exec_id = r.json().get("execution_id")
    print(f"  Execution ID: {exec_id}")

    # Poll for status
    status_url = f"{BASE}/execution/{exec_id}/status"
    for i in range(60):
        time.sleep(3)
        sr = requests.get(status_url, headers=HEADERS, timeout=15)
        state = sr.json().get("state", "")
        print(f"  Polling {i+1}: {state}")
        if state == "QUERY_STATE_COMPLETED":
            break
        if "FAILED" in state or "CANCELLED" in state:
            print(f"  Query failed: {state}")
            return None

    # Get results
    result_url = f"{BASE}/execution/{exec_id}/results"
    rr = requests.get(result_url, headers=HEADERS, timeout=30)
    return rr.json()

# ─── Known Dune queries for BTC ETF flow ────────────────────────────────────
QUERIES = {
    "btc_etf_netflow_v1": 3802960,   # Original in production code
    "btc_etf_daily_v2":   3615936,   # Alternative
    "btc_etf_glassnode":  2516589,   # Glassnode-style BTC ETF
}

print("="*60)
print("  Testing Dune queries for BTC ETF flow")
print("="*60)

best_data = None
best_qname = None

for qname, qid in QUERIES.items():
    print(f"\n[{qname}] query_id={qid}")
    data = get_query_results(qid, limit=5)
    if data is None:
        continue
    rows = data.get("result", {}).get("rows", [])
    if not rows:
        print(f"  Empty result")
        continue
    print(f"  Rows returned: {len(rows)}")
    print(f"  Columns: {list(rows[0].keys())}")
    print(f"  Sample row: {rows[0]}")
    if len(rows) > 0 and best_data is None:
        best_data = (qname, qid, data)

# ─── Fetch full historical data from best query ─────────────────────────────
print("\n" + "="*60)
if best_data is None:
    print("No valid query results found. Try executing fresh.")
    print("Attempting execution of query 3802960...")
    data = execute_and_wait(3802960)
    if data:
        rows = data.get("result", {}).get("rows", [])
        if rows:
            best_data = ("btc_etf_netflow_v1", 3802960, data)

if best_data:
    qname, qid, data = best_data
    # Fetch full dataset
    full_data = get_query_results(qid, limit=10000)
    if full_data:
        rows = full_data.get("result", {}).get("rows", [])
        print(f"\nFull dataset: {len(rows)} rows from query {qid}")

        if rows:
            df = pd.DataFrame(rows)
            print(f"Columns: {list(df.columns)}")
            print(df.head(5).to_string())
            print(f"...")
            print(df.tail(3).to_string())

            # Try to parse date column
            date_cols = [c for c in df.columns if any(k in c.lower() for k in ['date','day','time','period'])]
            print(f"\nDate-like columns: {date_cols}")

            if date_cols:
                dc = date_cols[0]
                df[dc] = pd.to_datetime(df[dc], utc=True)
                df = df.set_index(dc).sort_index()
                df = df[~df.index.duplicated(keep='last')]

                out_path = OUT_DIR / "etf_flow_dune.parquet"
                df.to_parquet(out_path)
                print(f"\nSaved: {out_path}")
                print(f"  {len(df)} rows | {df.index.min().date()} -> {df.index.max().date()}")
                print(f"  Cols: {list(df.columns)}")

                # Save metadata
                meta = {"query_id": qid, "query_name": qname, "rows": len(df),
                        "start": str(df.index.min().date()), "end": str(df.index.max().date()),
                        "columns": list(df.columns)}
                with open(OUT_DIR / "etf_flow_dune_meta.json", "w") as f:
                    json.dump(meta, f, indent=2)
else:
    print("Could not fetch any ETF flow data from Dune.")
    print()
    print("Manual steps:")
    print("1. Go to https://dune.com/queries/3802960")
    print("2. Check if query is active / has results")
    print("3. Try queries: 3802960, 3615936, 2516589")
