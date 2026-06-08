"""Quick test to find correct Dune ETF flow query."""
import requests, time

key = "V9XZW3Fz3XANxw8fpPrmkTUfr13BSL5U"

execs = {
    "01KTFRBMGQWQD0VT1VVBZZ5CW7": "4125015 (hildobby ETF)",
    "01KTFRBQJEMD2ZN9GVSSHCC00Z": "3228576 (Flow tracker)",
}

for eid, desc in execs.items():
    for i in range(8):
        time.sleep(2)
        resp = requests.get(
            f"https://api.dune.com/api/v1/execution/{eid}/status",
            headers={"X-Dune-API-Key": key}, timeout=10)
        if resp.status_code != 200:
            print(f"{desc}: HTTP {resp.status_code}")
            break
        state = resp.json().get("state", "")
        if state == "QUERY_STATE_COMPLETED":
            resp2 = requests.get(
                f"https://api.dune.com/api/v1/execution/{eid}/results",
                headers={"X-Dune-API-Key": key}, timeout=15)
            if resp2.status_code == 200:
                rows = resp2.json().get("result", {}).get("rows", [])
                print(f"{desc}: {len(rows)} rows")
                if rows:
                    print(f"  Keys: {list(rows[0].keys())}")
                    print(f"  First: {rows[0]}")
                    print(f"  Last: {rows[-1]}")
            else:
                print(f"{desc}: results HTTP {resp2.status_code}")
            break
        elif state == "QUERY_STATE_FAILED":
            print(f"{desc}: FAILED - {resp.text[:300]}")
            break
