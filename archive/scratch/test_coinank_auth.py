"""Test Coinank API authentication."""
import requests

KEY = "d15dc004e6de4a4c828dedf2220588e9"
BASE = "https://api.coinank.com/api"
ENDPOINTS = ["/v1/health", "/v1/public/time", "/v1/market/overview", "/health"]

headers_list = [
    {"apikey": KEY},
    {"x-api-key": KEY},
    {"X-API-Key": KEY},
    {"api-key": KEY},
    {"Authorization": f"Bearer {KEY}"},
    {"Authorization": KEY},
    {"token": KEY},
    {"coinank-api-key": KEY},
    {"x-coinank-apikey": KEY},
]

# Test header auth
print("=== HEADER AUTH ===")
for h in headers_list:
    for ep in ENDPOINTS:
        try:
            r = requests.get(f"{BASE}{ep}", headers=h, timeout=8)
            if r.status_code != 404:
                txt = r.text[:200]
                print(f"  {list(h.keys())[0]} => {ep}: HTTP {r.status_code} [{txt}]")
        except Exception as e:
            pass

# Test query param auth
print("\n=== QUERY AUTH ===")
params = ["apikey", "api_key", "token", "access_token", "key"]
for p in params:
    for ep in ENDPOINTS:
        try:
            r = requests.get(f"{BASE}{ep}?{p}={KEY}", timeout=8)
            if r.status_code != 404:
                print(f"  {p} => {ep}: HTTP {r.status_code} [{r.text[:200]}]")
        except Exception as e:
            pass

# Also check web frontend for API docs
print("\n=== WEB FRONTEND ===")
for path in ["/api", "/docs", "/api-docs", "/swagger", "/document"]:
    try:
        r = requests.get(f"https://www.coinank.com{path}", timeout=8)
        print(f"  {path}: HTTP {r.status_code} [{r.text[:100]}]")
    except:
        pass
