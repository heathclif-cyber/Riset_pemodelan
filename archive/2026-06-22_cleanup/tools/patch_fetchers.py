"""
tools/patch_fetchers.py — Fix fetch_fear_greed agar download data historis lengkap
dari Alternative.me API dengan limit=0 (all-time data)
"""
from pathlib import Path

fetchers_path = Path("core/fetchers.py")
content = fetchers_path.read_text(encoding="utf-8")

# Ganti hanya baris limit di fetch_fear_greed
# Alternative.me mendukung limit=0 untuk semua data historis (sejak 2018)
old = '        params={"limit": days_needed, "format": "json"},'
new = '        params={"limit": 0, "format": "json"},  # limit=0 = semua data historis (since 2018)'

if old in content:
    content_new = content.replace(old, new, 1)
    fetchers_path.write_text(content_new, encoding="utf-8")
    print("OK: fetch_fear_greed dipatch — limit=0 (all-time historical data)")
else:
    print(f"ERROR: Target string tidak ditemukan!")
    # Debug: cari sekitar baris 349
    lines = content.split('\n')
    for i, line in enumerate(lines[340:360], start=341):
        print(f"  {i}: {repr(line)}")
