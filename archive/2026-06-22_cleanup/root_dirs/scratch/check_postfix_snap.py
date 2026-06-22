import json, sqlite3
from pathlib import Path
db = Path("data/live_cache/app.db")
conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
rows = conn.execute(
    "SELECT signal_time, c.symbol, s.feature_snapshot "
    "FROM signal s JOIN coin c ON s.coin_id=c.id "
    "WHERE signal_time >= '2026-06-18 15:00' "
    "ORDER BY signal_time DESC LIMIT 5"
).fetchall()
for row in rows:
    s = json.loads(row[2]) if row[2] else {}
    print(row[0], row[1], "nkeys", len(s))
    for k in ["hmm_regime_enc", "long_short_ratio", "open_interest", "ofi_h4_delta", "rsi_h4"]:
        print(" ", k, s.get(k))