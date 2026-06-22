import sqlite3
from pathlib import Path

db = Path("data/live_cache/app.db")
con = sqlite3.connect(db)
con.row_factory = sqlite3.Row

r = con.execute("""
    SELECT MIN(s.signal_time) first_t, MAX(s.signal_time) last_t, COUNT(*) n
    FROM signal s JOIN model_meta m ON s.model_meta_id = m.id
    WHERE m.model_type = 'ic32_regime_v1'
""").fetchone()
print("ic32 signal range:", dict(r))

print("\nic32 samples:")
for row in con.execute("""
    SELECT s.signal_time, c.symbol, s.direction, s.entry_reason
    FROM signal s
    JOIN model_meta m ON s.model_meta_id = m.id
    JOIN coin c ON s.coin_id = c.id
    WHERE m.model_type = 'ic32_regime_v1'
    ORDER BY s.id DESC LIMIT 5
"""):
    d = dict(row)
    print(d["signal_time"], d["symbol"], d["direction"], (d["entry_reason"] or "")[:180])

print("\n2026-06-17 by model:")
for row in con.execute("""
    SELECT m.model_type, m.n_features, s.direction, COUNT(*) n
    FROM signal s JOIN model_meta m ON s.model_meta_id = m.id
    WHERE s.signal_time >= '2026-06-17'
    GROUP BY 1,2,3 ORDER BY n DESC
"""):
    print(dict(row))

print("\ncoin table sample:")
cols = [c[1] for c in con.execute("PRAGMA table_info(coin)")]
print("cols:", cols)
for row in con.execute("SELECT * FROM coin LIMIT 3"):
    print(dict(row))

print("\nModelSelection:")
try:
    for row in con.execute("""
        SELECT ms.coin_id, c.symbol, ms.model_meta_id, m.model_type, m.n_features
        FROM model_selection ms
        JOIN coin c ON ms.coin_id = c.id
        JOIN model_meta m ON ms.model_meta_id = m.id
        LIMIT 25
    """):
        print(dict(row))
    from collections import Counter
    types = [row["model_type"] for row in con.execute("""
        SELECT m.model_type FROM model_selection ms
        JOIN model_meta m ON ms.model_meta_id = m.id
    """)]
    print("selection counts:", Counter(types))
except Exception as e:
    print("err", e)

print("\nTimeline 2026-06-17:")
for row in con.execute("""
    SELECT strftime('%H', s.signal_time) hr, m.model_type, s.direction, COUNT(*) n
    FROM signal s JOIN model_meta m ON s.model_meta_id=m.id
    WHERE s.signal_time >= '2026-06-17'
    GROUP BY 1,2,3 ORDER BY 1,2
"""):
    print(dict(row))

print("\nLast ic32 ENTRY signal:")
for row in con.execute("""
    SELECT s.signal_time, c.symbol, s.direction, s.entry_reason
    FROM signal s JOIN model_meta m ON s.model_meta_id=m.id JOIN coin c ON s.coin_id=c.id
    WHERE m.model_type='ic32_regime_v1' AND s.direction != 'FLAT'
    ORDER BY s.id DESC LIMIT 3
"""):
    d=dict(row); print(d)

con.close()