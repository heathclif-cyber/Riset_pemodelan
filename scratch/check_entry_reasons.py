import sqlite3
from pathlib import Path

con = sqlite3.connect(Path(__file__).parents[1] / "data/live_cache/app.db")
rows = con.execute("""
    SELECT entry_reason, COUNT(*) n FROM signal
    WHERE entry_reason IS NOT NULL AND entry_reason != ''
    GROUP BY entry_reason ORDER BY n DESC LIMIT 20
""").fetchall()
print("entry_reason counts:")
for r in rows:
    print(f"  {r[1]:5d}  {r[0][:120]}")
con.close()