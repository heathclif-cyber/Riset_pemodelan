import sqlite3
from pathlib import Path

con = sqlite3.connect(Path(__file__).parents[1] / "data/live_cache/app.db")
con.row_factory = sqlite3.Row

# signal rejection reasons in feature_snapshot or entry_reason?
cols = [r[1] for r in con.execute("PRAGMA table_info(signal)").fetchall()]
print("signal cols:", cols)

# check if rejection logged in feature_snapshot json
rows = con.execute("""
    SELECT COUNT(*) n FROM signal
    WHERE feature_snapshot LIKE '%posisi_terbuka%'
       OR feature_snapshot LIKE '%pyramiding%'
""").fetchone()
print("signals with position reject in snapshot:", rows[0])

# open trades per coin now
open_by = con.execute("""
    SELECT c.symbol, COUNT(*) n, GROUP_CONCAT(t.direction) dirs
    FROM trade t JOIN coin c ON t.coin_id=c.id
    WHERE t.status='open' GROUP BY c.symbol
""").fetchall()
print("open trades by coin:", [dict(r) for r in open_by])

# same coin multiple signals while trade open - heuristic
dup = con.execute("""
    SELECT c.symbol, COUNT(*) n
    FROM signal s JOIN coin c ON s.coin_id=c.id
    WHERE s.direction IN ('LONG','SHORT')
    GROUP BY c.symbol, date(s.signal_time)
    HAVING n > 5
    ORDER BY n DESC LIMIT 10
""").fetchall()
print("coins with 5+ directional signals/day:", [dict(r) for r in dup])
con.close()