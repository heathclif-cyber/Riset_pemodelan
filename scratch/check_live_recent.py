import sqlite3
from pathlib import Path

con = sqlite3.connect(Path(__file__).parents[1] / "data/live_cache/app.db")
con.row_factory = sqlite3.Row

rows = con.execute("""
  SELECT t.id, t.opened_at, t.closed_at, t.pnl_net, t.exit_reason, t.is_live,
         t.quantity, t.leverage, t.entry_price, t.hold_bars, s.confidence, c.symbol
  FROM trade t
  JOIN signal s ON t.signal_id = s.id
  JOIN coin c ON s.coin_id = c.id
  WHERE t.opened_at >= '2026-06-18 00:00:00'
  ORDER BY t.opened_at DESC
  LIMIT 20
""").fetchall()
print(f"Trades since Jun 18: {len(rows)}")
for r in rows:
    d = dict(r)
    notional = (d["quantity"] or 0) * (d["entry_price"] or 0)
    margin = notional / d["leverage"] if d["leverage"] else None
    d["est_margin_usd"] = round(margin, 2) if margin else None
    print(d)

sig = con.execute(
    "SELECT COUNT(*) n, MAX(signal_time) last FROM signal WHERE signal_time >= '2026-06-18 00:00:00'"
).fetchone()
print("Signals since Jun 18:", dict(sig))

hb = con.execute("""
  SELECT exit_reason, ROUND(AVG(hold_bars),1) avg_hold, COUNT(*) n
  FROM trade WHERE is_live=1 AND closed_at >= '2026-06-17 00:00:00'
  GROUP BY exit_reason ORDER BY n DESC
""").fetchall()
print("Exit reasons since Jun 17:", [dict(x) for x in hb])
con.close()