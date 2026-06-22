import sqlite3, json
from pathlib import Path

db = Path("data/live_cache/app.db")
con = sqlite3.connect(db)
con.row_factory = sqlite3.Row

# Cek kolom tabel trade
cols = con.execute("PRAGMA table_info(trade)").fetchall()
print("Kolom trade:", [c["name"] for c in cols])
print()

# Closed trades Juni 2026
rows = con.execute("""
    SELECT t.entry_price, t.exit_price, t.direction,
           t.pnl_usd, t.status, t.entry_time, t.exit_time, c.symbol,
           t.margin_usd, t.notional_usd
    FROM trade t
    JOIN coin c ON t.coin_id = c.id
    WHERE t.status IN ('closed','CLOSED','LOSS','WIN','TP','SL')
    AND strftime('%Y-%m', t.entry_time) = '2026-06'
    ORDER BY t.entry_time
""").fetchall()

print(f"June 2026 closed trades: {len(rows)}")
if rows:
    pnls   = [r["pnl_usd"] for r in rows if r["pnl_usd"] is not None]
    notls  = [r["notional_usd"] for r in rows if r["notional_usd"] is not None]
    wins   = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]

    import numpy as np
    avg_not = float(np.mean(notls)) if notls else 0
    print(f"Total PnL   : ${sum(pnls):.3f}")
    print(f"Avg/trade   : ${np.mean(pnls):.4f}  (notional avg ${avg_not:.1f})")
    print(f"WR          : {len(wins)}/{len(pnls)} = {len(wins)/max(len(pnls),1)*100:.1f}%")
    if wins:   print(f"Avg Win     : ${np.mean(wins):.4f}")
    if losses: print(f"Avg Loss    : ${np.mean(losses):.4f}")
    print()
    for r in rows:
        print(f"  {r['symbol']:12} {r['direction']:5} entry={r['entry_price']:.5g} "
              f"exit={r['exit_price']:.5g if r['exit_price'] else 'open':>10} "
              f"pnl=${r['pnl_usd']:.4f if r['pnl_usd'] else 0:.4f}  {r['status']}")

con.close()
