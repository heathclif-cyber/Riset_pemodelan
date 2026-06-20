# -*- coding: utf-8 -*-
import json
import sqlite3
from pathlib import Path

import pandas as pd

DB = Path("data/live_cache/app.db")
conn = sqlite3.connect(DB)
live = pd.read_sql(
    """
    SELECT t.pnl_net, t.exit_reason, t.hold_bars, t.direction, t.quantity,
           t.entry_price, t.exit_price, t.sl_price, t.opened_at, c.symbol, s.confidence
    FROM trade t
    JOIN coin c ON c.id = t.coin_id
    LEFT JOIN signal s ON s.id = t.signal_id
    WHERE t.is_live = 1 AND t.status = 'closed' AND t.exit_reason = 'sl_hit'
    """,
    conn,
)
conn.close()

live["sl_dist_pct"] = abs(live["entry_price"] - live["sl_price"]) / live["entry_price"] * 100
live["actual_loss_pct"] = abs(live["entry_price"] - live["exit_price"]) / live["entry_price"] * 100
live["worse_than_sl"] = live["actual_loss_pct"] > live["sl_dist_pct"] + 0.05

print("=== LIVE sl_hit by modal ===")
for q, g in live.groupby("quantity"):
    print(f"  ${q}: n={len(g)} avg_pnl=${g.pnl_net.mean():.3f} avg_hold={g.hold_bars.mean():.1f}")

print("\n=== SL distance vs actual loss (close exit) ===")
print(f"avg planned SL dist: {live.sl_dist_pct.mean():.2f}%")
print(f"avg actual loss:     {live.actual_loss_pct.mean():.2f}%")
print(f"exit worse than SL:  {live.worse_than_sl.sum()}/{len(live)} trades")

print("\n=== Worst sl_hit ===")
cols = ["opened_at", "symbol", "direction", "quantity", "hold_bars", "confidence",
        "sl_dist_pct", "actual_loss_pct", "pnl_net"]
print(live.sort_values("pnl_net").head(8)[cols].to_string(index=False))