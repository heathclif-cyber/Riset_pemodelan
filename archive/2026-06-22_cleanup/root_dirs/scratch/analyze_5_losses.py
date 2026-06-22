# -*- coding: utf-8 -*-
"""Analyze recent live losses."""
import json
import sqlite3
from pathlib import Path

import pandas as pd

DB = Path(__file__).resolve().parents[1] / "data" / "live_cache" / "app.db"
con = sqlite3.connect(DB)

df = pd.read_sql("""
SELECT t.id, c.symbol, t.direction, t.entry_price, t.exit_price, t.pnl_net, t.pnl_pct,
       t.exit_reason, t.opened_at, t.closed_at, t.hold_bars, t.leverage, t.quantity,
       t.tp_price, t.sl_price, s.confidence, s.feature_snapshot, s.entry_reason
FROM trade t
JOIN coin c ON t.coin_id = c.id
LEFT JOIN signal s ON t.signal_id = s.id
WHERE t.status = 'closed' AND t.is_live = 1
ORDER BY t.closed_at DESC
LIMIT 20
""", con)

print("=== 20 TRADE LIVE TERAKHIR ===\n")
for _, r in df.iterrows():
    pnl = r["pnl_net"] or 0
    tag = "LOSS" if pnl < 0 else "WIN "
    print(
        f"{tag} {str(r['closed_at'])[:19]} | {r['symbol']:14s} | {r['direction']:5s} | "
        f"PnL {pnl:+.2f} | {r['exit_reason']} | conf {r['confidence']:.2f} | hold {r['hold_bars']}"
    )

losses = 0
for _, r in df.iterrows():
    if (r["pnl_net"] or 0) < 0:
        losses += 1
    else:
        break
print(f"\n>>> Streak loss terbaru: {losses} trade berturut-turut")

# Last 5 losses detail
loss_df = df[df["pnl_net"] < 0].head(5)
print("\n=== 5 LOSS TERBARU - DETAIL FITUR ===\n")
total_loss = 0
for _, r in loss_df.iterrows():
    total_loss += r["pnl_net"] or 0
    fs = json.loads(r["feature_snapshot"]) if r["feature_snapshot"] else {}
    health = fs.get("_feature_health", "N/A")
    print(f"--- {r['symbol']} {r['direction']} | PnL {r['pnl_net']:+.2f} | {r['exit_reason']}")
    print(f"    opened={str(r['opened_at'])[:19]} closed={str(r['closed_at'])[:19]} hold={r['hold_bars']}")
    print(f"    LSR={fs.get('long_short_ratio')} HMM={fs.get('hmm_regime_enc')} RSI_H4={fs.get('rsi_h4')}")
    print(f"    h4_trend={fs.get('h4_trend')} conf={r['confidence']:.3f}")
    print(f"    health={health}")
    reason = (r["entry_reason"] or "")[:120]
    print(f"    reason: {reason}")
    print()

print(f"Total 5 loss: ${total_loss:.2f}")

# Stats today and since fix deploy (~15:39 UTC Jun 18)
for label, since in [
    ("Hari ini (Jun 18)", "2026-06-18"),
    ("Sejak deploy fix (~15:39 UTC)", "2026-06-18 15:39:00"),
    ("Sejak Jun 17", "2026-06-17"),
]:
    s = pd.read_sql(f"""
        SELECT COUNT(*) n, SUM(pnl_net) pnl,
               SUM(CASE WHEN pnl_net>0 THEN 1 ELSE 0 END) wins
        FROM trade WHERE status='closed' AND is_live=1 AND closed_at >= '{since}'
    """, con).iloc[0]
    n, pnl, wins = int(s["n"]), s["pnl"] or 0, int(s["wins"] or 0)
    wr = 100 * wins / n if n else 0
    print(f"{label}: {n} trades, WR {wr:.0f}%, PnL ${pnl:+.2f}")

# Signals since fix - LSR distribution
sig = pd.read_sql("""
SELECT s.signal_time, c.symbol, s.direction, s.feature_snapshot
FROM signal s JOIN coin c ON s.coin_id=c.id
WHERE s.signal_time >= '2026-06-18 15:39:00'
ORDER BY s.signal_time DESC LIMIT 50
""", con)
lsr_vals = []
for _, r in sig.iterrows():
    if r["feature_snapshot"]:
        fs = json.loads(r["feature_snapshot"])
        lsr_vals.append(fs.get("long_short_ratio", -999))
if lsr_vals:
    print(f"\nSignals post-fix: n={len(lsr_vals)} LSR min={min(lsr_vals):.4f} max={max(lsr_vals):.4f} mean={sum(lsr_vals)/len(lsr_vals):.4f}")
else:
    print("\nBelum ada signal post-fix di DB (cron HH:12 berikutnya)")

con.close()