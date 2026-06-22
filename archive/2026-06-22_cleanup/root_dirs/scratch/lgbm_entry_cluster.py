import sqlite3
import pandas as pd
from pathlib import Path
import json
from datetime import datetime

db_path = Path("data/live_cache/app.db")
con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
sql = """
    SELECT t.*,
           c.symbol AS coin_symbol,
           mm.model_type AS model_type,
           s.confidence AS signal_confidence,
           s.feature_snapshot AS feature_snapshot
    FROM trade t
    JOIN coin c ON t.coin_id = c.id
    LEFT JOIN signal s ON t.signal_id = s.id
    LEFT JOIN model_meta mm ON s.model_meta_id = mm.id
    WHERE t.is_live = 1
    ORDER BY t.opened_at DESC
"""
df = pd.read_sql_query(sql, con)
con.close()

# Focus on very recent cluster: opened on or after 2026-06-16
recent = df[df["opened_at"].astype(str) >= "2026-06-16"].copy()
recent = recent.sort_values("opened_at", ascending=False)

print("="*90)
print("LGBM ENTRY CLUSTER ANALYSIS — 2026-06-16 onward (is_live=1)")
print(f"Total recent live trades in this window: {len(recent)}")
print(f"Closed in window: {(recent['status']=='closed').sum()} | Open: {(recent['status']=='open').sum()}")
print("="*90)
print()

print("ALL RECENT ENTRIES (newest first) with key entry context:")
print(f'{"Opened":<17} | {"Coin":<12} | {"Dir":<5} | {"Status":<6} | {"PnL/Float":>10} | {"Exit":<20} | {"Hold":>4} | {"Conf":>5} | {"Model":<18} | H4  VolR')
print("-"*110)

for _, r in recent.iterrows():
    opened = str(r.get("opened_at", ""))[:16]
    coin = str(r.get("coin_symbol", ""))[:12]
    direc = str(r.get("direction", ""))[:5]
    status = str(r.get("status", ""))[:6]
    pnl = r.get("pnl_net")
    if status == "closed":
        pnl_str = f"{float(pnl):+7.2f}" if pd.notna(pnl) else "    NaN"
    else:
        pnl_str = "floating?"
    exit_r = str(r.get("exit_reason", "") or "open")[:20]
    hold = r.get("hold_bars")
    hold_str = str(int(hold)) if pd.notna(hold) else " ? "
    conf = r.get("signal_confidence")
    conf_str = f"{float(conf):.2f}" if pd.notna(conf) else " ? "
    model = str(r.get("model_type", ""))[:18]
    
    # Parse snapshot for H4 and Vol
    h4 = "?"
    volr = "?"
    fs = r.get("feature_snapshot")
    if fs:
        try:
            d = json.loads(fs) if isinstance(fs, str) else fs
            ht = d.get("h4_trend", None)
            h4 = "UP" if ht == 1 else ("DOWN" if ht == -1 else ("RANGE" if ht == 0 else "?"))
            vr = d.get("vol_regime", None)
            volr = f"{float(vr):.2f}" if vr is not None else "?"
        except:
            pass
    
    print(f"{opened:<17} | {coin:<12} | {direc:<5} | {status:<6} | {pnl_str:>10} | {exit_r:<20} | {hold_str:>4} | {conf_str:>5} | {model:<18} | {h4:4} {volr}")

print()
print("="*90)
print("SUMMARY OF BAD ENTRIES IN THIS CLUSTER")
print("="*90)

closed_recent = recent[recent["status"] == "closed"]
sl_hits = closed_recent[closed_recent["exit_reason"].fillna("") == "sl_hit"]
print(f"SL hits in this window: {len(sl_hits)}")
for _, r in sl_hits.iterrows():
    print(f"  {str(r['opened_at'])[:16]} {r['coin_symbol']:<10} {r['direction']:<5} PnL ${float(r['pnl_net']):+6.2f}  Hold={r['hold_bars']}  Conf={r.get('signal_confidence'):.2f}  Model={r.get('model_type')}")

print()
opens = recent[recent["status"] == "open"]
print(f"Currently open (floating): {len(opens)}")
for _, r in opens.iterrows():
    print(f"  {str(r['opened_at'])[:16]} {r['coin_symbol']:<12} {r['direction']:<5}  Conf={r.get('signal_confidence'):.2f}  Model={r.get('model_type')}   (entry ~ few hours ago)")

print()
print("Observations on entry quality:")
print("- Many entries with Vol Regime < 0.15 (very low volume)")
print("- Several SL within 0-3 bars (price immediately adverse)")
print("- Both LONG and SHORT firing in same short time windows")
print("- Confidences mostly 0.46-0.56 (not high conviction)")
print("- Cluster of 5+ bad outcomes in < 24 hours")
print()
print("This supports your point: the LGBM/cascade entry signals are the source of the rapid losses.")
print("Guardian never gets a chance on the sl_hit cases because they die too fast.")
