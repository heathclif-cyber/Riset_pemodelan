import sqlite3
import pandas as pd
from pathlib import Path
from datetime import datetime
import json

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
    ORDER BY t.opened_at DESC
"""
df = pd.read_sql_query(sql, con)
con.close()

live = df[df["is_live"] == 1].copy()
closed = live[live["status"] == "closed"].copy()

print("="*85)
print("RECENT LIVE TRADES ANALYSIS (is_live=1 CLOSED ONLY) — FOCUS ON TAIL")
print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
print(f"Total live closed: {len(closed)} | Open live: {(live['status']=='open').sum()}")
print("="*85)
print()

# Recent windows
print("### PERFORMANCE BY RECENT WINDOW (most recent first)")
for n in [15, 25, 40, 60]:
    recent = closed.head(n)
    pnls = recent["pnl_net"].astype(float).dropna()
    if len(pnls) == 0: continue
    wins = int((pnls > 0).sum())
    wr = wins / len(pnls) * 100
    net = round(pnls.sum(), 2)
    gp = round(pnls[pnls > 0].sum(), 2)
    gl = round(abs(pnls[pnls <= 0].sum()), 2)
    pf = round(gp / gl, 2) if gl > 0 else 99.0
    # streak in this recent slice (chronological order)
    srt = recent.iloc[::-1]
    streak = 0
    maxs = 0
    for p in srt["pnl_net"].fillna(0).astype(float):
        if p <= 0:
            streak += 1
            if streak > maxs: maxs = streak
        else:
            streak = 0
    print(f"Last {n:2d}: Net ${net:+7.2f} | WR {wr:5.1f}% ({wins}W/{len(pnls)-wins}L) | PF {pf:5.2f} | Max consec loss in window: {maxs}")

print()

# Last 30 detailed list
print("="*85)
print("LAST 30 CLOSED LIVE TRADES (MOST RECENT FIRST)")
print(f'{"Opened":<17} | {"Coin":<12} | {"Dir":<5} | {"PnL $":>8} | {"Exit Reason":<22} | {"Conf":>5} | {"Hold":>4} | {"Model":<15}')
print("-"*95)
for _, r in closed.head(30).iterrows():
    opened = str(r.get("opened_at", ""))[:16]
    coin = str(r.get("coin_symbol", ""))[:12]
    direc = str(r.get("direction", ""))[:5]
    pnl = r.get("pnl_net")
    pnl_str = f"{float(pnl):+7.2f}" if pd.notna(pnl) else "    NaN"
    exit_r = str(r.get("exit_reason", ""))[:22]
    conf = r.get("signal_confidence")
    conf_str = f"{float(conf):.2f}" if pd.notna(conf) else "  ? "
    hold = r.get("hold_bars")
    hold_str = str(int(hold)) if pd.notna(hold) else " ? "
    model = str(r.get("model_type", ""))[:15]
    print(f"{opened:<17} | {coin:<12} | {direc:<5} | {pnl_str:>8} | {exit_r:<22} | {conf_str:>5} | {hold_str:>4} | {model}")

print()

# Focus on recent SHORTS and LONGS
print("="*85)
print("RECENT 40: DIRECTION BREAKDOWN + EXIT")
print("="*85)
rec40 = closed.head(40)
print("SHORTS in last 40:")
shorts = rec40[rec40["direction"] == "SHORT"]
if len(shorts) > 0:
    sp = shorts["pnl_net"].astype(float).dropna()
    sw = int((sp > 0).sum())
    print(f"  n={len(shorts)} | WR {sw/len(sp)*100:.1f}% | Net ${sp.sum():+.2f}")
    print("  By exit:")
    for er, g in shorts.groupby(shorts["exit_reason"].fillna("unknown")):
        gp = g["pnl_net"].astype(float).dropna()
        print(f"    {er}: {len(g)} trades, WR {(gp>0).sum()/len(gp)*100:.1f}%, Net ${gp.sum():+.2f}")

print()
print("LONGS in last 40:")
longs = rec40[rec40["direction"] == "LONG"]
if len(longs) > 0:
    lp = longs["pnl_net"].astype(float).dropna()
    lw = int((lp > 0).sum())
    print(f"  n={len(longs)} | WR {lw/len(lp)*100:.1f}% | Net ${lp.sum():+.2f}")
    print("  By exit:")
    for er, g in longs.groupby(longs["exit_reason"].fillna("unknown")):
        gp = g["pnl_net"].astype(float).dropna()
        print(f"    {er}: {len(g)} trades, WR {(gp>0).sum()/len(gp)*100:.1f}%, Net ${gp.sum():+.2f}")

print()
print("Coins bleeding in last 40:")
cp = rec40.groupby("coin_symbol")["pnl_net"].sum().sort_values()
for c, v in cp.items():
    n = len(rec40[rec40["coin_symbol"] == c])
    print(f"  {c}: ${v:+.2f} ({n} trades)")

print()
print("="*85)
print("CURRENT TAIL STREAK (from most recent backward until first win)")
print("="*85)
streak = 0
streak_pnl = 0.0
streak_coins = []
for _, r in closed.iterrows():
    p = r.get("pnl_net")
    if pd.isna(p) or float(p) > 0:
        break
    streak += 1
    streak_pnl += float(p)
    streak_coins.append(r.get("coin_symbol", "?"))
print(f"Current losing streak: {streak} trades, cumulative PnL ${streak_pnl:+.2f}")
print(f"Coins in streak: {', '.join(streak_coins)}")
print()

# Try to pull some H4 / Vol from feature_snapshot for the very recent if available
print("Sample feature context for last 10 (H4 Trend / Vol if present in snapshot):")
for _, r in closed.head(10).iterrows():
    fs = r.get("feature_snapshot")
    if fs:
        try:
            d = json.loads(fs) if isinstance(fs, str) else fs
            h4 = d.get("h4_trend", "?")
            vol = d.get("vol_regime", "?")
            print(f"  {str(r['opened_at'])[:16]} {r['coin_symbol'][:8]:<8} {r['direction']:<5} H4={h4} VolR={vol}")
        except:
            pass
print()
print("Note: H4 Trend 1=UP, -1=DOWN, 0=RANGE (if available). Many recent rows may have limited snapshot.")
