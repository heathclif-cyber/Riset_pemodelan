import sqlite3
import pandas as pd
from pathlib import Path
from datetime import datetime

db_path = Path("data/live_cache/app.db")
con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)

sql = """
    SELECT t.*,
           c.symbol AS coin_symbol,
           mm.model_type AS model_type
    FROM trade t
    JOIN coin c ON t.coin_id = c.id
    LEFT JOIN signal s ON t.signal_id = s.id
    LEFT JOIN model_meta mm ON s.model_meta_id = mm.id
    ORDER BY t.opened_at
"""
df = pd.read_sql_query(sql, con)
con.close()

live = df[df["is_live"] == 1].copy()
closed = live[live["status"] == "closed"].copy()
pnls = closed["pnl_net"].astype(float).dropna()
print("="*70)
print("LIVE TRADING ANALYSIS (is_live=1 ONLY) - FRESH FROM VPS")
print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
print(f"Total live trades: {len(live)} | Closed: {len(closed)} | Open: {(live['status']=='open').sum()}")
print("="*70)
print()

if len(closed) == 0:
    print("No closed live trades.")
else:
    net = round(pnls.sum(), 2)
    n = len(pnls)
    wins = int((pnls > 0).sum())
    losses = n - wins
    wr = round(wins / n * 100, 1)
    gp = round(pnls[pnls > 0].sum(), 2)
    gl = round(abs(pnls[pnls <= 0].sum()), 2)
    pf = round(gp / gl, 2) if gl > 0 else 99.0
    exp = round(pnls.mean(), 2)
    print("## 1. CORE METRICS (LIVE CLOSED)")
    print(f"Net PnL          : ${net:+.2f}")
    print(f"Win Rate         : {wr:.1f}%  ({wins}W / {losses}L out of {n})")
    print(f"Profit Factor    : {pf:.2f}")
    print(f"Expectancy       : ${exp:+.2f} per trade")
    print(f"Gross Profit     : ${gp:.2f}")
    print(f"Gross Loss       : ${gl:.2f}")
    # max consec
    srt = closed.sort_values("opened_at")
    streak = max_streak = 0
    cur = max_l = 0.0
    for p in srt["pnl_net"].fillna(0).astype(float):
        if p <= 0:
            streak += 1
            cur += p
            if streak > max_streak:
                max_streak = streak
                max_l = cur
        else:
            streak = 0
            cur = 0.0
    print(f"Max Consec Loss  : {max_streak} trades (${round(max_l,2):.2f})")
    print()

    print("## 2. PER DIRECTION")
    for d in ["LONG", "SHORT"]:
        ddf = closed[closed["direction"] == d]
        if len(ddf) > 0:
            dpnl = ddf["pnl_net"].astype(float).dropna()
            dw = int((dpnl > 0).sum())
            dpf = round( dpnl[dpnl>0].sum() / abs(dpnl[dpnl<=0].sum()) , 2) if (dpnl <= 0).sum() > 0 else 99.0
            print(f"  {d:5s}: {len(ddf):3d} trades | WR {dw/len(dpnl)*100:5.1f}% | PnL ${dpnl.sum():+7.2f} | PF {dpf:.2f}")
    print()

    print("## 3. PER EXIT_REASON (live closed)")
    if "exit_reason" in closed.columns:
        for er, edf in closed.groupby(closed["exit_reason"].fillna("unknown")):
            epnl = edf["pnl_net"].astype(float).dropna()
            ew = int((epnl > 0).sum())
            ewr = ew / len(epnl) * 100 if len(epnl) > 0 else 0
            epf = round( epnl[epnl>0].sum() / abs(epnl[epnl<=0].sum()) , 2) if (epnl<=0).sum()>0 else 99.0
            print(f"  {str(er):25s}: {len(edf):3d} | WR {ewr:5.1f}% | PnL ${epnl.sum():+8.2f} | PF {epf:.2f}")
    print()

    print("## 4. PER COIN (sorted by PnL asc, live closed only)")
    coin = closed.groupby("coin_symbol").agg(
        n=("pnl_net", "count"),
        wins=("pnl_net", lambda x: (x>0).sum()),
        pnl=("pnl_net", lambda x: round(x.sum(),2))
    )
    coin["wr"] = (coin["wins"] / coin["n"] * 100).round(1)
    coin = coin.sort_values("pnl")
    for c, r in coin.iterrows():
        print(f"  {c:12s}: {int(r.n):3d} trades | WR {r.wr:5.1f}% | PnL ${r.pnl:+8.2f}")
    print()

    # summary bad coins
    bad_coins = coin[coin["pnl"] < 0].sort_values("pnl")
    if len(bad_coins):
        print("  Coins with negative PnL (live):")
        for c, r in bad_coins.iterrows():
            print(f"    {c}: ${r.pnl:+.2f} ({int(r.n)} trades)")
    print()

print("## 5. OPEN POSITIONS (live only)")
openl = live[live["status"] == "open"]
print(f"Count: {len(openl)}")
if len(openl) > 0:
    for _, r in openl.iterrows():
        print(f"  {r['opened_at'][:19]} | {r['coin_symbol']:12s} | {r['direction']:5s} | entry {r['entry_price']:.6g} | leverage {r.get('leverage', '?')}")
print()

print("="*70)
print("Note: These are REAL LIVE (is_live=1) results from production DB.")
print("Compare to research holdout: WR 68%+ PF 2.79 for widyawardhana v2")
print("="*70)
