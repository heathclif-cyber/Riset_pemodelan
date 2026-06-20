import pandas as pd

p = "models/runs/ic32_regime_v1/holdout_scale_in_daily_apr_jun26.csv"
d = pd.read_csv(p).sort_values("date")
d["bad"] = d["pnl"] < 0

streaks = []
cur = []
for _, r in d.iterrows():
    if r["bad"]:
        cur.append(r)
    else:
        if cur:
            streaks.append(cur)
            cur = []
if cur:
    streaks.append(cur)

print("Max losing TRADING-day streak:", max(len(s) for s in streaks))
print("Streak length counts:", {n: sum(1 for s in streaks if len(s) == n) for n in sorted({len(s) for s in streaks})})
print()
print("All streaks with 2+ consecutive losing trading days:")
for s in streaks:
    if len(s) >= 2:
        tot = sum(x["pnl"] for x in s)
        print(f"  {len(s)} days | {s[0]['date']} to {s[-1]['date']} | combined PnL {tot:+.2f}")
        for x in s:
            print(f"    {x['date']}  {int(x['trades'])} trd  pnl {x['pnl']:+.2f}")