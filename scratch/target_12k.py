# -*- coding: utf-8 -*-
import math
import pandas as pd

tr = pd.read_excel(r"D:\Datatrade_ic32regime.xlsx", sheet_name="Trades_scale_in")
tr["ts_in"] = pd.to_datetime(tr["ts_in"])
tr["ts_out"] = pd.to_datetime(tr["ts_out"])
period_days = (tr["ts_in"].max() - tr["ts_in"].min()).days + 1
period_months = period_days / 30.44
total_pnl = float(tr["net_pnl"].sum())
monthly_holdout = total_pnl / period_months
monthly_oof = 5128.75 / 75

events = []
for _, r in tr.iterrows():
    events.append((r["ts_in"], float(r["modal_used"])))
    events.append((r["ts_out"], -float(r["modal_used"])))
events.sort(key=lambda x: x[0])
cur = peak = 0.0
samples = []
for i, (t, d) in enumerate(events):
    cur += d
    peak = max(peak, cur)
    if i + 1 < len(events):
        dt = (events[i + 1][0] - t).total_seconds() / 3600
        samples.append((cur, dt))
avg_concurrent = sum(c * h for c, h in samples) / sum(h for _, h in samples)


def months_to_target(start, target, monthly):
    r = monthly / start
    if r <= 0:
        return float("inf")
    return math.log(target / start) / math.log(1 + r)


def compound_timeline(start, monthly, target, max_m=48):
    rows = []
    bal = start
    for m in range(0, max_m + 1):
        rows.append((m, bal))
        if bal >= target:
            break
        bal += monthly * (bal / start)  # scale profit with bankroll
    return rows


target = 12000
print("RATES")
print(f"  holdout: ${monthly_holdout:.2f}/bln ({period_months:.1f} bln sample)")
print(f"  OOF:     ${monthly_oof:.2f}/bln (75 bln sample)")
print(f"  avg concurrent modal holdout: ${avg_concurrent:.2f}")
print()

print("TARGET: saldo akun $12,000 (compound, reinvest)")
for s in [210, 500, 1000, 2000, 5000]:
    mh = months_to_target(s, target, monthly_holdout)
    mo = months_to_target(s, target, monthly_oof)
    print(f"  start ${s:>5}: holdout {mh:5.1f} bln ({mh/12:4.1f} thn) | OOF {mo:5.1f} bln ({mo/12:4.1f} thn)")

print()
print("TARGET: profit kumulatif $12,000 (modal $10/trade tetap, tidak scale)")
print(f"  holdout: {target/monthly_holdout:5.0f} bln ({target/monthly_holdout/12:4.1f} thn)")
print(f"  OOF:     {target/monthly_oof:5.0f} bln ({target/monthly_oof/12:4.1f} thn)")

print()
print("KALENDER (compound holdout, start $210, asumsi rate holdout sustain)")
bal = 210.0
m = 0
while bal < target and m < 60:
    m += 1
    bal += monthly_holdout * (bal / 210.0)
print(f"  bulan ke-{m}: ~${bal:,.0f}")