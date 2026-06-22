# -*- coding: utf-8 -*-
import pandas as pd

tr = pd.read_excel(r"D:\Datatrade_ic32regime.xlsx", sheet_name="Trades_scale_in")
tr["ts_in"] = pd.to_datetime(tr["ts_in"])
tr["ts_out"] = pd.to_datetime(tr["ts_out"])
tr["hold_h"] = (tr["ts_out"] - tr["ts_in"]).dt.total_seconds() / 3600

period_days = (tr["ts_in"].max() - tr["ts_in"].min()).days + 1
period_months = period_days / 30.44
total_pnl = float(tr["net_pnl"].sum())
avg_modal = float(tr["modal_used"].mean())
sum_modal = float(tr["modal_used"].sum())

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

monthly_holdout = total_pnl / period_months
monthly_oof = 5128.75 / 75

bases = {
    "avg_concurrent_modal (realistic)": avg_concurrent,
    "peak_concurrent_modal": peak,
    "21_coins_x_10usd (max slots)": 210.0,
    "single_10usd_slot (min, unrealistic)": 10.0,
}

print("HOLDOUT scale_in stats")
print(f"  period_months={period_months:.2f} total_pnl={total_pnl:.2f}")
print(f"  monthly_pnl={monthly_holdout:.2f} avg_concurrent={avg_concurrent:.2f} peak={peak:.2f}")
print()
print("ROI projection (holdout rate)")
for name, cap in bases.items():
    print(f"\nCapital base: {name} = {cap:.2f} USD")
    for m in [3, 6, 12]:
        pnl = monthly_holdout * m
        roi = pnl / cap * 100
        print(f"  {m} bulan: PnL {pnl:8.2f} USD | ROI {roi:7.1f}%")

print("\nROI projection (OOF conservative rate)")
print(f"  monthly_pnl_oof={monthly_oof:.2f}")
for name, cap in bases.items():
    print(f"\nCapital base: {name} = {cap:.2f} USD")
    for m in [3, 6, 12]:
        pnl = monthly_oof * m
        roi = pnl / cap * 100
        print(f"  {m} bulan: PnL {pnl:8.2f} USD | ROI {roi:7.1f}%")