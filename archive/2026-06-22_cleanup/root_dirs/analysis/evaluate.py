"""Trade history evaluator. Usage: python analysis/evaluate.py <csv_path>"""
import pandas as pd, numpy as np, sys

path = sys.argv[1] if len(sys.argv) > 1 else 'reports/trade_history_holdout.csv'
df = pd.read_csv(path)

# Detect & normalize columns
cols = df.columns.tolist()
MAP = {
    'entry_time': 'Opened', 'exit_time': 'Closed',
    'coin': 'Coin', 'outcome': 'Exit Reason', 'entry_price': 'Entry',
    'exit_price': 'Exit', 'net_pnl': 'PnL ($)', 'direction': 'Direction',
}
for old, new in MAP.items():
    if old in cols and new not in cols:
        df.rename(columns={old: new}, inplace=True)

# Parse
if 'Opened' in df.columns:
    df['Opened'] = pd.to_datetime(df['Opened'])
    df['Closed'] = pd.to_datetime(df['Closed'], errors='coerce')

for c in ['PnL ($)']:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce')

if 'PnL (%)' in df.columns:
    df['PnL (%)'] = pd.to_numeric(df['PnL (%)'], errors='coerce')

# Filter closed only
if 'Status' in df.columns:
    closed = df[df['Status'] == 'closed'].copy()
elif 'Exit Reason' in df.columns:
    closed = df[df['Exit Reason'].notna()].copy()
elif 'Closed' in df.columns:
    closed = df[df['Closed'].notna()].copy()
else:
    closed = df.copy()

if len(closed) == 0:
    print("No closed trades found."); sys.exit(0)

wins = closed[closed['PnL ($)'] > 0]
losses = closed[closed['PnL ($)'] <= 0]
wr = len(wins) / len(closed) * 100
gross = wins['PnL ($)'].sum()
gross_loss = abs(losses['PnL ($)'].sum())
pf = gross / gross_loss if gross_loss > 0 else float('inf')

print(f"Trades: {len(closed)} closed", end="")
if 'Status' in df.columns:
    print(f" + {len(df[df['Status']=='open'])} open", end="")
print(f"\nPeriod: {df['Opened'].min().date()} to {df['Opened'].max().date()}")
print(f"Net PnL: ${closed['PnL ($)'].sum():.2f} | WR: {wr:.1f}% | PF: {pf:.2f} | Gross: +${gross:.0f} / -${gross_loss:.0f}")
if 'PnL (%)' in df.columns:
    print(f"Avg Win: +${wins['PnL ($)'].mean():.2f} ({wins['PnL (%)'].mean():.1f}%) | Avg Loss: -${abs(losses['PnL ($)'].mean()):.2f} ({abs(losses['PnL (%)'].mean()):.1f}%)")
else:
    print(f"Avg Win: +${wins['PnL ($)'].mean():.2f} | Avg Loss: -${abs(losses['PnL ($)'].mean()):.2f}")
print(f"Best: +${closed['PnL ($)'].max():.2f} | Worst: -${abs(closed['PnL ($)'].min()):.2f}")

# By model
if 'Model' in closed.columns:
    print("\n-- By Model --")
    for m in closed['Model'].unique():
        s = closed[closed['Model'] == m]; w = s[s['PnL ($)'] > 0]; l = s[s['PnL ($)'] <= 0]
        pf_m = abs(w['PnL ($)'].sum() / l['PnL ($)'].sum()) if len(l) > 0 else float('inf')
        print(f"  {str(m):12s} n={len(s):3d} WR={len(w)/len(s)*100:.1f}% Net=${s['PnL ($)'].sum():7.2f} PF={pf_m:.2f}")

# By exit reason
exit_col = 'Exit Reason' if 'Exit Reason' in closed.columns else 'ExitReason'
if exit_col in closed.columns:
    print(f"\n-- By Exit --")
    for e in closed[exit_col].dropna().unique():
        s = closed[closed[exit_col] == e]; w = s[s['PnL ($)'] > 0]
        wr_e = len(w)/len(s)*100 if len(s) > 0 else 0
        print(f"  {str(e):15s} n={len(s):3d} WR={wr_e:.1f}% Net=${s['PnL ($)'].sum():7.2f}")

# By coin
if 'Coin' in closed.columns:
    coin_col = 'Coin'
else:
    coin_col = 'coin' if 'coin' in closed.columns else None

if coin_col:
    print("\n-- By Direction --")
    if 'Direction' in closed.columns:
        for d in closed['Direction'].unique():
            s = closed[closed['Direction'] == d]; w = s[s['PnL ($)'] > 0]
            print(f"  {str(d):5s} n={len(s):3d} WR={len(w)/len(s)*100:.1f}% Net=${s['PnL ($)'].sum():7.2f}")

    print("\n-- By Coin (bottom 5 / top 5 by WR) --")
    coins = []
    for c in closed[coin_col].unique():
        s = closed[closed[coin_col] == c]
        if len(s) >= 2:
            wr_c = (s['PnL ($)'] > 0).sum()/len(s)*100
            coins.append((c, len(s), wr_c, s['PnL ($)'].sum()))
    coins.sort(key=lambda x: x[2])
    for c in coins[:5] + coins[-5:]:
        print(f"  {str(c[0]):15s} n={c[1]:2d} WR={c[2]:.0f}% Net=${c[3]:.2f}")

# Daily (last 14)
if 'Opened' in closed.columns:
    print("\n-- Daily (last 14) --")
    closed['Date'] = closed['Opened'].dt.date
    for d, grp in closed.groupby('Date'):
        if (closed['Date'].max() - d).days <= 14:
            net_d = grp['PnL ($)'].sum()
            wr_d = (grp['PnL ($)'] > 0).sum()/len(grp)*100
            print(f"  {d}  n={len(grp):2d}  net=${net_d:7.2f}  WR={wr_d:.0f}%")

# Streaks
if 'Opened' in closed.columns:
    print("\n-- Streaks --")
    pnl = np.sign(closed.sort_values('Opened')['PnL ($)'].values)
    streaks = []; cur = 1
    for i in range(1, len(pnl)):
        if pnl[i] == pnl[i-1]: cur += 1
        else: streaks.append((pnl[i-1], cur)); cur = 1
    streaks.append((pnl[-1], cur))
    for s, label in [(1, 'WIN'), (-1, 'LOSS')]:
        ss = [x[1] for x in streaks if x[0] == s]
        if ss: print(f"  {label} streaks: max={max(ss)} avg={np.mean(ss):.1f}")
