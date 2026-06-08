import pandas as pd
import numpy as np

df = pd.read_csv('livetrade.csv')
closed = df[df['Status'] == 'closed'].copy()

print("=" * 70)
print("LIVETRADE.CSV ANALYSIS — Real Performance vs EXPERIMENTS.md Backtests")
print("=" * 70)

print("\n=== OVERALL SUMMARY ===")
print(f"Total records: {len(df)}")
print(f"Closed trades: {(df['Status'] == 'closed').sum()}")
print(f"Open positions: {(df['Status'] == 'open').sum()}")
print(f"Date range: {df['Opened'].min()} to {df['Opened'].max()}")

print("\n=== MODEL DISTRIBUTION (all records) ===")
print(df['Model'].value_counts().to_string())

print("\n" + "=" * 70)
print("CLOSED TRADES PERFORMANCE")
print("=" * 70)

print(f"\nTotal closed: {len(closed)}")
print(f"Total PnL: ${closed['PnL ($)'].sum():.2f}")
print(f"Mean PnL/trade: ${closed['PnL ($)'].mean():.2f}")
print(f"Median PnL: ${closed['PnL ($)'].median():.2f}")

wins = (closed['PnL ($)'] > 0).sum()
losses = (closed['PnL ($)'] <= 0).sum()
wr = wins / len(closed) * 100
print(f"\nWins: {wins} | Losses: {losses} | Win Rate: {wr:.1f}%")

print("\n=== PER-MODEL PERFORMANCE (Closed Trades) ===")
model_stats = []
for model in closed['Model'].unique():
    mdf = closed[closed['Model'] == model]
    n = len(mdf)
    if n == 0: continue
    w = (mdf['PnL ($)'] > 0).sum()
    model_wr = w / n * 100
    total_pnl = mdf['PnL ($)'].sum()
    avg_pnl = mdf['PnL ($)'].mean()
    model_stats.append({
        'Model': model,
        'Trades': n,
        'WR': model_wr,
        'TotalPnL': total_pnl,
        'AvgPnL': avg_pnl
    })
    print(f"{model:15s} | {n:3d} trades | WR {model_wr:5.1f}% | PnL ${total_pnl:7.2f} | Avg ${avg_pnl:6.2f}")

print("\n=== EXIT REASON ANALYSIS ===")
print(closed['Exit Reason'].value_counts().to_string())

print("\n--- PnL by Exit Reason ---")
exit_pnl = closed.groupby('Exit Reason').agg(
    Trades=('PnL ($)', 'count'),
    TotalPnL=('PnL ($)', 'sum'),
    AvgPnL=('PnL ($)', 'mean'),
    WR=('PnL ($)', lambda x: (x > 0).sum() / len(x) * 100)
).round(2)
print(exit_pnl.to_string())

print("\n=== GUARDIAN EFFECTIVENESS ===")
guardian_exits = ['guardian_exit', 'guardian_momentum_exit']
is_g = closed['Exit Reason'].isin(guardian_exits)
g_count = is_g.sum()
g_pnl = closed[is_g]['PnL ($)'].sum()
g_wr = (closed[is_g]['PnL ($)'] > 0).sum() / g_count * 100 if g_count > 0 else 0

non_g = ~is_g & ~closed['Exit Reason'].isin(['migrated'])
ng_count = non_g.sum()
ng_pnl = closed[non_g]['PnL ($)'].sum()
ng_wr = (closed[non_g]['PnL ($)'] > 0).sum() / ng_count * 100 if ng_count > 0 else 0

print(f"Guardian exits (guardian_exit + momentum): {g_count} trades")
print(f"  Total PnL: ${g_pnl:.2f} | Win Rate: {g_wr:.1f}%")
print(f"Non-Guardian (TP/SL/Time/Manual): {ng_count} trades")
print(f"  Total PnL: ${ng_pnl:.2f} | Win Rate: {ng_wr:.1f}%")

print("\n=== SL HIT DAMAGE ===")
sl = closed[closed['Exit Reason'] == 'sl_hit']
print(f"SL hits: {len(sl)} trades")
print(f"Total loss from SL: ${sl['PnL ($)'].sum():.2f}")
print("SL hits by model:")
print(sl.groupby('Model').agg(Count=('PnL ($)', 'count'), Loss=('PnL ($)', 'sum')).to_string())

print("\n=== DIRECTION PERFORMANCE ===")
dir_stats = closed.groupby('Direction').agg(
    Trades=('PnL ($)', 'count'),
    TotalPnL=('PnL ($)', 'sum'),
    AvgPnL=('PnL ($)', 'mean'),
    WR=('PnL ($)', lambda x: (x > 0).sum() / len(x) * 100)
).round(2)
print(dir_stats.to_string())

print("\n=== RECENT 30 CLOSED TRADES ===")
recent = closed.tail(30)
print(f"PnL: ${recent['PnL ($)'].sum():.2f}")
print(f"WR: {(recent['PnL ($)'] > 0).sum() / len(recent) * 100:.1f}%")
print(f"Models in recent: {recent['Model'].value_counts().to_dict()}")

print("\n=== OPEN POSITIONS ===")
open_pos = df[df['Status'] == 'open']
print(open_pos[['Opened', 'Coin', 'Model', 'Direction', 'Conf', 'Entry', 'TP', 'SL']].to_string())

print("\n" + "=" * 70)
print("KEY INSIGHT: Live vs Backtest Gap")
print("=" * 70)
print("Backtest Guardian v3 (EXPERIMENTS.md): 88.9% WR, $169k on 21 coins / 11 months")
print(f"Live trading (this file): {wr:.1f}% WR, +${closed['PnL ($)'].sum():.2f} on ~27 days")
print("SL hits are destroying performance (32 hits = -$243.60)")
print("Guardian is helping but many trades hit SL before it can act.")
