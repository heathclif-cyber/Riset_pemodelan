import pandas as pd
import numpy as np

# ---- Run 092232 ----
print('=' * 70)
print('COMPARISON: Run 020556 vs Run 092232')
print('=' * 70)

for run_id in ['holdout_20260527_020556', 'holdout_20260527_092232']:
    label = 'BEFORE REVERT' if '020556' in run_id else 'AFTER REVERT (AKTIF)'
    df = pd.read_csv(f'reports/experiments/{run_id}_holdout_trade_history.csv')
    df['Opened'] = pd.to_datetime(df['Opened'])
    df = df[(df['Opened'] >= '2025-11-01') & (df['Opened'] < '2026-04-02')]

    pnl_col = 'PnL ($)'
    wins = df[df[pnl_col] > 0]
    losses = df[df[pnl_col] <= 0]
    wr = len(wins) / len(df) * 100

    print(f'\n### {run_id} -- {label}')
    print(f'Coins: {df["Coin"].nunique()} | Total Trades: {len(df)}')
    print(f'Wins: {len(wins)} | Losses: {len(losses)} | WR: {wr:.2f}%')
    print(f'Total PnL: ${df[pnl_col].sum():,.2f}')
    print(f'Avg Hold Bars: {df["Hold Bars"].mean():.1f} | Median: {df["Hold Bars"].median():.1f}')
    print(f'Avg Conf: {df["Conf"].mean():.3f}')

    # Direction
    for d in ['LONG', 'SHORT']:
        sub = df[df['Direction'] == d]
        w = (sub[pnl_col] > 0).sum()
        print(f'{d}: WR={w/len(sub)*100:.2f}% ({w}/{len(sub)}), PnL=${sub[pnl_col].sum():,.2f}')

    # Monthly
    df['Month'] = df['Opened'].dt.to_period('M')
    monthly = df.groupby('Month').agg(
        Trades=(pnl_col, 'count'),
        Wins=(pnl_col, lambda x: (x > 0).sum()),
        PnL=(pnl_col, 'sum'),
        WR=(pnl_col, lambda x: (x > 0).mean() * 100)
    ).sort_index()

    print(f'\n{"Month":<20s} {"Trades":>7s} {"Wins":>6s} {"PnL":>10s} {"WR":>8s}')
    for m, row in monthly.iterrows():
        print(f'{str(m):20s} {int(row.Trades):7d} {int(row.Wins):6d} ${row.PnL:9,.2f} {row.WR:7.2f}%')

    # Exit reason
    print('\n--- Exit Reason ---')
    exit_stats = df.groupby('Exit Reason').agg(
        Count=(pnl_col, 'count'),
        Wins=(pnl_col, lambda x: (x > 0).sum()),
        PnL=(pnl_col, 'sum'),
        WR=(pnl_col, lambda x: (x > 0).mean() * 100)
    )
    for e, row in exit_stats.iterrows():
        print(f'{e:25s} Count={int(row.Count):5d}  Wins={int(row.Wins):5d}  PnL=${row.PnL:9,.2f}  WR={row.WR:6.2f}%')

    # Max cons loss
    df_sorted = df.sort_values(['Coin', 'Opened'])
    max_cons_loss = 0
    curr_streak = 0
    for _, row in df_sorted.iterrows():
        if row[pnl_col] <= 0:
            curr_streak += 1
            max_cons_loss = max(max_cons_loss, curr_streak)
        else:
            curr_streak = 0
    print(f'Max Consecutive Loss: {max_cons_loss}')

    # Fees
    pos_size = 25 * 5
    fee_per_trade = pos_size * (0.0004 + 0.0005) * 2
    total_fee = fee_per_trade * len(df)
    gross_pnl = df[pnl_col].sum() + total_fee
    print(f'Total Fees: ${total_fee:,.2f} | Gross PnL: ${gross_pnl:,.2f}')

    # Top/bottom 5 coins
    coins = df.groupby('Coin').agg(
        Trades=(pnl_col, 'count'),
        Wins=(pnl_col, lambda x: (x > 0).sum()),
        PnL=(pnl_col, 'sum'),
        WR=(pnl_col, lambda x: (x > 0).mean() * 100)
    ).sort_values('PnL', ascending=False)
    print('\nTop 5 & Bottom 5 Coins:')
    for c, row in list(coins.head(5).iterrows()) + list(coins.tail(5).iterrows()):
        print(f'{c:15s} Trades={int(row.Trades):4d}  PnL=${row.PnL:8,.2f}  WR={row.WR:6.2f}%')

# ---- DELTA ----
print('\n' + '=' * 70)
print('DELTA (092232 - 020556)')
print('=' * 70)

for run_id in ['holdout_20260527_020556', 'holdout_20260527_092232']:
    df = pd.read_csv(f'reports/experiments/{run_id}_holdout_trade_history.csv')
    df['Opened'] = pd.to_datetime(df['Opened'])
    df = df[(df['Opened'] >= '2025-11-01') & (df['Opened'] < '2026-04-02')]
    pnl_col = 'PnL ($)'

    if '020556' in run_id:
        old_trades = len(df)
        old_pnl = df[pnl_col].sum()
        old_wr = (df[pnl_col] > 0).mean() * 100
    else:
        new_trades = len(df)
        new_pnl = df[pnl_col].sum()
        new_wr = (df[pnl_col] > 0).mean() * 100

print(f'Trades: {old_trades} -> {new_trades} (delta: {new_trades - old_trades:+d})')
print(f'WR: {old_wr:.2f}% -> {new_wr:.2f}% (delta: {new_wr - old_wr:+.2f}pp)')
print(f'PnL: ${old_pnl:,.2f} -> ${new_pnl:,.2f} (delta: ${new_pnl - old_pnl:+,.2f})')
