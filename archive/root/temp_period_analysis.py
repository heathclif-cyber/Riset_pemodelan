import pandas as pd
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('livetrade.csv')
closed = df[df['Status'] == 'closed'].copy()
closed['Opened'] = pd.to_datetime(closed['Opened'])
closed = closed.sort_values('Opened')

print('=== PERFORMANCE SPLIT BY PERIOD ===')
print()

early = closed[closed['Opened'] < '2026-05-15']
mid = closed[(closed['Opened'] >= '2026-05-15') & (closed['Opened'] < '2026-05-25')]
late = closed[closed['Opened'] >= '2026-05-25']

for name, period in [('Early (pre 15 May - v2 dominant)', early), 
                     ('Mid (15-24 May - transition)', mid), 
                     ('Late (25-31 May - newer configs)', late)]:
    if len(period) == 0: continue
    wins = (period['PnL ($)'] > 0).sum()
    wr = wins / len(period) * 100
    pnl = period['PnL ($)'].sum()
    print(f'{name}')
    print(f'  Trades: {len(period):3d} | WR: {wr:5.1f}% | PnL: ${pnl:7.2f} | Avg: ${pnl/len(period):6.2f}')
    print(f'  Models: {period["Model"].value_counts().to_dict()}')
    print()

print('=== SL HIT RATE BY PERIOD ===')
for name, period in [('Early', early), ('Mid', mid), ('Late', late)]:
    if len(period) == 0: continue
    sl = period[period['Exit Reason'] == 'sl_hit']
    sl_rate = len(sl) / len(period) * 100 if len(period) > 0 else 0
    loss = sl['PnL ($)'].sum() if len(sl) > 0 else 0
    print(f'{name:6s}: SL hits = {len(sl):2d}/{len(period):3d} ({sl_rate:5.1f}%) | Loss from SL: ${loss:7.2f}')

print()
print('=== LATEST 20 TRADES (most recent) ===')
latest = closed.tail(20)
print(f'PnL: ${latest["PnL ($)"].sum():.2f} | WR: {(latest["PnL ($)"] > 0).sum()/len(latest)*100:.1f}%')
print('Models:', latest['Model'].value_counts().to_dict())
print('Exit reasons:', latest['Exit Reason'].value_counts().to_dict())