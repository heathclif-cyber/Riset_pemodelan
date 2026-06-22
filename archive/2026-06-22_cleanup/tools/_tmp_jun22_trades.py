import pandas as pd

df = pd.read_csv('reports/experiments/2026-06-22_ic32_regime_v2_trades.csv')
df['entry_time'] = pd.to_datetime(df['entry_time'], utc=True)
df['exit_time']  = pd.to_datetime(df['exit_time'],  utc=True)

jun22 = pd.Timestamp('2026-06-22', tz='UTC')
jun23 = pd.Timestamp('2026-06-23', tz='UTC')

# trade yang entry di Jun 22, ATAU masih open dari sebelumnya dan exit di Jun 22
mask = (df['entry_time'] >= jun22) | ((df['exit_time'] >= jun22) & (df['entry_time'] < jun22))
d = df[mask].sort_values('entry_time').reset_index(drop=True)

wins = (d['net_pnl'] > 0).sum()
total = len(d)
pnl = d['net_pnl'].sum()
gp = d[d['net_pnl'] > 0]['net_pnl'].sum()
gl = d[d['net_pnl'] < 0]['net_pnl'].abs().sum()
pf = gp / gl if gl > 0 else float('inf')

print(f"Trades Jun 22 : {total}")
print(f"WR            : {wins}/{total} = {wins/total*100:.1f}%")
print(f"PF            : {pf:.3f}")
print(f"PnL           : ${pnl:.4f}")
print()

cols = ['coin','entry_time','exit_time','direction','entry_price','exit_price','rr','bars_held','net_pnl','outcome','lgbm_conf']
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 200)
pd.set_option('display.float_format', '{:.4f}'.format)
print(d[cols].to_string(index=True))
