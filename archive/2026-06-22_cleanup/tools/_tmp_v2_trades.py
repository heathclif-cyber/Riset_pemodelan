import pandas as pd

df = pd.read_csv('reports/experiments/2026-06-22_ic32_regime_v2_trades.csv')
df['entry_time'] = pd.to_datetime(df['entry_time'], utc=True)
df['exit_time']  = pd.to_datetime(df['exit_time'],  utc=True)

# 07:00 WITA Jun 22 = 23:00 UTC Jun 21
v2_start = pd.Timestamp('2026-06-21 23:00', tz='UTC')
end      = pd.Timestamp('2026-06-22 09:00', tz='UTC')

d = df[df['entry_time'] >= v2_start].sort_values('entry_time').reset_index(drop=True)

wins  = (d['net_pnl'] > 0).sum()
total = len(d)
pnl   = d['net_pnl'].sum()
gp    = d[d['net_pnl'] > 0]['net_pnl'].sum()
gl    = d[d['net_pnl'] < 0]['net_pnl'].abs().sum()
pf    = gp / gl if gl > 0 else float('inf')

print(f"== BACKTEST v2 sejak 07:00 WITA (23:00 UTC Jun 21) ==")
print(f"Trades  : {total}")
print(f"WR      : {wins}/{total} = {wins/total*100:.1f}%")
print(f"PF      : {pf:.3f}")
print(f"PnL     : ${pnl:.4f}")
print()

# konversi entry/exit ke WITA untuk display
d['entry_wita'] = d['entry_time'].dt.tz_convert('Asia/Makassar').dt.strftime('%m-%d %H:%M')
d['exit_wita']  = d['exit_time'].dt.tz_convert('Asia/Makassar').dt.strftime('%m-%d %H:%M')

cols = ['coin','entry_wita','exit_wita','direction','entry_price','exit_price','rr','bars_held','net_pnl','outcome','lgbm_conf']
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 220)
print(d[cols].to_string(index=True))
