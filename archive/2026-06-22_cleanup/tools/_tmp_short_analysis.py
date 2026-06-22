import pandas as pd, numpy as np

df = pd.read_csv('models/runs/ic32_regime_v2/holdout_v1_vs_v2_trades.csv')
v2 = df[df['model']=='v2'].copy()
shorts = v2[v2['dir']=='SHORT']
longs  = v2[v2['dir']=='LONG']

print("=== KOLOM ===")
print(df.columns.tolist())
print()

# Bias arah
print("=== BIAS ARAH v2 holdout ===")
print(f"  LONG  trades: {len(longs)} ({len(longs)/len(v2)*100:.0f}%)")
print(f"  SHORT trades: {len(shorts)} ({len(shorts)/len(v2)*100:.0f}%)")
print()

# Per-coin SHORT performance
symbols = ['ARBUSDT','AVAXUSDT','ONDOUSDT','DOGEUSDT']
print("=== SHORT HOLDOUT per-coin ===")
for sym in symbols:
    if 'symbol' in shorts.columns:
        sub = shorts[shorts['symbol']==sym]
    else:
        sub = pd.DataFrame()
    if len(sub) > 0:
        wins = sub[sub['pnl']>0]
        wr = len(wins)/len(sub)*100
        avg_hold = sub['bars'].mean()
        avg_pnl = sub['pnl'].mean()
        print(f"  {sym}: {len(sub)} trades  WR={wr:.0f}%  hold={avg_hold:.0f}bar  avg_pnl={avg_pnl:.4f}")
    else:
        print(f"  {sym}: no data (sym col: {'symbol' in shorts.columns})")

print()
# SL distance jika ada
if 'sl_pct' in df.columns:
    print("=== SL DISTANCE ===")
    print(shorts['sl_pct'].describe())
elif 'entry' in df.columns and 'sl' in df.columns:
    df['sl_dist'] = abs(df['sl'] - df['entry']) / df['entry'] * 100
    print("=== SL DISTANCE % ===")
    print(df[df['dir']=='SHORT']['sl_dist'].describe())

print()
# MAE proxy: avg losing trade magnitude
losing = shorts[shorts['pnl']<0]
print("=== LOSS TRADES SHORT ===")
print(f"  Count: {len(losing)}")
print(f"  AvgLoss ATR: {losing['pnl'].mean():.4f}")
print(f"  p50 ATR: {losing['pnl'].median():.4f}")
print(f"  MaxLoss ATR: {losing['pnl'].min():.4f}")
print(f"  Avg hold (losers): {losing['bars'].mean():.1f} bars")
print()

# How many bars until typical SHORT resolution?
print("=== HOLD DISTRIBUTION SHORT (bars) ===")
for pct in [25, 50, 75, 90, 95]:
    print(f"  p{pct}: {shorts['bars'].quantile(pct/100):.0f} bars")
