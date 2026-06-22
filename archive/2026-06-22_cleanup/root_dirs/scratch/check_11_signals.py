from tools.live_db_bridge import load_signals
import pandas as pd

sigs = load_signals()
sigs['signal_time'] = pd.to_datetime(sigs['signal_time'])

non_flat = sigs[sigs['direction'] != 'FLAT']

# Coin yang pernah non-FLAT (semua waktu)
print('=== Coin yang pernah generate non-FLAT signal ===')
coin_count = non_flat.groupby('coin_symbol')['direction'].count().sort_values(ascending=False)
print(coin_count.to_string())
print(f'\nTotal: {len(coin_count)} coin pernah non-FLAT dari {sigs["coin_symbol"].nunique()} coin aktif')

# Coin yang TIDAK PERNAH non-FLAT
never = set(sigs['coin_symbol'].unique()) - set(non_flat['coin_symbol'].unique())
print(f'\nCoin yang TIDAK PERNAH non-FLAT: {sorted(never)}')

# Proba LGBM terbaru per coin (untuk lihat siapa yang paling dekat threshold)
print()
print('=== LGBM proba LONG terbaru per coin (threshold ~0.59) ===')
import json
latest_per_coin = sigs.sort_values('signal_time').groupby('coin_symbol').last()
rows = []
for sym, row in latest_per_coin.iterrows():
    snap = {}
    if row['feature_snapshot']:
        try: snap = json.loads(row['feature_snapshot'])
        except: pass
    p = snap.get('_lgbm_proba')
    if p and len(p) == 3:
        rows.append({'coin': sym, 'p_long': p[2], 'p_flat': p[1], 'p_short': p[0]})

df_p = pd.DataFrame(rows).sort_values('p_long', ascending=False)
print(df_p.to_string(index=False))
