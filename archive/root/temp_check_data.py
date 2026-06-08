from config import PROC_DIR, LABEL_DIR

print('=== Checking engineered data availability ===')
print(f'PROC_DIR: {PROC_DIR}')
print(f'LABEL_DIR: {LABEL_DIR}')
print()

test_coins = ['SOLUSDT', 'BTCUSDT', 'ETHUSDT', 'DOGEUSDT']

print('In PROC_DIR (engineered.parquet):')
for coin in test_coins:
    path = PROC_DIR / f'{coin}_engineered.parquet'
    print(f'  {coin}: {"EXISTS" if path.exists() else "NOT FOUND"}')

print()
print('In LABEL_DIR (features_v3.parquet):')
for coin in test_coins:
    path = LABEL_DIR / f'{coin}_features_v3.parquet'
    print(f'  {coin}: {"EXISTS" if path.exists() else "NOT FOUND"}')
