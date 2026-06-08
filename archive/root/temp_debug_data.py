from config import PROC_DIR, TRAIN_CUTOFF_DATE
import pandas as pd

print("PROC_DIR:", PROC_DIR)
print("TRAIN_CUTOFF_DATE:", TRAIN_CUTOFF_DATE)
print()

test_coins = ["SOLUSDT", "BTCUSDT", "ETHUSDT"]

for coin in test_coins:
    path = PROC_DIR / f"{coin}_engineered.parquet"
    exists = path.exists()
    print(f"{coin}: exists = {exists}")
    
    if exists:
        df = pd.read_parquet(path)
        print(f"   rows: {len(df)}, cols: {len(df.columns)}")
        print(f"   has label: {'label' in df.columns}")
        if len(df) > 0:
            print(f"   date min: {df.index.min()}")
            print(f"   date max: {df.index.max()}")
        print()
