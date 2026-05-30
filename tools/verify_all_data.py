import pandas as pd
from pathlib import Path
import sys

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from config import TRAINING_COINS

def main():
    print("=== VERIFIKASI INTEGRITAS DATA SELURUH KOIN ===")
    
    # 1. Check Macro Data
    btc_dom_path = ROOT / "data" / "training" / "macro" / "btc_dominance.parquet"
    fg_path = ROOT / "data" / "training" / "macro" / "fear_greed_index.parquet"
    
    macro_ok = True
    for path, name in [(btc_dom_path, "BTC Dominance"), (fg_path, "Fear & Greed Index")]:
        if not path.exists():
            print(f"[ERROR] {name}: File TIDAK ditemukan!")
            macro_ok = False
        else:
            df = pd.read_parquet(path)
            null_count = df.isnull().sum().sum()
            # Fear & greed has 1 null from daily resampling, which is fine
            if null_count > 1:
                print(f"[WARN] {name}: Ditemukan {null_count} nilai Null/NaN.")
            else:
                print(f"[OK] {name}: OK ({len(df):,} baris, {null_count} Null).")
                
    # 2. Check Funding Rate per Coin
    print("\nChecking Funding Rate per symbol:")
    missing_coins = []
    invalid_coins = []
    
    for idx, symbol in enumerate(TRAINING_COINS, 1):
        fr_path = ROOT / "data" / "training" / "funding_rate" / f"{symbol}_8h.parquet"
        if not fr_path.exists():
            print(f"  [{idx:2d}/{len(TRAINING_COINS)}] {symbol:<14} [ERROR] File TIDAK ditemukan!")
            missing_coins.append(symbol)
            continue
            
        try:
            df = pd.read_parquet(fr_path)
            null_fr = df["funding_rate"].isnull().sum()
            all_zero = (df["funding_rate"] == 0.0).all()
            
            if len(df) == 0:
                print(f"  [{idx:2d}/{len(TRAINING_COINS)}] {symbol:<14} [ERROR] Data kosong (0 baris)!")
                invalid_coins.append(symbol)
            elif null_fr > 0:
                print(f"  [{idx:2d}/{len(TRAINING_COINS)}] {symbol:<14} [WARN] Ditemukan {null_fr} Null pada funding_rate.")
                invalid_coins.append(symbol)
            elif all_zero:
                print(f"  [{idx:2d}/{len(TRAINING_COINS)}] {symbol:<14} [ERROR] Data 100% bernilai 0.0!")
                invalid_coins.append(symbol)
            else:
                # Valid
                print(f"  [{idx:2d}/{len(TRAINING_COINS)}] {symbol:<14} [OK] OK ({len(df):,} baris, min={df['funding_rate'].min():.6f}, max={df['funding_rate'].max():.6f})")
        except Exception as e:
            print(f"  [{idx:2d}/{len(TRAINING_COINS)}] {symbol:<14} [ERROR] Error membaca file: {e}")
            invalid_coins.append(symbol)
            
    print("\n=== RINGKASAN VERIFIKASI ===")
    if macro_ok and len(missing_coins) == 0 and len(invalid_coins) == 0:
        print("[SUCCESS] SEMUA DATA VALID & SIAP UNTUK PIPELINE!")
    else:
        print("[FAILED] DITEMUKAN MASALAH DATA. Harap cek log di atas.")
        if missing_coins:
            print(f"  - File missing: {missing_coins}")
        if invalid_coins:
            print(f"  - Data invalid: {invalid_coins}")

if __name__ == "__main__":
    main()
