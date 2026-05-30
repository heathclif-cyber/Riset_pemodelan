import pandas as pd
from pathlib import Path

ROOT = Path(__file__).parent.parent

def check_file(path: Path, name: str):
    print(f"\n=== Memeriksa {name} ===")
    if not path.exists():
        print(f"File TIDAK ditemukan: {path}")
        return
    
    try:
        df = pd.read_parquet(path)
        print(f"File ditemukan di: {path}")
        print(f"Jumlah baris: {len(df):,}")
        print(f"Rentang waktu: {df.index.min()} s.d. {df.index.max()}")
        print(f"Jumlah baris Null/NaN:\n{df.isnull().sum()}")
        print(f"Statistik deskriptif:\n{df.describe()}")
        print("\nSampel data (5 baris pertama):")
        print(df.head(5))
    except Exception as e:
        print(f"Error saat membaca file: {e}")

def main():
    # 1. BTC Dominance
    btc_dom_path = ROOT / "data" / "training" / "macro" / "btc_dominance.parquet"
    check_file(btc_dom_path, "BTC Dominance")
    
    # 2. Fear & Greed Index
    fear_greed_path = ROOT / "data" / "training" / "macro" / "fear_greed_index.parquet"
    check_file(fear_greed_path, "Fear & Greed Index")
    
    # 3. Funding Rate (ambil sampel SOLUSDT)
    funding_sol_path = ROOT / "data" / "training" / "funding_rate" / "SOLUSDT_8h.parquet"
    check_file(funding_sol_path, "Funding Rate (SOLUSDT)")

if __name__ == "__main__":
    main()
