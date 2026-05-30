import sys
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from config import TRAIN_START, TRAIN_END
from core.fetchers import fetch_all_macro
from core.utils import setup_logger

logger = setup_logger("fetch_macro_only")

def main():
    print(f"Mengunduh data macro untuk periode: {TRAIN_START.date()} s.d. {TRAIN_END.date()}")
    
    # Kosongkan file lama atau force fetch dengan mereset progress key macro
    progress = {}
    
    # Hapus file lokal yang lama jika ada agar benar-benar ter-update
    btc_dom_path = ROOT / "data" / "training" / "macro" / "btc_dominance.parquet"
    fear_greed_path = ROOT / "data" / "training" / "macro" / "fear_greed_index.parquet"
    
    if btc_dom_path.exists():
        btc_dom_path.unlink()
        print(f"Menghapus file lama: {btc_dom_path.name}")
    if fear_greed_path.exists():
        fear_greed_path.unlink()
        print(f"Menghapus file lama: {fear_greed_path.name}")
        
    results = fetch_all_macro(TRAIN_START, TRAIN_END, progress=progress)
    
    print("\nHasil:")
    for key, df in results.items():
        if df is not None:
            print(f"- {key}: {len(df)} baris berhasil disimpan.")
        else:
            print(f"- {key}: GAGAL.")

if __name__ == "__main__":
    main()
