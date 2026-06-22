import sys
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from config import (
    TRAINING_COINS, TRAIN_START, TRAIN_END,
    BINANCE_BASE_URL, SLEEP_BETWEEN_REQUESTS,
    SLEEP_ON_RATE_LIMIT, MAX_RETRIES, RETRY_BACKOFF_BASE
)
from core.binance_client import BinanceClient
from core.fetchers import fetch_funding_rate
from core.utils import setup_logger

logger = setup_logger("fetch_funding_only")

def main():
    print(f"Mengunduh data Funding Rate untuk 20 koin...")
    print(f"Periode: {TRAIN_START.date()} s.d. {TRAIN_END.date()}")
    
    # Inisialisasi client
    client = BinanceClient(
        base_url         = BINANCE_BASE_URL,
        sleep_between    = SLEEP_BETWEEN_REQUESTS,
        sleep_rate_limit = SLEEP_ON_RATE_LIMIT,
        max_retries      = MAX_RETRIES,
        backoff_base     = RETRY_BACKOFF_BASE,
        verify_ssl       = False
    )
    
    if not client.test_connection():
        print("Koneksi ke Binance gagal. Pastikan VPN/WARP sudah aktif.")
        return

    print("Koneksi OK. Mulai mengunduh...")
    
    success_coins = []
    failed_coins = []
    
    # Hapus sub-progress khusus funding_rate dari progress file jika ada,
    # atau kita abaikan progress agar force fetch ulang
    for i, symbol in enumerate(TRAINING_COINS, 1):
        print(f"\n[{i}/{len(TRAINING_COINS)}] Fetching {symbol}...")
        
        # Hapus file lama jika ada agar benar-benar fresh
        raw_dir = ROOT / "data" / "training"
        fr_path = raw_dir / "funding_rate" / f"{symbol}_8h.parquet"
        if fr_path.exists():
            fr_path.unlink()
            print(f"  ↳ Menghapus file lama: {fr_path.name}")
            
        try:
            df = fetch_funding_rate(
                client        = client,
                symbol        = symbol,
                start         = TRAIN_START,
                end           = TRAIN_END,
                progress      = None,  # Pass None agar tidak ter-skip oleh cache progress lama
                funding_limit = 1000,
                raw_dir       = raw_dir
            )
            if df is not None:
                print(f"  ↳ Sukses: {len(df)} records disimpan.")
                success_coins.append(symbol)
            else:
                print(f"  ↳ Gagal: Mengembalikan None.")
                failed_coins.append(symbol)
        except Exception as e:
            print(f"  ↳ Error: {e}")
            failed_coins.append(symbol)

    print("\n" + "="*50)
    print("DOWNLOAD SELESAI")
    print(f"Sukses ({len(success_coins)}): {success_coins}")
    print(f"Gagal  ({len(failed_coins)}): {failed_coins}")
    print("="*50)

if __name__ == "__main__":
    main()
