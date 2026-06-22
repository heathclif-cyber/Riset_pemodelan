"""
tools/record_liquidations.py — Perekam Data Likuidasi Global Real-Time

Mendengarkan data force orders terlikuidasi dari Binance Futures via WebSocket:
  wss://fstream.binance.com/ws/!forceOrder@arr

Menyimpan hasil rekaman ke database SQLite lokal secara real-time:
  data/raw/global_liquidations.db
"""

import os
import sys
import json
import time
import sqlite3
import logging
import ssl
from datetime import datetime, timezone
from pathlib import Path
import websocket

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("record_liquidations")

# Paths & Settings
ROOT_DIR = Path(__file__).parent.parent
DB_DIR = ROOT_DIR / "data" / "raw"
DB_PATH = DB_DIR / "global_liquidations.db"
WS_URL = "wss://fstream.binance.com/ws/!forceOrder@arr"
RECONNECT_DELAY = 5.0  # seconds

def init_db():
    """Inisialisasi database SQLite dan tabel likuidasi."""
    DB_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()
    
    # Buat tabel jika belum ada
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS liquidations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp INTEGER,
            symbol TEXT,
            side TEXT,
            price REAL,
            quantity REAL,
            usd_value REAL
        )
    """)
    
    # Buat indeks untuk performa query
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_liq_symbol ON liquidations (symbol)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_liq_timestamp ON liquidations (timestamp)")
    
    conn.commit()
    conn.close()
    logger.info(f"Database SQLite terinisialisasi di {DB_PATH}")

def save_liquidation(timestamp: int, symbol: str, side: str, price: float, quantity: float):
    """Menyimpan satu record likuidasi ke database."""
    usd_value = price * quantity
    try:
        conn = sqlite3.connect(str(DB_PATH))
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO liquidations (timestamp, symbol, side, price, quantity, usd_value)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (timestamp, symbol, side, price, quantity, usd_value)
        )
        conn.commit()
        conn.close()
    except Exception as e:
        logger.error(f"Gagal menyimpan ke database: {e}")

def on_message(ws, message):
    try:
        payload = json.loads(message)
        
        # Binance stream can push with or without "data" wrapper depending on stream path
        event_data = payload.get("data", payload) if isinstance(payload, dict) else payload
        
        if not isinstance(event_data, dict) or event_data.get("e") != "forceOrder":
            return
            
        order = event_data.get("o", {})
        symbol = order.get("s")
        side = order.get("S")
        price = float(order.get("p", 0.0))
        qty = float(order.get("q", 0.0))
        # Gunakan transaction time (T) jika tersedia, fallback ke event time (E)
        ts = int(order.get("T", event_data.get("E", int(time.time() * 1000))))
        
        usd_value = price * qty
        dt = datetime.fromtimestamp(ts / 1000, tz=timezone.utc)
        
        # Simpan ke SQLite
        save_liquidation(ts, symbol, side, price, qty)
        
        # Log jika nilainya signifikan (misal > $1,000 USD) untuk mereduksi log spam,
        # tetapi simpan SEMUA likuidasi ke DB.
        if usd_value >= 1000.0:
            logger.info(
                f"🔥 LIQUIDATION: {symbol:<9} | {side:<4} | Price: {price:<9.2f} | "
                f"Qty: {qty:<10.3f} | USD: ${usd_value:,.2f} | Time: {dt.strftime('%H:%M:%S')}"
            )
            
    except Exception as e:
        logger.error(f"Error parsing message: {e}")

def on_error(ws, error):
    logger.error(f"WebSocket Error: {error}")

def on_close(ws, close_status_code, close_msg):
    logger.warning(f"Koneksi WebSocket terputus: {close_status_code} - {close_msg}")

def on_open(ws):
    logger.info("✅ Tersambung ke Binance Futures Global Liquidation Stream")

def run_recorder():
    """Menjalankan WebSocket logger dengan auto-reconnection loop."""
    init_db()
    
    # Disable websocket trace logs untuk menjaga kebersihan output
    websocket.enableTrace(False)
    
    while True:
        try:
            logger.info(f"Menghubungkan ke stream: {WS_URL} ...")
            ws = websocket.WebSocketApp(
                WS_URL,
                on_open=on_open,
                on_message=on_message,
                on_error=on_error,
                on_close=on_close
            )
            # Run forever blocks until connection is closed, bypass SSL validation for ISP redirects
            ws.run_forever(
                ping_interval=30,
                ping_timeout=10,
                sslopt={"cert_reqs": ssl.CERT_NONE, "check_hostname": False}
            )
        except KeyboardInterrupt:
            logger.info("Program dihentikan oleh pengguna (KeyboardInterrupt).")
            break
        except Exception as e:
            logger.error(f"Error di run loop: {e}")
            
        logger.info(f"Mencoba menyambung kembali dalam {RECONNECT_DELAY} detik...")
        time.sleep(RECONNECT_DELAY)

if __name__ == "__main__":
    try:
        run_recorder()
    except KeyboardInterrupt:
        logger.info("Perekam likuidasi dihentikan.")
        sys.exit(0)
