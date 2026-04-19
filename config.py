"""
config.py — Sentralisasi semua parameter proyek
Edit file ini untuk mengubah parameter training, fetch, atau feature engineering.
"""

from datetime import datetime, timezone
from pathlib import Path

# ─── Paths ───────────────────────────────────────────────────────────────────
ROOT_DIR   = Path(__file__).parent
DATA_DIR   = ROOT_DIR / "data"
RAW_DIR    = DATA_DIR / "raw"
PROC_DIR   = DATA_DIR / "processed"
LABEL_DIR  = DATA_DIR / "labeled"
MODEL_DIR  = ROOT_DIR / "models"
REPORT_DIR = ROOT_DIR / "reports"

# ─── Koin ────────────────────────────────────────────────────────────────────
# Koin training (5 koin awal)
TRAINING_COINS = [
    "SOLUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "DOGEUSDT",
]

# Koin baru untuk generalisasi (15 koin tambahan)
NEW_COINS = [
    "TONUSDT", "ADAUSDT", "TRXUSDT", "SHIBUSDT", "AVAXUSDT",
    "LINKUSDT", "DOTUSDT", "SUIUSDT", "POLUSDT", "NEARUSDT",
    "PEPEUSDT", "TAOUSDT", "APTOSUSDT", "ARBUSDT", "WLFIUSDT",
]

# Semua 20 koin
ALL_COINS = TRAINING_COINS + NEW_COINS

# Symbol → integer mapping (digunakan sebagai fitur 'symbol')
SYMBOL_MAP = {coin: i for i, coin in enumerate(ALL_COINS)}

# ─── Periode Data ─────────────────────────────────────────────────────────────
# Periode training (5 koin awal)
TRAIN_START = datetime(2022, 1, 1, tzinfo=timezone.utc)
TRAIN_END   = datetime(2025, 4, 1, tzinfo=timezone.utc)

# Periode koin baru (lebih pendek, mulai Apr 2023)
NEW_COINS_START = datetime(2023, 4, 1, tzinfo=timezone.utc)
NEW_COINS_END   = datetime(2025, 4, 1, tzinfo=timezone.utc)

# Alias untuk kompatibilitas
START_DATE = TRAIN_START
END_DATE   = TRAIN_END

# ─── Binance API ─────────────────────────────────────────────────────────────
BINANCE_BASE_URL    = "https://fapi.binance.com"
SLEEP_BETWEEN_REQUESTS = 0.12   # detik antar request
SLEEP_ON_RATE_LIMIT    = 60.0   # detik saat kena rate limit
MAX_RETRIES            = 3
RETRY_BACKOFF_BASE     = 2.0    # exponential backoff base

# Fetch limits per request
KLINE_LIMIT        = 1500
OI_LIMIT           = 500
FUNDING_LIMIT      = 1000
TAKER_RATIO_LIMIT  = 500
LONG_SHORT_LIMIT   = 500

# ─── Timeframes ───────────────────────────────────────────────────────────────
KLINE_INTERVALS = ["15m", "1h", "4h", "1d"]

# ─── Feature Engineering ──────────────────────────────────────────────────────
# Triple Barrier Labeling
TP_ATR_MULT      = 2.0   # take-profit = entry + TP_ATR_MULT × ATR
SL_ATR_MULT      = 1.0   # stop-loss   = entry - SL_ATR_MULT × ATR
MAX_HOLDING_BARS = 48    # maksimum holding = 48 bar M15 = 12 jam

# Volume Profile
VP_WINDOW = 96    # 24 jam dalam bar M15
VP_BINS   = 50    # jumlah price bins

# FVG (Fair Value Gap)
FVG_MIN_GAP_ATR = 0.5   # minimum gap dalam satuan ATR

# Order Block (CATATAN: OB_price di-drop dari feature set setelah EDA)
OB_LOOKBACK = 30

# Swing High/Low detection
SWING_LOOKBACK = 5   # bar di kiri dan kanan

# Synthetic OI (dari CVD)
SYNTHETIC_OI_CVD_WINDOW  = 96    # 24 jam rolling mean
SYNTHETIC_OI_NORM_WINDOW = 672   # 1 minggu normalisasi

# ─── Training ─────────────────────────────────────────────────────────────────
N_FOLDS        = 8
PURGE_GAP_BARS = 5   # bar M15 yang di-purge antara train dan test

# LightGBM
LGBM_PARAMS = {
    "objective":         "multiclass",
    "num_class":         3,
    "n_estimators":      1000,
    "learning_rate":     0.05,
    "max_depth":         6,
    "num_leaves":        31,
    "min_child_samples": 50,
    "subsample":         0.8,
    "colsample_bytree":  0.8,
    "verbose":           -1,
    "n_jobs":            -1,
    "random_state":      42,
}
LGBM_EARLY_STOPPING = 50

# LSTM
LSTM_SEQ_LEN    = 32
LSTM_HIDDEN     = 128
LSTM_LAYERS     = 2
LSTM_DROPOUT    = 0.3
LSTM_EPOCHS     = 30
LSTM_PATIENCE   = 5
LSTM_BATCH_SIZE = 512
LSTM_LR         = 0.001

# Label encoding
LABEL_MAP     = {"SHORT": 0, "FLAT": 1, "LONG": 2}
LABEL_MAP_INV = {v: k for k, v in LABEL_MAP.items()}
NUM_CLASSES   = 3

# ─── ML Signal Thresholds ─────────────────────────────────────────────────────
CONFIDENCE_FULL = 0.75   # >= FULL SIZE entry
CONFIDENCE_HALF = 0.60   # >= HALF SIZE entry
# < CONFIDENCE_HALF → SKIP

# ─── Macro Data ───────────────────────────────────────────────────────────────
COINGECKO_BASE_URL = "https://api.coingecko.com/api/v3"
FEAR_GREED_URL     = "https://api.alternative.me/fng/"
SLEEP_COINGECKO    = 2.0   # detik antar request CoinGecko (rate limit)

# ─── 58 Feature Columns (urutan wajib, TANPA OB_price) ───────────────────────
FEATURE_COLS = [
    "open", "high", "low", "close", "volume",
    "volume_delta", "cvd", "buy_volume", "sell_volume",
    "MSB_BOS", "CHoCH", "bars_since_BOS",
    "FVG_up", "FVG_down",
    "Buy_Liq", "Sell_Liq", "SFP_sweep",
    "open_interest", "funding_rate",
    "ema_7_m15", "ema_21_m15", "ema_50_m15", "ema_200_m15",
    "ema_7_h4", "ema_21_h4", "ema_50_h4", "ema_200_h4",
    "rsi_6", "stochrsi_k", "stochrsi_d",
    "atr_14_m15", "atr_14_h4",
    "PDH", "PDL", "PWH", "PWL",
    "Fib_618", "Fib_786",
    "POC", "VAH", "VAL",
    "btc_dominance", "fear_greed", "market_session",
    "log_ret_1", "log_ret_5", "log_ret_20",
    "vol_ratio_20",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
    "time_to_funding_norm",
    "long_short_ratio", "long_account_pct", "short_account_pct",
    "taker_buy_sell_ratio",
    "symbol",
]
