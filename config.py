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
TRAINING_COINS = [
    "SOLUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "DOGEUSDT",
]

NEW_COINS = [
    "TONUSDT", "ADAUSDT", "TRXUSDT", "1000SHIBUSDT", "AVAXUSDT",
    "LINKUSDT", "DOTUSDT", "SUIUSDT", "POLUSDT", "NEARUSDT",
    "1000PEPEUSDT", "TAOUSDT", "ARBUSDT",
]

ALL_COINS = TRAINING_COINS + NEW_COINS

SYMBOL_MAP = {coin: i for i, coin in enumerate(ALL_COINS)}

# ─── Periode Data ─────────────────────────────────────────────────────────────
# Semua koin — baik training maupun new coins — menggunakan periode yang sama.
# Koin yang listing setelah 2020 (SUI, TON, PEPE, TAO, ARB) akan otomatis
# mendapat data lebih pendek sesuai tanggal listing mereka di Binance.
TRAIN_START = datetime(2020, 1, 1, tzinfo=timezone.utc)
TRAIN_END   = datetime(2026, 4, 1, tzinfo=timezone.utc)

START_DATE = TRAIN_START
END_DATE   = TRAIN_END

# ─── Binance API ─────────────────────────────────────────────────────────────
BINANCE_BASE_URL       = "https://fapi.binance.com"
SLEEP_BETWEEN_REQUESTS = 0.12
SLEEP_ON_RATE_LIMIT    = 60.0
MAX_RETRIES            = 3
RETRY_BACKOFF_BASE     = 2.0

KLINE_LIMIT       = 1500
OI_LIMIT          = 500
FUNDING_LIMIT     = 1000
TAKER_RATIO_LIMIT = 500
LONG_SHORT_LIMIT  = 500

# ─── Timeframes ───────────────────────────────────────────────────────────────
KLINE_INTERVALS = ["1h", "4h", "1d"]

# ─── Feature Engineering ──────────────────────────────────────────────────────
TP_ATR_MULT      = 2.0       # TP untuk legacy path (non-swing)
SL_ATR_MULT      = 1.0       # SL untuk legacy path (non-swing)
MAX_HOLDING_BARS = 24        # bar H1 = 24 jam (sebelumnya 48)

# ── Swing-Based Labeling v3 ───────────────────────────────────────────────────
SWING_LABEL_MAX_HOLD = 24     # bar H1 = 24 jam (sebelumnya 48) — lever utama kurangi FLAT
SWING_LABEL_MIN_RR   = 1.2    # sebelumnya 1.5 — lever sekunder, lebih realistis untuk crypto
SWING_LABEL_MIN_TP   = 1.2    # sebelumnya 1.5 — lever utama, TP lebih mudah tercapai
SWING_LABEL_MAX_SL   = 3.0    # TETAP — SL ketat di crypto noisy meningkatkan false negatives
SWING_H4_LOOKBACK    = 5
SWING_ROLLING_BARS   = 24     # 24 jam rolling swing

# Volume Profile & FVG
VP_WINDOW       = 24
VP_BINS         = 50
FVG_MIN_GAP_ATR = 0.5
OB_LOOKBACK     = 30
SWING_LOOKBACK  = 5

# Synthetic OI (H1 Adjusted)
SYNTHETIC_OI_CVD_WINDOW  = 24
SYNTHETIC_OI_NORM_WINDOW = 168

# ─── Training & Purging ───────────────────────────────────────────────────────
N_FOLDS        = 8
PURGE_GAP_BARS = 5

# LightGBM Params (H1)
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

# LightGBM Params (H4 — binary: SHORT=0, LONG=1; FLAT ditentukan oleh threshold)
LGBM_H4_PARAMS = {
    "objective":         "binary",      # binary: hanya LONG vs SHORT (FLAT = below threshold)
    "n_estimators":      500,
    "learning_rate":     0.03,
    "max_depth":         6,             # dinaikkan dari 4 — lebih kompleks untuk regime H4
    "num_leaves":        31,            # dinaikkan dari 15
    "min_child_samples": 50,            # dinaikkan dari 30 — regulasi untuk cegah overfit
    "subsample":         0.8,
    "colsample_bytree":  0.7,
    "verbose":           -1,
    "n_jobs":            -1,
    "random_state":      42,
}

# ─── H4 Model Parameters ──────────────────────────────────────────────────────
# H4 menggunakan subset fitur dari FEATURE_COLS_V3 (30 fitur teratas)
# yang relevan untuk regime detection di timeframe lebih tinggi
H4_FEATURE_COLS = [
    # OHLCV base
    "open", "high", "low", "close", "volume",

    # EMA H4 (sangat relevan untuk H4 regime)
    "ema_7_h4", "ema_21_h4", "ema_50_h4", "ema_200_h4",

    # ATR H4
    "atr_14_h4",

    # Volume flow
    "cvd", "volume_delta",

    # Market structure
    "MSB_BOS", "CHoCH",

    # Momentum H4
    "rsi_h4", "rsi_divergence",

    # Trend dynamics (slope-based)
    "ema_21_slope_h4", "ema_50_slope_h4",
    "price_vs_ema_50_h4",
    "atr_percent_h4", "range_expansion_h4",
    "rsi_slope_h4",

    # Key levels
    "PDH", "PDL", "PWH", "PWL",

    # Market regime
    "h4_trend", "trend_strength", "vol_regime",

    # Higher Timeframe (D1) — trend makro, vol regime, alignment
    "ema_50_d1", "ema_200_d1",
    "ema_50_slope_d1", "ema_200_slope_d1",
    "price_vs_ema_50_d1",
    "atr_d1_percentile",
    "d1_trend", "d1_trend_strength",
    "htf_alignment",
    "d1_hh_hl_bias",

    # Smart money H4
    "cvd_div_h4", "cvd_slope_h4",

    # Macro
    "btc_dominance", "fear_greed", "market_session",

    # Returns
    "log_ret_1", "log_ret_5",

    # Open interest & funding
    "open_interest", "funding_rate",
]

# H4 Labeling — TP/SL proportionally larger for H4 timeframe
H4_SWING_LABEL_MIN_RR   = 0.6   # max theoretical RR = min_tp/max_sl = 2.0/3.0 ≈ 0.667
H4_SWING_LABEL_MIN_TP   = 2.0   # vs 1.2 di H1
H4_SWING_LABEL_MAX_SL   = 3.0   # sama dengan H1
H4_SWING_LABEL_MAX_HOLD = 6     # bar H4 = 24 jam (setara MAX_HOLDING_BARS H1)

# H4 CV — time-based boundaries (6 bar H4 purge ≈ 24 bar H1)
H4_N_FOLDS        = 8
H4_PURGE_GAP_BARS = 6   # 6 bar H4 ≈ 24 jam ≈ 24 bar H1

# H4 Calibration — dimatikan karena isotonic collapse (P50=0.518→1.000)
# Lihat AUDIT_REPORT.md § Isotonic Collapse
H4_USE_CALIBRATION = False

# ─── Hierarchical Decision Thresholds ─────────────────────────────────────────
# H4 Binary thresholds — binary model output: [prob_SHORT, prob_LONG]
# Distribusi probabilitas H4: P50≈0.506, P90≈0.572 → threshold 0.60 optimal.
# Target pass_rate ~8-15% (sebelumnya 0.65 → hanya 1.4%).
H4_BINARY_THRESHOLD_LONG  = 0.60  # diturunkan dari 0.65 (P90=0.572)
H4_BINARY_THRESHOLD_SHORT = 0.60  # diturunkan dari 0.65 (P90=0.572)
H4_BINARY_MARGIN          = 0.05  # bias hanya jika prob unggul >= margin atas lawan

# H4 Soft Filter — dinonaktifkan (arsitektur 2 model: LGBM + LSTM)
# H4 regime context kini ditangani langsung via fitur di LGBM:
# htf_alignment, d1_trend, trend_accel_4h, vol_price_confirm, dll.
H4_SOFT_FILTER_ENABLED      = False
H4_SOFT_ALIGN_BOOST         = 0.04
H4_SOFT_FLAT_PENALTY        = 0.015
H4_SOFT_OPPOSITE_PENALTY    = 0.035

# H4 Binary class weights — diturunkan dari 3.0 ke 1.5 (AUC rendah → overfit)
H4_BINARY_CLASS_WEIGHTS  = {0: 1.5, 1: 1.5}  # SHORT=0, LONG=1 (sebelumnya 3.0)

# H1 entry thresholds (3-class model tidak berubah)
LGBM_THRESHOLD_LONG  = 0.62   # LGBM minimum confidence untuk entry LONG
LGBM_THRESHOLD_SHORT = 0.62   # LGBM minimum confidence untuk entry SHORT
LSTM_CONFIRMATION_ENABLED = True  # LSTM digunakan sebagai confirmation vote
# ─── LSTM Soft Adjustment Penalties ────────────────────────────────────────────
# Tiered / absolute penalties menggantikan relative penalty (-0.15 × h1_conf)
# yang tidak adil untuk confidence moderate (lihat AUDIT_REPORT.md §2.4).
#
# Mode = "tiered":   penalty bervariasi berdasarkan margin di atas threshold
#   margin < 0.05 → heavy penalty  (borderline)
#   margin < 0.10 → medium penalty (moderate)
#   else         → light penalty   (confident)
#
# Mode = "absolute": penalty fixed terlepas dari h1_conf
#   agree_penalty diterapkan saat LSTM searah
#   neutral_penalty saat LSTM FLAT
#   opposite_penalty saat LSTM berlawanan arah
#
# Mode = "relative": original formula (0.05/0.05/0.15 × h1_conf) — dipertahankan
# untuk kompatibilitas.
LSTM_ADJUST_MODE         = "tiered"     # "relative" | "absolute" | "tiered"
LSTM_ADJUST_AGREE_BOOST  = 0.05         # boost saat agree (mode relative/absolute)
LSTM_ADJUST_NEUTRAL_PEN  = 0.05         # penalty saat LSTM FLAT
LSTM_ADJUST_OPPOSITE_PEN = 0.08         # penalty saat LSTM opposite (0.15 asli → 0.08)
# LSTM Params
LSTM_SEQ_LEN    = 16   # diturunkan dari 32 — lebih reaktif ke koreksi (32 jam → 16 jam)
LSTM_HIDDEN     = 128
LSTM_LAYERS     = 2
LSTM_DROPOUT    = 0.3
LSTM_EPOCHS     = 100
LSTM_PATIENCE   = 5
LSTM_BATCH_SIZE = 512
LSTM_LR         = 0.001

LABEL_MAP     = {"SHORT": 0, "FLAT": 1, "LONG": 2}
LABEL_MAP_INV = {v: k for k, v in LABEL_MAP.items()}
NUM_CLASSES   = 3

# ─── ML Signal Thresholds ─────────────────────────────────────────────────────
CONFIDENCE_FULL = 0.75
CONFIDENCE_HALF = 0.60

# ─── Signal Stability & Circuit Breaker (For Inference Deployment) ────────────
SIGNAL_FLIP_CONF_MIN       = 0.70
FLIP_CONFIRM_BARS          = 2
FLIP_COOLDOWN_SECS         = 1800
SAME_DIR_COOLDOWN_HOURS    = 2.5

VCB_ENABLED                = True
VCB_ATR_MULTIPLIER         = 3.0
VCB_LOOKBACK_BARS          = 24

MONITOR_POLL_INTERVAL_SECS = 300

# ─── Feature Columns v3 (H1 Base - 103 fitur) ────────────────────────────────
FEATURE_COLS_V3 = [
    # OHLCV base
    "open", "high", "low", "close", "volume",

    # Volume flow
    "volume_delta", "cvd", "buy_volume", "sell_volume",

    # Market structure
    "MSB_BOS", "CHoCH", "bars_since_BOS",
    "FVG_up", "FVG_down", "Buy_Liq", "Sell_Liq", "SFP_sweep",

    # Open interest & funding
    "open_interest", "funding_rate",

    # EMA H1
    "ema_7_h1", "ema_21_h1", "ema_50_h1", "ema_200_h1",

    # EMA H4
    "ema_7_h4", "ema_21_h4", "ema_50_h4", "ema_200_h4",

    # Momentum
    "rsi_6", "stochrsi_k", "stochrsi_d",

    # ATR
    "atr_14_h1", "atr_14_h4",

    # Key levels
    "PDH", "PDL", "PWH", "PWL", "Fib_618", "Fib_786",

    # Volume profile
    "POC", "VAH", "VAL",

    # Macro
    "btc_dominance", "fear_greed", "market_session",

    # Returns & volume ratio
    "log_ret_1", "log_ret_5", "log_ret_20", "vol_ratio_20",

    # Time cyclical
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
    "time_to_funding_norm",

    # Long/short ratio
    "long_short_ratio",

    # Swing structure (v2)
    "dist_swing_high", "dist_swing_low", "price_in_range", "swing_momentum",

    # Market regime (v2)
    "h4_trend", "trend_strength", "vol_regime",

    # Smart money v3
    "cvd_div_h4", "cvd_slope_h4",
    "vol_efficiency", "absorption_z",
    "funding_price_div",
    "rsi_h4", "rsi_divergence",
    "wyckoff_phase", "spring_upthrust",

    # Smart money v4 — OFI
    "ofi_raw", "ofi_acceleration", "ofi_z_score", "ofi_h4_delta",

    # Smart money v4 — VWDP
    "vwdp", "vwdp_smooth",

    # Smart money v4 — CVD hidden divergence
    "hidden_divergence", "cvd_momentum_adv",

    # Smart money v4 — Absorption at swing
    "absorption_at_swing",

    # Smart money v4 — VSA
    "spread_to_volume", "ultra_high_vol", "no_demand", "no_supply",
    "effort_vs_result",

    # H4 dynamics — slope & volatility (sebelumnya hilang dari parquet, fix bug H4)
    "ema_21_slope_h4", "ema_50_slope_h4", "price_vs_ema_50_h4",
    "rsi_slope_h4", "atr_percent_h4", "range_expansion_h4",

    # D1 higher timeframe context (HTF regime awareness, fix bug H4)
    "ema_50_d1", "ema_200_d1",
    "ema_50_slope_d1", "ema_200_slope_d1", "price_vs_ema_50_d1",
    "atr_d1_percentile",
    "d1_trend", "d1_trend_strength", "htf_alignment", "d1_hh_hl_bias",

    # Trend quality — correction detection
    "trend_accel_4h", "vol_price_confirm", "dist_from_8h_high",
]

# ─── Trading Simulation Parameters (Sesuai Klarifikasi Pengguna) ─────────────
MODAL_PER_TRADE            = 100.0    # 100 USD per trade (sebelumnya 1000)
LEVERAGE_SIM               = [5.0]    # leverage 5x = 500 USD exposure (sebelumnya [3.0, 5.0])
FEE_PER_SIDE               = 0.0004
SLIPPAGE_PER_SIDE          = 0.0005   # 0.05% slippage per trade side (entry/exit)
CONFIDENCE_THRESHOLD_ENTRY = 0.62     # minimum confidence untuk entry (sama dengan LGBM_THRESHOLD_*)
MIN_HOLD_BARS              = 2        # bar H1 = 2 jam minimum hold

# ─── LGBM Class Weights (Cost-Sensitive Learning) ────────────────────────────
# Format: {label_idx: weight} sesuai LABEL_MAP = {SHORT:0, FLAT:1, LONG:2}
LGBM_CLASS_WEIGHTS = {0: 3.0, 1: 1.5, 2: 3.0}   # SHORT=3x, FLAT=1.5x, LONG=3x
