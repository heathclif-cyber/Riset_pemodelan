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
# TRAINING_COINS sekarang berisi 20 koin crypto aktif (XAUTUSDT dikeluarkan karena emas).
# Untuk kemudahan, default training langsung mencakup semua koin aktif.
TRAINING_COINS = [
    "SOLUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "DOGEUSDT",
    "TONUSDT", "ADAUSDT", "TRXUSDT", "1000SHIBUSDT", "AVAXUSDT",
    "LINKUSDT", "DOTUSDT", "SUIUSDT", "POLUSDT", "NEARUSDT",
    "1000PEPEUSDT", "TAOUSDT", "ARBUSDT", "HBARUSDT", "ONDOUSDT",
]

NEW_COINS = []

ALL_COINS = TRAINING_COINS


SYMBOL_MAP = {coin: i for i, coin in enumerate(ALL_COINS)}

# ─── Periode Data ─────────────────────────────────────────────────────────────
# Semua koin — baik training maupun new coins — menggunakan periode yang sama.
# Koin yang listing setelah 2020 (SUI, TON, PEPE, TAO, ARB) akan otomatis
# mendapat data lebih pendek sesuai tanggal listing mereka di Binance.
TRAIN_START = datetime(2020, 1, 1, tzinfo=timezone.utc)
TRAIN_END   = datetime(2026, 4, 1, tzinfo=timezone.utc)

START_DATE = TRAIN_START
END_DATE   = TRAIN_END

# Cutoff untuk training — model HANYA dilatih di data sebelum tanggal ini.
# Holdout testing pakai data setelah cutoff (via 09_holdout_backtest.py).
# Ini memastikan TIDAK ADA data testing yang bocor ke training.
TRAIN_CUTOFF_DATE = datetime(2025, 11, 1, tzinfo=timezone.utc)

# ─── Binance API ─────────────────────────────────────────────────────────────
BINANCE_BASE_URL       = "https://fapi.binance.com"
SLEEP_BETWEEN_REQUESTS = 0.12
SLEEP_ON_RATE_LIMIT    = 60.0
MAX_RETRIES            = 3
RETRY_BACKOFF_BASE     = 2.0

KLINE_LIMIT       = 1000  # max Binance API per request
OI_LIMIT          = 500
FUNDING_LIMIT     = 1000
TAKER_RATIO_LIMIT = 500
LONG_SHORT_LIMIT  = 500

# ─── Timeframes ───────────────────────────────────────────────────────────────
KLINE_INTERVALS = ["1h", "4h", "1d"]

# ─── Feature Engineering ──────────────────────────────────────────────────────
TP_ATR_MULT      = 2.0       # TP untuk legacy path (non-swing)
SL_ATR_MULT      = 1.5       # SL untuk legacy path (non-swing)
MAX_HOLDING_BARS = 24        # bar H1 = 24 jam (sebelumnya 48)

# ── Swing-Based Labeling v3 ───────────────────────────────────────────────────
SWING_LABEL_MAX_HOLD = 24     # bar H1 = 24 jam (sebelumnya 48) — lever utama kurangi FLAT
SWING_LABEL_MIN_RR   = 0.45    # 0.45 = TP minimal 45% risiko (setelah Bumper SL)
SWING_LABEL_MIN_TP   = 1.2    # sebelumnya 1.5 — lever utama, TP lebih mudah tercapai
SWING_LABEL_MAX_SL   = 4.0    # 3.0→4.0 (2026-05-10): +697 trades, +$220 PnL, WR +0.14pp — lihat reports/PARAMETER_TEST_REPORT.md
SWING_H4_LOOKBACK    = 5
SWING_ROLLING_BARS   = 24     # 24 jam rolling swing

# ─── HMM Regime Detection ─────────────────────────────────────────────────────
# Dipakai oleh pipeline/11_regime_hmm.py + core/regime.py
HMM_N_STATES  = 4      # 4 states: TRENDING_DOWN, RANGING_LOW_VOL, RANGING_HIGH_VOL, TRENDING_UP
HMM_N_FOLDS   = 8      # walk-forward folds untuk OOF regime labels
HMM_PURGE_H4  = 6      # purge gap (H4 bars) antara train/val ≈ 24 jam
HMM_N_ITER    = 100    # max EM iterations untuk GaussianHMM

# Nama regime canonical (ordered by mean_return ascending)
REGIME_NAMES = ["TRENDING_DOWN", "RANGING_LOW_VOL", "RANGING_HIGH_VOL", "TRENDING_UP"]
REGIME_ENC   = {name: i for i, name in enumerate(REGIME_NAMES)}

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
# device_type="gpu" → pakai OpenCL (kompatibel AMD/Intel/NVIDIA)
# GPU mode: n_jobs diabaikan, max_bin default 63 (lebih cepat, sedikit kurang presisi vs CPU 255)
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
    "device_type":       "gpu",
    "gpu_platform_id":   0,
    "gpu_device_id":     0,
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

# H1 entry thresholds (3-class model tidak berubah)
LGBM_THRESHOLD_LONG  = 0.65   # LGBM minimum confidence untuk entry LONG
LGBM_THRESHOLD_SHORT = 0.65   # LGBM minimum confidence untuk entry SHORT
# FLAT review threshold — saat LGBM output FLAT dengan max_conf < threshold ini,
# LSTM dipanggil untuk review. Jika LSTM deteksi sinyal → override FLAT.
# Makin rendah → makin sering LSTM dipanggil → lebih banyak sinyal, lebih berat.
LGBM_FLAT_REVIEW_THRESHOLD = 0.90  # threshold untuk trigger FLAT review (hanya relevan jika FLAT_REVIEW_ENABLED)
LSTM_FLAT_REVIEW_ENABLED   = False # False = disable LSTM override FLAT (WR 78.8% vs 57.9%. Lihat EXPERIMENTS.md 2026-05-12)
LSTM_OVERRIDE_THRESHOLD    = 0.70  # minimum LSTM confidence untuk override FLAT
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
LSTM_ADJUST_MODE         = "hard_consensus"     # "relative" | "absolute" | "tiered"
LSTM_ADJUST_AGREE_BOOST  = 0.05         # boost saat agree (mode relative/absolute)
LSTM_ADJUST_NEUTRAL_PEN  = 0.00         # penalty saat LSTM FLAT
LSTM_ADJUST_OPPOSITE_PEN = 0.99         # penalty saat LSTM opposite (0.15 asli -> 0.08 -> 0.04 — kurangi blocked trades)
# Tiered multipliers (mode "tiered" only): penalty = pen × multiplier
# margin < 0.05 → borderline, < 0.10 → moderate, else → confident
LSTM_TIERED_MULTIPLIERS = [1.0, 0.5, 0.25]  # [borderline, moderate, confident] (was [1.5, 1.0, 0.5])
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

# ─── Feature Columns v3.1 (H1 Base - 93 fitur, cascade_v3.1_noD1) ─────────────
# Catatan: 10 fitur D1 dihapus (cascade_v3.1_noD1) karena lag berminggu-minggu
# dari slope EMA D1 menekan sinyal LONG saat H4 sudah bullish.
# hmm_regime_enc juga dikecualikan via NON_FEATURE_COLS di pipeline 05/06 (hardcoded 0).
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

    # Trend quality — correction detection
    "trend_accel_4h", "vol_price_confirm", "dist_from_8h_high",
]

# ─── TP/SL Architecture (Final — tested via ASPECT_COMPARISON.md) ───────────
# Lihat pipeline/test_all_aspects.py untuk detail pengujian per aspek.
# Arsitektur: Hybrid TP/SL + Swing Freshness + Structural Filter + RR Gate
#             + Slippage ON + Close SL Trigger + Fixed Sizing + Cooldown OFF

# #1: Hybrid TP/SL — max(swing, ATR) for TP, min(swing, ATR) for SL
TP_SL_HYBRID_MODE = True

# #2: Swing Freshness Check — tolak trade jika deviasi swing > 15% dari entry
TP_SL_SWING_FRESHNESS = True

# #3: Structural Filter — entry harus dalam [H4 Low, H4 High]
TP_SL_STRUCTURAL_FILTER = True
TP_SL_STRUCTURAL_TOLERANCE = 0.03  # Toleransi breakout 3%

# #4: RR Gate — validasi TP/SL sebelum entry
TP_SL_RR_GATE_ENABLED = True  # False = skip semua validasi RR
TP_SL_MIN_RR   = SWING_LABEL_MIN_RR   # 1.0
TP_SL_MIN_TP   = SWING_LABEL_MIN_TP   # 1.2 x ATR
TP_SL_MAX_SL   = SWING_LABEL_MAX_SL   # 4.0 x ATR (mengikuti SWING_LABEL_MAX_SL)

# #5: SL ATR Fallback Multiplier (hanya saat swing NaN)
TP_SL_FALLBACK_TP = 2.0  # TP = 2.0 x ATR
TP_SL_FALLBACK_SL = 1.5  # SL = 1.5 x ATR (Winrate lebih tinggi)

# #7: Slippage entry/exit
TP_SL_SLIPPAGE_ENABLED = True

# #8: SL Trigger — "highlow" karena eksekusi manual order book
TP_SL_TRIGGER_MODE = "highlow"  # "close" | "highlow"
TP_SL_SWING_BUMPER = 0.5        # Bumper SL 0.5x ATR pencegah stop-hunt

# #12: Position Sizing — "fixed" = $100/trade
TP_SL_SIZING_MODE = "fixed"  # "fixed" | "tiered"

# #15: Cooldown after exit — OFF (terlalu restriktif, buang 170 trade valid)
TP_SL_COOLDOWN_ENABLED = False

# ─── #16: VolR Conditional Max SL (Grup 1b/1c) ──────────────────────────────
# Saat VolR (volume ratio) < threshold, longgarkan max_sl_atr atau disable total.
# Mengatasi false positive di low ATR — SL struktural di-penalize max_sl_atr saat vol menyusut.
TP_SL_VOLR_CONDITIONAL_ENABLED = False  # enable conditional max_sl via VolR
TP_SL_VOLR_THRESHOLD           = 0.2    # threshold VolR untuk trigger low-vol
TP_SL_MAX_SL_VOLR_LOW          = 8.0    # max_sl_atr saat VolR < threshold (1b)
TP_SL_VOLR_DISABLE_MAX_SL      = False  # jika True, disable max_sl total di low-vol (1c)

# ─── #17: SL % Distance Cap (Grup 1d) ───────────────────────────────────────
# Alternatif metrik: batas SL berbasis persentase dari entry, bukan ATR.
TP_SL_MAX_SL_PCT_ENABLED = False  # enable SL % cap
TP_SL_MAX_SL_PCT         = 0.30   # max SL = 30% dari entry price

# ─── #18: Trend Alignment Penalties (Grup 2) ─────────────────────────────────
# With-trend trades WR rendah (33.3%) → penalty untuk kurangi sinyal with-trend.
# Counter-trend trades WR tinggi (77.8%) → boost untuk perlebar akses.
# Trend determined by h4_trend feature (>0 = UP, <0 = DOWN, ≈0 = RANGING).
TREND_ALIGNMENT_ENABLED  = False  # enable trend alignment adjustment
WITH_TREND_PENALTY       = 0.10   # penalty subtracted from confidence (2a)
COUNTER_TREND_BOOST      = 0.05   # boost added to confidence (2b)
WITH_TREND_BLOCK_CONF    = 0.95   # block all with-trend if conf < this (2c, 0 = disable)

# ─── #19: Max Swing Deviation (Grup 3b) ─────────────────────────────────────
# Tolak trade jika deviasi swing > threshold. Saat ini hardcoded 0.15.
TP_SL_MAX_SWING_DEVIATION_PCT = 0.15  # max deviasi swing H4 dari entry (3b: uji 0.12, 0.10)

# ─── #20: Individual Swing Freshness (Grup 3c) ───────────────────────────────
# Cek freshness masing-masing swing (high dan low) secara individual.
# Cegah TONUSDT-style leak — salah satu swing basi tetap lolos.
TP_SL_INDIVIDUAL_SWING_FRESHNESS = False  # jika True, tolak jika salah satu swing dev > max

# ─── #21: Conditional Sizing: Tiered + Half-Size for With-Trend (Grup 4b) ───
TP_SL_SIZING_WITH_TREND_HALF = False  # half-size untuk with-trend trades (hanya di mode tiered)

# ─── Exit Guardian (3rd Model — Dynamic Exit) ───────────────────────────────
# Model ke-3: Binary LGBM classifier untuk per-bar HOLD/EXIT decision.
# Aktif SETELAH entry — memonitor setiap bar dan menutup posisi saat momentum
# berbalik atau profit sudah optimal.
# Static TP/SL digantikan oleh guardian exit, dengan wide safety SL sebagai
# circuit breaker (5x ATR — bukan exit strategy utama).
GUARDIAN_ENABLED               = True   # master toggle — RUN C holdout
GUARDIAN_EXIT_THRESHOLD        = 0.65   # min EXIT proba mid-level
GUARDIAN_SL_EXIT_THRESHOLD     = 0.40   # min EXIT proba saat di swing SL (lebih longgar)
GUARDIAN_SL_SAFETY_ATR         = 1.5    # SL floor = 1.5x ATR dari entry
GUARDIAN_TP_ATR                = 2.0    # TP ceiling = 2.0x ATR (override swing)
GUARDIAN_MIN_HOLD_BARS         = 3      # guardian tidak boleh exit di 3 bar pertama
GUARDIAN_ACTIVATION_ATR        = 1.5    # guardian aktif setelah price bergerak 1.5x ATR

# ─── Trailing Stop (non-ML) ────────────────────────────────────────────────
TRAILING_STOP_ENABLED          = False  # Guardian solo — RUN C holdout
TRAILING_STOP_ATR              = 2.0    # jarak stop dari best price (× ATR)
TRAILING_STOP_MIN_BARS         = 2      # min bars sebelum trailing aktif

# Guardian static features — full FEATURE_COLS_V3 (93 fitur)
# Multiclass: 0=HOLD, 1=PARTIAL_EXIT, 2=FULL_EXIT
GUARDIAN_STATIC_FEATURES = FEATURE_COLS_V3

# Dynamic features — trade context (dihitung per bar saat simulasi)
GUARDIAN_DYNAMIC_FEATURES = [
    "bars_held_norm", "current_pnl_pct", "current_pnl_atr",
    "max_favorable_pnl_pct", "drawdown_from_peak_pct",
    "direction", "entry_price_ratio",
]

# Guardian LGBM training params (multiclass: 0=HOLD, 1=PARTIAL_EXIT, 2=FULL_EXIT)
GUARDIAN_LGBM_PARAMS = {
    "objective":          "multiclass",
    "num_class":          3,
    "n_estimators":       500,
    "learning_rate":      0.05,
    "max_depth":          6,
    "num_leaves":         31,
    "min_child_samples":  50,
    "subsample":          0.8,
    "colsample_bytree":   0.8,
    "verbose":            -1,
    "n_jobs":             -1,
    "random_state":       42,
}
GUARDIAN_PARTIAL_EXIT_RATIO = 0.5  # % posisi ditutup saat PARTIAL_EXIT
GUARDIAN_EARLY_STOPPING    = 50
GUARDIAN_N_FOLDS           = 8
GUARDIAN_PURGE_GAP_BARS    = 5
GUARDIAN_MIN_SAMPLES_COIN  = 30  # min in-trade bars per coin untuk training

# ─── Trading Simulation Parameters (Sesuai Klarifikasi Pengguna) ─────────────
MODAL_PER_TRADE            = 25.0    # 25 USD per trade (sesuai setting UI)
LEVERAGE_SIM               = [5.0]    # leverage 5x = 500 USD exposure (sebelumnya [3.0, 5.0])
FEE_PER_SIDE               = 0.0004
SLIPPAGE_PER_SIDE          = 0.0005   # 0.05% slippage per trade side (entry/exit)
CONFIDENCE_THRESHOLD_ENTRY = 0.65     # threshold entry disamakan dengan LGBM_THRESHOLD (tadinya 0.70 — SHORT killed di gap 0.62-0.69)
MIN_HOLD_BARS              = 2        # bar H1 = 2 jam minimum hold

# ─── LGBM Class Weights (Cost-Sensitive Learning) ────────────────────────────
# Format: {label_idx: weight} sesuai LABEL_MAP = {SHORT:0, FLAT:1, LONG:2}
LGBM_CLASS_WEIGHTS = {0: 3.0, 1: 1.5, 2: 3.0}   # SHORT=3x, FLAT=1.5x, LONG=3x

