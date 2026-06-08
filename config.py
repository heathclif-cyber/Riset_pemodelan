"""
config.py — Sentralisasi semua parameter proyek
Edit file ini untuk mengubah parameter training, fetch, atau feature engineering.
"""

# =============================================================================
# V2.5 HYBRID MODE — AKTIF (Decision Juni 2026)
# =============================================================================
# Tujuan hybrid ini: Mengatasi under-trading parah di live trading Mei 2026
# yang disebabkan oleh konfigurasi ultra-selektif (cascade_v3.1 / v4.1).
#
# Bukti utama:
#   - cascade_v2 (lebih agresif) secara konsisten lebih profit di livetrade.csv
#     selama periode choppy (54.7% WR, +$232 di early May vs -13 di late May).
#   - Perubahan besar setelah 12-15 Mei (matikan FLAT review + LONG 0.75 +
#     opposite penalty 0.99 + banyak filter baru) menyebabkan volume anjlok.
#
# Filosofi V2.5:
#   - Pertahankan 100% kekuatan modern:
#       • Guardian v3 multiclass (104 fitur + 7 dynamic)
#       • TP → Momentum Mode
#       • Instant Guardian activation (min_hold=0, activation_atr=0)
#       • Volatility Spike Detectors (v4.1)
#       • Trend Alignment + VCB + Structural Filter
#   - Longgarkan hanya gerbang ENTRY secara moderat agar sistem bisa
#     menangkap lebih banyak opportunity di regime sulit tanpa mengulang
#     kesalahan besar FLAT review.
#
# Parameter yang diubah di V2.5 Hybrid (lihat detail di bawah):
#   - LGBM_THRESHOLD_LONG          : 0.75  → 0.69
#   - LGBM_THRESHOLD_SHORT         : 0.60  → 0.59
#   - LSTM_ADJUST_OPPOSITE_PEN     : 0.99  → 0.65
#   - CONFIDENCE_THRESHOLD_ENTRY   : 0.60  → 0.59
#
# Parameter yang DIJAGA KETAT (jangan diubah tanpa bukti kuat):
#   - LSTM_FLAT_REVIEW_ENABLED = True + LSTM_DIRECTIONAL_REVIEW_THRESHOLD = 0.35 (eksperimen baru)
#   - Semua Guardian v3 settings
#   - Semua Volatility + Game Changer features
#
# Cara revert cepat: kembalikan 4 nilai di atas ke setting sebelum Juni 2026.
# =============================================================================

from datetime import datetime, timezone
from pathlib import Path

# ─── Paths ───────────────────────────────────────────────────────────────────
ROOT_DIR         = Path(__file__).parent
DATA_DIR         = ROOT_DIR / "data"

# Training data (2020-01-01 -> 2025-11-01)
TRAINING_DIR     = DATA_DIR / "training"          # root training
RAW_DIR          = TRAINING_DIR                   # alias — titik simpan klines, macro, dll.
PROC_DIR         = TRAINING_DIR / "processed"
LABEL_DIR        = TRAINING_DIR / "labeled"

# Holdout-test data (2025-11-01 -> 2026-04-01)
HOLDOUT_DIR      = DATA_DIR / "holdout-test"      # root holdout-test

MODEL_DIR        = ROOT_DIR / "models"
REPORT_DIR       = ROOT_DIR / "reports"

# ─── Koin ────────────────────────────────────────────────────────────────────
# TRAINING_COINS sekarang berisi 20 koin crypto aktif (XAUTUSDT dikeluarkan karena emas).
# Untuk kemudahan, default training langsung mencakup semua koin aktif.
TRAINING_COINS = [
    "BTCUSDT",
    "SOLUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "DOGEUSDT",
    "TONUSDT", "ADAUSDT", "TRXUSDT", "1000SHIBUSDT", "AVAXUSDT",
    "LINKUSDT", "DOTUSDT", "SUIUSDT", "POLUSDT", "NEARUSDT",
    "1000PEPEUSDT", "TAOUSDT", "ARBUSDT", "HBARUSDT", "ONDOUSDT",
]

NEW_COINS = []

ALL_COINS = TRAINING_COINS


SYMBOL_MAP = {coin: i for i, coin in enumerate(ALL_COINS)}

# ─── Periode Data ─────────────────────────────────────────────────────────────
#
# TRAINING DATA  : 2020-01-01 → TRAIN_CUTOFF_DATE (disimpan di data/raw/)
# OOS HOLDOUT    : OOS_START  → OOS_END           (disimpan di data/holdout/raw/)
#
# Fetch training : python pipeline/01_fetch.py --all
# Fetch OOS      : python pipeline/01_fetch.py --all --oos
#
# Koin yang listing setelah 2020 (SUI, TON, PEPE, TAO, ARB) akan otomatis
# mendapat data lebih pendek sesuai tanggal listing mereka di Binance.

# ── Training window ──────────────────────────────────────────────────────────
TRAIN_START = datetime(2020, 1,  1,  tzinfo=timezone.utc)

# Cutoff untuk training — model HANYA dilatih di data sebelum tanggal ini.
# Pastikan = OOS_START sehingga tidak ada overlap.
TRAIN_CUTOFF_DATE = datetime(2026, 4, 1, tzinfo=timezone.utc)
TRAIN_END         = TRAIN_CUTOFF_DATE   # alias — 01_fetch.py pakai TRAIN_END

START_DATE = TRAIN_START
END_DATE   = TRAIN_CUTOFF_DATE

# ── OOS / Hold-Out window ────────────────────────────────────────────────────
# Data ini di-fetch TERPISAH ke data/holdout/raw/ via --oos flag.
# TIDAK BOLEH tumpang tindih dengan training.
OOS_START = TRAIN_CUTOFF_DATE           # 2026-04-01 — sama persis dengan TRAIN_END
OOS_END   = datetime(2026, 7, 1, tzinfo=timezone.utc)  # 3 bulan OOS: Apr-Jun 2026

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
MAX_HOLDING_BARS = 36        # bar H1 = 36 jam — sync dengan SWING_LABEL_MAX_HOLD

# ── Swing-Based Labeling v3 ───────────────────────────────────────────────────
SWING_LABEL_MAX_HOLD = 36     # bar H1 = 36 jam — lebih panjang untuk capture momentum kuat
SWING_LABEL_MIN_RR   = 0.60    # 0.60 — dinaikkan dari 0.45: hapus label negatif EV
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

# PURGE_GAP_BARS untuk entry model (LGBM + LSTM v2-style)
# Sebelumnya 24 (sama dengan SWING_LABEL_MAX_HOLD).
# Diubah ke 16 bar (2026-05-31) untuk mengurangi pemborosan data di fold boundary
# sambil tetap memberikan buffer terhadap horizon labeling 24 bar.
# Catatan: Guardian menggunakan GUARDIAN_PURGE_GAP_BARS = 5 (terpisah).
PURGE_GAP_BARS = 20   # min safe = ceil(SWING_LABEL_MAX_HOLD/2) = 18 → pakai 20 untuk buffer
                      # Dengan max_hold=36: last label forward reach = T_fold-20+35 = T_fold+15
                      # Test fold starts = T_fold+20 → buffer 5 bar ✓

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
    "device_type":       "cpu",
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
# ─── V2.5 Hybrid Entry (2026-06 decision) ─────────────────────────────────────
# Tujuan: Mengembalikan sebagian volume entry yang hilang di regime choppy Mei 2026,
#         tanpa mengorbankan kekuatan Guardian v3 + Momentum Mode + Volatility Detectors.
#
# Bukti:
#   - cascade_v2 (lebih longgar) menghasilkan PnL jauh lebih baik di live May 2026.
#   - Ultra-selective config (LONG 0.75 + opposite_pen 0.99) menyebabkan under-trading.
#
# Prinsip V2.5 Hybrid:
#   - Pertahankan semua kekuatan exit (Guardian v3 multiclass 104 feat + instant + momentum).
#   - Longgarkan entry gate secara moderat agar sistem bisa "bernafas" di pasar sulit.
#   - JANGAN hidupkan kembali FLAT_REVIEW_ENABLED (terbukti buruk di EXPERIMENTS 2026-05-12).
#
# Nilai ini adalah sweet spot awal. Perlu diuji di backtest recent choppy period + paper trading.
LGBM_THRESHOLD_LONG  = 0.69   # V2.5 Hybrid: turun dari 0.75 (terlalu matikan LONG)
LGBM_THRESHOLD_SHORT = 0.59   # V2.5 Hybrid: sedikit lebih longgar dari 0.60
# ──────────────────────────────────────────────────────────────────────────────
# FLAT review threshold — saat LGBM output FLAT dengan max_conf < threshold ini,
# LSTM dipanggil untuk review. Jika LSTM deteksi sinyal → override FLAT.
# Makin rendah → makin sering LSTM dipanggil → lebih banyak sinyal, lebih berat.
LGBM_FLAT_REVIEW_THRESHOLD = 0.90  # threshold untuk trigger FLAT review (hanya relevan jika FLAT_REVIEW_ENABLED)

# ─── LSTM Review Activation (V2.5 Hybrid Experiment) ─────────────────────────
# LSTM_FLAT_REVIEW_ENABLED = True  → LSTM "hidup" untuk mereview / adjust keputusan LGBM
#
# Kondisi baru (per 2026-06):
#   LSTM akan aktif (direview) ketika:
#     - LGBM LONG score > LSTM_DIRECTIONAL_REVIEW_THRESHOLD (0.35), ATAU
#     - LGBM SHORT score > LSTM_DIRECTIONAL_REVIEW_THRESHOLD (0.35)
#
# Ini berbeda dari versi lama (hanya review ketika LGBM FLAT + max_conf rendah).
# Tujuannya: memberi LSTM kesempatan lebih besar untuk mengoreksi / menguatkan
# sinyal directional yang sedang "sedang-sedang saja" (>0.35), bukan hanya menyelamatkan FLAT.
LSTM_FLAT_REVIEW_ENABLED         = True   # Directional review active
LSTM_DIRECTIONAL_REVIEW_THRESHOLD = 0.35   # Threshold baru untuk mengaktifkan LSTM review pada sinyal directional

LSTM_OVERRIDE_THRESHOLD    = 0.70  # minimum LSTM confidence untuk override FLAT (masih dipakai di jalur lama)
LSTM_CONFIRMATION_ENABLED = True   # LSTM ON — survival filter: +$292, -42% volatility

# ─── HMM Gate: LSTM hanya aktif saat TRENDING ────────────────────────────
HMM_GATE_LSTM_ENABLED      = True   # NEW: LSTM only speaks in TRENDING (regime 0,3)
# Rationale: LSTM momentum OOF benefit +$125 over 63 months
# In RANGING: LSTM silent — LGBM runs solo (better WR in ranging)
# In TRENDING: LSTM active — survival filter (reduces counter-trend entries)
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
LSTM_ADJUST_OPPOSITE_PEN = 0.65         # V2.5 Hybrid: turun dari 0.99 (terlalu sering membunuh trade bagus di live)
                                           # Nilai 0.65 masih cukup keras untuk membatasi kasus berlawanan arah,
                                           # tapi memberi ruang lebih besar dibanding 0.99.
# Tiered multipliers (mode "tiered" only): penalty = pen × multiplier
# margin < 0.05 → borderline, < 0.10 → moderate, else → confident
LSTM_TIERED_MULTIPLIERS = [1.0, 0.5, 0.25]  # [borderline, moderate, confident] (was [1.5, 1.0, 0.5])

# ─── New Momentum Gate Fusion (Corrector + Booster mode, 2026-06) ─────────────
LSTM_FUSION_MODE = "hard_consensus"   # "hard_consensus" | "momentum_gate"
LSTM_CORRECTION_POWER = 0.35          # seberapa kuat LSTM boleh mengoreksi (0.0 - 1.0)
LSTM_BOOST_MULTIPLIER = 0.80          # seberapa kuat boost ketika aligned
LSTM_MIN_SCORE_TO_CORRECT = 0.65      # skor minimum LSTM agar diizinkan mengoreksi/boost
LSTM_MAX_PROB_SHIFT = 0.45            # batas maksimal pergeseran probabilitas oleh LSTM
# LSTM Params
LSTM_SEQ_LEN    = 32   # V4: naik dari 16 — 32 jam (8 candle H4) untuk capture momentum cycle penuh
LSTM_HIDDEN     = 128
LSTM_LAYERS     = 2
LSTM_DROPOUT    = 0.3
LSTM_EPOCHS     = 100
LSTM_PATIENCE   = 10
LSTM_BATCH_SIZE = 512
LSTM_LR         = 0.001

# ─── LSTM v2-style (khusus cascade_v2.5_hybrid_pruned) - Regularisasi ──────────
# Digunakan HANYA oleh pipeline/05_train_lstm_v2_style.py
#
# Riwayat probe (5 coin diagnostic):
#   Round 1 (original)  : Hidden=128, Dropout=0.3, WD=0      → Gap +0.42 ~ +0.49 (overfit berat)
#   Round 2 (medium)    : Hidden=96,  Dropout=0.4, WD=1e-4   → Gap +0.36 ~ +0.40 (terbaik sejauh ini)
#   Round 3 (terlalu agresif): Hidden=64, Dropout=0.5, WD=5e-4 → Val F1 hancur (0.16-0.20)
#   Round 4 (light Round 2): Hidden=96, Dropout=0.45, WD=2e-4, LR=0.0007  ← sekarang
#
# Tujuan: mencari sweet spot di sekitar Round 2 yang lebih stabil.
LSTM_V2_HIDDEN       = 96
LSTM_V2_LAYERS       = 2
LSTM_V2_DROPOUT      = 0.45
LSTM_V2_WEIGHT_DECAY = 2e-4
LSTM_V2_LR           = 0.0007

# ─── LSTM Momentum Detector Features (Central Source of Truth) ────────────────
# 18 fitur fokus akumulasi/flow (CVD, OFI variants, volume_delta, absorption, dynamic pressure, dll)
# + price trajectory (log_ret) + struktur sederhana. LSTM lebih kuat menangkap pola temporal
# akumulasi seiring waktu dibanding LGBM.
# Digunakan oleh: pipeline/05b_build_h1_sequences.py dan pipeline/05c_train_lstm_h1.py
# JANGAN duplikasi list fitur ini di dalam script pipeline manapun.
LSTM_MOMENTUM_FEATURES = [
    # === Akumulasi / Flow (Prioritas Tertinggi) ===
    "volume_delta",
    "cvd",
    "buy_volume",
    "sell_volume",
    "ofi_raw",
    "ofi_z_score",
    "ofi_acceleration",
    "cvd_momentum_adv",
    "dynamic_position_pressure",
    "absorption_z",

    # === Price Trajectory + Momentum yang evolve ===
    "log_ret_1",
    "log_ret_5",
    "log_ret_20",
    "rsi_6",
    "rsi_slope_h4",

    # === Structure sederhana ===
    "bars_since_BOS",
    "swing_momentum",

    # === Cross-market context ===
    "btc_h1_return",
]

# Alias untuk kompatibilitas
LSTM_SEQ_FEATURES = LSTM_MOMENTUM_FEATURES

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
    "open_interest", "dynamic_position_pressure", "funding_rate",

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
    "market_session", "btc_dominance", "fear_greed",

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

    # Game Changer Features (v4.0)
    "relative_strength_z", "relative_strength_momentum",
    "dist_liq_50x_long", "dist_liq_20x_long",
    "dist_liq_50x_short", "dist_liq_20x_short",
    "whale_retail_divergence",

    # Volatility Regime Features (v4.1) — Gejolak Market Detectors
    # atr_zscore_20d     : ATR H1 vs mean 20-hari (spike = gejolak)
    # atr_percentile_h1  : Volatility rank 30-hari (0=rendah, 1=tinggi)
    # vol_spike_zscore   : Volume H1 z-score vs 48-bar (spike = event besar)
    "atr_zscore_20d", "atr_percentile_h1", "vol_spike_zscore",

    # Momentum Acceleration Features (v4.2) — Pump/Dump Early Detection
    # price_accel_1h    : 2nd derivative log return — perubahan kecepatan harga
    # ofi_momentum_ratio: OFI 3-bar vs 24-bar — order flow surge vs baseline
    # vol_accel_3h      : perubahan vol_ratio_20 dalam 3 bar — volume acceleration
    "price_accel_1h", "ofi_momentum_ratio", "vol_accel_3h",

    # Kronos-mini Foundation Model Features (v4.3) — predict_len=36H = max_hold
    # kronos_pred_return_1h  : expected return 1H ke depan
    # kronos_pred_return_36h : expected return kumulatif 36H (match SWING_LABEL_MAX_HOLD)
    # kronos_direction       : arah trend 36H (-1/0/+1)
    # kronos_momentum        : slope linear prediksi
    # kronos_volatility      : uncertainty model
    # kronos_bull_prob       : probabilitas naik > 0.5% di t+36
    # kronos_range_ratio     : (pred_high - pred_low) / price
    # kronos_confidence      : 1 - CV (makin tinggi = makin yakin)
    "kronos_pred_return_1h", "kronos_pred_return_36h",
    "kronos_direction", "kronos_momentum", "kronos_volatility",
    "kronos_bull_prob", "kronos_range_ratio", "kronos_confidence",
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
TREND_ALIGNMENT_ENABLED  = False  # Disabled — digantikan REGIME_AWARE_ALIGNMENT
REGIME_AWARE_ALIGNMENT   = True   # FLIP: RANGING=counter-trend (swing), TRENDING=with-trend (momentum)

# ─── Positioning Engine (IC-validated, Binance public API) ──────────────────
POSITIONING_ENGINE_ENABLED  = True  # NEW: LS extreme → size × 0.50 (|IC|=0.16)
POSITIONING_SIZE_MULTIPLIER = 0.50   # Size reduction when LS extreme (>2σ)
POSITIONING_LS_EXTREME_THR  = 2.0    # Z-score threshold for extreme
# Columns from Coinank daily data (pre-computed, joined to df_slice):
#   pos_extreme:      LS z-score > 2.0 → reduces size (IC=0.16, predicts vol)
#   pos_ls_z20, pos_ls_d7, pos_oi_z20: for future regime bias
                                   # +$214 PnL (59% improvement) di extended backtest 63 bulan
WITH_TREND_PENALTY       = 0.10   # penalty subtracted from confidence (2a)
COUNTER_TREND_BOOST      = 0.05   # boost added to confidence (2b)
WITH_TREND_BLOCK_CONF    = 0.00   # block all with-trend if conf < this (2c, 0 = disable)

# ─── #19: Max Swing Deviation (Grup 3b) ─────────────────────────────────────
# Tolak trade jika deviasi swing > threshold. Saat ini hardcoded 0.15.
TP_SL_MAX_SWING_DEVIATION_PCT = 0.15  # max deviasi swing H4 dari entry (3b: uji 0.12, 0.10)

# ─── #20: Individual Swing Freshness (Grup 3c) ───────────────────────────────
# Cek freshness masing-masing swing (high dan low) secara individual.
# Cegah TONUSDT-style leak — salah satu swing basi tetap lolos.
TP_SL_INDIVIDUAL_SWING_FRESHNESS = False  # OFF — cek swing target saja (lebih longgar)

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
GUARDIAN_MIN_HOLD_BARS         = 2      # optimal: biarkan trade bernafas, kurangi exit prematur
GUARDIAN_ACTIVATION_ATR        = 0.0    # guardian aktif instant (tanpa ATR minimum)

# ─── Trailing Stop (non-ML) ────────────────────────────────────────────────
TRAILING_STOP_ENABLED          = False  # Guardian solo — RUN C holdout
TRAILING_STOP_ATR              = 2.0    # jarak stop dari best price (× ATR)
TRAILING_STOP_MIN_BARS         = 2      # min bars sebelum trailing aktif

# Guardian static features — full FEATURE_COLS_V3 (93 fitur)
# Multiclass: 0=HOLD, 1=PARTIAL_EXIT, 2=FULL_EXIT
GUARDIAN_STATIC_FEATURES = FEATURE_COLS_V3

# Extended static features untuk Guardian:
# 32 KEEP (ic32) + 10 REDUNDANT terbaik (IC 0.05-0.09)
# REDUNDANT berguna untuk Guardian via non-linear combinations exit timing
GUARDIAN_EXTENDED_STATIC = [
    # 32 KEEP features (same as LGBM ic32)
    "dist_from_8h_high", "rsi_6", "swing_momentum", "rsi_h4", "stochrsi_k",
    "dist_liq_50x_long", "trend_accel_4h", "rsi_slope_h4", "Fib_786", "Fib_618",
    "stochrsi_d", "ofi_h4_delta", "dist_liq_50x_short", "Buy_Liq",
    "relative_strength_z", "dist_liq_20x_long", "cvd_momentum_adv", "cvd_slope_h4",
    "ema_21_slope_h4", "ema_50_h1", "log_ret_20", "whale_retail_divergence",
    "Sell_Liq", "long_short_ratio",
    # 10 top REDUNDANT (punya IC 0.05-0.09, berguna non-linear untuk exit timing)
    "ema_7_h1", "ema_7_h4", "log_ret_5", "price_in_range",
    "VAH", "POC", "ema_21_h1", "dist_swing_low", "VAL", "dist_swing_high",
]

# Delta features — market change since entry (IC-validated)
# Clean v2: delta di-drop — config kept for reference only.
GUARDIAN_DELTA_MAP = {}

# Dynamic features — trade context (dihitung per bar saat simulasi)
# Clean v2: 7 original ONLY. Delta features di-drop — tidak menambah value OOS.
GUARDIAN_DYNAMIC_FEATURES = [
    "bars_held_norm", "current_pnl_pct", "current_pnl_atr",
    "max_favorable_pnl_pct", "drawdown_from_peak_pct",
    "direction", "entry_price_ratio",
]

# Guardian LGBM training params (multiclass: 0=HOLD, 1=PARTIAL_EXIT, 2=FULL_EXIT)
GUARDIAN_LGBM_PARAMS = {
    "objective":          "multiclass",
    "num_class":          3,
    "n_estimators":       2000,       # dinaikkan dari 500 — fold 4-8 mentok di batas lama
    "learning_rate":      0.02,       # diturunkan dari 0.05 — langkah lebih presisi
    "max_depth":          6,
    "num_leaves":         63,         # dinaikkan dari 31 — pohon lebih ekspresif
    "min_child_samples":  30,         # diturunkan dari 50 — split lebih granular
    "subsample":          0.8,
    "colsample_bytree":   0.8,
    "lambda_l1":          0.1,        # L1 regularization — cegah overfit saat model makin kompleks
    "lambda_l2":          0.1,        # L2 regularization
    "verbose":            -1,
    "n_jobs":             -1,
    "random_state":       42,
}
GUARDIAN_PARTIAL_EXIT_RATIO = 0.5  # % posisi ditutup saat PARTIAL_EXIT
GUARDIAN_EARLY_STOPPING    = 100   # dinaikkan dari 50 — kasih lebih banyak kesempatan konvergen
GUARDIAN_N_FOLDS           = 8
GUARDIAN_PURGE_GAP_BARS    = 5
GUARDIAN_MIN_SAMPLES_COIN  = 30  # min in-trade bars per coin untuk training

# ─── Trading Simulation Parameters (Sesuai Klarifikasi Pengguna) ─────────────
MODAL_PER_TRADE            = 10.0    # 10 USD per trade, 5x leverage = $50 exposure
LEVERAGE_SIM               = [5.0]    # leverage 5x = 500 USD exposure (sebelumnya [3.0, 5.0])
FEE_PER_SIDE               = 0.0004
SLIPPAGE_PER_SIDE          = 0.0005   # 0.05% slippage per trade side (entry/exit)
CONFIDENCE_THRESHOLD_ENTRY = 0.59     # V2.5 Hybrid: diselaraskan dengan threshold baru (0.69/0.59)
MIN_HOLD_BARS              = 2        # bar H1 = 2 jam minimum hold

# ─── LGBM Class Weights (Cost-Sensitive Learning) ────────────────────────────
# Format: {label_idx: weight} sesuai LABEL_MAP = {SHORT:0, FLAT:1, LONG:2}
LGBM_CLASS_WEIGHTS = {0: 3.0, 1: 1.5, 2: 3.0}   # SHORT=3x, FLAT=1.5x, LONG=3x

