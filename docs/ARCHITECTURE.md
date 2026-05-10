# Arsitektur: cascade_v2 — 2-Model Cascade + Dynamic TP/SL Regressor

## Overview Pipeline

```
FETCH        CLEAN       ENGINEER     TRAIN          EVALUATE       BACKTEST
01_fetch ──► 02_clean ──► 04_engineer ──► 05_train_lgbm ──► 07_evaluate ──► 08_backtest
                                          06_train_lstm                    09_holdout_backtest
                                          05b_tp_sl_regressor              10_visualize

03_analyze_swing  (opsional — grid search parameter swing)
analyze_min_hold (opsional — rekomendasi MIN_HOLD_BARS)
```

**Data flow:** Binance API → `data/raw/klines/` → `data/processed/` → `data/labeled/` → `models/`

---

## Model Cascade — Arsitektur 2-Model

```
BARISAN DATA H1 (544K bar × 103 fitur, 21 koin)
         │
         ▼
┌─────────────────────────────────────────────┐
│  STEP 1: LGBM Entry Signal Generator        │
│  ─────────────────────────────────────────  │
│  Input    : 103 fitur per bar (full H1)     │
│  Output   : proba[3] = [SHORT, FLAT, LONG]  │
│  Tipe     : Multiclass Classifier (3 kelas) │
│  Training : Walk-Forward CV 8-fold          │
│  Weights  : SHORT=3x, FLAT=1.5x, LONG=3x    │
│  Threshold: 0.62 (LGBM_THRESHOLD_LONG/SHORT)│
│  Role     : Primary — generate sinyal entry │
└────────────────────┬────────────────────────┘
                     │ confidence ≥ 0.62?
                     │ YES ──────────────────┐
                     │ NO  → FLAT (no trade) │
                     ▼                       │
┌─────────────────────────────────────────────┐
│  STEP 2: LSTM Soft Confirmation             │
│  ─────────────────────────────────────────  │
│  Input    : Sequence 16 bar H1 × 103 fitur  │
│  Output   : proba[3] → argmax = arah LSTM   │
│  Tipe     : Multiclass Classifier (3 kelas) │
│  Training : Purged Walk-Forward CV 8-fold   │
│  Arch     : 2-layer ManualLSTMCell (128D)   │
│  GPU      : DirectML (AMD RX 6600)          │
│  Role     : Soft adjustment — bukan veto    │
│                                             │
│  Mode "tiered":                             │
│    LSTM agree dgn LGBM   → +0.05 boost      │
│    LSTM netral (FLAT)    → -0.05 penalty    │
│    LSTM berlawanan arah  → -0.08 × margin   │
│    margin < 0.05  → heavy  (-0.12)          │
│    margin < 0.10  → medium (-0.08)          │
│    else           → light  (-0.04)          │
└────────────────────┬────────────────────────┘
                     │ adjusted_confidence ≥ 0.70?
                     │ YES → SINYAL LONG/SHORT
                     │ NO  → FLAT
                     ▼
               FINAL SIGNAL
```

**Kenapa cuma 2 model (bukan 3):**
- H4 LGBM (regime classifier) dihapus karena AUC hanya 0.55 (near-random)
- Regime context (H4 trend, D1 alignment, volume confirmation) sudah embedded sebagai **103 fitur** langsung di LGBM
- Stacked ensemble (LogReg + Isotonic) dihapus karena degradasi sinyal

---

## Feature Engineering — 103 Fitur

### Layer 1: OHLCV Base (5)
`open`, `high`, `low`, `close`, `volume`

### Layer 2: Volume Flow (4)
`volume_delta`, `cvd`, `buy_volume`, `sell_volume`

### Layer 3: Market Structure (8)
`MSB_BOS`, `CHoCH`, `bars_since_BOS`, `FVG_up`, `FVG_down`, `Buy_Liq`, `Sell_Liq`, `SFP_sweep`

### Layer 4: Open Interest & Funding (2)
`open_interest`, `funding_rate`

### Layer 5: EMA Multi-Timeframe (8)
H1: `ema_7_h1`, `ema_21_h1`, `ema_50_h1`, `ema_200_h1`
H4: `ema_7_h4`, `ema_21_h4`, `ema_50_h4`, `ema_200_h4`

### Layer 6: Momentum (3)
`rsi_6`, `stochrsi_k`, `stochrsi_d`

### Layer 7: ATR (2)
`atr_14_h1`, `atr_14_h4`

### Layer 8: Key Levels (6)
`PDH`, `PDL`, `PWH`, `PWL`, `Fib_618`, `Fib_786`

### Layer 9: Volume Profile (3)
`POC`, `VAH`, `VAL`

### Layer 10: Macro (3)
`btc_dominance`, `fear_greed`, `market_session`

### Layer 11: Returns & Time (8)
`log_ret_1`, `log_ret_5`, `log_ret_20`, `vol_ratio_20`
`hour_sin`, `hour_cos`, `dow_sin`, `dow_cos`, `time_to_funding_norm`

### Layer 12: Long/Short Ratio (1)
`long_short_ratio`

### Layer 13: Swing Structure (4)
`dist_swing_high`, `dist_swing_low`, `price_in_range`, `swing_momentum`

### Layer 14: Market Regime (3)
`h4_trend`, `trend_strength`, `vol_regime`

### Layer 15: Smart Money v3 — Core (6)
`cvd_div_h4`, `cvd_slope_h4`, `vol_efficiency`, `absorption_z`, `funding_price_div`, `rsi_h4`, `rsi_divergence`, `wyckoff_phase`, `spring_upthrust`

### Layer 16: Smart Money v4 — OFI (4)
`ofi_raw`, `ofi_acceleration`, `ofi_z_score`, `ofi_h4_delta`

### Layer 17: Smart Money v4 — VWDP (2)
`vwdp`, `vwdp_smooth`

### Layer 18: Smart Money v4 — CVD Hidden Divergence (2)
`hidden_divergence`, `cvd_momentum_adv`

### Layer 19: Smart Money v4 — Absorption (1)
`absorption_at_swing`

### Layer 20: Smart Money v4 — VSA (5)
`spread_to_volume`, `ultra_high_vol`, `no_demand`, `no_supply`, `effort_vs_result`

### Layer 21: H4 Dynamics (6)
`ema_21_slope_h4`, `ema_50_slope_h4`, `price_vs_ema_50_h4`, `rsi_slope_h4`, `atr_percent_h4`, `range_expansion_h4`

### Layer 22: D1 HTF Context (8)
`ema_50_d1`, `ema_200_d1`, `ema_50_slope_d1`, `ema_200_slope_d1`, `price_vs_ema_50_d1`, `atr_d1_percentile`, `d1_trend`, `d1_trend_strength`, `htf_alignment`, `d1_hh_hl_bias`

### Layer 23: Trend Quality — Correction Detection (3)
`trend_accel_4h`, `vol_price_confirm`, `dist_from_8h_high`

---

## TP/SL System — 3-Tier Priority

```
Setelah entry signal LONG/SHORT:

┌─────────────────────────────────────────────────┐
│  PRIORITY 1: Dynamic Regressor (jika 05b done)  │
│  ───────────────────────────────────────────    │
│  tp_regressor → prediksi tp_mult (momentum/trend)│
│  sl_regressor → prediksi sl_mult (vol/noise)     │
│  Training: win-conditional (hanya bar favorable) │
│  30+ fitur momentum → TP | 20+ fitur noise → SL │
│  Output: TP = entry ± predicted_tp × ATR        │
│          SL = entry ∓ predicted_sl × ATR        │
└────────────────────┬────────────────────────────┘
                     │ jika regressor tidak tersedia
                     ▼
┌─────────────────────────────────────────────────┐
│  PRIORITY 2: Swing-Based H4 (jika swing ada)    │
│  ───────────────────────────────────────────    │
│  TP = swing high H4 terdekat di atas (LONG)     │
│  SL = swing low H4 terdekat di bawah (LONG)     │
│  Validasi: RR ≥ 1.2, TP ≥ 1.2×ATR, SL ≤ 3×ATR │
│  Skip trade jika validasi gagal                  │
└────────────────────┬────────────────────────────┘
                     │ jika swing belum terbentuk (NaN)
                     ▼
┌─────────────────────────────────────────────────┐
│  PRIORITY 3: Fallback ATR Fixed                 │
│  ───────────────────────────────────────────    │
│  TP = 2.0 × ATR (TP_ATR_MULT)                   │
│  SL = 1.5 × ATR (SL_ATR_MULT)                   │
│  Selalu tersedia — tidak pernah skip             │
└─────────────────────────────────────────────────┘
```

---

## Decision Flow Lengkap (per bar H1)

```
Bar H1 ke-i
    │
    ├── Hitung 103 fitur
    │
    ├── LGBM predict_proba() → [prob_SHORT, prob_FLAT, prob_LONG]
    │
    ├── prob_LONG ≥ 0.62 AND prob_LONG ≥ prob_SHORT?
    │   ├── YES → LGBM_dir = LONG, LGBM_conf = prob_LONG
    │   └── NO  → cek SHORT:
    │       prob_SHORT ≥ 0.62 AND prob_SHORT > prob_LONG?
    │       ├── YES → LGBM_dir = SHORT, LGBM_conf = prob_SHORT
    │       └── NO  → FLAT (skip bar ini)
    │
    ├── LSTM confirmation (jika enabled):
    │   ├── Run LSTM inference pada sequence 16 bar terakhir
    │   ├── LSTM_dir = argmax(lstm_proba)
    │   ├── adj = _lstm_adjustment(LGBM_conf, LSTM_dir, LGBM_dir)
    │   │   Mode "tiered":
    │   │     agree: +0.05 × (1 - LGBM_conf)      → boost tipis
    │   │     neutral: -0.05                        → flat penalty
    │   │     opposite: -0.08 × tier_factor(margin) → stop signal
    │   │       margin < 0.05 → ×1.5 = -0.12 (heavy)
    │   │       margin < 0.10 → ×1.0 = -0.08 (medium)
    │   │       else          → ×0.5 = -0.04 (light)
    │   └── adj_conf = clip(LGBM_conf + adj, 0, 1)
    │
    ├── adj_conf ≥ CONFIDENCE_THRESHOLD_ENTRY (0.70)?
    │   ├── YES → SINYAL LONG/SHORT
    │   └── NO  → FLAT
    │
    ├── Jika sinyal:
    │   ├── Tentukan TP/SL (3-tier priority di atas)
    │   ├── Validasi RR (jika swing-based)
    │   ├── Entry dengan slippage
    │   └── Monitor hingga TP/SL hit atau MAX_HOLD (24 bar) timeout
    │
    └── Next bar
```

---

## Training Configuration

| Parameter | Value | Keterangan |
|-----------|-------|------------|
| Training coins | 5 (SOL, ETH, BNB, XRP, DOGE) | `TRAINING_COINS` |
| All coins | 21 | `ALL_COINS` — 5 training + 16 new |
| Period | 2020-01-01 s/d 2026-04-01 | `TRAIN_START` / `TRAIN_END` |
| Folds | 8 | Walk-forward (time-series) |
| Purge gap | 24 bar H1 | Hindari leakage fitur rolling |
| Min hold bars | 2 | Berdasarkan analisis P10 holding time |
| Max hold bars | 24 | 24 jam — timeout |

### LGBM Hyperparameters
| Parameter | Value |
|-----------|-------|
| Objective | multiclass (3 kelas) |
| n_estimators | 1000 |
| learning_rate | 0.05 |
| max_depth | 6 |
| num_leaves | 31 |
| min_child_samples | 50 |
| subsample | 0.8 |
| colsample_bytree | 0.8 |
| Device | GPU (OpenCL — AMD compatible) |
| Early stopping | 50 rounds |

### LSTM Hyperparameters
| Parameter | Value |
|-----------|-------|
| Sequence length | 16 bar H1 |
| Hidden size | 128 |
| Layers | 2 (ManualLSTMCell) |
| Dropout | 0.3 |
| Epochs | 100 |
| Patience | 5 |
| Batch size | 2048 |
| Learning rate | 0.001 |
| Device | DirectML (AMD RX 6600) |

---

## Key Thresholds

| Threshold | Value | Role |
|-----------|-------|------|
| `LGBM_THRESHOLD_LONG` | 0.62 | LGBM minimum confidence untuk LONG entry |
| `LGBM_THRESHOLD_SHORT` | 0.62 | LGBM minimum confidence untuk SHORT entry |
| `CONFIDENCE_THRESHOLD_ENTRY` | 0.70 | Final confidence setelah LSTM adjustment |
| `LSTM_ADJUST_AGREE_BOOST` | +0.05 | Boost saat LSTM searah |
| `LSTM_ADJUST_NEUTRAL_PEN` | -0.05 | Penalty saat LSTM netral |
| `LSTM_ADJUST_OPPOSITE_PEN` | -0.08 | Penalty saat LSTM berlawanan |

---

## Risk Parameters

| Parameter | Value |
|-----------|-------|
| Modal per trade | $100 |
| Leverage | 5x |
| Fee per side | 0.04% |
| Slippage per side | 0.05% |

---

## Model Files (models/)

```
models/
├── lgbm_baseline.pkl          # LGBM entry signal
├── lstm_best.pt               # LSTM confirmation (state dict)
├── lstm_scaler.pkl            # StandardScaler untuk LSTM
├── feature_cols_v2.json       # 103 feature names
├── cv_results.json            # LGBM CV metrics
├── lstm_cv_results.json       # LSTM CV metrics
├── tp_regressor.pkl           # TP dynamic regressor (opsional)
├── sl_regressor.pkl           # SL dynamic regressor (opsional)
├── tp_sl_regressor_meta.json  # Metadata regressor (opsional)
├── tp_sl_feat_cols.json       # Feature cols regressor (opsional)
├── inference_config.json      # Deployment config (dari 08_backtest)
├── model_registry.json        # Model registry
└── runs/
    └── run_YYYYMMDD_HHMMSS/
        ├── lgbm.pkl
        ├── lgbm_cv_results.json
        ├── lstm.pt
        ├── lstm_scaler.pkl
        ├── lstm_cv_results.json
        ├── backtest_results.json
        └── {SYMBOL}_trade_chart.svg
```

---

## Catatan Desain

1. **Prioritas recall > precision** — Class weights 3:1.5:3 mendorong LGBM lebih agresif deteksi LONG/SHORT. False positive masih bisa dipotong SL.

2. **LSTM bukan veto** — LSTM hanya memberikan soft adjustment, bukan hard gate. Ini mencegah false negative dari model kedua.

3. **H4 regime embedded, bukan separate model** — Semua informasi H4/D1 dijadikan fitur langsung di LGBM (103 fitur), menghindari akumulasi error dari stacking model.

4. **TP/SL regressor: win-conditional** — Hanya dilatih pada bar di mana `tp_mult > sl_mult`. Ini berarti regressor belajar "seberapa jauh bisa profit ketika setup favorable", bukan "apakah setup ini favorable".

5. **TP/SL Regressor sudah di-wire ke backtest** (08_backtest.py & 09_holdout_backtest.py) — Priority 1 sebelum swing H4 dan fallback ATR.

6. **Tidak ada mid-trade management** — Setelah entry, trade hanya dimonitor untuk TP/SL hit atau timeout. Tidak ada re-evaluation atau trailing stop berbasis ML.
