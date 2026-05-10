# MODEL_DEBUG_GUIDE.md — Arsitektur, Audit & Tuning Model SwingTrade v2

> **Tujuan:** Panduan lengkap untuk tuning model di **training repository** (terpisah dari repo ini).
> Hasil tuning disimpan di `models/inference_config.json` dan file model (`.pt`, `.pkl`).
> Repo ini hanya menjalankan **inference + trading**, bukan training.

---

## 1. Arsitektur Aplikasi (Terbaru — cascade_v2)

```
┌─────────────────────────────────────────────────────────┐
│                   TRAINING REPO (terpisah)               │
│  engineer_features() → labeling → train LGBM/LSTM       │
│  Output: lstm_best.pt, lgbm_baseline.pkl,               │
│          lstm_scaler.pkl, inference_config.json          │
└──────────────────────┬──────────────────────────────────┘
                       │ deploy_model.py
                       ▼
┌─────────────────────────────────────────────────────────┐
│                   REPO INI (swingtrade_v2)               │
│                                                         │
│  APScheduler (in-process)                               │
│  ┌──────────────┐  ┌──────────────┐  ┌───────────────┐  │
│  │ fetch_latest │  │generate_signal│  │check_positions│  │
│  │   (15 menit) │  │   (1 jam)    │  │  (5 menit)    │  │
│  └──────────────┘  └──────┬───────┘  └───────┬───────┘  │
│                            │                  │          │
│                            ▼                  ▼          │
│                   ┌─────────────────┐  ┌──────────────┐  │
│                   │ InferenceService│  │PaperTrading  │  │
│                   │ LGBM + LSTM     │  │Engine        │  │
│                   │ (103 fitur)     │  │TP/SL + VCB   │  │
│                   └────────┬────────┘  └──────┬───────┘  │
│                            │                  │          │
│                            ▼                  ▼          │
│                     Signal DB ──────────► Trade DB       │
│                                                         │
│  Config SSOT: models/inference_config.json               │
└─────────────────────────────────────────────────────────┘
```

**Alur sinyal ke trade (cascade_v2):**
```
features_df (103 kolom H1)
  → InferenceService.predict()     ← baca lgbm_baseline.pkl + lstm_best.pt
  → STEP 1: LGBM predict_proba()  → P(LONG), P(SHORT), P(FLAT)
  → STEP 2: LSTM soft adjustment  → boost/penalty confidence
  → Signal DB (direction + confidence + feature_snapshot)
  → PaperTradingEngine.process_signal()
     → confidence check → cooldown → VCB → TP/SL calc → sizing
     → Trade DB (entry/exit/TP/SL/PnL)
```

**Yang dibaca dari `inference_config.json` saat inference:**
- `inference.confidence_threshold_entry` → filter sinyal
- `inference.seq_len` → LSTM window (**sekarang 16**, bukan 32)
- `model_architecture` → n_features (**103**), hidden, layers
- `cascade.*` → LGBM dan LSTM thresholds
- `fallback_tp_sl.*` → TP/SL ATR multipliers
- `volatility_circuit_breaker.*` → VCB
- `risk.*` → modal, leverage, fee
- `label_map` → mapping arah ke index

---

## 2. Perubahan Arsitektur — hierarchical_v1 → cascade_v2

> **BREAKING CHANGES** — aplikasi inference **wajib** diupdate sebelum deploy model baru.

### 2.1 Arsitektur Model

| Aspek | hierarchical_v1 (lama) | cascade_v2 (baru) |
|-------|----------------------|-------------------|
| Model | H4 LGBM + H1 LGBM + LSTM | **LGBM + LSTM** |
| n_features | 84 | **103** |
| LSTM seq_len | 32 bar H1 | **16 bar H1** |
| H4 LGBM | Soft filter (AUC 0.55) | **Dihapus** |
| LGBM role | Entry signal (H1 context only) | Entry signal (full HTF context) |

### 2.2 Fitur Baru (103 − 84 = 19 fitur tambahan)

**16 fitur H4 dynamics & D1 HTF context** (sebelumnya dihitung tapi hilang dari parquet):

| Fitur | Keterangan |
|-------|-----------|
| `ema_21_slope_h4` | Rate of change EMA21 H4 per 4 bar |
| `ema_50_slope_h4` | Momentum medium-term H4 |
| `price_vs_ema_50_h4` | Posisi harga vs EMA50 H4 |
| `rsi_slope_h4` | Akselerasi momentum RSI H4 |
| `atr_percent_h4` | ATR sebagai % harga (volatilitas relatif) |
| `range_expansion_h4` | Ekspansi/kontraksi range H4 |
| `ema_50_d1` | Jarak harga ke EMA50 Daily |
| `ema_200_d1` | Jarak harga ke EMA200 Daily |
| `ema_50_slope_d1` | Momentum trend D1 (EMA50) |
| `ema_200_slope_d1` | Momentum trend D1 (EMA200) |
| `price_vs_ema_50_d1` | Harga di atas/bawah EMA50 Daily |
| `atr_d1_percentile` | Volatilitas sekarang vs 100 hari (percentile) |
| `d1_trend` | Trend D1: +1=bullish, -1=bearish, 0=flat |
| `d1_trend_strength` | Kekuatan trend D1 (EMA21 vs EMA50 gap) |
| `htf_alignment` | H4 dan D1 satu arah? 1=ya, 0=tidak |
| `d1_hh_hl_bias` | D1 structure: +1=HH/HL, -1=LH/LL, 0=flat |

**3 fitur Trend Quality** (correction detection — fix Temuan 1):

| Fitur | Keterangan | Relevansi |
|-------|-----------|-----------|
| `trend_accel_4h` | Rate of change `trend_strength` / 4 bar. Negatif saat trend UP = trend kelelahan | Deteksi koreksi awal |
| `vol_price_confirm` | `vol_regime × h4_trend`. Sangat negatif = volume tidak mendukung arah | DOGEUSDT case: VolR 0.01 × trend DOWN |
| `dist_from_8h_high` | Jarak harga dari highest high 8 bar terakhir (ATR-normalized). -0.3 = baru koreksi | Seberapa dalam koreksi sudah terjadi |

### 2.3 Perubahan Konfigurasi

| Parameter | Lama | Baru |
|-----------|------|------|
| `LSTM_SEQ_LEN` | 32 | **16** |
| `LGBM_CLASS_WEIGHTS[FLAT]` | 1.0x | **1.5x** |
| `H4_SOFT_FILTER_ENABLED` | True | **False** |
| `H1_THRESHOLD_LONG` | 0.62 | → rename **`LGBM_THRESHOLD_LONG`** = 0.62 |
| `H1_THRESHOLD_SHORT` | 0.62 | → rename **`LGBM_THRESHOLD_SHORT`** = 0.62 |

### 2.4 Perubahan File Model

| File | Status |
|------|--------|
| `lgbm_baseline.pkl` | Tetap (dipakai, retrain diperlukan) |
| `lstm_best.pt` | Tetap (dipakai, **retrain diperlukan** — n_features 84→103, seq_len 32→16) |
| `lstm_scaler.pkl` | Tetap (dipakai, **retrain diperlukan**) |
| `feature_cols_v2.json` | Tetap (akan berisi 103 fitur setelah retrain) |
| `lgbm_h4.pkl` | **Tidak dipakai lagi** |
| `h4_feature_cols.json` | **Tidak dipakai lagi** |

---

## 3. Compatibility Checklist — Aplikasi Inference

Sebelum deploy model cascade_v2, pastikan semua item ini diupdate di **repo inference**:

### 3.1 InferenceService (wajib)

- [ ] **`n_features` = 103** — pastikan `TradingLSTM` di-load dengan parameter yang benar dari `inference_config.json`
- [ ] **`seq_len` = 16** — update semua tempat yang memakai `LSTM_SEQ_LEN` atau hardcode 32
- [ ] **Hapus loading `lgbm_h4.pkl`** — jika ada kode yang load H4 model, hapus atau skip
- [ ] **Hapus H4 soft filter logic** — blok kode yang adjust confidence berdasarkan H4 bias
- [ ] **Update feature engineering** — panggil `engineer_features()` dari training repo versi terbaru yang menghasilkan 103 fitur
- [ ] **Verifikasi 19 fitur baru ada** — pastikan `feature_snapshot` include semua 103 kolom

### 3.2 Cascade Logic (wajib)

Cascade lama:
```python
# LAMA — HAPUS
h4_bias = get_h4_bias(lgbm_h4, df, h4_feat_cols)
h4_adjustment = H4_SOFT_ALIGN_BOOST if aligned else -H4_SOFT_OPPOSITE_PENALTY
adjusted_conf = h1_conf + h4_adjustment
# lanjut ke LSTM...
```

Cascade baru:
```python
# BARU
lgbm_proba = lgbm_model.predict_proba(features)   # (N, 3)
# Langsung ke LSTM soft adjustment, tanpa H4 layer
lstm_dir   = argmax(lstm_proba[i])
adj        = lstm_adjustment(lgbm_conf, lstm_dir, lgbm_dir)
final_conf = clip(lgbm_conf + adj, 0, 1)
if final_conf >= LGBM_THRESHOLD_LONG:   # 0.62
    signal = LONG
```

### 3.3 Constant Names (wajib jika import dari config)

| Lama | Baru |
|------|------|
| `H1_THRESHOLD_LONG` | `LGBM_THRESHOLD_LONG` |
| `H1_THRESHOLD_SHORT` | `LGBM_THRESHOLD_SHORT` |
| `H4_THRESHOLD_LONG` | dihapus |
| `H4_THRESHOLD_SHORT` | dihapus |
| `H4_SOFT_FILTER_ENABLED` | masih ada di config (False) |
| `H4_BINARY_THRESHOLD_*` | tidak dipakai di cascade |

### 3.4 inference_config.json (wajib)

Struktur yang berubah:

```json
// LAMA — "hierarchical_thresholds"
{
  "model_version": "hierarchical_v1",
  "hierarchical_thresholds": {
    "h4_binary_threshold_long": 0.60,
    "h4_binary_threshold_short": 0.60,
    "h1_threshold_long": 0.62,
    "h1_threshold_short": 0.62
  },
  "model_files": {
    "h4_lgbm": "lgbm_h4.pkl",
    "h1_lgbm": "lgbm_baseline.pkl",
    "h4_features": "h4_feature_cols.json"
  }
}
```

```json
// BARU — "cascade"
{
  "model_version": "cascade_v2",
  "cascade": {
    "lgbm_threshold_long": 0.62,
    "lgbm_threshold_short": 0.62,
    "lstm_confirmation": true,
    "lstm_adjust_mode": "tiered",
    "lstm_adjust_agree_boost": 0.05,
    "lstm_adjust_neutral_pen": 0.05,
    "lstm_adjust_opposite_pen": 0.08
  },
  "model_files": {
    "lgbm": "lgbm_baseline.pkl",
    "lstm": "lstm_best.pt",
    "lstm_scaler": "lstm_scaler.pkl",
    "features": "feature_cols_v2.json"
  },
  "model_architecture": {
    "n_features": 103,
    "lstm_hidden": 128,
    "lstm_layers": 2,
    "lstm_dropout": 0.3,
    "num_classes": 3
  },
  "inference": {
    "seq_len": 16
  }
}
```

### 3.5 Feature Engineering (wajib)

Pipeline engineer_features() di inference harus menghasilkan **103 fitur**. Cek dengan:

```python
from config import FEATURE_COLS_V3
assert len(FEATURE_COLS_V3) == 103, f"Ekspektasi 103, dapat {len(FEATURE_COLS_V3)}"

# Fitur baru yang wajib ada:
REQUIRED_NEW = [
    # H4 dynamics
    "ema_21_slope_h4", "ema_50_slope_h4", "price_vs_ema_50_h4",
    "rsi_slope_h4", "atr_percent_h4", "range_expansion_h4",
    # D1 HTF context
    "ema_50_d1", "ema_200_d1", "ema_50_slope_d1", "ema_200_slope_d1",
    "price_vs_ema_50_d1", "atr_d1_percentile",
    "d1_trend", "d1_trend_strength", "htf_alignment", "d1_hh_hl_bias",
    # Trend quality
    "trend_accel_4h", "vol_price_confirm", "dist_from_8h_high",
]
missing = [f for f in REQUIRED_NEW if f not in FEATURE_COLS_V3]
assert not missing, f"Fitur baru hilang: {missing}"
```

Pastikan juga cleaned parquet menyertakan kolom D1:
```
1d_open, 1d_high, 1d_low, 1d_close
```
Ini diperlukan untuk menghitung `ema_50_d1`, `d1_trend`, `atr_d1_percentile`, dll.

---

## 4. Audit Model — 4 Temuan dari Live Trading (May 2-7, 2026)

### [TEMUAN 1] Model Tidak Deteksi Koreksi Setelah Uptrend

**Data:**
```
May 5   → TONUSDT +39.7%, +42.8%, +57.1% (uptrend kuat)
May 6-7 → SUI -9.1%, SOL -6.8%, XRP -5.8%, 1000SHIB -6.0%, -6.1%
          7 open position LONG terjebak koreksi
```

**Bukti paling jelas — DOGEUSDT May 6, 21:05:**
```
DOGEUSDT  L  conf=0.70  Trd=DOWN  VolR=0.01  RR=0.37
```
H4 trend sudah DOWN, volume 1% dari rata-rata, RR 0.37 —
tapi model tetap predict LONG dengan confidence 0.70.

**Akar masalah (3 layer):**

| Layer | Masalah | Status |
|-------|---------|--------|
| **Feature gap** | Tidak ada "trend quality" metrics | ✅ **Fix: tambah `trend_accel_4h`, `vol_price_confirm`, `dist_from_8h_high`** |
| **HTF context** | D1 dan H4 slope tidak masuk FEATURE_COLS_V3 | ✅ **Fix: 16 fitur H4/D1 ditambahkan** |
| **LSTM window** | 32 bar terlalu lambat deteksi koreksi | ✅ **Fix: seq_len 32 → 16** |
| **Confidence paradox** | Model overconfident pada setup familiar | ✅ **Fix: FLAT weight 1x → 1.5x** |

**Tindakan setelah retrain:**
- [ ] Verifikasi `trend_accel_4h` muncul di top-20 SHAP importance
- [ ] Cek win rate LONG saat `htf_alignment=0` turun (model lebih hati-hati)

---

### [TEMUAN 2] TP/SL Selalu Fallback ke ATR (RR=1.33)

**Data:** 80%+ trade punya RR persis 1.33.

**Akar masalah:** `paper_trading.py:258` — saat breakout, H4 swing high sudah di-break sehingga kondisi `sh > entry` gagal → fallback ATR.

**Tindakan (di repo inference, tanpa retrain):**
- [ ] Saat breakout: `tp=3.0x, sl=2.0x` ATR → RR=1.5
- [ ] Atau: fallback ke rolling 24-bar high/low

**Parameter di `inference_config.json`:**
```json
{
  "fallback_tp_sl": { "tp_atr_mult": 3.0, "sl_atr_mult": 2.0 }
}
```

---

### [TEMUAN 3] 42% SL Hit adalah Wick Palsu

**Fix sudah deployed:** SL trigger pakai candle close (commit `1fdeea5`).
- [x] ~~SL trigger: low→close~~ ✅
- [ ] Tambah `sl_trigger: "close"` ke `inference_config.json` sebagai SSOT

---

### [TEMUAN 4] Confidence Tinggi ≠ Akurat

**Data:**
```
SUIUSDT   conf=0.97  loss -9.1%  (hold 6h)
SOLUSDT   conf=0.98  loss -6.8%  (hold 10h)
```

Model overconfident pada setup familiar (LONG di uptrend).

**Status:**
- [x] FLAT class weight dinaikkan 1x → 1.5x (model lebih ragu)
- [x] 3 fitur trend quality akan membantu kalibrasi
- [ ] Hitung ECE (Expected Calibration Error) setelah retrain — jika > 0.1, retrain calibrator

---

## 5. SSOT — Yang Harus Ada di `inference_config.json`

### ✅ Tetap (dipakai inference/trading)

| Section | Dipakai Oleh | Parameter Kunci |
|---------|-------------|-----------------|
| `inference` | `InferenceService.predict()` | confidence_threshold_entry, **seq_len=16**, label_map |
| `cascade` | Decision layer | lgbm_threshold_long/short, lstm_adjust_* |
| `labeling` | `swing_based_labeling()` | tp_atr_mult, sl_atr_mult, max_hold, min_rr |
| `fallback_tp_sl` | `_calculate_tp_sl()` | tp_atr_mult, sl_atr_mult |
| `feature_engineering` | `engineer_features()` | vp_window, vp_bins, swing_lookback |
| `model_architecture` | `_load_bundle()` | **n_features=103**, lstm_hidden/layers/dropout, **seq_len=16** |
| `model_files` | `_load_bundle()` | paths ke .pt/.pkl (tanpa H4) |
| `volatility_circuit_breaker` | `_circuit_breaker_active()` | enabled, atr_multiplier, lookback_bars |
| `risk` | `PaperTradingEngine` | modal_per_trade, leverage_recommended, fee_per_side |

### ➕ Tambah (belum ada, untuk SSOT)

```json
{
  "cooldown": {
    "tp_hit_hours": 2,
    "sl_hit_hours": 4,
    "time_exit_hours": 2,
    "default_hours": 4
  },
  "sl_trigger": "close",
  "label_distribution": { "LONG": 0.0, "SHORT": 0.0, "FLAT": 0.0 }
}
```

### ❌ Hapus / Tidak Relevan

| Section | Status |
|---------|--------|
| `backtest_summary` | Pindah ke `models/backtest_report.json` |
| `backtest_per_coin` | Pindah ke `models/backtest_report.json` |
| `coins_validated` | Pindah ke `models/backtest_report.json` |
| `hierarchical_thresholds` | Diganti `cascade` |
| `h4_binary_threshold_*` | Dihapus (H4 model tidak dipakai) |
| `h1_threshold_*` | Diganti `cascade.lgbm_threshold_*` |
| `signal_stability` | Scheduler config, bukan model |
| `same_dir_cooldown_hours` | Duplikat `cooldown.default_hours` |
| `created_at` | Metadata |
| `monitor` | Scheduler config |

---

## 6. Tuning Checklist (Jalankan di Training Repo)

### Sebelum training:

- [x] ~~`inference_config.json` di-cleanup~~ (dilakukan otomatis oleh 08_backtest.py)
- [x] ~~16 fitur H4/D1 masuk `FEATURE_COLS_V3`~~ ✅
- [x] ~~3 fitur baru "trend quality" ditambah di `core/features.py`~~ ✅
- [x] ~~`LSTM_SEQ_LEN` = 16~~ ✅
- [x] ~~`LGBM_CLASS_WEIGHTS` FLAT = 1.5x~~ ✅
- [ ] Update `TRAIN_END = 2026-05-07` (include data koreksi Mei 2026) — **paling impactful**
- [ ] Refetch data untuk semua koin hingga Mei 2026
- [ ] Label distribusi dicek — kalau LONG <20% atau >40%, adjust `labeling` parameters
- [ ] `labeling.min_rr` pertimbangkan naikkan ke 2.0 (dari 1.2)
- [ ] `labeling.max_sl_atr` pertimbangkan turunkan ke 2.5 (dari 3.0)
- [ ] Verifikasi kolom `1d_high/low/close` ada di cleaned parquet

### Saat training:

- [ ] Jalankan `python run_pipeline.py --train` (bukan `--train-h4`)
- [ ] Verifikasi `feature_cols_v2.json` berisi 103 fitur
- [ ] LGBM: class weights {SHORT:3x, FLAT:1.5x, LONG:3x} — sudah di config
- [ ] LSTM: weighted CrossEntropy + WeightedRandomSampler — sudah di `06_train_lstm.py`
- [ ] Hitung ECE (Expected Calibration Error) per class — jika > 0.1, retrain calibrator

### Setelah training:

- [ ] Jalankan `07_evaluate.py` — cek SHAP: `trend_accel_4h`, `htf_alignment` harus masuk top-20
- [ ] Jalankan `08_backtest.py` — target winrate > 60%, max DD < 25% (lev5x)
- [ ] `label_distribution` diisi dari hasil labeling aktual
- [ ] `cooldown` section ditambah ke `inference_config.json`
- [ ] `sl_trigger: "close"` ditambah
- [ ] Deploy: copy `.pt`, `.pkl`, `inference_config.json`, `feature_cols_v2.json` ke repo inference

---

## 7. Parameter yang Bisa Di-tuning via UI `/models` (Tanpa Retrain)

Parameter ini ada di `inference_config.json` dan bisa diedit langsung via UI:

- `inference.confidence_threshold_entry` — filter minimum confidence (sekarang 0.62)
- `cascade.lgbm_threshold_long` — LGBM minimum untuk entry LONG (sekarang 0.62)
- `cascade.lgbm_threshold_short` — LGBM minimum untuk entry SHORT (sekarang 0.62)
- `cascade.lstm_adjust_opposite_pen` — seberapa keras LSTM menolak (sekarang 0.08)
- `volatility_circuit_breaker.atr_multiplier` — VCB sensitivity (sekarang 3.0)
- `fallback_tp_sl.tp_atr_mult` — TP multiplier saat breakout (sekarang 2.0, target 3.0)

---

## 8. Kolom Diagnostik Trades (Output CSV)

```
Opened | Closed | Coin | Dir | Conf | Entry | Exit | TP | SL | ATR |
%Hi | %Lo | RR | Trd | VolR | PnL$ | PnL% | Hold | Reason | Status
```

### Panduan baca cepat:

| Pola | Artinya | Tindakan |
|------|---------|----------|
| **Conf >0.90 + loss >5%** | Model overconfident | Cek kalibrasi, naikkan threshold |
| **%Hi <1% + L + sl_hit** | Entry dekat resistance, gagal break | Filter entry dekat H4 swing high |
| **RR=1.33 semua** | Swing-based TP/SL tidak aktif | Cek Temuan 2 — breakout mode |
| **Trd=UP + VolR <0.3 + loss** | Low-vol correction | Monitor `vol_price_confirm` di SHAP |
| **Trd=DOWN + VolR <0.1 + tetap LONG** | Model tidak baca regime | Cek apakah `trend_accel_4h` masuk fitur top |
| **hold=0-1 + sl_hit** | Wick palsu | Seharusnya sudah difilter oleh close-based SL |
| **htf_alignment=0 + LONG entry** | H4 dan D1 berlawanan | Fitur baru — monitor apakah model lebih hati-hati |
