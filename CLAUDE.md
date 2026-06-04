# CLAUDE.md — Riset Pemodelan Crypto Trading

## Project Overview

Sistem trading kripto berbasis ML untuk Binance Futures.
*   **Active Production Version**: **`cascade_v4.1`** (104 fitur, FEATURE_COLS_V3 + 3 Volatility Spike Detectors).
*   **Active Research Version**: **`cascade_v4.3`** — LSTM Momentum Detector (H1 sequence, momentum labels, weighted fusion dengan LGBM).

Arsitektur: LGBM Classifier (entry) + LSTM Momentum Detector (weighted fusion) → **Guardian v3 (dynamic exit)**.

Periode data: 2020-01-01 s/d 2026-04-01. Timeframe: H1 base, H4 untuk swing + regime.
**TRAIN_CUTOFF_DATE = 2025-11-01** — training hanya di 2020 – Okt 2025. Holdout test di Nov 2025 – Apr 2026 (genuine temporal OOS, tanpa overlap).
21 koin: SOL, ETH, BNB, XRP, DOGE, TON, ADA, TRX, 1000SHIB, AVAX, LINK, DOT, SUI, POL, NEAR, 1000PEPE, TAO, ARB, XAUT, HBAR, ONDO.

## Architecture

```
┌─────────┐    ┌──────────┐    ┌──────────────┐    ┌───────────────┐
│  LGBM   │───▶│   LSTM   │───▶│  Confidence  │───▶│  Swing/ATR    │
│ Classif │    │  Soft    │    │  Filter      │    │  TP/SL Gate   │
│ 3-class │    │ Confirm  │    │  >= 0.65     │    │               │
└─────────┘    └──────────┘    └──────────────┘    └───────┬───────┘
                                                           │
                                            ┌──────────────┘
                                            ▼
                                    ┌──────────────┐
                                    │  RR Gate     │
                                    │  min_rr=0.5  │
                                    │  min_tp=1.2x │
                                    │  max_sl=4.0x │
                                    └──────┬───────┘
                                           ▼
                                    ┌──────────────┐
                                    │  EXECUTE     │
                                    │  max_hold=24 │
                                    └──────┬───────┘
                                           ▼
                                    ┌──────────────┐
                                    │ Guardian v3  │ ← per-bar dynamic exit
                                    │ HOLD / PART  │    104 feat + 7 dynamic
                                    │ / FULL EXIT  │    multiclass LGBM
                                    └──────────────┘
```

- **LGBM**: 3-class (SHORT=0, FLAT=1, LONG=2), **104 features** (`feature_cols_v2.json`), cost-sensitive weights {0:3, 1:1.5, 2:3}, GPU OpenCL. Asymmetric threshold: LONG ≥0.75, SHORT ≥0.60
- **LSTM Momentum Detector** (cascade_v4.3): `ManualLSTMCell` custom (DirectML compatible), seq_len=32 H1 bars, 12 fitur, hidden=128, 2-layer. Input: H1 sequence, Target: momentum labels N=8. Weighted fusion: `combined = 0.65×lgbm + 0.35×lstm`. Training via DirectML GPU, inference via CPU
- **Guardian v3**: Multiclass (0=HOLD, 1=PARTIAL_EXIT, 2=FULL_EXIT), **104 static + 7 dynamic features**, GPU OpenCL. Partial exit = 50% posisi. min_hold_bars=0, activation_atr=0.0 (instant activation, tidak ada zona buta)
- **TP/SL**: Hybrid H4 Swing + ATR Fallback (non-ML). `TP_SL_HYBRID_MODE=True`

## Cascade Flow (detail)

```
LGBM predict 3-class
  │
  ├─ LGBM LONG >= 0.75 / SHORT >= 0.60 → LSTM hard_consensus adjustment → conf >= 0.65? → ENTRY SIGNAL
  │       (lgbm_threshold_long = 0.75, lgbm_threshold_short = 0.60 — asymmetric)      FAIL → FLAT
  │
  │   LSTM hard_consensus mode (lstm_adjust_mode="hard_consensus"):
  │     agree    → +0.05 boost
  │     neutral  → +0.00 (tidak ada penalty)
  │     opposite → −0.99 penalty  ← efektif membunuh trade jika LSTM berlawanan
  │
  └─ Keduanya < 0.65 (LGBM FLAT) → FLAT (selesai)
       │
       └─ FLAT REVIEW = DISABLED (flat_review_threshold=0.0)
            Terbukti menambah 2,500+ trade dengan WR 39% — menurunkan WR dari 78% ke 57%.
            Detail: EXPERIMENTS.md § 2026-05-12

ENTRY → Guardian v3 per-bar check (sejak bar 0, tanpa batas ATR minimum — instant activation):
  ├─ HOLD (0)          → lanjut scan
  ├─ PARTIAL_EXIT (1)  → tutup 50% posisi, lanjut 50% sisanya
  └─ FULL_EXIT (2)     → tutup seluruh posisi
```

**Alur aktif**: LGBM entry → LSTM confirm → Trend Alignment gate → Guardian v3 dynamic exit.
TRAILING_STOP_ENABLED=False (Guardian solo beats trailing). LSTM TIDAK bisa meng-override FLAT.
TREND_ALIGNMENT_ENABLED=True (with_trend penalty=0.10, counter_trend boost=0.05).

## Final Results

> ⛔ **SEMUA METRIK DI BAWAH INI DICABUT — DATA LEAKAGE TERDETEKSI (2026-06-04)**
> Leakage ditemukan di tiga komponen: holdout split, feature engineering, dan Guardian training.
> Metrik 88.93% WR, $169,626 PnL, dan seluruh angka turunannya **tidak valid**.
> Jangan gunakan sebagai referensi performa. Akan diisi ulang setelah retrain bersih.

~~### 08 Backtest — Walk-Forward Purged CV (2020-2025, 20 koin)~~
~~### 09 Holdout — Genuine Temporal OOS (Nov 2025 – Apr 2026)~~
~~### Guardian v2 → v3 Transition~~

Detail leakage: `EXPERIMENTS.md § 2026-06-04`

### Cascade v4.1 (Production — Volatility-Aware + Asymmetric Entry + Instant Guardian)
*   **Status**: Production — deployed via `tools/deploy_model.py`.
*   **Training Date**: 2026-05-29
*   **Fitur**: **104 kolom** — FEATURE_COLS_V3 lengkap: H1/H4 EMA, Smart Money v3/v4 (OFI, VWDP, CVD, VSA), Game Changer v4.0 (Relative Strength, Liquidation Levels, Whale/Retail Divergence), dan Volatility Spike Detectors v4.1 (`atr_zscore_20d`, `atr_percentile_h1`, `vol_spike_zscore`).
*   **Training & Holdout OOS Setup**:
    *   **Training Period**: 2020-01-01 s/d 2025-11-01 (`TRAIN_CUTOFF_DATE = 2025-11-01`).
    *   **Holdout Temporal OOS**: 2025-11-01 s/d 2026-04-01.
    *   **Model Runs Path**: `models/runs/cascade_v4.1/`

### cascade_v4.3 (LSTM H1 Sequence — Selesai 2026-05-30)
*   **Status**: Selesai. Mean F1 = 0.3339 ≈ random baseline 0.333. Model tersimpan tapi tidak deploy.
*   **Fixes applied**: no_weighted_sampler, fold_scaler, weight_decay=1e-4, patience=15
*   **Root cause F1 rendah**: Fitur H4 (h4_trend, trend_strength, ema_21_slope_h4) hampir flat dalam 32 H1 bars → LSTM tidak bisa belajar pola temporal. Bukan leakage (sudah diaudit).
*   **Model Runs Path**: `models/runs/cascade_v4.3/`

### 💡 Research State: Cascade v4.4 (LSTM Trajectory Features — Next Run)
*   **Status**: Siap dijalankan setelah restart.
*   **Perubahan utama dari v4.3**:
    *   Fitur LSTM baru — hapus snapshot H4, ganti trajectory H1: `log_ret_5`, `log_ret_20`, `ofi_raw`, `ofi_acceleration`, `vwdp_smooth`, `vol_ratio_20`
    *   Labels N=12 (dari N=8) — horizon lebih panjang, label lebih decisive, FLAT turun ~48% → ~40%
    *   LR=0.001, batch_size=512 (dikembalikan ke default — 0.0014/1024 terlalu tinggi)
    *   Penalti LSTM FLAT = 0.03 (dari 0.0) — LGBM tidak bebas saat LSTM netral
*   **Pipeline**: `05a --n 12 → 05b → 05c --run-id cascade_v4.4`
*   **Model Runs Path**: `models/runs/cascade_v4.4/` (setelah retrain)
*   **Target F1**: > 0.36. Jika masih ≤ 0.35, evaluasi alternatif arsitektur.


## Cross-Repo: Production (swint_tradev2)

### 🎛️ Pusat Kendali Tunggal & Alur Kerja Satu Arah (One-Way Workflow)
Repositori **`Riset_pemodelan`** bertindak sebagai **Pusat Kendali Tunggal** (satu-satunya tempat kerja aktif developer). Repositori **`swint_tradev2`** adalah **lingkungan eksekusi pasif (runtime target)**. 

Seluruh aktivitas pengembangan, analisis, dan sinkronisasi bersifat **otomatis dan satu arah** dari repositori Riset ini:
*   **Analisis Tanpa Copy (Zero-Copy)**: Script `tools/trade_analyzer.py` di repo ini membaca data live trading langsung dari `D:\Apps-Dev\swint_tradev2\hasil_livetrading.csv`. Developer **tidak perlu** menyalin file secara manual atau berpindah repositori untuk melakukan analisis.
*   **Deployment Otomatis (Automated Handover)**: Pembaruan model, scaler, dan parameter dikirim secara otomatis ke produksi menggunakan `python tools/deploy_model.py` dari repo ini. Developer **tidak boleh** memodifikasi file atau parameter di repo produksi secara manual.

Dengan alur ini, Anda **hanya perlu membuka dan menjalankan perintah dari repositori Riset (`Riset_pemodelan`)**. Repo produksi cukup berjalan pasif menerima hasil deployment.

Repo production di `D:\Apps-Dev\swint_tradev2`. File kunci yang bisa langsung dibaca:

| File | Purpose |
|------|---------|
| `D:\Apps-Dev\swint_tradev2\CLAUDE.md` | Dokumentasi lengkap production system (Flask, scheduler, live trading) |
| `D:\Apps-Dev\swint_tradev2\hasil_livetrading.csv` | Live trade history (102+ trades, cascade_v2/v3/lstm) |
| `D:\Apps-Dev\swint_tradev2\TRADE_ANALYSIS_REPORT.md` | Laporan analisis trading terbaru oleh /analyze |
| `D:\Apps-Dev\swint_tradev2\AUDIT_REPORT.md` | Laporan audit terbaru oleh /audit |
| `D:\Apps-Dev\swint_tradev2\models\inference_config.json` | Konfigurasi inference production (parameter cascade, Guardian, filter) |
| `D:\Apps-Dev\swint_tradev2\app\services\paper_trading.py` | Exit logic + Guardian v3 implementation (EARLY + MOMENTUM mode) |
| `D:\Apps-Dev\swint_tradev2\app\services\signal_filter.py` | Filter chain (structural, VCB). R1 & R2 dihapus 2026-05-24. |
| `D:\Apps-Dev\swint_tradev2\app\services\inference.py` | Cascade v3.1 inference pipeline production |
| `D:\Apps-Dev\swint_tradev2\app\services\guardian_service.py` | Guardian v3 dynamic exit service |
| `D:\Apps-Dev\swint_tradev2\core\features.py` | Feature engineering production (104 feat, FEATURE_COLS_V3) |

**Konvensi**: Jika perlu membaca file dari repo production, gunakan absolute path `D:\Apps-Dev\swint_tradev2\...`.

## Holdout Setup — Parameter Live Aktual (2026-05-24)

Agar holdout backtest mendekati kondisi live, gunakan parameter berikut di `config.py` dan `evaluator.py`:

### Entry Filter Chain
| Parameter | Nilai Live | Catatan |
|-----------|-----------|---------|
| `lgbm_threshold_long` | **0.65** | |
| `lgbm_threshold_short` | **0.65** | |
| `confidence_threshold_entry` | **0.65** | Final threshold setelah LSTM adjustment |
| `flat_review_threshold` | **0.0** | Disabled — FLAT tidak di-review LSTM |
| R1 (SHORT block saat H4 UP) | **DIHAPUS** | Holdout 11 bulan: memblok 2,100+ SHORT profitable, WR 82-83%, rugi PnL ~$12,800 |
| R2 (vol_regime transition block) | **DIHAPUS** | Sama, dihapus bersamaan dengan R1 |

### LSTM Cascade Mode
| Parameter | Nilai Live | Catatan |
|-----------|-----------|---------|
| `lstm_adjust_mode` | **"hard_consensus"** | |
| `lstm_adjust_agree_boost` | **0.05** | LSTM searah → +0.05 |
| `lstm_adjust_neutral_pen` | **0.0** | LSTM neutral → tidak ada penalty |
| `lstm_adjust_opposite_pen` | **0.99** | LSTM berlawanan → −0.99 (efektif kill trade) |

### Filter Chain
| Filter | Status | Parameter |
|--------|--------|-----------|
| VCB | **enabled** | `atr_multiplier=3.0`, `lookback_bars=24` |
| Structural filter | **enabled** | `max_swing_deviation_pct=0.15`, `swing_max_age_hours=48`, `breakout_tolerance_pct=0.03` |
| Trend alignment | **disabled** | Double-counting H4 trend — menurunkan PnL |
| Cooldown | **disabled** | Cooldown off: PnL 2× lebih tinggi |

### RR Gate
| Parameter | Nilai Live |
|-----------|-----------|
| `min_rr` | **0.45** |
| `min_tp_atr` | **1.2** |
| `max_sl_atr` | **4.0** |
| `swing_bumper_atr` | **0.5** |

### Guardian v3
| Parameter | Nilai Live |
|-----------|-----------|
| `exit_threshold` | **0.65** |
| `min_hold_bars` | **3** |
| `activation_atr` | **1.5** |
| `partial_exit_ratio` | **0.50** |

### TP/SL Hybrid
| Parameter | Nilai Live |
|-----------|-----------|
| `tp_atr_mult` | **2.0** (fallback) |
| `sl_atr_mult` | **1.5** (fallback) |
| `swing_bumper_atr` | **0.5** |
| max holding | **24 bar** |

## Referensi Internal

- **MODEL_DEPLOYMENT_BRIDGE.md** — Kontrak sinkronisasi parameter & model lintas repositori.
- **EXPERIMENTS.md** — Logbook perubahan parameter & temuan eksperimen. Baca sebelum mengubah parameter.
- **Model registry**: `models/model_registry.json` — model aktif & metrik baseline
- **Holdout results**: `models/runs/holdout_20260515_001906/holdout_backtest_results.json`
- **Database Eksperimen**: `reports/experiments/` — Laporan-laporan point-in-time historis yang tersusun secara kronologis. Otomatis menghasilkan laporan `{run_id}_holdout_report.md` setiap kali backtest holdout selesai dijalankan.

## Key Files

| File | Role |
|------|------|
| `config.py` | Semua parameter terpusat — **source of truth**, jangan diduplikasi |
| `core/features.py` | Feature engineering + swing labeling v3 |
| `core/models.py` | `TradingLSTM`, `_CellLSTM`, `_ManualLSTMCell` |
| `core/evaluator.py` | `simulate_trades_swing()` + Guardian per-bar check + partial exit |
| `core/utils.py` | Logger, device utils, `chunk_time_range()` |
| `core/fetchers.py` | Binance data fetch (`KLINE_LIMIT=1000`) |
| `core/binance_client.py` | HTTP client Binance |
| `pipeline/01_fetch.py` | Fetch semua koin |
| `pipeline/02_clean.py` | Clean + resample |
| `pipeline/03_engineer.py` | Feature engineering & swing labeling pipeline |
| `pipeline/04_train_lgbm.py` | LGBM entry model training (TRAIN_CUTOFF_DATE filter) |
| `pipeline/05a_generate_momentum_labels.py` | Generate momentum labels H1 (N=8, majority direction + magnitude filter) |
| `pipeline/05b_build_h1_sequences.py` | Build H1 sequence dataset (32 bar × 12 fitur) untuk LSTM momentum |
| `pipeline/05c_train_lstm_h1.py` | **LSTM Momentum Detector training** — H1 sequence, momentum labels, no double-weighting |
| `pipeline/06_train_guardian.py` | **Guardian v3 training** — multiclass LGBM, TRAIN_CUTOFF_DATE filter |
| `pipeline/07_holdout_backtest.py` | Genuine OOS holdout backtest (Nov 2025 – Apr 2026) |
| `tools/generate_report.py` | **Retrospective Report Generator** — Merekonstruksi laporan Markdown premium komprehensif dari run historis (misal: `cascade_v3.1`) |
| `tools/train_all.py` | **Master Training Orchestrator** — Menanyakan versi secara interaktif, melakukan training lengkap, dan otomatis mendokumentasikan metrik OOS ke `model_registry.json` dan `EXPERIMENTS.md` |
| `pipeline/shared.py` | `SequenceDataset` + `build_purged_folds()` |
| `pipeline/backtest_utils.py` | `hierarchical_predict()` + feature alignment via `feature_name_` |
| `analysis/evaluate.py` | Penganalisis CSV hasil trading (Winrate, PF, streak, dll.) |
| `analysis/analyze_min_hold.py` | Optimal `MIN_HOLD_BARS` finder (distribusi holding aktual) |
| `analysis/compare_max_sl.py` | Tuning & perbandingan performa SL ATR 3.0 vs 4.0 |
| `analysis/compare_hybrid_vs_pure.py` | Perbandingan mode Hybrid TP/SL vs Pure Tier |
| `analysis/compare_aspects.py` | Pengujian 9 aspek arsitektural TP/SL di data holdout |
| `analysis/visualize_aspects.py` | Visualisasi batang delta winrate & heatmap matriks korelasi aspek |

## Pipeline Sequence (Order Matters)

```
01_fetch → 02_clean → 03_engineer → 04_train_lgbm
                                   → 05a_generate_momentum_labels
                                   → 05b_build_h1_sequences
                                   → 05c_train_lstm_h1
                                   → 06_train_guardian → 07_holdout_backtest
```

**LSTM Pipeline (05a → 05b → 05c)** — dijalankan paralel setelah 03_engineer, sebelum 06:
- `05a`: generate momentum labels dari parquet existing (N=8 H1, tidak perlu re-fetch)
- `05b`: build H1 sequence dataset (32 bar × 12 fitur, sliding window)
- `05c`: training LSTM dengan momentum labels, tanpa WeightedRandomSampler, patience=10

**Arsip** (tidak dipakai lagi, ada di `pipeline/archive/`):
- `05_train_lstm.py` — H1 flat features + swing labels (always output FLAT)
- `05_train_lstm_seq_sweep.py` — sweep variant H1
- `05b_build_h4_sequences.py` — H4 sequence (skala tidak selaras dengan target H1)
- `05c_train_lstm_momentum.py` — H4 LSTM training (double weighting, F1 ≈ random)

**Data flow**: Semua training script filter `df.index < TRAIN_CUTOFF_DATE` (2025-11-01).
Holdout test menggunakan data setelah cutoff — genuine temporal OOS.

## Key Learnings

### Guardian v3 — DICABUT (data leakage, 2026-06-04)

- Klaim WR 89%, PnL +57%, semua 21 koin positif **tidak valid** — ada leakage di holdout split, feature engineering, dan Guardian training
- Arsitektur Guardian v3 (multiclass, partial exit) tetap dipertahankan; hanya metrik hasil evaluasinya yang dicabut
- **Guardian > Trailing stop**: Guardian v3 mengalahkan trailing 2x ATR di semua metrik
- **Feature alignment robust**: `model.feature_name_` + zero-fill mencegah mismatch kolom
- **Partial exit belum optimal**: Minority class 4.5%, perlu monitoring lebih lanjut

### ML for TP/SL — ALL FAILED

| Approach | Result | Why |
|----------|--------|-----|
| TP/SL Regressor (LGBMRegressor) | WR 75%→37%, DD 64%→113% | SL R²=0.05 — entry bar has no signal for 24-bar ahead multiplier |
| Safe SL Classifier (Binary) | AUC=0.62, never triggered | Cannot predict if structural level holds from 1 bar |
| Regime Classifier (ML Binary) | Always predicts RANGING | Same problem — 1 bar can't predict 24-bar regime |
| Rule-Based Regime | Trend%=0-3%, no effect | Thresholds too conservative |

### What Works

- **Guardian v3** — dynamic exit, WR 89%, DD 42%, PnL +49% vs baseline di temporal OOS (21 koin)
- **TP → momentum mode** — TP tidak hard-close, trigger Guardian ride profit. Trade +72%, PnL +57% vs Guardian v2
- **Swing/ATR gate** — structural levels are real, statistically meaningful
- **Walk-forward purged CV** — prevents look-ahead leakage
- **Confidence filter** — reduces noise trades (threshold 0.62)
- **SHORT ≈ LONG** — model tidak bias arah; SHORT sedikit lebih akurat (+2.5%) karena market structure

### What We Learned About LSTM (2026-05-12)

- **LSTM FLAT review** menambah volume 2x tapi WR 39% — disabled
- **LSTM opposite penalty** 0.08 → 0.04 — kurangi blocked trades
- Detail: `EXPERIMENTS.md`

## Important Constraints

- **Python 3.12** on **Windows 10** with **AMD RX 6600 GPU** (DirectML)
- Shell: PowerShell (not bash). Use `;` not `&&` for chaining
- **Encoding**: Terminal is cp1252 — avoid unicode arrows (→) in logger messages
- **LSTM**: Custom `ManualLSTMCell` for DirectML compatibility. Train on GPU, infer on CPU
- **LGBM**: `device_type="gpu"` via OpenCL (compatible with AMD)
- **TRAIN_CUTOFF_DATE = 2025-11-01** — tidak boleh ada data testing bocor ke training
- **KLINE_LIMIT = 1000** — Binance max 1000 klines per request (sebelumnya 1500 → gap 21 hari)
- **TP/SL regressor/classifier files DELETED** — jangan re-implement tanpa diskusi
- **Feature alignment via `model.feature_name_`** — mencegah mismatch fitur 103 vs 104
- **Jangan duplikasi isi config.py** — baca langsung dari file
- **Jangan tulis riwayat perubahan di sini** — gunakan `EXPERIMENTS.md`
- **Retraining Protocol**: SEBELUM melakukan training model baru, tanyakan secara eksplisit kepada user mengenai nama versi model yang diinginkan (misalnya `cascade_v3.2`). Setelah ditentukan, dokumentasikan secara lengkap detail versi tersebut di `CLAUDE.md` mencakup: tanggal training, deskripsi fitur yang dipakai, parameter temporal holdout OOS, serta path penyimpanan model (`models/runs/{run_id}/`) untuk mencegah file sampah (*garbage*).

## Slash Commands

### /trade-analysis

Ketika user mengetik `/trade-analysis`, ikuti instruksi lengkap di `.claude/commands/trade-analysis.md`.

Ringkasan alur:
1. Parse argumen → jalankan `python tools/trade_analyzer.py`
2. Baca CSV sendiri → analisis mendalam (loss streak, counter-trend, Guardian, anomali coin, open positions)
3. Tampilkan: Scorecard → Temuan Kritis → Analisis Per Model → Rekomendasi
4. Rekonstruksi dampak rekomendasi: simulasikan tiap rekomendasi ke data historis dengan script Python, tampilkan tabel proyeksi scorecard per skenario
5. Evaluasi open positions risk
6. Simpan report ke `reports/TRADE_ANALYSIS_REPORT.md`

Contoh pemanggilan:
```
/trade-analysis                              → default livetrade.csv
/trade-analysis path/to/file.csv             → file lain
/trade-analysis vol=0.03 dev=0.10            → override threshold filter
/trade-analysis no-filter                    → baseline saja
```

Default: `--file livetrade.csv`, `--p2 0.05`, `--p4 0.08`, `--output reports/TRADE_ANALYSIS_REPORT.md`
