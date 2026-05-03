# Audit Report: Training Readiness — Pipeline Sinkronisasi & Kelayakan Eksekusi

**Tanggal:** 2026-05-03  
**Auditor:** AI Code Analysis  
**Scope:** Apakah pipeline siap dijalankan untuk training penuh (01_fetch → 08_backtest)

---

## Ringkasan Eksekutif

**Verdict: ✅ LAYAK DIJALANKAN**

Setelah verifikasi menyeluruh terhadap seluruh pipeline (01_fetch → 08_backtest), `run_pipeline.py`, `config.py`, dan sinkronisasi dengan `Riset_pemodelan.ipynb`, **tidak ditemukan showstopper**. Semua 5 perbaikan dari audit sebelumnya telah teraplikasi dengan benar. Terdapat 3 temuan minor yang perlu dicatat tetapi **tidak menghalangi eksekusi training**.

---

## Temuan Per Kategori

### ✅ PASS — Semua Berfungsi Dengan Benar

| Area | Status | Detail |
|------|--------|--------|
| Pipeline Execution Order | ✅ | `run_pipeline.py` urut: 01→02→03→04→05→06→07→08. `--train` = 04+05+06 |
| CLI Argument Parsing | ✅ | Semua script punya `--all`, `--run-id`. Argumen diverifikasi per script |
| H4 Walk-Forward Gap | ✅ | `TimeSeriesSplit(n_splits=8, gap=6)` = ~24 jam purge |
| H1 Walk-Forward Gap | ✅ | `TimeSeriesSplit(n_splits=4, gap=24)` = ~24 jam purge |
| LSTM Purge Gap | ✅ | `PURGE_GAP_BARS=5` dengan purge di kedua sisi fold |
| Model File Checks (08) | ✅ | Check: `lgbm_baseline.pkl`, `lstm_best.pt`, `lstm_scaler.pkl`, `feature_cols_v2.json`. H4 model opsional |
| Data Independence (H4↔H1) | ✅ | Kedua training baca dari `LABEL_DIR/*_features_v3.parquet` yang sama; tidak ada dependency silang |
| Previous Audit Fixes | ✅ | Semua 5 fix (percentile logging, pass rate, all-folds calibration, tiered LSTM, inference.py logging) terkonfirmasi |
| Binance Multi-Endpoint | ✅ | `core/binance_client.py:27-33` punya 5 endpoint fallback — fapi → data-api → api1 → api2 → api3 |
| requirements.txt | ✅ | Semua dependency kunci (lightgbm, torch, shap, scikit-learn, pandas) tercantum |
| No PostgreSQL Dep | ✅ | Pipeline training tidak bergantung pada database — independence terjamin |

### 🟡 MINOR — Perlu Dicatat, Tidak Blokir

#### Temuan 1: `feature_cols_v2.json` Overwrite Antar Script

| Field | Detail |
|-------|--------|
| **Lokasi** | `pipeline/05_train_lgbm_h1.py:207` dan `pipeline/06_train_lstm.py:288` |
| **Deskripsi** | Kedua script menyimpan `feature_cols_v2.json` ke `MODEL_DIR` yang sama. Karena 06 berjalan setelah 05, file ditimpa dengan feature cols dari LSTM. |
| **Dampak** | **Saat ini tidak masalah** karena kedua script menggunakan logika identik: `[c for c in df.columns if c not in NON_FEATURE_COLS]` dengan `NON_FEATURE_COLS = {"label", "h4_swing_high", "h4_swing_low"}`. Output feature list identik. |
| **Risiko** | Jika suatu saat `06_train_lstm.py` melakukan preprocessing yang berbeda (drop kolom, filter), feature cols akan divergen dan menyebabkan **dimension mismatch** saat `08_backtest.py` memuat `feature_cols_v2.json`. |
| **Rekomendasi** | Simpan feature cols per-model: `h1_feature_cols.json` dan `lstm_feature_cols.json`. Atau jika memang identik, simpan sekali saja di 03_engineer. |

#### Temuan 2: `sed` Workaround di Notebook Mungkin Redundan untuk Klines

| Field | Detail |
|-------|--------|
| **Lokasi** | `Riset_pemodelan.ipynb:817` — `sed -i 's|https://fapi.binance.com|https://data-api.binance.vision|g' config.py` |
| **Deskripsi** | Notebook mengubah `BINANCE_BASE_URL` di config.py untuk fallback kline. Namun `core/binance_client.py:27-33` sudah memiliki `KLINE_ENDPOINTS` dengan multi-endpoint fallback — data-api.binance.vision sudah sebagai fallback ke-2. |
| **Dampak** | sed **masih diperlukan untuk non-kline endpoints** (funding rate, OI) yang menggunakan `BINANCE_BASE_URL` dari config.py (`01_fetch.py:108` → `BinanceClient(base_url=BINANCE_BASE_URL)`). |
| **Verifikasi** | `config.py:44`: `BINANCE_BASE_URL = "https://fapi.binance.com"` — koneksi funding rate, OI, taker ratio tetap perlu fallback jika fapi diblokir. |

#### Temuan 3: `04_train_lgbm_h4.py` Tidak Punya `--coins` Flag

| Field | Detail |
|-------|--------|
| **Lokasi** | `pipeline/04_train_lgbm_h4.py:323-327` |
| **Deskripsi** | Hanya punya `--all` dan `--run-id`. Tidak seperti 05/06 yang juga punya mekanisme `--all`/`TRAINING_COINS` scoping (via `ALL_COINS` vs `TRAINING_COINS` di config). |
| **Dampak** | Konsisten dengan desain — H4 selalu dilatih pada koin yang tersedia. Bukan bug. |

---

## Jalur Eksekusi yang Teridentifikasi

```
run_pipeline.py --all
  │
  ├── 01_fetch.py ───────────────────────────────────── Binance API (multi-endpoint fallback)
  │     │
  │     └── core/binance_client.py · KLINE_ENDPOINTS [fapi → data-api → api1→api2→api3]
  │
  ├── 02_clean.py ────────────────────────────────────── Multi-TF alignment → H1 grid
  │     │
  │     └── audit_leakage() · fix_ohlc() · detect_gaps()
  │
  ├── 03_engineer.py ─────────────────────────────────── 85 fitur V3 + swing labels
  │     │
  │     └── engineer_features() · validate_features() → _features_v3.parquet
  │
  ├── 04_train_lgbm_h4.py ────────────────────────────── Resample H4 → Binary (LONG/SHORT)
  │     │                                                 Walk-Forward CV (8 fold, gap 6)
  │     │                                                 Isotonic Calibration (all folds)
  │     │
  │     └── Output: lgbm_h4.pkl + h4_calibrator.pkl + h4_feature_cols.json
  │
  ├── 05_train_lgbm_h1.py ────────────────────────────── 3-class LGBM (SHORT/FLAT/LONG)
  │     │                                                 Walk-Forward CV (4 fold, gap 24)
  │     │
  │     └── Output: lgbm_baseline.pkl + feature_cols_v2.json ← overwritten by 06
  │
  ├── 06_train_lstm.py ───────────────────────────────── Purged Walk-Forward CV (8 fold, gap 5)
  │     │                                                 LSTM (seq_len=32, hidden=128)
  │     │
  │     └── Output: lstm_best.pt + lstm_scaler.pkl + feature_cols_v2.json ← overwrites 05
  │
  ├── 07_evaluate.py ─────────────────────────────────── Multi-model eval (H4+H1+cascade)
  │
  └── 08_backtest.py ─────────────────────────────────── Walk-forward backtest
        │                                                 hierarchical_predict(H4→H1→LSTM)
        └── Output: backtest_results.json + charts
```

---

## Hipotesis Risiko (Diurutkan dari Paling Mungkin)

### 1. ⚠️ Feature Cols Divergence (Risiko Masa Depan)
**Jika** suatu saat `06_train_lstm.py` mengubah preprocessing (misal: drop kolom tertentu sebelum training), `feature_cols_v2.json` akan menyimpan daftar feature LSTM yang berbeda dari H1. Saat `08_backtest.py` memuat file ini, ia akan memberikan feature set LSTM ke H1 LGBM, menyebabkan:
- `ValueError: Number of features mismatch` jika feature list lebih pendek
- Silent wrong predictions jika feature list tidak akurat

**Probabilitas:** Rendah saat ini, tapi meningkat setiap kali ada modifikasi kode.

### 2. 🟢 Binance FAPI Blocking (Risiko Eksternal)
Jika IP cloud (Colab/VPS) memblokir `fapi.binance.com`:
- **Klines**: ✅ Aman — `binance_client.py` fallback ke `data-api.binance.vision` secara otomatis
- **Funding Rate / OI**: ⚠️ Bergantung pada `BINANCE_BASE_URL` yang di-sed di notebook
- **Test**: ✅ `test_connection()` mencoba semua 5 endpoint

### 3. 🟢 H4 Label Imbalance
`04_train_lgbm_h4.py:210` sudah memperingatkan jika LONG+SHORT < 2% dari total bars. Ini tidak blocking tapi bisa menyebabkan model bias ke FLAT (yang kemudian di-drop) atau binary classifier tidak balanced.

---

## Ringkasan Status Perbaikan Sebelumnya

| # | Item | Status | File |
|---|------|--------|------|
| 🔴 2.1 | Log percentile post-calibration | ✅ | `04_train_lgbm_h4.py:414-440` |
| 🟡 2.2 | All-folds concatenated calibration | ✅ | `04_train_lgbm_h4.py:406-446` |
| 🔴 2.3 | Log pass rate per layer | ✅ | `backtest_utils.py:213-219` |
| 🟡 2.4 | LSTM penalty tiered/absolute | ✅ | `config.py:197-200` |
| 🔴 2.1 | Percentile logging in inference.py | ✅ | `inference.py` (external project) |

---

## Checklist Training Readiness

| Step | Script | Ready? | Notes |
|------|--------|--------|-------|
| 01 — Fetch | `01_fetch.py` | ✅ | Multi-endpoint fallback, `--all` support |
| 02 — Clean | `02_clean.py` | ✅ | Gap detection, leakage audit, multi-TF alignment |
| 03 — Engineer | `03_engineer.py` | ✅ | 85 fitur V3, swing labels, NaN handling |
| 04 — H4 LGBM | `04_train_lgbm_h4.py` | ✅ | Binary classification, isotonic calibration, all-folds |
| 05 — H1 LGBM | `05_train_lgbm_h1.py` | ✅ | 3-class, cost-sensitive, walk-forward CV |
| 06 — LSTM | `06_train_lstm.py` | ✅ | Purged CV, early stopping, GPU support |
| 07 — Evaluate | `07_evaluate.py` | ✅ | Multi-model, SHAP, cascade evaluation |
| 08 — Backtest | `08_backtest.py` | ✅ | Hierarchical predict, pass rate logging, confidence filter |
| Orchestrator | `run_pipeline.py` | ✅ | `--all` = 01→08, `--train` = 04→06 |

---

## Rekomendasi (Deskriptif, Tanpa Kode)

1. **Pisahkan Feature Cols per Model** — Simpan `h1_feature_cols.json` dan `lstm_feature_cols.json` secara terpisah untuk menghindari risiko overwrite. `feature_cols_v2.json` dapat dipertahankan sebagai alias backward-compat atau dihapus jika sudah tidak digunakan.

2. **Dokumentasi sed Workaround** — Tambahkan komentar di notebook bahwa sed pada `config.py` diperlukan untuk non-kline endpoints (funding rate, OI), bukan untuk klines — karena `binance_client.py` sudah punya fallback internal.

3. **Validasi Post-Training** — Setelah semua training selesai, jalankan `07_evaluate.py` dengan `--run-id` yang sesuai untuk memverifikasi bahwa semua model dapat dimuat dan menghasilkan prediksi yang konsisten, sebelum lanjut ke `08_backtest.py`.

---

## Kesimpulan

**✅ PIPELINE LAYAK DIJALANKAN UNTUK TRAINING.**

Semua komponen kritis telah diverifikasi: argument parsing, walk-forward gap integrity, model file dependencies, multi-endpoint fallback, data independence antar training stage, dan seluruh 5 perbaikan dari audit sebelumnya. Tidak ada showstopper.

Tiga temuan minor (feature_cols overwrite, sed redundancy parsial, H4 CLI keterbatasan) perlu dicatat untuk pengembangan ke depan tetapi **tidak menghalangi eksekusi training sekarang**.
