# Audit Report: `10_visualize.py` — Stacked Ensemble Left Behind After Refactor

**Date:** 2026-05-03
**Topic:** `FileNotFoundError: models/ensemble_meta.pkl` saat menjalankan `10_visualize.py`

---

## RINGKASAN EKSEKUTIF

`10_visualize.py` crash di baris 68 dengan `FileNotFoundError` karena mencoba memuat `models/ensemble_meta.pkl` — sebuah file yang **tidak lagi diproduksi** oleh pipeline mana pun. File ini adalah artefak dari arsitektur *stacked ensemble* (LogReg meta-learner + Isotonic calibrator) yang telah dihapus dari `08_backtest.py` dan `09_holdout_backtest.py` karena terbukti mendegradasi sinyal (lihat docstring `08_backtest.py:10-11`). Kedua file tersebut kini menggunakan `hierarchical_predict()` dari `backtest_utils.py`. Namun `10_visualize.py` **tidak pernah diupdate** mengikuti perubahan ini, sehingga masih bergantung pada meta-learner dan calibrator yang sudah tidak ada.

---

## TEMUAN PER KATEGORI

```
[KONSISTENSI] Lokasi: pipeline/10_visualize.py:64-77
Deskripsi: Fungsi load_models() mencoba memuat ensemble_meta.pkl dan calibrator.pkl
Dampak: FileNotFoundError karena tidak ada pipeline step yang membuat file tersebut
Bukti:
  meta_learner = joblib.load(MODEL_DIR / "ensemble_meta.pkl")   # line 68 — TIDAK ADA
  calibrator   = ProbabilityCalibrator.load(MODEL_DIR / "calibrator.pkl")  # line 69 — TIDAK ADA
```

```
[KONSISTENSI] Lokasi: pipeline/10_visualize.py:83-137
Deskripsi: Fungsi run_inference() menggunakan meta_learner.predict_proba() dan
           calibrator.transform() — arsitektur ensemble lama
Dampak: Output inference tidak konsisten dengan 08_backtest/09_holdout_backtest
         yang menggunakan hierarchical_predict()
Bukti:
  meta_input = np.hstack([lgbm_proba, lstm_proba])              # line 123
  cal_proba  = calibrator.transform(meta_learner.predict_proba(meta_input))  # line 124
```

```
[KONSISTENSI] Lokasi: pipeline/08_backtest.py:10-11
Deskripsi: Docstring 08_backtest.py menyatakan stacked ensemble telah dihapus
Dampak: Dokumentasi mengatakan ensemble dihapus, tapi 10_visualize.py masih menggunakannya
Bukti:
  "Stacked ensemble (LogReg meta-learner + Isotonic calibrator) telah dihapus
   karena terbukti mendegradasi sinyal (lihat AUDIT_REPORT.md)."
```

```
[KONSISTENSI] Lokasi: pipeline/09_holdout_backtest.py:7
Deskripsi: Docstring 09_holdout_backtest.py juga menyatakan ensemble dihapus
Dampak: Sama seperti di atas
Bukti:
  "Stacked ensemble (LogReg + Isotonic) telah dihapus — lihat AUDIT_REPORT.md."
```

```
[DEPENDENSI] Lokasi: pipeline/10_visualize.py:50
Deskripsi: Masih mengimpor ProbabilityCalibrator yang sudah tidak diperlukan
Dampak: Import tidak terpakai (dead import) jika beralih ke hierarchical_predict()
Bukti:
  from core.models import load_lstm, ProbabilityCalibrator
```

```
[DEPENDENSI] Lokasi: pipeline/10_visualize.py:53
Deskripsi: Masih mengimpor SequenceDataset dari p05_utils
Dampak: Import tidak langsung terpakai di run_inference() — SequenceDataset
         digunakan oleh get_lstm_proba() di backtest_utils.py secara internal
Bukti:
  from pipeline.p05_utils import SequenceDataset
```

---

## JALUR EKSEKUSI YANG TERIDENTIFIKASI

```
10_visualize.py:502 main()
  → 10_visualize.py:478 load_models()
     → 10_visualize.py:68 joblib.load("models/ensemble_meta.pkl")
        ✗ FileNotFoundError — file tidak ada
```

Jalur yang SEHARUSNYA (mengikuti pola 08_backtest.py):
```
10_visualize.py:main()
  → load_models() (refactored)
     → joblib.load("models/lgbm_baseline.pkl")        ✅ H1 LGBM
     → load_lstm("models/lstm_best.pt")               ✅ LSTM
     → joblib.load("models/lstm_scaler.pkl")          ✅ LSTM Scaler
     → json.load("models/feature_cols_v2.json")       ✅ H1 features
     → joblib.load("models/lgbm_h4.pkl") (optional)   ✅ H4 LGBM
     → json.load("models/h4_feature_cols.json")       ✅ H4 features
  → process_symbol()
     → hierarchical_predict() dari backtest_utils.py  ✅ Hierarchical cascade
```

---

## HIPOTESIS PENYEBAB ROOT (diurutkan dari paling mungkin)

1. **Paling mungkin — Refactor遗漏 (refactor omission):**
   Ketika `08_backtest.py` dan `09_holdout_backtest.py` direfactor untuk menggunakan `hierarchical_predict()` (menghapus stacked ensemble), `10_visualize.py` tidak diikutkan dalam perubahan. Ini adalah *oversight* umum ketika beberapa file berbagi arsitektur yang sama tetapi hanya sebagian yang diupdate.

   **Bukti:** Docstring di `08_backtest.py:10-11` dan `09_holdout_backtest.py:7` secara eksplisit menyebutkan ensemble telah dihapus. Tidak ada docstring serupa di `10_visualize.py`.

2. **Kemungkinan — File pipeline/06_ensemble.py dihapus:**
   Berdasarkan komentar di `core/models.py:3` dan `pipeline/p05_utils.py:2`, dulu ada file `pipeline/06_ensemble.py` yang bertugas melatih meta-learner. File ini sudah tidak ada di struktur proyek saat ini, mengonfirmasi bahwa seluruh komponen ensemble telah dihapus.

   **Bukti:** `p05_utils.py:2` menyebut "pipeline/p05_utils.py — SequenceDataset shared antara 05_train_lstm dan 06_ensemble". File `06_ensemble.py` tidak ditemukan.

---

## PERTANYAAN KLARIFIKASI

Tidak ada — penyebab sudah jelas dan dapat diperbaiki tanpa informasi tambahan.

---

## REKOMENDASI PERBAIKAN (deskriptif, bukan kode)

### Perbaikan 1: Refactor `load_models()` — ikuti pola `08_backtest.py`

Ubah `load_models()` untuk:
- Hanya memuat model yang benar-benar ada: `lgbm_baseline.pkl` (H1), `lstm_best.pt`, `lstm_scaler.pkl`, `feature_cols_v2.json`
- Memuat H4 model (`lgbm_h4.pkl` dan `h4_feature_cols.json`) secara opsional (tidak wajib — fallback ke FLAT semua jika tidak ada)
- **Hapus** loading `ensemble_meta.pkl` dan `calibrator.pkl`

### Perbaikan 2: Ganti `run_inference()` dengan `hierarchical_predict()`

Hapus fungsi `run_inference()` yang lama (menggunakan stacked ensemble). Ganti dengan pemanggilan ke `hierarchical_predict()` dari `backtest_utils.py`. Fungsi ini sudah menerapkan:
- H4 bias direction (margin-based binary threshold)
- H1 entry signal (threshold-based)
- LSTM soft proportional adjustment
- H4 soft filter (boost/penalty based on alignment)

### Perbaikan 3: Update signatures dan imports

- `process_symbol()` dan `main()` — sesuaikan parameter signatures: hapus `meta_learner` dan `calibrator`, tambah `h4_model` dan `h4_feat_cols`
- Hapus import `ProbabilityCalibrator` (tidak lagi diperlukan)
- Tambah import `hierarchical_predict` dari `pipeline.backtest_utils`
- Impor `SequenceDataset` masih diperlukan oleh `backtest_utils.get_lstm_proba()` secara internal — tetapi bisa dihapus dari `10_visualize.py` jika tidak digunakan langsung
