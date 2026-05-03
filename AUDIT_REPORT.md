# Audit Report: H4 Isotonic Calibration Collapse

## Ringkasan Eksekutif

Kalibrasi isotonic untuk H4 binary model mengalami **complete collapse**: setelah kalibrasi, 100% probabilitas menjadi 1.0 (LONG). Ini membuat H4 filter selalu memberikan izin, sehingga cascade berubah menjadi `H4 ≈ always ON → system ≈ H1 only`. Penyebab utama: (1) concatenated all-fold calibration (data leakage), (2) AUC model ~0.55 (near-random) dengan distribusi prediksi sempit (P10=0.437, P90=0.593) yang membuat IsotonicRegression over-extrapolate, (3) penggunaan IsotonicRegression yang agresif untuk distribusi probabilitas sempit.

## Temuan Per Kategori

### [BUG — DATA LEAKAGE] `pipeline/04_train_lgbm_h4.py:410-451`
**Deskripsi:** Semua fold validation probabilities digabung (`np.concatenate(all_val_proba)`), lalu satu `IsotonicRegression` tunggal di-fit pada data gabungan tersebut. Ini adalah data leakage — kalibrator melihat data validasi dari semua fold sekaligus, bukan out-of-fold predictions per fold.
**Dampak:** Informasi dari fold masa depan bocor ke kalibrator. IsotonicRegression menganggap distribusi 49k samples sebagai satu set, bukan 8 independent folds.
**Bukti:**
```python
val_proba_all  = np.concatenate(all_val_proba)   # 48,928 samples
val_labels_all = np.concatenate(all_val_labels)
calibrator = ProbabilityCalibrator()
calibrator.fit(val_proba_all.reshape(-1, 1), val_labels_all)
```

### [BUG — ISOTONIC COLLAPSE] `core/models.py:99-130`
**Deskripsi:** `ProbabilityCalibrator` menggunakan `IsotonicRegression(out_of_bounds="clip")` default. Untuk distribusi probabilitas sempit (P10=0.437, P50=0.518, P90=0.593), isotonic memetakan input range ~0.15 ke output [0, 1]. Karena sebagian besar sample > 0.5, isotonic "memaksa" transformasi ke ekstrem 1.0.
**Dampak:** Semua probabilitas setelah kalibrasi = 1.0. Threshold 0.55 tidak berguna — 99.9% sample lolos.
**Bukti (log runtime):**
```
P50 0.518 → 1.000 (+48.2pp)
pass_rate 27.0% → 99.9%
```

### [KONFIGURASI] `pipeline/backtest_utils.py:121`
**Deskripsi:** Backtest (`get_h4_bias()`) tidak menggunakan calibrator sama sekali — langsung `h4_model.predict_proba()`. Ini berarti backtest results selama ini sebenarnya valid (tidak terpengaruh calibration collapse), tapi production inference di `inference.py` menggunakan calibrator.
**Dampak:** Backtest vs production mismatch. Backtest menunjukkan performa realistis, production collapse karena calibrator.
**Bukti:**
```python
# backtest_utils.py:121 — NO calibrator usage
h4_proba = h4_model.predict_proba(df_slice[valid_h4_cols])
```
```python
# inference.py:392 — WITH calibrator
if bundle.h4_calibrator is not None:
    h4_p_cal = bundle.h4_calibrator.transform(h4_p.reshape(1, -1))[0]
```

## Jalur Eksekusi

**Training:**
```
04_train_lgbm_h4.py:walk_forward_cv_h4() → concatenate all fold val_proba
→ ProbabilityCalibrator.fit(48k samples) → IsotonicRegression
→ save h4_calibrator.pkl  ← [BUG: data leakage + isotonic collapse]
```

**Backtest (tidak terpengaruh):**
```
backtest_utils.py:get_h4_bias()
→ h4_model.predict_proba() langsung → raw proba → threshold check
→ bias_dir = LONG/SHORT/FLAT
```

**Production (terpengaruh):**
```
inference.py:_hierarchical_proba()
→ h4_model.predict_proba() → raw proba
→ bundle.h4_calibrator.transform() → [COLLAPSE: semua jadi 1.0]
→ threshold check → selalu lolos → bias = LONG
```

## Hipotesis Penyebab Root

1. **IsotonicRegression + AUC rendah + distribusi sempit (PALING MUNGKIN):** Model H4 binary memiliki AUC ~0.55, nyaris random. Distribusi probabilitas output sangat sempit (range ~0.15). IsotonicRegression dirancang untuk distribusi probabilitas yang terkalibrasi baik; pada distribusi sempit ia "memaksa" mapping ke [0,1] sehingga terjadi ekstrapolasi agresif → semua > 0.5 menjadi 1.0.

2. **Concatenated folds (data leakage):** Meskipun masalah utama adalah isotonic collapse, concatenating all folds tetap salah secara metodologi. Kalibrator harus di-fit pada out-of-fold predictions per fold.

3. **Tidak perlu kalibrasi untuk binary threshold:** Untuk model binary dengan AUC rendah, kalibrasi isotonic tidak memberikan manfaat. Threshold sederhana pada raw probability lebih stabil dan interpretable. Sigmoid calibration (Platt scaling) lebih robust untuk distribusi sempit.

## Rekomendasi Perbaikan

### PRIORITAS 1 — Matikan H4 Calibration (EKSEKUSI SEKARANG)
**Lokasi:** `pipeline/04_train_lgbm_h4.py:410-453`
**Apa:** Tambahkan flag `H4_USE_CALIBRATION = False` di `config.py`. Di `04_train_lgbm_h4.py`, skip seluruh blok fitting & saving calibrator ketika flag False. Pertahankan logging percentile BEFORE calibration untuk monitoring.
**Mengapa:** Calibrator saat ini merusak inference. Raw probability langsung ke threshold sudah cukup untuk model binary dengan AUC rendah. Backtest sudah membuktikan ini (tidak pakai calibrator).

### PRIORITAS 2 — Tuning Threshold Langsung
**Lokasi:** `config.py:172-174`
**Apa:** Turunkan `H4_BINARY_THRESHOLD_LONG` dan `H4_BINARY_THRESHOLD_SHORT` dari 0.55 ke 0.52-0.53 (atau cari optimal via grid search pada validation set).
**Mengapa:** Tanpa calibrator, threshold di raw probability perlu disesuaikan. P50=0.518 berarti threshold 0.55 terlalu ketat untuk model saat ini.
**Rekomendasi:** Jalankan grid search threshold pada validation OOF predictions: coba threshold 0.48 s.d. 0.60 step 0.01, pilih yang memberikan profit factor + Sharpe ratio optimal.

### PRIORITAS 3 — Opsional: Ganti ke Sigmoid Jika Ingin Kalibrasi Ulang
**Lokasi:** `config.py` dan `pipeline/04_train_lgbm_h4.py`
**Apa:** Jika ingin kalibrasi di masa depan, ubah method dari `"isotonic"` ke `"sigmoid"` (Platt scaling / LogisticRegression). Sigmoid lebih stabil karena hanya mempelajari parameter logistik (skala + intercept), bukan non-parametric step function.
**Mengapa:** Logistic regression untuk binary calibration hanya punya 2 parameter (slope + bias), jauh lebih robust terhadap AUC rendah dan distribusi sempit dibanding IsotonicRegression yang bisa memiliki puluhan steps.
