# cascade_v2.5_hybrid — Review & Experiment Roadmap

**Tanggal Review**: 31 Mei 2026  
**Tujuan Utama V2.5 Hybrid**:
- Mengembalikan volume & performa seperti era cascade_v2 (yang terbukti lebih baik di live Mei 2026)
- Sambil mempertahankan kekuatan Guardian v3 + Volatility Detectors

---

## 1. Current State Review (Ringkasan)

**Keputusan Resmi (31 Mei 2026):**
> `cascade_v2.5_hybrid` secara resmi menggunakan **87 fitur** (pruned dari 104).

### Apa yang Sudah Ada
- LGBM model + CV results (versi 104 fitur dan versi pruned 87 fitur)
- Guardian model
- Feature importance analysis (sangat skewed ke liquidation distances)
- LSTM training script v2-style (sudah pakai RobustScaler + train metrics)
- Pruned feature list resmi disimpan di:
  - `models/runs/cascade_v2.5_hybrid_pruned/feature_cols_v2.json`
  - `experiments/cascade_v2.5_hybrid_pruned_features.json`

### Temuan Utama Saat Ini

**A. LGBM Overfitting**
- Train-Val Gap rata-rata ~0.19–0.20 pada versi 104 fitur
- Pruning 17 fitur rendah-importance menghasilkan performa hampir identik (bahkan sedikit lebih baik di best fold)

**B. Feature Importance**
- ~22% importance hanya dari 4 fitur liquidation distance
- 17 fitur advanced hampir mati total → berhasil dipruned tanpa kerugian performa signifikan

**C. Struktur**
- 5a/5b/5c sudah dipindah ke archive
- LSTM untuk hybrid ini harus pakai fitur yang sama dengan LGBM (87 fitur)

---

## 2. Proposed Experiments (Prioritas)

### Experiment 1: Feature Pruning (Prioritas Tertinggi)
**Tujuan**: Kurangi noise + kurangi overfitting dengan membuang fitur yang hampir tidak berguna.

**Rencana**:
- Buang fitur dengan importance < 100 (atau < 50)
- Retrain LGBM dengan fitur yang lebih sedikit
- Bandingkan CV performance + train-val gap
- Jika bagus → lanjut ke holdout

**Status**: Belum dimulai

### Experiment 2: Entry Logic Comparison (Inti V2.5)
**Tujuan**: Validasi apakah perubahan threshold + LSTM review benar-benar lebih baik.

**Varian yang akan diuji**:
- V2.5 Proposed: Long 0.69 / Short 0.59 + LSTM review > 0.35
- V2.5 Aggressive: Long 0.65 / Short 0.55 + LSTM review > 0.35
- V2.5 Conservative: Long 0.72 / Short 0.60 + LSTM review > 0.35
- Baseline (tanpa perubahan besar): Long 0.75 / Short 0.60 + LSTM review off atau penalty tinggi

**Metode**: Gunakan model LGBM yang sama + ubah hanya logic di backtest_utils.py atau evaluator, lalu bandingkan holdout.

**Status**: Belum dimulai

### Experiment 3: LSTM v2-style Training + Impact
**Tujuan**: Latih LSTM dengan fitur yang sama dengan LGBM (seperti era v2) dan ukur dampaknya.

**Langkah**:
1. Train LSTM v2-style untuk run `cascade_v2.5_hybrid`
2. Jalankan holdout backtest dengan 3 skenario:
   - Tanpa LSTM review
   - Dengan LSTM review (threshold 0.35)
   - Dengan LSTM review yang lebih agresif

**Status**: Belum dimulai

### Experiment 4: Regularisasi LGBM (jika masih overfit)
**Tujuan**: Kurangi train-val gap di LGBM.

**Ide**:
- Naikkan `min_child_samples`
- Turunkan `max_depth` atau `num_leaves`
- Tambah lebih banyak regularization (lambda, alpha)
- Bandingkan gap dan holdout performance

**Status**: Belum dimulai

---

## 3. Next Immediate Actions

1. [ ] Buat daftar fitur yang akan di-prune berdasarkan importance
2. [ ] Buat script kecil untuk retrain LGBM dengan fitur pruned
3. [ ] Jalankan training LSTM v2-style untuk run ini
4. [ ] Siapkan framework perbandingan holdout (beberapa varian logic entry)

---

## Catatan Penting

- Tujuan utama bukan mengejar F1 CV setinggi mungkin, tapi **performa live + holdout** yang stabil.
- Karena kita sedang revive spirit v2, kita harus berani menguji hal-hal yang "kurang optimal" di CV tapi terbukti lebih baik di real trading.

---

**Update terakhir**: 31 Mei 2026
