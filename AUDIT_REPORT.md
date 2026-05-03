# Audit Report: H4 Model Quality & Cascade Design

## Ringkasan Eksekutif

Setelah calibration collapse diperbaiki, H4 model menunjukkan performa yang masih belum memuaskan: Mean AUC = 0.5546 (sedikit di atas random), fold instability parah (Fold 1=0.6149, Fold 5=0.4692), dan pass rate 27% terlalu tinggi untuk H4 timeframe. Diagnosis: (1) Binary training (drop FLAT) memaksa model mempelajari noise, (2) Class weights 3.0 terlalu agresif, (3) Threshold 0.55 terlalu longgar, (4) Hard gate cascade (H4 veto) tidak cocok untuk model dengan signal quality rendah. Solusi: naikkan threshold ke 0.65, turunkan class weights ke 1.5, dan ubah H4 dari hard gate menjadi soft confidence adjuster.

## Temuan Per Kategori

### [DATA — DISTRIBUSI] `pipeline/04_train_lgbm_h4.py`
**Deskripsi:** H4 binary training membuang ~56% data (FLAT) dan hanya melatih LONG vs SHORT. Ini menyebabkan loss informasi — model tidak pernah belajar mengenali "no trade" regime.
**Dampak:** Model dipaksa mengklasifikasikan regime yang sebenarnya FLAT sebagai LONG atau SHORT, meningkatkan false positive rate.
**Bukti (log runtime):**
```
Label distribution: LONG≈22%, SHORT≈21%, FLAT≈56%
Binary training: SHORT=27k, LONG=27k (FLAT dropped)
```

### [KONFIGURASI — CLASS WEIGHTS TERLALU AGGRESIF] `pipeline/04_train_lgbm_h4.py:266`
**Deskripsi:** `BINARY_WEIGHTS = {0: 3.0, 1: 3.0}` — kedua kelas diberi bobot 3x. Untuk model dengan AUC~0.55, ini menyebabkan overfitting ke sample noise karena sample weight memperkuat sinyal lemah.
**Dampak:** Fold instability (Fold 5=0.4692, worse than random). Model overfit ke noise di beberapa fold.
**Bukti:**
```python
BINARY_WEIGHTS = {0: 3.0, 1: 3.0}  # SHORT=3x, LONG=3x
```

### [KONFIGURASI — THRESHOLD TERLALU RENDAH] `config.py:182-183`
**Deskripsi:** `H4_BINARY_THRESHOLD_LONG = 0.55` dengan distribusi probabilitas P50=0.518 menyebabkan pass rate 27% — terlalu tinggi untuk H4 timeframe yang seharusnya hanya melewati regime kuat.
**Dampak:** Cascade menjadi noisy. H1 entry quality turun karena terlalu banyak bar yang diberi bias.
**Bukti (log runtime):**
```
P50=0.518, threshold=0.55 → pass_rate=27%
Ideal untuk H4: pass_rate 10-15%
```

### [LOGIKA — HARD GATE TIDAK COCOK UNTUK MODEL LEMAH] `pipeline/backtest_utils.py:193-194`
**Deskripsi:** `if bias == 1: continue` — H4 FLAT = hard reject, tidak ada kesempatan untuk H1. Untuk model dengan AUC~0.55, hard decision (ya/tidak) terlalu berisiko.
**Dampak:** Cascade menderita karena H4 yang lemah memveto H1 yang kuat. Sistem = good entry + weak filter.
**Bukti:**
```python
if bias == 1:
    continue  # H4 FLAT → skip (hard reject!)
```

## Jalur Eksekusi

**Current (hard gate):**
```
get_h4_bias() → threshold 0.55 → pass_rate 27%
→ bias=FLAT? → HARD REJECT (H1 tidak pernah diperiksa)
→ bias=LONG/SHORT? → H1 check → LSTM → emit signal
```

**Proposed (soft filter):**
```
get_h4_bias() → threshold 0.65 → pass_rate ~12%
→ bias=FLAT? → h1_conf -= 0.02 (slight penalty, not reject)
→ bias=LONG? → h1_long_conf += 0.04 (slight boost)
→ bias=SHORT? → h1_short_conf += 0.04 (slight boost)
→ H1 check (dengan adjusted conf) → LSTM → emit signal
```

## Hipotesis Penyebab Root

1. **Binary training + class weights 3x (PALING MUNGKIN):** Membuang FLAT (56% data) dan memberikan bobot 3x ke kelas minoritas membuat model over-amplify noise. AUC 0.55 dan fold instability adalah konsekuensi langsung.

2. **Threshold terlalu rendah:** P50=0.518 dengan threshold 0.55 berarti ~50% probabilitas di atas threshold. H4 seharusnya jadi filter ketat, bukan gate yang longgar.

3. **Hard gate cascade tidak sesuai:** Arsitektur cascade yang memberikan H4 kekuatan veto tidak cocok untuk model dengan signal quality rendah. Soft filter (confidence adjustment) lebih robust.

## Rekomendasi Perbaikan

### PRIORITAS 1 — Naikkan H4 Threshold (EKSEKUSI SEKARANG)
**Lokasi:** `config.py:182-183`
**Apa:** Ubah `H4_BINARY_THRESHOLD_LONG` dan `H4_BINARY_THRESHOLD_SHORT` dari 0.55 → 0.65 (atau bahkan 0.70).
**Mengapa:** Menargetkan pass rate 10-15%, hanya mengambil regime yang benar-benar kuat. H4 yang lemah perlu threshold lebih ketat untuk menjaga precision.

### PRIORITAS 2 — Turunkan H4 Class Weights (EKSEKUSI SEKARANG)
**Lokasi:** `pipeline/04_train_lgbm_h4.py:266` dan `config.py`
**Apa:** Ubah `BINARY_WEIGHTS` dari `{0: 3.0, 1: 3.0}` → `{0: 1.5, 1: 1.5}`. Pindahkan ke `config.py` agar mudah di-tuning.
**Mengapa:** Bobot 3x terlalu agresif untuk model dengan AUC~0.55. Bobot 1.5x memberikan regularisasi alami, mengurangi fold instability.

### PRIORITAS 3 — Ubah H4 dari Hard Gate ke Soft Filter (EKSEKUSI SEKARANG)
**Lokasi:** `pipeline/backtest_utils.py:193-222` dan `inference.py:424-438`
**Apa:** Hapus hard reject (`if bias == 1: continue`). Ganti dengan confidence adjustment:
- Jika H4 bias align dengan H1 arah: `h1_conf += H4_SOFT_ALIGN_BOOST` (default +0.04)
- Jika H4 bias FLAT atau opposite: `h1_conf -= H4_SOFT_MISALIGN_PENALTY` (default -0.02)
- H1 tetap menjadi decision layer utama dengan threshold sendiri
**Mengapa:** H4 tidak cukup kuat untuk hard decision. Soft filter memungkinkan H1 tetap emit signal meskipun H4 ragu-ragu, hanya dengan confidence yang disesuaikan. Ini lebih robust.
