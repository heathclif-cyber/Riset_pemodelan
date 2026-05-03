# Audit Report: H4 Threshold & Soft Filter Penalty Tuning

**Date:** 2026-05-03  
**Session:** Post-Fix 6 (threshold=0.65, soft filter, class weights=1.5)  
**Trigger:** H4 pass_rate = 1.4% (terlalu rendah setelah threshold dinaikkan ke 0.65)

---

## RINGKASAN EKSEKUTIF

H4 binary model menunjukkan **pass_rate = 1.4%** setelah threshold dinaikkan ke 0.65 — artinya H4 hampir tidak pernah aktif sebagai filter. Dua masalah utama: (1) **threshold 0.65 terlalu agresif** untuk distribusi probabilitas H4 yang sempit (P50≈0.506, P90≈0.572) sehingga memotong hampir semua sampel, dan (2) **penalti FLAT dan opposite disamakan** (`-0.02`) padahal FLAT berarti "tidak punya opini" (bukan "salah"), sehingga perlu dibedakan. Tidak ada bug — ini murni masalah tuning dan desain soft filter.

---

## TEMUAN PER KATEGORI

### [KONFIGURASI] Lokasi: `config.py:183-184`
**Deskripsi:** `H4_BINARY_THRESHOLD_LONG = 0.65` dan `H4_BINARY_THRESHOLD_SHORT = 0.65`.
**Dampak:** Dengan distribusi probabilitas H4 P90=0.572, threshold 0.65 berada di ekstrem tail. Hanya ~1.4% sampel yang lolos, membuat H4 praktikalnya tidak pernah digunakan sebagai filter.
**Bukti:**
- P10=0.448, P50=0.506, P90=0.572 (dari live run)
- Pass rate = 1.4% (target ideal 8-15%)
- Threshold ideal berdasarkan distribusi: 0.58–0.62

### [LOGIKA] Lokasi: `pipeline/backtest_utils.py:226-231`
**Deskripsi:** Kondisi `bias == 1` (FLAT) dan `bias != h1_best_dir` (opposite) menggunakan penalty yang sama: `-H4_SOFT_MISALIGN_PENALTY` (-0.02).
**Dampak:** FLAT dihukum sama kerasnya dengan opposite. Ini tidak tepat secara konsep: FLAT berarti H4 tidak punya opini (netral), bukan berarti H4 berlawanan dengan H1. Hukuman yang sama menyamakan "tidak yakin" dengan "salah".
**Bukti:**
```python
elif bias == 1:
    # H4 is FLAT → slight penalty (but NOT hard reject)
    h4_adjustment = -H4_SOFT_MISALIGN_PENALTY  # -0.02
else:
    # H4 opposite to H1 → slightly bigger penalty
    h4_adjustment = -H4_SOFT_MISALIGN_PENALTY  # -0.02 (SAMA!)
```

### [KONSEP] Analisis: H4 AUC realistic ceiling
**Deskripsi:** H4 binary model memprediksi arah swing 24 jam ke depan dengan SL 3.0 ATR dan TP 2.0 ATR pada data crypto yang noisy. Ini inherently sulit — AUC 0.55-0.60 adalah realistic ceiling.
**Dampak:** Tidak perlu terus-menerus tuning H4. Fokus sebaiknya dialihkan ke:
1. Improve H1 (sudah F1 ~0.66)
2. Refine entry threshold
3. Optimasi risk management (DD 72% walau PF 6.3)

---

## JALUR EKSEKUSI YANG TERIDENTIFIKASI

```
hierarchical_predict() → H4 bias → [bias == aligned?] → +0.04 boost
                        ↘ [bias == FLAT?]              → -0.02 (sama dengan opposite!)
                        ↘ [bias == opposite?]          → -0.02
                        → H1 decision layer → LSTM adjustment → final signal
```

Masalah: FLAT (-0.02) = opposite (-0.02) secara logika tidak konsisten.

---

## HIPOTESIS PENYEBAB ROOT (diurutkan dari paling mungkin)

1. **Threshold 0.65 terlalu tinggi** — Distribusi probabilitas H4 sangat sempit (range 0.448-0.572). Threshold 0.65 memotong >98.6% sampel, menyebabkan pass_rate 1.4%. Target pass_rate 8-15% membutuhkan threshold 0.58-0.62.

2. **FLAT penalty tidak dibedakan dari opposite** — FLAT (bias=1) berarti H4 tidak memiliki conviction arah. Ini berbeda secara fundamental dari opposite (bias berlawanan dengan H1). Menyamakan keduanya membuat sistem kehilangan informasi: FLAT seharusnya penalty ringan (-0.01), opposite seharusnya penalty lebih berat (-0.04).

3. **Tidak ada config variable untuk FLAT penalty** — Saat ini `H4_SOFT_MISALIGN_PENALTY` digunakan untuk kedua kondisi. Perlu variabel terpisah: `H4_SOFT_FLAT_PENALTY` dan `H4_SOFT_OPPOSITE_PENALTY`.

---

## PERTANYAAN KLARIFIKASI

Tidak ada — data sudah cukup jelas dari live run metrics.

---

## REKOMENDASI PERBAIKAN

### 1. Threshold: 0.65 → 0.60
**Apa:** Turunkan `H4_BINARY_THRESHOLD_LONG` dan `H4_BINARY_THRESHOLD_SHORT` dari 0.65 ke 0.60 di `config.py`.
**Mengapa:** P90 distribusi probabilitas = 0.572. Threshold 0.60 memungkinkan ~8-15% sampel lolos (target ideal), sedangkan 0.65 terlalu ekstrem (hanya 1.4%). Ini akan mengaktifkan H4 sebagai filter tanpa memotong terlalu banyak.

### 2. Config: Tambah variabel penalty terpisah
**Apa:** Tambah `H4_SOFT_FLAT_PENALTY = 0.01` dan `H4_SOFT_OPPOSITE_PENALTY = 0.04` di `config.py`.
**Mengapa:** FLAT (netral) secara konsep berbeda dari opposite (berlawanan). Dengan variabel terpisah, tuning bisa dilakukan independen.

### 3. Logika: Bedakan FLAT vs opposite di soft filter
**Apa:** Ubah logika di `pipeline/backtest_utils.py:226-231`:
- `bias == 1` (FLAT): `h4_adjustment = -H4_SOFT_FLAT_PENALTY` (-0.01, ringan)
- `bias` opposite: `h4_adjustment = -H4_SOFT_OPPOSITE_PENALTY` (-0.04, lebih keras)
**Mengapa:** FLAT = "no opinion" → konsekuensi minimal. Opposite = "H4 yakin tapi berlawanan dengan H1" → konsekuensi lebih besar. Ini memberikan gradasi yang lebih realistis.

### 4. Komentar: Update threshold comment
**Apa:** Ubah komentar di `config.py:182` — threshold 0.60 dipilih berdasarkan distribusi probabilitas H4 (P90=0.572), bukan berdasarkan asumsi.
**Mengapa:** Dokumentasi yang akurat membantu debugging di masa depan.

---

*End of audit report.*
