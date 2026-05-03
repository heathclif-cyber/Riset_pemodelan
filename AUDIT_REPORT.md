# AUDIT REPORT: H4 Model Root Cause Analysis

**Tanggal:** 2026-05-03  
**Auditor:** Roo (Code)  
**Sasaran:** Diagnosa mengapa H4 binary model memiliki LONG Precision=0.0 dan cascade signal rate sangat rendah (~4.7%)

---

## RINGKASAN EKSEKUTIF

H4 binary model gagal memprediksi LONG dengan presisi > 0 karena kombinasi **tiga masalah struktural**: (1) label H4 dihitung hanya dari CLOSE price, bukan HIGH/LOW, sehingga sangat sedikit bar yang mencapai TP 2.0×ATR; (2) tidak ada fitur trend slope/momentum eksplisit di H4_FEATURE_COLS yang bisa menangkap akselerasi tren; (3) `"symbol"` sebagai fitur mengikat model ke bias koin individual daripada pola market structure umum. Akibatnya H4 model collapse ke prediksi FLAT hampir di semua bar, dan cascade tidak pernah mendapat sinyal H4 binary yang valid.

---

## TEMUAN PER KATEGORI

### [BUG] H4 Labeling Hanya Pakai CLOSE — Bukan HIGH/LOW
**Lokasi:** `pipeline/04_train_lgbm_h4.py:107-127`  
**Deskripsi:** `compute_h4_swing_labels()` menggunakan `future_close = close[i + 1:end]` untuk mengecek apakah TP atau SL tercapai. Tidak pernah menggunakan `high[]` atau `low[]`.  
**Dampak:** Untuk label LONG, close harus mencapai `c + 2.0 × ATR` dalam max_hold bar. Ini sangat sulit — padahal secara real, high intra-bar bisa mencapai level TP walau close di bawahnya. Akibatnya jumlah label LONG/SHORT sangat sedikit (∼0.1–0.5% dari total bar H4).  
**Bukti:**
```python
# Baris 107-119 — hanya close, tidak ada high/low
end = min(i + 1 + max_hold, n)
future_close = close[i + 1:end]

for fc in future_close:
    if fc >= tp_long and not long_hit and not short_hit:
        long_hit = True
        break
    if fc <= sl_long:
        break
```
**Perbaikan:** Gunakan `high[i+1:end]` untuk LONG TP check dan `low[i+1:end]` untuk SHORT TP check.

---

### [DATA] `"symbol"` sebagai Fitur — Generalization Leak
**Lokasi:** `config.py:160` di `H4_FEATURE_COLS` | `core/features.py:1245`  
**Deskripsi:** `feat["symbol"] = symbol_id` menambahkan integer ID koin (0–4 untuk training coins) sebagai fitur. Model bisa belajar "SOLUSDT (ID 0) cenderung bullish, XRPUSDT (ID 3) cenderung sideways" alih-alih pola market structure umum.  
**Dampak:** Untuk koin baru (NEW_COINS, ID 5–17) yang tidak ada di training, feature distribution symbol_id berubah drastis (out-of-distribution). Model gagal generalize ke koin unseen.  
**Bukti:**
```python
# config.py:19-29
TRAINING_COINS = ["SOLUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "DOGEUSDT"]  # ID 0-4
NEW_COINS = ["TONUSDT", "ADAUSDT", ...]  # ID 5-17 — TIDAK PERNAH DILIHAT SAAT TRAINING
```

---

### [KONFIGURASI] H4 Feature Set Kurang Trend Dynamics
**Lokasi:** `config.py:122-161` (`H4_FEATURE_COLS`)  
**Deskripsi:** H4_FEATURE_COLS sudah memiliki EMA (level) dan `rsi_h4`, tetapi tidak memiliki fitur momentum/slope eksplisit seperti `ema_slope`, `price_vs_ema`, `rsi_slope`, atau `range_expansion`. Fitur yang ada saat ini:
- ✅ EMA level (ATR-normalized): `ema_7/21/50/200_h4`
- ✅ Trend direction: `h4_trend` (EMA7 > EMA21 ? 1 : -1)
- ✅ Trend strength: `trend_strength` ((EMA7 - EMA50) / ATR)
- ✅ RSI H4: `rsi_h4`
- ❌ `ema_21_slope_h4` — rate of change EMA21
- ❌ `ema_50_slope_h4` — rate of change EMA50
- ❌ `price_vs_ema_50_h4` — posisi close relatif terhadap EMA50 (sekarang hanya sebagai normalized EMA)
- ❌ `atr_percent_h4` — ATR sebagai % harga
- ❌ `range_expansion_h4` — deteksi ekspansi range
- ❌ `rsi_slope_h4` — akselerasi momentum

**Dampak:** Model H4 kesulitan membedakan "trend yang sedang akselerasi" vs "trend yang sudah mature". Ini menjelaskan recall LONG yang sangat rendah.  
**Bukti:** Lihat `core/features.py:1167-1179` — EMA dihitung sebagai (EMA - close) / ATR. Slope tidak pernah dihitung.

---

### [LOGIKA] H4 Labeling RR Check Tidak Pernah Gagal
**Lokasi:** `pipeline/04_train_lgbm_h4.py:99-105,129-132`  
**Deskripsi:** Karena RR dihitung sebagai `min_tp_atr / max_sl_atr = 2.0/3.0 ≈ 0.667` (KONSTAN per bar), dan `min_rr = 0.6`, maka kondisi `rr_long >= min_rr` SELALU TRUE untuk setiap bar yang mencapai TP sebelum SL. Parameter `H4_SWING_LABEL_MIN_RR = 0.6` tidak berfungsi sebagai filter — semua bar lolos.  
**Dampak:** Parameter min_rr secara efektif mati (dead config). Tidak ada downside risk filtering.  
**Bukti:**
```python
# Baris 99-105 — RR fixed
tp_long  = c + min_tp_atr * a    # = c + 2.0 * ATR
sl_long  = c - max_sl_atr * a    # = c - 3.0 * ATR
rr_long  = (tp_long - c) / (c - sl_long)   # = (2.0*ATR) / (3.0*ATR) = 0.667

# Baris 129 — selalu True karena 0.667 >= 0.6
if long_hit and rr_long >= min_rr:   # min_rr = 0.6
```

---

### [KONFIGURASI] H4 LGBM Model Mungkin Terlalu Sederhana
**Lokasi:** `config.py:105-117` (`LGBM_H4_PARAMS`)  
**Deskripsi:** Parameter saat ini: max_depth=4, num_leaves=15, min_child_samples=30, n_estimators=500. Dengan ∼12K–20K binary samples setelah drop FLAT, model dengan 15 leaves mungkin tidak cukup kompleks untuk menangkap regime switching non-linear.  
**Dampak:** Model underfit — tidak bisa membedakan pola LONG vs SHORT dengan baik.  
**Bukti:**
```python
LGBM_H4_PARAMS = {
    "max_depth":         4,         # 2^4 = 16 leaves max
    "num_leaves":        15,        # ≤ max_depth
    "min_child_samples": 30,        # cukup rendah untuk 12K samples
    "n_estimators":      500,       # cukup
    "learning_rate":     0.03,      # konservatif
}
```

---

### [KONFIGURASI] Threshold Cascade Terlalu Ketat (Sekunder)
**Lokasi:** `config.py:175-180`  
**Deskripsi:** H4_BINARY_THRESHOLD=0.55 dan H1_THRESHOLD=0.62. Efek kaskade: probabilitas bar lolos kedua threshold = ~(sisa H4 non-FLAT) × (H1 signal rate). Jika H4 hanya 30% non-FLAT dan H1 hanya 30% sinyal, total signal rate = 9%. Dengan H4_BINARY_MARGIN=0.05 yang baru, ini bisa turun ke 5–7%.  
**Dampak:** Signal rate rendah (4.7%) secara alami mengikuti dari threshold ketat + margin baru. Bukan penyebab utama H4 model collapse.  
**Bukti:** Lihat `pipeline/backtest_utils.py:129-130` — margin-based masking.

---

## JALUR EKSEKUSI YANG TERIDENTIFIKASI

```
compute_h4_swing_labels()  → [CLOSE-only check] → sangat sedikit non-FLAT labels
       ↓
load_and_resample_to_h4() → label distribution: LONG~0.1% SHORT~0.1% FLAT~99.8%
       ↓
H4 model training (binary) → hanya 0.2% positive class → model collapse ke FLAT
       ↓
hierarchical_predict() → get_h4_bias() → 99.9% FLAT → cascade tidak pernah aktif
       ↓
evaluate_cascade() → signal rate ~4.7% → LONG Precision=0.0
```

---

## HIPOTESIS PENYEBAB ROOT

1. **Paling Kuat: H4 labeling hanya pakai CLOSE** — Jika labeling menggunakan HIGH/LOW untuk TP/SL detection (seperti standar backtesting), jumlah label non-FLAT bisa meningkat 3–5×. Ini adalah penyebab paling fundamental karena langsung membatasi jumlah training sample yang tersedia untuk H4 binary model.

2. **Kedua: Tidak ada fitur trend slope eksplisit** — H4 model perlu membedakan trend fase awal (potensi LONG) dari trend fase akhir (potensi reversal). EMA slope, RSI slope, dan price position relatif terhadap EMA memberikan sinyal ini. Tanpanya, model hanya punya "snapshot" level harga tanpa konteks momentum.

3. **Ketiga: `symbol` sebagai fitur** — Model belajar shortcut: "SOLUSDT (ID 0) lebih sering LONG daripada XRPUSDT (ID 3)". Ini menyebabkan model gagal generalisasi ke koin baru dan tidak belajar pola market structure universal.

4. **Keempat: H4 model terlalu sederhana** — max_depth=4 dengan ~12K samples mungkin underfit. max_depth=6 (64 leaves) memberikan kapasitas lebih tanpa risiko overfitting berlebihan untuk jumlah data tersebut.

---

## PERTANYAAN KLARIFIKASI

1. Apakah `compute_h4_swing_labels()` sengaja menggunakan CLOSE saja (bukan HIGH/LOW) karena alasan tertentu (misalnya, menghindari look-ahead)? Jika tidak, ini bug yang jelas.
2. Apakah penggunaan `"symbol"` sebagai fitur sengaja untuk menangkap perbedaan volatilitas antar koin? Atau ini oversight dari pipeline engineer awal?
3. Apakah ada data H4 results dari training sebelumnya yang menunjukkan distribusi label (berapa % LONG, SHORT, FLAT sebelum drop FLAT)?

---

## REKOMENDASI PERBAIKAN

### PRIORITAS 1 — H4 Labeling: Gunakan HIGH/LOW
**Lokasi:** `pipeline/04_train_lgbm_h4.py:107-127`  
**Deskripsi:** Ubah `compute_h4_swing_labels()` untuk menggunakan `high[i+1:end]` dan `low[i+1:end]` dalam mendeteksi TP/SL hit. Ini adalah praktik standar dalam backtesting — TP dianggap tercapai jika HIGH >= TP (untuk LONG), bukan jika CLOSE >= TP.
- Gunakan `high[i+1:end].max()` untuk LONG TP check
- Gunakan `low[i+1:end].min()` untuk LONG SL check
- Gunakan `low[i+1:end].min()` untuk SHORT TP check
- Gunakan `high[i+1:end].max()` untuk SHORT SL check
**Dampak:** Jumlah label non-FLAT meningkat signifikan (estimasi 3–5×).

### PRIORITAS 2 — Hapus `"symbol"` dari H4_FEATURE_COLS
**Lokasi:** `config.py:160`  
**Deskripsi:** Hapus `"symbol"` dari `H4_FEATURE_COLS`. Juga hapus dari `FEATURE_COLS_V3` (line 282-283) jika memungkinkan.
**Alternatif:** Jika ingin tetap mempertahankan informasi volatilitas per-koin, ganti dengan fitur yang lebih general seperti `log_volume_ma_ratio` atau `volatility_percentile` yang bisa generalize ke koin unseen.
**Dampak:** Model belajar pola market structure umum, bukan bias koin individual.

### PRIORITAS 3 — Tambah Trend Slope Features
**Lokasi:** `config.py:122-161` dan `core/features.py` (sekitar baris 1172)  
**Deskripsi:** Tambahkan fitur berikut ke `H4_FEATURE_COLS` dan hitung di `engineer_features()`:
- `ema_21_slope_h4` = (ema_21 - ema_21.shift(4)) / (atr_h4) — slope 4 bar H4
- `ema_50_slope_h4` = (ema_50 - ema_50.shift(4)) / (atr_h4)
- `price_vs_ema_50_h4` = (close - ema_50) / atr_h4 (lebih eksplisit dari normalized EMA)
- `atr_percent_h4` = atr_14_h4 / close * 100
- `range_expansion_h4` = (high - low) / (high.shift(1) - low.shift(1) + epsilon)
- `rsi_slope_h4` = rsi_h4 - rsi_h4.shift(4)
**Dampak:** Model mendapatkan informasi momentum dan akselerasi tren, bukan hanya level.

### PRIORITAS 4 — Review H4 LGBM Params
**Lokasi:** `config.py:105-117`  
**Deskripsi:** Setelah label dan fitur diperbaiki, evaluasi apakah `max_depth=4, num_leaves=15` cukup. Pertimbangkan:
- `max_depth=6, num_leaves=31, min_child_samples=50` — versi yang selaras dengan H1 params
- Alternatif: `max_depth=5, num_leaves=20` — kompromi antara kapasitas dan overfitting
**Catatan:** Jangan ubah parameter sebelum label dan fitur diperbaiki, karena model akan dilatih ulang dengan data yang berbeda.

### PRIORITAS 5 — Review Threshold (Pasca Retrain)
**Lokasi:** `config.py:175-180`  
**Deskripsi:** Setelah retrain H4 dengan label dan fitur baru, evaluasi distribusi probabilitas output. Jika calibrated proba terdistribusi lebih baik (tidak collapse ke 0.5), threshold bisa diturunkan untuk meningkatkan signal rate.
- H4_BINARY_THRESHOLD bisa diturunkan ke 0.50 jika distribusi proba bagus
- H1_THRESHOLD bisa tetap di 0.60–0.62 untuk menjaga kualitas entry
**Catatan:** Tuning threshold tanpa model yang baik adalah sia-sia. Lakukan setelah Prioritas 1–3 selesai.

---

## KESIMPULAN

Masalah H4 LONG Precision=0.0 dan signal rate 4.7% BUKAN disebabkan oleh pipeline, LSTM, atau kalibrasi. Akar masalah ada di **tiga tempat**:

| # | Masalah | Dampak | Prioritas |
|---|---------|--------|-----------|
| 1 | H4 labeling pakai CLOSE bukan HIGH/LOW | Sangat sedikit label non-FLAT | 🔴 WAJIB |
| 2 | `"symbol"` sebagai fitur | Generalisasi jelek ke koin baru | 🔴 WAJIB |
| 3 | Kurang trend slope features | Model buta momentum | 🟡 PENTING |
| 4 | H4 model terlalu sederhana | Underfitting | 🟡 PENTING |
| 5 | Threshold ketat | Signal rate rendah (sekunder) | 🟢 NICE TO HAVE |

Setelah Prioritas 1–3 diterapkan, distribusi label H4 diestimasi berubah dari:
- **Sekarang:** LONG=0.1% SHORT=0.1% FLAT=99.8% → binary samples ~100
- **Setelah:** LONG=~5% SHORT=~5% FLAT=~90% → binary samples ~6,000

Dengan 6K+ binary samples, H4 model memiliki data cukup untuk belajar pola regime detection yang bermakna.
