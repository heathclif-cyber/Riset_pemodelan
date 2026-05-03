# Audit Report: H4 Model — Missing Higher Timeframe (D1) Features

**Date:** 2026-05-03  
**Topic:** H4 model has no true higher timeframe awareness — D1 data is fetched but never used

---

## RINGKASAN EKSEKUTIF

H4 model saat ini hanya menggunakan fitur dari H1 dan H4 — tidak ada satupun fitur dari D1 (daily) meskipun data D1 sudah difetch dan tersedia di DataFrame. Ini menyebabkan H4 model tidak memiliki konteks higher timeframe: trend makro (D1), struktur swing harian, alignment multi-timeframe, dan regime volatilitas harian semuanya absen. Inilah mengapa AUC model mentok di ~0.55 — model hanya melihat versi resample dari H1, bukan informasi baru dari timeframe yang lebih tinggi.

---

## TEMUAN PER KATEGORI

### [DATA] Lokasi: `pipeline/02_clean.py:173-180`
**Deskripsi:** Data D1 (1d) sudah difetch dan di-join ke H1 master frame dengan prefix `1d_`. Kolom seperti `1d_open`, `1d_high`, `1d_low`, `1d_close`, `1d_volume` tersedia di cleaned parquet.
**Dampak:** Data tersedia tetapi tidak digunakan — sia-sia.
**Bukti:**
```python
# pipeline/02_clean.py:173-180
for tf in ("4h", "1d"):  # D1 ada di sini
    df_tf = klines.get(tf)
```

### [DATA] Lokasi: `core/features.py:1092-1095`
**Deskripsi:** `engineer_features()` mengekstrak H4 OHLCV (`4h_high`, `4h_low`, `4h_close`) tetapi TIDAK mengekstrak D1 OHLCV (`1d_high`, `1d_low`, `1d_close`).
**Dampak:** Seluruh fitur D1 tidak bisa dihitung karena data mentah tidak diambil.
**Bukti:**
```python
# Hanya H4 yang diekstrak:
h4_h = df.get("4h_high",  h)
h4_l = df.get("4h_low",   l)
h4_c = df.get("4h_close", c)
# Tidak ada: d1_h = df.get("1d_high", h)
```

### [FITUR] Lokasi: `config.py:122-164`
**Deskripsi:** `H4_FEATURE_COLS` tidak memiliki satupun fitur D1. Padahal reviewer merekomendasikan: HTF trend slope, volatility regime (ATR percentile), structure break (HH/HL), dan multi-timeframe alignment.
**Dampak:** H4 model hanya punya konteks 4 jam — tidak bisa membedakan swing harian vs intraday noise, tidak tahu trend makro.
**Bukti:**
```python
H4_FEATURE_COLS = [
    # ... semua fitur H1/H4 ...
    # Tidak ada: ema_50_d1, ema_200_d1, atr_d1_percentile, d1_trend, htf_alignment
]
```

### [ARSITEKTUR] Analisis: "H4 adalah H1 versi di-resample"
**Deskripsi:** Semua fitur H4 saat ini berasal dari data yang sama dengan H1 (di-resample ke 4h). Fitur seperti EMA H4, RSI H4, ATR H4 semuanya dihitung dari `4h_close` yang merupakan aggregasi dari `1h_close`. Tidak ada informasi baru yang independen dari H1.
**Dampak:** Model tidak bisa belajar pola yang hanya terlihat di daily (seperti trend mingguan, support/resistance mingguan, seasonal pattern).
**Rekomendasi:** D1 harus menjadi sumber fitur independen — bukan resample dari H1.

---

## JALUR EKSEKUSI

```
02_clean.py → join 1d_* columns ke DataFrame (✅ data ada)
03_engineer.py → engineer_features() → 
  [1092-1095] extract h4_h/l/c (✅)
  [??] extract d1_h/l/c (❌ TIDAK ADA)
  [1171-1186] EMA H4, slopes (✅)
  [??] EMA D1, slopes (❌ TIDAK ADA)
  [1309-1315] ATR H4, range expansion (✅)
  [??] ATR D1, percentile (❌ TIDAK ADA)
  [1261-1274] H4 trend, trend strength (✅)
  [??] D1 trend, HTF alignment (❌ TIDAK ADA)
  [??] D1 HH/HL (❌ TIDAK ADA)
→ 04_train_lgbm_h4.py → H4_FEATURE_COLS (tanpa D1 features)
```

---

## HIPOTESIS PENYEBAB ROOT

1. **D1 data diabaikan di `engineer_features()`** — Data tersedia (`1d_*` columns) tetapi tidak pernah diekstrak atau diolah. Ini adalah penyebab utama.

2. **Feature list tidak pernah diperbarui** — `H4_FEATURE_COLS` tidak pernah diperbarui untuk menyertakan D1 features sejak D1 ditambahkan ke pipeline fetching.

---

## PERTANYAAN KLARIFIKASI

Tidak ada — data sudah tersedia, tinggal dimanfaatkan.

---

## REKOMENDASI PERBAIKAN

### 1. Ekstrak D1 OHLCV di `engineer_features()`
**Apa:** Tambahkan ekstraksi `1d_high`, `1d_low`, `1d_close` setelah ekstraksi H4 (line ~1095).
**Mengapa:** Prasyarat untuk semua fitur D1.

### 2. Hitung D1 EMA & Slope
**Apa:** EMA 50 dan 200 dari D1 close, di-normalisasi dengan ATR H1, di-align ke H1 grid. Hitung slope price vs EMA.
**Mengapa:** Trend makro mingguan — informasi yang tidak bisa didapat dari H4.

### 3. Hitung D1 ATR & Percentile
**Apa:** ATR 14 dari D1. Lalu rank persentil dari ATR D1 terhadap rolling window 100 hari.
**Mengapa:** Volatilitas regime — apakah pasar sedang high/low vol secara harian. Ini memberikan konteks apakah H4 range expansion signifikan atau hanya noise.

### 4. Hitung D1 Trend & HTF Alignment
**Apa:** Trend direction dari D1 (EMA7 vs EMA21). Bandingkan dengan H4 trend — jika align, trend lebih kuat.
**Mengapa:** Multi-timeframe alignment adalah sinyal kuat: jika H4 dan D1 sama-sama bullish, conviction lebih tinggi.

### 5. Deteksi D1 HH/HL Structure
**Apa:** Simple swing high/low detection pada D1 (lookback 5 bar). Hitung bias net HH/HL.
**Mengapa:** Ini memberikan konteks "apakah D1 sedang uptrend atau downtrend" secara struktural — bukan hanya dari EMA.

### 6. Tambahkan ke `H4_FEATURE_COLS`
**Apa:** Tambahkan semua fitur baru ke `H4_FEATURE_COLS` di `config.py`.
**Mengapa:** Agar digunakan dalam training H4 model.

---

*End of audit report — recommendations ready for implementation.*
