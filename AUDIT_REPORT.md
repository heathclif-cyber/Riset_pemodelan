# Audit Report: Hasil Evaluasi Cascade — Validitas Metrik

**Tanggal:** 2026-05-03  
**Auditor:** AI Code Analysis  
**Mode:** Read-only audit (no code changes)

---

## Ringkasan Eksekutif

Pipeline `07_evaluate.py` berhasil menjalankan H1, H4, dan Cascade evaluation tanpa error. Namun metrik yang dihasilkan menunjukkan beberapa anomali serius:

1. **H4 LGBM LONG Precision = 0.0** — Model regime filter tidak pernah memprediksi LONG dengan benar.
2. **Winrate 81.45% dengan Sharpe 13.24** — Tidak realistis untuk sistem trading kripto; indikasi bias look-ahead atau inflasi metrik.
3. **Swing detection mengandung look-ahead** — `detect_h4_swing_points()` menggunakan data masa depan untuk konfirmasi swing.
4. **Sharpe ratio terinflasi** — Annualisasi menggunakan `sqrt(trades_per_year)` bukan `sqrt(periods_per_year)` standar.

Ini **bukan bug crash** seperti sebelumnya, tapi **masalah validitas metrik** yang membuat hasil evaluasi tidak bisa dipercaya untuk mengambil keputusan deployment.

---

## Temuan

### [BUG] H4 LGBM LONG Precision = 0.0 — Regime Filter Rusak

| Field | Detail |
|-------|--------|
| **Lokasi** | [`pipeline/07_evaluate.py:301`](pipeline/07_evaluate.py:301) — `evaluate_h4_lgbm()` output |
| **Bukti** | `LONG: Precision=0.0000, Recall=0.0000` |
| **Dampak** | H4 tidak pernah benar saat predict LONG. Namun cascade tetap menghasilkan 1.254 sinyal LONG — artinya sinyal H1/LSTM override H4. |

**Mekanisme override** di [`pipeline/backtest_utils.py:180-214`](pipeline/backtest_utils.py:180-214):
```python
if bias == 2:
    h1_conf = h1_long_conf
    ...
if h1_conf < h1_thr:
    continue  # H1 threshold not met
...
y_pred[i] = bias  # Final signal = H4 bias direction
```

Sinyal final SELALU mengikuti arah H4 bias (`y_pred[i] = bias`). Jadi 1.254 sinyal LONG berasal dari H4 predict LONG, tapi H4 evaluation menunjukkan LONG Precision=0. Artinya **semua 1.254 sinyal LONG punya regime bias yang salah**, tapi tetap menghasilkan winrate 81.45% karena TP/SL berbasis swing bisa menghasilkan profit meski arah biasnya salah.

### [BUG] Look-ahead Bias di Swing Detection

| Field | Detail |
|-------|--------|
| **Lokasi** | [`core/features.py:417`](core/features.py:417) — `detect_h4_swing_points()` |
| **Baris** | 434: `for i in range(lookback, n - lookback):` |
| **Lookback** | 3 bar H4 = 12 jam data masa depan |

```python
for i in range(lookback, n - lookback):
    window_h = h4_high.iloc[i - lookback: i + lookback + 1]
    #                                        ^^^^^^^^^^^^
    #                   3 bar KE KANAN = FUTURE DATA ⇢ LOOK-AHEAD
    if h4_high.iloc[i] == window_h.max():
        sh.iloc[i] = h4_high.iloc[i]
```

**Dampak:** Di [`pipeline/07_evaluate.py:362-363`](pipeline/07_evaluate.py:362-363), `h4_swing_high` dan `h4_swing_low` digunakan langsung di `simulate_trades_swing()` untuk TP/SL. Swing level di bar `i` dikonfirmasi menggunakan data sampai bar `i+3` — yang belum tersedia di skenario live. Ini membuat trade simulation seolah-olah punya "pengetahuan masa depan" tentang level support/resistance.

Contoh: Harga tembus swing high di bar `i+2` → bar `i` ditandai sebagai swing high → di bar `i` trader bisa pasang TP di level yang baru akan terkonfirmasi 2 bar kemudian.

### [KONFIGURASI] Inflasi Sharpe Ratio

| Field | Detail |
|-------|--------|
| **Lokasi** | [`core/evaluator.py:467`](core/evaluator.py:467) |
| **Baris** | `ann_factor = np.sqrt(trades_per_year)` |

**Rumus saat ini:**
```python
ann_factor = np.sqrt(trades_per_year)  # dengan ~349 trade/tahun → sqrt(349) ≈ 18.7x
```

**Standar industri untuk per-trade return:**
```python
ann_factor = np.sqrt(365 * 24 / avg_holding_hours)  # atau np.sqrt(periods_per_year)
```

Dengan 1.326 trades dalam ~3.8 tahun ≈ 349 trades/tahun, faktor annualisasi 18.7x mengubah Sharpe harian yang wajar (misal 0.7) menjadi 13.1. Ini menciptakan ilusi performa yang jauh lebih baik dari kenyataan.

### [DATA] Ketimpangan Sinyal LONG vs SHORT

**Dari output cascade:**
- LONG = 1.254 (82.2% dari sinyal non-FLAT)
- SHORT = 272 (17.8% dari sinyal non-FLAT)
- Rasio LONG/SHORT = 4.6:1

Ini menunjukkan model sangat bias ke arah LONG. Di pasar kripto bull market (data training didominasi tren naik), model belajar bahwa LONG hampir selalu benar. Akibatnya:

| Dampak | Detail |
|--------|--------|
| Di training | Model bagus karena sesuai tren pasar |
| Di live (sideways/bear) | Model akan rugi besar karena bias LONG |
| Evaluasi | Metrik tidak mencerminkan performa di semua regime pasar |

### [DATA] H4 Regime Accuracy Non-FLAT = 16.59%

Hanya 16.59% — barely di atas random (50% untuk binary). Ini berarti H4 model HAMPIR TIDAK BERGUNA sebagai regime filter. Tanpa H4 filter, trade tetap akan terjadi (H1 + LSTM), dan kemungkinan metriknya mirip.

---

## Jalur Eksekusi yang Teridentifikasi

```
evaluate_cascade()
  → load_data(SOLUSDT)                    # 33.397 bars H1
  → hierarchical_predict()                # STEP 1-4
      → get_h4_bias()                     # H4 predict: LONG/SHORT/FLAT
      → h1_model.predict_proba()          # H1 entry probability
      → get_lstm_proba()                  # LSTM soft adjustment
      → Decision layer                     # Final signal
  → full_trading_report()
      → simulate_trades_swing()           # Simulasi dengan swing TP/SL
          → detect_h4_swing_points()      # ← LOOK-AHEAD: pakai data masa depan
  → update_model_metrics("hierarchical_v1")
```

**Titik kegagalan validitas:**
1. `detect_h4_swing_points()` — look-ahead 3 bar H4 (12 jam)
2. `h4_swing_highs[i]` digunakan di bar `i` — padahal baru terkonfirmasi di bar `i+3`
3. Sharpe dihitung dengan annualisasi per-trade — tidak standar

---

## Hipotesis Penyebab Root (diurutkan dari paling mungkin)

1. **🟥 Look-ahead di swing detection adalah penyebab utama metrik tidak realistis.**  
   Bukti: `detect_h4_swing_points` menggunakan `i+lookback` (3 bar ke depan). Di simulation, swing level ini dipakai untuk TP/SL sebelum benar-benar terkonfirmasi. Ini memberi keuntungan tidak adil yang menjelaskan 81.45% winrate.

2. **🟥 H4 LGBM adalah random classifier (regime accuracy 16.59%).**  
   Bukti: LONG Precision=0.0, non-FLAT accuracy ≈ random. Menambah H4 filter tidak meningkatkan kualitas keputusan.

3. **🟡 Sharpe 13.24 adalah artefak matematis, bukan performa nyata.**  
   Bukti: Annualisasi `sqrt(349) ≈ 18.7x` menginflasi Sharpe dari ~0.7 menjadi 13.24.

4. **🟡 Bias LONG 4.6:1 adalah sinyal overfitting ke bull market.**  
   Bukti: 82.2% sinyal adalah LONG, model tidak siap untuk sideways/bear market.

---

## Rekomendasi Perbaikan (deskriptif, tanpa kode)

### 1. Perbaiki Look-ahead di Swing Detection
**Akar masalah:** `detect_h4_swing_points()` menggunakan `i ± lookback` untuk konfirmasi swing.

**Opsi yang bisa dilakukan:**
- **Opsi A (direkomendasikan):** Ubah `detect_h4_swing_points` menjadi hanya menggunakan data historis (`iloc[i-lookback:i+1]` saja, tanpa `i+lookback+1`). Konsekuensi: swing akan terdeteksi 3 bar lebih lambat, mengurangi jumlah trade tapi menghilangkan look-ahead bias.
- **Opsi B:** Simpan dua versi swing levels: `h4_swing_high_lookback` (untuk fitur) dan `h4_swing_high_nolookahead` (untuk evaluasi/backtest).
- **Opsi C:** Pindahkan logika konfirmasi swing ke dalam loop simulation (di `simulate_trades_swing`), di mana swing level di bar `i` hanya menggunakan data sampai bar `i`.

### 2. Perbaiki H4 Binary Training
**Akar masalah:** Model mendapat label yang benar (15% LONG) tapi tidak bisa belajar memprediksi LONG.

**Opsi:**
- Evaluasi ulang fitur H4 (`h4_feature_cols`) — apakah ada fitur yang informatif.
- Cek apakah ada data leakage antara fitur dan label H4.
- Pertimbangkan untuk merge H4 dan H1 training (multi-resolution) daripada pipeline terpisah.

### 3. Standarisasi Sharpe Ratio
**Akar masalah:** `sqrt(trades_per_year)` tidak standar.

**Opsi:**
- Ganti ke annualisasi harian: `np.sqrt(365)` untuk Sharpe harian standar.
- Atau pastikan return dihitung per periode waktu tetap (bukan per-trade) sebelum annualisasi.

### 4. Cross-Validation Multi-Symbol untuk Cascade
Saat ini cascade hanya dievaluasi di SOLUSDT. Evaluasi di semua 18 koin untuk memvalidasi generalisasi.

---

## Pertanyaan Klarifikasi

1. Apakah metrik cascade (81.45% winrate) perlu dipercaya untuk deployment? **Tidak disarankan** — terlalu tinggi dan ada indikasi look-ahead.
2. Apakah perlu memperbaiki look-ahead dulu sebelum lanjut ke fase 08 (backtest)? **Sangat disarankan** — backtest akan menderita bias yang sama.
3. Apakah label H4 perlu diperiksa ulang (mungkin labeling function perlu disesuaikan agar model bisa belajar)?
