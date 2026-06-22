# 📊 Balanced Benchmark Report: `simons_hybrid_v1` vs `ic32_regime_v1`
*Tanggal: 2026-06-08 | Periode Holdout: 2025-11-01 s/d 2026-04-01 (Out-of-Sample)*

Laporan ini membandingkan hasil backtest model **simons_hybrid_v1** (dengan 4-stage feature selection + training di filtered HMM regimes) melawan model baseline **ic32_regime_v1** (model global dengan 33 fitur). Kedua pengujian dijalankan dengan menggunakan setup kode yang sama, parameter TP/SL yang sama, dan penyelarasan fitur LSTM yang sudah diperbaiki secara dinamis.

## 📈 Perbandingan Metrik Portofolio

| Metrik | `ic32_regime_v1` (Baseline) | `simons_hybrid_v1` (Eksperimen) | Perubahan |
| :--- | :---: | :---: | :---: |
| **Total Net PnL (5x Leverage)** | **+$699.87 USD** | **+$369.27 USD** | -$330.60 USD |
| **Portfolio ROI (%)** | **+33.33%** | **+17.58%** | -15.75% |
| **Mean Win Rate (per koin)** | **63.38%** | **56.45%** | -6.93% |
| **Mean Sharpe Ratio** | **5.19** | **2.58** | -2.61 |
| **Mean Profit Factor** | **2.28** | **1.64** | -0.64 |
| **Mean Max DD (per koin, 5x)** | **56.59%** | **96.14%** | +39.55% |
| **Mean Trade/Bulan (per koin)** | **23.3** | **21.4** | -1.9 |
| **Total Trades** | **2,415** | **2,216** | -199 |
| **Worst Single-Trade Loss** | **-24.90%** | **-24.50%** | +0.40% |
| **Max Consecutive Loss** | **11** | **12** | +1 |

---

## 🔍 Temuan Utama

1. **Perbaikan Besar dari Perbaikan Bug**:
   Setelah memperbaiki bug regime routing (menghubungkan kolom `hmm_regime`) dan penyelarasan fitur LSTM yang tidak selaras di pengujian holdout, performa **simons_hybrid_v1** meningkat secara masif dari uji coba awal:
   * **Net PnL** meningkat dari **+$136.59 USD** menjadi **+$369.27 USD**.
   * **Sharpe Ratio** naik dari **1.41** menjadi **2.58**.
   * **Jumlah Trade** meningkat dari **1,405** menjadi **2,216** karena model tidak lagi buta terhadap regime-specific entry trigger.

2. **Dominasi Baseline `ic32_regime_v1`**:
   Meskipun model hybrid regime-specific mengalami peningkatan performa setelah bug diperbaiki, model global baseline **ic32_regime_v1** tetap mendominasi metrik performa utama:
   * **Net PnL hampir 2x lebih besar** (+$699.87 USD vs +$369.27 USD).
   * **Sharpe Ratio** (5.19 vs 2.58) dan **Win Rate** (63.38% vs 56.45%) jauh lebih tinggi.
   * **Max Drawdown** (56.59% vs 96.14%) jauh lebih aman dan terkontrol.

3. **Mengapa Model Global Lebih Baik?**:
   * **Feature Information Loss**: Multi-stage feature selection pada `simons_hybrid_v1` memotong jumlah fitur secara agresif (hanya ~15 fitur per model), yang menyebabkan hilangnya interaksi antar-fitur penting.
   * **Regime Boundary Noise**: Memisahkan model secara kaku pada state HMM 0-3 membuat model kehilangan transisi mulus antar regime. Sebaliknya, model global `ic32_regime_v1` memelajari transisi regime ini dengan menggunakan fitur `hmm_regime_enc` sebagai input, sehingga mempertahankan korelasi global.

---

## 💡 Rekomendasi

* **Gunakan Model Global (`ic32_regime_v1`)**: Untuk live trading, arsitektur global dengan fitur regime-aware (seperti `hmm_regime_enc`) terbukti jauh lebih robust dan efisien dibanding memecah model menjadi regime-specific model secara kaku.
* **Tinjau Ulang Feature Selection**: Menjaga 33 fitur lengkap memberikan informasi kontekstual yang jauh lebih kaya bagi LightGBM untuk membedakan sinyal.
