# 📊 Holdout Backtest Report: `ic32_guardian_delta_v1`

**Tanggal Pembuatan**: 2026-06-05 18:01:17 UTC
**Model Run ID**: `ic32_guardian_delta_v1`
**Periode Pengujian (Temporal OOS)**: `2025-11-01 - 2026-04-01`

> [!NOTE]
> **Ringkasan Portofolio Eksekutif**:
> *   **Total Net Profit**: **$+2,091.99 USD** (ROI Portofolio: **+99.62%**)
> *   **Rata-rata Win Rate**: **67.33%** | Total Trades: **2,426**
> *   **Rata-rata Max Drawdown (5x)**: **51.07%**
> *   **Risk-Adjusted**: Sharpe: **4.41** | Sortino: **8.14** | Calmar: **134.52** | Profit Factor: **2.72**

## 📈 Performa Scorecard Portofolio

| Metrik Utama | Nilai Portofolio | Catatan |
|:---|:---:|:---|
| **Total Net Profit ($)** | `$+2,091.99` | Akumulasi keuntungan bersih 5x leverage |
| **Portfolio ROI (%)** | `+99.62%` | ROI berdasarkan kapital portofolio $100/koin |
| **Overall Win Rate** | `67.33%` | Rasio kemenangan rata-rata seluruh aset |
| **Total Trades** | `2,426` | Jumlah total posisi yang dieksekusi |
| **Rata-rata Trade / Bulan** | `23.4` | Rata-rata frekuensi trade bulanan portofolio |
| **Rata-rata Trade / Hari** | `0.77` | Rata-rata frekuensi trade harian portofolio |
| **Max Drawdown (5x)** | `51.07%` | Rata-rata penurunan terdalam portofolio |
| **Sharpe Ratio** | `4.41` | Efisiensi profit terhadap volatilitas portofolio |
| **Sortino Ratio** | `8.14` | Efisiensi profit terhadap downside deviation |
| **Calmar Ratio** | `134.52` | Rasio return tahunan terhadap drawdown |
| **Profit Factor** | `2.72` | Rasio gross profit dibagi gross loss |
| **Max Consecutive Loss** | `9` trades | Streak kekalahan beruntun terpanjang |
| **Worst Single Trade PnL** | `-24.90%` | Kerugian terdalam dalam satu trade tunggal |
| **95% Trades Loss Under** | `10.47%` | Nilai risiko (VaR P95) kerugian maksimal |

## ↕️ Analisis Arah Signal (LONG vs SHORT)

| Arah Posisi | Jumlah Trade | Distribusi | Menang | Kalah | Win Rate | PnL Bersih ($) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **LONG** | 681 | 28.1% | 468 | 213 | 68.72% | +744.47 |
| **SHORT** | 1,745 | 71.9% | 1,164 | 581 | 66.70% | +1,347.53 |

### Rincian Rata-rata Profitabilitas per Trade

| Tipe Trade | PnL Rata-rata ($) | PnL Rata-rata (%) |
|:---|:---:|:---:|
| **Trade Menang (Wins)** | `$+2.1271` | `+8.51%` |
| **Trade Kalah (Losses)** | `$-1.7373` | `-6.95%` |

## 📅 Scorecard Bulanan Portofolio

| Bulan | Total Trades | Wins | Losses | Win Rate | Net PnL ($) |
|:---|:---:|:---:|:---:|:---:|:---:|
| 2025-11 | 550 | 364 | 186 | 66.18% | $+467.96 |
| 2025-12 | 550 | 387 | 163 | 70.36% | $+592.20 |
| 2026-01 | 491 | 317 | 174 | 64.56% | $+388.79 |
| 2026-02 | 369 | 254 | 115 | 68.83% | $+311.06 |
| 2026-03 | 466 | 310 | 156 | 66.52% | $+331.98 |

## 🚪 Distribusi Alasan Keluar Posisi (Exit Reasons)

| Alasan Exit | Jumlah | Persentase | Wins | Losses | Win Rate | PnL Bersih ($) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| `guardian_exit` | 1,954 | 80.5% | 1,597 | 357 | 81.73% | $+2,882.97 |
| `sl_hit` | 433 | 17.8% | 2 | 431 | 0.46% | $-806.30 |
| `time_exit` | 39 | 1.6% | 33 | 6 | 84.62% | $+15.33 |

## 🪙 Scorecard Per Koin (Detailed Assets)

| Token | Win Rate | Trades | LONG WR | SHORT WR | Net PnL ($) | Max DD | Sharpe | Sortino | Calmar | PF |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1000PEPE** | 60.94% | 128 | 71.0% | 57.7% | `$+65.00` | 116.55% | 2.63 | 2.42 | 18.52 | 1.54 |
| **1000SHIB** | 61.11% | 126 | 56.2% | 64.1% | `$+75.59` | 83.14% | 4.25 | 9.49 | 34.40 | 1.97 |
| **ADA** | 65.87% | 126 | 79.3% | 61.9% | `$+95.06` | 84.23% | 4.55 | 10.23 | 52.86 | 2.11 |
| **ARB** | 64.66% | 133 | 68.8% | 62.4% | `$+127.60` | 63.74% | 4.50 | 9.85 | 126.44 | 2.36 |
| **AVAX** | 65.96% | 141 | 68.6% | 65.1% | `$+129.21` | 42.42% | 5.82 | 11.33 | 194.96 | 2.80 |
| **BNB** | 71.93% | 114 | 62.5% | 74.4% | `$+69.00` | 39.16% | 4.71 | 8.09 | 61.53 | 2.54 |
| **BTC** | 72.12% | 104 | 78.8% | 69.0% | `$+68.08` | 29.99% | 5.23 | 6.01 | 78.37 | 2.74 |
| **DOGE** | 61.33% | 150 | 65.1% | 59.8% | `$+102.45` | 64.67% | 2.80 | 8.73 | 79.85 | 2.03 |
| **DOT** | 72.22% | 108 | 65.7% | 75.3% | `$+137.48` | 37.68% | 4.98 | 5.87 | 249.59 | 4.11 |
| **ETH** | 72.81% | 114 | 71.9% | 73.2% | `$+128.02` | 28.00% | 6.46 | 18.81 | 289.78 | 3.95 |
| **HBAR** | 71.43% | 105 | 73.9% | 70.7% | `$+77.29` | 25.30% | 4.69 | 6.23 | 117.89 | 2.34 |
| **LINK** | 76.58% | 111 | 85.2% | 73.8% | `$+156.96` | 49.90% | 5.63 | 17.14 | 248.92 | 4.34 |
| **NEAR** | 57.72% | 123 | 44.4% | 63.2% | `$+71.60` | 72.30% | 3.83 | 8.83 | 35.71 | 1.66 |
| **ONDO** | 70.93% | 86 | 70.0% | 71.4% | `$+122.96` | 34.85% | 4.39 | 3.30 | 214.31 | 3.40 |
| **POL** | 62.65% | 83 | 74.1% | 57.1% | `$+71.16` | 80.95% | 4.42 | 7.01 | 31.53 | 2.40 |
| **SOL** | 70.99% | 131 | 83.3% | 68.2% | `$+140.30` | 54.54% | 5.64 | 10.68 | 179.91 | 3.37 |
| **SUI** | 69.75% | 119 | 70.0% | 69.7% | `$+164.02` | 39.33% | 4.67 | 8.50 | 346.72 | 4.20 |
| **TAO** | 68.70% | 115 | 72.2% | 67.1% | `$+106.20` | 42.08% | 3.23 | 3.41 | 131.87 | 2.09 |
| **TON** | 55.79% | 95 | 51.8% | 57.4% | `$+40.24` | 27.26% | 2.55 | 5.78 | 34.18 | 1.66 |
| **TRX** | 65.00% | 100 | 64.5% | 65.2% | `$+15.52` | 28.29% | 2.13 | 1.39 | 7.91 | 1.47 |
| **XRP** | 75.44% | 114 | 81.2% | 73.2% | `$+128.26` | 28.12% | 5.52 | 7.77 | 289.66 | 3.93 |

## ⛓️ Daftar Fitur Aktif dalam Model

Total terdapat **33 fitur aktif** yang digunakan oleh LightGBM entry, LSTM Soft Confirmation, dan Exit Guardian v3:

<details>
<summary>▶ Klik untuk melihat daftar lengkap fitur aktif</summary>

1. `dist_from_8h_high`
2. `rsi_6`
3. `swing_momentum`
4. `rsi_h4`
5. `stochrsi_k`
6. `dist_liq_50x_long`
7. `trend_accel_4h`
8. `rsi_slope_h4`
9. `Fib_786`
10. `Fib_618`
11. `stochrsi_d`
12. `ofi_h4_delta`
13. `dist_liq_50x_short`
14. `Buy_Liq`
15. `relative_strength_z`
16. `dist_liq_20x_long`
17. `cvd_momentum_adv`
18. `Sell_Liq`
19. `long_short_ratio`
20. `cvd_slope_h4`
21. `ema_21_slope_h4`
22. `ema_50_h1`
23. `h4_trend`
24. `log_ret_20`
25. `whale_retail_divergence`
26. `dist_liq_20x_short`
27. `vol_price_confirm`
28. `ema_50_slope_h4`
29. `MSB_BOS`
30. `cvd`
31. `ofi_acceleration`
32. `cvd_div_h4`
33. `hmm_regime_enc`

</details>
