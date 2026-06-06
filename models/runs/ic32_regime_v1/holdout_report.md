# 📊 Holdout Backtest Report: `ic32_regime_v1`

**Tanggal Pembuatan**: 2026-06-05 06:05:40 UTC
**Model Run ID**: `ic32_regime_v1`
**Periode Pengujian (Temporal OOS)**: `2025-11-01 - 2026-04-01`

> [!NOTE]
> **Ringkasan Portofolio Eksekutif**:
> *   **Total Net Profit**: **$+2,100.50 USD** (ROI Portofolio: **+100.02%**)
> *   **Rata-rata Win Rate**: **69.20%** | Total Trades: **2,426**
> *   **Rata-rata Max Drawdown (5x)**: **49.73%**
> *   **Risk-Adjusted**: Sharpe: **6.44** | Sortino: **16.53** | Calmar: **23.18** | Profit Factor: **2.81**

## 📈 Performa Scorecard Portofolio

| Metrik Utama | Nilai Portofolio | Catatan |
|:---|:---:|:---|
| **Total Net Profit ($)** | `$+2,100.50` | Akumulasi keuntungan bersih 5x leverage |
| **Portfolio ROI (%)** | `+100.02%` | ROI berdasarkan kapital portofolio $100/koin |
| **Overall Win Rate** | `69.20%` | Rasio kemenangan rata-rata seluruh aset |
| **Total Trades** | `2,426` | Jumlah total posisi yang dieksekusi |
| **Rata-rata Trade / Bulan** | `23.4` | Rata-rata frekuensi trade bulanan portofolio |
| **Rata-rata Trade / Hari** | `0.77` | Rata-rata frekuensi trade harian portofolio |
| **Max Drawdown (5x)** | `49.73%` | Rata-rata penurunan terdalam portofolio |
| **Sharpe Ratio** | `6.44` | Efisiensi profit terhadap volatilitas portofolio |
| **Sortino Ratio** | `16.53` | Efisiensi profit terhadap downside deviation |
| **Calmar Ratio** | `23.18` | Rasio return tahunan terhadap drawdown |
| **Profit Factor** | `2.81` | Rasio gross profit dibagi gross loss |
| **Max Consecutive Loss** | `9` trades | Streak kekalahan beruntun terpanjang |
| **Worst Single Trade PnL** | `-24.90%` | Kerugian terdalam dalam satu trade tunggal |
| **95% Trades Loss Under** | `10.30%` | Nilai risiko (VaR P95) kerugian maksimal |

## ↕️ Analisis Arah Signal (LONG vs SHORT)

| Arah Posisi | Jumlah Trade | Distribusi | Menang | Kalah | Win Rate | PnL Bersih ($) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **LONG** | 681 | 28.1% | 465 | 216 | 68.28% | +701.44 |
| **SHORT** | 1,745 | 71.9% | 1,213 | 532 | 69.51% | +1,399.07 |

### Rincian Rata-rata Profitabilitas per Trade

| Tipe Trade | PnL Rata-rata ($) | PnL Rata-rata (%) |
|:---|:---:|:---:|
| **Trade Menang (Wins)** | `$+2.0337` | `+8.13%` |
| **Trade Kalah (Losses)** | `$-1.7541` | `-7.02%` |

## 📅 Scorecard Bulanan Portofolio

| Bulan | Total Trades | Wins | Losses | Win Rate | Net PnL ($) |
|:---|:---:|:---:|:---:|:---:|:---:|
| 2025-11 | 550 | 378 | 172 | 68.73% | $+446.40 |
| 2025-12 | 550 | 397 | 153 | 72.18% | $+619.47 |
| 2026-01 | 491 | 325 | 166 | 66.19% | $+402.40 |
| 2026-02 | 369 | 259 | 110 | 70.19% | $+298.50 |
| 2026-03 | 466 | 319 | 147 | 68.45% | $+333.73 |

## 🚪 Distribusi Alasan Keluar Posisi (Exit Reasons)

| Alasan Exit | Jumlah | Persentase | Wins | Losses | Win Rate | PnL Bersih ($) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| `guardian_exit` | 1,986 | 81.9% | 1,642 | 344 | 82.68% | $+2,844.94 |
| `sl_hit` | 399 | 16.4% | 1 | 398 | 0.25% | $-761.17 |
| `time_exit` | 41 | 1.7% | 35 | 6 | 85.37% | $+16.74 |

## 🪙 Scorecard Per Koin (Detailed Assets)

| Token | Win Rate | Trades | LONG WR | SHORT WR | Net PnL ($) | Max DD | Sharpe | Sortino | Calmar | PF |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1000PEPE** | 66.41% | 128 | 71.0% | 65.0% | `$+65.59` | 102.81% | 3.30 | 6.79 | 6.21 | 1.58 |
| **1000SHIB** | 62.70% | 126 | 56.2% | 66.7% | `$+65.41` | 85.98% | 4.35 | 11.07 | 7.41 | 1.85 |
| **ADA** | 67.46% | 126 | 72.4% | 66.0% | `$+96.17` | 71.53% | 5.39 | 15.86 | 13.10 | 2.15 |
| **ARB** | 65.41% | 133 | 68.8% | 63.5% | `$+113.92` | 63.74% | 5.44 | 12.22 | 17.41 | 2.22 |
| **AVAX** | 72.34% | 141 | 71.4% | 72.6% | `$+150.11` | 42.42% | 9.23 | 31.86 | 34.47 | 3.38 |
| **BNB** | 74.56% | 114 | 62.5% | 77.8% | `$+75.86` | 39.16% | 7.30 | 17.93 | 18.87 | 2.83 |
| **BTC** | 75.00% | 104 | 78.8% | 73.2% | `$+71.10` | 26.62% | 7.48 | 14.65 | 26.01 | 2.99 |
| **DOGE** | 63.33% | 150 | 67.4% | 61.7% | `$+122.54` | 56.10% | 6.46 | 16.21 | 21.28 | 2.35 |
| **DOT** | 72.22% | 108 | 62.9% | 76.7% | `$+135.11` | 37.68% | 8.43 | 17.57 | 34.92 | 4.14 |
| **ETH** | 71.93% | 114 | 68.8% | 73.2% | `$+119.19` | 28.00% | 8.33 | 19.50 | 41.46 | 3.76 |
| **HBAR** | 71.43% | 105 | 73.9% | 70.7% | `$+80.35` | 43.63% | 6.15 | 14.73 | 17.94 | 2.45 |
| **LINK** | 76.58% | 111 | 81.5% | 75.0% | `$+142.25` | 49.90% | 9.27 | 24.82 | 27.77 | 4.17 |
| **NEAR** | 59.35% | 123 | 44.4% | 65.5% | `$+86.04` | 72.30% | 4.33 | 11.20 | 11.59 | 1.85 |
| **ONDO** | 73.26% | 86 | 73.3% | 73.2% | `$+116.37` | 34.85% | 7.07 | 17.26 | 32.52 | 3.38 |
| **POL** | 63.86% | 83 | 74.1% | 58.9% | `$+64.01` | 83.24% | 4.52 | 9.91 | 7.49 | 2.27 |
| **SOL** | 72.52% | 131 | 87.5% | 69.2% | `$+135.58` | 56.44% | 8.05 | 23.22 | 23.40 | 3.35 |
| **SUI** | 73.11% | 119 | 66.7% | 75.3% | `$+162.10` | 31.28% | 8.77 | 23.82 | 50.48 | 4.48 |
| **TAO** | 68.70% | 115 | 66.7% | 69.6% | `$+106.17` | 42.08% | 4.87 | 9.54 | 24.58 | 2.09 |
| **TON** | 56.84% | 95 | 55.6% | 57.4% | `$+38.85` | 27.26% | 3.34 | 9.19 | 13.88 | 1.65 |
| **TRX** | 69.00% | 100 | 64.5% | 71.0% | `$+23.06` | 21.13% | 4.06 | 6.88 | 10.63 | 1.87 |
| **XRP** | 77.19% | 114 | 81.2% | 75.6% | `$+130.71` | 28.12% | 9.01 | 33.00 | 45.28 | 4.26 |

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
