# 📊 Holdout Backtest Report: `ic32_lgbm_v1`

**Tanggal Pembuatan**: 2026-06-05 00:25:58 UTC
**Model Run ID**: `ic32_lgbm_v1`
**Periode Pengujian (Temporal OOS)**: `2025-11-01 - 2026-04-01`

> [!NOTE]
> **Ringkasan Portofolio Eksekutif**:
> *   **Total Net Profit**: **$+2,095.54 USD** (ROI Portofolio: **+99.79%**)
> *   **Rata-rata Win Rate**: **68.79%** | Total Trades: **2,385**
> *   **Rata-rata Max Drawdown (5x)**: **53.31%**
> *   **Risk-Adjusted**: Sharpe: **6.33** | Sortino: **15.90** | Calmar: **20.69** | Profit Factor: **2.72**

## 📈 Performa Scorecard Portofolio

| Metrik Utama | Nilai Portofolio | Catatan |
|:---|:---:|:---|
| **Total Net Profit ($)** | `$+2,095.54` | Akumulasi keuntungan bersih 5x leverage |
| **Portfolio ROI (%)** | `+99.79%` | ROI berdasarkan kapital portofolio $100/koin |
| **Overall Win Rate** | `68.79%` | Rasio kemenangan rata-rata seluruh aset |
| **Total Trades** | `2,385` | Jumlah total posisi yang dieksekusi |
| **Rata-rata Trade / Bulan** | `23.1` | Rata-rata frekuensi trade bulanan portofolio |
| **Rata-rata Trade / Hari** | `0.76` | Rata-rata frekuensi trade harian portofolio |
| **Max Drawdown (5x)** | `53.31%` | Rata-rata penurunan terdalam portofolio |
| **Sharpe Ratio** | `6.33` | Efisiensi profit terhadap volatilitas portofolio |
| **Sortino Ratio** | `15.90` | Efisiensi profit terhadap downside deviation |
| **Calmar Ratio** | `20.69` | Rasio return tahunan terhadap drawdown |
| **Profit Factor** | `2.72` | Rasio gross profit dibagi gross loss |
| **Max Consecutive Loss** | `12` trades | Streak kekalahan beruntun terpanjang |
| **Worst Single Trade PnL** | `-24.90%` | Kerugian terdalam dalam satu trade tunggal |
| **95% Trades Loss Under** | `10.30%` | Nilai risiko (VaR P95) kerugian maksimal |

## ↕️ Analisis Arah Signal (LONG vs SHORT)

| Arah Posisi | Jumlah Trade | Distribusi | Menang | Kalah | Win Rate | PnL Bersih ($) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **LONG** | 672 | 28.2% | 456 | 216 | 67.86% | +666.86 |
| **SHORT** | 1,713 | 71.8% | 1,188 | 525 | 69.35% | +1,428.68 |

### Rincian Rata-rata Profitabilitas per Trade

| Tipe Trade | PnL Rata-rata ($) | PnL Rata-rata (%) |
|:---|:---:|:---:|
| **Trade Menang (Wins)** | `$+2.0671` | `+8.27%` |
| **Trade Kalah (Losses)** | `$-1.7581` | `-7.03%` |

## 📅 Scorecard Bulanan Portofolio

| Bulan | Total Trades | Wins | Losses | Win Rate | Net PnL ($) |
|:---|:---:|:---:|:---:|:---:|:---:|
| 2025-11 | 537 | 360 | 177 | 67.04% | $+417.07 |
| 2025-12 | 555 | 407 | 148 | 73.33% | $+635.94 |
| 2026-01 | 475 | 323 | 152 | 68.00% | $+449.52 |
| 2026-02 | 369 | 244 | 125 | 66.12% | $+257.86 |
| 2026-03 | 449 | 310 | 139 | 69.04% | $+335.16 |

## 🚪 Distribusi Alasan Keluar Posisi (Exit Reasons)

| Alasan Exit | Jumlah | Persentase | Wins | Losses | Win Rate | PnL Bersih ($) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| `guardian_exit` | 1,959 | 82.1% | 1,614 | 345 | 82.39% | $+2,817.51 |
| `sl_hit` | 393 | 16.5% | 2 | 391 | 0.51% | $-736.18 |
| `time_exit` | 33 | 1.4% | 28 | 5 | 84.85% | $+14.20 |

## 🪙 Scorecard Per Koin (Detailed Assets)

| Token | Win Rate | Trades | LONG WR | SHORT WR | Net PnL ($) | Max DD | Sharpe | Sortino | Calmar | PF |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1000PEPE** | 69.35% | 124 | 69.4% | 69.3% | `$+82.73` | 76.53% | 4.22 | 9.17 | 10.53 | 1.83 |
| **1000SHIB** | 63.49% | 126 | 56.5% | 67.5% | `$+59.04` | 93.06% | 4.20 | 10.68 | 6.18 | 1.80 |
| **ADA** | 70.63% | 126 | 78.1% | 68.1% | `$+139.10` | 55.32% | 7.62 | 23.18 | 24.49 | 2.90 |
| **ARB** | 65.85% | 123 | 71.1% | 62.8% | `$+108.51` | 87.64% | 5.56 | 13.30 | 12.06 | 2.30 |
| **AVAX** | 69.86% | 146 | 69.4% | 70.0% | `$+138.21` | 58.43% | 8.19 | 27.18 | 23.04 | 2.88 |
| **BNB** | 73.00% | 100 | 57.1% | 77.2% | `$+59.75` | 57.03% | 5.44 | 8.80 | 10.20 | 2.34 |
| **BTC** | 74.77% | 107 | 86.2% | 70.5% | `$+72.12` | 25.90% | 7.31 | 14.90 | 27.12 | 2.96 |
| **DOGE** | 62.86% | 140 | 65.9% | 61.5% | `$+117.67` | 57.47% | 6.09 | 15.96 | 19.94 | 2.37 |
| **DOT** | 70.91% | 110 | 58.8% | 76.3% | `$+130.83` | 37.68% | 7.95 | 16.41 | 33.82 | 3.63 |
| **ETH** | 77.50% | 120 | 75.9% | 78.0% | `$+144.48` | 29.35% | 10.49 | 20.68 | 47.95 | 4.97 |
| **HBAR** | 67.68% | 99 | 65.4% | 68.5% | `$+63.94` | 44.58% | 4.90 | 12.94 | 13.97 | 2.07 |
| **LINK** | 73.04% | 115 | 77.8% | 71.6% | `$+134.10` | 50.71% | 8.18 | 25.25 | 25.76 | 3.42 |
| **NEAR** | 62.73% | 110 | 50.0% | 68.4% | `$+113.52` | 79.09% | 6.02 | 16.51 | 13.98 | 2.51 |
| **ONDO** | 73.47% | 98 | 75.0% | 72.9% | `$+134.86` | 40.96% | 8.00 | 20.17 | 32.07 | 3.61 |
| **POL** | 61.36% | 88 | 69.2% | 58.1% | `$+67.29` | 86.25% | 4.75 | 10.34 | 7.60 | 2.31 |
| **SOL** | 69.72% | 109 | 82.6% | 66.3% | `$+104.01` | 50.42% | 6.26 | 14.30 | 20.09 | 2.84 |
| **SUI** | 68.91% | 119 | 65.6% | 70.1% | `$+137.34` | 41.56% | 6.86 | 15.81 | 32.19 | 3.10 |
| **TAO** | 69.57% | 115 | 69.0% | 69.8% | `$+102.70` | 56.38% | 4.96 | 9.47 | 17.74 | 2.08 |
| **TON** | 55.67% | 97 | 51.8% | 57.1% | `$+45.76` | 34.31% | 3.72 | 10.09 | 12.99 | 1.73 |
| **TRX** | 70.10% | 97 | 58.8% | 76.2% | `$+23.52` | 21.81% | 4.24 | 9.41 | 10.50 | 1.91 |
| **XRP** | 74.14% | 116 | 79.4% | 72.0% | `$+116.07` | 35.02% | 8.00 | 29.26 | 32.28 | 3.47 |

## ⛓️ Daftar Fitur Aktif dalam Model

Total terdapat **32 fitur aktif** yang digunakan oleh LightGBM entry, LSTM Soft Confirmation, dan Exit Guardian v3:

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

</details>
