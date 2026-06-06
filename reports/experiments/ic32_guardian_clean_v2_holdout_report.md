# 📊 Holdout Backtest Report: `ic32_guardian_clean_v2`

**Tanggal Pembuatan**: 2026-06-05 18:08:31 UTC
**Model Run ID**: `ic32_guardian_clean_v2`
**Periode Pengujian (Temporal OOS)**: `2025-11-01 - 2026-04-01`

> [!NOTE]
> **Ringkasan Portofolio Eksekutif**:
> *   **Total Net Profit**: **$+2,089.14 USD** (ROI Portofolio: **+99.48%**)
> *   **Rata-rata Win Rate**: **67.47%** | Total Trades: **2,426**
> *   **Rata-rata Max Drawdown (5x)**: **50.29%**
> *   **Risk-Adjusted**: Sharpe: **4.41** | Sortino: **7.91** | Calmar: **133.09** | Profit Factor: **2.72**

## 📈 Performa Scorecard Portofolio

| Metrik Utama | Nilai Portofolio | Catatan |
|:---|:---:|:---|
| **Total Net Profit ($)** | `$+2,089.14` | Akumulasi keuntungan bersih 5x leverage |
| **Portfolio ROI (%)** | `+99.48%` | ROI berdasarkan kapital portofolio $100/koin |
| **Overall Win Rate** | `67.47%` | Rasio kemenangan rata-rata seluruh aset |
| **Total Trades** | `2,426` | Jumlah total posisi yang dieksekusi |
| **Rata-rata Trade / Bulan** | `23.4` | Rata-rata frekuensi trade bulanan portofolio |
| **Rata-rata Trade / Hari** | `0.77` | Rata-rata frekuensi trade harian portofolio |
| **Max Drawdown (5x)** | `50.29%` | Rata-rata penurunan terdalam portofolio |
| **Sharpe Ratio** | `4.41` | Efisiensi profit terhadap volatilitas portofolio |
| **Sortino Ratio** | `7.91` | Efisiensi profit terhadap downside deviation |
| **Calmar Ratio** | `133.09` | Rasio return tahunan terhadap drawdown |
| **Profit Factor** | `2.72` | Rasio gross profit dibagi gross loss |
| **Max Consecutive Loss** | `9` trades | Streak kekalahan beruntun terpanjang |
| **Worst Single Trade PnL** | `-24.90%` | Kerugian terdalam dalam satu trade tunggal |
| **95% Trades Loss Under** | `10.50%` | Nilai risiko (VaR P95) kerugian maksimal |

## ↕️ Analisis Arah Signal (LONG vs SHORT)

| Arah Posisi | Jumlah Trade | Distribusi | Menang | Kalah | Win Rate | PnL Bersih ($) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **LONG** | 681 | 28.1% | 460 | 221 | 67.55% | +728.19 |
| **SHORT** | 1,745 | 71.9% | 1,176 | 569 | 67.39% | +1,360.95 |

### Rincian Rata-rata Profitabilitas per Trade

| Tipe Trade | PnL Rata-rata ($) | PnL Rata-rata (%) |
|:---|:---:|:---:|
| **Trade Menang (Wins)** | `$+2.1177` | `+8.47%` |
| **Trade Kalah (Losses)** | `$-1.7410` | `-6.96%` |

## 📅 Scorecard Bulanan Portofolio

| Bulan | Total Trades | Wins | Losses | Win Rate | Net PnL ($) |
|:---|:---:|:---:|:---:|:---:|:---:|
| 2025-11 | 550 | 363 | 187 | 66.00% | $+447.31 |
| 2025-12 | 550 | 389 | 161 | 70.73% | $+613.62 |
| 2026-01 | 491 | 316 | 175 | 64.36% | $+386.30 |
| 2026-02 | 369 | 254 | 115 | 68.83% | $+307.53 |
| 2026-03 | 466 | 314 | 152 | 67.38% | $+334.38 |

## 🚪 Distribusi Alasan Keluar Posisi (Exit Reasons)

| Alasan Exit | Jumlah | Persentase | Wins | Losses | Win Rate | PnL Bersih ($) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| `guardian_exit` | 1,965 | 81.0% | 1,601 | 364 | 81.48% | $+2,870.88 |
| `sl_hit` | 421 | 17.4% | 1 | 420 | 0.24% | $-797.52 |
| `time_exit` | 40 | 1.6% | 34 | 6 | 85.00% | $+15.79 |

## 🪙 Scorecard Per Koin (Detailed Assets)

| Token | Win Rate | Trades | LONG WR | SHORT WR | Net PnL ($) | Max DD | Sharpe | Sortino | Calmar | PF |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1000PEPE** | 64.06% | 128 | 71.0% | 61.9% | `$+69.37` | 102.81% | 2.75 | 2.55 | 23.68 | 1.60 |
| **1000SHIB** | 63.49% | 126 | 56.2% | 68.0% | `$+84.03` | 83.14% | 4.52 | 10.21 | 42.10 | 2.11 |
| **ADA** | 66.67% | 126 | 79.3% | 62.9% | `$+100.43` | 71.53% | 4.59 | 9.80 | 69.40 | 2.19 |
| **ARB** | 64.66% | 133 | 66.7% | 63.5% | `$+122.45` | 63.74% | 4.52 | 9.78 | 116.18 | 2.30 |
| **AVAX** | 65.96% | 141 | 68.6% | 65.1% | `$+129.41` | 42.42% | 5.77 | 10.98 | 195.58 | 2.79 |
| **BNB** | 71.05% | 114 | 62.5% | 73.3% | `$+69.64` | 39.16% | 4.74 | 8.24 | 62.60 | 2.52 |
| **BTC** | 73.08% | 104 | 78.8% | 70.4% | `$+68.05` | 29.99% | 5.28 | 5.92 | 78.30 | 2.80 |
| **DOGE** | 62.00% | 150 | 62.8% | 61.7% | `$+98.70` | 64.67% | 2.51 | 7.67 | 74.15 | 1.99 |
| **DOT** | 71.30% | 108 | 62.9% | 75.3% | `$+135.25` | 37.68% | 4.82 | 5.81 | 241.28 | 4.05 |
| **ETH** | 71.93% | 114 | 71.9% | 72.0% | `$+128.70` | 28.00% | 6.50 | 18.41 | 292.99 | 3.92 |
| **HBAR** | 71.43% | 105 | 73.9% | 70.7% | `$+78.25` | 25.30% | 4.53 | 6.12 | 120.71 | 2.36 |
| **LINK** | 77.48% | 111 | 85.2% | 75.0% | `$+153.67` | 49.90% | 6.11 | 15.02 | 238.03 | 4.40 |
| **NEAR** | 57.72% | 123 | 41.7% | 64.4% | `$+69.51` | 80.79% | 3.69 | 8.65 | 30.24 | 1.63 |
| **ONDO** | 70.93% | 86 | 70.0% | 71.4% | `$+122.98` | 34.85% | 4.21 | 3.26 | 214.38 | 3.39 |
| **POL** | 61.45% | 83 | 70.4% | 57.1% | `$+69.69` | 80.95% | 4.37 | 6.98 | 30.33 | 2.36 |
| **SOL** | 70.99% | 131 | 83.3% | 68.2% | `$+139.53` | 54.54% | 5.67 | 10.68 | 177.83 | 3.36 |
| **SUI** | 68.07% | 119 | 66.7% | 68.5% | `$+155.37` | 39.33% | 4.49 | 8.20 | 309.08 | 3.91 |
| **TAO** | 68.70% | 115 | 72.2% | 67.1% | `$+107.35` | 42.08% | 3.27 | 3.45 | 134.76 | 2.11 |
| **TON** | 54.74% | 95 | 48.1% | 57.4% | `$+39.57` | 28.89% | 2.53 | 5.72 | 31.37 | 1.65 |
| **TRX** | 64.00% | 100 | 61.3% | 65.2% | `$+15.93` | 28.29% | 2.18 | 1.42 | 8.20 | 1.48 |
| **XRP** | 77.19% | 114 | 81.2% | 75.6% | `$+131.26` | 28.12% | 5.60 | 7.26 | 303.83 | 4.14 |

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
