# 📊 Holdout Backtest Report: `ic32_guardian_ic_v1`

**Tanggal Pembuatan**: 2026-06-05 18:21:19 UTC
**Model Run ID**: `ic32_guardian_ic_v1`
**Periode Pengujian (Temporal OOS)**: `2025-11-01 - 2026-04-01`

> [!NOTE]
> **Ringkasan Portofolio Eksekutif**:
> *   **Total Net Profit**: **$+2,064.57 USD** (ROI Portofolio: **+98.31%**)
> *   **Rata-rata Win Rate**: **66.99%** | Total Trades: **2,426**
> *   **Rata-rata Max Drawdown (5x)**: **51.62%**
> *   **Risk-Adjusted**: Sharpe: **4.35** | Sortino: **7.82** | Calmar: **128.09** | Profit Factor: **2.68**

## 📈 Performa Scorecard Portofolio

| Metrik Utama | Nilai Portofolio | Catatan |
|:---|:---:|:---|
| **Total Net Profit ($)** | `$+2,064.57` | Akumulasi keuntungan bersih 5x leverage |
| **Portfolio ROI (%)** | `+98.31%` | ROI berdasarkan kapital portofolio $100/koin |
| **Overall Win Rate** | `66.99%` | Rasio kemenangan rata-rata seluruh aset |
| **Total Trades** | `2,426` | Jumlah total posisi yang dieksekusi |
| **Rata-rata Trade / Bulan** | `23.4` | Rata-rata frekuensi trade bulanan portofolio |
| **Rata-rata Trade / Hari** | `0.77` | Rata-rata frekuensi trade harian portofolio |
| **Max Drawdown (5x)** | `51.62%` | Rata-rata penurunan terdalam portofolio |
| **Sharpe Ratio** | `4.35` | Efisiensi profit terhadap volatilitas portofolio |
| **Sortino Ratio** | `7.82` | Efisiensi profit terhadap downside deviation |
| **Calmar Ratio** | `128.09` | Rasio return tahunan terhadap drawdown |
| **Profit Factor** | `2.68` | Rasio gross profit dibagi gross loss |
| **Max Consecutive Loss** | `9` trades | Streak kekalahan beruntun terpanjang |
| **Worst Single Trade PnL** | `-24.90%` | Kerugian terdalam dalam satu trade tunggal |
| **95% Trades Loss Under** | `10.40%` | Nilai risiko (VaR P95) kerugian maksimal |

## ↕️ Analisis Arah Signal (LONG vs SHORT)

| Arah Posisi | Jumlah Trade | Distribusi | Menang | Kalah | Win Rate | PnL Bersih ($) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **LONG** | 681 | 28.1% | 463 | 218 | 67.99% | +730.24 |
| **SHORT** | 1,745 | 71.9% | 1,161 | 584 | 66.53% | +1,334.33 |

### Rincian Rata-rata Profitabilitas per Trade

| Tipe Trade | PnL Rata-rata ($) | PnL Rata-rata (%) |
|:---|:---:|:---:|
| **Trade Menang (Wins)** | `$+2.1197` | `+8.48%` |
| **Trade Kalah (Losses)** | `$-1.7180` | `-6.87%` |

## 📅 Scorecard Bulanan Portofolio

| Bulan | Total Trades | Wins | Losses | Win Rate | Net PnL ($) |
|:---|:---:|:---:|:---:|:---:|:---:|
| 2025-11 | 550 | 365 | 185 | 66.36% | $+449.87 |
| 2025-12 | 550 | 381 | 169 | 69.27% | $+583.56 |
| 2026-01 | 491 | 315 | 176 | 64.15% | $+370.22 |
| 2026-02 | 369 | 253 | 116 | 68.56% | $+330.49 |
| 2026-03 | 466 | 310 | 156 | 66.52% | $+330.42 |

## 🚪 Distribusi Alasan Keluar Posisi (Exit Reasons)

| Alasan Exit | Jumlah | Persentase | Wins | Losses | Win Rate | PnL Bersih ($) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| `guardian_exit` | 1,948 | 80.3% | 1,586 | 362 | 81.42% | $+2,846.26 |
| `sl_hit` | 436 | 18.0% | 2 | 434 | 0.46% | $-798.19 |
| `time_exit` | 42 | 1.7% | 36 | 6 | 85.71% | $+16.51 |

## 🪙 Scorecard Per Koin (Detailed Assets)

| Token | Win Rate | Trades | LONG WR | SHORT WR | Net PnL ($) | Max DD | Sharpe | Sortino | Calmar | PF |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1000PEPE** | 63.28% | 128 | 71.0% | 60.8% | `$+71.15` | 116.55% | 2.71 | 2.37 | 21.89 | 1.62 |
| **1000SHIB** | 59.52% | 126 | 54.2% | 62.8% | `$+66.02` | 83.14% | 3.62 | 6.25 | 26.70 | 1.82 |
| **ADA** | 65.08% | 126 | 79.3% | 60.8% | `$+96.25` | 84.23% | 4.40 | 10.42 | 54.17 | 2.12 |
| **ARB** | 63.91% | 133 | 68.8% | 61.2% | `$+120.09` | 63.74% | 4.40 | 9.16 | 111.65 | 2.25 |
| **AVAX** | 68.09% | 141 | 68.6% | 67.9% | `$+129.84` | 42.42% | 5.98 | 11.78 | 196.94 | 2.91 |
| **BNB** | 71.93% | 114 | 62.5% | 74.4% | `$+71.88` | 39.16% | 4.86 | 8.36 | 66.43 | 2.61 |
| **BTC** | 72.12% | 104 | 78.8% | 69.0% | `$+68.55` | 29.99% | 5.24 | 6.10 | 79.38 | 2.78 |
| **DOGE** | 61.33% | 150 | 65.1% | 59.8% | `$+105.10` | 64.17% | 2.84 | 8.34 | 84.69 | 2.07 |
| **DOT** | 71.30% | 108 | 62.9% | 75.3% | `$+136.70` | 37.68% | 4.85 | 5.83 | 246.65 | 4.08 |
| **ETH** | 73.68% | 114 | 71.9% | 74.4% | `$+127.56` | 28.00% | 6.47 | 18.62 | 287.63 | 4.02 |
| **HBAR** | 71.43% | 105 | 73.9% | 70.7% | `$+86.35` | 25.30% | 4.87 | 6.58 | 145.85 | 2.50 |
| **LINK** | 75.68% | 111 | 85.2% | 72.6% | `$+155.41` | 49.90% | 5.61 | 16.82 | 243.75 | 4.29 |
| **NEAR** | 56.91% | 123 | 41.7% | 63.2% | `$+65.74` | 72.30% | 3.54 | 8.51 | 30.47 | 1.61 |
| **ONDO** | 70.93% | 86 | 70.0% | 71.4% | `$+123.82` | 34.85% | 4.24 | 3.17 | 217.39 | 3.45 |
| **POL** | 63.86% | 83 | 74.1% | 58.9% | `$+69.83` | 84.30% | 4.37 | 7.06 | 29.23 | 2.38 |
| **SOL** | 70.23% | 131 | 83.3% | 67.3% | `$+131.84` | 54.54% | 5.46 | 10.45 | 158.10 | 3.21 |
| **SUI** | 66.39% | 119 | 66.7% | 66.3% | `$+150.19` | 46.47% | 4.68 | 6.74 | 243.54 | 3.80 |
| **TAO** | 68.70% | 115 | 72.2% | 67.1% | `$+117.84` | 42.08% | 3.36 | 3.50 | 162.71 | 2.21 |
| **TON** | 54.74% | 95 | 51.8% | 55.9% | `$+35.19` | 28.89% | 2.55 | 5.48 | 25.90 | 1.57 |
| **TRX** | 64.00% | 100 | 61.3% | 65.2% | `$+16.21` | 28.29% | 2.17 | 1.46 | 8.40 | 1.49 |
| **XRP** | 73.68% | 114 | 81.2% | 70.7% | `$+119.00` | 28.12% | 5.08 | 7.30 | 248.43 | 3.59 |

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
