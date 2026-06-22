# 📊 Holdout Backtest Report: `cascade_v2.5_hybrid_accel_scenario_R`

**Tanggal Pembuatan**: 2026-06-02 20:51:39 UTC
**Model Run ID**: `cascade_v2.5_hybrid_accel_scenario_R`
**Periode Pengujian (Temporal OOS)**: `2025-11-01 - 2026-04-01`

> [!NOTE]
> **Ringkasan Portofolio Eksekutif**:
> *   **Total Net Profit**: **$+1,884.97 USD** (ROI Portofolio: **+89.76%**)
> *   **Rata-rata Win Rate**: **59.66%** | Total Trades: **4,623**
> *   **Rata-rata Max Drawdown (5x)**: **94.32%**
> *   **Risk-Adjusted**: Sharpe: **4.28** | Sortino: **12.53** | Calmar: **13.17** | Profit Factor: **2.36**

## 📈 Performa Scorecard Portofolio

| Metrik Utama | Nilai Portofolio | Catatan |
|:---|:---:|:---|
| **Total Net Profit ($)** | `$+1,884.97` | Akumulasi keuntungan bersih 5x leverage |
| **Portfolio ROI (%)** | `+89.76%` | ROI berdasarkan kapital portofolio $100/koin |
| **Overall Win Rate** | `59.66%` | Rasio kemenangan rata-rata seluruh aset |
| **Total Trades** | `4,623` | Jumlah total posisi yang dieksekusi |
| **Rata-rata Trade / Bulan** | `44.7` | Rata-rata frekuensi trade bulanan portofolio |
| **Rata-rata Trade / Hari** | `1.47` | Rata-rata frekuensi trade harian portofolio |
| **Max Drawdown (5x)** | `94.32%` | Rata-rata penurunan terdalam portofolio |
| **Sharpe Ratio** | `4.28` | Efisiensi profit terhadap volatilitas portofolio |
| **Sortino Ratio** | `12.53` | Efisiensi profit terhadap downside deviation |
| **Calmar Ratio** | `13.17` | Rasio return tahunan terhadap drawdown |
| **Profit Factor** | `2.36` | Rasio gross profit dibagi gross loss |
| **Max Consecutive Loss** | `17` trades | Streak kekalahan beruntun terpanjang |
| **Worst Single Trade PnL** | `-24.90%` | Kerugian terdalam dalam satu trade tunggal |
| **95% Trades Loss Under** | `10.99%` | Nilai risiko (VaR P95) kerugian maksimal |

## ↕️ Analisis Arah Signal (LONG vs SHORT)

| Arah Posisi | Jumlah Trade | Distribusi | Menang | Kalah | Win Rate | PnL Bersih ($) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **LONG** | 1,839 | 39.8% | 966 | 873 | 52.53% | +321.89 |
| **SHORT** | 2,784 | 60.2% | 1,763 | 1,021 | 63.33% | +1,563.09 |

### Rincian Rata-rata Profitabilitas per Trade

| Tipe Trade | PnL Rata-rata ($) | PnL Rata-rata (%) |
|:---|:---:|:---:|
| **Trade Menang (Wins)** | `$+1.9373` | `+7.75%` |
| **Trade Kalah (Losses)** | `$-1.7961` | `-7.18%` |

## 📅 Scorecard Bulanan Portofolio

| Bulan | Total Trades | Wins | Losses | Win Rate | Net PnL ($) |
|:---|:---:|:---:|:---:|:---:|:---:|
| 2025-11 | 840 | 474 | 366 | 56.43% | $+269.43 |
| 2025-12 | 946 | 589 | 357 | 62.26% | $+594.29 |
| 2026-01 | 927 | 588 | 339 | 63.43% | $+594.94 |
| 2026-02 | 887 | 455 | 432 | 51.30% | $+18.66 |
| 2026-03 | 1023 | 623 | 400 | 60.90% | $+407.65 |

## 🚪 Distribusi Alasan Keluar Posisi (Exit Reasons)

| Alasan Exit | Jumlah | Persentase | Wins | Losses | Win Rate | PnL Bersih ($) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| `guardian_exit` | 3,561 | 77.0% | 2,676 | 885 | 75.15% | $+3,804.81 |
| `sl_hit` | 1,000 | 21.6% | 2 | 998 | 0.20% | $-1,941.79 |
| `time_exit` | 62 | 1.3% | 51 | 11 | 82.26% | $+21.96 |

## 🪙 Scorecard Per Koin (Detailed Assets)

| Token | Win Rate | Trades | LONG WR | SHORT WR | Net PnL ($) | Max DD | Sharpe | Sortino | Calmar | PF |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1000PEPE** | 83.33% | 12 | 33.3% | 100.0% | `$+35.27` | 11.61% | 5.14 | 27.89 | 29.59 | 8.06 |
| **1000SHIB** | 77.78% | 9 | 50.0% | 85.7% | `$+8.62` | 4.88% | 2.63 | 5.42 | 17.21 | 7.79 |
| **ADA** | 64.50% | 169 | 56.5% | 70.0% | `$+137.92` | 51.87% | 6.68 | 21.10 | 25.90 | 2.28 |
| **ARB** | 58.21% | 268 | 52.0% | 63.6% | `$+156.90` | 85.52% | 5.58 | 14.09 | 17.87 | 1.73 |
| **AVAX** | 56.66% | 383 | 53.6% | 58.4% | `$+99.73` | 177.38% | 3.55 | 8.65 | 5.48 | 1.33 |
| **BNB** | 54.09% | 220 | 45.4% | 61.0% | `$+8.40` | 92.79% | 0.57 | 1.63 | 0.88 | 1.06 |
| **BTC** | 0.00% | 0 | 0.0% | 0.0% | `$+0.00` | 0.00% | 0.00 | 0.00 | 0.00 | 0.00 |
| **DOGE** | 83.87% | 31 | 80.0% | 87.5% | `$+57.20` | 13.85% | 6.32 | 22.95 | 40.22 | 6.60 |
| **DOT** | 58.31% | 319 | 47.9% | 64.5% | `$+91.30` | 103.57% | 3.60 | 9.17 | 8.59 | 1.38 |
| **ETH** | 66.48% | 176 | 61.2% | 69.7% | `$+118.92` | 107.03% | 6.29 | 16.75 | 10.82 | 2.09 |
| **HBAR** | 58.82% | 221 | 48.6% | 68.1% | `$+48.60` | 51.52% | 2.59 | 7.27 | 9.19 | 1.30 |
| **LINK** | 59.33% | 327 | 58.6% | 59.7% | `$+127.77` | 103.69% | 4.76 | 16.20 | 12.00 | 1.49 |
| **NEAR** | 59.69% | 320 | 43.2% | 69.3% | `$+157.37` | 84.36% | 5.50 | 12.65 | 18.17 | 1.66 |
| **ONDO** | 60.43% | 278 | 61.7% | 59.8% | `$+162.00` | 132.54% | 5.91 | 17.11 | 11.90 | 1.74 |
| **POL** | 55.04% | 258 | 50.0% | 59.0% | `$+106.10` | 164.34% | 4.35 | 12.36 | 6.29 | 1.51 |
| **SOL** | 54.85% | 361 | 50.0% | 57.3% | `$+90.26` | 157.20% | 3.19 | 9.71 | 5.59 | 1.30 |
| **SUI** | 61.87% | 299 | 53.8% | 66.3% | `$+174.75` | 186.08% | 6.29 | 15.63 | 9.15 | 1.82 |
| **TAO** | 55.00% | 260 | 47.7% | 65.4% | `$+83.73` | 258.39% | 2.83 | 6.10 | 3.16 | 1.33 |
| **TON** | 54.60% | 326 | 45.0% | 58.9% | `$+58.99` | 125.33% | 2.57 | 7.22 | 4.58 | 1.24 |
| **TRX** | 65.13% | 152 | 64.4% | 66.1% | `$+27.01` | 20.04% | 4.12 | 12.09 | 13.13 | 1.64 |
| **XRP** | 64.96% | 234 | 61.8% | 66.9% | `$+134.14` | 48.70% | 7.42 | 19.11 | 26.83 | 2.13 |

## ⛓️ Daftar Fitur Aktif dalam Model

Total terdapat **90 fitur aktif** yang digunakan oleh LightGBM entry, LSTM Soft Confirmation, dan Exit Guardian v3:

<details>
<summary>▶ Klik untuk melihat daftar lengkap fitur aktif</summary>

1. `dist_liq_50x_long`
2. `dist_liq_50x_short`
3. `dist_liq_20x_long`
4. `fear_greed`
5. `atr_percent_h4`
6. `dist_liq_20x_short`
7. `cvd_slope_h4`
8. `dist_from_8h_high`
9. `Buy_Liq`
10. `funding_rate`
11. `PWL`
12. `open_interest`
13. `Sell_Liq`
14. `PWH`
15. `dist_swing_low`
16. `dow_sin`
17. `atr_zscore_20d`
18. `PDL`
19. `ofi_h4_delta`
20. `log_ret_20`
21. `atr_percentile_h1`
22. `absorption_at_swing`
23. `PDH`
24. `cvd`
25. `dist_swing_high`
26. `ema_200_h1`
27. `whale_retail_divergence`
28. `log_ret_5`
29. `dow_cos`
30. `btc_dominance`
31. `trend_strength`
32. `bars_since_BOS`
33. `VAH`
34. `stochrsi_k`
35. `stochrsi_d`
36. `rsi_slope_h4`
37. `vwdp_smooth`
38. `POC`
39. `relative_strength_z`
40. `cvd_div_h4`
41. `VAL`
42. `cvd_momentum_adv`
43. `range_expansion_h4`
44. `ema_7_h1`
45. `hour_sin`
46. `trend_accel_4h`
47. `hour_cos`
48. `ema_21_slope_h4`
49. `swing_momentum`
50. `funding_price_div`
51. `atr_14_h1`
52. `rsi_h4`
53. `ema_50_slope_h4`
54. `rsi_6`
55. `price_in_range`
56. `relative_strength_momentum`
57. `vol_spike_zscore`
58. `atr_14_h4`
59. `absorption_z`
60. `ema_21_h1`
61. `Fib_786`
62. `time_to_funding_norm`
63. `log_ret_1`
64. `long_short_ratio`
65. `ofi_z_score`
66. `vol_ratio_20`
67. `spread_to_volume`
68. `Fib_618`
69. `ema_50_h1`
70. `effort_vs_result`
71. `open`
72. `vol_price_confirm`
73. `high`
74. `ofi_acceleration`
75. `vol_efficiency`
76. `vol_regime`
77. `ema_200_h4`
78. `low`
79. `buy_volume`
80. `close`
81. `price_vs_ema_50_h4`
82. `market_session`
83. `volume`
84. `volume_delta`
85. `vwdp`
86. `ema_7_h4`
87. `ofi_raw`
88. `price_accel_1h`
89. `ofi_momentum_ratio`
90. `vol_accel_3h`

</details>
