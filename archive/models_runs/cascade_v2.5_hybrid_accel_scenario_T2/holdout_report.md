# 📊 Holdout Backtest Report: `cascade_v2.5_hybrid_accel_scenario_T2`

**Tanggal Pembuatan**: 2026-06-02 21:12:11 UTC
**Model Run ID**: `cascade_v2.5_hybrid_accel_scenario_T2`
**Periode Pengujian (Temporal OOS)**: `2025-11-01 - 2026-04-01`

> [!NOTE]
> **Ringkasan Portofolio Eksekutif**:
> *   **Total Net Profit**: **$+1,566.72 USD** (ROI Portofolio: **+74.61%**)
> *   **Rata-rata Win Rate**: **60.62%** | Total Trades: **3,152**
> *   **Rata-rata Max Drawdown (5x)**: **82.22%**
> *   **Risk-Adjusted**: Sharpe: **4.07** | Sortino: **11.91** | Calmar: **24.82** | Profit Factor: **7.53**

## 📈 Performa Scorecard Portofolio

| Metrik Utama | Nilai Portofolio | Catatan |
|:---|:---:|:---|
| **Total Net Profit ($)** | `$+1,566.72` | Akumulasi keuntungan bersih 5x leverage |
| **Portfolio ROI (%)** | `+74.61%` | ROI berdasarkan kapital portofolio $100/koin |
| **Overall Win Rate** | `60.62%` | Rasio kemenangan rata-rata seluruh aset |
| **Total Trades** | `3,152` | Jumlah total posisi yang dieksekusi |
| **Rata-rata Trade / Bulan** | `30.5` | Rata-rata frekuensi trade bulanan portofolio |
| **Rata-rata Trade / Hari** | `1.00` | Rata-rata frekuensi trade harian portofolio |
| **Max Drawdown (5x)** | `82.22%` | Rata-rata penurunan terdalam portofolio |
| **Sharpe Ratio** | `4.07` | Efisiensi profit terhadap volatilitas portofolio |
| **Sortino Ratio** | `11.91` | Efisiensi profit terhadap downside deviation |
| **Calmar Ratio** | `24.82` | Rasio return tahunan terhadap drawdown |
| **Profit Factor** | `7.53` | Rasio gross profit dibagi gross loss |
| **Max Consecutive Loss** | `11` trades | Streak kekalahan beruntun terpanjang |
| **Worst Single Trade PnL** | `-24.90%` | Kerugian terdalam dalam satu trade tunggal |
| **95% Trades Loss Under** | `11.20%` | Nilai risiko (VaR P95) kerugian maksimal |

## ↕️ Analisis Arah Signal (LONG vs SHORT)

| Arah Posisi | Jumlah Trade | Distribusi | Menang | Kalah | Win Rate | PnL Bersih ($) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **LONG** | 1,499 | 47.6% | 801 | 698 | 53.44% | +347.76 |
| **SHORT** | 1,653 | 52.4% | 1,087 | 566 | 65.76% | +1,218.96 |

### Rincian Rata-rata Profitabilitas per Trade

| Tipe Trade | PnL Rata-rata ($) | PnL Rata-rata (%) |
|:---|:---:|:---:|
| **Trade Menang (Wins)** | `$+2.0398` | `+8.16%` |
| **Trade Kalah (Losses)** | `$-1.8072` | `-7.23%` |

## 📅 Scorecard Bulanan Portofolio

| Bulan | Total Trades | Wins | Losses | Win Rate | Net PnL ($) |
|:---|:---:|:---:|:---:|:---:|:---:|
| 2025-11 | 605 | 342 | 263 | 56.53% | $+232.65 |
| 2025-12 | 626 | 412 | 214 | 65.81% | $+522.65 |
| 2026-01 | 610 | 389 | 221 | 63.77% | $+451.13 |
| 2026-02 | 613 | 313 | 300 | 51.06% | $-4.87 |
| 2026-03 | 698 | 432 | 266 | 61.89% | $+365.17 |

## 🚪 Distribusi Alasan Keluar Posisi (Exit Reasons)

| Alasan Exit | Jumlah | Persentase | Wins | Losses | Win Rate | PnL Bersih ($) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| `guardian_exit` | 2,448 | 77.7% | 1,848 | 600 | 75.49% | $+2,844.31 |
| `sl_hit` | 658 | 20.9% | 1 | 657 | 0.15% | $-1,294.50 |
| `time_exit` | 46 | 1.5% | 39 | 7 | 84.78% | $+16.91 |

## 🪙 Scorecard Per Koin (Detailed Assets)

| Token | Win Rate | Trades | LONG WR | SHORT WR | Net PnL ($) | Max DD | Sharpe | Sortino | Calmar | PF |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1000PEPE** | 100.00% | 3 | 100.0% | 100.0% | `$+9.03` | 0.00% | 3.51 | 0.00 | 0.00 | 0.00 |
| **1000SHIB** | 66.67% | 3 | 0.0% | 66.7% | `$+6.02` | 0.20% | 2.15 | 0.00 | 293.38 | 121.97 |
| **ADA** | 65.08% | 126 | 60.0% | 69.7% | `$+118.41` | 42.32% | 6.34 | 18.91 | 27.25 | 2.50 |
| **ARB** | 54.10% | 183 | 50.5% | 59.0% | `$+80.09` | 110.04% | 3.24 | 8.32 | 7.09 | 1.48 |
| **AVAX** | 58.65% | 266 | 52.6% | 63.2% | `$+96.44` | 109.46% | 3.94 | 9.61 | 8.58 | 1.46 |
| **BNB** | 56.25% | 144 | 47.8% | 64.0% | `$+19.58` | 48.46% | 1.60 | 4.97 | 3.94 | 1.21 |
| **BTC** | 0.00% | 0 | 0.0% | 0.0% | `$+0.00` | 0.00% | 0.00 | 0.00 | 0.00 | 0.00 |
| **DOGE** | 83.33% | 12 | 33.3% | 100.0% | `$+18.86` | 13.85% | 3.76 | 40.10 | 13.27 | 6.45 |
| **DOT** | 57.42% | 209 | 48.5% | 65.5% | `$+66.73` | 117.81% | 3.09 | 7.37 | 5.52 | 1.41 |
| **ETH** | 70.00% | 140 | 66.1% | 72.8% | `$+122.29` | 35.74% | 7.49 | 23.80 | 33.33 | 2.71 |
| **HBAR** | 58.48% | 171 | 50.0% | 68.3% | `$+31.11` | 59.27% | 1.87 | 5.36 | 5.11 | 1.23 |
| **LINK** | 68.06% | 216 | 60.8% | 74.6% | `$+169.58` | 60.59% | 7.85 | 24.18 | 27.26 | 2.26 |
| **NEAR** | 57.14% | 203 | 41.1% | 69.9% | `$+112.44` | 62.81% | 4.75 | 11.79 | 17.44 | 1.71 |
| **ONDO** | 61.75% | 183 | 59.7% | 63.2% | `$+126.78` | 157.30% | 5.63 | 15.23 | 7.85 | 1.94 |
| **POL** | 54.14% | 181 | 49.5% | 59.3% | `$+73.92` | 190.01% | 3.47 | 9.66 | 3.79 | 1.49 |
| **SOL** | 54.32% | 243 | 49.5% | 57.5% | `$+83.24` | 155.25% | 3.32 | 10.98 | 5.22 | 1.40 |
| **SUI** | 63.33% | 210 | 55.2% | 69.1% | `$+146.21` | 190.29% | 5.92 | 15.50 | 7.48 | 1.98 |
| **TAO** | 56.32% | 190 | 51.6% | 65.1% | `$+91.89` | 203.82% | 3.52 | 7.95 | 4.39 | 1.52 |
| **TON** | 56.12% | 196 | 45.6% | 63.2% | `$+66.54` | 115.19% | 3.50 | 9.57 | 5.63 | 1.46 |
| **TRX** | 66.02% | 103 | 63.4% | 71.9% | `$+18.56` | 26.16% | 3.49 | 10.01 | 6.91 | 1.67 |
| **XRP** | 65.88% | 170 | 68.0% | 64.2% | `$+108.99` | 28.03% | 6.97 | 16.78 | 37.87 | 2.31 |

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
