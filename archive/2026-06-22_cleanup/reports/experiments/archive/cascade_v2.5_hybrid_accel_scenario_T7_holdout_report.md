# 📊 Holdout Backtest Report: `cascade_v2.5_hybrid_accel_scenario_T7`

**Tanggal Pembuatan**: 2026-06-02 21:18:23 UTC
**Model Run ID**: `cascade_v2.5_hybrid_accel_scenario_T7`
**Periode Pengujian (Temporal OOS)**: `2025-11-01 - 2026-04-01`

> [!NOTE]
> **Ringkasan Portofolio Eksekutif**:
> *   **Total Net Profit**: **$+1,353.02 USD** (ROI Portofolio: **+64.43%**)
> *   **Rata-rata Win Rate**: **61.92%** | Total Trades: **2,594**
> *   **Rata-rata Max Drawdown (5x)**: **71.38%**
> *   **Risk-Adjusted**: Sharpe: **3.63** | Sortino: **10.22** | Calmar: **9.74** | Profit Factor: **1.57**

## 📈 Performa Scorecard Portofolio

| Metrik Utama | Nilai Portofolio | Catatan |
|:---|:---:|:---|
| **Total Net Profit ($)** | `$+1,353.02` | Akumulasi keuntungan bersih 5x leverage |
| **Portfolio ROI (%)** | `+64.43%` | ROI berdasarkan kapital portofolio $100/koin |
| **Overall Win Rate** | `61.92%` | Rasio kemenangan rata-rata seluruh aset |
| **Total Trades** | `2,594` | Jumlah total posisi yang dieksekusi |
| **Rata-rata Trade / Bulan** | `25.1` | Rata-rata frekuensi trade bulanan portofolio |
| **Rata-rata Trade / Hari** | `0.82` | Rata-rata frekuensi trade harian portofolio |
| **Max Drawdown (5x)** | `71.38%` | Rata-rata penurunan terdalam portofolio |
| **Sharpe Ratio** | `3.63` | Efisiensi profit terhadap volatilitas portofolio |
| **Sortino Ratio** | `10.22` | Efisiensi profit terhadap downside deviation |
| **Calmar Ratio** | `9.74` | Rasio return tahunan terhadap drawdown |
| **Profit Factor** | `1.57` | Rasio gross profit dibagi gross loss |
| **Max Consecutive Loss** | `11` trades | Streak kekalahan beruntun terpanjang |
| **Worst Single Trade PnL** | `-24.90%` | Kerugian terdalam dalam satu trade tunggal |
| **95% Trades Loss Under** | `11.13%` | Nilai risiko (VaR P95) kerugian maksimal |

## ↕️ Analisis Arah Signal (LONG vs SHORT)

| Arah Posisi | Jumlah Trade | Distribusi | Menang | Kalah | Win Rate | PnL Bersih ($) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **LONG** | 1,230 | 47.4% | 655 | 575 | 53.25% | +291.81 |
| **SHORT** | 1,364 | 52.6% | 903 | 461 | 66.20% | +1,061.21 |

### Rincian Rata-rata Profitabilitas per Trade

| Tipe Trade | PnL Rata-rata ($) | PnL Rata-rata (%) |
|:---|:---:|:---:|
| **Trade Menang (Wins)** | `$+2.0698` | `+8.28%` |
| **Trade Kalah (Losses)** | `$-1.8067` | `-7.23%` |

## 📅 Scorecard Bulanan Portofolio

| Bulan | Total Trades | Wins | Losses | Win Rate | Net PnL ($) |
|:---|:---:|:---:|:---:|:---:|:---:|
| 2025-11 | 514 | 296 | 218 | 57.59% | $+241.64 |
| 2025-12 | 524 | 344 | 180 | 65.65% | $+457.58 |
| 2026-01 | 477 | 309 | 168 | 64.78% | $+384.71 |
| 2026-02 | 505 | 254 | 251 | 50.30% | $-38.76 |
| 2026-03 | 574 | 355 | 219 | 61.85% | $+307.86 |

## 🚪 Distribusi Alasan Keluar Posisi (Exit Reasons)

| Alasan Exit | Jumlah | Persentase | Wins | Losses | Win Rate | PnL Bersih ($) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| `guardian_exit` | 2,013 | 77.6% | 1,528 | 485 | 75.91% | $+2,420.17 |
| `sl_hit` | 547 | 21.1% | 0 | 547 | 0.00% | $-1,079.55 |
| `time_exit` | 34 | 1.3% | 30 | 4 | 88.24% | $+12.40 |

## 🪙 Scorecard Per Koin (Detailed Assets)

| Token | Win Rate | Trades | LONG WR | SHORT WR | Net PnL ($) | Max DD | Sharpe | Sortino | Calmar | PF |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1000PEPE** | 100.00% | 1 | 0.0% | 100.0% | `$+3.80` | 0.00% | 0.00 | 0.00 | 0.00 | 0.00 |
| **1000SHIB** | 100.00% | 2 | 0.0% | 100.0% | `$+6.07` | 0.00% | 2.65 | 0.00 | 0.00 | 0.00 |
| **ADA** | 64.36% | 101 | 60.0% | 67.9% | `$+96.96` | 36.80% | 5.66 | 16.67 | 25.66 | 2.52 |
| **ARB** | 52.67% | 150 | 50.0% | 56.2% | `$+54.52` | 132.39% | 2.41 | 6.42 | 4.01 | 1.38 |
| **AVAX** | 60.54% | 223 | 55.2% | 64.6% | `$+117.64` | 74.72% | 5.13 | 13.30 | 15.33 | 1.72 |
| **BNB** | 57.38% | 122 | 46.6% | 67.2% | `$+17.83` | 44.24% | 1.56 | 5.06 | 3.92 | 1.22 |
| **BTC** | 0.00% | 0 | 0.0% | 0.0% | `$+0.00` | 0.00% | 0.00 | 0.00 | 0.00 | 0.00 |
| **DOGE** | 75.00% | 8 | 33.3% | 100.0% | `$+5.74` | 13.85% | 1.76 | 14.93 | 4.03 | 2.66 |
| **DOT** | 59.76% | 169 | 45.7% | 72.7% | `$+80.58` | 77.48% | 4.04 | 9.63 | 10.13 | 1.66 |
| **ETH** | 68.70% | 115 | 64.4% | 71.4% | `$+102.20` | 30.76% | 6.78 | 21.57 | 32.36 | 2.70 |
| **HBAR** | 58.57% | 140 | 51.2% | 68.3% | `$+29.58` | 44.71% | 1.92 | 5.78 | 6.44 | 1.27 |
| **LINK** | 68.13% | 182 | 62.4% | 73.2% | `$+154.54` | 58.91% | 7.63 | 23.40 | 25.55 | 2.41 |
| **NEAR** | 57.14% | 161 | 39.7% | 69.9% | `$+103.54` | 60.17% | 4.81 | 12.24 | 16.76 | 1.84 |
| **ONDO** | 60.69% | 145 | 56.9% | 63.2% | `$+95.45` | 150.06% | 4.65 | 12.61 | 6.20 | 1.87 |
| **POL** | 55.33% | 150 | 51.2% | 60.0% | `$+79.13` | 143.82% | 4.05 | 12.07 | 5.36 | 1.67 |
| **SOL** | 52.74% | 201 | 48.1% | 55.8% | `$+57.68` | 114.55% | 2.58 | 8.19 | 4.90 | 1.33 |
| **SUI** | 64.04% | 178 | 54.9% | 70.1% | `$+127.64` | 172.07% | 5.48 | 13.90 | 7.23 | 2.00 |
| **TAO** | 55.56% | 162 | 51.8% | 63.0% | `$+64.97` | 182.62% | 2.69 | 5.95 | 3.46 | 1.42 |
| **TON** | 59.21% | 152 | 46.9% | 68.2% | `$+63.08` | 103.25% | 3.78 | 10.69 | 5.95 | 1.59 |
| **TRX** | 66.30% | 92 | 62.9% | 73.3% | `$+16.55` | 21.32% | 3.30 | 9.36 | 7.56 | 1.68 |
| **XRP** | 64.29% | 140 | 67.8% | 61.7% | `$+75.53` | 37.29% | 5.31 | 12.84 | 19.73 | 2.02 |

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
