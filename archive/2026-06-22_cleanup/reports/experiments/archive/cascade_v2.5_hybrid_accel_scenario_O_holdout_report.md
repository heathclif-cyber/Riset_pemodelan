# 📊 Holdout Backtest Report: `cascade_v2.5_hybrid_accel_scenario_O`

**Tanggal Pembuatan**: 2026-06-02 20:30:09 UTC
**Model Run ID**: `cascade_v2.5_hybrid_accel_scenario_O`
**Periode Pengujian (Temporal OOS)**: `2025-11-01 - 2026-04-01`

> [!NOTE]
> **Ringkasan Portofolio Eksekutif**:
> *   **Total Net Profit**: **$+1,875.13 USD** (ROI Portofolio: **+89.29%**)
> *   **Rata-rata Win Rate**: **59.57%** | Total Trades: **6,299**
> *   **Rata-rata Max Drawdown (5x)**: **125.25%**
> *   **Risk-Adjusted**: Sharpe: **3.90** | Sortino: **10.20** | Calmar: **10.46** | Profit Factor: **1.77**

## 📈 Performa Scorecard Portofolio

| Metrik Utama | Nilai Portofolio | Catatan |
|:---|:---:|:---|
| **Total Net Profit ($)** | `$+1,875.13` | Akumulasi keuntungan bersih 5x leverage |
| **Portfolio ROI (%)** | `+89.29%` | ROI berdasarkan kapital portofolio $100/koin |
| **Overall Win Rate** | `59.57%` | Rasio kemenangan rata-rata seluruh aset |
| **Total Trades** | `6,299` | Jumlah total posisi yang dieksekusi |
| **Rata-rata Trade / Bulan** | `60.9` | Rata-rata frekuensi trade bulanan portofolio |
| **Rata-rata Trade / Hari** | `2.00` | Rata-rata frekuensi trade harian portofolio |
| **Max Drawdown (5x)** | `125.25%` | Rata-rata penurunan terdalam portofolio |
| **Sharpe Ratio** | `3.90` | Efisiensi profit terhadap volatilitas portofolio |
| **Sortino Ratio** | `10.20` | Efisiensi profit terhadap downside deviation |
| **Calmar Ratio** | `10.46` | Rasio return tahunan terhadap drawdown |
| **Profit Factor** | `1.77` | Rasio gross profit dibagi gross loss |
| **Max Consecutive Loss** | `19` trades | Streak kekalahan beruntun terpanjang |
| **Worst Single Trade PnL** | `-25.90%` | Kerugian terdalam dalam satu trade tunggal |
| **95% Trades Loss Under** | `11.30%` | Nilai risiko (VaR P95) kerugian maksimal |

## ↕️ Analisis Arah Signal (LONG vs SHORT)

| Arah Posisi | Jumlah Trade | Distribusi | Menang | Kalah | Win Rate | PnL Bersih ($) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **LONG** | 2,667 | 42.3% | 1,377 | 1,290 | 51.63% | +255.93 |
| **SHORT** | 3,632 | 57.7% | 2,218 | 1,414 | 61.07% | +1,619.21 |

### Rincian Rata-rata Profitabilitas per Trade

| Tipe Trade | PnL Rata-rata ($) | PnL Rata-rata (%) |
|:---|:---:|:---:|
| **Trade Menang (Wins)** | `$+1.8887` | `+7.55%` |
| **Trade Kalah (Losses)** | `$-1.8175` | `-7.27%` |

## 📅 Scorecard Bulanan Portofolio

| Bulan | Total Trades | Wins | Losses | Win Rate | Net PnL ($) |
|:---|:---:|:---:|:---:|:---:|:---:|
| 2025-11 | 1235 | 682 | 553 | 55.22% | $+266.25 |
| 2025-12 | 1273 | 759 | 514 | 59.62% | $+621.32 |
| 2026-01 | 1237 | 763 | 474 | 61.68% | $+662.73 |
| 2026-02 | 1201 | 593 | 608 | 49.38% | $-105.61 |
| 2026-03 | 1353 | 798 | 555 | 58.98% | $+430.44 |

## 🚪 Distribusi Alasan Keluar Posisi (Exit Reasons)

| Alasan Exit | Jumlah | Persentase | Wins | Losses | Win Rate | PnL Bersih ($) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| `guardian_exit` | 4,733 | 75.1% | 3,516 | 1,217 | 74.29% | $+4,743.90 |
| `sl_hit` | 1,472 | 23.4% | 2 | 1,470 | 0.14% | $-2,902.17 |
| `time_exit` | 94 | 1.5% | 77 | 17 | 81.91% | $+33.41 |

## 🪙 Scorecard Per Koin (Detailed Assets)

| Token | Win Rate | Trades | LONG WR | SHORT WR | Net PnL ($) | Max DD | Sharpe | Sortino | Calmar | PF |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1000PEPE** | 73.13% | 67 | 64.3% | 79.5% | `$+86.08` | 38.87% | 5.90 | 15.94 | 21.57 | 3.19 |
| **1000SHIB** | 65.45% | 55 | 55.2% | 76.9% | `$+27.62` | 22.51% | 3.20 | 6.12 | 11.95 | 1.96 |
| **ADA** | 64.11% | 248 | 56.5% | 70.0% | `$+171.76` | 51.76% | 7.09 | 22.16 | 32.32 | 2.04 |
| **ARB** | 53.31% | 347 | 46.8% | 59.6% | `$+97.88` | 135.11% | 3.15 | 8.22 | 7.06 | 1.31 |
| **AVAX** | 55.81% | 482 | 53.0% | 57.5% | `$+111.90` | 254.71% | 3.59 | 8.89 | 4.28 | 1.29 |
| **BNB** | 53.04% | 296 | 45.7% | 58.7% | `$-4.40` | 109.91% | -0.25 | -0.70 | -0.39 | 0.98 |
| **BTC** | 80.95% | 21 | 81.8% | 80.0% | `$+26.45` | 11.27% | 5.71 | 12.24 | 22.86 | 5.99 |
| **DOGE** | 61.11% | 90 | 58.5% | 63.3% | `$+80.40` | 66.55% | 4.89 | 13.45 | 11.77 | 2.38 |
| **DOT** | 57.21% | 409 | 45.8% | 64.2% | `$+85.13` | 119.62% | 2.92 | 6.80 | 6.93 | 1.26 |
| **ETH** | 61.05% | 285 | 55.8% | 65.4% | `$+121.80` | 176.52% | 5.00 | 13.01 | 6.72 | 1.60 |
| **HBAR** | 56.46% | 294 | 48.6% | 64.0% | `$+46.17` | 70.14% | 2.13 | 5.84 | 6.41 | 1.20 |
| **LINK** | 56.05% | 430 | 54.3% | 57.1% | `$+113.52` | 161.23% | 3.70 | 11.41 | 6.86 | 1.31 |
| **NEAR** | 57.31% | 417 | 44.7% | 66.0% | `$+173.80` | 139.52% | 5.05 | 11.84 | 12.13 | 1.52 |
| **ONDO** | 57.71% | 350 | 57.8% | 57.7% | `$+145.84` | 170.48% | 4.72 | 12.79 | 8.33 | 1.48 |
| **POL** | 54.62% | 346 | 53.0% | 56.0% | `$+101.02` | 149.54% | 3.62 | 10.06 | 6.58 | 1.35 |
| **SOL** | 53.39% | 442 | 47.7% | 56.5% | `$+71.75` | 221.51% | 2.32 | 6.42 | 3.15 | 1.19 |
| **SUI** | 61.04% | 385 | 55.7% | 64.4% | `$+190.44` | 146.18% | 6.09 | 15.05 | 12.69 | 1.67 |
| **TAO** | 52.54% | 354 | 46.7% | 59.9% | `$+44.46` | 319.27% | 1.28 | 2.87 | 1.36 | 1.12 |
| **TON** | 50.23% | 428 | 40.9% | 55.2% | `$+17.49` | 181.56% | 0.67 | 1.84 | 0.94 | 1.05 |
| **TRX** | 64.29% | 238 | 66.4% | 61.2% | `$+38.44` | 35.39% | 4.98 | 14.72 | 10.58 | 1.61 |
| **XRP** | 62.22% | 315 | 57.4% | 65.6% | `$+127.58` | 48.70% | 6.14 | 15.33 | 25.52 | 1.71 |

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
