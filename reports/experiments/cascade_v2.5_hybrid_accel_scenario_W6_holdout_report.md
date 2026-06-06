# 📊 Holdout Backtest Report: `cascade_v2.5_hybrid_accel_scenario_W6`

**Tanggal Pembuatan**: 2026-06-03 21:21:00 UTC
**Model Run ID**: `cascade_v2.5_hybrid_accel_scenario_W6`
**Periode Pengujian (Temporal OOS)**: `2025-11-01 - 2026-04-01`

> [!NOTE]
> **Ringkasan Portofolio Eksekutif**:
> *   **Total Net Profit**: **$+219.74 USD** (ROI Portofolio: **+10.46%**)
> *   **Rata-rata Win Rate**: **50.32%** | Total Trades: **8,435**
> *   **Rata-rata Max Drawdown (5x)**: **213.22%**
> *   **Risk-Adjusted**: Sharpe: **0.60** | Sortino: **1.66** | Calmar: **1.48** | Profit Factor: **1.12**

## 📈 Performa Scorecard Portofolio

| Metrik Utama | Nilai Portofolio | Catatan |
|:---|:---:|:---|
| **Total Net Profit ($)** | `$+219.74` | Akumulasi keuntungan bersih 5x leverage |
| **Portfolio ROI (%)** | `+10.46%` | ROI berdasarkan kapital portofolio $100/koin |
| **Overall Win Rate** | `50.32%` | Rasio kemenangan rata-rata seluruh aset |
| **Total Trades** | `8,435` | Jumlah total posisi yang dieksekusi |
| **Rata-rata Trade / Bulan** | `81.5` | Rata-rata frekuensi trade bulanan portofolio |
| **Rata-rata Trade / Hari** | `2.68` | Rata-rata frekuensi trade harian portofolio |
| **Max Drawdown (5x)** | `213.22%` | Rata-rata penurunan terdalam portofolio |
| **Sharpe Ratio** | `0.60` | Efisiensi profit terhadap volatilitas portofolio |
| **Sortino Ratio** | `1.66` | Efisiensi profit terhadap downside deviation |
| **Calmar Ratio** | `1.48` | Rasio return tahunan terhadap drawdown |
| **Profit Factor** | `1.12` | Rasio gross profit dibagi gross loss |
| **Max Consecutive Loss** | `23` trades | Streak kekalahan beruntun terpanjang |
| **Worst Single Trade PnL** | `-25.90%` | Kerugian terdalam dalam satu trade tunggal |
| **95% Trades Loss Under** | `11.10%` | Nilai risiko (VaR P95) kerugian maksimal |

## ↕️ Analisis Arah Signal (LONG vs SHORT)

| Arah Posisi | Jumlah Trade | Distribusi | Menang | Kalah | Win Rate | PnL Bersih ($) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **LONG** | 3,777 | 44.8% | 1,631 | 2,146 | 43.18% | -977.20 |
| **SHORT** | 4,658 | 55.2% | 2,667 | 1,991 | 57.26% | +1,196.94 |

### Rincian Rata-rata Profitabilitas per Trade

| Tipe Trade | PnL Rata-rata ($) | PnL Rata-rata (%) |
|:---|:---:|:---:|
| **Trade Menang (Wins)** | `$+1.7544` | `+7.02%` |
| **Trade Kalah (Losses)** | `$-1.7695` | `-7.08%` |

## 📅 Scorecard Bulanan Portofolio

| Bulan | Total Trades | Wins | Losses | Win Rate | Net PnL ($) |
|:---|:---:|:---:|:---:|:---:|:---:|
| 2025-11 | 1390 | 733 | 657 | 52.73% | $+83.43 |
| 2025-12 | 1751 | 895 | 856 | 51.11% | $+231.27 |
| 2026-01 | 1750 | 930 | 820 | 53.14% | $+159.42 |
| 2026-02 | 1649 | 763 | 886 | 46.27% | $-275.75 |
| 2026-03 | 1895 | 977 | 918 | 51.56% | $+21.36 |

## 🚪 Distribusi Alasan Keluar Posisi (Exit Reasons)

| Alasan Exit | Jumlah | Persentase | Wins | Losses | Win Rate | PnL Bersih ($) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| `guardian_exit` | 5,774 | 68.5% | 4,185 | 1,589 | 72.48% | $+4,972.11 |
| `sl_hit` | 2,530 | 30.0% | 2 | 2,528 | 0.08% | $-4,798.60 |
| `time_exit` | 131 | 1.6% | 111 | 20 | 84.73% | $+46.24 |

## 🪙 Scorecard Per Koin (Detailed Assets)

| Token | Win Rate | Trades | LONG WR | SHORT WR | Net PnL ($) | Max DD | Sharpe | Sortino | Calmar | PF |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1000PEPE** | 58.33% | 12 | 0.0% | 58.3% | `$+11.13` | 17.85% | 1.61 | 6.09 | 6.07 | 2.41 |
| **1000SHIB** | 66.67% | 30 | 28.6% | 78.3% | `$+13.99` | 21.53% | 2.26 | 5.71 | 6.33 | 1.85 |
| **ADA** | 50.61% | 409 | 37.2% | 62.0% | `$+19.53` | 309.26% | 0.67 | 2.01 | 0.62 | 1.05 |
| **ARB** | 47.84% | 464 | 41.5% | 53.8% | `$-2.22` | 241.27% | -0.07 | -0.19 | -0.09 | 1.00 |
| **AVAX** | 50.99% | 553 | 42.3% | 57.1% | `$+44.20` | 265.30% | 1.42 | 3.92 | 1.62 | 1.10 |
| **BNB** | 48.08% | 547 | 42.3% | 52.8% | `$-69.94` | 343.68% | -3.08 | -8.15 | -1.98 | 0.82 |
| **BTC** | 0.00% | 0 | 0.0% | 0.0% | `$+0.00` | 0.00% | 0.00 | 0.00 | 0.00 | 0.00 |
| **DOGE** | 64.00% | 50 | 40.0% | 70.0% | `$+14.04` | 55.13% | 1.95 | 4.19 | 2.48 | 1.53 |
| **DOT** | 50.18% | 544 | 39.9% | 58.8% | `$-33.55` | 315.42% | -0.99 | -2.53 | -1.04 | 0.94 |
| **ETH** | 54.69% | 437 | 41.2% | 60.8% | `$+76.74` | 210.96% | 2.83 | 7.27 | 3.54 | 1.23 |
| **HBAR** | 48.45% | 419 | 41.3% | 57.1% | `$-29.31` | 165.86% | -1.17 | -3.15 | -1.72 | 0.92 |
| **LINK** | 52.05% | 584 | 45.5% | 56.8% | `$+57.93` | 144.80% | 1.74 | 5.05 | 3.90 | 1.12 |
| **NEAR** | 53.07% | 522 | 38.4% | 65.7% | `$+97.86` | 231.73% | 2.85 | 7.09 | 4.11 | 1.22 |
| **ONDO** | 52.61% | 460 | 48.5% | 55.7% | `$+67.05` | 226.91% | 2.05 | 5.84 | 2.88 | 1.16 |
| **POL** | 49.23% | 453 | 46.0% | 51.8% | `$+37.38` | 233.70% | 1.19 | 3.38 | 1.56 | 1.09 |
| **SOL** | 47.08% | 565 | 40.0% | 51.8% | `$-61.22` | 458.46% | -1.91 | -5.25 | -1.30 | 0.88 |
| **SUI** | 53.82% | 511 | 46.1% | 60.1% | `$+60.16` | 306.22% | 1.69 | 4.62 | 1.91 | 1.12 |
| **TAO** | 49.55% | 555 | 43.5% | 57.0% | `$-58.80` | 295.71% | -1.49 | -3.39 | -1.94 | 0.91 |
| **TON** | 46.27% | 549 | 43.4% | 48.3% | `$-97.66` | 409.91% | -3.49 | -9.77 | -2.32 | 0.81 |
| **TRX** | 59.26% | 297 | 53.0% | 67.4% | `$+21.59` | 60.60% | 2.54 | 7.11 | 3.47 | 1.24 |
| **XRP** | 54.01% | 474 | 46.8% | 60.2% | `$+50.84` | 163.40% | 2.00 | 5.00 | 3.03 | 1.15 |

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
