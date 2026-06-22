# 📊 Holdout Backtest Report: `cascade_v2.5_hybrid_accel_scenario_K`

**Tanggal Pembuatan**: 2026-06-02 20:10:48 UTC
**Model Run ID**: `cascade_v2.5_hybrid_accel_scenario_K`
**Periode Pengujian (Temporal OOS)**: `2025-11-01 - 2026-04-01`

> [!NOTE]
> **Ringkasan Portofolio Eksekutif**:
> *   **Total Net Profit**: **$+1,889.49 USD** (ROI Portofolio: **+89.98%**)
> *   **Rata-rata Win Rate**: **59.55%** | Total Trades: **6,251**
> *   **Rata-rata Max Drawdown (5x)**: **124.93%**
> *   **Risk-Adjusted**: Sharpe: **3.99** | Sortino: **10.50** | Calmar: **11.41** | Profit Factor: **1.62**

## 📈 Performa Scorecard Portofolio

| Metrik Utama | Nilai Portofolio | Catatan |
|:---|:---:|:---|
| **Total Net Profit ($)** | `$+1,889.49` | Akumulasi keuntungan bersih 5x leverage |
| **Portfolio ROI (%)** | `+89.98%` | ROI berdasarkan kapital portofolio $100/koin |
| **Overall Win Rate** | `59.55%` | Rasio kemenangan rata-rata seluruh aset |
| **Total Trades** | `6,251` | Jumlah total posisi yang dieksekusi |
| **Rata-rata Trade / Bulan** | `60.4` | Rata-rata frekuensi trade bulanan portofolio |
| **Rata-rata Trade / Hari** | `1.98` | Rata-rata frekuensi trade harian portofolio |
| **Max Drawdown (5x)** | `124.93%` | Rata-rata penurunan terdalam portofolio |
| **Sharpe Ratio** | `3.99` | Efisiensi profit terhadap volatilitas portofolio |
| **Sortino Ratio** | `10.50` | Efisiensi profit terhadap downside deviation |
| **Calmar Ratio** | `11.41` | Rasio return tahunan terhadap drawdown |
| **Profit Factor** | `1.62` | Rasio gross profit dibagi gross loss |
| **Max Consecutive Loss** | `13` trades | Streak kekalahan beruntun terpanjang |
| **Worst Single Trade PnL** | `-30.70%` | Kerugian terdalam dalam satu trade tunggal |
| **95% Trades Loss Under** | `11.00%` | Nilai risiko (VaR P95) kerugian maksimal |

## ↕️ Analisis Arah Signal (LONG vs SHORT)

| Arah Posisi | Jumlah Trade | Distribusi | Menang | Kalah | Win Rate | PnL Bersih ($) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **LONG** | 1,479 | 23.7% | 797 | 682 | 53.89% | +385.24 |
| **SHORT** | 4,772 | 76.3% | 2,824 | 1,948 | 59.18% | +1,504.25 |

### Rincian Rata-rata Profitabilitas per Trade

| Tipe Trade | PnL Rata-rata ($) | PnL Rata-rata (%) |
|:---|:---:|:---:|
| **Trade Menang (Wins)** | `$+1.8365` | `+7.35%` |
| **Trade Kalah (Losses)** | `$-1.8101` | `-7.24%` |

## 📅 Scorecard Bulanan Portofolio

| Bulan | Total Trades | Wins | Losses | Win Rate | Net PnL ($) |
|:---|:---:|:---:|:---:|:---:|:---:|
| 2025-11 | 1182 | 671 | 511 | 56.77% | $+332.31 |
| 2025-12 | 1288 | 754 | 534 | 58.54% | $+587.30 |
| 2026-01 | 1256 | 765 | 491 | 60.91% | $+608.26 |
| 2026-02 | 1222 | 671 | 551 | 54.91% | $+46.09 |
| 2026-03 | 1303 | 760 | 543 | 58.33% | $+315.53 |

## 🚪 Distribusi Alasan Keluar Posisi (Exit Reasons)

| Alasan Exit | Jumlah | Persentase | Wins | Losses | Win Rate | PnL Bersih ($) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| `guardian_exit` | 4,736 | 75.8% | 3,534 | 1,202 | 74.62% | $+4,587.31 |
| `sl_hit` | 1,409 | 22.5% | 2 | 1,407 | 0.14% | $-2,732.25 |
| `time_exit` | 106 | 1.7% | 85 | 21 | 80.19% | $+34.43 |

## 🪙 Scorecard Per Koin (Detailed Assets)

| Token | Win Rate | Trades | LONG WR | SHORT WR | Net PnL ($) | Max DD | Sharpe | Sortino | Calmar | PF |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1000PEPE** | 63.16% | 133 | 66.7% | 62.6% | `$+74.38` | 134.47% | 3.49 | 7.54 | 5.39 | 1.61 |
| **1000SHIB** | 62.26% | 106 | 64.5% | 61.3% | `$+61.01` | 71.45% | 4.72 | 12.67 | 8.32 | 2.06 |
| **ADA** | 63.03% | 238 | 58.3% | 64.6% | `$+141.36` | 64.16% | 5.71 | 17.18 | 21.46 | 1.81 |
| **ARB** | 58.68% | 288 | 52.5% | 62.0% | `$+121.28` | 123.66% | 4.45 | 10.83 | 9.55 | 1.52 |
| **AVAX** | 53.18% | 472 | 53.7% | 53.0% | `$+62.98` | 339.88% | 2.11 | 5.18 | 1.80 | 1.16 |
| **BNB** | 51.45% | 311 | 46.0% | 53.6% | `$-20.61` | 129.17% | -1.14 | -2.32 | -1.55 | 0.91 |
| **BTC** | 77.01% | 87 | 92.3% | 74.3% | `$+70.44` | 23.30% | 7.75 | 21.76 | 29.45 | 3.50 |
| **DOGE** | 65.29% | 121 | 72.0% | 63.5% | `$+116.43` | 33.95% | 6.62 | 16.07 | 33.40 | 2.77 |
| **DOT** | 56.84% | 424 | 51.1% | 58.4% | `$+89.07` | 116.84% | 3.08 | 6.51 | 7.42 | 1.27 |
| **ETH** | 65.12% | 215 | 59.5% | 66.5% | `$+145.57` | 55.70% | 7.05 | 20.02 | 25.45 | 2.14 |
| **HBAR** | 60.38% | 260 | 54.3% | 62.6% | `$+80.94` | 55.50% | 3.88 | 10.65 | 14.21 | 1.43 |
| **LINK** | 56.61% | 431 | 58.4% | 56.2% | `$+133.95` | 142.13% | 4.44 | 12.77 | 9.18 | 1.39 |
| **NEAR** | 59.66% | 414 | 42.4% | 64.6% | `$+207.33` | 136.01% | 6.06 | 14.47 | 14.85 | 1.64 |
| **ONDO** | 56.74% | 356 | 57.5% | 56.5% | `$+122.46` | 151.96% | 4.08 | 11.00 | 7.85 | 1.40 |
| **POL** | 53.65% | 315 | 50.6% | 54.8% | `$+59.55` | 167.71% | 2.28 | 6.20 | 3.46 | 1.22 |
| **SOL** | 52.86% | 454 | 50.0% | 53.5% | `$+22.06` | 300.87% | 0.72 | 1.65 | 0.71 | 1.06 |
| **SUI** | 63.82% | 387 | 59.0% | 65.1% | `$+222.31` | 85.20% | 7.12 | 17.47 | 25.41 | 1.80 |
| **TAO** | 51.25% | 281 | 43.0% | 57.5% | `$+3.03` | 214.74% | 0.10 | 0.20 | 0.14 | 1.01 |
| **TON** | 52.06% | 413 | 44.1% | 53.6% | `$+32.24` | 147.37% | 1.23 | 3.66 | 2.13 | 1.10 |
| **TRX** | 64.02% | 239 | 69.2% | 61.5% | `$+35.45` | 35.07% | 4.53 | 11.90 | 9.85 | 1.55 |
| **XRP** | 63.40% | 306 | 60.0% | 64.5% | `$+108.24` | 94.37% | 5.62 | 15.09 | 11.17 | 1.63 |

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
