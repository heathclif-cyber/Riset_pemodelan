# 📊 Holdout Backtest Report: `cascade_v2.5_hybrid_accel_scenario_V6`

**Tanggal Pembuatan**: 2026-06-03 20:48:43 UTC
**Model Run ID**: `cascade_v2.5_hybrid_accel_scenario_V6`
**Periode Pengujian (Temporal OOS)**: `2025-11-01 - 2026-04-01`

> [!NOTE]
> **Ringkasan Portofolio Eksekutif**:
> *   **Total Net Profit**: **$+1,731.16 USD** (ROI Portofolio: **+82.44%**)
> *   **Rata-rata Win Rate**: **59.55%** | Total Trades: **4,210**
> *   **Rata-rata Max Drawdown (5x)**: **93.58%**
> *   **Risk-Adjusted**: Sharpe: **4.11** | Sortino: **11.10** | Calmar: **12.07** | Profit Factor: **2.60**

## 📈 Performa Scorecard Portofolio

| Metrik Utama | Nilai Portofolio | Catatan |
|:---|:---:|:---|
| **Total Net Profit ($)** | `$+1,731.16` | Akumulasi keuntungan bersih 5x leverage |
| **Portfolio ROI (%)** | `+82.44%` | ROI berdasarkan kapital portofolio $100/koin |
| **Overall Win Rate** | `59.55%` | Rasio kemenangan rata-rata seluruh aset |
| **Total Trades** | `4,210` | Jumlah total posisi yang dieksekusi |
| **Rata-rata Trade / Bulan** | `40.7` | Rata-rata frekuensi trade bulanan portofolio |
| **Rata-rata Trade / Hari** | `1.34` | Rata-rata frekuensi trade harian portofolio |
| **Max Drawdown (5x)** | `93.58%` | Rata-rata penurunan terdalam portofolio |
| **Sharpe Ratio** | `4.11` | Efisiensi profit terhadap volatilitas portofolio |
| **Sortino Ratio** | `11.10` | Efisiensi profit terhadap downside deviation |
| **Calmar Ratio** | `12.07` | Rasio return tahunan terhadap drawdown |
| **Profit Factor** | `2.60` | Rasio gross profit dibagi gross loss |
| **Max Consecutive Loss** | `16` trades | Streak kekalahan beruntun terpanjang |
| **Worst Single Trade PnL** | `-24.90%` | Kerugian terdalam dalam satu trade tunggal |
| **95% Trades Loss Under** | `11.00%` | Nilai risiko (VaR P95) kerugian maksimal |

## ↕️ Analisis Arah Signal (LONG vs SHORT)

| Arah Posisi | Jumlah Trade | Distribusi | Menang | Kalah | Win Rate | PnL Bersih ($) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **LONG** | 1,755 | 41.7% | 918 | 837 | 52.31% | +294.27 |
| **SHORT** | 2,455 | 58.3% | 1,562 | 893 | 63.63% | +1,436.89 |

### Rincian Rata-rata Profitabilitas per Trade

| Tipe Trade | PnL Rata-rata ($) | PnL Rata-rata (%) |
|:---|:---:|:---:|
| **Trade Menang (Wins)** | `$+1.9553` | `+7.82%` |
| **Trade Kalah (Losses)** | `$-1.8023` | `-7.21%` |

## 📅 Scorecard Bulanan Portofolio

| Bulan | Total Trades | Wins | Losses | Win Rate | Net PnL ($) |
|:---|:---:|:---:|:---:|:---:|:---:|
| 2025-11 | 761 | 426 | 335 | 55.98% | $+242.91 |
| 2025-12 | 857 | 538 | 319 | 62.78% | $+553.74 |
| 2026-01 | 833 | 530 | 303 | 63.63% | $+551.76 |
| 2026-02 | 814 | 414 | 400 | 50.86% | $+0.46 |
| 2026-03 | 945 | 572 | 373 | 60.53% | $+382.29 |

## 🚪 Distribusi Alasan Keluar Posisi (Exit Reasons)

| Alasan Exit | Jumlah | Persentase | Wins | Losses | Win Rate | PnL Bersih ($) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| `guardian_exit` | 3,247 | 77.1% | 2,432 | 815 | 74.90% | $+3,488.29 |
| `sl_hit` | 907 | 21.5% | 2 | 905 | 0.22% | $-1,777.22 |
| `time_exit` | 56 | 1.3% | 46 | 10 | 82.14% | $+20.09 |

## 🪙 Scorecard Per Koin (Detailed Assets)

| Token | Win Rate | Trades | LONG WR | SHORT WR | Net PnL ($) | Max DD | Sharpe | Sortino | Calmar | PF |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1000PEPE** | 87.50% | 8 | 50.0% | 100.0% | `$+24.82` | 8.38% | 5.13 | 0.00 | 28.84 | 12.84 |
| **1000SHIB** | 71.43% | 7 | 0.0% | 83.3% | `$+7.59` | 4.88% | 2.30 | 5.41 | 15.16 | 6.98 |
| **ADA** | 64.20% | 162 | 56.7% | 69.5% | `$+130.72` | 51.87% | 6.48 | 20.18 | 24.55 | 2.27 |
| **ARB** | 56.85% | 241 | 51.7% | 62.0% | `$+127.95` | 85.52% | 4.74 | 11.99 | 14.57 | 1.64 |
| **AVAX** | 57.40% | 338 | 53.1% | 60.1% | `$+93.09` | 131.34% | 3.49 | 8.63 | 6.90 | 1.34 |
| **BNB** | 53.62% | 207 | 44.1% | 61.4% | `$+6.81` | 92.79% | 0.47 | 1.35 | 0.71 | 1.05 |
| **BTC** | 0.00% | 0 | 0.0% | 0.0% | `$+0.00` | 0.00% | 0.00 | 0.00 | 0.00 | 0.00 |
| **DOGE** | 85.71% | 21 | 71.4% | 92.9% | `$+40.44` | 13.85% | 6.04 | 26.90 | 28.44 | 7.69 |
| **DOT** | 56.54% | 283 | 46.5% | 63.3% | `$+59.04` | 126.89% | 2.47 | 6.20 | 4.53 | 1.26 |
| **ETH** | 68.45% | 168 | 65.6% | 70.1% | `$+129.26` | 76.29% | 7.14 | 19.46 | 16.50 | 2.37 |
| **HBAR** | 58.25% | 206 | 48.5% | 68.0% | `$+40.19` | 50.13% | 2.24 | 6.08 | 7.81 | 1.26 |
| **LINK** | 61.03% | 290 | 59.3% | 62.2% | `$+144.76` | 93.28% | 5.64 | 19.33 | 15.12 | 1.66 |
| **NEAR** | 58.36% | 293 | 42.9% | 68.0% | `$+123.45` | 84.36% | 4.51 | 10.35 | 14.25 | 1.54 |
| **ONDO** | 60.24% | 254 | 60.9% | 59.9% | `$+150.63` | 147.78% | 5.68 | 16.83 | 9.93 | 1.75 |
| **POL** | 55.46% | 238 | 50.0% | 60.3% | `$+101.58` | 179.02% | 4.32 | 12.15 | 5.53 | 1.54 |
| **SOL** | 56.16% | 333 | 51.3% | 58.8% | `$+106.88` | 152.93% | 3.87 | 12.00 | 6.81 | 1.40 |
| **SUI** | 62.45% | 269 | 53.4% | 68.1% | `$+167.61` | 190.33% | 6.22 | 15.87 | 8.58 | 1.87 |
| **TAO** | 54.62% | 249 | 49.0% | 63.0% | `$+83.84` | 261.37% | 2.89 | 6.27 | 3.12 | 1.35 |
| **TON** | 54.14% | 290 | 44.2% | 59.0% | `$+53.24` | 151.98% | 2.42 | 6.88 | 3.41 | 1.24 |
| **TRX** | 64.18% | 134 | 62.6% | 66.7% | `$+22.38` | 23.29% | 3.59 | 10.38 | 9.36 | 1.58 |
| **XRP** | 63.93% | 219 | 61.6% | 65.4% | `$+116.88` | 38.92% | 6.69 | 16.89 | 29.25 | 2.02 |

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
