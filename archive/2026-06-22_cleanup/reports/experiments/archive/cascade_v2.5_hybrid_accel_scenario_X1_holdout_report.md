# 📊 Holdout Backtest Report: `cascade_v2.5_hybrid_accel_scenario_X1`

**Tanggal Pembuatan**: 2026-06-03 21:25:01 UTC
**Model Run ID**: `cascade_v2.5_hybrid_accel_scenario_X1`
**Periode Pengujian (Temporal OOS)**: `2025-11-01 - 2026-04-01`

> [!NOTE]
> **Ringkasan Portofolio Eksekutif**:
> *   **Total Net Profit**: **$+1,217.09 USD** (ROI Portofolio: **+57.96%**)
> *   **Rata-rata Win Rate**: **61.45%** | Total Trades: **1,693**
> *   **Rata-rata Max Drawdown (5x)**: **45.51%**
> *   **Risk-Adjusted**: Sharpe: **4.13** | Sortino: **11.71** | Calmar: **12.40** | Profit Factor: **2.29**

## 📈 Performa Scorecard Portofolio

| Metrik Utama | Nilai Portofolio | Catatan |
|:---|:---:|:---|
| **Total Net Profit ($)** | `$+1,217.09` | Akumulasi keuntungan bersih 5x leverage |
| **Portfolio ROI (%)** | `+57.96%` | ROI berdasarkan kapital portofolio $100/koin |
| **Overall Win Rate** | `61.45%` | Rasio kemenangan rata-rata seluruh aset |
| **Total Trades** | `1,693` | Jumlah total posisi yang dieksekusi |
| **Rata-rata Trade / Bulan** | `16.4` | Rata-rata frekuensi trade bulanan portofolio |
| **Rata-rata Trade / Hari** | `0.54` | Rata-rata frekuensi trade harian portofolio |
| **Max Drawdown (5x)** | `45.51%` | Rata-rata penurunan terdalam portofolio |
| **Sharpe Ratio** | `4.13` | Efisiensi profit terhadap volatilitas portofolio |
| **Sortino Ratio** | `11.71` | Efisiensi profit terhadap downside deviation |
| **Calmar Ratio** | `12.40` | Rasio return tahunan terhadap drawdown |
| **Profit Factor** | `2.29` | Rasio gross profit dibagi gross loss |
| **Max Consecutive Loss** | `9` trades | Streak kekalahan beruntun terpanjang |
| **Worst Single Trade PnL** | `-23.60%` | Kerugian terdalam dalam satu trade tunggal |
| **95% Trades Loss Under** | `10.20%` | Nilai risiko (VaR P95) kerugian maksimal |

## ↕️ Analisis Arah Signal (LONG vs SHORT)

| Arah Posisi | Jumlah Trade | Distribusi | Menang | Kalah | Win Rate | PnL Bersih ($) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **LONG** | 403 | 23.8% | 240 | 163 | 59.55% | +241.48 |
| **SHORT** | 1,290 | 76.2% | 848 | 442 | 65.74% | +975.61 |

### Rincian Rata-rata Profitabilitas per Trade

| Tipe Trade | PnL Rata-rata ($) | PnL Rata-rata (%) |
|:---|:---:|:---:|
| **Trade Menang (Wins)** | `$+2.0836` | `+8.34%` |
| **Trade Kalah (Losses)** | `$-1.7353` | `-6.94%` |

## 📅 Scorecard Bulanan Portofolio

| Bulan | Total Trades | Wins | Losses | Win Rate | Net PnL ($) |
|:---|:---:|:---:|:---:|:---:|:---:|
| 2025-11 | 306 | 199 | 107 | 65.03% | $+249.49 |
| 2025-12 | 324 | 221 | 103 | 68.21% | $+353.39 |
| 2026-01 | 282 | 191 | 91 | 67.73% | $+260.02 |
| 2026-02 | 383 | 211 | 172 | 55.09% | $+79.79 |
| 2026-03 | 398 | 266 | 132 | 66.83% | $+274.39 |

## 🚪 Distribusi Alasan Keluar Posisi (Exit Reasons)

| Alasan Exit | Jumlah | Persentase | Wins | Losses | Win Rate | PnL Bersih ($) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| `guardian_exit` | 1,323 | 78.1% | 1,071 | 252 | 80.95% | $+1,856.07 |
| `sl_hit` | 352 | 20.8% | 1 | 351 | 0.28% | $-646.33 |
| `time_exit` | 18 | 1.1% | 16 | 2 | 88.89% | $+7.35 |

## 🪙 Scorecard Per Koin (Detailed Assets)

| Token | Win Rate | Trades | LONG WR | SHORT WR | Net PnL ($) | Max DD | Sharpe | Sortino | Calmar | PF |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1000PEPE** | 75.00% | 4 | 0.0% | 75.0% | `$+7.64` | 6.26% | 1.87 | 0.00 | 11.89 | 5.88 |
| **1000SHIB** | 55.56% | 9 | 33.3% | 66.7% | `$+2.27` | 14.17% | 0.67 | 1.80 | 1.56 | 1.44 |
| **ADA** | 66.67% | 69 | 64.3% | 67.3% | `$+68.61` | 58.62% | 4.80 | 16.83 | 11.40 | 2.62 |
| **ARB** | 59.30% | 86 | 62.1% | 57.9% | `$+66.56` | 63.59% | 4.06 | 12.57 | 10.19 | 1.98 |
| **AVAX** | 63.38% | 142 | 58.1% | 64.9% | `$+111.19` | 57.18% | 6.46 | 19.92 | 18.94 | 2.39 |
| **BNB** | 64.04% | 89 | 57.9% | 65.7% | `$+31.07` | 30.08% | 3.33 | 9.67 | 10.06 | 1.64 |
| **BTC** | 0.00% | 0 | 0.0% | 0.0% | `$+0.00` | 0.00% | 0.00 | 0.00 | 0.00 | 0.00 |
| **DOGE** | 61.54% | 13 | 0.0% | 66.7% | `$+3.30` | 17.04% | 0.66 | 1.40 | 1.89 | 1.36 |
| **DOT** | 62.50% | 104 | 41.7% | 68.8% | `$+68.51` | 41.50% | 4.26 | 10.40 | 16.08 | 1.92 |
| **ETH** | 78.82% | 85 | 75.0% | 79.2% | `$+109.19` | 28.37% | 8.94 | 28.94 | 37.49 | 4.67 |
| **HBAR** | 61.54% | 91 | 54.2% | 64.2% | `$+37.20` | 35.45% | 3.16 | 8.88 | 10.22 | 1.63 |
| **LINK** | 68.50% | 127 | 61.8% | 71.0% | `$+115.64` | 40.34% | 6.69 | 21.24 | 27.92 | 2.52 |
| **NEAR** | 67.68% | 99 | 50.0% | 72.2% | `$+118.60` | 53.45% | 7.52 | 23.51 | 21.61 | 3.43 |
| **ONDO** | 63.89% | 108 | 65.2% | 63.5% | `$+91.52` | 70.47% | 5.24 | 13.17 | 12.65 | 2.24 |
| **POL** | 59.26% | 81 | 64.0% | 57.1% | `$+69.07` | 89.18% | 4.75 | 14.07 | 7.54 | 2.22 |
| **SOL** | 54.93% | 142 | 50.0% | 56.2% | `$+37.16` | 108.03% | 2.08 | 7.14 | 3.35 | 1.33 |
| **SUI** | 65.60% | 125 | 62.1% | 66.7% | `$+89.84` | 71.80% | 4.74 | 11.80 | 12.19 | 2.08 |
| **TAO** | 60.81% | 74 | 58.3% | 62.0% | `$+39.17` | 60.45% | 2.54 | 6.03 | 6.31 | 1.60 |
| **TON** | 61.62% | 99 | 58.8% | 62.2% | `$+65.79` | 59.37% | 4.74 | 14.64 | 10.79 | 2.05 |
| **TRX** | 70.21% | 47 | 66.7% | 73.9% | `$+13.22` | 14.79% | 3.61 | 7.57 | 8.71 | 2.25 |
| **XRP** | 69.70% | 99 | 79.2% | 66.7% | `$+71.52` | 35.53% | 6.64 | 16.34 | 19.61 | 2.74 |

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
