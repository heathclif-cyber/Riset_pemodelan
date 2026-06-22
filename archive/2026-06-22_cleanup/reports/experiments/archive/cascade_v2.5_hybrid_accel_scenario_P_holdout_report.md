# 📊 Holdout Backtest Report: `cascade_v2.5_hybrid_accel_scenario_P`

**Tanggal Pembuatan**: 2026-06-02 20:30:42 UTC
**Model Run ID**: `cascade_v2.5_hybrid_accel_scenario_P`
**Periode Pengujian (Temporal OOS)**: `2025-11-01 - 2026-04-01`

> [!NOTE]
> **Ringkasan Portofolio Eksekutif**:
> *   **Total Net Profit**: **$+2,098.05 USD** (ROI Portofolio: **+99.91%**)
> *   **Rata-rata Win Rate**: **60.39%** | Total Trades: **5,761**
> *   **Rata-rata Max Drawdown (5x)**: **109.12%**
> *   **Risk-Adjusted**: Sharpe: **4.41** | Sortino: **11.92** | Calmar: **12.55** | Profit Factor: **1.73**

## 📈 Performa Scorecard Portofolio

| Metrik Utama | Nilai Portofolio | Catatan |
|:---|:---:|:---|
| **Total Net Profit ($)** | `$+2,098.05` | Akumulasi keuntungan bersih 5x leverage |
| **Portfolio ROI (%)** | `+99.91%` | ROI berdasarkan kapital portofolio $100/koin |
| **Overall Win Rate** | `60.39%` | Rasio kemenangan rata-rata seluruh aset |
| **Total Trades** | `5,761` | Jumlah total posisi yang dieksekusi |
| **Rata-rata Trade / Bulan** | `55.7` | Rata-rata frekuensi trade bulanan portofolio |
| **Rata-rata Trade / Hari** | `1.83` | Rata-rata frekuensi trade harian portofolio |
| **Max Drawdown (5x)** | `109.12%` | Rata-rata penurunan terdalam portofolio |
| **Sharpe Ratio** | `4.41` | Efisiensi profit terhadap volatilitas portofolio |
| **Sortino Ratio** | `11.92` | Efisiensi profit terhadap downside deviation |
| **Calmar Ratio** | `12.55` | Rasio return tahunan terhadap drawdown |
| **Profit Factor** | `1.73` | Rasio gross profit dibagi gross loss |
| **Max Consecutive Loss** | `15` trades | Streak kekalahan beruntun terpanjang |
| **Worst Single Trade PnL** | `-25.90%` | Kerugian terdalam dalam satu trade tunggal |
| **95% Trades Loss Under** | `11.10%` | Nilai risiko (VaR P95) kerugian maksimal |

## ↕️ Analisis Arah Signal (LONG vs SHORT)

| Arah Posisi | Jumlah Trade | Distribusi | Menang | Kalah | Win Rate | PnL Bersih ($) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **LONG** | 2,303 | 40.0% | 1,227 | 1,076 | 53.28% | +433.63 |
| **SHORT** | 3,458 | 60.0% | 2,140 | 1,318 | 61.89% | +1,664.43 |

### Rincian Rata-rata Profitabilitas per Trade

| Tipe Trade | PnL Rata-rata ($) | PnL Rata-rata (%) |
|:---|:---:|:---:|
| **Trade Menang (Wins)** | `$+1.9084` | `+7.63%` |
| **Trade Kalah (Losses)** | `$-1.8077` | `-7.23%` |

## 📅 Scorecard Bulanan Portofolio

| Bulan | Total Trades | Wins | Losses | Win Rate | Net PnL ($) |
|:---|:---:|:---:|:---:|:---:|:---:|
| 2025-11 | 1112 | 623 | 489 | 56.03% | $+290.09 |
| 2025-12 | 1161 | 709 | 452 | 61.07% | $+632.63 |
| 2026-01 | 1142 | 721 | 421 | 63.13% | $+681.27 |
| 2026-02 | 1102 | 574 | 528 | 52.09% | $+50.24 |
| 2026-03 | 1244 | 740 | 504 | 59.49% | $+443.82 |

## 🚪 Distribusi Alasan Keluar Posisi (Exit Reasons)

| Alasan Exit | Jumlah | Persentase | Wins | Losses | Win Rate | PnL Bersih ($) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| `guardian_exit` | 4,413 | 76.6% | 3,299 | 1,114 | 74.76% | $+4,564.50 |
| `sl_hit` | 1,268 | 22.0% | 2 | 1,266 | 0.16% | $-2,494.63 |
| `time_exit` | 80 | 1.4% | 66 | 14 | 82.50% | $+28.18 |

## 🪙 Scorecard Per Koin (Detailed Assets)

| Token | Win Rate | Trades | LONG WR | SHORT WR | Net PnL ($) | Max DD | Sharpe | Sortino | Calmar | PF |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1000PEPE** | 61.73% | 81 | 57.6% | 64.6% | `$+70.23` | 53.51% | 3.61 | 10.19 | 12.78 | 1.96 |
| **1000SHIB** | 68.00% | 75 | 61.4% | 77.4% | `$+49.22` | 50.25% | 4.35 | 10.00 | 9.54 | 2.33 |
| **ADA** | 65.16% | 221 | 59.3% | 69.2% | `$+171.73` | 48.12% | 7.39 | 23.71 | 34.76 | 2.21 |
| **ARB** | 55.87% | 315 | 50.3% | 60.7% | `$+135.83` | 127.01% | 4.53 | 11.54 | 10.42 | 1.51 |
| **AVAX** | 56.83% | 454 | 54.8% | 58.0% | `$+138.83` | 219.74% | 4.52 | 11.09 | 6.15 | 1.39 |
| **BNB** | 54.74% | 274 | 49.1% | 58.8% | `$+17.17` | 76.31% | 1.05 | 3.05 | 2.19 | 1.09 |
| **BTC** | 78.57% | 56 | 77.4% | 80.0% | `$+49.44` | 21.94% | 6.00 | 14.98 | 21.95 | 3.48 |
| **DOGE** | 67.42% | 89 | 71.9% | 64.9% | `$+105.04` | 24.06% | 6.99 | 19.77 | 42.52 | 3.36 |
| **DOT** | 57.71% | 376 | 46.5% | 64.5% | `$+97.35` | 108.15% | 3.49 | 8.25 | 8.77 | 1.33 |
| **ETH** | 62.45% | 229 | 57.5% | 65.9% | `$+111.02` | 170.18% | 5.11 | 13.77 | 6.35 | 1.70 |
| **HBAR** | 58.46% | 260 | 50.4% | 65.2% | `$+51.03` | 66.30% | 2.52 | 6.99 | 7.50 | 1.26 |
| **LINK** | 58.14% | 387 | 56.1% | 59.2% | `$+130.71` | 92.23% | 4.48 | 14.74 | 13.80 | 1.42 |
| **NEAR** | 57.87% | 375 | 44.2% | 65.8% | `$+146.75` | 118.01% | 4.74 | 10.91 | 12.11 | 1.48 |
| **ONDO** | 57.28% | 323 | 56.8% | 57.6% | `$+134.69` | 142.96% | 4.58 | 12.54 | 9.18 | 1.49 |
| **POL** | 54.52% | 310 | 51.8% | 56.6% | `$+103.11` | 147.64% | 3.88 | 10.94 | 6.80 | 1.41 |
| **SOL** | 54.95% | 404 | 49.2% | 57.7% | `$+87.99` | 196.92% | 3.01 | 8.49 | 4.35 | 1.27 |
| **SUI** | 61.60% | 349 | 53.6% | 66.1% | `$+201.56` | 151.31% | 6.59 | 17.01 | 12.97 | 1.79 |
| **TAO** | 54.84% | 310 | 49.5% | 62.5% | `$+79.54` | 277.98% | 2.49 | 5.37 | 2.79 | 1.26 |
| **TON** | 53.18% | 393 | 44.4% | 56.8% | `$+56.94` | 113.76% | 2.25 | 6.36 | 4.87 | 1.18 |
| **TRX** | 66.17% | 201 | 65.8% | 66.7% | `$+33.06` | 23.62% | 4.63 | 13.85 | 13.63 | 1.62 |
| **XRP** | 62.72% | 279 | 59.8% | 64.5% | `$+126.81` | 61.44% | 6.50 | 16.72 | 20.10 | 1.83 |

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
