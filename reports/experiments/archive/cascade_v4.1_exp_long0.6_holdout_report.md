# 📊 Holdout Backtest Report: `cascade_v4.1_exp_long0.6`

**Tanggal Pembuatan**: 2026-05-29 20:31:28 UTC
**Model Run ID**: `cascade_v4.1_exp_long0.6`
**Periode Pengujian (Temporal OOS)**: `2025-11-01 - 2026-04-01`

> [!NOTE]
> **Ringkasan Portofolio Eksekutif**:
> *   **Total Net Profit**: **$+895.83 USD** (ROI Portofolio: **+42.66%**)
> *   **Rata-rata Win Rate**: **57.58%** | Total Trades: **2,427**
> *   **Rata-rata Max Drawdown (5x)**: **110.81%**
> *   **Risk-Adjusted**: Sharpe: **2.45** | Sortino: **6.89** | Calmar: **4.57** | Profit Factor: **1.47**

## 📈 Performa Scorecard Portofolio

| Metrik Utama | Nilai Portofolio | Catatan |
|:---|:---:|:---|
| **Total Net Profit ($)** | `$+895.83` | Akumulasi keuntungan bersih 5x leverage |
| **Portfolio ROI (%)** | `+42.66%` | ROI berdasarkan kapital portofolio $100/koin |
| **Overall Win Rate** | `57.58%` | Rasio kemenangan rata-rata seluruh aset |
| **Total Trades** | `2,427` | Jumlah total posisi yang dieksekusi |
| **Rata-rata Trade / Bulan** | `23.4` | Rata-rata frekuensi trade bulanan portofolio |
| **Rata-rata Trade / Hari** | `0.77` | Rata-rata frekuensi trade harian portofolio |
| **Max Drawdown (5x)** | `110.81%` | Rata-rata penurunan terdalam portofolio |
| **Sharpe Ratio** | `2.45` | Efisiensi profit terhadap volatilitas portofolio |
| **Sortino Ratio** | `6.89` | Efisiensi profit terhadap downside deviation |
| **Calmar Ratio** | `4.57` | Rasio return tahunan terhadap drawdown |
| **Profit Factor** | `1.47` | Rasio gross profit dibagi gross loss |
| **Max Consecutive Loss** | `12` trades | Streak kekalahan beruntun terpanjang |
| **Worst Single Trade PnL** | `-28.90%` | Kerugian terdalam dalam satu trade tunggal |
| **95% Trades Loss Under** | `13.14%` | Nilai risiko (VaR P95) kerugian maksimal |

## ↕️ Analisis Arah Signal (LONG vs SHORT)

| Arah Posisi | Jumlah Trade | Distribusi | Menang | Kalah | Win Rate | PnL Bersih ($) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **LONG** | 1,509 | 62.2% | 801 | 708 | 53.08% | +206.26 |
| **SHORT** | 918 | 37.8% | 595 | 323 | 64.81% | +689.57 |

### Rincian Rata-rata Profitabilitas per Trade

| Tipe Trade | PnL Rata-rata ($) | PnL Rata-rata (%) |
|:---|:---:|:---:|
| **Trade Menang (Wins)** | `$+2.1194` | `+8.48%` |
| **Trade Kalah (Losses)** | `$-2.0008` | `-8.00%` |

## 📅 Scorecard Bulanan Portofolio

| Bulan | Total Trades | Wins | Losses | Win Rate | Net PnL ($) |
|:---|:---:|:---:|:---:|:---:|:---:|
| 2025-11 | 582 | 333 | 249 | 57.22% | $+146.67 |
| 2025-12 | 371 | 260 | 111 | 70.08% | $+343.25 |
| 2026-01 | 378 | 212 | 166 | 56.08% | $+132.94 |
| 2026-02 | 555 | 257 | 298 | 46.31% | $-126.88 |
| 2026-03 | 541 | 334 | 207 | 61.74% | $+399.86 |

## 🚪 Distribusi Alasan Keluar Posisi (Exit Reasons)

| Alasan Exit | Jumlah | Persentase | Wins | Losses | Win Rate | PnL Bersih ($) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| `guardian_exit` | 1,801 | 74.2% | 1,303 | 498 | 72.35% | $+2,003.18 |
| `sl_hit` | 520 | 21.4% | 1 | 519 | 0.19% | $-1,147.00 |
| `time_exit` | 106 | 4.4% | 92 | 14 | 86.79% | $+39.65 |

## 🪙 Scorecard Per Koin (Detailed Assets)

| Token | Win Rate | Trades | LONG WR | SHORT WR | Net PnL ($) | Max DD | Sharpe | Sortino | Calmar | PF |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1000PEPE** | 54.68% | 139 | 40.5% | 73.3% | `$+1.85` | 178.22% | 0.08 | 0.17 | 0.10 | 1.01 |
| **1000SHIB** | 57.38% | 122 | 50.6% | 71.8% | `$+16.39` | 84.53% | 1.12 | 2.47 | 1.89 | 1.17 |
| **ADA** | 63.48% | 115 | 55.0% | 82.9% | `$+70.04` | 85.86% | 3.73 | 10.45 | 7.94 | 1.81 |
| **ARB** | 52.34% | 128 | 47.4% | 66.7% | `$+38.89` | 91.15% | 1.90 | 4.46 | 4.16 | 1.31 |
| **AVAX** | 54.01% | 137 | 52.6% | 55.7% | `$+28.50` | 148.72% | 1.69 | 4.23 | 1.87 | 1.25 |
| **BNB** | 62.24% | 98 | 52.9% | 72.3% | `$+45.42` | 52.96% | 3.76 | 13.55 | 8.35 | 1.76 |
| **BTC** | 55.32% | 94 | 53.7% | 59.3% | `$+14.76` | 88.98% | 1.34 | 3.72 | 1.62 | 1.23 |
| **DOGE** | 56.03% | 116 | 51.4% | 63.6% | `$+58.52` | 129.62% | 3.06 | 8.77 | 4.40 | 1.58 |
| **DOT** | 58.87% | 124 | 51.4% | 69.2% | `$+51.05` | 132.17% | 2.59 | 6.46 | 3.76 | 1.45 |
| **ETH** | 55.45% | 101 | 49.1% | 63.6% | `$+34.35` | 169.87% | 1.97 | 6.10 | 1.97 | 1.39 |
| **HBAR** | 55.32% | 94 | 45.2% | 75.0% | `$+15.37` | 79.26% | 1.03 | 2.14 | 1.89 | 1.18 |
| **LINK** | 64.44% | 135 | 52.4% | 83.0% | `$+113.18` | 84.81% | 5.34 | 22.44 | 13.00 | 2.19 |
| **NEAR** | 59.66% | 119 | 58.9% | 60.9% | `$+68.67` | 103.52% | 3.31 | 8.08 | 6.46 | 1.64 |
| **ONDO** | 52.94% | 119 | 57.0% | 45.0% | `$+39.41` | 149.36% | 1.84 | 4.64 | 2.57 | 1.31 |
| **POL** | 61.98% | 121 | 66.7% | 50.0% | `$+64.79` | 64.30% | 3.95 | 7.93 | 9.81 | 1.80 |
| **SOL** | 51.54% | 130 | 45.8% | 56.3% | `$+43.03` | 183.44% | 2.05 | 6.52 | 2.28 | 1.34 |
| **SUI** | 53.68% | 136 | 51.9% | 56.1% | `$+37.60` | 208.29% | 1.57 | 3.86 | 1.76 | 1.27 |
| **TAO** | 60.71% | 112 | 59.4% | 62.5% | `$+53.86` | 134.90% | 2.22 | 4.71 | 3.89 | 1.43 |
| **TON** | 61.17% | 103 | 54.2% | 70.5% | `$+38.35` | 54.49% | 3.15 | 6.36 | 6.86 | 1.65 |
| **TRX** | 57.14% | 77 | 56.5% | 62.5% | `$+8.68` | 49.19% | 1.69 | 5.23 | 1.72 | 1.32 |
| **XRP** | 60.75% | 107 | 59.4% | 62.8% | `$+53.13` | 53.29% | 4.07 | 12.42 | 9.71 | 1.80 |

## ⛓️ Daftar Fitur Aktif dalam Model

Total terdapat **104 fitur aktif** yang digunakan oleh LightGBM entry, LSTM Soft Confirmation, dan Exit Guardian v3:

<details>
<summary>▶ Klik untuk melihat daftar lengkap fitur aktif</summary>

1. `open`
2. `high`
3. `low`
4. `close`
5. `volume`
6. `volume_delta`
7. `cvd`
8. `buy_volume`
9. `sell_volume`
10. `MSB_BOS`
11. `CHoCH`
12. `bars_since_BOS`
13. `FVG_up`
14. `FVG_down`
15. `Buy_Liq`
16. `Sell_Liq`
17. `SFP_sweep`
18. `open_interest`
19. `dynamic_position_pressure`
20. `funding_rate`
21. `ema_7_h1`
22. `ema_21_h1`
23. `ema_50_h1`
24. `ema_200_h1`
25. `ema_7_h4`
26. `ema_21_h4`
27. `ema_50_h4`
28. `ema_200_h4`
29. `rsi_6`
30. `stochrsi_k`
31. `stochrsi_d`
32. `atr_14_h1`
33. `atr_14_h4`
34. `PDH`
35. `PDL`
36. `PWH`
37. `PWL`
38. `Fib_618`
39. `Fib_786`
40. `POC`
41. `VAH`
42. `VAL`
43. `market_session`
44. `btc_dominance`
45. `fear_greed`
46. `log_ret_1`
47. `log_ret_5`
48. `log_ret_20`
49. `vol_ratio_20`
50. `hour_sin`
51. `hour_cos`
52. `dow_sin`
53. `dow_cos`
54. `time_to_funding_norm`
55. `long_short_ratio`
56. `dist_swing_high`
57. `dist_swing_low`
58. `price_in_range`
59. `swing_momentum`
60. `h4_trend`
61. `trend_strength`
62. `vol_regime`
63. `cvd_div_h4`
64. `cvd_slope_h4`
65. `vol_efficiency`
66. `absorption_z`
67. `funding_price_div`
68. `rsi_h4`
69. `rsi_divergence`
70. `wyckoff_phase`
71. `spring_upthrust`
72. `ofi_raw`
73. `ofi_acceleration`
74. `ofi_z_score`
75. `ofi_h4_delta`
76. `vwdp`
77. `vwdp_smooth`
78. `hidden_divergence`
79. `cvd_momentum_adv`
80. `absorption_at_swing`
81. `spread_to_volume`
82. `ultra_high_vol`
83. `no_demand`
84. `no_supply`
85. `effort_vs_result`
86. `ema_21_slope_h4`
87. `ema_50_slope_h4`
88. `price_vs_ema_50_h4`
89. `rsi_slope_h4`
90. `atr_percent_h4`
91. `range_expansion_h4`
92. `trend_accel_4h`
93. `vol_price_confirm`
94. `dist_from_8h_high`
95. `relative_strength_z`
96. `relative_strength_momentum`
97. `dist_liq_50x_long`
98. `dist_liq_20x_long`
99. `dist_liq_50x_short`
100. `dist_liq_20x_short`
101. `whale_retail_divergence`
102. `atr_zscore_20d`
103. `atr_percentile_h1`
104. `vol_spike_zscore`

</details>
