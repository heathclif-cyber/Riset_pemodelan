# 📊 Holdout Backtest Report: `lgbm_trending_v1`

**Tanggal Pembuatan**: 2026-06-08 09:34:30 UTC
**Model Run ID**: `lgbm_trending_v1`
**Periode Pengujian (Temporal OOS)**: `2025-11-01 - 2026-04-01`

> [!NOTE]
> **Ringkasan Portofolio Eksekutif**:
> *   **Total Net Profit**: **$+740.86 USD** (ROI Portofolio: **+35.28%**)
> *   **Rata-rata Win Rate**: **65.52%** | Total Trades: **2,281**
> *   **Rata-rata Max Drawdown (5x)**: **62.88%**
> *   **Risk-Adjusted**: Sharpe: **5.59** | Sortino: **14.71** | Calmar: **17.18** | Profit Factor: **2.43**

## 📈 Performa Scorecard Portofolio

| Metrik Utama | Nilai Portofolio | Catatan |
|:---|:---:|:---|
| **Total Net Profit ($)** | `$+740.86` | Akumulasi keuntungan bersih 5x leverage |
| **Portfolio ROI (%)** | `+35.28%` | ROI berdasarkan kapital portofolio $100/koin |
| **Overall Win Rate** | `65.52%` | Rasio kemenangan rata-rata seluruh aset |
| **Total Trades** | `2,281` | Jumlah total posisi yang dieksekusi |
| **Rata-rata Trade / Bulan** | `22.0` | Rata-rata frekuensi trade bulanan portofolio |
| **Rata-rata Trade / Hari** | `0.72` | Rata-rata frekuensi trade harian portofolio |
| **Max Drawdown (5x)** | `62.88%` | Rata-rata penurunan terdalam portofolio |
| **Sharpe Ratio** | `5.59` | Efisiensi profit terhadap volatilitas portofolio |
| **Sortino Ratio** | `14.71` | Efisiensi profit terhadap downside deviation |
| **Calmar Ratio** | `17.18` | Rasio return tahunan terhadap drawdown |
| **Profit Factor** | `2.43` | Rasio gross profit dibagi gross loss |
| **Max Consecutive Loss** | `10` trades | Streak kekalahan beruntun terpanjang |
| **Worst Single Trade PnL** | `-24.70%` | Kerugian terdalam dalam satu trade tunggal |
| **95% Trades Loss Under** | `11.20%` | Nilai risiko (VaR P95) kerugian maksimal |

## ↕️ Analisis Arah Signal (LONG vs SHORT)

| Arah Posisi | Jumlah Trade | Distribusi | Menang | Kalah | Win Rate | PnL Bersih ($) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **LONG** | 923 | 40.5% | 602 | 321 | 65.22% | +357.10 |
| **SHORT** | 1,358 | 59.5% | 893 | 465 | 65.76% | +383.76 |

### Rincian Rata-rata Profitabilitas per Trade

| Tipe Trade | PnL Rata-rata ($) | PnL Rata-rata (%) |
|:---|:---:|:---:|
| **Trade Menang (Wins)** | `$+0.8868` | `+8.87%` |
| **Trade Kalah (Losses)** | `$-0.7442` | `-7.44%` |

## 📅 Scorecard Bulanan Portofolio

| Bulan | Total Trades | Wins | Losses | Win Rate | Net PnL ($) |
|:---|:---:|:---:|:---:|:---:|:---:|
| 2025-11 | 574 | 362 | 212 | 63.07% | $+162.29 |
| 2025-12 | 477 | 338 | 139 | 70.86% | $+199.16 |
| 2026-01 | 436 | 301 | 135 | 69.04% | $+192.58 |
| 2026-02 | 300 | 192 | 108 | 64.00% | $+72.77 |
| 2026-03 | 494 | 302 | 192 | 61.13% | $+114.06 |

## 🚪 Distribusi Alasan Keluar Posisi (Exit Reasons)

| Alasan Exit | Jumlah | Persentase | Wins | Losses | Win Rate | PnL Bersih ($) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| `guardian_exit` | 1,826 | 80.1% | 1,464 | 362 | 80.18% | $+1,074.48 |
| `sl_hit` | 419 | 18.4% | 1 | 418 | 0.24% | $-339.32 |
| `time_exit` | 36 | 1.6% | 30 | 6 | 83.33% | $+5.70 |

## 🪙 Scorecard Per Koin (Detailed Assets)

| Token | Win Rate | Trades | LONG WR | SHORT WR | Net PnL ($) | Max DD | Sharpe | Sortino | Calmar | PF |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1000PEPE** | 65.31% | 147 | 55.0% | 69.2% | `$+51.58` | 103.22% | 5.33 | 11.86 | 12.17 | 2.03 |
| **1000SHIB** | 67.62% | 105 | 61.4% | 72.1% | `$+32.48` | 29.91% | 5.81 | 15.20 | 26.44 | 2.48 |
| **ADA** | 66.39% | 122 | 75.6% | 61.0% | `$+50.39` | 55.66% | 6.43 | 27.18 | 22.04 | 2.45 |
| **ARB** | 59.22% | 103 | 60.3% | 57.1% | `$+25.53` | 68.36% | 3.45 | 7.50 | 9.09 | 1.78 |
| **AVAX** | 58.04% | 112 | 67.3% | 49.1% | `$+26.16` | 78.54% | 3.93 | 11.58 | 8.11 | 1.75 |
| **BNB** | 67.95% | 78 | 54.8% | 76.6% | `$+21.46` | 75.96% | 4.71 | 11.93 | 6.88 | 2.23 |
| **BTC** | 79.46% | 112 | 88.6% | 75.3% | `$+38.93` | 34.35% | 9.75 | 26.68 | 27.60 | 3.95 |
| **DOGE** | 62.96% | 108 | 60.9% | 64.5% | `$+33.31` | 52.65% | 4.85 | 11.94 | 15.41 | 2.17 |
| **DOT** | 59.21% | 76 | 61.5% | 56.8% | `$+25.42` | 55.41% | 4.39 | 9.72 | 11.17 | 2.20 |
| **ETH** | 63.87% | 119 | 63.8% | 63.9% | `$+38.42` | 52.79% | 5.81 | 14.80 | 17.72 | 2.32 |
| **HBAR** | 72.15% | 79 | 59.0% | 85.0% | `$+32.20` | 45.57% | 6.44 | 15.04 | 17.20 | 3.00 |
| **LINK** | 75.22% | 113 | 79.5% | 73.0% | `$+57.31` | 43.42% | 9.00 | 29.33 | 32.14 | 3.80 |
| **NEAR** | 54.26% | 129 | 47.6% | 57.5% | `$+21.66` | 90.93% | 2.70 | 6.56 | 5.80 | 1.45 |
| **ONDO** | 59.57% | 94 | 57.9% | 60.7% | `$+38.39` | 97.65% | 4.97 | 12.53 | 9.57 | 2.27 |
| **POL** | 67.59% | 108 | 66.7% | 68.5% | `$+43.48` | 37.12% | 6.27 | 14.05 | 28.52 | 2.65 |
| **SOL** | 62.14% | 103 | 66.7% | 59.0% | `$+20.75` | 95.36% | 3.25 | 8.12 | 5.30 | 1.69 |
| **SUI** | 70.63% | 143 | 71.7% | 70.1% | `$+77.30` | 47.77% | 8.77 | 20.91 | 39.40 | 3.51 |
| **TAO** | 64.00% | 100 | 63.5% | 64.6% | `$+37.55` | 132.30% | 4.62 | 10.27 | 6.91 | 2.11 |
| **TON** | 57.29% | 96 | 57.5% | 57.1% | `$+25.47` | 48.27% | 4.60 | 16.27 | 12.85 | 2.07 |
| **TRX** | 77.01% | 87 | 75.0% | 78.7% | `$+13.08` | 8.99% | 7.37 | 17.28 | 35.43 | 3.17 |
| **XRP** | 65.99% | 147 | 82.3% | 61.1% | `$+29.99` | 66.22% | 4.89 | 10.19 | 11.03 | 1.85 |

## ⛓️ Daftar Fitur Aktif dalam Model

Total terdapat **108 fitur aktif** yang digunakan oleh LightGBM entry, LSTM Soft Confirmation, dan Exit Guardian v3:

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
105. `price_accel_1h`
106. `ofi_momentum_ratio`
107. `vol_accel_3h`
108. `hmm_regime_enc`

</details>
