# 📊 Holdout Backtest Report: `cascade_v4.1`

![Performance Scorecard Summary Charts](file:///D:/Apps-Dev/Riset_pemodelan/reports/experiments/cascade_v4.1_holdout_charts.png)


**Tanggal Pembuatan**: 2026-05-29 07:03:50 UTC
**Model Run ID**: `cascade_v4.1`
**Periode Pengujian (Temporal OOS)**: `2025-11-01 - 2026-04-01`

> [!NOTE]
> **Ringkasan Portofolio Eksekutif**:
> *   **Total Net Profit**: **$+1,040.61 USD** (ROI Portofolio: **+49.55%**)
> *   **Rata-rata Win Rate**: **65.06%** | Total Trades: **1,354**
> *   **Rata-rata Max Drawdown (5x)**: **54.82%**
> *   **Risk-Adjusted**: Sharpe: **3.71** | Sortino: **11.39** | Calmar: **12.46** | Profit Factor: **2.37**

## 📈 Performa Scorecard Portofolio

| Metrik Utama | Nilai Portofolio | Catatan |
|:---|:---:|:---|
| **Total Net Profit ($)** | `$+1,040.61` | Akumulasi keuntungan bersih 5x leverage |
| **Portfolio ROI (%)** | `+49.55%` | ROI berdasarkan kapital portofolio $100/koin |
| **Overall Win Rate** | `65.06%` | Rasio kemenangan rata-rata seluruh aset |
| **Total Trades** | `1,354` | Jumlah total posisi yang dieksekusi |
| **Rata-rata Trade / Bulan** | `13.1` | Rata-rata frekuensi trade bulanan portofolio |
| **Rata-rata Trade / Hari** | `0.43` | Rata-rata frekuensi trade harian portofolio |
| **Max Drawdown (5x)** | `54.82%` | Rata-rata penurunan terdalam portofolio |
| **Sharpe Ratio** | `3.71` | Efisiensi profit terhadap volatilitas portofolio |
| **Sortino Ratio** | `11.39` | Efisiensi profit terhadap downside deviation |
| **Calmar Ratio** | `12.46` | Rasio return tahunan terhadap drawdown |
| **Profit Factor** | `2.37` | Rasio gross profit dibagi gross loss |
| **Max Consecutive Loss** | `10` trades | Streak kekalahan beruntun terpanjang |
| **Worst Single Trade PnL** | `-24.90%` | Kerugian terdalam dalam satu trade tunggal |
| **95% Trades Loss Under** | `12.00%` | Nilai risiko (VaR P95) kerugian maksimal |

## ↕️ Analisis Arah Signal (LONG vs SHORT)

| Arah Posisi | Jumlah Trade | Distribusi | Menang | Kalah | Win Rate | PnL Bersih ($) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **LONG** | 368 | 27.2% | 232 | 136 | 63.04% | +287.34 |
| **SHORT** | 986 | 72.8% | 644 | 342 | 65.31% | +753.27 |

### Rincian Rata-rata Profitabilitas per Trade

| Tipe Trade | PnL Rata-rata ($) | PnL Rata-rata (%) |
|:---|:---:|:---:|
| **Trade Menang (Wins)** | `$+2.2701` | `+9.08%` |
| **Trade Kalah (Losses)** | `$-1.9832` | `-7.93%` |

## 📅 Scorecard Bulanan Portofolio

| Bulan | Total Trades | Wins | Losses | Win Rate | Net PnL ($) |
|:---|:---:|:---:|:---:|:---:|:---:|
| 2025-11 | 303 | 196 | 107 | 64.69% | $+217.57 |
| 2025-12 | 224 | 167 | 57 | 74.55% | $+247.73 |
| 2026-01 | 192 | 119 | 73 | 61.98% | $+173.04 |
| 2026-02 | 335 | 178 | 157 | 53.13% | $+21.25 |
| 2026-03 | 300 | 216 | 84 | 72.00% | $+381.02 |

## 🚪 Distribusi Alasan Keluar Posisi (Exit Reasons)

| Alasan Exit | Jumlah | Persentase | Wins | Losses | Win Rate | PnL Bersih ($) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| `guardian_exit` | 1,055 | 77.9% | 829 | 226 | 78.58% | $+1,577.08 |
| `sl_hit` | 249 | 18.4% | 0 | 249 | 0.00% | $-556.26 |
| `time_exit` | 50 | 3.7% | 47 | 3 | 94.00% | $+19.79 |

## 🪙 Scorecard Per Koin (Detailed Assets)

| Token | Win Rate | Trades | LONG WR | SHORT WR | Net PnL ($) | Max DD | Sharpe | Sortino | Calmar | PF |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1000PEPE** | 65.38% | 78 | 35.3% | 73.8% | `$+48.05` | 69.65% | 2.74 | 5.70 | 6.72 | 1.63 |
| **1000SHIB** | 64.06% | 64 | 45.5% | 73.8% | `$+31.31` | 27.72% | 3.03 | 6.80 | 11.00 | 1.80 |
| **ADA** | 83.93% | 56 | 85.7% | 83.3% | `$+107.41` | 16.92% | 7.38 | 22.68 | 61.83 | 6.24 |
| **ARB** | 68.25% | 63 | 69.0% | 67.7% | `$+77.48` | 37.49% | 5.25 | 15.87 | 20.13 | 2.97 |
| **AVAX** | 55.95% | 84 | 52.6% | 56.9% | `$+24.03` | 97.60% | 1.89 | 4.56 | 2.40 | 1.39 |
| **BNB** | 74.60% | 63 | 80.0% | 72.9% | `$+58.16` | 21.03% | 6.26 | 25.83 | 26.94 | 3.38 |
| **BTC** | 55.00% | 40 | 41.7% | 60.7% | `$+3.55` | 63.50% | 0.46 | 1.30 | 0.54 | 1.12 |
| **DOGE** | 67.69% | 65 | 76.5% | 64.6% | `$+72.95` | 59.23% | 4.99 | 15.43 | 12.00 | 2.95 |
| **DOT** | 67.61% | 71 | 73.3% | 66.1% | `$+55.86` | 86.05% | 3.97 | 7.85 | 6.32 | 2.14 |
| **ETH** | 60.94% | 64 | 50.0% | 63.5% | `$+43.29` | 105.35% | 2.94 | 9.64 | 4.00 | 1.80 |
| **HBAR** | 71.74% | 46 | 64.3% | 75.0% | `$+41.33` | 25.65% | 4.27 | 9.50 | 15.69 | 2.56 |
| **LINK** | 77.63% | 76 | 71.4% | 80.0% | `$+120.66` | 50.66% | 7.34 | 29.27 | 23.20 | 4.43 |
| **NEAR** | 64.47% | 76 | 60.9% | 66.0% | `$+60.01` | 59.28% | 3.74 | 9.78 | 9.86 | 1.98 |
| **ONDO** | 46.03% | 63 | 50.0% | 43.9% | `$+26.62` | 105.74% | 1.55 | 4.94 | 2.45 | 1.36 |
| **POL** | 62.07% | 58 | 79.0% | 53.8% | `$+46.94` | 42.65% | 3.71 | 8.70 | 10.72 | 2.22 |
| **SOL** | 60.67% | 89 | 75.0% | 57.5% | `$+61.45` | 76.94% | 3.91 | 13.69 | 7.78 | 1.95 |
| **SUI** | 56.25% | 80 | 54.5% | 56.9% | `$+44.80` | 65.52% | 2.27 | 6.24 | 6.66 | 1.58 |
| **TAO** | 66.67% | 72 | 76.5% | 63.6% | `$+47.44` | 55.11% | 2.90 | 5.65 | 8.38 | 1.78 |
| **TON** | 66.67% | 63 | 56.2% | 70.2% | `$+34.94` | 45.06% | 3.37 | 7.14 | 7.55 | 1.98 |
| **TRX** | 72.00% | 25 | 75.0% | 66.7% | `$+10.58` | 10.91% | 3.52 | 19.12 | 9.44 | 2.91 |
| **XRP** | 58.62% | 58 | 50.0% | 60.4% | `$+23.77` | 29.06% | 2.32 | 9.52 | 7.97 | 1.59 |

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
