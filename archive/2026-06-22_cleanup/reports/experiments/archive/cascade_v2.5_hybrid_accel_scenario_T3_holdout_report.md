# 📊 Holdout Backtest Report: `cascade_v2.5_hybrid_accel_scenario_T3`

**Tanggal Pembuatan**: 2026-06-02 21:12:33 UTC
**Model Run ID**: `cascade_v2.5_hybrid_accel_scenario_T3`
**Periode Pengujian (Temporal OOS)**: `2025-11-01 - 2026-04-01`

> [!NOTE]
> **Ringkasan Portofolio Eksekutif**:
> *   **Total Net Profit**: **$+1,583.97 USD** (ROI Portofolio: **+75.43%**)
> *   **Rata-rata Win Rate**: **60.86%** | Total Trades: **3,174**
> *   **Rata-rata Max Drawdown (5x)**: **83.04%**
> *   **Risk-Adjusted**: Sharpe: **4.31** | Sortino: **12.50** | Calmar: **11.44** | Profit Factor: **2.12**

## 📈 Performa Scorecard Portofolio

| Metrik Utama | Nilai Portofolio | Catatan |
|:---|:---:|:---|
| **Total Net Profit ($)** | `$+1,583.97` | Akumulasi keuntungan bersih 5x leverage |
| **Portfolio ROI (%)** | `+75.43%` | ROI berdasarkan kapital portofolio $100/koin |
| **Overall Win Rate** | `60.86%` | Rasio kemenangan rata-rata seluruh aset |
| **Total Trades** | `3,174` | Jumlah total posisi yang dieksekusi |
| **Rata-rata Trade / Bulan** | `30.7` | Rata-rata frekuensi trade bulanan portofolio |
| **Rata-rata Trade / Hari** | `1.01` | Rata-rata frekuensi trade harian portofolio |
| **Max Drawdown (5x)** | `83.04%` | Rata-rata penurunan terdalam portofolio |
| **Sharpe Ratio** | `4.31` | Efisiensi profit terhadap volatilitas portofolio |
| **Sortino Ratio** | `12.50` | Efisiensi profit terhadap downside deviation |
| **Calmar Ratio** | `11.44` | Rasio return tahunan terhadap drawdown |
| **Profit Factor** | `2.12` | Rasio gross profit dibagi gross loss |
| **Max Consecutive Loss** | `11` trades | Streak kekalahan beruntun terpanjang |
| **Worst Single Trade PnL** | `-24.90%` | Kerugian terdalam dalam satu trade tunggal |
| **95% Trades Loss Under** | `11.20%` | Nilai risiko (VaR P95) kerugian maksimal |

## ↕️ Analisis Arah Signal (LONG vs SHORT)

| Arah Posisi | Jumlah Trade | Distribusi | Menang | Kalah | Win Rate | PnL Bersih ($) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **LONG** | 1,513 | 47.7% | 807 | 706 | 53.34% | +347.48 |
| **SHORT** | 1,661 | 52.3% | 1,094 | 567 | 65.86% | +1,236.49 |

### Rincian Rata-rata Profitabilitas per Trade

| Tipe Trade | PnL Rata-rata ($) | PnL Rata-rata (%) |
|:---|:---:|:---:|
| **Trade Menang (Wins)** | `$+2.0419` | `+8.17%` |
| **Trade Kalah (Losses)** | `$-1.8049` | `-7.22%` |

## 📅 Scorecard Bulanan Portofolio

| Bulan | Total Trades | Wins | Losses | Win Rate | Net PnL ($) |
|:---|:---:|:---:|:---:|:---:|:---:|
| 2025-11 | 613 | 347 | 266 | 56.61% | $+236.36 |
| 2025-12 | 628 | 414 | 214 | 65.92% | $+528.07 |
| 2026-01 | 612 | 390 | 222 | 63.73% | $+455.77 |
| 2026-02 | 619 | 316 | 303 | 51.05% | $-8.85 |
| 2026-03 | 702 | 434 | 268 | 61.82% | $+372.61 |

## 🚪 Distribusi Alasan Keluar Posisi (Exit Reasons)

| Alasan Exit | Jumlah | Persentase | Wins | Losses | Win Rate | PnL Bersih ($) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| `guardian_exit` | 2,463 | 77.6% | 1,860 | 603 | 75.52% | $+2,869.71 |
| `sl_hit` | 663 | 20.9% | 1 | 662 | 0.15% | $-1,303.27 |
| `time_exit` | 48 | 1.5% | 40 | 8 | 83.33% | $+17.53 |

## 🪙 Scorecard Per Koin (Detailed Assets)

| Token | Win Rate | Trades | LONG WR | SHORT WR | Net PnL ($) | Max DD | Sharpe | Sortino | Calmar | PF |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1000PEPE** | 100.00% | 6 | 100.0% | 100.0% | `$+24.04` | 0.00% | 8.12 | 0.00 | 0.00 | 0.00 |
| **1000SHIB** | 71.43% | 7 | 0.0% | 83.3% | `$+7.59` | 4.88% | 2.30 | 5.41 | 15.16 | 6.98 |
| **ADA** | 64.06% | 128 | 59.0% | 68.7% | `$+114.56` | 51.87% | 6.08 | 18.52 | 21.51 | 2.38 |
| **ARB** | 54.05% | 185 | 50.5% | 59.0% | `$+80.70` | 110.04% | 3.26 | 8.26 | 7.14 | 1.48 |
| **AVAX** | 58.65% | 266 | 52.6% | 63.2% | `$+96.44` | 109.46% | 3.94 | 9.61 | 8.58 | 1.46 |
| **BNB** | 56.25% | 144 | 47.8% | 64.0% | `$+19.58` | 48.46% | 1.60 | 4.97 | 3.94 | 1.21 |
| **BTC** | 0.00% | 0 | 0.0% | 0.0% | `$+0.00` | 0.00% | 0.00 | 0.00 | 0.00 | 0.00 |
| **DOGE** | 85.71% | 14 | 60.0% | 100.0% | `$+24.53` | 13.85% | 4.70 | 48.27 | 17.25 | 8.09 |
| **DOT** | 57.42% | 209 | 48.5% | 65.5% | `$+66.73` | 117.81% | 3.09 | 7.37 | 5.52 | 1.41 |
| **ETH** | 70.21% | 141 | 66.1% | 73.2% | `$+124.20` | 35.74% | 7.60 | 24.08 | 33.85 | 2.74 |
| **HBAR** | 58.38% | 173 | 50.0% | 68.3% | `$+31.00` | 59.27% | 1.86 | 5.21 | 5.09 | 1.23 |
| **LINK** | 68.06% | 216 | 60.8% | 74.6% | `$+169.58` | 60.59% | 7.85 | 24.18 | 27.26 | 2.26 |
| **NEAR** | 56.59% | 205 | 40.2% | 69.9% | `$+107.09` | 62.81% | 4.49 | 10.93 | 16.61 | 1.66 |
| **ONDO** | 61.41% | 184 | 59.0% | 63.2% | `$+125.30` | 157.30% | 5.55 | 15.10 | 7.76 | 1.92 |
| **POL** | 54.40% | 182 | 50.0% | 59.3% | `$+74.65` | 190.01% | 3.51 | 9.73 | 3.83 | 1.50 |
| **SOL** | 54.51% | 244 | 50.0% | 57.5% | `$+86.54` | 155.25% | 3.44 | 11.39 | 5.43 | 1.42 |
| **SUI** | 63.33% | 210 | 55.2% | 69.1% | `$+146.21` | 190.29% | 5.92 | 15.50 | 7.48 | 1.98 |
| **TAO** | 56.32% | 190 | 51.6% | 65.1% | `$+91.89` | 203.82% | 3.52 | 7.95 | 4.39 | 1.52 |
| **TON** | 56.12% | 196 | 45.6% | 63.2% | `$+66.54` | 115.19% | 3.50 | 9.57 | 5.63 | 1.46 |
| **TRX** | 65.38% | 104 | 62.5% | 71.9% | `$+17.80` | 29.21% | 3.32 | 9.69 | 5.93 | 1.62 |
| **XRP** | 65.88% | 170 | 68.0% | 64.2% | `$+108.99` | 28.03% | 6.97 | 16.78 | 37.87 | 2.31 |

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
