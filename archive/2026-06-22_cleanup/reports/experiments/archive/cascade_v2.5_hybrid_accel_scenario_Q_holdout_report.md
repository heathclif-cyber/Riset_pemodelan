# 📊 Holdout Backtest Report: `cascade_v2.5_hybrid_accel_scenario_Q`

**Tanggal Pembuatan**: 2026-06-02 20:51:10 UTC
**Model Run ID**: `cascade_v2.5_hybrid_accel_scenario_Q`
**Periode Pengujian (Temporal OOS)**: `2025-11-01 - 2026-04-01`

> [!NOTE]
> **Ringkasan Portofolio Eksekutif**:
> *   **Total Net Profit**: **$+1,834.95 USD** (ROI Portofolio: **+87.38%**)
> *   **Rata-rata Win Rate**: **57.41%** | Total Trades: **4,526**
> *   **Rata-rata Max Drawdown (5x)**: **104.25%**
> *   **Risk-Adjusted**: Sharpe: **4.03** | Sortino: **11.98** | Calmar: **10.13** | Profit Factor: **1.69**

## 📈 Performa Scorecard Portofolio

| Metrik Utama | Nilai Portofolio | Catatan |
|:---|:---:|:---|
| **Total Net Profit ($)** | `$+1,834.95` | Akumulasi keuntungan bersih 5x leverage |
| **Portfolio ROI (%)** | `+87.38%` | ROI berdasarkan kapital portofolio $100/koin |
| **Overall Win Rate** | `57.41%` | Rasio kemenangan rata-rata seluruh aset |
| **Total Trades** | `4,526` | Jumlah total posisi yang dieksekusi |
| **Rata-rata Trade / Bulan** | `43.7` | Rata-rata frekuensi trade bulanan portofolio |
| **Rata-rata Trade / Hari** | `1.44` | Rata-rata frekuensi trade harian portofolio |
| **Max Drawdown (5x)** | `104.25%` | Rata-rata penurunan terdalam portofolio |
| **Sharpe Ratio** | `4.03` | Efisiensi profit terhadap volatilitas portofolio |
| **Sortino Ratio** | `11.98` | Efisiensi profit terhadap downside deviation |
| **Calmar Ratio** | `10.13` | Rasio return tahunan terhadap drawdown |
| **Profit Factor** | `1.69` | Rasio gross profit dibagi gross loss |
| **Max Consecutive Loss** | `19` trades | Streak kekalahan beruntun terpanjang |
| **Worst Single Trade PnL** | `-24.90%` | Kerugian terdalam dalam satu trade tunggal |
| **95% Trades Loss Under** | `11.30%` | Nilai risiko (VaR P95) kerugian maksimal |

## ↕️ Analisis Arah Signal (LONG vs SHORT)

| Arah Posisi | Jumlah Trade | Distribusi | Menang | Kalah | Win Rate | PnL Bersih ($) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **LONG** | 2,001 | 44.2% | 1,060 | 941 | 52.97% | +379.66 |
| **SHORT** | 2,525 | 55.8% | 1,599 | 926 | 63.33% | +1,455.28 |

### Rincian Rata-rata Profitabilitas per Trade

| Tipe Trade | PnL Rata-rata ($) | PnL Rata-rata (%) |
|:---|:---:|:---:|
| **Trade Menang (Wins)** | `$+1.9710` | `+7.88%` |
| **Trade Kalah (Losses)** | `$-1.8243` | `-7.30%` |

## 📅 Scorecard Bulanan Portofolio

| Bulan | Total Trades | Wins | Losses | Win Rate | Net PnL ($) |
|:---|:---:|:---:|:---:|:---:|:---:|
| 2025-11 | 833 | 459 | 374 | 55.10% | $+220.90 |
| 2025-12 | 901 | 577 | 324 | 64.04% | $+637.66 |
| 2026-01 | 905 | 574 | 331 | 63.43% | $+599.41 |
| 2026-02 | 880 | 451 | 429 | 51.25% | $+0.59 |
| 2026-03 | 1007 | 598 | 409 | 59.38% | $+376.39 |

## 🚪 Distribusi Alasan Keluar Posisi (Exit Reasons)

| Alasan Exit | Jumlah | Persentase | Wins | Losses | Win Rate | PnL Bersih ($) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| `guardian_exit` | 3,484 | 77.0% | 2,601 | 883 | 74.66% | $+3,743.90 |
| `sl_hit` | 976 | 21.6% | 2 | 974 | 0.20% | $-1,934.30 |
| `time_exit` | 66 | 1.5% | 56 | 10 | 84.85% | $+25.35 |

## 🪙 Scorecard Per Koin (Detailed Assets)

| Token | Win Rate | Trades | LONG WR | SHORT WR | Net PnL ($) | Max DD | Sharpe | Sortino | Calmar | PF |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1000PEPE** | 63.64% | 22 | 60.0% | 64.7% | `$+40.73` | 31.13% | 3.60 | 29.52 | 12.74 | 3.29 |
| **1000SHIB** | 68.75% | 32 | 46.7% | 88.2% | `$+23.08` | 22.99% | 3.20 | 8.91 | 9.78 | 2.58 |
| **ADA** | 63.69% | 179 | 58.5% | 68.0% | `$+140.68` | 60.41% | 6.47 | 21.00 | 22.68 | 2.17 |
| **ARB** | 55.60% | 259 | 48.9% | 62.5% | `$+122.04` | 103.08% | 4.31 | 10.81 | 11.53 | 1.54 |
| **AVAX** | 58.29% | 362 | 54.7% | 60.9% | `$+121.28` | 147.65% | 4.40 | 10.83 | 8.00 | 1.43 |
| **BNB** | 54.75% | 221 | 47.0% | 61.2% | `$+16.95` | 74.97% | 1.13 | 3.33 | 2.20 | 1.11 |
| **BTC** | 0.00% | 0 | 0.0% | 0.0% | `$+0.00` | 0.00% | 0.00 | 0.00 | 0.00 | 0.00 |
| **DOGE** | 72.22% | 54 | 79.0% | 68.6% | `$+65.04` | 25.09% | 5.39 | 17.02 | 25.25 | 3.21 |
| **DOT** | 57.81% | 301 | 49.2% | 64.2% | `$+85.04` | 117.81% | 3.38 | 8.57 | 7.03 | 1.37 |
| **ETH** | 64.04% | 203 | 56.5% | 70.3% | `$+122.72` | 128.12% | 5.95 | 16.71 | 9.33 | 1.95 |
| **HBAR** | 56.70% | 224 | 48.2% | 64.9% | `$+29.68` | 70.79% | 1.59 | 4.41 | 4.08 | 1.17 |
| **LINK** | 58.09% | 303 | 55.0% | 60.5% | `$+110.62` | 129.79% | 4.22 | 13.14 | 8.30 | 1.45 |
| **NEAR** | 57.58% | 297 | 43.0% | 67.6% | `$+131.22` | 115.01% | 4.65 | 11.08 | 11.11 | 1.55 |
| **ONDO** | 59.62% | 260 | 57.3% | 61.2% | `$+146.30` | 151.27% | 5.47 | 14.05 | 9.42 | 1.71 |
| **POL** | 57.31% | 253 | 51.6% | 62.6% | `$+105.65` | 155.98% | 4.34 | 11.70 | 6.60 | 1.52 |
| **SOL** | 54.57% | 339 | 48.3% | 58.0% | `$+86.65` | 193.01% | 3.09 | 9.02 | 4.37 | 1.30 |
| **SUI** | 62.96% | 270 | 55.5% | 68.1% | `$+186.45` | 180.27% | 6.67 | 17.87 | 10.07 | 1.97 |
| **TAO** | 55.47% | 265 | 52.7% | 60.0% | `$+81.00` | 263.72% | 2.63 | 5.59 | 2.99 | 1.30 |
| **TON** | 54.48% | 290 | 47.1% | 58.6% | `$+66.26` | 156.31% | 2.97 | 8.61 | 4.13 | 1.30 |
| **TRX** | 65.79% | 152 | 65.3% | 66.7% | `$+26.38` | 23.18% | 4.08 | 11.79 | 11.09 | 1.63 |
| **XRP** | 64.17% | 240 | 64.2% | 64.1% | `$+127.15` | 38.77% | 7.02 | 17.60 | 31.94 | 2.01 |

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
