# 📊 Holdout Backtest Report: `cascade_v2.5_hybrid_accel_scenario_Y1`

**Tanggal Pembuatan**: 2026-06-03 21:40:59 UTC
**Model Run ID**: `cascade_v2.5_hybrid_accel_scenario_Y1`
**Periode Pengujian (Temporal OOS)**: `2025-11-01 - 2026-04-01`

> [!NOTE]
> **Ringkasan Portofolio Eksekutif**:
> *   **Total Net Profit**: **$+2,074.97 USD** (ROI Portofolio: **+98.81%**)
> *   **Rata-rata Win Rate**: **62.58%** | Total Trades: **3,418**
> *   **Rata-rata Max Drawdown (5x)**: **82.23%**
> *   **Risk-Adjusted**: Sharpe: **5.08** | Sortino: **14.29** | Calmar: **14.01** | Profit Factor: **1.92**

## 📈 Performa Scorecard Portofolio

| Metrik Utama | Nilai Portofolio | Catatan |
|:---|:---:|:---|
| **Total Net Profit ($)** | `$+2,074.97` | Akumulasi keuntungan bersih 5x leverage |
| **Portfolio ROI (%)** | `+98.81%` | ROI berdasarkan kapital portofolio $100/koin |
| **Overall Win Rate** | `62.58%` | Rasio kemenangan rata-rata seluruh aset |
| **Total Trades** | `3,418` | Jumlah total posisi yang dieksekusi |
| **Rata-rata Trade / Bulan** | `33.0` | Rata-rata frekuensi trade bulanan portofolio |
| **Rata-rata Trade / Hari** | `1.09` | Rata-rata frekuensi trade harian portofolio |
| **Max Drawdown (5x)** | `82.23%` | Rata-rata penurunan terdalam portofolio |
| **Sharpe Ratio** | `5.08` | Efisiensi profit terhadap volatilitas portofolio |
| **Sortino Ratio** | `14.29` | Efisiensi profit terhadap downside deviation |
| **Calmar Ratio** | `14.01` | Rasio return tahunan terhadap drawdown |
| **Profit Factor** | `1.92` | Rasio gross profit dibagi gross loss |
| **Max Consecutive Loss** | `10` trades | Streak kekalahan beruntun terpanjang |
| **Worst Single Trade PnL** | `-24.90%` | Kerugian terdalam dalam satu trade tunggal |
| **95% Trades Loss Under** | `11.10%` | Nilai risiko (VaR P95) kerugian maksimal |

## ↕️ Analisis Arah Signal (LONG vs SHORT)

| Arah Posisi | Jumlah Trade | Distribusi | Menang | Kalah | Win Rate | PnL Bersih ($) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **LONG** | 959 | 28.1% | 568 | 391 | 59.23% | +599.77 |
| **SHORT** | 2,459 | 71.9% | 1,565 | 894 | 63.64% | +1,475.20 |

### Rincian Rata-rata Profitabilitas per Trade

| Tipe Trade | PnL Rata-rata ($) | PnL Rata-rata (%) |
|:---|:---:|:---:|
| **Trade Menang (Wins)** | `$+2.0706` | `+8.28%` |
| **Trade Kalah (Losses)** | `$-1.8222` | `-7.29%` |

## 📅 Scorecard Bulanan Portofolio

| Bulan | Total Trades | Wins | Losses | Win Rate | Net PnL ($) |
|:---|:---:|:---:|:---:|:---:|:---:|
| 2025-11 | 676 | 412 | 264 | 60.95% | $+454.99 |
| 2025-12 | 680 | 449 | 231 | 66.03% | $+582.49 |
| 2026-01 | 619 | 414 | 205 | 66.88% | $+554.55 |
| 2026-02 | 727 | 394 | 333 | 54.20% | $+77.05 |
| 2026-03 | 716 | 464 | 252 | 64.80% | $+405.88 |

## 🚪 Distribusi Alasan Keluar Posisi (Exit Reasons)

| Alasan Exit | Jumlah | Persentase | Wins | Losses | Win Rate | PnL Bersih ($) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| `guardian_exit` | 2,717 | 79.5% | 2,083 | 634 | 76.67% | $+3,341.22 |
| `sl_hit` | 647 | 18.9% | 1 | 646 | 0.15% | $-1,286.98 |
| `time_exit` | 54 | 1.6% | 49 | 5 | 90.74% | $+20.73 |

## 🪙 Scorecard Per Koin (Detailed Assets)

| Token | Win Rate | Trades | LONG WR | SHORT WR | Net PnL ($) | Max DD | Sharpe | Sortino | Calmar | PF |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1000PEPE** | 59.62% | 156 | 57.9% | 59.9% | `$+59.69` | 94.55% | 2.68 | 5.98 | 6.15 | 1.40 |
| **1000SHIB** | 60.77% | 130 | 67.7% | 58.6% | `$+60.16` | 83.34% | 4.17 | 11.67 | 7.03 | 1.78 |
| **ADA** | 67.92% | 159 | 64.7% | 69.4% | `$+146.04` | 43.58% | 6.92 | 22.68 | 32.64 | 2.42 |
| **ARB** | 57.58% | 165 | 57.3% | 57.8% | `$+105.93` | 114.76% | 4.45 | 11.15 | 8.99 | 1.78 |
| **AVAX** | 59.75% | 241 | 55.6% | 61.5% | `$+121.07` | 149.73% | 5.26 | 14.09 | 7.88 | 1.70 |
| **BNB** | 63.08% | 130 | 63.0% | 63.1% | `$+46.79` | 34.63% | 4.02 | 12.12 | 13.16 | 1.63 |
| **BTC** | 69.05% | 84 | 0.0% | 69.0% | `$+40.46` | 43.78% | 4.42 | 21.17 | 9.00 | 1.96 |
| **DOGE** | 62.42% | 149 | 76.0% | 59.7% | `$+105.61` | 61.37% | 5.58 | 13.97 | 16.76 | 2.14 |
| **DOT** | 62.07% | 174 | 51.8% | 66.7% | `$+109.68` | 50.24% | 5.40 | 13.50 | 21.26 | 1.91 |
| **ETH** | 71.35% | 171 | 69.6% | 72.0% | `$+177.85` | 49.50% | 9.51 | 30.11 | 35.00 | 3.10 |
| **HBAR** | 60.59% | 170 | 52.8% | 64.1% | `$+46.28` | 75.49% | 2.79 | 7.29 | 5.97 | 1.37 |
| **LINK** | 66.48% | 182 | 60.0% | 69.3% | `$+149.98` | 58.57% | 7.13 | 22.02 | 24.94 | 2.26 |
| **NEAR** | 61.49% | 174 | 45.6% | 67.2% | `$+135.24` | 59.84% | 5.99 | 13.69 | 22.01 | 2.08 |
| **ONDO** | 63.46% | 156 | 59.0% | 65.0% | `$+135.30` | 102.11% | 6.28 | 14.90 | 12.91 | 2.24 |
| **POL** | 60.00% | 165 | 59.7% | 60.2% | `$+101.96` | 154.57% | 4.95 | 12.79 | 6.42 | 1.85 |
| **SOL** | 57.59% | 224 | 53.6% | 58.9% | `$+85.77` | 211.34% | 3.64 | 10.04 | 3.95 | 1.47 |
| **SUI** | 64.74% | 190 | 58.5% | 67.2% | `$+136.31` | 105.89% | 5.67 | 13.78 | 12.54 | 2.01 |
| **TAO** | 59.72% | 144 | 54.9% | 62.4% | `$+88.66` | 79.88% | 3.66 | 8.21 | 10.81 | 1.64 |
| **TON** | 61.96% | 163 | 53.7% | 64.8% | `$+122.55` | 69.46% | 6.36 | 19.79 | 17.18 | 2.21 |
| **TRX** | 61.32% | 106 | 62.7% | 60.0% | `$+14.25` | 26.57% | 2.42 | 6.07 | 5.22 | 1.42 |
| **XRP** | 63.24% | 185 | 76.6% | 58.7% | `$+85.37` | 57.65% | 5.47 | 15.00 | 14.42 | 1.84 |

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
