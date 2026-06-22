# 📊 Holdout Backtest Report: `cascade_v2.5_hybrid_accel_scenario_Z3`

**Tanggal Pembuatan**: 2026-06-03 22:03:10 UTC
**Model Run ID**: `cascade_v2.5_hybrid_accel_scenario_Z3`
**Periode Pengujian (Temporal OOS)**: `2025-11-01 - 2026-04-01`

> [!NOTE]
> **Ringkasan Portofolio Eksekutif**:
> *   **Total Net Profit**: **$+1,518.66 USD** (ROI Portofolio: **+72.32%**)
> *   **Rata-rata Win Rate**: **64.81%** | Total Trades: **2,052**
> *   **Rata-rata Max Drawdown (5x)**: **66.10%**
> *   **Risk-Adjusted**: Sharpe: **4.77** | Sortino: **12.85** | Calmar: **16.66** | Profit Factor: **3.48**

## 📈 Performa Scorecard Portofolio

| Metrik Utama | Nilai Portofolio | Catatan |
|:---|:---:|:---|
| **Total Net Profit ($)** | `$+1,518.66` | Akumulasi keuntungan bersih 5x leverage |
| **Portfolio ROI (%)** | `+72.32%` | ROI berdasarkan kapital portofolio $100/koin |
| **Overall Win Rate** | `64.81%` | Rasio kemenangan rata-rata seluruh aset |
| **Total Trades** | `2,052` | Jumlah total posisi yang dieksekusi |
| **Rata-rata Trade / Bulan** | `19.8` | Rata-rata frekuensi trade bulanan portofolio |
| **Rata-rata Trade / Hari** | `0.65` | Rata-rata frekuensi trade harian portofolio |
| **Max Drawdown (5x)** | `66.10%` | Rata-rata penurunan terdalam portofolio |
| **Sharpe Ratio** | `4.77` | Efisiensi profit terhadap volatilitas portofolio |
| **Sortino Ratio** | `12.85` | Efisiensi profit terhadap downside deviation |
| **Calmar Ratio** | `16.66` | Rasio return tahunan terhadap drawdown |
| **Profit Factor** | `3.48` | Rasio gross profit dibagi gross loss |
| **Max Consecutive Loss** | `9` trades | Streak kekalahan beruntun terpanjang |
| **Worst Single Trade PnL** | `-24.90%` | Kerugian terdalam dalam satu trade tunggal |
| **95% Trades Loss Under** | `10.90%` | Nilai risiko (VaR P95) kerugian maksimal |

## ↕️ Analisis Arah Signal (LONG vs SHORT)

| Arah Posisi | Jumlah Trade | Distribusi | Menang | Kalah | Win Rate | PnL Bersih ($) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **LONG** | 1,091 | 53.2% | 618 | 473 | 56.65% | +517.97 |
| **SHORT** | 961 | 46.8% | 678 | 283 | 70.55% | +1,000.69 |

### Rincian Rata-rata Profitabilitas per Trade

| Tipe Trade | PnL Rata-rata ($) | PnL Rata-rata (%) |
|:---|:---:|:---:|
| **Trade Menang (Wins)** | `$+2.2172` | `+8.87%` |
| **Trade Kalah (Losses)** | `$-1.7921` | `-7.17%` |

## 📅 Scorecard Bulanan Portofolio

| Bulan | Total Trades | Wins | Losses | Win Rate | Net PnL ($) |
|:---|:---:|:---:|:---:|:---:|:---:|
| 2025-11 | 448 | 269 | 179 | 60.04% | $+284.68 |
| 2025-12 | 395 | 279 | 116 | 70.63% | $+493.83 |
| 2026-01 | 386 | 261 | 125 | 67.62% | $+411.52 |
| 2026-02 | 402 | 206 | 196 | 51.24% | $-31.26 |
| 2026-03 | 421 | 281 | 140 | 66.75% | $+359.89 |

## 🚪 Distribusi Alasan Keluar Posisi (Exit Reasons)

| Alasan Exit | Jumlah | Persentase | Wins | Losses | Win Rate | PnL Bersih ($) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| `guardian_exit` | 1,649 | 80.4% | 1,273 | 376 | 77.20% | $+2,232.53 |
| `sl_hit` | 376 | 18.3% | 0 | 376 | 0.00% | $-726.00 |
| `time_exit` | 27 | 1.3% | 23 | 4 | 85.19% | $+12.13 |

## 🪙 Scorecard Per Koin (Detailed Assets)

| Token | Win Rate | Trades | LONG WR | SHORT WR | Net PnL ($) | Max DD | Sharpe | Sortino | Calmar | PF |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1000PEPE** | 94.12% | 17 | 75.0% | 100.0% | `$+51.98` | 8.38% | 7.80 | 0.00 | 60.42 | 25.80 |
| **1000SHIB** | 64.29% | 14 | 60.0% | 66.7% | `$+7.49` | 6.32% | 2.01 | 6.70 | 11.55 | 2.90 |
| **ADA** | 68.54% | 89 | 62.0% | 76.9% | `$+113.88` | 20.87% | 7.10 | 22.24 | 53.15 | 3.43 |
| **ARB** | 54.84% | 124 | 53.2% | 57.8% | `$+75.13` | 99.32% | 3.52 | 8.50 | 7.37 | 1.70 |
| **AVAX** | 61.15% | 157 | 58.8% | 63.6% | `$+81.23` | 137.62% | 4.24 | 11.89 | 5.75 | 1.72 |
| **BNB** | 64.77% | 88 | 57.5% | 70.8% | `$+39.65` | 34.07% | 4.05 | 13.24 | 11.34 | 1.83 |
| **BTC** | 50.00% | 2 | 0.0% | 50.0% | `$+0.84` | 6.05% | 0.34 | 0.00 | 1.36 | 1.56 |
| **DOGE** | 83.87% | 31 | 80.0% | 87.5% | `$+57.20` | 13.85% | 6.32 | 22.95 | 40.22 | 6.60 |
| **DOT** | 61.02% | 118 | 51.6% | 72.2% | `$+73.00` | 80.42% | 4.23 | 10.58 | 8.84 | 1.89 |
| **ETH** | 69.23% | 104 | 68.6% | 69.8% | `$+101.64` | 49.50% | 7.05 | 23.74 | 20.00 | 2.91 |
| **HBAR** | 60.33% | 121 | 54.3% | 68.6% | `$+34.33` | 58.03% | 2.40 | 6.95 | 5.76 | 1.39 |
| **LINK** | 69.29% | 127 | 57.4% | 83.0% | `$+131.78` | 50.67% | 7.31 | 23.69 | 25.33 | 2.76 |
| **NEAR** | 57.94% | 126 | 40.3% | 75.0% | `$+94.93` | 66.61% | 4.53 | 11.55 | 13.88 | 1.92 |
| **ONDO** | 60.19% | 108 | 57.5% | 62.3% | `$+84.29` | 133.86% | 4.54 | 11.95 | 6.13 | 2.01 |
| **POL** | 57.52% | 113 | 53.9% | 64.9% | `$+86.80` | 108.01% | 4.70 | 12.88 | 7.83 | 2.01 |
| **SOL** | 62.86% | 140 | 53.6% | 71.8% | `$+113.39` | 108.40% | 5.77 | 19.27 | 10.19 | 2.20 |
| **SUI** | 64.86% | 148 | 56.1% | 72.0% | `$+134.07` | 132.96% | 6.10 | 14.93 | 9.82 | 2.37 |
| **TAO** | 58.27% | 127 | 53.1% | 67.4% | `$+79.27` | 162.35% | 3.50 | 7.89 | 4.76 | 1.66 |
| **TON** | 60.75% | 107 | 50.0% | 71.7% | `$+60.24` | 60.69% | 4.15 | 11.91 | 9.67 | 1.85 |
| **TRX** | 68.75% | 80 | 66.1% | 75.0% | `$+19.06` | 23.40% | 4.02 | 12.16 | 7.94 | 1.96 |
| **XRP** | 68.47% | 111 | 70.4% | 66.7% | `$+78.45` | 26.71% | 6.57 | 16.90 | 28.61 | 2.62 |

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
