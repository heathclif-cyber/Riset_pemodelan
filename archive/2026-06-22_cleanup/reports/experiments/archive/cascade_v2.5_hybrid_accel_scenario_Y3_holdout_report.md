# 📊 Holdout Backtest Report: `cascade_v2.5_hybrid_accel_scenario_Y3`

**Tanggal Pembuatan**: 2026-06-03 21:41:41 UTC
**Model Run ID**: `cascade_v2.5_hybrid_accel_scenario_Y3`
**Periode Pengujian (Temporal OOS)**: `2025-11-01 - 2026-04-01`

> [!NOTE]
> **Ringkasan Portofolio Eksekutif**:
> *   **Total Net Profit**: **$+1,924.77 USD** (ROI Portofolio: **+91.66%**)
> *   **Rata-rata Win Rate**: **59.82%** | Total Trades: **2,995**
> *   **Rata-rata Max Drawdown (5x)**: **75.70%**
> *   **Risk-Adjusted**: Sharpe: **4.80** | Sortino: **12.98** | Calmar: **13.45** | Profit Factor: **1.88**

## 📈 Performa Scorecard Portofolio

| Metrik Utama | Nilai Portofolio | Catatan |
|:---|:---:|:---|
| **Total Net Profit ($)** | `$+1,924.77` | Akumulasi keuntungan bersih 5x leverage |
| **Portfolio ROI (%)** | `+91.66%` | ROI berdasarkan kapital portofolio $100/koin |
| **Overall Win Rate** | `59.82%` | Rasio kemenangan rata-rata seluruh aset |
| **Total Trades** | `2,995` | Jumlah total posisi yang dieksekusi |
| **Rata-rata Trade / Bulan** | `28.9` | Rata-rata frekuensi trade bulanan portofolio |
| **Rata-rata Trade / Hari** | `0.95` | Rata-rata frekuensi trade harian portofolio |
| **Max Drawdown (5x)** | `75.70%` | Rata-rata penurunan terdalam portofolio |
| **Sharpe Ratio** | `4.80` | Efisiensi profit terhadap volatilitas portofolio |
| **Sortino Ratio** | `12.98` | Efisiensi profit terhadap downside deviation |
| **Calmar Ratio** | `13.45` | Rasio return tahunan terhadap drawdown |
| **Profit Factor** | `1.88` | Rasio gross profit dibagi gross loss |
| **Max Consecutive Loss** | `10` trades | Streak kekalahan beruntun terpanjang |
| **Worst Single Trade PnL** | `-24.90%` | Kerugian terdalam dalam satu trade tunggal |
| **95% Trades Loss Under** | `11.10%` | Nilai risiko (VaR P95) kerugian maksimal |

## ↕️ Analisis Arah Signal (LONG vs SHORT)

| Arah Posisi | Jumlah Trade | Distribusi | Menang | Kalah | Win Rate | PnL Bersih ($) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **LONG** | 853 | 28.5% | 505 | 348 | 59.20% | +548.40 |
| **SHORT** | 2,142 | 71.5% | 1,375 | 767 | 64.19% | +1,376.37 |

### Rincian Rata-rata Profitabilitas per Trade

| Tipe Trade | PnL Rata-rata ($) | PnL Rata-rata (%) |
|:---|:---:|:---:|
| **Trade Menang (Wins)** | `$+2.1051` | `+8.42%` |
| **Trade Kalah (Losses)** | `$-1.8232` | `-7.29%` |

## 📅 Scorecard Bulanan Portofolio

| Bulan | Total Trades | Wins | Losses | Win Rate | Net PnL ($) |
|:---|:---:|:---:|:---:|:---:|:---:|
| 2025-11 | 582 | 361 | 221 | 62.03% | $+429.02 |
| 2025-12 | 598 | 397 | 201 | 66.39% | $+544.84 |
| 2026-01 | 545 | 367 | 178 | 67.34% | $+518.91 |
| 2026-02 | 627 | 340 | 287 | 54.23% | $+60.50 |
| 2026-03 | 643 | 415 | 228 | 64.54% | $+371.50 |

## 🚪 Distribusi Alasan Keluar Posisi (Exit Reasons)

| Alasan Exit | Jumlah | Persentase | Wins | Losses | Win Rate | PnL Bersih ($) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| `guardian_exit` | 2,390 | 79.8% | 1,839 | 551 | 76.95% | $+3,008.92 |
| `sl_hit` | 560 | 18.7% | 1 | 559 | 0.18% | $-1,101.44 |
| `time_exit` | 45 | 1.5% | 40 | 5 | 88.89% | $+17.28 |

## 🪙 Scorecard Per Koin (Detailed Assets)

| Token | Win Rate | Trades | LONG WR | SHORT WR | Net PnL ($) | Max DD | Sharpe | Sortino | Calmar | PF |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1000PEPE** | 63.04% | 46 | 75.0% | 61.9% | `$+28.07` | 81.43% | 2.29 | 4.17 | 3.36 | 1.68 |
| **1000SHIB** | 60.38% | 53 | 30.0% | 67.4% | `$+25.28` | 52.21% | 2.77 | 7.03 | 4.72 | 1.80 |
| **ADA** | 66.44% | 149 | 63.0% | 68.0% | `$+131.85` | 48.52% | 6.39 | 20.95 | 26.47 | 2.31 |
| **ARB** | 58.60% | 157 | 60.3% | 57.3% | `$+112.26` | 97.86% | 4.80 | 11.87 | 11.17 | 1.90 |
| **AVAX** | 60.59% | 236 | 57.4% | 61.9% | `$+127.91` | 139.01% | 5.63 | 15.48 | 8.96 | 1.77 |
| **BNB** | 61.90% | 126 | 63.0% | 61.6% | `$+40.40` | 34.63% | 3.50 | 10.63 | 11.36 | 1.55 |
| **BTC** | 0.00% | 0 | 0.0% | 0.0% | `$+0.00` | 0.00% | 0.00 | 0.00 | 0.00 | 0.00 |
| **DOGE** | 65.52% | 87 | 80.0% | 63.6% | `$+82.94` | 46.48% | 5.55 | 12.97 | 17.38 | 2.77 |
| **DOT** | 62.79% | 172 | 53.8% | 66.7% | `$+113.85` | 50.24% | 5.64 | 13.90 | 22.07 | 1.98 |
| **ETH** | 71.69% | 166 | 72.1% | 71.5% | `$+171.43` | 41.52% | 9.38 | 28.88 | 40.22 | 3.12 |
| **HBAR** | 59.76% | 164 | 51.0% | 63.7% | `$+40.74` | 79.97% | 2.50 | 6.54 | 4.96 | 1.34 |
| **LINK** | 66.85% | 181 | 60.0% | 69.8% | `$+151.58` | 58.57% | 7.23 | 22.16 | 25.21 | 2.29 |
| **NEAR** | 61.90% | 168 | 44.2% | 68.0% | `$+126.21` | 59.84% | 5.69 | 13.21 | 20.54 | 2.02 |
| **ONDO** | 63.64% | 154 | 59.5% | 65.0% | `$+132.78` | 102.11% | 6.18 | 14.78 | 12.67 | 2.22 |
| **POL** | 59.74% | 154 | 59.7% | 59.8% | `$+96.35` | 154.57% | 4.71 | 12.88 | 6.07 | 1.82 |
| **SOL** | 57.21% | 222 | 53.6% | 58.4% | `$+83.48` | 212.96% | 3.54 | 9.82 | 3.82 | 1.46 |
| **SUI** | 66.12% | 183 | 60.8% | 68.2% | `$+147.62` | 105.89% | 6.31 | 16.31 | 13.58 | 2.21 |
| **TAO** | 59.86% | 142 | 54.9% | 62.6% | `$+86.04` | 70.52% | 3.62 | 8.01 | 11.88 | 1.64 |
| **TON** | 63.46% | 156 | 58.8% | 64.8% | `$+120.28` | 69.46% | 6.36 | 20.34 | 16.87 | 2.25 |
| **TRX** | 62.89% | 97 | 63.6% | 62.3% | `$+16.65` | 26.16% | 2.90 | 7.01 | 6.20 | 1.55 |
| **XRP** | 63.74% | 182 | 76.1% | 59.6% | `$+89.07` | 57.65% | 5.77 | 15.70 | 15.05 | 1.91 |

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
