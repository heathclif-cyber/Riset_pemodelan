# 📊 Holdout Backtest Report: `cascade_v2.5_hybrid_accel_scenario_U`

**Tanggal Pembuatan**: 2026-06-03 20:37:50 UTC
**Model Run ID**: `cascade_v2.5_hybrid_accel_scenario_U`
**Periode Pengujian (Temporal OOS)**: `2025-11-01 - 2026-04-01`

> [!NOTE]
> **Ringkasan Portofolio Eksekutif**:
> *   **Total Net Profit**: **$+1,415.94 USD** (ROI Portofolio: **+67.43%**)
> *   **Rata-rata Win Rate**: **58.31%** | Total Trades: **3,772**
> *   **Rata-rata Max Drawdown (5x)**: **103.39%**
> *   **Risk-Adjusted**: Sharpe: **3.46** | Sortino: **9.51** | Calmar: **8.82** | Profit Factor: **1.53**

## 📈 Performa Scorecard Portofolio

| Metrik Utama | Nilai Portofolio | Catatan |
|:---|:---:|:---|
| **Total Net Profit ($)** | `$+1,415.94` | Akumulasi keuntungan bersih 5x leverage |
| **Portfolio ROI (%)** | `+67.43%` | ROI berdasarkan kapital portofolio $100/koin |
| **Overall Win Rate** | `58.31%` | Rasio kemenangan rata-rata seluruh aset |
| **Total Trades** | `3,772` | Jumlah total posisi yang dieksekusi |
| **Rata-rata Trade / Bulan** | `36.5` | Rata-rata frekuensi trade bulanan portofolio |
| **Rata-rata Trade / Hari** | `1.20` | Rata-rata frekuensi trade harian portofolio |
| **Max Drawdown (5x)** | `103.39%` | Rata-rata penurunan terdalam portofolio |
| **Sharpe Ratio** | `3.46` | Efisiensi profit terhadap volatilitas portofolio |
| **Sortino Ratio** | `9.51` | Efisiensi profit terhadap downside deviation |
| **Calmar Ratio** | `8.82` | Rasio return tahunan terhadap drawdown |
| **Profit Factor** | `1.53` | Rasio gross profit dibagi gross loss |
| **Max Consecutive Loss** | `10` trades | Streak kekalahan beruntun terpanjang |
| **Worst Single Trade PnL** | `-24.90%` | Kerugian terdalam dalam satu trade tunggal |
| **95% Trades Loss Under** | `11.14%` | Nilai risiko (VaR P95) kerugian maksimal |

## ↕️ Analisis Arah Signal (LONG vs SHORT)

| Arah Posisi | Jumlah Trade | Distribusi | Menang | Kalah | Win Rate | PnL Bersih ($) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **LONG** | 1,895 | 50.2% | 989 | 906 | 52.19% | +215.79 |
| **SHORT** | 1,877 | 49.8% | 1,208 | 669 | 64.36% | +1,200.15 |

### Rincian Rata-rata Profitabilitas per Trade

| Tipe Trade | PnL Rata-rata ($) | PnL Rata-rata (%) |
|:---|:---:|:---:|
| **Trade Menang (Wins)** | `$+1.9274` | `+7.71%` |
| **Trade Kalah (Losses)** | `$-1.7896` | `-7.16%` |

## 📅 Scorecard Bulanan Portofolio

| Bulan | Total Trades | Wins | Losses | Win Rate | Net PnL ($) |
|:---|:---:|:---:|:---:|:---:|:---:|
| 2025-11 | 808 | 451 | 357 | 55.82% | $+259.68 |
| 2025-12 | 703 | 453 | 250 | 64.44% | $+496.95 |
| 2026-01 | 674 | 427 | 247 | 63.35% | $+431.95 |
| 2026-02 | 730 | 348 | 382 | 47.67% | $-131.85 |
| 2026-03 | 857 | 518 | 339 | 60.44% | $+359.22 |

## 🚪 Distribusi Alasan Keluar Posisi (Exit Reasons)

| Alasan Exit | Jumlah | Persentase | Wins | Losses | Win Rate | PnL Bersih ($) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| `guardian_exit` | 2,857 | 75.7% | 2,147 | 710 | 75.15% | $+3,066.42 |
| `sl_hit` | 861 | 22.8% | 1 | 860 | 0.12% | $-1,672.84 |
| `time_exit` | 54 | 1.4% | 49 | 5 | 90.74% | $+22.37 |

## 🪙 Scorecard Per Koin (Detailed Assets)

| Token | Win Rate | Trades | LONG WR | SHORT WR | Net PnL ($) | Max DD | Sharpe | Sortino | Calmar | PF |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1000PEPE** | 59.69% | 196 | 48.3% | 64.7% | `$+51.67` | 125.55% | 2.14 | 4.44 | 4.01 | 1.27 |
| **1000SHIB** | 54.71% | 170 | 56.1% | 53.4% | `$+30.08` | 122.51% | 1.79 | 4.37 | 2.39 | 1.24 |
| **ADA** | 63.58% | 162 | 60.7% | 67.1% | `$+124.10` | 47.55% | 6.15 | 17.75 | 25.42 | 2.16 |
| **ARB** | 46.96% | 181 | 44.5% | 50.7% | `$+8.85` | 207.58% | 0.40 | 0.96 | 0.42 | 1.05 |
| **AVAX** | 56.71% | 231 | 51.9% | 60.6% | `$+102.21` | 103.57% | 4.47 | 11.97 | 9.61 | 1.59 |
| **BNB** | 58.96% | 134 | 45.5% | 72.1% | `$+27.33` | 41.09% | 2.33 | 7.40 | 6.48 | 1.33 |
| **BTC** | 65.09% | 212 | 58.0% | 71.4% | `$+85.63` | 69.96% | 5.85 | 15.44 | 11.92 | 1.82 |
| **DOGE** | 56.15% | 187 | 51.9% | 59.1% | `$+83.52` | 115.56% | 3.84 | 9.46 | 7.04 | 1.58 |
| **DOT** | 56.99% | 186 | 46.1% | 70.2% | `$+65.70` | 100.07% | 3.10 | 8.24 | 6.39 | 1.43 |
| **ETH** | 67.46% | 169 | 53.2% | 79.3% | `$+131.95` | 42.40% | 7.20 | 25.44 | 30.31 | 2.38 |
| **HBAR** | 56.91% | 181 | 48.0% | 67.9% | `$+46.48` | 63.95% | 2.77 | 7.22 | 7.08 | 1.36 |
| **LINK** | 61.42% | 197 | 54.5% | 68.8% | `$+106.10` | 78.06% | 5.06 | 16.78 | 13.24 | 1.74 |
| **NEAR** | 55.56% | 180 | 44.3% | 66.3% | `$+101.74` | 55.79% | 4.61 | 11.99 | 17.76 | 1.74 |
| **ONDO** | 60.49% | 162 | 54.4% | 66.3% | `$+81.89` | 172.68% | 3.97 | 8.59 | 4.62 | 1.65 |
| **POL** | 53.89% | 180 | 50.0% | 59.5% | `$+52.17` | 177.34% | 2.51 | 5.93 | 2.87 | 1.34 |
| **SOL** | 51.22% | 205 | 48.3% | 53.4% | `$+34.24` | 165.77% | 1.59 | 5.21 | 2.01 | 1.19 |
| **SUI** | 64.89% | 188 | 58.8% | 69.9% | `$+103.46` | 158.18% | 4.57 | 10.85 | 6.37 | 1.76 |
| **TAO** | 58.86% | 175 | 55.6% | 65.5% | `$+80.41` | 94.70% | 3.14 | 7.25 | 8.27 | 1.48 |
| **TON** | 52.38% | 168 | 45.0% | 59.1% | `$+25.88` | 149.95% | 1.54 | 4.31 | 1.68 | 1.19 |
| **TRX** | 62.79% | 129 | 58.8% | 75.0% | `$+9.84` | 37.64% | 1.77 | 5.23 | 2.55 | 1.26 |
| **XRP** | 59.78% | 179 | 60.2% | 59.3% | `$+62.70` | 41.36% | 3.94 | 10.92 | 14.77 | 1.58 |

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
