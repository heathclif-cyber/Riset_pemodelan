# 📊 Holdout Backtest Report: `cascade_v2.5_hybrid_accel_scenario_T4`

**Tanggal Pembuatan**: 2026-06-02 21:12:55 UTC
**Model Run ID**: `cascade_v2.5_hybrid_accel_scenario_T4`
**Periode Pengujian (Temporal OOS)**: `2025-11-01 - 2026-04-01`

> [!NOTE]
> **Ringkasan Portofolio Eksekutif**:
> *   **Total Net Profit**: **$+1,568.74 USD** (ROI Portofolio: **+74.70%**)
> *   **Rata-rata Win Rate**: **60.68%** | Total Trades: **2,846**
> *   **Rata-rata Max Drawdown (5x)**: **79.66%**
> *   **Risk-Adjusted**: Sharpe: **4.30** | Sortino: **11.63** | Calmar: **13.44** | Profit Factor: **2.76**

## 📈 Performa Scorecard Portofolio

| Metrik Utama | Nilai Portofolio | Catatan |
|:---|:---:|:---|
| **Total Net Profit ($)** | `$+1,568.74` | Akumulasi keuntungan bersih 5x leverage |
| **Portfolio ROI (%)** | `+74.70%` | ROI berdasarkan kapital portofolio $100/koin |
| **Overall Win Rate** | `60.68%` | Rasio kemenangan rata-rata seluruh aset |
| **Total Trades** | `2,846` | Jumlah total posisi yang dieksekusi |
| **Rata-rata Trade / Bulan** | `27.5` | Rata-rata frekuensi trade bulanan portofolio |
| **Rata-rata Trade / Hari** | `0.90` | Rata-rata frekuensi trade harian portofolio |
| **Max Drawdown (5x)** | `79.66%` | Rata-rata penurunan terdalam portofolio |
| **Sharpe Ratio** | `4.30` | Efisiensi profit terhadap volatilitas portofolio |
| **Sortino Ratio** | `11.63` | Efisiensi profit terhadap downside deviation |
| **Calmar Ratio** | `13.44` | Rasio return tahunan terhadap drawdown |
| **Profit Factor** | `2.76` | Rasio gross profit dibagi gross loss |
| **Max Consecutive Loss** | `10` trades | Streak kekalahan beruntun terpanjang |
| **Worst Single Trade PnL** | `-24.90%` | Kerugian terdalam dalam satu trade tunggal |
| **95% Trades Loss Under** | `11.20%` | Nilai risiko (VaR P95) kerugian maksimal |

## ↕️ Analisis Arah Signal (LONG vs SHORT)

| Arah Posisi | Jumlah Trade | Distribusi | Menang | Kalah | Win Rate | PnL Bersih ($) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **LONG** | 1,408 | 49.5% | 765 | 643 | 54.33% | +405.51 |
| **SHORT** | 1,438 | 50.5% | 961 | 477 | 66.83% | +1,163.23 |

### Rincian Rata-rata Profitabilitas per Trade

| Tipe Trade | PnL Rata-rata ($) | PnL Rata-rata (%) |
|:---|:---:|:---:|
| **Trade Menang (Wins)** | `$+2.0888` | `+8.36%` |
| **Trade Kalah (Losses)** | `$-1.8183` | `-7.27%` |

## 📅 Scorecard Bulanan Portofolio

| Bulan | Total Trades | Wins | Losses | Win Rate | Net PnL ($) |
|:---|:---:|:---:|:---:|:---:|:---:|
| 2025-11 | 575 | 335 | 240 | 58.26% | $+278.75 |
| 2025-12 | 564 | 373 | 191 | 66.13% | $+502.78 |
| 2026-01 | 532 | 344 | 188 | 64.66% | $+445.29 |
| 2026-02 | 556 | 283 | 273 | 50.90% | $-31.53 |
| 2026-03 | 619 | 391 | 228 | 63.17% | $+373.45 |

## 🚪 Distribusi Alasan Keluar Posisi (Exit Reasons)

| Alasan Exit | Jumlah | Persentase | Wins | Losses | Win Rate | PnL Bersih ($) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| `guardian_exit` | 2,218 | 77.9% | 1,692 | 526 | 76.28% | $+2,720.10 |
| `sl_hit` | 589 | 20.7% | 0 | 589 | 0.00% | $-1,166.85 |
| `time_exit` | 39 | 1.4% | 34 | 5 | 87.18% | $+15.49 |

## 🪙 Scorecard Per Koin (Detailed Assets)

| Token | Win Rate | Trades | LONG WR | SHORT WR | Net PnL ($) | Max DD | Sharpe | Sortino | Calmar | PF |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1000PEPE** | 87.50% | 8 | 50.0% | 100.0% | `$+24.82` | 8.38% | 5.13 | 0.00 | 28.84 | 12.84 |
| **1000SHIB** | 71.43% | 7 | 0.0% | 83.3% | `$+7.59` | 4.88% | 2.30 | 5.41 | 15.16 | 6.98 |
| **ADA** | 66.12% | 121 | 63.3% | 68.8% | `$+117.87` | 32.55% | 6.50 | 19.09 | 35.27 | 2.62 |
| **ARB** | 54.12% | 170 | 52.9% | 55.9% | `$+75.41` | 126.49% | 3.12 | 8.03 | 5.81 | 1.48 |
| **AVAX** | 59.00% | 239 | 52.7% | 64.3% | `$+103.83` | 129.08% | 4.35 | 10.98 | 7.83 | 1.55 |
| **BNB** | 58.40% | 125 | 47.5% | 68.2% | `$+23.69` | 44.24% | 2.03 | 6.65 | 5.22 | 1.29 |
| **BTC** | 0.00% | 0 | 0.0% | 0.0% | `$+0.00` | 0.00% | 0.00 | 0.00 | 0.00 | 0.00 |
| **DOGE** | 82.61% | 23 | 66.7% | 92.9% | `$+46.59` | 13.85% | 5.93 | 26.73 | 32.77 | 7.39 |
| **DOT** | 60.77% | 181 | 49.5% | 72.2% | `$+91.95` | 86.47% | 4.52 | 10.84 | 10.36 | 1.72 |
| **ETH** | 69.92% | 133 | 66.1% | 73.0% | `$+119.03` | 41.52% | 7.51 | 23.44 | 27.92 | 2.77 |
| **HBAR** | 58.33% | 156 | 50.0% | 69.1% | `$+30.55` | 45.96% | 1.91 | 5.37 | 6.47 | 1.26 |
| **LINK** | 66.67% | 189 | 60.4% | 72.5% | `$+145.70` | 68.79% | 7.00 | 21.59 | 20.63 | 2.20 |
| **NEAR** | 57.06% | 177 | 40.5% | 70.4% | `$+104.39` | 62.81% | 4.58 | 11.26 | 16.19 | 1.74 |
| **ONDO** | 60.90% | 156 | 56.9% | 63.7% | `$+104.59` | 159.24% | 4.96 | 12.98 | 6.40 | 1.89 |
| **POL** | 54.94% | 162 | 51.1% | 60.0% | `$+73.08` | 187.66% | 3.54 | 9.95 | 3.79 | 1.54 |
| **SOL** | 55.09% | 216 | 52.7% | 56.9% | `$+82.86` | 134.40% | 3.52 | 11.32 | 6.01 | 1.47 |
| **SUI** | 64.25% | 193 | 56.1% | 70.3% | `$+143.17` | 181.34% | 5.94 | 15.42 | 7.69 | 2.05 |
| **TAO** | 56.98% | 172 | 52.2% | 66.1% | `$+88.85` | 182.62% | 3.55 | 7.89 | 4.74 | 1.56 |
| **TON** | 58.28% | 163 | 47.2% | 67.0% | `$+67.42` | 103.25% | 3.81 | 10.97 | 6.36 | 1.57 |
| **TRX** | 66.34% | 101 | 62.9% | 74.2% | `$+18.24` | 24.38% | 3.51 | 10.20 | 7.29 | 1.68 |
| **XRP** | 65.58% | 154 | 70.0% | 61.9% | `$+99.10` | 35.02% | 6.51 | 16.17 | 27.56 | 2.27 |

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
