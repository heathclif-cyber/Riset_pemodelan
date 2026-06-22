# 📊 Holdout Backtest Report: `cascade_v2.5_hybrid_accel_scenario_M`

**Tanggal Pembuatan**: 2026-06-02 20:29:04 UTC
**Model Run ID**: `cascade_v2.5_hybrid_accel_scenario_M`
**Periode Pengujian (Temporal OOS)**: `2025-11-01 - 2026-04-01`

> [!NOTE]
> **Ringkasan Portofolio Eksekutif**:
> *   **Total Net Profit**: **$+1,926.86 USD** (ROI Portofolio: **+91.76%**)
> *   **Rata-rata Win Rate**: **62.78%** | Total Trades: **4,700**
> *   **Rata-rata Max Drawdown (5x)**: **93.36%**
> *   **Risk-Adjusted**: Sharpe: **4.45** | Sortino: **12.84** | Calmar: **14.27** | Profit Factor: **2.53**

## 📈 Performa Scorecard Portofolio

| Metrik Utama | Nilai Portofolio | Catatan |
|:---|:---:|:---|
| **Total Net Profit ($)** | `$+1,926.86` | Akumulasi keuntungan bersih 5x leverage |
| **Portfolio ROI (%)** | `+91.76%` | ROI berdasarkan kapital portofolio $100/koin |
| **Overall Win Rate** | `62.78%` | Rasio kemenangan rata-rata seluruh aset |
| **Total Trades** | `4,700` | Jumlah total posisi yang dieksekusi |
| **Rata-rata Trade / Bulan** | `45.4` | Rata-rata frekuensi trade bulanan portofolio |
| **Rata-rata Trade / Hari** | `1.49` | Rata-rata frekuensi trade harian portofolio |
| **Max Drawdown (5x)** | `93.36%` | Rata-rata penurunan terdalam portofolio |
| **Sharpe Ratio** | `4.45` | Efisiensi profit terhadap volatilitas portofolio |
| **Sortino Ratio** | `12.84` | Efisiensi profit terhadap downside deviation |
| **Calmar Ratio** | `14.27` | Rasio return tahunan terhadap drawdown |
| **Profit Factor** | `2.53` | Rasio gross profit dibagi gross loss |
| **Max Consecutive Loss** | `17` trades | Streak kekalahan beruntun terpanjang |
| **Worst Single Trade PnL** | `-24.90%` | Kerugian terdalam dalam satu trade tunggal |
| **95% Trades Loss Under** | `10.90%` | Nilai risiko (VaR P95) kerugian maksimal |

## ↕️ Analisis Arah Signal (LONG vs SHORT)

| Arah Posisi | Jumlah Trade | Distribusi | Menang | Kalah | Win Rate | PnL Bersih ($) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **LONG** | 1,866 | 39.7% | 983 | 883 | 52.68% | +349.38 |
| **SHORT** | 2,834 | 60.3% | 1,796 | 1,038 | 63.37% | +1,577.48 |

### Rincian Rata-rata Profitabilitas per Trade

| Tipe Trade | PnL Rata-rata ($) | PnL Rata-rata (%) |
|:---|:---:|:---:|
| **Trade Menang (Wins)** | `$+1.9349` | `+7.74%` |
| **Trade Kalah (Losses)** | `$-1.7960` | `-7.18%` |

## 📅 Scorecard Bulanan Portofolio

| Bulan | Total Trades | Wins | Losses | Win Rate | Net PnL ($) |
|:---|:---:|:---:|:---:|:---:|:---:|
| 2025-11 | 872 | 492 | 380 | 56.42% | $+277.94 |
| 2025-12 | 959 | 599 | 360 | 62.46% | $+608.42 |
| 2026-01 | 934 | 593 | 341 | 63.49% | $+603.20 |
| 2026-02 | 902 | 467 | 435 | 51.77% | $+28.73 |
| 2026-03 | 1033 | 628 | 405 | 60.79% | $+408.57 |

## 🚪 Distribusi Alasan Keluar Posisi (Exit Reasons)

| Alasan Exit | Jumlah | Persentase | Wins | Losses | Win Rate | PnL Bersih ($) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| `guardian_exit` | 3,617 | 77.0% | 2,724 | 893 | 75.31% | $+3,884.39 |
| `sl_hit` | 1,019 | 21.7% | 2 | 1,017 | 0.20% | $-1,980.47 |
| `time_exit` | 64 | 1.4% | 53 | 11 | 82.81% | $+22.94 |

## 🪙 Scorecard Per Koin (Detailed Assets)

| Token | Win Rate | Trades | LONG WR | SHORT WR | Net PnL ($) | Max DD | Sharpe | Sortino | Calmar | PF |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1000PEPE** | 90.48% | 21 | 66.7% | 100.0% | `$+56.43` | 11.61% | 6.73 | 33.74 | 47.34 | 12.29 |
| **1000SHIB** | 60.00% | 15 | 50.0% | 66.7% | `$+5.69` | 7.22% | 1.42 | 3.89 | 7.67 | 1.99 |
| **ADA** | 64.53% | 172 | 57.1% | 69.6% | `$+137.70` | 51.41% | 6.63 | 20.98 | 26.09 | 2.26 |
| **ARB** | 57.61% | 276 | 51.2% | 63.1% | `$+153.08` | 85.52% | 5.39 | 13.72 | 17.43 | 1.69 |
| **AVAX** | 56.85% | 387 | 53.9% | 58.5% | `$+104.72` | 167.87% | 3.71 | 9.06 | 6.08 | 1.34 |
| **BNB** | 54.71% | 223 | 45.9% | 61.6% | `$+12.47` | 81.84% | 0.84 | 2.40 | 1.48 | 1.08 |
| **BTC** | 75.00% | 4 | 0.0% | 75.0% | `$+6.49` | 6.05% | 2.32 | 0.00 | 10.44 | 5.29 |
| **DOGE** | 83.87% | 31 | 80.0% | 87.5% | `$+57.20` | 13.85% | 6.32 | 22.95 | 40.22 | 6.60 |
| **DOT** | 58.13% | 320 | 47.9% | 64.2% | `$+89.22` | 103.57% | 3.52 | 8.98 | 8.39 | 1.37 |
| **ETH** | 66.48% | 179 | 60.9% | 70.0% | `$+120.58` | 115.47% | 6.34 | 16.97 | 10.17 | 2.09 |
| **HBAR** | 58.74% | 223 | 49.1% | 67.5% | `$+47.33` | 51.52% | 2.52 | 7.09 | 8.95 | 1.29 |
| **LINK** | 59.10% | 335 | 58.5% | 59.5% | `$+123.02` | 114.35% | 4.53 | 15.40 | 10.48 | 1.46 |
| **NEAR** | 59.63% | 322 | 43.7% | 69.0% | `$+155.31` | 74.03% | 5.38 | 12.19 | 20.43 | 1.64 |
| **ONDO** | 60.50% | 281 | 61.7% | 59.9% | `$+163.41` | 132.54% | 5.95 | 17.24 | 12.01 | 1.74 |
| **POL** | 54.79% | 261 | 49.1% | 59.3% | `$+106.57` | 164.34% | 4.36 | 12.27 | 6.32 | 1.51 |
| **SOL** | 54.97% | 362 | 50.0% | 57.4% | `$+90.97` | 157.20% | 3.21 | 9.77 | 5.64 | 1.31 |
| **SUI** | 61.84% | 304 | 54.1% | 66.1% | `$+179.86` | 177.44% | 6.42 | 16.08 | 9.87 | 1.83 |
| **TAO** | 54.96% | 262 | 47.7% | 65.1% | `$+83.02` | 262.77% | 2.80 | 6.03 | 3.08 | 1.33 |
| **TON** | 54.85% | 330 | 45.1% | 59.2% | `$+69.11` | 113.32% | 2.97 | 8.35 | 5.94 | 1.28 |
| **TRX** | 66.24% | 157 | 65.2% | 67.7% | `$+30.46` | 20.04% | 4.61 | 13.41 | 14.81 | 1.72 |
| **XRP** | 65.11% | 235 | 61.8% | 67.1% | `$+134.25` | 48.70% | 7.42 | 19.08 | 26.85 | 2.13 |

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
