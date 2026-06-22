# 📊 Holdout Backtest Report: `cascade_v2.5_hybrid_accel_scenario_W`

**Tanggal Pembuatan**: 2026-06-03 21:13:05 UTC
**Model Run ID**: `cascade_v2.5_hybrid_accel_scenario_W`
**Periode Pengujian (Temporal OOS)**: `2025-11-01 - 2026-04-01`

> [!NOTE]
> **Ringkasan Portofolio Eksekutif**:
> *   **Total Net Profit**: **$-11,110.97 USD** (ROI Portofolio: **-529.09%**)
> *   **Rata-rata Win Rate**: **41.85%** | Total Trades: **39,840**
> *   **Rata-rata Max Drawdown (5x)**: **2270.70%**
> *   **Risk-Adjusted**: Sharpe: **-8.59** | Sortino: **-21.19** | Calmar: **-2.01** | Profit Factor: **0.70**

## 📈 Performa Scorecard Portofolio

| Metrik Utama | Nilai Portofolio | Catatan |
|:---|:---:|:---|
| **Total Net Profit ($)** | `$-11,110.97` | Akumulasi keuntungan bersih 5x leverage |
| **Portfolio ROI (%)** | `-529.09%` | ROI berdasarkan kapital portofolio $100/koin |
| **Overall Win Rate** | `41.85%` | Rasio kemenangan rata-rata seluruh aset |
| **Total Trades** | `39,840` | Jumlah total posisi yang dieksekusi |
| **Rata-rata Trade / Bulan** | `385.0` | Rata-rata frekuensi trade bulanan portofolio |
| **Rata-rata Trade / Hari** | `12.65` | Rata-rata frekuensi trade harian portofolio |
| **Max Drawdown (5x)** | `2270.70%` | Rata-rata penurunan terdalam portofolio |
| **Sharpe Ratio** | `-8.59` | Efisiensi profit terhadap volatilitas portofolio |
| **Sortino Ratio** | `-21.19` | Efisiensi profit terhadap downside deviation |
| **Calmar Ratio** | `-2.01` | Rasio return tahunan terhadap drawdown |
| **Profit Factor** | `0.70` | Rasio gross profit dibagi gross loss |
| **Max Consecutive Loss** | `62` trades | Streak kekalahan beruntun terpanjang |
| **Worst Single Trade PnL** | `-36.30%` | Kerugian terdalam dalam satu trade tunggal |
| **95% Trades Loss Under** | `11.90%` | Nilai risiko (VaR P95) kerugian maksimal |

## ↕️ Analisis Arah Signal (LONG vs SHORT)

| Arah Posisi | Jumlah Trade | Distribusi | Menang | Kalah | Win Rate | PnL Bersih ($) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **LONG** | 19,871 | 49.9% | 7,566 | 12,305 | 38.08% | -9,246.62 |
| **SHORT** | 19,969 | 50.1% | 9,873 | 10,096 | 49.44% | -1,864.34 |

### Rincian Rata-rata Profitabilitas per Trade

| Tipe Trade | PnL Rata-rata ($) | PnL Rata-rata (%) |
|:---|:---:|:---:|
| **Trade Menang (Wins)** | `$+1.6756` | `+6.70%` |
| **Trade Kalah (Losses)** | `$-1.8004` | `-7.20%` |

## 📅 Scorecard Bulanan Portofolio

| Bulan | Total Trades | Wins | Losses | Win Rate | Net PnL ($) |
|:---|:---:|:---:|:---:|:---:|:---:|
| 2025-11 | 7153 | 3456 | 3697 | 48.32% | $-1,224.63 |
| 2025-12 | 8357 | 3744 | 4613 | 44.80% | $-816.78 |
| 2026-01 | 8251 | 3607 | 4644 | 43.72% | $-1,946.43 |
| 2026-02 | 7457 | 2838 | 4619 | 38.06% | $-4,728.19 |
| 2026-03 | 8622 | 3794 | 4828 | 44.00% | $-2,394.95 |

## 🚪 Distribusi Alasan Keluar Posisi (Exit Reasons)

| Alasan Exit | Jumlah | Persentase | Wins | Losses | Win Rate | PnL Bersih ($) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| `guardian_exit` | 23,863 | 59.9% | 16,476 | 7,387 | 69.04% | $+17,079.12 |
| `sl_hit` | 14,949 | 37.5% | 25 | 14,924 | 0.17% | $-28,637.56 |
| `time_exit` | 1,028 | 2.6% | 938 | 90 | 91.25% | $+447.48 |

## 🪙 Scorecard Per Koin (Detailed Assets)

| Token | Win Rate | Trades | LONG WR | SHORT WR | Net PnL ($) | Max DD | Sharpe | Sortino | Calmar | PF |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1000PEPE** | 43.75% | 48 | 33.3% | 44.4% | `$-15.72` | 115.28% | -1.46 | -3.78 | -1.33 | 0.72 |
| **1000SHIB** | 47.59% | 166 | 35.9% | 54.9% | `$-15.49` | 196.78% | -0.98 | -2.36 | -0.77 | 0.89 |
| **ADA** | 43.04% | 2,126 | 34.9% | 52.4% | `$-620.97` | 2663.15% | -9.94 | -25.26 | -2.27 | 0.72 |
| **ARB** | 43.19% | 2,283 | 37.2% | 49.5% | `$-585.86` | 2555.42% | -8.10 | -20.81 | -2.23 | 0.77 |
| **AVAX** | 45.24% | 2,292 | 39.7% | 50.3% | `$-411.24` | 2165.07% | -6.57 | -16.94 | -1.85 | 0.81 |
| **BNB** | 42.59% | 2,510 | 39.1% | 45.5% | `$-626.36` | 2560.35% | -12.91 | -32.20 | -2.38 | 0.68 |
| **BTC** | 0.00% | 0 | 0.0% | 0.0% | `$+0.00` | 0.00% | 0.00 | 0.00 | 0.00 | 0.00 |
| **DOGE** | 44.05% | 336 | 29.3% | 52.6% | `$-69.59` | 399.60% | -3.57 | -9.63 | -1.70 | 0.74 |
| **DOT** | 43.57% | 2,435 | 35.0% | 53.3% | `$-759.80` | 3289.60% | -10.64 | -25.22 | -2.25 | 0.72 |
| **ETH** | 46.22% | 1,969 | 42.6% | 48.5% | `$-319.75` | 1691.80% | -5.73 | -13.34 | -1.84 | 0.82 |
| **HBAR** | 41.99% | 2,286 | 35.6% | 50.6% | `$-784.89` | 3194.09% | -12.70 | -31.07 | -2.39 | 0.67 |
| **LINK** | 45.59% | 2,395 | 38.9% | 51.8% | `$-508.32` | 2167.33% | -7.86 | -19.57 | -2.28 | 0.78 |
| **NEAR** | 43.69% | 2,369 | 36.6% | 51.7% | `$-798.67` | 3414.66% | -10.42 | -23.64 | -2.28 | 0.72 |
| **ONDO** | 42.55% | 2,402 | 37.0% | 48.2% | `$-873.94` | 3518.15% | -12.50 | -30.55 | -2.42 | 0.68 |
| **POL** | 41.91% | 2,343 | 35.4% | 48.8% | `$-646.55` | 2845.19% | -9.45 | -24.76 | -2.21 | 0.75 |
| **SOL** | 43.03% | 2,359 | 37.2% | 48.0% | `$-771.24` | 3188.40% | -12.37 | -28.63 | -2.36 | 0.68 |
| **SUI** | 45.01% | 2,446 | 39.8% | 50.8% | `$-581.47` | 2588.28% | -7.66 | -18.57 | -2.19 | 0.79 |
| **TAO** | 42.70% | 2,487 | 40.4% | 45.6% | `$-1,179.07` | 4798.26% | -13.95 | -31.46 | -2.39 | 0.65 |
| **TON** | 43.90% | 2,467 | 40.3% | 47.1% | `$-680.04` | 2751.22% | -11.45 | -31.03 | -2.41 | 0.71 |
| **TRX** | 45.21% | 1,922 | 43.5% | 46.6% | `$-237.29` | 1065.10% | -10.73 | -29.01 | -2.17 | 0.70 |
| **XRP** | 43.97% | 2,199 | 37.8% | 51.3% | `$-624.69` | 2517.07% | -11.33 | -27.09 | -2.42 | 0.69 |

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
