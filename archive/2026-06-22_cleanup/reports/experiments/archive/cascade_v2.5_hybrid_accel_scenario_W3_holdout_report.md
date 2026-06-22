# 📊 Holdout Backtest Report: `cascade_v2.5_hybrid_accel_scenario_W3`

**Tanggal Pembuatan**: 2026-06-03 21:15:02 UTC
**Model Run ID**: `cascade_v2.5_hybrid_accel_scenario_W3`
**Periode Pengujian (Temporal OOS)**: `2025-11-01 - 2026-04-01`

> [!NOTE]
> **Ringkasan Portofolio Eksekutif**:
> *   **Total Net Profit**: **$+257.23 USD** (ROI Portofolio: **+12.25%**)
> *   **Rata-rata Win Rate**: **50.08%** | Total Trades: **8,311**
> *   **Rata-rata Max Drawdown (5x)**: **211.74%**
> *   **Risk-Adjusted**: Sharpe: **0.62** | Sortino: **1.75** | Calmar: **1.38** | Profit Factor: **1.12**

## 📈 Performa Scorecard Portofolio

| Metrik Utama | Nilai Portofolio | Catatan |
|:---|:---:|:---|
| **Total Net Profit ($)** | `$+257.23` | Akumulasi keuntungan bersih 5x leverage |
| **Portfolio ROI (%)** | `+12.25%` | ROI berdasarkan kapital portofolio $100/koin |
| **Overall Win Rate** | `50.08%` | Rasio kemenangan rata-rata seluruh aset |
| **Total Trades** | `8,311` | Jumlah total posisi yang dieksekusi |
| **Rata-rata Trade / Bulan** | `80.3` | Rata-rata frekuensi trade bulanan portofolio |
| **Rata-rata Trade / Hari** | `2.64` | Rata-rata frekuensi trade harian portofolio |
| **Max Drawdown (5x)** | `211.74%` | Rata-rata penurunan terdalam portofolio |
| **Sharpe Ratio** | `0.62` | Efisiensi profit terhadap volatilitas portofolio |
| **Sortino Ratio** | `1.75` | Efisiensi profit terhadap downside deviation |
| **Calmar Ratio** | `1.38` | Rasio return tahunan terhadap drawdown |
| **Profit Factor** | `1.12` | Rasio gross profit dibagi gross loss |
| **Max Consecutive Loss** | `18` trades | Streak kekalahan beruntun terpanjang |
| **Worst Single Trade PnL** | `-25.90%` | Kerugian terdalam dalam satu trade tunggal |
| **95% Trades Loss Under** | `11.15%` | Nilai risiko (VaR P95) kerugian maksimal |

## ↕️ Analisis Arah Signal (LONG vs SHORT)

| Arah Posisi | Jumlah Trade | Distribusi | Menang | Kalah | Win Rate | PnL Bersih ($) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **LONG** | 3,852 | 46.3% | 1,685 | 2,167 | 43.74% | -900.77 |
| **SHORT** | 4,459 | 53.7% | 2,548 | 1,911 | 57.14% | +1,158.01 |

### Rincian Rata-rata Profitabilitas per Trade

| Tipe Trade | PnL Rata-rata ($) | PnL Rata-rata (%) |
|:---|:---:|:---:|
| **Trade Menang (Wins)** | `$+1.7703` | `+7.08%` |
| **Trade Kalah (Losses)** | `$-1.7745` | `-7.10%` |

## 📅 Scorecard Bulanan Portofolio

| Bulan | Total Trades | Wins | Losses | Win Rate | Net PnL ($) |
|:---|:---:|:---:|:---:|:---:|:---:|
| 2025-11 | 1363 | 719 | 644 | 52.75% | $+69.39 |
| 2025-12 | 1707 | 880 | 827 | 51.55% | $+272.44 |
| 2026-01 | 1689 | 903 | 786 | 53.46% | $+207.93 |
| 2026-02 | 1646 | 742 | 904 | 45.08% | $-357.20 |
| 2026-03 | 1906 | 989 | 917 | 51.89% | $+64.67 |

## 🚪 Distribusi Alasan Keluar Posisi (Exit Reasons)

| Alasan Exit | Jumlah | Persentase | Wins | Losses | Win Rate | PnL Bersih ($) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| `guardian_exit` | 5,669 | 68.2% | 4,121 | 1,548 | 72.69% | $+4,978.96 |
| `sl_hit` | 2,511 | 30.2% | 2 | 2,509 | 0.08% | $-4,767.91 |
| `time_exit` | 131 | 1.6% | 110 | 21 | 83.97% | $+46.18 |

## 🪙 Scorecard Per Koin (Detailed Assets)

| Token | Win Rate | Trades | LONG WR | SHORT WR | Net PnL ($) | Max DD | Sharpe | Sortino | Calmar | PF |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1000PEPE** | 54.55% | 11 | 0.0% | 54.5% | `$+10.97` | 17.85% | 1.59 | 6.27 | 5.98 | 2.39 |
| **1000SHIB** | 66.67% | 27 | 28.6% | 80.0% | `$+11.54` | 21.53% | 2.01 | 4.80 | 5.22 | 1.80 |
| **ADA** | 50.61% | 413 | 39.8% | 59.9% | `$+23.53` | 270.78% | 0.81 | 2.39 | 0.85 | 1.06 |
| **ARB** | 47.55% | 469 | 41.0% | 53.8% | `$+8.57` | 272.01% | 0.26 | 0.73 | 0.31 | 1.02 |
| **AVAX** | 51.77% | 537 | 44.2% | 57.6% | `$+61.89` | 233.29% | 2.01 | 5.53 | 2.58 | 1.14 |
| **BNB** | 49.72% | 533 | 44.4% | 53.8% | `$-48.32` | 274.45% | -2.13 | -5.53 | -1.72 | 0.87 |
| **BTC** | 0.00% | 0 | 0.0% | 0.0% | `$+0.00` | 0.00% | 0.00 | 0.00 | 0.00 | 0.00 |
| **DOGE** | 63.83% | 47 | 40.0% | 70.3% | `$+13.98` | 49.74% | 1.98 | 4.28 | 2.74 | 1.56 |
| **DOT** | 48.73% | 550 | 38.6% | 58.0% | `$-56.70` | 369.51% | -1.67 | -4.12 | -1.49 | 0.90 |
| **ETH** | 54.74% | 411 | 41.8% | 60.2% | `$+68.68` | 268.62% | 2.56 | 6.65 | 2.49 | 1.22 |
| **HBAR** | 48.13% | 428 | 41.7% | 56.4% | `$-38.16` | 177.47% | -1.51 | -4.01 | -2.09 | 0.90 |
| **LINK** | 52.36% | 571 | 46.4% | 57.0% | `$+65.48` | 143.94% | 1.97 | 5.71 | 4.43 | 1.14 |
| **NEAR** | 53.74% | 521 | 41.3% | 65.5% | `$+105.88` | 218.08% | 3.07 | 7.66 | 4.73 | 1.24 |
| **ONDO** | 51.98% | 454 | 47.3% | 55.9% | `$+59.21` | 227.95% | 1.81 | 5.19 | 2.53 | 1.14 |
| **POL** | 49.68% | 471 | 46.6% | 52.4% | `$+57.73` | 220.58% | 1.82 | 5.21 | 2.55 | 1.13 |
| **SOL** | 47.51% | 543 | 42.1% | 51.6% | `$-48.91` | 443.79% | -1.53 | -4.35 | -1.07 | 0.90 |
| **SUI** | 54.78% | 502 | 47.0% | 61.4% | `$+68.66` | 285.06% | 1.95 | 5.30 | 2.35 | 1.15 |
| **TAO** | 49.81% | 540 | 43.9% | 57.4% | `$-70.56` | 341.20% | -1.81 | -4.05 | -2.01 | 0.89 |
| **TON** | 46.53% | 548 | 44.4% | 48.2% | `$-78.03` | 329.28% | -2.78 | -7.72 | -2.31 | 0.84 |
| **TRX** | 56.36% | 275 | 49.4% | 67.0% | `$+11.08` | 74.32% | 1.35 | 3.72 | 1.45 | 1.13 |
| **XRP** | 52.61% | 460 | 45.7% | 59.0% | `$+30.72` | 207.03% | 1.22 | 3.10 | 1.45 | 1.09 |

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
