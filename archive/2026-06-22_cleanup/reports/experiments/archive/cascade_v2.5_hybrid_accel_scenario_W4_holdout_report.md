# 📊 Holdout Backtest Report: `cascade_v2.5_hybrid_accel_scenario_W4`

**Tanggal Pembuatan**: 2026-06-03 21:15:37 UTC
**Model Run ID**: `cascade_v2.5_hybrid_accel_scenario_W4`
**Periode Pengujian (Temporal OOS)**: `2025-11-01 - 2026-04-01`

> [!NOTE]
> **Ringkasan Portofolio Eksekutif**:
> *   **Total Net Profit**: **$+672.89 USD** (ROI Portofolio: **+32.04%**)
> *   **Rata-rata Win Rate**: **51.40%** | Total Trades: **6,531**
> *   **Rata-rata Max Drawdown (5x)**: **154.77%**
> *   **Risk-Adjusted**: Sharpe: **1.39** | Sortino: **3.83** | Calmar: **2.44** | Profit Factor: **1.18**

## 📈 Performa Scorecard Portofolio

| Metrik Utama | Nilai Portofolio | Catatan |
|:---|:---:|:---|
| **Total Net Profit ($)** | `$+672.89` | Akumulasi keuntungan bersih 5x leverage |
| **Portfolio ROI (%)** | `+32.04%` | ROI berdasarkan kapital portofolio $100/koin |
| **Overall Win Rate** | `51.40%` | Rasio kemenangan rata-rata seluruh aset |
| **Total Trades** | `6,531` | Jumlah total posisi yang dieksekusi |
| **Rata-rata Trade / Bulan** | `63.1` | Rata-rata frekuensi trade bulanan portofolio |
| **Rata-rata Trade / Hari** | `2.07` | Rata-rata frekuensi trade harian portofolio |
| **Max Drawdown (5x)** | `154.77%` | Rata-rata penurunan terdalam portofolio |
| **Sharpe Ratio** | `1.39` | Efisiensi profit terhadap volatilitas portofolio |
| **Sortino Ratio** | `3.83` | Efisiensi profit terhadap downside deviation |
| **Calmar Ratio** | `2.44` | Rasio return tahunan terhadap drawdown |
| **Profit Factor** | `1.18` | Rasio gross profit dibagi gross loss |
| **Max Consecutive Loss** | `16` trades | Streak kekalahan beruntun terpanjang |
| **Worst Single Trade PnL** | `-25.90%` | Kerugian terdalam dalam satu trade tunggal |
| **95% Trades Loss Under** | `11.05%` | Nilai risiko (VaR P95) kerugian maksimal |

## ↕️ Analisis Arah Signal (LONG vs SHORT)

| Arah Posisi | Jumlah Trade | Distribusi | Menang | Kalah | Win Rate | PnL Bersih ($) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **LONG** | 3,054 | 46.8% | 1,392 | 1,662 | 45.58% | -512.33 |
| **SHORT** | 3,477 | 53.2% | 2,044 | 1,433 | 58.79% | +1,185.21 |

### Rincian Rata-rata Profitabilitas per Trade

| Tipe Trade | PnL Rata-rata ($) | PnL Rata-rata (%) |
|:---|:---:|:---:|
| **Trade Menang (Wins)** | `$+1.7953` | `+7.18%` |
| **Trade Kalah (Losses)** | `$-1.7757` | `-7.10%` |

## 📅 Scorecard Bulanan Portofolio

| Bulan | Total Trades | Wins | Losses | Win Rate | Net PnL ($) |
|:---|:---:|:---:|:---:|:---:|:---:|
| 2025-11 | 1059 | 576 | 483 | 54.39% | $+143.04 |
| 2025-12 | 1306 | 715 | 591 | 54.75% | $+346.57 |
| 2026-01 | 1309 | 717 | 592 | 54.77% | $+254.00 |
| 2026-02 | 1308 | 600 | 708 | 45.87% | $-222.29 |
| 2026-03 | 1549 | 828 | 721 | 53.45% | $+151.56 |

## 🚪 Distribusi Alasan Keluar Posisi (Exit Reasons)

| Alasan Exit | Jumlah | Persentase | Wins | Losses | Win Rate | PnL Bersih ($) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| `guardian_exit` | 4,567 | 69.9% | 3,361 | 1,206 | 73.59% | $+4,217.01 |
| `sl_hit` | 1,879 | 28.8% | 2 | 1,877 | 0.11% | $-3,575.42 |
| `time_exit` | 85 | 1.3% | 73 | 12 | 85.88% | $+31.30 |

## 🪙 Scorecard Per Koin (Detailed Assets)

| Token | Win Rate | Trades | LONG WR | SHORT WR | Net PnL ($) | Max DD | Sharpe | Sortino | Calmar | PF |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1000PEPE** | 55.56% | 9 | 0.0% | 55.6% | `$+5.37` | 17.85% | 1.07 | 2.99 | 2.93 | 1.89 |
| **1000SHIB** | 68.18% | 22 | 28.6% | 86.7% | `$+13.18` | 21.53% | 2.66 | 8.12 | 5.96 | 2.39 |
| **ADA** | 53.55% | 310 | 41.1% | 64.6% | `$+60.65` | 178.15% | 2.39 | 6.92 | 3.32 | 1.23 |
| **ARB** | 49.31% | 363 | 42.5% | 56.6% | `$+46.55` | 160.66% | 1.58 | 4.49 | 2.82 | 1.13 |
| **AVAX** | 54.10% | 427 | 48.4% | 58.7% | `$+97.94` | 157.19% | 3.46 | 9.54 | 6.07 | 1.29 |
| **BNB** | 52.67% | 412 | 47.0% | 57.1% | `$-2.95` | 147.65% | -0.15 | -0.39 | -0.19 | 0.99 |
| **BTC** | 0.00% | 0 | 0.0% | 0.0% | `$+0.00` | 0.00% | 0.00 | 0.00 | 0.00 | 0.00 |
| **DOGE** | 59.46% | 37 | 37.5% | 65.5% | `$+5.78` | 59.93% | 0.89 | 1.92 | 0.94 | 1.26 |
| **DOT** | 49.88% | 423 | 39.5% | 59.2% | `$-12.38` | 246.66% | -0.41 | -1.02 | -0.49 | 0.97 |
| **ETH** | 56.45% | 310 | 42.7% | 63.3% | `$+60.04` | 197.50% | 2.55 | 6.45 | 2.96 | 1.25 |
| **HBAR** | 49.86% | 363 | 43.2% | 57.9% | `$-5.45` | 123.54% | -0.23 | -0.61 | -0.43 | 0.98 |
| **LINK** | 53.02% | 447 | 49.0% | 56.4% | `$+46.96` | 161.42% | 1.58 | 4.87 | 2.83 | 1.12 |
| **NEAR** | 54.31% | 418 | 42.4% | 65.6% | `$+91.49` | 242.51% | 2.92 | 7.36 | 3.67 | 1.25 |
| **ONDO** | 55.04% | 367 | 50.9% | 58.4% | `$+99.14` | 148.54% | 3.30 | 9.04 | 6.50 | 1.30 |
| **POL** | 50.96% | 363 | 49.2% | 52.7% | `$+60.59` | 184.78% | 2.20 | 6.06 | 3.19 | 1.19 |
| **SOL** | 47.10% | 431 | 42.2% | 50.6% | `$-35.35` | 367.97% | -1.22 | -3.55 | -0.94 | 0.91 |
| **SUI** | 57.25% | 393 | 49.1% | 63.5% | `$+112.94` | 179.69% | 3.58 | 9.82 | 6.12 | 1.33 |
| **TAO** | 50.47% | 424 | 43.4% | 59.3% | `$-40.59` | 306.18% | -1.17 | -2.61 | -1.29 | 0.91 |
| **TON** | 48.51% | 437 | 45.6% | 50.8% | `$-29.94` | 170.42% | -1.18 | -3.33 | -1.71 | 0.92 |
| **TRX** | 56.94% | 216 | 49.3% | 70.0% | `$+9.61` | 62.65% | 1.29 | 3.62 | 1.49 | 1.14 |
| **XRP** | 56.82% | 359 | 51.8% | 61.1% | `$+89.28` | 115.36% | 3.97 | 10.65 | 7.54 | 1.38 |

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
