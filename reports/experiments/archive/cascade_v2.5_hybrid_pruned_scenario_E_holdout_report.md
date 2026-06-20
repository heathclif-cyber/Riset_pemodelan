# 📊 Holdout Backtest Report: `cascade_v2.5_hybrid_pruned_scenario_E`

**Tanggal Pembuatan**: 2026-06-01 14:35:47 UTC
**Model Run ID**: `cascade_v2.5_hybrid_pruned_scenario_E`
**Periode Pengujian (Temporal OOS)**: `2025-11-01 - 2026-04-01`

> [!NOTE]
> **Ringkasan Portofolio Eksekutif**:
> *   **Total Net Profit**: **$-285.94 USD** (ROI Portofolio: **-13.62%**)
> *   **Rata-rata Win Rate**: **52.37%** | Total Trades: **11,169**
> *   **Rata-rata Max Drawdown (5x)**: **351.74%**
> *   **Risk-Adjusted**: Sharpe: **-0.63** | Sortino: **-1.37** | Calmar: **-0.14** | Profit Factor: **0.96**

## 📈 Performa Scorecard Portofolio

| Metrik Utama | Nilai Portofolio | Catatan |
|:---|:---:|:---|
| **Total Net Profit ($)** | `$-285.94` | Akumulasi keuntungan bersih 5x leverage |
| **Portfolio ROI (%)** | `-13.62%` | ROI berdasarkan kapital portofolio $100/koin |
| **Overall Win Rate** | `52.37%` | Rasio kemenangan rata-rata seluruh aset |
| **Total Trades** | `11,169` | Jumlah total posisi yang dieksekusi |
| **Rata-rata Trade / Bulan** | `107.9` | Rata-rata frekuensi trade bulanan portofolio |
| **Rata-rata Trade / Hari** | `3.55` | Rata-rata frekuensi trade harian portofolio |
| **Max Drawdown (5x)** | `351.74%` | Rata-rata penurunan terdalam portofolio |
| **Sharpe Ratio** | `-0.63` | Efisiensi profit terhadap volatilitas portofolio |
| **Sortino Ratio** | `-1.37` | Efisiensi profit terhadap downside deviation |
| **Calmar Ratio** | `-0.14` | Rasio return tahunan terhadap drawdown |
| **Profit Factor** | `0.96` | Rasio gross profit dibagi gross loss |
| **Max Consecutive Loss** | `21` trades | Streak kekalahan beruntun terpanjang |
| **Worst Single Trade PnL** | `-29.80%` | Kerugian terdalam dalam satu trade tunggal |
| **95% Trades Loss Under** | `11.90%` | Nilai risiko (VaR P95) kerugian maksimal |

## ↕️ Analisis Arah Signal (LONG vs SHORT)

| Arah Posisi | Jumlah Trade | Distribusi | Menang | Kalah | Win Rate | PnL Bersih ($) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **LONG** | 6,989 | 62.6% | 3,378 | 3,611 | 48.33% | -1,572.69 |
| **SHORT** | 4,180 | 37.4% | 2,497 | 1,683 | 59.74% | +1,286.75 |

### Rincian Rata-rata Profitabilitas per Trade

| Tipe Trade | PnL Rata-rata ($) | PnL Rata-rata (%) |
|:---|:---:|:---:|
| **Trade Menang (Wins)** | `$+1.6339` | `+6.54%` |
| **Trade Kalah (Losses)** | `$-1.8673` | `-7.47%` |

## 📅 Scorecard Bulanan Portofolio

| Bulan | Total Trades | Wins | Losses | Win Rate | Net PnL ($) |
|:---|:---:|:---:|:---:|:---:|:---:|
| 2025-11 | 2090 | 1086 | 1004 | 51.96% | $-336.25 |
| 2025-12 | 2146 | 1258 | 888 | 58.62% | $+546.42 |
| 2026-01 | 2307 | 1190 | 1117 | 51.58% | $-210.75 |
| 2026-02 | 2163 | 1083 | 1080 | 50.07% | $-326.22 |
| 2026-03 | 2463 | 1258 | 1205 | 51.08% | $+40.85 |

## 🚪 Distribusi Alasan Keluar Posisi (Exit Reasons)

| Alasan Exit | Jumlah | Persentase | Wins | Losses | Win Rate | PnL Bersih ($) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| `guardian_exit` | 8,052 | 72.1% | 5,599 | 2,453 | 69.54% | $+5,219.12 |
| `sl_hit` | 2,761 | 24.7% | 5 | 2,756 | 0.18% | $-5,612.43 |
| `time_exit` | 356 | 3.2% | 271 | 85 | 76.12% | $+107.37 |

## 🪙 Scorecard Per Koin (Detailed Assets)

| Token | Win Rate | Trades | LONG WR | SHORT WR | Net PnL ($) | Max DD | Sharpe | Sortino | Calmar | PF |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1000PEPE** | 54.33% | 289 | 44.3% | 72.1% | `$-35.47` | 353.66% | -1.28 | -2.56 | -0.98 | 0.89 |
| **1000SHIB** | 47.59% | 311 | 41.1% | 64.7% | `$-77.93` | 388.78% | -3.22 | -7.96 | -1.95 | 0.76 |
| **ADA** | 54.74% | 517 | 52.4% | 59.0% | `$+94.75` | 284.93% | 2.80 | 7.71 | 3.24 | 1.21 |
| **ARB** | 53.32% | 557 | 49.6% | 60.1% | `$-37.04` | 463.26% | -1.07 | -2.32 | -0.78 | 0.93 |
| **AVAX** | 52.31% | 648 | 46.3% | 59.9% | `$+34.66` | 290.93% | 1.03 | 2.55 | 1.16 | 1.06 |
| **BNB** | 50.19% | 540 | 46.2% | 55.9% | `$-60.40` | 241.58% | -2.60 | -6.25 | -2.44 | 0.85 |
| **BTC** | 48.17% | 301 | 48.5% | 25.0% | `$-65.04` | 290.34% | -3.76 | -9.02 | -2.18 | 0.72 |
| **DOGE** | 52.91% | 412 | 47.9% | 64.3% | `$+53.87` | 217.54% | 1.90 | 4.55 | 2.41 | 1.15 |
| **DOT** | 54.20% | 631 | 47.8% | 64.3% | `$-17.60` | 247.80% | -0.49 | -1.11 | -0.69 | 0.97 |
| **ETH** | 51.65% | 546 | 44.1% | 63.3% | `$-48.00` | 606.12% | -1.49 | -3.21 | -0.77 | 0.90 |
| **HBAR** | 50.39% | 514 | 46.1% | 59.4% | `$-73.12` | 436.32% | -2.45 | -5.83 | -1.63 | 0.85 |
| **LINK** | 54.35% | 655 | 51.1% | 58.6% | `$+72.20` | 220.78% | 1.94 | 5.56 | 3.19 | 1.13 |
| **NEAR** | 56.37% | 644 | 51.2% | 64.1% | `$+118.32` | 364.25% | 3.08 | 7.17 | 3.16 | 1.22 |
| **ONDO** | 52.73% | 567 | 49.5% | 57.3% | `$-53.03` | 561.82% | -1.46 | -3.62 | -0.92 | 0.91 |
| **POL** | 54.65% | 602 | 55.8% | 52.6% | `$+2.59` | 201.57% | 0.08 | 0.17 | 0.13 | 1.00 |
| **SOL** | 50.08% | 615 | 47.8% | 52.9% | `$-1.29` | 402.17% | -0.04 | -0.09 | -0.03 | 1.00 |
| **SUI** | 52.33% | 600 | 46.2% | 61.0% | `$-8.14` | 347.97% | -0.22 | -0.47 | -0.23 | 0.99 |
| **TAO** | 51.62% | 556 | 46.7% | 61.2% | `$-114.46` | 591.95% | -2.79 | -5.68 | -1.88 | 0.83 |
| **TON** | 50.00% | 652 | 47.9% | 52.5% | `$-76.27` | 402.10% | -2.53 | -6.08 | -1.85 | 0.86 |
| **TRX** | 52.10% | 476 | 48.3% | 60.4% | `$-16.81` | 211.18% | -1.55 | -4.09 | -0.78 | 0.90 |
| **XRP** | 55.78% | 536 | 50.1% | 65.6% | `$+22.26` | 261.56% | 0.82 | 1.74 | 0.83 | 1.06 |

## ⛓️ Daftar Fitur Aktif dalam Model

Total terdapat **87 fitur aktif** yang digunakan oleh LightGBM entry, LSTM Soft Confirmation, dan Exit Guardian v3:

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

</details>
