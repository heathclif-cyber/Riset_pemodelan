# 📊 Holdout Backtest Report: `cascade_v2.5_hybrid_pruned_scenario_D`

**Tanggal Pembuatan**: 2026-06-01 14:35:16 UTC
**Model Run ID**: `cascade_v2.5_hybrid_pruned_scenario_D`
**Periode Pengujian (Temporal OOS)**: `2025-11-01 - 2026-04-01`

> [!NOTE]
> **Ringkasan Portofolio Eksekutif**:
> *   **Total Net Profit**: **$-54.15 USD** (ROI Portofolio: **-2.58%**)
> *   **Rata-rata Win Rate**: **52.69%** | Total Trades: **9,998**
> *   **Rata-rata Max Drawdown (5x)**: **322.29%**
> *   **Risk-Adjusted**: Sharpe: **-0.24** | Sortino: **-0.44** | Calmar: **0.09** | Profit Factor: **0.98**

## 📈 Performa Scorecard Portofolio

| Metrik Utama | Nilai Portofolio | Catatan |
|:---|:---:|:---|
| **Total Net Profit ($)** | `$-54.15` | Akumulasi keuntungan bersih 5x leverage |
| **Portfolio ROI (%)** | `-2.58%` | ROI berdasarkan kapital portofolio $100/koin |
| **Overall Win Rate** | `52.69%` | Rasio kemenangan rata-rata seluruh aset |
| **Total Trades** | `9,998` | Jumlah total posisi yang dieksekusi |
| **Rata-rata Trade / Bulan** | `96.6` | Rata-rata frekuensi trade bulanan portofolio |
| **Rata-rata Trade / Hari** | `3.17` | Rata-rata frekuensi trade harian portofolio |
| **Max Drawdown (5x)** | `322.29%` | Rata-rata penurunan terdalam portofolio |
| **Sharpe Ratio** | `-0.24` | Efisiensi profit terhadap volatilitas portofolio |
| **Sortino Ratio** | `-0.44` | Efisiensi profit terhadap downside deviation |
| **Calmar Ratio** | `0.09` | Rasio return tahunan terhadap drawdown |
| **Profit Factor** | `0.98` | Rasio gross profit dibagi gross loss |
| **Max Consecutive Loss** | `21` trades | Streak kekalahan beruntun terpanjang |
| **Worst Single Trade PnL** | `-29.80%` | Kerugian terdalam dalam satu trade tunggal |
| **95% Trades Loss Under** | `12.00%` | Nilai risiko (VaR P95) kerugian maksimal |

## ↕️ Analisis Arah Signal (LONG vs SHORT)

| Arah Posisi | Jumlah Trade | Distribusi | Menang | Kalah | Win Rate | PnL Bersih ($) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **LONG** | 6,331 | 63.3% | 3,083 | 3,248 | 48.70% | -1,272.14 |
| **SHORT** | 3,667 | 36.7% | 2,206 | 1,461 | 60.16% | +1,217.99 |

### Rincian Rata-rata Profitabilitas per Trade

| Tipe Trade | PnL Rata-rata ($) | PnL Rata-rata (%) |
|:---|:---:|:---:|
| **Trade Menang (Wins)** | `$+1.6518` | `+6.61%` |
| **Trade Kalah (Losses)** | `$-1.8668` | `-7.47%` |

## 📅 Scorecard Bulanan Portofolio

| Bulan | Total Trades | Wins | Losses | Win Rate | Net PnL ($) |
|:---|:---:|:---:|:---:|:---:|:---:|
| 2025-11 | 1877 | 991 | 886 | 52.80% | $-240.29 |
| 2025-12 | 1895 | 1138 | 757 | 60.05% | $+620.05 |
| 2026-01 | 2091 | 1068 | 1023 | 51.08% | $-216.07 |
| 2026-02 | 1976 | 979 | 997 | 49.54% | $-344.93 |
| 2026-03 | 2159 | 1113 | 1046 | 51.55% | $+127.10 |

## 🚪 Distribusi Alasan Keluar Posisi (Exit Reasons)

| Alasan Exit | Jumlah | Persentase | Wins | Losses | Win Rate | PnL Bersih ($) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| `guardian_exit` | 7,173 | 71.7% | 5,035 | 2,138 | 70.19% | $+4,940.56 |
| `sl_hit` | 2,498 | 25.0% | 5 | 2,493 | 0.20% | $-5,095.14 |
| `time_exit` | 327 | 3.3% | 249 | 78 | 76.15% | $+100.44 |

## 🪙 Scorecard Per Koin (Detailed Assets)

| Token | Win Rate | Trades | LONG WR | SHORT WR | Net PnL ($) | Max DD | Sharpe | Sortino | Calmar | PF |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1000PEPE** | 53.73% | 268 | 44.2% | 71.3% | `$-40.36` | 318.84% | -1.50 | -2.96 | -1.23 | 0.87 |
| **1000SHIB** | 48.55% | 276 | 41.2% | 69.4% | `$-50.16` | 319.53% | -2.19 | -5.43 | -1.53 | 0.82 |
| **ADA** | 54.03% | 472 | 51.8% | 58.4% | `$+79.02` | 290.55% | 2.43 | 6.66 | 2.65 | 1.18 |
| **ARB** | 53.37% | 504 | 50.0% | 60.0% | `$-32.70` | 435.96% | -0.99 | -2.11 | -0.73 | 0.93 |
| **AVAX** | 51.45% | 585 | 46.2% | 58.1% | `$+18.13` | 332.19% | 0.56 | 1.38 | 0.53 | 1.04 |
| **BNB** | 51.96% | 460 | 47.4% | 58.4% | `$-27.06` | 144.70% | -1.24 | -2.90 | -1.82 | 0.92 |
| **BTC** | 49.24% | 262 | 49.6% | 25.0% | `$-40.59` | 204.05% | -2.49 | -5.92 | -1.94 | 0.79 |
| **DOGE** | 53.13% | 367 | 47.4% | 66.1% | `$+59.39` | 206.94% | 2.18 | 5.08 | 2.80 | 1.19 |
| **DOT** | 55.50% | 564 | 48.2% | 68.1% | `$+5.93` | 249.34% | 0.17 | 0.40 | 0.23 | 1.01 |
| **ETH** | 51.83% | 492 | 44.7% | 63.2% | `$-39.48` | 611.22% | -1.26 | -2.69 | -0.63 | 0.91 |
| **HBAR** | 51.61% | 467 | 47.1% | 62.1% | `$-41.54` | 353.81% | -1.46 | -3.40 | -1.14 | 0.90 |
| **LINK** | 54.39% | 592 | 51.5% | 58.1% | `$+80.49` | 199.26% | 2.24 | 6.49 | 3.93 | 1.16 |
| **NEAR** | 56.17% | 559 | 51.0% | 63.8% | `$+90.26` | 424.68% | 2.51 | 5.72 | 2.07 | 1.19 |
| **ONDO** | 52.14% | 514 | 49.7% | 55.9% | `$-45.06` | 525.44% | -1.29 | -3.15 | -0.84 | 0.92 |
| **POL** | 56.37% | 534 | 57.5% | 54.2% | `$+39.71` | 167.22% | 1.24 | 2.76 | 2.31 | 1.09 |
| **SOL** | 50.18% | 556 | 48.0% | 52.8% | `$+18.51` | 355.18% | 0.57 | 1.41 | 0.51 | 1.04 |
| **SUI** | 52.46% | 549 | 46.2% | 62.0% | `$+2.81` | 359.22% | 0.08 | 0.17 | 0.08 | 1.01 |
| **TAO** | 52.50% | 480 | 47.7% | 62.4% | `$-66.70` | 505.67% | -1.72 | -3.52 | -1.28 | 0.88 |
| **TON** | 50.34% | 594 | 48.5% | 52.6% | `$-63.12` | 352.54% | -2.20 | -5.25 | -1.74 | 0.87 |
| **TRX** | 52.57% | 428 | 49.1% | 59.7% | `$-10.82` | 174.10% | -1.03 | -2.71 | -0.61 | 0.93 |
| **XRP** | 54.95% | 475 | 49.2% | 65.3% | `$+9.21` | 237.75% | 0.36 | 0.75 | 0.38 | 1.03 |

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
