# 📊 Holdout Backtest Report: `cascade_v2.5_hybrid_pruned_scenario_G`

**Tanggal Pembuatan**: 2026-06-01 14:36:45 UTC
**Model Run ID**: `cascade_v2.5_hybrid_pruned_scenario_G`
**Periode Pengujian (Temporal OOS)**: `2025-11-01 - 2026-04-01`

> [!NOTE]
> **Ringkasan Portofolio Eksekutif**:
> *   **Total Net Profit**: **$+306.01 USD** (ROI Portofolio: **+14.57%**)
> *   **Rata-rata Win Rate**: **53.67%** | Total Trades: **8,249**
> *   **Rata-rata Max Drawdown (5x)**: **239.73%**
> *   **Risk-Adjusted**: Sharpe: **0.34** | Sortino: **1.05** | Calmar: **1.12** | Profit Factor: **1.04**

## 📈 Performa Scorecard Portofolio

| Metrik Utama | Nilai Portofolio | Catatan |
|:---|:---:|:---|
| **Total Net Profit ($)** | `$+306.01` | Akumulasi keuntungan bersih 5x leverage |
| **Portfolio ROI (%)** | `+14.57%` | ROI berdasarkan kapital portofolio $100/koin |
| **Overall Win Rate** | `53.67%` | Rasio kemenangan rata-rata seluruh aset |
| **Total Trades** | `8,249` | Jumlah total posisi yang dieksekusi |
| **Rata-rata Trade / Bulan** | `79.7` | Rata-rata frekuensi trade bulanan portofolio |
| **Rata-rata Trade / Hari** | `2.62` | Rata-rata frekuensi trade harian portofolio |
| **Max Drawdown (5x)** | `239.73%` | Rata-rata penurunan terdalam portofolio |
| **Sharpe Ratio** | `0.34` | Efisiensi profit terhadap volatilitas portofolio |
| **Sortino Ratio** | `1.05` | Efisiensi profit terhadap downside deviation |
| **Calmar Ratio** | `1.12` | Rasio return tahunan terhadap drawdown |
| **Profit Factor** | `1.04` | Rasio gross profit dibagi gross loss |
| **Max Consecutive Loss** | `19` trades | Streak kekalahan beruntun terpanjang |
| **Worst Single Trade PnL** | `-29.80%` | Kerugian terdalam dalam satu trade tunggal |
| **95% Trades Loss Under** | `11.80%` | Nilai risiko (VaR P95) kerugian maksimal |

## ↕️ Analisis Arah Signal (LONG vs SHORT)

| Arah Posisi | Jumlah Trade | Distribusi | Menang | Kalah | Win Rate | PnL Bersih ($) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **LONG** | 5,165 | 62.6% | 2,558 | 2,607 | 49.53% | -834.17 |
| **SHORT** | 3,084 | 37.4% | 1,887 | 1,197 | 61.19% | +1,140.18 |

### Rincian Rata-rata Profitabilitas per Trade

| Tipe Trade | PnL Rata-rata ($) | PnL Rata-rata (%) |
|:---|:---:|:---:|
| **Trade Menang (Wins)** | `$+1.6598` | `+6.64%` |
| **Trade Kalah (Losses)** | `$-1.8591` | `-7.44%` |

## 📅 Scorecard Bulanan Portofolio

| Bulan | Total Trades | Wins | Losses | Win Rate | Net PnL ($) |
|:---|:---:|:---:|:---:|:---:|:---:|
| 2025-11 | 1539 | 823 | 716 | 53.48% | $-115.58 |
| 2025-12 | 1544 | 923 | 621 | 59.78% | $+468.30 |
| 2026-01 | 1687 | 897 | 790 | 53.17% | $-40.70 |
| 2026-02 | 1612 | 810 | 802 | 50.25% | $-228.90 |
| 2026-03 | 1867 | 992 | 875 | 53.13% | $+222.89 |

## 🚪 Distribusi Alasan Keluar Posisi (Exit Reasons)

| Alasan Exit | Jumlah | Persentase | Wins | Losses | Win Rate | PnL Bersih ($) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| `guardian_exit` | 6,002 | 72.8% | 4,250 | 1,752 | 70.81% | $+4,272.47 |
| `sl_hit` | 1,991 | 24.1% | 5 | 1,986 | 0.25% | $-4,041.48 |
| `time_exit` | 256 | 3.1% | 190 | 66 | 74.22% | $+75.02 |

## 🪙 Scorecard Per Koin (Detailed Assets)

| Token | Win Rate | Trades | LONG WR | SHORT WR | Net PnL ($) | Max DD | Sharpe | Sortino | Calmar | PF |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1000PEPE** | 54.64% | 183 | 44.4% | 72.1% | `$-16.86` | 239.22% | -0.71 | -1.36 | -0.69 | 0.92 |
| **1000SHIB** | 52.36% | 191 | 45.5% | 68.4% | `$+2.81` | 186.83% | 0.14 | 0.35 | 0.15 | 1.02 |
| **ADA** | 59.04% | 354 | 57.2% | 62.7% | `$+146.36` | 104.09% | 5.16 | 15.17 | 13.70 | 1.53 |
| **ARB** | 54.95% | 424 | 51.4% | 62.3% | `$-16.68` | 294.89% | -0.56 | -1.16 | -0.55 | 0.96 |
| **AVAX** | 52.36% | 508 | 46.6% | 59.4% | `$+32.22` | 192.73% | 1.11 | 2.83 | 1.63 | 1.08 |
| **BNB** | 52.14% | 374 | 47.0% | 60.4% | `$-24.32` | 176.93% | -1.24 | -2.87 | -1.34 | 0.91 |
| **BTC** | 46.11% | 167 | 46.3% | 33.3% | `$-46.00` | 222.94% | -3.70 | -8.54 | -2.01 | 0.65 |
| **DOGE** | 55.47% | 247 | 49.7% | 68.9% | `$+69.92` | 166.20% | 2.94 | 7.18 | 4.10 | 1.34 |
| **DOT** | 54.88% | 523 | 48.8% | 64.7% | `$-0.89` | 201.72% | -0.03 | -0.06 | -0.04 | 1.00 |
| **ETH** | 53.78% | 344 | 46.9% | 64.4% | `$+10.45` | 326.08% | 0.40 | 0.94 | 0.31 | 1.04 |
| **HBAR** | 52.33% | 365 | 47.4% | 62.7% | `$-41.53` | 283.84% | -1.71 | -4.08 | -1.43 | 0.88 |
| **LINK** | 54.10% | 536 | 52.1% | 56.7% | `$+73.28` | 152.10% | 2.17 | 6.50 | 4.69 | 1.16 |
| **NEAR** | 55.32% | 526 | 49.2% | 64.2% | `$+81.17` | 461.79% | 2.26 | 5.39 | 1.71 | 1.18 |
| **ONDO** | 54.00% | 437 | 51.7% | 57.6% | `$+17.57` | 338.73% | 0.55 | 1.35 | 0.51 | 1.04 |
| **POL** | 55.01% | 449 | 56.3% | 52.6% | `$-0.87` | 182.01% | -0.03 | -0.06 | -0.05 | 1.00 |
| **SOL** | 51.73% | 491 | 48.5% | 55.6% | `$+29.86` | 255.16% | 0.97 | 2.36 | 1.14 | 1.07 |
| **SUI** | 54.23% | 485 | 48.1% | 63.3% | `$+56.73` | 260.29% | 1.72 | 3.88 | 2.12 | 1.13 |
| **TAO** | 52.08% | 409 | 47.0% | 63.7% | `$-65.15` | 485.31% | -1.83 | -3.83 | -1.31 | 0.87 |
| **TON** | 51.88% | 532 | 49.5% | 54.8% | `$-28.62` | 239.65% | -1.06 | -2.76 | -1.16 | 0.93 |
| **TRX** | 53.73% | 335 | 48.2% | 65.1% | `$-8.18` | 136.53% | -0.93 | -2.41 | -0.58 | 0.93 |
| **XRP** | 56.91% | 369 | 51.1% | 66.9% | `$+34.73` | 127.34% | 1.61 | 3.26 | 2.66 | 1.14 |

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
