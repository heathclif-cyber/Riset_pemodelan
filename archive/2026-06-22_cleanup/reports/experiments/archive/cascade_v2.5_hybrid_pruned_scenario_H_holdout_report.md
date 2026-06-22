# 📊 Holdout Backtest Report: `cascade_v2.5_hybrid_pruned_scenario_H`

**Tanggal Pembuatan**: 2026-06-01 14:47:25 UTC
**Model Run ID**: `cascade_v2.5_hybrid_pruned_scenario_H`
**Periode Pengujian (Temporal OOS)**: `2025-11-01 - 2026-04-01`

> [!NOTE]
> **Ringkasan Portofolio Eksekutif**:
> *   **Total Net Profit**: **$-693.94 USD** (ROI Portofolio: **-33.04%**)
> *   **Rata-rata Win Rate**: **52.89%** | Total Trades: **11,118**
> *   **Rata-rata Max Drawdown (5x)**: **398.06%**
> *   **Risk-Adjusted**: Sharpe: **-0.67** | Sortino: **-1.55** | Calmar: **0.82** | Profit Factor: **1.04**

## 📈 Performa Scorecard Portofolio

| Metrik Utama | Nilai Portofolio | Catatan |
|:---|:---:|:---|
| **Total Net Profit ($)** | `$-693.94` | Akumulasi keuntungan bersih 5x leverage |
| **Portfolio ROI (%)** | `-33.04%` | ROI berdasarkan kapital portofolio $100/koin |
| **Overall Win Rate** | `52.89%` | Rasio kemenangan rata-rata seluruh aset |
| **Total Trades** | `11,118` | Jumlah total posisi yang dieksekusi |
| **Rata-rata Trade / Bulan** | `107.4` | Rata-rata frekuensi trade bulanan portofolio |
| **Rata-rata Trade / Hari** | `3.53` | Rata-rata frekuensi trade harian portofolio |
| **Max Drawdown (5x)** | `398.06%` | Rata-rata penurunan terdalam portofolio |
| **Sharpe Ratio** | `-0.67` | Efisiensi profit terhadap volatilitas portofolio |
| **Sortino Ratio** | `-1.55` | Efisiensi profit terhadap downside deviation |
| **Calmar Ratio** | `0.82` | Rasio return tahunan terhadap drawdown |
| **Profit Factor** | `1.04` | Rasio gross profit dibagi gross loss |
| **Max Consecutive Loss** | `18` trades | Streak kekalahan beruntun terpanjang |
| **Worst Single Trade PnL** | `-28.20%` | Kerugian terdalam dalam satu trade tunggal |
| **95% Trades Loss Under** | `11.60%` | Nilai risiko (VaR P95) kerugian maksimal |

## ↕️ Analisis Arah Signal (LONG vs SHORT)

| Arah Posisi | Jumlah Trade | Distribusi | Menang | Kalah | Win Rate | PnL Bersih ($) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **LONG** | 6,456 | 58.1% | 3,066 | 3,390 | 47.49% | -1,473.84 |
| **SHORT** | 4,662 | 41.9% | 2,635 | 2,027 | 56.52% | +779.90 |

### Rincian Rata-rata Profitabilitas per Trade

| Tipe Trade | PnL Rata-rata ($) | PnL Rata-rata (%) |
|:---|:---:|:---:|
| **Trade Menang (Wins)** | `$+1.5987` | `+6.39%` |
| **Trade Kalah (Losses)** | `$-1.8106` | `-7.24%` |

## 📅 Scorecard Bulanan Portofolio

| Bulan | Total Trades | Wins | Losses | Win Rate | Net PnL ($) |
|:---|:---:|:---:|:---:|:---:|:---:|
| 2025-11 | 1890 | 1040 | 850 | 55.03% | $-73.93 |
| 2025-12 | 2205 | 1194 | 1011 | 54.15% | $+333.79 |
| 2026-01 | 2437 | 1247 | 1190 | 51.17% | $-138.19 |
| 2026-02 | 2097 | 989 | 1108 | 47.16% | $-624.19 |
| 2026-03 | 2489 | 1231 | 1258 | 49.46% | $-191.40 |

## 🚪 Distribusi Alasan Keluar Posisi (Exit Reasons)

| Alasan Exit | Jumlah | Persentase | Wins | Losses | Win Rate | PnL Bersih ($) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| `guardian_exit` | 7,592 | 68.3% | 5,428 | 2,164 | 71.50% | $+5,542.33 |
| `sl_hit` | 3,174 | 28.5% | 5 | 3,169 | 0.16% | $-6,343.70 |
| `time_exit` | 352 | 3.2% | 268 | 84 | 76.14% | $+107.43 |

## 🪙 Scorecard Per Koin (Detailed Assets)

| Token | Win Rate | Trades | LONG WR | SHORT WR | Net PnL ($) | Max DD | Sharpe | Sortino | Calmar | PF |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1000PEPE** | 63.56% | 118 | 51.0% | 73.1% | `$+38.21` | 148.84% | 1.88 | 3.66 | 2.50 | 1.30 |
| **1000SHIB** | 62.04% | 108 | 57.1% | 67.3% | `$+51.13` | 56.27% | 3.44 | 7.91 | 8.85 | 1.71 |
| **ADA** | 49.42% | 518 | 42.7% | 61.1% | `$+3.74` | 340.41% | 0.11 | 0.34 | 0.11 | 1.01 |
| **ARB** | 52.29% | 568 | 48.9% | 57.0% | `$-39.95` | 364.61% | -1.21 | -2.79 | -1.07 | 0.92 |
| **AVAX** | 47.93% | 699 | 43.6% | 52.9% | `$-42.95` | 504.59% | -1.25 | -3.08 | -0.83 | 0.93 |
| **BNB** | 49.01% | 604 | 46.0% | 54.3% | `$-87.97` | 384.56% | -4.01 | -10.39 | -2.23 | 0.79 |
| **BTC** | 58.33% | 36 | 58.8% | 50.0% | `$+5.10` | 63.54% | 0.80 | 1.73 | 0.78 | 1.23 |
| **DOGE** | 61.02% | 118 | 56.9% | 66.0% | `$+70.39` | 38.70% | 4.24 | 12.11 | 17.71 | 1.89 |
| **DOT** | 50.48% | 731 | 44.0% | 59.4% | `$-156.86` | 691.45% | -4.14 | -10.36 | -2.21 | 0.79 |
| **ETH** | 51.38% | 508 | 47.5% | 54.8% | `$-17.93` | 535.76% | -0.59 | -1.40 | -0.33 | 0.96 |
| **HBAR** | 51.44% | 486 | 47.6% | 60.1% | `$-21.07` | 216.48% | -0.79 | -2.15 | -0.95 | 0.95 |
| **LINK** | 53.19% | 767 | 52.3% | 54.2% | `$+62.65` | 350.81% | 1.61 | 4.57 | 1.74 | 1.09 |
| **NEAR** | 53.17% | 709 | 46.7% | 61.3% | `$+12.61` | 608.19% | 0.31 | 0.73 | 0.20 | 1.02 |
| **ONDO** | 50.63% | 634 | 50.7% | 50.5% | `$-68.38` | 576.09% | -1.89 | -4.80 | -1.16 | 0.89 |
| **POL** | 53.28% | 580 | 52.5% | 54.5% | `$-11.06` | 306.28% | -0.33 | -0.81 | -0.35 | 0.98 |
| **SOL** | 47.80% | 728 | 45.7% | 50.1% | `$-72.43` | 774.67% | -2.02 | -5.14 | -0.91 | 0.89 |
| **SUI** | 51.76% | 709 | 44.6% | 62.5% | `$-17.17` | 383.83% | -0.44 | -1.02 | -0.44 | 0.97 |
| **TAO** | 45.82% | 622 | 41.4% | 55.7% | `$-261.49` | 1052.67% | -6.17 | -13.06 | -2.42 | 0.68 |
| **TON** | 49.02% | 763 | 48.7% | 49.4% | `$-145.86` | 618.46% | -4.63 | -11.23 | -2.30 | 0.78 |
| **TRX** | 55.84% | 591 | 53.0% | 59.5% | `$+15.80` | 155.18% | 1.35 | 3.52 | 0.99 | 1.09 |
| **XRP** | 53.36% | 521 | 48.3% | 61.9% | `$-10.44` | 187.89% | -0.42 | -0.93 | -0.54 | 0.97 |

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
