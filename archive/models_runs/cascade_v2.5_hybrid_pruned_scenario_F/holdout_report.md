# 📊 Holdout Backtest Report: `cascade_v2.5_hybrid_pruned_scenario_F`

**Tanggal Pembuatan**: 2026-06-01 14:36:19 UTC
**Model Run ID**: `cascade_v2.5_hybrid_pruned_scenario_F`
**Periode Pengujian (Temporal OOS)**: `2025-11-01 - 2026-04-01`

> [!NOTE]
> **Ringkasan Portofolio Eksekutif**:
> *   **Total Net Profit**: **$-299.80 USD** (ROI Portofolio: **-14.28%**)
> *   **Rata-rata Win Rate**: **52.49%** | Total Trades: **11,408**
> *   **Rata-rata Max Drawdown (5x)**: **360.45%**
> *   **Risk-Adjusted**: Sharpe: **-0.66** | Sortino: **-1.42** | Calmar: **-0.13** | Profit Factor: **0.96**

## 📈 Performa Scorecard Portofolio

| Metrik Utama | Nilai Portofolio | Catatan |
|:---|:---:|:---|
| **Total Net Profit ($)** | `$-299.80` | Akumulasi keuntungan bersih 5x leverage |
| **Portfolio ROI (%)** | `-14.28%` | ROI berdasarkan kapital portofolio $100/koin |
| **Overall Win Rate** | `52.49%` | Rasio kemenangan rata-rata seluruh aset |
| **Total Trades** | `11,408` | Jumlah total posisi yang dieksekusi |
| **Rata-rata Trade / Bulan** | `110.2` | Rata-rata frekuensi trade bulanan portofolio |
| **Rata-rata Trade / Hari** | `3.62` | Rata-rata frekuensi trade harian portofolio |
| **Max Drawdown (5x)** | `360.45%` | Rata-rata penurunan terdalam portofolio |
| **Sharpe Ratio** | `-0.66` | Efisiensi profit terhadap volatilitas portofolio |
| **Sortino Ratio** | `-1.42` | Efisiensi profit terhadap downside deviation |
| **Calmar Ratio** | `-0.13` | Rasio return tahunan terhadap drawdown |
| **Profit Factor** | `0.96` | Rasio gross profit dibagi gross loss |
| **Max Consecutive Loss** | `21` trades | Streak kekalahan beruntun terpanjang |
| **Worst Single Trade PnL** | `-29.80%` | Kerugian terdalam dalam satu trade tunggal |
| **95% Trades Loss Under** | `11.90%` | Nilai risiko (VaR P95) kerugian maksimal |

## ↕️ Analisis Arah Signal (LONG vs SHORT)

| Arah Posisi | Jumlah Trade | Distribusi | Menang | Kalah | Win Rate | PnL Bersih ($) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **LONG** | 7,123 | 62.4% | 3,448 | 3,675 | 48.41% | -1,606.99 |
| **SHORT** | 4,285 | 37.6% | 2,564 | 1,721 | 59.84% | +1,307.19 |

### Rincian Rata-rata Profitabilitas per Trade

| Tipe Trade | PnL Rata-rata ($) | PnL Rata-rata (%) |
|:---|:---:|:---:|
| **Trade Menang (Wins)** | `$+1.6268` | `+6.51%` |
| **Trade Kalah (Losses)** | `$-1.8681` | `-7.47%` |

## 📅 Scorecard Bulanan Portofolio

| Bulan | Total Trades | Wins | Losses | Win Rate | Net PnL ($) |
|:---|:---:|:---:|:---:|:---:|:---:|
| 2025-11 | 2144 | 1112 | 1032 | 51.87% | $-352.70 |
| 2025-12 | 2196 | 1286 | 910 | 58.56% | $+544.65 |
| 2026-01 | 2355 | 1220 | 1135 | 51.80% | $-209.12 |
| 2026-02 | 2200 | 1106 | 1094 | 50.27% | $-326.31 |
| 2026-03 | 2513 | 1288 | 1225 | 51.25% | $+43.68 |

## 🚪 Distribusi Alasan Keluar Posisi (Exit Reasons)

| Alasan Exit | Jumlah | Persentase | Wins | Losses | Win Rate | PnL Bersih ($) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| `guardian_exit` | 8,240 | 72.2% | 5,729 | 2,511 | 69.53% | $+5,303.52 |
| `sl_hit` | 2,805 | 24.6% | 5 | 2,800 | 0.18% | $-5,713.69 |
| `time_exit` | 363 | 3.2% | 278 | 85 | 76.58% | $+110.37 |

## 🪙 Scorecard Per Koin (Detailed Assets)

| Token | Win Rate | Trades | LONG WR | SHORT WR | Net PnL ($) | Max DD | Sharpe | Sortino | Calmar | PF |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1000PEPE** | 54.76% | 294 | 44.4% | 72.9% | `$-29.81` | 341.30% | -1.07 | -2.14 | -0.85 | 0.91 |
| **1000SHIB** | 48.44% | 320 | 42.3% | 65.1% | `$-74.50` | 381.64% | -3.06 | -7.55 | -1.90 | 0.78 |
| **ADA** | 54.77% | 535 | 51.8% | 60.0% | `$+85.89` | 286.94% | 2.51 | 6.82 | 2.92 | 1.18 |
| **ARB** | 53.33% | 570 | 49.7% | 59.8% | `$-38.96` | 491.22% | -1.11 | -2.42 | -0.77 | 0.93 |
| **AVAX** | 52.04% | 661 | 45.9% | 59.6% | `$+30.02` | 293.41% | 0.89 | 2.20 | 1.00 | 1.05 |
| **BNB** | 50.73% | 550 | 47.2% | 55.8% | `$-59.75` | 238.99% | -2.56 | -6.06 | -2.44 | 0.85 |
| **BTC** | 47.73% | 308 | 48.0% | 25.0% | `$-68.48` | 297.77% | -3.94 | -9.39 | -2.24 | 0.71 |
| **DOGE** | 53.21% | 421 | 48.3% | 64.6% | `$+60.30` | 219.74% | 2.10 | 5.03 | 2.67 | 1.17 |
| **DOT** | 54.35% | 644 | 48.1% | 64.1% | `$-12.72` | 247.97% | -0.35 | -0.80 | -0.50 | 0.98 |
| **ETH** | 52.14% | 560 | 44.7% | 63.6% | `$-40.83` | 611.50% | -1.26 | -2.70 | -0.65 | 0.92 |
| **HBAR** | 50.86% | 525 | 46.8% | 59.6% | `$-72.59` | 449.64% | -2.42 | -5.74 | -1.57 | 0.85 |
| **LINK** | 54.79% | 668 | 51.6% | 58.9% | `$+84.84` | 228.78% | 2.26 | 6.44 | 3.61 | 1.15 |
| **NEAR** | 56.40% | 656 | 51.5% | 63.6% | `$+114.65` | 379.58% | 2.96 | 6.92 | 2.94 | 1.20 |
| **ONDO** | 52.17% | 575 | 48.8% | 57.0% | `$-68.74` | 610.44% | -1.88 | -4.64 | -1.10 | 0.89 |
| **POL** | 54.72% | 614 | 55.2% | 53.9% | `$+10.87` | 201.22% | 0.32 | 0.70 | 0.53 | 1.02 |
| **SOL** | 50.63% | 630 | 48.1% | 53.6% | `$+5.14` | 382.33% | 0.15 | 0.37 | 0.13 | 1.01 |
| **SUI** | 51.72% | 609 | 45.2% | 61.0% | `$-16.44` | 360.06% | -0.44 | -0.94 | -0.44 | 0.97 |
| **TAO** | 51.94% | 568 | 47.3% | 60.8% | `$-123.98` | 647.53% | -2.98 | -6.01 | -1.86 | 0.82 |
| **TON** | 50.15% | 668 | 48.2% | 52.5% | `$-81.27` | 417.06% | -2.65 | -6.35 | -1.90 | 0.86 |
| **TRX** | 51.75% | 485 | 47.9% | 60.1% | `$-20.18` | 212.31% | -1.85 | -4.86 | -0.93 | 0.88 |
| **XRP** | 55.58% | 547 | 49.9% | 65.5% | `$+16.76` | 269.99% | 0.62 | 1.31 | 0.60 | 1.04 |

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
