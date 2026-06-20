# 📊 Holdout Backtest Report: `cascade_v2.5_hybrid_accel_scenario_I`

**Tanggal Pembuatan**: 2026-06-02 19:04:17 UTC
**Model Run ID**: `cascade_v2.5_hybrid_accel_scenario_I`
**Periode Pengujian (Temporal OOS)**: `2025-11-01 - 2026-04-01`

> [!NOTE]
> **Ringkasan Portofolio Eksekutif**:
> *   **Total Net Profit**: **$+2,198.54 USD** (ROI Portofolio: **+104.69%**)
> *   **Rata-rata Win Rate**: **63.29%** | Total Trades: **3,460**
> *   **Rata-rata Max Drawdown (5x)**: **79.19%**
> *   **Risk-Adjusted**: Sharpe: **5.55** | Sortino: **15.08** | Calmar: **15.97** | Profit Factor: **2.04**

## 📈 Performa Scorecard Portofolio

| Metrik Utama | Nilai Portofolio | Catatan |
|:---|:---:|:---|
| **Total Net Profit ($)** | `$+2,198.54` | Akumulasi keuntungan bersih 5x leverage |
| **Portfolio ROI (%)** | `+104.69%` | ROI berdasarkan kapital portofolio $100/koin |
| **Overall Win Rate** | `63.29%` | Rasio kemenangan rata-rata seluruh aset |
| **Total Trades** | `3,460` | Jumlah total posisi yang dieksekusi |
| **Rata-rata Trade / Bulan** | `33.4` | Rata-rata frekuensi trade bulanan portofolio |
| **Rata-rata Trade / Hari** | `1.10` | Rata-rata frekuensi trade harian portofolio |
| **Max Drawdown (5x)** | `79.19%` | Rata-rata penurunan terdalam portofolio |
| **Sharpe Ratio** | `5.55` | Efisiensi profit terhadap volatilitas portofolio |
| **Sortino Ratio** | `15.08` | Efisiensi profit terhadap downside deviation |
| **Calmar Ratio** | `15.97` | Rasio return tahunan terhadap drawdown |
| **Profit Factor** | `2.04` | Rasio gross profit dibagi gross loss |
| **Max Consecutive Loss** | `10` trades | Streak kekalahan beruntun terpanjang |
| **Worst Single Trade PnL** | `-24.90%` | Kerugian terdalam dalam satu trade tunggal |
| **95% Trades Loss Under** | `10.70%` | Nilai risiko (VaR P95) kerugian maksimal |

## ↕️ Analisis Arah Signal (LONG vs SHORT)

| Arah Posisi | Jumlah Trade | Distribusi | Menang | Kalah | Win Rate | PnL Bersih ($) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **LONG** | 945 | 27.3% | 575 | 370 | 60.85% | +644.53 |
| **SHORT** | 2,515 | 72.7% | 1,600 | 915 | 63.62% | +1,554.01 |

### Rincian Rata-rata Profitabilitas per Trade

| Tipe Trade | PnL Rata-rata ($) | PnL Rata-rata (%) |
|:---|:---:|:---:|
| **Trade Menang (Wins)** | `$+2.0629` | `+8.25%` |
| **Trade Kalah (Losses)** | `$-1.7808` | `-7.12%` |

## 📅 Scorecard Bulanan Portofolio

| Bulan | Total Trades | Wins | Losses | Win Rate | Net PnL ($) |
|:---|:---:|:---:|:---:|:---:|:---:|
| 2025-11 | 739 | 455 | 284 | 61.57% | $+495.89 |
| 2025-12 | 693 | 477 | 216 | 68.83% | $+714.34 |
| 2026-01 | 634 | 410 | 224 | 64.67% | $+514.16 |
| 2026-02 | 697 | 400 | 297 | 57.39% | $+129.97 |
| 2026-03 | 697 | 433 | 264 | 62.12% | $+344.18 |

## 🚪 Distribusi Alasan Keluar Posisi (Exit Reasons)

| Alasan Exit | Jumlah | Persentase | Wins | Losses | Win Rate | PnL Bersih ($) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| `guardian_exit` | 2,725 | 78.8% | 2,138 | 587 | 78.46% | $+3,520.73 |
| `sl_hit` | 689 | 19.9% | 1 | 688 | 0.15% | $-1,339.13 |
| `time_exit` | 46 | 1.3% | 36 | 10 | 78.26% | $+16.93 |

## 🪙 Scorecard Per Koin (Detailed Assets)

| Token | Win Rate | Trades | LONG WR | SHORT WR | Net PnL ($) | Max DD | Sharpe | Sortino | Calmar | PF |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1000PEPE** | 61.39% | 158 | 55.6% | 62.6% | `$+69.09` | 140.61% | 3.02 | 6.67 | 4.79 | 1.46 |
| **1000SHIB** | 63.20% | 125 | 64.1% | 62.8% | `$+78.53` | 88.55% | 5.15 | 15.02 | 8.64 | 2.16 |
| **ADA** | 65.75% | 146 | 67.4% | 65.0% | `$+134.63` | 46.65% | 6.51 | 23.82 | 28.11 | 2.32 |
| **ARB** | 54.75% | 179 | 56.2% | 53.8% | `$+75.81` | 130.28% | 3.18 | 7.99 | 5.67 | 1.47 |
| **AVAX** | 57.44% | 242 | 57.1% | 57.5% | `$+111.78` | 155.35% | 4.91 | 12.98 | 7.01 | 1.65 |
| **BNB** | 67.77% | 121 | 65.5% | 68.5% | `$+61.58` | 31.93% | 5.68 | 15.49 | 18.78 | 2.06 |
| **BTC** | 71.93% | 114 | 74.1% | 71.3% | `$+78.99` | 43.38% | 7.03 | 19.90 | 17.74 | 2.67 |
| **DOGE** | 63.70% | 135 | 70.0% | 61.9% | `$+116.14` | 55.89% | 6.29 | 15.94 | 20.24 | 2.49 |
| **DOT** | 64.40% | 191 | 51.0% | 69.3% | `$+154.97` | 43.38% | 7.60 | 18.54 | 34.79 | 2.40 |
| **ETH** | 72.03% | 143 | 69.2% | 73.1% | `$+151.21` | 33.92% | 9.11 | 27.84 | 43.42 | 3.38 |
| **HBAR** | 62.18% | 156 | 56.2% | 64.8% | `$+64.78` | 50.67% | 3.87 | 10.97 | 12.45 | 1.60 |
| **LINK** | 65.76% | 184 | 63.6% | 66.4% | `$+140.65` | 84.56% | 6.71 | 21.01 | 16.20 | 2.12 |
| **NEAR** | 62.67% | 217 | 48.0% | 67.1% | `$+169.89` | 105.32% | 6.77 | 16.54 | 15.71 | 2.11 |
| **ONDO** | 59.26% | 162 | 56.8% | 60.2% | `$+108.29` | 97.88% | 5.03 | 12.67 | 10.78 | 1.86 |
| **POL** | 61.90% | 168 | 61.9% | 61.9% | `$+118.14` | 102.21% | 5.88 | 15.47 | 11.26 | 2.04 |
| **SOL** | 61.54% | 195 | 59.6% | 62.2% | `$+103.18` | 147.59% | 4.73 | 12.25 | 6.81 | 1.74 |
| **SUI** | 65.52% | 203 | 66.0% | 65.4% | `$+179.78` | 79.50% | 7.31 | 16.90 | 22.03 | 2.41 |
| **TAO** | 62.12% | 132 | 58.0% | 64.6% | `$+85.52` | 79.34% | 3.77 | 8.03 | 10.50 | 1.70 |
| **TON** | 55.56% | 189 | 50.0% | 57.0% | `$+88.21` | 70.31% | 4.55 | 15.26 | 12.22 | 1.68 |
| **TRX** | 65.57% | 122 | 68.6% | 63.4% | `$+24.48` | 16.24% | 4.08 | 9.94 | 14.68 | 1.73 |
| **XRP** | 64.61% | 178 | 74.4% | 61.9% | `$+82.86` | 59.35% | 5.42 | 13.43 | 13.60 | 1.84 |

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
