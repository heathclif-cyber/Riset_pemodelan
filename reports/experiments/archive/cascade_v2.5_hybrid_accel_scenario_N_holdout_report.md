# 📊 Holdout Backtest Report: `cascade_v2.5_hybrid_accel_scenario_N`

**Tanggal Pembuatan**: 2026-06-02 20:29:33 UTC
**Model Run ID**: `cascade_v2.5_hybrid_accel_scenario_N`
**Periode Pengujian (Temporal OOS)**: `2025-11-01 - 2026-04-01`

> [!NOTE]
> **Ringkasan Portofolio Eksekutif**:
> *   **Total Net Profit**: **$+1,947.50 USD** (ROI Portofolio: **+92.74%**)
> *   **Rata-rata Win Rate**: **60.57%** | Total Trades: **4,888**
> *   **Rata-rata Max Drawdown (5x)**: **106.41%**
> *   **Risk-Adjusted**: Sharpe: **4.35** | Sortino: **12.07** | Calmar: **11.97** | Profit Factor: **1.80**

## 📈 Performa Scorecard Portofolio

| Metrik Utama | Nilai Portofolio | Catatan |
|:---|:---:|:---|
| **Total Net Profit ($)** | `$+1,947.50` | Akumulasi keuntungan bersih 5x leverage |
| **Portfolio ROI (%)** | `+92.74%` | ROI berdasarkan kapital portofolio $100/koin |
| **Overall Win Rate** | `60.57%` | Rasio kemenangan rata-rata seluruh aset |
| **Total Trades** | `4,888` | Jumlah total posisi yang dieksekusi |
| **Rata-rata Trade / Bulan** | `47.2` | Rata-rata frekuensi trade bulanan portofolio |
| **Rata-rata Trade / Hari** | `1.55` | Rata-rata frekuensi trade harian portofolio |
| **Max Drawdown (5x)** | `106.41%` | Rata-rata penurunan terdalam portofolio |
| **Sharpe Ratio** | `4.35` | Efisiensi profit terhadap volatilitas portofolio |
| **Sortino Ratio** | `12.07` | Efisiensi profit terhadap downside deviation |
| **Calmar Ratio** | `11.97` | Rasio return tahunan terhadap drawdown |
| **Profit Factor** | `1.80` | Rasio gross profit dibagi gross loss |
| **Max Consecutive Loss** | `19` trades | Streak kekalahan beruntun terpanjang |
| **Worst Single Trade PnL** | `-24.90%` | Kerugian terdalam dalam satu trade tunggal |
| **95% Trades Loss Under** | `11.30%` | Nilai risiko (VaR P95) kerugian maksimal |

## ↕️ Analisis Arah Signal (LONG vs SHORT)

| Arah Posisi | Jumlah Trade | Distribusi | Menang | Kalah | Win Rate | PnL Bersih ($) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **LONG** | 2,128 | 43.5% | 1,138 | 990 | 53.48% | +448.40 |
| **SHORT** | 2,760 | 56.5% | 1,735 | 1,025 | 62.86% | +1,499.10 |

### Rincian Rata-rata Profitabilitas per Trade

| Tipe Trade | PnL Rata-rata ($) | PnL Rata-rata (%) |
|:---|:---:|:---:|
| **Trade Menang (Wins)** | `$+1.9568` | `+7.83%` |
| **Trade Kalah (Losses)** | `$-1.8235` | `-7.29%` |

## 📅 Scorecard Bulanan Portofolio

| Bulan | Total Trades | Wins | Losses | Win Rate | Net PnL ($) |
|:---|:---:|:---:|:---:|:---:|:---:|
| 2025-11 | 946 | 519 | 427 | 54.86% | $+217.98 |
| 2025-12 | 977 | 632 | 345 | 64.69% | $+684.59 |
| 2026-01 | 966 | 612 | 354 | 63.35% | $+649.76 |
| 2026-02 | 944 | 491 | 453 | 52.01% | $+30.29 |
| 2026-03 | 1055 | 619 | 436 | 58.67% | $+364.88 |

## 🚪 Distribusi Alasan Keluar Posisi (Exit Reasons)

| Alasan Exit | Jumlah | Persentase | Wins | Losses | Win Rate | PnL Bersih ($) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| `guardian_exit` | 3,766 | 77.0% | 2,812 | 954 | 74.67% | $+4,011.45 |
| `sl_hit` | 1,052 | 21.5% | 2 | 1,050 | 0.19% | $-2,090.98 |
| `time_exit` | 70 | 1.4% | 59 | 11 | 84.29% | $+27.03 |

## 🪙 Scorecard Per Koin (Detailed Assets)

| Token | Win Rate | Trades | LONG WR | SHORT WR | Net PnL ($) | Max DD | Sharpe | Sortino | Calmar | PF |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1000PEPE** | 63.08% | 65 | 63.2% | 63.0% | `$+69.98` | 56.55% | 3.87 | 15.92 | 12.05 | 2.18 |
| **1000SHIB** | 62.50% | 64 | 53.3% | 70.6% | `$+29.78` | 40.36% | 2.95 | 6.53 | 7.19 | 1.82 |
| **ADA** | 63.35% | 191 | 60.0% | 66.0% | `$+147.84` | 51.31% | 6.61 | 22.00 | 28.06 | 2.15 |
| **ARB** | 54.48% | 279 | 48.2% | 60.7% | `$+110.81` | 127.52% | 3.82 | 9.73 | 8.46 | 1.44 |
| **AVAX** | 57.70% | 383 | 55.1% | 59.5% | `$+122.31` | 152.31% | 4.36 | 10.69 | 7.82 | 1.41 |
| **BNB** | 55.22% | 230 | 47.6% | 61.6% | `$+20.64` | 71.10% | 1.34 | 4.00 | 2.83 | 1.13 |
| **BTC** | 80.00% | 45 | 93.3% | 73.3% | `$+46.25` | 16.89% | 6.48 | 16.53 | 26.67 | 4.53 |
| **DOGE** | 68.06% | 72 | 74.1% | 64.4% | `$+81.95` | 25.31% | 5.95 | 18.60 | 31.54 | 3.10 |
| **DOT** | 56.83% | 315 | 48.1% | 63.2% | `$+77.02` | 115.29% | 3.01 | 7.72 | 6.51 | 1.31 |
| **ETH** | 63.51% | 211 | 56.1% | 69.9% | `$+120.88` | 150.53% | 5.79 | 16.45 | 7.82 | 1.88 |
| **HBAR** | 56.47% | 232 | 48.7% | 63.9% | `$+26.80` | 70.79% | 1.41 | 3.95 | 3.69 | 1.15 |
| **LINK** | 57.72% | 324 | 54.9% | 59.7% | `$+112.75` | 125.08% | 4.18 | 12.96 | 8.78 | 1.42 |
| **NEAR** | 58.33% | 312 | 44.9% | 67.6% | `$+139.91` | 100.21% | 4.85 | 11.38 | 13.60 | 1.56 |
| **ONDO** | 58.89% | 270 | 54.6% | 61.7% | `$+137.50` | 151.27% | 5.01 | 12.67 | 8.85 | 1.61 |
| **POL** | 57.25% | 269 | 51.6% | 62.4% | `$+113.29` | 141.14% | 4.52 | 12.31 | 7.82 | 1.53 |
| **SOL** | 54.68% | 342 | 48.8% | 57.9% | `$+84.97` | 193.01% | 3.02 | 8.80 | 4.29 | 1.30 |
| **SUI** | 62.85% | 288 | 55.6% | 67.6% | `$+197.04` | 160.35% | 6.83 | 18.12 | 11.97 | 1.95 |
| **TAO** | 55.56% | 270 | 53.0% | 59.6% | `$+80.69` | 265.27% | 2.61 | 5.51 | 2.96 | 1.30 |
| **TON** | 53.90% | 308 | 46.3% | 58.0% | `$+68.46` | 160.11% | 2.98 | 8.73 | 4.16 | 1.29 |
| **TRX** | 67.06% | 170 | 66.3% | 68.2% | `$+31.11` | 21.36% | 4.67 | 13.37 | 14.19 | 1.70 |
| **XRP** | 64.52% | 248 | 64.3% | 64.7% | `$+127.55` | 38.77% | 7.00 | 17.57 | 32.04 | 2.00 |

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
