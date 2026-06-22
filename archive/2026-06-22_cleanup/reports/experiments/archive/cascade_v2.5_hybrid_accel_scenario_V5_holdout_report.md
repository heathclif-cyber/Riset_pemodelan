# 📊 Holdout Backtest Report: `cascade_v2.5_hybrid_accel_scenario_V5`

**Tanggal Pembuatan**: 2026-06-03 20:48:18 UTC
**Model Run ID**: `cascade_v2.5_hybrid_accel_scenario_V5`
**Periode Pengujian (Temporal OOS)**: `2025-11-01 - 2026-04-01`

> [!NOTE]
> **Ringkasan Portofolio Eksekutif**:
> *   **Total Net Profit**: **$+1,748.80 USD** (ROI Portofolio: **+83.28%**)
> *   **Rata-rata Win Rate**: **59.85%** | Total Trades: **4,227**
> *   **Rata-rata Max Drawdown (5x)**: **93.55%**
> *   **Risk-Adjusted**: Sharpe: **4.17** | Sortino: **11.28** | Calmar: **12.79** | Profit Factor: **2.71**

## 📈 Performa Scorecard Portofolio

| Metrik Utama | Nilai Portofolio | Catatan |
|:---|:---:|:---|
| **Total Net Profit ($)** | `$+1,748.80` | Akumulasi keuntungan bersih 5x leverage |
| **Portfolio ROI (%)** | `+83.28%` | ROI berdasarkan kapital portofolio $100/koin |
| **Overall Win Rate** | `59.85%` | Rasio kemenangan rata-rata seluruh aset |
| **Total Trades** | `4,227` | Jumlah total posisi yang dieksekusi |
| **Rata-rata Trade / Bulan** | `40.9` | Rata-rata frekuensi trade bulanan portofolio |
| **Rata-rata Trade / Hari** | `1.34` | Rata-rata frekuensi trade harian portofolio |
| **Max Drawdown (5x)** | `93.55%` | Rata-rata penurunan terdalam portofolio |
| **Sharpe Ratio** | `4.17` | Efisiensi profit terhadap volatilitas portofolio |
| **Sortino Ratio** | `11.28` | Efisiensi profit terhadap downside deviation |
| **Calmar Ratio** | `12.79` | Rasio return tahunan terhadap drawdown |
| **Profit Factor** | `2.71` | Rasio gross profit dibagi gross loss |
| **Max Consecutive Loss** | `16` trades | Streak kekalahan beruntun terpanjang |
| **Worst Single Trade PnL** | `-24.90%` | Kerugian terdalam dalam satu trade tunggal |
| **95% Trades Loss Under** | `11.00%` | Nilai risiko (VaR P95) kerugian maksimal |

## ↕️ Analisis Arah Signal (LONG vs SHORT)

| Arah Posisi | Jumlah Trade | Distribusi | Menang | Kalah | Win Rate | PnL Bersih ($) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **LONG** | 1,768 | 41.8% | 927 | 841 | 52.43% | +304.33 |
| **SHORT** | 2,459 | 58.2% | 1,566 | 893 | 63.68% | +1,444.46 |

### Rincian Rata-rata Profitabilitas per Trade

| Tipe Trade | PnL Rata-rata ($) | PnL Rata-rata (%) |
|:---|:---:|:---:|
| **Trade Menang (Wins)** | `$+1.9543` | `+7.82%` |
| **Trade Kalah (Losses)** | `$-1.8012` | `-7.21%` |

## 📅 Scorecard Bulanan Portofolio

| Bulan | Total Trades | Wins | Losses | Win Rate | Net PnL ($) |
|:---|:---:|:---:|:---:|:---:|:---:|
| 2025-11 | 766 | 429 | 337 | 56.01% | $+243.70 |
| 2025-12 | 859 | 540 | 319 | 62.86% | $+560.94 |
| 2026-01 | 838 | 534 | 304 | 63.72% | $+558.30 |
| 2026-02 | 815 | 415 | 400 | 50.92% | $+0.62 |
| 2026-03 | 949 | 575 | 374 | 60.59% | $+385.24 |

## 🚪 Distribusi Alasan Keluar Posisi (Exit Reasons)

| Alasan Exit | Jumlah | Persentase | Wins | Losses | Win Rate | PnL Bersih ($) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| `guardian_exit` | 3,260 | 77.1% | 2,444 | 816 | 74.97% | $+3,510.29 |
| `sl_hit` | 910 | 21.5% | 2 | 908 | 0.22% | $-1,782.09 |
| `time_exit` | 57 | 1.3% | 47 | 10 | 82.46% | $+20.60 |

## 🪙 Scorecard Per Koin (Detailed Assets)

| Token | Win Rate | Trades | LONG WR | SHORT WR | Net PnL ($) | Max DD | Sharpe | Sortino | Calmar | PF |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1000PEPE** | 87.50% | 8 | 50.0% | 100.0% | `$+24.82` | 8.38% | 5.13 | 0.00 | 28.84 | 12.84 |
| **1000SHIB** | 77.78% | 9 | 50.0% | 85.7% | `$+8.62` | 4.88% | 2.63 | 5.42 | 17.21 | 7.79 |
| **ADA** | 64.42% | 163 | 57.4% | 69.5% | `$+131.49` | 51.87% | 6.51 | 20.23 | 24.69 | 2.28 |
| **ARB** | 56.85% | 241 | 51.7% | 62.0% | `$+127.95` | 85.52% | 4.74 | 11.99 | 14.57 | 1.64 |
| **AVAX** | 57.40% | 338 | 53.1% | 60.1% | `$+93.09` | 131.34% | 3.49 | 8.63 | 6.90 | 1.34 |
| **BNB** | 53.62% | 207 | 44.1% | 61.4% | `$+6.81` | 92.79% | 0.47 | 1.35 | 0.71 | 1.05 |
| **BTC** | 0.00% | 0 | 0.0% | 0.0% | `$+0.00` | 0.00% | 0.00 | 0.00 | 0.00 | 0.00 |
| **DOGE** | 85.71% | 28 | 76.9% | 93.3% | `$+59.14` | 13.85% | 7.07 | 30.75 | 41.59 | 9.12 |
| **DOT** | 56.69% | 284 | 47.0% | 63.3% | `$+59.44` | 126.89% | 2.49 | 6.23 | 4.56 | 1.27 |
| **ETH** | 67.65% | 170 | 63.5% | 70.1% | `$+125.63` | 76.29% | 6.88 | 19.14 | 16.04 | 2.28 |
| **HBAR** | 58.45% | 207 | 48.5% | 68.3% | `$+40.35` | 49.48% | 2.25 | 6.09 | 7.94 | 1.26 |
| **LINK** | 61.03% | 290 | 59.3% | 62.2% | `$+144.76` | 93.28% | 5.64 | 19.33 | 15.12 | 1.66 |
| **NEAR** | 58.36% | 293 | 42.9% | 68.0% | `$+123.45` | 84.36% | 4.51 | 10.35 | 14.25 | 1.54 |
| **ONDO** | 60.24% | 254 | 60.9% | 59.9% | `$+150.63` | 147.78% | 5.68 | 16.83 | 9.93 | 1.75 |
| **POL** | 55.46% | 238 | 50.0% | 60.3% | `$+101.58` | 179.02% | 4.32 | 12.15 | 5.53 | 1.54 |
| **SOL** | 56.16% | 333 | 51.3% | 58.8% | `$+106.88` | 152.93% | 3.87 | 12.00 | 6.81 | 1.40 |
| **SUI** | 62.22% | 270 | 52.9% | 68.1% | `$+167.18` | 190.33% | 6.21 | 15.72 | 8.56 | 1.87 |
| **TAO** | 54.62% | 249 | 49.0% | 63.0% | `$+83.84` | 261.37% | 2.89 | 6.27 | 3.12 | 1.35 |
| **TON** | 54.14% | 290 | 44.2% | 59.0% | `$+53.24` | 151.98% | 2.42 | 6.88 | 3.41 | 1.24 |
| **TRX** | 64.71% | 136 | 63.1% | 67.3% | `$+23.01` | 23.29% | 3.69 | 10.59 | 9.62 | 1.60 |
| **XRP** | 63.93% | 219 | 61.6% | 65.4% | `$+116.88` | 38.92% | 6.69 | 16.89 | 29.25 | 2.02 |

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
