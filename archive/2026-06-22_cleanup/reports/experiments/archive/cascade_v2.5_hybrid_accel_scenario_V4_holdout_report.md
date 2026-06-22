# 📊 Holdout Backtest Report: `cascade_v2.5_hybrid_accel_scenario_V4`

**Tanggal Pembuatan**: 2026-06-03 20:47:52 UTC
**Model Run ID**: `cascade_v2.5_hybrid_accel_scenario_V4`
**Periode Pengujian (Temporal OOS)**: `2025-11-01 - 2026-04-01`

> [!NOTE]
> **Ringkasan Portofolio Eksekutif**:
> *   **Total Net Profit**: **$+1,756.93 USD** (ROI Portofolio: **+83.66%**)
> *   **Rata-rata Win Rate**: **59.67%** | Total Trades: **4,240**
> *   **Rata-rata Max Drawdown (5x)**: **93.55%**
> *   **Risk-Adjusted**: Sharpe: **4.17** | Sortino: **10.95** | Calmar: **13.18** | Profit Factor: **2.62**

## 📈 Performa Scorecard Portofolio

| Metrik Utama | Nilai Portofolio | Catatan |
|:---|:---:|:---|
| **Total Net Profit ($)** | `$+1,756.93` | Akumulasi keuntungan bersih 5x leverage |
| **Portfolio ROI (%)** | `+83.66%` | ROI berdasarkan kapital portofolio $100/koin |
| **Overall Win Rate** | `59.67%` | Rasio kemenangan rata-rata seluruh aset |
| **Total Trades** | `4,240` | Jumlah total posisi yang dieksekusi |
| **Rata-rata Trade / Bulan** | `41.0` | Rata-rata frekuensi trade bulanan portofolio |
| **Rata-rata Trade / Hari** | `1.35` | Rata-rata frekuensi trade harian portofolio |
| **Max Drawdown (5x)** | `93.55%` | Rata-rata penurunan terdalam portofolio |
| **Sharpe Ratio** | `4.17` | Efisiensi profit terhadap volatilitas portofolio |
| **Sortino Ratio** | `10.95` | Efisiensi profit terhadap downside deviation |
| **Calmar Ratio** | `13.18` | Rasio return tahunan terhadap drawdown |
| **Profit Factor** | `2.62` | Rasio gross profit dibagi gross loss |
| **Max Consecutive Loss** | `16` trades | Streak kekalahan beruntun terpanjang |
| **Worst Single Trade PnL** | `-24.90%` | Kerugian terdalam dalam satu trade tunggal |
| **95% Trades Loss Under** | `11.00%` | Nilai risiko (VaR P95) kerugian maksimal |

## ↕️ Analisis Arah Signal (LONG vs SHORT)

| Arah Posisi | Jumlah Trade | Distribusi | Menang | Kalah | Win Rate | PnL Bersih ($) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **LONG** | 1,776 | 41.9% | 932 | 844 | 52.48% | +308.63 |
| **SHORT** | 2,464 | 58.1% | 1,569 | 895 | 63.68% | +1,448.30 |

### Rincian Rata-rata Profitabilitas per Trade

| Tipe Trade | PnL Rata-rata ($) | PnL Rata-rata (%) |
|:---|:---:|:---:|
| **Trade Menang (Wins)** | `$+1.9549` | `+7.82%` |
| **Trade Kalah (Losses)** | `$-1.8012` | `-7.21%` |

## 📅 Scorecard Bulanan Portofolio

| Bulan | Total Trades | Wins | Losses | Win Rate | Net PnL ($) |
|:---|:---:|:---:|:---:|:---:|:---:|
| 2025-11 | 773 | 431 | 342 | 55.76% | $+241.92 |
| 2025-12 | 861 | 542 | 319 | 62.95% | $+562.77 |
| 2026-01 | 840 | 536 | 304 | 63.81% | $+564.68 |
| 2026-02 | 817 | 417 | 400 | 51.04% | $+2.32 |
| 2026-03 | 949 | 575 | 374 | 60.59% | $+385.24 |

## 🚪 Distribusi Alasan Keluar Posisi (Exit Reasons)

| Alasan Exit | Jumlah | Persentase | Wins | Losses | Win Rate | PnL Bersih ($) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| `guardian_exit` | 3,270 | 77.1% | 2,452 | 818 | 74.98% | $+3,523.83 |
| `sl_hit` | 913 | 21.5% | 2 | 911 | 0.22% | $-1,787.50 |
| `time_exit` | 57 | 1.3% | 47 | 10 | 82.46% | $+20.60 |

## 🪙 Scorecard Per Koin (Detailed Assets)

| Token | Win Rate | Trades | LONG WR | SHORT WR | Net PnL ($) | Max DD | Sharpe | Sortino | Calmar | PF |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1000PEPE** | 90.91% | 11 | 50.0% | 100.0% | `$+32.66` | 8.38% | 5.97 | 0.00 | 37.96 | 16.58 |
| **1000SHIB** | 72.73% | 11 | 66.7% | 75.0% | `$+8.64` | 4.88% | 2.47 | 6.38 | 17.24 | 4.69 |
| **ADA** | 64.42% | 163 | 57.4% | 69.5% | `$+131.49` | 51.87% | 6.51 | 20.23 | 24.69 | 2.28 |
| **ARB** | 56.61% | 242 | 51.2% | 62.0% | `$+125.50` | 85.52% | 4.64 | 11.78 | 14.29 | 1.62 |
| **AVAX** | 57.40% | 338 | 53.1% | 60.1% | `$+93.09` | 131.34% | 3.49 | 8.63 | 6.90 | 1.34 |
| **BNB** | 53.62% | 207 | 44.1% | 61.4% | `$+6.81` | 92.79% | 0.47 | 1.35 | 0.71 | 1.05 |
| **BTC** | 0.00% | 0 | 0.0% | 0.0% | `$+0.00` | 0.00% | 0.00 | 0.00 | 0.00 | 0.00 |
| **DOGE** | 83.87% | 31 | 80.0% | 87.5% | `$+57.20` | 13.85% | 6.32 | 22.95 | 40.22 | 6.60 |
| **DOT** | 56.69% | 284 | 47.0% | 63.3% | `$+59.44` | 126.89% | 2.49 | 6.23 | 4.56 | 1.27 |
| **ETH** | 67.65% | 170 | 63.5% | 70.1% | `$+125.63` | 76.29% | 6.88 | 19.14 | 16.04 | 2.28 |
| **HBAR** | 58.65% | 208 | 49.0% | 68.3% | `$+40.93` | 49.48% | 2.28 | 6.16 | 8.06 | 1.26 |
| **LINK** | 61.03% | 290 | 59.3% | 62.2% | `$+144.76` | 93.28% | 5.64 | 19.33 | 15.12 | 1.66 |
| **NEAR** | 58.50% | 294 | 43.4% | 68.0% | `$+130.05` | 84.36% | 4.70 | 10.89 | 15.01 | 1.56 |
| **ONDO** | 60.24% | 254 | 60.9% | 59.9% | `$+150.63` | 147.78% | 5.68 | 16.83 | 9.93 | 1.75 |
| **POL** | 55.23% | 239 | 49.6% | 60.3% | `$+100.96` | 179.02% | 4.30 | 11.99 | 5.49 | 1.53 |
| **SOL** | 56.16% | 333 | 51.3% | 58.8% | `$+106.88` | 152.93% | 3.87 | 12.00 | 6.81 | 1.40 |
| **SUI** | 62.22% | 270 | 52.9% | 68.1% | `$+167.18` | 190.33% | 6.21 | 15.72 | 8.56 | 1.87 |
| **TAO** | 54.62% | 249 | 49.0% | 63.0% | `$+83.84` | 261.37% | 2.89 | 6.27 | 3.12 | 1.35 |
| **TON** | 53.95% | 291 | 43.8% | 59.0% | `$+51.36` | 151.98% | 2.33 | 6.65 | 3.29 | 1.23 |
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
