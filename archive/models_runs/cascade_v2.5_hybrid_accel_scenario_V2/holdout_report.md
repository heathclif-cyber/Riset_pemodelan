# 📊 Holdout Backtest Report: `cascade_v2.5_hybrid_accel_scenario_V2`

**Tanggal Pembuatan**: 2026-06-03 20:47:00 UTC
**Model Run ID**: `cascade_v2.5_hybrid_accel_scenario_V2`
**Periode Pengujian (Temporal OOS)**: `2025-11-01 - 2026-04-01`

> [!NOTE]
> **Ringkasan Portofolio Eksekutif**:
> *   **Total Net Profit**: **$+1,765.73 USD** (ROI Portofolio: **+84.08%**)
> *   **Rata-rata Win Rate**: **59.79%** | Total Trades: **4,345**
> *   **Rata-rata Max Drawdown (5x)**: **90.75%**
> *   **Risk-Adjusted**: Sharpe: **4.18** | Sortino: **11.28** | Calmar: **12.91** | Profit Factor: **2.70**

## 📈 Performa Scorecard Portofolio

| Metrik Utama | Nilai Portofolio | Catatan |
|:---|:---:|:---|
| **Total Net Profit ($)** | `$+1,765.73` | Akumulasi keuntungan bersih 5x leverage |
| **Portfolio ROI (%)** | `+84.08%` | ROI berdasarkan kapital portofolio $100/koin |
| **Overall Win Rate** | `59.79%` | Rasio kemenangan rata-rata seluruh aset |
| **Total Trades** | `4,345` | Jumlah total posisi yang dieksekusi |
| **Rata-rata Trade / Bulan** | `42.0` | Rata-rata frekuensi trade bulanan portofolio |
| **Rata-rata Trade / Hari** | `1.38` | Rata-rata frekuensi trade harian portofolio |
| **Max Drawdown (5x)** | `90.75%` | Rata-rata penurunan terdalam portofolio |
| **Sharpe Ratio** | `4.18` | Efisiensi profit terhadap volatilitas portofolio |
| **Sortino Ratio** | `11.28` | Efisiensi profit terhadap downside deviation |
| **Calmar Ratio** | `12.91` | Rasio return tahunan terhadap drawdown |
| **Profit Factor** | `2.70` | Rasio gross profit dibagi gross loss |
| **Max Consecutive Loss** | `16` trades | Streak kekalahan beruntun terpanjang |
| **Worst Single Trade PnL** | `-24.90%` | Kerugian terdalam dalam satu trade tunggal |
| **95% Trades Loss Under** | `10.98%` | Nilai risiko (VaR P95) kerugian maksimal |

## ↕️ Analisis Arah Signal (LONG vs SHORT)

| Arah Posisi | Jumlah Trade | Distribusi | Menang | Kalah | Win Rate | PnL Bersih ($) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **LONG** | 1,777 | 40.9% | 931 | 846 | 52.39% | +301.61 |
| **SHORT** | 2,568 | 59.1% | 1,626 | 942 | 63.32% | +1,464.12 |

### Rincian Rata-rata Profitabilitas per Trade

| Tipe Trade | PnL Rata-rata ($) | PnL Rata-rata (%) |
|:---|:---:|:---:|
| **Trade Menang (Wins)** | `$+1.9430` | `+7.77%` |
| **Trade Kalah (Losses)** | `$-1.7912` | `-7.17%` |

## 📅 Scorecard Bulanan Portofolio

| Bulan | Total Trades | Wins | Losses | Win Rate | Net PnL ($) |
|:---|:---:|:---:|:---:|:---:|:---:|
| 2025-11 | 790 | 444 | 346 | 56.20% | $+256.15 |
| 2025-12 | 879 | 551 | 328 | 62.68% | $+562.93 |
| 2026-01 | 871 | 549 | 322 | 63.03% | $+550.05 |
| 2026-02 | 833 | 426 | 407 | 51.14% | $+13.01 |
| 2026-03 | 972 | 587 | 385 | 60.39% | $+383.59 |

## 🚪 Distribusi Alasan Keluar Posisi (Exit Reasons)

| Alasan Exit | Jumlah | Persentase | Wins | Losses | Win Rate | PnL Bersih ($) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| `guardian_exit` | 3,346 | 77.0% | 2,507 | 839 | 74.93% | $+3,578.17 |
| `sl_hit` | 941 | 21.7% | 2 | 939 | 0.21% | $-1,833.37 |
| `time_exit` | 58 | 1.3% | 48 | 10 | 82.76% | $+20.92 |

## 🪙 Scorecard Per Koin (Detailed Assets)

| Token | Win Rate | Trades | LONG WR | SHORT WR | Net PnL ($) | Max DD | Sharpe | Sortino | Calmar | PF |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1000PEPE** | 87.50% | 8 | 50.0% | 100.0% | `$+24.82` | 8.38% | 5.13 | 0.00 | 28.84 | 12.84 |
| **1000SHIB** | 77.78% | 9 | 50.0% | 85.7% | `$+8.62` | 4.88% | 2.63 | 5.42 | 17.21 | 7.79 |
| **ADA** | 64.42% | 163 | 57.4% | 69.5% | `$+131.49` | 51.87% | 6.51 | 20.23 | 24.69 | 2.28 |
| **ARB** | 56.68% | 247 | 51.7% | 61.4% | `$+127.60` | 86.81% | 4.70 | 11.96 | 14.32 | 1.62 |
| **AVAX** | 57.51% | 353 | 53.4% | 59.9% | `$+102.04` | 123.32% | 3.76 | 9.27 | 8.06 | 1.37 |
| **BNB** | 53.77% | 212 | 44.7% | 61.0% | `$+6.06` | 92.79% | 0.42 | 1.19 | 0.64 | 1.04 |
| **BTC** | 0.00% | 0 | 0.0% | 0.0% | `$+0.00` | 0.00% | 0.00 | 0.00 | 0.00 | 0.00 |
| **DOGE** | 85.71% | 28 | 76.9% | 93.3% | `$+59.14` | 13.85% | 7.07 | 30.75 | 41.59 | 9.12 |
| **DOT** | 57.34% | 293 | 47.0% | 64.0% | `$+68.57` | 103.57% | 2.84 | 7.11 | 6.45 | 1.30 |
| **ETH** | 67.44% | 172 | 63.5% | 69.7% | `$+126.71` | 76.29% | 6.86 | 19.23 | 16.18 | 2.26 |
| **HBAR** | 58.88% | 214 | 48.5% | 68.5% | `$+43.52` | 48.91% | 2.39 | 6.52 | 8.67 | 1.28 |
| **LINK** | 60.33% | 305 | 59.3% | 60.9% | `$+137.92` | 88.96% | 5.29 | 18.41 | 15.10 | 1.59 |
| **NEAR** | 58.39% | 298 | 43.0% | 67.9% | `$+124.56` | 84.36% | 4.54 | 10.31 | 14.38 | 1.54 |
| **ONDO** | 60.15% | 261 | 61.4% | 59.5% | `$+151.19` | 137.98% | 5.65 | 16.80 | 10.67 | 1.73 |
| **POL** | 54.88% | 246 | 49.6% | 59.4% | `$+102.34` | 180.44% | 4.27 | 12.05 | 5.52 | 1.52 |
| **SOL** | 55.36% | 345 | 50.8% | 57.7% | `$+98.31` | 155.75% | 3.52 | 10.68 | 6.15 | 1.35 |
| **SUI** | 62.18% | 275 | 52.9% | 67.8% | `$+172.22` | 190.10% | 6.36 | 15.91 | 8.82 | 1.89 |
| **TAO** | 54.58% | 251 | 48.7% | 63.4% | `$+86.75` | 242.72% | 2.97 | 6.49 | 3.48 | 1.36 |
| **TON** | 53.51% | 299 | 44.2% | 57.8% | `$+50.11` | 148.28% | 2.27 | 6.29 | 3.29 | 1.22 |
| **TRX** | 65.00% | 140 | 63.1% | 67.9% | `$+24.13` | 23.29% | 3.84 | 11.06 | 10.09 | 1.61 |
| **XRP** | 64.16% | 226 | 60.9% | 66.2% | `$+119.65` | 43.19% | 6.79 | 17.23 | 26.98 | 2.02 |

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
