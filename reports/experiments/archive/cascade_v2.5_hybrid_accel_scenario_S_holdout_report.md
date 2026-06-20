# 📊 Holdout Backtest Report: `cascade_v2.5_hybrid_accel_scenario_S`

**Tanggal Pembuatan**: 2026-06-02 21:08:53 UTC
**Model Run ID**: `cascade_v2.5_hybrid_accel_scenario_S`
**Periode Pengujian (Temporal OOS)**: `2025-11-01 - 2026-04-01`

> [!NOTE]
> **Ringkasan Portofolio Eksekutif**:
> *   **Total Net Profit**: **$+1,645.55 USD** (ROI Portofolio: **+78.36%**)
> *   **Rata-rata Win Rate**: **59.69%** | Total Trades: **3,654**
> *   **Rata-rata Max Drawdown (5x)**: **86.52%**
> *   **Risk-Adjusted**: Sharpe: **4.13** | Sortino: **11.33** | Calmar: **12.65** | Profit Factor: **2.56**

## 📈 Performa Scorecard Portofolio

| Metrik Utama | Nilai Portofolio | Catatan |
|:---|:---:|:---|
| **Total Net Profit ($)** | `$+1,645.55` | Akumulasi keuntungan bersih 5x leverage |
| **Portfolio ROI (%)** | `+78.36%` | ROI berdasarkan kapital portofolio $100/koin |
| **Overall Win Rate** | `59.69%` | Rasio kemenangan rata-rata seluruh aset |
| **Total Trades** | `3,654` | Jumlah total posisi yang dieksekusi |
| **Rata-rata Trade / Bulan** | `35.3` | Rata-rata frekuensi trade bulanan portofolio |
| **Rata-rata Trade / Hari** | `1.16` | Rata-rata frekuensi trade harian portofolio |
| **Max Drawdown (5x)** | `86.52%` | Rata-rata penurunan terdalam portofolio |
| **Sharpe Ratio** | `4.13` | Efisiensi profit terhadap volatilitas portofolio |
| **Sortino Ratio** | `11.33` | Efisiensi profit terhadap downside deviation |
| **Calmar Ratio** | `12.65` | Rasio return tahunan terhadap drawdown |
| **Profit Factor** | `2.56` | Rasio gross profit dibagi gross loss |
| **Max Consecutive Loss** | `13` trades | Streak kekalahan beruntun terpanjang |
| **Worst Single Trade PnL** | `-24.90%` | Kerugian terdalam dalam satu trade tunggal |
| **95% Trades Loss Under** | `11.10%` | Nilai risiko (VaR P95) kerugian maksimal |

## ↕️ Analisis Arah Signal (LONG vs SHORT)

| Arah Posisi | Jumlah Trade | Distribusi | Menang | Kalah | Win Rate | PnL Bersih ($) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **LONG** | 1,653 | 45.2% | 879 | 774 | 53.18% | +338.06 |
| **SHORT** | 2,001 | 54.8% | 1,285 | 716 | 64.22% | +1,307.50 |

### Rincian Rata-rata Profitabilitas per Trade

| Tipe Trade | PnL Rata-rata ($) | PnL Rata-rata (%) |
|:---|:---:|:---:|
| **Trade Menang (Wins)** | `$+2.0069` | `+8.03%` |
| **Trade Kalah (Losses)** | `$-1.8103` | `-7.24%` |

## 📅 Scorecard Bulanan Portofolio

| Bulan | Total Trades | Wins | Losses | Win Rate | Net PnL ($) |
|:---|:---:|:---:|:---:|:---:|:---:|
| 2025-11 | 682 | 384 | 298 | 56.30% | $+239.99 |
| 2025-12 | 736 | 476 | 260 | 64.67% | $+545.74 |
| 2026-01 | 711 | 453 | 258 | 63.71% | $+505.64 |
| 2026-02 | 712 | 357 | 355 | 50.14% | $-29.42 |
| 2026-03 | 813 | 494 | 319 | 60.76% | $+383.60 |

## 🚪 Distribusi Alasan Keluar Posisi (Exit Reasons)

| Alasan Exit | Jumlah | Persentase | Wins | Losses | Win Rate | PnL Bersih ($) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| `guardian_exit` | 2,829 | 77.4% | 2,120 | 709 | 74.94% | $+3,144.74 |
| `sl_hit` | 775 | 21.2% | 2 | 773 | 0.26% | $-1,517.67 |
| `time_exit` | 50 | 1.4% | 42 | 8 | 84.00% | $+18.48 |

## 🪙 Scorecard Per Koin (Detailed Assets)

| Token | Win Rate | Trades | LONG WR | SHORT WR | Net PnL ($) | Max DD | Sharpe | Sortino | Calmar | PF |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1000PEPE** | 85.71% | 7 | 50.0% | 100.0% | `$+21.94` | 8.38% | 4.50 | 0.00 | 25.51 | 11.47 |
| **1000SHIB** | 71.43% | 7 | 0.0% | 83.3% | `$+7.59` | 4.88% | 2.30 | 5.41 | 15.16 | 6.98 |
| **ADA** | 64.38% | 146 | 59.4% | 68.3% | `$+125.24` | 51.87% | 6.33 | 19.48 | 23.52 | 2.33 |
| **ARB** | 55.09% | 216 | 50.9% | 59.8% | `$+102.52` | 88.85% | 3.94 | 9.93 | 11.24 | 1.55 |
| **AVAX** | 56.80% | 294 | 52.5% | 59.9% | `$+84.93` | 139.56% | 3.31 | 8.36 | 5.93 | 1.35 |
| **BNB** | 55.68% | 176 | 46.8% | 62.9% | `$+18.37` | 69.95% | 1.36 | 4.32 | 2.56 | 1.15 |
| **BTC** | 0.00% | 0 | 0.0% | 0.0% | `$+0.00` | 0.00% | 0.00 | 0.00 | 0.00 | 0.00 |
| **DOGE** | 84.21% | 19 | 66.7% | 92.3% | `$+36.19` | 13.85% | 5.42 | 25.31 | 25.45 | 6.99 |
| **DOT** | 56.96% | 237 | 48.1% | 64.1% | `$+64.39` | 117.81% | 2.86 | 7.05 | 5.32 | 1.35 |
| **ETH** | 69.87% | 156 | 66.7% | 71.9% | `$+135.25` | 45.68% | 8.02 | 25.14 | 28.84 | 2.73 |
| **HBAR** | 57.95% | 195 | 49.0% | 67.7% | `$+35.91` | 44.64% | 2.03 | 5.69 | 7.84 | 1.24 |
| **LINK** | 63.01% | 246 | 59.6% | 65.7% | `$+142.92` | 80.11% | 6.11 | 20.46 | 17.38 | 1.81 |
| **NEAR** | 57.38% | 237 | 43.1% | 68.2% | `$+112.44` | 70.83% | 4.40 | 10.06 | 15.46 | 1.59 |
| **ONDO** | 61.90% | 210 | 61.0% | 62.5% | `$+147.52` | 147.42% | 6.15 | 17.27 | 9.75 | 1.95 |
| **POL** | 56.04% | 207 | 52.4% | 59.8% | `$+95.37` | 194.17% | 4.28 | 11.82 | 4.78 | 1.59 |
| **SOL** | 54.30% | 291 | 49.5% | 57.2% | `$+86.04` | 136.43% | 3.22 | 10.42 | 6.14 | 1.35 |
| **SUI** | 62.08% | 240 | 53.6% | 67.8% | `$+151.14` | 188.00% | 5.86 | 14.98 | 7.83 | 1.87 |
| **TAO** | 55.71% | 219 | 51.5% | 62.6% | `$+85.67` | 216.10% | 3.10 | 6.97 | 3.86 | 1.41 |
| **TON** | 54.43% | 237 | 44.6% | 60.7% | `$+58.12` | 144.69% | 2.85 | 7.81 | 3.91 | 1.32 |
| **TRX** | 65.29% | 121 | 63.0% | 70.0% | `$+22.80` | 23.29% | 3.89 | 10.79 | 9.54 | 1.69 |
| **XRP** | 65.28% | 193 | 64.6% | 65.8% | `$+111.20` | 30.41% | 6.80 | 16.68 | 35.62 | 2.14 |

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
