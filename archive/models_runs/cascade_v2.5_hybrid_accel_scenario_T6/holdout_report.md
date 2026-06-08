# 📊 Holdout Backtest Report: `cascade_v2.5_hybrid_accel_scenario_T6`

**Tanggal Pembuatan**: 2026-06-02 21:15:58 UTC
**Model Run ID**: `cascade_v2.5_hybrid_accel_scenario_T6`
**Periode Pengujian (Temporal OOS)**: `2025-11-01 - 2026-04-01`

> [!NOTE]
> **Ringkasan Portofolio Eksekutif**:
> *   **Total Net Profit**: **$+1,647.41 USD** (ROI Portofolio: **+78.45%**)
> *   **Rata-rata Win Rate**: **62.76%** | Total Trades: **3,236**
> *   **Rata-rata Max Drawdown (5x)**: **84.66%**
> *   **Risk-Adjusted**: Sharpe: **4.36** | Sortino: **11.36** | Calmar: **14.51** | Profit Factor: **3.04**

## 📈 Performa Scorecard Portofolio

| Metrik Utama | Nilai Portofolio | Catatan |
|:---|:---:|:---|
| **Total Net Profit ($)** | `$+1,647.41` | Akumulasi keuntungan bersih 5x leverage |
| **Portfolio ROI (%)** | `+78.45%` | ROI berdasarkan kapital portofolio $100/koin |
| **Overall Win Rate** | `62.76%` | Rasio kemenangan rata-rata seluruh aset |
| **Total Trades** | `3,236` | Jumlah total posisi yang dieksekusi |
| **Rata-rata Trade / Bulan** | `31.3` | Rata-rata frekuensi trade bulanan portofolio |
| **Rata-rata Trade / Hari** | `1.03` | Rata-rata frekuensi trade harian portofolio |
| **Max Drawdown (5x)** | `84.66%` | Rata-rata penurunan terdalam portofolio |
| **Sharpe Ratio** | `4.36` | Efisiensi profit terhadap volatilitas portofolio |
| **Sortino Ratio** | `11.36` | Efisiensi profit terhadap downside deviation |
| **Calmar Ratio** | `14.51` | Rasio return tahunan terhadap drawdown |
| **Profit Factor** | `3.04` | Rasio gross profit dibagi gross loss |
| **Max Consecutive Loss** | `11` trades | Streak kekalahan beruntun terpanjang |
| **Worst Single Trade PnL** | `-24.90%` | Kerugian terdalam dalam satu trade tunggal |
| **95% Trades Loss Under** | `11.20%` | Nilai risiko (VaR P95) kerugian maksimal |

## ↕️ Analisis Arah Signal (LONG vs SHORT)

| Arah Posisi | Jumlah Trade | Distribusi | Menang | Kalah | Win Rate | PnL Bersih ($) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **LONG** | 1,554 | 48.0% | 835 | 719 | 53.73% | +385.27 |
| **SHORT** | 1,682 | 52.0% | 1,110 | 572 | 65.99% | +1,262.13 |

### Rincian Rata-rata Profitabilitas per Trade

| Tipe Trade | PnL Rata-rata ($) | PnL Rata-rata (%) |
|:---|:---:|:---:|
| **Trade Menang (Wins)** | `$+2.0438` | `+8.18%` |
| **Trade Kalah (Losses)** | `$-1.8031` | `-7.21%` |

## 📅 Scorecard Bulanan Portofolio

| Bulan | Total Trades | Wins | Losses | Win Rate | Net PnL ($) |
|:---|:---:|:---:|:---:|:---:|:---:|
| 2025-11 | 639 | 361 | 278 | 56.49% | $+246.78 |
| 2025-12 | 635 | 420 | 215 | 66.14% | $+536.21 |
| 2026-01 | 622 | 399 | 223 | 64.15% | $+474.33 |
| 2026-02 | 628 | 323 | 305 | 51.43% | $-1.45 |
| 2026-03 | 712 | 442 | 270 | 62.08% | $+391.55 |

## 🚪 Distribusi Alasan Keluar Posisi (Exit Reasons)

| Alasan Exit | Jumlah | Persentase | Wins | Losses | Win Rate | PnL Bersih ($) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| `guardian_exit` | 2,513 | 77.7% | 1,902 | 611 | 75.69% | $+2,949.94 |
| `sl_hit` | 673 | 20.8% | 1 | 672 | 0.15% | $-1,321.03 |
| `time_exit` | 50 | 1.5% | 42 | 8 | 84.00% | $+18.50 |

## 🪙 Scorecard Per Koin (Detailed Assets)

| Token | Win Rate | Trades | LONG WR | SHORT WR | Net PnL ($) | Max DD | Sharpe | Sortino | Calmar | PF |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1000PEPE** | 93.75% | 16 | 75.0% | 100.0% | `$+45.86` | 8.38% | 7.23 | 0.00 | 53.30 | 22.88 |
| **1000SHIB** | 69.23% | 13 | 60.0% | 75.0% | `$+8.13` | 6.32% | 2.23 | 6.65 | 12.52 | 3.45 |
| **ADA** | 64.89% | 131 | 60.9% | 68.7% | `$+119.67` | 51.87% | 6.31 | 19.12 | 22.47 | 2.44 |
| **ARB** | 53.44% | 189 | 49.5% | 59.0% | `$+76.25` | 110.04% | 3.06 | 7.79 | 6.75 | 1.44 |
| **AVAX** | 58.65% | 266 | 52.6% | 63.2% | `$+96.44` | 109.46% | 3.94 | 9.61 | 8.58 | 1.46 |
| **BNB** | 56.25% | 144 | 47.8% | 64.0% | `$+19.58` | 48.46% | 1.60 | 4.97 | 3.94 | 1.21 |
| **BTC** | 50.00% | 2 | 0.0% | 50.0% | `$+0.84` | 6.05% | 0.34 | 0.00 | 1.36 | 1.56 |
| **DOGE** | 83.87% | 31 | 80.0% | 87.5% | `$+57.20` | 13.85% | 6.32 | 22.95 | 40.22 | 6.60 |
| **DOT** | 57.62% | 210 | 49.0% | 65.5% | `$+67.14` | 117.81% | 3.11 | 7.40 | 5.55 | 1.41 |
| **ETH** | 69.18% | 146 | 64.1% | 73.2% | `$+120.45` | 54.25% | 7.23 | 23.21 | 21.63 | 2.55 |
| **HBAR** | 58.86% | 175 | 50.5% | 68.8% | `$+31.73` | 59.27% | 1.90 | 5.31 | 5.21 | 1.24 |
| **LINK** | 68.06% | 216 | 60.8% | 74.6% | `$+169.58` | 60.59% | 7.85 | 24.18 | 27.26 | 2.26 |
| **NEAR** | 56.80% | 206 | 40.9% | 69.9% | `$+113.69` | 62.81% | 4.70 | 11.58 | 17.63 | 1.70 |
| **ONDO** | 61.41% | 184 | 59.0% | 63.2% | `$+125.30` | 157.30% | 5.55 | 15.10 | 7.76 | 1.92 |
| **POL** | 54.05% | 185 | 49.5% | 59.3% | `$+73.77` | 190.01% | 3.46 | 9.48 | 3.78 | 1.49 |
| **SOL** | 54.69% | 245 | 50.5% | 57.5% | `$+88.40` | 155.25% | 3.51 | 11.61 | 5.55 | 1.43 |
| **SUI** | 63.21% | 212 | 55.1% | 69.1% | `$+146.53` | 190.29% | 5.93 | 15.34 | 7.50 | 1.98 |
| **TAO** | 56.32% | 190 | 51.6% | 65.1% | `$+91.89` | 203.82% | 3.52 | 7.95 | 4.39 | 1.52 |
| **TON** | 56.06% | 198 | 45.7% | 63.2% | `$+69.64` | 115.19% | 3.61 | 10.01 | 5.89 | 1.48 |
| **TRX** | 66.04% | 106 | 63.0% | 72.7% | `$+18.42` | 28.81% | 3.44 | 9.94 | 6.23 | 1.64 |
| **XRP** | 65.50% | 171 | 68.0% | 63.5% | `$+106.91` | 28.03% | 6.79 | 16.46 | 37.15 | 2.25 |

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
