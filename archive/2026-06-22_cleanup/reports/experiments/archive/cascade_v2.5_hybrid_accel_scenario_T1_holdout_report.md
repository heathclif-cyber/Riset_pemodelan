# 📊 Holdout Backtest Report: `cascade_v2.5_hybrid_accel_scenario_T1`

**Tanggal Pembuatan**: 2026-06-02 21:11:49 UTC
**Model Run ID**: `cascade_v2.5_hybrid_accel_scenario_T1`
**Periode Pengujian (Temporal OOS)**: `2025-11-01 - 2026-04-01`

> [!NOTE]
> **Ringkasan Portofolio Eksekutif**:
> *   **Total Net Profit**: **$+1,539.11 USD** (ROI Portofolio: **+73.29%**)
> *   **Rata-rata Win Rate**: **61.33%** | Total Trades: **2,822**
> *   **Rata-rata Max Drawdown (5x)**: **78.49%**
> *   **Risk-Adjusted**: Sharpe: **4.37** | Sortino: **12.65** | Calmar: **12.03** | Profit Factor: **2.18**

## 📈 Performa Scorecard Portofolio

| Metrik Utama | Nilai Portofolio | Catatan |
|:---|:---:|:---|
| **Total Net Profit ($)** | `$+1,539.11` | Akumulasi keuntungan bersih 5x leverage |
| **Portfolio ROI (%)** | `+73.29%` | ROI berdasarkan kapital portofolio $100/koin |
| **Overall Win Rate** | `61.33%` | Rasio kemenangan rata-rata seluruh aset |
| **Total Trades** | `2,822` | Jumlah total posisi yang dieksekusi |
| **Rata-rata Trade / Bulan** | `27.3` | Rata-rata frekuensi trade bulanan portofolio |
| **Rata-rata Trade / Hari** | `0.90` | Rata-rata frekuensi trade harian portofolio |
| **Max Drawdown (5x)** | `78.49%` | Rata-rata penurunan terdalam portofolio |
| **Sharpe Ratio** | `4.37` | Efisiensi profit terhadap volatilitas portofolio |
| **Sortino Ratio** | `12.65` | Efisiensi profit terhadap downside deviation |
| **Calmar Ratio** | `12.03` | Rasio return tahunan terhadap drawdown |
| **Profit Factor** | `2.18` | Rasio gross profit dibagi gross loss |
| **Max Consecutive Loss** | `10` trades | Streak kekalahan beruntun terpanjang |
| **Worst Single Trade PnL** | `-24.90%` | Kerugian terdalam dalam satu trade tunggal |
| **95% Trades Loss Under** | `11.20%` | Nilai risiko (VaR P95) kerugian maksimal |

## ↕️ Analisis Arah Signal (LONG vs SHORT)

| Arah Posisi | Jumlah Trade | Distribusi | Menang | Kalah | Win Rate | PnL Bersih ($) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **LONG** | 1,393 | 49.4% | 753 | 640 | 54.06% | +386.36 |
| **SHORT** | 1,429 | 50.6% | 954 | 475 | 66.76% | +1,152.75 |

### Rincian Rata-rata Profitabilitas per Trade

| Tipe Trade | PnL Rata-rata ($) | PnL Rata-rata (%) |
|:---|:---:|:---:|
| **Trade Menang (Wins)** | `$+2.0884` | `+8.35%` |
| **Trade Kalah (Losses)** | `$-1.8169` | `-7.27%` |

## 📅 Scorecard Bulanan Portofolio

| Bulan | Total Trades | Wins | Losses | Win Rate | Net PnL ($) |
|:---|:---:|:---:|:---:|:---:|:---:|
| 2025-11 | 565 | 327 | 238 | 57.88% | $+267.15 |
| 2025-12 | 562 | 372 | 190 | 66.19% | $+503.91 |
| 2026-01 | 529 | 341 | 188 | 64.46% | $+433.16 |
| 2026-02 | 552 | 280 | 272 | 50.72% | $-31.40 |
| 2026-03 | 614 | 387 | 227 | 63.03% | $+366.30 |

## 🚪 Distribusi Alasan Keluar Posisi (Exit Reasons)

| Alasan Exit | Jumlah | Persentase | Wins | Losses | Win Rate | PnL Bersih ($) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| `guardian_exit` | 2,196 | 77.8% | 1,673 | 523 | 76.18% | $+2,686.65 |
| `sl_hit` | 587 | 20.8% | 0 | 587 | 0.00% | $-1,163.03 |
| `time_exit` | 39 | 1.4% | 34 | 5 | 87.18% | $+15.49 |

## 🪙 Scorecard Per Koin (Detailed Assets)

| Token | Win Rate | Trades | LONG WR | SHORT WR | Net PnL ($) | Max DD | Sharpe | Sortino | Calmar | PF |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1000PEPE** | 100.00% | 6 | 100.0% | 100.0% | `$+24.04` | 0.00% | 8.12 | 0.00 | 0.00 | 0.00 |
| **1000SHIB** | 71.43% | 7 | 0.0% | 83.3% | `$+7.59` | 4.88% | 2.30 | 5.41 | 15.16 | 6.98 |
| **ADA** | 65.25% | 118 | 61.4% | 68.8% | `$+112.76` | 32.55% | 6.27 | 18.49 | 33.74 | 2.55 |
| **ARB** | 53.85% | 169 | 52.5% | 55.9% | `$+74.25` | 126.49% | 3.07 | 7.93 | 5.72 | 1.47 |
| **AVAX** | 59.00% | 239 | 52.7% | 64.3% | `$+103.83` | 129.08% | 4.35 | 10.98 | 7.83 | 1.55 |
| **BNB** | 58.40% | 125 | 47.5% | 68.2% | `$+23.69` | 44.24% | 2.03 | 6.65 | 5.22 | 1.29 |
| **BTC** | 0.00% | 0 | 0.0% | 0.0% | `$+0.00` | 0.00% | 0.00 | 0.00 | 0.00 | 0.00 |
| **DOGE** | 85.71% | 14 | 60.0% | 100.0% | `$+24.53` | 13.85% | 4.70 | 48.27 | 17.25 | 8.09 |
| **DOT** | 60.56% | 180 | 48.9% | 72.2% | `$+91.55` | 86.47% | 4.50 | 10.82 | 10.31 | 1.71 |
| **ETH** | 70.23% | 131 | 66.7% | 73.0% | `$+120.70` | 30.99% | 7.69 | 24.26 | 37.94 | 2.87 |
| **HBAR** | 58.06% | 155 | 50.0% | 68.7% | `$+30.39` | 46.61% | 1.90 | 5.36 | 6.35 | 1.25 |
| **LINK** | 66.67% | 189 | 60.4% | 72.5% | `$+145.70` | 68.79% | 7.00 | 21.59 | 20.63 | 2.20 |
| **NEAR** | 57.06% | 177 | 40.5% | 70.4% | `$+104.39` | 62.81% | 4.58 | 11.26 | 16.19 | 1.74 |
| **ONDO** | 60.90% | 156 | 56.9% | 63.7% | `$+104.59` | 159.24% | 4.96 | 12.98 | 6.40 | 1.89 |
| **POL** | 54.66% | 161 | 50.5% | 60.0% | `$+72.50` | 187.66% | 3.51 | 9.90 | 3.76 | 1.54 |
| **SOL** | 54.88% | 215 | 52.2% | 56.9% | `$+81.01` | 134.40% | 3.45 | 11.09 | 5.87 | 1.46 |
| **SUI** | 64.06% | 192 | 55.6% | 70.3% | `$+142.42` | 181.34% | 5.90 | 15.38 | 7.65 | 2.05 |
| **TAO** | 56.98% | 172 | 52.2% | 66.1% | `$+88.85` | 182.62% | 3.55 | 7.89 | 4.74 | 1.56 |
| **TON** | 58.28% | 163 | 47.2% | 67.0% | `$+67.42` | 103.25% | 3.81 | 10.97 | 6.36 | 1.57 |
| **TRX** | 66.00% | 100 | 62.9% | 73.3% | `$+17.71` | 24.38% | 3.41 | 9.95 | 7.08 | 1.66 |
| **XRP** | 66.01% | 153 | 70.0% | 62.6% | `$+101.18` | 28.63% | 6.69 | 16.51 | 34.42 | 2.33 |

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
