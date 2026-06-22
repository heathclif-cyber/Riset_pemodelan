# 📊 Holdout Backtest Report: `cascade_v2.5_hybrid_accel_scenario_T`

**Tanggal Pembuatan**: 2026-06-02 21:09:14 UTC
**Model Run ID**: `cascade_v2.5_hybrid_accel_scenario_T`
**Periode Pengujian (Temporal OOS)**: `2025-11-01 - 2026-04-01`

> [!NOTE]
> **Ringkasan Portofolio Eksekutif**:
> *   **Total Net Profit**: **$+1,521.87 USD** (ROI Portofolio: **+72.47%**)
> *   **Rata-rata Win Rate**: **61.10%** | Total Trades: **2,800**
> *   **Rata-rata Max Drawdown (5x)**: **78.08%**
> *   **Risk-Adjusted**: Sharpe: **4.12** | Sortino: **12.06** | Calmar: **25.34** | Profit Factor: **7.59**

## 📈 Performa Scorecard Portofolio

| Metrik Utama | Nilai Portofolio | Catatan |
|:---|:---:|:---|
| **Total Net Profit ($)** | `$+1,521.87` | Akumulasi keuntungan bersih 5x leverage |
| **Portfolio ROI (%)** | `+72.47%` | ROI berdasarkan kapital portofolio $100/koin |
| **Overall Win Rate** | `61.10%` | Rasio kemenangan rata-rata seluruh aset |
| **Total Trades** | `2,800` | Jumlah total posisi yang dieksekusi |
| **Rata-rata Trade / Bulan** | `27.1` | Rata-rata frekuensi trade bulanan portofolio |
| **Rata-rata Trade / Hari** | `0.89` | Rata-rata frekuensi trade harian portofolio |
| **Max Drawdown (5x)** | `78.08%` | Rata-rata penurunan terdalam portofolio |
| **Sharpe Ratio** | `4.12` | Efisiensi profit terhadap volatilitas portofolio |
| **Sortino Ratio** | `12.06` | Efisiensi profit terhadap downside deviation |
| **Calmar Ratio** | `25.34` | Rasio return tahunan terhadap drawdown |
| **Profit Factor** | `7.59` | Rasio gross profit dibagi gross loss |
| **Max Consecutive Loss** | `10` trades | Streak kekalahan beruntun terpanjang |
| **Worst Single Trade PnL** | `-24.90%` | Kerugian terdalam dalam satu trade tunggal |
| **95% Trades Loss Under** | `11.20%` | Nilai risiko (VaR P95) kerugian maksimal |

## ↕️ Analisis Arah Signal (LONG vs SHORT)

| Arah Posisi | Jumlah Trade | Distribusi | Menang | Kalah | Win Rate | PnL Bersih ($) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **LONG** | 1,379 | 49.2% | 747 | 632 | 54.17% | +386.64 |
| **SHORT** | 1,421 | 50.7% | 947 | 474 | 66.64% | +1,135.23 |

### Rincian Rata-rata Profitabilitas per Trade

| Tipe Trade | PnL Rata-rata ($) | PnL Rata-rata (%) |
|:---|:---:|:---:|
| **Trade Menang (Wins)** | `$+2.0864` | `+8.35%` |
| **Trade Kalah (Losses)** | `$-1.8196` | `-7.28%` |

## 📅 Scorecard Bulanan Portofolio

| Bulan | Total Trades | Wins | Losses | Win Rate | Net PnL ($) |
|:---|:---:|:---:|:---:|:---:|:---:|
| 2025-11 | 557 | 322 | 235 | 57.81% | $+263.43 |
| 2025-12 | 560 | 370 | 190 | 66.07% | $+498.48 |
| 2026-01 | 527 | 340 | 187 | 64.52% | $+428.52 |
| 2026-02 | 546 | 277 | 269 | 50.73% | $-27.43 |
| 2026-03 | 610 | 385 | 225 | 63.11% | $+358.86 |

## 🚪 Distribusi Alasan Keluar Posisi (Exit Reasons)

| Alasan Exit | Jumlah | Persentase | Wins | Losses | Win Rate | PnL Bersih ($) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| `guardian_exit` | 2,181 | 77.9% | 1,661 | 520 | 76.16% | $+2,661.25 |
| `sl_hit` | 582 | 20.8% | 0 | 582 | 0.00% | $-1,154.25 |
| `time_exit` | 37 | 1.3% | 33 | 4 | 89.19% | $+14.87 |

## 🪙 Scorecard Per Koin (Detailed Assets)

| Token | Win Rate | Trades | LONG WR | SHORT WR | Net PnL ($) | Max DD | Sharpe | Sortino | Calmar | PF |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1000PEPE** | 100.00% | 3 | 100.0% | 100.0% | `$+9.03` | 0.00% | 3.51 | 0.00 | 0.00 | 0.00 |
| **1000SHIB** | 66.67% | 3 | 0.0% | 66.7% | `$+6.02` | 0.20% | 2.15 | 0.00 | 293.38 | 121.97 |
| **ADA** | 66.38% | 116 | 62.5% | 70.0% | `$+116.61` | 29.97% | 6.55 | 18.86 | 37.90 | 2.69 |
| **ARB** | 53.89% | 167 | 52.5% | 55.9% | `$+73.64` | 126.49% | 3.05 | 8.01 | 5.67 | 1.47 |
| **AVAX** | 59.00% | 239 | 52.7% | 64.3% | `$+103.83` | 129.08% | 4.35 | 10.98 | 7.83 | 1.55 |
| **BNB** | 58.40% | 125 | 47.5% | 68.2% | `$+23.69` | 44.24% | 2.03 | 6.65 | 5.22 | 1.29 |
| **BTC** | 0.00% | 0 | 0.0% | 0.0% | `$+0.00` | 0.00% | 0.00 | 0.00 | 0.00 | 0.00 |
| **DOGE** | 83.33% | 12 | 33.3% | 100.0% | `$+18.86` | 13.85% | 3.76 | 40.10 | 13.27 | 6.45 |
| **DOT** | 60.56% | 180 | 48.9% | 72.2% | `$+91.55` | 86.47% | 4.50 | 10.82 | 10.31 | 1.71 |
| **ETH** | 70.00% | 130 | 66.7% | 72.6% | `$+118.79` | 30.99% | 7.58 | 23.97 | 37.34 | 2.84 |
| **HBAR** | 58.17% | 153 | 50.0% | 68.7% | `$+30.50` | 48.31% | 1.91 | 5.52 | 6.15 | 1.26 |
| **LINK** | 66.67% | 189 | 60.4% | 72.5% | `$+145.70` | 68.79% | 7.00 | 21.59 | 20.63 | 2.20 |
| **NEAR** | 57.71% | 175 | 41.6% | 70.4% | `$+109.75` | 62.81% | 4.86 | 12.19 | 17.02 | 1.81 |
| **ONDO** | 61.29% | 155 | 57.8% | 63.7% | `$+106.07` | 159.24% | 5.04 | 13.12 | 6.49 | 1.92 |
| **POL** | 54.37% | 160 | 50.0% | 60.0% | `$+71.77` | 187.66% | 3.47 | 9.83 | 3.73 | 1.53 |
| **SOL** | 54.67% | 214 | 51.6% | 56.9% | `$+77.70` | 134.40% | 3.32 | 10.67 | 5.63 | 1.44 |
| **SUI** | 64.06% | 192 | 55.6% | 70.3% | `$+142.42` | 181.34% | 5.90 | 15.38 | 7.65 | 2.05 |
| **TAO** | 56.98% | 172 | 52.2% | 66.1% | `$+88.85` | 182.62% | 3.55 | 7.89 | 4.74 | 1.56 |
| **TON** | 58.28% | 163 | 47.2% | 67.0% | `$+67.42` | 103.25% | 3.81 | 10.97 | 6.36 | 1.57 |
| **TRX** | 66.67% | 99 | 63.8% | 73.3% | `$+18.48` | 21.32% | 3.58 | 10.28 | 8.44 | 1.71 |
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
