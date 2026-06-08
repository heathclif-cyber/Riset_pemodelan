# 📊 Holdout Backtest Report: `cascade_v2.5_hybrid_accel_scenario_Y2`

**Tanggal Pembuatan**: 2026-06-03 21:41:20 UTC
**Model Run ID**: `cascade_v2.5_hybrid_accel_scenario_Y2`
**Periode Pengujian (Temporal OOS)**: `2025-11-01 - 2026-04-01`

> [!NOTE]
> **Ringkasan Portofolio Eksekutif**:
> *   **Total Net Profit**: **$+2,038.77 USD** (ROI Portofolio: **+97.08%**)
> *   **Rata-rata Win Rate**: **62.46%** | Total Trades: **3,322**
> *   **Rata-rata Max Drawdown (5x)**: **80.33%**
> *   **Risk-Adjusted**: Sharpe: **4.99** | Sortino: **13.92** | Calmar: **14.09** | Profit Factor: **1.91**

## 📈 Performa Scorecard Portofolio

| Metrik Utama | Nilai Portofolio | Catatan |
|:---|:---:|:---|
| **Total Net Profit ($)** | `$+2,038.77` | Akumulasi keuntungan bersih 5x leverage |
| **Portfolio ROI (%)** | `+97.08%` | ROI berdasarkan kapital portofolio $100/koin |
| **Overall Win Rate** | `62.46%` | Rasio kemenangan rata-rata seluruh aset |
| **Total Trades** | `3,322` | Jumlah total posisi yang dieksekusi |
| **Rata-rata Trade / Bulan** | `32.1` | Rata-rata frekuensi trade bulanan portofolio |
| **Rata-rata Trade / Hari** | `1.05` | Rata-rata frekuensi trade harian portofolio |
| **Max Drawdown (5x)** | `80.33%` | Rata-rata penurunan terdalam portofolio |
| **Sharpe Ratio** | `4.99` | Efisiensi profit terhadap volatilitas portofolio |
| **Sortino Ratio** | `13.92` | Efisiensi profit terhadap downside deviation |
| **Calmar Ratio** | `14.09` | Rasio return tahunan terhadap drawdown |
| **Profit Factor** | `1.91` | Rasio gross profit dibagi gross loss |
| **Max Consecutive Loss** | `10` trades | Streak kekalahan beruntun terpanjang |
| **Worst Single Trade PnL** | `-24.90%` | Kerugian terdalam dalam satu trade tunggal |
| **95% Trades Loss Under** | `11.10%` | Nilai risiko (VaR P95) kerugian maksimal |

## ↕️ Analisis Arah Signal (LONG vs SHORT)

| Arah Posisi | Jumlah Trade | Distribusi | Menang | Kalah | Win Rate | PnL Bersih ($) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **LONG** | 945 | 28.4% | 559 | 386 | 59.15% | +589.14 |
| **SHORT** | 2,377 | 71.6% | 1,513 | 864 | 63.65% | +1,449.64 |

### Rincian Rata-rata Profitabilitas per Trade

| Tipe Trade | PnL Rata-rata ($) | PnL Rata-rata (%) |
|:---|:---:|:---:|
| **Trade Menang (Wins)** | `$+2.0842` | `+8.34%` |
| **Trade Kalah (Losses)** | `$-1.8237` | `-7.30%` |

## 📅 Scorecard Bulanan Portofolio

| Bulan | Total Trades | Wins | Losses | Win Rate | Net PnL ($) |
|:---|:---:|:---:|:---:|:---:|:---:|
| 2025-11 | 657 | 402 | 255 | 61.19% | $+455.64 |
| 2025-12 | 668 | 440 | 228 | 65.87% | $+573.24 |
| 2026-01 | 600 | 402 | 198 | 67.00% | $+534.28 |
| 2026-02 | 698 | 377 | 321 | 54.01% | $+81.56 |
| 2026-03 | 699 | 451 | 248 | 64.52% | $+394.06 |

## 🚪 Distribusi Alasan Keluar Posisi (Exit Reasons)

| Alasan Exit | Jumlah | Persentase | Wins | Losses | Win Rate | PnL Bersih ($) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| `guardian_exit` | 2,638 | 79.4% | 2,026 | 612 | 76.80% | $+3,278.00 |
| `sl_hit` | 634 | 19.1% | 1 | 633 | 0.16% | $-1,258.24 |
| `time_exit` | 50 | 1.5% | 45 | 5 | 90.00% | $+19.02 |

## 🪙 Scorecard Per Koin (Detailed Assets)

| Token | Win Rate | Trades | LONG WR | SHORT WR | Net PnL ($) | Max DD | Sharpe | Sortino | Calmar | PF |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1000PEPE** | 59.85% | 132 | 62.5% | 59.5% | `$+51.53` | 87.32% | 2.47 | 5.42 | 5.75 | 1.40 |
| **1000SHIB** | 60.00% | 115 | 61.5% | 59.6% | `$+53.45` | 71.20% | 4.00 | 11.03 | 7.31 | 1.80 |
| **ADA** | 67.72% | 158 | 64.7% | 69.2% | `$+145.56` | 45.51% | 6.90 | 22.68 | 31.15 | 2.41 |
| **ARB** | 57.41% | 162 | 57.5% | 57.3% | `$+103.78` | 109.77% | 4.38 | 10.95 | 9.21 | 1.78 |
| **AVAX** | 59.75% | 241 | 55.6% | 61.5% | `$+121.07` | 149.73% | 5.26 | 14.09 | 7.88 | 1.70 |
| **BNB** | 62.79% | 129 | 63.0% | 62.7% | `$+44.61` | 34.63% | 3.85 | 11.60 | 12.55 | 1.60 |
| **BTC** | 65.38% | 52 | 0.0% | 65.4% | `$+21.31` | 37.91% | 2.75 | 13.78 | 5.48 | 1.71 |
| **DOGE** | 64.23% | 137 | 76.0% | 61.6% | `$+113.25` | 43.73% | 6.17 | 15.35 | 25.23 | 2.41 |
| **DOT** | 62.07% | 174 | 51.8% | 66.7% | `$+109.68` | 50.24% | 5.40 | 13.50 | 21.26 | 1.91 |
| **ETH** | 71.18% | 170 | 69.6% | 71.8% | `$+173.57` | 49.50% | 9.34 | 29.47 | 34.15 | 3.05 |
| **HBAR** | 60.36% | 169 | 52.8% | 63.8% | `$+43.38` | 81.53% | 2.63 | 6.86 | 5.18 | 1.35 |
| **LINK** | 66.48% | 182 | 60.0% | 69.3% | `$+149.98` | 58.57% | 7.13 | 22.02 | 24.94 | 2.26 |
| **NEAR** | 61.85% | 173 | 45.6% | 67.7% | `$+135.79` | 59.84% | 6.02 | 13.82 | 22.10 | 2.09 |
| **ONDO** | 63.46% | 156 | 59.0% | 65.0% | `$+135.30` | 102.11% | 6.28 | 14.90 | 12.91 | 2.24 |
| **POL** | 59.51% | 163 | 58.5% | 60.2% | `$+98.84` | 154.57% | 4.81 | 12.47 | 6.23 | 1.82 |
| **SOL** | 57.59% | 224 | 53.6% | 58.9% | `$+85.77` | 211.34% | 3.64 | 10.04 | 3.95 | 1.47 |
| **SUI** | 65.08% | 189 | 58.5% | 67.7% | `$+140.26` | 105.89% | 5.88 | 14.44 | 12.90 | 2.07 |
| **TAO** | 59.72% | 144 | 54.9% | 62.4% | `$+88.66` | 79.88% | 3.66 | 8.21 | 10.81 | 1.64 |
| **TON** | 62.73% | 161 | 56.4% | 64.8% | `$+123.36` | 69.46% | 6.41 | 20.65 | 17.30 | 2.23 |
| **TRX** | 61.32% | 106 | 62.7% | 60.0% | `$+14.25` | 26.57% | 2.42 | 6.07 | 5.22 | 1.42 |
| **XRP** | 63.24% | 185 | 76.6% | 58.7% | `$+85.37` | 57.65% | 5.47 | 15.00 | 14.42 | 1.84 |

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
