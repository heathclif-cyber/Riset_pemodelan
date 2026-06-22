# 📊 Holdout Backtest Report: `cascade_v2.5_hybrid_accel_scenario_X3`

**Tanggal Pembuatan**: 2026-06-03 21:25:33 UTC
**Model Run ID**: `cascade_v2.5_hybrid_accel_scenario_X3`
**Periode Pengujian (Temporal OOS)**: `2025-11-01 - 2026-04-01`

> [!NOTE]
> **Ringkasan Portofolio Eksekutif**:
> *   **Total Net Profit**: **$+1,165.37 USD** (ROI Portofolio: **+55.49%**)
> *   **Rata-rata Win Rate**: **60.99%** | Total Trades: **1,645**
> *   **Rata-rata Max Drawdown (5x)**: **45.51%**
> *   **Risk-Adjusted**: Sharpe: **4.00** | Sortino: **11.28** | Calmar: **11.90** | Profit Factor: **2.26**

## 📈 Performa Scorecard Portofolio

| Metrik Utama | Nilai Portofolio | Catatan |
|:---|:---:|:---|
| **Total Net Profit ($)** | `$+1,165.37` | Akumulasi keuntungan bersih 5x leverage |
| **Portfolio ROI (%)** | `+55.49%` | ROI berdasarkan kapital portofolio $100/koin |
| **Overall Win Rate** | `60.99%` | Rasio kemenangan rata-rata seluruh aset |
| **Total Trades** | `1,645` | Jumlah total posisi yang dieksekusi |
| **Rata-rata Trade / Bulan** | `15.9` | Rata-rata frekuensi trade bulanan portofolio |
| **Rata-rata Trade / Hari** | `0.52` | Rata-rata frekuensi trade harian portofolio |
| **Max Drawdown (5x)** | `45.51%` | Rata-rata penurunan terdalam portofolio |
| **Sharpe Ratio** | `4.00` | Efisiensi profit terhadap volatilitas portofolio |
| **Sortino Ratio** | `11.28` | Efisiensi profit terhadap downside deviation |
| **Calmar Ratio** | `11.90` | Rasio return tahunan terhadap drawdown |
| **Profit Factor** | `2.26` | Rasio gross profit dibagi gross loss |
| **Max Consecutive Loss** | `9` trades | Streak kekalahan beruntun terpanjang |
| **Worst Single Trade PnL** | `-23.60%` | Kerugian terdalam dalam satu trade tunggal |
| **95% Trades Loss Under** | `10.20%` | Nilai risiko (VaR P95) kerugian maksimal |

## ↕️ Analisis Arah Signal (LONG vs SHORT)

| Arah Posisi | Jumlah Trade | Distribusi | Menang | Kalah | Win Rate | PnL Bersih ($) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **LONG** | 387 | 23.5% | 228 | 159 | 58.91% | +223.06 |
| **SHORT** | 1,258 | 76.5% | 823 | 435 | 65.42% | +942.32 |

### Rincian Rata-rata Profitabilitas per Trade

| Tipe Trade | PnL Rata-rata ($) | PnL Rata-rata (%) |
|:---|:---:|:---:|
| **Trade Menang (Wins)** | `$+2.0903` | `+8.36%` |
| **Trade Kalah (Losses)** | `$-1.7366` | `-6.95%` |

## 📅 Scorecard Bulanan Portofolio

| Bulan | Total Trades | Wins | Losses | Win Rate | Net PnL ($) |
|:---|:---:|:---:|:---:|:---:|:---:|
| 2025-11 | 297 | 191 | 106 | 64.31% | $+231.19 |
| 2025-12 | 317 | 215 | 102 | 67.82% | $+344.86 |
| 2026-01 | 275 | 186 | 89 | 67.64% | $+254.64 |
| 2026-02 | 367 | 200 | 167 | 54.50% | $+70.47 |
| 2026-03 | 389 | 259 | 130 | 66.58% | $+264.21 |

## 🚪 Distribusi Alasan Keluar Posisi (Exit Reasons)

| Alasan Exit | Jumlah | Persentase | Wins | Losses | Win Rate | PnL Bersih ($) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| `guardian_exit` | 1,285 | 78.1% | 1,036 | 249 | 80.62% | $+1,791.80 |
| `sl_hit` | 344 | 20.9% | 1 | 343 | 0.29% | $-633.03 |
| `time_exit` | 16 | 1.0% | 14 | 2 | 87.50% | $+6.60 |

## 🪙 Scorecard Per Koin (Detailed Assets)

| Token | Win Rate | Trades | LONG WR | SHORT WR | Net PnL ($) | Max DD | Sharpe | Sortino | Calmar | PF |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1000PEPE** | 75.00% | 4 | 0.0% | 75.0% | `$+7.64` | 6.26% | 1.87 | 0.00 | 11.89 | 5.88 |
| **1000SHIB** | 55.56% | 9 | 33.3% | 66.7% | `$+2.27` | 14.17% | 0.67 | 1.80 | 1.56 | 1.44 |
| **ADA** | 65.67% | 67 | 64.3% | 66.0% | `$+65.80` | 58.62% | 4.62 | 16.38 | 10.93 | 2.56 |
| **ARB** | 58.82% | 85 | 60.7% | 57.9% | `$+64.53` | 63.59% | 3.94 | 12.26 | 9.88 | 1.95 |
| **AVAX** | 62.32% | 138 | 53.6% | 64.5% | `$+99.59` | 57.18% | 5.91 | 18.10 | 16.96 | 2.24 |
| **BNB** | 62.35% | 85 | 57.9% | 63.6% | `$+23.42` | 30.08% | 2.58 | 7.46 | 7.58 | 1.48 |
| **BTC** | 0.00% | 0 | 0.0% | 0.0% | `$+0.00` | 0.00% | 0.00 | 0.00 | 0.00 | 0.00 |
| **DOGE** | 58.33% | 12 | 0.0% | 63.6% | `$+2.99` | 18.28% | 0.60 | 1.32 | 1.59 | 1.33 |
| **DOT** | 62.50% | 104 | 41.7% | 68.8% | `$+68.51` | 41.50% | 4.26 | 10.40 | 16.08 | 1.92 |
| **ETH** | 78.05% | 82 | 75.0% | 78.4% | `$+100.43` | 28.37% | 8.37 | 27.10 | 34.48 | 4.37 |
| **HBAR** | 62.65% | 83 | 56.5% | 65.0% | `$+37.73` | 35.45% | 3.34 | 8.88 | 10.37 | 1.72 |
| **LINK** | 68.00% | 125 | 61.8% | 70.3% | `$+111.95` | 40.34% | 6.48 | 20.73 | 27.03 | 2.47 |
| **NEAR** | 67.68% | 99 | 50.0% | 72.2% | `$+118.60` | 53.45% | 7.52 | 23.51 | 21.61 | 3.43 |
| **ONDO** | 63.21% | 106 | 61.9% | 63.5% | `$+85.14` | 70.47% | 4.91 | 12.36 | 11.77 | 2.15 |
| **POL** | 57.89% | 76 | 59.1% | 57.4% | `$+66.62` | 79.81% | 4.64 | 13.89 | 8.13 | 2.23 |
| **SOL** | 54.01% | 137 | 48.3% | 55.6% | `$+35.11` | 116.22% | 1.98 | 6.82 | 2.94 | 1.31 |
| **SUI** | 66.12% | 121 | 63.0% | 67.0% | `$+90.91` | 71.80% | 4.82 | 12.01 | 12.33 | 2.13 |
| **TAO** | 59.72% | 72 | 58.3% | 60.4% | `$+32.13` | 60.45% | 2.14 | 5.02 | 5.18 | 1.49 |
| **TON** | 61.86% | 97 | 60.0% | 62.2% | `$+65.11` | 59.37% | 4.72 | 14.54 | 10.68 | 2.05 |
| **TRX** | 70.21% | 47 | 66.7% | 73.9% | `$+13.22` | 14.79% | 3.61 | 7.57 | 8.71 | 2.25 |
| **XRP** | 70.83% | 96 | 82.6% | 67.1% | `$+73.65` | 35.53% | 7.02 | 16.80 | 20.19 | 2.97 |

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
