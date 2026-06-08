# 📊 Holdout Backtest Report: `cascade_v2.5_hybrid_accel_scenario_L`

**Tanggal Pembuatan**: 2026-06-02 20:28:35 UTC
**Model Run ID**: `cascade_v2.5_hybrid_accel_scenario_L`
**Periode Pengujian (Temporal OOS)**: `2025-11-01 - 2026-04-01`

> [!NOTE]
> **Ringkasan Portofolio Eksekutif**:
> *   **Total Net Profit**: **$+1,930.54 USD** (ROI Portofolio: **+91.93%**)
> *   **Rata-rata Win Rate**: **61.17%** | Total Trades: **4,783**
> *   **Rata-rata Max Drawdown (5x)**: **106.98%**
> *   **Risk-Adjusted**: Sharpe: **4.38** | Sortino: **11.98** | Calmar: **11.96** | Profit Factor: **1.98**

## 📈 Performa Scorecard Portofolio

| Metrik Utama | Nilai Portofolio | Catatan |
|:---|:---:|:---|
| **Total Net Profit ($)** | `$+1,930.54` | Akumulasi keuntungan bersih 5x leverage |
| **Portfolio ROI (%)** | `+91.93%` | ROI berdasarkan kapital portofolio $100/koin |
| **Overall Win Rate** | `61.17%` | Rasio kemenangan rata-rata seluruh aset |
| **Total Trades** | `4,783` | Jumlah total posisi yang dieksekusi |
| **Rata-rata Trade / Bulan** | `46.2` | Rata-rata frekuensi trade bulanan portofolio |
| **Rata-rata Trade / Hari** | `1.52` | Rata-rata frekuensi trade harian portofolio |
| **Max Drawdown (5x)** | `106.98%` | Rata-rata penurunan terdalam portofolio |
| **Sharpe Ratio** | `4.38` | Efisiensi profit terhadap volatilitas portofolio |
| **Sortino Ratio** | `11.98` | Efisiensi profit terhadap downside deviation |
| **Calmar Ratio** | `11.96` | Rasio return tahunan terhadap drawdown |
| **Profit Factor** | `1.98` | Rasio gross profit dibagi gross loss |
| **Max Consecutive Loss** | `19` trades | Streak kekalahan beruntun terpanjang |
| **Worst Single Trade PnL** | `-24.90%` | Kerugian terdalam dalam satu trade tunggal |
| **95% Trades Loss Under** | `11.30%` | Nilai risiko (VaR P95) kerugian maksimal |

## ↕️ Analisis Arah Signal (LONG vs SHORT)

| Arah Posisi | Jumlah Trade | Distribusi | Menang | Kalah | Win Rate | PnL Bersih ($) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **LONG** | 2,127 | 44.5% | 1,138 | 989 | 53.50% | +449.91 |
| **SHORT** | 2,656 | 55.5% | 1,677 | 979 | 63.14% | +1,480.63 |

### Rincian Rata-rata Profitabilitas per Trade

| Tipe Trade | PnL Rata-rata ($) | PnL Rata-rata (%) |
|:---|:---:|:---:|
| **Trade Menang (Wins)** | `$+1.9608` | `+7.84%` |
| **Trade Kalah (Losses)** | `$-1.8238` | `-7.30%` |

## 📅 Scorecard Bulanan Portofolio

| Bulan | Total Trades | Wins | Losses | Win Rate | Net PnL ($) |
|:---|:---:|:---:|:---:|:---:|:---:|
| 2025-11 | 928 | 511 | 417 | 55.06% | $+222.50 |
| 2025-12 | 943 | 610 | 333 | 64.69% | $+672.09 |
| 2026-01 | 943 | 601 | 342 | 63.73% | $+650.43 |
| 2026-02 | 925 | 477 | 448 | 51.57% | $+8.36 |
| 2026-03 | 1044 | 616 | 428 | 59.00% | $+377.15 |

## 🚪 Distribusi Alasan Keluar Posisi (Exit Reasons)

| Alasan Exit | Jumlah | Persentase | Wins | Losses | Win Rate | PnL Bersih ($) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| `guardian_exit` | 3,688 | 77.1% | 2,755 | 933 | 74.70% | $+3,943.53 |
| `sl_hit` | 1,027 | 21.5% | 2 | 1,025 | 0.19% | $-2,039.31 |
| `time_exit` | 68 | 1.4% | 58 | 10 | 85.29% | $+26.32 |

## 🪙 Scorecard Per Koin (Detailed Assets)

| Token | Win Rate | Trades | LONG WR | SHORT WR | Net PnL ($) | Max DD | Sharpe | Sortino | Calmar | PF |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1000PEPE** | 64.81% | 54 | 63.2% | 65.7% | `$+64.60` | 56.55% | 3.83 | 15.61 | 11.13 | 2.37 |
| **1000SHIB** | 66.07% | 56 | 53.3% | 80.8% | `$+33.07` | 29.04% | 3.44 | 7.36 | 11.09 | 2.13 |
| **ADA** | 63.64% | 187 | 60.0% | 66.7% | `$+147.36` | 51.31% | 6.64 | 21.89 | 27.97 | 2.18 |
| **ARB** | 54.71% | 276 | 48.2% | 61.3% | `$+111.66` | 127.52% | 3.86 | 9.81 | 8.53 | 1.45 |
| **AVAX** | 58.09% | 377 | 55.1% | 60.2% | `$+125.09` | 150.91% | 4.48 | 11.03 | 8.07 | 1.43 |
| **BNB** | 55.46% | 229 | 47.6% | 62.1% | `$+21.71` | 71.10% | 1.41 | 4.21 | 2.97 | 1.14 |
| **BTC** | 86.67% | 30 | 93.3% | 80.0% | `$+36.21` | 11.27% | 6.62 | 13.72 | 31.30 | 7.78 |
| **DOGE** | 67.65% | 68 | 74.1% | 63.4% | `$+78.96` | 27.72% | 5.81 | 18.12 | 27.74 | 3.14 |
| **DOT** | 56.96% | 309 | 48.1% | 63.6% | `$+76.84` | 115.29% | 3.03 | 7.71 | 6.49 | 1.32 |
| **ETH** | 63.51% | 211 | 56.1% | 69.9% | `$+120.88` | 150.53% | 5.79 | 16.45 | 7.82 | 1.88 |
| **HBAR** | 57.02% | 228 | 49.1% | 64.7% | `$+33.19` | 70.79% | 1.77 | 4.91 | 4.57 | 1.19 |
| **LINK** | 58.01% | 312 | 54.9% | 60.3% | `$+109.71` | 125.08% | 4.12 | 12.87 | 8.54 | 1.43 |
| **NEAR** | 58.03% | 305 | 44.9% | 67.4% | `$+136.77` | 115.01% | 4.78 | 11.18 | 11.58 | 1.56 |
| **ONDO** | 58.74% | 269 | 54.6% | 61.5% | `$+135.83` | 151.27% | 4.95 | 12.54 | 8.75 | 1.61 |
| **POL** | 56.98% | 265 | 51.6% | 62.0% | `$+113.15` | 141.14% | 4.52 | 12.34 | 7.81 | 1.54 |
| **SOL** | 54.84% | 341 | 48.8% | 58.2% | `$+87.66` | 193.01% | 3.13 | 9.10 | 4.42 | 1.31 |
| **SUI** | 62.99% | 281 | 55.6% | 68.1% | `$+192.74` | 171.64% | 6.82 | 18.33 | 10.94 | 1.97 |
| **TAO** | 55.56% | 270 | 53.0% | 59.6% | `$+80.69` | 265.27% | 2.61 | 5.51 | 2.96 | 1.30 |
| **TON** | 53.92% | 306 | 46.3% | 58.1% | `$+67.92` | 160.11% | 2.96 | 8.66 | 4.13 | 1.29 |
| **TRX** | 66.46% | 164 | 66.3% | 66.7% | `$+29.36` | 23.18% | 4.45 | 12.73 | 12.34 | 1.67 |
| **XRP** | 64.49% | 245 | 64.3% | 64.6% | `$+127.14` | 38.77% | 7.00 | 17.52 | 31.94 | 2.00 |

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
