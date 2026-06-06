# 📊 Holdout Backtest Report: `cascade_v2.5_hybrid_accel_scenario_W2`

**Tanggal Pembuatan**: 2026-06-03 21:14:20 UTC
**Model Run ID**: `cascade_v2.5_hybrid_accel_scenario_W2`
**Periode Pengujian (Temporal OOS)**: `2025-11-01 - 2026-04-01`

> [!NOTE]
> **Ringkasan Portofolio Eksekutif**:
> *   **Total Net Profit**: **$-338.02 USD** (ROI Portofolio: **-16.10%**)
> *   **Rata-rata Win Rate**: **48.70%** | Total Trades: **10,180**
> *   **Rata-rata Max Drawdown (5x)**: **311.16%**
> *   **Risk-Adjusted**: Sharpe: **-0.28** | Sortino: **-0.65** | Calmar: **0.58** | Profit Factor: **1.02**

## 📈 Performa Scorecard Portofolio

| Metrik Utama | Nilai Portofolio | Catatan |
|:---|:---:|:---|
| **Total Net Profit ($)** | `$-338.02` | Akumulasi keuntungan bersih 5x leverage |
| **Portfolio ROI (%)** | `-16.10%` | ROI berdasarkan kapital portofolio $100/koin |
| **Overall Win Rate** | `48.70%` | Rasio kemenangan rata-rata seluruh aset |
| **Total Trades** | `10,180` | Jumlah total posisi yang dieksekusi |
| **Rata-rata Trade / Bulan** | `98.4` | Rata-rata frekuensi trade bulanan portofolio |
| **Rata-rata Trade / Hari** | `3.23` | Rata-rata frekuensi trade harian portofolio |
| **Max Drawdown (5x)** | `311.16%` | Rata-rata penurunan terdalam portofolio |
| **Sharpe Ratio** | `-0.28` | Efisiensi profit terhadap volatilitas portofolio |
| **Sortino Ratio** | `-0.65` | Efisiensi profit terhadap downside deviation |
| **Calmar Ratio** | `0.58` | Rasio return tahunan terhadap drawdown |
| **Profit Factor** | `1.02` | Rasio gross profit dibagi gross loss |
| **Max Consecutive Loss** | `19` trades | Streak kekalahan beruntun terpanjang |
| **Worst Single Trade PnL** | `-25.90%` | Kerugian terdalam dalam satu trade tunggal |
| **95% Trades Loss Under** | `11.20%` | Nilai risiko (VaR P95) kerugian maksimal |

## ↕️ Analisis Arah Signal (LONG vs SHORT)

| Arah Posisi | Jumlah Trade | Distribusi | Menang | Kalah | Win Rate | PnL Bersih ($) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **LONG** | 4,692 | 46.1% | 1,977 | 2,715 | 42.14% | -1,441.80 |
| **SHORT** | 5,488 | 53.9% | 3,074 | 2,414 | 56.01% | +1,103.78 |

### Rincian Rata-rata Profitabilitas per Trade

| Tipe Trade | PnL Rata-rata ($) | PnL Rata-rata (%) |
|:---|:---:|:---:|
| **Trade Menang (Wins)** | `$+1.7360` | `+6.94%` |
| **Trade Kalah (Losses)** | `$-1.7755` | `-7.10%` |

## 📅 Scorecard Bulanan Portofolio

| Bulan | Total Trades | Wins | Losses | Win Rate | Net PnL ($) |
|:---|:---:|:---:|:---:|:---:|:---:|
| 2025-11 | 1702 | 871 | 831 | 51.18% | $-13.19 |
| 2025-12 | 2098 | 1063 | 1035 | 50.67% | $+227.57 |
| 2026-01 | 2083 | 1089 | 994 | 52.28% | $+132.47 |
| 2026-02 | 1992 | 881 | 1111 | 44.23% | $-553.53 |
| 2026-03 | 2305 | 1147 | 1158 | 49.76% | $-131.34 |

## 🚪 Distribusi Alasan Keluar Posisi (Exit Reasons)

| Alasan Exit | Jumlah | Persentase | Wins | Losses | Win Rate | PnL Bersih ($) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| `guardian_exit` | 6,821 | 67.0% | 4,908 | 1,913 | 71.95% | $+5,664.63 |
| `sl_hit` | 3,196 | 31.4% | 3 | 3,193 | 0.09% | $-6,062.01 |
| `time_exit` | 163 | 1.6% | 140 | 23 | 85.89% | $+59.36 |

## 🪙 Scorecard Per Koin (Detailed Assets)

| Token | Win Rate | Trades | LONG WR | SHORT WR | Net PnL ($) | Max DD | Sharpe | Sortino | Calmar | PF |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1000PEPE** | 50.00% | 14 | 0.0% | 50.0% | `$+8.04` | 17.85% | 1.12 | 4.59 | 4.39 | 1.73 |
| **1000SHIB** | 67.65% | 34 | 37.5% | 76.9% | `$+17.12` | 18.83% | 2.58 | 6.85 | 8.85 | 1.93 |
| **ADA** | 48.85% | 522 | 36.9% | 59.1% | `$-11.92` | 400.13% | -0.36 | -1.07 | -0.29 | 0.98 |
| **ARB** | 48.86% | 571 | 42.5% | 54.7% | `$+26.69` | 259.33% | 0.74 | 2.06 | 1.00 | 1.05 |
| **AVAX** | 50.47% | 640 | 42.4% | 56.8% | `$+36.78` | 312.48% | 1.09 | 3.02 | 1.15 | 1.07 |
| **BNB** | 48.63% | 656 | 42.9% | 52.9% | `$-79.15` | 394.33% | -3.18 | -8.35 | -1.96 | 0.83 |
| **BTC** | 0.00% | 0 | 0.0% | 0.0% | `$+0.00` | 0.00% | 0.00 | 0.00 | 0.00 | 0.00 |
| **DOGE** | 60.00% | 60 | 36.4% | 65.3% | `$+8.96` | 61.49% | 1.12 | 2.37 | 1.42 | 1.24 |
| **DOT** | 46.47% | 652 | 35.4% | 56.9% | `$-127.18` | 634.15% | -3.47 | -8.65 | -1.95 | 0.81 |
| **ETH** | 52.80% | 517 | 42.0% | 57.8% | `$+49.91` | 352.28% | 1.69 | 4.34 | 1.38 | 1.12 |
| **HBAR** | 47.25% | 527 | 40.1% | 55.9% | `$-71.78` | 306.36% | -2.54 | -6.62 | -2.28 | 0.85 |
| **LINK** | 50.80% | 689 | 46.2% | 54.4% | `$+13.77` | 244.11% | 0.39 | 1.10 | 0.55 | 1.02 |
| **NEAR** | 53.01% | 632 | 41.1% | 64.4% | `$+90.99` | 295.82% | 2.43 | 6.05 | 3.00 | 1.16 |
| **ONDO** | 49.82% | 570 | 45.1% | 53.7% | `$+9.70` | 389.07% | 0.27 | 0.76 | 0.24 | 1.02 |
| **POL** | 47.93% | 580 | 43.7% | 51.6% | `$+19.27` | 369.09% | 0.55 | 1.58 | 0.51 | 1.03 |
| **SOL** | 45.74% | 680 | 40.0% | 50.0% | `$-108.13` | 658.47% | -3.16 | -8.42 | -1.60 | 0.83 |
| **SUI** | 51.94% | 620 | 42.4% | 60.6% | `$+10.95` | 401.60% | 0.28 | 0.76 | 0.27 | 1.02 |
| **TAO** | 48.69% | 651 | 42.9% | 55.8% | `$-121.87` | 487.50% | -2.87 | -6.34 | -2.44 | 0.84 |
| **TON** | 44.63% | 670 | 42.2% | 46.5% | `$-148.01` | 596.49% | -4.82 | -13.48 | -2.42 | 0.76 |
| **TRX** | 56.38% | 337 | 49.7% | 65.1% | `$+10.99` | 85.86% | 1.22 | 3.28 | 1.25 | 1.10 |
| **XRP** | 52.87% | 558 | 43.7% | 61.0% | `$+26.84` | 249.04% | 0.97 | 2.46 | 1.05 | 1.06 |

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
