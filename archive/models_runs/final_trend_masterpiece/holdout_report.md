# 📊 Holdout Backtest Report: `cascade_v3.1`

**Tanggal Pembuatan**: 2026-05-27 23:08:25 UTC
**Model Run ID**: `final_trend_masterpiece`
**Periode Pengujian (Temporal OOS)**: `2025-11-01 - 2026-04-01`

> [!NOTE]
> **Ringkasan Portofolio Eksekutif**:
> *   **Total Net Profit**: **$+2,527.54 USD** (ROI Portofolio: **+126.38%**)
> *   **Rata-rata Win Rate**: **81.63%** | Total Trades: **1,735**
> *   **Rata-rata Max Drawdown (5x)**: **34.37%**
> *   **Risk-Adjusted**: Sharpe: **6.98** | Sortino: **16.34** | Calmar: **19.78** | Profit Factor: **6.76**

## 📈 Performa Scorecard Portofolio

| Metrik Utama | Nilai Portofolio | Catatan |
|:---|:---:|:---|
| **Total Net Profit ($)** | `$+2,527.54` | Akumulasi keuntungan bersih 5x leverage |
| **Portfolio ROI (%)** | `+126.38%` | ROI berdasarkan kapital portofolio $100/koin |
| **Overall Win Rate** | `81.63%` | Rasio kemenangan rata-rata seluruh aset |
| **Total Trades** | `1,735` | Jumlah total posisi yang dieksekusi |
| **Rata-rata Trade / Bulan** | `7.9` | Rata-rata frekuensi trade bulanan portofolio |
| **Rata-rata Trade / Hari** | `0.26` | Rata-rata frekuensi trade harian portofolio |
| **Max Drawdown (5x)** | `34.37%` | Rata-rata penurunan terdalam portofolio |
| **Sharpe Ratio** | `6.98` | Efisiensi profit terhadap volatilitas portofolio |
| **Sortino Ratio** | `16.34` | Efisiensi profit terhadap downside deviation |
| **Calmar Ratio** | `19.78` | Rasio return tahunan terhadap drawdown |
| **Profit Factor** | `6.76` | Rasio gross profit dibagi gross loss |
| **Max Consecutive Loss** | `8` trades | Streak kekalahan beruntun terpanjang |
| **Worst Single Trade PnL** | `-18.50%` | Kerugian terdalam dalam satu trade tunggal |
| **95% Trades Loss Under** | `8.00%` | Nilai risiko (VaR P95) kerugian maksimal |

## ↕️ Analisis Arah Signal (LONG vs SHORT)

| Arah Posisi | Jumlah Trade | Distribusi | Menang | Kalah | Win Rate | PnL Bersih ($) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **LONG** | 607 | 35.0% | 468 | 139 | 77.10% | +681.97 |
| **SHORT** | 1,128 | 65.0% | 955 | 173 | 84.66% | +1,845.57 |

### Rincian Rata-rata Profitabilitas per Trade

| Tipe Trade | PnL Rata-rata ($) | PnL Rata-rata (%) |
|:---|:---:|:---:|
| **Trade Menang (Wins)** | `$+2.1095` | `+8.44%` |
| **Trade Kalah (Losses)** | `$-1.5201` | `-6.08%` |

## 📅 Scorecard Bulanan Portofolio

| Bulan | Total Trades | Wins | Losses | Win Rate | Net PnL ($) |
|:---|:---:|:---:|:---:|:---:|:---:|
| 2025-05 | 343 | 308 | 35 | 89.80% | $+590.54 |
| 2025-06 | 310 | 289 | 21 | 93.23% | $+547.03 |
| 2025-07 | 320 | 292 | 28 | 91.25% | $+644.49 |
| 2025-08 | 264 | 233 | 31 | 88.26% | $+422.71 |
| 2025-09 | 66 | 39 | 27 | 59.09% | $+35.81 |
| 2025-10 | 88 | 65 | 23 | 73.86% | $+147.22 |
| 2025-11 | 85 | 49 | 36 | 57.65% | $+16.58 |
| 2025-12 | 69 | 40 | 29 | 57.97% | $+51.61 |
| 2026-01 | 90 | 52 | 38 | 57.78% | $+36.67 |
| 2026-02 | 38 | 21 | 17 | 55.26% | $+7.34 |
| 2026-03 | 62 | 35 | 27 | 56.45% | $+27.52 |

## 🚪 Distribusi Alasan Keluar Posisi (Exit Reasons)

| Alasan Exit | Jumlah | Persentase | Wins | Losses | Win Rate | PnL Bersih ($) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| `guardian_exit` | 1,571 | 90.5% | 1,409 | 162 | 89.69% | $+2,782.45 |
| `sl_hit` | 148 | 8.5% | 1 | 147 | 0.68% | $-259.84 |
| `time_exit` | 16 | 0.9% | 13 | 3 | 81.25% | $+4.92 |

## 🪙 Scorecard Per Koin (Detailed Assets)

| Token | Win Rate | Trades | LONG WR | SHORT WR | Net PnL ($) | Max DD | Sharpe | Sortino | Calmar | PF |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1000PEPE** | 81.55% | 103 | 80.0% | 81.8% | `$+251.70` | 30.93% | 10.04 | 22.83 | 35.60 | 11.61 |
| **1000SHIB** | 85.45% | 110 | 87.1% | 84.8% | `$+188.52` | 20.32% | 10.42 | 25.66 | 40.58 | 11.89 |
| **ADA** | 86.25% | 80 | 87.5% | 85.7% | `$+131.41` | 15.99% | 7.74 | 11.73 | 35.95 | 10.23 |
| **ARB** | 81.40% | 86 | 75.0% | 87.0% | `$+161.27` | 25.12% | 7.36 | 23.43 | 28.08 | 7.49 |
| **AVAX** | 84.27% | 89 | 78.8% | 87.5% | `$+136.50` | 29.64% | 8.42 | 21.97 | 20.14 | 7.35 |
| **BNB** | 83.91% | 87 | 92.9% | 82.2% | `$+84.56` | 35.38% | 6.61 | 17.67 | 10.46 | 6.60 |
| **DOGE** | 83.61% | 122 | 75.6% | 88.3% | `$+207.07` | 57.72% | 8.21 | 23.62 | 15.69 | 6.02 |
| **DOT** | 79.59% | 98 | 80.4% | 78.8% | `$+114.76` | 33.55% | 5.91 | 11.10 | 14.96 | 4.34 |
| **ETH** | 85.57% | 97 | 75.9% | 89.7% | `$+154.89` | 17.96% | 8.71 | 17.39 | 37.72 | 9.26 |
| **HBAR** | 84.72% | 72 | 72.0% | 91.5% | `$+110.02` | 63.73% | 7.33 | 15.56 | 7.55 | 6.57 |
| **LINK** | 79.73% | 74 | 69.2% | 85.4% | `$+89.49` | 34.55% | 4.89 | 14.11 | 11.33 | 4.37 |
| **NEAR** | 83.13% | 83 | 79.5% | 86.4% | `$+145.37` | 45.32% | 6.50 | 13.41 | 14.03 | 5.63 |
| **ONDO** | 84.34% | 83 | 72.2% | 93.6% | `$+145.72` | 43.22% | 7.61 | 12.97 | 14.75 | 8.53 |
| **POL** | 76.70% | 103 | 73.3% | 79.3% | `$+77.81` | 46.84% | 4.64 | 7.26 | 7.27 | 2.97 |
| **SOL** | 78.67% | 75 | 71.4% | 87.9% | `$+99.20` | 23.67% | 5.77 | 14.61 | 18.33 | 5.23 |
| **SUI** | 76.39% | 72 | 65.6% | 85.0% | `$+81.40` | 50.30% | 4.59 | 9.84 | 7.08 | 3.58 |
| **TAO** | 86.75% | 83 | 77.8% | 87.8% | `$+149.05` | 20.46% | 7.67 | 16.72 | 31.87 | 8.13 |
| **TON** | 73.85% | 65 | 96.3% | 57.9% | `$+59.12` | 46.81% | 4.79 | 17.77 | 5.52 | 3.81 |
| **TRX** | 67.16% | 67 | 54.2% | 74.4% | `$+30.70` | 32.00% | 3.75 | 13.98 | 4.20 | 3.01 |
| **XRP** | 89.53% | 86 | 92.0% | 88.5% | `$+108.97` | 13.86% | 8.55 | 15.27 | 34.39 | 8.61 |

## ⛓️ Daftar Fitur Aktif dalam Model

Total terdapat **93 fitur aktif** yang digunakan oleh LightGBM entry, LSTM Soft Confirmation, dan Exit Guardian v3:

<details>
<summary>▶ Klik untuk melihat daftar lengkap fitur aktif</summary>

1. `open`
2. `high`
3. `low`
4. `close`
5. `volume`
6. `volume_delta`
7. `cvd`
8. `buy_volume`
9. `sell_volume`
10. `MSB_BOS`
11. `CHoCH`
12. `bars_since_BOS`
13. `FVG_up`
14. `FVG_down`
15. `Buy_Liq`
16. `Sell_Liq`
17. `SFP_sweep`
18. `open_interest`
19. `funding_rate`
20. `ema_7_h1`
21. `ema_21_h1`
22. `ema_50_h1`
23. `ema_200_h1`
24. `ema_7_h4`
25. `ema_21_h4`
26. `ema_50_h4`
27. `ema_200_h4`
28. `rsi_6`
29. `stochrsi_k`
30. `stochrsi_d`
31. `atr_14_h1`
32. `atr_14_h4`
33. `PDH`
34. `PDL`
35. `PWH`
36. `PWL`
37. `Fib_618`
38. `Fib_786`
39. `POC`
40. `VAH`
41. `VAL`
42. `btc_dominance`
43. `fear_greed`
44. `market_session`
45. `log_ret_1`
46. `log_ret_5`
47. `log_ret_20`
48. `vol_ratio_20`
49. `hour_sin`
50. `hour_cos`
51. `dow_sin`
52. `dow_cos`
53. `time_to_funding_norm`
54. `long_short_ratio`
55. `dist_swing_high`
56. `dist_swing_low`
57. `price_in_range`
58. `swing_momentum`
59. `h4_trend`
60. `trend_strength`
61. `vol_regime`
62. `cvd_div_h4`
63. `cvd_slope_h4`
64. `vol_efficiency`
65. `absorption_z`
66. `funding_price_div`
67. `rsi_h4`
68. `rsi_divergence`
69. `wyckoff_phase`
70. `spring_upthrust`
71. `ofi_raw`
72. `ofi_acceleration`
73. `ofi_z_score`
74. `ofi_h4_delta`
75. `vwdp`
76. `vwdp_smooth`
77. `hidden_divergence`
78. `cvd_momentum_adv`
79. `absorption_at_swing`
80. `spread_to_volume`
81. `ultra_high_vol`
82. `no_demand`
83. `no_supply`
84. `effort_vs_result`
85. `ema_21_slope_h4`
86. `ema_50_slope_h4`
87. `price_vs_ema_50_h4`
88. `rsi_slope_h4`
89. `atr_percent_h4`
90. `range_expansion_h4`
91. `trend_accel_4h`
92. `vol_price_confirm`
93. `dist_from_8h_high`

</details>
