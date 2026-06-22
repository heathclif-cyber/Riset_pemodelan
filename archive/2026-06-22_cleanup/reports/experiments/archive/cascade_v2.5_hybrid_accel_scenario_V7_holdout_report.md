# 📊 Holdout Backtest Report: `cascade_v2.5_hybrid_accel_scenario_V7`

**Tanggal Pembuatan**: 2026-06-03 20:49:03 UTC
**Model Run ID**: `cascade_v2.5_hybrid_accel_scenario_V7`
**Periode Pengujian (Temporal OOS)**: `2025-11-01 - 2026-04-01`

> [!NOTE]
> **Ringkasan Portofolio Eksekutif**:
> *   **Total Net Profit**: **$+1,586.49 USD** (ROI Portofolio: **+75.55%**)
> *   **Rata-rata Win Rate**: **60.89%** | Total Trades: **2,870**
> *   **Rata-rata Max Drawdown (5x)**: **80.02%**
> *   **Risk-Adjusted**: Sharpe: **4.35** | Sortino: **11.47** | Calmar: **14.12** | Profit Factor: **2.78**

## 📈 Performa Scorecard Portofolio

| Metrik Utama | Nilai Portofolio | Catatan |
|:---|:---:|:---|
| **Total Net Profit ($)** | `$+1,586.49` | Akumulasi keuntungan bersih 5x leverage |
| **Portfolio ROI (%)** | `+75.55%` | ROI berdasarkan kapital portofolio $100/koin |
| **Overall Win Rate** | `60.89%` | Rasio kemenangan rata-rata seluruh aset |
| **Total Trades** | `2,870` | Jumlah total posisi yang dieksekusi |
| **Rata-rata Trade / Bulan** | `27.7` | Rata-rata frekuensi trade bulanan portofolio |
| **Rata-rata Trade / Hari** | `0.91` | Rata-rata frekuensi trade harian portofolio |
| **Max Drawdown (5x)** | `80.02%` | Rata-rata penurunan terdalam portofolio |
| **Sharpe Ratio** | `4.35` | Efisiensi profit terhadap volatilitas portofolio |
| **Sortino Ratio** | `11.47` | Efisiensi profit terhadap downside deviation |
| **Calmar Ratio** | `14.12` | Rasio return tahunan terhadap drawdown |
| **Profit Factor** | `2.78` | Rasio gross profit dibagi gross loss |
| **Max Consecutive Loss** | `10` trades | Streak kekalahan beruntun terpanjang |
| **Worst Single Trade PnL** | `-24.90%` | Kerugian terdalam dalam satu trade tunggal |
| **95% Trades Loss Under** | `11.20%` | Nilai risiko (VaR P95) kerugian maksimal |

## ↕️ Analisis Arah Signal (LONG vs SHORT)

| Arah Posisi | Jumlah Trade | Distribusi | Menang | Kalah | Win Rate | PnL Bersih ($) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **LONG** | 1,425 | 49.7% | 776 | 649 | 54.46% | +412.54 |
| **SHORT** | 1,445 | 50.3% | 966 | 479 | 66.85% | +1,173.95 |

### Rincian Rata-rata Profitabilitas per Trade

| Tipe Trade | PnL Rata-rata ($) | PnL Rata-rata (%) |
|:---|:---:|:---:|
| **Trade Menang (Wins)** | `$+2.0872` | `+8.35%` |
| **Trade Kalah (Losses)** | `$-1.8169` | `-7.27%` |

## 📅 Scorecard Bulanan Portofolio

| Bulan | Total Trades | Wins | Losses | Win Rate | Net PnL ($) |
|:---|:---:|:---:|:---:|:---:|:---:|
| 2025-11 | 584 | 338 | 246 | 57.88% | $+278.07 |
| 2025-12 | 568 | 377 | 191 | 66.37% | $+511.81 |
| 2026-01 | 538 | 349 | 189 | 64.87% | $+450.81 |
| 2026-02 | 558 | 285 | 273 | 51.08% | $-29.83 |
| 2026-03 | 622 | 393 | 229 | 63.18% | $+375.63 |

## 🚪 Distribusi Alasan Keluar Posisi (Exit Reasons)

| Alasan Exit | Jumlah | Persentase | Wins | Losses | Win Rate | PnL Bersih ($) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| `guardian_exit` | 2,236 | 77.9% | 1,707 | 529 | 76.34% | $+2,746.38 |
| `sl_hit` | 594 | 20.7% | 0 | 594 | 0.00% | $-1,175.88 |
| `time_exit` | 40 | 1.4% | 35 | 5 | 87.50% | $+15.99 |

## 🪙 Scorecard Per Koin (Detailed Assets)

| Token | Win Rate | Trades | LONG WR | SHORT WR | Net PnL ($) | Max DD | Sharpe | Sortino | Calmar | PF |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1000PEPE** | 90.91% | 11 | 50.0% | 100.0% | `$+32.66` | 8.38% | 5.97 | 0.00 | 37.96 | 16.58 |
| **1000SHIB** | 72.73% | 11 | 66.7% | 75.0% | `$+8.64` | 4.88% | 2.47 | 6.38 | 17.24 | 4.69 |
| **ADA** | 66.12% | 121 | 63.3% | 68.8% | `$+117.87` | 32.55% | 6.50 | 19.09 | 35.27 | 2.62 |
| **ARB** | 53.80% | 171 | 52.4% | 55.9% | `$+72.96` | 126.49% | 3.01 | 7.79 | 5.62 | 1.46 |
| **AVAX** | 59.00% | 239 | 52.7% | 64.3% | `$+103.83` | 129.08% | 4.35 | 10.98 | 7.83 | 1.55 |
| **BNB** | 58.40% | 125 | 47.5% | 68.2% | `$+23.69` | 44.24% | 2.03 | 6.65 | 5.22 | 1.29 |
| **BTC** | 0.00% | 0 | 0.0% | 0.0% | `$+0.00` | 0.00% | 0.00 | 0.00 | 0.00 | 0.00 |
| **DOGE** | 83.87% | 31 | 80.0% | 87.5% | `$+57.20` | 13.85% | 6.32 | 22.95 | 40.22 | 6.60 |
| **DOT** | 60.77% | 181 | 49.5% | 72.2% | `$+91.95` | 86.47% | 4.52 | 10.84 | 10.36 | 1.72 |
| **ETH** | 68.89% | 135 | 63.9% | 73.0% | `$+115.40` | 49.50% | 7.19 | 23.07 | 22.71 | 2.63 |
| **HBAR** | 58.60% | 157 | 50.6% | 69.1% | `$+31.13` | 45.96% | 1.95 | 5.45 | 6.60 | 1.26 |
| **LINK** | 66.67% | 189 | 60.4% | 72.5% | `$+145.70` | 68.79% | 7.00 | 21.59 | 20.63 | 2.20 |
| **NEAR** | 57.30% | 178 | 41.2% | 70.4% | `$+110.99` | 62.81% | 4.80 | 11.93 | 17.21 | 1.79 |
| **ONDO** | 60.90% | 156 | 56.9% | 63.7% | `$+104.59` | 159.24% | 4.96 | 12.98 | 6.40 | 1.89 |
| **POL** | 54.60% | 163 | 50.5% | 60.0% | `$+72.47` | 187.66% | 3.50 | 9.78 | 3.76 | 1.53 |
| **SOL** | 55.09% | 216 | 52.7% | 56.9% | `$+82.86` | 134.40% | 3.52 | 11.32 | 6.01 | 1.47 |
| **SUI** | 63.92% | 194 | 55.4% | 70.3% | `$+142.74` | 181.34% | 5.92 | 15.21 | 7.67 | 2.05 |
| **TAO** | 56.98% | 172 | 52.2% | 66.1% | `$+88.85` | 182.62% | 3.55 | 7.89 | 4.74 | 1.56 |
| **TON** | 57.93% | 164 | 46.6% | 67.0% | `$+65.53` | 103.25% | 3.69 | 10.71 | 6.18 | 1.55 |
| **TRX** | 66.67% | 102 | 63.4% | 74.2% | `$+18.34` | 23.98% | 3.53 | 10.21 | 7.45 | 1.68 |
| **XRP** | 65.58% | 154 | 70.0% | 61.9% | `$+99.10` | 35.02% | 6.51 | 16.17 | 27.56 | 2.27 |

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
