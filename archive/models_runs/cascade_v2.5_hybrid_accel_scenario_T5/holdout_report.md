# 📊 Holdout Backtest Report: `cascade_v2.5_hybrid_accel_scenario_T5`

**Tanggal Pembuatan**: 2026-06-02 21:15:35 UTC
**Model Run ID**: `cascade_v2.5_hybrid_accel_scenario_T5`
**Periode Pengujian (Temporal OOS)**: `2025-11-01 - 2026-04-01`

> [!NOTE]
> **Ringkasan Portofolio Eksekutif**:
> *   **Total Net Profit**: **$+1,602.55 USD** (ROI Portofolio: **+76.31%**)
> *   **Rata-rata Win Rate**: **63.22%** | Total Trades: **2,884**
> *   **Rata-rata Max Drawdown (5x)**: **80.38%**
> *   **Risk-Adjusted**: Sharpe: **4.42** | Sortino: **11.51** | Calmar: **14.72** | Profit Factor: **3.10**

## 📈 Performa Scorecard Portofolio

| Metrik Utama | Nilai Portofolio | Catatan |
|:---|:---:|:---|
| **Total Net Profit ($)** | `$+1,602.55` | Akumulasi keuntungan bersih 5x leverage |
| **Portfolio ROI (%)** | `+76.31%` | ROI berdasarkan kapital portofolio $100/koin |
| **Overall Win Rate** | `63.22%` | Rasio kemenangan rata-rata seluruh aset |
| **Total Trades** | `2,884` | Jumlah total posisi yang dieksekusi |
| **Rata-rata Trade / Bulan** | `27.9` | Rata-rata frekuensi trade bulanan portofolio |
| **Rata-rata Trade / Hari** | `0.92` | Rata-rata frekuensi trade harian portofolio |
| **Max Drawdown (5x)** | `80.38%` | Rata-rata penurunan terdalam portofolio |
| **Sharpe Ratio** | `4.42` | Efisiensi profit terhadap volatilitas portofolio |
| **Sortino Ratio** | `11.51` | Efisiensi profit terhadap downside deviation |
| **Calmar Ratio** | `14.72` | Rasio return tahunan terhadap drawdown |
| **Profit Factor** | `3.10` | Rasio gross profit dibagi gross loss |
| **Max Consecutive Loss** | `10` trades | Streak kekalahan beruntun terpanjang |
| **Worst Single Trade PnL** | `-24.90%` | Kerugian terdalam dalam satu trade tunggal |
| **95% Trades Loss Under** | `11.20%` | Nilai risiko (VaR P95) kerugian maksimal |

## ↕️ Analisis Arah Signal (LONG vs SHORT)

| Arah Posisi | Jumlah Trade | Distribusi | Menang | Kalah | Win Rate | PnL Bersih ($) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **LONG** | 1,434 | 49.7% | 781 | 653 | 54.46% | +424.15 |
| **SHORT** | 1,450 | 50.3% | 970 | 480 | 66.90% | +1,178.40 |

### Rincian Rata-rata Profitabilitas per Trade

| Tipe Trade | PnL Rata-rata ($) | PnL Rata-rata (%) |
|:---|:---:|:---:|
| **Trade Menang (Wins)** | `$+2.0894` | `+8.36%` |
| **Trade Kalah (Losses)** | `$-1.8146` | `-7.26%` |

## 📅 Scorecard Bulanan Portofolio

| Bulan | Total Trades | Wins | Losses | Win Rate | Net PnL ($) |
|:---|:---:|:---:|:---:|:---:|:---:|
| 2025-11 | 591 | 341 | 250 | 57.70% | $+277.56 |
| 2025-12 | 569 | 378 | 191 | 66.43% | $+512.04 |
| 2026-01 | 539 | 350 | 189 | 64.94% | $+451.72 |
| 2026-02 | 561 | 287 | 274 | 51.16% | $-24.01 |
| 2026-03 | 624 | 395 | 229 | 63.30% | $+385.23 |

## 🚪 Distribusi Alasan Keluar Posisi (Exit Reasons)

| Alasan Exit | Jumlah | Persentase | Wins | Losses | Win Rate | PnL Bersih ($) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| `guardian_exit` | 2,246 | 77.9% | 1,715 | 531 | 76.36% | $+2,766.88 |
| `sl_hit` | 597 | 20.7% | 0 | 597 | 0.00% | $-1,180.78 |
| `time_exit` | 41 | 1.4% | 36 | 5 | 87.80% | $+16.46 |

## 🪙 Scorecard Per Koin (Detailed Assets)

| Token | Win Rate | Trades | LONG WR | SHORT WR | Net PnL ($) | Max DD | Sharpe | Sortino | Calmar | PF |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1000PEPE** | 93.75% | 16 | 75.0% | 100.0% | `$+45.86` | 8.38% | 7.23 | 0.00 | 53.30 | 22.88 |
| **1000SHIB** | 69.23% | 13 | 60.0% | 75.0% | `$+8.13` | 6.32% | 2.23 | 6.65 | 12.52 | 3.45 |
| **ADA** | 66.12% | 121 | 63.3% | 68.8% | `$+117.87` | 32.55% | 6.50 | 19.09 | 35.27 | 2.62 |
| **ARB** | 53.18% | 173 | 51.4% | 55.9% | `$+69.80` | 126.49% | 2.87 | 7.44 | 5.37 | 1.43 |
| **AVAX** | 59.00% | 239 | 52.7% | 64.3% | `$+103.83` | 129.08% | 4.35 | 10.98 | 7.83 | 1.55 |
| **BNB** | 58.40% | 125 | 47.5% | 68.2% | `$+23.69` | 44.24% | 2.03 | 6.65 | 5.22 | 1.29 |
| **BTC** | 50.00% | 2 | 0.0% | 50.0% | `$+0.84` | 6.05% | 0.34 | 0.00 | 1.36 | 1.56 |
| **DOGE** | 83.87% | 31 | 80.0% | 87.5% | `$+57.20` | 13.85% | 6.32 | 22.95 | 40.22 | 6.60 |
| **DOT** | 60.77% | 181 | 49.5% | 72.2% | `$+91.95` | 86.47% | 4.52 | 10.84 | 10.36 | 1.72 |
| **ETH** | 69.12% | 136 | 64.5% | 73.0% | `$+116.95` | 49.50% | 7.29 | 23.29 | 23.01 | 2.65 |
| **HBAR** | 58.60% | 157 | 50.6% | 69.1% | `$+31.13` | 45.96% | 1.95 | 5.45 | 6.60 | 1.26 |
| **LINK** | 66.67% | 189 | 60.4% | 72.5% | `$+145.70` | 68.79% | 7.00 | 21.59 | 20.63 | 2.20 |
| **NEAR** | 57.30% | 178 | 41.2% | 70.4% | `$+110.99` | 62.81% | 4.80 | 11.93 | 17.21 | 1.79 |
| **ONDO** | 60.90% | 156 | 56.9% | 63.7% | `$+104.59` | 159.24% | 4.96 | 12.98 | 6.40 | 1.89 |
| **POL** | 54.27% | 164 | 50.0% | 60.0% | `$+71.62` | 187.66% | 3.46 | 9.62 | 3.72 | 1.52 |
| **SOL** | 55.09% | 216 | 52.7% | 56.9% | `$+82.86` | 134.40% | 3.52 | 11.32 | 6.01 | 1.47 |
| **SUI** | 63.92% | 194 | 55.4% | 70.3% | `$+142.74` | 181.34% | 5.92 | 15.21 | 7.67 | 2.05 |
| **TAO** | 56.98% | 172 | 52.2% | 66.1% | `$+88.85` | 182.62% | 3.55 | 7.89 | 4.74 | 1.56 |
| **TON** | 58.18% | 165 | 47.3% | 67.0% | `$+70.51` | 103.25% | 3.92 | 11.48 | 6.65 | 1.59 |
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
