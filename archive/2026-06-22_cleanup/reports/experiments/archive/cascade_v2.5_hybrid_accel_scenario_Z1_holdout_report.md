# 📊 Holdout Backtest Report: `cascade_v2.5_hybrid_accel_scenario_Z1`

**Tanggal Pembuatan**: 2026-06-03 22:02:32 UTC
**Model Run ID**: `cascade_v2.5_hybrid_accel_scenario_Z1`
**Periode Pengujian (Temporal OOS)**: `2025-11-01 - 2026-04-01`

> [!NOTE]
> **Ringkasan Portofolio Eksekutif**:
> *   **Total Net Profit**: **$+1,701.68 USD** (ROI Portofolio: **+81.03%**)
> *   **Rata-rata Win Rate**: **62.03%** | Total Trades: **3,715**
> *   **Rata-rata Max Drawdown (5x)**: **87.16%**
> *   **Risk-Adjusted**: Sharpe: **4.32** | Sortino: **11.24** | Calmar: **14.66** | Profit Factor: **3.10**

## 📈 Performa Scorecard Portofolio

| Metrik Utama | Nilai Portofolio | Catatan |
|:---|:---:|:---|
| **Total Net Profit ($)** | `$+1,701.68` | Akumulasi keuntungan bersih 5x leverage |
| **Portfolio ROI (%)** | `+81.03%` | ROI berdasarkan kapital portofolio $100/koin |
| **Overall Win Rate** | `62.03%` | Rasio kemenangan rata-rata seluruh aset |
| **Total Trades** | `3,715` | Jumlah total posisi yang dieksekusi |
| **Rata-rata Trade / Bulan** | `35.9` | Rata-rata frekuensi trade bulanan portofolio |
| **Rata-rata Trade / Hari** | `1.18` | Rata-rata frekuensi trade harian portofolio |
| **Max Drawdown (5x)** | `87.16%` | Rata-rata penurunan terdalam portofolio |
| **Sharpe Ratio** | `4.32` | Efisiensi profit terhadap volatilitas portofolio |
| **Sortino Ratio** | `11.24` | Efisiensi profit terhadap downside deviation |
| **Calmar Ratio** | `14.66` | Rasio return tahunan terhadap drawdown |
| **Profit Factor** | `3.10` | Rasio gross profit dibagi gross loss |
| **Max Consecutive Loss** | `13` trades | Streak kekalahan beruntun terpanjang |
| **Worst Single Trade PnL** | `-24.90%` | Kerugian terdalam dalam satu trade tunggal |
| **95% Trades Loss Under** | `11.10%` | Nilai risiko (VaR P95) kerugian maksimal |

## ↕️ Analisis Arah Signal (LONG vs SHORT)

| Arah Posisi | Jumlah Trade | Distribusi | Menang | Kalah | Win Rate | PnL Bersih ($) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **LONG** | 1,692 | 45.5% | 903 | 789 | 53.37% | +370.36 |
| **SHORT** | 2,023 | 54.5% | 1,300 | 723 | 64.26% | +1,331.32 |

### Rincian Rata-rata Profitabilitas per Trade

| Tipe Trade | PnL Rata-rata ($) | PnL Rata-rata (%) |
|:---|:---:|:---:|
| **Trade Menang (Wins)** | `$+2.0115` | `+8.05%` |
| **Trade Kalah (Losses)** | `$-1.8053` | `-7.22%` |

## 📅 Scorecard Bulanan Portofolio

| Bulan | Total Trades | Wins | Losses | Win Rate | Net PnL ($) |
|:---|:---:|:---:|:---:|:---:|:---:|
| 2025-11 | 704 | 395 | 309 | 56.11% | $+244.55 |
| 2025-12 | 746 | 483 | 263 | 64.75% | $+560.78 |
| 2026-01 | 721 | 462 | 259 | 64.08% | $+525.65 |
| 2026-02 | 723 | 364 | 359 | 50.35% | $-23.32 |
| 2026-03 | 821 | 499 | 322 | 60.78% | $+394.02 |

## 🚪 Distribusi Alasan Keluar Posisi (Exit Reasons)

| Alasan Exit | Jumlah | Persentase | Wins | Losses | Win Rate | PnL Bersih ($) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| `guardian_exit` | 2,873 | 77.3% | 2,157 | 716 | 75.08% | $+3,221.86 |
| `sl_hit` | 790 | 21.3% | 2 | 788 | 0.25% | $-1,539.63 |
| `time_exit` | 52 | 1.4% | 44 | 8 | 84.62% | $+19.45 |

## 🪙 Scorecard Per Koin (Detailed Assets)

| Token | Win Rate | Trades | LONG WR | SHORT WR | Net PnL ($) | Max DD | Sharpe | Sortino | Calmar | PF |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1000PEPE** | 94.12% | 17 | 75.0% | 100.0% | `$+51.98` | 8.38% | 7.80 | 0.00 | 60.42 | 25.80 |
| **1000SHIB** | 64.29% | 14 | 60.0% | 66.7% | `$+7.49` | 6.32% | 2.01 | 6.70 | 11.55 | 2.90 |
| **ADA** | 64.63% | 147 | 60.0% | 68.3% | `$+126.00` | 51.87% | 6.37 | 19.54 | 23.66 | 2.34 |
| **ARB** | 54.30% | 221 | 50.0% | 59.2% | `$+97.91` | 88.85% | 3.74 | 9.37 | 10.73 | 1.51 |
| **AVAX** | 56.80% | 294 | 52.5% | 59.9% | `$+84.93` | 139.56% | 3.31 | 8.36 | 5.93 | 1.35 |
| **BNB** | 55.68% | 176 | 46.8% | 62.9% | `$+18.37` | 69.95% | 1.36 | 4.32 | 2.56 | 1.15 |
| **BTC** | 50.00% | 2 | 0.0% | 50.0% | `$+0.84` | 6.05% | 0.34 | 0.00 | 1.36 | 1.56 |
| **DOGE** | 83.87% | 31 | 80.0% | 87.5% | `$+57.20` | 13.85% | 6.32 | 22.95 | 40.22 | 6.60 |
| **DOT** | 57.14% | 238 | 48.6% | 64.1% | `$+64.80` | 117.81% | 2.87 | 7.08 | 5.36 | 1.35 |
| **ETH** | 68.75% | 160 | 64.1% | 71.9% | `$+130.54` | 56.21% | 7.59 | 24.14 | 22.62 | 2.55 |
| **HBAR** | 58.00% | 200 | 49.5% | 67.4% | `$+35.64` | 45.06% | 2.00 | 5.64 | 7.70 | 1.23 |
| **LINK** | 63.01% | 246 | 59.6% | 65.7% | `$+142.92` | 80.11% | 6.11 | 20.46 | 17.38 | 1.81 |
| **NEAR** | 57.56% | 238 | 43.7% | 68.2% | `$+119.03` | 70.83% | 4.60 | 10.63 | 16.37 | 1.62 |
| **ONDO** | 61.79% | 212 | 60.7% | 62.5% | `$+150.66` | 147.42% | 6.26 | 17.26 | 9.95 | 1.97 |
| **POL** | 55.71% | 210 | 51.8% | 59.8% | `$+97.93` | 178.07% | 4.36 | 11.98 | 5.36 | 1.60 |
| **SOL** | 54.11% | 292 | 49.1% | 57.2% | `$+83.25` | 147.57% | 3.11 | 10.05 | 5.50 | 1.33 |
| **SUI** | 61.83% | 241 | 53.1% | 67.8% | `$+150.71` | 188.00% | 5.84 | 14.82 | 7.81 | 1.87 |
| **TAO** | 55.71% | 219 | 51.5% | 62.6% | `$+85.67` | 216.10% | 3.10 | 6.97 | 3.86 | 1.41 |
| **TON** | 54.39% | 239 | 44.7% | 60.7% | `$+61.22` | 144.69% | 2.96 | 8.23 | 4.12 | 1.34 |
| **TRX** | 65.85% | 123 | 63.4% | 70.7% | `$+23.43` | 23.29% | 4.00 | 10.99 | 9.80 | 1.71 |
| **XRP** | 65.13% | 195 | 64.6% | 65.5% | `$+111.14` | 30.41% | 6.75 | 16.63 | 35.60 | 2.11 |

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
