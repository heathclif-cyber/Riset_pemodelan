# 📊 Holdout Backtest Report: `cascade_v2.5_hybrid_accel_scenario_T9`

**Tanggal Pembuatan**: 2026-06-02 21:19:08 UTC
**Model Run ID**: `cascade_v2.5_hybrid_accel_scenario_T9`
**Periode Pengujian (Temporal OOS)**: `2025-11-01 - 2026-04-01`

> [!NOTE]
> **Ringkasan Portofolio Eksekutif**:
> *   **Total Net Profit**: **$+1,441.41 USD** (ROI Portofolio: **+68.64%**)
> *   **Rata-rata Win Rate**: **60.99%** | Total Trades: **3,414**
> *   **Rata-rata Max Drawdown (5x)**: **80.82%**
> *   **Risk-Adjusted**: Sharpe: **3.52** | Sortino: **9.98** | Calmar: **8.89** | Profit Factor: **1.45**

## 📈 Performa Scorecard Portofolio

| Metrik Utama | Nilai Portofolio | Catatan |
|:---|:---:|:---|
| **Total Net Profit ($)** | `$+1,441.41` | Akumulasi keuntungan bersih 5x leverage |
| **Portfolio ROI (%)** | `+68.64%` | ROI berdasarkan kapital portofolio $100/koin |
| **Overall Win Rate** | `60.99%` | Rasio kemenangan rata-rata seluruh aset |
| **Total Trades** | `3,414` | Jumlah total posisi yang dieksekusi |
| **Rata-rata Trade / Bulan** | `33.0` | Rata-rata frekuensi trade bulanan portofolio |
| **Rata-rata Trade / Hari** | `1.08` | Rata-rata frekuensi trade harian portofolio |
| **Max Drawdown (5x)** | `80.82%` | Rata-rata penurunan terdalam portofolio |
| **Sharpe Ratio** | `3.52` | Efisiensi profit terhadap volatilitas portofolio |
| **Sortino Ratio** | `9.98` | Efisiensi profit terhadap downside deviation |
| **Calmar Ratio** | `8.89` | Rasio return tahunan terhadap drawdown |
| **Profit Factor** | `1.45` | Rasio gross profit dibagi gross loss |
| **Max Consecutive Loss** | `11` trades | Streak kekalahan beruntun terpanjang |
| **Worst Single Trade PnL** | `-24.90%` | Kerugian terdalam dalam satu trade tunggal |
| **95% Trades Loss Under** | `11.00%` | Nilai risiko (VaR P95) kerugian maksimal |

## ↕️ Analisis Arah Signal (LONG vs SHORT)

| Arah Posisi | Jumlah Trade | Distribusi | Menang | Kalah | Win Rate | PnL Bersih ($) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **LONG** | 1,482 | 43.4% | 774 | 708 | 52.23% | +233.07 |
| **SHORT** | 1,932 | 56.6% | 1,231 | 701 | 63.72% | +1,208.34 |

### Rincian Rata-rata Profitabilitas per Trade

| Tipe Trade | PnL Rata-rata ($) | PnL Rata-rata (%) |
|:---|:---:|:---:|
| **Trade Menang (Wins)** | `$+1.9851` | `+7.94%` |
| **Trade Kalah (Losses)** | `$-1.8017` | `-7.21%` |

## 📅 Scorecard Bulanan Portofolio

| Bulan | Total Trades | Wins | Losses | Win Rate | Net PnL ($) |
|:---|:---:|:---:|:---:|:---:|:---:|
| 2025-11 | 626 | 349 | 277 | 55.75% | $+206.60 |
| 2025-12 | 697 | 447 | 250 | 64.13% | $+498.46 |
| 2026-01 | 658 | 420 | 238 | 63.83% | $+455.33 |
| 2026-02 | 664 | 330 | 334 | 49.70% | $-37.74 |
| 2026-03 | 769 | 459 | 310 | 59.69% | $+318.77 |

## 🚪 Distribusi Alasan Keluar Posisi (Exit Reasons)

| Alasan Exit | Jumlah | Persentase | Wins | Losses | Win Rate | PnL Bersih ($) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| `guardian_exit` | 2,635 | 77.2% | 1,965 | 670 | 74.57% | $+2,857.63 |
| `sl_hit` | 734 | 21.5% | 2 | 732 | 0.27% | $-1,431.62 |
| `time_exit` | 45 | 1.3% | 38 | 7 | 84.44% | $+15.40 |

## 🪙 Scorecard Per Koin (Detailed Assets)

| Token | Win Rate | Trades | LONG WR | SHORT WR | Net PnL ($) | Max DD | Sharpe | Sortino | Calmar | PF |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1000PEPE** | 100.00% | 1 | 0.0% | 100.0% | `$+3.80` | 0.00% | 0.00 | 0.00 | 0.00 | 0.00 |
| **1000SHIB** | 100.00% | 2 | 0.0% | 100.0% | `$+6.07` | 0.00% | 2.65 | 0.00 | 0.00 | 0.00 |
| **ADA** | 62.99% | 127 | 56.0% | 67.5% | `$+105.10` | 43.80% | 5.57 | 17.18 | 23.37 | 2.24 |
| **ARB** | 54.31% | 197 | 48.5% | 60.2% | `$+82.78` | 104.13% | 3.36 | 8.67 | 7.74 | 1.48 |
| **AVAX** | 57.91% | 278 | 54.6% | 60.0% | `$+98.74` | 95.07% | 3.98 | 10.29 | 10.12 | 1.45 |
| **BNB** | 54.91% | 173 | 46.2% | 62.1% | `$+12.50` | 69.95% | 0.94 | 2.96 | 1.74 | 1.11 |
| **BTC** | 0.00% | 0 | 0.0% | 0.0% | `$+0.00` | 0.00% | 0.00 | 0.00 | 0.00 | 0.00 |
| **DOGE** | 75.00% | 8 | 33.3% | 100.0% | `$+5.74` | 13.85% | 1.76 | 14.93 | 4.03 | 2.66 |
| **DOT** | 56.19% | 226 | 45.4% | 64.3% | `$+53.43` | 108.82% | 2.42 | 5.91 | 4.78 | 1.30 |
| **ETH** | 68.35% | 139 | 63.8% | 70.7% | `$+115.80` | 45.30% | 7.11 | 22.51 | 24.90 | 2.58 |
| **HBAR** | 58.33% | 180 | 50.0% | 67.4% | `$+35.11` | 44.64% | 2.05 | 6.03 | 7.66 | 1.25 |
| **LINK** | 64.02% | 239 | 61.2% | 66.2% | `$+151.76` | 69.07% | 6.61 | 22.20 | 21.40 | 1.92 |
| **NEAR** | 57.47% | 221 | 42.9% | 67.7% | `$+111.58` | 72.26% | 4.56 | 10.59 | 15.04 | 1.64 |
| **ONDO** | 61.81% | 199 | 61.3% | 62.1% | `$+138.37` | 138.24% | 5.89 | 17.13 | 9.75 | 1.93 |
| **POL** | 56.41% | 195 | 52.7% | 59.8% | `$+101.41` | 149.20% | 4.77 | 13.87 | 6.62 | 1.70 |
| **SOL** | 52.54% | 276 | 45.5% | 56.5% | `$+60.86` | 142.99% | 2.37 | 7.58 | 4.15 | 1.25 |
| **SUI** | 61.78% | 225 | 52.3% | 67.6% | `$+135.61` | 178.73% | 5.42 | 13.57 | 7.39 | 1.82 |
| **TAO** | 54.55% | 209 | 51.1% | 60.3% | `$+61.79` | 216.10% | 2.30 | 5.16 | 2.78 | 1.30 |
| **TON** | 54.87% | 226 | 44.0% | 61.3% | `$+53.78` | 144.69% | 2.75 | 7.43 | 3.62 | 1.32 |
| **TRX** | 65.49% | 113 | 63.0% | 70.0% | `$+21.63` | 23.29% | 3.80 | 10.28 | 9.05 | 1.70 |
| **XRP** | 63.89% | 180 | 62.0% | 65.1% | `$+85.55` | 37.01% | 5.52 | 13.27 | 22.51 | 1.89 |

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
