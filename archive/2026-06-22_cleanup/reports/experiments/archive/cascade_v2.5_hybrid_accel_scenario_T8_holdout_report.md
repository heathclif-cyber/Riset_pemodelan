# 📊 Holdout Backtest Report: `cascade_v2.5_hybrid_accel_scenario_T8`

**Tanggal Pembuatan**: 2026-06-02 21:18:44 UTC
**Model Run ID**: `cascade_v2.5_hybrid_accel_scenario_T8`
**Periode Pengujian (Temporal OOS)**: `2025-11-01 - 2026-04-01`

> [!NOTE]
> **Ringkasan Portofolio Eksekutif**:
> *   **Total Net Profit**: **$+1,397.88 USD** (ROI Portofolio: **+66.57%**)
> *   **Rata-rata Win Rate**: **61.45%** | Total Trades: **2,946**
> *   **Rata-rata Max Drawdown (5x)**: **77.71%**
> *   **Risk-Adjusted**: Sharpe: **3.58** | Sortino: **10.08** | Calmar: **9.31** | Profit Factor: **1.51**

## 📈 Performa Scorecard Portofolio

| Metrik Utama | Nilai Portofolio | Catatan |
|:---|:---:|:---|
| **Total Net Profit ($)** | `$+1,397.88` | Akumulasi keuntungan bersih 5x leverage |
| **Portfolio ROI (%)** | `+66.57%` | ROI berdasarkan kapital portofolio $100/koin |
| **Overall Win Rate** | `61.45%` | Rasio kemenangan rata-rata seluruh aset |
| **Total Trades** | `2,946` | Jumlah total posisi yang dieksekusi |
| **Rata-rata Trade / Bulan** | `28.5` | Rata-rata frekuensi trade bulanan portofolio |
| **Rata-rata Trade / Hari** | `0.94` | Rata-rata frekuensi trade harian portofolio |
| **Max Drawdown (5x)** | `77.71%` | Rata-rata penurunan terdalam portofolio |
| **Sharpe Ratio** | `3.58` | Efisiensi profit terhadap volatilitas portofolio |
| **Sortino Ratio** | `10.08` | Efisiensi profit terhadap downside deviation |
| **Calmar Ratio** | `9.31` | Rasio return tahunan terhadap drawdown |
| **Profit Factor** | `1.51` | Rasio gross profit dibagi gross loss |
| **Max Consecutive Loss** | `11` trades | Streak kekalahan beruntun terpanjang |
| **Worst Single Trade PnL** | `-24.90%` | Kerugian terdalam dalam satu trade tunggal |
| **95% Trades Loss Under** | `11.17%` | Nilai risiko (VaR P95) kerugian maksimal |

## ↕️ Analisis Arah Signal (LONG vs SHORT)

| Arah Posisi | Jumlah Trade | Distribusi | Menang | Kalah | Win Rate | PnL Bersih ($) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **LONG** | 1,350 | 45.8% | 709 | 641 | 52.52% | +252.93 |
| **SHORT** | 1,596 | 54.2% | 1,043 | 553 | 65.35% | +1,144.95 |

### Rincian Rata-rata Profitabilitas per Trade

| Tipe Trade | PnL Rata-rata ($) | PnL Rata-rata (%) |
|:---|:---:|:---:|
| **Trade Menang (Wins)** | `$+2.0214` | `+8.09%` |
| **Trade Kalah (Losses)** | `$-1.7953` | `-7.18%` |

## 📅 Scorecard Bulanan Portofolio

| Bulan | Total Trades | Wins | Losses | Win Rate | Net PnL ($) |
|:---|:---:|:---:|:---:|:---:|:---:|
| 2025-11 | 562 | 316 | 246 | 56.23% | $+210.86 |
| 2025-12 | 590 | 386 | 204 | 65.42% | $+481.74 |
| 2026-01 | 560 | 358 | 202 | 63.93% | $+407.31 |
| 2026-02 | 572 | 290 | 282 | 50.70% | $-16.21 |
| 2026-03 | 662 | 402 | 260 | 60.73% | $+314.17 |

## 🚪 Distribusi Alasan Keluar Posisi (Exit Reasons)

| Alasan Exit | Jumlah | Persentase | Wins | Losses | Win Rate | PnL Bersih ($) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| `guardian_exit` | 2,280 | 77.4% | 1,715 | 565 | 75.22% | $+2,603.23 |
| `sl_hit` | 623 | 21.1% | 1 | 622 | 0.16% | $-1,219.80 |
| `time_exit` | 43 | 1.5% | 36 | 7 | 83.72% | $+14.45 |

## 🪙 Scorecard Per Koin (Detailed Assets)

| Token | Win Rate | Trades | LONG WR | SHORT WR | Net PnL ($) | Max DD | Sharpe | Sortino | Calmar | PF |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1000PEPE** | 100.00% | 1 | 0.0% | 100.0% | `$+3.80` | 0.00% | 0.00 | 0.00 | 0.00 | 0.00 |
| **1000SHIB** | 100.00% | 2 | 0.0% | 100.0% | `$+6.07` | 0.00% | 2.65 | 0.00 | 0.00 | 0.00 |
| **ADA** | 63.06% | 111 | 57.1% | 67.7% | `$+98.76` | 58.22% | 5.48 | 16.72 | 16.52 | 2.34 |
| **ARB** | 53.01% | 166 | 47.8% | 59.5% | `$+60.97` | 124.70% | 2.62 | 6.81 | 4.76 | 1.39 |
| **AVAX** | 60.00% | 250 | 55.0% | 63.3% | `$+110.25` | 74.83% | 4.67 | 11.66 | 14.35 | 1.60 |
| **BNB** | 55.32% | 141 | 47.1% | 63.0% | `$+13.72` | 48.46% | 1.14 | 3.52 | 2.76 | 1.14 |
| **BTC** | 0.00% | 0 | 0.0% | 0.0% | `$+0.00` | 0.00% | 0.00 | 0.00 | 0.00 | 0.00 |
| **DOGE** | 75.00% | 8 | 33.3% | 100.0% | `$+5.74` | 13.85% | 1.76 | 14.93 | 4.03 | 2.66 |
| **DOT** | 56.57% | 198 | 45.6% | 65.7% | `$+55.77` | 108.82% | 2.64 | 6.23 | 4.99 | 1.36 |
| **ETH** | 68.80% | 125 | 63.8% | 71.8% | `$+105.70` | 35.39% | 6.72 | 21.47 | 29.09 | 2.58 |
| **HBAR** | 58.86% | 158 | 51.2% | 68.1% | `$+30.19` | 59.27% | 1.87 | 5.58 | 4.96 | 1.25 |
| **LINK** | 69.38% | 209 | 62.5% | 75.2% | `$+178.42` | 53.11% | 8.46 | 25.95 | 32.72 | 2.45 |
| **NEAR** | 56.61% | 189 | 39.5% | 69.4% | `$+106.24` | 50.68% | 4.69 | 11.77 | 20.42 | 1.73 |
| **ONDO** | 61.27% | 173 | 59.2% | 62.7% | `$+116.16` | 148.12% | 5.27 | 14.85 | 7.64 | 1.90 |
| **POL** | 54.97% | 171 | 50.6% | 59.3% | `$+81.28` | 146.17% | 4.02 | 11.76 | 5.42 | 1.61 |
| **SOL** | 52.61% | 230 | 46.0% | 56.6% | `$+63.21` | 147.03% | 2.63 | 8.58 | 4.19 | 1.31 |
| **SUI** | 63.27% | 196 | 54.5% | 68.9% | `$+131.43` | 181.01% | 5.50 | 14.05 | 7.07 | 1.93 |
| **TAO** | 55.00% | 180 | 51.3% | 62.3% | `$+68.01` | 203.82% | 2.69 | 6.06 | 3.25 | 1.39 |
| **TON** | 56.76% | 185 | 45.1% | 64.0% | `$+62.20` | 115.19% | 3.44 | 9.23 | 5.26 | 1.47 |
| **TRX** | 65.62% | 96 | 62.5% | 71.9% | `$+16.63` | 26.16% | 3.21 | 9.12 | 6.19 | 1.63 |
| **XRP** | 64.33% | 157 | 65.6% | 63.4% | `$+83.34` | 37.01% | 5.65 | 13.30 | 21.93 | 2.02 |

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
