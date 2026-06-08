# 📊 Holdout Backtest Report: `cascade_v2.5_hybrid_pruned_scenario_J`

**Tanggal Pembuatan**: 2026-06-01 17:02:57 UTC
**Model Run ID**: `cascade_v2.5_hybrid_pruned_scenario_J`
**Periode Pengujian (Temporal OOS)**: `2025-11-01 - 2026-04-01`

> [!NOTE]
> **Ringkasan Portofolio Eksekutif**:
> *   **Total Net Profit**: **$+748.12 USD** (ROI Portofolio: **+35.62%**)
> *   **Rata-rata Win Rate**: **56.21%** | Total Trades: **5,293**
> *   **Rata-rata Max Drawdown (5x)**: **157.03%**
> *   **Risk-Adjusted**: Sharpe: **1.68** | Sortino: **4.51** | Calmar: **4.34** | Profit Factor: **1.26**

## 📈 Performa Scorecard Portofolio

| Metrik Utama | Nilai Portofolio | Catatan |
|:---|:---:|:---|
| **Total Net Profit ($)** | `$+748.12` | Akumulasi keuntungan bersih 5x leverage |
| **Portfolio ROI (%)** | `+35.62%` | ROI berdasarkan kapital portofolio $100/koin |
| **Overall Win Rate** | `56.21%` | Rasio kemenangan rata-rata seluruh aset |
| **Total Trades** | `5,293` | Jumlah total posisi yang dieksekusi |
| **Rata-rata Trade / Bulan** | `51.1` | Rata-rata frekuensi trade bulanan portofolio |
| **Rata-rata Trade / Hari** | `1.68` | Rata-rata frekuensi trade harian portofolio |
| **Max Drawdown (5x)** | `157.03%` | Rata-rata penurunan terdalam portofolio |
| **Sharpe Ratio** | `1.68` | Efisiensi profit terhadap volatilitas portofolio |
| **Sortino Ratio** | `4.51` | Efisiensi profit terhadap downside deviation |
| **Calmar Ratio** | `4.34` | Rasio return tahunan terhadap drawdown |
| **Profit Factor** | `1.26` | Rasio gross profit dibagi gross loss |
| **Max Consecutive Loss** | `18` trades | Streak kekalahan beruntun terpanjang |
| **Worst Single Trade PnL** | `-28.20%` | Kerugian terdalam dalam satu trade tunggal |
| **95% Trades Loss Under** | `11.30%` | Nilai risiko (VaR P95) kerugian maksimal |

## ↕️ Analisis Arah Signal (LONG vs SHORT)

| Arah Posisi | Jumlah Trade | Distribusi | Menang | Kalah | Win Rate | PnL Bersih ($) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **LONG** | 2,935 | 55.5% | 1,490 | 1,445 | 50.77% | -149.85 |
| **SHORT** | 2,358 | 44.5% | 1,434 | 924 | 60.81% | +897.97 |

### Rincian Rata-rata Profitabilitas per Trade

| Tipe Trade | PnL Rata-rata ($) | PnL Rata-rata (%) |
|:---|:---:|:---:|
| **Trade Menang (Wins)** | `$+1.7123` | `+6.85%` |
| **Trade Kalah (Losses)** | `$-1.7976` | `-7.19%` |

## 📅 Scorecard Bulanan Portofolio

| Bulan | Total Trades | Wins | Losses | Win Rate | Net PnL ($) |
|:---|:---:|:---:|:---:|:---:|:---:|
| 2025-11 | 949 | 572 | 377 | 60.27% | $+254.49 |
| 2025-12 | 938 | 556 | 382 | 59.28% | $+365.41 |
| 2026-01 | 1104 | 589 | 515 | 53.35% | $+59.88 |
| 2026-02 | 1086 | 554 | 532 | 51.01% | $-138.55 |
| 2026-03 | 1216 | 653 | 563 | 53.70% | $+206.90 |

## 🚪 Distribusi Alasan Keluar Posisi (Exit Reasons)

| Alasan Exit | Jumlah | Persentase | Wins | Losses | Win Rate | PnL Bersih ($) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| `guardian_exit` | 3,773 | 71.3% | 2,803 | 970 | 74.29% | $+3,397.08 |
| `sl_hit` | 1,361 | 25.7% | 4 | 1,357 | 0.29% | $-2,694.75 |
| `time_exit` | 159 | 3.0% | 117 | 42 | 73.58% | $+45.78 |

## 🪙 Scorecard Per Koin (Detailed Assets)

| Token | Win Rate | Trades | LONG WR | SHORT WR | Net PnL ($) | Max DD | Sharpe | Sortino | Calmar | PF |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1000PEPE** | 63.56% | 118 | 51.0% | 73.1% | `$+38.21` | 148.84% | 1.88 | 3.66 | 2.50 | 1.30 |
| **1000SHIB** | 62.50% | 104 | 58.5% | 66.7% | `$+52.08` | 48.37% | 3.52 | 8.16 | 10.49 | 1.74 |
| **ADA** | 62.50% | 200 | 59.4% | 66.0% | `$+123.28` | 53.91% | 5.85 | 19.58 | 22.27 | 1.90 |
| **ARB** | 55.80% | 276 | 51.5% | 62.4% | `$+21.25` | 196.35% | 0.90 | 1.88 | 1.05 | 1.09 |
| **AVAX** | 52.63% | 342 | 48.2% | 57.0% | `$+67.09` | 128.89% | 2.76 | 7.39 | 5.07 | 1.26 |
| **BNB** | 50.00% | 244 | 43.6% | 58.7% | `$-20.90` | 164.71% | -1.34 | -3.72 | -1.24 | 0.88 |
| **BTC** | 58.33% | 36 | 58.8% | 50.0% | `$+5.10` | 63.54% | 0.80 | 1.73 | 0.78 | 1.23 |
| **DOGE** | 63.92% | 97 | 60.0% | 67.3% | `$+79.62` | 31.89% | 4.91 | 13.75 | 24.32 | 2.25 |
| **DOT** | 57.94% | 378 | 50.0% | 67.7% | `$+37.36` | 118.40% | 1.38 | 3.41 | 3.07 | 1.11 |
| **ETH** | 57.14% | 140 | 56.4% | 57.6% | `$+44.72` | 107.78% | 2.57 | 7.52 | 4.04 | 1.44 |
| **HBAR** | 52.06% | 194 | 46.6% | 60.3% | `$-10.55` | 146.30% | -0.63 | -1.62 | -0.70 | 0.94 |
| **LINK** | 55.50% | 400 | 54.9% | 56.2% | `$+100.06` | 164.33% | 3.34 | 10.06 | 5.93 | 1.30 |
| **NEAR** | 58.13% | 375 | 50.2% | 67.0% | `$+122.40` | 297.74% | 3.95 | 10.33 | 4.00 | 1.41 |
| **ONDO** | 53.57% | 308 | 53.0% | 54.3% | `$+28.64` | 248.69% | 1.08 | 2.86 | 1.12 | 1.10 |
| **POL** | 54.68% | 278 | 56.7% | 51.0% | `$+16.50` | 198.38% | 0.72 | 1.57 | 0.81 | 1.07 |
| **SOL** | 53.78% | 331 | 51.5% | 56.2% | `$+53.39` | 196.07% | 2.09 | 5.37 | 2.65 | 1.20 |
| **SUI** | 53.31% | 347 | 47.2% | 61.3% | `$+41.47` | 190.29% | 1.51 | 3.28 | 2.12 | 1.14 |
| **TAO** | 49.64% | 276 | 43.0% | 63.3% | `$-52.15` | 310.71% | -1.80 | -3.84 | -1.63 | 0.84 |
| **TON** | 50.00% | 402 | 46.2% | 53.7% | `$-56.26` | 312.33% | -2.57 | -6.26 | -1.75 | 0.83 |
| **TRX** | 57.01% | 221 | 51.2% | 65.2% | `$+8.39` | 78.41% | 1.21 | 3.45 | 1.04 | 1.13 |
| **XRP** | 58.41% | 226 | 51.5% | 67.7% | `$+48.44` | 91.69% | 3.07 | 6.09 | 5.15 | 1.39 |

## ⛓️ Daftar Fitur Aktif dalam Model

Total terdapat **87 fitur aktif** yang digunakan oleh LightGBM entry, LSTM Soft Confirmation, dan Exit Guardian v3:

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

</details>
