# 📊 Holdout Backtest Report: `cascade_v2.5_hybrid_accel_scenario_W5`

**Tanggal Pembuatan**: 2026-06-03 21:20:15 UTC
**Model Run ID**: `cascade_v2.5_hybrid_accel_scenario_W5`
**Periode Pengujian (Temporal OOS)**: `2025-11-01 - 2026-04-01`

> [!NOTE]
> **Ringkasan Portofolio Eksekutif**:
> *   **Total Net Profit**: **$-953.42 USD** (ROI Portofolio: **-45.40%**)
> *   **Rata-rata Win Rate**: **48.74%** | Total Trades: **12,075**
> *   **Rata-rata Max Drawdown (5x)**: **397.70%**
> *   **Risk-Adjusted**: Sharpe: **-0.95** | Sortino: **-2.26** | Calmar: **0.54** | Profit Factor: **1.03**

## 📈 Performa Scorecard Portofolio

| Metrik Utama | Nilai Portofolio | Catatan |
|:---|:---:|:---|
| **Total Net Profit ($)** | `$-953.42` | Akumulasi keuntungan bersih 5x leverage |
| **Portfolio ROI (%)** | `-45.40%` | ROI berdasarkan kapital portofolio $100/koin |
| **Overall Win Rate** | `48.74%` | Rasio kemenangan rata-rata seluruh aset |
| **Total Trades** | `12,075` | Jumlah total posisi yang dieksekusi |
| **Rata-rata Trade / Bulan** | `116.7` | Rata-rata frekuensi trade bulanan portofolio |
| **Rata-rata Trade / Hari** | `3.83` | Rata-rata frekuensi trade harian portofolio |
| **Max Drawdown (5x)** | `397.70%` | Rata-rata penurunan terdalam portofolio |
| **Sharpe Ratio** | `-0.95` | Efisiensi profit terhadap volatilitas portofolio |
| **Sortino Ratio** | `-2.26` | Efisiensi profit terhadap downside deviation |
| **Calmar Ratio** | `0.54` | Rasio return tahunan terhadap drawdown |
| **Profit Factor** | `1.03` | Rasio gross profit dibagi gross loss |
| **Max Consecutive Loss** | `24` trades | Streak kekalahan beruntun terpanjang |
| **Worst Single Trade PnL** | `-28.80%` | Kerugian terdalam dalam satu trade tunggal |
| **95% Trades Loss Under** | `11.40%` | Nilai risiko (VaR P95) kerugian maksimal |

## ↕️ Analisis Arah Signal (LONG vs SHORT)

| Arah Posisi | Jumlah Trade | Distribusi | Menang | Kalah | Win Rate | PnL Bersih ($) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **LONG** | 5,483 | 45.4% | 2,267 | 3,216 | 41.35% | -1,961.90 |
| **SHORT** | 6,592 | 54.6% | 3,626 | 2,966 | 55.01% | +1,008.48 |

### Rincian Rata-rata Profitabilitas per Trade

| Tipe Trade | PnL Rata-rata ($) | PnL Rata-rata (%) |
|:---|:---:|:---:|
| **Trade Menang (Wins)** | `$+1.7100` | `+6.84%` |
| **Trade Kalah (Losses)** | `$-1.7843` | `-7.14%` |

## 📅 Scorecard Bulanan Portofolio

| Bulan | Total Trades | Wins | Losses | Win Rate | Net PnL ($) |
|:---|:---:|:---:|:---:|:---:|:---:|
| 2025-11 | 2065 | 1031 | 1034 | 49.93% | $-200.49 |
| 2025-12 | 2501 | 1217 | 1284 | 48.66% | $+86.16 |
| 2026-01 | 2550 | 1315 | 1235 | 51.57% | $+100.34 |
| 2026-02 | 2297 | 1033 | 1264 | 44.97% | $-656.94 |
| 2026-03 | 2662 | 1297 | 1365 | 48.72% | $-282.49 |

## 🚪 Distribusi Alasan Keluar Posisi (Exit Reasons)

| Alasan Exit | Jumlah | Persentase | Wins | Losses | Win Rate | PnL Bersih ($) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| `guardian_exit` | 7,989 | 66.2% | 5,707 | 2,282 | 71.44% | $+6,354.70 |
| `sl_hit` | 3,874 | 32.1% | 4 | 3,870 | 0.10% | $-7,385.84 |
| `time_exit` | 212 | 1.8% | 182 | 30 | 85.85% | $+77.72 |

## 🪙 Scorecard Per Koin (Detailed Assets)

| Token | Win Rate | Trades | LONG WR | SHORT WR | Net PnL ($) | Max DD | Sharpe | Sortino | Calmar | PF |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1000PEPE** | 61.11% | 18 | 0.0% | 61.1% | `$+12.93` | 14.42% | 1.79 | 6.51 | 8.73 | 2.18 |
| **1000SHIB** | 71.05% | 38 | 37.5% | 80.0% | `$+23.76` | 17.45% | 3.44 | 8.99 | 13.26 | 2.29 |
| **ADA** | 47.81% | 638 | 35.1% | 59.2% | `$-49.69` | 503.34% | -1.39 | -4.04 | -0.96 | 0.92 |
| **ARB** | 48.13% | 667 | 42.9% | 52.7% | `$-30.31` | 280.72% | -0.79 | -2.17 | -1.05 | 0.95 |
| **AVAX** | 47.90% | 737 | 40.5% | 53.3% | `$-40.18` | 411.44% | -1.14 | -3.05 | -0.95 | 0.94 |
| **BNB** | 46.70% | 788 | 42.4% | 50.1% | `$-120.77` | 550.05% | -4.40 | -11.38 | -2.14 | 0.80 |
| **BTC** | 0.00% | 0 | 0.0% | 0.0% | `$+0.00` | 0.00% | 0.00 | 0.00 | 0.00 | 0.00 |
| **DOGE** | 59.72% | 72 | 38.5% | 64.4% | `$+13.12` | 72.58% | 1.47 | 3.44 | 1.76 | 1.30 |
| **DOT** | 45.91% | 771 | 34.7% | 56.0% | `$-187.06` | 887.21% | -4.66 | -11.18 | -2.05 | 0.78 |
| **ETH** | 51.54% | 648 | 44.2% | 55.0% | `$+30.35` | 371.74% | 0.92 | 2.34 | 0.80 | 1.06 |
| **HBAR** | 47.45% | 607 | 40.0% | 56.0% | `$-79.89` | 373.97% | -2.57 | -6.86 | -2.08 | 0.86 |
| **LINK** | 50.81% | 805 | 46.0% | 54.5% | `$+25.31` | 229.98% | 0.67 | 1.87 | 1.07 | 1.04 |
| **NEAR** | 50.88% | 741 | 39.8% | 61.5% | `$+6.31` | 519.35% | 0.16 | 0.38 | 0.12 | 1.01 |
| **ONDO** | 49.78% | 687 | 42.7% | 55.1% | `$+14.70` | 441.33% | 0.37 | 1.03 | 0.32 | 1.02 |
| **POL** | 48.64% | 664 | 42.6% | 53.5% | `$+25.29` | 342.13% | 0.67 | 1.94 | 0.72 | 1.04 |
| **SOL** | 45.11% | 807 | 37.6% | 50.2% | `$-151.98` | 807.53% | -4.20 | -10.69 | -1.83 | 0.80 |
| **SUI** | 50.60% | 747 | 39.9% | 60.4% | `$-8.23` | 459.13% | -0.20 | -0.53 | -0.17 | 0.99 |
| **TAO** | 46.58% | 803 | 41.9% | 52.4% | `$-240.66` | 962.64% | -5.15 | -11.14 | -2.44 | 0.75 |
| **TON** | 45.24% | 778 | 42.6% | 47.3% | `$-179.23` | 737.62% | -5.44 | -14.74 | -2.37 | 0.75 |
| **TRX** | 58.13% | 406 | 50.9% | 66.1% | `$+16.82` | 90.50% | 1.70 | 4.59 | 1.81 | 1.13 |
| **XRP** | 50.54% | 653 | 44.0% | 56.4% | `$-34.00` | 278.51% | -1.16 | -2.85 | -1.19 | 0.93 |

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
