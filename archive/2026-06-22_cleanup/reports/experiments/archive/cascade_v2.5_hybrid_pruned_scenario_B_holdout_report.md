# 📊 Holdout Backtest Report: `cascade_v2.5_hybrid_pruned_scenario_B`

**Tanggal Pembuatan**: 2026-06-01 14:25:25 UTC
**Model Run ID**: `cascade_v2.5_hybrid_pruned_scenario_B`
**Periode Pengujian (Temporal OOS)**: `2025-11-01 - 2026-04-01`

> [!NOTE]
> **Ringkasan Portofolio Eksekutif**:
> *   **Total Net Profit**: **$+670.78 USD** (ROI Portofolio: **+31.94%**)
> *   **Rata-rata Win Rate**: **55.02%** | Total Trades: **7,072**
> *   **Rata-rata Max Drawdown (5x)**: **198.97%**
> *   **Risk-Adjusted**: Sharpe: **1.17** | Sortino: **2.97** | Calmar: **2.51** | Profit Factor: **1.14**

## 📈 Performa Scorecard Portofolio

| Metrik Utama | Nilai Portofolio | Catatan |
|:---|:---:|:---|
| **Total Net Profit ($)** | `$+670.78` | Akumulasi keuntungan bersih 5x leverage |
| **Portfolio ROI (%)** | `+31.94%` | ROI berdasarkan kapital portofolio $100/koin |
| **Overall Win Rate** | `55.02%` | Rasio kemenangan rata-rata seluruh aset |
| **Total Trades** | `7,072` | Jumlah total posisi yang dieksekusi |
| **Rata-rata Trade / Bulan** | `68.3` | Rata-rata frekuensi trade bulanan portofolio |
| **Rata-rata Trade / Hari** | `2.25` | Rata-rata frekuensi trade harian portofolio |
| **Max Drawdown (5x)** | `198.97%` | Rata-rata penurunan terdalam portofolio |
| **Sharpe Ratio** | `1.17` | Efisiensi profit terhadap volatilitas portofolio |
| **Sortino Ratio** | `2.97` | Efisiensi profit terhadap downside deviation |
| **Calmar Ratio** | `2.51` | Rasio return tahunan terhadap drawdown |
| **Profit Factor** | `1.14` | Rasio gross profit dibagi gross loss |
| **Max Consecutive Loss** | `15` trades | Streak kekalahan beruntun terpanjang |
| **Worst Single Trade PnL** | `-29.80%` | Kerugian terdalam dalam satu trade tunggal |
| **95% Trades Loss Under** | `11.60%` | Nilai risiko (VaR P95) kerugian maksimal |

## ↕️ Analisis Arah Signal (LONG vs SHORT)

| Arah Posisi | Jumlah Trade | Distribusi | Menang | Kalah | Win Rate | PnL Bersih ($) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **LONG** | 4,296 | 60.7% | 2,183 | 2,113 | 50.81% | -350.56 |
| **SHORT** | 2,776 | 39.3% | 1,693 | 1,083 | 60.99% | +1,021.34 |

### Rincian Rata-rata Profitabilitas per Trade

| Tipe Trade | PnL Rata-rata ($) | PnL Rata-rata (%) |
|:---|:---:|:---:|
| **Trade Menang (Wins)** | `$+1.6856` | `+6.74%` |
| **Trade Kalah (Losses)** | `$-1.8344` | `-7.34%` |

## 📅 Scorecard Bulanan Portofolio

| Bulan | Total Trades | Wins | Losses | Win Rate | Net PnL ($) |
|:---|:---:|:---:|:---:|:---:|:---:|
| 2025-11 | 1317 | 725 | 592 | 55.05% | $+48.70 |
| 2025-12 | 1306 | 792 | 514 | 60.64% | $+479.40 |
| 2026-01 | 1440 | 780 | 660 | 54.17% | $+41.97 |
| 2026-02 | 1413 | 712 | 701 | 50.39% | $-204.14 |
| 2026-03 | 1596 | 867 | 729 | 54.32% | $+304.86 |

## 🚪 Distribusi Alasan Keluar Posisi (Exit Reasons)

| Alasan Exit | Jumlah | Persentase | Wins | Losses | Win Rate | PnL Bersih ($) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| `guardian_exit` | 5,135 | 72.6% | 3,710 | 1,425 | 72.25% | $+4,054.86 |
| `sl_hit` | 1,715 | 24.3% | 5 | 1,710 | 0.29% | $-3,447.26 |
| `time_exit` | 222 | 3.1% | 161 | 61 | 72.52% | $+63.18 |

## 🪙 Scorecard Per Koin (Detailed Assets)

| Token | Win Rate | Trades | LONG WR | SHORT WR | Net PnL ($) | Max DD | Sharpe | Sortino | Calmar | PF |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1000PEPE** | 61.08% | 167 | 50.5% | 76.5% | `$+47.92` | 171.92% | 2.03 | 3.86 | 2.71 | 1.27 |
| **1000SHIB** | 58.23% | 158 | 52.5% | 67.8% | `$+48.25` | 69.67% | 2.67 | 6.45 | 6.75 | 1.40 |
| **ADA** | 60.87% | 299 | 60.1% | 62.2% | `$+156.79` | 83.82% | 5.93 | 17.47 | 18.22 | 1.70 |
| **ARB** | 55.28% | 360 | 51.9% | 62.0% | `$-16.59` | 279.48% | -0.61 | -1.21 | -0.58 | 0.95 |
| **AVAX** | 52.85% | 456 | 48.8% | 57.7% | `$+44.17` | 147.10% | 1.61 | 4.11 | 2.92 | 1.12 |
| **BNB** | 52.41% | 311 | 46.2% | 61.6% | `$-10.60` | 151.23% | -0.60 | -1.63 | -0.68 | 0.95 |
| **BTC** | 46.34% | 123 | 46.7% | 33.3% | `$-27.53` | 136.84% | -2.48 | -5.84 | -1.96 | 0.71 |
| **DOGE** | 61.96% | 184 | 57.1% | 70.8% | `$+99.01` | 118.95% | 4.67 | 11.24 | 8.11 | 1.70 |
| **DOT** | 55.94% | 463 | 49.8% | 65.2% | `$+35.50` | 144.99% | 1.14 | 2.85 | 2.38 | 1.09 |
| **ETH** | 55.56% | 261 | 48.9% | 63.3% | `$+34.01` | 172.69% | 1.53 | 3.58 | 1.92 | 1.17 |
| **HBAR** | 52.07% | 290 | 47.1% | 61.4% | `$-33.00` | 222.38% | -1.59 | -3.65 | -1.45 | 0.87 |
| **LINK** | 53.35% | 493 | 51.4% | 56.0% | `$+64.48` | 208.60% | 1.96 | 5.90 | 3.01 | 1.15 |
| **NEAR** | 56.80% | 456 | 49.6% | 66.3% | `$+106.95` | 393.43% | 3.18 | 7.60 | 2.65 | 1.28 |
| **ONDO** | 53.56% | 379 | 52.6% | 55.2% | `$+37.78` | 275.45% | 1.25 | 3.38 | 1.34 | 1.10 |
| **POL** | 55.70% | 386 | 58.0% | 51.1% | `$+16.58` | 210.20% | 0.63 | 1.33 | 0.77 | 1.05 |
| **SOL** | 53.11% | 418 | 51.3% | 55.1% | `$+43.48` | 207.58% | 1.52 | 3.76 | 2.04 | 1.12 |
| **SUI** | 54.40% | 432 | 48.4% | 62.6% | `$+49.00` | 297.99% | 1.61 | 3.51 | 1.60 | 1.13 |
| **TAO** | 51.83% | 355 | 47.2% | 62.6% | `$-49.73` | 360.18% | -1.50 | -3.18 | -1.34 | 0.88 |
| **TON** | 51.04% | 482 | 49.0% | 53.4% | `$-37.76` | 307.11% | -1.48 | -3.86 | -1.20 | 0.90 |
| **TRX** | 54.51% | 288 | 48.1% | 65.4% | `$-1.03` | 107.19% | -0.12 | -0.32 | -0.09 | 0.99 |
| **XRP** | 58.52% | 311 | 52.6% | 68.1% | `$+63.11` | 111.53% | 3.28 | 6.91 | 5.51 | 1.34 |

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
