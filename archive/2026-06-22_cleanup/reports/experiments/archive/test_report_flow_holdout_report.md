# 📊 Holdout Backtest Report: `test_report_flow`

**Tanggal Pembuatan**: 2026-05-27 23:09:23 UTC
**Model Run ID**: `test_report_flow`
**Periode Pengujian (Temporal OOS)**: `2025-11-01 - 2026-04-01`

> [!NOTE]
> **Ringkasan Portofolio Eksekutif**:
> *   **Total Net Profit**: **$+99.20 USD** (ROI Portofolio: **+99.20%**)
> *   **Rata-rata Win Rate**: **78.67%** | Total Trades: **75**
> *   **Rata-rata Max Drawdown (5x)**: **23.67%**
> *   **Risk-Adjusted**: Sharpe: **5.77** | Sortino: **14.61** | Calmar: **18.33** | Profit Factor: **5.23**

## 📈 Performa Scorecard Portofolio

| Metrik Utama | Nilai Portofolio | Catatan |
|:---|:---:|:---|
| **Total Net Profit ($)** | `$+99.20` | Akumulasi keuntungan bersih 5x leverage |
| **Portfolio ROI (%)** | `+99.20%` | ROI berdasarkan kapital portofolio $100/koin |
| **Overall Win Rate** | `78.67%` | Rasio kemenangan rata-rata seluruh aset |
| **Total Trades** | `75` | Jumlah total posisi yang dieksekusi |
| **Rata-rata Trade / Bulan** | `6.8` | Rata-rata frekuensi trade bulanan portofolio |
| **Rata-rata Trade / Hari** | `0.22` | Rata-rata frekuensi trade harian portofolio |
| **Max Drawdown (5x)** | `23.67%` | Rata-rata penurunan terdalam portofolio |
| **Sharpe Ratio** | `5.77` | Efisiensi profit terhadap volatilitas portofolio |
| **Sortino Ratio** | `14.61` | Efisiensi profit terhadap downside deviation |
| **Calmar Ratio** | `18.33` | Rasio return tahunan terhadap drawdown |
| **Profit Factor** | `5.23` | Rasio gross profit dibagi gross loss |
| **Max Consecutive Loss** | `3` trades | Streak kekalahan beruntun terpanjang |
| **Worst Single Trade PnL** | `-11.60%` | Kerugian terdalam dalam satu trade tunggal |
| **95% Trades Loss Under** | `8.11%` | Nilai risiko (VaR P95) kerugian maksimal |

## ↕️ Analisis Arah Signal (LONG vs SHORT)

| Arah Posisi | Jumlah Trade | Distribusi | Menang | Kalah | Win Rate | PnL Bersih ($) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **LONG** | 42 | 56.0% | 30 | 12 | 71.43% | +34.54 |
| **SHORT** | 33 | 44.0% | 29 | 4 | 87.88% | +64.66 |

### Rincian Rata-rata Profitabilitas per Trade

| Tipe Trade | PnL Rata-rata ($) | PnL Rata-rata (%) |
|:---|:---:|:---:|
| **Trade Menang (Wins)** | `$+2.0789` | `+8.32%` |
| **Trade Kalah (Losses)** | `$-1.4659` | `-5.87%` |

## 📅 Scorecard Bulanan Portofolio

| Bulan | Total Trades | Wins | Losses | Win Rate | Net PnL ($) |
|:---|:---:|:---:|:---:|:---:|:---:|
| 2025-05 | 14 | 14 | 0 | 100.00% | $+29.07 |
| 2025-06 | 12 | 9 | 3 | 75.00% | $+15.40 |
| 2025-07 | 8 | 8 | 0 | 100.00% | $+14.47 |
| 2025-08 | 18 | 14 | 4 | 77.78% | $+26.86 |
| 2025-09 | 1 | 0 | 1 | 0.00% | $-0.07 |
| 2025-10 | 4 | 4 | 0 | 100.00% | $+4.00 |
| 2025-11 | 5 | 5 | 0 | 100.00% | $+7.17 |
| 2025-12 | 3 | 1 | 2 | 33.33% | $-1.58 |
| 2026-01 | 4 | 1 | 3 | 25.00% | $-1.35 |
| 2026-02 | 4 | 2 | 2 | 50.00% | $-0.69 |
| 2026-03 | 2 | 1 | 1 | 50.00% | $+5.94 |

## 🚪 Distribusi Alasan Keluar Posisi (Exit Reasons)

| Alasan Exit | Jumlah | Persentase | Wins | Losses | Win Rate | PnL Bersih ($) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| `guardian_exit` | 66 | 88.0% | 59 | 7 | 89.39% | $+111.82 |
| `sl_hit` | 9 | 12.0% | 0 | 9 | 0.00% | $-12.62 |

## 🪙 Scorecard Per Koin (Detailed Assets)

| Token | Win Rate | Trades | LONG WR | SHORT WR | Net PnL ($) | Max DD | Sharpe | Sortino | Calmar | PF |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **SOL** | 78.67% | 75 | 71.4% | 87.9% | `$+99.20` | 23.67% | 5.77 | 14.61 | 18.33 | 5.23 |

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
