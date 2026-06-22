# 📊 Holdout Backtest Report: `simons_hybrid_v1`

**Tanggal Pembuatan**: 2026-06-08 20:03:35 UTC
**Model Run ID**: `simons_hybrid_v1`
**Periode Pengujian (Temporal OOS)**: `2025-11-01 - 2026-04-01`

> [!NOTE]
> **Ringkasan Portofolio Eksekutif**:
> *   **Total Net Profit**: **$+369.27 USD** (ROI Portofolio: **+17.58%**)
> *   **Rata-rata Win Rate**: **56.45%** | Total Trades: **2,216**
> *   **Rata-rata Max Drawdown (5x)**: **96.14%**
> *   **Risk-Adjusted**: Sharpe: **2.58** | Sortino: **8.09** | Calmar: **5.93** | Profit Factor: **1.64**

## 📈 Performa Scorecard Portofolio

| Metrik Utama | Nilai Portofolio | Catatan |
|:---|:---:|:---|
| **Total Net Profit ($)** | `$+369.27` | Akumulasi keuntungan bersih 5x leverage |
| **Portfolio ROI (%)** | `+17.58%` | ROI berdasarkan kapital portofolio $100/koin |
| **Overall Win Rate** | `56.45%` | Rasio kemenangan rata-rata seluruh aset |
| **Total Trades** | `2,216` | Jumlah total posisi yang dieksekusi |
| **Rata-rata Trade / Bulan** | `21.4` | Rata-rata frekuensi trade bulanan portofolio |
| **Rata-rata Trade / Hari** | `0.70` | Rata-rata frekuensi trade harian portofolio |
| **Max Drawdown (5x)** | `96.14%` | Rata-rata penurunan terdalam portofolio |
| **Sharpe Ratio** | `2.58` | Efisiensi profit terhadap volatilitas portofolio |
| **Sortino Ratio** | `8.09` | Efisiensi profit terhadap downside deviation |
| **Calmar Ratio** | `5.93` | Rasio return tahunan terhadap drawdown |
| **Profit Factor** | `1.64` | Rasio gross profit dibagi gross loss |
| **Max Consecutive Loss** | `12` trades | Streak kekalahan beruntun terpanjang |
| **Worst Single Trade PnL** | `-24.50%` | Kerugian terdalam dalam satu trade tunggal |
| **95% Trades Loss Under** | `11.70%` | Nilai risiko (VaR P95) kerugian maksimal |

## ↕️ Analisis Arah Signal (LONG vs SHORT)

| Arah Posisi | Jumlah Trade | Distribusi | Menang | Kalah | Win Rate | PnL Bersih ($) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **LONG** | 177 | 8.0% | 84 | 93 | 47.46% | -5.78 |
| **SHORT** | 2,039 | 92.0% | 1,162 | 877 | 56.99% | +375.05 |

### Rincian Rata-rata Profitabilitas per Trade

| Tipe Trade | PnL Rata-rata ($) | PnL Rata-rata (%) |
|:---|:---:|:---:|
| **Trade Menang (Wins)** | `$+0.8717` | `+8.72%` |
| **Trade Kalah (Losses)** | `$-0.7391` | `-7.39%` |

## 📅 Scorecard Bulanan Portofolio

| Bulan | Total Trades | Wins | Losses | Win Rate | Net PnL ($) |
|:---|:---:|:---:|:---:|:---:|:---:|
| 2025-11 | 653 | 328 | 325 | 50.23% | $+37.26 |
| 2025-12 | 365 | 174 | 191 | 47.67% | $+36.26 |
| 2026-01 | 624 | 396 | 228 | 63.46% | $+174.24 |
| 2026-02 | 317 | 210 | 107 | 66.25% | $+120.67 |
| 2026-03 | 257 | 138 | 119 | 53.70% | $+0.83 |

## 🚪 Distribusi Alasan Keluar Posisi (Exit Reasons)

| Alasan Exit | Jumlah | Persentase | Wins | Losses | Win Rate | PnL Bersih ($) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| `guardian_exit` | 1,786 | 80.6% | 1,209 | 577 | 67.69% | $+668.24 |
| `sl_hit` | 393 | 17.7% | 0 | 393 | 0.00% | $-306.54 |
| `time_exit` | 37 | 1.7% | 37 | 0 | 100.00% | $+7.57 |

## 🪙 Scorecard Per Koin (Detailed Assets)

| Token | Win Rate | Trades | LONG WR | SHORT WR | Net PnL ($) | Max DD | Sharpe | Sortino | Calmar | PF |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1000PEPE** | 53.01% | 83 | 27.3% | 56.9% | `$+13.19` | 189.39% | 1.61 | 5.57 | 1.70 | 1.33 |
| **1000SHIB** | 57.14% | 168 | 26.7% | 60.1% | `$+15.16` | 113.98% | 2.16 | 4.97 | 3.24 | 1.30 |
| **ADA** | 72.15% | 79 | 53.8% | 75.8% | `$+37.16` | 44.32% | 6.34 | 22.72 | 20.41 | 3.21 |
| **ARB** | 53.64% | 110 | 43.8% | 55.3% | `$+15.28` | 93.20% | 2.15 | 7.37 | 3.99 | 1.38 |
| **AVAX** | 55.12% | 127 | 60.0% | 54.7% | `$+13.49` | 76.94% | 2.05 | 4.85 | 4.27 | 1.33 |
| **BNB** | 55.61% | 187 | 57.1% | 55.5% | `$+22.01` | 126.39% | 3.84 | 12.53 | 4.24 | 1.54 |
| **BTC** | 50.00% | 112 | 50.0% | 50.0% | `$+7.81` | 89.13% | 1.54 | 5.60 | 2.13 | 1.23 |
| **DOGE** | 47.13% | 157 | 72.7% | 45.2% | `$+6.75` | 126.12% | 0.83 | 2.20 | 1.30 | 1.11 |
| **DOT** | 58.82% | 153 | 33.3% | 61.6% | `$+32.99` | 108.32% | 4.00 | 9.80 | 7.42 | 1.65 |
| **ETH** | 72.64% | 106 | 100.0% | 71.6% | `$+48.48` | 50.01% | 7.87 | 30.47 | 23.60 | 3.76 |
| **HBAR** | 56.16% | 146 | 62.5% | 55.4% | `$+9.22` | 107.97% | 1.33 | 3.79 | 2.08 | 1.18 |
| **LINK** | 56.04% | 91 | 33.3% | 56.8% | `$+13.21` | 88.62% | 2.12 | 5.83 | 3.63 | 1.41 |
| **NEAR** | 50.00% | 90 | 50.0% | 50.0% | `$+7.76` | 130.66% | 1.03 | 2.57 | 1.45 | 1.21 |
| **ONDO** | 62.90% | 62 | 100.0% | 61.7% | `$+20.44` | 48.28% | 3.73 | 12.52 | 10.31 | 2.08 |
| **POL** | 60.00% | 40 | 100.0% | 59.0% | `$-0.76` | 55.22% | -0.17 | -0.35 | -0.34 | 0.96 |
| **SOL** | 56.57% | 99 | 66.7% | 55.9% | `$+30.47` | 122.93% | 4.07 | 12.35 | 6.04 | 1.94 |
| **SUI** | 62.03% | 79 | 0.0% | 65.3% | `$+29.10` | 59.02% | 4.63 | 13.54 | 12.01 | 2.26 |
| **TAO** | 63.22% | 87 | 75.0% | 62.6% | `$+36.35` | 95.10% | 4.93 | 11.69 | 9.31 | 2.29 |
| **TON** | 38.24% | 68 | 22.2% | 40.7% | `$-6.60` | 136.06% | -1.18 | -4.23 | -1.18 | 0.81 |
| **TRX** | 47.14% | 70 | 37.5% | 48.4% | `$-6.44` | 100.61% | -2.62 | -5.24 | -1.56 | 0.63 |
| **XRP** | 57.84% | 102 | 20.0% | 59.8% | `$+24.20` | 56.57% | 3.87 | 11.26 | 10.42 | 1.79 |

## ⛓️ Daftar Fitur Aktif dalam Model

Total terdapat **40 fitur aktif** yang digunakan oleh LightGBM entry, LSTM Soft Confirmation, dan Exit Guardian v3:

<details>
<summary>▶ Klik untuk melihat daftar lengkap fitur aktif</summary>

1. `dist_from_8h_high`
2. `cvd_slope_h4`
3. `ofi_h4_delta`
4. `ultra_high_vol`
5. `POC`
6. `absorption_at_swing`
7. `swing_momentum`
8. `ema_50_h1`
9. `ema_50_slope_h4`
10. `log_ret_20`
11. `vol_price_confirm`
12. `vol_ratio_20`
13. `ema_21_slope_h4`
14. `dist_liq_50x_long`
15. `PDL`
16. `dist_liq_50x_short`
17. `log_ret_1`
18. `dist_swing_low`
19. `stochrsi_d`
20. `etf_gbtc_change_usd`
21. `etf_total_change_usd`
22. `dist_liq_20x_long`
23. `dist_liq_20x_short`
24. `ema_7_h1`
25. `PDH`
26. `log_ret_5`
27. `vol_accel_3h`
28. `rsi_6`
29. `dow_cos`
30. `long_short_ratio`
31. `ofi_raw`
32. `absorption_z`
33. `atr_percentile_h1`
34. `ema_200_h1`
35. `vol_efficiency`
36. `dist_swing_high`
37. `ema_21_h1`
38. `price_in_range`
39. `Buy_Liq`
40. `ofi_z_score`

</details>
