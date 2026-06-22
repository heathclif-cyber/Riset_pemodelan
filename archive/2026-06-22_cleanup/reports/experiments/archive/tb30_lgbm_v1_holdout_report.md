# 📊 Holdout Backtest Report: `tb30_lgbm_v1`

**Tanggal Pembuatan**: 2026-06-05 17:21:40 UTC
**Model Run ID**: `tb30_lgbm_v1`
**Periode Pengujian (Temporal OOS)**: `2025-11-01 - 2026-04-01`

> [!NOTE]
> **Ringkasan Portofolio Eksekutif**:
> *   **Total Net Profit**: **$+1,835.60 USD** (ROI Portofolio: **+87.41%**)
> *   **Rata-rata Win Rate**: **86.39%** | Total Trades: **980**
> *   **Rata-rata Max Drawdown (5x)**: **14.19%**
> *   **Risk-Adjusted**: Sharpe: **4.70** | Sortino: **21.00** | Calmar: **433.25** | Profit Factor: **18.72**

## 📈 Performa Scorecard Portofolio

| Metrik Utama | Nilai Portofolio | Catatan |
|:---|:---:|:---|
| **Total Net Profit ($)** | `$+1,835.60` | Akumulasi keuntungan bersih 5x leverage |
| **Portfolio ROI (%)** | `+87.41%` | ROI berdasarkan kapital portofolio $100/koin |
| **Overall Win Rate** | `86.39%` | Rasio kemenangan rata-rata seluruh aset |
| **Total Trades** | `980` | Jumlah total posisi yang dieksekusi |
| **Rata-rata Trade / Bulan** | `9.5` | Rata-rata frekuensi trade bulanan portofolio |
| **Rata-rata Trade / Hari** | `0.31` | Rata-rata frekuensi trade harian portofolio |
| **Max Drawdown (5x)** | `14.19%` | Rata-rata penurunan terdalam portofolio |
| **Sharpe Ratio** | `4.70` | Efisiensi profit terhadap volatilitas portofolio |
| **Sortino Ratio** | `21.00` | Efisiensi profit terhadap downside deviation |
| **Calmar Ratio** | `433.25` | Rasio return tahunan terhadap drawdown |
| **Profit Factor** | `18.72` | Rasio gross profit dibagi gross loss |
| **Max Consecutive Loss** | `6` trades | Streak kekalahan beruntun terpanjang |
| **Worst Single Trade PnL** | `-20.40%` | Kerugian terdalam dalam satu trade tunggal |
| **95% Trades Loss Under** | `5.60%` | Nilai risiko (VaR P95) kerugian maksimal |

## ↕️ Analisis Arah Signal (LONG vs SHORT)

| Arah Posisi | Jumlah Trade | Distribusi | Menang | Kalah | Win Rate | PnL Bersih ($) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **LONG** | 24 | 2.4% | 22 | 2 | 91.67% | +42.27 |
| **SHORT** | 956 | 97.6% | 827 | 129 | 86.51% | +1,793.33 |

### Rincian Rata-rata Profitabilitas per Trade

| Tipe Trade | PnL Rata-rata ($) | PnL Rata-rata (%) |
|:---|:---:|:---:|
| **Trade Menang (Wins)** | `$+2.3533` | `+9.41%` |
| **Trade Kalah (Losses)** | `$-1.2393` | `-4.96%` |

## 📅 Scorecard Bulanan Portofolio

| Bulan | Total Trades | Wins | Losses | Win Rate | Net PnL ($) |
|:---|:---:|:---:|:---:|:---:|:---:|
| 2025-11 | 216 | 183 | 33 | 84.72% | $+414.27 |
| 2025-12 | 329 | 295 | 34 | 89.67% | $+740.00 |
| 2026-01 | 185 | 160 | 25 | 86.49% | $+314.02 |
| 2026-02 | 120 | 110 | 10 | 91.67% | $+222.73 |
| 2026-03 | 130 | 101 | 29 | 77.69% | $+144.58 |

## 🚪 Distribusi Alasan Keluar Posisi (Exit Reasons)

| Alasan Exit | Jumlah | Persentase | Wins | Losses | Win Rate | PnL Bersih ($) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| `guardian_exit` | 876 | 89.4% | 835 | 41 | 95.32% | $+1,945.33 |
| `sl_hit` | 94 | 9.6% | 4 | 90 | 4.26% | $-118.09 |
| `time_exit` | 10 | 1.0% | 10 | 0 | 100.00% | $+8.36 |

## 🪙 Scorecard Per Koin (Detailed Assets)

| Token | Win Rate | Trades | LONG WR | SHORT WR | Net PnL ($) | Max DD | Sharpe | Sortino | Calmar | PF |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1000PEPE** | 88.24% | 51 | 0.0% | 88.2% | `$+115.05` | 7.33% | 4.18 | 122.00 | 889.72 | 19.77 |
| **1000SHIB** | 77.27% | 44 | 100.0% | 75.0% | `$+44.52` | 48.56% | 4.17 | 7.53 | 22.74 | 3.64 |
| **ADA** | 95.56% | 45 | 0.0% | 95.6% | `$+133.17` | 8.89% | 4.93 | 0.00 | 990.24 | 50.27 |
| **ARB** | 85.29% | 34 | 0.0% | 85.3% | `$+100.47` | 10.47% | 4.54 | 34.27 | 474.47 | 19.55 |
| **AVAX** | 85.71% | 56 | 100.0% | 85.2% | `$+117.99` | 17.52% | 5.28 | 5.21 | 391.86 | 12.51 |
| **BNB** | 95.24% | 63 | 0.0% | 95.2% | `$+96.61` | 11.32% | 5.89 | 30.17 | 406.08 | 28.29 |
| **BTC** | 90.70% | 43 | 0.0% | 90.7% | `$+60.60` | 5.28% | 4.64 | 13.04 | 359.51 | 18.24 |
| **DOGE** | 84.91% | 53 | 0.0% | 84.9% | `$+110.61` | 12.11% | 5.62 | 0.00 | 497.32 | 27.96 |
| **DOT** | 87.50% | 40 | 100.0% | 86.5% | `$+77.24` | 14.40% | 5.03 | 19.61 | 206.86 | 9.78 |
| **ETH** | 83.67% | 49 | 0.0% | 85.4% | `$+98.37` | 8.33% | 4.38 | 8.30 | 571.88 | 12.74 |
| **HBAR** | 94.59% | 37 | 100.0% | 93.9% | `$+75.79` | 7.74% | 5.86 | 13.12 | 371.26 | 29.79 |
| **LINK** | 88.89% | 54 | 0.0% | 88.9% | `$+129.26` | 12.75% | 5.01 | 18.45 | 649.14 | 17.18 |
| **NEAR** | 88.24% | 34 | 100.0% | 87.9% | `$+94.29` | 9.25% | 4.52 | 86.82 | 473.65 | 22.07 |
| **ONDO** | 86.21% | 29 | 100.0% | 85.2% | `$+72.07` | 7.20% | 3.62 | 9.23 | 363.03 | 33.82 |
| **POL** | 95.83% | 48 | 0.0% | 95.8% | `$+118.04` | 8.35% | 5.44 | 9.68 | 822.87 | 35.01 |
| **SOL** | 83.67% | 49 | 50.0% | 85.1% | `$+78.18` | 8.66% | 4.37 | 16.36 | 351.97 | 10.98 |
| **SUI** | 71.70% | 53 | 0.0% | 71.7% | `$+68.93` | 39.43% | 4.21 | 5.63 | 60.99 | 3.89 |
| **TAO** | 88.57% | 35 | 0.0% | 88.6% | `$+95.04` | 5.78% | 3.56 | 10.59 | 769.87 | 19.09 |
| **TON** | 74.42% | 43 | 100.0% | 73.8% | `$+47.94` | 33.19% | 3.83 | 9.12 | 37.78 | 3.74 |
| **TRX** | 80.82% | 73 | 100.0% | 79.7% | `$+34.05` | 14.58% | 4.67 | 6.74 | 48.68 | 4.31 |
| **XRP** | 87.23% | 47 | 0.0% | 87.2% | `$+67.40` | 6.82% | 4.95 | 15.08 | 338.26 | 10.50 |

## ⛓️ Daftar Fitur Aktif dalam Model

Total terdapat **33 fitur aktif** yang digunakan oleh LightGBM entry, LSTM Soft Confirmation, dan Exit Guardian v3:

<details>
<summary>▶ Klik untuk melihat daftar lengkap fitur aktif</summary>

1. `dist_from_8h_high`
2. `rsi_6`
3. `swing_momentum`
4. `rsi_h4`
5. `stochrsi_k`
6. `dist_liq_50x_long`
7. `trend_accel_4h`
8. `rsi_slope_h4`
9. `Fib_786`
10. `Fib_618`
11. `stochrsi_d`
12. `ofi_h4_delta`
13. `dist_liq_50x_short`
14. `Buy_Liq`
15. `relative_strength_z`
16. `dist_liq_20x_long`
17. `cvd_momentum_adv`
18. `Sell_Liq`
19. `long_short_ratio`
20. `cvd_slope_h4`
21. `ema_21_slope_h4`
22. `ema_50_h1`
23. `h4_trend`
24. `log_ret_20`
25. `whale_retail_divergence`
26. `dist_liq_20x_short`
27. `vol_price_confirm`
28. `ema_50_slope_h4`
29. `MSB_BOS`
30. `cvd`
31. `ofi_acceleration`
32. `cvd_div_h4`
33. `hmm_regime_enc`

</details>
