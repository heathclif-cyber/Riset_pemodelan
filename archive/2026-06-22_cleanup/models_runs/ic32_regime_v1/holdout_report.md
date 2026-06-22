# 📊 Holdout Backtest Report: `ic32_regime_v1`

**Tanggal Pembuatan**: 2026-06-08 20:04:45 UTC
**Model Run ID**: `ic32_regime_v1`
**Periode Pengujian (Temporal OOS)**: `2025-11-01 - 2026-04-01`

> [!NOTE]
> **Ringkasan Portofolio Eksekutif**:
> *   **Total Net Profit**: **$+699.87 USD** (ROI Portofolio: **+33.33%**)
> *   **Rata-rata Win Rate**: **63.38%** | Total Trades: **2,415**
> *   **Rata-rata Max Drawdown (5x)**: **56.59%**
> *   **Risk-Adjusted**: Sharpe: **5.19** | Sortino: **14.02** | Calmar: **18.59** | Profit Factor: **2.28**

## 📈 Performa Scorecard Portofolio

| Metrik Utama | Nilai Portofolio | Catatan |
|:---|:---:|:---|
| **Total Net Profit ($)** | `$+699.87` | Akumulasi keuntungan bersih 5x leverage |
| **Portfolio ROI (%)** | `+33.33%` | ROI berdasarkan kapital portofolio $100/koin |
| **Overall Win Rate** | `63.38%` | Rasio kemenangan rata-rata seluruh aset |
| **Total Trades** | `2,415` | Jumlah total posisi yang dieksekusi |
| **Rata-rata Trade / Bulan** | `23.3` | Rata-rata frekuensi trade bulanan portofolio |
| **Rata-rata Trade / Hari** | `0.77` | Rata-rata frekuensi trade harian portofolio |
| **Max Drawdown (5x)** | `56.59%` | Rata-rata penurunan terdalam portofolio |
| **Sharpe Ratio** | `5.19` | Efisiensi profit terhadap volatilitas portofolio |
| **Sortino Ratio** | `14.02` | Efisiensi profit terhadap downside deviation |
| **Calmar Ratio** | `18.59` | Rasio return tahunan terhadap drawdown |
| **Profit Factor** | `2.28` | Rasio gross profit dibagi gross loss |
| **Max Consecutive Loss** | `11` trades | Streak kekalahan beruntun terpanjang |
| **Worst Single Trade PnL** | `-24.90%` | Kerugian terdalam dalam satu trade tunggal |
| **95% Trades Loss Under** | `10.70%` | Nilai risiko (VaR P95) kerugian maksimal |

## ↕️ Analisis Arah Signal (LONG vs SHORT)

| Arah Posisi | Jumlah Trade | Distribusi | Menang | Kalah | Win Rate | PnL Bersih ($) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **LONG** | 626 | 25.9% | 403 | 223 | 64.38% | +258.00 |
| **SHORT** | 1,789 | 74.1% | 1,123 | 666 | 62.77% | +441.87 |

### Rincian Rata-rata Profitabilitas per Trade

| Tipe Trade | PnL Rata-rata ($) | PnL Rata-rata (%) |
|:---|:---:|:---:|
| **Trade Menang (Wins)** | `$+0.8524` | `+8.52%` |
| **Trade Kalah (Losses)** | `$-0.6759` | `-6.76%` |

## 📅 Scorecard Bulanan Portofolio

| Bulan | Total Trades | Wins | Losses | Win Rate | Net PnL ($) |
|:---|:---:|:---:|:---:|:---:|:---:|
| 2025-11 | 520 | 330 | 190 | 63.46% | $+153.39 |
| 2025-12 | 547 | 347 | 200 | 63.44% | $+179.61 |
| 2026-01 | 491 | 301 | 190 | 61.30% | $+137.15 |
| 2026-02 | 384 | 240 | 144 | 62.50% | $+95.66 |
| 2026-03 | 473 | 308 | 165 | 65.12% | $+134.06 |

## 🚪 Distribusi Alasan Keluar Posisi (Exit Reasons)

| Alasan Exit | Jumlah | Persentase | Wins | Losses | Win Rate | PnL Bersih ($) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| `guardian_exit` | 1,886 | 78.1% | 1,478 | 408 | 78.37% | $+1,037.65 |
| `sl_hit` | 476 | 19.7% | 1 | 475 | 0.21% | $-346.60 |
| `time_exit` | 53 | 2.2% | 47 | 6 | 88.68% | $+8.82 |

## 🪙 Scorecard Per Koin (Detailed Assets)

| Token | Win Rate | Trades | LONG WR | SHORT WR | Net PnL ($) | Max DD | Sharpe | Sortino | Calmar | PF |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1000PEPE** | 62.79% | 129 | 72.4% | 60.0% | `$+25.59` | 82.00% | 3.26 | 6.40 | 7.60 | 1.59 |
| **1000SHIB** | 55.64% | 133 | 48.9% | 59.3% | `$+21.62` | 80.93% | 3.13 | 8.19 | 6.50 | 1.56 |
| **ADA** | 60.32% | 126 | 64.3% | 59.2% | `$+29.19` | 74.69% | 4.29 | 12.22 | 9.52 | 1.79 |
| **ARB** | 62.70% | 126 | 61.0% | 63.5% | `$+43.03` | 63.74% | 4.99 | 11.93 | 16.44 | 2.14 |
| **AVAX** | 58.27% | 139 | 60.0% | 57.8% | `$+38.46` | 53.70% | 5.65 | 17.38 | 17.44 | 2.07 |
| **BNB** | 64.04% | 114 | 50.0% | 67.0% | `$+21.55` | 71.36% | 4.88 | 14.90 | 7.35 | 1.95 |
| **BTC** | 66.97% | 109 | 76.5% | 62.7% | `$+22.07` | 63.00% | 4.84 | 9.42 | 8.53 | 2.01 |
| **DOGE** | 56.62% | 136 | 63.2% | 54.1% | `$+24.45` | 70.22% | 3.54 | 9.02 | 8.48 | 1.65 |
| **DOT** | 62.86% | 105 | 59.3% | 64.1% | `$+35.88` | 21.78% | 5.67 | 16.62 | 40.12 | 2.60 |
| **ETH** | 70.18% | 114 | 69.0% | 70.6% | `$+49.82` | 21.51% | 8.40 | 21.77 | 56.40 | 3.74 |
| **HBAR** | 66.67% | 120 | 71.4% | 65.7% | `$+34.00` | 35.16% | 5.44 | 15.78 | 23.54 | 2.12 |
| **LINK** | 72.17% | 115 | 73.3% | 71.8% | `$+51.90` | 49.90% | 8.62 | 22.90 | 25.33 | 3.43 |
| **NEAR** | 54.39% | 114 | 46.4% | 57.0% | `$+13.97` | 132.10% | 1.89 | 5.26 | 2.58 | 1.33 |
| **ONDO** | 70.93% | 86 | 73.3% | 69.6% | `$+45.62` | 34.85% | 6.79 | 17.48 | 31.88 | 3.17 |
| **POL** | 63.10% | 84 | 70.4% | 59.7% | `$+23.85` | 79.34% | 4.07 | 9.22 | 7.32 | 2.13 |
| **SOL** | 65.04% | 123 | 82.6% | 61.0% | `$+40.19` | 54.54% | 5.99 | 15.90 | 17.94 | 2.36 |
| **SUI** | 61.29% | 124 | 63.3% | 60.6% | `$+63.92` | 68.48% | 8.09 | 23.14 | 22.73 | 3.52 |
| **TAO** | 68.81% | 109 | 72.4% | 67.5% | `$+41.42` | 42.08% | 5.09 | 9.47 | 23.97 | 2.17 |
| **TON** | 57.29% | 96 | 43.5% | 61.6% | `$+20.33` | 33.05% | 3.66 | 11.16 | 14.98 | 1.78 |
| **TRX** | 62.24% | 98 | 60.0% | 63.2% | `$+8.72` | 22.72% | 3.69 | 9.24 | 9.35 | 1.72 |
| **XRP** | 68.70% | 115 | 75.0% | 66.3% | `$+44.30` | 33.18% | 6.98 | 27.11 | 32.51 | 3.05 |

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
