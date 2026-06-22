# 📊 Holdout Backtest Report: `ic32_guardian_ext_v1`

**Tanggal Pembuatan**: 2026-06-05 18:53:44 UTC
**Model Run ID**: `ic32_guardian_ext_v1`
**Periode Pengujian (Temporal OOS)**: `2025-11-01 - 2026-04-01`

> [!NOTE]
> **Ringkasan Portofolio Eksekutif**:
> *   **Total Net Profit**: **$+1,851.87 USD** (ROI Portofolio: **+88.18%**)
> *   **Rata-rata Win Rate**: **67.50%** | Total Trades: **2,426**
> *   **Rata-rata Max Drawdown (5x)**: **51.34%**
> *   **Risk-Adjusted**: Sharpe: **4.26** | Sortino: **7.47** | Calmar: **104.59** | Profit Factor: **2.53**

## 📈 Performa Scorecard Portofolio

| Metrik Utama | Nilai Portofolio | Catatan |
|:---|:---:|:---|
| **Total Net Profit ($)** | `$+1,851.87` | Akumulasi keuntungan bersih 5x leverage |
| **Portfolio ROI (%)** | `+88.18%` | ROI berdasarkan kapital portofolio $100/koin |
| **Overall Win Rate** | `67.50%` | Rasio kemenangan rata-rata seluruh aset |
| **Total Trades** | `2,426` | Jumlah total posisi yang dieksekusi |
| **Rata-rata Trade / Bulan** | `23.4` | Rata-rata frekuensi trade bulanan portofolio |
| **Rata-rata Trade / Hari** | `0.77` | Rata-rata frekuensi trade harian portofolio |
| **Max Drawdown (5x)** | `51.34%` | Rata-rata penurunan terdalam portofolio |
| **Sharpe Ratio** | `4.26` | Efisiensi profit terhadap volatilitas portofolio |
| **Sortino Ratio** | `7.47` | Efisiensi profit terhadap downside deviation |
| **Calmar Ratio** | `104.59` | Rasio return tahunan terhadap drawdown |
| **Profit Factor** | `2.53` | Rasio gross profit dibagi gross loss |
| **Max Consecutive Loss** | `10` trades | Streak kekalahan beruntun terpanjang |
| **Worst Single Trade PnL** | `-24.90%` | Kerugian terdalam dalam satu trade tunggal |
| **95% Trades Loss Under** | `10.30%` | Nilai risiko (VaR P95) kerugian maksimal |

## ↕️ Analisis Arah Signal (LONG vs SHORT)

| Arah Posisi | Jumlah Trade | Distribusi | Menang | Kalah | Win Rate | PnL Bersih ($) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **LONG** | 681 | 28.1% | 467 | 214 | 68.58% | +689.41 |
| **SHORT** | 1,745 | 71.9% | 1,169 | 576 | 66.99% | +1,162.46 |

### Rincian Rata-rata Profitabilitas per Trade

| Tipe Trade | PnL Rata-rata ($) | PnL Rata-rata (%) |
|:---|:---:|:---:|
| **Trade Menang (Wins)** | `$+1.9737` | `+7.90%` |
| **Trade Kalah (Losses)** | `$-1.7431` | `-6.97%` |

## 📅 Scorecard Bulanan Portofolio

| Bulan | Total Trades | Wins | Losses | Win Rate | Net PnL ($) |
|:---|:---:|:---:|:---:|:---:|:---:|
| 2025-11 | 550 | 381 | 169 | 69.27% | $+409.48 |
| 2025-12 | 550 | 379 | 171 | 68.91% | $+515.99 |
| 2026-01 | 491 | 317 | 174 | 64.56% | $+337.28 |
| 2026-02 | 369 | 252 | 117 | 68.29% | $+296.38 |
| 2026-03 | 466 | 307 | 159 | 65.88% | $+292.73 |

## 🚪 Distribusi Alasan Keluar Posisi (Exit Reasons)

| Alasan Exit | Jumlah | Persentase | Wins | Losses | Win Rate | PnL Bersih ($) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| `guardian_exit` | 1,964 | 81.0% | 1,596 | 368 | 81.26% | $+2,613.53 |
| `sl_hit` | 417 | 17.2% | 1 | 416 | 0.24% | $-779.27 |
| `time_exit` | 45 | 1.9% | 39 | 6 | 86.67% | $+17.61 |

## 🪙 Scorecard Per Koin (Detailed Assets)

| Token | Win Rate | Trades | LONG WR | SHORT WR | Net PnL ($) | Max DD | Sharpe | Sortino | Calmar | PF |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1000PEPE** | 64.84% | 128 | 74.2% | 61.9% | `$+72.88` | 102.46% | 2.61 | 2.36 | 26.05 | 1.64 |
| **1000SHIB** | 61.90% | 126 | 56.2% | 65.4% | `$+80.20` | 81.12% | 4.11 | 6.81 | 39.45 | 2.05 |
| **ADA** | 65.08% | 126 | 79.3% | 60.8% | `$+81.19` | 97.75% | 4.47 | 8.16 | 33.52 | 1.95 |
| **ARB** | 66.92% | 133 | 66.7% | 67.1% | `$+107.78` | 63.74% | 4.21 | 8.42 | 89.69 | 2.19 |
| **AVAX** | 64.54% | 141 | 65.7% | 64.1% | `$+109.91` | 42.42% | 5.29 | 9.92 | 140.16 | 2.48 |
| **BNB** | 73.68% | 114 | 66.7% | 75.6% | `$+63.48` | 39.16% | 4.98 | 7.58 | 52.75 | 2.45 |
| **BTC** | 74.04% | 104 | 81.8% | 70.4% | `$+65.47` | 29.99% | 5.13 | 5.79 | 72.90 | 2.73 |
| **DOGE** | 60.67% | 150 | 65.1% | 58.9% | `$+86.25` | 60.00% | 2.40 | 7.08 | 61.36 | 1.82 |
| **DOT** | 72.22% | 108 | 65.7% | 75.3% | `$+127.19` | 37.68% | 4.77 | 5.42 | 212.49 | 3.96 |
| **ETH** | 75.44% | 114 | 71.9% | 76.8% | `$+118.04` | 28.00% | 6.44 | 16.88 | 245.38 | 3.84 |
| **HBAR** | 67.62% | 105 | 65.2% | 68.3% | `$+69.96` | 35.52% | 4.47 | 6.25 | 69.62 | 2.14 |
| **LINK** | 78.38% | 111 | 88.9% | 75.0% | `$+140.67` | 32.03% | 7.10 | 17.04 | 308.00 | 4.32 |
| **NEAR** | 57.72% | 123 | 50.0% | 60.9% | `$+60.89` | 75.40% | 3.42 | 8.85 | 25.40 | 1.56 |
| **ONDO** | 69.77% | 86 | 70.0% | 69.6% | `$+96.24` | 34.85% | 3.96 | 3.00 | 130.89 | 2.83 |
| **POL** | 65.06% | 83 | 74.1% | 60.7% | `$+63.24` | 79.28% | 4.13 | 6.50 | 25.88 | 2.32 |
| **SOL** | 72.52% | 131 | 83.3% | 70.1% | `$+119.92` | 63.80% | 5.24 | 9.38 | 111.22 | 3.02 |
| **SUI** | 68.91% | 119 | 70.0% | 68.5% | `$+135.34` | 40.55% | 4.80 | 7.15 | 224.48 | 3.60 |
| **TAO** | 66.96% | 115 | 69.4% | 65.8% | `$+101.70` | 42.24% | 2.96 | 3.25 | 120.48 | 2.04 |
| **TON** | 52.63% | 95 | 48.1% | 54.4% | `$+29.66` | 31.82% | 2.07 | 4.97 | 17.94 | 1.46 |
| **TRX** | 64.00% | 100 | 61.3% | 65.2% | `$+12.86` | 28.29% | 1.83 | 1.28 | 6.17 | 1.38 |
| **XRP** | 74.56% | 114 | 81.2% | 72.0% | `$+109.01` | 32.05% | 5.01 | 10.85 | 182.47 | 3.45 |

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
