# 📊 Holdout Backtest Report: `ic32_lstm_strong_v1`

**Tanggal Pembuatan**: 2026-06-05 05:44:05 UTC
**Model Run ID**: `ic32_lstm_strong_v1`
**Periode Pengujian (Temporal OOS)**: `2025-11-01 - 2026-04-01`

> [!NOTE]
> **Ringkasan Portofolio Eksekutif**:
> *   **Total Net Profit**: **$+2,011.70 USD** (ROI Portofolio: **+95.80%**)
> *   **Rata-rata Win Rate**: **61.27%** | Total Trades: **4,002**
> *   **Rata-rata Max Drawdown (5x)**: **88.57%**
> *   **Risk-Adjusted**: Sharpe: **4.87** | Sortino: **13.11** | Calmar: **11.65** | Profit Factor: **1.78**

## 📈 Performa Scorecard Portofolio

| Metrik Utama | Nilai Portofolio | Catatan |
|:---|:---:|:---|
| **Total Net Profit ($)** | `$+2,011.70` | Akumulasi keuntungan bersih 5x leverage |
| **Portfolio ROI (%)** | `+95.80%` | ROI berdasarkan kapital portofolio $100/koin |
| **Overall Win Rate** | `61.27%` | Rasio kemenangan rata-rata seluruh aset |
| **Total Trades** | `4,002` | Jumlah total posisi yang dieksekusi |
| **Rata-rata Trade / Bulan** | `38.7` | Rata-rata frekuensi trade bulanan portofolio |
| **Rata-rata Trade / Hari** | `1.27` | Rata-rata frekuensi trade harian portofolio |
| **Max Drawdown (5x)** | `88.57%` | Rata-rata penurunan terdalam portofolio |
| **Sharpe Ratio** | `4.87` | Efisiensi profit terhadap volatilitas portofolio |
| **Sortino Ratio** | `13.11` | Efisiensi profit terhadap downside deviation |
| **Calmar Ratio** | `11.65` | Rasio return tahunan terhadap drawdown |
| **Profit Factor** | `1.78` | Rasio gross profit dibagi gross loss |
| **Max Consecutive Loss** | `13` trades | Streak kekalahan beruntun terpanjang |
| **Worst Single Trade PnL** | `-24.90%` | Kerugian terdalam dalam satu trade tunggal |
| **95% Trades Loss Under** | `10.39%` | Nilai risiko (VaR P95) kerugian maksimal |

## ↕️ Analisis Arah Signal (LONG vs SHORT)

| Arah Posisi | Jumlah Trade | Distribusi | Menang | Kalah | Win Rate | PnL Bersih ($) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **LONG** | 1,944 | 48.6% | 1,006 | 938 | 51.75% | +323.32 |
| **SHORT** | 2,058 | 51.4% | 1,441 | 617 | 70.02% | +1,688.37 |

### Rincian Rata-rata Profitabilitas per Trade

| Tipe Trade | PnL Rata-rata ($) | PnL Rata-rata (%) |
|:---|:---:|:---:|
| **Trade Menang (Wins)** | `$+1.9098` | `+7.64%` |
| **Trade Kalah (Losses)** | `$-1.7117` | `-6.85%` |

## 📅 Scorecard Bulanan Portofolio

| Bulan | Total Trades | Wins | Losses | Win Rate | Net PnL ($) |
|:---|:---:|:---:|:---:|:---:|:---:|
| 2025-11 | 748 | 503 | 245 | 67.25% | $+537.53 |
| 2025-12 | 799 | 521 | 278 | 65.21% | $+618.63 |
| 2026-01 | 931 | 533 | 398 | 57.25% | $+333.49 |
| 2026-02 | 695 | 372 | 323 | 53.53% | $+105.70 |
| 2026-03 | 829 | 518 | 311 | 62.48% | $+416.34 |

## 🚪 Distribusi Alasan Keluar Posisi (Exit Reasons)

| Alasan Exit | Jumlah | Persentase | Wins | Losses | Win Rate | PnL Bersih ($) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| `guardian_exit` | 3,050 | 76.2% | 2,387 | 663 | 78.26% | $+3,629.46 |
| `sl_hit` | 885 | 22.1% | 2 | 883 | 0.23% | $-1,646.05 |
| `time_exit` | 67 | 1.7% | 58 | 9 | 86.57% | $+28.28 |

## 🪙 Scorecard Per Koin (Detailed Assets)

| Token | Win Rate | Trades | LONG WR | SHORT WR | Net PnL ($) | Max DD | Sharpe | Sortino | Calmar | PF |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1000PEPE** | 69.11% | 123 | 66.7% | 70.2% | `$+81.22` | 62.79% | 4.17 | 9.43 | 12.60 | 1.82 |
| **1000SHIB** | 62.50% | 136 | 50.9% | 69.9% | `$+62.93` | 55.18% | 4.13 | 9.90 | 11.11 | 1.77 |
| **ADA** | 59.57% | 235 | 52.4% | 67.6% | `$+119.31` | 89.97% | 5.18 | 15.25 | 12.92 | 1.67 |
| **ARB** | 62.98% | 208 | 57.4% | 69.0% | `$+146.96` | 75.78% | 6.30 | 16.71 | 18.89 | 2.04 |
| **AVAX** | 63.96% | 222 | 53.6% | 72.0% | `$+144.91` | 95.80% | 7.24 | 23.41 | 14.73 | 2.15 |
| **BNB** | 61.80% | 178 | 43.8% | 74.3% | `$+50.27` | 69.68% | 3.74 | 11.93 | 7.03 | 1.51 |
| **BTC** | 61.36% | 176 | 56.9% | 64.4% | `$+51.74` | 71.62% | 3.94 | 8.58 | 7.04 | 1.57 |
| **DOGE** | 57.32% | 157 | 50.0% | 62.4% | `$+80.28` | 72.76% | 4.07 | 10.75 | 10.75 | 1.71 |
| **DOT** | 62.62% | 214 | 54.4% | 74.2% | `$+133.93` | 97.86% | 5.87 | 14.71 | 13.33 | 1.88 |
| **ETH** | 66.98% | 215 | 55.3% | 77.7% | `$+137.51` | 102.97% | 7.36 | 19.33 | 13.01 | 2.22 |
| **HBAR** | 58.38% | 197 | 47.6% | 70.2% | `$+42.56` | 105.86% | 2.48 | 6.34 | 3.92 | 1.30 |
| **LINK** | 64.10% | 195 | 57.5% | 70.3% | `$+148.10` | 46.62% | 6.75 | 21.77 | 30.94 | 2.18 |
| **NEAR** | 56.77% | 192 | 45.7% | 70.1% | `$+110.23` | 114.65% | 4.88 | 12.56 | 9.36 | 1.79 |
| **ONDO** | 62.86% | 175 | 53.3% | 72.9% | `$+151.22` | 94.86% | 6.95 | 20.78 | 15.53 | 2.29 |
| **POL** | 62.01% | 179 | 57.4% | 68.0% | `$+114.12` | 63.38% | 6.21 | 16.97 | 17.54 | 2.07 |
| **SOL** | 60.11% | 188 | 51.3% | 66.4% | `$+112.91` | 82.78% | 5.33 | 14.00 | 13.29 | 1.90 |
| **SUI** | 57.14% | 245 | 45.9% | 70.5% | `$+89.26` | 169.91% | 3.51 | 9.19 | 5.12 | 1.43 |
| **TAO** | 58.15% | 184 | 44.9% | 73.3% | `$+82.36` | 92.83% | 3.26 | 7.02 | 8.64 | 1.48 |
| **TON** | 50.80% | 187 | 39.6% | 62.6% | `$+23.39` | 147.41% | 1.40 | 4.25 | 1.55 | 1.16 |
| **TRX** | 68.07% | 166 | 63.2% | 71.4% | `$+32.12` | 34.07% | 4.63 | 8.80 | 9.18 | 1.71 |
| **XRP** | 60.00% | 230 | 50.8% | 70.0% | `$+96.36` | 113.14% | 4.87 | 13.72 | 8.29 | 1.68 |

## ⛓️ Daftar Fitur Aktif dalam Model

Total terdapat **7 fitur aktif** yang digunakan oleh LightGBM entry, LSTM Soft Confirmation, dan Exit Guardian v3:

<details>
<summary>▶ Klik untuk melihat daftar lengkap fitur aktif</summary>

1. `dist_liq_20x_long`
2. `rsi_slope_h4`
3. `log_ret_20`
4. `dist_liq_50x_long`
5. `long_short_ratio`
6. `cvd_slope_h4`
7. `ofi_h4_delta`

</details>
