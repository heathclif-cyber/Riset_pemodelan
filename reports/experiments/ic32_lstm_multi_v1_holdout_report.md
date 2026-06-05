# 📊 Holdout Backtest Report: `ic32_lstm_multi_v1`

**Tanggal Pembuatan**: 2026-06-05 11:09:19 UTC
**Model Run ID**: `ic32_lstm_multi_v1`
**Periode Pengujian (Temporal OOS)**: `2025-11-01 - 2026-04-01`

> [!NOTE]
> **Ringkasan Portofolio Eksekutif**:
> *   **Total Net Profit**: **$+2,463.97 USD** (ROI Portofolio: **+117.33%**)
> *   **Rata-rata Win Rate**: **64.84%** | Total Trades: **3,683**
> *   **Rata-rata Max Drawdown (5x)**: **69.78%**
> *   **Risk-Adjusted**: Sharpe: **6.14** | Sortino: **15.87** | Calmar: **18.15** | Profit Factor: **2.13**

## 📈 Performa Scorecard Portofolio

| Metrik Utama | Nilai Portofolio | Catatan |
|:---|:---:|:---|
| **Total Net Profit ($)** | `$+2,463.97` | Akumulasi keuntungan bersih 5x leverage |
| **Portfolio ROI (%)** | `+117.33%` | ROI berdasarkan kapital portofolio $100/koin |
| **Overall Win Rate** | `64.84%` | Rasio kemenangan rata-rata seluruh aset |
| **Total Trades** | `3,683` | Jumlah total posisi yang dieksekusi |
| **Rata-rata Trade / Bulan** | `35.6` | Rata-rata frekuensi trade bulanan portofolio |
| **Rata-rata Trade / Hari** | `1.17` | Rata-rata frekuensi trade harian portofolio |
| **Max Drawdown (5x)** | `69.78%` | Rata-rata penurunan terdalam portofolio |
| **Sharpe Ratio** | `6.14` | Efisiensi profit terhadap volatilitas portofolio |
| **Sortino Ratio** | `15.87` | Efisiensi profit terhadap downside deviation |
| **Calmar Ratio** | `18.15` | Rasio return tahunan terhadap drawdown |
| **Profit Factor** | `2.13` | Rasio gross profit dibagi gross loss |
| **Max Consecutive Loss** | `11` trades | Streak kekalahan beruntun terpanjang |
| **Worst Single Trade PnL** | `-24.90%` | Kerugian terdalam dalam satu trade tunggal |
| **95% Trades Loss Under** | `10.40%` | Nilai risiko (VaR P95) kerugian maksimal |

## ↕️ Analisis Arah Signal (LONG vs SHORT)

| Arah Posisi | Jumlah Trade | Distribusi | Menang | Kalah | Win Rate | PnL Bersih ($) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **LONG** | 1,334 | 36.2% | 780 | 554 | 58.47% | +673.39 |
| **SHORT** | 2,349 | 63.8% | 1,614 | 735 | 68.71% | +1,790.58 |

### Rincian Rata-rata Profitabilitas per Trade

| Tipe Trade | PnL Rata-rata ($) | PnL Rata-rata (%) |
|:---|:---:|:---:|
| **Trade Menang (Wins)** | `$+1.9585` | `+7.83%` |
| **Trade Kalah (Losses)** | `$-1.7258` | `-6.90%` |

## 📅 Scorecard Bulanan Portofolio

| Bulan | Total Trades | Wins | Losses | Win Rate | Net PnL ($) |
|:---|:---:|:---:|:---:|:---:|:---:|
| 2025-11 | 739 | 494 | 245 | 66.85% | $+525.97 |
| 2025-12 | 832 | 570 | 262 | 68.51% | $+748.50 |
| 2026-01 | 786 | 497 | 289 | 63.23% | $+495.04 |
| 2026-02 | 597 | 364 | 233 | 60.97% | $+255.97 |
| 2026-03 | 729 | 469 | 260 | 64.33% | $+438.50 |

## 🚪 Distribusi Alasan Keluar Posisi (Exit Reasons)

| Alasan Exit | Jumlah | Persentase | Wins | Losses | Win Rate | PnL Bersih ($) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| `guardian_exit` | 2,878 | 78.1% | 2,333 | 545 | 81.06% | $+3,817.08 |
| `sl_hit` | 738 | 20.0% | 1 | 737 | 0.14% | $-1,380.35 |
| `time_exit` | 67 | 1.8% | 60 | 7 | 89.55% | $+27.24 |

## 🪙 Scorecard Per Koin (Detailed Assets)

| Token | Win Rate | Trades | LONG WR | SHORT WR | Net PnL ($) | Max DD | Sharpe | Sortino | Calmar | PF |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1000PEPE** | 64.34% | 129 | 63.0% | 64.7% | `$+61.89` | 114.47% | 3.15 | 7.19 | 5.27 | 1.54 |
| **1000SHIB** | 62.69% | 134 | 51.0% | 69.9% | `$+60.49` | 38.36% | 4.30 | 10.73 | 15.36 | 1.80 |
| **ADA** | 70.22% | 178 | 68.9% | 71.2% | `$+162.39` | 58.05% | 8.19 | 21.25 | 27.25 | 2.62 |
| **ARB** | 61.14% | 193 | 55.7% | 65.7% | `$+142.00` | 96.08% | 5.85 | 16.75 | 14.39 | 2.00 |
| **AVAX** | 68.72% | 195 | 64.5% | 70.7% | `$+173.69` | 75.40% | 8.89 | 26.93 | 22.44 | 2.67 |
| **BNB** | 70.43% | 186 | 51.8% | 78.0% | `$+94.94` | 56.46% | 7.10 | 19.88 | 16.38 | 2.16 |
| **BTC** | 61.68% | 167 | 58.2% | 64.0% | `$+53.88` | 74.29% | 4.04 | 9.97 | 7.06 | 1.59 |
| **DOGE** | 65.36% | 179 | 67.9% | 64.2% | `$+137.61` | 70.51% | 6.83 | 16.78 | 19.01 | 2.28 |
| **DOT** | 67.68% | 198 | 53.4% | 76.0% | `$+166.15` | 61.13% | 8.08 | 17.63 | 26.47 | 2.57 |
| **ETH** | 67.84% | 199 | 59.7% | 72.0% | `$+168.79` | 95.12% | 8.94 | 20.93 | 17.28 | 2.81 |
| **HBAR** | 62.50% | 160 | 54.2% | 67.3% | `$+56.86` | 82.20% | 3.45 | 10.37 | 6.74 | 1.49 |
| **LINK** | 68.18% | 198 | 67.3% | 68.5% | `$+174.39` | 59.60% | 8.37 | 22.65 | 28.50 | 2.60 |
| **NEAR** | 62.83% | 191 | 57.1% | 65.6% | `$+139.09` | 78.55% | 6.08 | 15.30 | 17.25 | 2.02 |
| **ONDO** | 64.20% | 162 | 55.1% | 71.0% | `$+151.26` | 58.13% | 7.32 | 19.95 | 25.34 | 2.55 |
| **POL** | 61.25% | 160 | 62.1% | 60.6% | `$+102.01` | 61.10% | 5.39 | 14.14 | 16.26 | 1.95 |
| **SOL** | 64.85% | 202 | 60.7% | 66.4% | `$+165.26` | 59.84% | 7.80 | 21.19 | 26.90 | 2.46 |
| **SUI** | 64.67% | 184 | 52.1% | 72.6% | `$+153.10` | 44.59% | 6.56 | 17.00 | 33.44 | 2.29 |
| **TAO** | 59.30% | 172 | 44.0% | 71.1% | `$+96.88` | 92.78% | 3.77 | 7.95 | 10.17 | 1.60 |
| **TON** | 52.73% | 165 | 48.6% | 55.8% | `$+32.10` | 117.85% | 2.07 | 5.46 | 2.65 | 1.27 |
| **TRX** | 71.23% | 146 | 69.6% | 72.2% | `$+33.30` | 18.99% | 5.13 | 10.30 | 17.08 | 1.91 |
| **XRP** | 69.73% | 185 | 69.2% | 70.1% | `$+137.86` | 51.98% | 7.69 | 20.85 | 25.83 | 2.53 |

## ⛓️ Daftar Fitur Aktif dalam Model

Total terdapat **15 fitur aktif** yang digunakan oleh LightGBM entry, LSTM Soft Confirmation, dan Exit Guardian v3:

<details>
<summary>▶ Klik untuk melihat daftar lengkap fitur aktif</summary>

1. `rsi_6`
2. `stochrsi_k`
3. `stochrsi_d`
4. `rsi_slope_h4`
5. `rsi_h4`
6. `ema_21_slope_h4`
7. `cvd_slope_h4`
8. `ofi_h4_delta`
9. `cvd_momentum_adv`
10. `swing_momentum`
11. `dist_from_8h_high`
12. `price_in_range`
13. `long_short_ratio`
14. `dist_liq_50x_long`
15. `hmm_regime_enc`

</details>
