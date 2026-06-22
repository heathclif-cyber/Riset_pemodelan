# 📊 Holdout Backtest Report: `lgbm_v2_nonlinear`

**Tanggal Pembuatan**: 2026-06-08 23:50:46 UTC
**Model Run ID**: `lgbm_v2_nonlinear`
**Periode Pengujian (Temporal OOS)**: `2025-11-01 - 2026-04-01`

> [!NOTE]
> **Ringkasan Portofolio Eksekutif**:
> *   **Total Net Profit**: **$+416.63 USD** (ROI Portofolio: **+19.84%**)
> *   **Rata-rata Win Rate**: **60.14%** | Total Trades: **2,064**
> *   **Rata-rata Max Drawdown (5x)**: **80.57%**
> *   **Risk-Adjusted**: Sharpe: **3.15** | Sortino: **8.42** | Calmar: **7.15** | Profit Factor: **1.75**

## 📈 Performa Scorecard Portofolio

| Metrik Utama | Nilai Portofolio | Catatan |
|:---|:---:|:---|
| **Total Net Profit ($)** | `$+416.63` | Akumulasi keuntungan bersih 5x leverage |
| **Portfolio ROI (%)** | `+19.84%` | ROI berdasarkan kapital portofolio $100/koin |
| **Overall Win Rate** | `60.14%` | Rasio kemenangan rata-rata seluruh aset |
| **Total Trades** | `2,064` | Jumlah total posisi yang dieksekusi |
| **Rata-rata Trade / Bulan** | `19.9` | Rata-rata frekuensi trade bulanan portofolio |
| **Rata-rata Trade / Hari** | `0.66` | Rata-rata frekuensi trade harian portofolio |
| **Max Drawdown (5x)** | `80.57%` | Rata-rata penurunan terdalam portofolio |
| **Sharpe Ratio** | `3.15` | Efisiensi profit terhadap volatilitas portofolio |
| **Sortino Ratio** | `8.42` | Efisiensi profit terhadap downside deviation |
| **Calmar Ratio** | `7.15` | Rasio return tahunan terhadap drawdown |
| **Profit Factor** | `1.75` | Rasio gross profit dibagi gross loss |
| **Max Consecutive Loss** | `12` trades | Streak kekalahan beruntun terpanjang |
| **Worst Single Trade PnL** | `-24.50%` | Kerugian terdalam dalam satu trade tunggal |
| **95% Trades Loss Under** | `11.60%` | Nilai risiko (VaR P95) kerugian maksimal |

## ↕️ Analisis Arah Signal (LONG vs SHORT)

| Arah Posisi | Jumlah Trade | Distribusi | Menang | Kalah | Win Rate | PnL Bersih ($) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **LONG** | 272 | 13.2% | 174 | 98 | 63.97% | +40.13 |
| **SHORT** | 1,792 | 86.8% | 1,069 | 723 | 59.65% | +376.50 |

### Rincian Rata-rata Profitabilitas per Trade

| Tipe Trade | PnL Rata-rata ($) | PnL Rata-rata (%) |
|:---|:---:|:---:|
| **Trade Menang (Wins)** | `$+0.8003` | `+8.00%` |
| **Trade Kalah (Losses)** | `$-0.7043` | `-7.04%` |

## 📅 Scorecard Bulanan Portofolio

| Bulan | Total Trades | Wins | Losses | Win Rate | Net PnL ($) |
|:---|:---:|:---:|:---:|:---:|:---:|
| 2025-11 | 523 | 317 | 206 | 60.61% | $+97.78 |
| 2025-12 | 305 | 180 | 125 | 59.02% | $+103.98 |
| 2026-01 | 672 | 410 | 262 | 61.01% | $+171.59 |
| 2026-02 | 377 | 250 | 127 | 66.31% | $+60.84 |
| 2026-03 | 187 | 86 | 101 | 45.99% | $-17.57 |

## 🚪 Distribusi Alasan Keluar Posisi (Exit Reasons)

| Alasan Exit | Jumlah | Persentase | Wins | Losses | Win Rate | PnL Bersih ($) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| `guardian_exit` | 1,662 | 80.5% | 1,187 | 475 | 71.42% | $+665.31 |
| `sl_hit` | 345 | 16.7% | 0 | 345 | 0.00% | $-261.84 |
| `time_exit` | 57 | 2.8% | 56 | 1 | 98.25% | $+13.16 |

## 🪙 Scorecard Per Koin (Detailed Assets)

| Token | Win Rate | Trades | LONG WR | SHORT WR | Net PnL ($) | Max DD | Sharpe | Sortino | Calmar | PF |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1000PEPE** | 49.23% | 65 | 22.2% | 53.6% | `$-6.16` | 158.68% | -1.28 | -2.01 | -0.95 | 0.77 |
| **1000SHIB** | 63.39% | 112 | 39.1% | 69.7% | `$+29.32` | 59.81% | 5.04 | 10.99 | 11.94 | 2.13 |
| **ADA** | 48.72% | 78 | 55.6% | 47.8% | `$+4.67` | 66.28% | 0.84 | 2.39 | 1.72 | 1.16 |
| **ARB** | 65.56% | 90 | 72.0% | 63.1% | `$+23.58` | 37.73% | 3.93 | 8.40 | 15.22 | 1.96 |
| **AVAX** | 55.19% | 154 | 84.6% | 52.5% | `$+17.31` | 84.42% | 2.59 | 5.39 | 4.99 | 1.39 |
| **BNB** | 54.21% | 107 | 78.6% | 50.5% | `$+20.66` | 55.35% | 4.57 | 17.36 | 9.09 | 2.01 |
| **BTC** | 55.70% | 79 | 56.2% | 55.6% | `$+12.64` | 46.02% | 3.06 | 10.40 | 6.69 | 1.70 |
| **DOGE** | 57.26% | 117 | 28.6% | 61.2% | `$+12.89` | 73.87% | 1.95 | 4.93 | 4.25 | 1.34 |
| **DOT** | 66.41% | 128 | 53.3% | 68.1% | `$+31.03` | 144.30% | 4.13 | 9.56 | 5.24 | 1.83 |
| **ETH** | 61.95% | 113 | 80.0% | 58.1% | `$+34.06` | 80.94% | 5.48 | 14.77 | 10.25 | 2.30 |
| **HBAR** | 59.76% | 82 | 100.0% | 56.6% | `$+11.56` | 60.09% | 2.20 | 6.51 | 4.68 | 1.45 |
| **LINK** | 60.95% | 105 | 81.8% | 58.5% | `$+18.70` | 73.85% | 2.94 | 8.03 | 6.17 | 1.58 |
| **NEAR** | 60.26% | 78 | 100.0% | 59.2% | `$+19.63` | 103.46% | 2.76 | 5.65 | 4.62 | 1.70 |
| **ONDO** | 67.19% | 64 | 75.0% | 64.6% | `$+22.27` | 40.39% | 3.92 | 8.97 | 13.43 | 2.13 |
| **POL** | 70.67% | 75 | 55.6% | 72.7% | `$+26.87` | 40.59% | 4.74 | 9.41 | 16.12 | 2.45 |
| **SOL** | 62.83% | 113 | 70.0% | 62.1% | `$+30.72` | 97.79% | 4.62 | 17.26 | 7.65 | 2.02 |
| **SUI** | 73.74% | 99 | 66.7% | 74.4% | `$+43.09` | 71.15% | 6.58 | 17.56 | 14.75 | 2.94 |
| **TAO** | 72.73% | 99 | 75.0% | 72.4% | `$+40.37` | 110.19% | 5.39 | 9.86 | 8.92 | 2.39 |
| **TON** | 52.43% | 103 | 60.0% | 51.1% | `$+8.11` | 64.35% | 1.40 | 4.01 | 3.07 | 1.26 |
| **TRX** | 47.62% | 84 | 70.6% | 41.8% | `$-4.66` | 87.01% | -1.96 | -3.69 | -1.30 | 0.73 |
| **XRP** | 57.14% | 119 | 57.1% | 57.1% | `$+19.95` | 135.63% | 3.31 | 11.15 | 3.58 | 1.59 |

## ⛓️ Daftar Fitur Aktif dalam Model

Total terdapat **19 fitur aktif** yang digunakan oleh LightGBM entry, LSTM Soft Confirmation, dan Exit Guardian v3:

<details>
<summary>▶ Klik untuk melihat daftar lengkap fitur aktif</summary>

1. `dist_liq_50x_short`
2. `vol_price_confirm`
3. `dist_from_8h_high`
4. `dist_swing_low`
5. `absorption_at_swing`
6. `ema_7_h1`
7. `Sell_Liq`
8. `dist_liq_20x_short`
9. `cvd_slope_h4`
10. `ofi_h4_delta`
11. `rsi_h4`
12. `log_ret_1`
13. `dist_swing_high`
14. `etf_gbtc_change_usd`
15. `rsi_slope_h4`
16. `ema_21_slope_h4`
17. `ema_50_h1`
18. `etf_total_change_usd`
19. `hmm_regime_enc`

</details>
