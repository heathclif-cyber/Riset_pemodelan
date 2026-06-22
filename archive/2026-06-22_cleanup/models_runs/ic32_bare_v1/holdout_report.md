# 📊 Holdout Backtest Report: `ic32_bare_v1`

**Tanggal Pembuatan**: 2026-06-08 20:20:08 UTC
**Model Run ID**: `ic32_bare_v1`
**Periode Pengujian (Temporal OOS)**: `2025-11-01 - 2026-04-01`

> [!NOTE]
> **Ringkasan Portofolio Eksekutif**:
> *   **Total Net Profit**: **$+309.93 USD** (ROI Portofolio: **+14.76%**)
> *   **Rata-rata Win Rate**: **59.27%** | Total Trades: **1,527**
> *   **Rata-rata Max Drawdown (5x)**: **56.19%**
> *   **Risk-Adjusted**: Sharpe: **2.84** | Sortino: **7.95** | Calmar: **8.26** | Profit Factor: **1.80**

## 📈 Performa Scorecard Portofolio

| Metrik Utama | Nilai Portofolio | Catatan |
|:---|:---:|:---|
| **Total Net Profit ($)** | `$+309.93` | Akumulasi keuntungan bersih 5x leverage |
| **Portfolio ROI (%)** | `+14.76%` | ROI berdasarkan kapital portofolio $100/koin |
| **Overall Win Rate** | `59.27%` | Rasio kemenangan rata-rata seluruh aset |
| **Total Trades** | `1,527` | Jumlah total posisi yang dieksekusi |
| **Rata-rata Trade / Bulan** | `14.8` | Rata-rata frekuensi trade bulanan portofolio |
| **Rata-rata Trade / Hari** | `0.48` | Rata-rata frekuensi trade harian portofolio |
| **Max Drawdown (5x)** | `56.19%` | Rata-rata penurunan terdalam portofolio |
| **Sharpe Ratio** | `2.84` | Efisiensi profit terhadap volatilitas portofolio |
| **Sortino Ratio** | `7.95` | Efisiensi profit terhadap downside deviation |
| **Calmar Ratio** | `8.26` | Rasio return tahunan terhadap drawdown |
| **Profit Factor** | `1.80` | Rasio gross profit dibagi gross loss |
| **Max Consecutive Loss** | `11` trades | Streak kekalahan beruntun terpanjang |
| **Worst Single Trade PnL** | `-26.90%` | Kerugian terdalam dalam satu trade tunggal |
| **95% Trades Loss Under** | `11.30%` | Nilai risiko (VaR P95) kerugian maksimal |

## ↕️ Analisis Arah Signal (LONG vs SHORT)

| Arah Posisi | Jumlah Trade | Distribusi | Menang | Kalah | Win Rate | PnL Bersih ($) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **LONG** | 236 | 15.5% | 156 | 80 | 66.10% | +79.96 |
| **SHORT** | 1,291 | 84.5% | 752 | 539 | 58.25% | +229.97 |

### Rincian Rata-rata Profitabilitas per Trade

| Tipe Trade | PnL Rata-rata ($) | PnL Rata-rata (%) |
|:---|:---:|:---:|
| **Trade Menang (Wins)** | `$+0.8183` | `+8.18%` |
| **Trade Kalah (Losses)** | `$-0.6996` | `-6.99%` |

## 📅 Scorecard Bulanan Portofolio

| Bulan | Total Trades | Wins | Losses | Win Rate | Net PnL ($) |
|:---|:---:|:---:|:---:|:---:|:---:|
| 2025-11 | 328 | 189 | 139 | 57.62% | $+60.64 |
| 2025-12 | 355 | 216 | 139 | 60.85% | $+89.95 |
| 2026-01 | 313 | 165 | 148 | 52.72% | $+12.04 |
| 2026-02 | 254 | 154 | 100 | 60.63% | $+68.57 |
| 2026-03 | 277 | 184 | 93 | 66.43% | $+78.74 |

## 🚪 Distribusi Alasan Keluar Posisi (Exit Reasons)

| Alasan Exit | Jumlah | Persentase | Wins | Losses | Win Rate | PnL Bersih ($) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| `guardian_exit` | 1,190 | 77.9% | 885 | 305 | 74.37% | $+551.18 |
| `sl_hit` | 314 | 20.6% | 1 | 313 | 0.32% | $-245.73 |
| `time_exit` | 23 | 1.5% | 22 | 1 | 95.65% | $+4.48 |

## 🪙 Scorecard Per Koin (Detailed Assets)

| Token | Win Rate | Trades | LONG WR | SHORT WR | Net PnL ($) | Max DD | Sharpe | Sortino | Calmar | PF |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1000PEPE** | 56.16% | 73 | 75.0% | 52.5% | `$+5.07` | 84.21% | 0.92 | 1.69 | 1.47 | 1.18 |
| **1000SHIB** | 60.00% | 75 | 46.2% | 62.9% | `$+22.40` | 38.80% | 3.96 | 10.90 | 14.06 | 2.15 |
| **ADA** | 62.16% | 74 | 61.5% | 62.3% | `$+22.55` | 54.81% | 3.91 | 11.39 | 10.02 | 2.06 |
| **ARB** | 61.43% | 70 | 81.8% | 57.6% | `$+22.56` | 38.09% | 3.87 | 11.59 | 14.42 | 2.26 |
| **AVAX** | 61.63% | 86 | 77.8% | 59.7% | `$+21.57` | 41.85% | 4.12 | 10.53 | 12.55 | 1.99 |
| **BNB** | 64.38% | 73 | 55.6% | 65.6% | `$+15.37` | 56.54% | 4.40 | 16.23 | 6.62 | 2.20 |
| **BTC** | 57.14% | 49 | 66.7% | 54.0% | `$-1.15` | 60.91% | -0.38 | -0.77 | -0.46 | 0.92 |
| **DOGE** | 56.38% | 94 | 63.6% | 55.4% | `$+14.57` | 69.64% | 2.55 | 6.94 | 5.09 | 1.52 |
| **DOT** | 69.33% | 75 | 66.7% | 69.6% | `$+25.07` | 46.73% | 4.47 | 8.68 | 13.06 | 2.37 |
| **ETH** | 66.10% | 59 | 76.9% | 63.0% | `$+21.61` | 25.25% | 5.23 | 14.49 | 20.84 | 3.15 |
| **HBAR** | 55.84% | 77 | 37.5% | 58.0% | `$+10.34` | 71.04% | 1.94 | 5.51 | 3.54 | 1.39 |
| **LINK** | 62.65% | 83 | 90.9% | 58.3% | `$+17.30` | 103.28% | 3.17 | 9.45 | 4.08 | 1.67 |
| **NEAR** | 51.76% | 85 | 60.0% | 50.7% | `$+2.26` | 84.38% | 0.34 | 0.69 | 0.65 | 1.06 |
| **ONDO** | 50.98% | 51 | 66.7% | 44.4% | `$+2.73` | 57.95% | 0.58 | 1.45 | 1.15 | 1.14 |
| **POL** | 58.93% | 56 | 78.6% | 52.4% | `$+12.13` | 60.79% | 2.56 | 5.92 | 4.86 | 1.75 |
| **SOL** | 64.29% | 98 | 81.8% | 62.1% | `$+29.90` | 73.31% | 4.93 | 13.98 | 9.93 | 2.26 |
| **SUI** | 58.90% | 73 | 46.7% | 62.1% | `$+29.50` | 24.77% | 4.92 | 11.56 | 29.00 | 2.56 |
| **TAO** | 58.33% | 48 | 55.6% | 59.0% | `$+0.05` | 81.21% | 0.01 | 0.02 | 0.02 | 1.00 |
| **TON** | 48.65% | 74 | 44.4% | 49.2% | `$+7.42` | 41.18% | 1.64 | 5.21 | 4.39 | 1.34 |
| **TRX** | 56.41% | 78 | 71.4% | 53.1% | `$+3.77` | 21.78% | 1.85 | 4.73 | 4.21 | 1.35 |
| **XRP** | 63.16% | 76 | 72.7% | 61.5% | `$+24.93` | 43.50% | 4.59 | 16.65 | 13.95 | 2.45 |

## ⛓️ Daftar Fitur Aktif dalam Model

Total terdapat **24 fitur aktif** yang digunakan oleh LightGBM entry, LSTM Soft Confirmation, dan Exit Guardian v3:

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
10. `stochrsi_d`
11. `ofi_h4_delta`
12. `dist_liq_50x_short`
13. `Buy_Liq`
14. `relative_strength_z`
15. `dist_liq_20x_long`
16. `cvd_momentum_adv`
17. `Sell_Liq`
18. `cvd_slope_h4`
19. `ema_21_slope_h4`
20. `ema_50_h1`
21. `h4_trend`
22. `log_ret_20`
23. `whale_retail_divergence`
24. `hmm_regime_enc`

</details>
