# 📊 Holdout Backtest Report: `ic32_hybrid_ic_v1`

**Tanggal Pembuatan**: 2026-06-09 00:27:57 UTC
**Model Run ID**: `ic32_hybrid_ic_v1`
**Periode Pengujian (Temporal OOS)**: `2025-11-01 - 2026-04-01`

> [!NOTE]
> **Ringkasan Portofolio Eksekutif**:
> *   **Total Net Profit**: **$+805.68 USD** (ROI Portofolio: **+38.37%**)
> *   **Rata-rata Win Rate**: **60.56%** | Total Trades: **3,427**
> *   **Rata-rata Max Drawdown (5x)**: **82.75%**
> *   **Risk-Adjusted**: Sharpe: **4.93** | Sortino: **12.94** | Calmar: **11.99** | Profit Factor: **1.88**

## 📈 Performa Scorecard Portofolio

| Metrik Utama | Nilai Portofolio | Catatan |
|:---|:---:|:---|
| **Total Net Profit ($)** | `$+805.68` | Akumulasi keuntungan bersih 5x leverage |
| **Portfolio ROI (%)** | `+38.37%` | ROI berdasarkan kapital portofolio $100/koin |
| **Overall Win Rate** | `60.56%` | Rasio kemenangan rata-rata seluruh aset |
| **Total Trades** | `3,427` | Jumlah total posisi yang dieksekusi |
| **Rata-rata Trade / Bulan** | `33.1` | Rata-rata frekuensi trade bulanan portofolio |
| **Rata-rata Trade / Hari** | `1.09` | Rata-rata frekuensi trade harian portofolio |
| **Max Drawdown (5x)** | `82.75%` | Rata-rata penurunan terdalam portofolio |
| **Sharpe Ratio** | `4.93` | Efisiensi profit terhadap volatilitas portofolio |
| **Sortino Ratio** | `12.94` | Efisiensi profit terhadap downside deviation |
| **Calmar Ratio** | `11.99` | Rasio return tahunan terhadap drawdown |
| **Profit Factor** | `1.88` | Rasio gross profit dibagi gross loss |
| **Max Consecutive Loss** | `14` trades | Streak kekalahan beruntun terpanjang |
| **Worst Single Trade PnL** | `-31.80%` | Kerugian terdalam dalam satu trade tunggal |
| **95% Trades Loss Under** | `11.17%` | Nilai risiko (VaR P95) kerugian maksimal |

## ↕️ Analisis Arah Signal (LONG vs SHORT)

| Arah Posisi | Jumlah Trade | Distribusi | Menang | Kalah | Win Rate | PnL Bersih ($) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **LONG** | 612 | 17.9% | 384 | 228 | 62.75% | +197.43 |
| **SHORT** | 2,815 | 82.1% | 1,688 | 1,127 | 59.96% | +608.25 |

### Rincian Rata-rata Profitabilitas per Trade

| Tipe Trade | PnL Rata-rata ($) | PnL Rata-rata (%) |
|:---|:---:|:---:|
| **Trade Menang (Wins)** | `$+0.8517` | `+8.52%` |
| **Trade Kalah (Losses)** | `$-0.7078` | `-7.08%` |

## 📅 Scorecard Bulanan Portofolio

| Bulan | Total Trades | Wins | Losses | Win Rate | Net PnL ($) |
|:---|:---:|:---:|:---:|:---:|:---:|
| 2025-11 | 738 | 434 | 304 | 58.81% | $+171.57 |
| 2025-12 | 644 | 383 | 261 | 59.47% | $+183.42 |
| 2026-01 | 955 | 583 | 372 | 61.05% | $+184.54 |
| 2026-02 | 511 | 335 | 176 | 65.56% | $+174.35 |
| 2026-03 | 579 | 337 | 242 | 58.20% | $+91.80 |

## 🚪 Distribusi Alasan Keluar Posisi (Exit Reasons)

| Alasan Exit | Jumlah | Persentase | Wins | Losses | Win Rate | PnL Bersih ($) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| `guardian_exit` | 2,730 | 79.7% | 1,995 | 735 | 73.08% | $+1,267.62 |
| `sl_hit` | 614 | 17.9% | 1 | 613 | 0.16% | $-477.75 |
| `time_exit` | 83 | 2.4% | 76 | 7 | 91.57% | $+15.82 |

## 🪙 Scorecard Per Koin (Detailed Assets)

| Token | Win Rate | Trades | LONG WR | SHORT WR | Net PnL ($) | Max DD | Sharpe | Sortino | Calmar | PF |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1000PEPE** | 63.46% | 156 | 59.4% | 64.5% | `$+32.78` | 153.03% | 3.72 | 6.99 | 5.22 | 1.62 |
| **1000SHIB** | 56.44% | 163 | 64.7% | 54.3% | `$+30.59` | 63.59% | 4.35 | 12.58 | 11.71 | 1.70 |
| **ADA** | 60.45% | 177 | 63.3% | 59.9% | `$+42.86` | 70.94% | 5.05 | 14.56 | 14.71 | 1.79 |
| **ARB** | 59.39% | 165 | 65.8% | 57.5% | `$+39.55` | 111.83% | 4.15 | 10.69 | 8.61 | 1.69 |
| **AVAX** | 56.08% | 189 | 58.1% | 55.7% | `$+42.65` | 75.23% | 5.15 | 11.55 | 13.80 | 1.79 |
| **BNB** | 60.37% | 164 | 38.7% | 65.4% | `$+31.91` | 62.10% | 6.06 | 18.39 | 12.51 | 2.07 |
| **BTC** | 62.99% | 154 | 80.0% | 58.9% | `$+34.70` | 69.90% | 6.11 | 17.16 | 12.09 | 2.10 |
| **DOGE** | 53.07% | 179 | 51.6% | 53.4% | `$+26.84` | 117.62% | 3.22 | 9.73 | 5.56 | 1.46 |
| **DOT** | 62.32% | 138 | 61.9% | 62.4% | `$+43.12` | 71.35% | 5.55 | 15.43 | 14.72 | 2.16 |
| **ETH** | 69.23% | 195 | 71.0% | 68.8% | `$+75.90` | 73.26% | 9.60 | 29.57 | 25.23 | 3.09 |
| **HBAR** | 58.44% | 154 | 73.3% | 56.8% | `$+25.17` | 91.81% | 3.31 | 8.39 | 6.67 | 1.50 |
| **LINK** | 64.97% | 177 | 69.0% | 64.2% | `$+58.12` | 62.83% | 7.33 | 18.38 | 22.53 | 2.35 |
| **NEAR** | 56.55% | 145 | 63.6% | 55.3% | `$+24.21` | 76.94% | 2.70 | 6.26 | 7.66 | 1.46 |
| **ONDO** | 69.77% | 129 | 63.0% | 71.6% | `$+56.88` | 59.12% | 7.45 | 16.08 | 23.43 | 2.75 |
| **POL** | 59.85% | 137 | 56.0% | 60.7% | `$+31.21` | 77.50% | 4.12 | 10.30 | 9.81 | 1.78 |
| **SOL** | 56.99% | 186 | 74.2% | 53.5% | `$+31.02` | 118.35% | 3.78 | 10.78 | 6.38 | 1.55 |
| **SUI** | 62.11% | 161 | 65.6% | 61.2% | `$+55.72` | 94.38% | 5.99 | 13.06 | 14.38 | 2.15 |
| **TAO** | 67.70% | 161 | 76.0% | 66.2% | `$+50.40` | 95.42% | 5.14 | 8.47 | 12.86 | 1.89 |
| **TON** | 51.98% | 177 | 39.3% | 54.4% | `$+16.01` | 68.79% | 2.29 | 6.41 | 5.67 | 1.29 |
| **TRX** | 60.33% | 121 | 65.4% | 59.0% | `$+6.76` | 33.32% | 2.57 | 5.92 | 4.94 | 1.40 |
| **XRP** | 59.30% | 199 | 61.1% | 58.9% | `$+49.27` | 90.43% | 5.87 | 21.13 | 13.27 | 1.95 |

## ⛓️ Daftar Fitur Aktif dalam Model

Total terdapat **32 fitur aktif** yang digunakan oleh LightGBM entry, LSTM Soft Confirmation, dan Exit Guardian v3:

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
14. `dist_liq_20x_long`
15. `cvd_momentum_adv`
16. `Sell_Liq`
17. `cvd_slope_h4`
18. `ema_21_slope_h4`
19. `etf_total_change_usd`
20. `ema_50_h1`
21. `etf_gbtc_change_usd`
22. `h4_trend`
23. `log_ret_20`
24. `whale_retail_divergence`
25. `dist_liq_20x_short`
26. `vol_price_confirm`
27. `MSB_BOS`
28. `ema_50_slope_h4`
29. `cvd`
30. `ofi_acceleration`
31. `cvd_div_h4`
32. `hmm_regime_enc`

</details>
