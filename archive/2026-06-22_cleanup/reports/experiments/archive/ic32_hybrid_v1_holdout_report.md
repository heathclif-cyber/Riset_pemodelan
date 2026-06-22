# 📊 Holdout Backtest Report: `ic32_hybrid_v1`

**Tanggal Pembuatan**: 2026-06-09 00:16:42 UTC
**Model Run ID**: `ic32_hybrid_v1`
**Periode Pengujian (Temporal OOS)**: `2025-11-01 - 2026-04-01`

> [!NOTE]
> **Ringkasan Portofolio Eksekutif**:
> *   **Total Net Profit**: **$+782.03 USD** (ROI Portofolio: **+37.24%**)
> *   **Rata-rata Win Rate**: **60.46%** | Total Trades: **3,435**
> *   **Rata-rata Max Drawdown (5x)**: **77.92%**
> *   **Risk-Adjusted**: Sharpe: **4.78** | Sortino: **12.70** | Calmar: **12.84** | Profit Factor: **1.84**

## 📈 Performa Scorecard Portofolio

| Metrik Utama | Nilai Portofolio | Catatan |
|:---|:---:|:---|
| **Total Net Profit ($)** | `$+782.03` | Akumulasi keuntungan bersih 5x leverage |
| **Portfolio ROI (%)** | `+37.24%` | ROI berdasarkan kapital portofolio $100/koin |
| **Overall Win Rate** | `60.46%` | Rasio kemenangan rata-rata seluruh aset |
| **Total Trades** | `3,435` | Jumlah total posisi yang dieksekusi |
| **Rata-rata Trade / Bulan** | `33.2` | Rata-rata frekuensi trade bulanan portofolio |
| **Rata-rata Trade / Hari** | `1.09` | Rata-rata frekuensi trade harian portofolio |
| **Max Drawdown (5x)** | `77.92%` | Rata-rata penurunan terdalam portofolio |
| **Sharpe Ratio** | `4.78` | Efisiensi profit terhadap volatilitas portofolio |
| **Sortino Ratio** | `12.70` | Efisiensi profit terhadap downside deviation |
| **Calmar Ratio** | `12.84` | Rasio return tahunan terhadap drawdown |
| **Profit Factor** | `1.84` | Rasio gross profit dibagi gross loss |
| **Max Consecutive Loss** | `11` trades | Streak kekalahan beruntun terpanjang |
| **Worst Single Trade PnL** | `-28.20%` | Kerugian terdalam dalam satu trade tunggal |
| **95% Trades Loss Under** | `11.20%` | Nilai risiko (VaR P95) kerugian maksimal |

## ↕️ Analisis Arah Signal (LONG vs SHORT)

| Arah Posisi | Jumlah Trade | Distribusi | Menang | Kalah | Win Rate | PnL Bersih ($) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **LONG** | 572 | 16.7% | 372 | 200 | 65.03% | +195.48 |
| **SHORT** | 2,863 | 83.3% | 1,706 | 1,157 | 59.59% | +586.54 |

### Rincian Rata-rata Profitabilitas per Trade

| Tipe Trade | PnL Rata-rata ($) | PnL Rata-rata (%) |
|:---|:---:|:---:|
| **Trade Menang (Wins)** | `$+0.8390` | `+8.39%` |
| **Trade Kalah (Losses)** | `$-0.7084` | `-7.08%` |

## 📅 Scorecard Bulanan Portofolio

| Bulan | Total Trades | Wins | Losses | Win Rate | Net PnL ($) |
|:---|:---:|:---:|:---:|:---:|:---:|
| 2025-11 | 729 | 441 | 288 | 60.49% | $+161.64 |
| 2025-12 | 669 | 398 | 271 | 59.49% | $+177.79 |
| 2026-01 | 931 | 582 | 349 | 62.51% | $+211.71 |
| 2026-02 | 541 | 345 | 196 | 63.77% | $+168.99 |
| 2026-03 | 565 | 312 | 253 | 55.22% | $+61.90 |

## 🚪 Distribusi Alasan Keluar Posisi (Exit Reasons)

| Alasan Exit | Jumlah | Persentase | Wins | Losses | Win Rate | PnL Bersih ($) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| `guardian_exit` | 2,751 | 80.1% | 2,006 | 745 | 72.92% | $+1,235.79 |
| `sl_hit` | 607 | 17.7% | 1 | 606 | 0.16% | $-468.75 |
| `time_exit` | 77 | 2.2% | 71 | 6 | 92.21% | $+14.99 |

## 🪙 Scorecard Per Koin (Detailed Assets)

| Token | Win Rate | Trades | LONG WR | SHORT WR | Net PnL ($) | Max DD | Sharpe | Sortino | Calmar | PF |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1000PEPE** | 60.37% | 164 | 54.5% | 61.8% | `$+25.37` | 142.22% | 2.96 | 5.69 | 4.34 | 1.44 |
| **1000SHIB** | 54.37% | 160 | 59.4% | 53.1% | `$+27.82` | 87.59% | 3.80 | 11.15 | 7.73 | 1.62 |
| **ADA** | 60.57% | 175 | 64.0% | 60.0% | `$+41.93` | 89.56% | 4.94 | 14.65 | 11.40 | 1.79 |
| **ARB** | 60.12% | 163 | 61.1% | 59.8% | `$+37.74` | 128.68% | 4.08 | 9.36 | 7.14 | 1.69 |
| **AVAX** | 51.85% | 189 | 55.2% | 51.2% | `$+22.10` | 98.93% | 2.78 | 5.98 | 5.44 | 1.36 |
| **BNB** | 60.61% | 165 | 50.0% | 62.4% | `$+31.51` | 45.98% | 6.00 | 16.68 | 16.69 | 2.03 |
| **BTC** | 65.14% | 175 | 81.2% | 61.5% | `$+40.24` | 58.50% | 6.72 | 17.57 | 16.75 | 2.16 |
| **DOGE** | 59.76% | 164 | 76.9% | 56.5% | `$+35.17` | 72.65% | 4.37 | 14.45 | 11.79 | 1.71 |
| **DOT** | 58.78% | 148 | 52.4% | 59.8% | `$+27.88` | 83.31% | 3.50 | 9.63 | 8.15 | 1.61 |
| **ETH** | 68.23% | 192 | 80.7% | 65.8% | `$+70.33` | 65.41% | 8.78 | 25.97 | 26.18 | 2.88 |
| **HBAR** | 59.73% | 149 | 66.7% | 58.8% | `$+24.54` | 76.97% | 3.22 | 8.12 | 7.76 | 1.49 |
| **LINK** | 63.79% | 174 | 65.4% | 63.5% | `$+53.32` | 58.60% | 6.54 | 16.83 | 22.16 | 2.16 |
| **NEAR** | 55.10% | 147 | 56.5% | 54.8% | `$+20.91` | 76.18% | 2.39 | 5.41 | 6.68 | 1.39 |
| **ONDO** | 67.36% | 144 | 67.9% | 67.2% | `$+59.73` | 44.92% | 7.30 | 17.65 | 32.38 | 2.56 |
| **POL** | 62.00% | 150 | 70.8% | 60.3% | `$+35.57` | 84.79% | 4.46 | 11.65 | 10.21 | 1.85 |
| **SOL** | 58.03% | 193 | 75.9% | 54.9% | `$+38.33` | 95.28% | 4.45 | 13.22 | 9.80 | 1.66 |
| **SUI** | 61.49% | 161 | 71.4% | 59.4% | `$+60.80` | 72.02% | 6.57 | 14.56 | 20.56 | 2.32 |
| **TAO** | 68.21% | 151 | 73.9% | 67.2% | `$+50.93` | 88.29% | 5.56 | 9.74 | 14.05 | 2.04 |
| **TON** | 55.56% | 162 | 41.7% | 58.0% | `$+26.57` | 53.66% | 3.90 | 11.12 | 12.06 | 1.59 |
| **TRX** | 57.85% | 121 | 66.7% | 54.9% | `$+6.07` | 36.66% | 2.30 | 5.61 | 4.03 | 1.35 |
| **XRP** | 60.64% | 188 | 66.7% | 59.5% | `$+45.16` | 76.18% | 5.72 | 21.60 | 14.43 | 1.96 |

## ⛓️ Daftar Fitur Aktif dalam Model

Total terdapat **36 fitur aktif** yang digunakan oleh LightGBM entry, LSTM Soft Confirmation, dan Exit Guardian v3:

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
34. `absorption_at_swing`
35. `etf_gbtc_change_usd`
36. `etf_total_change_usd`

</details>
