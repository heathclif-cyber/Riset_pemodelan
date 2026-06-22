# Holdout Per-Trade Comparison: ic32 vs TB

**Generated**: 2026-06-17
**Period**: 2026-04-01 - 2026-07-01 (~2.5 bulan, data s/d Jun 13)
**Coins**: 21 | **Granularity**: per trade (bukan per hari)

## 1. Scorecard Agregat

| Metrik | ic32_regime_v1 | tb_lstm_cond | Delta (TB-ic32) |
|--------|---------------:|-------------:|----------------:|
| Total trades | 936 | 1,596 | +660 |
| Win rate % | 62.1 | 54.7 | -7.4 |
| Net PnL | $+207.22 | $+247.05 | $+39.83 |
| PnL/trade | 0.2 | 0.2 | -0.1 |
| Profit factor | 1.96 | 1.30 | -0.65 |
| LONG trades | 348 | 734 | +386 |
| LONG WR % | 66.1 | 54.4 | -11.7 |
| SHORT trades | 588 | 862 | +274 |
| SHORT WR % | 59.7 | 55.0 | -4.7 |
| Avg hold bars | 10.8 | 12.1 | +1.3 |
| Avg modal | $10.00 | $14.46 | $+4.46 |
| SL hit % | 18.8 | 37.6 | +18.8 |
| Guardian exit % | 73.3 | 54.6 | -18.7 |

### Config (frozen)
- **ic32**: LGBM thr 0.69/0.59, conf>=0.59, Guardian clean_v2 exit 0.65, fixed $10
- **TB**: LGBM 36f + HMM Config B + LSTM conditional_momentum + Guardian v2 + DynSize

## 2. Breakdown Per Exit Reason (per trade)

### ic32

| exit_norm | n | wr | net_pnl | avg_pnl | pf | avg_hold |
| --- | --- | --- | --- | --- | --- | --- |
| sl_hit | 176 | 1.70 | -116.22 | -0.66 | 0.00 | 10.68 |
| guardian_exit | 344 | 48.55 | -13.77 | -0.04 | 0.86 | 7.73 |
| time_exit | 74 | 97.30 | 13.00 | 0.18 | 101.02 | 34.36 |
| guardian_momentum_partial | 47 | 100.00 | 14.46 | 0.31 | inf | 34.60 |
| guardian_momentum_exit | 295 | 98.98 | 309.76 | 1.05 | 323.73 | 4.86 |

### TB

| exit_norm | n | wr | net_pnl | avg_pnl | pf | avg_hold |
| --- | --- | --- | --- | --- | --- | --- |
| sl_hit | 600 | 0.67 | -764.80 | -1.27 | 0.00 | 10.47 |
| time_exit | 125 | 60.80 | 14.92 | 0.12 | 4.24 | 33.79 |
| guardian_momentum_partial | 41 | 100.00 | 17.87 | 0.44 | inf | 35.00 |
| guardian_exit | 368 | 80.71 | 213.77 | 0.58 | 6.06 | 8.92 |
| guardian_momentum_exit | 462 | 98.48 | 765.29 | 1.66 | 239.78 | 8.92 |

## 3. Breakdown Per Arah + Trend Alignment

### ic32

| direction | trend_align | n | wr | net_pnl | avg_pnl | pf | avg_hold |
| --- | --- | --- | --- | --- | --- | --- | --- |
| LONG | with | 81 | 59.26 | 9.19 | 0.11 | 1.42 | 10.28 |
| SHORT | with | 148 | 57.43 | 25.84 | 0.17 | 1.77 | 9.28 |
| LONG | counter | 267 | 68.16 | 75.46 | 0.28 | 2.47 | 11.83 |
| SHORT | counter | 440 | 60.45 | 96.74 | 0.22 | 1.88 | 10.86 |

### TB

| direction | trend_align | n | wr | net_pnl | avg_pnl | pf | avg_hold |
| --- | --- | --- | --- | --- | --- | --- | --- |
| LONG | counter | 329 | 56.84 | 18.93 | 0.06 | 1.09 | 12.94 |
| SHORT | with | 465 | 52.04 | 42.23 | 0.09 | 1.19 | 11.38 |
| LONG | with | 405 | 52.35 | 57.75 | 0.14 | 1.27 | 12.59 |
| SHORT | counter | 397 | 58.44 | 128.14 | 0.32 | 1.74 | 11.83 |

## 4. Breakdown Confidence Bucket (per trade)

### ic32

| conf_bucket | n | wr | net_pnl | avg_pnl | pf | avg_hold |
| --- | --- | --- | --- | --- | --- | --- |
| [0.59, 0.65) | 14 | 28.57 | -2.28 | -0.16 | 0.59 | 9.14 |
| nan | 1 | 100.00 | 1.05 | 1.05 | inf | 8.00 |
| [0.65, 0.7) | 103 | 64.08 | 17.12 | 0.17 | 1.82 | 10.41 |
| [0.7, 0.75) | 218 | 57.80 | 23.50 | 0.11 | 1.40 | 10.30 |
| [0.75, 0.8) | 253 | 56.92 | 36.68 | 0.14 | 1.54 | 11.10 |
| [0.8, 1.0) | 347 | 69.16 | 131.16 | 0.38 | 3.06 | 11.18 |

### TB

| conf_bucket | n | wr | net_pnl | avg_pnl | pf | avg_hold |
| --- | --- | --- | --- | --- | --- | --- |
| [0.55, 0.6) | 652 | 51.23 | -30.38 | -0.05 | 0.91 | 11.82 |
| [0.45, 0.5) | 0 | 0.00 | 0.00 | nan | inf | nan |
| [0.5, 0.55) | 0 | 0.00 | 0.00 | nan | inf | nan |
| [0.6, 0.65) | 387 | 51.68 | 31.43 | 0.08 | 1.14 | 12.21 |
| [0.65, 1.0) | 557 | 60.86 | 246.00 | 0.44 | 2.04 | 12.41 |

## 5. Breakdown Hold Bars (losers only, per trade)

### ic32

| hold_bucket | n | wr | net_pnl | avg_pnl | pf | avg_hold |
| --- | --- | --- | --- | --- | --- | --- |
| (5, 10] | 99 | 0.00 | -66.76 | -0.67 | 0.00 | 7.62 |
| (2, 5] | 96 | 0.00 | -62.76 | -0.65 | 0.00 | 3.99 |
| (10, 20] | 68 | 0.00 | -37.59 | -0.55 | 0.00 | 14.57 |
| (0, 2] | 56 | 0.00 | -35.99 | -0.64 | 0.00 | 1.89 |
| (20, 100] | 36 | 0.00 | -13.77 | -0.38 | 0.00 | 26.58 |

### TB

| hold_bucket | n | wr | net_pnl | avg_pnl | pf | avg_hold |
| --- | --- | --- | --- | --- | --- | --- |
| (5, 10] | 189 | 0.00 | -226.99 | -1.20 | 0.00 | 7.93 |
| (2, 5] | 167 | 0.00 | -205.77 | -1.23 | 0.00 | 4.07 |
| (10, 20] | 167 | 0.00 | -189.45 | -1.13 | 0.00 | 14.60 |
| (20, 100] | 118 | 0.00 | -108.96 | -0.92 | 0.00 | 29.62 |
| (0, 2] | 82 | 0.00 | -84.68 | -1.03 | 0.00 | 1.79 |

## 6. Kelemahan Per Trade Pattern (tag aggregation, bukan per hari)

Setiap trade bisa punya beberapa tag. Angka di bawah = jumlah **trade** yang membawa tag tersebut.

### ic32 — weakness tags (sorted by net PnL)

| tag | trades_tagged | loss_trades | wr | net_pnl | avg_pnl |
| --- | --- | --- | --- | --- | --- |
| medium_loss | 209 | 209 | 0.00 | -116.59 | -0.56 |
| sl_loss | 173 | 173 | 0.00 | -116.43 | -0.67 |
| guardian_loss | 180 | 180 | 0.00 | -100.31 | -0.56 |
| large_loss | 81 | 81 | 0.00 | -89.07 | -1.10 |
| early_guardian_cut | 53 | 53 | 0.00 | -28.80 | -0.54 |
| quick_sl | 24 | 24 | 0.00 | -18.02 | -0.75 |
| small_loss | 65 | 65 | 0.00 | -11.21 | -0.17 |
| time_exit_loss | 2 | 2 | 0.00 | -0.13 | -0.07 |
| long_with_trend | 81 | 33 | 59.26 | 9.19 | 0.11 |
| short_counter_trend | 440 | 174 | 60.45 | 96.74 | 0.22 |

### TB — weakness tags (sorted by net PnL)

| tag | trades_tagged | loss_trades | wr | net_pnl | avg_pnl |
| --- | --- | --- | --- | --- | --- |
| sl_loss | 596 | 596 | 0.00 | -765.83 | -1.28 |
| large_loss | 446 | 446 | 0.00 | -695.57 | -1.56 |
| medium_loss | 189 | 189 | 0.00 | -108.90 | -0.58 |
| quick_sl | 60 | 60 | 0.00 | -68.88 | -1.15 |
| guardian_loss | 78 | 78 | 0.00 | -45.43 | -0.58 |
| marginal_entry | 652 | 318 | 51.23 | -30.38 | -0.05 |
| early_guardian_cut | 30 | 30 | 0.00 | -22.04 | -0.73 |
| small_loss | 88 | 88 | 0.00 | -11.38 | -0.13 |
| time_exit_loss | 49 | 49 | 0.00 | -4.60 | -0.09 |
| long_with_trend | 405 | 193 | 52.35 | 57.75 | 0.14 |
| short_counter_trend | 397 | 165 | 58.44 | 128.14 | 0.32 |
| oversized | 1050 | 453 | 56.86 | 269.65 | 0.26 |

## 7. Worst 15 Trades (per trade)

### ic32

| coin | entry_time | direction | confidence | outcome | exit_norm | hold_bars | net_pnl | modal_used | trend_align | weakness_tags |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| NEARUSDT | 2026-06-08 11:00:00+00:00 | SHORT | 0.73 | LOSS | sl_hit | 4 | -2.54 | 10.00 | counter | sl_loss|large_loss|short_counter_trend |
| ONDOUSDT | 2026-05-04 15:00:00+00:00 | SHORT | 0.75 | LOSS | sl_hit | 4 | -2.18 | 10.00 | counter | sl_loss|large_loss|short_counter_trend |
| TONUSDT | 2026-06-09 03:00:00+00:00 | SHORT | 0.83 | GUARDIAN_EXIT | guardian_exit | 2 | -1.74 | 10.00 | with | guardian_loss|early_guardian_cut|large_loss |
| ARBUSDT | 2026-04-10 09:00:00+00:00 | SHORT | 0.86 | LOSS | sl_hit | 12 | -1.58 | 10.00 | counter | sl_loss|large_loss|short_counter_trend |
| DOGEUSDT | 2026-05-15 08:00:00+00:00 | LONG | 0.81 | LOSS | sl_hit | 5 | -1.57 | 10.00 | counter | sl_loss|large_loss |
| ARBUSDT | 2026-04-22 09:00:00+00:00 | SHORT | 0.86 | LOSS | sl_hit | 8 | -1.51 | 10.00 | counter | sl_loss|large_loss|short_counter_trend |
| 1000SHIBUSDT | 2026-06-04 16:00:00+00:00 | LONG | 0.75 | GUARDIAN_EXIT | guardian_exit | 5 | -1.43 | 10.00 | counter | guardian_loss|large_loss |
| ARBUSDT | 2026-06-02 02:00:00+00:00 | LONG | 0.78 | LOSS | sl_hit | 4 | -1.43 | 10.00 | counter | sl_loss|large_loss |
| TONUSDT | 2026-06-12 10:00:00+00:00 | SHORT | 0.67 | LOSS | sl_hit | 5 | -1.38 | 10.00 | counter | sl_loss|large_loss|short_counter_trend |
| TONUSDT | 2026-05-31 21:00:00+00:00 | SHORT | 0.67 | LOSS | sl_hit | 3 | -1.35 | 10.00 | counter | sl_loss|large_loss|short_counter_trend |
| NEARUSDT | 2026-05-14 21:00:00+00:00 | LONG | 0.80 | LOSS | sl_hit | 8 | -1.32 | 10.00 | with | sl_loss|large_loss|long_with_trend |
| SUIUSDT | 2026-06-11 09:00:00+00:00 | SHORT | 0.73 | LOSS | sl_hit | 8 | -1.26 | 10.00 | counter | sl_loss|large_loss|short_counter_trend |
| DOTUSDT | 2026-04-17 12:00:00+00:00 | SHORT | 0.66 | LOSS | sl_hit | 2 | -1.24 | 10.00 | counter | sl_loss|quick_sl|large_loss|short_counter_trend |
| TAOUSDT | 2026-05-29 05:00:00+00:00 | LONG | 0.79 | LOSS | sl_hit | 7 | -1.24 | 10.00 | counter | sl_loss|large_loss |
| POLUSDT | 2026-04-29 22:00:00+00:00 | SHORT | 0.69 | GUARDIAN_EXIT | guardian_exit | 3 | -1.24 | 10.00 | counter | guardian_loss|early_guardian_cut|large_loss|short_counter_trend |

### TB

| coin | entry_time | direction | confidence | outcome | exit_norm | hold_bars | net_pnl | modal_used | trend_align | weakness_tags |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| NEARUSDT | 2026-06-03 21:00:00+00:00 | LONG | 0.64 | LOSS | sl_hit | 4 | -5.80 | 16.00 | with | sl_loss|large_loss|oversized|long_with_trend |
| DOTUSDT | 2026-06-05 09:00:00+00:00 | LONG | 0.59 | LOSS | sl_hit | 9 | -5.35 | 14.16 | counter | sl_loss|large_loss|marginal_entry|oversized |
| ONDOUSDT | 2026-06-04 20:00:00+00:00 | LONG | 0.65 | LOSS | sl_hit | 10 | -4.92 | 16.00 | counter | sl_loss|large_loss|oversized |
| TONUSDT | 2026-06-01 22:00:00+00:00 | LONG | 0.61 | LOSS | sl_hit | 9 | -4.83 | 16.00 | with | sl_loss|large_loss|oversized|long_with_trend |
| SOLUSDT | 2026-06-05 09:00:00+00:00 | LONG | 0.64 | LOSS | sl_hit | 9 | -4.39 | 16.00 | counter | sl_loss|large_loss|oversized |
| 1000PEPEUSDT | 2026-06-04 05:00:00+00:00 | LONG | 0.56 | LOSS | sl_hit | 25 | -4.31 | 10.81 | counter | sl_loss|large_loss|marginal_entry |
| AVAXUSDT | 2026-06-04 05:00:00+00:00 | LONG | 0.59 | LOSS | sl_hit | 21 | -4.25 | 13.66 | counter | sl_loss|large_loss|marginal_entry |
| DOTUSDT | 2026-06-04 06:00:00+00:00 | LONG | 0.57 | LOSS | sl_hit | 24 | -4.20 | 12.40 | counter | sl_loss|large_loss|marginal_entry |
| DOTUSDT | 2026-05-24 00:00:00+00:00 | LONG | 0.61 | LOSS | sl_hit | 21 | -4.20 | 16.00 | with | sl_loss|large_loss|oversized|long_with_trend |
| TAOUSDT | 2026-06-05 10:00:00+00:00 | LONG | 0.68 | LOSS | sl_hit | 8 | -4.16 | 16.00 | counter | sl_loss|large_loss|oversized |
| ONDOUSDT | 2026-05-04 13:00:00+00:00 | SHORT | 0.62 | LOSS | sl_hit | 6 | -4.14 | 16.00 | counter | sl_loss|large_loss|oversized|short_counter_trend |
| NEARUSDT | 2026-06-08 11:00:00+00:00 | SHORT | 0.68 | LOSS | sl_hit | 4 | -4.07 | 16.00 | counter | sl_loss|large_loss|oversized|short_counter_trend |
| SOLUSDT | 2026-06-04 05:00:00+00:00 | LONG | 0.57 | LOSS | sl_hit | 25 | -3.81 | 11.60 | counter | sl_loss|large_loss|marginal_entry |
| DOGEUSDT | 2026-06-04 05:00:00+00:00 | LONG | 0.57 | LOSS | sl_hit | 24 | -3.64 | 12.24 | counter | sl_loss|large_loss|marginal_entry |
| TONUSDT | 2026-05-05 08:00:00+00:00 | SHORT | 0.56 | LOSS | sl_hit | 3 | -3.62 | 10.67 | counter | sl_loss|large_loss|marginal_entry|short_counter_trend |

## 8. Per Coin (top 5 best / worst net PnL)

### ic32 — worst 5 coins

| coin | n | wr | net_pnl | avg_pnl | pf | avg_hold |
| --- | --- | --- | --- | --- | --- | --- |
| SUIUSDT | 66 | 56.06 | 0.53 | 0.01 | 1.03 | 15.77 |
| BTCUSDT | 40 | 47.50 | 0.69 | 0.02 | 1.08 | 10.32 |
| TRXUSDT | 41 | 58.54 | 1.90 | 0.05 | 1.41 | 8.12 |
| HBARUSDT | 40 | 60.00 | 4.21 | 0.11 | 1.45 | 9.50 |
| ADAUSDT | 42 | 54.76 | 5.10 | 0.12 | 1.44 | 12.83 |

### ic32 — best 5 coins

| coin | n | wr | net_pnl | avg_pnl | pf | avg_hold |
| --- | --- | --- | --- | --- | --- | --- |
| ETHUSDT | 37 | 72.97 | 19.30 | 0.52 | 5.07 | 6.32 |
| AVAXUSDT | 53 | 69.81 | 18.37 | 0.35 | 3.17 | 10.13 |
| ONDOUSDT | 38 | 60.53 | 16.18 | 0.43 | 3.28 | 13.16 |
| NEARUSDT | 42 | 54.76 | 16.05 | 0.38 | 2.22 | 9.60 |
| 1000SHIBUSDT | 44 | 72.73 | 15.30 | 0.35 | 2.70 | 6.18 |

### TB — worst 5 coins

| coin | n | wr | net_pnl | avg_pnl | pf | avg_hold |
| --- | --- | --- | --- | --- | --- | --- |
| AVAXUSDT | 81 | 46.91 | -3.25 | -0.04 | 0.94 | 12.68 |
| TAOUSDT | 83 | 51.81 | -3.11 | -0.04 | 0.95 | 10.27 |
| ADAUSDT | 73 | 52.05 | 1.07 | 0.01 | 1.03 | 12.52 |
| SUIUSDT | 77 | 45.45 | 1.15 | 0.01 | 1.02 | 12.87 |
| 1000SHIBUSDT | 74 | 50.00 | 3.30 | 0.04 | 1.08 | 13.27 |

### TB — best 5 coins

| coin | n | wr | net_pnl | avg_pnl | pf | avg_hold |
| --- | --- | --- | --- | --- | --- | --- |
| NEARUSDT | 90 | 64.44 | 47.92 | 0.53 | 2.00 | 7.81 |
| TONUSDT | 85 | 60.00 | 37.20 | 0.44 | 1.76 | 10.22 |
| ONDOUSDT | 86 | 53.49 | 33.94 | 0.39 | 1.68 | 9.48 |
| SOLUSDT | 70 | 54.29 | 16.53 | 0.24 | 1.50 | 13.67 |
| 1000PEPEUSDT | 77 | 61.04 | 16.24 | 0.21 | 1.55 | 13.03 |

## 9. Kesimpulan Kelemahan Masing-masing Model

### ic32_regime_v1

- **Volume lebih rendah** (936 vs 1596 trade) — gate conf 0.59 + thr tinggi memfilter banyak sinyal.
- **SL hit** 176 trade (18.8%) — avg loss $-0.660
- **Guardian exit rugi** 180 trade — keluar dini di chop, avg $-0.557
- **LONG** 348 trade WR 66.1% vs SHORT WR 59.7%
- Kelemahan dominan: **precision tinggi tapi trade count rendah**; rugi terkonsentrasi di SL hit + guardian cut kecil.

### tb_lstm_cond

- **Volume tinggi** (1596 trade) — thr HMM rendah (0.45-0.55) + LSTM boost banyak entry marginal.
- **SL hit** 600 trade (37.6%) — sumber utama loss absolut, avg $-1.275
- **Marginal entry** (conf margin <0.05): 652 trade, net $-30.38
- **Oversized** (modal >1.4x): 1050 trade, net $269.65
- **LONG** 734 trade WR 54.4% — dynsize sering boost modal di TRENDING_UP
- Kelemahan dominan: **banyak trade marginal + SL hit**; dynsize amplifikasi loss saat entry lemah.

## 10. File Output

- `holdout_ic32_trades_apr_jun26.csv` — {ic32_sc['n']:,} baris, 1 baris = 1 trade
- `holdout_tb_lstm_cond_trades_apr_jun26.csv` — {tb_sc['n']:,} baris, 1 baris = 1 trade

*Generated by tools/holdout_ic32_vs_tb_per_trade.py (frozen config, export-only)*