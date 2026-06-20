# Feature Selection Report — 2026-06-15

## Metodologi
- Walk-forward IC (Spearman rank) per tahun per koin
- IC gate: |mean_IC| > 0.02 dan t_stat > 2.0
- Fitur ditest: 95 total
- Koin: 21  |  Periode: 2020-2026-04

## Rekomendasi

### Tambah ke LGBM (NEW, tidak ada di current 27)

| Fitur | IC | t-stat | hit% | Arah |
|-------|-----|--------|------|------|
| `wyckoff_phase` | -0.0534 | -14.5 | 8% | mean-rev |
| `ret_30d` | -0.0462 | -9.2 | 20% | mean-rev |
| `hmm_regime_enc` | +0.0417 | 9.3 | 82% | momentum |
| `ret_14d` | -0.0319 | -6.4 | 18% | mean-rev |
| `ema_21_slope_h4` | +0.0273 | 6.4 | 74% | momentum |
| `rsi_h4` | +0.0271 | 6.9 | 79% | momentum |
| `Sell_Liq` | -0.0265 | -7.2 | 19% | mean-rev |
| `spread_to_volume` | -0.0260 | -6.1 | 23% | mean-rev |
| `price_vs_ema_50_h4` | +0.0255 | 6.0 | 76% | momentum |
| `cvd_div_h4` | -0.0253 | -8.9 | 19% | mean-rev |
| `price_in_range` | +0.0240 | 5.7 | 74% | momentum |
| `ema_50_slope_h4` | +0.0238 | 5.7 | 75% | momentum |
| `stochrsi_k` | +0.0236 | 7.9 | 86% | momentum |
| `atr_zscore_20d` | +0.0234 | 5.8 | 72% | momentum |
| `trend_strength` | +0.0224 | 5.5 | 71% | momentum |
| `VAL` | -0.0214 | -5.1 | 26% | mean-rev |
| `dist_swing_low` | +0.0212 | 4.9 | 68% | momentum |
| `vol_price_confirm` | +0.0210 | 5.9 | 70% | momentum |
| `dist_pdl` | +0.0206 | 5.4 | 68% | momentum |

### Drop dari LGBM (current 27 yang lemah)

| Fitur | IC | t-stat |
|-------|-----|--------|
| `dist_liq_50x_long` | +0.0099 | 2.5 |
| `vol_spike_zscore` | +0.0070 | 2.7 |
| `vol_ratio_20` | +0.0069 | 2.9 |
| `effort_vs_result` | +0.0054 | 2.3 |
| `no_demand` | +0.0053 | 3.3 |
| `ultra_high_vol` | +0.0045 | 2.4 |
| `absorption_z` | +0.0030 | 1.4 |
| `range_expansion_h4` | +0.0025 | 2.4 |
| `vol_accel_3h` | -0.0014 | -1.1 |
| `no_supply` | +0.0004 | 0.3 |

## Tabel Lengkap (top-50 by |IC|)

| # | Fitur | IC | t-stat | IC_IR | hit% | sig% | current? |
|---|-------|----|--------|-------|------|------|----------|
| 1 | `ofi_h4_delta` | +0.1140 | 36.20 | 3.64 | 100% | 98% | YES |
| 2 | `cvd_slope_h4` | +0.1088 | 30.21 | 3.04 | 100% | 98% | YES |
| 3 | `wyckoff_phase` | -0.0534 | -14.49 | 1.46 | 8% | 78% |  |
| 4 | `ret_30d` | -0.0462 | -9.17 | 0.94 | 20% | 69% |  |
| 5 | `hmm_regime_enc` | +0.0417 | 9.25 | 0.93 | 82% | 77% |  |
| 6 | `dist_liq_20x_short` | -0.0360 | -10.06 | 1.01 | 16% | 62% | YES |
| 7 | `dist_liq_50x_short` | -0.0330 | -10.25 | 1.03 | 17% | 63% | YES |
| 8 | `dow_cos` | -0.0328 | -6.04 | 0.61 | 28% | 67% | YES |
| 9 | `ret_14d` | -0.0319 | -6.42 | 0.65 | 18% | 62% |  |
| 10 | `ema_21_slope_h4` | +0.0273 | 6.43 | 0.65 | 74% | 64% |  |
| 11 | `rsi_h4` | +0.0271 | 6.88 | 0.69 | 79% | 59% |  |
| 12 | `stochrsi_d` | +0.0268 | 9.29 | 0.93 | 88% | 62% | YES |
| 13 | `Sell_Liq` | -0.0265 | -7.21 | 0.72 | 19% | 63% |  |
| 14 | `spread_to_volume` | -0.0260 | -6.14 | 0.62 | 23% | 57% |  |
| 15 | `price_vs_ema_50_h4` | +0.0255 | 5.98 | 0.60 | 76% | 62% |  |
| 16 | `cvd_div_h4` | -0.0253 | -8.89 | 0.89 | 19% | 53% |  |
| 17 | `atr_percentile_h1` | +0.0246 | 5.50 | 0.55 | 71% | 65% | YES |
| 18 | `price_in_range` | +0.0240 | 5.67 | 0.57 | 74% | 58% |  |
| 19 | `Buy_Liq` | +0.0239 | 6.18 | 0.62 | 76% | 60% | YES |
| 20 | `ema_50_slope_h4` | +0.0238 | 5.68 | 0.57 | 75% | 60% |  |
| 21 | `stochrsi_k` | +0.0236 | 7.89 | 0.79 | 86% | 61% |  |
| 22 | `atr_zscore_20d` | +0.0234 | 5.78 | 0.58 | 72% | 58% |  |
| 23 | `cvd_momentum_adv` | +0.0234 | 8.00 | 0.80 | 80% | 55% | YES |
| 24 | `trend_strength` | +0.0224 | 5.48 | 0.55 | 71% | 59% |  |
| 25 | `atr_percent_h4` | +0.0222 | 4.47 | 0.45 | 69% | 64% | YES |
| 26 | `trend_accel_4h` | +0.0221 | 7.44 | 0.75 | 80% | 54% | YES |
| 27 | `dist_swing_high` | +0.0219 | 5.34 | 0.54 | 72% | 61% | YES |
| 28 | `VAH` | -0.0215 | -5.31 | 0.53 | 30% | 60% | YES |
| 29 | `VAL` | -0.0214 | -5.06 | 0.51 | 26% | 53% |  |
| 30 | `dist_swing_low` | +0.0212 | 4.89 | 0.49 | 68% | 56% |  |
| 31 | `vol_price_confirm` | +0.0210 | 5.88 | 0.59 | 70% | 56% |  |
| 32 | `dist_pdl` | +0.0206 | 5.35 | 0.54 | 68% | 61% |  |
| 33 | `h4_trend` | +0.0196 | 4.56 | 0.46 | 64% | 58% |  |
| 34 | `dist_from_8h_high` | +0.0190 | 6.20 | 0.62 | 77% | 49% | YES |
| 35 | `relative_strength_z` | +0.0189 | 4.50 | 0.47 | 68% | 59% |  |
| 36 | `funding_rate` | -0.0187 | -4.32 | 0.44 | 39% | 52% | YES |
| 37 | `rsi_6` | +0.0186 | 5.70 | 0.57 | 77% | 56% |  |
| 38 | `MSB_BOS` | +0.0183 | 5.34 | 0.54 | 72% | 48% |  |
| 39 | `POC` | -0.0173 | -4.06 | 0.41 | 33% | 57% |  |
| 40 | `dow_sin` | -0.0171 | -3.30 | 0.33 | 42% | 59% |  |
| 41 | `dist_ema200_h4` | -0.0167 | -4.17 | 0.42 | 34% | 45% |  |
| 42 | `dist_ema200_h1` | -0.0167 | -4.17 | 0.42 | 34% | 45% |  |
| 43 | `whale_retail_divergence` | +0.0160 | 4.87 | 0.49 | 70% | 41% | YES |
| 44 | `log_ret_5` | +0.0154 | 5.18 | 0.52 | 77% | 46% |  |
| 45 | `dist_ema50_h4` | -0.0145 | -4.19 | 0.42 | 32% | 53% |  |
| 46 | `vol_efficiency` | +0.0140 | 3.47 | 0.35 | 63% | 51% |  |
| 47 | `swing_momentum` | +0.0140 | 6.26 | 0.63 | 83% | 39% |  |
| 48 | `dist_pwh` | -0.0132 | -2.33 | 0.23 | 38% | 64% |  |
| 49 | `ret_7d` | -0.0129 | -3.06 | 0.31 | 42% | 45% |  |
| 50 | `log_ret_20` | +0.0123 | 2.99 | 0.30 | 62% | 51% | YES |