# IC Test LGBM — ic_test_lgbm_v1
*2026-06-04 22:11 | cutoff=2025-11-01*

**Rows**: 785,185 | **Effective N**: 32,716 (N/24) | **Features tested**: 90

**Thresholds**: standalone>=0.02, t-stat>=2.0, marginal>=0.01

## Summary

| Verdict | Count | Artinya |
|---------|-------|--------|
| **KEEP** | 22 | Standalone dan marginal IC lolos — masuk model |
| **REDUNDANT** | 22 | Standalone lolos tapi marginal kecil — duplikasi sinyal |
| **WEAK** | 8 | Standalone gagal tapi marginal lolos — suppressor variable |
| **DROP** | 38 | Tidak ada sinyal — buang |

## KEEP Features (22)

`dist_from_8h_high`, `rsi_6`, `swing_momentum`, `rsi_h4`, `stochrsi_k`, `dist_liq_50x_long`, `trend_accel_4h`, `price_in_range`, `rsi_slope_h4`, `dist_swing_low`, `stochrsi_d`, `ofi_h4_delta`, `dist_liq_50x_short`, `log_ret_1`, `relative_strength_z`, `dist_liq_20x_long`, `cvd_momentum_adv`, `cvd_slope_h4`, `ema_21_slope_h4`, `ema_50_h1`, `log_ret_20`, `whale_retail_divergence`

## KEEP (22)

| Feature | Standalone IC | t-stat | Marginal IC | Verdict |
|---------|:------------:|:------:|:-----------:|:-------:|
| `dist_from_8h_high` | -0.1040 | -18.91 | -0.1040 | **KEEP** |
| `rsi_6` | -0.0966 | -17.56 | -0.0388 | **KEEP** |
| `swing_momentum` | -0.0784 | -14.22 | -0.0310 | **KEEP** |
| `rsi_h4` | -0.0779 | -14.12 | -0.0114 | **KEEP** |
| `stochrsi_k` | -0.0717 | -13.00 | +0.0410 | **KEEP** |
| `dist_liq_50x_long` | -0.0698 | -12.66 | -0.0160 | **KEEP** |
| `trend_accel_4h` | -0.0687 | -12.45 | -0.0143 | **KEEP** |
| `price_in_range` | -0.0654 | -11.86 | -0.0249 | **KEEP** |
| `rsi_slope_h4` | -0.0620 | -11.23 | +0.0228 | **KEEP** |
| `dist_swing_low` | -0.0605 | -10.97 | -0.0125 | **KEEP** |
| `stochrsi_d` | -0.0542 | -9.81 | +0.0184 | **KEEP** |
| `ofi_h4_delta` | +0.0530 | 9.60 | +0.0599 | **KEEP** |
| `dist_liq_50x_short` | +0.0500 | 9.05 | +0.0354 | **KEEP** |
| `log_ret_1` | -0.0480 | -8.69 | -0.0150 | **KEEP** |
| `relative_strength_z` | -0.0476 | -8.62 | -0.0103 | **KEEP** |
| `dist_liq_20x_long` | -0.0455 | -8.25 | +0.0196 | **KEEP** |
| `cvd_momentum_adv` | -0.0452 | -8.19 | -0.0127 | **KEEP** |
| `cvd_slope_h4` | +0.0378 | 6.85 | +0.0686 | **KEEP** |
| `ema_21_slope_h4` | -0.0365 | -6.61 | +0.0143 | **KEEP** |
| `ema_50_h1` | +0.0287 | 5.20 | -0.0299 | **KEEP** |
| `log_ret_20` | -0.0260 | -4.70 | -0.0205 | **KEEP** |
| `whale_retail_divergence` | -0.0244 | -4.41 | -0.0461 | **KEEP** |

## REDUNDANT (22)

| Feature | Standalone IC | t-stat | Marginal IC | Verdict |
|---------|:------------:|:------:|:-----------:|:-------:|
| `ema_7_h1` | +0.0923 | 16.76 | -0.0058 | **REDUNDANT** |
| `ema_7_h4` | +0.0923 | 16.76 | +0.0000 | **REDUNDANT** |
| `log_ret_5` | -0.0807 | -14.65 | -0.0070 | **REDUNDANT** |
| `VAH` | +0.0618 | 11.20 | +0.0034 | **REDUNDANT** |
| `POC` | +0.0613 | 11.11 | -0.0000 | **REDUNDANT** |
| `ema_21_h1` | +0.0608 | 11.02 | +0.0073 | **REDUNDANT** |
| `Fib_786` | +0.0592 | 10.72 | +0.0017 | **REDUNDANT** |
| `VAL` | +0.0574 | 10.41 | -0.0024 | **REDUNDANT** |
| `Fib_618` | +0.0574 | 10.40 | -0.0094 | **REDUNDANT** |
| `dist_swing_high` | -0.0557 | -10.09 | -0.0078 | **REDUNDANT** |
| `Buy_Liq` | -0.0479 | -8.67 | -0.0023 | **REDUNDANT** |
| `Sell_Liq` | +0.0443 | 8.03 | +0.0057 | **REDUNDANT** |
| `long_short_ratio` | -0.0431 | -7.81 | +0.0072 | **REDUNDANT** |
| `relative_strength_momentum` | -0.0418 | -7.57 | +0.0006 | **REDUNDANT** |
| `volume_delta` | -0.0416 | -7.53 | -0.0007 | **REDUNDANT** |
| `PDL` | +0.0298 | 5.40 | -0.0008 | **REDUNDANT** |
| `absorption_at_swing` | -0.0294 | -5.32 | -0.0099 | **REDUNDANT** |
| `price_vs_ema_50_h4` | -0.0287 | -5.20 | +0.0000 | **REDUNDANT** |
| `PDH` | +0.0252 | 4.56 | +0.0026 | **REDUNDANT** |
| `ofi_z_score` | -0.0234 | -4.23 | +0.0030 | **REDUNDANT** |
| `vwdp` | -0.0233 | -4.22 | -0.0019 | **REDUNDANT** |
| `ofi_raw` | -0.0217 | -3.92 | -0.0003 | **REDUNDANT** |

## WEAK (8)

| Feature | Standalone IC | t-stat | Marginal IC | Verdict |
|---------|:------------:|:------:|:-----------:|:-------:|
| `dist_liq_20x_short` | +0.0193 | 3.49 | -0.0276 | **WEAK** |
| `ema_50_slope_h4` | -0.0133 | -2.40 | -0.0348 | **WEAK** |
| `cvd` | -0.0077 | -1.40 | -0.0111 | **WEAK** |
| `ofi_acceleration` | -0.0070 | -1.27 | -0.0106 | **WEAK** |
| `dow_cos` | -0.0065 | -1.17 | -0.0106 | **WEAK** |
| `cvd_div_h4` | -0.0061 | -1.10 | -0.0433 | **WEAK** |
| `trend_strength` | -0.0050 | -0.91 | +0.0135 | **WEAK** |
| `atr_14_h1` | +0.0011 | 0.19 | +0.0137 | **WEAK** |

## DROP (38)

| Feature | Standalone IC | t-stat | Marginal IC | Verdict |
|---------|:------------:|:------:|:-----------:|:-------:|
| `vol_price_confirm` | -0.0169 | -3.06 | +0.0049 | **DROP** |
| `atr_percentile_h1` | +0.0127 | 2.30 | +0.0043 | **DROP** |
| `atr_zscore_20d` | +0.0121 | 2.18 | -0.0009 | **DROP** |
| `atr_percent_h4` | +0.0119 | 2.16 | +0.0007 | **DROP** |
| `price_accel_1h` | -0.0103 | -1.86 | -0.0088 | **DROP** |
| `ofi_momentum_ratio` | +0.0095 | 1.72 | +0.0036 | **DROP** |
| `vwdp_smooth` | -0.0074 | -1.34 | +0.0063 | **DROP** |
| `open_interest` | +0.0064 | 1.15 | +0.0003 | **DROP** |
| `vol_ratio_20` | +0.0064 | 1.16 | +0.0070 | **DROP** |
| `vol_regime` | +0.0064 | 1.16 | -0.0004 | **DROP** |
| `market_session` | +0.0064 | 1.16 | +0.0021 | **DROP** |
| `vol_spike_zscore` | +0.0055 | 0.99 | +0.0009 | **DROP** |
| `effort_vs_result` | +0.0049 | 0.88 | -0.0013 | **DROP** |
| `ema_200_h1` | +0.0048 | 0.87 | -0.0021 | **DROP** |
| `hour_sin` | -0.0048 | -0.86 | +0.0016 | **DROP** |
| `ema_200_h4` | +0.0048 | 0.87 | +0.0000 | **DROP** |
| `funding_price_div` | +0.0046 | 0.83 | +0.0013 | **DROP** |
| `PWL` | +0.0032 | 0.58 | -0.0025 | **DROP** |
| `range_expansion_h4` | +0.0027 | 0.50 | +0.0010 | **DROP** |
| `funding_rate` | -0.0022 | -0.40 | -0.0074 | **DROP** |
| `time_to_funding_norm` | +0.0021 | 0.38 | +0.0032 | **DROP** |
| `absorption_z` | -0.0019 | -0.34 | -0.0060 | **DROP** |
| `low` | -0.0019 | -0.34 | -0.0067 | **DROP** |
| `close` | -0.0019 | -0.34 | -0.0030 | **DROP** |
| `open` | -0.0018 | -0.33 | -0.0054 | **DROP** |
| `high` | -0.0018 | -0.33 | +0.0026 | **DROP** |
| `PWH` | -0.0017 | -0.30 | +0.0032 | **DROP** |
| `btc_dominance` | -0.0016 | -0.28 | -0.0019 | **DROP** |
| `dow_sin` | +0.0014 | 0.25 | +0.0004 | **DROP** |
| `vol_accel_3h` | +0.0012 | 0.22 | +0.0022 | **DROP** |
| `volume` | +0.0009 | 0.16 | -0.0007 | **DROP** |
| `bars_since_BOS` | +0.0008 | 0.14 | +0.0040 | **DROP** |
| `fear_greed` | -0.0006 | -0.10 | +0.0015 | **DROP** |
| `hour_cos` | -0.0005 | -0.09 | +0.0028 | **DROP** |
| `atr_14_h4` | +0.0005 | 0.09 | -0.0021 | **DROP** |
| `spread_to_volume` | +0.0002 | 0.04 | -0.0059 | **DROP** |
| `buy_volume` | -0.0002 | -0.04 | -0.0043 | **DROP** |
| `vol_efficiency` | -0.0001 | -0.02 | +0.0028 | **DROP** |

