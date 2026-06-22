# IC Test LGBM — tb_widyawardhana_fs_v3
*2026-06-14 15:36 | cutoff=2026-04-01*

**Rows**: 785,185 | **Effective N**: 32,716 (N/24) | **Features tested**: 115

**Thresholds**: standalone>=0.02, t-stat>=2.0, marginal>=0.01

## Summary

| Verdict | Count | Artinya |
|---------|-------|--------|
| **KEEP** | 23 | Standalone dan marginal IC lolos — masuk model |
| **REDUNDANT** | 27 | Standalone lolos tapi marginal kecil — duplikasi sinyal |
| **WEAK** | 15 | Standalone gagal tapi marginal lolos — suppressor variable |
| **DROP** | 50 | Tidak ada sinyal — buang |

## KEEP Features (23)

`dist_from_8h_high`, `rsi_6`, `swing_momentum`, `dist_liq_50x_long`, `trend_accel_4h`, `rsi_slope_h4`, `dist_swing_low`, `dist_swing_high`, `stochrsi_d`, `ofi_h4_delta`, `dist_liq_50x_short`, `Buy_Liq`, `dist_liq_20x_long`, `cvd_momentum_adv`, `Sell_Liq`, `long_short_ratio`, `rsi_divergence`, `cvd_slope_h4`, `time_below_entry_8`, `ema_50_h1`, `h4_trend`, `log_ret_20`, `whale_retail_divergence`

## KEEP (23)

| Feature | Standalone IC | t-stat | Marginal IC | Verdict |
|---------|:------------:|:------:|:-----------:|:-------:|
| `dist_from_8h_high` | -0.1040 | -18.91 | -0.1040 | **KEEP** |
| `rsi_6` | -0.0966 | -17.56 | -0.0399 | **KEEP** |
| `swing_momentum` | -0.0784 | -14.22 | -0.0240 | **KEEP** |
| `dist_liq_50x_long` | -0.0698 | -12.66 | -0.0105 | **KEEP** |
| `trend_accel_4h` | -0.0687 | -12.45 | -0.0123 | **KEEP** |
| `rsi_slope_h4` | -0.0620 | -11.23 | +0.0372 | **KEEP** |
| `dist_swing_low` | -0.0605 | -10.97 | -0.0117 | **KEEP** |
| `dist_swing_high` | -0.0557 | -10.09 | -0.0110 | **KEEP** |
| `stochrsi_d` | -0.0542 | -9.81 | +0.0216 | **KEEP** |
| `ofi_h4_delta` | +0.0530 | 9.60 | +0.0599 | **KEEP** |
| `dist_liq_50x_short` | +0.0500 | 9.05 | +0.0192 | **KEEP** |
| `Buy_Liq` | -0.0479 | -8.67 | -0.0104 | **KEEP** |
| `dist_liq_20x_long` | -0.0455 | -8.25 | +0.0327 | **KEEP** |
| `cvd_momentum_adv` | -0.0452 | -8.19 | -0.0194 | **KEEP** |
| `Sell_Liq` | +0.0443 | 8.03 | +0.0156 | **KEEP** |
| `long_short_ratio` | -0.0431 | -7.81 | +0.0119 | **KEEP** |
| `rsi_divergence` | -0.0379 | -6.86 | +0.0112 | **KEEP** |
| `cvd_slope_h4` | +0.0378 | 6.85 | +0.0686 | **KEEP** |
| `time_below_entry_8` | +0.0302 | 5.46 | +0.0330 | **KEEP** |
| `ema_50_h1` | +0.0287 | 5.20 | -0.0163 | **KEEP** |
| `h4_trend` | -0.0261 | -4.72 | -0.0188 | **KEEP** |
| `log_ret_20` | -0.0260 | -4.70 | -0.0335 | **KEEP** |
| `whale_retail_divergence` | -0.0244 | -4.41 | -0.0461 | **KEEP** |

## REDUNDANT (27)

| Feature | Standalone IC | t-stat | Marginal IC | Verdict |
|---------|:------------:|:------:|:-----------:|:-------:|
| `ema_7_h1` | +0.0923 | 16.76 | -0.0011 | **REDUNDANT** |
| `ema_7_h4` | +0.0923 | 16.76 | +0.0000 | **REDUNDANT** |
| `log_ret_5` | -0.0807 | -14.65 | -0.0038 | **REDUNDANT** |
| `rsi_h4` | -0.0779 | -14.12 | -0.0090 | **REDUNDANT** |
| `stochrsi_k` | -0.0717 | -13.00 | -0.0063 | **REDUNDANT** |
| `price_in_range` | -0.0654 | -11.86 | -0.0035 | **REDUNDANT** |
| `VAH` | +0.0618 | 11.20 | +0.0037 | **REDUNDANT** |
| `POC` | +0.0613 | 11.11 | -0.0006 | **REDUNDANT** |
| `ema_21_h1` | +0.0608 | 11.02 | -0.0001 | **REDUNDANT** |
| `ema_21_h4` | +0.0608 | 11.02 | +0.0000 | **REDUNDANT** |
| `Fib_786` | +0.0592 | 10.72 | +0.0042 | **REDUNDANT** |
| `Fib_618` | +0.0574 | 10.40 | -0.0087 | **REDUNDANT** |
| `VAL` | +0.0574 | 10.41 | -0.0023 | **REDUNDANT** |
| `log_ret_1` | -0.0480 | -8.69 | -0.0077 | **REDUNDANT** |
| `relative_strength_z` | -0.0476 | -8.62 | -0.0097 | **REDUNDANT** |
| `relative_strength_momentum` | -0.0418 | -7.57 | +0.0014 | **REDUNDANT** |
| `volume_delta` | -0.0416 | -7.53 | -0.0006 | **REDUNDANT** |
| `ema_21_slope_h4` | -0.0365 | -6.61 | +0.0095 | **REDUNDANT** |
| `time_above_entry_8` | -0.0299 | -5.40 | -0.0014 | **REDUNDANT** |
| `PDL` | +0.0298 | 5.40 | +0.0007 | **REDUNDANT** |
| `absorption_at_swing` | -0.0294 | -5.32 | -0.0024 | **REDUNDANT** |
| `ema_50_h4` | +0.0287 | 5.20 | +0.0000 | **REDUNDANT** |
| `price_vs_ema_50_h4` | -0.0287 | -5.20 | +0.0000 | **REDUNDANT** |
| `PDH` | +0.0252 | 4.56 | +0.0006 | **REDUNDANT** |
| `ofi_z_score` | -0.0234 | -4.23 | +0.0009 | **REDUNDANT** |
| `vwdp` | -0.0233 | -4.22 | -0.0005 | **REDUNDANT** |
| `ofi_raw` | -0.0217 | -3.92 | -0.0030 | **REDUNDANT** |

## WEAK (15)

| Feature | Standalone IC | t-stat | Marginal IC | Verdict |
|---------|:------------:|:------:|:-----------:|:-------:|
| `dist_liq_20x_short` | +0.0193 | 3.49 | -0.0180 | **WEAK** |
| `mae_8` | +0.0189 | 3.41 | -0.0425 | **WEAK** |
| `mfe_8` | -0.0178 | -3.22 | +0.0110 | **WEAK** |
| `vol_price_confirm` | -0.0169 | -3.06 | +0.0135 | **WEAK** |
| `MSB_BOS` | -0.0145 | -2.62 | +0.0422 | **WEAK** |
| `ema_50_slope_h4` | -0.0133 | -2.40 | -0.0267 | **WEAK** |
| `atr_percent_h4` | +0.0119 | 2.16 | -0.0194 | **WEAK** |
| `etf_gbtc_change_usd` | +0.0106 | 1.92 | -0.0116 | **WEAK** |
| `price_accel_1h` | -0.0103 | -1.86 | -0.0154 | **WEAK** |
| `cvd` | -0.0077 | -1.40 | -0.0108 | **WEAK** |
| `mae_12` | -0.0077 | -1.39 | -0.0237 | **WEAK** |
| `ofi_acceleration` | -0.0070 | -1.27 | -0.0113 | **WEAK** |
| `cvd_div_h4` | -0.0061 | -1.10 | -0.0429 | **WEAK** |
| `atr_14_h1` | +0.0011 | 0.19 | +0.0126 | **WEAK** |
| `wyckoff_phase` | -0.0001 | -0.02 | -0.0181 | **WEAK** |

## DROP (50)

| Feature | Standalone IC | t-stat | Marginal IC | Verdict |
|---------|:------------:|:------:|:-----------:|:-------:|
| `etf_total_change_usd` | +0.0136 | 2.47 | +0.0031 | **DROP** |
| `atr_percentile_h1` | +0.0127 | 2.30 | +0.0088 | **DROP** |
| `atr_zscore_20d` | +0.0121 | 2.18 | -0.0009 | **DROP** |
| `ofi_momentum_ratio` | +0.0095 | 1.72 | +0.0036 | **DROP** |
| `vwdp_smooth` | -0.0074 | -1.34 | +0.0058 | **DROP** |
| `dow_cos` | -0.0065 | -1.17 | -0.0096 | **DROP** |
| `open_interest` | +0.0064 | 1.15 | +0.0012 | **DROP** |
| `market_session` | +0.0064 | 1.16 | +0.0020 | **DROP** |
| `vol_ratio_20` | +0.0064 | 1.16 | +0.0047 | **DROP** |
| `vol_regime` | +0.0064 | 1.16 | +0.0003 | **DROP** |
| `dynamic_position_pressure` | +0.0058 | 1.04 | +0.0028 | **DROP** |
| `vol_spike_zscore` | +0.0055 | 0.99 | +0.0009 | **DROP** |
| `mfe_12` | +0.0051 | 0.92 | +0.0066 | **DROP** |
| `trend_strength` | -0.0050 | -0.91 | +0.0061 | **DROP** |
| `effort_vs_result` | +0.0049 | 0.88 | -0.0013 | **DROP** |
| `ema_200_h1` | +0.0048 | 0.87 | +0.0007 | **DROP** |
| `ema_200_h4` | +0.0048 | 0.87 | +0.0000 | **DROP** |
| `hour_sin` | -0.0048 | -0.86 | +0.0013 | **DROP** |
| `funding_price_div` | +0.0046 | 0.83 | +0.0016 | **DROP** |
| `hidden_divergence` | +0.0043 | 0.78 | +0.0021 | **DROP** |
| `PWL` | +0.0032 | 0.58 | -0.0032 | **DROP** |
| `range_expansion_h4` | +0.0027 | 0.50 | +0.0014 | **DROP** |
| `no_demand` | +0.0026 | 0.48 | +0.0033 | **DROP** |
| `funding_rate` | -0.0022 | -0.40 | -0.0082 | **DROP** |
| `time_to_funding_norm` | +0.0021 | 0.38 | +0.0031 | **DROP** |
| `spring_upthrust` | +0.0021 | 0.38 | +0.0087 | **DROP** |
| `low` | -0.0019 | -0.34 | -0.0057 | **DROP** |
| `close` | -0.0019 | -0.34 | -0.0032 | **DROP** |
| `absorption_z` | -0.0019 | -0.34 | -0.0047 | **DROP** |
| `no_supply` | +0.0019 | 0.35 | +0.0025 | **DROP** |
| `open` | -0.0018 | -0.33 | -0.0029 | **DROP** |
| `high` | -0.0018 | -0.33 | +0.0023 | **DROP** |
| `ultra_high_vol` | +0.0018 | 0.33 | -0.0028 | **DROP** |
| `PWH` | -0.0017 | -0.30 | +0.0024 | **DROP** |
| `dow_sin` | +0.0014 | 0.25 | +0.0012 | **DROP** |
| `SFP_sweep` | +0.0012 | 0.22 | -0.0022 | **DROP** |
| `vol_accel_3h` | +0.0012 | 0.22 | +0.0017 | **DROP** |
| `volume` | +0.0009 | 0.16 | -0.0010 | **DROP** |
| `sell_volume` | +0.0009 | 0.16 | +0.0000 | **DROP** |
| `bars_since_BOS` | +0.0008 | 0.14 | +0.0005 | **DROP** |
| `atr_14_h4` | +0.0005 | 0.09 | -0.0018 | **DROP** |
| `hour_cos` | -0.0005 | -0.09 | +0.0053 | **DROP** |
| `CHoCH` | +0.0004 | 0.08 | +0.0009 | **DROP** |
| `buy_volume` | -0.0002 | -0.04 | -0.0042 | **DROP** |
| `spread_to_volume` | +0.0002 | 0.04 | -0.0050 | **DROP** |
| `vol_efficiency` | -0.0001 | -0.02 | +0.0009 | **DROP** |
| `FVG_up` | +0.0000 | 0.00 | +0.0000 | **DROP** |
| `FVG_down` | +0.0000 | 0.00 | +0.0000 | **DROP** |
| `btc_dominance` | +0.0000 | 0.00 | +0.0000 | **DROP** |
| `fear_greed` | +0.0000 | 0.00 | +0.0000 | **DROP** |

