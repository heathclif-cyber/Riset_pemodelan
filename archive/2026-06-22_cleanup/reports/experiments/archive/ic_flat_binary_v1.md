# IC Test LGBM — ic_flat_binary_v1
*2026-06-14 16:49 | cutoff=2026-04-01*

**Rows**: 785,185 | **Effective N**: 32,716 (N/24) | **Features tested**: 115

**Thresholds**: standalone>=0.01, t-stat>=1.5, marginal>=0.005

## Summary

| Verdict | Count | Artinya |
|---------|-------|--------|
| **KEEP** | 31 | Standalone dan marginal IC lolos — masuk model |
| **REDUNDANT** | 12 | Standalone lolos tapi marginal kecil — duplikasi sinyal |
| **WEAK** | 32 | Standalone gagal tapi marginal lolos — suppressor variable |
| **DROP** | 40 | Tidak ada sinyal — buang |

## KEEP Features (31)

`ultra_high_vol`, `absorption_z`, `vol_spike_zscore`, `dist_liq_50x_short`, `vol_accel_3h`, `no_supply`, `dist_liq_50x_long`, `no_demand`, `range_expansion_h4`, `effort_vs_result`, `dist_from_8h_high`, `vol_ratio_20`, `dow_cos`, `dist_liq_20x_short`, `spring_upthrust`, `dist_liq_20x_long`, `VAH`, `PDH`, `PDL`, `atr_percent_h4`, `dist_swing_high`, `mae_8`, `time_to_funding_norm`, `mae_12`, `etf_gbtc_change_usd`, `dow_sin`, `SFP_sweep`, `vol_price_confirm`, `POC`, `ema_200_h1`, `funding_rate`

## KEEP (31)

| Feature | Standalone IC | t-stat | Marginal IC | Verdict |
|---------|:------------:|:------:|:-----------:|:-------:|
| `ultra_high_vol` | -0.3114 | -59.27 | -0.3114 | **KEEP** |
| `absorption_z` | -0.0776 | -14.08 | -0.0227 | **KEEP** |
| `vol_spike_zscore` | -0.0772 | -14.00 | -0.0093 | **KEEP** |
| `dist_liq_50x_short` | -0.0671 | -12.16 | -0.0860 | **KEEP** |
| `vol_accel_3h` | -0.0625 | -11.32 | -0.0074 | **KEEP** |
| `no_supply` | -0.0598 | -10.83 | -0.0092 | **KEEP** |
| `dist_liq_50x_long` | -0.0591 | -10.71 | -0.0854 | **KEEP** |
| `no_demand` | -0.0585 | -10.60 | -0.0085 | **KEEP** |
| `range_expansion_h4` | -0.0529 | -9.57 | -0.0128 | **KEEP** |
| `effort_vs_result` | -0.0516 | -9.35 | +0.0132 | **KEEP** |
| `dist_from_8h_high` | +0.0515 | 9.32 | +0.0414 | **KEEP** |
| `vol_ratio_20` | -0.0503 | -9.10 | +0.0966 | **KEEP** |
| `dow_cos` | -0.0473 | -8.57 | -0.0297 | **KEEP** |
| `dist_liq_20x_short` | -0.0321 | -5.80 | +0.0147 | **KEEP** |
| `spring_upthrust` | +0.0298 | 5.39 | +0.0351 | **KEEP** |
| `dist_liq_20x_long` | -0.0267 | -4.83 | +0.0316 | **KEEP** |
| `VAH` | -0.0266 | -4.82 | -0.0143 | **KEEP** |
| `PDH` | +0.0254 | 4.60 | +0.0282 | **KEEP** |
| `PDL` | -0.0231 | -4.18 | -0.0382 | **KEEP** |
| `atr_percent_h4` | -0.0218 | -3.95 | -0.1806 | **KEEP** |
| `dist_swing_high` | +0.0197 | 3.56 | +0.0100 | **KEEP** |
| `mae_8` | -0.0178 | -3.22 | +0.0552 | **KEEP** |
| `time_to_funding_norm` | +0.0171 | 3.10 | +0.0132 | **KEEP** |
| `mae_12` | -0.0159 | -2.87 | +0.0375 | **KEEP** |
| `etf_gbtc_change_usd` | -0.0159 | -2.88 | -0.0065 | **KEEP** |
| `dow_sin` | -0.0153 | -2.76 | -0.0109 | **KEEP** |
| `SFP_sweep` | +0.0135 | 2.44 | +0.0132 | **KEEP** |
| `vol_price_confirm` | -0.0125 | -2.26 | -0.0111 | **KEEP** |
| `POC` | -0.0120 | -2.17 | +0.0149 | **KEEP** |
| `ema_200_h1` | +0.0118 | 2.13 | +0.0072 | **KEEP** |
| `funding_rate` | -0.0102 | -1.85 | -0.0111 | **KEEP** |

## REDUNDANT (12)

| Feature | Standalone IC | t-stat | Marginal IC | Verdict |
|---------|:------------:|:------:|:-----------:|:-------:|
| `vol_regime` | -0.0567 | -10.28 | -0.0048 | **REDUNDANT** |
| `atr_zscore_20d` | -0.0434 | -7.86 | -0.0019 | **REDUNDANT** |
| `atr_percentile_h1` | -0.0401 | -7.25 | +0.0030 | **REDUNDANT** |
| `volume` | -0.0173 | -3.14 | -0.0043 | **REDUNDANT** |
| `sell_volume` | -0.0173 | -3.14 | +0.0000 | **REDUNDANT** |
| `buy_volume` | -0.0172 | -3.12 | +0.0008 | **REDUNDANT** |
| `PWH` | +0.0163 | 2.95 | -0.0039 | **REDUNDANT** |
| `etf_total_change_usd` | -0.0146 | -2.65 | +0.0016 | **REDUNDANT** |
| `dynamic_position_pressure` | -0.0118 | -2.13 | -0.0029 | **REDUNDANT** |
| `ema_200_h4` | +0.0118 | 2.13 | +0.0000 | **REDUNDANT** |
| `time_below_entry_8` | -0.0115 | -2.08 | -0.0023 | **REDUNDANT** |
| `stochrsi_d` | +0.0107 | 1.93 | -0.0023 | **REDUNDANT** |

## WEAK (32)

| Feature | Standalone IC | t-stat | Marginal IC | Verdict |
|---------|:------------:|:------:|:-----------:|:-------:|
| `mfe_12` | -0.0099 | -1.80 | +0.0295 | **WEAK** |
| `vol_efficiency` | -0.0098 | -1.77 | +0.0239 | **WEAK** |
| `time_above_entry_8` | +0.0096 | 1.73 | -0.0156 | **WEAK** |
| `stochrsi_k` | +0.0083 | 1.49 | -0.0093 | **WEAK** |
| `relative_strength_z` | -0.0079 | -1.42 | -0.0061 | **WEAK** |
| `price_in_range` | +0.0078 | 1.42 | -0.0079 | **WEAK** |
| `atr_14_h4` | -0.0076 | -1.38 | -0.0136 | **WEAK** |
| `Buy_Liq` | +0.0075 | 1.36 | +0.0167 | **WEAK** |
| `ema_7_h1` | -0.0070 | -1.26 | +0.0356 | **WEAK** |
| `ofi_momentum_ratio` | -0.0066 | -1.19 | +0.0114 | **WEAK** |
| `rsi_6` | +0.0063 | 1.14 | -0.0057 | **WEAK** |
| `Fib_618` | -0.0057 | -1.03 | -0.0221 | **WEAK** |
| `ema_50_slope_h4` | -0.0054 | -0.98 | +0.0079 | **WEAK** |
| `rsi_h4` | +0.0053 | 0.97 | -0.0182 | **WEAK** |
| `mfe_8` | -0.0048 | -0.87 | +0.0256 | **WEAK** |
| `absorption_at_swing` | -0.0047 | -0.85 | -0.0086 | **WEAK** |
| `Sell_Liq` | -0.0043 | -0.77 | +0.0141 | **WEAK** |
| `bars_since_BOS` | +0.0042 | 0.75 | -0.0509 | **WEAK** |
| `hour_sin` | +0.0042 | 0.76 | +0.0111 | **WEAK** |
| `high` | -0.0040 | -0.73 | +0.0103 | **WEAK** |
| `open` | -0.0039 | -0.70 | -0.0056 | **WEAK** |
| `close` | -0.0039 | -0.70 | -0.0094 | **WEAK** |
| `low` | -0.0037 | -0.67 | -0.0202 | **WEAK** |
| `h4_trend` | -0.0036 | -0.66 | +0.0068 | **WEAK** |
| `log_ret_20` | -0.0029 | -0.52 | -0.0097 | **WEAK** |
| `dist_swing_low` | -0.0028 | -0.51 | +0.0156 | **WEAK** |
| `CHoCH` | +0.0027 | 0.49 | +0.0099 | **WEAK** |
| `ema_21_h1` | -0.0027 | -0.49 | +0.0064 | **WEAK** |
| `Fib_786` | -0.0014 | -0.25 | +0.0121 | **WEAK** |
| `spread_to_volume` | +0.0013 | 0.23 | -0.0101 | **WEAK** |
| `PWL` | +0.0005 | 0.10 | +0.0073 | **WEAK** |
| `whale_retail_divergence` | +0.0001 | 0.01 | +0.0051 | **WEAK** |

## DROP (40)

| Feature | Standalone IC | t-stat | Marginal IC | Verdict |
|---------|:------------:|:------:|:-----------:|:-------:|
| `trend_strength` | -0.0091 | -1.64 | -0.0012 | **DROP** |
| `vwdp_smooth` | -0.0087 | -1.58 | +0.0012 | **DROP** |
| `atr_14_h1` | -0.0074 | -1.33 | -0.0035 | **DROP** |
| `ofi_z_score` | +0.0071 | 1.28 | +0.0004 | **DROP** |
| `ema_7_h4` | -0.0070 | -1.26 | +0.0000 | **DROP** |
| `funding_price_div` | +0.0064 | 1.15 | +0.0040 | **DROP** |
| `wyckoff_phase` | -0.0058 | -1.05 | -0.0001 | **DROP** |
| `cvd_momentum_adv` | +0.0058 | 1.05 | -0.0046 | **DROP** |
| `long_short_ratio` | +0.0057 | 1.04 | +0.0015 | **DROP** |
| `price_accel_1h` | +0.0055 | 1.00 | +0.0022 | **DROP** |
| `market_session` | -0.0054 | -0.98 | -0.0047 | **DROP** |
| `ema_50_h1` | +0.0051 | 0.92 | -0.0048 | **DROP** |
| `ema_50_h4` | +0.0051 | 0.92 | +0.0000 | **DROP** |
| `price_vs_ema_50_h4` | -0.0051 | -0.92 | +0.0000 | **DROP** |
| `log_ret_1` | +0.0048 | 0.87 | -0.0011 | **DROP** |
| `open_interest` | -0.0040 | -0.72 | +0.0008 | **DROP** |
| `rsi_slope_h4` | -0.0040 | -0.73 | +0.0024 | **DROP** |
| `ofi_h4_delta` | -0.0032 | -0.59 | +0.0019 | **DROP** |
| `volume_delta` | +0.0030 | 0.53 | -0.0001 | **DROP** |
| `rsi_divergence` | -0.0029 | -0.52 | -0.0010 | **DROP** |
| `hidden_divergence` | +0.0028 | 0.51 | +0.0023 | **DROP** |
| `ema_21_h4` | -0.0027 | -0.49 | +0.0000 | **DROP** |
| `MSB_BOS` | +0.0019 | 0.35 | -0.0043 | **DROP** |
| `ema_21_slope_h4` | +0.0019 | 0.35 | +0.0004 | **DROP** |
| `swing_momentum` | +0.0018 | 0.32 | -0.0032 | **DROP** |
| `relative_strength_momentum` | -0.0017 | -0.31 | +0.0044 | **DROP** |
| `log_ret_5` | +0.0015 | 0.26 | +0.0010 | **DROP** |
| `cvd` | +0.0011 | 0.20 | +0.0040 | **DROP** |
| `ofi_acceleration` | -0.0011 | -0.20 | -0.0023 | **DROP** |
| `vwdp` | +0.0008 | 0.14 | -0.0008 | **DROP** |
| `ofi_raw` | +0.0007 | 0.13 | +0.0024 | **DROP** |
| `VAL` | +0.0003 | 0.05 | +0.0027 | **DROP** |
| `hour_cos` | +0.0003 | 0.06 | -0.0034 | **DROP** |
| `cvd_div_h4` | -0.0002 | -0.03 | -0.0018 | **DROP** |
| `cvd_slope_h4` | -0.0002 | -0.04 | +0.0020 | **DROP** |
| `FVG_up` | +0.0000 | 0.00 | +0.0000 | **DROP** |
| `FVG_down` | +0.0000 | 0.00 | +0.0000 | **DROP** |
| `btc_dominance` | +0.0000 | 0.00 | +0.0000 | **DROP** |
| `fear_greed` | +0.0000 | 0.00 | +0.0000 | **DROP** |
| `trend_accel_4h` | +0.0000 | 0.00 | -0.0046 | **DROP** |

