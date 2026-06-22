# IC Test LGBM — ic32_hybrid_v1_ic
*2026-06-09 00:20 | cutoff=2026-04-01*

**Rows**: 785,185 | **Effective N**: 32,716 (N/24) | **Features tested**: 35

**Thresholds**: standalone>=0.02, t-stat>=2.0, marginal>=0.01

## Summary

| Verdict | Count | Artinya |
|---------|-------|--------|
| **KEEP** | 24 | Standalone dan marginal IC lolos — masuk model |
| **REDUNDANT** | 4 | Standalone lolos tapi marginal kecil — duplikasi sinyal |
| **WEAK** | 7 | Standalone gagal tapi marginal lolos — suppressor variable |
| **DROP** | 0 | Tidak ada sinyal — buang |

## KEEP Features (24)

`dist_from_8h_high`, `rsi_6`, `swing_momentum`, `rsi_h4`, `stochrsi_k`, `dist_liq_50x_long`, `trend_accel_4h`, `rsi_slope_h4`, `Fib_786`, `stochrsi_d`, `ofi_h4_delta`, `dist_liq_50x_short`, `Buy_Liq`, `dist_liq_20x_long`, `cvd_momentum_adv`, `Sell_Liq`, `cvd_slope_h4`, `ema_21_slope_h4`, `etf_total_change_usd`, `ema_50_h1`, `etf_gbtc_change_usd`, `h4_trend`, `log_ret_20`, `whale_retail_divergence`

## KEEP (24)

| Feature | Standalone IC | t-stat | Marginal IC | Verdict |
|---------|:------------:|:------:|:-----------:|:-------:|
| `dist_from_8h_high` | -0.1040 | -18.91 | -0.1040 | **KEEP** |
| `rsi_6` | -0.0966 | -17.56 | -0.0388 | **KEEP** |
| `swing_momentum` | -0.0784 | -14.22 | -0.0260 | **KEEP** |
| `rsi_h4` | -0.0779 | -14.12 | -0.0105 | **KEEP** |
| `stochrsi_k` | -0.0717 | -13.00 | +0.0410 | **KEEP** |
| `dist_liq_50x_long` | -0.0698 | -12.66 | -0.0448 | **KEEP** |
| `trend_accel_4h` | -0.0687 | -12.45 | -0.0139 | **KEEP** |
| `rsi_slope_h4` | -0.0620 | -11.23 | +0.0248 | **KEEP** |
| `Fib_786` | +0.0592 | 10.72 | +0.0111 | **KEEP** |
| `stochrsi_d` | -0.0542 | -9.81 | +0.0219 | **KEEP** |
| `ofi_h4_delta` | +0.0530 | 9.60 | +0.0599 | **KEEP** |
| `dist_liq_50x_short` | +0.0500 | 9.05 | +0.0175 | **KEEP** |
| `Buy_Liq` | -0.0479 | -8.67 | -0.0141 | **KEEP** |
| `dist_liq_20x_long` | -0.0455 | -8.25 | +0.0118 | **KEEP** |
| `cvd_momentum_adv` | -0.0452 | -8.19 | -0.0134 | **KEEP** |
| `Sell_Liq` | +0.0443 | 8.03 | +0.0129 | **KEEP** |
| `cvd_slope_h4` | +0.0378 | 6.85 | +0.0686 | **KEEP** |
| `ema_21_slope_h4` | -0.0365 | -6.61 | +0.0151 | **KEEP** |
| `etf_total_change_usd` | +0.0295 | 5.33 | +0.0324 | **KEEP** |
| `ema_50_h1` | +0.0287 | 5.20 | -0.0162 | **KEEP** |
| `etf_gbtc_change_usd` | +0.0280 | 5.07 | +0.0252 | **KEEP** |
| `h4_trend` | -0.0261 | -4.72 | -0.0115 | **KEEP** |
| `log_ret_20` | -0.0260 | -4.70 | -0.0399 | **KEEP** |
| `whale_retail_divergence` | -0.0244 | -4.41 | -0.0461 | **KEEP** |

## REDUNDANT (4)

| Feature | Standalone IC | t-stat | Marginal IC | Verdict |
|---------|:------------:|:------:|:-----------:|:-------:|
| `Fib_618` | +0.0574 | 10.40 | -0.0037 | **REDUNDANT** |
| `relative_strength_z` | -0.0476 | -8.62 | -0.0074 | **REDUNDANT** |
| `long_short_ratio` | -0.0431 | -7.81 | +0.0051 | **REDUNDANT** |
| `absorption_at_swing` | -0.0294 | -5.32 | -0.0052 | **REDUNDANT** |

## WEAK (7)

| Feature | Standalone IC | t-stat | Marginal IC | Verdict |
|---------|:------------:|:------:|:-----------:|:-------:|
| `dist_liq_20x_short` | +0.0193 | 3.49 | -0.0190 | **WEAK** |
| `vol_price_confirm` | -0.0169 | -3.06 | +0.0180 | **WEAK** |
| `MSB_BOS` | -0.0145 | -2.62 | +0.0300 | **WEAK** |
| `ema_50_slope_h4` | -0.0133 | -2.40 | -0.0287 | **WEAK** |
| `cvd` | -0.0077 | -1.40 | -0.0105 | **WEAK** |
| `ofi_acceleration` | -0.0070 | -1.27 | -0.0107 | **WEAK** |
| `cvd_div_h4` | -0.0061 | -1.10 | -0.0433 | **WEAK** |

