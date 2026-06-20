# 🔬 4-Stage Non-Linear Feature Selection — `lgbm_v2_nonlinear`

*2026-06-08 20:42 | cutoff=2026-04-01*

| Stage | Method | Threshold | Input | Output |
|:---|:---|:---:|:---:|:---:|
| 1 | Spearman IC + Mutual Info | IC>=0.015 OR MI>=0.008 | 106 | 60 |
| 2 | Gram-Schmidt Marginal IC | >=0.008 | 60 | 32 |
| 3 | OOF MDA Permutation | F1 drop>=0.0002 | 32 | 18 |
| 4 | SHAP TreeExplainer | (Ranking) | 18 | 19 |

**Final feature count: 19**

---

## Final Feature List (SHAP-ranked)

| Rank | Feature | Standalone IC | Mutual Info | Marginal IC | MDA Drop | SHAP |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|
| 1 | `dist_liq_50x_short` | +0.0500 | 0.0182 | +0.0120 | +0.00092 | 0.285561 |
| 2 | `vol_price_confirm` | -0.0169 | 0.0413 | +0.0130 | +0.04883 | 0.246539 |
| 3 | `dist_from_8h_high` | -0.1040 | 0.0261 | -0.1040 | +0.01271 | 0.223648 |
| 4 | `dist_swing_low` | -0.0605 | 0.0178 | -0.0082 | +0.00929 | 0.194287 |
| 5 | `absorption_at_swing` | -0.0294 | 0.0344 | -0.0125 | +0.01205 | 0.171194 |
| 6 | `ema_7_h1` | +0.0923 | 0.0432 | -0.0152 | +0.00835 | 0.170375 |
| 7 | `Sell_Liq` | +0.0443 | 0.0177 | +0.0085 | +0.01019 | 0.104677 |
| 8 | `dist_liq_20x_short` | +0.0193 | 0.0057 | -0.0175 | +0.00545 | 0.097111 |
| 9 | `cvd_slope_h4` | +0.0378 | 0.0022 | +0.0686 | +0.00646 | 0.080832 |
| 10 | `ofi_h4_delta` | +0.0530 | 0.0000 | +0.0599 | +0.00400 | 0.071879 |
| 11 | `rsi_h4` | -0.0779 | 0.0193 | -0.0116 | +0.00262 | 0.059645 |
| 12 | `log_ret_1` | -0.0480 | 0.0007 | -0.0141 | +0.00365 | 0.047843 |
| 13 | `dist_swing_high` | -0.0557 | 0.0183 | -0.0088 | +0.00381 | 0.042207 |
| 14 | `etf_gbtc_change_usd` | +0.0280 | 0.0052 | +0.0256 | +0.00180 | 0.034061 |
| 15 | `rsi_slope_h4` | -0.0620 | 0.0000 | +0.0214 | +0.00110 | 0.031910 |
| 16 | `ema_21_slope_h4` | -0.0365 | 0.0027 | -0.0122 | +0.00149 | 0.024344 |
| 17 | `ema_50_h1` | +0.0287 | 0.0000 | -0.0258 | +0.00172 | 0.020912 |
| 18 | `etf_total_change_usd` | +0.0295 | 0.0053 | +0.0328 | +0.00104 | 0.019250 |
| - | `hmm_regime_enc` | — | — | — | — | (added manually) |

---

## Features Dropped per Stage

### Stage 1 Drop (46 fitur) — IC+MI too weak
`CHoCH`, `FVG_down`, `FVG_up`, `MSB_BOS`, `PWH`, `PWL`, `SFP_sweep`, `atr_14_h1`, `atr_14_h4`, `atr_percent_h4`, `atr_percentile_h1`, `atr_zscore_20d`, `bars_since_BOS`, `btc_dominance`, `buy_volume`, `coinank_oi_change_24h`, `cvd`, `cvd_div_h4`, `dow_cos`, `dow_sin`, `dynamic_position_pressure`, `ema_200_h1`, `ema_200_h4`, `ema_50_slope_h4`, `fear_greed`, `funding_price_div`, `hidden_divergence`, `hour_cos`, `hour_sin`, `market_session`, `no_demand`, `no_supply`, `ofi_acceleration`, `ofi_momentum_ratio`, `open_interest`, `price_accel_1h`, `range_expansion_h4`, `sell_volume`, `spread_to_volume`, `spring_upthrust`, `time_to_funding_norm`, `trend_strength`, `vol_efficiency`, `volume`, `vwdp_smooth`, `wyckoff_phase`

### Stage 2 Drop (28 fitur) — Redundant
`Buy_Liq`, `Fib_618`, `Fib_786`, `PDH`, `PDL`, `POC`, `VAH`, `VAL`, `absorption_z`, `effort_vs_result`, `ema_21_h4`, `ema_50_h4`, `ema_7_h4`, `long_short_ratio`, `ofi_raw`, `ofi_z_score`, `price_in_range`, `price_vs_ema_50_h4`, `relative_strength_momentum`, `relative_strength_z`, `stochrsi_k`, `ultra_high_vol`, `vol_accel_3h`, `vol_ratio_20`, `vol_regime`, `vol_spike_zscore`, `volume_delta`, `vwdp`

### Stage 3 Drop (14 fitur) — No non-linear contribution
`cvd_momentum_adv`, `dist_liq_20x_long`, `dist_liq_50x_long`, `ema_21_h1`, `funding_rate`, `h4_trend`, `log_ret_20`, `log_ret_5`, `rsi_6`, `rsi_divergence`, `stochrsi_d`, `swing_momentum`, `trend_accel_4h`, `whale_retail_divergence`

