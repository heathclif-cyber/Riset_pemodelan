# IC Test Per-Regime — ic_test_trending_regimes
*2026-06-08 09:16 | cutoff=2026-04-01*

Laporan ini menguji korelasi fitur terhadap **Triple Barrier Continuation Label** (TP=2.0, SL=1.5) khusus untuk regime pasar trending.

**Thresholds**: Standalone IC >= 0.02 | t-stat >= 2.0 | Marginal IC >= 0.01

## Regime: TRENDING_UP

- **Total Rows**: 88,039 | **Effective N**: 3,668
- **Summary**: **KEEP**: 5 | **REDUNDANT**: 1 | **WEAK**: 29 | **DROP**: 67

### KEEP Features

`ofi_h4_delta`, `cvd_slope_h4`, `funding_rate`, `wyckoff_phase`, `atr_zscore_20d`

| Feature | Standalone IC | t-stat | Marginal IC | Verdict |
|---------|:------------:|:------:|:-----------:|:-------:|
| `ofi_h4_delta` | +0.0854 | 5.19 | +0.0854 | **KEEP** |
| `cvd_slope_h4` | +0.0788 | 4.79 | +0.0788 | **KEEP** |
| `funding_rate` | -0.0786 | -4.78 | -0.0188 | **KEEP** |
| `wyckoff_phase` | -0.0375 | -2.27 | -0.0225 | **KEEP** |
| `atr_zscore_20d` | +0.0370 | 2.24 | +0.0382 | **KEEP** |
| `atr_percentile_h1` | +0.0347 | 2.10 | +0.0050 | **REDUNDANT** |
| `stochrsi_d` | +0.0318 | 1.92 | +0.0327 | **WEAK** |
| `dist_liq_20x_short` | -0.0298 | -1.80 | -0.0166 | **WEAK** |
| `cvd_div_h4` | -0.0261 | -1.58 | -0.0887 | **WEAK** |
| `vol_spike_zscore` | +0.0236 | 1.43 | +0.0207 | **WEAK** |
| `price_in_range` | +0.0224 | 1.36 | +0.0243 | **WEAK** |
| `rsi_h4` | +0.0223 | 1.35 | +0.0109 | **WEAK** |
| `ema_7_h1` | -0.0221 | -1.34 | -0.0156 | **WEAK** |
| `dist_swing_high` | +0.0209 | 1.26 | +0.0123 | **WEAK** |
| `VAL` | -0.0200 | -1.21 | -0.0173 | **WEAK** |
| `atr_14_h1` | -0.0184 | -1.11 | -0.0155 | **WEAK** |
| `dow_cos` | -0.0165 | -1.00 | -0.0192 | **WEAK** |
| `swing_momentum` | +0.0161 | 0.98 | +0.0121 | **WEAK** |
| `relative_strength_momentum` | +0.0144 | 0.87 | +0.0103 | **WEAK** |
| `dow_sin` | -0.0139 | -0.84 | -0.0131 | **WEAK** |
| `log_ret_5` | +0.0134 | 0.81 | -0.0146 | **WEAK** |
| `VAH` | -0.0132 | -0.80 | +0.0162 | **WEAK** |
| `cvd_momentum_adv` | +0.0117 | 0.71 | -0.0448 | **WEAK** |
| `sell_volume` | +0.0114 | 0.69 | -0.0273 | **WEAK** |
| `open_interest` | +0.0102 | 0.62 | +0.0148 | **WEAK** |
| `whale_retail_divergence` | +0.0081 | 0.49 | -0.0236 | **WEAK** |
| `trend_strength` | +0.0079 | 0.48 | +0.0100 | **WEAK** |
| `h4_trend` | +0.0068 | 0.41 | -0.0316 | **WEAK** |
| `log_ret_20` | +0.0066 | 0.40 | -0.0547 | **WEAK** |
| `price_accel_1h` | -0.0057 | -0.35 | -0.0152 | **WEAK** |
| `rsi_slope_h4` | +0.0043 | 0.26 | -0.0282 | **WEAK** |
| `ofi_acceleration` | -0.0031 | -0.19 | -0.0234 | **WEAK** |
| `ema_200_h1` | +0.0028 | 0.17 | +0.0137 | **WEAK** |
| `log_ret_1` | -0.0008 | -0.05 | -0.0143 | **WEAK** |
| `PWH` | -0.0004 | -0.03 | +0.0104 | **WEAK** |

---

## Regime: TRENDING_DOWN

- **Total Rows**: 199,127 | **Effective N**: 8,296
- **Summary**: **KEEP**: 8 | **REDUNDANT**: 7 | **WEAK**: 29 | **DROP**: 58

### KEEP Features

`cvd_slope_h4`, `ofi_h4_delta`, `wyckoff_phase`, `stochrsi_d`, `ema_21_slope_h4`, `trend_accel_4h`, `PDH`, `ema_50_h1`

| Feature | Standalone IC | t-stat | Marginal IC | Verdict |
|---------|:------------:|:------:|:-----------:|:-------:|
| `cvd_slope_h4` | +0.0906 | 8.29 | +0.0906 | **KEEP** |
| `ofi_h4_delta` | +0.0855 | 7.81 | +0.0844 | **KEEP** |
| `wyckoff_phase` | -0.0481 | -4.39 | -0.0170 | **KEEP** |
| `stochrsi_d` | +0.0265 | 2.41 | +0.0141 | **KEEP** |
| `ema_21_slope_h4` | +0.0241 | 2.20 | +0.0322 | **KEEP** |
| `trend_accel_4h` | +0.0237 | 2.16 | +0.0103 | **KEEP** |
| `PDH` | -0.0233 | -2.12 | -0.0202 | **KEEP** |
| `ema_50_h1` | -0.0224 | -2.04 | -0.0142 | **KEEP** |
| `ema_21_h1` | -0.0259 | -2.36 | +0.0036 | **REDUNDANT** |
| `ema_21_h4` | -0.0259 | -2.36 | +0.0000 | **REDUNDANT** |
| `rsi_h4` | +0.0251 | 2.29 | -0.0088 | **REDUNDANT** |
| `stochrsi_k` | +0.0249 | 2.27 | +0.0049 | **REDUNDANT** |
| `Sell_Liq` | -0.0228 | -2.08 | -0.0074 | **REDUNDANT** |
| `ema_50_h4` | -0.0224 | -2.04 | +0.0000 | **REDUNDANT** |
| `price_vs_ema_50_h4` | +0.0224 | 2.04 | +0.0000 | **REDUNDANT** |
| `ema_7_h1` | -0.0219 | -2.00 | -0.0138 | **WEAK** |
| `VAH` | -0.0197 | -1.79 | +0.0132 | **WEAK** |
| `Fib_618` | -0.0192 | -1.75 | -0.0122 | **WEAK** |
| `cvd_div_h4` | -0.0192 | -1.75 | -0.0898 | **WEAK** |
| `POC` | -0.0184 | -1.68 | -0.0130 | **WEAK** |
| `price_in_range` | +0.0183 | 1.67 | -0.0186 | **WEAK** |
| `h4_trend` | +0.0181 | 1.65 | -0.0194 | **WEAK** |
| `cvd_momentum_adv` | +0.0178 | 1.63 | -0.0173 | **WEAK** |
| `Fib_786` | -0.0174 | -1.58 | +0.0131 | **WEAK** |
| `swing_momentum` | +0.0159 | 1.44 | +0.0182 | **WEAK** |
| `PWH` | +0.0144 | 1.31 | +0.0175 | **WEAK** |
| `cvd` | -0.0142 | -1.29 | -0.0285 | **WEAK** |
| `relative_strength_z` | +0.0142 | 1.29 | -0.0104 | **WEAK** |
| `dow_sin` | -0.0139 | -1.26 | -0.0151 | **WEAK** |
| `dist_liq_20x_short` | -0.0134 | -1.22 | -0.0110 | **WEAK** |
| `dow_cos` | -0.0128 | -1.17 | -0.0121 | **WEAK** |
| `MSB_BOS` | +0.0127 | 1.15 | -0.0131 | **WEAK** |
| `hour_sin` | -0.0119 | -1.08 | -0.0122 | **WEAK** |
| `whale_retail_divergence` | +0.0119 | 1.08 | -0.0251 | **WEAK** |
| `open_interest` | -0.0106 | -0.97 | -0.0135 | **WEAK** |
| `log_ret_20` | +0.0102 | 0.93 | -0.0599 | **WEAK** |
| `btc_dominance` | +0.0094 | 0.86 | +0.0108 | **WEAK** |
| `rsi_slope_h4` | +0.0079 | 0.72 | -0.0134 | **WEAK** |
| `atr_percentile_h1` | +0.0063 | 0.57 | +0.0183 | **WEAK** |
| `ofi_acceleration` | -0.0051 | -0.46 | -0.0297 | **WEAK** |
| `vol_accel_3h` | +0.0050 | 0.45 | +0.0107 | **WEAK** |
| `PWL` | +0.0023 | 0.21 | -0.0171 | **WEAK** |
| `ema_200_h1` | +0.0015 | 0.14 | +0.0163 | **WEAK** |
| `long_short_ratio` | +0.0003 | 0.02 | -0.0266 | **WEAK** |

---

