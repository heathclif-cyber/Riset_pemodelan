# 🧪 Simons Feature Selection Pipeline Report
*2026-06-08 12:23 | training_cutoff=2026-04-01*

Laporan ini berisi hasil penyaringan fitur menggunakan pipa 3 tahap: **Mutual Information + Standalone IC**, **Gram-Schmidt Marginal IC**, dan **OOF Mean Decrease Accuracy (MDA)**.

## Model: LGBM_TRENDING_UP

- **Total Fitur Terpilih**: 20 / 102
### Daftar Fitur Terpilih

`ofi_h4_delta`, `cvd_slope_h4`, `cvd_div_h4`, `ema_50_h1`, `stochrsi_d`, `VAH`, `atr_zscore_20d`, `whale_retail_divergence`, `ofi_acceleration`, `vol_accel_3h`, `vol_spike_zscore`, `dist_from_8h_high`, `log_ret_5`, `ema_7_h1`, `ema_21_h1`, `atr_14_h4`, `cvd`, `open_interest`, `vol_price_confirm`, `Sell_Liq`

| Feature | Standalone IC | Mutual Info | Marginal IC | MDA (F1 Drop) |
| :--- | :---: | :---: | :---: | :---: |
| `cvd_slope_h4` | +0.0821 | 0.0557 | +0.0827 | +0.03190 (**KEEP**) |
| `ofi_h4_delta` | +0.0853 | 0.0475 | +0.0853 | +0.02066 (**KEEP**) |
| `atr_zscore_20d` | +0.0353 | 0.0059 | +0.0296 | +0.01602 (**KEEP**) |
| `cvd_div_h4` | -0.0232 | 0.0000 | -0.0486 | +0.00933 (**KEEP**) |
| `stochrsi_d` | +0.0354 | 0.0000 | +0.0361 | +0.00395 (**KEEP**) |
| `cvd` | -0.0083 | 0.0530 | -0.0117 | +0.00304 (**KEEP**) |
| `whale_retail_divergence` | +0.0151 | 0.0057 | -0.0275 | +0.00288 (**KEEP**) |
| `Sell_Liq` | -0.0311 | 0.0017 | -0.0105 | +0.00271 (**KEEP**) |
| `ema_50_h1` | -0.0156 | 0.0000 | +0.0545 | +0.00221 (**KEEP**) |
| `atr_14_h4` | -0.0174 | 0.0000 | -0.0129 | +0.00182 (**KEEP**) |
| `open_interest` | +0.0118 | 0.0099 | +0.0157 | +0.00176 (**KEEP**) |
| `ema_21_h1` | -0.0266 | 0.0077 | +0.0145 | +0.00156 (**KEEP**) |
| `log_ret_5` | +0.0226 | 0.0024 | -0.0150 | +0.00120 (**KEEP**) |
| `vol_spike_zscore` | +0.0202 | 0.0000 | +0.0204 | +0.00118 (**KEEP**) |
| `VAH` | -0.0213 | 0.0017 | +0.0444 | +0.00111 (**KEEP**) |
| `ofi_acceleration` | -0.0022 | 0.0125 | -0.0245 | +0.00102 (**KEEP**) |
| `vol_accel_3h` | -0.0236 | 0.0085 | -0.0182 | +0.00066 (**KEEP**) |
| `vol_price_confirm` | +0.0240 | 0.0027 | -0.0114 | +0.00063 (**KEEP**) |
| `ema_7_h1` | -0.0298 | 0.0000 | -0.0155 | +0.00045 (**KEEP**) |
| `dist_from_8h_high` | +0.0255 | 0.0000 | +0.0191 | +0.00037 (**KEEP**) |
| `cvd_momentum_adv` | +0.0152 | 0.0060 | -0.0378 | +0.00004 (DROP) |
| `wyckoff_phase` | -0.0453 | 0.0000 | -0.0240 | +0.00000 (DROP) |
| `stochrsi_k` | +0.0317 | 0.0062 | -0.0219 | -0.00011 (DROP) |
| `long_short_ratio` | +0.0042 | 0.0154 | -0.0167 | -0.00031 (DROP) |
| `dist_swing_low` | +0.0293 | 0.0000 | -0.0160 | -0.00067 (DROP) |
| `rsi_6` | +0.0239 | 0.0043 | -0.0241 | -0.00080 (DROP) |
| `ema_21_slope_h4` | +0.0229 | 0.0000 | -0.0220 | -0.00170 (DROP) |
| `VAL` | -0.0278 | 0.0063 | -0.0145 | -0.00170 (DROP) |
| `Fib_618` | -0.0300 | 0.0074 | -0.0164 | -0.00172 (DROP) |

---

## Model: LGBM_TRENDING_DOWN

- **Total Fitur Terpilih**: 18 / 102
### Daftar Fitur Terpilih

`ofi_h4_delta`, `cvd_slope_h4`, `log_ret_20`, `cvd_div_h4`, `cvd`, `ema_21_slope_h4`, `cvd_momentum_adv`, `stochrsi_d`, `price_in_range`, `ema_21_h1`, `btc_dominance`, `ofi_acceleration`, `log_ret_5`, `vol_price_confirm`, `trend_accel_4h`, `dist_swing_low`, `VAL`, `wyckoff_phase`

| Feature | Standalone IC | Mutual Info | Marginal IC | MDA (F1 Drop) |
| :--- | :---: | :---: | :---: | :---: |
| `cvd_slope_h4` | +0.0903 | 0.0304 | +0.0898 | +0.03448 (**KEEP**) |
| `ofi_h4_delta` | +0.0905 | 0.0197 | +0.0905 | +0.01587 (**KEEP**) |
| `cvd_div_h4` | -0.0209 | 0.0005 | -0.0914 | +0.01037 (**KEEP**) |
| `log_ret_20` | +0.0106 | 0.0086 | -0.0596 | +0.00638 (**KEEP**) |
| `cvd` | -0.0137 | 0.0359 | -0.0289 | +0.00299 (**KEEP**) |
| `dist_swing_low` | +0.0166 | 0.0000 | -0.0102 | +0.00244 (**KEEP**) |
| `price_in_range` | +0.0187 | 0.0000 | -0.0250 | +0.00187 (**KEEP**) |
| `VAL` | -0.0187 | 0.0081 | -0.0113 | +0.00147 (**KEEP**) |
| `ema_21_slope_h4` | +0.0278 | 0.0010 | +0.0279 | +0.00138 (**KEEP**) |
| `log_ret_5` | +0.0171 | 0.0044 | -0.0144 | +0.00125 (**KEEP**) |
| `cvd_momentum_adv` | +0.0197 | 0.0000 | -0.0286 | +0.00107 (**KEEP**) |
| `ema_21_h1` | -0.0300 | 0.0054 | -0.0190 | +0.00107 (**KEEP**) |
| `ofi_acceleration` | +0.0012 | 0.0091 | -0.0163 | +0.00070 (**KEEP**) |
| `vol_price_confirm` | +0.0157 | 0.0086 | -0.0136 | +0.00068 (**KEEP**) |
| `trend_accel_4h` | +0.0260 | 0.0000 | +0.0126 | +0.00057 (**KEEP**) |
| `stochrsi_d` | +0.0300 | 0.0000 | +0.0247 | +0.00057 (**KEEP**) |
| `btc_dominance` | +0.0164 | 0.0000 | +0.0173 | +0.00051 (**KEEP**) |
| `wyckoff_phase` | -0.0427 | 0.0000 | -0.0107 | +0.00028 (**KEEP**) |
| `rsi_h4` | +0.0283 | 0.0067 | -0.0236 | -0.00014 (DROP) |
| `VAH` | -0.0203 | 0.0000 | +0.0127 | -0.00076 (DROP) |

---

## Model: LGBM_RANGING_GLOBAL

- **Total Fitur Terpilih**: 17 / 102
### Daftar Fitur Terpilih

`cvd_slope_h4`, `ofi_h4_delta`, `log_ret_20`, `cvd_div_h4`, `cvd`, `ofi_z_score`, `atr_percentile_h1`, `stochrsi_d`, `whale_retail_divergence`, `log_ret_5`, `cvd_momentum_adv`, `atr_zscore_20d`, `ema_200_h1`, `buy_volume`, `VAL`, `dist_swing_high`, `atr_percent_h4`

| Feature | Standalone IC | Mutual Info | Marginal IC | MDA (F1 Drop) |
| :--- | :---: | :---: | :---: | :---: |
| `cvd_slope_h4` | +0.0908 | 0.0174 | +0.0908 | +0.04262 (**KEEP**) |
| `cvd_div_h4` | -0.0220 | 0.0041 | -0.0909 | +0.01433 (**KEEP**) |
| `ofi_h4_delta` | +0.0752 | 0.0133 | +0.0744 | +0.00947 (**KEEP**) |
| `log_ret_20` | +0.0155 | 0.0033 | -0.0521 | +0.00626 (**KEEP**) |
| `atr_percentile_h1` | +0.0239 | 0.0047 | +0.0261 | +0.00468 (**KEEP**) |
| `atr_zscore_20d` | +0.0184 | 0.0046 | -0.0150 | +0.00342 (**KEEP**) |
| `atr_percent_h4` | +0.0108 | 0.0084 | -0.0103 | +0.00159 (**KEEP**) |
| `whale_retail_divergence` | +0.0211 | 0.0000 | -0.0276 | +0.00131 (**KEEP**) |
| `dist_swing_high` | +0.0197 | 0.0000 | -0.0108 | +0.00104 (**KEEP**) |
| `stochrsi_d` | +0.0269 | 0.0050 | +0.0234 | +0.00098 (**KEEP**) |
| `ofi_z_score` | -0.0048 | 0.0091 | -0.0284 | +0.00087 (**KEEP**) |
| `log_ret_5` | +0.0083 | 0.0119 | -0.0265 | +0.00058 (**KEEP**) |
| `cvd` | -0.0219 | 0.0098 | -0.0290 | +0.00033 (**KEEP**) |
| `cvd_momentum_adv` | +0.0223 | 0.0000 | -0.0168 | +0.00030 (**KEEP**) |
| `ema_200_h1` | -0.0216 | 0.0000 | +0.0146 | +0.00028 (**KEEP**) |
| `VAL` | -0.0219 | 0.0102 | -0.0106 | +0.00025 (**KEEP**) |
| `buy_volume` | -0.0042 | 0.0113 | +0.0107 | +0.00023 (**KEEP**) |
| `price_in_range` | +0.0231 | 0.0037 | +0.0112 | +0.00020 (DROP) |
| `spring_upthrust` | +0.0155 | 0.0075 | +0.0170 | +0.00013 (DROP) |
| `ema_50_slope_h4` | +0.0287 | 0.0000 | +0.0286 | -0.00003 (DROP) |
| `wyckoff_phase` | -0.0534 | 0.0000 | -0.0274 | -0.00005 (DROP) |
| `dow_cos` | -0.0171 | 0.0000 | -0.0149 | -0.00007 (DROP) |
| `h4_trend` | +0.0193 | 0.0119 | -0.0256 | -0.00010 (DROP) |
| `stochrsi_k` | +0.0213 | 0.0000 | -0.0111 | -0.00021 (DROP) |
| `Buy_Liq` | +0.0181 | 0.0057 | -0.0159 | -0.00026 (DROP) |
| `VAH` | -0.0189 | 0.0000 | +0.0194 | -0.00036 (DROP) |
| `ema_7_h1` | -0.0203 | 0.0009 | -0.0152 | -0.00052 (DROP) |

---

## Model: TradingLSTM

- **Total Fitur Terpilih**: 8 / 102
### Daftar Fitur Terpilih

`cvd_momentum_adv`, `ofi_h4_delta`, `cvd_slope_h4`, `atr_zscore_20d`, `vwdp_smooth`, `stochrsi_d`, `hour_sin`, `swing_momentum`

| Feature | Standalone IC | Mutual Info | Marginal IC | MDA (F1 Drop) |
| :--- | :---: | :---: | :---: | :---: |
| `cvd_momentum_adv` | -0.2583 | 0.0409 | -0.2583 | +0.05773 (**KEEP**) |
| `cvd_slope_h4` | -0.0871 | 0.0208 | +0.0752 | +0.01579 (**KEEP**) |
| `ofi_h4_delta` | +0.1106 | 0.0055 | +0.1023 | +0.00479 (**KEEP**) |
| `stochrsi_d` | -0.1521 | 0.0210 | +0.0226 | +0.00315 (**KEEP**) |
| `hour_sin` | -0.0224 | 0.0000 | -0.0179 | +0.00166 (**KEEP**) |
| `swing_momentum` | -0.0632 | 0.0060 | +0.0117 | +0.00101 (**KEEP**) |
| `vwdp_smooth` | -0.0980 | 0.0041 | -0.0265 | +0.00094 (**KEEP**) |
| `atr_zscore_20d` | +0.0226 | 0.0000 | +0.0290 | +0.00094 (**KEEP**) |
| `wyckoff_phase` | +0.1266 | 0.0152 | -0.0122 | -0.00003 (DROP) |
| `ultra_high_vol` | +0.0168 | 0.0018 | +0.0166 | -0.00007 (DROP) |
| `PDH` | +0.1423 | 0.0093 | -0.0119 | -0.00073 (DROP) |
| `dist_swing_low` | -0.1643 | 0.0132 | -0.0102 | -0.00095 (DROP) |
| `vwdp` | -0.0319 | 0.0000 | -0.0150 | -0.00105 (DROP) |
| `ema_7_h1` | +0.1862 | 0.0204 | -0.0110 | -0.00113 (DROP) |
| `rsi_slope_h4` | -0.0657 | 0.0054 | -0.0155 | -0.00114 (DROP) |
| `cvd_div_h4` | -0.0308 | 0.0064 | -0.0580 | -0.00117 (DROP) |
| `log_ret_1` | -0.0610 | 0.0116 | +0.0230 | -0.00118 (DROP) |
| `cvd` | -0.0135 | 0.0117 | -0.0205 | -0.00122 (DROP) |
| `long_short_ratio` | -0.0694 | 0.0084 | -0.0465 | -0.00129 (DROP) |
| `PWH` | +0.0353 | 0.0000 | +0.0140 | -0.00191 (DROP) |
| `price_in_range` | -0.1778 | 0.0165 | +0.0320 | -0.00204 (DROP) |
| `whale_retail_divergence` | -0.1995 | 0.0256 | -0.0910 | -0.00256 (DROP) |
| `rsi_h4` | -0.2108 | 0.0205 | -0.0794 | -0.00280 (DROP) |
| `ema_21_h1` | +0.2042 | 0.0381 | -0.0140 | -0.00280 (DROP) |
| `log_ret_20` | -0.1558 | 0.0172 | -0.0626 | -0.00520 (DROP) |
| `VAH` | +0.1873 | 0.0209 | +0.0127 | -0.00646 (DROP) |

---

