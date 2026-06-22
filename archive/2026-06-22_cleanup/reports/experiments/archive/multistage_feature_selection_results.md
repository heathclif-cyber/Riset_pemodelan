# 🧪 Multi-Stage Feature Selection Results (Coinank Integrated)
*2026-06-08 14:09 | training_cutoff=2026-04-01*

Laporan ini berisi hasil penyaringan fitur multi-tahap: **Spearman IC + Mutual Info**, **Gram-Schmidt Marginal IC**, **OOF MDA**, dan **SHAP Validation**.

## Model: LGBM_TRENDING_UP

- **Total Fitur Terpilih**: 18 / 105
### Daftar Fitur Terpilih

`dist_from_8h_high`, `cvd_slope_h4`, `ofi_h4_delta`, `ultra_high_vol`, `POC`, `absorption_at_swing`, `swing_momentum`, `ema_50_h1`, `ema_50_slope_h4`, `log_ret_20`, `vol_price_confirm`, `vol_ratio_20`, `ema_21_slope_h4`, `dist_liq_50x_long`, `PDL`, `dist_liq_50x_short`, `log_ret_1`, `dist_swing_low`

| Feature | Standalone IC | Mutual Info | Marginal IC | MDA (F1 Drop) | Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `vol_ratio_20` | +0.0083 | 0.0372 | -0.0170 | +0.08291 | **KEEP** |
| `ultra_high_vol` | +0.0434 | 0.0404 | +0.0369 | +0.08136 | **KEEP** |
| `dist_liq_50x_short` | +0.0629 | 0.0306 | +0.0163 | +0.01453 | **KEEP** |
| `dist_swing_low` | -0.0509 | 0.0120 | +0.0143 | +0.01210 | **KEEP** |
| `vol_price_confirm` | -0.0142 | 0.0460 | +0.0128 | +0.00817 | **KEEP** |
| `cvd_slope_h4` | +0.0298 | 0.0198 | +0.0602 | +0.00464 | **KEEP** |
| `ofi_h4_delta` | +0.0515 | 0.0238 | +0.0495 | +0.00391 | **KEEP** |
| `dist_from_8h_high` | -0.1049 | 0.0192 | -0.1049 | +0.00328 | **KEEP** |
| `dist_liq_50x_long` | -0.0709 | 0.0315 | -0.0127 | +0.00227 | **KEEP** |
| `PDL` | +0.0441 | 0.0043 | +0.0161 | +0.00146 | **KEEP** |
| `log_ret_20` | -0.0298 | 0.0065 | -0.0125 | +0.00129 | **KEEP** |
| `log_ret_1` | -0.0466 | 0.0091 | -0.0137 | +0.00125 | **KEEP** |
| `ema_21_slope_h4` | -0.0414 | 0.0047 | +0.0135 | +0.00090 | **KEEP** |
| `ema_50_slope_h4` | -0.0251 | 0.0082 | -0.0201 | +0.00089 | **KEEP** |
| `swing_momentum` | -0.0744 | 0.0000 | -0.0204 | +0.00083 | **KEEP** |
| `ema_50_h1` | +0.0369 | 0.0015 | -0.0173 | +0.00066 | **KEEP** |
| `absorption_at_swing` | -0.0325 | 0.0373 | -0.0197 | +0.00052 | **KEEP** |
| `POC` | +0.0693 | 0.0095 | +0.0217 | +0.00024 | **KEEP** |
| `etf_gbtc_change_usd` | +0.0139 | 0.0114 | +0.0123 | +0.00017 | DROP |
| `hour_sin` | -0.0166 | 0.0041 | -0.0146 | +0.00007 | DROP |
| `h4_trend` | -0.0350 | 0.0000 | -0.0153 | +0.00000 | DROP |
| `no_supply` | +0.0192 | 0.0000 | +0.0129 | -0.00000 | DROP |
| `whale_retail_divergence` | -0.0348 | 0.0015 | -0.0644 | -0.00004 | DROP |
| `trend_strength` | -0.0151 | 0.0038 | +0.0189 | -0.00005 | DROP |
| `rsi_divergence` | -0.0276 | 0.0036 | +0.0182 | -0.00011 | DROP |
| `rsi_6` | -0.0892 | 0.0252 | -0.0352 | -0.00016 | DROP |
| `Buy_Liq` | -0.0362 | 0.0211 | +0.0125 | -0.00027 | DROP |
| `cvd_momentum_adv` | -0.0469 | 0.0040 | -0.0129 | -0.00035 | DROP |
| `stochrsi_d` | -0.0456 | 0.0112 | +0.0432 | -0.00046 | DROP |
| `ema_7_h1` | +0.0825 | 0.0344 | -0.0247 | -0.00047 | DROP |
| `atr_zscore_20d` | +0.0307 | 0.0021 | +0.0147 | -0.00067 | DROP |
| `spread_to_volume` | -0.0151 | 0.0097 | +0.0191 | -0.00080 | DROP |
| `rsi_slope_h4` | -0.0528 | 0.0040 | +0.0350 | -0.00096 | DROP |
| `long_short_ratio` | -0.0416 | 0.0175 | +0.0151 | -0.00098 | DROP |
| `trend_accel_4h` | -0.0605 | 0.0014 | -0.0194 | -0.00123 | DROP |
| `atr_percent_h4` | +0.0226 | 0.0064 | +0.0301 | -0.00144 | DROP |
| `dist_liq_20x_long` | -0.0557 | 0.0017 | +0.0272 | -0.00163 | DROP |
| `dist_liq_20x_short` | +0.0295 | 0.0063 | -0.0422 | -0.00179 | DROP |

---

## Model: LGBM_TRENDING_DOWN

- **Total Fitur Terpilih**: 17 / 105
### Daftar Fitur Terpilih

`dist_from_8h_high`, `cvd_slope_h4`, `ofi_h4_delta`, `stochrsi_d`, `etf_gbtc_change_usd`, `ultra_high_vol`, `etf_total_change_usd`, `dist_liq_50x_long`, `dist_liq_20x_long`, `dist_liq_20x_short`, `ema_7_h1`, `dist_swing_low`, `PDH`, `absorption_at_swing`, `log_ret_5`, `vol_accel_3h`, `vol_price_confirm`

| Feature | Standalone IC | Mutual Info | Marginal IC | MDA (F1 Drop) | Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `ultra_high_vol` | -0.0261 | 0.0357 | -0.0331 | +0.08054 | **KEEP** |
| `vol_price_confirm` | -0.0163 | 0.0437 | +0.0123 | +0.06267 | **KEEP** |
| `cvd_slope_h4` | +0.0413 | 0.0149 | +0.0728 | +0.01267 | **KEEP** |
| `vol_accel_3h` | -0.0020 | 0.0233 | +0.0107 | +0.00741 | **KEEP** |
| `dist_swing_low` | -0.0641 | 0.0178 | -0.0174 | +0.00517 | **KEEP** |
| `dist_from_8h_high` | -0.0992 | 0.0239 | -0.0992 | +0.00367 | **KEEP** |
| `ema_7_h1` | +0.0891 | 0.0380 | -0.0181 | +0.00349 | **KEEP** |
| `PDH` | +0.0250 | 0.0000 | +0.0180 | +0.00211 | **KEEP** |
| `absorption_at_swing` | -0.0341 | 0.0348 | -0.0142 | +0.00193 | **KEEP** |
| `dist_liq_20x_long` | -0.0392 | 0.0000 | +0.0285 | +0.00187 | **KEEP** |
| `ofi_h4_delta` | +0.0557 | 0.0051 | +0.0620 | +0.00172 | **KEEP** |
| `dist_liq_20x_short` | +0.0213 | 0.0021 | -0.0368 | +0.00111 | **KEEP** |
| `etf_gbtc_change_usd` | +0.0373 | 0.0076 | +0.0401 | +0.00106 | **KEEP** |
| `dist_liq_50x_long` | -0.0674 | 0.0179 | -0.0279 | +0.00075 | **KEEP** |
| `etf_total_change_usd` | +0.0376 | 0.0149 | +0.0318 | +0.00070 | **KEEP** |
| `log_ret_5` | -0.0791 | 0.0090 | -0.0107 | +0.00033 | **KEEP** |
| `stochrsi_d` | -0.0490 | 0.0117 | +0.0445 | +0.00025 | **KEEP** |
| `trend_accel_4h` | -0.0653 | 0.0072 | -0.0197 | +0.00015 | DROP |
| `h4_trend` | -0.0212 | 0.0000 | -0.0140 | +0.00000 | DROP |
| `log_ret_20` | -0.0210 | 0.0000 | -0.0171 | -0.00015 | DROP |
| `swing_momentum` | -0.0759 | 0.0000 | -0.0218 | -0.00020 | DROP |
| `coinank_oi_change_24h` | +0.0222 | 0.0063 | +0.0147 | -0.00027 | DROP |
| `rsi_divergence` | -0.0378 | 0.0046 | +0.0114 | -0.00032 | DROP |
| `ema_21_slope_h4` | -0.0306 | 0.0000 | -0.0231 | -0.00036 | DROP |
| `vwdp` | -0.0189 | 0.0018 | -0.0100 | -0.00039 | DROP |
| `rsi_slope_h4` | -0.0616 | 0.0018 | +0.0335 | -0.00043 | DROP |
| `rsi_6` | -0.0944 | 0.0181 | -0.0399 | -0.00052 | DROP |
| `rsi_h4` | -0.0737 | 0.0357 | -0.0216 | -0.00052 | DROP |
| `btc_dominance` | +0.0036 | 0.0087 | +0.0137 | -0.00053 | DROP |
| `ema_50_h1` | +0.0220 | 0.0079 | -0.0252 | -0.00068 | DROP |
| `ema_21_h1` | +0.0550 | 0.0043 | -0.0199 | -0.00073 | DROP |
| `dist_liq_50x_short` | +0.0465 | 0.0244 | +0.0230 | -0.00077 | DROP |
| `price_in_range` | -0.0636 | 0.0076 | -0.0163 | -0.00077 | DROP |
| `cvd_momentum_adv` | -0.0448 | 0.0015 | -0.0102 | -0.00077 | DROP |
| `cvd` | -0.0109 | 0.0200 | -0.0238 | -0.00131 | DROP |
| `whale_retail_divergence` | -0.0234 | 0.0000 | -0.0456 | -0.00171 | DROP |

---

## Model: LGBM_RANGING_LOW_VOL

- **Total Fitur Terpilih**: 16 / 105
### Daftar Fitur Terpilih

`dist_from_8h_high`, `cvd_slope_h4`, `ofi_h4_delta`, `rsi_6`, `etf_total_change_usd`, `dow_cos`, `swing_momentum`, `long_short_ratio`, `vol_price_confirm`, `ofi_raw`, `dist_liq_20x_long`, `vol_ratio_20`, `absorption_z`, `ema_50_slope_h4`, `atr_percentile_h1`, `ema_200_h1`

| Feature | Standalone IC | Mutual Info | Marginal IC | MDA (F1 Drop) | Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `vol_ratio_20` | +0.0094 | 0.0352 | +0.0122 | +0.06754 | **KEEP** |
| `vol_price_confirm` | -0.0282 | 0.0392 | +0.0135 | +0.03514 | **KEEP** |
| `absorption_z` | -0.0063 | 0.0107 | -0.0164 | +0.01479 | **KEEP** |
| `cvd_slope_h4` | +0.0466 | 0.0053 | +0.0749 | +0.01227 | **KEEP** |
| `dow_cos` | -0.0224 | 0.0180 | -0.0264 | +0.00478 | **KEEP** |
| `rsi_6` | -0.0958 | 0.0308 | -0.0476 | +0.00371 | **KEEP** |
| `atr_percentile_h1` | +0.0171 | 0.0000 | +0.0105 | +0.00296 | **KEEP** |
| `ema_50_slope_h4` | -0.0270 | 0.0012 | -0.0104 | +0.00287 | **KEEP** |
| `dist_from_8h_high` | -0.0994 | 0.0223 | -0.0994 | +0.00177 | **KEEP** |
| `ema_200_h1` | +0.0183 | 0.0042 | +0.0110 | +0.00175 | **KEEP** |
| `ofi_h4_delta` | +0.0646 | 0.0113 | +0.0723 | +0.00172 | **KEEP** |
| `long_short_ratio` | -0.0432 | 0.0148 | +0.0187 | +0.00150 | **KEEP** |
| `etf_total_change_usd` | +0.0388 | 0.0084 | +0.0431 | +0.00089 | **KEEP** |
| `dist_liq_20x_long` | -0.0493 | 0.0000 | +0.0125 | +0.00080 | **KEEP** |
| `ofi_raw` | -0.0275 | 0.0074 | -0.0130 | +0.00073 | **KEEP** |
| `swing_momentum` | -0.0733 | 0.0000 | -0.0199 | +0.00041 | **KEEP** |
| `price_accel_1h` | -0.0173 | 0.0000 | -0.0305 | +0.00020 | DROP |
| `Sell_Liq` | +0.0490 | 0.0142 | +0.0148 | +0.00017 | DROP |
| `Fib_618` | +0.0608 | 0.0128 | -0.0213 | +0.00012 | DROP |
| `rsi_slope_h4` | -0.0563 | 0.0055 | +0.0206 | +0.00011 | DROP |
| `etf_gbtc_change_usd` | +0.0230 | 0.0012 | +0.0208 | +0.00010 | DROP |
| `ema_21_slope_h4` | -0.0441 | 0.0000 | +0.0291 | +0.00008 | DROP |
| `atr_percent_h4` | +0.0184 | 0.0000 | +0.0118 | +0.00004 | DROP |
| `log_ret_20` | -0.0291 | 0.0095 | -0.0201 | +0.00003 | DROP |
| `MSB_BOS` | -0.0171 | 0.0098 | +0.0372 | +0.00000 | DROP |
| `h4_trend` | -0.0387 | 0.0000 | -0.0165 | +0.00000 | DROP |
| `rsi_divergence` | -0.0263 | 0.0046 | +0.0269 | -0.00002 | DROP |
| `price_in_range` | -0.0688 | 0.0231 | -0.0315 | -0.00005 | DROP |
| `dist_swing_low` | -0.0641 | 0.0040 | -0.0216 | -0.00029 | DROP |
| `price_vs_ema_50_h4` | -0.0412 | 0.0015 | +0.0212 | -0.00033 | DROP |
| `trend_strength` | -0.0179 | 0.0000 | +0.0145 | -0.00042 | DROP |
| `dist_liq_50x_long` | -0.0704 | 0.0201 | -0.0199 | -0.00053 | DROP |
| `whale_retail_divergence` | -0.0245 | 0.0000 | -0.0512 | -0.00056 | DROP |
| `cvd_div_h4` | +0.0030 | 0.0087 | -0.0401 | -0.00065 | DROP |
| `stochrsi_k` | -0.0628 | 0.0123 | +0.0496 | -0.00078 | DROP |
| `ema_21_h1` | +0.0672 | 0.0159 | -0.0137 | -0.00102 | DROP |
| `stochrsi_d` | -0.0499 | 0.0123 | +0.0102 | -0.00110 | DROP |

---

## Model: LGBM_RANGING_HIGH_VOL

- **Total Fitur Terpilih**: 15 / 105
### Daftar Fitur Terpilih

`cvd_slope_h4`, `ofi_h4_delta`, `etf_total_change_usd`, `rsi_6`, `vol_efficiency`, `ema_50_h1`, `vol_price_confirm`, `dist_liq_50x_long`, `dist_liq_20x_short`, `log_ret_20`, `etf_gbtc_change_usd`, `PDH`, `long_short_ratio`, `dist_swing_low`, `dist_swing_high`

| Feature | Standalone IC | Mutual Info | Marginal IC | MDA (F1 Drop) | Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `vol_price_confirm` | -0.0131 | 0.0490 | +0.0267 | +0.04758 | **KEEP** |
| `PDH` | +0.0268 | 0.0051 | +0.0123 | +0.01575 | **KEEP** |
| `cvd_slope_h4` | +0.0394 | 0.0080 | +0.0672 | +0.01066 | **KEEP** |
| `ema_50_h1` | +0.0200 | 0.0000 | -0.0231 | +0.00866 | **KEEP** |
| `dist_swing_low` | -0.0569 | 0.0177 | -0.0138 | +0.00604 | **KEEP** |
| `dist_swing_high` | -0.0532 | 0.0108 | -0.0123 | +0.00521 | **KEEP** |
| `log_ret_20` | -0.0221 | 0.0000 | -0.0202 | +0.00408 | **KEEP** |
| `dist_liq_20x_short` | +0.0150 | 0.0045 | -0.0308 | +0.00272 | **KEEP** |
| `vol_efficiency` | -0.0212 | 0.0077 | -0.0239 | +0.00138 | **KEEP** |
| `etf_total_change_usd` | +0.0443 | 0.0124 | +0.0449 | +0.00086 | **KEEP** |
| `ofi_h4_delta` | +0.0558 | 0.0078 | +0.0627 | +0.00085 | **KEEP** |
| `dist_liq_50x_long` | -0.0697 | 0.0173 | -0.0184 | +0.00071 | **KEEP** |
| `long_short_ratio` | -0.0418 | 0.0112 | +0.0207 | +0.00070 | **KEEP** |
| `rsi_6` | -0.0928 | 0.0177 | -0.0436 | +0.00034 | **KEEP** |
| `etf_gbtc_change_usd` | +0.0202 | 0.0085 | +0.0142 | +0.00032 | **KEEP** |
| `rsi_h4` | -0.0739 | 0.0213 | +0.0191 | +0.00017 | DROP |
| `MSB_BOS` | -0.0166 | 0.0026 | +0.0312 | +0.00000 | DROP |
| `h4_trend` | -0.0270 | 0.0017 | -0.0204 | +0.00000 | DROP |
| `VAH` | +0.0567 | 0.0082 | -0.0143 | -0.00001 | DROP |
| `swing_momentum` | -0.0721 | 0.0042 | -0.0132 | -0.00003 | DROP |
| `dow_cos` | -0.0163 | 0.0000 | -0.0199 | -0.00015 | DROP |
| `stochrsi_k` | -0.0700 | 0.0109 | +0.0354 | -0.00018 | DROP |
| `rsi_divergence` | -0.0344 | 0.0000 | +0.0145 | -0.00042 | DROP |
| `log_ret_5` | -0.0759 | 0.0026 | -0.0121 | -0.00064 | DROP |
| `rsi_slope_h4` | -0.0566 | 0.0066 | +0.0258 | -0.00092 | DROP |
| `whale_retail_divergence` | -0.0218 | 0.0007 | -0.0450 | -0.00104 | DROP |
| `dist_from_8h_high` | -0.0980 | 0.0274 | -0.0980 | -0.00137 | DROP |
| `stochrsi_d` | -0.0544 | 0.0048 | +0.0106 | -0.00228 | DROP |
| `dist_liq_20x_long` | -0.0458 | 0.0026 | +0.0201 | -0.00291 | DROP |
| `trend_accel_4h` | -0.0711 | 0.0017 | -0.0230 | -0.00342 | DROP |
| `relative_strength_z` | -0.0486 | 0.0091 | -0.0130 | -0.00483 | DROP |
| `dist_liq_50x_short` | +0.0490 | 0.0122 | +0.0517 | -0.00526 | DROP |

---

## Model: LGBM_GLOBAL_FALLBACK

- **Total Fitur Terpilih**: 15 / 105
### Daftar Fitur Terpilih

`dist_from_8h_high`, `cvd_slope_h4`, `etf_gbtc_change_usd`, `stochrsi_d`, `rsi_6`, `ema_50_h1`, `swing_momentum`, `etf_total_change_usd`, `long_short_ratio`, `ema_21_h1`, `price_in_range`, `Buy_Liq`, `ema_21_slope_h4`, `vol_price_confirm`, `ofi_z_score`

| Feature | Standalone IC | Mutual Info | Marginal IC | MDA (F1 Drop) | Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `vol_price_confirm` | -0.0171 | 0.0432 | +0.0104 | +0.03833 | **KEEP** |
| `Buy_Liq` | -0.0535 | 0.0143 | -0.0116 | +0.01297 | **KEEP** |
| `cvd_slope_h4` | +0.0398 | 0.0028 | +0.0730 | +0.01121 | **KEEP** |
| `ema_50_h1` | +0.0214 | 0.0027 | -0.0317 | +0.00819 | **KEEP** |
| `dist_from_8h_high` | -0.1087 | 0.0257 | -0.1087 | +0.00683 | **KEEP** |
| `long_short_ratio` | -0.0366 | 0.0162 | +0.0230 | +0.00647 | **KEEP** |
| `rsi_6` | -0.0976 | 0.0250 | -0.0389 | +0.00583 | **KEEP** |
| `ofi_z_score` | -0.0296 | 0.0147 | -0.0105 | +0.00571 | **KEEP** |
| `ema_21_h1` | +0.0585 | 0.0097 | -0.0225 | +0.00526 | **KEEP** |
| `etf_total_change_usd` | +0.0321 | 0.0102 | +0.0235 | +0.00159 | **KEEP** |
| `ema_21_slope_h4` | -0.0323 | 0.0039 | -0.0110 | +0.00075 | **KEEP** |
| `swing_momentum` | -0.0817 | 0.0035 | -0.0273 | +0.00074 | **KEEP** |
| `price_in_range` | -0.0672 | 0.0153 | -0.0153 | +0.00032 | **KEEP** |
| `stochrsi_d` | -0.0515 | 0.0178 | +0.0363 | +0.00028 | **KEEP** |
| `etf_gbtc_change_usd` | +0.0376 | 0.0084 | +0.0383 | +0.00026 | **KEEP** |
| `cvd_momentum_adv` | -0.0458 | 0.0000 | -0.0147 | +0.00016 | DROP |
| `whale_retail_divergence` | -0.0215 | 0.0107 | -0.0517 | +0.00014 | DROP |
| `volume_delta` | -0.0429 | 0.0008 | -0.0147 | +0.00012 | DROP |
| `h4_trend` | -0.0244 | 0.0071 | -0.0199 | +0.00000 | DROP |
| `cvd` | -0.0131 | 0.0109 | -0.0164 | -0.00005 | DROP |
| `dist_liq_20x_short` | +0.0226 | 0.0027 | -0.0273 | -0.00016 | DROP |
| `rsi_divergence` | -0.0372 | 0.0064 | +0.0132 | -0.00016 | DROP |
| `dist_liq_50x_long` | -0.0684 | 0.0147 | -0.0173 | -0.00018 | DROP |
| `ofi_h4_delta` | +0.0397 | 0.0027 | +0.0413 | -0.00022 | DROP |
| `log_ret_1` | -0.0419 | 0.0026 | -0.0114 | -0.00029 | DROP |
| `log_ret_5` | -0.0827 | 0.0178 | -0.0135 | -0.00051 | DROP |
| `Sell_Liq` | +0.0452 | 0.0153 | +0.0128 | -0.00061 | DROP |
| `rsi_slope_h4` | -0.0646 | 0.0045 | +0.0354 | -0.00094 | DROP |
| `log_ret_20` | -0.0247 | 0.0033 | -0.0143 | -0.00094 | DROP |
| `trend_accel_4h` | -0.0777 | 0.0000 | -0.0350 | -0.00247 | DROP |
| `dist_liq_20x_long` | -0.0418 | 0.0050 | +0.0217 | -0.00321 | DROP |
| `range_expansion_h4` | +0.0112 | 0.0089 | +0.0136 | -0.00423 | DROP |
| `dist_liq_50x_short` | +0.0524 | 0.0222 | +0.0311 | -0.00575 | DROP |

---

## Model: TradingLSTM

- **Total Fitur Terpilih**: 15 / 105
### Daftar Fitur Terpilih

`dist_from_8h_high`, `cvd_slope_h4`, `etf_gbtc_change_usd`, `stochrsi_d`, `rsi_6`, `ema_50_h1`, `swing_momentum`, `etf_total_change_usd`, `long_short_ratio`, `ema_21_h1`, `price_in_range`, `Buy_Liq`, `ema_21_slope_h4`, `vol_price_confirm`, `ofi_z_score`

| Feature | Standalone IC | Mutual Info | Marginal IC | MDA (F1 Drop) | Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `vol_price_confirm` | -0.0171 | 0.0432 | +0.0104 | +0.03833 | **KEEP** |
| `Buy_Liq` | -0.0535 | 0.0143 | -0.0116 | +0.01297 | **KEEP** |
| `cvd_slope_h4` | +0.0398 | 0.0028 | +0.0730 | +0.01121 | **KEEP** |
| `ema_50_h1` | +0.0214 | 0.0027 | -0.0317 | +0.00819 | **KEEP** |
| `dist_from_8h_high` | -0.1087 | 0.0257 | -0.1087 | +0.00683 | **KEEP** |
| `long_short_ratio` | -0.0366 | 0.0162 | +0.0230 | +0.00647 | **KEEP** |
| `rsi_6` | -0.0976 | 0.0250 | -0.0389 | +0.00583 | **KEEP** |
| `ofi_z_score` | -0.0296 | 0.0147 | -0.0105 | +0.00571 | **KEEP** |
| `ema_21_h1` | +0.0585 | 0.0097 | -0.0225 | +0.00526 | **KEEP** |
| `etf_total_change_usd` | +0.0321 | 0.0102 | +0.0235 | +0.00159 | **KEEP** |
| `ema_21_slope_h4` | -0.0323 | 0.0039 | -0.0110 | +0.00075 | **KEEP** |
| `swing_momentum` | -0.0817 | 0.0035 | -0.0273 | +0.00074 | **KEEP** |
| `price_in_range` | -0.0672 | 0.0153 | -0.0153 | +0.00032 | **KEEP** |
| `stochrsi_d` | -0.0515 | 0.0178 | +0.0363 | +0.00028 | **KEEP** |
| `etf_gbtc_change_usd` | +0.0376 | 0.0084 | +0.0383 | +0.00026 | **KEEP** |
| `cvd_momentum_adv` | -0.0458 | 0.0000 | -0.0147 | +0.00016 | DROP |
| `whale_retail_divergence` | -0.0215 | 0.0107 | -0.0517 | +0.00014 | DROP |
| `volume_delta` | -0.0429 | 0.0008 | -0.0147 | +0.00012 | DROP |
| `h4_trend` | -0.0244 | 0.0071 | -0.0199 | +0.00000 | DROP |
| `cvd` | -0.0131 | 0.0109 | -0.0164 | -0.00005 | DROP |
| `dist_liq_20x_short` | +0.0226 | 0.0027 | -0.0273 | -0.00016 | DROP |
| `rsi_divergence` | -0.0372 | 0.0064 | +0.0132 | -0.00016 | DROP |
| `dist_liq_50x_long` | -0.0684 | 0.0147 | -0.0173 | -0.00018 | DROP |
| `ofi_h4_delta` | +0.0397 | 0.0027 | +0.0413 | -0.00022 | DROP |
| `log_ret_1` | -0.0419 | 0.0026 | -0.0114 | -0.00029 | DROP |
| `log_ret_5` | -0.0827 | 0.0178 | -0.0135 | -0.00051 | DROP |
| `Sell_Liq` | +0.0452 | 0.0153 | +0.0128 | -0.00061 | DROP |
| `rsi_slope_h4` | -0.0646 | 0.0045 | +0.0354 | -0.00094 | DROP |
| `log_ret_20` | -0.0247 | 0.0033 | -0.0143 | -0.00094 | DROP |
| `trend_accel_4h` | -0.0777 | 0.0000 | -0.0350 | -0.00247 | DROP |
| `dist_liq_20x_long` | -0.0418 | 0.0050 | +0.0217 | -0.00321 | DROP |
| `range_expansion_h4` | +0.0112 | 0.0089 | +0.0136 | -0.00423 | DROP |
| `dist_liq_50x_short` | +0.0524 | 0.0222 | +0.0311 | -0.00575 | DROP |

---

