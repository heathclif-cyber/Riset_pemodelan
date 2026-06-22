# IC Decay Test — ic_decay_weak_v1
*2026-06-04 23:56 | 6 windows*

**Thresholds**: IC_IR >= 0.5, sign konsisten >= 5/6 window

## Summary

| Verdict | Count | Artinya |
|---------|-------|--------|
| **STABLE** | 7 | Sinyal konsisten lintas semua regime — masuk model |
| **MARGINAL** | 0 | Sinyal cukup konsisten — masuk dengan catatan |
| **UNSTABLE** | 3 | Sinyal tidak konsisten — regime-specific, hindari |
| **INSUFFICIENT_DATA** | 0 | Data tidak cukup di beberapa window |

## STABLE Features (7)

`dist_liq_20x_short`, `vol_price_confirm`, `ema_50_slope_h4`, `MSB_BOS`, `cvd`, `ofi_acceleration`, `cvd_div_h4`

## STABLE (7)

| Feature | Mean IC | Std | IC_IR | Sign | IC 2020 | IC 2021 | IC 2022 | IC 2023 | IC 2024 | IC 2025 | Verdict |
|---------|--------:|----:|------:|-----:|------:|------:|------:|------:|------:|------:|--------|
| `dist_liq_20x_short` | +0.0245 | 0.0140 | 1.74 | 6/6 | +0.0106 | +0.0445 | +0.0162 | +0.0129 | +0.0245 | +0.0382 | **STABLE** |
| `vol_price_confirm` | -0.0177 | 0.0095 | 1.86 | 5/6 | -0.0184 | -0.0212 | -0.0173 | -0.0268 | +0.0005 | -0.0231 | **STABLE** |
| `ema_50_slope_h4` | -0.0157 | 0.0143 | 1.09 | 5/6 | -0.0277 | -0.0251 | -0.0117 | -0.0211 | +0.0114 | -0.0201 | **STABLE** |
| `MSB_BOS` | -0.0152 | 0.0078 | 1.95 | 6/6 | -0.0164 | -0.0104 | -0.0157 | -0.0248 | -0.0028 | -0.0208 | **STABLE** |
| `cvd` | -0.0087 | 0.0057 | 1.52 | 6/6 | -0.0142 | -0.0123 | -0.0001 | -0.0127 | -0.0033 | -0.0093 | **STABLE** |
| `ofi_acceleration` | -0.0071 | 0.0043 | 1.65 | 6/6 | -0.0049 | -0.0141 | -0.0050 | -0.0109 | -0.0044 | -0.0035 | **STABLE** |
| `cvd_div_h4` | -0.0061 | 0.0060 | 1.02 | 5/6 | -0.0039 | -0.0077 | -0.0147 | -0.0002 | -0.0105 | +0.0006 | **STABLE** |

## UNSTABLE (3)

| Feature | Mean IC | Std | IC_IR | Sign | IC 2020 | IC 2021 | IC 2022 | IC 2023 | IC 2024 | IC 2025 | Verdict |
|---------|--------:|----:|------:|-----:|------:|------:|------:|------:|------:|------:|--------|
| `dow_cos` | -0.0055 | 0.0227 | 0.24 | 5/6 | -0.0041 | -0.0167 | +0.0390 | -0.0239 | -0.0123 | -0.0147 | **UNSTABLE** |
| `wyckoff_phase` | +0.0018 | 0.0148 | 0.12 | 4/6 | +0.0156 | +0.0064 | -0.0075 | +0.0072 | -0.0236 | +0.0127 | **UNSTABLE** |
| `atr_14_h1` | -0.0011 | 0.0074 | 0.15 | 2/6 | -0.0101 | -0.0106 | +0.0012 | +0.0065 | +0.0044 | +0.0019 | **UNSTABLE** |

