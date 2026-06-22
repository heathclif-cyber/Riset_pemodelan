# Post-Signal Path Analysis — ic32_regime_v1

Empiris: pergerakan harga **setelah** bar entry (holdout Apr-Jun 2026).

## Holdout (936 trades)

- SL tersentuh **bar +1**: 0.64%
- SL tersentuh **bar +2**: 1.92%
- Losers yang arah benar di +12h: 31.27%

### Forward return rata-rata (semua trade, directional %)

- +1h: +0.4628%
- +2h: +0.7264%
- +3h: +0.8864%
- +6h: +0.9072%
- +12h: +0.9308%
- +24h: +0.8750%

### Slice comparison

| Slice | n | WR% | SL bar1% | dir_ok_12% | loss+dir_ok_12% | fwd+1% | mae_12% |
|-------|--:|----:|---------:|-----------:|----------------:|-------:|--------:|
| all | 936 | 62.1 | 0.6 | 65.1 | 11.9 | 0.463 | 1.212 |
| losers | 355 | 0.0 | 1.7 | 31.3 | 31.3 | 0.205 | 2.213 |
| winners | 581 | 100.0 | 0.0 | 85.7 | 0.0 | 0.621 | 0.6 |
| sl_touch_within_2bars | 24 | 0.0 | 25.0 | 37.5 | 37.5 | -0.434 | 3.219 |
| losers_but_dir_ok_12h | 111 | 0.0 | 1.8 | 100.0 | 100.0 | 0.306 | 1.236 |
| losers_sl_bar1 | 6 | 0.0 | 100.0 | 33.3 | 33.3 | -0.892 | 2.064 |
| sl_bar1_dir_ok_12h | 2 | 0.0 | 100.0 | 100.0 | 100.0 | 0.826 | 1.172 |
| vol_spike_ge_2 | 113 | 65.5 | 0.9 | 70.8 | 11.5 | 0.646 | 1.016 |
| vol_spike_ge_2_losers | 39 | 0.0 | 2.6 | 33.3 | 33.3 | 0.386 | 1.945 |
| repeat_entry_1h_gap | 190 | 62.6 | 1.6 | 64.2 | 11.6 | 0.482 | 1.191 |
| repeat_1h_losers | 71 | 0.0 | 4.2 | 31.0 | 31.0 | 0.193 | 2.133 |

## Live ic32 (62 trades)

- SL bar +1: 3.23%
- Losers dir ok +12h: 23.33%