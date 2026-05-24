# Trade Analysis Report

**Generated:** 2026-05-22 23:09  
**File:** `D:\Apps-Dev\Riset_pemodelan\reports\experiments\holdout_20260522_230614_holdout_trade_history.csv`  
**Period:** 2025-05-01 s/d 2026-03-31  
**Closed:** 22914 | **Open:** 0  
**Filter params:** P2 VolR >= 0.05 | P4 swing_dev <= 8%

---

## 1. Overview

| Metrik | Nilai |
|---|---|
| Total trades (closed) | 22914 |
| Open positions | 0 |
| Win / Loss | 20447 / 2467 |
| **Win Rate** | **89.2%** |
| **Net PnL** | **$+169625.85** |
| Gross Profit | $188701.17 |
| Gross Loss | $19075.33 |
| **Profit Factor** | **9.89** |
| Sharpe-like | 1.05 |
| Avg Win | $+9.23 |
| Avg Loss | $-7.73 |
| Avg Win / Avg Loss | 1.19x |
| Max consec. losses | 27 trades ($-232.28) |

## 2. Per Model

| Model | N | WR | Net PnL | PF | Avg Conf | Avg Hold |
|---|---|---|---|---|---|---|
| cascade_v3 | 22914 | 89.2% | $+169625.85 | 9.89 | 0.81 | 5.5h |

## 3. Exit Reasons

| Exit Reason | N | % | WR | Avg PnL | Total PnL |
|---|---|---|---|---|---|
| tp_hit | 8829 | 38.5% | 100.0% | $+10.71 | $+94572.33 |
| sl_hit | 1093 | 4.8% | 0.0% | $-9.14 | $-9985.19 |
| guardian_exit | 12800 | 55.9% | 90.6% | $+6.65 | $+85056.13 |
| time_exit | 192 | 0.8% | 12.5% | $-0.09 | $-17.43 |

## 4. Confidence Bands

| Band | N | WR | Avg PnL | PF |
|---|---|---|---|---|
| 0.60-0.70 | 2882 | 81.9% | $+5.94 | 5.18 |
| 0.70-0.80 | 7542 | 85.2% | $+6.58 | 6.75 |
| 0.80-0.90 | 6841 | 91.2% | $+7.66 | 12.17 |
| 0.90-1.00 | 5649 | 96.0% | $+8.93 | 31.44 |

## 5. Volume Regime

| Vol Regime | N | WR | Avg PnL | PF |
|---|---|---|---|---|
| < 0.05 | 2 | 0.0% | $-0.65 | 0.00 |
| 0.05-0.20 | 106 | 93.4% | $+7.87 | 21.21 |
| 0.20-0.50 | 5156 | 87.2% | $+6.84 | 7.66 |
| 0.50-1.00 | 11435 | 88.6% | $+7.09 | 9.09 |
| > 1.00 | 6215 | 92.1% | $+8.43 | 15.16 |

## 6. H4 Trend Alignment

| Scenario | N | WR | Net PnL | PF |
|---|---|---|---|---|
| LONG + UP  (with-trend) | 5202 | 87.2% | $+34575.33 | 7.78 |
| LONG + DOWN (counter) | 4717 | 88.8% | $+37117.45 | 9.39 |
| SHORT + DOWN (with-trend) | 7316 | 89.5% | $+55417.15 | 10.57 |
| SHORT + UP  (counter) | 5679 | 91.2% | $+42515.91 | 12.30 |

## 7. Hold Duration

| Hold | N | WR | Avg PnL | Top Exit |
|---|---|---|---|---|
| 0-2h | 3333 | 93.3% | $+9.03 | tp_hit |
| 3-6h | 13301 | 92.5% | $+7.62 | guardian_exit |
| 7-12h | 4723 | 83.6% | $+6.44 | guardian_exit |
| 13-24h | 1557 | 69.7% | $+4.93 | guardian_exit |

## 8. H4 Swing Deviation Distribution

_max(|entry-H4Hi|, |entry-H4Lo|) / entry_

| Swing Dev | N | WR | Net PnL | PF | Note |
|---|---|---|---|---|---|
| <=5% | 22181 | 89.1% | $+157392.99 | 9.56 |  |
| 5-8% | 662 | 91.8% | $+10374.48 | 16.99 |  |
| 8-15% | 71 | 93.0% | $+1858.38 | 41.70 | ⚠️ stale |

## 9. Coin Concentration

| Coin | N | WR | Net PnL | % Total |
|---|---|---|---|---|
| NEARUSDT | 1253 | 88.6% | $+11042.39 | +7% |
| TAOUSDT | 1125 | 89.9% | $+10940.57 | +6% |
| ARBUSDT | 1234 | 87.5% | $+10489.60 | +6% |
| SUIUSDT | 1223 | 90.4% | $+10429.54 | +6% |
| POLUSDT | 1300 | 90.2% | $+10334.50 | +6% |
| 1000PEPEUSDT | 959 | 87.1% | $+9759.51 | +6% |
| ONDOUSDT | 1175 | 89.4% | $+9732.68 | +6% |
| DOGEUSDT | 1164 | 90.5% | $+9309.33 | +5% |
| ADAUSDT | 1220 | 88.7% | $+9161.15 | +5% |
| DOTUSDT | 1190 | 89.3% | $+8885.61 | +5% |
| AVAXUSDT | 1147 | 90.3% | $+8877.36 | +5% |
| LINKUSDT | 1093 | 90.9% | $+8707.07 | +5% |
| HBARUSDT | 1107 | 90.9% | $+8510.38 | +5% |
| SOLUSDT | 1221 | 89.1% | $+8365.56 | +5% |
| 1000SHIBUSDT | 1077 | 89.5% | $+8153.58 | +5% |
| XRPUSDT | 1177 | 88.4% | $+7252.59 | +4% |
| TONUSDT | 1118 | 89.4% | $+6878.89 | +4% |
| ETHUSDT | 996 | 88.2% | $+5885.53 | +3% |
| BNBUSDT | 1107 | 88.7% | $+4731.77 | +3% |
| TRXUSDT | 1016 | 87.6% | $+2141.70 | +1% |
| XAUTUSDT | 12 | 83.3% | $+36.51 | +0% |

## 10. Simulasi Filter P2 + P4

_P2: Vol Regime >= 0.05 | P4: Swing dev <= 8%_

| Metrik | Sebelum | Sesudah | Delta |
|---|---|---|---|
| Jumlah trades | 22914 | 22841 | -73 |
| Win / Loss | 20447/2467 | 20381/2460 | — |
| **Win Rate** | **89.2%** | **89.2%** | **0.0pp** |
| **Net PnL** | **$+169625.85** | **$+167768.77** | **-1857.08 ↓** |
| Gross Profit | $188701.17 | $186797.13 | -1904.04 ↓ |
| Gross Loss | $19075.33 | $19028.37 | -46.96 ↑ |
| **Profit Factor** | **9.89** | **9.82** | **-0.07 ↓** |
| Avg Win | $+9.23 | $+9.17 | -0.06 ↓ |
| Avg Loss | $-7.73 | $-7.74 | -0.01 ↑ |
| Max consec. losses | 27 ($-232.28) | 27 ($-232.28) | +0 trades |

### Trade Diblokir: 73

| | N | Total PnL |
|---|---|---|
| Wins diblokir | 66 | $+1904.04 |
| Losses dicegah | 7 | $-46.96 |
| **Net PnL diblokir** | **73** | **$+1857.08** |

### Detail Trade Diblokir

| Tgl | Coin | Dir | Vol | Hi_dev | Lo_dev | PnL | Alasan Block |
|---|---|---|---|---|---|---|---|
| 2025-05-11 | ARBUSDT | SHORT | 0.800 | 5.0% | 10.7% | $+22.91 ✅ WIN | P4(dev=10.7%) |
| 2025-05-19 | DOGEUSDT | LONG | 0.760 | 8.7% | 1.3% | $+15.22 ✅ WIN | P4(dev=8.7%) |
| 2025-05-19 | DOGEUSDT | LONG | 0.560 | 8.6% | 1.4% | $+14.94 ✅ WIN | P4(dev=8.6%) |
| 2025-05-23 | 1000PEPEUSDT | SHORT | 0.920 | 1.7% | 11.9% | $+44.91 ✅ WIN | P4(dev=11.9%) |
| 2025-05-23 | 1000PEPEUSDT | SHORT | 1.000 | 3.3% | 10.5% | $+37.72 ✅ WIN | P4(dev=10.5%) |
| 2025-05-23 | 1000PEPEUSDT | SHORT | 0.670 | 3.8% | 10.1% | $+35.37 ✅ WIN | P4(dev=10.1%) |
| 2025-06-06 | DOGEUSDT | LONG | 0.510 | 10.2% | 2.2% | $+31.21 ✅ WIN | P4(dev=10.2%) |
| 2025-06-06 | DOGEUSDT | LONG | 0.820 | 9.7% | 2.7% | $+28.50 ✅ WIN | P4(dev=9.7%) |
| 2025-07-18 | XRPUSDT | SHORT | 0.990 | 1.6% | 8.9% | $+20.63 ✅ WIN | P4(dev=8.9%) |
| 2025-07-18 | HBARUSDT | SHORT | 0.640 | 5.5% | 13.0% | $+22.82 ✅ WIN | P4(dev=13.0%) |
| 2025-07-18 | XRPUSDT | SHORT | 0.550 | 2.0% | 8.5% | $+18.68 ✅ WIN | P4(dev=8.5%) |
| 2025-07-18 | XRPUSDT | SHORT | 0.760 | 1.2% | 9.2% | $+22.39 ✅ WIN | P4(dev=9.2%) |
| 2025-07-18 | HBARUSDT | SHORT | 0.460 | 5.0% | 13.4% | $+27.85 ✅ WIN | P4(dev=13.4%) |
| 2025-07-18 | HBARUSDT | SHORT | 0.630 | 6.8% | 12.0% | $+25.73 ✅ WIN | P4(dev=12.0%) |
| 2025-07-18 | DOGEUSDT | LONG | 1.340 | 4.7% | 9.1% | $+18.62 ✅ WIN | P4(dev=9.1%) |
| 2025-08-14 | ADAUSDT | SHORT | 1.200 | 2.8% | 8.9% | $+43.64 ✅ WIN | P4(dev=8.9%) |
| 2025-08-14 | ADAUSDT | LONG | 0.220 | 8.2% | 1.8% | $+30.08 ✅ WIN | P4(dev=8.2%) |
| 2025-09-22 | AVAXUSDT | LONG | 1.350 | 8.4% | 3.7% | $+12.84 ✅ WIN | P4(dev=8.4%) |
| 2025-09-22 | AVAXUSDT | LONG | 0.780 | 8.1% | 4.1% | $+25.39 ✅ WIN | P4(dev=8.1%) |
| 2025-09-22 | TONUSDT | SHORT | 0.550 | 4.7% | 9.7% | $+9.25 ✅ WIN | P4(dev=9.7%) |
| 2025-10-12 | ARBUSDT | LONG | 0.580 | 8.2% | 2.3% | $+22.54 ✅ WIN | P4(dev=8.2%) |
| 2025-10-12 | 1000PEPEUSDT | LONG | 0.820 | 8.5% | 1.4% | $+17.32 ✅ WIN | P4(dev=8.5%) |
| 2025-10-12 | TAOUSDT | LONG | 0.450 | 8.6% | 1.5% | $+26.33 ✅ WIN | P4(dev=8.6%) |
| 2025-10-12 | 1000PEPEUSDT | LONG | 0.770 | 8.8% | 1.1% | $+18.90 ✅ WIN | P4(dev=8.8%) |
| 2025-10-12 | ARBUSDT | LONG | 0.630 | 8.4% | 2.0% | $+23.77 ✅ WIN | P4(dev=8.4%) |
| 2025-10-12 | TAOUSDT | LONG | 0.400 | 8.1% | 2.0% | $+15.15 ✅ WIN | P4(dev=8.1%) |
| 2025-10-14 | TAOUSDT | LONG | 1.210 | 11.8% | 5.1% | $+58.43 ✅ WIN | P4(dev=11.8%) |
| 2025-10-14 | TAOUSDT | LONG | 0.960 | 9.4% | 7.2% | $+46.31 ✅ WIN | P4(dev=9.4%) |
| 2025-10-14 | TAOUSDT | LONG | 1.190 | 8.0% | 8.4% | $+42.47 ✅ WIN | P4(dev=8.4%) |
| 2025-10-14 | TAOUSDT | LONG | 2.380 | 8.4% | 9.2% | $+53.66 ✅ WIN | P4(dev=9.2%) |
| 2025-10-14 | ONDOUSDT | LONG | 1.280 | 8.1% | 5.9% | $+26.86 ✅ WIN | P4(dev=8.1%) |
| 2025-10-14 | BNBUSDT | LONG | 1.510 | 8.2% | 1.5% | $+6.45 ✅ WIN | P4(dev=8.2%) |
| 2025-10-16 | TAOUSDT | SHORT | 1.630 | 2.0% | 8.9% | $+43.82 ✅ WIN | P4(dev=8.9%) |
| 2025-10-16 | TAOUSDT | SHORT | 1.370 | 2.8% | 8.2% | $+40.56 ✅ WIN | P4(dev=8.2%) |
| 2025-11-05 | ETHUSDT | LONG | 1.390 | 8.5% | 5.8% | $+22.92 ✅ WIN | P4(dev=8.5%) |
| 2025-11-08 | NEARUSDT | LONG | 1.120 | 12.6% | 4.8% | $+30.34 ✅ WIN | P4(dev=12.6%) |
| 2025-11-08 | DOTUSDT | LONG | 1.070 | 4.4% | 9.9% | $+30.06 ✅ WIN | P4(dev=9.9%) |
| 2025-11-08 | NEARUSDT | LONG | 0.920 | 9.4% | 7.6% | $+38.99 ✅ WIN | P4(dev=9.4%) |
| 2025-11-08 | NEARUSDT | LONG | 0.680 | 8.6% | 8.2% | $+35.44 ✅ WIN | P4(dev=8.6%) |
| 2025-11-08 | NEARUSDT | LONG | 0.480 | 10.5% | 3.5% | $+44.41 ✅ WIN | P4(dev=10.5%) |
| 2025-11-08 | NEARUSDT | LONG | 0.310 | 9.3% | 4.5% | $+38.80 ✅ WIN | P4(dev=9.3%) |
| 2025-11-20 | NEARUSDT | SHORT | 0.530 | 1.6% | 8.3% | $+11.52 ✅ WIN | P4(dev=8.3%) |
| 2025-12-19 | 1000PEPEUSDT | LONG | 0.490 | 8.4% | 1.4% | $+28.96 ✅ WIN | P4(dev=8.4%) |
| 2025-12-19 | ARBUSDT | LONG | 0.340 | 9.0% | 1.4% | $+34.37 ✅ WIN | P4(dev=9.0%) |
| 2025-12-19 | NEARUSDT | LONG | 0.610 | 12.2% | 0.5% | $+28.19 ✅ WIN | P4(dev=12.2%) |
| 2025-12-19 | 1000PEPEUSDT | LONG | 0.620 | 8.1% | 1.4% | $+27.53 ✅ WIN | P4(dev=8.1%) |
| 2025-12-20 | BNBUSDT | LONG | 0.040 | 0.9% | 1.3% | $-0.65 ❌ LOSS | P2(vol=0.040) |
| 2025-12-20 | BNBUSDT | LONG | 0.040 | 1.1% | 1.1% | $-0.65 ❌ LOSS | P2(vol=0.040) |
| 2026-01-14 | 1000PEPEUSDT | SHORT | 0.840 | 1.5% | 10.5% | $+34.48 ✅ WIN | P4(dev=10.5%) |
| 2026-01-14 | 1000PEPEUSDT | SHORT | 1.080 | 3.5% | 8.7% | $+25.40 ✅ WIN | P4(dev=8.7%) |
| 2026-01-19 | POLUSDT | SHORT | 1.710 | 3.8% | 9.4% | $+16.37 ✅ WIN | P4(dev=9.4%) |
| 2026-01-31 | ONDOUSDT | LONG | 0.770 | 9.1% | 7.5% | $+23.49 ✅ WIN | P4(dev=9.1%) |
| 2026-01-31 | SUIUSDT | LONG | 0.600 | 10.4% | 7.1% | $+17.84 ✅ WIN | P4(dev=10.4%) |
| 2026-01-31 | DOTUSDT | LONG | 0.490 | 4.5% | 8.2% | $-0.65 ❌ LOSS | P4(dev=8.2%) |
| 2026-01-31 | SUIUSDT | LONG | 0.340 | 5.9% | 8.2% | $-0.65 ❌ LOSS | P4(dev=8.2%) |
| 2026-01-31 | TAOUSDT | LONG | 0.360 | 4.0% | 8.2% | $-15.53 ❌ LOSS | P4(dev=8.2%) |
| 2026-01-31 | ARBUSDT | LONG | 0.380 | 4.4% | 8.9% | $-0.65 ❌ LOSS | P4(dev=8.9%) |
| 2026-02-01 | ETHUSDT | SHORT | 0.670 | 4.1% | 8.2% | $+13.62 ✅ WIN | P4(dev=8.2%) |
| 2026-02-05 | POLUSDT | SHORT | 1.100 | 8.7% | 5.6% | $+38.83 ✅ WIN | P4(dev=8.7%) |
| 2026-02-05 | XRPUSDT | SHORT | 1.570 | 8.1% | 6.3% | $-28.18 ❌ LOSS | P4(dev=8.1%) |
| 2026-02-06 | NEARUSDT | LONG | 0.330 | 6.6% | 12.4% | $+36.56 ✅ WIN | P4(dev=12.4%) |
| 2026-02-06 | AVAXUSDT | LONG | 0.410 | 6.1% | 8.6% | $+26.31 ✅ WIN | P4(dev=8.6%) |
| 2026-02-06 | ARBUSDT | LONG | 0.410 | 5.4% | 12.4% | $+36.67 ✅ WIN | P4(dev=12.4%) |
| 2026-02-06 | SOLUSDT | LONG | 0.770 | 7.2% | 12.1% | $+32.83 ✅ WIN | P4(dev=12.1%) |
| 2026-02-06 | TONUSDT | LONG | 0.360 | 7.9% | 9.4% | $+38.98 ✅ WIN | P4(dev=9.4%) |
| 2026-02-06 | 1000PEPEUSDT | LONG | 0.540 | 5.2% | 10.9% | $+42.18 ✅ WIN | P4(dev=10.9%) |
| 2026-02-06 | DOTUSDT | LONG | 0.360 | 5.6% | 11.3% | $+33.34 ✅ WIN | P4(dev=11.3%) |
| 2026-02-06 | TONUSDT | LONG | 0.520 | 6.2% | 10.8% | $+33.82 ✅ WIN | P4(dev=10.8%) |
| 2026-02-06 | TONUSDT | LONG | 0.410 | 6.6% | 10.5% | $+33.10 ✅ WIN | P4(dev=10.5%) |
| 2026-02-06 | ETHUSDT | LONG | 1.080 | 4.0% | 8.0% | $+20.00 ✅ WIN | P4(dev=8.0%) |
| 2026-02-06 | BNBUSDT | LONG | 0.430 | 3.6% | 8.0% | $+18.56 ✅ WIN | P4(dev=8.0%) |
| 2026-03-20 | TAOUSDT | SHORT | 1.150 | 8.5% | 3.8% | $+29.68 ✅ WIN | P4(dev=8.5%) |
| 2026-03-25 | TAOUSDT | SHORT | 0.920 | 1.8% | 8.2% | $+29.16 ✅ WIN | P4(dev=8.2%) |

### Tanpa TONUSDT (apples-to-apples)

| Metrik | Sebelum | Sesudah | Delta |
|---|---|---|---|
| Win Rate | 89.2% | 89.2% | 0.0pp |
| Net PnL | $+162746.95 | $+161005.02 | -1741.93 ↓ |
| Profit Factor | 9.88 | 9.81 | -0.07 ↓ |

## 11. Open Positions

_Tidak ada posisi terbuka._

## 12. 20 Trade Terakhir

| Metrik | Nilai |
|---|---|
| Trades | 20 |
| Win Rate | 60.0% |
| Net PnL | $+126.10 |
| Profit Factor | 25.25 |
| Max consec. losses | 6 ($-3.90) |

| Tgl | Coin | Dir | PnL | Exit |
|---|---|---|---|---|
| 2026-03-31 | BNBUSDT | LONG | ✅ $+7.76 | tp_hit |
| 2026-03-31 | 1000SHIBUSDT | LONG | ✅ $+11.47 | tp_hit |
| 2026-03-31 | LINKUSDT | LONG | ✅ $+12.58 | tp_hit |
| 2026-03-31 | HBARUSDT | LONG | ✅ $+13.27 | tp_hit |
| 2026-03-31 | ETHUSDT | LONG | ✅ $+12.68 | tp_hit |
| 2026-03-31 | ONDOUSDT | LONG | ✅ $+13.47 | tp_hit |
| 2026-03-31 | SUIUSDT | LONG | ✅ $+11.98 | tp_hit |
| 2026-03-31 | POLUSDT | LONG | ✅ $+11.52 | tp_hit |
| 2026-03-31 | NEARUSDT | LONG | ❌ $-0.65 | time_exit |
| 2026-03-31 | SOLUSDT | LONG | ✅ $+14.48 | tp_hit |
| 2026-03-31 | 1000PEPEUSDT | LONG | ✅ $+9.60 | guardian_exit |
| 2026-03-31 | XRPUSDT | LONG | ✅ $+9.33 | tp_hit |
| 2026-03-31 | ADAUSDT | LONG | ❌ $-0.65 | time_exit |
| 2026-03-31 | TONUSDT | SHORT | ✅ $+3.18 | guardian_exit |
| 2026-03-31 | TONUSDT | SHORT | ❌ $-0.65 | time_exit |
| 2026-03-31 | TONUSDT | SHORT | ❌ $-0.65 | time_exit |
| 2026-03-31 | DOTUSDT | SHORT | ❌ $-0.65 | time_exit |
| 2026-03-31 | NEARUSDT | LONG | ❌ $-0.65 | time_exit |
| 2026-03-31 | NEARUSDT | LONG | ❌ $-0.65 | time_exit |
| 2026-03-31 | 1000PEPEUSDT | LONG | ❌ $-0.65 | time_exit |

---
_Generated by tools/trade_analyzer.py — SwingTrade v2_