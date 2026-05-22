# Trade Analysis Report
**Generated:** 2026-05-22
**Data:** 106 closed trades (2026-05-04 s/d 2026-05-21, migrated excluded)
**Hipotesis user:** Model bias ke SHORT padahal market berbalik arah — TERKONFIRMASI

---

## A. SCORECARD

| Metrik | Sebelum Filter | Sesudah Filter | Delta |
|--------|---------------|----------------|-------|
| Trades | 106 | 75 | -31 |
| Win Rate | 51.9% | 57.1% | +5.2% |
| Net PnL | +$230.50 | +$139.02 | -$91.48 |
| Profit Factor | 1.65 | 1.78 | +0.13 |
| Max Streak Loss | 7 | 5 | -2 |

Tanpa TONUSDT (outlier H4 basi): WR 53.5%→58.1%, PnL $+136→$+150, PF 1.51→1.89

### Per Model
| Model | N | WR | Net PnL | Catatan |
|-------|---|-----|---------|---------|
| lstm | 5 LONG | 80% | +$65.85 | Best WR, sample kecil |
| cascade_v2 | 48L/0S | 56% | +$118.91 | Solid saat market bullish |
| cascade_v3 | 1L/30S | 45% | -$1.71 | BERMASALAH — SHORT bias |
| cascade | 22L/0S | 45% | +$47.45 | Legacy, stabil |

---

## B. TEMUAN KRITIS



---

## C. ANALISIS PER MODEL

### [CASCADE_V3] — 31t | WR 45% | PF 0.98 | Net -$1.71 — BERMASALAH

SHORT bias 97%. Breakdown H4 Trend:
- SHORT + H4_UP (counter-trend): 7t, WR 29%, Net **-$14.32**
- SHORT + H4_DOWN (with-trend): 23t, WR 52%, Net **+$19.89**

VolR regime:
- VolR 0.05-0.2: 19t, WR 47%, Net +$1.88
- VolR 0.2-0.5: 6t, WR **17%**, Net **-$16.34** (berbahaya)

Root cause: LGBM ditraining di periode bearish — feature importance condong ke setup SHORT.
Saat May 21 market reversal ke UP, sinyal SHORT terus keluar dan semua loss.

Rekomendasi tuning:
- Naikkan  dari 0.62 ke 0.68
- Turunkan sedikit  ke 0.60
- Atau retrain dengan data balanced (bullish + bearish)

### [CASCADE_V2] — 48t | WR 56% | PF 1.84 | Net +$118.91

Performa terbaik. Bias LONG yang kuat tepat sasaran karena digunakan saat market bullish (May 8-14).
1 SHORT (AVAXUSDT) langsung loss -$8.52. Model tidak fleksibel untuk SHORT — gunakan hanya saat H4 UP.

---

## D. REKOMENDASI PRIORITAS



---

## E. REKONSTRUKSI DAMPAK (cascade_v3)

| Skenario | Trades | Win Rate | Net PnL | PF | Max Streak | vs Baseline |
|----------|--------|----------|---------|-----|------------|-------------|
| Baseline | 31 | 45.2% | -$1.71 | 0.98 | 6 | — |
| R1 (blokir SHORT+H4_UP) | 24 | 50.0% | +$12.61 | 1.30 | 5 | +$14.32 |
| R2 (blokir VolR 0.2-0.5) | 25 | 52.0% | +$14.63 | 1.33 | 4 | +$16.34 |
| R3 (conf >= 0.72) | 20 | 50.0% | +$5.06 | 1.10 | 4 | +$6.77 |
| **R1 + R2 (REKOMENDASI)** | **20** | **60.0%** | **+$33.70** | **2.63** | **4** | **+$35.41** |
| R1 + R3 | 16 | 50.0% | +$6.57 | 1.24 | 4 | +$8.28 |
| R1 + R2 + R3 (best case) | 14 | 57.1% | +$21.23 | 2.62 | 3 | +$22.94 |

**Kesimpulan:**
R1 + R2 adalah kombinasi terbaik — dari -$1.71 menjadi +$33.70, PF dari 0.98 menjadi **2.63**.
Trade count turun 35% (31→20) tapi kualitas jauh meningkat. R3 menambah sedikit nilai tambahan.

---

## F. OPEN POSITIONS

Tidak ada posisi terbuka saat laporan dibuat. Semua May 21 trades sudah closed.

---

## KESIMPULAN

Hipotesis user **TERKONFIRMASI**: cascade_v3 bias SHORT (97% dari 31 trades).
Saat market reversal UP pada May 21, model tetap generate SHORT → 5 losses berturut-turut.

Tindakan mendesak:
1. **R1 + R2 filter** di signal_filter.py — dampak simulasi +$35 dari baseline
2. **Retrain** cascade_v3 dengan data balanced untuk menangkap LONG setup
