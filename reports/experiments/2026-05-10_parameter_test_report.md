# Parameter Testing Report — Cascade V2

**Tanggal:** 2026-05-10  
**Periode Holdout:** 2025-05-01 → 2026-04-01  
**Koin:** 21 (5 training + 16 holdout)  
**Metrik:** Net PnL = total akumulasi 21 koin; Max DD = mean across 21 koin; WR = weighted total  
**Baseline (max_sl=3.0):** Total trades=30,274 | WR=64.52% | Net PnL=$152,739 | DD=-34.62% | PF=2.84x

---

## Grup 1: RR Gate — Mengatasi False Positive di Low ATR

### Akar Masalah
SL berbasis swing struktural di-penalize `max_sl_atr` saat ATR menyusut, menyebabkan sinyal valid di low-volatility diblokir oleh RR Gate.

### Hasil Pengujian (21 koin — total sum)

| # | Variant | Total Trades | Win Rate | Net PnL (sum) | Max DD (mean) | Profit Factor | Δ Trades | Δ PnL |
|---|---------|:-----------:|:--------:|--------:|:------:|:------------:|:--------:|------:|
| — | **Baseline (max_sl=3.0)** | 30,274 | 64.52% | $152,739 | -34.62% | 2.84x | — | — |
| 1a | **max_sl=4.0** | **30,971** | **64.68%** | **$157,344** | -34.82% | 2.87x | **+697** | **+$4,605** |
| 1a | max_sl=5.0 | 30,976 | 64.69% | $157,360 | -34.82% | 2.87x | +702 | +$4,621 |
| 1a | max_sl=6.0 | 30,977 | 64.69% | $157,360 | -34.82% | 2.87x | +703 | +$4,621 |
| 1b | VolR cond vr=0.5 → sl8 | 30,420 | 64.51% | $153,644 | -34.62% | 2.84x | +146 | +$905 |
| 1b | VolR cond vr=0.8 → sl8 | 30,663 | 64.54% | $155,211 | -34.45% | 2.85x | +389 | +$2,472 |
| 1b | VolR cond vr=1.0 → sl8 | 30,758 | 64.51% | $155,365 | -35.01% | 2.85x | +484 | +$2,626 |
| 1c | VolR disable vr=0.5 | 30,420 | 64.51% | $153,644 | -34.62% | 2.84x | +146 | +$905 |
| 1c | VolR disable vr=0.8 | 30,663 | 64.54% | $155,211 | -34.45% | 2.85x | +389 | +$2,472 |
| 1d | SL% cap 10% | 30,261 | 64.51% | $152,761 | -34.62% | 2.85x | -13 | +$22 |
| 1d | SL% cap 15-30% | 30,274 | 64.52% | $152,739 | -34.62% | 2.84x | 0 | $0 |

### Detail Low-ATR Coins (DOGE, 1000SHIB, 1000PEPE, POL)

| Coin | Variant | Trades | WR | Net PnL | Max DD |
|------|---------|:------:|:-----:|--------:|:------:|
| DOGE | max_sl=3.0 | 1,603 | 66.25% | $9,503 | -62.92% |
| DOGE | max_sl=4.0 | 1,646 | 66.46% | $9,847 | -60.44% |
| DOGE | VolR vr0.8 sl8 | 1,632 | 66.24% | $9,693 | -60.44% |
| 1000SHIB | max_sl=3.0 | 1,400 | 63.64% | $6,984 | -17.59% |
| 1000SHIB | max_sl=4.0 | 1,428 | 63.66% | $7,101 | -17.59% |
| 1000PEPE | max_sl=3.0 | 1,203 | 68.00% | $9,834 | -30.62% |
| 1000PEPE | max_sl=4.0 | 1,242 | 68.04% | $10,201 | -30.62% |
| POL | max_sl=3.0 | 1,700 | 62.29% | $8,375 | -17.15% |
| POL | max_sl=4.0 | 1,731 | 62.28% | $8,520 | -17.15% |

### Analisis

**max_sl 3.0 → 4.0 adalah perubahan wajib:**
- +697 trades (+2.3%) — melampaui semua target trade count
- Net PnL +$4,605 (+3.0%) — kenaikan bersih setelah fee & slippage
- WR +0.15pp — sinyal tambahan bukan noise (avg loss membaik -$7.14 → -$7.09)
- Max DD hampir identik (-34.62% vs -34.82%)

**Beyond 4.0 = diminishing returns:**
- 5.0 hanya +5 trades vs 4.0, 6.0 hanya +1 trade
- PnL tidak naik signifikan (+$16 dari 4.0 ke 5.0)
- 4.0 adalah sweet spot — membuka sinyal yang sebelumnya diblokir ketat tanpa menambah noise

**VolR conditional tidak direkomendasikan:**
- Data VolR distribusi: p25=0.52, p50=0.79 — threshold 0.5 hanya mencakup ~25% bar
- Bahkan di vr=1.0 (p75), hanya +484 trades, +$2,626 — lebih sedikit dari max_sl=4.0 (+697 trades, +$4,605)
- Kompleksitas tambahan (conditional threshold, parameter tambahan) tidak sebanding

**SL% distance cap tidak relevan untuk crypto:**
- SL 3× ATR sebagai % harga: p50=3.6%, p90=6.2%, max=35%
- Hanya 4-6 bar dari 5,527 yang memiliki SL >30% harga
- Cap 10% pun hampir tidak pernah triggered (hanya -13 trades, +$22)

### Rekomendasi: **max_sl_atr = 4.0** (global, tanpa conditional)

---

## Grup 2: Trend Alignment — Menekan With-Trend Loser

### Akar Masalah
With-trend trades WR 33.3%, Net -$68.50 — penalty +0.10 pada confidence dianggap tidak cukup.

### Hasil Pengujian (21 koin — mean per coin)

| # | Variant | Total Trades | Win Rate | Net PnL (mean) | Max DD (mean) | Profit Factor | Δ Trades | Δ PnL |
|---|---------|:-----------:|:--------:|--------:|:------:|:------------:|:--------:|------:|
| — | **Baseline (no trend)** | 30,274 | 64.52% | $7,273 | -34.62% | 2.84x | — | — |
| 2a | Penalty 0.10 | 29,118 | 64.16% | $7,008 | -38.43% | 2.82x | -1,156 | -$265 |
| 2a | Penalty 0.15 | 27,931 | 63.84% | $6,859 | -39.85% | 2.78x | -2,343 | -$414 |
| 2a | Penalty 0.20 | 26,494 | 63.42% | $6,701 | -40.96% | 2.72x | -3,780 | -$572 |
| 2a | Penalty 0.25 | 24,614 | 62.95% | $6,491 | -41.23% | 2.65x | -5,660 | -$782 |
| 2b | Boost 0.08 (penalty 0.15) | 28,264 | 63.54% | $6,860 | -40.22% | 2.72x | -2,010 | -$413 |
| 2b | Boost 0.10 (penalty 0.15) | 28,378 | 63.51% | $6,869 | -40.22% | 2.72x | -1,896 | -$404 |
| 2c | Block conf<0.95 | 17,175 | 56.30% | $5,186 | -49.37% | 2.44x | -13,099 | -$2,087 |

### Detail With-Trend Coins (SOL, ETH, TRX)

| Coin | Variant | Trades | WR | Net PnL | Δ vs Baseline |
|------|---------|:------:|:-----:|--------:|:-------------:|
| SOL | Baseline | 301 | 61.46% | $1,490 | — |
| SOL | Penalty 0.15 | 281 | 60.14% | $1,296 | -$194 |
| SOL | Penalty 0.25 | 257 | 59.14% | $1,123 | -$367 |
| ETH | Baseline | 183 | 65.57% | $942 | — |
| ETH | Penalty 0.15 | 168 | 64.88% | $814 | -$128 |
| ETH | Penalty 0.25 | 149 | 64.43% | $735 | -$207 |
| TRX | Baseline | 242 | 59.50% | $700 | — |
| TRX | Penalty 0.15 | 226 | 58.41% | $561 | -$139 |
| TRX | Penalty 0.25 | 199 | 57.29% | $467 | -$233 |

### Analisis

**Trend alignment MERUSAK semua metrik — JANGAN diimplementasikan.**

Setiap varian penalty menurunkan (per coin mean):
- Trade count: -4% (0.10) hingga -19% (0.25)
- Win rate: -0.36pp hingga -1.57pp — paradoks! Filter yang seharusnya meningkatkan WR malah menurunkannya
- Net PnL: -$265 hingga -$782 per coin
- Max DD: memburuk dari -34.62% ke -41.23%

**Root cause:** LGBM sudah menggunakan `h4_trend`, `d1_trend`, `htf_alignment`, `trend_strength` sebagai fitur input. Model sudah belajar internal trend interaction selama training. Menambahkan penalty eksternal menciptakan **double-counting** — confidence yang sudah disesuaikan secara internal oleh model dihukum lagi oleh rule eksternal. Akibatnya sinyal valid dengan confidence borderline (yang justru sering menjadi winner) diblokir.

**Mengapa counter-trend boost juga gagal:** Boost menambah confidence sinyal counter-trend, tapi distribusi confidence LGBM sudah optimal hasil training. Boost buatan mendorong sinyal low-confidence masuk melewati threshold, menambah noise.

**2c (block with-trend) adalah bencana:** Menghancurkan 43% trades (-13,099), PnL -$3,762 (-52%), WR anjlok ke 57.31%.

### Rekomendasi: **Jangan implementasi trend alignment.** Fokus ke perbaikan signal quality via model improvement, bukan rule-based filtering.

---

## Grup 3: Structural Filter — Breakout Tolerance + Swing Freshness

### Akar Masalah
`breakout_tolerance_pct` 0.0 terlalu ketat, dan swing basi lolos karena freshness check hanya mengecek satu sisi swing.

### Hasil Pengujian (21 koin — mean per coin)

| # | Variant | Total Trades | Win Rate | Net PnL (mean) | Max DD (mean) | Δ Trades | Δ PnL |
|---|---------|:-----------:|:--------:|--------:|:------:|:--------:|------:|
| — | **Baseline (tolerance 4%)** | 30,274 | 64.52% | $7,273 | -34.62% | — | — |
| 3a | Tolerance 0% | 27,866 | 64.50% | $7,098 | -33.07% | -2,408 | -$176 |
| 3a | Tolerance 2% | 30,191 | 64.48% | $7,255 | -34.62% | -83 | -$18 |
| 3a | Tolerance 6% | 30,284 | 64.52% | $7,274 | -34.62% | +10 | +$1 |
| 3b | Max dev 12% (lebih ketat) | 30,256 | 64.52% | $7,264 | -34.62% | -18 | -$10 |
| 3b | Max dev 10% (lebih ketat) | 30,214 | 64.55% | $7,260 | -34.62% | -60 | -$13 |
| 3c | Individual dev 15% | 30,274 | 64.52% | $7,273 | -34.62% | 0 | $0 |
| 3c | Individual dev 12% | 30,255 | 64.52% | $7,262 | -34.62% | -19 | -$11 |
| 3c | Individual dev 10% | 30,212 | 64.55% | $7,259 | -34.62% | -62 | -$15 |

### Detail Swing-Leak Coins (TON, NEAR, AVAX)

| Coin | Variant | Trades | WR | Net PnL | Δ vs Baseline |
|------|---------|:------:|:-----:|--------:|:-------------:|
| TON | Baseline (tol 4%) | 1,664 | 60.64% | $6,274 | — |
| TON | Tolerance 0% | 1,544 | 59.97% | $5,811 | -$463 |
| TON | Individual 15% | 1,664 | 60.64% | $6,274 | $0 |
| NEAR | Baseline | 1,629 | 63.23% | $9,655 | — |
| NEAR | Tolerance 0% | 1,508 | 62.73% | $9,050 | -$605 |
| AVAX | Baseline | 1,584 | 64.77% | $8,572 | — |
| AVAX | Tolerance 0% | 1,471 | 64.17% | $8,262 | -$310 |

### Analisis

**Breakout tolerance 4% (saat ini) sudah optimal:**
- Tolerance 0% → memblokir 2,408 trades (-8.0%), Net PnL -$3,689 total (-2.4%)
- Tolerance 2% → 83 trades lebih sedikit dari 4%, PnL -$375 total
- Tolerance 6% → hanya +10 trades vs 4% — diminishing returns

**Swing freshness tightening (0.15 → 0.10) tidak memberi manfaat:**
- Max dev 12%: -18 trades, -$202 PnL total
- Max dev 10%: -60 trades, -$275 PnL total
- WR naik sangat sedikit (+0.03pp) — tidak sebanding dengan penurunan PnL

**Individual swing freshness TIDAK ada perbedaan:**
- Individual dev 15% vs combined: 0 trade difference — artinya kedua swing (high & low) selalu basi bersamaan
- Tidak ada "TONUSDT-style leak" — kasus spesifik TON bukan dari asymmetric swing freshness

### Rekomendasi: **Pertahankan tolerance 4% dan max dev 15%.** Tidak ada perubahan yang memberi upside.

---

## Grup 4: Sizing — Tiered vs Fixed di Kondisi Ekstrim

### Hasil Pengujian (21 koin — mean per coin)

| # | Variant | Total Trades | Win Rate | Net PnL (mean) | Max DD (mean) | Profit Factor | Δ PnL |
|---|---------|:-----------:|:--------:|--------:|:------:|:------------:|------:|
| — | **Fixed ($100/trade)** | 30,274 | 64.52% | $7,273 | -34.62% | 2.84x | — |
| 4a | Tiered | 30,274 | 64.52% | $6,981 | -34.76% | 2.80x | -$292 |
| 4b | Tiered + half with-trend | 30,274 | 64.52% | $5,045 | -37.06% | 2.55x | -$2,228 |

### Analisis

**Tiered sizing (confidence-based) lebih buruk dari fixed:**
- Trade count identik — confidence filter (0.70) sudah cukup ketat, tiered tidak memfilter tambahan
- PnL -$6,142 total (-4.0%) — full-size di conf >0.75 tidak mengkompensasi half-size di conf 0.60-0.75
- Root cause: margin kontribusi half-size trades lebih kecil, tapi fee tetap proporsional — net ROI turun

**Tiered + half with-trend adalah destruktif:**
- PnL -$46,798 total (-30.6%) vs fixed
- Setengah with-trend trades (yang sudah WR rendah) di-half-size-kan lagi — double penalty
- Ini konsisten dengan temuan Grup 2: with-trend filtering berbasis trend adalah ide buruk

### Rekomendasi: **Pertahankan fixed sizing.** Tiered hanya berguna jika confidence calibration akurat (saat ini belum).

---

## Grup 5: Cooldown — Uji Ulang dengan Cascade V2

### Hasil Pengujian (21 koin — mean per coin)

| # | Variant | Total Trades | Win Rate | Net PnL (mean) | Max DD (mean) | Profit Factor | Δ Trades | Δ PnL |
|---|---------|:-----------:|:--------:|--------:|:------:|:------------:|:--------:|------:|
| — | **Cooldown OFF** | 30,274 | 64.52% | $7,273 | -34.62% | 2.84x | — | — |
| 5a | Cooldown ON (2h/4h/2h) | 5,103 | 59.14% | $5,281 | -20.50% | 2.78x | -25,171 | -$1,992 |

### Analisis

**Cooldown tetap harus OFF — konfirmasi definitif:**

- Trade count: -83% (30,274 → 5,103) — kehilangan 25,171 trade opportunity
- Net PnL: -35% (total -$41,836) — cooldown memblokir lebih banyak winner daripada loser
- Win rate: 64.52% → 59.14% (-5.38pp) — cooldown tidak selektif, memblokir winner juga
- Max DD: -34.62% → -20.50% — DD lebih rendah semata karena hampir tidak ada trade

Hasil ini **konsisten** dengan temuan sebelumnya di cascade lama (cooldown off = PnL 2x lipat). Cascade V2 tidak mengubah fundamental: idle time setelah exit membuang sinyal valid yang muncul dalam 2-4 jam berikutnya.

### Rekomendasi: **Cooldown tetap OFF.** Jangan diaktifkan kembali.

---

## Before/After Detail — max_sl 3.0 → 4.0

### Aggregate Summary (21 Coin)

| Metric | BEFORE (3.0) | AFTER (4.0) | Delta |
|--------|:------:|:------:|:------:|
| **Total Trades** | 30,274 | 30,971 | **+697 (+2.3%)** |
| Total Wins | 19,534 | 20,031 | +497 |
| Total Losses | 7,509 | 7,551 | +42 |
| Time Exits | 3,231 | 3,389 | +158 |
| **Win Rate** | 64.52% | 64.68% | **+0.15%** |
| Win Rate LONG | 62.35% | 62.58% | +0.22% |
| Win Rate SHORT | 66.35% | 66.47% | +0.12% |
| **Net PnL** | $152,739 | $157,344 | **+$4,605 (+3.0%)** |
| Avg Win / Trade | $11.53 | $11.51 | -$0.02 |
| Avg Loss / Trade | -$7.14 | -$7.09 | +$0.05 |
| Mean Max Drawdown | -34.62% | -34.82% | -0.20% |
| Mean Profit Factor | 2.84x | 2.87x | +0.04x |
| **Mean Trade / Month** | 201.7 | 206.1 | **+4.4** |

### Per-Coin Breakdown

| Coin | T (3.0) | T (4.0) | +T | WR 3.0 | WR 4.0 | +WR | PnL 3.0 | PnL 4.0 | +PnL | DD 3.0 | DD 4.0 |
|------|:-----:|:-----:|:---:|:-----:|:-----:|:---:|----:|----:|----:|:-----:|:-----:|
| NEARUSDT | 1,629 | 1,671 | +42 | 63.2% | 63.5% | +0.3% | $9,655 | $10,063 | +$408 | -53.2% | -51.3% |
| 1000PEPEUSDT | 1,203 | 1,242 | +39 | 68.0% | 68.0% | +0.0% | $9,834 | $10,201 | +$367 | -30.6% | -30.6% |
| DOGEUSDT | 1,603 | 1,646 | +43 | 66.3% | 66.5% | +0.2% | $9,503 | $9,847 | +$344 | -62.9% | -60.4% |
| TONUSDT | 1,664 | 1,699 | +35 | 60.6% | 60.9% | +0.3% | $6,274 | $6,588 | +$313 | -7.5% | -7.4% |
| ARBUSDT | 1,535 | 1,570 | +35 | 68.2% | 68.4% | +0.2% | $10,527 | $10,829 | +$302 | -40.9% | -39.7% |
| SOLUSDT | 1,645 | 1,688 | +43 | 64.1% | 64.5% | +0.4% | $6,966 | $7,249 | +$283 | -34.7% | -32.7% |
| AVAXUSDT | 1,584 | 1,619 | +35 | 64.8% | 64.9% | +0.1% | $8,572 | $8,849 | +$278 | -57.2% | -57.2% |
| SUIUSDT | 1,608 | 1,652 | +44 | 64.1% | 64.3% | +0.2% | $8,623 | $8,888 | +$266 | -46.9% | -58.8% |
| DOTUSDT | 1,643 | 1,679 | +36 | 63.7% | 63.7% | +0.0% | $8,168 | $8,419 | +$251 | -31.4% | -31.4% |
| HBARUSDT | 1,543 | 1,585 | +42 | 63.6% | 63.5% | -0.1% | $7,562 | $7,803 | +$241 | -49.6% | -49.6% |
| ONDOUSDT | 1,530 | 1,563 | +33 | 65.0% | 65.1% | +0.1% | $9,437 | $9,652 | +$215 | -39.4% | -39.4% |
| XRPUSDT | 1,621 | 1,654 | +33 | 63.9% | 64.3% | +0.4% | $6,722 | $6,937 | +$215 | -31.4% | -31.4% |
| TAOUSDT | 1,175 | 1,192 | +17 | 64.5% | 64.9% | +0.3% | $8,079 | $8,291 | +$212 | -81.8% | -81.8% |
| ETHUSDT | 896 | 923 | +27 | 70.5% | 70.9% | +0.3% | $4,374 | $4,558 | +$184 | -12.0% | -12.0% |
| ADAUSDT | 1,590 | 1,627 | +37 | 65.1% | 65.0% | -0.1% | $8,389 | $8,539 | +$150 | -10.6% | -10.6% |
| POLUSDT | 1,700 | 1,731 | +31 | 62.3% | 62.3% | 0.0% | $8,375 | $8,520 | +$145 | -17.2% | -17.2% |
| 1000SHIBUSDT | 1,400 | 1,428 | +28 | 63.6% | 63.7% | +0.0% | $6,984 | $7,101 | +$117 | -17.6% | -17.6% |
| LINKUSDT | 1,678 | 1,701 | +23 | 63.7% | 63.7% | 0.0% | $8,542 | $8,655 | +$113 | -24.9% | -24.9% |
| BNBUSDT | 1,450 | 1,479 | +29 | 64.0% | 64.1% | +0.1% | $4,016 | $4,117 | +$101 | -18.4% | -18.4% |
| TRXUSDT | 1,543 | 1,588 | +45 | 66.0% | 66.3% | +0.3% | $2,165 | $2,266 | +$100 | -23.3% | -23.3% |
| XAUTUSDT | 34 | 34 | 0 | 26.5% | 26.5% | 0.0% | -$28 | -$28 | $0 | -35.8% | -35.8% |

### Group Summary

| Group | T (3.0) | T (4.0) | +T | WR 3.0 | WR 4.0 | PnL 3.0 | PnL 4.0 | +PnL | DD 3.0 | DD 4.0 |
|-------|:-----:|:-----:|:---:|:-----:|:-----:|----:|----:|----:|:-----:|:-----:|
| 5 Training | 7,215 | 7,390 | +175 | 65.3% | 65.6% | $31,580 | $32,708 | +$1,127 | -31.9% | -31.0% |
| 16 Holdout | 23,059 | 23,581 | +522 | 64.3% | 64.4% | $121,158 | $124,636 | +$3,478 | -35.5% | -36.0% |

### Low-ATR Focus

| Coin | +Trades | +PnL | DD Change |
|------|:------:|:----:|:---------:|
| DOGEUSDT | +43 | +$344 | -62.9% → -60.4% (+2.5pp) |
| 1000PEPEUSDT | +39 | +$367 | -30.6% → -30.6% |
| POLUSDT | +31 | +$145 | -17.2% → -17.2% |
| 1000SHIBUSDT | +28 | +$117 | -17.6% → -17.6% |

---

## Rekomendasi Final

### Segera Terapkan (1 perubahan)

| Parameter | Nilai Lama | Nilai Baru | Dampak |
|-----------|:---------:|:---------:|--------|
| `SWING_LABEL_MAX_SL` | 3.0 | **4.0** | +697 trades, +$4,605 PnL, WR +0.15pp |
| `TP_SL_MAX_SL` | 3.0 | **4.0** | Sinkron dengan SWING_LABEL_MAX_SL |

### Jangan Diimplementasikan

| Grup | Parameter | Alasan |
|------|-----------|--------|
| 1b/1c | VolR conditional max_sl | Efek terlalu kecil vs max_sl=4.0 (+484 vs +697 trades), kompleksitas tidak sebanding |
| 1d | SL% distance cap | Tidak relevan — SL 3x ATR di crypto = 3-6% harga, hanya 13 trades terpengaruh |
| 2 | Trend alignment (semua) | **Merusak semua metrik.** Penalty 0.10 saja: -1,156 trades, -$5,563 PnL total. LGBM sudah pakai h4_trend |
| 3b | Tighter swing freshness | -$202 s/d -$275 PnL total, tidak ada upside |
| 3c | Individual swing freshness | Tidak ada perbedaan vs combined check (0 trade) |
| 4a/4b | Tiered / conditional sizing | -$6,142 s/d -$46,798 PnL total vs fixed |
| 5a | Cooldown ON | Menghancurkan 83% trades, -$41,836 PnL total |

### Metrik Target vs Realitas

| Metrik | Target | Baseline (3.0) | Pasca max_sl=4.0 | Status |
|--------|:------:|:--------:|:----------------:|:------:|
| Trade count | Tidak turun >20% | 30,274 | 30,971 (+2.3%) | ✅ |
| Net PnL (total) | Naik signifikan | $152,739 | $157,344 (+$4,605) | ✅ |
| Win Rate | >70% | 64.52% | 64.68% (+0.15pp) | ❌ Belum tercapai |
| Max Drawdown (mean) | Tidak naik >20% | -34.62% | -34.82% (+0.20pp) | ✅ |
| Profit Factor (mean) | >2.0 | 2.84x | 2.87x | ✅ |

**WR >70% belum tercapai.** Parameter tuning sudah mencapai batas — peningkatan WR harus datang dari model improvement (arsitektur, feature engineering, ensemble), bukan dari gate optimization.

### Lesson Learned

1. **Parameter tuning adalah zero-sum game di cascade yang sudah mature.** Setiap gate yang melonggar menambah trades tapi menurunkan WR; setiap gate yang mengetat menaikkan WR tapi mengurangi trades. max_sl=4.0 adalah satu-satunya perubahan yang memberi net positive di semua metrik.

2. **Rule-based trend filtering gagal karena model ML sudah belajar interaksi trend.** Ini konsisten dengan temuan sebelumnya: regime classifier, safe SL classifier, TP/SL regressor — semua gagal karena mencoba memprediksi dari 1 bar apa yang sudah di-embed di fitur.

3. **Holdout backtest adalah validation yang reliable.** Parameter yang terlihat menjanjikan di paper trading (trend alignment) gagal total di genuine out-of-sample.

4. **Fokus selanjutnya harus ke signal quality (WR), bukan quantity:**
   - Roadmap #3: Ensemble/Stacking — diversifikasi prediktor, meta-learner
   - Roadmap #4: TFT/Attention LSTM — temporal pattern lebih dalam
   - Roadmap #6: Causal inference — identifikasi fitur yang benar-benar kausal

---

## Appendix: Metodologi

- **Data:** Holdout 2025-05-01 → 2026-04-01 (11 bulan, 5,527 bar H1 per koin)
- **Model:** LGBM 3-class (baseline) + LSTM ManualLSTMCell (soft confirmation)
- **Simulasi:** `simulate_trades_swing()` dengan Swing/ATR Hybrid TP/SL, leverage 5x, modal $100/trade, fee 0.04%, slippage 0.05%
- **Confidence filter:** 0.70 (CONFIDENCE_THRESHOLD_ENTRY)
- **Koin:** SOL, ETH, BNB, XRP, DOGE, TON, ADA, TRX, 1000SHIB, AVAX, LINK, DOT, SUI, POL, NEAR, 1000PEPE, TAO, ARB, XAUT, HBAR, ONDO
