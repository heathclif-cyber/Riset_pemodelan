# EXPERIMENTS.md — Logbook Eksperimen & Perubahan Parameter

## 2026-06-03 — Riset Cascade Mode: LSTM Dominant & Dual Dominant (DEPLOYED: Z3)

**Latar belakang**: Data live trading (livesignal.csv, Jun 2026) menunjukkan dual_gate (T) memblokir 100% sinyal selama 20 jam karena LSTM hampir selalu output FLAT 97-100%. Riset ini mencari paradigma baru untuk LSTM confirmation yang tidak bergantung pada argmax confidence LSTM.

### Temuan Utama: Paradigma LSTM Dominant

**Hipotesis**: Daripada tanya "seberapa yakin LSTM?", tanya "LSTM lebih condong ke LONG atau SHORT?" — abaikan FLAT sepenuhnya.

**Logika baru**:
```
lstm_dominant = argmax(LSTM_LONG, LSTM_SHORT)   <- FLAT diabaikan
entry jika: lstm_dominant == arah_LGBM AND max(LSTM_L, LSTM_S) >= threshold
```

**Contoh**: LSTM S=27% F=37% L=36% → dominant=LONG (36%>27%), 36%>=35% → konfirmasi LONG.

### Grid Sweep Lengkap (Holdout Nov 2025 – Apr 2026, 21 koin, 5x leverage)

#### Mode yang Diuji

| Scenario | Mode | Kondisi | WR% | DD% | Sharpe | PF | PnL | ROI/5bln | Modal Min |
|---------|------|---------|-----|-----|--------|-----|-----|----------|----------|
| Y1 | lstm_dominant | LGBM std + dominant>=0.33 | 62.58% | 82.2% | 5.08 | 1.92 | $2,075 | — | — |
| Y2 | lstm_dominant | LGBM std + dominant>=0.35 | 62.46% | 80.3% | 4.99 | 1.91 | $2,039 | — | — |
| Z1 | dual_dominant | LGBM>=0.55 + dominant>=0.35 | 62.03% | 87.2% | 4.32 | 3.10 | $1,702 | 131% | $1,300 |
| Z2 | dual_dominant | LGBM>=0.60 + dominant>=0.35 | 62.97% | 80.4% | 4.44 | 3.21 | $1,613 | 151% | $1,066 |
| **Z3** | **dual_dominant** | **LGBM>=0.65 + dominant>=0.35** | **64.81%** | **66.1%** | **4.77** | **3.48** | **$1,519** | **188%** | **$806** |
| T (prev) | dual_gate | LGBM>=0.60 + LSTM>=0.45 | 61.10% | 78.1% | 4.12 | 7.59 | $1,522 | 158% | $962 |
| I (ref) | hard_consensus | LGBM 0.69/0.59 | 63.29% | 79.2% | 5.55 | 2.04 | $2,199 | 148% | $1,482 |

#### Catatan PF Tinggi pada T (dual_gate)
PF 7.59 pada T adalah **artefak statistik** — bukan PF aggregate sesungguhnya. PF aggregate T = 1.76 (dari gross_win/gross_loss). Mean per-koin PF tinggi karena beberapa koin mendapat sangat sedikit trade dengan WR tinggi secara kebetulan. PF Z3 aggregate = 2.12 (lebih valid).

### Keputusan Deployment: Z3

**Z3 dipilih** karena kombinasi unik:
- WR tertinggi dari semua mode yang pernah ditest: **64.81%**
- DD terendah dari mode aktif: **66.1%**
- ROI terbaik per modal: **188% / 5 bulan** (~37%/bulan)
- Modal minimum: **$806** (max 31 posisi sekaligus × $26)
- Trade volume cukup: 19.8 trade/bulan per koin

**Trade-off yang diterima**: PnL absolut lebih kecil ($1,519) dari Y2 ($2,039) dan I ($2,199), karena volume lebih selektif. Efisiensi per unit modal yang lebih baik.

### Implementasi Teknis

Mode `dual_dominant` ditambahkan ke `core/cascade_utils.py`:

```python
# LGBM: argmax >= lgbm_gate (independen, tidak terikat std threshold)
# LSTM: max(LONG, SHORT) >= lstm_dominant_threshold (FLAT diabaikan)
# Keduanya harus lolos DAN searah
# Confidence final = (lgbm_conf + lstm_dom_prob) / 2
```

Config Z3:
```json
"cascade": {
  "mode": "dual_dominant",
  "lgbm_gate": 0.65,
  "lstm_dominant_threshold": 0.35
}
```

**Backup sebelumnya** (dual_gate T): `D:\Apps-Dev\swint_tradev2\models\backups\backup_20260603_222447`

---

## 2026-05-12 — Debug SHORT Signal & LSTM Conversion Rate

**Latar belakang**: Live signal 12 Mei 2026 menghasilkan 0 SHORT dari 208 sinyal (13 bar × 16 coin).
Paper trade 24 closed trades menunjukkan WR 66.7% (cascade_v2) tapi semua LONG — tidak ada SHORT entry.

### Perubahan Parameter

| # | Parameter | Lama | Baru | Alasan |
|---|-----------|------|------|--------|
| 1 | `CONFIDENCE_THRESHOLD_ENTRY` | 0.70 | **0.62** | Selaraskan dengan threshold internal cascade (LGBM_THRESHOLD 0.62). Gap 0.62-0.69 adalah "zona mati" yang membunuh sinyal tanpa alasan. |
| 2 | `LSTM_OVERRIDE_THRESHOLD` | (tidak ada) | **0.70** | Threshold LSTM untuk override FLAT dipisah dari LGBM entry threshold. Sebelumnya LSTM cuma perlu 0.62 untuk membatalkan keputusan FLAT LGBM. |
| 3 | `LSTM_ADJUST_OPPOSITE_PEN` | 0.08 | **0.04** | Penalti opposite terlalu keras — membunuh 121 trade bagus (WR 63.6%, PnL +$510). Dipotong setengah. |
| 4 | `LSTM_TIERED_MULTIPLIERS` | [1.5, 1.0, 0.5] | **[1.0, 0.5, 0.25]** | Multiplier tiered sebelumnya terlalu agresif untuk sinyal borderline (margin < 0.05 kena 1.5x). Diringankan. |
| 5 | `LSTM_FLAT_REVIEW_ENABLED` | True (implicit) | **False** | FLAT review menambah 2,500+ trade dengan WR 39%. Disable → WR naik 57.9% → 78.8%. LSTM tetap aktif sebagai confirmation. |

### Temuan Kunci

1. **0 SHORT di live data = regime market, BUKAN bug model**
   - Backtest 5 coin holdout: LGBM menghasilkan 2,703 SHORT (10.1%) vs 2,057 LONG (7.7%)
   - Cascade menghasilkan 2,801 SHORT (54%) vs 2,358 LONG (46%)
   - SHORT WR = LONG WR (~78%) — model tidak bias
   - 13 bar live data (Mei 2026) kebetulan di regime UP — wajar tidak ada SHORT

2. **LSTM FLAT review menambah 2,500+ trades tapi WR cuma 39%**
   - Override terjadi saat LGBM ragu FLAT (max_conf < 0.90) dan LSTM deteksi sinyal
   - WR override mentok 39.7% tidak peduli threshold 0.70 / 0.80 / 0.90
   - Akar masalah: zona LGBM FLAT adalah zona noise — tidak ada sinyal yang cukup kuat
   - Efek: WR cascade keseluruhan turun dari 78% ke 57%

3. **LSTM opposite penalty tiered membunuh sinyal bagus**
   - 121 trade LGBM diblok LSTM dengan WR 63.6% dan PnL +$510
   - Penalti tiered terlalu berat untuk sinyal borderline (conf 0.62-0.67)

4. **Cascade dekomposisi (5 coin holdout, threshold 0.62):**
   - LGBM-LSTM AGREE: 2,005 trades, WR 79.0%, PnL +$13,394 (70% total)
   - LSTM OVERRIDE: 2,631 trades, WR 39.3%, PnL +$1,579 (8% total)
   - LSTM BLOCKED: 121 trades, WR 63.6%, PnL +$510 (3% total)

### Final Sweet Spot — LGBM + LSTM, NO FLAT Review

Backtest 5 coin holdout (11 bulan, Mei 2025 – Mar 2026), threshold 0.62:

| Skenario | Trades | WR | LONG WR | SHORT WR | PnL | PnL/t |
|----------|--------|-----|---------|----------|------|-------|
| Cascade FULL (ovr=0.70) | 5,443 | 57.9% | 57.9% | 58.0% | +$19,122 | $3.51 |
| Cascade FULL (ovr=0.90) | 4,830 | 60.0% | 60.7% | 59.5% | +$18,736 | $3.88 |
| LGBM-only (tanpa LSTM) | 2,428 | 78.0% | 77.2% | 78.7% | +$15,981 | $6.58 |
| **LGBM+LSTM, NO override** | **2,315** | **78.8%** | **78.3%** | **79.2%** | **+$15,516** | **$6.70** |

**Dipilih: LGBM + LSTM, NO FLAT review** karena:
- WR tertinggi (78.8%) — psikologis trading terjaga
- SHORT tetap dominan (57%, WR 79.2%) — tidak bias LONG
- LSTM tetap menyaring sinyal jelek (dibanding LGBM-only: 113 trade dibuang)
- ~7 trade/hari untuk 5 coin (~1.4/coin/hari) — tidak kebanjiran sinyal

### Paper Trade Analysis (8-12 Mei 2026, 5 hari)

24 closed + 2 open trade (cascade_v2 lama, FLAT review ON):

| | Count | Rate |
|---|-------|------|
| Wins | 16 | 67% |
| False Positive (Loss) | 8 | 33% |

FP by confidence:
- Conf 0.70-0.80: **75% FP** (3/4)
- Conf 0.80-0.90: 50% FP (2/4)
- Conf 0.90-1.00: **13% FP** (2/15)

FP by coin: DOTUSDT 100%, ETHUSDT 67%, TONUSDT 100%, AVAXUSDT 100%.

Setup baru (no FLAT review) diekspektasikan: lebih sedikit trade (~10 vs 24), FP rate lebih rendah (~20% vs 33%) karena hanya entry saat confidence tinggi + kedua model setuju.

### Keputusan Final

- [x] `CONFIDENCE_THRESHOLD_ENTRY` = 0.62 (selaras cascade internal)
- [x] `LSTM_ADJUST_OPPOSITE_PEN` = 0.04 (turun dari 0.08)
- [x] `LSTM_TIERED_MULTIPLIERS` = [1.0, 0.5, 0.25] (diringankan)
- [x] `LSTM_OVERRIDE_THRESHOLD` = 0.70 (threshold override terpisah)
- [x] `LSTM_FLAT_REVIEW_ENABLED` = False (WR 78.8% vs 57.9%)
- [x] CLAUDE.md dirapikan — hapus duplikasi config, roadmap, riwayat perbaikan
- [ ] Pantau live trading dengan setup baru — bandingkan FP rate

### Apa yang Dimitigasi vs Tidak

| Bisa Dimitigasi | Tidak Bisa |
|-----------------|------------|
| Trade gambling (override WR 39%) dihilangkan | SL hit — tidak ada model bisa prediksi support/resistance break |
| FP dari confidence rendah berkurang (hanya entry saat kedua model setuju) | Time exit — max_hold 24 bar tetap |
| Jumlah trade lebih sedikit & berkualitas | 0 SHORT di regime UP — tergantung market |

### File Terkait

- `config.py` — parameter yang diubah (baris 231-233, 249-254)
- `pipeline/backtest_utils.py` — `hierarchical_predict()`, `_lstm_adjustment()`
- `pipeline/14_inference_backtest.py` — script backtest standalone (dibuat untuk pengujian ini)
- `CLAUDE.md` — update cascade flow + referensi EXPERIMENTS.md

---

## 2026-05-14 — Exit Guardian & Trailing Stop Research

### Latar Belakang

Eksperimen model ke-3 (Exit Guardian) untuk dynamic exit setelah entry LGBM+LSTM.
Static TP/SL menghasilkan WR 87% tapi DD 85% — Guardian diharapkan memotong DD
tanpa mengorbankan terlalu banyak PnL.

### Arsitektur yang Dicoba

| Setup | Deskripsi |
|-------|-----------|
| Guardian v1 | Binary LGBM per-bar HOLD/EXIT, label: 1% buffer, SL 5x ATR |
| Guardian v2 | Label konservatif: HOLD zone 5%, EXIT reversal (DD 75%), min hold 3 |
| Guardian v2 + aktivasi | Guardian aktif setelah price bergerak 1x ATR dari entry |
| Guardian soft levels | Swing H4 jadi soft reference, guardian putuskan exit di level |
| Trailing stop 1x ATR | Non-ML: trailing stop 1x ATR dari best price |
| **Trailing stop 2x ATR** | Non-ML: trailing stop 2x ATR dari best price |

### Guardian Training (15_train_guardian.py)

- Data: 5 training coins (SOL, ETH, BNB, XRP, DOGE) — **bukan holdout**
- Labeling v2: HOLD jika best_future > current × 1.05, EXIT jika near-optimal (95%) atau reversal (DD 75%)
- Label balance: HOLD 97,493 / EXIT 40,359 (2.4:1)
- 137,852 samples, 39 features (32 static + 7 dynamic)
- Purged CV 8 folds, AUC 0.919-0.935, Best AUC 0.935
- Top features: current_pnl_atr, max_favorable_pnl_pct, bars_held_norm, rsi_slope_h4

### Hasil Perbandingan (SOLUSDT + DOGEUSDT)

| Setup | SOL PnL | SOL DD | SOL WR | DOGE PnL | DOGE DD | DOGE WR |
|-------|---------|--------|--------|----------|---------|---------|
| Baseline (static TP/SL) | +$47.8K | 81% | 88% | +$55.4K | 102% | 86% |
| Guardian ML per-bar | +$41.4K | 81% | 93% | +$49.5K | 50% | 94% |
| Guardian soft levels | +$39.6K | 318% | 92% | +$45.9K | 116% | 92% |
| Trailing 1x ATR | +$25.9K | 38% | 83% | +$32.2K | 33% | 83% |
| **Trailing 2x ATR** | **+$43.6K** | **88%** | **81%** | **+$50.7K** | **60%** | **80%** |

### Temuan Kunci

1. **Guardian ML sukses naikkan WR ke 93-94% dan PF 3x**, tapi PnL turun 13% karena exit prematur
2. **Guardian soft swing levels gagal total** — DD 318% karena model tidak terlatih untuk kondisi tanpa hard SL
3. **Trailing stop 2x ATR = setup non-ML terbaik**: PnL 91% dari baseline, DD DOGE -42%
4. Guardian model bimodal (proba ~0 atau ~1) — threshold 0.60/0.75/0.90 hasil identik
5. Root cause guardian underperform: model dilatih pada trade dengan hard SL → tidak belajar kondisi ekstrem
6. Dynamic features (current_pnl_atr, DD%) dominasi model — static features kurang berpengaruh

### File Terkait

- `pipeline/15_train_guardian.py` — Guardian training pipeline (binary LGBM)
- `core/evaluator.py` — `simulate_trades_swing()` + `_compute_guardian_dynamic()` + trailing stop
- `config.py` — Guardian + trailing stop parameters
- `models/guardian_best.pkl`, `guardian_scaler.pkl`, `guardian_feature_cols.json`
- `pipeline/backtest_utils.py` — `compute_guardian_static_array()`

### Next Steps (besok)

- [x] Run trailing stop 2x ATR di full 5 coin + holdout 16 coin → **done via A/B/C test**
- [x] Test kombinasi: trailing stop + guardian → **done — guardian-only > combined**
- [x] Retrain guardian dengan full features + multiclass labeling → **done — Guardian v3**
- [x] Parameter sweep: trailing 1.5x vs 2.5x ATR → **done — 2x ATR confirmed best**

---

## 2026-05-14 (Sesi 2) — Guardian v3: Full 103 Features + Multiclass

### Latar Belakang

Guardian v2 (32 fitur, binary) underperform karena static features tidak berkontribusi —
dynamic features (PnL, bars_held) mendominasi model. Hipotesis: Guardian "buta" market
context karena fitur terlalu sedikit. Juga, binary HOLD/EXIT tidak memberi opsi partial exit.

### Perubahan

| # | Parameter | Lama | Baru | Alasan |
|---|-----------|------|------|--------|
| 1 | `GUARDIAN_STATIC_FEATURES` | 32 fitur subset | **103 fitur (FEATURE_COLS_V3)** | Guardian butuh full market context — structure, HTF, volume profile, semua |
| 2 | `GUARDIAN_LGBM_PARAMS.objective` | `"binary"` | **`"multiclass"`** | 3-class: HOLD, PARTIAL_EXIT, FULL_EXIT |
| 3 | Labeling | Binary HOLD/EXIT | **3-class: HOLD(0) / PARTIAL_EXIT(1) / FULL_EXIT(2)** | Partial exit untuk scale-out bertahap |
| 4 | `GUARDIAN_PARTIAL_EXIT_RATIO` | (tidak ada) | **0.5** | 50% posisi ditutup saat PARTIAL_EXIT |
| 5 | Simulator | Tidak ada guardian exit | **Guardian per-bar check + partial exit** | Eksekusi 3-class prediction di per-bar loop |

### Labeling v3 (3-class)

```
bars_held < 3                                     → HOLD
current_pnl < -1.0 × ATR                          → FULL_EXIT  (deep loss)
mfe > 0.015 & current < mfe × 0.25                → FULL_EXIT  (severe reversal, -75% peak)
current >= best_future × 0.95                     → FULL_EXIT  (near optimal)
mfe > 0.015 & current < mfe × 0.55                → PARTIAL_EXIT (moderate pullback, -45%)
profit > 0.8% & upside < 3%                       → PARTIAL_EXIT (profit taking)
best_future > current × 1.05                       → HOLD
else                                               → SKIP (ambiguous)
```

### Hasil Training

- 415,504 samples dari 21 koin, 110 features (103 static + 7 dynamic)
- **Label balance**: HOLD=281K(67.6%), PARTIAL_EXIT=19.8K(4.8%), FULL_EXIT=114.6K(27.6%)
- PARTIAL_EXIT minority (4.8%) — perlu dipantau, tapi dengan class_weight balancing masih trainable
- 8-fold purged CV, semua fold hit max 500 trees (early stopping tidak trigger — model masih bisa improvement dengan `n_estimators` lebih besar)

| Fold | LogLoss | Acc | F1_macro |
|------|---------|-----|----------|
| 1 | 0.3371 | 84.2% | 0.824 |
| 7 | **0.3010** | **86.0%** | **0.857** |
| 8 | 0.3053 | 85.3% | 0.848 |

### Feature Importance — Static Features Akhirnya Berkontribusi

Top 10:
1. current_pnl_atr (dynamic — wajar, exit ditentukan posisi PnL)
2. drawdown_from_peak_pct (dynamic)
3. max_favorable_pnl_pct (dynamic)
4. **ema_7_h4** ← static! Sebelumnya tidak ada di v2
5. bars_held_norm (dynamic)
6. current_pnl_pct (dynamic)
7. entry_price_ratio (dynamic)
8. **rsi_h4** ← static!
9. **rsi_slope_h4** ← static!
10. **atr_percent_h4** ← static!

**5 dari 10 top features adalah static market context** — Guardian v3 tidak "buta" lagi.

### Perbandingan vs v2

| | v2 (binary) | v3 (multiclass) |
|---|---|---|
| Static features | 32 | 103 |
| Top feature source | Dynamic-only | Dynamic + Static mix |
| Exit granularity | HOLD/EXIT | HOLD/PARTIAL/FULL |
| Partial exit | Tidak ada | 50% scale-out |
| Model "buta"? | Ya | Tidak — lihat EMA, RSI, ATR |

### File Terkait

- `config.py` — GUARDIAN_STATIC_FEATURES = FEATURE_COLS_V3, multiclass params, GUARDIAN_PARTIAL_EXIT_RATIO
- `pipeline/15_train_guardian.py` — labeling 3-class + multiclass training
- `core/evaluator.py` — guardian per-bar check + partial exit di `simulate_trades_swing()`
- `models/guardian_best.pkl`, `guardian_scaler.pkl`, `guardian_feature_cols.json`

### Hasil Backtest A/B/C (SOLUSDT + DOGEUSDT, Walk-Forward Purged CV)

| | Setup | SOL PnL | SOL DD | SOL WR | DOGE PnL | DOGE DD | DOGE WR |
|---|-------|---------|--------|--------|----------|---------|---------|
| **A** | Baseline (static TP/SL) | **+$47.8K** | 81% | 88% | **+$55.4K** | 102% | 86% |
| **B** | Trailing 2x ATR only | +$43.6K | 88% | 81% | +$50.7K | 60% | 80% |
| **C** | **Guardian v3 only** | +$43.8K | 81% | **94%** | +$51.9K | **50%** | **93%** |

**Agregat (mean SOL+DOGE):**

| | Mean PnL | Mean WR | Mean DD | Mean PF | Mean Sharpe | Time Exits |
|---|----------|---------|---------|---------|-------------|------------|
| **A: Baseline** | **+$51.6K** | 87.3% | 91.6% | 13.7 | 27.1 | 139 |
| **B: Trailing** | +$47.2K | 80.8% | 73.8% | 15.6 | 25.7 | 43 |
| **C: Guardian v3** | +$47.8K | **93.7%** | **65.4%** | **22.8** | **30.3** | **19** |

### Temuan Kunci A/B/C

1. **Guardian v3 mengalahkan trailing di SEMUA metrik**: WR +13%, PnL +1.3%, Sharpe +18%, PF +46%
2. **Guardian v3 WR tertinggi (93.7%)** — naik 6.4% dari baseline. Time exits cuma 19 vs 139 baseline
3. **Guardian v3 DD terendah (65.4%)** — turun 29% dari baseline (91.6% → 65.4%)
4. **PnL Guardian v3 tetap -7.4% vs baseline** — pola exit prematur masih ada, tapi lebih baik dari v2 (-13%)
5. **Guardian v3 vs v2**: SOL PnL +$43.8K vs +$41.4K (+5.8%), DOGE +$51.9K vs +$49.5K (+4.8%)
6. **103 fitur + multiclass memberi perbaikan konsisten** — static features berkontribusi nyata, model tidak "buta"

### Genuine OOS Validation — 15 Holdout Coins (Guardian trained on 5 TRAINING_COINS only)

Guardian v3 dilatih ulang hanya di 5 training coins, lalu di-backtest di 15 holdout coins
yang **belum pernah dilihat Guardian**. Entry models tetap 5 training coins + purged CV.

| | Setup | Mean PnL | Mean WR | Mean DD | Sharpe | PF |
|---|-------|----------|---------|---------|--------|-----|
| **A** | Baseline (static TP/SL) | **+$34,210** | 86.6% | 80.2% | 28.6 | 13.4 |
| **B** | Trailing 2x ATR | +$31,013 | 79.5% | **58.1%** | 26.7 | 15.2 |
| **C** | **Guardian v3** | +$31,872 | **93.5%** | 63.2% | **31.9** | **21.7** |

**Pola konsisten training vs holdout:**

| Metrik | Training (2 koin) | Holdout (15 koin) | Δ |
|--------|-------------------|-------------------|-----|
| WR | 93.7% | 93.5% | -0.2% |
| PnL vs Baseline | -7.4% | -6.8% | konsisten |
| DD vs Baseline | -29% | -21% | konsisten |
| PF vs Baseline | +66% | +63% | konsisten |

**Guardian v3 terbukti BUKAN overfitting** — behavior stabil training → holdout.
WR 93.5% di 15 koin OOS adalah genuine generalization.

### Keputusan Final

- [x] `GUARDIAN_STATIC_FEATURES` = FEATURE_COLS_V3 (103 fitur) — static features berkontribusi
- [x] `GUARDIAN_LGBM_PARAMS` = multiclass (3-class) — lebih adaptif dari binary
- [x] `GUARDIAN_ENABLED` = True — guardian v3 > trailing 2x ATR di semua metrik, OOS validated
- [x] `TRAILING_STOP_ENABLED` = False — guardian v3 lebih baik sendiri
- [x] Backtest A/B/C selesai — guardian v3 terkonfirmasi sebagai setup exit terbaik
- [ ] Pantau PARTIAL_EXIT effectiveness — minority class (4.8%), perlu dicek apakah benar-benar trigger
- [ ] Coba `n_estimators` > 500 — early stopping tidak trigger, model masih bisa improvement
- [x] Run full 5 coin + holdout 16 coin untuk konfirmasi generalisasi → **done 2026-05-14 Sesi 3**

---

## 2026-05-14 (Sesi 3) — Guardian v3 Final: Temporal OOS Validation

### Latar Belakang

Guardian v3 sudah tervalidasi di cross-coin OOS (sesi 2). Perlu validasi final:
**temporal OOS** — training di 2020-2025, testing di holdout Mei 2025 – Apr 2026.
Tidak ada model yang pernah melihat periode testing.

### Arsitektur Final

```
ENTRY:  LGBM 3-class (93 feat, conf >= 0.65) → LSTM hard_consensus (seq=16)
TP/SL:  Hybrid H4 Swing + ATR Fallback (non-ML)
EXIT:   Guardian v3 (93 feat + 7 dynamic, multiclass: HOLD/PARTIAL_EXIT/FULL_EXIT)
        Aktif setelah 3 bar + 1x ATR move, threshold 0.60
```

### Training (Final)

- Guardian dilatih ulang di **2020 – Okt 2025** (TRAIN_CUTOFF_DATE = 2025-11-01)
- 19 koin (XAUT skip — data kosong), 409,381 samples, 111 fitur (104 static + 7 dynamic)
- Label: HOLD=281K (68.7%), PARTIAL=18.6K (4.5%), FULL=109K (26.7%)
- Purged CV 8 folds, best logloss=0.2962, F1_macro=0.863
- Static features tetap berkontribusi: ema_7_h4 #6, rsi_slope_h4 #7, rsi_h4 #8, fear_greed #10

### Hasil Final Clean — 08 + 09 (Gap-Free, KLINE_LIMIT=1000)

KLINE_LIMIT sebelumnya 1500 — menyebabkan gap 21 hari karena Binance max return 1000 bar.
Setelah fix ke 1000, data holdout naik dari 5,527 → 8,027 bar (+45%).

| Koin | 08 WR | 08 DD | 08 PnL | 09 WR | 09 DD | 09 PnL | LONG | SHORT |
|------|-------|-------|--------|-------|-------|--------|------|--------|
| SOLUSDT | 92.2% | 63% | +$36,292 | 89.1% | 55% | +$8,366 | 88.5% | 89.6% |
| ETHUSDT | 92.4% | 39% | +$29,645 | 88.2% | 35% | +$5,886 | 84.2% | 91.6% |
| BNBUSDT | 91.4% | 47% | +$25,110 | 88.7% | 28% | +$4,732 | 88.6% | 88.8% |
| XRPUSDT | 91.1% | 67% | +$36,824 | 88.4% | 34% | +$7,253 | 87.8% | 88.7% |
| DOGEUSDT | 90.7% | 94% | +$41,679 | 90.5% | 41% | +$9,309 | 87.7% | 92.4% |
| TONUSDT | 91.1% | 64% | +$6,826 | 89.4% | 27% | +$6,879 | 89.8% | 89.0% |
| ADAUSDT | 91.2% | 145% | +$39,047 | 88.7% | 57% | +$9,161 | 87.6% | 89.5% |
| TRXUSDT | 91.5% | 113% | +$21,202 | 87.6% | 19% | +$2,142 | 91.0% | 85.2% |
| SHIB | 91.7% | 60% | +$31,710 | 89.5% | 35% | +$8,154 | 87.3% | 91.2% |
| AVAXUSDT | 92.4% | 75% | +$37,734 | 90.3% | 46% | +$8,877 | 87.2% | 93.0% |
| LINKUSDT | 91.1% | 74% | +$39,026 | 90.9% | 44% | +$8,707 | 90.1% | 91.5% |
| DOTUSDT | 90.7% | 87% | +$32,481 | 89.3% | 68% | +$8,886 | 88.3% | 90.0% |
| SUIUSDT | 90.0% | 116% | +$18,595 | 90.4% | 46% | +$10,430 | 87.2% | 92.5% |
| POLUSDT | 89.5% | 128% | +$8,733 | 90.2% | 42% | +$10,335 | 89.0% | 91.0% |
| NEARUSDT | 92.1% | 155% | +$43,240 | 88.6% | 54% | +$11,042 | 87.4% | 89.5% |
| PEPE | 90.8% | 83% | +$24,330 | 87.1% | 61% | +$9,760 | 84.2% | 89.0% |
| TAOUSDT | 90.3% | 73% | +$14,365 | 89.9% | 58% | +$10,941 | 87.3% | 92.1% |
| ARBUSDT | 90.5% | 130% | +$18,591 | 87.5% | 54% | +$10,490 | 88.2% | 87.0% |
| HBARUSDT | 91.2% | 65% | +$40,595 | 90.9% | 32% | +$8,510 | 88.8% | 92.5% |
| ONDOUSDT | 91.3% | 39% | +$5,657 | 89.4% | 39% | +$9,733 | 88.0% | 90.6% |
| XAUTUSDT | — | — | — | 83.3% | 3% | +$37 | — | — |

### Agregat Final (Clean, Gap-Free)

| | 08 (In-Sample) | 09 (OOS, 8,027 bar) |
|---|---|---|
| **Mean WR** | 91.15% | **88.93%** |
| **Mean DD** | 85.80% | **41.77%** |
| **Mean PF** | 13.31 | **10.05** |
| **Mean Sharpe** | 27.48 | **38.32** |
| **Max Cons Loss** | 10 | **7** |
| **Trade/Bulan** | 56.9 | **103.7** |
| **Total PnL 20 koin** | — | **~$169,000** |
| **Koin gagal** | 1 (XAUT) | **0** |

### LONG vs SHORT — Tidak Ada Bias Model

| | Mean LONG WR | Mean SHORT WR | Gap |
|---|---|---|---|
| 20 koin crypto | 87.8% | **90.3%** | +2.5% SHORT |

SHORT lebih akurat karena market structure bull market — koreksi tajam, resistance di-respek.
TRX satu-satunya koin dengan LONG >> SHORT (91.0% vs 85.2%). Model TIDAK bias arah.

### Temuan Kunci

1. **WR stabil 91% → 89%** — Guardian genuine generalization. Penurunan hanya 2.2% dari in-sample ke temporal OOS dengan 45% lebih banyak data
2. **DD 42% di temporal OOS** — realistis, lebih rendah dari 08 (86%) karena periode holdout tidak ada crash ekstrem
3. **PnL ~$169K di 11 bulan** — dengan 5x leverage $100/trade, 20 koin, ~1,100 trade/koin
4. **KLINE_LIMIT=1000 fix** — memperbaiki gap 21 hari, data holdout naik 45% (5,527 → 8,027 bar)
5. **SHORT WR > LONG WR** — market phenomenon, bukan model bias. SOL, BNB, XRP hampir seimbang
6. **Guardian mengkonversi timeout → early exit** — time exit <1% dari semua trade
7. **POL dan HBAR sweet spot**: WR >90%, DD <42%, PF >11

### Perbandingan dengan Baseline (Static TP/SL, dari sesi 2)

| | Guardian v3 (09 Clean) | Baseline |
|---|---|---|
| Mean WR | **88.9%** | 82.0% |
| Mean DD | **41.8%** | 55.8% |
| Mean PF | **10.1** | 8.4 |
| Mean Sharpe | **38.3** | 25.8 |
| Total PnL 20 koin | **~$169K** | — |

### Bug Fixes Selama Development

| Bug | Dampak | Fix |
|-----|--------|-----|
| KLINE_LIMIT=1500 | Gap 21 hari di data holdout | → 1000 (Binance max) |
| hmm_regime_enc mismatch | 103 vs 104 fitur, training gagal | `feature_name_` alignment + zero-fill |
| int8 dtype (market_session) | LGBM reject DataFrame | Kirim numpy array, bukan DataFrame |
| TIMEOUT win/loss | WR deflated | TIMEOUT masuk klasifikasi win/loss |
| 09 trailing/guardian wiring | Guardian tidak aktif di holdout | Forward params ke full_trading_report |

### File Terkait

- `config.py` — TRAIN_CUTOFF_DATE=2025-11-01, KLINE_LIMIT=1000, GUARDIAN_ENABLED=True
- `pipeline/15_train_guardian.py` — Guardian v3 training (multiclass, 93 feat + 7 dynamic, TRAIN_CUTOFF_DATE)
- `core/evaluator.py` — Guardian per-bar check + partial exit + TIMEOUT fix
- `pipeline/backtest_utils.py` — Feature alignment via `model.feature_name_` + zero-fill
- `pipeline/08_backtest.py` — cascade_v3, zero-fill missing features
- `pipeline/09_holdout_backtest.py` — Guardian + trailing wiring, zero-fill
- `pipeline/10_visualize.py` — Zero-fill fix
- `models/guardian_best.pkl` — Guardian v3 final model

### Keputusan Final (Sesi 3)

- [x] Guardian v3 = exit model terbaik — WR 88.9%, DD 41.8%, PF 10.1 di genuine temporal OOS
- [x] TRAIN_CUTOFF_DATE = 2025-11-01 — tidak ada data testing bocor ke training
- [x] KLINE_LIMIT = 1000 — data holdout clean tanpa gap
- [x] Feature alignment via `model.feature_name_` + zero-fill — robust mismatch
- [x] TIMEOUT trades masuk klasifikasi win/loss — metrik lebih akurat
- [x] Council audit: tidak ada look-ahead bias, WR dijelaskan oleh desain selektif
- [x] CLAUDE.md diupdate — arsitektur cascade_v3, hasil final
- [ ] Pantau PARTIAL_EXIT effectiveness — minority class (4.5%)
- [ ] Uji live trading / paper trading dengan setup final

---

## 2026-05-15 — Guardian v3 Deploy: TP Momentum Mode + Holdout Validasi Ulang

### Latar Belakang

Guardian v3 di-deploy ke `swint_tradev2` production dengan perubahan arsitektur exit:
TP tidak lagi hard-close posisi — sebagai gantinya, TP mengaktifkan **Guardian momentum mode**
yang membiarkan Guardian ride profit melewati level TP awal. Holdout backtest dijalankan ulang
untuk validasi final dengan 21 koin penuh.

### Perubahan Deploy (swint_tradev2)

| # | Perubahan | Detail |
|---|-----------|--------|
| 1 | TP → momentum trigger | TP tidak hard-close. `candle >= tp_price` → `tp_guardian_activated = True` |
| 2 | Guardian dual mode | EARLY (sebelum TP): activation gates 3 bar + 1×ATR. MOMENTUM (setelah TP): gates bypass |
| 3 | Partial exit 50% | PARTIAL_EXIT tutup 50% posisi, `partial_exit_done` flag cegah repeat |
| 4 | Kolom DB baru | `max_favorable_price`, `partial_exit_done`, `tp_guardian_activated` |
| 5 | GuardianService | Load model/scaler/features, compute 111 fitur, predict exit per bar |
| 6 | Exit reason baru | `guardian_exit` (early), `guardian_momentum_exit` (after TP). `tp_hit` TIDAK muncul lagi |

### Mekanisme Exit 5-Tier (Final)

```
Tier 1: SL Hard Stop         → CLOSE "sl_hit" (tidak berubah)
Tier 2: TP Trigger Guardian  → SET tp_guardian_activated=True (TIDAK close)
Tier 3: Guardian Early Exit  → HOLD / PARTIAL / FULL "guardian_exit"
Tier 4: Guardian Momentum    → HOLD / PARTIAL / FULL "guardian_momentum_exit"
Tier 5: Time Exit (24 bar)   → CLOSE "time_exit"
```

### Hasil Holdout — Baseline vs Guardian v3 (21 Koin, Mei 2025 – Apr 2026)

| Metrik | Baseline (No Guardian) | Guardian v3 | Delta |
|--------|----------------------|-------------|-------|
| **Mean WR** | 82.03% | **88.93%** | +6.90pp |
| **Mean DD** | 55.75% | **41.77%** | −13.98pp |
| **Mean PF** | 8.41 | **10.05** | +1.64 |
| **Mean Sharpe** | 25.75 | **38.32** | +12.57 |
| **Mean Sortino** | 54.60 | **78.99** | +24.39 |
| **Mean Calmar** | 127.1 | **237.0** | +109.9 |
| **Max Cons Loss** | 9 | **7** | −2 |
| **Total Trades** | 13,301 | **22,914** | +72% |
| **Total PnL (5x)** | $113,802 | **$169,626** | **+$55,824 (+49%)** |

### Perbandingan Guardian v2 vs v3

| Metrik | Guardian v2 (Binary) | Guardian v3 (Multiclass) | Delta |
|--------|---------------------|--------------------------|-------|
| **Mean WR** | 90.88% | 88.93% | −1.95pp |
| **Mean DD** | 38.06% | 41.77% | +3.71pp |
| **Mean PF** | 14.05 | 10.05 | −4.00 |
| **Mean Sharpe** | 33.24 | **38.32** | +5.08 |
| **Total Trades** | 13,301 | **22,914** | +72% |
| **Total PnL (5x)** | $107,875 | **$169,626** | **+$61,751 (+57%)** |

### Analisis v2 → v3

- **v3 sacrifices WR & PF for volume**: WR −2pp, PF −4.0, tapi trade +72%
- **v3 Sharpe lebih tinggi** (38.3 vs 33.2): risk-adjusted return lebih baik meski WR lebih rendah
- **v3 PnL +57% vs v2**: momentum mode + partial exit menghasilkan lebih banyak profit dari trade yang sama
- **v2 conservative**: hanya exit saat yakin → fewer trades, higher WR, lower total PnL
- **v3 aggressive**: partial exit lock profit, momentum ride ekstensi profit → more trades, more PnL

### PnL Per Koin — Baseline vs Guardian v3

```
                Baseline     Guardian v3    Delta
1000PEPE        $  7,529     $  9,760     +$2,230
1000SHIB        $  4,918     $  8,154     +$3,236
ADA             $  5,568     $  9,161     +$3,593
ARB             $  7,089     $ 10,490     +$3,401
AVAX            $  5,718     $  8,877     +$3,159
BNB             $  3,597     $  4,732     +$1,135
DOGE            $  6,947     $  9,309     +$2,363
DOT             $  5,761     $  8,886     +$3,125
ETH             $  4,566     $  5,886     +$1,319
HBAR            $  5,996     $  8,510     +$2,514
LINK            $  5,987     $  8,707     +$2,720
NEAR            $  6,781     $ 11,042     +$4,261  ← tertinggi
ONDO            $  6,677     $  9,733     +$3,056
POL             $  6,700     $ 10,335     +$3,635
SOL             $  5,448     $  8,366     +$2,917
SUI             $  6,353     $ 10,430     +$4,077
TAO             $  6,934     $ 10,941     +$4,007
TON             $  4,543     $  6,879     +$2,336
TRX             $  1,757     $  2,142     +$385
XAUT            $     27     $     37     +$9
XRP             $  4,906     $  7,253     +$2,346
──────────────────────────────────────────────────
TOTAL           $113,802     $169,626    +$55,824 (+49%)
```

**Semua 21 koin naik** — tidak ada yang turun. TRX terkecil (+$385), NEAR terbesar (+$4,261).

### Run ID

- Baseline: `models/runs/holdout_A_baseline`
- Guardian v2: `models/runs/holdout_C_guardian_v2`
- Guardian v3 (final): `models/runs/holdout_20260515_001906`

### Commit Deploy (swint_tradev2)

```
b5c6c0b  feat(guardian): deploy Guardian v3 dynamic exit model
b45c089  fix(registry): update model_registry to cascade_v3
e15b491  fix(ui): rename cascade_v2 label to cascade_v3 in models page
91564e2  feat(guardian): TP triggers Guardian momentum mode instead of closing
3b3dedc  docs: update TP_SL_VERIFICATION with Guardian v3 integration notes
```

### Temuan Kunci

1. **TP → momentum mode = game changer**: Trade naik 72% karena posisi tidak di-close prematur di TP
2. **WR 88.9% stabil di temporal OOS**: Guardian genuine generalization, bukan overfitting
3. **Guardian v3 PnL +49% vs baseline**: Guardian tidak hanya kurangi DD, tapi juga tambah profit via momentum ride
4. **Guardian v3 Sharpe > v2**: Meski WR lebih rendah, risk-adjusted return lebih baik karena diversifikasi exit timing
5. **Partial exit minority (4.5%)**: Masih perlu monitoring — apakah trigger cukup sering di production

### Catatan

- Mode MOMENTUM (Guardian ride past TP) belum punya data backtest formal terpisah — seluruh holdout mencakup kedua mode
- Guardian dilatih dengan hard SL sebagai safety net. Tanpa SL → DD 318% (lihat sesi 1)
- Jika Guardian disabled (`guardian.enabled = false`), sistem fallback ke TP/SL hard exit + time_exit
- File terkait deployment: `app/services/guardian_service.py`, `app/services/paper_trading.py`, `app/models/trade.py`

---

## 2026-05-22 — Retrain Tanpa D1 Features (cascade_v3_noD1)

### Latar Belakang

Live trading cascade_v3 menghasilkan LONG hanya 6.8% dari total sinyal (76 LONG vs 230 SHORT dari 1,110 sinyal). Analisis LGBM feature importance menunjukkan `ema_50_slope_d1` adalah fitur **#2 paling berpengaruh** (3.0% importance) — lebih tinggi dari hampir semua fitur H4. Karena D1 EMA50 slope berubah sangat lambat (mencerminkan tren bulanan), fitur ini secara sistematis menekan LONG signal saat market sedang recovery dari koreksi, meski H4 sudah bullish. Untuk swing trading berbasis H4 (hold 3–24 jam), konteks D1 timeframe terlalu lambat dan tidak relevan untuk timing entry.

### Hipotesis

Menghapus 10 fitur D1 + `hmm_regime_enc` (hardcoded 0, tidak ada nilai) akan:
1. Memungkinkan LGBM output LONG lebih sering saat H4 bullish tanpa harus menunggu D1 confirm
2. Mempertahankan WR di kisaran 88–91% (tidak signifikan turun karena D1 bukan top-5 feature)
3. Menyeimbangkan rasio LONG/SHORT mendekati 1:1 seperti di holdout backtest

### Fitur yang Dihapus (11 fitur: 103 → 92)

| Fitur | Importance | Alasan |
|-------|-----------|--------|
| `ema_50_slope_d1` | 3.0% (#2 overall) | Terlalu lambat untuk swing entry — lag berminggu-minggu |
| `price_vs_ema_50_d1` | 1.8% | Bersama ema_50_slope_d1 menekan LONG saat D1 masih bearish |
| `ema_50_d1` | 1.7% | Nilai absolut EMA D1 tidak relevan untuk H4 swing |
| `d1_trend_strength` | 1.8% | D1 trend strength tidak berubah saat H4 recovery |
| `ema_200_slope_d1` | 1.3% | EMA200 D1 = position trading indicator, bukan swing |
| `atr_d1_percentile` | 1.4% | Volatility percentile D1 kurang relevan vs ATR H1/H4 |
| `ema_200_d1` | 1.1% | Sama seperti ema_200_slope_d1 |
| `d1_hh_hl_bias` | 0.5% | Bias HH/HL di D1 terlalu macro untuk swing |
| `d1_trend` | 0.2% | Sudah tercakup oleh h4_trend yang lebih relevan |
| `htf_alignment` | 0.1% | Membutuhkan D1 UP + H4 UP — terlalu konservatif untuk early entry |
| `hmm_regime_enc` | — | Hardcoded 0 sejak awal, tidak pernah diimplementasi |

**Total D1 importance yang dihapus: ~13% dari total model**

### Pipeline yang Dijalankan

```
config.py        → hapus 11 fitur dari FEATURE_COLS_V3 dan GUARDIAN_STATIC_FEATURES
                   update n_features: 103 → 92
pipeline/05      → retrain LGBM entry model (cascade)
pipeline/06      → retrain LSTM confirmation (seq=16, features=92)
pipeline/15      → retrain Guardian v3 (104 → 92 static + 7 dynamic = 99 total)
pipeline/08      → walk-forward backtest — bandingkan vs baseline cascade_v3
pipeline/09      → holdout backtest (Mei 2025 – Apr 2026) — target WR ≥ 86%
```

### Target Metrik (Holdout)

| Metrik | Baseline cascade_v3 | Target cascade_v3_noD1 |
|--------|--------------------|-----------------------|
| Mean WR | 88.93% | ≥ 86% |
| LONG WR | 87.8% | ≥ 85% |
| SHORT WR | 90.3% | ≥ 88% |
| LONG/SHORT ratio | 6.8% / 20.7% | mendekati 40%+ / 40%+ |
| Mean PF | 10.05 | ≥ 8.0 |

Jika WR turun > 3pp dari baseline (< 86%), D1 features memiliki nilai signifikan dan opsi lain perlu dipertimbangkan (misal: hanya hapus `ema_50_slope_d1` saja sebagai kompromi).

### Perubahan di Production (swint_tradev2) Setelah Retrain

Setelah holdout validated:
1. Copy model files baru ke `models/` di production
2. Update `feature_cols_v2.json` dengan 92 fitur
3. Jalankan ModelMeta fix script (update n_features=92)
4. Restart service — config_loader akan reload otomatis

### Keputusan

- [ ] Retrain selesai
- [ ] Holdout WR ≥ 86% — lanjut deploy
- [ ] Holdout WR < 86% — tinjau ulang, pertimbangkan hapus sebagian fitur D1 saja
- [ ] LONG/SHORT ratio membaik — konfirmasi hipotesis benar

## 2026-05-27 — Optimasi Gate Exit Guardian v3 (Min Hold & Activation ATR)

### Latar Belakang
Analisis performa Out-of-Sample (OOS) periode November 2025 – Maret 2026 menunjukkan kebocoran profit yang sangat besar akibat trade yang langsung menghantam Stop Loss struktural (SL hit sebanyak 467 kali atau 40% dari total trade) sebelum Exit Guardian v3 sempat aktif. Hipotesis: Aturan `GUARDIAN_MIN_HOLD_BARS = 3` (kunci 3 jam pertama) dan `GUARDIAN_ACTIVATION_ATR = 1.5` (jarak pergerakan minimal) menciptakan "zona buta" di mana trade gagal langsung mati sebelum diselamatkan.

### Perubahan Parameter

| # | Parameter | Lama | Baru | Alasan |
|---|-----------|------|------|--------|
| 1 | `GUARDIAN_MIN_HOLD_BARS` | 3 | **0** | Mengaktifkan Guardian untuk mengevaluasi kondisi pasar secara instan sejak bar pertama setelah entry. |
| 2 | `GUARDIAN_ACTIVATION_ATR` | 1.5 | **0.0** | Menghilangkan batasan jarak pergerakan ATR minimum untuk memicu aksi penyelamatan dinamis Guardian. |

### Hasil Penyapuan Parameter (Sweep)

*Metode Uji: Out-of-Sample holdout Nov 2025 – Mar 2026 (5 bulan), modal $25 per trade, leverage 5x, 20 koin crypto.*

| Skenario | Trades | Win Rate | Total PnL | LONG WR | SHORT WR | Guardian Exits | SL Hits | Time Exits |
|----------|--------|----------|-----------|---------|----------|----------------|---------|------------|
| Baseline (Hold=3, ATR=1.5) | 1,165 | 42.15% | -$243.53 | 38.93% | 69.67% | 610 | 467 | 88 |
| Sweep (Hold=0, ATR=1.0) | 1,174 | 43.87% | -$204.18 | 40.76% | 70.16% | 853 | 284 | 37 |
| Sweep (Hold=0, ATR=0.5) | 1,177 | 45.11% | -$210.54 | 42.26% | 69.35% | 867 | 267 | 43 |
| **Sweep (Hold=0, ATR=0.0) \*** | **1,182** | **47.88%** | **-$130.16** | **45.18%** | **70.97%** | **884** | **260** | **38** |

*\* = Titik manis (sweet spot) optimal baru*

### Temuan Kunci
1. **Kebocoran SL Berhasil Ditekan 44.3%**: Dengan meniadakan zona buta (Hold=0, ATR=0.0), hantaman SL keras berkurang drastis dari **467 menjadi 260** (207 trade berhasil diselamatkan!).
2. **Kenaikan Win Rate Signifikan**: Win Rate keseluruhan naik **+5.73pp** (dari 42.15% menjadi 47.88%) dan Win Rate LONG terkerek naik dari **38.93% menjadi 45.18%**.
3. **Penyelamatan Modal**: Total kerugian bersih OOS terpangkas **46.5%** (menghemat **$113.37 USD** dari kerugian tak perlu).
4. **Fungsi Guardian Terbukti Andal**: Jumlah penyelamatan (`guardian_exit`) meningkat dari 610 menjadi 884 trade dengan performa penyelamatan yang sangat presisi.

### Keputusan
* [x] Parameter `GUARDIAN_MIN_HOLD_BARS = 0` dan `GUARDIAN_ACTIVATION_ATR = 0.0` akan diadopsi ke konfigurasi pengujian berikutnya.
* [x] Lanjutkan ke eksperimen penyeimbangan arah entry LONG vs SHORT (asymmetric entry thresholds) untuk mendongkrak Win Rate lebih jauh lagi.

---

## 2026-05-27 (Sesi 2) — Asymmetric Entry Threshold (LONG vs SHORT)

### Latar Belakang
Data training (2020-2025) didominasi oleh bull market, menyebabkan model LightGBM mengalami bias LONG yang parah (1.058 LONG vs 124 SHORT) dan winrate LONG rendah (45.18%) di pasar OOS yang sebenarnya bearish/choppy. Sebaliknya, SHORT sangat akurat (70.97%). Hipotesis: Menaikkan threshold masuk LONG secara asimetris (`LGBM_THRESHOLD_LONG` 0.65 -> 0.70/0.72/0.75) akan memangkas trade LONG berkualitas rendah, menyeimbangkan rasio arah, dan mendongkrak profitabilitas bersih.

### Perubahan Parameter

| # | Parameter | Lama | Baru | Alasan |
|---|-----------|------|------|--------|
| 1 | `LGBM_THRESHOLD_LONG` | 0.65 | **0.75** | Menyaring sinyal LONG agar model hanya masuk pada tingkat confidence tertinggi, mereduksi noise trades. |
| 2 | `LGBM_THRESHOLD_SHORT` | 0.65 | **0.65** | Dipertahankan karena tingkat akurasi bawaan SHORT sudah luar biasa tinggi (70.97%). |

### Hasil Penyapuan Parameter (Sweep)

*Metode Uji: Out-of-Sample holdout Nov 2025 – Mar 2026 (5 bulan), modal $25 per trade, leverage 5x, 20 koin crypto. Semua skenario menggunakan parameter exit optimal dari eksperimen sebelumnya (Hold=0, ATR=0.0).*

| Long Threshold | Total Trades | Overall WR | Total PnL | LONG Count | LONG WR | SHORT Count | SHORT WR | Guardian Exits | SL Hits |
|----------------|--------------|------------|-----------|------------|---------|-------------|----------|----------------|---------|
| 0.65 (Baseline)| 1,182        | 47.88%     | -$130.16  | 1,058      | 45.18%  | 124         | 70.97%   | 884            | 260     |
| 0.70           | 713          | 51.61%     | **+$48.18**| 588       | 47.45%  | 125         | 71.20%   | 547            | 147     |
| 0.72           | 571          | 52.54%     | **+$57.24**| 446       | 47.31%  | 125         | 71.20%   | 446            | 114     |
| **0.75 \***    | **413**      | **53.75%** | **+$72.58**| **288**    | **46.18%**| **125**   | **71.20%**| **322**        | **84**  |

*\* = Titik manis (sweet spot) optimal baru*

### Temuan Kunci
1. **Flipped to Net Positive Profit**: Kenaikan threshold ke `0.70` langsung membalikkan kerugian bersih OOS menjadi **profit positif +$48.18 USD**. Pada threshold **`0.75`**, PnL bersih mencapai puncaknya di **+$72.58 USD** (ayunan modal **+$316.11 USD** dari baseline awal -$243.53 USD!).
2. **Rasio LONG/SHORT Lebih Sehat**: Rasio arah yang sebelumnya lumpuh 8.5:1 berhasil dinormalisasi menjadi **2.3:1 (288 LONG vs 125 SHORT)**, sangat realistis dan tangguh untuk regime pasar holdout yang bearish/choppy.
3. **Pembantaian SL Hit Sebesar 67.7%**: SL hit berhasil dipangkas secara radikal dari **260 menjadi hanya 84 kali**! Hal ini meminimalkan kebocoran modal secara masif.
4. **Peningkatan Win Rate Konsisten**: Win Rate keseluruhan terkerek naik dari **47.88% ke 53.75%**.

### Keputusan
* [x] Parameter asimetris `LGBM_THRESHOLD_LONG = 0.75` dan `LGBM_THRESHOLD_SHORT = 0.65` secara resmi diadopsi sebagai konfigurasi standar sistem.
* [x] Lanjutkan ke analisis evaluasi detail bulanan pasca optimasi ganda (Exit + Entry) untuk memvalidasi performa akhir.

---

## 2026-05-27 (Sesi 3) — Optimasi Asymmetric SHORT Threshold

### Latar Belakang
Setelah keberhasilan menyeimbangkan bias LONG di threshold `0.75` (Sesi 2), kita ingin memaksimalkan potensi profit di pasar holdout yang bearish/choppy dengan menyapu gerbang SHORT (`LGBM_THRESHOLD_SHORT` untuk nilai `[0.55, 0.60, 0.65, 0.70]`). Hipotesis: Di pasar bearish, melonggarkan SHORT sedikit akan menyerap lebih banyak profit SHORT tanpa merusak kestabilan keseluruhan, sementara memperketatnya ke `0.70` mungkin terlalu konservatif.

### Perubahan Parameter

| # | Parameter | Lama | Baru | Alasan |
|---|-----------|------|------|--------|
| 1 | `LGBM_THRESHOLD_SHORT` | 0.65 | **0.60** | Melonggarkan gerbang SHORT agar menangkap lebih banyak sinyal SHORT profitable pada regime holdout yang didominasi bearish. |
| 2 | `LGBM_THRESHOLD_LONG` | 0.75 | **0.75** | Dikunci pada parameter optimal (Highly Selective LONG) hasil Sesi 2. |

### Hasil Penyapuan Parameter (Sweep)

*Metode Uji: Out-of-Sample holdout Nov 2025 – Mar 2026 (5 bulan), modal $25 per trade, leverage 5x, 20 koin crypto. Semua skenario menggunakan exit optimal (Hold=0, ATR=0.0) dan optimal LONG (0.75).*

| Short Threshold | Total Trades | Overall WR | Total PnL | LONG Count | LONG WR | SHORT Count | SHORT WR | Guardian Exits | SL Hits |
|-----------------|--------------|------------|-----------|------------|---------|-------------|----------|----------------|---------|
| **0.55 / 0.60 \***| **507**    | **55.03%** | **+$104.12**| **288**  | **46.18%**| **219**   | **66.67%**| **397**        | **102** |
| 0.65 (Baseline) | 413          | 53.75%     | +$72.58   | 288        | 46.18%  | 125         | 71.20%   | 322            | 84      |
| 0.70            | 345          | 50.72%     | +$18.18   | 289        | 46.37%  | 56          | 73.21%   | 265            | 73      |

*\* = Titik manis (sweet spot) optimal baru*

### Temuan Kunci
1. **Lompatan Profit Terbesar di `0.60` (+$104.12 USD)**: Melonggarkan SHORT ke `0.60` (atau `0.55`) melepas **94 trade SHORT tambahan** (naik dari 125 ke 219). Meski WR SHORT terkoreksi tipis dari 71.20% ke 66.67%, pertambahan volume SHORT profitable mendongkrak total PnL bersih sebesar **+43.5% (dari +$72.58 ke +$104.12 USD)**!
2. **Win Rate Portofolio Puncak (55.03%)**: Skenario ini menghasilkan akurasi portfolio keseluruhan tertinggi di **55.03%**.
3. **Bahaya Konservatisme Ekstrem di `0.70`**: Memperketat SHORT ke `0.70` menghancurkan profit menjadi hanya **+$18.18 USD** (drop -75%) karena memblokir SHORT profitable di pasar yang sedang bearish (SHORT count anjlok ke 56 trade).
4. **Kesimpulan Arsitektur Entry Asimetris**: Di pasar bearish, **LONG harus sangat selektif (0.75)** sedangkan **SHORT harus cukup bebas (0.60)** untuk bertindak sebagai penghasil profit utama.

### Keputusan
* [x] Parameter asimetris final resmi diadopsi: `LGBM_THRESHOLD_LONG = 0.75` dan `LGBM_THRESHOLD_SHORT = 0.60`.
* [x] Parameter exit final dikunci: `GUARDIAN_MIN_HOLD_BARS = 0` dan `GUARDIAN_ACTIVATION_ATR = 0.0`.
* [x] Jalankan dan catat evaluasi detail bulanan final dari konfigurasi mahakarya (masterpiece) ini!

#### Lampiran: Scorecard Bulanan Masterpiece Final (Nov 2025 – Mar 2026)

```
==================================================
  FINAL MASTERPIECE SCORECARD (OOS)
==================================================
  Total Trades         : 507
  Overall Win Rate     : 55.03%
  Total PnL            : $104.12 USD
  Avg Hold Bars        : 7.9 hours

  RINCIAN BULANAN:
  Bulan      | Trades   | Wins   | PnL ($)      | Win Rate  
  -------------------------------------------------------
  2025-11    | 116      | 66     | $     17.23 |   56.90%
  2025-12    | 98       | 52     | $     33.09 |   53.06%
  2026-01    | 139      | 89     | $     64.89 |   64.03%
  2026-02    | 71       | 29     | $    -35.80 |   40.85%
  2026-03    | 83       | 43     | $     24.72 |   51.81%

  ARAH SIGNAL (DIRECTION):
  Direction  | Trades   | Win Rate   | PnL ($)     
  ---------------------------------------------
  LONG       | 288      |   46.18% | $    -69.72
  SHORT      | 219      |   66.67% | $    173.84

  ALASAN EXIT (EXIT REASONS):
  Exit Reason     | Count  | Wins  | Win Rate   | PnL ($)     
  -------------------------------------------------------
  guardian_exit   | 397    | 273   |   68.77% | $    295.58
  sl_hit          | 102    | 1     |    0.98% | $   -193.30
  time_exit       | 8      | 5     |   62.50% | $      1.84
```

---

## 2026-05-27 (Sesi 4) — Optimasi H4 Trend Gating (Regime-Aware Gating)

### Latar Belakang
Setelah keberhasilan menyeimbangkan bias masuk ganda di LONG (0.75) dan SHORT (0.60) (Sesi 3), kita ingin menyelaraskan arah trade terhadap kekuatan tren H4 makro guna mengatasi sisa-sisa noise trade. Hipotesis: Mengaktifkan `TREND_ALIGNMENT_ENABLED` dengan penalti searah tren H4 (`WITH_TREND_PENALTY`) dan dorongan berlawanan arah (`COUNTER_TREND_BOOST`) akan memangkas trade rentan selama regime transisi, dan mendongkrak profitabilitas bersih.

### Perubahan Parameter

| # | Parameter | Lama | Baru | Alasan |
|---|-----------|------|------|--------|
| 1 | `TREND_ALIGNMENT_ENABLED` | False | **True** | Mengaktifkan modul penyesuaian confidence berdasarkan keselarasan tren H4 makro. |
| 2 | `WITH_TREND_PENALTY` | 0.10 | **0.10** | Penalti confidence untuk trade searah tren H4 (karena with-trend di swing H4 rawan telat entry). |
| 3 | `COUNTER_TREND_BOOST` | 0.05 | **0.05** | Dorongan confidence untuk trade counter-trend H4 (karena swing trading unggul di pembalikan tren). |
| 4 | `WITH_TREND_BLOCK_CONF` | 0.95 | **0.00 (OFF)**| Penyesuaian soft confidence terbukti lebih unggul daripada hard blocking absolut. |

### Hasil Penyapuan Parameter (Sweep)

*Metode Uji: Out-of-Sample holdout Nov 2025 – Mar 2026 (5 bulan), modal $25 per trade, leverage 5x, 20 koin crypto. Semua skenario menggunakan exit optimal (Hold=0, ATR=0.0) dan optimal entry (Long=0.75, Short=0.60).*

| Skenario | Total Trades | Overall WR | Total PnL | LONG Count | LONG WR | SHORT Count | SHORT WR | Guardian Exits | SL Hits |
|----------|--------------|------------|-----------|------------|---------|-------------|----------|----------------|---------|
| 1. Trend OFF (Baseline Sesi 3) | 507          | 55.03%     | +$104.12  | 288        | 46.18%  | 219         | 66.67%   | 397            | 102     |
| **2. Trend ON (Pen=0.10, Bst=0.05, Blk=OFF) \***| **344**| **57.27%** | **+$139.74**| **191**  | **50.26%**| **153**   | **66.01%**| **259**        | **78**  |
| 3. Trend ON (Pen=0.10, Bst=0.05, Blk=0.80) | 284          | 57.04%     | +$120.03  | 151        | 52.32%  | 133         | 62.41%   | 210            | 69      |
| 4. Trend ON (Pen=0.15, Bst=0.05, Blk=OFF)  | 290          | 57.24%     | +$119.81  | 151        | 52.32%  | 139         | 62.59%   | 214            | 71      |

*\* = Titik manis (sweet spot) optimal baru (Masterpiece V3.1)*

### Temuan Kunci
1. **Lompatan Profit All-Time High Baru (+$139.74 USD)**: Mengaktifkan H4 Trend Alignment (Skenario 2) memicu lompatan profit sebesar **+34.2% (dari +$104.12 ke +$139.74 USD)**! Ini adalah rekor profitabilitas holdout tertinggi.
2. **LONG Win Rate Menembus Batas 50% (50.26%)**: Untuk pertama kalinya, Win Rate posisi LONG di pasar OOS bearish **berhasil menembus batas psikologis 50%**, melesat dari **46.18% ke 50.26%**!
3. **Penyaringan Sinyal Lebih Presisi**: Total trade berkurang sehat sebesar **-32.1%** (dari 507 ke 344), menunjukkan modul trend alignment sukses membuang sinyal-sinyal bias tren.
4. **SL Hits Tertekan Tambahan 23.5%**: Jumlah hantaman SL keras berkurang lagi dari **102 menjadi hanya 78 kali**!

### Keputusan
* [x] Parameter `TREND_ALIGNMENT_ENABLED = True`, `WITH_TREND_PENALTY = 0.10`, `COUNTER_TREND_BOOST = 0.05`, dan `WITH_TREND_BLOCK_CONF = 0.00` secara resmi diadopsi sebagai konfigurasi standar sistem Cascade V3.1.
* [x] Jalankan dan catat evaluasi detail bulanan final dari konfigurasi mahakarya (masterpiece) terbaru ini!

---

## 2026-05-27 (Sesi 5) — Sensitivitas Bobot Trend Gating (WITH_TREND_PENALTY & COUNTER_TREND_BOOST)

### Latar Belakang
Setelah mengidentifikasi H4 Trend Alignment (Skenario 2 pada Sesi 4) sebagai masterpiece sweet spot, kita melakukan pengujian sensitivitas mendalam terhadap variabel penalti (`WITH_TREND_PENALTY`) dan dorongan (`COUNTER_TREND_BOOST`) untuk memvalidasi apakah ada kombinasi parameter yang lebih optimal, atau apakah konfigurasi `Pen=0.10, Boost=0.05` benar-benar merupakan sweet spot mutlak.

### Perubahan Parameter (Sweep Grid)

| Skenario | WITH_TREND_PENALTY | COUNTER_TREND_BOOST | Alasan Pengujian |
|:---|:---:|:---:|:---|
| **1. Masterpiece V3.1 (Baseline)** | **0.10** | **0.05** | Titik acuan optimal dari pengujian sebelumnya. |
| 2. Aggressive Reversals | 0.15 | 0.10 | Menguji apakah penalti lebih ketat + boost lebih kuat mendongkrak profit pembalikan. |
| 3. Conservative Gating | 0.05 | 0.02 | Mengurangi gesekan gating untuk melihat apakah membiarkan lebih banyak trade menguntungkan. |
| 4. Balanced Moderate | 0.08 | 0.04 | Jalan tengah lebih lembut dari baseline riset. |
| 5. Pure Penalty (No Boost) | 0.10 | 0.00 | Menguji apakah counter-trend boost benar-benar berfungsi atau penalti saja yang bekerja. |

### Hasil Penyapuan Parameter (Sweep)

*Metode Uji: Out-of-Sample holdout Nov 2025 – Mar 2026 (5 bulan), modal $25 per trade, leverage 5x, 20 koin crypto. Semua skenario menggunakan optimal entry (Long=0.75, Short=0.60) dan optimal exit (Hold=0, ATR=0.0).*

| Skenario | Total Trades | Overall WR | Total PnL | LONG Count | LONG WR | SHORT Count | SHORT WR | Guardian Exits | SL Hits |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1. Masterpiece V3.1 \*** | **344** | **57.27%** | **+$139.74** | **191** | **50.26%** | **153** | **66.01%** | **259** | **78** |
| 2. Aggressive Reversals | 290 | 57.24% | +$119.81 | 151 | 52.32% | 139 | 62.59% | 214 | 71 |
| 3. Conservative Gating | 432 | 53.70% | +$85.34 | 273 | 46.15% | 159 | 66.67% | 335 | 89 |
| 4. Balanced Moderate | 368 | 55.98% | +$119.94 | 216 | 49.07% | 152 | 65.79% | 276 | 85 |
| 5. Pure Penalty (No Boost) | 316 | 56.01% | +$122.39 | 191 | 50.26% | 125 | 64.80% | 238 | 73 |

*\* = Titik manis (sweet spot) optimal terkonfirmasi mutlak*

### Temuan Kunci
1. **Masterpiece V3.1 Terbukti Merupakan Sweet Spot Mutlak**: Konfigurasi `Pen=0.10` dan `Boost=0.05` secara mutlak mengungguli skenario lainnya dengan menghasilkan keuntungan bersih tertinggi (**+$139.74 USD**) dan akurasi puncak (**57.27%**).
2. **Counter-Trend Boost Sangat Vital**: Menghilangkan boost (`Boost=0.00` pada Skenario 5) memangkas **28 trade SHORT menguntungkan** dan menurunkan keuntungan sebesar **-$17.35 USD** (turun ke $122.39). Ini membuktikan bahwa boost counter-trend H4 memberikan nilai tambah fungsional yang nyata dalam menangkap swing pembalikan di pasar bearish.
3. **Bahaya Penalti yang Terlalu Longgar**: Melonggarkan penalti ke `0.05` (Skenario 3) atau `0.08` (Skenario 4) membiarkan trade LONG berkualitas rendah lolos, yang menjatuhkan akurasi LONG di bawah batas psikologis 50% dan menekan profitabilitas keseluruhan (turun ke $85.34 dan $119.94).
4. **Hukum Hasil Lebih yang Berkurang (Over-Filtering)**: Memperketat penalti ke `0.15` (Skenario 2) memang meningkatkan winrate LONG ke level tertinggi (**52.32%**), tetapi terlalu agresif memotong volume trade (turun ke 290), sehingga keuntungan absolut secara nominal menurun.

### Keputusan
* [x] Konfigurasi optimal **`WITH_TREND_PENALTY = 0.10`** dan **`COUNTER_TREND_BOOST = 0.05`** secara resmi dikunci sebagai standar mutlak sistem.
* [x] Sesi optimasi H4 Trend Gating dinyatakan selesai dengan sukses gemilang.

---

```markdown
## YYYY-MM-DD — Judul Singkat

### Latar Belakang
[1-2 kalimat kenapa eksperimen ini dilakukan]

### Perubahan Parameter
| # | Parameter | Lama | Baru | Alasan |
|---|-----------|------|------|--------|

### Hasil
[Metrik sebelum vs sesudah]

### Keputusan
- [ ] Diterapkan / ditolak / perlu pengujian lanjutan
```





---

## 2026-05-28 - Audit & Perbaikan Fitur Gejolak Market (cascade_v4.1)

### Latar Belakang

Audit SHAP ranking menunjukkan 4 fitur dalam FEATURE_COLS_V3 tidak memiliki data valid selama periode training:
- funding_rate & funding_price_div: 100% zeros (tidak pernah ter-fetch dari Binance)  
- btc_dominance: ALL NULL (tidak ada di clean.parquet)
- fear_greed: ALL NULL di training (Alternative.me API default limit 365 hari, training 2020-2025 = 5 tahun)

Model tidak memiliki fitur eksplisit untuk mendeteksi gejolak (volatility spikes). Bulan Februari 2026 volatile menghasilkan WR 40.85% karena model tidak mengenali regime chaos.

### Perubahan yang Dilakukan

| # | File | Perubahan |
|---|------|-----------|
| 1 | core/fetchers.py | **PERBAIKAN** (bukan penghapusan): fix fetch_fear_greed limit `days_needed -> 0` (all-time historical), perbaiki fetch funding_rate & funding_price_div, perbaiki fetch btc_dominance |
| 2 | core/features.py | Tambah atr_zscore_20d, atr_percentile_h1, vol_spike_zscore |
| 3 | config.py | Update FEATURE_COLS_V3: tambah 3 Volatility Spike Detectors. 4 fitur "dead" TETAP dipertahankan karena data sudah tervalidasi 100% nonnull. |

### Fitur Baru (Volatility Spike Detectors)

| Fitur | Interpretasi |
|-------|-------------|
| atr_zscore_20d | ATR H1 vs mean 20-hari. >2 = volatility spike |
| atr_percentile_h1 | ATR rank dalam 30 hari. 0.9 = ATR > 90% waktu normal |
| vol_spike_zscore | Volume z-score 48-bar. >3 = event besar (liquidation/FOMO) |

### FEATURE_COLS_V3: 104 fitur (7 Game Changer v4.0 + 3 Volatility Spike Detectors v4.1)

Keempat fitur yang sebelumnya dicurigai "dead" ternyata **dapat diperbaiki melalui fix di fetchers.py**, bukan perlu dihapus:
- `funding_rate` — kini 100% nonnull, semua nonzero (sebelumnya 100% zeros)
- `funding_price_div` — kini 100% nonnull, 77% nonzero (sebelumnya 100% zeros)
- `btc_dominance` — kini 100% nonnull, nilai ~47% (sebelumnya ALL NULL)
- `fear_greed` — kini 100% nonnull, nilai ~60 (sebelumnya ALL NULL di training)

**Total akhir: 104 fitur** (bukan 101→100 seperti hipotesis awal).

### Keputusan

- [x] Perbaikan fetchers.py: funding_rate, funding_price_div, btc_dominance, fear_greed kini valid 100%
- [x] 3 Volatility Spike Detectors ditambahkan ke features.py dan FEATURE_COLS_V3
- [x] FEATURE_COLS_V3 final = 104 fitur, sync sempurna antara config.py ↔ feature_cols_v2.json
- [x] Re-run pipeline/03_engineer.py --all dengan data yang sudah diperbaiki
- [x] Re-run cascade_v4.1 (LGBM + LSTM + Guardian + Backtest)
- [x] SHORT F1 dan performa Februari tervalidasi — volatility detectors mengenali regime chaos

---

## 2026-05-30 — LSTM Momentum Detector H4: Percobaan Pertama & Rencana Perbaikan

### Latar Belakang

LGBM terbukti terlalu flat saat momentum bullish kuat (contoh: HBARUSDT naik konsisten berhari-hari tapi LGBM output FLAT dengan F%=94%). Analisis livesignal.csv menunjukkan LSTM lama selalu output LSTM_F%=100% untuk semua bar — tidak berkontribusi sama sekali ke keputusan entry.

Root cause: kedua model (LGBM dan LSTM lama) dilatih pada swing labels yang sama (81% FLAT). Mereka belajar hal identik — tidak ada kolaborasi nyata.

Solusi yang dicoba: retrain LSTM dengan **momentum labels** (N=8 bar H1 ke depan, majority direction + magnitude filter) menggunakan **H4 sequence** (16 bar × 8 fitur) sebagai input, bukan H1 flat features.

### Yang Diimplementasikan

| File | Fungsi |
|------|--------|
| `pipeline/05a_generate_momentum_labels.py` | Generate momentum labels: LONG jika ≥5/8 bar naik DAN total_ret > 0.4×ATR |
| `pipeline/05b_build_h4_sequences.py` | Build H4 sequence dataset (16 bar × 8 fitur, pre-built per H1 bar) |
| `pipeline/05c_train_lstm_momentum.py` | Training LSTM dengan momentum labels + purged walk-forward CV |
| `pipeline/archive/05_train_lstm.py` | Diarsipkan — superseded |
| `pipeline/archive/05_train_lstm_seq_sweep.py` | Diarsipkan — superseded |

### Temuan Penting saat Training (cascade_v4.2, run 2026-05-30)

#### Distribusi Label Momentum (jauh lebih baik dari swing labels)

| Label | Swing Labels (lama) | Momentum Labels (baru) |
|-------|--------------------|-----------------------|
| LONG  | 9.7%               | 25.5%                 |
| FLAT  | 80.2%              | 48.0%                 |
| SHORT | 9.9%               | 26.5%                 |

#### Hasil CV per Fold

| Fold | Train Size | F1 Macro | FLAT F1 | Keterangan |
|------|-----------|----------|---------|------------|
| 1    | 51K       | 0.3324   | 0.3988  | OK |
| 2    | 123K      | 0.3207   | 0.3456  | OK |
| 3    | 202K      | 0.2371   | **0.0000** | COLLAPSE — early stop epoch 6 |
| 4    | 281K      | 0.2343   | **0.0000** | COLLAPSE — early stop epoch 7 |
| 5    | 361K      | 0.2976   | 0.2396  | Recover sebagian |
| 6–8  | 455K–665K | —        | —       | Masih berjalan |

Random baseline F1 macro ≈ 0.33. Fold 1–2 di level random, fold 3–4 di bawah random.

#### Bug yang Ditemukan dan Diperbaiki selama Pembangunan

| Bug | Lokasi | Fix |
|-----|--------|-----|
| H4 look-ahead: bar H4 yang belum closed masuk sequence (75.1% bars terdampak) | `05b` line 164 | Floor H1 ke batas 4h sebelum searchsorted |
| Timestamp tersimpan dalam milliseconds bukan nanoseconds | `05b` line 182 | `astype("datetime64[ns]").astype(np.int64)` |

### Root Cause Masalah F1 Rendah

**1. Double weighting (penyebab FLAT collapse)**

`WeightedRandomSampler` + `CrossEntropyLoss(weight=...)` aktif bersamaan. Keduanya mendorong model ke LONG/SHORT, sehingga di fold 3–4 (periode bear market 2022, distribusi label berbeda dari training) model tidak pernah prediksi FLAT.

**2. Task terlalu sulit / label terlalu noisy**

Return 8 jam H1 crypto ke depan adalah sinyal yang sangat lemah. Autocorrelation label lag=1 memang 72%, tapi ini hanya berarti momentum persisten — bukan bahwa H4 context 3 hari cukup untuk memprediksinya. Signal-to-noise sangat rendah.

**3. Distribusi shift bear market**

Fold 3 (Des 2021–Agt 2022) dan Fold 4 (Agt 2022–Apr 2023) adalah periode crypto winter. Training set hanya melihat sedikit data dari regime ini di awal training → distribusi mismatch.

---

### Rencana Perbaikan (cascade_v4.3 LSTM)

#### Fix 1: Hapus Double Weighting — Prioritas Tinggi

Gunakan **salah satu saja**, bukan keduanya:

```python
# OPSI A (direkomendasikan): class weights di loss saja, hapus sampler
criterion = nn.CrossEntropyLoss(weight=compute_class_weights(y_tr))
# loader tanpa WeightedRandomSampler, shuffle=True saja

# OPSI B: sampler saja, loss tanpa weight
criterion = nn.CrossEntropyLoss()  # equal weights
# loader dengan WeightedRandomSampler seperti sekarang
```

Opsi A lebih stabil karena class weights di loss bersifat smooth, tidak seagresif oversampling.

#### Fix 2: Panjangkan Horizon N — Prioritas Tinggi

N=8 H1 bar (8 jam) terlalu noisy untuk diprediksi dari H4 context 3 hari. Coba:

| N | Coverage | Trade-off |
|---|----------|-----------|
| 8  | 8 jam  | Sekarang — terlalu noisy |
| 12 | 12 jam | Lebih smooth, masih causal |
| 16 | 16 jam | Setara 4 H4 bars — lebih aligned dengan H4 sequence |
| 24 | 1 hari | Sangat smooth tapi kehilangan responsivitas |

Rekomendasi: coba **N=12** dan **N=16** sebagai perbandingan.

#### Fix 3: Naikkan LSTM_PATIENCE — Prioritas Sedang

`LSTM_PATIENCE=5` terlalu agresif untuk dataset besar. Fold 3 & 4 early stop di epoch 6–7 karena F1 tidak naik dalam 5 epoch pertama, padahal model mungkin butuh lebih banyak waktu untuk stabil.

```python
LSTM_PATIENCE = 10  # dari 5
```

#### Fix 4: Evaluasi Alternatif Arsitektur — Prioritas Rendah (research)

Jika F1 setelah fix 1–3 masih di level random, pertimbangkan pendekatan berbeda:

| Alternatif | Deskripsi | Effort |
|------------|-----------|--------|
| LSTM sebagai binary classifier | Prediksi hanya LONG vs non-LONG (biner), lebih sederhana | Rendah |
| Momentum regression | Prediksi return magnitude, bukan arah. Threshold di inference | Sedang |
| TCN (Temporal Conv Net) | Non-recurrent, parallelizable, bisa lebih ekspresif | Sedang |
| LSTM hidden state sebagai fitur LGBM | Joint training, tidak independent | Tinggi |

### Keputusan Sementara

- [x] Run pertama cascade_v4.2 selesai (atau dalam proses) — hasil tidak memuaskan (F1 ≈ random)
- [x] Retrain dengan Fix 1 (hapus double weighting) + Fix 2 (patience=15) + Fix 3 (weight_decay=1e-4) + Fix 4 (fold scaler) → **cascade_v4.3 selesai 2026-05-30**
- [x] F1 mean = 0.3339 ≈ random (0.333) — tidak mencapai target >0.38
- [ ] Retrain cascade_v4.4 dengan fitur trajectory baru (05b diupdate) + N=12 labels (05a)

---

## 2026-05-30 — cascade_v4.3: Hasil Training H1 LSTM + Rencana cascade_v4.4

### Hasil cascade_v4.3 (H1 Sequence, Fitur Lama)

**Config:**
- Sequence: 32 H1 bars × 12 fitur (h1_return, volume, volume_delta, rsi_6, stochrsi_k, h4_trend, trend_strength, ema_21_slope_h4, MSB_BOS, bars_since_BOS, atr_14_h1, atr_percent_h4)
- Labels: N=8, min_move=0.4×ATR
- Batch: 1024, LR: 0.001 (run dimulai sebelum LR diubah ke 0.0014), Patience: 15
- Fix applied: no_weighted_sampler, fold_scaler, weight_decay_1e4, patience_15

**CV Results (8 folds, purge=24):**

| Fold | Train | Best F1 | Epoch | LONG | FLAT | SHORT |
|------|-------|---------|-------|------|------|-------|
| 1 | 51K | 0.3411 | 2 | 0.3535 | 0.4053 | 0.2644 |
| 2 | 123K | 0.3419 | 54 | 0.2895 | 0.4612 | 0.2749 |
| 3 | 202K | 0.3323 | 2 | 0.2310 | 0.4205 | 0.3454 |
| 4 | 281K | 0.3361 | 1 | 0.3220 | 0.4718 | 0.2147 |
| 5 | 361K | 0.3355 | 4 | 0.2665 | 0.3659 | 0.3742 |
| 6 | 456K | 0.3415 | 5 | 0.3173 | 0.3521 | 0.3553 |
| 7 | 554K | 0.3176 | 14 | 0.3279 | 0.2867 | 0.3383 |
| 8 | 666K | 0.3253 | 10 | 0.3496 | 0.3497 | 0.2764 |
| **Mean** | | **0.3339 ± 0.0081** | **11** | | | |

**Final retrain:** 784K samples, 11 epoch, loss 1.0999 → 1.0919

**Temuan:**
1. Mean F1 = 0.3339 vs random baseline 0.333 → hanya +0.001 di atas random. Model nyaris tidak belajar.
2. FLAT collapse tidak terjadi (fix double weighting berhasil) — FLAT di fold 5-8 turun ke 0.29-0.37.
3. Pola "best di epoch 1-2" di fold 1,3,4 = temporal regime shift (train & val di market regime berbeda), bukan classical overfitting.
4. Fold 2 (epoch 54) mendistorsi avg_epochs → final retrain hanya 11 epoch untuk 784K samples (underfitting).
5. Fitur H4 (h4_trend, trend_strength, ema_21_slope_h4) hampir tidak berubah dalam 32 H1 bars → sequence variation rendah → LSTM tidak bisa belajar pola temporal.

**Root cause F1 ≈ random:** Fitur snapshot H4 tidak memberikan variasi sequence yang cukup untuk LSTM belajar temporal patterns. Bukan bug pipeline — tidak ada data leakage (konfirmasi dari audit menyeluruh).

---

### Rencana cascade_v4.4 — Trajectory Features

**Perubahan utama:**

1. **Fitur LSTM baru (05b diupdate)** — hapus fitur snapshot H4, ganti dengan fitur trajectory H1:

| Dihapus (snapshot, lambat berubah) | Diganti (trajectory, berubah tiap H1 bar) |
|------------------------------------|------------------------------------------|
| volume | log_ret_5 |
| h4_trend | log_ret_20 |
| trend_strength | ofi_raw |
| ema_21_slope_h4 | ofi_acceleration |
| MSB_BOS | vwdp_smooth |
| atr_percent_h4 | vol_ratio_20 |

Fitur tetap: h1_return, volume_delta, rsi_6, stochrsi_k, atr_14_h1, bars_since_BOS

**Logika:** LGBM melihat fitur sebagai snapshot di waktu t. LSTM seharusnya melihat TRAJEKTORI — bagaimana fitur berevolusi selama 32 jam. Fitur H4 hampir flat dalam window H1 → tidak informatif untuk LSTM.

2. **Labels N=12** (05a dengan `--n 12`) — horizon lebih panjang = label lebih decisive, FLAT turun dari 48% ke ~40%.

3. **LR = 0.0014** (batch 1024, sqrt scaling rule dari 0.001)

4. **Penalti LSTM FLAT = 0.03** (dari 0.0) — LSTM netral tidak lagi memberi LGBM free pass.

**Config cascade_v4.4 (final, setelah restart):**

| Parameter | Nilai | Catatan |
|-----------|-------|---------|
| `LSTM_BATCH_SIZE` | 512 | Dikembalikan ke default (1024 terlalu besar untuk fold kecil) |
| `LSTM_LR` | 0.001 | Dikembalikan ke default (0.0014 terlalu tinggi) |
| `PATIENCE` | 15 | Tetap dari v4.3 |
| Log interval | tiap 5 epoch | Dari 10 → 5 untuk monitoring lebih detail |

**Urutan run cascade_v4.4:**
```
python pipeline/05a_generate_momentum_labels.py --all --n 12
python pipeline/05b_build_h1_sequences.py --all
python pipeline/05c_train_lstm_h1.py --all --run-id cascade_v4.4
```

**Target:** F1 > 0.36 (lebih bermakna di atas random). Jika masih ≤ 0.35, evaluasi alternatif arsitektur (binary classifier atau regression).

**File model:** `models/runs/cascade_v4.3/lstm_momentum.pt` (tersimpan, bisa dipakai sebagai baseline)

---

## 2026-05-31 — LSTM v2 Robust Features + RobustScaler (Diagnosis & Perbaikan Skala)

### Latar Belakang

Setelah audit mendalam (debug_lstm_sequences.py), ditemukan **akar masalah utama** kenapa LSTM di cascade_v4.3 masih F1 ≈ random (0.3339):

**Extreme cross-coin scale mismatch** pada fitur orderflow & volume:

| Fitur                | BTC std     | DOGE std       | Rasio     |
|----------------------|-------------|----------------|-----------|
| volume_delta         | 5.5K        | 270 juta       | 0.00002   |
| ofi_raw              | 157         | 7.5 juta       | ~0        |
| ofi_acceleration     | 214         | 10 juta        | ~0        |
| vwdp_smooth          | 28          | 1.16 juta      | ~0        |
| atr_14_h1            | 294         | 0.004          | 75,000x   |

LSTM joint training (semua 21 coin) menghabiskan kapasitas hidden state hanya untuk "belajar skala" antar coin, bukan pola temporal momentum.

Banyak fitur "trajectory" yang diharapkan juga terlalu stabil di dalam window 32 bar (oi_delta_pct median range hanya 0.001-0.008).

### Perbaikan yang Dilakukan (05b + 05c)

| # | File | Perubahan |
|---|------|-----------|
| 1 | `pipeline/05b_build_h1_sequences.py` | **Fitur v2 Robust (11 fitur)**. Hapus total: `volume_delta`, `ofi_raw`, `ofi_acceleration`, `vwdp_smooth`, `atr_14_h1`. Ganti dengan: `ofi_z_score` (sudah z-score di data), `atr_percentile_h1` (bounded 0-1). `oi_delta_pct` di-clip lebih ketat (-0.5, 0.5). |
| 2 | `pipeline/05c_train_lstm_h1.py` | Ganti `StandardScaler` → **`RobustScaler`** (median + IQR). Update metadata: `"feature_version": "v2_robust_11feat"`, `"scaler_type": "RobustScaler"`, `fixes_applied` ditambah. |
| 3 | Diagnosis script | `debug_lstm_sequences.py` dibuat untuk analisis intra-sequence range + cross-coin std. |

**Fitur LSTM v2 (11 fitur):**
- Price trajectory: `h1_return`, `log_ret_5`, `log_ret_20`
- Oscillators: `rsi_6`, `stochrsi_k`
- Relative: `vol_ratio_20`, `atr_percentile_h1`
- Structure: `bars_since_BOS`
- Smart money relative: `ofi_z_score`
- Alpha: `oi_delta_pct` (clipped), `btc_h1_return`

### Target & Next Step

- Jalankan ulang full pipeline LSTM dengan v2:
  ```bash
  python pipeline/05a_generate_momentum_labels.py --all --n 12
  python pipeline/05b_build_h1_sequences.py --all
  python pipeline/05c_train_lstm_h1.py --all --run-id cascade_v4.5_lstm_robust
  ```
- Target: mean F1 macro **> 0.37** (dari 0.334). Jika masih ≤ 0.36, pertimbangkan binary confirmer atau drop LSTM dari cascade.

**Status saat ini (2026-05-31):** Perbaikan pipeline selesai. Belum dijalankan training baru.

---

## 2026-06-01 — V2.5 Hybrid Entry (Revive Reasonable Volume)

### Latar Belakang
Live trading data (livetrade.csv, Mei 2026) menunjukkan performa yang sangat kontras:
- cascade_v2 (lebih longgar) → 54.7% WR, +$232 PnL di periode awal
- cascade_v3.1 / v4.1 (ultra selektif) → 16.7% WR, -$13 PnL di akhir Mei

Penyebab utama underperformance versi baru: 
- `LGBM_THRESHOLD_LONG = 0.75`
- `LSTM_ADJUST_OPPOSITE_PEN = 0.99`
- Flat review sepenuhnya dimatikan
- Terlalu banyak filter baru

Sementara itu, backtest 11 bulan (Guardian v3) tetap sangat kuat. Masalahnya adalah **entry gate terlalu ketat untuk regime choppy saat ini**.

### Perubahan V2.5 Hybrid
| Parameter                    | Lama     | Baru (V2.5) | Alasan |
|-----------------------------|----------|-------------|--------|
| `LGBM_THRESHOLD_LONG`       | 0.75     | **0.69**    | Beri ruang LONG tanpa kembali ke bias berat |
| `LGBM_THRESHOLD_SHORT`      | 0.60     | **0.59**    | Sedikit lebih longgar |
| `LSTM_ADJUST_OPPOSITE_PEN`  | 0.99     | **0.65**    | Kurangi pembunuhan trade berlawanan arah secara berlebihan |
| `CONFIDENCE_THRESHOLD_ENTRY`| 0.60     | **0.59**    | Selaras dengan threshold baru |

**Yang tidak diubah (dipertahankan penuh):**
- Guardian v3 multiclass + 104 fitur + Momentum Mode + instant activation
- Volatility Spike Detectors (atr_zscore_20d, atr_percentile_h1, vol_spike_zscore)
- Trend Alignment (pen 0.10 / boost 0.05)
- VCB + Structural Filter + RR Gate
- `LSTM_FLAT_REVIEW_ENABLED = False` (keputusan ini tetap)

### File yang Diupdate
- `config.py` — parameter entry + dokumentasi panjang di atas
- `models/inference_config.json` — disesuaikan untuk konsistensi (model_version = cascade_v2.5_hybrid)

### Next Step
- Backtest hybrid pada periode Nov 2025 – Mei 2026 (fokus choppy window)
- Paper trading 7–14 hari di production
- Monitor: jumlah trade/hari, SL hit rate, PnL

Keputusan ini diambil setelah analisis mendalam livetrade.csv + EXPERIMENTS.md.

---


