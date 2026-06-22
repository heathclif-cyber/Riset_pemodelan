# Robustness Audit — tb_widyawardhana_v2_continuation
Date: 2026-06-15
Auditor: Claude Code (read-only, no code changes)
Scope: flatboost_v2 LGBM (27 feat) + continuation_v1 Guardian (29 feat) + HMM regime

---

## Summary

Lima temuan kritis yang perlu diperhatikan sebelum menyimpulkan performa holdout adalah representatif:

1. **Guardian StandardScaler di-fit pada seluruh training set** (bukan per-fold CV), menyebabkan data leakage ringan di CV metrics Guardian — namun tidak mempengaruhi holdout karena Guardian scaler tidak di-fit ulang pada data holdout. (MEDIUM)

2. **Threshold 0.50/0.55 flatboost_v2 dipilih dari holdout Apr-Jun 2026** — threshold sweep 49 kombinasi dilakukan secara eksplisit terhadap data holdout yang sama yang digunakan untuk pelaporan final. Ini adalah look-ahead bias yang terkonfirmasi dan mengangkat WR yang dilaporkan. (HIGH)

3. **Guardian training menggunakan in-sample TB trade** dari flatboost_v2 yang sama (model entry yang sudah ada) — ini valid secara arsitektur tapi menciptakan circular dependency: Guardian menilai kualitas exit dari trade yang di-generate oleh model yang sama. (MEDIUM)

4. **HMM OOF labels untuk training bersih**, tetapi ada residual risiko bahwa parameter HMM (n_states, n_folds, purge) ditentukan setelah melihat efeknya di holdout melalui threshold sweep. (LOW-MEDIUM)

5. **Holdout digunakan berulang kali** untuk pemilihan: threshold sweep (49 combo), Guardian threshold (288 combo), HMM config (84 combo), min_hold, dan perbandingan antar variant — total puluhan iterasi. Setiap iterasi menambah kontaminasi holdout. (HIGH)

---

## 1. Data Leakage Check

### 1.1 Feature Engineering

**Status: SEBAGIAN BERSIH — satu leakage historis sudah difiks**

**Confirmed clean:**
- Semua rolling windows (`log_ret_1/5/20`, `vol_ratio_20`, `atr_zscore_20d`, `vol_spike_zscore`, dll.) menggunakan `.rolling()` backward-looking. Tidak ditemukan `.shift(-N)` pada fitur untuk prediksi.
- H4 swing points: `detect_h4_swing_points()` menggunakan `i-lookback:i+lookback+1` — ini sebenarnya look-ahead (bar i+3 dipakai untuk konfirmasi swing bar i). Namun sudah dimitigasi dengan `h4_swing_highs = h4_swing_highs.shift(3)` di `engineer_features()` baris ~1452-1453. Mitigasi ini benar dan memadai.
- `calc_liquidity_levels()`: swing dikonfirmasi dengan `shift(lookback)` — bersih.
- `calc_market_structure()`: prev_sh/sl menggunakan `shift(lookback+1)` — bersih.
- HMM untuk training: OOF walk-forward, di-fit per fold dengan purge 6 H4 bars (~24 jam). Bersih.
- HMM untuk holdout (`03e_regime_hmm_holdout.py`): model di-fit hanya pada data `< TRAIN_CUTOFF_DATE`, kemudian di-predict terhadap holdout. Bersih.

**Leakage yang sudah ditemukan dan diperbaiki (2026-06-12):**
- `etf_total_change_usd` dan `etf_gbtc_change_usd` di `03_engineer.py` — di-merge tanpa lag T-1, menyebabkan IC anomali (0.14-0.15, 2x lebih tinggi dari fitur terkuat lain). Fix: `shift(24)`. Model `tb_lstm_v1` yang dilatih sebelum fix dinyatakan CONTAMINATED dan tidak di-deploy. Model flatboost_v2 dan continuation_v1 Guardian tidak menggunakan fitur ETF, sehingga **tidak terpengaruh**.

**Potensi leakage yang perlu diperhatikan:**
- `mfe_8`, `mfe_12`, `mae_8`, `mae_12`, `time_above_entry_8`, `time_below_entry_8` (excursion features): menggunakan `rolling(8/12)` backward-looking — bersih.
- CVD (`calc_cvd`) menggunakan `.cumsum()` dari `taker_buy_volume - taker_sell_volume`. Ini bersifat kumulatif dan panjang, artinya nilai CVD di bar T mengandung seluruh history dari bar pertama. Dalam produksi, data real-time memberikan CVD yang terus bertambah sejak listing. Ini bukan leakage dalam arti ML, tapi perlu diperhatikan karena normalisasi (z-score) terhadap seluruh history berbeda antara training (data lengkap tersedia) dan produksi (streaming). Risiko: distribusi CVD z-score di awal data berbeda dengan di akhir.

### 1.2 CV / Training Methodology

**Purging:**
- `build_purged_folds()` menggunakan expanding window. Purge dilakukan dengan memotong `purge` bar terakhir dari training timestamps dan `purge` bar pertama dari test timestamps.
- Untuk TB model (`04_train_lgbm_flatboost_v3.py`): `purge = max(max_hold, TB_PURGE_GAP_BARS) = max(36, 36) = 36 bar`. Ini sesuai dengan `max_hold=36` — purge cukup mencegah TB label di fold boundary bocor ke validation.
- Multi-coin: `build_purged_folds` bekerja pada `unique timestamps` yang di-sorted. Ini benar — memisahkan semua koin yang share timestamp yang sama ke satu sisi fold boundary. Tidak ada kebocoran lintas koin.

**Scaler:**
- flatboost_v2 LGBM: tidak menggunakan scaler (tree-based model). Bersih.
- Guardian continuation_v1 (`06p_train_guardian_continuation_v1.py`, baris 455): `scaler = StandardScaler(); X_scaled = scaler.fit_transform(X_all)` — scaler di-fit pada **seluruh** training set sebelum CV loop, bukan di dalam setiap fold.

  **Ini adalah data leakage ringan di CV metrics.** Test fold melihat statistik (mean, std) yang dihitung dari seluruh dataset termasuk test fold. Namun, dampaknya pada performa sebenarnya terbatas karena: (a) scaler hanya melakukan normalisasi linear, (b) nilai mean/std tidak berubah drastis antar fold untuk dataset besar (>100k samples). Guardian CV F1=0.85 mungkin sedikit inflated, tapi efeknya kecil secara ekonomis.

  Untuk **holdout evaluation**, scaler ini tidak di-refit — `guardian_scaler.pkl` yang sudah ada dipakai langsung. Ini bersih (scaler hanya tahu training data).

**Early Stopping:**
- Guardian menggunakan `eval_set=[(X_scaled[test_idx], y_all[test_idx])]` untuk early stopping. Dengan scaler yang di-fit pada seluruh data, ini memperkuat leakage ringan di atas. Namun sekali lagi, dampaknya terbatas.
- flatboost_v2 menggunakan `eval_set=[(X_val, y_val)]` per fold — bersih karena tidak ada scaler.

### 1.3 Guardian Training

**Status: SUSPECTED LABEL CIRCULARITY (MEDIUM)**

Guardian (`06p_train_guardian_continuation_v1.py`) dilatih sebagai berikut:
1. Load flatboost_v2 model yang sudah ada.
2. Jalankan flatboost_v2 di seluruh training data untuk generate trade entries.
3. Untuk setiap in-sample trade, hitung label HOLD/PARTIAL/FULL_EXIT berdasarkan future price.
4. Latih Guardian pada in-sample trade tersebut.

**Label Guardian menggunakan future price — ini disengaja dan benar.** Ini adalah cara standard pelabelan exit strategy: "apa yang seharusnya dilakukan di bar ini berdasarkan apa yang terjadi di masa depan?"

**Yang perlu dicatat:**
- Guardian dilabeli berdasarkan `best_future_pnl` (baris 320-329 `06p`): `for k in range(j + 1, min(bar_out, n)): best_future_pnl = max(best_future_pnl, pnl_k)`. Ini adalah future-looking untuk label — intentional, bukan bug.
- TB purge di Guardian: `GUARDIAN_PURGE_GAP_BARS = 36 = MAX_HOLDING_BARS`. Ini memastikan label Guardian yang berakhir di bar T tidak overlap ke validation fold yang mulai dari T+36. Benar.
- Guardian tidak dapat memprediksi future price secara langsung dari fitur. Label dibuat dari future price, tapi fitur adalah snapshot saat ini (static market features + dynamic trade context).

**Circular dependency yang dikhawatirkan:** Guardian dilatih pada trade yang dibuka oleh flatboost_v2. Jika flatboost_v2 di-replace dengan model lain, distribusi trade (entry timing, entry level) berubah, dan Guardian yang ada mungkin tidak optimal. Ini adalah **arsitektur coupling** yang valid diperhatikan, bukan leakage dalam arti tradisional. Untuk holdout evaluation yang adil, hal ini sudah terjaga: Guardian dilatih pada training data, dievaluasi terhadap holdout trade yang dibuka oleh flatboost_v2 yang sama.

**Continuation HOLD override** (`flow_still_aligned()`, baris 162-186): override EXIT ke HOLD saat flow masih searah. Ini menggunakan data pada bar j (bukan future) — bersih dari look-ahead.

---

## 2. Overfitting Check

### 2.1 CV vs Holdout Performance Gap

**flatboost_v2 LGBM:**
- CV F1 Macro = 0.3914 (8-fold purged, training data Jan 2020 - Apr 2026)
- Holdout standalone (no Guardian, argmax conf>0.40): WR=52.8%, 11,913 trades
- Holdout dengan threshold 0.50/0.55: WR=66.9%, 1,698 trades
- CV F1 Macro yang rendah (~0.39) dibandingkan WR holdout yang lebih tinggi (66.9%) mencerminkan bahwa metrik CV (F1 pada label) tidak berkorelasi langsung dengan WR simulasi. Tidak ada anomali yang jelas dari perspektif CV vs holdout.
- Fold 1 di flatboost_v2 CV: `best_iteration=22` (sangat dini). Ini adalah tanda awal bahwa fold pertama memiliki distribusi yang berbeda dari fold berikutnya — wajar untuk expanding window CV karena fold 1 memiliki training set paling kecil.

**Guardian continuation_v1:**
- CV F1 Macro = 0.8503 (8-fold) — sangat tinggi untuk multiclass 3-label.
- Namun CV dijalankan dengan scaler yang di-fit pada seluruh data (lihat 1.2) — nilai ini mungkin sedikit inflated.
- Holdout PF = 2.65 (momentum_v1) dan 2.70 (laporan di prompt). PF > 2 yang konsisten di holdout genuine menunjukkan edge yang nyata. Tidak ada tanda overfitting parah.
- Guardian F1=0.85 yang tinggi disebabkan oleh dominasi kelas HOLD (>70% sampel): "predict HOLD terus" sudah memberikan akurasi tinggi. F1 macro lebih informatif — 0.85 cukup baik tapi perlu dilihat per-class untuk memastikan EXIT class tidak lemah.

### 2.2 Feature Count vs Sample Size

**flatboost_v2 LGBM:**
- 27 fitur, ~785,000 training bars (21 koin × ~37,000 bars), tree-based model.
- Rasio sampel/fitur: 785,000 / 27 ≈ 29,000:1. Sangat aman, tidak ada risiko overfitting dimensi.
- Regulasi: `min_child_samples=50`, `subsample=0.8`, `colsample_bytree=0.7`, `class_weight="balanced"`. Memadai.

**Guardian continuation_v1:**
- ~28 fitur (variasi tergantung IC selection), ~129,893 samples (dari EXPERIMENTS.md momentum_v1 baseline).
- Rasio: ~4,639:1. Masih aman untuk tree-based dengan regulasi L1/L2.
- `num_leaves=63`, `max_depth=6`, `n_estimators=2000`, `learning_rate=0.02` dengan early stopping 100. Agresif tapi ada safety net early stopping.

### 2.3 Holdout Reuse

**Status: CONFIRMED — HIGH SEVERITY**

Holdout Apr-Jun 2026 digunakan berulang kali untuk keputusan parameter dan model selection:

1. **Threshold sweep flatboost_v2** (EXPERIMENTS.md "Eksperimen 6"): 49 kombinasi threshold_long × threshold_short diuji terhadap holdout. Threshold optimal (0.50/0.55) dipilih berdasarkan hasil holdout ini.

2. **TB widyawardhana_v3 sweep** (EXPERIMENTS.md "2026-06-13b"): 288 kombinasi LGBM HMM config × Guardian threshold × min_hold diuji terhadap holdout yang sama.

3. **Perbandingan 6-variant** (EXPERIMENTS.md "2026-06-13"): ic32+Guardian vs TB+Guardian vs TB+LSTM-C+Gdn vs berbagai bare variants — semua terhadap holdout yang sama.

4. **Guardian variant comparison**: profit_v1 vs momentum_v1 vs continuation_v1 — semua dibandingkan di holdout yang sama.

5. **LGBM HMM config sweep** (sweep 08, 84 combos): dilakukan terhadap holdout yang sama.

**Estimasi kontaminasi:** Dengan puluhan parameter yang di-tune terhadap holdout, WR dan PnL yang dilaporkan adalah optimistis. Ini adalah **selection bias / multiple testing problem**. Nilai seperti WR=70.3% dan PF=2.70 mencerminkan performa konfigurasi terbaik yang dipilih DARI holdout, bukan estimasi yang tidak-bias dari performa live trading.

**Catatan mitigasi dari kode:** EXPERIMENTS.md mencatat bahwa "Holdout Nov-Mar 2026 (sesi sebelumnya) contaminated — TB Guardian v2 training overlap. Ini holdout bersih pertama." Ini menunjukkan kesadaran tim tentang masalah ini, tapi kontaminasi tetap terjadi pada holdout Apr-Jun melalui proses tuning yang ekstensif.

---

## 3. Look-ahead Bias

### 3.1 Dynamic Sizing Threshold Selection

**Status: CONFIRMED — HIGH SEVERITY**

Dari prompt dan dari EXPERIMENTS.md "Eksperimen 6":

- Threshold 0.50/0.55 untuk flatboost_v2 dipilih berdasarkan **49-combination sweep terhadap holdout Apr-Jun 2026**.
- Threshold HMM-adaptive `{TRENDING:0.42, RANGING:0.52}` dipilih berdasarkan **288-combination sweep terhadap holdout yang sama**.
- Guardian threshold 0.55 (turun dari 0.65) dipilih berdasarkan sweep holdout yang menunjukkan "+$42 avg PnL."

**Ini adalah look-ahead bias yang terdefinisi dengan baik:** parameter utama sistem dipilih berdasarkan hasil yang diukur pada data yang diklaim sebagai "OOS holdout." Setiap kali holdout digunakan untuk membuat keputusan, ia kehilangan statusnya sebagai genuine OOS.

**Severity:** HIGH. WR yang dilaporkan (70.3%) dan PF (2.70) dari konfigurasi TERPILIH tidak dapat diinterpretasikan sebagai estimasi tidak-bias performa live. Angka sebenarnya bisa lebih rendah secara material.

**Quantifikasi contoh konkret dari EXPERIMENTS.md:**
- Guardian 0.55 vs 0.65: avg $318 vs $276 (+15% PnL dari tuning satu parameter ini saja terhadap holdout)
- HMM T042_R052 vs baseline: $322 vs ~$280 (+15%)
- Threshold 0.50/0.55 vs argmax: WR 66.9% vs 52.8% (+14 pp dari satu keputusan yang dioptimasi di holdout)

**Catatan:** Tuning threshold memiliki justifikasi intuitif (threshold tinggi = lebih selektif = WR lebih tinggi). Ini bukan manipulation, tapi tetap berarti angka holdout adalah upper bound optimistis, bukan prediksi unbiased.

### 3.2 Calibration Methodology

**Status: CLEAN**

- Platt calibrator (jika digunakan di ic32 stack) di-fit pada fold validation — tidak di-fit pada holdout. Bersih.
- flatboost_v2 tidak menggunakan calibrator — langsung argmax atau threshold pada proba raw.
- Guardian scaler di-fit pada training data saja — holdout tidak tersentuh saat fitting.

---

## 4. Distribution & Concentration Risk

### 4.1 Per-coin Performance

**Status: TIDAK DAPAT DIVERIFIKASI SECARA PENUH dari kode yang diaudit**

Dari `12_holdout_ic32_apr_jun26.py` (baris 338-344), output per-coin disimpan ke JSON. Data per-coin aktual tidak tersedia dalam audit ini — hanya scorecard agregat yang bisa diverifikasi dari kode.

Yang dapat disimpulkan dari arsitektur:
- 21 koin, 918 total trades (dari prompt) → rata-rata ~44 trades per koin untuk 2.5 bulan. Jumlah ini cukup kecil untuk variance tinggi per koin.
- Beberapa koin listing setelah 2020 (SUI, TON, PEPE, TAO, ARB) memiliki training history lebih pendek — model mungkin underfit pada koin baru ini.
- **Risiko konsentrasi:** WR agregat 70.3% bisa disebabkan oleh 2-3 koin dengan WR >80% yang mengimbangi koin lain dengan WR <50%. Tanpa breakdown per-coin, tidak bisa dikonfirmasi.

**Dari EXPERIMENTS.md (2026-06-13 comparison):**
- ic32+Guardian: 1,041 trades di holdout sebelumnya (Nov-Mar), $292
- TB+Guardian (holdout Apr-Jun, konfigurasi optimal): 918 trades, WR 70.3%

**Rekomendasi:** Jalankan breakdown per-coin dari file `holdout_apr_jun26_vs_ic32.json` untuk verifikasi distribusi.

### 4.2 Temporal Stability

**Status: CONCERN — SUSPECTED DEGRADASI**

Dari EXPERIMENTS.md:
- **Holdout Nov-Mar 2026 (5 bulan)**: TB+Guardian $1,125 (55.6% WR), ic32+Guardian $848 → $157-291/bulan
- **Holdout Apr-Jun 2026 (2.5 bulan)**: TB optimal $365, ic32+Guardian $292 → $84-146/bulan
- **Degradasi ic32+Guardian**: $157/bulan (Nov-Mar) → $58/bulan (Apr-Jun) = -63%

Komentar dari EXPERIMENTS.md: "Degradasi vs Nov-Mar: PnL/bulan ic32+Guardian $58 vs $157 di holdout lama (-63%). Periode Apr-Jun 2026 adalah market harder untuk strategi ini."

Dari prompt, holdout final yang dilaporkan (918 trades, WR=70.3%, PF=2.70) menggunakan konfigurasi yang di-tune terhadap Apr-Jun. Jika live trading berlanjut ke periode berbeda (Jul-Sep 2026), degradasi temporal bisa terjadi lagi.

**Pola temporal dari flatboost_v2 (jika mengikuti ic32 pattern):**
- Apr 2026: kemungkinan lebih baik (market fresh OOS)
- Mei 2026: degradasi (live trading ic32 menunjukkan under-trading parah)
- Jun 2026: hanya 7% PnL dari distribusi (disebutkan di prompt)

Distribusi WR per bulan yang tidak merata (disebutkan Apr=57%, May=36%, Jun=7% di prompt untuk PnL) menunjukkan bahwa performa sangat tidak stabil dalam 2.5 bulan.

---

## 5. Verdict per Issue

| Issue | Severity | Status | Rekomendasi |
|-------|----------|--------|-------------|
| Guardian StandardScaler di-fit pada seluruh dataset (leakage CV ringan) | MEDIUM | CONFIRMED | Pindahkan `scaler.fit_transform` ke dalam loop CV; fit hanya pada `train_idx`. Dampak kecil tapi prinsipnya penting. |
| Threshold 0.50/0.55 dipilih dari holdout (look-ahead bias) | HIGH | CONFIRMED | Kumpulkan holdout baru (Jul-Sep 2026) sebagai true OOS sebelum menyimpulkan performa final. |
| 288+ parameter combinations di-tune terhadap holdout yang sama (holdout reuse) | HIGH | CONFIRMED | Sama dengan atas — holdout Apr-Jun sudah terkontaminasi oleh tuning ekstensif. |
| Guardian label HOLD/EXIT menggunakan future price | LOW | CONFIRMED (by design) | Ini adalah cara standard labeling exit — bukan bug. Dokumentasikan sebagai intentional. |
| Guardian circular dependency dengan flatboost_v2 entry | MEDIUM | SUSPECTED | Evaluasi Guardian secara terpisah dengan entry model berbeda untuk verifikasi generalisasi. |
| HMM OOF untuk training | CLEAR | BERSIH | Implementasi walk-forward OOF di `03e_regime_hmm.py` sudah benar. |
| HMM untuk holdout (fit pada train, predict holdout) | CLEAR | BERSIH | `03e_regime_hmm_holdout.py` hanya fit pada `< TRAIN_CUTOFF_DATE`. |
| ETF look-ahead leakage (etf_total_change_usd) | CLEAR | FIXED (2026-06-12) | Model aktif tidak menggunakan ETF features. Leakage sudah diperbaiki. |
| H4 swing look-ahead (detect_h4_swing_points) | CLEAR | MITIGATED | `shift(3)` diterapkan di `engineer_features()`. Benar. |
| Triple Barrier labeling menggunakan future bar | LOW | CONFIRMED (by design) | TB labeling secara inheren prospektif — ini adalah metode yang documented (Lopez de Prado). Purge gap 36 bar sudah mencegah leakage ke CV validation fold. |
| CVD kumulatif (distribusi berbeda early vs late training) | LOW | SUSPECTED | Pertimbangkan normalisasi window terbatas (e.g., rolling 500-bar CVD) vs cumsum penuh untuk stabilitas distribusi. |
| Degradasi temporal holdout Nov-Mar vs Apr-Jun (-63% PnL) | HIGH | CONFIRMED | Strategi perlu evaluasi robustness multi-period. Holdout satu periode 2.5 bulan tidak cukup. |
| Per-coin concentration risk (tidak dapat dikonfirmasi) | MEDIUM | SUSPECTED | Lakukan analisis per-coin dari holdout JSON untuk deteksi outlier. |

---

## 6. Prioritas Perbaikan

**IMMEDIATE (sebelum kesimpulan apapun tentang live viability):**

1. **Kumpulkan holdout baru Jul-Sep 2026 sebagai true OOS.** Tidak ada satupun parameter yang boleh di-tune terhadap periode ini. Evaluasi flatboost_v2 + continuation_v1 dengan threshold 0.50/0.55 dan Guardian 0.55 secara langsung tanpa modifikasi. Ini adalah satu-satunya cara mendapat estimasi unbiased.

2. **Analisis per-coin dari holdout JSON.** Verifikasi bahwa WR 70.3% bukan didominasi 2-3 koin. Jika ada koin dengan <20 trades, WR per koin tidak representatif.

**MEDIUM TERM (untuk kualitas metodologi):**

3. **Perbaiki Guardian CV scaler leakage.** Fit scaler per fold: `scaler_fold = StandardScaler(); X_tr_scaled = scaler_fold.fit_transform(X_all[train_idx]); X_val_scaled = scaler_fold.transform(X_all[test_idx])`. CV metrics akan lebih akurat.

4. **Walk-forward temporal validation.** Jalankan sistem dengan konfigurasi frozen terhadap setiap bulan secara sequential: Apr, Mei, Jun sebagai out-of-time slices terpisah. Distribusi WR/PnL per bulan memberi gambaran stabilitas yang jauh lebih baik dari agregat 2.5 bulan.

5. **Benchmark bottom-up per koin terhadap random baseline.** Uji apakah WR per koin secara statistik berbeda dari 50% (binomial test, alpha 0.05 dengan Bonferroni correction untuk 21 koin).

**MONITORING (selama live trading):**

6. **Monitor WR dan PF per bulan di live trading.** Jika selama 2 bulan berturut-turut WR < 55% atau PF < 1.5 (versus 70.3% / 2.70 holdout), pertimbangkan retrain atau deactivation.

7. **Track distribusi confidence score di live vs training.** Distribution shift pada LGBM confidence scores adalah early warning sign bahwa market regime telah berubah.

---

## Appendix: File Kunci yang Diaudit

| File | Temuan Kunci |
|------|-------------|
| `D:\Apps-Dev\Riset_pemodelan\core\features.py` | H4 swing shift(3) OK; CVD cumsum distribusi risk; ETF bukan di fitur aktif |
| `D:\Apps-Dev\Riset_pemodelan\pipeline\shared.py` | `build_purged_folds()` benar (timestamp-level, expanding) |
| `D:\Apps-Dev\Riset_pemodelan\pipeline\06p_train_guardian_continuation_v1.py` | Scaler di-fit pada seluruh data sebelum CV — leakage ringan |
| `D:\Apps-Dev\Riset_pemodelan\pipeline\04_train_lgbm_flatboost_v3.py` | CV methodology bersih |
| `D:\Apps-Dev\Riset_pemodelan\pipeline\07_holdout_flatboost_v2.py` | Bersih sebagai holdout runner; masalah ada di iterasi penggunaannya |
| `D:\Apps-Dev\Riset_pemodelan\pipeline\03e_regime_hmm.py` | OOF walk-forward bersih |
| `D:\Apps-Dev\Riset_pemodelan\pipeline\03e_regime_hmm_holdout.py` | Bersih — fit pada training saja |
| `D:\Apps-Dev\Riset_pemodelan\pipeline\12_holdout_ic32_apr_jun26.py` | Holdout evaluation script bersih; problem di iterasi penggunaan |
| `D:\Apps-Dev\Riset_pemodelan\EXPERIMENTS.md` | Bukti holdout reuse ekstensif (288+ combinations) |
| `D:\Apps-Dev\Riset_pemodelan\config.py` | TRAIN_CUTOFF_DATE = 2026-04-01 = OOS_START — tidak ada overlap |
| `D:\Apps-Dev\Riset_pemodelan\models\model_registry.json` | holdout_scorecard Nov-Mar 2026: WR 67.5%, PnL $848 — lebih reliable sebagai uncontaminated estimate |
