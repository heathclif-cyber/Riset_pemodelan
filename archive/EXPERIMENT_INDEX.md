# ARCHIVE INDEX — Daftar Eksperimen Diarsipkan

Format: `YYYY-MM-DD-N_category_object` — diurutkan kronologis.

---

## 2026-05-09

### 2026-05-09-1_extended_backtest_scorecard
- **File**: `scratch/extended_2026_scorecard.py`, `scratch/extended_scorecard_v2.py`
- **Deskripsi**: Extended backtest scorecard generation untuk berbagai konfigurasi cascade
- **Hasil**: ❌ Metrik lama — dijalankan sebelum leak fix 2026-06-04. Semua metrik INVALID (in-sample leakage)

---

## 2026-05-14

### 2026-05-14-1_guardian_v2_binary
- **File**: `pipeline/06_train_guardian.py`
- **Deskripsi**: Guardian exit model pertama — binary HOLD/EXIT, 32 static + 7 dynamic features
- **Hasil**: ❌ WR 93-94% tapi PnL -13% vs baseline. Exit prematur + model "buta" market context

### 2026-05-14-2_guardian_v3_multiclass
- **File**: (lihat `pipeline/06b_train_guardian_clean_v2.py` yang aktif, `models/runs/ic32_guardian_clean_v2/`)
- **Deskripsi**: Guardian multiclass 3-class (HOLD/PARTIAL_EXIT/FULL_EXIT), 103 static features
- **Hasil**: ✅ Terbaik — WR 88.9%, DD 41.8%, PF 10.1 di temporal OOS. Ini adalah Guardian yang AKTIF sekarang (clean_v2 variant)

### 2026-05-14-3_trailing_stop_atr
- **File**: (parameter di `config.py` — TRAILING_STOP_ENABLED, TRAILING_STOP_ATR)
- **Deskripsi**: Trailing stop non-ML: 1x ATR, 2x ATR
- **Hasil**: ❌ PnL 91% dari baseline, DD DOGE -42%. Trailing 2x ATR terbaik dari non-ML tapi Guardian v3 superior

---

## 2026-05-22

### 2026-05-22-1_cascade_v3_noD1_retrain
- **File**: (perubahan di `config.py` — FEATURE_COLS_V3 dari 103 → 92)
- **Deskripsi**: Retrain cascade_v3 tanpa 10 fitur D1. Hipotesis: D1 features terlalu lambat untuk H4 swing
- **Hasil**: ⚠️ Holdout WR target ≥86%. LONG/SHORT ratio belum tervalidasi di production

---

## 2026-05-27

### 2026-05-27-1_guardian_minhold_sweep
- **File**: (parameter sweep di `config.py` — GUARDIAN_MIN_HOLD_BARS, GUARDIAN_ACTIVATION_ATR)
- **Deskripsi**: Optimasi Guardian gate: min_hold 0/2/6/8 + activation_atr 0.0/0.5/1.0/1.5
- **Hasil**: ⚠️ **DATA LEAKAGE** — parameter dipilih dari sweep pada data holdout yang sama. min_hold=0, activation_atr=0.0 adalah sweet spot tapi TIDAK valid sebagai OOS estimate

### 2026-05-27-2_lgbm_asymmetric_entry
- **File**: `scratch/test_custom_config.py`
- **Deskripsi**: Asymmetric entry threshold: LONG=0.75, SHORT=0.60. Menyeimbangkan bias LONG di bear market
- **Hasil**: ⚠️ **DATA LEAKAGE** — dipilih dari holdout yang sama. Terbukti gagal di live: WR 16.7% Jun 2026

### 2026-05-27-3_h4_trend_alignment_gating
- **File**: `scratch/test_flip_trend.py`, `scratch/test_flip_scorecard.py`
- **Deskripsi**: TREND_ALIGNMENT_ENABLED — penalti with-trend, boost counter-trend
- **Hasil**: ⚠️ **DATA LEAKAGE** — "Masterpiece V3.1 sweet spot mutlak" adalah circular validation. Parameter sesungguhnya: FLIP alignment (REGIME_AWARE_ALIGNMENT) — RANGING=counter-trend, TRENDING=with-trend

### 2026-05-27-4_short_threshold_sweep
- **File**: (embedded di `scratch/test_custom_config.py`)
- **Deskripsi**: Sweep LGBM_THRESHOLD_SHORT [0.55, 0.60, 0.65, 0.70]
- **Hasil**: ⚠️ **DATA LEAKAGE**. SHORT=0.60 dipilih — overtrade di trending market

### 2026-05-27-5_trend_gating_sensitivity
- **File**: `scratch/test_flip_scorecard.py`
- **Deskripsi**: Sensitivitas WITH_TREND_PENALTY & COUNTER_TREND_BOOST
- **Hasil**: ⚠️ **DATA LEAKAGE**. Pen=0.10, Boost=0.05 "terkonfirmasi" via circular validation

---

## 2026-05-28

### 2026-05-28-1_cascade_v4.1_volatility_detectors
- **File**: `core/features.py` (3 Volatility Spike Detectors), `models/runs/cascade_v4.1/`
- **Deskripsi**: 3 fitur baru: atr_zscore_20d, atr_percentile_h1, vol_spike_zscore. Fix dead features (funding_rate, btc_dominance, fear_greed). Total: 104 fitur
- **Hasil**: ✅ Fitur volatility valid. Tapi cascade_v4.1 ultra-selektif → WR 16.7% di live Jun 2026

---

## 2026-05-30

### 2026-05-30-1_lstm_momentum_h4_v4.2
- **File**: `pipeline/05a_momentum_labels_v2.py`, `pipeline/05b_train_lstm_momentum_v2.py`
- **Deskripsi**: LSTM momentum dengan H4 sequence (16 bar × 8 fitur), N=8 labels. Fix: H4 look-ahead bug
- **Hasil**: ❌ F1 0.33 ≈ random. Double weighting + noisy labels. COLLAPSE di fold 3-4

### 2026-05-30-2_lstm_momentum_h1_v4.3
- **File**: `pipeline/05c_train_lstm_momentum_v3.py`, `models/runs/cascade_v4.3/`
- **Deskripsi**: LSTM H1 sequence (32 bar × 12 fitur), fix double weighting + fold scaler + patience=15
- **Hasil**: ❌ Mean F1 0.3339 ≈ random (0.333). Fitur H4 snapshot flat dalam window H1 → sequence variation rendah

### 2026-05-30-3_lstm_momentum_v4_trajectory
- **File**: `pipeline/05d_train_lstm_momentum_v4.py`
- **Deskripsi**: Trajectory features (ganti H4 snapshot dengan H1 trajectory), N=12 labels
- **Hasil**: ❌ Masih plateau F1 ~0.41. OHLCV ceiling informasi

---

## 2026-05-31

### 2026-05-31-1_lstm_v5_robust_scaler
- **File**: `pipeline/05e_train_lstm_momentum_v5.py`
- **Deskripsi**: RobustScaler + 11 fitur v2 (hapus volume_delta, ofi_raw, dll — extreme cross-coin scale mismatch)
- **Hasil**: ❌ F1 masih plateau ~0.41. Root cause: signal-to-noise terlalu rendah untuk 3-class momentum

### 2026-05-31-2_lstm_v6_final
- **File**: `pipeline/05f_train_lstm_momentum_v6.py`
- **Deskripsi**: Final attempt LSTM 3-class momentum — berbagai kombinasi parameter
- **Hasil**: ❌ Tidak ada improvement. PLATEAU final. **Keputusan**: Hentikan LSTM 3-class momentum

---

## 2026-06-01

### 2026-06-01-1_cascade_v2.5_hybrid_entry
- **File**: (perubahan di `config.py` — V2.5 Hybrid), `models/runs/cascade_v2.5_hybrid*/`
- **Deskripsi**: Longgarkan entry gate: LONG 0.75→0.69, SHORT 0.60→0.59, opposite_pen 0.99→0.65
- **Hasil**: ⚠️ Revive reasonable volume — WR 67.5%, PF 2.54 di holdout. Tapi 50+ scenario sweep = overfitting. Diarsipkan karena ic32_regime_v1 adalah hasil final yang lebih clean

---

## 2026-06-03

### 2026-06-03-1_cascade_dual_dominant_Z3
- **File**: `scratch/test_dual_mode.py`, `scratch/test_dual_model.py`, `core/cascade_utils.py`
- **Deskripsi**: Paradigma baru: LSTM dominant (abaikan FLAT, max(LONG,SHORT)). Mode dual_dominant Z3: LGBM≥0.65 + dominant≥0.35
- **Hasil**: ⚠️ Z3 dipilih dari holdout (test-set selection bias). Deploy ke production. Terlalu selektif: 81 trade/5koin/bln. **Dicabut** setelah ic32_regime_v1

---

## 2026-06-05

### 2026-06-05-1_simon_methodology_pipeline
- **File**: `pipeline/03b_ic_test.py` (AKTIF), `pipeline/03b_guardian_ic_test.py` (diarsipkan), `pipeline/03c_ic_decay_test.py`, `pipeline/03d_temporal_ic_test.py`
- **Deskripsi**: IC test Spearman rank + Marginal IC (Gram-Schmidt) + IC decay stability + temporal half-life
- **Hasil**: ✅ **107 fitur → 32 KEEP** + HMM argmax = 33 fitur ic32_regime_v1. STANDAR BARU feature selection

### 2026-06-05-2_triple_barrier_exploration
- **File**: `pipeline/03f_triple_barrier_relabel.py`, `pipeline/03g_rr_sweep.py`, `pipeline/03h_hybrid_relabel.py`
- **Deskripsi**: Triple barrier labeling (TP/SL/time) + RR ratio sweep + hybrid swing+TB
- **Hasil**: ❌ **Ditinggalkan**. TB labels 95% correlated dengan swing + bimodal FLAT (15% atau 80%). Swing labels lebih appropriate untuk crypto

### 2026-06-05-3_logistic_baseline_simon_step4
- **File**: `pipeline/04b_logistic_baseline.py`
- **Deskripsi**: Logistic Regression sebagai batas bawah (Simon Step 4: "does the model beat a simple linear model?")
- **Hasil**: ✅ F1 0.347 vs random 0.333 — konfirmasi ada non-linear signal. LGBM gain +0.243 F1

### 2026-06-05-4_meta_labeling_v1
- **File**: `pipeline/08_generate_meta_labels.py`, `pipeline/09_train_meta_model.py`
- **Deskripsi**: Meta-labeling: binary LGBM secondary filter prediksi trade outcome
- **Hasil**: ❌ AUC 0.63 tapi WR improvement OOS sangat kecil (+1.4pp). Root cause: meta-labels dari training simulation → target leak ke training set

### 2026-06-05-5_ic_test_per_regime
- **File**: `pipeline/12_ic_test_per_regime.py`
- **Deskripsi**: IC test features pada subset regime (TRENDING_UP, TRENDING_DOWN, RANGING)
- **Hasil**: ⚠️ 23 fitur trending UP/DN teridentifikasi. Digunakan untuk LGBM trending (lihat 2026-06-08-1)

---

## 2026-06-06

### 2026-06-06-1_meta_labeling_v2_oof_fix
- **File**: `pipeline/08_generate_meta_labels_v2.py`, `pipeline/09_train_lstm_meta.py`
- **Deskripsi**: Meta-labeling dengan walk-forward OOF labels (hindari in-sample bias). LSTM binary + Attention
- **Hasil**: ⚠️ AUC 0.594 (naik dari 0.580 setelah leak fix!). Konsep TERBUKTI — tapi tidak improve PnL di cascade

### 2026-06-06-2_meta_positioning
- **File**: `pipeline/10_train_meta_positioning.py`
- **Deskripsi**: Meta-model + positioning features dari Coinank
- **Hasil**: ❌ AUC degraded 0.594 → 0.534. Positioning features justru merusak sinyal meta-model

### 2026-06-06-3_logreg_meta_combiner
- **File**: `pipeline/12_train_logreg_meta.py`
- **Deskripsi**: Logistic Regression meta-combiner untuk ensemble LGBM + LSTM + HMM
- **Hasil**: ❌ **AUC 0.499 ≈ random**. LogReg tidak bisa combine sinyal non-linear

---

## 2026-06-07

### 2026-06-07-1_lstm_daily_binary_momentum
- **File**: `pipeline/11_train_lstm_daily.py` (TETAP — untuk arsitektur professional_v2)
- **Deskripsi**: LSTM Daily binary classifier (17 features, BiLSTM 96 hidden, seq=32 daily bars)
- **Hasil**: ⚠️ AUC 0.611 (8-fold CV). Signal valid tapi belum diintegrasikan ke production. Disimpan untuk `ic32_professional_v2`

### 2026-06-07-2_positioning_data_mining_start
- **File**: `pipeline/01c_fetch_positioning.py` (AKTIF — cron hourly)
- **Deskripsi**: 4 endpoint Binance+Bybit: taker_ratio, top_trader, global_ls, OI
- **Hasil**: ✅ 83 file, 21 koin. Target 4,000+ bar dalam 6 bulan (Des 2026)

### 2026-06-07-3_coinank_ic_test
- **File**: `scratch/ic_test_coinank.py`, `scratch/ic_test_coinank_v2.py`
- **Deskripsi**: IC test fitur Coinank (OI delta, LS ratio) vs swing labels
- **Hasil**: ⚠️ IC rendah, tidak prediktif. Coinank API tidak reliable → digantikan Binance public API

### 2026-06-07-4_kelly_regime_sizing
- **File**: `scratch/test_kelly_regime.py`, `core/position_sizing.py`
- **Deskripsi**: Kelly Criterion + Regime-Based position sizing
- **Hasil**: ❌ **GAGAL** — Kelly amplifies losses pada sistem tanpa genuine OOF edge. Fixed sizing $10/trade lebih aman

### 2026-06-07-5_lstm_daily_cascade_test
- **File**: `scratch/test_lstm_daily_cascade.py`
- **Deskripsi**: Integrasi LSTM Daily ke cascade sebagai bias filter
- **Hasil**: ⚠️ 0 trade diblokir — holdout 100% RANGING. Perlu trending market untuk test

### 2026-06-07-6_hmm_gate_lstm_test
- **File**: `scratch/test_hmmgate_sizemult.py`
- **Deskripsi**: HMM Gate: LSTM hanya aktif di TRENDING. Positioning size multiplier
- **Hasil**: ⚠️ HMM Gate ON vs OFF = 0 trade berubah. LSTM terlalu lemah untuk pengaruhi keputusan

---

## 2026-06-08

### 2026-06-08-1_lgbm_trending_regime_router
- **File**: `pipeline/04_train_lgbm_trending.py`, `pipeline/04_train_trend_momentum.py`, `pipeline/04_train_triple_barrier_lgbm.py`, `pipeline/04_train_momentum_ic38.py`, `pipeline/04_train_momentum_lgbm.py`, `pipeline/04_train_lgbm_hmm_probs.py`
- **Deskripsi**: 3 model spesialis per regime: SWING (ranging), TRENDING_UP (continuation, dir=1), TRENDING_DOWN (continuation, dir=-1)
- **Hasil**: ❌ **GAGAL di genuine WFV.** Pre-trained models +$7,747 di in-sample → -$762 di genuine OOF retrain. Overtrade 40x (2,880 vs 66 baseline). Solusi: FLIP alignment (REGIME_AWARE_ALIGNMENT)

### 2026-06-08-2_cascade_sweep_50_scenarios
- **File**: `experiments/cascade_sweep.py`, `experiments/run_pruned_lgbm.py`, `experiments/test_*.py` (20+ test files), `experiments/cascade_sweep_phase*.json`, `models/runs/cascade_v2.5_hybrid*/` (50+ directories)
- **Deskripsi**: Grid sweep masif: 56 konfigurasi cascade × 5 koin → 3 terbaik × 21 koin. Parameter: LSTM mode, trend alignment, threshold asymmetric, min_hold, activation_atr
- **Hasil**: ⚠️ Sweet spot ditemukan: hard_consensus + h4_trend, WR 67.5%, PF 2.54. Tapi 50+ scenario pada holdout yang sama = **MULTIPLE TESTING BIAS**. ic32_regime_v1 adalah hasil final yang lebih clean

### 2026-06-08-3_lstm_attention_bilstm_experiments
- **File**: `pipeline/A1_lstm_attention.py`, `pipeline/B1_label_adx_bilstm.py`, `pipeline/B2_adx_bilstm.py`
- **Deskripsi**: LSTM + Attention mechanism. BiLSTM dengan ADX-based labels
- **Hasil**: ❌ Tidak menambah value vs LSTM vanilla. Label ADX terlalu noisy

### 2026-06-08-4_coinank_fetch_attempts
- **File**: `pipeline/02c_fetch_coinank.py`, `pipeline/02d_fetch_coinank_extended.py`, `pipeline/02e_fetch_coinank_final.py`, `pipeline/02f_fetch_free_features.py`, `scratch/test_coinank_auth.py`
- **Deskripsi**: Berbagai attempt fetch data OI/LS dari Coinank API (free tier)
- **Hasil**: ❌ **Ditinggalkan**. API key expired, rate limit ketat, data tidak lengkap. Digantikan Binance/Bybit public API (`01c_fetch_positioning.py`)

### 2026-06-08-5_hmm_probs_vs_argmax
- **File**: `pipeline/03e_hmm_probs.py`
- **Deskripsi**: 4 HMM posterior probabilities vs 1 kolom argmax sebagai fitur LGBM
- **Hasil**: ❌ 4 probs PnL -45% vs argmax. 4 probs redundan (sum to 1). HMM argmax (`hmm_regime_enc`) tetap superior

### 2026-06-08-6_wfv_genuine_validation_scripts
- **File**: `scratch/wf_validation_genuine.py`, `scratch/wfv_jan2022_21coins.py`, `scratch/genuine_oof_3coins.py`, `scratch/extended_regime_genuine_oof.py`, `scratch/extended_regime_router.py`, `scratch/extended_oof_genuine.py`, `scratch/check_equivalence.py`, `scratch/compare_lstm_seq_results.py`
- **Deskripsi**: Berbagai script validasi Walk-Forward genuine OOF — expanding window, purged CV, retrain per fold
- **Hasil**: Semua konfirmasi: regime router GAGAL, LSTM tidak menambah value, trending models overtrade. **Pelajaran**: genuine WFV adalah standar emas — tidak bisa digantikan fixed model test

### 2026-06-08-7_positioning_engine_tests
- **File**: `scratch/test_pos_ab.py`, `scratch/test_pos_overlap.py`, `scratch/test_positioning_cascade.py`, `scratch/test_guardian_speed.py`, `scratch/fe_live_ic_test.py`
- **Deskripsi**: A/B test positioning engine, overlap check, cascade integration, speed benchmark
- **Hasil**: ⚠️ Positioning LS extreme → size × 0.50 valid (IC=0.16). Tapi dampak ke PnL kecil karena jarang trigger

### 2026-06-08-8_repo_cleanup
- **File**: Semua yang diarsipkan (265+ file)
- **Deskripsi**: Pembersihan masif — fokus ke `ic32_regime_v1` + arsitektur `ic32_professional_v2`
- **Hasil**: ✅ Repo bersih: 6 root files, 15 pipeline, 12 model runs, 12 model files

---

## 2026-06-15

### 2026-06-15-1_meta_fb_v2_closed
- **File**: `pipeline/08–09`, `14–16_*_meta_fb_v2.py`, `core/meta_labeling.py`
- **Deskripsi**: Meta-labeling LGBM binary gate di atas `tb_widyawardhana_v2` (flatboost_v2 + HMM). Simon Gate #1 + ablation + 3 varian fitur + soft multiplier
- **Hasil**: ❌ **CLOSED / NO-GO**. Holdout marginal IC t=0.9 (FAIL). Best meta arm -$4 vs primary_hmm +$276. Gate #2 tidak dijalankan. Detail: `EXPERIMENTS.md` §11

---

## Ringkasan Status Eksperimen

| Status | Count | Contoh |
|--------|-------|--------|
| ✅ **AKTIF (production)** | 12 | ic32_regime_v1, Guardian clean_v2, FLIP alignment, positioning engine |
| ✅ **TERBUKTI (research)** | 5 | IC test methodology, LSTM Daily AUC 0.611, meta-labeling konsep |
| ⚠️ **DATA LEAKAGE** | 10+ | Semua sweep 2026-05-27, cascade_v3.1 metrik, Guardian v3 88.9% |
| ❌ **GAGAL** | 30+ | LSTM momentum 3-class (6 versi), LGBM trending, LogReg meta, Kelly, Coinank |

## Pelajaran Kunci

1. **Genuine WFV > Fixed Model Test** — pre-trained model selalu overfit di in-sample
2. **OHLCV ceiling untuk momentum** — LSTM 3-class mentok F1 0.41. Perlu positioning data
3. **FLIP alignment > Regime router** — alignment sederhana lebih robust dari model spesialis
4. **Jangan sweep di holdout** — multiple testing bias. Parameter harus divalidasi di validation set independen
5. **LSTM H1 nyaris 0 kontribusi** — soft multiplier tidak pernah cross threshold
6. **Meta-labeling entry gate ditutup** — `tb_meta_fb_v2` gagal holdout IC; redundan dengan LGBM conf. Fokus: Guardian exit + fitur primary baru
7. **Swing labels > Triple Barrier** — untuk crypto dengan liquidation mechanics
