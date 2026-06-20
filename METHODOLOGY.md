# Metodologi Training yang Benar
**Berlaku untuk:** SwingTrade pipeline (LGBM + Guardian + HMM + Dynamic Sizing)
**Dibuat:** 2026-06-15 — berdasarkan audit robustness dan lessons learned

---

## Prinsip Utama

**Satu aturan yang tidak boleh dilanggar:**
> Setiap keputusan (threshold, parameter, pilihan model) harus dibuat
> HANYA berdasarkan data training period. Holdout adalah amplop tersegel
> yang dibuka sekali di akhir untuk konfirmasi — bukan alat development.

---

## A. Pembagian Data

```
|──────────── Training Period ────────────|── Holdout ──|── Live ──|
  Jan 2020                      Mar 2026   Apr-Jun 2026   Jul 2026+
                                           (tersegel)
```

### Aturan pembagian
- Split SELALU berdasarkan waktu, tidak pernah random
- Training period: semua data sebelum `TRAIN_CUTOFF_DATE`
- Holdout: minimal 3 bulan, idealnya 6 bulan, berisi setidaknya ~500 trades
- Holdout dipilih sebelum development dimulai — jangan disesuaikan setelah melihat hasilnya
- Live trading secara otomatis menjadi "holdout berjalan" setelah holdout resmi habis

### Berapa lama holdout yang cukup?
| Trades per bulan | Holdout minimum | Alasan |
|-----------------|-----------------|--------|
| < 100 | 6 bulan | Butuh ~600 trades untuk WR estimate ±5% |
| 100-300 | 3 bulan | ~600 trades cukup |
| > 300 | 2 bulan | ~600 trades terpenuhi |

Untuk pipeline ini (~367 trades/bulan di 21 koin): **2-3 bulan holdout cukup**,
asalkan tidak dipakai untuk keputusan apapun.

---

## B. Feature Engineering (tanpa leakage)

### Aturan wajib
1. Semua fitur harus hanya menggunakan data dari bar ≤ T untuk prediksi di bar T
2. Rolling windows: `df["feat"].rolling(N).mean()` — backward-looking, aman
3. Lag/shift: `df["feat"].shift(N)` dengan N ≥ 1 — aman. `shift(-N)` adalah leakage
4. Fitur berbasis H4/swing: konfirmasi bar i memerlukan bar i+lookback →
   wajib `shift(lookback)` setelah komputasi sebelum dipakai sebagai fitur
5. CVD cumsum: aman secara temporal tapi distribusinya bergeser seiring waktu.
   Pertimbangkan rolling z-score dengan window terbatas (e.g., 500 bar) daripada
   z-score terhadap seluruh history untuk stabilitas distribusi

### Checklist sebelum menambah fitur baru
- [ ] Apakah fitur ini bisa dihitung pada saat entry tanpa melihat masa depan?
- [ ] Apakah ada `.shift(-N)` tersembunyi dalam fungsi helper?
- [ ] Apakah fitur ini menggunakan label/outcome sebagai input?
- [ ] Jika fitur memakai future data secara sengaja (e.g., H4 swing confirmation),
      apakah sudah di-lag dengan benar?

---

## C. Walk-forward CV dengan Purging

### Struktur fold yang benar

```
Fold 1: [──train──]  [purge] [──val──]
Fold 2: [────train────]  [purge] [──val──]
Fold 3: [──────train──────]  [purge] [──val──]
```

- **Expanding window** (bukan rolling): setiap fold menambah data training
- **Purge gap**: minimal `max_holding_bars` bar antara akhir train dan awal val.
  Untuk pipeline ini: purge = 36 bar (= MAX_HOLDING_BARS).
  Alasan: label Triple Barrier bar T menggunakan bar T+1 sampai T+36.
  Tanpa purge, bar di training yang labelnya overlap ke val = leakage.
- **Multi-coin**: purge berdasarkan timestamp universal, bukan per-coin.
  Semua koin yang share timestamp T masuk ke sisi yang sama.
- **Minimum folds**: 5-8 fold. Lebih dari 8 fold: fold awal terlalu kecil.

### Scaler dalam CV

```python
# SALAH — scaler melihat val saat fitting
scaler = StandardScaler()
X_all_scaled = scaler.fit_transform(X_all)  # ← leakage
for train_idx, val_idx in folds:
    model.fit(X_all_scaled[train_idx], y[train_idx])

# BENAR — scaler hanya melihat training fold
for train_idx, val_idx in folds:
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_all[train_idx])
    X_val   = scaler.transform(X_all[val_idx])      # ← transform only
    model.fit(X_train, y[train_idx])
```

### Early stopping dalam CV

```python
# BENAR: eval_set adalah val fold yang sudah di-transform terpisah
model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],   # val dari fold yang sama, scaler yang sama
    callbacks=[lgb.early_stopping(100)]
)
```

---

## D. OOF Predictions — Pusat Semua Keputusan

Setelah CV selesai, gabungkan prediksi dari setiap val fold → OOF predictions.
Ini adalah prediksi yang "jujur" karena setiap bar diprediksi tanpa melihat dirinya sendiri saat training.

```python
oof_preds  = np.zeros((N_bars, N_classes))
oof_confs  = np.zeros(N_bars)
oof_labels = y_all.copy()

for fold_idx, (train_idx, val_idx) in enumerate(folds):
    # ... train model ...
    oof_preds[val_idx]  = model.predict_proba(X_val)
    oof_confs[val_idx]  = oof_preds[val_idx].max(axis=1)
```

### Apa yang boleh dan harus dilakukan dengan OOF:

**BOLEH (semua keputusan ini aman menggunakan OOF):**
- Pilih threshold LGBM (0.50 vs 0.55 vs 0.58) → sweep OOF simulation
- Bandingkan Guardian variant (profit vs momentum vs continuation) → OOF trades
- Tune dynamic sizing parameter (HIGH_THR, LOW_THR) → OOF trades
- Kalibrasi confidence (Platt scaling) → fit calibrator pada OOF predictions
- Pilih fitur (IC selection, importance) → OOF IC score
- Bandingkan arsitektur model (LGBM vs LSTM vs cascade) → OOF simulation

**TIDAK BOLEH:**
- Melihat angka holdout, lalu memutuskan untuk mengubah sesuatu
- Menggunakan holdout sebagai "validation set tambahan"

### OOF Simulation untuk threshold sweep

```python
# Contoh: sweep threshold menggunakan OOF
results = {}
for thr_long in [0.50, 0.52, 0.55, 0.58]:
    for thr_short in [0.50, 0.52, 0.55]:
        # Buat signal dari OOF predictions
        signals = np.where(oof_confs[:, 2] >= thr_long, LONG,
                  np.where(oof_confs[:, 0] >= thr_short, SHORT, FLAT))
        # Simulasi trading pada OOF bars
        rep = full_trading_report(signals, oof_labels, ...)
        results[(thr_long, thr_short)] = rep["pnl"]

best_thr = max(results, key=results.get)
# → Gunakan best_thr sebagai konfigurasi final
```

---

## E. Urutan Training yang Benar

### Langkah 1 — Feature engineering (sekali, tidak berulang-ulang)

- Tambah/hapus fitur berdasarkan domain knowledge dan IC analysis
- IC dihitung terhadap OOF predictions atau label CV fold
- Jangan tambah fitur karena "terlihat bagus di holdout"

### Langkah 2 — Train LGBM entry model via walk-forward CV

```
Input  : training bars, semua koin, fitur-fitur final
Output : OOF predictions (proba per bar) + OOF confidence
         + model terbaik (trained pada seluruh training period)
```

- Hyperparameter tuning (num_leaves, learning_rate, dll) via OOF metric (F1/PnL)
- Threshold selection (thr_long, thr_short) via OOF simulation
- Simpan: `lgbm_final.pkl` ditraining pada **seluruh** training period
  dengan hyperparameter terbaik dari CV

### Langkah 3 — Generate OOF trades untuk Guardian training

```
Input  : OOF signals (dari threshold terbaik), bar data training
Output : Daftar trade OOF (entry_bar, exit_bar, pnl_aktual, bar-by-bar features)
```

- Jalankan `full_trading_report()` pada OOF signals → OOF trades
- Ini penting: Guardian harus dilatih pada **OOF trades**, bukan in-sample trades
- Alasan: kalau Guardian dilatih pada in-sample trade, model entry sudah "melihat"
  data itu saat training → Guardian belajar dari trade yang biased

```python
# OOF trades (benar)
oof_signals = apply_threshold(oof_preds, best_thr)
oof_trades  = full_trading_report(oof_signals, oof_labels, ...)["trades"]
# → Gunakan oof_trades sebagai training data Guardian

# In-sample trades (salah untuk training Guardian)
insample_signals = lgbm_final.predict(X_all_train)
insample_trades  = full_trading_report(insample_signals, ...)["trades"]
# → Jangan pakai ini — model sudah "hafal" X_all_train
```

### Langkah 4 — Label OOF trades untuk Guardian

Label Guardian (HOLD / PARTIAL_EXIT / FULL_EXIT) per bar dalam setiap OOF trade
dibuat berdasarkan future bars dalam trade tersebut — ini intentional dan benar.

Purge Guardian: gap antara akhir trade training dan awal trade validasi ≥ MAX_HOLDING_BARS.

### Langkah 5 — Train Guardian via walk-forward CV

```
Input  : OOF trades dengan label, static features, dynamic features
Output : OOF Guardian predictions + Guardian model final
```

- Scaler StandardScaler harus di-fit PER FOLD, bukan pada seluruh dataset
- Bandingkan Guardian variant (min_hold, exit_threshold, activation_atr)
  menggunakan OOF simulation Guardian — bukan holdout
- Guardian final di-train pada seluruh OOF trades setelah variant terpilih

### Langkah 6 — Train HMM regime

```
Input  : Training bars, fitur regime (vol, trend, dll)
Output : HMM model (fitted hanya pada training period)
         OOF regime labels (walk-forward OOF, fit per fold)
```

- OOF HMM labels dipakai sebagai fitur untuk LGBM (jika `hmm_regime_enc` adalah fitur)
- Untuk holdout: HMM di-predict menggunakan model yang di-fit hanya pada training period
- Tidak ada parameter HMM yang di-tune terhadap holdout

### Langkah 7 — Kalibrasi confidence (Platt / Isotonic)

```
Input  : OOF confidence scores + OOF outcomes (win/loss)
Output : Kalibrasi model (Platt atau Isotonic)
```

- Fit calibrator pada OOF (conf, win_label) pairs → ini bersih dan unbiased
- Bandingkan Platt vs Isotonic via OOF calibration error (MAE/ECE)
- Simpan calibrator untuk inference

### Langkah 8 — Dynamic sizing (sweep via OOF simulation)

```
Input  : OOF trades, OOF confidence, OOF regime, OOF calibrated p
Output : Optimal sizing config (HIGH_THR, LOW_THR, atau Kelly params)
```

- Sweep semua parameter sizing menggunakan OOF trades — tidak menyentuh holdout
- Pilih konfigurasi terbaik berdasarkan OOF risk-adjusted return

### Langkah 9 — Freeze semua konfigurasi

Setelah langkah 1-8 selesai, catat dengan tepat:
- Model files (lgbm.pkl, guardian.pkl, scaler.pkl)
- Threshold final (thr_long, thr_short, guardian_thr)
- Sizing config (HIGH_THR, LOW_THR atau Kelly params)
- HMM config
- Semua parameter inference_config.json

**Tidak ada yang boleh diubah setelah ini**, kecuali kembali ke Fase development
dengan holdout baru.

### Langkah 10 — Evaluasi holdout SEKALI

```
Input  : Config frozen, holdout bars (belum pernah dilihat)
Output : Angka final: WR, PF, PnL, Sharpe — estimasi unbiased
```

- Jalankan sekali tanpa modifikasi apapun
- Catat hasilnya
- Jika hasilnya tidak memuaskan: JANGAN tune → kembali ke development
  dengan holdout baru di periode berikutnya

---

## F. Kapan Harus Retrain?

### Trigger retrain
1. WR live turun di bawah 55% selama 2 bulan berturut-turut
2. Distribusi confidence live bergeser signifikan dari training
   (KL-divergence atau mean conf turun >10%)
3. PF live < 1.5 selama 6 minggu berturut-turut
4. Ada perubahan pasar struktural (delisting koin, perubahan regulasi besar)

### Proses retrain
- Pindahkan `TRAIN_CUTOFF_DATE` maju (tambah data baru ke training)
- Ulangi langkah 1-10 dengan holdout baru (periode setelah training cutoff baru)
- Data live trading selama periode lama otomatis masuk sebagai training data baru

---

## G. Ringkasan: Mana yang Pakai Data Apa?

| Keputusan | Data yang dipakai | Alasan |
|-----------|------------------|--------|
| Feature selection | OOF IC / CV metric | OOF unbiased terhadap bar itu sendiri |
| Hyperparameter LGBM | OOF PnL simulation | Aman karena setiap bar diprediksi OOS |
| Threshold thr_long/short | OOF PnL simulation | Sama di atas |
| Pilih Guardian variant | OOF Guardian simulation | OOF trades dari OOF signals |
| Guardian hyperparameter | OOF Guardian CV metric | Guardian CV dengan scaler per fold |
| Calibration (Platt) | OOF (conf, win) pairs | Fit pada data yang sudah "OOS" |
| Dynamic sizing params | OOF trades simulation | Tidak perlu menyentuh holdout |
| Evaluasi akhir | Holdout (sekali saja) | Konfirmasi generalisasi temporal |
| Monitoring live | Live trades | Holdout de facto berjalan |

---

## H. Kesalahan yang Paling Sering Terjadi

| Kesalahan | Akibat | Pencegahan |
|-----------|--------|------------|
| Tune threshold di holdout | Angka holdout = upper bound optimistik, bukan prediksi | Pakai OOF untuk semua tuning |
| Lihat holdout → tidak suka → ganti model → eval holdout lagi | Multiple testing contamination | Satu model, satu evaluasi |
| Scaler di-fit pada seluruh dataset | CV metric inflated | Fit scaler dalam loop fold |
| Guardian dilatih pada in-sample trades | Guardian belajar dari trade biased | Pakai OOF trades untuk train Guardian |
| Holdout terlalu pendek (<200 trades) | Variance terlalu tinggi, WR bisa ±15% dari chance | Minimal 3 bulan atau 600 trades |
| CV fold terlalu sedikit (3 fold) | OOF coverage rendah, estimate tidak stabil | Minimal 5 fold |
| Tidak ada purge gap | Label bocor lintas fold | Purge = MAX_HOLDING_BARS |

---

## I. Struktur File yang Disarankan

```
pipeline/
  01_fetch.py              # data fetching
  02_label.py              # Triple Barrier labeling
  03_features.py           # feature engineering
  04_cv_lgbm.py            # walk-forward CV → OOF predictions + model final
  05_oof_trades.py         # generate OOF trades dari OOF signals
  06_train_guardian.py     # train Guardian pada OOF trades (CV per fold, scaler per fold)
  07_select_thresholds.py  # sweep threshold via OOF simulation → pilih best
  08_select_sizing.py      # sweep dynamic sizing via OOF trades → pilih best
  09_calibrate.py          # fit Platt calibrator pada OOF (conf, win) pairs
  10_train_final.py        # train semua model pada SELURUH training period
  11_holdout_eval.py       # evaluasi holdout SEKALI — jangan dijalankan berulang
  12_deploy.py             # package model files untuk production

config_frozen/             # snapshot config saat holdout dievaluasi
  inference_config.json
  threshold_config.json
  sizing_config.json
```

`11_holdout_eval.py` sebaiknya memiliki guard:
```python
HOLDOUT_ALREADY_EVALUATED = True  # ← set True setelah pertama kali dijalankan
if HOLDOUT_ALREADY_EVALUATED:
    raise RuntimeError("Holdout sudah dievaluasi. Jangan jalankan ulang.")
```

---

*Metodologi ini mengikuti prinsip dari Advances in Financial Machine Learning (Lopez de Prado, 2018)
dan best practices temporal cross-validation untuk time series.*
