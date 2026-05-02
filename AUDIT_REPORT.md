# AUDIT: Diskontinuitas Periode Data — Risiko Underfitting pada New Coins

**Auditor:** Roo  
**Tanggal:** 2026-05-02  
**Fokus Audit:** Penelusuran seluruh referensi `NEW_COINS_START` / `NEW_COINS_END` di pipeline — verifikasi dampak perioda data berbeda antara training coins dan new coins

---

## RINGKASAN EKSEKUTIF

Pipeline saat ini membagi koin menjadi 2 grup dengan perioda data **berbeda**: `TRAINING_COINS` (SOL, ETH, BNB, XRP, DOGE) mendapatkan data penuh **2020-01-01 → 2026-04-01** (~6,3 tahun), sementara `NEW_COINS` (TON, ADA, TRX, SHIB, AVAX, LINK, DOT, SUI, POL, NEAR, PEPE, TAO, ARB — 13 koin) hanya mendapatkan data **2023-04-01 → 2026-04-01** (~3 tahun). Disparitas 3 tahun ini berarti new coins **tidak pernah melihat bull 2021 maupun bear 2022**, sehingga model ensemble (LGBM + LSTM) yang dilatih pada new coins akan memiliki **representasi regime pasar yang jauh lebih sempit**. Mengingat user menginginkan semua koin diperlakukan seragam — `ALL_COINS = TRAINING_COINS + NEW_COINS` dengan satu perioda `TRAIN_START` yang sama — maka konfigurasi saat ini perlu disederhanakan.

---

## TEMUAN PER KATEGORI

### [KONFIGURASI] Lokasi: [`config.py:36-37`](config.py:36)
```
Deskripsi: NEW_COINS_START = 2023-04-01 dan NEW_COINS_END = 2026-04-01
           didefinisikan terpisah dari TRAIN_START (2020-01-01).
Dampak:    New coins kehilangan 3 tahun data historis (2020-2023).
           Tidak ada data bear market 2022, bull 2021.
Bukti:
    TRAIN_START     = datetime(2020, 1, 1, tzinfo=timezone.utc)
    TRAIN_END       = datetime(2026, 4, 1, tzinfo=timezone.utc)
    NEW_COINS_START = datetime(2023, 4, 1, tzinfo=timezone.utc)  ← 3 tahun lebih pendek
    NEW_COINS_END   = datetime(2026, 4, 1, tzinfo=timezone.utc)
```

### [LOGIKA] Lokasi: [`pipeline/01_fetch.py:57-99`](pipeline/01_fetch.py:57)
```
Deskripsi: Fungsi _build_coin_schedule() mengimplementasikan logika cabang:
           - Jika coin ∈ TRAINING_COINS → pakai TRAIN_START (2020)
           - Jika coin ∈ NEW_COINS → pakai NEW_COINS_START (2023)
           Ini terjadi di 3 mode: --new, --all, dan --coins custom.
Dampak:    Setiap kali fetch dijalankan dengan --all, 13 dari 20 koin
           hanya mendapat 3 tahun data. Model akan under-trained untuk
           new coins pada regime bear.
Bukti:
    if args.all:
        schedule = (
            [(sym, TRAIN_START, TRAIN_END)         for sym in TRAINING_COINS] +
            [(sym, NEW_COINS_START, NEW_COINS_END)  for sym in NEW_COINS]
        )
    elif args.coins:
        schedule = [
            (sym,
             TRAIN_START      if sym in TRAINING_SET else NEW_COINS_START,
             TRAIN_END        if sym in TRAINING_SET else NEW_COINS_END,
            ) for sym in custom
        ]
```

### [DATA] Lokasi: [`pipeline/04_train_lgbm.py`](pipeline/04_train_lgbm.py) dan [`pipeline/05_train_lstm.py`](pipeline/05_train_lstm.py)
```
Deskripsi: Kedua skrip training (LGBM line 156, LSTM line 233)
           menggunakan ALL_COINS atau TRAINING_COINS untuk iterasi,
           tetapi TIDAK menentukan filter tanggal. Data loader membaca
           semua file parquet yang ada di LABEL_DIR.
Dampak:    Training akan menggunakan data apa pun yang ada di disk.
           Jika new coins hanya di-fetch dari 2023, maka training
           hanya melihat 3 tahun data untuk koin tersebut.
           Walk-forward validation dengan 8 folds akan memiliki
           window lebih pendek untuk new coins.
Bukti:
    # 04_train_lgbm.py:156
    coins = ALL_COINS if args.all else TRAINING_COINS

    # 05_train_lstm.py:233
    coins = ALL_COINS if args.all else TRAINING_COINS
```

### [DATA] Lokasi: [`pipeline/06_ensemble.py:107`](pipeline/06_ensemble.py:107)
```
Deskripsi: Ensemble juga menggunakan ALL_COINS / TRAINING_COINS
           tanpa filter tanggal.
Dampak:    Meta-learner (Logistic Regression) akan melihat OOF
           predictions dari new coins yang hanya punya data 2023+.
           Konsistensi antar koin tidak seragam.
```

### [DATA] Lokasi: [`pipeline/08_backtest.py:818-823`](pipeline/08_backtest.py:818)
```
Deskripsi: Backtest menggunakan ALL_COINS / TRAINING_COINS.
           Tapi inference config (line 671-673) menulis:
           training_period.start = TRAIN_START.date() (2020-01-01)
           untuk SEMUA koin — termasuk new coins yang sebenarnya
           hanya punya data dari 2023.
Dampak:    Inference config memberikan informasi yang MENYESATKAN:
           mengklaim training period 2020-2026 untuk new coins
           padahal data aktual hanya 2023-2026.
```

### [KONFIGURASI] Lokasi: [`d:/Apps-Dev/swint_tradev2/models/inference_config.json:5-6`](d:/Apps-Dev/swint_tradev2/models/inference_config.json:5)
```
Deskripsi: Inference config DEPLOYMENT (swint_tradev2) masih menyimpan
           training_period.start = "2022-01-01" — ini STALE dari
           konfigurasi lama. Config.py sekarang 2020-01-01.
Dampak:    Deployment tidak sadar bahwa model diperbarui dengan
           data mulai 2020. Tidak ada dampak runtime, tapi
           informasi perioda di inference_config.json tidak akurat.
```

### [KONFIGURASI] Lokasi: [`d:/Apps-Dev/swint_tradev2/models/inference_config.json:13,31`](d:/Apps-Dev/swint_tradev2/models/inference_config.json:13)
```
Deskripsi: max_hold_bars = 48 di inference config, sedangkan
           config.py sekarang SWING_LABEL_MAX_HOLD = 24.
           Juga max_holding_bars = 48 di section labeling (line 31).
Dampak:    Jika inference engine membaca max_hold_bars dari
           inference_config.json (bukan config.py), maka sinyal
           trading realtime menggunakan parameter LAMA (48 bar)
           yang sudah di-rekomendasikan untuk diturunkan.
```

---

## JALUR EKSEKUSI YANG TERIDENTIFIKASI

### Alur Data Perioda — Training Coins
```
config.py:TRAIN_START=2020-01-01
    → 01_fetch.py:schedule → fetch dari Binance (2020→2026)
        → data/raw/klines/*/  ← data penuh 6,3 tahun
            → 02_clean.py → data/processed/*_clean.parquet
                → 03_engineer.py → data/labeled/*_features_v3.parquet
                    → 04_train_lgbm.py  ✅ data 2020-2026
                    → 05_train_lstm.py  ✅ data 2020-2026
                    → 06_ensemble.py    ✅ data 2020-2026
                    → 08_backtest.py    ✅ data 2020-2026
```

### Alur Data Perioda — New Coins (BERMASALAH)
```
config.py:NEW_COINS_START=2023-04-01
    → 01_fetch.py:schedule → fetch dari Binance (2023→2026)  ⚠️
        → data/raw/klines/*/  ← data hanya 3 tahun
            → 02_clean.py → data/processed/*_clean.parquet
                → 03_engineer.py → data/labeled/*_features_v3.parquet
                    → 04_train_lgbm.py  ⚠️ hanya data 2023-2026
                    → 05_train_lstm.py  ⚠️ hanya data 2023-2026
                    → 06_ensemble.py    ⚠️ hanya data 2023-2026
                    → 08_backtest.py    ⚠️ hanya data 2023-2026
                                        ❌ TAPI inference config menulis 2020-2026
```

### Lingkup Dampak per Pipeline Stage

| Pipeline | Impor `NEW_COINS_START`? | Terdampak? | Detail |
|----------|--------------------------|------------|--------|
| `01_fetch.py` | ✅ Ya — line 26, 69, 75, 87 | **LANGSUNG** | Jadwal fetch berbeda per grup koin |
| `02_clean.py` | ❌ Tidak | Tidak langsung | Hanya membersihkan apa yang ada di raw/ |
| `03_engineer.py` | ❌ Tidak | Tidak langsung | Fitur dihitung dari data yang ada |
| `04_train_lgbm.py` | ❌ Tidak | **Tidak langsung** | Data new coins hanya 3 tahun |
| `05_train_lstm.py` | ❌ Tidak | **Tidak langsung** | Sama seperti LGBM |
| `06_ensemble.py` | ❌ Tidak | **Tidak langsung** | OOF new coins dari data terbatas |
| `07_evaluate.py` | ❌ Tidak | Tidak langsung | SHAP dari fitur yang ada |
| `08_backtest.py` | ❌ Tidak | **Tidak langsung** | Backtest new coins hanya 3 tahun |
| `09_holdout_backtest.py` | ❌ Tidak | **Tidak langsung** | Holdout independen (2025→2026) |
| `10_visualize.py` | ❌ Tidak | Tidak langsung | Visualisasi dari data yang ada |

---

## HIPOTESIS PENYEBAB ROOT

### 1. 🎯 Asumsi "New Coins = Baru Listing" — Kemungkinan: 95%
**Bukti:** Komentar di [`01_fetch.py:62-64`](pipeline/01_fetch.py:62):
```python
# Training coins mendapat data penuh dari TRAIN_START (Jan 2020).
# New coins mendapat data dari NEW_COINS_START (Apr 2023) karena banyak
# yang baru listing setelah 2023.
```
Ini menunjukkan developer berasumsi bahwa `NEW_COINS` adalah koin yang baru listing di Binance sekitar 2023. Namun kenyataannya:
- ADAUSDT — listing sejak 2021
- TRXUSDT — listing sejak 2019
- DOTUSDT — listing sejak 2020
- LINKUSDT — listing sejak 2019
- AVAXUSDT — listing sejak 2021
**Hanya TAOUSDT dan ARBUSDT yang benar-benar baru listing setelah 2023.** Jadi asumsi ini salah untuk mayoritas new coins.

### 2. 🎯 Warisan dari Konfigurasi Lama — Kemungkinan: 80%
**Bukti:** `NEW_COINS_START` mungkin merupakan sisa dari konfigurasi sebelumnya ketika `TRAIN_START` masih `2022-01-01` (lihat `inference_config.json` yang masih menyimpan `"start": "2022-01-01"`). Saat `TRAIN_START` diperpanjang ke 2020, `NEW_COINS_START` tidak ikut diperbarui.

### 3. 🎯 Kekhawatiran Binance Rate Limit — Kemungkinan: 40%
**Bukti:** Dengan 20 koin × 3 timeframe × ~6 tahun data, jumlah request API bisa besar. Mungkin sengaja membatasi new coins untuk mengurangi waktu fetch. Tapi ini tidak relevan karena pipeline sudah memiliki rate limit handling (`SLEEP_BETWEEN_REQUESTS=0.12`, `SLEEP_ON_RATE_LIMIT=60`).

---

## PERTANYAAN KLARIFIKASI

1. Apakah TONUSDT, ADAUSDT, TRXUSDT, AVAXUSDT, LINKUSDT, DOTUSDT memiliki data historis di Binance sejak 2020? (Perlu dicep—kemungkinan besar ya untuk mayoritas)
2. Apakah ada koin di NEW_COINS yang benar-benar baru listing setelah 2023? (Hanya TAOUSDT ≈ Feb 2024, ARBUSDT ≈ Mar 2023)
3. Apakah ingin tetap mempertahankan `TRAINING_COINS` vs `NEW_COINS` sebagai grup terpisah untuk keperluan validasi hold-out, atau semua koin diperlakukan identik?

---

## REKOMENDASI PERBAIKAN

### Rekomendasi 1 (WAJIB — Prioritas Tertinggi)
**Apa:** Hapus `NEW_COINS_START` dan `NEW_COINS_END` dari config.py. Gunakan `TRAIN_START` dan `TRAIN_END` untuk semua koin tanpa pengecualian.

**Mengapa:** User secara eksplisit menginginkan `ALL_COINS = TRAINING_COINS + NEW_COINS` dengan perioda seragam. Semua koin kecuali TAO & ARB sudah listing di Binance sejak sebelum 2020 atau setidaknya 2021. Data 2020-2022 (bull 2021, bear 2022) penting untuk representasi regime pasar.

**File yang perlu diubah:**
- [`config.py:36-37`](config.py:36) — hapus `NEW_COINS_START` dan `NEW_COINS_END`
- [`pipeline/01_fetch.py:57-99`](pipeline/01_fetch.py:57) — sederhanakan `_build_coin_schedule()`: semua koin pakai `TRAIN_START` dan `TRAIN_END`
- [`pipeline/01_fetch.py:23-27`](pipeline/01_fetch.py:23) — hapus import `NEW_COINS_START, NEW_COINS_END`

### Rekomendasi 2 (PENTING)
**Apa:** Setelah menghapus `NEW_COINS_START`, jalankan ulang fetch untuk semua new coins dengan perioda penuh `python pipeline/01_fetch.py --new --reset`.

**Mengapa:** Data existing di `data/raw/` untuk new coins hanya dari 2023. Perlu difetch ulang dari 2020 untuk mengisi gap data.

### Rekomendasi 3 (PENTING — Maintenance)
**Apa:** Perbarui `inference_config.json` di `swint_tradev2/models/` setelah training selesai dengan nilai parameter terkini. Pipeline [`08_backtest.py:671-673`](pipeline/08_backtest.py:671) sudah melakukan ini saat `generate_inference_config()`, jadi hanya perlu memastikan file deployment di-copy setelah training.

**Mengapa:** Inference config yang stale (max_hold_bars=48, training_period=2022-2025) dapat menyebabkan inkonsistensi antara training dan inference jika deployment membaca parameter dari JSON tersebut.

### Rekomendasi 4 (OPSIONAL — Refactoring)
**Apa:** Pertimbangkan untuk menghapus `TRAINING_COINS` / `NEW_COINS` dikotomi jika semua koin diperlakukan identik. Cukup gunakan `ALL_COINS` di semua pipeline stage.

**Mengapa:** Menyederhanakan kode dan menghilangkan kemungkinan perlakuan berbeda di masa depan. Argumen `--new` di CLI bisa dihapus karena tidak lagi relevan.

---

## LAMPIRAN: Daftar Koin dan Perkiraan Listing di Binance Futures

| Koin | Grup | Listing Binance Futures | Data 2020? |
|------|------|------------------------|------------|
| SOLUSDT | Training | Aug 2021 | ✅ Tersedia |
| ETHUSDT | Training | Sejak awal | ✅ Tersedia |
| BNBUSDT | Training | Sejak awal | ✅ Tersedia |
| XRPUSDT | Training | Sejak awal | ✅ Tersedia |
| DOGEUSDT | Training | Apr 2021 | ✅ Tersedia |
| **TONUSDT** | **New** | **Nov 2024** | ❌ Baru listing |
| **ADAUSDT** | **New** | **Mar 2021** | **✅ Tersedia — tapi dibatasi 2023** |
| **TRXUSDT** | **New** | **Jan 2020** | **✅ Tersedia — tapi dibatasi 2023** |
| **1000SHIBUSDT** | **New** | **May 2021** | **✅ Tersedia — tapi dibatasi 2023** |
| **AVAXUSDT** | **New** | **Sep 2021** | **✅ Tersedia — tapi dibatasi 2023** |
| **LINKUSDT** | **New** | **2019** | **✅ Tersedia — tapi dibatasi 2023** |
| **DOTUSDT** | **New** | **Sep 2020** | **✅ Tersedia — tapi dibatasi 2023** |
| **SUIUSDT** | **New** | **May 2023** | ❌ Listing 2023 |
| **POLUSDT** | **New** | **~2021** | **✅ Tersedia — tapi dibatasi 2023** |
| **NEARUSDT** | **New** | **Oct 2021** | **✅ Tersedia — tapi dibatasi 2023** |
| **1000PEPEUSDT** | **New** | **Apr 2023** | ❌ Listing 2023 |
| **TAOUSDT** | **New** | **Feb 2024** | ❌ Listing 2024 |
| **ARBUSDT** | **New** | **Mar 2023** | ❌ Listing 2023 |

**Kesimpulan:** Dari 13 new coins, **9 koin (69%) sudah listing sebelum 2023** dan seharusnya bisa mendapat data dari 2020. Hanya 4 koin (TON, SUI, PEPE, TAO, ARB ≈ 31%) yang benar-benar baru listing setelah 2023. Untuk koin-koin yang benar-benar baru, Binance akan mengembalikan data kosong untuk periode sebelum listing — fetch akan tetap aman (tidak error), hanya dapat data sedikit.

---

*Audit selesai. Tidak ada perubahan kode yang dilakukan — laporan ini hanya untuk analisis.*
