# Simon Methodology — IC Test & Research Pipeline
*2026-06-04 | Riset Pemodelan Crypto Trading*

---

## Apa itu IC?

**Information Coefficient (IC)** = seberapa kuat sebuah fitur memprediksi target.
Diukur dengan Spearman rank correlation antara nilai fitur dan label ke depan.

```
IC = spearman_corr(fitur, target)

IC = 0.00  → tidak ada sinyal
IC = 0.05  → sinyal lemah tapi ada
IC = 0.15  → sinyal kuat (jarang di pasar)
IC = 1.00  → prediksi sempurna (tidak mungkin)
```

Target di project ini: label ordinal `SHORT=-1 / FLAT=0 / LONG=+1`
IC negatif artinya fitur tinggi → SHORT (mean-reversion pattern, wajar untuk RSI).

---

## Dua Jenis IC

### 1. Standalone IC
> "Apakah fitur ini berguna kalau diukur sendiri?"

Korelasi langsung antara fitur dan target tanpa memperhitungkan fitur lain.
**Masalah**: tidak mendeteksi duplikasi. Dua fitur yang sangat berkorelasi satu sama lain bisa keduanya punya standalone IC tinggi, padahal hanya membawa satu sinyal.

### 2. Marginal IC
> "Apakah fitur ini masih berguna setelah fitur lain sudah diketahui?"

Diukur setelah fitur-fitur lain di-*orthogonalisasi* keluar (Gram-Schmidt sequential).
Ini yang dipakai Jim Simons di Renaissance — **marginal IC** yang menentukan apakah fitur layak masuk model.

---

## Algoritma: Sequential Orthogonalization (Gram-Schmidt)

```
1. Hitung standalone IC semua fitur terhadap target
2. Pilih fitur dengan IC tertinggi → masukkan ke set S
3. Orthogonalisasi semua fitur sisa terhadap fitur terpilih:
     fitur_j_baru = fitur_j - (cov(j, pivot) / var(pivot)) × pivot
4. Orthogonalisasi target juga (hapus varians bersama)
5. Ulangi dari langkah 1 dengan fitur dan target yang sudah di-orthogonalisasi
6. IC yang dihitung di setiap iterasi = MARGINAL IC
```

Hasilnya: setiap fitur dapat satu nilai marginal IC yang benar-benar mengukur kontribusi **unik**-nya.

---

## Threshold Keputusan

| Kriteria | Threshold | Alasan |
|----------|-----------|--------|
| Standalone IC | ≥ 0.02 | IC di bawah ini terlalu kecil untuk dipercaya |
| t-statistic | ≥ 2.0 | Signifikansi statistik (dengan koreksi autocorrelation) |
| Marginal IC | ≥ 0.01 | Kontribusi unik minimal yang masih bermakna |

**Catatan t-stat**: data H1 memiliki autocorrelation tinggi. Effective N dihitung sebagai N/24
(bukan N mentah) untuk menghindari false significance dari 156k samples.

### Kategori Keputusan

| Verdict | Kondisi | Tindakan |
|---------|---------|----------|
| **KEEP** | Standalone ✓ DAN Marginal ✓ | Masukkan ke model |
| **REDUNDANT** | Standalone ✓ TAPI Marginal ✗ | Pertimbangkan drop atau orthogonalisasi |
| **DROP** | Standalone ✗ | Buang — tidak ada sinyal bahkan secara standalone |
| **WEAK** | Standalone ✗ TAPI Marginal ✓ | Suppressor variable — perlu investigasi lebih |

---

## Hasil: LSTM H1 (12 Fitur)
*156,480 bars, 20 koin, holdout Nov 2025 – Apr 2026*

| Feature | Standalone IC | t-stat | Marginal IC | Verdict |
|---------|--------------|--------|-------------|---------|
| `rsi_6` | -0.1519 | -12.41 | -0.1519 | **KEEP** |
| `log_ret_5` | -0.1424 | -11.62 | -0.0797 | **KEEP** |
| `stochrsi_k` | -0.1308 | -10.65 | +0.0082 | **REDUNDANT** |
| `log_ret_20` | -0.0774 | -6.26 | -0.0188 | **KEEP** |
| `h1_return` | -0.0534 | -4.32 | -0.0343 | **KEEP** |
| `volume_delta` | -0.0396 | -3.20 | -0.4602 | **KEEP** |
| `ofi_raw` | -0.0247 | -2.00 | -0.4679 | **DROP** |
| `bars_since_BOS` | -0.0210 | -1.70 | +0.0217 | **DROP** |
| `vol_ratio_20` | +0.0196 | +1.58 | +0.0090 | **DROP** |
| `vwdp_smooth` | -0.0168 | -1.36 | -0.4667 | **DROP** |
| `atr_14_h1` | -0.0091 | -0.73 | -0.2541 | **DROP** |
| `ofi_acceleration` | -0.0049 | -0.40 | +0.4725 | **DROP** |

### Temuan Kunci LSTM

**KEEP (5 fitur)**: `rsi_6`, `log_ret_5`, `log_ret_20`, `h1_return`, `volume_delta`

**REDUNDANT (1)**: `stochrsi_k`
- Standalone IC tinggi (0.13) tapi marginal IC hanya 0.008
- StochRSI secara matematis diturunkan dari RSI → informasinya sudah tercakup di `rsi_6`
- Rekomendasi: drop sepenuhnya

**DROP (6 fitur)**: `ofi_raw`, `bars_since_BOS`, `vol_ratio_20`, `vwdp_smooth`, `atr_14_h1`, `ofi_acceleration`
- Tidak melewati threshold standalone — tidak ada sinyal prediktif terhadap label momentum
- Menjelaskan sebagian mengapa F1 LSTM v4.3 ≈ random: 6 dari 12 slot adalah noise

**Implikasi**:
- 7 slot kosong = peluang untuk fitur yang lebih bermakna
- Kandidat pengisi harus lolos IC test dulu sebelum masuk (lihat roadmap di bawah)

---

## Pipeline Riset: Saat Ini vs Jim Simons

### Pendekatan Saat Ini (Bottom-Up)

```
Intuisi/Domain Knowledge
        │
        ▼
  Feature Engineering      ← buat fitur dulu karena "masuk akal"
        │
        ▼
  Training Model           ← latih semua fitur sekaligus
        │
        ▼
  Evaluasi (F1, WR)        ← baru tahu apakah fitur berguna
        │
        ▼
  Tune / Iterasi
```

**Masalah**: baru tahu fitur berguna atau tidak *setelah* melatih model.
Kalau model jelek, tidak jelas: masalah di fitur, arsitektur, label, atau data?

---

### Pendekatan Jim Simons (Signal-First, Top-Down)

```
Hipotesis / Observasi pasar
        │
        ▼
① SIGNAL DISCOVERY          ← apakah sinyal ini ada di data?
   Ukur IC mentah dulu.
   Kalau IC < threshold → STOP, jangan lanjut.
        │
        ▼
② SIGNAL VALIDATION         ← apakah sinyal ini stabil lintas waktu?
   IC Decay test (bagi data jadi 6 window temporal)
   Kalau IC tidak konsisten → STOP.
        │
        ▼
③ SIGNAL INDEPENDENCE       ← apakah sinyal ini benar-benar baru?
   Marginal IC vs sinyal yang sudah ada.
   Kalau marginal IC kecil → DROP.
        │
        ▼
④ MODEL SEDERHANA DULU      ← validasi dengan model paling simpel
   Linear / logistic regression.
   Kalau model simpel tidak bisa tangkap → arsitektur kompleks tidak akan menolong.
        │
        ▼
⑤ KOMPLEKSITAS BERTAHAP     ← tambah kompleksitas hanya kalau terbukti perlu
   Tree → Shallow NN → Deep NN
   Setiap langkah harus terbukti meningkatkan OOS metric.
        │
        ▼
⑥ ENSEMBLE TEST             ← apakah sinyal ini meningkatkan ensemble?
   Kalau ensemble IC tidak naik → tidak masuk production.
```

---

### Perbandingan Langkah per Langkah

| Langkah | Saat Ini | Jim Simons |
|---------|----------|------------|
| **Mulai dari** | Intuisi → langsung buat fitur | Hipotesis → ukur IC dulu |
| **Gate pertama** | Tidak ada — semua fitur masuk training | IC test — fitur gagal tidak dilanjut |
| **Tahu fitur berguna** | Setelah training selesai | Sebelum training dimulai |
| **Model pertama** | Langsung kompleks (LGBM + LSTM) | Linear regression dulu |
| **Tambah kompleksitas** | Berdasarkan intuisi arsitektur | Hanya kalau OOS terbukti naik |
| **Fitur baru** | Tambah ke model yang ada | Test marginal IC dulu |
| **Keputusan drop fitur** | Berdasarkan feature importance LGBM | Berdasarkan marginal IC |

---

### Kalau Simons Mulai dari Nol di Project Ini

**Fase 1 — Signal Discovery (sebelum coding apapun)**

```
Hipotesis A: "RSI H1 memprediksi arah 8 jam ke depan"
  → IC(rsi_6, label_8h) = -0.15, t = -12.4  ✓ ADA SINYAL → lanjut

Hipotesis B: "OFI order flow memprediksi arah 8 jam ke depan"
  → IC(ofi_raw, label_8h) = -0.025, t = -2.0  ✗ SINYAL LEMAH → STOP

Hipotesis C: "ETF outflow 7 hari memprediksi arah BTC 7 hari ke depan"
  → Ukur IC(etf_flow_7d, btc_fwd_7d) dulu
  → Kalau IC < 0.02 → tidak perlu GRU apapun
  → Kalau IC ≥ 0.05 → lanjut ke fase berikutnya
```

Simons tidak akan pernah membangun GRU sebelum langkah ini.

**Fase 2 — Signal Validation (IC Decay)**

```
Bagi data menjadi 6 window temporal, hitung IC rsi_6 di setiap window:
  [-0.14, -0.17, -0.13, -0.16, -0.15, -0.14]  → konsisten ✓

Hitung IC ofi_raw di setiap window:
  [-0.03, +0.01, -0.02, +0.02, -0.01, +0.03]  → tidak konsisten ✗ → DROP
```

**Fase 3 — Model Sederhana Dulu**

```
Logistic regression dengan 5 fitur KEEP → F1 = 0.38 (baseline linear)
Tambah LSTM                              → F1 = 0.41 (+0.03) ✓ worth it
Tambah GRU regime                        → F1 = 0.41 (+0.00) ✗ tidak worth
```

---

## Label Validation — Langkah Paling Kritis yang Sering Dilewati

Sebelum IC test fitur, Simons tanya hal yang lebih mendasar:

> **"Apakah label yang kita latih benar-benar mengukur apa yang menentukan profit?"**

Label salah = model sempurna yang memprediksi hal yang salah. Lebih berbahaya dari fitur buruk.

### Tiga Pertanyaan Label Simons

**① Apakah 8-bar ke depan target yang tepat?**

Di sistem ini ada disconnect fundamental:

```
LSTM dilatih memprediksi:
  "apakah harga naik/turun dalam 8 bar ke depan?"

Yang sebenarnya menentukan profit:
  "berapa P&L trade setelah Guardian exit, TP/SL, dan partial exit?"
```

Contoh kasus nyata:

```
Bar t=0 : label LONG (8-bar forward return = +5%)
Bar t=1 : harga naik +0.5%
Bar t=2 : Guardian → FULL_EXIT karena momentum berbalik
Hasil   : P&L = +0.5%, bukan +5%

Label bilang BENAR. Trade hanya dapat 10% dari yang dijanjikan label.
```

Guardian memotong trade jauh sebelum 8 bar. Label mengukur pasar bebas, bukan hasil trade nyata.

**② Apakah 48% FLAT masalah labeling atau memang perilaku pasar?**

```
Test A: apakah bar berlabel FLAT → P&L trade ≈ 0?
  Kalau ya  → 48% FLAT akurat, memang zona tanpa edge
  Kalau tidak → threshold terlalu konservatif, terlalu banyak
                momen profitable yang salah masuk FLAT

Test B: apakah distribusi FLAT konsisten lintas koin?
  SOL: FLAT 41%   DOGE: FLAT 48%   XAUT: FLAT 67%
  Kalau sangat bervariasi → threshold universal tidak valid
```

Masalah turunan: 48% FLAT membuat LSTM cenderung prediksi FLAT karena "aman."
Inilah akar dari flat_review dimatikan — LSTM terlalu sering prediksi FLAT.

**③ Apakah threshold ±3% optimal?**

Simons tidak pakai intuisi. Test range threshold, pilih yang IC-nya paling **stabil**:

```
threshold 1% → FLAT=24%, IC decay: [0.13, 0.11, 0.12, 0.12, 0.11, 0.13]  ✓ stabil
threshold 2% → FLAT=35%, IC decay: [0.14, 0.15, 0.13, 0.14, 0.14, 0.15]  ✓ terbaik
threshold 3% → FLAT=48%, IC decay: [0.12, 0.07, 0.13, 0.08, 0.11, 0.09]  ✗ fluktuatif
threshold 4% → FLAT=60%, IC decay: [noise]
```

Threshold optimal bukan yang IC-nya tertinggi — tapi yang paling konsisten lintas waktu.

### Kenapa LONG dan FLAT Bisa Menghasilkan P&L Hampir Sama

Ini tanda label tidak berfungsi sebagai pemisah kualitas trade.

```
Label bekerja:     LONG=+2.5%  FLAT=0.0%  SHORT=-1.8%
Label bermasalah:  LONG=+1.2%  FLAT=+1.0%  SHORT=+0.9%  ← hampir sama
```

Tiga penyebab utama:

```
Penyebab 1 — Guardian mengubah segalanya
  Label LONG berdasarkan 8-bar pasar bebas.
  Guardian exit di bar ke-2. Trade dapat sepersepuluh dari yang dijanjikan label.

Penyebab 2 — TP/SL tidak selaras dengan threshold label
  TP aktual ≈ 1.5–2% (1.2× ATR)
  Threshold FLAT = antara -3% dan +3%
  → trade profitable (TP hit di 1.8%) terhitung bar berlabel FLAT
  → LSTM dilatih mengabaikan kondisi yang sebenarnya menguntungkan

Penyebab 3 — Threshold universal untuk 21 koin berbeda volatilitas
  SOL bisa bergerak 3% dalam 2 jam → threshold 3% wajar
  XAUT bergerak 0.5% dalam 8 jam → threshold 3% hampir tidak pernah tercapai
  → label LONG untuk XAUT hampir tidak ada, model tidak punya sinyal
```

### Test Diagnostik — Validasi Label dari Data Holdout

```
Ambil semua trade di holdout backtest
Kelompokkan berdasarkan label bar entry-nya
Ukur actual P&L per kelompok

Harapan kalau label valid:
  LONG entry bars  → WR tinggi, P&L positif
  FLAT entry bars  → WR rendah, P&L ≈ 0

Kalau LONG ≈ FLAT → label tidak membedakan kualitas trade
→ akar masalah F1 ≈ random, bukan masalah arsitektur atau fitur
```

---

## Algoritma ML yang Simons Gunakan

Simons tidak memilih satu algoritma — dia ensemble banyak algoritma lemah yang sinyalnya **benar-benar independen**. Tidak ada algoritma yang dominan di semua kondisi. Yang menang adalah kombinasi.

> *"Setiap parameter tambahan butuh 10x lebih banyak data untuk divalidasi."*

### 1. Linear Model — Selalu Titik Mulai

Ridge Regression, Lasso, Logistic Regression.

IC dari model linear adalah **batas atas teoritis** sinyal yang bisa dipelajari. Kalau model linear tidak bisa menangkap sinyal, model kompleks hanya akan overfit noise — bukan menemukan pola baru.

```
Gate: Logistic Regression dengan fitur KEEP → F1 baseline
  F1 << random → sinyal tidak ada, bukan masalah arsitektur
  F1 > random  → ada yang bisa dipelajari, lanjut ke non-linear
```

### 2. Tree Ensemble — Tangkap Non-Linearitas

LGBM, XGBoost, Random Forest. Sudah ada di sistem saat ini.

Simons tetap pakai ini — tapi hanya setelah IC test memfilter fitur. LGBM bisa belajar dari noise kalau dibiarkan dengan 104 fitur tanpa gate.

### 3. Hidden Markov Model (HMM) — Regime Detection

Gap terbesar di sistem saat ini. HMM dirancang khusus untuk pertanyaan *"pasar sedang dalam kondisi apa?"*

```
HMM belajar secara unsupervised:
  State 1: low volatility, trending
  State 2: high volatility, ranging
  State 3: crisis / breakdown

Bukan kamu yang mendefinisikan regime — data yang menentukannya.
```

Lebih tepat dari GRU untuk regime karena:
- Tidak butuh label (unsupervised)
- Jumlah state dicari secara empiris
- Dirancang matematis untuk transisi antar kondisi
- Interpretable: "sekarang state 2, artinya ranging"

Untuk concern ETF outflow: Simons tidak langsung pakai data ETF sebagai fitur.
Dia tanya dulu — *"apakah ETF outflow terjadi di state HMM tertentu?"*
Kalau iya, HMM sudah menangkap regime itu secara implisit dari data harga.

### 4. Sequence Model (LSTM/GRU) — Hanya Setelah Dibuktikan Perlu

```
Test wajib sebelum pakai LSTM:
  Snapshot → Logistic Regression → F1 = A
  Sequential → LSTM             → F1 = B

  B > A secara signifikan → ada pola temporal, LSTM worth it
  B ≈ A                  → LSTM hanya overfit, cukup snapshot
```

Untuk GRU regime — Simons test HMM dulu. HMM lebih parsimonious untuk task yang sama dengan data yang jauh lebih sedikit.

### 5. Survival Analysis — Untuk Exit Timing (Guardian)

Guardian sekarang adalah multiclass LGBM: HOLD / PARTIAL / FULL EXIT.
Tapi pertanyaan Guardian sebenarnya adalah *"kapan waktu terbaik untuk keluar?"* — ini **time-to-event problem**, bukan klasifikasi.

```
Cox Hazard Model:
  Input : kondisi trade saat ini (P&L, bars held, ATR, dll.)
  Output: probabilitas exit optimal dalam N bar ke depan

  Bar 3 : hazard rate = 0.12  → tahan
  Bar 7 : hazard rate = 0.45  → pertimbangkan exit
  Bar 11: hazard rate = 0.78  → exit sekarang
```

Lebih natural dari multiclass karena durasi holding adalah variabel bermakna,
bukan keputusan di titik fixed.

### 6. PCA / ICA — Reduksi 104 Fitur LGBM

Alih-alih IC test satu per satu untuk 104 fitur, dekomposisi dulu:

```
PCA pada 104 fitur:
  PC1 = 35% varians → "momentum factor"
  PC2 = 18% varians → "volatility factor"
  PC3 = 11% varians → "structure factor"
  ...
  PC20+ = < 0.5% masing-masing → noise

Hasil: 104 fitur → 15–20 principal components yang guaranteed orthogonal
```

Cara otomatis mendapatkan fitur independen tanpa manual IC test satu per satu.

### 7. Meta-Labeling — Pisahkan Dua Keputusan

Saat ini LGBM memutuskan dua hal sekaligus: arah trade DAN kualitas sinyal.
Simons pisahkan keduanya:

```
Model Primer : prediksi arah → LONG atau SHORT (binary, tidak ada FLAT)
Model Sekunder: prediksi apakah model primer benar → Ya / Tidak

Entry hanya kalau:
  Model primer → LONG
  DAN model sekunder → "ya, kemungkinan benar"
```

Confidence threshold yang sudah ada (0.65) adalah versi sederhana dari ini —
tapi meta-labeling melatihnya secara eksplisit sebagai model terpisah.

### Peta Algoritma untuk Sistem Ini

| Komponen | Sekarang | Simons Rekomendasikan |
|----------|----------|----------------------|
| Entry signal | LGBM 104 fitur | LGBM + PCA preprocessing + IC filter |
| Momentum confirm | LSTM 12 fitur | LSTM 5 fitur (post IC test) |
| Regime detection | Tidak ada | **HMM dulu**, GRU hanya kalau HMM tidak cukup |
| Exit timing | Guardian LGBM multiclass | **Survival Analysis** + LGBM hybrid |
| Label design | Fixed N-bar return | Triple Barrier + ATR-normalized |
| Ensemble fusion | Fixed weights (0.65/0.35) | Weights proporsional marginal IC |

### Urutan Implementasi (effort rendah → tinggi)

```
1. PCA pada 104 fitur LGBM        → reduksi redundansi, tidak perlu retrain
2. HMM untuk regime detection     → ganti rencana GRU, lebih parsimonious
3. Triple Barrier untuk label     → selaraskan label dengan sistem trading nyata
4. Survival Analysis untuk Guardian → ganti multiclass LGBM exit timing
```

---

## Ensemble Model — Cara Simons Merakit Semua Komponen

### Apa itu Ensemble?

Kombinasi beberapa model di mana hasilnya lebih baik dari model manapun secara individual.

```
Model A benar 70% — salah di kondisi volatilitas tinggi
Model B benar 65% — salah di kondisi ranging market
Model C benar 68% — salah di kondisi news spike

Saat A salah → B dan C kemungkinan benar
Saat B salah → A dan C kemungkinan benar
→ Ensemble benar lebih sering dari 70%
```

Kuncinya bukan seberapa akurat setiap model — tapi **seberapa berbeda kesalahan yang mereka buat**.
Model yang selalu salah bersamaan tidak membantu satu sama lain.

---

### Bagaimana Simons Merakit Ensemble

**Langkah 1 — Setiap model harus lolos gate IC dulu**

```
Marginal IC model baru terhadap ensemble yang sudah ada ≥ 0.01
  → sinyal ini unik, belum ditangkap komponen lain → masuk

Marginal IC ≈ 0
  → menduplikasi yang sudah ada → tidak masuk,
    tidak peduli seberapa akurat standalone-nya
```

**Langkah 2 — Bobot ditentukan marginal IC, bukan intuisi**

```
Bobot LGBM = f(marginal IC LGBM)
Bobot LSTM = f(marginal IC LSTM | LGBM sudah ada)
Bobot HMM  = f(marginal IC HMM  | LGBM + LSTM sudah ada)

Semakin besar kontribusi unik → semakin besar bobot
```

Bukan 0.65/0.35 karena "kelihatannya masuk akal" — tapi karena data berkata demikian.

**Langkah 3 — Bobot berubah per regime**

```
HMM tentukan: "market sedang di state mana?"

State trending : w_lgbm=0.60  w_lstm=0.35  w_hmm=0.05
State ranging  : w_lgbm=0.70  w_lstm=0.15  w_hmm=0.15
State crisis   : w_lgbm=0.30  w_lstm=0.10  w_hmm=0.60
```

---

### Tiga Jenis Ensemble Simons

**1. Signal-Level Ensemble** — ratusan sinyal kecil independen

```
Sinyal RSI reversal      bobot = 0.003
Sinyal volume spike      bobot = 0.002
Sinyal OI divergence     bobot = 0.004
... (ratusan sinyal lagi)
```

Tidak ada single point of failure. Satu sinyal gagal, ratusan lainnya jalan.

**2. Model-Level Ensemble** — algoritma berbeda saling melengkapi

```
Linear model   → pola linear yang stabil
Tree model     → non-linearitas dan interaksi fitur
HMM            → regime dan state pasar
Sequence model → pola temporal
```

Setiap algoritma punya blind spot berbeda. Kombinasinya saling menutupi.

**3. Regime-Conditional Ensemble** — bobot dinamis per kondisi pasar

HMM menentukan state → bobot ensemble disesuaikan otomatis per state.

---

### Model Final Ensemble untuk Sistem Ini

Kalau Simons membangun ulang dari awal:

```
┌──────────────────────────────────────────────────────┐
│                  ENSEMBLE FINAL                       │
│                                                       │
│  Layer 1 — Linear Baseline                            │
│  Logistic Regression (5 fitur KEEP)                   │
│  "Batas bawah" — sinyal paling stabil dan reliable    │
│  Gate: kalau layer ini tidak bisa → sinyal tidak ada  │
│                                                       │
│  Layer 2 — Non-Linear Entry                           │
│  LGBM (PCA-reduced, IC-filtered)                      │
│  Tangkap interaksi fitur yang linear tidak bisa       │
│  Bobot: marginal IC vs Layer 1                        │
│                                                       │
│  Layer 3 — Temporal Momentum                          │
│  LSTM (5 fitur KEEP, bukan 12)                        │
│  Tangkap pola sequential H1                           │
│  Bobot: marginal IC vs Layer 1+2                      │
│                                                       │
│  Layer 4 — Regime Context                             │
│  HMM (unsupervised, belajar state dari data)          │
│  Atur bobot Layer 1-3 secara dinamis                  │
│  Gate tambahan: state crisis → tahan entry            │
│                                                       │
│  Layer 5 — Exit Timing (Guardian)                     │
│  Survival Analysis (Cox Hazard) + LGBM hybrid         │
│  Time-to-event problem, bukan klasifikasi             │
│  Pipeline terpisah dari entry, metric berbeda         │
└──────────────────────────────────────────────────────┘
```

**Kenapa urutan ini?**

```
Layer 1 wajib ada dulu:
  F1 > random → ada sinyal → lanjut ke Layer 2
  F1 ≈ random → sinyal tidak ada → stop, jangan lanjut

Layer 2 diuji marginal IC vs Layer 1:
  Tidak tambah apapun → tidak perlu

Layer 3 diuji marginal IC vs Layer 1+2:
  Snapshot sudah cukup → LSTM tidak perlu

Layer 4 bukan diuji IC biasa:
  HMM adalah meta-component yang mengatur bobot layer lain
  Dievaluasi dari: apakah IC ensemble lebih stabil lintas regime?

Layer 5 independen dari entry:
  Dievaluasi dengan metric berbeda — holding time, P&L distribution
```

---

### Perbandingan Ensemble Saat Ini vs Simons

| Aspek | Sekarang | Simons |
|-------|----------|--------|
| Jumlah komponen | 2 (LGBM + LSTM) | 5 layer terstruktur |
| Bobot ditentukan | Intuisi (0.65/0.35) | Marginal IC |
| Bobot berubah? | Tidak | Ya — per regime via HMM |
| Gate masuk ensemble | Tidak ada | Marginal IC ≥ 0.01 |
| Linear baseline | Tidak ada | Selalu ada sebagai anchor |
| Regime awareness | Partial (trend alignment) | Eksplisit via HMM state |
| Exit model | LGBM multiclass | Survival Analysis + LGBM |

---

## Prinsip Simons

> *"If you're adding a feature because it makes intuitive sense, you're not doing science."*

> *"The signal has to exist in the data before you build the model. Otherwise you're fitting noise with a complex function."*

> *"Each new signal must improve the ensemble. Standalone performance is irrelevant."*

> *"The label is your most important decision. A perfect model trained on the wrong label is perfectly wrong."*

> *"We're right 50.75% of the time, but we're right 50.75% of the time ALL THE TIME."*

---

## Adopsi Simons sebagai Retail Quant Trader

### Keunggulan Struktural yang RenTech Tidak Punya

Sebelum bicara adopsi, pahami posisi retail quant:

| Aspek | RenTech | Retail Quant |
|-------|---------|--------------|
| Kapasitas | Masalah besar — strategi mati kalau terlalu besar | **Tidak ada masalah kapasitas** |
| Tekanan investor | LP redemption, quarterly reporting | **Tidak ada** |
| Pasar kecil | Tidak bisa — akan gerakkan harga sendiri | **Bisa masuk yang RenTech tidak bisa** |
| Iterasi | Butuh committee approval | **Deploy besok kalau mau** |
| Insentif | Fee-based | **100% uang sendiri — alignment sempurna** |
| Tim | 300+ PhD | Solo — tapi AI bisa jadi peer reviewer |

Kecil itu **keunggulan struktural**, bukan kelemahan.

---

### Adaptasi untuk Retail

**Kualitas di atas kuantitas sinyal**

```
RenTech : 500 sinyal × edge 0.1% masing-masing
Retail  : 10–15 sinyal × edge 1–2% masing-masing

Total expected value bisa setara.
Lebih mudah dikelola, divalidasi, dan diperbaiki.
```

**Manfaatkan kecepatan iterasi**

```
RenTech butuh bulan untuk deploy perubahan.
Retail bisa:
  Senin : temukan sinyal baru
  Selasa: IC test
  Rabu  : backtest purged CV
  Kamis : holdout validation
  Jumat : deploy (kalau lolos semua gate)
```

Kecepatan tanpa proses = gambling. Kecepatan dengan proses = keunggulan.

**Gunakan AI sebagai "tim"**

```
AI untuk review kode       → second opinion sebelum deploy
AI untuk brainstorm sinyal → hipotesis yang belum terpikirkan
AI untuk debug logic       → cegah bias konfirmasi
AI untuk dokumentasi       → semua keputusan tercatat dengan alasan
```

Ini cara retail quant mengimplementasikan "culture of peer review" ala RenTech.

**Position sizing lebih kritis dari entry signal**

```
Kelly Criterion (simplified):
  f* = edge / variance

  edge     = win_rate × avg_win - loss_rate × avg_loss
  variance = standar deviasi P&L

  Bet tidak pernah melebihi f* dari portfolio

Drawdown rule:
  Portfolio turun 15% dari peak → reduce position size 50%
  Portfolio turun 25% dari peak → stop trading, review sistem
  Tidak pernah average down pada sistem yang sedang drawdown
```

---

### Manifesto Retail Quant — 10 Prinsip Simons

```
① Tidak ada sinyal tanpa IC test. Tidak ada pengecualian.

② Tidak ada model baru tanpa marginal IC test terhadap ensemble.

③ Tidak ada training tanpa validasi label terlebih dahulu.

④ Model adalah otoritas tertinggi.
   Override hanya untuk anomali teknis — bukan opini pasar.

⑤ Position size ditentukan formula, bukan perasaan.

⑥ Drawdown limit ditentukan sebelum trading, ditaati saat trading.

⑦ Setiap keputusan riset didokumentasikan dengan justifikasi data.
   (EXPERIMENTS.md adalah implementasi dari prinsip ini)

⑧ Kompound dulu, withdraw belakangan.

⑨ Hanya deploy apa yang bisa dijalankan 5 tahun ke depan.

⑩ Review IC decay setiap kuartal — pensiun sinyal yang melemah.
```

---

### Satu Hal yang Membedakan Retail Quant Sukses

Bukan kecerdasan algoritma. Bukan akses data eksklusif.

> **Disiplin untuk tidak melakukan apa yang tidak didukung data.**

Sebagian besar retail trader gagal bukan karena sistemnya buruk — tapi karena override sistem saat drawdown, menambah fitur karena panik, atau mengubah parameter setelah satu minggu yang buruk.

Simons berhasil karena seluruh organisasinya dibangun untuk mencegah hal itu. Sebagai retail quant, kamu harus membangun hal yang sama — di dalam dirimu sendiri.

---

## Roadmap IC Test per Model

| Model | Status | Fitur | Next Step |
|-------|--------|-------|-----------|
| LSTM | ✅ Selesai | 12 → 5 KEEP | IC Decay test lintas 6 window |
| LGBM | 🔲 Belum | 104 fitur | `03b_ic_test.py --model lgbm` |
| Guardian v3 | 🔲 Belum | 104+7 fitur | `03b_ic_test.py --model guardian` |
| GRU (rencana) | 🔲 Tunggu IC signal | 6 fitur D1 | Ukur IC(etf_flow/OI, fwd_return) dulu |

---

## Integrasi ke Pipeline

```
01_fetch → 02_clean → 03_engineer → 03b_ic_test  ← gate wajib
                                         │
                          ┌──────────────┼──────────────┐
                          ▼              ▼              ▼
                   04_train_lgbm   05a→05b→05c   06_train_guardian
```

```bash
python pipeline/03b_ic_test.py --model lstm
python pipeline/03b_ic_test.py --model lgbm
python pipeline/03b_ic_test.py --model all
```

Output: `reports/experiments/ic_test_{model}_{run_id}.md` dan `.json`

---

*Script: `pipeline/03b_ic_test.py`*
*Hasil LSTM: `reports/experiments/ic_test_lstm_ic_test_v1.json`*
