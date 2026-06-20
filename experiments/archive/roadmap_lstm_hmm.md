# Roadmap LSTM & HMM — Menuju Ensemble Simons

*2026-06-06 | Disusun setelah cascade sweep, Guardian retrain, LSTM momentum v2, HMM Controller*

---

## Pelajaran dari Eksperimen

### LSTM: Bukan Model Jelek — Salah Peran

| Eksperimen | Hasil | Pelajaran |
|------------|-------|-----------|
| LSTM swing labels | F1=0.334 ≈ random | Prediksi hal yang sama dengan LGBM → marginal IC=0 |
| LSTM momentum v2 (flow labels) | **F1=0.415** (+23% vs random) | Model genuine belajar! Flow-based labels work |
| LSTM as hard gate | **-$1,131 PnL** vs LSTM=OFF | Gate bunuh trade bagus saat LSTM salah (59% disagreement) |
| LSTM as soft modulator | **-$103 PnL** vs LSTM=OFF | Lebih baik dari hard gate, tetap tidak nambah value |
| LSTM support vs outcome | **Spearman=0.005 (p=0.90)** | LSTM tidak bisa prediksi apakah LGBM benar |

**Akar masalah**: LSTM prediksi ARAH flow (BEARISH/NEUTRAL/BULLISH) — pertanyaan yang SAMA dengan LGBM. Seharusnya LSTM prediksi OUTCOME trade. Simons: "Model Primer prediksi arah, Model Sekunder prediksi apakah model primer benar."

### HMM: Bukan Jelek — Salah Integrasi

| Eksperimen | Hasil | Pelajaran |
|------------|-------|-----------|
| HMM sebagai fitur argmax | Importance paling rendah (582) | 1 integer tidak cukup — buang 75% informasi |
| HMM Controller (H4) | Kalah dari h4_trend | H4 terlalu lambat untuk per-bar entry |
| HMM Controller (soft) | WR +1.2pp, PnL -$466 | Tidak cukup baik untuk gantikan h4_trend |
| HMM train vs holdout distribution | BERBEDA SIGNIFIKAN | Walk-forward OOF valid — tidak ada leakage |

**Akar masalah**: Argmax membuang informasi distribusi. Renaissance pakai FULL probability distribution + transition matrix untuk risk management dan position sizing.

---

## Roadmap

### Phase 1 — HMM Probabilities (Effort: ~1 jam)

**Goal**: 4 kolom probabilitas HMM menggantikan 1 kolom argmax.

**Implementasi**:
- Generate `hmm_prob_0` s/d `hmm_prob_3` dari GaussianHMM posterior
- Simpan ke `{coin}_hmm_probs.parquet`
- IC test: apakah 4 prob punya marginal IC > 0 vs `hmm_regime_enc`?
- Jika ya → tambahkan ke LGBM feature cols
- Target: feature importance naik dari 582 → 2000+

**Ukuran sukses**: PnL holdout naik vs baseline $2,120

---

### Phase 2 — LSTM Binary Meta-Labeling — ❌ CLOSED (2026-06-15)

> **Update**: Jalur meta entry gate ditutup setelah `tb_meta_fb_v2` gagal Simon Gate #1 di holdout
> (marginal IC t=0.9, corr meta-conf 0.53) dan eksplorasi 3 varian fitur + soft multiplier
> tidak beat baseline PnL. Konsisten dengan kegagalan `tb_lstm_binary_meta_v1` dan `tb_meta_v1`.
> Lihat `EXPERIMENTS.md` §11. **Jangan lanjut hard gate / multiplier meta di atas stack v2.**

**Goal** (asli): LSTM sebagai quality scorer — prediksi apakah trade akan profit.

**Perubahan fundamental**:

```
SEKARANG                          SEHARUSNYA
─────────                         ──────────
Label: BEARISH/NEUTRAL/BULLISH    Label: LOSE(0) / WIN(1)
Model: 3-class classifier         Model: binary classifier
Output: arah flow                 Output: probabilitas profit
Integrasi: direction gate         Integrasi: confidence multiplier
Peran: duplikasi LGBM             Peran: quality scorer INDEPENDEN
```

**Implementasi**:

1. **Generate meta-labels dari trade outcomes**:
   ```
   Training: 2020-2025, walk-forward OOF (hindari in-sample bias)
   Label: 1 jika trade net_pnl > 0, 0 jika tidak
   Fitur: 11 flow features + 7 additional context features
          (hmm_probs, regime上下文, cross-asset context)
   ```

2. **Arsitektur LSTM baru**:
   ```
   Input: 16-32 bar sequence × 18 features
   Model: LSTM + Attention (bukan vanilla LSTM)
   Output: 1 sigmoid → prob(profit)
   ```

3. **Integrasi ensemble**:
   ```
   LGBM direction + LSTM profit_score → soft confidence multiplier
   
   LSTM profit_score 0.8 → boost confidence 10%
   LSTM profit_score 0.5 → no change
   LSTM profit_score 0.2 → reduce confidence 8%
   
   TIDAK PERNAH BLOKIR TRADE — hanya modulasi
   ```

4. **Marginal IC test**:
   ```
   IC(LSTM_profit_score | LGBM_already_known) harus > 0
   Kalau tidak → LSTM tidak masuk ensemble (Simons gate)
   ```

**Ukuran sukses**: PnL+LSTM > PnL tanpa LSTM untuk pertama kalinya — **tidak tercapai**.

**Pengalihan fokus** (ganti Phase 2):
- Guardian exit (`continuation_v1`) — bukan meta entry
- LGBM fitur baru via Simon IC gate (positioning, macro)
- `tb_lstm_macro_v1` — LSTM dengan fitur non-OHLCV, bukan binary meta gate

---

### Phase 3 — Cross-Asset + Frequency Features (Effort: ~2-3 hari)

**Cross-Asset**:
- Fix `btc_h1_return` (sekarang broken — zero-filled)
- Tambah: `btc_h4_return`, `eth_h1_return`
- Sector average return (Layer 1 coins)
- BTC dominance change
- Correlation regime (semua koin naik bareng vs divergen)

**Frequency Features**:
- FFT/Wavelet pada close price 64-bar window
- Output: dominant cycle period, cycle strength, cycle phase
- Cocok untuk LSTM yang bisa capture cyclical patterns

**Ukuran sukses**: IC test per fitur — hanya yang lolos yang masuk

---

### Phase 4 — Autoencoder Feature Extraction (Effort: ~2 hari)

**Goal**: Non-linear feature combinations tanpa manual engineering.

**Implementasi**:
```
104 fitur → Encoder → 12-16 bottleneck features → Decoder → 104 fitur
                         ↓
                 Fitur baru untuk LGBM
```

**Kenapa**: LGBM bisa capture pairwise interactions (tree splits), tapi tidak bisa capture 3-way, 4-way interactions. Autoencoder bisa.

**Ukuran sukses**: Marginal IC autoencoder features vs existing LGBM features > 0

---

### Phase 5 — Attention-Based Architecture (Effort: ~3-5 hari)

**Goal**: LSTM dengan attention mechanism yang fokus ke time steps informatif.

**Kenapa**: Vanilla LSTM memperlakukan semua time steps sama penting. Padahal di trading, beberapa momen jauh lebih informatif (volatility spike, regime change, news event).

**Arsitektur**:
```
Input: 48-64 bar sequence × flow + structure features
Layer 1: Bidirectional LSTM (96 hidden)
Layer 2: Multi-Head Attention (4 heads)
Layer 3: Global Average Pooling
Output: Binary — profit(1) / loss(0)
```

**Ukuran sukses**: F1 > 0.45, marginal IC vs LGBM > 0

---

## Prioritas & Urutan Eksekusi

```
MINGGU 1-2 (deploy + kumpulkan data live):
  ├── Deploy config final (LSTM=OFF, Guardian ON)
  └── Phase 1: HMM Probabilities

MINGGU 3-4 (setelah 100+ live trades):
  ├── Analisis live trade outcomes
  └── Phase 2: LSTM Binary Meta-Labeling

BULAN 2-3 (setelah 300+ live trades):
  ├── Phase 3: Cross-Asset + Frequency Features
  └── Phase 4: Autoencoder Features

BULAN 4-6 (setelah 500+ live trades):
  ├── Retrain ensemble dengan data live
  └── Phase 5: Attention Architecture
```

## Prinsip Simons yang Harus Dipegang

1. **Tidak ada model baru tanpa marginal IC test** terhadap ensemble existing
2. **NN bukan untuk gantikan LGBM** — NN untuk pola yang TIDAK BISA ditangkap LGBM
3. **HMM untuk meta-control** — position sizing, regime switching, bukan entry gate
4. **Ensemble dari banyak small edges** — tidak ada single model dominan
5. **Setiap fitur baru harus lolos IC test** sebelum masuk model apapun
