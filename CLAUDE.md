# CLAUDE.md — Riset Pemodelan Crypto Trading

## Project Overview

Sistem trading kripto berbasis ML untuk Binance Futures.
- **Versi aktif**: lihat `models/model_registry.json` untuk production version & metrik terkini
- **Research state**: lihat `EXPERIMENTS.md` untuk eksperimen yang sedang/akan berjalan
- **Model aktif & scorecard**: lihat `reports/widyawardhana_model.md`
- **Data**: 2020-01-01 – 2026-04-01, 21 koin, H1 base + H4 swing/regime
- **TRAIN_CUTOFF_DATE = 2026-04-01** — training sampai Mar 2026; holdout Apr – Jun 2026

## Critical Constraints

- **Python 3.12**, Windows, AMD RX 6600 — DirectML untuk LSTM, OpenCL untuk LGBM
- **Shell**: PowerShell. Gunakan `;` bukan `&&` untuk chaining
- **Encoding**: Terminal cp1252 — jangan pakai unicode arrow (`→`) di logger messages
- **KLINE_LIMIT = 1000** — Binance max per request (bukan 1500)
- **TRAIN_CUTOFF_DATE** — tidak boleh ada data post-cutoff bocor ke training, di mana pun
- **LSTM**: Custom `ManualLSTMCell` — train GPU (DirectML), infer CPU

## 📋 ALUR KERJA EKSPERIMEN — WAJIB DIIKUTI

Setiap eksperimen mengikuti 4 tahap ini tanpa pengecualian. Jangan loncat ke tahap berikutnya sebelum tahap sebelumnya selesai.

---

### TAHAP 1 — Cek Logbook (SEBELUM apapun)

Baca `EXPERIMENTS.md` dari bawah ke atas (entri terbaru dulu):
- Apakah eksperimen serupa pernah dicoba? Kenapa tidak dipakai?
- Berapa baseline metrik model aktif saat ini?
- Apa yang berbeda kali ini yang membuat hasilnya mungkin berbeda?

Jika tidak bisa menjawab pertanyaan ketiga → baca logbook lagi, jangan lanjut.

---

### TAHAP 2 — Tulis Rencana di EXPERIMENTS.md (SEBELUM jalankan script)

Tambahkan entri baru di `EXPERIMENTS.md` dengan format:

```markdown
## YYYY-MM-DD — [Nama Eksperimen]

**Status**: PLANNED

### Hipotesis
[Apa yang diduga akan terjadi dan mengapa]

### Yang Diubah
- [Parameter / arsitektur / fitur yang berbeda dari sebelumnya]

### Target
- Metrik yang ingin dicapai: [WR > X%, PF > Y, trades > Z]
- Baseline yang akan dibandingkan: Widyawardhana vN (lihat widyawardhana_model.md)

### Script
- [script yang akan dijalankan]
```

**Jangan jalankan script sebelum rencana ini ditulis.** Ini bukan formalitas — ini memaksa berpikir sebelum komputasi mahal dimulai.

---

### TAHAP 3 — Jalankan & Catat Hasil di EXPERIMENTS.md

Setelah selesai, update entri yang sama:

```markdown
**Status**: COMPLETED  ← atau ABANDONED (dengan alasan)

### Hasil CV
| Metrik | Nilai |
|--------|-------|
| Mean F1 Macro | X.XXX |

### Hasil Holdout (jika dievaluasi)
| Metrik | Nilai | vs Widyawardhana |
|--------|-------|-----------------|
| Trades | N | +/- delta |
| WR | X% | +/- delta |
| PF | X.XX | +/- delta |
| PnL | $X | +/- delta |

### Kesimpulan
[Apakah hipotesis terbukti? Kenapa berhasil / tidak berhasil?]
```

---

### TAHAP 4 — Update Model Aktif (JIKA lebih baik)

Bandingkan hasil dengan model aktif di `reports/widyawardhana_model.md`.
**Kriteria upgrade** (harus penuhi SEMUA):
- WR holdout >= WR Widyawardhana saat ini pada periode yang sama
- PF holdout >= PF Widyawardhana saat ini
- Trades >= 80% jumlah Widyawardhana (jangan trade terlalu sedikit)
- Metodologi genuine OOF (tidak ada kontaminasi holdout)

Jika kriteria terpenuhi:
1. Arsip `widyawardhana_model.md` ke `reports/experiments/YYYY-MM-DD_widyawardhana_vN.md`
2. Update `widyawardhana_model.md` dengan model baru (versi, arsitektur, scorecard)
3. Catat di `EXPERIMENTS.md` bahwa model aktif telah diganti

Jika tidak terpenuhi: **jangan update** — catat alasan di EXPERIMENTS.md dan lanjut riset.

---

## ⛔ ATURAN METODOLOGI — MUTLAK TIDAK BOLEH DILANGGAR

Aturan berikut bukan preferensi — melanggarnya membuat seluruh hasil riset tidak bisa dipercaya.

**Referensi lengkap + contoh kode: `METHODOLOGY.md`**

### ATURAN 1 — Holdout adalah amplop tersegel. Dibuka sekali, di akhir.

Setiap keputusan (threshold, parameter, pilihan model) HARUS dibuat berdasarkan data training saja.
Holdout dipakai **sekali** setelah semua config di-freeze. Jika holdout sudah dipakai → terkontaminasi.

**Yang menggunakan OOF (bukan holdout):** threshold sweep, Guardian variant, sizing params, feature selection.

### ATURAN 2 — Guardian dilatih pada OOF trades, bukan in-sample trades.

Lihat contoh kode di `METHODOLOGY.md § Aturan 2`.

### ATURAN 3 — Scaler di-fit di dalam loop fold, bukan sebelumnya.

Lihat contoh kode di `METHODOLOGY.md § Aturan 3`.

### ATURAN 4 — Semua extended backtest HARUS pakai Purged CV OOF.

- Retrain LGBM per fold dari nol — model TIDAK BOLEH lihat data uji
- Fixed model di 2020-2025 = **IN-SAMPLE LEAKAGE** (hasil tidak valid)
- **Purge gap = MAX_HOLDING_BARS** (36 bar) antara akhir train fold dan awal val fold

### ATURAN 5 — Tidak ada fitur yang menggunakan data masa depan.

- Semua rolling window harus backward-looking
- `shift(-N)` pada fitur = leakage — dilarang tanpa lag kompensasi
- Fitur IC > 0.10 tanpa justifikasi domain → investigasi leakage dulu

---

## Larangan (Do NOT)

- **Jangan duplikasi isi `config.py`** — baca langsung dari file; `config.py` adalah source of truth
- **Jangan tulis riwayat perubahan di sini** — gunakan `EXPERIMENTS.md`
- **Jangan re-implement TP/SL regressor/classifier** — file sudah dihapus; diskusi dulu sebelum membuat ulang
- **Jangan pakai fixed model untuk extended backtest** — IN-SAMPLE LEAKAGE. Harus purged CV OOF retrain per fold.
- **Jangan modifikasi file di `swint_tradev2` secara manual** — deployment via `tools/deploy_production.py`
- **Metrik lama TIDAK VALID** — WR 88.93%, PnL $169k dicabut karena data leakage (2026-06-04). Detail: `EXPERIMENTS.md § 2026-06-04`
- **Jangan tune parameter berdasarkan holdout** — holdout bukan development set. Lihat Aturan 1.

## Penamaan Model — Standar Baku

Format: `{universe}.{pipeline}.{component}.{specs}`

```
ic32.rv2.lgbm.36f.sw4.r8       # LGBM: 36 feat, swing H4 label, 8-fold rolling
ic32.rv2.lstm.11f.s72.c55      # LSTM v2: 11 feat, seq=72, cand_thr≥0.55
ic32.rv2.lstm.14f.s36.c55      # LSTM v3: 14 feat, seq=36, cand_thr≥0.55
ic32.rv2.guard.oof              # Guardian: dilatih di OOF trades
```

**Field guide:**

| Field | Format | Arti |
|---|---|---|
| `ic32` | universe | 21-coin H1 config |
| `rv{N}` | pipeline | regime_v{N} — HMM regime features |
| `lgbm` / `lstm` / `guard` | component | komponen model |
| `{N}f` | fitur | jumlah fitur input |
| `sw4` / `sw1` | label | swing H4 atau H1 label |
| `tb` | label | triple barrier (jika dipakai) |
| `r{K}` | fold (lgbm) | K-fold rolling walk-forward |
| `s{S}` | seq len (lstm) | sequence length |
| `c{T}` | cand thr (lstm) | LGBM confident bar thr × 100 (c55 = 0.55) |
| `oof` | domain (guard) | dilatih hanya pada OOF trades |

Run directory: `models/runs/{nama_model_tanpa_titik}/` — ganti titik dengan underscore.
Contoh: `ic32.rv2.lstm.14f.s36.c55` → `models/runs/ic32_rv2_lstm_14f_s36_c55/`

## Retraining Protocol

Sebelum training model baru, tanyakan nama versi secara eksplisit menggunakan format baku di atas. Setelah ditentukan, catat di sini: tanggal training, fitur yang dipakai, periode holdout OOS, dan path model (`models/runs/{run_id}/`).

## Cross-Repo: Production (swint_tradev2)

Repo produksi di `D:\Apps-Dev\swint_tradev2`. Alur kerja **satu arah** dari repo ini:
- Analisis live: tarik DB live dari VPS via `tools/live_db_bridge.py`, lalu `tools/trade_analyzer.py`
- Deployment ke VPS: `python tools/deploy_production.py` (satu perintah)
- Salin lokal saja (tanpa VPS): `python tools/deploy_model.py`

### Deploy Production

```powershell
cd D:\Apps-Dev\Riset_pemodelan
python tools/deploy_production.py
```

| Flag | Kapan dipakai |
|------|----------------|
| `--code-only` | Hanya ubah kode di swint — git push + VPS restart |
| `--models-only` | Hanya update model/config — scp + restart |
| `--local-only` | Stop setelah salin ke swint lokal |
| `--dry-run` | Preview langkah tanpa eksekusi |
| `-m "pesan"` | Custom git commit message |

**Prasyarat:** SSH key-based ke VPS (`root@139.180.157.176`).
**Kenapa dua jalur?** File `.pkl`/`.pt` dan `inference_config.json` di-`.gitignore` swint — dikirim via scp.
Detail kontrak deploy: `MODEL_DEPLOYMENT_BRIDGE.md`

### ⚠️ Data live ada di VPS, BUKAN di file lokal

Web App jalan di **VPS** (`139.180.157.176`). File lokal **BASI** — jangan dipakai untuk analisis live.

**Jembatan DB live** (`tools/live_db_bridge.py`):
- `python tools/live_db_bridge.py` — scp `app.db` dari VPS → cache `data/live_cache/app.db`
- `trade_analyzer.py` otomatis prioritaskan CSV cache live
- API: `from tools.live_db_bridge import pull_live_db, load_trades, load_signals`

## Key Files

| File | Role |
|------|------|
| `config.py` | **Source of truth** — semua parameter terpusat |
| `EXPERIMENTS.md` | Logbook perubahan & temuan — baca sebelum mengubah parameter |
| `METHODOLOGY.md` | Aturan metodologi lengkap + contoh kode |
| `reports/widyawardhana_model.md` | **Model meta aktif** — arsitektur, fitur, config, scorecard |
| `core/evaluator.py` | `simulate_trades_swing()` + Guardian per-bar check + partial exit |
| `core/models.py` | `TradingLSTM`, `ManualLSTMCell` |
| `core/features.py` | Feature engineering + swing labeling v3 |
| `pipeline/07_holdout_backtest.py` | Genuine OOS holdout backtest |
| `tools/deploy_production.py` | **Deploy seamless** Riset → swint lokal → git + scp → VPS restart |
| `tools/deploy_model.py` | Salin ke swint lokal saja (merge inference_config) |
| `tools/live_db_bridge.py` | Tarik DB live (signal/trade) dari VPS via scp → DataFrame + CSV |
| `models/model_registry.json` | Model aktif & metrik baseline |

## Pipeline Sequence

Pipeline dirapikan 2026-06-20. Script lama diarsipkan ke `pipeline/archive/` (180+ file).
Core pipeline: 13 script. LSTM di-restore untuk full retrain ic32_regime_v2+.

```
01_fetch.py                     → Fetch data training (2020 → TRAIN_CUTOFF_DATE)
01c_fetch_positioning.py        → Positioning data hourly cron (Binance + Bybit)
02_clean.py                     → Cleaning + normalization
03_engineer.py                  → Feature engineering → data/training/labeled/
03e_regime_hmm.py               → HMM regime labels (OOF walk-forward)
03e_regime_hmm_holdout.py       → HMM regime labels untuk holdout period

04_train_lgbm_ic32_regime_v2.py     → [GENUINE] LGBM rolling walk-forward CV + OOF + threshold sweep
05_train_lstm_ic32_regime_v2.py     → [GENUINE] LSTM momentum filter training
06_train_guardian_ic32_regime_v2.py → [GENUINE] Guardian training pada OOF trades
07_holdout_ic32_regime_v2.py        → [SEKALI] Holdout eval — set HOLDOUT_EVALUATED=True setelah run
```

Urutan eksekusi:
```powershell
python pipeline/01_fetch.py --all
python pipeline/02_clean.py
python pipeline/03_engineer.py
python pipeline/03e_regime_hmm.py
python pipeline/04_train_lgbm_ic32_regime_v2.py
python pipeline/05_train_lstm_ic32_regime_v2.py
python pipeline/06_train_guardian_ic32_regime_v2.py
python pipeline/07_holdout_ic32_regime_v2.py  # SEKALI, freeze dulu
```

## Slash Commands

- `/trade-analysis` — analisis performa trading live. Detail: `.claude/commands/trade-analysis.md`

## Eksperimen Baru — Wajib Benchmark

Setiap train model baru wajib benchmark terhadap model terbaik yang ada (`ic32_regime_v1`) atau model yang ditentukan user. Gunakan data holdout yang sama. Tampilkan scorecard + leak audit + identifikasi kejanggalan. Simpan hasil di `models/runs/{run_name}/`.
