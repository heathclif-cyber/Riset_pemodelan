# CLAUDE.md — Riset Pemodelan Crypto Trading

## Project Overview

Sistem trading kripto berbasis ML untuk Binance Futures.
- **Versi aktif**: lihat `models/model_registry.json` untuk production version & metrik terkini
- **Research state**: lihat `EXPERIMENTS.md` untuk eksperimen yang sedang/akan berjalan
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
| ... | ... |

### Hasil Holdout (jika dievaluasi)
| Metrik | Nilai | vs Widyawardhana |
|--------|-------|-----------------|
| Trades | N | +/- delta |
| WR | X% | +/- delta |
| PF | X.XX | +/- delta |
| PnL | $X | +/- delta |

### Kesimpulan
[Apakah hipotesis terbukti? Kenapa berhasil / tidak berhasil?]
[Apakah akan dilanjutkan?]
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
Setiap pelanggaran menghasilkan angka yang terlihat bagus tapi tidak akan terbukti di live trading.

**Referensi lengkap: `METHODOLOGY.md`**

---

### ATURAN 1 — Holdout adalah amplop tersegel. Dibuka sekali, di akhir.

> **Setiap keputusan (threshold, parameter, pilihan model, sizing config) HARUS dibuat
> berdasarkan data training period saja. Holdout TIDAK BOLEH digunakan untuk keputusan apapun.**

- Holdout dipakai **sekali** setelah semua config di-freeze, untuk konfirmasi
- Jika holdout sudah dipakai untuk satu keputusan → holdout itu sudah terkontaminasi
- Solusi satu-satunya: gunakan periode holdout BARU untuk evaluasi berikutnya
- Melanggar aturan ini: angka WR/PF yang dilaporkan adalah upper bound optimistik, bukan prediksi live

**Yang menggunakan OOF (bukan holdout):**
- Threshold sweep LGBM (thr_long, thr_short)
- Pemilihan Guardian variant
- Dynamic sizing parameter (HIGH_THR, LOW_THR, Kelly params)
- Kalibrasi confidence (Platt/Isotonic) — fit pada OOF (conf, win) pairs
- Feature selection (IC analysis)
- Perbandingan arsitektur (LGBM vs cascade vs ensemble)

---

### ATURAN 2 — Guardian dilatih pada OOF trades, bukan in-sample trades.

```python
# ❌ SALAH — model entry sudah "hafal" data ini
insample_trades = run_trading(lgbm_final, X_train)  # model dilatih di X_train
train_guardian(insample_trades)                      # Guardian belajar dari sinyal biased

# ✅ BENAR — setiap trade diprediksi tanpa model melihat bar itu saat training
oof_signals = apply_threshold(oof_predictions, best_thr)   # dari CV
oof_trades  = run_trading(oof_signals, X_train)             # OOF signals
train_guardian(oof_trades)                                  # Guardian clean
```

---

### ATURAN 3 — Scaler di-fit di dalam loop fold, bukan sebelumnya.

```python
# ❌ SALAH — val fold "melihat" statistiknya sendiri saat scaler di-fit
scaler = StandardScaler()
X_all_scaled = scaler.fit_transform(X_all)   # leakage: val ikut di-fit

# ✅ BENAR — scaler hanya melihat training fold
for train_idx, val_idx in folds:
    scaler = StandardScaler()
    X_tr  = scaler.fit_transform(X[train_idx])
    X_val = scaler.transform(X[val_idx])     # transform only, tidak fit
```

---

### ATURAN 4 — Semua extended backtest HARUS pakai Purged CV OOF.

- Retrain LGBM per fold dari nol — model TIDAK BOLEH lihat data uji
- Testing dengan fixed model (`lgbm_baseline.pkl`) di 2020-2025 = **IN-SAMPLE LEAKAGE**
- Hasil in-sample (PnL +$9,878, WR 69%) TIDAK VALID sebagai ekspektasi OOF
- Hasil genuine OOF: PnL sekitar -$362 s/d -$148 (EXPERIMENTS.md §2026-06-06)
- **Purge gap = MAX_HOLDING_BARS** (36 bar) antara akhir train fold dan awal val fold
- Multi-coin: purge berdasarkan timestamp universal, bukan per-coin

```python
# ❌ SALAH — fixed model, in-sample leakage
lgbm = joblib.load("lgbm_baseline.pkl")  # trained on 2020-2025
test on 2020-2025                         # same data!

# ✅ BENAR — purged CV OOF
for fold in purged_folds:
    fold_model = LGBMClassifier()
    fold_model.fit(train_fold)            # train on folds != k
    test on fold_k                        # model never saw this
```

---

### ATURAN 5 — Tidak ada fitur yang menggunakan data masa depan.

- Semua rolling window harus backward-looking
- `shift(-N)` pada fitur = leakage langsung — dilarang tanpa lag kompensasi
- H4 swing yang dikonfirmasi dengan bar ke depan: wajib `shift(lookback)` sebelum dipakai
- Fitur apapun yang IC-nya > 0.10 tanpa justifikasi domain yang jelas → investigasi leakage dulu

---

## Backtest Methodology (ringkasan)

**Cara validasi yang benar:**
```python
# ✅ BENAR — purged CV OOF
for fold in purged_folds:
    fold_model = LGBMClassifier()
    fold_model.fit(train_fold)            # train on folds != k
    test on fold_k                        # model never saw this
```

## Larangan (Do NOT)

- **Jangan duplikasi isi `config.py`** — baca langsung dari file; `config.py` adalah source of truth
- **Jangan tulis riwayat perubahan di sini** — gunakan `EXPERIMENTS.md`
- **Jangan re-implement TP/SL regressor/classifier** — file sudah dihapus; diskusi dulu sebelum membuat ulang
- **Jangan pakai fixed model untuk extended backtest** — IN-SAMPLE LEAKAGE. Harus purged CV OOF retrain per fold.
- **Jangan modifikasi file di `swint_tradev2` secara manual** — deployment hanya via `tools/deploy_model.py`
- **Metrik lama TIDAK VALID** — WR 88.93%, PnL $169k dicabut karena data leakage (2026-06-04). Detail: `EXPERIMENTS.md § 2026-06-04`
- **Jangan tune parameter berdasarkan holdout** — holdout bukan development set. Lihat Aturan 1 di atas.

## Retraining Protocol

Sebelum training model baru, tanyakan nama versi secara eksplisit (contoh: `cascade_v4.4`). Setelah ditentukan, catat di sini: tanggal training, fitur yang dipakai, periode holdout OOS, dan path model (`models/runs/{run_id}/`).

## Cross-Repo: Production (swint_tradev2)

Repo produksi di `D:\Apps-Dev\swint_tradev2`. Alur kerja **satu arah** dari repo ini:
- Analisis live: tarik DB live dari VPS via `tools/live_db_bridge.py`, lalu `tools/trade_analyzer.py`
- Deployment: `python tools/deploy_model.py` (jangan edit manual di repo produksi)

### ⚠️ Data live ada di VPS, BUKAN di file lokal

Web App produksi jalan di **VPS** (`139.180.157.176`), menyimpan signal & trade di SQLite
`/home/swint/swint_tradev2/instance/app.db`. File lokal di mesin dev **BASI** — jangan dipakai
untuk analisis live:
- `D:\Apps-Dev\swint_tradev2\instance\app.db` — DB dev lama
- `D:\Apps-Dev\swint_tradev2\hasil_livetrading.csv` — snapshot lama, tidak ter-sync dari VPS

**Jembatan DB live** (`tools/live_db_bridge.py`):
- `python tools/live_db_bridge.py` — scp `app.db` dari VPS → cache `data/live_cache/app.db`
  (di-`.gitignore`), cetak ringkasan, export CSV live format `trades_export_csv()` web app.
- `trade_analyzer.py` otomatis memprioritaskan CSV cache live ini di atas CSV swint basi.
- API Python: `from tools.live_db_bridge import pull_live_db, load_trades, load_signals`.
- Skema: tabel `signal` (17 kol) + `trade` (29 kol; `is_live`, `pnl_net`, `exit_reason`,
  `tp_guardian_activated`, …). Bridge join `coin` + `model_meta` + `signal` → DataFrame siap pakai.
- **Prasyarat:** SSH key-based auth ke VPS (`root@139.180.157.176`). scp pakai BatchMode (gagal
  cepat, tak hang). Override via env `SWINT_VPS_HOST/USER/DB/SSH_KEY`.
- DB live `journal_mode=delete` (bukan WAL), write tiap 5 mnt → scp file tunggal aman/konsisten.

File kunci produksi yang sering dibaca:

| File | Purpose |
|------|---------|
| `D:\Apps-Dev\swint_tradev2\CLAUDE.md` | Dokumentasi lengkap sistem produksi |
| `tools/live_db_bridge.py` | **Sumber data live** — tarik app.db dari VPS (CSV lokal BASI) |
| `D:\Apps-Dev\swint_tradev2\models\inference_config.json` | Parameter inference aktif |
| `D:\Apps-Dev\swint_tradev2\app\services\paper_trading.py` | Exit logic + Guardian v3 |

## Key Files

| File | Role |
|------|------|
| `config.py` | **Source of truth** — semua parameter terpusat |
| `EXPERIMENTS.md` | Logbook perubahan & temuan — baca sebelum mengubah parameter |
| `core/evaluator.py` | `simulate_trades_swing()` + Guardian per-bar check + partial exit |
| `core/models.py` | `TradingLSTM`, `ManualLSTMCell` |
| `core/features.py` | Feature engineering + swing labeling v3 |
| `pipeline/07_holdout_backtest.py` | Genuine OOS holdout backtest |
| `tools/deploy_model.py` | Deployment ke swint_tradev2 (merge inference_config — preserve key operasional) |
| `tools/live_db_bridge.py` | Tarik DB live (signal/trade) dari VPS via scp → DataFrame + CSV |
| `models/model_registry.json` | Model aktif & metrik baseline |
| `reports/widyawardhana_model.md` | **Model meta aktif** — arsitektur, fitur, config, scorecard. Diupdate saat model baru ditetapkan |
| `reports/experiments/` | Laporan holdout point-in-time historis |

## Pipeline Sequence

Pipeline telah dirapikan (2026-06-15). Hanya script aktif yang tersisa. Script lama/eksperimen
diarsipkan ke `pipeline/archive/` (107 file) — masih bisa dirujuk jika perlu.

### Pipeline Genuine OOF (aktif)

```
01_fetch.py                     → Fetch data training (2020 → TRAIN_CUTOFF_DATE)
01c_fetch_positioning.py        → Positioning data hourly cron (Binance + Bybit)
02_clean.py                     → Cleaning + normalization
03_engineer.py                  → Feature engineering → data/training/labeled/
03e_regime_hmm.py               → HMM regime labels (OOF walk-forward)
03e_regime_hmm_holdout.py       → HMM regime labels untuk holdout period

04_train_lgbm_genuine_v1.py     → [GENUINE] LGBM CV + OOF predictions + threshold sweep
                                   Output: lgbm.pkl, oof_predictions.parquet, best_thresholds.json
06_train_guardian_genuine_v1.py → [GENUINE] Guardian training pada OOF trades, scaler per fold
                                   Input:  oof_predictions.parquet, best_thresholds.json
                                   Output: guardian.pkl, guardian_scaler.pkl
07_holdout_genuine_v1.py        → [SEKALI] Holdout eval — set HOLDOUT_EVALUATED=True setelah run
                                   Guard built-in: RuntimeError jika HOLDOUT_EVALUATED=True
```

### Mengapa "Genuine OOF"?

- `04_` lama: tidak menyimpan OOF predictions, threshold dipilih dari holdout (leakage Aturan 1)
- `06l_` lama: scaler di-fit pada seluruh data sebelum CV loop (leakage Aturan 3),
               trades dari final model bukan OOF (leakage Aturan 2)
- Script baru memastikan: setiap keputusan (threshold, Guardian params) dari OOF saja

### Urutan Eksekusi

```bash
# 1. Data (jika belum ada)
python pipeline/01_fetch.py --all
python pipeline/02_clean.py
python pipeline/03_engineer.py
python pipeline/03e_regime_hmm.py

# 2. Training genuine OOF
python pipeline/04_train_lgbm_genuine_v1.py
python pipeline/06_train_guardian_genuine_v1.py

# 3. Freeze config, lalu evaluasi holdout SEKALI
python pipeline/07_holdout_genuine_v1.py
```

## LSTM Momentum — OHLCV Information Ceiling

LSTM V3 (16 IC-validated features, 5 koin, 8-fold purged CV):
- **Mean Val F1: 0.407 ± 0.007** — plateau, tidak ada improvement vs V2 (F1=0.415)
- Random baseline: 0.333. Gain: +0.074.
- NEUTRAL class selalu lemah (F1 ~0.22-0.30) — LSTM tidak bisa bedakan netral vs directional dari OHLCV
- BULLISH/BEARISH classes lumayan (F1 ~0.46-0.53)
- **Kesimpulan**: OHLCV telah mencapai ceiling informasi untuk prediksi momentum.
  Solusi: positioning data (OI delta, Taker ratio, Top Trader L/S) — dikumpulkan via 01c_fetch_positioning.py.
  Estimasi 6 bulan dari 2026-06-07 → Desember 2026 cukup data untuk IC38 momentum retrain.

## Active Configuration (2026-06-14) — Widyawardhana v2

```
Research model : tb_widyawardhana_v2
Entry          : flatboost_v2 LGBM (27 feat, Triple Barrier labels)
                 models/runs/tb_lgbm_flatboost_v2/lgbm.pkl
HMM Regime     : Adaptive threshold T50_R55
                 TRENDING (state 0,3): thr_long=0.50, thr_short=0.55
                 RANGING  (state 1,2): thr_long=0.55, thr_short=0.60
LSTM           : Soft veto di TRENDING bars saja (opp_conf >= 0.50)
                 models/lstm_best.pt
Guardian       : profit_v1 (18 static + 7 dynamic = 25 feat)
                 models/runs/tb_guardian_profit_v1/guardian.pkl
                 Labeling: profit-only, HAPUS loss-based EXIT
Data           : Positioning mining aktif — 4 endpoint Binance+Bybit hourly
```

**Fitur LGBM flatboost_v2 (27):**
```
Liquidity/Distance : dist_liq_50x_long, dist_liq_50x_short, dist_liq_20x_short,
                     dist_from_8h_high, dist_swing_high, VAH
CVD/Order Flow     : cvd_slope_h4, ofi_h4_delta, cvd_momentum_adv
Momentum/Trend     : trend_accel_4h, stochrsi_d, log_ret_20, atr_percent_h4, atr_percentile_h1
Volume Analysis    : whale_retail_divergence, Buy_Liq, vol_spike_zscore, range_expansion_h4,
                     ultra_high_vol, absorption_z, vol_accel_3h, vol_ratio_20
Wyckoff            : no_supply, no_demand, effort_vs_result
Macro/Waktu        : dow_cos, funding_rate
```

**Fitur Guardian profit_v1 (18 static + 7 dynamic):**
```
Static  : etf_gbtc_change_usd, etf_total_change_usd, cvd_slope_h4, ofi_h4_delta,
          wyckoff_phase, Sell_Liq, atr_percentile_h1, stochrsi_k, dist_liq_50x_short,
          funding_rate, ema_7_h1, dow_cos, cvd_div_h4, dist_swing_low, VAH,
          cvd_momentum_adv, dist_from_8h_high, ema_200_h1
Dynamic : bars_held_norm, current_pnl_pct, current_pnl_atr,
          max_favorable_pnl_pct, drawdown_from_peak_pct, direction, entry_price_ratio
```

**Parameters:**
| Parameter | Value |
|-----------|-------|
| THR_TRENDING_LONG | 0.50 |
| THR_TRENDING_SHORT | 0.55 |
| THR_RANGING_LONG | 0.55 |
| THR_RANGING_SHORT | 0.60 |
| LSTM_VETO_THRESHOLD | 0.50 |
| LSTM_ACTIVE_IN | TRENDING bars only |
| GUARDIAN_EXIT_THRESHOLD | 0.70 |
| GUARDIAN_MIN_HOLD_BARS | 3 |
| MODAL_PER_TRADE | $10 |
| LEVERAGE | 5x |

> ic32_regime_v1 tetap sebagai model produksi di swint_tradev2. Widyawardhana v2 adalah kandidat riset terbaik saat ini.
> Detail lengkap: `reports/widyawardhana_model.md` — diupdate setiap kali model meta baru ditetapkan

## Scorecard — Holdout Apr 1 – Jun 13, 2026 (21 koin, ~2.5 bulan)

| Metrik | Widyawardhana v2 | ic32 (same period) |
|--------|:----------------:|:------------------:|
| **Total Trades** | **905** | 936 |
| Trades/bulan | 362 | 374 |
| **Win Rate** | **68.2%** | 62.1% |
| **Profit Factor** | **2.79** | 2.54 |
| **Net PnL ($10/trade, 5x)** | **+$301** | +$207 |
| PnL/bulan | +$120 | +$83 |
| **PnL/trade** | **+$0.332** | +$0.221 |
| Guardian Exit % | 65.1% | — |
| SL Hit Rate | 0.0% | — |

> Scorecard menggunakan live-like eval (SL + time exit, tanpa TP).
> ic32 benchmark pada periode sama dipakai sebagai baseline perbandingan.
> Scorecard ic32 lama (Nov 2025 – Apr 2026, 5 bulan): lihat `reports/experiments/2026-06-09_widyawardhana_v1.md`
> Model meta sebelumnya tersimpan di `reports/experiments/` dengan tanggal penetapan.

## Slash Commands

- `/trade-analysis` — analisis performa trading live. Detail: `.claude/commands/trade-analysis.md`

## Eksperimen Baru — Wajib Benchmark

Setiap train model baru wajib benchmark terhadap model terbaik yang ada (`ic32_regime_v1`) atau model yang ditentukan user. Gunakan data holdout yang sama. Tampilkan scorecard + leak audit + identifikasi kejanggalan. Simpan hasil di `models/runs/{run_name}/`.
