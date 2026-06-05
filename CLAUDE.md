# CLAUDE.md — Riset Pemodelan Crypto Trading

## Project Overview

Sistem trading kripto berbasis ML untuk Binance Futures.
- **Versi aktif**: lihat `models/model_registry.json` untuk production version & metrik terkini
- **Research state**: lihat `EXPERIMENTS.md` untuk eksperimen yang sedang/akan berjalan
- **Data**: 2020-01-01 – 2026-04-01, 21 koin, H1 base + H4 swing/regime
- **TRAIN_CUTOFF_DATE = 2025-11-01** — training sampai Okt 2025; holdout Nov 2025 – Apr 2026

## Critical Constraints

- **Python 3.12**, Windows, AMD RX 6600 — DirectML untuk LSTM, OpenCL untuk LGBM
- **Shell**: PowerShell. Gunakan `;` bukan `&&` untuk chaining
- **Encoding**: Terminal cp1252 — jangan pakai unicode arrow (`→`) di logger messages
- **KLINE_LIMIT = 1000** — Binance max per request (bukan 1500)
- **TRAIN_CUTOFF_DATE** — tidak boleh ada data post-cutoff bocor ke training, di mana pun
- **LSTM**: Custom `ManualLSTMCell` — train GPU (DirectML), infer CPU

## Larangan (Do NOT)

- **Jangan duplikasi isi `config.py`** — baca langsung dari file; `config.py` adalah source of truth
- **Jangan tulis riwayat perubahan di sini** — gunakan `EXPERIMENTS.md`
- **Jangan re-implement TP/SL regressor/classifier** — file sudah dihapus; diskusi dulu sebelum membuat ulang
- **Jangan modifikasi file di `swint_tradev2` secara manual** — deployment hanya via `tools/deploy_model.py`
- **Metrik lama TIDAK VALID** — WR 88.93%, PnL $169k dicabut karena data leakage (2026-06-04). Detail: `EXPERIMENTS.md § 2026-06-04`

## Retraining Protocol

Sebelum training model baru, tanyakan nama versi secara eksplisit (contoh: `cascade_v4.4`). Setelah ditentukan, catat di sini: tanggal training, fitur yang dipakai, periode holdout OOS, dan path model (`models/runs/{run_id}/`).

## Cross-Repo: Production (swint_tradev2)

Repo produksi di `D:\Apps-Dev\swint_tradev2`. Alur kerja **satu arah** dari repo ini:
- Analisis: `tools/trade_analyzer.py` membaca langsung `D:\Apps-Dev\swint_tradev2\hasil_livetrading.csv`
- Deployment: `python tools/deploy_model.py` (jangan edit manual di repo produksi)

File kunci produksi yang sering dibaca:

| File | Purpose |
|------|---------|
| `D:\Apps-Dev\swint_tradev2\CLAUDE.md` | Dokumentasi lengkap sistem produksi |
| `D:\Apps-Dev\swint_tradev2\hasil_livetrading.csv` | Live trade history |
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
| `tools/deploy_model.py` | Deployment ke swint_tradev2 |
| `models/model_registry.json` | Model aktif & metrik baseline |
| `reports/experiments/` | Laporan holdout point-in-time historis |

## Pipeline Sequence

```
01_fetch → 02_clean → 03_engineer → 04_train_lgbm
                                   → 05a_generate_momentum_labels
                                   → 05b_build_h1_sequences
                                   → 05c_train_lstm_h1
                                   → 06_train_guardian → 07_holdout_backtest
```

## Slash Commands

- `/trade-analysis` — analisis performa trading live. Detail: `.claude/commands/trade-analysis.md`
