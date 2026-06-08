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

## Backtest Methodology — MANDATORY

**SEMUA extended backtest HARUS pakai Purged CV OOF.**
- Retrain LGBM per fold dari nol — model TIDAK BOLEH lihat data uji
- Testing dengan fixed model (`lgbm_baseline.pkl`) di 2020-2025 = **IN-SAMPLE LEAKAGE**
- Hasil in-sample (PnL +$9,878, WR 69%) TIDAK VALID sebagai ekspektasi OOF
- Hasil genuine OOF: PnL sekitar -$362 s/d -$148 (EXPERIMENTS.md §2026-06-06)

**Cara validasi yang benar:**
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

## Larangan (Do NOT)

- **Jangan duplikasi isi `config.py`** — baca langsung dari file; `config.py` adalah source of truth
- **Jangan tulis riwayat perubahan di sini** — gunakan `EXPERIMENTS.md`
- **Jangan re-implement TP/SL regressor/classifier** — file sudah dihapus; diskusi dulu sebelum membuat ulang
- **Jangan pakai fixed model untuk extended backtest** — IN-SAMPLE LEAKAGE. Harus purged CV OOF retrain per fold.
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
01_fetch → 02_clean → 03_engineer → 03e_regime_hmm → 04_train_lgbm
                                 → 06b_train_guardian_clean_v2 → 07_holdout_backtest

01c_fetch_positioning → hourly cron (Windows Task Scheduler: FetchPositioningData)
                        4 endpoint: Binance taker_ratio + top_trader + global_ls + Bybit OI
                        Output: data/positioning/{coin}_{type}.parquet
                        Target: 4,000+ bar/jam dalam 6 bulan untuk training momentum model

05_train_lstm_v2_style → LSTM training (v2-style, seq=32)
11_train_lstm_daily → LSTM Daily binary momentum (AUC 0.611) — cadangan untuk professional_v2
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

## Active Configuration (2026-06-07)

```
Entry   : LGBM ic32_regime_v1 (33 feat: 32 KEEP IC-test + HMM argmax)
LSTM    : ON — survival filter (11 feat temporal, soft multiplier 0.70-1.30)
Cascade : hard_consensus + FLIP alignment (regime-aware: RANGING=counter-trend, TRENDING=with-trend)
Guardian: ic32_guardian_clean_v2 (40 feat, min_hold=2)
HMM     : argmax sebagai fitur LGBM ke-33 (IC=-0.074)
Data    : Positioning mining aktif — 4 endpoint Binance+Bybit hourly (83 file, 21 koin)
```

**Parameters:**
| Parameter | Value |
|-----------|-------|
| LGBM_THRESHOLD_LONG | 0.69 |
| LGBM_THRESHOLD_SHORT | 0.59 |
| CONFIDENCE_THRESHOLD_ENTRY | 0.59 |
| LSTM_CONFIRMATION_ENABLED | True |
| LSTM_FLAT_REVIEW_ENABLED | True |
| TREND_ALIGNMENT_ENABLED | False |
| REGIME_AWARE_ALIGNMENT | True |
| GUARDIAN_ENABLED | True |
| GUARDIAN_MIN_HOLD_BARS | 2 |
| GUARDIAN_EXIT_THRESHOLD | 0.65 |
| MODAL_PER_TRADE | $10 |
| LEVERAGE | 5x |

## Scorecard — Holdout Nov 2025 – Apr 2026 (21 koin, 5 bulan)

| Metrik | Nilai |
|--------|-------|
| **Total Trades** | 2,434 |
| Trades/bulan | 487 |
| **Win Rate** | **67.5%** |
| LONG WR | 67.6% (682 trades, 28.0%) |
| SHORT WR | 67.4% (1,752 trades) |
| **Net PnL ($10/trade, 5x)** | **$848** |
| PnL/bulan | $170 |
| PnL/trade | $0.35 |
| **Profit Factor** | **2.54** |
| Sharpe Ratio (daily) | ~15 * |
| Sharpe Ratio (live est.) | ~1.5-2.5 |
| Max Drawdown | ~15% |
> *Sharpe backtest inflated: 5 bln short period + 21 koin diversification + sqrt(365) annualization + perfect execution assumption. Live trading expect Sharpe 1.5-2.5. PF dan WR lebih reliable sebagai metrik backtest.
| Max Consecutive Loss | 22 |
| Avg Hold Bars | 9.6 |
| SL Hit Rate | 17.3% (422 trades) |
| Guardian Exit WR | 79.6% (1,792 trades) |
| Time Exit | 9.0% (220 trades) |

> Note: Sharpe/Sortino/Calmar dihitung dari daily equity curve $10/trade.
> Sharpe = mean(daily_return) / std(daily_return) * sqrt(365).
> Sortino = mean(daily_return) / std(neg_daily_return) * sqrt(365).
> Calmar = annualized_PnL / max_drawdown.
> Max DD dihitung dari peak-to-trough equity curve.

## Slash Commands

- `/trade-analysis` — analisis performa trading live. Detail: `.claude/commands/trade-analysis.md`
