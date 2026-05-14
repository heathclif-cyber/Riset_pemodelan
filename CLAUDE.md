# CLAUDE.md — Riset Pemodelan Crypto Trading

## Project Overview

Sistem trading kripto berbasis ML untuk Binance Futures. Arsitektur **3-model cascade**:
LGBM Classifier (entry) → LSTM Soft Confirmation → **Guardian v3 (dynamic exit)**.

Periode data: 2020-01-01 s/d 2026-04-01. Timeframe: H1 base, H4 untuk swing + regime, D1 untuk HTF context.
**TRAIN_CUTOFF_DATE = 2025-05-01** — training hanya di 2020-2025, holdout test di Mei 2025 – Apr 2026.
21 koin: SOL, ETH, BNB, XRP, DOGE, TON, ADA, TRX, 1000SHIB, AVAX, LINK, DOT, SUI, POL, NEAR, 1000PEPE, TAO, ARB, XAUT, HBAR, ONDO.

## Architecture

```
┌─────────┐    ┌──────────┐    ┌──────────────┐    ┌───────────────┐
│  LGBM   │───▶│   LSTM   │───▶│  Confidence  │───▶│  Swing/ATR    │
│ Classif │    │  Soft    │    │  Filter      │    │  TP/SL Gate   │
│ 3-class │    │ Confirm  │    │  >= 0.62     │    │               │
└─────────┘    └──────────┘    └──────────────┘    └───────┬───────┘
                                                           │
                                            ┌──────────────┘
                                            ▼
                                    ┌──────────────┐
                                    │  RR Gate     │
                                    │  min_rr=1.0  │
                                    │  min_tp=1.2x │
                                    │  max_sl=4.0x │
                                    └──────┬───────┘
                                           ▼
                                    ┌──────────────┐
                                    │  EXECUTE     │
                                    │  max_hold=24 │
                                    └──────┬───────┘
                                           ▼
                                    ┌──────────────┐
                                    │ Guardian v3  │ ← per-bar dynamic exit
                                    │ HOLD / PART  │    104 feat + 7 dynamic
                                    │ / FULL EXIT  │    multiclass LGBM
                                    └──────────────┘
```

- **LGBM**: 3-class (SHORT=0, FLAT=1, LONG=2), 104 features, cost-sensitive weights {0:3, 1:1.5, 2:3}, GPU OpenCL
- **LSTM**: `ManualLSTMCell` custom (DirectML compatible), seq_len=16, hidden=128, 2-layer. Training via DirectML GPU, inference via CPU
- **Guardian v3**: Multiclass (0=HOLD, 1=PARTIAL_EXIT, 2=FULL_EXIT), 104 static + 7 dynamic features, GPU OpenCL. Partial exit = 50% posisi
- **TP/SL**: Hybrid H4 Swing + ATR Fallback (non-ML). `TP_SL_HYBRID_MODE=True`

## Cascade Flow (detail)

```
LGBM predict 3-class
  │
  ├─ LGBM LONG/SHORT >= 0.62 → LSTM tiered adjustment → conf >= 0.62? → ENTRY SIGNAL
  │                                                                FAIL → FLAT
  │
  └─ Keduanya < 0.62 (LGBM FLAT) → FLAT (selesai)
       │
       └─ FLAT REVIEW = DISABLED (LSTM_FLAT_REVIEW_ENABLED=False)
            Terbukti menambah 2,500+ trade dengan WR 39% — menurunkan WR dari 78% ke 57%.
            Detail: EXPERIMENTS.md § 2026-05-12

ENTRY → Guardian v3 per-bar check (setelah 3 bar + 1×ATR move):
  ├─ HOLD (0)          → lanjut scan
  ├─ PARTIAL_EXIT (1)  → tutup 50% posisi, lanjut 50% sisanya
  └─ FULL_EXIT (2)     → tutup seluruh posisi
```

**Alur aktif**: LGBM entry → LSTM confirm → Guardian v3 dynamic exit.
TRAILING_STOP_ENABLED=False (Guardian solo beats trailing). LSTM TIDAK bisa meng-override FLAT.

## Final Results (2026-05-15)

### 08 Backtest — Walk-Forward Purged CV (2020-2025, 20 koin)

| Metrik | Nilai |
|--------|-------|
| Mean WR | 91.15% |
| Mean DD | 85.80% |
| Mean PF | 13.31 |

### 09 Holdout — Genuine Temporal OOS (Mei 2025 – Apr 2026, 21 koin, 8,027 bar/koin)

| Metrik | Guardian v3 | Baseline (no Guardian) | Delta |
|--------|-------------|------------------------|-------|
| Mean WR | **88.93%** | 82.03% | +6.90pp |
| Mean DD | **41.77%** | 55.75% | −13.98pp |
| Mean PF | **10.05** | 8.41 | +1.64 |
| Mean Sharpe | **38.32** | 25.75 | +12.57 |
| Mean Sortino | **78.99** | 54.60 | +24.39 |
| Max Cons Loss | **7** | 9 | −2 |
| Trades/bulan | **103.7** | 62.0 | +67% |
| **Total PnL (5x, 21 koin)** | **$169,626** | $113,802 | **+$55,824 (+49%)** |
| LONG WR | 87.8% | — | — |
| SHORT WR | **90.3%** | — | — |

SHORT > LONG +2.5% — bukan bias model, market structure bull market (koreksi tajam → SHORT TP cepat).
**Semua 21 koin naik PnL** vs baseline, NEAR tertinggi (+$4,261), TRX terendah (+$385).

### Guardian v2 → v3 Transition

| Metrik | Guardian v2 (Binary) | Guardian v3 (Multiclass) | Delta |
|--------|---------------------|--------------------------|-------|
| Mean WR | 90.88% | 88.93% | −1.95pp |
| Mean DD | 38.06% | 41.77% | +3.71pp |
| Mean PF | 14.05 | 10.05 | −4.00 |
| Mean Sharpe | 33.24 | **38.32** | +5.08 |
| Total Trades | 13,301 | **22,914** | +72% |
| Total PnL | $107,875 | **$169,626** | **+$61,751 (+57%)** |

v3 korbankan WR/PF demi volume 72% lebih banyak — Sharpe lebih tinggi, PnL +57%.

Detail lengkap: `EXPERIMENTS.md § 2026-05-14 (Sesi 3)` dan `§ 2026-05-15`

## Referensi Eksternal

- **EXPERIMENTS.md** — Logbook perubahan parameter & temuan eksperimen. Baca sebelum mengubah parameter.
- **Inference config**: `D:\Apps-Dev\swint_tradev2\models\inference_config.json` — setup produksi yang sudah tervalidasi.
- **Model registry**: `models/model_registry.json` — model aktif & metrik baseline
- **Holdout results**: `models/runs/holdout_20260515_001906/holdout_backtest_results.json`

## Key Files

| File | Role |
|------|------|
| `config.py` | Semua parameter terpusat — **source of truth**, jangan diduplikasi |
| `core/features.py` | Feature engineering + swing labeling v3 |
| `core/models.py` | `TradingLSTM`, `_CellLSTM`, `_ManualLSTMCell` |
| `core/evaluator.py` | `simulate_trades_swing()` + Guardian per-bar check + partial exit |
| `core/utils.py` | Logger, device utils, `chunk_time_range()` |
| `core/fetchers.py` | Binance data fetch (`KLINE_LIMIT=1000`) |
| `core/binance_client.py` | HTTP client Binance |
| `pipeline/01_fetch.py` | Fetch semua koin |
| `pipeline/02_clean.py` | Clean + resample |
| `pipeline/03_analyze_swing.py` | H4 swing detection analysis |
| `pipeline/04_engineer.py` | Feature engineering pipeline |
| `pipeline/05_train_lgbm.py` | LGBM entry model training (TRAIN_CUTOFF_DATE filter) |
| `pipeline/06_train_lstm.py` | LSTM soft confirmation training (TRAIN_CUTOFF_DATE filter) |
| `pipeline/07_evaluate.py` | Evaluasi cascade (SOLUSDT) |
| `pipeline/08_backtest.py` | Walk-forward backtest (cascade_v3) |
| `pipeline/09_holdout_backtest.py` | Genuine OOS holdout backtest (Mei 2025 – Apr 2026) |
| `pipeline/10_visualize.py` | Visualisasi hasil |
| `pipeline/shared.py` | `SequenceDataset` + `build_purged_folds()` |
| `pipeline/backtest_utils.py` | `hierarchical_predict()` + feature alignment via `feature_name_` |
| `pipeline/15_train_guardian.py` | **Guardian v3 training** — multiclass LGBM, TRAIN_CUTOFF_DATE filter |

## Pipeline Sequence (Order Matters)

```
01_fetch → 02_clean → 03_analyze_swing → 04_engineer → 05_train_lgbm → 06_train_lstm → 15_train_guardian → 07_evaluate → 08_backtest → 09_holdout_backtest → 10_visualize
```

**Data flow**: Semua training script filter `df.index < TRAIN_CUTOFF_DATE` (2025-05-01).
Holdout test menggunakan data setelah cutoff — genuine temporal OOS.

## Key Learnings

### Guardian v3 — SUCCESS (2026-05-15)

- **104 feat + multiclass > 32 feat binary**: Static features (ema_7_h4, rsi_h4, rsi_slope_h4, atr_percent_h4) berkontribusi nyata
- **WR 89% di temporal OOS**: Guardian genuine generalization, bukan overfitting. Semua 21 koin PnL positif
- **TP → momentum mode = game changer**: Trade +72%, PnL +57% vs Guardian v2 binary. TP tidak hard-close posisi
- **Guardian > Trailing stop**: Guardian v3 mengalahkan trailing 2x ATR di semua metrik
- **Feature alignment robust**: `model.feature_name_` + zero-fill mencegah mismatch kolom
- **Partial exit belum optimal**: Minority class 4.5%, perlu monitoring lebih lanjut

### ML for TP/SL — ALL FAILED

| Approach | Result | Why |
|----------|--------|-----|
| TP/SL Regressor (LGBMRegressor) | WR 75%→37%, DD 64%→113% | SL R²=0.05 — entry bar has no signal for 24-bar ahead multiplier |
| Safe SL Classifier (Binary) | AUC=0.62, never triggered | Cannot predict if structural level holds from 1 bar |
| Regime Classifier (ML Binary) | Always predicts RANGING | Same problem — 1 bar can't predict 24-bar regime |
| Rule-Based Regime | Trend%=0-3%, no effect | Thresholds too conservative |

### What Works

- **Guardian v3** — dynamic exit, WR 89%, DD 42%, PnL +49% vs baseline di temporal OOS (21 koin)
- **TP → momentum mode** — TP tidak hard-close, trigger Guardian ride profit. Trade +72%, PnL +57% vs Guardian v2
- **Swing/ATR gate** — structural levels are real, statistically meaningful
- **Walk-forward purged CV** — prevents look-ahead leakage
- **Confidence filter** — reduces noise trades (threshold 0.62)
- **SHORT ≈ LONG** — model tidak bias arah; SHORT sedikit lebih akurat (+2.5%) karena market structure

### What We Learned About LSTM (2026-05-12)

- **LSTM FLAT review** menambah volume 2x tapi WR 39% — disabled
- **LSTM opposite penalty** 0.08 → 0.04 — kurangi blocked trades
- Detail: `EXPERIMENTS.md`

## Important Constraints

- **Python 3.12** on **Windows 10** with **AMD RX 6600 GPU** (DirectML)
- Shell: PowerShell (not bash). Use `;` not `&&` for chaining
- **Encoding**: Terminal is cp1252 — avoid unicode arrows (→) in logger messages
- **LSTM**: Custom `ManualLSTMCell` for DirectML compatibility. Train on GPU, infer on CPU
- **LGBM**: `device_type="gpu"` via OpenCL (compatible with AMD)
- **TRAIN_CUTOFF_DATE = 2025-05-01** — tidak boleh ada data testing bocor ke training
- **KLINE_LIMIT = 1000** — Binance max 1000 klines per request (sebelumnya 1500 → gap 21 hari)
- **TP/SL regressor/classifier files DELETED** — jangan re-implement tanpa diskusi
- **Feature alignment via `model.feature_name_`** — mencegah mismatch fitur 103 vs 104
- **Jangan duplikasi isi config.py** — baca langsung dari file
- **Jangan tulis riwayat perubahan di sini** — gunakan `EXPERIMENTS.md`
