# CLAUDE.md — Riset Pemodelan Crypto Trading

## Project Overview

Sistem trading kripto berbasis ML untuk Binance Futures. Arsitektur **2-model cascade**:
LGBM Classifier (entry signal) → LSTM Soft Confirmation → Swing/ATR Gate (TP/SL).

Periode data: 2020-01-01 s/d 2026-04-01. Timeframe: H1 base, H4 untuk swing + regime, D1 untuk HTF context.
20 koin: SOL, ETH, BNB, XRP, DOGE, TON, ADA, TRX, 1000SHIB, AVAX, LINK, DOT, SUI, POL, NEAR, 1000PEPE, TAO, ARB, XAUT, HBAR, ONDO.

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
                                    │  min_rr=0.5  │
                                    │  min_tp=1.2x │
                                    │  max_sl=4.0x │
                                    └──────┬───────┘
                                           ▼
                                    ┌──────────────┐
                                    │  EXECUTE     │
                                    │  max_hold=24 │
                                    └──────────────┘
```

- **LGBM**: 3-class (SHORT=0, FLAT=1, LONG=2), 103 features, cost-sensitive weights {0:3, 1:1.5, 2:3}
- **LSTM**: `ManualLSTMCell` custom (DirectML compatible), seq_len=16, hidden=128, 2-layer. Training via DirectML GPU, inference via CPU
- **TP/SL**: Swing H4 structural levels + ATR fallback only. **NO ML, NO regime detection** — all approaches degraded performance

## Cascade Flow (detail)

```
LGBM predict 3-class
  │
  ├─ LGBM LONG/SHORT >= 0.62 → LSTM tiered adjustment → conf >= 0.62? → SIGNAL
  │                                                                FAIL → FLAT
  │
  └─ Keduanya < 0.62 (LGBM FLAT) → FLAT (selesai)
       │
       └─ FLAT REVIEW = DISABLED (LSTM_FLAT_REVIEW_ENABLED=False)
            Terbukti menambah 2,500+ trade dengan WR 39% — menurunkan WR dari 78% ke 57%.
            Detail: EXPERIMENTS.md § 2026-05-12
```

**Alur aktif**: LGBM entry signal → LSTM confirm/adjust → entry jika keduanya setuju.
LSTM TIDAK bisa meng-override keputusan FLAT LGBM.

## Referensi Eksternal

- **EXPERIMENTS.md** — Logbook perubahan parameter & temuan eksperimen. Baca sebelum mengubah parameter.
- **Inference config**: `D:\Apps-Dev\swint_tradev2\models\inference_config.json` — setup produksi yang sudah tervalidasi. Parameter di config.py training HARUS konsisten dengan ini.
- **Model registry**: `models/model_registry.json` — model aktif & metrik baseline

## Key Files

| File | Role |
|------|------|
| `config.py` | Semua parameter terpusat — **source of truth**, jangan diduplikasi |
| `core/features.py` | Feature engineering + swing labeling v3 |
| `core/models.py` | `TradingLSTM`, `_CellLSTM`, `_ManualLSTMCell`, `ProbabilityCalibrator` |
| `core/evaluator.py` | `simulate_trades_swing()` + `full_trading_report()` — evaluasi training |
| `core/utils.py` | Logger, device utils |
| `core/fetchers.py` | Binance data fetch |
| `core/binance_client.py` | HTTP client Binance |
| `pipeline/01_fetch.py` | Fetch semua koin |
| `pipeline/02_clean.py` | Clean + resample |
| `pipeline/03_analyze_swing.py` | H4 swing detection analysis |
| `pipeline/04_engineer.py` | Feature engineering pipeline |
| `pipeline/05_train_lgbm.py` | LGBM entry model training |
| `pipeline/06_train_lstm.py` | LSTM soft confirmation training |
| `pipeline/07_evaluate.py` | Evaluasi cascade (SOLUSDT) |
| `pipeline/08_backtest.py` | Walk-forward backtest |
| `pipeline/09_holdout_backtest.py` | Genuine OOS holdout backtest |
| `pipeline/10_visualize.py` | Visualisasi hasil |
| `pipeline/shared.py` | `SequenceDataset` + `build_purged_folds()` |
| `pipeline/backtest_utils.py` | `hierarchical_predict()` + `get_lstm_proba()` + `_lstm_adjustment()` |
| `pipeline/test_inference_backtest.py` | **Backtest mandiri pakai inference config** — untuk uji parameter tanpa polusi pipeline training |

## Pipeline Sequence (Order Matters)

```
01_fetch → 02_clean → 03_analyze_swing → 04_engineer → 05_train_lgbm → 06_train_lstm → 07_evaluate → 08_backtest → 09_holdout_backtest → 10_visualize
```

## Key Learnings (What Failed & Why)

### ML for TP/SL — ALL FAILED

| Approach | Result | Why |
|----------|--------|-----|
| TP/SL Regressor (LGBMRegressor) | WR 75%→37%, DD 64%→113% | SL R²=0.05 — entry bar has no signal for 24-bar ahead multiplier |
| Safe SL Classifier (Binary) | AUC=0.62, never triggered | Cannot predict if structural level holds from 1 bar |
| Regime Classifier (ML Binary) | Always predicts RANGING | Same problem — 1 bar can't predict 24-bar regime |
| Rule-Based Regime | Trend%=0-3%, no effect | Thresholds too conservative, but at least doesn't degrade |

**Root cause**: Entry bar features cannot predict what happens 24 bars ahead. Signal-to-noise ratio too low.

### What Works

- **Swing/ATR gate** — structural levels are real, statistically meaningful
- **Walk-forward purged CV** — prevents look-ahead leakage
- **Confidence filter** — reduces noise trades (threshold 0.62, selaras dgn cascade internal)
- **SHORT = LONG** — model tidak bias arah; WR kedua arah identik (~78% di holdout)

### What We Learned About LSTM (2026-05-12)

- **LSTM FLAT review** menambah volume trade 2x tapi menurunkan WR dari 78% ke 57%. Zona LGBM FLAT adalah zona noise — LSTM tidak bisa memprediksi di sana.
- **LSTM opposite penalty** yang terlalu keras (0.08) membunuh sinyal bagus. Diturunkan ke 0.04.
- Detail di `EXPERIMENTS.md`

## Important Constraints

- **Python 3.12** on **Windows 10** with **AMD RX 6600 GPU** (DirectML)
- Shell: PowerShell (not bash). Use `;` not `&&` for chaining
- **Encoding**: Terminal is cp1252 — avoid unicode arrows (→) in logger messages
- **LSTM**: Custom `ManualLSTMCell` for DirectML compatibility. Train on GPU, infer on CPU
- **LGBM**: `device_type="gpu"` via OpenCL (compatible with AMD)
- **Data**: 5 training coins (SOL, ETH, BNB, XRP, DOGE) + 15 holdout coins
- **TP/SL regressor/classifier files DELETED** — do not re-implement without discussing why previous attempts failed
- **Jangan duplikasi isi config.py** — baca langsung dari file. Duplikasi di doc akan stale.
- **Jangan tulis riwayat perubahan di sini** — gunakan `EXPERIMENTS.md`
