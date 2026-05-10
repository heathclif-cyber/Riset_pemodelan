# CLAUDE.md — Riset Pemodelan Crypto Trading

## Project Overview

Sistem trading kripto berbasis ML untuk Binance Futures. Arsitektur **2-model cascade**:
LGBM Classifier (entry signal) → LSTM Soft Confirmation → Swing/ATR Gate (TP/SL).

Periode data: 2020-01-01 s/d 2026-04-01. Timeframe: H1 base, H4 untuk swing + regime, D1 untuk HTF context.
20 koin: SOL, ETH, BNB, XRP, DOGE, TON, ADA, TRX, 1000SHIB, AVAX, LINK, DOT, SUI, POL, NEAR, 1000PEPE, TAO, ARB, XAUT, HBAR, ONDO.

## Architecture (Final)

```
┌─────────┐    ┌──────────┐    ┌──────────────┐    ┌───────────────┐
│  LGBM   │───▶│   LSTM   │───▶│  Confidence  │───▶│  Swing/ATR    │
│ Classif │    │  Soft    │    │  Filter      │    │  TP/SL Gate   │
│ 3-class │    │ Confirm  │    │  ≥ 0.70      │    │               │
└─────────┘    └──────────┘    └──────────────┘    └───────┬───────┘
                                                           │
                                            ┌──────────────┘
                                            ▼
                                    ┌──────────────┐
                                    │  RR Gate     │
                                    │  min_rr=1.0  │
                                    │  min_tp=1.2× │
                                    │  max_sl=3.0× │
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

## TP/SL Determination (Final)

```
Tier 1 — H4 Swing Points (dengan Bumper):
  LONG:  TP = h4_swing_high[i]
         SL = h4_swing_low[i] - (0.5 × ATR)  [Bumper proteksi wick]
  SHORT: TP = h4_swing_low[i]
         SL = h4_swing_high[i] + (0.5 × ATR) [Bumper proteksi wick]

Tier 2 — ATR Fallback (jika swing NaN):
  TP = price ± 2.0 × ATR
  SL = price ∓ 1.5 × ATR

RR Gate (skip trade jika gagal):
  TP_dist ≥ 1.2 × ATR
  SL_dist ≤ 3.0 × ATR
  TP_dist / SL_dist ≥ 0.5
  
Eksekusi Manual:
  Trigger SL = highlow (harga ekstrem menyentuh level SL)
```

## Performance (Baseline Swing/ATR, No ML TP/SL)

| | XRPUSDT | ETHUSDT |
|---|---------|---------|
| In-sample WR | 75.0% | 77.2% |
| In-sample DD | 64.2% | 55.8% |
| Holdout WR | 72.2% | 75.4% |
| Holdout DD | 41.8% | 32.9% |

## Key Files

| File | Role |
|------|------|
| `config.py` | Semua parameter terpusat |
| `core/features.py` | Feature engineering + swing labeling v3 |
| `core/models.py` | `TradingLSTM`, `_CellLSTM`, `_ManualLSTMCell` |
| `core/evaluator.py` | `simulate_trades_swing()` + `full_trading_report()` |
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
| `pipeline/backtest_utils.py` | `hierarchical_predict()` + `get_lstm_proba()` |
| `test_regime.py` | Quick test: regime vs baseline |

## Pipeline Sequence (Order Matters)

```
01_fetch → 02_clean → 03_analyze_swing → 04_engineer → 05_train_lgbm → 06_train_lstm → 07_evaluate → 08_backtest → 09_holdout_backtest → 10_visualize
```

## Config Reference (config.py)

| Param | Value | Notes |
|-------|-------|-------|
| `N_FOLDS` | 8 | Walk-forward folds |
| `PURGE_GAP_BARS` | 5 | Purge both sides of fold boundary |
| `MAX_HOLDING_BARS` | 24 | Max trade duration (24h) |
| `LSTM_SEQ_LEN` | 16 | LSTM lookback window |
| `LSTM_HIDDEN` | 128 | Hidden dimension |
| `LSTM_LAYERS` | 2 | LSTM layers |
| `CONFIDENCE_THRESHOLD_ENTRY` | 0.70 | Min confidence for entry |
| `SWING_LABEL_MIN_RR` | 0.5 | Toleransi RR minimal 0.5 (setengah risiko) |
| `SWING_LABEL_MIN_TP` | 1.2 | Min TP (× ATR) |
| `SWING_LABEL_MAX_SL` | 3.0 | Max SL (× ATR) |
| `TP_SL_SWING_BUMPER` | 0.5 | Bumper SL 0.5x ATR pencegah stop-hunt |
| `MODAL_PER_TRADE` | $100 | Per trade |
| `LEVERAGE_SIM` | [5.0] | 5× leverage |
| `FEE_PER_SIDE` | 0.0004 | 0.04% |
| `SLIPPAGE_PER_SIDE` | 0.0005 | 0.05% |
| `TP_SL_HYBRID_MODE` | True | max(swing,ATR) TP / min(swing,ATR) SL |
| `TP_SL_TRIGGER_MODE` | "highlow" | SL trigger manual order book (menyentuh wick) |
| `TP_SL_FALLBACK_SL` | 1.5 | SL fallback 1.5×ATR (memberi ruang nafas) |
| `TP_SL_COOLDOWN_ENABLED` | False | Cooldown OFF (terlalu restriktif) |
| `TP_SL_RR_GATE_ENABLED` | True | Validasi RR sebelum entry |
| `TP_SL_SWING_FRESHNESS` | True | Tolak trade jika swing >15% dari entry |
| `TP_SL_STRUCTURAL_FILTER` | True | Entry harus dalam [H4 Low, H4 High] |
| `TP_SL_SIZING_MODE` | "fixed" | Fixed $100/trade |
| `TP_SL_SLIPPAGE_ENABLED` | True | Slippage entry/exit 0.05% |


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
- **Rule-based regime** — reads current state (not predicts future), safe failure mode
- **Walk-forward purged CV** — prevents look-ahead leakage
- **Confidence filter (0.70)** — reduces noise trades

## Fixes Applied (2026-05-09)

1. **Swing look-ahead fix**: `h4_swing_high/low` shifted 3 bars forward in `engineer_features()` — removes confirmation bias from feature engineering
2. **Unified fold builder**: All models use `build_purged_folds()` from `pipeline/shared.py` — consistent purge on both sides
3. **Tail protection**: Last `max_hold` bars forced to FLAT (was `max_hold//4`)
4. **FVG fix**: `calc_fvg()` no longer uses `i+1` bar
5. **Regressor + Safe SL + Regime ML**: Fully deleted — code, models, metadata, config sections

---

## Development Roadmap

### 1. Purged Walk-Forward + Robust Validation ★★★★★ — Selesai
- [x] Unify fold builder across all models
- [x] Fix swing look-ahead in feature engineering
- [x] Fix tail protection for labels
- [x] Fix FVG look-ahead
- [ ] Add embargo parameter (`EMBARGO_BARS` in config)
- [ ] Add combinatorial purge (purge after test before next train)

### 2. Regime Detection / Regime-Aware Model ★★★★★ — Segera
Swing/ATR gate works well but doesn't adapt to market regime:
- **Approach A**: Train separate models per regime (Bull/Bear/Sideways)
  - Label regimes via HMM or clustering on H4/D1 features
  - One LGBM+LSTM per regime → regime-specific entry signals
- **Approach B**: Regime as input feature
  - Add `regime_label` as categorical feature to LGBM
  - Let model learn regime-specific patterns internally
- **Approach C**: Meta-model regime switch
  - Simple classifier: detect regime → switch TP/SL strategy
  - TRENDING: wider TP, tighter SL | RANGING: swing TP/SL | HIGH_VOL: skip trade

### 3. Ensemble / Stacking ★★★★ — 1-2 Minggu
- **Meta-learner**: LGBM that takes LGBM+LSTM+XGBoost predictions as input
- **Feature diversity**: Different lookback windows, different feature subsets per model
- **Bagging**: Train same architecture on different coin subsets
- Expected: Sharpe improvement 10-20%, DD reduction

### 4. Temporal Fusion Transformer / Attention LSTM ★★★★ — 2-4 Minggu
- **TFT**: Built for financial time series — variable selection networks + multi-horizon
- **Attention-enhanced LSTM**: Add multi-head attention to existing ManualLSTMCell
- **Informer/Autoformer**: Better for long sequences (if extending lookback)
- Prerequisite: Purged walk-forward must be solid first

### 5. Feature Engineering Lanjutan ★★★ — Berkelanjutan
- **Cross-sectional**: Ranking features across coins (relative strength, correlation breakdown)
- **Microstructure**: Order book imbalance, trade flow toxicity (if available)
- **Lag features**: Systematic lag optimization per coin
- **Volatility regime**: GARCH, Parkinson volatility, realized vol
- **On-chain**: Exchange reserves, whale alerts (if API available)

### 6. Causal Inference Layer ★★★ — Setelah Stabil
- **Refutation tests**: Does the feature actually cause the prediction?
- **DoWhy/EconML**: Causal effect estimation
- **Counterfactual**: What if we DIDN'T enter this trade?

## Important Constraints

- **Python 3.12** on **Windows 10** with **AMD RX 6600 GPU** (DirectML)
- Shell: PowerShell (not bash). Use `;` not `&&` for chaining
- **Encoding**: Terminal is cp1252 — avoid unicode arrows (→) in logger messages
- **LSTM**: Custom `ManualLSTMCell` for DirectML compatibility. Train on GPU, infer on CPU
- **LGBM**: `device_type="gpu"` via OpenCL (compatible with AMD)
- **Data**: 5 training coins (SOL, ETH, BNB, XRP, DOGE) + 15 holdout coins
- **TP/SL regressor/classifier files DELETED** — do not re-implement without discussing why previous attempts failed
