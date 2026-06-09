# 2026-06-09 — Widyawardhana v1: Dual-Model Architecture

**Tanggal**: 2026-06-09 | **Periode Holdout**: Nov 2025 – Apr 2026 (5 bulan OOS)
**Author**: Heathclif-cyber | **Status**: LGBM Production-ready, LSTM in-training

---

## 1. LGBM Entry Model — Champion Comparison

### 1.1 Scorecard Head-to-Head

| Metrik | `ic32_hybrid_ic_v1` | `ic32_regime_v1` |
|:---|---:|:---|
| **Fitur** | **32** | 33 |
| **Total Net PnL (5x)** | **$+805.68** 🥇 | $+699.87 |
| **Portfolio ROI** | **+38.37%** | +33.33% |
| **Win Rate** | 60.56% | **63.38%** 🥇 |
| **Total Trades** | **3,427** | 2,415 |
| **Trade/Bulan** | **33.1** | 23.3 |
| **Sharpe Ratio** | 4.93 | **5.19** 🥇 |
| **Sortino Ratio** | 12.94 | **14.02** 🥇 |
| **Calmar Ratio** | 11.99 | **18.59** 🥇 |
| **Profit Factor** | 1.88 | **2.28** 🥇 |
| **Max Drawdown (5x)** | 82.75% | **56.59%** 🥇 |
| **Max Consecutive Loss** | 14 | **11** 🥇 |
| **Avg Win PnL** | +$0.80 (8.0%) | +$0.85 (8.5%) |
| **Avg Loss PnL** | -$0.71 (7.1%) | -$0.68 (6.8%) |

### 1.2 Exit Distribution

| Exit Type | `ic32_hybrid_ic_v1` | `ic32_regime_v1` |
|:---|---:|---:|
| Guardian Exit | 79.9% (WR 70.9%) | 78.1% (WR 78.4%) |
| SL Hit | 17.9% (WR 0.0%) | 19.7% (WR 0.2%) |
| Time Exit | 2.2% (WR 98.7%) | 2.2% (WR 88.7%) |

### 1.3 Direction Analysis

| | `ic32_hybrid_ic_v1` | `ic32_regime_v1` |
|:---|---:|---:|
| LONG Trades | 362 (10.6%, WR 65.2%) | 626 (25.9%, WR 64.4%) |
| SHORT Trades | 3,065 (89.4%, WR 59.9%) | 1,789 (74.1%, WR 62.8%) |

### 1.4 `ic32_hybrid_ic_v1` — 32 Features

```
═══ 23 IC32 KEEP (IC-test verified) ═══
dist_from_8h_high, rsi_6, swing_momentum, rsi_h4, stochrsi_k,
dist_liq_50x_long, trend_accel_4h, rsi_slope_h4, Fib_786, stochrsi_d,
ofi_h4_delta, dist_liq_50x_short, Buy_Liq, dist_liq_20x_long,
cvd_momentum_adv, Sell_Liq, cvd_slope_h4, ema_21_slope_h4,
ema_50_h1, h4_trend, log_ret_20, whale_retail_divergence

═══ 2 ETF Flow (yfinance, FREE) ═══
etf_total_change_usd, etf_gbtc_change_usd

═══ 7 WEAK (retained for non-linear LGBM interaction) ═══
dist_liq_20x_short, vol_price_confirm, MSB_BOS, ema_50_slope_h4,
cvd, ofi_acceleration, cvd_div_h4

═══ Regime ═══
hmm_regime_enc
```

### 1.5 `ic32_regime_v1` — 33 Features

```
32 IC32 KEEP + hmm_regime_enc
(Tidak termasuk ETF flow)
```

Full list: `models/feature_cols_ic32_regime.json`

---

## 2. LSTM Survival Filter

### 2.1 Current Production (`ic32_regime_v1`)

| Parameter | Value |
|:---|:---|
| Features | 11 temporal OHLCV (default) |
| seq_len | 32 |
| Role | Soft survival filter (hard_consensus mode) |
| Contribution | **MARGINAL** — near 0 trade impact |
| Recommendation | Deactivate / replace with new model |

### 2.2 New LSTM — `ic32_hybrid_lstm` (in-training)

| Parameter | Value |
|:---|:---|
| **Features** | **18** (MDA-selected from 31 IC+MI candidates) |
| **seq_len** | **72** (3 hari — ETF flow berubah 3× dalam window) |
| **hidden** | 96 |
| **layers** | 2 |
| **dropout** | 0.45 |
| **weight_decay** | 0.0002 |
| **lr** | 0.0007 |
| **step** | 3 |
| **Dataset** | 261,176 seq × 72 × 18 |
| **Status** | 🔄 Training (8-fold CV) |

#### 18 Features (MDA > 0, seq_len=72 validated)

```
═══ Order Flow (5) ═══
cvd_slope_h4, cvd_momentum_adv, volume_delta, ofi_h4_delta, ofi_z_score

═══ Momentum & Trend (4) ═══
swing_momentum, trend_accel_4h, rsi_slope_h4, log_ret_20

═══ Absorption (3) ═══
absorption_at_swing, vol_price_confirm, whale_retail_divergence

═══ Liquidity (3) ═══
long_short_ratio, Buy_Liq, dist_liq_50x_long

═══ ETF Macro (2) ═══
etf_total_change_usd, etf_gbtc_change_usd    ← FREE (yfinance)

═══ Oscillator (1) ═══
stochrsi_d
```

#### Feature Selection Pipeline

```
36 candidates → Stage 1 (IC + Mutual Info) → 31 KEEP + 1 WEAK
                                          → Stage 2 (MDA, seq_len=72) → 18 KEEP
```

#### 13 Features DROPPED (MDA negative — LSTM cannot use)

| Feature | SA_IC | Marg_IC | Why Dropped |
|:---|---:|---:|:---|
| `rsi_6` | -0.143 | -0.154 | Oscillator — no trend in sequence |
| `stochrsi_k` | -0.105 | -0.049 | Oscillator — no trend |
| `rsi_h4` | -0.116 | -0.065 | Oscillator — no trend |
| `dist_from_8h_high` | -0.153 | -0.153 | Static level — LGBM domain |
| `price_in_range` | -0.095 | -0.085 | Static level — LGBM domain |
| `dist_liq_50x_short` | +0.074 | -0.059 | Static level — LGBM domain |
| `ema_7_h1` | +0.139 | +0.099 | Redundant with trend_accel_4h |
| `ema_21_slope_h4` | -0.052 | -0.036 | Redundant with trend_accel_4h |
| `Sell_Liq` | +0.066 | +0.007 | Liquidation level — tree model |
| `hmm_regime_enc` | +0.030 | +0.008 | Constant in 72-bar window |
| `h4_trend` | -0.036 | -0.027 | Signal too weak |
| `log_ret_5` | -0.115 | +0.047 | Signal too weak |
| `ofi_raw` | -0.030 | +0.079 | Signal too weak |

> **Key insight**: All 13 dropped features PASSED IC Marginal test (>|0.005|) but FAILED MDA.
> IC only measures linear correlation; MDA measures contribution INSIDE the trained LSTM.
> Static features (oscillators, price levels, regime) are LGBM's domain, not LSTM's.

---

## 3. Guardian — Exit Model

### 3.1 Current: `ic32_guardian_clean_v2`

| Parameter | Value |
|:---|:---|
| Features | 34 static + 7 dynamic |
| Classes | HOLD / PARTIAL_EXIT(50%) / FULL_EXIT |
| Threshold | 0.65 |
| Min Hold Bars | 2 |
| Run Dir | `models/runs/ic32_guardian_clean_v2/` |

### 3.2 Guardian Performance

| Model | Guardian WR | Guardian Exit % | SL Hit % |
|:---|---:|---:|---:|
| `ic32_hybrid_ic_v1` | 70.9% | 79.9% | 17.9% |
| `ic32_regime_v1` | **78.4%** | 78.1% | 19.7% |

### 3.3 Guardian Plan

| Item | Priority | Notes |
|:---|:---|:---|
| Retrain Guardian dengan ETF features | High | ETF flow sebagai konteks makro exit |
| Optimasi threshold per-regime | Medium | RANGING vs TRENDING berbeda exit behavior |
| Tambah macro cross-asset features | Low | SPY, VIX, GLD sebagai risk-off trigger exit |

---

## 4. Realisasi & Rencana

### 4.1 Realisasi per 2026-06-09

| Component | Status | Detail |
|:---|:---|:---|
| LGBM Entry — `ic32_hybrid_ic_v1` | ✅ Production-ready | 32 fitur, PnL $806, 100% free |
| LGBM Entry — `ic32_regime_v1` | ✅ Fallback | 33 fitur, risk-adjusted superior |
| LSTM — `ic32_hybrid_lstm` | 🔄 Training | 18 fitur MDA, seq_len=72 |
| Guardian — `clean_v2` | ✅ Active | Shared across both models |
| ETF Flow (yfinance) | ✅ Free | Pipeline migrated from Coinank |
| Feature Selection Pipeline | ✅ Complete | LGBM: IC+MI, LSTM: IC+MI+MDA |

### 4.2 Immediate Next Steps

1. **Selesaikan LSTM training** — 8-fold CV dengan 18 fitur + seq_len=72
2. **Integrasi LSTM baru** — ganti LSTM lama (F1 ~0.42) dengan model baru
3. **Full backtest cascade** — LGBM + LSTM baru + Guardian
4. **Risk management** — atasi MaxDD 82.8% di `ic32_hybrid_ic_v1`

### 4.3 Architecture Target — Widyawardhana v1

```
┌─────────────────────────────────────────────────┐
│              WIDYAWARDHANA v1                     │
│                                                   │
│  Entry: LGBM ic32_hybrid_ic_v1 (32 feat)         │
│         ├─ 23 IC32 KEEP                          │
│         ├─ 2 ETF Flow (yfinance, FREE)           │
│         └─ 7 WEAK (non-linear interaction)       │
│                                                   │
│  Filter: LSTM ic32_hybrid_lstm (18 feat, 72-seq) │
│         ├─ 5 Order Flow                          │
│         ├─ 4 Momentum/Trend                      │
│         ├─ 3 Absorption                          │
│         ├─ 3 Liquidity                           │
│         ├─ 2 ETF Macro                           │
│         └─ 1 Oscillator                          │
│                                                   │
│  Exit: Guardian clean_v2 (40 feat)               │
│         └─ Dynamic per-bar HOLD/PARTIAL/FULL     │
│                                                   │
│  100% FREE — No paid API dependencies            │
└─────────────────────────────────────────────────┘
```

---

## 5. Data Pipeline — All Free

| Data Source | Fetch Script | Output | Frequency |
|:---|:---|:---|:---|
| Binance Klines | `01_fetch.py` | OHLCV H1 | Hourly |
| ETF Flow (yfinance) | `01d_fetch_macro_yfinance.py` | `data/macro/etf_flow_btc.parquet` | Daily |
| Macro Cross-Asset | `01d_fetch_macro_yfinance.py` | `data/macro/macro_cross_asset.parquet` | Daily |
| Positioning | `01c_fetch_positioning.py` | `data/positioning/` | Hourly |

> **No Coinank dependency** — all data now sourced from free APIs (Binance + yfinance).

---

*Generated by Claude Code | 🤖 Generated with [Claude Code](https://claude.com/claude-code)*
