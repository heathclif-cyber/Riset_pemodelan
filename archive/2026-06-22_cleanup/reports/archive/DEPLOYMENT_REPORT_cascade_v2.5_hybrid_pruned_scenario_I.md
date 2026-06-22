# Laporan Deployment: cascade_v2.5_hybrid_pruned — Scenario I
**Tanggal Deploy**: 2026-06-01  
**Versi**: `cascade_v2.5_hybrid_pruned_scenario_I`  
**Status**: AKTIF di produksi (`swint_tradev2`)  
**Backup**: `D:\Apps-Dev\swint_tradev2\models\backups\backup_20260601_154737`

---

## 1. Ringkasan Eksekutif

Model ini menggantikan cascade_v4.1 (104 fitur) dengan pendekatan yang lebih seimbang antara volume trade dan kualitas sinyal. Berdasarkan analisis live trading Mei 2026, konfigurasi ultra-selektif sebelumnya menyebabkan under-trading di regime choppy. Versi ini mengembalikan spirit cascade_v2 sambil mempertahankan Guardian v3 modern.

**Hasil OOS Holdout (Nov 2025 – Mar 2026, 21 koin, 5x leverage):**

| Metrik | Nilai |
|--------|-------|
| Mean Win Rate | **56.21%** |
| Total PnL Portfolio | **+$748.12** |
| Trade/Bulan per Koin | 51.1 |
| Trade/Bulan Total Portfolio | ~1,074 |
| Trade/Hari Total Portfolio | ~35.3 |
| Mean Sharpe Ratio | **1.68** |
| Mean Profit Factor | **1.26** |
| Mean Max Drawdown | 157.03% |
| Max Consecutive Loss | 18 |
| Worst Single Trade Loss | −28.2% |
| 95% Trade Loss Under | 11.3% |

---

## 2. Arsitektur Model

### 2.1 Entry LGBM (Primary Signal)

| Parameter | Nilai |
|-----------|-------|
| Tipe | LightGBM Classifier (3-class: SHORT/FLAT/LONG) |
| Fitur | **87** (pruned dari 104 berdasarkan feature importance) |
| n_estimators (final) | 781 |
| Device | CPU (OpenCL GPU saat training) |
| Class weights | SHORT=3×, FLAT=1.5×, LONG=3× |
| Training period | 2020-01-01 s/d 2025-11-01 |
| CV Mean F1-macro | 0.5987 |
| CV Mean Accuracy | 81.28% |

**File**: `models/lgbm_baseline.pkl`  
**Feature list**: `models/feature_cols_v2.json` (87 fitur)

### 2.2 LSTM Confirmation (Soft Modifier)

| Parameter | Nilai |
|-----------|-------|
| Tipe | TradingLSTM — ManualLSTMCell (DirectML compatible) |
| Arsitektur | hidden=96, layers=2, dropout=0.45 |
| Input features | **87** (sama dengan LGBM) |
| Seq length | 16 H1 bars (16 jam) |
| Num classes | 3 (SHORT/FLAT/LONG) |
| Scaler | RobustScaler |
| Training period | 2020-01-01 s/d 2025-11-01 |
| CV Mean Val F1-macro | 0.4687 |
| Final retrain epochs | 29 |
| Gap train/val (fold akhir) | +0.003 (near zero overfitting) |

**File**: `models/lstm_best.pt`, `models/lstm_scaler.pkl`

### 2.3 Guardian Exit (Dynamic Exit)

| Parameter | Nilai |
|-----------|-------|
| Tipe | LightGBM Classifier (3-class: HOLD/PARTIAL_EXIT/FULL_EXIT) |
| Static features | 87 (sama dengan LGBM entry) |
| Dynamic features | 7 (bars_held, current_pnl, MFE, drawdown, direction, entry_ratio, pnl_atr) |
| Total features | **94** |
| n_estimators (final) | 473 |
| Training samples | 665,798 in-trade bars (21 koin) |
| CV F1-macro mean | ~0.812 |
| F1 HOLD | 0.874–0.888 |
| F1 PARTIAL_EXIT | 0.838–0.865 |
| F1 FULL_EXIT | 0.662–0.754 |

**File**: `models/guardian_best.pkl`, `models/guardian_scaler.pkl`, `models/guardian_feature_cols.json`

---

## 3. Konfigurasi Cascade

### 3.1 Entry Signal Logic

```
LGBM predict 3-class
  │
  ├─ LGBM LONG >= 0.69  →  LSTM adjustment  →  conf >= 0.59? → ENTRY LONG
  ├─ LGBM SHORT >= 0.59 →  LSTM adjustment  →  conf >= 0.59? → ENTRY SHORT
  │
  └─ LGBM dibawah threshold → FLAT
       └─ Directional review: jika LGBM score > 0.35 → LSTM bisa override
          (LSTM LONG/SHORT >= 0.70 → entry dengan override confidence)
```

### 3.2 Parameter Cascade

| Parameter | Nilai | Keterangan |
|-----------|-------|-----------|
| `lgbm_threshold_long` | **0.69** | LGBM harus yakin minimal 69% LONG |
| `lgbm_threshold_short` | **0.59** | LGBM harus yakin minimal 59% SHORT |
| `confidence_threshold_entry` | **0.59** | Final threshold setelah LSTM adjustment |
| `lstm_adjust_mode` | hard_consensus | Mode LSTM adjustment |
| `lstm_adjust_agree_boost` | **+0.05** | LSTM setuju → confidence naik 0.05 |
| `lstm_adjust_neutral_pen` | **0.00** | LSTM netral → tidak ada penalti |
| `lstm_adjust_opposite_pen` | **−0.65** | LSTM berlawanan → confidence turun 0.65 |
| `lstm_flat_review_enabled` | true | LSTM bisa review sinyal di bawah threshold |
| `lstm_directional_review_threshold` | 0.35 | Minimum LGBM score untuk aktivasi review |
| `lstm_override_threshold` | 0.70 | Minimum LSTM confidence untuk override |

### 3.3 Guardian Exit

| Parameter | Nilai | Keterangan |
|-----------|-------|-----------|
| `enabled` | true | Guardian aktif |
| `exit_threshold` | 0.65 | Minimum proba EXIT untuk trigger |
| `min_hold_bars` | **2** | Minimum 2 bar sebelum Guardian bisa exit |
| `activation_atr` | 0.0 | Instant activation (tanpa ATR minimum) |
| `partial_exit_ratio` | 0.50 | Partial exit = tutup 50% posisi |

### 3.4 TP/SL & Risk Management

| Parameter | Nilai |
|-----------|-------|
| `min_rr` | 0.45 |
| `min_tp_atr` | 1.2× ATR |
| `max_sl_atr` | 4.0× ATR |
| `swing_bumper_atr` | 0.5 |
| `tp_atr_mult` (fallback) | 2.0× |
| `sl_atr_mult` (fallback) | 1.5× |
| `max_holding_bars` | 24 H1 |
| `modal_per_trade` | $26 |
| `leverage_recommended` | 5× |
| `fee_per_side` | 0.04% |
| `slippage_per_side` | 0.05% |

### 3.5 Filter Aktif

| Filter | Status | Parameter |
|--------|--------|-----------|
| Trend Alignment | **enabled** | penalty=0.10, boost=0.05 |
| Structural Filter | **enabled** | max_deviation=15%, swing_age=48h |
| VCB (Volatility Circuit Breaker) | **enabled** | ATR mult=3.0, lookback=24 |
| RR Gate | **enabled** | min_rr=0.45 |
| Cooldown | N/A | Dikelola production |

---

## 4. Fitur Utama (87 fitur LGBM + LSTM)

Fitur dikelompokkan berdasarkan kategori:

| Kategori | Contoh Fitur | Jumlah |
|----------|-------------|--------|
| Price & Volume | open, high, low, close, volume, log_ret_1/5/20 | 8 |
| EMA H1 | ema_7/21/50/200_h1 | 4 |
| EMA H4 | ema_7/21/50/200_h4, slopes | 7 |
| Momentum H1 | rsi_6, stochrsi_k/d, vol_ratio_20 | 4 |
| ATR & Volatility | atr_14_h1/h4, atr_zscore_20d, atr_percentile_h1, vol_spike_zscore | 5 |
| Market Structure | BOS, CHoCH, bars_since_BOS, FVG | 5 |
| Liquidity | Buy_Liq, Sell_Liq, SFP_sweep | 3 |
| Order Flow (OFI) | ofi_raw, ofi_accel, ofi_z_score, ofi_h4_delta | 4 |
| Smart Money v3 | cvd, cvd_div_h4, cvd_slope_h4, absorption_z | 5 |
| Smart Money v4 | vwdp, vwdp_smooth, hidden_divergence, VSA (5 feat) | 8 |
| Swing Structure | dist_swing_high/low, price_in_range, swing_momentum | 4 |
| Market Regime | h4_trend, trend_strength, vol_regime | 3 |
| HTF Context | rsi_h4, rsi_slope_h4, atr_percent_h4, range_expansion_h4 | 4 |
| Game Changer v4 | relative_strength_z/momentum, liquidation levels (4), whale_retail_divergence | 7 |
| Macro & Waktu | fear_greed, btc_dominance, market_session, hour/dow sin/cos | 7 |
| Lain-lain | PDH/L, PWH/L, Fib_618/786, POC/VAH/VAL | 9 |
| **Total** | | **87** |

Guardian menggunakan 87 fitur di atas + 7 dynamic features:
`bars_held_norm`, `current_pnl_pct`, `current_pnl_atr`, `max_favorable_pnl_pct`, `drawdown_from_peak_pct`, `direction`, `entry_price_ratio`

---

## 5. Performa Per Koin (Holdout OOS)

| Koin | WR | Trades | Trade/Bln | PnL (5×) |
|------|----|----|----|----|
| ADA | 62.50% | 200 | 40.6 | **+$123.28** |
| NEAR | 58.13% | 375 | 76.1 | **+$122.40** |
| LINK | 55.50% | 400 | 81.2 | **+$100.06** |
| DOGE | 63.92% | 97 | 19.7 | +$79.62 |
| AVAX | 52.63% | 342 | 69.4 | +$67.09 |
| SOL | 53.78% | 331 | 67.2 | +$53.39 |
| SHIB | 62.50% | 104 | 21.1 | +$52.08 |
| XRP | 58.41% | 226 | 45.9 | +$48.44 |
| ETH | 57.14% | 140 | 28.4 | +$44.72 |
| SUI | 53.31% | 347 | 70.4 | +$41.47 |
| PEPE | 63.56% | 118 | 23.9 | +$38.21 |
| DOT | 57.94% | 378 | 76.7 | +$37.36 |
| ONDO | 53.57% | 308 | 62.5 | +$28.64 |
| POL | 54.68% | 278 | 56.4 | +$16.50 |
| ARB | 55.80% | 276 | 56.0 | +$21.25 |
| TRX | 57.01% | 221 | 44.9 | +$8.39 |
| BTC | 58.33% | 36 | 7.3 | +$5.10 |
| HBAR | 52.06% | 194 | 39.4 | −$10.55 |
| BNB | 50.00% | 244 | 49.5 | −$20.90 |
| TAO | 49.64% | 276 | 56.0 | −$52.15 |
| TON | 50.00% | 402 | 81.6 | −$56.26 |
| **TOTAL** | **56.21%** | **5,273** | **1,074/bln** | **+$748.12** |

**Catatan**: TAO dan TON secara konsisten merugi di semua skenario. Pertimbangkan untuk dikecualikan dari universe trading live.

---

## 6. Perbandingan Skenario (Ringkasan Eksperimen)

| Skenario | Deskripsi | WR | Total PnL | Sharpe | PF | DD |
|----------|-----------|----|-----------|---------|----|-----|
| **I (deployed)** | A + dynamic threshold | **56.21%** | **+$748** | **1.68** | **1.26** | 157% |
| A | v2.5hybrid original | 56.19% | +$731 | 1.64 | 1.26 | 158% |
| B | relaxed 0.60/0.55 | 55.02% | +$671 | 1.17 | 1.14 | 199% |
| C | LGBM 0.45, LSTM strict | 52.48% | rugi | −0.66 | 0.96 | 361% |
| D | C + conf=0.65 | 52.69% | rugi | −0.24 | 0.98 | 322% |
| E | Production v3.1 exact | 52.37% | rugi | −0.63 | 0.96 | 352% |
| F | LGBM 0.45, neutral longgar | — | — | — | — | — |
| G | LGBM 0.55, LSTM strict | — | — | — | — | — |
| H | A + LSTM rescue | 52.89% | −$694 | −0.67 | 1.04 | 398% |

---

## 7. Perbedaan dari Versi Sebelumnya (cascade_v4.1)

| Aspek | cascade_v4.1 | cascade_v2.5_hybrid_pruned (ini) |
|-------|-------------|----------------------------------|
| Fitur LGBM | 104 fitur | **87 fitur** (pruned) |
| LGBM threshold LONG | 0.75 | **0.69** |
| LGBM threshold SHORT | 0.60 | **0.59** |
| LSTM opposite penalty | 0.99 (kill) | **0.65** (kurangi) |
| LSTM neutral penalty | 0.99 (kill) | **0.00** (bebas) |
| LSTM architecture | hidden=128 | **hidden=96** |
| Guardian min_hold | 3 | **2** |
| Guardian activation_atr | 1.5 | **0.0** (instant) |
| Trade volume | sangat rendah | **+34% lebih banyak** |

---

## 8. Fitur Tambahan (Belum di Deploy — Next Update)

### Dynamic Threshold Momentum (Scenario I enhancement)
Sudah diimplementasi di `pipeline/backtest_utils.py` tapi belum aktif di production `swint_tradev2/app/services/inference.py`.

Cara kerja: saat `vol_spike_zscore >= 2.0`, LGBM threshold otomatis turun 0.04 (momentum pump/dump detection).

Untuk mengaktifkan di production, tambahkan ke `inference.py`:
```python
# Setelah hitung lgbm_proba, sebelum threshold check:
vol_spike = float(features.get("vol_spike_zscore", 0.0))
momentum_reduce = 0.07 if vol_spike >= 3.0 else (0.04 if vol_spike >= 2.0 else 0.0)
eff_long_thr  = max(0.0, LGBM_THRESHOLD_LONG  - momentum_reduce)
eff_short_thr = max(0.0, LGBM_THRESHOLD_SHORT - momentum_reduce)
```

Dampak terukur di holdout: WR +0.02pp, Total PnL +$17, Sharpe +0.04.

### Price Acceleration Features (Solusi 2 — in progress)
Tiga fitur baru sudah ditambahkan ke `core/features.py` (versi production):
- `price_accel_1h` — 2nd derivative harga
- `ofi_momentum_ratio` — OFI short vs long-term
- `vol_accel_3h` — volume acceleration

LGBM belum diretrain dengan fitur ini. Pipeline: re-run `03_engineer.py` → `04_train_lgbm.py` → `06_train_guardian.py` → holdout.

---

## 9. File yang Di-deploy

| File | Deskripsi | Ukuran (estimasi) |
|------|-----------|-------------------|
| `models/lgbm_baseline.pkl` | LGBM entry, 87 fitur, 781 trees | ~5 MB |
| `models/lstm_best.pt` | LSTM v2-style, hidden=96, 87 fitur | ~0.8 MB |
| `models/lstm_scaler.pkl` | RobustScaler untuk LSTM | ~0.01 MB |
| `models/guardian_best.pkl` | Guardian exit, 94 fitur, 473 trees | ~3 MB |
| `models/guardian_scaler.pkl` | StandardScaler untuk Guardian | ~0.01 MB |
| `models/feature_cols_v2.json` | List 87 fitur | ~3 KB |
| `models/guardian_feature_cols.json` | List 94 fitur Guardian | ~3 KB |
| `models/inference_config.json` | Konfigurasi cascade lengkap | ~3 KB |
| `core/features.py` | Feature engineering (+ 3 fitur baru) | ~60 KB |
| `core/models.py` | TradingLSTM architecture | ~20 KB |

---

## 10. Catatan Operasional

- **Rollback**: Jika performa live buruk, restore dari `backup_20260601_154737`
- **Monitoring**: Pantau WR per koin selama 2 minggu pertama. Flag koin yang WR < 45% secara konsisten (kandidat exclude: TAO, TON, BNB)
- **Guardian**: min_hold_bars=2 berarti Guardian tidak akan exit di 2 bar pertama setelah entry. Ini mencegah exit terlalu dini di momentum awal
- **LSTM feature mismatch**: `backtest_utils.py` sudah di-patch untuk auto-handle jika LGBM retrain dengan lebih banyak fitur (new features di-append di akhir, LSTM auto-truncate)
