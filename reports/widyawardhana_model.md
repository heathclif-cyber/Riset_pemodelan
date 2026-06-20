# Widyawardhana — Model Meta Aktif

> **Golden Benchmark** — file ini adalah standar yang harus dilampaui oleh eksperimen baru.
> Setiap model kandidat dibandingkan dengan scorecard di file ini sebelum dinyatakan "lebih baik".
>
> **Cara update**: arsip versi lama ke `reports/experiments/YYYY-MM-DD_widyawardhana_vN.md`,
> lalu tulis versi baru di sini. Update HANYA jika model baru lebih baik di SEMUA kriteria:
> WR >= saat ini, PF >= saat ini, trades >= 80% saat ini, metodologi genuine OOF.
>
> **Jangan update file ini berdasarkan hasil in-sample atau OOF saja** —
> harus ada konfirmasi holdout genuinee (lihat `CLAUDE.md` § Alur Kerja Eksperimen Tahap 4).

---

**Versi**: v2  
**Ditetapkan**: 2026-06-14  
**Run ID**: tb_widyawardhana_v2 (gabungan: flatboost_v2 + HMM T50_R55 + Guardian profit_v1)  
**Holdout**: Apr 1 – Jun 13, 2026 (~2.5 bulan, 21 koin)  
**Benchmark**: ic32_regime_v1 pada periode sama (936 trades | WR 62.1% | PF 2.54 | +$207)

---

## Arsitektur Stack

```
[Bar baru]
    |
    v
[1] LGBM flatboost_v2
    Triple Barrier labels (TP=2xATR, SL=1.5xATR, MaxHold=36)
    27 fitur -> prediksi LONG/SHORT/FLAT + confidence
    |
    v
[2] HMM Adaptive Threshold
    4 state: TRENDING_DOWN=0, RANGING_LOW=1, RANGING_HIGH=2, TRENDING_UP=3
    TRENDING (0,3): thr_long=0.50, thr_short=0.55
    RANGING  (1,2): thr_long=0.55, thr_short=0.60
    |
    v
[3] Entry Filter
    confidence >= threshold? -> lanjut
    |
    v (jika TRENDING)
[4] LSTM Soft Veto
    Aktif hanya di bar TRENDING
    Jika LSTM conf arah berlawanan >= 0.50 -> veto masuk
    |
    v
[5] OPEN TRADE
    |
    v (setiap bar)
[6] Guardian profit_v1
    25 fitur (18 static + 7 dynamic)
    Labeling: profit-only, tidak ada loss-based EXIT
    exit_thr=0.70, min_hold=3 bar
    Output: HOLD / PARTIAL / EXIT
```

---

## Model Files

| Komponen | File | Ukuran |
|----------|------|--------|
| Entry LGBM | `models/runs/tb_lgbm_flatboost_v2/lgbm.pkl` | — |
| LSTM | `models/lstm_best.pt` | — |
| LSTM Scaler | `models/lstm_scaler.pkl` | — |
| Guardian | `models/runs/tb_guardian_profit_v1/guardian.pkl` | — |
| Guardian Scaler | `models/runs/tb_guardian_profit_v1/guardian_scaler.pkl` | — |
| HMM | `models/hmm_regime_model.pkl` | — |

---

## Fitur LGBM flatboost_v2 (27 fitur)

**Kategori: Liquidity & Distance (6)**
```
dist_liq_50x_long, dist_liq_50x_short, dist_liq_20x_short,
dist_from_8h_high, dist_swing_high, VAH
```

**Kategori: CVD & Order Flow (4)**
```
cvd_slope_h4, ofi_h4_delta, cvd_momentum_adv, ofi_h4_delta
```

**Kategori: Momentum & Trend (5)**
```
trend_accel_4h, stochrsi_d, log_ret_20, atr_percent_h4, atr_percentile_h1
```

**Kategori: Volume Analysis (8)**
```
whale_retail_divergence, Buy_Liq, vol_spike_zscore, range_expansion_h4,
ultra_high_vol, absorption_z, vol_accel_3h, vol_ratio_20
```

**Kategori: Wyckoff / Supply-Demand (3)**
```
no_supply, no_demand, effort_vs_result
```

**Kategori: Macro / Waktu (2)**
```
dow_cos, funding_rate
```

Training: 8-fold Purged CV, Triple Barrier labels, 21 koin, 2020-2025-10

---

## Fitur LSTM (11 temporal features)

Dari `models/feature_cols_lstm_temporal.json` — fitur H1 temporal sequence (seq_len=32).  
Peran: soft veto di bar TRENDING saja (bukan hard entry signal).

---

## Fitur Guardian profit_v1 (25 fitur)

### Static (18) — Market Context per bar
```
etf_gbtc_change_usd    -- ETF BTC inflow/outflow
etf_total_change_usd   -- Total ETF flow
cvd_slope_h4           -- Arah tekanan beli/jual H4
ofi_h4_delta           -- Order flow imbalance delta H4
wyckoff_phase          -- Fase Wyckoff
Sell_Liq               -- Likuiditas sisi jual
atr_percentile_h1      -- Volatilitas relatif (rank)
stochrsi_k             -- Stochastic RSI K
dist_liq_50x_short     -- Jarak ke likuidasi short massal
funding_rate           -- Tekanan long/short futures
ema_7_h1               -- EMA cepat H1
dow_cos                -- Hari dalam seminggu
cvd_div_h4             -- Divergensi CVD H4
dist_swing_low         -- Jarak dari swing low
VAH                    -- Value Area High
cvd_momentum_adv       -- CVD momentum advanced
dist_from_8h_high      -- Jarak dari high 8 jam
ema_200_h1             -- EMA lambat H1 (trend makro)
```

### Dynamic (7) — Trade Progress per bar
```
bars_held_norm         -- Seberapa lama sudah hold (0=baru, 1=hampir time exit)
current_pnl_pct        -- PnL saat ini dalam %
current_pnl_atr        -- PnL saat ini dalam satuan ATR
max_favorable_pnl_pct  -- MFE: PnL tertinggi yang pernah dicapai
drawdown_from_peak_pct -- Turun berapa % dari puncak MFE
direction              -- Arah trade (1=LONG, 0=SHORT)
entry_price_ratio      -- Harga sekarang / harga entry
```

### Labeling Guardian (profit-only)
```
Rule 1: bars_held < 3           -> HOLD  (terlalu dini)
Rule 2: mfe>2%, pos, balik 60%  -> EXIT  (profit-lock) [BARU]
Rule 3: mfe>1.5%, balik 75%     -> EXIT  (hampir habis)
Rule 4: at-peak & pnl>0.5%     -> EXIT  (waktu terbaik, look-ahead)
Rule 5: mfe>1.5%, balik 45%     -> PARTIAL
Rule 6: pnl>0.8%, upside<3%    -> PARTIAL
Rule 7: future > current*1.05   -> HOLD  (masih ada potensi)
HAPUS: current_pnl < -1 ATR -> EXIT  (bug lama, exit saat rugi)
```
CV F1 macro: 0.8509 (8-fold purged)  
EXIT-2 saat PnL positif: 78% (vs bug lama: banyak negatif)

---

## HMM Adaptive Threshold

**Konfigurasi T50_R55:**
```
TRENDING_DOWN (0) + TRENDING_UP (3):
    thr_long  = 0.50
    thr_short = 0.55

RANGING_LOW (1) + RANGING_HIGH (2):
    thr_long  = 0.55
    thr_short = 0.60
```

**Distribusi holdout Apr-Jun 2026:**
```
TRENDING_DOWN : ~20%
RANGING_LOW   : ~54%
RANGING_HIGH  : ~20%
TRENDING_UP   :  ~6%
```

**LSTM Veto (hanya TRENDING bars):**
```
opp_conf = LSTM confidence arah berlawanan
if opp_conf >= 0.50:
    adjusted = entry_conf - opp_conf * (entry_conf - thr_l + 0.05)
    if adjusted < thr_l: VETO (skip entry)
```

---

## Parameter Config

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

---

## Scorecard Holdout — Apr 1 – Jun 13, 2026 (2.5 bulan, 21 koin)

| Metrik | Widyawardhana v2 | ic32 (same period) | Delta |
|--------|:----------------:|:------------------:|:-----:|
| **Total Trades** | **905** | 936 | -31 |
| Trades/bulan | 362 | 374 | — |
| **Win Rate** | **68.2%** | 62.1% | **+6.1%** |
| **Profit Factor** | **2.79** | 2.54 | **+0.25** |
| **Net PnL ($10/trade, 5x)** | **+$301** | +$207 | **+$94 (+45%)** |
| PnL/bulan | +$120 | +$83 | — |
| **PnL/trade** | **+$0.332** | +$0.221 | **+$0.111** |
| Guardian Exit % | 65.1% | — | — |
| SL Hit Rate | 0.0% | — | — |
| Time Exit | (via Guardian) | — | — |

> Guardian profit_v1 menambahkan +$20 vs tanpa Guardian (base no_gd = +$281).
> SL 0% karena Guardian selalu exit sebelum SL tercapai.

---

## Sweep Summary

### HMM Threshold Sweep (42 combo + fixed)
Best total PnL: T42_R50 (+$545) — tapi Guardian mengurangi PnL  
Best balanced (Guardian membantu + PF tinggi): **T50_R55** (+$301, PF 2.79)  
Full hasil: `models/runs/tb_lgbm_flatboost_v2/hmm_guardian_sweep.json`

### Guardian Variants Comparison (pada T50_R55)
| Variant | Fitur | WR | PF | PnL |
|---------|-------|----:|----:|----:|
| profit_v1 | 18s+7d | 68.2% | **2.79** | **+$301** |
| fb_union_v1 | 36s+7d | 69.1% | 2.77 | +$295 |
| fb_v1 | 27s+7d | 68.2% | 2.73 | +$292 |
| no_guardian | — | 69.6% | 2.43 | +$281 |

Full hasil: `models/runs/tb_lgbm_flatboost_v2/guardian_variants_eval.json`

---

## Catatan & Temuan Kunci

1. **Guardian bug lama (clean_v2)**: Rule `current_pnl < -1 ATR -> EXIT` mengajarkan model exit saat rugi. Fix: hapus rule ini, ganti dengan profit-lock.

2. **HMM insight**: thr_R=0.50 (semua RANGING masuk) memberi volume tertinggi tapi PF rendah. Sweet spot ada di thr_R=0.55 di mana selektivitas cukup untuk PF meningkat.

3. **Guardian paling efektif di config medium volume** (900-1000 trades). Pada config high-volume (>2000 trades), Guardian malah memotong winners prematur.

4. **fb_v1 / fb_union_v1 lebih lemah dari profit_v1**: Fitur entry LGBM (volume analysis, wave) kurang relevan untuk keputusan exit. Fitur market context (ETF flow, EMA 200, Wyckoff) lebih informatif untuk exit timing.

5. **Meta-labeling entry (`tb_meta_fb_v2`) — CLOSED 2026-06-15**: Binary LGBM take/skip di atas flatboost_v2 + HMM gagal Simon Gate #1 di holdout (marginal IC t=0.9). Eksplorasi 3 varian fitur + soft multiplier tidak beat `primary_hmm` PnL (+$276). **Tidak deploy.** Detail: `EXPERIMENTS.md` §11.

---

## Meta-Labeling — Status & Arah Lain

| Eksperimen | Status | Kesimpulan singkat |
|------------|--------|-------------------|
| `tb_meta_v1` (LGBM gate) | ❌ Closed | PnL < baseline, volume loss |
| `tb_lstm_binary_meta_v1` | ❌ Closed | AUC 0.56, corr LGBM 80% |
| `tb_meta_fb_v2` (flatboost_v2) | ❌ **Closed 2026-06-15** | Gate #1 FAIL holdout; best meta arm -$4 vs baseline |
| Hard gate / soft multiplier | ❌ Ditolak | PF naik, PnL turun |

**Fokus riset pengganti meta entry:**

1. **Guardian exit** — model aktif `continuation_v1` (19s+10d); iterasi labeling & fitur flow since-entry
2. **LGBM primary** — fitur baru (positioning, macro temporal) via walk-forward IC gate
3. **LSTM macro** — `tb_lstm_macro_v1` (7 fitur IC-validated, bukan duplikasi arah LGBM)
4. **Positioning pipeline** — data sudah di-fetch di repo produksi; training saat history cukup

Stack entry produksi **tetap**: flatboost_v2 → HMM T50_R55 → LSTM soft veto → Guardian continuation_v1. Tidak menambah layer meta gate.
