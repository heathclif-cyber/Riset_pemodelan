# Rencana Eksperimen: Cascade Sweep

*2026-06-06 | Validasi konfigurasi cascade fusion di holdout bersih*

---

## Latar Belakang

Konfigurasi cascade saat ini (V2.5 Hybrid) adalah **emergency fix** dari live under-trading
Mei 2026, bukan hasil eksperimen formal. Parameter dipilih dari observasi live + tuning
holdout yang sudah dicabut (data leakage 2026-06-04).

**Tujuan**: menemukan konfigurasi cascade yang tervalidasi di holdout bersih
(Nov 2025 – Apr 2026) dengan model ic32 yang baru.

---

## Parameter yang Di-sweep

### A. CASCADE MODE (3 mode utama)

| Mode | Cara Kerja | Kekuatan | Kelemahan |
|------|-----------|----------|-----------|
| **hard_consensus** | LGBM std threshold → LSTM soft adjust (agree boost / opposite penalty) → trend alignment | Sederhana, WR tinggi (78.8% di eksperimen lama) | LSTM FLAT tidak berkontribusi |
| **dual_dominant** | LGBM argmax ≥ gate → LSTM max(L,S) ≥ threshold → harus searah | WR tertinggi di Z3 (64.81%), dual gate ketat | Trade count rendah, gate mungkin terlalu ketat |
| **lstm_dominant** | LGBM std threshold → LSTM dominant(L,S) ≥ threshold → harus searah | LSTM FLAT diabaikan, lebih banyak sinyal dari dual_dominant | WR lebih rendah dari dual_dominant (62.58%) |

### B. THRESHOLD (per mode)

#### hard_consensus
```
opposite_penalty  : [0.35, 0.50, 0.65, 0.80]
threshold_long    : [0.65, 0.69, 0.72, 0.75]
threshold_short   : [0.55, 0.59, 0.62, 0.65]
```

#### dual_dominant
```
lgbm_gate          : [0.55, 0.60, 0.65, 0.70]
lstm_dominant_thr  : [0.30, 0.33, 0.35, 0.38]
```

#### lstm_dominant
```
threshold_long     : [0.65, 0.69, 0.72]
threshold_short    : [0.55, 0.59, 0.62]
lstm_dominant_thr  : [0.30, 0.33, 0.35, 0.38]
```

### C. TREND ALIGNMENT (semua mode)

```
trend_alignment_enabled : [True, False]
with_trend_penalty      : [0.05, 0.10]    (hanya jika enabled)
counter_trend_boost     : [0.03, 0.05]    (hanya jika enabled)
```

### D. LSTM DIRECTIONAL REVIEW (hard_consensus only)

```
lstm_directional_review : [True, False]
directional_threshold   : [0.30, 0.35]    (hanya jika True)
```

---

## Desain Eksperimen

### Fase 1 — 5 Koin, Narrowing (estimasi 30-60 menit)

Tujuan: eliminasi kombinasi yang jelas buruk.

```
Koin: SOLUSDT, ETHUSDT, BNBUSDT, DOGEUSDT, BTCUSDT
Periode: Nov 2025 – Apr 2026
Modal: $25/trade, 5x leverage

Total kombinasi: terlalu banyak untuk brute force (~3 mode × 4×4 thr × 2 trend × ...)
→ Gunakan narrowing bertahap:
```

#### Step 1.1 — Mode Baseline (4 run)
```
Run 1: hard_consensus, opp=0.65, thr 0.69/0.59, trend=ON  (V2.5 Hybrid — current)
Run 2: hard_consensus, opp=0.65, thr 0.69/0.59, trend=OFF (tanpa trend alignment)
Run 3: dual_dominant, gate=0.65, lstm_dom=0.35             (Z3 reference)
Run 4: lstm_dominant, thr 0.69/0.59, lstm_dom=0.35         (Y1 reference)

→ Pilih 2 mode terbaik berdasarkan WR × trade_count (bukan PnL saja)
```

#### Step 1.2 — Threshold Sweep per Mode Terpilih (8 run per mode)
```
hard_consensus:
  Sweep opp=[0.35, 0.50, 0.65, 0.80] × thr_long=[0.65, 0.69]
  
dual_dominant:
  Sweep gate=[0.55, 0.60, 0.65, 0.70] × lstm_dom=[0.30, 0.33, 0.35, 0.38]
```

#### Step 1.3 — Trend Alignment ON/OFF untuk top 3 kombinasi
```
6 run: 3 kombinasi terbaik × trend=[ON, OFF]
```

### Fase 2 — 21 Koin, Validasi (estimasi 15-30 menit)

```
Koin: semua 21 koin
Periode: Nov 2025 – Apr 2026
Kombinasi: top 3 dari Fase 1
```

### Fase 3 — Decision Matrix

Dari hasil Fase 2, pilih 1 konfigurasi final berdasarkan:

```
Skor = (WR - 0.60) × 0.4 + (Sharpe / 5) × 0.3 + (trades_per_bulan × 0.5) × 0.2 + (LONG_WR - SHORT_WR_abs) × -0.1

Prioritas:
  1. WR ≥ 62% (minimum viable)
  2. Trade count ≥ 100/bulan/21koin (statistical significance)
  3. LONG:SHORT ratio antara 0.5:1 sampai 2:1 (tidak bias)
  4. DD ≤ 75% (risk tolerance)
```

---

## Metrik yang Dicatat per Run

```
wr_overall        — win rate keseluruhan
wr_long           — win rate LONG trades
wr_short          — win rate SHORT trades
total_trades      — jumlah trade
trades_per_bulan  — rata-rata per bulan
net_pnl           — total PnL ($)
profit_factor     — gross_win / gross_loss
sharpe_ratio      — dari daily equity curve
max_drawdown_pct  — max drawdown %
avg_hold_bars     — rata-rata hold time
guardian_exit_wr  — WR saat guardian exit
sl_hit_wr         — WR saat SL hit (harus ≈ 0)
sl_hit_pct        — % trade yang kena SL
long_pct          — % LONG trades
```

---

## Script

### File: `experiments/cascade_sweep.py`

```python
"""
experiments/cascade_sweep.py — Cascade fusion parameter sweep
Holdout bersih Nov 2025 – Apr 2026, 5 koin narrowing → 21 koin validasi

Usage:
  python experiments/cascade_sweep.py --phase 1 --coins 5    # narrowing
  python experiments/cascade_sweep.py --phase 2 --coins 21   # validation
"""
```

Script akan:
1. Load model ic32 yang sama untuk semua run (LGBM + LSTM + Guardian tidak berubah)
2. Variasikan HANYA parameter cascade fusion (`hierarchical_predict` args)
3. Simpan hasil per run ke `experiments/cascade_sweep_results.json`
4. Generate summary markdown

---

## Catatan Penting

### JANGAN:
- ❌ Jangan retrain model apapun — ini pure inference sweep
- ❌ Jangan ubah Guardian params — Guardian tetap clean_v2
- ❌ Jangan pakai data selain holdout (Nov 2025 – Apr 2026)
- ❌ Jangan pilih parameter dari holdout lalu klaim sebagai OOS — ini tuning,
     bukan genuine OOS. Hasil ini untuk memilih konfigurasi TERBAIK di holdout,
     tapi metriknya harus divalidasi ulang di live trading.

### Kenapa ini valid (meski tuning di holdout):
- Holdout belum pernah dipakai untuk sweep cascade fusion sebelumnya
  (sweep 2026-05-27 pakai model cascade_v4.1 yang berbeda — leakage berasal
   dari holdout split, bukan dari parameter tuning di holdout)
- Model ic32 dilatih 2020-2025, holdout 2025-2026 adalah genuine unseen
- Kita memilih konfigurasi cascade (bukan melatih model) — ini hyperparameter
  tuning yang lebih ringan dari training

### Ekspektasi:
- hard_consensus: WR tertinggi, trade count moderate
- dual_dominant: WR moderate-tinggi, trade count rendah
- lstm_dominant: WR moderate, trade count tertinggi
- Trend alignment: small improvement, mungkin noise
- Tidak ada kombinasi yang akan menghasilkan WR 88% — itu artefak leakage lama
