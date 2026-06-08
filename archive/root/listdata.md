# listdata.md — Data Live yang Dibutuhkan

*2026-06-05 | Untuk validasi statistik 100+ live trades*

---

## Prinsip

Setiap perubahan model membutuhkan 100+ closed trades untuk validasi statistik bermakna.

```
Dengan WR live ~65% dan n=100:
  95% CI = 65% ± 9.3%  → 55.7% – 74.3% (terlalu lebar)

Dengan n=200:
  95% CI = 65% ± 6.6%

Dengan n=400:
  95% CI = 65% ± 4.7%  ← baru bisa bedakan sistem A vs B yang beda 5%
```

Untuk membuktikan perubahan model memberi **+5% WR**, butuh minimal 300–400 trades per konfigurasi.

---

## A. MENU SIGNAL (Setiap Bar H1, Per Koin — Sebelum Entry)

Dicatat saat sinyal di-generasi, **termasuk yang tidak entry**.
Ini untuk validasi: apakah cascade filter benar menyaring sinyal jelek?

```
signal_id         — unique ID (format: {coin}_{timestamp}_signal, contoh: BTCUSDT_20260605_1400_signal)
timestamp         — waktu bar H1 (contoh: 2026-06-05 14:00)
coin              — simbol (BTCUSDT, SOLUSDT, ...)
close             — harga close bar ini
```

> **signal_id → trade_id linkage**: saat cascade memutuskan ENTRY, trade diberi
> `trade_id` dengan format `{signal_id}_trade` (contoh: `BTCUSDT_20260605_1400_signal_trade`).
> Dengan ini, setiap trade bisa di-trace balik ke signal asalnya — tahu persis
> LGBM prob, LSTM dominant, dan HMM regime saat sinyal pertama kali muncul.

### Output Model Entry

```
lgbm_label        — LGBM output: LONG(2) / FLAT(1) / SHORT(0)
lgbm_long_prob    — probabilitas LONG dari LGBM
lgbm_short_prob   — probabilitas SHORT dari LGBM
lgbm_flat_prob    — probabilitas FLAT dari LGBM
lgbm_confidence   — max(lgbm_long, lgbm_short, lgbm_flat)
```

### LSTM Confirmation

```
lstm_label        — LSTM output: LONG(2) / FLAT(1) / SHORT(0)
lstm_long_prob    — probabilitas LONG dari LSTM
lstm_short_prob   — probabilitas SHORT dari LSTM
lstm_flat_prob    — probabilitas FLAT dari LSTM
lstm_dominant     — max(lstm_long_prob, lstm_short_prob) — FLAT diabaikan
```

### Cascade Decision

```
cascade_mode      — mode aktif (dual_dominant / hard_consensus)
cascade_decision  — final: ENTRY_LONG / ENTRY_SHORT / NO_ENTRY
cascade_conf      — confidence final setelah cascade fusion
```

### Regime & Market Context

```
hmm_regime        — 0=TRENDING_DOWN, 1=RANGING_LOW_VOL, 2=RANGING_HIGH_VOL, 3=TRENDING_UP
h4_trend          — tren H4 (+1 up, -1 down, 0 ranging)
vol_regime        — volume regime (vol / rolling avg vol)
atr_14_h1         — ATR H1 saat ini
```

### Structural Context

```
h4_swing_high     — level swing high H4 terdekat
h4_swing_low      — level swing low H4 terdekat
dist_swing_high   — jarak % ke swing high
dist_swing_low    — jarak % ke swing low
```

---

## B. MENU TRADE (Per Trade, Open → Close)

### B1. Saat Entry (1 baris per trade)

```
trade_id          — unique ID ({signal_id}_trade)
signal_id         — referensi ke signal asal (untuk traceback)
opened            — timestamp entry
coin              — simbol
direction         — LONG / SHORT
entry_price       — harga entry
```

#### Model State Saat Entry

```
lgbm_long_prob    — prob LONG LGBM saat bar entry
lgbm_short_prob   — prob SHORT LGBM saat bar entry
lstm_dominant     — dominant LSTM probability
cascade_conf      — final cascade confidence
```

#### Context Saat Entry

```
hmm_regime        — regime HMM (0-3)
h4_trend          — tren H4
vol_regime        — vol regime
atr_entry         — ATR H1 saat entry
h4_swing_high     — swing H4
h4_swing_low      — swing H4
```

#### TP/SL Structure

```
tp_price          — take profit price
sl_price          — stop loss price
tp_atr_mult       — TP dalam × ATR
sl_atr_mult       — SL dalam × ATR
rr_ratio          — risk/reward ratio
```

#### Position

```
position_size     — $ size posisi (saat ini $25 fixed)
leverage          — 5x
```

---

### B2. Per-Bar Update (Setiap Bar H1 Selama Trade Aktif)

**Ini data paling penting untuk validasi Guardian live vs backtest.**
Menjawab: "Apakah Guardian IC dynamic features (0.33 di backtest) sama kuatnya di live?"

```
trade_id          — referensi ke trade
bar_timestamp     — waktu bar H1 ini
bar_number        — bar ke-berapa sejak entry (1, 2, 3, ...)
close_price       — harga close bar ini
```

#### P&L State (Real-Time)

```
current_pnl_pct   — unrealized P&L (%)
current_pnl_atr   — unrealized P&L dalam × ATR entry
mfe_pct           — max favorable excursion sejauh ini (% dari entry)
drawdown_pct      — drawdown dari MFE saat ini (%)
```

#### Guardian Output (Per Bar)

```
guardian_label         — HOLD(0) / PARTIAL_EXIT(1) / FULL_EXIT(2)
guardian_hold_prob     — probabilitas HOLD
guardian_partial_prob  — probabilitas PARTIAL
guardian_full_prob     — probabilitas FULL
guardian_action        — NO_ACTION / PARTIAL_CLOSE / FULL_CLOSE
```

#### Decision

```
exit_triggered    — True/False apakah bar ini trigger exit
exit_reason       — guardian_exit / sl_hit / tp_hit / time_exit / manual
```

---

### B3. Saat Close (1 Baris Final Per Trade)

```
trade_id          — referensi
closed            — timestamp exit
exit_price        — harga exit
exit_reason       — guardian_exit / sl_hit / tp_hit / time_exit
hold_bars         — total bar dipegang
```

#### P&L Final

```
gross_pnl         — P&L kotor sebelum fee
fee_entry         — biaya entry
fee_exit          — biaya exit
slippage_entry    — slippage entry
slippage_exit     — slippage exit
net_pnl           — P&L bersih ($)
net_pnl_pct       — P&L bersih (%)
```

#### Guardian Behavior

```
partial_exit_done        — apakah partial exit terjadi
partial_exit_bar         — di bar ke berapa
partial_exit_pnl         — P&L saat partial exit
tp_guardian_activated    — apakah TP trigger guardian momentum mode
max_favorable_price      — harga tertinggi/terendah yang pernah dicapai
```

#### Outcome Classification (untuk validasi)

```
is_win            — True jika net_pnl > 0
is_counter_trend  — LONG di h4_trend DOWN, atau SHORT di h4_trend UP
is_ranging_regime — hmm_regime = 1 atau 2
```

---

## C. MENU FEATURE STORE (untuk Retraining — BUKAN di UI, tapi Log/Database)

Ini yang membedakan sistem yang cuma bisa divalidasi vs sistem yang bisa **diretrain ulang**
dari data live. Probabilitas adalah lossy compression — cukup untuk validasi, tidak cukup
untuk training ulang model.

Disimpan ke file terpisah (Parquet atau SQLite), bukan ditampilkan di UI.
Volume data per hari untuk 21 koin: ~500 sinyal × 33 floats + ~10 trade × 40 floats × 8 bar ≈ ~20 KB.
Setahun cuma ~7 MB — sangat murah.

### C1. Feature Snapshot di Signal Bar (untuk Retrain LGBM)

Setiap kali signal di-generasi (termasuk yang di-reject cascade), simpan semua 33 fitur
dari `models/feature_cols_v2.json`:

```
signal_id              — unique ID (link ke signal log)
timestamp              — waktu bar
coin                   — simbol

# Semua 33 fitur aktif LGBM (dari feature_cols_v2.json)
dist_from_8h_high      rsi_6               swing_momentum
rsi_h4                 stochrsi_k          dist_liq_50x_long
trend_accel_4h         rsi_slope_h4        Fib_786
Fib_618                stochrsi_d          ofi_h4_delta
dist_liq_50x_short     Buy_Liq             relative_strength_z
dist_liq_20x_long      cvd_momentum_adv    cvd_slope_h4
ema_21_slope_h4        ema_50_h1           log_ret_20
whale_retail_divergence Sell_Liq           long_short_ratio
hmm_regime_enc         ... (33 total)

# Label (untuk supervised training)
label_ordinal          — SHORT=-1 / FLAT=0 / LONG=+1 (swing label H1)
```

> Kenapa signal yang di-reject juga disimpan? Karena mereka adalah **negative samples**
> untuk melatih cascade filter — "fitur seperti ini → cascade benar menolak"

### C2. LSTM Sequence di Entry Bar (untuk Retrain LSTM)

Saat cascade memutuskan ENTRY, simpan 16 bar sequence × 7 fitur dari
`models/feature_cols_lstm_temporal.json`:

```
trade_id               — referensi ke trade
entry_timestamp        — timestamp bar entry

# 16 bar × 7 fitur (112 floats), oldest → newest
seq_bar_{t-15}_dist_liq_20x_long
seq_bar_{t-15}_rsi_slope_h4
seq_bar_{t-15}_log_ret_20
seq_bar_{t-15}_dist_liq_50x_long
seq_bar_{t-15}_long_short_ratio
seq_bar_{t-15}_cvd_slope_h4
seq_bar_{t-15}_ofi_h4_delta
... (repeat untuk t-14 sampai t-0)

# Label momentum (N=8 bar forward return, arah mayoritas)
momentum_label         — LONG(2) / FLAT(1) / SHORT(0)
```

### C3. Per-Bar Guardian Features Selama Trade (untuk Retrain Guardian)

Setiap bar H1 selama trade aktif, simpan fitur Guardian lengkap:

```
trade_id               — referensi
bar_timestamp          — waktu bar
bar_number             — bar ke-berapa sejak entry

# Dynamic features (7)
bars_held_norm         — bar_number / MAX_HOLDING_BARS
current_pnl_pct        — unrealized P&L %
current_pnl_atr        — unrealized P&L / ATR
max_favorable_pnl_pct  — MFE sejauh ini
drawdown_from_peak_pct — DD dari MFE
direction              — 1=LONG, 0=SHORT
entry_price_ratio      — (close - entry) / ATR

# Static features at this bar (33 dari feature_cols_v2.json)
# — semua 33 fitur yang sama seperti C1, tapi nilainya di bar ini
# — ini kritis: Guardian butuh market context saat memutuskan exit
dist_from_8h_high      rsi_6               swing_momentum
rsi_h4                 stochrsi_k          ... (33 total)

# Label Guardian (apa yang terjadi SETELAH bar ini)
future_best_pnl        — P&L terbaik dari bar ini sampai exit
future_worst_pnl       — P&L terburuk dari bar ini sampai exit
guardian_label         — HOLD(0) / PARTIAL_EXIT(1) / FULL_EXIT(2)
```

---

## D. Gap Analysis — Sudah Ada vs Belum

### Data Validasi (ringan, wajib segera)

| Data | Sudah Ada? | Prioritas |
|------|-----------|-----------|
| trade_id, opened, closed, coin, direction | ✅ | — |
| entry, exit, PnL, exit_reason, hold_bars | ✅ | — |
| h4_trend, vol_regime | ✅ | — |
| **hmm_regime** | ❌ | **HIGH** |
| **lgbm_long/short/flat prob** | ❌ | **HIGH** |
| **lstm_long/short/flat prob** | ❌ | **HIGH** |
| **per-bar guardian prob** | ❌ | **HIGH** |
| **per-bar unrealized P&L (current_pnl_atr, mfe, dd)** | ❌ | MEDIUM |
| **rr_ratio saat entry** | ❌ | LOW |

### Data Retraining (feature vector, mulai kumpulin dari hari pertama)

| Data | Sudah Ada? | Prioritas | Untuk Retrain |
|------|-----------|-----------|---------------|
| **Feature snapshot di signal bar (33 feat)** | ❌ | **HIGH** | LGBM |
| **LSTM sequence di entry (16×7)** | ❌ | **HIGH** | LSTM |
| **Per-bar Guardian features (33+7)** | ❌ | **HIGH** | Guardian |
| **Label ordinal per signal** | ❌ | MEDIUM | LGBM target |
| **Label momentum per entry** | ❌ | MEDIUM | LSTM target |
| **Guardian label per bar (HOLD/PARTIAL/FULL)** | ❌ | MEDIUM | Guardian target |
| **Delta features (5) — market change since entry** | ❌ | LOW | Guardian ext |
| **Feature distributions随时间 (drift detection)** | ❌ | LOW | IC decay monitoring |

---

## E. Minimum Viable — Mulai Dengan Ini

### Tier 1 — Validasi (4 kolom, bisa langsung)

Tambahkan ke trade entry:

```
1. hmm_regime          — di trade entry
2. lgbm_long_prob      — di trade entry (ganti single "Conf" jadi 3 prob)
3. lgbm_short_prob     — di trade entry
4. lstm_dominant       — di trade entry
```

Dengan ini, setelah 100 trade closed bisa langsung:

| Analisis | Data yang Dipakai |
|----------|-------------------|
| WR per HMM regime | `hmm_regime` + `net_pnl` |
| Confidence calibration | `lgbm_*_prob` bucket vs `is_win` |
| LSTM contribution | `lstm_dominant` agree vs disagree vs `is_win` |
| Counter-trend risk | `is_counter_trend` vs `is_win` |

### Tier 2 — Feature Store (mulai kumpulin dari hari pertama)

Ini simpan ke file terpisah (`.parquet`), bukan di UI. Volume: ~20 KB/hari, ~7 MB/tahun.

```
1. Signal features (33 feat × semua signal)       → untuk retrain LGBM nanti
2. LSTM sequence (16×7 feat × setiap entry)       → untuk retrain LSTM nanti
3. Guardian bar features (33+7 feat × setiap bar)  → untuk retrain Guardian nanti
```

Tanpa Tier 2, 400 trade kemudian kamu mau retrain — tapi cuma punya probabilitas.
Prob tidak bisa dipakai training. Harus mulai kumpulin dari nol lagi. Rugi waktu 3-6 bulan.

Per-bar Guardian data (B2) di UI bisa ditambahkan setelah sistem stabil 1–2 minggu.

---

## F. Analisis yang Bisa Dilakukan Setelah Data Terkumpul

### Setelah 50 trades

```
- WR kasar (CI ±13pp — masih noise)
- Deteksi anomali besar: apakah ada koin dengan WR 0%?
- Apakah cascade memblokir terlalu banyak sinyal?
```

### Setelah 100 trades

```
- WR per hmm_regime — regime mana yang profit, mana yang loss
- WR saat lstm_dominant > 0.5 vs < 0.5
- Confidence calibration: apakah conf 0.9 benar WR > conf 0.6?
- Exit reason breakdown: guardian_exit vs sl_hit WR
```

### Setelah 200 trades — Analisis Lanjutan

```
- Guardian IC test live (butuh per-bar data B2)
- Per-koin WR — koin mana yang underperform?
- Regime-conditional ensemble weight validasi
- Cek feature drift: bandingkan distribusi fitur live vs training
```

### Setelah 400 trades — Signifikan Statistik

```
- Bisa bandingkan sistem A vs B dengan CI < 5%
- Kelly Criterion position sizing dikalibrasi dari data live
- Deteksi sinyal yang melemah: IC decay di data live vs training
```

### Setelah 1,000+ trades — Retraining Penuh

```
- Retrain LGBM dengan feature store C1 (33 feat × semua signal)
- Retrain LSTM dengan sequence store C2 (16×7 feat × setiap entry)
- Retrain Guardian dengan bar store C3 (33+7 feat × setiap bar in-trade)
- Meta-labeling walk-forward OOF (pakai signal features + actual outcome)
- Full Simon pipeline diulang dari data live — bukan dari simulation
```

---

## G. Target Metrik Live (dari Holdout Baseline)

| Metrik | Holdout (target) | Acceptable Range |
|--------|-----------------|-----------------|
| Overall WR | 69% | 60–75% |
| guardian_exit WR | 65%+ | 55%+ |
| sl_hit rate | < 15% | < 25% |
| Trades/hari (21 koin) | 8–15 | 5–20 |
| Avg hold bars | 6–10 | 4–14 |

Jika metrik live di luar acceptable range secara konsisten → investigasi sebelum lanjut modifikasi model.
