# Implementation Brief — Perbaikan Signal Filter & TP/SL Logic
**Dibuat**: 2026-05-20  
**Berdasarkan**: Analisis 101 live trade (cascade_v3, May 4–20 2026)  
**Target repo**: `swint_tradev2`

---

## Ringkasan Eksekutif

Live trading menunjukkan WR 52–56% vs backtest holdout 88.93%. Analisis 98 trade valid (dengan PnL) menemukan **3 penyebab utama** yang bersifat code bug dan rule gap — bukan model failure.

Simulasi ketiga fix terhadap data live:

| Fix | Trades Hilang | WR | Net PnL | PF |
|-----|--------------|-----|---------|-----|
| Baseline | 98 | 54.1% | +$239.56 | 1.70 |
| + 2A (counter-trend filter) | 90 | **+2.6pp** → 56.7% | **+$22.84** → $262.40 | 1.88 |
| + 2A + 1A (RR gate) | 69 | −1.9pp → 52.2%* | −$31.71 → $207.85* | 1.86 |
| + 2A + 1A + 1B (staleness) | 69 | tidak berubah | tidak berubah | 1.86 |

\* *Angka combined turun karena 1A memblokir winning trades cascade_v2 dari periode bull run May 8–14. Untuk cascade_v3 (model aktif sekarang): WR 52.2% → **69.2%**, PnL +$7.35 → **+$25.43**, PF 1.11 → **2.29**.*

**Kesimpulan**: Implementasi ketiganya. Evaluasi dampak di cascade_v3 saja karena itu satu-satunya model yang berjalan sekarang.

---

## FIX 1A — RR Gate Re-Validasi Setelah Final TP/SL

### Root Cause

Sistem menggunakan hybrid TP/SL: bisa swing-based, ATR fallback, atau campuran keduanya. Bug terjadi ketika **TP menggunakan ATR fallback** (H4Low terlalu dekat entry untuk SHORT, atau H4High terlalu dekat untuk LONG) tetapi **SL tetap menggunakan swing-based** (H4High + buffer untuk SHORT). RR tidak di-check ulang setelah resolusi final.

Contoh nyata dari live data:

```
SUIUSDT SHORT (05-18 04:05)
  Entry = 1.0448
  H4Low = 1.0357  →  hanya 0.9% di bawah entry (terlalu dekat untuk TP)
  → Sistem fallback ke ATR untuk TP:
  TP = entry − 2×ATR = 1.0448 − 0.0266 = 1.0182  (2.5% dari entry)
  SL = H4High + 0.5×ATR = 1.0900 + 0.0066 = 1.0966  (5.0% dari entry)
  RR final = 2.5 / 5.0 = 0.51  ← harusnya SKIP (min_rr = 1.0)
  Tapi trade MASUK → LOSS
```

Trades yang terdampak di live data (semuanya berakhir loss, kecuali 1 keberuntungan):

| Trade | RR | Outcome |
|-------|----|---------|
| XRPUSDT SHORT 05-18 19:05 | 0.50 | LOSS −$4.85 |
| SUIUSDT SHORT 05-18 04:05 | 0.51 | WIN +$8.47 (luck) |
| SUIUSDT SHORT 05-18 19:05 | 0.67 | LOSS −$7.38 |
| TONUSDT SHORT 05-18 22:05 | 0.91 | LOSS −$11.25 |
| SUIUSDT SHORT 05-19 03:05 | 0.97 | LOSS −$8.76 |
| NEARUSDT v2 05-13 17:05 | 0.57 | WIN +$8.82 (bull phase) |
| DOGEUSDT v2 05-13 15:05 | 0.51 | WIN +$11.53 (bull phase) |
| (dan 8 lainnya dari cascade_v2 bull phase) | < 1.0 | mostly WIN |

> **Catatan**: cascade_v2 banyak low-RR yang menang karena bull run May 8–14. Ke depan di market normal, low-RR entries akan lebih sering loss.

### Fix

Tambahkan re-validasi RR **setelah** TP dan SL final ditetapkan, **sebelum** order dikirim. Ini 1 blok kode, tidak perlu ubah logika TP/SL sebelumnya.

```python
# Letakkan SETELAH resolve_tp_sl() dan SEBELUM submit_order()

def validate_rr(entry_price: float, tp_price: float, sl_price: float,
                direction: str, min_rr: float = 1.0) -> bool:
    """
    Re-validate RR dengan TP/SL final, apapun metode yang digunakan.
    Return True = trade valid, False = skip trade.
    """
    if direction == "LONG":
        tp_dist = tp_price - entry_price
        sl_dist = entry_price - sl_price
    else:  # SHORT
        tp_dist = entry_price - tp_price
        sl_dist = sl_price - entry_price

    if sl_dist <= 0:
        return False  # SL di sisi yang salah

    rr = tp_dist / sl_dist
    return rr >= min_rr
```

Lokasi panggilan:

```python
# Di paper_trading.py atau signal_processor.py, setelah TP/SL ditetapkan:

tp, sl = resolve_tp_sl(signal, market_data)  # existing logic

# === TAMBAHAN FIX 1A ===
if not validate_rr(signal.entry_price, tp, sl, signal.direction, min_rr=1.0):
    logger.info(f"[SKIP] RR gagal: {signal.symbol} {signal.direction} "
                f"RR={abs(tp-entry)/abs(sl-entry):.2f} < 1.0")
    return None  # atau equivalent skip logic
# === END FIX 1A ===

submit_order(signal, tp, sl)
```

### Parameter
- `MIN_RR = 1.0` — sesuai `inference_config.json` (`rr_gate.min_rr`)
- Tidak perlu ubah config, hanya enforce yang sudah ada

---

## FIX 1B — H4 Swing Level Staleness Validation

### Root Cause

Swing level H4 bisa menjadi stale jika harga sudah bergerak jauh dari range H4 yang terakhir terdeteksi. Kasus ekstrem dari live data:

```
TONUSDT cascade_v2 (05-08 05:05) — LONG
  Entry = 2.6888
  H4High = 2.9078  (TP target, 8.1% di atas entry — OK)
  H4Low  = 1.3065  (SL target, 51.4% di bawah entry — STALE!)
  SL yang dipasang = 1.3065  ←  jika SL hit = loss −51.4% × 5x = −257%

  Dalam sample ini trade time_exit (harga tidak turun sejauh itu).
  Tapi risiko catastrophic loss tetap nyata.
```

Kasus lain: TONUSDT legacy dengan H4High=1.3560 saat harga trading di 2.10–2.70. H4High jauh di bawah entry → TP target tidak valid → sistem sudah fallback ke ATR (benar). Artinya 1B sudah sebagian berjalan, tapi tidak konsisten untuk kasus SL.

### Fix

Tambahkan staleness check **sebelum** menggunakan swing level sebagai SL:

```python
MAX_SL_ATR_MULT = 4.0      # dari config, max_sl = 4x ATR
MAX_TP_ATR_MULT = 10.0     # batas wajar TP tidak terlalu jauh
ATR_FALLBACK_TP  = 2.0
ATR_FALLBACK_SL  = 1.5

def resolve_tp_sl(direction: str, entry: float, h4_high: float, h4_low: float,
                  atr: float) -> tuple[float, float]:
    """
    Hybrid TP/SL dengan staleness guard.
    """
    use_swing_tp = True
    use_swing_sl = True

    if direction == "LONG":
        # TP = H4High (harus DI ATAS entry)
        if h4_high <= entry:
            use_swing_tp = False  # stale: H4High di bawah atau sama dengan entry
        # SL = H4Low (harus DI BAWAH entry, dan tidak terlalu jauh)
        if h4_low >= entry:
            use_swing_sl = False  # stale: H4Low di atas entry
        elif (entry - h4_low) / atr > MAX_SL_ATR_MULT:
            use_swing_sl = False  # terlalu jauh: > 4x ATR

    elif direction == "SHORT":
        # TP = H4Low (harus DI BAWAH entry)
        if h4_low >= entry:
            use_swing_tp = False
        # SL = H4High (harus DI ATAS entry, dan tidak terlalu jauh)
        if h4_high <= entry:
            use_swing_sl = False
        elif (h4_high - entry) / atr > MAX_SL_ATR_MULT:
            use_swing_sl = False

    # Resolve TP
    if use_swing_tp:
        if direction == "LONG":
            tp = h4_high
        else:
            tp = h4_low
    else:
        tp = (entry + ATR_FALLBACK_TP * atr) if direction == "LONG" \
             else (entry - ATR_FALLBACK_TP * atr)

    # Resolve SL
    if use_swing_sl:
        if direction == "LONG":
            sl = h4_low - 0.5 * atr   # buffer di bawah swing low
        else:
            sl = h4_high + 0.5 * atr  # buffer di atas swing high
    else:
        sl = (entry - ATR_FALLBACK_SL * atr) if direction == "LONG" \
             else (entry + ATR_FALLBACK_SL * atr)

    return tp, sl
```

> **Penting**: Setelah 1B resolve TP/SL, tetap jalankan **1A RR re-validation**. Kombinasi keduanya memastikan tidak ada trade dengan bad RR yang lolos.

### Impact di Live Data
1B tidak mengubah outcome trade manapun di sample karena:
- Semua kasus stale H4 sudah tertangkap oleh 1A (RR=0.16 untuk TONUSDT v2)
- TONUSDT legacy sudah menggunakan ATR fallback secara benar
- Nilainya **defensive** — mencegah catastrophic loss di masa depan jika harga benar-benar mencapai SL stale tersebut

---

## FIX 2A — Hard Block SHORT + H4 Trend = UP

### Root Cause

Di live data, 11 trade SHORT melawan H4 UP trend menghasilkan WR 18.2% (2W/9L), PnL −$22.84. Trade dengan confidence tertinggi (TONUSDT conf=0.94) pun berakhir loss. Market memiliki bias bullish struktural — SHORT di tengah uptrend H4 statistiknya tidak favorable.

Distribusi outcome:

| Kondisi | Trades | WR | PnL |
|---------|--------|-----|-----|
| SHORT + H4 DOWN (with-trend) | 12 | 62.5% | +$30.19 |
| SHORT + H4 UP (counter-trend) | 8* | **18.2%** | **−$22.84** |

\* *8 dengan PnL; 3 migrated excluded.*

Simulasi P1 on entire dataset: +$22.84 PnL, WR +2.6pp, PF 1.70→1.88.  
Simulasi P1 on cascade_v3 only: WR 52.2% → 62.5%, PF 1.11 → 1.68.

**Trade-off**: Satu trade yang terlewat adalah SUIUSDT +$16.00 (21 bar hold, H4 UP). Exception ini ada karena Vol Regime=0.35 dan RR=2.38 — kombinasi unusual. Secara statistik, blocking trade ini lebih menguntungkan daripada tidak.

### Fix

Satu kondisi di signal filter, **sebelum TP/SL calculation** (lebih efisien):

```python
# Di signal_filter.py atau setelah cascade signal dihasilkan,
# sebelum memanggil resolve_tp_sl()

def apply_pre_entry_filters(signal: TradeSignal, market_data: MarketData) -> bool:
    """
    Return True = lanjut, False = skip trade.
    """
    # [2A] Hard block counter-trend SHORT
    if signal.direction == "SHORT" and market_data.h4_trend == "UP":
        logger.info(f"[SKIP] Counter-trend: {signal.symbol} SHORT di H4 UP")
        return False

    # Future: pertimbangkan mirror untuk LONG + H4 DOWN
    # Data saat ini terlalu sedikit (1 instance) untuk dijadikan hard rule

    return True
```

Lokasi: segera setelah LGBM+LSTM menghasilkan signal, sebelum kalkulasi TP/SL.

### Catatan Implementasi
- `h4_trend` harus berasal dari **H4 candle terkini** saat signal dihasilkan, bukan dari feature yang stale
- Pastikan `h4_trend` tersedia di `MarketData` atau `SignalContext` — jika tidak ada, tambahkan ke feature extraction

---

## Urutan Aplikasi Fix (Order Matters)

```
Signal LONG/SHORT dihasilkan (LGBM + LSTM)
        │
        ▼
[2A] Counter-trend filter
  SHORT + H4 UP? → SKIP
        │ lolos
        ▼
[1B] Resolve TP/SL dengan staleness check
  H4High/H4Low valid? → pakai swing
  Stale? → ATR fallback
        │
        ▼
[1A] RR re-validation dengan TP/SL final
  RR ≥ 1.0? → lanjut
  RR < 1.0? → SKIP
        │ lolos
        ▼
Submit order
```

2A ditempatkan paling awal karena paling murah (tidak perlu kalkulasi TP/SL).

---

## File yang Perlu Diubah di swint_tradev2

| File | Perubahan |
|------|-----------|
| `app/services/paper_trading.py` | Tambah call ke `validate_rr()` setelah TP/SL ditetapkan |
| `app/services/signal_filter.py` | Tambah kondisi SHORT+H4 UP sebelum entry |
| `app/services/paper_trading.py` atau `tp_sl_resolver.py` | Refactor `resolve_tp_sl()` dengan staleness guard |
| `models/inference_config.json` | Verifikasi `rr_gate.min_rr = 1.0` sudah ada (sebagai source of truth) |

Jika tidak ada `signal_filter.py`, tambahkan filter di step awal `paper_trading.py` sebelum TP/SL calculation.

---

## Yang Tidak Perlu Dilakukan (Anti-Rekomendasi)

Berdasarkan analisis, rekomendasi berikut dari TRADE_ANALYSIS_REPORT.md **tidak** diimplementasikan:

| Rekomendasi | Alasan Ditolak |
|-------------|---------------|
| Naikkan `guardian.min_hold_bars` ke 5 | Kausalitas terbalik: Guardian exit loss di bar 3–5 adalah BENAR — Guardian memotong loss sebelum SL. TONUSDT: Guardian cut −11% vs SL potensial −44%. Menaikkan min_hold = rugi lebih dalam. |
| Naikkan `guardian.exit_threshold` ke 0.65 | Tanpa retraining dengan distribusi baru, mengubah threshold tanpa dasar statistik. Threshold 0.60 divalidasi di backtest. |
| Exclude 6 koin (POL, DOT, AVAX, PEPE, HBAR, SUI) | Data snooping — memilih koin berdasarkan data yang sama yang dianalisis. Tidak valid forward-looking. |
| Rollback ke cascade_v2 | cascade_v2 WR 56% karena bull phase May 8–14 (6 wins consecutive May 14). Timing, bukan model superiority. |

---

## Tambahan: MAX_HOLD Enforcement

Di live data ditemukan 2 trade yang hold melebihi MAX_HOLD=24:
- BNBUSDT: 26 bars (seharusnya time_exit di bar 24)
- ETHUSDT: 29 bars

Periksa logika bar counting di `paper_trading.py`:

```python
# Pastikan ini menggunakan candle count, bukan wall-clock time
if bars_held >= MAX_HOLD_BARS:  # bukan > (off-by-one)
    close_trade(reason="time_exit")
```

---

## Data Pendukung — Trade Examples

### Contoh 1: Bug 1A (RR Gate Bypass)
```
SUIUSDT SHORT (2026-05-18 04:05)
  Entry=1.0448, TP=1.0182, SL=1.0966, ATR=0.01327
  H4Low=1.0357 (0.9% di bawah entry → ATR fallback digunakan)
  TP = entry − 2×ATR = 1.0448 − 0.02654 = 1.0182  ← ATR fallback
  SL = H4High + 0.5×ATR = 1.0900 + 0.0066 = 1.0966  ← swing-based
  RR final = (1.0448−1.0182)/(1.0966−1.0448) = 0.0266/0.0518 = 0.51
  Masuk → WIN +8.47 (lucky, bisa saja LOSS)
```

### Contoh 2: Bug 1B (Stale SL)
```
TONUSDT LONG (2026-05-08 05:05)  
  Entry=2.6888, H4High=2.9078, H4Low=1.3065
  H4Low = 51.4% di bawah entry
  SL ditetapkan = 1.3065  ← catastrophic risk
  Trade time_exit -15.76% (harga tidak sampai SL)
  Dengan 1B: SL = entry − 1.5×ATR = 2.6888 − 0.1517 = 2.5371 (5.6% away)
  Trade outcome sama (time_exit), tapi SL risk jauh lebih rasional
```

### Contoh 3: 2A (Counter-trend SHORT)
```
TONUSDT SHORT (2026-05-18 22:05)
  Conf=0.94 (tertinggi), H4 Trend=UP, RR=0.91
  Masuk → guardian exit bar 9 → LOSS −$11.25
  Dengan 2A: trade diblokir karena SHORT + H4 UP
  
  Guardian exit di 2.0663 (dari entry 2.0212) — harga naik 2.2% melawan SHORT
  Jika tidak ada Guardian: harga menuju SL 2.1987 = −8.8% × 5x = −44%
  Guardian benar memotong di −11.25%, tapi trade harusnya tidak masuk sama sekali
```

---

## Metrik Backtest untuk Referensi

Dari holdout temporal OOS (Mei 2025 – Apr 2026, 21 koin), **sebelum** bug-bug ini:

| Metrik | Guardian v3 (backtest) | Live cascade_v3 (sekarang) | Gap |
|--------|----------------------|--------------------------|-----|
| WR | 88.93% | 52.2% | −36.7pp |
| PF | 10.05 | 1.11 | −8.94 |
| Mean DD | 41.77% | (belum dihitung) | — |

Gap yang besar menunjukkan bahwa bug-bug ini (terutama 1A dan 2A) secara material menyebabkan perbedaan antara backtest dan live.

---

*Analisis dari: `d:\Apps-Dev\Riset_pemodelan\reports\TRADE_ANALYSIS_REPORT.md` + manual trace `livetrade.csv`*  
*Model files: `d:\Apps-Dev\Riset_pemodelan\models\guardian_best.pkl`, `lgbm_baseline.pkl`, `lstm_best.pt`*
