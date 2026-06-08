# Audit Overfitting & Data Leakage — LSTM Momentum Pipeline

**Tanggal audit**: 2026-05-31  
**Scope**: Pipeline LSTM H1 (`05a` → `05b` → `05c`)  
**Auditor**: Claude Sonnet 4.6  
**Status**: BUG DITEMUKAN DAN DIPERBAIKI — pipeline bersih setelah fix

---

## Temuan 1 — BUG KRITIS: Off-by-one Sequence vs Label (05b)

**Severity**: HIGH — penyebab utama F1 ≈ random di v4.3

### Masalah

Di `05b_build_h1_sequences.py`, sequence diambil:

```python
# SEBELUM (salah)
seq = feat_arr[i - seq_len : i]   # bars i-32 s.d. i-1
```

Label di bar `i` (dari `05a`) menggunakan `close[i]` sebagai titik referensi:

```python
c0 = close_arr[i]
total_ret = close_arr[i+n] - c0   # diukur DARI close[i]
bar_rets = close_arr[i+1:i+n+1] - close_arr[i:i+n]
```

**Akibatnya**: model diminta memprediksi arah dari `close[i]` tanpa melihat bar `i` sama sekali. Dua sequence identik (bars `i-32` s.d. `i-1`) bisa mendapat label berbeda bergantung apa yang bar `i` lakukan — ini adalah **irreducible label noise** yang membuat model tidak bisa belajar pola yang konsisten.

### Akibat Tambahan: Train/Inference Mismatch

Production (`data_service.py:167`):
```python
seq = X_scaled[-lstm_seq_len:]  # ambil 32 bar TERAKHIR termasuk current bar i
```

Production sudah benar mengambil sequence ending at bar `i`. Training yang salah (ending at `i-1`). Model ditraining pada distribusi berbeda dari yang diterima saat inference → mismatch struktural.

### Fix yang Diterapkan

```python
# SESUDAH (benar)
seq = feat_arr[i - seq_len + 1 : i + 1]   # bars i-31 s.d. i (inclusive)
```

Bar `i` sekarang masuk sequence. Model melihat `h1_return[i]`, `rsi_6[i]`, `ofi_raw[i]`, dll. — sinyal momentum terkini yang menjadi dasar prediksi. Semua fitur bar `i` bersifat backward-looking, tidak ada look-ahead.

---

## Temuan 2 — BUG MINOR: `feature_order` Stale di 05c

**Severity**: MEDIUM — bisa menyebabkan salah feature alignment di inference

`05c_train_lstm_h1.py` menyimpan `meta["feature_order"]` dengan 12 fitur hardcoded, padahal `05b` sudah menghasilkan 14 fitur (ditambah `oi_delta_pct` dan `btc_h1_return`).

**Fix**: `feature_order` di-update ke 14 fitur agar sync dengan `H1_SEQ_FEATURES` di `05b`.

---

## Audit Leakage — Semua 14 Fitur LSTM

Semua fitur diverifikasi backward-looking (tidak pakai data masa depan):

| Fitur | Definisi | Leakage? |
|-------|---------|----------|
| `h1_return` | `close.pct_change()` | Tidak ✓ |
| `log_ret_5` | `log(c / c.shift(5))` | Tidak ✓ |
| `log_ret_20` | `log(c / c.shift(20))` | Tidak ✓ |
| `volume_delta` | `taker_buy - taker_sell` (bar saat ini) | Tidak ✓ |
| `ofi_raw` | `buy_vol - sell_vol` (bar saat ini) | Tidak ✓ |
| `ofi_acceleration` | `ofi_raw.diff(3)` | Tidak ✓ |
| `rsi_6` | `calc_rsi(c, 6)` — rolling backward | Tidak ✓ |
| `stochrsi_k` | rolling min/max RSI + `.rolling(k).mean()` | Tidak ✓ |
| `vwdp_smooth` | `vwdp.rolling(window).mean()` — rolling backward | Tidak ✓ |
| `atr_14_h1` | ATR 14-period backward | Tidak ✓ |
| `vol_ratio_20` | `v / v.rolling(20).mean()` | Tidak ✓ |
| `bars_since_BOS` | `shift(lookback+1)` — bar `i` pakai swing confirmed up to `i-1` | Tidak ✓ |
| `oi_delta_pct` | `open_interest.pct_change().clip().fillna(0)` | Tidak ✓ |
| `btc_h1_return` | BTC `pct_change` + `reindex(ffill)` — backward | Tidak ✓ |

### Catatan `bars_since_BOS`

Fitur paling rawan karena `detect_swing_highs_lows` menggunakan window `[i-5 : i+6]` (forward 5 bar). Namun `calc_market_structure` menerapkan `shift(lookback+1)` sebelum digunakan, sehingga:

- Swing di bar `k` dikonfirmasi pada bar `k+5`, tersedia via `shift(6)` di bar `k+6`
- `bars_since_BOS[i]` maksimal menggunakan data hingga bar `i-1`

Aman untuk bar `i` di dalam sequence. ✓

---

## Audit Label Generation (05a)

| Check | Status |
|-------|--------|
| Filter `TRAIN_CUTOFF_DATE` sebelum labeling | ✓ |
| `labels[-n:] = "FLAT"` — tail N bar dipaksa FLAT | ✓ |
| Label untuk bar `i = length-n-1` pakai `close[length-1]` (masih dalam training window) | ✓ |
| Formula `up_count` dan `total_ret` tidak menggunakan future beyond `i+N` | ✓ |

---

## Audit Purge Gap (05c + shared.py)

`PURGE_GAP_BARS = 24`, `SEQ_LEN = 32`, `N_label = 12`

Setelah fix (sequence ending at bar `i`):

| | Nilai |
|---|---|
| Last training label | ~`T_fold - 25` |
| Last training sequence (akhir) | `T_fold - 25` |
| First test label | ~`T_fold + 24` |
| First test sequence (awal) | `T_fold - 7` |
| Gap antara akhir train seq dan awal test seq | **18 bar** — tidak ada overlap ✓ |
| Last training label forward horizon | `T_fold - 13` — sebelum test fold ✓ |

Purge 24 bar dari masing-masing sisi (dead zone ~48 bar) sudah cukup untuk mencegah sequence overlap (butuh minimal 31 bar) dan label horizon leakage (butuh minimal 12 bar).

---

## Audit Scaler

| Check | Status |
|-------|--------|
| `fold_scaler = fit_scaler(X_tr)` — hanya training fold | ✓ |
| `final_scaler = fit_scaler(X)` — semua pre-cutoff data (benar untuk final model) | ✓ |
| Val fold di-scale dengan `fold_scaler` (bukan full scaler) | ✓ |

---

## Audit `retrain_final`

`avg_epochs = int(np.mean([m["best_epoch"] for m in all_metrics]))` lalu retrain tanpa early stopping. Ini adalah pola standar — epoch dihitung dari CV yang valid. Risiko overfit moderat karena tidak ada validasi di final retrain, namun `weight_decay=1e-4` dan `dropout=0.3` menjadi regularizer.

**Rekomendasi monitoring**: perhatikan apakah training loss di final retrain masih turun stabil di epoch terakhir atau sudah datar — kalau masih turun tajam, `avg_epochs` mungkin terlalu rendah; kalau sudah sangat datar sejak awal, mungkin terlalu tinggi.

---

## Sliding Window Autocorrelation — Inheren, Bukan Bug

Setiap sample berurutan berbagi 31 dari 32 bar. Jumlah sample efektif independen jauh lebih kecil dari nominal. Ini inheren untuk LSTM time-series dengan sliding window dan tidak bisa dihilangkan tanpa ganti arsitektur. CV F1 bisa sedikit optimistik di dekat fold boundary karena alasan ini.

---

## Tindakan yang Sudah Diambil

1. `05b` line 153: `feat_arr[i - seq_len : i]` → `feat_arr[i - seq_len + 1 : i + 1]`
2. `05c` `feature_order`: sync 12 → 14 fitur (tambah `oi_delta_pct`, `btc_h1_return`)
3. Cache `data/training/h1_sequences/all_coins_seq.npz` dihapus — akan direbuild saat `05b` jalan ulang

## Langkah Selanjutnya

```
python pipeline/05b_build_h1_sequences.py --all
python pipeline/05c_train_lstm_h1.py --all --run-id cascade_v4.4
```

Target F1 > 0.36. Jika masih ≤ 0.35 setelah fix ini, kemungkinan besar sinyal H1 momentum memang tidak cukup kuat untuk dipelajari LSTM pada horizon 12 bar (masalah fundamental EMH, bukan masalah pipeline).
