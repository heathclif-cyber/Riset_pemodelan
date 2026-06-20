# ic32 Scale-In Pyramiding — Laporan Lengkap

**Tanggal**: 2026-06-18  
**Status**: DEPLOYED ke VPS  
**Model**: ic32_regime_v1 (B-dir + hard_consensus + continuation_v1 + min_hold=4 + SL close)

---

## 1. Ringkasan Eksekutif

Scale-in menggantikan multi-leg `shared_sl_first` yang sebelumnya live di VPS.

| Aspek | Multi-leg (lama) | Scale-in (baru) |
|-------|------------------|-----------------|
| DB record per posisi | 2 row (leg 1 + leg 2) | **1 row** (modal numpuk) |
| Entry price | Per leg | **VWAP** semua leg |
| TP/SL | Leg 2+ anchor SL leg 1 | **Semua exposure** pakai TP/SL leg 1 |
| Guardian hold_bars | Per leg | Dari **entry leg 1** |
| Exit | Per leg (bisa beda bar) | **Satu exit** untuk seluruh modal |

**Holdout Apr 1 – Jun 12 2026** (21 koin, $10/trade, 5x):

| Variant | Trades | WR | PF | ppt_norm | PnL |
|---------|-------:|---:|---:|---------:|----:|
| no_pyr | 355 | 71.8% | 3.79 | $0.516 | +$183 |
| pyr2_shared_sl_first | 446 | 72.0% | 3.92 | $0.517 | +$231 |
| **pyr2_scale_in** | **356** | **71.6%** | **3.90** | **$0.547** | **+$249** |

Scale-in menang: PnL +8% vs multi-leg, +36% vs baseline, dengan trade count flat (+0.3%).

---

## 2. Hipotesis & Motivasi

### Masalah multi-leg
- UX: 2 transaksi terpisah di DB untuk 1 "posisi mental" trader
- +25% trade count di holdout (446 vs 355) — noise di laporan
- PPT flat ($0.517 vs $0.516) — edge per $10 exposure tidak naik

### Teori scale-in
- Sinyal lanjutan searah = tambah conviction, bukan posisi baru
- 1 exit unified → Guardian/SL/hold dari entry pertama (anchor)
- Modal numpuk → PnL absolut naik tanpa inflasi trade count

### Hipotesis yang DITOLAK (forensics OOF)
> "Pyramiding mencegah SL entry awal"

Leg 1 SL rate **sama** (~19.7%) dengan atau tanpa pyramiding. Pyramiding tidak mengubah probabilitas SL leg pertama.

---

## 3. Arsitektur Simulasi (Riset)

### File kunci
| File | Fungsi |
|------|--------|
| `core/scale_in_sim.py` | Simulator OOF/holdout — 1 trade record, VWAP, unified exit |
| `core/evaluator.py` | Dispatch `pyramiding_exit_mode == "scale_in"` |
| `pipeline/08j_oof_ic32_scale_in_sweep.py` | OOF sweep 3 variant |
| `pipeline/07h_holdout_ic32_scale_in_diag.py` | Holdout diagnostic sekali |

### Alur scale-in (backtest)
```
Bar i: sinyal LONG, tidak ada posisi
  -> Buka leg 1: vwap=price, total_modal=$10, TP/SL dari bar ini

Bar j: sinyal LONG lagi, posisi aktif, legs < 2
  -> Scale-in: vwap = (old*vwap_old + price*modal) / (old+modal)
  -> total_modal += $10, legs += 1
  -> TP/SL TIDAK dihitung ulang (anchor leg 1)

Exit (Guardian/SL/timeout):
  -> Satu exit untuk seluruh total_modal
  -> hold_bars dari entry_bar leg 1
  -> PnL = pct_move(vwap, exit) * total_modal * leverage - fees
```

### Hasil OOF (6,381 baseline trades)
| Variant | Trades | WR | PF | PPT | ppt_norm | avg modal | 2-leg% | PnL |
|---------|-------:|---:|---:|----:|---------:|----------:|-------:|----:|
| no_pyr | 6,381 | 69.4% | 3.12 | $0.570 | $0.570 | $10 | 0% | $3,636 |
| pyr2_shared_sl_first | 7,931 | 69.4% | 3.07 | $0.565 | $0.565 | $10 | — | $4,480 |
| **pyr2_scale_in** | **6,388** | **68.4%** | **3.28** | **$0.803** | **$0.640** | **$12.55** | **25.5%** | **$5,129** |

Artefak: `models/runs/ic32_regime_v1/ic32_scale_in_sweep_oof.json`

---

## 4. Implementasi Live (swint_tradev2)

### Config (`inference_config.json`)
```json
"pyramiding": {
  "enabled": true,
  "max_positions_per_coin": 2,
  "same_direction_only": true,
  "exit_mode": "scale_in"
}
```

`max_positions_per_coin` = **max legs**, bukan max DB rows. Scale-in selalu 1 row per koin.

### Kode (`app/services/paper_trading.py`)

**Entry leg 1** — flow normal:
- Hitung TP/SL hybrid swing+ATR
- RR gate
- INSERT `Trade` baru

**Entry leg 2+ (scale-in)** — `_apply_scale_in()`:
- Skip TP/SL & RR gate (anchor leg 1)
- VWAP: `new_vwap = (entry * qty + price * add_modal) / (qty + add_modal)`
- `quantity += add_modal`
- `fee_total += add_modal * leverage * fee_per_side`
- TP/SL, `opened_at`, `hold_bars` **tidak berubah**
- Signal reason: `SCALE-IN leg N trade #ID ...`

**Exit** — tidak berubah:
- `check_open_positions()` eval per open trade
- Dengan scale-in hanya ada 1 trade → 1 exit untuk seluruh exposure
- Guardian `hold_bars` dari `opened_at` leg 1 (sudah benar)

**Batasan live**:
- `is_live=True`: scale-in add-on **ditolak** (belum ada market add-order di Binance)
- Dynamic sizing: leg count estimasi dari `quantity / modal_per_trade`

---

## 5. Perbandingan 3 Mode Exit

| Mode | DB rows | Entry leg 2+ | Exit | Holdout PnL | Holdout ppt_norm |
|------|---------|--------------|------|------------:|-----------------:|
| `no_pyr` | 1 | — | Normal | +$183 | $0.516 |
| `shared_sl_first` | 2 | TP/Guardian sendiri, SL anchor leg 1 | Per leg | +$231 | $0.517 |
| **`scale_in`** | **1** | **VWAP qty numpuk, TP/SL leg 1** | **Unified** | **+$249** | **$0.547** |
| `independent` | 2 | TP/SL/Guardian sendiri tiap leg | Per leg | OOF PPT -2.2% | — |
| `close_with_first` | 2 | Max hold = exit leg 1 | Forced sync | OOF WR -3.2pp | — |

---

## 6. Scorecard Holdout Detail (scale_in)

**Periode**: 2026-04-01 s/d 2026-06-12 (73 hari kalender, 72 hari trading)

| Metrik | Nilai |
|--------|------:|
| Total trades | 356 |
| Trade/hari aktif | 4.9 |
| Trade/bulan | 148 |
| WR overall | 71.6% |
| WR LONG | 71.5% (193 trd, 54.2%) |
| WR SHORT | 71.8% (163 trd, 45.8%) |
| PF overall | 3.90 |
| PF LONG | 3.47 |
| PF SHORT | 4.65 |
| PnL total | +$249.05 |
| PPT (per trade record) | +$0.700 |
| ppt_norm (@$10 exposure) | +$0.547 |
| Avg modal | $12.78 |
| 2-leg stacks | 27.8% |
| SL rate | 18.8% |
| Guardian exit | 68.3% |
| PnL/hari aktif | +$3.46 |
| Hari +PnL / -PnL | 52 / 20 |

Artefak: `models/runs/ic32_regime_v1/holdout_scale_in_diag_apr_jun26.json`

---

## 7. Gate Keputusan & Deploy

### Kriteria PROMOTE (semua terpenuhi)
- [x] `ppt_norm` holdout >= no_pyr (+6.1%)
- [x] PF holdout >= no_pyr (3.90 vs 3.79)
- [x] Total PnL >= multi-leg (+8%)
- [x] Trade count >= 80% baseline (356 vs 355 = 100%)
- [x] UX: 1 row DB per posisi

### Deploy checklist
1. `paper_trading.py` — `_apply_scale_in()`, `_scale_in_legs()`, branch `exit_mode=scale_in`
2. `inference_config.json` — `exit_mode: scale_in`
3. VPS: `deploy_production.py` (git push kode + scp config + restart)
4. Verifikasi: `scratch/verify_pyramiding_deploy.py`

---

## 8. Monitoring Live

Setelah deploy, pantau:
- Proporsi trade dengan `quantity > modal_per_trade` (~25-28% expected)
- Signal reason `SCALE-IN leg 2` di tabel signal
- PnL/trade vs baseline pre-deploy
- Tidak boleh ada 2 open trades per koin (bug jika terjadi)

Perintah verifikasi VPS:
```powershell
python scratch/verify_pyramiding_deploy.py
```

---

## 9. Rollback

Jika performa live mengecewakan:
```json
"pyramiding": {
  "exit_mode": "shared_sl_first"
}
```
+ redeploy config. Kode multi-leg masih ada di `paper_trading.py` (branch `shared_sl_first`).

Untuk disable total:
```json
"pyramiding": { "enabled": false, "max_positions_per_coin": 1 }
```

---

## 10. Referensi

- Logbook: `EXPERIMENTS.md` entri `ic32_scale_in_pyramiding`
- Forensics SL: `scratch/pyramiding_forensics.py` → `pyramiding_forensics_oof.json`
- Multi-leg holdout: `models/runs/ic32_regime_v1/holdout_pyramiding_diag_apr_jun26.json`