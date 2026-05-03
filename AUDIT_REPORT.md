# Audit Report: H4 Training Still Failing — Config Change Not Applied in Cloud

**Tanggal:** 2026-05-03  
**Auditor:** AI Code Analysis  
**Mode:** Read-only audit (no code changes)

---

## Ringkasan Eksekutif

**Hasil tetap sama (100% FLAT) karena file [`config.py`](config.py:164) di cloud Colab belum diubah.** Perbaikan `H4_SWING_LABEL_MIN_RR = 2.0 → 0.6` sudah diidentifikasi, diterapkan sementara di lokal, lalu di-revert kembali sesuai instruksi "jangan ubah code". Perubahan ini tidak pernah di-push ke GitHub, sehingga Colab masih menggunakan nilai buggy `2.0`.

---

## Temuan

### [BUG] `H4_SWING_LABEL_MIN_RR = 2.0` masih aktif

| Field | Detail |
|-------|--------|
| **Lokasi** | [`config.py:164`](config.py:164) |
| **Nilai saat ini** | `H4_SWING_LABEL_MIN_RR = 2.0` (belum diubah) |
| **Nilai seharusnya** | `H4_SWING_LABEL_MIN_RR = 0.6` (agar ≤ max RR teoritis `0.667`) |
| **Status di GitHub** | Belum di-push — Colab masih pakai kode lama |
| **Dampak** | 100% FLAT terus berlanjut karena perubahan hanya di lokal, tidak di cloud |

### [DATA] Bukti matematis — masih bug

`04_train_lgbm_h4.py` jam 08:08:03 menunjukkan output identik dengan sebelumnya:

```
RR = min_tp_atr / max_sl_atr = 2.0 / 3.0 = 0.667
Condition: 0.667 >= min_rr(2.0) = FALSE → 100% FLAT
```

---

## Langkah Perbaikan di Colab

Edit [`config.py`](config.py:164) langsung di lingkungan Colab:

**Sebelum (baris 164):**
```python
H4_SWING_LABEL_MIN_RR   = 2.0   # vs 1.2 di H1
```

**Sesudah:**
```python
H4_SWING_LABEL_MIN_RR   = 0.6   # max theoretical RR = 2.0/3.0 ≈ 0.667
```

Kemudian jalankan ulang:
```bash
python pipeline/04_train_lgbm_h4.py --all
```
