# Audit Report: H4 Soft Filter — Opsi A (Balanced) Penalty Refinement

**Date:** 2026-05-03  
**Topic:** Fine-tuning soft filter penalty values for neutral bias

---

## RINGKASAN EKSEKUTIF

Current soft filter penalties (flat=-0.01, opposite=-0.04) create a large gap between FLAT and OPPOSITE treatment, biasing the system towards "trade unless H4 strongly opposes." Reviewer proposes **Opsi A (balanced)**: flat=-0.015, opposite=-0.035. This narrows the gap, making the filter more neutral — FLAT gets slightly more penalty (no longer ignorable), opposite gets slightly less penalty (no longer catastrophic). Net effect: system is less "bullish" on trade execution, more balanced.

---

## TEMUAN PER KATEGORI

### [KONFIGURASI] Lokasi: `config.py:194-196`
**Deskripsi:** Current soft filter penalty values are aggressive — FLAT penalty is very light (0.01), opposite penalty is heavy (0.04).
**Dampak:** Gap of 0.03 between FLAT and OPPOSITE is wide. When H4 is FLAT (~70%+ of samples for AUC 0.55), the -0.01 penalty is negligible, so almost all H1 decisions pass. System is biased toward action.
**Bukti:**
```python
H4_SOFT_ALIGN_BOOST      = 0.04   # strong boost
H4_SOFT_FLAT_PENALTY     = 0.01   # very light
H4_SOFT_OPPOSITE_PENALTY = 0.04   # heavy
```

### [KONSEP] Analisis bias: "terlalu bullish"
**Deskripsi:** Current design heavily favors trade execution:
- Aligned: +0.04 (reward)
- FLAT: -0.01 (barely penalized)
- Opposing: -0.04 (harsh)

Net effect: When H4 is FLAT (most common), system barely penalizes. This creates a "trading bias" — system defaults toward action unless H4 actively opposes.
**Dampak:** May increase trade frequency at the cost of quality.
**Rekomendasi:** Opsi A balanced — narrower gap, more neutral stance.

---

## JALUR EKSEKUSI

```
config.py → H4_SOFT_FLAT_PENALTY, H4_SOFT_OPPOSITE_PENALTY
         → backtest_utils.py: hierarchical_predict() → H4 adjustment
```

---

## HIPOTESIS PENYEBAB ROOT

1. **Penalty gap terlalu lebar** — 0.03 antara FLAT (-0.01) dan OPPOSITE (-0.04) membuat treatment tidak proporsional untuk weak model.

---

## REKOMENDASI PERBAIKAN

### Opsi A (Balanced) — Rekomendasi reviewer
| Condition | Current | Proposed | Delta |
|-----------|---------|----------|-------|
| Aligned   | +0.04   | +0.04    | same  |
| FLAT      | -0.01   | **-0.015** | 50% heavier |
| Opposite  | -0.04   | **-0.035** | 12.5% lighter |

**Mengapa:** Gap mengecil dari 0.03 ke 0.02. FLAT tidak lagi bisa diabaikan, opposite tidak lagi terlalu keras. Sistem lebih netral — tidak "bullish" terhadap trade execution.

### Implementation
Ubah nilai di `config.py:195-196`:
```python
H4_SOFT_FLAT_PENALTY     = 0.015  # dinaikkan dari 0.01
H4_SOFT_OPPOSITE_PENALTY = 0.035  # diturunkan dari 0.04
```

Tidak perlu perubahan di `backtest_utils.py` — logika sudah menggunakan variabel terpisah.
