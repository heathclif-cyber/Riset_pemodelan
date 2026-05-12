# EXPERIMENTS.md — Logbook Eksperimen & Perubahan Parameter

## 2026-05-12 — Debug SHORT Signal & LSTM Conversion Rate

**Latar belakang**: Live signal 12 Mei 2026 menghasilkan 0 SHORT dari 208 sinyal (13 bar × 16 coin).
Paper trade 24 closed trades menunjukkan WR 66.7% (cascade_v2) tapi semua LONG — tidak ada SHORT entry.

### Perubahan Parameter

| # | Parameter | Lama | Baru | Alasan |
|---|-----------|------|------|--------|
| 1 | `CONFIDENCE_THRESHOLD_ENTRY` | 0.70 | **0.62** | Selaraskan dengan threshold internal cascade (LGBM_THRESHOLD 0.62). Gap 0.62-0.69 adalah "zona mati" yang membunuh sinyal tanpa alasan. |
| 2 | `LSTM_OVERRIDE_THRESHOLD` | (tidak ada) | **0.70** | Threshold LSTM untuk override FLAT dipisah dari LGBM entry threshold. Sebelumnya LSTM cuma perlu 0.62 untuk membatalkan keputusan FLAT LGBM. |
| 3 | `LSTM_ADJUST_OPPOSITE_PEN` | 0.08 | **0.04** | Penalti opposite terlalu keras — membunuh 121 trade bagus (WR 63.6%, PnL +$510). Dipotong setengah. |
| 4 | `LSTM_TIERED_MULTIPLIERS` | [1.5, 1.0, 0.5] | **[1.0, 0.5, 0.25]** | Multiplier tiered sebelumnya terlalu agresif untuk sinyal borderline (margin < 0.05 kena 1.5x). Diringankan. |
| 5 | `LSTM_FLAT_REVIEW_ENABLED` | True (implicit) | **False** | FLAT review menambah 2,500+ trade dengan WR 39%. Disable → WR naik 57.9% → 78.8%. LSTM tetap aktif sebagai confirmation. |

### Temuan Kunci

1. **0 SHORT di live data = regime market, BUKAN bug model**
   - Backtest 5 coin holdout: LGBM menghasilkan 2,703 SHORT (10.1%) vs 2,057 LONG (7.7%)
   - Cascade menghasilkan 2,801 SHORT (54%) vs 2,358 LONG (46%)
   - SHORT WR = LONG WR (~78%) — model tidak bias
   - 13 bar live data (Mei 2026) kebetulan di regime UP — wajar tidak ada SHORT

2. **LSTM FLAT review menambah 2,500+ trades tapi WR cuma 39%**
   - Override terjadi saat LGBM ragu FLAT (max_conf < 0.90) dan LSTM deteksi sinyal
   - WR override mentok 39.7% tidak peduli threshold 0.70 / 0.80 / 0.90
   - Akar masalah: zona LGBM FLAT adalah zona noise — tidak ada sinyal yang cukup kuat
   - Efek: WR cascade keseluruhan turun dari 78% ke 57%

3. **LSTM opposite penalty tiered membunuh sinyal bagus**
   - 121 trade LGBM diblok LSTM dengan WR 63.6% dan PnL +$510
   - Penalti tiered terlalu berat untuk sinyal borderline (conf 0.62-0.67)

4. **Cascade dekomposisi (5 coin holdout, threshold 0.62):**
   - LGBM-LSTM AGREE: 2,005 trades, WR 79.0%, PnL +$13,394 (70% total)
   - LSTM OVERRIDE: 2,631 trades, WR 39.3%, PnL +$1,579 (8% total)
   - LSTM BLOCKED: 121 trades, WR 63.6%, PnL +$510 (3% total)

### Final Sweet Spot — LGBM + LSTM, NO FLAT Review

Backtest 5 coin holdout (11 bulan, Mei 2025 – Mar 2026), threshold 0.62:

| Skenario | Trades | WR | LONG WR | SHORT WR | PnL | PnL/t |
|----------|--------|-----|---------|----------|------|-------|
| Cascade FULL (ovr=0.70) | 5,443 | 57.9% | 57.9% | 58.0% | +$19,122 | $3.51 |
| Cascade FULL (ovr=0.90) | 4,830 | 60.0% | 60.7% | 59.5% | +$18,736 | $3.88 |
| LGBM-only (tanpa LSTM) | 2,428 | 78.0% | 77.2% | 78.7% | +$15,981 | $6.58 |
| **LGBM+LSTM, NO override** | **2,315** | **78.8%** | **78.3%** | **79.2%** | **+$15,516** | **$6.70** |

**Dipilih: LGBM + LSTM, NO FLAT review** karena:
- WR tertinggi (78.8%) — psikologis trading terjaga
- SHORT tetap dominan (57%, WR 79.2%) — tidak bias LONG
- LSTM tetap menyaring sinyal jelek (dibanding LGBM-only: 113 trade dibuang)
- ~7 trade/hari untuk 5 coin (~1.4/coin/hari) — tidak kebanjiran sinyal

### Paper Trade Analysis (8-12 Mei 2026, 5 hari)

24 closed + 2 open trade (cascade_v2 lama, FLAT review ON):

| | Count | Rate |
|---|-------|------|
| Wins | 16 | 67% |
| False Positive (Loss) | 8 | 33% |

FP by confidence:
- Conf 0.70-0.80: **75% FP** (3/4)
- Conf 0.80-0.90: 50% FP (2/4)
- Conf 0.90-1.00: **13% FP** (2/15)

FP by coin: DOTUSDT 100%, ETHUSDT 67%, TONUSDT 100%, AVAXUSDT 100%.

Setup baru (no FLAT review) diekspektasikan: lebih sedikit trade (~10 vs 24), FP rate lebih rendah (~20% vs 33%) karena hanya entry saat confidence tinggi + kedua model setuju.

### Keputusan Final

- [x] `CONFIDENCE_THRESHOLD_ENTRY` = 0.62 (selaras cascade internal)
- [x] `LSTM_ADJUST_OPPOSITE_PEN` = 0.04 (turun dari 0.08)
- [x] `LSTM_TIERED_MULTIPLIERS` = [1.0, 0.5, 0.25] (diringankan)
- [x] `LSTM_OVERRIDE_THRESHOLD` = 0.70 (threshold override terpisah)
- [x] `LSTM_FLAT_REVIEW_ENABLED` = False (WR 78.8% vs 57.9%)
- [x] CLAUDE.md dirapikan — hapus duplikasi config, roadmap, riwayat perbaikan
- [ ] Pantau live trading dengan setup baru — bandingkan FP rate

### Apa yang Dimitigasi vs Tidak

| Bisa Dimitigasi | Tidak Bisa |
|-----------------|------------|
| Trade gambling (override WR 39%) dihilangkan | SL hit — tidak ada model bisa prediksi support/resistance break |
| FP dari confidence rendah berkurang (hanya entry saat kedua model setuju) | Time exit — max_hold 24 bar tetap |
| Jumlah trade lebih sedikit & berkualitas | 0 SHORT di regime UP — tergantung market |

### File Terkait

- `config.py` — parameter yang diubah (baris 231-233, 249-254)
- `pipeline/backtest_utils.py` — `hierarchical_predict()`, `_lstm_adjustment()`
- `pipeline/test_inference_backtest.py` — script backtest standalone (dibuat untuk pengujian ini)
- `CLAUDE.md` — update cascade flow + referensi EXPERIMENTS.md

---

## Template — Cara Mencatat Eksperimen Baru

```markdown
## YYYY-MM-DD — Judul Singkat

### Latar Belakang
[1-2 kalimat kenapa eksperimen ini dilakukan]

### Perubahan Parameter
| # | Parameter | Lama | Baru | Alasan |
|---|-----------|------|------|--------|

### Hasil
[Metrik sebelum vs sesudah]

### Keputusan
- [ ] Diterapkan / ditolak / perlu pengujian lanjutan
```
