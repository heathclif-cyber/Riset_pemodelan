# Widyawardhana — Model Meta Aktif

> **Golden Benchmark** — file ini adalah standar yang harus dilampaui oleh eksperimen baru.
> Setiap model kandidat dibandingkan dengan scorecard di file ini sebelum dinyatakan "lebih baik".
>
> **Cara update**: arsip versi lama ke `reports/experiments/YYYY-MM-DD_widyawardhana_vN.md`,
> lalu tulis versi baru di sini. Update HANYA jika model baru lebih baik di SEMUA kriteria:
> WR >= saat ini, PF >= saat ini, trades >= 80% saat ini, metodologi genuine OOF.

---

## Versi Aktif: ic32_regime_v2_parity (2026-06-22)

**Nama model**: `ic32.rv2.lgbm.33f.r8` + `ic32.rv2.guard.oof`
**Deployed**: 2026-06-22
**Status**: BASELINE BERSIH — model pertama tanpa lookahead bias

### Stack

| Komponen | Run ID | Detail |
|----------|--------|--------|
| LGBM | `ic32_regime_v2_parity` | 33 fitur, 8-fold rolling CV, thr 0.75/0.70 |
| Guardian | `ic32_rv2_parity_guard_oof` | 30 fitur, CV F1 0.847, exit_thr=0.90, min_hold=2 |
| LSTM | `ic32_lstm_regime_v2` | DINONAKTIFKAN di production |
| HMM | per-coin 4-state | Per-state threshold, state3-short=0.80 |

### Fitur LGBM (33)

Lihat `models/runs/ic32_regime_v2_parity/features.json` untuk list lengkap.
Fitur kunci: `cvd_slope_h4` (kausal), `ofi_h4_delta` (kausal), `cvd_momentum_adv`, `hmm_regime_enc`, `wyckoff_phase`, `volume_delta` (real taker).

### Fixes vs model sebelumnya

| Bug | Status |
|-----|--------|
| H4 lookahead (`resample` label di awal window) | FIXED — label digeser +4h |
| CVD proxy (`cumsum(sign*vol)`) | FIXED — `cumsum(buy_taker - sell_taker)` |
| LSR sintetik | FIXED — real data dari Binance Vision |

### Scorecard

**OOF (2020-01-01 – 2026-04-01, genuine purged CV):**
| Metrik | Nilai |
|--------|-------|
| WR | 50.8% |
| PF | ~1.12 |
| Trades | 4,812 |
| PnL | $583 |
| $/trade | $0.12 |

**Holdout (Apr 1 – Jun 22, 2026) — SEALED, satu kali buka:**
| Metrik | No Guardian | +Guardian |
|--------|-------------|-----------|
| Trades | 56 | 56 |
| WR | 42.9% | 41.1% |
| PF | 0.796 | 0.599 |
| PnL | -$3.28 | -$5.34 |

### Kriteria upgrade model berikutnya

Model kandidat harus melampaui **SEMUA** ini:
- Holdout WR > 42.9% (dengan Guardian)
- Holdout PF > 0.60
- Holdout Trades >= 45 (80% dari 56)
- Metodologi: genuine OOF purge gap 36 bar, holdout sealed

### Catatan penting

Model ini di-deploy bukan karena performanya luar biasa, tapi karena **jujur**.
Metrik lama (WR 65%, PF 2.88) adalah artefak lookahead H4. Baseline ini adalah
titik start yang valid untuk iterasi improvement yang terukur.
