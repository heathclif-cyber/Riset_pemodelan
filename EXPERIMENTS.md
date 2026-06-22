# EXPERIMENTS.md — Logbook Riset ic32 Regime v2+

> Histori penuh (pre-2026-06-22) diarsipkan di `archive/2026-06-22_cleanup/root_files/EXPERIMENTS_full_history.md`

---

## Baseline Aktif — ic32_regime_v2_parity (Deployed 2026-06-22)

**Status**: PRODUCTION — baseline bersih untuk semua eksperimen berikutnya

### Arsitektur
- **LGBM**: `ic32_regime_v2_parity` — 33 fitur, rolling 8-fold walk-forward CV, thr long=0.75 / short=0.70
- **LSTM**: `ic32_lstm_regime_v2` — DINONAKTIFKAN (`lstm_confirmation_enabled: false`)
- **Guardian**: `ic32_rv2_parity_guard_oof` — 30 fitur, CV F1 0.847, exit_thr=0.90, min_hold=2
- **HMM**: Per-coin 4-state, per-state threshold

### Fitur kunci yang difix vs versi sebelumnya
| Fix | Detail |
|-----|--------|
| H4 lookahead | `resample(4h)` label digeser +4h, kausal. Terdampak: `cvd_slope_h4`, `ofi_h4_delta`, `cvd_div_h4` |
| CVD real | Ganti `cumsum(sign*vol)` proxy → `cumsum(buy_taker - sell_taker)` = identik live |
| LSR real | Merge data real dari `data.binance.vision/daily/metrics/` ke training + holdout |

### Metrik Jujur (OOF + Holdout)

**OOF (training 2020-01-01 – 2026-04-01):**
| Metrik | Nilai |
|--------|-------|
| WR | 50.8% |
| Trades | 4,812 |
| PnL | $583 |
| $/trade | $0.12 |

**Holdout Apr 1 – Jun 22, 2026 (SEALED):**
| Metrik | No Guardian | +Guardian |
|--------|-------------|-----------|
| Trades | 56 | 56 |
| WR | 42.9% | 41.1% |
| PF | 0.796 | 0.599 |
| PnL | -$3.28 | -$5.34 |

> Model sebelumnya menunjukkan WR 65% yang sebagian besar artefak lookahead H4.
> Ini adalah baseline jujur — improvement berikutnya diukur dari sini.

---

## Template Eksperimen Berikutnya

```markdown
## YYYY-MM-DD — [Nama Eksperimen]

**Status**: PLANNED

### Hipotesis
[Apa yang diduga akan terjadi dan mengapa]

### Yang Diubah
- [vs ic32_regime_v2_parity sebagai baseline]

### Target
- WR > 52%, PF > 1.0, Trades >= 45 (80% dari 56 baseline)
- Metodologi: genuine OOF, purge gap 36 bar

### Script
- [script yang akan dijalankan]
```
