# EXPERIMENTS.md — Logbook Riset ic32 Regime v2+

> Histori penuh (pre-2026-06-22) diarsipkan di `archive/2026-06-22_cleanup/root_files/EXPERIMENTS_full_history.md`

---

## 2026-07-04 — Fix inference_config: structural_filter OFF di source deploy (bukan cuma live UI)

**Status**: **APPLIED** — `models/inference_config.json` (Riset + swint), `deploy_model.py` enforce dari `config.py`

**Stack**: `LGBM opt2_plus_trend + HMM 0.65/0.10 + Guardian guard_opt2_plus_trend_hmm`

**Masalah**: `config.py` sudah `TP_SL_STRUCTURAL_FILTER=False` dan VPS pernah di-OFF via UI, tapi **source deploy** `models/inference_config.json` masih `structural_filter.enabled=true` → setiap `deploy_model.py` / scp config bisa menyalakan lagi. Menu Models menampilkan pill Structural ON.

**Perbaikan**:
- `models/inference_config.json`: `structural_filter.enabled=false` + catatan OOF sweep
- `tools/ops/deploy_model.py` `merge_inference_config`: paksa `enabled` dari `config.TP_SL_STRUCTURAL_FILTER` setiap deploy
- VPS: scp `inference_config.json` + restart

**Benchmark operasional (tetap)**: OOF minimal HMM+RR — 5,884 trade, PF **2.287** (tanpa structural/swing-fresh)

---

## 2026-07-04 — OOF filter sweep: structural/swing-fresh OFF, HMM+RR only (selaras live)

**Status**: **APPLIED** — `config.py` `TP_SL_STRUCTURAL_FILTER=False`, `TP_SL_SWING_FRESHNESS=False`

**Stack**: `LGBM opt2_plus_trend + HMM 0.65/0.10 + Guardian guard_opt2_plus_trend_hmm`

**Konteks**: Live VPS sudah `structural_filter.enabled=false`. Benchmark OOF lama (struct+swing ON) tidak merepresentasikan operasional.

### OOF sweep (`tools/model/oof_filter_sweep_minimal.py`)

| Skenario | Trades | WR | PF | PnL | MaxDD |
|----------|--------|-----|-----|------|-------|
| Baseline (struct+swing ON) | 5,261 | 66.5% | 2.108 | $2,438 | −$27.9 |
| **Minimal (HMM+RR only)** | **5,884** | 65.9% | **2.287** | **$3,386** | **−$25.5** |
| No RR gate | 10,200 | 67.7% | 3.043 | $6,811 | −$17.7 |
| No HMM regime thr | 5,271 | 64.4% | 2.226 | $3,077 | −$46.7 |
| LGBM only (no gates) | 7,477 | 66.2% | 2.500 | $4,463 | −$32.3 |

**Pseudo-holdout**: baseline PF 1.741 → minimal **1.896** (+9%), PnL $158 → **$223**.

### Keputusan

- **OFF**: structural filter, swing-freshness (OOF + live selaras, edge model tidak terpotong)
- **ON**: HMM gate per-regime (0.65/0.10), RR gate (no HMM → MaxDD −$46.7; no RR → backtest terinflasi)

**Artefak**: `models/runs/guard_opt2_plus_trend_hmm/oof_filter_sweep_minimal.json`

**Benchmark baru OOF full stack + guardian**: 5,884 trade, WR 65.9%, PF **2.287**, PnL $3,386, MaxDD −$25.5

---

## 2026-07-04 — Fix `trend_accel_4h` double-ATR + deploy v6.1.1 (threshold 0.65/0.10, min_hold=0)

**Status**: **DEPLOYED** — VPS `e299287` (git) + inference cache rebuild 2026-07-04T04:35Z

**Stack**: `LGBM opt2_plus_trend (lgbm38f) + HMM 0.65/0.10 (hmm24/48) + Guardian guard_opt2_plus_trend_hmm (guard28f, min_hold=0)`

### Bug

`trend_accel_4h = trend_strength.diff(4) / atr_h4` — **normalisasi ATR ganda** karena
`trend_strength` sudah `(ema7-ema50)/atr_h4`. Koin harga rendah (1000SHIB/PEPE, DOGE, TRX, …)
meledak (live s/d 3549 vs training mean ~0); BTC/ETH/TAO hampir tidak terasa.

**Fix** (`core/features.py`): `return trend_strength.diff(window)` — `atr_h4` dipertahankan di signature.

### Retrain + validasi

| Tahap | Hasil |
|-------|-------|
| Engineer `--all` + relabel `labeled_opt2` + sync join + holdout | 21/21 OK |
| LGBM `opt2_plus_trend` retrain | label `Fix trend_accel_4h double-ATR-normalization 2026-07-04` |
| Guardian `guard_opt2_plus_trend_hmm` retrain | 143,693 samples |
| OOF full stack + guardian | 5,261 trade, WR 66.5%, PF **2.108**, PnL $2,438, MaxDD -$27.92 |
| Sealed holdout OOS (2026-04-01 → 2026-07-03) | 290 trade, WR 58.6%, PF **1.424**, PnL **$46.14**, Long PF 1.258 |

**vs OOS sebelum fix** (187 trade, PF 1.158, PnL $11.42, Long PF 0.432): PnL ~4×, LONG untung.

**Trade 1–2 Jul (model sesudah fix)**: 6 trade, PnL -$0.54 (vs insiden live 19 trade -$8.57).

### Deploy

- `deploy_model.py` → swint lokal; SCP model + standards ke VPS; `features.py` fix ke VPS
- `training_feature_standards.json` regenerasi (`trend_accel_4h` std 2.04 → **0.168**)
- `systemctl restart swint-trade.service`; `POST /api/features/refresh` (20/21 cache OK)
- GitHub `swint_tradev2` push `e299287`

### Feature Monitor pasca-rebuild

| Metrik | Sebelum | Sesudah |
|--------|---------|---------|
| Pipeline OK | 9 | **19** |
| Warning | 11 (`trend_accel_4h` semua) | **1** (`1000PEPEUSDT` `coin_mkt_sync_24h` z=6) |
| Error | 1 (GRAMUSDT) | 1 (GRAMUSDT — klines < 82 bar) |

`trend_accel_4h` parity: **semua 20 koin ber-cache OK** (contoh 1000SHIB live 0.29, dulu 3549).

### Script

- `pipeline/data/run_engineer.py --all`
- `tools/model/relabel_labeled_opt2.py`, `tools/model/join_sync_training.py`
- `pipeline/model/experiments/train_lgbm_custom.py`
- `tools/model/train_guardian_opt2_plus_trend.py`
- `pipeline/model/run_oof_scorecard.py --stack fs38_28f`
- `pipeline/model/run_holdout_oos.py --stack fs38_28f --end-date 2026-07-03`
- `tools/ops/deploy_model.py`

---

## 2026-07-04 — Deploy v6.1 / fs38_28f (H4-closed + GRAMUSDT)

**Status**: **DEPLOYED** — snapshot `2026-07-03 18:05:00 UTC`, VPS `ff37118`, rollback v6.0 snapshot `2026-07-03 11:50:45`

**Stack**: `LGBM opt2_plus_trend (lgbm38f) + HMM 0.65/0.05 (hmm24/48) + Guardian guard_opt2_plus_trend_hmm (guard28f)`

| Metrik | v6.0 (live lama) | v6.1 (deploy ini) |
|--------|------------------|-------------------|
| OOF trades | 12,354 | **4,602** |
| OOF PF | 2.04 | **2.236** |
| OOF PnL | $5,332 | **$2,327** |
| Sealed holdout OOS | −$18 / 571 (post-fix JSON) | **+$42 / 257** PF 1.484 |

**Perubahan vs v6.0:**
- H4-closed + shift(4h) seragam (`core/features.py`)
- `funding_rate` fix di parquet labeled_opt2
- `training_feature_standards.json` regenerasi (`model_type: ic32_regime_v6.1`)
- **TONUSDT → GRAMUSDT** (rebrand Binance 2026-07-02; histori TON dibawa, bar GRAM asli append)
- `monitor.pairs` di `inference_config.json` (21 koin, GRAM mengganti TON)

**Deploy:** `python tools/ops/deploy_production.py -m "deploy v6.1 H4-closed GRAMUSDT"` — swint lokal + git push + SCP model + VPS restart OK.

**Audit pre-deploy:** HMM parity PASS (BTC/ETH/DOGE 0 mismatch). Feature value audit: 11 FLAG (mayoritas sinyal live TONUSDT lama + LSR flat) — pantau pasca-deploy.

**DB coin:** lokal `coin.id=2` TON→GRAM manual; VPS idem via SSH.

---

## 2026-07-03 — Refactor H4 closed + shift(4h) (seragamkan EMA/RSI/trend/swing)

**Status**: RETRAIN DONE (2026-07-03) — deploy v6.1 2026-07-04 (lihat entry di atas)

**Stack terdampak**: `ic32_regime_v6` / `fs38_28f` — semua fitur HTF di `core/features.py`

### Masalah

Fitur EMA/RSI/trend H4 memakai **partial-H4** (candle berjalan, update tiap jam) padahal
prinsip desain = **closed H4 sebelumnya** (sama seperti `ofi_h4_delta` / `cvd_slope_h4`).

### Perubahan (`core/features.py`)

- Helper baru: `_shift_h4_index`, `_align_h4_to_h1`, `_closed_h4_ohlc_on_h1`, `_ema_h4_on_h1`
- Pola tunggal: `resample("4h")` → hitung → index `+4h` → ffill ke H1
- Terdampak: `ema_*_h4`, `ema_*_slope_h4`, `h4_trend`, `trend_strength`, `rsi_h4`,
  `rsi_slope_h4`, `range_expansion_h4`, `atr_14_h4`, swing H4 (labeling + `dist_liq_*`),
  `calc_cvd_divergence` (input H1 close, bukan partial)

### Validasi (ETHUSDT train)

| Cek | Hasil |
|-----|-------|
| Closed H4 stabil intra-periode (08–11 UTC) | 1 nilai unik ✓ |
| Truncation `rsi_h4`, `h4_trend`, `ema_21_slope_h4`, `ofi_h4_delta`, `cvd_slope_h4` | PASS semua |

### Retrain (2026-07-03, H4 closed)

| Tahap | Hasil |
|-------|-------|
| Engineer `--all` + `labeled_opt2` relabel (ofi_z gate) + sync join | 21/21 OK |
| LGBM `opt2_plus_trend` 38f | OOF cov 88.9%, F1-macro fold ~0.57–0.58 |
| Guardian `guard_opt2_plus_trend_hmm` 28f | 148,564 samples (vs 408k pre-H4), logloss ~0.30, EXIT neg PnL 0% |
| Backup model lama | `opt2_plus_trend_bak_20260703_1505`, `guard_opt2_plus_trend_hmm_bak_20260703_1505` |

**Catatan**: Guardian sample turun drastis → entry trade count HMM-gated kemungkinan jauh di bawah baseline (12k OOF). Wajib jalankan full-stack sim sebelum deploy.

### Script

- `tools/model/retrain_v6_h4closed.py` (orchestrator)
- `tools/model/relabel_labeled_opt2.py`, `tools/model/join_sync_training.py`
- `pipeline/model/experiments/train_lgbm_custom.py --label-dir labeled_opt2`
- `tools/model/train_guardian_opt2_plus_trend.py`

### Kriteria lolos retrain

- OOF PF ≥ baseline v6 (2.04 full-stack) atau trade-off WR/PF acceptable
- Holdout OOS tidak regresi vs pre-refactor
- `audit_feature_value_parity.py` PASS setelah sync `features.py` ke swint

### Risiko

- Definisi fitur berubah → model v6 production **tidak valid** sampai retrain + deploy
- Label swing bisa bergeser sedikit (swing deteksi di grid H4 closed, bukan partial)

---

## 2026-07-01 — Guardian k5_mom **v7** (pnl_constrained_exit — SWEET SPOT TERCAPAI)

**Status**: OOF VALIDATED — lokal only, belum deploy VPS

**Stack**: `LGBM (ic32_rv2_lgbm_mkt_sync_v2) + HMM + k5_mom (K5_MOM_TR1, btc_mom=2%) + guard_k5mom_v7`

### Hipotesis

GUARDIAN_EXIT WR rendah di v6 (22%) karena model belajar dari label L1_loss_exit yang meng-EXIT
saat `current_pnl < LOSS_EXIT_THR = -0.006`. Fix: relabel L1 → HOLD (0) bukan EXIT (2). Dengan
constraint ini, semua EXIT label di training dijamin `cur_pnl >= 0` (M2/M6 sudah enforce `> 0.003`).
Model tidak lagi belajar "EXIT saat rugi" → GUARDIAN_EXIT inference hanya terpicu di bar positif.

### Perubahan vs v6 (satu-satunya perubahan)

```python
# v6: L1 → EXIT (2) saat current_pnl < -0.006
# v7: L1 → HOLD (0) — model tidak lagi cut losses early
if (current_pnl < LOSS_EXIT_THR and bars_held >= LOSS_EXIT_BARS
        and best_future_pnl < 0.002 and not mom_strong):
    return 0  # HOLD, bukan EXIT
```

### Label distribution

| Versi | HOLD | PARTIAL | EXIT | EXIT neg PnL |
|-------|------|---------|------|--------------|
| v6 | 397,095 (66.6%) | 754 | 198,488 (33.3%) | >0% |
| **v7** | **511,536 (85.8%)** | 754 | **84,047 (14.1%)** | **0.0%** |

L1_relabeled_hold: ~5,000–7,700 per koin (total ~115,000 sampel pindah dari EXIT → HOLD).

### CV Metrics

| Metrik | v6 | **v7** |
|--------|-----|--------|
| CV logloss | 0.470 | **0.3224** |
| CV F1 macro | 0.621 | **0.6013** |

Logloss v7 jauh lebih rendah (model lebih confident) meski F1 sedikit turun — konsisten dengan
distribusi label lebih clean (EXIT hanya positif).

### OOF Scorecard (default params: exit_thr=0.65, floor=0.70, min_hold=0)

Entry signals: **39,833** bar-jam · modal $10 · leverage 5×

| Guardian | Trades | WR | PF | PnL | PnL/trade |
|----------|--------|-----|-----|------|-----------|
| **guard_k5mom_v7** | **18,256** | **65.59%** | **2.054** | **$8,113.94** | **$0.445** |
| guard_k5mom_v6 | ~19,984 | 52.0% | 1.684 | $5,947 | $0.298 |
| guard_k5mom_v2 (binary) | 17,629 | 55.0% | 1.731 | $6,054 | $0.343 |
| *guard_momentum_v2* (unfair) | 19,449 | 70.2% | 1.782 | $5,669 | $0.291 |
| Tanpa guardian | 17,645 | 55.6% | 1.711 | $5,881 | $0.333 |

**Sweet spot (WR≥62%, PnL≥$5k, PF≥1.60): v7 LOLOS SEMUA ✅**

| Kriteria | Target | v7 | Status |
|----------|--------|----|--------|
| WR | ≥62% | **65.59%** | ✅ |
| PnL | ≥$5,000 | **$8,113.94** | ✅ |
| PF | ≥1.60 | **2.054** | ✅ |

### Analisis

v7 adalah guardian pertama yang **melampaui momentum_v2 di semua dimensi fair** (no p_bull):
- WR 65.59% vs momentum_v2 70.2% (gap 4.6pp vs sebelumnya gap 18pp pada v6)
- PF 2.054 vs momentum_v2 1.782 — v7 **lebih tinggi PF** meski WR sedikit di bawah
- PnL $8,114 vs momentum_v2 $5,669 — **+$2,445 lebih**

Root cause fix: dengan menghilangkan L1_loss_exit dari label EXIT, model belajar bahwa
"EXIT = puncak momentum" bukan "EXIT = cut loss". Pada inference, GUARDIAN_EXIT hanya terpicu
saat model confident di peak positif → WR naik drastis (estimasi dari 22% → >65%).

Tradeoff yang diterima: trade yang seharusnya di-cut early (L1 kondisi) sekarang lanjut ke SL.
Net effect: lebih sedikit micro-loss-cut, tapi exit pada momentum peak lebih akurat → PnL naik.

### Artefak

| Path | Isi |
|------|-----|
| `pipeline/model/experiments/train_guardian_k5mom_v7.py` | Train v7 |
| `models/runs/ic32_rv2_guard_k5mom_v7/` | Model artefak |
| `models/runs/ic32_rv2_lgbm_mkt_sync_v2/oof_k5mom_v7_compare.json` | OOF comparison |

### Langkah berikutnya

- [ ] Sweep params v7 (exit_thr × floor × min_hold) untuk cari titik optimal
- [ ] Pseudo-holdout v7 (2025-10-01 s/d 2026-03-31) untuk verifikasi
- [ ] Pertimbangkan deploy jika pseudo-holdout konsisten

---

## 2026-07-01 — Ringkasan: Guardian vs stack **k5_mom embedded** (kesalahan & hasil)

**Status**: COMPLETE — tidak ada kandidat sweet spot; tidak deploy

**Stack tetap**: `LGBM (ic32_rv2_lgbm_mkt_sync_v2) + HMM + k5_mom (K5_MOM_TR1, btc_mom=2%)`  
**Constraint**: single guardian, **no p_bull**, no VPS deploy tanpa approval  
**Target gagal**: WR ≥ 62%, PnL ≥ $5,000, PF ≥ 1.60 (sweet spot) — **0 config lolos** di semua fase

### Kesalahan metodologi & eksekusi

| # | Kesalahan | Dampak | Koreksi |
|---|-----------|--------|---------|
| K1 | **Bandingkan WR `momentum_v2` (live) dengan guardian fair no-p_bull** | Ekspektasi WR ~70% pada v6/v2/v3 — tidak apples-to-apples | Audit trade-level: WR gap = kualitas pre-TP `GUARDIAN_EXIT` (100% vs 22%), bukan entry k5 |
| K2 | **Anggap `p_bull=0.5` penyebab WR rendah** | Waktu terbuang tuning p_bull | Ablation: delta PnL kecil; `daily_pbull` OOF **0% hit** — bukan driver utama |
| K3 | **Target sweet WR 62% + PnL $5k simultan** | 216+ sweep configs, semua gagal | Kurva Pareto: WR↑ hanya dengan PnL↓ pada guardian fair |
| K4 | **Retrain v5 Opsi A (TP-phase 3-class)** tanpa validasi label horizon dulu | CV F1 0.866 menipu; OOF PnL **$4,571** | Buang v5; label `bar_out` bug di run pertama |
| K5 | **Hybrid dual guardian** (`momentum_v2` pre + `k5mom_v2` peak) | Melanggar constraint single guardian + masih pakai pre-p_bull | Ditolak user; PnL hybrid $5,616 < v2 solo $6,054 |
| K6 | **Pre-TP emergency `conf_delta`** pada v3 | WR **14.5%**, PnL $2.3k — conf hourly turun alami pasca-entry k5 | Fase emergency **GAGAL** 0/48 |
| K7 | **Phase 2 sweep 768 configs** + `ProcessPoolExecutor` Windows | Macet ~22 menit tanpa output; 33 config terbuang | Grid dipangkas ke 162; sequential mode |
| K8 | **`min_hold` 0/2/4** dalam grid besar padahal 0≈2 identik | ~3× runtime redundan | Phase 2: hanya `min_hold` [0, 2] |
| K9 | **LGBM score guardian v1** — exit tiap bar tanpa gate | WR tampak **0%** (bug) + PnL $459 | Fix `guardian_outcomes`; v2 rules tetap gagal |
| K10 | **Confuse live WR tinggi dengan edge buffer k5 fair** | Salah interpretasi hasil sweep | Live = `momentum_v2` + HMM entry + pre-TP micro-win; bukan template no-p_bull |

### Hasil guardian pada entry k5_mom (OOF `< 2026-04-01`)

Entry signals: **39,833** bar-jam · modal $10 · leverage 5×

#### A. Retrain no-p_bull (default OOF params kecuali dinyatakan)

| Guardian | Arsitektur | Train entry | Trades | WR | PF | PnL | Verdict |
|----------|------------|-------------|--------|-----|-----|------|---------|
| **k5mom_v2** | binary peak escort | k5_mom | 17,629 | **55.0%** | **1.731** | **$6,054** | **terbaik PnL** no-p_bull |
| **k5mom_v6** | 3-class + label_end fix | k5_mom | 19,984 | 52.0% | 1.684 | **$5,947** | terbaik balance PnL vs fair |
| k5mom_v3 | 3-class momentum | k5_mom | 21,580 | **57.6%** | 1.490 | $3,871 | WR ok, PnL lemah |
| k5mom_v5 | 3-class TP-phase | k5_mom | 18,287 | 52.6% | 1.507 | $4,571 | **BUANG** |
| k5mom_v1 | 3-class (awal) | k5_mom | 21,426 | 56.6% | 1.507 | $4,042 | **BUANG** |
| *momentum_v2* | 3-class + p_bull slot | HMM/LSTM path | 19,616 | *70.9%* | 1.768 | $5,496 | *bukan fair ablation* |

#### B. Param sweep (guardian sudah ada, tanpa retrain)

| Fase | Grid | Sweet | Best result | Output |
|------|------|-------|-------------|--------|
| post-TP thr×floor (v2) | 30 | 0/30 | PnL **$6,109** (thr=0.80 floor=0.40), WR 54.2% | `k5mom_post_tp_sweep.json` |
| min_hold×floor (v6/v3/v2/momv2) | 32 | — | v6 mh=0 floor=0.2: PnL $6,617 WR 49.5% | `k5mom_min_hold_sweep.json` |
| **sweet-spot Phase 1** | 216 | **0/216** | Best PnL v6: **$7,080** WR 47.6% (thr=0.70 floor=0.20) | `k5mom_guard_sweetspot_sweep.json` |
| sweet-spot Phase 2 pre-TP gate (v6) | 162 planned | **dibatalkan** @24/162 | pre-gate tidak ubah trade-off WR↔PnL | `k5mom_guard_sweetspot_progress.json` |
| hybrid momv2+k5v2 | 16 | 0/16 | PnL $5,616 WR 71.0% — ditolak (dual + p_bull pre) | `k5mom_hybrid_sweep.json` |
| emergency pre-TP (v3) | 48 | 0/48 | WR **14.5%** | `k5mom_emergency_pre_tp_sweep.json` |
| **LGBM score guardian** (rule, no ML) | 27 | 0/27 | vs no-guard WR 55.6% PnL $5,881 → best rule WR 55.8% PnL **$1,522** | `lgbm_score_guardian_sweep.json` |

#### C. Tanpa guardian (baseline k5_mom)

| Mode | Trades | WR | PF | PnL |
|------|--------|-----|-----|------|
| SL/TP swing saja | 17,645 | 55.6% | 1.711 | $5,881 |

Guardian ML fair **tidak mengalahkan** baseline WR; hanya v2/v6 menambah PnL dengan WR sedikit lebih rendah.

### Akar masalah WR rendah (guardian fair)

1. **`wr_strict_tp = 0%`** — tidak ada win dari swing TP keras; WR = kualitas exit guardian.
2. **Pre-TP `GUARDIAN_EXIT` v6: WR 22%** vs momentum_v2: **100%** (micro-profit) — mekanisme yang sama dengan gap live vs eksperimen.
3. **Tuning threshold/floor/min_hold/pre-TP gate** hanya geser kurva Pareto — tidak memperbaiki label/model pre-TP.

### Keputusan

| Item | Status |
|------|--------|
| Deploy guardian baru ke VPS | **TIDAK** |
| `guard_momentum_v2` pada k5 entry (WR 70%, ada p_bull) | **DITOLAK** user |
| Hybrid dual guardian | **DITOLAK** |
| `guard_k5mom_v5` | **BUANG** |
| LGBM score rule guardian | **BUANG** — churn + PnL collapse |
| Kandidat paper no-p_bull | **`guard_k5mom_v2`** PnL $6,054 atau **v6** $5,947; params opsional thr=0.80 floor=0.40 (+$56) |
| Jalur WR buffer tanpa p_bull | **Belum ada** — butuh **v7 retrain** (pre-TP EXIT label hanya jika `cur_pnl ≥ 0`) atau entry filter |

### Artefak & script

| Path | Isi |
|------|-----|
| `pipeline/model/experiments/train_guardian_k5mom_v6.py` | Train v6 |
| `tools/model/sweep_k5mom_guard_sweetspot.py` | Sweet-spot Phase 1–2 |
| `tools/model/sim_lgbm_score_guardian_oof.py` | LGBM score rule sim |
| `tools/model/audit_guardian_momentum_v2_wr.py` | Forensics WR momentum_v2 |
| `core/lgbm_score_guardian.py` | Rule exit (experimental) |
| `core/evaluator.py` | `guardian_lgbm_score_enabled`, `guardian_pre_exit_min_pnl` |

---

## 2026-07-01 — Audit momentum_v2 WR (trade-level forensics)

**Status**: COMPLETE — root cause identified

**Stack audit**: OOF `< 2026-04-01` + live parity params | k5_mom + HMM baseline

### Hipotesis hasil

| ID | Hipotesis | Verdict |
|----|-----------|---------|
| H1 | WR inflated by partial half-TP | **REJECTED** — `partial_rate` 0.14% mom_v2 vs 0.24% v6; gap WR−WR_noP hanya **0.08pp** |
| H2 | Class weight 20× over-trigger PARTIAL | **REJECTED** — hampir tidak ada partial di inference |
| H3 | Pre-TP exit quality beda model | **CONFIRMED** — lihat histogram di bawah |
| H4 | live parity = OOF legacy | **REJECTED** — live floor=0.2 mengubah angka (v6 live PnL $7,075) |
| H5 | daily_pbull cover OOF | **CONFIRMED** — **0% hit**, 100% fallback 0.5 |

### OOF legacy (`mh=0 floor=0.7 pbull=0.5`)

| Guardian | Trades | WR std | WR no-partial | partial% | PnL |
|----------|--------|--------|---------------|----------|-----|
| momentum_v2 | 19,616 | **70.9%** | 70.8% | 0.14% | $5,496 |
| k5mom_v6 | 19,984 | 52.0% | 51.9% | 0.24% | **$5,947** |
| k5mom_v2 binary | 17,629 | 55.0% | 55.0% | 0% | $6,054 |

**`wr_strict_tp` = 0%** untuk semua run — tidak ada hard-TP WIN (momentum mode).

### Outcome histogram — penyebab WR gap (OOF legacy)

| Outcome | mom_v2 count | mom_v2 WR | v6 count | v6 WR |
|---------|-------------|-----------|----------|-------|
| **GUARDIAN_EXIT** (pre-TP) | **10,217** | **100%** | 8,793 | **22.2%** |
| GUARDIAN_MOMENTUM_EXIT | 2,122 | 100% | 1,795 | 100% |
| GUARDIAN_MOMENTUM_FLOOR | 1,623 | 96.1% | 6,770 | 98.0% |
| LOSS | 5,153 | 0% | 2,586 | 0% |

**Kesimpulan**: WR 71% momentum_v2 **bukan artefak partial/p_bull konstan**. Model pre-TP exit (`GUARDIAN_EXIT`) hampir selalu keluar dengan `net_pnl > 0` (10,217/10,217). v6 pada bucket yang sama **78% rugi** → WR rendah meski PnL total lebih tinggi (floor escort + fewer SL).

### Train/sim mismatch (dokumentasi)

| | momentum_v2 train | Sim OOF |
|--|-------------------|---------|
| LGBM | `ic32_regime_v2_parity` | `ic32_rv2_lgbm_mkt_sync_v2` |
| Entry labels | LSTM thr + p_bull | k5_mom |
| PARTIAL labels | 38 / 71,547 | — |

### Rekomendasi

1. **Jangan bandingkan WR** momentum_v2 vs v6 sebagai satu metrik — definisi win sama tapi **distribusi outcome berbeda radical**.
2. **Metrik utama paper no-p_bull**: PnL + PF (`guard_k5mom_v6` atau v2 binary).
3. Jika butuh WR tinggi tanpa p_bull: perlu retrain pre-TP exit agar `GUARDIAN_EXIT` tidak 22% WR (v6) — bukan tambah p_bull konstan.
4. Fix `daily_pbull` coverage untuk OOF/live parity test yang meaningful.

Script: `tools/model/audit_guardian_momentum_v2_wr.py`  
Output: `models/runs/ic32_rv2_lgbm_mkt_sync_v2/audit_momentum_v2_wr.json`, `audit_momentum_v2_trades_sample.csv`

---

## 2026-07-01 — Guardian k5_mom v6 (fair ablation no p_bull, label_end fix)

**Status**: OOF VALIDATED — lokal only

**Stack**: `LGBM + HMM + k5_mom + guard_k5mom_v6`

| Komponen | Detail |
|----------|--------|
| Entry | k5_mom_tr1 |
| Labeling | momentum_escort_v2 **tanpa** cabang LSTM (sama spirit v3) |
| Horizon | `label_end = min(bar_in+MAX_HOLD, n)` — fix post-TP samples (vs v3 `bar_out`) |
| Fitur | **30**, **no p_bull** |
| Samples | 596,337 — HOLD=397,095 PARTIAL=754 EXIT=198,488 |
| CV F1 macro | **0.621** (logloss 0.470) |

**OOF** (`min_hold=0`, `floor=0.70`, `exit=0.65`):

| Guardian | Trades | WR | PF | PnL |
|----------|--------|-----|-----|------|
| **guard_k5mom_v6** | 19,984 | 52.0% | **1.684** | **$5,947** |
| guard_k5mom_v2 binary | 17,629 | 55.0% | 1.731 | $6,054 |
| guard_momentum_v2 (+p_bull slot) | 19,616 | 70.9% | 1.768 | $5,496 |
| guard_k5mom_v3 | 21,580 | 57.6% | 1.490 | $3,871 |

v6: PnL hampir v2 binary (−$107), **+$451 vs momentum_v2** tanpa p_bull; WR rendah (52%) karena tanpa partial inflation.

Script: `pipeline/model/experiments/train_guardian_k5mom_v6.py`  
OOF JSON: `models/runs/ic32_rv2_lgbm_mkt_sync_v2/oof_k5_mom_guard_k5mom_v6.json`

---

## 2026-07-01 — Guardian k5_mom v5 retrain (Opsi A: TP-phase 3-class, no p_bull)

**Status**: OOF VALIDATED — lokal only, belum deploy VPS

**Stack**: `LGBM + HMM + k5_mom + guard_k5mom_v5`

| Komponen | Detail |
|----------|--------|
| Entry | k5_mom_tr1 (top_k=5, floor=0.52, BTC mom 2%) |
| Exit | **Opsi A** — single 3-class HOLD / **PARTIAL** / EXIT + fitur `tp_phase` (0/1/2) |
| Labeling | PRE=conservative hold · AT=99% PARTIAL · POST=peak escort (M2/M6) |
| Fitur | **31** (21 static + 10 dynamic incl. `tp_phase`), **tanpa p_bull** |
| Train samples | 599,357 — HOLD=418,367 PARTIAL=12,298 EXIT=168,692 |
| Phase split | PRE 343k / AT 10.5k / POST 245k |
| CV F1 macro | **0.866** (logloss 0.288) |
| Class weights | HOLD=1, PARTIAL=8, EXIT=2 |

**Bug fix**: label horizon pakai `min(bar_in + MAX_HOLDING_BARS, n)` bukan `bar_out` — run pertama hanya PRE (AT=0, POST=0).

**OOF trade sim** (`< 2026-04-01`, default `min_hold=4`, `floor=0.70`):

| Guardian | Trades | WR | PF | PnL |
|----------|--------|-----|-----|------|
| **guard_k5mom_v5** | 18,287 | 52.6% | 1.507 | **$4,571** |
| guard_k5mom_v3 | 21,426 | 56.6% | 1.507 | $4,042 |
| guard_k5mom_v2 (binary) | 17,629 | 55.0% | 1.731 | $6,054 |

**Kesimpulan**: v5 +$529 PnL vs v3 pada default OOF params; WR lebih rendah (52.6% vs 56.6%) karena lebih sedikit trade + skema half-TP. Binary v2 masih unggul PnL tapi tanpa PARTIAL escort. **Tidak pakai p_bull** — memenuhi constraint v6.

Script: `pipeline/model/experiments/train_guardian_k5mom_v5.py`  
Output: `models/runs/ic32_rv2_guard_k5mom_v5/`  
OOF JSON: `models/runs/ic32_rv2_lgbm_mkt_sync_v2/oof_k5_mom_guardian.json`

---

## 2026-07-01 — Guardian k5_mom v3 retrain (3-class, no p_bull)

**Status**: OOF VALIDATED — lokal only, belum deploy VPS

**Stack**: `LGBM + HMM + k5_mom + guard_k5mom_v3`

| Komponen | Detail |
|----------|--------|
| Entry | k5_mom_tr1 (top_k=5, floor=0.52, BTC mom 2%) |
| Exit | 3-class momentum escort v2 — HOLD / **PARTIAL (half TP)** / EXIT |
| Fitur | **30** (21 static + 9 dynamic), **tanpa p_bull**, tanpa LSTM |
| Train samples | 223,488 — HOLD=170,489 PARTIAL=146 EXIT=52,853 |
| CV F1 macro | **0.593** (logloss 0.379) |

**OOF trade sim** (`< 2026-04-01`):

| Config | min_hold | floor | Trades | WR | PF | PnL |
|--------|----------|-------|--------|-----|-----|------|
| oof_default | 4 | 0.70 | 21,426 | 56.6% | 1.507 | $4,042 |
| live_parity | 2 | 0.20 | 21,498 | 57.1% | 1.499 | $3,942 |

**Bandingkan** `LGBM + HMM + k5_mom + guard_momentum_v2` (31f + p_bull slot): WR **70.2%**, PnL $5,545 — delta WR besar, tapi stack tidak self-contained (fitur dari model lain).

Script: `pipeline/model/experiments/train_guardian_k5mom_v3.py`  
Output: `models/runs/ic32_rv2_guard_k5mom_v3/`  
OOF JSON: `models/runs/ic32_rv2_lgbm_mkt_sync_v2/oof_k5_mom_guard_k5mom_v3.json`

---

## 2026-07-01 — Guardian momentum_v2 live parity re-sim

**Status**: OOF VALIDATED — tidak deploy

Stack: `LGBM sync_v2 + HMM + guard_momentum_v2` dengan artefak deploy (`md5=24543b94dfbd`, 31 feat).

| Entry | p_bull | min_hold | floor | Trades | WR | PF | PnL |
|-------|--------|----------|-------|--------|-----|-----|------|
| **HMM baseline** | daily_pbull | **2 (live)** | 0.20 | 17,839 | **68.9%** | 1.623 | **$4,250** |
| HMM baseline | 0.5 legacy | 4 | 0.70 | 17,755 | 68.9% | 1.634 | $4,379 |
| **k5_mom** | daily_pbull | **2 (live)** | 0.20 | 19,544 | **70.2%** | 1.774 | **$5,545** |
| k5_mom | 0.5 legacy | 4 | 0.70 | 19,449 | 70.2% | 1.782 | $5,669 |

**min_hold sweep** (daily_pbull, floor=0.20): `min_hold` **0 / 1 / 2 identik** — Guardian pre-TP jarang memicu sebelum bar 2; perbedaan baru terlihat di `min_hold=4` (k5: WR 69.6%, PnL $5,730).

**Kesimpulan**:
- `daily_pbull` vs `p_bull=0.5` → WR hampir sama; delta PnL kecil (−$125 k5, −$129 HMM) vs legacy floor=0.7
- Angka k5_mom WR **70.2%** robust — bukan artefak `p_bull` salah semata; **`floor` + `min_hold`** yang lebih berpengaruh
- Live default `min_hold=2` ≈ `min_hold=0` pada data ini

Script: `tools/model/run_guard_momentum_live_parity.py`  
Output: `models/runs/ic32_rv2_lgbm_mkt_sync_v2/guard_momentum_live_parity_sweep.json`

---

## 2026-06-30 — Guardian k5_mom v2 (binary peak escort)

**Status**: OOF VALIDATED — belum deploy

### Stack lengkap (entry + exit)

**Entry** — `LGBM + HMM + k5_mom` (tanpa LSTM, tanpa p_bull):

| Layer | Komponen | Run / config |
|-------|----------|--------------|
| LGBM | 3-class OOF proba p0/p2 | `ic32_rv2_lgbm_mkt_sync_v2` |
| HMM | Per-state threshold gate | `REGIME_THR` (bull/bear/sideways) |
| k5_mom | Cross-section top-K + BTC momentum day | `K5_MOM_TR1`: top_k=5, floor=0.52, require_trigger=True, BTC mom **2%** |

**Exit** — Guardian (varian dibandingkan di bawah):

| Guardian run | Arsitektur | Train entry path | Catatan |
|--------------|------------|------------------|---------|
| `ic32_rv2_guard_momentum_v2` | 3-class momentum escort | HMM baseline (bukan k5) | Sim pakai p_bull=0.5 netral |
| `ic32_rv2_guard_k5mom_v1` | 3-class momentum escort | k5_mom | **GAGAL** — exit terlalu agresif, PARTIAL noise |
| `ic32_rv2_guard_k5mom_v2` | **binary peak escort** | k5_mom | Train peak zone only (pnl>0.3%, pnl≥80% MFE), post-TP inference |

### OOF scorecard — stack lengkap `< 2026-04-01`

Modal $10/trade, leverage 5×. Entry signals k5_mom: **39,833** bar-jam.

| Stack entry + exit | Trades | WR | PF | PnL | PnL/trade |
|--------------------|--------|-----|-----|------|-----------|
| LGBM + HMM (baseline) + `guard_momentum_v2` | 17,755 | 68.9% | 1.634 | $4,379 | $0.247 |
| **LGBM + HMM + k5_mom + `guard_momentum_v2`** | **19,449** | **70.2%** | **1.782** | **$5,669** | **$0.291** |
| LGBM + HMM + k5_mom + `guard_k5mom_v1` | 21,426 | 56.6% | 1.507 | $4,042 | $0.189 |
| **LGBM + HMM + k5_mom + `guard_k5mom_v2`** | **17,629** | **55.0%** | **1.731** | **$6,054** | **$0.343** |

Delta vs baseline HMM + guard_v2: k5_mom + guard_v2 → +1,694 trades, PF +0.15, PnL **+$1,290**.

Delta vs k5_mom + guard_v2: k5_mom + guard_k5mom_v2 → −1,820 trades, PF −0.05, PnL **+$385** (lebih sedikit trade, PnL lebih tinggi).

### Pseudo-holdout — `2025-10-01` s/d `2026-03-31` (bukan holdout tersegel)

k5_mom entry 2%, signals **3,657** bar-jam.

| Stack entry + exit | Trades | WR | PF | PnL |
|--------------------|--------|-----|-----|------|
| LGBM + HMM + k5_mom + `guard_momentum_v2` | 1,936 | 66.8% | 1.544 | $368 |
| LGBM + HMM + k5_mom + `guard_k5mom_v1` | 2,129 | 54.4% | 1.291 | $210 |
| **LGBM + HMM + k5_mom + `guard_k5mom_v2`** | **1,809** | **53.7%** | **1.560** | **$426** |

### Train Guardian k5mom_v2

| Metrik | Nilai |
|--------|-------|
| Samples raw (k5 entry path) | 223,342 |
| Samples peak zone (train) | 51,022 (HOLD 34,970 / EXIT 16,052) |
| CV AUC | 0.860 |
| CV logloss | 0.444 |
| Fitur | 30 (9 dynamic + 21 static, no p_bull) |

Script: `pipeline/model/experiments/train_guardian_k5mom_v2.py`  
Output: `models/runs/ic32_rv2_guard_k5mom_v2/`  
OOF JSON: `models/runs/ic32_rv2_lgbm_mkt_sync_v2/oof_k5_mom_guardian.json`

### Keputusan sementara

- **Production tetap** `ic32_regime_v5` (belum deploy k5_mom).
- Kandidat paper: **LGBM + HMM + k5_mom + guard_k5mom_v2** — PnL OOF tertinggi ($6,054), PF solid (1.73).
- Alternatif konservatif: tetap `guard_momentum_v2` pada entry k5_mom (WR lebih tinggi, PF sedikit lebih baik).
- v1 dibuang — arsitektur 3-class pada path k5 salah.

### Catatan runtime sim

Satu OOF full-stack ~5–10 menit (21 koin × `simulate_trades_swing` bar-by-bar + Guardian). Perbandingan 3 Guardian ≈ 3× waktu. Setelah bundle cache + parallel sweep config: ~5 min / 48 config.

### Fase 1 — pre-TP emergency sweep (2026-07-01) GAGAL

Stack: `LGBM + HMM + k5_mom + guard_k5mom_v3` + emergency `lgbm_conf_delta`.

Grid ketat 48 config: `conf_delta` −0.14..−0.20, near −0.20..−0.28, prox 0.85/0.90/0.95.

| Stack entry + exit | Trades | WR | PF | PnL |
|--------------------|--------|-----|-----|------|
| post-TP only (baseline) | 17,628 | 55.0% | 1.732 | $6,065 |
| + emergency (semua grid) | ~22,700–22,900 | **14.5–14.7%** | 1.35–1.36 | $2,296–$2,358 |

**Pass 0/48** (WR≥60%, PF≥1.70, PnL≥$5,800).

Diagnosa: `conf_delta` turun alami setelah entry (conf rank k5 tinggi, conf hourly lebih rendah) — bukan reversal. Emergency → exit dini → re-entry → trades naik, WR jeblok.

Output: `models/runs/ic32_rv2_lgbm_mkt_sync_v2/k5mom_emergency_pre_tp_sweep.json`

### Fase 3 — post-TP threshold/floor sweep (2026-07-01)

Stack: `LGBM + HMM + k5_mom + guard_k5mom_v2` (tanpa pre-TP emergency).

Grid: `exit_thr` 0.55–0.80 × `floor_frac` 0.40–0.80 (30 combo). Runtime ~105s (6 workers).

| Config (thr / floor) | Trades | WR | PF | PnL | vs default |
|----------------------|--------|-----|-----|------|------------|
| **default 0.65 / 0.70** | 17,629 | 55.0% | 1.731 | $6,054 | — |
| **best PnL 0.80 / 0.40** | 17,481 | 54.2% | **1.739** | **$6,109** | +$56, −148 trades |
| runner-up 0.80 / 0.50 | 17,555 | 54.6% | 1.737 | $6,106 | +$52 |
| runner-up 0.55 / 0.40 | 17,598 | 54.8% | 1.735 | $6,092 | +$38 |

**Pass 0/30** sweet (WR≥58%, PF≥1.65, PnL≥$5,500) — semua lolos PF+PnL, **WR tetap ~54–55%** di seluruh grid.

Diagnosa: tuning thr/floor hanya geser trade-off PnL↔trades; WR rendah berasal dari arsitektur binary peak escort (post-TP only), bukan dari default threshold. `floor_frac` rendah (0.40) sedikit naikkan PnL; `exit_thr` tinggi (0.80) sedikit naikkan PF.

Output: `models/runs/ic32_rv2_lgbm_mkt_sync_v2/k5mom_post_tp_sweep.json`

**Rekomendasi sementara**: jika tetap `guard_k5mom_v2`, kandidat params **thr=0.80 floor=0.40** (+$56 PnL OOF, PF 1.739). Untuk WR ~70%, tetap `guard_momentum_v2` pada entry k5_mom.

### Fase 4 — Hybrid dual Guardian (2026-07-01)

Arsitektur: **pre-TP 3-class** + **post-TP binary peak** (`guardian_hybrid_dual` di `evaluator.py`).

| Variant pre / peak | Trades | WR | PF | PnL | Catatan |
|--------------------|--------|-----|-----|------|---------|
| v1 pre + v2 peak (smoke) | 21,427 | 56.7% | 1.507 | $4,044 | ≈ v1 saja — pre dominan |
| v4 pre + v2 peak (retrain pre-zone) | 19,857 | **46.2%** | 1.463 | $4,209 | pre-zone train **gagal** |
| **momentum_v2 pre + k5mom_v2 peak** | **19,766** | **71.2%** | **1.785** | **$5,589** | **kandidat hybrid terbaik** |
| Ref: momentum_v2 only (k5 entry) | 19,449 | 70.2% | 1.782 | $5,669 | baseline WR |
| Ref: k5mom_v2 only | 17,629 | 55.0% | 1.731 | $6,054 | baseline PnL |

Sweep post-TP (momentum_v2 pre fixed, peak thr × floor): **best PnL** peak=0.65 floor=0.40 → **$5,616** WR=71.0% PF=1.789. **0/16 sweet** (PnL≥$5,800).

**Keputusan**: Hybrid `momentum_v2` pre + `k5mom_v2` peak naikkan WR (+1pp vs momentum_v2 solo) tapi **PnL belum menyalip k5mom_v2** (−$465 vs $6,054). Retrain pre k5-only (v4) tidak membantu.

Output:
- `models/runs/ic32_rv2_guard_k5mom_v4/` (pre specialist — tidak dipakai)
- `models/runs/ic32_rv2_lgbm_mkt_sync_v2/oof_k5_mom_hybrid.json`
- `models/runs/ic32_rv2_lgbm_mkt_sync_v2/k5mom_hybrid_sweep.json`

Script: `tools/model/run_k5mom_hybrid.py`, `sweep_k5mom_hybrid.py`, `train_guardian_k5mom_v4.py`

**Next**: pseudo-holdout hybrid terbaik (momv2 pre + k5v2 peak, peak_thr=0.65 floor=0.40) jika disetujui.

---

## 2026-06-27 — ic32.rv5.coin_fs.v1

**Status**: IN_PROGRESS

### Hipotesis
Saat momentum market luas, hanya 1-2 coin dapat sinyal karena (a) gate HMM/threshold terlalu ketat per coin, atau (b) LGBM conf alt/meme tidak tembus — bukan karena meme merusak pool (PEPE/SHIB OOF PF kuat).

Campur BTC+alt+meme dalam satu pool berisiko jika ranking fitur IC divergen antar tier (>30%). Urutan: diagnostik funnel -> D (tier threshold) -> C (coin meta) -> B (cluster).

### Yang Diubah
- Arsip tersegel: `archive/ic32_regime_v4_baseline/`
- Pipeline eksperimen: `pipeline/experiments/ic32_rv5_coin_fs/` (00-04)
- Tier coin T1/T2/T3 di `config_rv5.json`
- Fetch kandidat coin baru (Binance top volume) — belum auto-include

### Target
- Signal rate naik (lebih banyak coin aktif per hari momentum)
- PF OOF >= 1.2, WR tidak turun >3pp vs v4
- Evaluasi: OOF + live forward (holdout Apr-Jun terkontaminasi)

### Script
```powershell
python pipeline/12_rv5_coin_fs.py 00   # fetch candidates
python pipeline/12_rv5_coin_fs.py 01   # coin profile
python pipeline/12_rv5_coin_fs.py 02   # signal funnel
python pipeline/12_rv5_coin_fs.py 03   # feature IC per tier
python pipeline/12_rv5_coin_fs.py 04   # select universe
```

### Hasil Diagnostik Awal (2026-06-27)

| Tahap | Temuan |
|-------|--------|
| Coin profile | Signal rate OOF merata ~29-43/1k bars semua tier; meme TIDAK dominan rendah |
| Funnel recent 90d | **verdict_hint = LGBM_confidence** — p2_p90 ~0.43-0.49, di bawah thr long HMM 0.60 |
| Funnel | hmm_killed = 0 di semua coin — HMM gate BUKAN bottleneck di OOF/recent |
| Live vs OOF | Gejala 1-2 coin live kemungkinan conf live lebih rendah lagi — perlu banding live signals |

**Keputusan sementara**: skip pendekatan D (tier threshold) sebagai prioritas; fokus **C (coin meta + retrain)** atau turunkan base thr di OOF sweep rv5.

### Hasil Eksekusi (2026-06-27)

| Step | Hasil |
|------|-------|
| 00b meta | 4 kolom ditambah ke 21 parquet |
| 03 IC gate | Meta baru **GAGAL** IC/ICIR/stab — tidak masuk training |
| features_rv5 | **13 fitur** (marginal greedy dari 33f baseline) |
| 04t retrain | OOF 765k bars, thr 0.75/0.70 |
| 05 eval | PF **1.304** PASS; density median **3 coin/hari** FAIL (target 5) |
| 02b live | median **20 coin/hari** (55 hari cache) — gejala 1-2 coin mungkin bukan generasi sinyal |

**Status**: IN_PROGRESS

### Revisi seleksi fitur (2026-06-27 pm)

IC ketat (13f) **salah acuan** — user: gate = **OOF scorecard** (density momentum + PF + PnL), IC rendah OK.

| Model | Fitur | Cascade flat 0.65 | Trades | PF | PnL @$10 | Coin/hari (momentum) |
|-------|-------|-------------------|--------|-----|----------|----------------------|
| rv5 strict | 13 | 0.65/0.65 | 3,164 | 1.28 | $484 | **2** |
| rv5 relaxed | **37** | 0.65/0.65 | 10,373 | 1.32 | **$1,857** | **6** |
| v4 baseline OOF | 33 | 0.65/0.65 | 12,959 | 1.35 | $2,512 | **7** |

Relaxed 37f lolos target density (>=5). Deploy ditunda — bandingkan live forward vs v4.

Output: `models/runs/ic32_rv5_coin_fs_v1/`

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

## 2026-06-23 — Holdout: LGBM Parity + Guardian Momentum Escort v2

**Status**: COMPLETED ✓ — HOLDOUT_EVALUATED=True, amplop dikunci

### Hipotesis
Guardian v2 (momentum escort labeling) akan meningkatkan WR holdout vs Guardian lama
(profit-lock labeling) karena model belajar "kawal momentum, exit saat peak" bukan "lock profit awal".
Baseline holdout dengan Guardian lama: 56 trades, WR 41.1%, PF 0.599, PnL -$5.34.

### Yang Diubah vs Baseline
- **Guardian lama**: `ic32_rv2_parity_guard_oof` (profit-lock, exit_thr=0.90)
- **Guardian v2**: `ic32_rv2_guard_momentum_v2` (momentum escort, exit_thr=0.65, floor=0.20)
  - Labeling: EXIT hanya di momentum peak (≥88% best_future), HOLD selama pullback
  - Fitur tambahan: `p_bull` dari LSTM daily (nilai real dari inference pada holdout)
  - Class weights: EXIT=5×, PARTIAL=20×
- **LSTM threshold adj**: alpha=0.10, zone=0.12 — coinank OI/LS s.d. Jun 7, macro ffill; ETF=0.0
- **No HMM filter** pada SHORT threshold (parity model tidak pakai filter ini)
- OOF CV F1 Guardian v2: 0.4945
- Sweep OOF: floor=0.20, exit_thr=0.65 → PnL=$657.51, PF=1.586

### Hasil Holdout (Apr 1 – Jun 22, 2026)

| Metrik | No Guardian | +Guardian v2 | vs Baseline Guardian lama |
|--------|-------------|--------------|---------------------------|
| Trades | 266 | 275 | baseline=56 |
| WR | 47.0% | **68.7%** | baseline=41.1% |
| PF | 1.171 | **1.426** | baseline=0.599 |
| PnL | +$10.68 | **+$17.68** | baseline=-$5.34 |
| PnL/Month | +$3.96 | +$6.56 | - |
| MaxDD | -$5.55 | -$3.44 | - |

Exit breakdown (dengan Guardian): GUARDIAN_EXIT=151, LOSS=75, GUARDIAN_MOMENTUM_EXIT=34, TIMEOUT=8, FLOOR=7

**Catatan**: LSTM adjustment menghasilkan 5× lebih banyak trade (266 vs 56 tanpa LSTM).
Pasar Apr-Jun 2026 dominan SHORT (265/266 = 99.6% SHORT trades) — LSTM deteksi bear bias
→ turunkan thr_short secara simetris → lebih banyak sinyal SHORT.

### Kesimpulan
✅ **Hipotesis terbukti**: Guardian v2 momentum escort mengubah holdout dari rugi (-$5.34)
   menjadi profit (+$17.68) dengan WR naik dari 41.1% → 68.7% (+27.6%).
✅ **LSTM adjustment** terbukti bernilai: menghasilkan 5× lebih banyak trade dengan WR 47%
   (profitable) tanpa Guardian, dan 68.7% dengan Guardian.

### Kriteria Upgrade (CLAUDE.md)
- [x] WR: 68.7% >> WR lama 41.1%
- [x] PF: 1.426 >> PF lama 0.599
- [x] Trades: 275 >> 80% × 56 = 44.8
- [x] Metodologi genuine OOF (Guardian dilatih pada OOF trades, bukan holdout)

→ **SEMUA kriteria terpenuhi.**

### Tahap 4 — Model Aktif Diupgrade (2026-06-23)
- Arsip: `reports/experiments/2026-06-23_widyawardhana_v1_parity.md`
- Update: `reports/widyawardhana_model.md` → v2 (LGBM parity + LSTM daily + Guardian momentum v2)
- **Belum di-deploy ke production** — perlu persetujuan eksplisit user

### Script
- `pipeline/07b_holdout_parity_guard_v2.py` (HOLDOUT_EVALUATED=True)

---

## 2026-06-24 — DEPLOY ic32_regime_v4 (LGBM + HMM gate 0.65/0.05 + Guardian v2)

**Status**: DEPLOYED ke production VPS (keputusan eksplisit user, override kriteria upgrade)

### Ringkasan
Setelah holdout base=0.74/0.06 gagal (60 trades), dilakukan sweep extended (08f, step 0.05,
base 0.55–0.80). Dipilih **base=0.65, delta=0.05** (OOF: 16,226 trades, WR 65.6%, PF 1.408).
Holdout 07c (base=0.65/0.05): **480 trades, WR 67.7%, PF 1.294, +$22.01**.

### Investigasi "kenapa OOF tinggi tapi holdout rendah"
- Audit drift fitur (`tools/diag_feature_drift.py`): 0 fitur drift |z|>1, semua 33 fitur dalam range
  aman. Tidak ada fitur hilang/bocor.
- Penyebab gap OOF vs holdout: **OOF pakai 8 fold models (lebih confident, signal rate ~2%),
  holdout pakai 1 final lgbm.pkl (lebih konservatif, signal rate ~1.1%)**. Bukan bug — properti
  normal fold-vs-final model. Sudah terverifikasi via distribusi p0/p2.

### Keputusan Deploy (override kriteria)
v4 GAGAL kriteria upgrade CLAUDE.md:
- WR 67.7% < 68.7% baseline ❌
- PF 1.294 < 1.426 baseline ❌
- Trades 480 >= 220 ✅
- Holdout terkontaminasi (dibuka 3x: 07, 07b, 07c) — angka tidak bisa diklaim OOS murni.

User memutuskan deploy tetap dilakukan dengan alasan: volume trade jauh lebih tinggi (480 vs 275),
S:L lebih seimbang, dan lepas dari ketergantungan coinank/LSTM daily (p_bull). **Benchmark
tervalidasi tetap Widyawardhana v2.**

---

## 2026-06-25 — Feature Selection: Stability Analysis + Ablation + Reduced Retrain (ic32.rv2.lgbm.24f)

**Status**: IN PROGRESS

### Hipotesis
33 fitur ic32_regime_v2_parity mengandung redundancy signifikan (pasangan korelasi r>0.90)
dan beberapa fitur dengan rank importance tidak stabil antar fold (rank range >12). Mengurangi
ke 24 fitur dengan menghapus yang redundant/tidak stabil akan mempertahankan atau meningkatkan
OOF PF & PnL/trade karena regularisasi efektif meningkat dan noise fitur berkurang.

### Yang Diubah
- **Fitur dihapus (9)**: h4_trend (imp negatif), Fib_618 (r=0.986 dgn Fib_786), stochrsi_d
  (r=0.929 dgn stochrsi_k), ema_50_h1 (r=-0.962 dgn ema_50_slope_h4), cvd_div_h4 (rank 29.6),
  MSB_BOS (rank 31.5, r>0.6 dgn banyak fitur), cvd_momentum_adv (rank 26.1, r=0.677 dgn WD),
  ofi_h4_delta (rank 25.9, parity issue live-sim), swing_momentum (rank 26.0, r=0.857 dgn rsi_slope)
- **Fitur dipertahankan (24)**: semua Liquidity & Positioning (kecuali relative_strength_z yang
  dipertahankan meski unstable), top momentum, CVD group, hmm_regime_enc
- Semua parameter lain identik: 8-fold purged walk-forward, purge=20, LGBM params sama

### Evidence dari Feature Selection Analysis (2026-06-25)
- Mean inter-fold rank corr: 0.830 (sehat)
- Grouped ablation delta F1: Liquidity +0.469, CVD +0.073, Momentum +0.051, Struktur +0.031, Regime +0.008
- Pasangan sangat redundant: Fib r=0.986, stochrsi r=0.929, ema_50 r=0.962, dist_liq_50x/20x r=0.918

### Target
- OOF WR >= 50.8% (baseline)
- OOF PF >= baseline (baseline OOF: $583 / 4812 trades = $0.12/trade)
- OOF trades >= 80% × 4812 = 3850 (tidak boleh terlalu sedikit)
- F1 macro >= 0.550 (baseline 0.557)

### Hasil OOF — 24f vs Baseline 33f

| Metrik | 33f Baseline | 24f (best thr) | Delta |
|--------|-------------|----------------|-------|
| OOF F1 macro | 0.5571 | **0.5612** | +0.0041 |
| OOF F1 std | 0.0059 | 0.0069 | +0.0010 |
| OOF Trades | 4,812 | 3,060 | -36.4% |
| OOF WR | 50.8% | 50.5% | -0.3pp |
| OOF PF | n/a | **1.404** | – |
| OOF PPT | $0.1210 | $0.1099 | **-$0.0111 (-9.2%)** |
| Best thr L/S | 0.75/0.70 | 0.75/0.70 | sama |

Sweep tambahan 24f (untuk perbandingan volume):

| Threshold | Trades | WR | PF | PPT |
|-----------|--------|----|----|-----|
| 0.75/0.70 | 3,060 | 50.5% | 1.404 | $0.1099 |
| 0.70/0.70 | 4,364 | 50.0% | 1.361 | $0.1036 |
| 0.75/0.65 | 6,271 | 51.3% | 1.399 | $0.1023 |

### Jawaban 4 Pertanyaan Evaluasi Medallion

**Q1 — Delta F1 +0.469 Liquidity diukur pada entry LGBM** (multiclass), bukan Guardian.

**Q2 — Delta OOF PPT setelah hapus 9 fitur: -$0.0111** pada threshold sama (0.75/0.70).
F1 naik +0.004 tapi PPT turun -9.2%. Trade count turun 36.4%. 9 fitur "rank rendah" ternyata
berkontribusi sebagai diversifier dalam LGBM ensemble meskipun individual importance rendah.

**Q3 — Dari 11 fitur unstable, 2 fitur sangat state-specific:**
- `rsi_h4`: imp=0.00016 di State 0 (TRENDING_DOWN) vs 0.00833–0.00863 di Ranging → kandidat
  regime-conditional weighting
- `ofi_acceleration`: imp=0.00632 di State 0, nyaris nol (0.00025) di State 3 (TRENDING_UP)
- Liquidity TETAP dominan di semua 4 state — tidak ada pergeseran ke momentum di trending state

**Q4 — Rank corr Liquidity group antar fold: 0.954** (vs 0.830 overall).
Liquidity jauh lebih stabil dari rata-rata — ini membuktikan edge Liquidity adalah structural
bukan regime-specific. Ordering dalam grup (50x > 20x > Buy/Sell > LSR > WD) identik semua fold.

`hmm_regime_enc` = 0.000 dalam SETIAP state secara individual, tapi 0.01049 di full data →
HMM encode informasi ANTAR state (untuk gating threshold), bukan discriminator entry DALAM state.

### Perbandingan OOF — Basis Threshold 0.65/0.65 (apple-to-apple)

| Skenario | Trades | WR | PF | PnL | PPT | MaxDD | S:L | Long WR | Short WR | Long PF | Short PF |
|---|---|---|---|---|---|---|---|---|---|---|---|
| LGBM 33f @ 0.65/0.65 | 12,959 | 49.7% | **1.350** | $1,256 | **$0.0969** | $27.03 | 5962:6997 | 48.3% | 50.9% | **1.332** | 1.368 |
| LGBM 24f @ 0.65/0.65 | 9,924 | 50.0% | 1.329 | $910 | $0.0917 | $26.28 | 4577:5347 | 48.1% | **51.6%** | 1.269 | **1.400** |

### Kesimpulan

**Hipotesis TIDAK terbukti**: mengurangi 33→24 fitur tidak meningkatkan PPT (turun -5.4% pada basis sama).
- F1 naik kecil (+0.004) — model tidak degraded dari sisi F1 macro
- Tapi 9 fitur yang dihapus berkontribusi sebagai "recall diversifier" — menambah coverage trade
  valid yang sekarang tidak ter-capture
- Alternatif yang lebih konservatif (hanya hapus pasangan r>0.95): Fib_618, stochrsi_d, ema_50_h1
  → belum diuji; kemungkinan trade-off lebih kecil

**Keputusan: TIDAK upgrade ke 24f** — PPT turun dan trade count turun tanpa benefit PF yang
jelas (PF baseline tidak tersedia untuk perbandingan langsung).

**Status**: COMPLETED (sebagian) — model 24f disimpan di `models/runs/ic32_rv2_lgbm_24f_sw4_r8/`
tapi tidak di-deploy. Benchmark tetap `ic32_regime_v2_parity` (33f).

### Follow-up: 4 Pertanyaan Lanjutan (2026-06-25)

**Q3 — Liquidity importance per sub-periode:**
| Sub-periode | Total Liq Imp | F1 baseline |
|---|---|---|
| 2020-2021 Bull Run | 0.568 | 0.638 |
| 2022 Bear/Crash | **0.544** | 0.631 |
| 2023-mid2024 Recovery | 0.681 | 0.659 |
| late2024-2026 Mixed | 0.586 | 0.652 |
Kesimpulan: Liquidity dominan di SEMUA periode termasuk 2022 bear. Tidak ada hidden regime risk.
Variasi ~20% (0.544 bear vs 0.681 recovery) — fluktuasi normal, bukan collapse.

**Q4 — Stability comparison 24f vs 33f:**
| Model | Rank Corr | Unstable fitur | Liq rank corr |
|---|---|---|---|
| 33f | 0.809 | 11 | 0.954 |
| 24f | **0.822** | **7** | **0.964** |
Kesimpulan: 24f lebih stabil (+0.013 rank corr). Trade-off: stability naik tapi recall turun.

**Q1 — 29f (hapus h4_trend, Fib_618, stochrsi_d, ema_50_h1 saja) @ 0.65/0.65:**
- Trades: 10,293 (-20.6%)  WR: 49.9%  PF: 1.326  PPT: $0.0909 (delta -$0.0060 = -6.2%)
- OOF F1: 0.5616 (33f: 0.5571) — F1 naik tipis tapi PPT tetap turun
- Disimpan di `models/runs/ic32_rv2_lgbm_29f_sw4_r8/`

**Q2 — 31f+interactions (rsi_h4 × state, ofi_acc × state) @ 0.65/0.65:**
- Trades: 10,065 (-22.3%)  WR: 49.9%  PF: 1.315  PPT: $0.0873 (delta -$0.0096 = -9.9%)
- LEBIH BURUK dari 29f dan 24f — interaction features tidak membantu
- LGBM sudah menangkap interaksi HMM×fitur melalui tree splitting alami
- Disimpan di `models/runs/ic32_rv2_lgbm_31f_sw4_r8/`

**Keputusan final**: Semua versi reduced fitur lebih buruk dari 33f pada PPT dan volume trade.
Arah (B) — terima 33f sebagai baseline, fokus ke komponen lain (Guardian / threshold / risk).

---

## 2026-06-25 — WD tanpa Komponen L/S (WD = CVD z-score saja)

**Status**: PLANNED

### Hipotesis
Saat holdout Jun 24 menggunakan L/S statis 1.8058 (coinank stale), secara tidak sengaja
`ls_z = 0` (karena L/S konstan → std = 0 → ls_z = NaN → 0), sehingga:
`whale_retail_divergence = cvd_z - 0 = cvd_z`
Model secara efektif berjalan TANPA pengaruh L/S di WD, dan hasil trade selection berbeda
(meski lebih bagus di Jun 24 karena selection bias 1 hari). Hipotesis: komponen ls_z di WD
adalah **noise bukan signal** — model 33f dengan WD = cvd_z (zero-out ls_z) secara konsisten
menghasilkan OOF metrics lebih baik daripada WD = cvd_z - ls_z.

### Yang Diubah
- **Tetap 33 fitur** — tidak ada fitur dihapus/ditambah
- `whale_retail_divergence` di-recompute per-coin: WD = cvd_z.clip(-5, 5) (ls_z = 0)
- `long_short_ratio` tetap ada sebagai fitur terpisah
- Semua parameter lain identik: 8-fold purged walk-forward, LGBM params sama

### Target
- Bandingkan vs baseline 33f @ 0.65/0.65: trades=12959, WR=49.7%, PF=1.350, PPT=$0.0969
- Jika PPT naik, berarti ls_z adalah noise yang mengganggu WD signal
- Jika PPT turun, berarti ls_z memang informatif dan WD formula saat ini sudah benar

### Script
- Retrain: `scratchpad/retrain_lgbm_33f_no_lsz.py` → run: `ic32_rv2_lgbm_33f_wd_cvz`

### Hasil OOF @ 0.65/0.65

| Skenario | Trades | WR | PF | PPT | Delta PPT |
|---|---|---|---|---|---|
| 33f baseline (WD = cvz - lsz) | 12,959 | 49.7% | 1.350 | $0.0969 | — |
| 33f WD = cvz saja (ls_z = 0) | 10,094 | 49.2% | 1.288 | $0.0808 | **-16.6%** |

### Kesimpulan
Hipotesis TIDAK TERBUKTI. ls_z berkontribusi POSITIF dalam whale_retail_divergence — bukan noise.
Delta PPT -16.6% adalah yang terbesar dari semua variasi yang diuji. WR pun ikut turun.
Holdout Jun24 lebih baik dengan L/S statis = pure selection bias (1 hari).
Baseline 33f formula WD asli (cvz - lsz) adalah konfigurasi optimal.

**Status**: COMPLETED — tidak ada perubahan ke produksi.

### Scorecard Lengkap Semua Skenario @ 0.65/0.65

| Skenario | Trades | WR | PF | PnL | PPT | Delta PPT |
|---|---|---|---|---|---|---|
| **33f baseline** | **12,959** | **49.7%** | **1.350** | **$1,256** | **$0.0969** | — |
| 24f (hapus 9) | 9,924 | 50.0% | 1.329 | $910 | $0.0917 | -5.4% |
| 29f (hapus 4) | 10,293 | 49.9% | 1.326 | $936 | $0.0909 | -6.2% |
| 31f+interactions | 10,065 | 49.9% | 1.315 | $879 | $0.0873 | -9.9% |

Catatan metodologi: fold structure 24f menggunakan `shared.build_rolling_folds` (uniform chunk
95698/fold) sedangkan baseline menggunakan logic lama (unequal splits). Coverage OOF sama
(861K rows), purge gap sama (20). Tidak mempengaruhi validitas perbandingan.

### Script
- Feature selection analysis: `scratchpad/feature_selection_analysis.py`
- Regime-conditional + Liquidity rank corr: `scratchpad/regime_conditional_importance.py`
- Retrain 24f: `scratchpad/retrain_lgbm_24f.py` → run: `ic32_rv2_lgbm_24f_sw4_r8`

### Perubahan teknis (config-only, tanpa ubah kode live)
- `inference_config.json`: model_version=ic32_regime_v4; lstm_daily_adjust.enabled=**false**;
  hmm.per_state_thresholds = {0:[0.70,0.60], 1:[0.675,0.625], 2:[0.625,0.675], 3:[0.60,0.70], -1:[0.65,0.65]}
- Parity entry terverifikasi: `apply_hmm_gate_single` live identik logika sim 07c; encoding HMM
  canonical (regime.py di-deploy); argmax-equiv (thr>=0.60); momentum_reduce off.
- Catatan parity Guardian: fitur p_bull (1/31) di backtest 07c ~0, di live pakai p_bull asli →
  exit bisa sedikit beda. Second-order, entry 100% parity.
- Deploy: `tools/deploy_production.py`. Live VPS terverifikasi (ic32_regime_v4, p_bull adjust off,
  per_state benar). Backup rollback: `swint/models/backups/backup_20260624_002652`.

### Script
- `pipeline/08f_hmm_sweep_extended.py` (sweep), `pipeline/07c_holdout_hmm_base65_d05.py` (holdout),
  `tools/diag_feature_drift.py` (audit drift)

---

## 2026-06-23 — Holdout: LGBM + HMM Gate + Guardian v2 (base=0.74, delta=0.06)

**Status**: COMPLETED — HOLDOUT_EVALUATED=True, amplop dikunci

### Hipotesis
HMM regime gate (4-state, full gating) akan memperbaiki SHORT bias dan meningkatkan
WR/PF holdout vs Widyawardhana v2 (LGBM+LSTM+Guardian). OOF menunjukkan WR 68.2%, PF 1.565,
5,129 trades — target: lampaui holdout baseline v2 (WR 68.7%, PF 1.426, 275 trades).

### Yang Diubah vs Baseline
- **Ganti LSTM threshold adj** → **HMM regime gate** (4 state: TRENDING_DOWN/RANGING_LOW/RANGING_HIGH/TRENDING_UP)
- **Config**: base=0.74, delta=0.06 (dipilih dari sweep grid, trade-off tinggi PnL vs PF terbaik)
- **Regime thresholds**:
  - TRENDING_UP:   LONG>=0.68, SHORT>=0.80
  - RANGING_HIGH:  LONG>=0.71, SHORT>=0.77
  - RANGING_LOW:   LONG>=0.77, SHORT>=0.71
  - TRENDING_DOWN: LONG>=0.80, SHORT>=0.68
- **OOF** (full 21 koin, 8-fold purged CV): WR 68.2%, PF 1.565, 5,129 trades, PnL $598

### Hasil Holdout (Apr 1 – Jun 22, 2026)

| Metrik | LGBM+HMM+Guard | Widyawardhana v2 (baseline) | Lolos? |
|--------|---------------|----------------------------|--------|
| Trades | 60 | 275 | **GAGAL** (<=27%) |
| WR | 63.3% | 68.7% | **GAGAL** |
| PF | 1.042 | 1.426 | **GAGAL** |
| PnL | +$0.57 | +$17.68 | **GAGAL** |
| PnL/trade | $0.0094 | $0.0643 | **GAGAL** |
| MaxDD | -$6.22 | -$3.44 | **GAGAL** |
| S:L | 2.75x | ~pure SHORT | — |
| Long WR | 37.5% | — | — |
| Short WR | 72.7% | — | — |
| Long PF | 0.483 | — | — |
| Short PF | 1.561 | — | — |

### Kesimpulan
**Hipotesis TIDAK terbukti.** LGBM+HMM+Guardian GAGAL semua 3 kriteria upgrade.

**Root cause:**
1. **Trade volume kolaps**: 60 vs 275 baseline (4.6x lebih sedikit). HMM threshold terlalu
   ketat di holdout — distribusi regime Apr-Jun 2026 berbeda dari distribusi training (2020-2026).
   Threshold yang dioptimasi via OOF overfit ke distribusi historis.
2. **LONG direction rusak**: 16 LONG, WR 37.5%, PF 0.483. Apr-Jun 2026 dominan RANGING/TRENDING_DOWN
   → threshold LONG naik ke 0.77-0.80 → LGBM tidak cukup confident → sedikit LONG, dan yang masuk kalah.
3. **SHORT bagus tapi sedikit**: Short WR 72.7%, PF 1.561 — signal SHORT masih bekerja, tapi
   volume tidak cukup untuk menutupi kerugian LONG.
4. **Temuan penting**: OOF HMM (WR 68.2%, PF 1.565) TIDAK generalisasi ke holdout OOS.
   Ini menunjukkan HMM threshold sweep (meski dilakukan OOF) mengadaptasi ke distribusi pasar
   multi-tahun yang tidak representatif untuk periode tertentu.

**Model aktif tetap Widyawardhana v2.**

### Script
- `pipeline/07b_holdout_hmm_guardian_v2.py` (HOLDOUT_EVALUATED=True)
- Hasil: `models/runs/ic32_regime_v2_parity/holdout_hmm_guardian_v2.json`

---

## 2026-06-23 — Daily LSTM v2: Binance-direct L/S + Automasi p_bull Harian

**Status**: PLANNED

### Hipotesis
Data L/S coinank (`top_trader_position_ls`, `top_trader_account_ls`) terbukti IDENTIK untuk semua
21 koin sepanjang 499 hari — bukan per-koin, melainkan satu nilai global. Akibatnya 3 dari 15 fitur
riil di daily LSTM (`ls_z20`, `ls_d7`, `smart_retail`) tidak memberikan sinyal coin-discriminative.
Dengan mengganti sumber ke Binance-direct (`/futures/data/topLongShortPositionRatio` dan
`topLongShortAccountRatio`, period=1d, per-koin benar), model akan untuk pertama kalinya
mendapatkan sinyal L/S yang sesungguhnya per koin.

Hipotesis: OOF AUC daily LSTM v2 lebih tinggi dari v1, dan p_bull akan lebih berkorelasi dengan
pergerakan harga per koin (bukan agregat global).

### Yang Diubah vs v1
- **Sumber L/S**: `data/coinank/{coin}_ls_position.parquet` (global-corrupt) →
  `data/positioning/{coin}_ls_daily.parquet` (Binance-direct, coin-specific, kausal)
- **Account L/S**: juga dari Binance `/futures/data/topLongShortAccountRatio` (per-koin)
- **Sumber OI**: tetap dari `data/coinank/{coin}_oi.parquet` (masih benar, coin-specific)
- **RUN_ID**: `ic32_daily_lstm_17f_s30` → `ic32_daily_lstm_17f_s30_v2`
- **Infrastructure baru**:
  - `tools/backfill_binance_daily_ls.py` — historical daily L/S dari Binance API
  - `pipeline/11_train_lstm_daily_v2.py` — training script dgn Binance data
  - `tools/refresh_pbull.py` — daily automation (fetch → compute → scp VPS)
  - `01c_fetch_positioning.py` — tambah `topLongShortAccountRatio` endpoint

### Verifikasi Kausalitas Fitur L/S
- `topLongShortPositionRatio` period=1d: Binance melabeli bar harian pada AKHIR hari (00:00 UTC
  berikutnya), bukan awal. Dengan `shift(1)` dalam `build_coin_df()`, fitur D-1 dipakai untuk
  prediksi hari D → kausal.
- Account L/S: sama, `shift(1)`.

### Target
- OOF AUC daily LSTM v2 > v1 (v1 unknown but expected ~0.55-0.60)
- Holdout WR dengan p_bull v2 >= 68.7% (baseline ic32_regime_v3)
- Holdout PF >= 1.426

### Script (urutan)
1. `tools/backfill_binance_daily_ls.py` — backfill historical L/S ke data/positioning/
2. `pipeline/11_train_lstm_daily_v2.py` — retrain daily LSTM
3. `tools/compute_daily_pbull.py --use-binance --lstm-run ic32_daily_lstm_17f_s30_v2`
4. `pipeline/07b_holdout_parity_guard_v2.py` — re-eval holdout (hanya jika deploy dipertimbangkan)

---

## 2026-06-23 — ic32_regime_v4: HMM-Gated Direction Threshold

**Status**: PLANNED

### Hipotesis
Model ic32_regime_v3 sangat bias SHORT (274 SHORT vs 1 LONG di holdout Apr-Jun 2026).
Analisis OOF mengungkap HMM regime sudah encode directional edge dengan sempurna:

| Regime | SHORT prec | LONG prec | Natural dominance |
|---|---|---|---|
| TRENDING_DOWN | 0.451 | 0.430 | SHORT 1.75× lebih banyak |
| RANGING_LOW_VOL | 0.493 | 0.428 | SHORT 1.73× lebih banyak |
| RANGING_HIGH_VOL | 0.433 | 0.451 | LONG 1.39× lebih banyak |
| TRENDING_UP | 0.410 | 0.492 | LONG 2.63× lebih banyak |

SHORT di TRENDING_UP = precision hanya 0.388–0.410 (paling buruk). Threshold flat (0.75/0.70)
ditambah p_bull bearish → eksaserbasi SHORT bias menjadi 274:1 di holdout bear market.

Hipotesis: adaptive threshold per-regime (HMM sebagai direction gate) akan:
1. Mengurangi SHORT:LONG bias dari 2.5:1 → mendekati 1:1 pada OOF
2. Meningkatkan presisi dengan tidak ambil counter-trend trades
3. Lebih adaptif noise H1 (RANGING_LOW_VOL dengan thr lebih tinggi)

### Yang Diubah vs ic32_regime_v3
- **LGBM + Guardian**: TIDAK diubah — sama persis
- **Threshold system**: parameterized by `(base_thr, regime_delta)`
  - `base_thr` ∈ [0.68, 0.70, 0.72]: baseline threshold kedua arah
  - `regime_delta` ∈ [0, 0.08, 0.12, 0.16, 0.20]: seberapa kuat gate per-regime
  - TRENDING states: aligned_dir = base_thr - δ/2, counter_dir = base_thr + δ/2
  - RANGING states: half-delta (gating lebih lemah karena noise tinggi)
- **p_bull**: tetap sebagai micro-adjustment (±0.03) di atas HMM thresholds

### Verifikasi Kausalitas
- `hmm_regime_enc` adalah fitur LGBM existing → tidak ada lookahead baru
- HMM difit dengan purged walk-forward → regime terkini tidak bocor ke training
- Post-prediction gate berbasis regime TIDAK mengubah kausalitas model

### Target
- OOF trades: >= 2,000 (tidak terlalu sedikit)
- OOF SHORT:LONG ratio: <= 3:1 (dari 2.51:1 baseline)
- OOF WR: >= 50.8% (baseline ic32_regime_v2_parity OOF)
- OOF PnL/trade: >= $0.12 (baseline)
- Baseline dibandingkan: ic32_regime_v3 (holdout WR 68.7%, PF 1.426, +$17.68, 275 trades)

### Hasil OOF Sweep

**Status**: COMPLETED (OOF only — holdout belum tersedia)

Sweep 24 kombinasi (4 base_thr × 6 regime_delta). Hasil diurutkan PnL/trade:

| Rank | base_thr | delta | Trades | WR | PnL | PnL/trade | S:L |
|------|----------|-------|--------|----|-----|-----------|-----|
| Baseline | 0.75/0.70 flat | — | 4,812 | 50.8% | $583 | $0.1212 | 2.5x |
| **#1** | **0.74** | **0.06** | **3,903** | **52.3%** | **$604** | **$0.1547** | **1.1x** |
| #2 | 0.74 | 0.10 | 4,560 | 52.1% | $658 | $0.1442 | 1.2x |
| #3 | 0.74 | 0.00 | 3,281 | 50.3% | $450 | $0.1371 | 1.0x |
| #4 | 0.74 | 0.14 | 5,387 | 51.7% | $735 | $0.1364 | 1.2x |
| #5 | 0.72 | 0.10 | 6,171 | 51.5% | $829 | $0.1344 | 1.2x |

**Config terbaik (base=0.74, delta=0.06):**
```
TRENDING_DOWN  : thr_long=0.770  thr_short=0.710
RANGING_LOW    : thr_long=0.755  thr_short=0.725
RANGING_HIGH   : thr_long=0.725  thr_short=0.755
TRENDING_UP    : thr_long=0.710  thr_short=0.770
```
Disimpan: `models/runs/ic32_regime_v2_parity/hmm_regime_thresholds.json`

### Kesimpulan OOF

- **SHORT bias turun drastis**: S:L dari 2.5x → 1.1x (hampir seimbang)
- **PnL/trade naik +27.6%**: $0.1212 → $0.1547
- **WR naik**: 50.8% → 52.3%
- **Trades lebih sedikit tapi berkualitas**: 3,903 vs 4,812 (lebih selektif)
- Perbaikan datang dari **2 sumber**: (a) base threshold dinaikkan ke 0.74 (lebih selektif
  dari thr_short=0.70 saat ini), (b) regime gating ringan (delta=0.06) menyeimbangkan arah

**Hipotesis terbukti di OOF.** Untuk validasi final perlu holdout baru (Jul-Sep 2026).

### Hasil OOF Full Stack — Scorecard Lengkap (Guardian aktif semua skenario)

Script: `pipeline/08c_oof_full_stack_eval.py`

| Metrik | A. Baseline | **B. HMM-only** | C. HMM+LSTM | D. HMM+LSTM gated |
|---|---|---|---|---|
| **Trades** | 4,962 | **4,010** | 15,001 | 12,687 |
| **WR** | 68.8% | **69.2%** | 61.3% | 62.6% |
| **PF** | 1.589 | **1.694** | 1.317 | 1.369 |
| **Net PnL** | $577 | $548 | **$1,112** | $1,041 |
| **PnL/trade** | $0.1163 | **$0.1368** | $0.0741 | $0.0821 |
| Avg Win | $0.4561 | $0.4830 | $0.5019 | $0.4865 |
| Avg Loss | -$0.6322 | -$0.6392 | -$0.6037 | -$0.5937 |
| Max Drawdown | -$11.40 | -$14.58 | -$25.99 | -$25.82 |
| SHORT count | 3,688 | 2,252 | 1,918 | 2,042 |
| LONG count | 1,274 | 1,758 | 13,083 | 10,645 |
| S:L ratio | 2.9x | **1.3x** | 0.15x | 0.19x |
| LONG WR | 68.7% | **69.8%** | 60.2% | 61.4% |
| SHORT WR | 68.8% | 68.7% | 69.2% | 68.7% |
| LONG PF | 1.820 | **1.923** | 1.289 | 1.346 |
| SHORT PF | 1.496 | 1.491 | 1.561 | 1.513 |
| LONG PnL/trade | $0.1817 | **$0.1947** | $0.0695 | $0.0790 |
| SHORT PnL/trade | $0.0937 | $0.0916 | $0.1055 | $0.0978 |
| GUARDIAN_EXIT | 52.1% | 51.6% | 44.4% | 45.1% |
| LOSS | 28.0% | 27.5% | 35.2% | 33.8% |

**Temuan kritis — LSTM complement LONG bias:**
- LSTM complement dilatih dengan `bull_thr=0.38` untuk mengimbangi LGBM SHORT-heavy.
- Dengan HMM gating yang sudah menyeimbangkan LGBM (S:L 1.3x), LSTM malah membalikkan
  bias ke LONG ekstrem: S:L=0.15x (6.8× lebih banyak LONG dari SHORT di skenario C).
- LONG WR turun dari 69.8% (B) ke 60.2% (C) — kualitas LONG dari LSTM jauh lebih rendah.
- HMM direction lock di skenario D hanya sedikit memperbaiki.
- **LSTM complement perlu retrain** dengan HMM-gated flat definition & bull_thr ~0.50.

**Kandidat terbaik ic32_regime_v4: Skenario B (HMM-only)**
- PF +0.105 dan PnL/trade +17.6% vs baseline, S:L seimbang (1.3x)
- LSTM complement tidak kompatibel tanpa recalibration

### Script
- `pipeline/08_oof_hmm_regime_threshold_sweep.py` — OOF sweep per-regime threshold
- `pipeline/08c_oof_full_stack_eval.py` — OOF full stack comparison (4 skenario)

---

## 2026-06-24 — OOF: Guardian Momentum v2 Exit-Param Sweep di Stack v4 (HMM 0.65/0.05)

**Status**: COMPLETED — kesimpulan: **frontier PF jenuh, tidak ada upgrade.**

### Hasil (20 kombinasi, OOF, entry HMM 0.65/0.05 fixed)

Baseline v4 (exit_thr=0.65, floor=0.20): PF 1.399, payoff 0.758, WR 64.9%, PnL $1,414, PPT $0.0874, margin +8.0pp, 16,187 trades.

| exit_thr | floor | Trades | WR | PF | payoff | PnL | PPT | margin | LOSS% |
|---|---|---|---|---|---|---|---|---|---|
| 0.55 | 0.20 | 16,544 | 67.4% | 1.383 | 0.669 | $1,290 | 0.0780 | +7.5pp | 29.3% |
| 0.60 | 0.20 | 16,364 | 66.1% | 1.392 | 0.713 | $1,356 | 0.0829 | +7.8pp | 30.2% |
| **0.65** | **0.20** | **16,187** | **64.9%** | **1.399** | **0.758** | **$1,414** | **0.0874** | **+8.0pp** | **31.2%** |
| 0.70 | 0.20 | 15,999 | 63.2% | 1.395 | 0.813 | $1,440 | 0.0900 | +8.0pp | 32.4% |
| 0.75 | 0.20 | 15,811 | 61.8% | 1.398 | 0.865 | $1,481 | 0.0937 | +8.1pp | 33.4% |

(floor_frac 0.10–0.40 efeknya kecil: <0.01 PF, <0.02 payoff di semua exit_thr.)

### Kesimpulan
- **PF jenuh di ~1.40 di seluruh grid.** Tidak ada kombinasi yang melampaui PF baseline secara material;
  maksimum 1.400 (vs baseline 1.399). Exit-param BUKAN lever untuk PF.
- Yang bisa digeser hanya **profil WR↔payoff** di sepanjang garis PF datar:
  - Naikkan `exit_thr` 0.65→0.75 → payoff 0.758→0.865, PnL +$67 (+4.7%), PPT +7%, **tapi WR turun 64.9%→61.8%**.
  - `floor_frac` praktis tidak berpengaruh.
- **Margin breakeven nyaris konstan (+7.5 s.d. +8.2pp)** di seluruh grid — menggeser exit tidak menebalkan
  bantalan. Kelemahan payoff bersifat **struktural** (desain Guardian escort + threshold), bukan soal tuning exit.

### Tahap 4 — TIDAK upgrade
Tidak ada config yang memenuhi kriteria (WR >= 64.9% DAN PF >= 1.399 bersamaan). Baseline v4 dipertahankan.
**Implikasi:** tidak ada cara volume-neutral menaikkan PF v4. Satu-satunya lever PF terbukti = perketat
threshold entry (frontier OOF: 0.70/0.05 → PF 1.519/8,672 trades; 0.75/0.10 → PF 1.578/6,356) — menukar volume.

### Keputusan threshold-frontier (2026-06-24)
Frontier volume-PF disajikan ke user. Semua titik di atas v4 perbaiki WR+PF+MaxDD tapi volume −47% s.d. −76%.
**User memilih TIDAK menggeser** — volume/total PnL diprioritaskan (konsisten alasan deploy v4 awal).
v4 0.65/0.05 dipertahankan sebagai operating point. Sisi LONG lemah = artefak holdout (di OOF LONG sehat,
PF 1.394/payoff 0.807) → JANGAN tuning entry ke holdout. Lever PF volume-neutral yang tersisa hanya
**riset alpha entry baru** atau **retrain Guardian** (effort sedang-besar) — belum dijalankan.

### Script
- scratchpad `v4_guard_exit_sweep.py`; hasil: `models/runs/ic32_rv2_guard_momentum_v2/v4_exit_param_sweep.json`

### Rencana Awal (hipotesis & desain — diarsipkan)

#### Hipotesis
v4 OOF payoff = 0.74 (<1) — edge murni WR-driven, margin breakeven cuma +8.1pp. Parameter exit
Guardian (`floor_frac=0.20`, `exit_thr=0.65`) dulu di-optimasi pada stack **PARITY+LSTM**
(~5,767 OOF trades), BUKAN pada stack **v4 HMM-gated** (16,226 trades, short-heavy, no LSTM).
Populasi trade berbeda → titik exit optimal bisa berbeda. Re-optimasi `exit_thr × floor_frac`
pada populasi v4 mungkin mengangkat payoff/PF.

**Kunci:** exit-param TIDAK mengubah entry → **volume tetap 16,226 by construction** (volume-neutral).
Yang dipertaruhkan: win lebih besar (payoff naik) vs give-back (GUARDIAN_EXIT berubah jadi LOSS).

#### Yang Diubah
- Entry: **FIXED** — HMM gate 0.65/0.05, identik v4. Tidak ada fitur baru, tidak ada non-H1 baru
  (ATURAN 6 tidak berlaku — tidak ada resample/timeframe baru).
- Guardian model: **FIXED** — `ic32_rv2_guard_momentum_v2` (tidak retrain).
- Hanya 2 parameter inference exit di-sweep: `exit_thr` ∈ [0.55–0.75], `floor_frac` ∈ [0.10–0.40] (20 kombinasi).

#### Target (asli)
- PF > 1.408 DAN payoff > 0.74; margin >= +8.1pp; trades ~16,226 (volume-neutral).
- Metodologi: OOF only (ATURAN 1); holdout terkontaminasi → TIDAK disentuh; validasi final perlu holdout baru > Jun 2026.

---

## 2026-06-24 — MODEL BARU: Trade-Quality Sizing (volume-neutral payoff lift)

**Status**: COMPLETED (OOF) — sinyal NYATA tapi LEMAH; PnL/PF naik tapi MaxDD memburuk → belum deploy.

**Nama**: `ic32.rv2.qsize.lgbm` (LightGBM regressor R-multiple). Script: scratchpad `qsize_oof.py` (belum dipromosikan ke pipeline).

### Hasil OOF (walk-forward 6-fold purged, n_eval=11,359, capital-neutral)
**Gate IC — LOLOS:** IC_lgbm = **+0.0431** overall, folds [0.051, 0.067, 0.026, 0.040, 0.055, 0.018] —
semua POSITIF, sign stabil. Floor `IC_conf` (size by entry-confidence) = **−0.0205** (negatif!) →
LightGBM **mengalahkan heuristik confidence telak**. Kualitas trade memang terprediksi (lemah).

Sizing sweep (range multiplier, mean=1.0, baseline uniform PnL $794 / PF 1.357 / payoff 0.722 / MaxDD −$22.14):
| range | PnL | dPnL | PF | payoff | MaxDD | dMaxDD |
|---|---|---|---|---|---|---|
| [0.85,1.15] | $821 | +$27 | 1.367 | 0.727 | −$23.9 | −$1.8 |
| [0.80,1.20] | $830 | +$36 | 1.370 | 0.729 | −$24.5 | −$2.4 |
| [0.70,1.30] | $848 | +$54 | 1.377 | 0.732 | −$25.7 | −$3.6 |
| [0.50,2.00] | $902 | +$108 | 1.397 | 0.743 | −$29.3 | −$7.2 |

### Kesimpulan (jujur)
- ✅ Hipotesis terbukti SEBAGIAN: kualitas trade terprediksi (IC +0.043 stabil, kalahkan floor), dan sizing
  menaikkan PnL/PF/payoff **volume-neutral** persis seperti target.
- ❌ TAPI **MaxDD selalu memburuk** di semua range — tidak ada titik PnL-up & DD-netral. Rasio PnL/MaxDD
  malah TURUN (uniform 35.9 → [0.50,2.0] 30.7): konsentrasi modal ke predicted-winner menambah risiko
  lebih cepat dari tambahan PnL. IC 0.043 terlalu lemah untuk menang di basis risk-adjusted.

### Tahap 4 — BELUM deploy
PF naik tapi MaxDD memburuk → tidak lolos bersih kriteria upgrade. **Tapi ini sinyal positif PERTAMA**
di seluruh investigasi v4 — kualitas trade ternyata punya struktur terprediksi. Nilai sebenarnya baru
muncul kalau **IC dinaikkan** (0.043 → ~0.10) lewat fitur khusus-kualitas / target lebih baik, sehingga
gain PnL menutup biaya DD. Lever lanjut: perkuat model kualitas, bukan sizing function.

### Validasi final
Belum di holdout. Holdout baru (>Jun 2026) hanya jika IC versi diperkuat tembus & risk-adjusted positif.

### Iterasi 2 (COMPLETED) — Perkuat model kualitas
**Yang diubah:** fitur 24 → **49** (union LGBM-33 + guardian-static + p0/p2 + entry-context: rr, sl_dist,
dir, counter_trend). 4 varian target diuji. Semua fitur EXISTING & kausal (ATURAN 6 N/A).

**Hasil IC (OOF walk-forward, n=11,359):**
| Target | IC |
|---|---|
| R-multiple (49 feat) | **+0.0482** |
| R winsorized [-2,2] | +0.0467 |
| P(win) classifier | +0.0159 |
| Long/Short terpisah | +0.0395 |
| *baseline iter-1 (24 feat)* | *+0.0431* |

**Temuan 1 (mengecewakan):** ekspansi fitur 24→49 + 4 target **TIDAK** menaikkan IC ke target (0.06–0.10).
IC cuma 0.043 → 0.048. Target engineering (winsor/biner/LS-split) malah lebih buruk. **Edge kualitas
mentok ~0.05** dengan fitur kausal yang ada. Goal "naikkan IC" GAGAL.

**Temuan 2 (bonus):** model lebih kaya membuat sizing **risk-adjusted bersih** — rasio PnL/MaxDD KONSTAN
~36 di semua range (iter-1 turun 35.9→30.7). Sizing [0.50,2.0]: PnL +$107 (+13.5%), PF 1.357→1.395,
payoff +0.020, **MaxDD cuma −$2.9 (vs −$7.2 iter-1)**. Gain PnL/PF kini datang tanpa degradasi risk-adjusted.

| range | PnL | dPnL | PF | payoff | MaxDD | dMaxDD |
|---|---|---|---|---|---|---|
| uniform | $794 | — | 1.357 | 0.722 | −$22.1 | — |
| [0.80,1.20] | $830 | +$36 | 1.370 | 0.729 | −$23.1 | −$1.0 |
| [0.50,2.00] | $901 | +$107 | 1.395 | 0.742 | −$25.0 | −$2.9 |

### Uji Statistik IC & Marginal Value (2026-06-24)
**IC significance:**
- Overall IC=+0.0482, n=11,359, **t=5.14, p=2.8e-7** (sangat signifikan, bukan nol).
- Per-fold IC=[0.070,0.040,0.063,0.047,0.016,0.060], **100% positif**, t(5)=6.21, p=0.0016.
- Block-bootstrap by-coin (B=1000) 95% CI=[+0.033,+0.063], **100% sampel >0** (robust thd klaster koin).
- **IC orthogonal-to-confidence = +0.049** (tak berubah) → sinyal qsize INDEPENDEN dari confidence LGBM (info baru).

**Marginal value test (sizing edge nyata vs kebetulan):**
- Extra PnL [0.5,2.0] = **+$106.86**. Permutation null (acak pred): mean≈$0, std $23 → **z=4.62, p=0.0005**.
- Block-bootstrap 95% CI extra PnL = **[+$68, +$139], 100% >0**.

### Tahap 4 — Verdict (REVISI setelah uji statistik)
Edge **statistik ROBUST, bukan noise**: IC signifikan (p~3e-7), 100% fold positif, CI bootstrap seluruhnya >0,
independen dari confidence; gain sizing lolos permutation (p=0.0005) & bootstrap (CI >0). Yang benar adalah
edge ini **kecil secara ekonomi (IC ~0.05, PnL +13.5% max), bukan rapuh secara statistik.**
Sisa gap = konfirmasi OOS segar (OOF di sini genuine purged WF, sah sbg basis keputusan; tapi Apr–Jun
terkontaminasi & data >Jun belum ada). **Belum deploy** — magnitudo kecil + belum ada OOS live. Tapi kualitas
trade kini terbukti **edge nyata, signifikan, independen, volume-neutral.**

### Hipotesis
Payoff v4 = 0.74 (<1) bersifat struktural — sweep (exit & threshold) terbukti tidak bisa memperbaikinya.
Hipotesis baru: jika **kualitas tiap trade (realized R-multiple) bisa diprediksi walau lemah** dari fitur
**entry-time** (target IC > 0.03), maka **realokasi modal** ke trade berkualitas tinggi — dengan total
kapital & jumlah trade IDENTIK — menaikkan $-weighted PF & payoff **tanpa kehilangan volume**.
Ini menyerang langsung kelemahan payoff, sesuai batasan user (volume tidak boleh turun).

Kenapa beda dari yang gagal: TIDAK memprediksi arah (plafon LGBM ~51%, LSTM ≈ random). Memprediksi
*magnitude* dari sinyal yang SUDAH lolos entry LGBM+HMM — problem berbeda & lebih tractable.

### Yang Diubah
- Komponen BARU: regressor kualitas (LightGBM) → prediksi realized R-multiple per sinyal.
- Entry (LGBM+HMM 0.65/0.05) & exit (Guardian v2) **TIDAK diubah** — 16,226 trade tetap diambil semua.
- Sizing: `modal_mult = f(pred_R)`, dinormalisasi **mean=1.0 per fold** (capital-neutral; bukan naikkan bet).
  Range awal [0.5, 2.0], monotonik terhadap rank prediksi.

### Data & Label
- Dataset: 16,226 OOF v4 trades (genuine, config 0.65/0.05).
- X = fitur **entry-time SAJA**: LGBM p0/p2/conf, hmm_regime_enc, atr, ~30 static guardian feats, vol_regime, dst.
  NOL fitur masa depan trade (tanpa exit/MFE/bars_held/outcome). Map via `bar_in` ke baris fitur.
- y = realized R-multiple = `net_pnl / planned_risk_$` (planned_risk = |entry−sl|/entry × modal × lev).

### Metodologi (kritis — anti-leakage)
- Sizing model dilatih **PURGED WALK-FORWARD di level-waktu trade**: train trade masa lalu → prediksi
  kualitas trade masa depan. Purge gap = MAX_HOLDING_BARS (36 bar). Model tidak pernah lihat outcome
  fold ujinya (ATURAN 1/2/4). Scaler fit di dalam fold (ATURAN 3).
- Tidak ada fitur non-H1 baru → ATURAN 6 N/A (semua fitur sudah ada & terverifikasi di stack v4).
- Capital-neutral: mean(modal_mult)=1.0 → perbandingan adil (bukan karena modal lebih besar).

### Target (kriteria lanjut/abandon)
- **Gate diagnostik**: IC (Spearman pred vs realized R) > 0.03 dengan sign stabil lintas fold.
  Jika IC ≈ 0 → outcome tak terprediksi → **ABANDON jujur** (sizing cuma tambah variance).
- Jika lolos gate: $-weighted PF > 1.408 DAN payoff > 0.74 pada OOF, trades = 16,226 (identik),
  MaxDD tidak memburuk vs −$19.84.
- Baseline: v4 uniform sizing (PF 1.408, payoff 0.74, PnL $1,443, 16,226 trades).

### Script
- `pipeline/09_train_qsize_oof.py` (baru) — build trade dataset → walk-forward train → eval sizing vs uniform.

### Validasi final
- Jika OOF lolos: validasi di **holdout BARU (>Jun 2026)**, bukan Apr–Jun (terkontaminasi).

---

## 2026-06-24 — Investigasi Lever RR: min_rr / Entry Timing / TP Extension

**Status**: IN PROGRESS

### Hipotesis
Payoff v4 = 0.74 bersifat struktural karena desain labeling+exit. Ada tiga lever yang belum dicoba:
1. **Lever 1 — min_rr naik**: Trade dengan planned RR lebih tinggi punya realized R lebih baik → filter setup RR rendah meningkatkan payoff/PF.
2. **Lever 2 — entry lebih dekat SL (price_in_range)**: Trade yang masuk saat harga dekat SL (bottom range untuk LONG) punya RR entry lebih baik → geometris naikkan planned RR tanpa ubah swing levels.
3. **Lever 3 — TP extension ke swing berikutnya**: Target TP di swing H4 kedua (lebih jauh) → realized R lebih besar saat TP tercapai, tapi WR turun karena TP lebih jauh.

**Catatan metodologi**: Lever 1 & 2 adalah diagnostic sweep post-hoc pada OOF v4 trades (16,226 trade) — BUKAN retrain. Menjawab "apakah trade dengan karakteristik X punya outcome lebih baik?" Jika ada sinyal kuat, eskalasi ke retrain. Lever 3 memerlukan modifikasi simulator.

### Yang Diubah
- Entry stack (LGBM+HMM+Guardian): **TIDAK diubah** — 16,226 trade v4 yang sama
- Lever 1: filter post-hoc `rr >= threshold` di [0.6, 0.8, 1.0, 1.2, 1.5, 2.0]
- Lever 2: filter `price_in_range` — LONG masuk hanya jika pir < X, SHORT hanya jika pir > (1-X)
- Lever 3: modify `simulate_trades_swing` untuk gunakan swing H4 kedua sebagai TP

### Baseline untuk dibandingkan
v4 OOF (0.65/0.05): WR 65.6%, PF 1.408, payoff 0.74, PnL $1,443, 16,226 trades, MaxDD −$19.84

### Target diagnostik (per lever)
- Jika PF naik >= 0.05 dengan trade reduction <= 30%: eskalasi ke retrain penuh
- Jika PF flat atau turun: lever tersebut tidak efektif, tidak perlu retrain

### Script
- scratchpad: `lever_rr_sweep.py` (Lever 1), `lever_entry_timing.py` (Lever 2), `lever_tp_extension.py` (Lever 3)

### Verifikasi Kausalitas (ATURAN 6)
- Lever 1 & 2: tidak ada fitur baru, tidak ada resample baru → N/A
- Lever 3: TP extension menggunakan swing H4 yang sama (sudah terverifikasi, +4h shift ada) → N/A

### Hasil

**Status**: COMPLETED — semua lever tidak menaikkan PF secara viable.

**Lever 1 — min_rr sweep (OOF v4 post-hoc filter):**
IC(planned_rr, realized_R) = +0.055 (ada korelasi lemah positif)

| min_rr | Trades | %keep | WR | PF | payoff |
|---|---|---|---|---|---|
| baseline (0.6) | 16,226 | 100% | 65.7% | 1.408 | 0.737 |
| 0.8 | 13,996 | 86% | 64.8% | **1.380** ↓ | 0.751 |
| 1.0 | 10,837 | 67% | 62.9% | **1.327** ↓ | 0.783 |
| 1.2 | 6,801 | 42% | 61.2% | **1.315** ↓ | 0.835 |
| **1.5** | **781** | **5%** | 60.3% | **1.696** ↑ | **1.117** |
| 2.0 | 334 | 2% | 57.8% | 1.711 | 1.250 |

**Temuan kritis:** Filter min_rr 0.8–1.2 justru MENURUNKAN PF. Trade RR rendah (0.6–0.9) punya WR 70–72% karena TP dekat → Guardian escort sering exit kecil-kecil, tapi profitable secara agregat. Membuangnya = membuang trade yang bekerja. PF baru naik di min_rr ≥ 1.5 (payoff > 1) tapi hanya 5% volume — tidak viable.

**Lever 2 — price_in_range:**
price_in_range tidak ter-capture di trade log → tidak dapat dievaluasi. Perlu modifikasi simulator. Ditunda.

**Lever 3 — TP extension ke swing H4 berikutnya:**

| Konfigurasi | Trades | WR | PF | payoff | PnL | vs baseline |
|---|---|---|---|---|---|---|
| Baseline (nearest, hold=36) | 16,226 | 65.7% | 1.408 | 0.737 | $1,443 | — |
| 3a: next swing, hold=36 | 16,121 | 65.5% | **1.397** ↓ | 0.735 | $1,391 | PF −0.011 |
| 3b: next swing, hold=72 | 15,958 | 66.9% | **1.344** ↓ | 0.665 | $1,255 | PF −0.065 |

**Root cause:** TP extension TIDAK membantu karena Guardian exit sebelum TP tercapai. Dengan TP lebih jauh, GUARDIAN_EXIT justru naik (7,582 → 7,996 → 8,120) — Guardian semakin sering intercept sebelum TP. TP extension = hanya memperpanjang "runway" yang tidak pernah dipakai karena Guardian keluar lebih dulu.

### Kesimpulan

**Semua lever gagal menaikkan PF secara viable.** Root cause tunggal: **Guardian exit mechanism adalah bottleneck**. Tidak peduli seberapa jauh TP ditetapkan, Guardian mengintercepi trade lebih awal sehingga realized R tidak berubah. Ini bukan bug — ini adalah desain Guardian escort (exit di momentum peak). Payoff 0.74 adalah properti dari arsitektur, bukan parameter yang bisa di-tune.

Implikasi: jika ingin payoff > 1, harus mengubah desain Guardian (retrain dengan objective berbeda) ATAU mengubah entry seleksi secara fundamental (LGBM label baru). Keduanya adalah effort besar dan berisiko menurunkan WR.

**Tahap 4:** TIDAK ada upgrade — tidak ada konfigurasi yang memenuhi kriteria. Baseline v4 dipertahankan.

---

## 2026-06-24 — Guardian Min-Gain Gate (Opsi A): Retrain dengan Patience Gate

**Status**: IN PROGRESS

### Hipotesis
Payoff v4 = 0.74 karena Guardian exit terlalu dini — EXIT dipicu bahkan saat trade baru naik 10–15% dari planned TP. Jika Guardian dilatih dengan constraint "EXIT hanya boleh setelah mencapai min X% dari planned TP distance", model belajar bahwa sinyal momentum-exit di fase awal trade = HOLD, bukan EXIT. Hasilnya: realized R per trade naik (winners lari lebih jauh), payoff naik, PF naik. WR sedikit turun karena lebih banyak trade jalan ke SL.

### Yang Diubah vs Guardian v2 (aktif)
1. **Min-Gain Gate**: sebelum semua EXIT rule, cek `current_pnl < tp_pct × min_gain_frac` → override HOLD. Gate = 40% TP distance (default, sweep 0.25–0.50).
2. **Training signal source**: v4 OOF (HMM 0.65/0.05, no LSTM/p_bull) — lebih representatif dari trade yang guardian operasi dalam prod. v2 dilatih pada parity+LSTM signals.
3. **Fitur baru**: `tp_progress_pct = current_pnl / tp_pct` — agar model bisa eksplisit belajar "trade progress menuju TP = X%". Kausal (current_pnl backward-looking, tp_pct dari entry).
4. **p_bull**: 0.5 (neutral) — v4 p_bull OFF, konsisten dengan inference.

### Metodologi (ATURAN 1–4)
- Guardian dilatih pada OOF v4 trades (ATURAN 2) — bukan in-sample
- Scaler di-fit per fold di dalam loop (ATURAN 3)
- Purge gap = GUARDIAN_PURGE_GAP_BARS (ATURAN 4)
- Holdout TIDAK disentuh (ATURAN 1)

### Target
- OOF PF >= 1.50 (naik dari 1.408 baseline v4)
- OOF payoff >= 0.85 (naik dari 0.74)
- OOF WR >= 60% (margin turun dari 65.7% tapi masih profitable)
- OOF trades >= 14,000 (>=86% dari 16,226 — Guardian hanya filter exit, entry tetap)

### Script
- `pipeline/06f_train_guardian_min_gain.py` (baru)

### Verifikasi Kausalitas (ATURAN 6)
- `tp_progress_pct` = current_pnl / tp_pct: keduanya dihitung dari data backward-looking (current_pnl dari harga saat ini, tp_pct dari entry & sl yang sudah diketahui saat masuk) → KAUSAL, N/A ATURAN 6.

### Hasil Iterasi 1 (min_gain_frac=0.40)

Label distribution training: HOLD 80.2%, PARTIAL 0%, EXIT 19.8%

| Metrik | Baseline v4 | Min-Gain 40% | Delta |
|---|---|---|---|
| Trades | 16,226 | 14,848 | −8.5% |
| WR | 65.7% | **46.0%** | **−19.7pp** |
| PF | 1.408 | **1.165** | **−0.243** |
| Payoff | 0.737 | **1.367** | **+0.63** |
| PnL | $1,443 | $606 | −$837 |
| MaxDD | −$19.84 | −$40.77 | −$21 |

Outcomes: GUARDIAN_MOMENTUM_FLOOR=7,224 / LOSS=5,634 / TIMEOUT=1,740 (tidak ada GUARDIAN_EXIT)

**Analisis:** Gate 40% berhasil menaikkan payoff dari 0.74 → 1.37 (avg win > avg loss). Tapi WR crash dari 65.7% → 46.0% karena baseline punya 7,582 GUARDIAN_EXIT positif — trade yang momentum-peak di +0.3R lalu exit tipis positif. Dengan gate 40%, trade-trade ini tidak bisa exit → lanjut ke SL. PF turun dari 1.408 → 1.165.

**Root cause:** Gate 40% terlalu ketat. Terlalu banyak trade yang sebelumnya "Guardian exit kecil-positif" sekarang berakhir di SL. Payoff membaik tapi WR drop terlalu besar untuk PF naik.

**Breakeven WR** di payoff 1.367 = 42.3%. WR 46% > 42.3% → masih profitable, tapi margin hanya 3.7pp (vs baseline 8.1pp). Total PnL turun 58%.

### Iterasi berikutnya: min_gain_frac lebih rendah

Perlu temukan "sweet spot" gate yang cukup tinggi untuk angkat payoff tapi tidak terlalu memotong WR. Coba sweep [0.10, 0.15, 0.20, 0.25, 0.30] — mungkin gate ringan (15–20%) cukup untuk filter exit terlalu dini tanpa korbankan WR drastis.

---

## 2026-06-25 — Holdout 24 Juni 2026 WIB: ic32_regime_v4 (model live)

**Status**: COMPLETED

### Hipotesis
Evaluasi performa model yang di-deploy pada 24 Juni 2026 (ic32_regime_v4) untuk hari pertama
operasional. Data Jun 22 sebelumnya tidak tersedia di holdout; data di-fetch fresh dari Binance.

### Yang Diubah
- Bukan eksperimen baru — evaluasi OOS untuk model live (07d_holdout_v4_jun24.py)
- Period: 24 Juni 2026 WIB = 2026-06-23 17:00 UTC s.d. 2026-06-24 17:00 UTC (entry window)
- Data di-fetch 2026-06-25 untuk 21 koin, pipeline: extend raw -> clean -> engineer -> HMM

### Hasil — 24 Juni 2026 WIB (ic32_regime_v4: LGBM+HMM 0.65/0.05+Guardian v2)

| Metrik | Nilai |
|--------|-------|
| Trades | 6 |
| Short / Long | 5 / 1 |
| Win Rate | 83.3% (5W / 1L) |
| Profit Factor | 2.645 |
| Net PnL | +$0.96 |
| PnL/trade | +$0.1604 |
| Max Drawdown | -$0.59 |
| Long WR | 100.0% (1 trade, DOGEUSDT) |
| Short WR | 80.0% (5 trades) |

**Per-koin aktif:**
| Koin | Dir | Outcome | PnL |
|------|-----|---------|-----|
| BTCUSDT | SHORT | GUARDIAN_MOMENTUM_EXIT | +$0.41 |
| ARBUSDT | SHORT | GUARDIAN_EXIT | +$0.37 |
| ONDOUSDT | SHORT | GUARDIAN_EXIT | +$0.33 |
| SUIUSDT | SHORT | GUARDIAN_EXIT | +$0.23 |
| DOGEUSDT | LONG | GUARDIAN_EXIT | +$0.21 |
| TAOUSDT | SHORT | LOSS | -$0.59 |

**Exit breakdown:** GUARDIAN_EXIT=4 (66.7%), GUARDIAN_MOMENTUM_EXIT=1 (16.7%), LOSS=1 (16.7%)

### Kesimpulan
Hari pertama operasional v4: 6 trades, 5 menang. PF 2.645, PnL +$0.96. Short-heavy (5/6 trades)
konsisten dengan kondisi bearish/ranging pasar. Satu-satunya loss (TAOUSDT, SHORT) adalah anomali
kecil. Volume rendah (6 trade/hari dari 21 koin) normal untuk evaluasi satu hari.

Catatan: sample satu hari terlalu kecil untuk kesimpulan statistik. Ini snapshot hari pertama,
bukan validasi long-term.

### Script
- `pipeline/07d_holdout_v4_jun24.py`
- Hasil: `models/runs/ic32_regime_v2_parity/holdout_v4_jun24.json`

---

## 2026-07-02 — Ablation Dekorelasi Geometri + Prinsip No-Cross-Model-Features (37f/33f)

**Status**: DONE (OOF) — kandidat 33f layak lanjut fase threshold sweep

### Latar Belakang
Insiden live 1–2 Jul: 19/20 trade SHORT saat BTC pump +5.66%, 8 SL beruntun (-$14.63).
Deep-dive SHAP pada `_lgbm_proba` live (parity eksak 2.2e-16): keluarga geometri-range
(`dist_liq_50x/20x_*`, `Buy_Liq`, `Sell_Liq`, `dist_from_8h_high`) = 56.5% total gain,
`dist_liq_50x_long` sendirian +1.04 log-odds ke SHORT saat pump. Fitur konteks market
hanya 4.4% gain. Detail: `reports/TRADE_ANALYSIS_REPORT.md` bagian D.

### Hipotesis
1. `hmm_regime_enc` sebagai fitur input melanggar prinsip no-cross-model-features;
   drop tidak menurunkan PF (informasi regime tetap masuk via gating eksternal).
2. Duplikat geometri (ρ=0.91 antar pasangan 50x/20x; IC 20x lebih lemah: -0.043 vs
   -0.070, +0.028 vs +0.057) membuat model over-trust mean-reversion → drop 4 duplikat
   (`dist_liq_20x_long/short`, `Buy_Liq`, `Sell_Liq`) menyeimbangkan long/short.

### Yang Diubah
- `abl_no_hmm_37f` = 38f v5 − `hmm_regime_enc`
- `abl_no_hmm_geo_33f` = 37f − 4 duplikat geometri
- Trainer: `pipeline/model/experiments/train_lgbm_custom.py` (8-fold rolling purged, identik baseline)

### Hasil (OOF, gate HMM 0.65/0.05, Guardian OFF, evaluator identik)

| Run | Trades | WR | PF | PnL | Long PF | Short PF | MaxDD |
|---|---|---|---|---|---|---|---|
| baseline 38f v5 | 16,202 | 53.6% | 1.580 | $4,588 | 1.496 | 1.667 | -43.5 |
| 37f | 14,624 | 53.8% | 1.581 | $4,129 | 1.527 | 1.634 | -53.6 |
| **33f** | 10,456 | 52.1% | **1.601** | $3,016 | **1.573** | 1.629 | **-34.7** |

Replay insiden (proba pada snapshot sinyal live 1–2 Jul, n=815):
- p_short rata2 pada 14 short yang rugi: live 0.650 → 37f 0.622 → **33f 0.538**
- Masih fire SHORT≥0.65: live 6/14 → 37f 3/14 → **33f 1/14**
- Kandidat thr 0.65 semua sinyal: live S7/L5 → 37f S4/L7 → **33f S1/L4** (bias terbalik ke LONG saat pump)

Fold F1 stabil (37f: 0.551–0.580; 33f: 0.524–0.553). Trade count 33f turun 35% pada
operating point 0.65/0.05 (kalibrasi proba berubah) → butuh sweep base/delta ulang
sebelum kesimpulan PnL final.

### Kesimpulan
- Drop `hmm_regime_enc` ~gratis (PF 1.581 vs 1.580) → prinsip no-cross-model terpenuhi.
- Dekorelasi geometri: PF naik, keseimbangan Long/Short PF terbaik (gap 0.17→0.06),
  MaxDD terbaik, dan hampir mengeliminasi short-saat-pump pada replay insiden.
- Biaya: partisipasi turun pada threshold lama — perlu operating point baru.

### Script & Artefak
- `models/runs/abl_no_hmm_37f/`, `models/runs/abl_no_hmm_geo_33f/`
- Scorecard: `models/runs/abl_no_hmm_geo_33f/scorecard_compare.json`

---

## 2026-07-03 — IC Test 6 Kandidat Fitur Baru + Marginal Test (basis 37f)

**Status**: DONE (OOF) — `ofi_z_score` lolos & bernilai tambah nyata; TIDAK memperbaiki bias short-saat-pump

### Hipotesis
User minta uji 6 kandidat fitur konteks: `atr_percentile_h1/h4`, `funding_rate_extreme`,
`cross_asset_correlation_spike`, `recent_hmm_transition`, `volume_anomaly`/`ofi_acceleration`
(cek skala), `realized_vol_spike`. Basis **37f** (bukan 33f — dipakai belakangan/ditunda per arahan user).

### Yang Diuji (IC, 8 slice kronologis, Spearman vs arah label, 861.289 baris/21 koin)

| Fitur | IC_mean | ICIR | Sign stability | Lolos? |
|---|---|---|---|---|
| `ofi_z_score` (sudah ada, nganggur) | -0.0235 | **-5.01** | **100%** | ✅ kuat |
| `ofi_accel_z` (baru, z-score per-koin) | -0.0056 | -1.47 | 87.5% | lemah |
| `vol_spike_zscore` (sudah ada, nganggur) | 0.0081 | 0.98 | 75% | borderline |
| `atr_percentile_h1` (sudah ada, nganggur) | 0.0099 | 0.69 | 75% | borderline |
| `recent_hmm_transition` (baru) | 0.0035 | 0.58 | 75% | borderline — **fitur output-HMM, langgar prinsip no-cross-model** |
| `atr_percentile_h4`, `realized_vol_spike_24h`, `corr_btc/mkt_spike_24h` | 0.003–0.005 | 0.24–0.32 | 50–62.5% | gagal (tanda tak konsisten) |
| `funding_rate_z` / `funding_rate_extreme_bin` | 0 | — | — | **tidak bisa diuji** — `funding_rate` 100% NaN di `features_v3.parquet` meski data mentah ada di `data/training/funding_rate/{coin}_8h.parquet` (gap join pipeline) |

Pembanding: fitur market yang sudah dipakai di 37f — `btc_ret_24h` IC -0.0185, `mkt_breadth_4h` IC -0.0593.
`ofi_z_score` mengungguli `btc_ret_24h` yang sudah dipakai produksi.

### Marginal Test (OOF, gate HMM 0.65/0.05, Guardian OFF, evaluator identik)

| Run | Trades | WR | PF | PnL | MaxDD |
|---|---|---|---|---|---|
| baseline 37f | 14,624 | 53.8% | 1.581 | $4,129 | -53.6 |
| **38f** (37f + `ofi_z_score`) | 15,339 | 53.9% | 1.581 | **$4,369** (+$240) | **-39.9** (lebih baik) |
| **40f** (37f + `ofi_z_score` + `vol_spike_zscore` + `atr_percentile_h1`) | 16,744 | 53.6% | **1.591** (+0.010) | **$4,766** (+$637) | -42.7 |

### Replay Insiden 1–2 Jul (n=815 snapshot sinyal live) — TEMUAN PENTING

| Model | Masih fire SHORT≥0.65 (dari 14 short live) | Kandidat SHORT/LONG @thr 0.65 (semua sinyal) |
|---|---|---|
| 37f baseline | 3/14 | S=4 / L=7 |
| 38f (+ofi_z_score) | 4/14 | S=7 / L=6 |
| 40f (+3 fitur) | 3/14 | S=9 / L=8 |

**`ofi_z_score` dan kombinasi 3 fitur memperbaiki metrik OOF full-period (PnL, MaxDD, PF sedikit)
TAPI memperbanyak kandidat SHORT saat insiden pump 1–2 Jul, bukan mengurangi** — berlawanan
dengan efek dekorelasi geometri (33f) yang menghapus hampir semua short-saat-pump di replay
yang sama (lihat eksperimen di atas). Fitur order-flow/volatilitas ini menambah edge yang
independen dari akar penyebab insiden (dominasi keluarga geometri mean-reversion); tidak
menggantikan fix 33f.

### Kesimpulan
- `ofi_z_score` **layak masuk fitur produksi** — IC terkuat & paling stabil dari seluruh kandidat
  baru, marginal PnL +$240 dengan MaxDD membaik, gratis (sudah ter-engineer).
- `vol_spike_zscore` + `atr_percentile_h1` menambah marginal lift lebih kecil (+$397, PF +0.010)
  di atas `ofi_z_score` saja — borderline, bukan prioritas tinggi.
- `recent_hmm_transition` sengaja TIDAK dimasukkan kandidat retrain — melanggar prinsip
  no-cross-model-features user meski ICIR borderline lolos.
- `funding_rate_extreme` tidak bisa dievaluasi — perlu perbaikan pipeline join `funding_rate`
  dulu (gap terpisah, bukan gap data).
- 4 kandidat lain (atr_pct_h4, realized_vol_spike, corr spike BTC/mkt) gagal syarat stabilitas.

### Script & Artefak
- `models/runs/marg_37f_plus_ofiz/`, `models/runs/marg_37f_plus3/`
- Scorecard: `models/runs/marg_37f_plus3/scorecard_compare.json`

---

## 2026-07-03 — Audit Leakage + Marginal Test di atas 33f (ofi_z_score, atr_percentile_h1, vol_spike_zscore)

**Status**: DONE (OOF) — leakage `vol_spike_zscore` terkonfirmasi kode DAN terbukti empiris; `ofi_z_score`+`atr_percentile_h1` bersih & bernilai tambah nyata

### Audit Leakage (code review, sebelum training)
User minta kombinasi 33f + `ofi_z_score` + `vol_spike_zscore` + `atr_percentile_h1`, dengan syarat
pastikan tidak ada data leakage. Hasil audit:

- **`vol_spike_zscore` — LEAKAGE TERKONFIRMASI.** Dipakai langsung sebagai parameter
  `momentum_vol_spike` di dalam `swing_based_labeling()` ([core/features.py:1922](../core/features.py#L1922),
  dipanggil dari `engineer_features()` saat `add_label=True` — jalur produksi asli, bukan
  hipotetis). Saat `|vol_spike_zscore| >= 1.5`, label bar itu beralih dari jalur swing-based
  ke jalur momentum (TP/SL berbasis ATR, RR dipaksa 1.33, kedua arah selalu valid, tie-break
  via `price_accel_1h`). Fitur ini secara struktural ikut MEMBENTUK kolom `label` — bukan
  cuma memprediksinya. Model berisiko belajar mendeteksi rezim-pelabelan, bukan sinyal pasar.
- **`ofi_z_score`, `atr_percentile_h1` — bersih.** Tidak muncul di `swing_based_labeling()`
  maupun `structural_label_filter()` (yang hanya pakai `price_in_range`). Aman dipakai sbg fitur.

### Marginal Test (OOF, gate HMM 0.65/0.05, Guardian OFF, basis 33f — bukan 37f)

| Run | Trades | WR | PF | PnL | MaxDD |
|---|---|---|---|---|---|
| baseline 33f | 10,456 | 52.1% | 1.601 | $3,016 | -34.7 |
| **35f_clean** (33f + `ofi_z_score` + `atr_percentile_h1`) | 9,853 | 52.7% | **1.650** (+0.049) | $3,049 | -35.1 |
| **36f_FLAGGED** (35f_clean + `vol_spike_zscore`) | 11,422 | 51.9% | **1.597** (-0.053 vs clean) | $3,317 | -39.5 (lebih buruk) |

**Leakage terbukti empiris**: menambah `vol_spike_zscore` menaikkan volume trade (+1.569 vs
clean) dan PnL nominal, tapi **menurunkan PF di bawah bahkan baseline 33f murni** (1.597 <
1.601 < 1.650). Pola persis yang diprediksi audit kode: fitur ini membuat lebih banyak trade
"lolos" (mendeteksi rezim-pelabelan momentum-path yang selalu valid dua arah) namun kualitas
riil trade (PF, MaxDD) memburuk — bukan edge pasar asli.

### Replay Insiden 1–2 Jul (n=815, cek fix 33f tidak rusak oleh fitur baru)

| Model | Masih fire SHORT≥0.65 (dari 14 short live) | Kandidat SHORT/LONG @thr 0.65 |
|---|---|---|
| 33f baseline | 1/14 | S=1 / L=4 |
| 35f_clean | 1/14 | S=1 / L=6 |
| 36f_flagged | 1/14 | S=2 / L=5 |

Fix dekorelasi geometri (33f) tetap terjaga di kedua varian — short-saat-pump tidak muncul
kembali akibat penambahan fitur ini.

### Kesimpulan
- **`ofi_z_score` + `atr_percentile_h1` di atas 33f = kandidat LGBM terbaik sejauh ini**:
  PF 1.650 (tertinggi dari semua run — 38f produksi 1.580, 33f 1.601, 37f+3 1.591), trade
  count & MaxDD hampir tak berubah, dan fix short-saat-pump tetap utuh.
- **`vol_spike_zscore` TIDAK direkomendasikan sebagai fitur LGBM** — leakage terkonfirmasi
  kode + empiris. Untuk dipakai dengan aman, `swing_based_labeling()` perlu diubah agar
  tidak lagi memakainya sebagai gate label (di luar scope eksperimen ini).

### Script & Artefak
- `models/runs/marg_33f_plus2_clean/`, `models/runs/marg_33f_plus3_flagged/`
- Scorecard: `models/runs/marg_33f_plus3_flagged/scorecard_compare.json`

---

## 2026-07-03 — Otak-atik Kombinasi Dekorelasi Keluarga Geometri (basis 37f)

**Status**: DONE (OOF) — `v2` (keep 20x, buang 50x+Buy/Sell_Liq) ternyata TERBAIK, mengalahkan `v1`/33f

### Hipotesis
User minta uji kombinasi alternatif drop/keep dari 3 pasangan kolinear
(`dist_liq_50x_*`, `dist_liq_20x_*`, `Buy_Liq`/`Sell_Liq`) selain pilihan `v1` (33f) yang
sudah ada. IC screen sebelumnya menunjukkan `50x` selalu ber-IC lebih kuat dari `20x`
(mis. -0.0697 vs -0.0430 sisi long) — hipotesis naifnya `v1` (keep 50x) akan selalu menang.

### Varian (semua basis 37f, drop dari 7 fitur keluarga geometri)

| Var | Drop | Keep | n |
|---|---|---|---|
| v1 (33f, existing) | 20x_long, 20x_short, Buy_Liq, Sell_Liq | 50x_long, 50x_short, dist_from_8h_high | 33f |
| v2 (swap leverage) | 50x_long, 50x_short, Buy_Liq, Sell_Liq | 20x_long, 20x_short, dist_from_8h_high | 33f |
| v3 (keep buy/sell) | 20x_long, 20x_short | 50x_long, 50x_short, Buy_Liq, Sell_Liq, dist_from_8h_high | 35f |
| v4 (keep buy/sell) | 50x_long, 50x_short | 20x_long, 20x_short, Buy_Liq, Sell_Liq, dist_from_8h_high | 35f |

### Hasil (OOF, gate HMM 0.65/0.05, Guardian OFF)

| Run | Trades | PF | PnL | Long PF | Short PF | MaxDD |
|---|---|---|---|---|---|---|
| 37f (semua 7 fitur) | 14,624 | 1.581 | $4,129 | 1.527 | 1.634 | -53.6 |
| v1/33f (keep 50x) | 10,456 | 1.601 | $3,016 | 1.573 | 1.629 | -34.7 |
| **v2 (keep 20x, no buy/sell)** | 9,233 | **1.633** | $2,868 | **1.633** | **1.634** | -43.6 |
| v3 (keep 50x+buy/sell) | 11,139 | 1.574 | $3,049 | 1.500 | 1.654 | -36.1 |
| v4 (keep 20x+buy/sell) | 9,198 | 1.562 | $2,544 | 1.592 | 1.533 | -51.3 |

Replay insiden 1–2 Jul (n=815, 14 short live-fire): `p_short` mean 37f 0.622 → v1 0.538 →
**v2 0.513** (terendah) → v3 0.552 → v4 0.543. Masih fire ≥0.65: 37f 3/14, v1/v2/v3 1/14,
v4 2/14.

### Temuan Penting — IC ranking TIDAK memprediksi hasil ensemble

- **`v2` (keep 20x, bukan 50x) mengungguli `v1`/33f** meski IC single-feature `20x` lebih
  lemah dari `50x` (-0.0430 vs -0.0697). PF +0.032, dan Long/Short PF `v2` paling seimbang
  dari SEMUA varian yang pernah diuji (1.633 vs 1.634 — gap 0.001). Trade count/PnL lebih
  rendah dari v1 (kalibrasi lebih ketat lagi), tapi kualitas per-trade tertinggi.
- **`Buy_Liq`/`Sell_Liq` konsisten MERUGIKAN saat digabung dengan pasangan leverage manapun**:
  menambahkannya ke 50x (v3 vs v1) menurunkan PF -0.027; menambahkannya ke 20x (v4 vs v2)
  menurunkan PF **-0.071** — penurunan terbesar dari seluruh eksperimen ablation. `v4` bahkan
  lebih buruk dari baseline 37f (1.562 < 1.581).
- Insiden pump tetap teratasi baik di v1 maupun v2 (1/14 fire); v3/v4 (yang menyimpan
  Buy_Liq/Sell_Liq) sedikit lebih buruk (3/14, 2/14) — konsisten dgn PF-nya yang lebih rendah.

### Kesimpulan (revisi setelah v5, lihat lanjutan di bawah)
- Pelajaran metodologi: screening IC single-feature bagus untuk filter awal (buang yang jelas
  lemah/tak stabil), tapi urutan/pemilihan akhir antar fitur berkorelasi tinggi harus diuji
  langsung di ensemble — IC ranking bisa salah arah.

### Lanjutan — v5: buang KEDUA pasangan leverage, keep hanya Buy_Liq+Sell_Liq (33f)

| Run | Trades | PF | PnL | Long PF | Short PF | MaxDD |
|---|---|---|---|---|---|---|
| 37f (semua 7) | 14,624 | 1.581 | $4,129 | 1.527 | 1.634 | -53.6 |
| v1/33f (keep 50x) | 10,456 | 1.601 | $3,016 | 1.573 | 1.629 | -34.7 |
| v2 (keep 20x) | 9,233 | 1.633 | $2,868 | 1.633 | 1.634 | -43.6 |
| **v5 (keep Buy/Sell_Liq saja)** | **4,090** | **1.679** | $1,371 | 1.660 | 1.702 | -42.2 |

Replay insiden: `p_short` mean turun terus 37f 0.622 → v1 0.538 → v2 0.513 → **v5 0.475**
(terendah). Masih fire ≥0.65: 37f 3/14 → v1/v2 1/14 → **v5 0/14** — insiden pump nol short palsu.

### Pola yang terungkap — PF naik terus, trade count runtuh terus

```
37f (7 fitur) : trades 14,624 | PF 1.581
v1 (3 fitur)  : trades 10,456 | PF 1.601
v2 (3 fitur)  : trades  9,233 | PF 1.633
v5 (3 fitur)  : trades  4,090 | PF 1.679   <- PF tertinggi, trade paling sedikit (-72% dari 37f)
```

Makin sedikit fitur "vote berulang" untuk sinyal overextension yang sama, makin jarang
probabilitas menembus threshold 0.65 tetap (kalibrasi makin ketat — lihat penjelasan
kalibrasi di percakapan sebelumnya), TAPI trade yang lolos makin berkualitas. **v5 bukan
otomatis "terbaik"** — PnL absolut ($1.371) jauh di bawah semua varian lain karena volume
terlalu tipis di threshold saat ini. Untuk menilai v5 secara adil butuh sweep base/delta
HMM khusus (belum dieksekusi, sama seperti catatan di eksperimen v1/33f).

### Kesimpulan Final
- **`v5` = PF terbaik mutlak (1.679) dan satu-satunya varian dengan 0 short palsu saat
  insiden pump**, tapi trade count paling sedikit — kandidat menarik HANYA jika habis
  di-threshold-sweep ulang (belum dikerjakan).
- **`v2` tetap kandidat paling seimbang tanpa perlu sweep ulang** — PF 1.633 dgn trade count
  masih 63% dari 37f (vs v5 cuma 28%).
- Urutan PF murni keluarga geometri (opsi 3-fitur): v5 (Buy/Sell_Liq) 1.679 > v2 (20x) 1.633
  > v1 (50x) 1.601. Pola konsisten: makin jauh dari "duplikat leverage" (50x/20x, korelasi
  0.91), makin tinggi PF tapi makin sedikit trade.

### Script & Artefak
- `models/runs/geo_v2_swap50to20/`, `geo_v3_keep_buysell/`, `geo_v4_keep20_buysell/`, `geo_v5_only_buysell/`
- Scorecard: `models/runs/geo_v5_only_buysell/scorecard_compare.json`

---

## 2026-07-03 — KOREKSI: Bug Threshold Flat 0.65 di Semua Script "Replay Insiden"

**Status**: DONE — angka drill-down insiden 1-2 Jul di eksperimen SEBELUMNYA (marginal test,
audit leakage, otak-atik geometri) terlalu optimis; scorecard OOF utama TIDAK terpengaruh.

### Bug
Semua skrip ad-hoc `replay_*.py` / `trade_by_trade_compare.py` sesi ini memakai **threshold
flat 0.65** untuk menentukan keputusan SHORT/LONG per model kandidat. Sistem live yang
sebenarnya (dan seluruh scorecard OOF resmi via `sweep_hmm_mkt_breadth.run_config`) memakai
**gating per-state HMM** dari `inference_config.json` (`base=0.65, delta=0.05`):

| State | thr_long | thr_short |
|---|---|---|
| 0 TRENDING_DOWN | 0.70 | **0.60** |
| 1 RANGING_LOW_VOL | 0.675 | 0.625 |
| 2 RANGING_HIGH_VOL | 0.625 | 0.675 |
| 3 TRENDING_UP | 0.60 | 0.70 |

Selama insiden 1-2 Jul, `hmm_regime_enc` didominasi state 0 (thr_short cuma 0.60, lebih
longgar dari flat 0.65) — jadi beberapa sinyal yang tadinya dianggap "tersaring" (SUI 0.601,
ADA 0.637) ternyata tetap lolos di gating asli.

### Dampak — angka koreksi (33f, 35f-clean, v2; per-state gate benar)

| | Klaim SEBELUMNYA (flat 0.65, salah) | **Terkoreksi (per-state, benar)** |
|---|---|---|
| 33f masih SHORT dari 14 live-fire | 1/14 | **3/14** |
| 35f-clean masih SHORT dari 14 live-fire | 1/14 | **3/14** |
| v2 masih SHORT dari 14 live-fire | 1/14 | **2/14** |
| PnL 35f-clean pada 11 trade riil 1-2 Jul | -$0.88 (proyeksi keliru) | **-$3.95** |
| Perbaikan vs live (-$13.11) | +$12.23 (keliru) | **+$9.17** |

v5 (0/14, klaim "nol short palsu") BELUM diverifikasi ulang dgn per-state gate — kemungkinan
juga naik dari 0, catat sebagai **belum dikonfirmasi**.

### Yang TIDAK berubah
- Seluruh scorecard OOF (PF 1.580/1.601/1.633/1.650/1.679 dst., trades, PnL, MaxDD di ratusan
  ribu baris) — **valid**, sudah pakai per-state gate sejak awal.
- Urutan kualitas kandidat (35f-clean > v2 > 33f > 37f > live 38f) — tidak berubah.
- Arah kesimpulan (35f-clean jauh mengurangi short-saat-pump) — tetap benar, cuma
  magnitude drill-down insiden yang direvisi turun.

### Pelajaran
Skrip ilustratif/ad-hoc untuk drill-down harus tetap pakai gating identik dengan evaluator
resmi (`sweep_hmm_mkt_breadth`), bukan simplifikasi threshold tunggal — meski cuma untuk
"ilustrasi", angka yang salah tetap bisa menyesatkan keputusan.

### Script & Artefak
- `replay_full_correct.py`, `trade_by_trade_v2_correct.py` (scratchpad session ini)

---

## 2026-07-03 — Riset Fitur Regime-Adaptive (basis 35f-clean) — HASIL NEGATIF, didokumentasikan

**Status**: DONE (OOF + replay insiden, per-state gate benar) — `trend_strength`/`no_demand`
DITOLAK sbg tambahan; 35f-clean tetap kandidat terbaik

### Tujuan
User minta cari fitur agar model tidak condong ke satu arah terus saat regime berubah
(short saat pasar mulai bullish). Basis tetap 35f-clean.

### Kandidat diaudit (8, semua dicek leakage di `swing_based_labeling`/`structural_label_filter`)

| Fitur | Bersih? | IC standar (ICIR) |
|---|---|---|
| `wyckoff_phase` | **TIDAK** — dibangun dari `price_in_range` yg dipakai label filter | 1.41 (dikesampingkan) |
| `spring_upthrust` | **TIDAK** — idem | 1.12 (dikesampingkan) |
| `effort_vs_result` | Bersih | 1.11 |
| `trend_strength` = (ema7_h4-ema50_h4)/atr_h4 | Bersih | -0.38 (lemah overall) |
| `no_demand` | Bersih | 0.37 |
| `no_supply` | Bersih | 0.23 |
| `CHoCH` | Bersih | 0.17 |
| `bars_since_BOS` | Bersih | -0.10 |

`dist_swing_high`/`dist_swing_low` dikesampingkan tanpa diuji — konsep sama persis
`Buy_Liq`/`Sell_Liq` yg terbukti merugikan (eksperimen sebelumnya).

### Temuan awal yang menjanjikan (SEBELUM validasi OOF)
IC `trend_strength` melonjak drastis di 6 bar setelah `CHoCH` (structural reversal):
**-0.008 (jauh dari reversal) → +0.032 (dekat reversal)**, delta terbesar dari semua kandidat.
Distribusi label juga mengungkap nuansa: setelah CHoCH BULLISH, SHORT masih 13.7% vs LONG
11.0% (reversal palsu umum secara historis) — sebaliknya setelah CHoCH BEARISH, LONG 14.1%
vs SHORT 13.1%. Hipotesis: `trend_strength` bisa membantu model membedakan reversal genuine
vs gagal tepat saat regime baru berubah.

### Validasi OOF + Replay Insiden — HIPOTESIS DITOLAK

| Run | Trades | PF | PnL | MaxDD |
|---|---|---|---|---|
| 35f-clean (baseline) | 9,853 | **1.650** | $3,049 | -35.1 |
| 36f (+`trend_strength`) | 10,446 | 1.617 (turun) | $3,059 | -39.9 (turun) |
| 37f (+`trend_strength`+`no_demand`) | 10,234 | 1.581 (turun lebih) | $2,869 (turun) | -40.3 |

Replay insiden 1-2 Jul (per-state gate BENAR, bukan flat 0.65):
- Masih SHORT dari 14 live-fire: 35f-clean **3/14** → 36f/37f **4/14** (lebih buruk)
- Proyeksi PnL 11 trade riil: 35f-clean **-$3.95** → 36f/37f **-$5.10** (lebih buruk)

**`trend_strength` dan `no_demand` MEMPERBURUK baik performa umum maupun insiden spesifik**,
berlawanan dengan sinyal IC-kondisional yang menjanjikan.

### Analisis kegagalan
`trend_strength` = (ema7_h4-ema50_h4)/atr_h4 adalah indikator **lagging** (berbasis EMA).
Screening IC "6 bar setelah CHoCH" menangkap efek RATA-RATA di seluruh jendela 6 bar —
kemungkinan sinyal positifnya baru muncul di bar ke-4-6 (setelah EMA sempat menyesuaikan),
BUKAN di bar 0-2 (awal reversal, persis situasi insiden 1-2 Jul) di mana EMA7 masih
mencerminkan tren LAMA. Screening IC-kondisional yang saya buat tidak cukup granular untuk
menangkap perbedaan ini — pelajaran metodologi baru, senada dgn temuan "IC ranking bisa
salah arah" di eksperimen geometri sebelumnya.

### Kesimpulan
- `trend_strength`, `no_demand` **DITOLAK** sbg tambahan fitur — regresi di OOF dan insiden.
- **35f-clean tetap kandidat LGBM terbaik** (PF 1.650, tidak berubah).
- Fitur regime-adaptive yang genuine belum ditemukan lewat 8 kandidat ini; kemungkinan perlu
  indikator LEADING (bukan lagging berbasis EMA) — mis. rate-of-change volume/OFI yang
  presisi di bar pertama breakout, bukan rata-rata beberapa bar. Belum diuji.
- Screening IC-kondisional (dekat vs jauh dari event) perlu dipersempit granularitasnya
  (per-bar-offset, bukan window rata-rata 6 bar) sebelum dipercaya untuk kandidat berikutnya.

### Script & Artefak
- `models/runs/marg_35f_plus_trend/`, `marg_35f_plus_trend_nodemand/`
- Scorecard: `models/runs/marg_35f_plus_trend_nodemand/scorecard_compare.json`

---

## 2026-07-03 — Riset Fitur Leading Indicator Gen-2 (basis 35f-clean) — HASIL POSITIF

**Status**: DONE (OOF + replay insiden per-state gate benar) — `absorption_at_swing` +
`vwdp` bersama-sama mengungguli 35f-clean di scorecard DAN insiden. Kandidat baru terkuat.

### Latar belakang
Setelah `trend_strength`/`no_demand` gagal (entri sebelumnya), dicari kandidat lain dengan
metodologi diperbaiki: leakage-check di CALL SITE `engineer_features` (bukan cuma dalam
fungsi labeling — `price_accel_1h` HAMPIR lolos jadi kandidat, ternyata dipakai sbg
`momentum_price_accel` di `swing_based_labeling`, dikecualikan) + IC per-OFFSET-bar dari
event `MSB_BOS` (bar 0,1,2...5 terpisah, bukan rata-rata window — pelajaran dari kegagalan
`trend_strength`).

### Kandidat diuji (5, semua confirmed bersih dari label construction)

| Fitur | IC standar (ICIR) | Korelasi vs geometry family tersisa |
|---|---|---|
| **`absorption_at_swing`** | **-0.029 (ICIR -7.70, stabil 100%)** — terkuat sesi ini | 0.06-0.11 (aman) |
| **`vwdp`** | -0.024 (ICIR -4.37, 100%) | 0.15-0.26 (aman) |
| `hidden_divergence` | +0.019 (ICIR 2.53, 100%) | -0.17 s/d 0.22 |
| `vwdp_smooth` | -0.008 (ICIR -2.48) | — |
| `SFP_sweep` | +0.002 (lemah, stabilitas 62.5%) | paling rendah, tapi tak signifikan |

Pola per-offset-bar: TIDAK ada yg murni "leading" (kuat di bar-0, lalu melemah) — semua
menguat bertahap seiring waktu (karakter window-based). Beda dgn `trend_strength`: **arah
sudah benar sejak bar-0**, tidak berbalik, cuma lebih lemah di awal — risiko lebih kecil.

### Hasil Marginal Test (OOF, gate HMM 0.65/0.05, Guardian OFF)

| Run | Trades | PF | PnL | Long PF | Short PF | MaxDD |
|---|---|---|---|---|---|---|
| 35f-clean (baseline) | 9,853 | 1.650 | $3,049 | 1.601 | 1.698 | -35.1 |
| 36f (+`absorption_at_swing` saja) | 11,256 | 1.600 (turun) | $3,239 | 1.562 | 1.637 | -33.9 |
| **37f (+`absorption_at_swing`+`vwdp`)** | 11,272 | **1.641** | **$3,410** (+$360) | 1.616 | 1.668 | -35.3 |

`vwdp` memulihkan PF yg turun akibat `absorption_at_swing` sendirian (1.600→1.641), sambil
mempertahankan trade count lebih tinggi dan PnL tertinggi dari SEMUA kandidat 35f-family
yg pernah diuji.

### Replay Insiden 1-2 Jul (per-state gate BENAR sejak awal — tidak perlu koreksi ulang)

| Model | Masih SHORT dari 14 live-fire | PnL proyeksi 11 trade riil |
|---|---|---|
| 35f-clean (baseline) | 3/14 | -$3.95 |
| 36f (+absorption saja) | 3/14 (sama) | -$3.69 (tipis membaik) |
| **37f (+absorption+vwdp)** | **2/14 (terbaik)** | **-$2.04 (terbaik)** |

### Kesimpulan
- **`absorption_at_swing` + `vwdp` bersama = kandidat LGBM terkuat baru**, mengungguli
  35f-clean di scorecard umum (PnL +$360, PF hampir setara) DAN insiden spesifik (SHORT
  palsu 3→2, PnL insiden -$3.95→-$2.04).
- Sinergi menarik: `absorption_at_swing` sendirian justru menurunkan PF (1.650→1.600),
  tapi begitu digabung `vwdp`, PF pulih ke 1.641 sambil PnL naik lebih tinggi lagi —
  kedua fitur saling melengkapi (order-flow di swing level + wick-adjusted pressure).
- Belum diuji: `hidden_divergence` (IC positif, ICIR 2.53) sbg tambahan lebih lanjut.
- 34-37f (35f-clean+2f) ini BELUM di-threshold-sweep ulang — trade count naik 14% vs
  35f-clean, jadi PF sudah lebih fair dibanding perbandingan 35f-clean vs 33f/37f
  sebelumnya (yg volume-nya beda jauh).

### Script & Artefak
- `models/runs/marg_35f_plus_absorb/`, `marg_35f_plus_absorb_vwdp/`
- Scorecard: `models/runs/marg_35f_plus_absorb_vwdp/scorecard_compare.json`

---

## 2026-07-03 — Round 3: IC Test Pool Luas + hidden_divergence/ofi_momentum_ratio (DITOLAK) + Replay Jendela 30 Jun-2 Jul

**Status**: DONE — kandidat baru ditolak; **37f (35f-clean+`absorption_at_swing`+`vwdp`) tetap
kandidat LGBM terbaik**. Replay jendela lebih lebar (30 Jun-2 Jul) mengonfirmasi 37f punya
perilaku adaptif per-hari yang persis diminta user (short di downtrend, mundur bertahap saat
reversal, nol short di puncak pump).

### IC Test Round 3 — pool luas (23 kandidat, basis: 37f = 35f-clean+absorb+vwdp)

| Fitur | IC (ICIR) | Korelasi vs fitur terpakai | Keputusan |
|---|---|---|---|
| `VAL` | 0.059 (**ICIR 5.89**, terkuat sesi) | **-0.52 s/d -0.55** vs geometry family | **DITOLAK** — kolinearitas parah |
| `POC` | 0.062 (ICIR 5.50) | **-0.50 s/d -0.58** | **DITOLAK** |
| `VAH` | 0.063 (ICIR 4.51) | **-0.58 s/d -0.65** (terparah) | **DITOLAK** |
| `relative_strength_momentum` | -0.041 (ICIR -3.94) | 0.27-0.40 vs geometry+`relative_strength_z` | Ditolak (kolinearitas) |
| `rsi_divergence` | -0.036 (ICIR -3.45) | 0.22-0.39 | Ditolak |
| `PDL`/`PDH` | 0.032/0.026 | 0.35-0.50 vs geometry+`relative_strength_z` | Ditolak |
| `mae_8`/`mfe_8` | 0.021/-0.020 | 0.31-0.47 | Ditolak |
| `time_above/below_entry_8` | ±0.031 | 0.31-0.49 | Ditolak |
| **`hidden_divergence`** | 0.019 (ICIR 2.53) | **≤0.22** — bersih | Lolos screening → **diuji, GAGAL** (lihat bawah) |
| **`ofi_momentum_ratio`** | 0.010 (ICIR 2.21) | **≤0.10** — bersih | Lolos screening → **diuji, GAGAL** |

**Catatan penting (near-miss terbesar sesi ini)**: `VAH`/`VAL`/`POC` (volume profile levels)
punya IC/ICIR TERKUAT dari seluruh sesi, tapi korelasinya dgn `dist_from_8h_high` sampai
**-0.65** — persis pola "duplikat vote" yg motivasi seluruh proyek dekorelasi geometri.
Ditolak tanpa diuji OOF sama sekali — pelajaran dari eksperimen sebelumnya sudah cukup kuat
utk tidak membuang waktu training pada kandidat berkorelasi tinggi.

### Marginal Test (OOF, gate HMM 0.65/0.05, Guardian OFF) — hidden_divergence GAGAL

| Run | Trades | PF | PnL | MaxDD |
|---|---|---|---|---|
| **37f (baseline: 35f-clean+absorb+vwdp)** | 11,272 | **1.641** | **$3,410** | -35.3 |
| 38f (+`hidden_divergence`) | 10,824 | 1.614 (turun) | $3,153 (turun) | -45.6 (turun) |
| 39f (+`hidden_divergence`+`ofi_momentum_ratio`) | 10,455 | 1.611 (turun) | $3,121 (turun) | -40.8 |

### Replay Insiden — hidden_divergence MEMBALIKKAN perbaikan yg sudah didapat

| Model | Masih SHORT dari 14 (1-2 Jul) | SHORT di 2 Jul (puncak pump) |
|---|---|---|
| **37f (baseline)** | **2/14** | **0** (nol total!) |
| 38f (+hidden_divergence) | 5/14 (lebih buruk) | 2 (muncul lagi) |
| 39f (+hd+ofi_momentum_ratio) | 4/14 | 1 |

PnL proyeksi 19 trade riil 30 Jun-2 Jul: 37f **-$1.52** → 38f **-$7.73** (nyaris balik ke level
live -$8.57) → 39f -$3.51. `hidden_divergence` MENGHANCURKAN properti terbaik 37f (nol short
saat puncak pump) meski lolos IC screening DAN cek kolinearitas — kandidat ke-3 yg gagal
validasi OOF/insiden meski tampak bersih di screening awal (setelah `trend_strength`, `no_demand`).

### Replay Jendela Lebar 30 Jun-2 Jul (konfirmasi properti adaptif 37f)

| Hari | Live SHORT | **37f (absorb+vwdp) SHORT** | Konteks |
|---|---|---|---|
| 30 Jun (downtrend valid, profit) | 15 | **6** (proporsional lebih selektif) | BTC low jam 12:00 UTC |
| 1 Jul (reversal mulai) | 7 | **3** | Mulai pump |
| 2 Jul (puncak pump) | 7 | **0** | BTC 61k+ |

Trade riil 30 Jun-2 Jul (n=19): Live PnL -$8.57 → **37f PnL -$1.52** (perbaikan $7.05, terbaik
dari SEMUA kandidat yg pernah diuji sesi ini termasuk 35f-clean sendiri yg cuma -$3.59).

### Kesimpulan
- **37f (35f-clean+`absorption_at_swing`+`vwdp`) tetap kandidat LGBM final terbaik** — bukan
  cuma unggul di scorecard, tapi terbukti PALING ADAPTIF: mengurangi SHORT secara gradual
  seiring regime berubah (15→6 di downtrend valid, →3 saat reversal mulai, →0 di puncak pump),
  bukan all-or-nothing.
- 3 dari 3 percobaan "tambah fitur lagi di atas 37f" gagal (`trend_strength`, `no_demand`,
  `hidden_divergence`, `ofi_momentum_ratio`) — sinyal kuat bahwa 37f sudah mendekati titik
  optimal utk feature set berbasis fitur yg tersedia saat ini.
- `VAH`/`VAL`/`POC` layak dicatat sbg pelajaran: IC/ICIR terkuat TIDAK berarti kandidat
  terbaik jika kolinear dgn fitur retained — screening kolinearitas HARUS dilakukan sebelum
  training, bukan sesudah.

### Script & Artefak
- `models/runs/marg_37f_plus_hd/`, `marg_37f_plus_hd_ofimom/`
- Scorecard: `models/runs/marg_37f_plus_hd_ofimom/scorecard_compare.json`

---

## 2026-07-03 — Opsi 2: Redesain Label Struktural (gate momentum ofi_z_score, ganti vol_spike_zscore)

**Status**: DONE (OOF + replay insiden) — hipotesis inti TERVALIDASI di level probabilitas,
efek praktis positif tapi campur (MIXED) di metrik SHORT-saat-pump spesifik. Belum
menggantikan 37f label lama sbg kandidat final — perlu threshold sweep dulu.

### Akar Masalah (diagnosis sebelum eksperimen)
`swing_based_labeling()` normal path: `long_valid` mensyaratkan `tp_dist_long =
swing_high - harga > 0`. Begitu harga **sudah melewati** swing high lama (breakout
genuine), `tp_dist_long` negatif → LONG **mustahil** dilabeli via jalur normal, berapa
pun kuat momentumnya. Satu-satunya escape hatch: **jalur momentum** (TP/SL proyeksi ATR
dari harga sekarang, arah-netral), diaktifkan gate `momentum_vol_spike` >= 1.5. Gate asli
memakai `vol_spike_zscore` (lonjakan volume mentah) — masalahnya lonjakan volume biasanya
cuma di awal breakout lalu mereda, PADAHAL harga masih lanjut naik → gate mati di saat
paling dibutuhkan.

### Eksperimen: ganti gate ke `ofi_z_score` (order-flow imbalance, sudah confirmed bersih)
Direplikasi PERSIS logika `swing_based_labeling` di script terpisah (TIDAK mengubah
`core/features.py` produksi), cuma ganti sumber gate momentum + tiebreak
(`MSB_BOS` ganti `price_accel_1h`, sama-sama leaky di path lama).

**Validasi diagnostik (6 koin)**: saat kondisi "fresh bullish BOS + `ofi_z_score`>1.5"
(breakout genuine terkonfirmasi order-flow) — distribusi label:
`LONG 12.6%→44.3%, SHORT 13.2%→38.1%, FLAT ~74%→17.5%`. Distribusi label keseluruhan
(semua bar) cuma bergeser tipis (LONG 13.1%→13.6%, SHORT 14.1%→14.6%) — perbaikan presisi
di kondisi yang ditarget, bukan mengacak keseimbangan umum.

**Relabel penuh 21 koin** (861.289 baris, ~20 detik) dgn pola pergeseran serupa di semua
koin (LONG +0.5-0.7pp, SHORT +0.5-0.8pp per koin).

### Training OOF LGBM 37f di atas label baru (Opsi 2)

| Run | Trades | PF | PnL | Long count | Short count | Long PF | MaxDD |
|---|---|---|---|---|---|---|---|
| 37f label LAMA (baseline) | 11,272 | 1.641 | $3,410 | 5,200 | 6,072 | 1.616 | -35.3 |
| **37f label BARU (Opsi 2)** | 11,081 | **1.661** | $3,413 | **4,561** (turun) | **6,520** (naik) | 1.638 | **-25.4** (membaik jauh) |

**Kejutan**: LONG count OOF penuh justru TURUN (bukan naik seperti hipotesis awal), SHORT
naik. PF & MaxDD tetap membaik. Interpretasi: gate baru menambah label continuation di
KEDUA arah (LONG dan SHORT) secara simetris — dan crypto historis lebih sering crash tajam
drpd pump tajam, jadi net penambahan label condong ke SHORT di skala 5+ tahun data.

### Replay Insiden 30 Jun-2 Jul (per-state gate benar) — hasil CAMPUR, bukan clean win

| Hari | Live SHORT | 37f label lama SHORT | **37f label baru SHORT** |
|---|---|---|---|
| 30 Jun (downtrend valid) | 15 | 6 | **13** (mendekati live, lebih baik menangkap downtrend valid) |
| 1 Jul (reversal mulai) | 7 | 3 | **2** (lebih baik) |
| 2 Jul (puncak pump) | 10 | 1 | **4** (LEBIH BURUK utk metrik ini spesifik) |

PnL proyeksi 19 trade riil 30 Jun-2 Jul: label lama **-$1.52** → **label baru -$0.37**
(lebih baik secara agregat, meski hari 2 Jul individual lebih banyak short).

**Temuan kunci — hipotesis probabilitas TERVALIDASI**: `p_long` untuk banyak koin saat
pump 1-2 Jul melonjak drastis dari ~0.001 ke 0.5-0.6 di bawah label baru (1000SHIBUSDT
0.001→0.624, ADAUSDT 0.000→0.565, dst.) — **persis** memperbaiki akar masalah "p_long
collapse begitu harga breakout" yang diidentifikasi sebelumnya. TAPI nilai 0.5-0.6 ini
**masih di bawah threshold gate per-state** (0.675-0.70 utk state 0/1 yg dominan saat itu)
— jadi belum cukup untuk benar-benar fire LONG. Threshold yg sama (dikalibrasi utk label
LAMA) kembali jadi penghalang, persis pola yang berulang kali ditemukan sesi ini
(kalibrasi geometri, kalibrasi absorb+vwdp, sekarang kalibrasi label baru).

### Kesimpulan
- **Akar masalah "p_long collapse saat breakout" berhasil diperbaiki di level model** —
  tervalidasi lewat lonjakan p_long yg jelas saat pump.
- **Belum diterjemahkan jadi hasil trading yg bersih lebih baik** khusus utk metrik
  short-saat-puncak-pump — karena threshold gate per-state FIXED (0.65/0.05) belum
  dikalibrasi ulang utk PROBABILITAS BARU ini (sama seperti masalah kalibrasi yg sudah
  3x ditemukan sesi ini: dekorelasi geometri, tambah fitur, sekarang label baru).
- Metrik agregat (PF, MaxDD, PnL 19-trade) tetap membaik meski tidak dramatis.
- **Threshold sweep bukan lagi "nice to have" — sudah jadi kebutuhan mendesak.** Tiga
  perbaikan independen (geometri, fitur, label) semuanya terhambat oleh gate yang sama.
- Belum menggantikan 37f label lama sbg kandidat final tanpa threshold sweep lebih dulu.

### Script & Artefak
- `models/runs/opt2_relabel_ofiz_37f/`, `data/training/labeled_opt2/` (relabel, bukan produksi)
- Scorecard: `models/runs/opt2_relabel_ofiz_37f/scorecard_compare.json`

---

## 2026-07-03 — Retest Fitur Gagal vs Label Baru: `trend_strength` Berhasil, Insiden Sempit Tetap Campur

**Status**: DONE (IC + OOF + replay 2 jendela) — `trend_strength` terbukti genuinely membantu
saat dilatih ULANG dgn label baru Opsi 2 (sebelumnya gagal di label lama). Perbaikan solid
di skala penuh, TIDAK konsisten terlihat di insiden sempit 19-trade 30 Jun-2 Jul.

### Hipotesis
Fitur yg gagal sebelumnya (`trend_strength`, `hidden_divergence`, dll.) mungkin gagal karena
"bentrok" dgn label lama yg mean-reversion-biased, bukan karena fitur itu sendiri buruk.
Diuji ulang terhadap label baru (Opsi 2, gate `ofi_z_score`).

### IC Retest — 2 pola berbeda ditemukan

**IC agregat (semua 861k baris) vs label baru — nyaris tidak berubah** dari label lama utk
semua kandidat (delta -0.002 s/d +0.002) — masuk akal karena cuma 8.08% bar yg labelnya
benar-benar berubah (69,628 dari 861,289).

**IC KHUSUS di 69,628 bar yg berubah label — pola berbeda muncul**:

| Fitur | IC agregat (lama→baru) | **IC khusus bar berubah** | Arah |
|---|---|---|---|
| **`trend_strength`** | -0.0064 → -0.0084 (lemah) | **+0.0215** | Berbalik jadi searah — **lolos** |
| **`CHoCH`** | +0.0010 → +0.0034 (lemah) | **+0.0115** | Membaik, searah — lolos |
| `hidden_divergence` | +0.0193 (positif) | **-0.0193** | **Terbalik arah** — jelaskan kenapa dulu merusak |
| `ofi_momentum_ratio` | +0.0103 (positif) | **-0.0122** | Terbalik arah juga |
| `no_demand`, `effort_vs_result` | lemah | tetap lemah | Tidak lolos |

### Marginal Test OOF (basis: opt2_relabel_ofiz_37f, label baru)

| Run | Trades | PF | PnL | Long count | Long PF | MaxDD |
|---|---|---|---|---|---|---|
| opt2 baseline (37f, label baru) | 11,081 | 1.661 | $3,413 | 4,561 | 1.638 | -25.4 |
| **opt2 + `trend_strength`** | 12,070 | **1.692** | **$3,853** (+$440) | **5,042** (+10.5%) | **1.705** | -24.6 |
| opt2 + trend_strength + CHoCH | 12,408 | 1.666 | **$3,890** (tertinggi) | **5,202** (tertinggi) | 1.700 | -30.2 (turun) |

**Pertama kalinya jumlah LONG naik (bukan turun) bersamaan dgn PF & Long PF naik** — beda
total dari semua percobaan sebelumnya (geometri, absorb+vwdp, hidden_divergence di label
lama) yg selalu trade-off volume vs kualitas.

### Replay Insiden 30 Jun-2 Jul (per-state gate benar) — HASIL CAMPUR, tidak sesuai ekspektasi

| | opt2 baseline | **opt2 + trend_strength** |
|---|---|---|
| PnL proyeksi 19 trade riil | **-$0.37 (terbaik)** | -$4.08 (lebih buruk) |
| LONG di 1 Jul | 3 | **0 (hilang semua)** |
| Kandidat LONG baru selama pump 1-2 Jul | — | cuma 1 (ARBUSDT, p_long 0.648, masih di bawah gate) |

### Kesimpulan
- **Hipotesis tervalidasi secara statistik skala penuh**: `trend_strength` (dan `CHoCH`)
  memang gagal karena bentrok dgn label lama, bukan krn fitur itu sendiri lemah — setelah
  label diganti, kontribusinya positif & robust (ribuan trade OOF, 5+ tahun data).
- **TAPI perbaikan ini tidak terlihat di insiden 19-trade 30 Jun-2 Jul** — kemungkinan
  besar karena sampel terlalu kecil utk merefleksikan perbaikan yg berasal dari pola
  breakout lain di periode/koin berbeda, bukan kontradiksi nyata. Pola serupa (menang di
  agregat, beda di insiden sempit) sudah berulang beberapa kali sesi ini (v1 vs v2 geometri).
- Tidak ada satu kandidat yg menang di KEDUA metrik (OOF penuh vs insiden sempit)
  sekaligus — `opt2+trend_strength` terbaik di OOF, `opt2 baseline` terbaik di insiden.
- Threshold sweep (masih tertunda, diusulkan berkali-kali) makin krusial — perlu utk
  menilai kandidat-kandidat ini secara adil di volume yg sepadan sebelum keputusan final.

### Script & Artefak
- `models/runs/opt2_plus_trend/`, `opt2_plus_trend_choch/`
- Scorecard: `models/runs/opt2_plus_trend_choch/scorecard_compare.json`

---

## 2026-07-03 — Threshold Sweep HMM + Uji HMM Fast-React (vol/mom window 6/12)

**Status**: DONE — kandidat final terpilih **opt2_plus_trend** (35f-clean+absorb+vwdp+
trend_strength, label Opsi 2). Threshold sweep menemukan titik operasi yg genuinely
lebih baik dari live di volume sepadan. HMM fast-react diuji sbg fix akar-masalah
regime-lag — hasil kecil & campur, BUKAN solusi definitif.

### Threshold Sweep (24 kombinasi base/delta, HMM per-state gating, model opt2_plus_trend)

**Apples-to-apples vs live produksi (16,202 trades, PF 1.580, PnL $4,588)**:

| base/delta | Trades | PF | PnL | vs Live |
|---|---|---|---|---|
| 0.65/0.05 (default) | 12,070 | 1.692 | $3,853 | Volume -25% |
| **0.65/0.15** | 18,081 | 1.580 (sama) | **$5,109** | **+11.4% di volume mirip** |
| 0.60/0.05 | 20,047 | 1.558 | **$5,400** | **+17.7% di volume lebih besar** |

**Ini pertama kali membuktikan** perbaikan fitur+label sesi ini genuinely mengungguli
live BUKAN cuma "kualitas naik tapi volume dikorbankan" — di volume yg sepadan/lebih
besar, PnL tetap lebih tinggi.

### KOREKSI PENTING — threshold "terbaik agregat" ternyata LEBIH BURUK di insiden spesifik

Rekomendasi awal (0.65/0.15) diuji ulang khusus di insiden 30 Jun-3 Jul (19 trade riil,
live PnL -$8.57):

| Threshold | PnL insiden | WR |
|---|---|---|
| 0.65/0.05 (default) | **-$4.08** (terbaik di antara delta>0) | 40% |
| 0.65/0.15 (rekomendasi awal, SALAH) | **-$9.67** (lebih buruk dari live!) | 33.3% |
| 0.60/0.05 | -$7.20 | 37.5% |

Sebab: delta besar melonggarkan threshold SHORT di state RANGING_LOW_VOL (state yg
dominan selama regime-lag), meloloskan lebih banyak short beracun. **Delta=0 (flat,
regime diabaikan sepenuhnya) sempat terlihat "menang"** (satu2nya PnL insiden positif,
+$0.36) TAPI user dgn tepat mengoreksi: itu membuang fungsi HMM sepenuhnya, bukan solusi
— HMM ada untuk memberi variasi threshold per-regime, delta=0 meniadakan tujuan itu.

**Kesimpulan sweep**: di antara opsi yg genuinely mempertahankan fungsi HMM (delta>0),
default 0.65/0.05 sudah jadi titik seimbang terbaik yg ditemukan — menaikkan delta
memperbaiki OOF agregat tapi memperburuk ketahanan saat regime-transition-lag.

### Uji Akar Masalah: HMM Fast-React (vol_window=6, mom_window=12 vs default 24/48)

Hipotesis: bukan threshold yg salah, tapi HMM-nya sendiri LAMBAT mendeteksi regime
berubah (window 24/48 bar = berhari-hari). Fit ulang HMM 13 koin dgn window lebih
pendek, fitting pakai data lokal `data/training/klines/` (< cutoff) + fetch API utk
data terbaru, predict via SSOT method (shift(1), predict >= cutoff).

**Hasil per koin (state 30 Jun-3 Jul) — CAMPUR, bukan universal**:

| Kategori | Koin | n |
|---|---|---|
| Membaik jelas (capai TRENDING_UP) | ETHUSDT, ADAUSDT | 2 |
| Membaik sebagian (menjauh dari TRENDING_DOWN) | BTCUSDT, BNBUSDT, HBARUSDT, XRPUSDT | 4 |
| Tidak berubah | ONDOUSDT | 1 |
| Memburuk (menuju TRENDING_DOWN, makin salah) | AVAXUSDT, TAOUSDT | 2 |
| Berisik/tak jelas pola | SOLUSDT, SUIUSDT, DOTUSDT, NEARUSDT | 4 |

### Dampak Konkret ke Trading (gate opt2_plus_trend dgn state fast-react vs live)

| Gate | Trade diambil | PnL insiden |
|---|---|---|
| HMM live (default) | 5 | -$4.08 |
| **HMM fast-react** | 3 | **-$2.78** (membaik +$1.30) |

Perbaikan kecil: fast-react benar menghindari 1 loss (ADAUSDT -$1.66, state benar
terdeteksi TRENDING_UP) TAPI juga membuang 1 win (XRPUSDT +$0.36, state berubah dari
RANGING_LOW_VOL ke RANGING_HIGH_VOL). Net +$1.30 dari swap satu lawan satu, sampel
sangat tipis (2 dari 19 trade berubah keputusan).

### Kesimpulan
- **Kandidat final: opt2_plus_trend**, threshold **0.65/0.05 tetap dipertahankan**
  (titik seimbang terbaik yg ditemukan di antara opsi yg mempertahankan fungsi HMM).
- Threshold 0.65/0.15 atau lebih lebar TERBUKTI meningkatkan OOF agregat tapi
  memperburuk insiden regime-lag — trade-off yg harus disadari, bukan otomatis "lebih
  lebar = lebih baik".
- HMM fast-react: **hasil kecil & tidak konklusif** — bukan direkomendasikan sbg
  perubahan produksi tanpa validasi jauh lebih besar (retrain OOF regime penuh, bukan
  cuma satu insiden n=19). Tidak dilanjutkan ke full-scale karena sinyal awal lemah.
- Di volume sepadan dgn live (base/delta lebih lebar spt 0.65/0.15 atau 0.60/0.05),
  opt2_plus_trend TETAP mengungguli live 11-18% — ini tetap valid sbg validasi kandidat
  final, terlepas dari nuansa insiden-spesifik di atas.

### Script & Artefak
- `models/runs/opt2_plus_trend/hmm_threshold_sweep.csv`, `sweep_combined_oof_incident.csv`
- `e:/Widyawardhana_Capital/scratch_fastreact_hmm/` (encoding fast-react per koin, riset saja)

---

## 2026-07-03 — Skenario A: Monotone Constraints via One-vs-Rest (SHORT-vs-rest, LONG-vs-rest)

**Status**: DONE — constraint terverifikasi bekerja BENAR (partial dependence), OOF PnL naik
signifikan tapi PF turun (trade-off kuantitas vs kualitas). Belum menggantikan
`opt2_plus_trend` sbg kandidat final tanpa keputusan lebih lanjut.

### Masalah teknis yang diselesaikan dulu
Native LightGBM multiclass (`objective=multiclass`) menerapkan `monotone_constraints`
SAMA untuk semua kelas (SHORT/FLAT/LONG) — tidak bisa "paksa LONG naik, SHORT turun"
secara asimetris pada model multiclass tunggal. Solusi: **2 model one-vs-rest terpisah**
(SHORT-vs-rest, LONG-vs-rest), masing-masing binary classifier dgn constraint sendiri:
- Model SHORT: `monotone_constraints=-1` pada `btc_ret_24h`, `mkt_breadth_4h`
  (P(SHORT) tidak boleh naik saat market makin bullish)
- Model LONG: `monotone_constraints=+1` pada fitur yg sama
  (P(LONG) tidak boleh turun saat market makin bullish)

Basis: 38 fitur `opt2_plus_trend`, label Opsi 2, OOF walk-forward 8-fold sama.
AUC per-fold sehat: SHORT 0.84-0.87, LONG 0.85-0.88.

### Verifikasi Constraint — Partial Dependence (BUKAN korelasi mentah)

Korelasi Spearman mentah p_long vs btc_ret_24h di data insiden **masih negatif** (-0.131)
— ini SEMPAT terlihat sbg kegagalan constraint. Setelah ditelusuri: `monotone_constraints`
LightGBM adalah constraint **partial dependence** (menahan fitur lain tetap), BUKAN
korelasi marginal lintas-sampel. Korelasi mentah terkontaminasi variasi lintas-koin
(koin lain punya fitur lain yg sangat berbeda). Uji ulang dgn partial dependence yang
benar (3 sampel riil, cuma btc_ret_24h/mkt_breadth_4h divariasi, fitur lain TETAP):

**SUIUSDT** — btc_ret_24h -0.05→+0.05: p_long **0.6455→0.6705 (naik monoton)**,
p_short **0.6231→0.5985 (turun monoton)** ✅
**AVAXUSDT** — p_long 0.0029→0.0035 (naik tipis, tetap monoton), p_short 0.6050→0.5829 (turun) ✅
**NEARUSDT** — p_long 0.6681→0.7075 (naik), p_short 0.7739→0.7481 (turun) ✅

**Constraint terbukti bekerja PERSIS sesuai desain di ketiga sampel** — cuma pengaruhnya
modest (perubahan beberapa poin persen) dibanding 36 fitur lain, jadi tidak mendominasi
korelasi agregat, tapi tetap memberi lantai/plafon yang benar.

### Evaluasi Utama: OOF Penuh (gate HMM 0.65/0.05, Guardian OFF)

| Run | Trades | PF | PnL | Long count | Long PF | MaxDD |
|---|---|---|---|---|---|---|
| opt2_plus_trend (baseline) | 12,070 | **1.692** | $3,853 | 5,042 | **1.705** | **-24.6** |
| **opt2_trend_monotone_ovr** | **34,286** (+184%) | 1.369 (turun) | **$6,737** (+75%) | **18,130** (+260%) | 1.338 | -48.6 (turun) |

**Trade-off besar dan jelas**: volume trade + PnL absolut melonjak signifikan (terutama
partisipasi LONG naik 260%), tapi kualitas per-trade (PF) turun dan MaxDD memburuk.
PF tetap komfortabel di atas 1.0 (masih profitable secara struktural).

### Replay Insiden 30 Jun-3 Jul (tambahan, bukan penentu utama)

| Hari | Live | base | **ovr_monotone** |
|---|---|---|---|
| 2 Jul (puncak pump) | S=10,L=1 | S=5,L=1 | **S=49, L=80 (LONG jadi dominan!)** |

PnL 19 trade riil: base n=5 PnL=-$4.08 WR=40% → **ovr_monotone n=12 PnL=-$6.75 WR=50%**
(trade lebih banyak, WR lebih tinggi, tapi PnL dolar sedikit lebih buruk — asimetri
menang/kalah di sampel kecil ini).

### Kesimpulan
- **Hipotesis tervalidasi teknis penuh**: constraint OVR bekerja benar (partial dependence
  terverifikasi), dan secara OOF penuh berhasil membuka partisipasi LONG jauh lebih besar
  (+260%) — paling langsung menyerang akar masalah "LONG terhambat struktural" dari semua
  skenario yg diuji sesi ini.
- **Trade-off kuantitas vs kualitas nyata**: PF turun dari 1.692 ke 1.369, MaxDD memburuk.
  PnL absolut tetap naik signifikan karena volume, tapi ini bukan "menang tanpa biaya".
- Insiden spesifik: WR membaik (50% vs 40%) tapi PnL dolar sedikit lebih buruk di sampel
  kecil ini — konsisten dgn pola sesi ini bahwa OOF & insiden sempit bisa berbeda arah.
- Arsitektur OVR (2 model terpisah, bukan 1 multiclass) adalah perubahan STRUKTURAL, bukan
  cuma parameter — perlu dipertimbangkan matang sbg kandidat produksi (kompleksitas deploy
  2x lipat: 2 model, 2 file, kalibrasi probabilitas yg berbeda dari single-multiclass).

### Script & Artefak
- `models/runs/opt2_trend_monotone_ovr/` (lgbm_short.pkl, lgbm_long.pkl)
- Scorecard: `models/runs/opt2_trend_monotone_ovr/scorecard_compare.json`

### Verdict user (setelah presentasi)
**DITOLAK.** Kenaikan trade count (2.84x) tidak sebanding kenaikan PnL (1.75x) — PnL per
trade turun 38% ($0.319→$0.196), tercermin di PF (1.692→1.369). Trade lebih banyak dgn
kualitas diencerkan, bukan perbaikan genuine. Dugaan penyebab: arsitektur OVR (2 model
biner independen, tidak saling menekan spt softmax multiclass) mungkin berkontribusi
lebih besar drpd constraint itu sendiri -- belum diisolasi (butuh uji OVR tanpa
constraint utk pisahkan efek arsitektur vs constraint, belum dikerjakan).

---

## 2026-07-03 — Skenario B: Validasi Fast-React HMM SKALA PENUH (regenerasi OOF walk-forward)

**Status**: DONE — **DITOLAK**. Uji insiden kecil sebelumnya (n=19) MENYESATKAN; skala
penuh (12,070 trade, apples-to-apples) menunjukkan fast-react kalah di SEMUA metrik.

### Metodologi (beda dari uji cepat sebelumnya)
Uji sebelumnya (entri "Threshold Sweep HMM + Uji HMM Fast-React") cuma fit-sekali +
predict pada 1 jendela kecil (30 Jun-3 Jul, n=19 trade) -- BUKAN walk-forward OOF yg
proper. Kali ini: regenerasi PENUH via `generate_oof_regime_labels()` (methodology SAMA
persis dgn `pipeline/data/core/regime_hmm.py` yg menghasilkan `hmm_regime_enc` produksi)
utk semua 21 koin, walk-forward 8-fold, purge 6 bar H4, `n_iter=100` -- window diganti
vol=6/mom=12 (fast-react) vs default vol=24/mom=48. Selesai 63 detik (21 koin, cepat).
Beberapa fold (POLUSDT fold 1, ONDOUSDT fold 4) HMM gagal konvergen, otomatis fallback
(mekanisme bawaan `generate_oof_regime_labels`, sama seperti produksi).

### Hasil — Apples-to-apples SEMPURNA (trade count identik, cuma input regime beda)

| Metrik | Default HMM (24/48) | **Fast-React HMM (6/12)** |
|---|---|---|
| Trades | 12,070 | 12,070 (identik) |
| **PF** | **1.692** | 1.651 (turun) |
| PnL | $3,853 | $3,659 (turun 5%) |
| **Long count** | 5,042 | **4,595 (turun 9% -- BUKAN naik)** |
| Short count | 7,028 | 7,475 (naik) |
| MaxDD | -24.6 | -29.7 (memburuk) |

**Fast-react kalah di SETIAP metrik** -- bukan cuma tidak membantu, partisipasi LONG
malah turun (kontra hipotesis), semua metrik kualitas memburuk.

### Kesimpulan
- **Kontras tajam dgn uji insiden kecil sebelumnya** (n=19, hasil +$1.30 "membaik") --
  itu KEBETULAN baik di sampel kecil, terbukti MENYESATKAN begitu diuji skala penuh.
  Bukti konkret kenapa evaluasi utama harus OOF (pelajaran yg sudah ditekankan user).
- Window fast-react (6/12) kemungkinan besar terlalu sensitif thd fluktuasi jangka
  pendek -- menangkap NOISE (state bolak-balik, sudah terlihat dari trace per-koin
  sebelumnya: SOL/SUI/DOT/NEAR "berisik") lebih banyak drpd sinyal regime genuine.
- **HMM fast-react DITOLAK sbg perbaikan** -- default (24/48) tetap dipertahankan.
- Opsi B dan A (skenario sebelumnya) sama-sama ditolak setelah diuji rigorous di skala
  penuh -- `opt2_plus_trend` (baseline, HMM default, threshold 0.65/0.05 atau hasil
  sweep 0.65/0.15 dsb.) tetap kandidat final terbaik yg tervalidasi sejauh ini.

### Script & Artefak
- `data/training/labeled_fastreact_hmm/` (regenerasi regime, riset saja, bukan produksi)
- `models/runs/opt2_plus_trend/scorecard_fastreact_full_oof.json`

---

## 2026-07-03 — Skenario C: Perbesar Radius Gate Momentum (ofi_z_score 1.5 -> 1.2)

**Status**: DONE — **DITOLAK** (verdict user). Pola sama dgn Skenario A yg ditolak
(trade naik lebih cepat drpd PnL, PF turun), lebih ringan tapi MaxDD memburuk signifikan
DAN tujuan utama (tambah partisipasi LONG) tidak tercapai.

### Perubahan
Threshold gate momentum path (`ofi_z_score` di `swing_based_labeling` reimplementation)
diturunkan 1.5->1.2, memperbesar coverage bar yg masuk jalur momentum dari ~11% ke
~16.4% (sample 8 koin). Relabel penuh 21 koin (22.7s). Distribusi label bergeser lebih
besar dari Opsi 2 asli (LONG/SHORT +2-3pp per koin, vs +0.5-0.7pp sebelumnya). Fitur
SAMA PERSIS dgn `opt2_plus_trend` (38f termasuk trend_strength) -- isolasi variabel
cuma soal radius label.

### Hasil OOF (gate HMM 0.65/0.05, Guardian OFF)

| Metrik | opt2_plus_trend (radius 1.5, baseline) | **opt2c (radius 1.2)** |
|---|---|---|
| Trades | 12,070 | 13,279 (+10%) |
| **PF** | **1.692** | 1.646 (turun) |
| PnL | $3,853 | $4,056 (+5.3%) |
| **Long count** | 5,042 | **5,261 (+4.3% saja)** |
| Short count | 7,028 | 8,018 (+14.1%) |
| **MaxDD** | -24.6 | **-40.3 (memburuk 64%)** |

**PnL per trade**: $0.3193 -> $0.3055 (**-4.3%**) -- pola sama dgn Skenario A (trade naik
lebih cepat drpd PnL), jauh lebih ringan (-4.3% vs -38%) tapi arah sama.

**Tujuan utama gagal tercapai**: partisipasi LONG cuma naik 4.3%, SEDANGKAN short count
naik 14.1% -- radius gate yg diperbesar mayoritas menangkap trade SHORT baru, bukan LONG
spt yg diharapkan.

### Kesimpulan
- **DITOLAK** dgn kriteria yg sama persis dipakai user utk menolak Skenario A (PF turun,
  trade/PnL tidak proporsional) -- kali ini lebih ringan tapi arah identik, plus MaxDD
  memburuk signifikan dan tujuan (tangkap lebih banyak LONG) tidak tercapai.
- **Ketiga skenario (A: monotone OVR, B: fast-react HMM, C: radius gate lebih besar)
  semuanya diuji rigorous di skala OOF penuh dan DITOLAK.**
- `opt2_plus_trend` (38f, label Opsi 2 radius 1.5, HMM default 24/48) tetap kandidat
  final terbaik yg tervalidasi sesi ini -- tidak ada penantang yg mengalahkannya tanpa
  trade-off signifikan.

### Script & Artefak
- `data/training/labeled_opt2c/`, `models/runs/opt2c_relabel_ofiz12_trend/`
- Scorecard: `models/runs/opt2c_relabel_ofiz12_trend/scorecard_compare.json`

---

## 2026-07-03 — Guardian Baru utk opt2_plus_trend + HMM (no cross-model features)

**Status**: DONE — **full-stack OOF & pseudo-holdout membaik signifikan, proporsional
(bukan sekadar naik trade)**. Belum dibandingkan ke produksi live / belum di-deploy.

### Hipotesis
`opt2_plus_trend` (entry LGBM + HMM gate) tervalidasi sbg kandidat final entry model.
Selama ini dievaluasi dengan exit fixed (TP/SL swing + max-hold), belum ada exit model
(Guardian) yang match dengan entry baru ini. Guardian lama (`ic32_rv2_guard_momentum_v2`
dkk.) dilatih dari sample trade entry LAMA (k5_mom_tr1 proxy) — tidak matched. Perlu
Guardian baru yang dilatih langsung dari trade riil `opt2_plus_trend`+HMM, dengan
fitur bersih sesuai prinsip no-cross-model-feature ([[no-cross-model-features]]).

### Yang Diubah
- Sample generation: `simulate_trades_swing(guardian_enabled=False)` pada trade riil
  hasil gating PER_STATE (opt2_plus_trend proba + HMM 0.65/0.05), bukan proxy k5_mom.
  Label: reuse `_label_escort_no_lstm_pnl_constrained` (momentum escort v2,
  pnl_constrained) dari `train_guardian_k5mom_v7.py`.
- Fitur: 21 static v7 **minus** `hmm_regime_enc` (20) + 9 dynamic v7 **minus**
  `lgbm_entry_conf` (8) = **28 fitur, 0 cross-model output**. Di-assert eksplisit
  (`hmm_regime_enc`/`lgbm_entry_conf`/`p_bull` tidak boleh ada di feat_cols).
- Training: `train_guardian_with_oof()` (8-fold walk-forward, sama seperti entry LGBM).

### Hasil Training (408,021 sample: HOLD 350,236 / PARTIAL 482 / EXIT 57,303)
CV logloss=0.321, F1-macro=0.606, stabil di 8 fold (logloss 0.297-0.338).

### Hasil Full-Stack Simulasi (apples-to-apples, trade source identik)

**Full OOF (2020 → 2026-04-01):**

| Metrik | Tanpa Guardian | **Dengan Guardian baru** | Delta |
|---|---|---|---|
| Trades | 12,070 | 12,354 | +2.35% |
| WR | 53.45% | **64.20%** | +10.75pp |
| PF | 1.692 | **2.040** | +20.6% |
| PnL | $3,853.40 | **$5,332.48** | +38.4% |
| PnL/trade | $0.3193 | **$0.4316** | +35.2% |
| Max DD | -24.6% | **-19.77%** | membaik 19.6% |
| Long PF | 1.705 | 2.181 | +27.9% |
| Short PF | 1.680 | 1.915 | +14.0% |

**Pseudo Holdout (2025-10-01 → 2026-04-01):**

| Metrik | Tanpa Guardian | **Dengan Guardian baru** | Delta |
|---|---|---|---|
| Trades | 1,275 | 1,313 | +2.98% |
| WR | 51.22% | **62.38%** | +11.16pp |
| PF | 1.466 | **1.943** | +32.5% |
| PnL | $248.22 | **$478.30** | +92.7% |
| Max DD | -19.71% | **-9.36%** | membaik 52.5% |
| Long PF | 1.119 (lemah) | **1.544** | +38.0% |
| Short PF | 2.088 | 2.682 | +28.4% |

### Kesimpulan
- **Kenaikan trade count kecil dan proporsional** (+2.35% OOF, +2.98% pseudo holdout) —
  BUKAN pola disproporsional yang menyebabkan Skenario A/C ditolak. PnL/trade justru
  naik +35% (OOF) — perbaikan genuine, bukan dilusi kuantitas.
- Guardian memperbaiki **kedua sisi** (LONG dan SHORT PF naik di kedua evaluasi) dan
  MaxDD membaik signifikan di kedua evaluasi (-19.6% OOF, -52.5% pseudo holdout).
- Sisi LONG di pseudo holdout — titik lemah yang diidentifikasi sebelumnya (PF 1.119,
  lihat laporan konsolidasi 2026-07-03) — membaik ke PF 1.544 dengan Guardian, meski
  masih di bawah SHORT PF (2.682). Belum menutup celah sepenuhnya tapi arah positif.
- **Belum dibandingkan terhadap kasus insiden 30 Jun-2 Jul 2026** maupun terhadap
  Guardian produksi saat ini — perlu keputusan eksplisit user sebelum promosi/deploy.

### Script & Artefak
- Training: `train_guardian_opt2_plus_trend.py` (scratchpad sesi ini)
- Simulasi: `sim_full_stack_guardian.py` (scratchpad sesi ini)
- Model: `models/runs/guard_opt2_plus_trend_hmm/{guardian.pkl, guardian_scaler.pkl, guardian_features.json, guardian_cv_results.json}`
- Hasil: `models/runs/guard_opt2_plus_trend_hmm/full_stack_vs_no_guardian.json`

---

## 2026-07-03 — DEPLOY ic32_regime_v6 / fs38_28f ke Production

**Status**: DONE — live di VPS sejak 2026-07-03 07:49 UTC (15:49 WITA).

### Stack (alias baru)
- **lgbm38f** = `opt2_plus_trend` (38f, label Opsi 2 + trend_strength)
- **hmm24/48** = HMM default per-coin, gate 0.65/0.05 (tidak berubah dari v5)
- **guard28f** = `guard_opt2_plus_trend_hmm` (28f, NO cross-model)
- LSTM tetap OFF. Prinsip no-cross-model kini berlaku penuh: v5 masih punya
  `hmm_regime_enc` di LGBM & `p_bull`+`lgbm_entry_conf` di Guardian — v6 bersih semua.

### Audit pra-deploy (sinkronisasi fitur)
- 38 fitur lgbm38f + 20 static guard28f: semua hadir & non-null di snapshot live
  (40/40; kecuali `VAH` null 2/40 — pre-existing, live isi 0.0).
- 5 fitur baru vs v5 (`absorption_at_swing`, `atr_percentile_h1`, `ofi_z_score`,
  `trend_strength`, `vwdp`): implementasi identik riset<->swint (git-sync).
- `guardian_service.py` live feature-list-driven -> guard28f kompatibel tanpa ubah kode.
- Param guardian disesuaikan ke nilai tervalidasi sim: `min_hold_bars` 2->4,
  `momentum_floor_frac` 0.2->0.7 (exit 0.65, activation 0, partial 0.5 tetap).
- Skew `bars_held_norm` train(/36) vs infer(/24): pre-existing sejak v5, sim validasi
  memakai /24 yg sama dgn live -> hasil sim = perilaku live. Fix opsional harus
  serentak train+infer (kedua sisi), bukan saat deploy.

### Verifikasi pasca-deploy
- VPS: git pull + restart OK, `/api/health` merespons, registry swint auto-update v6 active.
- HMM parity (`verify_hmm_feature_parity.py`, pointer sudah diarahkan ke
  `opt2_plus_trend/features.json`): **PASS 0 mismatch** (BTC/ETH/DOGE sejak 2026-06-24).
- Feature check menandai 4 fitur panel market (`btc_ret_24h`, `btc_minus_mkt_24h`,
  `mkt_breadth_1h/4h`) ALL_NAN — **di file holdout riset** (0/2200 sejak April), BUKAN
  di live (live menghitung sendiri, tervalidasi "sync 5f parity" saat v5 + audit snapshot
  hari ini). Follow-up: backfill panel market ke holdout labeled agar harness parity
  bisa memverifikasi 4 fitur ini juga.
- `compare_holdout_live.py` belum bisa jalan (KeyError 'match') — belum ada trade v6
  utk dibandingkan; ulangi setelah ada trade live v6.

### Konfigurasi kunci
- `_snapshot_time` = `2026-07-03 07:49:19` UTC -> scorecard live v6 terpisah dari v5.
- Rollback: v5 = `ic32_rv2_lgbm_mkt_sync_v2` + `ic32_rv2_guard_momentum_v2`
  (file masih ada di swint/models + backup timestamped otomatis deploy).

---

## 2026-07-03 — Sweep guard28f `guardian_min_hold_bars`: 4 vs 0

**Status**: DONE — **tidak ada efek berarti, param 4 (live) dipertahankan.**

### Hipotesis
`min_hold_bars=4` menahan Guardian dari cek exit di 4 bar pertama trade. Diuji apakah
menurunkan ke 0 (exit bisa lebih dini) memberi perbaikan PnL/DD.

### Hasil (apples-to-apples, guard28f + fs38_28f)

| Metrik | min_hold=4 (prod) | min_hold=0 | Delta |
|---|---|---|---|
| Full OOF trades | 12,354 | 12,355 | +1 |
| Full OOF PF | 2.040 | 2.044 | +0.004 |
| Full OOF PnL | $5,332.48 | $5,345.21 | +0.24% |
| Full OOF MaxDD | -19.77% | -19.77% | sama |
| Pseudo holdout (semua metrik) | identik | identik | 0 |

Distribusi exit reason (`GUARDIAN_MOMENTUM_FLOOR`, `LOSS`, `GUARDIAN_EXIT`, dll) juga
hampir sama persis — beda 1-4 trade dari ribuan.

### Kesimpulan
- **`min_hold_bars` BUKAN lever berpengaruh di stack ini** — Guardian secara alami
  jarang mau EXIT/PARTIAL di 4 bar pertama walau diizinkan lebih awal (kemungkinan
  `bars_held_norm` sbg fitur membuat proba EXIT masih rendah di awal trade).
- **Tidak ada perubahan config** — `min_hold_bars=4` (live, `inference_config.json`)
  dipertahankan. Tidak ada upside maupun downside dari 0.

### Script & Artefak
- `sweep_guardian_minhold0.py` (scratchpad sesi ini)
- Hasil: `models/runs/guard_opt2_plus_trend_hmm/sweep_min_hold_bars.json`

---

## 2026-07-03 — LSTM Confirmation Cascade (v1 win/loss, v2 continuation magnitude)

**Status**: DONE — **KEDUA VARIAN DITOLAK** (AUC ~0.54, hampir acak). Tidak lanjut ke evaluasi cascade.

### Konteks
Setelah `fs38_28f` (lgbm38f+hmm24/48+guard28f) dikunci sbg produksi (v6), dieksplorasi
arah model baru: LSTM sbg **cascade confirmation layer terpisah** (BUKAN fitur input ke
LGBM/Guardian — patuh [[no-cross-model-features]]). Desain: LSTM jalan HANYA saat
LGBM+HMM sudah signal LONG/SHORT, tugas cuma confirm/veto (binary genuine-move),
integrasi via mekanisme `hard_consensus` yang sudah ada di `core/cascade_utils.py`.

Riset riwayat (via subagent) menemukan project ini SUDAH BERKALI-KALI mencoba LSTM
sequence dari fitur market/order-flow — semua plateau di F1~0.33-0.49 ("OHLCV/order-flow
ceiling"), tidak pernah spesifik LONG-confirmation. User pilih tetap coba (skop simetris
LONG+SHORT), dgn urutan isolasi variabel: label dulu, baru fitur/arsitektur kalau perlu.

### Setup umum (v1 & v2)
- Populasi entry: trade riil dari `simulate_trades_swing(guardian_enabled=False)` pada
  opt2_plus_trend OOF proba + HMM PER_STATE gate (SAMA populasi dgn guard28f).
- Fitur sequence: 11 fitur order-flow/positioning (`ofi_z_score, ofi_acceleration,
  cvd_momentum_adv, absorption_z, volume_delta, vol_ratio_20, log_ret_1/5/20, rsi_6,
  btc_ret_1h`) — BUKAN OHLCV mentah (hindari ceiling yg sudah terbukti gagal),
  seq_len=72 H1 bar (referensi `ic32_lstm_regime_v2`).
- Arsitektur: `TradingLSTM` (`core/models.py`, sudah ada) hidden=32, layers=2, dropout=0.3,
  num_classes=2 (binary).
- Validasi: walk-forward 8-fold, fold berbasis waktu kalender (bukan posisi baris, krn
  sample sparse/event-based), purge_gap=36 jam (=MAX_HOLDING_BARS, cegah label leakage).
- n_samples=12,046 (match trade count opt2_plus_trend tanpa guardian).

### v1 — Label win/loss (net_pnl trade > 0)
Label dari hasil trade riil (net_pnl), bisa terkontaminasi mekanik SL-hunt/TP-fallback.

| Fold | n_train | AUC | F1-macro |
|---|---|---|---|
| 2-8 | 634 → 10,160 | 0.515-0.564 | 0.457-0.512 |

**Overall OOF: AUC=0.5418, F1-macro=0.4867** (SHORT AUC 0.547, LONG AUC 0.534).

### v2 — Label continuation magnitude (MFE >= 1.2 ATR dlm 36 bar, bebas mekanik trade)
Isolasi variabel murni di label (fitur & arsitektur v1=v2). Label lebih bersih (murni
price-path, dihitung langsung dari close/atr), tapi imbalanced (genuine 71%/29%) —
dipakai class-weighted CrossEntropyLoss. Korelasi label v2 vs v1 = 0.663 (terkait tapi beda).

| Fold | n_train | AUC | F1-macro |
|---|---|---|---|
| 2-8 | 634 → 10,160 | 0.516-0.566 | 0.491-0.538 |

**Overall OOF: AUC=0.5428, F1-macro=0.5225** (SHORT AUC 0.546, LONG AUC 0.537).

### Kesimpulan
- **AUC v1 vs v2 hampir identik (0.5418 vs 0.5428)** — F1-macro naik (0.487→0.523, efek
  class-weighting membuat prediksi lebih seimbang) tapi **AUC (metrik diskriminasi murni,
  independen threshold) TIDAK bergerak sama sekali**. Ini bukti kuat: **bottleneck BUKAN
  di label** (hipotesis awal v2 salah) — mengganti label sebersih apa pun tidak membantu.
- **Kedua varian gagal melewati ceiling ~0.54 AUC** (hampir acak, 0.5=coin flip) — konsisten
  & mengonfirmasi ulang pola historis project (LSTM momentum 3-class F1~0.33-0.41 Mei 2026,
  `ic32_lstm_regime_v2` LONG F1 0.28-0.37) meski target/label kali ini benar2 baru.
- **Tidak lanjut ke evaluasi cascade** — AUC 0.54 tidak punya sinyal cukup utk memberi efek
  nyata di PnL/trade, buang compute utk hasil yang sudah predictable dari AUC semata.

### v3 — Arsitektur diperbesar (hidden 32->128, layers 2->3, + attention pooling, RX6600/DirectML)
Fitur & label SAMA PERSIS dgn v1 (11f, win/loss) — isolasi murni di kapasitas model.
Reuse sample v1. Training via GPU RX6600 (`torch_directml`, device `privateuseone:0`,
tanpa error — infrastruktur `_CellLSTM` di `core/models.py` memang didesain DirectML-compatible).

| Fold | n_train | AUC | F1-macro |
|---|---|---|---|
| 2-8 | 634 → 10,160 | 0.499-0.544 | 0.423-0.503 |

**Overall OOF: AUC=0.5332, F1-macro=0.4833** (SHORT AUC 0.521, LONG AUC 0.550) —
**LEBIH LEMAH** dari arsitektur kecil v1 (0.5418), bukan lebih baik. Fold 5 malah AUC
0.499 (persis coin-flip). Kemungkinan overfit ringan krn kapasitas naik tapi data tetap
~12k sample.

### Kesimpulan akhir (v1+v2+v3)
- **Tiga variasi independen, tiga hasil identik-mentok**: label win/loss (0.5418), label
  continuation magnitude (0.5428), arsitektur besar+attention (0.5332 — malah turun).
  Tidak satu pun lolos ceiling ~0.53-0.54 AUC (hampir coin-flip).
- **Bottleneck bukan di label maupun kapasitas model** — sinyal genuine-vs-fake move
  memang tidak cukup ada di 11 fitur order-flow + sequence 72 bar H1 ini, utk populasi
  entry manapun (LGBM+HMM sudah signal). Konsisten & mengonfirmasi ulang pola historis
  project yg sudah berkali-kali gagal dgn feature set/arsitektur berbeda-beda.
- **LSTM confirmation cascade DITOLAK sepenuhnya** — opsi ke-4 (perluas fitur ke 38f)
  tidak dilanjutkan; probabilitas berhasil dinilai sangat rendah mengingat pola konsisten
  di 3 percobaan + riwayat panjang project. Arah eksplorasi berikutnya (keputusan user):
  algoritma entry beda, model sizing, atau ensemble regime-specialist LGBM.

### Script & Artefak
- `build_lstm_confirm_samples.py`, `train_lstm_confirm.py` (v1)
- `build_lstm_confirm_samples_v2.py`, `train_lstm_confirm_v2.py` (v2)
- `train_lstm_confirm_v3_big.py` (v3, reuse sample v1, DirectML/RX6600) — scratchpad sesi ini
- Model: `models/runs/lstm_confirm_v1/`, `lstm_confirm_v2/`, `lstm_confirm_v3_big/` (cv_results.json)

---

## 2026-07-03 — Rasa Penasaran: XGBoost & CatBoost vs LightGBM (algoritma entry)

**Status**: DONE — **DITOLAK**, LightGBM (locked) tetap terbaik. Eksperimen rasa
penasaran user, isolasi murni di algoritma (fitur 38f + label Opsi 2 IDENTIK ke
`opt2_plus_trend`, walk-forward 8-fold purge=20 sama persis).

### Setup
`train_catboost_xgboost_opt2.py`: reuse `pipeline.model.core.train_lgbm` untuk load data
(`LABEL_DIR` diarahkan ke `data/training/labeled_opt2`) + `build_rolling_folds` sama,
`LGBM_CLASS_WEIGHTS` (SHORT/LONG=3x, FLAT=1.5x) dipakai sbg sample_weight utk kedua
algoritma. Params disamakan sebisa mungkin: n_estimators/iterations=1500, lr=0.05,
depth/max_depth=6, subsample=0.8, colsample=0.8, early_stopping=50.

### Hasil training (F1-macro per fold) — hampir identik di training
LightGBM/XGBoost/CatBoost F1-macro per fold semuanya di rentang 0.555-0.587 — perbedaan
antar-algoritma nyaris tidak ada di level klasifikasi murni (sesuai dugaan awal).

### Hasil OOF Scorecard (HMM gate 0.65/0.05, metodologi identik)

| Model | Trades | WR | PF | PnL | PnL/trade | MaxDD |
|---|---|---|---|---|---|---|
| **lgbm38f (LOCKED)** | 12,070 | 53.45% | **1.692** | $3,853.40 | **$0.3193** | **-24.6** |
| xgb38f | 14,419 | 52.81% | 1.639 | $4,330.73 | $0.3003 | **-51.72** |
| catboost38f | 10,441 | 53.92% | 1.668 | $3,316.66 | $0.3177 | -45.98 |

### Kesimpulan
- **F1-macro training nyaris identik**, tapi begitu diterjemahkan ke trading, kedua
  alternatif KALAH — bukan cuma "tidak lebih baik", tapi **MaxDD membengkak ~2x lipat**
  di keduanya (XGB -51.72, CatBoost -45.98 vs LGBM -24.6), meski PF/PnL-per-trade
  berdekatan atau sedikit lebih rendah.
- XGBoost: pola "trade naik (+19.5%), PF & PnL/trade turun" — sama dgn pola disproporsional
  yg sudah berkali-kali ditolak sesi ini (Skenario A/C).
- Kemungkinan penyebab: kalibrasi probabilitas antar-library berbeda saat berinteraksi
  dgn threshold gate HMM (0.65/0.05) yang sensitif terhadap bentuk distribusi proba,
  bukan cuma akurasi klasifikasi rata-rata.
- **Konfirmasi hipotesis awal**: berganti algoritma GBM tidak menembus ceiling data yang
  sama — LightGBM bukan cuma setara, tapi justru unggul di kontrol risiko (MaxDD).
  `opt2_plus_trend` (lgbm38f) tetap kandidat entry terbaik yg tervalidasi.

### Script & Artefak
- `train_catboost_xgboost_opt2.py`, `score_catboost_xgboost.py` (scratchpad sesi ini)
- Model: `models/runs/xgb_opt2_plus_trend/`, `models/runs/catboost_opt2_plus_trend/`
- Scorecard: `models/runs/xgb_opt2_plus_trend/compare_algo_scorecard.json`

---

## 2026-07-03 — Ensemble Regime-Specialist LGBM (4 model per HMM state)

**Status**: DONE — **DITOLAK**. Partisipasi LONG naik signifikan sesuai tujuan, tapi
kualitas (LongPF) justru turun — pola sama dgn skenario2 yg sudah ditolak sebelumnya.

### Hipotesis
LGBM generalis harus berkompromi di 4 kondisi pasar sekaligus. Spesialis per-regime
(4 LGBM kecil, 1 per HMM state) diduga bisa mempelajari hubungan fitur-ke-label lebih
tajam per-kondisi — khususnya model TRENDING_UP diharapkan lebih baik menangkap breakout
LONG tanpa "diencerkan" pola dari 3 regime lain. Beda sifat dari LSTM (nambah sinyal
baru, gagal) -- ini cuma memodelkan ulang sinyal SAMA dgn lebih presisi per-kondisi.

### Setup
Fitur (38f) & label (Opsi 2) IDENTIK `opt2_plus_trend`. Data difilter per `hmm_regime_enc`
(0=TRENDING_DOWN n=169,810; 1=RANGING_LOW_VOL n=305,830; 2=RANGING_HIGH_VOL n=256,710;
3=TRENDING_UP n=128,939), masing2 walk-forward 8-fold purge=20 independen (data tetap
terurut kalender sblm filter, purge gap tetap valid). OOF 4 regime distitch jadi 1
array global utk evaluasi standar (HMM gate 0.65/0.05).

### Hasil training — F1 combined OOF 0.566, sebanding baseline (~0.5715 avg per-fold)

### Hasil OOF Scorecard

| Model | Trades | WR | PF | PnL | PnL/trade | MaxDD | LongPF | Long count |
|---|---|---|---|---|---|---|---|---|
| **lgbm38f (LOCKED)** | 12,070 | 53.45% | **1.692** | $3,853.40 | **$0.3193** | **-24.6** | **1.705** | 5,042 |
| regime_specialist | 15,211 | 51.98% | 1.565 | $4,152.36 | $0.2730 | -36.76 | 1.546 | **7,220** |

### Breakdown per-regime (audit lanjutan) — semua arah

| Regime | Model | Trades | WR | PF | PnL |
|---|---|---|---|---|---|
| TRENDING_DOWN | generalis | 2,581 | 53.70% | **1.647** | $853.69 |
| | spesialis | 3,152 | 51.11% | 1.449 | $769.88 |
| RANGING_LOW_VOL | generalis | 4,477 | 53.61% | **1.540** | $971.45 |
| | spesialis | 5,143 | 52.30% | 1.476 | $968.89 |
| RANGING_HIGH_VOL | generalis | 3,179 | 52.94% | **1.656** | $931.96 |
| | spesialis | 4,274 | 52.53% | 1.594 | $1,160.12 |
| TRENDING_UP | generalis | 1,833 | 53.63% | **2.062** | $1,096.30 |
| | spesialis | 2,642 | 51.48% | 1.763 | $1,253.47 |

### Breakdown per-regime, LONG saja

| Regime | Model | Trades | WR | PF | PnL |
|---|---|---|---|---|---|
| TRENDING_DOWN | generalis | 528 | 45.64% | **1.307** | $123.24 |
| | spesialis | 493 | 43.41% | 1.124 | $46.50 |
| RANGING_LOW_VOL | generalis | 1,108 | 50.99% | **1.347** | $196.33 |
| | spesialis | 1,306 | 48.01% | 1.222 | $133.96 |
| RANGING_HIGH_VOL | generalis | 1,917 | 54.83% | **1.734** | $629.81 |
| | spesialis | 3,075 | 52.42% | 1.560 | $798.95 |
| **TRENDING_UP** | generalis | 1,489 | 54.94% | **2.103** | $932.19 |
| | spesialis | 2,346 | 51.88% | 1.777 | $1,119.77 |

### Kesimpulan
- **Tujuan tercapai sebagian**: partisipasi LONG naik signifikan (+43.2%, 5,042→7,220)
  — persis yg diharapkan dari spesialisasi regime TRENDING_UP.
- **TAPI kualitas LONG justru turun** (LongPF 1.705→1.546, **-9.3%**), bukan naik. Trade
  count +26%, PF -7.5%, PnL/trade -14.5%, MaxDD memburuk ~50% (-36.76 vs -24.6).
- **Bukti definitif dari breakdown per-regime**: spesialis kalah PF di **SEMUA 4 regime
  tanpa kecuali** — termasuk di regime TRENDING_UP miliknya sendiri (LONG PF 2.103→1.777,
  turun juga), tempat seharusnya ia paling unggul. Hipotesis dasar (spesialisasi per-regime
  menangkap pola lebih tajam) **terbukti salah** — mempersempit data training per-regime
  cuma mengurangi jumlah sample belajar (~129k vs 861k total) tanpa manfaat spesialisasi
  yang diharapkan. Generalis unggul justru karena volume data lebih besar utk belajar pola
  lintas-regime.
- **Pola identik dgn Skenario A/C dan XGBoost** yang sudah ditolak sebelumnya: setiap kali
  sistem "dilonggarkan" utk menangkap lebih banyak LONG (radius gate, algoritma beda,
  spesialisasi regime), hasilnya konsisten kuantitas naik tapi kualitas turun proporsional
  atau lebih buruk.
- **Bukti tambahan yg memperkuat**: masalah short-bias/LONG-catching lemah kemungkinan
  butuh sinyal/informasi BARU yang belum ada di fitur saat ini (bukan restrukturisasi
  cara model existing dipakai) — LSTM (percobaan menambah sinyal baru) juga sudah gagal
  (ceiling AUC~0.54). Kombinasi bukti dari 5 percobaan independen sesi ini (monotone OVR,
  radius gate, fast-react HMM, LSTM confirmation, algoritma beda, regime-specialist)
  semuanya menunjuk ke arah yang sama: `opt2_plus_trend` + Guardian (fs38_28f) sudah
  dekat batas optimal yg bisa dicapai dgn data & fitur yg tersedia saat ini.
- **Status model**: `regime_specialist_lgbm` TIDAK dipakai produksi, disimpan sbg arsip
  riset saja di `models/runs/regime_specialist_lgbm/`.

### Script & Artefak
- `train_regime_specialist_lgbm.py` (scratchpad sesi ini)
- Model: `models/runs/regime_specialist_lgbm/` (4x `lgbm_regime{0-3}.pkl`, cv_results.json)
- Scorecard: `models/runs/regime_specialist_lgbm/compare_vs_generalist.json`, `per_regime_breakdown_meta.json`

---

## 2026-07-03 — Investigasi OOS Holdout Rugi (PF<1) — Root Cause + Fix `positioning_mode`

**Status**: DONE — **bug produksi ditemukan & diperbaiki**. OOS holdout membaik signifikan
tapi TETAP rugi tipis — regime shift (bukan bug) tetap dominan.

### Konteks
OOS holdout penuh (1 Apr – 2 Jul 2026, sealed, genuinely never-trained-on) menunjukkan
`fs38_28f` rugi (PF 0.829) — jauh di bawah OOF (PF 2.04). User mendesak investigasi
krn gap sebesar ini tidak masuk akal cuma dari "regime shift".

### Temuan 1 — Regime shift (nyata, terkonfirmasi, TAPI bukan penyebab tunggal)
Distribusi `hmm_regime_enc`: TRENDING_UP anjlok dari 14.97% (training 2020-Mar2026) ke
6.49% (holdout Apr-Jul2026), tren gradual sejak Okt 2025 (lihat entri holdout sebelumnya).
Live aktual (v4 PF 0.560, v5 PF 0.448, keduanya dari DB riil) mengonfirmasi ini bukan
cuma masalah backtest — pasar genuinely sulit utk SEMUA versi model periode ini.

### Temuan 2 — BUG: `positioning_mode="training_parity"` usang (root cause dominan kedua)
`long_short_ratio` (fitur **#6 terpenting**, 5,772 split dari 38 fitur) di live **ADA &
NON-NULL** tapi nyaris konstan (~1.0, std=0.0055) krn `_apply_training_parity_overrides()`
meng-clip ke `training_feature_standards.json` — file itu **snapshot model v1, 2026-06-19**,
dari saat training MEMANG masih synthetic. Training sebenarnya (`labeled`/`labeled_opt2`)
sudah lama pakai data ASLI Binance (mean 1.52-1.90, range 0.1-5.0, terverifikasi 5 koin).
Live TIDAK PERNAH di-update mengikuti perubahan ini — mismatch train/serve utk fitur top-6,
berlangsung sejak v4 (bukan spesifik v6).

**Uji sebab-akibat** (model sama, holdout sungguhan, entry-only):
| Skenario | Trades | PF | PnL |
|---|---|---|---|
| LSR ter-strip (kondisi lama) | 506 | 0.877 | -$33.12 |
| LSR asli (Binance) | 526 | 0.918 | -$22.20 |

Bug nyata (~33% dari kerugian) tapi BUKAN dominan — gap OOF-OOS mayoritas dari regime shift.

### Fix yang diterapkan
1. **`inference_config.json`**: `feature_engineering.positioning_mode` `training_parity`→`live`.
   Live sekarang pakai data asli dari `data/positioning/*.parquet` (fresh, real-time) alih2
   clip ke bounds usang. Deploy ke swint+VPS (sempat konflik git file model uncommitted lagi
   — pola sama isu deploy sebelumnya — diselesaikan via checkpoint commit + merge).
2. **`pipeline/data/core/engineer.py`**: `_INFERENCE_PARITY` utk `--holdout-test` diubah
   `True`→`False` (dulu strip data asli biar "match" live yg synthetic — asumsi usang).
   Holdout-test 21 koin diregenerasi + `join_sync_features_holdout.py` dijalankan ulang
   (utk restore `coin_mkt_sync_24h` dkk yg butuh market panel merge terpisah).

### Hasil OOS holdout SETELAH fix (full-stack, HMM gate 0.65/0.05 + guard28f)

| Metrik | Sebelum fix | Sesudah fix |
|---|---|---|
| Trades | 506 | 571 |
| WR | 47.63% | 50.44% |
| **PF** | 0.829 | **0.927** |
| PnL | -$43.06 | **-$18.11** |
| Max DD | -62.74 | **-41.47** |
| Long PF | 0.661 | 0.567 |
| Short PF | 0.998 | 1.154 |

### Kesimpulan
- Fix `positioning_mode` **nyata dan signifikan**: kerugian berkurang >2x lipat, MaxDD
  membaik ~34%. Ini bug produksi lama (sejak v4), bukan spesifik v6 — kemungkinan ikut
  berkontribusi pada kerugian live v4/v5 yg diamati juga.
- **Tetap net rugi (PF 0.927)** setelah fix — regime shift (TRENDING_UP langka) tetap
  faktor dominan yg tidak bisa "diperbaiki" spt bug biasa; itu kondisi pasar nyata.
- 3 bug produksi ditemukan sesi ini (lihat `FEATURE_AUDIT.md`): `long_short_ratio`
  (**FIXED**), TONUSDT ledakan EMA/trend (belum), 1000PEPEUSDT ledakan OFI/vwdp (belum).
- Prosedur audit baru (`tools/model/audit_feature_value_parity.py`) WAJIB dijalankan
  sebelum deploy ke depan — lihat `FEATURE_AUDIT.md`.

### Script & Artefak
- Fix: `models/inference_config.json`, `pipeline/data/core/engineer.py`
- Regenerasi: `pipeline/03_engineer.py --holdout-test --all`, `pipeline/experiments/join_sync_features_holdout.py`
- Scorecard: `models/runs/guard_opt2_plus_trend_hmm/oos_holdout_full_scorecard.json`, `oos_holdout_full_trades_detail.csv`
- Dokumentasi prosedur: `FEATURE_AUDIT.md`, `tools/model/audit_feature_value_parity.py`

---

## 2026-07-03 — UI Pemantauan Fitur Live (Feature Monitoring Dashboard)

**Status**: DONE — dideploy ke production, terverifikasi visual (Playwright) + terverifikasi
langsung menangkap bug TONUSDT yang masih terbuka.

### Konteks
User minta perbaikan menu fitur di live app supaya bisa pantau data per-signal per-koin,
agar insiden spt `long_short_ratio`/TONUSDT/1000PEPEUSDT tidak terulang tanpa terdeteksi.
Riset menemukan infrastruktur monitoring SUDAH ADA (`app/services/feature_monitor.py`,
`/features` page) dan sudah cukup matang (per-fitur check type: bounds/categorical/strict/
moderate/loose/skip, z-score based, Telegram alert) — masalahnya cuma acuannya
(`training_feature_standards.json`) snapshot model v1 (2026-06-19), usang total.

### 3 Tahap yang dikerjakan
1. **Regenerasi `training_feature_standards.json`** (`regen_training_feature_standards.py`,
   scratchpad) — 38 fitur v6 dari `labeled_opt2` (861,289 baris, 21 koin). check-type
   dipertahankan dari judgment file lama utk fitur overlap, fitur baru (btc_ret_24h dkk)
   diberi tipe berdasar range aktual.
2. **Modal "Semua Fitur" per-signal** di `/paper/signals` — tombol kecil "fitur" per baris
   membuka modal AJAX (`GET /paper/signals/<id>`, diperluas return `parity` dari
   `check_feature_parity()`) menampilkan SEMUA 38 fitur + status vs training (bukan cuma
   7 chip hardcoded yang sudah ada).
3. **Tab "Tren Fitur"** di `/features` — endpoint baru `GET /api/features/history` query
   histori `Signal.feature_snapshot` per koin+fitur, dirender Chart.js dgn garis referensi
   p5/p95 training — visual langsung kelihatan kalau ada ledakan/flat-lining nilai.

### Temuan tambahan saat build (2 hal penting)
- **Kontaminasi outlier EKSTRIM di training data sendiri** (bukan cuma live!): `trend_accel_4h`
  punya puluhan ribu baris |value|>1000 di 1000SHIBUSDT (41,302/54,723=75%!), 1000PEPEUSDT
  (24,132), DOGEUSDT (6,918), TRXUSDT (2,463), ADAUSDT (83), HBARUSDT (486) — max absolut
  sampai **173 miliar**. `vwdp` lebih parah lagi: 653,876/861,289 (76%!) baris SEMUA koin
  |value|>1000, max **4.7 triliun**. Mean/std mentah jadi tidak berguna (mean bisa >1 juta
  padahal p5-p95 cuma puluhan). **Fix**: pakai median+MAD (median absolute deviation, robust
  sampai ~50% kontaminasi) utk "mean"/"std" di reference file, BUKAN mean/std mentah;
  `vwdp` direklasifikasi jadi "skip" (skala tidak stabil by design, sama spt cvd/ofi).
- **Root cause bug TON/PEPE (dari sesi audit sebelumnya) kemungkinan BUKAN cuma isu live** —
  kontaminasi historis di training corpus utk coin harga rendah/1000X-prefix menunjukkan
  ini kemungkinan bug lama di `core/features.py` (bukan insiden baru 2026-07-03), sudah ada
  sejak data historis 2020-2025 utk beberapa coin. **Belum di-root-cause** — perlu investigasi
  terpisah kenapa `trend_accel_4h`/`vwdp`/EMA-slope meledak utk koin tertentu.

### Verifikasi
- Playwright headless: 3 halaman (modal per-signal, tab Parity, tab Tren) — screenshot OK,
  0 console error (kecuali warning Tailwind CDN, pre-existing).
- Post-deploy `curl /api/features/parity?refresh=true`: **`long_short_ratio` 0 error di
  21 koin** (fix positioning_mode terkonfirmasi bekerja lintas semua koin, bukan cuma
  sample). **TONUSDT langsung ter-flag** (`dist_liq_50x_*`, `trend_accel_4h` s.d. 2.29e+25,
  `ema_*`) — bukti dashboard baru ini BEKERJA menangkap bug yang masih terbuka.
- 12 koin lain menunjukkan warning ringan `trend_accel_4h` (nilai puluhan-ribuan, bukan
  meledak) — kemungkinan sifat fitur ini yang emang fat-tailed, bukan bug baru; perlu
  observasi lanjutan sebelum reklasifikasi check-type.

### Kesimpulan
- Dashboard monitoring fitur per-signal per-koin sudah live dan **terbukti bekerja** (langsung
  menangkap 1 bug yang masih ada). Auto-refresh 60 detik, alert Telegram utk error/warning
  (existing infra, cuma reference-nya yang diperbaiki).
- **Follow-up terbuka**: root-cause bug TONUSDT/1000PEPEUSDT/1000SHIBUSDT (ledakan
  `trend_accel_4h`/`vwdp`/EMA-slope) — kemungkinan bug lama di feature engineering utk
  coin harga rendah, bukan insiden baru. Belum diinvestigasi/diperbaiki.

### Script & Artefak
- `regen_training_feature_standards.py` (scratchpad sesi ini)
- Kode: `app/api/signals.py`, `app/api/features_bp.py`, `app/templates/signals.html`,
  `app/templates/features.html` (swint_tradev2)
- Data: `models/training_feature_standards.json` (regenerasi)

---

## 2026-07-04 — DEPLOY ic32_regime_v6.1 (H4-closed + funding_rate + TON→GRAM) + INSIDEN market-panel holdout NaN

**Status**: DONE — v6.1 live di VPS (commit swint `051a516`, snapshot `2026-07-03 18:05:00` UTC).

### Stack v6.1 (nama run TIDAK berubah, model di-retrain in-place)
| Komponen | Run | Perubahan |
|----------|-----|-----------|
| LGBM | `opt2_plus_trend` (lgbm38f) | Retrain 2026-07-04 01:05 — fitur H4 closed-candle seragam + label dari swing H4 yang benar |
| Guardian | `guard_opt2_plus_trend_hmm` (guard28f) | Retrain 2026-07-04 01:18 — + `funding_rate` real (backfill 2020-2026, dulu 100% NaN) |
| HMM | hmm24/48, gate 0.65/0.05 | Tidak berubah (fitur internal H4 native sudah kausal; diverifikasi empiris Viterbi konsisten) |
| Koin | 21 | **TONUSDT → GRAMUSDT** (rebrand 1:1, TON `SETTLING`, GRAM listing 2026-07-02; histori TON dibawa, disambung data GRAM real) |

### Akar perubahan (kronologi sesi 2026-07-03→04)
1. **Fix H4 tidak seragam** (`core/features.py`): EMA/RSI "H4" dulu dihitung dari close H1 (salah timeframe);
   swing points H4 pakai expanding bucket berjalan + shift salah satuan (3 jam, harusnya 3 candle=12 jam).
   Sekarang: resample 4h → shift(+4h) → ffill ke H1 (kausal, = live). Label training (TP/SL dari swing H4)
   otomatis ikut benar. `ofi_h4_delta`/`cvd_div_h4`/`cvd_slope_h4` TIDAK terdampak (sudah benar sejak awal).
2. **`funding_rate` backfill** — bukan bug join, tapi data mentah tidak pernah di-fetch penuh (cuma Apr-Jun 2026).
   Backfill fapi 2020→sekarang, 21 koin, training+holdout.
3. **INSIDEN BARU (tertangkap audit pra-deploy)**: `_MKT_PANEL_PATH` di `engineer.py` selalu menunjuk panel
   TRAINING — di mode `--holdout-test`, panel (index < cutoff) di-merge ke frame holdout (index >= cutoff)
   → `btc_ret_24h`/`mkt_breadth_*`/`btc_minus_mkt_24h`/`mkt_ret_*` **NaN semua** di holdout labeled sejak
   regenerasi 2026-07-03 22:58. SEMUA angka OOS yang dihitung 2026-07-03 22:58 → 2026-07-04 09:49 tercemar
   (fitur market ke-nol-kan/NaN saat eval). Fix: panel holdout dibangun dari data holdout + cache di dir holdout.
   Training TIDAK terdampak (retrain valid).

### Scorecard v6.1 (angka BENAR, setelah fix panel)
| Metrik | OOF full (2020→Apr26) | OOS holdout (Apr→3 Jul 26) | OOS v6.0 pembanding (pra-H4-fix) |
|---|---|---|---|
| Trades | 4,602 | 187 | 571 |
| WR | 66.8% | 58.8% | 50.4% |
| PF | **2.236** | **1.158** | 0.927 (rugi) |
| PnL | $2,327 | +$11.42 | -$18.11 |
| MaxDD | -$22.30 | -$12.50 | -$41.47 |
| Long/Short PF | 2.141 / 2.302 | 0.551 / 1.439 | 0.567 / 1.154 |

> ⚠️ Angka OOS "257 trades / PF 1.484 / +$42.10" yang sempat dilaporkan & tercatat di config = **TERCEMAR
> bug panel NaN, JANGAN disitir**. Registry & inference_config sudah dikoreksi ke 187/1.158/+$11.42.
> OOS per bulan (benar): Apr 64tr PF 1.623 +$11.22 · Mei 41tr PF 0.988 -$0.24 · Jun 78tr PF 1.063 +$2.06 · Jul(3hr) 4tr -$1.62.

### Deploy & verifikasi
- `training_feature_standards.json` regenerasi dari labeled_opt2 baru (median/MAD; `dist_liq_50x_*` & `trend_strength` std ×2.2 — efek H4 fix).
- `verify_hmm_feature_parity.py`: PASS 38/38 (setelah fix panel; sebelumnya FAIL 34/38 — itulah yang membongkar insiden).
- VPS: git `051a516`, `core/features.py` identik konten (beda CRLF saja), `positioning_mode=live`,
  GRAMUSDT di pairs + `GRAMUSDT_hmm.pkl` ter-scp, `TONUSDT_hmm.pkl` dihapus. DB VPS: koin GRAMUSDT id=23 aktif
  (auto-register), TONUSDT id=8 dibiarkan sbg arsip histori (unique constraint, 1.051 signal lama tetap utuh).
- Follow-up terbuka: LONG masih lemah di OOS (PF 0.551, 33 trade — asimetri loss besar vs win kecil); GRAM
  data live baru ~2 hari; TONUSDT lama masih muncul di arsip UI.

### Script & Artefak
- Fix: `core/features.py` (`cache_market_panel` + proc_dir/label_dir), `pipeline/data/core/engineer.py` (`_MKT_PANEL_PATH` holdout)
- `models/runs/guard_opt2_plus_trend_hmm/oos_holdout_full_scorecard.json` (+ trades detail + monthly CSV)
- Migrasi GRAM: `extend_gram_data.py` (scratchpad), rename 23 file/dir TON→GRAM

### INSIDEN #3 (pasca-deploy, 2026-07-04 08:20 WITA) — `h4_trend`/`MSB_BOS` ERR 21/21 di dashboard live

**Sebab**: regenerasi `training_feature_standards.json` (langkah di atas) cuma meng-copy key `check`+`max_z`
dari file lama, tidak ikut copy `valid` (daftar kategori sah, mis. `[-1,0,1]`) untuk fitur `check="categorical"`.
Akibat: `feature_monitor.check_feature_parity()` baca `spec.get("valid", [])` → list kosong → SEMUA nilai
live dianggap "di luar valid []", 21/21 koin ERROR di alert Telegram + dashboard `/features`.

**Fix**: tambahkan kembali `"valid": [-1, 0, 1]` ke `h4_trend` DAN `MSB_BOS` (keduanya ternary -1/0/1 —
`MSB_BOS` sebelumnya "moderate" di file sangat lama, sempat berubah jadi "categorical" di regenerasi
antara sesi ini, dipertahankan krn memang lebih tepat utk variabel diskret). Deploy: scp file ke swint
lokal + VPS, **restart `systemctl restart swint-trade.service`** (cache module-level `_stds_cache` tidak
auto-reload), verifikasi via `curl .../api/features/parity?refresh=1` (endpoint tanpa `refresh=1` baca
`PARITY_REPORT` cache disk, bukan live-compute — WAJIB pakai `refresh=1` utk verifikasi pasca-fix).

**Hasil setelah fix**: 9 OK, 11 WARNING (`trend_accel_4h` — bug lama TONUSDT/1000PEPEUSDT dkk, **sudah
terdokumentasi, bukan baru**), 1 ERROR (`GRAMUSDT` — "no cache", lihat di bawah). 0 error utk h4_trend/MSB_BOS.

**Pelajaran**: saat regenerasi file reference dgn field campuran (numerik + kategorikal), jangan cuma
whitelist 2 key (`check`,`max_z`) — copy SELURUH dict lama lalu overwrite field numerik saja, supaya field
lain (`valid`, dst.) yang tidak di-regenerasi otomatis ikut terbawa.

### GRAMUSDT: "1h klines tidak cukup (42 bars, min 82)" / parity "no cache"

**Bukan bug** — GRAM listing 2026-07-02 08:00 UTC, baru ~2 hari saat insiden ini (perlu ~3.4 hari utk
82 bar H1). Pipeline live BENAR menolak generate sinyal sampai histori cukup — akan resolve sendiri
dlm ~1.5 hari tanpa tindakan. TONUSDT lama (id=8 di DB) sengaja dibiarkan sbg arsip, tidak dihapus.

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
