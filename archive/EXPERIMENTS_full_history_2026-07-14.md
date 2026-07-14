# EXPERIMENTS.md -- Arsip Histori Lengkap (2026-07-08 s.d. 2026-07-12)

> Dipindah dari EXPERIMENTS.md 2026-07-14 (file utama menembus ~1900 baris lagi).
> Index 1-baris-per-entry ada di EXPERIMENTS.md -- file ini isi LENGKAP tiap entry.

---

## 2026-07-08 — Insiden: Drift Tak Terdokumentasi 07-06 (partial-H4 + Guardian OFF) — Audit + Rollback ke v6.1

**Status**: DONE — root cause ditemukan, production di-rollback ke v6.1 (verified), repo dibersihkan.

### Konteks
Alert monitor Telegram melaporkan `long_short_ratio`/`cvd_slope_h4`/`cvd_momentum_adv` ERROR di 21/21 koin. Investigasi awal (`training_feature_standards.json` basi, generated 2026-06-19 vs versi benar 2026-07-04) memicu audit lebih dalam: ditemukan production sejak **2026-07-06 22:25 UTC** diam-diam menjalankan `ic32_regime_v6.2_baseline_noguardian` (LGBM `opt2_plus_trend_local`, Guardian **disabled**) — deploy nyata (commit swint `9c9ff70`, model benar-benar retrain) tapi **nol jejak** di `EXPERIMENTS.md`/`models/model_registry.json` repo ini. Commit yang sama membawa perubahan `core/features.py` yang tidak pernah disinkronkan balik ke Riset:
1. `trend_accel_4h`: regresi bug double-ATR-normalization (sempat "diperbaiki" 07-04, balik rusak 07-06) — **terbukti aktif di live** (z-score puluhan ribu di 1000PEPEUSDT/1000SHIBUSDT dkk saat diverifikasi via `/api/features/parity` + parquet cache langsung).
2. H4 trend/EMA/RSI dibalik dari closed-candle → partial/expanding candle (membalikkan fix headline v6.1).
3. Swing high/low look-ahead shift direlokasi (posisi shift beda, tidak tervalidasi ulang).
4. `cache_market_panel()` kehilangan param `proc_dir`/`label_dir` (membalikkan fix insiden market-panel-NaN 07-04, dead code di live tapi berbahaya kalau dipakai lagi utk holdout).

Ditemukan juga bug independen di `app/services/paper_trading.py` (swint): TP-touch selalu mengaktifkan Guardian momentum-mode + trailing floor **terlepas dari `guardian.enabled`** — jadi Guardian yang "dimatikan" tetap mempengaruhi exit behavior.

### Tindakan
1. Sync `training_feature_standards.json` benar ke VPS + restart.
2. Fix `paper_trading.py`: TP-touch close normal (`tp_hit`) saat `guardian.enabled=false` (commit swint `10fb41c`).
3. User putuskan: closed-candle adalah standar yang benar → **rollback penuh** via `tools/ops/deploy_production.py`: `core/features.py` (closed-H4, sudah benar di Riset) + model `opt2_plus_trend`/`guard_opt2_plus_trend_hmm` (v6.1, Guardian ON, param 0.65/0.7) — commit swint `033ed81`.
4. Verifikasi pasca-deploy: `core/features.py` Riset vs VPS byte-identical; `trend_accel_4h` di parquet cache kembali wajar (dulu ribuan, sekarang -0.3 s/d 0.3); posisi live tidak terganggu (`active_trades` tetap jalan selama restart).

### Pelajaran
- **Model/config yang di-scp langsung ke VPS (bukan lewat `deploy_production.py` dari Riset) lolos dari semua tracking** — `model_registry.json`/`EXPERIMENTS.md` di repo ini jadi tidak sinkron dengan kenyataan produksi. Kalau ada eksperimen yang mau dicoba live di luar alur normal, **wajib dicatat manual** di `EXPERIMENTS.md` + `model_registry.json`, bukan cuma commit message di repo swint.
- **Dashboard monitoring live (`/api/features/parity`) cache per-jam** (`check_features` job, `HH:20`) — jangan simpulkan fix gagal/berhasil dari dashboard tanpa cek `checked_at` timestamp-nya dulu; cek parquet cache (`data/inference/{symbol}.parquet`) langsung kalau butuh verifikasi cepat.
- Repo Riset ini dan `swint_tradev2/core/features.py` **harus selalu diverifikasi identik** (`diff`/`md5sum`, bukan asumsi) sebelum & sesudah deploy — insiden ini murni karena driftnya tidak pernah dicek selama 2 hari.

### Cleanup Repo (bersamaan)
- 86 folder `models/runs/` (2.1GB) tidak direferensikan → `archive/models_runs_2026-07-08/`
- 25 script `pipeline/experiments/` tanpa jejak di `EXPERIMENTS.md` → `archive/pipeline_experiments_2026-07-08/`
- `terminals/` (11MB) → `archive/terminals_2026-07-08/`
- Detail lengkap: `archive/EXPERIMENT_INDEX.md` § 2026-07-08

---

## 2026-07-08 — KANDIDAT DIKUNCI: fs38_18coin_spotconfirm (18-koin + spot-confirm + regime-disable + Guardian)

**Status**: **CANDIDATE — dikunci sementara**, bukan production. SSOT: `model/stacks/fs38_18coin_spotconfirm/stack.json`.

### Ringkasan
Hasil rangkaian eksperimen hari ini (universe 18-koin baru, model spot-trade eksternal
`Riset_spottrade` sbg confirmation, tuning HMM/regime). Arsitektur:

```
LGBM (opt2_plus_trend_18coin, 38f)
  -> spot-confirm fusion (agree_boost=0.08, opposite_pen=0.35, thr=0.60)
  -> HMM gate default (vol=24/mom=48, base=0.65/delta=0.10)
  -> block LONG jika regime=TRENDING_DOWN
  -> entry
  -> Guardian exit (guard_opt2_plus_trend_hmm_18coin, 28f, no cross-model) -- lihat § Update lanjutan
```

Guardian awalnya di luar scope (lihat § Update 2026-07-08 lanjutan utk hasil pengujian).

### Hasil (OOF genuine walk-forward + OOS genuine holdout 2026-04-01 s.d. 2026-07-07)

| Tahap | OOF trades | OOF PF | OOS trades | OOS PF |
|---|---|---|---|---|
| LGBM saja | 1.631 | 1.0898 | — | — |
| + HMM default | 5.436 | 1.1668 | 270 | 1.0532 |
| + spot-confirm | 5.375 | 1.3293 | 240 | 1.1344 |
| **+ regime-disable (final)** | **5.327** | **1.3516** | **239** | **1.1441** |

Arah membaik konsisten OOF & OOS (tidak berbalik) — beda dgn HMM fast-react yg ditolak
karena menang OOF (PF 1.49) tapi kalah OOS (PF 1.06).

### Ditolak (sudah diuji, jangan ulang tanpa info baru)
- HMM fast-react (vol=6/mom=12) — kontradiksi OOF-vs-OOS.
- `max_proba` ceiling (ide dari `hmm_orchestrator.py` Riset_spottrade) — tidak ada pola
  "exhaustion" di data ic32.
- **Guardian rule "TP-proximity tolerance"** (exit kalau sisa jarak ke TP <30% & momentum
  lemah, hold kalau momentum kuat) — fitur `tp_progress_pct` causal (reuse `tp_proximity()`
  dari `guardian_k5mom_inference.py`, sudah pernah dicoba dulu di saga k5mom v1-v7 dan
  ditolak). Diuji ulang 2026-07-08 di context baru (18-koin+spot-confirm), 3 iterasi:
  rule dicek sebelum M2/M3 (motong winner, PF turun), tidak sengaja tukar prioritas M2/M3
  (lebih rusak), murni ditambah setelah M1/M2/M3 asli (**M1b tidak pernah aktif, 0 sample
  di semua koin** — M2+M3 sudah exhaustive utk kasus current_pnl>0.003, tidak ada celah
  tersisa). Kesimpulan: rule M1/M2/M3 yang ada SUDAH mengimplementasikan "hold saat
  momentum, exit dekat puncak" — cuma pakai proxy hindsight (`best_future_pnl`), bukan
  TP tetap. Tidak ada ruang buat rule tambahan tanpa mengubah prioritas rule lama.
- **Metodologi OOF Guardian scorecard TERBUKTI tidak stabil**: v3 di atas pakai label
  IDENTIK dgn baseline (M1b 0 sample) + 1 fitur tambahan inert, tapi PF OOF berubah drastis
  (2.088→1.565) padahal CV genuine walk-forward hampir sama. Bukti kuat metodologi
  "final model discore ke populasi sendiri" didominasi noise overfitting kapasitas tinggi
  (2000 tree), BUKAN sinyal skill asli. Jangan percaya angka OOF trade-level utk
  membandingkan varian Guardian — cuma OOS ablation yang valid.

### Model spot-trade eksternal (Riset_spottrade) — read-only, tidak pernah diubah
2 LightGBM binary (LONG/SHORT terpisah), 20 fitur H4/D1/macro/cross-market — beda total
dari fitur ic32. Backtest historis pakai file OOF genuine walk-forward mereka sendiri
(`data/spottrade/oof_{long,short}.csv`, 8-fold + purge 84 bar), BUKAN model deployed
(itu di-fit single-split, in-sample utk sebagian besar histori — ketahuan & diperbaiki
saat validasi, lihat detail di `reports/oof_18coin_vs_v6.1.md`).

### Belum
- Data eksperimen masih **isolated** (`pipeline/experiments/guardian_reeval_2026-07-08/data_18coin/`),
  belum dipromosikan ke `data/training/` produksi.
- Belum wiring ke `models/inference_config.json` / live — butuh approval terpisah.

### Update 2026-07-08 (lanjutan) — Guardian diuji, OOF positif

Guardian baru `guard_opt2_plus_trend_hmm_18coin` (guard28f_18coin, 28f no cross-model)
dilatih dari entry population **stack penuh** (LGBM+spot-confirm+HMM+regime-disable),
bukan cuma LGBM+HMM polos seperti guard28f production lama. Recipe fitur & label sama
persis dgn guard28f (`tools/model/train_guardian_opt2_plus_trend.py`).

CV genuine walk-forward (8-fold purge): logloss 0.2765-0.3308, F1 macro 0.51-0.65 — sehat,
tidak ada fold yang njomplang (indikasi tidak overfit parah).

| Skenario | Trades | WR | PF | PnL | MaxDD |
|---|---|---|---|---|---|
| Full OOF, no Guardian | 5.327 | 43.7% | 1.348 | $1.030,80 | -$63,29 |
| Full OOF, + Guardian | 5.742 | 67.3% | 2.088 | $2.859,54 | -$24,08 |
| Pseudo-holdout (Okt25-Mar26), no Guardian | 399 | 44.4% | 1.409 | $70,32 | -$12,28 |
| Pseudo-holdout, + Guardian | 427 | 66.5% | 2.463 | $225,81 | -$8,87 |

**Caveat metodologi** (sama dgn benchmark resmi guard28f production): angka trade-level di
atas pakai Guardian final (fit ke SEMUA sample) discore ke populasi yang sama — semi
in-sample utk kebijakan exit, BUKAN klaim walk-forward murni utk exit-policy. LGBM entry
tetap genuine OOF.

### Update 2026-07-08 (lanjutan lagi) — TP-proximity rule DITOLAK + OOS Guardian genuine SELESAI

**TP-proximity tolerance rule** (ide: exit kalau sisa jarak ke TP <30% & momentum lemah,
hold kalau momentum kuat) — diuji 3 iterasi, **DITOLAK**. Detail lengkap di § Ditolak di
atas. Ringkas: rule jadi 0-sample (tidak pernah aktif) begitu prioritas M2/M3 asli
dikembalikan — sudah redundant dgn logic existing. Sekalian ketemu bukti kuat metodologi
OOF trade-level scorecard tidak stabil (didominasi noise overfitting).

**OOS ablation genuine** (holdout-test 2026-04-01 s.d. 2026-07-07, data belum pernah
dilihat training) — baseline Guardian (`guard_opt2_plus_trend_hmm_18coin`, BUKAN varian
tpprox yg ditolak):

| | Trades | WR | PF | PnL | MaxDD | Peak modal bersamaan |
|---|---|---|---|---|---|---|
| Tanpa Guardian | 239 | 41.0% | 1.144 | $13,59 | -$12,88 | $110 (11 trade) |
| **+ Guardian** | 249 | **57.4%** | **1.315** | **$29,61** | **-$8,31** | $100 (10 trade) |

Delta genuine: PF +0.171, PnL +$16,02, WR +16,4pp, MaxDD membaik $4,57. **Guardian
sungguh membantu** — tapi delta PF OOS (+0.171) cuma **~15% dari delta OOF in-sample**
(+1.135, dari 1.348→2.088) — konsisten dgn pola production (delta OOS production +0.334
dari delta OOF +1.135, ~29%). Pelajaran: **jangan pernah pakai angka OOF Guardian
trade-level sebagai ekspektasi riil** — selalu validasi OOS sebelum percaya besaran lift.

Status final: `model/stacks/fs38_18coin_spotconfirm/stack.json` → `guardian.enabled=true`,
`oos_ablation_genuine` terisi. Arsitektur candidate SELESAI diuji tahap ini. Belum ke
`models/inference_config.json` / live.

### Update 2026-07-08 (lanjutan lagi) — Keputusan: TIDAK deploy sekarang

Dibandingkan OOS-to-OOS (genuine, apples-to-apples) dgn v6.1 production:

| | v6.1 production (s.d. 07-02) | Kandidat 18-koin (s.d. 07-07) |
|---|---|---|
| Trades | 308 | 249 |
| PF | 1.264 | 1.315 |
| PnL | $32,94 | $29,61 |
| MaxDD | -$12,21 | -$8,31 |
| Long PF | **0.667** (lemah, kelemahan terbuka v6.1) | **1.433** |
| Short PF | 1.575 | 1.269 |

Baca: PF/MaxDD sedikit lebih baik, PnL sebanding, **asimetri LONG v6.1 yang selama ini jadi
kelemahan terbuka diperbaiki signifikan** — tapi short PF kandidat lebih lemah dari v6.1,
dan sample cuma ~250-300 trade (jangan overclaim signifikansi statistik).

**Keputusan**: kandidat riset kuat & menjanjikan (khususnya perbaikan long/short balance),
tapi **TIDAK deploy sekarang** — banyak pekerjaan engineering & validasi tersisa. Status
tetap `"candidate"`.

### Update 2026-07-08 (lanjutan lagi) — Prep kerja "siap deploy" (bagian aman/additive)

User minta siapkan semuanya biar layak deploy. Dikerjakan bagian yang **aman & additive**
(tidak sentuh artefak production yang sedang dipakai v6.1 live), sisanya butuh akses/
keputusan production terpisah:

**Selesai (additive, diverifikasi tidak overwrite production):**
- Promosi data 6 koin baru (LTCUSDT, ATOMUSDT, UNIUSDT, FILUSDT, ETCUSDT, BCHUSDT):
  `labeled_opt2` (training) + `labeled`+`regime_h1` (holdout-test) dicopy dari isolated
  experiment dir ke `data/training/labeled_opt2/` & `data/holdout-test/labeled/`. Recipe
  dikonfirmasi identik production (pakai fungsi asli `pipeline.data.core.engineer` +
  `core.features.swing_based_labeling`/`structural_label_filter`, bukan reimplementasi).
- Ditemukan saat verifikasi: fetch+clean (`data/training/processed/`, `data/holdout-test/processed/`)
  dan HMM per-coin fit SSOT (`models/hmm/{coin}_hmm.pkl`, via `regime_hmm_holdout.py`) utk
  6 koin baru **SUDAH ADA** dari kerja sesi sebelumnya — bukan kerja baru, cuma dikonfirmasi.
  12 koin overlap dgn production **tidak disentuh** (timestamp & git status dicek, bersih).
- `models/inference_config_candidate_18coin.json` — draft baru LENGKAP (universe 18-koin,
  spot_confirm block, regime_disable block, scorecard OOF+OOS genuine), file TERPISAH,
  tidak overwrite `models/inference_config.json` production.
- Audit parity fitur: `audit_feature_value_parity.py` **tidak bisa dipakai apa adanya** —
  hardcoded ke `config.ALL_COINS` (21-koin) + baca live signal DB yang cuma berisi koin yang
  SUDAH live (chicken-and-egg: 6 koin baru belum pernah live, tidak ada snapshot buat
  dibandingkan). Sbg gantinya dijalankan diagnostik presence/null-rate/degenerate utk 48
  fitur unik (38 LGBM + 28 Guardian) di 6 koin baru — **bersih**, tidak ada kolom
  hilang/null tinggi/konstan. Ini BUKAN pengganti penuh audit live-vs-training resmi.
- Rencana rollback ditulis (`stack.json` § `rollback_plan`): balik `ACTIVE_STACK` ke
  `fs38_28f`, wajib backup artefak v6.1 sebelum deploy kandidat.

**BELUM — sengaja tidak dikerjakan otomatis (butuh keputusan/akses terpisah):**
- **Live feed 6 koin baru** — perlu cek VPS/swint_tradev2 (di luar scope repo riset).
- **Wiring spot-confirm ke live** (`live_db_bridge.py`/`paper_trading.py`) — ini
  memodifikasi kode yang dipakai monitoring v6.1 yang SEDANG BERJALAN; sengaja tidak
  disentuh tanpa approval eksplisit terpisah karena risiko mengganggu monitoring aktif.
- **`deploy_production.py`** — tidak akan dijalankan tanpa approval eksplisit terpisah.

Detail lengkap per-item: `model/stacks/fs38_18coin_spotconfirm/stack.json` §
`deploy_readiness_checklist`.

### Artefak
- Stack SSOT: `model/stacks/fs38_18coin_spotconfirm/stack.json`
- Report lengkap: `reports/oof_18coin_vs_v6.1.md`
- Model LGBM: `models/runs/opt2_plus_trend_18coin/`
- Model Guardian: `models/runs/guard_opt2_plus_trend_hmm_18coin/` (`guardian.pkl`, `guardian_scaler.pkl`,
  `guardian_features.json`, `guardian_cv_results.json`, `oof_scorecard_guardian_18coin.json`,
  `oos_holdout_guardian_ablation_18coin.json`)
- Guardian tpprox (DITOLAK, disimpan utk referensi): `models/runs/guard_opt2_plus_trend_hmm_18coin_tpprox{,_v2,_v3}/`
- Spot-confirm: `core/spottrade_confirm.py`, `core/cascade_utils.py::apply_spot_confirm_fusion_pre`,
  `models/spottrade_confirm/`, `data/spottrade/`
- Config kandidat: `models/inference_config_candidate_18coin.json` (draft, belum live)
- Data produksi (6 koin baru, additive): `data/training/labeled_opt2/{6coin}_features_v3.parquet`,
  `data/holdout-test/labeled/{6coin}_{features_v3,regime_h1}.parquet`, `models/hmm/{6coin}_hmm.pkl`
- Script eksperimen: `pipeline/experiments/guardian_reeval_2026-07-08/run_*.py`
  (termasuk `run_train_guardian_18coin.py`, `run_oof_scorecard_guardian_18coin.py`,
  `run_oos_scorecard_guardian_full_18coin.py`)

---

## 2026-07-09 — Investigasi fitur market-context (btc_ret_24h fade, R5 monotone, R1, coin_mkt_sync_24h)

**Status**: SELESAI diinvestigasi. **v6.3 tetap live tanpa perubahan** — R5 disimpan sbg
kandidat riset tervalidasi, TIDAK di-deploy.

### Pemicu
Audit `audit_feature_value_parity.py` pasca-deploy v6.3 menemukan 2 fitur ter-flag:
`btc_minus_mkt_24h`, `coin_mkt_sync_24h` (std_ratio rendah, live vs training). User
sudah lama curiga ke 2 fitur ini.

### Temuan 1 — disparitas audit: bukan bug, artefak jendela waktu
`btc_minus_mkt_24h` itu fitur MARKET-WIDE (sama utk semua koin di jam yg sama, dari
`build_market_panel_from_closes`), jadi sampel live "108 baris" itu cuma 6 nilai unik
diulang 18x. Dibanding training (ribuan titik waktu independen 6 tahun), std pasti jomplang.
Nilai aktualnya TERBUKTI tidak konstan (mkt_ret_24h bervariasi -0.013 s.d +0.009 dlm 6 jam) —
bukan di-strip spt insiden Juli lalu.

### Temuan 2 — riwayat lama: grup market-context lemah + 1 fitur ada fade bug
Dari `reports/TRADE_ANALYSIS_REPORT.md` (2 Juli, model v5 lama, belum pernah
ditindaklanjuti): grup 6 fitur market/regime cuma 4.4% gain, dan `btc_ret_24h` "belajar
polaritas fade" (BTC naik → model malah dorong SHORT). Verifikasi ulang di v6.3
(SHAP, sample 12rb baris 6 koin): **fade MASIH ADA** — `btc_ret_24h` corr(val,SHAP_LONG)=-0.386,
`mkt_breadth_4h`=-0.163. `btc_minus_mkt_24h` (+0.316) & `coin_mkt_sync_24h` (-0.223) tidak
separah itu.

### R5 — monotone_constraints pada btc_ret_24h + mkt_breadth_4h
Retrain isolated `opt2_plus_trend_18coin_mono` (identik baseline, +`monotone_constraints=+1`
pada 2 fitur itu). Hasil:
- **Fade hilang total**: btc_ret_24h -0.386→**+0.938**, mkt_breadth_4h -0.163→**+0.877**
- **OOF (tanpa Guardian, semua tahap)**: R5 konsisten menang — LGBM saja PF 1.090→**1.136**,
  +HMM 1.167→**1.217**, +spot-confirm 1.329→**1.363**, +regime-disable 1.352→**1.371**
  (trade lebih sedikit tapi lebih selektif/akurat)
- **OOS genuine (2026-04-01 s.d. 07-07), tanpa Guardian**: R5 menang — PF 1.144→**1.203**,
  PnL $13.59→**$18.35**, MaxDD -$12.88→**-$11.79** (tapi Long PF turun 0.999→0.787)
- **OOS genuine, DENGAN Guardian (retrain baru `guard_..._18coin_mono`)**: baseline sedikit
  menang — PF **1.315** vs 1.293, PnL **$29.61** vs $26.22, MaxDD **-$8.31** vs -$10.98
- **Kesimpulan**: hasil campuran di full-stack (sample kecil ~230-250 trade, dalam rentang
  noise). Fix fade-nya riil & tervalidasi, tapi tidak cukup meyakinkan utk gantikan v6.3.
  **Keputusan: tetap v6.3, simpan R5 sbg kandidat riset tervalidasi OOF+OOS.**

### R1 — dekorelasi geometri: TIDAK BERLAKU LAGI, tidak dieksekusi
Cek ulang: 4 dari 7 fitur geometri target R1 lama (`dist_liq_20x_long/short`, `Buy_Liq`,
`Sell_Liq`) **sudah tidak ada** di feature set v6.3 (38f) — entah kapan dihapus, bukan hasil
kerja R1. Sisa 2 fitur (`dist_liq_50x_long/short`, 39.9% gain gabungan) TIDAK kolinear kuat
lagi (corr -0.27). Replikasi skenario insiden asli (stretched + BTC pump) di label training:
rasio SHORT:LONG cuma 1.17:1 (dulu 1.3:1) dan mayoritas (89.5%) malah FLAT, bukan SHORT —
label recipe "Opsi 2" (`ofi_z_score>1.5` + `structural_label_filter`) kemungkinan sudah
menyelesaikan masalah ini duluan lewat cara lain. **Ditutup, tidak perlu ablation.**

### coin_mkt_sync_24h — dicek terpisah, TIDAK ada bug
Fitur ini `coin_r24 × mkt_ret_24h` — perkalian 2 return, jadi nilai TINGGI bisa berarti
"bullish bareng" ATAU "bearish bareng" (ambigu arah by design). Cek label pada nilai
tinggi (top 10%), dipecah proksi `h4_trend`: LONG selalu sedikit > SHORT di kedua sub-grup
(bullish-proxy 8.0% vs 6.8%; bearish-proxy 7.5% vs 5.8%) — tidak ada bias SHORT. Pola
"dorong ke FLAT" (86% FLAT saat nilai tinggi vs baseline 78%) konsisten dgn label asli,
bukan pola salah yang dipelajari model. **Tidak perlu tindakan.**

### Artefak
- LGBM: `models/runs/opt2_plus_trend_18coin_mono/` (features.json, lgbm.pkl, cv_results.json,
  oof_scorecard_progression.json, holdout_predictions.parquet)
- Guardian: `models/runs/guard_opt2_plus_trend_hmm_18coin_mono/` (guardian.pkl, guardian_scaler.pkl,
  oof_scorecard_guardian_18coin.json, oos_holdout_guardian_ablation_18coin.json)
- Script: `pipeline/experiments/guardian_reeval_2026-07-08/run_train_lgbm_18coin_monotone.py`,
  `run_train_guardian_18coin_mono.py`, `run_oof_scorecard_mono_progression_18coin.py`,
  `run_score_holdout_18coin_mono.py`, `run_oos_scorecard_guardian_full_18coin_mono.py`
- Referensi: `reports/TRADE_ANALYSIS_REPORT.md` (analisis kausal asli, rekomendasi R1-R6)

---

## 2026-07-10 — Audit parity live-vs-riset (trigger XRPUSDT SHORT@1.0911 salah arah) + fix

**Status**: SELESAI. 2 bug live DIFIX & DIDEPLOY. 1 fitur di-takedown dari feature set.

### Pemicu
User komplain entry SHORT XRPUSDT@1.0911 (07-10 09:05 WITA) tepat sebelum pump. Audit
trade-by-trade (STAR-style, `tools/ops/live_db_bridge.py` + trace manual `feature_snapshot`)
menemukan p_short murni LGBM live=0.5662 vs offline-recompute=0.4719 di bar yang sama —
gap besar di fitur order-flow.

### Bug 1 — taker buy/sell volume (FIXED & DEPLOYED)
Root cause: `data_service._attach_real_positioning` (live) override `1h_taker_buy/sell_volume`
dari `data/positioning/{sym}_taker_ratio.parquet` (job fetch HH:02) via ffill murni — row
jam baru sering belum publish/parsial saat `generate_signals` baca HH:05, ffill nempel bar
sebelumnya. Rusak ~10/38 fitur LGBM (ofi_z_score, ofi_acceleration, vwdp, cvd*) — bisa
balik tanda. Fix: hapus override, pakai kline `taker_buy_volume` langsung (final saat bar
tutup, identik riset). Deploy: scp manual ke VPS (`swint-tradev2-shared-filesystem` utk
kode ternyata TIDAK auto-sync, beda dari observasi sesi sebelumnya) + restart `swint-trade`.
Terverifikasi: `buy_volume+sell_volume == volume` persis di bar live terbaru pasca-restart.

### Bug 2 — spot_confirm staleness (FIXED & DEPLOYED)
`live_spottrade` (app terpisah) refresh per H4 close (00/04/08/12/16/20 UTC) di +15m(fetch)/
+20m(feature)/+25m(scan) — bukan tiap jam. `spot_confirm_live.py::get_spot_confirm_proba`
tidak pernah cek `open_time` respons API vs H4 saat ini, jadi di 6 jam boundary H4/hari
(persis kasus XRPUSDT 00:05) ic32 dapat skor spot sampai 4 jam basi tanpa terdeteksi (bukan
error, fail-open lama tidak ke-trigger). Fix: `_expected_h4_open()` + skip fusion (fail-open)
kalau bar spot belum ter-refresh utk H4 saat ini. Tidak mengubah jadwal service manapun.
Diverifikasi: jam normal (bukan boundary) tetap dapat proba valid (tidak ada false-positive
reject); path stale belum sempat teramati langsung (nunggu boundary H4 berikutnya), pantau
via `journalctl -u swint-trade | grep "spot_confirm.*stale"`.

### Audit 7 fitur "market-panel/cross-source" (mkt_breadth_1h/4h, btc_ret_24h,
btc_minus_mkt_24h, coin_mkt_sync_24h, whale_retail_divergence, relative_strength_z)

Awalnya diduga bug (live 18-koin vs riset `config.ALL_COINS` 21-koin) — **user klarifikasi
18-koin live itu MEMANG universe current yang benar**, 21-koin riset yang usang. Setelah
riset rebuild market panel pakai 18-koin yang sama (fetch 6 koin baru ATOM/BCH/ETC/FIL/LTC/
UNI ke holdout-test): **4/5 fitur match EXACT** sampai banyak desimal begitu universe
disamakan — TIDAK ADA BUG. `whale_retail_divergence` ikut ke-fix via Bug 1 (formula pakai
`cvd`). `roll_corr_btc_48h/mkt_48h` (bukan input 38f, cuma perantara `coin_mkt_sync_24h`
yang hasil akhirnya match): root cause KETEMU & terverifikasi persis — `MKT_PANEL_BARS=32`
di live lebih pendek dari `_SYNC_WINDOW=48` yang dibutuhkan, ~16 bar rolling window
ter-nolkan. Simulasi panel 32-bar di riset menghasilkan angka match ke digit terakhir
dgn live. Tidak ada dampak trading (bukan input model), tidak difix (prioritas rendah).

**Ablasi OOF genuine walk-forward (LGBM+HMM, tanpa Guardian) 7 fitur ini vs baseline**:
JANGAN hapus semua — PF turun 1,229→1,098 (`opt2_plus_trend_ablate7mkt`, 31f), fiturnya
genuinely bermanfaat, gap sebelumnya murni artefak universe koin, bukan kualitas fitur.

### `relative_strength_z` — TAKEDOWN (keputusan user)

Satu-satunya dari 7 fitur yang gap-nya TIDAK terjelaskan (bukan universe koin, bukan
`MKT_PANEL_BARS` — cuma butuh close XRP+BTC, BTC di-fetch histori penuh terpisah).
Residual ~9% (live -1.1636 vs riset-recompute -1.2686), dugaan cache `_fetch_btc_close`
1 jam, tidak terbukti. Ablasi OOF genuine (`opt2_plus_trend_ablate_rsz`, 37f) vs baseline:
PF 1,2294→1,2260 (nyaris noise), PnL $758,35→$683,57 (-9,9%), MaxDD -$140,51→-$119,72
(membaik). Argumen performa murni netral/sedikit merugikan hapus — **user tetap pilih
takedown**: fitur dgn sumber tidak akuntabel & gap tak terjelaskan harus dibuang meski
OOF-nya netral, setara prioritas dgn feature selection formal (lihat memory
`feedback-feature-source-accountability`). **Status: riset selesai (37f jadi baseline
baru), belum di-deploy ke live** — deploy production perlu keputusan/approval terpisah
(audit parity value wajib dulu per `FEATURE_AUDIT.md`).

### Artefak
- LGBM 31f (ablasi 7 fitur, ditolak — jangan pakai): `models/runs/opt2_plus_trend_ablate7mkt/`
- LGBM 37f (takedown `relative_strength_z`, DITERIMA — baseline baru): `models/runs/opt2_plus_trend_ablate_rsz/`
- Kode live diubah: `app/services/data_service.py` (`_attach_real_positioning`),
  `app/services/spot_confirm_live.py` (`_expected_h4_open`) — swint_tradev2, deploy via scp manual
- Memory: `project-ofi-feature-parity-gap`, `project-spot-confirm-live-anomaly`,
  `feedback-feature-source-accountability`, `swint-tradev2-shared-filesystem` (koreksi)

### Belum
- Deploy LGBM 37f (`opt2_plus_trend_ablate_rsz`) ke production — butuh approval eksplisit
  + audit parity value (`audit_feature_value_parity.py`) sebelum jalan.
- Keputusan struktural: `config.ALL_COINS` riset (21 koin) vs universe live current
  (18 koin) — di luar scope hari ini, perlu keputusan eksplisit kapan disamakan.
- `roll_corr_btc_48h/mkt_48h` (`MKT_PANEL_BARS=32`) — tidak difix, tidak berdampak model.

---

## 2026-07-10 (lanjutan) — Jadwal live digeser HH:05→HH:15 + entry price M15 di OOF/OOS

**Status**: SELESAI. Jadwal live sudah diubah. OOF/OOS entry-M15 sudah disajikan, belum
ada keputusan pakai yang mana sbg acuan resmi.

### Latar belakang
User geser fetch `live_spottrade` dari +15m ke +3m setelah H4 close (bukti nyata dari
`pipeline_logs`: kerja fetch+refresh cuma ~55-80 detik, +15m lama itu buffer konservatif,
bukan kebutuhan riil). Konsekuensinya: `generate_signals` ic32 digeser HH:05→**HH:15**
(`app/jobs/__init__.py`, deploy scp+restart) — semantik bar/fitur TIDAK berubah
(`_complete_h1_bar_end()` tetap treat bar HH:00 sbg "belum settle" di HH:15 juga, sama
seperti di HH:05 dulu; keputusan tetap pakai bar HH-1:00, identik holdout).

### Entry price M15 — karena eksekusi nyata sekarang di HH:15, bukan HH:00
Backtest lama asumsikan entry = close bar H1 (harga di HH:00). Realitasnya entry order
baru dieksekusi di HH:15 (15 menit kemudian). Fetch M15 klines penuh 2020-01-01 s.d.
2026-07-10 (21 koin, `data/entry_price_m15/`) → `entry_price_hh15` = close candle M15
pertama tiap jam (open=HH:00, span HH:00-HH:15). Parameter `entry_price_override` di
`simulate_trades_swing` **sudah ada dari sesi sebelumnya** (comment eksplisit "harga
entry M15 per bar H1 opsional") — tinggal dipakai, tidak perlu ubah `core/evaluator.py`.

### Hasil — OOF (LGBM+HMM, genuine walk-forward, tanpa Guardian)

| | Entry H1-close (lama) | Entry M15@HH:15 (baru) |
|---|---|---|
| 38f baseline | PF 1,229 / PnL $758 / trades 5945 | PF **1,264** / PnL **$912** / trades 6125 |
| 37f (takedown rsz) | PF 1,226 / PnL $684 / trades 5630 | PF **1,290** / PnL **$920** / trades 5804, MaxDD -$100,78 |

### Hasil — OOS (full stack LGBM+HMM+Guardian, holdout 2026-04-01 s.d. 07-10)

| | Entry H1-close (lama) | Entry M15@HH:15 (baru) |
|---|---|---|
| 38f baseline | PF 1,128 / PnL $16,93 / MaxDD -$17,00 | PF **1,304** / PnL **$39,55** / MaxDD **-$14,17** |
| 37f (takedown rsz) | PF 1,208 / PnL $26,79 / MaxDD -$14,77 | PF **1,312** / PnL **$40,50** / MaxDD **-$12,55** |

**Baca:** entry M15@HH:15 memperbaiki PF & PnL substansial di KEDUA model (OOF & OOS) —
konsisten, bukan noise satu sisi. 37f tetap sedikit unggul dari 38f di kedua metodologi
setelah entry disesuaikan (OOS PF 1,312 vs 1,304, MaxDD lebih baik). Kenaikan performa
ini murni efek metodologi backtest jadi lebih realistis (entry lebih dekat ke harga
eksekusi nyata) — BUKAN klaim bahwa strategi jadi lebih untung di live; ini soal
menghilangkan bias optimis lama (entry di harga close H1 yang sebenarnya tidak pernah
benar-benar dieksekusi persis di situ).

### Belum
- Keputusan resmi: jadikan `entry_price_override` (M15) sbg metodologi STANDARD utk
  semua evaluasi OOF/OOS berikutnya (ganti benchmark resmi), atau tetap dua angka
  paralel. Belum ada keputusan eksplisit user.
- Update `model/CLAUDE.md` / `pipeline/model/CLAUDE.md` § Benchmark kalau metodologi
  entry-M15 dijadikan standar baru — belum dilakukan, tunggu keputusan di atas.

### Artefak
- `data/entry_price_m15/{coin}_15m.parquet`raw, `{coin}_entry_hh15.parquet` (harga entry per bar H1)
- Kode live diubah: `app/jobs/__init__.py` (`generate_signals` HH:15) — swint_tradev2
- Script eksperimen (scratchpad session, belum dipindah ke repo): `compare_oof_m15entry.py`,
  `compare_oos_m15entry.py`, `build_entry_hh15.py`, `fetch_m15_entry.py`

---

## 2026-07-10 (lanjutan lagi) — KOREKSI: takedown relative_strength_z diulang di universe 18-koin (bukan 21)

**Status**: SELESAI. Koreksi kesalahan — eksperimen takedown & entry-M15 sebelumnya (di atas)
salah pakai universe 21-koin (`opt2_plus_trend`), padahal live SUDAH pindah ke 18-koin
(`opt2_plus_trend_18coin`, per `tools/ops/deploy_model.py::ACTIVE_STACK`, ketahuan saat mau
deploy). Diulang penuh di universe 18-koin yang benar, hasil = konsisten menang lebih kuat.

### Kenapa terjadi
`model/stacks/fs38_28f/stack.json` & `models/model_registry.json` (dibaca sbg SSOT sepanjang
sesi) masih menunjuk `opt2_plus_trend` 21-koin — TIDAK sinkron dgn `deploy_model.py::ACTIVE_STACK`
yang sudah `opt2_plus_trend_18coin` (stack candidate `fs38_18coin_spotconfirm`, 07-08). Dokumentasi
SSOT belum diupdate merefleksikan pergeseran live ke 18-koin. **Belum diperbaiki** (di luar scope
hari ini) — lihat § Belum.

### Setup ulang (18-koin, isolated — TIDAK sentuh data/training/ production)
1. Lengkapi holdout-test 6 koin baru (ATOM/BCH/ETC/FIL/LTC/UNI) s.d. 2026-07-10: engineer + regime_holdout
   + sync-join (panel 18-koin, bukan 21).
2. Copy `data/training/labeled_opt2/` 18 koin relevan ke
   `pipeline/experiments/takedown_rsz_18coin/labeled_opt2/` (isolated), rebuild 4 kolom sync
   (`coin_mkt_sync_24h` dkk) pakai panel 18-koin konsisten di semua 18 (blocked otomatis coba
   overwrite production in-place — sengaja dialihkan ke isolated dir, lihat § insiden kecil).
3. Retrain 2 model dari data isolated: `opt2_plus_trend_18coin_iso38f` (38f, baseline apple-to-apple)
   & `opt2_plus_trend_18coin_iso37f` (37f, takedown `relative_strength_z`) — genuine walk-forward,
   purge gap sama persis metodologi production.
4. Fetch M15 tambahan 6 koin baru (2020-2025+OOS) — sebelumnya cuma 21-koin lama ter-cover.
5. Guardian: reuse `guard_opt2_plus_trend_hmm_18coin` (existing artifact 07-08) utk kedua varian —
   no-cross-model, konsisten dgn desain guard28f.
6. 6 koin baru tidak punya `hmm_regime_enc` OOF-fold di training (gap dari promosi data 07-08) —
   fallback state=1 utk OOF gating only (bukan bagian 38f), sama utk kedua model jadi tidak bias
   perbandingan.

### Hasil — OOF (LGBM+HMM, 18-koin, genuine walk-forward, tanpa Guardian)

| | 38f baseline | 37f takedown | Delta |
|---|---|---|---|
| H1-close entry | PF 1,126 / PnL $366,82 / MaxDD -$125,86 | PF 1,127 / PnL $351,84 / MaxDD -$113,28 | PF nyaris sama, PnL sedikit turun |
| M15@HH:15 entry | PF 1,184 / PnL $558,40 / MaxDD -$119,47 | **PF 1,203 / PnL $579,67 / MaxDD -$100,10** | PF/PnL/MaxDD semua membaik |

### Hasil — OOS (full stack LGBM+HMM+Guardian, 18-koin, holdout 2026-04-01 s.d. 07-10)

| | 38f baseline | 37f takedown | Delta |
|---|---|---|---|
| H1-close entry | 257 trade, PF 1,236, PnL $22,75, MaxDD -$9,73, LongPF 0,736 | **275 trade, PF 1,362, PnL $36,02, MaxDD -$9,11, LongPF 1,061** | PF +0,126, PnL +58%, LongPF pulih |
| M15@HH:15 entry | 265 trade, PF 1,679, PnL $60,72, MaxDD -$9,50, LongPF 1,167 | **278 trade, PF 1,816, PnL $73,57, MaxDD -$7,58, LongPF 1,696** | PF +0,137, PnL +21%, MaxDD & LongPF membaik |

**Baca:** di universe 18-koin yang BENAR (bukan 21-koin sesi sebelumnya), takedown
`relative_strength_z` menang **lebih tegas** dari yang terlihat di eksperimen 21-koin —
terutama di OOS full-stack (PF +0,126 s.d +0,137 di kedua konvensi entry, konsisten). Yang
paling mencolok: `long_pf` yang selama ini jadi kelemahan terbuka stack production (0,66-0,74)
**pulih signifikan** di 37f (1,06-1,70) — kemungkinan `relative_strength_z` yang gap live-vs-
riset-nya tak terjelaskan itu memang mengganggu kualitas sinyal LONG secara spesifik.

### Insiden kecil — classifier blokir overwrite data production
Percobaan pertama sync-join 18-koin salah sasaran ke `data/training/labeled_opt2/` (in-place,
data production) — diblokir otomatis oleh permission classifier sebelum tereksekusi. Dialihkan
ke copy isolated (`pipeline/experiments/takedown_rsz_18coin/`) sesuai konfirmasi user. Tidak ada
data production yang tersentuh.

### Belum
- **Sinkronisasi SSOT dokumentasi**: `model/stacks/fs38_28f/stack.json`, `models/model_registry.json`
  masih bilang production = 21-koin `opt2_plus_trend`, padahal `deploy_model.py::ACTIVE_STACK`
  sudah 18-koin `opt2_plus_trend_18coin` sejak kapan (tidak diketahui persis). Perlu investigasi +
  perbaikan terpisah — bukan cuma soal model ini.
- Deploy `opt2_plus_trend_18coin_iso37f` ke production — audit parity value dulu wajib
  (`audit_feature_value_parity.py`), lalu approval eksplisit terpisah.
- `hmm_regime_enc` OOF-fold masih hilang utk 6 koin baru di `data/training/labeled_opt2/` —
  kalau mau eksperimen OOF-gate lain yg butuh regime akurat per-coin, perlu digenerate dulu
  (bukan cuma fallback state=1).

### Artefak
- LGBM: `models/runs/opt2_plus_trend_18coin_iso38f/`, `models/runs/opt2_plus_trend_18coin_iso37f/`
- Data isolated: `pipeline/experiments/takedown_rsz_18coin/labeled_opt2/` (18 koin, sync fixed)
- M15 entry 6 koin baru: `data/entry_price_m15/{ATOM,BCH,ETC,FIL,LTC,UNI}USDT_15m.parquet` + `_entry_hh15.parquet`
- Script (scratchpad, belum dipindah ke repo): `train_18coin_variants.py`, `compare_oof_18coin.py`,
  `compare_oos_18coin.py`, `fetch_m15_entry_6new.py`

---

## 2026-07-10 (lanjutan lagi) — DEPLOY: ic32_regime_v6.4 (lgbm37f_18coin, takedown relative_strength_z)

**Status**: SELESAI, LIVE. Deploy production pertama hari ini setelah rangkaian audit/fix di atas.

### Yang di-deploy
- LGBM: `opt2_plus_trend_18coin` (38f) → **`opt2_plus_trend_18coin_iso37f`** (37f, takedown `relative_strength_z`)
- Guardian: tetap `guard_opt2_plus_trend_hmm_18coin` (reuse, tidak retrain)
- `ACTIVE_STACK` (`tools/ops/deploy_model.py`) & `models/inference_config.json` diupdate: `model_version` v6.3→**v6.4**, `n_features` 38→37, `model_files.features`, `_snapshot_time`, `scorecard.holdout_oos` (angka 37f terbaru).

### Pre-deploy check
- `audit_feature_value_parity.py --run opt2_plus_trend_18coin_iso37f --label-dir <isolated>`: 36/37 OK, 1 FLAG (`coin_mkt_sync_24h`, std_ratio=0.064). **Diinvestigasi langsung**: ambil 4 snapshot live UNIUSDT terbaru, recompute offline pakai panel 18-koin — **match exact 4/4 titik data** (sampai digit terakhir). FLAG = false-positive artefak sample-size utk fitur market-wide (sama seperti temuan 07-09), bukan bug. Lanjut deploy.

### Insiden saat deploy — git conflict di VPS (RESOLVED)
`deploy_production.py` step scp sukses (model+config ter-upload), tapi step VPS `git pull` GAGAL: `app/jobs/__init__.py` & `app/services/data_service.py` di VPS masih berstatus "uncommitted local changes" (dari fix manual scp pagi ini — taker-volume fix & jadwal HH:15 — yang belum sempat di-commit resmi via git). Akibatnya restart service tertunda, sempat ada window singkat model lama masih jalan meski file baru sudah di disk.
**Resolusi**: diverifikasi dulu (`diff` ignore line-ending) isi VPS working-tree = isi commit yang mau di-pull (0 baris beda) → aman `git checkout -- <2 file>` (buang status uncommitted, BUKAN buang isi fix) → `git pull` sukses fast-forward → restart bersih. Tidak ada fix yang hilang, cuma proses git-nya sempat berantakan karena 2 jalur deploy (scp manual + git resmi) dipakai bergantian hari ini.

### Verifikasi pasca-deploy
- `tools/ops/verify_deployment.py`: **23/38 FAIL** — TERNYATA script ini untuk model lain sama sekali ("TB Widyawardhana v3", 18f, threshold 0.42 dst), bukan ic32_regime_v6.x. Bukan indikasi masalah, cuma tooling stale yang belum digeneralisasi. Diabaikan.
- Verifikasi manual langsung (lebih relevan): load `get_inference_config()` di proses live → `model_version=ic32_regime_v6.4`, `feature_cols=37` (relative_strength_z absen, benar). Jalankan inference sungguhan utk BTCUSDT → `LGBM loaded dari .../lgbm_opt2_plus_trend_18coin_iso37f.pkl`, prediksi jalan tanpa error (FLAT conf=1.0, valid). `active_trades` tidak berkurang (posisi lama aman).

### Belum
- `compare_holdout_live.py` — butuh beberapa jam/hari trade baru dgn model v6.4 dulu sebelum ada yang bisa dibandingkan. Monitor terpisah.
- `tools/ops/verify_deployment.py` & `verify_hmm_feature_parity.py` sebaiknya digeneralisasi/diupdate biar tidak hardcode ke 1 model — di luar scope hari ini.
- Rollback cepat kalau perlu: `ACTIVE_STACK["lgbm"]` balik ke `"opt2_plus_trend_18coin"` di `tools/ops/deploy_model.py`, lalu `deploy_production.py` lagi.

### Artefak
- Backup pre-deploy: `E:\Widyawardhana_Capital\swint_tradev2\models\backups\backup_20260710_212538`
- Commit: `976f349` (swint_tradev2, "deploy: lgbm37f_18coin (takedown relative_strength_z) v6.4")
- `models/inference_config.json` (Riset_pemodelan, sudah di-commit ke swint via deploy)

---

## 2026-07-11/12 — Reproduksi OOS v6.4 vs live 11 Juli: 3 bug ditemukan & difix (2 penuh, 1 parsial)

**Status**: SELESAI utk tujuan reproduksi. 2 bug difix permanen (ada guard/pencegah).
1 bug (indexing spot-confirm) difix di script reproduksi, **BELUM** difix di `core/spottrade_confirm.py`
sumber — butuh keputusan eksplisit krn menyentuh benchmark OOF/OOS spot-confirm yang sudah tercatat.

### Pemicu
User minta reproduksi OOS riset (BUKAN app.db live) utk sinyal 11 Juli 2026, lalu bandingkan
trade-by-trade vs live sungguhan. Awalnya salah pakai universe 21-koin (`config.ALL_COINS`,
stale) + stack `fs38_28f` (v6.1) alih-alih v6.4/18-koin yang sebenarnya live — dikoreksi user
("kenapa ada PEPE... harusnya 18 coin"). Setelah dikoreksi ke universe & stack yang benar
(v6.4 = `opt2_plus_trend_18coin_iso37f` + spot-confirm + regime-disable + HMM 0.65/0.10 +
`guard_opt2_plus_trend_hmm_18coin`, config diambil langsung dari `inference_config.json` VPS),
5-7 trade hasil reproduksi masih tidak cocok persis dengan 4 trade live sungguhan (SOLUSDT,
ADAUSDT, AVAXUSDT, TRXUSDT) — ADAUSDT malah hilang total dari reproduksi.

### Bug 1 — Market-panel corrupt akibat `--coins` subset (FIXED, guard permanen)
`run_engineer.py --coins ATOMUSDT BCHUSDT ETCUSDT FILUSDT LTCUSDT UNIUSDT --holdout-test`
(niatnya cuma nambah data 6 koin baru) diam-diam me-rebuild `_market_panel_h1.parquet`
(file BERSAMA dipakai semua koin) dari cuma 6 koin itu — merusak `mkt_breadth_*`/
`btc_ret_24h`/`btc_minus_mkt_24h`/`coin_mkt_sync_24h` di SEMUA 18 koin. Ini insiden ULANG
dari fix serupa 2026-07-10 (lihat entry 07-10 di atas) — sempat ke-timpa lagi krn agent
tidak baca EXPERIMENTS.md dulu sebelum jalan.
- **Fix**: rebuild panel manual dgn 18 koin lengkap dalam 1 command. Diverifikasi: 4/5
  fitur cocok EXACT sampai presisi mesin setelah fix.
- **Guard permanen**: `pipeline/data/core/engineer.py` — `--coins` (subset) TIDAK lagi
  rebuild panel diam-diam, wajib flag `--rebuild-panel` eksplisit + warning.

### Bug 2 — Spot-confirm panel lokal basi (FIXED, sumber diganti)
`data/spottrade/panel.parquet` (salinan manual dari `Riset_spottrade`, dipakai
`core/spottrade_confirm.py`) berhenti di **6 Juli** — 5+ hari basi. Efeknya: fusion
spot-confirm salah arah (penalty padahal harusnya boost, atau sebaliknya) utk bar-bar
belakangan, mengubah keputusan LONG/SHORT/FLAT.
- Coba salinan lebih baru dari `E:\Widyawardhana_Capital\Riset_spottrade\data\features\panel.parquet`
  (10 Juli 04:00) — lebih baik tapi masih tidak cukup baru.
- **Sumber benar ditemukan**: `live_spottrade` (app terpisah di VPS yang SAMA dgn
  `swint_tradev2`) punya panel sendiri di `/opt/riset/data/features/panel.parquet`,
  di-refresh tiap H4 close (job `pipeline_logs` VPS, jadwal **HH:03**, selesai ~1 menit —
  BUKAN 15-25 menit seperti dokumentasi lama, sudah digeser 2026-07-10). Data sampai
  **11 Juli 12:00 UTC**.
- **Fix**: scp panel VPS → merge (concat + dedup keep-VPS-version di overlap, bukan
  overwrite total) ke `data/spottrade/panel.parquet` — histori 2020-2025 tetap utuh,
  Des2025-Jul2026 sekarang dari sumber VPS asli.
- **Guard permanen**: `core/spottrade_confirm.py::load_spot_panel()` cetak peringatan
  kalau panel > 2 hari basi.

### Bug 3 — Indexing 1 jam salah di `get_spot_confirm_h1` (FIXED di script repro, BELUM di source)
Root cause paling halus. Bar H1 berindeks `open_time=T` (candle `[T, T+1h)`) baru
"selesai"/actionable di `T+1h` (saat candle itu tutup) — tapi semua caller
`get_spot_confirm_h1(coin, df.index)` di repo ini (script reproduksi + kemungkinan
`pipeline/experiments/guardian_reeval_2026-07-08/run_oos_trace_1_7jul_18coin.py` dan
turunannya) query pakai `df.index` (=`T`, jam BUKA) langsung — 1 jam lebih awal dari
kapan keputusan sungguhan terjadi. Akibatnya dapat H4 spot data 1 window (hingga 4 jam)
lebih lama dari yang live pakai.
- **Bukti empiris (TRXUSDT bar 11:00 UTC, 11 Juli)**: query @ jam buka (11:00) → H4 lama →
  spot_short=0.603 (lolos threshold 0.60) → salah kasih PENALTY ke sinyal LONG. Query
  @ jam tutup (12:00, = kapan live benar2 memutuskan) → H4 segar → spot_long=0.530/
  spot_short=0.483 → NETRAL, PERSIS cocok dgn live (`_score_lgbm == _score_final`,
  no adjustment). Diverifikasi juga TIDAK bocor ke H4 masa depan (query jam 14:00
  tetap pakai H4 yang tutup jam 12:00, bukan bucket 12:00-16:00 yang baru tutup 16:00).
- **Fix (di script reproduksi saja)**: `get_spot_confirm_h1(coin, df.index + pd.Timedelta("1h"))`,
  lalu reindex balik ke `df.index`.
- **BELUM difix di `core/spottrade_confirm.py`/`core/cascade_utils.py` sumber** — dampaknya
  lebih luas dari reproduksi ini: script yang sama (`get_spot_confirm_h1(coin, m.index)`
  tanpa shift) dipakai di `pipeline/experiments/guardian_reeval_2026-07-08/*.py`, yaitu
  sumber angka OOF/OOS resmi yang jadi dasar approve fitur spot-confirm & deploy v6.3/v6.4
  (lihat entry 07-08 "KANDIDAT DIKUNCI fs38_18coin_spotconfirm" di atas). Kemungkinan
  angka² itu understate/overstate kontribusi spot-confirm krn pakai window H4 yang salah.
  **Perlu keputusan eksplisit user** sebelum fix di source + re-run benchmark² terkait.

### Hasil akhir reproduksi (setelah Bug 1+2+3 difix di level reproduksi)
4 dari 4 trade live 11 Juli 2026 tereproduksi akurat:

| Coin | Live | Reproduksi | Match |
|------|------|------------|-------|
| SOLUSDT | SHORT 77.4413/76.1177/79.1509 | SHORT 77.4413/76.1177/79.1509 | Persis |
| ADAUSDT | SHORT (trade asli sesi ini) | SHORT — TP/SL identik | Persis |
| AVAXUSDT | LONG 6.7134/6.8330/6.6322 | LONG 6.7134/6.8330/6.6322 | Persis |
| TRXUSDT | LONG 0.3307/0.3335/0.3288 | LONG 0.330665/0.333490/0.328791 | Persis |
| FILUSDT | FLAT (tidak trade) | Sinyal LONG palsu (residual) | **Root cause diketahui, TIDAK bisa difix**: `long_short_ratio` (+80%) & turunan `whale_retail_divergence` — data snapshot real-time Binance, tidak bisa direproduksi retroaktif (lihat [[project-ofi-feature-parity-gap]]). 35/37 fitur lain cocok exact. |

Metodologi verifikasi kunci: feed fitur snapshot LIVE (`feature_snapshot` JSON di tabel
`signal` app.db, ada field debug `_lgbm_proba`/`_score_final`/dll) langsung ke `lgbm.pkl`
lokal → proba identik sampai banyak desimal di semua kasus. Ini membuktikan MODEL selalu
sama; semua gap yang ditemukan murni fitur/data input, bukan versi model beda.

### Tool baru (protokol pencegahan, lihat juga [[feedback-shared-cache-scoped-rebuild]])
- `pipeline/data/core/engineer.py --rebuild-panel` flag (Bug 1)
- `core/spottrade_confirm.py` staleness warning (Bug 2)
- `tools/ops/check_ssot_drift.py` — bandingkan model_type/n_features/coin universe
  live vs `model_registry.json`/`config.ALL_COINS`, jalan di awal sesi

### Belum
- Fix Bug 3 di source (`core/spottrade_confirm.py`/`core/cascade_utils.py`) + re-run
  benchmark OOF/OOS spot-confirm yang terpengaruh — butuh approval eksplisit.
- `model_registry.json` masih v6.1/38f/21-koin, live sudah v6.4/37f/18-koin — SSOT
  belum diupdate (item lama, sudah dicatat 07-10, masih belum dikerjakan).

### Artefak
- `reports/oos_v6_4_trades_2026-07-11_summary.csv`, `_full_features.csv`
- Script reproduksi (scratchpad session, belum dipindah ke repo)

---

## 2026-07-12 — Sweep threshold HMM v6.4 (OOF) + FIX live: SL close-mode pakai harga intrabar, bukan candle settled

**Status**: Sweep SELESAI + divalidasi OOS (kandidat ditolak). Fix live SELESAI & DIDEPLOY.

### Sweep threshold HMM (base/delta) — stack v6.4, OOF genuine walk-forward

Grid base=[0.55..0.80] x delta=[0.00..0.15] (21/24 valid, 3 skip krn threshold <0.50),
18 koin, LGBM `opt2_plus_trend_18coin_iso37f` + spot-confirm (fix indexing 1 jam dari
entry sebelumnya) + regime-disable + Guardian `guard_opt2_plus_trend_hmm_18coin`.
Prasyarat: 6 koin baru (LTC/ATOM/UNI/FIL/ETC/BCH) belum punya `hmm_regime_enc` OOF
(beda dari holdout yg sudah dikerjakan sesi sebelumnya) — dilengkapi via
`run_regime_train.py --coins <6>` + `tools/model/relabel_labeled_opt2.py` (dipanggil
langsung per-coin, BUKAN via `main()`-nya yang hardcode `config.ALL_COINS` 21-koin lama).

Production (0.65/0.10) di OOF: trades=5,415 PF=1.952 PnL=$2,405 — peringkat 15/21 by PF.
Kandidat terbaik by PF: 0.80/0.15 (PF=2.522, tapi cuma 1,487 trade). Full tabel:
`models/runs/guard_opt2_plus_trend_hmm_18coin/hmm_threshold_sweep_v6_4.csv`.

**Validasi OOS (base=0.70/delta=0.10 vs production 0.65/0.10)** — window 2026-04-01 s.d.
11/12 Juli, data `data/holdout-test/labeled`:

| | Production 0.65/0.10 | Kandidat 0.70/0.10 |
|---|---|---|
| Trades | 286 | 171 |
| PF | **1.177** (untung) | **0.887** (RUGI) |
| PnL | +$19.73 | -$8.55 |
| Long-PF | 1.076 | 0.528 (ambruk) |

**Kandidat DITOLAK** — pola klasik overfit OOF: threshold ketat cuma valid di histori
2020-2026, tidak menggeneralisasi ke OOS genuine. **Production 0.65/0.10 dipertahankan.**
Artefak: `models/runs/guard_opt2_plus_trend_hmm_18coin/oos_compare_070_vs_production.json`.

### Fix live — SL trigger_mode="close" pakai harga live intrabar, bukan candle settled (FIXED & DEPLOYED)

Ditemukan saat user tanya "apakah live sudah pakai closed candle utk SL?". Jawaban:
**tidak sepenuhnya**. `app/jobs/check_positions.py` (swint_tradev2) jalan **tiap 5 menit**,
fetch candle H1 "terbaru" dari Binance — kalau dipanggil di tengah jam (hampir selalu),
candle itu **masih berjalan** (belum tutup), `close`-nya = harga live saat itu, bukan close
settle. Padahal komentar kode eksplisit klaim "parity dengan holdout evaluator
(`core/evaluator.py`)" — evaluator riset SL "close" hanya kena kalau candle H1 **benar-benar
tutup** melewati level. Ini root cause tepat kenapa trade ADAUSDT (dibahas dari awal sesi
ini) closed cuma 54 detik setelah harga mendekati SL — bukan 1 jam.

**Fix**: `_fetch_current_candle()` sekarang return `close_settled` (close candle H1 terakhir
yang `close_time` sudah lewat now) TERPISAH dari `close` (harga live, tetap dipakai MFE
tracking & TP-touch — bukan scope fix ini, intrabar by design). `paper_trading.py` pakai
`close_settled` khusus utk cek SL mode "close".

- Commit: `96dbe83` (swint_tradev2) — 2 file saja (`check_positions.py`, `paper_trading.py`)
- Deploy: git push → VPS `git pull` (fast-forward bersih) → `systemctl restart swint-trade`
- Verifikasi: `/api/health` status ok, `app_revision=96dbe83`, tidak ada error di log
- **Efek langsung ke posisi terbuka saat deploy** (SOLUSDT, AVAXUSDT, TRXUSDT +lainnya):
  SL sekarang nunggu candle H1 tutup, bukan reaktif ke harga live tiap 5 menit

### Belum
- Monitor beberapa hari: apakah SL live jadi lebih jarang trigger dini spt insiden ADA,
  dan apakah frekuensi SL-hit turun mendekati pola OOS (bukan pola lama yang lebih reaktif).

---

## 2026-07-12 (lanjutan) — Fix Bug 3 di source (spot-confirm indexing) + auto-refresh panel + config.py disamakan ke 18-koin

**Status**: SELESAI, ketiganya diterapkan.

### Fix Bug 3 — indexing 1 jam di `core/spottrade_confirm.py::get_spot_confirm_h1()` (FIXED di source)

Sebelumnya cuma difix di script reproduksi (lihat entry sebelumnya). Sekarang dipindah ke
SOURCE — `get_spot_confirm_h1()` otomatis geser `h1_index + 1h` sebelum align ke H4, lalu
reindex balik ke `h1_index` asli, jadi **semua caller** (termasuk script lama di
`pipeline/experiments/guardian_reeval_2026-07-08/*.py` yang belum diubah) otomatis dapat
alignment yang benar tanpa perlu diedit satu-satu. Diverifikasi ulang: TRXUSDT bar 11:00 UTC
→ spot_long=0.530/spot_short=0.483 (netral), cocok persis hasil manual sebelumnya.

**Dampak**: nilai fusion spot-confirm SEMUA script yang pakai fungsi ini sekarang berubah
(termasuk kalau ada yang re-run benchmark OOF/OOS spot-confirm lama — angka lama di
`model/stacks/fs38_18coin_spotconfirm/stack.json`/EXPERIMENTS.md 07-08 belum di-refresh,
tidak otomatis berubah krn cuma angka statis tersimpan, bukan re-run).

### Auto-refresh panel spot-confirm dari VPS (tool baru)

`tools/ops/refresh_spot_panel.py` — scp panel `live_spottrade` VPS
(`/opt/riset/data/features/panel.parquet`, sumber PALING segar) → merge (bukan overwrite)
ke `data/spottrade/panel.parquet`, histori lama tetap utuh. Sama pola dgn
`live_db_bridge.py::pull_live_db()`. Diuji: panel dari basi 5,7 hari → basi 0,2 hari sekali
jalan. Jalankan sebelum OOF/OOS yang pakai spot-confirm.

### `config.py` TRAINING_COINS disamakan ke 18-koin live (KEPUTUSAN STRUKTURAL, akhirnya diambil)

Item yang berkali-kali dicatat "belum diambil, perlu approval eksplisit" (07-10, 07-11/12)
— user beri approval eksplisit sesi ini. `TRAINING_COINS`/`ALL_COINS` diubah dari 21-koin
lama ke 18-koin yang PERSIS cocok live: drop GRAMUSDT/1000SHIBUSDT/SUIUSDT/POLUSDT/
1000PEPEUSDT/TAOUSDT/ARBUSDT/HBARUSDT/ONDOUSDT, tambah LTCUSDT/ATOMUSDT/UNIUSDT/FILUSDT/
ETCUSDT/BCHUSDT.

**Verifikasi sebelum ubah**: `symbol_id` (turunan `SYMBOL_MAP`) BUKAN fitur model (cuma
parameter internal `engineer_features()`, tidak ada di 37f/38f) — aman diubah urutannya.
`MEME_COINS`/`NON_MEME_COINS` cuma dipakai 3 script eksperimen lama, tidak crash.

**Verifikasi setelah ubah**: `tools/ops/check_ssot_drift.py` — drift koin **0 isu** (18 vs
18, match persis). Sisa 2 isu drift (model_type v6.1 vs v6.4, n_features 38 vs 37 di
`model_registry.json`) — **belum disentuh**, topik terpisah, belum diminta eksplisit.

**Caveat penting**: stack lama `fs38_28f`/`opt2_plus_trend` (v6.1, rollback path resmi di
`model/stacks/fs38_28f/stack.json`) dilatih di 21-koin lama. Kalau ada yang butuh
evaluasi/rollback ke stack itu, `config.ALL_COINS` default (18-koin) sekarang TIDAK cocok
lagi — perlu override manual/`--coins` eksplisit, jangan andalkan default.

### Artefak
- `core/spottrade_confirm.py::get_spot_confirm_h1()` — fix permanen
- `tools/ops/refresh_spot_panel.py` — tool baru
- `config.py` — `TRAINING_COINS` 18-koin

---

## 2026-07-12 (lanjutan lagi) — Audit trade-by-trade 12 Juli: 4 gap baru ditemukan (3 metodologi, 1 bug dicatat belum fix, 1 kebijakan bukan bug)

**Status**: Semua ditemukan & didokumentasikan. Fix Guardian (item C) **SENGAJA DITUNDA** atas
keputusan user ("jangan dulu, cukup dicatat"). Meme-coin exclusion (item D) **DIKONFIRMASI
TETAP** — bukan bug, jangan disamakan ke riset.

### Latar belakang
Reproduksi OOS 12 Juli 2026 (stack v6.4) dibandingkan trade-by-trade vs live sungguhan (8 trade
live vs 4 di reproduksi awal). Audit mendalam pakai `feature_snapshot`/`entry_reason` live
(app.db) menemukan 4 sumber gap:

### A. Same-bar re-entry gap (BUKAN bug, TIDAK PERLU difix)
`core/evaluator.py::simulate_trades_swing` (mode non-pyramiding, default) menyimpan
`exit_bar+1` sbg bar tercepat boleh entry baru per-koin — kalau sinyal baru persis di bar yang
SAMA dengan exit posisi lama, backtest skip 1 kesempatan; live (cek independen tiap siklus)
tidak punya batasan ini. Terverifikasi 3 kasus (AVAXUSDT, FILUSDT, SOLUSDT, 12 Juli, semua bar
target = exit bar posisi lama persis). Fitur & model 100% cocok di ketiganya (diverifikasi via
feed-live-feature-ke-model-lokal, proba identik) — murni limitasi struktur backtest.
**Keputusan**: JANGAN dipaksa cocok — live yang benar (bisa entry lebih cepat), memaksa live
meniru keterbatasan backtest = sengaja buang kesempatan profit demi kecocokan angka semata.

### B. Meme-coin exclusion di live (BUKAN bug — kebijakan risiko sengaja, DIKONFIRMASI TETAP)
DOGEUSDT hasilkan sinyal SHORT valid (conf 0.72, lolos threshold) di kedua sisi, tapi live
**menolak** via `LIVE_MEME_TRADING=false` (swint_tradev2) — koin meme (DOGE/SHIB/PEPE) sengaja
tidak boleh entry live meski model bilang layak. Repo riset TIDAK punya exclusion setara —
backtest bebas trading DOGE (satu-satunya dari 3 meme coin yang masih ada di universe 18-koin
saat ini; SHIB/PEPE sudah keluar dari 21→18 koin duluan). **Keputusan user**: pertahankan
larangan di live (JANGAN dihidupkan lagi demi "samakan ke riset" — ini risiko modal nyata,
bukan technical mismatch). **Belum dikerjakan**: adjust script OOF/OOS masa depan exclude
DOGEUSDT dari perhitungan kalau mau perbandingan live-vs-riset yang adil (semua angka OOF/OOS
yang sudah dilaporkan sepanjang sesi ini — WR 64.4%/PF 1.369 dkk — sedikit overstate krn ikut
menghitung trade DOGE yang live tidak pernah ambil).

### C. Guardian exit pakai harga live intrabar, BUKAN settled (BUG sejenis SL — DITEMUKAN, BELUM DIFIX)
Root cause ADAUSDT trade pertama (12 Juli) exit lebih cepat di live (~45-60 menit) drpd
reproduksi riset. Kode dicek langsung (`app/services/paper_trading.py`): Guardian exit check
(`self._guardian.check_exit(trade, features_df, close, atr, ...)`), trailing floor
(`guardian_momentum_floor`), dan TP-momentum-activation SEMUA pakai `close` = harga live candle
H1 yang masih berjalan (sama var yang dipakai SL sebelum fix) — BUKAN `close_settled` yang
sudah ditambahkan utk SL (lihat entry sebelumnya, commit `96dbe83`). MFE tracking (high/low)
**TIDAK termasuk masalah ini** — itu sengaja intrabar, sesuai desain `core/evaluator.py` sendiri
(evaluator juga pakai high/low intrabar utk MFE, bukan cuma close).
**Fix yang diperlukan (BELUM dieksekusi)**: ganti `close` → `close_settled` di 3 titik
(`paper_trading.py` sekitar baris 529, 544/547/549, 559/565) — pola sama persis dgn fix SL
`96dbe83`. **Keputusan user**: DITUNDA, cukup dicatat dulu — tidak dideploy sesi ini.

### D. `coin_mkt_sync_24h` zero-filled (item lama, masih berlaku, tidak baru)
Sama seperti sebelumnya — gap pipeline holdout-test lama, tidak spesifik ke tanggal ini.

### Belum
- Fix item C (Guardian close_settled) — tertunda, butuh keputusan eksplisit terpisah kapan mau dikerjakan.
- Adjust script OOF/OOS exclude DOGEUSDT utk perbandingan adil vs live (item B) — belum dikerjakan.
- Re-run angka OOF/OOS resmi (WR/PF yang sudah dilaporkan) dgn DOGE dikeluarkan, kalau user minta.

---

## 2026-07-12 (koreksi) — Item A di atas ("same-bar re-entry gap") SALAH — root cause asli: swing sidedness, bukan gap timing. Fix diuji OOF+OOS, DITOLAK (default TIDAK berubah)

**Status**: SELESAI diinvestigasi. Root cause asli ditemukan & diverifikasi. Fix (`swing_sidedness_check`)
ditambahkan ke `core/evaluator.py` sbg parameter opsional, **default tetap `False`** (perilaku lama)
krn hasil uji OOF+OOS menunjukkan "fix" ini justru MEMPERBURUK performa signifikan.

### Koreksi atas entry "2026-07-12 (lanjutan lagi)" § A
Klaim sebelumnya ("backtest skip 1 kesempatan krn exit_bar sama dgn bar sinyal baru") **SALAH**
— diverifikasi ulang langsung: filter `open_positions = [p for p in open_positions if
p["exit_bar"] > i]` di `core/evaluator.py` SUDAH membuang posisi yg exit persis di bar `i`
SEBELUM cek blocking, jadi re-entry di bar yang sama TIDAK pernah diblokir oleh mekanisme itu.
Dicek langsung dgn print per-bar (AVAXUSDT/FILUSDT/SOLUSDT, 11-12 Juli): tidak ada posisi lain
yang overlap di bar 2434 utk ketiga koin itu — signal SHORT valid ada (`y_pred=SHORT`), tapi
trade tetap tidak muncul di backtest. Artinya blokirnya bukan dari cek pyramiding/gap sama sekali.

### Root cause asli: swing sidedness — `core/evaluator.py` pakai H4 swing walau sudah basi/ditembus harga
Dicek manual RR gate di bar 2434 (AVAXUSDT SHORT): `h4_swing_low=6.648` ternyata **DI ATAS**
`entry=6.546` (harga sudah menembus swing low lama — struktur basi). `core/evaluator.py`
(sebelum fix) tidak validasi sisi ini — `use_swing = not NaN(sh) and not NaN(sl)` saja, jadi swing
basi tetap dipakai apa adanya: `swing_sl = h4_swing_high + 0.5*ATR = 6.859` (SL sangat lebar,
~5.9×ATR) → `rr=0.33` < `min_rr=0.6` → **trade ditolak gate RR**.
`swint_tradev2/app/services/paper_trading.py::_calculate_tp_sl` (live) **sudah** validasi
`sl_lvl < entry and sh > entry` sebelum pakai swing; kalau gagal → fallback ATR murni
(`sl_mult≈1.5×ATR`, jauh lebih ketat) → RR jadi bagus (~1.43) → **live entry**. Threshold gate
sendiri (`max_sl_atr=4`, `min_rr=0.6`, `min_tp_atr=1.2`) **identik** di kedua sisi (dicek
`models/inference_config.json` rr_gate vs `config.py` — sama persis) — bukan config drift.
Diverifikasi FILUSDT & SOLUSDT: pola identik (swing_low > entry, live fallback ATR, riset pakai
swing basi).

### Fix diuji: `swing_sidedness_check` param baru (match validasi live), full OOF+OOS 18-koin
Ditambahkan param opsional `swing_sidedness_check` ke `simulate_trades_swing` — kalau `True`,
`use_swing` juga require `sh_i > price and sl_i < price` (persis logic live). Diuji A/B penuh
(bukan cuma 3 trade) — stack v6.4, 18 koin, spot-confirm+HMM+Guardian:

| Scope | Metric | BEFORE (bug lama) | AFTER (match live) | Delta |
|---|---|---|---|---|
| OOF | trades | 5,415 | 12,579 | **+7,164 (2.3x)** |
| OOF | WR | 66.1% | 59.5% | **-6.6pp** |
| OOF | PF | 1.952 | 1.582 | **-0.37** |
| OOS | trades | 303 | 591 | **+288 (1.95x)** |
| OOS | WR | 59.7% | 53.3% | **-6.4pp** |
| OOS | PF | 1.351 | 1.085 | **-0.27 (nyaris breakeven)** |
| OOS | MaxDD | -$7.50 | -$23.02 | **3.1x lebih buruk** |

Trade TAMBAHAN yang lolos krn fix (OOF n=6,783, OOS n=263) kualitasnya jelas lebih rendah dari
baseline: OOF WR cuma 54.5% (vs 66.1% baseline), OOS malah **net PnL negatif** (-$16.63, WR
46.0%). OOF dan OOS **sepakat** (bukan kasus OOF-menang-OOS-kalah spt kandidat HMM 0.70/0.10
sebelumnya) — sinyal konsisten & sampel besar (bukan n=2 anekdot spt awal investigasi).

### Keputusan: default `core/evaluator.py` TIDAK diubah (tetap `swing_sidedness_check=False`)
Perilaku "lama" (reject saat swing basi, bukan fallback ATR) ternyata **secara tidak sengaja
lebih baik** — bukan bug yang perlu di-"benerin" ke arah live. Param baru disimpan aktif (opt-in)
utk kebutuhan riset "match live literally" ke depan, tapi SSOT benchmark tetap default lama.

### Implikasi baru (BELUM dieksekusi, perlu keputusan terpisah)
Temuan ini membalik arah investigasi: bukan riset yang perlu "dibenerin" ke perilaku live —
justru **live's fallback-ke-ATR saat H4 swing basi berpotensi jadi kelemahan nyata** (persis
konsisten dgn 2 trade live yang closed dari kasus asal investigasi ini: AVAXUSDT -2.75%,
FILUSDT -7.76%, keduanya rugi). Opsi ke depan (belum diputuskan, butuh persetujuan eksplisit
sebelum sentuh live): ubah `paper_trading.py::_calculate_tp_sl` supaya **reject** entry (bukan
fallback ATR) saat swing H4 basi/sudah ditembus — match perilaku riset yang terbukti lebih baik.

### File
- `core/evaluator.py` — param baru `swing_sidedness_check` (default `False`)
- Script uji: scratchpad `swing_sidedness_test.py` (OOF+OOS A/B, 18 koin)
- Hasil: `models/runs/guard_opt2_plus_trend_hmm_18coin/swing_sidedness_fix_before_after.json`

### Belum
- Keputusan eksplisit: mau investigasi/fix live's ATR-fallback-on-stale-swing atau tidak.
- Item C (Guardian close_settled) masih tertunda dari entry sebelumnya, terpisah dari ini.
- Item B (exclude DOGEUSDT dari OOF/OOS utk perbandingan adil) masih belum dikerjakan.

---

## 2026-07-12 (lanjutan) — Root cause kenapa "tolak swing basi" menang: cocok dgn label training model. Varian HMM-conditional diuji & KALAH. Fix DITERAPKAN ke live (lokal, BELUM deploy)

**Status**: Root cause dikonfirmasi. Varian granular (HMM regime-conditional) diuji, ditolak.
Fix diterapkan ke `swint_tradev2` (edit lokal) — **BELUM di-deploy ke VPS**, butuh konfirmasi
terpisah.

### Kenapa "tolak swing basi" menang — bukan kebetulan
`core/features.py::swing_based_labeling` (fungsi label TRAINING model) mensyaratkan
`tp_dist > 0 and sl_dist > 0` — persis sama dgn cek sidedness (`swing_high > price and
swing_low < price`). Kalau gagal & bukan jalur momentum (vol-spike), bar dilabeli **FLAT** saat
training. Jadi model **dilatih** menganggap kondisi swing-basi = bukan setup valid.
`core/evaluator.py` (sebelum fix hari ini, default lama) kebetulan menolak trade di kondisi
sama (via RR gate gagal krn SL lebar) — cocok dgn training. Fallback ATR (spt live) menyodorkan
model ke kondisi yg TIDAK ia pelajari sbg valid — train/inference mismatch, bukan cuma soal
"entry telat/chasing" spt hipotesis awal.

### Varian granular diuji: HMM RANGING→tolak, TRENDING→fallback ATR — KALAH
Sesuai ide "izinkan fallback ATR pas market trending", diuji param baru
`swing_sidedness_regime_arr` + `swing_sidedness_active_states=(TRENDING_DOWN, TRENDING_UP)`.
Hasil OOF+OOS 18-koin (3 arm):

| Scope | Skenario | Trades | WR | PF | PnL | MaxDD |
|---|---|---|---|---|---|---|
| OOF | Tolak semua (baseline) | 5,415 | 66.1% | 1.952 | $2,405 | -$29.41 |
| OOF | Fallback ATR semua (=live) | 12,579 | 59.5% | 1.582 | $3,854 | -$29.92 |
| OOF | **Hybrid HMM (RANGING tolak/TRENDING fallback)** | 9,189 | 61.4% | 1.671 | $3,364 | **-$33.45 (terburuk)** |
| OOS | Tolak semua (baseline) | 303 | 59.7% | 1.351 | $39.00 | -$7.50 |
| OOS | Fallback ATR semua (=live) | 591 | 53.3% | 1.085 | $21.70 | -$23.02 |
| OOS | **Hybrid HMM (RANGING tolak/TRENDING fallback)** | 463 | 55.5% | 1.154 | $30.13 | -$14.75 |

Hybrid kalah dari "tolak semua" di SEMUA metrik OOS, dan MaxDD-nya paling buruk di OOF juga
(meski PnL absolut sedikit lebih tinggi dari baseline). **Keputusan: pakai versi simple (tolak
semua), bukan granular.**

### Fix DITERAPKAN ke live (lokal, BELUM deploy)
`swint_tradev2/app/services/paper_trading.py::_calculate_tp_sl` dan
`app/services/signal_filter.py::_calculate_tp_sl` — ditambah `else` branch: kalau swing H4 ADA
datanya tapi sisi-nya invalid (gagal `sh>entry and sl_lvl<entry`), `return None, None` (sinyal
ditolak) — **bukan** lanjut ke "Fallback ATR murni". Kasus swing benar-benar tidak ada data
(NaN) TIDAK berubah — tetap fallback ATR seperti sebelumnya (skenario beda, tidak diuji &
tidak terbukti bermasalah).
Cek historis live (348 trade closed, difilter 18-koin = 132 trade) sbg pembanding SEBELUM fix
diterapkan menunjukkan hasil ambigu (grup swing-basi n=20 malah sedikit lebih baik dari grup
normal) — tapi sampel kecil & tercampur bug lain yg baru difix (SL, spot-confirm), jadi tidak
dianggap bantahan kuat vs bukti OOF+OOS terkontrol.

**BELUM di-deploy ke VPS** — edit baru di working tree lokal `swint_tradev2`, butuh review +
commit + push + deploy resmi (pola sama spt commit `96dbe83`), MASIH PERLU KONFIRMASI TERPISAH
sebelum dieksekusi.

### Rencana masa depan (BELUM dikerjakan, dicatat saja): retrain dgn label ATR aktif saat trending
User usul: buat model baru dgn label training yang PAKAI ATR (bukan cuma reject) khusus saat
market sedang trending — perluasan konsep jalur momentum yg sudah ada di `swing_based_labeling`
(saat ini dipicu `vol_spike_zscore`, bukan regime HMM eksplisit). Ide: tambah kondisi entry ATR
berbasis regime HMM TRENDING (bukan cuma vol-spike sesaat) sbg jalur label alternatif saat swing
basi, supaya model betulan belajar mengenali & memanfaatkan kondisi ini alih-alih menolaknya
begitu saja. **Ini pekerjaan retraining terpisah, besar, belum dieksekusi** — perlu dibahas
alternatif konkret dulu (sesuai gaya kerja: paparkan opsi sebelum eksekusi) baru jalan.

### File
- `swint_tradev2/app/services/paper_trading.py`, `app/services/signal_filter.py` — fix diterapkan lokal
- `core/features.py::swing_based_labeling` (line ~1146) — bukti sidedness match label training
- Script uji granular: scratchpad `swing_sidedness_hmm_variant.py`, `swing_sidedness_regime_check.py`
- Hasil: `models/runs/guard_opt2_plus_trend_hmm_18coin/swing_sidedness_hmm_variant.json`

### Belum
- ~~Deploy fix live ke VPS~~ **SELESAI** — commit `ea28ee8`, dideploy 2026-07-12.
- ~~Retrain model dgn label ATR-trending-aware~~ **SELESAI** — lihat entry di bawah (`lgbm37f_trend`).
- Item C (Guardian close_settled), Item B (exclude DOGEUSDT dari OOF/OOS) — masih tertunda, tidak berubah.

---

## 2026-07-12 (lanjutan) — Model baru `lgbm37f_trend` (label triple-barrier ATR, khusus TRENDING_UP): dibangun, diuji OOF+OOS, DIDEPLOY LIVE (regime_model_routing)

**Status**: SELESAI & LIVE. Model baru dilatih, dituning, divalidasi OOF+OOS, diintegrasikan ke
`swint_tradev2` dengan routing berbasis regime HMM, dan **diaktifkan di produksi**
(`regime_model_routing.enabled=true`, commit `4938ddb`).

### Latar belakang
Menindaklanjuti temuan swing-sidedness (entry sebelumnya): `core/features.py::swing_based_labeling`
melabeli bar FLAT saat swing H4 basi (kecuali jalur momentum vol-spike). User usul: bikin model
LGBM terpisah dengan label triple-barrier ATR murni, khusus dipakai saat regime HMM trending
(bukan filter vol-spike sesaat), supaya model benar-benar belajar memanfaatkan kondisi swing-basi
alih-alih menolaknya begitu saja.

### Label & training
- `triple_barrier_labeling` (sudah ada di `core/features.py`, tidak perlu ditulis ulang) —
  TP/SL = 2.0×/1.5× ATR (RR 1.33), `max_hold=36` — SAMA persis dgn parameter jalur momentum yang
  sudah dipakai `swing_based_labeling` produksi (`momentum_tp_atr=2.0, momentum_sl_atr=1.5`).
- Label dibuat di FULL continuous series (forward-scan barrier akurat), lalu **difilter** ke bar
  regime HMM TRENDING_DOWN(0)/TRENDING_UP(3) saja utk training set final (bukan re-label,
  murni exclude bar RANGING dari training).
- 18 koin, 326,324 bar trending, class balance jauh lebih seimbang dari label swing produksi:
  LONG 39.6% / SHORT 42.1% / FLAT 18.3%.
- Training: reuse `pipeline/model/experiments/train_lgbm_custom.py::train_run()` (LGBM_PARAMS,
  walk-forward CV `build_rolling_folds`, sama persis pola training model produksi), fitur SAMA
  37f dgn `opt2_plus_trend_18coin_iso37f` (isolasi efek label, bukan feature selection baru).
- Run name: `lgbm37f_trend`, run dir `models/runs/lgbm37f_trend/`.
- **Peringatan metodologi**: `oof_threshold_sweep()` bawaan `train_lgbm.py` pakai
  `simulate_trades_swing` dgn swing ASLI (utk model produksi) — TIDAK VALID utk model ini (dilatih
  triple-barrier ATR, dievaluasi pakai swing basi = mismatch, WR keluar 24.4% yang menyesatkan).
  Eval yang benar: paksa `h4_swing_highs/lows=NaN` supaya `simulate_trades_swing` pakai Tier-2 ATR
  fallback (`TP_SL_FALLBACK_TP/SL` kebetulan = 2.0/1.5, identik param label) — WR jadi 54-61%,
  breakeven cuma 42.9% (1/(1+1.33)), PF 1.75-2.12 di kandidat threshold terbaik OOF standalone.

### Routing & threshold — HMM 0.65±0.10 dipakai utk KEDUA model (bukan flat threshold)
User klarifikasi: HMM harus nentuin MODEL mana DAN threshold dinamisnya sekaligus, pakai base
yang sudah ada (0.65±0.10) — bukan flat threshold baru. `build_regime_thresholds(0.65,0.10)`
sudah punya entry utk state 0/3 (TRENDING_DOWN=(0.75,0.55), TRENDING_UP=(0.55,0.75)) yang
sebelumnya cuma dipakai utk gating model lama; sekarang dipakai jg utk model baru saat regime
trending.

### Iterasi pengujian (OOF+OOS 18-koin, semua diverifikasi bukan cuma OOF)
1. **Full routing (TRENDING_DOWN+UP -> model baru)**, threshold flat 0.55/0.60: OOF positif
   (PnL +7.8%, MaxDD -18%) TAPI **OOS gagal total** — PF turun ke 1.070, MaxDD 2x lebih buruk
   (-$7.03->-$15.15). Breakdown per state ternyata TRENDING_DOWN yang jadi biang rugi (693 trade,
   71% dari total, PF 0.984 OOS) sementara TRENDING_UP justru untung (103 trade, PF 1.332 OOS).
   Market OOS period 62.1% ranging, sisa trending 32.4% TRENDING_DOWN vs cuma 5.6% TRENDING_UP.
2. **Narrow routing (TRENDING_UP SAJA -> model baru)**, threshold per-state 0.65±0.10, Guardian ON,
   spot-confirm OFF (dibuktikan miscalibrated di uji terpisah — lihat di bawah): OOF nyaris flat
   vs baseline (PnL -1.8%, MaxDD membaik 18%), **OOS genuinely positif**: WR 54.8%->56.1%, PF
   1.247->1.303, PnL $28.45->$47.19 (+66%), MaxDD naik tipis (-$7.03->-$8.11, jauh lebih kecil dari
   percobaan #1). Ini kandidat yang di-deploy.
3. **Guardian ON vs OFF** (isolasi trade origin=trend): Guardian ON tetap lebih baik (WR+5.3pp,
   PF+0.053) meski manfaatnya jauh lebih tipis dari porsi model lama (WR+26pp) — bukan mismatch,
   tetap dipakai apa adanya (reuse `guard_opt2_plus_trend_hmm_18coin`, no retrain).
4. **spot-confirm fusion** (agree_boost=0.08/opposite_pen=0.35, dikalibrasi utk model lama):
   MISCALIBRATED utk model baru — distribusi proba model baru jauh lebih sempit (std≈0.066 vs
   model lama), jadi boost tetap 0.08 menggeser ~1.2 std (bukan sedikit), trade meledak 2.4x
   dgn kualitas turun (WR 62.7%->58.4%, PF 1.923->1.616). **spot-confirm di-skip khusus utk model
   baru** di kode live (bukan dihapus dari model lama).

### Kesimpulan penempatan: swing vs ATR bukan soal trending/ranging, tapi ARAH trending
- TRENDING_UP: model ATR unggul (breakout lebih "grinding", target dekat lebih realistis
  drpd nunggu swing jauh yg sudah basi).
- TRENDING_DOWN: model swing (opt2) lebih tahan (structural SL lebih jauh dari lonjakan
  sesaat/relief-rally yg sering terjadi di downtrend crypto, ATR ketat gampang whipsaw).
- Penjelasan ini plausibel & konsisten dgn data tapi BELUM diverifikasi mendalam (blm dicek pola
  spesifik trade rugi TRENDING_DOWN) — dicatat sbg hipotesis kerja, bukan fakta final.

### Deploy ke live (`swint_tradev2`)
- Commit `4938ddb` (kode, routing disabled) -> commit config `enabled:true` (deploy 2 tahap:
  push kode dulu dgn routing OFF, verifikasi sehat, baru flip ON terpisah setelah konfirmasi
  eksplisit ulang).
- `app/services/inference.py`: `_ModelBundle` +`lgbm_trend` slot, `_load_bundle` load model
  opsional (config-gated), `_select_lgbm_for_regime()` pilih model berdasar `hmm_enc` +
  `regime_model_routing.trend_states`, spot-confirm di-skip otomatis saat model trend dipakai.
- `app/jobs/generate_signals.py`: tag "Label=Swing/ATR(trend)" di signal reason.
- `app/templates/trades.html`: badge label basis di halaman trades.
- Model file `models/lgbm_lgbm37f_trend.pkl` (SAMA fitur 37f dgn model utama, reuse
  `feats_lgbm_opt2_plus_trend_18coin_iso37f.json`, tidak perlu file fitur terpisah).
- `models/inference_config.json`: `models.lgbm_trend` path + `regime_model_routing` block
  (`enabled=true`, `trend_states=[3]`).
- **Feature parity audit** (`audit_feature_value_parity.py --run lgbm37f_trend --label-dir
  labeled_trend_tb`): 2 flag (`coin_mkt_sync_24h`, `vwdp`) — SAMA PERSIS dgn 2 dari 4 flag yang
  sudah ada di model produksi yang SEDANG live (`opt2_plus_trend_18coin_iso37f` juga flag
  `ofi_h4_delta`/`ofi_acceleration` tambahan) — bukan parity issue baru, sudah terdokumentasi lama
  (DOGEUSDT extreme values, `coin_mkt_sync_24h` zero-filled).
- **Catatan tooling**: `deploy_production.py`/`deploy_model.py` (tooling resmi) TIDAK dipakai utk
  deploy ini krn `merge_inference_config()` FORCE `models` section ikut `ACTIVE_STACK` (yg belum
  tahu ttg `lgbm_trend`) — akan menghapus config baru ini kalau dijalankan. Deploy manual: git
  push (kode) + scp (model+config) + ssh restart, pola sama dgn fix SL/swing-sidedness sebelumnya.
  **TODO masa depan**: integrasikan `lgbm_trend`/`regime_model_routing` ke `ACTIVE_STACK` resmi
  kalau mau pakai `deploy_production.py` lagi tanpa risiko config ke-overwrite.
- Verifikasi: `/api/health` `app_revision=4938ddb`, `status=ok`, config `regime_model_routing.
  enabled=True` terkonfirmasi terbaca live, 5 posisi terbuka aman, tidak ada error log.

### File
- `pipeline/experiments/guardian_reeval_2026-07-08/data_18coin/labeled_trend_tb/` — label baru
- `models/runs/lgbm37f_trend/` — model, features.json, OOF predictions, semua hasil uji
- Script riset: scratchpad `gen_trend_tb_labels.py`, `train_lgbm37f_trend.py`,
  `eval_lgbm37f_trend_oof.py`, `eval_lgbm37f_trend_scorecard.py`, `routing_combined_oof.py`,
  `routing_combined_tune.py`, `routing_combined_regime_thr.py`, `routing_guardian_onoff.py`,
  `routing_combined_oos.py`, `oos_regime_breakdown.py`, `routing_narrow_trendup_only.py`

### Belum
- Monitoring live pasca-deploy: apakah frekuensi & kualitas trade TRENDING_UP sesuai proyeksi OOS.
- Validasi hipotesis "kenapa ATR unggul di uptrend, swing unggul di downtrend" — belum dicek pola
  trade spesifik.
- Integrasi `lgbm_trend` ke `ACTIVE_STACK`/`deploy_model.py` resmi (saat ini bypass tooling itu).
- Item C (Guardian close_settled), Item B (exclude DOGEUSDT dari OOF/OOS) — masih tertunda, tidak berubah.

---

## 2026-07-12 (lanjutan) — Tool baru `compare_oos_live_signals.py` + fix `long_short_ratio` utk 6 koin baru (window-length-dependent bug)

**Status**: SELESAI & DIDEPLOY. Tool audit per-sinyal per-fitur dibangun, dipakai langsung, nemu
1 bug nyata (bukan di tool doang, tapi genuine parity gap), difix di kedua repo, dideploy.

### Tool baru: `tools/ops/compare_oos_live_signals.py`
Beda dari `compare_holdout_live.py` (agregat WR/PF trade-level, saat ini rusak/import error) —
tool ini bandingkan LIVE vs reproduksi OOS riset **per sinyal, per fitur** (nilai, bukan cuma
ada/tidak — insiden 2026-07-03 `positioning_mode` sudah pernah kena kelas bug yang sama).
Reusable: `--start-date --end-date --coins`. Alur: tarik sinyal live (SSH query langsung ke
`instance/app.db`, bukan sync seluruh DB), reproduksi bar riset dari holdout-test (stack v6.4
penuh termasuk `regime_model_routing`), cocokkan per (koin, waktu), diff tiap fitur.

### Bug #1 (di tool sendiri): off-by-one-hour bar matching
Percobaan pertama: 26,511 baris fitur ter-flag beda, 21 sinyal beda arah — kelihatan seperti gap
sistemik raksasa (btc_ret_1h, mkt_ret_1h, funding_rate, ofi_*, dll, hampir 100% sinyal kena).
Ternyata **93% cuma bug di tool saya sendiri**: bar riset dicocokkan pakai `index <= signal_time`
(waktu OPEN), padahal sinyal live dibuat SETELAH bar CLOSE (+1h) — persis kelas bug yang sama
dengan insiden spot-confirm indexing sebelumnya. Fix: `index <= (signal_time - 1h)`. Setelah fix:
1,787 baris ter-flag (turun 93%), 1 sinyal beda arah dari 342 (99,7% cocok).

### Bug #2 (genuine, di production): `long_short_ratio` fallback synthetic utk 6 koin baru
Dari 1,787 baris tersisa, dipilah mana yang benar-benar dipakai model (37f aktif) vs tidak:
`etf_*`, `funding_rate`, `relative_strength_*` — **TIDAK aktif**, sudah lama di-takedown/tidak
dipakai, tidak berdampak prediksi. Yang aktif & bermasalah: **`long_short_ratio`** dan
**`whale_retail_divergence`** (turunan langsung LSR+CVD) — beda di ~30% sinyal, khusus 6 koin:
LTCUSDT/ATOMUSDT/UNIUSDT/FILUSDT/ETCUSDT/BCHUSDT (yang "6 koin baru", positioning collection baru
mulai 2026-06-30/07-09).

**Root cause**: `core/features.py` (di-deploy 1:1 Riset→swint via `DEPLOY_MAPPING`) — cek
`real_ls.notna().mean() > 0.1` dihitung atas **SELURUH window fetch** (live: 3000 bar/125 hari).
Utk 6 koin ini, data real cuma ~280-300 bar → 300/3000 = **9.4%**, jatuh di BAWAH ambang 10% →
kolom REAL dibuang semua, fallback ke rumus synthetic (`1.0 + vol_delta/vol_ma24`, nilai selalu
~1.0) — padahal data TERBARU (yang dipakai keputusan trading detik ini) 100% valid & sudah
diverifikasi cocok dgn riset (1.4-2.9). Diverifikasi langsung: file
`data/positioning/{coin}_global_ls.parquet` di VPS berisi data asli benar, cuma tidak lolos gate
persentase. Bug ini AKAN TERUS ADA berbulan-bulan (rasio 300/3000 naik pelan-pelan tiap hari,
butuh ~10 bulan utk koin baru "matang" secara alami) — bukan sesuatu yang self-resolve cepat.

**Fix**: cek validitas di window RECENT (300 bar, ~N_BARS_KEEP) bukan seluruh window fetch.
ATOMUSDT: 9.4% → 94.0% setelah fix (tail-300 dari 282 bar data asli = hampir semua valid).
Diterapkan IDENTIK di `Riset_pemodelan/core/features.py` DAN `swint_tradev2/core/features.py`
(harus sinkron, yang kedua di-overwrite dari yang pertama tiap `deploy_model.py` jalan). Tidak
berdampak ke training historis (positioning data utk 6 koin ini baru mulai SETELAH
`TRAIN_CUTOFF_DATE`, jadi label training tetap 100% pakai synthetic scr benar utk periode itu).

**Deploy**: commit `d4d884e` (swint_tradev2), git push + pull + restart (tanpa scp model, cuma
1 file kode). Verifikasi: `/api/health` app_revision=d4d884e, status ok, 5 posisi aman.

### File
- `tools/ops/compare_oos_live_signals.py` (baru, reusable)
- `core/features.py` (kedua repo) — fix `use_real` LSR window
- Laporan: `reports/oos_live_feature_diff_2026-07-12_2026-07-12.csv`,
  `reports/oos_live_unmatched_2026-07-12_2026-07-12.csv`
- Data holdout-test di-refresh manual dgn `--end-date` eksplisit (lihat catatan `OOS_END` di
  `config.py` — bukan otomatis "sampai sekarang", perlu override manual tiap butuh data terbaru)

### Belum
- Cek apakah 6 koin ini pernah entry trade SAAT LSR masih salah (sebelum fix hari ini) — audit
  retrospektif dampak trading real, belum dikerjakan.
- Fitur mati (`etf_*`, `funding_rate`, `relative_strength_*`) — masih dihitung tapi tidak dipakai
  model; boleh dibersihkan dari pipeline kalau mau, tidak urgent (tidak berdampak prediksi).

---
