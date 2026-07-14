# EXPERIMENTS.md — Logbook Riset ic32 Regime v2+

> **Histori penuh diarsipkan** (isi lengkap tiap entry, bukan cuma ringkasan):
> - Pre-2026-06-22: `archive/2026-06-22_cleanup/root_files/EXPERIMENTS_full_history.md`
> - 2026-06-22 s.d. 2026-07-04: `archive/EXPERIMENTS_full_history_2026-07-08.md`
> - 2026-07-08 s.d. 2026-07-12: `archive/EXPERIMENTS_full_history_2026-07-14.md`
>
> File ini cuma **index** (1 baris/eksperimen: tanggal — judul — status — link ke detail).
> Dipecah 2026-07-08 karena versi lama (3269 baris) terlalu boros token untuk dibaca rutin.
> **Entry BARU tulis lengkap di sini dulu** (pakai Template di bagian bawah) — baru dipadatkan
> jadi 1 baris index + dipindah ke archive kalau file ini mulai membengkak lagi.

---

## 2026-07-14 (lanjutan lagi) — Fix: dashboard /models tidak ikut update setelah deploy polos + 2 skrip riset baru (swing-basi rescue)

**Status**: SELESAI, diverifikasi langsung di VPS.

**Pemicu**: user cek `http://139.180.157.176:5000/models` setelah deploy "polos" — dashboard MASIH
menampilkan angka lama (OOF 5.742 trade PF 2.088, OOS 278 trade PF 1.816), padahal config sudah
diupdate. Root cause GANDA:
1. **Salah desain saya sendiri**: angka baru ditambahkan sbg key SIBLING baru (`oof_polos_2026_07_14`
   dkk), bukan menimpa key `scorecard.oof`/`scorecard.holdout_oos` yang ASLI dibaca dashboard
   (`active_model_info.py`). Fix: SWAP — `oof`/`holdout_oos` sekarang isi angka polos (3.450 trade
   PF 2.022 / 161 trade PF 1.680), angka lama dipindah ke `oof_pre_polos_2026_07_08`/
   `holdout_oos_pre_polos_2026_07_08` sbg arsip. Sama di `model_registry.json`.
2. **Bug independen, TIDAK terkait deploy hari ini**: `active_model_info.py` baris ~245 hardcode
   string `"base 0.65/δ0.10"` di badge "Filter Chain" (HMM), bukan baca dinamis dari
   `per_state_thresholds` — sudah salah sejak base naik ke 0.70 kemarin (07-13), baru ketahuan
   sekarang. Fix: derive `base`/`delta` dari `per_state_thresholds["-1"]` & `["0"]`.

**Deploy**: commit swint `0f05c01` (`app/services/active_model_info.py`), scp config. Ditambah ke
`SWINT_NATIVE_CODE` (`tools/ops/deploy_production.py`) supaya ikut ter-track deploy berikutnya.
Verifikasi langsung SSH ke VPS: `scorecard.oof`/`holdout_oos` = 3450/2.022 & 161/1.680, badge HMM
= "base 0.70/δ0.10".

**Pelajaran**: kalau update info scorecard krn deploy config baru, JANGAN tambah key baru di
samping — TIMPA key yang dibaca dashboard, pindahkan angka lama ke key arsip bertanggal. Per
instruksi user: "protokol saat deploy model baru, informasi itu di live juga harusnya ikut
terupdate" — dicatat sbg kebiasaan wajib ke depan.

### Riset baru (belum dites OOS, jangan dianggap final) — "swing-basi rescue"

User tanya: model tanpa regime-routing gimana caranya tetap optimal saat momentum kuat tapi swing
H4 sudah tidak ada/basi? Jawaban: **TIDAK ada mekanisme khusus** — dikonfirmasi baca kode
`paper_trading.py::_calculate_tp_sl`, TP/SL dihitung fungsi yang SAMA utk model utama maupun
`lgbm_trend` (routing cuma ganti model probabilitas, bukan cara TP/SL), jadi celah ini nyata &
tidak tertutup oleh routing sama sekali.

**Ukur skala celah** (`measure_swing_basi_gap.py`, scratchpad): dari 9.697 sinyal lolos threshold,
32,4% (3.146) ditolak krn swing basi — TRENDING_UP paling parah (46,0% dari sinyal state ini,
711 sinyal). 60,6% dari semua yg ditolak-basi ada di state TRENDING (UP+DOWN).

**Ide baru diuji** (`pipeline/model/run_oof_swing_basi_rescue.py`): kalau sinyal ditolak krn swing
basi DI STATE TRENDING_UP, cek `lgbm_trend` di bar yang sama sbg "opini kedua" — kalau setuju arah
sama, izinkan masuk pakai ATR fallback murni. Beda dari fallback-ATR-buta (gagal 07-12) dan routing
penuh (tidak bypass reject-swing-basi).

Sweep ambang konfirmasi `lgbm_trend` (OOF, base=0.70/delta=0.10 utk model utama):

| trend_thr | rescued | trades | WR | PF | PnL | MaxDD |
|---|---|---|---|---|---|---|
| *(baseline, tanpa rescue)* | — | 2.550 | 65,5% | 1,903 | $1.028,85 | -$19,98 |
| 0,60/0,80 (sama ambang state 3) | 24 (3,4%) | 2.570 | 65,6% | 1,919 | $1.055,97 | -$19,98 |
| 0,55/0,75 | 119 (16,7%) | 2.649 | 65,4% | 1,911 | $1.095,23 | -$19,98 |
| 0,50/0,70 | 290 (40,8%) | 2.771 | 65,4% | 1,919 | $1.183,77 | -$19,98 |
| **0,45/0,65** | 433 (60,9%) | 2.877 | 65,5% | **1,987** | $1.340,15 | -$23,69 |
| **0,40/0,60** | 495 (69,6%) | 2.924 | 65,5% | **1,987** | **$1.382,94** | -$23,69 |
| 0,35/0,55 | 509 (71,6%) | 2.935 | 65,3% | 1,977 | $1.382,30 | -$23,69 |

Ambang 0,60/0,80 (sama dgn model utama) TERLALU KETAT (cuma 3,4% lolos, efek nyaris nol). Turunkan
ke 0,40-0,45 -> WR STABIL (tidak rusak sama sekali) sambil PF naik +4,4%, PnL naik ~34%. Sinyal
kuat trade yg diselamatkan memang berkualitas, bukan cuma nambah kuantitas.

**PENTING — baseline 2.550 trade di tabel INI BUKAN angka resmi "polos" yang live** (yg resmi
3.450 trade OOF). Beda krn skrip uji ini pre-filter LEBIH KETAT: tolak eksplisit SEMUA sinyal
swing-basi sebelum simulasi, bukan biarkan mekanisme RR-gate bawaan yg sedikit lebih longgar
menyaring. Perbandingan DI DALAM tabel ini (rescue vs tidak, metodologi sama) tetap valid & adil,
tapi jangan disandingkan dgn angka 3.450 resmi.

### OOS validasi — DITOLAK (OOF vs OOS bertentangan)

Script: `pipeline/model/run_oos_swing_basi_rescue.py`. Sweep trend_thr (0,40/0,60) & (0,45/0,65)
di window OOS genuine (2026-04-01 s.d. 2026-07-13 16:00 UTC):

| | Trade | WR | PF | PnL | MaxDD |
|---|---|---|---|---|---|
| baseline "polos" (tanpa rescue) | 138 | 64,5% | 1,926 | $35,20 | -$6,64 |
| + rescue (0,40/0,60 & 0,45/0,65, identik) | 147 (+9) | 62,6% | **1,734 (-0,192)** | $32,72 (-$2,48) | -$9,02 (lebih buruk) |

**OOF bilang PF naik (+4,4%, +$354), OOS bilang PF turun (-10%).** Sampel OOS sangat kecil (cuma
12 kandidat TRENDING_UP+swing-basi total di seluruh window OOS, 9-10 di-rescue) — tidak cukup utk
jadi bukti kuat "idenya jelek", tapi CUKUP utk melanggar syarat baku proyek ini: OOF dan OOS harus
SEPAKAT dulu sebelum dipertimbangkan lanjut (pola sama persis dgn penolakan HMM fast-react
sebelumnya). **Keputusan: DITOLAK/disimpan dulu**, tidak lanjut ke implementasi kode live.

**Rincian 12 kandidat** (dicek manual atas permintaan user, verifikasi long/short sudah benar
dipertimbangkan): **SEMUA 12 kandidat arahnya LONG — nol kejadian SHORT** di window OOS ini.
Jadi sisi SHORT dari mekanisme rescue ini belum pernah teruji sama sekali secara empiris (bukan
bug — logika `confirmed = (is_long and trend_long) or (is_short and trend_short)` sudah simetris
utk 2 arah, cuma kebetulan tidak ada kejadian SHORT muncul di periode ini). 2 dari 12 kandidat
LONG TIDAK diselamatkan krn `lgbm_trend` justru condong SHORT di bar yg sama (BTCUSDT 2026-06-02,
BCHUSDT 2026-06-15) — konfirmasi logika bekerja benar (menangkap disagreement asli, bukan rubber-stamp).

### Belum
- Kalau nanti window OOS lebih panjang tersedia (lebih banyak kejadian TRENDING_UP+swing-basi,
  termasuk kasus SHORT), boleh diuji ulang. Untuk sekarang datanya tidak cukup mendukung lanjut.

---

## 2026-07-14 (lanjutan) — DEPLOYED ke live: matikan regime-routing + regime-disable + spot-confirm ("polos")

**Status**: LIVE, terverifikasi langsung di VPS. `/api/health` OK, `active_trades` tidak terganggu
(0 posisi terbuka saat deploy).

**Keputusan**: berdasarkan OOF+OOS di entry di bawah (dua uji beda sampel sepakat penuh), matikan
`regime_model_routing`, `regime_disable`, `spot_confirm` — HMM cara baru base=0.70/delta=0.10
TIDAK berubah (sudah benar dari deploy kemarin). Bukan retrain, murni toggle config.

### Yang dikerjakan

1. **Fix blocker sebelum deploy**: `tools/ops/deploy_model.py::PRESERVE_KEYS` sempat berisi
   `regime_model_routing` & `regime_disable` (ditambah kemarin utk kasus BEDA — key hilang total
   dari source). Kalau tidak dicabut, perubahan `enabled=false` hari ini akan DIAM-DIAM ditimpa
   balik ke `true` oleh nilai lama di target saat deploy. Dicabut dari PRESERVE_KEYS (root cause
   asli sudah dibereskan permanen dgn cara lain: block sekarang selalu ada penuh di source Riset).
2. **3 saklar dimatikan** di `models/inference_config.json` (+ note masing-masing dgn alasan &
   angka): `regime_model_routing.enabled`, `regime_disable.enabled`, `spot_confirm.enabled` semua
   `true`→`false`. `lgbm_trend` model TETAP di-deploy (tidak dihapus, siap direaktivasi kalau nanti
   direkalibrasi ulang thd HMM cara baru).
3. **Fix dokumentasi basi**: note HMM di `inference_config.json`/`model_registry.json` masih bilang
   "belum disinkronkan ke live" padahal sudah (commit 4b6ef26, 2026-07-13) — dibetulkan.
4. **Rename display_name**: `fs37_18coin_spotconfirm` → `fs37_18coin_polos` (di `inference_config.json`
   & `model_registry.json`) — cuma label UI (`app/templates/models.html`), tidak ada dependensi kode.
5. **Scorecard OOF/OOS diupdate**: ditambah section baru `oof_polos_2026_07_14` /
   `holdout_oos_polos_2026_07_14` (inference_config) dan `oof_scorecard_polos_2026_07_14` /
   `sealed_holdout_oos_polos_2026_07_14` (model_registry) berisi angka setup yang SEKARANG live.
   Section lama (dgn spot-confirm/regime-disable) DIBIARKAN sbg referensi historis, ditandai
   caveat "sudah tidak mencerminkan setup live sekarang".
6. **Audit sebelum deploy**: `verify_hmm_feature_parity.py` PASS (0 mismatch, 37/37 fitur ada).
   `audit_feature_value_parity.py`: 1 FLAG (`coin_mkt_sync_24h`, std_ratio 0.036) — diinvestigasi
   (lihat di bawah), disimpulkan bukan bug, lanjut deploy.

### Investigasi flag `coin_mkt_sync_24h` (sebelum lanjut deploy)

Fitur = `coin_ret_24h * mkt_ret_24h` (hasil kali, bukan nilai mentah) — kalau market lagi tenang,
hasil kalinya mengecil KUADRATIK, jauh lebih ekstrem drpd fitur input tunggalnya. Buktinya:
- `btc_ret_24h` (salah satu input serupa) std_ratio 0.258 — sudah OK (tidak ter-flag), turun dari
  ter-flag kemarin. `btc_minus_mkt_24h` std_ratio 0.137 — juga sudah OK.
- `coin_mkt_sync_24h` std_ratio 0.036 masih ter-flag krn efek kuadratik dari perkalian 2 fitur yg
  sama-sama sedang kecil.
- Cek 200 sinyal live terakhir (2026-07-13 23:16 - 2026-07-14 10:16 UTC): 200/200 nilai BEDA semua
  (tidak macet di 1 angka), koin ekstrem gonta-ganti tiap jam (FILUSDT/UNIUSDT/ADAUSDT/SOLUSDT...) —
  bukan pola strip/synthetic klasik (beda dari insiden `long_short_ratio` lama).
- 2 dari 3 fitur terkait (`btc_ret_24h`, `btc_minus_mkt_24h`) sudah PULIH ke rentang normal sejak
  kemarin ter-flag bareng — pola pemulihan bertahap ini konsisten dgn "market lagi tenang", BUKAN
  bug yang macet permanen (bug strip akan tetap datar terus, tidak ikut pulih).

**User tanya apakah fitur ini (turunan/perkalian) sebaiknya di-takedown krn "problematik".**
Dicek importance gain di model: **peringkat 17 dari 37 fitur (2.83%)** — kontributor menengah yang
nyata, lebih penting dari 20 fitur lain (whale_retail_divergence, vol_price_confirm, Fib levels,
stochRSI, dst). **Rekomendasi: JANGAN ditakedown** — beda kasus dgn `relative_strength_z` (gap
TAK TERJELASKAN, itu kriteria takedown wajib per [[feedback-feature-source-accountability]]); gap
fitur ini SUDAH terjelaskan penuh (matematis + pola pemulihan), dan fiturnya kontributor nyata,
bukan marginal. Kalau nanti mau ditakedown juga, itu perlu jalur formal (IC→ICIR→Stability→Marginal
→ retrain ablasi), bukan keputusan sekali-jalan — dicatat sbg opsi terbuka, bukan dieksekusi.

### Verifikasi pasca-deploy

```
/api/health: status=ok, scheduler_running=true, active_trades=0 (tidak ada posisi terganggu)
SSH VPS langsung cek inference_config.json: regime_model_routing/regime_disable/spot_confirm
  semua enabled=False, hmm.per_state_thresholds base=0.70/delta=0.10 (tidak berubah), 
  display_name=fs37_18coin_polos, scorecard.oof_polos_2026_07_14/holdout_oos_polos_2026_07_14 ada.
```

Git push: tidak ada perubahan kode (`[INFO] Tidak ada perubahan kode — lewati commit`) — deploy ini
murni config (scp), sesuai ekspektasi (tidak ada retrain/kode baru).

### Belum / follow-up terbuka

- Pantau performa live 1-2 minggu ke depan dgn setup baru ini sebelum evaluasi lanjut.
- `regime_model_routing`/`spot_confirm` bisa direaktivasi nanti KALAU direkalibrasi ulang formal
  thd HMM cara baru (bukan dipakai lagi begitu saja dgn angka kalibrasi lama).
- Takedown `coin_mkt_sync_24h` (kalau mau dipertimbangkan lagi nanti): perlu IC→ICIR→Stability→
  Marginal + retrain ablasi formal, BUKAN keputusan cepat.

---

## 2026-07-14 — OOF: isolasi ulang regime-routing & spot-confirm di HMM cara baru (base 0.65 vs 0.70)

**Status**: OOF selesai (walk-forward genuine, sampel besar 3-6 ribu trade/varian). OOS sampel
kecil (dicatat sesi 07-13/14 sebelumnya) SEPAKAT arahnya. **Belum dieksekusi ke live** — menunggu
keputusan user.

**Pemicu**: lanjutan temuan kemarin (regime-model-routing narik turun performa begitu digabung
HMM cara baru). User minta OOF khusus utk LGBM+HMM+Guardian+spot-confirm (tanpa
regime-routing/regime-disable) di base 0.65 & 0.70, plus cek ulang alasan spot-confirm awalnya
dipakai.

**Script**: `pipeline/model/run_oof_full_stack_sweep.py` — ditambah flag `--no-routing`/
`--no-disable`/`--no-spot`/`--tag` biar reusable per-kombinasi (sebelumnya hardcoded full-stack).

### Hasil OOF (base=0.65 & 0.70, delta=0.10, TANPA regime-routing & TANPA regime-disable di semua baris)

| Varian | Base | Trades | WR | PF | PnL | MaxDD |
|---|---|---|---|---|---|---|
| + spot-confirm | 0.65 | 5,258 | 65.2% | 1.779 | $2,013.15 | -$28.20 |
| + spot-confirm | 0.70 | 3,120 | 66.7% | 1.954 | $1,382.28 | -$25.04 |
| **polos (tanpa spot-confirm)** | 0.65 | 6,349 | 61.4% | 1.810 | $2,543.82 | -$36.22 |
| **polos (tanpa spot-confirm)** | **0.70** | **3,450** | 62.9% | **2.022** | **$1,631.56** | **-$22.97** |

Referensi (regime-routing+regime-disable+spot-confirm SEMUA nyala = versi live sekarang):
- base 0.65: trades=4,803 WR=65.0% PF=1.770 PnL=$1,794.61 MaxDD=-$28.22
- base 0.70: trades=2,755 WR=66.1% PF=1.886 PnL=$1,124.90 MaxDD=-$25.04

**Kesimpulan OOF**: matikan regime-routing+regime-disable saja (spot-confirm tetap nyala) sudah
menaikkan semua metrik vs live sekarang di base yang sama. Matikan spot-confirm JUGA menang PF &
PnL lebih tinggi lagi di base 0.70 (PF 2.022 vs 1.954, PnL $1,631.56 vs $1,382.28) — trade-off:
WR turun (62.9% vs 66.7%) tapi MaxDD malah membaik (-$22.97 vs -$25.04, lebih kecil).

**Cocok dengan OOS** (sampel kecil, `run_oos_hmm_causal_vs_viterbi.py`, ditambah varian D/E/H biar
matriks 2x2-nya sama persis dgn OOF di atas, semua TANPA regime-routing & regime-disable):

| Varian | Base | Trades | WR | PF | PnL | MaxDD |
|---|---|---|---|---|---|---|
| + spot-confirm (E) | 0.65 | 280 | 59.6% | 1.197 | $21.82 | -$11.67 |
| + spot-confirm (H) | 0.70 | 185 | 56.8% | 1.068 | $5.13 | -$8.83 |
| **polos (D)** | 0.65 | 295 | 55.2% | 1.213 | $24.83 | -$14.32 |
| **polos (C)** | **0.70** | **161** | **60.9%** | **1.680** | **$32.96** | **-$6.19** |

Pola SAMA PERSIS dgn OOF: di base 0.70, spot-confirm menyeret PF turun drastis (1.680→1.068,
-36%) — malah lebih parah dari OOF (yang cuma turun ~3.5%, mungkin krn sampel OOS kecil +
window terkini lebih sensitif). Di base 0.65 efeknya kecil & campuran (WR/MaxDD spot-confirm
sedikit lebih baik, PF/PnL sedikit lebih jelek) — konsisten juga dgn OOF.

**polos di base 0.70 (C) adalah SATU-SATUNYA skenario (dari 8 yang diuji OOS) yang menang di
SEMUA metrik sekaligus.** Dua uji beda (OOF sampel besar 3-6 ribu trade, OOS sampel kecil 161-295
trade) SEPAKAT arah: LGBM+HMM(cara baru, base 0.70)+Guardian polos (tanpa regime-routing,
regime-disable, spot-confirm) adalah kombinasi terbaik yang sudah diuji sejauh ini.

### Kenapa spot-confirm dulu disetujui (dicek ulang atas permintaan user)

Disetujui 2026-07-08 (lihat entry "KANDIDAT DIKUNCI fs38_18coin_spotconfirm"): OOF PF naik
1.17→1.33, OOS PF naik 1.05→1.13 — naik konsisten di OOF **dan** OOS, itu jadi dasar approve saat
itu (metodologi sah). TAPI: benchmark itu pakai HMM cara lama + threshold 0.65/0.05, dan baru
belakangan (2026-07-11/12) ketahuan spot-confirm punya bug indexing 1 jam (fixed 2026-07-12) —
angka approval ASLI itu **tidak pernah dihitung ulang** pakai kode yang sudah difix. Jadi
keputusan awal metodologinya sah waktu itu, tapi angkanya sendiri berpotensi bias & belum
diverifikasi ulang sejak fix bug + sejak ganti ke HMM cara baru.

**Follow-up terbuka**: spot-confirm butuh audit ulang formal (OOF+OOS, HMM cara baru, kode sudah
difix) sebelum bisa disimpulkan "masih worth it" atau tidak — data hari ini justru condong ke
arah TIDAK menambah nilai lagi di base 0.70 (lihat tabel di atas, per [[feedback-feature-source-accountability]]).

### Belum
- Keputusan final: live matikan regime-routing+regime-disable+spot-confirm semua (full-polos)
  ATAU cuma regime-routing+regime-disable (tetap pakai spot-confirm) — user belum konfirmasi.
- Kalau full-polos dipilih: perlu regenerasi `inference_config.json`/live (matikan 3 block
  config) + audit parity fitur sebelum deploy, prosedur sama seperti deploy HMM kemarin.
- Artefak: `models/runs/guard_opt2_plus_trend_hmm_18coin/oof_full_stack_sweep_lgbm_hmm_guard_spot.json`,
  `...oof_full_stack_sweep_lgbm_hmm_guard_only.json`.

---

## 2026-07-13 (lanjutan lagi lagi lagi) — DEPLOYED ke live: HMM causal filtering + base 0.70/delta 0.10

**Status**: LIVE, terverifikasi. VPS `app_revision=4b6ef26`, health OK, feature check 18/18 koin OK,
tidak ada error pasca-restart, `active_trades` lama tidak terganggu.

### Ditemukan sebelum deploy — 2 isu tambahan (ditangani sebelum lanjut)

1. **`coin_mkt_sync_24h` (dan sync-cols lain) hilang total di holdout-test** — ternyata ini
   memang gap lama yang terdokumentasi (`pipeline/experiments/join_sync_features_holdout.py`,
   root cause: `tools/ops/join_sync_features_incremental.py` cuma nge-patch training, tidak
   pernah dijalankan utk holdout). Refresh data holdout-test pagi ini (fetch/clean/engineer)
   TIDAK otomatis include langkah ini, jadi fitur ini diam-diam nol di SEMUA simulasi OOS sesi
   ini sebelum ketahuan. **Fix**: jalankan `join_sync_features_holdout.py` (18/18 koin OK).
   **Dampak**: baseline OOS produksi yang benar ternyata LEBIH BAGUS dari yang sempat dilaporkan
   (PF 1.373 PnL $41.48, bukan PF 1.262 PnL $30.34) -- semua perbandingan causal 0.65-0.75 di atas
   perlu dibaca dgn baseline yang sudah dikoreksi ini (lihat tabel final di bawah).
2. **3 fitur BTC/market-wide ter-flag** di `audit_feature_value_parity.py` (`btc_ret_24h`,
   `btc_minus_mkt_24h`, `coin_mkt_sync_24h`, std_ratio live/train < 0.1). Investigasi: pola nilai
   live (range sempit, tidak macet di angka bulat) konsisten dgn "pasar memang lagi tenang di
   200 sinyal terakhir", BUKAN pola strip/synthetic klasik (beda dari kasus `long_short_ratio`
   lama). Fitur ini TIDAK disentuh oleh perubahan HMM sesi ini. **Keputusan user**: lanjut deploy
   HMM-only, fitur ini diinvestigasi terpisah nanti (BELUM selesai, follow-up terbuka).

### Scorecard final (dgn data lengkap, sync-fix + leak-fix, apples-to-apples)

| | trades | WR | PF | PnL | MaxDD |
|--|--------|-----|-----|-----|-------|
| Viterbi 0.65/0.10 (lama, data lengkap) | 303 | 56.4% | 1.373 | $41.48 | -$9.56 |
| **Causal 0.70/0.10 (DEPLOYED)** | **159 (-47.5%)** | **61.0% (+4.6pp)** | **1.703 (+24.0%)** | **$33.27 (-19.8%)** | **-$5.60 (41.4% lebih baik)** |

**Baca jujur**: ini trade-off nyata, BUKAN "menang telak" seperti sempat disangka di iterasi
sebelumnya (sebelum sync-fix). User pilih trade less-often-but-better-quality (PF/WR/MaxDD
semua naik signifikan) meski PnL total turun ~20% di window OOS ini (159 trade, sampel sedang).

### Perubahan yang di-deploy

1. **Riset** (`Riset_pemodelan`, commit lokal belum di-push ke git Riset -- reminder: commit
   terpisah dari deploy production):
   - `core/regime.py`: `predict_hmm_causal()` baru (forward filtering) + fix bocor `fillna(ret.std())`
     -> `fillna(0.0)`. `generate_oof_regime_labels()` pakai causal di tiap fold.
   - `pipeline/data/core/regime_hmm_holdout.py` (SSOT holdout): `predict_hmm` -> `predict_hmm_causal`.
   - `pipeline/data/core/regime_hmm.py`: blok holdout vestigial dikasih peringatan (bukan SSOT).
   - `tools/model/verify_hmm_feature_parity.py`: diupdate cek causal (bukan Viterbi) + feature
     list diupdate ke `opt2_plus_trend_18coin_iso37f` (37f, sebelumnya rujuk model lama 38f).
   - `data/holdout-test/labeled/*_regime_h1.parquet` + `models/hmm/*.pkl`: diregenerasi 18 koin.
   - `models/inference_config.json`, `models/model_registry.json`: base 0.65->0.70, didokumentasikan.
2. **Live** (`swint_tradev2`, commit `4b6ef26`, pushed ke `main`, VPS git pull sukses):
   - `app/services/data_service.py::_compute_hmm_regime()`: `predict_hmm` -> `predict_hmm_causal`.
   - `core/regime.py`, `core/features.py`: sync 1:1 dari Riset.
   - `models/inference_config.json` (scp): per_state_thresholds base 0.70/delta 0.10.
   - `models/hmm/*.pkl` (scp, 21 file termasuk legacy coins): isi sama (fit tidak berubah).
   - Backup otomatis tersimpan: `models/backups/backup_20260713_191636/`.

### Verifikasi pasca-deploy

```
app_revision: 4b6ef26
status: ok, scheduler_running: true
feature_monitor: {'total': 18, 'ok': 18, 'warning': 0, 'error': 0}
active_trades: 1 (posisi lama tidak terganggu)
```

### Belum / follow-up terbuka

- 3 fitur BTC/market-wide ter-flag (lihat di atas) -- investigasi terpisah, bukan blocker HMM.
- Commit perubahan `core/regime.py` dkk ke git Riset_pemodelan (belum dilakukan sesi ini, cuma
  deploy production yg jalan -- riset masih uncommitted working tree per kebiasaan sesi ini).
- Pantau performa live 1-2 minggu ke depan sebelum menilai apakah trade-off (PF naik, PnL turun)
  ini benar-benar sepadan di kondisi live sungguhan (sampel OOS 159 trade tergolong sedang).

---

## 2026-07-13 (lanjutan lagi lagi) — DIPUTUSKAN: base 0.65->0.70 + causal filtering, DITERAPKAN ke SSOT riset (BELUM live)

**Status**: DITERAPKAN di riset (SSOT + config), BELUM disinkron ke live/VPS -- itu keputusan
terpisah, tunggu approval eksplisit lanjutan.

User pilih kandidat **causal 0.70/0.10** dari sweep di atas sbg keputusan final.

### Audit tambahan sebelum menerapkan (diminta user: "cek ulang apakah sudah tidak leakage")

Ditemukan 1 bocor kecil LAGI di `core/regime.py::_build_hmm_features()`:
`vol = ret.rolling(vol_window,...).std().fillna(ret.std())` -- fallback NaN pakai std SELURUH
series yang dipass (termasuk observasi sesudah bar yang ditambal). Cuma ~5 bar pertama tiap
window kena (porsi kecil), tapi tetap leakage. Fix: `.fillna(0.0)`, konsisten dgn `mom`/`vr` yang
sudah begitu. **Angka berubah stlh fix**: causal_0.70_0.10 PnL turun dari $35.15 ke $30.43 (dari
"menang di semua metrik" jadi "untung nyaris sama, tapi lebih efisien & risiko sedikit lebih
kecil") -- bukti bocor kecil pun berpengaruh nyata ke kesimpulan, bukan cuma formalitas.

### Perubahan yang DITERAPKAN (repo riset)

1. `pipeline/data/core/regime_hmm_holdout.py` (SSOT holdout/live) -- ganti `predict_hmm`
   (Viterbi) jadi `predict_hmm_causal`. `pipeline/data/core/regime_hmm.py` -- blok holdout
   vestigial (bukan SSOT, beda 2 hal dari SSOT: Viterbi + pakai btc_h4) dikasih peringatan
   jangan dipakai, tidak dihapus (backward-compat).
2. Regenerasi `data/holdout-test/labeled/{coin}_regime_h1.parquet` + `models/hmm/{coin}_hmm.pkl`
   utk 18 koin (`run_regime_holdout.py --all`) -- fit model IDENTIK (belum berubah), cuma cara
   decode/predict yang beda. Verifikasi: DOTUSDT jam 07-12 05:00-23:00 (dulu 19/19 beda vs live)
   sekarang cocok 15/19 dgn histori live -- bukti perbaikan nyata, meski belum sempurna.
3. `models/inference_config.json` `hmm.per_state_thresholds` -- base 0.65->0.70 (delta tetap
   0.10): state0 [0.80,0.60], state1 [0.75,0.65], state2 [0.65,0.75], state3 [0.60,0.80],
   fallback [0.70,0.70]. Note lengkap ditambahkan menjelaskan alasan + status sinkron live.
4. `models/model_registry.json` `stack.hmm` -- tambah field `base`, `decode_method`,
   `decode_method_note`.

### Angka final (OOS genuine, setelah fix bocor kedua)

| | trades | WR | PF | PnL | MaxDD |
|--|--------|-----|-----|-----|-------|
| Viterbi 0.65/0.10 (lama) | 315 | 55.9% | 1.262 | $30.34 | -$6.93 |
| **Causal 0.70/0.10 (baru)** | **169 (-46%)** | **59.2% (+3.3pp)** | **1.572 (+25%)** | **$30.43 (nyaris sama)** | **-$6.38 (8% lebih baik)** |

### BELUM dilakukan -- perlu keputusan terpisah

- **`swint_tradev2/app/services/data_service.py::_compute_hmm_regime()`** MASIH import & pakai
  `predict_hmm` (Viterbi) langsung, base masih 0.65 (di config live VPS, terpisah dari
  `models/inference_config.json` riset). Live TIDAK ikut berubah dari sesi ini.
- Deploy resmi (`deploy_production.py`, scp model+config ke VPS, restart) belum dijalankan --
  butuh audit parity fitur (`audit_feature_value_parity.py`) + approval eksplisit terpisah
  sebelum sentuh live, sesuai `tools/ops/CLAUDE.md`.
- Sample OOS 169 trade tergolong sedang (bukan besar) -- worth dipantau lagi setelah data
  makin panjang sebelum terlalu yakin.

---

## 2026-07-13 (lanjutan lagi) — Re-tuning threshold HMM pakai regime causal — TIDAK ADA kandidat yang menang telak, keputusan trade-off

**Status**: SELESAI. Tidak ada kandidat yang unggul di SEMUA metrik -- keputusan produksi diserahkan
ke user (belum diubah).

Follow-up dari straight-swap yang ditolak (entry di atas). User setuju lanjut: threshold
`hmm.per_state_thresholds` disetel ULANG pakai regime causal (bukan Viterbi bocor), bukan cuma
tempel cara baru ke pengaturan lama.

### Sweep OOF (leak-free, causal filtering di tiap fold walk-forward)

**Script baru**: `pipeline/model/run_oof_hmm_threshold_retune_causal.py`. `core/regime.py`
`generate_oof_regime_labels()` diubah pakai `predict_hmm_causal()` (bukan `predict_hmm`
Viterbi) di tiap fold -- OOF regime encoding sekarang juga leak-free, bukan cuma di sisi holdout.
Grid base 0.55-0.80 x delta 0-0.15 (24 kombinasi, sama seperti sweep legacy).

Topby PF: base=0.80/delta=0.00 (PF 2.80, tapi cuma 301 trade/6th/18koin -- sangat jarang),
base=0.70/delta=0.05 (PF 2.094, 2469 trade), base=0.65/delta=0.05 (PF 1.872, 4984 trade).
**Current prod (base=0.65/delta=0.10) dengan regime causal**: PF 1.799, trades 6364 (baseline
pembanding, BUKAN yang tertinggi PF-nya -- pola sama seperti sweep original 2026-07-08: pilihan
produksi selalu bukan PF tertinggi OOF, karena PF tertinggi = trade paling sedikit/rapuh).

### Validasi OOS (data belum pernah dilihat training, 2026-04-01 s.d. 07-12/13, 315 trade baseline)

**Script**: `pipeline/model/run_oos_hmm_causal_vs_viterbi.py` (extended, multi-kandidat).

| Kandidat | trades | WR | PF | PnL | MaxDD |
|---|---|---|---|---|---|
| viterbi 0.65/0.10 (**prod saat ini**) | 315 | 55.9% | 1.262 | $30.34 | -$6.93 |
| causal 0.65/0.10 (regime baru, threshold lama) | 306 (-2.9%) | 53.9% | 1.107 (-12.3%) | $13.07 (-57%) | -$12.52 (+81% lbh buruk) |
| causal 0.65/0.05 | 225 (-28.6%) | 56.9% | 1.324 (+4.9%) | $26.15 (-14%) | -$11.89 (+72% lbh buruk) |
| causal 0.70/0.05 | 122 (-61.3%) | 58.2% | 1.560 (+23.6%) | $23.62 (-22%) | -$7.51 (+8% lbh buruk) |
| causal 0.75/0.05 | 55 (-82.5%) | 61.8% | 1.790 (+41.8%) | $15.47 (-49%) | **-$5.04 (27% LEBIH BAIK)** |

**Baca**: pola jelas, bukan bug -- makin tinggi threshold (makin selektif), PF & WR makin bagus
dan bahkan MaxDD ikut membaik di 0.75/0.05 (lebih baik dari prod!), TAPI jumlah trade jatuh
drastis (sampai -82.5%) dan PnL total tetap lebih rendah dari prod di SEMUA kandidat -- lebih
selektif = lebih jarang trading = untung total lebih kecil meski kualitas per-trade naik.
**Tidak ada kandidat yang menang di SEMUA metrik sekaligus** dibanding prod. Sampel OOS juga
kecil (315 turun ke 55 utk kandidat paling agresif) -- beda-beda ini indikatif, bukan konklusif.

### Kesimpulan & rekomendasi

1. **Regime Viterbi yang bocor (lookahead) MEMANG bikin angka OOF/OOS historis proyek ini sedikit
   digelembungkan** -- causal filtering (jujur, leak-free) adalah cara evaluasi yang lebih benar
   secara metodologi ke depan, terlepas dari keputusan threshold.
2. **TAPI tidak ada bukti kuat utk ganti konfigurasi produksi** -- semua kandidat causal
   menang di kualitas (PF/WR) tapi kalah di volume/total profit vs setup existing. Trade-off,
   bukan free upgrade.
3. **Rekomendasi**: TETAP pakai threshold+regime production existing (Viterbi 0.65/0.10) untuk
   saat ini -- tidak cukup bukti utk pindah. `predict_hmm_causal()` disimpan di kode sbg metode
   yang tersedia & lebih benar utk EVALUASI/backtest riset ke depan (hindari overclaim PF dari
   lookahead), tapi TIDAK dipakai SSOT/live saat ini. Keputusan akhir threshold produksi
   diserahkan ke user -- belum ada perubahan ke `models/inference_config.json`/deploy.

Artefak: `models/runs/guard_opt2_plus_trend_hmm_18coin/hmm_threshold_sweep_causal_oof.csv`,
`oos_hmm_causal_vs_viterbi.json`.

---

## 2026-07-13 (lanjutan) — Implementasi HMM causal filtering (ganti Viterbi) — DITOLAK straight-swap, OOS turun

**Status**: KODE DIBUAT & TERVALIDASI BENAR, TAPI PERFORMA OOS TURUN — jangan deploy tanpa re-tuning.
User minta tindak lanjuti temuan `hmm_regime_enc` mismatch (entry di atas) dengan mengganti metode
prediksi dari Viterbi ke causal filtering.

### Implementasi
`core/regime.py` — fungsi baru `predict_hmm_causal()` (forward filtering murni via
`hmmlearn._hmmc.forward_log`, argmax alpha_t) sbg pengganti `predict_hmm()` (Viterbi,
`model.predict()`) UNTUK use-case holdout/live (window terus bertambah panjang). `predict_hmm()`
lama TIDAK dihapus — masih dipakai `generate_oof_regime_labels()` (tiap fold window TETAP,
tidak pernah diperpanjang, jadi Viterbi di situ sudah stabil/aman).

### Validasi kebenaran (sebelum lihat dampak performa)
1. **Stabilitas** (properti inti yang dicari): predict causal atas prefix N vs prefix N+20 —
   hasil utk 0..N-1 IDENTIK 100% (dites DOTUSDT). Viterbi tidak punya jaminan ini.
2. **Sanity distribusi**: causal vs Viterbi state distribution DOTUSDT masuk akal (TRENDING_DOWN
   53.7% vs 49.7%, tidak degenerate).
3. **Magnitude vs Viterbi** (18 koin, window OOS Apr-Jul): causal beda 1.1%-13.1% dari Viterbi per
   koin (rata2 ~8%) — BUKAN cuma di 3 koin yang ke-flag di audit (DOT/BCH/ADA), across-the-board.
   Artinya switch ini reklasifikasi state utk sebagian besar sejarah, bukan cuma nge-fix 3 koin.

### Dampak performa — OOS (18-koin, stack v6.4, H1-close entry, 2026-04-01 s.d. 2026-07-13)

**Script**: `pipeline/model/run_oos_hmm_causal_vs_viterbi.py` (tidak overwrite artefak resmi
`models/hmm/*.pkl`/`regime_h1.parquet` — regime causal dibangun in-memory utk perbandingan).

| | trades | WR | PF | PnL | MaxDD |
|--|--------|-----|-----|-----|-------|
| Viterbi (current) | 315 | 55.9% | **1.262** | $30.34 | -$6.93 |
| Causal (baru) | 306 | 53.9% | 1.107 (-12.3%) | $13.07 (-57%) | -$12.52 (+81% lebih buruk) |

**Baca**: causal filtering LEBIH BENAR secara matematis (stabil, tidak pernah lihat masa depan)
TAPI performanya lebih jelek kalau langsung tukar tanpa re-tuning. Penyebab paling mungkin:
`hmm.per_state_thresholds` (base=0.65, delta=0.10) dikalibrasi lewat sweep OOF pakai regime
Viterbi -- begitu state encoding berubah (rata2 ~8% bar reklasifikasi), threshold yang sama
tidak lagi cocok utk distribusi state yang baru. Bukan berarti causal "salah", tapi straight-swap
tanpa re-tuning threshold jelas merugikan.

### Keputusan
**JANGAN deploy causal filtering apa adanya** — turun 12% PF, 57% PnL, MaxDD 81% lebih dalam di
OOS. Kalau mau dikejar lebih jauh, perlu sweep ulang `hmm.per_state_thresholds` di OOF (bukan
OOS) dgn regime causal, baru validasi OOS lagi -- effort besar & hasil belum tentu lebih baik dari
Viterbi. Belum diputuskan/dieksekusi, tunggu keputusan user. `predict_hmm_causal()` disimpan di
`core/regime.py` sbg opsi yang tersedia utk eksperimen lanjutan kapanpun, tidak dipakai SSOT saat ini.

---

## 2026-07-13 — Audit sinyal live vs riset (refresh OOS + compare_oos_live_signals.py, 11-13 Juli)

**Status**: SELESAI. **Koreksi penting atas laporan awal saya sendiri** — root cause 1 ternyata
sudah fixed (bukan bug baru), root cause 2 genuinely ditemukan & dijelaskan (bukan bug kode,
karakteristik struktural HMM Viterbi).

### Alur
1. Refresh `data/holdout-test/` penuh (fetch→clean→engineer→regime, `--all`, s.d. 2026-07-13) —
   sebelumnya mentok 2026-07-12 11:00 UTC.
2. `tools/ops/compare_oos_live_signals.py --start-date 2026-07-11 --end-date 2026-07-13`: 1022
   sinyal live, 5379 baris fitur ter-flag (>5% beda), 5 sinyal beda arah, 0 unmatched.

### Temuan 1 — `long_short_ratio`/`whale_retail_divergence` (6 koin baru): SUDAH FIXED, BUKAN bug aktif

Laporan awal saya (ke user) SALAH menyimpulkan fix `d4d884e` (2026-07-12) belum tuntas. Setelah
cek **distribusi waktu** baris ter-flag: SEMUA 260 baris `long_short_ratio` + 255 baris
`whale_retail_divergence` (ATOM/ETC/BCH/LTC/FIL/UNI) berhenti tepat di **2026-07-12 11:16 UTC**
— persis setelah commit `d4d884e` (2026-07-12 11:12 UTC). Nol baris ter-flag sesudah itu.
Diverifikasi langsung: jalankan live code path (`InferenceDataService.prepare_latest_features`)
di VPS via SSH utk ATOMUSDT sekarang → `last_LSR=1.6497` (real, bukan synthetic ~1.0).
**Pelajaran metodologi**: kalau nemu gap, cek dulu distribusi WAKTU vs tanggal deploy fix
terakhir sebelum simpulkan "belum fixed" — kalau tidak, gampang salah lapor false-positive
dari data historis pra-fix yang kebetulan masuk rentang tanggal yang dipilih.

### Temuan 2 — `hmm_regime_enc` mismatch (DOTUSDT 19 jam berturut-turut, BCH 7 jam, ADA 7 jam): STRUKTURAL, bukan bug kode

Model `.pkl` byte-identical live vs riset (md5 match). Harga H4 juga cocok (close riset vs live
selisih <0.5%). Root cause: `core/regime.py::predict_hmm()` pakai `model.predict(X)` (hmmlearn
GaussianHMM) = **Viterbi decoding atas SELURUH sequence sekaligus** (global, bukan per-bar
independen). Konsekuensi: state yang di-assign ke bar LAMA bisa BERUBAH kalau sequence
diperpanjang dengan observasi BARU di re-decode berikutnya — sifat inheren Viterbi, bukan bug.
Live decode terus-menerus dgn data s.d. "sekarang" (real-time); riset decode dgn cutoff snapshot
refresh saya (s.d. 2026-07-13 00:00-01:00) — beda titik "as-of" ini cukup utk menggeser state
bar2 dekat batas waktu retroaktif, meski observasi H4 mentahnya sama persis di kedua sisi.

**Bukan bug yang bisa di-"fix" sederhana** — dua opsi kalau mau dibereskan struktural:
1. **Terima sebagai noise wajar** dekat batas real-time (Recommended, effort rendah) — cocok
   dgn fakta 5 sinyal beda-arah cuma 0.5% dari 1022 sinyal, dan HMM state cuma dipakai utk
   threshold gating (bukan input LGBM 37f langsung).
2. Ganti metodologi predict ke **filtering causal murni** (bukan Viterbi batch) supaya state
   bar lama stabil selamanya — butuh retrain-adjacent validation (OOF/OOS ulang), scope besar,
   BELUM diputuskan/dieksekusi, tunggu approval eksplisit terpisah.

### Keputusan
Tidak ada kode yang diubah sesi ini (temuan 1 sudah fixed sendiri sebelum sesi ini; temuan 2
strukturaL, butuh keputusan user dulu). Direkomendasikan: opsi 1 (terima sbg noise near-boundary),
dicatat di sini sbg baseline pemahaman utk audit berikutnya — jangan panik kalau `hmm_regime_enc`
beda di beberapa jam terakhir sebelum cutoff reproduksi riset, itu ekspektasi Viterbi bukan bug.

---

## 2026-07-13 — OOF: pyramiding max2 (scale_in) dengan gate WAKTU antar-leg (1/2/3/5/7/9 jam)

**Status**: OOF COMPLETE (not deployed). **Kesimpulan: TIDAK cukup kuat, pyramiding TETAP CLOSED.**
**Stack**: lgbm37f_18coin (`opt2_plus_trend_18coin_iso37f`) + HMM 0.65/0.10 + guard28f_18coin (`guard_opt2_plus_trend_hmm_18coin`, reuse), tanpa spot_confirm/regime_disable (fusion live-only).
**Script baru**: `pipeline/model/run_oof_pyramiding_time_gap_sweep.py` (sweep `--gaps`).
**Kode baru**: `core/scale_in_sim.py` param `pyramiding_min_bars_gap` (jeda minimum sejak leg TERAKHIR, bukan sejak entry awal) + thread ke `core/evaluator.py::simulate_trades_swing`.
**Artifact**: `models/runs/guard_opt2_plus_trend_hmm_18coin/pyramiding_time_gap_sweep_oof.json`

Konteks: pyramiding max2 dgn gate HARGA (rugi>=2.5%) sudah diuji tuntas & ditolak (PF 1.952->1.745
tanpa gate, ~1.72 dgn gate). Eksperimen ini sudut BEDA: gate WAKTU (jam) minimum antar-leg, bukan harga.

### Full OOF (2020-01-01 s.d. 2026-04-01, 5330 trade baseline)

| | trades | WR | PF | PnL | MaxDD |
|--|--------|-----|-----|-----|-------|
| A baseline (no pyramiding) | 5330 | 63.1% | **1.946** | $2356 | -$33.02 |
| gap 1h | 5321 | 63.0% | 1.830 (-6.0%) | $2971 (+26%) | -$46.05 (+40% lebih buruk) |
| gap 2h | 5326 | 62.9% | 1.834 (-5.8%) | $2831 (+20%) | -$42.84 (+30% lebih buruk) |
| gap 3h | 5331 | 63.0% | 1.815 (-6.7%) | $2686 (+14%) | -$45.95 (+39% lebih buruk) |
| gap 5h | 5331 | 63.2% | 1.802 (-7.4%) | $2503 (+6%) | -$42.66 (+29% lebih buruk) |
| gap 7h | 5333 | 63.2% | 1.766 (-9.2%) | $2315 (-2%) | -$44.21 (+34% lebih buruk) |
| gap 9h | 5340 | 63.4% | 1.767 (-9.2%) | $2247 (-5%) | -$43.56 (+32% lebih buruk) |

**Baca**: di sampel besar (paling bisa dipercaya), gate waktu berapapun panjangnya TIDAK
menyelamatkan pyramiding — PF selalu lebih rendah dari baseline, MaxDD selalu lebih buruk
29-40%. Gap pendek (1-3h) malah PnL dolar mentah lebih tinggi (lebih banyak leg lolos, market
sedang tren jadi nambah ke posisi untung menguntungkan secara nominal) tapi PF/risk-adjusted
tetap kalah — pola yang sama persis dengan temuan gate harga sebelumnya: pyramiding di sistem
ini "nambah ke winner", bukan "nolongin entry salah", apapun jenis gate-nya.

### Pseudo-holdout (2025-10-01 s.d. 2026-04-01, 414 trade baseline — SEMI IN-SAMPLE utk Guardian, sampel kecil)

| | trades | WR | PF | PnL | MaxDD |
|--|--------|-----|-----|-----|-------|
| A baseline | 414 | 61.4% | 2.049 | $167.53 | -$8.13 |
| gap 1h | 412 | 61.2% | 2.212 (+8.0%) | $258.97 (+55%) | -$10.78 (+33% lebih buruk) |
| gap 2h | 413 | 62.0% | 2.202 (+7.5%) | $242.64 (+45%) | -$10.63 (+31% lebih buruk) |
| gap 3h | 415 | 62.2% | 2.196 (+7.2%) | $230.54 (+38%) | -$8.37 (+3%, nyaris flat) |
| gap 5h | 415 | 61.9% | 2.002 (-2.3%) | $186.97 (+12%) | -$8.37 (flat) |
| gap 7h | 415 | 61.2% | 1.894 (-7.6%) | $169.99 (+1.5%) | -$9.08 (+12% lebih buruk) |
| gap 9h | 414 | 61.4% | 1.868 (-8.8%) | $162.57 (-3%) | -$8.71 (+7% lebih buruk) |

**Baca**: BERTOLAK BELAKANG dari full OOF — di window kecil & terbaru ini, gap pendek (1-3h)
justru PF/WR/PnL semua membaik, MaxDD nyaris flat di gap 3h. Tapi ini window YANG SAMA dipakai
melatih Guardian (semi in-sample, caveat yang sama berlaku di semua angka pseudo_holdout lain
di proyek ini) dan sampelnya kecil (414 trade, cuma ~1-2 trade multi-leg per bulan). Pola umum
di proyek ini: angka OOF/pseudo yang bagus SERING menyusut/berbalik di OOS genuine (lihat delta
Guardian OOF +1.135 PF vs OOS genuine cuma +0.171-0.334 PF). Tidak cukup kuat untuk override
kesimpulan full OOF tanpa uji OOS genuine — dan OOS butuh approval eksplisit user dulu.

### Follow-up SAMA HARI — OOS genuine (holdout-test, 2026-04-01 s.d. 2026-07-12, 314 trade baseline)

User minta ditindaklanjuti dengan OOS asli utk cek apakah sinyal positif pseudo-holdout di atas
nyata. **Script baru**: `pipeline/model/run_oos_pyramiding_time_gap_sweep.py` (adaptasi
`model/eval/holdout_oos.py::evaluate_coin`, entry H1-close, data `data/holdout-test/labeled/`).

| | trades | WR | PF | PnL | MaxDD |
|--|--------|-----|-----|-----|-------|
| A baseline | 314 | 55.1% | **1.248** | $28.68 | -$7.03 |
| gap 1h | 311 | 54.3% | 1.096 (-12.2%) | $16.75 (-42%) | -$13.13 (+87% lebih buruk) |
| gap 2h | 312 | 54.8% | 1.154 (-7.5%) | $24.60 (-14%) | -$13.03 (+85% lebih buruk) |
| gap 3h | 314 | 54.8% | 1.099 (-11.9%) | $15.68 (-45%) | -$12.75 (+82% lebih buruk) |
| gap 5h | 312 | 54.8% | 1.140 (-8.7%) | $20.58 (-28%) | -$11.90 (+69% lebih buruk) |
| gap 7h | 314 | 55.7% | 1.098 (-12.0%) | $13.78 (-52%) | -$11.11 (+58% lebih buruk) |
| gap 9h | 316 | 56.0% | 1.103 (-11.6%) | $13.90 (-52%) | -$11.11 (+58% lebih buruk) |

**Baca — KONKLUSIF, sinyal pseudo-holdout TIDAK terkonfirmasi.** Di data yang genuinely belum
pernah dilihat training (bukan window yang dipakai Guardian belajar), gap 1-3h yang tadinya
kelihatan bagus di pseudo-holdout justru SEMUA gap (1-9h) lebih jelek dari baseline di PF & PnL,
dan MaxDD 58-87% LEBIH DALAM. Ini persis pola yang diantisipasi: angka pseudo-holdout yang
bagus adalah optimisme sampel kecil/semi in-sample, tidak generalisasi ke data baru.

**Artifact**: `models/runs/guard_opt2_plus_trend_hmm_18coin/pyramiding_time_gap_sweep_oos.json`

### Keputusan FINAL

**Pyramiding CLOSED, ketiga sudut sudah diuji tuntas dgn hasil konsisten menolak**: (1) mode
`independent` salah parity, (2) gate harga (rugi>=2.5%), (3) gate waktu (1-9h) — OOF besar DAN
OOS genuine dua-duanya sepakat PF lebih rendah & MaxDD lebih buruk di semua varian. Tidak ada
alasan lagi untuk mengulang eksperimen pyramiding tanpa temuan struktural baru (misal: arsitektur
sizing berbeda, bukan cuma gate tambahan).

---

## 2026-07-13 — Fix gap: config Riset tidak tahu soal cooldown/limit_exit live (nyaris ke-hapus saat deploy berikutnya)

**Status**: APPLIED (config sync + safety net, tidak ada deploy ke VPS)
**Trigger**: user minta cek "setup model terbaik dengan cooldown".

Ditemukan: `models/inference_config.json` (Riset) draft v6.4 (belum commit) sudah include model 18-koin/37f,
tapi TIDAK punya section `cooldown`/`limit_exit` sama sekali — dua fitur yang sudah live di swint_tradev2
sejak 2026-07-12/13 (lihat [[project-guardian-floor-cooldown-deploy]]). Kode keduanya cuma ada di
swint_tradev2 (`paper_trading.py`, `signal_filter.py`), tidak pernah direplikasi ke Riset. `PRESERVE_KEYS`
di `tools/ops/deploy_model.py` juga tidak melindungi kedua key ini — kalau draft v6.4 di-commit lalu
`deploy_production.py` dijalankan, config baru akan menimpa target dan MENGHAPUS cooldown+floor stop-limit
dari production (regresi ke bug re-entry ala-AVAX yang baru diperbaiki).

**Fix**:
1. `models/inference_config.json` — tambah section `cooldown` (enabled/profit_only/profit_hours=1) dan
   `limit_exit` (enabled/floor_tp_frac=0.7), nilai disalin persis dari mirror lokal swint_tradev2.
2. `tools/ops/deploy_model.py` `PRESERVE_KEYS` — tambah `"limit_exit"` dan `"cooldown"` (whole-object preserve)
   supaya config operasional ini tidak pernah ketimpa dari source Riset di deploy manapun ke depan.

3. `models/model_registry.json` — diupdate penuh dari v6.1/38f (21-koin, stale sejak 2026-06-22 draft)
   ke v6.4/37f (18-koin): `active`/`display_name`/`benchmark`/`architecture` ganti, `stack.lgbm` ->
   `opt2_plus_trend_18coin_iso37f`, `stack.guardian` -> `guard_opt2_plus_trend_hmm_18coin` (cv_f1_macro/
   cv_logloss dihitung ulang dari `guardian_cv_results.json` aktual, BUKAN dicopy dari guard28f lama),
   tambah `stack.spot_confirm` + `stack.regime_disable` (komponen baru yang belum ada di schema lama),
   `oof_scorecard`/`sealed_holdout_oos` disamakan dgn angka di `inference_config.json.scorecard`.
   Ditambah field `post_deploy_operational_note` menjelaskan cooldown/limit_exit ditambahkan terpisah
   setelah deploy model ini (tidak mengubah model/n_features).

**Belum dibereskan** (di luar scope task ini): banner atas `CLAUDE.md` ("Production live: ic32_regime_v6
/ fs38_28f, deploy 2026-07-03") juga masih menyebut stack lama — belum disamakan ke v6.4/fs37_18coin_spotconfirm.

---

## 2026-07-12 — OOF: cooldown 1h after profit + pyramiding max2

**Status**: OOF COMPLETE (not deployed)  
**Stack**: `LGBM opt2_plus_trend + HMM + guard_opt2_plus_trend_hmm` (fs38_28f, guardian ON)  
**Script**: `pipeline/model/run_oof_cooldown_pyramiding.py`  
**Artifact**: `models/runs/guard_opt2_plus_trend_hmm/oof_cooldown_pyramiding.json`

### Variants
- **A** baseline: single pos, no cooldown  
- **B** cooldown 1 bar (1h) only if `net_pnl > 0`  
- **C** pyramiding max 2 same-dir  
- **D** C+B  

### Full OOF (→2026-04)
| | trades | WR | PF | PnL | MaxDD |
|--|--------|-----|-----|-----|-------|
| A | 4522 | 66.0% | 2.35 | $2699 | $-26 |
| B | 4481 | 66.1% | 2.37 | $2689 | $-26 (−0.4% PnL) |
| C | 6374 | 66.2% | 2.35 | $3766 | $-40 (+40% PnL, DD+52%) |
| D | 5037 | 60.3% | 1.81 | $2115 | $-42 (−22% PnL) |

### Keputusan
- **B** netral OOF → OK sebagai policy live anti re-entry pasca profit (AVAX case).  
- **C** PnL naik, DD naik — jangan deploy tanpa risk budget.  
- **D** tolak.  
Belum enable di live `inference_config` / `signal_filter` (butuh approval).

---

## Index Eksperimen (2026-06-22 s.d. 2026-07-04, kronologis per blok)

- 2026-07-04 — Fix inference_config: structural_filter OFF di source deploy (bukan cuma live UI) — Status: **APPLIED** — `models/inference_config.json` (Riset + swint), `deploy_model.py` enforce dari `config.py` — [detail](archive/EXPERIMENTS_full_history_2026-07-08.md#L7)
- 2026-07-04 — OOF filter sweep: structural/swing-fresh OFF, HMM+RR only (selaras live) — Status: **APPLIED** — `config.py` `TP_SL_STRUCTURAL_FILTER=False`, `TP_SL_SWING_FRESHNESS=False` — [detail](archive/EXPERIMENTS_full_history_2026-07-08.md#L24)
- 2026-07-04 — Fix `trend_accel_4h` double-ATR + deploy v6.1.1 (threshold 0.65/0.10, min_hold=0) — Status: **DEPLOYED** — VPS `e299287` (git) + inference cache rebuild 2026-07-04T04:35Z — [detail](archive/EXPERIMENTS_full_history_2026-07-08.md#L55)
- 2026-07-04 — Deploy v6.1 / fs38_28f (H4-closed + GRAMUSDT) — Status: **DEPLOYED** — snapshot `2026-07-03 18:05:00 UTC`, VPS `ff37118`, rollback v6.0 snapshot `2026-07-03 11:50:45` — [detail](archive/EXPERIMENTS_full_history_2026-07-08.md#L112)
- 2026-07-03 — Refactor H4 closed + shift(4h) (seragamkan EMA/RSI/trend/swing) — Status: RETRAIN DONE — deploy v6.1 2026-07-04 — [detail](archive/EXPERIMENTS_full_history_2026-07-08.md#L140)
- 2026-07-01 — Guardian k5_mom **v7** (pnl_constrained_exit — SWEET SPOT TERCAPAI) — Status: OOF VALIDATED — lokal only, belum deploy VPS — [detail](archive/EXPERIMENTS_full_history_2026-07-08.md#L197)
- 2026-07-01 — Ringkasan: Guardian vs stack **k5_mom embedded** (kesalahan & hasil) — Status: COMPLETE — tidak ada kandidat sweet spot; tidak deploy — [detail](archive/EXPERIMENTS_full_history_2026-07-08.md#L289)
- 2026-07-01 — Audit momentum_v2 WR (trade-level forensics) — Status: COMPLETE — root cause identified — [detail](archive/EXPERIMENTS_full_history_2026-07-08.md#L378)
- 2026-07-01 — Guardian k5_mom v6 (fair ablation no p_bull, label_end fix) — Status: OOF VALIDATED — lokal only — [detail](archive/EXPERIMENTS_full_history_2026-07-08.md#L435)
- 2026-07-01 — Guardian k5_mom v5 retrain (Opsi A: TP-phase 3-class, no p_bull) — Status: OOF VALIDATED — lokal only, belum deploy VPS — [detail](archive/EXPERIMENTS_full_history_2026-07-08.md#L466)
- 2026-07-01 — Guardian k5_mom v3 retrain (3-class, no p_bull) — Status: OOF VALIDATED — lokal only, belum deploy VPS — [detail](archive/EXPERIMENTS_full_history_2026-07-08.md#L501)
- 2026-07-01 — Guardian momentum_v2 live parity re-sim — Status: OOF VALIDATED — tidak deploy — [detail](archive/EXPERIMENTS_full_history_2026-07-08.md#L530)
- 2026-06-30 — Guardian k5_mom v2 (binary peak escort) — Status: OOF VALIDATED — belum deploy — [detail](archive/EXPERIMENTS_full_history_2026-07-08.md#L555)
- 2026-06-27 — ic32.rv5.coin_fs.v1 — Status: IN_PROGRESS (masih aktif, lihat `pipeline/experiments/ic32_rv5_coin_fs/CLAUDE.md`) — [detail](archive/EXPERIMENTS_full_history_2026-07-08.md#L692)
- Baseline Aktif — ic32_regime_v2_parity (Deployed 2026-06-22) — Status: PRODUCTION (saat itu) — baseline bersih untuk semua eksperimen berikutnya — [detail](archive/EXPERIMENTS_full_history_2026-07-08.md#L761)
- 2026-06-23 — Holdout: LGBM Parity + Guardian Momentum Escort v2 — Status: COMPLETED — HOLDOUT_EVALUATED=True, amplop dikunci — [detail](archive/EXPERIMENTS_full_history_2026-07-08.md#L801)
- 2026-06-24 — DEPLOY ic32_regime_v4 (LGBM + HMM gate 0.65/0.05 + Guardian v2) — Status: DEPLOYED ke production VPS (keputusan eksplisit user, override kriteria upgrade) — [detail](archive/EXPERIMENTS_full_history_2026-07-08.md#L862)
- 2026-06-25 — Feature Selection: Stability Analysis + Ablation + Reduced Retrain (ic32.rv2.lgbm.24f) — Status: (lihat detail — superseded oleh opt2_plus_trend) — [detail](archive/EXPERIMENTS_full_history_2026-07-08.md#L891)
- 2026-06-25 — WD tanpa Komponen L/S (WD = CVD z-score saja) — Status: (lihat detail) — [detail](archive/EXPERIMENTS_full_history_2026-07-08.md#L1019)
- 2026-06-23 — Holdout: LGBM + HMM Gate + Guardian v2 (base=0.74, delta=0.06) — Status: COMPLETED — HOLDOUT_EVALUATED=True, amplop dikunci — [detail](archive/EXPERIMENTS_full_history_2026-07-08.md#L1095)
- 2026-06-23 — Daily LSTM v2: Binance-direct L/S + Automasi p_bull Harian — Status: (lihat detail — LSTM daily tidak dipakai di v6+) — [detail](archive/EXPERIMENTS_full_history_2026-07-08.md#L1153)
- 2026-06-23 — ic32_regime_v4: HMM-Gated Direction Threshold — Status: (lihat detail — superseded oleh HMM per-state v6) — [detail](archive/EXPERIMENTS_full_history_2026-07-08.md#L1199)
- 2026-06-24 — OOF: Guardian Momentum v2 Exit-Param Sweep di Stack v4 (HMM 0.65/0.05) — Status: COMPLETED — kesimpulan: **frontier PF jenuh, tidak ada upgrade** — [detail](archive/EXPERIMENTS_full_history_2026-07-08.md#L1322)
- 2026-06-24 — MODEL BARU: Trade-Quality Sizing (volume-neutral payoff lift) — Status: COMPLETED (OOF) — sinyal nyata tapi lemah; PnL/PF naik tapi MaxDD memburuk → tidak deploy — [detail](archive/EXPERIMENTS_full_history_2026-07-08.md#L1388)
- 2026-06-24 — Investigasi Lever RR: min_rr / Entry Timing / TP Extension — Status: (lihat detail — kesimpulan dipakai di v6 structural_filter OFF) — [detail](archive/EXPERIMENTS_full_history_2026-07-08.md#L1513)
- 2026-06-24 — Guardian Min-Gain Gate (Opsi A): Retrain dengan Patience Gate — Status: (lihat detail — superseded) — [detail](archive/EXPERIMENTS_full_history_2026-07-08.md#L1586)
- 2026-06-25 — Holdout 24 Juni 2026 WIB: ic32_regime_v4 (model live) — Status: COMPLETED — [detail](archive/EXPERIMENTS_full_history_2026-07-08.md#L1644)
- 2026-07-02 — Ablation Dekorelasi Geometri + Prinsip No-Cross-Model-Features (37f/33f) — Status: DONE (OOF) — kandidat 33f layak lanjut fase threshold sweep — [detail](archive/EXPERIMENTS_full_history_2026-07-08.md#L1697)
- 2026-07-03 — IC Test 6 Kandidat Fitur Baru + Marginal Test (basis 37f) — Status: DONE (OOF) — `ofi_z_score` lolos & bernilai tambah nyata; tidak memperbaiki bias short-saat-pump — [detail](archive/EXPERIMENTS_full_history_2026-07-08.md#L1749)
- 2026-07-03 — Audit Leakage + Marginal Test di atas 33f (ofi_z_score, atr_percentile_h1, vol_spike_zscore) — Status: DONE — **leakage `vol_spike_zscore` terkonfirmasi** (dipakai di label-gen); `ofi_z_score`+`atr_percentile_h1` bersih — [detail](archive/EXPERIMENTS_full_history_2026-07-08.md#L1813)
- 2026-07-03 — Otak-atik Kombinasi Dekorelasi Keluarga Geometri (basis 37f) — Status: DONE (OOF) — `v2` (keep 20x, buang 50x+Buy/Sell_Liq) terbaik, mengalahkan `v1`/33f — [detail](archive/EXPERIMENTS_full_history_2026-07-08.md#L1870)
- 2026-07-03 — KOREKSI: Bug Threshold Flat 0.65 di Semua Script "Replay Insiden" — Status: DONE — angka drill-down sesi ini pakai threshold salah, tidak berdampak ke scorecard resmi — [detail](archive/EXPERIMENTS_full_history_2026-07-08.md#L1965)
- 2026-07-03 — Riset Fitur Regime-Adaptive (basis 35f-clean) — HASIL NEGATIF — Status: DONE — `trend_strength`/`no_demand` awalnya DITOLAK di basis ini, `wyckoff_phase` "tidak bersih" — [detail](archive/EXPERIMENTS_full_history_2026-07-08.md#L2017)
- 2026-07-03 — Riset Fitur Leading Indicator Gen-2 (basis 35f-clean) — HASIL POSITIF — Status: DONE — `absorption_at_swing` lolos — [detail](archive/EXPERIMENTS_full_history_2026-07-08.md#L2089)
- 2026-07-03 — Round 3: IC Test Pool Luas + hidden_divergence/ofi_momentum_ratio (DITOLAK) + Replay 30 Jun-2 Jul — Status: DONE — kandidat baru ditolak; 37f (35f-clean+absorption_at_swing+vwdp) tetap — [detail](archive/EXPERIMENTS_full_history_2026-07-08.md#L2154)
- 2026-07-03 — Opsi 2: Redesain Label Struktural (gate momentum ofi_z_score, ganti vol_spike_zscore) — Status: DONE — hipotesis inti tervalidasi — [detail](archive/EXPERIMENTS_full_history_2026-07-08.md#L2232)
- 2026-07-03 — Retest Fitur Gagal vs Label Baru: `trend_strength` Berhasil, Insiden Sempit Tetap Campur — Status: DONE — `trend_strength` terbukti genuinely membantu dgn label Opsi 2 — [detail](archive/EXPERIMENTS_full_history_2026-07-08.md#L2312)
- 2026-07-03 — Threshold Sweep HMM + Uji HMM Fast-React (vol/mom window 6/12) — Status: DONE — kandidat final terpilih **opt2_plus_trend** — [detail](archive/EXPERIMENTS_full_history_2026-07-08.md#L2378)
- 2026-07-03 — Skenario A: Monotone Constraints via One-vs-Rest (SHORT-vs-rest, LONG-vs-rest) — Status: DONE — constraint bekerja benar, OOF PnL naik — [detail](archive/EXPERIMENTS_full_history_2026-07-08.md#L2468)
- 2026-07-03 — Skenario B: Validasi Fast-React HMM Skala Penuh — Status: DONE — **DITOLAK**, uji kecil sebelumnya (n=19) menyesatkan — [detail](archive/EXPERIMENTS_full_history_2026-07-08.md#L2553)
- 2026-07-03 — Skenario C: Perbesar Radius Gate Momentum (ofi_z_score 1.5→1.2) — Status: DONE — **DITOLAK** (verdict user) — [detail](archive/EXPERIMENTS_full_history_2026-07-08.md#L2600)
- 2026-07-03 — Guardian Baru utk opt2_plus_trend + HMM (no cross-model features) — Status: DONE — full-stack OOF & pseudo-holdout membaik signifikan, proporsional — [detail](archive/EXPERIMENTS_full_history_2026-07-08.md#L2648)
- 2026-07-03 — DEPLOY ic32_regime_v6 / fs38_28f ke Production — Status: DONE — live di VPS sejak 2026-07-03 07:49 UTC — [detail](archive/EXPERIMENTS_full_history_2026-07-08.md#L2721)
- 2026-07-03 — Sweep guard28f `guardian_min_hold_bars`: 4 vs 0 — Status: DONE — tidak ada efek berarti, param 4 (kemudian diubah ke 0, lihat entry v6.1.1) — [detail](archive/EXPERIMENTS_full_history_2026-07-08.md#L2763)
- 2026-07-03 — LSTM Confirmation Cascade (v1 win/loss, v2 continuation magnitude) — Status: DONE — **KEDUA VARIAN DITOLAK** (AUC ~0.54, hampir acak) — [detail](archive/EXPERIMENTS_full_history_2026-07-08.md#L2797)
- 2026-07-03 — Rasa Penasaran: XGBoost & CatBoost vs LightGBM (algoritma entry) — Status: DONE — **DITOLAK**, LightGBM tetap terbaik — [detail](archive/EXPERIMENTS_full_history_2026-07-08.md#L2892)
- 2026-07-03 — Ensemble Regime-Specialist LGBM (4 model per HMM state) — Status: DONE — **DITOLAK** — [detail](archive/EXPERIMENTS_full_history_2026-07-08.md#L2938)
- 2026-07-03 — Investigasi OOS Holdout Rugi (PF<1) — Root Cause + Fix `positioning_mode` — Status: DONE — **bug produksi ditemukan & diperbaiki** (LSR di-strip jadi konstan) — [detail](archive/EXPERIMENTS_full_history_2026-07-08.md#L3025)
- 2026-07-03 — UI Pemantauan Fitur Live (Feature Monitoring Dashboard) — Status: DONE — dideploy ke production — [detail](archive/EXPERIMENTS_full_history_2026-07-08.md#L3099)
- 2026-07-04 — DEPLOY ic32_regime_v6.1 (H4-closed + funding_rate + TON→GRAM) + INSIDEN market-panel holdout NaN — Status: DONE — v6.1 live di VPS, snapshot `2026-07-03 18:05:00 UTC` — [detail](archive/EXPERIMENTS_full_history_2026-07-08.md#L3167)

---

## Index Eksperimen (2026-07-08 s.d. 2026-07-12, dipadatkan 2026-07-14)

- 2026-07-08 — Insiden: Drift Tak Terdokumentasi 07-06 (partial-H4 + Guardian OFF) — Audit + Rollback ke v6.1 — Status: DONE — root cause ditemukan, production di-rollback ke v6.1 (verified), repo dibersihkan — [detail](archive/EXPERIMENTS_full_history_2026-07-14.md#L8)
- 2026-07-08 — KANDIDAT DIKUNCI: fs38_18coin_spotconfirm (18-koin + spot-confirm + regime-disable + Guardian) — Status: CANDIDATE dikunci sementara, bukan production — [detail](archive/EXPERIMENTS_full_history_2026-07-14.md#L40)
- 2026-07-09 — Investigasi fitur market-context (btc_ret_24h fade, R5 monotone, R1, coin_mkt_sync_24h) — Status: SELESAI diinvestigasi, v6.3 tetap live tanpa perubahan, R5 disimpan sbg kandidat riset — [detail](archive/EXPERIMENTS_full_history_2026-07-14.md#L231)
- 2026-07-10 — Audit parity live-vs-riset (trigger XRPUSDT SHORT@1.0911 salah arah) + fix — Status: SELESAI, 2 bug live DIFIX & DIDEPLOY, 1 fitur di-takedown dari feature set — [detail](archive/EXPERIMENTS_full_history_2026-07-14.md#L300)
- 2026-07-10 (lanjutan) — Jadwal live digeser HH:05→HH:15 + entry price M15 di OOF/OOS — Status: SELESAI, jadwal live sudah diubah — [detail](archive/EXPERIMENTS_full_history_2026-07-14.md#L380)
- 2026-07-10 (lanjutan lagi) — KOREKSI: takedown relative_strength_z diulang di universe 18-koin (bukan 21) — Status: SELESAI, koreksi kesalahan eksperimen sebelumnya — [detail](archive/EXPERIMENTS_full_history_2026-07-14.md#L438)
- 2026-07-10 (lanjutan lagi) — DEPLOY: ic32_regime_v6.4 (lgbm37f_18coin, takedown relative_strength_z) — Status: SELESAI, LIVE — [detail](archive/EXPERIMENTS_full_history_2026-07-14.md#L516)
- 2026-07-11/12 — Reproduksi OOS v6.4 vs live 11 Juli: 3 bug ditemukan & difix (2 penuh, 1 parsial) — Status: SELESAI utk tujuan reproduksi, 2 bug difix permanen — [detail](archive/EXPERIMENTS_full_history_2026-07-14.md#L548)
- 2026-07-12 — Sweep threshold HMM v6.4 (OOF) + FIX live: SL close-mode pakai harga intrabar, bukan candle settled — Status: sweep SELESAI + divalidasi OOS (kandidat ditolak), fix live SELESAI & DIDEPLOY — [detail](archive/EXPERIMENTS_full_history_2026-07-14.md#L652)
- 2026-07-12 (lanjutan) — Fix Bug 3 di source (spot-confirm indexing) + auto-refresh panel + config.py disamakan ke 18-koin — Status: SELESAI, ketiganya diterapkan — [detail](archive/EXPERIMENTS_full_history_2026-07-14.md#L712)
- 2026-07-12 (lanjutan lagi) — Audit trade-by-trade 12 Juli: 4 gap baru ditemukan (3 metodologi, 1 bug dicatat belum fix, 1 kebijakan bukan bug) — Status: semua ditemukan & didokumentasikan, fix Guardian (item C) SENGAJA DITUNDA — [detail](archive/EXPERIMENTS_full_history_2026-07-14.md#L766)
- 2026-07-12 (koreksi) — Item A di atas ("same-bar re-entry gap") SALAH — root cause asli: swing sidedness, bukan gap timing. Fix diuji OOF+OOS, DITOLAK — Status: SELESAI diinvestigasi, root cause asli ditemukan & diverifikasi — [detail](archive/EXPERIMENTS_full_history_2026-07-14.md#L822)
- 2026-07-12 (lanjutan) — Root cause kenapa "tolak swing basi" menang: cocok dgn label training model. Varian HMM-conditional diuji & KALAH. Fix DITERAPKAN ke live (lokal, BELUM deploy) — Status: root cause dikonfirmasi, varian granular diuji & ditolak — [detail](archive/EXPERIMENTS_full_history_2026-07-14.md#L896)
- 2026-07-12 (lanjutan) — Model baru `lgbm37f_trend` (label triple-barrier ATR, khusus TRENDING_UP): dibangun, diuji OOF+OOS, DIDEPLOY LIVE (regime_model_routing) — Status: SELESAI & LIVE — [detail](archive/EXPERIMENTS_full_history_2026-07-14.md#L968)
- 2026-07-12 (lanjutan) — Tool baru `compare_oos_live_signals.py` + fix `long_short_ratio` utk 6 koin baru (window-length-dependent bug) — Status: SELESAI & DIDEPLOY — [detail](archive/EXPERIMENTS_full_history_2026-07-14.md#L1080)

---

## Template Eksperimen Berikutnya

```markdown
## YYYY-MM-DD — [Nama Eksperimen]

**Status**: PLANNED

### Hipotesis
[Apa yang diduga akan terjadi dan mengapa]

### Yang Diubah
- [vs baseline production saat ini]

### Target
- WR > 52%, PF > 1.0, Trades >= 45 (80% dari 56 baseline)
- Metodologi: genuine OOF, purge gap 36 bar

### Script
- [script yang akan dijalankan]
```
