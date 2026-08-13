# EXPERIMENTS.md — Logbook Riset ic32 Regime v2+

> **Histori penuh diarsipkan** (isi lengkap tiap entry, bukan cuma ringkasan):
> - Pre-2026-06-22: `archive/2026-06-22_cleanup/root_files/EXPERIMENTS_full_history.md`
> - 2026-06-22 s.d. 2026-07-04: `archive/EXPERIMENTS_full_history_2026-07-08.md`
> - 2026-07-08 s.d. 2026-07-12: `archive/EXPERIMENTS_full_history_2026-07-14.md`
> - 2026-07-26 s.d. 2026-08-05: `archive/EXPERIMENTS_full_history_2026-08-06.md` (dipindah 2026-08-06 -- file utama sempat 9.786 baris/656KB, terlalu boros token)
>
> File ini cuma **index** (1 baris/eksperimen: tanggal — judul — status — link ke detail).
> Dipecah 2026-07-08 karena versi lama (3269 baris) terlalu boros token untuk dibaca rutin.
> **Entry BARU tulis lengkap di sini dulu** (pakai Template di bagian bawah) — baru dipadatkan
> jadi 1 baris index + dipindah ke archive kalau file ini mulai membengkak lagi.
>
> **AMBANG ARSIP (wajib dicek proaktif, bukan tunggu ditegur user)**: kalau file ini lewat
> **~2.500 baris atau ~150KB**, TIDAK ADA hook otomatis yg menegur (beda dgn MEMORY.md yg
> punya hook) — jadi WAJIB `wc -l`/`wc -c` cek sendiri di awal sesi yg akan menulis eksperimen
> baru ke sini. Kalau lewat ambang: pindah SEMUA entry lama (kecuali Template & sesi
> aktif/hari berjalan) ke `archive/EXPERIMENTS_full_history_{tanggal-hari-ini}.md` (lossless,
> verifikasi jumlah baris cocok), tambah 1 baris baru di daftar arsip di atas, dan cek/perbaiki
> pointer di `EXPERIMENTS_INDEX.jsonl` yg mungkin jadi basi nunjuk ke entry yg baru dipindah.
> Detail protokol: memori `feedback-efisiensi-file-log-proaktif`.

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

---

## 2026-08-06 — Fase 107-110: `ExecutionSimulator` (paritas riset↔live) + ISOLASI Mode A/B/C — akar masalah sebenarnya BUKAN cek 5-menit, tapi HARGA FILL EXIT

**Rencana disetujui (`RENCANA_PARITAS_EKSEKUSI_RISET_LIVE_20260805.md`, Opus).** Tahap 1-4: `ExecutionCfg` di `config.py` (flag `monitor_enabled`, `monitor_interval_min`, `entry_regate_at_market`, `exit_fill_at_market`, dst — semua default = perilaku live SEKARANG, tidak mengubah apa pun); `hitung_net_pnl()` diekstrak dari `paper_engine._finalize_trade` (byte-identik, diverifikasi via test parity 2.156-trade existing); `execution_sim.py`/`ExecutionSimulator` — mesin replay BARU yg memakai KODE LIVE ASLI (`position_monitor._alasan_exit`, `live_executor._entry_gate`) via import langsung, bukan duplikasi logika; `harness_eksekusi.py` (Riset_pemodelan) memanggil mesin yg sama utk OOF/holdout. Semua commit `live_dualbin_ft` worktree `real-money-execution`: `3d751b2`, `5a77e35`, `52ce481`, `656182c`.

**Fase 107 (holdout, model takedown, harness baru).** Mode-1 (H1-only) tervalidasi PERSIS sama dgn Fase 105 (3.195 trade, PF 1,9238, PnL $1.518,17) — membuktikan harness benar. Mode-2 (harness PENUH: delay+regate+monitor+exit-fill-market, SEMUA REALISTIS spt live):

| | Mode-1 (H1 saja) | Mode-2 (harness penuh, realistis) |
|---|---:|---:|
| Trades | 3.195 | 3.364 |
| PF | 1,9238 | **0,7325** |
| PnL | $1.518,17 | **-$649,00** |
| entry_ditolak | - | 3,10% |

Penurunan JAUH lebih parah dari perkiraan Fase 103/105 (yg cuma isolasi 1 mekanisme). Diverifikasi di Fase 108 (OOF W1/W2/W3, robustness) — pola SAMA konsisten di 3 window independen: W1 PF 0,6327/PnL -$1.257,55; W2 PF 0,7260/PnL -$774,58; W3 PF 0,6698/PnL -$981,50 (angka ini jadi BASELINE rujukan Fase 116 di bawah).

**Fase 109 — isolasi Mode A/B/C (kunci diagnosis).** Utk cari mekanisme mana yg dominan:
- Mode A (semua realistis, spt Fase 107 Mode-2): PF 0,7325 / PnL -$649
- Mode B (HANYA cek-5-menit dimatikan, sisanya realistis): PF 0,7598 / PnL -$559 — **hampir tidak membaik**
- Mode C (cek-5-menit MATI + harga fill exit dikembalikan ke level ideal bar H1, bukan harga pasar real): PF **1,8426** / PnL **+$1.402** — hampir balik ke baseline awal

**Kesimpulan Fase 109 (memperbaiki hipotesis awal user sendiri, dgn bukti keras):** penyebab dominan BUKAN cek 5-menit (Mode B nyaris tak membantu), tapi HARGA FILL EXIT (real market price vs harga ideal bar) — Mode C sendirian mengembalikan hampir seluruh performa. Ini mengarahkan investigasi ke cara LABEL TRAINING dihitung, bukan cuma cara backtest mengukur.

**Fase 110 — uji cepat mitigasi cek-SL (N-konfirmasi 1x/2x/3x, SL-only-H1).** Semua varian TIDAK membantu berarti (PnL -$3.070 s/d -$2.965, semua dlm derau) — mengonfirmasi ulang Fase 109: masalahnya bukan di cadence monitor.

**Temuan tambahan (pembacaan kode, bukan pengukuran).** `core/features.py::swing_based_labeling` (label training SEMUA model entry DualBin, historis) py LOOK-AHEAD BIAS: `if h_arr[j] >= tp_long: outcome_long = "LONG"` — mengklaim WIN begitu wick masa depan MANAPUN menyentuh TP, tanpa model delay-deteksi/harga-fill realistis. Ini SECARA STRUKTURAL sama dgn bias yg baru diperbaiki di mesin backtest (idealisasi harga exit). Ditemukan tambahan: `paper_engine._finalize_trade` membukukan loss SL di `pos.sl_price` (level SL persis), bukan `close` bar yg benar2 menembusnya — bias optimis lain yg sama arahnya.

**Fitur baru sekalian (permintaan user).** `execution.monitor_sl_enabled` (default `true`, TIDAK mengubah live) — kalau `false`, monitor 5-menit MENGABAIKAN sentuhan SL (SL hanya diputuskan via cek H1 close), TP tetap instan via monitor. Diimplementasi identik di `position_monitor.py` (live) & `execution_sim.py` (riset), TDD penuh. Commit `656182c`.

**Keputusan.** Fokus pindah dari "perbaiki mekanisme eksekusi" ke "perbaiki cara label training dihitung" — lihat Fase 111-116 di bawah. Rencana lama (`RENCANA_PARITAS_EKSEKUSI...`) Tahap 5+ (re-baseline dgn mekanisme realistis) DITUNDA, kalah prioritas dari temuan leakage label.

**Artefak.** `app/config.py` (`ExecutionCfg`), `app/services/execution_sim.py` (`ExecutionSimulator`, `simulasi_entry_paksa`), `app/services/paper_engine.py` (`hitung_net_pnl`), `app/services/position_monitor.py`, `app/services/scheduler.py` — semua `live_dualbin_ft` worktree `real-money-execution`. `pipeline/experiments/dualbin_universe/harness_eksekusi.py`, `fase107_harness_holdout_reproduksi.py`, `fase107b_diagnosa_selisih_trade_count.py`, `fase108_harness_oof_robustness.py`, `fase109_cek_cepat_monitor_off.py`, `fase110_uji_sl_konfirmasi_2x.py` (semua +log).

---

## 2026-08-06 — Fase 111-116: Label-by-Replay (anti-leakage) — Gerbang Oracle LOLOS, tapi OOF retrain DITOLAK di semua window

**Rencana disetujui (`RENCANA_LABEL_ANTI_LEAKAGE_20260806.md`, Opus).** Alih-alih rule tangan (`swing_based_labeling`), label dihasilkan dari mesin eksekusi yg SAMA yg sudah divalidasi Fase 107-110: `simulasi_entry_paksa(coin, bars, i, arah, cfg, engine)` — paksa keputusan LONG/SHORT di bar `i`, replay jujur ke depan (delay+regate+monitor+fill market, via `ExecutionSimulator(..., ketat_m5=True)` — mode ketat baru: RAISE `M5TidakTersedia` kalau data M5 tak cukup, bukan diam2 fallback ke harga ideal), label = `net_pnl > 0`. Keputusan eksplisit user: label baru pakai `monitor_sl_enabled=False` (SL hanya via H1) — bukan berarti live diubah, ini asumsi label generation.

**Fase 112 (benchmark).** 6,25 ms/kandidat — dipakai hitung biaya sebelum commit ke generasi penuh (bukan asumsi).

**Fase 113 — Gerbang Oracle (W3, ~5 menit, tes murah-tapi-menentukan SEBELUM generasi penuh).** Kalau trader "tahu masa depan" (pilih top-k% kandidat via label jujur) tetap tidak profitable, seluruh rencana gugur di sini — hemat berhari-hari. Gerbang: top-5% selektivitas harus PF≥1,5 DAN PnL>0.

| Metrik | Nilai |
|---|---:|
| PF (top-5%) | **inf** (nol trade rugi) |
| PnL (top-5%) | **+$7.957** / 121 hari (~20 trade/hari) |
| PnL ceiling (semua kandidat profitable) | $26.666 |
| Entry ditolak (gate re-check harga pasar) | 53,19% (mengoreksi klaim Fase 106 lama 10,58% — bug timestamp: `ts+5menit` bukan `ts+1jam+5menit`; indeks bar H1 = waktu OPEN candle, bukan close) |

**LOLOS spektakuler** → lanjut ke generasi label penuh.

**Fase 114 (generasi label penuh).** 2024-04-01 s/d ~2026-07-29, 18 koin, `monitor_sl_enabled=False`, ketat_m5=True. 43,9 menit. Output: `data/training/label_eksekusi_jujur/{coin}_label.parquet` (`net_pnl_long/short`, `exit_channel_long/short`, `ditolak_long/short`).

**Fase 115 (diagnosa label lama vs baru — gerbang kewarasan wajib sebelum retrain).**

| | LONG | SHORT |
|---|---:|---:|
| positive_rate lama → baru | 0,0959 → 0,2719 | 0,1055 → 0,2678 |
| Kesepakatan (lama vs baru) | 75,81% | 76,44% |
| Cohen kappa | 0,2337 | 0,2562 |
| Rata2 net_pnl jujur di label-1 LAMA | $0,65 | $0,62 |
| Rata2 net_pnl jujur di label-1 BARU | $1,08 | $1,01 |
| % bar label-1 LAMA yg simulasi jujurnya RUGI | 34,34% | 34,77% |

Dua gerbang kewarasan (kesepakatan TIDAK boleh >95% [lolos, ~76%] & net_pnl baru HARUS lebih tinggi dari lama [lolos, +65-70%]) SAMA-SAMA LOLOS — label baru terbukti membawa informasi baru, bukan re-labeling kosmetik. Lebih dari 1/3 label "WIN" versi lama ternyata RUGI kalau dieksekusi jujur — bukti kuantitatif langsung dari leakage.

**Fase 116 — retrain OOF, 2 arm terkunci × 3 window (W1/W2/W3), gerbang ABSOLUT (PF>1,00 DAN PnL>0 di SETIAP window, bukan "lebih baik dari baseline").** Arm A1 = target biner tanpa bobot; Arm A2 = `sample_weight=|net_pnl|`. Baseline rujukan = angka Fase 108 (model live, label lama, harness realistis).

| Window | Arm | Trades | PF | PnL | Baseline (label lama) | Vonis |
|---|---|---:|---:|---:|---:|---|
| W1 | A1 | 3.606 | 0,6952 | -$1.009,52 | PF 0,6327 / -$1.257,55 | GAGAL gerbang absolut |
| W1 | A2 | 840 | 0,6984 | -$236,09 | idem | GAGAL gerbang absolut |
| W2 | A1 | 3.617 | 0,7327 | -$865,77 | PF 0,7260 / -$774,58 | GAGAL gerbang absolut |
| W2 | A2 | 1.050 | 0,7222 | -$300,68 | idem | GAGAL gerbang absolut |
| W3 | A1 | 3.732 | 0,7299 | -$853,04 | PF 0,6698 / -$981,50 | GAGAL gerbang absolut |
| W3 | A2 | 1.240 | 0,7130 | -$311,69 | idem | GAGAL gerbang absolut |

Catatan tambahan: AUC train arm A2 jelek (0,54-0,71, LONG di W2/W3 nyaris tebak-acak) vs A1 (0,90-0,91) — pembobotan `|net_pnl|` membuat pola arah jauh lebih sulit dipelajari, "menang" cuma krn lebih selektif (entry_ditolak 56-72 vs 23-29, trade jauh lebih sedikit).

**VONIS: TOLAK KEDUA ARM. Tidak lanjut ke holdout tersegel.** Label anti-leakage TERBUKTI membantu (rugi mengecil di 5/6 perbandingan, arm A2 rugi turun 70-80%) — bug leakage nyata & perbaikannya bukan sia-sia — tapi TIDAK CUKUP membuat model entry `swing` ini profitable di bawah asumsi eksekusi jujur berdiri sendiri. PF mentok konsisten 0,69-0,73 di semua window kedua arm. Faktor lain (fitur entry, ambang keputusan, atau problem di exit/Guardian) masih menggerus performa.

**Status live**: TETAP paper (atas perintah eksplisit user, bukan dampak temuan ini) sampai ada stack yg lolos gerbang jujur penuh (OOF absolut + holdout tersegel).

**Artefak.** `pipeline/experiments/dualbin_universe/fase111_fetch_intrabar_m5_train.py`, `fase112_benchmark_simulasi_entry_paksa.py`, `fase113_gerbang_oracle_w3.py` (+`models/runs/dualbin_gerbang_oracle_w3_20260806/oracle_w3.parquet`), `fase114_label_eksekusi_jujur.py` (+`data/training/label_eksekusi_jujur/`), `fase115_diagnosa_label_lama_vs_baru.py` (+`models/runs/dualbin_diagnosa_label_lama_vs_baru_20260806.json`), `fase116_oof_retrain_label_jujur.py` (+`models/runs/dualbin_oof_retrain_label_jujur_20260806.json`). `app/services/execution_sim.py::simulasi_entry_paksa`, `M5TidakTersedia`.

---

## 2026-08-06 — Fase 117-119: mekanisme order (limit/exchange-stop) GAGAL, akar sesungguhnya = `breakeven_lock` memotong keputusan Guardian sendiri

**Pemicu.** Setelah Fase 107-116 menunjukkan model live (`long22f_swing`/`short20f_swing`) rugi di bawah eksekusi jujur, user minta gali lebih dalam: (1) apakah ganti mekanisme order (limit alih-alih market) bisa mendekatkan ke performa idealisasi lama, (2) SL monitor 5-menit dicoret dari opsi ("jangan pakai SL monitor"), (3) akhirnya user memberi petunjuk kuat dari pengalaman live: "awal deploy performanya gak jelek-jelek amat... ada yang beberapa puluh persen loss tapi gak langsung SL, bisa berbalik TP... setelah perubahan-perubahan SL lebih cepat kena."

**Fase 117 — entry via LIMIT order resting (1 jam) + TP via limit (wick M5), SL/Guardian tetap realistis.** Dites jujur pakai data M5 asli (bukan idealisasi) di holdout, model & window sama dgn Fase 109. Hasil: PF **0,6519** (LEBIH JELEK dari Mode A 0,7325). Kanal `monitor_sl` (622 trade) menyerap hampir semua rugi (-$1.983,17) sementara kanal `h1` (yg lolos dari SL cepat) UNTUNG (+$940,15). Penjelasan: order limit di entry kena **adverse selection** — sinyal momentum yg BENERAN bagus harganya lanjut jalan (limit gak pernah keisi, trade bagus HILANG), sementara sinyal yg harganya sempat balik dulu (baru limit keisi) justru tanda lemah/mau gagal. **Ide DITOLAK utk entry.**

**Fase 118 — hapus jeda 5 menit KHUSUS exit-jalur-H1 (entry tetap 5 menit, krn OI/LSR butuh nunggu; exit TIDAK butuh OI/LSR, dicek langsung `guardian_features.json` 28 fitur, TIDAK ADA open_interest/long_short_ratio).** Hasil: PF 0,7598→0,7642 (+$14 doang). **Nyaris tidak membantu.** Root cause SESUNGGUHNYA (dibaca dari kode, bukan diukur): `paper_engine.py::_manage_bar` baris 378, `pos.raw_exit = pos.sl_price` -- versi "idealisasi" SELALU membukukan SL persis di level SL, PADAHAL syarat pemicunya `close_j <= sl_price` (candle bisa closed JAUH di bawah level itu). Overshoot ini TIDAK bisa dihindari kalau SL cuma dicek 1x/jam (baik App-side polling maupun order stop bursa asli MEMBUTUHKAN deteksi lebih cepat dari 1 jam utk menutup overshoot ini) -- user eksplisit menolak deteksi lebih cepat dari 1 jam apapun caranya, jadi PF idealisasi (1,84, "Mode C" Fase 109) DIKONFIRMASI TIDAK REALISTIS dicapai, bukan target yg valid.

**Fase 119 — titik balik: uji apakah mekanisme Guardian yg BERUBAH (bukan cara eksekusi) yg jadi penyebab, sesuai intuisi user dari pengalaman live.** Bandingkan config Guardian ASLI (sblm 08-03: `momentum_floor_frac=0,30`, near-TP floor & breakeven-lock MATI) vs SEKARANG (`momentum_floor_frac=0,10` + near-TP floor ON + breakeven-lock ON), model & holdout & harness SAMA (SL monitor mati, sesuai keputusan user):

| Config | Trade | PF | PnL |
|---|---:|---:|---:|
| A -- Asli (sblm 08-03) | 1.639 | 0,8824 | -$260,25 |
| B -- + near-TP floor saja | 1.647 | 0,8671 | -$297,75 |
| C -- + breakeven-lock saja | 3.143 | 0,7598 | -$558,53 |
| D -- Sekarang (keduanya) | 3.143 | 0,7598 | -$558,53 (identik C -- near-TP floor jadi tak relevan lagi) |

**Near-TP floor cuma nyumbang rugi kecil (-$37,50).** **Breakeven-lock adalah penyebab dominan**, TAPI bukan lewat "SL lebih cepat kena" -- SL biasa (`LOSS`) malah HAMPIR SAMA (520→506 trade, -$3,47→-$3,37/trade). Yang terjadi: breakeven-lock MEMOTONG DULUAN 82% dari semua trade SEBELUM Guardian sempat memutuskan sendiri (`GUARDIAN_EXIT`/`GUARDIAN_MOMENTUM_EXIT`, dulu rata2 +$1,44/+$3,34 per trade, kini tinggal 12+4 trade) -- breakeven-lock menguncinya di rata2 cuma +$0,52/trade, di 2.569 trade sekaligus. Bukan "mencegah rugi", tapi **memotong untung besar jadi untung kecil** secara masif. Jumlah trade juga hampir 2x lipat (1.639→3.143) krn posisi ditutup lebih cepat = slot koin lebih cepat bebas utk entry baru.

**Catatan penting**: config ASLI (A) pun, diukur JUJUR (bukan idealisasi), SUDAH rugi (PF 0,88) -- ingatan user "awal deploy gak jelek amat" kemungkinan berdasar angka lama yg dihitung idealisasi (skrg terbukti optimistis), BUKAN bukti config asli itu profitable. Tapi perbandingan apples-to-apples (sama2 diukur jujur) tetap membuktikan breakeven-lock aktif MEMPERBURUK, bukan cuma tidak membantu -- padahal waktu deploy (`project-dualbin-guardian-breakeven-lock-deploy.md`) dia LOLOS OOF+OOS pakai metode LAMA (idealized `replay_arrays()`), metode yg SESI INI terbukti sistematis optimistis.

**BELUM ada keputusan produksi** (matikan/tune ulang breakeven-lock) -- perlu dibahas dgn user dulu, keduanya (near-TP floor & breakeven-lock) tetap LIVE di config saat ini.

**Artefak.** `pipeline/experiments/dualbin_universe/fase117_mode_d_limit_order.py`, `fase118_mode_f_exit_tanpa_delay.py`, `fase119_dampak_floor_breakeven.py` (semua +log).

---

## 2026-08-06 — Fase 120-129: sweep ambang breakeven + lookback swing H4 -- kandidat 3/3 window BERKALI-KALI runtuh, akhir ketemu BUG FITUR yg membatalkan seluruh narasi lookback

**Fase 120 -- sweep halus ambang `breakeven_lock_mfe_pct` (OOF, near-TP floor tetap ON).** Titik terbaik konsisten (kasar & halus): **mfe_pct=1,5% / buffer=3%** -- PF 0,9111/-$198,45 (holdout), jauh lebih baik dari 0,5% skrg (-$558,53) TAPI tetap rugi.

**Fase 121 -- sweep selektivitas entry (TL/TS) di atas titik Fase 120.** TL=0,93/TS=0,90 tembus **PF 1,2271/+$48,85** (n=203, sampel kecil) -- KANDIDAT pertama malam ini yg untung, tapi ditemukan lewat **tuning di HOLDOUT** (kesalahan proses diakui eksplisit, lihat entri berikutnya).

**Kesalahan proses diakui (sesi ini)**: Fase 120+121 dijalankan di HOLDOUT tersegel, padahal itu PENCARIAN parameter (bukan diagnosa) -- melanggar "jangan tune berdasarkan holdout". Diperbaiki: sisa investigasi (Fase 122+) dipindah SELURUHNYA ke OOF (W1/W2/W3), holdout tidak disentuh sampai kandidat final dikunci.

**Fase 122 (OOF) -- B1/B2: benarkah SL kena krn market volatile lalu balik ke TP?** 1.434 trade kena SL biasa; 18,55% ($828 dari $6.094 total rugi SL) memang balik ke TP kalau dibiarkan -- MINORITAS. **B2 kebalikan dugaan**: kuartil volatilitas RENDAH justru paling sering balik ke TP (24,79%), kuartil TINGGI paling jarang (12,64%) & paling mahal per kejadian. "Lebarin SL pas volatile" TIDAK didukung data.

**Fase 123 -- sweep `lookback` di `detect_h4_swing_points` (default 3), OOF, model+config SEKARANG.** 2,3,5,7,10 lalu 12,15,20,25,30: rugi mengecil TAJAM seiring lookback naik (3→2.923,70 rugi; 10→411,60 rugi total 3 window), plateau/berisik >15 (trade makin sedikit, 150-300/window). **[LIHAT KOREKSI BUG DI BAWAH -- kesimpulan ini TERCAMPUR bug fitur, jangan dipercaya mentah.]**

**Fase 124 -- gabung lookback (15,20 dikunci user) x sweep breakeven.** lookback=20+breakeven=0%: W1 lolos (+$24), W2 nyaris lolos (-$4,87, PF 0,9852), W3 lolos kuat (+$173) -- PALING DEKAT 3/3 saat itu.

**Fase 125 -- baseline LGBM MURNI tanpa Guardian sama sekali** (SL/TP/timeout doang, atas arahan user "sistematis, LGBM dulu"), sweep lookback(10,15,20) x max_hold(24,36,48,60): tidak ada kombinasi lolos 3/3; terbaik lookback=20/max_hold=24 (+$157,18, 2/3). Guardian (near-TP floor) terbukti nambah nilai di atas sinyal mentah (+$157 murni vs +$193-256 dgn Guardian).

**Fase 126 -- Guardian ON + lookback=20 + max_hold=24 (kombinasi belum pernah dicoba).** **LOLOS 3/3 window OOF pertama kali malam ini**: W1 +$54,33, W2 +$19,70, W3 +$181,82, total +$255,85.

**Fase 127 -- HOLDOUT SEKALI (sentuhan pertama), kandidat lookback=20+max_hold=24.** **GAGAL**: PF 0,9802/-$7,64 keseluruhan, 2/4 bulan gagal (Mei PF 0,52 terparah). User lalu menyoroti: **volume trade jatuh 91,6%** (29,56→2,49 trade/hari) -- sampel terlalu tipis utk dipercaya, kemungkinan besar penyebab OOF "lolos" tapi holdout gagal.

**Fase 128 -- geser ke lookback moderat (3 default, 5,6,7,10,15), max_hold=24, scorecard lengkap (WR/PF/LONG-SHORT/peak/MaxDD).** lookback=5 & 7 (juga 6) lolos 3/3 window dgn volume lebih sehat (5,5-8/hari) -- lookback=5 kandidat terkuat: PF gab 1,0939, PnL +$449,66, MaxDD/peak rasio 57% (terbaik). **Pola LONG vs SHORT konsisten di SEMUA lookback**: LONG selalu PF<1 (beban), SHORT selalu PF>1,1 (kuat) -- temuan robust lintas-konfigurasi.

**BUG DITEMUKAN sebelum holdout sentuhan kedua (audit atas permintaan eksplisit user "pastikan bebas leakage")**: `dist_liq_50x_long`/`dist_liq_50x_short` (2 dari 22/20 fitur model entry LONG/SHORT) dihitung dari `h4_swing_low*0,98`/`h4_swing_high*1,02` (rumus IDENTIK live, `app/core/features.py` baris 1863-1870, diverifikasi) -- **TIDAK PERNAH ikut dihitung ulang di Fase 123-128** saat lookback disweep, cuma `h4_swing_high`/`h4_swing_low` mentah yg diupdate. BUKAN leakage (tidak ada info masa depan dipakai), tapi inkonsistensi fitur-vs-label yg mengotori SEMUA hasil sejak Fase 123.

**Fase 129 (dibatalkan sebelum jalan, terganti retest) -- setelah fix (`perbarui_fitur_turunan_swing`), OOF lookback=5 diulang: HASIL BERBALIK TOTAL.**

| | Sebelum fix (fitur basi) | Sesudah fix (fitur benar) |
|---|---|---|
| W1 | Lolos, PF 1,02 | GAGAL, PF 0,8826, -$335,33 |
| W2 | Lolos, PF 1,15 | GAGAL, PF 0,9292, -$187,90 |
| W3 | Lolos, PF 1,13 | Lolos, PF 1,1497, +$321,95 |
| Gabungan (4.506 trade) | PF 1,0939, **+$449,66** | PF 0,9737, **-$201,29** |

**Kesimpulan kritis: seluruh narasi "lookback lebih panjang = lebih baik" (Fase 123-128) TERCAMPUR/kemungkinan besar SEBAGIAN BESAR ARTIFAK bug fitur ini, bukan sinyal asli murni.** Holdout sentuhan kedua (lookback=5) DIBATALKAN sebelum dijalankan -- tepat waktu, tidak jadi menyentuh holdout dgn evaluasi cacat. Ketemu SEBELUM holdout krn user eksplisit minta audit "leakage, fitur, environment masuk akal" sebelum lanjut -- kalau tidak diminta, hampir pasti kebablasan.

**Fase 128 diulang PENUH dgn fitur benar (`perbarui_fitur_turunan_swing`), 7 nilai lookback (3,5,6,7,10,15,20), OOF W1/W2/W3, scorecard lengkap:**

| Lookback | W1 | W2 | W3 | PF gabungan | PnL gabungan | Lolos 3/3? |
|---|---|---|---|---:|---:|---|
| 3 | GAGAL 0,964 | Lolos 1,125 | GAGAL 0,993 | 1,024 | +$191,74 | Tidak |
| 5 | GAGAL 0,883 | GAGAL 0,929 | Lolos 1,150 | 0,974 | -$201,29 | Tidak |
| 6 | GAGAL 0,895 | GAGAL 0,947 | Lolos 1,122 | 0,979 | -$155,84 | Tidak |
| **7** | GAGAL 0,933 | Lolos 1,039 | Lolos 1,100 | **1,018** | **+$120,34** | Tidak |
| 10 | GAGAL 0,927 | GAGAL 0,939 | Lolos 1,041 | 0,965 | -$214,76 | Tidak |
| 15 | GAGAL 0,896 | GAGAL 0,848 | Lolos 1,070 | 0,931 | -$340,35 | Tidak |
| 20 | GAGAL 0,914 | GAGAL 0,887 | GAGAL 0,982 | 0,926 | -$302,25 | Tidak |

**TIDAK ADA lookback yg lolos gerbang absolut (untung di SEMUA window terpisah) setelah fitur dibetulkan.** Narasi "lookback lebih panjang = lebih baik" (Fase 123-128 versi lama) TERBUKTI sebagian besar artefak bug fitur -- pola aslinya jauh lebih berantakan: W1 HAMPIR SELALU gagal apapun lookback-nya, W3 hampir selalu lolos, W2 cuma lolos di 2/7 titik (lookback 3 & 7).

**Temuan yg BERTAHAN, tidak terkait bug (terkonfirmasi tetap sama arah sebelum & sesudah fix):** LONG konsisten lemah (PF<1, kadang ekstrem -- lookback 15/20 di W2: PF LONG 0,54-0,55) di HAMPIR SEMUA window/lookback; SHORT konsisten kuat (PF>1,1) di HAMPIR SEMUA kombinasi. Ini sinyal ASLI (robust lintas-konfigurasi), bukan artefak -- kandidat paling menjanjikan utk digali lanjut, BUKAN lookback.

**Keputusan penutup sesi**: lookback=7 (PF gabungan tertinggi di antara yg gagal, +$120,34) **TIDAK dipakai sbg kandidat deploy/holdout** -- eksplisit dikonfirmasi user sbg TITIK AWAL riset lanjutan saja (fokus asimetri LONG/SHORT), bukan validasi. Holdout TIDAK disentuh lagi malam ini (2 sentuhan sejauh ini: lookback=20 gagal Fase 127, lookback=5 dibatalkan sblm jalan Fase 129 krn bug fitur ketemu duluan).

**Artefak.** `pipeline/experiments/dualbin_universe/fase120_sweep_breakeven_threshold.py`, `fase121_sweep_selektivitas_entry.py` (holdout, JANGAN dipakai acuan lagi), `fase122_diagnosa_sl_volatilitas_oof.py` (+`models/runs/dualbin_fase122_diagnosa_sl_oof.parquet`), `fase123_sweep_lookback_swing_h4.py` (KOTOR, bug fitur), `fase124_gabung_lookback_breakeven.py` (KOTOR), `fase125_lgbm_murni_tanpa_guardian.py` (KOTOR), `fase126_guardian_lookback20_maxhold24.py` (KOTOR), `fase127_holdout_final_lookback20_maxhold24.py` (KOTOR, tapi kesimpulan "gagal holdout" tetap berdiri terlepas bug krn arahnya sama/lebih buruk), `fase128_lookback_moderat_10_15.py` (versi FINAL sudah pakai `perbarui_fitur_turunan_swing` -- SSOT utk lanjutan), `fase129_holdout_final_lookback5_fitur_konsisten.py` (ditulis, TIDAK dijalankan -- template holdout kalau ada kandidat bersih nanti).

**Fase 130-136 -- ronde kedua pencarian kandidat lb7 (lookback=7, PF gabungan 1,018 dari Fase 128), semua OOF-only, TIDAK ada holdout disentuh:**

- **Fase 130** SHORT-only vs gabungan: SHORT-only total +$547,51 (2/3 window) vs gabungan +$120,34 (2/3), tapi SHORT-only bikin W1 LEBIH BURUK, bukan solusi bersih.
- **Fase 131** lookback terpisah per arah (SHORT tetap 7, LONG disweep 3/5/7/10/15/20): terbaik LONG=10 (+$131,06), tetap 0/3 window -- W1 gagal di SEMUA 6 kombinasi.
- **Fase 132** retrain Guardian khusus lb7 (12 bulan trajectory data, pipeline SSOT `train_guardian_with_oof`, simpan ke `models/runs/dualbin_guardian_lb7_20260806/` -- BUKAN timpa produksi): PF~0,84 PnL -$1.201,27, 0/3 window -- LEBIH BURUK dari Guardian lama (1,018/+$120,34). **CATATAN PENTING (ditemukan Fase C di bawah): retrain ini memuat fitur ETF (`etf_gbtc_change_usd`/`etf_total_change_usd`) LANGSUNG dari file TIER1 tanpa merge ETF asli -- kolom itu TERNYATA nol/placeholder total di TIER1 (lihat audit di bawah). Kesimpulan "lebih buruk krn data kurang" jadi TIDAK PASTI -- bisa juga krn Guardian dilatih seolah ETF selalu nol, padahal ablation sebelumnya (`project-dualbin-guardian-etf-real-fetch.md`) menunjukkan ETF asli > ETF nol. BELUM diulang dengan ETF asli.**
- **Fase 133** LGBM standalone (tanpa Guardian) lb7: 12 bulan PnL -$1.137,05 (0/3); 4 tahun (kontaminasi LSR palsu Apr-Nov2021 belum diketahui saat itu) PnL -$857,32 (0/3, membaik tapi tetap gagal).
- **Fase 134** SL/TP murni ATR (1,5x & 2,0x, simetris, tanpa swing) menggantikan label swing: label mendekati acak (positive_rate rerata 0,496, cuma 1,67% sampel >0,80) -- model benar mendeteksi tak ada sinyal, trade nyaris nol (24 & 16 trade total). **Ditutup: SL/TP murni ATR bukan arah yg layak**, kembali ke label berbasis swing.
- **Fase 135** sweep swing berbasis H1 (bukan H4), lookback 10/15/20/30/40, latih 4 tahun: SEMUA GAGAL (0-1/3 window), terbaik lookback=10 di -$963,98 -- lebih buruk dari H4 lb7 (-$857,32). H4 tetap lebih baik dari H1 utk deteksi swing.
- **Fase 136** diagnosa langsung AUC + feature importance LONG vs SHORT (lb7, latih 4 tahun, per-window): AUC val bagus utk keduanya (LONG 0,906-0,925, SHORT 0,931-0,943) -- **BUKAN masalah kurang sinyal**. Tapi gap AUC train-val LONG (+0,021 s/d +0,037, MEMBURUK tiap window) 2-6x lebih besar dari SHORT (+0,002 s/d +0,015, STABIL) -- **LONG overfit jauh lebih parah, makin parah di window terbaru (W3)**. Feature importance nyaris identik LONG/SHORT (top: `dist_liq_50x_*`, `ofi_z_score`, `log_ret_20`, `long_short_ratio`, `atr_percentile_h1`) -- bukan soal fitur beda, soal generalisasi.

**Audit data sintetis (permintaan eksplisit user "cek fitur dominan bebas leakage... bukan data original Binance Vision") -- opsi C dari 3 opsi lanjutan (A: latih ulang jendela bersih, B: perbaiki overfitting LONG, C: audit fitur lain):**

1. `long_short_ratio` (dicek sesi sebelumnya): PALSU/kosong 17/18 koin sebelum **2021-11-01** (BTC sejak Agu 2020) -- lihat `project-dualbin-longshortratio-fake-data-before-nov2021.md`.
2. **`funding_rate`** (dipakai Guardian produksi): titik mulai data asli **PER KOIN BERBEDA-BEDA**, dari 2020-01 (BTC, LINK) sampai 2022-02 (BCH) -- BUKAN batas seragam spt LSR. Training window manapun yg pakai batas tunggal berisiko mencemari sebagian koin.
3. **`open_interest`** (TIDAK dipakai LONG/SHORT/Guardian produksi saat ini): titik mulai 2020-01 (LTC/ETC/BCH) s/d **2022-01** utk 11/18 koin (ETH,BNB,SOL,XRP,ADA,DOGE,AVAX,LINK,DOT,TRX,NEAR). Aman krn tidak dipakai sekarang, tapi WAJIB dicek ulang kalau mau dipakai lagi di eksperimen fitur.
4. **`etf_total_change_usd` / `etf_gbtc_change_usd`** (dipakai Guardian produksi): **NOL/placeholder total di file TIER1 utk SEMUA 18 koin, SEPANJANG histori** -- bukan "kurang panjang", tapi memang tidak pernah diisi data asli di file ini. Ini SESUAI desain (`core/features.py` baris ~2090-2100: `engineer_features()` zero-fill ETF sbg fallback, data ASLI di-overwrite belakangan oleh `DataService.compute_features_batch()` di live_dualbin_ft -- TIDAK pernah ditulis balik ke TIER1). **Konsekuensi: skrip riset APAPUN yg load fitur ETF langsung dari TIER1 (spt Fase 132 di atas) otomatis melatih dgn ETF=0 di SELURUH data, termasuk periode pasca-2024 yg sebetulnya ada data ETF asli.**
5. `btc_dominance`, `fear_greed`: kolom mati, TIDAK dipakai model manapun (LONG/SHORT/Guardian saat ini), nol/placeholder total -- aman diabaikan.
6. `ofi_z_score`, `ofi_h4_delta`, `ofi_raw`, `cvd`, `cvd_slope_h4` (fitur top-importance LONG & SHORT): **BERSIH**, data asli dari Jan-Okt 2020 tergantung koin -- jauh sebelum batas training manapun yg realistis. Ofi/CVD **bukan** sumber masalah LONG.

**Fase 137 -- opsi A (setelah audit): ulang uji "4 tahun latih" TAPI jendela di-clamp supaya tidak pernah mundur sebelum 2021-11-01** (bukan 12 bulan bersih, bukan 4 tahun tercemar -- pakai SEMUA histori bersih yg tersedia, maks 48 bulan). LGBM standalone lb7, OOF W1/W2/W3:

| Jendela | Lama latih | PF | PnL | LONG PF | SHORT PF |
|---|---|---:|---:|---:|---:|
| W1 | 41,0bln dari 2021-11-01 | 0,8653 | -$386,27 | 0,90 | 0,83 |
| W2 | 45,0bln dari 2021-11-01 | 0,9111 | -$242,89 | 0,76 | 1,12 |
| W3 | 48,0bln dari 2021-12-01 | 0,9672 | -$78,49 | 0,80 | 1,27 |
| **Total** | | | **-$707,65** | | |

Progresi PnL gabungan: 12bln bersih -$1.137,05 → 4th tercemar -$857,32 → 4th **bersih** -$707,65 -- **membaik monoton dgn data lebih bersih & lebih panjang**, mengkonfirmasi kontaminasi LSR memang sebagian menahan hasil. **Tetap gagal gerbang (0/3 window)**, tapi W3 nyaris impas (PF 0,9672). **Pola LONG/SHORT makin tajam**: LONG PF 0,76-0,90 di SEMUA window (masih beban), SHORT PF 0,83→1,12→1,27 (TREN NAIK, dua window terakhir sudah >1) -- makin menguatkan Fase 136: soal LONG overfitting, bukan window/data kotor.

**Artefak lanjutan.** `fase130_short_only_vs_gabungan.py`, `fase131_lookback_terpisah_per_arah.py`, `fase132_retrain_guardian_lb7.py` (+`models/runs/dualbin_guardian_lb7_20260806/`, hasil DIRAGUKAN krn ETF nol, belum diulang), `fase133_cek_lgbm_standalone_lb7.py` (ada bug var `LOOKBACK` tak terdefinisi di versi tersimpan -- JANGAN dijalankan ulang apa adanya), `fase134_sl_atr_murni.py`, `fase135_sweep_swing_h1.py`, `fase136_diagnosa_long_lemah.py`, `fase137_lgbm_window_bersih.py` (SSOT terbaru utk standalone lb7 + jendela latih bersih).

**Fase 138 -- Opsi B: grid LONG (regularisasi x jumlah fitur x panjang jendela latih), SHORT tetap.** 27 kombinasi (3 level regularisasi baseline/moderat/ketat x 3 level fitur full23/top12/top8 x 3 panjang jendela bersih_maks/12bln/6bln) x 3 window OOF = 81 fit LONG, replay gabungan LONG+SHORT penuh tiap kombinasi (bukan skor LONG terisolasi -- SHORT & LONG terkopel via aturan konflik margin).

**Hasil: TIDAK ADA kombinasi yg lolos gerbang 3/3 -- maksimal 1/3 window (dua kombinasi lolos W3 SAJA, PF 1,01-1,03, tipis).** Bahkan di kedua kasus "lolos" itu, LONG sendiri PF masih 0,78-0,79 -- yg bikin gabungan lolos adalah SHORT (sudah kuat), bukan LONG membaik. LONG PF tetap 0,64-0,92 di SEMUA 81 kombinasi, semua window -- regularisasi & pengurangan fitur TIDAK memperbaikinya sama sekali.

Temuan per sumbu:
- **Regularisasi lebih ketat (num_leaves/max_depth/min_child_samples lebih kecil) TIDAK membantu, malah SEDIKIT LEBIH BURUK** di kebanyakan kombinasi fitur/jendela (mis. full23/bersih_maks: PnL total baseline -$822 vs moderat -$881 vs ketat -$899). LONG PF per-window juga nyaris tak bergerak (mis. W1: 0,88/0,88/0,88 di 3 level regularisasi) -- overfitting LONG BUKAN sekadar "model kelewat kompleks" dlm arti mekanis biasa.
- **Kurangi fitur (23->12->8) TIDAK membantu** -- top12 kadang malah lebih buruk dari full23 (mis. baseline/12bln: full23 -$911 vs top12 **-$1.181** (terburuk di seluruh grid) vs top8 -$970).
- **Jendela latih lebih pendek (6 bulan, data terbaru saja) SATU-SATUNYA sumbu yg konsisten membantu** -- hampir di semua kombinasi reg/fitur, 6bln > bersih_maks(sampai 48bln) > umumnya juga > 12bln (12bln sering JADI YG TERBURUK, bukan di tengah -- pola tak monoton, kemungkinan ada rentang buruk spesifik ~12 bulan ke belakang yg tercakup di jendela itu tapi ter-encer di jendela lbh pendek/lbh panjang). Kombinasi terbaik keseluruhan: `top12/6bln/baseline` PnL total -$677,58 (vs baseline Fase137 -$707,65 -- beda ~$30, DALAM lantai derau ±$80, BUKAN perbaikan nyata).

**Kesimpulan: Opsi B (regularisasi, fitur, jendela latih -- pendekatan model-complexity klasik) GAGAL memperbaiki LONG.** Sinyal LONG tampaknya bukan soal model kelewat rumit menghafal noise, tapi lebih ke pola yg genuinely time-varying/regime-dependent (konsisten dgn indikasi "jendela lebih baru sedikit lebih baik" tapi efeknya kecil & tak cukup). Dibutuhkan pendekatan berbeda dari model-complexity tuning kalau mau lanjut ke LONG secara langsung.

**Artefak.** `fase138_opsi_b_grid_long.py` (+`models/runs/dualbin_fase138_opsi_b_grid.parquet`, 81 baris hasil per kombinasi -- 2 bug kecil sempat ketemu & diperbaiki: unpacking `product()` salah jumlah variabel, kolom `high`/`low`/`atr_14_h1` sempat tak ikut ter-subset shg `bars_dari_panel` crash).

**Fase 139 -- HOLDOUT (sentuhan ke-3 sesi ini), lb7 standalone (tanpa Guardian, config Fase137: latih 48bln di-clamp >=2021-11-01), atas permintaan eksplisit user MESKI OOF gagal 3/3.** Ditanya dulu via konfirmasi eksplisit sblm jalan (sesuai [[feedback-oof-gates-before-oos]]/[[no-oos-without-approval]]) -- user pilih tetap lanjut, murni sbg info tambahan, BUKAN validasi resmi.

| Bulan | Trade | WR | PF | PnL | Status |
|---|---:|---:|---:|---:|---|
| 2026-04 | 444 | 60,6% | 1,3011 | +$149,10 | Lolos |
| 2026-05 | 411 | 49,9% | 0,6258 | -$247,11 | **Gagal parah** |
| 2026-06 | 291 | 55,3% | 1,0364 | +$16,42 | Lolos |
| 2026-07 | 352 | 57,7% | 1,1475 | +$58,34 | Lolos |
| **Total** | **1.498** | **55,9%** | **0,9884** | **-$23,24** | **Gagal (1/4 bulan)** |

LONG: 679 trade, WR 54,5%, PF 0,9671, PnL -$30,41. SHORT: 819 trade, WR 57,1%, PF 1,0066, PnL +$7,17.
Peak equity $186,81, MaxDD -$369,56.

**Catatan penting**: PF total 0,9884 jauh lebih dekat ke impas drpd ketiga window OOF (0,87/0,91/0,97) -- dalam lantai derau ±0,02 yg sudah diukur sblmnya ([[feedback-ukur-lantai-derau-sebelum-menafsir]]), LONG holdout (PF 0,97) bahkan jauh lebih baik dari LONG OOF manapun (0,76-0,90). TAPI **gerbang keseluruhan tetap GAGAL** -- 1 bulan (Mei) sendirian rugi -$247 dan menyeret 3 bulan lain yg positif jadi total minus. Ini TIDAK mengubah kesimpulan: OOF sudah gagal 3/3 sblm holdout dijalankan, holdout cuma info tambahan spt yg disepakati di depan, BUKAN alasan meninjau ulang kelayakan lb7.

**Breakdown per-bulan per-arah** (diminta user): Mei rugi di KEDUA sisi (LONG -$134, SHORT -$113) -- bulan itu buruk secara umum, bukan masalah LONG saja. Pola tiap bulan BERGANTIAN, bukan LONG selalu kalah: April LONG sangat kuat (PF 2,22, WR 66%) tapi SHORT rugi tipis; Juni kebalikannya (SHORT PF 1,51, LONG PF 0,75). Beda dgn pola OOF (LONG selalu lemah, SHORT selalu kuat, konsisten di 3 window) -- holdout menunjukkan gambaran lebih berantakan/bergantian. Dicatat sbg petunjuk BELUM digali, bukan kesimpulan baru (lihat [[feedback-protokol-evaluasi-berkelanjutan]]).

**Artefak.** `fase139_holdout_lb7_standalone.py` (+`models/runs/dualbin_fase139_holdout_lb7_standalone_20260806.parquet`).

---

## 2026-08-06 — Audit lanjutan: bug fetch `funding_rate`, `dist_liq_50x_*` BUKAN order book, katalog Binance Vision

**Pemicu.** User minta telusuri kenapa `funding_rate` kita punya gap vs Binance Vision (temuan sesi audit sebelumnya), sekalian audit semua fitur turunan lain -- khusus curiga `dist_liq_50x_long/short` ("harusnya dari order book, tebal bid/ask, support/resistance") dan `ofi_z_score`.

**Bug `funding_rate` ditemukan (BELUM diperbaiki).** `core/fetchers.py::fetch_funding_rate()` incremental fetch cuma maju ke depan (`start_ms = max(start_ms, last_ms + step_ms)` begitu ada `existing`), tak pernah backfill mundur. Dicek langsung ke Binance Vision (S3 listing `data/futures/um/monthly/fundingRate/{coin}/`): data ASLI tersedia sejak listing tiap koin (BTC/XRP/BCH sejak 2020-01, DOT sejak 2020-08) -- jauh lebih awal dari TIER1 kita (XRP genuine baru 2021-08, BCH baru 2022-02, gap 19-25 bulan). Klines sudah termigrasi ke Vision (lengkap), funding_rate TIDAK. Beda dgn `long_short_ratio` yg REST-nya genuinely hard-limited 500 jam oleh Binance sendiri (bukan bug) -- funding_rate REST-nya sebenarnya BISA mundur jauh, cuma pipeline kita tak pernah memanfaatkannya.

**`dist_liq_50x_long/short` dikonfirmasi BUKAN dari order book** (`core/features.py` baris 1898-1909, komentar source sendiri bilang "Liquidation Cascade Proxy"): `liq_50x_long = h4_swing_low*0,98`, `liq_50x_short = h4_swing_high*1,02` -- asumsi sembarang "leverage 50x = 2% gerak = liquidasi", dihitung dari swing candle H4 (bukan order book/likuidasi asli). Datanya sendiri aman (dari candle asli), tapi validitas konsep sbg "jarak ke likuidasi" lemah. Fitur ini importance #1 LONG (9,16%) dan #1 SHORT (9,87%) -- dampak potensial besar kalau diganti.

**`ofi_z_score`/`cvd` dikonfirmasi**: dari `taker_buy_volume`/`taker_sell_volume` bawaan kline Binance (bukan endpoint terpisah) -- data ASLI & aman, tapi itu agregat volume transaksi, BUKAN kedalaman order book (beda konsep dari yg dicurigai user, tapi datanya sendiri tidak bermasalah).

**Katalog Binance Vision dicek langsung** (S3 listing `data/futures/um/{daily,monthly}/`): tersedia klines, fundingRate, aggTrades, trades, bookTicker, markPriceKlines, indexPriceKlines, premiumIndexKlines (daily+monthly), plus bookDepth & metrics (CUMA daily). **TIDAK ADA arsip likuidasi historis** (`liquidationSnapshot`/`forceOrder`) di Vision utk USD-M futures -- data likuidasi asli cuma live via WebSocket, tak bisa ditarik mundur. `bookDepth` (order book asli, kandidat pengganti `dist_liq_50x_*`) baru tersedia sejak **2023-01-01** (BTCUSDT) -- jauh lebih pendek dari jendela latih yg dibutuhkan (2021/2022+).

**Audit fitur turunan lain (top-15 LONG & SHORT)**: dari 26 fitur unik, cuma `long_short_ratio` yg genuinely bermasalah datanya (sudah diketahui). Sisanya semua dari candle Binance asli (OHLCV, panel lintas-koin, atau volume taker buy/sell) -- aman dari sisi provenance data, walau `dist_liq_50x_*` lemah secara konsep (bukan soal data palsu).

**Status: BELUM ada perbaikan dieksekusi** -- user eksplisit minta `long_short_ratio` (batas Nov2021 sudah confirmed dinding Binance, tak bisa diperpanjang) dan `dist_liq_50x_long/short` (perlu pendekatan baru, opsi blm dievaluasi) diperbaiki. Rencana perbaikan BELUM ditulis/disetujui.

**Artefak.** Tidak ada script baru (audit via WebFetch S3 listing langsung + baca kode `core/fetchers.py`/`core/features.py`). Memori baru: `project-dualbin-funding-rate-fetch-gap-bug.md`, `project-dualbin-dist-liq-proxy-not-orderbook.md`, `reference-binance-vision-data-catalog.md`.

**Cek exchange lain utk long_short_ratio/OI lebih panjang (OKX, Bybit).** OKX (`rubik/stat/contracts/long-short-account-ratio-contract`, dicoba query langsung Jan2020/Jan2022/Jan2025) -- SEMUA kosong, cuma data terbaru yg keluar; retensi jauh lebih pendek dari Binance Vision. Bybit (`public.bybit.com`) -- arsip bulk publiknya cuma kline/premium_index/spot_index/trading/spot, TIDAK ADA kategori open interest/long-short ratio sama sekali. **Kesimpulan: tidak ada exchange dgn data posisi trader lebih panjang dari Binance** -- Des2021 tetap dinding final, tidak bisa diperpanjang dari sumber lain.

**Cek ketersediaan macro (BTC dominance, stablecoin mcap, Fear&Greed) -- permintaan user.** `core/fetchers.py` SUDAH punya `fetch_btc_dominance()` (CoinGecko) & `fetch_fear_greed()` (Alternative.me), dan hasil fetch-nya ADA & bagus: `fear_greed.parquet` 2018-02-01 s/d 2026-08-03 (3.102 baris), `btc_dominance.parquet`/`fear_greed_index.parquet` 2020-01-01 s/d 2026-04-01 harian (2.283 baris, 0 NaN). **Tapi 0% terisi di TIER1** -- disimulasikan manual merge-nya (`_load()`+`ffill_macro()` dari `clean.py`), TERNYATA bekerja sempurna (54.736 baris, 100% non-null) saat dites langsung. Root cause: **staleness murni** -- `BTCUSDT_clean.parquet` mtime 1 Agustus, macro source mtime 2 Agustus (SETELAH clean.py terakhir jalan). Bukan bug kode, cuma pipeline `clean->engineer` belum di-run ulang sejak macro data itu ada. **Stablecoin market cap: TIDAK ADA fetcher sama sekali**, perlu ditulis baru kalau mau.

---

## 2026-08-06 — Eksekusi perbaikan (fear_greed, long_short_ratio, dist_liq_50x): SEBAGIAN, terhenti krn 2 temuan besar + krisis memori sistem

**Konteks.** User setuju eksekusi 3 perbaikan sekaligus: (1) fear_greed + btc_dominance masuk TIER1, (2) tegakkan batas data asli `long_short_ratio`, (3) ganti `dist_liq_50x_long/short` pakai bookDepth asli (user pilih eksplisit: terima jendela latih lebih pendek 2023+ demi data order book sungguhan).

**KRITIS #1 -- `btc_dominance_pct` ternyata HARDCODE PALSU** (`core/fetchers.py`, baris ~419 & ~350): `{2020:63.0, 2021:45.0, ...}.get(year, 55.0)` -- tabel tebakan per-tahun, BUKAN dihitung dari market cap sungguhan (endpoint historis CoinGecko utk total market cap butuh paket berbayar, dicek 401 Unauthorized). `btc_market_cap_usd` di kolom sebelah ITU asli. User pilih **skip btc_dominance dulu**, lanjut fear_greed (dikonfirmasi genuinely asli, tidak ada hardcode) + LSR + dist_liq.

**Fear & Greed: re-fetch berhasil.** `fetch_fear_greed()` dipanggil ulang manual (start=2020-01-01, end=2026-08-06) -> `data/training/macro/fear_greed_index.parquet` sekarang 2.410 baris, cover training+holdout penuh.

**`long_short_ratio`: TIDAK perlu enforcement kode tambahan** -- dicek langsung, kolom ini SUDAH genuine NaN (bukan placeholder 0/konstan) sblm batas Nov2021 di SEMUA koin dicek (ETHUSDT 16.036/16.036 baris pra-boundary = 100% NaN). Semua skrip riset malam ini SUDAH pakai `.notna()` filter yg otomatis benar. Ditambahkan `LSR_GENUINE_START = 2021-11-01` di `config.py` sbg dokumentasi eksplisit (bukan fix logika, krn logikanya sudah benar).

**KRITIS #2 -- `data/training/labeled_opt2_tier1/` (dipakai SEMUA riset dualbin) BUKAN output `run_engineer.py --all`.** Regenerasi `clean.py --all` + `engineer.py --coins ...` yg dijalankan (setelah insiden memori di bawah) BENAR menghasilkan `fear_greed` 100% terisi -- tapi di `data/training/labeled/`, direktori BEDA. `labeled_opt2_tier1` ternyata hasil GRAFT PARSIAL dari `data/training/labeled_opt2/` (135 kolom dipertahankan apa adanya, cuma 3 kolom LSR di-timpa fresh via `pipeline/experiments/dualbin_entry_fitur/regen_tier1_columns.py`). Root cause bug 0%-terisi ditelusuri langkah-demi-langkah (engineer_features -> merge_market_panel -> augment_coin_sync -> ETF merge -> HMM merge -> cols_to_keep -> dropna, SEMUA benar & preserve fear_greed) sampai ketemu: skrip yg saya jalankan menulis ke `LABEL_DIR` (`data/training/labeled/`), BUKAN `labeled_opt2_tier1`. **`fear_greed` BELUM benar-benar masuk ke TIER1 yg dipakai riset dualbin** -- perlu skrip graft baru serupa `regen_tier1_columns.py`, BELUM ditulis.

**Insiden memori sistem.** `engineer.py --all` (18 koin, `n_workers=cpu_count()-1` utk mode training -> ~11 worker paralel) gagal 13/18 koin dgn `MemoryError` -- physical memory tersisa 1,8GB/16GB, pagefile 457MB/46,7GB (99% penuh), akumulasi dari berjam-jam proses background sesi ini. User diberi 3 opsi (restart, coba bersihkan proses, tunda) -> jawab "lanjutkan saja". Diidentifikasi 11 proses python zombie (semua start ~22:50-22:51, cocok dgn command yg baru gagal) -> di-kill, ~4GB memori kembali (1,8GB->3,85GB free). Lanjut dgn batch kecil (`--coins` 3 koin sekaligus, bukan `--all`) -- BERHASIL utk 3 koin (BTC/ETH/BNB) di `labeled/`.

**Status akhir**: fear_greed source data FIXED & lengkap. `labeled/` (3/18 koin) sudah benar tapi BUKAN direktori yg dipakai riset. `labeled_opt2_tier1` (yg dipakai riset) BELUM tersentuh sama sekali. `dist_liq_50x_long/short` via bookDepth BELUM dikerjakan (skema kolom sudah dicek & didokumentasikan: `timestamp,percentage,depth,notional`, ±5% dari mid-price, snapshot ~30 detik, histori BTCUSDT dari 2023-01-01 -- lihat `reference-binance-vision-bookdepth-schema.md`). `btc_dominance` SENGAJA di-skip (data sumbernya palsu).

**Artefak.** `config.py` (+`LSR_GENUINE_START`), `data/training/macro/fear_greed_index.parquet` (diperbarui), `data/training/labeled/{BTC,ETH,BNB}USDT_features_v3.parquet` (fear_greed terisi, TAPI bukan dir yg dipakai riset). Memori baru: `project-dualbin-btc-dominance-hardcoded-fake.md`, `project-dualbin-tier1-three-parallel-datasets.md`, `reference-binance-vision-bookdepth-schema.md`.

**Lanjutan (sama malam) -- graft `fear_greed` ke TIER1 yg BENAR, tuntas.** Setelah root cause dipahami (§ di atas): (1) `data/holdout-test/raw/macro/fear_greed_index.parquet` disalin dari training (sblmnya kosong total -- `clean.py --holdout-test` pakai `_SRC_DIR` beda dari training, tidak fallback), (2) `clean.py --all --holdout-test` di-run ulang -> holdout `processed/` sekarang punya `macro_fear_greed_index_fear_greed`, (3) ditulis `regen_tier1_fear_greed.py` + `regen_tier1_fear_greed_holdout.py` (pola identik `regen_tier1_columns.py`/`_holdout.py`, HANYA cangkok `fear_greed`, 137 kolom lain apa adanya). Dijalankan sekuensial (bukan paralel, aman memori) utk 18 koin kedua arah.

**Hasil terverifikasi langsung**: training TIER1 `fear_greed` 0/946.413 (0%) -> 946.413/946.413 (100%). Holdout TIER1 0/51.444 (0%) -> 51.444/51.444 (100%). Dicek ulang manual baca file (bukan cuma self-report skrip): BTCUSDT training 54.723/54.723, holdout 2.858/2.858, jumlah kolom tetap 138 di kedua TIER1 (tidak ada kolom lain berubah).

**Artefak lanjutan.** `pipeline/experiments/dualbin_entry_fitur/regen_tier1_fear_greed.py`, `regen_tier1_fear_greed_holdout.py`. `data/holdout-test/raw/macro/fear_greed_index.parquet` (baru). `data/training/labeled_opt2_tier1/*.parquet` + `data/holdout-test/labeled_tier1/*.parquet` (fear_greed sekarang asli, seluruh 18 koin, kedua arah).

**Sisa kerja** (belum dikerjakan malam ini): `dist_liq_50x_long/short` via bookDepth asli (skema sudah dicek, fetch+desain+implementasi blm mulai), `btc_dominance` (sengaja ditunda, data sumber palsu).

**Artefak lanjutan.** Tidak ada script baru. Memori baru: `project-dualbin-macro-data-stale-not-fetcher-bug.md`.

---

## 2026-08-06/07 — bookDepth asli (Binance Vision): fitur BARU (bukan overwrite dist_liq_50x), fetch 18 koin berjalan lama di background

**Konteks.** User eksplisit setuju lanjut sendirian (akan tidur, "begitu ada token lanjutkan dan langsung training atas fitur perbaikan"). Auto-mode -- keputusan desain diambil sendiri, didokumentasikan di sini utk ditinjau, BUKAN dieksekusi ke holdout/produksi tanpa gerbang biasa.

**Keputusan desain penting (BUKAN overwrite fitur produksi).** `dist_liq_50x_long/short` TETAP DIPERTAHANKAN apa adanya (masih dipakai model produksi `long23f_swing`/`short21f_swing` -- mengubahnya diam-diam = train-serve mismatch). Fitur BARU ditambahkan terpisah: `book_depth_support_z`, `book_depth_resistance_z`.

**Formula** (setelah 2 iterasi desain, dites langsung di data BTC): (1) coba "wall distance" (bucket dgn marginal notional terbesar) -- HASIL TERLALU KASAR (cuma 5 nilai diskrit -5..-1/1..5, kebanyakan 1-2, tidak informatif). (2) **DIPILIH**: z-score notional depth di bucket -2%/+2% vs rolling 30 hari (720 bar H1) sendiri -- kontinu, well-distributed (mean~0, std~1,1, hampir semua nilai unik), causal (rolling backward-looking, `min_periods=48`). Dicek di sampel BTC 2023: 7.717 baris terisi sejauh fetch berjalan, distribusi sehat.

**NaN genuine sebelum 2023-01-01** (bukan 0.0) -- ikuti pola `long_short_ratio` yg sudah benar malam ini, biar `.notna()` filter otomatis eksklusi periode tanpa bookDepth, bukan pipeline lain harus ingat batas tanggal manual.

**Implementasi** (`pipeline/data/core/engineer.py`, setelah blok merge ETF): baca `data/training/bookdepth_h1/{coin}.parquet` (kalau ada), reindex ke `feat_df.index`, hitung z-score, tambah 2 kolom BARU ke `extra_cols` (bukan `FEATURE_COLS_V3` -- itu whitelist fitur produksi, sengaja tidak disentuh).

**Fetcher baru** (`pipeline/experiments/dualbin_entry_fitur/fetch_bookdepth.py`): Binance Vision `daily/bookDepth/{coin}/` (2023-01-01 s/d skrg, cuma harian, TIDAK ADA versi bulanan). Per hari: unduh zip, resample snapshot ~30 detik -> H1 (ambil snapshot TERAKHIR sblm jam tutup, kausal), simpan 10 kolom notional per bucket (-5%..-1%,1%..5%) ke `data/training/bookdepth_h1/{coin}.parquet`. Resumable (skip hari yg sudah ada), checkpoint tiap 30 hari per koin. **Estimasi durasi total 18 koin: ~4,5-5 jam** (BTC 1 tahun ~4 menit, 18 koin x ~2,5 tahun). **DIJALANKAN DI BACKGROUND, MASIH BERJALAN saat entry ini ditulis** -- cek progres via `tail data/training/bookdepth_h1/*.parquet` mtime atau log proses.

**Rencana lanjutan setelah fetch selesai** (BELUM dieksekusi): (1) graft `book_depth_support_z`/`book_depth_resistance_z` ke `labeled_opt2_tier1` + `labeled_tier1` (pola sama persis `regen_tier1_fear_greed.py`), (2) retrain LGBM LONG/SHORT research (OOF W1/W2/W3, BUKAN holdout) bandingkan: baseline (fitur skrg) vs +fear_greed vs +book_depth vs kombinasi, sesuai gerbang absolut yg sudah berlaku sepanjang sesi ini. HOLDOUT TETAP TIDAK DISENTUH tanpa kandidat lolos OOF 3/3 dulu (aturan tak berubah meski user sedang tidur).

**Artefak.** `pipeline/experiments/dualbin_entry_fitur/fetch_bookdepth.py`, `pipeline/data/core/engineer.py` (+blok merge bookDepth), `data/training/bookdepth_h1/*.parquet` (sedang terisi).

---

## 2026-08-07 — bookDepth fetch selesai (326 menit) + graft ke TIER1, tuntas

**Fetch selesai**: 18 koin, 2023-01-01 s/d 2026-08-06, total 326,2 menit (~5,4 jam). Ukuran total cuma ~55MB (disimpan ringkasan H1 10-kolom, bukan snapshot mentah 30-detik). Cakupan bagus: BTCUSDT 31.270 baris H1, ETCUSDT/BCHUSDT dst serupa, non-null ~99,97% dlm rentang tsb (celah kecil wajar dari hari fetch gagal/koneksi).

**Graft ke TIER1** (`regen_tier1_bookdepth.py` + `_holdout.py`, pola identik `regen_tier1_fear_greed.py` -- HANYA cangkok `book_depth_support_z`/`book_depth_resistance_z`, `dist_liq_50x_long/short` & 138 kolom lain TETAP apa adanya, diverifikasi manual tidak berubah):

| | Training TIER1 | Holdout TIER1 |
|---|---:|---:|
| Baris total | 946.413 | 51.444 |
| `book_depth_support_z` kosong | 441.856 (46,7%) -- genuine NaN sblm 2023-01-01 | 0 (0,0%) -- holdout seluruhnya 2026, dlm cakupan |
| `book_depth_resistance_z` kosong | 440.505 (46,5%) | 0 (0,0%) |
| Total kolom | 140 (138 lama + 2 baru) | 140 |

Terverifikasi langsung baca file (bukan cuma self-report skrip): BTCUSDT training 28.165/54.723 terisi, distribusi sehat (mean 0,09, std 1,09, rentang -5..5 stlh clip). `dist_liq_50x_long` BTCUSDT dicek sample -- nilai identik sblm/sesudah graft (tidak tersentuh, spt yg direncanakan).

**Selanjutnya**: retrain OOF (`fase140_retrain_fear_greed_bookdepth.py`) -- lihat entri berikutnya.

**Artefak.** `pipeline/experiments/dualbin_entry_fitur/regen_tier1_bookdepth.py`, `regen_tier1_bookdepth_holdout.py`.

---

## 2026-08-07 — Fase 140: retrain OOF baseline vs +fear_greed vs +book_depth vs +kombinasi -- SEMUA GAGAL, book_depth malah MEMPERBURUK

**Hasil (lb7, OOF W1/W2/W3, gerbang absolut PF>1 & PnL>0 tiap window terpisah):**

| Lengan | Total trade | Total PnL | W1 PF | W2 PF | W3 PF | Lolos? |
|---|---:|---:|---:|---:|---:|---|
| baseline | 4.349 | -$707,65 | 0,87 | 0,91 | 0,97 | 0/3 |
| +fear_greed | 4.220 | -$678,67 | 0,86 | 0,94 | 0,95 | 0/3 |
| +book_depth | 4.169 | -$899,02 | 0,85 | 0,89 | 0,93 | 0/3 |
| +kombinasi | 3.936 | -$826,90 | 0,89 | 0,89 | 0,89 | 0/3 |

**TIDAK ADA yg lolos.** `+fear_greed` sedikit lebih baik dari baseline (+$29) TAPI dalam lantai derau ±$80 yg sudah diukur sblmnya ([[feedback-ukur-lantai-derau-sebelum-menafsir]]) -- bukan perbaikan nyata. **`+book_depth` justru MEMPERBURUK** (-$899 vs -$708 baseline, PF turun di semua window) -- data order book ASLI (bukan proxy swing) TIDAK terbukti membantu, malah sedikit merugikan. `+kombinasi` juga lebih buruk dari baseline.

**Breakdown LONG/SHORT (PF) per lengan:**

| Lengan | LONG W1 | LONG W2 | LONG W3 | SHORT W1 | SHORT W2 | SHORT W3 |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 0,90 | 0,76 | 0,80 | 0,83 | 1,12 | 1,27 |
| +fear_greed | 0,90 | 0,79 | 0,77 | 0,83 | 1,15 | 1,22 |
| +book_depth | 0,88 | 0,74 | 0,76 | 0,81 | 1,11 | 1,22 |
| +kombinasi | 0,93 | 0,75 | 0,76 | 0,84 | 1,09 | 1,07 |

Pola menarik: `+kombinasi` LONG W1 membaik (0,93 vs 0,90 baseline) tapi SHORT W3 rusak parah (1,07 vs 1,27 baseline) -- kombinasi fitur baru tampaknya menggeser trade-off LONG/SHORT, bukan perbaikan bersih di kedua sisi.

**Kesimpulan**: fitur data-kualitas yg diperbaiki malam ini (fear_greed genuinely asli, book_depth genuinely order book asli -- BUKAN proxy) **TIDAK terbukti memperbaiki lb7**. `+fear_greed` netral (dlm derau), `+book_depth` sedikit merugikan. Ini KONSISTEN dgn kesimpulan Opsi A/B/C sebelumnya (data sintetis BUKAN akar masalah utama LONG lemah) -- akar masalahnya tetap overfitting/time-varying spt didiagnosa Fase 136, bukan kualitas data. `dist_liq_50x_long/short` (proxy swing) TIDAK diganti di produksi (tidak ada bukti order book asli lebih baik).

**TIDAK ada tindakan lanjut malam ini** -- holdout TIDAK disentuh (0/3 gerbang OOF di semua lengan), produksi TIDAK diubah. Keputusan lanjutan (kalau ada) menunggu user.

**Artefak.** `pipeline/experiments/dualbin_universe/fase140_retrain_fear_greed_bookdepth.py` (+`models/runs/dualbin_fase140_fear_greed_bookdepth.parquet`).

---

## 2026-08-07 — Fase 141-142: kerangka 8-kategori fitur user (ADX/MACD/candle structure/interaksi) -- 1 kandidat menarik ditemukan, sedang diverifikasi

**Konteks.** User beri kerangka lengkap 8-kategori fitur quant crypto (return/momentum, candle structure, trend, volatility, volume, derivatives, regime, time -- disimpan verbatim di memori `reference-quant-crypto-8category-feature-framework.md`). Dibandingkan ke 140 fitur existing: mayoritas sudah tercakup (kadang lebih canggih -- OFI/CVD utk order-flow, mkt_breadth/dispersion utk regime). Celah nyata yg dipilih utk dicoba (SEMUA dari OHLC yg sudah ada, TIDAK perlu fetch baru): ADX_14/+DI/-DI, MACD+signal+histogram, candle structure sederhana (body_ratio, wick_ratio, close_position, range_pct, gap_from_prev_close), atr_percent_h1 (blm ada versi H1, cuma H4 & percentile), risk_adjusted_momentum (interaksi log_ret_20/atr_percent_h1).

**Fase 141 -- fit GABUNGAN (metodologi B2: fit gabungan -> baca importance -> ablation utk yg bukan nol), lb7, OOF W1/W2/W3:**

| Lengan | Total PnL | Lolos? |
|---|---:|---|
| baseline | -$707,65 | Tidak |
| +14 kandidat gabungan | -$702,49 | Tidak (beda ~$5, dalam derau) |

**Feature importance kandidat baru (rata-rata LONG, 3 window):**

| Fitur | Importance |
|---|---:|
| **atr_percent_h1** | **17,21%** (jauh di atas semua kandidat lain, bahkan di atas banyak fitur lama) |
| adx_14 | 2,45% |
| minus_di_14 | 1,92% |
| macd | 1,89% |
| macd_signal | 1,80% |
| macd_histogram | 1,41% |
| range_pct | 1,29% |
| plus_di_14 | 0,99% |
| risk_adjusted_momentum | 0,83% |
| lower_wick_ratio / gap_from_prev_close / close_position / body_ratio / upper_wick_ratio | semua <0,5% |

Total 14 kandidat = 31,76% dari total importance model -- TAPI PnL model gabungan HAMPIR SAMA dgn baseline (-$702 vs -$707). Ini pola klasik "menyerap importance tanpa menambah sinyal baru" -- dicurigai `atr_percent_h1` (atr/close mentah) cuma menggantikan peran `atr_percentile_h1` (sudah ada, rank-based) yg informasinya tumpang tindih, bukan sinyal baru asli. Candle structure (body/wick/gap) semua <0,5% -- dalam derau, TIDAK layak diablasi lanjut.

**Fase 142 -- ablation OOF: `atr_percent_h1` sendirian vs grup trend (ADX+MACD gabungan), lb7:**

| Lengan | W1 PF | W2 PF | W3 PF | Total PnL | Lolos? |
|---|---:|---:|---:|---:|---|
| baseline | 0,87 | 0,91 | 0,97 | -$707,65 | Tidak |
| +atr_percent_h1 | 0,84 | **0,99** | 0,94 | **-$640,10** (+$67,55) | Tidak |
| +trend (ADX+MACD) | 0,85 | 0,91 | 0,93 | -$870,12 (-$162,47) | Tidak |

**`atr_percent_h1` sendirian**: perbaikan $67,55 -- angka terbesar malam ini dari satu fitur. TAPI dibedah per window: W1 memburuk (-$72), **W2 membaik BESAR (+$205, hampir tembus PF 1,0)**, W3 memburuk (-$65). Sama persis pola yg ditemukan pas `+fear_greed` semalam -- **1 window (W2) yg nge-drive semua perbaikan, 2 window lain malah lebih jelek.** Sesuai aturan "signifikan di 1 jendela = kandidat, bukan temuan" -- ini BUKAN efek konsisten, sama seperti fear_greed. Kemungkinan besar `atr_percent_h1` (atr mentah/close) memang tumpang tindih dgn `atr_percentile_h1` yg sudah ada (importance combined 17,2% tapi PnL gabungan nyaris sama dgn baseline di Fase 141 -- pola "menyerap importance tanpa nambah sinyal").

**Grup trend (ADX+MACD)**: KONSISTEN memburuk di SEMUA 3 window (beda dari atr_percent_h1) -- ini justru bukti BERSIH melawan penggunaannya, bukan sekadar "tidak membantu".

**Kesimpulan Fase 141-142 (kerangka 8-kategori fitur user)**: dari 14 kandidat yg dicoba (ADX/DI, MACD+signal+histogram, 6 fitur candle structure, atr_percent_h1, risk_adjusted_momentum) -- **TIDAK ADA yg lolos gerbang, dan tidak ada yg terbukti sbg perbaikan nyata setelah dicek konsistensi antar-window.** atr_percent_h1 punya angka $ terbesar tapi gagal uji konsistensi (sama seperti fear_greed sebelumnya); ADX+MACD malah terbukti konsisten merugikan; candle structure importance-nya dalam derau, tak layak diuji lanjut. Pola malam ini terus berulang: **setiap perbaikan/tambahan fitur yg dicoba (data sintetis, book_depth, fear_greed, ADX/MACD, candle structure) semuanya gagal memperbaiki lb7** -- memperkuat kesimpulan Fase 136 bahwa akar masalah LONG adalah overfitting/pola time-varying, bukan kekurangan fitur atau kualitas data.

**TIDAK ada tindakan holdout/produksi** -- semua gagal gerbang OOF.

**Artefak.** `pipeline/experiments/dualbin_universe/fase141_kandidat_fitur_baru.py` (+`models/runs/dualbin_fase141_kandidat_fitur_baru.json`), `fase142_ablation_atr_percent_trend.py` (+`models/runs/dualbin_fase142_ablation.json`). Memori baru: `reference-quant-crypto-8category-feature-framework.md`.

---

## 2026-08-07 — Fase 143: analisis kalibrasi probabilitas — TEMUAN BARU, model overconfident parah di KEDUA sisi

**Pemicu.** Setelah 14 kandidat fitur (Fase 141-142) semua gagal, user minta "evaluasi dan analisis lagi" -- bukan coba fitur baru lagi, tapi cari pola dari semua temuan malam ini. AUC LONG tinggi (0,90-0,93, Fase 136) tapi PF trading tetap <1 di HAMPIR SEMUA konfigurasi -- AUC ukur RANKING, bukan KALIBRASI probabilitas di ambang keputusan (TL=0,80/TS=0,75).

**Metode**: latih model lb7 standar (fitur baseline, TANPA kandidat baru) per window W1/W2/W3, kumpulkan probabilitas prediksi di validation set, bucketkan 10 bin, bandingkan rata-rata probabilitas prediksi vs realisasi win rate LABEL asli per bin.

**Hasil -- MISKALIBRASI PARAH, KEDUA SISI:**

| Bin probabilitas | Rata2 prediksi | LONG realisasi WR | SHORT realisasi WR |
|---|---:|---:|---:|
| 0,80-0,85 | 82,8% | 27,6% | 29,2% |
| 0,85-0,90 | 87,4% | 32,0% | 36,4% |
| 0,90+ | 91,0% | **36,7%** | **45,8%** |
| **Di atas ambang trading** | 85-86% | **30,7%** (n=22.659) | **32,6%** (n=22.248) |

Model bilang "85-91% yakin" tapi realisasi cuma 27-46% -- gap konsisten -50 s/d -55 poin persentase di SEMUA bin tinggi, KEDUA arah (LONG & SHORT).

**Analisis mendalam:**
1. **Ini BUKAN penjelasan langsung kenapa LONG lemah** -- miskalibrasi menimpa KEDUA sisi dengan magnitude serupa (gap -0,55 LONG vs -0,52 SHORT rata2). Kalau ini akar masalah LONG spesifik, harusnya SHORT jauh lebih terkalibrasi -- tidak.
2. **TAPI ada sinyal korobatif**: di bin PALING TINGGI (0,90+), gap LONG-SHORT justru MELEBAR (36,7% vs 45,8%, beda 9,1 poin) -- lebih besar dari gap di bin lain (~1-4 poin). Konsisten dgn temuan AUC/overfitting Fase 136 (LONG genuinely lebih lemah di ekor confidence tinggi), TAPI ini BUKAN penjelasan tunggal.
3. **Kemungkinan penyebab miskalibrasi**: `scale_pos_weight` (dipakai tangani label imbalance ~10-27% positive rate) mendorong output probabilitas LightGBM ke ekstrem (dekat 0/1) demi pemisahan kelas lebih baik -- efek samping umum, TIDAK ada langkah kalibrasi ulang (Platt scaling/isotonic) di pipeline manapun yg dicek malam ini.
4. **Implikasi operasional**: ambang trading (TL=0,80/TS=0,75) dipilih dgn asumsi angka mencerminkan peluang menang asli -- padahal "0,80" artinya realisasi cuma ~30%. Threshold saat ini mungkin BUKAN titik optimal krn dipilih di ruang probabilitas yg salah kalibrasi.

**Arah baru yg BELUM pernah dicoba malam ini**: SEMUA percobaan sebelumnya (data sintetis, book_depth, ADX/MACD, candle structure) soal FITUR. Ini soal CARA MODEL MENGHASILKAN PROBABILITAS -- kategori masalah berbeda total. Kalibrasi ulang (Platt/isotonic) berpotensi memperbaiki SISTEM SECARA UMUM (bukan cuma LONG), tapi belum terbukti -- BELUM dieksekusi, sengaja BERHENTI di sini utk cek-in dgn user dulu (protokol: jangan lanjut eksperimen otonom tanpa lapor stlh temuan besar).

**TIDAK ada tindakan holdout/produksi.**

**Artefak.** `pipeline/experiments/dualbin_universe/fase143_analisis_kalibrasi.py`.

---

## 2026-08-07 — Fase 144: sweep kalibrasi komprehensif (none/sigmoid/isotonic x 14 threshold) -- membantu LONG, MERUGIKAN SHORT, TETAP TIDAK LOLOS

**Rancangan (biar tidak "mengintip" OOF saat pilih threshold):** tiap window training dipotong -- 3 bulan TERAKHIR jadi "irisan kalibrasi" (terpisah dari data fit model DAN dari OOF asli). Model dilatih di sisa training (sblm irisan). Kalibrasi (sigmoid/isotonic) di-fit HANYA dari irisan kalibrasi. Threshold (grid 0,15-0,80) DIPILIH via trade simulation HANYA di irisan kalibrasi (kriteria PnL, min 30 trade). Baru (metode+threshold) yg "terkunci" itu diterapkan ke OOF asli.

**CATATAN METODOLOGI PENTING**: evaluasi Fase 144 pakai replay SATU SISI SEDERHANA (bukan `predict_dual`+aturan konflik margin+basi_reject+min_gap spt evaluasi resmi lb7 sepanjang malam) -- angka PF/PnL di sini **TIDAK bisa dibandingkan langsung** ke baseline -$707,65 yg dipakai di Fase 137-142. Perbandingan yg SAH cuma ANTAR metode kalibrasi DI DALAM fase ini (none vs sigmoid vs isotonic, kerangka sama).

**Hasil LONG (standalone):**

| Kalibrasi | W1 | W2 | W3 | Total PnL | Lolos? |
|---|---|---|---|---:|---|
| none (baseline) | trades=845, PF=0,93 | trades=945, PF=0,76 | trades=1088, PF=0,74 | -$1.118,66 | 0/3 |
| sigmoid | trades=140, PF=0,81 | trades=959, PF=0,76 | trades=431, PF=0,79 | -$728,06 | 0/3 |
| **isotonic** | **trades=28, PF=1,01 LOLOS** | trades=251, PF=0,80 | trades=1093, PF=0,74 | **-$637,24** | 1/3 (nyaris nol trade di window yg lolos) |

**Hasil SHORT (standalone):**

| Kalibrasi | W1 | W2 | W3 | Total PnL | Lolos? |
|---|---|---|---|---:|---|
| **none (baseline)** | trades=931, PF=0,80 | trades=787, PF=1,24 LOLOS | trades=861, PF=1,18 LOLOS | **+$195,23** | 2/3 |
| sigmoid | trades=945, PF=0,81 | trades=188, PF=1,54 LOLOS | trades=580, PF=1,23 LOLOS | -$16,10 | 2/3 (tp volume anjlok) |
| isotonic | trades=982, PF=0,79 | trades=120, PF=1,39 LOLOS | trades=689, PF=1,21 LOLOS | -$109,25 | 2/3 (tp volume anjlok) |

**Analisis:**
1. **Kalibrasi MEMBANTU LONG** -- isotonic memperbaiki total PnL dari -$1.119 (none) jadi -$637 (perbaikan ~$481), bahkan bikin 1 window "lolos" (tapi cuma 28 trade, PnL $0,77 -- lolos teknis, bukan bukti kuat). LONG masih GAGAL 2/3 window meski dgn kalibrasi terbaik.
2. **Kalibrasi MERUGIKAN SHORT** -- baik sigmoid maupun isotonic membuat total PnL SHORT jadi NEGATIF (-$16 dan -$109) padahal versi TANPA kalibrasi (none) sudah POSITIF (+$195, 2/3 window lolos). Penyebabnya: threshold hasil kalibrasi terlalu konservatif utk SHORT, volume trade anjlok drastis di W2 (787->188 sigmoid, ->120 isotonic) walau PF per-trade yg tersisa lebih tinggi.
3. **Konsisten dgn Fase 143**: miskalibrasi menimpa kedua sisi scr mentah, TAPI dampak PERBAIKANNYA asimetris -- LONG genuinely diuntungkan (raw threshold-nya memang memilih trade kualitas rendah), SHORT JUSTRU dirugikan (raw threshold 0,75 SHORT ternyata sudah cukup baik scr empiris meski "salah" scr probabilitas murni).
4. **TIDAK ADA kombinasi yg lolos gerbang 3/3** utk kedua sisi manapun. Arah kalibrasi TIDAK menyelesaikan masalah lb7 secara keseluruhan, tapi memberi info baru: LONG & SHORT butuh PERLAKUAN BERBEDA (LONG mgkn perlu kalibrasi, SHORT jangan disentuh) -- bukan 1 resep sama utk keduanya.

**Sisa kerja (BELUM dieksekusi)**: kalau mau lanjut, perlu re-evaluasi kombinasi "LONG isotonic-calibrated + SHORT tanpa kalibrasi (none/raw)" pakai ENGINE RESMI (predict_dual+margin conflict penuh, bukan replay sederhana single-sisi Fase 144 ini) sebelum bisa diklaim sbg kandidat -- BELUM dilakukan.

**TIDAK ada tindakan holdout/produksi.**

**Artefak.** `pipeline/experiments/dualbin_universe/fase144_sweep_kalibrasi.py` (+`models/runs/dualbin_fase144_sweep_kalibrasi.json`).

---

## 2026-08-07 — Fase 145: kombinasi LONG-kalibrasi + SHORT-asli via mesin RESMI -- perbaikan TERBESAR malam ini ($393), TAPI threshold LONG TIDAK STABIL antar-window

**Rancangan**: LONG dilatih di fit-slice (sblm irisan kalibrasi 3 bulan), dikalibrasi isotonic dari irisan kalibrasi, threshold LONG dipilih via grid search PnL DI IRISAN KALIBRASI SAJA (bukan diintip dari OOF). SHORT tetap standar (TS=0,75, latih penuh spt biasa, TANPA kalibrasi -- sesuai temuan Fase 144 bahwa kalibrasi merugikan SHORT). Kombinasi dievaluasi lewat MESIN RESMI (`predict_dual` rule=margin + `apply_basi_reject` + `apply_min_gap` + replay gabungan dgn aturan konflik) -- BUKAN replay sederhana Fase 144.

**Hasil:**

| Window | TL (dari kalibrasi) | Trade | PF | PnL | LONG (n, PF, $) | SHORT (n, PF, $) | Lolos? |
|---|---:|---:|---:|---:|---|---|---|
| W1 | 0,60 | 903 | 0,8485 | -$260,20 | n=22, PF=0,97, -$1 | n=881, PF=0,85, -$259 | Tidak |
| W2 | 0,45 | 912 | **1,0832** | **+$144,57** | n=188, PF=0,81, -$90 | n=724, PF=1,19, +$235 | **LOLOS** |
| W3 | 0,20 | 1.543 | 0,9236 | -$199,03 | n=953, PF=0,76, -$425 | n=590, PF=1,27, +$226 | Tidak |
| **Total** | | **3.358** | | **-$314,65** | | | **0/3** |

**Pembanding baseline (Fase 137, TL=0,80/TS=0,75 keduanya asli)**: PnL -$707,65, 0/3. **Perbaikan +$393 -- TERBESAR dari SEMUA percobaan malam ini** (lebih besar dari fear_greed +$29, atr_percent_h1 +$68). W2 bahkan LOLOS gerbang absolut sendirian.

**TAPI ada masalah serius yg bikin ini TIDAK bisa langsung dipercaya**: threshold LONG hasil kalibrasi **SANGAT TIDAK STABIL** antar window -- 0,60 (W1) -> 0,45 (W2) -> 0,20 (W3). Ini bukan variasi kecil, itu rentang HAMPIR 3x lipat. Konsekuensinya jumlah trade LONG melonjak liar: 22 (W1) -> 188 (W2) -> 953 (W3). Di W3, threshold serendah 0,20 meloloskan 953 trade LONG kualitas rendah (PF 0,76, rugi -$425 SENDIRIAN) -- ini yg bikin W3 tetap gagal. Pola yg terlihat: **perbaikan total PnL kemungkinan besar bukan krn LONG jadi genuinely lebih akurat, tapi krn threshold tinggi (W1: 0,60) KEBETULAN membuat LONG nyaris "opt-out" (cuma 22 trade, nyaris breakeven) di window yg SHORT-nya jg lagi jelek** -- bukan LONG yg membaik, tapi LONG yg "diam" pas sedang tidak berguna.

**Kesimpulan jujur**: pilihan threshold via irisan kalibrasi 3-bulan TIDAK STABIL/general -- kemungkinan overfitting ke idiosinkrasi tiap irisan kalibrasi spesifik, bukan menemukan titik operasi yg genuinely robust. Angka $393 real tapi TIDAK BOLEH ditafsirkan sbg "kalibrasi menyelesaikan masalah LONG" -- itu premature. Kalau mau lanjut, perlu threshold LONG yg LEBIH STABIL (mis. rata-rata/median dari beberapa irisan kalibrasi berbeda, bukan cuma 1 irisan 3-bulan per window) sblm bisa diklaim sbg kandidat kuat.

**TIDAK ada tindakan holdout/produksi** (0/3 gerbang, apalagi dgn keraguan stabilitas threshold ini).

**Artefak.** `pipeline/experiments/dualbin_universe/fase145_kombinasi_long_kalibrasi_short_asli.py`.

---

## 2026-08-07 — Fase 146: jawab "apakah ada overfitting?" -- YA, tapi bukan di tempat yg diduga

**User tanya langsung setelah Fase 145. Dua uji: (1) kurva PnL-vs-threshold penuh di tiap irisan kalibrasi, (2) uji silang -- threshold "milik" 1 window dipaksa dites di 2 window lain.**

**Kurva kalibrasi -- ternyata SEBAGIAN threshold yg "dipilih" Fase145 cuma "paling tidak rugi" dari kumpulan yg SEMUA rugi:**
- **W1**: SEMUA 12 threshold (0,15-0,70) hasilnya NEGATIF di irisan kalibrasi -- bahkan **PF=0,000 di SEMUA titik (nol trade menang sama sekali)**. Threshold 0,60 "terpilih" cuma krn PALING SEDIKIT rugi (-$8,10), bukan krn benar-benar bagus.
- **W2**: pola SEHAT -- PF naik konsisten seiring threshold naik (1,06 di 0,15 -> 2,78 di 0,50), PnL positif hampir semua titik. Ini SATU-SATUNYA irisan kalibrasi yg genuinely punya sinyal.
- **W3**: mirip W1 -- 4 threshold pertama (0,15-0,30) SEMUA negatif (-$393 s/d -$422). Ada 1 titik menjanjikan (th=0,35, PF=1,98, PnL+$8,75) TAPI cuma 11 trade, DIBUANG krn di bawah syarat minimal 30 -- threshold 0,20 yg akhirnya "terpilih" cuma yg paling sedikit rugi dari kumpulan buruk.

**Uji silang -- INI KUNCI TEMUANNYA:**

| Threshold dari | Diuji di | PF | PnL | Lolos? |
|---|---|---:|---:|---|
| W1 (0,60) | W1 (asli) | 0,85 | -$260 | Gagal |
| W1 (0,60) | W2 (asing) | **1,24** | **+$317** | **LOLOS** |
| W1 (0,60) | W3 (asing) | **1,22** | **+$252** | **LOLOS** |
| W2 (0,45) | W1 (asing) | 0,85 | -$276 | Gagal |
| W2 (0,45) | W2 (asli) | 1,08 | +$145 | LOLOS |
| W2 (0,45) | W3 (asing) | **1,22** | **+$252** | **LOLOS** |
| W3 (0,20) | W1 (asing) | 0,89 | -$329 | Gagal |
| W3 (0,20) | W2 (asing) | 0,92 | -$222 | Gagal |
| W3 (0,20) | W3 (asli) | 0,92 | -$199 | Gagal |

**Pola yg terungkap**: threshold TINGGI (0,45 atau 0,60) justru performanya SAMA BAGUSNYA -- bahkan LEBIH BAIK -- di window "asing" drpd di window "asalnya sendiri". Threshold 0,60 dari W1 GAGAL di W1 sendiri tapi LOLOS di W2 & W3. Threshold RENDAH (0,20 dari W3) gagal DI MANA PUN, termasuk di rumahnya sendiri.

**Kesimpulan jujur soal overfitting:**
1. **YA ada overfitting -- tapi di PROSES PEMILIHAN THRESHOLD PER-WINDOW, bukan di sinyal LONG itu sendiri.** Karena syarat minimal 30 trade, di W1 & W3 (irisan kalibrasi yg genuinely buruk) proses ini terpaksa memilih "yg paling tidak rugi" dari kumpulan yg semuanya rugi -- bukan menemukan sinyal asli. Threshold W3 (0,20) adalah PRODUK LANGSUNG dari overfitting ini: dipilih krn "OK" di kalibrasinya sendiri yg buruk, ternyata gagal di MANA PUN saat diuji.
2. **TAPI ada sinyal ASLI yg tersembunyi**: threshold TINGGI & STABIL (0,45-0,60), diterapkan SERAGAM (bukan dipilih ulang tiap window), meloloskan **W2 & W3 secara konsisten** (2 dari 3 window) -- ini BUKAN kebetulan, terbukti dari 2 threshold berbeda (0,45 dan 0,60) sama-sama meloloskan W2 & W3.
3. **W1 gagal di threshold BERAPAPUN yg dicoba** (0,20/0,45/0,60 semua gagal di W1) -- konsisten dgn pola yg sudah berulang kali ditemukan sepanjang malam sejak Fase128 ("W1 HAMPIR SELALU gagal apapun konfigurasinya"). Ini kelihatannya bukan soal threshold/kalibrasi sama sekali -- April-Juli 2025 tampaknya periode yg secara struktural buruk utk LONG, apapun caranya.

**Rekomendasi yg lebih jujur & robust**: JANGAN pilih threshold ulang tiap window (itu yg overfitting). Pakai 1 threshold TETAP & TINGGI (0,45 atau 0,60) di SEMUA window -- hasilnya 2/3 window lolos scr KONSISTEN, lebih bisa dipercaya drpd hasil Fase145 yg re-select per window. Tapi TETAP 0/3 scr gerbang absolut krn W1 tak tertembus threshold apapun -- kemungkinan perlu didekati beda (bukan soal ambang), atau diterima sbg periode yg memang tak bisa ditembus dgn setup lb7 ini.

**TIDAK ada tindakan holdout/produksi.**

**Artefak.** `pipeline/experiments/dualbin_universe/fase146_cek_overfitting_threshold.py` (+`models/runs/dualbin_fase146_cek_overfitting.json`).

---

## 2026-08-07 — Investigasi W1: KETEMU -- W1 gagal krn broad market BULL, bukan soal volatilitas

**Pemicu.** Rekomendasi sblmnya: selidiki W1 (April-Juli 2025) scr khusus krn gagal di SEMUA konfigurasi sepanjang malam. User bilang "lanjutkan saja".

**Cek kondisi pasar per window (18 koin, agregat):**

| Window | Return rata2 | Return median | Koin naik | ATR percentile rata2 | trend_strength rata2 |
|---|---:|---:|---:|---:|---:|
| **W1** | **+25,3%** | +22,5% | **14/18 naik** | 0,506 | +0,244 |
| W2 | -30,2% | -33,5% | 1/18 naik | 0,465 | -0,422 |
| W3 | -28,6% | -31,4% | 1/18 naik | 0,416 | -0,536 |

**Temuan (mengoreksi dugaan awal soal volatilitas)**: hipotesis pertama (W1 = volatilitas rendah, itu sebabnya) **TERBANTAH** -- ATR percentile W1 (0,506) justru PALING TINGGI dari ketiga window, bukan paling rendah. **Pembeda sebenarnya adalah ARAH PASAR**: **W1 = broad bull market murni** (14/18 koin naik rata2 +25%), **W2 & W3 = broad bear market** (nyaris semua koin turun, rata2 -29% s/d -30%).

**Pola yg terungkap (KONTRA-INTUITIF)**: LONG performanya PALING BURUK justru di window yg pasarnya BENERAN naik (W1), dan SHORT performanya PALING BAIK di window yg pasarnya turun (W2/W3). Kalau logika naif "LONG untung kalau market naik" berlaku, harusnya kebalikannya.

**Interpretasi**: ini menunjukkan masalahnya BUKAN "LONG secara umum lebih lemah dari SHORT", tapi lebih spesifik: **mekanisme entry berbasis swing (`swing_based_labeling`) kesulitan spesifik menangkap KENAIKAN BROAD-MARKET yg kuat & konsisten** -- kemungkinan krn di bull market yg solid, pullback ke swing low jarang & dangkal (sinyal entry LONG jarang muncul di titik yg genuinely bagus), beda dgn bear market yg triggernya (swing high utk SHORT) lebih jelas & sering muncul selama market memang lagi turun terus. Konsisten dgn hipotesis awal sesi ini (pergerakan turun kripto -- likuidasi, panic selling -- py pola lebih seragam/mekanis drpd pergerakan naik yg lebih beragam bentuknya).

**Implikasi**: ini BUKAN sesuatu yg bisa ditambal cepat malam ini (butuh desain ulang mekanisme label/entry LONG scr spesifik, bukan sekadar fitur/threshold/kalibrasi -- SEMUA sudah dicoba malam ini & gagal, konsisten dgn temuan baru ini krn semuanya menyasar simtom bukan akar). Layak jadi topik sesi terpisah dgn kepala dingin: cara paling masuk akal adalah desain label/entry LONG yg BEDA utk kondisi broad-bull (bukan swing-based yg sama dgn SHORT).

**TIDAK ada tindakan holdout/produksi.**

**Artefak.** Tidak ada script baru -- analisis langsung dari `data/training/labeled_opt2_tier1/*.parquet` (kolom close, atr_percentile_h1, trend_strength).

---

## 2026-08-08 — Fase 147+148: scorecard lengkap arsitektur "terbaik sementara" (OOF+OOS) -- LONG ternyata NYARIS MATI, bukan diperbaiki

**Rancangan.** Arsitektur = LONG isotonic-calibrated dgn threshold TETAP 0,60 (bukan re-select per window, sesuai rekomendasi Fase146) + SHORT asli TS=0,75 (kalibrasi terbukti merugikan SHORT, Fase144). Mesin resmi `predict_dual`+`apply_basi_reject`+`apply_min_gap`. User minta scorecard LENGKAP (trades/WR/PF/PnL per LONG-SHORT + peak equity/MaxDD, bkn cuma ringkasan) utk OOF, lalu eksplisit setuju jalankan OOS/holdout jg "sbg info tambahan" (BUKAN validasi -- OOF sblmnya cuma 2/3, gerbang absolut blm lolos).

**OOF (Fase147, 3 window Apr2025-Apr2026):**

| Window | Trade | WR | PF | PnL | Peak | MaxDD | LONG (n/WR/PF/$) | SHORT (n/WR/PF/$) | Lolos? |
|---|---:|---:|---:|---:|---:|---:|---|---|---|
| W1 (bull) | 903 | 51,3% | 0,8485 | -$260,20 | $197 | -$548 | n=22, WR=50%, PF=0,97, -$1,41 | n=881, WR=51,3%, PF=0,845, -$258,79 | Gagal |
| W2 (bear) | 759 | 56,5% | 1,2432 | +$317,41 | $317 | -$204 | n=1, WR=100%, PF=inf, +$5,07 | n=758, WR=56,5%, PF=1,239, +$312,34 | **LOLOS** |
| W3 (bear) | 726 | 59,0% | 1,2170 | +$251,56 | $344 | -$178 | n=14, WR=57,1%, PF=1,122, +$5,57 | n=712, WR=59,0%, PF=1,221, +$245,99 | **LOLOS** |
| **Total** | **2.388** | | | **+$308,78** | | | **37 trade LONG total (1,5%)** | **2.351 trade SHORT (98,5%)** | **2/3** |

**Temuan kunci yg baru kelihatan lewat scorecard detail (tidak kelihatan di Fase145/146 yg cuma lihat PF/PnL agregat)**: threshold tetap 0,60 pada probabilitas LONG yg sudah dikalibrasi ternyata membuat LONG **nyaris tidak pernah entry** -- 37 trade dari 2.388 total (1,5%). "Perbaikan" PnL di W2/W3 HAMPIR SELURUHNYA dari SHORT, bukan LONG. ini bukan "LONG diperbaiki", ini "LONG dimatikan scr de-facto, SHORT (yg memang sudah tak disentuh) yg menopang".

**OOS/Holdout (Fase148, 2026-04-01 s/d 2026-07-29, data BARU/tak pernah dilihat sama sekali):**

Data latih LONG: 2022-04-01 s/d 2025-12-30 (n=591.551) + irisan kalibrasi 2026-01-01 s/d 2026-03-30 (n=38.232). Latih SHORT penuh: 2022-04-01 s/d 2026-03-30 (n=630.431).

| Total | Trade | WR | PF | PnL | Peak | MaxDD | Status |
|---|---:|---:|---:|---:|---:|---:|---|
| Semua | 933 | 56,5% | 0,9796 | **-$26,29** | $64 | -$328 | GAGAL |
| LONG | **0** | - | - | $0,00 | | | **NOL trade sama sekali** |
| SHORT | 933 | 56,5% | 0,980 | -$26,29 | | | |

Per bulan: Apr gagal (PF 0,85, -$62), Mei gagal (PF 0,74, -$98), Jun lolos (PF 1,43, +$95), Jul lolos (PF 1,15, +$40) -- 2/4 bulan, tak konsisten.

**Kesimpulan jujur -- arsitektur ini TIDAK menyelesaikan apa-apa, cuma menyamarkan masalah:**
1. **LONG resmi mati di holdout** (0 dari 933 trade) -- lebih ekstrem drpd OOF (masih ada 37). Threshold 0,60 pada skala probabilitas kalibrasi + data holdout yg belum pernah dilihat = LONG benar2 tak pernah cukup percaya diri utk masuk.
2. **Karena LONG mati, hasil holdout = SHORT SENDIRIAN**. PF 0,98/-$26 di holdout ini **NYARIS IDENTIK** dgn baseline lb7 standalone holdout lama (Fase139: PF 0,9884/-$23,24) -- artinya seluruh rangkaian kalibrasi+threshold Fase143-147 TIDAK mengubah apapun scr substansial saat diuji di data benar2 baru. "Perbaikan +$393" yg terlihat di OOF (Fase145/147) kemungkinan besar cuma artefak dari window OOF spesifik yg dipakai, bukan perbaikan genuine yg generalize.
3. **Menguatkan (bukan menggantikan) temuan W1 sebelumnya**: [[project-dualbin-w1-bull-market-long-failure]] sudah bilang mekanisme swing-entry LONG kesulitan di broad-bull. Temuan hari ini menambahkan: bahkan di luar W1, LONG terlalu lemah utk lolos threshold tinggi manapun yg genuinely robust -- "solusi" kalibrasi+threshold-tetap cuma geser masalah dari "LONG buruk" jadi "LONG dimatikan".

**Rekomendasi**: arsitektur kalibrasi+threshold-tetap-0,60 ini **BUKAN kandidat** -- gerbang OOF tetap 2/3 (bukan 3/3) DAN OOS menunjukkan hasilnya = SHORT-only yg tak lebih baik dari baseline lama. Jalur kalibrasi/threshold utk LONG (Fase143-148) dianggap **SELESAI DIEKSPLORASI, tidak membuahkan kandidat lolos**. Kalau mau lanjut LONG, perlu kembali ke rekomendasi [[project-dualbin-w1-bull-market-long-failure]]: desain ulang label/entry LONG (bukan lagi threshold/kalibrasi), topik sesi terpisah.

**TIDAK ada tindakan produksi** -- holdout ini murni informasi tambahan sesuai permintaan user, bukan validasi (OOF sudah gagal gerbang duluan).

**Artefak.** `pipeline/experiments/dualbin_universe/fase147_oof_final_scorecard.py` (+`models/runs/dualbin_fase147_oof_final_scorecard.json`), `fase148_holdout_final_scorecard.py` (+`models/runs/dualbin_fase148_holdout_final_scorecard.json`).

---

## 2026-08-08 — Fase 149: mekanisme PERSIS kenapa LONG 0 trade di holdout -- ambang 0,60 = artefak ekor sempit, bukan sinyal

**Pemicu.** User tanya langsung "kenapa tidak ada trade long?" setelah Fase148. Dijawab dgn angka, bukan dugaan -- reproduksi training LONG yg sama persis, cetak statistik kurva kalibrasi & probabilitas holdout mentah.

**Irisan kalibrasi holdout (2026-01-01 s/d 2026-03-30, n=38.232):**
- Win rate keseluruhan: 6,13% (LONG memang jarang & kebanyakan jelek di periode ini).
- Win rate REALISASI per bucket raw-probability tertinggi: top10%=30,5%, top5%=32,3%, top1%=34,2%, **top0,1%=25,6%** (n=39 -- TURUN dari top1%, tanda klasik derau di sampel kecil, bukan pola asli).
- Kurva isotonic MELOMPAT ke 1,0 di titik ekor tsb (segelintir sampel yg kebetulan semua menang) -- **inilah sumber angka "0,60" yg "terpilih" di Fase145/146**, murni keberuntungan statistik di sampel kecil, bukan sinyal asli.

**Holdout (2026-04-01 s/d 2026-07-29, n=51.444):**
- Raw probability LONG PALING TINGGI yg pernah dicapai model: 0,9351 -- **sedikit di bawah** zona ekor sempit tempat kurva kalibrasi melompat ke 1,0.
- Karena tak masuk zona sempit itu, nilai kalibrasi tertinggi yg dicapai SELURUH holdout cuma **0,3812**.
- Ambang tetap = 0,60. Krn 0,38 < 0,60, **0 dari 51.444 bar lolos** -- persis match dgn hasil Fase148 (0 trade LONG).

**Kesimpulan**: rekomendasi Fase146 ("pakai ambang TETAP, bukan re-select per window") betul mengurangi 1 jenis overfitting (pemilihan ulang per-window), TAPI angka 0,60 itu sendiri ternyata SUMBERNYA rapuh -- produk kebetulan di ekor sempit 1 irisan kalibrasi spesifik, bukan titik operasi genuinely robust. Ini memperkuat (bukan mengubah) kesimpulan Fase147/148: jalur kalibrasi+threshold utk LONG sudah selesai dieksplorasi & tidak menghasilkan kandidat, krn masalahnya bukan di ANGKA ambang, tapi di sinyal LONG mentahnya yg genuinely lemah (win rate tak pernah tembus jauh di atas 30% bahkan di prediksi paling percaya diri sekalipun).

**TIDAK ada tindakan produksi.**

**Artefak.** `pipeline/experiments/dualbin_universe/fase149_diagnosa_kenapa_long_nol.py`.

---

## 2026-08-08 — Fase 150: SHORT sendirian (LONG dimatikan total) -- KONFIRMASI: kalibrasi semalam TIDAK menambah nilai apapun

**Pemicu.** User tanya "kalau short saja berapa oof dan oosnya?" -- dijalankan baseline SHORT MURNI (bukan LONG-diberi-ambang-tinggi-sampai-nyaris-mati spt Fase147/148, tapi LONG benar2 tak ada dlm keputusan), TS=0,75 asli tanpa kalibrasi, jendela training identik Fase147/148 spy apple-to-apple.

**Hasil:**

| | Trade | WR | PF | PnL | Status |
|---|---:|---:|---:|---:|---|
| OOF W1 | 881 | 51,3% | 0,8446 | -$259,66 | Gagal |
| OOF W2 | 758 | 56,5% | 1,2393 | +$312,34 | **LOLOS** |
| OOF W3 | 713 | 59,0% | 1,2274 | +$253,29 | **LOLOS** |
| **OOF Total** | **2.352** | | | **+$305,97** | **2/3** |
| **OOS/Holdout** | **933** | 56,5% | 0,9796 | **-$26,29** | Gagal |

**Perbandingan ke Fase147/148 (LONG isotonic-calib + ambang tetap 0,60, bukan SHORT murni)**:
- OOF: 2.388 trade/+$308,78 (Fase147) vs 2.352 trade/+$305,97 (Fase150) -- **selisih cuma $2,81**.
- OOS: 933 trade/-$26,29 (Fase148) vs 933 trade/-$26,29 (Fase150) -- **IDENTIK sampai 2 desimal** (krn LONG memang 0 trade di holdout Fase148).

**Kesimpulan**: ini konfirmasi paling bersih yg bisa didapat -- seluruh rangkaian kalibrasi Fase143-149 (semalam sampai pagi) secara matematis SAMA DENGAN SHORT-only baseline yg sudah ada sejak Fase137. Bukan "mendekati", tapi identik. Status lb7 SHORT-only tetap seperti sebelumnya: OOF 2/3 (W1 selalu gagal), OOS gagal (PF 0,98) -- BELUM ADA perbaikan apapun dari titik ini sejak awal sesi.

**TIDAK ada tindakan produksi.**

**Artefak.** `pipeline/experiments/dualbin_universe/fase150_short_only_oof_oos.py`.

---

## 2026-08-08 — RENCANA (belum dieksekusi): sguardian, Guardian khusus SHORT utk lb7

**Pemicu.** Entry lb7 tetap gagal gerbang (OOF 2/3, W1 selalu gagal -- masalah desain label
entry, bukan threshold/kalibrasi, lihat Fase143-150). User usul arah baru: pecah Guardian
(model exit) yg SAAT INI satu model gabungan LONG+SHORT, jadi model TERPISAH per-arah -- mulai
dari SHORT (`sguardian`). Ini menyimpang dari aturan "selesaikan LGBM entry dulu sblm Guardian" --
dikonfirmasi ke user, dijawab lanjut sadar (bukan LGBM dianggap selesai, tapi keputusan pindah
fokus sementara). **Sumber populasi trade: lb7** (dipilih user meski saya rekomendasikan model
live `short21f_swing` -- lb7 riset murni, BELUM tentu bisa dideploy langsung krn entry-nya sendiri
blm lolos gerbang).

**Riset arsitektur Guardian saat ini (dibaca langsung, `pipeline/experiments/guardian_reeval_
2026-07-08/run_train_guardian_18coin.py` + `pipeline/model/core/train_guardian.py`):**
- Sample training SUDAH py kolom `direction` (1.0=LONG/0.0=SHORT) sbg SALAH SATU dari ~28 fitur
  -- model SAAT INI bukan buta arah, cuma arahnya dicampur jadi satu model.
  P&L/MFE/drawdown SUDAH dinormalisasi searah profit (SHORT untung = angka positif, sama sprt
  LONG) -- jadi memisah per-arah TIDAK perlu ubah rumus label/fitur P&L sama sekali.
- Populasi generate: entry LGBM (opt2_plus_trend production, BUKAN lb7) -> `simulate_trades_swing`
  -> per-trade, per-bar sample (HOLD/PARTIAL/EXIT via `_label_pnl_constrained`).
- Training: `train_guardian_with_oof` -- purged K-fold pd posisi baris (BUKAN walk-forward
  waktu antar-window) utk pilih iterasi/logloss internal. **Ini beda dari evaluasi FINAL** (yg
  py masalah lantai derau, lihat [[project-dualbin-riset-guardian-tidak-layak]]) -- evaluasi
  akhir masih perlu backtest PF/PnL terpisah, bukan cuma logloss CV internal.
- Guardian lb7 yg SUDAH pernah dilatih (`models/runs/dualbin_guardian_lb7_20260806`) DIRAGUKAN
  krn fitur ETF placeholder saat itu (blm diperbaiki) -- TIDAK dipakai sbg baseline apa adanya.

**Rancangan (blm dieksekusi, tunggu konfirmasi akhir user):**

1. **Populasi**: trade SHORT dari lb7 (lookback=7, max_hold=24, TS=0,75 raw -- persis resep
   Fase150), di-generate ulang lintas 3 window OOF yg SUDAH established sesi ini (W1 Apr-Ags
   2025, W2 Ags-Des 2025, W3 Des2025-Apr2026) -- otomatis dpt ~2.352 trade SHORT total, JAUH di
   atas syarat lantai-derau lama (983 trade tak cukup; ini >2x lipat, tersebar 3 window
   independen pula).
2. **Fitur**: 28 fitur `guard28f` APA ADANYA, MINUS `direction` (selalu 0 utk populasi SHORT-only,
   nol informasi) -> 27 fitur. **Sengaja TIDAK nambah fitur baru bersamaan** -- prinsip "satu
   komponen dulu": isolasi dulu efek "model khusus arah", baru nanti (kalau ini menang) baru
   eksplorasi fitur tambahan sbg langkah terpisah.
3. **Label**: `_label_pnl_constrained` APA ADANYA, tak diubah.
4. **Training WALK-FORWARD per window** (beda dari resep `guard28f` production yg cuma sekali
   latih+purged-CV internal): sguardian window W_i dilatih dari trade SHORT SEBELUM W_i (rolling,
   pola sama persis entry lb7), skor exit dipakai KHUSUS trade SHORT di W_i itu. Kasih 3 titik
   ukur independen (bukan 1 angka gabungan) -- sejalan gerbang absolut yg sudah dipakai sepanjang
   sesi ini.
5. **3 lengan pembanding (populasi SAMA, bukan cuma 2)**, per [[feedback-ablation-samakan-populasi-latih]]:
   - (a) TANPA Guardian sama sekali (fixed TP/SL/timeout) -- angka SUDAH ada, itu Fase150 apa
     adanya, tinggal dipakai lagi.
   - (b) Guardian GABUNGAN (LONG+SHORT dicampur, direction jadi fitur) dilatih ULANG bersih
     khusus lb7 (BUKAN pakai `dualbin_guardian_lb7_20260806` yg lama/diragukan) -- baseline "exit
     model biasa" yg adil.
   - (c) **sguardian** (SHORT-only, 27 fitur) -- kandidat yg diuji.
   Kalau (c) > (b) > (a): splitting per-arah genuinely nambah nilai. Kalau (c) ≈ (b): splitting
   TIDAK menambah apa-apa, exit model gabungan sudah cukup (fitur `direction` sudah menangkap
   semua yg perlu). Kalau (b) ≈ (a): exit model APAPUN tidak menambah nilai di lb7 -- beda topik,
   balik ke entry.
6. **Gerbang keputusan**: PF>1 & PnL>0 di SEMUA 3 window, sama seperti entry -- **CATATAN
   PENTING**: exit yg lebih baik TIDAK BISA memperbaiki masalah ENTRY. W1 (bull market, entry
   LONG/gate lemah) kemungkinan besar TETAP gagal apa pun Guardian-nya -- itu sudah didiagnosis
   sbg masalah desain label ENTRY (lihat [[project-dualbin-w1-bull-market-long-failure]]), bukan
   exit. Jangan berharap sguardian "menyelamatkan" W1.
7. **Setelah SHORT tuntas** (menang/kalah/perlu iterasi lanjut) -- BARU mulai `lguardian` (LONG),
   satu per satu, sesuai aturan sistematis yg sudah berlaku sepanjang sesi ini.

**Belum diputuskan / perlu konfirmasi eksplisit user sblm eksekusi**: apakah lanjut latih (b) dan
(c) sekarang, atau ada penyesuaian rancangan dulu.

**TIDAK ada tindakan holdout/produksi direncanakan di tahap ini -- OOF dulu.**

**UPDATE -- DIEKSEKUSI (2026-08-08).** Trajectory 12 bulan sblm W1 (2024-04-01→2025-04-01,
n=157.032), 3.259 trade trajectory → 60.758 sampel (LONG=28.691, SHORT=32.067). Data ETF TIER1
dicek dulu SEBELUM latih -- SUDAH asli (nunique 19-23/bulan, bukan placeholder nol), jadi arm (b)
kali ini TIDAK kena masalah lama fase132 (guardian lb7 lama diragukan krn ETF nol).

**Gotcha yg ketemu & DIPERBAIKI di skrip (bukan ubah fungsi SSOT bersama)**: kelas PARTIAL_EXIT
sangat jarang di populasi SHORT-only (33/32.067 = 0,1%) -- 1 dari 8 fold purged kebagian validasi
py kelas yg tak pernah muncul di training-nya, `LabelEncoder` LightGBM meledak
(`ValueError: y contains previously unseen labels: [1]`). Fix: salinan LOKAL fungsi training
(`train_guardian_robust` di skrip ini, BUKAN edit `train_guardian.py` SSOT) yg skip fold semacam
itu -- selebihnya identik persis (hyperparameter/class-weight/purge/refit).

**Hasil OOF (populasi entry SHORT-only identik ke Fase150, cuma exit policy beda):**

| Lengan | Trade total | PnL total | Window lolos |
|---|---:|---:|---:|
| (a) Tanpa Guardian [Fase150] | 2.352 | +$305,97 | 2/3 |
| (b) Guardian gabungan (LONG+SHORT, `direction` sbg fitur) | 5.527 | **-$1.502,38** | 0/3 |
| (c) sguardian (SHORT-only, 27f tanpa `direction`) | 5.504 | **-$1.512,12** | 0/3 |

**Temuan 1 (menjawab pertanyaan awal)**: (b) vs (c) HAMPIR IDENTIK (-$1.502 vs -$1.512, selisih
$9,74 dari total >$1.500 rugi) -- pola SAMA PERSIS spt semua variasi kecil sepanjang sesi ini
(lihat [[feedback-wajib-dua-jendela-penilaian]]). **Memisah Guardian per-arah TIDAK menambah
nilai apa pun** dgn resep ini -- fitur `direction` di guardian gabungan sudah cukup menangkap
info yg dibutuhkan, split tak memberi keunggulan.

**Temuan 2 (jauh lebih besar & mengejutkan, TAPI BELUM BISA dipercaya penuh)**: KEDUA lengan
Guardian jauh LEBIH BURUK drpd tanpa Guardian sama sekali -- PF turun dari >1 jadi ~0,6-0,7, DAN
jumlah trade nyaris DOBEL (2.352 → ~5.500) meski hanya exit policy yg berubah (entry population
SAMA). Diperiksa: PARTIAL_EXIT TIDAK dobel-hitung sbg trade terpisah (posisi tetap 1 baris,
`partial_pnl` terlipat ke `net_pnl` final) -- BUKAN bug penghitungan.

**Kecurigaan kuat (BELUM diverifikasi, jangan simpulkan "Guardian buruk" dulu)**: `exit_threshold`
(0,55) + parameter Guardian lain yg dipakai evaluasi diambil APA ADANYA dari `config.json`
produksi (`muat_cfg_live()`) -- angka itu di-tuning utk Guardian PRODUKSI LAMA (model beda,
distribusi probabilitas beda), BUKAN utk 2 model BARU yg baru dilatih di sini. Ambang yg salah
skala bisa bikin Guardian baru exit KELEWAT dini/sering -- otomatis menjelaskan DUA gejala
sekaligus (trade lebih banyak krn turnover lebih cepat, PF anjlok krn exit prematur motong
posisi yg akan profitable). Pola sama fase132 (guardian lb7 lama JUGA "lebih buruk", dulu
disalahkan ke ETF placeholder -- ETF sekarang sudah dipastikan asli, jadi kalau masih buruk,
threshold yg belum di-tuning kandidat kuat penyebabnya, bukan ETF lagi).

**Belum diputuskan**: apakah lanjut tuning `exit_threshold` (+ param terkait) utk 2 model baru
ini via irisan kalibrasi (pola sama entry threshold tuning Fase144-146) sblm menyimpulkan
"Guardian tak berguna di lb7", ATAU terima temuan 1 (split per-arah tak berguna) sbg jawaban
final & hentikan jalur Guardian di sini.

**TIDAK ada tindakan holdout/produksi.**

**Artefak.** `pipeline/experiments/dualbin_universe/fase151_sguardian_vs_combined.py`.
`models/runs/dualbin_guardian_lb7_combined_20260808/`, `models/runs/dualbin_sguardian_lb7_20260808/`.

---

## 2026-08-08 — Fase 152: sguardian, isolasi exit_threshold (near_tp/breakeven MATI), sweep OOF

**Pemicu.** Fase151 curiga `exit_threshold=0,55` dkk (dari `config.json` produksi, di-tuning utk
Guardian LAMA) menjelaskan kenapa KEDUA lengan Guardian lebih buruk drpd tanpa Guardian. Berhenti
banding ke Guardian gabungan (Temuan 1 Fase151 sudah jawab: hampir identik ke sguardian) --
fokus HANYA sguardian, model yg SUDAH terlatih (tidak dilatih ulang), sweep parameter EKSEKUSI.

**Rancangan**: `near_tp_arm_frac`/`breakeven_lock_mfe_pct` dimatikan (0,0 = isolasi murni
keputusan model sguardian sendiri), `exit_threshold` di-sweep 0,35→0,99 (10 titik) langsung di 3
window OOF -- BUKAN kalibrasi terpisah (data M5 intrabar cuma ada Apr2025-Apr2026, PERSIS rentang
OOF, tak ada ruang irisan kalibrasi sblm W1). Ditandai eksploratif, kurva PENUH dicetak.

**Hasil (total PnL 3 window per ambang, populasi entry SHORT identik Fase150/151):**

| Ambang | Total PnL | Window lolos |
|---|---:|---:|
| 0,35 | -$222,11 | 1/3 |
| 0,55 | -$145,82 | 1/3 |
| 0,65 | -$2,35 | 2/3 |
| 0,75 | +$229,35 | 2/3 |
| 0,85 | +$380,06 | 2/3 |
| 0,90 | +$432,87 | 2/3 |
| 0,95 | +$496,08 | 2/3 |
| **0,99** | **+$510,77** | **2/3** |
| Tanpa Guardian [Fase150] | +$305,97 | 2/3 |

Kurva naik halus (bukan spike 1 titik), konsisten di ketiga window scr terpisah -- W1 tetap rugi
tapi mengecil (-$259,66→-$216,55), W2/W3 membaik. W1 tetap gagal gerbang di SEMUA ambang (problem
struktural bull market di entry, bukan Guardian, sudah terdiagnosis lama).

**Masalah**: 0,99 adalah UJUNG grid yg diuji, bukan puncak di tengah -- kurva belum kelihatan
mendatar/berbalik. User curiga ("agak kurang meyakinkan kenapa bisa 0,99"), minta training period
LGBM SHORT + sguardian dipaparkan. Ditelusuri: entry SHORT dilatih ULANG per window (rolling 48
bulan, dipotong 2021-11-01), tapi sguardian dilatih SEKALI dari trajectory 2024-04-01→2025-03-30
lalu dipakai APA ADANYA di W1/W2/W3 -- utk W3, model "melihat" pola trade berumur 8-9 bulan.
Asimetri ini (entry walk-forward, sguardian statis) belum ditandai eksplisit sblm ini.

**TIDAK ada tindakan holdout/produksi. Lanjut ke Fase153 (cek OOS) sblm simpulkan apa pun.**

**Artefak.** `pipeline/experiments/dualbin_universe/fase152_sguardian_threshold_sweep.py`.
`models/runs/dualbin_fase152_sguardian_threshold_sweep.json` (grid 0,35-0,75),
`models/runs/dualbin_fase152_sguardian_threshold_sweep_tinggi.json` (grid 0,80-0,99).

---

## 2026-08-08 — Fase 153: sguardian exit_threshold di OOS -- pola Fase152 TIDAK bertahan (negatif)

**Pemicu.** User minta sweep 0,75 ke atas (kelipatan 0,05) langsung di OOS/holdout (2026-04-01 s/d
2026-07-29), data yg BENAR-BENAR belum pernah dilihat sguardian maupun entry SHORT window
manapun -- jawaban langsung atas kecurigaan "kurva 0,99 belum ketemu puncak, apa ini bertahan di
data baru?". Model sguardian SAMA (tak dilatih ulang), entry SHORT dilatih 1x khusus OOS (rolling
48 bulan: 2022-04-01→2026-03-30, sama pola Fase150). Grid diperluas sampai 1,00 (guardian nyaris
mati total -- anchor pembanding langsung ke "tanpa Guardian").

**Hasil:**

| Ambang | Trade | WR | PF | PnL |
|---|---:|---:|---:|---:|
| 0,75 | 1.036 | 41,4% | 0,9627 | -$51,17 |
| 0,80 | 1.011 | 40,8% | 0,9662 | -$46,33 |
| 0,85 | 975 | 41,3% | 0,9595 | -$54,49 |
| 0,90 | 958 | 41,5% | 0,9582 | -$55,94 |
| 0,95 | 918 | 42,7% | 0,9675 | -$43,00 |
| 1,00 (Guardian nyaris mati) | 855 | 45,3% | 0,9908 | -$11,70 |
| Tanpa Guardian [Fase150 OOS] | 933 | 56,5% | 0,9796 | -$26,29 |

**Kesimpulan -- pola OOF TIDAK bertahan, kecurigaan user TERBUKTI BENAR.** Di OOF, ambang tinggi
jelas menang (kurva naik halus -$222→+$511). Di OOS, SEMUA ambang gagal gerbang (PF selalu <1),
DAN polanya justru **kebalikan**: makin sering Guardian ikut campur (ambang rendah, 0,75-0,95),
makin buruk hasilnya dibanding hampir tidak ikut campur sama sekali (1,00, paling dekat ke "tanpa
Guardian"). Tidak ada titik ambang manapun yg mendekati keunggulan +$500 yg terlihat di OOF.

**Penjelasan paling masuk akal**: sguardian dilatih SEKALI dari periode 2024-04-01→2025-03-30 dan
TIDAK di-refresh -- saat dipakai ke OOS (mulai 2026-04-01), model itu sudah berumur ~13 bulan dari
titik latihnya (lebih basi drpd saat dipakai ke W3 yg "cuma" 8-9 bulan). Pola yg tampak kuat di
OOF kemungkinan besar spesifik ke rezim pasar 2025 yg dilihat model saat training, bukan sinyal
exit yg genuinely bisa dipakai umum. Ini konsisten dgn asimetri yg sudah ditandai di Fase152 (entry
walk-forward per window, sguardian statis) -- kali ini terbukti asimetri itu berakibat nyata.

**Status jalur sguardian: NEGATIF/TIDAK LAYAK dgn resep saat ini.** Tidak direkomendasikan lanjut
ke produksi dlm bentuk apa pun. Kalau mau dilanjutkan, perlu perubahan struktural (latih ULANG
sguardian per window scr walk-forward, sama disiplin dgn entry SHORT-nya) -- bukan cuma sweep
parameter eksekusi lagi, dan itu langkah besar yg butuh persetujuan eksplisit sblm dieksekusi.

**TIDAK ada tindakan holdout/produksi.**

**Artefak.** `pipeline/experiments/dualbin_universe/fase153_sguardian_oos_threshold_sweep.py`.
`models/runs/dualbin_fase153_sguardian_oos_threshold_sweep.json`.

---

## 2026-08-08 — Fase 154: sguardian v2 -- dibangun ulang dari 4 prinsip Guardian user (bukan resep lama)

**Pemicu.** User curiga sguardian v1 (Fase151-153, gagal OOS) BUKAN krn parameter salah tapi krn
prinsip Guardian-nya sendiri salah. Diminta bandingkan ke Guardian ic32 (`train_guardian.py`,
"Momentum Escort v2") -- ternyata resep itu MILIK ic32, bukan dualbin (skrip sampel dualbin asli
sudah HILANG, dicatat di fase6). User lalu menjelaskan 4 prinsip sendiri: (P1) Guardian hanya exit
saat profit, (P2) trend berbalik->keluar tapi momentum kuat->tetap hold walau lewat TP, (P3) sentuh
TP->partial 50% otomatis, (P4) sisa posisi tetap dikawal Guardian. Diminta bangun ulang dari sini,
BUKAN dari resep lama manapun. Rencana penuh: `witty-dreaming-kahn.md` (Plan Mode, Opus).

**Implementasi** (TDD penuh, `live_dualbin_ft` commit `937ba83`): 3 mekanisme baru di `_manage_bar`
(`paper_engine.py`) + `GuardianCfg`, SEMUA default MATI (parity byte-for-byte `replay_arrays()`
tetap hijau tanpa diubah, 34 test parity + 10 test baru lulus, suite penuh 448 lulus/1 gagal
pra-eksisting tak terkait):
- `tp_partial_frac` (P3) -- partial DETERMINISTIK persis saat TP tersentuh, pakai ulang mekanika
  partial lama (`_apply_partial_exit`, diekstrak jadi fungsi bersama -- SATU rumus PnL).
- `escort_after_partial` (P4) -- gate `position_remaining>0.5` diperluas `or escort_after_partial`,
  memperbaiki bug lama: Guardian berhenti mengevaluasi posisi SELAMANYA setelah partial.
- `exit_only_when_profit` (P1) -- keputusan EXIT model diabaikan kalau `current_pnl<=0`.

**Metodologi**: sguardian v2 dilatih ULANG WALK-FORWARD per window (rolling 12 bulan trajectory,
berhenti 36 jam sblm window mulai) -- BUKAN sekali latih spt v1, memperbaiki akar penyebab gagal
OOS Fase153. Label 2-kelas fungsional (HOLD/EXIT) via `mom_strong` dari `cvd_momentum_adv` (P2).

**Bug ditemukan & diperbaiki SEBELUM hasil final** (run pertama, 17:40-17:48, HARUS dibuang):
melewati nilai label 1 sepenuhnya (HOLD=0, EXIT=2 langsung) merusak `LabelEncoder` internal
sklearn -- mereindeks nilai yg ADA jadi kolom berurutan (0,2->kolom 0,1), TAPI `_manage_bar`
membaca `_Booster.predict()` mentah (mengasumsikan indeks kolom == nilai label). Akibatnya
keputusan EXIT mendarat di kolom "PARTIAL" dan DITOLAK DIAM-DIAM oleh `izinkan_partial=False` --
P2 jadi TIDAK PERNAH benar2 tereksekusi, CV logloss meledak ke ~3,0 (bukti: log produksi "Guardian
memilih PARTIAL p=0,99+ tapi jalur real-time tidak bisa eksekusi" muncul di HAMPIR SETIAP keputusan
EXIT). Direproduksi & diverifikasi via isolasi `LGBMClassifier` 10 baris sblm memperbaiki. **Fix**:
kembali ke 3-kelas utuh (kelas 1 diisi placeholder teknis dari aturan give-back v1, TIDAK PERNAH
benar2 dieksekusi krn `izinkan_partial=False`) + `train_guardian_robust` lokal (skip fold CV kelas
langka, sama pola v1) -- deviasi dari rencana ("tanpa salinan lokal"), perlu berdasar bukti
empiris. CV logloss run kedua: 0,25-0,36 (waras, malah lebih baik dari v1 kombinasi 0,44-0,51).

**Hasil OOF (headline exit_threshold=0,55, TIDAK dipilih dari kurva):**

| Lengan | Trade | PnL total | Window lolos |
|---|---:|---:|---:|
| (a) Tanpa Guardian [Fase150] | 2.352 | +$305,97 | 2/3 |
| (b) v2 + lantai lama (momentum_floor/near_tp/breakeven_lock config live APA ADANYA) | 4.542 | **-$1.234,66** | 0/3 |
| (c) v2 murni model (lantai lama SEMUA mati, cuma 4 prinsip) | 2.278 | **+$342,57** | 2/3 |

Per window (c) vs (a): W1 -$190,08 vs -$259,66 (+$69,58, tetap gagal gerbang -- problem entry bull
market, bukan exit), W2 +$257,06 vs +$312,34 (**-$55,28**, lolos tapi lebih buruk), W3 +$275,58 vs
+$253,29 (+$22,29, lolos & sedikit lebih baik). **Arah TIDAK konsisten 3/3 window** (2 window
membaik, 1 memburuk) -- persis pola yg [[feedback-wajib-dua-jendela-penilaian]] tandai sbg BUKAN
edge stabil. Selisih total (+$36,60, ~12%) berada di bawah/dekat lantai derau ~$80 per-window yg
sudah ditemukan sesi-sesi sebelumnya.

**(b) gagal total** -- floor lama (momentum_floor_frac=0,1 dkk) di-tuning utk Guardian LAMA,
"bentrok" dgn mekanisme baru (`tp_partial_frac`/`escort_after_partial`). Pelajaran: JANGAN campur
parameter lama dgn model/mekanisme baru tanpa tuning ulang -- pola sama v1 Fase151.

**Kurva diagnosa (c), TIDAK dipakai memilih ambang**: 0,35=$283,88 (=0,45) -> 0,55=$342,57 ->
0,65=$529,14 -> 0,75=$559,23 -> **puncak 0,85=$578,64** -> 0,95=$500,14 (TURUN). Beda kualitatif
dari Fase152 (v1): ada **puncak di TENGAH grid**, bukan naik terus sampai ujung -- pola lebih sehat
(bukan "makin dekat mati makin baik"), TAPI ini baru observasi, BUKAN validasi ambang 0,85 (sama
sekali belum diuji OOS, dan memilih dari kurva OOF persis kesalahan yg dihindari).

**Kesimpulan jujur**: 4 prinsip user SEKARANG genuinely diimplementasikan & tereksekusi benar
(dibuktikan lewat TDD + fix bug encoding) -- beda dari v1 yg diam-diam menyimpang dari prinsip
manapun. Tapi edeknya di headline ambang KECIL & arahnya TIDAK konsisten antar window -- lebih
jujur disebut "belum terbukti", bukan "menang" ataupun "kalah telak" spt v1. TIDAK direkomendasikan
promosi ke OOS/produksi dari hasil ini saja.

**TIDAK ada tindakan OOS/holdout/produksi -- butuh persetujuan terpisah.**

**Artefak.** `pipeline/experiments/dualbin_universe/fase154_sguardian_v2_prinsip.py`.
`models/runs/dualbin_sguardian_v2/{W1,W2,W3}_20260808/guardian/`. Kode mesin:
`live_dualbin_ft` commit `937ba83` (`app/services/paper_engine.py`, `app/config.py`,
`tests/test_guardian_v2_principles.py`).

---

## 2026-08-08 — Fase 155: sguardian v2 di OOS -- TIDAK reversal spt v1, tapi TETAP belum terbukti

**Pemicu.** User minta cek OOS setelah lihat hasil Fase154 (OOF +$342,57 vs baseline +$305,97,
arah tak konsisten). Metodologi SAMA PERSIS Fase154 (walk-forward): sguardian v2 dilatih ULANG
khusus utk window OOS (trajectory 2025-04-01→2026-03-30, entry SHORT rolling 48bln berhenti
2026-03-30) -- BUKAN reuse model W3. Cuma lengan (c) v2 murni model diuji (lengan (b) sudah
terbukti gagal total di Fase154, tidak diulang).

**Hasil (headline exit_threshold=0,55, TIDAK dipilih dari kurva):**

| | Trade | WR | PF | PnL | Status |
|---|---:|---:|---:|---:|---|
| (c) v2 murni model | 910 | 57,7% | 0,9897 | **-$12,65** | Gagal (PF<1) |
| Tanpa Guardian [Fase150 OOS] | 933 | 56,5% | 0,9796 | -$26,29 | Gagal |

v2 GAGAL gerbang (PF 0,9897 < 1,00), tapi rugi LEBIH KECIL drpd baseline (-$12,65 vs -$26,29,
selisih +$13,64) -- BUKAN kemenangan, tapi BUKAN pula reversal total spt v1 (v1 KALAH di SEMUA
ambang OOS, sguardian v2 sedikit lebih baik drpd baseline di ambang headline).

**Kurva diagnosa (TIDAK dipakai memilih ambang, cuma transparansi)**: 0,35/0,45 PF=1,0067
PnL=+$8,25 (lolos gerbang), 0,55=-$12,65, 0,65=-$31,26 (terburuk), 0,75=-$8,09, 0,85 PF=1,0004
PnL=+$0,44 (lolos tipis), 0,95 PF=1,0022 PnL=+$2,68 (lolos tipis). **Kurva BERGELOMBANG** (turun
lalu naik lagi), bukan tren bersih naik/turun spt Fase152/153/154 -- tanda paling jelas bahwa
seluruh rentang ini ada DI DALAM lantai derau, bukan sinyal sistematis.

**Menghubungkan ke 4 window independen (W1/W2/W3 OOF + OOS)**: v2 mengalahkan baseline di 3/4
(W1 +$69,58, W3 +$22,29, OOS +$13,64), kalah di 1/4 (W2 -$55,28). Mayoritas arah positif, TAPI
semua selisih kecil (di bawah/dekat lantai derau ~$80/window) DAN tidak ada satu pun window yg
menunjukkan efek besar & jelas. Beda kualitatif dari v1 (menang OOF DRAMATIS lalu kalah OOS
DRAMATIS di SEMUA ambang) -- v2 dari awal tidak pernah mengklaim kemenangan besar.

**Kesimpulan jujur**: sguardian v2 TIDAK menunjukkan reversal OOF->OOS yang menjatuhkan v1 --
tapi efeknya di semua 4 window terlalu kecil & konsisten HANYA mayoritas (3/4, bukan 4/4) utk
disebut edge terbukti. Status: **belum terbukti, condong netral-ke-sedikit-positif**, bukan
kemenangan bersih. Tidak lolos gerbang PF>1 di OOS pada ambang headline manapun yg wajar (0,55).

**TIDAK direkomendasikan promosi ke produksi.** Kalau mau dilanjutkan: perbesar sampel evaluasi
(lebih banyak koin/periode) utk menurunkan lantai derau sebelum menafsir lebih jauh -- sweep
ambang lagi TIDAK akan menjawab apa-apa selama semuanya ada di dalam derau. Juga diingat: P3
(partial 50% di TP) MASIH belum bisa dieksekusi sungguhan di live (order `reduceOnly` sebagian) --
promosi ke produksi tetap perlu proyek terpisah itu dulu, terlepas dari hasil riset ini.

**TIDAK ada tindakan holdout/produksi lebih lanjut.**

**Artefak.** `pipeline/experiments/dualbin_universe/fase155_sguardian_v2_oos.py`.
`models/runs/dualbin_sguardian_v2/OOS_20260808/guardian/`,
`models/runs/dualbin_fase155_sguardian_v2_oos.json`.

---

## 2026-08-08 — Fase 156: KOREKSI -- baseline Fase154/155 pakai mesin BEDA, v2 sebenarnya KALAH $175

**Pemicu.** User minta pecah hasil per bulan. Membangunnya memaksa baseline "tanpa Guardian"
dihitung ULANG lewat mesin yang SAMA dgn v2 (`ExecutionSimulator`) -- dan ketahuan baseline yang
dipakai Fase154/155 (`BASELINE_A`, angka teks dari Fase150) itu dihasilkan `replay_gabungan()`,
fungsi simulasi TULIS TANGAN TERPISAH (TP/SL/timeout sederhana, TANPA mekanika `ExecutionSimulator`
penuh -- fill M5 intrabar, cek likuidasi, force-close `MAX_HOLDING_BARS`, dst). **Seluruh
perbandingan "v2 menang 3/4 window" di Fase154/155 membandingkan DUA MESIN SIMULASI BERBEDA**,
bukan cuma kebijakan exit yang beda -- pelanggaran [[feedback-ablation-samakan-populasi-latih]]
versi lebih dasar (mesinnya sendiri beda, bukan cuma populasi).

**Baseline dihitung ulang lewat `ExecutionSimulator` (mesin SAMA dgn v2, `exit_threshold=1,01` +
SEMUA mekanisme guardian dimatikan -- "tanpa Guardian" yang genuinely apple-to-apple):**

| | PnL total (Apr2025-Jul2026, 16 bulan) |
|---|---:|
| Tanpa Guardian [Fase150, mesin LAMA -- SALAH dibandingkan] | +$305,97 (OOF) + -$26,29 (OOS) = +$279,68 |
| Tanpa Guardian [ExecutionSimulator, mesin SAMA -- BENAR] | **+$505,26** |
| sguardian v2 murni model | +$329,92 |
| **Selisih v2 vs baseline BENAR** | **-$175,33** |

**v2 KALAH, bukan menang.** 10 dari 16 bulan v2 lebih buruk drpd baseline yang benar; kerugian
terbesar Okt 2025 (-$119,56, satu bulan itu saja menyumbang 68% dari total selisih negatif).
Trade v2 SELALU lebih banyak drpd baseline tiap bulan (Guardian ikut campur -> siklus posisi lebih
cepat -> lebih banyak entry per periode, pola sama yg diamati Fase152/153) -- tapi PnL per-trade
turun.

**Kesimpulan REVISI: sguardian v2 TIDAK unggul dari baseline yang benar.** Kesimpulan "belum
terbukti, condong netral-ke-sedikit-positif" di Fase154/155 DICABUT -- itu artefak salah
membandingkan dua mesin simulasi. Dengan mesin yang sama, v2 kalah jelas ($175 dari $505, ~35%
lebih buruk). **Status akhir: NEGATIF.** Tidak direkomendasikan promosi ke produksi maupun
riset lanjutan tanpa desain baru.

**Pelajaran metodologi**: kapan pun membandingkan kandidat ke "baseline"/"tanpa X", WAJIB hitung
baseline itu lewat mesin evaluasi YANG SAMA PERSIS dgn kandidat -- jangan pinjam angka teks dari
skrip lain walau judulnya "baseline resmi". Berlaku mundur ke Fase152/153 juga (baseline OOS
Fase153 dari Fase150 kemungkinan sama-sama tercemar mesin beda, TAPI konklusi Fase153 (v1 gagal
total OOS) sudah cukup ekstrem hingga koreksi ini TIDAK mengubah kesimpulan v1 -- v1 tetap NEGATIF).

**TIDAK ada tindakan holdout/produksi.**

**Artefak.** `pipeline/experiments/dualbin_universe/fase156_sguardian_v2_bulanan.py`.
`models/runs/dualbin_fase156_sguardian_v2_bulanan.json`.

---

## 2026-08-08 — Fase 157 (Fase A rencana "untung tipis"): GERBANG GAGAL, tesis mati di data

**Pemicu.** User usul arah baru pasca sguardian v2 NEGATIF: bukan menebak reversal, tapi kejar
untung tipis (target ATR dekat) lalu cabut cepat (0-3 jam), Guardian cuma "mengawal arah". Rencana
`witty-dreaming-kahn.md` (Plan Mode, Opus/Fable) menyusun 4 fase bergerbang (A=diagnostik murah
tanpa retrain, B=ubah mesin eksekusi, C=retrain entry, D=sguardian pengawal). Fase A dieksekusi
dulu -- SENGAJA murah, supaya tidak lanjut ke retrain kalau tesisnya sendiri sudah mati di data.

**Metodologi**: 24.278 sinyal SHORT dari periode SEBELUM W1 (2024-04-01→2025-03-30, entry model
resep identik Fase150/154, TIDAK mengintip OOF). Simulasi bracket ATR murni (TP/SL berbasis
ATR di 8 kombinasi 0,5-1,5×ATR), aturan keluar sesuai keputusan user: kena TP/SL → keluar; bar≥3
& untung → keluar ambil untung; bar≥3 & masih rugi → tahan (tak pernah dipaksa rugi krn waktu).

**Run 1 (min_profit_frac=0,0, "untung sedikit pun cabut", versi literal permintaan user)**: SEMUA
8 kombinasi PnL rata-rata NEGATIF (-0,12% s/d -0,15%/trade), walau beberapa win-rate kelihatan di
atas ambang impas naif. Diverifikasi penyebabnya: aturan "cabut begitu untung > biaya" memotong
kemenangan jadi RECEHAN (rata-rata kemenangan jauh di bawah TP penuh) sementara kerugian tetap
jalan sampai SL penuh -- asimetri yang timpang. Bukti: bracket 1,0/0,7 WR=59,2% seharusnya hasilkan
+0,169%/trade kalau kemenangan genuinely sebesar TP penuh, tapi aktual -0,140%/trade.

**Run 2 (Opsi 1, atas persetujuan user: cabut hanya kalau untung ≥ min_profit_frac × TP, bukan
cuma "di atas biaya")**: 32 kombinasi (8 bracket × 4 fraksi 0,3/0,5/0,7/1,0) -- **SEMUA TETAP
NEGATIF, 0/32 lolos**. Pola jelas & konsisten: (1) menaikkan `min_profit_frac` MEMBAIKKAN hasil
di tiap bracket (mengonfirmasi diagnosis di atas -- makin lambat "cabut untung", makin baik), TAPI
(2) bahkan di frac=1,0 (murni TP/SL/timeout, TANPA cabut-untung-dini sama sekali) **tetap negatif
di SEMUA 8 bracket** (-0,077% s/d -0,135%/trade). (3) Pola arah bracket: makin LEBAR (1,5/1,5)
makin MENDEKATI impas (-0,077%), makin TIPIS (0,5/0,5) makin RUGI (-0,135%) -- arahnya PERSIS
KEBALIKAN dari tesis awal ("tipis lebih baik"). Data justru menunjuk balik ke arah sistem swing
yang SUDAH ADA (TP jauh), bukan ke arah baru.

**Kesimpulan**: **Gerbang A GAGAL total, bukan cuma "belum meyakinkan".** Bukan soal aturan
keluarnya kurang tepat (sudah dicoba versi paling sabar sekalipun) -- di level fundamental,
lintasan harga H1 pasca-entry SHORT tidak punya cukup edge di skala ATR-tipis utk menutup biaya
transaksi, WALAU sinyal entry-nya sama. **Sesuai gerbang yang disepakati di rencana ("kalau tidak
ada kombinasi yang lolos, BERHENTI -- tidak lanjut retrain"): jalur ini distop di Fase A.** Fase
B/C/D (ubah mesin eksekusi, retrain entry+Guardian) TIDAK dijalankan -- akan sia-sia menghabiskan
waktu retrain utk tesis yang sudah terbukti mati di data mentah.

**SUSULAN (hari sama)**: user minta perluas grid ke bracket LEBAR (2,0/2,0 s/d 3,0/3,0 + varian
asimetris 2,0/3,0 & 3,0/2,0), utk cek apa tren "makin lebar makin baik" (yg kelihatan di run
sebelumnya, 0,5→1,5×ATR) berlanjut. **Jawabannya TIDAK** -- trennya BERBALIK setelah 1,5×ATR:

| Bracket (frac=1,0, paling sabar) | PnL/trade | timeout% |
|---|---:|---:|
| 1,0 / 1,5 | -0,087% | 4,5% |
| **1,5 / 1,5** | **-0,077%** | 9,6% |
| 2,0 / 2,0 | -0,088% | 21,3% |
| 2,5 / 2,5 | -0,096% | 33,0% |
| 3,0 / 3,0 | -0,110% | 44,8% |
| 2,0 / 3,0 (SL>TP) | -0,134% (terburuk) | 30,8% |
| **3,0 / 2,0 (TP>SL)** | **-0,069% (terbaik)** | 33,9% |

Titik paling ringan ada di sekitar **1,5/1,5 s/d 3,0/2,0** (-0,069% s/d -0,077%/trade) -- TETAP
NEGATIF, cuma paling MENDEKATI impas. Lewat titik itu, `timeout%` melonjak tajam (9,6%→44,8%)
krn `MAX_HOLD=24` jam terlalu pendek utk target sejauh 2,5-3×ATR -- hampir separuh trade di
bracket 3,0/3,0 tidak pernah selesai secara alami, dipaksa tutup di harga apa adanya jam ke-24.
Pola RR juga konsisten: TP>SL (3,0/2,0) SELALU lebih baik drpd SL>TP (2,0/3,0) di bobot yg sama.

**Kesimpulan diperkuat**: sudah dijelajahi rentang PENUH 0,5-3,0×ATR (simetris & asimetris,
agresif & sabar) -- TIDAK ADA satu pun titik yang positif. Bentuk kurvanya sendiri (lembah, bukan
naik terus) menunjukkan ini eksplorasi yang SUDAH TUNTAS di ruang bracket ATR-murni, bukan
berhenti prematur. Kesimpulan Gerbang A GAGAL semakin kokoh, bukan cuma dari 1 titik data.

**Susulan ke-2 (hari sama)**: user usul "untung di atas 5%" + tanya kenapa tidak cek per-menit
(kekhawatiran valid: simulasi sblm ini cuma cek harga PENUTUPAN H1, bisa melewatkan target yang
sempat tersentuh lalu berbalik sebelum jam tutup). Diklarifikasi 5% = **untung AKUN** (bukan
harga mentah) — dgn leverage 15x, itu cuma **0,333% pergerakan harga** (lebih tipis dari bracket
manapun yg sudah diuji). Diperbaiki metodologi: pakai HIGH/LOW H1 (bukan cuma close) utk deteksi
tersentuh — jauh lebih dekat ke "cek tiap menit" tanpa perlu data M5 baru (M5 cuma tersedia
≥2025-04-01, akan mengintip OOF kalau dipakai di periode diagnostik ini).

**Efek koreksi high/low pada grid ATR yg sudah ada**: ambang tersentuh JAUH lebih sering terdeteksi
(untung ≥0,5×ATR dlm 3 jam: 41,6%→**63,3%**; rugi ≥0,6×ATR: 33,6%→**52,6%**) — TAPI PnL/trade
malah SEDIKIT LEBIH NEGATIF di hampir semua bracket (mis. 1,5/1,5 frac=1,0: -0,077%→-0,0375%,
justru membaik di titik ini; tapi 0,5/0,5: -0,135%→-0,186%, memburuk) — SL & TP sama-sama lebih
sering tersentuh dgn deteksi lebih akurat, jadi bukan cuma satu sisi yg diuntungkan. Kesimpulan
Gerbang A GAGAL TIDAK berubah dgn metodologi yg lebih akurat ini — bukan artefak under-sampling.

**Target akun "5%" (0,333% harga), 3 lebar SL (3%/5%/8% akun), high/low H1:**

| TP akun | SL akun | TP/SL harga | WR | PnL/trade | PnL$/trade (modal $10) |
|---|---|---|---:|---:|---:|
| 5% | 3% | 0,333%/0,200% | 25,8% | -0,243% | **-$0,364** |
| 5% | 5% | 0,333%/0,333% | 38,7% | -0,256% | **-$0,384** |
| 5% | 8% | 0,333%/0,533% | 53,4% | -0,251% | **-$0,377** |

**SEMUA GAGAL, dan ini yang TERBURUK dari seluruh eksplorasi hari ini** (dibanding rentang
-0,04% s/d -0,19% di bracket ATR 0,5-3,0×). Sesuai prediksi tren yang sudah ditemukan: target
0,333% harga LEBIH TIPIS dari bracket tertipis manapun yg sudah diuji (0,5×ATR≈0,57%), dan
tren "makin tipis makin rugi" berlanjut konsisten sampai ke titik paling ekstrem ini. `frac`
(kesabaran cabut-untung) nyaris tak berpengaruh di sini krn target sangat tipis — hampir semua
trade selesai lewat TP/SL murni dlm hitungan jam, bukan lewat aturan cabut-dini atau timeout.

**Kesimpulan akhir (setelah 3 putaran uji hari ini)**: kekhawatiran metodologi user (cek per-menit)
VALID dan sudah diperbaiki (high/low, bukan cuma close) — tapi TIDAK mengubah arah kesimpulan.
Target berbasis leverage/akun yang tipis (spt "5% akun") justru PALING BURUK dari semua yang
diuji. Edge (kalau ada) di populasi sinyal SHORT ini butuh KESABARAN (bracket lebar/jendela
panjang), bukan kecepatan — mengarah balik ke filosofi sistem swing yang sudah ada, bukan menjauh
darinya. Jalur "kejar untung cepat" ini sudah diuji habis dari berbagai sudut (agresif/sabar,
tipis/lebar, ATR-relatif/akun-tetap, close/high-low) — rekomendasi: hentikan eksplorasi arah ini.

**KOREKSI BESAR (susulan ke-3, hari sama)**: user tegur -- SEMUA simulasi di atas pakai target
TETAP (bracket statis, TP/SL diset saat entry & tidak pernah berubah). Itu BUKAN "mengawal dan
mengikuti arah" yang diminta sejak awal -- belum pernah diuji TRAILING STOP (SL mengikuti puncak,
TANPA batas atas, keluar hanya kalau tren genuinely berbalik). Ditambahkan `simulasi_trailing()`:
SL awal tetap sampai untung (MFE) capai `arm_atr`, lalu "bersenjata" -- SL berpindah mengikuti
`trail_frac x MFE` (dikunci naik terus, tak pernah melonggar), posisi dibiarkan lari TANPA target
atas, keluar kalau harga mundur menembus trailing atau MAX_HOLD habis.

**HASIL PERTAMA POSITIF dari SELURUH eksplorasi hari ini (sguardian v1, v2, bracket tetap,
trailing):**

| Bersenjata | Trail (kunci dari puncak) | SL awal | WR | PnL/trade | PnL$/trade |
|---|---|---|---:|---:|---:|
| 0,3×ATR | 0,7×MFE | 0,7×ATR | 69,2% | **+0,0389%** | **+$0,0583** |
| 0,3×ATR | 0,7×MFE | 1,0×ATR | 74,5% | **+0,0460%** | **+$0,0689** |

Kombinasi lain (arm 0,5/0,8×ATR, atau trail 0,3/0,5×MFE) SEMUA tetap negatif -- pola jelas:
**bersenjata CEPAT (0,3×ATR, jangan tunggu lama) + trailing LONGGAR (kunci cuma 70% dari puncak,
kasih ruang 30% mundur)** yang berhasil. Trailing KETAT (kunci 30-50%) selalu kalah -- keluar
kena noise/whipsaw sebelum tren sungguhan berkembang.

**Status: KANDIDAT, BELUM DIVALIDASI.** Baru 1 populasi (pra-W1, 24.278 sinyal) -- WAJIB dicek
di 3 window OOF independen ([[feedback-wajib-dua-jendela-penilaian]]) sebelum dipercaya sbg
temuan nyata, BUKAN cuma cocok kebetulan di 1 periode. Ini langkah berikutnya, belum dieksekusi.

**TIDAK ada tindakan holdout/produksi. TIDAK ada retrain/perubahan kode -- masih diagnostik murni.**

**Artefak.** `pipeline/experiments/dualbin_universe/fase157_diag_untung_tipis.py`.

---

## 2026-08-08 — Fase B0 (Fase159): dua koreksi penting sebelum lanjut ke Fase B1

**Bug ditemukan & diperbaiki**: `simulasi_trailing` versi awal meng-update MFE pakai `low[j]` DULU
baru cek trailing-SL thd `high[j]` di BAR YANG SAMA -- artinya 1 bar boleh sekaligus "menetapkan
puncak baru" DAN "keluar nyaris tepat di puncak itu", tidak realistis. Diperbaiki: cek exit dulu
pakai status/MFE dari SEBELUM bar ini, baru update MFE untuk bar berikutnya. Efek: PnL/trade turun
~25% (mis. pra-W1 trail=0,99: 0,2412%→0,1818%) tapi pola & kesimpulan arah TIDAK berubah.

**Cek ujung grid (0,7→1,0, titik 1,0 = batas matematis literal)**: kurva **MENDATAR (plateau)**
mendekati 1,0, BUKAN meledak -- pra-W1 trail=0,99→1,00 cuma naik 0,1818%→0,1881% (W2:
0,1912%→0,1970%, W3: 0,1423%→0,1481%). Ini BEDA dari pola berbahaya sguardian v1 (naik terus tanpa
tanda melambat) -- di sini genuinely mendekati asimtot, gerbang B0 LOLOS dengan syarat.

**TEMUAN PENTING (bukan bug, tapi mengubah interpretasi)**: `bars_held_median = 2,0` -- **SAMA
PERSIS di SEMUA trail_frac (0,70 s/d 1,00) dan SEMUA window**, begitu juga rincian alasan
keluar (sl_awal%/trail%) IDENTIK persis di semua trail_frac. Artinya `trail_frac` di rentang ini
TIDAK mengubah KAPAN posisi keluar (median tetap ~2 jam sejak entry) -- cuma mengubah harga
persisnya (kunci lebih ketat = tangkap harga lebih dekat ke puncak dalam jendela ~2 jam yang
sama). **Mekanismenya BUKAN "ikuti tren panjang" spt namanya** -- ini genuinely "tangkap gerakan
awal, keluar dekat puncak lokal dlm ~2 jam", jauh lebih dekat ke ide "untung cepat" yang sudah
dicoba sebelumnya (Fase157 bracket) -- bedanya di SINI berhasil krn menangkap HARGA TERBAIK (via
peak-following) bukan target sembarang, bukan krn benar2 menahan tren lama.

**Implikasi ke Fase B1**: parameter existing `near_tp_arm_frac`/`near_tp_lock_frac` bersenjata
relatif JARAK-TP (bisa berjam-jam), BUKAN relatif ATR + horison pendek (~2 jam) spt temuan ini --
perlu diverifikasi eksplisit apa parameter existing bisa mereplikasi perilaku "arm cepat, exit ~2
jam kemudian dekat puncak" ini, atau apa horison waktunya secara struktural berbeda.

**TIDAK ada tindakan holdout/produksi. TIDAK ada retrain/perubahan kode -- masih diagnostik murni.**

**Artefak.** `pipeline/experiments/dualbin_universe/fase159_b0_grid_ujung.py`.

---

## 2026-08-08 — Fase 160: KOREKSI TOTAL -- trailing stop di M5 asli TERBALIK jadi negatif

**Pemicu.** User minta "buka kemungkinan cek data tiap beberapa menit". Ditemukan lebih dulu:
sistem SUDAH punya `position_monitor` polling 5 menit (`app/config.py` `ExecutionCfg.
monitor_interval_min=5`, `monitor_enabled=True`), TAPI scope-nya `monitor_scope="sl_tp"` --
logika Guardian/trailing MASIH cuma dievaluasi tiap jam (H1), tidak pernah di 5 menit. M5 asli
kebetulan tersedia PERSIS di window OOF (2025-04-01→2026-03-31) -- tidak perlu data baru.

**Metodologi**: mekanisme trailing SAMA PERSIS Fase159 (arm=0,3×ATR, SL awal=1,0×ATR, urutan
cek-lalu-update yg sudah diperbaiki), cuma resolusi pengecekan diganti dari H1 high/low (per jam)
jadi M5 asli (per 5 menit) -- data sungguhan, bukan pendekatan.

**Hasil: TERBALIK TOTAL jadi NEGATIF di semua grid & semua window.**

| Window | trail=0,99 (terbaik) | Lama ditahan (median) |
|---|---:|---:|
| W1 | -$0,089/trade | 25 menit |
| W2 | -$0,021/trade | 25 menit |
| W3 | -$0,062/trade | 30 menit |

Lama ditahan anjlok dari **~2 jam** (Fase159, resolusi H1) jadi **25-30 menit** (M5 asli) --
SEMUA titik grid (0,70 s/d 1,00) NEGATIF di SEMUA 3 window, tidak ada satu pun yang lolos.

**Penjelasan**: temuan "trailing stop menang" di Fase157-159 adalah **artefak kekasaran
pengecekan per-jam**, BUKAN edge sungguhan. Cek H1 (bahkan pakai high/low) tidak bisa bereaksi
ke goyangan DALAM jam itu -- efeknya seperti "masa tenggang" 1 jam yang menyembunyikan noise
jangka-pendek, membuat mekanisme kelihatan "menahan tren" padahal sebenarnya cuma tidak sempat
bereaksi ke pembalikan cepat. Begitu dicek di resolusi sungguhan (5 menit, SAMA dgn yang akan
dipakai kalau ini benar2 di-deploy lewat `position_monitor`), goyangan dalam 25-30 menit pertama
ternyata cukup sering membalik & biaya transaksi memakan seluruh untungnya.

**Konsekuensi**: **Fase B1 (sweep `near_tp_arm_frac`/`near_tp_lock_frac`/`momentum_floor_frac`
existing) DIBATALKAN** -- tidak ada gunanya menyetel parameter produksi (yang notabene cuma
dievaluasi tiap jam) untuk mereplikasi mekanisme yang justru TERBUKTI HANYA menang krn resolusi
kasar. Ini juga BUKAN argumen "coba pasang di resolusi 5 menit sungguhan" -- sudah diuji LANGSUNG
di resolusi itu (M5 asli), hasilnya negatif, bukan cuma didekati.

**Status jalur trailing stop: NEGATIF, sudah diuji tuntas** (H1 approksimasi awal salah [bug
same-bar], H1 diperbaiki [masih positif tapi ternyata artefak], M5 asli [negatif, jawaban
final]). Menutup seluruh cabang eksplorasi hari ini (sguardian v1, v2, bracket tetap, target
akun, trailing H1, trailing M5) — SEMUA berujung negatif atau tidak terbukti di resolusi yang
realistis.

**TIDAK ada tindakan holdout/produksi. TIDAK ada retrain/perubahan kode.**

**Artefak.** `pipeline/experiments/dualbin_universe/fase160_trailing_m5.py`.

---

## 2026-08-09 — Fase 161: peta target profit M5 sebelum LSTM

**Tujuan.** Mengukur kelayakan target profit yang diusulkan user secara M5 nyata sebelum membuat label sequence LSTM. SHORT entry dibangun walk-forward seperti Fase158/160 pada tiga window OOF W1/W2/W3; tidak ada holdout, retrain production, atau deploy.

**Hasil.** Pada leverage 15x, +5% margin ROI (= +0,333% harga) disentuh oleh 90,5–91,7% sinyal dalam median 0,5–0,67 jam; +10% (= +0,667% harga) 82,0–83,2% dalam 1,67–1,92 jam; +15% (= +1,0% harga) 73,7–75,3% dalam 3,0–3,5 jam. Sebaliknya target notional/harga: +2% hanya 52,9–55,0% (median 6,5–7,7 jam), +5% hanya 15,6–20,8% (11,7–15,1 jam), dan +10% hanya 1,3–3,8% (17,1–17,7 jam).

**Keputusan gerbang.** +5% margin terlalu mudah/cepat untuk membedakan tren; +5–10% notional terlalu jarang dan lambat untuk klaim “cepat keluar”. Sesudah Fase160 menunjukkan trailing M5 negatif, LSTM **tidak** dilatih untuk meniru kebijakan itu. Tahap LSTM ditunda sampai ada desain exit M5 yang lolos baseline causal; tidak ada perubahan produksi.

**Artefak.** `pipeline/experiments/dualbin_universe/fase161_peta_profit_threshold_m5.py`; `models/runs/dualbin_fase161_profit_threshold_m5.json`.

---

## 2026-08-09 — Fase 162: TP-limit langsung 5–15% margin ROI di M5 juga NEGATIF

**Pertanyaan user.** Bukan trailing: begitu trade dimulai, langsung pasang `reduce-only` take-profit limit pada +5–10% margin ROI. Ini diuji literal pada SHORT entry walk-forward yang sama, M5 nyata, maksimum tahan 24 jam, fee+slippage dua sisi, dan protective stop. Jika TP dan SL sama-sama disentuh dalam satu candle M5, SL diasumsikan lebih dahulu (konservatif karena urutan tick tidak tersedia).

**Grid.** TP margin ROI 5/10/15% (pada 15x berarti gerak harga +0,333/+0,667/+1,000%) × SL 3/5/8% ROI. W1/W2/W3 OOF, tanpa holdout atau perubahan produksi.

**Hasil.** Seluruh 27 cell (9 kombinasi × 3 window) negatif. Kombinasi paling ringan, TP=15%/SL=8%, masih W1=-0,1455%, W2=-0,1108%, W3=-0,1398% per trade. TP=5%/SL=8% mencatat win rate 62,6–65,2%, tetapi tetap -0,1481%/-0,1479%/-0,1705% per trade.

**Mengapa.** Target +5% margin hanya +0,333% harga. Setelah fee+slippage dua sisi, kemenangan kecil itu tidak cukup menutup kerugian yang terjadi sebelum TP tercapai; melonggarkan SL memang menaikkan win rate, tetapi memperbesar kerugian saat kalah. Target 10–15% memberi payoff lebih besar, namun proporsi sinyal yang terkena stop terlebih dahulu naik menjadi sekitar 52–83% tergantung konfigurasi. Jadi ini bukan masalah trailing: direct TP-limit pun tidak memiliki edge net pada populasi entry saat ini.

**Keputusan gerbang.** Jangan retrain LSTM untuk entry/exit ini sebelum ada baseline M5 yang positif. LSTM tidak boleh dipakai sebagai alasan untuk melewati kegagalan policy dasar. Tidak ada perubahan produksi.

**Artefak.** `pipeline/experiments/dualbin_universe/fase162_tp_limit_m5.py`; `models/runs/dualbin_fase162_tp_limit_m5.json`.

---

## 2026-08-09 — Fase 163: filter arah H4 memperbaiki, tetapi belum membentuk populasi trend-following yang layak

**Tujuan.** Menjawab koreksi user bahwa strategi harus mengikuti tren saat itu, bukan memberi exit cepat pada semua sinyal SHORT. Exit M5 TP-limit identik Fase162 (TP 5/10/15% margin ROI, SL 8%); yang diubah HANYA populasi entry. Tiga populasi: raw LGBM; `h4_aligned` (trend_strength negatif, slope EMA21/EMA50 H4 negatif, harga di bawah EMA50 H4); dan `h4_breakdown_24h` (h4_aligned + close menembus low 24H sebelumnya, hanya sinyal pertama per 24 jam).

**Hasil arah H4.** Dari 8.165/6.505/6.788 raw signal menjadi 482/995/1.174 sinyal. Hasil membaik, terutama TP15/SL8: W1 -0,1455% → -0,0602%, W2 -0,1108% → -0,0758%, W3 -0,1398% → -0,1224% per trade. Namun semua masih negatif; filter arah saja belum cukup menutup biaya M5.

**Hasil breakout 24H.** Populasi hanya 3/7/9 trade di W1/W2/W3. Ada hasil positif pada W2/W3 untuk TP10/SL8 (+0,1438%/+0,2200%), tetapi sampel ini terlalu kecil untuk disebut edge atau menjadi data LSTM. W1 hanya 3 trade; tidak sah untuk tuning.

**Keputusan gerbang.** Hipotesis utama benar sebagian: raw LGBM sebelumnya bukan populasi tren yang bersih, dan arah H4 membantu. Namun definisi breakout 24H terlalu jarang, sedangkan arah H4 tanpa breakout tetap negatif. Jangan melatih LSTM dari 19 trade breakout atau dari populasi h4_aligned yang masih rugi. Tahap berikut bila dilanjutkan harus mencari definisi *continuation* yang cukup sering (bukan breakout ekstrem) dan mengujinya dengan baseline M5 sebelum LSTM. Tidak ada perubahan produksi.

**Artefak.** `pipeline/experiments/dualbin_universe/fase163_filter_trend_breakout_m5.py`; `models/runs/dualbin_fase163_filter_trend_breakout_m5.json`.

---

## 2026-08-09 — Fase 164: cross-sectional momentum + regime BTC tidak konsisten, tidak lanjut trend-swing/LSTM

**Tujuan.** Mencari alternatif trend-following yang tidak memberi TP cepat kepada semua sinyal: SHORT hanya pada koin relatif terlemah ketika BTC sendiri H4 turun. Entry per koin dideduplikasi (maksimum satu per 24 jam pada arus sama). Exit M5 TP/SL tetap Fase162 supaya perubahan hanya berasal dari filter arus.

**Filter.** `cs24_bottom3_btc_down` = bottom-3 return 24H + BTC trend_strength/slope EMA21/slope EMA50/posisi EMA50 semuanya bearish. Varian `cs_combo` memakai ranking gabungan return 6H+24H. Dibandingkan dengan raw entry yang juga dideduplikasi.

**Hasil.** Raw fresh tetap negatif di seluruh window. `cs24_bottom3_btc_down` sempat positif W1 pada TP10/SL8 +0,0581% dan TP15/SL8 +0,1629%, tetapi hanya 14 trade dan runtuh pada W2 (36 trade, -0,1800%/-0,1746%) serta W3 (47 trade, -0,1772%/-0,1913%). Varian combo negatif di semua window. Ini bukan edge lintas-window; W1 adalah kandidat sampel kecil yang tidak terkonfirmasi.

**Keputusan gerbang.** Fase 164 GAGAL. Jangan lanjut Fase165 trend-swing maupun LSTM, karena alternatif entry trend yang diuji tidak positif konsisten. Tidak ada holdout, perubahan produksi, atau retrain production.

**Artefak.** `pipeline/experiments/dualbin_universe/fase164_cross_section_trend_m5.py`; `models/runs/dualbin_fase164_cross_section_trend_m5.json`.

---

## 2026-08-09 — Fase 165: oracle trend-swing H4 gagal; LSTM continuation tidak dilanjutkan

**Tujuan.** Mengikuti rekomendasi reset strategi: sebelum LSTM, uji apakah kandidat yang secara kausal sudah searah tren H4 memang mempunyai peluang dasar untuk swing. Kandidat LONG/SHORT dipilih saat trend_strength, slope EMA21 H4, slope EMA50 H4, dan posisi terhadap EMA50 H4 semuanya searah; satu kandidat per koin tiap 24 jam. Oracle hanya memberi LABEL masa depan (bukan strategi live): apakah +2% harga disentuh sebelum -1% harga, melalui M5 nyata, dalam 48/96 jam. TP dan SL dalam candle M5 sama diasumsikan SL dulu.

**Hasil.** Gabungan LONG+SHORT TP sebelum SL hanya W1 29,6%/30,6% (48/96 jam), W2 32,7%/33,3%, dan W3 31,9%/32,7%, dari 444–480 kandidat per window. LONG 25,6–34,9%; SHORT 31,5–36,1%. Horizon lebih panjang nyaris tidak mengubah hasil karena banyak posisi sudah terkena -1% lebih dahulu.

**Keputusan gerbang.** Dengan target +2%, stop -1%, dan biaya dua sisi, tingkat impas sekitar 39%; oracle 30–33% berada jauh di bawahnya. Ini berarti definisi trend H4 sederhana sendiri belum mengandung peluang yang cukup bahkan SEBELUM kesalahan prediksi LSTM. Jangan buat label/train LSTM continuation dari populasi ini; LSTM tidak dapat secara jujur memperbaiki base rate yang gagal tanpa bukti filter baru. Tidak ada holdout, perubahan produksi, atau deploy.

**Artefak.** `pipeline/experiments/dualbin_universe/fase165_oracle_trend_swing_m5.py`; `models/runs/dualbin_fase165_oracle_trend_swing_m5.json`.

---

## 2026-08-09 — Fase 166A/B: event breakout trend menghasilkan baseline positif lintas-window

**Tujuan.** Memperbaiki kesalahan formulasi sebelumnya: bukan sinyal berulang setiap jam, melainkan satu event trend-following yang dapat diperdagangkan. Aturan dibekukan sebelum run: arah H4 sejalan (trend_strength, slope EMA21/50, posisi EMA50), range H1 12 jam terkompresi <=80% median 72 jam sebelumnya, close H1 breakout range, dan volume >=1,2x rerata 24 jam. Satu event per arah/koin/24 jam. Entry hanya memakai informasi yang sudah selesai; M5 dipakai untuk replay exit nyata.

**Eksekusi baseline.** Stop = sisi lawan range kompresi (risiko dibatasi 0,5–3,0% harga), TP = 2R, horizon maksimum 96 jam. Bila TP dan stop tersentuh dalam M5 yang sama, stop diasumsikan dahulu. Ini disebut oracle hanya karena hasil masa depan dipakai sebagai label kelanjutan berikutnya; replay bracket baseline-nya sendiri dapat dieksekusi secara kausal.

| Window | Event | TP sebelum SL | PF | Net/trade |
|---|---:|---:|---:|---:|
| W1 | 244 | 36,1% | **1,023** | **+0,0323%** |
| W2 | 225 | 36,9% | **1,045** | **+0,0596%** |
| W3 | 271 | 43,2% | **1,380** | **+0,4376%** |

Per arah: LONG positif di seluruh window (PF 1,118/1,226/1,313); SHORT negatif W1/W2 (PF 0,938/0,905) lalu kuat W3 (1,411). Karena itu baseline gabungan lolos sangat tipis di W1 dan belum layak dipromosikan, tetapi ini adalah **baseline pertama yang positif konsisten** setelah seluruh jalur raw-signal/scalp gagal.

**Keputusan.** Jangan deploy dan jangan simpulkan edge matang: W1 masih dekat impas dan SHORT tidak konsisten. Namun jalur event breakout layak menjadi basis LSTM continuation, bukan entry lama. LSTM belum dilatih karena M5 label hanya tersedia mulai 2025-04-01; untuk train walk-forward W1 yang jujur diperlukan M5 pre-W1 atau sumber label eksekusi yang setara. Mengganti training dengan label H1 akan menciptakan train/eval mismatch, sehingga tidak dilakukan.

**Artefak.** `pipeline/experiments/dualbin_universe/fase166_event_breakout_oracle_m5.py`; `models/runs/dualbin_fase166_event_breakout_oracle_m5.json`.

---

## 2026-08-09 — Fase 167: LSTM continuation tidak menambah nilai; baseline event tetap anchor

**Data & protokol.** Histori M5 pre-W1 ternyata sudah tersedia (`m5_train_20240401_20250401`), sehingga dibuat 1.401 label event M5 2024-04..2026-04. LSTM kecil (sequence 48 H1) hanya membaca data sampai timestamp event dan memprediksi TP 2R sebelum stop struktural. Fitur: return 1/6H searah posisi, ATR%, ratio volume, trend_strength dan slope/posisi EMA H4 searah posisi. Per fold, label train dipurge 96 jam sebelum validasi. Baseline Fase166 dibekukan; ambang `p>=0,60` dipra-tetapkan, BUKAN dipilih dari OOF profit.

| Window | Baseline PF / net-trade | LSTM p>=0,60 | Vonis |
|---|---:|---:|---|
| W1 | 1,033 / +0,0451% (243) | 0 event | gagal memilih |
| W2 | 1,045 / +0,0596% (225) | PF 0,541 / -0,6318% (8) | merugikan |
| W3 | 1,356 / +0,4148% (273) | PF 0,373 / -0,7099% (14) | merugikan |

**Keputusan.** LSTM tidak memperbaiki event baseline dan tidak boleh dipakai/deploy. Tidak dilakukan threshold tuning setelah melihat hasil: setiap profit tunggal bukan anchor tuning. **Anchor yang berlaku** adalah baseline event Fase166 yang dibekukan; perubahan hanya boleh diterima jika meningkatkan PF/net-trade secara konsisten pada W1/W2/W3 dengan populasi dan biaya yang sama. Tidak ada holdout atau perubahan produksi.

**Artefak.** `pipeline/experiments/dualbin_universe/fase167_lstm_event_continuation_oof.py`; `models/runs/dualbin_fase167_lstm_event_continuation_oof.json`.

---

## 2026-08-09 — Fase 168: hanya compression breakout lolos; Donchian/pullback ditolak

**Protokol frozen.** Tiga keluarga event diuji dengan replay M5, TP 2R, stop struktural, 96 jam, biaya sama, dan LONG/SHORT terpisah: (1) compression breakout Fase166; (2) Donchian 24H breakout + volume; (3) pullback EMA21 lalu continuation. Ini pemilihan keluarga event, bukan sweep parameter.

| Keluarga | W1 PF / net | W2 PF / net | W3 PF / net | Vonis |
|---|---:|---:|---:|---|
| Compression breakout | **1,023 / +0,0323%** | **1,045 / +0,0596%** | **1,380 / +0,4376%** | Lolos, anchor tetap |
| Donchian24 breakout | 0,929 / -0,0951% | 1,084 / +0,1119% | 1,585 / +0,6105% | Ditolak: W1 negatif |
| EMA21 pullback continuation | 0,920 / -0,0938% | 0,834 / -0,2062% | 0,914 / -0,0988% | Ditolak |

**Keputusan.** Jangan gabungkan atau tune keluarga yang gagal untuk mengejar hasil W2/W3. Hanya compression breakout dilanjutkan ke audit jumlah koin/universe. Tidak ada holdout atau perubahan produksi.

**Artefak.** `pipeline/experiments/dualbin_universe/fase168_event_family_oracle_m5.py`; `models/runs/dualbin_fase168_event_family_oracle_m5.json`.

---

## 2026-08-09 — Fase 169: 18-koin LONG-only lolos; core-8 tidak konsisten

**Tujuan.** Audit jumlah koin dan arah pada compression breakout frozen Fase166. Grup ditetapkan sebelum hasil: seluruh 18 koin long+short, seluruh 18 long-only, core-8 likuid long+short, core-8 long-only.

| Universe | W1 PF / net | W2 PF / net | W3 PF / net | Vonis |
|---|---:|---:|---:|---|
| 18 both | 1,023 / +0,0323% | 1,045 / +0,0596% | 1,380 / +0,4376% | baseline lama |
| **18 LONG-only** | **1,118 / +0,1443%** | **1,226 / +0,2841%** | **1,313 / +0,3437%** | Lolos konsisten |
| Core-8 both | 1,052 / +0,0685% | 1,297 / +0,3669% | 1,679 / +0,7133% | Kandidat, belum dipilih karena jumlah koin berubah |
| Core-8 LONG-only | 0,936 / -0,0813% | 2,134 / +1,0664% | 1,883 / +0,8423% | Ditolak: W1 negatif |

**Keputusan.** Anchor kerja berikutnya adalah **compression breakout, 18 koin, LONG-only**. SHORT dikeluarkan dari jalur ini karena tidak stabil sebelumnya. Core-8 tidak dipromosikan meski hasil gabungan hijau; ini masih satu audit dan memerlukan konfirmasi terpisah, bukan alasan memilih universe sesudah melihat hasil. Tidak ada perubahan produksi.

**Artefak.** `pipeline/experiments/dualbin_universe/fase169_universe_event_audit_m5.py`; `models/runs/dualbin_fase169_universe_event_audit_m5.json`.

---

## 2026-08-09 — Fase 170: LightGBM event ranking gagal konsistensi; ML ditutup

**Protokol.** Event anchor Fase169 (18-koin LONG-only) dipakai tanpa perubahan exit. LightGBM membaca snapshot, mean, std, dan perubahan fitur sequence 48H. Tidak ada threshold yang dipilih dari profit: setiap window mengambil top 40% probabilitas prediksi, rasio partisipasi dibekukan sebelum run.

| Window | Baseline PF / net (n) | LGBM top-40% PF / net (n) |
|---|---:|---:|
| W1 | 1,118 / +0,1443% (131) | 1,144 / +0,1784% (53) |
| W2 | 1,226 / +0,2841% (104) | **0,851 / -0,2151% (42)** |
| W3 | 1,313 / +0,3437% (90) | 1,390 / +0,3498% (36) |

**Keputusan.** Ditolak. Perbaikan W1/W3 tidak boleh menutupi pembalikan W2. Bersama Fase167 (LSTM juga gagal), dua keluarga model ML tidak menambah nilai di atas rule anchor. Jangan lanjut TCN/LSTM/seed/threshold sweep; itu hanya mengejar hasil historis. Baseline rule tetap satu-satunya kandidat riset yang bertahan. Tidak ada holdout atau perubahan produksi.

**Artefak.** `pipeline/experiments/dualbin_universe/fase170_lgbm_long_event_rank_oof.py`; `models/runs/dualbin_fase170_lgbm_long_event_rank_oof.json`.

---

## 2026-08-09 — Fase 171: bootstrap menolak promosi anchor LONG-only

**Tujuan.** Menguji apakah profit point-estimate anchor Fase169 (compression breakout, 18 koin, LONG-only) cukup stabil setelah dependensi event diperhitungkan. Tidak ada perubahan parameter, model, exit, atau universe.

**Metodologi.** Cluster/block bootstrap sebanyak 3.000 kali pada unit `coin-month`, agar beberapa event dari koin dan bulan yang sama tidak diperlakukan sebagai observasi independen. Setiap replikasi menghitung kembali net return per trade dan profit factor; interval 95% persentil menjadi gerbang robustness.

| Window | Cluster | Point estimate net / PF | 95% CI net | 95% CI PF |
|---|---:|---:|---:|---:|
| W1 | 53 | +0,1443% / 1,118 | -0,4577% s.d. +0,7512% | 0,690 s.d. 1,743 |
| W2 | 51 | +0,2841% / 1,226 | -0,4156% s.d. +1,0190% | 0,731 s.d. 2,032 |
| W3 | 44 | +0,3437% / 1,313 | -0,2012% s.d. +0,9256% | 0,849 s.d. 2,007 |

**Keputusan.** Belum lolos. Ketiga interval mencakup net negatif dan PF di bawah 1, sehingga tiga point estimate hijau Fase169 belum merupakan bukti edge yang cukup kuat. Anchor dibekukan hanya sebagai **kandidat riset**, bukan sinyal untuk dipromosikan, dituning lagi, atau dideploy. Langkah yang sah berikutnya adalah evaluasi pada periode data benar-benar baru / paper-forward dengan aturan ini dibekukan; bukan mencoba parameter atau model tambahan pada data yang sama. Tidak ada holdout tersegel atau produksi disentuh.

**Artefak.** `pipeline/experiments/dualbin_universe/fase171_anchor_robustness.py`; `models/runs/dualbin_fase171_anchor_robustness.json`; manifest pembekuan `models/runs/dualbin_trend_event_long_candidate_frozen_20260809.json`; scorecard terkonsolidasi `models/runs/dualbin_trend_event_long_candidate_scorecard_20260809.json`.

---

## 2026-08-09 — Fase 172: paper-forward pasif kandidat trend-event dimulai

**Tujuan.** Mengumpulkan bukti independen untuk kandidat beku `trend_event_long_18coin_v1_frozen_20260809`, tanpa order riil dan tanpa menyentuh model/config/DB produksi.

**Protokol beku.** Monitor membaca public Binance USD-M pada candle tertutup: 18 koin, LONG-only; H4 trend positif (EMA7–50/ATR, slope EMA21, slope EMA50, harga vs EMA50), H1 compression 12 jam <=80% median 72 jam, breakout high range dengan volume >=1,2x rata-rata 24H. Stop = low range sebelumnya, hanya risiko 0,5–3%; target 2R; maksimum 96H; M5 replay, biaya+slippage, dan SL dulu jika TP+SL dalam M5 yang sama. Event disimpan sekali per koin/24H. Tidak ada API key ataupun fungsi pengiriman order dalam skrip.

**Start snapshot.** 2026-08-09 01:46:20 UTC. Dry-run dan run resmi terhadap seluruh 18 koin berhasil; **0 event** memenuhi rule pada snapshot awal. Ini bukan hasil PnL dan bukan kegagalan: kandidat memang event-driven dan selektif. Scorecard awal: 0 closed/open, belum eligible.

**Gerbang.** Periode tidak boleh dituning. Evaluasi paling cepat setelah >=4 bulan dan >=100 event closed; lalu wajib net positif serta lower CI bootstrap untuk net >0 dan PF >1. Sampai gerbang itu tercapai, status tetap `research_candidate_frozen_unconfirmed`, bukan paper model lama dan bukan produksi.

**Operasional.** Task Scheduler lokal `RisetTrendEventPaperForward` aktif tiap jam pada menit 05 WITA (start 2026-08-09 10:05 WITA); batas eksekusi 10 menit, instance baru diabaikan jika run lama belum selesai. Ia hanya menjalankan skrip riset read-only ini, tanpa API key/order.

**Artefak.** `pipeline/experiments/dualbin_universe/paper_forward_trend_event_long.py`; `models/runs/dualbin_trend_event_long_paper_forward_20260809/state.json`; `models/runs/dualbin_trend_event_long_paper_forward_20260809/scorecard.json`.

---

## 2026-08-09 — Fase 164: breakeven-lock diuji ULANG dgn metodologi BENAR -- KONFIRMASI: memperburuk di semua varian, semua window

**Catatan penomoran.** Nomor fase 161-163 di atas dipakai sesi paralel lain (topik beda: target
profit M5/LSTM) -- entri ini melompat ke 164 utk hindari tabrakan. Skrip riset entri ini bernama
`fase161_breakeven_lock_oof_correct.py` (dibuat sebelum tabrakan nomor ketahuan), TIDAK diubah
namanya -- isi & hasil skrip tidak terpengaruh, cuma label dokumentasi ini.

**Pemicu.** User marah: performa live dualbin makin memburuk sejak awal deploy. Investigasi
temukan `breakeven_lock_mfe_pct` (dideploy 2026-08-05) TERBUKTI memperburuk sehari kemudian
(2026-08-06, fase119, metodologi jujur) tapi TAK PERNAH dimatikan -- 3 hari aktif tanpa keputusan.
**Dimatikan di live** (config VPS, `breakeven_lock_mfe_pct: 0,005→0,0`, restart terverifikasi,
DEPLOY_LOG.md). User minta: uji ulang dengan metodologi yang benar SEJAK AWAL (bukan cuma
reproduksi temuan lama) -- apakah breakeven-lock genuinely tak bisa diselamatkan, atau cuma
parameter lama yang salah?

**Metodologi (3 koreksi dari sweep asli yg dulu keliru "menang")**: (1) `ExecutionSimulator` M5
real, bukan `replay_arrays()` optimistis. (2) `execution.monitor_enabled=True` -- dicek LANGSUNG
ke config VPS, monitor SL/TP 5-menit MEMANG aktif di live sekarang (default, tak ada override),
BEDA dari fase119 yang mematikannya "sesuai keputusan user saat itu". (3) OOF 3 window (W1/W2/W3),
BUKAN holdout tersegel -- entry LONG+SHORT dilatih ULANG per window (rolling 48bln,
LSR_GENUINE_START). Baseline dihitung ULANG di mesin yang SAMA (pelajaran Fase156).

**Hasil (total 3 window, gerbang PF>1 & PnL>0):**

| Konfigurasi | Trade | PnL total | Window lolos |
|---|---:|---:|---:|
| **Mati (baseline)** | 4.703 | **-$648,87** | 0/3 |
| Lama (nilai live 08-05, `mfe=0,005/buffer=0,03`) | 9.250 | -$2.895,78 | 0/3 |
| mfe lebih tinggi (0,015), buffer sama | 5.530 | -$1.671,71 | 0/3 |
| mfe lama, buffer lebih kecil (0,01) | 9.161 | -$2.790,59 | 0/3 |
| mfe & buffer lebih tinggi (0,015/0,01) | 5.425 | -$1.364,84 | 0/3 |

**Baseline (mati) MENANG di SEMUA 3 window secara terpisah**, bukan cuma total -- 4 varian
breakeven-lock yang diuji (termasuk 2 titik BARU yang belum pernah dicoba) SEMUA lebih buruk,
2,1x sampai 4,5x lipat kerugian baseline. Bukan soal parameter yang salah -- mekanismenya sendiri
(mengunci SL begitu MFE nyentuh ambang) struktural merugikan di sistem ini, konsisten dgn root
cause fase119 (memotong keputusan Guardian sendiri yang justru lebih baik).

**Temuan tambahan (di luar scope breakeven-lock, PERLU DITINDAKLANJUTI TERPISAH)**: baseline
(breakeven-lock mati) SENDIRI masih PF<1 di ketiga window (W1 0,88 / W2 0,91 / W3 0,97 -- membaik
tapi belum lolos gerbang manapun). Ini kombinasi entry LONG+SHORT + Guardian produksi APA ADANYA
(momentum_floor_frac=0,10, near_tp_arm=0,95/lock=0,20) diuji jujur dgn monitor 5-menit ON --
BUKAN cuma soal breakeven-lock. Kemungkinan kontributor lain thd keluhan user "makin memburuk":
`momentum_floor_frac` 0,30→0,10 (deploy sama 08-05), near-TP floor (08-03), atau entry model
`long22f_swing`/`short20f_swing` (08-05) itu sendiri. **Belum diinvestigasi -- perlu sesi
terpisah**, jangan simpulkan breakeven-lock sbg SATU-SATUNYA penyebab.

**Kesimpulan: KONFIRMASI TUNTAS.** Mematikan `breakeven_lock_mfe_pct` di live (08-09) adalah
keputusan yang benar dan divalidasi ulang dgn metodologi yang benar-benar jujur sejak awal --
bukan cuma reproduksi temuan lama. Jangan coba tuning ulang breakeven-lock lagi tanpa perubahan
desain (mis. jadi exchange-side resting order spt swint_tradev2, bukan polling in-loop -- lihat
memory `reference-swint-floor-tp-exchange-side-order`).

**TIDAK ada tindakan holdout/produksi lebih lanjut (holdout tersegel TIDAK disentuh).**

**Artefak.** `pipeline/experiments/dualbin_universe/fase161_breakeven_lock_oof_correct.py`.
`models/runs/dualbin_fase161_breakeven_lock_oof_correct.json`.

---

## 2026-08-09 — Fase 166: sguardian v2 diuji ULANG dgn `monitor_enabled=True` -- BUKAN netral, TERBUKTI memperburuk

**Catatan penomoran.** Skrip bernama `fase164_sguardian_v2_monitor_correct.py` (dibuat sebelum
tabrakan nomor dgn sesi paralel ketahuan) -- entri log ini pakai 166 supaya tidak tabrakan dgn
entri "Fase 164" (breakeven-lock) di atas maupun Fase165 sesi paralel.

**Pemicu.** Ditemukan 2026-08-09: SEMUA skrip sguardian hari sebelumnya (Fase151-156) eksplisit
`monitor_enabled=False` -- tidak cocok kondisi live (SL/TP dicek tiap 5 menit, bukan nunggu H1).
Kesimpulan lama Fase156 ("belum terbukti, condong netral") jadi diragukan, sama seperti
breakeven-lock & trailing stop yang terbalik begitu diuji dgn setting yg benar. User minta
diuji ulang.

**Metodologi.** `muat_cfg_live(monitor_enabled=True, monitor_scope="sl_tp")` -- cocok config VPS
live sekarang. Guardian v2 (model sudah tersimpan dari Fase154, TIDAK dilatih ulang) vs baseline
tanpa Guardian, keduanya dihitung ULANG di mesin (`ExecutionSimulator`) & window yang SAMA
(`fase154`/`fase156` punya `cfg_baseline`/`cfg_lengan_c`, dipakai apa adanya). Entry SHORT
dilatih ulang per window (murah, rolling 48bln, seed=42).

**Hasil (3 window OOF, gerbang PF>1 & PnL>0):**

| | W1 | W2 | W3 | Total |
|---|---:|---:|---:|---:|
| Tanpa Guardian (baseline) | -$293,78 (PF 0,82) | +$330,37 (PF 1,26) ✅ | +$166,32 (PF 1,15) ✅ | **+$202,92**, 2/3 lolos |
| sguardian v2 (thr 0,55) | -$345,43 (PF 0,79) | +$231,31 (PF 1,19) ✅ | +$149,66 (PF 1,14) ✅ | **+$35,55**, 2/3 lolos |
| Selisih (v2 − baseline) | -$51,65 | -$99,06 | -$16,66 | **-$167,37** |

**Guardian v2 KALAH di SEMUA 3 window secara terpisah**, tidak cuma total -- bukan lagi "condong
netral" (Fase156) begitu monitor 5-menit dihidupkan sesuai kondisi live nyata. Konsisten dgn pola
breakeven-lock & trailing-H1 hari ini: metodologi lama (monitor mati / resolusi kasar) systematically
melebih-lebihkan mekanisme tambahan yang reaktif terhadap harga.

**Temuan sampingan (bug infra, bukan soal PF).** Log run memunculkan error berulang: *"Guardian
memilih PARTIAL EXIT tapi jalur real-time tidak bisa mengeksekusinya ke bursa -- posisi DIBIARKAN
UTUH"*. Artinya sebagian keputusan partial-exit Guardian v2 di simulasi ini TIDAK benar-benar
tereksekusi (posisi dibiarkan utuh, cuma diawasi ulang) -- kalau Guardian v2 pernah didorong ke
live, partial-exit perlu diimplementasikan sungguhan (reduceOnly sebagian) dulu, bukan
diasumsikan jalan.

**Keputusan gerbang.** Fase166 GAGAL (guardian v2 kalah di 3/3 window). Jangan pakai sguardian v2
sbg tambahan live. Baseline (tanpa Guardian filter tambahan, cuma Guardian produksi apa adanya)
tetap yang TERBAIK dari semua yang diuji ulang dgn metodologi benar hari ini -- tapi baseline
itu sendiri masih GAGAL gerbang di W1 (PF 0,82), jadi "terbaik" di sini bukan berarti "lolos".

**Sintesis lintas Fase160/161(164)/166 hari ini (metodologi monitor/resolusi benar, entry LONG+SHORT
+ Guardian produksi apa adanya sbg baseline bersama):** trailing stop (H1→M5), breakeven-lock, dan
sguardian v2 SEMUA terbukti memperburuk begitu diuji jujur. Tidak ada satu pun mekanisme tambahan
yang diuji hari ini yang mengalahkan baseline. Sinyal kuat: masalah "makin memburuk sejak deploy"
yang dikeluhkan user kemungkinan besar BUKAN krn kurang mekanisme pengaman, tapi krn parameter
Guardian produksi (momentum_floor_frac, near_tp_arm/lock) atau entry model itu sendiri -- **PR
terbuka, belum diinvestigasi, sesi terpisah** (sama seperti dicatat di Fase164/breakeven-lock).

**TIDAK ada tindakan holdout/produksi (OOF saja, model Guardian v2 tidak dilatih ulang).**

**Artefak.** `pipeline/experiments/dualbin_universe/fase164_sguardian_v2_monitor_correct.py`.

---

## 2026-08-09 — Fase 167 Tahap 1: parity config dualbin vs swint (`momentum_floor_frac`/`exit_threshold`/`near_tp`) -- TIDAK ADA yang lolos

**Pemicu.** User menegaskan dualbin seharusnya "sama dengan swint, bedanya cuma 2 model LGBM
LONG/SHORT terpisah". Dicek: dua repo **sengaja diisolasi total** (`live_dualbin_ft/CLAUDE.md`),
bukan fork kode — tapi konsep arsitekturnya memang paralel, dan dualbin lebih muda (beberapa
mekanisme swint belum pernah ada di dualbin). Rencana disetujui via Plan Mode (Opus) — uji tiap
perbedaan config satu per satu dulu (Tahap 1, murah) sebelum port mekanisme baru (Tahap 2+).

**Bug ketemu sebelum hasil valid.** Run pertama baseline TIDAK cocok Fase164 (-$2.895,78 alih-alih
-$648,87) -- ternyata `config.json` worktree riset lokal (gitignored) masih `breakeven_lock_mfe_pct
=0,005`, TIDAK PERNAH ter-update saat perbaikan 2026-08-09 dilakukan via SSH langsung ke VPS.
Pola SSOT-drift yang sama seperti `feedback-ssot-drift-pattern`. **Diperbaiki**: config lokal
disamakan (0,0), skrip dikeraskan supaya SEMUA lengan eksplisit set
`breakeven_lock_mfe_pct/buffer_pct=0,0` (tidak bergantung diam-diam ke isi file). Rerun: baseline
cocok persis (4.703 trade, PnL -$648,87, PF 0,88/0,91/0,97).

**Hasil (3 window OOF, gerbang PF>1 & PnL>0, mengalahkan baseline ≥2/3 window):**

| Lengan | Total PnL | vs baseline | W1 | W2 | W3 |
|---|---:|---:|---:|---:|---:|
| Baseline (live apa adanya) | -$648,87 | — | PF 0,881 | PF 0,911 | PF 0,966 |
| 1a floor 0,10→0,70 (nilai swint) | -$746,74 | **-$97,87** | lebih buruk | lebih buruk | lebih buruk |
| 1b exit_threshold 0,55→0,65 (swint) | -$632,24 | +$16,63 | lebih buruk | lebih baik (+$60,70) | ~flat |
| 1c near_tp dimatikan (swint tak punya) | -$620,53 | +$28,34 | ~flat | ~flat | ~flat (semua +$2-17) |
| 1d gabungan (mirip-swint) | -$745,38 | -$96,51 | lebih buruk | ~flat | lebih buruk |

**1a (nilai floor SWINT SENDIRI, 0,70) KALAH konsisten di ketiga window** -- bukan cuma total.
Floor yang lebih ketat (mengunci lebih banyak untung) yang bekerja baik di swint (TP/SL berbasis
ATR, 2,0/1,5) ternyata memperburuk di dualbin (TP/SL berbasis swing, `tp_sl_balance_ratio=1,0`,
TP jauh lebih jauh dari SL). Ini bukti awal bahwa **nilai parameter tidak bisa dipindah 1:1**
antar dua struktur TP/SL yang berbeda, walau nama parameternya sama persis.

1b & 1d ikut terseret jelek oleh komponen floor=0,70 di dalamnya. 1c (near_tp dimatikan total)
efeknya nyaris nol di ketiga window (+$2 s.d. +$17) -- jauh di bawah lantai derau OOF dualbin
(PnL ±8,1%, lihat `feedback-ukur-lantai-derau-sebelum-menafsir`), artinya near_tp_arm/lock yang
sudah aktif di live (0,95/0,20) itu sendiri hampir tidak berpengaruh, bukan menang atau kalah.

**Keputusan gerbang.** Tahap 1 TIDAK ADA yang lolos (0/4 lengan mengalahkan baseline ≥2/3 window
dengan margin di atas derau). Baseline (live apa adanya) tetap terbaik dari semua yang diuji.
Menegaskan lagi kesimpulan Fase164/166: bukan kurang mekanisme, PR akar masalah W1 (PF<1 di
SEMUA konfigurasi yang dicoba) masih terbuka.

**TIDAK ada tindakan holdout/produksi (OOF saja, config.json LOKAL riset yang dibetulkan, VPS
tidak disentuh).**

**Artefak.** `pipeline/experiments/dualbin_universe/fase167_parity_swint_config.py`;
`models/runs/dualbin_fase167_parity_swint_config.json`.

## 2026-08-11 — swint (ic32): scorecard OOF/OOS diperbaiki match exit mechanic live + funding cost ditambahkan

**Pemicu.** User minta cek ulang scorecard OOF/OOS swint (ic32_regime_v6.4) krn insiden DualBin
(evaluasi riset ternyata tidak menggambarkan live). Audit kode (bukan cuma baca dokumentasi)
menemukan 2 mekanisme exit yang sudah LIVE sejak 12-13 Juli TAPI tidak pernah dioper ke
`simulate_trades_swing()` di `run_oof_full_stack_sweep.py` / `model/eval/holdout_oos.py`:
1. Guardian floor = FIXED 0,7xTP_pnl (STOP-LIMIT exchange-side begitu TP tersentuh) -- simulasi
   diam2 masih pakai default lama trailing 0,7xMFE (`guardian_momentum_floor_frac`, param baru
   `guardian_floor_replace_with_tp`/`guardian_momentum_floor_tp_frac` sudah ada di
   `core/evaluator.py` sejak 12 Juli tapi tak pernah dioper scorecard resmi).
2. Cooldown profit-only 1 jam (live sejak 13 Juli) -- `config.py.TP_SL_COOLDOWN_ENABLED=False`
   default, tak pernah di-override di skrip scorecard resmi.

Juga ditemukan: **funding-rate cost TIDAK PERNAH dihitung** di simulasi manapun (fee_per_side +
slippage saja) -- funding cuma dipakai sbg fitur ML (`core/features.py`), bukan biaya. Di 15x
leverage, hold sampai `MAX_HOLDING_BARS`=36 jam bisa lewat 4 settlement (00:00/08:00/16:00 UTC).

**Metode.** Tambah flag `--live-parity-exit` opt-in ke `model/eval/holdout_oos.py` (non-breaking,
default off = perilaku lama persis). Skrip baru `pipeline/model/run_oof_live_parity_check.py`
(4 varian A/B/C/D) utk OOF. Tool baru `tools/model/funding_cost_overlay.py` (post-hoc, pakai data
funding-rate ASLI `data/training/funding_rate/` & `data/holdout-test/raw/funding_rate/`, bukan
re-run simulasi) -- konvensi Binance: funding positif = LONG bayar SHORT, notional=modal*leverage.

**Sanity check OOS**: baseline reproduksi (exit lama, tanpa fix) = 167 trade, PF 1,611 -- **cocok
persis** dgn angka yg sebelumnya tercatat di `inference_config.json`. Memvalidasi mesin simulasi
konsisten.

**Hasil (stack fs37_18coin_polos = live sekarang, exit PERSIS live + funding cost, MAE-aware @15x):**

| | Trades | WR | PF | PnL | MaxDD | LongPF | ShortPF |
|---|---|---|---|---|---|---|---|
| OOF lama (trailing floor, no cooldown/funding) | 3.339* | 63,6% | 1,869 | $4.410,21 | -$63,57 | 1,728 | 1,985 |
| **OOF baru (live-parity + funding)** | **2.837** | **65,7%** | **2,020** | **$4.279,35** | **-$65,56** | **1,869** | **2,167** |
| OOS lama (trailing floor, no cooldown/funding) | 167 | 64,7% | 1,611 | $94,41 | -$26,89 | 0,530 | 1,967 |
| **OOS baru (live-parity + funding)** | **163** | **64,4%** | **1,643** | **$98,58** | **-$25,15** | **0,535** | **2,012** |

**Kesimpulan: BUKAN insiden spt DualBin.** Exit mechanic yang PERSIS live menghasilkan PF SEDIKIT
LEBIH BAIK di kedua basis (OOF +0,151/+8%, OOS +0,032/+2%), bukan lebih buruk -- gap yg ditemukan
tidak menyembunyikan performa yg lebih jelek. Dampak funding rate **negligible** (<0,2% dari PnL
di kedua basis) -- ditambahkan demi kelengkapan akuntansi, bukan krn mengubah kesimpulan. Fee
sendiri (0,04%/sisi) sudah fee-on-notional yg benar (scale dgn leverage), sudah tervalidasi lewat
audit kode terpisah.

**Temuan independen yg TIDAK hilang oleh perbaikan ini**: sisi LONG OOS tetap lemah (PF 0,530 ->
0,535, nyaris tak berubah) -- konfirmasi ini masalah entry model/kondisi pasar, BUKAN artefak
simulasi exit.

**Belum terselesaikan**: trade count OOF baru (2.837) TIDAK cocok dgn angka OOF lama yg terpasang
sebelumnya di `inference_config.json` (3.339) -- source 3.339 tidak berhasil direkonsiliasi ke
skrip manapun (skrip baru + `run_oof_full_stack_sweep.py` konsisten hasilkan ~2.837-2.860, cocok
dgn `model_registry.json.oof_scorecard` versi lama 2.858). Kemungkinan 3.339 dihitung dari
kombinasi param/populasi berbeda yg tidak diketahui persis -- **perlu investigasi terpisah**
sebelum dipercaya penuh. OOS TIDAK punya masalah ini (baseline reproduksi cocok persis).

**SSOT diupdate**: `model_registry.json.oof_scorecard` + `inference_config.json.scorecard.oof`/
`scorecard.holdout_oos` (dashboard live) -- angka lama diarsip di key `*_pre_live_parity_fix_2026_08_11`.
Model/HMM/Guardian/threshold **TIDAK berubah** -- murni perbaikan metodologi pengukuran, bukan
deploy baru, tidak perlu approval deploy terpisah.

**Artefak.** `pipeline/model/run_oof_live_parity_check.py` (baru); `model/eval/holdout_oos.py`
(+`--live-parity-exit`); `tools/model/funding_cost_overlay.py` (baru); trade-level CSV:
`data/live_cache/oof_live_parity_D_trades_funding.csv`, `data/live_cache/oos_live_parity_D_trades_funding.csv`;
`models/runs/guard_opt2_plus_trend_hmm_18coin_clean/{oof_live_parity_check.json,oos_holdout_full_scorecard.json}`.

## 2026-08-12 — swint (ic32): HMM base threshold 0,70 -> 0,65 (delta tetap 0,10) -- DEPLOYED

**Pemicu.** User tanya threshold dasar sebelum HMM (jawab: 0,70, dari `hmm.per_state_thresholds["-1"]`
live). Lanjut tanya "bagaimana jika diturunkan tanpa regime itu jadi 0,65" -- ditemukan state "-1"
itu sendiri INERT di backtest riset (tidak pernah direproduksi, cuma dipakai live sbg fallback), jadi
pertanyaan direframe ke yang bermakna: turunkan BASE utk SEMUA 4 state regime (bukan cuma fallback).
User setuju uji base=0,65, lalu minta tambah pembanding base=0,60.

**Metode.** Reuse `--hmm-base`/`--hmm-delta` override (sudah ada dari kerja sebelumnya) di
`pipeline/model/run_oof_live_parity_check.py` (varian D = live-parity: floor FIXED 0,7xTP + cooldown)
dan `model/eval/holdout_oos.py --live-parity-exit`. Stack skrg = `fs37_18coin_polos` (`guard28f_18coin_clean`,
window 18/36, MAE-aware @15x, funding cost). Delta tetap 0,10 (tidak diubah) -- hanya base yang digeser.
3 titik dibandingkan: base 0,60 / 0,65 / 0,70 (baseline live sblm perubahan ini).

**Hasil OOF (varian D, live-parity penuh, basis native/leverage-invariant, apples-to-apples):**

| base | Trades | WR | PF | PnL | MaxDD | LongPF | ShortPF |
|---|---|---|---|---|---|---|---|
| 0,70 (lama) | 2.837 | 67,2% | 2,609 | $5.576,30 | -$45,40 | 2,834 | 2,426 |
| **0,65 (baru)** | **5.365** | **63,6%** | **2,066** | **$8.122,78** | **-$78,73** | **2,155** | **1,983** |
| 0,60 (ditolak) | 9.289 | 61,6% | 1,841 | $11.843,32 | -$109,44 | 1,917 | 1,766 |

**Hasil OOS (holdout, live-parity penuh, MAE-aware @15x, funding -- angka final yg disimpan SSOT):**

| base | Trades | WR | PF | PnL | MaxDD | LongPF | ShortPF |
|---|---|---|---|---|---|---|---|
| 0,70 (lama) | 163 | 64,4% | 1,643 | $98,58 | -$25,15 | 0,535 | 2,012 |
| **0,65 (baru)** | **344** | **59,6%** | **1,367** | **$139,54** | **-$33,21** | **0,851** | **1,585** |
| 0,60 (ditolak) | -- | -- | -- | $81,96 | -$62,80 | -- | -- |

Pola di OOF vs OOS terbalik: makin base diturunkan, OOF (data latih) makin "membaik" (PF naik, trade
makin banyak) TAPI OOS (data belum pernah dilihat) base=0,60 malah PALING JELEK (PnL & MaxDD) --
ciri khas overfitting, bukan sinyal asli. Ini alasan utama 0,60 ditolak walau OOF-nya paling menarik.

**Keputusan: base=0,65 DIPILIH, base=0,60 DITOLAK.** Pola klasik overfitting pada 0,60: OOF makin
membaik seiring base diturunkan (lebih banyak trade, PF makin naik krn threshold makin longgar) TAPI
OOS-nya justru MEMBURUK dibanding 0,65 (PnL lebih rendah, MaxDD jauh lebih dalam) -- sinyal base
terlalu longgar mulai menangkap sinyal derau yang tidak generalize ke luar-sample. base=0,65 adalah
titik tengah yang trade-off-nya user terima sadar: trades naik 163->344 (+111%), WR turun 64,4%->59,6%,
PF turun 1,643->1,367 (-17%), PnL naik $98,58->$139,54 (+42%), MaxDD lebih dalam -$25,15->-$33,21 (+32%).
LongPF justru MEMBAIK 0,535->0,851 (masih <1, LONG tetap net rugi tapi jauh kurang parah) -- ShortPF
turun 2,012->1,585 (msh >1, net untung).

**Audit wajib pre-deploy (`tools/model/verify_hmm_feature_parity.py`, `audit_feature_value_parity.py
--run opt2_plus_trend_18coin_iso37f`):** 2 temuan, KEDUANYA pre-existing & tidak terkait perubahan ini
(dicek ulang, tidak berubah krn base HMM) -- **perlu investigasi terpisah, tidak memblokir deploy ini**:
1. ETHUSDT HMM classification mismatch (12/855 bar sejak 2026-06-24) -- terkait modifikasi lokal
   belum di-commit di `core/regime.py` (120 baris, sudah ada sblm sesi ini mulai).
2. Fitur `coin_mkt_sync_24h` nilainya abnormal flat di live vs training (std_ratio=0,0304) --
   mengindikasikan ADAUSDT spesifik.

**Dependensi dicek**: `regime_model_routing` & `regime_disable` sudah MATI (live "polos" sejak
2026-07-14) -- perubahan base HMM tidak butuh re-validasi jalur itu.

**SSOT diupdate & di-deploy.** `model_registry.json`: `stack.hmm.base` 0,70->0,65 +
`oof_scorecard` (base=0,65 native: trades=5.365, WR=63,6%, PF=2,066, PnL=$8.122,78, MaxDD=-$78,73,
LongPF=2,155, ShortPF=1,983); lama diarsip `oof_scorecard_pre_base065_2026_08_12`.
`inference_config.json`: `hmm.per_state_thresholds` -> `{"0":[0.75,0.55],"1":[0.7,0.6],"2":[0.6,0.7],
"3":[0.55,0.75],"-1":[0.65,0.65]}`; `scorecard.oof`/`scorecard.holdout_oos` (angka MAE-aware+funding
di tabel atas) + `monthly` baru; lama diarsip `*_pre_base065_2026_08_12`; `_snapshot_time` ->
"2026-08-12 01:06:35". Deploy ke VPS via merge surgical (pull live config dulu, replace HANYA key
di atas, assert block `models`/`regime_model_routing`/`regime_disable`/`spot_confirm`/`limit_exit`/
`cooldown`/`risk` live-only tetap utuh) -- SCP + `systemctl restart swint-trade` (bukan `update.sh`,
krn file ini drift independen dari git di VPS). Diverifikasi post-deploy: `/api/health` sehat
(`scheduler_running: true`), dashboard `/models` render threshold & catatan perubahan dgn benar.

**Rollback**: base=0,70, delta=0,10 (angka lama tersimpan di key arsip `*_pre_base065_2026_08_12`
di kedua file SSOT).

## 2026-08-12 — swint (ic32): tambah constraint eksekusi real (max_open_positions/daily_loss_limit) ke OOF+OOS -- MaxDD hampir 2x lipat

**Pemicu.** User minta cek apakah KONFIGURASI OOF/OOS realistis diterapkan di live -- persis
kekhawatiran insiden DualBin (PF riset 1,9 tidak terwujud di real), tapi fokus pada MEKANISME
eksekusi, BUKAN perbandingan angka live historis (itu diminta terpisah sebelumnya & sudah
dijawab). Audit kode eksekusi live (`swint_tradev2/app/services/execution.py`) menemukan 2
constraint NYATA yang menggerbang setiap entry (`place_entry()`) tapi TIDAK PERNAH dimodelkan
di simulasi manapun (`core/evaluator.py`, `model/eval/holdout_oos.py`,
`run_oof_live_parity_check.py` -- dicek nol referensi):
1. `max_open_positions=10` -- `Trade.query.filter_by(status="open", is_live=True).count()`,
   cap GLOBAL lintas semua koin/arah (bukan per-koin). Backtest diam-diam asumsi modal tanpa
   batas, bisa buka SEMUA sinyal 18 koin sekaligus.
2. `daily_loss_limit=8` -- `_get_daily_loss_count()`: setelah 8 trade rugi (`pnl_net<0`)
   closed hari ini (WITA, midnight-to-midnight), entry baru DITOLAK sisa hari itu. Backtest
   terus trading tanpa jeda apa pun.

(Temuan ke-3 dari audit yang sama, SL polling 5 menit tanpa exchange-side stop, TIDAK
dikuantifikasi -- perlu data intrabar terpisah, di luar scope malam ini.)

**Metode.** Fungsi baru `apply_portfolio_execution_limits()` (`core/evaluator.py`) -- filter
POST-HOC: kumpulkan semua kandidat trade lintas-koin (sudah disimulasikan independen per-koin,
punya `entry_time`/`exit_time`/`net_pnl`), urutkan kronologis, replay slot posisi terbuka via
min-heap by `exit_time` + hitung ulang rugi harian WITA persis query live
(`_get_daily_loss_count`). Kandidat yang ditolak TIDAK diganti kandidat lain (sama seperti
live: sinyal hilang, bukan dicoba ulang). Smoke-test unit (2 skenario sintetis: cap posisi &
cap rugi harian) lulus sebelum run penuh. Wired via flag baru `--portfolio-limits` di
`model/eval/holdout_oos.py` (OOS) & `pipeline/model/run_oof_live_parity_check.py` (OOF, varian
baru **E** = D + portfolio limits, D & varian A/B/C tetap ada sbg pembanding). Default values
dari `config.LIVE_MAX_OPEN_POSITIONS=10` / `LIVE_DAILY_LOSS_LIMIT=8` (sudah ada di config.py
sblm sesi ini, ternyata belum pernah dipakai di mana pun).

**Hasil (stack live skrg, base HMM 0,65, exit live-parity + funding, MAE-aware @15x):**

| | Trades | WR | PF | PnL | MaxDD | LongPF | ShortPF |
|---|---|---|---|---|---|---|---|
| OOF tanpa portfolio_limits | 5.365 | 62,4% | 1,712 | $6.253,27 | -$84,71 | 1,600 | 1,831 |
| **OOF + portfolio_limits (real)** | **5.343** | **62,5%** | **1,721** | **$6.267,20** | **-$159,68** | **1,618** | **1,828** |
| OOS tanpa portfolio_limits | 344 | 59,6% | 1,367 | $139,54 | -$33,21 | 0,851 | 1,585 |
| **OOS + portfolio_limits (real)** | **333** | **59,5%** | **1,343** | **$129,94** | **-$62,17** | **0,837** | **1,555** |

Cuma 8-22 kandidat trade ditolak (hampir semua krn `max_open_positions`, `daily_loss_limit`
nyaris tak pernah kena -- 0/8 di OOS, 14/22 di OOF krn rentang 6 tahun jauh lebih panjang).
Trade count/WR/PF/PnL nyaris tak berubah (<1% di OOS, <2% di OOF). **TAPI MaxDD naik ~88% di
KEDUA dataset SECARA INDEPENDEN** -- OOF -84,71->-159,68, OOS -33,21->-62,17. Pola identik di
6 tahun data latih maupun 4 bulan holdout yang belum pernah dilihat model -- bukan derau,
struktural.

**Interpretasi.** Membatasi posisi konkuren menghapus efek diversifikasi persis di saat paling
dibutuhkan: periode sinyal padat/berkorelasi (biasanya pergerakan pasar serentak lintas koin) --
kalau ada rentetan rugi di periode itu, dampaknya jadi lebih terkonsentrasi drpd yg diasumsikan
backtest tanpa batas (yg bisa "menyebar" ke sebanyak-banyaknya koin). Efeknya nyaris seluruhnya
di MaxDD (risiko), BUKAN return -- return (PF/WR/PnL) nyaris tidak terpengaruh krn sinyal yang
ditolak jumlahnya kecil & tidak sistematis condong untung/rugi.

**Bukan insiden spt DualBin** (PF tidak jatuh, model tidak "gagal diwujudkan") -- tapi
**scorecard yang SEBELUMNYA jadi SSOT meremehkan risiko riil (MaxDD) ~2x lipat**. Signifikan
utk keputusan ukuran modal/leverage, krn MaxDD adalah metrik yg dipakai menilai toleransi risiko.

**Keputusan user**: adopsi angka portfolio_limits sbg SSOT baru ("environment riset harus
masuk akal dan representatif live").

**SSOT diupdate.** `model_registry.json.oof_scorecard` (native+funding, portfolio_limits ON) --
lama diarsip `oof_scorecard_pre_portfolio_limits_2026_08_12`. `inference_config.json.scorecard.
oof`/`scorecard.holdout_oos` (MAE-aware+funding, portfolio_limits ON) + `monthly` baru -- lama
diarsip `*_pre_portfolio_limits_2026_08_12`. `_snapshot_time`/threshold/model **TIDAK berubah**
-- murni perbaikan metodologi pengukuran (constraint eksekusi, bukan parameter model), sama
kelasnya dgn perbaikan live-parity-exit 2026-08-11.

**Artefak.** `core/evaluator.py` (+`apply_portfolio_execution_limits`); `model/eval/
holdout_oos.py` (+`--portfolio-limits`/`--max-open-positions`/`--daily-loss-limit`);
`pipeline/model/run_oof_live_parity_check.py` (+varian E, flag sama); trade-level CSV:
`data/live_cache/oof_hmmbase065_E_trades.csv` (+`_funding.csv`),
`data/live_cache/oos_hmmbase065_portfoliolimits_trades_funding.csv`;
`models/runs/guard_opt2_plus_trend_hmm_18coin_clean/{oof_live_parity_check_hmmbase0.65.json,
oos_holdout_full_scorecard.json,oos_holdout_h4closed_full.json}`.

**Belum diselesaikan** (di luar scope malam ini, dicatat sbg utang): kuantifikasi slippage SL
5-menit-polling (temuan ke-3 audit realisme eksekusi) -- butuh data intrabar M1/M5 utk estimasi
realistis, belum ada di repo ini.

## 2026-08-12 — REJECT: pyramiding max2 time-gap untuk LGBM37f + HMM 0,65 + Guardian clean (polos)

**Tujuan.** Menaikkan utilisasi modal tanpa melonggarkan threshold entry. Uji scale-in leg ke-2
arah sama dengan jeda minimum 1/2/4/8 jam, maksimum dua leg per koin.

**Metode.** OOF genuine 2020-01-01 s.d. 2026-04-01 + pseudo-holdout internal
2025-10-01 s.d. 2026-04-01. Evaluator diselaraskan dengan Guardian
`guard_opt2_plus_trend_hmm_18coin_clean`, floor fixed 0,7xTP, cooldown profit-only 1 jam,
dan constraint live global max 10 posisi/daily-loss 8. Tidak menjalankan atau men-tune holdout
tersegel. Artefak: `pipeline/model/run_oof_pyramiding_time_gap_sweep.py` dan
`models/runs/guard_opt2_plus_trend_hmm_18coin_clean/pyramiding_time_gap_sweep_oof_live_parity.json`.

| Varian | OOF trade | PF | PnL | MaxDD | Delta MaxDD vs baseline |
|---|---:|---:|---:|---:|---:|
| Baseline single-leg | 5.258 | 2,129 | $8.190,84 | -$138,22 | — |
| max2, gap 1j | 5.094 | 2,022 | $10.139,54 | -$228,46 | +65,3% |
| max2, gap 2j | 5.092 | 2,034 | $9.729,08 | -$213,92 | +54,8% |
| max2, gap 4j | 5.087 | 1,983 | $8.797,95 | -$191,10 | +38,3% |
| max2, gap 8j | 5.091 | 1,949 | $7.913,67 | -$159,63 | +15,5% |

**Keputusan.** REJECT/CLOSE opsi pyramiding time-gap. Semua varian menurunkan jumlah trade
selesai ~3% (scale-in menggabungkan exposure, bukan menciptakan entry independen), tidak mencapai
target utilisasi, dan tidak memenuhi kriteria PF (maksimum -5%) + MaxDD (maksimum +15%). Gap 1j
memang kuat pada pseudo-holdout (PF 2,392 vs 2,176; PnL +43,5%; MaxDD membaik 6,8%), tetapi gagal
robustness 6 tahun OOF sehingga tidak boleh dipromosikan. Angka ini hanya perbandingan internal
karena overlay funding dan MAE-aware belum diterapkan ke varian pyramiding; kegagalan sudah jelas
sebelum overlay tersebut, jadi tidak ada alasan melakukan simulasi tambahan/deploy.

## 2026-08-13 — OOF candidate: second-tier near-miss sleeve 25% modal (belum validasi OOS/deploy)

**Tujuan.** Tambah frekuensi tanpa mengubah atau melonggarkan core `LGBM37f + HMM causal
0,65/0,10 + Guardian clean (polos)`. Sleeve hanya mengambil sinyal yang gagal ambang HMM core
sebesar paling banyak band yang diuji; core tetap prioritas dan tidak mendapat perubahan threshold.

**Metode.** Genuine OOF 2020-01-01 s.d. 2026-04-01, 18 koin. Band near-miss 0,02/0,03/0,05,
modal sleeve 25% dari core. Simulator menggunakan fixed floor 0,7xTP, cooldown profit-only 1 jam,
Guardian `guard_opt2_plus_trend_hmm_18coin_clean`, serta max 10 posisi dan daily-loss 8 persis
live. Posisi sleeve tetap mengambil satu slot global; simulator tidak mengizinkan posisi kedua
pada koin yang sama. **Tidak memakai holdout tersegel untuk memilih band.**

| Band | Sleeve trade | PF sleeve | Trade gabungan | Delta trade | PF gabungan | MaxDD gabungan |
|---|---:|---:|---:|---:|---:|---:|
| Core saja | — | — | 5.343 | — | 2,056 | -$133,76 |
| 0,02 | 1.928 | 1,774 | 6.696 | +25,3% | 2,033 | -$117,16 |
| 0,03 | 2.934 | 1,678 | 7.455 | +39,5% | 1,999 | -$113,93 |
| 0,05 | 5.059 | 1,636 | 9.091 | +70,2% | 1,963 | -$106,59 |

**Interpretasi awal.** Semua band memenuhi gate OOF awal: sleeve PF >1,10, trade gabungan naik
>=25%, PF gabungan turun <=5%, dan MaxDD tidak memburuk. Band **0,02** kandidat konservatif:
frekuensi +25,3% (dari 2,74 menjadi ~3,44 trade/hari kalender) dengan PF hanya -1,1%. Band 0,05
memaksimalkan volume tetapi PF turun -4,5%; tidak dipilih sebelum validasi lanjut. PnL gabungan
lebih rendah daripada core dalam ketiga band karena sleeve kadang menduduki slot/koin sebelum
sinyal core berikutnya; jadi alasan kandidat adalah utilisasi dan PF/DD, bukan kenaikan PnL mentah.

**Batas bukti / keputusan.** Kandidat OOF saja, **BELUM** perubahan config/deploy dan belum layak
diaktifkan. Funding overlay dan MAE-aware belum diterapkan pada sleeve; berikutnya hanya setelah
keputusan user: validasi kandidat 0,02 pada OOS yang sudah tersedia sebagai evaluasi final,
kemudian MAE-aware+funding dan audit konsentrasi per-koin/arah sebelum ada usulan paper/live.

**Artefak.** `pipeline/model/run_oof_second_tier_sleeve.py`;
`models/runs/guard_opt2_plus_trend_hmm_18coin_clean/second_tier_sleeve_oof_live_parity.json`.

## 2026-08-13 — KANDIDAT: matikan Guardian momentum floor total (floor OFF) — lolos 2 jendela, MENUNGGU REVIEW OPUS/FABLE

**Status: BELUM DIADOPSI, BELUM DIDEPLOY.** User setuju arah adopsi, tapi memilih review
Opus/Fable dulu sesuai aturan "perubahan production wajib direncanakan model perencana".
Entry ini = paket bukti untuk review itu.

**Pemicu.** Insiden nyata FILUSDT 2026-08-12: Guardian exit 20:10 UTC @0,6880 (paper, SHORT),
lalu candle 21:15 UTC anjlok ke low 0,6569 — untung ~6pp lebih besar terlewat. User tanya
seberapa sering pola ini terjadi.

**Langkah 1 — kuantifikasi "big-move terlewat"** (`tools/model/_scratch_guardian_missed_move.py`,
6 tahun OOF, 3.424 trade Guardian-early-exit, threshold big-move 3%):

| Jendela | Big-move TERLEWAT | Big-move TERHINDAR (floor menyelamatkan) |
|---|---|---|
| +6h | 17,8% | 17,3% |
| **+12h** | **28,2%** | **30,0%** |
| +24h | 39,6% | 44,3% |
| +48h | 51,8% | 58,1% |

Insiden FILUSDT BUKAN langka (~1 dari 4 kejadian), tapi hampir seimbang dgn kebalikannya —
konsisten dgn Guardian sbg peredam variance, bukan penebak arah. Breakdown per-outcome:
`GUARDIAN_MOMENTUM_FLOOR` (mekanisme yg kena di FILUSDT) rate terlewat **36,6%**, jelas lebih
tinggi dari `GUARDIAN_EXIT` biasa (26,4%) — floor inilah tersangka utama, bukan Guardian umum.

**Langkah 2 — sweep floor_frac** (`pipeline/model/run_oof_floor_frac_sweep.py`, baru):

| floor_frac | OOF PF | OOF PnL | OOS PF | OOS PnL | OOS MaxDD |
|---|---|---|---|---|---|
| 0,50 | 2,090 | $8.287,58 | 1,300 | $114,09 | -$62,48 |
| 0,60 | 2,064 | $8.076,66 | 1,338 | $127,63 | -$61,73 |
| 0,70 (live) | 2,056 | $8.014,24 | 1,350 | $132,10 | -$60,12 |
| 0,80 | 2,026 | $7.812,37 | — | — | — |
| 0,90 | 2,011 | $7.696,52 | — | — | — |

**Melonggarkan floor (0,5/0,6) DITOLAK** — pola overfitting klasik yg sama dgn base HMM 0,60:
menang OOF, KALAH OOS di semua metrik. Rate terlewat/terhindar nyaris TIDAK berubah lintas
floor_frac (27,6-28,7% vs 29,7-30,0%) — hipotesis awal "floor longgar kurangi kasus FILUSDT"
TIDAK TERBUKTI; pergerakan pasca-exit ditentukan pasar, bukan posisi floor.

**Langkah 3 — floor OFF total** (beda dari floor longgar: `guardian_floor_replace_with_tp=False`
+ `guardian_momentum_floor_frac=0.0`, jadi TIDAK ADA jaring pengaman pasca-TP sama sekali;
exit murni dari sinyal probabilitas Guardian / TIMEOUT). Angka final MAE-aware @15x + funding
+ portfolio_limits (metodologi PERSIS sama dgn SSOT saat ini):

| | OOF Trades | OOF PF | OOF PnL | OOF MaxDD | OOS Trades | OOS PF | OOS PnL | OOS MaxDD |
|---|---|---|---|---|---|---|---|---|
| **Floor OFF** | 5.277 | **1,773** | **$6.849,97** | -$159,98 | 329 | **1,390** | **$150,53** | **-$51,27** |
| 0,70 (live) | 5.343 | 1,721 | $6.253,27 | -$159,68 | 333 | 1,343 | $139,54 | -$62,17 |

**Menang di 2 jendela sekaligus, di SEMUA metrik utama** — OOF PF +3,0% PnL +9,3% MaxDD netral
($0,30 beda, derau); OOS PF +3,5% PnL +15,9% **MaxDD 17,5% LEBIH DANGKAL**. Ini satu-satunya
kandidat malam ini yg lolos gerbang 2-jendela (base 0,60 gagal, floor 0,5/0,6 gagal).

**Cek risiko ekor** (kekhawatiran: tanpa floor, posisi "lari" bisa rugi besar): 10 kerugian
terbesar OOS varian floor-OFF SEMUANYA `LOSS` (SL biasa), bukan Guardian-exit-lalu-berbalik.
Terburuk -$13,36 — wajar utk SL hit @15x, tidak ada pola bahaya baru.

**Kenapa belum diadopsi meski angkanya lolos:** ini perubahan ARSITEKTUR eksekusi (hapus total
jaring pengaman mekanis, sepenuhnya percaya model probabilitas Guardian), bukan geser parameter.
Sampel OOS 329 trade / 4 bulan — lantai derau utk sistem ini belum diukur spesifik (bandingkan
disiplin dualbin: `feedback-ukur-lantai-derau-sebelum-menafsir`). Butuh keputusan model
perencana, bukan cuma "angka menang".

**Pertanyaan terbuka utk review Opus/Fable:**
1. Apakah selisih PF +0,047 OOS (1,343->1,390) di atas lantai derau utk n=329? Belum diukur.
2. Live saat ini floor = STOP-LIMIT exchange-side ASLI (resting order, bukan polling). Matikan
   floor = hapus satu-satunya order pelindung sisi-bursa yg ada — sisa exit semuanya polling
   5 menit (lihat catatan `check_positions.py`: TIDAK ADA SL exchange-side). Trade-off
   operasional ini TIDAK terwakili di backtest manapun.
3. Perlu window OOS ketiga (mis. potong holdout jadi 2 sub-periode) sebelum promosi?

**Artefak.** `pipeline/model/run_oof_floor_frac_sweep.py` (baru, dukung `off`);
`tools/model/_scratch_guardian_missed_move.py` (baru); `model/eval/holdout_oos.py`
(+`--floor-frac`, dukung `off`); `data/live_cache/{oof,oos}_floorOFF_final.json`,
`data/live_cache/oof_floorOFF_trades.csv`, `data/live_cache/guardian_missed_move_analysis.csv`,
`data/live_cache/floor_frac_sweep_result.csv`.

### ADDENDUM review Opus 2026-08-13 — KOREKSI: klaim "lolos gerbang 2-jendela" TIDAK VALID

Menjawab 3 pertanyaan terbuka di atas. **Klaim utama entry ini dikoreksi**, sesuai disiplin
`feedback-correctness-over-favorable-numbers` (perbaiki catatan walau melemahkan kandidat yang
user inginkan).

**Pertanyaan #1 (lantai derau) — TERJAWAB, dan menggugurkan klaim OOS.**
`tools/model/_scratch_floor_noise_floor.py` (bootstrap 20.000x, seed 42):

| Uji | Hasil |
|---|---|
| **Lantai derau PF OOS (n=329)** | **+-0,537** |
| Selisih PF OOS teramati (OFF - 0,7) | +0,047 -- **11x DI BAWAH lantai derau** |
| P(OFF > 0,7) dari bootstrap | 0,563 (nyaris lempar koin) |
| Berpasangan OOS (n=328, cocok per coin+entry_time) | +$0,043/trade, CI95 [-$0,040, +$0,130], **p=0,31 TIDAK SIGNIFIKAN** |
| Trade yang exit-nya benar-benar berubah di OOS | **cuma 37** -- 20 lebih baik vs 17 lebih buruk |

Seluruh rentang PF OOS lintas SEMUA varian floor (1,300 s.d. 1,397 = rentang 0,097) **5,5x lebih
sempit dari lantai derau**. Artinya peringkat OOS mana pun (termasuk "OFF menang") adalah DERAU,
bukan temuan. "Menang di 2 jendela" **tidak terbukti** -- yang benar: menang di 1 jendela (OOF),
dan jendela kedua (OOS) TIDAK PUNYA DAYA untuk mengonfirmasi maupun membantah.

**TAPI efeknya kemungkinan besar NYATA -- OOF berpasangan sangat kuat** (n=5.263 cocok, 6 tahun):
+$0,128/trade, CI95 [+$0,089, +$0,169], **p<0,0001**. Dari 541 trade yang exit-nya berubah:
**350 lebih baik vs 191 lebih buruk** (65:35, arah konsisten). Total +$674,77. Lolos koreksi
multi-perbandingan dgn mudah. CI OOS [-0,040, +0,130] JUSTRU MEMUAT titik-estimasi OOF (+0,128) --
OOS tidak membantah, cuma terlalu kecil (37 trade terdampak) untuk meresolusi efek sebesar itu.

**Pertanyaan #2 (order pelindung sisi-bursa) — lebih bernuansa dari dugaan awal.**
Baca `paper_trading.py:578-601` + `check_positions.py`: floor STOP-LIMIT (`_place_floor_stop`)
cuma dipasang SETELAH TP tersentuh. Untuk SELURUH masa hidup pra-TP setiap trade, **sudah TIDAK
ADA proteksi sisi-bursa sama sekali** (terdokumentasi, risiko yang sudah diterima user sadar --
memory `project-no-exchange-side-stop-loss`). Jadi mematikan floor BUKAN "menghapus satu-satunya
jaring pengaman untuk semua trade", melainkan menghapusnya untuk subset pasca-TP saja (posisi
yang sudah UNTUNG). Marginal, tapi tetap penurunan: `floor_tp_frac=0` melewati SELURUH blok itu,
jadi tidak ada resting order MAUPUN polling floor. Bukti langsung risiko ini nyata: server
**macet total malam ini** (butuh restart manual, ~1 jam+).

**Pertanyaan #3 (lubang sweep) — DITEMUKAN & DIISI.** Rentang 0,1-0,3 tak pernah diuji:

| floor_frac | OOF PF | OOF PnL | OOS PF | OOS PnL | OOS MaxDD | Resting order di bursa? |
|---|---|---|---|---|---|---|
| OFF | 2,118 | $8.638,67 | 1,397 | $152,66 | -$49,20 | **TIDAK** |
| 0,10 | 2,105 | $8.483,26 | 1,340 | $131,35 | -$54,64 | YA |
| 0,20 | 2,107 | $8.455,59 | 1,336 | $129,59 | -$54,64 | YA |
| 0,30 | 2,106 | $8.428,94 | 1,330 | $126,28 | -$55,92 | YA |
| 0,50 | 2,090 | $8.287,58 | 1,300 | $114,09 | -$62,48 | YA |
| 0,70 (live) | 2,056 | $8.014,24 | 1,350 | $132,10 | -$60,12 | YA |

floor 0,1-0,3 menangkap **~82% keuntungan PF OOF** dari OFF (2,056->2,107 vs 2,056->2,118) sambil
TETAP memasang STOP-LIMIT di bursa. Di OOS ketiganya sedikit di bawah 0,70 -- tapi seperti di
atas, SEMUA selisih OOS ada di dalam derau, jadi itu bukan bukti apa pun.

**PUTUSAN REVIEW: JANGAN adopsi sekarang** (bukan karena buktinya jelek):
1. Bukti OOF kuat & kemungkinan nyata, TAPI satu-satunya jendela luar-sampel yang ada tidak bisa
   mengonfirmasi. Aturan `feedback-wajib-dua-jendela-penilaian`: signifikan di 1 jendela =
   KANDIDAT, bukan temuan. Status kandidat TIDAK naik hanya karena arah OOS kebetulan searah.
2. Nilai efek ~+$112/thn di sizing sekarang (~8% PnL). Bukan nol, tapi kecil dibanding
   **kesenjangan yang belum terselesaikan: backtest PF ~1,4-1,6 vs rekam jejak uang riil PF 0,68**
   (162 trade, 18 koin, periode sama). Menyetel exit demi +8% sementara diskrepansi ~2x di sistem
   yang sama belum dibongkar = menyetel hal yang salah lebih dulu.
3. Trading uang riil sedang MATI. Tidak ada urgensi mengubah mekanisme risiko sekarang.

**Kalau nanti tetap mau maju**, urutan yang disarankan: (a) bongkar dulu kesenjangan backtest vs
uang-riil (`tools/ops/compare_oos_live_signals.py`); (b) kalau tetap mau longgar, **pilih
floor_frac 0,2, bukan OFF** -- ~82% keuntungan OOF, resting order di bursa TETAP ADA, dan MaxDD
OOS lebih dangkal dari 0,70 (-$54,64 vs -$60,12); (c) jendela OOS ketiga / paper-forward sebelum
uang riil.

**Artefak review.** `tools/model/_scratch_floor_noise_floor.py`;
`data/live_cache/oos_floorfrac{01,02,03,05,06,07}_trades_detail.csv`,
`data/live_cache/oos_floorOFF_trades_detail.csv`, `data/live_cache/oof_floorfrac07_trades.csv`.

## 2026-08-13 — BIAS BARU DITEMUKAN: floor STOP-LIMIT dimodelkan SALAH di backtest (scorecard live overstated ~7% OOF)

**Pemicu.** User bertanya: "memang pakai stop limit, bagaimana scorecard jika stop limit itu
diaktifkan?" -- pertanyaan tepat sasaran. Ternyata backtest TIDAK PERNAH memodelkan floor
sebagaimana ia benar-benar tereksekusi di live.

**Ketidakcocokan (dibaca dari kode kedua sisi, bukan asumsi):**

| | Live (`paper_trading.py::_place_floor_stop`) | Backtest LAMA (`core/evaluator.py`) |
|---|---|---|
| Bentuk | STOP-LIMIT resting di bursa, `trigger=limit=floor_price` | pengecekan in-loop |
| Kapan trigger | **INTRABAR**, saat harga menyentuh level | hanya saat **CLOSE** bar |
| Harga exit | **DI `floor_price`** | **DI `close[j]`** |
| vs SL di bar sama | floor pasti duluan (floor di sisi untung, SL di sisi rugi) | SL menang (dicek lebih dulu) |

`raw_exit = close[j]` itu menurut definisi SUDAH DI BAWAH floor (syarat trigger-nya `cur_pnl <
floor_pnl`). Bandingkan **SL yang SUDAH benar** di file yang sama: `raw_exit = sl_price` (harga
level, bukan close). Jadi dua stop order diperlakukan beda di satu simulator yang sama.

**Perbaikan.** Param baru `guardian_floor_intrabar` (default False = perilaku lama, non-breaking,
scorecard lama tetap bisa direproduksi -- diverifikasi regresi OOF PF 2,056/$8.014,24 & OOS PF
1,350/$132,10 cocok PERSIS). True = model seperti live: trigger saat wick menyentuh `floor_price`,
exit DI `floor_price`, dan dicek SEBELUM SL.

**Hasil -- ARAH BERLAWANAN dari hipotesis awal.** Dugaan saya: model benar akan MENGUNTUNGKAN
floor (fill di level, bukan di close yang lebih jelek). **SALAH.** Efek WAKTU jauh mengalahkan
efek HARGA: order resting kena "sabet wick" -- ter-trigger oleh celupan sesaat yang sebenarnya
pulih di bar yang sama, memotong pemenang lebih dini. Itu memang perilaku asli stop order.

OOF native (base 0,65, portfolio_limits ON):

| Varian | PF | PnL | MaxDD |
|---|---|---|---|
| floor 0,7 model LAMA (= dasar SSOT sekarang) | 2,056 | $8.014,24 | -$133,76 |
| floor 0,7 model LIVE BENAR | **2,001** | **$7.623,97** | -$128,04 |
| floor 0,2 model LIVE BENAR | 2,009 | $7.683,14 | -$132,09 |
| floor OFF (tak terpengaruh) | 2,118 | $8.638,67 | -$134,07 |

**Konsekuensi 1 -- SCORECARD LIVE SEKARANG OVERSTATED.** Angka SSOT dihitung dgn model floor yang
salah. Basis MAE-aware+funding (metodologi SSOT):

| | SSOT terpasang | Model floor BENAR | Selisih |
|---|---|---|---|
| OOF PF | 1,712* / 1,721 | **1,664** | -3,3% |
| OOF PnL | $6.253 / $6.267 | **$5.810,33** | **-7,3%** |
| OOS PF | 1,343 | **1,340** | -0,2% |
| OOS PnL | $129,94 | **$127,70** | -1,7% |

Kelas kesalahan SAMA dgn temuan portfolio_limits semalam: backtest memodelkan lingkungan eksekusi
lebih ramah dari kenyataan. Dampak OOF nyata (-7,3% PnL), OOS kecil (-1,7%).

**Konsekuensi 2 -- kandidat floor OFF jadi LEBIH kuat, bukan lebih lemah.** Pembanding yang adil
(kedua-duanya model benar), basis MAE-aware+funding:

| | OOF PF | OOF PnL | OOS PF | OOS PnL |
|---|---|---|---|---|
| floor 0,7 (live, model benar) | 1,664 | $5.810,33 | 1,340 | $127,70 |
| floor OFF | **1,773** | **$6.849,97** | **1,390** | **$150,53** |
| selisih | +0,109 | **+17,9%** | +0,050 | **+17,9%** |

Semalam gap OOF cuma +$624 (model salah); sekarang **+$1.040 (+17,9%)**. Jadi "harga" yang dibayar
untuk mempertahankan proteksi sisi-bursa itu ~18% PnL, bukan ~8% seperti dikira.

**Konsekuensi 3 -- tuning floor_frac ternyata TIDAK relevan.** Dgn model benar, 0,2 (2,009) vs 0,7
(2,001) nyaris identik. Yang penting cuma floor ADA vs TIDAK ADA, bukan angkanya. Rekomendasi
"pilih 0,2 daripada OFF" di addendum sebelumnya dgn demikian kehilangan dasarnya -- 0,2 tidak lagi
menangkap "82% keuntungan"; nyaris tidak menangkap apa pun.

**Yang TIDAK berubah:** lantai derau OOS tetap +-0,537, jadi OOS tetap TIDAK BISA meresolusi
selisih +0,050. Status floor OFF tetap: kuat di 1 jendela (OOF), jendela kedua tak berdaya.

**Pelajaran metodologi.** Ini persis `feedback-uji-resolusi-eksekusi-asli`: mekanisme reaktif-waktu
(stop order) yang diuji pada resolusi kasar bisa **MEMBALIK arah kesimpulan**, bukan sekadar kurang
presisi. Hipotesis saya sendiri (model benar menguntungkan floor) terbalik setelah diukur. Aturan
turunan: **setiap mekanisme exit yang di live berupa ORDER DI BURSA wajib dimodelkan sbg order
(trigger intrabar + fill di level), bukan sbg pengecekan di close.** SL sudah benar sejak awal;
floor terlewat 1 bulan (sejak 2026-07-12).

**Artefak.** `core/evaluator.py` (+`guardian_floor_intrabar`); `pipeline/model/
run_oof_floor_frac_sweep.py` & `model/eval/holdout_oos.py` (+`--floor-intrabar`);
`data/live_cache/{oof,oos}_intrabar_ff07_trades.csv`.

**BELUM dikerjakan:** koreksi SSOT (`model_registry.json`, `inference_config.json`, dashboard VPS)
ke angka model-benar -- menunggu keputusan user.
