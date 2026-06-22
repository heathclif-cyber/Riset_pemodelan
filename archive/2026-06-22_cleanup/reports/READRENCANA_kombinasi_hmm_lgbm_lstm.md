# RENCANA — Kombinasi HMM + LGBM + LSTM (edge tiap model, WR/PF naik, PnL & volume nyaris tetap)

> Dokumen rencana. Belum dieksekusi. Saat eksekusi dimulai, salin tiap Track ke
> `EXPERIMENTS.md` (TAHAP 2) sebelum jalankan script. Holdout tetap amplop tersegel
> (ATURAN 1) — semua keputusan di sini diambil dari OOF saja.
>
> Dibuat: 2026-06-22 · Universe: ic32 · Pipeline: regime_v2

---

## 1. Tujuan & definisi sukses

**Goal user:** gabungkan HMM, LGBM, LSTM sehingga **setiap model menyumbang edge nyata**,
output akhir **WR & PF tinggi**, dengan **perubahan PnL dan volume (jumlah trade) minimal**.

Ini bukan sekadar "tambah filter". Filter naif (LSTM veto) memang menaikkan WR/PF tapi
**memotong volume dan PnL** — lihat §3. Yang dicari: kombinasi yang **menukar trade jelek
dengan trade bagus** (volume-neutral), bukan yang **membuang** trade.

### Gate kelulusan (semua harus terpenuhi, diukur OOF vs baseline LGBM-only)

| Kriteria | Target |
|---|---|
| WR | ≥ +2.0 pp vs baseline |
| PF | ≥ +0.15 vs baseline |
| Volume (trades) | dalam **±10%** baseline (bukan turun) |
| PnL total | ≥ **−5%** baseline (idealnya ≥ baseline) |
| PPT (PnL/trade) | ≥ baseline |
| **Edge tiap layer** | drop-one OOF: tiap model punya kontribusi marginal positif (§6) |

> Gate "edge tiap layer" adalah inti permintaan user: kalau sebuah layer hanya memotong
> volume tanpa menaikkan PF/PPT, layer itu **tidak punya edge** → buang, jangan dipaksakan.

---

## 2. Baseline (angka aktual, OOF 2020-01 → 2026-03, purged CV)

**LGBM `ic32_regime_v2`** (33 fitur, `hmm_regime_enc` sudah jadi salah satu fitur),
thr_long=0.75 / thr_short=0.70 — pemenang sweep OOF:

| Metrik | Nilai |
|---|---|
| Trades | 12,861 |
| WR | 64.7% |
| PnL | $3,632 |
| PPT | $0.282 |
| F1 macro CV | 0.5962 |

**LSTM `ic32_lstm_candidate_v3_seq36`** (14 fitur, seq=36, candidate domain p≥0.55):
F1 macro 0.4114 (naik dari v2 0.3748). Edge nyata tapi lemah — cocok sebagai **konfirmasi**,
bukan sebagai sinyal arah mandiri.

**HMM `hmm_regime_model.pkl`** (config B). Di v2 dipakai sebagai **fitur** LGBM
(`hmm_regime_enc`). Peran regime-conditioning (threshold/sizing per-state) **belum** dipakai di v2.

---

## 3. Diagnosa — kenapa ensemble naif gagal goal ini

Bukti dari sweep & forensik yang sudah ada:

1. **LSTM sebagai hard veto** (`combined_sweep_results.json` v1, lstm_thr=0.33):
   9,692 trade · WR 65.4% · PF 2.60 · $2,715.
   → vs baseline: WR **+0.7pp**, tapi trades **−25%**, PnL **−25%**. ❌ langgar gate volume/PnL.

2. **LSTM veto ketat** (v2, lstm_thr=0.60): WR 67.5% · PF 2.67 tapi **280 trade**.
   → WR/PF cantik, volume hancur. ❌

3. **LR meta-stacking** (`ic32_rv2_lr_meta_c55`, argmax): WR 48.5% · PF 1.46.
   → stacking lewat argmax malah **over-trade FLAT-adjacent**. Stacking butuh threshold,
   dan cenderung kolaps ke LGBM (korelasi tinggi). ❌

4. **Forensik LGBM↔LSTM** (731k bar): LSTM **berlawanan arah LGBM 50.2%** dari waktu.
   LSTM bukan prediktor arah yang independen — dia **filter survival/kualitas**, harus
   diperlakukan sebagai **skor lunak**, bukan suara arah setara.

**Kesimpulan diagnosa:** setiap kali LSTM dipakai sebagai gerbang biner, ia membuang trade →
volume & PnL turun. Untuk memenuhi goal, LSTM/HMM harus **mengubah komposisi & ukuran** trade,
bukan **mengurangi jumlahnya**. Tiga jalur memenuhi ini: sizing lunak (volume identik),
swap berkompensasi-threshold (volume netral), dan regime-conditioning dua arah.

---

## 4. Prinsip desain — pemisahan peran (tanpa redundansi)

Setiap model menjawab **pertanyaan berbeda**. Tidak boleh ada dua model menjawab hal yang sama
(itu sumber korelasi & redundansi yang membuat ensemble tak punya edge tambahan).

| Model | Pertanyaan | Sifat | Peran di stack |
|---|---|---|---|
| **LGBM** | "Arah apa, seberapa yakin?" | snapshot cross-sectional | **sinyal entry + confidence** |
| **HMM** | "Kapan sinyal LGBM layak dipercaya?" | konteks rezim | **conditioning threshold/sizing dua arah** |
| **LSTM** | "Apakah trajectory 36 bar mendukung?" | path-dependency temporal | **skor konfirmasi lunak (sizing/re-rank)** |
| **Guardian** | "Kapan keluar?" | manajemen posisi | **exit** (di luar scope ini, sudah min_hold=4) |

Implikasi: **HMM tidak hanya menaikkan threshold** (yang memotong volume di ranging). Ia juga
boleh **menurunkan** threshold di rezim favorable untuk mengembalikan volume pada bar berkualitas
→ net volume netral, komposisi membaik.

---

## 5. Track eksperimen (urut prioritas; tiap track = entri EXPERIMENTS.md sendiri)

> Semua diukur OOF. Jangan sentuh holdout sampai satu kandidat lolos §1 dan di-freeze.

### Track A — LSTM sebagai sizing multiplier (volume IDENTIK) ★ prioritas 1

**Hipotesis:** entry (LGBM+HMM) tidak diubah → trade 100% identik. LSTM hanya mengatur **ukuran**:
- LSTM agree (p arah sama ≥ τ_hi) → modal ×(1+a), clamp ≤ 2.0
- LSTM ragu/lawan (p ≤ τ_lo) → modal ×(1−b), clamp ≥ 0.5
- selebihnya → modal ×1.0

**Kenapa cocok goal:** terbukti pola DynSize (`ic32_dynsize_sweep`) menjaga **trades 100% identik**;
PnL absolut/PPT_norm naik karena modal mengalir ke trade berkualitas. Di sini "kualitas" =
konfirmasi LSTM, bukan sekadar confidence LGBM. WR mentah tak berubah, tapi **PnL-weighted WR & PF naik**
karena loser dikecilkan. Volume & jumlah trade = tak berubah sama sekali.

**Diubah:** hanya `modal_arr` per bar. Grid: τ_hi {0.40,0.45}, τ_lo {0.34,0.37}, a {0.3,0.5}, b {0.3,0.5}.
**Baseline:** LGBM+HMM fixed modal. **Gate:** PPT_norm ≥ +$0.01, trades identik, PF ≥ baseline.
**Script baru:** `pipeline/08_oof_ic32_lstm_sizing_sweep.py` (pola dari `08h_oof_ic32_dynsize_sweep.py`).

---

### Track B — Swap berkompensasi-threshold (volume NETRAL) ★ prioritas 2

**Hipotesis:** buang N trade terburuk (LSTM lawan kuat) **lalu** turunkan threshold LGBM
secukupnya agar masuk ~N trade baru yang LSTM-konfirmasi. Net trades ≈ baseline, tapi komposisi
bergeser ke WR lebih tinggi.

**Mekanik:**
1. LSTM veto: drop bar di mana `lstm_opp ≥ v` (arah lawan kuat).
2. Recovery: turunkan thr_long/thr_short step-by-step **hanya pada bar LSTM-konfirmasi**
   sampai `trades ≈ baseline ±10%`.

**Diubah:** pasangan (veto threshold v, thr_recovery). **Gate:** trades dalam ±10%, WR +≥2pp, PF +≥0.15, PnL ≥ −5%.
**Script baru:** `pipeline/08_oof_ic32_lstm_swap_sweep.py` (manfaatkan grid `05h_sweep_lgbm_lstm_combined.py`,
tambahkan loop recovery threshold + kendala volume-netral).

---

### Track C — HMM regime-conditioning dua arah (volume NETRAL) ★ prioritas 3

**Hipotesis:** per-state threshold yang **asimetris naik-turun** mengalihkan volume dari rezim
ber-PF-rendah (ranging) ke rezim ber-PF-tinggi (trending) tanpa mengubah total volume.

**Diubah:** `per_state_thresholds` (turunkan di TRENDING favorable, naikkan di RANGING),
dikalibrasi agar total trades ≈ baseline. Opsional: `regime_mult` sizing per-state.
**Catatan:** HMM sudah jadi fitur LGBM (`hmm_regime_enc`); track ini menambah peran
**conditioning** di atas fitur — uji apakah memberi edge **marginal** di luar yang sudah
diserap LGBM (kalau tidak → HMM-as-feature sudah cukup, conditioning tak punya edge tambahan).
**Gate:** trades ±10%, WR/PF naik sesuai §1, **dan** drop-one membuktikan conditioning > sekadar fitur.
**Script baru:** `pipeline/08_oof_ic32_hmm_conditioning_sweep.py`.

---

### Track D — Stacking meta dengan threshold + abstain (eksploratif)

**Hipotesis:** LR/GBM meta atas [lgbm_p*, lstm_p*, lstm_in_domain, hmm_state] bisa kalahkan
gating manual **jika** dipakai dengan threshold (bukan argmax) + kelas abstain untuk jaga volume.

**Diubah:** ganti argmax → threshold per-kelas pada `oof_lr_predictions.parquet`; tambah fitur
HMM state ke meta; sweep threshold agar volume-netral.
**Status:** jalankan **hanya jika** A–C belum tembus gate. Risiko kolaps ke LGBM tinggi (§3.3).
**Script:** perluas `pipeline/05i_train_lr_meta_lgbm_lstm.py` + sweep threshold OOF.

---

## 6. Audit "edge tiap model" — drop-one OOF (WAJIB, sebelum holdout)

Sebelum kandidat mana pun diajukan, jalankan ablation drop-one pada konfigurasi pemenang:

| Konfigurasi | Trades | WR | PF | PPT | Δ PF vs full | Δ PPT vs full |
|---|---|---|---|---|---|---|
| Full (LGBM+HMM+LSTM) | | | | | — | — |
| − LSTM | | | | | | |
| − HMM conditioning | | | | | | |
| LGBM only | | | | | | |

**Aturan keputusan:** sebuah layer "punya edge" hanya jika menghapusnya **menurunkan** PF atau PPT
**tanpa** menaikkan volume secara berarti. Kalau menghapus layer membuat metrik sama/lebih baik →
layer itu tak punya edge di stack ini → **keluarkan** (jujur ke goal user, jangan dipaksa masuk).
Artefak: `models/runs/ic32_regime_v2/drop_one_ablation.json`.

---

## 7. Protokol metodologi (mengikat — METHODOLOGY.md)

- **OOF saja** untuk semua sweep/threshold/sizing/veto/conditioning (ATURAN 1, 4).
- LSTM & LGBM OOF dari **purged CV retrain per fold**, purge = MAX_HOLDING_BARS (ATURAN 4).
  OOF yang dipakai sudah ada: `ic32_regime_v2/oof_predictions.parquet`,
  `ic32_lstm_candidate_v3_seq36/oof_lstm_predictions.parquet`.
- Scaler fit **di dalam** fold (ATURAN 3). Tidak ada fitur look-ahead (ATURAN 5).
- **Holdout (Apr–Jun 2026) tetap tersegel.** Dibuka **sekali**, hanya untuk kandidat tunggal
  pemenang OOF yang sudah di-freeze (config tidak boleh diubah setelah lihat holdout).
- Benchmark wajib vs `ic32_regime_v1` (live alpha, frozen — hanya dibaca) pada periode sama.

---

## 8. Urutan eksekusi & decision tree

```
1. Track A (sizing)  ──lolos §1?──► kandidat utama (volume identik = paling aman)
        │ tidak
        ▼
2. Track B (swap)    ──lolos §1?──► kandidat (volume netral)
        │ tidak
        ▼
3. Track C (HMM)     ──lolos §1?──► gabungkan dengan A bila ortogonal
        │ tidak
        ▼
4. Track D (meta)    ──lolos §1?──► kandidat; jika tidak → stack tetap LGBM+HMM (+min_hold4)
        │
        ▼
5. Drop-one ablation (§6) pada pemenang  ──semua layer ber-edge?──► freeze config
        │ ada layer tanpa edge
        ▼
   keluarkan layer itu, ulang §6
        │
        ▼
6. Holdout SEKALI (07_holdout_ic32_regime_v2.py)  ──lolos TAHAP 4 CLAUDE.md?──► update widyawardhana_model.md
```

Prioritas A→B→C karena: A menjamin volume identik (risiko paling kecil terhadap goal),
B menjaga volume netral, C mengalihkan volume; D paling berisiko redundansi.

---

## 9. Skrip & artefak

**Sudah ada (dipakai ulang):**
- `pipeline/05h_sweep_lgbm_lstm_combined.py` — basis grid gating LGBM×LSTM
- `pipeline/05i_train_lr_meta_lgbm_lstm.py` — basis Track D
- `pipeline/08h_oof_ic32_dynsize_sweep.py` — pola sizing untuk Track A
- OOF: `ic32_regime_v2/oof_predictions.parquet`, `ic32_lstm_candidate_v3_seq36/oof_lstm_predictions.parquet`

**Baru (akan dibuat saat eksekusi):**
- `pipeline/08_oof_ic32_lstm_sizing_sweep.py` (Track A)
- `pipeline/08_oof_ic32_lstm_swap_sweep.py` (Track B)
- `pipeline/08_oof_ic32_hmm_conditioning_sweep.py` (Track C)
- artefak ablation: `models/runs/ic32_regime_v2/drop_one_ablation.json`

---

## 10. Catatan

- LSTM **belum** dilatih ulang final untuk produksi v2 (candidate v3 seq36 ada sebagai OOF + model fold).
  Rencana ini berjalan di atas OOF yang sudah tersedia; training final LSTM produksi menyusul
  hanya jika satu Track lolos holdout.
- Jika **tidak ada** Track yang lolos §1 (mungkin: LGBM v2 sudah menyerap edge HMM via fitur, dan
  LSTM terlalu lemah untuk menambah tanpa memotong) → kesimpulan jujur: **stack tetap LGBM+HMM-feature
  + Guardian min_hold=4**, dan LSTM/HMM-conditioning tidak menambah edge yang sepadan. Itu hasil valid,
  bukan kegagalan — dicatat di EXPERIMENTS.md.
