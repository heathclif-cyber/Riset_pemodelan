# AGENTS.md

All project instructions, constraints, and workflow are in [`CLAUDE.md`](CLAUDE.md).

**Read `CLAUDE.md` first**, lalu domain CLAUDE sesuai tugas (`features/`, `model/`, `tools/ops/`, `pipeline/data/`, `pipeline/model/`).

**JANGAN baca semua file `.md` di repo ini.** Repo ini punya 30+ file `.md` — cukup ikuti tabel
"Decision Tree" di `CLAUDE.md` utk pilih 1-2 file yang relevan dgn tugas saat ini, jangan glob/baca
semuanya. Khusus `EXPERIMENTS.md` (logbook, ribuan baris): cari entry spesifik via grep
tanggal/topik, JANGAN baca file itu utuh dari awal. `archive/` hanya dibaca kalau user minta
eksplisit. `.claude/skills/` khusus Claude Code — abaikan kalau bukan Claude Code.

---

## Workflow Agent — Wajib

1. **Baca `CLAUDE.md`** + domain CLAUDE sebelum tugas apa pun.
2. **Paparkan rencana lengkap ke user SEBELUM eksekusi** — sim, retrain, sweep, ubah production.
3. **Tunggu persetujuan** (atau koreksi) user, baru jalankan script / ubah kode.
4. **Catat hasil** di `EXPERIMENTS.md` (bukan file ad-hoc), dengan nama stack eksplisit: `LGBM + HMM + k5_mom + Guardian`.
5. **Jangan tune di holdout tersegel** (`TRAIN_CUTOFF_DATE = 2026-04-01`).
6. **Jangan deploy** tanpa approval eksplisit.

Format rencana minimal:

```
Tujuan | Baseline angka | Yang diubah | Script | Estimasi runtime | Kriteria lolos | Risiko
```