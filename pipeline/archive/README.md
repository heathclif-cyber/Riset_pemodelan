# Archive — Old LSTM Training Pipelines

Folder ini berisi script LSTM training lama yang sudah **tidak digunakan lagi**.

## Dua Filosofi LSTM yang Pernah Dipakai

### 1. V2-Style (Original cascade_v2 era — yang perform bagus di live Mei 2026)
- LSTM **menggunakan fitur yang sama persis** dengan LGBM entry model (`feature_cols_v2.json` / FEATURE_COLS_V3, 93–104 fitur).
- Label biasanya sama atau sangat mirip dengan LGBM (swing labels atau sederhana).
- Peran utama: confirmation + flat review / soft adjustment.
- Script lama yang pakai pendekatan ini: `05_train_lstm.py` (dan sweep-nya).
- **Inilah yang user maksud** ketika bilang "di v2 itu sama dengan lgbm fiturnya".

### 2. Advanced Momentum Detector (5a/5b/5c — eksperimen v4.x)
- LSTM pakai **fitur terpisah** (LSTM_MOMENTUM_FEATURES, ~18 fitur trajectory + orderflow).
- Label khusus: momentum labels (N=8 atau 12) dengan 05a.
- Tujuan: LSTM sebagai independent corrector/booster yang belajar pola temporal sendiri.
- Pipeline saat ini: 05a → 05b → 05c.

## Struktur Saat Ini (2026-06)

**Untuk cascade_v2.5_hybrid dan revival spirit v2** → sebaiknya pakai pendekatan #1 (fitur sama dengan LGBM).

**Untuk eksperimen advanced** → pakai 05a/05b/05c (pendekatan #2).

## File yang Diarsipkan

| File Lama                        | Filosofi     | Alasan Diarsipkan |
|----------------------------------|--------------|-------------------|
| `05_train_lstm.py`               | V2-style     | Pakai swing labels + fitur flat (selalu cenderung FLAT) |
| `05_train_lstm_seq_sweep.py`     | V2-style     | Versi sweep lama |
| `05b_build_h4_sequences.py`      | Advanced     | H4 sequence tidak align dengan target H1 |
| `05c_train_lstm_momentum.py`     | Advanced     | Versi H4 + double weighting (F1 ≈ random) |

**Rekomendasi saat ini (Juni 2026):**
- Untuk `cascade_v2.5_hybrid` → gunakan `pipeline/05_train_lstm_v2_style.py` (sama fitur dengan LGBM + RobustScaler).
- Jalur advanced (05a/05b/05c) sudah dipindah ke `archive/experimental_momentum_detector/`.
- Script `05_train_lstm_v2_style.py` sudah diperbaiki: per-fold RobustScaler + training loop lengkap.

Jangan jalankan file lama di folder ini kecuali untuk referensi historis atau audit.
