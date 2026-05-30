# 🌉 MODEL DEPLOYMENT BRIDGE

Dokumen kontrak sinkronisasi untuk memastikan parameter hasil riset pemodelan (`Riset_pemodelan`) diterapkan secara sempurna ke Web App produksi (`swint_tradev2`).

---

## 🚀 Active Model: Cascade v4.1 (Deployed: 2026-05-29)

*   **Model Version:** `cascade_v4.1` (LGBM Entry → LSTM Confirmation → Exit Guardian v3 + Volatility Regime)
*   **Trained Date:** 2026-05-29
*   **Active Run Folder (Riset):** `models/runs/cascade_v4.1/`
*   **Target Web App Directory:** `D:\Apps-Dev\swint_tradev2\`

---

## 📋 Parameter Checklist (SSOT via models/inference_config.json)

Semua parameter di bawah ini secara otomatis di-load oleh Web App melalui file `models/inference_config.json`. **Dilarang keras menyalin nilai secara manual ke dalam kode program.**

| Parameter | Nilai Riset (SSOT) | Status di Web App | Tanggal Verifikasi | Catatan / Justifikasi |
|---|---|---|---|---|
| `lgbm_threshold_long` | **0.75** | `[x] Applied` | 2026-05-27 | Selective LONG entries to filter false breakout noise. |
| `lgbm_threshold_short` | **0.60** | `[x] Applied` | 2026-05-27 | Relaxed SHORT threshold to capture bear-regime profits. |
| `confidence_threshold_entry` | **0.65** | `[x] Applied` | 2026-05-23 | Optimal threshold from OOS temporal parameter sweep. |
| `lstm_adjust_mode` | **"hard_consensus"** | `[x] Applied` | 2026-05-23 | LSTM opposite pen block active. |
| `lstm_adjust_neutral_pen` | **0.00** | `[x] Applied` | 2026-05-23 | LSTM FLAT does not penalize LGBM. |
| `lstm_adjust_opposite_pen` | **0.99** | `[x] Applied` | 2026-05-23 | Hard block opposite signals. |
| `guardian.enabled` | **True** | `[x] Applied` | 2026-05-23 | Using Guardian v3 (multiclass exit model). |
| `guardian.threshold` | **0.65** | `[x] Applied` | 2026-05-27 | Exit threshold matching config.py. |
| `guardian.min_hold_bars` | **0** | `[x] Applied` | 2026-05-27 | Meniadakan zone buta guardian untuk emergency exit instan. |
| `guardian.activation_atr` | **0.0** | `[x] Applied` | 2026-05-27 | Meniadakan batas ATR minimum pemicu penyelamatan Guardian. |
| `trend_alignment.enabled` | **True** | `[x] Applied` | 2026-05-27 | Regime-Aware Gating H4 (Pen=0.10, Boost=0.05, Block=0.0). |
| `tp_sl_hybrid_mode` | **True** | `[x] Applied` | 2026-05-23 | Menggabungkan level Swing H4 terdekat + ATR Fallback. |
| `tp_sl_fallback_sl` | **1.5** | `[x] Applied` | 2026-05-23 | Diperlebar dari 1.0 ke 1.5 untuk menahan wick market. |
| `tp_sl_fallback_tp` | **2.0** | `[x] Applied` | 2026-05-23 | Standar kelonggaran target profit. |
| `tp_sl_max_sl` | **4.0** | `[x] Applied` | 2026-05-23 | Dinaikkan dari 3.0 ke 4.0. |
| `tp_sl_cooldown_enabled` | **False** | `[x] Applied` | 2026-05-23 | Terlalu agresif memblokir trade winner berturut-turut. |
| `tp_sl_trigger_mode` | **"close"** | `[x] Applied` | 2026-05-23 | (Riset) Close-based stop. **Catatan:** Khusus manual trading set 'highlow'. |
| `tp_sl_sizing_mode` | **"fixed"** | `[x] Applied` | 2026-05-23 | Pukul rata $100 per trade untuk manajemen risiko portofolio. |

---

## 🛠️ Code Compatibility Checklist (Kesesuaian Kode Program)

Sebelum menyalin file model baru, pastikan kode Web App (`swint_tradev2`) telah diupdate agar kompatibel dengan fitur dan struktur logika terbaru:

1.  **[OK] Fitur Baru (93 Fitur):**
    *   *Detail:* Web App harus memiliki modul `core/features.py` versi terbaru yang menghasilkan 93 fitur (termasuk slope H4, HTF daily context, dan 3 fitur trend quality baru).
    *   *Status:* **Terpenuhi** (di-deploy via commit `b5c6c0b`).
2.  **[OK] Guardian Dual Mode (EARLY + MOMENTUM):**
    *   *Detail:* Logic exit di web app tidak boleh langsung menutup posisi ketika menyentuh level TP. TP bertindak sebagai pemicu (trigger) untuk mengaktifkan **Guardian Momentum Mode** guna membiarkan profit berlari (ride momentum).
    *   *Status:* **Terpenuhi** (di-deploy via commit `91564e2` di `app/services/paper_trading.py`).
3.  **[OK] LSTM Sequence Length (16 Bar):**
    *   *Detail:* Web App harus menggunakan LSTM window 16 bar H1 (sebelumnya 32 bar).
    *   *Status:* **Terpenuhi** (telah disinkronkan di `app/services/inference.py`).

---

## 📅 Riwayat Update & Deployment

*   **2026-05-27 (Masterpiece V3.1 Gating Optimization & Deployment):** 
    Optimasi besar-besaran terhadap gerbang keluar-masuk sistem Cascade V3.1 di data temporal OOS (Nov 2025 – Mar 2026, 20 koin):
    *   *Exit Guardian Gate:* Mengatur `min_hold_bars = 0` dan `activation_atr = 0.0` (menghapus blind spot), menekan SL hits sebesar **44.3%** dan menyelamatkan modal dari kebocoran.
    *   *Asymmetric Entry Thresholds:* Menetapkan `LGBM_THRESHOLD_LONG = 0.75` dan `LGBM_THRESHOLD_SHORT = 0.60`, menyeimbangkan rasio arah LONG/SHORT menjadi 1.3:1 dan membalikkan PnL ke area positif.
    *   *H4 Trend Gating (Regime-Aware):* Mengaktifkan `TREND_ALIGNMENT_ENABLED = True` (`Pen=0.10, Boost=0.05, Block=0.00`). Mengangkat winrate LONG di atas batas psikologis ke **50.26%**, memangkas SL hits ke level terendah **78 kali**, dan menorehkan rekor profitabilitas holdout tertinggi sebesar **+$139.74 USD** dengan overall WR **57.27%**.
    *   *Deployment:* Seluruh parameter optimal baru telah sukses disinkronisasikan dan aktif 100% di bursa live produksi `swint_tradev2`.
*   **2026-05-15 (Cascade v3.1 Deploy):** Pengenalan Guardian v3 Multiclass + TP Momentum Mode. Trade bertambah +72%, Net PnL meroket +57% dibandingkan Guardian v2 Binary.
*   **2026-05-14 (Temporal OOS Validation):** Validasi model temporal OOS (Genuine Out-of-Sample) pada data Mei 2025 – April 2026 selesai. Hasil stabil: Winrate 88.93%, Drawdown 41.77%, Profit Factor 10.05.
*   **2026-05-12 (Sweet Spot LGBM+LSTM):** Penyesuaian `confidence_threshold_entry=0.62` dan menonaktifkan `LSTM_FLAT_REVIEW_ENABLED` guna memulihkan Winrate keseluruhan dari 57% kembali ke 78%.

---
*Dokumen ini dibuat otomatis oleh pipeline riset dan bertindak sebagai manual penyerahan (handover) model.*
