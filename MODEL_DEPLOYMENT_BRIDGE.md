# 🌉 MODEL DEPLOYMENT BRIDGE

Dokumen kontrak sinkronisasi untuk memastikan parameter hasil riset pemodelan (`Riset_pemodelan`) diterapkan secara sempurna ke Web App produksi (`swint_tradev2`).

---

## 🚀 Active Model: Cascade v3 (Deployed: 2026-05-15)

*   **Model Version:** `cascade_v3` (LGBM Entry → LSTM Confirmation → Guardian v3 Dynamic Exit)
*   **Trained Date:** 2026-05-15
*   **Active Run Folder (Riset):** `models/runs/holdout_20260515_001906/`
*   **Target Web App Directory:** `D:\Apps-Dev\swint_tradev2\`

---

## 📋 Parameter Checklist (SSOT via models/inference_config.json)

Semua parameter di bawah ini secara otomatis di-load oleh Web App melalui file `models/inference_config.json`. **Dilarang keras menyalin nilai secara manual ke dalam kode program.**

| Parameter | Nilai Riset (SSOT) | Status di Web App | Tanggal Verifikasi | Catatan / Justifikasi |
|---|---|---|---|---|
| `confidence_threshold_entry` | **0.65** | `[x] Applied` | 2026-05-23 | Optimal threshold from OOS temporal parameter sweep. |
| `lstm_adjust_mode` | **"hard_consensus"** | `[x] Applied` | 2026-05-23 | LSTM opposite pen block active. |
| `lstm_adjust_neutral_pen` | **0.00** | `[x] Applied` | 2026-05-23 | LSTM FLAT does not penalize LGBM. |
| `lstm_adjust_opposite_pen` | **0.99** | `[x] Applied` | 2026-05-23 | Hard block opposite signals. |
| `guardian.enabled` | **True** | `[x] Applied` | 2026-05-23 | Using Guardian v3 (multiclass exit model). |
| `guardian.threshold` | **0.60** | `[x] Applied` | 2026-05-23 | Hasil optimal dari parameter sweep OOS. |
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

*   **2026-05-15 (Cascade v3 Deploy):** Pengenalan Guardian v3 Multiclass + TP Momentum Mode. Trade bertambah +72%, Net PnL meroket +57% dibandingkan Guardian v2 Binary.
*   **2026-05-14 (Temporal OOS Validation):** Validasi model temporal OOS (Genuine Out-of-Sample) pada data Mei 2025 – April 2026 selesai. Hasil stabil: Winrate 88.93%, Drawdown 41.77%, Profit Factor 10.05.
*   **2026-05-12 (Sweet Spot LGBM+LSTM):** Penyesuaian `confidence_threshold_entry=0.62` dan menonaktifkan `LSTM_FLAT_REVIEW_ENABLED` guna memulihkan Winrate keseluruhan dari 57% kembali ke 78%.

---
*Dokumen ini dibuat otomatis oleh pipeline riset dan bertindak sebagai manual penyerahan (handover) model.*
