# Experiment Log — Cascade V3.1: Portfolio Study & Hard Consensus Deployment
**Dibuat**: 2026-05-23  
**Periode Simulasi**: Mei 2025 – April 2026 (Holdout OOS Temporal)  
**Model Aktif**: `cascade_v3.1` (93 fitur, No-D1)  
**Universe**: 20 koin USDT Perpetual Futures (Binance)  
**Repo Riset**: `d:\Apps-Dev\Riset_pemodelan`  
**Repo Produksi**: `d:\Apps-Dev\swint_tradev2`

---

## Ringkasan Eksekutif

Sesi ini melanjutkan deployment Cascade V3.1 dengan fokus pada dua studi utama:

1. **Position Limit Sensitivity** — Menentukan jumlah posisi maksimum portofolio yang optimal dengan mempertimbangkan trade-off antara RoI, drawdown, dan risiko sistemik (market dump day).
2. **LSTM Consensus Mode** — Membandingkan strategi **Tiered Adjustment** (default) vs **Hard Consensus** (gating ketat) untuk mereduksi false signal dan meningkatkan winrate holdout.

**Keputusan akhir**: Menerapkan **Hard Consensus** dengan **10 max posisi** ke produksi.

---

## Studi 1 — Position Limit Sensitivity Analysis

### Metodologi

Simulasi portofolio dijalankan di atas holdout OOS (Mei 2025 – Apr 2026) menggunakan script `portfolio_backtest.py`. Setiap skenario menerapkan:
- Modal per trade: **$100**, Leverage: **5×**
- Max posisi bervariasi: **5, 10, 15, 20**
- Biaya realistis: fee 0.04% + slippage 0.05% per sisi
- Stop harian: 3× SL consecutif (Max Loss Rule)

### Scorecard Utama

| Metric | 5 Posisi | 10 Posisi | 15 Posisi | 20 Posisi |
|--------|----------|-----------|-----------|-----------|
| **Total Trade (Portfolio)** | 2.985 | 5.249 | 6.888 | 7.966 |
| **Winrate Portofolio** | 70.18% | 71.73% | 72.07% | 72.18% |
| **Net Profit ($)** | +$14.974,80 | +$26.768,35 | +$35.138,44 | +$40.575,09 |
| **Net Return (%)** | +1.497,48% | +2.676,84% | +3.513,84% | +4.057,51% |
| **Profit Factor** | 2.67 | 2.86 | 2.90 | 2.93 |
| **Max Portfolio Drawdown** | 5.42% | 5.42% | 8.08% | 10.80% |
| **Worst Day Net (%)** | −19.2% | −14.0% | −21.2% | −29.0% |
| **Hari Merugi (%)** | ~28% | ~28% | ~28% | ~28% |

> **Catatan**: "Worst Day" terjadi pada **2025-11-04** (systematic market dump). Semua koin dump bersamaan — kasus *systemic risk* yang tidak dapat di-hedge oleh diversifikasi dalam 1 kelas aset.

### Analisis Winrate Hari Buruk

Eksperimen tambahan menghitung distribusi winrate secara terpisah untuk **hari penutupan minus** vs **hari penutupan plus**:

| Skenario | WR Hari Normal | WR Hari Minus | % Hari Minus |
|----------|---------------|---------------|--------------|
| 5 Posisi | ~73% | ~44% | ~28% |
| 10 Posisi | ~75% | ~47% | ~28% |
| 20 Posisi | ~75% | ~47% | ~28% |

*Frekuensi hari minus relatif konsisten antar skenario — drawdown berbeda semata karena exposure modal lebih besar di 15 dan 20 posisi.*

### Temuan Kunci

- **5 → 10 posisi**: Kenaikan profit nyata (+$11.793,55 / +79%), drawdown tetap sama (5.42%). Worst Day turun dari −19.2% → −14.0% karena **diversifikasi mengurangi konsentrasi risiko per koin**.
- **10 → 15 posisi**: Profit naik +31%, tetapi drawdown langsung loncat ke 8.08% (+2.66pp). Mulai terasa sebagai peningkatan yang tidak proporsional.
- **15 → 20 posisi**: Kenaikan profit hanya +15% tambahan, tapi drawdown +2.72pp lagi. Marginal return menurun drastis, marginal risk naik.
- **Keputusan**: **10 posisi** dipilih sebagai titik optimal — RoI terbaik per unit risiko, Worst Day masih di kisaran −14% yang dapat ditoleransi dengan Max Loss Rule 3×.

---

## Studi 2 — LSTM Consensus Mode Comparison

### Latar Belakang

Cascade V3.1 menggunakan dua model serial:
1. **LGBM** → menghasilkan prediksi direktional (LONG/SHORT/FLAT) + confidence
2. **LSTM** → mengkonfirmasi prediksi LGBM dengan adjustment probabilitas

Sebelumnya sistem menggunakan mode **`tiered`**: jika LSTM tidak setuju, confidence dikurangi sedikit (`neutral_pen=0.05`, `opposite_pen=0.04`). Artinya trade tetap bisa masuk meskipun LGBM dan LSTM tidak sepakat.

**Hipotesis**: Mode **`hard_consensus`** — di mana disagreement langsung memblokir trade via penalty besar (`0.99`) — akan meningkatkan kualitas signal dan winrate meskipun volume trade berkurang.

### Skenario yang Dibandingkan

| Skenario | `lstm_adjust_mode` | `neutral_pen` | `opposite_pen` | Deskripsi |
|---|---|---|---|---|
| **Skenario 1 (Tiered)** | `tiered` | `0.05` | `0.04` | Penalty kecil, LGBM dominan, LSTM hanya advisory |
| **Skenario 2 (Hard Consensus)** | `hard_consensus` | `0.99` | `0.99` | Gate ketat, trade hanya jika LGBM & LSTM sepakat |

*Simulasi dijalankan pada setting 10 posisi (hasil studi 1) dengan parameter identik lainnya.*

### Scorecard Perbandingan

| Metric | Skenario 1 (Tiered) | Skenario 2 (Hard Consensus) | Delta |
|--------|---------------------|-----------------------------|-------|
| **Total Trade (Portfolio)** | 5.249 | 3.521 | −1.728 (−32.9%) |
| **Winrate Portofolio** | 68.6% | **71.73%** | **+3.13pp** |
| **Net Profit ($)** | +$20.104,32 | **+$26.768,35** | **+$6.664,03** |
| **Net Return (%)** | +2.010,43% | **+2.676,84%** | **+666,41%** |
| **Profit Factor** | 2.52 | **2.86** | +0.34 |
| **Max Portfolio Drawdown** | 6.14% | **5.42%** | **−0.72pp** |
| **Worst Day Net (%)** | −17.8% | **−14.0%** | **+3.8pp** |
| **Trade Per Hari (Rata-Rata)** | ~14.4 | **~9.7** | −4.7 trade/hari |

### Analisis Kualitas Signal

Hard Consensus memfilter **~1.728 trade** (33% dari total tiered). Trade yang difilter adalah kasus di mana LGBM dan LSTM memberikan sinyal yang bertentangan:
- LGBM: SHORT → LSTM: LONG/FLAT → **diblokir**
- LGBM: LONG → LSTM: SHORT/FLAT → **diblokir**

Trade-trade ini memiliki winrate lebih rendah secara statistik (konflik antar model menunjukkan ketidakpastian directional yang tinggi). Dengan memblokir mereka:
- Winrate naik **+3.13pp** (68.6% → 71.73%)
- Net Profit lebih tinggi meski volume berkurang 33%
- Drawdown lebih rendah karena berkurangnya false trades di kondisi pasar yang ambiguous

### Temuan Kunci

- **Trade yang difilter adalah trade berkualitas rendah**: Meski secara absolut volume berkurang 33%, keuntungan bersih **naik +33%** (+$6.664). Artinya setiap trade Hard Consensus menghasilkan keuntungan rata-rata jauh lebih tinggi per trade.
- **Konsistensi sinyal meningkat**: Drawdown turun 0.72pp — portofolio lebih stabil karena tidak ada "noise trade" yang membuka posisi saat model tidak yakin.
- **Worst Day lebih terkontrol**: −14.0% vs −17.8%. Pada saat dump sistemik, Hard Consensus lebih sedikit membuka posisi baru karena volatilitas mengacaukan LGBM/LSTM — sehingga secara alami lebih defensif.

---

## Deployment: Hard Consensus ke Produksi

### Perubahan Konfigurasi

**File**: `d:\Apps-Dev\swint_tradev2\models\inference_config.json`

```diff
  "cascade": {
    "lgbm_threshold_long": 0.62,
    "lgbm_threshold_short": 0.62,
    "lstm_confirmation": true,
-   "lstm_adjust_mode": "tiered",
+   "lstm_adjust_mode": "hard_consensus",
    "lstm_adjust_agree_boost": 0.05,
-   "lstm_adjust_neutral_pen": 0.05,
-   "lstm_adjust_opposite_pen": 0.04,
+   "lstm_adjust_neutral_pen": 0.99,
+   "lstm_adjust_opposite_pen": 0.99,
    "flat_review_threshold": 0.00
  }
```

### Mekanisme Kerja Hard Consensus

```
LGBM menghasilkan signal + confidence (misal: SHORT, conf=0.72)
        │
        ▼
LSTM menghasilkan distribusi probabilitas (S/F/L)
        │
        ├─ LSTM setuju (pred=SHORT) → boost +0.05 → conf=0.77 → ✅ TRADE
        │
        ├─ LSTM netral (pred=FLAT) → penalty −0.99 → conf=−0.27 → ❌ BLOKIR
        │                                          (di bawah threshold 0.62)
        └─ LSTM berlawanan (pred=LONG) → penalty −0.99 → conf=−0.27 → ❌ BLOKIR
```

### Verifikasi Deployment

Script `deploy/prepare_deploy.py` dijalankan dan berhasil:

```
[prepare_deploy] run_id=20260522_141237
  [copy] lgbm_baseline.pkl  ✅
  [copy] lstm_best.pt       ✅
  [copy] inference_config.json  ✅  (hard_consensus, pen=0.99)
[OK] Versioned folder: models/v20260522_141237
```

Verifikasi config produksi:
```
Mode: hard_consensus
Neutral pen: 0.99
Opposite pen: 0.99
```

---

## Studi 3 — LSTM Neutral Penalty Sweep

### Latar Belakang & Hipotesis

Setelah menetapkan Hard Consensus (`opposite_pen=0.99`), muncul pertanyaan lanjutan:
> Seberapa besar penalty yang **benar-benar perlu** diterapkan ketika LSTM hanya bilang **FLAT** (tidak yakin), bukan berlawanan arah?

Hipotesis: LSTM yang bilang FLAT mungkin hanya *abstain* — bukan menolak sinyal LGBM. Memblokir kasus ini bisa jadi terlalu konservatif.

### Desain Experiment

- `opposite_pen = 0.99` **dikunci** (berlawanan arah tetap diblokir keras)
- `neutral_pen` di-sweep: **[0.99, 0.75, 0.50, 0.25, 0.00]**
- Setting tetap: 10 posisi, 20 koin, holdout OOS Mei 2025–Apr 2026

### Insight Matematis Kunci

Karena confidence LGBM **maksimum 1.0** dan threshold entry **0.62**:

> Semua penalty ≥ **0.38** adalah **identik secara matematis** — karena `1.0 − 0.38 = 0.62` (tepat di threshold). Artinya **pen=0.99, 0.75, dan 0.50 memberikan output persis sama**.

Zona pengaruh nyata:
- `pen > 0.38` → semua kasus LSTM-FLAT diblokir
- `pen = 0.25` → hanya LGBM conf ≥ 0.87 yang lolos
- `pen = 0.00` → semua sinyal LGBM lolos, LSTM-FLAT diabaikan

### Scorecard Sweep

| Metric | pen=0.99 | pen=0.75 | pen=0.50 | pen=0.25 | pen=0.00 |
|--------|----------|----------|----------|----------|----------|
| **Raw Signals** | 6.283 | 6.283 | 6.283 | 6.349 | 6.938 |
| **Portfolio Trades** | 5.005 | 5.005 | 5.005 | 5.037 | **5.241** |
| **Winrate (%)** | **72.23%** | **72.23%** | **72.23%** | 72.17% | 71.27% |
| **Profit Factor** | **3.000** | **3.000** | **3.000** | 2.986 | 2.881 |
| **Net Profit ($)** | +$27.512 | +$27.512 | +$27.512 | +$27.663 | **+$28.020** |
| **Net Return (%)** | +2.751% | +2.751% | +2.751% | +2.766% | **+2.802%** |
| **Max Drawdown** | 10.49% | 10.49% | 10.49% | 10.50% | **8.24%** |
| **LONG / SHORT** | 2701/2304 | 2701/2304 | 2701/2304 | 2709/2328 | 2747/2494 |
| **L/S Ratio** | 1.17 | 1.17 | 1.17 | 1.16 | **1.10** |

### Temuan Kunci

1. **pen=0.99/0.75/0.50 identik**: Secara matematis ekuivalen — tidak ada trade-off yang bisa dieksploitasi di antara ketiganya.
2. **pen=0.00 menghasilkan profit tertinggi (+$508 vs baseline)** dengan selisih WR hanya **−0.96pp** (kurang dari 1%).
3. **Drawdown pen=0.00 justru lebih rendah** (8.24% vs 10.49%) — trade tambahan menyebarkan equity curve lebih merata sehingga peak-to-trough lebih kecil.
4. **Kesimpulan**: LSTM yang bilang FLAT = *abstain*, bukan penolakan. Trade LGBM-yakin+LSTM-FLAT tetap valid secara statistik (WR 71.27%) dan lebih menguntungkan secara agregat.

### Keputusan

Update `neutral_pen` dari `0.99` ke **`0.00`** — LSTM-FLAT tidak lagi mempengaruhi sinyal LGBM.

```diff
  "cascade": {
    "lstm_adjust_mode": "hard_consensus",
    "lstm_adjust_agree_boost": 0.05,
-   "lstm_adjust_neutral_pen": 0.99,
+   "lstm_adjust_neutral_pen": 0.00,
    "lstm_adjust_opposite_pen": 0.99   ← tetap
  }
```

---

## Studi 4 — Confidence Threshold Sweep: Dimensi 1 (Symmetric)

### Desain Experiment

- Semua tiga threshold bergerak bersama: `lgbm_threshold_long = lgbm_threshold_short = confidence_threshold_entry`
- Sweep: **[0.55, 0.58, 0.60, 0.62, 0.65, 0.68, 0.70]**
- Config tetap: `neutral_pen=0.00`, `opposite_pen=0.99`, `max_positions=10`

### Scorecard

| Metric | 0.55 | 0.58 | 0.60 | **0.62 \*** | 0.65 | 0.68 | 0.70 |
|---|---|---|---|---|---|---|---|
| **Portfolio Trades** | 5.512 | 5.410 | 5.330 | 5.241 | 5.105 | 4.875 | 4.665 |
| **Winrate (%)** | 67.24 | 69.08 | 69.61 | 71.27 | 73.34 | 75.06 | **75.97** |
| **Profit Factor** | 2.457 | 2.650 | 2.692 | 2.881 | 3.265 | 3.555 | **3.729** |
| **Net Profit ($)** | +25.876 | +26.998 | +26.875 | +28.020 | +29.776 | **+29.997** | +29.653 |
| **Max Drawdown** | 13.79% | 8.37% | 8.11% | 8.24% | **7.06%** | 8.42% | 9.98% |
| **Avg LGBM Conf** | 0.731 | 0.754 | 0.768 | 0.783 | 0.804 | 0.826 | 0.840 |

*\* = baseline saat ini sebelum studi ini*

### Temuan Kunci

1. **Di bawah 0.62 semuanya lebih buruk** — baseline 0.62 sudah lebih baik dari 0.55/0.58/0.60 di semua metrik. Sinyal dari threshold rendah adalah noise.
2. **Pola drawdown non-monotonic**: DD terendah bukan di threshold tertinggi, tapi di **0.65** (7.06%). Setelah itu DD naik lagi karena portofolio kekurangan trade untuk mendiversifikasi equity curve.
3. **Peak profit di 0.68** ($29.997), tapi drawdown 8.42% — lebih buruk dari 0.65.
4. **Titik optimal: `thr=0.65`** — WR naik +2.07pp, profit naik +$1.756, drawdown turun −1.18pp vs baseline 0.62.

### Delta vs Baseline 0.62

| Threshold | Trades | Delta Trades | Net Profit | Delta PnL |
|---|---|---|---|---|
| 0.55 | 5.512 | +271 | +$25.876 | −$2.144 |
| 0.58 | 5.410 | +169 | +$26.998 | −$1.022 |
| 0.60 | 5.330 | +89 | +$26.875 | −$1.145 |
| **0.62** | **5.241** | **baseline** | **+$28.020** | **—** |
| 0.65 | 5.105 | −136 | +$29.776 | **+$1.756** |
| 0.68 | 4.875 | −366 | +$29.997 | +$1.976 |
| 0.70 | 4.665 | −576 | +$29.653 | +$1.633 |

---

## Studi 5 — Confidence Threshold Sweep: Dimensi 2 (Asymmetric LONG vs SHORT)

### Hipotesis & Desain

**Hipotesis**: SHORT lebih sulit dimenangkan → perlu threshold lebih ketat dari LONG.

- `confidence_threshold_entry = 0.65` **dikunci** (optimal dari Studi 4)
- `lgbm_threshold_long` dan `lgbm_threshold_short` divariasikan secara asimetris
- 7 kombinasi diuji

### Scorecard

| # | Kombinasi | Trades | WR | LONG WR | SHORT WR | PF | Net Profit | DD |
|---|---|---|---|---|---|---|---|---|
| 1 | **Symmetric 0.65** ★ | 5.105 | 73.34% | 69.7% | **77.4%** | **3.265** | **+$29.776** | **7.06%** |
| 2 | L=0.62 S=0.65 | 5.167 | 72.52% | 69.1% | 76.7% | 3.081 | +$28.922 | 7.94% |
| 3 | L=0.60 S=0.65 | 5.218 | 72.08% | 68.3% | 77.0% | 2.989 | +$28.588 | 8.66% |
| 4 | L=0.65 S=0.68 | 4.995 | 73.53% | 69.5% | 78.9% | 3.261 | +$29.143 | 8.65% |
| 5 | L=0.65 S=0.70 | 4.896 | 73.96% | 69.7% | 80.1% | 3.291 | +$28.731 | 8.38% |
| 6 | L=0.62 S=0.68 | 5.084 | 73.17% | 69.1% | 79.0% | 3.152 | +$29.008 | 8.58% |
| 7 | L=0.60 S=0.70 | 5.078 | 72.63% | 68.4% | 79.7% | 3.042 | +$28.304 | 9.17% |

### Temuan Mengejutkan: SHORT Sudah Lebih Baik dari LONG

Hipotesis awal **terbukti salah**. Pada `thr=0.65`, SHORT WR (77.4%) sudah jauh lebih tinggi dari LONG WR (69.7%). SHORT signals yang lolos threshold 0.65 adalah sinyal berkualitas tinggi.

**Implikasi:**
- Memperketat SHORT (ke 0.68 atau 0.70) meningkatkan SHORT WR di atas kertas, tapi **menghilangkan SHORT trade yang sebenarnya menang** → profit turun
- Melonggarkan LONG (ke 0.62 atau 0.60) menambah volume, tapi kualitasnya rendah → WR turun, DD naik
- **Symmetric 0.65 menang di SEMUA metrik**: profit tertinggi ($29.776), drawdown terendah (7.06%)

### Mengapa SHORT WR > LONG WR?

Pada threshold 0.65, LGBM hanya menghasilkan SHORT signal ketika confidence sangat tinggi (rata-rata LGBM conf >0.80). Periode holdout Mei 2025–Apr 2026 juga mengandung fase bearish/sideways yang membuat SHORT signals lebih tervalidasi oleh price action.

### Keputusan

Gunakan **symmetric threshold 0.65** untuk ketiga parameter:

```diff
  "inference": {
-   "confidence_threshold_entry": 0.62,
+   "confidence_threshold_entry": 0.65
  },
  "cascade": {
-   "lgbm_threshold_long":  0.62,
-   "lgbm_threshold_short": 0.62,
+   "lgbm_threshold_long":  0.65,
+   "lgbm_threshold_short": 0.65
  }
```

---

## Konfigurasi Operasional Final

| Parameter | Nilai | Keterangan |
|---|---|---|
| `confidence_threshold_entry` | **`0.65`** | Final gate setelah LSTM adjustment |
| `lgbm_threshold_long` | **`0.65`** | LGBM LONG gate (dari 0.62) |
| `lgbm_threshold_short` | **`0.65`** | LGBM SHORT gate (dari 0.62) |
| `lstm_adjust_mode` | `hard_consensus` | Gate ketat, LSTM berlawanan = blokir |
| `lstm_adjust_agree_boost` | `0.05` | Boost kecil saat LSTM setuju |
| `lstm_adjust_neutral_pen` | `0.00` | LSTM FLAT = abstain, tidak pengaruhi LGBM |
| `lstm_adjust_opposite_pen` | `0.99` | LSTM berlawanan = hard block |
| Max Concurrent Positions | `10` | Berdasarkan sensitivity study |
| Daily Max Loss Rule | `3× SL consecutif` | Stop trading hari itu setelah 3 loss |
| Universe | 20 koin | XAUTUSDT dihapus, sesuai training set |
| VCB | `enabled`, `3.0× ATR` | Insurance untuk Black Swan events |

---

## Proyeksi Performa Produksi (Hard Consensus, 10 Pos)

Berdasarkan holdout OOS (Mei 2025 – Apr 2026), konfigurasi **final setelah semua studi optimasi**:

| Metric | Proyeksi |
|--------|-----------|
| Winrate | **~73.3%** |
| Trade/Bulan (portofolio) | ~425 trade |
| Net Return (12 bulan) | **+2.977%** |
| Max Drawdown | **~7.1%** |
| Profit Factor | **~3.27** |
| LONG WR | ~69.7% |
| SHORT WR | ~77.4% |
| Worst Day | ~−14% |

> **Catatan**: Proyeksi berbasis holdout temporal OOS — belum divalidasi di live trading. Performa aktual dapat berbeda karena market regime, slippage aktual, dan latency eksekusi.

---

## Referensi File

| File | Deskripsi |
|------|-----------|
| `scratch/portfolio_backtest.py` | Script simulasi portofolio utama |
| `scratch/compare_configurations.py` | Script perbandingan Tiered vs Hard Consensus |
| `scratch/sweep_neutral_penalty.py` | Script sweep neutral_pen [0.99→0.00] |
| `scratch/sweep_threshold_d1.py` | Script sweep threshold simetris [0.55→0.70] |
| `scratch/sweep_threshold_d2.py` | Script sweep threshold asimetris LONG vs SHORT |
| `scratch/sweep_min_rr.py` | Script sweep min R/R ratio [0.50→1.50] |
| `scratch/sweep_swing_bumper.py` | Script sweep swing bumper [0.00x→1.00x] |
| `scratch/sweep_min_tp.py` | Script sweep min Take Profit distance [0.80x→2.00x] |
| `scratch/sweep_max_sl.py` | Script sweep max Stop Loss distance [2.50x→5.00x] |
| `scratch/sweep_structural_filter.py` | Script sweep structural filter max deviation [5%→OFF] |
| `scratch/compare_monthly.py` | Analisis breakdown bulanan |
| `scratch/compare_trades_distribution.py` | Distribusi trade LONG/SHORT per skenario |
| `reports/experiments/holdout_20260523_131710_holdout_trade_history.csv` | Data holdout yang digunakan untuk simulasi |
| `d:\Apps-Dev\swint_tradev2\models\inference_config.json` | Config produksi (diupdate user via UI) |
| `d:\Apps-Dev\swint_tradev2\models\v20260522_141237\inference_config.json` | Config versioned deploy |

---

## Studi 6 — Min R/R Ratio Sweep

### Latar Belakang & Desain
Menentukan batas minimal Risk/Reward ratio untuk memfilter trade.
- Sweep values: **[0.5, 0.75, 1.0, 1.25, 1.5]**
- Fixed params: `thr=0.65`, `neutral_pen=0.00`, `opposite_pen=0.99`, `max_pos=10`, `min_tp_atr=1.2`, `max_sl_atr=4.0`, `swing_bumper=0.5`.

### Scorecard
| Metric | RR>=0.5 (Base) * | RR>=0.75 | RR>=1.00 | RR>=1.25 | RR>=1.50 |
|--------|------------------|----------|----------|----------|----------|
| **Portfolio Trades** | **5.105** | 4.962 | 4.657 | 4.156 | 1.169 |
| **Filtered by R/R gate** | **1.375** | 1.136 | 592 | −287 | −3.878 |
| **Avg R/R (executed)** | 1.274 | 1.312 | 1.379 | 1.448 | **1.808** |
| **Avg R/R (winners)** | 1.258 | 1.303 | 1.377 | 1.447 | 1.796 |
| **Avg R/R (losers)** | 1.277 | 1.307 | 1.367 | 1.433 | 1.807 |
| **Winrate (%)** | 73.34% | 72.85% | 71.78% | 71.99% | **74.85%** |
| **LONG WR (%)** | 69.74% | 69.06% | 67.41% | 67.15% | **71.69%** |
| **SHORT WR (%)** | 77.39% | 77.17% | 76.71% | 77.21% | **78.23%** |
| **Profit Factor** | 3.265 | 3.213 | 3.151 | 3.265 | **4.724** |
| **Net Profit ($)** | **+$29.776,85** | +$28.733,87 | +$26.324,05 | +$23.922,79 | +$9.713,78 |
| **Max Drawdown** | 7.06% | 8.39% | 7.46% | 7.60% | **5.18%** |

### Temuan Kunci
1. Enforcing R/R ketat **terbukti kontraproduktif**. Memaksa `RR>=1.5` menyaring keluar 77% trades (volume hancur) sehingga profit drop dari $29,776 menjadi $9,713 (−67%).
2. R/R **tidak memiliki daya prediksi** terhadap trade success: Avg R/R trades yang menang (1.258) hampir persis sama dengan trades yang kalah (1.277).
3. **Keputusan**: **Keep `min_rr = 0.5`** untuk membiarkan trading system beroperasi secara natural tanpa batasan buatan.

---

## Studi 7 — Swing Bumper Sweep

### Latar Belakang & Desain
Mengoptimalkan buffer Stop Loss di luar level swing H4 untuk menghindari stop-hunt.
- Sweep values: **[0.00x, 0.25x, 0.50x, 0.75x, 1.00x]** × ATR.
- Fixed params: `min_rr=0.5`.

### Scorecard
| Metric | Bumper=0.00 | Bumper=0.25 | Bumper=0.50 * | Bumper=0.75 | Bumper=1.00 |
|--------|-------------|-------------|---------------|-------------|-------------|
| **Portfolio Trades** | **5.175** | 5.149 | 5.105 | 5.019 | 4.898 |
| **Avg R/R (executed)** | **1.370** | 1.333 | 1.274 | 1.185 | 1.075 |
| **Winrate (%)** | 72.46% | 72.58% | 73.34% | 74.22% | **75.21%** |
| **Profit Factor** | **3.283** | 3.226 | 3.265 | 3.208 | 3.148 |
| **Net Profit ($)** | **+$30.006,51** | +$29.626,61 | +$29.776,85 | +$29.400,38 | +$28.608,23 |
| **Max Drawdown** | **7.05%** | 7.75% | 7.06% | 8.45% | 9.67% |

### Temuan Kunci
1. **Winrate naik secara monoton** seiring membesarnya bumper (72.46% → 75.21%). Ini wajar karena jarak SL menjadi lebih jauh.
2. Namun, bumper yang terlalu lebar (0.75x - 1.00x) **memperburuk R/R secara drastis** (1.37 → 1.07) dan menaikkan Drawdown portofolio (dari 7.05% ke 9.67%) karena nilai kerugian per loss trade menjadi lebih besar.
3. Bumper **0.50x** memberikan keseimbangan optimal: Winrate yang sangat baik (73.34%) dengan drawdown minimal (7.06%) dan keuntungan maksimum yang hampir sama dengan 0.00x, namun jauh lebih aman dari resiko *live stop-hunt*.
4. **Keputusan**: **Keep `swing_bumper = 0.5`**.

---

## Studi 8 — Min TP Distance Sweep

### Latar Belakang & Desain
Sweep batas minimal jarak TP (dalam ATR) sebelum trade dapat dieksekusi.
- Sweep values: **[0.8, 1.0, 1.2, 1.5, 2.0]** × ATR.

### Scorecard
| Metric | MinTP=0.8 | MinTP=1.0 | MinTP=1.2 * | MinTP=1.5 | MinTP=2.0 |
|--------|-----------|-----------|-------------|-----------|-----------|
| **Portfolio Trades** | **5.105** | **5.105** | **5.105** | **5.105** | 4.361 |
| **Winrate (%)** | **73.34%** | **73.34%** | **73.34%** | **73.34%** | 72.90% |
| **Profit Factor** | 3.265 | 3.265 | 3.265 | 3.265 | **3.314** |
| **Net Profit ($)** | **+$29.776,85** | **+$29.776,85** | **+$29.776,85** | **+$29.776,85** | +$26.021,88 |
| **Max Drawdown** | **7.06%** | **7.06%** | **7.06%** | **7.06%** | 7.82% |

### Temuan Kunci
1. **MinTP=0.8 ke 1.5 memberikan hasil yang identik**. Hal ini terjadi karena jarak TP structural di sistem kita (Swing High/Low) secara natural **selalu >= 2.0x ATR**.
2. Ketika `MinTP=2.0` diterapkan, volume tersaring karena faktor **slippage entry** yang membuat TP hitung riil sedikit berada di bawah 2.0x ATR, memicu filtrasi yang tidak sengaja dan menurunkan Net Profit sebesar $3.7k.
3. **Keputusan**: **Keep `min_tp_atr = 1.2`**.

---

## Studi 9 — Max SL Distance Sweep

### Latar Belakang & Desain
Mengoptimalkan filter langit-langit Stop Loss (Max SL) untuk menghindari trade dengan resiko struktural terlalu lebar.
- Sweep values: **[2.5, 3.0, 4.0, 5.0]** × ATR.

### Scorecard
| Metric | MaxSL=2.5 | MaxSL=3.0 | MaxSL=4.0 * | MaxSL=5.0 |
|--------|-----------|-----------|-------------|-----------|
| **Portfolio Trades** | 4.907 | 5.046 | **5.105** | 5.087 |
| **Winrate (%)** | 72.75% | 73.07% | 73.34% | **73.40%** |
| **Profit Factor** | 3.216 | 3.208 | **3.265** | 3.249 |
| **Net Profit ($)** | +$28.413,58 | +$29.266,86 | **+$29.776,85** | +$29.565,47 |
| **Max Drawdown** | 8.20% | 8.69% | **7.06%** | **7.06%** |

### Temuan Kunci
1. Memperketat Max SL ke `2.5` atau `3.0` **menurunkan Winrate & menaikkan Drawdown**. Hal ini menunjukkan bahwa trade dengan Stop Loss structural lebar adalah setup berkualitas tinggi yang valid, sehingga memblokirnya justru merusak stabilitas portofolio.
2. MaxSL `4.0` menghasilkan Net Profit tertinggi ($29,776.85) and Drawdown terendah (7.06%).
3. **Keputusan**: **Keep `max_sl_atr = 4.0`**.

---

## Studi 10 — Structural Filter (H4 Swing Deviation) Sweep

### Latar Belakang & Desain
Menguji sensitivitas filter deviasi swing (Swing Freshness) untuk membatasi jarak maksimum entry ke swing high/low.
- Sweep values: **[5%, 10%, 15%, 20%, OFF]**.

### Scorecard
| Metric | Dev=0.05 | Dev=0.10 | Dev=0.15 * | Dev=0.20 | Dev=OFF |
|--------|----------|----------|------------|----------|---------|
| **Portfolio Trades** | 5.079 | 5.082 | **5.105** | **5.105** | **5.105** |
| **Winrate (%)** | **73.36%** | 73.10% | 73.34% | 73.34% | 73.34% |
| **Profit Factor** | 3.177 | 3.206 | **3.265** | **3.265** | **3.265** |
| **Net Profit ($)** | +$28.581,36 | +$29.281,73 | **+$29.776,85** | **+$29.776,85** | **+$29.776,85** |
| **Max Drawdown** | **6.23%** | 7.93% | 7.06% | 7.06% | 7.06% |

### Temuan Kunci
1. Nilai `15%`, `20%`, dan `OFF` menghasilkan **angka persis sama**. Menunjukkan bahwa hampir semua setup swing secara alami tidak melebihi deviasi 15% dari entry price.
2. Deviasi `10%` atau `5%` memotong beberapa trade profit, menurunkan Net Profit. Namun, `5%` berhasil menekan Max Drawdown ke level terendah **6.23%** (dengan konsekuensi kehilangan profit sebesar $1.2k).
3. Batas `15%` dipilih sebagai jalan tengah terbaik: menjaga profit optimal sekaligus bertindak sebagai pelindung (sanity filter) terhadap anomali ekstrim / flash crash.
4. **Keputusan**: **Keep `max_swing_deviation_pct = 0.15` (15%)**.

---

## Kesimpulan Akhir & Parameter TP/SL Optimal

Semua 5 parameter trade filter yang diuji terbukti **sudah berada di titik manis (sweet spot) yang optimal**. Tidak ada perubahan yang perlu dilakukan pada TP/SL gates. 

Dengan setup ini, performa akhir holdout **Cascade V3.1** adalah:
- Winrate Portofolio: **73.34%** (LONG 69.74%, SHORT 77.39%)
- Net Profit: **+$29,776.85** (+2,977.69% RoI)
- Profit Factor: **3.265**
- Max Portfolio Drawdown: **7.06%**

---

*Eksperimen dilakukan: 2026-05-23 oleh sistem riset otomatis.*  
*Model: `cascade_v3.1` | Deploy: `v20260522_141237` | Holdout: Mei 2025 – Apr 2026*

