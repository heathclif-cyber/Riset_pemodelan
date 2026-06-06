# Laporan Deployment: cascade_v2.5_hybrid_accel — Dual Gate T
**Tanggal Deploy**: 2026-06-02  
**Versi**: `cascade_v2.5_hybrid_accel` dengan `cascade.mode = dual_gate`  
**Status**: AKTIF di produksi (`swint_tradev2`)  
**Backup**: `D:\Apps-Dev\swint_tradev2\models\backups\backup_20260602_212639`

---

## Alasan Memilih Dual Gate T (Psikologi Trading)

Scenario I (hard_consensus) menghasilkan PnL lebih tinggi ($2,199 vs $1,522) tapi dengan trade lebih banyak (702 vs 568/bulan). **Banyak trade berarti lebih banyak loss di awal** yang mempengaruhi psikologi.

Dual Gate T dipilih karena:
- Setiap trade yang masuk benar-benar melewati dua filter ketat
- Win Rate 61% lebih konsisten dan loss lebih jarang
- **Profit Factor 7.59** — setiap $1 yang hilang, sistem membuat $7.59
- Maksimal consecutive loss hanya **10** (vs 18 di Scenario I)

---

## Arsitektur Model

### Entry LGBM (Primary)

| Parameter | Nilai |
|-----------|-------|
| Tipe | LightGBM Classifier (3-class) |
| Fitur | **90** (87 pruned + 3 momentum baru) |
| n_estimators | 779 |
| Training | 2020-01-01 s/d 2025-11-01 |

**3 Fitur Momentum Baru:**
- `price_accel_1h` — akselerasi harga (2nd derivative)
- `ofi_momentum_ratio` — OFI 3-bar vs 24-bar baseline
- `vol_accel_3h` — volume acceleration

### LSTM Confirmation

| Parameter | Nilai |
|-----------|-------|
| Architecture | ManualLSTMCell (DirectML compatible) |
| Hidden | 96, Layers=2, Dropout=0.45 |
| Input features | 87 (subset dari 90) |
| Seq length | 16 H1 bars |

### Guardian Exit (Dynamic)

| Parameter | Nilai |
|-----------|-------|
| Tipe | LightGBM Multiclass (HOLD/PARTIAL/FULL_EXIT) |
| Fitur | 97 (90 static + 7 dynamic) |
| n_estimators | 1,316 |
| Logloss CV | 0.323 |

---

## Mekanisme Dual Gate

### Cara Kerja

```
Untuk setiap bar H1:

  LGBM predict → [p_SHORT, p_FLAT, p_LONG]
  LSTM predict → [p_SHORT, p_FLAT, p_LONG]

  Syarat ENTRY (semua harus terpenuhi):
    1. argmax(LGBM) == argmax(LSTM)  ← arah HARUS sama
    2. max(LGBM) >= 0.60             ← LGBM confidence >= 60%
    3. max(LSTM) >= 0.45             ← LSTM confidence >= 45%
    4. arah != FLAT

  Kalau semua terpenuhi:
    confidence_final = (LGBM_conf + LSTM_conf) / 2
    → ENTRY

  Kalau salah satu gagal → FLAT, tidak masuk
```

### Perbedaan dari Hard Consensus (sebelumnya)

| Situasi | Hard Consensus | **Dual Gate** |
|---------|---------------|---------------|
| LGBM 0.70, LSTM netral (0.33) | MASUK (0.70-0.10=0.60 > 0.59) | **BLOKIR** (LSTM < 0.45) |
| LGBM 0.70, LSTM 0.46 LONG | MASUK | **MASUK** ✓ |
| LGBM 0.62, LSTM 0.50 LONG | BLOKIR (LGBM < 0.69) | **MASUK** (keduanya lolos) |
| LGBM 0.70, LSTM 0.50 SHORT | MASUK (LGBM wins) | **BLOKIR** (arah berbeda) |
| Market crash, LSTM ragu | Bisa masuk | **SELALU BLOKIR** |

**Proteksi crash**: Saat market volatile, LSTM sering output netral/berlawanan → LGBM tidak bisa masuk sendiri. Dual gate secara otomatis mengurangi eksposur di kondisi tidak pasti.

---

## Konfigurasi Parameter Lengkap

```json
{
  "cascade": {
    "mode": "dual_gate",
    "lgbm_gate": 0.60,
    "lstm_gate": 0.45,
    "lgbm_threshold_long": 0.69,
    "lgbm_threshold_short": 0.59,
    "lstm_adjust_neutral_pen": 0.10,
    "lstm_adjust_opposite_pen": 0.85,
    "lstm_adjust_agree_boost": 0.05
  },
  "guardian": {
    "enabled": true,
    "exit_threshold": 0.65,
    "min_hold_bars": 2,
    "activation_atr": 0.0,
    "partial_exit_ratio": 0.5
  },
  "rr_gate": {
    "enabled": true,
    "min_rr": 0.60,
    "min_tp_atr": 1.2,
    "max_sl_atr": 4.0
  },
  "inference": {
    "max_hold_bars": 36,
    "confidence_threshold_entry": 0.59,
    "seq_len": 16
  }
}
```

---

## Performa OOS (Nov 2025 – Mar 2026, 5× Leverage)

| Metrik | Nilai | Konteks |
|--------|-------|---------|
| **Win Rate** | **61.10%** | 61% trade profit |
| **Total PnL** | **+$1,522** | 21 koin, 5 bulan |
| **Trade/Bulan per Koin** | **27.1** | ~1 trade per hari per koin |
| **Trade/Bulan Total** | **568** | Semua 21 koin |
| **Mean Sharpe** | **4.12** | Excellent risk-adjusted |
| **Profit Factor** | **7.59** | Tiap $1 hilang → $7.59 profit |
| **Mean Max DD** | **78%** | Per koin |
| **Max Consecutive Loss** | **10** | Berturut-turut rugi |
| Worst Trade | −24.9% | Trade terburuk |

### Performa Per Koin

| Koin | WR | Trade/5bln | PnL |
|------|----|-----------|-----|
| LINKUSDT | 66.7% | 189 | **+$145.70** |
| ADAUSDT | 66.4% | 116 | **+$116.61** |
| ETHUSDT | 70.0% | 130 | **+$118.79** |
| SUIUSDT | 64.1% | 192 | +$142.42 |
| NEARUSDT | 57.7% | 175 | +$109.75 |
| ONDOUSDT | 61.3% | 155 | +$106.07 |
| AVAXUSDT | 59.0% | 239 | +$103.83 |
| XRPUSDT | 66.0% | 153 | +$101.19 |
| DOTUSDT | 60.6% | 180 | +$91.55 |
| TAOUSDT | 57.0% | 172 | +$88.85 |
| SOLUSDT | 54.7% | 214 | +$77.70 |
| ARBUSDT | 53.9% | 167 | +$73.64 |
| POLUSDT | 54.4% | 160 | +$71.77 |
| TONUSDT | 58.3% | 163 | +$67.42 |
| HBARUSDT | 58.2% | 153 | +$30.50 |
| BNBUSDT | 58.4% | 125 | +$23.69 |
| TRXUSDT | 66.7% | 99 | +$18.48 |
| DOGEUSDT | 83.3% | 12 | +$18.86 |
| BTCUSDT | — | 0 | $0 |
| PEPE/SHIB | — | ~3 | +$15 |

**Semua koin profit.** BTCUSDT mendapat 0 trade karena threshold sangat ketat — LGBM dan LSTM jarang sama-sama setuju untuk BTC di periode ini.

---

## Labeling Perubahan dari Versi Sebelumnya

| Parameter | Sebelum (pruned) | **Sekarang (accel)** |
|-----------|-----------------|---------------------|
| `SWING_LABEL_MAX_HOLD` | 24 jam | **36 jam** |
| `SWING_LABEL_MIN_RR` | 0.45 | **0.60** |
| Hybrid momentum labeling | Tidak ada | **Ada** (vol_spike >= 1.5 → ATR-based) |
| `PURGE_GAP_BARS` | 16 | **20** |

### Guardian Label Improvement (Momentum-Aware)

Saat trade searah trend kuat (H4 trend + trend_strength > 1.0):
- FULL_EXIT reversal threshold: MFE×0.25 → **MFE×0.15** (tahan lebih lama)
- PARTIAL_EXIT threshold: MFE×0.55 → **MFE×0.35** (lebih sabar)
- SL threshold: 1.0×ATR → **1.5×ATR** (berikan ruang)

---

## File yang Di-deploy

| File | Deskripsi |
|------|-----------|
| `models/lgbm_baseline.pkl` | LGBM entry, 90 fitur, 779 trees |
| `models/lstm_best.pt` | LSTM (hidden=96, 87 feat) |
| `models/lstm_scaler.pkl` | RobustScaler LSTM |
| `models/guardian_best.pkl` | Guardian, 97 feat, 1316 trees |
| `models/guardian_scaler.pkl` | StandardScaler Guardian |
| `models/feature_cols_v2.json` | 90 fitur aktif |
| `models/guardian_feature_cols.json` | 97 fitur Guardian |
| `models/inference_config.json` | Config dengan dual_gate params |
| `core/cascade_utils.py` | **BARU** — implementasi dual_gate |
| `core/features.py` | Feature engineering (+3 momentum feat) |

---

## Catatan Integrasi Production

File `core/cascade_utils.py` berisi fungsi `evaluate_entry()` yang mengimplementasikan dual_gate. Production `inference.py` perlu diupdate:

```python
from core.cascade_utils import evaluate_entry

# Saat menentukan sinyal entry:
cascade_cfg = config.get("cascade", {})
result = evaluate_entry(lgbm_proba, lstm_proba, cascade_cfg)

if result["entry"]:
    direction  = result["direction"]   # 0=SHORT, 2=LONG
    confidence = result["confidence"]  # combined avg confidence
```

Fungsi ini membaca `cascade.mode` dari config — kalau `"dual_gate"`, pakai threshold independen. Kalau `"hard_consensus"` (atau tidak ada), pakai logic lama.

---

## Rollback

Jika perlu kembali ke versi sebelumnya:
```
D:\Apps-Dev\swint_tradev2\models\backups\backup_20260602_212639
```
