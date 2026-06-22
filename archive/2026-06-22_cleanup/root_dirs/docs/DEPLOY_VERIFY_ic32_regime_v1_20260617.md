# DEPLOY VERIFY PROMPT — Revert to ic32_regime_v1 (2026-06-17)

**Tujuan**: Verifikasi bahwa production (swint_tradev2) sudah benar-benar berjalan dengan model ic32_regime_v1 sesuai snapshot yang direstore.

**Konteks**: 
- Revert dari tb_genuine_v2_dynsize_lstm_cond (low conf 0.45 base) karena live degradation (quick SL + floating loss pada conf 0.46-0.56).
- Target: ic32_regime_v1 dengan threshold lebih tinggi (0.69/0.59), hard_consensus LSTM, regime FLIP, structural filter, Guardian clean_v2.

**Prasyarat sebelum verifikasi**:
- Deploy selesai: `python tools/deploy_production.py` (disarankan) atau `python tools/deploy_model.py` + VPS manual.
- Service sudah di-restart (otomatis jika pakai `deploy_production.py`).
- Backup folder ada: `models/backups/backup_YYYYMMDD_HHMMSS`
- Ownership VPS: file `models/` harus `swint:swint` (bukan `root:root`) agar UI bisa simpan Risk & Sizing.

---

## Langkah Verifikasi (Jalankan satu per satu)

### 1. Cek File & Timestamp
```bash
cd /path/to/swint_tradev2   # atau D:\Apps-Dev\swint_tradev2 di Windows

ls -l models/lgbm_baseline.pkl models/inference_config.json `
      models/feature_cols_v2.json models/guardian_feature_cols.json `
      models/guardian_best.pkl models/guardian_scaler.pkl

# Cek backup terbaru
ls -l models/backups/ | tail -5
```

**Expected**:
- File lgbm_baseline.pkl, inference_config, feature_cols harus lebih baru dari backup timestamp.
- Tidak ada error permission.

### 2. Validasi inference_config.json (Paling Penting)
```powershell
python -c "
import json
cfg = json.load(open('models/inference_config.json'))
print('=== MODEL IDENTITY ===')
print('model_type:', cfg.get('model_type'))
print('model_version:', cfg.get('model_version'))
print('note:', cfg.get('note', '')[:100], '...')

print('\n=== CASCADE / ENTRY THRESHOLDS (HARUS ic32) ===')
c = cfg['cascade']
print('lgbm_threshold_long:', c['lgbm_threshold_long'])   # harus 0.69
print('lgbm_threshold_short:', c['lgbm_threshold_short']) # harus 0.59
print('confidence_threshold_entry:', c['confidence_threshold_entry']) # harus 0.59
print('lstm_fusion_mode:', c['lstm_fusion_mode'])         # harus hard_consensus
print('lstm_confirmation_enabled:', c['lstm_confirmation_enabled'])
print('lstm_adjust_opposite_pen:', c['lstm_adjust_opposite_pen']) # 0.65

print('\n=== INFERENCE / SIZE ===')
i = cfg['inference']
print('confidence_full_size:', i['confidence_full_size']) # 0.69
print('confidence_half_size:', i['confidence_half_size']) # 0.59

print('\n=== GUARDIAN ===')
g = cfg['guardian']
print('enabled:', g['enabled'])
print('exit_threshold:', g['exit_threshold'])  # 0.65
print('min_hold_bars:', g['min_hold_bars'])    # 2

print('\n=== REGIME & ALIGNMENT ===')
print('hmm.enabled:', cfg['hmm']['enabled'])
ra = cfg.get('regime_alignment', {})
print('regime_alignment.enabled:', ra.get('enabled'))  # True (FLIP)

print('\n=== STRUCTURAL & RISK FILTERS ===')
print('structural_filter.enabled:', cfg['structural_filter']['enabled'])
print('rr_gate.enabled:', cfg['rr_gate']['enabled'])
print('volatility_circuit_breaker.enabled:', cfg['volatility_circuit_breaker']['enabled'])
print('risk.modal_per_trade:', cfg['risk']['modal_per_trade'])
print('risk.leverage_recommended:', cfg['risk']['leverage_recommended'])
"
```

**Expected values (harus match persis)**:
- model_version: ic32_regime_v1
- lgbm_threshold_long: 0.69
- lgbm_threshold_short: 0.59
- lstm_fusion_mode: hard_consensus
- guardian exit_threshold: 0.65
- regime_alignment.enabled: true
- confidence_threshold_entry: 0.59

Jika ada nilai yang berbeda → **STOP** dan perbaiki sebelum trading live.

### 3. Cek Feature Columns
```powershell
python -c "
import json
lgbm_f = json.load(open('models/feature_cols_v2.json'))
print('LGBM features count:', len(lgbm_f))
print('Sample first 5:', lgbm_f[:5])
print('Contains hmm_regime_enc:', 'hmm_regime_enc' in lgbm_f)
print('Contains h4_trend:', 'h4_trend' in lgbm_f)

g_f = json.load(open('models/guardian_feature_cols.json'))
print('\nGuardian features count:', len(g_f))
print('Contains dynamic bars_held_norm:', 'bars_held_norm' in g_f)
"
```

**Expected**: 33 untuk LGBM, 40 untuk Guardian (lihat daftar lengkap di config snapshot 2026-06-06).

### 4. Load Model (Smoke Test)
```powershell
python -c "
import joblib

print('Loading LGBM...')
lgbm = joblib.load('models/lgbm_baseline.pkl')
print('LGBM loaded OK. n_classes:', getattr(lgbm, 'n_classes_', '?'))

print('\nLoading Guardian...')
g = joblib.load('models/guardian_best.pkl')
print('Guardian loaded OK.')

# Optional: coba load LSTM jika dipakai
try:
    import torch
    lstm = torch.load('models/lstm_best.pt', map_location='cpu')
    print('LSTM loaded OK')
except Exception as e:
    print('LSTM load note:', str(e)[:80])
"
```

### 5. Cek Log / Service Status (setelah restart)
```powershell
# Linux / VPS
journalctl -u swint-trade -n 50 --no-pager | tail -20

# Atau cari log aplikasi (sesuaikan path)
# grep -i "ic32|model_version|inference_config" logs/*.log 2>/dev/null | tail -10
```

Cari baris yang menyebut "ic32_regime_v1" atau "confidence_threshold_entry: 0.59".

### 6. Verifikasi Live Behavior (setelah beberapa jam)
```powershell
# Dari research repo (bukan live repo)
python tools/live_db_bridge.py
python tools/trade_analyzer.py --file data/live_cache/hasil_livetrading.csv --no-save
```

- Cek apakah entry baru punya `Conf >= 0.59` (atau sesuai threshold di cascade).
- Perhatikan apakah SL rate turun dan hold time lebih reasonable dibanding cluster 16-17 Juni.
- Bandingkan dengan snapshot sebelum revert.

---

## Checklist Cepat (copy ke chat / issue)

- [ ] inference_config model_version = ic32_regime_v1
- [ ] LGBM thresholds 0.69 / 0.59
- [ ] lstm_fusion_mode = hard_consensus
- [ ] regime_alignment.enabled = true
- [ ] Guardian exit 0.65 + min_hold 2
- [ ] LGBM & Guardian load tanpa error
- [ ] Backup folder dibuat
- [ ] Service restart sukses
- [ ] Feature count LGBM=33, Guardian=40
- [ ] confidence_threshold_entry = 0.59

**Jika semua hijau** → deployment berhasil. Mulai monitor live trades baru.

**Jika ada yang merah** → rollback ke backup terbaru dan laporkan.

---

**Referensi Snapshot Asli**:
- inference_config snapshot_time: "2026-06-06 22:30:00"
- EXPERIMENTS.md entry: 2026-06-17 — Emergency Revert to ic32_regime_v1
- Revert reason: Live low-confidence TB entries (conf 0.46-0.56) causing rapid SL & floating losses.
