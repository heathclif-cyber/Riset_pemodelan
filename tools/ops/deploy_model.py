# -*- coding: utf-8 -*-
"""
-------------------------------------------------------------
 MODEL DEPLOYMENT AUTOMATION SCRIPT
-------------------------------------------------------------
Fungsi: Menyalin file model, scaler, config, dan dokumen bridge secara aman 
        dari repositori Riset_pemodelan ke repositori Web App produksi swint_tradev2.
Fitur:
  1. Validasi keberadaan path asal dan target.
  2. Backup otomatis seluruh file di target sebelum ditimpa (disimpan di subfolder timestamped).
  3. Proses penyalinan 9 file kunci (LGBM, LSTM, Guardian, Scalers, Configs, Bridge).
  4. Validasi pasca-salin (ukuran file & struktur JSON inference_config).
  5. Log visual terperinci dengan standard ASCII untuk mencegah UnicodeEncodeError.
"""

import os
import shutil
import json
import glob as _glob
from datetime import datetime

# ==================== CONFIGURATION ====================
SOURCE_REPO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
TARGET_REPO_DIR = r"E:\Widyawardhana_Capital\swint_tradev2"

# -------------------------------------------------------
# ACTIVE STACK — ubah ini saat ganti model
# Nama harus cocok dengan direktori di models/runs/
# -------------------------------------------------------
ACTIVE_STACK = {
    "lgbm":     "opt2_plus_trend_18coin_iso37f",       # lgbm37f 18-koin — takedown relative_strength_z (2026-07-10), gap live-vs-riset tak terjelaskan; OOS PF +8-10% vs opt2_plus_trend_18coin di semua kombinasi entry/regime_disable
    "lstm":     "ic32_lstm_regime_v2",                 # momentum LSTM tetap DISABLED (cascade)
    "guardian": "guard_opt2_plus_trend_hmm_18coin",    # guard28f_18coin — no cross-model, reuse (tidak retrain, tidak butuh relative_strength_z)
}
# Rollback cepat: kembalikan lgbm ke "opt2_plus_trend_18coin" (38f, pre-takedown).
# Catatan ic32_regime_v3: daily LSTM `ic32_daily_lstm_17f_s30` TIDAK di-deploy sbg
# model live (Opsi B). p_bull dihitung sisi-riset -> models/daily_pbull.parquet -> scp.
# Live baca artefak via daily_pbull_service.py. Lihat tools/compute_daily_pbull.py.

# Derived filenames — jangan edit manual, ikut ACTIVE_STACK
def _named(component, run_id, ext):
    return f"models/{component}_{run_id}{ext}"

_L  = ACTIVE_STACK["lgbm"]
_LS = ACTIVE_STACK["lstm"]
_G  = ACTIVE_STACK["guardian"]

# File model, scaler, dan code yang akan disalin (source relative to SOURCE_REPO_DIR, target relative to TARGET_REPO_DIR)
DEPLOY_MAPPING = {
    # 1. Models & Scalers — named by run_id agar traceable
    f"models/runs/{_L}/lgbm.pkl":                        _named("lgbm", _L, ".pkl"),
    f"models/runs/{_LS}/lstm_momentum.pt":               _named("lstm", _LS, ".pt"),
    f"models/runs/{_LS}/lstm_momentum_scaler.pkl":       _named("scaler_lstm", _LS, ".pkl"),
    f"models/runs/{_G}/guardian.pkl":                    _named("guard", _G, ".pkl"),
    f"models/runs/{_G}/guardian_scaler.pkl":             _named("scaler_guard", _G, ".pkl"),

    # 2. Feature files — named by run_id
    f"models/runs/{_L}/features.json":                   _named("feats_lgbm", _L, ".json"),
    f"models/runs/{_LS}/lstm_v4_selected_features.json": _named("feats_lstm", _LS, ".json"),
    f"models/runs/{_G}/guardian_features.json":          _named("feats_guard", _G, ".json"),

    # 3. Inference config & misc
    "models/training_feature_standards.json": "models/training_feature_standards.json",
    "models/inference_config.json": "models/inference_config.json",

    # 3b. ic32_regime_v3 — artefak p_bull harian (dihitung sisi-riset, dibaca live)
    "models/daily_pbull.parquet": "models/daily_pbull.parquet",

    # 3c. Scorecard OOF/OOS untuk menu Models (ringkasan di inference_config.scorecard,
    # detail per-trade OOS di file ini)
    f"models/runs/{_G}/oos_holdout_trades.json": "models/oos_holdout_trades.json",
    
    # 3. Communication Bridge Contract
    "MODEL_DEPLOYMENT_BRIDGE.md": "MODEL_DEPLOYMENT_BRIDGE.md",

    # 4. Core Feature Engineering & Model Architecture Code
    "core/features.py": "core/features.py",
    "core/models.py": "core/models.py",
    "core/utils.py": "core/utils.py",
    "core/regime.py": "core/regime.py",
    "core/cascade_utils.py": "core/cascade_utils.py",

    # 5. Positioning Data Mining (momentum model Phase 4)
    "pipeline/01c_fetch_positioning.py": "pipeline/01c_fetch_positioning.py",
    "pipeline/_wrapper.py": "pipeline/_wrapper.py",
    "pipeline/_bootstrap.py": "pipeline/_bootstrap.py",
    "pipeline/data/core/fetch_positioning.py": "pipeline/data/core/fetch_positioning.py",

    # 6. Production Guardian service (continuation_v1 feature builder)
    "tools/production/guardian_service.py": "app/services/guardian_service.py",
}

# 6. Per-symbol HMM models (dynamic — add all found *.pkl in models/hmm/)
for _fp in _glob.glob(os.path.join(SOURCE_REPO_DIR, "models", "hmm", "*_hmm.pkl")):
    _rel = "models/hmm/" + os.path.basename(_fp)
    DEPLOY_MAPPING[_rel] = _rel

# ---- Key operasional milik operator/VPS (di-set via UI atau fitur web-app-side) ----
# Saat menyalin inference_config.json, nilai key ini DIPRESERVE dari config target
# lama — deploy model TIDAK menimpanya. Mencegah:
#   * risk.modal_per_trade / leverage  ter-reset ke default Riset (insiden modal=10)
#   * rr_gate.sl_trigger_mode          ikut terhapus (fitur exit sisi web-app)
#   * limit_exit / cooldown           terhapus — kode & keputusan approve HANYA ada
#                                      di swint_tradev2 (paper_trading.py, signal_filter.py),
#                                      Riset tidak reproduce, jadi WAJIB dipreserve utuh
#                                      (insiden ditemukan 2026-07-13: config Riset ketinggalan,
#                                      nyaris ke-deploy ulang tanpa 2 fitur ini).
# Format: dotted path. Hanya dipreserve bila path-nya ADA di config target.
PRESERVE_KEYS = [
    "risk.modal_per_trade",
    "risk.leverage_recommended",
    "risk.max_open_positions",
    "risk.daily_loss_limit",
    "rr_gate.sl_trigger_mode",
    "limit_exit",
    "cooldown",
]

# File yang di-merge (bukan copy buta) — dikecualikan dari cek kesetaraan ukuran.
MERGE_FILES = {"models/inference_config.json"}


def _get_nested(d, dotted):
    cur = d
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return (False, None)
        cur = cur[part]
    return (True, cur)


def _set_nested(d, dotted, value):
    parts = dotted.split(".")
    cur = d
    for part in parts[:-1]:
        cur = cur.setdefault(part, {})
    cur[parts[-1]] = value


def _build_models_section():
    """Buat `models` section di inference_config berdasarkan ACTIVE_STACK."""
    L, LS, G = _L, _LS, _G
    return {
        "lgbm":             f"lgbm_{L}.pkl",
        "lgbm_features":    f"feats_lgbm_{L}.json",
        "lstm":             f"lstm_{LS}.pt",
        "lstm_features":    f"feats_lstm_{LS}.json",
        "lstm_scaler":      f"scaler_lstm_{LS}.pkl",
        "guardian":         f"guard_{G}.pkl",
        "guardian_features": f"feats_guard_{G}.json",
        "guardian_scaler":  f"scaler_guard_{G}.pkl",
    }


def update_swint_registry():
    """Auto-update model_registry.json di swint berdasarkan ACTIVE_STACK + inference_config.

    Dipanggil setelah copy_files() agar registry selalu sinkron dengan model yang di-deploy.
    Format output: {"versions": [...]} dengan entry baru sebagai active, entry lama archived.
    """
    reg_path = os.path.join(TARGET_REPO_DIR, "models", "model_registry.json")
    ic_path  = os.path.join(TARGET_REPO_DIR, "models", "inference_config.json")

    with open(ic_path, encoding="utf-8") as f:
        ic = json.load(f)

    model_version = ic.get("model_version", _L)
    trained_at    = datetime.now().strftime("%Y-%m-%dT%H:%M:%S+00:00")
    lstm_on       = ic.get("cascade", {}).get("lstm_confirmation_enabled", True)

    # Baca registry lama, fallback ke struktur kosong
    data = {"versions": []}
    if os.path.exists(reg_path):
        try:
            with open(reg_path, encoding="utf-8") as f:
                raw = json.load(f)
            # Dukung dua format: {"versions":[...]} atau flat {"active":...}
            if isinstance(raw.get("versions"), list):
                data = raw
        except (json.JSONDecodeError, OSError):
            pass

    # Archive semua entry active lama
    for v in data["versions"]:
        if isinstance(v, dict) and v.get("status") == "active":
            v["status"] = "archived"

    models_sec = _build_models_section()
    # Hitung n_features dari features file yang sudah tersalin
    n_feat = 0
    feat_path = os.path.join(TARGET_REPO_DIR, "models", models_sec["lgbm_features"])
    if os.path.exists(feat_path):
        try:
            with open(feat_path, encoding="utf-8") as f:
                n_feat = len(json.load(f))
        except Exception:
            pass

    new_entry = {
        "run_id":     model_version,
        "model_type": model_version,
        "status":     "active",
        "n_features": n_feat,
        "version":    model_version,
        "trained_at": trained_at,
        "note": (
            f"Deployed via deploy_model.py | "
            f"LGBM={_L} LSTM={_LS}(enabled={lstm_on}) Guardian={_G}"
        ),
        "paths": {
            "lgbm":              f"models/{models_sec['lgbm']}",
            "lstm":              f"models/{models_sec['lstm']}",
            "scaler":            f"models/{models_sec['lstm_scaler']}",
            "guardian":          f"models/{models_sec['guardian']}",
            "guardian_scaler":   f"models/{models_sec['guardian_scaler']}",
            "lgbm_features":     f"models/{models_sec['lgbm_features']}",
            "lstm_features":     f"models/{models_sec['lstm_features']}",
            "guardian_features": f"models/{models_sec['guardian_features']}",
            "inference_config":  "models/inference_config.json",
        },
    }

    data["versions"].insert(0, new_entry)

    with open(reg_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")

    print(f"     [REGISTRY] model_registry.json diupdate: {model_version} -> active")


def merge_inference_config(src_path, tgt_path):
    """Salin config dari source, tapi preserve PRESERVE_KEYS dari target lama.

    Memastikan parameter operasional (modal/leverage UI, sl_trigger_mode) tidak
    ter-reset saat deploy model baru. Ditulis dgn indent=2, ensure_ascii=False
    agar konsisten dgn gaya source (tanpa escape unicode \\u2014).
    """
    with open(src_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    preserved = []
    if os.path.exists(tgt_path):
        try:
            with open(tgt_path, "r", encoding="utf-8") as f:
                old = json.load(f)
            for key in PRESERVE_KEYS:
                found, val = _get_nested(old, key)
                if found:
                    _, src_val = _get_nested(cfg, key)
                    _set_nested(cfg, key, val)
                    if src_val != val:
                        preserved.append(f"{key}={val} (source={src_val})")
                    else:
                        preserved.append(f"{key}={val}")
        except (json.JSONDecodeError, OSError) as e:
            print(f"     [WARNING] Gagal baca config target untuk preserve: {e}")

    # Paksa models section ikut ACTIVE_STACK agar inference_config selalu konsisten
    cfg["models"] = _build_models_section()
    print(f"     [STACK] models section diupdate: {cfg['models']}")

    # Paksa filter chain selaras config.py (jangan ikut UI lama / template enabled=true)
    try:
        import sys
        if SOURCE_REPO_DIR not in sys.path:
            sys.path.insert(0, SOURCE_REPO_DIR)
        import config as _cfg
        struct = cfg.setdefault("structural_filter", {})
        struct["enabled"] = bool(getattr(_cfg, "TP_SL_STRUCTURAL_FILTER", False))
        if not struct["enabled"]:
            struct.setdefault(
                "note",
                "DIMATIKAN — enforced deploy_model.py dari config.py TP_SL_STRUCTURAL_FILTER.",
            )
        print(f"     [FILTER] structural_filter.enabled={struct['enabled']} (config.py)")
    except Exception as e:
        print(f"     [WARNING] Gagal sync structural_filter dari config.py: {e}")

    with open(tgt_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)
        f.write("\n")

    if preserved:
        print("     [PRESERVE] Nilai operasional dipertahankan dari config target:")
        for p in preserved:
            print(f"                - {p}")
    else:
        print("     [PRESERVE] Tidak ada key operasional di target (deploy bersih).")
# =======================================================

def print_banner():
    banner = """
=============================================================
 SINKRONISASI LINTAS REPO: RISET -> PRODUCTION (swint_tradev2)
=============================================================
    """
    print(banner)

def validate_paths():
    print("[1/5] Memverifikasi struktur repositori...")
    
    # Validasi Source
    if not os.path.exists(SOURCE_REPO_DIR):
        print(f"[ERROR] Repositori Riset asal tidak ditemukan di {SOURCE_REPO_DIR}")
        return False
    print(f"   - [OK] Source Repo found at: {SOURCE_REPO_DIR}")
    
    # Validasi Target
    if not os.path.exists(TARGET_REPO_DIR):
        print(f"[ERROR] Repositori Produksi target tidak ditemukan di {TARGET_REPO_DIR}")
        print("   Pastikan path E:\\Widyawardhana_Capital\\swint_tradev2 sudah benar.")
        return False
    print(f"   - [OK] Target Repo found at: {TARGET_REPO_DIR}")
    
    # Validasi keberadaan file source
    missing_sources = []
    for rel_src in DEPLOY_MAPPING.keys():
        src_path = os.path.join(SOURCE_REPO_DIR, rel_src)
        if not os.path.exists(src_path):
            missing_sources.append(rel_src)
            
    if missing_sources:
        print("[ERROR] Beberapa file sumber di repositori riset tidak ditemukan:")
        for missing in missing_sources:
            print(f"   - {missing}")
        return False
        
    print("   - [OK] Semua file sumber valid dan siap disalin.")
    return True

def create_backups():
    print("\n[2/5] Menjalankan Backup Otomatis di Target...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir_name = f"backup_{timestamp}"
    backup_dir = os.path.join(TARGET_REPO_DIR, "models", "backups", backup_dir_name)
    
    # Pastikan direktori backup dibuat
    os.makedirs(backup_dir, exist_ok=True)
    print(f"   - Membuat folder backup di: {backup_dir}")
    
    backed_up_count = 0
    for rel_tgt in DEPLOY_MAPPING.values():
        tgt_path = os.path.join(TARGET_REPO_DIR, rel_tgt)
        if os.path.exists(tgt_path):
            # Salin ke folder backup
            filename = os.path.basename(rel_tgt)
            # Jika itu bridge di root, atau file config
            if rel_tgt.endswith(".md"):
                backup_file_path = os.path.join(backup_dir, f"ROOT_{filename}")
            else:
                backup_file_path = os.path.join(backup_dir, filename)
                
            shutil.copy2(tgt_path, backup_file_path)
            print(f"     [+] Backed up: {rel_tgt} -> {os.path.basename(backup_file_path)}")
            backed_up_count += 1
            
    if backed_up_count == 0:
        print("   - [INFO] Tidak ada file target yang ada sebelumnya. Backup dilewati.")
    else:
        print(f"   - [OK] Berhasil mencadangkan {backed_up_count} file lama ke target backup.")
    
    return backup_dir

def copy_files():
    print("\n[3/5] Melakukan Sinkronisasi File...")
    copied_files = []
    
    for rel_src, rel_tgt in DEPLOY_MAPPING.items():
        src_path = os.path.join(SOURCE_REPO_DIR, rel_src)
        tgt_path = os.path.join(TARGET_REPO_DIR, rel_tgt)
        
        # Buat subdirektori target jika belum ada (misal models/)
        os.makedirs(os.path.dirname(tgt_path), exist_ok=True)

        if rel_tgt in MERGE_FILES:
            # Merge: salin config baru tapi preserve key operasional dari target lama
            merge_inference_config(src_path, tgt_path)
            print(f"     [~>] Merged: {rel_src} ===> {rel_tgt}")
        else:
            # Salin file beserta metadata
            shutil.copy2(src_path, tgt_path)
            print(f"     [->] Copied: {rel_src} ===> {rel_tgt}")
        copied_files.append(tgt_path)
        
    print(f"   - [OK] Berhasil menyalin {len(copied_files)} file ke produksi.")
    return copied_files

def validate_deployment():
    print("\n[4/5] Memulai Validasi Pasca-Penyalinan...")
    
    # 1. Validasi ukuran file cocok
    size_checks = True
    for rel_src, rel_tgt in DEPLOY_MAPPING.items():
        # File merge sengaja berbeda dari source (preserve key operasional) —
        # validasi strukturnya ditangani terpisah di bawah, bukan via ukuran.
        if rel_tgt in MERGE_FILES:
            continue

        src_path = os.path.join(SOURCE_REPO_DIR, rel_src)
        tgt_path = os.path.join(TARGET_REPO_DIR, rel_tgt)

        src_size = os.path.getsize(src_path)
        tgt_size = os.path.getsize(tgt_path)

        if src_size != tgt_size:
            print(f"   [WARNING] Mismatch Ukuran File: {rel_tgt} (Source: {src_size}B vs Target: {tgt_size}B)")
            size_checks = False
            
    if size_checks:
        print("   - [OK] Ukuran seluruh file 100% cocok (Integritas Data Terjamin).")
    else:
        return False
        
    # 2. Validasi struktur JSON inference_config.json
    tgt_config_path = os.path.join(TARGET_REPO_DIR, "models/inference_config.json")
    try:
        with open(tgt_config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
            
        required_keys = ["model_version", "inference", "tp_sl", "cascade"]
        missing_keys = [k for k in required_keys if k not in config]
        
        if missing_keys:
            print(f"   [ERROR] Validasi JSON Gagal: Key berikut tidak ditemukan di target config: {missing_keys}")
            return False
            
        # Tampilkan ringkasan konfigurasi target untuk verifikasi visual
        print("   - [OK] Struktur file JSON `inference_config.json` valid.")
        print("\nRingkasan Parameter Aktif Terverifikasi:")
        print(f"     * Model Version           : {config.get('model_version')}")
        print(f"     * Confidence Threshold    : {config['inference'].get('confidence_threshold_entry')}")
        print(f"     * Sequence Length (LSTM)   : {config['inference'].get('seq_len')}")
        print(f"     * LSTM Adjust Mode        : {config['cascade'].get('lstm_adjust_mode')}")
        print(f"     * LSTM Opposite Penalty   : {config['cascade'].get('lstm_adjust_opposite_pen')}")
        print(f"     * TP/SL Max SL ATR        : {config['tp_sl'].get('max_sl_atr')}")
        
    except Exception as e:
        print(f"   [ERROR] Gagal memvalidasi JSON `inference_config.json`: {str(e)}")
        return False
        
    return True

def main():
    print_banner()
    
    start_time = datetime.now()
    
    # Langkah 1: Validasi awal
    if not validate_paths():
        print("\n[ERROR] SINKRONISASI DIBATALKAN: Validasi awal gagal.")
        return
        
    # Langkah 2: Buat backup
    backup_path = create_backups()
    
    # Langkah 3: Salin file
    try:
        copy_files()
    except Exception as e:
        print(f"\n[ERROR] Gagal saat melakukan penyalinan file: {str(e)}")
        print("[!] Memulihkan dari backup sangat disarankan jika file dalam keadaan rusak.")
        return

    # Langkah 3b: Update model_registry.json di swint (agar signal page tampil versi benar)
    try:
        update_swint_registry()
    except Exception as e:
        print(f"     [WARNING] Gagal update registry: {e} — tidak fatal, lanjut.")

    # Langkah 4: Validasi pasca-salin
    if validate_deployment():
        duration = (datetime.now() - start_time).total_seconds()
        print(f"\n[5/5] SINKRONISASI SUKSES 100%! (Durasi: {duration:.2f} detik)")
        print("-------------------------------------------------------------")
        print(f"   Note: Seluruh model & parameter terbaru Cascade v3")
        print(f"         serta MODEL_DEPLOYMENT_BRIDGE.md sudah aktif di swint_tradev2.")
        print("   Lokasi backup jika diperlukan pemulihan:")
        print(f"      -> {backup_path}")
        print("=============================================================")
    else:
        print("\n[WARNING] Validasi Pasca-Penyalinan bermasalah. Periksa log di atas.")

if __name__ == "__main__":
    main()
