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
from datetime import datetime

# ==================== CONFIGURATION ====================
SOURCE_REPO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TARGET_REPO_DIR = r"D:\Apps-Dev\swint_tradev2"

# File model, scaler, dan code yang akan disalin (source relative to SOURCE_REPO_DIR, target relative to TARGET_REPO_DIR)
DEPLOY_MAPPING = {
    # 1. Models & Scalers (Source -> Target)
    "models/lgbm_baseline.pkl": "models/lgbm_baseline.pkl",
    "models/lstm_best.pt": "models/lstm_best.pt",
    "models/lstm_scaler.pkl": "models/lstm_scaler.pkl",
    "models/guardian_best.pkl": "models/guardian_best.pkl",
    "models/guardian_scaler.pkl": "models/guardian_scaler.pkl",
    
    # 2. Configs & Features
    "models/feature_cols_v2.json": "models/feature_cols_v2.json",
    "models/guardian_feature_cols.json": "models/guardian_feature_cols.json",
    "models/inference_config.json": "models/inference_config.json",
    
    # 3. Communication Bridge Contract
    "MODEL_DEPLOYMENT_BRIDGE.md": "MODEL_DEPLOYMENT_BRIDGE.md",

    # 4. Core Feature Engineering & Model Architecture Code
    "core/features.py": "core/features.py",
    "core/models.py": "core/models.py",
    "core/utils.py": "core/utils.py",
    "core/regime.py": "core/regime.py"
}
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
        print("   Pastikan path D:\\Apps-Dev\\swint_tradev2 sudah benar.")
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
