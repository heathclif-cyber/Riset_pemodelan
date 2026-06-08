"""
scratch/compare_lstm_seq_results.py — Bandingkan hasil CV sweep seq_len LSTM
Membaca models/runs/lstm_seq{N}/lstm_cv_results.json dan baseline seq=16
dari models/lstm_cv_results.json

Jalankan setelah scp hasil dari RunPod ke lokal:
  python scratch/compare_lstm_seq_results.py
"""
import json
import sys
from pathlib import Path

ROOT = Path("d:/Apps-Dev/Riset_pemodelan")
sys.path.insert(0, str(ROOT))

MODEL_DIR = ROOT / "models"

# Konfigurasi: (label, path ke cv_results)
CONFIGS = [
    ("seq=16 (BASELINE)", MODEL_DIR / "lstm_cv_results.json"),
    ("seq=32", MODEL_DIR / "runs" / "lstm_seq32" / "lstm_cv_results.json"),
    ("seq=128", MODEL_DIR / "runs" / "lstm_seq128" / "lstm_cv_results.json"),
]

results = []
for label, path in CONFIGS:
    if not path.exists():
        print(f"[SKIP] {label}: file tidak ditemukan ({path})")
        continue
    with open(path) as f:
        data = json.load(f)
    folds = data.get("folds", [])
    f1_macros = [fold["f1_macro"] for fold in folds]
    f1_shorts  = [fold.get("f1_SHORT", 0) for fold in folds]
    f1_longs   = [fold.get("f1_LONG", 0) for fold in folds]
    results.append({
        "label": label,
        "seq_len": data.get("seq_len", 16),
        "mean_f1_macro": data.get("mean_f1_macro", 0),
        "std_f1_macro": data.get("std_f1_macro", 0),
        "best_f1": data.get("best_f1_macro", 0),
        "best_fold": data.get("best_fold", "-"),
        "mean_f1_short": sum(f1_shorts) / len(f1_shorts) if f1_shorts else 0,
        "mean_f1_long": sum(f1_longs) / len(f1_longs) if f1_longs else 0,
        "final_epochs": data.get("final_retrain_epochs", "-"),
        "n_folds": len(folds),
    })

if not results:
    print("Tidak ada hasil yang ditemukan. Pastikan training sudah selesai.")
    sys.exit(1)

print()
print("=" * 90)
print("  PERBANDINGAN HASIL SWEEP LSTM SEQ_LEN")
print("=" * 90)
print(f"{'Konfigurasi':<22} | {'Mean F1-macro':<14} | {'Best F1':<8} | {'F1-SHORT':<9} | {'F1-LONG':<8} | {'Epochs'}")
print("-" * 90)
for r in results:
    print(f"{r['label']:<22} | {r['mean_f1_macro']:.4f} +/- {r['std_f1_macro']:.4f}  | {r['best_f1']:<8.4f} | {r['mean_f1_short']:<9.4f} | {r['mean_f1_long']:<8.4f} | {r['final_epochs']}")
print("=" * 90)

# Rekomendasi
best = max(results, key=lambda x: x["mean_f1_macro"])
print(f"\nREKOMENDASI: Gunakan {best['label']} dengan Mean F1-macro = {best['mean_f1_macro']:.4f}")

# Per-fold detail
print()
for label, path in CONFIGS:
    if not path.exists():
        continue
    with open(path) as f:
        data = json.load(f)
    label_str = label
    print(f"\n--- Detail per fold: {label_str} ---")
    print(f"{'Fold':<6} | {'F1-macro':<9} | {'F1-SHORT':<9} | {'F1-FLAT':<9} | {'F1-LONG':<9} | {'Best Epoch'}")
    print("-" * 65)
    for fold in data.get("folds", []):
        print(f"{fold['fold']:<6} | {fold['f1_macro']:<9.4f} | {fold.get('f1_SHORT',0):<9.4f} | {fold.get('f1_FLAT',0):<9.4f} | {fold.get('f1_LONG',0):<9.4f} | {fold.get('best_epoch', '-')}")
