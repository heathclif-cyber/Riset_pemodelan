#!/bin/bash
# runpod_setup.sh — Setup environment di RunPod (jalankan satu kali setelah upload)
set -euo pipefail

WORKDIR=/workspace/Riset_pemodelan
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # .../runpod/
cd "$WORKDIR"

echo ""
echo "════════════════════════════════════════════════"
echo "  RunPod Setup — Riset Pemodelan LSTM"
echo "════════════════════════════════════════════════"

# ── 1. Cek GPU ────────────────────────────────────────────────────────────────
echo ""
echo "[1/4] Cek GPU..."
if command -v nvidia-smi &>/dev/null; then
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader \
        | awk -F',' '{printf "  GPU    : %s\n  VRAM   : %s\n  Driver : %s\n", $1, $2, $3}'
else
    echo "  WARNING: nvidia-smi tidak tersedia"
fi

# ── 2. Cek PyTorch + CUDA ─────────────────────────────────────────────────────
echo ""
echo "[2/4] Cek PyTorch CUDA..."

CUDA_OK=0
python3 - <<'PYCHECK' && CUDA_OK=1 || true
import torch
assert torch.cuda.is_available(), "CUDA tidak tersedia"
print(f"  PyTorch : {torch.__version__}")
print(f"  CUDA    : {torch.version.cuda}")
print(f"  GPU     : {torch.cuda.get_device_name(0)}")
print(f"  VRAM    : {torch.cuda.get_device_properties(0).total_memory // 1024**3} GB")
PYCHECK

# ── 3. Install dependencies ───────────────────────────────────────────────────
echo ""
echo "[3/4] Install dependencies..."

REQS="$SCRIPT_DIR/requirements_runpod.txt"

if [ "$CUDA_OK" = "1" ]; then
    echo "  PyTorch CUDA sudah ada — skip reinstall torch"
    grep -vE "^torch($|[^-])" "$REQS" | pip install -r /dev/stdin -q --no-warn-script-location
else
    echo "  PyTorch CUDA belum ada — install torch + cu121..."
    pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu121 -q
    grep -vE "^torch($|[^-])" "$REQS" | pip install -r /dev/stdin -q --no-warn-script-location
fi

echo "  Dependencies OK"

# ── 4. Verifikasi final ────────────────────────────────────────────────────────
echo ""
echo "[4/4] Verifikasi project..."

python3 - <<'PYVERIFY'
import sys, torch
from pathlib import Path
sys.path.insert(0, "/workspace/Riset_pemodelan")

ok = True

print(f"  Python  : {sys.version.split()[0]}")
print(f"  PyTorch : {torch.__version__}")
cuda_ok = torch.cuda.is_available()
print(f"  CUDA    : {'OK — ' + torch.cuda.get_device_name(0) if cuda_ok else 'TIDAK TERSEDIA (akan pakai CPU)'}")
if not cuda_ok:
    print("  WARNING: Training akan sangat lambat tanpa GPU!")
    ok = False

try:
    from config import LABEL_DIR, TRAINING_COINS, MODEL_DIR
    from core.utils import setup_logger
    from core.models import TradingLSTM
    print("  Import  : OK")
except Exception as e:
    print(f"  Import  : GAGAL — {e}")
    ok = False

try:
    missing = [s for s in TRAINING_COINS if not (LABEL_DIR / f"{s}_features_v3.parquet").exists()]
    if missing:
        print(f"  Data    : WARNING — tidak ada untuk: {missing}")
    else:
        sizes = sum((LABEL_DIR / f"{s}_features_v3.parquet").stat().st_size for s in TRAINING_COINS)
        print(f"  Data    : OK ({len(TRAINING_COINS)} coins, {sizes/1024**2:.0f} MB total)")
except Exception as e:
    print(f"  Data    : ERROR — {e}")
    ok = False

for d in [MODEL_DIR, Path("/workspace/Riset_pemodelan/logs")]:
    Path(d).mkdir(parents=True, exist_ok=True)
print("  Dirs    : OK (models/, logs/)")

sys.exit(0 if ok else 1)
PYVERIFY

echo ""
echo "════════════════════════════════════════════════"
echo "  Setup selesai!"
echo ""
echo "  Mulai training:"
echo "    bash $WORKDIR/runpod/runpod_train.sh"
echo ""
echo "  Training semua 20 koin:"
echo "    bash $WORKDIR/runpod/runpod_train.sh --all"
echo "════════════════════════════════════════════════"
echo ""
