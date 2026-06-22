#!/bin/bash
# runpod/setup_runpod.sh
# Jalankan SEKALI setelah pod pertama kali aktif
# Usage: bash runpod/setup_runpod.sh

set -e
echo "============================================"
echo " Setting up RunPod environment..."
echo "============================================"

# Update pip
pip install --upgrade pip

# Install PyTorch (CUDA 12.1 — sesuai dengan kebanyakan RunPod GPU)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install dependencies utama
pip install \
  pandas>=2.0 \
  numpy>=1.26 \
  pyarrow>=14.0 \
  lightgbm>=4.0 \
  scikit-learn>=1.4 \
  joblib>=1.3 \
  hmmlearn>=0.3.3

echo ""
echo "Setup selesai!"
python -c "import torch; print(f'PyTorch: {torch.__version__} | CUDA: {torch.cuda.is_available()} | GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"
