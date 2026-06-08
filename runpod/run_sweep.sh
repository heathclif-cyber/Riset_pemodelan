#!/bin/bash
# runpod/run_sweep.sh
# Jalankan sweep training seq_len 32 dan 128
# Usage: bash runpod/run_sweep.sh

set -e
cd /workspace/Riset_pemodelan

echo "============================================"
echo " LSTM Seq_Len Sweep (32 dan 128)"
echo " Dibandingkan dengan baseline seq_len=16"
echo "============================================"

START=$(date +%s)

for SEQ in 32 128; do
  echo ""
  echo "--------------------------------------------"
  echo " Mulai training: seq_len = $SEQ"
  echo " Waktu: $(date)"
  echo "--------------------------------------------"

  python pipeline/05_train_lstm_seq_sweep.py --seq-len $SEQ --all

  echo " Selesai: seq_len = $SEQ | $(date)"
done

END=$(date +%s)
ELAPSED=$(( (END - START) / 60 ))

echo ""
echo "============================================"
echo " SEMUA TRAINING SELESAI!"
echo " Total waktu: ${ELAPSED} menit"
echo " Hasil tersimpan di: models/runs/lstm_seq32/"
echo "                dan: models/runs/lstm_seq128/"
echo "============================================"
