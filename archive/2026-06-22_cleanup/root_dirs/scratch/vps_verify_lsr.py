# -*- coding: utf-8 -*-
"""Quick VPS/local verify: inference LSR must be > 0 in training_parity mode."""
import os
import sys

ROOT = sys.argv[1] if len(sys.argv) > 1 else "/home/swint/swint_tradev2"
os.chdir(ROOT)
sys.path.insert(0, ROOT)

from app.services.data_service import InferenceDataService, _get_positioning_mode

mode = _get_positioning_mode()
print(f"mode={mode}")
svc = InferenceDataService()
df = svc.prepare_latest_features("SOLUSDT", n_bars=500)
if df is None:
    print("FAIL: features None")
    sys.exit(1)
lsr = float(df["long_short_ratio"].iloc[-1])
hmm = int(df["hmm_regime_enc"].iloc[-1])
print(f"LSR={lsr:.4f} HMM={hmm}")
if lsr <= 0:
    print("FAIL: LSR <= 0")
    sys.exit(1)
print("OK")