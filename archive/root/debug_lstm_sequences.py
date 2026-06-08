"""
Debug script: Analyze why LSTM momentum detector stays at random F1.
Focus: intra-sequence variance & scaling of the 14 trajectory features.
"""
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))

from config import TRAINING_DIR

seq_dir = TRAINING_DIR / "h1_sequences"
features = [
    "h1_return", "log_ret_5", "log_ret_20",
    "volume_delta", "ofi_raw", "ofi_acceleration",
    "rsi_6", "stochrsi_k", "vwdp_smooth",
    "atr_14_h1", "vol_ratio_20", "bars_since_BOS",
    "oi_delta_pct", "btc_h1_return"
]

print("=" * 70)
print("LSTM SEQUENCE DIAGNOSIS - Intra-window dynamics (32 H1 bars)")
print("=" * 70)

coins = ["SOLUSDT", "BTCUSDT", "DOGEUSDT", "HBARUSDT", "NEARUSDT"]

for coin in coins:
    path = seq_dir / f"{coin}_seq.npz"
    if not path.exists():
        print(f"\n{coin}: FILE NOT FOUND")
        continue
    
    data = np.load(path)
    X = data["X"]          # (N, 32, 14)
    y = data["y"]
    
    print(f"\n{'='*70}")
    print(f"{coin}")
    print(f"{'='*70}")
    print(f"Total sequences: {len(y):,}")
    
    # Label distribution
    unique, counts = np.unique(y, return_counts=True)
    label_names = ["SHORT", "FLAT", "LONG"]
    print("Label distribution:")
    for u, c in zip(unique, counts):
        print(f"  {label_names[u]:6s}: {c:7,} ({c/len(y)*100:5.1f}%)")
    
    # Key diagnosis: how much does each feature actually MOVE inside one 32-bar window?
    print("\nIntra-sequence dynamics (max - min inside each 32-bar window):")
    print(f"{'Feature':<20} {'Mean Range':>14} {'Median Range':>14} {'P95 Range':>14}")
    print("-" * 65)
    
    for i, f in enumerate(features):
        window_ranges = np.max(X[:, :, i], axis=1) - np.min(X[:, :, i], axis=1)
        mean_r = np.mean(window_ranges)
        med_r = np.median(window_ranges)
        p95_r = np.percentile(window_ranges, 95)
        print(f"{f:<20} {mean_r:14.6f} {med_r:14.6f} {p95_r:14.6f}")
    
    # Also show global std (across all data)
    print("\nGlobal std (all samples, all timesteps):")
    for i, f in enumerate(features):
        global_std = np.std(X[:, :, i])
        print(f"  {f:<20}: {global_std:.6f}")

print("\n" + "=" * 70)
print("KEY INSIGHT: If 'Mean Range' inside 32 bars is tiny relative to the")
print("feature's natural scale, LSTM has almost no temporal signal to learn.")
print("=" * 70)

# ==================== SCALER & CROSS-COIN ANALYSIS ====================
print("\n\n" + "=" * 70)
print("SCALER & CROSS-COIN CONSISTENCY CHECK")
print("=" * 70)

import joblib
scaler_path = Path("models/runs/cascade_v4.3/lstm_momentum_scaler.pkl")
if scaler_path.exists():
    scaler = joblib.load(scaler_path)
    print(f"\nScaler found: {scaler_path}")
    if hasattr(scaler, "mean_"):
        print(f"  Number of features scaled: {len(scaler.mean_)}")
        print(f"  Scaler means (first 8): {scaler.mean_[:8]}")
        print(f"  Scaler scales (first 8): {scaler.scale_[:8]}")
else:
    print("\n[WARNING] No scaler found from v4.3 run")

# Check cross-coin scale differences (this is deadly for joint training)
print("\n=== Cross-coin scale mismatch (Global std comparison) ===")
print(f"{'Feature':<20} {'BTC':>12} {'SOL':>12} {'DOGE':>12} {'Ratio BTC/DOGE':>16}")
print("-" * 75)

btc_data = np.load(seq_dir / "BTCUSDT_seq.npz")["X"]
sol_data = np.load(seq_dir / "SOLUSDT_seq.npz")["X"]
doge_data = np.load(seq_dir / "DOGEUSDT_seq.npz")["X"]

for i, f in enumerate(features):
    btc_std = np.std(btc_data[:, :, i])
    sol_std = np.std(sol_data[:, :, i])
    doge_std = np.std(doge_data[:, :, i])
    ratio = btc_std / doge_std if doge_std > 0 else float("inf")
    print(f"{f:<20} {btc_std:12.4f} {sol_std:12.4f} {doge_std:12.4f} {ratio:16.1f}")

print("\n[CRITICAL] Ratio >> 1 or << 1 means the same 'feature' lives on completely")
print("different numerical scales across coins. LSTM trained jointly will waste")
print("capacity learning scale differences instead of temporal patterns.")

# Final diagnosis
print("\n\n" + "=" * 70)
print("FINAL DIAGNOSIS SUMMARY")
print("=" * 70)
print("""
ROOT CAUSES why LSTM F1 stays at random (~0.333):

1. EXTREME CROSS-COIN SCALE MISMATCH (biggest killer)
   - volume_delta, ofi_*, vwdp_smooth, atr_14_h1 have wildly different
     magnitudes between BTC vs midcap vs memecoins.
   - Even with per-fold StandardScaler, the joint distribution across
     all coins is extremely heavy-tailed and non-stationary.

2. MANY "TRAJECTORY" FEATURES ARE STILL TOO STABLE IN 32-BAR WINDOWS
   - oi_delta_pct: median range only ~0.001-0.008 in 32 hours
   - Many orderflow features have long stretches of near-zero movement
     punctuated by rare huge spikes (classic heavy-tail problem).

3. LABEL NOISE IS HIGH
   - Even with N=8/12 + magnitude filter, 40-44% FLAT means the "signal"
     (decisive momentum) is still minority and hard to separate from noise.

4. LSTM CAPACITY WASTED ON SCALE LEARNING
   - With features on 10^-5 vs 10^8 scales mixed together, the model
     spends most of its hidden state trying to normalize internally.

RECOMMENDED FIXES (in order of impact):

A. Per-coin or per-regime scaling + robust scaling (RobustScaler or
   quantile transform) instead of plain StandardScaler.

B. Remove or heavily transform the worst offenders:
   - volume_delta / ofi_raw / ofi_acceleration / vwdp_smooth
   - Replace with rank-based or log(1 + x) + clip versions, or
     z-score *relative to recent 20-50 bar history* (local normalization).

C. Consider training separate LSTMs per volatility regime or per
   market-cap bucket (BTC/ETH vs mid vs small), then ensemble.

D. If after A+B F1 still < 0.36-0.37, seriously consider abandoning
   3-class LSTM and move to a simpler binary "momentum confirmer"
   (LONG vs not-LONG) or even drop the LSTM entirely.
""")

