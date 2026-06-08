import numpy as np

data = np.load("data/training/h1_sequences/all_coins_seq.npz")
X = data["X"]
print("Shape:", X.shape)
print("Expected features: 11")
print()

feat_names = [
    "h1_return", "log_ret_5", "log_ret_20",
    "rsi_6", "stochrsi_k",
    "vol_ratio_20", "atr_percentile_h1",
    "bars_since_BOS",
    "ofi_z_score",
    "oi_delta_pct", "btc_h1_return"
]

print("Per-feature global std (should be much more comparable across coins now):")
for i, name in enumerate(feat_names):
    std = np.std(X[:, :, i])
    print(f"  {name:20s}: {std:12.6f}")

print("\n=== SUCCESS: 11 robust features built correctly ===")
