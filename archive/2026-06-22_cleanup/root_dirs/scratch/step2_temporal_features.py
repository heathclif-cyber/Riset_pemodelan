"""
scratch/step2_temporal_features.py
Compute temporal trajectory features untuk setiap OOF bar.

Hypothesis: LGBM melihat SNAPSHOT di bar t. LSTM melihat 32-bar trajectory.
Jika trajectory features punya marginal IC >= 0.02 vs LGBM conf, LSTM worth building.

Trajectory features yang dihitung per coin per bar:
  - slope_k  : linear regression slope dari feature selama k bar terakhir
  - delta_k  : feat[t] - feat[t-k]
  - accel    : slope(k=8) - slope(k=16) — "acceleration"
  - z_k      : (feat[t] - mean(k)) / std(k)  — z-score posisi relatif

Target features dari TB_V3_FEATURES yang relevan untuk trajectory:
  funding_rate, cvd_slope_h4, ofi_h4_delta, cvd_div_h4,
  cvd_momentum_adv, atr_percentile_h1, stochrsi_k

Output: data/meta_labels/tb_v3_temporal_feats.parquet
"""
import sys, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import config

OOF_PATH    = ROOT / "data" / "meta_labels" / "tb_lgbm_v3_oof.parquet"
OUT_PATH    = ROOT / "data" / "meta_labels" / "tb_v3_temporal_feats.parquet"
LABEL_DIR   = Path(config.LABEL_DIR)

# Auto-detect dari features_v3.parquet — exclude non-continuous
EXCLUDE_COLS = {
    # Raw OHLCV prices (absolute levels not useful as trajectory source)
    "open", "high", "low", "close", "volume",
    # Target/label
    "label", "tb_label",
    # Categorical
    "hmm_regime", "hmm_regime_enc", "wyckoff_phase",
    "market_session", "vol_regime", "h4_trend",
    # Cyclic time (no meaningful slope)
    "dow_cos", "dow_sin", "hour_cos", "hour_sin",
    # Absolute price levels
    "h4_swing_high", "h4_swing_low", "PDH", "PDL",
    "PWH", "PWL", "Fib_618", "Fib_786",
    "POC", "VAH", "VAL",
    # Binary event flags (0/1 — slope meaningless for single bar)
    "CHoCH", "MSB_BOS", "SFP_sweep", "FVG_up", "FVG_down",
    "spring_upthrust", "ultra_high_vol", "hidden_divergence",
    "rsi_divergence", "no_demand", "no_supply",
    # Coin identifier
    "coin",
}
WINDOWS = [4, 8, 16, 32]


def banner(t, w=65):
    print(f"\n{'='*w}\n  {t}\n{'='*w}")


def fast_rolling_slope(arr: np.ndarray, w: int) -> np.ndarray:
    """
    Vectorized OLS slope via sliding window (numpy stride tricks).
    O(N*w) but fully vectorized — ~100x faster than pandas apply(linregress).
    """
    from numpy.lib.stride_tricks import sliding_window_view
    n = len(arr)
    x = np.arange(w, dtype=float) - (w - 1) / 2.0   # centered x
    denom = float((x ** 2).sum())
    if denom == 0:
        return np.full(n, np.nan)

    # Pad front with NaN so output length == input length
    pad = np.full(w - 1, np.nan)
    wins = sliding_window_view(arr, w)  # shape (n-w+1, w)

    # Mask rows with any NaN
    valid = ~np.isnan(wins).any(axis=1)
    slopes = np.full(len(wins), np.nan)
    if valid.any():
        slopes[valid] = (wins[valid] * x).sum(axis=1) / denom

    return np.concatenate([pad, slopes])


def compute_traj_features(df: pd.DataFrame, feat_cols: list) -> pd.DataFrame:
    """Per-coin: tambahkan slope, delta, z-score per window."""
    result_cols = {}
    for feat in feat_cols:
        if feat not in df.columns:
            continue
        arr = df[feat].ffill().fillna(0).values.astype(float)
        s   = pd.Series(arr, index=df.index)
        for w in WINDOWS:
            roll_mean = s.rolling(w, min_periods=max(2, w // 2)).mean()
            roll_std  = s.rolling(w, min_periods=max(2, w // 2)).std()
            result_cols[f"{feat}_delta{w}"] = arr - np.concatenate([np.full(w, np.nan), arr[:-w]])
            result_cols[f"{feat}_slope{w}"] = fast_rolling_slope(arr, w)
            result_cols[f"{feat}_z{w}"]     = ((arr - roll_mean.values) / (roll_std.values + 1e-10))
        result_cols[f"{feat}_accel"] = fast_rolling_slope(arr, 4) - fast_rolling_slope(arr, 16)

    out = pd.DataFrame(result_cols, index=df.index)
    return out


banner("Step 2: Temporal Trajectory Features")

# Load OOF
print("\nLoading OOF predictions...")
oof = pd.read_parquet(OOF_PATH)
oof.index = pd.to_datetime(oof.index, utc=True)
print(f"  OOF rows: {len(oof):,} | Coins: {oof['coin'].nunique()}")

# Load raw features per coin, compute trajectory, join with OOF
print("\nComputing trajectory features per coin...")
result_pieces = []
TRAJ_SOURCES = None   # will be detected from first coin

for sym in oof["coin"].unique():
    fpath = LABEL_DIR / f"{sym}_features_v3.parquet"
    if not fpath.exists():
        print(f"  [SKIP] {sym} — features_v3.parquet not found")
        continue

    df = pd.read_parquet(fpath)
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, utc=True)
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")

    # Auto-detect continuous numeric features on first coin
    if TRAJ_SOURCES is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        TRAJ_SOURCES = [c for c in numeric_cols if c not in EXCLUDE_COLS]
        print(f"\n  Auto-detected {len(TRAJ_SOURCES)} continuous numeric source features")
        print(f"  (from {len(df.columns)} total, excluded {len(df.columns) - len(TRAJ_SOURCES)})")
        print(f"  Trajectory cols akan dibuat: {len(TRAJ_SOURCES) * (len(WINDOWS)*3 + 1):,}\n")

    # Compute trajectory on full coin history (for clean rolling windows)
    traj = compute_traj_features(df, TRAJ_SOURCES)

    # Subset to OOF timestamps only for this coin
    oof_coin = oof[oof["coin"] == sym].copy()
    traj_sub = traj.reindex(oof_coin.index)

    # Concat OOF columns + trajectory columns side-by-side (same index)
    coin_merged = pd.concat([oof_coin, traj_sub], axis=1)
    result_pieces.append(coin_merged)

    n_traj_cols = len(traj.columns)
    print(f"  {sym}: {len(oof_coin)} OOF bars, {n_traj_cols} traj features")

merged = pd.concat(result_pieces).sort_index()
traj_cols_final = [c for c in merged.columns
                   if any(c.startswith(f) for f in TRAJ_SOURCES)]
print(f"\n  Merged rows : {len(merged):,}")
print(f"  Traj cols   : {len(traj_cols_final)}")

# Save
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
merged.to_parquet(OUT_PATH)

banner("Step 2 Complete")
print(f"\n  Saved -> {OUT_PATH}")
traj_cols = [c for c in merged.columns
             if any(c.startswith(f) for f in TRAJ_SOURCES)]
print(f"  Trajectory features: {len(traj_cols)}")
print(f"  Sample: {traj_cols[:8]}")
print(f"\n  Next: python scratch/step3_marginal_ic_test.py")
