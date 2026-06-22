"""Part 3: Fit HMM fresh, cek transition matrix dan means per state."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from core.regime import _build_hmm_features, _align_states, REGIME_NAMES_4

PROC_DIR = Path("data/training/processed")

sample_coins = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]

for coin in sample_coins:
    fp = PROC_DIR / f"{coin}_clean.parquet"
    if not fp.exists():
        print(f"  {coin}: tidak ditemukan di {fp}")
        continue

    df = pd.read_parquet(fp)
    # Rename kolom 1h_ prefix jika ada
    df = df.rename(columns={
        "1h_open": "open", "1h_high": "high", "1h_low": "low",
        "1h_close": "close", "1h_volume": "volume"
    })
    if "close" not in df.columns:
        print(f"  {coin}: kolom close tidak ada, kolom: {df.columns.tolist()[:8]}")
        continue

    # Resample H1 -> H4
    ohlcv = df[["open","high","low","close","volume"]].resample("4h").agg({
        "open": "first", "high": "max", "low": "min",
        "close": "last", "volume": "sum"
    }).dropna()

    X = _build_hmm_features(ohlcv)
    model = GaussianHMM(n_components=4, covariance_type="diag",
                        n_iter=100, random_state=42, tol=1e-3)
    model.fit(X)
    state_map = _align_states(model, 4)
    raw_labels = model.predict(X)
    labels = [state_map[int(s)] for s in raw_labels]

    dist = pd.Series(labels).value_counts(normalize=True).round(3)
    diag = np.diag(model.transmat_)

    print(f"\n{coin} (H4, {len(ohlcv)} bars):")
    print(f"  Distribusi state: {dist.to_dict()}")
    print(f"  Persistence diagonal: min={diag.min():.3f}  mean={diag.mean():.3f}")
    print(f"  {'STABIL' if diag.min() >= 0.7 else 'TIDAK STABIL - state loncat-loncat'}")

    means_by_state = {}
    for raw_idx, name in state_map.items():
        means_by_state[name] = model.means_[raw_idx]

    print(f"  {'State':22} {'ret':>10} {'vol':>10} {'mom':>10} {'vol_r':>10}")
    for name in REGIME_NAMES_4:
        if name in means_by_state:
            m = means_by_state[name]
            print(f"  {name:22} {m[0]:+10.5f} {m[1]:10.5f} {m[2]:+10.5f} {m[3]:+10.5f}")
