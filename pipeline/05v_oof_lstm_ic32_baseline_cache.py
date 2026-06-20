"""
Cache LSTM proba from production lstm_best.pt aligned to ic32 LGBM OOF bars.

Fixed model inference (same as live) -- NOT purged CV retrain.
Output: models/runs/ic32_regime_v1/oof_lstm_baseline_predictions.parquet

Usage:
  python pipeline/05v_oof_lstm_ic32_baseline_cache.py
"""
import json
import sys
import warnings
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from pipeline.backtest_utils import get_lstm_proba
from core.models import load_lstm
from core.utils import setup_logger, ensure_utc_index
from config import ALL_COINS, LABEL_DIR, MODEL_DIR, TRAIN_CUTOFF_DATE, LABEL_MAP

logger = setup_logger("05v_lstm_baseline_cache")
RUN_DIR = MODEL_DIR / "runs" / "ic32_regime_v1"
OOF_PATH = RUN_DIR / "oof_predictions.parquet"
OUT_PATH = RUN_DIR / "oof_lstm_baseline_predictions.parquet"
LSTM_PROD_N_FEAT = 11


def main():
    if not OOF_PATH.exists():
        raise FileNotFoundError(f"Missing {OOF_PATH} -- run 04_train_lgbm_ic32_genuine_oof first")

    oof_all = pd.read_parquet(OOF_PATH)
    if not isinstance(oof_all.index, pd.DatetimeIndex):
        oof_all.index = pd.to_datetime(oof_all.index, utc=True)

    with open(MODEL_DIR / "feature_cols_lstm_temporal.json", encoding="utf-8") as f:
        lstm_feat_cols = json.load(f)[:LSTM_PROD_N_FEAT]

    lstm_model = load_lstm(MODEL_DIR / "lstm_best.pt", device="cpu")
    lstm_scaler = joblib.load(MODEL_DIR / "lstm_scaler.pkl")

    rows = []
    total_bars = 0
    for sym in ALL_COINS:
        path = LABEL_DIR / f"{sym}_features_v3.parquet"
        if not path.exists():
            continue
        df = pd.read_parquet(path)
        df = ensure_utc_index(df).sort_index()
        df = df[df.index < TRAIN_CUTOFF_DATE]
        df = df[df["label"].astype(str).isin(LABEL_MAP)]
        if df.empty:
            continue

        oof_sym = oof_all[oof_all["coin"] == sym]
        if oof_sym.empty:
            continue
        merged = df.join(oof_sym[["has_oof"]], how="left")
        has_oof = merged["has_oof"].fillna(False).values.astype(bool)
        if has_oof.sum() < 10:
            continue

        n = len(df)
        X = np.zeros((n, len(lstm_feat_cols)), dtype=np.float64)
        for idx, col in enumerate(lstm_feat_cols):
            if col in df.columns:
                X[:, idx] = df[col].ffill().fillna(0).values.astype(np.float64)

        lstm_proba = get_lstm_proba(lstm_model, lstm_scaler, X, n)
        vol_spike = (
            df["vol_spike_zscore"].fillna(0).values.astype(np.float32)
            if "vol_spike_zscore" in df.columns
            else np.zeros(n, dtype=np.float32)
        )

        for i in range(n):
            if not has_oof[i]:
                continue
            rows.append({
                "ts": df.index[i],
                "coin": sym,
                "p0": float(lstm_proba[i, 0]),
                "p1": float(lstm_proba[i, 1]),
                "p2": float(lstm_proba[i, 2]),
                "vol_spike": float(vol_spike[i]),
                "has_oof": True,
            })
        total_bars += int(has_oof.sum())
        logger.info(f"  {sym}: {has_oof.sum():,} OOF bars cached")

    if not rows:
        raise RuntimeError("No LSTM OOF rows generated")

    out = pd.DataFrame(rows)
    out["ts"] = pd.to_datetime(out["ts"], utc=True)
    out = out.set_index("ts").sort_index()
    out.to_parquet(OUT_PATH)

    meta = {
        "created": datetime.now().isoformat(),
        "model": "lstm_best.pt",
        "n_features": LSTM_PROD_N_FEAT,
        "n_rows": len(out),
        "n_coins": out["coin"].nunique(),
        "methodology": "fixed_model_inference_aligned_to_lgbm_oof_bars",
        "note": "Same as production -- not purged CV LSTM OOF",
    }
    with open(RUN_DIR / "oof_lstm_baseline_cache_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"\nCached {len(out):,} LSTM rows ({out['coin'].nunique()} coins)")
    print(f"Saved: {OUT_PATH}")


if __name__ == "__main__":
    main()