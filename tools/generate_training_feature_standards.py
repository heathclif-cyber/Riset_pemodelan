# -*- coding: utf-8 -*-
"""
Hitung distribusi fitur training ic32 dari parquet labeled.
Output: models/training_feature_standards.json (deploy ke swint untuk parity monitor).
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.utils import ensure_utc_index

TRAIN = ROOT / "data/training/labeled"
OUT = ROOT / "models/training_feature_standards.json"
FEAT = json.load(open(ROOT / "models/feature_cols_v2.json", encoding="utf-8"))

# Cara cek per fitur di live monitor
CHECK_RULES: dict[str, dict] = {
    "long_short_ratio": {"check": "strict", "max_z": 3.0, "note": "training_parity harus ~1.0"},
    "whale_retail_divergence": {"check": "strict", "max_z": 4.0},
    "hmm_regime_enc": {"check": "categorical", "valid": [0, 1, 2, 3]},
    "h4_trend": {"check": "categorical", "valid": [-1, 0, 1]},
    "rsi_h4": {"check": "bounds", "min": 0, "max": 100},
    "rsi_6": {"check": "bounds", "min": 0, "max": 100},
    "stochrsi_d": {"check": "bounds", "min": 0, "max": 100},
    "stochrsi_k": {"check": "bounds", "min": 0, "max": 100},
    "cvd": {"check": "skip", "note": "skala live beda; IC lemah"},
    "ofi_h4_delta": {"check": "skip"},
    "ofi_acceleration": {"check": "skip"},
    "cvd_div_h4": {"check": "skip"},
    "cvd_momentum_adv": {"check": "loose", "max_z": 6.0},
    # ATR-normalized; low-vol coins (e.g. TRX) spike z-score without bad data.
    "dist_liq_50x_long": {"check": "loose", "max_z": 7.0},
    "dist_liq_20x_long": {"check": "loose", "max_z": 7.0},
    "dist_liq_50x_short": {"check": "loose", "max_z": 7.0},
    "dist_liq_20x_short": {"check": "loose", "max_z": 7.0},
}
DEFAULT_CHECK = {"check": "moderate", "max_z": 4.5}


def main() -> None:
    frames = []
    for p in sorted(TRAIN.glob("*_features_v3.parquet")):
        df = ensure_utc_index(pd.read_parquet(p))
        cols = [c for c in FEAT if c in df.columns]
        if cols:
            frames.append(df[cols])
    if not frames:
        raise SystemExit(f"Tidak ada parquet di {TRAIN}")

    all_df = pd.concat(frames, axis=0, ignore_index=False)
    n_rows = len(all_df)
    n_coins = len(frames)

    features: dict[str, dict] = {}
    for f in FEAT:
        if f not in all_df.columns:
            continue
        s = all_df[f].replace([np.inf, -np.inf], np.nan).dropna()
        if s.empty:
            continue
        rule = {**DEFAULT_CHECK, **CHECK_RULES.get(f, {})}
        features[f] = {
            "mean": round(float(s.mean()), 6),
            "std": round(float(s.std()), 6),
            "p5": round(float(np.percentile(s, 5)), 6),
            "p25": round(float(np.percentile(s, 25)), 6),
            "p50": round(float(np.percentile(s, 50)), 6),
            "p75": round(float(np.percentile(s, 75)), 6),
            "p95": round(float(np.percentile(s, 95)), 6),
            "p99": round(float(np.percentile(s, 99)), 6),
            "min": round(float(s.min()), 6),
            "max": round(float(s.max()), 6),
            "n": int(len(s)),
            **rule,
        }

    payload = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "model_type": "ic32_regime_v1",
        "feature_cols": FEAT,
        "source": "data/training/labeled/*_features_v3.parquet",
        "n_rows": n_rows,
        "n_coins": n_coins,
        "features": features,
    }
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved {OUT} ({n_coins} coins, {n_rows:,} rows, {len(features)} features)")
    lsr = features.get("long_short_ratio", {})
    print(f"LSR training: mean={lsr.get('mean')} p5={lsr.get('p5')} p95={lsr.get('p95')}")


if __name__ == "__main__":
    main()