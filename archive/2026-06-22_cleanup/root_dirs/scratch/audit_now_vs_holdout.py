# -*- coding: utf-8 -*-
"""Compare VPS fetch sample (now) vs holdout bar at same UTC timestamp."""
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import HOLDOUT_DIR
from core.utils import ensure_utc_index

VPS_SAMPLE = {
    "SOLUSDT": {"hmm_regime_enc": 1.0, "long_short_ratio": 3.0193, "open_interest": 10474511.58,
                "ofi_h4_delta": 242111.27, "cvd": -4490287.38, "rsi_h4": 35.82, "h4_trend": -1.0},
    "ADAUSDT": {"hmm_regime_enc": 2.0, "long_short_ratio": 2.0516, "open_interest": 424864828.0,
                "ofi_h4_delta": 3716525.0, "cvd": -530494170.0, "rsi_h4": 32.14, "h4_trend": -1.0},
    "BTCUSDT": {"hmm_regime_enc": 2.0, "long_short_ratio": 1.8058, "open_interest": 100431.151,
                "ofi_h4_delta": 2033.82, "cvd": -25087.68, "rsi_h4": 34.12, "h4_trend": -1.0},
    "LINKUSDT": {"hmm_regime_enc": 0.0, "long_short_ratio": 1.7855, "open_interest": 8216781.64,
                 "ofi_h4_delta": 355713.6, "cvd": -8003473.51, "rsi_h4": 42.77, "h4_trend": -1.0},
}
TS = pd.Timestamp("2026-06-18 15:00:00", tz="UTC")
HOLDOUT = HOLDOUT_DIR / "labeled"
KEYS = list(next(iter(VPS_SAMPLE.values())).keys())


def holdout_bar(sym):
    p = HOLDOUT / f"{sym}_features_v3.parquet"
    df = ensure_utc_index(pd.read_parquet(p)).sort_index()
    rp = HOLDOUT / f"{sym}_regime_h1.parquet"
    if rp.exists():
        reg = ensure_utc_index(pd.read_parquet(rp))
        if "hmm_regime_enc" in df.columns:
            df = df.drop(columns=["hmm_regime_enc"])
        df = df.join(reg[["hmm_regime_enc"]], how="left")
        df["hmm_regime_enc"] = df["hmm_regime_enc"].fillna(1).astype("int32")
    sub = df[df.index <= TS]
    return sub.iloc[-1] if len(sub) else None


rows = []
for sym, live in VPS_SAMPLE.items():
    h = holdout_bar(sym)
    if h is None:
        continue
    row = {"symbol": sym}
    for k in KEYS:
        lv, hv = live[k], float(h.get(k, 0) or 0)
        row[f"{k}_live"] = lv
        row[f"{k}_hold"] = round(hv, 4)
        row[f"{k}_delta"] = round(abs(lv - hv), 4)
        row[f"{k}_ok"] = row[f"{k}_delta"] < (0.05 if k == "hmm_regime_enc" else max(abs(hv) * 0.15, 1.0))
    rows.append(row)

df = pd.DataFrame(rows)
print("=== VPS NOW (post-fix) vs HOLDOUT @ 2026-06-18 15:00 UTC ===\n")
print(df.to_string(index=False))
print("\n--- Summary ---")
for k in KEYS:
    ok = df[f"{k}_ok"].sum() if f"{k}_ok" in df else 0
    print(f"{k}: {ok}/{len(df)} within tolerance")