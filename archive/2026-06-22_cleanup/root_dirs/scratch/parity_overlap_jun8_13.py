# -*- coding: utf-8 -*-
"""Compare live DB vs holdout for Jun 8-13 overlap — non-positioning features."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import HOLDOUT_DIR, MODEL_DIR
from core.utils import ensure_utc_index
from tools.live_db_bridge import load_signals, LOCAL_DB

FEAT = json.load(open(MODEL_DIR / "feature_cols_ic32_regime.json", encoding="utf-8"))
# Fitur yang seharusnya match jika klines+H4 shift sama (bukan positioning)
STABLE = [
    "dist_from_8h_high", "rsi_6", "swing_momentum", "stochrsi_k",
    "dist_liq_50x_long", "dist_liq_50x_short", "Fib_786", "Fib_618",
    "dist_liq_20x_long", "dist_liq_20x_short", "log_ret_20", "ema_50_h1",
    "MSB_BOS", "hmm_regime_enc", "h4_trend",
]
POSITIONING = [
    "long_short_ratio", "ofi_h4_delta", "cvd", "cvd_slope_h4",
    "cvd_momentum_adv", "ofi_acceleration", "cvd_div_h4",
    "Buy_Liq", "Sell_Liq", "whale_retail_divergence",
]
H = HOLDOUT_DIR / "labeled"


def holdout_bar(sym, ts):
    df = ensure_utc_index(pd.read_parquet(H / f"{sym}_features_v3.parquet")).sort_index()
    rp = H / f"{sym}_regime_h1.parquet"
    if rp.exists():
        reg = ensure_utc_index(pd.read_parquet(rp))
        if "hmm_regime_enc" in df.columns:
            df = df.drop(columns=["hmm_regime_enc"])
        df = df.join(reg[["hmm_regime_enc"]], how="left")
        df["hmm_regime_enc"] = df["hmm_regime_enc"].fillna(1).astype("int32")
    sub = df[df.index <= ts]
    return sub.iloc[-1] if len(sub) else None


def main():
    sig = load_signals(LOCAL_DB)
    overlap = sig[(sig["signal_time"] >= "2026-06-08") & (sig["signal_time"] < "2026-06-14")]
    deltas = {f: [] for f in FEAT}
    hmm_ok = 0
    n = 0
    for _, r in overlap.iterrows():
        snap = json.loads(r["feature_snapshot"] or "{}")
        if not snap:
            continue
        ts = pd.Timestamp(r["signal_time"])
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        hb = holdout_bar(r["coin_symbol"], ts)
        if hb is None:
            continue
        n += 1
        if int(snap.get("hmm_regime_enc", -99)) == int(hb.get("hmm_regime_enc", -98)):
            hmm_ok += 1
        for f in FEAT:
            lv, hv = snap.get(f), hb.get(f)
            if lv is None or hv is None or (isinstance(lv, float) and np.isnan(lv)):
                continue
            deltas[f].append(abs(float(lv) - float(hv)))

    print(f"Overlap Jun 8-13: {n} signals compared\n")
    print(f"HMM match: {hmm_ok}/{n} ({100*hmm_ok/max(n,1):.1f}%)\n")

    def summarize(names):
        rows = []
        for f in names:
            v = deltas[f]
            if not v:
                continue
            med = float(np.median(v))
            rows.append((f, med, len(v)))
        return sorted(rows, key=lambda x: x[1])

    print("STABLE feats (median abs delta, lower=better):")
    for f, med, cnt in summarize(STABLE):
        flag = "OK" if (f == "hmm_regime_enc" and med < 0.5) or (f != "hmm_regime_enc" and med < 0.1) else "DRIFT"
        print(f"  [{flag}] {f}: median={med:.4g} n={cnt}")

    print("\nPOSITIONING feats (expected large drift pre-fix):")
    for f, med, cnt in summarize(POSITIONING):
        print(f"  {f}: median={med:.4g} n={cnt}")


if __name__ == "__main__":
    main()