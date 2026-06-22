# -*- coding: utf-8 -*-
"""Parity audit: VPS pipeline vs holdout vs DB signals."""
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
H = HOLDOUT_DIR / "labeled"

# VPS manual fetch 2026-06-18 15:31 UTC
VPS_PARTIAL = {
    "SOLUSDT": {
        "hmm_regime_enc": 1.0, "long_short_ratio": 3.0193, "ofi_h4_delta": 242111.27,
        "cvd": -4490287.38, "rsi_h4": 26.05, "h4_trend": -1.0, "cvd_slope_h4": -0.042,
        "stochrsi_d": 40.27, "log_ret_20": -0.0252,
    },
    "ADAUSDT": {
        "hmm_regime_enc": 2.0, "long_short_ratio": 2.0516, "ofi_h4_delta": 3716525.0,
        "cvd": -530494170.0, "rsi_h4": 24.97, "h4_trend": -1.0,
    },
    "BTCUSDT": {
        "hmm_regime_enc": 2.0, "long_short_ratio": 1.8058, "ofi_h4_delta": 2033.82,
        "cvd": -25087.68, "rsi_h4": 27.67, "h4_trend": -1.0,
    },
    "LINKUSDT": {
        "hmm_regime_enc": 0.0, "long_short_ratio": 1.7855, "ofi_h4_delta": 355713.6,
        "cvd": -8003473.51, "rsi_h4": 34.29, "h4_trend": -1.0,
    },
}
TS = pd.Timestamp("2026-06-18 15:00:00", tz="UTC")


def holdout_bar(sym: str, ts: pd.Timestamp) -> pd.Series | None:
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


def tol(f: str, hv: float) -> float:
    if f == "hmm_regime_enc":
        return 0.5
    if f == "long_short_ratio":
        return max(0.15, abs(hv) * 0.15)
    if f in ("ofi_h4_delta", "cvd", "cvd_momentum_adv", "Buy_Liq", "Sell_Liq"):
        return max(5e4, abs(hv) * 0.35)
    return max(0.05, abs(hv) * 0.15)


def main():
    sig = load_signals(LOCAL_DB)

    print("=== A) VPS inference pipeline vs HOLDOUT @ 15:00 UTC ===\n")
    all_mismatch = {}
    for sym, live in VPS_PARTIAL.items():
        h = holdout_bar(sym, TS)
        if h is None:
            print(f"{sym}: no holdout bar")
            continue
        bad, ok = [], []
        for f in FEAT:
            if f not in live:
                continue
            lv, hv = float(live[f]), float(h.get(f, 0) or 0)
            d = abs(lv - hv)
            t = tol(f, hv)
            entry = (sym, f, lv, hv, d, t)
            (ok if d <= t else bad).append(entry)
            if d > t:
                all_mismatch.setdefault(f, []).append(d)
        print(f"{sym}: {len(ok)} OK / {len(bad)} MISMATCH (sampled {len(live)} feats)")
        for e in sorted(bad, key=lambda x: -x[4])[:6]:
            print(f"  {e[1]}: live={e[2]:.4g} hold={e[3]:.4g} delta={e[4]:.4g}")

    print("\n=== B) Holdout LSR/OI baseline (Jun 10-13) ===")
    for sym in ["SOLUSDT", "BTCUSDT"]:
        h = holdout_bar(sym, TS)
        if h is not None:
            print(
                f"  {sym}: LSR={float(h.get('long_short_ratio',0)):.4f} "
                f"OFI={float(h.get('ofi_h4_delta',0)):.0f} CVD={float(h.get('cvd',0)):.0f}"
            )

    print("\n=== C) DB signals Jun 17+ ===")
    jun17 = sig[sig["signal_time"] >= "2026-06-17"]
    lsr = []
    oi_synthetic = 0
    hmm_m, hmm_n = 0, 0
    for _, r in jun17.iterrows():
        s = json.loads(r["feature_snapshot"] or "{}")
        lsr.append(s.get("long_short_ratio"))
        oi = s.get("open_interest")
        if oi is not None and 0.5 < float(oi) < 2.0:
            oi_synthetic += 1
        ts = pd.Timestamp(r["signal_time"])
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        hb = holdout_bar(r["coin_symbol"], ts)
        if hb is not None and "hmm_regime_enc" in s:
            hmm_n += 1
            if int(s["hmm_regime_enc"]) == int(hb["hmm_regime_enc"]):
                hmm_m += 1
    lsr_s = pd.Series(lsr)
    print(f"  n={len(jun17)} LSR_zero={(lsr_s == 0).sum()} LSR_mean={lsr_s.replace(0, np.nan).mean()}")
    print(f"  OI looks synthetic (~1.0): {oi_synthetic}/{len(jun17)}")
    print(f"  HMM match holdout: {hmm_m}/{hmm_n} ({100*hmm_m/max(hmm_n,1):.1f}%)")

    print("\n=== D) DB signals Jun 18 (latest batch) ===")
    today = sig[sig["signal_time"].astype(str).str.startswith("2026-06-18")]
    for _, r in today.tail(5).iterrows():
        s = json.loads(r["feature_snapshot"] or "{}")
        print(
            f"  {r['signal_time']} {r['coin_symbol']} "
            f"LSR={s.get('long_short_ratio')} HMM={s.get('hmm_regime_enc')} "
            f"OI={round(float(s.get('open_interest') or 0), 2)}"
        )

    print("\n=== E) Top mismatch features (VPS vs holdout, aggregated) ===")
    for f, ds in sorted(all_mismatch.items(), key=lambda x: -np.mean(x[1]))[:10]:
        print(f"  {f}: mean_delta={np.mean(ds):.4g} n={len(ds)}")

    out = ROOT / "reports" / "experiments" / "parity_audit_now.json"
    payload = {
        "vps_vs_holdout_mismatch_feats": {k: float(np.mean(v)) for k, v in all_mismatch.items()},
        "jun17_lsr_zero": int((lsr_s == 0).sum()),
        "jun17_hmm_match_pct": round(100 * hmm_m / max(hmm_n, 1), 1),
        "note": "VPS pipeline manual OK; DB snapshots pre-16:05 UTC still stale",
    }
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()