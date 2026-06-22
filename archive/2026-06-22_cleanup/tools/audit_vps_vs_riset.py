# -*- coding: utf-8 -*-
"""
Audit parity VPS live vs Riset holdout setelah fix positioning.

Output: reports/experiments/vps_vs_riset_audit.json
"""
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
from tools.live_db_bridge import load_signals, LOCAL_DB, pull_live_db

OUT = ROOT / "reports" / "experiments"
FEAT = json.load(open(MODEL_DIR / "feature_cols_ic32_regime.json", encoding="utf-8"))
KEY = [
    "hmm_regime_enc", "long_short_ratio", "open_interest", "ofi_h4_delta",
    "cvd", "rsi_h4", "h4_trend", "cvd_slope_h4", "stochrsi_d", "log_ret_20",
    "vol_ratio_20", "atr_percentile_h1",
]
HOLDOUT = HOLDOUT_DIR / "labeled"
# Sinyal pasca-fix (post positioning fetch ~15:07 UTC)
POST_FIX_START = "2026-06-18 15:00"


def _parse_snap(fs):
    try:
        return json.loads(fs) if fs else {}
    except json.JSONDecodeError:
        return {}


def _holdout_bar(sym: str, ts_utc: pd.Timestamp) -> pd.Series | None:
    p = HOLDOUT / f"{sym}_features_v3.parquet"
    if not p.exists():
        return None
    df = ensure_utc_index(pd.read_parquet(p)).sort_index()
    rp = HOLDOUT / f"{sym}_regime_h1.parquet"
    if rp.exists():
        reg = ensure_utc_index(pd.read_parquet(rp))
        if "hmm_regime_enc" in df.columns:
            df = df.drop(columns=["hmm_regime_enc"])
        df = df.join(reg[["hmm_regime_enc"]], how="left")
        df["hmm_regime_enc"] = df["hmm_regime_enc"].fillna(1).astype("int32")
    sub = df[df.index <= ts_utc]
    return sub.iloc[-1] if len(sub) else None


def audit_post_fix_signals(signals: pd.DataFrame) -> dict:
    post = signals[signals["signal_time"] >= POST_FIX_START].copy()
    hmm_dist = {}
    lsr_vals = []
    oi_vals = []
    records = []

    for _, r in post.iterrows():
        snap = _parse_snap(r["feature_snapshot"])
        if not snap:
            continue
        hmm = snap.get("hmm_regime_enc", "NA")
        hmm_dist[hmm] = hmm_dist.get(hmm, 0) + 1
        lsr_vals.append(snap.get("long_short_ratio"))
        if "open_interest" in snap:
            oi_vals.append(snap.get("open_interest"))

        ts = pd.Timestamp(r["signal_time"]).tz_localize("Asia/Makassar").tz_convert("UTC")
        h = _holdout_bar(r["coin_symbol"], ts)
        if h is None:
            continue
        row = {"signal_time": r["signal_time"], "symbol": r["coin_symbol"], "direction": r["direction"]}
        for k in KEY:
            lv, hv = snap.get(k), h.get(k)
            if lv is not None and hv is not None and not (pd.isna(lv) or pd.isna(hv)):
                row[f"{k}_live"] = float(lv)
                row[f"{k}_hold"] = float(hv)
                row[f"{k}_delta"] = abs(float(lv) - float(hv))
        records.append(row)

    cmp_df = pd.DataFrame(records)
    summary = {
        "post_fix_signals": len(post),
        "with_snapshot": int(post["feature_snapshot"].notna().sum()),
        "hmm_dist": hmm_dist,
        "lsr_live_mean": float(pd.Series(lsr_vals).mean()) if lsr_vals else None,
        "lsr_live_zero_pct": float((pd.Series(lsr_vals) == 0).mean()) if lsr_vals else None,
        "oi_in_snapshot_pct": len(oi_vals) / max(len(post), 1),
    }
    if len(cmp_df):
        delta_cols = [c for c in cmp_df.columns if c.endswith("_delta")]
        summary["compared"] = len(cmp_df)
        summary["hmm_match"] = float(
            (cmp_df.get("hmm_regime_enc_live", pd.Series()) == cmp_df.get("hmm_regime_enc_hold", pd.Series())).mean()
        ) if "hmm_regime_enc_live" in cmp_df else None
        summary["mean_delta"] = {
            c.replace("_delta", ""): round(float(cmp_df[c].mean()), 6)
            for c in delta_cols if cmp_df[c].notna().any()
        }
        cmp_df.to_csv(OUT / "vps_vs_riset_post_fix.csv", index=False)
    return summary, cmp_df


def audit_overlap_historical(signals: pd.DataFrame) -> dict:
    """Overlap 8-13 Jun — sinyal lama (pre-fix) untuk referensi."""
    overlap = signals[
        (signals["signal_time"] >= "2026-06-08") & (signals["signal_time"] < "2026-06-14")
    ]
    hmm = []
    lsr = []
    for fs in overlap["feature_snapshot"]:
        s = _parse_snap(fs)
        hmm.append(s.get("hmm_regime_enc"))
        lsr.append(s.get("long_short_ratio"))
    return {
        "n": len(overlap),
        "hmm_dist": pd.Series(hmm).value_counts().to_dict(),
        "lsr_zero_pct": float((pd.Series(lsr) == 0).mean()) if lsr else None,
        "note": "Data historis pre-fix — hanya referensi",
    }


def main():
    pull_live_db(force=True)
    signals = load_signals(LOCAL_DB)

    post_summary, _ = audit_post_fix_signals(signals)
    hist = audit_overlap_historical(signals)

    # Coverage: semua 33 LGBM cols di snapshot post-fix
    post = signals[signals["signal_time"] >= POST_FIX_START]
    miss_any = {}
    for fs in post["feature_snapshot"]:
        s = _parse_snap(fs)
        for f in FEAT:
            if f not in s:
                miss_any[f] = miss_any.get(f, 0) + 1

    report = {
        "audit_time": pd.Timestamp.now(tz="UTC").isoformat(),
        "post_fix": post_summary,
        "overlap_pre_fix": hist,
        "missing_feats_post_fix": miss_any,
        "findings": [],
    }

    if post_summary.get("lsr_live_zero_pct", 1) == 0:
        report["findings"].append({"ok": True, "msg": "long_short_ratio tidak lagi 0 di post-fix"})
    elif post_summary.get("post_fix_signals", 0) == 0:
        report["findings"].append({"warn": True, "msg": "Belum ada sinyal post-fix — tunggu cron berikutnya"})
    else:
        report["findings"].append({"critical": True, "msg": f"LSR masih 0: {post_summary.get('lsr_live_zero_pct')}"})

    hmm_d = post_summary.get("hmm_dist", {})
    if hmm_d and len(hmm_d) > 1:
        report["findings"].append({"ok": True, "msg": f"HMM bervariasi post-fix: {hmm_d}"})
    elif post_summary.get("post_fix_signals", 0) > 0:
        report["findings"].append({"warn": True, "msg": f"HMM masih dominan satu state: {hmm_d}"})

    md = post_summary.get("mean_delta", {})
    if md:
        big = {k: v for k, v in md.items() if v > 0.5 and k not in ("cvd", "ofi_h4_delta", "open_interest")}
        if big:
            report["findings"].append({"warn": True, "msg": f"Delta besar vs holdout: {big}"})
        report["findings"].append({
            "info": True,
            "msg": "cvd/ofi/oi delta besar diharapkan — holdout pakai synthetic, live pakai data asli",
        })

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "vps_vs_riset_audit.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)

    print(json.dumps(report, indent=2, default=str))
    print(f"\nReport: {path}")


if __name__ == "__main__":
    main()