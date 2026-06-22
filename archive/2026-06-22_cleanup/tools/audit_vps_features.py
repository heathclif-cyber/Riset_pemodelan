# -*- coding: utf-8 -*-
"""
Audit fitur live VPS vs holdout parquet (ic32_regime_v1).

Membandingkan feature_snapshot dari DB live dengan bar holdout pada timestamp
yang sama. Output: reports/experiments/vps_feature_audit.json + .csv

Usage:
    python tools/audit_vps_features.py
    python tools/audit_vps_features.py --no-pull
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import HOLDOUT_DIR, MODEL_DIR
from core.utils import ensure_utc_index
from tools.live_db_bridge import LOCAL_DB, pull_live_db, load_signals

OUT_DIR = ROOT / "reports" / "experiments"
FEAT_COLS: list[str] = json.load(
    open(MODEL_DIR / "feature_cols_ic32_regime.json", encoding="utf-8")
)
HOLDOUT_LABEL = HOLDOUT_DIR / "labeled"
OVERLAP_START = "2026-06-08"
OVERLAP_END = "2026-06-14"

# Fitur yang diharapkan beda by design (dokumentasi swint) — flag terpisah
DESIGN_DIFF = {"long_short_ratio"}

# Fitur kritis untuk threshold/regime
CRITICAL = [
    "hmm_regime_enc", "h4_trend", "ofi_h4_delta", "cvd_slope_h4",
    "vol_ratio_20", "vol_spike_zscore", "rsi_h4", "stochrsi_d",
    "long_short_ratio", "log_ret_20", "atr_percentile_h1",
]


def _parse_snapshot(fs: str | None) -> dict:
    if not fs:
        return {}
    try:
        return json.loads(fs)
    except (json.JSONDecodeError, TypeError):
        return {}


def _load_holdout_bar(symbol: str, ts_utc: pd.Timestamp) -> pd.Series | None:
    p = HOLDOUT_LABEL / f"{symbol}_features_v3.parquet"
    if not p.exists():
        return None
    df = ensure_utc_index(pd.read_parquet(p)).sort_index()
    rp = HOLDOUT_LABEL / f"{symbol}_regime_h1.parquet"
    if rp.exists():
        reg = ensure_utc_index(pd.read_parquet(rp))
        if "hmm_regime_enc" in df.columns:
            df = df.drop(columns=["hmm_regime_enc"])
        df = df.join(reg[["hmm_regime_enc"]], how="left")
        df["hmm_regime_enc"] = df["hmm_regime_enc"].fillna(1).astype("int32")
    sub = df[df.index <= ts_utc]
    if sub.empty:
        return None
    return sub.iloc[-1]


def _signal_ts_utc(signal_time: str) -> pd.Timestamp:
    """signal_time di DB = WITA (Asia/Makassar), bar inference = jam sebelumnya UTC."""
    ts = pd.Timestamp(signal_time)
    if ts.tzinfo is None:
        ts = ts.tz_localize("Asia/Makassar")
    return ts.tz_convert("UTC")


def audit_snapshot_coverage(signals: pd.DataFrame) -> dict:
    """Cek kolom fitur yang hilang dari feature_snapshot."""
    overlap = signals[
        (signals["signal_time"] >= OVERLAP_START)
        & (signals["signal_time"] < OVERLAP_END)
    ]
    missing_per_feat: Counter = Counter()
    present_count = 0
    for fs in overlap["feature_snapshot"]:
        snap = _parse_snapshot(fs)
        if not snap:
            continue
        present_count += 1
        for f in FEAT_COLS:
            if f not in snap:
                missing_per_feat[f] += 1

    n = len(overlap)
    return {
        "overlap_signals": n,
        "with_snapshot": present_count,
        "missing_any_feat": {
            f: missing_per_feat[f] for f in FEAT_COLS if missing_per_feat[f] > 0
        },
        "all_33_present": [
            f for f in FEAT_COLS if missing_per_feat[f] == 0 and present_count > 0
        ],
    }


def audit_hmm_distribution(signals: pd.DataFrame) -> dict:
    rows = []
    for _, r in signals.iterrows():
        snap = _parse_snapshot(r["feature_snapshot"])
        hmm = snap.get("hmm_regime_enc", "MISSING")
        rows.append({
            "signal_time": r["signal_time"],
            "symbol": r["coin_symbol"],
            "hmm": hmm,
            "date": str(r["signal_time"])[:10],
        })
    df = pd.DataFrame(rows)
    out = {}
    for period, mask in [
        ("overlap_8_13", (df["date"] >= OVERLAP_START) & (df["date"] < OVERLAP_END)),
        ("jun17_plus", df["date"] >= "2026-06-17"),
        ("all", pd.Series(True, index=df.index)),
    ]:
        sub = df[mask]
        out[period] = {
            "n": len(sub),
            "hmm_dist": sub["hmm"].value_counts().to_dict(),
        }
    return out


def compare_features_vs_holdout(signals: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Bandingkan setiap signal overlap dengan bar holdout terdekat."""
    overlap = signals[
        (signals["signal_time"] >= OVERLAP_START)
        & (signals["signal_time"] < OVERLAP_END)
    ].copy()

    records = []
    feat_deltas: dict[str, list[float]] = {f: [] for f in CRITICAL}

    for _, r in overlap.iterrows():
        sym = r["coin_symbol"]
        snap = _parse_snapshot(r["feature_snapshot"])
        if not snap:
            continue
        ts_utc = _signal_ts_utc(str(r["signal_time"]))
        hold = _load_holdout_bar(sym, ts_utc)
        if hold is None:
            continue

        row = {
            "signal_time": r["signal_time"],
            "symbol": sym,
            "direction": r["direction"],
            "live_hmm": snap.get("hmm_regime_enc"),
            "hold_hmm": int(hold.get("hmm_regime_enc", -1)),
            "hmm_match": snap.get("hmm_regime_enc") == int(hold.get("hmm_regime_enc", -2)),
        }
        for f in CRITICAL:
            lv = snap.get(f)
            hv = hold.get(f)
            if lv is None or (isinstance(lv, float) and np.isnan(lv)):
                row[f"{f}_live"] = None
            else:
                row[f"{f}_live"] = float(lv)
            if hv is None or (isinstance(hv, float) and np.isnan(hv)):
                row[f"{f}_hold"] = None
            else:
                row[f"{f}_hold"] = float(hv)
            if row[f"{f}_live"] is not None and row[f"{f}_hold"] is not None:
                d = abs(row[f"{f}_live"] - row[f"{f}_hold"])
                row[f"{f}_delta"] = d
                feat_deltas[f].append(d)
        records.append(row)

    cmp_df = pd.DataFrame(records)
    summary = {
        "compared": len(cmp_df),
        "hmm_match_rate": float(cmp_df["hmm_match"].mean()) if len(cmp_df) else 0,
        "hmm_mismatch_count": int((~cmp_df["hmm_match"]).sum()) if len(cmp_df) else 0,
        "mean_abs_delta": {
            f: round(float(np.mean(v)), 6) if v else None
            for f, v in feat_deltas.items()
        },
        "median_abs_delta": {
            f: round(float(np.median(v)), 6) if v else None
            for f, v in feat_deltas.items()
        },
    }
    # Top mismatches by total delta
    if len(cmp_df):
        delta_cols = [c for c in cmp_df.columns if c.endswith("_delta")]
        cmp_df["_total_delta"] = cmp_df[delta_cols].sum(axis=1, skipna=True)
        top = cmp_df.nlargest(10, "_total_delta")[
            ["signal_time", "symbol", "direction", "live_hmm", "hold_hmm", "_total_delta"]
        ]
        summary["top_mismatch_cases"] = top.to_dict(orient="records")

    return cmp_df, summary


def audit_holdout_lsr_baseline() -> dict:
    """Distribusi long_short_ratio di holdout (training expectation)."""
    vals = []
    for sym in ["BTCUSDT", "SOLUSDT", "ADAUSDT"]:
        p = HOLDOUT_LABEL / f"{sym}_features_v3.parquet"
        if not p.exists():
            continue
        df = ensure_utc_index(pd.read_parquet(p))
        if "long_short_ratio" in df.columns:
            sub = df.loc[OVERLAP_START:OVERLAP_END, "long_short_ratio"].dropna()
            vals.extend(sub.tolist())
    if not vals:
        return {"status": "no_data"}
    s = pd.Series(vals)
    return {
        "n": len(s),
        "mean": round(float(s.mean()), 4),
        "std": round(float(s.std()), 4),
        "min": round(float(s.min()), 4),
        "max": round(float(s.max()), 4),
        "pct_zero": round(float((s == 0).mean()), 4),
        "note": "Holdout pakai nilai historis dari parquet (bukan hardcoded 0)",
    }


def build_findings(coverage, hmm_dist, compare_summary, lsr_holdout) -> list[dict]:
    findings = []

    overlap_hmm = hmm_dist.get("overlap_8_13", {}).get("hmm_dist", {})
    if overlap_hmm.get(0, 0) == hmm_dist.get("overlap_8_13", {}).get("n", 0) and overlap_hmm:
        findings.append({
            "severity": "CRITICAL",
            "issue": "hmm_regime_enc stuck at 0 during overlap 8-13 Jun",
            "detail": (
                "Semua sinyal overlap punya hmm_regime_enc=0. Holdout punya state 0/1/2/3. "
                "HMM .pkl baru deploy ~17 Jun (Birth timestamp VPS). Sebelum itu fallback "
                "on-the-fly atau df_4h=None -> default 0."
            ),
        })

    jun17_hmm = hmm_dist.get("jun17_plus", {}).get("hmm_dist", {})
    if len(jun17_hmm) > 1:
        findings.append({
            "severity": "INFO",
            "issue": "HMM recovered after Jun 17 deploy",
            "detail": f"Distribusi Jun17+: {jun17_hmm}",
        })

    if compare_summary.get("hmm_match_rate", 0) < 0.5:
        findings.append({
            "severity": "CRITICAL",
            "issue": "HMM mismatch vs holdout",
            "detail": (
                f"Match rate {compare_summary.get('hmm_match_rate', 0):.1%}, "
                f"mismatch={compare_summary.get('hmm_mismatch_count', 0)}"
            ),
        })

    findings.append({
        "severity": "HIGH",
        "issue": "long_short_ratio hardcoded 0 in live data_service.py",
        "detail": (
            f"Live selalu 0. Holdout overlap mean={lsr_holdout.get('mean', 'N/A')}, "
            f"pct_zero={lsr_holdout.get('pct_zero', 'N/A')}. "
            "Meski swint docs bilang 'by design', holdout ic32 pakai nilai historis ~1.0."
        ),
    })

    top_delta = compare_summary.get("mean_abs_delta", {})
    for f in ("ofi_h4_delta", "vol_ratio_20", "vol_spike_zscore", "rsi_h4"):
        d = top_delta.get(f)
        if d is not None and d > 0.1:
            findings.append({
                "severity": "HIGH",
                "issue": f"{f} large delta vs holdout",
                "detail": f"mean_abs_delta={d}",
            })

    miss = coverage.get("missing_any_feat", {})
    if miss:
        findings.append({
            "severity": "CRITICAL",
            "issue": "Features missing from live snapshots",
            "detail": miss,
        })

    return findings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-pull", action="store_true")
    args = parser.parse_args()

    if not args.no_pull:
        pull_live_db(force=True)

    signals = load_signals(LOCAL_DB)
    signals = signals.rename(columns={"coin_symbol": "coin_symbol"})

    coverage = audit_snapshot_coverage(signals)
    hmm_dist = audit_hmm_distribution(signals)
    cmp_df, compare_summary = compare_features_vs_holdout(signals)
    lsr_holdout = audit_holdout_lsr_baseline()
    findings = build_findings(coverage, hmm_dist, compare_summary, lsr_holdout)

    report = {
        "audit_time": pd.Timestamp.now(tz="UTC").isoformat(),
        "feature_cols": FEAT_COLS,
        "n_features": len(FEAT_COLS),
        "coverage": coverage,
        "hmm_distribution": hmm_dist,
        "holdout_lsr_baseline": lsr_holdout,
        "compare_summary": compare_summary,
        "findings": findings,
        "design_diff_features": list(DESIGN_DIFF),
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = OUT_DIR / "vps_feature_audit.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)

    if len(cmp_df):
        csv_path = OUT_DIR / "vps_feature_audit_compare.csv"
        cmp_df.to_csv(csv_path, index=False)
        print(f"Compare CSV: {csv_path}")

    print(f"Report: {json_path}")
    print("\n=== FINDINGS ===")
    for fi in findings:
        print(f"[{fi['severity']}] {fi['issue']}")
        print(f"  {fi['detail']}")
    print(f"\nHMM overlap: {hmm_dist.get('overlap_8_13', {})}")
    print(f"HMM Jun17+: {hmm_dist.get('jun17_plus', {})}")
    print(f"Compare: {compare_summary.get('compared', 0)} signals, "
          f"HMM match {compare_summary.get('hmm_match_rate', 0):.1%}")


if __name__ == "__main__":
    main()