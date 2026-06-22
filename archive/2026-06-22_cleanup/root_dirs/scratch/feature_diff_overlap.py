# -*- coding: utf-8 -*-
"""
Bandingkan fitur holdout parquet vs live feature_snapshot pada jam mismatch.
Fokus overlap 8-13 Jun 2026.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import HOLDOUT_DIR, MODEL_DIR, LABEL_MAP
from core.utils import ensure_utc_index
from pipeline.ic32_fusion_shared import build_per_bar_thresholds, load_b_dir_hmm_cfg


OUT = ROOT / "reports" / "experiments"
HOLDOUT_LABEL = HOLDOUT_DIR / "labeled"
FEAT_COLS = json.load(open(MODEL_DIR / "feature_cols_ic32_regime.json", encoding="utf-8"))
LSTM_FEATS = json.load(open(MODEL_DIR / "feature_cols_lstm_temporal.json", encoding="utf-8"))[:11]
OVERLAP = ["2026-06-08", "2026-06-09", "2026-06-10", "2026-06-11", "2026-06-12", "2026-06-13"]

# Kasus: holdout trade, live FLAT di jam sama (dari overlap_mismatch_detail)
CASES = [
    ("2026-06-08", "ADAUSDT", 9, "LONG"),
    ("2026-06-08", "SUIUSDT", 9, "SHORT"),
    ("2026-06-08", "LINKUSDT", 12, "LONG"),
    ("2026-06-09", "LINKUSDT", 13, "SHORT"),
    ("2026-06-10", "SUIUSDT", 6, "LONG"),
    ("2026-06-11", "SOLUSDT", 8, "LONG"),
    ("2026-06-11", "ADAUSDT", 8, "LONG"),
    ("2026-06-11", "LINKUSDT", 8, "LONG"),
    ("2026-06-11", "SUIUSDT", 8, "LONG"),
    ("2026-06-11", "ARBUSDT", 8, "LONG"),
    ("2026-06-11", "TAOUSDT", 8, "LONG"),
    ("2026-06-11", "XRPUSDT", 8, "LONG"),
    ("2026-06-11", "TONUSDT", 9, "LONG"),
    ("2026-06-12", "TAOUSDT", 20, "SHORT"),
]

KEY_FEATS = [
    "hmm_regime_enc", "h4_trend", "cvd_slope_h4", "ofi_h4_delta", "vol_ratio_20",
    "vol_spike_zscore", "stochrsi_d", "log_ret_20", "atr_percentile_h1",
    "dist_from_8h_high", "rsi_h4", "long_short_ratio", "close",
]


def _wita_to_utc_ts(date_wita: str, hour_wita: int) -> pd.Timestamp:
    return pd.Timestamp(f"{date_wita} {hour_wita:02d}:00:00", tz="Asia/Makassar").tz_convert("UTC")


def _load_holdout_row(symbol: str, ts_utc: pd.Timestamp) -> pd.Series | None:
    p = HOLDOUT_LABEL / f"{symbol}_features_v3.parquet"
    if not p.exists():
        return None
    df = ensure_utc_index(pd.read_parquet(p)).sort_index()
    rp = HOLDOUT_LABEL / f"{symbol}_regime_h1.parquet"
    if rp.exists():
        reg = pd.read_parquet(rp)
        if "hmm_regime_enc" in df.columns:
            df = df.drop(columns=["hmm_regime_enc"])
        df = df.join(reg[["hmm_regime_enc"]], how="left")
        df["hmm_regime_enc"] = df["hmm_regime_enc"].fillna(1).astype("int32")
    # nearest bar at or before ts
    sub = df[df.index <= ts_utc]
    if sub.empty:
        return None
    return sub.iloc[-1]


def _load_live_signal(signals: pd.DataFrame, symbol: str, date_wita: str, hour_wita: int) -> dict | None:
    sub = signals[
        (signals["symbol"] == symbol)
        & (signals["date_wita"] == date_wita)
        & (signals["hour_wita"] == hour_wita)
    ]
    if sub.empty:
        return None
    row = sub.iloc[0]
    snap = {}
    if pd.notna(row.get("feature_snapshot")) and row["feature_snapshot"]:
        try:
            snap = json.loads(row["feature_snapshot"])
        except json.JSONDecodeError:
            pass
    return {
        "direction": row["direction"],
        "confidence": row.get("confidence"),
        "entry_reason": row.get("entry_reason"),
        "ts_wita": row.get("ts_wita"),
        "snapshot": snap,
    }


def _lgbm_decision(row: pd.Series, lgbm, hmm_cfg: dict) -> dict:
    feat_cols = FEAT_COLS
    X = pd.DataFrame(
        [[float(row.get(col, 0) or 0) for col in feat_cols]],
        columns=feat_cols,
    )
    proba = lgbm.predict_proba(X)[0]
    hmm_enc = int(row.get("hmm_regime_enc", 1))
    thr_l, thr_s = hmm_cfg.get(hmm_enc, hmm_cfg[-1])
    inv = {0: "BEARISH", 1: "NEUTRAL", 2: "BULLISH"}
    idx = int(np.argmax(proba))
    return {
        "lgbm_pred": inv[idx],
        "lgbm_conf": float(proba[idx]),
        "lgbm_proba": [round(float(p), 4) for p in proba],
        "hmm_enc": hmm_enc,
        "hmm_thr_long": thr_l,
        "hmm_thr_short": thr_s,
    }


def main():
    sigs_raw = pd.read_csv(OUT / "live_ic32_signals.csv")
    # Re-load with snapshots from DB
    sys.path.insert(0, str(ROOT))
    from tools.live_db_bridge import load_signals
    sigs_db = load_signals()
    sigs_db = sigs_db[sigs_db["model_type"] == "ic32_regime_v1"].copy()
    sigs_db["ts_wita"] = pd.to_datetime(sigs_db["signal_time"], utc=True).dt.tz_convert("Asia/Makassar")
    sigs_db["date_wita"] = sigs_db["ts_wita"].dt.strftime("%Y-%m-%d")
    sigs_db["hour_wita"] = sigs_db["ts_wita"].dt.hour
    sigs_db["symbol"] = sigs_db["coin_symbol"]

    hmm_cfg = load_b_dir_hmm_cfg()
    lgbm = joblib.load(MODEL_DIR / "runs/ic32_regime_v1/lgbm.pkl")


    rows = []
    feat_diffs_all = []

    for date_wita, symbol, hour_wita, hold_dir in CASES:
        ts_utc = _wita_to_utc_ts(date_wita, hour_wita)
        h_row = _load_holdout_row(symbol, ts_utc)
        live = _load_live_signal(sigs_db, symbol, date_wita, hour_wita)

        rec = {
            "date_wita": date_wita,
            "hour_wita": hour_wita,
            "symbol": symbol,
            "holdout_expected_dir": hold_dir,
            "holdout_ts_utc": str(ts_utc),
            "live_direction": live["direction"] if live else "NO_SIGNAL",
            "live_confidence": live.get("confidence") if live else None,
            "live_entry_reason": live.get("entry_reason") if live else None,
        }

        if h_row is not None:
            p = HOLDOUT_LABEL / f"{symbol}_features_v3.parquet"
            df = ensure_utc_index(pd.read_parquet(p)).sort_index()
            rp = HOLDOUT_LABEL / f"{symbol}_regime_h1.parquet"
            if rp.exists():
                reg = pd.read_parquet(rp)
                if "hmm_regime_enc" in df.columns:
                    df = df.drop(columns=["hmm_regime_enc"])
                df = df.join(reg[["hmm_regime_enc"]], how="left")
                df["hmm_regime_enc"] = df["hmm_regime_enc"].fillna(1).astype("int32")
            idx = df.index.get_indexer([ts_utc], method="pad")[0]
            pred = _lgbm_decision(h_row, lgbm, hmm_cfg)
            rec["holdout_lgbm_pred"] = pred["lgbm_pred"]
            rec["holdout_lgbm_conf"] = round(pred["lgbm_conf"], 4)
            rec["holdout_lgbm_proba"] = pred["lgbm_proba"]
            rec["holdout_hmm_enc"] = pred["hmm_enc"]
            rec["holdout_thr_l_s"] = f"{pred['hmm_thr_long']:.2f}/{pred['hmm_thr_short']:.2f}"

            snap = live["snapshot"] if live else {}
            for feat in KEY_FEATS:
                hv = h_row.get(feat)
                lv = snap.get(feat)
                if hv is None and lv is None:
                    continue
                try:
                    hv_f = float(hv) if pd.notna(hv) else None
                    lv_f = float(lv) if lv is not None else None
                except (TypeError, ValueError):
                    continue
                if hv_f is not None and lv_f is not None:
                    diff = lv_f - hv_f
                    if abs(diff) > 1e-6 or feat == "hmm_regime_enc":
                        feat_diffs_all.append({
                            "case": f"{date_wita} {hour_wita:02d} {symbol}",
                            "feature": feat,
                            "holdout": round(hv_f, 6),
                            "live": round(lv_f, 6),
                            "delta": round(diff, 6),
                        })
                rec[f"h_{feat}"] = hv_f
                rec[f"l_{feat}"] = lv_f

            if snap:
                rec["live_lgbm_dec"] = snap.get("_lgbm_decision")
                rec["live_lgbm_conf"] = snap.get("_lgbm_conf")
                rec["live_lstm_dec"] = snap.get("_lstm_decision")
                rec["live_lstm_conf"] = snap.get("_lstm_conf")
                rec["live_cascade_stage"] = snap.get("_cascade_stage")
                lgbm_p = snap.get("_lgbm_proba")
                if lgbm_p:
                    rec["live_lgbm_proba"] = [round(x, 4) for x in lgbm_p]

        rows.append(rec)

    df_out = pd.DataFrame(rows)
    df_out.to_csv(OUT / "feature_diff_overlap_cases.csv", index=False)

    # Top feature deltas
    if feat_diffs_all:
        fd = pd.DataFrame(feat_diffs_all)
        fd["abs_delta"] = fd["delta"].abs()
        top = (
            fd.groupby("feature")["abs_delta"]
            .agg(["mean", "max", "count"])
            .sort_values("mean", ascending=False)
            .reset_index()
        )
        top.to_csv(OUT / "feature_diff_overlap_top_features.csv", index=False)
    else:
        top = pd.DataFrame()

    # Root cause tags per case
    causes = []
    for r in rows:
        tags = []
        if r.get("live_direction") == "FLAT":
            reason = str(r.get("live_entry_reason") or "")
            if "flat" in reason.lower() or "FLAT" in reason:
                tags.append("live_cascade_flat")
            if r.get("holdout_hmm_enc") != r.get("l_hmm_regime_enc"):
                tags.append("hmm_regime_mismatch")
            if r.get("live_lgbm_dec") and r["live_lgbm_dec"] != r.get("holdout_expected_dir"):
                tags.append("lgbm_decision_diff")
            if r.get("live_lstm_dec") == "BEARISH" and r.get("holdout_expected_dir") == "LONG":
                tags.append("lstm_opposite")
            if r.get("live_lstm_dec") == "BULLISH" and r.get("holdout_expected_dir") == "SHORT":
                tags.append("lstm_opposite")
        if r.get("holdout_lgbm_pred") and r["holdout_lgbm_pred"] != r.get("holdout_expected_dir"):
            tags.append("holdout_lgbm_vs_trade_dir")
        causes.append({
            "case": f"{r['date_wita']} {r['hour_wita']:02d} {r['symbol']}",
            "hold_dir": r.get("holdout_expected_dir"),
            "live_dir": r.get("live_direction"),
            "hold_lgbm": r.get("holdout_lgbm_pred"),
            "hold_lgbm_conf": r.get("holdout_lgbm_conf"),
            "live_lgbm": r.get("live_lgbm_dec"),
            "live_lstm": r.get("live_lstm_dec"),
            "live_reason": (r.get("live_entry_reason") or "")[:120],
            "root_cause_tags": "|".join(tags) if tags else "ok",
        })

    report = {
        "cases_analyzed": len(CASES),
        "top_feature_divergence": top.head(15).to_dict("records") if not top.empty else [],
        "per_case": causes,
        "findings": [],
    }

    # Auto findings
    if not top.empty:
        hmm_row = top[top["feature"] == "hmm_regime_enc"]
        if not hmm_row.empty and hmm_row.iloc[0]["count"] > 0:
            report["findings"].append(
                f"hmm_regime_enc beda di {int(hmm_row.iloc[0]['count'])} kasus — "
                "holdout pakai precomputed parquet, live hitung ulang per jam"
            )
    lsr = top[top["feature"] == "long_short_ratio"] if not top.empty else pd.DataFrame()
    if not lsr.empty:
        report["findings"].append(
            "long_short_ratio live selalu 0 (by design data_service.py); holdout pakai nilai historis parquet"
        )

    flat_lstm = [c for c in causes if "lstm_opposite" in c.get("root_cause_tags", "")]
    if flat_lstm:
        report["findings"].append(
            f"LSTM opposite review -> FLAT di {len(flat_lstm)} kasus holdout-LONG"
        )

    with open(OUT / "feature_diff_overlap_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)

    print("Saved feature_diff_overlap_cases.csv, feature_diff_overlap_report.json")
    print("\n=== PER CASE ===")
    for c in causes:
        print(f"{c['case']}: hold={c['hold_dir']} h_lgbm={c['hold_lgbm']}({c['hold_lgbm_conf']}) "
              f"live={c['live_dir']} lgbm={c['live_lgbm']} lstm={c['live_lstm']}")
        print(f"  reason: {c['live_reason']}")
        print(f"  tags: {c['root_cause_tags']}")
    if not top.empty:
        print("\n=== TOP FEATURE DELTA (mean abs) ===")
        print(top.head(10).to_string(index=False))


if __name__ == "__main__":
    main()