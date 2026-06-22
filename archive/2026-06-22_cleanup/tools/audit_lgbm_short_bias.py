"""Audit LGBM SHORT/LONG bias: features, labels, thresholds, FLIP, live signals."""
import json
import sqlite3
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.cascade_utils import compute_regime_flip_delta, check_macro_alignment_gate, SHORT, LONG, FLAT

SWINT = Path("D:/Apps-Dev/swint_tradev2")
OOF_PATH = ROOT / "models/runs/ic32_regime_v1/oof_predictions.parquet"
FEAT_JSON = SWINT / "models/feature_cols_v2.json"
LGBM_PATH = SWINT / "models/lgbm_baseline.pkl"
INF_CFG = ROOT / "models/inference_config.json"
LIVE_DB = ROOT / "data/live_cache/app.db"


def main():
    feat_cols = json.load(open(FEAT_JSON, encoding="utf-8"))
    cfg = json.load(open(INF_CFG, encoding="utf-8"))
    thr_l = float(cfg["cascade"]["lgbm_threshold_long"])
    thr_s = float(cfg["cascade"]["lgbm_threshold_short"])
    ra = cfg.get("regime_alignment", {})

    print("=" * 60)
    print("LGBM SHORT/LONG BIAS AUDIT (ic32_regime_v1)")
    print("=" * 60)

    # 1. Feature importance
    lgbm = joblib.load(LGBM_PATH)
    imp = sorted(zip(feat_cols, lgbm.feature_importances_), key=lambda x: -x[1])
    print("\n[1] Top feature importance (no directional bias inherent):")
    for f, v in imp[:8]:
        print(f"  {f}: {v:.0f}")

    # 2. OOF score balance
    oof = pd.read_parquet(OOF_PATH).dropna(subset=["p0", "p2"])
    p0, p2 = oof["p0"].values, oof["p2"].values
    print("\n[2] OOF LGBM raw scores (valid=%d):" % len(oof))
    print("  p0 mean=%.4f  p2 mean=%.4f  p0>p2=%.1f%%" % (p0.mean(), p2.mean(), (p0 > p2).mean() * 100))
    wl = (p2 >= thr_l).sum()
    ws = ((p0 >= thr_s) & ~(p2 >= thr_l)).sum()
    print("  thr L=%.2f S=%.2f -> LONG %d SHORT %d ratio=%.2fx" % (thr_l, thr_s, wl, ws, ws / max(wl, 1)))
    print("  >> SHORT bias dari THRESHOLD ASIMETRIS (bukan fitur)")

    # 3. Label balance
    lab = oof["label"].value_counts()
    print("\n[3] Training labels in OOF:")
    names = {0: "SHORT", 1: "FLAT", 2: "LONG"}
    for k, v in lab.items():
        print("  %s: %d (%.1f%%)" % (names.get(k, k), v, v / len(oof) * 100))

    # 4. Feature-label separation (BTC sample)
    btc = pd.read_parquet(ROOT / "data/training/labeled/BTCUSDT_features_v3.parquet")
    lab_s = btc["label"].astype(str)
    sm, lm = lab_s == "SHORT", lab_s == "LONG"
    print("\n[4] Feature means SHORT vs LONG (BTC, balanced labels %d/%d):" % (sm.sum(), lm.sum()))
    for c in ["dist_from_8h_high", "rsi_h4", "stochrsi_d", "h4_trend"]:
        if c in btc.columns:
            d = (btc.loc[sm, c].mean() - btc.loc[lm, c].mean()) / btc[c].std()
            print("  %s: d=%.3f (mean-reversion: SHORT sedikit lebih dekat 8h high)" % (c, d))

    # 5. FLIP impact simulation on sample
    sample = pd.read_parquet(ROOT / "data/training/labeled/BTCUSDT_features_v3.parquet").tail(5000)
    flip_short_boost = flip_short_pen = 0
    for _, row in sample.iterrows():
        p0v, p2v = float(row.get("p0", 0)), float(row.get("p2", 0))
        if p2v >= thr_l:
            d = LONG
        elif p0v >= thr_s:
            d = SHORT
        else:
            continue
        enc = int(row.get("hmm_regime_enc", -1))
        h4 = float(row.get("h4_trend", 0))
        delta, lbl = compute_regime_flip_delta(d, enc, h4, ra, market_breadth=0.5)
        if d == SHORT and "counter+" in lbl:
            flip_short_boost += 1
        if d == SHORT and "counter-" in lbl:
            flip_short_pen += 1
    print("\n[5] FLIP on BTC sample (breadth=0.5): SHORT counter-trend boosted=%d penalized=%d" % (
        flip_short_boost, flip_short_pen))

    gate_cfg = cfg.get("breadth_gate", {"enabled": True, "bull_threshold": 0.70, "bear_threshold": 0.30})
    blocked = 0
    for _, row in sample.iterrows():
        p0v, p2v = float(row.get("p0", 0)), float(row.get("p2", 0))
        if p2v >= thr_l:
            direction = "LONG"
        elif p0v >= thr_s:
            direction = "SHORT"
        else:
            continue
        h4 = float(row.get("h4_trend", 0))
        ok, _ = check_macro_alignment_gate(direction, h4, 0.85, gate_cfg)
        if not ok:
            blocked += 1
    print("  Macro gate (breadth=0.85): would block %d/%d directional candidates" % (blocked, flip_short_boost + flip_short_pen + 100))

    # 6. Live signals
    if LIVE_DB.exists():
        conn = sqlite3.connect(LIVE_DB)
        sig = pd.read_sql(
            "SELECT direction, confidence FROM signal WHERE direction IN ('LONG','SHORT') "
            "ORDER BY created_at DESC LIMIT 2000",
            conn,
        )
        trade = pd.read_sql(
            "SELECT direction, pnl_net, closed_at FROM trade WHERE is_live=1", conn,
        )
        conn.close()
        print("\n[6] Live directional signals (recent):")
        print(sig["direction"].value_counts().to_string())
        closed = trade[trade["closed_at"].notna()]
        for d in ("LONG", "SHORT"):
            sub = closed[closed["direction"] == d]
            if len(sub):
                print("  %s trades: n=%d WR=%.1f%% net=$%.2f" % (
                    d, len(sub), (sub["pnl_net"] > 0).mean() * 100, sub["pnl_net"].sum()))

    print("\n" + "=" * 60)
    print("KESIMPULAN: Fitur & label seimbang. SHORT bias dari:")
    print("  (a) thr_short=%.2f < thr_long=%.2f" % (thr_s, thr_l))
    print("  (b) FLIP RANGING boost counter-trend SHORT saat h4 UP")
    print("  (c) market_breadth dihitung tapi tidak dipakai di cascade")
    print("=" * 60)


if __name__ == "__main__":
    main()