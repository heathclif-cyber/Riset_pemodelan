# -*- coding: utf-8 -*-
"""
Full 33-feature parity: VPS pipeline (post-fix) vs Riset holdout vs DB snapshot.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import HOLDOUT_DIR, MODEL_DIR
from core.utils import ensure_utc_index
from tools.live_db_bridge import LOCAL_DB, pull_live_db, load_signals

FEAT = json.load(open(MODEL_DIR / "feature_cols_ic32_regime.json", encoding="utf-8"))
H = HOLDOUT_DIR / "labeled"
VPS = "root@139.180.157.176"
COINS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "ADAUSDT", "DOGEUSDT",
    "DOTUSDT", "LINKUSDT", "1000PEPEUSDT",
]

# Fitur yang memang expected beda (training synthetic vs live path)
EXPECTED_DIFF = {
    "long_short_ratio",  # training ~1.0; live pre-fix=0; post-fix ~1.0; manual VPS bisa real 2-3
    "cvd", "ofi_h4_delta", "ofi_acceleration", "cvd_div_h4", "cvd_momentum_adv",
    "whale_retail_divergence",  # derived dari LSR/CVD
}

# OHLCV/liquidity — harus match ketat
STRICT_MATCH = {
    "dist_from_8h_high", "rsi_6", "swing_momentum", "rsi_h4", "stochrsi_k",
    "dist_liq_50x_long", "trend_accel_4h", "rsi_slope_h4", "Fib_786", "Fib_618",
    "stochrsi_d", "dist_liq_50x_short", "Buy_Liq", "relative_strength_z",
    "dist_liq_20x_long", "Sell_Liq", "cvd_slope_h4", "ema_21_slope_h4",
    "ema_50_h1", "h4_trend", "log_ret_20", "dist_liq_20x_short",
    "vol_price_confirm", "ema_50_slope_h4", "MSB_BOS",
}


def holdout_bar(sym: str, ts: pd.Timestamp) -> pd.Series | None:
    p = H / f"{sym}_features_v3.parquet"
    if not p.exists():
        return None
    df = ensure_utc_index(pd.read_parquet(p)).sort_index()
    rp = H / f"{sym}_regime_h1.parquet"
    if rp.exists():
        reg = ensure_utc_index(pd.read_parquet(rp))
        if "hmm_regime_enc" in df.columns:
            df = df.drop(columns=["hmm_regime_enc"])
        df = df.join(reg[["hmm_regime_enc"]], how="left")
        df["hmm_regime_enc"] = df["hmm_regime_enc"].fillna(1).astype("int32")
    sub = df[df.index <= ts]
    return sub.iloc[-1] if len(sub) else None


def tol(feat: str, hold_val: float) -> float:
    if feat == "hmm_regime_enc":
        return 0.5
    if feat == "long_short_ratio":
        return max(0.2, abs(hold_val) * 0.2)
    if feat in ("ofi_h4_delta", "cvd", "cvd_momentum_adv", "Buy_Liq", "Sell_Liq"):
        return max(1e5, abs(hold_val) * 0.5)  # loose — expected diff
    if feat in ("rsi_h4", "stochrsi_d", "stochrsi_k", "rsi_6"):
        return max(8.0, abs(hold_val) * 0.25)
    return max(0.08, abs(hold_val) * 0.2) if hold_val else 0.1


def fetch_vps_features(symbols: list[str]) -> dict[str, dict]:
    """SSH: run inference on VPS, return last-bar features per coin."""
    sym_json = json.dumps(symbols)
    py = f"""
import json, sys, os
os.chdir('/home/swint/swint_tradev2')
sys.path.insert(0, '.')
from app.services.data_service import InferenceDataService, _get_positioning_mode
svc = InferenceDataService()
mode = _get_positioning_mode()
out = {{'positioning_mode': mode, 'coins': {{}}}}
for sym in {sym_json}:
    df = svc.prepare_latest_features(sym, n_bars=500)
    if df is None:
        out['coins'][sym] = None
        continue
    row = df.iloc[-1]
    d = {{}}
    for c in df.columns:
        v = row[c]
        if hasattr(v, 'item'):
            v = v.item()
        d[c] = None if (isinstance(v, float) and (v != v)) else v
    out['coins'][sym] = d
print(json.dumps(out))
"""
    r = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=25", VPS,
         "sudo -u swint /home/swint/swint_tradev2/.venv/bin/python -c " + repr(py)],
        capture_output=True, text=True, timeout=180,
    )
    if r.returncode != 0:
        print("VPS fetch stderr:", r.stderr[:500], file=sys.stderr)
        raise RuntimeError(f"VPS inference failed: {r.returncode}")
    # last line should be JSON
    line = [ln for ln in r.stdout.strip().splitlines() if ln.startswith("{")][-1]
    return json.loads(line)


def classify_feat(feat: str, deltas: list[float], matches: int, total: int) -> str:
    if feat in EXPECTED_DIFF:
        return "EXPECTED_DIFF"
    if feat == "hmm_regime_enc":
        return "REGIME" if matches / max(total, 1) < 0.6 else "MATCH"
    rate = matches / max(total, 1)
    if rate >= 0.75:
        return "MATCH"
    if rate >= 0.4:
        return "INVESTIGATE"
    return "MISMATCH"


def main():
    print("Pulling live DB...")
    pull_live_db()

    # Compare at holdout last bar (Jun 13 06:00 UTC — end of holdout)
    TS = pd.Timestamp("2026-06-13 06:00:00", tz="UTC")

    print("Fetching VPS pipeline (post-fix)...")
    vps = fetch_vps_features(COINS)
    print(f"VPS positioning_mode={vps.get('positioning_mode')}")

    # Per-feature aggregation: VPS now vs holdout @ TS
    feat_stats: dict[str, dict] = {f: {"match": 0, "total": 0, "deltas": []} for f in FEAT}

    print(f"\n=== VPS pipeline NOW vs Riset holdout @ {TS} ===\n")
    print(f"{'Coin':14s} {'Match':>5s} {'Miss':>5s} {'LSR_live':>9s} {'LSR_hold':>9s} {'HMM_l':>4s} {'HMM_h':>4s}")
    for sym in COINS:
        live = vps["coins"].get(sym)
        hold = holdout_bar(sym, TS)
        if live is None or hold is None:
            print(f"{sym:14s} SKIP (live={live is not None}, hold={hold is not None})")
            continue
        m, miss = 0, 0
        for f in FEAT:
            if f not in live or pd.isna(hold.get(f)):
                continue
            lv, hv = float(live[f]), float(hold[f])
            d = abs(lv - hv)
            t = tol(f, hv)
            feat_stats[f]["total"] += 1
            feat_stats[f]["deltas"].append(d)
            if d <= t:
                feat_stats[f]["match"] += 1
                m += 1
            else:
                miss += 1
        print(
            f"{sym:14s} {m:5d} {miss:5d} "
            f"{float(live.get('long_short_ratio', 0)):9.4f} {float(hold.get('long_short_ratio', 0)):9.4f} "
            f"{int(live.get('hmm_regime_enc', -1)):4d} {int(hold.get('hmm_regime_enc', -1)):4d}"
        )

    # Cannot compare VPS NOW to holdout Jun 13 at same wall clock — different dates!
    # Also run: VPS vs holdout at SAME calendar approach — use latest holdout bar for reference only
    print("\n[Catatan] VPS=now (Jun 18+), holdout=Jun 13 — perbandingan lintas waktu, bukan point-in-time.")

    # DB latest signals vs VPS inference cache
    print("\n=== DB signal snapshot vs VPS pipeline (same coin, approximate) ===\n")
    sig = load_signals(LOCAL_DB)
    sig = sig.sort_values("signal_time", ascending=False)
    db_lsr = []
    for sym in COINS:
        sub = sig[sig["coin_symbol"] == sym].head(1)
        if sub.empty:
            continue
        r = sub.iloc[0]
        fs = json.loads(r["feature_snapshot"] or "{}")
        live = vps["coins"].get(sym) or {}
        db_lsr.append(fs.get("long_short_ratio"))
        d_lsr = abs(float(fs.get("long_short_ratio") or 0) - float(live.get("long_short_ratio") or 0))
        print(
            f"{sym:14s} DB@{str(r['signal_time'])[:16]} LSR_db={fs.get('long_short_ratio')} "
            f"LSR_vps_now={live.get('long_short_ratio', '?'):.4f} delta={d_lsr:.4f} "
            f"HMM_db={fs.get('hmm_regime_enc')} HMM_vps={live.get('hmm_regime_enc')}"
        )

    # Summary table all 33 features
    print("\n=== RINGKASAN 33 FITUR (VPS now vs Holdout Jun13) ===\n")
    rows = []
    for f in FEAT:
        st = feat_stats[f]
        n, m = st["total"], st["match"]
        mean_d = float(np.mean(st["deltas"])) if st["deltas"] else 0
        status = classify_feat(f, st["deltas"], m, n)
        rows.append({"feature": f, "status": status, "match_rate": f"{100*m/max(n,1):.0f}%", "mean_delta": round(mean_d, 4), "n": n})
        print(f"{status:14s} | {f:28s} | match {m}/{n} | mean_delta={mean_d:.4g}")

    # Overall verdict
    match_n = sum(1 for r in rows if r["status"] == "MATCH")
    exp_n = sum(1 for r in rows if r["status"] == "EXPECTED_DIFF")
    inv_n = sum(1 for r in rows if r["status"] in ("INVESTIGATE", "MISMATCH", "REGIME"))
    print(f"\n>>> MATCH: {match_n}/33 | EXPECTED_DIFF: {exp_n}/33 | INVESTIGATE/MISMATCH/REGIME: {inv_n}/33")

    lsr_db_zero = sum(1 for x in db_lsr if x == 0 or x == 0.0)
    print(f">>> DB latest per coin: LSR=0 pada {lsr_db_zero}/{len(db_lsr)} koin (snapshot lama pre-fix)")

    out = ROOT / "reports" / "experiments" / "full_feature_parity.json"
    payload = {
        "vps_positioning_mode": vps.get("positioning_mode"),
        "holdout_ts": str(TS),
        "note": "VPS=current time; holdout=Jun13 bar — cross-time reference",
        "summary": {"match": match_n, "expected_diff": exp_n, "other": inv_n},
        "features": rows,
        "db_lsr_zero_coins": lsr_db_zero,
    }
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()