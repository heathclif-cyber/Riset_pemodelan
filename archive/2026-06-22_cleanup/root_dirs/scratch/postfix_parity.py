# -*- coding: utf-8 -*-
"""Post-fix only: apa yang MASIH beda antara live pipeline vs riset training."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import HOLDOUT_DIR
from core.utils import ensure_utc_index

FEAT = json.load(open(ROOT / "models/feature_cols_v2.json", encoding="utf-8"))
TRAIN = ROOT / "data/training/labeled"
HOLD = HOLDOUT_DIR / "labeled"

# VPS health snapshot (post-fix pipeline)
vps_path = ROOT / "scratch/vps_health_now.json"
vps = json.loads(vps_path.read_text(encoding="utf-8-sig")) if vps_path.exists() else None


def load_lsr_stats(base: Path, label: str) -> dict:
    lsrs = []
    for p in sorted(base.glob("*_features_v3.parquet")):
        df = ensure_utc_index(pd.read_parquet(p))
        if "long_short_ratio" in df.columns:
            lsrs.extend(df["long_short_ratio"].dropna().tolist())
    arr = np.array(lsrs)
    return {
        "label": label,
        "n": len(arr),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "p5": float(np.percentile(arr, 5)),
        "p95": float(np.percentile(arr, 95)),
    }


def vps_key_features() -> pd.DataFrame:
    rows = []
    for c in vps.get("coins", []):
        kf = c.get("key_features", {})
        row = {"symbol": c["symbol"], **kf}
        rows.append(row)
    return pd.DataFrame(rows)


def compare_distributions(train_df: pd.DataFrame, vps_df: pd.DataFrame, feats: list[str]) -> list[dict]:
    out = []
    for f in feats:
        if f not in train_df.columns:
            continue
        tr = train_df[f].dropna()
        if tr.empty:
            continue
        if f in vps_df.columns:
            lv = vps_df[f].dropna()
            live_mean = float(lv.mean()) if len(lv) else None
            live_std = float(lv.std()) if len(lv) > 1 else 0
        else:
            live_mean = live_std = None
        tr_mean, tr_std = float(tr.mean()), float(tr.std())
        if live_mean is None:
            status = "NO_LIVE"
        elif f in ("long_short_ratio",):
            # training sangat ketat ~1.0
            if abs(live_mean - tr_mean) > 0.15 or live_std > tr_std * 3:
                status = "STILL_WRONG"
            else:
                status = "OK"
        elif f in ("cvd", "ofi_h4_delta", "ofi_acceleration", "cvd_div_h4", "cvd_momentum_adv"):
            status = "EXPECTED_DIFF"
        elif f in ("rsi_h4", "stochrsi_d", "stochrsi_k"):
            if abs(live_mean - tr_mean) > 15:
                status = "STILL_WRONG"
            else:
                status = "TIMING_DIFF"
        elif f == "hmm_regime_enc":
            status = "OK" if live_std > 0.3 else "STILL_WRONG"
        else:
            rel = abs(live_mean - tr_mean) / max(abs(tr_mean), 1e-6)
            status = "OK" if rel < 0.5 else "CHECK"
        out.append({
            "feature": f,
            "status": status,
            "train_mean": round(tr_mean, 4),
            "train_std": round(tr_std, 4),
            "live_mean": round(live_mean, 4) if live_mean is not None else None,
            "live_std": round(live_std, 4) if live_std is not None else None,
        })
    return out


def sample_train_last_bars(n_coins: int = 21) -> pd.DataFrame:
    rows = []
    for p in sorted(TRAIN.glob("*_features_v3.parquet"))[:n_coins]:
        sym = p.stem.replace("_features_v3", "")
        df = ensure_utc_index(pd.read_parquet(p))
        r = df.iloc[-1]
        row = {"symbol": sym}
        for f in FEAT:
            if f in r.index:
                row[f] = r[f]
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    print("=== POST-FIX PARITY (bukan data lama) ===\n")

    lsr_train = load_lsr_stats(TRAIN, "training")
    lsr_hold = load_lsr_stats(HOLD, "holdout")
    print("LSR distribution riset:")
    for s in (lsr_train, lsr_hold):
        print(f"  {s['label']:8s} mean={s['mean']:.4f} std={s['std']:.4f} p5={s['p5']:.4f} p95={s['p95']:.4f}")

    if vps is None:
        print("VPS health tidak ada")
        return

    vdf = vps_key_features()
    lsr_live = vdf["long_short_ratio"].dropna()
    print(f"\nLSR live post-fix (21 koin): mean={lsr_live.mean():.4f} min={lsr_live.min():.4f} max={lsr_live.max():.4f}")
    print(f"  SOL (referensi parity): {float(vdf.loc[vdf.symbol=='SOLUSDT','long_short_ratio'].iloc[0]):.4f}")

    hmm = vdf["hmm_regime_enc"].value_counts().to_dict()
    print(f"HMM live post-fix dist: {hmm}")

    # Compare key feature means: live 21 coins vs training last-bar sample
    train_last = sample_train_last_bars()
    compare_feats = [
        "long_short_ratio", "hmm_regime_enc", "rsi_h4", "stochrsi_d", "h4_trend",
        "cvd", "ofi_h4_delta", "cvd_slope_h4", "whale_retail_divergence",
        "vol_price_confirm", "dist_liq_50x_long", "log_ret_20",
    ]
    rows = compare_distributions(train_last, vdf, compare_feats)

    print("\nFitur yang MASIH bermasalah post-fix:")
    still_wrong = [r for r in rows if r["status"] in ("STILL_WRONG", "CHECK")]
    ok = [r for r in rows if r["status"] in ("OK", "TIMING_DIFF", "EXPECTED_DIFF")]
    for r in still_wrong:
        print(f"  {r['feature']:28s} | train_mean={r['train_mean']} live_mean={r['live_mean']} | {r['status']}")
    print(f"\nSudah OK / expected diff: {len(ok)}/{len(rows)}")

    print("\nYang belum diverifikasi end-to-end:")
    print("  - Belum ada signal DB post-fix (cron HH:12 belum jalan setelah deploy)")
    print("  - Trade yang open sekarang masih pakai snapshot lama")

    out = ROOT / "reports/experiments/postfix_parity.json"
    payload = {
        "lsr_train": lsr_train,
        "lsr_holdout": lsr_hold,
        "lsr_live": {
            "mean": float(lsr_live.mean()),
            "min": float(lsr_live.min()),
            "max": float(lsr_live.max()),
            "values": lsr_live.to_dict(),
        },
        "hmm_live_dist": hmm,
        "feature_compare": rows,
        "still_wrong": still_wrong,
    }
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()