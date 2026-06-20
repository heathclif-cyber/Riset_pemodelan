# -*- coding: utf-8 -*-
"""
Diagnostic holdout ic32 B-dir-combined pada holdout rebuilt (positioning fix).
TIDAK menulis flag .holdout_b_dir_combined_evaluated — bukan eval resmi.
"""
from __future__ import annotations

import json
import sys
import warnings
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import importlib.util

from core.models import load_lstm
from core.utils import setup_logger, ensure_utc_index
from config import ALL_COINS, HOLDOUT_DIR, MODEL_DIR, GUARDIAN_DYNAMIC_FEATURES

logger = setup_logger("holdout_rebuild_diag")

_hmod_path = ROOT / "pipeline" / "07_holdout_ic32_b_dir_combined.py"
_spec = importlib.util.spec_from_file_location("holdout_ic32_mod", _hmod_path)
hmod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(hmod)

RUN_DIR = MODEL_DIR / "runs" / "ic32_regime_v1"
FROZEN_PATH = RUN_DIR / "b_dir_combined_frozen.json"
OLD_PATH = RUN_DIR / "holdout_b_dir_combined_apr_jun26.json"
OUT_PATH = ROOT / "reports" / "experiments" / "holdout_rebuild_positioning_diag.json"
HOLDOUT_MONTHS = 2.5


def _aggregate(all_trades: list) -> dict:
    n_total = len(all_trades)
    n_wins = sum(1 for t in all_trades if t.get("net_pnl", 0) > 0)
    total_pnl = sum(t.get("net_pnl", 0) for t in all_trades)
    long_trades = [t for t in all_trades if t.get("direction") == "LONG"]
    short_trades = [t for t in all_trades if t.get("direction") == "SHORT"]
    gpnl = sum(t["net_pnl"] for t in all_trades if t.get("net_pnl", 0) > 0)
    gloss = abs(sum(t["net_pnl"] for t in all_trades if t.get("net_pnl", 0) <= 0))
    pf = gpnl / gloss if gloss > 0 else float("inf")
    wr_pct = n_wins / n_total * 100
    outcome_counts = {}
    for t in all_trades:
        oc = t.get("outcome", "UNKNOWN")
        outcome_counts[oc] = outcome_counts.get(oc, 0) + 1
    gd_rate = sum(v for k, v in outcome_counts.items() if "GUARDIAN" in k) / n_total * 100
    return {
        "total_trades": n_total,
        "trades_per_month": round(n_total / HOLDOUT_MONTHS, 1),
        "win_rate": round(wr_pct, 2),
        "long_pct": round(len(long_trades) / n_total * 100, 2),
        "short_long_ratio": round(len(short_trades) / max(len(long_trades), 1), 3),
        "total_pnl": round(total_pnl, 2),
        "pnl_per_trade": round(total_pnl / n_total, 4),
        "profit_factor": round(pf, 3),
        "guardian_exit_pct": round(gd_rate, 2),
        "outcome_counts": outcome_counts,
    }


def _delta(new: dict, old: dict, key: str) -> float | None:
    if old is None or key not in old or key not in new:
        return None
    ov, nv = old[key], new[key]
    if isinstance(ov, (int, float)) and isinstance(nv, (int, float)):
        return round(nv - ov, 4)
    return None


def main():
    if not FROZEN_PATH.exists():
        raise FileNotFoundError(FROZEN_PATH)

    hmm_cfg = hmod._load_frozen_cfg()
    live_cfg = hmod._apply_live_config()

    with open(MODEL_DIR / "feature_cols_ic32_regime.json", encoding="utf-8") as f:
        feat_cols = json.load(f)
    with open(MODEL_DIR / "feature_cols_lstm_temporal.json", encoding="utf-8") as f:
        lstm_feat_cols = json.load(f)

    lgbm_model = joblib.load(RUN_DIR / "lgbm.pkl")
    lstm_model = load_lstm(MODEL_DIR / "lstm_best.pt", device="cpu")
    lstm_scaler = joblib.load(MODEL_DIR / "lstm_scaler.pkl")
    guardian_model = joblib.load(MODEL_DIR / "guardian_clean_v2.pkl")
    guardian_scaler = joblib.load(MODEL_DIR / "guardian_clean_v2_scaler.pkl")
    with open(MODEL_DIR / "guardian_clean_v2_feature_cols.json", encoding="utf-8") as f:
        guardian_feat_cols = json.load(f)
    g_static = [c for c in guardian_feat_cols if c not in set(GUARDIAN_DYNAMIC_FEATURES)]

    old_agg = None
    if OLD_PATH.exists():
        with open(OLD_PATH, encoding="utf-8") as f:
            old_agg = json.load(f).get("aggregate")

    print("=" * 80)
    print("  DIAGNOSTIC — ic32 holdout pada data rebuilt (positioning fix)")
    print("  BUKAN eval resmi — flag holdout tidak disentuh")
    print("=" * 80)

    results = {}
    all_trades = []
    failed = []

    for sym in ALL_COINS:
        try:
            r = hmod.backtest_coin(
                sym, hmm_cfg, live_cfg,
                lgbm_model, lstm_model, lstm_scaler, lstm_feat_cols,
                feat_cols, guardian_model, guardian_scaler, g_static,
            )
            if r is None:
                failed.append(sym)
                continue
            results[sym] = r
            trades = r.get("trades", [])
            all_trades.extend(trades)
            pnl = sum(t.get("net_pnl", 0) for t in trades)
            logger.info(
                f"  [{sym}] {r.get('total_trades', 0)} trades | "
                f"WR={r.get('winrate', 0)*100:.1f}% | PnL=${pnl:+.2f}"
            )
        except Exception as exc:
            logger.error(f"  [{sym}] {exc}")
            failed.append(sym)

    if not all_trades:
        raise RuntimeError("No holdout trades generated")

    new_agg = _aggregate(all_trades)

    print("\n--- NEW (rebuilt holdout) ---")
    for k, v in new_agg.items():
        if k != "outcome_counts":
            print(f"  {k}: {v}")

    if old_agg:
        print("\n--- OLD (pre-rebuild, official) ---")
        for k in ("total_trades", "win_rate", "profit_factor", "total_pnl", "pnl_per_trade"):
            print(f"  {k}: {old_agg.get(k)}")
        print("\n--- DELTA (new - old) ---")
        for k in ("total_trades", "win_rate", "profit_factor", "total_pnl", "pnl_per_trade"):
            d = _delta(new_agg, old_agg, k)
            if d is not None:
                print(f"  {k}: {d:+.4f}")

    # LSR coverage in holdout features
    lsr_stats = {}
    holdout_dir = HOLDOUT_DIR / "labeled"
    for sym in ALL_COINS[:5]:
        p = holdout_dir / f"{sym}_features_v3.parquet"
        if p.exists():
            df = ensure_utc_index(pd.read_parquet(p))
            if "long_short_ratio" in df.columns:
                s = pd.to_numeric(df["long_short_ratio"], errors="coerce")
                lsr_stats[sym] = {
                    "mean": round(float(s.mean()), 4),
                    "realish_pct": round(float((s > 1.5).mean() * 100), 2),
                }

    out = {
        "meta": {
            "type": "holdout_rebuild_positioning_diagnostic",
            "methodology": "NOT official holdout — do not use for model decisions",
            "holdout_period": "2026-04-01 to 2026-06-13",
            "frozen_config": str(FROZEN_PATH),
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "official_holdout_preserved": str(RUN_DIR / ".holdout_b_dir_combined_evaluated"),
        },
        "aggregate_new": new_agg,
        "aggregate_old_official": old_agg,
        "delta_new_minus_old": {
            k: _delta(new_agg, old_agg, k)
            for k in ("total_trades", "win_rate", "profit_factor", "total_pnl", "pnl_per_trade")
        },
        "lsr_sample_stats": lsr_stats,
        "per_coin": {
            sym: {
                "trades": r.get("total_trades", 0),
                "wr": round(r.get("winrate", 0) * 100, 2),
                "pnl": round(sum(t.get("net_pnl", 0) for t in r.get("trades", [])), 2),
            }
            for sym, r in results.items()
        },
        "failed": failed,
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print(f"\nSaved -> {OUT_PATH}")
    print("=" * 80)


if __name__ == "__main__":
    main()