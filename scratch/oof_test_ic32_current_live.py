#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
scratch/oof_test_ic32_current_live.py

OOF Test for the EXACT current live ic32_regime_v1 config
(Pre-dynamic size, from 2026-06-06 snapshot).

Matches the live snapshot the user is running right now:
- LGBM 33 feats + thr 0.69/0.59 + conf 0.59
- hard_consensus LSTM (opposite_pen 0.65, etc.)
- regime FLIP (regime_alignment)
- Guardian clean_v2 (exit 0.65, min_hold 2)
- structural_filter, rr_gate, vol circuit breaker
- NO dynamic sizing (fixed modal $10 / 5x)
- Positioning data mining enabled in config

This is the OOF baseline for the pre-dynsize ic32 that the user confirmed is currently live
(and for which no OOF test had been done yet).

Run only after the plan is written in EXPERIMENTS.md (2026-06-17 section).

Usage examples:
  python scratch/oof_test_ic32_current_live.py --use-holdout-data
  python scratch/oof_test_ic32_current_live.py --data data/holdout/2026-04-01_to_2026-06-13_features.parquet

It will produce a scorecard + oof_simulated_trades.parquet using the exact live parameters.
"""

import argparse
import json
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

# === EXACT LIVE CONFIG (copy of the 2026-06-06 snapshot the user is running) ===
# Thresholds, fusion, FLIP, Guardian, filters — all from the pasted inference_config
LIVE_CFG = {
    "lgbm_threshold_long": 0.69,
    "lgbm_threshold_short": 0.59,
    "confidence_threshold_entry": 0.59,
    "lstm_fusion_mode": "hard_consensus",
    "lstm_adjust_opposite_pen": 0.65,
    "lstm_adjust_agree_boost": 0.05,
    "lstm_directional_review_threshold": 0.35,
    "lstm_confirmation_enabled": True,
    "lstm_flat_review_enabled": True,
    "lstm_no_veto_threshold": 0.5,
    "regime_alignment": {
        "enabled": True,
        "ranging": {"counter_trend_boost": 0.05, "with_trend_penalty": 0.1},
        "trending": {"counter_trend_penalty": 0.05, "with_trend_boost": 0.1}
    },
    "guardian": {
        "exit_threshold": 0.65,
        "min_hold_bars": 2,
        "partial_exit_ratio": 0.5,
        "activation_atr": 0
    },
    "structural_filter": {
        "enabled": True,
        "max_swing_deviation_pct": 0.15,
        "require_entry_in_swing_range": True,
        "swing_max_age_hours": 48,
        "breakout_tolerance_pct": 0.03
    },
    "rr_gate": {
        "enabled": True,
        "min_rr": 0.6,
        "min_tp_atr": 1.2,
        "max_sl_atr": 4,
        "swing_bumper_atr": 0.5
    },
    "volatility_circuit_breaker": {
        "enabled": True,
        "atr_multiplier": 3,
        "lookback_bars": 24
    },
    "tp_sl": {
        "tp_atr_mult": 2.0,
        "sl_atr_mult": 1.5,
        "min_rr": 0.6,
        "min_tp_atr": 1.2,
        "max_sl_atr": 4
    },
    "risk": {
        "modal_per_trade": 10,
        "leverage_recommended": 5
    },
    "note": "NO dynamic sizing. Exact pre-dynsize ic32 snapshot the user is currently running live."
}

IC32_RUN = Path("models/runs/ic32_regime_v1")
LIVE_RUN = Path("models/runs/ic32_live_current_oof")   # we prepared the exact config here

def load_live_config():
    cfg_path = LIVE_RUN / "inference_config_live_snapshot.json"
    if cfg_path.exists():
        with open(cfg_path) as f:
            return json.load(f)
    return LIVE_CFG  # fallback to the dict above

def load_ic32_lgbm():
    p = IC32_RUN / "lgbm.pkl"
    if not p.exists():
        p = Path("models/lgbm_baseline.pkl")
    print(f"[ic32-oof] Loading LGBM from {p}")
    return joblib.load(p)

def main():
    parser = argparse.ArgumentParser(description="OOF test for current live ic32_regime_v1 (pre-dynsize)")
    parser.add_argument("--data", type=str, default=None, help="Parquet with features + labels for OOF period")
    parser.add_argument("--use-holdout-data", action="store_true", help="Use holdout_trade_history.csv from the ic32 run")
    args = parser.parse_args()

    print("="*70)
    print("OOF TEST — EXACT CURRENT LIVE ic32_regime_v1 (PRE-DYNAMIC SIZE)")
    print("User confirmation: no dynamic size entry, OOF not done yet for this live config")
    print("="*70)

    cfg = load_live_config()
    print("Using thresholds:", cfg.get("cascade", {}).get("lgbm_threshold_long"), 
          cfg.get("cascade", {}).get("lgbm_threshold_short"))
    print("Note:", cfg.get("note", LIVE_CFG["note"]))

    lgbm = load_ic32_lgbm()

    if args.use_holdout_data:
        holdout_csv = IC32_RUN / "holdout_trade_history.csv"
        print(f"Loading archived holdout data: {holdout_csv}")
        df = pd.read_csv(holdout_csv)
        print("Holdout columns sample:", list(df.columns)[:12])
        print("Rows:", len(df))
        # The csv contains pre-simulated results. 
        # For true "exact live config" we ideally re-apply the full inference (LGBM proba -> thr -> hard LSTM -> FLIP -> Guardian + filters)
        # on the raw features if they are in the csv or if we have the raw OOF bars.
        print("\n[TODO] Re-apply the EXACT live stack (thresholds + hard_consensus + FLIP + Guardian + structural/rr/vol filters)")
        print("on top of the raw OOF predictions or the holdout features.")
    else:
        print("Provide --data path to a features parquet that has the 33 ic32 LGBM features + regime + any needed dynamic fields.")
        print("The script will load the trained ic32 LGBM, get proba on the data, apply the live thresholds/fusion/filters, and simulate trades.")

    print("\nPrepared skeleton complete.")
    print("Next: implement the full simulation loop using the live config values above.")
    print("Target output: OOF scorecard (WR, PF, PnL $10/5x, trades, SL rate, Guardian %, max consec loss, per direction, per coin).")
    print("Compare against the archived numbers in the ic32_regime_v1 run and against the recent TB live disaster.")

if __name__ == "__main__":
    main()
