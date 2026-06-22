# -*- coding: utf-8 -*-
"""Bandingkan inference_config Riset vs VPS live."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RISET_CFG = REPO / "models" / "inference_config.json"
FROZEN = REPO / "models" / "runs" / "ic32_regime_v1" / "b_dir_combined_frozen.json"
VPS_HOST = "root@139.180.157.176"
VPS_CFG = "/home/swint/swint_tradev2/models/inference_config.json"

# Keys yang wajib match deploy B-dir-combined
CHECK_PATHS = [
    ("model_version", None),
    ("model_type", None),
    ("cascade.mode", "cascade"),
    ("cascade.lgbm_threshold_long", "cascade"),
    ("cascade.lgbm_threshold_short", "cascade"),
    ("cascade.lstm_confirmation_enabled", "cascade"),
    ("cascade.lstm_adjust_opposite_pen", "cascade"),
    ("inference.confidence_threshold_entry", "inference"),
    ("hmm.enabled", "hmm"),
    ("hmm.per_state_thresholds", "hmm"),
    ("guardian.enabled", "guardian"),
    ("guardian.exit_threshold", "guardian"),
    ("guardian.min_hold_bars", "guardian"),
    ("rr_gate.enabled", "rr_gate"),
    ("rr_gate.sl_trigger_mode", "rr_gate"),
    ("rr_gate.min_rr", "rr_gate"),
    ("regime_alignment.enabled", "regime_alignment"),
    ("structural_filter.enabled", "structural_filter"),
    ("volatility_circuit_breaker.enabled", "volatility_circuit_breaker"),
    ("tp_sl.sl_atr_mult", "tp_sl"),
    ("tp_sl.tp_atr_mult", "tp_sl"),
    ("risk.modal_per_trade", "risk"),
    ("risk.max_open_positions", "risk"),
    ("risk.daily_loss_limit", "risk"),
    ("risk.leverage_recommended", "risk"),
    ("feature_engineering.positioning_mode", "feature_engineering"),
    ("models.lgbm", "models"),
    ("models.guardian", "models"),
    ("models.lstm", "models"),
]


def get_nested(d: dict, path: str):
    cur = d
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def fetch_vps_config() -> dict:
    cmd = [
        "ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=25",
        VPS_HOST, f"cat {VPS_CFG}",
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=40)
    if r.returncode != 0:
        print(f"[ERROR] SSH gagal: {r.stderr or r.stdout}")
        sys.exit(1)
    return json.loads(r.stdout)


def main():
    riset = json.loads(RISET_CFG.read_text(encoding="utf-8"))
    frozen = json.loads(FROZEN.read_text(encoding="utf-8"))
    live = fetch_vps_config()

    print("=" * 60)
    print(" PERBANDINGAN Riset vs VPS Live")
    print("=" * 60)

    ok = 0
    fail = 0
    warn = 0

    for path, _ in CHECK_PATHS:
        rv = get_nested(riset, path)
        lv = get_nested(live, path)
        if rv == lv:
            status = "OK"
            ok += 1
        else:
            status = "MISMATCH"
            fail += 1
        print(f"  [{status:8}] {path}")
        if rv != lv:
            print(f"             Riset: {rv!r}")
            print(f"             Live:  {lv!r}")

    # Frozen HMM vs live
    f_thr = frozen.get("per_state_thresholds")
    l_thr = get_nested(live, "hmm.per_state_thresholds")
    print()
    print("--- Frozen b_dir_combined vs Live HMM ---")
    if json.dumps(f_thr, sort_keys=True) == json.dumps(l_thr, sort_keys=True):
        print("  [OK] per_state_thresholds match frozen")
        ok += 1
    else:
        print("  [MISMATCH] per_state_thresholds")
        print(f"    Frozen: {f_thr}")
        print(f"    Live:   {l_thr}")
        fail += 1

    # Operational keys (boleh beda jika di-set via UI VPS)
    print()
    print("--- Operasional (boleh beda dari Riset default) ---")
    for path in ["risk.modal_per_trade", "risk.leverage_recommended", "trading_mode"]:
        rv = get_nested(riset, path)
        lv = get_nested(live, path)
        if rv == lv:
            print(f"  [OK] {path} = {lv}")
        else:
            print(f"  [INFO] {path}: Riset={rv} Live={lv}")
            warn += 1

    print()
    print("=" * 60)
    print(f" Hasil: {ok} OK, {fail} MISMATCH, {warn} operasional beda")
    if fail == 0:
        print(" LIVE SESUAI dengan config Riset (deploy target)")
    else:
        print(" ADA PERBEDAAN — perlu sinkron ulang")
    print("=" * 60)
    return 1 if fail else 0


if __name__ == "__main__":
    raise SystemExit(main())