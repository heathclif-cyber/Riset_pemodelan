"""
05q — Compare tb_lstm_genuine_v2 vs tb_lstm_lgbm_seq_v1 + single-param tweaks.

Phase 1: slice comparison (signal-level, frozen 05j winner config)
Phase 2: tweak one param at a time on lgbm_seq LSTM, pipeline PPT_norm

Genuine OOF only. Present before promote.

Usage:
  python pipeline/05q_lstm_lgbm_seq_compare_tweak.py
  python pipeline/05q_lstm_lgbm_seq_compare_tweak.py --tweaks-only
"""
import argparse
import copy
import json
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from config import MODEL_DIR, MODAL_PER_TRADE
from core.cascade_utils import apply_conditional_momentum_fusion_pre
from pipeline.lstm_fusion_shared import (
    LGBM_DIR, GUARDIAN_DIR, DYNAMIC_FEATS, FLAT, SHORT,
    attach_cross_section, build_y_pred,
    load_guardian_params, load_hmm_cfg, preload_coins,
    precompute_cross_section,
)

import importlib.util
_spec_path = ROOT / "pipeline" / "05j_lstm_conditional_momentum_eval.py"
_spec = importlib.util.spec_from_file_location("eval05j", _spec_path)
_eval05j = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_eval05j)
eval_pipeline = _eval05j.eval_pipeline

REF_RUN = "tb_lstm_genuine_v2"
SEQ_RUN = "tb_lstm_lgbm_seq_v1"
OUT_PATH = LGBM_DIR / "lstm_lgbm_seq_compare_tweak.json"
VOL_THR = 2.0
BULL_THR = 0.38

FROZEN = {
    "fusion": "lstm",
    "mode": "conditional_momentum",
    "bull_thr": 0.38,
    "bear_thr": 0.50,
    "boost": 0.10,
    "opposite_pen": 0.14,
    "near_miss_gap": 0.03,
    "vol_thr": VOL_THR,
    "proportional": True,
    "enable_boost": True,
    "enable_penalty": True,
    "label": "ref_05j_frozen",
}

# Tweaks applied one-at-a-time on SEQ_RUN only (baseline = FROZEN)
TWEAKS = [
    {"label": "tweak_opp_pen_16", "opposite_pen": 0.16},
    {"label": "tweak_opp_pen_18", "opposite_pen": 0.18},
    {"label": "tweak_bear_thr_55", "bear_thr": 0.55},
    {"label": "tweak_bear_thr_58", "bear_thr": 0.58},
    {"label": "tweak_bull_thr_40", "bull_thr": 0.40},
    {"label": "tweak_boost_08", "boost": 0.08},
    {"label": "tweak_near_gap_05", "near_miss_gap": 0.05},
    {"label": "tweak_pen_only_opp16", "opposite_pen": 0.16, "enable_boost": False},
]


def load_coins_for_run(lstm_run: str) -> list:
    lgbm_oof = pd.read_parquet(LGBM_DIR / "oof_predictions.parquet")
    lstm_dir = MODEL_DIR / "runs" / lstm_run
    lstm_oof = pd.read_parquet(lstm_dir / "oof_lstm_predictions.parquet")
    return preload_coins(lgbm_oof, lstm_oof)


def slice_stats(coins: list, cfg: dict, hmm_cfg: dict, cs: dict) -> dict:
    attach_cross_section(coins, cs)
    y_base = {c["sym"]: build_y_pred(c, {"fusion": "baseline"}, hmm_cfg) for c in coins}
    y_fused = {c["sym"]: build_y_pred(c, cfg, hmm_cfg) for c in coins}

    boost = pen = 0
    conflict_total = conflict_penalized = 0
    rally_short_base = rally_short_fused = 0
    pump_short_base = pump_short_fused = 0
    lstm_bull_on_short = 0

    for c in coins:
        sym = c["sym"]
        yb = y_base[sym]
        yf = y_fused[sym]
        mom = c["vol_spike"] >= VOL_THR
        rally = c["frac_up"] >= 0.8

        p0, p2 = c["p0"], c["p2"]
        lstm_p = c["lstm_p"]
        lstm_bull = lstm_p[:, 2] >= BULL_THR
        lstm_bear = lstm_p[:, 0] >= cfg.get("bear_thr", 0.50)

        for i in range(len(yb)):
            if not mom[i] or not c["lstm_valid"][i]:
                continue
            if yb[i] == FLAT and yf[i] != FLAT:
                boost += 1
            if yb[i] != FLAT and yf[i] == FLAT:
                pen += 1
            if yb[i] == SHORT:
                if lstm_bull[i]:
                    conflict_total += 1
                    if yf[i] == FLAT:
                        conflict_penalized += 1
                    lstm_bull_on_short += 1
            if rally[i]:
                if yb[i] == SHORT:
                    rally_short_base += 1
                if yf[i] == SHORT:
                    rally_short_fused += 1
            if mom[i] and yb[i] == SHORT:
                pump_short_base += 1
                if yf[i] == SHORT:
                    pump_short_fused += 1

    n_short_base = sum(int((y_base[c["sym"]] == SHORT).sum()) for c in coins)
    n_short_fused = sum(int((y_fused[c["sym"]] == SHORT).sum()) for c in coins)

    return {
        "boost_unlock": boost,
        "penalty_block": pen,
        "n_short_base": n_short_base,
        "n_short_fused": n_short_fused,
        "conflict_lstm_bull_on_lgbm_short": conflict_total,
        "conflict_penalized_to_flat": conflict_penalized,
        "conflict_pen_pct": round(conflict_penalized / max(conflict_total, 1) * 100, 1),
        "rally_short_base": rally_short_base,
        "rally_short_fused": rally_short_fused,
        "rally_short_delta": rally_short_fused - rally_short_base,
        "pump_short_base": pump_short_base,
        "pump_short_fused": pump_short_fused,
        "pump_short_delta": pump_short_fused - pump_short_base,
    }


def run_pipeline_cfg(coins, cfg, hmm_cfg, g_model, g_scaler, g_params) -> dict:
    base_cfg = {"fusion": "baseline"}
    base_keys = set()
    for c in coins:
        yb = build_y_pred(c, base_cfg, hmm_cfg)
        for i, sig in enumerate(yb):
            if sig != FLAT:
                base_keys.add((c["sym"], i))

    rep = eval_pipeline(coins, cfg, hmm_cfg, g_model, g_scaler, g_params, base_keys)
    base_rep = eval_pipeline(coins, base_cfg, hmm_cfg, g_model, g_scaler, g_params, base_keys)
    rep["delta_port_ppt"] = rep["portfolio"]["ppt_norm"] - base_rep["portfolio"]["ppt_norm"]
    return rep


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tweaks-only", action="store_true")
    args = parser.parse_args()

    t0 = time.time()
    hmm_cfg = load_hmm_cfg()
    g_params = load_guardian_params()

    print("\n" + "=" * 78)
    print("  05q: LSTM ref vs LGBM-seq compare + single-param tweaks")
    print("=" * 78)

    slices = {}
    if not args.tweaks_only:
        print("\n  PHASE 1 — Slice comparison (frozen 05j winner config)")
        coins_ref = load_coins_for_run(REF_RUN)
        coins_seq = load_coins_for_run(SEQ_RUN)
        cs_ref = precompute_cross_section(coins_ref)
        cs_seq = precompute_cross_section(coins_seq)

        for name, coins, cs in [
            ("ref_" + REF_RUN, coins_ref, cs_ref),
            ("seq_" + SEQ_RUN, coins_seq, cs_seq),
        ]:
            slices[name] = slice_stats(coins, FROZEN, hmm_cfg, cs)

        print(f"\n  {'Model':<28} {'boost':>6} {'pen':>6} {'conflict':>9} {'pen%':>6} "
              f"{'rally_S':>8} {'pump_S d':>8}")
        for name, s in slices.items():
            print(
                f"  {name:<28} {s['boost_unlock']:>6} {s['penalty_block']:>6} "
                f"{s['conflict_lstm_bull_on_lgbm_short']:>9} {s['conflict_pen_pct']:>5.1f}% "
                f"{s['rally_short_fused']:>8} {s['pump_short_delta']:>+8}"
            )

        print("\n  Interpretasi slice:")
        print("    conflict = LGBM SHORT + LSTM bull>=0.38 pada vol>=2")
        print("    pen%     = % conflict yang berhasil di-block ke FLAT")
        print("    rally_S  = SHORT signals saat frac_up>=0.8 (wide pump)")
        print("    pump_S d = delta SHORT di vol>=2 bars (fused vs base)")

    # Phase 2: tweaks on SEQ only
    print("\n  PHASE 2 — Single-param tweaks (LSTM=tb_lstm_lgbm_seq_v1)")
    g_model = joblib.load(GUARDIAN_DIR / "guardian.pkl")
    g_scaler = joblib.load(GUARDIAN_DIR / "guardian_scaler.pkl")
    with open(GUARDIAN_DIR / "guardian_features.json", encoding="utf-8") as f:
        g_static = [feat for feat in json.load(f) if feat not in DYNAMIC_FEATS]
    lgbm_oof = pd.read_parquet(LGBM_DIR / "oof_predictions.parquet")
    lstm_oof = pd.read_parquet(MODEL_DIR / "runs" / SEQ_RUN / "oof_lstm_predictions.parquet")
    coins_seq = preload_coins(lgbm_oof, lstm_oof, g_static)

    ref_pipeline = run_pipeline_cfg(coins_seq, FROZEN, hmm_cfg, g_model, g_scaler, g_params)
    ref_ppt = ref_pipeline["portfolio"]["ppt_norm"]
    ref_delta = ref_pipeline["delta_port_ppt"]

    print(f"\n  Baseline seq+frozen: port PPT_norm={ref_ppt:.4f} delta={ref_delta:+.4f}")
    print(f"  {'Tweak':<28} {'dPPT':>8} {'pen_blk':>8} {'boost':>6} {'newMom':>7} {'PASS':>5}")

    tweak_results = [{
        "label": "seq_frozen_baseline",
        **FROZEN,
        "pipeline": ref_pipeline,
        "delta_port_ppt": ref_delta,
    }]

    cs = precompute_cross_section(coins_seq)
    for tw in TWEAKS:
        cfg = {**FROZEN, **tw}
        cfg["label"] = tw["label"]
        sig = slice_stats(coins_seq, cfg, hmm_cfg, cs)
        pipe = run_pipeline_cfg(coins_seq, cfg, hmm_cfg, g_model, g_scaler, g_params)
        d = pipe["delta_port_ppt"]
        passes = d >= ref_delta - 1e-6 and d > 0
        row = {
            **tw,
            "slice": sig,
            "pipeline": {
                "portfolio": pipe["portfolio"],
                "new_momentum": pipe["new_momentum"],
                "n_new_momentum": pipe["n_new_momentum"],
            },
            "delta_port_ppt": d,
            "delta_vs_seq_frozen": d - ref_delta,
            "passes": passes,
        }
        tweak_results.append(row)
        print(
            f"  {tw['label']:<28} {d:>+8.4f} {sig['penalty_block']:>8} "
            f"{sig['boost_unlock']:>6} {pipe['n_new_momentum']:>7} {'Y' if passes else 'N':>5}"
        )

    best = max(tweak_results[1:], key=lambda r: r["delta_port_ppt"])
    out = {
        "created": datetime.now().isoformat(),
        "frozen_config": FROZEN,
        "ref_run": REF_RUN,
        "seq_run": SEQ_RUN,
        "phase1_slices": slices or None,
        "seq_frozen_baseline": {
            "delta_port_ppt": ref_delta,
            "portfolio_ppt_norm": ref_ppt,
        },
        "ref_05j_original": {
            "delta_port_ppt": 0.0007515894262979383,
            "note": "tb_lstm_genuine_v2 best from lstm_conditional_momentum_eval.json",
        },
        "tweak_results": tweak_results,
        "best_tweak": best["label"],
        "best_delta_port_ppt": best["delta_port_ppt"],
        "elapsed_sec": round(time.time() - t0, 1),
    }
    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2, default=str)

    print(f"\n  Best tweak: {best['label']} delta={best['delta_port_ppt']:+.4f}")
    print(f"  vs ref genuine_v2 (+0.00075): {best['delta_port_ppt'] - 0.0007515894262979383:+.4f}")
    print(f"  Saved: {OUT_PATH}")
    print("=" * 78 + "\n")


if __name__ == "__main__":
    main()