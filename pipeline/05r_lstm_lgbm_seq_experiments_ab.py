"""
05r — Experiments A & B for tb_lstm_lgbm_seq_v1 fusion tweaks.

Exp A: Full 05j-style sweep, opposite_pen FIXED at 0.18
Exp B: Full sweep, opposite_pen=0.18 + bear_thr=0.55 FIXED

Genuine OOF, HMM Config B frozen, holdout sealed.

Usage:
  python pipeline/05r_lstm_lgbm_seq_experiments_ab.py
  python pipeline/05r_lstm_lgbm_seq_experiments_ab.py --exp A
  python pipeline/05r_lstm_lgbm_seq_experiments_ab.py --exp B
"""
import argparse
import importlib.util
import itertools
import json
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from pipeline.lstm_fusion_shared import (
    LGBM_DIR, GUARDIAN_DIR, DYNAMIC_FEATS, FLAT,
    build_y_pred, config_label, genuine_audit_block,
    load_guardian_params, load_hmm_cfg, preload_coins,
)

_spec = importlib.util.spec_from_file_location(
    "eval05j", ROOT / "pipeline" / "05j_lstm_conditional_momentum_eval.py"
)
_eval05j = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_eval05j)

LSTM_RUN = "tb_lstm_lgbm_seq_v1"
REF_DELTA = 0.0007515894262979383
MOMENTUM_VOL_THR = 2.0
TOP_K_PIPELINE = 10

BULL_THR = [0.38, 0.42, 0.46]
BEAR_THR_FULL = [0.50, 0.55, 0.58]
BOOST = [0.06, 0.08, 0.10]
NEAR_GAP = [0.03, 0.05, 0.08]
MODES = [
    {"enable_boost": True, "enable_penalty": True},
    {"enable_boost": True, "enable_penalty": False},
    {"enable_boost": False, "enable_penalty": True},
]


def build_sweep_configs(exp: str) -> list[dict]:
    opp_pen = [0.18]
    bear_thr = [0.55] if exp == "B" else BEAR_THR_FULL

    configs = [{"fusion": "baseline", "label": "baseline_no_lstm"}]
    for mode, bu, be, b, o, g in itertools.product(MODES, BULL_THR, bear_thr, BOOST, opp_pen, NEAR_GAP):
        if mode["enable_penalty"] is False and o != opp_pen[0]:
            continue
        if mode["enable_boost"] is False and b != BOOST[0]:
            continue
        cfg = {
            "fusion": "lstm",
            "mode": "conditional_momentum",
            "bull_thr": bu,
            "bear_thr": be,
            "boost": b,
            "opposite_pen": o,
            "near_miss_gap": g,
            "vol_thr": MOMENTUM_VOL_THR,
            "proportional": True,
            **mode,
        }
        cfg["label"] = config_label(cfg)
        configs.append(cfg)
    return configs


def run_experiment(exp: str, coins, coins_p, hmm_cfg, g_model, g_scaler, g_params, base_keys) -> dict:
    configs = build_sweep_configs(exp)
    y_base = {c["sym"]: build_y_pred(c, {"fusion": "baseline"}, hmm_cfg) for c in coins}
    base_sig = _eval05j.signal_stats(coins, y_base, y_base, hmm_cfg)
    base_dir = base_sig["n_dir"]

    signal_rows = [{"fusion": "baseline", "label": "baseline_no_lstm", **base_sig, "rank": 0.0}]
    for cfg in [c for c in configs if c["fusion"] != "baseline"]:
        y_map = {c["sym"]: build_y_pred(c, cfg, hmm_cfg) for c in coins}
        sig = _eval05j.signal_stats(coins, y_map, y_base, hmm_cfg)
        row = {"label": cfg["label"], **cfg, **sig, "rank": _eval05j.rank_signal({**sig, **cfg}, base_dir)}
        signal_rows.append(row)
    signal_rows.sort(key=lambda x: x["rank"], reverse=True)
    top_signal = [r for r in signal_rows if r.get("fusion") != "baseline"][:TOP_K_PIPELINE]

    pipe_candidates = [{"fusion": "baseline", "label": "baseline_no_lstm"}] + [
        {k: v for k, v in r.items() if k in (
            "fusion", "mode", "label", "bull_thr", "bear_thr", "boost", "opposite_pen",
            "near_miss_gap", "vol_thr", "proportional", "enable_boost", "enable_penalty",
        )} for r in top_signal
    ]

    pipe_results = []
    baseline_pipe = None
    print(f"\n  EXP {exp} — {len(configs)} signal configs, pipeline top-{len(pipe_candidates)-1}")
    if exp == "A":
        print("  Fixed: opposite_pen=0.18 | sweep bull/bear/boost/near_gap/modes")
    else:
        print("  Fixed: opposite_pen=0.18, bear_thr=0.55 | sweep bull/boost/near_gap/modes")

    for i, cfg in enumerate(pipe_candidates):
        t_c = time.time()
        met = _eval05j.eval_pipeline(coins_p, cfg, hmm_cfg, g_model, g_scaler, g_params, base_keys)
        row = {"label": cfg["label"], **cfg, **met}
        if cfg.get("fusion") == "baseline":
            baseline_pipe = row
        elif baseline_pipe:
            row["delta_port_ppt"] = met["portfolio"]["ppt_norm"] - baseline_pipe["portfolio"]["ppt_norm"]
            row["delta_new_mom_ppt"] = (
                met["new_momentum"]["ppt_norm"] - baseline_pipe["new_momentum"]["ppt_norm"]
                if met["n_new_momentum"] > 0 else None
            )
            row["passes"] = (
                met["new_momentum"]["n"] >= 30
                and met["new_momentum"]["ppt_norm"] > 0
                and met["portfolio"]["ppt_norm"] >= baseline_pipe["portfolio"]["ppt_norm"] - 0.003
            )
            row["beats_ref"] = row["delta_port_ppt"] >= REF_DELTA - 1e-6
        pipe_results.append(row)
        dt = time.time() - t_c
        if cfg.get("fusion") != "baseline":
            print(
                f"  [{i}] {cfg['label'][:55]}"
                f"\n      port={met['portfolio']['ppt_norm']:+.4f} "
                f"dPort={row.get('delta_port_ppt', 0):+.4f} "
                f"newMom={met['n_new_momentum']} ({dt:.0f}s)"
            )

    ranked = sorted(
        [r for r in pipe_results if r.get("fusion") != "baseline"],
        key=lambda x: (x.get("delta_port_ppt", -999), x["portfolio"]["ppt_norm"]),
        reverse=True,
    )
    winners = [r for r in pipe_results if r.get("passes")]

    print(f"\n  EXP {exp} TOP 5 by delta_port_ppt:")
    print(f"  {'Label':<42} {'dPort':>8} {'portPPT':>9} {'newMom':>7} {'beats_ref':>9}")
    for r in ranked[:5]:
        print(
            f"  {r['label']:<42} {r.get('delta_port_ppt', 0):>+8.4f} "
            f"{r['portfolio']['ppt_norm']:>+9.4f} {r.get('n_new_momentum', 0):>7} "
            f"{'Y' if r.get('beats_ref') else 'N':>9}"
        )

    best = ranked[0] if ranked else None
    return {
        "experiment": exp,
        "description": (
            "opp_pen=0.18 fixed, full 05j sweep other params"
            if exp == "A" else
            "opp_pen=0.18 + bear_thr=0.55 fixed, sweep other params"
        ),
        "n_signal_configs": len(configs),
        "baseline_signal": base_sig,
        "top_signal": top_signal,
        "baseline_pipeline": baseline_pipe,
        "pipeline_results": pipe_results,
        "winners": winners,
        "best": best,
        "decision": "PROMOTE_CANDIDATE" if winners and best and best.get("beats_ref") else (
            "PROMOTE_CANDIDATE" if winners else "NO_PROMOTE"
        ),
        "beats_ref_genuine_v2": bool(best and best.get("beats_ref")),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", choices=["A", "B", "both"], default="both")
    args = parser.parse_args()
    exps = ["A", "B"] if args.exp == "both" else [args.exp]

    t0 = time.time()
    hmm_cfg = load_hmm_cfg()
    g_params = load_guardian_params()
    lstm_dir = ROOT / "models" / "runs" / LSTM_RUN

    print("\n" + "=" * 78)
    print(f"  05r: LGBM-seq fusion experiments A/B | LSTM={LSTM_RUN}")
    print(f"  Ref delta target: {REF_DELTA:+.4f} (tb_lstm_genuine_v2)")
    print("=" * 78)

    lgbm_oof = pd.read_parquet(LGBM_DIR / "oof_predictions.parquet")
    lstm_oof = pd.read_parquet(lstm_dir / "oof_lstm_predictions.parquet")
    coins = preload_coins(lgbm_oof, lstm_oof)

    g_model = joblib.load(GUARDIAN_DIR / "guardian.pkl")
    g_scaler = joblib.load(GUARDIAN_DIR / "guardian_scaler.pkl")
    with open(GUARDIAN_DIR / "guardian_features.json", encoding="utf-8") as f:
        g_static = [f for f in json.load(f) if f not in DYNAMIC_FEATS]
    coins_p = preload_coins(lgbm_oof, lstm_oof, g_static)

    base_keys = set()
    for c in coins_p:
        y = build_y_pred(c, {"fusion": "baseline"}, hmm_cfg)
        for i in range(len(y)):
            if y[i] != FLAT:
                base_keys.add((c["sym"], i))

    results = {}
    for exp in exps:
        results[exp] = run_experiment(exp, coins, coins_p, hmm_cfg, g_model, g_scaler, g_params, base_keys)

    audit = genuine_audit_block()
    audit["lstm_source"] = f"{LSTM_RUN}/oof_lstm_predictions.parquet"
    out = {
        **audit,
        "lstm_run": LSTM_RUN,
        "ref_delta_tb_lstm_genuine_v2": REF_DELTA,
        "created": datetime.now().isoformat(),
        "elapsed_sec": round(time.time() - t0, 1),
        "experiments": results,
    }

    out_path = LGBM_DIR / "lstm_lgbm_seq_experiments_ab.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=str)

    print("\n" + "=" * 78)
    print("  SUMMARY")
    for exp in exps:
        r = results[exp]
        b = r.get("best") or {}
        print(
            f"  Exp {exp}: decision={r['decision']} | best={b.get('label', 'n/a')[:40]} "
            f"dPort={b.get('delta_port_ppt', 0):+.4f} beats_ref={r.get('beats_ref_genuine_v2')}"
        )
    print(f"  Saved: {out_path}")
    print("=" * 78 + "\n")


if __name__ == "__main__":
    main()