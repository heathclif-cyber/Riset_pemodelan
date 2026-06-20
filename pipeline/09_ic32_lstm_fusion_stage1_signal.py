"""
Stage 1 -- ic32 LGBM+LSTM fusion: SIGNAL-ONLY OOF sweep.

Genuine: OOF only, B-dir frozen, holdout NOT touched.
Output: top candidates for Stage 2 pipeline eval.

Usage:
  python pipeline/09_ic32_lstm_fusion_stage1_signal.py
"""
import itertools
import json
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from pipeline.ic32_fusion_shared import (
    IC32_DIR, COMPLEMENT_DIR,
    build_signals, config_label, count_signals,
    genuine_audit_block, load_b_dir_hmm_cfg, load_production_defaults,
    preload_ic32_coins,
)

TOP_K = 12
STAGE1_OUT = IC32_DIR / "ic32_lstm_fusion_stage1_signal.json"

AGREE_BOOST = [0.03, 0.05, 0.07]
OPPOSITE_PEN = [0.50, 0.55, 0.65, 0.75]
FLAT_REVIEW = [True, False]
HMM_GATE = [True, False]
FLIP = [True, False]


def build_sweep_configs() -> list[dict]:
    prod = load_production_defaults()
    configs = [{
        "fusion": "baseline",
        "label": "baseline_production",
        **prod,
    }]
    for ab, op, fr, hg, fl in itertools.product(
        AGREE_BOOST, OPPOSITE_PEN, FLAT_REVIEW, HMM_GATE, FLIP,
    ):
        cfg = {
            "fusion": "lstm",
            "agree_boost": ab,
            "opposite_pen": op,
            "neutral_pen": prod["neutral_pen"],
            "conf_entry": prod["conf_entry"],
            "flat_review": fr,
            "hmm_gate_lstm": hg,
            "flip": fl,
            "dir_review_thr": prod["dir_review_thr"],
        }
        cfg["label"] = config_label(cfg)
        configs.append(cfg)

    for op in [0.55, 0.65, 0.75]:
        cfg = {
            "fusion": "lstm",
            "agree_boost": 0.05,
            "opposite_pen": op,
            "neutral_pen": 0.0,
            "conf_entry": prod["conf_entry"],
            "flat_review": True,
            "hmm_gate_lstm": True,
            "flip": True,
            "dual_complement": True,
            "vol_thr": 2.0,
            "bull_thr": 0.38,
            "bear_thr": 0.50,
            "boost": 0.10,
            "comp_opposite_pen": 0.14,
        }
        cfg["label"] = config_label(cfg)
        configs.append(cfg)

    seen = set()
    unique = []
    for c in configs:
        if c["label"] in seen:
            continue
        seen.add(c["label"])
        unique.append(c)
    return unique


def rank_score(row: dict, baseline_dir: int) -> float:
    if row.get("fusion") == "baseline":
        return 0.0
    dir_ratio = row["n_dir"] / baseline_dir if baseline_dir else 1.0
    penalty = max(0.0, dir_ratio - 1.20) * 300.0
    return row["delta_dir"] * 0.3 - abs(row["delta_dir"]) * 0.1 - penalty


def main():
    sep = "=" * 78
    t0 = time.time()
    hmm_cfg = load_b_dir_hmm_cfg()
    configs = build_sweep_configs()

    lstm_path = IC32_DIR / "oof_lstm_baseline_predictions.parquet"
    if not lstm_path.exists():
        raise FileNotFoundError(
            f"Missing {lstm_path}\nRun: python pipeline/05v_oof_lstm_ic32_baseline_cache.py"
        )

    print(f"\n{sep}")
    print("  STAGE 1: ic32 LGBM+LSTM Fusion -- SIGNAL ONLY (OOF genuine)")
    print(f"  HMM B-dir frozen | {len(configs)} configs | Top-{TOP_K} -> Stage 2")
    print(sep)

    lgbm_oof = pd.read_parquet(IC32_DIR / "oof_predictions.parquet")
    lstm_oof = pd.read_parquet(lstm_path)
    if "coin" not in lstm_oof.columns:
        lstm_oof = lstm_oof.reset_index().rename(columns={"index": "ts"})
    comp_path = COMPLEMENT_DIR / "oof_lstm_predictions.parquet"
    complement_oof = None
    if comp_path.exists():
        complement_oof = pd.read_parquet(comp_path)
        if "coin" not in complement_oof.columns:
            complement_oof = complement_oof.reset_index().rename(columns={"index": "ts"})

    coins = preload_ic32_coins(lgbm_oof, lstm_oof, complement_oof=complement_oof)
    print(f"  Coins: {len(coins)} | OOF bars: {sum(len(c['p0']) for c in coins):,}")

    y_base = {c["sym"]: build_signals(c, {"fusion": "baseline"}, hmm_cfg)[0] for c in coins}
    baseline_sig = count_signals(coins, y_base, y_base)
    baseline_dir = baseline_sig["n_dir"]
    print(f"  BASELINE signals: LONG={baseline_sig['n_long']:,} "
          f"SHORT={baseline_sig['n_short']:,} DIR={baseline_dir:,}")

    results = [{
        "label": "baseline_production",
        "fusion": "baseline",
        **baseline_sig,
        "rank_score": 0.0,
        "dir_ratio_vs_base": 1.0,
    }]

    lstm_cfgs = [c for c in configs if c["fusion"] != "baseline"]
    for i, cfg in enumerate(lstm_cfgs):
        y_map = {c["sym"]: build_signals(c, cfg, hmm_cfg)[0] for c in coins}
        sig = count_signals(coins, y_map, y_base)
        row = {"label": cfg["label"], **cfg, **sig}
        row["rank_score"] = rank_score(row, baseline_dir)
        row["dir_ratio_vs_base"] = sig["n_dir"] / baseline_dir if baseline_dir else 1.0
        results.append(row)
        if (i + 1) % 40 == 0:
            print(f"  ... {i+1}/{len(lstm_cfgs)} configs done")

    ranked = sorted(
        [r for r in results if r["fusion"] != "baseline"],
        key=lambda x: (x["rank_score"], -abs(x["dir_ratio_vs_base"] - 1.0)),
        reverse=True,
    )
    top_k = ranked[:TOP_K]

    print(f"\n{sep}")
    print(f"  TOP {TOP_K} BY SIGNAL SCORE")
    print(sep)
    print(f"  {'Label':<44} {'dDIR':>6} {'n_dir':>8} {'ratio':>6}")
    for r in top_k:
        print(f"  {r['label']:<44} {r['delta_dir']:>+6,} {r['n_dir']:>8,} {r['dir_ratio_vs_base']:>6.3f}")

    elapsed = time.time() - t0
    out = {
        **genuine_audit_block(),
        "stage": 1,
        "created": datetime.now().isoformat(),
        "elapsed_sec": round(elapsed, 1),
        "n_configs": len(configs),
        "baseline_signals": baseline_sig,
        "top_k": TOP_K,
        "top_candidates": top_k,
        "all_results": results,
    }
    with open(STAGE1_OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\n  Elapsed: {elapsed:.1f}s")
    print(f"  Saved: {STAGE1_OUT}")
    print(f"  Next: python pipeline/09_ic32_lstm_fusion_stage2_pipeline.py")
    print(f"{sep}\n")


if __name__ == "__main__":
    main()