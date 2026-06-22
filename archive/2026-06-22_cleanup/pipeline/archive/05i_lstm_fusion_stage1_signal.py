"""
Stage 1 — LSTM score fusion: SIGNAL-ONLY OOF sweep (fast).

Tidak menjalankan Guardian/backtest. Hanya hitung perubahan entry signal
setelah LSTM boost/penalty + HMM Config B.

Genuine: OOF only, TRAIN_CUTOFF enforced, holdout NOT touched.
Output: top candidates untuk Stage 2 pipeline eval.
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

from pipeline.lstm_fusion_shared import (
    LGBM_DIR, LSTM_DIR, LGBM_RUN, LSTM_RUN,
    SHORT, FLAT, LONG,
    build_y_pred, config_label, genuine_audit_block,
    load_hmm_cfg, preload_coins,
)

FUSION_MODES = ["pre_hmm", "post_hmm"]
GATE_MODES = ["all_oof", "vol_spike2"]
AGREE_BOOST = [0.04, 0.06, 0.08, 0.10]
NEUTRAL_PEN = [0.04, 0.06, 0.08]
OPPOSITE_PEN = [0.06, 0.10, 0.14]
TOP_K = 12
STAGE1_OUT = LGBM_DIR / "lstm_fusion_stage1_signal.json"


def build_sweep_configs() -> list[dict]:
    configs = [{"fusion": "baseline", "label": "baseline_no_lstm"}]
    for mode, gate, ab, np_, op in itertools.product(
        FUSION_MODES, GATE_MODES, AGREE_BOOST, NEUTRAL_PEN, OPPOSITE_PEN,
    ):
        cfg = {
            "fusion": "lstm", "mode": mode, "gate": gate,
            "agree_boost": ab, "neutral_pen": np_, "opposite_pen": op,
        }
        cfg["label"] = config_label(cfg)
        configs.append(cfg)
    return configs


def precompute_baseline_y(coins: list, hmm_cfg: dict) -> dict[str, np.ndarray]:
    return {c["sym"]: build_y_pred(c, {"fusion": "baseline"}, hmm_cfg) for c in coins}


def precompute_rally_ts(coins: list) -> set:
    """Rally bars: frac_up >= 0.8 AND cross-coin max p2 < 0.45 (baseline scores)."""
    rows = []
    for c in coins:
        for i, ts in enumerate(c["ts"]):
            rows.append({"ts": ts, "coin": c["sym"], "p2": float(c["p2"][i]), "ret4": float(c["ret4"][i])})
    panel = pd.DataFrame(rows)
    pivot_p2 = panel.pivot_table(index="ts", columns="coin", values="p2", aggfunc="first")
    pivot_ret = panel.pivot_table(index="ts", columns="coin", values="ret4", aggfunc="first")
    frac_up = (pivot_ret > 0).mean(axis=1)
    max_p2 = pivot_p2.max(axis=1)
    rally_idx = frac_up.index[(frac_up >= 0.8) & (max_p2 < 0.45)]
    return set(rally_idx)


def count_signals(coins: list, y_by_sym: dict[str, np.ndarray], rally_ts: set,
                  y_base_by_sym: dict[str, np.ndarray]) -> dict:
    n_long = n_short = n_flat = 0
    rally_unlock = rally_extra_short = 0
    delta_long = delta_short = 0

    for c in coins:
        sym = c["sym"]
        y = y_by_sym[sym]
        yb = y_base_by_sym[sym]
        n_long += int((y == LONG).sum())
        n_short += int((y == SHORT).sum())
        n_flat += int((y == FLAT).sum())
        delta_long += int(((y == LONG) & (yb != LONG)).sum())
        delta_short += int(((y == SHORT) & (yb != SHORT)).sum())

        if not rally_ts:
            continue
        ts_arr = c["ts"]
        for i, ts in enumerate(ts_arr):
            if ts not in rally_ts:
                continue
            if y[i] == LONG and yb[i] != LONG:
                rally_unlock += 1
            if y[i] == SHORT and yb[i] != SHORT:
                rally_extra_short += 1

    n_dir = n_long + n_short
    return {
        "n_long": n_long, "n_short": n_short, "n_flat": n_flat,
        "n_dir": n_dir,
        "delta_long": delta_long, "delta_short": delta_short,
        "delta_dir": delta_long + delta_short,
        "rally_unlock_long": rally_unlock,
        "rally_extra_short": rally_extra_short,
        "rally_bars": len(rally_ts),
    }


def rank_score(row: dict, baseline_dir: int) -> float:
    """Prioritize rally unlock without exploding total directional signals."""
    if row.get("fusion") == "baseline":
        return 0.0
    dir_ratio = row["n_dir"] / baseline_dir if baseline_dir else 1.0
    penalty = max(0.0, dir_ratio - 1.15) * 500.0
    return (
        row["rally_unlock_long"] * 3.0
        + row["delta_long"] * 0.5
        - row["delta_short"] * 0.3
        - penalty
    )


def main():
    SEP = "=" * 78
    t0 = time.time()
    hmm_cfg = load_hmm_cfg()
    configs = build_sweep_configs()

    print(f"\n{SEP}")
    print("  STAGE 1: LSTM Score Fusion — SIGNAL ONLY (OOF genuine)")
    print(f"  LGBM OOF + LSTM OOF | HMM Config B | NO Guardian | holdout sealed")
    print(f"  Configs: {len(configs)} | Top-{TOP_K} -> Stage 2")
    print(SEP)

    lgbm_oof = pd.read_parquet(LGBM_DIR / "oof_predictions.parquet")
    lstm_oof = pd.read_parquet(LSTM_DIR / "oof_lstm_predictions.parquet")
    print("  Preloading coins (TRAIN_CUTOFF enforced)...")
    coins = preload_coins(lgbm_oof, lstm_oof)
    lstm_bars = sum(int(c["lstm_valid"].sum()) for c in coins)
    print(f"  Coins: {len(coins)} | LGBM OOF bars: {sum(len(c['p0']) for c in coins):,}")
    print(f"  LSTM OOF bars aligned: {lstm_bars:,}")

    y_base = precompute_baseline_y(coins, hmm_cfg)
    rally_ts = precompute_rally_ts(coins)
    print(f"  Rally diagnostic bars (frac_up>=0.8, max_p2<0.45): {len(rally_ts):,}")

    baseline_sig = count_signals(coins, y_base, rally_ts, y_base)
    baseline_dir = baseline_sig["n_dir"]
    print(f"\n  BASELINE signals: LONG={baseline_sig['n_long']:,} "
          f"SHORT={baseline_sig['n_short']:,} FLAT={baseline_sig['n_flat']:,}")

    results = [{
        "label": "baseline_no_lstm", "fusion": "baseline",
        **baseline_sig, "rank_score": 0.0,
    }]

    for i, cfg in enumerate([c for c in configs if c["fusion"] != "baseline"]):
        y_map = {c["sym"]: build_y_pred(c, cfg, hmm_cfg) for c in coins}
        sig = count_signals(coins, y_map, rally_ts, y_base)
        row = {"label": cfg["label"], **cfg, **sig}
        row["rank_score"] = rank_score(row, baseline_dir)
        row["dir_ratio_vs_base"] = sig["n_dir"] / baseline_dir if baseline_dir else 1.0
        results.append(row)
        if (i + 1) % 30 == 0:
            print(f"  ... {i+1}/{len(configs)-1} configs done")

    ranked = sorted(
        [r for r in results if r["fusion"] != "baseline"],
        key=lambda x: x["rank_score"], reverse=True,
    )
    top_k = ranked[:TOP_K]

    print(f"\n{SEP}")
    print(f"  TOP {TOP_K} BY RALLY/MOMENTUM SCORE (Stage 1)")
    print(SEP)
    print(f"  {'Label':<42} {'dLONG':>6} {'dSHORT':>7} {'Rally+':>6} {'n_dir':>8} {'ratio':>6}")
    for r in top_k:
        print(f"  {r['label']:<42} {r['delta_long']:>+6,} {r['delta_short']:>+7,} "
              f"{r['rally_unlock_long']:>6,} {r['n_dir']:>8,} {r['dir_ratio_vs_base']:>6.3f}")

    elapsed = time.time() - t0
    out = {
        **genuine_audit_block(),
        "stage": 1,
        "stage_name": "signal_only_sweep",
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
    print(f"  Next: python pipeline/05i_lstm_fusion_stage2_pipeline.py")
    print(f"{SEP}\n")


if __name__ == "__main__":
    main()