"""
05n tune — Rally Boost round 2 (focused grid after v1 probe).

v1 lesson: no-LSTM = too many trades; LSTM = -0.09 sen (close).
Tune levers:
  - rally_min_gap: near-miss only (not all FLAT)
  - rally_prop_rank: decay boost by cross-coin rank
  - Always rally_require_lstm=True

Genuine OOF, holdout sealed.
"""
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

from config import (
    MODAL_PER_TRADE, LEVERAGE_SIM, FEE_PER_SIDE, SLIPPAGE_PER_SIDE,
    SWING_LABEL_MIN_RR, SWING_LABEL_MIN_TP, SWING_LABEL_MAX_SL,
    TP_SL_FALLBACK_TP, TP_SL_FALLBACK_SL, MAX_HOLDING_BARS,
    GUARDIAN_ACTIVATION_ATR,
)
import importlib.util

from pipeline.lstm_fusion_shared import (
    LGBM_DIR, LSTM_DIR, GUARDIAN_DIR, DYNAMIC_FEATS, FLAT,
    apply_fused_scores, attach_cross_section, genuine_audit_block,
    load_guardian_params, load_hmm_cfg, preload_coins, precompute_cross_section,
    rally_boost_label, build_predictions_full,
)

_spec = importlib.util.spec_from_file_location(
    "eval05n", ROOT / "pipeline" / "05n_rally_boost_eval.py"
)
_eval05n = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_eval05n)
eval_pipeline = _eval05n.eval_pipeline
signal_stats = _eval05n.signal_stats

OUT_PATH = LGBM_DIR / "rally_boost_tune.json"
TOP_K_PIPELINE = 14
RALLY_LOOSE_FRAC = 0.75

FROZEN_05J = {
    "fusion": "lstm",
    "mode": "conditional_momentum",
    "bull_thr": 0.38,
    "bear_thr": 0.50,
    "boost": 0.10,
    "opposite_pen": 0.14,
    "near_miss_gap": 0.03,
    "vol_thr": 2.0,
    "proportional": True,
    "enable_boost": True,
    "enable_penalty": True,
    "rally_boost_enabled": False,
    "rally_require_lstm": True,
}


def build_tune_configs() -> list[dict]:
    configs = [
        {"fusion": "baseline", "label": "baseline_no_lstm"},
        {**FROZEN_05J, "label": "ref_05j_winner"},
    ]
    seen = {c["label"] for c in configs}

    # v1 best anchor
    anchor = {
        **FROZEN_05J,
        "rally_boost_enabled": True,
        "rally_frac": 0.80,
        "rally_top_k": 5,
        "rally_boost_amt": 0.10,
        "rally_require_lstm": True,
    }
    anchor["label"] = rally_boost_label(anchor)
    if anchor["label"] not in seen:
        seen.add(anchor["label"])
        configs.append(anchor)

    for rf, rk, ra, mg, pr in itertools.product(
        [0.80, 0.85],
        [2, 3],
        [0.05, 0.06, 0.08],
        [0.05, 0.08],
        [False, True],
    ):
        cfg = {
            **FROZEN_05J,
            "rally_boost_enabled": True,
            "rally_frac": rf,
            "rally_top_k": rk,
            "rally_boost_amt": ra,
            "rally_min_gap": mg,
            "rally_prop_rank": pr,
            "rally_require_lstm": True,
        }
        cfg["label"] = rally_boost_label(cfg)
        if cfg["label"] not in seen:
            seen.add(cfg["label"])
            configs.append(cfg)
    return configs


def rank_signal_tune(row: dict, ref_row: dict) -> float:
    if row.get("fusion") == "baseline":
        return 0.0
    ref_dir = ref_row.get("n_dir", 1)
    ratio_pen = max(0.0, row["n_dir"] / ref_dir - 1.08) * 80.0
    return (
        row["rally_entry_unlock"] * 4.0
        + row["rally_boost_fires"] * 0.02
        - ratio_pen
    )


def main():
    hmm_cfg = load_hmm_cfg()
    g_params = load_guardian_params()
    configs = build_tune_configs()
    t0 = time.time()

    SEP = "=" * 78
    print(f"\n{SEP}")
    print("  05n TUNE: Rally Boost round 2 (near-miss + prop rank)")
    print(f"  Configs: {len(configs)} | Pipeline top-{TOP_K_PIPELINE}")
    print(SEP)

    lgbm_oof = pd.read_parquet(LGBM_DIR / "oof_predictions.parquet")
    lstm_oof = pd.read_parquet(LSTM_DIR / "oof_lstm_predictions.parquet")
    coins = preload_coins(lgbm_oof, lstm_oof)
    cs = precompute_cross_section(coins)
    attach_cross_section(coins, cs)
    rally_loose_ts = set(cs["frac_up"].index[cs["frac_up"] >= RALLY_LOOSE_FRAC])

    y_base = {c["sym"]: apply_fused_scores(c, {"fusion": "baseline"}, hmm_cfg)[0] for c in coins}
    ref_cfg = next(c for c in configs if c["label"] == "ref_05j_winner")
    y_ref, _, _ = build_predictions_full(coins, ref_cfg, hmm_cfg)
    ref_sig = signal_stats(coins, y_ref, y_base, y_ref, rally_loose_ts, 0)

    signal_rows = []
    for cfg in configs:
        if cfg.get("fusion") == "baseline":
            sig = signal_stats(coins, y_base, y_base, y_ref, rally_loose_ts, 0)
            row = {"label": cfg["label"], **cfg, **sig, "rank": 0.0}
        else:
            y_map, _, rf = build_predictions_full(coins, cfg, hmm_cfg)
            sig = signal_stats(coins, y_map, y_base, y_ref, rally_loose_ts, rf)
            row = {"label": cfg["label"], **cfg, **sig, "rank": rank_signal_tune(sig, ref_sig)}
        signal_rows.append(row)
    signal_rows.sort(key=lambda x: x["rank"], reverse=True)
    top_signal = [r for r in signal_rows if r.get("fusion") != "baseline"][:TOP_K_PIPELINE]

    print(f"\n  STAGE A — TOP 10 (by rally unlock, controlled dir growth)")
    for r in top_signal[:10]:
        print(f"  {r['label'][:58]:<58} rally+={r['rally_entry_unlock']:>4,} "
              f"fires={r['rally_boost_fires']:>5,} n_dir={r['n_dir']:>6,}")

    g_model = joblib.load(GUARDIAN_DIR / "guardian.pkl")
    g_scaler = joblib.load(GUARDIAN_DIR / "guardian_scaler.pkl")
    with open(GUARDIAN_DIR / "guardian_features.json", encoding="utf-8") as f:
        g_static = [f for f in json.load(f) if f not in DYNAMIC_FEATS]
    coins_p = preload_coins(lgbm_oof, lstm_oof, g_static)
    attach_cross_section(coins_p, cs)

    base_keys = set()
    for c in coins_p:
        y, _, _ = apply_fused_scores(c, {"fusion": "baseline"}, hmm_cfg)
        for i in range(len(y)):
            if y[i] != FLAT:
                base_keys.add((c["sym"], i))

    cfg_keys = (
        "fusion", "mode", "label", "bull_thr", "bear_thr", "boost", "opposite_pen",
        "near_miss_gap", "vol_thr", "proportional", "enable_boost", "enable_penalty",
        "rally_boost_enabled", "rally_frac", "rally_top_k", "rally_boost_amt",
        "rally_require_lstm", "rally_min_gap", "rally_prop_rank",
    )
    pipe_candidates = [{"fusion": "baseline", "label": "baseline_no_lstm"}, ref_cfg] + [
        {k: v for k, v in r.items() if k in cfg_keys}
        for r in top_signal if r.get("label") != "ref_05j_winner"
    ][:TOP_K_PIPELINE]

    pipe_results, ref_pipe = [], None
    print(f"\n  STAGE B — PIPELINE ({len(pipe_candidates)} configs)")
    for i, cfg in enumerate(pipe_candidates):
        t_c = time.time()
        met = eval_pipeline(coins_p, cfg, hmm_cfg, g_model, g_scaler, g_params, base_keys)
        row = {"label": cfg["label"], **cfg, **met}
        if cfg.get("label") == "ref_05j_winner":
            ref_pipe = row
        elif ref_pipe and cfg.get("label") != "ref_05j_winner":
            d_port = met["portfolio"]["ppt_norm"] - ref_pipe["portfolio"]["ppt_norm"]
            row["delta_port_ppt"] = d_port
            row["passes_strict"] = bool(d_port > 0.0001 and met["new_rally"]["ppt_norm"] > 0.30)
            row["passes_close"] = bool(
                d_port >= -0.0005
                and met["n_new_rally"] >= 20
                and met["new_rally"]["ppt_norm"] > 0.25
            )
        pipe_results.append(row)
        print(f"  [{i+1}] {cfg['label'][:56]}")
        print(f"       port={met['portfolio']['ppt_norm']:+.4f} dPort={row.get('delta_port_ppt',0):+.4f} | "
              f"newRally N={met['n_new_rally']:,} PPT={met['new_rally']['ppt_norm']:+.4f} ({time.time()-t_c:.0f}s)")

    ranked = sorted(
        [r for r in pipe_results if r.get("label") not in ("baseline_no_lstm", "ref_05j_winner")],
        key=lambda x: x.get("portfolio", {}).get("ppt_norm", -999),
        reverse=True,
    )
    strict_win = [r for r in pipe_results if r.get("passes_strict")]
    close_win = [r for r in pipe_results if r.get("passes_close") and not r.get("passes_strict")]

    elapsed = time.time() - t0
    decision = "PROMOTE_CANDIDATE" if strict_win else ("TUNE_MORE" if close_win else "NO_PROMOTE")
    out = {
        **genuine_audit_block(),
        "eval": "rally_boost_tune_r2",
        "created": datetime.now().isoformat(),
        "elapsed_sec": round(elapsed, 1),
        "n_configs": len(configs),
        "ref_pipeline": ref_pipe,
        "pipeline_results": pipe_results,
        "strict_winners": strict_win,
        "close_candidates": close_win,
        "decision": decision,
        "best": ranked[0] if ranked else None,
    }
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\n  Decision: {decision} | strict={len(strict_win)} close={len(close_win)}")
    print(f"  Saved: {OUT_PATH}")
    print(f"{SEP}\n")


if __name__ == "__main__":
    main()