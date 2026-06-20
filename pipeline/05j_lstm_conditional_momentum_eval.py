"""
05j — Conditional LSTM momentum fusion (genuine OOF).

Spesialisasi:
  BOOST   : LGBM FLAT/near-miss + vol_spike + LSTM momentum confident
  PENALTY : LGBM entry berlawanan LSTM dominant pada vol_spike (reversal)

Stage A: signal sweep (fast)
Stage B: pipeline top-10 + new-trade quality breakdown

Genuine: OOF only, HMM Config B frozen, holdout sealed.

Usage:
  python pipeline/05j_lstm_conditional_momentum_eval.py
  python pipeline/05j_lstm_conditional_momentum_eval.py --lstm-run tb_lstm_complement_v1
"""
import argparse
import itertools
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

from config import (
    MODEL_DIR, MODAL_PER_TRADE, LEVERAGE_SIM, FEE_PER_SIDE, SLIPPAGE_PER_SIDE,
    SWING_LABEL_MIN_RR, SWING_LABEL_MIN_TP, SWING_LABEL_MAX_SL,
    TP_SL_FALLBACK_TP, TP_SL_FALLBACK_SL, MAX_HOLDING_BARS,
    GUARDIAN_ACTIVATION_ATR,
)
from core.cascade_utils import apply_conditional_momentum_fusion_pre
from core.evaluator import simulate_trades_swing
from pipeline.lstm_fusion_shared import (
    LGBM_DIR, GUARDIAN_DIR, DYNAMIC_FEATS, DYNSIZE_CFG, FLAT, LONG, SHORT,
    apply_hmm_thr, build_y_pred, config_label, genuine_audit_block,
    compute_dynamic_modal, load_guardian_params, load_hmm_cfg,
    preload_coins, summarize_trades,
)

DEFAULT_LSTM_RUN = "tb_lstm_genuine_v2"
MOMENTUM_VOL_THR = 2.0
TOP_K_PIPELINE = 10

BULL_THR = [0.38, 0.42, 0.46]
BEAR_THR = [0.50, 0.55, 0.58]
BOOST = [0.06, 0.08, 0.10]
OPP_PEN = [0.10, 0.14]
NEAR_GAP = [0.03, 0.05, 0.08]
MODES = [
    {"enable_boost": True, "enable_penalty": True},
    {"enable_boost": True, "enable_penalty": False},
    {"enable_boost": False, "enable_penalty": True},
]


def build_sweep_configs() -> list[dict]:
    configs = [{"fusion": "baseline", "label": "baseline_no_lstm"}]
    for mode, bu, be, b, o, g in itertools.product(MODES, BULL_THR, BEAR_THR, BOOST, OPP_PEN, NEAR_GAP):
        if mode["enable_penalty"] is False and o != OPP_PEN[0]:
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


def signal_stats(coins: list, y_map: dict, y_base: dict, hmm_cfg: dict) -> dict:
    boost_unlock = penalty_block = 0
    n_long = n_short = 0
    for c in coins:
        sym = c["sym"]
        y = y_map[sym]
        yb = y_base[sym]
        n_long += int((y == LONG).sum())
        n_short += int((y == SHORT).sum())
        mom = c["vol_spike"] >= MOMENTUM_VOL_THR
        for i in range(len(y)):
            if not mom[i]:
                continue
            if yb[i] == FLAT and y[i] != FLAT:
                boost_unlock += 1
            if yb[i] != FLAT and y[i] == FLAT:
                penalty_block += 1
    return {
        "n_long": n_long, "n_short": n_short, "n_dir": n_long + n_short,
        "boost_unlock": boost_unlock,
        "penalty_block": penalty_block,
    }


def rank_signal(row: dict, base_dir: int) -> float:
    if row.get("fusion") == "baseline":
        return 0.0
    ratio_pen = max(0.0, row["n_dir"] / base_dir - 1.05) * 200.0
    return (
        row["boost_unlock"] * 2.0
        + row["penalty_block"] * 1.5
        - ratio_pen
    )


def eval_pipeline(coins, cfg, hmm_cfg, g_model, g_scaler, g_params, base_keys: set) -> dict:
    all_trades = []
    common = dict(
        modal=MODAL_PER_TRADE, leverage=LEVERAGE_SIM[0],
        fee_per_side=FEE_PER_SIDE, slippage=SLIPPAGE_PER_SIDE,
        max_hold=MAX_HOLDING_BARS,
        min_rr=SWING_LABEL_MIN_RR, min_tp_atr=SWING_LABEL_MIN_TP,
        max_sl_atr=SWING_LABEL_MAX_SL,
        tp_fallback_atr=TP_SL_FALLBACK_TP, sl_fallback_atr=TP_SL_FALLBACK_SL,
    )
    for c in coins:
        y = build_y_pred(c, cfg, hmm_cfg)
        p0, p2 = c["p0"].copy(), c["p2"].copy()
        if cfg.get("mode") == "conditional_momentum":
            _, _, tl, ts = apply_hmm_thr(p0, p2, c["hmm"], hmm_cfg)
            p0, p2 = apply_conditional_momentum_fusion_pre(
                p0, p2, c["lstm_p"], tl, ts, c["vol_spike"],
                vol_thr=cfg.get("vol_thr", 2.0),
                bull_thr=cfg["bull_thr"], bear_thr=cfg["bear_thr"],
                near_miss_gap=cfg["near_miss_gap"],
                boost=cfg["boost"], opposite_pen=cfg["opposite_pen"],
                enable_boost=cfg.get("enable_boost", True),
                enable_penalty=cfg.get("enable_penalty", True),
                lstm_valid=c["lstm_valid"],
            )
        _, conf, tl, ts = apply_hmm_thr(p0, p2, c["hmm"], hmm_cfg)
        p0e, p2e = p0, p2
        modal_arr = compute_dynamic_modal(p0e, p2e, c["hmm"], y, MODAL_PER_TRADE, DYNSIZE_CFG, tl, ts)
        rep = simulate_trades_swing(
            y_pred=y, guardian_enabled=True,
            guardian_model=g_model, guardian_scaler=g_scaler, X_guardian=c["X_grd"],
            guardian_exit_threshold=g_params["exit_threshold"],
            guardian_min_hold_bars=g_params["min_hold_bars"],
            guardian_activation_atr=GUARDIAN_ACTIVATION_ATR,
            modal_arr=modal_arr,
            close=c["close"], high=c["high"], low=c["low"], atr=c["atr"],
            h4_swing_highs=c["h4_sh"], h4_swing_lows=c["h4_sl"],
            **common,
        )
        for t in rep.get("trades", []):
            bi = t.get("bar_in", 0)
            key = (c["sym"], bi)
            t2 = dict(t)
            t2["is_new"] = key not in base_keys
            t2["momentum_entry"] = bi < len(c["vol_spike"]) and c["vol_spike"][bi] >= MOMENTUM_VOL_THR
            all_trades.append(t2)
    port = summarize_trades(all_trades)
    mom = summarize_trades([t for t in all_trades if t.get("momentum_entry")])
    new_t = summarize_trades([t for t in all_trades if t.get("is_new")])
    new_mom = summarize_trades([t for t in all_trades if t.get("is_new") and t.get("momentum_entry")])
    blocked = len(base_keys)  # placeholder
    return {
        "portfolio": port, "momentum": mom,
        "new_trades": new_t, "new_momentum": new_mom,
        "n_new": new_t["n"], "n_new_momentum": new_mom["n"],
    }


def main():
    parser = argparse.ArgumentParser(description="Conditional LSTM momentum fusion OOF eval")
    parser.add_argument(
        "--lstm-run",
        default=DEFAULT_LSTM_RUN,
        help="LSTM run folder under models/runs/ (default: tb_lstm_genuine_v2)",
    )
    parser.add_argument(
        "--out-name",
        default=None,
        help="Output JSON filename under tb_lgbm_genuine_v2/ (default: lstm_conditional_momentum_eval.json)",
    )
    args = parser.parse_args()

    lstm_run = args.lstm_run
    lstm_dir = MODEL_DIR / "runs" / lstm_run
    out_name = args.out_name or (
        "lstm_conditional_momentum_eval.json"
        if lstm_run == DEFAULT_LSTM_RUN
        else f"lstm_conditional_momentum_eval_{lstm_run}.json"
    )
    out_path = LGBM_DIR / out_name

    SEP = "=" * 78
    hmm_cfg = load_hmm_cfg()
    g_params = load_guardian_params()
    configs = build_sweep_configs()
    t0 = time.time()

    print(f"\n{SEP}")
    print("  05j: Conditional LSTM Momentum Fusion (genuine OOF)")
    print(f"  LSTM run: {lstm_run}")
    print(f"  BOOST flat/near-miss | PENALTY wrong-way entry | vol_spike>={MOMENTUM_VOL_THR}")
    print(f"  Stage A signal configs: {len(configs)} | Stage B pipeline top-{TOP_K_PIPELINE}")
    print(SEP)

    lgbm_oof = pd.read_parquet(LGBM_DIR / "oof_predictions.parquet")
    lstm_oof = pd.read_parquet(lstm_dir / "oof_lstm_predictions.parquet")
    coins = preload_coins(lgbm_oof, lstm_oof)
    print(f"  Coins: {len(coins)}")

    y_base = {c["sym"]: build_y_pred(c, {"fusion": "baseline"}, hmm_cfg) for c in coins}
    base_sig = signal_stats(coins, y_base, y_base, hmm_cfg)
    base_dir = base_sig["n_dir"]
    print(f"  Baseline signals: dir={base_dir:,} (mom vol bars tracked in sweep)")

    # ── Stage A: signal ───────────────────────────────────────────────────────
    signal_rows = [{"fusion": "baseline", "label": "baseline_no_lstm", **base_sig, "rank": 0.0}]
    for cfg in [c for c in configs if c["fusion"] != "baseline"]:
        y_map = {c["sym"]: build_y_pred(c, cfg, hmm_cfg) for c in coins}
        sig = signal_stats(coins, y_map, y_base, hmm_cfg)
        row = {"label": cfg["label"], **cfg, **sig, "rank": rank_signal({**sig, **cfg}, base_dir)}
        signal_rows.append(row)
    signal_rows.sort(key=lambda x: x["rank"], reverse=True)
    top_signal = [r for r in signal_rows if r.get("fusion") != "baseline"][:TOP_K_PIPELINE]

    print(f"\n  STAGE A — TOP 8 SIGNAL")
    print(f"  {'Label':<52} {'boost+':>6} {'pen_blk':>7} {'n_dir':>7}")
    for r in top_signal[:8]:
        print(f"  {r['label']:<52} {r['boost_unlock']:>6,} {r['penalty_block']:>7,} {r['n_dir']:>7,}")

    # ── Stage B: pipeline ─────────────────────────────────────────────────────
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

    pipe_candidates = [{"fusion": "baseline", "label": "baseline_no_lstm"}] + [
        {k: v for k, v in r.items() if k in (
            "fusion", "mode", "label", "bull_thr", "bear_thr", "boost", "opposite_pen",
            "near_miss_gap", "vol_thr", "proportional", "enable_boost", "enable_penalty",
        )} for r in top_signal
    ]

    pipe_results = []
    baseline_pipe = None
    print(f"\n  STAGE B — PIPELINE ({len(pipe_candidates)} configs)")
    for i, cfg in enumerate(pipe_candidates):
        t_c = time.time()
        met = eval_pipeline(coins_p, cfg, hmm_cfg, g_model, g_scaler, g_params, base_keys)
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
        pipe_results.append(row)
        dt = time.time() - t_c
        print(f"  [{i+1}] {cfg['label'][:50]}")
        print(f"       port PPT={met['portfolio']['ppt_norm']:+.4f} | "
              f"new_mom N={met['n_new_momentum']:,} PPT={met['new_momentum']['ppt_norm']:+.4f} ({dt:.0f}s)")

    winners = [r for r in pipe_results if r.get("passes")]
    ranked = sorted(
        [r for r in pipe_results if r.get("fusion") != "baseline"],
        key=lambda x: (x.get("new_momentum", {}).get("ppt_norm", -999), x["portfolio"]["ppt_norm"]),
        reverse=True,
    )

    print(f"\n{SEP}")
    print("  STAGE B SUMMARY (new momentum trades = LSTM-caused entries)")
    print(SEP)
    b = baseline_pipe
    print(f"  BASELINE: port PPT={b['portfolio']['ppt_norm']:+.4f} | "
          f"mom PPT={b['momentum']['ppt_norm']:+.4f}")
    print(f"  {'Label':<40} {'newMomN':>7} {'newMomPPT':>10} {'portPPT':>9} {'dPort':>7} PASS")
    for r in ranked[:10]:
        nm = r.get("new_momentum", {})
        pas = "Y" if r.get("passes") else "N"
        print(f"  {r['label']:<40} {r.get('n_new_momentum',0):>7,} "
              f"{nm.get('ppt_norm',0):>+10.4f} {r['portfolio']['ppt_norm']:>+9.4f} "
              f"{r.get('delta_port_ppt',0):>+7.4f} {pas:>4}")

    elapsed = time.time() - t0
    audit = genuine_audit_block()
    audit["lstm_source"] = f"{lstm_run}/oof_lstm_predictions.parquet (has_oof=True only)"
    out = {
        **audit,
        "lstm_run": lstm_run,
        "eval": "conditional_momentum_fusion",
        "created": datetime.now().isoformat(),
        "elapsed_sec": round(elapsed, 1),
        "n_signal_configs": len(configs),
        "baseline_signal": base_sig,
        "top_signal": top_signal,
        "baseline_pipeline": baseline_pipe,
        "pipeline_results": pipe_results,
        "winners": winners,
        "decision": "PROMOTE_CANDIDATE" if winners else "NO_PROMOTE",
        "best": ranked[0] if ranked else None,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\n  Decision: {out['decision']} | Elapsed: {elapsed/60:.1f} min")
    print(f"  Saved: {out_path}")
    print(f"{SEP}\n")


if __name__ == "__main__":
    main()