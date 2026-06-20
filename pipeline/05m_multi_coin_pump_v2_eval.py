"""
05m — Multi-coin pump v2: loose rally (frac_up only) + top-K without breadth gate.

Variants on frozen 05j conditional_momentum:
  A) top-K only (3/5/8) — cap LONG per bar, no breadth gate
  B) frac_up-only breadth gate (0.7/0.75/0.8) — no max_p2 cap
  C) frac_up + top-K combined
  D) wider near_miss_gap (0.08/0.10) — no breadth, no top-K

Genuine: OOF only, HMM Config B frozen, holdout sealed.
"""
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
    MODAL_PER_TRADE, LEVERAGE_SIM, FEE_PER_SIDE, SLIPPAGE_PER_SIDE,
    SWING_LABEL_MIN_RR, SWING_LABEL_MIN_TP, SWING_LABEL_MAX_SL,
    TP_SL_FALLBACK_TP, TP_SL_FALLBACK_SL, MAX_HOLDING_BARS,
    GUARDIAN_ACTIVATION_ATR,
)
from core.evaluator import simulate_trades_swing
from pipeline.lstm_fusion_shared import (
    LGBM_DIR, LSTM_DIR, GUARDIAN_DIR, DYNAMIC_FEATS, DYNSIZE_CFG, FLAT, LONG, SHORT,
    apply_fused_scores, apply_hmm_thr, apply_top_k_per_bar,
    attach_cross_section, compute_dynamic_modal, genuine_audit_block,
    load_guardian_params, load_hmm_cfg, preload_coins, precompute_cross_section,
    pump_config_label, summarize_trades,
)

OUT_PATH = LGBM_DIR / "multi_coin_pump_v2_eval.json"
MOMENTUM_VOL_THR = 2.0
TOP_K_PIPELINE = 14
RALLY_LOOSE_FRAC = 0.75

FROZEN = {
    "fusion": "lstm",
    "mode": "conditional_momentum",
    "bull_thr": 0.38,
    "bear_thr": 0.50,
    "boost": 0.10,
    "opposite_pen": 0.14,
    "vol_thr": MOMENTUM_VOL_THR,
    "proportional": True,
    "enable_boost": True,
    "enable_penalty": True,
    "breadth_gate_side": "long",
    "near_miss_gap": 0.03,
    "breadth_frac": None,
    "max_p2_cap": None,
    "top_k_long": None,
    "top_k_short": None,
}


def build_sweep_configs() -> list[dict]:
    configs = [
        {"fusion": "baseline", "label": "baseline_no_lstm"},
        {**FROZEN, "label": "ref_05j_winner"},
    ]
    seen = {c["label"] for c in configs}

    def add(cfg: dict) -> None:
        cfg["label"] = pump_config_label(cfg)
        if cfg["label"] not in seen:
            seen.add(cfg["label"])
            configs.append(cfg)

    # A) top-K only
    for kl in [3, 5, 8]:
        add({**FROZEN, "top_k_long": kl})

    # B) frac_up-only breadth (no max_p2)
    for br in [0.7, 0.75, 0.8]:
        for ng in [0.03, 0.05, 0.08]:
            add({**FROZEN, "breadth_frac": br, "near_miss_gap": ng})

    # C) frac_up + top-K
    for br, kl, ng in itertools.product([0.75, 0.8], [3, 5], [0.03, 0.05]):
        add({**FROZEN, "breadth_frac": br, "top_k_long": kl, "near_miss_gap": ng})

    # D) wider near_miss only
    for ng in [0.08, 0.10]:
        add({**FROZEN, "near_miss_gap": ng})

    return configs


def rally_ts_sets(cs: dict) -> dict:
    frac = cs["frac_up"]
    max_p2 = cs["max_p2"]
    return {
        "strict": set(frac.index[(frac >= 0.8) & (max_p2 < 0.45)]),
        "loose": set(frac.index[frac >= RALLY_LOOSE_FRAC]),
        "br70": set(frac.index[frac >= 0.7]),
        "br75": set(frac.index[frac >= 0.75]),
        "br80": set(frac.index[frac >= 0.8]),
    }


def rally_ts_for_cfg(cfg: dict, rally_sets: dict) -> set:
    br = cfg.get("breadth_frac")
    if br is None:
        return rally_sets["loose"]
    if br >= 0.8:
        return rally_sets["br80"]
    if br >= 0.75:
        return rally_sets["br75"]
    return rally_sets["br70"]


def build_predictions(coins: list, cfg: dict, hmm_cfg: dict) -> tuple[dict, dict]:
    y_map, scores = {}, {}
    for c in coins:
        y, p0, p2 = apply_fused_scores(c, cfg, hmm_cfg)
        y_map[c["sym"]] = y
        scores[c["sym"]] = {"p0": p0, "p2": p2}
    k_long = cfg.get("top_k_long") or 0
    k_short = cfg.get("top_k_short") or 0
    if k_long > 0 or k_short > 0:
        y_map = apply_top_k_per_bar(coins, y_map, scores, k_long=k_long, k_short=k_short)
    return y_map, scores


def signal_stats(coins: list, y_map: dict, y_base: dict, rally_sets: dict, cfg: dict) -> dict:
    rally_ts = rally_ts_for_cfg(cfg, rally_sets)
    rally_strict = rally_sets["strict"]
    boost_unlock = penalty_block = 0
    rally_loose_unlock = rally_strict_unlock = 0
    n_long = n_short = 0
    per_bar_long = {}

    for c in coins:
        sym = c["sym"]
        y = y_map[sym]
        yb = y_base[sym]
        n_long += int((y == LONG).sum())
        n_short += int((y == SHORT).sum())
        mom = c["vol_spike"] >= MOMENTUM_VOL_THR
        for i, ts in enumerate(c["ts"]):
            if y[i] == LONG:
                per_bar_long[ts] = per_bar_long.get(ts, 0) + 1
            if mom[i]:
                if yb[i] == FLAT and y[i] != FLAT:
                    boost_unlock += 1
                if yb[i] != FLAT and y[i] == FLAT:
                    penalty_block += 1
            if yb[i] != LONG and y[i] == LONG:
                if ts in rally_ts:
                    rally_loose_unlock += 1
                if ts in rally_strict:
                    rally_strict_unlock += 1

    multi_long_bars = sum(1 for n in per_bar_long.values() if n >= 2)
    max_long_per_bar = max(per_bar_long.values()) if per_bar_long else 0
    return {
        "n_long": n_long, "n_short": n_short, "n_dir": n_long + n_short,
        "boost_unlock": boost_unlock,
        "penalty_block": penalty_block,
        "rally_unlock_loose": rally_loose_unlock,
        "rally_unlock_strict": rally_strict_unlock,
        "rally_bars_loose": len(rally_ts),
        "rally_bars_strict": len(rally_strict),
        "multi_long_bars": multi_long_bars,
        "max_long_per_bar": max_long_per_bar,
    }


def rank_signal(row: dict, ref_row: dict) -> float:
    if row.get("fusion") == "baseline":
        return 0.0
    ref_dir = ref_row.get("n_dir", 1)
    ratio_pen = max(0.0, row["n_dir"] / ref_dir - 1.10) * 120.0
    return (
        row["rally_unlock_loose"] * 5.0
        + row["multi_long_bars"] * 2.0
        + row["boost_unlock"] * 1.5
        + row["penalty_block"] * 0.5
        - ratio_pen
    )


def eval_pipeline(coins, cfg, hmm_cfg, g_model, g_scaler, g_params, base_keys: set) -> dict:
    y_map, scores = build_predictions(coins, cfg, hmm_cfg)
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
        sym = c["sym"]
        y = y_map[sym]
        p0, p2 = scores[sym]["p0"], scores[sym]["p2"]
        _, _, tl, ts = apply_hmm_thr(p0, p2, c["hmm"], hmm_cfg)
        modal_arr = compute_dynamic_modal(p0, p2, c["hmm"], y, MODAL_PER_TRADE, DYNSIZE_CFG, tl, ts)
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
            key = (sym, bi)
            t2 = dict(t)
            t2["is_new"] = key not in base_keys
            t2["momentum_entry"] = bi < len(c["vol_spike"]) and c["vol_spike"][bi] >= MOMENTUM_VOL_THR
            all_trades.append(t2)
    port = summarize_trades(all_trades)
    mom = summarize_trades([t for t in all_trades if t.get("momentum_entry")])
    new_t = summarize_trades([t for t in all_trades if t.get("is_new")])
    new_mom = summarize_trades([t for t in all_trades if t.get("is_new") and t.get("momentum_entry")])
    return {
        "portfolio": port, "momentum": mom,
        "new_trades": new_t, "new_momentum": new_mom,
        "n_new": new_t["n"], "n_new_momentum": new_mom["n"],
    }


def main():
    hmm_cfg = load_hmm_cfg()
    g_params = load_guardian_params()
    configs = build_sweep_configs()
    t0 = time.time()

    SEP = "=" * 78
    print(f"\n{SEP}")
    print("  05m: Multi-Coin Pump v2 (frac_up only + top-K no breadth)")
    print(f"  Configs: {len(configs)} | Pipeline top-{TOP_K_PIPELINE}")
    print(SEP)

    lgbm_oof = pd.read_parquet(LGBM_DIR / "oof_predictions.parquet")
    lstm_oof = pd.read_parquet(LSTM_DIR / "oof_lstm_predictions.parquet")
    coins = preload_coins(lgbm_oof, lstm_oof)
    cs = precompute_cross_section(coins)
    attach_cross_section(coins, cs)
    rally_sets = rally_ts_sets(cs)
    print(f"  Coins: {len(coins)}")
    print(f"  Rally strict (br>=0.8, mp<0.45): {len(rally_sets['strict']):,}")
    print(f"  Rally loose  (br>={RALLY_LOOSE_FRAC}): {len(rally_sets['loose']):,}")

    y_base = {}
    for c in coins:
        y, _, _ = apply_fused_scores(c, {"fusion": "baseline"}, hmm_cfg)
        y_base[c["sym"]] = y

    ref_cfg = next(c for c in configs if c["label"] == "ref_05j_winner")
    y_ref, _ = build_predictions(coins, ref_cfg, hmm_cfg)
    ref_sig = signal_stats(coins, y_ref, y_base, rally_sets, ref_cfg)
    print(f"  Ref 05j: boost+={ref_sig['boost_unlock']:,} rally_loose+={ref_sig['rally_unlock_loose']:,} "
          f"multi_long_bars={ref_sig['multi_long_bars']:,} max_long/bar={ref_sig['max_long_per_bar']}")

    signal_rows = []
    for cfg in configs:
        if cfg.get("fusion") == "baseline":
            sig = signal_stats(coins, y_base, y_base, rally_sets, cfg)
            row = {"label": cfg["label"], **cfg, **sig, "rank": 0.0}
        else:
            y_map, _ = build_predictions(coins, cfg, hmm_cfg)
            sig = signal_stats(coins, y_map, y_base, rally_sets, cfg)
            row = {"label": cfg["label"], **cfg, **sig, "rank": rank_signal(sig, ref_sig)}
        signal_rows.append(row)
    signal_rows.sort(key=lambda x: x["rank"], reverse=True)
    top_signal = [r for r in signal_rows if r.get("fusion") != "baseline"][:TOP_K_PIPELINE]

    print(f"\n  STAGE A — TOP 10 SIGNAL")
    print(f"  {'Label':<54} {'rallyL+':>6} {'boost+':>6} {'mltBar':>6} {'n_dir':>7}")
    for r in top_signal[:10]:
        print(f"  {r['label']:<54} {r['rally_unlock_loose']:>6,} {r['boost_unlock']:>6,} "
              f"{r['multi_long_bars']:>6,} {r['n_dir']:>7,}")

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
        "breadth_frac", "max_p2_cap", "top_k_long", "top_k_short", "breadth_gate_side",
    )
    pipe_candidates = [{"fusion": "baseline", "label": "baseline_no_lstm"}, ref_cfg] + [
        {k: v for k, v in r.items() if k in cfg_keys}
        for r in top_signal if r.get("label") != "ref_05j_winner"
    ]
    pipe_candidates = pipe_candidates[:TOP_K_PIPELINE + 2]

    pipe_results = []
    ref_pipe = baseline_pipe = None
    print(f"\n  STAGE B — PIPELINE ({len(pipe_candidates)} configs)")
    for i, cfg in enumerate(pipe_candidates):
        t_c = time.time()
        met = eval_pipeline(coins_p, cfg, hmm_cfg, g_model, g_scaler, g_params, base_keys)
        row = {"label": cfg["label"], **cfg, **met}
        if cfg.get("label") == "ref_05j_winner":
            ref_pipe = row
        if cfg.get("fusion") == "baseline":
            baseline_pipe = row
        elif ref_pipe and cfg.get("label") != "ref_05j_winner":
            row["delta_port_ppt"] = met["portfolio"]["ppt_norm"] - ref_pipe["portfolio"]["ppt_norm"]
            row["delta_new_mom_ppt"] = (
                met["new_momentum"]["ppt_norm"] - ref_pipe["new_momentum"]["ppt_norm"]
                if met["n_new_momentum"] > 0 else None
            )
            d_port = met["portfolio"]["ppt_norm"] - ref_pipe["portfolio"]["ppt_norm"]
            row["passes"] = bool(
                d_port > 0.0001
                and met["new_momentum"]["n"] >= 15
                and met["new_momentum"]["ppt_norm"] >= ref_pipe["new_momentum"]["ppt_norm"] - 0.02
            )
        pipe_results.append(row)
        dt = time.time() - t_c
        print(f"  [{i+1}] {cfg['label'][:52]}")
        print(f"       port PPT={met['portfolio']['ppt_norm']:+.4f} | "
              f"new_mom N={met['n_new_momentum']:,} PPT={met['new_momentum']['ppt_norm']:+.4f} ({dt:.0f}s)")

    winners = [r for r in pipe_results if r.get("passes") is True]
    ranked = sorted(
        [r for r in pipe_results if r.get("label") not in ("baseline_no_lstm", "ref_05j_winner")],
        key=lambda x: (x.get("portfolio", {}).get("ppt_norm", -999), x.get("new_momentum", {}).get("ppt_norm", -999)),
        reverse=True,
    )

    print(f"\n{SEP}")
    print("  STAGE B SUMMARY (vs ref_05j_winner)")
    print(SEP)
    if ref_pipe:
        print(f"  REF 05j: port PPT={ref_pipe['portfolio']['ppt_norm']:+.4f} | "
              f"new_mom N={ref_pipe['n_new_momentum']:,} PPT={ref_pipe['new_momentum']['ppt_norm']:+.4f}")
    print(f"  {'Label':<46} {'newMomN':>7} {'newMomPPT':>10} {'portPPT':>9} {'dPort':>7} PASS")
    for r in ranked[:12]:
        nm = r.get("new_momentum", {})
        pas = "Y" if r.get("passes") is True else "N"
        print(f"  {r['label']:<46} {r.get('n_new_momentum',0):>7,} "
              f"{nm.get('ppt_norm',0):>+10.4f} {r['portfolio']['ppt_norm']:>+9.4f} "
              f"{r.get('delta_port_ppt',0):>+7.4f} {pas:>4}")

    elapsed = time.time() - t0
    out = {
        **genuine_audit_block(),
        "eval": "multi_coin_pump_v2",
        "created": datetime.now().isoformat(),
        "elapsed_sec": round(elapsed, 1),
        "n_configs": len(configs),
        "rally_loose_frac": RALLY_LOOSE_FRAC,
        "rally_bars_strict": len(rally_sets["strict"]),
        "rally_bars_loose": len(rally_sets["loose"]),
        "ref_05j_signal": ref_sig,
        "top_signal": top_signal,
        "ref_pipeline": ref_pipe,
        "baseline_pipeline": baseline_pipe,
        "pipeline_results": pipe_results,
        "winners": winners,
        "decision": "PROMOTE_CANDIDATE" if winners else "NO_PROMOTE",
        "best": ranked[0] if ranked else None,
    }
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\n  Decision: {out['decision']} | Elapsed: {elapsed/60:.1f} min")
    print(f"  Saved: {OUT_PATH}")
    print(f"{SEP}\n")


if __name__ == "__main__":
    main()