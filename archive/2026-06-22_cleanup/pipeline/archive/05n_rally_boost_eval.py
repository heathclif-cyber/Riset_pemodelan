"""
05n — Rally Boost eval (additive cross-coin ranking on pump bars).

Layer stack:
  1. ref 05j conditional_momentum (unchanged)
  2. rally boost: frac_up>=threshold, rank raw-LGBM-FLAT by p2, boost top-K

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
    apply_fused_scores, apply_hmm_thr, attach_cross_section, compute_dynamic_modal,
    genuine_audit_block, load_guardian_params, load_hmm_cfg, preload_coins,
    precompute_cross_section, rally_boost_label, build_predictions_full, summarize_trades,
)

OUT_PATH = LGBM_DIR / "rally_boost_eval.json"
MOMENTUM_VOL_THR = 2.0
TOP_K_PIPELINE = 12
RALLY_LOOSE_FRAC = 0.75

FROZEN_05J = {
    "fusion": "lstm",
    "mode": "conditional_momentum",
    "bull_thr": 0.38,
    "bear_thr": 0.50,
    "boost": 0.10,
    "opposite_pen": 0.14,
    "near_miss_gap": 0.03,
    "vol_thr": MOMENTUM_VOL_THR,
    "proportional": True,
    "enable_boost": True,
    "enable_penalty": True,
    "rally_boost_enabled": False,
}


def build_sweep_configs() -> list[dict]:
    configs = [
        {"fusion": "baseline", "label": "baseline_no_lstm"},
        {**FROZEN_05J, "label": "ref_05j_winner"},
    ]
    seen = {c["label"] for c in configs}

    for rf, rk, ra, lstm in itertools.product(
        [0.75, 0.80],
        [3, 5],
        [0.08, 0.10],
        [True, False],
    ):
        cfg = {
            **FROZEN_05J,
            "rally_boost_enabled": True,
            "rally_frac": rf,
            "rally_top_k": rk,
            "rally_boost_amt": ra,
            "rally_require_lstm": lstm,
        }
        cfg["label"] = rally_boost_label(cfg)
        if cfg["label"] not in seen:
            seen.add(cfg["label"])
            configs.append(cfg)
    return configs


def signal_stats(
    coins: list,
    y_map: dict,
    y_base: dict,
    y_ref: dict,
    rally_loose_ts: set,
    rally_fires: int = 0,
) -> dict:
    boost_unlock = penalty_block = rally_entry_unlock = 0
    n_long = n_short = 0
    per_bar_long = {}

    for c in coins:
        sym = c["sym"]
        y = y_map[sym]
        yb = y_base[sym]
        yr = y_ref[sym]
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
            if ts in rally_loose_ts and yr[i] != LONG and y[i] == LONG:
                rally_entry_unlock += 1

    return {
        "n_long": n_long, "n_short": n_short, "n_dir": n_long + n_short,
        "boost_unlock": boost_unlock,
        "penalty_block": penalty_block,
        "rally_entry_unlock": rally_entry_unlock,
        "rally_boost_fires": rally_fires,
        "multi_long_bars": sum(1 for n in per_bar_long.values() if n >= 2),
        "max_long_per_bar": max(per_bar_long.values()) if per_bar_long else 0,
    }


def rank_signal(row: dict, ref_row: dict) -> float:
    if row.get("fusion") == "baseline":
        return 0.0
    ref_dir = ref_row.get("n_dir", 1)
    ratio_pen = max(0.0, row["n_dir"] / ref_dir - 1.12) * 100.0
    return (
        row["rally_entry_unlock"] * 6.0
        + row["rally_boost_fires"] * 0.05
        + row["boost_unlock"] * 1.0
        - ratio_pen
    )


def eval_pipeline(coins, cfg, hmm_cfg, g_model, g_scaler, g_params, base_keys: set) -> dict:
    y_map, scores, rally_fires = build_predictions_full(coins, cfg, hmm_cfg)
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
            t2["rally_entry"] = (
                bi < len(c["frac_up"])
                and c["frac_up"][bi] >= cfg.get("rally_frac", RALLY_LOOSE_FRAC)
                and t2["momentum_entry"]
            )
            all_trades.append(t2)
    port = summarize_trades(all_trades)
    mom = summarize_trades([t for t in all_trades if t.get("momentum_entry")])
    rally = summarize_trades([t for t in all_trades if t.get("rally_entry")])
    new_t = summarize_trades([t for t in all_trades if t.get("is_new")])
    new_mom = summarize_trades([t for t in all_trades if t.get("is_new") and t.get("momentum_entry")])
    new_rally = summarize_trades([t for t in all_trades if t.get("is_new") and t.get("rally_entry")])
    return {
        "portfolio": port, "momentum": mom, "rally_subset": rally,
        "new_trades": new_t, "new_momentum": new_mom, "new_rally": new_rally,
        "n_new": new_t["n"], "n_new_momentum": new_mom["n"], "n_new_rally": new_rally["n"],
        "rally_boost_fires": rally_fires,
    }


def main():
    hmm_cfg = load_hmm_cfg()
    g_params = load_guardian_params()
    configs = build_sweep_configs()
    t0 = time.time()

    SEP = "=" * 78
    print(f"\n{SEP}")
    print("  05n: Rally Boost (additive cross-coin top-K on pump bars)")
    print(f"  Configs: {len(configs)} | Pipeline top-{TOP_K_PIPELINE}")
    print(SEP)

    lgbm_oof = pd.read_parquet(LGBM_DIR / "oof_predictions.parquet")
    lstm_oof = pd.read_parquet(LSTM_DIR / "oof_lstm_predictions.parquet")
    coins = preload_coins(lgbm_oof, lstm_oof)
    cs = precompute_cross_section(coins)
    attach_cross_section(coins, cs)
    rally_loose_ts = set(cs["frac_up"].index[cs["frac_up"] >= RALLY_LOOSE_FRAC])
    print(f"  Coins: {len(coins)} | Rally bars (frac_up>={RALLY_LOOSE_FRAC}): {len(rally_loose_ts):,}")

    y_base = {}
    for c in coins:
        y, _, _ = apply_fused_scores(c, {"fusion": "baseline"}, hmm_cfg)
        y_base[c["sym"]] = y

    ref_cfg = next(c for c in configs if c["label"] == "ref_05j_winner")
    y_ref, _, _ = build_predictions_full(coins, ref_cfg, hmm_cfg)
    ref_sig = signal_stats(coins, y_ref, y_base, y_ref, rally_loose_ts, 0)
    print(f"  Ref 05j: boost+={ref_sig['boost_unlock']:,} rally_entry+={ref_sig['rally_entry_unlock']:,}")

    signal_rows = []
    for cfg in configs:
        if cfg.get("fusion") == "baseline":
            sig = signal_stats(coins, y_base, y_base, y_ref, rally_loose_ts, 0)
            row = {"label": cfg["label"], **cfg, **sig, "rank": 0.0}
        else:
            y_map, _, rf = build_predictions_full(coins, cfg, hmm_cfg)
            sig = signal_stats(coins, y_map, y_base, y_ref, rally_loose_ts, rf)
            row = {"label": cfg["label"], **cfg, **sig, "rank": rank_signal(sig, ref_sig)}
        signal_rows.append(row)
    signal_rows.sort(key=lambda x: x["rank"], reverse=True)
    top_signal = [r for r in signal_rows if r.get("fusion") != "baseline"][:TOP_K_PIPELINE]

    print(f"\n  STAGE A — TOP 10 SIGNAL")
    print(f"  {'Label':<56} {'rally+':>6} {'fires':>6} {'n_dir':>7}")
    for r in top_signal[:10]:
        print(f"  {r['label']:<56} {r['rally_entry_unlock']:>6,} {r['rally_boost_fires']:>6,} {r['n_dir']:>7,}")

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
        "rally_boost_enabled", "rally_frac", "rally_top_k", "rally_boost_amt", "rally_require_lstm",
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
            d_port = met["portfolio"]["ppt_norm"] - ref_pipe["portfolio"]["ppt_norm"]
            row["delta_port_ppt"] = d_port
            row["delta_new_rally_ppt"] = (
                met["new_rally"]["ppt_norm"] - ref_pipe["new_rally"]["ppt_norm"]
                if met["n_new_rally"] > 0 else None
            )
            row["passes"] = bool(
                d_port > 0.0001
                and met["n_new_rally"] >= 10
                and met["new_rally"]["ppt_norm"] > 0.0
            )
        pipe_results.append(row)
        dt = time.time() - t_c
        print(f"  [{i+1}] {cfg['label'][:54]}")
        print(f"       port PPT={met['portfolio']['ppt_norm']:+.4f} | "
              f"new_rally N={met['n_new_rally']:,} PPT={met['new_rally']['ppt_norm']:+.4f} "
              f"fires={met['rally_boost_fires']:,} ({dt:.0f}s)")

    winners = [r for r in pipe_results if r.get("passes") is True]
    ranked = sorted(
        [r for r in pipe_results if r.get("label") not in ("baseline_no_lstm", "ref_05j_winner")],
        key=lambda x: (x.get("portfolio", {}).get("ppt_norm", -999), x.get("new_rally", {}).get("ppt_norm", -999)),
        reverse=True,
    )

    print(f"\n{SEP}")
    print("  STAGE B SUMMARY (vs ref_05j_winner)")
    print(SEP)
    if ref_pipe:
        print(f"  REF 05j: port PPT={ref_pipe['portfolio']['ppt_norm']:+.4f} | "
              f"new_rally N={ref_pipe['n_new_rally']:,}")
    print(f"  {'Label':<44} {'newRallyN':>9} {'newRallyPPT':>11} {'portPPT':>9} {'dPort':>7} PASS")
    for r in ranked[:10]:
        nr = r.get("new_rally", {})
        pas = "Y" if r.get("passes") is True else "N"
        print(f"  {r['label']:<44} {r.get('n_new_rally',0):>9,} "
              f"{nr.get('ppt_norm',0):>+11.4f} {r['portfolio']['ppt_norm']:>+9.4f} "
              f"{r.get('delta_port_ppt',0):>+7.4f} {pas:>4}")

    elapsed = time.time() - t0
    out = {
        **genuine_audit_block(),
        "eval": "rally_boost_additive",
        "created": datetime.now().isoformat(),
        "elapsed_sec": round(elapsed, 1),
        "n_configs": len(configs),
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