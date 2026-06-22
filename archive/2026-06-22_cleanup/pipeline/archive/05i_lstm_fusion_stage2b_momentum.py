"""
Stage 2b — LSTM momentum overlay OOF eval (genuine, strict).

Fokus: pump/dump gate ONLY (vol_spike2 / pump_dump), boost-only + targeted pre_hmm.
Metrik utama: PPT pada trade subset momentum (entry bar vol_spike>=2).
Metrik sekunder: portfolio PPT_norm (delta kecil acceptable).

Genuine: OOF only, TRAIN_CUTOFF, HMM Config B frozen, holdout NOT touched.
"""
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
from core.cascade_utils import apply_lstm_boost_only_pre, apply_lstm_proba_fusion_pre
from core.evaluator import simulate_trades_swing
from pipeline.lstm_fusion_shared import (
    LGBM_DIR, LSTM_DIR, GUARDIAN_DIR, DYNAMIC_FEATS, DYNSIZE_CFG, FLAT, LONG, SHORT,
    apply_hmm_thr, build_y_pred, config_label, genuine_audit_block,
    compute_dynamic_modal, gate_mask, load_guardian_params, load_hmm_cfg,
    preload_coins, summarize_trades,
)

STAGE2B_OUT = LGBM_DIR / "lstm_fusion_stage2b_momentum.json"
MOMENTUM_VOL_THR = 2.0
PORTFOLIO_PPT_MAX_DROP = 0.005
MOMENTUM_PPT_MIN_GAIN = 0.001

CANDIDATES = [
    {"fusion": "baseline", "label": "baseline_no_lstm"},
    # vol_spike2 pre_hmm (Stage 1 top, penalties included)
    {"fusion": "lstm", "mode": "pre_hmm", "gate": "vol_spike2",
     "agree_boost": 0.08, "neutral_pen": 0.04, "opposite_pen": 0.06},
    {"fusion": "lstm", "mode": "pre_hmm", "gate": "vol_spike2",
     "agree_boost": 0.10, "neutral_pen": 0.04, "opposite_pen": 0.06},
    {"fusion": "lstm", "mode": "pre_hmm", "gate": "vol_spike2",
     "agree_boost": 0.08, "neutral_pen": 0.04, "opposite_pen": 0.10},
    # boost-only asymmetric (momentum overlay design)
    {"fusion": "lstm", "mode": "boost_only", "gate": "vol_spike2",
     "boost_side": "both", "agree_boost": 0.06},
    {"fusion": "lstm", "mode": "boost_only", "gate": "vol_spike2",
     "boost_side": "both", "agree_boost": 0.08},
    {"fusion": "lstm", "mode": "boost_only", "gate": "vol_spike2",
     "boost_side": "both", "agree_boost": 0.10},
    {"fusion": "lstm", "mode": "boost_only", "gate": "vol_spike2",
     "boost_side": "long", "agree_boost": 0.08},
    {"fusion": "lstm", "mode": "boost_only", "gate": "vol_spike2",
     "boost_side": "short", "agree_boost": 0.08},
    {"fusion": "lstm", "mode": "boost_only", "gate": "pump_dump",
     "boost_side": "both", "agree_boost": 0.08},
]
for c in CANDIDATES:
    if c.get("fusion") != "baseline":
        c["label"] = config_label(c)


def build_entry_arrays(coin: dict, cfg: dict, y: np.ndarray, hmm_cfg: dict):
    p0, p2 = coin["p0"].copy(), coin["p2"].copy()
    active = gate_mask(
        coin["vol_spike"], coin["lstm_valid"], cfg.get("gate", "vol_spike2"),
        coin.get("is_gate"),
    )
    if cfg.get("fusion") != "baseline":
        if cfg.get("mode") == "boost_only":
            p0, p2 = apply_lstm_boost_only_pre(
                p0, p2, coin["lstm_p"],
                agree_boost=cfg["agree_boost"],
                active_mask=active,
                boost_side=cfg.get("boost_side", "both"),
            )
        elif cfg.get("mode") == "pre_hmm":
            p0, p2 = apply_lstm_proba_fusion_pre(
                p0, p2, coin["lstm_p"],
                agree_boost=cfg["agree_boost"],
                neutral_pen=cfg["neutral_pen"],
                opposite_pen=cfg["opposite_pen"],
                active_mask=active,
            )
    _, conf, tl, ts = apply_hmm_thr(p0, p2, coin["hmm"], hmm_cfg)
    p0e = np.zeros_like(p0)
    p2e = np.zeros_like(p2)
    p0e[y == SHORT] = conf[y == SHORT]
    p2e[y == LONG] = conf[y == LONG]
    if cfg.get("fusion") != "baseline" and cfg.get("mode") in ("pre_hmm", "boost_only"):
        p0e, p2e = p0, p2
    return p0e, p2e, tl, ts


def tag_momentum_trades(trades: list, vol_spike: np.ndarray) -> list:
    out = []
    for t in trades:
        bi = t.get("bar_in", 0)
        t2 = dict(t)
        t2["momentum_entry"] = bool(bi < len(vol_spike) and vol_spike[bi] >= MOMENTUM_VOL_THR)
        out.append(t2)
    return out


def eval_config(coins: list, cfg: dict, hmm_cfg: dict, g_model, g_scaler, g_params) -> list:
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
        p0e, p2e, tl, ts = build_entry_arrays(c, cfg, y, hmm_cfg)
        modal_arr = compute_dynamic_modal(
            p0e, p2e, c["hmm"], y, MODAL_PER_TRADE, DYNSIZE_CFG, tl, ts,
        )
        rep = simulate_trades_swing(
            y_pred=y, guardian_enabled=True,
            guardian_model=g_model, guardian_scaler=g_scaler,
            X_guardian=c["X_grd"],
            guardian_exit_threshold=g_params["exit_threshold"],
            guardian_min_hold_bars=g_params["min_hold_bars"],
            guardian_activation_atr=GUARDIAN_ACTIVATION_ATR,
            modal_arr=modal_arr,
            close=c["close"], high=c["high"], low=c["low"], atr=c["atr"],
            h4_swing_highs=c["h4_sh"], h4_swing_lows=c["h4_sl"],
            **common,
        )
        tagged = tag_momentum_trades(rep.get("trades", []), c["vol_spike"])
        all_trades.extend(tagged)
    return all_trades


def momentum_subset_summary(trades: list) -> dict:
    mom = [t for t in trades if t.get("momentum_entry")]
    return summarize_trades(mom)


def passes_momentum_gate(variant: dict, baseline: dict) -> bool:
    vm = variant["momentum"]
    bm = baseline["momentum"]
    vp = variant["portfolio"]
    bp = baseline["portfolio"]
    mom_ok = (
        vm["n"] >= 50
        and vm["ppt_norm"] >= bm["ppt_norm"] + MOMENTUM_PPT_MIN_GAIN
    )
    port_ok = vp["ppt_norm"] >= bp["ppt_norm"] - PORTFOLIO_PPT_MAX_DROP
    return mom_ok and port_ok


def main():
    SEP = "=" * 78
    hmm_cfg = load_hmm_cfg()
    g_params = load_guardian_params()

    print(f"\n{SEP}")
    print("  STAGE 2b: LSTM Momentum Overlay — OOF genuine (strict)")
    print(f"  Gate: vol_spike2 / pump_dump | Metrics: momentum subset PRIMARY")
    print(f"  Stack: HMM Config B + Guardian + DynSize | holdout sealed")
    print(f"  Candidates: {len(CANDIDATES)}")
    print(SEP)

    g_model = joblib.load(GUARDIAN_DIR / "guardian.pkl")
    g_scaler = joblib.load(GUARDIAN_DIR / "guardian_scaler.pkl")
    with open(GUARDIAN_DIR / "guardian_features.json", encoding="utf-8") as f:
        g_static = [f for f in json.load(f) if f not in DYNAMIC_FEATS]

    lgbm_oof = pd.read_parquet(LGBM_DIR / "oof_predictions.parquet")
    lstm_oof = pd.read_parquet(LSTM_DIR / "oof_lstm_predictions.parquet")
    coins = preload_coins(lgbm_oof, lstm_oof, g_static)
    mom_bars = sum(int((c["vol_spike"] >= MOMENTUM_VOL_THR).sum()) for c in coins)
    print(f"  Coins: {len(coins)} | bars vol_spike>={MOMENTUM_VOL_THR}: {mom_bars:,}")

    t0 = time.time()
    results = []
    baseline_row = None

    for i, cfg in enumerate(CANDIDATES):
        t_cfg = time.time()
        trades = eval_config(coins, cfg, hmm_cfg, g_model, g_scaler, g_params)
        port = summarize_trades(trades)
        mom = momentum_subset_summary(trades)
        row = {
            "label": cfg["label"], **cfg,
            "portfolio": port,
            "momentum": mom,
            "momentum_n": mom["n"],
            "momentum_ppt_norm": mom["ppt_norm"],
            "portfolio_ppt_norm": port["ppt_norm"],
        }
        if cfg.get("fusion") == "baseline":
            baseline_row = row
        elif baseline_row:
            row["delta_momentum_ppt"] = mom["ppt_norm"] - baseline_row["momentum_ppt_norm"]
            row["delta_portfolio_ppt"] = port["ppt_norm"] - baseline_row["portfolio_ppt_norm"]
            row["delta_n"] = port["n"] - baseline_row["portfolio"]["n"]
            row["delta_momentum_n"] = mom["n"] - baseline_row["momentum_n"]
            row["passes_momentum_gate"] = passes_momentum_gate(row, baseline_row)
        results.append(row)
        dt = time.time() - t_cfg
        print(f"  [{i+1}/{len(CANDIDATES)}] {cfg['label']}")
        print(f"       portfolio: N={port['n']:,} PPT_norm={port['ppt_norm']:+.4f} | "
              f"momentum: N={mom['n']:,} PPT_norm={mom['ppt_norm']:+.4f} ({dt:.0f}s)")

    ranked = sorted(
        [r for r in results if r.get("fusion") != "baseline"],
        key=lambda x: x["momentum_ppt_norm"], reverse=True,
    )
    winners = [r for r in ranked if r.get("passes_momentum_gate")]

    print(f"\n{SEP}")
    print("  RESULTS — MOMENTUM SUBSET (PRIMARY)")
    print(SEP)
    b = baseline_row
    print(f"  BASELINE momentum: N={b['momentum_n']:,} PPT_norm={b['momentum_ppt_norm']:+.4f} "
          f"PF={b['momentum']['pf']:.3f}")
    print(f"  BASELINE portfolio: N={b['portfolio']['n']:,} PPT_norm={b['portfolio_ppt_norm']:+.4f}")
    print(f"\n  {'Label':<38} {'momN':>5} {'dmomN':>6} {'momPPT':>8} {'dmomPPT':>8} "
          f"{'portPPT':>8} {'dport':>7} {'PASS':>4}")
    for r in ranked:
        pas = "Y" if r.get("passes_momentum_gate") else "N"
        print(f"  {r['label']:<38} {r['momentum_n']:>5,} {r.get('delta_momentum_n',0):>+6,} "
              f"{r['momentum_ppt_norm']:>+8.4f} {r.get('delta_momentum_ppt',0):>+8.4f} "
              f"{r['portfolio_ppt_norm']:>+8.4f} {r.get('delta_portfolio_ppt',0):>+7.4f} {pas:>4}")

    decision = "NO_PROMOTE"
    best = winners[0] if winners else (ranked[0] if ranked else None)
    if winners:
        decision = "PROMOTE_CANDIDATE"
        print(f"\n  WINNER: {winners[0]['label']}")
        print(f"    momentum PPT_norm {winners[0]['momentum_ppt_norm']:+.4f} "
              f"(delta {winners[0]['delta_momentum_ppt']:+.4f})")
        print(f"    portfolio PPT_norm {winners[0]['portfolio_ppt_norm']:+.4f} "
              f"(delta {winners[0]['delta_portfolio_ppt']:+.4f})")
    elif best:
        print(f"\n  No winner. Best momentum PPT: {best['label']}")

    elapsed = time.time() - t0
    out = {
        **genuine_audit_block(),
        "stage": "2b",
        "stage_name": "momentum_overlay_subset_eval",
        "created": datetime.now().isoformat(),
        "elapsed_sec": round(elapsed, 1),
        "momentum_vol_thr": MOMENTUM_VOL_THR,
        "gates": {
            "momentum_ppt_min_gain": MOMENTUM_PPT_MIN_GAIN,
            "portfolio_ppt_max_drop": PORTFOLIO_PPT_MAX_DROP,
        },
        "baseline": baseline_row,
        "decision": decision,
        "n_winners": len(winners),
        "best": best,
        "winners": winners,
        "all_results": results,
    }
    with open(STAGE2B_OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\n  Elapsed: {elapsed/60:.1f} min | Saved: {STAGE2B_OUT}")
    print(f"{SEP}\n")


if __name__ == "__main__":
    main()