"""
05s — Slice eval pump/SHORT untuk 3 stack (genuine OOF).

Stacks:
  1. ref_genuine_v2   : tb_lstm_genuine_v2 + 05j frozen (opp_pen=0.14)
  2. seq_frozen       : tb_lstm_lgbm_seq_v1 + 05j frozen (opp_pen=0.14)
  3. seq_exp_a        : tb_lstm_lgbm_seq_v1 + Exp A winner (opp_pen=0.18)

Slices (signal + trade):
  - conflict: LGBM SHORT + LSTM bull>=0.38 pada vol>=2
  - pump_SHORT: sinyal/trade SHORT pada vol_spike>=2
  - rally_SHORT: sinyal/trade SHORT pada frac_up>=0.8 (wide pump)

Genuine: OOF LGBM+LSTM, HMM Config B frozen, Guardian OOF, holdout sealed.

Usage:
  python pipeline/05s_pump_short_slice_eval.py
"""
import importlib.util
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

from config import MODEL_DIR, TRAIN_CUTOFF_DATE
from pipeline.lstm_fusion_shared import (
    LGBM_DIR, GUARDIAN_DIR, DYNAMIC_FEATS, FLAT, LONG, SHORT,
    attach_cross_section, build_y_pred, genuine_audit_block,
    load_guardian_params, load_hmm_cfg, preload_coins,
    precompute_cross_section, summarize_trades,
)

_spec = importlib.util.spec_from_file_location(
    "eval05j", ROOT / "pipeline" / "05j_lstm_conditional_momentum_eval.py"
)
_eval05j = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_eval05j)

VOL_THR = 2.0
BULL_THR = 0.38
RALLY_FRAC = 0.8
DUMP_FRAC = 0.2   # frac_up <= 0.2 ~ 80% koin turun (simetris rally)
OUT_PATH = LGBM_DIR / "pump_short_slice_eval.json"

STACKS = [
    {
        "id": "ref_genuine_v2",
        "lstm_run": "tb_lstm_genuine_v2",
        "fusion": {
            "fusion": "lstm", "mode": "conditional_momentum",
            "bull_thr": 0.38, "bear_thr": 0.50, "boost": 0.10,
            "opposite_pen": 0.14, "near_miss_gap": 0.03, "vol_thr": VOL_THR,
            "proportional": True, "enable_boost": True, "enable_penalty": True,
        },
    },
    {
        "id": "seq_frozen_o14",
        "lstm_run": "tb_lstm_lgbm_seq_v1",
        "fusion": {
            "fusion": "lstm", "mode": "conditional_momentum",
            "bull_thr": 0.38, "bear_thr": 0.50, "boost": 0.10,
            "opposite_pen": 0.14, "near_miss_gap": 0.03, "vol_thr": VOL_THR,
            "proportional": True, "enable_boost": True, "enable_penalty": True,
        },
    },
    {
        "id": "seq_exp_a_o18",
        "lstm_run": "tb_lstm_lgbm_seq_v1",
        "fusion": {
            "fusion": "lstm", "mode": "conditional_momentum",
            "bull_thr": 0.38, "bear_thr": 0.50, "boost": 0.10,
            "opposite_pen": 0.18, "near_miss_gap": 0.03, "vol_thr": VOL_THR,
            "proportional": True, "enable_boost": True, "enable_penalty": True,
        },
    },
]


def signal_slices(coins: list, cfg: dict, hmm_cfg: dict, cs: dict) -> dict:
    attach_cross_section(coins, cs)
    y_base = {c["sym"]: build_y_pred(c, {"fusion": "baseline"}, hmm_cfg) for c in coins}
    y_fused = {c["sym"]: build_y_pred(c, cfg, hmm_cfg) for c in coins}

    boost = pen = 0
    conflict_short = conflict_short_pen = 0
    conflict_long = conflict_long_pen = 0
    rally_short_base = rally_short_fused = 0
    dump_long_base = dump_long_fused = 0
    pump_short_base = pump_short_fused = 0
    wide_dump_long_base = wide_dump_long_fused = 0

    bear_thr = cfg.get("bear_thr", 0.50)

    for c in coins:
        yb, yf = y_base[c["sym"]], y_fused[c["sym"]]
        mom = c["vol_spike"] >= VOL_THR
        rally = c["frac_up"] >= RALLY_FRAC
        dump_wide = c["frac_up"] <= DUMP_FRAC
        lstm_bull = c["lstm_p"][:, 2] >= BULL_THR
        lstm_bear = c["lstm_p"][:, 0] >= bear_thr

        for i in range(len(yb)):
            if not c["lstm_valid"][i]:
                continue
            if mom[i]:
                if yb[i] == FLAT and yf[i] != FLAT:
                    boost += 1
                if yb[i] != FLAT and yf[i] == FLAT:
                    pen += 1
                if yb[i] == SHORT:
                    pump_short_base += 1
                    if yf[i] == SHORT:
                        pump_short_fused += 1
                    if lstm_bull[i]:
                        conflict_short += 1
                        if yf[i] == FLAT:
                            conflict_short_pen += 1
                if yb[i] == LONG:
                    dump_long_base += 1
                    if yf[i] == LONG:
                        dump_long_fused += 1
                    if lstm_bear[i]:
                        conflict_long += 1
                        if yf[i] == FLAT:
                            conflict_long_pen += 1
            if rally[i]:
                if yb[i] == SHORT:
                    rally_short_base += 1
                if yf[i] == SHORT:
                    rally_short_fused += 1
            if dump_wide[i]:
                if yb[i] == LONG:
                    wide_dump_long_base += 1
                if yf[i] == LONG:
                    wide_dump_long_fused += 1

    return {
        "boost_unlock": boost,
        "penalty_block": pen,
        "pump_short_signals_base": pump_short_base,
        "pump_short_signals_fused": pump_short_fused,
        "pump_short_blocked": pump_short_base - pump_short_fused,
        "dump_long_signals_base": dump_long_base,
        "dump_long_signals_fused": dump_long_fused,
        "dump_long_blocked": dump_long_base - dump_long_fused,
        "rally_short_signals_base": rally_short_base,
        "rally_short_signals_fused": rally_short_fused,
        "rally_short_blocked": rally_short_base - rally_short_fused,
        "wide_dump_long_base": wide_dump_long_base,
        "wide_dump_long_fused": wide_dump_long_fused,
        "wide_dump_long_blocked": wide_dump_long_base - wide_dump_long_fused,
        "conflict_short_bars": conflict_short,
        "conflict_short_penalized": conflict_short_pen,
        "conflict_short_pen_pct": round(conflict_short_pen / max(conflict_short, 1) * 100, 1),
        "conflict_long_bars": conflict_long,
        "conflict_long_penalized": conflict_long_pen,
        "conflict_long_pen_pct": round(conflict_long_pen / max(conflict_long, 1) * 100, 1),
        # backward compat
        "conflict_bars": conflict_short,
        "conflict_penalized": conflict_short_pen,
        "conflict_pen_pct": round(conflict_short_pen / max(conflict_short, 1) * 100, 1),
    }


def collect_trades(coins_p, cfg, hmm_cfg, g_model, g_scaler, g_params) -> list:
    from config import (
        MODAL_PER_TRADE, LEVERAGE_SIM, FEE_PER_SIDE, SLIPPAGE_PER_SIDE,
        SWING_LABEL_MIN_RR, SWING_LABEL_MIN_TP, SWING_LABEL_MAX_SL,
        TP_SL_FALLBACK_TP, TP_SL_FALLBACK_SL, MAX_HOLDING_BARS,
        GUARDIAN_ACTIVATION_ATR,
    )
    from core.cascade_utils import apply_conditional_momentum_fusion_pre
    from core.evaluator import simulate_trades_swing
    from pipeline.lstm_fusion_shared import apply_hmm_thr, compute_dynamic_modal, DYNSIZE_CFG

    all_trades = []
    common = dict(
        modal=MODAL_PER_TRADE, leverage=LEVERAGE_SIM[0],
        fee_per_side=FEE_PER_SIDE, slippage=SLIPPAGE_PER_SIDE,
        max_hold=MAX_HOLDING_BARS,
        min_rr=SWING_LABEL_MIN_RR, min_tp_atr=SWING_LABEL_MIN_TP,
        max_sl_atr=SWING_LABEL_MAX_SL,
        tp_fallback_atr=TP_SL_FALLBACK_TP, sl_fallback_atr=TP_SL_FALLBACK_SL,
    )
    for c in coins_p:
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
            bi = t["bar_in"]
            t2 = dict(t)
            t2["sym"] = c["sym"]
            t2["vol_spike_entry"] = float(c["vol_spike"][bi])
            t2["frac_up_entry"] = float(c["frac_up"][bi])
            t2["is_pump_short"] = t["direction"] == "SHORT" and c["vol_spike"][bi] >= VOL_THR
            t2["is_rally_short"] = t["direction"] == "SHORT" and c["frac_up"][bi] >= RALLY_FRAC
            t2["is_dump_long"] = t["direction"] == "LONG" and c["vol_spike"][bi] >= VOL_THR
            t2["is_wide_dump_long"] = t["direction"] == "LONG" and c["frac_up"][bi] <= DUMP_FRAC
            all_trades.append(t2)
    return all_trades


def trade_slices(trades: list) -> dict:
    pump_short = [t for t in trades if t["is_pump_short"]]
    rally_short = [t for t in trades if t["is_rally_short"]]
    dump_long = [t for t in trades if t["is_dump_long"]]
    wide_dump_long = [t for t in trades if t["is_wide_dump_long"]]
    all_short = [t for t in trades if t["direction"] == "SHORT"]
    all_long = [t for t in trades if t["direction"] == "LONG"]
    port = summarize_trades(trades)
    return {
        "portfolio_all": port,
        "all_short": summarize_trades(all_short),
        "all_long": summarize_trades(all_long),
        "pump_short_trades": summarize_trades(pump_short),
        "rally_short_trades": summarize_trades(rally_short),
        "dump_long_trades": summarize_trades(dump_long),
        "wide_dump_long_trades": summarize_trades(wide_dump_long),
        "n_pump_short": len(pump_short),
        "n_rally_short": len(rally_short),
        "n_dump_long": len(dump_long),
        "n_wide_dump_long": len(wide_dump_long),
    }


def main():
    t0 = time.time()
    hmm_cfg = load_hmm_cfg()
    g_params = load_guardian_params()
    lgbm_oof = pd.read_parquet(LGBM_DIR / "oof_predictions.parquet")

    g_model = joblib.load(GUARDIAN_DIR / "guardian.pkl")
    g_scaler = joblib.load(GUARDIAN_DIR / "guardian_scaler.pkl")
    with open(GUARDIAN_DIR / "guardian_features.json", encoding="utf-8") as f:
        g_static = [feat for feat in json.load(f) if feat not in DYNAMIC_FEATS]

    print("\n" + "=" * 78)
    print("  05s: Pump/SHORT + Dump/LONG slice eval (genuine OOF)")
    print(f"  Stacks: {len(STACKS)} | vol_thr={VOL_THR} | rally>={RALLY_FRAC} | dump<={DUMP_FRAC}")
    print("=" * 78)

    results = []
    for stack in STACKS:
        print(f"\n  [{stack['id']}] LSTM={stack['lstm_run']} opp_pen={stack['fusion']['opposite_pen']}")
        lstm_oof = pd.read_parquet(MODEL_DIR / "runs" / stack["lstm_run"] / "oof_lstm_predictions.parquet")
        coins = preload_coins(lgbm_oof, lstm_oof)
        cs = precompute_cross_section(coins)
        sig = signal_slices(coins, stack["fusion"], hmm_cfg, cs)

        coins_p = preload_coins(lgbm_oof, lstm_oof, g_static)
        attach_cross_section(coins_p, cs)
        t1 = time.time()
        trades = collect_trades(coins_p, stack["fusion"], hmm_cfg, g_model, g_scaler, g_params)
        tr = trade_slices(trades)
        print(f"    pump/SHORT: conflict={sig['conflict_short_bars']} pen={sig['conflict_short_penalized']} "
              f"({sig['conflict_short_pen_pct']}%) block={sig['pump_short_blocked']}")
        print(f"    dump/LONG:  conflict={sig['conflict_long_bars']} pen={sig['conflict_long_penalized']} "
              f"({sig['conflict_long_pen_pct']}%) block={sig['dump_long_blocked']} "
              f"| wide_dump block={sig['wide_dump_long_blocked']}")
        ps = tr["pump_short_trades"]
        dl = tr["dump_long_trades"]
        print(f"    trades: pump_SHORT n={tr['n_pump_short']} WR={ps['wr']:.1f}% PPT={ps['ppt_norm']:+.4f}")
        print(f"            dump_LONG  n={tr['n_dump_long']} WR={dl['wr']:.1f}% PPT={dl['ppt_norm']:+.4f} "
              f"({time.time()-t1:.0f}s)")

        results.append({
            **stack,
            "signal_slices": sig,
            "trade_slices": tr,
        })

    audit = genuine_audit_block()
    out = {
        **audit,
        "created": datetime.now().isoformat(),
        "train_cutoff": str(TRAIN_CUTOFF_DATE.date()),
        "vol_thr": VOL_THR,
        "rally_frac": RALLY_FRAC,
        "dump_frac": DUMP_FRAC,
        "stacks": results,
        "elapsed_sec": round(time.time() - t0, 1),
    }
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=str)

    print("\n" + "=" * 78)
    print("  PUMP/SHORT")
    print(f"  {'Stack':<22} {'S_conf%':>7} {'pump_blk':>8} {'pump_WR':>7} {'pump_PPT':>9}")
    for r in results:
        s, t = r["signal_slices"], r["trade_slices"]
        ps = t["pump_short_trades"]
        print(
            f"  {r['id']:<22} {s['conflict_short_pen_pct']:>6.1f}% {s['pump_short_blocked']:>8} "
            f"{ps['wr']:>6.1f}% {ps['ppt_norm']:>+9.4f}"
        )
    print("\n  DUMP/LONG (simetris)")
    print(f"  {'Stack':<22} {'L_conf%':>7} {'dump_blk':>8} {'wide_blk':>8} {'dump_WR':>7} {'dump_PPT':>9}")
    for r in results:
        s, t = r["signal_slices"], r["trade_slices"]
        dl = t["dump_long_trades"]
        print(
            f"  {r['id']:<22} {s['conflict_long_pen_pct']:>6.1f}% {s['dump_long_blocked']:>8} "
            f"{s['wide_dump_long_blocked']:>8} {dl['wr']:>6.1f}% {dl['ppt_norm']:>+9.4f}"
        )
    print(f"\n  Saved: {OUT_PATH}")
    print("=" * 78 + "\n")


if __name__ == "__main__":
    main()