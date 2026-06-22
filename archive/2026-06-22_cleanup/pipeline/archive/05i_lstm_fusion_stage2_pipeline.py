"""
Stage 2 — LSTM score fusion: FULL PIPELINE OOF eval (top candidates only).

Input: lstm_fusion_stage1_signal.json top candidates + baseline.
Stack: HMM Config B + Guardian v2 + DynSize cm_0.60 (frozen).

Genuine: OOF only, holdout NOT touched.
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
from core.evaluator import simulate_trades_swing
from pipeline.lstm_fusion_shared import (
    LGBM_DIR, LSTM_DIR, GUARDIAN_DIR, LGBM_RUN, LSTM_RUN, GUARDIAN_RUN,
    DYNAMIC_FEATS, DYNSIZE_CFG, FLAT, LONG, SHORT,
    apply_hmm_thr, build_y_pred, config_label, genuine_audit_block,
    compute_dynamic_modal, load_guardian_params, load_hmm_cfg,
    preload_coins, summarize_trades,
)

STAGE1_OUT = LGBM_DIR / "lstm_fusion_stage1_signal.json"
STAGE2_OUT = LGBM_DIR / "lstm_fusion_stage2_pipeline.json"
PPT_NORM_GATE = 0.002
MIN_TRADE_RATIO = 0.80


def build_entry_conf(coin: dict, cfg: dict, y: np.ndarray, hmm_cfg: dict):
    p0, p2 = coin["p0"].copy(), coin["p2"].copy()
    if cfg.get("fusion") != "baseline" and cfg.get("mode") == "pre_hmm":
        from pipeline.lstm_fusion_shared import gate_mask
        from core.cascade_utils import apply_lstm_proba_fusion_pre
        active = gate_mask(coin["vol_spike"], coin["lstm_valid"], cfg.get("gate", "all_oof"))
        p0, p2 = apply_lstm_proba_fusion_pre(
            p0, p2, coin["lstm_p"],
            agree_boost=cfg["agree_boost"], neutral_pen=cfg["neutral_pen"],
            opposite_pen=cfg["opposite_pen"], active_mask=active,
        )
    _, conf, tl, ts = apply_hmm_thr(p0, p2, coin["hmm"], hmm_cfg)
    if cfg.get("fusion") != "baseline" and cfg.get("mode") == "post_hmm":
        from pipeline.lstm_fusion_shared import gate_mask
        from core.cascade_utils import apply_lstm_conf_fusion_post
        active = gate_mask(coin["vol_spike"], coin["lstm_valid"], cfg.get("gate", "all_oof"))
        y, conf = apply_lstm_conf_fusion_post(
            y, conf, coin["lstm_p"], tl, ts,
            agree_boost=cfg["agree_boost"], neutral_pen=cfg["neutral_pen"],
            opposite_pen=cfg["opposite_pen"], active_mask=active,
        )
    p0_eff = np.zeros_like(p0)
    p2_eff = np.zeros_like(p2)
    p0_eff[y == SHORT] = conf[y == SHORT]
    p2_eff[y == LONG] = conf[y == LONG]
    if cfg.get("fusion") != "baseline" and cfg.get("mode") == "pre_hmm":
        p0_eff, p2_eff = p0, p2
    return p0_eff, p2_eff, tl, ts


def eval_pipeline(coins: list, cfg: dict, hmm_cfg: dict, g_model, g_scaler, g_params) -> list:
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
        p0e, p2e, tl, ts = build_entry_conf(c, cfg, y, hmm_cfg)
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
        all_trades.extend(rep.get("trades", []))
    return all_trades


def passes_gate(variant: dict, baseline: dict) -> bool:
    trade_ratio = variant["n"] / baseline["n"] if baseline["n"] else 0
    return (
        variant["ppt_norm"] >= baseline["ppt_norm"] + PPT_NORM_GATE
        and trade_ratio >= MIN_TRADE_RATIO
        and variant["pf"] >= baseline["pf"] * 0.98
    )


def main():
    SEP = "=" * 78
    if not STAGE1_OUT.exists():
        raise FileNotFoundError(
            f"Stage 1 belum dijalankan: {STAGE1_OUT}\n"
            "Jalankan: python pipeline/05i_lstm_fusion_stage1_signal.py"
        )

    with open(STAGE1_OUT, encoding="utf-8") as f:
        stage1 = json.load(f)

    hmm_cfg = load_hmm_cfg()
    g_params = load_guardian_params()
    candidates = [{"fusion": "baseline", "label": "baseline_no_lstm"}]
    for c in stage1["top_candidates"]:
        candidates.append({
            "fusion": "lstm", "mode": c["mode"], "gate": c["gate"],
            "agree_boost": c["agree_boost"], "neutral_pen": c["neutral_pen"],
            "opposite_pen": c["opposite_pen"], "label": c["label"],
            "stage1_rank_score": c.get("rank_score"),
            "stage1_rally_unlock": c.get("rally_unlock_long"),
        })

    print(f"\n{SEP}")
    print("  STAGE 2: LSTM Score Fusion — FULL PIPELINE (OOF genuine)")
    print(f"  Candidates: {len(candidates)} (baseline + top-{len(candidates)-1} from Stage 1)")
    print(f"  Stack: HMM-B + Guardian + DynSize cm_0.60 | holdout sealed")
    print(SEP)

    g_model = joblib.load(GUARDIAN_DIR / "guardian.pkl")
    g_scaler = joblib.load(GUARDIAN_DIR / "guardian_scaler.pkl")
    with open(GUARDIAN_DIR / "guardian_features.json", encoding="utf-8") as f:
        g_all = json.load(f)
    g_static = [f for f in g_all if f not in DYNAMIC_FEATS]

    lgbm_oof = pd.read_parquet(LGBM_DIR / "oof_predictions.parquet")
    lstm_oof = pd.read_parquet(LSTM_DIR / "oof_lstm_predictions.parquet")
    print("  Preloading coins...")
    coins = preload_coins(lgbm_oof, lstm_oof, g_static)
    print(f"  Coins: {len(coins)}")

    t0 = time.time()
    results = []
    baseline_sm = None

    for i, cfg in enumerate(candidates):
        t_cfg = time.time()
        trades = eval_pipeline(coins, cfg, hmm_cfg, g_model, g_scaler, g_params)
        sm = summarize_trades(trades)
        row = {"label": cfg["label"], **cfg, **sm}
        if cfg.get("fusion") == "baseline":
            baseline_sm = sm
        elif baseline_sm:
            row["delta_ppt_norm"] = sm["ppt_norm"] - baseline_sm["ppt_norm"]
            row["delta_n"] = sm["n"] - baseline_sm["n"]
            row["passes_gate"] = passes_gate(sm, baseline_sm)
        results.append(row)
        dt = time.time() - t_cfg
        print(f"  [{i+1}/{len(candidates)}] {cfg['label']}: "
              f"N={sm['n']:,} PPT_norm={sm['ppt_norm']:+.4f} PF={sm['pf']:.3f} ({dt:.0f}s)")

    ranked = sorted(
        [r for r in results if r.get("fusion") != "baseline"],
        key=lambda x: x["ppt_norm"], reverse=True,
    )
    winners = [r for r in ranked if r.get("passes_gate")]

    print(f"\n{SEP}")
    print("  STAGE 2 RESULTS vs BASELINE")
    print(SEP)
    b = baseline_sm
    print(f"  BASELINE: N={b['n']:,} WR={b['wr']:.1f}% PPT_norm={b['ppt_norm']:+.4f} PF={b['pf']:.3f}")
    print(f"\n  {'Label':<42} {'N':>7} {'dN':>6} {'PPT_norm':>8} {'dPPT':>7} {'PF':>5} {'PASS':>4}")
    for r in ranked:
        pf = f"{r['pf']:.3f}" if r["pf"] != float("inf") else " INF"
        pas = "Y" if r.get("passes_gate") else "N"
        print(f"  {r['label']:<42} {r['n']:>7,} {r.get('delta_n',0):>+6,} "
              f"{r['ppt_norm']:>+8.4f} {r.get('delta_ppt_norm',0):>+7.4f} {pf:>5} {pas:>4}")

    elapsed = time.time() - t0
    decision = "NO_PROMOTE"
    best = winners[0] if winners else (ranked[0] if ranked else None)
    if winners:
        decision = "PROMOTE_CANDIDATE"
        print(f"\n  WINNER (passes gate): {winners[0]['label']}")
    elif best:
        print(f"\n  No config passes gate. Best PPT_norm: {best['label']}")

    out = {
        **genuine_audit_block(),
        "stage": 2,
        "stage_name": "full_pipeline_top_candidates",
        "created": datetime.now().isoformat(),
        "elapsed_sec": round(elapsed, 1),
        "stage1_source": str(STAGE1_OUT),
        "baseline": baseline_sm,
        "ppt_norm_gate": PPT_NORM_GATE,
        "min_trade_ratio": MIN_TRADE_RATIO,
        "n_candidates": len(candidates),
        "n_winners": len(winners),
        "decision": decision,
        "best": best,
        "winners": winners,
        "all_results": results,
        "guardian_retrain_needed": (
            best is not None and baseline_sm is not None
            and abs(best.get("delta_n", 0)) > baseline_sm["n"] * 0.05
        ),
    }
    with open(STAGE2_OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\n  Elapsed: {elapsed/60:.1f} min")
    print(f"  Saved: {STAGE2_OUT}")
    if out["guardian_retrain_needed"]:
        print("  NOTE: trade mix changed >5% — Guardian retrain recommended before deploy")
    print(f"{SEP}\n")


if __name__ == "__main__":
    main()