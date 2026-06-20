"""
core/meta_labeling.py — Shared helpers for ic32 meta-labeling pipeline.

Spec: pipeline/meta_label_spec.json
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
SPEC_PATH = ROOT / "pipeline" / "meta_label_spec.json"


def load_spec() -> dict:
    with open(SPEC_PATH, encoding="utf-8") as f:
        return json.load(f)


def directional_candidate(proba: np.ndarray, candidate_thr: float = 0.45) -> tuple[int, float]:
    """
    Pick direction from OOF LGBM proba using candidate threshold.
    Returns (sig, confidence): sig in {0=SHORT, 2=LONG, -1=FLAT}.
    """
    p_short, _p_flat, p_long = float(proba[0]), float(proba[1]), float(proba[2])
    if p_long >= candidate_thr and p_long > p_short:
        return 2, p_long
    if p_short >= candidate_thr and p_short > p_long:
        return 0, p_short
    return -1, 0.0


def build_meta_row(
    proba: np.ndarray,
    direction: int,
    df_row: pd.Series,
    context_feats: list[str],
) -> dict:
    """Build one meta-model feature row at entry bar."""
    p_short, p_flat, p_long = float(proba[0]), float(proba[1]), float(proba[2])
    conf = max(p_short, p_long)
    row = {
        "p_short": p_short,
        "p_flat": p_flat,
        "p_long": p_long,
        "confidence": conf,
        "direction": 1 if direction == 2 else (-1 if direction == 0 else 0),
    }
    for feat in context_feats:
        row[feat] = float(df_row[feat]) if feat in df_row.index else 0.0
    return row


def profit_factor(trades: list[dict]) -> float:
    gross_win = sum(t.get("net_pnl", 0) for t in trades if t.get("net_pnl", 0) > 0)
    gross_loss = abs(sum(t.get("net_pnl", 0) for t in trades if t.get("net_pnl", 0) < 0))
    if gross_loss < 1e-9:
        return float("inf") if gross_win > 0 else 0.0
    return gross_win / gross_loss


def pass_fail_check(
    baseline: dict,
    variant: dict,
    max_trade_drop_pct: float = 30.0,
    min_pf_delta: float = 0.10,
) -> dict:
    """
    Evaluate variant vs baseline per meta_label_spec.json pass_fail rules.
    baseline/variant: {trades, pnl, pf, wr}
    """
    b_trades = max(baseline.get("trades", 0), 1)
    v_trades = variant.get("trades", 0)
    trade_drop = (b_trades - v_trades) / b_trades * 100.0

    b_pnl = baseline.get("pnl", 0.0)
    v_pnl = variant.get("pnl", 0.0)
    b_pf = baseline.get("pf", 0.0)
    v_pf = variant.get("pf", 0.0)

    pnl_pass = v_pnl >= b_pnl
    volume_ok = trade_drop <= max_trade_drop_pct
    pf_improves = (
        isinstance(v_pf, (int, float))
        and isinstance(b_pf, (int, float))
        and b_pf > 0
        and v_pf > b_pf * (1.0 + (min_pf_delta if not volume_ok else 0.0))
    )

    passed = pnl_pass or (volume_ok and pf_improves)
    reason = []
    if pnl_pass:
        reason.append(f"PnL ${v_pnl:+.0f} >= baseline ${b_pnl:+.0f}")
    if volume_ok:
        reason.append(f"trade drop {trade_drop:.1f}% <= {max_trade_drop_pct}%")
    else:
        reason.append(f"trade drop {trade_drop:.1f}% > {max_trade_drop_pct}%")
    if v_pf > b_pf:
        reason.append(f"PF {v_pf:.2f} > baseline {b_pf:.2f}")
    else:
        reason.append(f"PF {v_pf:.2f} <= baseline {b_pf:.2f}")

    return {
        "passed": passed,
        "trade_drop_pct": round(trade_drop, 2),
        "pnl_delta": round(v_pnl - b_pnl, 2),
        "pf_delta": round(v_pf - b_pf, 3),
        "reason": reason,
    }


# ── TB Widyawardhana v2 entry (flatboost_v2 + HMM T50_R55) ───────────────────
TRENDING_STATES = {0, 3}
THR_TRENDING_LONG = 0.50
THR_TRENDING_SHORT = 0.55
THR_RANGING_LONG = 0.55
THR_RANGING_SHORT = 0.60


def hmm_thresholds(hmm_state: int) -> tuple[float, float]:
    if int(hmm_state) in TRENDING_STATES:
        return THR_TRENDING_LONG, THR_TRENDING_SHORT
    return THR_RANGING_LONG, THR_RANGING_SHORT


def hmm_entry_from_proba(proba: np.ndarray, hmm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Production HMM-adaptive entry: per-bar sig in {0,1,2} + directional confidence."""
    n = len(proba)
    yp = np.ones(n, dtype=np.int32)
    conf = np.zeros(n, dtype=np.float64)
    for i in range(n):
        tl, ts = hmm_thresholds(hmm[i])
        p = proba[i]
        if p[2] >= tl and p[2] >= p[0]:
            yp[i] = 2
            conf[i] = float(p[2])
        elif p[0] >= ts and p[0] > p[2]:
            yp[i] = 0
            conf[i] = float(p[0])
    return yp, conf


def apply_lstm_soft_veto(
    yp: np.ndarray,
    conf: np.ndarray,
    proba_lstm: np.ndarray,
    agree_boost: float = 0.05,
    neutral_pen: float = 0.05,
    opposite_pen: float = 0.08,
    no_veto_thr: float = 0.50,
) -> tuple[np.ndarray, np.ndarray]:
    """Match inference_config tb_widyawardhana_v2: soft confidence adjust, then re-gate."""
    yp_out = yp.copy()
    conf_out = conf.copy()
    for i in range(len(yp_out)):
        if yp_out[i] == 1:
            continue
        sig = int(yp_out[i])
        adj = float(conf_out[i])
        li = int(np.argmax(proba_lstm[i]))
        if li == sig:
            adj += agree_boost
        elif li == 1:
            adj -= neutral_pen
        else:
            if adj > no_veto_thr:
                adj -= opposite_pen
            else:
                adj -= opposite_pen
        adj = float(np.clip(adj, 0.0, 1.0))
        thr = THR_TRENDING_LONG if sig == 2 else THR_TRENDING_SHORT
        if adj < thr:
            yp_out[i] = 1
            conf_out[i] = 0.0
        else:
            conf_out[i] = adj
    return yp_out, conf_out


def apply_meta_mask(
    yp: np.ndarray,
    conf: np.ndarray,
    meta_proba: Optional[np.ndarray],
    threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Zero-out bars where meta p_win < threshold (in-place safe copy)."""
    yp_out = yp.copy()
    conf_out = conf.copy()
    if meta_proba is None:
        return yp_out, conf_out
    for i in range(len(yp_out)):
        if yp_out[i] == 1:
            continue
        if meta_proba[i] < threshold:
            yp_out[i] = 1
            conf_out[i] = 0.0
    return yp_out, conf_out