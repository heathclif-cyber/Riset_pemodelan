"""Shared helpers for ic32 LGBM+LSTM fusion sweep (genuine OOF protocol)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from config import ALL_COINS, TRAIN_CUTOFF_DATE, LABEL_DIR, MODEL_DIR, MODAL_PER_TRADE, LABEL_MAP
from core.cascade_utils import (
    apply_conditional_momentum_fusion_pre,
    SHORT,
    FLAT,
    LONG,
)

IC32_RUN = "ic32_regime_v1"
COMPLEMENT_RUN = "ic32_lstm_swing_complement_v2"
IC32_DIR = MODEL_DIR / "runs" / IC32_RUN
COMPLEMENT_DIR = MODEL_DIR / "runs" / COMPLEMENT_RUN
FROZEN_PATH = IC32_DIR / "b_dir_combined_frozen.json"
INF_CFG_PATH = MODEL_DIR / "inference_config.json"

RANGING = {1, 2}
TRENDING = {0, 3}
LGBM_FLAT_REVIEW_THR = 0.90
LSTM_OVERRIDE_THR = 0.70


def load_b_dir_hmm_cfg() -> dict:
    with open(FROZEN_PATH, encoding="utf-8") as f:
        data = json.load(f)
    raw = data["per_state_thresholds"]
    return {int(k): (float(v[0]), float(v[1])) for k, v in raw.items()}


def load_production_defaults() -> dict:
    with open(INF_CFG_PATH, encoding="utf-8") as f:
        cfg = json.load(f)
    cascade = cfg.get("cascade", {})
    ra = cfg.get("regime_alignment", {})
    return {
        "agree_boost": float(cascade.get("lstm_adjust_agree_boost", 0.05)),
        "neutral_pen": float(cascade.get("lstm_adjust_neutral_pen", 0.0)),
        "opposite_pen": float(cascade.get("lstm_adjust_opposite_pen", 0.65)),
        "conf_entry": float(cascade.get("confidence_threshold_entry", 0.59)),
        "flat_review": bool(cascade.get("lstm_flat_review_enabled", True)),
        "dir_review_thr": float(cascade.get("lstm_directional_review_threshold", 0.35)),
        "hmm_gate_lstm": True,
        "flip": bool(ra.get("enabled", True)),
        "flip_ranging_with": 0.10,
        "flip_ranging_counter": 0.05,
        "flip_trending_with": 0.10,
        "flip_trending_counter": 0.05,
    }


def build_per_bar_thresholds(hmm_enc: np.ndarray, hmm_cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    n = len(hmm_enc)
    default_tl, default_ts = hmm_cfg[-1]
    tl_arr = np.full(n, default_tl, dtype=np.float32)
    ts_arr = np.full(n, default_ts, dtype=np.float32)
    for state, (tl, ts) in hmm_cfg.items():
        if state == -1:
            continue
        mask = hmm_enc == state
        tl_arr[mask] = tl
        ts_arr[mask] = ts
    return tl_arr, ts_arr


def _apply_flip(
    y: np.ndarray,
    conf: np.ndarray,
    hmm: np.ndarray,
    h4t: np.ndarray,
    cfg: dict,
) -> np.ndarray:
    if not cfg.get("flip", True):
        return conf
    adj = np.zeros(len(y), dtype=np.float32)
    dir_mask = y != FLAT
    if not dir_mask.any():
        return conf

    is_long = y == LONG
    is_short = y == SHORT
    ranging = np.isin(hmm, list(RANGING))
    trending = np.isin(hmm, list(TRENDING))
    h4_up = h4t > 0
    h4_dn = h4t < 0

    wt_r = dir_mask & ranging & ((is_long & h4_up) | (is_short & h4_dn))
    ct_r = dir_mask & ranging & ((is_long & h4_dn) | (is_short & h4_up))
    wt_t = dir_mask & trending & ((is_long & h4_up) | (is_short & h4_dn))
    ct_t = dir_mask & trending & ((is_long & h4_dn) | (is_short & h4_up))

    adj[wt_r] -= cfg.get("flip_ranging_with", 0.10)
    adj[ct_r] += cfg.get("flip_ranging_counter", 0.05)
    adj[wt_t] += cfg.get("flip_trending_with", 0.10)
    adj[ct_t] -= cfg.get("flip_trending_counter", 0.05)
    return np.clip(conf + adj, 0.0, 1.0).astype(np.float32)


def _apply_dual_complement(
    p0: np.ndarray,
    p2: np.ndarray,
    coin: dict,
    cfg: dict,
    tl_arr: np.ndarray,
    ts_arr: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if not cfg.get("dual_complement") or coin.get("lstm_p_comp") is None:
        return p0, p2
    flat_mask = (p2 < tl_arr) & (p0 < ts_arr)
    p0c, p2c = apply_conditional_momentum_fusion_pre(
        p0.copy(),
        p2.copy(),
        coin["lstm_p_comp"],
        tl_arr,
        ts_arr,
        coin["vol_spike"],
        vol_thr=cfg.get("vol_thr", 2.0),
        bull_thr=cfg.get("bull_thr", 0.38),
        bear_thr=cfg.get("bear_thr", 0.50),
        near_miss_gap=cfg.get("near_miss_gap", 0.05),
        boost=cfg.get("boost", 0.10),
        opposite_pen=cfg.get("comp_opposite_pen", 0.14),
        enable_boost=cfg.get("enable_boost", True),
        enable_penalty=cfg.get("enable_penalty", True),
        lstm_valid=coin.get("lstm_valid_comp", coin.get("lstm_valid")),
        proportional=True,
    )
    comp_valid = coin.get("lstm_valid_comp", coin.get("lstm_valid"))
    use = flat_mask & comp_valid if comp_valid is not None else flat_mask
    p0_out, p2_out = p0.copy(), p2.copy()
    p0_out[use] = p0c[use]
    p2_out[use] = p2c[use]
    return p0_out, p2_out


def apply_dual_complement_to_proba(
    oof_proba: np.ndarray,
    lstm_comp: np.ndarray,
    vol_spike: np.ndarray,
    lstm_comp_valid: np.ndarray,
    hmm_enc: np.ndarray,
    hmm_cfg: dict,
    cfg: dict,
) -> np.ndarray:
    """Pre-adjust LGBM OOF proba on FLAT+vol_spike bars before hierarchical_predict."""
    if not cfg.get("dual_complement"):
        return oof_proba
    p0 = oof_proba[:, 0].astype(np.float32).copy()
    p2 = oof_proba[:, 2].astype(np.float32).copy()
    tl_arr, ts_arr = build_per_bar_thresholds(hmm_enc, hmm_cfg)
    stub = {
        "lstm_p_comp": lstm_comp.astype(np.float32),
        "lstm_valid_comp": lstm_comp_valid.astype(bool),
        "vol_spike": vol_spike.astype(np.float32),
    }
    p0_out, p2_out = _apply_dual_complement(p0, p2, stub, cfg, tl_arr, ts_arr)
    out = np.array(oof_proba, dtype=np.float64, copy=True)
    out[:, 0] = p0_out
    out[:, 2] = p2_out
    return out


def build_signals(coin: dict, cfg: dict, hmm_cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized hard_consensus + FLIP + flat review (approx production path)."""
    p0 = coin["p0"].astype(np.float32).copy()
    p2 = coin["p2"].astype(np.float32).copy()
    n = len(p0)
    hmm = coin["hmm"]
    h4t = coin.get("h4t", np.zeros(n, dtype=np.float64))
    lstm_p = coin["lstm_p"]
    lstm_valid = coin["lstm_valid"]

    tl_arr, ts_arr = build_per_bar_thresholds(hmm, hmm_cfg)
    p0, p2 = _apply_dual_complement(p0, p2, coin, cfg, tl_arr, ts_arr)

    lgbm_long = p2 >= tl_arr
    lgbm_short = (p0 >= ts_arr) & ~lgbm_long
    dir_mask = lgbm_long | lgbm_short

    y = np.full(n, FLAT, dtype=np.int32)
    conf = np.zeros(n, dtype=np.float32)
    lgbm_dir = np.where(lgbm_long, LONG, np.where(lgbm_short, SHORT, FLAT))
    lgbm_conf = np.where(lgbm_long, p2, np.where(lgbm_short, p0, 0.0)).astype(np.float32)

    if cfg.get("fusion") == "baseline":
        y[dir_mask] = lgbm_dir[dir_mask]
        conf[dir_mask] = lgbm_conf[dir_mask]
        conf = _apply_flip(y, conf, hmm, h4t, cfg)
        below = (y != FLAT) & (conf < cfg.get("conf_entry", 0.59))
        y[below] = FLAT
        return y, conf

    lstm_active = np.ones(n, dtype=bool)
    if cfg.get("hmm_gate_lstm", True):
        lstm_active = np.isin(hmm, list(TRENDING))

    adj_conf = lgbm_conf.copy()
    if lstm_valid.any():
        lstm_dir = np.argmax(lstm_p, axis=1).astype(np.int32)
        active = dir_mask & lstm_active & lstm_valid
        agree = active & (lstm_dir == lgbm_dir)
        neutral = active & (lstm_dir == FLAT)
        opposite = active & ~agree & ~neutral
        adj_conf[agree] += cfg.get("agree_boost", 0.05)
        adj_conf[neutral] -= cfg.get("neutral_pen", 0.0)
        adj_conf[opposite] -= cfg.get("opposite_pen", 0.65)
        adj_conf = np.clip(adj_conf, 0.0, 1.0)

    y[dir_mask] = lgbm_dir[dir_mask]
    conf[dir_mask] = adj_conf[dir_mask]
    conf = _apply_flip(y, conf, hmm, h4t, cfg)

    if cfg.get("flat_review", True) and lstm_valid.any():
        flat_mask = ~dir_mask
        lgbm_max = np.maximum(p0, p2)
        lgbm_dir_score = np.maximum(p0, p2)
        review = flat_mask & lstm_valid & (
            (lgbm_max < LGBM_FLAT_REVIEW_THR)
            | (lgbm_dir_score > cfg.get("dir_review_thr", 0.35))
        )
        if cfg.get("hmm_gate_lstm", True):
            review &= lstm_active
        idxs = np.where(review)[0]
        for i in idxs:
            lstm_dir_i = int(np.argmax(lstm_p[i]))
            if lstm_dir_i == LONG and lstm_p[i, 2] >= LSTM_OVERRIDE_THR:
                oc = float(lstm_p[i, 2])
                y[i] = LONG
                conf[i] = oc
            elif lstm_dir_i == SHORT and lstm_p[i, 0] >= LSTM_OVERRIDE_THR:
                oc = float(lstm_p[i, 0])
                y[i] = SHORT
                conf[i] = oc
        conf = _apply_flip(y, conf, hmm, h4t, cfg)

    below = (y != FLAT) & (conf < cfg.get("conf_entry", 0.59))
    y[below] = FLAT
    return y, conf


def preload_ic32_coins(
    lgbm_oof: pd.DataFrame,
    lstm_oof: pd.DataFrame,
    g_static: Optional[list] = None,
    complement_oof: Optional[pd.DataFrame] = None,
) -> list:
    coins = []
    for sym in ALL_COINS:
        path = LABEL_DIR / f"{sym}_features_v3.parquet"
        if not path.exists():
            continue
        df = pd.read_parquet(path)
        df = df.sort_index()
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        df = df[df.index < TRAIN_CUTOFF_DATE]
        df = df[df["label"].astype(str).isin(LABEL_MAP)]
        if df.empty:
            continue

        rp = LABEL_DIR / f"{sym}_regime_h1.parquet"
        if rp.exists():
            try:
                reg = pd.read_parquet(rp)
                if "hmm_regime_enc" in df.columns:
                    df = df.drop(columns=["hmm_regime_enc"])
                df = df.join(reg[["hmm_regime_enc"]], how="left")
                df["hmm_regime_enc"] = df["hmm_regime_enc"].fillna(1).astype("int32")
            except Exception:
                df["hmm_regime_enc"] = 1
        elif "hmm_regime_enc" not in df.columns:
            df["hmm_regime_enc"] = 1

        sym_lgbm = lgbm_oof[(lgbm_oof["coin"] == sym) & (lgbm_oof["has_oof"] == True)][["p0", "p2"]]
        proba = sym_lgbm.reindex(df.index)
        has_oof = proba["p0"].notna()
        if has_oof.sum() < 30:
            continue

        lstm_cols = ["p0", "p1", "p2", "has_oof"]
        if "vol_spike" in lstm_oof.columns:
            lstm_cols.append("vol_spike")
        sym_lstm = lstm_oof[lstm_oof["coin"] == sym][lstm_cols].copy()
        if "ts" in sym_lstm.columns:
            sym_lstm = sym_lstm.set_index("ts")
        lstm_aligned = sym_lstm.reindex(df.index[has_oof])
        lstm_p = lstm_aligned[["p0", "p1", "p2"]].values.astype(np.float32)
        lstm_valid = lstm_aligned["has_oof"].fillna(False).values.astype(bool)

        df_oof = df[has_oof].copy()
        n = len(df_oof)
        if "vol_spike_zscore" in df_oof.columns:
            vol_spike = df_oof["vol_spike_zscore"].fillna(-99).values.astype(np.float32)
        elif "vol_spike" in lstm_aligned.columns:
            vol_spike = lstm_aligned["vol_spike"].fillna(-99).values.astype(np.float32)
        else:
            vol_spike = np.full(n, -99.0, dtype=np.float32)

        entry = {
            "sym": sym,
            "ts": df_oof.index,
            "p0": proba["p0"][has_oof].values.astype(np.float32),
            "p2": proba["p2"][has_oof].values.astype(np.float32),
            "hmm": df_oof["hmm_regime_enc"].fillna(-1).values.astype(np.int8),
            "h4t": df_oof["h4_trend"].fillna(0).values.astype(np.float64)
            if "h4_trend" in df_oof.columns else np.zeros(n, dtype=np.float64),
            "lstm_p": lstm_p,
            "lstm_valid": lstm_valid,
            "vol_spike": vol_spike,
            "close": df_oof["close"].values.astype(np.float64),
            "high": df_oof["high"].values.astype(np.float64),
            "low": df_oof["low"].values.astype(np.float64),
            "atr": df_oof["atr_14_h1"].values.astype(np.float64),
            "h4_sh": df_oof["h4_swing_high"].values.astype(np.float64)
            if "h4_swing_high" in df_oof.columns else np.full(n, np.nan),
            "h4_sl": df_oof["h4_swing_low"].values.astype(np.float64)
            if "h4_swing_low" in df_oof.columns else np.full(n, np.nan),
            "h4t_full": df_oof["h4_trend"].values.astype(np.float64)
            if "h4_trend" in df_oof.columns else None,
            "volr": df_oof["vol_ratio_20"].values.astype(np.float64)
            if "vol_ratio_20" in df_oof.columns else None,
        }

        if complement_oof is not None:
            sym_comp = complement_oof[complement_oof["coin"] == sym][["p0", "p1", "p2", "has_oof"]].copy()
            if "ts" in sym_comp.columns:
                sym_comp = sym_comp.set_index("ts")
            comp_aligned = sym_comp.reindex(df.index[has_oof])
            entry["lstm_p_comp"] = comp_aligned[["p0", "p1", "p2"]].values.astype(np.float32)
            entry["lstm_valid_comp"] = comp_aligned["has_oof"].fillna(False).values.astype(bool)

        if g_static is not None:
            X_grd = np.zeros((n, len(g_static)), dtype=np.float64)
            for idx, col in enumerate(g_static):
                if col in df_oof.columns:
                    X_grd[:, idx] = df_oof[col].ffill().fillna(0).values.astype(np.float64)
            entry["X_grd"] = X_grd
            entry["df_oof"] = df_oof
        coins.append(entry)
    return coins


def config_label(cfg: dict) -> str:
    if cfg.get("fusion") == "baseline":
        return "baseline_production"
    parts = [
        f"hc_a{int(cfg.get('agree_boost', 0.05) * 100)}",
        f"o{int(cfg.get('opposite_pen', 0.65) * 100)}",
    ]
    if not cfg.get("flat_review", True):
        parts.append("nofr")
    if not cfg.get("hmm_gate_lstm", True):
        parts.append("nogate")
    if not cfg.get("flip", True):
        parts.append("noflip")
    if cfg.get("dual_complement"):
        parts.append("dualcomp")
    return "_".join(parts)


def count_signals(coins: list, y_by_sym: dict, y_base: dict) -> dict:
    n_long = n_short = delta_long = delta_short = 0
    for c in coins:
        sym = c["sym"]
        y = y_by_sym[sym]
        yb = y_base[sym]
        n_long += int((y == LONG).sum())
        n_short += int((y == SHORT).sum())
        delta_long += int(((y == LONG) & (yb != LONG)).sum())
        delta_short += int(((y == SHORT) & (yb != SHORT)).sum())
    n_dir = n_long + n_short
    return {
        "n_long": n_long,
        "n_short": n_short,
        "n_dir": n_dir,
        "delta_long": delta_long,
        "delta_short": delta_short,
        "delta_dir": delta_long + delta_short,
    }


def summarize_trades(trades: list, base_modal: float = MODAL_PER_TRADE) -> dict:
    if not trades:
        return {"n": 0, "wr": 0, "pnl": 0, "ppt": 0, "pf": 0, "sl_pct": 0}
    n = len(trades)
    wins = sum(1 for t in trades if t["net_pnl"] > 0)
    sl_hit = sum(1 for t in trades if t.get("outcome") == "LOSS")
    gpnl = sum(t["net_pnl"] for t in trades if t["net_pnl"] > 0)
    lloss = sum(abs(t["net_pnl"]) for t in trades if t["net_pnl"] < 0)
    tpnl = sum(t["net_pnl"] for t in trades)
    pf = gpnl / lloss if lloss > 0 else float("inf")
    return {
        "n": n,
        "wr": wins / n * 100,
        "pnl": tpnl,
        "ppt": tpnl / n,
        "pf": pf,
        "sl_pct": sl_hit / n * 100,
    }


def genuine_audit_block() -> dict:
    return {
        "protocol": "genuine_oof",
        "holdout_used": False,
        "train_cutoff": str(TRAIN_CUTOFF_DATE.date()),
        "lgbm_source": f"{IC32_RUN}/oof_predictions.parquet (has_oof=True only)",
        "lstm_source": f"{IC32_RUN}/oof_lstm_baseline_predictions.parquet",
        "hmm_config": "B-dir-combined frozen (b_dir_combined_frozen.json)",
        "guardian_source": "ic32_guardian_continuation_v1",
        "no_lgbm_retrain": True,
        "no_holdout_tuning": True,
    }