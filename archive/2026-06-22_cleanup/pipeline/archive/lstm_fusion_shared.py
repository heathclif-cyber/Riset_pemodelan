"""Shared helpers for LSTM score fusion OOF eval (genuine protocol)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from config import ALL_COINS, TRAIN_CUTOFF_DATE, LABEL_DIR, MODEL_DIR, MODAL_PER_TRADE
from core.cascade_utils import (
    apply_conditional_momentum_fusion_pre,
    apply_lstm_boost_only_pre,
    apply_lstm_conf_fusion_post,
    apply_lstm_proba_fusion_pre,
)
from core.utils import ensure_utc_index

LGBM_RUN = "tb_lgbm_genuine_v2"
LSTM_RUN = "tb_lstm_genuine_v2"
GUARDIAN_RUN = "tb_guardian_genuine_v2_hmm_v2"
LGBM_DIR = MODEL_DIR / "runs" / LGBM_RUN
LSTM_DIR = MODEL_DIR / "runs" / LSTM_RUN
GUARDIAN_DIR = MODEL_DIR / "runs" / GUARDIAN_RUN
INFERENCE_CFG = MODEL_DIR / "inference_config.json"

HMM_CONFIG_B = {
    0: (0.55, 0.55),
    1: (0.55, 0.55),
    2: (0.50, 0.50),
    3: (0.45, 0.50),
    -1: (0.45, 0.45),
}

DYNAMIC_FEATS = [
    "bars_held_norm", "current_pnl_pct", "current_pnl_atr",
    "max_favorable_pnl_pct", "drawdown_from_peak_pct",
    "direction", "entry_price_ratio",
]

DYNSIZE_CFG = {
    "conf_window": 0.10,
    "conf_max_mult": 0.6,
    "regime_mult": {0: 0.75, 1: 1.0, 2: 1.0, "3_long": 1.5, "3_short": 0.75, -1: 0.80},
    "clamp_min": 0.5,
    "clamp_max": 2.0,
}

SHORT, FLAT, LONG = 0, 1, 2


def load_hmm_cfg() -> dict:
    """Frozen deploy HMM Config B from inference_config — NOT sweep winner (Config D)."""
    with open(INFERENCE_CFG, encoding="utf-8") as f:
        inf = json.load(f)
    pst = inf.get("hmm", {}).get("per_state_thresholds", {})
    if pst:
        return {int(k): (float(v[0]), float(v[1])) for k, v in pst.items()}
    return dict(HMM_CONFIG_B)


def load_guardian_params() -> dict:
    with open(INFERENCE_CFG, encoding="utf-8") as f:
        inf = json.load(f)
    g = inf.get("guardian", {})
    return {
        "exit_threshold": float(g.get("exit_threshold", 0.55)),
        "min_hold_bars": int(g.get("min_hold_bars", 2)),
    }


def apply_hmm_thr(p0, p2, hmm_enc, hmm_cfg):
    n = len(p0)
    default_tl, default_ts = hmm_cfg[-1]
    tl_arr = np.full(n, default_tl, dtype=np.float32)
    ts_arr = np.full(n, default_ts, dtype=np.float32)
    for state, (tl, ts) in hmm_cfg.items():
        if state == -1:
            continue
        mask = hmm_enc == state
        tl_arr[mask] = tl
        ts_arr[mask] = ts
    long_mask = p2 >= tl_arr
    short_mask = (p0 >= ts_arr) & ~long_mask
    y = np.ones(n, dtype=np.int32)
    y[long_mask] = LONG
    y[short_mask] = SHORT
    conf = np.where(long_mask, p2, np.where(short_mask, p0, 0.0)).astype(np.float32)
    return y, conf, tl_arr, ts_arr


def gate_mask(
    vol_spike: np.ndarray,
    lstm_valid: np.ndarray,
    gate_mode: str,
    is_gate: Optional[np.ndarray] = None,
) -> np.ndarray:
    if gate_mode == "all_oof":
        return lstm_valid
    if gate_mode == "vol_spike2":
        return lstm_valid & (vol_spike >= 2.0)
    if gate_mode == "pump_dump" and is_gate is not None:
        return lstm_valid & is_gate.astype(bool)
    return lstm_valid


def build_y_pred(coin: dict, cfg: dict, hmm_cfg: dict) -> np.ndarray:
    p0, p2 = coin["p0"].copy(), coin["p2"].copy()
    if cfg.get("fusion") == "baseline":
        y, _, _, _ = apply_hmm_thr(p0, p2, coin["hmm"], hmm_cfg)
        return y

    active = gate_mask(
        coin["vol_spike"], coin["lstm_valid"], cfg.get("gate", "all_oof"),
        coin.get("is_gate"),
    )
    if cfg["mode"] == "boost_only":
        p0, p2 = apply_lstm_boost_only_pre(
            p0, p2, coin["lstm_p"],
            agree_boost=cfg["agree_boost"],
            active_mask=active,
            boost_side=cfg.get("boost_side", "both"),
        )
        y, _, _, _ = apply_hmm_thr(p0, p2, coin["hmm"], hmm_cfg)
        return y
    if cfg["mode"] == "conditional_momentum":
        _, _, tl, ts = apply_hmm_thr(p0, p2, coin["hmm"], hmm_cfg)
        p0, p2 = apply_conditional_momentum_fusion_pre(
            p0, p2, coin["lstm_p"], tl, ts, coin["vol_spike"],
            vol_thr=cfg.get("vol_thr", 2.0),
            bull_thr=cfg.get("bull_thr", 0.40),
            bear_thr=cfg.get("bear_thr", 0.55),
            near_miss_gap=cfg.get("near_miss_gap", 0.05),
            boost=cfg.get("boost", 0.08),
            opposite_pen=cfg.get("opposite_pen", 0.12),
            enable_boost=cfg.get("enable_boost", True),
            enable_penalty=cfg.get("enable_penalty", True),
            lstm_valid=coin["lstm_valid"],
            proportional=cfg.get("proportional", True),
        )
        y, _, _, _ = apply_hmm_thr(p0, p2, coin["hmm"], hmm_cfg)
        return y
    if cfg["mode"] == "pre_hmm":
        p0, p2 = apply_lstm_proba_fusion_pre(
            p0, p2, coin["lstm_p"],
            agree_boost=cfg["agree_boost"],
            neutral_pen=cfg["neutral_pen"],
            opposite_pen=cfg["opposite_pen"],
            active_mask=active,
        )
        y, _, _, _ = apply_hmm_thr(p0, p2, coin["hmm"], hmm_cfg)
        return y

    y, conf, tl, ts = apply_hmm_thr(p0, p2, coin["hmm"], hmm_cfg)
    y, _ = apply_lstm_conf_fusion_post(
        y, conf, coin["lstm_p"], tl, ts,
        agree_boost=cfg["agree_boost"],
        neutral_pen=cfg["neutral_pen"],
        opposite_pen=cfg["opposite_pen"],
        active_mask=active,
    )
    return y


def preload_coins(lgbm_oof: pd.DataFrame, lstm_oof: pd.DataFrame,
                  g_static_feats: Optional[list] = None) -> list:
    coins = []
    for sym in ALL_COINS:
        path = LABEL_DIR / f"{sym}_features_v3.parquet"
        if not path.exists():
            continue
        df = pd.read_parquet(path)
        df = ensure_utc_index(df).sort_index()
        df = df[df.index < TRAIN_CUTOFF_DATE]
        if df.empty:
            continue

        sym_lgbm = lgbm_oof[(lgbm_oof["coin"] == sym) & (lgbm_oof["has_oof"] == True)][["p0", "p2"]]
        proba = sym_lgbm.reindex(df.index)
        has_oof = proba["p0"].notna()
        if has_oof.sum() < 30:
            continue

        lstm_cols = ["p0", "p1", "p2", "vol_spike", "has_oof"]
        if "is_gate" in lstm_oof.columns:
            lstm_cols.append("is_gate")
        sym_lstm = lstm_oof[lstm_oof["coin"] == sym][lstm_cols]
        lstm_aligned = sym_lstm.reindex(df.index[has_oof])
        lstm_p = lstm_aligned[["p0", "p1", "p2"]].values.astype(np.float32)
        lstm_valid = lstm_aligned["has_oof"].fillna(False).values.astype(bool)
        vol_spike = lstm_aligned["vol_spike"].fillna(-99).values.astype(np.float32)
        is_gate = (
            lstm_aligned["is_gate"].fillna(0).values.astype(bool)
            if "is_gate" in lstm_aligned.columns else (vol_spike >= 2.0)
        )

        df_oof = df[has_oof].copy()
        n = len(df_oof)
        if "log_ret_4" in df_oof.columns:
            ret4 = df_oof["log_ret_4"].fillna(0).values.astype(np.float32)
        else:
            ret4 = df_oof["close"].pct_change(4).fillna(0).values.astype(np.float32)

        entry = {
            "sym": sym,
            "ts": df_oof.index,
            "p0": proba["p0"][has_oof].values.astype(np.float32),
            "p2": proba["p2"][has_oof].values.astype(np.float32),
            "hmm": df_oof["hmm_regime_enc"].fillna(-1).values.astype(np.int8)
            if "hmm_regime_enc" in df_oof.columns else np.full(n, -1, np.int8),
            "lstm_p": lstm_p,
            "lstm_valid": lstm_valid,
            "vol_spike": vol_spike,
            "is_gate": is_gate,
            "ret4": ret4,
            "close": df_oof["close"].values.astype(np.float64),
            "high": df_oof["high"].values.astype(np.float64),
            "low": df_oof["low"].values.astype(np.float64),
            "atr": df_oof["atr_14_h1"].values.astype(np.float64),
            "h4_sh": df_oof["h4_swing_high"].values.astype(np.float64)
            if "h4_swing_high" in df_oof.columns else np.full(n, np.nan),
            "h4_sl": df_oof["h4_swing_low"].values.astype(np.float64)
            if "h4_swing_low" in df_oof.columns else np.full(n, np.nan),
        }
        if g_static_feats is not None:
            X_grd = np.zeros((n, len(g_static_feats)), dtype=np.float64)
            for idx, col in enumerate(g_static_feats):
                if col in df_oof.columns:
                    X_grd[:, idx] = df_oof[col].ffill().fillna(0).values.astype(np.float64)
            entry["X_grd"] = X_grd
        coins.append(entry)
    return coins


def precompute_cross_section(coins: list) -> dict:
    """Per-timestamp frac_up and max baseline p2 across OOF coins."""
    rows = []
    for c in coins:
        for i, ts in enumerate(c["ts"]):
            rows.append({
                "ts": ts,
                "coin": c["sym"],
                "ret4": float(c["ret4"][i]),
                "p2": float(c["p2"][i]),
            })
    panel = pd.DataFrame(rows)
    pivot_ret = panel.pivot_table(index="ts", columns="coin", values="ret4", aggfunc="first")
    pivot_p2 = panel.pivot_table(index="ts", columns="coin", values="p2", aggfunc="first")
    frac_up = (pivot_ret > 0).mean(axis=1)
    max_p2 = pivot_p2.max(axis=1)
    return {"frac_up": frac_up, "max_p2": max_p2}


def attach_cross_section(coins: list, cs: dict) -> None:
    frac_s = cs["frac_up"]
    max_p2_s = cs["max_p2"]
    for c in coins:
        idx = c["ts"]
        c["frac_up"] = frac_s.reindex(idx).fillna(0.5).values.astype(np.float32)
        c["cs_max_p2"] = max_p2_s.reindex(idx).fillna(1.0).values.astype(np.float32)


def build_breadth_mask(coin: dict, cfg: dict) -> Optional[np.ndarray]:
    n = len(coin["p0"])
    mask = np.ones(n, dtype=bool)
    if cfg.get("breadth_frac") is not None:
        mask &= coin["frac_up"] >= float(cfg["breadth_frac"])
    if cfg.get("max_p2_cap") is not None:
        mask &= coin["cs_max_p2"] < float(cfg["max_p2_cap"])
    if not mask.any():
        return mask
    return mask


def apply_fused_scores(coin: dict, cfg: dict, hmm_cfg: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return y_pred, fused p0, fused p2 after conditional_momentum (if enabled)."""
    p0, p2 = coin["p0"].copy(), coin["p2"].copy()
    if cfg.get("fusion") == "baseline":
        y, _, _, _ = apply_hmm_thr(p0, p2, coin["hmm"], hmm_cfg)
        return y, p0, p2
    if cfg.get("mode") == "conditional_momentum":
        _, _, tl, ts = apply_hmm_thr(p0, p2, coin["hmm"], hmm_cfg)
        breadth = build_breadth_mask(coin, cfg) if cfg.get("breadth_frac") is not None else None
        if cfg.get("max_p2_cap") is not None and breadth is None:
            breadth = build_breadth_mask(coin, cfg)
        p0, p2 = apply_conditional_momentum_fusion_pre(
            p0, p2, coin["lstm_p"], tl, ts, coin["vol_spike"],
            vol_thr=cfg.get("vol_thr", 2.0),
            bull_thr=cfg.get("bull_thr", 0.40),
            bear_thr=cfg.get("bear_thr", 0.55),
            near_miss_gap=cfg.get("near_miss_gap", 0.05),
            boost=cfg.get("boost", 0.08),
            opposite_pen=cfg.get("opposite_pen", 0.12),
            enable_boost=cfg.get("enable_boost", True),
            enable_penalty=cfg.get("enable_penalty", True),
            lstm_valid=coin["lstm_valid"],
            proportional=cfg.get("proportional", True),
            breadth_mask=breadth,
            breadth_gate_boost_side=cfg.get("breadth_gate_side", "long"),
        )
        y, _, _, _ = apply_hmm_thr(p0, p2, coin["hmm"], hmm_cfg)
        return y, p0, p2
    y = build_y_pred(coin, cfg, hmm_cfg)
    return y, p0, p2


def rally_boost_label(cfg: dict) -> str:
    if not cfg.get("rally_boost_enabled"):
        return config_label(cfg) if cfg.get("mode") == "conditional_momentum" else cfg.get("label", "baseline")
    parts = [
        config_label(cfg) if cfg.get("mode") == "conditional_momentum" else "base",
        f"rbr{int(cfg.get('rally_frac', 0.75) * 100)}",
        f"rk{cfg.get('rally_top_k', 5)}",
        f"ra{int(cfg.get('rally_boost_amt', 0.08) * 100)}",
    ]
    if cfg.get("rally_require_lstm", True):
        parts.append("lst")
    if cfg.get("rally_min_gap") is not None:
        parts.append(f"mg{int(cfg['rally_min_gap'] * 100)}")
    if cfg.get("rally_prop_rank"):
        parts.append("pr")
    return "_".join(parts)


def apply_rally_boost_panel(
    coins: list,
    scores_by_sym: dict[str, dict[str, np.ndarray]],
    hmm_cfg: dict,
    cfg: dict,
) -> tuple[dict[str, dict[str, np.ndarray]], int]:
    """
    Additive cross-coin rally boost (pump specialist).

    On bars where frac_up >= rally_frac:
      1. Find coins where RAW LGBM is FLAT (before rally layer)
      2. Rank by raw p2 (LONG conviction)
      3. Add rally_boost_amt to fused p2 for top-K only

    Runs AFTER conditional_momentum fusion. Does not remove existing boosts.
    """
    if not cfg.get("rally_boost_enabled"):
        return scores_by_sym, 0

    from collections import defaultdict

    from core.cascade_utils import _lstm_dominant_direction

    rally_frac = float(cfg.get("rally_frac", 0.75))
    top_k = int(cfg.get("rally_top_k", 5))
    boost_amt = float(cfg.get("rally_boost_amt", 0.08))
    vol_thr = float(cfg.get("vol_thr", 2.0))
    require_lstm = bool(cfg.get("rally_require_lstm", True))
    bull_thr = float(cfg.get("bull_thr", 0.38))
    min_gap = cfg.get("rally_min_gap")  # only near-miss FLAT: p2 >= tl - gap
    prop_rank = bool(cfg.get("rally_prop_rank", False))

    out = {
        sym: {"p0": sc["p0"].copy(), "p2": sc["p2"].copy()}
        for sym, sc in scores_by_sym.items()
    }
    buckets: dict = defaultdict(list)
    n_fires = 0

    for c in coins:
        sym = c["sym"]
        p0_raw = c["p0"]
        p2_raw = c["p2"]
        _, _, tl, ts = apply_hmm_thr(p0_raw, p2_raw, c["hmm"], hmm_cfg)

        for i, ts_bar in enumerate(c["ts"]):
            if float(c["frac_up"][i]) < rally_frac:
                continue
            if float(c["vol_spike"][i]) < vol_thr:
                continue
            if p2_raw[i] >= tl[i] or p0_raw[i] >= ts[i]:
                continue
            if min_gap is not None and p2_raw[i] < tl[i] - float(min_gap):
                continue
            if require_lstm:
                if not c["lstm_valid"][i]:
                    continue
                lstm_dir, lstm_dom = _lstm_dominant_direction(c["lstm_p"][i : i + 1])
                if int(lstm_dir[0]) != LONG or float(lstm_dom[0]) < bull_thr:
                    continue
            buckets[ts_bar].append((sym, i, float(p2_raw[i])))

    for _ts, entries in buckets.items():
        if not entries:
            continue
        entries.sort(key=lambda x: x[2], reverse=True)
        for rank, (sym, idx, _) in enumerate(entries[:top_k], start=1):
            amt = boost_amt
            if prop_rank and top_k > 1:
                amt = boost_amt * (top_k - rank + 1) / top_k
            out[sym]["p2"][idx] = min(1.0, float(out[sym]["p2"][idx]) + amt)
            n_fires += 1

    return out, n_fires


def build_predictions_full(
    coins: list,
    cfg: dict,
    hmm_cfg: dict,
) -> tuple[dict[str, np.ndarray], dict[str, dict[str, np.ndarray]], int]:
    """Layer 1: per-coin fusion. Layer 2: optional rally boost panel. Layer 3: HMM gate."""
    scores_by_sym = {}
    for c in coins:
        _, p0, p2 = apply_fused_scores(c, cfg, hmm_cfg)
        scores_by_sym[c["sym"]] = {"p0": p0, "p2": p2}

    rally_fires = 0
    if cfg.get("rally_boost_enabled"):
        scores_by_sym, rally_fires = apply_rally_boost_panel(coins, scores_by_sym, hmm_cfg, cfg)

    y_map = {}
    for c in coins:
        sym = c["sym"]
        p0, p2 = scores_by_sym[sym]["p0"], scores_by_sym[sym]["p2"]
        y, _, _, _ = apply_hmm_thr(p0, p2, c["hmm"], hmm_cfg)
        y_map[sym] = y

    return y_map, scores_by_sym, rally_fires


def apply_top_k_per_bar(
    coins: list,
    y_by_sym: dict[str, np.ndarray],
    scores_by_sym: dict[str, dict[str, np.ndarray]],
    k_long: int = 0,
    k_short: int = 0,
) -> dict[str, np.ndarray]:
    if k_long <= 0 and k_short <= 0:
        return y_by_sym
    from collections import defaultdict

    buckets: dict[tuple, list] = defaultdict(list)
    for c in coins:
        sym = c["sym"]
        y = y_by_sym[sym]
        sc = scores_by_sym[sym]
        for i, ts in enumerate(c["ts"]):
            if k_long > 0 and y[i] == LONG:
                buckets[(ts, LONG)].append((sym, i, float(sc["p2"][i])))
            elif k_short > 0 and y[i] == SHORT:
                buckets[(ts, SHORT)].append((sym, i, float(sc["p0"][i])))

    y_out = {sym: arr.copy() for sym, arr in y_by_sym.items()}
    for (ts, direction), entries in buckets.items():
        k = k_long if direction == LONG else k_short
        if len(entries) <= k:
            continue
        entries.sort(key=lambda x: x[2], reverse=True)
        keep = {(sym, idx) for sym, idx, _ in entries[:k]}
        for sym, idx, _ in entries:
            if (sym, idx) not in keep:
                y_out[sym][idx] = FLAT
    return y_out


def pump_config_label(cfg: dict) -> str:
    if cfg.get("fusion") == "baseline":
        return "baseline_no_lstm"
    parts = [config_label(cfg)]
    if cfg.get("breadth_frac") is not None:
        parts.append(f"br{int(cfg['breadth_frac'] * 100)}")
    if cfg.get("max_p2_cap") is not None:
        parts.append(f"mp{int(cfg['max_p2_cap'] * 100)}")
    if cfg.get("top_k_long"):
        parts.append(f"kl{cfg['top_k_long']}")
    if cfg.get("top_k_short"):
        parts.append(f"ks{cfg['top_k_short']}")
    return "_".join(parts)


def config_label(cfg: dict) -> str:
    if cfg.get("fusion") == "baseline":
        return "baseline_no_lstm"
    if cfg.get("mode") == "boost_only":
        side = cfg.get("boost_side", "both")
        return f"boost_{side}_{cfg['gate']}_b{int(cfg['agree_boost'] * 100)}"
    if cfg.get("mode") == "conditional_momentum":
        bp = "BP" if cfg.get("enable_boost") and cfg.get("enable_penalty") else (
            "B" if cfg.get("enable_boost") else "P"
        )
        return (
            f"cond_{bp}_bu{int(cfg.get('bull_thr', 0.4)*100)}"
            f"_be{int(cfg.get('bear_thr', 0.55)*100)}"
            f"_g{int(cfg.get('near_miss_gap', 0.05)*100)}"
            f"_b{int(cfg.get('boost', 0.08)*100)}"
            f"_o{int(cfg.get('opposite_pen', 0.12)*100)}"
        )
    return (
        f"{cfg['mode']}_{cfg['gate']}"
        f"_b{int(cfg['agree_boost'] * 100)}"
        f"_n{int(cfg['neutral_pen'] * 100)}"
        f"_o{int(cfg['opposite_pen'] * 100)}"
    )


def summarize_trades(trades: list, base_modal: float = MODAL_PER_TRADE) -> dict:
    if not trades:
        return {"n": 0, "wr": 0, "pnl": 0, "ppt": 0, "pf": 0,
                "sl_pct": 0, "avg_modal": base_modal, "ppt_norm": 0}
    n = len(trades)
    wins = sum(1 for t in trades if t["net_pnl"] > 0)
    sl_hit = sum(1 for t in trades if t["outcome"] == "LOSS")
    gpnl = sum(t["net_pnl"] for t in trades if t["net_pnl"] > 0)
    lloss = sum(abs(t["net_pnl"]) for t in trades if t["net_pnl"] < 0)
    tpnl = sum(t["net_pnl"] for t in trades)
    pf = gpnl / lloss if lloss > 0 else float("inf")
    modals = [t.get("modal_used", base_modal) for t in trades]
    avg_modal = float(np.mean(modals))
    ppt_norm = (tpnl / n) * (base_modal / avg_modal) if avg_modal > 0 else 0.0
    return {
        "n": n, "wr": wins / n * 100, "pnl": tpnl,
        "ppt": tpnl / n, "pf": pf, "sl_pct": sl_hit / n * 100,
        "avg_modal": avg_modal, "ppt_norm": ppt_norm,
    }


def compute_dynamic_modal(p0, p2, hmm_enc, y_pred, base_modal, ds_cfg, tl_arr, ts_arr):
    n = len(p0)
    rm = ds_cfg["regime_mult"]
    long_mask = y_pred == LONG
    short_mask = y_pred == SHORT
    conf = np.where(long_mask, p2, np.where(short_mask, p0, 0.0)).astype(np.float32)
    thr = np.where(long_mask, tl_arr, ts_arr)
    c_mult = 1.0 + np.clip((conf - thr) / ds_cfg["conf_window"], 0.0, ds_cfg["conf_max_mult"])
    r_mult = np.full(n, rm[-1], dtype=np.float64)
    r_mult[hmm_enc == 0] = rm[0]
    r_mult[hmm_enc == 1] = rm[1]
    r_mult[hmm_enc == 2] = rm[2]
    r_mult[(hmm_enc == 3) & long_mask] = rm.get("3_long", 1.5)
    r_mult[(hmm_enc == 3) & short_mask] = rm.get("3_short", 0.75)
    total_mult = np.clip(r_mult * c_mult, ds_cfg["clamp_min"], ds_cfg["clamp_max"])
    modal_arr = (base_modal * total_mult).astype(np.float32)
    modal_arr[y_pred == FLAT] = base_modal
    return modal_arr


def genuine_audit_block() -> dict:
    return {
        "protocol": "genuine_oof",
        "holdout_used": False,
        "train_cutoff": str(TRAIN_CUTOFF_DATE.date()),
        "lgbm_source": f"{LGBM_RUN}/oof_predictions.parquet (has_oof=True only)",
        "lstm_source": f"{LSTM_RUN}/oof_lstm_predictions.parquet (has_oof=True only)",
        "hmm_config": "Config B frozen from inference_config.json per_state_thresholds",
        "guardian_source": GUARDIAN_RUN,
        "guardian_trained_on": "OOF trades from LGBM+HMM-B (not holdout)",
        "no_lgbm_retrain": True,
        "no_holdout_tuning": True,
    }