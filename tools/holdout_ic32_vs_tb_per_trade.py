"""
Holdout per-trade comparison: ic32_regime_v1 vs tb_genuine_v2_dynsize_lstm_cond.

Frozen configs only — export + analysis, NOT tuning.
Outputs:
  reports/experiments/holdout_ic32_trades_apr_jun26.csv
  reports/experiments/holdout_tb_lstm_cond_trades_apr_jun26.csv
  reports/experiments/holdout_ic32_vs_tb_per_trade_report.md
"""
from __future__ import annotations

import json
import sys
import warnings
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pipeline.backtest_utils as btu
from config import (
    ALL_COINS, HOLDOUT_DIR, MODEL_DIR, REPORT_DIR, LABEL_MAP,
    MODAL_PER_TRADE, LEVERAGE_SIM, FEE_PER_SIDE, SLIPPAGE_PER_SIDE,
    MAX_HOLDING_BARS, CONFIDENCE_THRESHOLD_ENTRY,
    SWING_LABEL_MIN_RR, SWING_LABEL_MIN_TP, SWING_LABEL_MAX_SL,
    TP_SL_FALLBACK_TP, TP_SL_FALLBACK_SL,
    GUARDIAN_EXIT_THRESHOLD, GUARDIAN_DYNAMIC_FEATURES,
    TRAILING_STOP_ENABLED, TRAILING_STOP_ATR, TRAILING_STOP_MIN_BARS,
    OOS_START, OOS_END, LSTM_SEQ_LEN, GUARDIAN_ACTIVATION_ATR,
)
from core.cascade_utils import apply_conditional_momentum_fusion_pre
from core.evaluator import full_trading_report, simulate_trades_swing
from core.models import load_lstm
from core.utils import ensure_utc_index, setup_logger
from pipeline.backtest_utils import hierarchical_predict, compute_guardian_static_array

GUARDIAN_COMPUTED = set(GUARDIAN_DYNAMIC_FEATURES) | {
    "cvd_slope_h4_delta_entry", "ofi_h4_delta_entry", "flow_momentum_3bar",
}
from pipeline.lstm_fusion_shared import (
    DYNAMIC_FEATS, DYNSIZE_CFG, FLAT, LONG, SHORT,
    apply_hmm_thr, build_y_pred, compute_dynamic_modal,
    load_guardian_params, load_hmm_cfg, summarize_trades,
)

logger = setup_logger("holdout_ic32_vs_tb")

OUT_DIR = REPORT_DIR / "experiments"
IC32_RUN = MODEL_DIR / "runs" / "ic32_regime_v1"
LSTM_RUN = MODEL_DIR / "runs" / "tb_lstm_genuine_v2"
HOLDOUT_LABEL_DIR = HOLDOUT_DIR / "labeled"

REF_FUSION = {
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
    "label": "ref_lstm_cond",
}

HOLDOUT_MONTHS = 2.5


# ── ic32 stack ────────────────────────────────────────────────────────────────

def _load_ic32_models():
    btu.SMART_ENTRY_MODE = "disabled"
    btu.MOMENTUM_DYNAMIC_THRESHOLD_ENABLED = False
    btu.TREND_DYNAMIC_THRESHOLD_ENABLED = False
    btu.LSTM_STANDALONE_ENABLED = False

    lgbm = joblib.load(IC32_RUN / "lgbm.pkl")
    lstm = load_lstm(MODEL_DIR / "lstm_best.pt", device="cpu")
    lstm_scaler = joblib.load(MODEL_DIR / "lstm_scaler.pkl")
    with open(MODEL_DIR / "feature_cols_ic32_regime.json") as f:
        feat_cols = json.load(f)
    with open(MODEL_DIR / "feature_cols_lstm_temporal.json") as f:
        lstm_feat_cols = json.load(f)
    guardian = joblib.load(MODEL_DIR / "guardian_clean_v2.pkl")
    guardian_scaler = joblib.load(MODEL_DIR / "guardian_clean_v2_scaler.pkl")
    with open(MODEL_DIR / "guardian_clean_v2_feature_cols.json") as f:
        guardian_feat_cols = json.load(f)
    g_static = [c for c in guardian_feat_cols if c not in GUARDIAN_DYNAMIC_FEATURES]
    return {
        "lgbm": lgbm, "lstm": lstm, "lstm_scaler": lstm_scaler,
        "feat_cols": feat_cols, "lstm_feat_cols": lstm_feat_cols,
        "guardian": guardian, "guardian_scaler": guardian_scaler,
        "g_static": g_static,
    }


def run_ic32_holdout() -> list[dict]:
    models = _load_ic32_models()
    all_trades: list[dict] = []

    for sym in ALL_COINS:
        p = HOLDOUT_LABEL_DIR / f"{sym}_features_v3.parquet"
        if not p.exists():
            continue
        df = pd.read_parquet(p)
        df = ensure_utc_index(df).sort_index()
        df = df[(df.index >= OOS_START) & (df.index < OOS_END)]

        rp = HOLDOUT_LABEL_DIR / f"{sym}_regime_h1.parquet"
        if rp.exists():
            try:
                reg = pd.read_parquet(rp)
                for col in ["hmm_regime_enc", "hmm_regime"]:
                    if col in df.columns:
                        df = df.drop(columns=[col])
                df = df.join(reg[["hmm_regime_enc", "hmm_regime"]], how="left")
                df["hmm_regime_enc"] = df["hmm_regime_enc"].fillna(1).astype("int32")
            except Exception:
                pass
        if "hmm_regime_enc" not in df.columns:
            df["hmm_regime_enc"] = 1

        mask = df["label"].astype(str).isin(LABEL_MAP)
        df = df[mask].copy()
        n = len(df)
        if n < 50:
            continue

        feat_cols = models["feat_cols"]
        X = np.zeros((n, len(feat_cols)), dtype=np.float64)
        for idx, col in enumerate(feat_cols):
            if col in df.columns:
                X[:, idx] = df[col].ffill().fillna(0).values.astype(np.float64)

        y_pred, confidence = hierarchical_predict(
            None, models["lgbm"], models["lstm"], models["lstm_scaler"],
            X, feat_cols, [], df, model_dir=IC32_RUN,
            lstm_feat_cols=models["lstm_feat_cols"],
        )
        below = (y_pred != 1) & (confidence < CONFIDENCE_THRESHOLD_ENTRY)
        y_pred[below] = 1

        atr = df["atr_14_h1"].values if "atr_14_h1" in df.columns else np.ones(n)
        close = df["close"].values
        high = df["high"].values if "high" in df.columns else close
        low = df["low"].values if "low" in df.columns else close
        h4_sh = df["h4_swing_high"].values if "h4_swing_high" in df.columns else None
        h4_sl = df["h4_swing_low"].values if "h4_swing_low" in df.columns else None
        h4t = df["h4_trend"].values if "h4_trend" in df.columns else None
        volr = df["vol_ratio_20"].values if "vol_ratio_20" in df.columns else None
        X_guardian = compute_guardian_static_array(df, models["g_static"])

        report = full_trading_report(
            y_pred=y_pred, y_actual=df["label"].map(LABEL_MAP).values.astype(np.int64),
            atr=atr, close=close, high=high, low=low,
            h4_swing_highs=h4_sh, h4_swing_lows=h4_sl,
            index=df.index, modal=MODAL_PER_TRADE, leverages=LEVERAGE_SIM,
            fee_per_side=FEE_PER_SIDE, slippage=SLIPPAGE_PER_SIDE,
            min_rr=SWING_LABEL_MIN_RR, min_tp_atr=SWING_LABEL_MIN_TP,
            max_sl_atr=SWING_LABEL_MAX_SL,
            tp_fallback_atr=TP_SL_FALLBACK_TP, sl_fallback_atr=TP_SL_FALLBACK_SL,
            max_hold=MAX_HOLDING_BARS, symbol=sym, confidence=confidence,
            guardian_model=models["guardian"], guardian_scaler=models["guardian_scaler"],
            X_guardian=X_guardian, guardian_exit_threshold=GUARDIAN_EXIT_THRESHOLD,
            trailing_stop_enabled=TRAILING_STOP_ENABLED,
            trailing_stop_atr=TRAILING_STOP_ATR,
            trailing_stop_min_bars=TRAILING_STOP_MIN_BARS,
            h4_trend=h4t, vol_ratio=volr,
        )

        ts = df.index
        hmm = df["hmm_regime_enc"].values
        for t in report.get("trades", []):
            bi = t["bar_in"]
            h4_tr = float(h4t[bi]) if h4t is not None and bi < len(h4t) else np.nan
            direction = t["direction"]
            align = _trend_align(direction, h4_tr)
            all_trades.append({
                "model": "ic32_regime_v1",
                "coin": sym,
                "entry_time": ts[bi],
                "exit_time": ts[min(t["bar_out"], len(ts) - 1)],
                "direction": direction,
                "confidence": round(float(confidence[bi]), 4),
                "entry_price": t["entry"],
                "exit_price": t["exit"],
                "tp": t.get("tp"),
                "sl": t.get("sl"),
                "rr": t.get("rr"),
                "outcome": t["outcome"],
                "net_pnl": t["net_pnl"],
                "modal_used": t.get("modal_used", MODAL_PER_TRADE),
                "hold_bars": t["bar_out"] - t["bar_in"],
                "hmm_state": int(hmm[bi]) if bi < len(hmm) else -1,
                "h4_trend": h4_tr,
                "trend_align": align,
                "vol_ratio_20": float(volr[bi]) if volr is not None and bi < len(volr) else np.nan,
            })
        logger.info(f"ic32 {sym}: {len(report.get('trades', []))} trades")

    return all_trades


# ── TB stack ──────────────────────────────────────────────────────────────────

def _load_tb_models():
    lgbm = joblib.load(MODEL_DIR / "lgbm_baseline.pkl")
    with open(MODEL_DIR / "feature_cols_v2.json", encoding="utf-8") as f:
        lgbm_feats = json.load(f)
    guard = joblib.load(MODEL_DIR / "guardian_best.pkl")
    scaler = joblib.load(MODEL_DIR / "guardian_scaler.pkl")
    with open(MODEL_DIR / "guardian_feature_cols.json", encoding="utf-8") as f:
        guard_feats = json.load(f)
    g_static = [f for f in guard_feats if f not in GUARDIAN_COMPUTED]
    with open(LSTM_RUN / "lstm_v4_selected_features.json", encoding="utf-8") as f:
        lstm_feats = json.load(f)
    lstm_model = load_lstm(LSTM_RUN / "lstm_momentum.pt", device="cpu")
    lstm_scaler = joblib.load(LSTM_RUN / "lstm_momentum_scaler.pkl")
    with open(MODEL_DIR / "inference_config.json", encoding="utf-8") as f:
        inf = json.load(f)
    mom = inf.get("cascade", {}).get("lstm_momentum", REF_FUSION)
    fusion_cfg = {
        "fusion": "lstm", "mode": "conditional_momentum",
        "bull_thr": float(mom.get("bull_thr", 0.38)),
        "bear_thr": float(mom.get("bear_thr", 0.50)),
        "boost": float(mom.get("boost", 0.10)),
        "opposite_pen": float(mom.get("opposite_pen", 0.14)),
        "near_miss_gap": float(mom.get("near_miss_gap", 0.03)),
        "vol_thr": float(mom.get("vol_thr", 2.0)),
        "proportional": bool(mom.get("proportional", True)),
        "enable_boost": bool(mom.get("enable_boost", True)),
        "enable_penalty": bool(mom.get("enable_penalty", True)),
        "label": "ref_lstm_cond",
    }
    return {
        "lgbm": lgbm, "lgbm_feats": lgbm_feats,
        "guard": guard, "guard_scaler": scaler,
        "guard_feats": guard_feats, "g_static": g_static,
        "lstm_model": lstm_model, "lstm_scaler": lstm_scaler, "lstm_feats": lstm_feats,
        "fusion_cfg": fusion_cfg,
    }


def _lstm_predict_proba(X_raw, lstm_model, lstm_scaler, seq_len):
    n, f = X_raw.shape
    probs = np.full((n, 3), 1.0 / 3.0, dtype=np.float32)
    if n < seq_len:
        return probs
    X_sc = lstm_scaler.transform(X_raw.reshape(-1, f)).reshape(n, f).astype(np.float32)
    seqs = np.stack([X_sc[i - seq_len + 1: i + 1] for i in range(seq_len - 1, n)])
    chunks = []
    with torch.no_grad():
        for b in range(0, len(seqs), 512):
            t = torch.from_numpy(seqs[b: b + 512])
            lg = lstm_model(t)
            chunks.append(torch.softmax(lg, dim=1).cpu().numpy())
    probs[seq_len - 1:] = np.concatenate(chunks, axis=0)
    return probs


def _load_tb_coin(sym: str, models: dict) -> dict | None:
    path = HOLDOUT_LABEL_DIR / f"{sym}_features_v3.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    df = ensure_utc_index(df).sort_index()
    df = df[(df.index >= OOS_START) & (df.index < OOS_END)]
    if len(df) < LSTM_SEQ_LEN + 10:
        return None
    n = len(df)
    lgbm_feats = models["lgbm_feats"]
    X_lgbm = np.zeros((n, len(lgbm_feats)), dtype=np.float64)
    for i, col in enumerate(lgbm_feats):
        if col in df.columns:
            X_lgbm[:, i] = df[col].ffill().fillna(0).values
    proba = models["lgbm"].predict_proba(X_lgbm)
    p0, p2 = proba[:, 0].astype(np.float32), proba[:, 2].astype(np.float32)

    lstm_feats = models["lstm_feats"]
    X_lstm = np.zeros((n, len(lstm_feats)), dtype=np.float64)
    for i, col in enumerate(lstm_feats):
        if col in df.columns:
            X_lstm[:, i] = df[col].ffill().fillna(0).values
    lstm_p = _lstm_predict_proba(X_lstm, models["lstm_model"], models["lstm_scaler"], LSTM_SEQ_LEN)
    lstm_valid = np.isfinite(lstm_p).all(axis=1)
    vol_spike = (
        df["vol_spike_zscore"].fillna(-99).values.astype(np.float32)
        if "vol_spike_zscore" in df.columns else np.full(n, -99.0, np.float32)
    )
    h4t = df["h4_trend"].values if "h4_trend" in df.columns else None

    flow_arr = (
        df["flow_momentum_3bar"].ffill().fillna(0).values.astype(np.float64)
        if "flow_momentum_3bar" in df.columns else np.zeros(n)
    )

    return {
        "sym": sym, "ts": df.index,
        "p0": p0, "p2": p2,
        "hmm": df["hmm_regime_enc"].fillna(-1).values.astype(np.int8)
        if "hmm_regime_enc" in df.columns else np.full(n, -1, np.int8),
        "lstm_p": lstm_p, "lstm_valid": lstm_valid, "vol_spike": vol_spike,
        "close": df["close"].values.astype(np.float64),
        "high": df["high"].values.astype(np.float64),
        "low": df["low"].values.astype(np.float64),
        "atr": df["atr_14_h1"].values.astype(np.float64),
        "h4_sh": df["h4_swing_high"].values.astype(np.float64)
        if "h4_swing_high" in df.columns else np.full(n, np.nan),
        "h4_sl": df["h4_swing_low"].values.astype(np.float64)
        if "h4_swing_low" in df.columns else np.full(n, np.nan),
        "h4_trend": h4t,
        "df": df,
        "flow_arr": flow_arr,
    }


def run_tb_holdout() -> list[dict]:
    models = _load_tb_models()
    hmm_cfg = load_hmm_cfg()
    g_params = load_guardian_params()
    cfg = models["fusion_cfg"]
    all_trades: list[dict] = []

    common = dict(
        modal=MODAL_PER_TRADE, leverage=LEVERAGE_SIM[0],
        fee_per_side=FEE_PER_SIDE, slippage=SLIPPAGE_PER_SIDE,
        max_hold=MAX_HOLDING_BARS,
        min_rr=SWING_LABEL_MIN_RR, min_tp_atr=SWING_LABEL_MIN_TP,
        max_sl_atr=SWING_LABEL_MAX_SL,
        tp_fallback_atr=TP_SL_FALLBACK_TP, sl_fallback_atr=TP_SL_FALLBACK_SL,
    )

    for sym in ALL_COINS:
        c = _load_tb_coin(sym, models)
        if c is None:
            continue
        y = build_y_pred(c, cfg, hmm_cfg)
        p0, p2 = c["p0"].copy(), c["p2"].copy()
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
            proportional=cfg.get("proportional", True),
        )
        _, conf, tl, ts = apply_hmm_thr(p0, p2, c["hmm"], hmm_cfg)
        modal_arr = compute_dynamic_modal(p0, p2, c["hmm"], y, MODAL_PER_TRADE, DYNSIZE_CFG, tl, ts)

        X_grd = compute_guardian_static_array(c["df"], models["g_static"])
        rep = simulate_trades_swing(
            y_pred=y, guardian_enabled=True,
            guardian_model=models["guard"], guardian_scaler=models["guard_scaler"],
            X_guardian=X_grd,
            guardian_feat_cols=models["guard_feats"],
            guardian_static_names=models["g_static"],
            flow_momentum_arr=c["flow_arr"],
            guardian_exit_threshold=g_params["exit_threshold"],
            guardian_min_hold_bars=g_params["min_hold_bars"],
            guardian_activation_atr=GUARDIAN_ACTIVATION_ATR,
            modal_arr=modal_arr,
            close=c["close"], high=c["high"], low=c["low"], atr=c["atr"],
            h4_swing_highs=c["h4_sh"], h4_swing_lows=c["h4_sl"],
            pyramiding_enabled=True,
            pyramiding_max_per_coin=1,
            **common,
        )

        ts_idx = c["ts"]
        h4t = c.get("h4_trend")
        for t in rep.get("trades", []):
            bi = t["bar_in"]
            direction = "LONG" if y[bi] == LONG else "SHORT"
            h4_tr = float(h4t[bi]) if h4t is not None and bi < len(h4t) else np.nan
            thr = float(tl[bi]) if direction == "LONG" else float(ts[bi])
            all_trades.append({
                "model": "tb_lstm_cond",
                "coin": sym,
                "entry_time": ts_idx[bi],
                "exit_time": ts_idx[min(t["bar_out"], len(ts_idx) - 1)],
                "direction": direction,
                "confidence": round(float(conf[bi]), 4),
                "thr_used": round(thr, 4),
                "conf_margin": round(float(conf[bi]) - thr, 4),
                "entry_price": t["entry"],
                "exit_price": t["exit"],
                "tp": t.get("tp"),
                "sl": t.get("sl"),
                "rr": t.get("rr"),
                "outcome": t["outcome"],
                "net_pnl": t["net_pnl"],
                "modal_used": t.get("modal_used", MODAL_PER_TRADE),
                "hold_bars": t["bar_out"] - t["bar_in"],
                "hmm_state": int(c["hmm"][bi]) if bi < len(c["hmm"]) else -1,
                "h4_trend": h4_tr,
                "trend_align": _trend_align(direction, h4_tr),
                "vol_spike": float(c["vol_spike"][bi]) if bi < len(c["vol_spike"]) else np.nan,
            })
        logger.info(f"tb {sym}: {len(rep.get('trades', []))} trades")

    return all_trades


# ── helpers ───────────────────────────────────────────────────────────────────

def _trend_align(direction: str, h4_trend: float) -> str:
    if not np.isfinite(h4_trend):
        return "unknown"
    if direction == "LONG":
        return "with" if h4_trend > 0 else "counter"
    if direction == "SHORT":
        return "with" if h4_trend < 0 else "counter"
    return "unknown"


def _normalize_outcome(oc: str) -> str:
    oc = str(oc).upper()
    if oc == "LOSS":
        return "sl_hit"
    if "GUARDIAN" in oc:
        if "MOMENTUM" in oc and "PARTIAL" in oc:
            return "guardian_momentum_partial"
        if "MOMENTUM" in oc:
            return "guardian_momentum_exit"
        return "guardian_exit"
    if oc in ("TIMEOUT", "TIMEOUT_MOMENTUM"):
        return "time_exit"
    if oc == "WIN":
        return "tp_hit"
    if oc == "TRAILING_STOP":
        return "trailing_stop"
    return oc.lower()


def _tag_weakness(row: pd.Series, model: str) -> str:
    tags = []
    if row["net_pnl"] <= 0:
        if row["exit_norm"] == "sl_hit":
            tags.append("sl_loss")
            if row["hold_bars"] <= 2:
                tags.append("quick_sl")
        elif row["exit_norm"].startswith("guardian"):
            tags.append("guardian_loss")
            if row["hold_bars"] <= 3:
                tags.append("early_guardian_cut")
        elif row["exit_norm"] == "time_exit":
            tags.append("time_exit_loss")
        if row["net_pnl"] < -0.8:
            tags.append("large_loss")
        elif row["net_pnl"] < -0.3:
            tags.append("medium_loss")
        else:
            tags.append("small_loss")
    else:
        tags.append("winner")

    if model == "ic32_regime_v1":
        if row["confidence"] < CONFIDENCE_THRESHOLD_ENTRY + 0.05:
            tags.append("near_conf_gate")
    else:
        margin = row.get("conf_margin", row["confidence"] - 0.45)
        if margin < 0.05:
            tags.append("marginal_entry")
        if row.get("modal_used", MODAL_PER_TRADE) > MODAL_PER_TRADE * 1.4:
            tags.append("oversized")

    if row["direction"] == "LONG" and row.get("trend_align") == "with":
        tags.append("long_with_trend")
    if row["direction"] == "SHORT" and row.get("trend_align") == "counter":
        tags.append("short_counter_trend")

    return "|".join(tags)


def trades_to_df(trades: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(trades)
    if df.empty:
        return df
    df["exit_norm"] = df["outcome"].map(_normalize_outcome)
    df["is_win"] = df["net_pnl"] > 0
    model = df["model"].iloc[0]
    df["weakness_tags"] = df.apply(lambda r: _tag_weakness(r, model), axis=1)
    return df


def scorecard_from_df(df: pd.DataFrame) -> dict:
    if df.empty:
        return {}
    wins = df[df["is_win"]]
    losses = df[~df["is_win"]]
    gpnl = wins["net_pnl"].sum()
    lloss = abs(losses["net_pnl"].sum())
    return {
        "n": len(df),
        "wr": len(wins) / len(df) * 100,
        "pnl": df["net_pnl"].sum(),
        "ppt": df["net_pnl"].mean(),
        "pf": gpnl / lloss if lloss > 0 else float("inf"),
        "long_n": int((df["direction"] == "LONG").sum()),
        "short_n": int((df["direction"] == "SHORT").sum()),
        "long_wr": (df[df["direction"] == "LONG"]["is_win"].mean() * 100)
        if (df["direction"] == "LONG").any() else 0,
        "short_wr": (df[df["direction"] == "SHORT"]["is_win"].mean() * 100)
        if (df["direction"] == "SHORT").any() else 0,
        "avg_hold": df["hold_bars"].mean(),
        "avg_modal": df["modal_used"].mean() if "modal_used" in df.columns else MODAL_PER_TRADE,
        "sl_pct": (df["exit_norm"] == "sl_hit").mean() * 100,
        "guardian_pct": df["exit_norm"].str.startswith("guardian").mean() * 100,
    }


def breakdown_table(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    rows = []
    for keys, g in df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        wins = g[g["is_win"]]
        losses = g[~g["is_win"]]
        gpnl = wins["net_pnl"].sum()
        lloss = abs(losses["net_pnl"].sum())
        rows.append({
            **dict(zip(group_cols, keys)),
            "n": len(g),
            "wr": len(wins) / len(g) * 100 if len(g) else 0,
            "net_pnl": g["net_pnl"].sum(),
            "avg_pnl": g["net_pnl"].mean(),
            "pf": gpnl / lloss if lloss > 0 else float("inf"),
            "avg_hold": g["hold_bars"].mean(),
        })
    return pd.DataFrame(rows).sort_values("net_pnl")


def weakness_summary(df: pd.DataFrame) -> list[dict]:
    """Aggregate per-trade weakness tags (not per day)."""
    tag_stats = defaultdict(lambda: {"n": 0, "wins": 0, "net": 0.0, "losses_only": 0})
    for _, row in df.iterrows():
        for tag in str(row["weakness_tags"]).split("|"):
            if not tag or tag == "winner":
                continue
            s = tag_stats[tag]
            s["n"] += 1
            if row["is_win"]:
                s["wins"] += 1
            else:
                s["losses_only"] += 1
            s["net"] += row["net_pnl"]
    out = []
    for tag, s in tag_stats.items():
        out.append({
            "tag": tag,
            "trades_tagged": s["n"],
            "loss_trades": s["losses_only"],
            "wr": s["wins"] / s["n"] * 100 if s["n"] else 0,
            "net_pnl": s["net"],
            "avg_pnl": s["net"] / s["n"] if s["n"] else 0,
        })
    return sorted(out, key=lambda x: x["net_pnl"])


def worst_trades(df: pd.DataFrame, n: int = 15) -> pd.DataFrame:
    cols = ["coin", "entry_time", "direction", "confidence", "outcome", "exit_norm",
            "hold_bars", "net_pnl", "modal_used", "trend_align", "weakness_tags"]
    cols = [c for c in cols if c in df.columns]
    return df.nsmallest(n, "net_pnl")[cols]


def md_table(df: pd.DataFrame, float_cols: int = 2) -> str:
    if df.empty:
        return "_empty_"
    d = df.copy()
    for c in d.columns:
        if pd.api.types.is_float_dtype(d[c]):
            d[c] = d[c].map(lambda x: f"{x:.{float_cols}f}" if np.isfinite(x) else str(x))
        elif isinstance(d[c].dtype, pd.CategoricalDtype):
            d[c] = d[c].astype(str)
    headers = list(d.columns)
    lines = [
        "| " + " | ".join(str(h) for h in headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for _, row in d.iterrows():
        lines.append("| " + " | ".join(str(row[h]) for h in headers) + " |")
    return "\n".join(lines)


def build_report(ic32_df: pd.DataFrame, tb_df: pd.DataFrame) -> str:
    ic32_sc = scorecard_from_df(ic32_df)
    tb_sc = scorecard_from_df(tb_df)
    today = datetime.now().strftime("%Y-%m-%d")

    lines = [
        f"# Holdout Per-Trade Comparison: ic32 vs TB",
        "",
        f"**Generated**: {today}",
        f"**Period**: {OOS_START.date()} - {OOS_END.date()} (~{HOLDOUT_MONTHS} bulan, data s/d Jun 13)",
        f"**Coins**: 21 | **Granularity**: per trade (bukan per hari)",
        "",
        "## 1. Scorecard Agregat",
        "",
        "| Metrik | ic32_regime_v1 | tb_lstm_cond | Delta (TB-ic32) |",
        "|--------|---------------:|-------------:|----------------:|",
    ]

    def _row(label, k, fmt="{:.1f}"):
        a, b = ic32_sc.get(k, 0), tb_sc.get(k, 0)
        if k == "n":
            d = f"{int(b - a):+d}"
            lines.append(f"| {label} | {int(a):,} | {int(b):,} | {d} |")
        elif k == "pnl":
            lines.append(f"| {label} | ${a:+.2f} | ${b:+.2f} | ${b-a:+.2f} |")
        elif k == "pf":
            lines.append(f"| {label} | {a:.2f} | {b:.2f} | {b-a:+.2f} |")
        else:
            lines.append(f"| {label} | {fmt.format(a)} | {fmt.format(b)} | {fmt.format(b-a)} |")

    for label, key, fmt in [
        ("Total trades", "n", "{:d}"),
        ("Win rate %", "wr", "{:.1f}"),
        ("Net PnL ($10 base)", "pnl", None),
        ("PnL/trade", "ppt", None),
        ("Profit factor", "pf", None),
        ("LONG trades", "long_n", "{:d}"),
        ("LONG WR %", "long_wr", "{:.1f}"),
        ("SHORT trades", "short_n", "{:d}"),
        ("SHORT WR %", "short_wr", "{:.1f}"),
        ("Avg hold bars", "avg_hold", "{:.1f}"),
        ("Avg modal used", "avg_modal", "${:.2f}"),
        ("SL hit %", "sl_pct", "{:.1f}"),
        ("Guardian exit %", "guardian_pct", "{:.1f}"),
    ]:
        if key == "n":
            lines.append(f"| Total trades | {ic32_sc['n']:,} | {tb_sc['n']:,} | {tb_sc['n']-ic32_sc['n']:+,} |")
        elif key == "long_n":
            lines.append(f"| LONG trades | {ic32_sc['long_n']:,} | {tb_sc['long_n']:,} | {tb_sc['long_n']-ic32_sc['long_n']:+,} |")
        elif key == "short_n":
            lines.append(f"| SHORT trades | {ic32_sc['short_n']:,} | {tb_sc['short_n']:,} | {tb_sc['short_n']-ic32_sc['short_n']:+,} |")
        elif key == "pnl":
            lines.append(f"| Net PnL | ${ic32_sc['pnl']:+.2f} | ${tb_sc['pnl']:+.2f} | ${tb_sc['pnl']-ic32_sc['pnl']:+.2f} |")
        elif key == "pf":
            lines.append(f"| Profit factor | {ic32_sc['pf']:.2f} | {tb_sc['pf']:.2f} | {tb_sc['pf']-ic32_sc['pf']:+.2f} |")
        elif key == "avg_modal":
            lines.append(f"| Avg modal | ${ic32_sc['avg_modal']:.2f} | ${tb_sc['avg_modal']:.2f} | ${tb_sc['avg_modal']-ic32_sc['avg_modal']:+.2f} |")
        elif key == "ppt":
            lines.append(f"| PnL/trade | ${ic32_sc['ppt']:+.3f} | ${tb_sc['ppt']:+.3f} | ${tb_sc['ppt']-ic32_sc['ppt']:+.3f} |")
        else:
            a, b = ic32_sc[key], tb_sc[key]
            lines.append(f"| {label} | {a:.1f} | {b:.1f} | {b-a:+.1f} |")

    lines += [
        "",
        "### Config (frozen)",
        "- **ic32**: LGBM thr 0.69/0.59, conf>=0.59, Guardian clean_v2 exit 0.65, fixed $10",
        "- **TB**: LGBM 36f + HMM Config B + LSTM conditional_momentum + Guardian v2 + DynSize",
        "",
        "## 2. Breakdown Per Exit Reason (per trade)",
        "",
        "### ic32",
        "",
        md_table(breakdown_table(ic32_df, ["exit_norm"])),
        "",
        "### TB",
        "",
        md_table(breakdown_table(tb_df, ["exit_norm"])),
        "",
        "## 3. Breakdown Per Arah + Trend Alignment",
        "",
        "### ic32",
        "",
        md_table(breakdown_table(ic32_df, ["direction", "trend_align"])),
        "",
        "### TB",
        "",
        md_table(breakdown_table(tb_df, ["direction", "trend_align"])),
        "",
        "## 4. Breakdown Confidence Bucket (per trade)",
        "",
    ]

    for name, df, bins in [
        ("ic32", ic32_df, [0.59, 0.65, 0.70, 0.75, 0.80, 1.0]),
        ("TB", tb_df, [0.45, 0.50, 0.55, 0.60, 0.65, 1.0]),
    ]:
        d = df.copy()
        d["conf_bucket"] = pd.cut(d["confidence"], bins=bins, right=False, include_lowest=True)
        lines += [f"### {name}", "", md_table(breakdown_table(d, ["conf_bucket"])), ""]

    lines += [
        "## 5. Breakdown Hold Bars (losers only, per trade)",
        "",
        "### ic32",
        "",
    ]
    ic32_loss = ic32_df[~ic32_df["is_win"]].copy()
    ic32_loss["hold_bucket"] = pd.cut(ic32_loss["hold_bars"], bins=[0, 2, 5, 10, 20, 100], right=True)
    lines.append(md_table(breakdown_table(ic32_loss, ["hold_bucket"])))
    lines += ["", "### TB", ""]
    tb_loss = tb_df[~tb_df["is_win"]].copy()
    tb_loss["hold_bucket"] = pd.cut(tb_loss["hold_bars"], bins=[0, 2, 5, 10, 20, 100], right=True)
    lines.append(md_table(breakdown_table(tb_loss, ["hold_bucket"])))

    lines += [
        "",
        "## 6. Kelemahan Per Trade Pattern (tag aggregation, bukan per hari)",
        "",
        "Setiap trade bisa punya beberapa tag. Angka di bawah = jumlah **trade** yang membawa tag tersebut.",
        "",
        "### ic32 — weakness tags (sorted by net PnL)",
        "",
    ]
    ic32_weak = pd.DataFrame(weakness_summary(ic32_df))
    lines.append(md_table(ic32_weak))
    lines += [
        "",
        "### TB — weakness tags (sorted by net PnL)",
        "",
    ]
    tb_weak = pd.DataFrame(weakness_summary(tb_df))
    lines.append(md_table(tb_weak))

    lines += [
        "",
        "## 7. Worst 15 Trades (per trade)",
        "",
        "### ic32",
        "",
        md_table(worst_trades(ic32_df)),
        "",
        "### TB",
        "",
        md_table(worst_trades(tb_df)),
        "",
        "## 8. Per Coin (top 5 best / worst net PnL)",
        "",
    ]

    for name, df in [("ic32", ic32_df), ("TB", tb_df)]:
        pc = breakdown_table(df, ["coin"])
        lines += [
            f"### {name} — worst 5 coins",
            "",
            md_table(pc.head(5)),
            "",
            f"### {name} — best 5 coins",
            "",
            md_table(pc.tail(5).iloc[::-1]),
            "",
        ]

    lines += [
        "## 9. Kesimpulan Kelemahan Masing-masing Model",
        "",
        "### ic32_regime_v1",
        "",
    ]

    ic32_sl = ic32_df[ic32_df["exit_norm"] == "sl_hit"]
    ic32_gd_loss = ic32_df[(ic32_df["exit_norm"].str.startswith("guardian")) & (~ic32_df["is_win"])]
    lines += [
        f"- **Volume lebih rendah** ({ic32_sc['n']} vs {tb_sc['n']} trade) — gate conf 0.59 + thr tinggi memfilter banyak sinyal.",
        f"- **SL hit** {len(ic32_sl)} trade ({ic32_sc['sl_pct']:.1f}%) — avg loss ${ic32_sl['net_pnl'].mean():.3f}" if len(ic32_sl) else "- SL hit minimal",
        f"- **Guardian exit rugi** {len(ic32_gd_loss)} trade — keluar dini di chop, avg ${ic32_gd_loss['net_pnl'].mean():.3f}" if len(ic32_gd_loss) else "",
        f"- **LONG** {ic32_sc['long_n']} trade WR {ic32_sc['long_wr']:.1f}% vs SHORT WR {ic32_sc['short_wr']:.1f}%",
        "- Kelemahan dominan: **precision tinggi tapi trade count rendah**; rugi terkonsentrasi di SL hit + guardian cut kecil.",
        "",
        "### tb_lstm_cond",
        "",
    ]

    tb_sl = tb_df[tb_df["exit_norm"] == "sl_hit"]
    tb_marg = tb_df[tb_df["weakness_tags"].str.contains("marginal_entry", na=False)]
    tb_oversz = tb_df[tb_df["weakness_tags"].str.contains("oversized", na=False)]
    lines += [
        f"- **Volume tinggi** ({tb_sc['n']} trade) — thr HMM rendah (0.45-0.55) + LSTM boost banyak entry marginal.",
        f"- **SL hit** {len(tb_sl)} trade ({tb_sc['sl_pct']:.1f}%) — sumber utama loss absolut, avg ${tb_sl['net_pnl'].mean():.3f}" if len(tb_sl) else "",
        f"- **Marginal entry** (conf margin <0.05): {len(tb_marg)} trade, net ${tb_marg['net_pnl'].sum():.2f}",
        f"- **Oversized** (modal >1.4x): {len(tb_oversz)} trade, net ${tb_oversz['net_pnl'].sum():.2f}",
        f"- **LONG** {tb_sc['long_n']} trade WR {tb_sc['long_wr']:.1f}% — dynsize sering boost modal di TRENDING_UP",
        "- Kelemahan dominan: **banyak trade marginal + SL hit**; dynsize amplifikasi loss saat entry lemah.",
        "",
        "## 10. File Output",
        "",
        f"- `holdout_ic32_trades_apr_jun26.csv` — {ic32_sc['n']:,} baris, 1 baris = 1 trade",
        f"- `holdout_tb_lstm_cond_trades_apr_jun26.csv` — {tb_sc['n']:,} baris, 1 baris = 1 trade",
        "",
        "*Generated by tools/holdout_ic32_vs_tb_per_trade.py (frozen config, export-only)*",
    ]

    return "\n".join(lines)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Running ic32 holdout...")
    ic32_trades = run_ic32_holdout()
    print("Running TB holdout...")
    tb_trades = run_tb_holdout()

    ic32_df = trades_to_df(ic32_trades)
    tb_df = trades_to_df(tb_trades)

    ic32_csv = OUT_DIR / "holdout_ic32_trades_apr_jun26.csv"
    tb_csv = OUT_DIR / "holdout_tb_lstm_cond_trades_apr_jun26.csv"
    ic32_df.to_csv(ic32_csv, index=False)
    tb_df.to_csv(tb_csv, index=False)

    report = build_report(ic32_df, tb_df)
    report_path = OUT_DIR / "holdout_ic32_vs_tb_per_trade_report.md"
    report_path.write_text(report, encoding="utf-8")

    ic32_sc = scorecard_from_df(ic32_df)
    tb_sc = scorecard_from_df(tb_df)
    print(f"\nic32: {ic32_sc['n']} trades | WR {ic32_sc['wr']:.1f}% | PnL ${ic32_sc['pnl']:+.2f} | PF {ic32_sc['pf']:.2f}")
    print(f"tb  : {tb_sc['n']} trades | WR {tb_sc['wr']:.1f}% | PnL ${tb_sc['pnl']:+.2f} | PF {tb_sc['pf']:.2f}")
    print(f"\nCSV  : {ic32_csv}")
    print(f"CSV  : {tb_csv}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()