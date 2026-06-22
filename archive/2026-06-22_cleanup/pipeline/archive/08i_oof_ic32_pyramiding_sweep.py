"""
OOF sweep: pyramiding max positions + exit schema on ic32 B-dir stack.

Stack frozen: B-dir + hard_consensus + continuation_v1 + min_hold=4 + SL close.
Tidak menyentuh holdout.

Exit schemas:
  independent      — tiap leg punya TP/SL/Guardian sendiri (live default bila pyramiding ON)
  shared_sl_first  — leg 2+ pakai SL harga leg pertama (anchor entry pertama)
  close_with_first — leg 2+ max hold = exit bar leg pertama (tutup bersamaan)

Usage:
  python pipeline/08i_oof_ic32_pyramiding_sweep.py
"""
import json
import sys
import warnings
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import pipeline.backtest_utils as btu
from pipeline.backtest_utils import hierarchical_predict, compute_guardian_static_array
from core.evaluator import full_trading_report
from core.models import load_lstm
from core.utils import setup_logger, ensure_utc_index
from pipeline.ic32_fusion_shared import build_per_bar_thresholds, load_b_dir_hmm_cfg
from pipeline import ic32_fusion_shared as ifs
from config import (
    ALL_COINS, LABEL_DIR, MODEL_DIR, LABEL_MAP, TRAIN_CUTOFF_DATE,
    MODAL_PER_TRADE, LEVERAGE_SIM, FEE_PER_SIDE, SLIPPAGE_PER_SIDE,
    MAX_HOLDING_BARS, GUARDIAN_DYNAMIC_FEATURES, GUARDIAN_MIN_HOLD_BARS,
    SWING_LABEL_MIN_RR, SWING_LABEL_MIN_TP, SWING_LABEL_MAX_SL,
    TP_SL_FALLBACK_TP, TP_SL_FALLBACK_SL,
    TRAILING_STOP_ENABLED, TRAILING_STOP_ATR, TRAILING_STOP_MIN_BARS,
)

logger = setup_logger("08i_pyramiding")
RUN_DIR = MODEL_DIR / "runs" / "ic32_regime_v1"
OOF_PATH = RUN_DIR / "oof_predictions.parquet"
OUT_JSON = RUN_DIR / "ic32_pyramiding_sweep_oof.json"
FLOW_MOM_WINDOW = 3
DYN_EXTRA = {"cvd_slope_h4_delta_entry", "ofi_h4_delta_entry", "flow_momentum_3bar"}

VARIANTS = [
    {"label": "no_pyr", "enabled": False, "max_per_coin": 1, "exit_mode": "independent"},
    {"label": "pyr2_independent", "enabled": True, "max_per_coin": 2, "exit_mode": "independent"},
    {"label": "pyr3_independent", "enabled": True, "max_per_coin": 3, "exit_mode": "independent"},
    {"label": "pyr2_shared_sl_first", "enabled": True, "max_per_coin": 2, "exit_mode": "shared_sl_first"},
    {"label": "pyr2_close_with_first", "enabled": True, "max_per_coin": 2, "exit_mode": "close_with_first"},
]


def _apply_live_config():
    prod = ifs.load_production_defaults()
    import config as project_config
    for mod in (project_config, btu):
        mod.CONFIDENCE_THRESHOLD_ENTRY = prod["conf_entry"]
        mod.LSTM_ADJUST_AGREE_BOOST = prod["agree_boost"]
        mod.LSTM_ADJUST_NEUTRAL_PEN = prod["neutral_pen"]
        mod.LSTM_ADJUST_OPPOSITE_PEN = prod["opposite_pen"]
        mod.LSTM_DIRECTIONAL_REVIEW_THRESHOLD = prod["dir_review_thr"]
        mod.LSTM_FLAT_REVIEW_ENABLED = prod["flat_review"]
        mod.LSTM_CONFIRMATION_ENABLED = True
        mod.REGIME_AWARE_ALIGNMENT = prod["flip"]
        mod.HMM_GATE_LSTM_ENABLED = prod["hmm_gate_lstm"]
    btu.SMART_ENTRY_MODE = "disabled"
    return {"conf_entry": prod["conf_entry"]}


def _prep_df(sym: str) -> pd.DataFrame | None:
    path = LABEL_DIR / f"{sym}_features_v3.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    df = ensure_utc_index(df).sort_index()
    df = df[df.index < TRAIN_CUTOFF_DATE]
    rp = LABEL_DIR / f"{sym}_regime_h1.parquet"
    if rp.exists():
        reg = pd.read_parquet(rp)
        if "hmm_regime_enc" in df.columns:
            df = df.drop(columns=["hmm_regime_enc"])
        df = df.join(reg[["hmm_regime_enc"]], how="left")
        df["hmm_regime_enc"] = df["hmm_regime_enc"].fillna(1).astype("int32")
    if "cvd_slope_h4" in df.columns:
        df["cvd_slope_h4_delta"] = df["cvd_slope_h4"].diff(1)
    if "ofi_z_score" in df.columns:
        df["flow_momentum_3bar"] = df["ofi_z_score"].rolling(FLOW_MOM_WINDOW, min_periods=1).mean()
    df = df[df["label"].astype(str).isin(LABEL_MAP)].copy()
    return df if len(df) >= 50 else None


def _load_guardian() -> dict:
    run = MODEL_DIR / "runs" / "ic32_guardian_continuation_v1"
    with open(run / "guardian_feature_cols.json", encoding="utf-8") as f:
        feats = json.load(f)
    dyn = set(GUARDIAN_DYNAMIC_FEATURES) | DYN_EXTRA
    static = [c for c in feats if c not in dyn]
    return {
        "model": joblib.load(run / "guardian.pkl"),
        "scaler": joblib.load(run / "guardian_scaler.pkl"),
        "feats": feats,
        "static": static,
    }


def _run_coin(sym, oof_sym, feat_cols, lgbm, lstm, lstm_scaler, lstm_feats,
              hmm_cfg, live_cfg, gdn, variant) -> list:
    df = _prep_df(sym)
    if df is None:
        return []
    n = len(df)
    merged = df.join(oof_sym[["p0", "p1", "p2", "has_oof"]], how="left")
    has_oof = merged["has_oof"].fillna(False).values.astype(bool)
    if has_oof.sum() < 30:
        return []

    oof_proba = np.column_stack([merged["p0"].values, merged["p1"].values, merged["p2"].values])
    X = np.zeros((n, len(feat_cols)), dtype=np.float64)
    for idx, col in enumerate(feat_cols):
        if col in df.columns:
            X[:, idx] = df[col].ffill().fillna(0).values.astype(np.float64)

    hmm_enc = df["hmm_regime_enc"].values.astype(np.int32)
    thr_l, thr_s = build_per_bar_thresholds(hmm_enc, hmm_cfg)
    y_pred, confidence = hierarchical_predict(
        None, lgbm, lstm, lstm_scaler, X, feat_cols, [], df,
        model_dir=RUN_DIR, lstm_feat_cols=lstm_feats, lgbm_proba=oof_proba,
        per_bar_thr_long=thr_l, per_bar_thr_short=thr_s,
    )
    below = has_oof & (y_pred != 1) & (confidence < live_cfg["conf_entry"])
    y_pred[below] = 1
    y_pred[~has_oof] = 1

    flow = df["flow_momentum_3bar"].ffill().fillna(0).values.astype(np.float64) \
        if "flow_momentum_3bar" in df.columns else np.zeros(n)
    X_gd = compute_guardian_static_array(df, gdn["static"])
    rep = full_trading_report(
        y_pred=y_pred, y_actual=df["label"].map(LABEL_MAP).values.astype(np.int64),
        atr=df["atr_14_h1"].values if "atr_14_h1" in df.columns else np.ones(n),
        close=df["close"].values, high=df["high"].values, low=df["low"].values,
        h4_swing_highs=df["h4_swing_high"].values if "h4_swing_high" in df.columns else None,
        h4_swing_lows=df["h4_swing_low"].values if "h4_swing_low" in df.columns else None,
        index=df.index, modal=MODAL_PER_TRADE, leverages=LEVERAGE_SIM,
        fee_per_side=FEE_PER_SIDE, slippage=SLIPPAGE_PER_SIDE,
        min_rr=SWING_LABEL_MIN_RR, min_tp_atr=SWING_LABEL_MIN_TP, max_sl_atr=SWING_LABEL_MAX_SL,
        tp_fallback_atr=TP_SL_FALLBACK_TP, sl_fallback_atr=TP_SL_FALLBACK_SL,
        max_hold=MAX_HOLDING_BARS, confidence=confidence, symbol=sym,
        guardian_model=gdn["model"], guardian_scaler=gdn["scaler"],
        X_guardian=X_gd, guardian_exit_threshold=0.65,
        guardian_min_hold_bars=GUARDIAN_MIN_HOLD_BARS,
        guardian_feat_cols=gdn["feats"], guardian_static_names=gdn["static"],
        flow_momentum_arr=flow,
        trailing_stop_enabled=TRAILING_STOP_ENABLED,
        trailing_stop_atr=TRAILING_STOP_ATR, trailing_stop_min_bars=TRAILING_STOP_MIN_BARS,
        h4_trend=df["h4_trend"].values if "h4_trend" in df.columns else None,
        vol_ratio=df["vol_ratio_20"].values if "vol_ratio_20" in df.columns else None,
        sl_trigger_mode="close",
        pyramiding_enabled=variant["enabled"],
        pyramiding_max_per_coin=variant["max_per_coin"],
        pyramiding_same_dir=True,
        pyramiding_exit_mode=variant["exit_mode"],
    )
    trades = rep.get("lev5x", rep).get("trades", [])
    for t in trades:
        t["symbol"] = sym
    return trades


def _pyramid_stats(trades: list) -> dict:
    """Leg overlap / stack depth per coin."""
    if not trades:
        return {"avg_legs_per_stack": 0, "pct_stacks_multi_leg": 0, "signals_blocked_est": 0}

    by_sym = defaultdict(list)
    for t in trades:
        by_sym[t["symbol"]].append(t)

    stacks_multi = 0
    total_stacks = 0
    leg_counts = []
    add_on_legs = 0

    for sym, ts in by_sym.items():
        ts = sorted(ts, key=lambda x: x["bar_in"])
        i = 0
        while i < len(ts):
            stack = [ts[i]]
            j = i + 1
            while j < len(ts):
                if ts[j]["bar_in"] < stack[-1]["bar_out"]:
                    stack.append(ts[j])
                    j += 1
                else:
                    break
            total_stacks += 1
            leg_counts.append(len(stack))
            if len(stack) > 1:
                stacks_multi += 1
                add_on_legs += len(stack) - 1
            i = j if j > i + 1 else i + 1

    return {
        "avg_legs_per_stack": round(float(np.mean(leg_counts)), 2) if leg_counts else 0,
        "pct_stacks_multi_leg": round(stacks_multi / total_stacks * 100, 1) if total_stacks else 0,
        "add_on_legs": add_on_legs,
        "total_stacks": total_stacks,
    }


def _agg(trades: list, variant: dict) -> dict:
    if not trades:
        return {"trades": 0, "label": variant["label"]}
    n = len(trades)
    wins = [t for t in trades if t.get("net_pnl", 0) > 0]
    losses = [t for t in trades if t.get("net_pnl", 0) <= 0]
    gpnl = sum(t["net_pnl"] for t in wins)
    gloss = abs(sum(t["net_pnl"] for t in losses))
    total_modal = sum(t.get("modal_used", MODAL_PER_TRADE) for t in trades)
    pyr = _pyramid_stats(trades)
    return {
        "label": variant["label"],
        "config": variant,
        "trades": n,
        "wr": round(len(wins) / n * 100, 2),
        "pf": round(gpnl / gloss, 3) if gloss > 0 else None,
        "pnl": round(sum(t.get("net_pnl", 0) for t in trades), 2),
        "ppt": round(sum(t.get("net_pnl", 0) for t in trades) / n, 4),
        "ppt_norm": round(sum(t.get("net_pnl", 0) for t in trades) / total_modal, 4) if total_modal else 0,
        "avg_modal": round(total_modal / n, 2),
        "avg_hold_bars": round(float(np.mean([
            t.get("bar_out", 0) - t.get("bar_in", 0) for t in trades
        ])), 2),
        **pyr,
    }


def main():
    live_cfg = _apply_live_config()
    hmm_cfg = load_b_dir_hmm_cfg()
    gdn = _load_guardian()
    oof_all = pd.read_parquet(OOF_PATH)
    with open(MODEL_DIR / "feature_cols_ic32_regime.json", encoding="utf-8") as f:
        feat_cols = json.load(f)
    with open(MODEL_DIR / "feature_cols_lstm_temporal.json", encoding="utf-8") as f:
        lstm_feats = json.load(f)[:11]
    lgbm = joblib.load(RUN_DIR / "lgbm.pkl")
    lstm = load_lstm(MODEL_DIR / "lstm_best.pt", device="cpu")
    lstm_scaler = joblib.load(MODEL_DIR / "lstm_scaler.pkl")

    results = {}
    for variant in VARIANTS:
        logger.info(f"Running {variant['label']}...")
        trades = []
        for sym in ALL_COINS:
            oof_sym = oof_all[oof_all["coin"] == sym]
            trades.extend(_run_coin(sym, oof_sym, feat_cols, lgbm, lstm, lstm_scaler,
                                  lstm_feats, hmm_cfg, live_cfg, gdn, variant))
        sc = _agg(trades, variant)
        results[variant["label"]] = sc
        logger.info(sc)

    base = results["no_pyr"]
    for label, sc in results.items():
        if base.get("trades"):
            sc["delta_trades_pct"] = round((sc["trades"] - base["trades"]) / base["trades"] * 100, 1)
            sc["delta_ppt"] = round(sc.get("ppt", 0) - base.get("ppt", 0), 4)
            sc["delta_ppt_norm"] = round(sc.get("ppt_norm", 0) - base.get("ppt_norm", 0), 4)

    out = {
        "protocol": "genuine_oof",
        "holdout_used": False,
        "stack": "B-dir + continuation_v1 + min_hold=4 + SL close",
        "exit_schema_doc": {
            "independent": "Tiap leg: TP/SL/Guardian dari bar entry leg itu (live paper_trading saat ini)",
            "shared_sl_first": "Leg 2+ pakai SL price leg pertama (anchor risiko)",
            "close_with_first": "Leg 2+ tidak boleh hold lebih lama dari exit bar leg pertama",
        },
        "created": datetime.now().isoformat(),
        "baseline": base,
        "results": results,
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print("\n=== OOF Pyramiding sweep ===")
    print(f"{'label':<22} {'trades':>7} {'WR%':>6} {'PF':>6} {'PPT':>8} {'norm':>8} {'multi%':>7} {'dPPT':>8}")
    for v in VARIANTS:
        sc = results[v["label"]]
        print(
            f"{sc['label']:<22} {sc.get('trades', 0):>7} {sc.get('wr', 0):>5.1f}% "
            f"{sc.get('pf', 0):>6.2f} ${sc.get('ppt', 0):>+7.4f} ${sc.get('ppt_norm', 0):>+7.4f} "
            f"{sc.get('pct_stacks_multi_leg', 0):>6.1f}% {sc.get('delta_ppt', 0):>+8.4f}"
        )
    print(f"\nSaved {OUT_JSON}")


if __name__ == "__main__":
    main()