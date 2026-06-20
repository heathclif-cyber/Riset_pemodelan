"""
Diagnostic holdout: fixed modal vs DynSize cm_0.60 (OOF winner).

Sekali saja -- konfirmasi setelah OOF 08h.
Stack: B-dir + hard_consensus + continuation_v1 + min_hold=4.

Usage:
  python pipeline/07f_holdout_ic32_dynsize_diag.py
"""
import json
import sys
import warnings
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import pipeline.backtest_utils as btu
from pipeline.backtest_utils import hierarchical_predict, compute_guardian_static_array, apply_training_parity
from core.evaluator import full_trading_report
from core.models import load_lstm
from core.utils import setup_logger, ensure_utc_index
from pipeline.ic32_fusion_shared import build_per_bar_thresholds, load_b_dir_hmm_cfg
from pipeline import ic32_fusion_shared as ifs
from pipeline.lstm_fusion_shared import compute_dynamic_modal
from config import (
    ALL_COINS, HOLDOUT_DIR, MODEL_DIR, LABEL_MAP,
    MODAL_PER_TRADE, LEVERAGE_SIM, FEE_PER_SIDE, SLIPPAGE_PER_SIDE,
    MAX_HOLDING_BARS, GUARDIAN_DYNAMIC_FEATURES, GUARDIAN_MIN_HOLD_BARS,
    SWING_LABEL_MIN_RR, SWING_LABEL_MIN_TP, SWING_LABEL_MAX_SL,
    TP_SL_FALLBACK_TP, TP_SL_FALLBACK_SL,
    TRAILING_STOP_ENABLED, TRAILING_STOP_ATR, TRAILING_STOP_MIN_BARS,
)

logger = setup_logger("07f_dynsize_holdout")
HOLDOUT_LABEL_DIR = HOLDOUT_DIR / "labeled"
RUN_DIR = MODEL_DIR / "runs" / "ic32_regime_v1"
INF_CFG_PATH = MODEL_DIR / "inference_config.json"
OUT_JSON = RUN_DIR / "holdout_dynsize_diag_apr_jun26.json"
FLOW_MOM_WINDOW = 3
HOLDOUT_MONTHS = 2.5
DYN_EXTRA = {"cvd_slope_h4_delta_entry", "ofi_h4_delta_entry", "flow_momentum_3bar"}

WINNER_DYNSIZE = {
    "conf_window": 0.10,
    "conf_max_mult": 0.6,
    "regime_mult": {0: 0.75, 1: 1.0, 2: 1.0, "3_long": 1.5, "3_short": 0.75, -1: 0.80},
    "clamp_min": 0.5,
    "clamp_max": 2.0,
}


def _apply_live_config() -> dict:
    prod = ifs.load_production_defaults()
    import config as project_config

    with open(INF_CFG_PATH, encoding="utf-8") as f:
        inf = json.load(f)
    rr = inf.get("rr_gate", {})

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
    project_config.GUARDIAN_EXIT_THRESHOLD = 0.65
    project_config.GUARDIAN_MIN_HOLD_BARS = GUARDIAN_MIN_HOLD_BARS
    btu.SMART_ENTRY_MODE = "disabled"
    btu.MOMENTUM_DYNAMIC_THRESHOLD_ENABLED = False
    btu.TREND_DYNAMIC_THRESHOLD_ENABLED = False
    btu.LSTM_STANDALONE_ENABLED = False

    return {
        "conf_entry": prod["conf_entry"],
        "sl_trigger_mode": str(rr.get("sl_trigger_mode", "close")),
    }


def _add_momentum_feats(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "cvd_slope_h4" in df.columns:
        df["cvd_slope_h4_delta"] = df["cvd_slope_h4"].diff(1)
    if "ofi_z_score" in df.columns:
        df["flow_momentum_3bar"] = df["ofi_z_score"].rolling(FLOW_MOM_WINDOW, min_periods=1).mean()
    else:
        df["flow_momentum_3bar"] = 0.0
    return df


def _load_guardian_cont() -> dict:
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


def _scorecard(trades: list, base_modal: float = MODAL_PER_TRADE) -> dict:
    if not trades:
        return {"trades": 0}
    n = len(trades)
    wins = [t for t in trades if t.get("net_pnl", 0) > 0]
    losses = [t for t in trades if t.get("net_pnl", 0) <= 0]
    gpnl = sum(t["net_pnl"] for t in wins)
    gloss = abs(sum(t["net_pnl"] for t in losses))
    pf = gpnl / gloss if gloss > 0 else float("inf")
    tpnl = sum(t.get("net_pnl", 0) for t in trades)
    modals = [t.get("modal_used", base_modal) for t in trades]
    avg_modal = float(np.mean(modals))
    ppt = tpnl / n
    return {
        "trades": n,
        "trades_per_month": round(n / HOLDOUT_MONTHS, 1),
        "win_rate": round(len(wins) / n * 100, 2),
        "total_pnl": round(tpnl, 2),
        "pnl_per_trade": round(ppt, 4),
        "ppt_norm": round(ppt * (base_modal / avg_modal), 4) if avg_modal > 0 else 0.0,
        "profit_factor": round(pf, 3),
        "avg_modal": round(avg_modal, 2),
    }


def _run_holdout(sym, hmm_cfg, live_cfg, gdn, feat_cols, lstm_feats, lgbm, lstm, lstm_scaler,
                 use_dynsize: bool) -> list:
    p = HOLDOUT_LABEL_DIR / f"{sym}_features_v3.parquet"
    if not p.exists():
        return []
    df = pd.read_parquet(p)
    df = ensure_utc_index(df).sort_index()
    df = _add_momentum_feats(df)
    rp = HOLDOUT_LABEL_DIR / f"{sym}_regime_h1.parquet"
    if rp.exists():
        try:
            reg = pd.read_parquet(rp)
            if "hmm_regime_enc" in df.columns:
                df = df.drop(columns=["hmm_regime_enc"])
            df = df.join(reg[["hmm_regime_enc"]], how="left")
            df["hmm_regime_enc"] = df["hmm_regime_enc"].fillna(1).astype("int32")
        except Exception:
            df["hmm_regime_enc"] = 1
    mask = df["label"].astype(str).isin(LABEL_MAP)
    df = df[mask].copy()
    n = len(df)
    if n < 30:
        return []

    # Apply same training parity overrides seperti live inference
    df = apply_training_parity(df)

    X = np.zeros((n, len(feat_cols)), dtype=np.float64)
    for idx, col in enumerate(feat_cols):
        if col in df.columns:
            X[:, idx] = df[col].ffill().fillna(0).values.astype(np.float64)

    proba = lgbm.predict_proba(X)
    p0 = proba[:, 0].astype(np.float32)
    p2 = proba[:, 2].astype(np.float32)
    hmm_enc = df["hmm_regime_enc"].values.astype(np.int32)
    thr_l, thr_s = build_per_bar_thresholds(hmm_enc, hmm_cfg)
    y_pred, confidence = hierarchical_predict(
        None, lgbm, lstm, lstm_scaler, X, feat_cols, [], df,
        model_dir=RUN_DIR, lstm_feat_cols=lstm_feats,
        per_bar_thr_long=thr_l, per_bar_thr_short=thr_s,
    )
    below = (y_pred != 1) & (confidence < live_cfg["conf_entry"])
    y_pred[below] = 1

    modal_arr = None
    if use_dynsize:
        modal_arr = compute_dynamic_modal(
            p0, p2, hmm_enc, y_pred, MODAL_PER_TRADE, WINNER_DYNSIZE, thr_l, thr_s,
        )

    flow = df["flow_momentum_3bar"].ffill().fillna(0).values.astype(np.float64)
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
        max_hold=MAX_HOLDING_BARS, symbol=sym, confidence=confidence,
        guardian_model=gdn["model"], guardian_scaler=gdn["scaler"],
        X_guardian=X_gd, guardian_exit_threshold=0.65,
        guardian_min_hold_bars=GUARDIAN_MIN_HOLD_BARS,
        guardian_feat_cols=gdn["feats"], guardian_static_names=gdn["static"],
        flow_momentum_arr=flow, modal_arr=modal_arr,
        trailing_stop_enabled=TRAILING_STOP_ENABLED,
        trailing_stop_atr=TRAILING_STOP_ATR, trailing_stop_min_bars=TRAILING_STOP_MIN_BARS,
        h4_trend=df["h4_trend"].values if "h4_trend" in df.columns else None,
        vol_ratio=df["vol_ratio_20"].values if "vol_ratio_20" in df.columns else None,
        sl_trigger_mode=live_cfg["sl_trigger_mode"],
    )
    return rep.get("lev5x", rep).get("trades", [])


def main():
    live_cfg = _apply_live_config()
    hmm_cfg = load_b_dir_hmm_cfg()
    gdn = _load_guardian_cont()
    with open(MODEL_DIR / "feature_cols_ic32_regime.json", encoding="utf-8") as f:
        feat_cols = json.load(f)
    with open(MODEL_DIR / "feature_cols_lstm_temporal.json", encoding="utf-8") as f:
        lstm_feats = json.load(f)[:11]
    lgbm = joblib.load(RUN_DIR / "lgbm.pkl")
    lstm = load_lstm(MODEL_DIR / "lstm_best.pt", device="cpu")
    lstm_scaler = joblib.load(MODEL_DIR / "lstm_scaler.pkl")

    results = {}
    for label, dyn in (("fixed_modal", False), ("dynsize_cm_0.60", True)):
        trades = []
        for sym in ALL_COINS:
            trades.extend(_run_holdout(sym, hmm_cfg, live_cfg, gdn, feat_cols, lstm_feats,
                                       lgbm, lstm, lstm_scaler, dyn))
        results[label] = _scorecard(trades)
        logger.info(f"{label}: {results[label]}")

    b, d = results["fixed_modal"], results["dynsize_cm_0.60"]
    out = {
        "methodology": "holdout_diagnostic_once",
        "period": "Apr-Jun 2026",
        "stack": "B-dir + continuation_v1 + min_hold=4",
        "dynsize_winner": "cm_0.60",
        "results": results,
        "delta_ppt": round(d.get("pnl_per_trade", 0) - b.get("pnl_per_trade", 0), 4),
        "delta_ppt_norm": round(d.get("ppt_norm", 0) - b.get("ppt_norm", 0), 4),
        "created": datetime.now().isoformat(),
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=str)

    print("\n=== Holdout DynSize diagnostic ===")
    print(f"{'variant':<18} {'trades':>7} {'WR%':>7} {'PPT':>8} {'PPTn':>8} {'avg$':>6} {'PF':>6}")
    for name, sc in results.items():
        if not sc.get("trades"):
            continue
        print(f"{name:<18} {sc['trades']:>7} {sc['win_rate']:>6.1f}% "
              f"${sc['pnl_per_trade']:>+7.4f} ${sc['ppt_norm']:>+7.4f} "
              f"${sc['avg_modal']:>5.1f} {sc['profit_factor']:>6.3f}")
    print(f"\nSaved {OUT_JSON}")


if __name__ == "__main__":
    main()