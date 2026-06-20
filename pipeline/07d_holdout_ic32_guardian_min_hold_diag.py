"""
Diagnostic holdout: Guardian min_hold 2 (prod) vs 4.

BUKAN grid -- konfirmasi sekali setelah OOF 08g.
Entry frozen B-dir-combined + production cascade.

Usage:
  python pipeline/07d_holdout_ic32_guardian_min_hold_diag.py
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
from config import (
    ALL_COINS, HOLDOUT_DIR, MODEL_DIR, LABEL_MAP,
    MODAL_PER_TRADE, LEVERAGE_SIM, FEE_PER_SIDE, SLIPPAGE_PER_SIDE,
    MAX_HOLDING_BARS, GUARDIAN_DYNAMIC_FEATURES,
    SWING_LABEL_MIN_RR, SWING_LABEL_MIN_TP, SWING_LABEL_MAX_SL,
    TP_SL_FALLBACK_TP, TP_SL_FALLBACK_SL,
    TRAILING_STOP_ENABLED, TRAILING_STOP_ATR, TRAILING_STOP_MIN_BARS,
)

logger = setup_logger("07d_min_hold_holdout")
HOLDOUT_LABEL_DIR = HOLDOUT_DIR / "labeled"
RUN_DIR = MODEL_DIR / "runs" / "ic32_regime_v1"
FROZEN_PATH = RUN_DIR / "b_dir_combined_frozen.json"
INF_CFG_PATH = MODEL_DIR / "inference_config.json"
OUT_JSON = RUN_DIR / "holdout_guardian_min_hold_diag_apr_jun26.json"
FLOW_MOM_WINDOW = 3
HOLDOUT_MONTHS = 2.5
DYN_EXTRA = {"cvd_slope_h4_delta_entry", "ofi_h4_delta_entry", "flow_momentum_3bar"}


def _apply_live_config(min_hold: int) -> dict:
    prod = ifs.load_production_defaults()
    import config as project_config

    with open(INF_CFG_PATH, encoding="utf-8") as f:
        inf = json.load(f)
    rr = inf.get("rr_gate", {})

    project_config.CONFIDENCE_THRESHOLD_ENTRY = prod["conf_entry"]
    project_config.LSTM_ADJUST_AGREE_BOOST = prod["agree_boost"]
    project_config.LSTM_ADJUST_NEUTRAL_PEN = prod["neutral_pen"]
    project_config.LSTM_ADJUST_OPPOSITE_PEN = prod["opposite_pen"]
    project_config.LSTM_DIRECTIONAL_REVIEW_THRESHOLD = prod["dir_review_thr"]
    project_config.LSTM_FLAT_REVIEW_ENABLED = prod["flat_review"]
    project_config.LSTM_CONFIRMATION_ENABLED = True
    project_config.REGIME_AWARE_ALIGNMENT = prod["flip"]
    project_config.HMM_GATE_LSTM_ENABLED = prod["hmm_gate_lstm"]
    project_config.GUARDIAN_EXIT_THRESHOLD = 0.65
    project_config.GUARDIAN_MIN_HOLD_BARS = min_hold

    btu.CONFIDENCE_THRESHOLD_ENTRY = prod["conf_entry"]
    btu.LSTM_ADJUST_AGREE_BOOST = prod["agree_boost"]
    btu.LSTM_ADJUST_NEUTRAL_PEN = prod["neutral_pen"]
    btu.LSTM_ADJUST_OPPOSITE_PEN = prod["opposite_pen"]
    btu.LSTM_DIRECTIONAL_REVIEW_THRESHOLD = prod["dir_review_thr"]
    btu.LSTM_FLAT_REVIEW_ENABLED = prod["flat_review"]
    btu.LSTM_CONFIRMATION_ENABLED = True
    btu.REGIME_AWARE_ALIGNMENT = prod["flip"]
    btu.HMM_GATE_LSTM_ENABLED = prod["hmm_gate_lstm"]
    btu.SMART_ENTRY_MODE = "disabled"
    btu.MOMENTUM_DYNAMIC_THRESHOLD_ENABLED = False
    btu.TREND_DYNAMIC_THRESHOLD_ENABLED = False
    btu.LSTM_STANDALONE_ENABLED = False

    return {
        "conf_entry": prod["conf_entry"],
        "gdn_min_hold": min_hold,
        "sl_trigger_mode": str(rr.get("sl_trigger_mode", "close")),
    }


def _load_frozen_cfg() -> dict:
    with open(FROZEN_PATH, encoding="utf-8") as f:
        data = json.load(f)
    raw = data["per_state_thresholds"]
    return {int(k): (float(v[0]), float(v[1])) for k, v in raw.items()}


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


def _scorecard(trades: list) -> dict:
    if not trades:
        return {"trades": 0}
    n = len(trades)
    wins = [t for t in trades if t.get("net_pnl", 0) > 0]
    losses = [t for t in trades if t.get("net_pnl", 0) <= 0]
    gpnl = sum(t["net_pnl"] for t in wins)
    gloss = abs(sum(t["net_pnl"] for t in losses))
    pf = gpnl / gloss if gloss > 0 else float("inf")
    hold_bars = [t.get("bar_out", 0) - t.get("bar_in", 0) for t in trades]
    early_gdn_los = sum(
        1 for t in losses
        if "GUARDIAN" in t.get("outcome", "")
        and (t.get("bar_out", 0) - t.get("bar_in", 0)) <= 3
    )
    return {
        "trades": n,
        "trades_per_month": round(n / HOLDOUT_MONTHS, 1),
        "win_rate": round(len(wins) / n * 100, 2),
        "total_pnl": round(sum(t.get("net_pnl", 0) for t in trades), 2),
        "pnl_per_trade": round(sum(t.get("net_pnl", 0) for t in trades) / n, 4),
        "profit_factor": round(pf, 3),
        "avg_hold_bars": round(float(np.mean(hold_bars)), 2),
        "median_hold_bars": round(float(np.median(hold_bars)), 2),
        "early_gdn_losers": early_gdn_los,
    }


def _run_holdout(sym: str, hmm_cfg: dict, live_cfg: dict, gdn: dict,
                 feat_cols: list, lstm_feats: list, lgbm, lstm, lstm_scaler) -> list:
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

    hmm_enc = df["hmm_regime_enc"].values.astype(np.int32)
    thr_l, thr_s = build_per_bar_thresholds(hmm_enc, hmm_cfg)
    y_pred, confidence = hierarchical_predict(
        None, lgbm, lstm, lstm_scaler, X, feat_cols, [], df,
        model_dir=RUN_DIR, lstm_feat_cols=lstm_feats,
        per_bar_thr_long=thr_l, per_bar_thr_short=thr_s,
    )
    below = (y_pred != 1) & (confidence < live_cfg["conf_entry"])
    y_pred[below] = 1

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
        guardian_min_hold_bars=live_cfg["gdn_min_hold"],
        guardian_feat_cols=gdn["feats"], guardian_static_names=gdn["static"],
        flow_momentum_arr=flow,
        trailing_stop_enabled=TRAILING_STOP_ENABLED,
        trailing_stop_atr=TRAILING_STOP_ATR, trailing_stop_min_bars=TRAILING_STOP_MIN_BARS,
        h4_trend=df["h4_trend"].values if "h4_trend" in df.columns else None,
        vol_ratio=df["vol_ratio_20"].values if "vol_ratio_20" in df.columns else None,
        sl_trigger_mode=live_cfg["sl_trigger_mode"],
    )
    return rep.get("lev5x", rep).get("trades", [])


def main():
    hmm_cfg = _load_frozen_cfg()
    gdn = _load_guardian_cont()
    with open(MODEL_DIR / "feature_cols_ic32_regime.json", encoding="utf-8") as f:
        feat_cols = json.load(f)
    with open(MODEL_DIR / "feature_cols_lstm_temporal.json", encoding="utf-8") as f:
        lstm_feats = json.load(f)[:11]
    lgbm = joblib.load(RUN_DIR / "lgbm.pkl")
    lstm = load_lstm(MODEL_DIR / "lstm_best.pt", device="cpu")
    lstm_scaler = joblib.load(MODEL_DIR / "lstm_scaler.pkl")

    results = {}
    for mh in (2, 4):
        live_cfg = _apply_live_config(mh)
        trades = []
        for sym in ALL_COINS:
            trades.extend(_run_holdout(sym, hmm_cfg, live_cfg, gdn, feat_cols, lstm_feats,
                                       lgbm, lstm, lstm_scaler))
        results[f"min_hold_{mh}"] = _scorecard(trades)
        logger.info(f"min_hold={mh}: {results[f'min_hold_{mh}']}")

    b = results["min_hold_2"]
    c = results["min_hold_4"]
    out = {
        "methodology": "holdout_diagnostic_once",
        "period": "Apr-Jun 2026",
        "guardian": "ic32_guardian_continuation_v1",
        "entry": "B-dir-combined frozen",
        "results": results,
        "delta_ppt": round(c.get("pnl_per_trade", 0) - b.get("pnl_per_trade", 0), 4),
        "created": datetime.now().isoformat(),
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=str)

    print("\n=== Holdout Guardian min_hold diagnostic ===")
    print(f"{'variant':<16} {'trades':>7} {'WR%':>7} {'PPT':>8} {'PF':>6} {'early_gdn_los':>14}")
    for name, sc in results.items():
        if not sc.get("trades"):
            continue
        print(f"{name:<16} {sc['trades']:>7} {sc['win_rate']:>6.1f}% "
              f"${sc['pnl_per_trade']:>+7.4f} {sc['profit_factor']:>6.3f} "
              f"{sc.get('early_gdn_losers', 0):>14}")
    print(f"\nSaved {OUT_JSON}")


if __name__ == "__main__":
    main()