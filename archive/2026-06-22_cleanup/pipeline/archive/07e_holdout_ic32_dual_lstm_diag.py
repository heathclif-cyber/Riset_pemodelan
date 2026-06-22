"""
Diagnostic holdout: best dual-LSTM vs baseline (sekali).

Usage:
  python pipeline/07e_holdout_ic32_dual_lstm_diag.py
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
from pipeline.ic32_fusion_shared import (
    IC32_DIR, COMPLEMENT_DIR,
    apply_dual_complement_to_proba, build_per_bar_thresholds,
    load_b_dir_hmm_cfg, load_production_defaults,
)
from config import (
    ALL_COINS, HOLDOUT_DIR, MODEL_DIR, LABEL_MAP,
    MODAL_PER_TRADE, LEVERAGE_SIM, FEE_PER_SIDE, SLIPPAGE_PER_SIDE,
    MAX_HOLDING_BARS, GUARDIAN_DYNAMIC_FEATURES,
    SWING_LABEL_MIN_RR, SWING_LABEL_MIN_TP, SWING_LABEL_MAX_SL,
    TP_SL_FALLBACK_TP, TP_SL_FALLBACK_SL,
    TRAILING_STOP_ENABLED, TRAILING_STOP_ATR, TRAILING_STOP_MIN_BARS,
)

logger = setup_logger("07e_dual_lstm_holdout")
HOLDOUT_LABEL_DIR = HOLDOUT_DIR / "labeled"
STAGE2_OUT = IC32_DIR / "ic32_dual_lstm_stage2_pipeline.json"
OUT_JSON = IC32_DIR / "holdout_dual_lstm_diag_apr_jun26.json"
INF_CFG_PATH = MODEL_DIR / "inference_config.json"
FLOW_MOM_WINDOW = 3
DYN_EXTRA = {"cvd_slope_h4_delta_entry", "ofi_h4_delta_entry", "flow_momentum_3bar"}
HOLDOUT_MONTHS = 2.5


def _apply_production_cascade() -> dict:
    prod = load_production_defaults()
    import config as project_config

    with open(INF_CFG_PATH, encoding="utf-8") as f:
        inf = json.load(f)
    sl_mode = str(inf.get("rr_gate", {}).get("sl_trigger_mode", "close"))

    project_config.CONFIDENCE_THRESHOLD_ENTRY = prod["conf_entry"]
    project_config.LSTM_ADJUST_AGREE_BOOST = prod["agree_boost"]
    project_config.LSTM_ADJUST_NEUTRAL_PEN = prod["neutral_pen"]
    project_config.LSTM_ADJUST_OPPOSITE_PEN = prod["opposite_pen"]
    project_config.LSTM_DIRECTIONAL_REVIEW_THRESHOLD = prod["dir_review_thr"]
    project_config.LSTM_FLAT_REVIEW_ENABLED = prod["flat_review"]
    project_config.LSTM_CONFIRMATION_ENABLED = True
    project_config.REGIME_AWARE_ALIGNMENT = prod["flip"]
    project_config.HMM_GATE_LSTM_ENABLED = prod["hmm_gate_lstm"]

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
    return {"conf_entry": prod["conf_entry"], "sl_trigger_mode": sl_mode}


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
    return {
        "trades": n,
        "trades_per_month": round(n / HOLDOUT_MONTHS, 1),
        "win_rate": round(len(wins) / n * 100, 2),
        "total_pnl": round(sum(t.get("net_pnl", 0) for t in trades), 2),
        "pnl_per_trade": round(sum(t.get("net_pnl", 0) for t in trades) / n, 4),
        "profit_factor": round(pf, 3),
    }


def _run_holdout(sym: str, hmm_cfg: dict, live_cfg: dict, gdn: dict,
                 feat_cols: list, lstm_feats: list, lgbm, lstm, lstm_scaler,
                 comp_model, comp_scaler, comp_feats: list, dual_cfg: dict | None) -> list:
    p = HOLDOUT_LABEL_DIR / f"{sym}_features_v3.parquet"
    if not p.exists():
        return []
    df = pd.read_parquet(p)
    df = ensure_utc_index(df).sort_index()
    if "cvd_slope_h4" in df.columns:
        df["cvd_slope_h4_delta"] = df["cvd_slope_h4"].diff(1)
    if "ofi_z_score" in df.columns:
        df["flow_momentum_3bar"] = df["ofi_z_score"].rolling(FLOW_MOM_WINDOW, min_periods=1).mean()
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

    gbm_feats = lgbm.feature_name_
    X_pred = np.zeros((n, len(gbm_feats)), dtype=np.float64)
    for idx, col in enumerate(gbm_feats):
        if col in df.columns:
            X_pred[:, idx] = df[col].ffill().fillna(0).values.astype(np.float64)
    oof_proba = lgbm.predict_proba(X_pred)

    hmm_enc = df["hmm_regime_enc"].values.astype(np.int32)
    thr_l, thr_s = build_per_bar_thresholds(hmm_enc, hmm_cfg)

    if dual_cfg and dual_cfg.get("dual_complement"):
        X_comp = np.zeros((n, len(comp_feats)), dtype=np.float64)
        for idx, col in enumerate(comp_feats):
            if col in df.columns:
                X_comp[:, idx] = df[col].ffill().fillna(0).values.astype(np.float64)
        from pipeline.backtest_utils import get_lstm_proba
        lstm_comp = get_lstm_proba(comp_model, comp_scaler, X_comp, n)
        vol_spike = df["vol_spike_zscore"].fillna(-99).values.astype(np.float32) \
            if "vol_spike_zscore" in df.columns else np.full(n, -99.0, dtype=np.float32)
        comp_valid = np.ones(n, dtype=bool)
        oof_proba = apply_dual_complement_to_proba(
            oof_proba, lstm_comp, vol_spike, comp_valid, hmm_enc, hmm_cfg, dual_cfg,
        )

    y_pred, confidence = hierarchical_predict(
        None, lgbm, lstm, lstm_scaler, X, feat_cols, [], df,
        model_dir=IC32_DIR, lstm_feat_cols=lstm_feats, lgbm_proba=oof_proba,
        per_bar_thr_long=thr_l, per_bar_thr_short=thr_s,
    )
    below = (y_pred != 1) & (confidence < live_cfg["conf_entry"])
    y_pred[below] = 1

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
        max_hold=MAX_HOLDING_BARS, symbol=sym, confidence=confidence,
        guardian_model=gdn["model"], guardian_scaler=gdn["scaler"],
        X_guardian=X_gd, guardian_exit_threshold=0.65, guardian_min_hold_bars=2,
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
    if not STAGE2_OUT.exists():
        raise FileNotFoundError(f"Run dual lstm stage2 first: {STAGE2_OUT}")

    with open(STAGE2_OUT, encoding="utf-8") as f:
        stage2 = json.load(f)
    best = stage2.get("best")
    if not best or not best.get("dual_complement"):
        print("No dual complement winner -- skip holdout diag")
        return

    dual_cfg = {k: best[k] for k in best if k in (
        "label", "dual_complement", "vol_thr", "bull_thr", "bear_thr",
        "near_miss_gap", "boost", "comp_opposite_pen",
    )}
    live_cfg = _apply_production_cascade()
    hmm_cfg = load_b_dir_hmm_cfg()
    gdn = _load_guardian_cont()

    with open(MODEL_DIR / "feature_cols_ic32_regime.json", encoding="utf-8") as f:
        feat_cols = json.load(f)
    with open(MODEL_DIR / "feature_cols_lstm_temporal.json", encoding="utf-8") as f:
        lstm_feats = json.load(f)[:11]
    comp_feat_path = COMPLEMENT_DIR / "ic32_lstm_swing_complement_v2_features.json"
    with open(comp_feat_path, encoding="utf-8") as f:
        comp_feats = json.load(f)

    lgbm = joblib.load(IC32_DIR / "lgbm.pkl")
    lstm = load_lstm(MODEL_DIR / "lstm_best.pt", device="cpu")
    lstm_scaler = joblib.load(MODEL_DIR / "lstm_scaler.pkl")
    comp_model = load_lstm(COMPLEMENT_DIR / "lstm_momentum.pt", device="cpu")
    comp_scaler = joblib.load(COMPLEMENT_DIR / "lstm_momentum_scaler.pkl")

    results = {}
    for name, cfg in [("baseline_no_dual", None), (dual_cfg["label"], dual_cfg)]:
        trades = []
        for sym in ALL_COINS:
            trades.extend(_run_holdout(
                sym, hmm_cfg, live_cfg, gdn, feat_cols, lstm_feats,
                lgbm, lstm, lstm_scaler, comp_model, comp_scaler, comp_feats, cfg,
            ))
        results[name] = _scorecard(trades)

    out = {
        "methodology": "holdout_diagnostic_once",
        "candidate": dual_cfg["label"],
        "results": results,
        "delta_ppt": round(
            results[dual_cfg["label"]].get("pnl_per_trade", 0)
            - results["baseline_no_dual"].get("pnl_per_trade", 0), 4,
        ),
        "created": datetime.now().isoformat(),
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=str)

    print("\n=== Holdout dual-LSTM diagnostic ===")
    for name, sc in results.items():
        print(f"{name}: {sc}")
    print(f"Saved {OUT_JSON}")


if __name__ == "__main__":
    main()