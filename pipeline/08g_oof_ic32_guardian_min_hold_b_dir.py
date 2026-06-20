"""
OOF: guardian min_hold {2, 4} on B-dir + continuation_v1 full stack.

Apples-to-apples dengan production stack (8833 trade baseline).
Tidak menyentuh holdout.

Usage:
  python pipeline/08g_oof_ic32_guardian_min_hold_b_dir.py
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
from pipeline.backtest_utils import hierarchical_predict, compute_guardian_static_array
from core.evaluator import full_trading_report
from core.models import load_lstm
from core.utils import setup_logger, ensure_utc_index
from pipeline.ic32_fusion_shared import build_per_bar_thresholds, load_b_dir_hmm_cfg
from pipeline import ic32_fusion_shared as ifs
from config import (
    ALL_COINS, LABEL_DIR, MODEL_DIR, LABEL_MAP, TRAIN_CUTOFF_DATE,
    MODAL_PER_TRADE, LEVERAGE_SIM, FEE_PER_SIDE, SLIPPAGE_PER_SIDE,
    MAX_HOLDING_BARS, GUARDIAN_DYNAMIC_FEATURES,
    SWING_LABEL_MIN_RR, SWING_LABEL_MIN_TP, SWING_LABEL_MAX_SL,
    TP_SL_FALLBACK_TP, TP_SL_FALLBACK_SL,
    TRAILING_STOP_ENABLED, TRAILING_STOP_ATR, TRAILING_STOP_MIN_BARS,
)

logger = setup_logger("08g_min_hold_b_dir")
RUN_DIR = MODEL_DIR / "runs" / "ic32_regime_v1"
OOF_PATH = RUN_DIR / "oof_predictions.parquet"
OUT_JSON = RUN_DIR / "guardian_min_hold_b_dir_oof.json"
FLOW_MOM_WINDOW = 3
MIN_HOLD_GRID = (2, 4)
DYN_EXTRA = {"cvd_slope_h4_delta_entry", "ofi_h4_delta_entry", "flow_momentum_3bar"}


def _apply_live_config():
    prod = ifs.load_production_defaults()
    import config as project_config

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
        try:
            reg = pd.read_parquet(rp)
            if "hmm_regime_enc" in df.columns:
                df = df.drop(columns=["hmm_regime_enc"])
            df = df.join(reg[["hmm_regime_enc"]], how="left")
            df["hmm_regime_enc"] = df["hmm_regime_enc"].fillna(1).astype("int32")
        except Exception:
            df["hmm_regime_enc"] = 1
    if "cvd_slope_h4" in df.columns:
        df["cvd_slope_h4_delta"] = df["cvd_slope_h4"].diff(1)
    if "ofi_z_score" in df.columns:
        df["flow_momentum_3bar"] = df["ofi_z_score"].rolling(FLOW_MOM_WINDOW, min_periods=1).mean()
    mask = df["label"].astype(str).isin(LABEL_MAP)
    df = df[mask].copy()
    return df if len(df) >= 50 else None


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


def _agg(trades: list) -> dict:
    if not trades:
        return {"trades": 0, "wr": 0, "pf": 0, "pnl": 0, "ppt": 0, "early_gdn_los": 0}
    n = len(trades)
    wins = [t for t in trades if t.get("net_pnl", 0) > 0]
    losses = [t for t in trades if t.get("net_pnl", 0) <= 0]
    gpnl = sum(t["net_pnl"] for t in wins)
    gloss = abs(sum(t["net_pnl"] for t in losses))
    early_los = sum(
        1 for t in losses
        if "GUARDIAN" in t.get("outcome", "")
        and (t.get("bar_out", 0) - t.get("bar_in", 0)) <= 3
    )
    return {
        "trades": n,
        "wr": round(len(wins) / n * 100, 2),
        "pf": round(gpnl / gloss, 3) if gloss > 0 else float("inf"),
        "pnl": round(sum(t.get("net_pnl", 0) for t in trades), 2),
        "ppt": round(sum(t.get("net_pnl", 0) for t in trades) / n, 4),
        "early_gdn_los": early_los,
    }


def _run_coin(sym, oof_sym, feat_cols, lgbm, lstm, lstm_scaler, lstm_feats,
              hmm_cfg, live_cfg, gdn, min_hold):
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
        X_guardian=X_gd, guardian_exit_threshold=0.65, guardian_min_hold_bars=min_hold,
        guardian_feat_cols=gdn["feats"], guardian_static_names=gdn["static"],
        flow_momentum_arr=flow,
        trailing_stop_enabled=TRAILING_STOP_ENABLED,
        trailing_stop_atr=TRAILING_STOP_ATR, trailing_stop_min_bars=TRAILING_STOP_MIN_BARS,
        h4_trend=df["h4_trend"].values if "h4_trend" in df.columns else None,
        vol_ratio=df["vol_ratio_20"].values if "vol_ratio_20" in df.columns else None,
    )
    return rep.get("lev5x", rep).get("trades", [])


def main():
    live_cfg = _apply_live_config()
    hmm_cfg = load_b_dir_hmm_cfg()
    gdn = _load_guardian_cont()
    oof_all = pd.read_parquet(OOF_PATH)
    with open(MODEL_DIR / "feature_cols_ic32_regime.json", encoding="utf-8") as f:
        feat_cols = json.load(f)
    with open(MODEL_DIR / "feature_cols_lstm_temporal.json", encoding="utf-8") as f:
        lstm_feats = json.load(f)[:11]
    lgbm = joblib.load(RUN_DIR / "lgbm.pkl")
    lstm = load_lstm(MODEL_DIR / "lstm_best.pt", device="cpu")
    lstm_scaler = joblib.load(MODEL_DIR / "lstm_scaler.pkl")

    results = {}
    for mh in MIN_HOLD_GRID:
        trades = []
        for sym in ALL_COINS:
            oof_sym = oof_all[oof_all["coin"] == sym]
            trades.extend(_run_coin(sym, oof_sym, feat_cols, lgbm, lstm, lstm_scaler,
                                  lstm_feats, hmm_cfg, live_cfg, gdn, mh))
        sc = _agg(trades)
        results[str(mh)] = sc
        logger.info(f"min_hold={mh} B-dir: {sc}")

    b = results["2"]
    for k, sc in results.items():
        if b.get("trades"):
            sc["trade_pct_vs_baseline"] = round(sc["trades"] / b["trades"] * 100, 1)
            sc["delta_ppt"] = round(sc["ppt"] - b["ppt"], 4)

    out = {
        "guardian": "ic32_guardian_continuation_v1",
        "stack": "B-dir-combined + production cascade",
        "grid": list(MIN_HOLD_GRID),
        "baseline_min_hold": 2,
        "created": datetime.now().isoformat(),
        "results": results,
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print("\n=== OOF min_hold B-dir (continuation_v1) ===")
    print(f"{'min_hold':>8} {'trades':>8} {'WR%':>7} {'PF':>6} {'PPT':>8} {'early_gdn_los':>14}")
    for mh in MIN_HOLD_GRID:
        sc = results[str(mh)]
        print(f"{mh:>8} {sc['trades']:>8} {sc['wr']:>6.1f}% {sc['pf']:>6.2f} "
              f"${sc['ppt']:>+7.4f} {sc['early_gdn_los']:>14}")
    print(f"\nSaved {OUT_JSON}")


if __name__ == "__main__":
    main()