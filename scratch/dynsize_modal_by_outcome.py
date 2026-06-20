"""Forensics: avg modal_used by win/loss/SL under DynSize cm_0.60 (OOF stack)."""
import json
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# reuse 08h helpers
from pipeline import ic32_fusion_shared as ifs  # noqa
import pipeline.backtest_utils as btu
from pipeline.backtest_utils import hierarchical_predict, compute_guardian_static_array
from core.evaluator import full_trading_report
from core.models import load_lstm
from core.utils import ensure_utc_index
from pipeline.ic32_fusion_shared import build_per_bar_thresholds, load_b_dir_hmm_cfg
from pipeline.lstm_fusion_shared import compute_dynamic_modal
from config import (
    ALL_COINS, LABEL_DIR, MODEL_DIR, LABEL_MAP, TRAIN_CUTOFF_DATE,
    MODAL_PER_TRADE, LEVERAGE_SIM, FEE_PER_SIDE, SLIPPAGE_PER_SIDE,
    MAX_HOLDING_BARS, GUARDIAN_DYNAMIC_FEATURES, GUARDIAN_MIN_HOLD_BARS,
    SWING_LABEL_MIN_RR, SWING_LABEL_MIN_TP, SWING_LABEL_MAX_SL,
    TP_SL_FALLBACK_TP, TP_SL_FALLBACK_SL,
    TRAILING_STOP_ENABLED, TRAILING_STOP_ATR, TRAILING_STOP_MIN_BARS,
)

RUN_DIR = MODEL_DIR / "runs" / "ic32_regime_v1"
OOF_PATH = RUN_DIR / "oof_predictions.parquet"
DYN = {
    "conf_window": 0.10, "conf_max_mult": 0.6,
    "regime_mult": {0: 0.75, 1: 1.0, 2: 1.0, "3_long": 1.5, "3_short": 0.75, -1: 0.80},
    "clamp_min": 0.5, "clamp_max": 2.0,
}
DYN_EXTRA = {"cvd_slope_h4_delta_entry", "ofi_h4_delta_entry", "flow_momentum_3bar"}
FLOW_MOM_WINDOW = 3


def _apply():
    prod = ifs.load_production_defaults()
    import config as pc
    for m in (pc, btu):
        m.CONFIDENCE_THRESHOLD_ENTRY = prod["conf_entry"]
        m.LSTM_ADJUST_AGREE_BOOST = prod["agree_boost"]
        m.LSTM_ADJUST_OPPOSITE_PEN = prod["neutral_pen"]
        m.LSTM_ADJUST_OPPOSITE_PEN = prod["opposite_pen"]
        m.LSTM_DIRECTIONAL_REVIEW_THRESHOLD = prod["dir_review_thr"]
        m.LSTM_FLAT_REVIEW_ENABLED = prod["flat_review"]
        m.LSTM_CONFIRMATION_ENABLED = True
        m.REGIME_AWARE_ALIGNMENT = prod["flip"]
        m.HMM_GATE_LSTM_ENABLED = prod["hmm_gate_lstm"]
    btu.SMART_ENTRY_MODE = "disabled"


def _prep(sym):
    p = LABEL_DIR / f"{sym}_features_v3.parquet"
    if not p.exists():
        return None
    df = pd.read_parquet(p)
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


def main():
    _apply()
    hmm_cfg = load_b_dir_hmm_cfg()
    run = MODEL_DIR / "runs" / "ic32_guardian_continuation_v1"
    with open(run / "guardian_feature_cols.json") as f:
        feats = json.load(f)
    dyn = set(GUARDIAN_DYNAMIC_FEATURES) | DYN_EXTRA
    static = [c for c in feats if c not in dyn]
    gdn = {
        "model": joblib.load(run / "guardian.pkl"),
        "scaler": joblib.load(run / "guardian_scaler.pkl"),
        "feats": feats, "static": static,
    }
    oof_all = pd.read_parquet(OOF_PATH)
    with open(MODEL_DIR / "feature_cols_ic32_regime.json") as f:
        feat_cols = json.load(f)
    with open(MODEL_DIR / "feature_cols_lstm_temporal.json") as f:
        lstm_feats = json.load(f)[:11]
    lgbm = joblib.load(RUN_DIR / "lgbm.pkl")
    lstm = load_lstm(MODEL_DIR / "lstm_best.pt", device="cpu")
    lstm_scaler = joblib.load(MODEL_DIR / "lstm_scaler.pkl")
    prod = ifs.load_production_defaults()

    all_trades = []
    for sym in ALL_COINS:
        df = _prep(sym)
        if df is None:
            continue
        oof_sym = oof_all[oof_all["coin"] == sym]
        merged = df.join(oof_sym[["p0", "p1", "p2", "has_oof"]], how="left")
        has_oof = merged["has_oof"].fillna(False).values.astype(bool)
        if has_oof.sum() < 30:
            continue
        n = len(df)
        p0 = merged["p0"].values.astype(np.float32)
        p2 = merged["p2"].values.astype(np.float32)
        oof_proba = np.column_stack([p0, merged["p1"].values, p2])
        X = np.zeros((n, len(feat_cols)))
        for i, col in enumerate(feat_cols):
            if col in df.columns:
                X[:, i] = df[col].ffill().fillna(0).values
        hmm_enc = df["hmm_regime_enc"].values.astype(np.int32)
        thr_l, thr_s = build_per_bar_thresholds(hmm_enc, hmm_cfg)
        y_pred, confidence = hierarchical_predict(
            None, lgbm, lstm, lstm_scaler, X, feat_cols, [], df,
            model_dir=RUN_DIR, lstm_feat_cols=lstm_feats, lgbm_proba=oof_proba,
            per_bar_thr_long=thr_l, per_bar_thr_short=thr_s,
        )
        below = has_oof & (y_pred != 1) & (confidence < prod["conf_entry"])
        y_pred[below] = 1
        y_pred[~has_oof] = 1
        modal_arr = compute_dynamic_modal(p0, p2, hmm_enc, y_pred, MODAL_PER_TRADE, DYN, thr_l, thr_s)
        flow = df["flow_momentum_3bar"].ffill().fillna(0).values if "flow_momentum_3bar" in df.columns else np.zeros(n)
        X_gd = compute_guardian_static_array(df, static)
        rep = full_trading_report(
            y_pred=y_pred, y_actual=df["label"].map(LABEL_MAP).values.astype(np.int64),
            atr=df["atr_14_h1"].values if "atr_14_h1" in df.columns else np.ones(n),
            close=df["close"].values, high=df["high"].values, low=df["low"].values,
            h4_swing_highs=df.get("h4_swing_high"), h4_swing_lows=df.get("h4_swing_low"),
            index=df.index, modal=MODAL_PER_TRADE, leverages=LEVERAGE_SIM,
            fee_per_side=FEE_PER_SIDE, slippage=SLIPPAGE_PER_SIDE,
            min_rr=SWING_LABEL_MIN_RR, min_tp_atr=SWING_LABEL_MIN_TP, max_sl_atr=SWING_LABEL_MAX_SL,
            tp_fallback_atr=TP_SL_FALLBACK_TP, sl_fallback_atr=TP_SL_FALLBACK_SL,
            max_hold=MAX_HOLDING_BARS, confidence=confidence, symbol=sym,
            guardian_model=gdn["model"], guardian_scaler=gdn["scaler"],
            X_guardian=X_gd, guardian_exit_threshold=0.65,
            guardian_min_hold_bars=GUARDIAN_MIN_HOLD_BARS,
            guardian_feat_cols=gdn["feats"], guardian_static_names=static,
            flow_momentum_arr=flow, modal_arr=modal_arr,
            trailing_stop_enabled=TRAILING_STOP_ENABLED,
            trailing_stop_atr=TRAILING_STOP_ATR, trailing_stop_min_bars=TRAILING_STOP_MIN_BARS,
            h4_trend=df["h4_trend"].values if "h4_trend" in df.columns else None,
            vol_ratio=df["vol_ratio_20"].values if "vol_ratio_20" in df.columns else None,
        )
        all_trades.extend(rep.get("lev5x", rep).get("trades", []))

    def bucket(label, pred):
        sub = [t for t in all_trades if pred(t)]
        if not sub:
            return {"n": 0}
        modals = [t.get("modal_used", MODAL_PER_TRADE) for t in sub]
        pnls = [t["net_pnl"] for t in sub]
        return {
            "n": len(sub),
            "avg_modal": round(float(np.mean(modals)), 2),
            "median_modal": round(float(np.median(modals)), 2),
            "avg_pnl": round(float(np.mean(pnls)), 4),
            "pct_of_trades": round(len(sub) / len(all_trades) * 100, 1),
        }

    gdn_out = ("GUARDIAN_EXIT", "GUARDIAN_FULL", "GUARDIAN_MOMENTUM_EXIT",
               "GUARDIAN_MOMENTUM_PARTIAL", "GUARDIAN_DELTA_EXIT", "TRAILING_STOP")
    wins = [t for t in all_trades if t["net_pnl"] > 0]
    losses = [t for t in all_trades if t["net_pnl"] <= 0]
    sl_losses = [t for t in all_trades if t.get("outcome") == "LOSS"]
    gdn_losses = [t for t in all_trades if t.get("outcome") in gdn_out and t["net_pnl"] <= 0]
    gdn_wins = [t for t in all_trades if t.get("outcome") in gdn_out and t["net_pnl"] > 0]

    out = {
        "total_trades": len(all_trades),
        "all_avg_modal": round(float(np.mean([t.get("modal_used", 10) for t in all_trades])), 2),
        "profit_net_pnl_gt_0": bucket("win", lambda t: t["net_pnl"] > 0),
        "loss_net_pnl_le_0": bucket("loss", lambda t: t["net_pnl"] <= 0),
        "sl_hit_LOSS": bucket("sl", lambda t: t.get("outcome") == "LOSS"),
        "guardian_exit_profit": bucket("gdn_win", lambda t: t.get("outcome") in gdn_out and t["net_pnl"] > 0),
        "guardian_exit_loss": bucket("gdn_loss", lambda t: t.get("outcome") in gdn_out and t["net_pnl"] <= 0),
        "delta_win_vs_loss_modal": round(
            float(np.mean([t.get("modal_used", 10) for t in wins]))
            - float(np.mean([t.get("modal_used", 10) for t in losses])), 2
        ) if wins and losses else None,
        "delta_win_vs_sl_modal": round(
            float(np.mean([t.get("modal_used", 10) for t in wins]))
            - float(np.mean([t.get("modal_used", 10) for t in sl_losses])), 2
        ) if wins and sl_losses else None,
    }
    path = RUN_DIR / "dynsize_modal_by_outcome_oof.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))
    print(f"Saved {path}")


if __name__ == "__main__":
    main()