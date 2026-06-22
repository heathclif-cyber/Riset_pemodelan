"""Forensics v2: modal by outcome + quartile SL rate (OOF + holdout, cm_0.60)."""
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

from pipeline import ic32_fusion_shared as ifs
import pipeline.backtest_utils as btu
from pipeline.backtest_utils import hierarchical_predict, compute_guardian_static_array
from core.evaluator import full_trading_report
from core.models import load_lstm
from core.utils import ensure_utc_index
from pipeline.ic32_fusion_shared import build_per_bar_thresholds, load_b_dir_hmm_cfg
from pipeline.lstm_fusion_shared import compute_dynamic_modal
from config import (
    ALL_COINS, LABEL_DIR, HOLDOUT_DIR, MODEL_DIR, LABEL_MAP, TRAIN_CUTOFF_DATE,
    MODAL_PER_TRADE, LEVERAGE_SIM, FEE_PER_SIDE, SLIPPAGE_PER_SIDE,
    MAX_HOLDING_BARS, GUARDIAN_DYNAMIC_FEATURES, GUARDIAN_MIN_HOLD_BARS,
    SWING_LABEL_MIN_RR, SWING_LABEL_MIN_TP, SWING_LABEL_MAX_SL,
    TP_SL_FALLBACK_TP, TP_SL_FALLBACK_SL,
    TRAILING_STOP_ENABLED, TRAILING_STOP_ATR, TRAILING_STOP_MIN_BARS,
)

RUN_DIR = MODEL_DIR / "runs" / "ic32_regime_v1"
OOF_PATH = RUN_DIR / "oof_predictions.parquet"
HOLDOUT_LABEL_DIR = HOLDOUT_DIR / "labeled"
DYN = {
    "conf_window": 0.10, "conf_max_mult": 0.6,
    "regime_mult": {0: 0.75, 1: 1.0, 2: 1.0, "3_long": 1.5, "3_short": 0.75, -1: 0.80},
    "clamp_min": 0.5, "clamp_max": 2.0,
}
DYN_EXTRA = {"cvd_slope_h4_delta_entry", "ofi_h4_delta_entry", "flow_momentum_3bar"}
FLOW_MOM_WINDOW = 3
GDN_OUT = ("GUARDIAN_EXIT", "GUARDIAN_FULL", "GUARDIAN_MOMENTUM_EXIT",
           "GUARDIAN_MOMENTUM_PARTIAL", "GUARDIAN_DELTA_EXIT", "TRAILING_STOP")


def _apply():
    prod = ifs.load_production_defaults()
    import config as pc
    for m in (pc, btu):
        m.CONFIDENCE_THRESHOLD_ENTRY = prod["conf_entry"]
        m.LSTM_ADJUST_AGREE_BOOST = prod["agree_boost"]
        m.LSTM_ADJUST_NEUTRAL_PEN = prod["neutral_pen"]
        m.LSTM_ADJUST_OPPOSITE_PEN = prod["opposite_pen"]
        m.LSTM_DIRECTIONAL_REVIEW_THRESHOLD = prod["dir_review_thr"]
        m.LSTM_FLAT_REVIEW_ENABLED = prod["flat_review"]
        m.LSTM_CONFIRMATION_ENABLED = True
        m.REGIME_AWARE_ALIGNMENT = prod["flip"]
        m.HMM_GATE_LSTM_ENABLED = prod["hmm_gate_lstm"]
    btu.SMART_ENTRY_MODE = "disabled"


def _prep(sym, holdout=False):
    base = HOLDOUT_LABEL_DIR if holdout else LABEL_DIR
    p = base / f"{sym}_features_v3.parquet"
    if not p.exists():
        return None
    df = pd.read_parquet(p)
    df = ensure_utc_index(df).sort_index()
    if not holdout:
        df = df[df.index < TRAIN_CUTOFF_DATE]
    rp = (HOLDOUT_LABEL_DIR if holdout else LABEL_DIR) / f"{sym}_regime_h1.parquet"
    if not rp.exists():
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


def _stats(trades, pred):
    sub = [t for t in trades if pred(t)]
    if not sub:
        return {"n": 0}
    modals = np.array([t.get("modal_used", MODAL_PER_TRADE) for t in sub])
    pnls = np.array([t["net_pnl"] for t in sub])
    return {
        "n": len(sub),
        "pct": round(len(sub) / len(trades) * 100, 1) if trades else 0,
        "avg_modal": round(float(modals.mean()), 2),
        "median_modal": round(float(np.median(modals)), 2),
        "p25_modal": round(float(np.percentile(modals, 25)), 2),
        "p75_modal": round(float(np.percentile(modals, 75)), 2),
        "total_pnl": round(float(pnls.sum()), 2),
        "avg_pnl": round(float(pnls.mean()), 4),
    }


def _quartile_sl_rate(trades):
    modals = np.array([t.get("modal_used", MODAL_PER_TRADE) for t in trades])
    q25, q50, q75 = np.percentile(modals, [25, 50, 75])
    buckets = [
        ("Q1_low", lambda m: m <= q25),
        ("Q2", lambda m: (m > q25) & (m <= q50)),
        ("Q3", lambda m: (m > q50) & (m <= q75)),
        ("Q4_high", lambda m: m > q75),
    ]
    out = {}
    for name, fn in buckets:
        idx = fn(modals)
        n = int(idx.sum())
        if n == 0:
            continue
        sl_n = sum(1 for t, i in zip(trades, idx) if i and t.get("outcome") == "LOSS")
        win_n = sum(1 for t, i in zip(trades, idx) if i and t["net_pnl"] > 0)
        out[name] = {
            "n": n,
            "avg_modal": round(float(modals[idx].mean()), 2),
            "sl_rate_pct": round(sl_n / n * 100, 1),
            "win_rate_pct": round(win_n / n * 100, 1),
            "avg_pnl": round(float(np.mean([t["net_pnl"] for t, i in zip(trades, idx) if i])), 4),
        }
    return out


def _run_trades(holdout=False):
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
    oof_all = pd.read_parquet(OOF_PATH) if not holdout else None
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
        df = _prep(sym, holdout=holdout)
        if df is None:
            continue
        n = len(df)
        if holdout:
            X = np.zeros((n, len(feat_cols)))
            for i, col in enumerate(feat_cols):
                if col in df.columns:
                    X[:, i] = df[col].ffill().fillna(0).values
            hmm_enc = df["hmm_regime_enc"].values.astype(np.int32)
            thr_l, thr_s = build_per_bar_thresholds(hmm_enc, hmm_cfg)
            y_pred, confidence = hierarchical_predict(
                None, lgbm, lstm, lstm_scaler, X, feat_cols, [], df,
                model_dir=RUN_DIR, lstm_feat_cols=lstm_feats,
                per_bar_thr_long=thr_l, per_bar_thr_short=thr_s,
            )
            p0 = np.zeros(n, dtype=np.float32)
            p2 = np.zeros(n, dtype=np.float32)
            for i, row in enumerate(y_pred):
                if row == 2:
                    p2[i] = confidence[i]
                elif row == 0:
                    p0[i] = confidence[i]
        else:
            oof_sym = oof_all[oof_all["coin"] == sym]
            merged = df.join(oof_sym[["p0", "p1", "p2", "has_oof"]], how="left")
            has_oof = merged["has_oof"].fillna(False).values.astype(bool)
            if has_oof.sum() < 30:
                continue
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
    return all_trades


def _analyze(trades, label):
    wins = [t for t in trades if t["net_pnl"] > 0]
    losses = [t for t in trades if t["net_pnl"] <= 0]
    sl = [t for t in trades if t.get("outcome") == "LOSS"]
    gdn_w = [t for t in trades if t.get("outcome") in GDN_OUT and t["net_pnl"] > 0]
    gdn_l = [t for t in trades if t.get("outcome") in GDN_OUT and t["net_pnl"] <= 0]

    avg_win_m = float(np.mean([t.get("modal_used", 10) for t in wins])) if wins else 0
    avg_sl_m = float(np.mean([t.get("modal_used", 10) for t in sl])) if sl else 0
    avg_loss_m = float(np.mean([t.get("modal_used", 10) for t in losses])) if losses else 0

    total_sl_loss = sum(t["net_pnl"] for t in sl)
    total_win_pnl = sum(t["net_pnl"] for t in wins)

    return {
        "period": label,
        "total_trades": len(trades),
        "all_avg_modal": round(float(np.mean([t.get("modal_used", 10) for t in trades])), 2),
        "profit": _stats(trades, lambda t: t["net_pnl"] > 0),
        "loss": _stats(trades, lambda t: t["net_pnl"] <= 0),
        "sl_hit": _stats(trades, lambda t: t.get("outcome") == "LOSS"),
        "guardian_profit": _stats(trades, lambda t: t.get("outcome") in GDN_OUT and t["net_pnl"] > 0),
        "guardian_loss": _stats(trades, lambda t: t.get("outcome") in GDN_OUT and t["net_pnl"] <= 0),
        "delta_profit_vs_sl_modal": round(avg_win_m - avg_sl_m, 2),
        "delta_profit_vs_loss_modal": round(avg_win_m - avg_loss_m, 2),
        "total_sl_dollar_loss": round(total_sl_loss, 2),
        "total_profit_dollar": round(total_win_pnl, 2),
        "sl_loss_per_dollar_modal": round(total_sl_loss / (avg_sl_m * len(sl)), 4) if sl else None,
        "profit_per_dollar_modal": round(total_win_pnl / (avg_win_m * len(wins)), 4) if wins else None,
        "quartile_breakdown": _quartile_sl_rate(trades),
    }


def main():
    print("Running OOF forensics...")
    oof_trades = _run_trades(holdout=False)
    print("Running holdout forensics...")
    ho_trades = _run_trades(holdout=True)

    out = {
        "dynsize_config": DYN,
        "base_modal": MODAL_PER_TRADE,
        "live_modal_equiv": {
            "min": MODAL_PER_TRADE * DYN["clamp_min"],
            "max": MODAL_PER_TRADE * DYN["clamp_max"],
            "note": "Live $5 -> range $2.50-$10",
        },
        "oof": _analyze(oof_trades, "OOF_train"),
        "holdout": _analyze(ho_trades, "holdout_apr_jun26"),
    }

    path = RUN_DIR / "dynsize_modal_forensics_v2.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))
    print(f"Saved {path}")


if __name__ == "__main__":
    main()