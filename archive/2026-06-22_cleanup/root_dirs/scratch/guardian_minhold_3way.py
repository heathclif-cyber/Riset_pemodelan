"""OOF + holdout: no_guardian vs min_hold {2,4,5} on ic32 B-dir stack."""
import json
import sys
import warnings
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pipeline.backtest_utils as btu
from pipeline.backtest_utils import hierarchical_predict, compute_guardian_static_array
from core.evaluator import full_trading_report
from core.models import load_lstm
from core.utils import ensure_utc_index
from pipeline.ic32_fusion_shared import build_per_bar_thresholds, load_b_dir_hmm_cfg
from pipeline import ic32_fusion_shared as ifs
from config import (
    ALL_COINS, LABEL_DIR, HOLDOUT_DIR, MODEL_DIR, LABEL_MAP, TRAIN_CUTOFF_DATE,
    MODAL_PER_TRADE, LEVERAGE_SIM, FEE_PER_SIDE, SLIPPAGE_PER_SIDE,
    MAX_HOLDING_BARS, GUARDIAN_DYNAMIC_FEATURES,
    SWING_LABEL_MIN_RR, SWING_LABEL_MIN_TP, SWING_LABEL_MAX_SL,
    TP_SL_FALLBACK_TP, TP_SL_FALLBACK_SL,
    TRAILING_STOP_ENABLED, TRAILING_STOP_ATR, TRAILING_STOP_MIN_BARS,
)

RUN_DIR = MODEL_DIR / "runs" / "ic32_regime_v1"
OOF_PATH = RUN_DIR / "oof_predictions.parquet"
HOLDOUT_LABEL_DIR = HOLDOUT_DIR / "labeled"
OUT_JSON = RUN_DIR / "guardian_minhold_3way_compare.json"
FLOW_MOM_WINDOW = 3
DYN_EXTRA = {"cvd_slope_h4_delta_entry", "ofi_h4_delta_entry", "flow_momentum_3bar"}
VARIANTS = [
    ("no_guardian", None, False),
    ("min_hold_2", 2, True),
    ("min_hold_4", 4, True),
    ("min_hold_5", 5, True),
]
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
    return prod


def _prep(sym, holdout=False):
    base = HOLDOUT_LABEL_DIR if holdout else LABEL_DIR
    p = base / f"{sym}_features_v3.parquet"
    if not p.exists():
        return None
    df = pd.read_parquet(p)
    df = ensure_utc_index(df).sort_index()
    if not holdout:
        df = df[df.index < TRAIN_CUTOFF_DATE]
    rp = HOLDOUT_LABEL_DIR / f"{sym}_regime_h1.parquet"
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


def _agg(trades):
    if not trades:
        return {"trades": 0}
    n = len(trades)
    wins = [t for t in trades if t.get("net_pnl", 0) > 0]
    losses = [t for t in trades if t.get("net_pnl", 0) <= 0]
    gpnl = sum(t["net_pnl"] for t in wins)
    gloss = abs(sum(t["net_pnl"] for t in losses))
    holds = [t.get("bar_out", 0) - t.get("bar_in", 0) for t in trades]
    early_gdn_los = sum(
        1 for t in losses
        if t.get("outcome") in GDN_OUT and (t.get("bar_out", 0) - t.get("bar_in", 0)) <= 3
    )
    sl_n = sum(1 for t in trades if t.get("outcome") == "LOSS")
    gdn_n = sum(1 for t in trades if t.get("outcome") in GDN_OUT)
    to_n = sum(1 for t in trades if "TIMEOUT" in t.get("outcome", ""))
    return {
        "trades": n,
        "wr": round(len(wins) / n * 100, 2),
        "pf": round(gpnl / gloss, 3) if gloss > 0 else None,
        "pnl": round(sum(t.get("net_pnl", 0) for t in trades), 2),
        "ppt": round(sum(t.get("net_pnl", 0) for t in trades) / n, 4),
        "avg_hold_bars": round(float(np.mean(holds)), 2),
        "median_hold_bars": round(float(np.median(holds)), 1),
        "sl_rate_pct": round(sl_n / n * 100, 1),
        "guardian_exit_pct": round(gdn_n / n * 100, 1),
        "timeout_pct": round(to_n / n * 100, 1),
        "early_gdn_losers": early_gdn_los,
    }


def _run(holdout, label, min_hold, use_guardian, hmm_cfg, live_cfg, gdn, oof_all,
         feat_cols, lgbm, lstm, lstm_scaler, lstm_feats):
    trades = []
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
        else:
            oof_sym = oof_all[oof_all["coin"] == sym]
            merged = df.join(oof_sym[["p0", "p1", "p2", "has_oof"]], how="left")
            has_oof = merged["has_oof"].fillna(False).values.astype(bool)
            if has_oof.sum() < 30:
                continue
            oof_proba = np.column_stack([merged["p0"].values, merged["p1"].values, merged["p2"].values])
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
            below = has_oof & (y_pred != 1) & (confidence < live_cfg["conf_entry"])
            y_pred[below] = 1
            y_pred[~has_oof] = 1

        flow = df["flow_momentum_3bar"].ffill().fillna(0).values if "flow_momentum_3bar" in df.columns else np.zeros(n)
        X_gd = compute_guardian_static_array(df, gdn["static"]) if use_guardian else None
        kw = dict(
            y_pred=y_pred, y_actual=df["label"].map(LABEL_MAP).values.astype(np.int64),
            atr=df["atr_14_h1"].values if "atr_14_h1" in df.columns else np.ones(n),
            close=df["close"].values, high=df["high"].values, low=df["low"].values,
            h4_swing_highs=df.get("h4_swing_high"), h4_swing_lows=df.get("h4_swing_low"),
            index=df.index, modal=MODAL_PER_TRADE, leverages=LEVERAGE_SIM,
            fee_per_side=FEE_PER_SIDE, slippage=SLIPPAGE_PER_SIDE,
            min_rr=SWING_LABEL_MIN_RR, min_tp_atr=SWING_LABEL_MIN_TP, max_sl_atr=SWING_LABEL_MAX_SL,
            tp_fallback_atr=TP_SL_FALLBACK_TP, sl_fallback_atr=TP_SL_FALLBACK_SL,
            max_hold=MAX_HOLDING_BARS, confidence=confidence, symbol=sym,
            guardian_enabled=use_guardian,
            trailing_stop_enabled=TRAILING_STOP_ENABLED,
            trailing_stop_atr=TRAILING_STOP_ATR, trailing_stop_min_bars=TRAILING_STOP_MIN_BARS,
            h4_trend=df["h4_trend"].values if "h4_trend" in df.columns else None,
            vol_ratio=df["vol_ratio_20"].values if "vol_ratio_20" in df.columns else None,
            sl_trigger_mode="close",
        )
        if use_guardian:
            kw.update(
                guardian_model=gdn["model"], guardian_scaler=gdn["scaler"],
                X_guardian=X_gd, guardian_exit_threshold=0.65,
                guardian_min_hold_bars=min_hold,
                guardian_feat_cols=gdn["feats"], guardian_static_names=gdn["static"],
                flow_momentum_arr=flow,
            )
        rep = full_trading_report(**kw)
        trades.extend(rep.get("lev5x", rep).get("trades", []))
    return _agg(trades)


def main():
    live_cfg = _apply()
    hmm_cfg = load_b_dir_hmm_cfg()
    run = MODEL_DIR / "runs" / "ic32_guardian_continuation_v1"
    with open(run / "guardian_feature_cols.json") as f:
        feats = json.load(f)
    dyn = set(GUARDIAN_DYNAMIC_FEATURES) | DYN_EXTRA
    gdn = {
        "model": joblib.load(run / "guardian.pkl"),
        "scaler": joblib.load(run / "guardian_scaler.pkl"),
        "feats": feats, "static": [c for c in feats if c not in dyn],
    }
    oof_all = pd.read_parquet(OOF_PATH)
    with open(MODEL_DIR / "feature_cols_ic32_regime.json") as f:
        feat_cols = json.load(f)
    with open(MODEL_DIR / "feature_cols_lstm_temporal.json") as f:
        lstm_feats = json.load(f)[:11]
    lgbm = joblib.load(RUN_DIR / "lgbm.pkl")
    lstm = load_lstm(MODEL_DIR / "lstm_best.pt", device="cpu")
    lstm_scaler = joblib.load(MODEL_DIR / "lstm_scaler.pkl")

    out = {
        "stack": "ic32 B-dir + hard_consensus + continuation_v1",
        "guardian_exit_threshold": 0.65,
        "created": datetime.now().isoformat(),
        "oof": {},
        "holdout": {},
    }

    for period, holdout in [("oof", False), ("holdout", True)]:
        print(f"\n=== {period.upper()} ===")
        for label, mh, use_gdn in VARIANTS:
            print(f"  running {label}...")
            sc = _run(holdout, label, mh, use_gdn, hmm_cfg, live_cfg, gdn, oof_all,
                      feat_cols, lgbm, lstm, lstm_scaler, lstm_feats)
            out[period][label] = sc
            print(f"    {sc}")

    base = out["oof"]["min_hold_2"]
    for period in ("oof", "holdout"):
        for label, sc in out[period].items():
            if base.get("ppt"):
                sc["delta_ppt_vs_min_hold_2"] = round(sc["ppt"] - base["ppt"], 4)

    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved {OUT_JSON}")


if __name__ == "__main__":
    main()