"""Forensics: SL early-entry hypothesis + stack-level PnL (no_pyr vs pyr2 variants)."""
import json
import sys
import warnings
from collections import defaultdict
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
    ALL_COINS, LABEL_DIR, MODEL_DIR, LABEL_MAP, TRAIN_CUTOFF_DATE,
    MODAL_PER_TRADE, LEVERAGE_SIM, FEE_PER_SIDE, SLIPPAGE_PER_SIDE,
    MAX_HOLDING_BARS, GUARDIAN_DYNAMIC_FEATURES, GUARDIAN_MIN_HOLD_BARS,
    SWING_LABEL_MIN_RR, SWING_LABEL_MIN_TP, SWING_LABEL_MAX_SL,
    TP_SL_FALLBACK_TP, TP_SL_FALLBACK_SL,
    TRAILING_STOP_ENABLED, TRAILING_STOP_ATR, TRAILING_STOP_MIN_BARS,
)

RUN_DIR = MODEL_DIR / "runs" / "ic32_regime_v1"
OOF_PATH = RUN_DIR / "oof_predictions.parquet"
OUT_JSON = RUN_DIR / "pyramiding_forensics_oof.json"
FLOW_MOM_WINDOW = 3
DYN_EXTRA = {"cvd_slope_h4_delta_entry", "ofi_h4_delta_entry", "flow_momentum_3bar"}
GDN_OUT = ("GUARDIAN_EXIT", "GUARDIAN_FULL", "GUARDIAN_MOMENTUM_EXIT",
           "GUARDIAN_MOMENTUM_PARTIAL", "GUARDIAN_DELTA_EXIT", "TRAILING_STOP")

VARIANTS = [
    ("no_pyr", False, 1, "independent"),
    ("pyr2_independent", True, 2, "independent"),
    ("pyr2_shared_sl", True, 2, "shared_sl_first"),
]


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
    return df[df["label"].astype(str).isin(LABEL_MAP)].copy()


def _run_coin(sym, oof_sym, feat_cols, lgbm, lstm, lstm_scaler, lstm_feats,
              hmm_cfg, live_cfg, gdn, enabled, max_p, exit_mode):
    df = _prep(sym)
    if df is None or len(df) < 50:
        return []
    n = len(df)
    merged = df.join(oof_sym[["p0", "p1", "p2", "has_oof"]], how="left")
    has_oof = merged["has_oof"].fillna(False).values.astype(bool)
    if has_oof.sum() < 30:
        return []
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
    X_gd = compute_guardian_static_array(df, gdn["static"])
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
        guardian_feat_cols=gdn["feats"], guardian_static_names=gdn["static"],
        flow_momentum_arr=flow, sl_trigger_mode="close",
        pyramiding_enabled=enabled, pyramiding_max_per_coin=max_p,
        pyramiding_same_dir=True, pyramiding_exit_mode=exit_mode,
    )
    trades = rep.get("lev5x", rep).get("trades", [])
    for t in trades:
        t["symbol"] = sym
    return trades


def _build_stacks(trades):
    by_sym = defaultdict(list)
    for t in trades:
        by_sym[t["symbol"]].append(t)
    stacks = []
    for sym, ts in by_sym.items():
        ts = sorted(ts, key=lambda x: x["bar_in"])
        i = 0
        while i < len(ts):
            stack = [ts[i]]
            j = i + 1
            while j < len(ts) and ts[j]["bar_in"] < stack[-1]["bar_out"]:
                stack.append(ts[j])
                j += 1
            stacks.append({"symbol": sym, "legs": stack, "n": len(stack)})
            i = j if j > i + 1 else i + 1
    return stacks


def _analyze(label, trades):
    n = len(trades)
    if not n:
        return {"label": label, "trades": 0}
    sl_n = sum(1 for t in trades if t.get("outcome") == "LOSS")
    wins = [t for t in trades if t["net_pnl"] > 0]
    losses = [t for t in trades if t["net_pnl"] <= 0]
    gpnl = sum(t["net_pnl"] for t in wins)
    gloss = abs(sum(t["net_pnl"] for t in losses))
    stacks = _build_stacks(trades)
    multi = [s for s in stacks if s["n"] > 1]

    leg1_sl_leg2_profit = 0
    leg1_sl_leg2_loss = 0
    stack_pnl_multi = []
    stack_pnl_single = []
    addon_legs = []
    leg1_only = []

    for s in stacks:
        legs = s["legs"]
        spnl = sum(t["net_pnl"] for t in legs)
        if len(legs) == 1:
            stack_pnl_single.append(spnl)
            leg1_only.append(legs[0])
        else:
            stack_pnl_multi.append(spnl)
            addon_legs.extend(legs[1:])
            l1, l2 = legs[0], legs[1]
            if l1.get("outcome") == "LOSS":
                if l2["net_pnl"] > 0:
                    leg1_sl_leg2_profit += 1
                else:
                    leg1_sl_leg2_loss += 1

    def leg_stats(arr, name):
        if not arr:
            return {"n": 0}
        sl = sum(1 for t in arr if t.get("outcome") == "LOSS")
        return {
            "n": len(arr),
            "sl_rate_pct": round(sl / len(arr) * 100, 1),
            "avg_pnl": round(float(np.mean([t["net_pnl"] for t in arr])), 4),
            "avg_hold": round(float(np.mean([t["bar_out"] - t["bar_in"] for t in arr])), 1),
        }

    return {
        "label": label,
        "trades": n,
        "stacks": len(stacks),
        "multi_leg_stacks": len(multi),
        "sl_rate_pct": round(sl_n / n * 100, 1),
        "wr_pct": round(len(wins) / n * 100, 1),
        "pf": round(gpnl / gloss, 3) if gloss else None,
        "total_pnl": round(sum(t["net_pnl"] for t in trades), 2),
        "ppt": round(sum(t["net_pnl"] for t in trades) / n, 4),
        "stack_pnl_single_avg": round(float(np.mean(stack_pnl_single)), 4) if stack_pnl_single else 0,
        "stack_pnl_multi_avg": round(float(np.mean(stack_pnl_multi)), 4) if stack_pnl_multi else 0,
        "leg1_stats": leg_stats(leg1_only + [s["legs"][0] for s in multi], "leg1"),
        "addon_leg_stats": leg_stats(addon_legs, "addon"),
        "leg1_sl_leg2_profit": leg1_sl_leg2_profit,
        "leg1_sl_leg2_loss": leg1_sl_leg2_loss,
        "early_sl_rescued_by_addon": leg1_sl_leg2_profit,
    }


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

    out = {}
    for label, en, mx, em in VARIANTS:
        print(f"Running {label}...")
        trades = []
        for sym in ALL_COINS:
            oof_sym = oof_all[oof_all["coin"] == sym]
            trades.extend(_run_coin(sym, oof_sym, feat_cols, lgbm, lstm, lstm_scaler,
                                    lstm_feats, hmm_cfg, live_cfg, gdn, en, mx, em))
        out[label] = _analyze(label, trades)
        print(json.dumps(out[label], indent=2))

    base = out["no_pyr"]
    for k, v in out.items():
        if k != "no_pyr" and base.get("ppt"):
            v["delta_ppt"] = round(v["ppt"] - base["ppt"], 4)
            v["delta_pnl_pct"] = round((v["total_pnl"] - base["total_pnl"]) / base["total_pnl"] * 100, 1)
            v["delta_sl_rate"] = round(v["sl_rate_pct"] - base["sl_rate_pct"], 1)

    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved {OUT_JSON}")


if __name__ == "__main__":
    main()