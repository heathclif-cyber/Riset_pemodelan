"""
Compare Guardian clean_v2 vs ic32_guardian_continuation_v1 on full-stack OOF ic32.
"""
import json
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from pipeline.backtest_utils import hierarchical_predict, compute_guardian_static_array
from core.evaluator import full_trading_report
from core.models import load_lstm
from core.utils import setup_logger, ensure_utc_index
from config import (
    ALL_COINS, LABEL_DIR, MODEL_DIR, LABEL_MAP, TRAIN_CUTOFF_DATE,
    MODAL_PER_TRADE, LEVERAGE_SIM, FEE_PER_SIDE, SLIPPAGE_PER_SIDE,
    MAX_HOLDING_BARS, GUARDIAN_DYNAMIC_FEATURES,
    SWING_LABEL_MIN_RR, SWING_LABEL_MIN_TP, SWING_LABEL_MAX_SL,
    TP_SL_FALLBACK_TP, TP_SL_FALLBACK_SL,
    TRAILING_STOP_ENABLED, TRAILING_STOP_ATR, TRAILING_STOP_MIN_BARS,
)

logger = setup_logger("08b_guardian_compare")
RUN_DIR = MODEL_DIR / "runs" / "ic32_regime_v1"
OOF_PATH = RUN_DIR / "oof_predictions.parquet"
FLOW_MOM_WINDOW = 3


def _prep_df(sym):
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
    mask = df["label"].astype(str).isin(LABEL_MAP)
    df = df[mask].copy()
    if len(df) < 50:
        return None
    if "ofi_z_score" in df.columns:
        df["flow_momentum_3bar"] = df["ofi_z_score"].rolling(FLOW_MOM_WINDOW, min_periods=1).mean()
    else:
        df["flow_momentum_3bar"] = 0.0
    return df


def _signals(sym, df, oof_all, feat_cols, lgbm, lstm, lstm_scaler, lstm_feats, live_cfg):
    n = len(df)
    oof_sym = oof_all[oof_all["coin"] == sym]
    merged = df.join(oof_sym[["p0", "p1", "p2", "has_oof"]], how="left")
    has_oof = merged["has_oof"].fillna(False).values.astype(bool)
    if has_oof.sum() < 30:
        return None
    oof_proba = np.column_stack([merged["p0"].values, merged["p1"].values, merged["p2"].values])
    X = np.zeros((n, len(feat_cols)), dtype=np.float64)
    for idx, col in enumerate(feat_cols):
        if col in df.columns:
            X[:, idx] = df[col].ffill().fillna(0).values.astype(np.float64)
    y_pred, confidence = hierarchical_predict(
        None, lgbm, lstm, lstm_scaler, X, feat_cols, [], df,
        model_dir=RUN_DIR, lstm_feat_cols=lstm_feats, lgbm_proba=oof_proba,
    )
    below = has_oof & (y_pred != 1) & (confidence < live_cfg["conf_entry"])
    y_pred[below] = 1
    y_pred[~has_oof] = 1
    return y_pred, confidence, df


def _agg(trades):
    if not trades:
        return {"trades": 0, "wr": 0, "pf": 0, "pnl": 0, "mom_exit": 0, "mom_avg_bars": 0}
    wins = [t for t in trades if t.get("net_pnl", 0) > 0]
    losses = [t for t in trades if t.get("net_pnl", 0) <= 0]
    gp = sum(t["net_pnl"] for t in wins)
    gl = abs(sum(t["net_pnl"] for t in losses))
    mom = [t for t in trades if t.get("outcome") in ("GUARDIAN_MOMENTUM_EXIT", "GUARDIAN_MOMENTUM_PARTIAL")]
    mom_bars = [t.get("bar_out", 0) - t.get("bar_in", 0) for t in mom]
    return {
        "trades": len(trades),
        "wr": len(wins) / len(trades) * 100,
        "pf": gp / gl if gl > 0 else float("inf"),
        "pnl": sum(t.get("net_pnl", 0) for t in trades),
        "mom_exit": len(mom),
        "mom_avg_bars": float(np.mean(mom_bars)) if mom_bars else 0,
    }


def _run_guardian(df, y_pred, confidence, gdn, use_cont):
    n = len(df)
    y = df["label"].map(LABEL_MAP).values.astype(np.int64)
    close, high, low = df["close"].values, df["high"].values, df["low"].values
    atr = df["atr_14_h1"].values if "atr_14_h1" in df.columns else np.ones(n)
    h4_sh = df["h4_swing_high"].values if "h4_swing_high" in df.columns else None
    h4_sl = df["h4_swing_low"].values if "h4_swing_low" in df.columns else None
    h4t = df["h4_trend"].values if "h4_trend" in df.columns else None
    volr = df["vol_ratio_20"].values if "vol_ratio_20" in df.columns else None
    X_gd = compute_guardian_static_array(df, gdn["static"])
    kw = dict(
        y_pred=y_pred, y_actual=y, atr=atr, close=close, high=high, low=low,
        h4_swing_highs=h4_sh, h4_swing_lows=h4_sl, index=df.index,
        modal=MODAL_PER_TRADE, leverages=LEVERAGE_SIM, fee_per_side=FEE_PER_SIDE,
        slippage=SLIPPAGE_PER_SIDE, min_rr=SWING_LABEL_MIN_RR, min_tp_atr=SWING_LABEL_MIN_TP,
        max_sl_atr=SWING_LABEL_MAX_SL, tp_fallback_atr=TP_SL_FALLBACK_TP,
        sl_fallback_atr=TP_SL_FALLBACK_SL, max_hold=MAX_HOLDING_BARS,
        confidence=confidence, h4_trend=h4t, vol_ratio=volr,
        guardian_enabled=True, guardian_model=gdn["model"], guardian_scaler=gdn["scaler"],
        X_guardian=X_gd, guardian_exit_threshold=0.65, guardian_min_hold_bars=2,
        trailing_stop_enabled=TRAILING_STOP_ENABLED, trailing_stop_atr=TRAILING_STOP_ATR,
        trailing_stop_min_bars=TRAILING_STOP_MIN_BARS,
    )
    if use_cont:
        flow = df["flow_momentum_3bar"].ffill().fillna(0).values.astype(np.float64)
        kw.update(
            guardian_feat_cols=gdn["feats"],
            guardian_static_names=gdn["static"],
            flow_momentum_arr=flow,
        )
    rep = full_trading_report(**kw)
    return rep.get("lev5x", rep).get("trades", [])


def _load_gdn(run_name, fallback_files=None):
    d = MODEL_DIR / "runs" / run_name
    if not (d / "guardian.pkl").exists():
        return None
    with open(d / "guardian_feature_cols.json", encoding="utf-8") as f:
        feats = json.load(f)
    dyn = set(GUARDIAN_DYNAMIC_FEATURES) | {
        "cvd_slope_h4_delta_entry", "ofi_h4_delta_entry", "flow_momentum_3bar",
    }
    static = [c for c in feats if c not in dyn]
    return {
        "model": joblib.load(d / "guardian.pkl"),
        "scaler": joblib.load(d / "guardian_scaler.pkl"),
        "feats": feats,
        "static": static,
        "name": run_name,
    }


def main():
    # Reuse live config helper from 08
    sys.path.insert(0, str(ROOT / "pipeline"))
    import importlib.util
    spec = importlib.util.spec_from_file_location("oof08", ROOT / "pipeline" / "08_oof_ic32_full_stack.py")
    oof08 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(oof08)
    live_cfg = oof08._apply_live_config()

    oof_all = pd.read_parquet(OOF_PATH)
    with open(MODEL_DIR / "feature_cols_ic32_regime.json", encoding="utf-8") as f:
        feat_cols = json.load(f)
    with open(MODEL_DIR / "feature_cols_lstm_temporal.json", encoding="utf-8") as f:
        lstm_feats = json.load(f)
    lgbm = joblib.load(RUN_DIR / "lgbm.pkl")
    lstm = load_lstm(MODEL_DIR / "lstm_best.pt", device="cpu")
    lstm_scaler = joblib.load(MODEL_DIR / "lstm_scaler.pkl")

    g_clean = _load_gdn("ic32_guardian_clean_v2")
    if g_clean is None:
        g_clean = {
            "model": joblib.load(MODEL_DIR / "guardian_best.pkl"),
            "scaler": joblib.load(MODEL_DIR / "guardian_scaler.pkl"),
            "feats": json.load(open(MODEL_DIR / "guardian_feature_cols.json")),
            "static": [c for c in json.load(open(MODEL_DIR / "guardian_feature_cols.json"))
                       if c not in GUARDIAN_DYNAMIC_FEATURES],
            "name": "guardian_best",
        }
    g_cont = _load_gdn("ic32_guardian_continuation_v1")
    if g_cont is None:
        raise FileNotFoundError("Run 06_train_guardian_ic32_continuation_v1.py first")

    trades_clean, trades_cont = [], []
    for sym in ALL_COINS:
        df = _prep_df(sym)
        if df is None:
            continue
        sig = _signals(sym, df, oof_all, feat_cols, lgbm, lstm, lstm_scaler, lstm_feats, live_cfg)
        if sig is None:
            continue
        y_pred, confidence, df = sig
        trades_clean.extend(_run_guardian(df, y_pred, confidence, g_clean, False))
        trades_cont.extend(_run_guardian(df, y_pred, confidence, g_cont, True))

    a_clean = _agg(trades_clean)
    a_cont = _agg(trades_cont)
    print("\n=== OOF Full Stack Guardian Compare ===")
    print(f"{'variant':<25} {'trades':>8} {'WR%':>7} {'PF':>6} {'PnL':>10} {'mom_exit':>9} {'mom_bars':>9}")
    for name, a in [(g_clean["name"], a_clean), (g_cont["name"], a_cont)]:
        print(f"{name:<25} {a['trades']:>8} {a['wr']:>6.1f}% {a['pf']:>6.2f} ${a['pnl']:>+8.0f} "
              f"{a['mom_exit']:>9} {a['mom_avg_bars']:>9.1f}")

    out = MODEL_DIR / "runs" / "ic32_guardian_continuation_v1" / "oof_compare_vs_clean_v2.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"clean_v2": a_clean, "continuation_v1": a_cont}, f, indent=2)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()