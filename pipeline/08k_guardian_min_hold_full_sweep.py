"""
OOF + holdout diagnostic: guardian min_hold full sweep.

Stack frozen: B-dir + hard_consensus + continuation_v1 + SL close.
Holdout tambah scale_in (setup live). Keputusan dari OOF saja.

Usage:
  python pipeline/08k_guardian_min_hold_full_sweep.py
  python pipeline/08k_guardian_min_hold_full_sweep.py --oof-only
  python pipeline/08k_guardian_min_hold_full_sweep.py --holdout-only --mh 4 5
"""
from __future__ import annotations

import argparse
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
    ALL_COINS, LABEL_DIR, HOLDOUT_DIR, MODEL_DIR, LABEL_MAP, TRAIN_CUTOFF_DATE,
    MODAL_PER_TRADE, LEVERAGE_SIM, FEE_PER_SIDE, SLIPPAGE_PER_SIDE,
    MAX_HOLDING_BARS, GUARDIAN_DYNAMIC_FEATURES,
    SWING_LABEL_MIN_RR, SWING_LABEL_MIN_TP, SWING_LABEL_MAX_SL,
    TP_SL_FALLBACK_TP, TP_SL_FALLBACK_SL,
    TRAILING_STOP_ENABLED, TRAILING_STOP_ATR, TRAILING_STOP_MIN_BARS,
)

logger = setup_logger("08k_min_hold_sweep")
RUN_DIR = MODEL_DIR / "runs" / "ic32_regime_v1"
OOF_PATH = RUN_DIR / "oof_predictions.parquet"
INF_CFG = MODEL_DIR / "inference_config.json"
OUT_OOF = RUN_DIR / "guardian_min_hold_full_sweep_oof.json"
OUT_HOLD = RUN_DIR / "guardian_min_hold_full_sweep_holdout.json"
FLOW_MOM_WINDOW = 3
MIN_HOLD_GRID = (0, 1, 2, 3, 4, 5, 6)
DYN_EXTRA = {"cvd_slope_h4_delta_entry", "ofi_h4_delta_entry", "flow_momentum_3bar"}
SCALE_IN = {"enabled": True, "max_per_coin": 2, "exit_mode": "scale_in"}


def _apply_live_config() -> dict:
    prod = ifs.load_production_defaults()
    import config as project_config

    with open(INF_CFG, encoding="utf-8") as f:
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
    btu.SMART_ENTRY_MODE = "disabled"
    btu.MOMENTUM_DYNAMIC_THRESHOLD_ENABLED = False
    btu.TREND_DYNAMIC_THRESHOLD_ENABLED = False
    btu.LSTM_STANDALONE_ENABLED = False

    return {
        "conf_entry": prod["conf_entry"],
        "sl_trigger_mode": str(rr.get("sl_trigger_mode", "close")),
    }


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


def _add_momentum(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "cvd_slope_h4" in df.columns:
        df["cvd_slope_h4_delta"] = df["cvd_slope_h4"].diff(1)
    if "ofi_z_score" in df.columns:
        df["flow_momentum_3bar"] = df["ofi_z_score"].rolling(FLOW_MOM_WINDOW, min_periods=1).mean()
    else:
        df["flow_momentum_3bar"] = 0.0
    return df


def _prep_train(sym: str) -> pd.DataFrame | None:
    path = LABEL_DIR / f"{sym}_features_v3.parquet"
    if not path.exists():
        return None
    df = ensure_utc_index(pd.read_parquet(path)).sort_index()
    df = df[df.index < TRAIN_CUTOFF_DATE]
    df = _add_momentum(df)
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
    return df if len(df) >= 50 else None


def _prep_holdout(sym: str) -> pd.DataFrame | None:
    path = HOLDOUT_DIR / "labeled" / f"{sym}_features_v3.parquet"
    if not path.exists():
        return None
    df = ensure_utc_index(pd.read_parquet(path)).sort_index()
    df = _add_momentum(df)
    rp = HOLDOUT_DIR / "labeled" / f"{sym}_regime_h1.parquet"
    if rp.exists():
        try:
            reg = pd.read_parquet(rp)
            if "hmm_regime_enc" in df.columns:
                df = df.drop(columns=["hmm_regime_enc"])
            df = df.join(reg[["hmm_regime_enc"]], how="left")
            df["hmm_regime_enc"] = df["hmm_regime_enc"].fillna(1).astype("int32")
        except Exception:
            df["hmm_regime_enc"] = 1
    else:
        df["hmm_regime_enc"] = 1
    mask = df["label"].astype(str).isin(LABEL_MAP)
    df = df[mask].copy()
    return df if len(df) >= 30 else None


def _agg(trades: list) -> dict:
    if not trades:
        return {"trades": 0}
    n = len(trades)
    wins = [t for t in trades if t.get("net_pnl", 0) > 0]
    losses = [t for t in trades if t.get("net_pnl", 0) <= 0]
    gpnl = sum(t["net_pnl"] for t in wins)
    gloss = abs(sum(t["net_pnl"] for t in losses))
    hold = [t.get("bar_out", 0) - t.get("bar_in", 0) for t in trades]

    early_gdn_los = 0
    gdn_exit_n = gdn_exit_w = 0
    sl_n = 0
    gdn_los_n = 0
    for t in trades:
        oc = t.get("outcome", "")
        hb = t.get("bar_out", 0) - t.get("bar_in", 0)
        pnl = t.get("net_pnl", 0)
        if oc == "LOSS":
            sl_n += 1
        if oc == "GUARDIAN_EXIT":
            gdn_exit_n += 1
            if pnl > 0:
                gdn_exit_w += 1
        if "GUARDIAN" in oc and pnl <= 0:
            gdn_los_n += 1
            if hb <= 3:
                early_gdn_los += 1

    return {
        "trades": n,
        "wr": round(len(wins) / n * 100, 2),
        "pf": round(gpnl / gloss, 3) if gloss > 0 else None,
        "pnl": round(sum(t.get("net_pnl", 0) for t in trades), 2),
        "ppt": round(sum(t.get("net_pnl", 0) for t in trades) / n, 4),
        "sl_rate_pct": round(sl_n / n * 100, 2),
        "gdn_exit_wr_pct": round(gdn_exit_w / gdn_exit_n * 100, 2) if gdn_exit_n else None,
        "gdn_exit_n": gdn_exit_n,
        "gdn_losers": gdn_los_n,
        "early_gdn_losers": early_gdn_los,
        "avg_hold_bars": round(float(np.mean(hold)), 2),
        "median_hold_bars": round(float(np.median(hold)), 2),
    }


def _simulate(df: pd.DataFrame, sym: str, feat_cols, lstm_feats, lgbm, lstm, lstm_scaler,
              hmm_cfg, live_cfg, gdn, min_hold, oof_sym: pd.DataFrame | None,
              scale_in: bool) -> list:
    n = len(df)
    if oof_sym is not None:
        merged = df.join(oof_sym[["p0", "p1", "p2", "has_oof"]], how="left")
        has_oof = merged["has_oof"].fillna(False).values.astype(bool)
        if has_oof.sum() < 30:
            return []
        oof_proba = np.column_stack([merged["p0"].values, merged["p1"].values, merged["p2"].values])
        lgbm_proba = oof_proba
    else:
        has_oof = np.ones(n, dtype=bool)
        lgbm_proba = None

    X = np.zeros((n, len(feat_cols)), dtype=np.float64)
    for idx, col in enumerate(feat_cols):
        if col in df.columns:
            X[:, idx] = df[col].ffill().fillna(0).values.astype(np.float64)

    hmm_enc = df["hmm_regime_enc"].values.astype(np.int32)
    thr_l, thr_s = build_per_bar_thresholds(hmm_enc, hmm_cfg)
    y_pred, confidence = hierarchical_predict(
        None, lgbm, lstm, lstm_scaler, X, feat_cols, [], df,
        model_dir=RUN_DIR, lstm_feat_cols=lstm_feats,
        lgbm_proba=lgbm_proba,
        per_bar_thr_long=thr_l, per_bar_thr_short=thr_s,
    )
    if oof_sym is not None:
        below = has_oof & (y_pred != 1) & (confidence < live_cfg["conf_entry"])
        y_pred[below] = 1
        y_pred[~has_oof] = 1
    else:
        below = (y_pred != 1) & (confidence < live_cfg["conf_entry"])
        y_pred[below] = 1

    flow = df["flow_momentum_3bar"].ffill().fillna(0).values.astype(np.float64)
    X_gd = compute_guardian_static_array(df, gdn["static"])
    kwargs = dict(
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
        guardian_min_hold_bars=min_hold,
        guardian_feat_cols=gdn["feats"], guardian_static_names=gdn["static"],
        flow_momentum_arr=flow,
        trailing_stop_enabled=TRAILING_STOP_ENABLED,
        trailing_stop_atr=TRAILING_STOP_ATR, trailing_stop_min_bars=TRAILING_STOP_MIN_BARS,
        h4_trend=df["h4_trend"].values if "h4_trend" in df.columns else None,
        vol_ratio=df["vol_ratio_20"].values if "vol_ratio_20" in df.columns else None,
        sl_trigger_mode=live_cfg["sl_trigger_mode"],
    )
    if scale_in:
        kwargs.update(
            pyramiding_enabled=SCALE_IN["enabled"],
            pyramiding_max_per_coin=SCALE_IN["max_per_coin"],
            pyramiding_same_dir=True,
            pyramiding_exit_mode=SCALE_IN["exit_mode"],
        )
    rep = full_trading_report(**kwargs)
    return rep.get("lev5x", rep).get("trades", [])


def run_oof(grid: tuple, live_cfg, hmm_cfg, gdn, feat_cols, lstm_feats, lgbm, lstm, lstm_scaler):
    oof_all = pd.read_parquet(OOF_PATH)
    results = {}
    for mh in grid:
        trades = []
        for sym in ALL_COINS:
            df = _prep_train(sym)
            if df is None:
                continue
            oof_sym = oof_all[oof_all["coin"] == sym]
            trades.extend(_simulate(df, sym, feat_cols, lstm_feats, lgbm, lstm, lstm_scaler,
                                    hmm_cfg, live_cfg, gdn, mh, oof_sym, scale_in=False))
        results[str(mh)] = _agg(trades)
        logger.info(f"OOF min_hold={mh}: {results[str(mh)]}")

    base = results.get("4") or results.get("2") or next(iter(results.values()))
    for sc in results.values():
        if base.get("trades"):
            sc["trade_pct_vs_mh4"] = round(sc["trades"] / base["trades"] * 100, 1)
            sc["delta_ppt_vs_mh4"] = round(sc["ppt"] - base["ppt"], 4)

    out = {
        "phase": "oof",
        "guardian": "ic32_guardian_continuation_v1",
        "stack": "B-dir + hard_consensus + continuation_v1 + SL close",
        "grid": list(grid),
        "baseline_min_hold": 4,
        "modal": MODAL_PER_TRADE,
        "created": datetime.now().isoformat(),
        "results": results,
    }
    with open(OUT_OOF, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    _print_table("OOF", grid, results)
    print(f"Saved {OUT_OOF}")
    return results


def run_holdout(grid: tuple, live_cfg, hmm_cfg, gdn, feat_cols, lstm_feats, lgbm, lstm, lstm_scaler):
    results = {}
    for mh in grid:
        trades = []
        for sym in ALL_COINS:
            df = _prep_holdout(sym)
            if df is None:
                continue
            trades.extend(_simulate(df, sym, feat_cols, lstm_feats, lgbm, lstm, lstm_scaler,
                                    hmm_cfg, live_cfg, gdn, mh, None, scale_in=True))
        results[str(mh)] = _agg(trades)
        logger.info(f"Holdout min_hold={mh}: {results[str(mh)]}")

    base = results.get("4", {})
    for sc in results.values():
        if base.get("trades"):
            sc["delta_ppt_vs_mh4"] = round(sc["ppt"] - base["ppt"], 4)

    out = {
        "phase": "holdout_diagnostic",
        "guardian": "ic32_guardian_continuation_v1",
        "stack": "B-dir + scale_in + SL close",
        "grid": list(grid),
        "baseline_min_hold": 4,
        "modal": MODAL_PER_TRADE,
        "created": datetime.now().isoformat(),
        "results": results,
    }
    with open(OUT_HOLD, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    _print_table("Holdout scale_in", grid, results)
    print(f"Saved {OUT_HOLD}")
    return results


def _print_table(title: str, grid: tuple, results: dict):
    print(f"\n=== {title} guardian min_hold sweep ===")
    print(
        f"{'mh':>3} {'trades':>7} {'WR%':>6} {'PF':>5} {'PPT':>8} "
        f"{'SL%':>5} {'gdnWR':>6} {'early':>6} {'hold':>5}"
    )
    for mh in grid:
        sc = results.get(str(mh), {})
        if not sc.get("trades"):
            continue
        gwr = sc.get("gdn_exit_wr_pct")
        gwr_s = f"{gwr:5.1f}" if gwr is not None else "  n/a"
        print(
            f"{mh:>3} {sc['trades']:>7} {sc['wr']:>5.1f}% {sc['pf']:>5.2f} "
            f"${sc['ppt']:>+7.4f} {sc['sl_rate_pct']:>4.1f}% {gwr_s}% "
            f"{sc.get('early_gdn_losers', 0):>6} {sc.get('avg_hold_bars', 0):>5.1f}"
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oof-only", action="store_true")
    ap.add_argument("--holdout-only", action="store_true")
    ap.add_argument("--mh", nargs="*", type=int, default=None)
    args = ap.parse_args()

    grid = tuple(args.mh) if args.mh else MIN_HOLD_GRID
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

    if not args.holdout_only:
        run_oof(grid, live_cfg, hmm_cfg, gdn, feat_cols, lstm_feats, lgbm, lstm, lstm_scaler)
    if not args.oof_only:
        run_holdout(grid, live_cfg, hmm_cfg, gdn, feat_cols, lstm_feats, lgbm, lstm, lstm_scaler)


if __name__ == "__main__":
    main()