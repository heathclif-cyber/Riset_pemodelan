"""
Stage 2 -- ic32 LGBM+LSTM fusion: FULL PIPELINE OOF (top candidates).

Stack: B-dir HMM + hard_consensus + continuation_v1 Guardian + SL close.
Uses hierarchical_predict for accuracy (per_bar B-dir thresholds).

Usage:
  python pipeline/09_ic32_lstm_fusion_stage2_pipeline.py
"""
import json
import sys
import time
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
from pipeline.ic32_fusion_shared import (
    IC32_DIR, COMPLEMENT_DIR,
    build_per_bar_thresholds, genuine_audit_block,
    load_b_dir_hmm_cfg, load_production_defaults, summarize_trades,
)
from config import (
    ALL_COINS, LABEL_DIR, MODEL_DIR, LABEL_MAP, TRAIN_CUTOFF_DATE,
    MODAL_PER_TRADE, LEVERAGE_SIM, FEE_PER_SIDE, SLIPPAGE_PER_SIDE,
    MAX_HOLDING_BARS, GUARDIAN_DYNAMIC_FEATURES,
    SWING_LABEL_MIN_RR, SWING_LABEL_MIN_TP, SWING_LABEL_MAX_SL,
    TP_SL_FALLBACK_TP, TP_SL_FALLBACK_SL,
    TRAILING_STOP_ENABLED, TRAILING_STOP_ATR, TRAILING_STOP_MIN_BARS,
)

logger = setup_logger("09_ic32_fusion_stage2")
STAGE1_OUT = IC32_DIR / "ic32_lstm_fusion_stage1_signal.json"
STAGE2_OUT = IC32_DIR / "ic32_lstm_fusion_stage2_pipeline.json"
OOF_PATH = IC32_DIR / "oof_predictions.parquet"
FLOW_MOM_WINDOW = 3
DYN_EXTRA = {"cvd_slope_h4_delta_entry", "ofi_h4_delta_entry", "flow_momentum_3bar"}
PPT_GATE = 0.01
MIN_TRADE_RATIO = 0.80


def _apply_cascade_cfg(cfg: dict) -> dict:
    prod = load_production_defaults()
    import config as project_config

    project_config.LSTM_ADJUST_AGREE_BOOST = float(cfg.get("agree_boost", prod["agree_boost"]))
    project_config.LSTM_ADJUST_NEUTRAL_PEN = float(cfg.get("neutral_pen", prod["neutral_pen"]))
    project_config.LSTM_ADJUST_OPPOSITE_PEN = float(cfg.get("opposite_pen", prod["opposite_pen"]))
    project_config.LSTM_DIRECTIONAL_REVIEW_THRESHOLD = float(cfg.get("dir_review_thr", prod["dir_review_thr"]))
    project_config.LSTM_FLAT_REVIEW_ENABLED = bool(cfg.get("flat_review", prod["flat_review"]))
    project_config.LSTM_CONFIRMATION_ENABLED = True
    project_config.CONFIDENCE_THRESHOLD_ENTRY = float(cfg.get("conf_entry", prod["conf_entry"]))
    project_config.REGIME_AWARE_ALIGNMENT = bool(cfg.get("flip", prod["flip"]))
    project_config.HMM_GATE_LSTM_ENABLED = bool(cfg.get("hmm_gate_lstm", prod["hmm_gate_lstm"]))
    project_config.GUARDIAN_EXIT_THRESHOLD = 0.65
    project_config.GUARDIAN_MIN_HOLD_BARS = 2

    btu.LSTM_ADJUST_AGREE_BOOST = project_config.LSTM_ADJUST_AGREE_BOOST
    btu.LSTM_ADJUST_NEUTRAL_PEN = project_config.LSTM_ADJUST_NEUTRAL_PEN
    btu.LSTM_ADJUST_OPPOSITE_PEN = project_config.LSTM_ADJUST_OPPOSITE_PEN
    btu.LSTM_DIRECTIONAL_REVIEW_THRESHOLD = project_config.LSTM_DIRECTIONAL_REVIEW_THRESHOLD
    btu.LSTM_FLAT_REVIEW_ENABLED = project_config.LSTM_FLAT_REVIEW_ENABLED
    btu.LSTM_CONFIRMATION_ENABLED = project_config.LSTM_CONFIRMATION_ENABLED
    btu.CONFIDENCE_THRESHOLD_ENTRY = project_config.CONFIDENCE_THRESHOLD_ENTRY
    btu.REGIME_AWARE_ALIGNMENT = project_config.REGIME_AWARE_ALIGNMENT
    btu.HMM_GATE_LSTM_ENABLED = project_config.HMM_GATE_LSTM_ENABLED
    btu.SMART_ENTRY_MODE = "disabled"
    btu.MOMENTUM_DYNAMIC_THRESHOLD_ENABLED = False
    btu.TREND_DYNAMIC_THRESHOLD_ENABLED = False
    btu.LSTM_STANDALONE_ENABLED = False

    return {"conf_entry": project_config.CONFIDENCE_THRESHOLD_ENTRY}


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


def _run_coin(sym: str, oof_sym: pd.DataFrame, feat_cols: list, lstm_feats: list,
              lgbm, lstm, lstm_scaler, hmm_cfg: dict, live_cfg: dict, gdn: dict) -> list:
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
        model_dir=IC32_DIR, lstm_feat_cols=lstm_feats, lgbm_proba=oof_proba,
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
        X_guardian=X_gd, guardian_exit_threshold=0.65, guardian_min_hold_bars=2,
        guardian_feat_cols=gdn["feats"], guardian_static_names=gdn["static"],
        flow_momentum_arr=flow,
        trailing_stop_enabled=TRAILING_STOP_ENABLED,
        trailing_stop_atr=TRAILING_STOP_ATR, trailing_stop_min_bars=TRAILING_STOP_MIN_BARS,
        h4_trend=df["h4_trend"].values if "h4_trend" in df.columns else None,
        vol_ratio=df["vol_ratio_20"].values if "vol_ratio_20" in df.columns else None,
    )
    trades = rep.get("lev5x", rep).get("trades", [])
    for t in trades:
        t["symbol"] = sym
    return trades


def eval_config(cfg: dict, oof_all: pd.DataFrame, feat_cols: list, lstm_feats: list,
                lgbm, lstm, lstm_scaler, hmm_cfg: dict, gdn: dict) -> list:
    live_cfg = _apply_cascade_cfg(cfg)
    all_trades = []
    for sym in ALL_COINS:
        oof_sym = oof_all[oof_all["coin"] == sym]
        all_trades.extend(
            _run_coin(sym, oof_sym, feat_cols, lstm_feats, lgbm, lstm, lstm_scaler,
                      hmm_cfg, live_cfg, gdn)
        )
    return all_trades


def passes_gate(variant: dict, baseline: dict) -> bool:
    ratio = variant["n"] / baseline["n"] if baseline["n"] else 0
    return (
        variant["ppt"] >= baseline["ppt"] + PPT_GATE
        and ratio >= MIN_TRADE_RATIO
        and variant["pf"] >= baseline["pf"] * 0.98
    )


def main():
    sep = "=" * 78
    if not STAGE1_OUT.exists():
        raise FileNotFoundError(f"Run stage1 first: {STAGE1_OUT}")

    with open(STAGE1_OUT, encoding="utf-8") as f:
        stage1 = json.load(f)

    hmm_cfg = load_b_dir_hmm_cfg()
    prod = load_production_defaults()
    candidates = [{"fusion": "baseline", "label": "baseline_production", **prod}]
    for c in stage1["top_candidates"]:
        row = {k: c[k] for k in c if k not in ("rank_score", "n_long", "n_short", "n_dir",
                                                "delta_long", "delta_short", "delta_dir",
                                                "dir_ratio_vs_base")}
        candidates.append(row)

    print(f"\n{sep}")
    print("  STAGE 2: ic32 LGBM+LSTM Fusion -- FULL PIPELINE (OOF genuine)")
    print(f"  Candidates: {len(candidates)} | B-dir + continuation_v1 | holdout sealed")
    print(sep)

    with open(MODEL_DIR / "feature_cols_ic32_regime.json", encoding="utf-8") as f:
        feat_cols = json.load(f)
    with open(MODEL_DIR / "feature_cols_lstm_temporal.json", encoding="utf-8") as f:
        lstm_feats = json.load(f)[:11]

    lgbm = joblib.load(IC32_DIR / "lgbm.pkl")
    lstm = load_lstm(MODEL_DIR / "lstm_best.pt", device="cpu")
    lstm_scaler = joblib.load(MODEL_DIR / "lstm_scaler.pkl")
    gdn = _load_guardian_cont()
    oof_all = pd.read_parquet(OOF_PATH)

    t0 = time.time()
    results = []
    baseline_sm = None

    for i, cfg in enumerate(candidates):
        t_cfg = time.time()
        trades = eval_config(cfg, oof_all, feat_cols, lstm_feats, lgbm, lstm, lstm_scaler, hmm_cfg, gdn)
        sm = summarize_trades(trades)
        row = {"label": cfg["label"], **cfg, **sm}
        if cfg.get("fusion") == "baseline":
            baseline_sm = sm
        elif baseline_sm:
            row["delta_ppt"] = sm["ppt"] - baseline_sm["ppt"]
            row["delta_n"] = sm["n"] - baseline_sm["n"]
            row["passes_gate"] = passes_gate(sm, baseline_sm)
        results.append(row)
        dt = time.time() - t_cfg
        print(f"  [{i+1}/{len(candidates)}] {cfg['label']}: "
              f"N={sm['n']:,} PPT={sm['ppt']:+.4f} PF={sm['pf']:.3f} ({dt:.0f}s)")

    ranked = sorted(
        [r for r in results if r.get("fusion") != "baseline"],
        key=lambda x: x["ppt"], reverse=True,
    )
    winners = [r for r in ranked if r.get("passes_gate")]

    print(f"\n{sep}")
    print("  STAGE 2 vs BASELINE")
    print(sep)
    b = baseline_sm
    print(f"  BASELINE: N={b['n']:,} WR={b['wr']:.1f}% PPT={b['ppt']:+.4f} PF={b['pf']:.3f}")
    print(f"\n  {'Label':<44} {'N':>7} {'dN':>6} {'PPT':>8} {'dPPT':>7} {'PF':>5} {'PASS':>4}")
    for r in ranked:
        pf = f"{r['pf']:.3f}" if r["pf"] != float("inf") else " INF"
        pas = "Y" if r.get("passes_gate") else "N"
        print(f"  {r['label']:<44} {r['n']:>7,} {r.get('delta_n',0):>+6,} "
              f"{r['ppt']:>+8.4f} {r.get('delta_ppt',0):>+7.4f} {pf:>5} {pas:>4}")

    elapsed = time.time() - t0
    best = winners[0] if winners else (ranked[0] if ranked else None)
    decision = "PROMOTE_CANDIDATE" if winners else "NO_PROMOTE"

    out = {
        **genuine_audit_block(),
        "stage": 2,
        "created": datetime.now().isoformat(),
        "elapsed_sec": round(elapsed, 1),
        "stage1_source": str(STAGE1_OUT),
        "baseline": baseline_sm,
        "ppt_gate": PPT_GATE,
        "min_trade_ratio": MIN_TRADE_RATIO,
        "n_candidates": len(candidates),
        "n_winners": len(winners),
        "decision": decision,
        "best": best,
        "winners": winners,
        "all_results": results,
    }
    with open(STAGE2_OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\n  Elapsed: {elapsed/60:.1f} min | Decision: {decision}")
    print(f"  Saved: {STAGE2_OUT}")
    print(f"{sep}\n")


if __name__ == "__main__":
    main()