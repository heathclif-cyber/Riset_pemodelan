"""
Full pipeline OOF: dual-LSTM complement wired to hierarchical_predict.

Baseline lstm_best.pt hard_consensus unchanged.
Complement v2 conditional_momentum pre-adjusts LGBM OOF on FLAT+vol_spike bars only.

Genuine: OOF only, B-dir frozen, holdout NOT touched.

Usage:
  python pipeline/09_ic32_dual_lstm_stage2_pipeline.py
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
    IC32_DIR, COMPLEMENT_DIR, COMPLEMENT_RUN,
    apply_dual_complement_to_proba, build_per_bar_thresholds,
    genuine_audit_block, load_b_dir_hmm_cfg, load_production_defaults,
    summarize_trades,
)
from config import (
    ALL_COINS, LABEL_DIR, MODEL_DIR, LABEL_MAP, TRAIN_CUTOFF_DATE,
    MODAL_PER_TRADE, LEVERAGE_SIM, FEE_PER_SIDE, SLIPPAGE_PER_SIDE,
    MAX_HOLDING_BARS, GUARDIAN_DYNAMIC_FEATURES,
    SWING_LABEL_MIN_RR, SWING_LABEL_MIN_TP, SWING_LABEL_MAX_SL,
    TP_SL_FALLBACK_TP, TP_SL_FALLBACK_SL,
    TRAILING_STOP_ENABLED, TRAILING_STOP_ATR, TRAILING_STOP_MIN_BARS,
)

logger = setup_logger("09_dual_lstm_stage2")
OUT_JSON = IC32_DIR / "ic32_dual_lstm_stage2_pipeline.json"
OOF_PATH = IC32_DIR / "oof_predictions.parquet"
COMP_OOF_PATH = COMPLEMENT_DIR / "oof_lstm_predictions.parquet"
FLOW_MOM_WINDOW = 3
DYN_EXTRA = {"cvd_slope_h4_delta_entry", "ofi_h4_delta_entry", "flow_momentum_3bar"}
PPT_GATE = 0.01
MIN_TRADE_RATIO = 0.80

DUAL_CONFIGS = [
    {"label": "baseline_no_dual", "dual_complement": False},
    {
        "label": "dual_b10_o14", "dual_complement": True,
        "vol_thr": 2.0, "bull_thr": 0.38, "bear_thr": 0.50,
        "near_miss_gap": 0.05, "boost": 0.10, "comp_opposite_pen": 0.14,
    },
    {
        "label": "dual_b08_o14", "dual_complement": True,
        "vol_thr": 2.0, "bull_thr": 0.38, "bear_thr": 0.50,
        "near_miss_gap": 0.05, "boost": 0.08, "comp_opposite_pen": 0.14,
    },
    {
        "label": "dual_b10_o10", "dual_complement": True,
        "vol_thr": 2.0, "bull_thr": 0.38, "bear_thr": 0.50,
        "near_miss_gap": 0.05, "boost": 0.10, "comp_opposite_pen": 0.10,
    },
    {
        "label": "dual_b12_o14", "dual_complement": True,
        "vol_thr": 2.0, "bull_thr": 0.38, "bear_thr": 0.50,
        "near_miss_gap": 0.05, "boost": 0.12, "comp_opposite_pen": 0.14,
    },
]


def _apply_production_cascade() -> dict:
    prod = load_production_defaults()
    import config as project_config

    project_config.LSTM_ADJUST_AGREE_BOOST = prod["agree_boost"]
    project_config.LSTM_ADJUST_NEUTRAL_PEN = prod["neutral_pen"]
    project_config.LSTM_ADJUST_OPPOSITE_PEN = prod["opposite_pen"]
    project_config.LSTM_DIRECTIONAL_REVIEW_THRESHOLD = prod["dir_review_thr"]
    project_config.LSTM_FLAT_REVIEW_ENABLED = prod["flat_review"]
    project_config.LSTM_CONFIRMATION_ENABLED = True
    project_config.CONFIDENCE_THRESHOLD_ENTRY = prod["conf_entry"]
    project_config.REGIME_AWARE_ALIGNMENT = prod["flip"]
    project_config.HMM_GATE_LSTM_ENABLED = prod["hmm_gate_lstm"]

    btu.LSTM_ADJUST_AGREE_BOOST = prod["agree_boost"]
    btu.LSTM_ADJUST_NEUTRAL_PEN = prod["neutral_pen"]
    btu.LSTM_ADJUST_OPPOSITE_PEN = prod["opposite_pen"]
    btu.LSTM_DIRECTIONAL_REVIEW_THRESHOLD = prod["dir_review_thr"]
    btu.LSTM_FLAT_REVIEW_ENABLED = prod["flat_review"]
    btu.LSTM_CONFIRMATION_ENABLED = True
    btu.CONFIDENCE_THRESHOLD_ENTRY = prod["conf_entry"]
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


def _align_comp(sym: str, df_index: pd.Index, comp_all: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sym_comp = comp_all[comp_all["coin"] == sym][["p0", "p1", "p2", "has_oof", "vol_spike"]].copy()
    if sym_comp.index.name != "ts" and "ts" not in sym_comp.columns:
        sym_comp = sym_comp.reset_index().rename(columns={"index": "ts"}).set_index("ts")
    aligned = sym_comp.reindex(df_index)
    lstm_comp = aligned[["p0", "p1", "p2"]].fillna(0).values.astype(np.float32)
    valid = aligned["has_oof"].fillna(False).values.astype(bool)
    if "vol_spike" in aligned.columns:
        vol = aligned["vol_spike"].fillna(-99).values.astype(np.float32)
    else:
        vol = np.full(len(df_index), -99.0, dtype=np.float32)
    return lstm_comp, valid, vol


def _run_coin(sym: str, oof_sym: pd.DataFrame, comp_all: pd.DataFrame,
              feat_cols: list, lstm_feats: list, lgbm, lstm, lstm_scaler,
              hmm_cfg: dict, live_cfg: dict, gdn: dict, dual_cfg: dict) -> list:
    df = _prep_df(sym)
    if df is None:
        return []
    n = len(df)
    merged = df.join(oof_sym[["p0", "p1", "p2", "has_oof"]], how="left")
    has_oof = merged["has_oof"].fillna(False).values.astype(bool)
    if has_oof.sum() < 30:
        return []

    oof_proba = np.column_stack([merged["p0"].values, merged["p1"].values, merged["p2"].values])
    hmm_enc = df["hmm_regime_enc"].values.astype(np.int32)
    thr_l, thr_s = build_per_bar_thresholds(hmm_enc, hmm_cfg)

    if dual_cfg.get("dual_complement"):
        lstm_comp, comp_valid, vol_spike = _align_comp(sym, df.index, comp_all)
        if "vol_spike_zscore" in df.columns:
            vol_spike = df["vol_spike_zscore"].fillna(-99).values.astype(np.float32)
        oof_proba = apply_dual_complement_to_proba(
            oof_proba, lstm_comp, vol_spike, comp_valid, hmm_enc, hmm_cfg, dual_cfg,
        )

    X = np.zeros((n, len(feat_cols)), dtype=np.float64)
    for idx, col in enumerate(feat_cols):
        if col in df.columns:
            X[:, idx] = df[col].ffill().fillna(0).values.astype(np.float64)

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


def main():
    sep = "=" * 78
    if not COMP_OOF_PATH.exists():
        raise FileNotFoundError(f"Missing {COMP_OOF_PATH} -- train {COMPLEMENT_RUN} first")

    live_cfg = _apply_production_cascade()
    hmm_cfg = load_b_dir_hmm_cfg()

    print(f"\n{sep}")
    print("  DUAL-LSTM Stage 2 -- FULL PIPELINE OOF (genuine)")
    print(f"  B-dir + continuation_v1 + complement v2 on FLAT+vol_spike")
    print(f"  Configs: {len(DUAL_CONFIGS)} | holdout sealed")
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
    comp_all = pd.read_parquet(COMP_OOF_PATH)

    t0 = time.time()
    results = []
    baseline_sm = None

    for i, cfg in enumerate(DUAL_CONFIGS):
        t_cfg = time.time()
        trades = []
        for sym in ALL_COINS:
            oof_sym = oof_all[oof_all["coin"] == sym]
            trades.extend(_run_coin(
                sym, oof_sym, comp_all, feat_cols, lstm_feats,
                lgbm, lstm, lstm_scaler, hmm_cfg, live_cfg, gdn, cfg,
            ))
        sm = summarize_trades(trades)
        row = {**cfg, **sm}
        if not cfg.get("dual_complement"):
            baseline_sm = sm
        elif baseline_sm:
            row["delta_ppt"] = sm["ppt"] - baseline_sm["ppt"]
            row["delta_n"] = sm["n"] - baseline_sm["n"]
            row["passes_gate"] = (
                sm["ppt"] >= baseline_sm["ppt"] + PPT_GATE
                and sm["n"] >= baseline_sm["n"] * MIN_TRADE_RATIO
                and sm["pf"] >= baseline_sm["pf"] * 0.98
            )
        results.append(row)
        dt = time.time() - t_cfg
        print(f"  [{i+1}/{len(DUAL_CONFIGS)}] {cfg['label']}: "
              f"N={sm['n']:,} PPT={sm['ppt']:+.4f} PF={sm['pf']:.3f} ({dt:.0f}s)")

    ranked = sorted(
        [r for r in results if r.get("dual_complement")],
        key=lambda x: x["ppt"], reverse=True,
    )
    winners = [r for r in ranked if r.get("passes_gate")]
    best = winners[0] if winners else (ranked[0] if ranked else None)
    decision = "PROMOTE_CANDIDATE" if winners else "NO_PROMOTE"

    print(f"\n{sep}")
    b = baseline_sm
    print(f"  BASELINE: N={b['n']:,} PPT={b['ppt']:+.4f} PF={b['pf']:.3f}")
    for r in ranked:
        pas = "Y" if r.get("passes_gate") else "N"
        print(f"  {r['label']:<20} N={r['n']:>6,} dN={r.get('delta_n',0):>+5} "
              f"PPT={r['ppt']:+.4f} dPPT={r.get('delta_ppt',0):+.4f} PASS={pas}")

    out = {
        **genuine_audit_block(),
        "track": "dual_lstm_full_pipeline",
        "complement_run": COMPLEMENT_RUN,
        "created": datetime.now().isoformat(),
        "elapsed_sec": round(time.time() - t0, 1),
        "baseline": baseline_sm,
        "decision": decision,
        "best": best,
        "winners": winners,
        "all_results": results,
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\n  Decision: {decision} | Saved: {OUT_JSON}")
    print(f"{sep}\n")


if __name__ == "__main__":
    main()