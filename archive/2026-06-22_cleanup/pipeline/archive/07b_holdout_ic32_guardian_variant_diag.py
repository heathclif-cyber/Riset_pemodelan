"""
Diagnostic holdout: Guardian clean_v2 vs ic32_guardian_continuation_v1.

BUKAN evaluasi holdout baru untuk tuning entry — entry frozen B-dir-combined
sama dengan 07_holdout_ic32_b_dir_combined.py. Hanya membandingkan variant Guardian
pada periode Apr-Jun 2026 agar production (continuation_v1) punya scorecard holdout.

Usage:
  python pipeline/07b_holdout_ic32_guardian_variant_diag.py
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
from config import (
    ALL_COINS, HOLDOUT_DIR, MODEL_DIR, LABEL_MAP,
    MODAL_PER_TRADE, LEVERAGE_SIM, FEE_PER_SIDE, SLIPPAGE_PER_SIDE,
    MAX_HOLDING_BARS, GUARDIAN_DYNAMIC_FEATURES,
    SWING_LABEL_MIN_RR, SWING_LABEL_MIN_TP, SWING_LABEL_MAX_SL,
    TP_SL_FALLBACK_TP, TP_SL_FALLBACK_SL,
    TRAILING_STOP_ENABLED, TRAILING_STOP_ATR, TRAILING_STOP_MIN_BARS,
)

logger = setup_logger("07b_guardian_variant_diag")
HOLDOUT_LABEL_DIR = HOLDOUT_DIR / "labeled"
RUN_DIR = MODEL_DIR / "runs" / "ic32_regime_v1"
FROZEN_PATH = RUN_DIR / "b_dir_combined_frozen.json"
INF_CFG_PATH = MODEL_DIR / "inference_config.json"
OUT_JSON = RUN_DIR / "holdout_guardian_variant_diag_apr_jun26.json"
OUT_TRADES_CONT = ROOT / "reports/experiments/holdout_ic32_cont_v1_trades_apr_jun26.csv"
FLOW_MOM_WINDOW = 3
HOLDOUT_MONTHS = 2.5

DYN_EXTRA = {"cvd_slope_h4_delta_entry", "ofi_h4_delta_entry", "flow_momentum_3bar"}


def _apply_live_config() -> dict:
    with open(INF_CFG_PATH, encoding="utf-8") as f:
        cfg = json.load(f)
    cascade = cfg.get("cascade", {})
    guardian = cfg.get("guardian", {})
    ra = cfg.get("regime_alignment", {})
    rr = cfg.get("rr_gate", {})

    import config as project_config
    project_config.CONFIDENCE_THRESHOLD_ENTRY = float(cascade.get("confidence_threshold_entry", 0.59))
    project_config.LSTM_ADJUST_AGREE_BOOST = float(cascade.get("lstm_adjust_agree_boost", 0.05))
    project_config.LSTM_ADJUST_NEUTRAL_PEN = float(cascade.get("lstm_adjust_neutral_pen", 0.0))
    project_config.LSTM_ADJUST_OPPOSITE_PEN = float(cascade.get("lstm_adjust_opposite_pen", 0.65))
    project_config.LSTM_DIRECTIONAL_REVIEW_THRESHOLD = float(
        cascade.get("lstm_directional_review_threshold", 0.35)
    )
    project_config.LSTM_FLAT_REVIEW_ENABLED = bool(cascade.get("lstm_flat_review_enabled", True))
    project_config.LSTM_CONFIRMATION_ENABLED = bool(cascade.get("lstm_confirmation_enabled", True))
    project_config.REGIME_AWARE_ALIGNMENT = bool(ra.get("enabled", True))
    project_config.GUARDIAN_EXIT_THRESHOLD = float(guardian.get("exit_threshold", 0.65))
    project_config.GUARDIAN_MIN_HOLD_BARS = int(guardian.get("min_hold_bars", 2))
    sl_mode = str(rr.get("sl_trigger_mode", "close"))

    btu.CONFIDENCE_THRESHOLD_ENTRY = project_config.CONFIDENCE_THRESHOLD_ENTRY
    btu.LSTM_ADJUST_AGREE_BOOST = project_config.LSTM_ADJUST_AGREE_BOOST
    btu.LSTM_ADJUST_NEUTRAL_PEN = project_config.LSTM_ADJUST_NEUTRAL_PEN
    btu.LSTM_ADJUST_OPPOSITE_PEN = project_config.LSTM_ADJUST_OPPOSITE_PEN
    btu.LSTM_DIRECTIONAL_REVIEW_THRESHOLD = project_config.LSTM_DIRECTIONAL_REVIEW_THRESHOLD
    btu.LSTM_FLAT_REVIEW_ENABLED = project_config.LSTM_FLAT_REVIEW_ENABLED
    btu.LSTM_CONFIRMATION_ENABLED = project_config.LSTM_CONFIRMATION_ENABLED
    btu.REGIME_AWARE_ALIGNMENT = project_config.REGIME_AWARE_ALIGNMENT
    btu.SMART_ENTRY_MODE = "disabled"
    btu.MOMENTUM_DYNAMIC_THRESHOLD_ENABLED = False
    btu.TREND_DYNAMIC_THRESHOLD_ENABLED = False
    btu.LSTM_STANDALONE_ENABLED = False

    return {
        "conf_entry": project_config.CONFIDENCE_THRESHOLD_ENTRY,
        "gdn_exit": project_config.GUARDIAN_EXIT_THRESHOLD,
        "gdn_min_hold": project_config.GUARDIAN_MIN_HOLD_BARS,
        "flip": project_config.REGIME_AWARE_ALIGNMENT,
        "sl_trigger_mode": sl_mode,
    }


def _load_frozen_cfg() -> dict:
    with open(FROZEN_PATH, encoding="utf-8") as f:
        data = json.load(f)
    raw = data["per_state_thresholds"]
    return {int(k): (float(v[0]), float(v[1])) for k, v in raw.items()}


def _build_per_bar_thresholds(hmm_enc: np.ndarray, hmm_cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    default_tl, default_ts = hmm_cfg[-1]
    tl_arr = np.full(len(hmm_enc), default_tl, dtype=np.float64)
    ts_arr = np.full(len(hmm_enc), default_ts, dtype=np.float64)
    for state, (tl, ts) in hmm_cfg.items():
        if state == -1:
            continue
        mask = hmm_enc == state
        tl_arr[mask] = tl
        ts_arr[mask] = ts
    return tl_arr, ts_arr


def _add_momentum_feats(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "cvd_slope_h4" in df.columns:
        df["cvd_slope_h4_delta"] = df["cvd_slope_h4"].diff(1)
    else:
        df["cvd_slope_h4_delta"] = 0.0
    if "ofi_h4_delta" in df.columns:
        df["ofi_h4_accel"] = df["ofi_h4_delta"].diff(2)
    else:
        df["ofi_h4_accel"] = 0.0
    if "rsi_h4" in df.columns:
        df["rsi_h4_slope"] = df["rsi_h4"].diff(2)
    else:
        df["rsi_h4_slope"] = 0.0
    if "dist_liq_50x_long" not in df.columns:
        df["dist_liq_50x_long"] = 0.0
    if "ofi_z_score" in df.columns:
        df["flow_momentum_3bar"] = df["ofi_z_score"].rolling(FLOW_MOM_WINDOW, min_periods=1).mean()
    else:
        df["flow_momentum_3bar"] = 0.0
    return df


def _load_guardian_clean() -> dict:
    with open(MODEL_DIR / "guardian_clean_v2_feature_cols.json", encoding="utf-8") as f:
        feats = json.load(f)
    static = [c for c in feats if c not in set(GUARDIAN_DYNAMIC_FEATURES)]
    return {
        "name": "guardian_clean_v2",
        "model": joblib.load(MODEL_DIR / "guardian_clean_v2.pkl"),
        "scaler": joblib.load(MODEL_DIR / "guardian_clean_v2_scaler.pkl"),
        "feats": feats,
        "static": static,
        "use_cont": False,
    }


def _load_guardian_cont() -> dict:
    run = MODEL_DIR / "runs" / "ic32_guardian_continuation_v1"
    with open(run / "guardian_feature_cols.json", encoding="utf-8") as f:
        feats = json.load(f)
    dyn = set(GUARDIAN_DYNAMIC_FEATURES) | DYN_EXTRA
    static = [c for c in feats if c not in dyn]
    return {
        "name": "ic32_guardian_continuation_v1",
        "model": joblib.load(run / "guardian.pkl"),
        "scaler": joblib.load(run / "guardian_scaler.pkl"),
        "feats": feats,
        "static": static,
        "use_cont": True,
    }


def _scorecard(trades: list[dict]) -> dict:
    if not trades:
        return {"trades": 0}
    n = len(trades)
    wins = [t for t in trades if t.get("net_pnl", 0) > 0]
    losses = [t for t in trades if t.get("net_pnl", 0) <= 0]
    gpnl = sum(t["net_pnl"] for t in wins)
    gloss = abs(sum(t["net_pnl"] for t in losses))
    pf = gpnl / gloss if gloss > 0 else float("inf")
    outcome_counts: dict[str, int] = {}
    hold_bars = []
    for t in trades:
        oc = t.get("outcome", "UNKNOWN")
        outcome_counts[oc] = outcome_counts.get(oc, 0) + 1
        hold_bars.append(t.get("bar_out", 0) - t.get("bar_in", 0))
    gd_n = sum(v for k, v in outcome_counts.items() if "GUARDIAN" in k)
    sl_n = outcome_counts.get("LOSS", 0)
    mom_n = sum(v for k, v in outcome_counts.items() if "MOMENTUM" in k)
    return {
        "trades": n,
        "trades_per_month": round(n / HOLDOUT_MONTHS, 1),
        "win_rate": round(len(wins) / n * 100, 2),
        "total_pnl": round(sum(t.get("net_pnl", 0) for t in trades), 2),
        "pnl_per_trade": round(sum(t.get("net_pnl", 0) for t in trades) / n, 4),
        "profit_factor": round(pf, 3),
        "guardian_exit_pct": round(gd_n / n * 100, 2),
        "momentum_exit_pct": round(mom_n / n * 100, 2),
        "sl_rate_pct": round(sl_n / n * 100, 2),
        "avg_hold_bars": round(float(np.mean(hold_bars)), 2),
        "median_hold_bars": round(float(np.median(hold_bars)), 2),
        "outcome_counts": outcome_counts,
    }


def backtest_coin(sym: str, hmm_cfg: dict, live_cfg: dict, gdn: dict,
                  lgbm_model, lstm_model, lstm_scaler, lstm_feat_cols, feat_cols) -> tuple[list[dict], pd.Index]:
    p = HOLDOUT_LABEL_DIR / f"{sym}_features_v3.parquet"
    if not p.exists():
        return [], pd.Index([])

    df = pd.read_parquet(p)
    df = ensure_utc_index(df).sort_index()
    if gdn["use_cont"]:
        df = _add_momentum_feats(df)

    rp = HOLDOUT_LABEL_DIR / f"{sym}_regime_h1.parquet"
    if rp.exists():
        try:
            reg = pd.read_parquet(rp)
            for col in ["hmm_regime_enc", "hmm_regime"]:
                if col in df.columns:
                    df = df.drop(columns=[col])
            cols = [c for c in ["hmm_regime_enc", "hmm_regime"] if c in reg.columns]
            df = df.join(reg[cols], how="left")
            df["hmm_regime_enc"] = df["hmm_regime_enc"].fillna(1).astype("int32")
        except Exception:
            df["hmm_regime_enc"] = 1
    else:
        df["hmm_regime_enc"] = 1

    mask = df["label"].astype(str).isin(LABEL_MAP)
    df = df[mask].copy()
    n = len(df)
    if n < 30:
        return [], df.index

    # Apply same training parity overrides seperti live inference
    df = apply_training_parity(df)

    X = np.zeros((n, len(feat_cols)), dtype=np.float64)
    for idx, col in enumerate(feat_cols):
        if col in df.columns:
            X[:, idx] = df[col].ffill().fillna(0).values.astype(np.float64)

    hmm_enc = df["hmm_regime_enc"].values.astype(np.int32)
    thr_l, thr_s = _build_per_bar_thresholds(hmm_enc, hmm_cfg)
    y_pred, confidence = hierarchical_predict(
        None, lgbm_model, lstm_model, lstm_scaler,
        X, feat_cols, [], df, model_dir=RUN_DIR, lstm_feat_cols=lstm_feat_cols,
        per_bar_thr_long=thr_l, per_bar_thr_short=thr_s,
    )
    below = (y_pred != 1) & (confidence < live_cfg["conf_entry"])
    y_pred[below] = 1

    atr = df["atr_14_h1"].values if "atr_14_h1" in df.columns else np.ones(n)
    h4t = df["h4_trend"].values if "h4_trend" in df.columns else None
    volr = df["vol_ratio_20"].values if "vol_ratio_20" in df.columns else None
    X_guardian = compute_guardian_static_array(df, gdn["static"])

    kw = dict(
        y_pred=y_pred, y_actual=df["label"].map(LABEL_MAP).values.astype(np.int64),
        atr=atr, close=df["close"].values, high=df["high"].values, low=df["low"].values,
        h4_swing_highs=df["h4_swing_high"].values if "h4_swing_high" in df.columns else None,
        h4_swing_lows=df["h4_swing_low"].values if "h4_swing_low" in df.columns else None,
        index=df.index, modal=MODAL_PER_TRADE, leverages=LEVERAGE_SIM,
        fee_per_side=FEE_PER_SIDE, slippage=SLIPPAGE_PER_SIDE,
        min_rr=SWING_LABEL_MIN_RR, min_tp_atr=SWING_LABEL_MIN_TP, max_sl_atr=SWING_LABEL_MAX_SL,
        tp_fallback_atr=TP_SL_FALLBACK_TP, sl_fallback_atr=TP_SL_FALLBACK_SL,
        max_hold=MAX_HOLDING_BARS, symbol=sym, confidence=confidence,
        guardian_model=gdn["model"], guardian_scaler=gdn["scaler"], X_guardian=X_guardian,
        guardian_exit_threshold=live_cfg["gdn_exit"],
        guardian_min_hold_bars=live_cfg["gdn_min_hold"],
        trailing_stop_enabled=TRAILING_STOP_ENABLED,
        trailing_stop_atr=TRAILING_STOP_ATR,
        trailing_stop_min_bars=TRAILING_STOP_MIN_BARS,
        h4_trend=h4t, vol_ratio=volr,
        sl_trigger_mode=live_cfg["sl_trigger_mode"],
    )
    if gdn["use_cont"]:
        flow = df["flow_momentum_3bar"].ffill().fillna(0).values.astype(np.float64)
        kw.update(
            guardian_feat_cols=gdn["feats"],
            guardian_static_names=gdn["static"],
            flow_momentum_arr=flow,
        )

    report = full_trading_report(**kw)
    return report.get("trades", []), df.index


def _trades_to_df(trades: list[dict], sym: str, gdn_name: str, index) -> list[dict]:
    rows = []
    for t in trades:
        bi = t.get("bar_in", 0)
        bo = t.get("bar_out", bi)
        rows.append({
            "guardian_variant": gdn_name,
            "coin": sym,
            "direction": t.get("direction"),
            "outcome": t.get("outcome"),
            "net_pnl": t.get("net_pnl"),
            "hold_bars": bo - bi,
            "is_win": t.get("net_pnl", 0) > 0,
            "entry_time": index[bi] if bi < len(index) else None,
            "exit_time": index[min(bo, len(index) - 1)] if len(index) else None,
        })
    return rows


def main():
    if not FROZEN_PATH.exists():
        raise FileNotFoundError(f"Missing {FROZEN_PATH}")

    hmm_cfg = _load_frozen_cfg()
    live_cfg = _apply_live_config()
    g_clean = _load_guardian_clean()
    g_cont = _load_guardian_cont()

    with open(MODEL_DIR / "feature_cols_ic32_regime.json", encoding="utf-8") as f:
        feat_cols = json.load(f)
    with open(MODEL_DIR / "feature_cols_lstm_temporal.json", encoding="utf-8") as f:
        lstm_feat_cols = json.load(f)
    lgbm_model = joblib.load(RUN_DIR / "lgbm.pkl")
    lstm_model = load_lstm(MODEL_DIR / "lstm_best.pt", device="cpu")
    lstm_scaler = joblib.load(MODEL_DIR / "lstm_scaler.pkl")

    ref_path = RUN_DIR / "holdout_b_dir_combined_apr_jun26.json"
    ref_clean = {}
    if ref_path.exists():
        with open(ref_path, encoding="utf-8") as f:
            ref_clean = json.load(f).get("aggregate", {})

    results = {}
    trade_rows_cont = []
    for gdn in (g_clean, g_cont):
        all_trades = []
        for sym in ALL_COINS:
            tr, ts = backtest_coin(
                sym, hmm_cfg, live_cfg, gdn,
                lgbm_model, lstm_model, lstm_scaler, lstm_feat_cols, feat_cols,
            )
            all_trades.extend(tr)
            if gdn["name"] == g_cont["name"]:
                trade_rows_cont.extend(_trades_to_df(tr, sym, gdn["name"], ts))
            logger.info(f"  [{gdn['name']}] {sym}: {len(tr)} trades")

        sc = _scorecard(all_trades)
        results[gdn["name"]] = sc
        print(f"\n{gdn['name']}: {sc['trades']} trades | WR {sc['win_rate']}% | "
              f"PPT ${sc['pnl_per_trade']:+.4f} | PF {sc['profit_factor']}")

    out = {
        "meta": {
            "type": "guardian_variant_diagnostic",
            "note": "Entry frozen B-dir-combined; NOT a new holdout tune. Guardian variant compare only.",
            "holdout_period": "2026-04-01 to 2026-06-13",
            "sl_trigger_mode": live_cfg["sl_trigger_mode"],
            "generated_at": datetime.utcnow().isoformat(),
        },
        "variants": results,
        "vs_prior_clean_v2_holdout": ref_clean,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    if trade_rows_cont:
        pd.DataFrame(trade_rows_cont).to_csv(OUT_TRADES_CONT, index=False)

    print(f"\nSaved {OUT_JSON}")
    if trade_rows_cont:
        print(f"Saved {OUT_TRADES_CONT}")


if __name__ == "__main__":
    main()