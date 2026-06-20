"""
pipeline/08d_oof_ic32_adaptive_full_stack.py — Full-stack OOF for adaptive HMM thresholds

Evaluates top candidates from 05f_hmm_adaptive_on_config_c_ic32.py with full
LSTM + FLIP + Guardian stack.

Jalankan:
  python pipeline/08d_oof_ic32_adaptive_full_stack.py
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
from config import (
    ALL_COINS, LABEL_DIR, MODEL_DIR, LABEL_MAP, TRAIN_CUTOFF_DATE,
    MODAL_PER_TRADE, LEVERAGE_SIM, FEE_PER_SIDE, SLIPPAGE_PER_SIDE,
    MAX_HOLDING_BARS, GUARDIAN_DYNAMIC_FEATURES,
    SWING_LABEL_MIN_RR, SWING_LABEL_MIN_TP, SWING_LABEL_MAX_SL,
    TP_SL_FALLBACK_TP, TP_SL_FALLBACK_SL,
    TRAILING_STOP_ENABLED, TRAILING_STOP_ATR, TRAILING_STOP_MIN_BARS,
)

logger = setup_logger("08d_ic32_adaptive_oof")
RUN_DIR = MODEL_DIR / "runs" / "ic32_regime_v1"
OOF_PATH = RUN_DIR / "oof_predictions.parquet"
ADAPTIVE_PATH = RUN_DIR / "hmm_adaptive_config_c_ic32.json"
INF_CFG_PATH = MODEL_DIR / "inference_config.json"
BASELINE_STATIC_PATH = RUN_DIR / "oof_simons_full_stack_scorecard.json"

RANGING = {1, 2}
TRENDING = {0, 3}


def _apply_live_config():
    with open(INF_CFG_PATH, encoding="utf-8") as f:
        cfg = json.load(f)
    cascade = cfg.get("cascade", {})
    guardian = cfg.get("guardian", {})
    ra = cfg.get("regime_alignment", {})

    import config as project_config
    project_config.LGBM_THRESHOLD_LONG = float(cascade.get("lgbm_threshold_long", 0.69))
    project_config.LGBM_THRESHOLD_SHORT = float(cascade.get("lgbm_threshold_short", 0.59))
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

    btu.LGBM_THRESHOLD_LONG = project_config.LGBM_THRESHOLD_LONG
    btu.LGBM_THRESHOLD_SHORT = project_config.LGBM_THRESHOLD_SHORT
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
    }


def _parse_cfg(raw: dict) -> dict:
    return {int(k): (float(v[0]), float(v[1])) for k, v in raw.items()}


def _build_per_bar_thresholds(hmm_enc: np.ndarray, h4: np.ndarray | None,
                            base_cfg: dict, offsets: dict | None) -> tuple[np.ndarray, np.ndarray]:
    n = len(hmm_enc)
    default_tl, default_ts = base_cfg[-1]
    tl_arr = np.full(n, default_tl, dtype=np.float64)
    ts_arr = np.full(n, default_ts, dtype=np.float64)
    for state, (tl, ts) in base_cfg.items():
        if state == -1:
            continue
        mask = hmm_enc == state
        tl_arr[mask] = tl
        ts_arr[mask] = ts

    if not offsets or h4 is None:
        return tl_arr, ts_arr

    rng_wt = float(offsets.get("ranging_wt_tight", 0.0))
    rng_ct = float(offsets.get("ranging_ct_ease", 0.0))
    trd_wt = float(offsets.get("trending_wt_ease", 0.0))
    trd_ct = float(offsets.get("trending_ct_tight", 0.0))

    for i in range(n):
        state = int(hmm_enc[i])
        h = float(h4[i])
        if abs(h) < 1e-9:
            continue
        tl, ts = float(tl_arr[i]), float(ts_arr[i])
        if state in RANGING:
            if h > 0:
                tl -= rng_ct
                ts -= rng_ct
            else:
                ts -= rng_ct
                tl -= rng_ct
                tl += rng_wt
        elif state in TRENDING:
            if state == 3:
                if h > 0:
                    tl -= trd_wt
                    ts += trd_ct
                else:
                    tl += trd_ct
                    ts -= trd_wt
            else:
                if h < 0:
                    ts -= trd_wt
                    tl += trd_ct
                else:
                    ts += trd_ct
                    tl -= trd_wt
        tl_arr[i] = np.clip(tl, 0.35, 0.85)
        ts_arr[i] = np.clip(ts, 0.35, 0.85)
    return tl_arr, ts_arr


def _load_guardian():
    for model_name, feat_name, scaler_name in (
        ("guardian_clean_v2.pkl", "guardian_clean_v2_feature_cols.json", "guardian_clean_v2_scaler.pkl"),
        ("guardian_best.pkl", "guardian_feature_cols.json", "guardian_scaler.pkl"),
    ):
        mp, fp, sp = MODEL_DIR / model_name, MODEL_DIR / feat_name, MODEL_DIR / scaler_name
        if mp.exists() and fp.exists() and sp.exists():
            with open(fp, encoding="utf-8") as f:
                feat_cols = json.load(f)
            return joblib.load(mp), joblib.load(sp), feat_cols, model_name
    raise FileNotFoundError("Guardian model files not found")


def _aggregate(all_trades: list) -> dict:
    n = len(all_trades)
    if n == 0:
        return {}
    wins = sum(1 for t in all_trades if t.get("net_pnl", 0) > 0)
    total_pnl = sum(t.get("net_pnl", 0) for t in all_trades)
    long_t = [t for t in all_trades if t.get("direction") == "LONG"]
    short_t = [t for t in all_trades if t.get("direction") == "SHORT"]
    gpnl = sum(t["net_pnl"] for t in all_trades if t.get("net_pnl", 0) > 0)
    gloss = abs(sum(t["net_pnl"] for t in all_trades if t.get("net_pnl", 0) <= 0))
    pf = gpnl / gloss if gloss > 0 else float("inf")
    return {
        "total_trades": n,
        "win_rate": round(wins / n * 100, 2),
        "long_pct": round(len(long_t) / n * 100, 2),
        "short_long_ratio": round(len(short_t) / max(len(long_t), 1), 3),
        "total_pnl": round(total_pnl, 2),
        "pnl_per_trade": round(total_pnl / n, 4),
        "profit_factor": round(pf, 3),
    }


def eval_candidate(name: str, base_cfg: dict, offsets: dict | None,
                   oof_all, feat_cols, models, live_cfg) -> dict:
    lgbm_model, lstm_model, lstm_scaler, lstm_feat_cols = models["lgbm"], models["lstm"], models["lstm_scaler"], models["lstm_feat"]
    guardian_model, guardian_scaler, g_static = models["guard"], models["guard_scaler"], models["g_static"]

    all_trades = []
    for sym in ALL_COINS:
        path = LABEL_DIR / f"{sym}_features_v3.parquet"
        if not path.exists():
            continue
        df = pd.read_parquet(path)
        df = ensure_utc_index(df).sort_index()
        df = df[df.index < TRAIN_CUTOFF_DATE]
        rp = LABEL_DIR / f"{sym}_regime_h1.parquet"
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
        if n < 50:
            continue

        oof_sym = oof_all[oof_all["coin"] == sym]
        if oof_sym.empty:
            continue
        merged = df.join(oof_sym[["p0", "p1", "p2", "has_oof"]], how="left")
        has_oof = merged["has_oof"].fillna(False).values.astype(bool)
        if has_oof.sum() < 30:
            continue

        oof_proba = np.column_stack([merged["p0"], merged["p1"], merged["p2"]]).astype(np.float64)
        hmm_enc = df["hmm_regime_enc"].values.astype(np.int32)
        h4 = df["h4_trend"].values.astype(np.float64) if "h4_trend" in df.columns else None
        thr_l, thr_s = _build_per_bar_thresholds(hmm_enc, h4, base_cfg, offsets)

        X = np.zeros((n, len(feat_cols)), dtype=np.float64)
        for idx, col in enumerate(feat_cols):
            if col in df.columns:
                X[:, idx] = df[col].ffill().fillna(0).values.astype(np.float64)

        y_pred, confidence = hierarchical_predict(
            None, lgbm_model, lstm_model, lstm_scaler,
            X, feat_cols, [], df,
            model_dir=RUN_DIR, lstm_feat_cols=lstm_feat_cols,
            lgbm_proba=oof_proba,
            per_bar_thr_long=thr_l, per_bar_thr_short=thr_s,
        )
        conf_thr = live_cfg["conf_entry"]
        below = has_oof & (y_pred != 1) & (confidence < conf_thr)
        y_pred[below] = 1
        y_pred[~has_oof] = 1

        y = df["label"].map(LABEL_MAP).values.astype(np.int64)
        atr = df["atr_14_h1"].values if "atr_14_h1" in df.columns else np.ones(n)
        h4t = df["h4_trend"].values if "h4_trend" in df.columns else None
        volr = df["vol_ratio_20"].values if "vol_ratio_20" in df.columns else None
        X_guardian = compute_guardian_static_array(df, g_static)

        report = full_trading_report(
            y_pred=y_pred, y_actual=y, atr=atr,
            close=df["close"].values, high=df["high"].values, low=df["low"].values,
            h4_swing_highs=df["h4_swing_high"].values if "h4_swing_high" in df.columns else None,
            h4_swing_lows=df["h4_swing_low"].values if "h4_swing_low" in df.columns else None,
            index=df.index, modal=MODAL_PER_TRADE, leverages=LEVERAGE_SIM,
            fee_per_side=FEE_PER_SIDE, slippage=SLIPPAGE_PER_SIDE,
            min_rr=SWING_LABEL_MIN_RR, min_tp_atr=SWING_LABEL_MIN_TP, max_sl_atr=SWING_LABEL_MAX_SL,
            tp_fallback_atr=TP_SL_FALLBACK_TP, sl_fallback_atr=TP_SL_FALLBACK_SL,
            max_hold=MAX_HOLDING_BARS, symbol=sym, confidence=confidence,
            guardian_model=guardian_model, guardian_scaler=guardian_scaler,
            X_guardian=X_guardian, guardian_exit_threshold=live_cfg["gdn_exit"],
            guardian_min_hold_bars=live_cfg["gdn_min_hold"],
            trailing_stop_enabled=TRAILING_STOP_ENABLED,
            trailing_stop_atr=TRAILING_STOP_ATR, trailing_stop_min_bars=TRAILING_STOP_MIN_BARS,
            h4_trend=h4t, vol_ratio=volr,
        )
        all_trades.extend(report.get("trades", []))

    agg = _aggregate(all_trades)
    logger.info(f"  [{name}] {agg.get('total_trades', 0)} trades | WR={agg.get('win_rate', 0):.1f}% | "
                f"PPT=${agg.get('pnl_per_trade', 0):+.4f} | PF={agg.get('profit_factor', 0):.2f}")
    return {"name": name, "aggregate": agg, "base_cfg": {str(k): list(v) for k, v in base_cfg.items()},
            "offsets": offsets}


def main():
    if not ADAPTIVE_PATH.exists():
        raise FileNotFoundError(f"Missing {ADAPTIVE_PATH} — run 05f first")

    with open(ADAPTIVE_PATH, encoding="utf-8") as f:
        adapt = json.load(f)

    live_cfg = _apply_live_config()
    oof_all = pd.read_parquet(OOF_PATH)
    if not isinstance(oof_all.index, pd.DatetimeIndex):
        oof_all.index = pd.to_datetime(oof_all.index, utc=True)

    with open(MODEL_DIR / "feature_cols_ic32_regime.json", encoding="utf-8") as f:
        feat_cols = json.load(f)
    with open(MODEL_DIR / "feature_cols_lstm_temporal.json", encoding="utf-8") as f:
        lstm_feat_cols = json.load(f)

    guardian_model, guardian_scaler, guardian_feat_cols, gdn_name = _load_guardian()
    _gdn_dyn = set(GUARDIAN_DYNAMIC_FEATURES)
    g_static = [c for c in guardian_feat_cols if c not in _gdn_dyn]

    models = {
        "lgbm": joblib.load(RUN_DIR / "lgbm.pkl"),
        "lstm": load_lstm(MODEL_DIR / "lstm_best.pt", device="cpu"),
        "lstm_scaler": joblib.load(MODEL_DIR / "lstm_scaler.pkl"),
        "lstm_feat": lstm_feat_cols,
        "guard": guardian_model,
        "guard_scaler": guardian_scaler,
        "g_static": g_static,
    }

    base_c = _parse_cfg(adapt["base_config_c"])
    candidates = [
        ("C-static", base_c, None),
        ("A-H4-adaptive", base_c, adapt["phase_a_best_offsets"]),
        ("B-dir-combined", _parse_cfg(adapt["phase_b_combined_cfg"]), None),
        ("C-combined-adaptive", _parse_cfg(adapt["winner_cfg"]), adapt["winner_offsets"]),
    ]

    sep = "=" * 88
    print(f"\n{sep}")
    print("  ic32 — FULL-STACK OOF adaptive HMM threshold candidates")
    print(f"  Guardian: {gdn_name} | FLIP: {live_cfg['flip']}")
    print(f"{sep}\n")

    results = []
    for name, cfg, offsets in candidates:
        print(f"  Evaluating: {name}...", flush=True)
        results.append(eval_candidate(name, cfg, offsets, oof_all, feat_cols, models, live_cfg))

    print(f"\n{sep}")
    print("  FULL-STACK SCORECARD")
    print(f"{sep}")
    print(f"  {'Config':<22} {'Trades':>7} {'WR%':>6} {'PPT':>8} {'PF':>6} {'S/L':>5} {'PnL':>10}")
    print("  " + "-" * 72)
    for r in results:
        a = r["aggregate"]
        print(f"  {r['name']:<22} {a['total_trades']:>7,} {a['win_rate']:>6.1f} "
              f"${a['pnl_per_trade']:>+7.4f} {a['profit_factor']:>6.2f} "
              f"{a['short_long_ratio']:>5.2f} ${a['total_pnl']:>+9.2f}")

    static_ref = None
    if BASELINE_STATIC_PATH.exists():
        with open(BASELINE_STATIC_PATH, encoding="utf-8") as f:
            static_ref = json.load(f)["aggregate"]
        print(f"\n  vs Config C full-stack (08c): {static_ref['total_trades']} trades | "
              f"WR {static_ref['win_rate']:.1f}% | PPT ${static_ref['pnl_per_trade']:.4f} | "
              f"PF {static_ref['profit_factor']:.2f}")

    out = {
        "meta": {
            "run": "ic32_regime_v1",
            "methodology": "genuine_oof_full_stack_adaptive_hmm",
            "adaptive_source": str(ADAPTIVE_PATH),
            "guardian_model": gdn_name,
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        },
        "candidates": results,
        "vs_config_c_static_full_stack": static_ref,
    }
    out_path = RUN_DIR / "oof_adaptive_full_stack_scorecard.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\n  Saved -> {out_path}")
    print(f"{sep}\n")


if __name__ == "__main__":
    main()