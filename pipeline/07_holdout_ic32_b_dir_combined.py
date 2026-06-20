"""
pipeline/07_holdout_ic32_b_dir_combined.py — Holdout eval SEKALI (B-dir-combined)

Frozen HMM per-state thresholds (direction-aware) dari OOF sweep.
Tidak boleh diulang / tune pada holdout.

Prerequisite:
  data/holdout-test/labeled/*_features_v3.parquet
  models/runs/ic32_regime_v1/b_dir_combined_frozen.json

Jalankan:
  python pipeline/07_holdout_ic32_b_dir_combined.py
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

logger = setup_logger("07_ic32_b_dir_holdout")
HOLDOUT_LABEL_DIR = HOLDOUT_DIR / "labeled"
RUN_DIR = MODEL_DIR / "runs" / "ic32_regime_v1"
FROZEN_PATH = RUN_DIR / "b_dir_combined_frozen.json"
INF_CFG_PATH = MODEL_DIR / "inference_config.json"
PROD_HOLDOUT_REF = RUN_DIR / "holdout_apr_jun26.json"
OUT_PATH = RUN_DIR / "holdout_b_dir_combined_apr_jun26.json"
HOLDOUT_EVAL_FLAG = RUN_DIR / ".holdout_b_dir_combined_evaluated"

HOLDOUT_MONTHS = 2.5


def _guard_already_evaluated():
    if HOLDOUT_EVAL_FLAG.exists():
        raise RuntimeError(
            "Holdout B-dir-combined sudah dievaluasi (Aturan 1).\n"
            f"Flag: {HOLDOUT_EVAL_FLAG}\n"
            "Jangan jalankan ulang. Buat periode holdout baru jika perlu eval lagi."
        )


def _apply_live_config():
    with open(INF_CFG_PATH, encoding="utf-8") as f:
        cfg = json.load(f)
    cascade = cfg.get("cascade", {})
    guardian = cfg.get("guardian", {})
    ra = cfg.get("regime_alignment", {})

    import config as project_config
    project_config.CONFIDENCE_THRESHOLD_ENTRY = float(
        cascade.get("confidence_threshold_entry", 0.59)
    )
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


def _load_frozen_cfg() -> dict:
    with open(FROZEN_PATH, encoding="utf-8") as f:
        data = json.load(f)
    raw = data["per_state_thresholds"]
    return {int(k): (float(v[0]), float(v[1])) for k, v in raw.items()}


def _build_per_bar_thresholds(hmm_enc: np.ndarray, hmm_cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    n = len(hmm_enc)
    default_tl, default_ts = hmm_cfg[-1]
    tl_arr = np.full(n, default_tl, dtype=np.float64)
    ts_arr = np.full(n, default_ts, dtype=np.float64)
    for state, (tl, ts) in hmm_cfg.items():
        if state == -1:
            continue
        mask = hmm_enc == state
        tl_arr[mask] = tl
        ts_arr[mask] = ts
    return tl_arr, ts_arr


def backtest_coin(sym: str, hmm_cfg: dict, live_cfg: dict,
                  lgbm_model, lstm_model, lstm_scaler, lstm_feat_cols,
                  feat_cols, guardian_model, guardian_scaler, g_static) -> dict | None:
    p = HOLDOUT_LABEL_DIR / f"{sym}_features_v3.parquet"
    if not p.exists():
        return None

    df = pd.read_parquet(p)
    df = ensure_utc_index(df).sort_index()

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
            if "hmm_regime" in df.columns:
                df["hmm_regime"] = df["hmm_regime"].fillna("RANGING_LOW_VOL")
        except Exception:
            df["hmm_regime_enc"] = 1
    else:
        df["hmm_regime_enc"] = 1
        df["hmm_regime"] = "RANGING_LOW_VOL"

    # Apply same training parity overrides seperti live inference
    # (LSR clip ke ~1.0, CVD features clamp ke ±3σ training distribution)
    df = apply_training_parity(df)

    mask = df["label"].astype(str).isin(LABEL_MAP)
    df = df[mask].copy()
    n = len(df)
    if n < 30:
        return None

    X = np.zeros((n, len(feat_cols)), dtype=np.float64)
    for idx, col in enumerate(feat_cols):
        if col in df.columns:
            X[:, idx] = df[col].ffill().fillna(0).values.astype(np.float64)

    hmm_enc = df["hmm_regime_enc"].values.astype(np.int32)
    thr_l, thr_s = _build_per_bar_thresholds(hmm_enc, hmm_cfg)

    y_pred, confidence = hierarchical_predict(
        None, lgbm_model, lstm_model, lstm_scaler,
        X, feat_cols, [], df,
        model_dir=RUN_DIR,
        lstm_feat_cols=lstm_feat_cols,
        per_bar_thr_long=thr_l,
        per_bar_thr_short=thr_s,
    )

    conf_thr = live_cfg["conf_entry"]
    below = (y_pred != 1) & (confidence < conf_thr)
    y_pred[below] = 1

    y = df["label"].map(LABEL_MAP).values.astype(np.int64)
    atr = df["atr_14_h1"].values if "atr_14_h1" in df.columns else np.ones(n)
    h4t = df["h4_trend"].values if "h4_trend" in df.columns else None
    volr = df["vol_ratio_20"].values if "vol_ratio_20" in df.columns else None
    X_guardian = compute_guardian_static_array(df, g_static)

    report = full_trading_report(
        y_pred=y_pred, y_actual=y,
        atr=atr, close=df["close"].values, high=df["high"].values, low=df["low"].values,
        h4_swing_highs=df["h4_swing_high"].values if "h4_swing_high" in df.columns else None,
        h4_swing_lows=df["h4_swing_low"].values if "h4_swing_low" in df.columns else None,
        index=df.index, modal=MODAL_PER_TRADE, leverages=LEVERAGE_SIM,
        fee_per_side=FEE_PER_SIDE, slippage=SLIPPAGE_PER_SIDE,
        min_rr=SWING_LABEL_MIN_RR, min_tp_atr=SWING_LABEL_MIN_TP, max_sl_atr=SWING_LABEL_MAX_SL,
        tp_fallback_atr=TP_SL_FALLBACK_TP, sl_fallback_atr=TP_SL_FALLBACK_SL,
        max_hold=MAX_HOLDING_BARS, symbol=sym, confidence=confidence,
        guardian_model=guardian_model, guardian_scaler=guardian_scaler,
        X_guardian=X_guardian,
        guardian_exit_threshold=live_cfg["gdn_exit"],
        guardian_min_hold_bars=live_cfg["gdn_min_hold"],
        trailing_stop_enabled=TRAILING_STOP_ENABLED,
        trailing_stop_atr=TRAILING_STOP_ATR,
        trailing_stop_min_bars=TRAILING_STOP_MIN_BARS,
        h4_trend=h4t, vol_ratio=volr,
    )
    report["n_filtered"] = int(below.sum())
    report["n_bars"] = n
    return report


def main():
    _guard_already_evaluated()

    if not FROZEN_PATH.exists():
        raise FileNotFoundError(f"Missing {FROZEN_PATH}")

    hmm_cfg = _load_frozen_cfg()
    live_cfg = _apply_live_config()

    with open(MODEL_DIR / "feature_cols_ic32_regime.json", encoding="utf-8") as f:
        feat_cols = json.load(f)
    with open(MODEL_DIR / "feature_cols_lstm_temporal.json", encoding="utf-8") as f:
        lstm_feat_cols = json.load(f)

    lgbm_model = joblib.load(RUN_DIR / "lgbm.pkl")
    lstm_model = load_lstm(MODEL_DIR / "lstm_best.pt", device="cpu")
    lstm_scaler = joblib.load(MODEL_DIR / "lstm_scaler.pkl")
    guardian_model = joblib.load(MODEL_DIR / "guardian_clean_v2.pkl")
    guardian_scaler = joblib.load(MODEL_DIR / "guardian_clean_v2_scaler.pkl")
    with open(MODEL_DIR / "guardian_clean_v2_feature_cols.json", encoding="utf-8") as f:
        guardian_feat_cols = json.load(f)
    g_static = [c for c in guardian_feat_cols if c not in set(GUARDIAN_DYNAMIC_FEATURES)]

    sep = "=" * 80
    print(f"\n{sep}")
    print("  ic32 B-dir-combined — HOLDOUT Apr-Jun 2026 (SEKALI)")
    print(f"  HMM thr: {hmm_cfg}")
    print(f"  CONF={live_cfg['conf_entry']} GDN={live_cfg['gdn_exit']} FLIP={live_cfg['flip']}")
    print(f"{sep}\n")

    results = {}
    all_trades = []
    success, failed = [], []

    for sym in ALL_COINS:
        try:
            r = backtest_coin(
                sym, hmm_cfg, live_cfg,
                lgbm_model, lstm_model, lstm_scaler, lstm_feat_cols,
                feat_cols, guardian_model, guardian_scaler, g_static,
            )
            if r is None:
                failed.append(sym)
                continue
            results[sym] = r
            success.append(sym)
            trades = r.get("trades", [])
            all_trades.extend(trades)
            pnl = sum(t.get("net_pnl", 0) for t in trades)
            logger.info(
                f"  [{sym}] {r.get('total_trades', 0)} trades | "
                f"WR={r.get('winrate', 0)*100:.1f}% | PnL=${pnl:+.2f}"
            )
        except Exception as exc:
            import traceback
            logger.error(f"  [{sym}] Error: {exc}")
            logger.error(traceback.format_exc())
            failed.append(sym)

    if not all_trades:
        logger.error("No holdout trades.")
        sys.exit(1)

    n_total = len(all_trades)
    n_wins = sum(1 for t in all_trades if t.get("net_pnl", 0) > 0)
    total_pnl = sum(t.get("net_pnl", 0) for t in all_trades)
    long_trades = [t for t in all_trades if t.get("direction") == "LONG"]
    short_trades = [t for t in all_trades if t.get("direction") == "SHORT"]
    gpnl = sum(t["net_pnl"] for t in all_trades if t.get("net_pnl", 0) > 0)
    gloss = abs(sum(t["net_pnl"] for t in all_trades if t.get("net_pnl", 0) <= 0))
    pf = gpnl / gloss if gloss > 0 else float("inf")
    wr_pct = n_wins / n_total * 100
    ppt = total_pnl / n_total

    outcome_counts = {}
    for t in all_trades:
        oc = t.get("outcome", "UNKNOWN")
        outcome_counts[oc] = outcome_counts.get(oc, 0) + 1
    gd_rate = sum(v for k, v in outcome_counts.items() if "GUARDIAN" in k) / n_total * 100

    print(f"\n{sep}")
    print("  HOLDOUT SCORECARD — B-dir-combined")
    print(f"{sep}")
    print(f"  Total Trades   : {n_total:,}")
    print(f"  Trades/bulan   : {n_total / HOLDOUT_MONTHS:.0f}")
    print(f"  Win Rate       : {wr_pct:.1f}%")
    print(f"  LONG %         : {len(long_trades)/n_total*100:.1f}%")
    print(f"  S/L ratio      : {len(short_trades)/max(len(long_trades),1):.2f}")
    print(f"  Net PnL        : ${total_pnl:+.2f}")
    print(f"  PnL/trade      : ${ppt:+.4f}")
    print(f"  Profit Factor  : {pf:.3f}")
    print(f"  Guardian exits : {gd_rate:.1f}%")

    prod_ref = None
    if PROD_HOLDOUT_REF.exists():
        with open(PROD_HOLDOUT_REF, encoding="utf-8") as f:
            prod_ref = json.load(f)["aggregate"]
        print(f"\n  vs Production holdout (0.69/0.59):")
        print(f"    Prod : {prod_ref['total_trades']} trades | WR {prod_ref['win_rate']:.1f}% | "
              f"PPT ${prod_ref['pnl_per_trade']:.4f} | PF {prod_ref['profit_factor']:.2f} | "
              f"PnL ${prod_ref['total_pnl']:+.2f}")
        print(f"    B-dir: {n_total} trades | WR {wr_pct:.1f}% | "
              f"PPT ${ppt:.4f} | PF {pf:.2f} | PnL ${total_pnl:+.2f}")

        min_trades = int(prod_ref["total_trades"] * 0.8)
        upgrade = {
            "wr_pass": wr_pct >= prod_ref["win_rate"],
            "pf_pass": pf >= prod_ref["profit_factor"],
            "trades_pass": n_total >= min_trades,
        }
        print(f"\n  Kriteria upgrade vs production holdout:")
        print(f"    WR >= {prod_ref['win_rate']:.1f}%: {'PASS' if upgrade['wr_pass'] else 'FAIL'} ({wr_pct:.1f}%)")
        print(f"    PF >= {prod_ref['profit_factor']:.2f}: {'PASS' if upgrade['pf_pass'] else 'FAIL'} ({pf:.2f})")
        print(f"    Trades >= {min_trades}: {'PASS' if upgrade['trades_pass'] else 'FAIL'} ({n_total})")

    out = {
        "meta": {
            "run": "ic32_b_dir_combined",
            "holdout_period": "2026-04-01 to 2026-06-13",
            "frozen_config": str(FROZEN_PATH),
            "hmm_per_state_thresholds": {str(k): list(v) for k, v in hmm_cfg.items()},
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        },
        "aggregate": {
            "total_trades": n_total,
            "trades_per_month": round(n_total / HOLDOUT_MONTHS, 1),
            "win_rate": round(wr_pct, 2),
            "long_pct": round(len(long_trades) / n_total * 100, 2),
            "short_long_ratio": round(len(short_trades) / max(len(long_trades), 1), 3),
            "total_pnl": round(total_pnl, 2),
            "pnl_per_trade": round(ppt, 4),
            "profit_factor": round(pf, 3),
            "guardian_exit_pct": round(gd_rate, 2),
            "outcome_counts": outcome_counts,
        },
        "vs_production_holdout": prod_ref,
        "per_coin": {
            sym: {
                "trades": r.get("total_trades", 0),
                "wr": round(r.get("winrate", 0) * 100, 2),
                "pnl": round(sum(t.get("net_pnl", 0) for t in r.get("trades", [])), 2),
            }
            for sym, r in results.items()
        },
        "failed": failed,
    }

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    HOLDOUT_EVAL_FLAG.write_text(datetime.now().isoformat(), encoding="utf-8")

    print(f"\n  Saved -> {OUT_PATH}")
    print(f"  Flag  -> {HOLDOUT_EVAL_FLAG}")
    print(f"{sep}\n")


if __name__ == "__main__":
    main()