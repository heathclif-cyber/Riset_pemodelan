"""
pipeline/08_oof_ic32_full_stack.py — Full-Stack Genuine OOF for ic32_regime_v1 (live config)

Simulates the EXACT live stack on OOF LGBM predictions:
  OOF LGBM proba -> hard_consensus LSTM -> regime FLIP -> Guardian clean_v2
  + structural/rr/vol filters (via full_trading_report defaults)
  NO dynamic sizing (fixed $10 / 5x)

Prerequisite:
  python pipeline/04_train_lgbm_ic32_genuine_oof.py

Jalankan:
  python pipeline/08_oof_ic32_full_stack.py
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

logger = setup_logger("08_ic32_oof")
RUN_DIR = MODEL_DIR / "runs" / "ic32_regime_v1"
OOF_PATH = RUN_DIR / "oof_predictions.parquet"
INF_CFG_PATH = MODEL_DIR / "inference_config.json"


def _apply_live_config():
    """Override config.py defaults with exact live inference_config snapshot."""
    with open(INF_CFG_PATH, encoding="utf-8") as f:
        cfg = json.load(f)
    cascade = cfg.get("cascade", {})
    guardian = cfg.get("guardian", {})
    ra = cfg.get("regime_alignment", {})

    import config as project_config
    project_config.LGBM_THRESHOLD_LONG = float(cascade.get("lgbm_threshold_long", 0.69))
    project_config.LGBM_THRESHOLD_SHORT = float(cascade.get("lgbm_threshold_short", 0.59))
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
        "thr_long": project_config.LGBM_THRESHOLD_LONG,
        "thr_short": project_config.LGBM_THRESHOLD_SHORT,
        "conf_entry": project_config.CONFIDENCE_THRESHOLD_ENTRY,
        "gdn_exit": project_config.GUARDIAN_EXIT_THRESHOLD,
        "gdn_min_hold": project_config.GUARDIAN_MIN_HOLD_BARS,
        "flip": project_config.REGIME_AWARE_ALIGNMENT,
    }


def _load_guardian():
    """Prefer guardian_clean_v2 (ic32 holdout baseline), fallback guardian_best."""
    for model_name, feat_name, scaler_name in (
        ("guardian_clean_v2.pkl", "guardian_clean_v2_feature_cols.json", "guardian_clean_v2_scaler.pkl"),
        ("guardian_best.pkl", "guardian_feature_cols.json", "guardian_scaler.pkl"),
    ):
        mp = MODEL_DIR / model_name
        fp = MODEL_DIR / feat_name
        sp = MODEL_DIR / scaler_name
        if mp.exists() and fp.exists() and sp.exists():
            with open(fp, encoding="utf-8") as f:
                feat_cols = json.load(f)
            return joblib.load(mp), joblib.load(sp), feat_cols, model_name
    raise FileNotFoundError("Guardian model files not found")


def backtest_coin_oof(sym: str, oof_all: pd.DataFrame, feat_cols: list,
                      lgbm_model, lstm_model, lstm_scaler, lstm_feat_cols,
                      guardian_model, guardian_scaler, g_static, live_cfg) -> dict | None:
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
            for col in ["hmm_regime_enc", "hmm_regime"]:
                if col in df.columns:
                    df = df.drop(columns=[col])
            cols = [c for c in ["hmm_regime_enc", "hmm_regime"] if c in reg.columns]
            df = df.join(reg[cols], how="left")
            df["hmm_regime_enc"] = df["hmm_regime_enc"].fillna(1).astype("int32")
            if "hmm_regime" in df.columns:
                df["hmm_regime"] = df["hmm_regime"].fillna("RANGING_LOW_VOL")
        except Exception:
            pass
    else:
        df["hmm_regime_enc"] = 1
        df["hmm_regime"] = "RANGING_LOW_VOL"

    mask = df["label"].astype(str).isin(LABEL_MAP)
    df = df[mask].copy()
    n = len(df)
    if n < 50:
        return None

    oof_sym = oof_all[oof_all["coin"] == sym]
    if oof_sym.empty:
        return None

    merged = df.join(oof_sym[["p0", "p1", "p2", "has_oof"]], how="left")
    has_oof = merged["has_oof"].fillna(False).values.astype(bool)
    if has_oof.sum() < 30:
        return None

    oof_proba = np.column_stack([
        merged["p0"].values,
        merged["p1"].values,
        merged["p2"].values,
    ]).astype(np.float64)

    X = np.zeros((n, len(feat_cols)), dtype=np.float64)
    for idx, col in enumerate(feat_cols):
        if col in df.columns:
            X[:, idx] = df[col].ffill().fillna(0).values.astype(np.float64)

    y_pred, confidence = hierarchical_predict(
        None, lgbm_model, lstm_model, lstm_scaler,
        X, feat_cols, [], df,
        model_dir=RUN_DIR,
        lstm_feat_cols=lstm_feat_cols,
        lgbm_proba=oof_proba,
    )

    conf_thr = live_cfg["conf_entry"]
    below = has_oof & (y_pred != 1) & (confidence < conf_thr)
    y_pred[below] = 1
    y_pred[~has_oof] = 1
    confidence[~has_oof] = 0.0

    y = df["label"].map(LABEL_MAP).values.astype(np.int64)
    atr = df["atr_14_h1"].values if "atr_14_h1" in df.columns else np.ones(n)
    close = df["close"].values
    high = df["high"].values
    low = df["low"].values
    h4_sh = df["h4_swing_high"].values if "h4_swing_high" in df.columns else None
    h4_sl = df["h4_swing_low"].values if "h4_swing_low" in df.columns else None
    h4t = df["h4_trend"].values if "h4_trend" in df.columns else None
    volr = df["vol_ratio_20"].values if "vol_ratio_20" in df.columns else None

    X_guardian = compute_guardian_static_array(df, g_static)

    report = full_trading_report(
        y_pred=y_pred,
        y_actual=y,
        atr=atr,
        close=close,
        high=high,
        low=low,
        h4_swing_highs=h4_sh,
        h4_swing_lows=h4_sl,
        index=df.index,
        modal=MODAL_PER_TRADE,
        leverages=LEVERAGE_SIM,
        fee_per_side=FEE_PER_SIDE,
        slippage=SLIPPAGE_PER_SIDE,
        min_rr=SWING_LABEL_MIN_RR,
        min_tp_atr=SWING_LABEL_MIN_TP,
        max_sl_atr=SWING_LABEL_MAX_SL,
        tp_fallback_atr=TP_SL_FALLBACK_TP,
        sl_fallback_atr=TP_SL_FALLBACK_SL,
        max_hold=MAX_HOLDING_BARS,
        symbol=sym,
        confidence=confidence,
        guardian_model=guardian_model,
        guardian_scaler=guardian_scaler,
        X_guardian=X_guardian,
        guardian_exit_threshold=live_cfg["gdn_exit"],
        guardian_min_hold_bars=live_cfg["gdn_min_hold"],
        trailing_stop_enabled=TRAILING_STOP_ENABLED,
        trailing_stop_atr=TRAILING_STOP_ATR,
        trailing_stop_min_bars=TRAILING_STOP_MIN_BARS,
        h4_trend=h4t,
        vol_ratio=volr,
    )
    report["n_oof_bars"] = int(has_oof.sum())
    report["n_filtered_conf"] = int(below.sum())
    return report


def main():
    if not OOF_PATH.exists():
        raise FileNotFoundError(
            f"{OOF_PATH} missing — run pipeline/04_train_lgbm_ic32_genuine_oof.py first"
        )

    live_cfg = _apply_live_config()
    oof_all = pd.read_parquet(OOF_PATH)
    if not isinstance(oof_all.index, pd.DatetimeIndex):
        oof_all.index = pd.to_datetime(oof_all.index, utc=True)

    with open(MODEL_DIR / "feature_cols_ic32_regime.json", encoding="utf-8") as f:
        feat_cols = json.load(f)
    with open(MODEL_DIR / "feature_cols_lstm_temporal.json", encoding="utf-8") as f:
        lstm_feat_cols = json.load(f)

    lgbm_model = joblib.load(RUN_DIR / "lgbm.pkl")
    lstm_model = load_lstm(MODEL_DIR / "lstm_best.pt", device="cpu")
    lstm_scaler = joblib.load(MODEL_DIR / "lstm_scaler.pkl")
    guardian_model, guardian_scaler, guardian_feat_cols, gdn_name = _load_guardian()

    _gdn_dyn = set(GUARDIAN_DYNAMIC_FEATURES)
    g_static = [c for c in guardian_feat_cols if c not in _gdn_dyn]

    sep = "=" * 80
    print(f"\n{sep}")
    print("  ic32_regime_v1 — FULL-STACK GENUINE OOF (live config)")
    print(f"  Period  : training until {TRAIN_CUTOFF_DATE.date()}")
    print(f"  LGBM thr: {live_cfg['thr_long']}/{live_cfg['thr_short']} | CONF: {live_cfg['conf_entry']}")
    print(f"  Guardian: {gdn_name} exit={live_cfg['gdn_exit']} min_hold={live_cfg['gdn_min_hold']}")
    print(f"  FLIP    : {live_cfg['flip']} | Dynamic size: OFF")
    print(f"{sep}\n")

    results = {}
    all_trades = []
    success, failed = [], []

    for sym in ALL_COINS:
        try:
            r = backtest_coin_oof(
                sym, oof_all, feat_cols,
                lgbm_model, lstm_model, lstm_scaler, lstm_feat_cols,
                guardian_model, guardian_scaler, g_static, live_cfg,
            )
            if r is None:
                failed.append(sym)
                continue
            results[sym] = r
            success.append(sym)
            trades = r.get("trades", [])
            all_trades.extend(trades)
            pnl = sum(t.get("net_pnl", 0) for t in trades)
            n_tr = r.get("total_trades", 0)
            wr = r.get("winrate", 0) * 100
            logger.info(
                f"  [{sym}] {n_tr} trades | WR={wr:.1f}% | PnL=${pnl:+.2f} | "
                f"oof_bars={r.get('n_oof_bars', 0)}"
            )
        except Exception as exc:
            import traceback
            logger.error(f"  [{sym}] Error: {exc}")
            logger.error(traceback.format_exc())
            failed.append(sym)

    if not all_trades:
        logger.error("No OOF trades generated.")
        sys.exit(1)

    n_total = len(all_trades)
    n_wins = sum(1 for t in all_trades if t.get("net_pnl", 0) > 0)
    total_pnl = sum(t.get("net_pnl", 0) for t in all_trades)
    long_trades = [t for t in all_trades if t.get("direction") == "LONG"]
    short_trades = [t for t in all_trades if t.get("direction") == "SHORT"]
    n_long_win = sum(1 for t in long_trades if t.get("net_pnl", 0) > 0)
    n_short_win = sum(1 for t in short_trades if t.get("net_pnl", 0) > 0)

    wins_pnl = [t["net_pnl"] for t in all_trades if t.get("net_pnl", 0) > 0]
    losses_pnl = [t["net_pnl"] for t in all_trades if t.get("net_pnl", 0) <= 0]
    gross_profit = sum(wins_pnl)
    gross_loss = abs(sum(losses_pnl))
    pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    outcome_counts = {}
    for t in all_trades:
        oc = t.get("outcome", "UNKNOWN")
        outcome_counts[oc] = outcome_counts.get(oc, 0) + 1

    sl_hits = sum(
        1 for t in all_trades
        if "SL" in str(t.get("outcome", "")) or t.get("exit_reason") == "sl"
    )
    gd_exits = sum(1 for t in all_trades if "GUARDIAN" in str(t.get("outcome", "")))

    wr_pct = n_wins / n_total * 100
    long_wr = n_long_win / len(long_trades) * 100 if long_trades else 0
    short_wr = n_short_win / len(short_trades) * 100 if short_trades else 0
    long_pct = len(long_trades) / n_total * 100

    train_years = (TRAIN_CUTOFF_DATE.year - 2020) + (TRAIN_CUTOFF_DATE.month - 1) / 12
    trades_per_month = n_total / max(train_years * 12, 1)

    print(f"\n{sep}")
    print("  OOF SCORECARD — ic32_regime_v1 full stack (live config)")
    print(f"{sep}")
    print(f"  Total Trades     : {n_total:,}")
    print(f"  Trades/bulan     : {trades_per_month:.0f}")
    print(f"  Win Rate         : {wr_pct:.1f}%")
    print(f"  LONG WR          : {long_wr:.1f}% ({len(long_trades)} trades, {long_pct:.1f}%)")
    print(f"  SHORT WR         : {short_wr:.1f}% ({len(short_trades)} trades)")
    print(f"  Net PnL          : ${total_pnl:+.2f}")
    print(f"  PnL/trade        : ${total_pnl / n_total:+.4f}")
    print(f"  Profit Factor    : {pf:.3f}")
    print(f"  SL hit rate      : {sl_hits / n_total * 100:.1f}%")
    print(f"  Guardian exits   : {gd_exits / n_total * 100:.1f}%")

    holdout_ref = RUN_DIR / "holdout_apr_jun26.json"
    if holdout_ref.exists():
        with open(holdout_ref, encoding="utf-8") as f:
            h = json.load(f)["aggregate"]
        print(f"\n  vs Holdout archived (Apr-Jun 2026):")
        print(f"    Holdout: {h['total_trades']} trades | WR {h['win_rate']:.1f}% | PF {h['profit_factor']:.2f}")
        print(f"    OOF    : {n_total} trades | WR {wr_pct:.1f}% | PF {pf:.2f}")

    out = {
        "meta": {
            "run": "ic32_regime_v1",
            "methodology": "genuine_oof_full_stack",
            "period": f"2020-01-01 to {TRAIN_CUTOFF_DATE.date()}",
            "live_config_source": str(INF_CFG_PATH),
            "guardian_model": gdn_name,
            "dynamic_sizing": False,
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        },
        "live_params": live_cfg,
        "aggregate": {
            "total_trades": n_total,
            "trades_per_month": round(trades_per_month, 1),
            "win_rate": round(wr_pct, 2),
            "long_wr": round(long_wr, 2),
            "short_wr": round(short_wr, 2),
            "long_pct": round(long_pct, 2),
            "total_pnl": round(total_pnl, 2),
            "pnl_per_trade": round(total_pnl / n_total, 4),
            "profit_factor": round(pf, 3),
            "sl_rate_pct": round(sl_hits / n_total * 100, 2),
            "guardian_exit_pct": round(gd_exits / n_total * 100, 2),
            "outcome_counts": outcome_counts,
        },
        "per_coin": {
            sym: {
                "trades": r.get("total_trades", 0),
                "wr": round(r.get("winrate", 0) * 100, 2),
                "pnl": round(sum(t.get("net_pnl", 0) for t in r.get("trades", [])), 2),
                "oof_bars": r.get("n_oof_bars", 0),
            }
            for sym, r in results.items()
        },
        "failed": failed,
    }

    out_path = RUN_DIR / "oof_full_stack_scorecard.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    trades_path = RUN_DIR / "oof_trade_history.csv"
    pd.DataFrame(all_trades).to_csv(trades_path, index=False)

    print(f"\n  Saved -> {out_path}")
    print(f"  Saved -> {trades_path}")
    print(f"{sep}\n")


if __name__ == "__main__":
    main()