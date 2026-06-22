"""
pipeline/07_holdout_backtest_tb.py — Holdout Backtest Triple Barrier LGBM

Membandingkan TB LGBM (widyawardhana_v1) vs ic32_regime_v1 dalam kondisi
IDENTIK: standalone LGBM only, static TP/SL, no LSTM, no Guardian.

Jalankan:
  python pipeline/07_holdout_backtest_tb.py
  python pipeline/07_holdout_backtest_tb.py --coins SOLUSDT ETHUSDT
  python pipeline/07_holdout_backtest_tb.py --baseline  # re-run baseline juga
"""

import argparse
import json
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from core.features import triple_barrier_labeling
from core.evaluator import full_trading_report
from core.utils import setup_logger, ensure_utc_index

from config import (
    TRAINING_COINS, ALL_COINS,
    LABEL_DIR, HOLDOUT_DIR, MODEL_DIR,
    MODAL_PER_TRADE, LEVERAGE_SIM, FEE_PER_SIDE, SLIPPAGE_PER_SIDE,
    MAX_HOLDING_BARS,
    SWING_LABEL_MIN_RR, SWING_LABEL_MIN_TP, SWING_LABEL_MAX_SL,
    TP_SL_FALLBACK_TP, TP_SL_FALLBACK_SL,
    LGBM_THRESHOLD_LONG, LGBM_THRESHOLD_SHORT, CONFIDENCE_THRESHOLD_ENTRY,
)

logger = setup_logger("07_holdout_backtest_tb")

RUN_NAME = "tb_lgbm_widyawardhana_v1"
HOLDOUT_LABEL_DIR = HOLDOUT_DIR / "labeled"
LABEL_MAP = {"SHORT": 0, "FLAT": 1, "LONG": 2}
TB_LABEL_MAP = {"SHORT": 0, "FLAT": 1, "LONG": 2}


# ═══════════════════════════════════════════════════════════════════════════════
# HELPER
# ═══════════════════════════════════════════════════════════════════════════════

def load_model_and_features(run_dir: Path) -> tuple:
    """Load trained TB LGBM model and its feature list."""
    model_path = run_dir / "lgbm.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    model = joblib.load(model_path)

    feat_path = run_dir / f"{RUN_NAME}_keep_features.json"
    if not feat_path.exists():
        raise FileNotFoundError(f"Feature list not found: {feat_path}")

    with open(feat_path) as f:
        features = json.load(f)

    logger.info(f"Model loaded: {model_path}")
    logger.info(f"Features: {len(features)}")
    return model, features


def load_baseline_model_and_features() -> tuple:
    """Load ic32_regime_v1 baseline model and its feature list."""
    model_path = MODEL_DIR / "lgbm_baseline.pkl"
    if not model_path.exists():
        # Try from run directory
        model_path = MODEL_DIR / "runs" / "ic32_regime_v1" / "lgbm.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"Baseline model not found. Tried: {MODEL_DIR}/lgbm_baseline.pkl")

    model = joblib.load(model_path)

    # Use model's own feature_name_ (exact features used during training)
    if hasattr(model, 'feature_name_') and model.feature_name_:
        features = list(model.feature_name_)
    else:
        # Fallback: try JSON
        feat_path = MODEL_DIR / "feature_cols_v2.json"
        if not feat_path.exists():
            feat_path = MODEL_DIR / "runs" / "ic32_regime_v1" / "feature_cols_v2.json"
        with open(feat_path) as f:
            features = json.load(f)

    logger.info(f"Baseline model loaded: {model_path}")
    logger.info(f"Baseline features: {len(features)}")
    return model, features


def predict_lgbm_standalone(model, X: np.ndarray, feat_cols: list,
                            df: pd.DataFrame) -> tuple:
    """
    Simple LGBM-only prediction (no cascade, no LSTM).
    Returns (y_pred, confidence) arrays.

    y_pred: 0=SHORT, 1=FLAT, 2=LONG
    """
    # Build feature matrix
    X_infer = np.zeros((len(df), len(feat_cols)), dtype=np.float64)
    for idx, col in enumerate(feat_cols):
        if col in df.columns:
            X_infer[:, idx] = df[col].ffill().fillna(0).values.astype(np.float64)

    proba = model.predict_proba(X_infer)  # shape: (n, 3) → SHORT, FLAT, LONG
    confidence = np.max(proba, axis=1)
    y_pred = np.argmax(proba, axis=1)

    # Apply threshold filtering
    # Convert to: if argmax=SHORT and proba < threshold → FLAT
    y_pred_filtered = y_pred.copy()
    for i in range(len(y_pred)):
        if y_pred[i] == 2 and proba[i, 2] < LGBM_THRESHOLD_LONG:  # LONG
            y_pred_filtered[i] = 1  # → FLAT
        elif y_pred[i] == 0 and proba[i, 0] < LGBM_THRESHOLD_SHORT:  # SHORT
            y_pred_filtered[i] = 1  # → FLAT

    # Secondary confidence filter
    below = (y_pred_filtered != 1) & (confidence < CONFIDENCE_THRESHOLD_ENTRY)
    y_pred_filtered[below] = 1

    return y_pred_filtered, confidence


def backtest_symbol(symbol: str, model, feat_cols: list,
                    tp_atr: float, sl_atr: float, max_hold: int) -> Optional[dict]:
    """Run backtest for one symbol."""
    path = HOLDOUT_LABEL_DIR / f"{symbol}_features_v3.parquet"
    if not path.exists():
        logger.warning(f"[{symbol}] Holdout features not found, skip")
        return None

    df = pd.read_parquet(path)
    df = ensure_utc_index(df)
    df = df.sort_index()

    # Generate TB labels for holdout (same as training)
    required = ["close", "high", "low", "atr_14_h1"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        logger.warning(f"[{symbol}] Missing columns: {missing}, skip")
        return None

    tb_labels = triple_barrier_labeling(
        close=df["close"], high=df["high"], low=df["low"],
        atr_base=df["atr_14_h1"],
        tp_atr_mult=tp_atr, sl_atr_mult=sl_atr, max_hold=max_hold,
    )
    df["tb_label"] = tb_labels.map(TB_LABEL_MAP)
    df = df.dropna(subset=["tb_label"])

    mask = df["tb_label"].astype(int).isin([0, 1, 2])
    df = df[mask].copy()
    if len(df) < 50:
        logger.warning(f"[{symbol}] Too few rows after TB labeling: {len(df)}, skip")
        return None

    y_actual = df["tb_label"].astype(np.int32).values

    # Predict
    y_pred, confidence = predict_lgbm_standalone(model, None, feat_cols, df)

    # Trading arrays
    atr_arr = df["atr_14_h1"].values if "atr_14_h1" in df.columns else np.ones(len(df))
    close_arr = df["close"].values if "close" in df.columns else np.ones(len(df))
    high_arr = df["high"].values if "high" in df.columns else close_arr
    low_arr = df["low"].values if "low" in df.columns else close_arr
    h4_sh = df["h4_swing_high"].values if "h4_swing_high" in df.columns else None
    h4_sl = df["h4_swing_low"].values if "h4_swing_low" in df.columns else None
    h4_trend_arr = df["h4_trend"].values if "h4_trend" in df.columns else None

    report = full_trading_report(
        y_pred=y_pred,
        y_actual=y_actual,
        atr=atr_arr,
        close=close_arr,
        high=high_arr,
        low=low_arr,
        h4_swing_highs=h4_sh,
        h4_swing_lows=h4_sl,
        index=df.index,
        modal=MODAL_PER_TRADE,
        leverages=LEVERAGE_SIM,
        fee_per_side=FEE_PER_SIDE,
        slippage=SLIPPAGE_PER_SIDE,
        min_rr=SWING_LABEL_MIN_RR,
        min_tp_atr=SWING_LABEL_MIN_TP,
        max_sl_atr=max_hold,
        tp_fallback_atr=tp_atr,
        sl_fallback_atr=sl_atr,
        max_hold=max_hold,
        symbol=symbol,
        confidence=confidence,
        guardian_enabled=False,  # No Guardian for clean comparison
        trailing_stop_enabled=False,
        h4_trend=h4_trend_arr,
    )

    # Extract key metrics
    lev5 = report.get("lev5x", report)
    trades = lev5.get("trades", []) if isinstance(lev5, dict) else []

    # Count wins/losses
    wins = sum(1 for t in trades if t.get("outcome") == "WIN")
    losses = sum(1 for t in trades if t.get("outcome") == "LOSS")
    timeouts = sum(1 for t in trades if t.get("outcome") == "TIMEOUT")
    total_trades = len(trades)
    wr = wins / max(total_trades, 1) * 100

    # Direction breakdown
    long_trades = [t for t in trades if t.get("direction") == "LONG"]
    short_trades = [t for t in trades if t.get("direction") == "SHORT"]
    long_wr = sum(1 for t in long_trades if t.get("outcome") == "WIN") / max(len(long_trades), 1) * 100
    short_wr = sum(1 for t in short_trades if t.get("outcome") == "WIN") / max(len(short_trades), 1) * 100

    # PnL
    total_pnl = sum(t.get("net_pnl", 0) for t in trades)
    sl_hit = sum(1 for t in trades if t.get("outcome") == "LOSS" and t.get("exit_reason", "").startswith("sl"))
    avg_hold = np.mean([t.get("hold_bars", t.get("bars_held", 0)) for t in trades]) if trades else 0

    result = {
        "symbol": symbol,
        "n_bars": len(df),
        "total_trades": total_trades,
        "win_rate_pct": round(wr, 2),
        "long_wr_pct": round(long_wr, 2),
        "long_trades": len(long_trades),
        "short_wr_pct": round(short_wr, 2),
        "short_trades": len(short_trades),
        "timeout_trades": timeouts,
        "net_pnl": round(total_pnl, 2),
        "sl_hit": sl_hit,
        "avg_hold_bars": round(float(avg_hold), 1),
    }

    logger.info(
        f"[{symbol}] {total_trades} trades | WR={wr:.1f}% | "
        f"LONG={len(long_trades)} ({long_wr:.1f}%) | SHORT={len(short_trades)} ({short_wr:.1f}%) | "
        f"PnL=${total_pnl:+.0f}"
    )
    return result


def compute_aggregate(results: list) -> dict:
    """Compute aggregate metrics across all symbols."""
    valid = [r for r in results if r is not None and r["total_trades"] > 0]
    if not valid:
        return {"error": "No valid results"}

    total_trades = sum(r["total_trades"] for r in valid)
    total_pnl = sum(r["net_pnl"] for r in valid)
    mean_wr = np.mean([r["win_rate_pct"] for r in valid])

    # Long/short aggregate
    total_long = sum(r["long_trades"] for r in valid)
    total_short = sum(r["short_trades"] for r in valid)
    long_wins = sum(
        int(r["long_wr_pct"] * r["long_trades"] / 100) for r in valid if r["long_trades"] > 0
    )
    short_wins = sum(
        int(r["short_wr_pct"] * r["short_trades"] / 100) for r in valid if r["short_trades"] > 0
    )
    agg_long_wr = long_wins / max(total_long, 1) * 100
    agg_short_wr = short_wins / max(total_short, 1) * 100

    # PnL per trade
    pnl_per_trade = total_pnl / max(total_trades, 1)

    # PF
    gross_win = sum(
        sum(t.get("net_pnl", 0) for t in r.get("trades", []) if t.get("net_pnl", 0) > 0)
        for r in [lev5_for_result(x) for x in valid]
    ) if False else sum(r["net_pnl"] for r in valid if r["net_pnl"] > 0)
    gross_loss = abs(sum(r["net_pnl"] for r in valid if r["net_pnl"] < 0))
    pf = gross_win / gross_loss if gross_loss > 0 else 999

    return {
        "n_coins": len(valid),
        "total_trades": total_trades,
        "trades_per_coin": total_trades // max(len(valid), 1),
        "mean_win_rate_pct": round(mean_wr, 2),
        "agg_long_wr_pct": round(agg_long_wr, 2),
        "agg_short_wr_pct": round(agg_short_wr, 2),
        "total_long_trades": total_long,
        "total_short_trades": total_short,
        "total_pnl": round(total_pnl, 2),
        "pnl_per_trade": round(pnl_per_trade, 2),
        "profit_factor": round(pf, 2),
        "per_coin": valid,
    }


def lev5_for_result(result: dict) -> dict:
    """Placeholder — result already contains flattened metrics."""
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="TB LGBM Holdout Backtest")
    parser.add_argument("--all", action="store_true", help="All coins")
    parser.add_argument("--coins", nargs="+", default=None, help="Specific coins")
    parser.add_argument("--baseline", action="store_true",
                        help="Also re-run ic32_regime_v1 baseline standalone")
    parser.add_argument("--tp", type=float, default=TP_SL_FALLBACK_TP)
    parser.add_argument("--sl", type=float, default=TP_SL_FALLBACK_SL)
    parser.add_argument("--max-hold", type=int, default=MAX_HOLDING_BARS)
    args = parser.parse_args()

    if args.coins:
        coins = args.coins
    elif args.all:
        coins = ALL_COINS
    else:
        coins = TRAINING_COINS

    tp_atr = args.tp
    sl_atr = args.sl
    max_hold = args.max_hold

    run_dir = MODEL_DIR / "runs" / RUN_NAME
    sep = "=" * 70

    # ── Verify holdout data ────────────────────────────────────────────────────
    available_coins = []
    for sym in coins:
        path = HOLDOUT_LABEL_DIR / f"{sym}_features_v3.parquet"
        if path.exists():
            available_coins.append(sym)
        else:
            logger.warning(f"[{sym}] Holdout data not found, skip")

    logger.info(f"Holdout coins available: {len(available_coins)}/{len(coins)}")

    # ═══════════════════════════════════════════════════════════════════════════
    # TB LGBM widyawardhana_v1
    # ═══════════════════════════════════════════════════════════════════════════

    print(f"\n{sep}")
    print(f"  HOLDOUT BACKTEST — TB LGBM {RUN_NAME}")
    print(f"{sep}")
    print(f"  TP={tp_atr}xATR  SL={sl_atr}xATR  MaxHold={max_hold}")
    print(f"  Coins: {len(available_coins)} | ${MODAL_PER_TRADE}/trade, {LEVERAGE_SIM[0]}x leverage")
    print(f"  Mode: LGBM standalone (no LSTM, no Guardian)")
    print(f"{sep}\n")

    tb_model, tb_feats = load_model_and_features(run_dir)

    print("Running TB LGBM backtest...")
    tb_results = []
    for sym in available_coins:
        try:
            r = backtest_symbol(sym, tb_model, tb_feats, tp_atr, sl_atr, max_hold)
            tb_results.append(r)
        except Exception as e:
            logger.error(f"[{sym}] Error: {e}")
            import traceback
            traceback.print_exc()

    tb_agg = compute_aggregate(tb_results)

    # ═══════════════════════════════════════════════════════════════════════════
    # BASELINE: ic32_regime_v1 (standalone LGBM only)
    # ═══════════════════════════════════════════════════════════════════════════

    baseline_agg = None
    if args.baseline:
        print(f"\n{sep}")
        print(f"  HOLDOUT BACKTEST — BASELINE ic32_regime_v1 (standalone)")
        print(f"{sep}\n")

        try:
            bl_model, bl_feats = load_baseline_model_and_features()

            print("Running BASELINE backtest...")
            bl_results = []
            for sym in available_coins:
                try:
                    r = backtest_symbol(sym, bl_model, bl_feats, tp_atr, sl_atr, max_hold)
                    bl_results.append(r)
                except Exception as e:
                    logger.error(f"[{sym}] Baseline error: {e}")

            baseline_agg = compute_aggregate(bl_results)

            # Save baseline results
            bl_path = run_dir / "holdout_baseline_ic32_standalone.json"
            with open(bl_path, "w") as f:
                json.dump(baseline_agg, f, indent=2, default=str)
            logger.info(f"Baseline results -> {bl_path}")

        except FileNotFoundError as e:
            logger.warning(f"Baseline model not found: {e}")
            logger.warning("Skipping baseline comparison. Run with --baseline after fixing model path.")

    # ═══════════════════════════════════════════════════════════════════════════
    # SAVE
    # ═══════════════════════════════════════════════════════════════════════════

    # TB results
    tb_path = run_dir / "holdout_backtest_results.json"
    with open(tb_path, "w") as f:
        json.dump(tb_agg, f, indent=2, default=str)
    logger.info(f"TB results -> {tb_path}")

    # ═══════════════════════════════════════════════════════════════════════════
    # HEAD-TO-HEAD COMPARISON
    # ═══════════════════════════════════════════════════════════════════════════

    print(f"\n{sep}")
    print(f"  COMPARISON TABLE")
    print(f"{sep}")

    # Load ic32_regime_v1 reference metrics from model_registry
    registry_path = MODEL_DIR / "model_registry.json"
    registry_metrics = {}
    if registry_path.exists():
        with open(registry_path) as f:
            reg = json.load(f)
        scorecard = reg.get("holdout_scorecard_nov2025_apr2026", {})
        registry_metrics = {
            "total_trades": scorecard.get("total_trades", "-"),
            "win_rate_pct": scorecard.get("win_rate_pct", "-"),
            "net_pnl_usd": scorecard.get("net_pnl_usd", "-"),
            "profit_factor": scorecard.get("profit_factor", "-"),
            "max_drawdown_pct": scorecard.get("max_drawdown_pct", "-"),
            "pnl_per_trade_usd": scorecard.get("pnl_per_trade_usd", "-"),
        }

    # Print comparison
    print(f"\n{'Metric':<30} {'ic32_regime_v1':>18} {'TB widyawardhana_v1':>22}")
    print("-" * 72)
    print(f"{'Coins tested':<30} {len(available_coins):>18} {tb_agg.get('n_coins', 0):>22}")
    print(f"{'Total Trades':<30} {str(registry_metrics.get('total_trades', '-')):>18} "
          f"{str(tb_agg.get('total_trades', '-')):>22}")
    print(f"{'Win Rate %':<30} {str(registry_metrics.get('win_rate_pct', '-')):>18} "
          f"{str(tb_agg.get('mean_win_rate_pct', '-')):>22}")

    if baseline_agg:
        print(f"{'  -> Baseline standalone':<30} "
              f"{str(baseline_agg.get('mean_win_rate_pct', '-')):>18} {'-':>22}")

    print(f"{'LONG WR %':<30} {'-':>18} "
          f"{str(tb_agg.get('agg_long_wr_pct', '-')):>22}")
    print(f"{'SHORT WR %':<30} {'-':>18} "
          f"{str(tb_agg.get('agg_short_wr_pct', '-')):>22}")
    print(f"{'Net PnL ($)':<30} {str(registry_metrics.get('net_pnl_usd', '-')):>18} "
          f"{str(tb_agg.get('total_pnl', '-')):>22}")
    print(f"{'PnL/Trade ($)':<30} {str(registry_metrics.get('pnl_per_trade_usd', '-')):>18} "
          f"{str(tb_agg.get('pnl_per_trade', '-')):>22}")
    print(f"{'Profit Factor':<30} {str(registry_metrics.get('profit_factor', '-')):>18} "
          f"{str(tb_agg.get('profit_factor', '-')):>22}")

    if baseline_agg:
        print(f"{'  -> Baseline standalone':<30} "
              f"{str(baseline_agg.get('total_pnl', '-')):>18} {'-':>22}")

    print("-" * 72)
    print(f"\nNOTE: ic32_regime_v1 metrics from model_registry are WITH cascade+LSTM+FLIP.")
    print(f"      TB widyawardhana_v1 is standalone LGBM only.")
    if baseline_agg:
        print(f"      'Baseline standalone' = ic32_regime_v1 LGBM-only (no cascade).")

    # Save comparison
    comparison = {
        "run_name": RUN_NAME,
        "timestamp": datetime.now().isoformat(),
        "parameters": {
            "tp_atr_mult": tp_atr,
            "sl_atr_mult": sl_atr,
            "max_hold": max_hold,
            "modal_per_trade": MODAL_PER_TRADE,
            "leverage": LEVERAGE_SIM[0],
        },
        "tb_widyawardhana_v1": tb_agg,
        "ic32_regime_v1_reference": registry_metrics,
        "baseline_standalone": baseline_agg,
    }
    comp_path = run_dir / "holdout_comparison.json"
    with open(comp_path, "w") as f:
        json.dump(comparison, f, indent=2, default=str)
    logger.info(f"Comparison -> {comp_path}")

    print(f"\nFull results saved to: {run_dir}")


if __name__ == "__main__":
    main()
