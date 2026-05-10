"""
Test script: Bandingkan Hybrid vs Pure Tier TP/SL pada Holdout Backtest

Aspect #1 — Sumber TP/SL saat swing ada:
  HYBRID (sekarang):  max(swing, ATR) untuk TP, min(swing, ATR) untuk SL
  PURE   (proposal):  Tier murni: swing ada → swing only, tanpa campur ATR

Jalankan: python pipeline/test_hybrid_vs_pure.py
"""

import json
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from config import (
    ALL_COINS, LABEL_MAP, NUM_CLASSES,
    MODAL_PER_TRADE, LEVERAGE_SIM, FEE_PER_SIDE, SLIPPAGE_PER_SIDE,
    MAX_HOLDING_BARS, CONFIDENCE_THRESHOLD_ENTRY,
    SWING_LABEL_MIN_RR, SWING_LABEL_MIN_TP, SWING_LABEL_MAX_SL,
)
from core.models import load_lstm
from core.evaluator import simulate_trades_swing
from core.utils import setup_logger
from pipeline.backtest_utils import hierarchical_predict

logger = setup_logger("test_hybrid")
DEVICE = torch.device("cpu")
HOLDOUT_LABEL_DIR = ROOT / "data" / "holdout" / "labeled"
MODEL_DIR = ROOT / "models"


def load_models():
    lgbm = joblib.load(MODEL_DIR / "lgbm_baseline.pkl")
    lstm = load_lstm(MODEL_DIR / "lstm_best.pt").to(DEVICE)
    scaler = joblib.load(MODEL_DIR / "lstm_scaler.pkl")
    with open(MODEL_DIR / "feature_cols_v2.json") as f:
        feat_cols = json.load(f)
    return lgbm, lstm, scaler, feat_cols


def run_backtest_for_mode(symbols, lgbm, lstm, scaler, feat_cols, hybrid_mode):
    mode_name = "HYBRID" if hybrid_mode else "PURE"
    results = {}

    for symbol in symbols:
        path = HOLDOUT_LABEL_DIR / f"{symbol}_features_v3.parquet"
        if not path.exists():
            continue

        df = pd.read_parquet(path)
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index, utc=True)
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        df = df.sort_index()

        mask = df["label"].astype(str).isin(LABEL_MAP)
        df = df[mask].copy()

        valid_cols = [c for c in feat_cols if c in df.columns]
        df[valid_cols] = df[valid_cols].ffill().fillna(0)
        X = df[valid_cols].values.astype(np.float64)

        y_pred, confidence = hierarchical_predict(
            None, lgbm, lstm, scaler, X, valid_cols, [], df[valid_cols],
        )

        below = (y_pred != 1) & (confidence < CONFIDENCE_THRESHOLD_ENTRY)
        y_pred[below] = 1

        close_arr = df["close"].values
        high_arr  = df["high"].values  if "high"  in df.columns else close_arr
        low_arr   = df["low"].values   if "low"   in df.columns else close_arr
        atr_arr   = df["atr_14_h1"].values if "atr_14_h1" in df.columns else np.ones(len(df))
        sh_arr    = df["h4_swing_high"].values if "h4_swing_high" in df.columns else np.full(len(df), np.nan)
        sl_arr    = df["h4_swing_low"].values  if "h4_swing_low"  in df.columns else np.full(len(df), np.nan)

        sim = simulate_trades_swing(
            y_pred          = y_pred,
            close           = close_arr,
            high            = high_arr,
            low             = low_arr,
            atr             = atr_arr,
            h4_swing_highs  = sh_arr,
            h4_swing_lows   = sl_arr,
            modal           = MODAL_PER_TRADE,
            leverage        = LEVERAGE_SIM[0],
            fee_per_side    = FEE_PER_SIDE,
            slippage        = SLIPPAGE_PER_SIDE,
            min_rr          = SWING_LABEL_MIN_RR,
            min_tp_atr      = SWING_LABEL_MIN_TP,
            max_sl_atr      = SWING_LABEL_MAX_SL,
            max_hold        = MAX_HOLDING_BARS,
            hybrid_mode     = hybrid_mode,
        )

        if sim.get("error"):
            continue

        trades = sim.get("trades", [])
        n_swing_trades = sum(1 for t in trades if not t.get("fallback", False))

        results[symbol] = {
            "winrate":        sim["winrate"],
            "total_trades":   sim["total_trades"],
            "total_pnl":      sim["total_pnl"],
            "max_drawdown":   sim.get("max_drawdown", 0),
            "avg_rr":         sim.get("avg_rr", 0),
            "wins":           sim["wins"],
            "losses":         sim["losses"],
        }

        logger.info(
            f"[{mode_name}] {symbol}: WR={sim['winrate']:.2%} "
            f"Trades={sim['total_trades']} PnL=${sim['total_pnl']:+.2f} "
            f"DD={sim.get('max_drawdown', 0):.2%}"
        )

    return results


def main():
    print("=" * 70)
    print("  HOLD-OUT BACKTEST: HYBRID vs PURE TIER TP/SL")
    print("=" * 70)

    lgbm, lstm, scaler, feat_cols = load_models()
    symbols = [p.stem.replace("_features_v3", "") for p in HOLDOUT_LABEL_DIR.glob("*_features_v3.parquet")]
    logger.info(f"Coins: {len(symbols)} — {symbols}")

    # ── Run HYBRID mode (sekarang) ──────────────────────────────────────────
    print("\n--- MODE: HYBRID (max(swing,ATR) TP / min(swing,ATR) SL) ---")
    hybrid_results = run_backtest_for_mode(symbols, lgbm, lstm, scaler, feat_cols, hybrid_mode=True)

    # ── Run PURE mode (proposal) ────────────────────────────────────────────
    print("\n--- MODE: PURE TIER (swing only) ---")
    pure_results = run_backtest_for_mode(symbols, lgbm, lstm, scaler, feat_cols, hybrid_mode=False)

    # ── Compare ─────────────────────────────────────────────────────────────
    common = set(hybrid_results) & set(pure_results)
    if not common:
        print("No common coins to compare.")
        return

    print("\n" + "=" * 90)
    print("  PERBANDINGAN HYBRID vs PURE TIER")
    print("=" * 90)
    print(f"  {'Coin':<14} {'Hybrid WR':>10} {'Pure WR':>10} {'Delta WR':>10} "
          f"{'Hybrid Tr':>10} {'Pure Tr':>10} {'Hybrid DD':>10} {'Pure DD':>10}")
    print(f"  {'-'*14} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")

    for sym in sorted(common):
        h = hybrid_results[sym]
        p = pure_results[sym]
        dwr = p["winrate"] - h["winrate"]
        dtr = p["total_trades"] - h["total_trades"]
        ddd = p["max_drawdown"] - h["max_drawdown"]
        print(f"  {sym:<14} {h['winrate']:>10.2%} {p['winrate']:>10.2%} {dwr:>+10.2%} "
              f"{h['total_trades']:>10} {p['total_trades']:>10} "
              f"{h['max_drawdown']:>10.2%} {p['max_drawdown']:>10.2%}")

    # ── Aggregate ───────────────────────────────────────────────────────────
    h_wr  = [r["winrate"] for r in hybrid_results.values()]
    p_wr  = [r["winrate"] for r in pure_results.values()]
    h_tr  = [r["total_trades"] for r in hybrid_results.values()]
    p_tr  = [r["total_trades"] for r in pure_results.values()]
    h_pnl = [r["total_pnl"] for r in hybrid_results.values()]
    p_pnl = [r["total_pnl"] for r in pure_results.values()]
    h_dd  = [r["max_drawdown"] for r in hybrid_results.values()]
    p_dd  = [r["max_drawdown"] for r in pure_results.values()]

    print(f"\n  {'─'*60}")
    print(f"  {'AGGREGATE':<20} {'HYBRID':>15} {'PURE':>15} {'DELTA':>15}")
    print(f"  {'─'*60}")
    print(f"  {'Mean Winrate':<20} {np.mean(h_wr):>15.2%} {np.mean(p_wr):>15.2%} "
          f"{np.mean(p_wr)-np.mean(h_wr):>+15.2%}")
    print(f"  {'Mean Trades':<20} {np.mean(h_tr):>15.1f} {np.mean(p_tr):>15.1f} "
          f"{np.mean(p_tr)-np.mean(h_tr):>+15.1f}")
    print(f"  {'Mean PnL ($)':<20} {np.mean(h_pnl):>+15.2f} {np.mean(p_pnl):>+15.2f} "
          f"{np.mean(p_pnl)-np.mean(h_pnl):>+15.2f}")
    print(f"  {'Mean Max DD':<20} {np.mean(h_dd):>15.2%} {np.mean(p_dd):>15.2%} "
          f"{np.mean(p_dd)-np.mean(h_dd):>+15.2%}")
    print(f"  {'─'*60}")


if __name__ == "__main__":
    main()
