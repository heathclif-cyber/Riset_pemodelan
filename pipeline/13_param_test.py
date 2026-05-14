"""
pipeline/13_param_test.py — Parameter Testing Matrix (Grup 1-5)

Menguji parameter dari Quality Check & Trade Analysis:
  Grup 1: RR Gate low-ATR (max_sl_atr, VolR conditional, SL% cap)
  Grup 2: Trend Alignment (with_trend_penalty, counter_trend_boost)
  Grup 3: Structural Filter (breakout tolerance, swing freshness)
  Grup 4: Sizing (tiered, conditional half-size)
  Grup 5: Cooldown re-test

Metode:
  1. Load holdout labeled data + baseline models
  2. Run cascade inference (cached per symbol)
  3. Untuk Grup 1,3,4,5: uji varian parameter di simulate_trades_swing()
  4. Untuk Grup 2: re-run hierarchical_predict() dengan trend alignment
  5. Bandingkan metrik: trade count, winrate, net PnL, max DD, profit factor

Jalankan:
  python pipeline/13_param_test.py
  python pipeline/13_param_test.py --group 1
  python pipeline/13_param_test.py --coins SOLUSDT ETHUSDT
"""

import argparse
import json
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
import torch

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from config import (
    TRAINING_COINS, ALL_COINS, SYMBOL_MAP, LABEL_MAP, NUM_CLASSES,
    MODAL_PER_TRADE, LEVERAGE_SIM, FEE_PER_SIDE, SLIPPAGE_PER_SIDE,
    MAX_HOLDING_BARS, CONFIDENCE_THRESHOLD_ENTRY,
    SWING_LABEL_MIN_RR, SWING_LABEL_MIN_TP, SWING_LABEL_MAX_SL,
    SWING_LABEL_MAX_HOLD,
    FEATURE_COLS_V3, MODEL_DIR,
    TP_SL_HYBRID_MODE, TP_SL_SWING_FRESHNESS, TP_SL_STRUCTURAL_FILTER,
    TP_SL_RR_GATE_ENABLED, TP_SL_STRUCTURAL_TOLERANCE,
    TP_SL_MAX_SWING_DEVIATION_PCT,
    TP_SL_SIZING_MODE, TP_SL_COOLDOWN_ENABLED, TP_SL_SLIPPAGE_ENABLED,
)
from core.evaluator import simulate_trades_swing, full_trading_report
from core.utils import setup_logger, ensure_utc_index
from pipeline.backtest_utils import hierarchical_predict

logger = setup_logger("13_param_test")

HOLDOUT_LABEL_DIR = ROOT / "data" / "holdout" / "labeled"
REPORT_DIR        = ROOT / "reports" / "param_tests"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# ── Koin fokus per grup ────────────────────────────────────────────────────────
LOW_ATR_COINS = ["DOGEUSDT", "1000SHIBUSDT", "1000PEPEUSDT", "POLUSDT"]
WITH_TREND_COINS = ["SOLUSDT", "ETHUSDT", "TRXUSDT"]
SWING_LEAK_COINS = ["TONUSDT", "NEARUSDT", "AVAXUSDT"]


# ═══════════════════════════════════════════════════════════════════════════════════
# PHASE 0: Load data & run baseline cascade
# ═══════════════════════════════════════════════════════════════════════════════════

def load_holdout_data(symbol: str) -> Optional[pd.DataFrame]:
    path = HOLDOUT_LABEL_DIR / f"{symbol}_features_v3.parquet"
    if not path.exists():
        logger.warning(f"[{symbol}] Holdout features not found: {path}")
        return None
    df = pd.read_parquet(path)
    df = ensure_utc_index(df).sort_index()
    mask = df["label"].astype(str).isin(LABEL_MAP)
    return df[mask].copy()


def run_baseline_cascade(
    df: pd.DataFrame,
    lgbm_model,
    lstm_model,
    lstm_scaler,
    feat_cols: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Run baseline cascade (no trend alignment). Returns (y_pred, confidence)."""
    valid_cols = [c for c in feat_cols if c in df.columns]
    df_valid = df.copy()
    df_valid[valid_cols] = df_valid[valid_cols].ffill().fillna(0)
    X = df_valid[valid_cols].values.astype(np.float64)

    y_pred, confidence = hierarchical_predict(
        None, lgbm_model, lstm_model, lstm_scaler,
        X, valid_cols, [], df_valid,
        trend_alignment_enabled=False,  # baseline: no trend alignment
    )
    return y_pred, confidence


def apply_confidence_filter(y_pred: np.ndarray, confidence: np.ndarray) -> np.ndarray:
    yf = y_pred.copy()
    below = (yf != 1) & (confidence < CONFIDENCE_THRESHOLD_ENTRY)
    yf[below] = 1  # FLAT
    return yf


# ═══════════════════════════════════════════════════════════════════════════════════
# PHASE 1: Parameter variant runners
# ═══════════════════════════════════════════════════════════════════════════════════

def extract_arrays(df: pd.DataFrame) -> dict:
    """Extract numpy arrays from feature DataFrame for simulator."""
    arr = {
        "close": df["close"].values if "close" in df.columns else None,
        "high": df["high"].values if "high" in df.columns else None,
        "low": df["low"].values if "low" in df.columns else None,
        "atr": df["atr_14_h1"].values if "atr_14_h1" in df.columns else None,
        "h4_swing_high": df["h4_swing_high"].values if "h4_swing_high" in df.columns else None,
        "h4_swing_low": df["h4_swing_low"].values if "h4_swing_low" in df.columns else None,
        "vol_ratio": df["vol_ratio_20"].values if "vol_ratio_20" in df.columns else None,
        "h4_trend": df["h4_trend"].values if "h4_trend" in df.columns else None,
        "label": df["label"].map(LABEL_MAP).values.astype(np.int64),
    }
    return arr


def run_sim_variant(
    y_pred: np.ndarray,
    confidence: np.ndarray,
    arr: dict,
    symbol: str,
    variant_name: str,
    **kwargs,
) -> dict:
    """Run simulate_trades_swing with override params."""
    yf = apply_confidence_filter(y_pred, confidence)

    # Build defaults, allow kwargs to override
    params = dict(
        y_pred=yf,
        close=arr["close"],
        high=arr["high"],
        low=arr["low"],
        atr=arr["atr"],
        h4_swing_highs=arr["h4_swing_high"],
        h4_swing_lows=arr["h4_swing_low"],
        modal=MODAL_PER_TRADE,
        leverage=LEVERAGE_SIM[0],
        fee_per_side=FEE_PER_SIDE,
        slippage=SLIPPAGE_PER_SIDE,
        max_hold=MAX_HOLDING_BARS,
        min_rr=SWING_LABEL_MIN_RR,
        min_tp_atr=SWING_LABEL_MIN_TP,
        max_sl_atr=SWING_LABEL_MAX_SL,
        confidence=confidence,
        # Default toggles
        hybrid_mode=TP_SL_HYBRID_MODE,
        swing_freshness_check=TP_SL_SWING_FRESHNESS,
        structural_filter=TP_SL_STRUCTURAL_FILTER,
        structural_tolerance_pct=TP_SL_STRUCTURAL_TOLERANCE,
        slippage_enabled=TP_SL_SLIPPAGE_ENABLED,
        sizing_mode=TP_SL_SIZING_MODE,
        cooldown_enabled=TP_SL_COOLDOWN_ENABLED,
        swing_sl_bumper_atr=0.5,
        max_swing_deviation_pct=TP_SL_MAX_SWING_DEVIATION_PCT,
    )
    params.update(kwargs)
    result = simulate_trades_swing(**params)

    if result.get("error"):
        return {"variant": variant_name, "error": result["error"], "total_trades": 0}

    # Use max_drawdown from the function (already correctly computed on equity)
    max_dd = result.get("max_drawdown", 0.0)
    if max_dd < -1.0:
        max_dd = -1.0  # clamp unrealistic DD

    wins = sum(1 for t in result["trades"] if t["outcome"] == "WIN")
    total = len(result["trades"])
    wr = wins / total if total > 0 else 0.0

    # Profit factor
    win_sum = sum(t["net_pnl"] for t in result["trades"] if t["net_pnl"] > 0)
    loss_sum = abs(sum(t["net_pnl"] for t in result["trades"] if t["net_pnl"] < 0))
    pf = win_sum / loss_sum if loss_sum > 0 else 0.0

    n_months = max((arr["close"].shape[0] / 24 / 30.44), 0.1)
    tpm = round(total / n_months, 2)

    return {
        "variant": variant_name,
        "symbol": symbol,
        "total_trades": total,
        "wins": wins,
        "losses": result.get("losses", 0),
        "winrate": round(wr, 4),
        "net_pnl": round(result.get("net_pnl_total", 0), 2),
        "max_drawdown": round(max_dd, 4),
        "profit_factor": round(pf, 4),
        "trade_per_month": tpm,
        "avg_rr": result.get("avg_rr", 0),
    }


# ═══════════════════════════════════════════════════════════════════════════════════
# GRUP 1: RR Gate low-ATR
# ═══════════════════════════════════════════════════════════════════════════════════

def test_group_1(y_pred, confidence, arr, symbol):
    """Test max_sl_atr variants + VolR conditional + SL % cap."""
    results = []

    # 1a: Global max_sl_atr sweep
    for max_sl in [3.0, 4.0, 5.0, 6.0]:
        label = f"1a_max_sl_{max_sl}"
        r = run_sim_variant(y_pred, confidence, arr, symbol, label, max_sl_atr=max_sl)
        results.append(r)

    # 1b: VolR conditional — longgarkan max_sl saat vol mati
    for volr_thresh in [0.5, 0.8, 1.0]:  # sweep threshold (data shows p25=0.52, p50=0.79)
        r = run_sim_variant(y_pred, confidence, arr, symbol,
                            f"1b_volr_cond_vr{volr_thresh}_sl8",
                            vol_ratio=arr["vol_ratio"],
                            volr_conditional_enabled=True,
                            volr_threshold=volr_thresh,
                            max_sl_volr_low=8.0,
                            volr_disable_max_sl=False)
        results.append(r)

    # 1c: VolR conditional — disable max_sl total di low vol
    for volr_thresh in [0.5, 0.8]:
        r = run_sim_variant(y_pred, confidence, arr, symbol,
                            f"1c_volr_disable_vr{volr_thresh}",
                            vol_ratio=arr["vol_ratio"],
                            volr_conditional_enabled=True,
                            volr_threshold=volr_thresh,
                            volr_disable_max_sl=True)
        results.append(r)

    # 1d: SL % distance cap (alternatif ATR) — tighter range
    for sl_pct in [0.10, 0.15, 0.20, 0.30]:  # 10-30% (p50 SL is 3.5-5%)
        r = run_sim_variant(y_pred, confidence, arr, symbol,
                            f"1d_sl_pct_cap_{int(sl_pct*100)}pct",
                            max_sl_pct_enabled=True,
                            max_sl_pct=sl_pct)
        results.append(r)

    return results


# ═══════════════════════════════════════════════════════════════════════════════════
# GRUP 2: Trend Alignment (requires re-running cascade)
# ═══════════════════════════════════════════════════════════════════════════════════

def test_group_2(df, lgbm_model, lstm_model, lstm_scaler, feat_cols, arr, symbol):
    """Test trend alignment penalties — re-runs hierarchical_predict for each variant."""
    results = []

    valid_cols = [c for c in feat_cols if c in df.columns]
    df_valid = df.copy()
    df_valid[valid_cols] = df_valid[valid_cols].ffill().fillna(0)
    X = df_valid[valid_cols].values.astype(np.float64)

    variants = [
        # (name, enabled, penalty, boost, block_conf)
        ("2a_baseline_no_trend", False, 0.10, 0.05, 0.0),
        ("2a_penalty_0.10", True, 0.10, 0.05, 0.0),
        ("2a_penalty_0.15", True, 0.15, 0.05, 0.0),
        ("2a_penalty_0.20", True, 0.20, 0.05, 0.0),
        ("2a_penalty_0.25", True, 0.25, 0.05, 0.0),
        ("2b_boost_0.08", True, 0.15, 0.08, 0.0),
        ("2b_boost_0.10", True, 0.15, 0.10, 0.0),
        ("2c_block_0.95", True, 0.15, 0.05, 0.95),
    ]

    for name, enabled, penalty, boost, block in variants:
        y_pred, confidence = hierarchical_predict(
            None, lgbm_model, lstm_model, lstm_scaler,
            X, valid_cols, [], df_valid,
            trend_alignment_enabled=enabled,
            with_trend_penalty=penalty,
            counter_trend_boost=boost,
            with_trend_block_conf=block,
        )
        r = run_sim_variant(y_pred, confidence, arr, symbol, name)
        results.append(r)

    return results


# ═══════════════════════════════════════════════════════════════════════════════════
# GRUP 3: Structural Filter — Breakout Tolerance + Swing Freshness
# ═══════════════════════════════════════════════════════════════════════════════════

def test_group_3(y_pred, confidence, arr, symbol):
    """Test breakout tolerance, swing deviation, individual freshness."""
    results = []

    # 3a: Breakout tolerance sweep
    for tol in [0.0, 0.02, 0.04, 0.06]:
        r = run_sim_variant(y_pred, confidence, arr, symbol,
                            f"3a_tolerance_{int(tol*100)}pct",
                            structural_tolerance_pct=tol)
        results.append(r)

    # 3b: Max swing deviation (lebih ketat)
    for dev in [0.15, 0.12, 0.10]:
        r = run_sim_variant(y_pred, confidence, arr, symbol,
                            f"3b_max_dev_{int(dev*100)}pct",
                            max_swing_deviation_pct=dev)
        results.append(r)

    # 3c: Individual swing freshness check
    for dev in [0.15, 0.12, 0.10]:
        r = run_sim_variant(y_pred, confidence, arr, symbol,
                            f"3c_individual_dev_{int(dev*100)}pct",
                            max_swing_deviation_pct=dev,
                            individual_swing_freshness=True)
        results.append(r)

    return results


# ═══════════════════════════════════════════════════════════════════════════════════
# GRUP 4: Sizing — Tiered vs Fixed di Kondisi Ekstrim
# ═══════════════════════════════════════════════════════════════════════════════════

def test_group_4(y_pred, confidence, arr, symbol):
    """Test sizing modes."""
    results = []

    # 4a: Tiered sizing
    r = run_sim_variant(y_pred, confidence, arr, symbol,
                        "4a_tiered", sizing_mode="tiered")
    results.append(r)

    # 4b: Tiered + half-size for with-trend
    r = run_sim_variant(y_pred, confidence, arr, symbol,
                        "4b_tiered_half_with_trend",
                        sizing_mode="tiered",
                        h4_trend=arr["h4_trend"],
                        sizing_with_trend_half=True)
    results.append(r)

    return results


# ═══════════════════════════════════════════════════════════════════════════════════
# GRUP 5: Cooldown re-test
# ═══════════════════════════════════════════════════════════════════════════════════

def test_group_5(y_pred, confidence, arr, symbol):
    """Test cooldown enabled vs disabled."""
    results = []

    r = run_sim_variant(y_pred, confidence, arr, symbol,
                        "5a_cooldown_off", cooldown_enabled=False)
    results.append(r)

    r = run_sim_variant(y_pred, confidence, arr, symbol,
                        "5a_cooldown_on", cooldown_enabled=True)
    results.append(r)

    return results


# ═══════════════════════════════════════════════════════════════════════════════════
# Aggregation & Reporting
# ═══════════════════════════════════════════════════════════════════════════════════

def aggregate_results(all_results: list[dict], baseline_key: str) -> dict:
    """Aggregate across coins, grouped by variant name."""
    groups: dict[str, list[dict]] = {}
    for r in all_results:
        name = r["variant"]
        groups.setdefault(name, []).append(r)

    summary = {}
    baseline = None
    for name, items in groups.items():
        trades = [i["total_trades"] for i in items]
        wrs = [i["winrate"] for i in items]
        pnls = [i["net_pnl"] for i in items]
        dds = [i["max_drawdown"] for i in items]
        pfs = [i["profit_factor"] for i in items]
        tpms = [i["trade_per_month"] for i in items]

        agg = {
            "variant": name,
            "n_coins": len(items),
            "total_trades": sum(trades),
            "mean_trades_per_coin": round(np.mean(trades), 1) if trades else 0,
            "mean_winrate": round(np.mean(wrs), 4),
            "mean_net_pnl": round(np.mean(pnls), 2),
            "mean_max_dd": round(np.mean(dds), 4),
            "mean_profit_factor": round(np.mean(pfs), 2),
            "mean_trade_per_month": round(np.mean(tpms), 2),
            # Std dev untuk stabilitas
            "std_winrate": round(np.std(wrs), 4),
            "std_net_pnl": round(np.std(pnls), 2),
        }
        summary[name] = agg

        if name == baseline_key:
            baseline = agg

    # Hitung delta dari baseline
    if baseline:
        for name, agg in summary.items():
            if name == baseline_key:
                continue
            agg["delta_trades"] = agg["total_trades"] - baseline["total_trades"]
            agg["delta_trades_pct"] = round(
                (agg["total_trades"] - baseline["total_trades"]) / max(baseline["total_trades"], 1), 4)
            agg["delta_wr"] = round(agg["mean_winrate"] - baseline["mean_winrate"], 4)
            agg["delta_pnl"] = round(agg["mean_net_pnl"] - baseline["mean_net_pnl"], 2)
            agg["delta_dd"] = round(agg["mean_max_dd"] - baseline["mean_max_dd"], 4)
            agg["delta_pf"] = round(agg["mean_profit_factor"] - baseline["mean_profit_factor"], 2)

    return summary


def print_aggregate_table(summary: dict, title: str):
    """Print formatted comparison table."""
    print(f"\n{'=' * 110}")
    print(f"  {title}")
    print(f"{'=' * 110}")
    header = f"{'Variant':<38} {'Trades':>7} {'WR':>8} {'Net PnL':>10} {'Max DD':>8} {'PF':>6} {'T/mo':>6}"
    print(header)
    print("-" * 110)

    for name, s in summary.items():
        delta = ""
        if "delta_wr" in s:
            d_wr = s["delta_wr"]
            d_pnl = s["delta_pnl"]
            d_t = s.get("delta_trades_pct", 0)
            delta = f"  [dT:{d_t:+.0%} dWR:{d_wr:+.2%} dPnL:${d_pnl:+.0f}]"
        print(f"{name:<38} {s['total_trades']:>7} {s['mean_winrate']:>7.2%} "
              f"${s['mean_net_pnl']:>9.0f} {s['mean_max_dd']:>7.2%} "
              f"{s['mean_profit_factor']:>5.1f} {s['mean_trade_per_month']:>5.1f}{delta}")

    print("-" * 110)


# ═══════════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(description="Parameter Testing Matrix")
    parser.add_argument("--group", type=int, choices=[1, 2, 3, 4, 5],
                        help="Run specific group only")
    parser.add_argument("--coins", nargs="+", metavar="SYMBOL",
                        help="Specific coins (default: depends on group)")
    parser.add_argument("--all-coins", action="store_true",
                        help="Run on all 20 coins")
    parser.add_argument("--output", default=None,
                        help="Output JSON path")
    return parser.parse_args()


def main():
    args = parse_args()
    t0 = time.time()

    # Determine coins to test
    if args.coins:
        coins = [c.upper() for c in args.coins]
    elif args.all_coins:
        coins = ALL_COINS
    else:
        coins = ALL_COINS  # Default: all coins for comprehensive test

    target_groups = [args.group] if args.group else [1, 2, 3, 4, 5]

    # Load models
    print("Loading models...")
    lgbm_model = joblib.load(MODEL_DIR / "lgbm_baseline.pkl")
    from core.models import load_lstm
    lstm_model = load_lstm(MODEL_DIR / "lstm_best.pt", device="cpu")
    lstm_scaler = joblib.load(MODEL_DIR / "lstm_scaler.pkl")
    with open(MODEL_DIR / "feature_cols_v2.json") as f:
        feat_cols = json.load(f)

    all_results: dict[int, list] = {g: [] for g in target_groups}

    # Process each coin
    for i, symbol in enumerate(coins, 1):
        print(f"\n[{i}/{len(coins)}] {symbol}...")
        df = load_holdout_data(symbol)
        if df is None:
            continue

        arr = extract_arrays(df)
        if arr["close"] is None or arr["atr"] is None:
            logger.warning(f"[{symbol}] Missing critical columns — skip")
            continue

        # Run baseline cascade (no trend alignment)
        y_pred_base, conf_base = run_baseline_cascade(
            df, lgbm_model, lstm_model, lstm_scaler, feat_cols)

        yf_base = apply_confidence_filter(y_pred_base, conf_base)

        # Test Group 1
        if 1 in target_groups:
            results = test_group_1(y_pred_base, conf_base, arr, symbol)
            all_results[1].extend(results)

        # Test Group 2 (re-runs cascade)
        if 2 in target_groups:
            results = test_group_2(df, lgbm_model, lstm_model, lstm_scaler,
                                   feat_cols, arr, symbol)
            all_results[2].extend(results)

        # Test Group 3
        if 3 in target_groups:
            results = test_group_3(y_pred_base, conf_base, arr, symbol)
            all_results[3].extend(results)

        # Test Group 4
        if 4 in target_groups:
            results = test_group_4(y_pred_base, conf_base, arr, symbol)
            all_results[4].extend(results)

        # Test Group 5
        if 5 in target_groups:
            results = test_group_5(y_pred_base, conf_base, arr, symbol)
            all_results[5].extend(results)

    # Print aggregate results per group
    report_data = {}
    group_titles = {
        1: "GRUP 1: RR Gate low-ATR (max_sl, VolR conditional, SL% cap)",
        2: "GRUP 2: Trend Alignment (with_trend_penalty, counter_trend_boost)",
        3: "GRUP 3: Structural Filter (breakout tolerance, swing freshness)",
        4: "GRUP 4: Sizing (tiered vs fixed, conditional half-size)",
        5: "GRUP 5: Cooldown re-test",
    }
    group_baselines = {
        1: "1a_max_sl_3.0", 2: "2a_baseline_no_trend", 3: "3a_tolerance_4pct",
        4: "4a_tiered", 5: "5a_cooldown_off",
    }

    for g in target_groups:
        if all_results[g]:
            summary = aggregate_results(all_results[g], group_baselines[g])
            print_aggregate_table(summary, group_titles[g])
            report_data[f"group_{g}"] = summary

            # Print per-coin detail for focused coins
            if g == 1:
                print("\n  [Grup 1 — Detail Low-ATR Coins]")
                for r in all_results[g]:
                    if r["symbol"] in LOW_ATR_COINS:
                        print(f"  {r['symbol']:<14} {r['variant']:<35} "
                              f"T={r['total_trades']:>4} WR={r['winrate']:.2%} "
                              f"PnL=${r['net_pnl']:>8.0f} DD={r['max_drawdown']:.2%}")

    # Save report
    all_results_serializable = {
        f"group_{g}": items
        for g, items in all_results.items()
    }
    out_path = args.output or (
        REPORT_DIR / f"param_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(out_path, "w") as f:
        json.dump(all_results_serializable, f, indent=2, default=str)

    elapsed = time.time() - t0
    print(f"\n{'=' * 80}")
    print(f"  PARAMETER TEST SELESAI — {elapsed:.1f}s")
    print(f"  Report: {out_path}")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
