"""
experiments/cascade_sweep.py — Cascade fusion parameter sweep
Holdout bersih Nov 2025 – Apr 2026

Usage:
  python experiments/cascade_sweep.py --phase 1    # 5 koin narrowing
  python experiments/cascade_sweep.py --phase 2    # 21 koin validation (top 3)
"""

import argparse
import json
import os
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd

from config import (
    TRAINING_COINS, LABEL_DIR, MODEL_DIR, HOLDOUT_DIR,
    MODAL_PER_TRADE, LEVERAGE_SIM, FEE_PER_SIDE, SLIPPAGE_PER_SIDE,
    MAX_HOLDING_BARS, SWING_LABEL_MIN_RR, SWING_LABEL_MIN_TP, SWING_LABEL_MAX_SL,
    TP_SL_FALLBACK_TP, TP_SL_FALLBACK_SL,
    LGBM_THRESHOLD_LONG as _DEF_LGBM_THR_LONG,
    LGBM_THRESHOLD_SHORT as _DEF_LGBM_THR_SHORT,
    LABEL_MAP, GUARDIAN_ENABLED, CONFIDENCE_THRESHOLD_ENTRY,
    GUARDIAN_EXIT_THRESHOLD, GUARDIAN_DYNAMIC_FEATURES,
)

from core.utils import setup_logger
from core.models import load_lstm
from core.evaluator import simulate_trades_swing
from pipeline.backtest_utils import compute_guardian_static_array

logger = setup_logger("cascade_sweep")

NON_FEATURE_COLS = {"label", "h4_swing_high", "h4_swing_low"}

# Guardian static columns — computed once from guardian_feature_cols.json
_g_feats_all = json.load(open(MODEL_DIR / "guardian_feature_cols.json"))
GUARDIAN_STATIC_COLS = [c for c in _g_feats_all if c not in set(GUARDIAN_DYNAMIC_FEATURES)]

# ── Load models once ──────────────────────────────────────────────────────────
def load_all_models():
    lgbm = joblib.load(MODEL_DIR / "lgbm_baseline.pkl")
    lstm_model = load_lstm(MODEL_DIR / "lstm_best.pt")
    lstm_scaler_obj = joblib.load(MODEL_DIR / "lstm_scaler.pkl")
    guardian = joblib.load(MODEL_DIR / "guardian_best.pkl")
    g_scaler = joblib.load(MODEL_DIR / "guardian_scaler.pkl")
    g_feats = json.load(open(MODEL_DIR / "guardian_feature_cols.json"))
    feat_cols = json.load(open(MODEL_DIR / "feature_cols_v2.json"))
    return lgbm, lstm_model, lstm_scaler_obj, guardian, g_scaler, g_feats, feat_cols


def load_holdout_data(coins: list[str]):
    """Load holdout parquet for specified coins, merge HMM regime."""
    data = {}
    for coin in coins:
        path = HOLDOUT_DIR / "labeled" / f"{coin}_features_v3.parquet"
        if not path.exists():
            logger.warning(f"Holdout missing: {coin}")
            continue
        df = pd.read_parquet(path).sort_index()
        # Merge HMM regime
        reg_path = HOLDOUT_DIR / "labeled" / f"{coin}_regime_h1.parquet"
        if reg_path.exists():
            reg = pd.read_parquet(reg_path)
            if "hmm_regime_enc" in df.columns:
                df = df.drop(columns=["hmm_regime_enc"])
            df = df.join(reg[["hmm_regime_enc"]], how="left")
            df["hmm_regime_enc"] = df["hmm_regime_enc"].fillna(1).astype("int32")
        else:
            df["hmm_regime_enc"] = 1
        # Filter labeled bars
        mask = df["label"].astype(str).isin(LABEL_MAP)
        df = df[mask].copy()
        if len(df) < 50:
            continue
        data[coin] = df
    return data


def run_one_config(
    config_label: str,
    coins_data: dict,
    lgbm, lstm_model, lstm_scaler_obj,
    guardian, g_scaler, g_feats, feat_cols,
    **btu_overrides,
) -> dict:
    """Run one cascade config and return aggregate metrics."""
    import pipeline.backtest_utils as btu

    # Apply monkey-patches directly to backtest_utils module globals
    for k, v in btu_overrides.items():
        if hasattr(btu, k):
            setattr(btu, k, v)
        else:
            logger.warning(f"Unknown param for backtest_utils: {k}")

    # hierarchical_predict reads module-level globals at call time,
    # so patching btu.* before calling is sufficient
    from pipeline.backtest_utils import hierarchical_predict

    all_trades = []
    for coin, df in coins_data.items():
        n = len(df)
        # Build X from ALL parquet feature cols (not just feat_cols).
        # hierarchical_predict/gel_lstm_proba auto-slice what they need:
        # - LGBM uses feat_cols (33) via model.feature_name_
        # - LSTM uses first N cols via scaler.n_features_in_
        parquet_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]
        X = np.zeros((n, len(parquet_cols)), dtype=np.float64)
        for i, col in enumerate(parquet_cols):
            X[:, i] = df[col].ffill().fillna(0).values

        try:
            y_pred, confidence = hierarchical_predict(
                None, lgbm, lstm_model, lstm_scaler_obj,
                X, feat_cols, [], df,
                trend_alignment_enabled=btu_overrides.get("TREND_ALIGNMENT_ENABLED", False),
                with_trend_penalty=btu_overrides.get("WITH_TREND_PENALTY", 0.10),
                counter_trend_boost=btu_overrides.get("COUNTER_TREND_BOOST", 0.05),
            )
        except Exception as e:
            logger.error(f"  {config_label} | {coin}: hierarchical_predict error: {e}")
            continue

        # Confidence threshold
        below = (y_pred != 1) & (confidence < CONFIDENCE_THRESHOLD_ENTRY)
        y_pred[below] = 1

        # Pre-compute Guardian static feature array
        X_guardian = None
        if guardian is not None:
            X_guardian = compute_guardian_static_array(df, GUARDIAN_STATIC_COLS)

        close = df["close"].values
        high = df["high"].values if "high" in df.columns else close
        low = df["low"].values if "low" in df.columns else close
        atr = df["atr_14_h1"].values if "atr_14_h1" in df.columns else np.ones(n)
        sh = df["h4_swing_high"].values if "h4_swing_high" in df.columns else np.full(n, np.nan)
        sl = df["h4_swing_low"].values if "h4_swing_low" in df.columns else np.full(n, np.nan)

        result = simulate_trades_swing(
            y_pred=y_pred, close=close, high=high, low=low, atr=atr,
            h4_swing_highs=sh, h4_swing_lows=sl,
            modal=MODAL_PER_TRADE, leverage=LEVERAGE_SIM[0],
            fee_per_side=FEE_PER_SIDE, slippage=SLIPPAGE_PER_SIDE,
            max_hold=MAX_HOLDING_BARS,
            min_rr=SWING_LABEL_MIN_RR, min_tp_atr=SWING_LABEL_MIN_TP,
            max_sl_atr=SWING_LABEL_MAX_SL,
            tp_fallback_atr=TP_SL_FALLBACK_TP, sl_fallback_atr=TP_SL_FALLBACK_SL,
            confidence=confidence,
            guardian_enabled=GUARDIAN_ENABLED,
            guardian_model=guardian,
            guardian_scaler=g_scaler,
            X_guardian=X_guardian,
            guardian_exit_threshold=GUARDIAN_EXIT_THRESHOLD,
        )

        trades = result.get("trades", [])
        for t in trades:
            t["coin"] = coin
        all_trades.extend(trades)

    # Aggregate metrics
    if not all_trades:
        return {"config": config_label, "total_trades": 0, "wr_overall": 0, "wr_long": 0, "wr_short": 0,
                "total_pnl": 0, "profit_factor": 0, "sharpe": 0, "max_drawdown_pct": 0,
                "avg_hold_bars": 0, "guardian_exit_wr": 0, "sl_hit_pct": 0, "long_pct": 0,
                "trades_per_bulan": 0, "sl_hit_count": 0, "guardian_exit_count": 0}

    wins = [t for t in all_trades if t.get("net_pnl", 0) > 0]
    losses = [t for t in all_trades if t.get("net_pnl", 0) <= 0]
    n = len(all_trades)
    wr = len(wins) / n * 100 if n > 0 else 0

    long_trades = [t for t in all_trades if t.get("direction") == "LONG"]
    short_trades = [t for t in all_trades if t.get("direction") == "SHORT"]
    long_wr = len([t for t in long_trades if t.get("net_pnl", 0) > 0]) / len(long_trades) * 100 if long_trades else 0
    short_wr = len([t for t in short_trades if t.get("net_pnl", 0) > 0]) / len(short_trades) * 100 if short_trades else 0

    total_pnl = sum(t.get("net_pnl", 0) for t in all_trades)
    gross_win = sum(t.get("net_pnl", 0) for t in wins)
    gross_loss = abs(sum(t.get("net_pnl", 0) for t in losses))
    pf = gross_win / gross_loss if gross_loss > 0 else float("inf")

    # Exit reason breakdown — use 'outcome' field from evaluator
    guardian_exits = [t for t in all_trades if "guardian" in str(t.get("outcome", "")).lower()]
    sl_hits = [t for t in all_trades if str(t.get("outcome", "")).lower() in ("loss", "sl_hit")]
    gx_wr = len([t for t in guardian_exits if t.get("net_pnl", 0) > 0]) / len(guardian_exits) * 100 if guardian_exits else 0

    # Hold bars: bar_out - bar_in
    hold_bars = [t.get("bar_out", 0) - t.get("bar_in", 0) for t in all_trades if "bar_in" in t and "bar_out" in t]
    avg_hold = np.mean(hold_bars) if hold_bars else 0

    # Sharpe from trade-level PnL (approximate)
    pnls = [t.get("net_pnl", 0) for t in all_trades]
    sharpe = float(np.mean(pnls) / np.std(pnls) * np.sqrt(len(pnls))) if len(pnls) > 1 and np.std(pnls) > 0 else 0

    # Max consecutive loss
    max_cl = 0; cur_cl = 0
    for t in all_trades:
        if t.get("net_pnl", 0) <= 0:
            cur_cl += 1; max_cl = max(max_cl, cur_cl)
        else:
            cur_cl = 0

    # DD from equity curve
    equity = 0; peak = 0; max_dd = 0
    for t in all_trades:
        equity += t.get("net_pnl", 0)
        peak = max(peak, equity)
        dd = (equity - peak) / peak if peak > 0 else 0
        max_dd = min(max_dd, dd)
    max_dd_pct = abs(max_dd) * 100

    months = 5.0  # Nov 2025 – Mar 2026

    return {
        "config": config_label,
        "total_trades": n,
        "trades_per_bulan": round(n / months, 1),
        "wr_overall": round(wr, 2),
        "wr_long": round(long_wr, 2),
        "wr_short": round(short_wr, 2),
        "total_pnl": round(total_pnl, 2),
        "profit_factor": round(pf, 2),
        "sharpe": round(sharpe, 2),
        "max_drawdown_pct": round(max_dd_pct, 1),
        "avg_hold_bars": round(avg_hold, 1),
        "guardian_exit_wr": round(gx_wr, 2),
        "sl_hit_count": len(sl_hits),
        "sl_hit_pct": round(len(sl_hits) / n * 100, 1) if n > 0 else 0,
        "long_pct": round(len(long_trades) / n * 100, 1) if n > 0 else 0,
        "max_consecutive_loss": max_cl,
        "guardian_exit_count": len(guardian_exits),
    }


# ─── Phase 1 Configs ──────────────────────────────────────────────────────────

def get_phase1_configs():
    """Generate narrowing configs for Phase 1."""
    configs = []

    # Step 1.1 — Mode Baseline (4 configs)
    configs.append({
        "label": "1_hard_consensus_current",
        "SMART_ENTRY_MODE": "disabled",
        "LSTM_FLAT_REVIEW_ENABLED": True,
        "LSTM_DIRECTIONAL_REVIEW_THRESHOLD": 0.35,
        "LSTM_ADJUST_OPPOSITE_PEN": 0.65,
        "LGBM_THRESHOLD_LONG": 0.69,
        "LGBM_THRESHOLD_SHORT": 0.59,
        "TREND_ALIGNMENT_ENABLED": True,
        "WITH_TREND_PENALTY": 0.10,
        "COUNTER_TREND_BOOST": 0.05,
    })

    configs.append({
        "label": "2_hard_consensus_no_trend",
        "SMART_ENTRY_MODE": "disabled",
        "LSTM_FLAT_REVIEW_ENABLED": True,
        "LSTM_DIRECTIONAL_REVIEW_THRESHOLD": 0.35,
        "LSTM_ADJUST_OPPOSITE_PEN": 0.65,
        "LGBM_THRESHOLD_LONG": 0.69,
        "LGBM_THRESHOLD_SHORT": 0.59,
        "TREND_ALIGNMENT_ENABLED": False,
    })

    configs.append({
        "label": "3_dual_dominant_Z3",
        "SMART_ENTRY_MODE": "dual_dominant",
        "SMART_ENTRY_LGBM_GATE": 0.65,
        "LSTM_DOMINANT_THRESHOLD": 0.35,
        "LGBM_THRESHOLD_LONG": 0.69,
        "LGBM_THRESHOLD_SHORT": 0.59,
        "TREND_ALIGNMENT_ENABLED": False,
    })

    configs.append({
        "label": "4_lstm_dominant_Y1",
        "SMART_ENTRY_MODE": "lstm_dominant",
        "LSTM_DOMINANT_THRESHOLD": 0.35,
        "LGBM_THRESHOLD_LONG": 0.69,
        "LGBM_THRESHOLD_SHORT": 0.59,
        "TREND_ALIGNMENT_ENABLED": False,
    })

    return configs


def get_phase1b_configs(best_modes: list[str]):
    """Threshold sweep for best 2 modes."""
    configs = []

    if "hard_consensus" in best_modes:
        for opp in [0.35, 0.50, 0.65, 0.80]:
            for thr_l in [0.65, 0.69]:
                for thr_s in [0.55, 0.59]:
                    configs.append({
                        "label": f"hc_opp{opp}_l{thr_l}_s{thr_s}",
                        "SMART_ENTRY_MODE": "disabled",
                        "LSTM_FLAT_REVIEW_ENABLED": False,
                        "LSTM_ADJUST_OPPOSITE_PEN": opp,
                        "LGBM_THRESHOLD_LONG": thr_l,
                        "LGBM_THRESHOLD_SHORT": thr_s,
                        "TREND_ALIGNMENT_ENABLED": False,
                    })

    if "dual_dominant" in best_modes:
        for gate in [0.55, 0.60, 0.65, 0.70]:
            for dom in [0.30, 0.33, 0.35, 0.38]:
                configs.append({
                    "label": f"dd_gate{gate}_dom{dom}",
                    "SMART_ENTRY_MODE": "dual_dominant",
                    "SMART_ENTRY_LGBM_GATE": gate,
                    "LSTM_DOMINANT_THRESHOLD": dom,
                    "LGBM_THRESHOLD_LONG": 0.69,
                    "LGBM_THRESHOLD_SHORT": 0.59,
                    "TREND_ALIGNMENT_ENABLED": False,
                })

    if "lstm_dominant" in best_modes:
        for thr_l in [0.65, 0.69, 0.72]:
            for thr_s in [0.55, 0.59, 0.62]:
                for dom in [0.30, 0.33, 0.35, 0.38]:
                    configs.append({
                        "label": f"ld_l{thr_l}_s{thr_s}_dom{dom}",
                        "SMART_ENTRY_MODE": "lstm_dominant",
                        "LSTM_DOMINANT_THRESHOLD": dom,
                        "LGBM_THRESHOLD_LONG": thr_l,
                        "LGBM_THRESHOLD_SHORT": thr_s,
                        "TREND_ALIGNMENT_ENABLED": False,
                    })

    return configs


def get_phase2_configs():
    """Top 3 configs from Phase 1 for 21-coin validation."""
    return [
        {
            "label": "P2_hard_consensus_current",
            "SMART_ENTRY_MODE": "disabled",
            "LSTM_FLAT_REVIEW_ENABLED": True,
            "LSTM_DIRECTIONAL_REVIEW_THRESHOLD": 0.35,
            "LSTM_ADJUST_OPPOSITE_PEN": 0.65,
            "LGBM_THRESHOLD_LONG": 0.69,
            "LGBM_THRESHOLD_SHORT": 0.59,
            "TREND_ALIGNMENT_ENABLED": True,
            "WITH_TREND_PENALTY": 0.10,
            "COUNTER_TREND_BOOST": 0.05,
        },
        {
            "label": "P2_hard_consensus_no_trend",
            "SMART_ENTRY_MODE": "disabled",
            "LSTM_FLAT_REVIEW_ENABLED": True,
            "LSTM_DIRECTIONAL_REVIEW_THRESHOLD": 0.35,
            "LSTM_ADJUST_OPPOSITE_PEN": 0.65,
            "LGBM_THRESHOLD_LONG": 0.69,
            "LGBM_THRESHOLD_SHORT": 0.59,
            "TREND_ALIGNMENT_ENABLED": False,
        },
        {
            "label": "P2_lstm_dominant_tight",
            "SMART_ENTRY_MODE": "lstm_dominant",
            "LSTM_DOMINANT_THRESHOLD": 0.35,
            "LGBM_THRESHOLD_LONG": 0.69,
            "LGBM_THRESHOLD_SHORT": 0.62,
            "TREND_ALIGNMENT_ENABLED": False,
        },
    ]


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", type=int, default=1)
    parser.add_argument("--coins", type=int, default=5)
    parser.add_argument("--best-modes", nargs="+", default=None,
                        help="Modes to sweep in phase 1b, e.g. hard_consensus dual_dominant")
    args = parser.parse_args()

    print(f"\n{'='*65}")
    print(f" CASCADE SWEEP — Phase {args.phase} | {args.coins} coins")
    print(f" Period: Nov 2025 – Apr 2026 (clean holdout)")
    print(f" Models: ic32_regime_v1 LGBM + ic32_lstm_multi_v1 + Guardian clean_v2")
    print(f"{'='*65}\n")

    # Load models once
    print("[0] Loading models...")
    lgbm, lstm_model, lstm_scaler, guardian, g_scaler, g_feats, feat_cols = load_all_models()
    print(f"    LGBM features: {len(feat_cols)} | Guardian features: {len(g_feats)}")

    # Load holdout data
    coins = TRAINING_COINS[:args.coins] if args.coins <= len(TRAINING_COINS) else TRAINING_COINS
    print(f"[0] Loading holdout data for {len(coins)} coins...")
    coins_data = load_holdout_data(coins)
    print(f"    Loaded: {list(coins_data.keys())}")

    if args.phase == 1:
        if args.best_modes:
            configs = get_phase1b_configs(args.best_modes)
            phase_label = "1b"
        else:
            configs = get_phase1_configs()
            phase_label = "1a"
    elif args.phase == 2:
        configs = get_phase2_configs()
        phase_label = "2"
    else:
        print(f"Unknown phase: {args.phase}")
        return

    print(f"\n[1] Running {len(configs)} configs (Phase {phase_label})...\n")

    results = []
    for i, cfg in enumerate(configs):
        label = cfg.pop("label")
        t0 = time.perf_counter()
        res = run_one_config(label, coins_data, lgbm, lstm_model, lstm_scaler,
                            guardian, g_scaler, g_feats, feat_cols, **cfg)
        elapsed = time.perf_counter() - t0
        status = f"{res.get('total_trades', 0)} trades, WR={res.get('wr_overall', 0):.1f}%"
        print(f"  [{i+1}/{len(configs)}] {label:<40} {status:<40} ({elapsed:.0f}s)")
        results.append(res)

    # Print summary
    print(f"\n{'='*80}")
    print(f" SUMMARY — Phase {phase_label}")
    print(f"{'='*80}")
    header = f"  {'Config':<35} {'Trades':>6} {'WR%':>6} {'L_WR%':>6} {'S_WR%':>6} {'PnL':>8} {'PF':>6} {'Sharpe':>7} {'DD%':>6} {'SL%':>6} {'L%':>6}"
    print(header)
    print("-" * len(header))
    for r in sorted(results, key=lambda x: (x.get("wr_overall", 0) * x.get("total_trades", 0)), reverse=True):
        print(f"  {r['config']:<35} {r['total_trades']:>6} {r['wr_overall']:>6.1f} {r['wr_long']:>6.1f} {r['wr_short']:>6.1f} {r['total_pnl']:>8.1f} {r['profit_factor']:>6.1f} {r['sharpe']:>7.1f} {r['max_drawdown_pct']:>6.1f} {r['sl_hit_pct']:>6.1f} {r['long_pct']:>6.1f}")

    # Save results
    os.makedirs("experiments", exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = f"experiments/cascade_sweep_phase{phase_label}_{ts}.json"
    with open(out_path, "w") as f:
        json.dump({"phase": phase_label, "coins": coins, "results": results}, f, indent=2)
    print(f"\nSaved: {out_path}")

    # Recommendation for next phase
    if phase_label == "1a":
        # Find best modes
        hc = [r for r in results if "hard_consensus" in r["config"]]
        dd = [r for r in results if "dual_dominant" in r["config"]]
        ld = [r for r in results if "lstm_dominant" in r["config"]]

        def score(r):
            return (r["wr_overall"] - 60) * r["total_trades"] / 100

        best_hc = sorted(hc, key=score, reverse=True)[0] if hc else None
        best_dd = sorted(dd, key=score, reverse=True)[0] if dd else None
        best_ld = sorted(ld, key=score, reverse=True)[0] if ld else None

        print("\n--- Phase 1b recommendation ---")
        ranked = sorted(
            [(m, r) for m, r in [("hard_consensus", best_hc), ("dual_dominant", best_dd), ("lstm_dominant", best_ld)] if r],
            key=lambda x: score(x[1]), reverse=True
        )
        top2 = [m for m, _ in ranked[:2]]
        print(f"  Best 2 modes: {top2}")
        print(f"  Run: python experiments/cascade_sweep.py --phase 1 --best-modes {' '.join(top2)}")


if __name__ == "__main__":
    main()
