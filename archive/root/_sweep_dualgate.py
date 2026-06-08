"""
Dual Gate Grid Sweep — LGBM gate x LSTM gate
  LGBM: [0.40, 0.50, 0.60]
  LSTM: [0.37, 0.39, 0.41]  (step 0.02 dalam zone 0.37-0.42)

Jalankan:
  python _sweep_dualgate.py
"""

import sys, json, warnings, numpy as np, pandas as pd
from pathlib import Path
from itertools import product
from datetime import datetime

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

# ── grid ───────────────────────────────────────────────────────────────────────
LGBM_GATES = [0.40, 0.50, 0.60]
LSTM_GATES  = [0.37, 0.39, 0.41]

# ── load semua yang dibutuhkan dari 07_holdout_backtest ───────────────────────
from config import (
    ALL_COINS, MODEL_DIR,
    GUARDIAN_ENABLED, GUARDIAN_EXIT_THRESHOLD,
    TRAILING_STOP_ENABLED, TRAILING_STOP_ATR, TRAILING_STOP_MIN_BARS,
)
import pipeline.backtest_utils as _bu
import joblib, torch
from core.models import load_lstm

RUN_DIR = MODEL_DIR / "runs" / "cascade_v2.5_hybrid_accel"

print("Loading models...", flush=True)
lgbm_model      = joblib.load(RUN_DIR / "lgbm.pkl")
guardian_model  = joblib.load(RUN_DIR / "guardian.pkl")
guardian_scaler = joblib.load(RUN_DIR / "guardian_scaler.pkl")
lstm_scaler     = joblib.load(RUN_DIR / "lstm_v2_style_scaler.pkl")
lstm_model      = load_lstm(str(RUN_DIR / "lstm_v2_style.pt"), device=torch.device("cpu"))
lstm_model.eval()

with open(RUN_DIR / "feature_cols_v2.json") as f:
    feat_cols = json.load(f)

print(f"Models loaded | {len(feat_cols)} features", flush=True)

# ── Gunakan backtest_holdout_symbol dari 07_holdout_backtest ─────────────────
# Import fungsi langsung — lebih aman karena sudah handle semua args
import importlib.util, types

spec = importlib.util.spec_from_file_location(
    "holdout07", ROOT / "pipeline" / "07_holdout_backtest.py"
)
mod = types.ModuleType("holdout07")
spec.loader.exec_module(mod)        # jalankan module-level code
backtest_coin = mod.backtest_holdout_symbol

# ── run sweep ──────────────────────────────────────────────────────────────────
print()
print("=" * 70)
print("  DUAL GATE SWEEP: LGBM gate x LSTM gate (dual_gate mode)")
print("=" * 70)
print(f"{'LGBM':>6} {'LSTM':>6} | {'WR%':>6} {'DD%':>7} {'Sharpe':>7} {'PF':>6} {'PnL ($)':>9} {'Trd/mo':>7} {'N':>6}")
print("-" * 70)

results = []

for lgbm_gate, lstm_gate in product(LGBM_GATES, LSTM_GATES):
    # Patch globals
    _bu.SMART_ENTRY_MODE      = "dual_gate"
    _bu.SMART_ENTRY_LGBM_GATE = lgbm_gate
    _bu.SMART_ENTRY_LSTM_GATE = lstm_gate
    _bu.LSTM_STANDALONE_ENABLED           = False
    _bu.MOMENTUM_DYNAMIC_THRESHOLD_ENABLED = False
    _bu.TREND_DYNAMIC_THRESHOLD_ENABLED   = False

    print(f"  Running LGBM={lgbm_gate:.2f} LSTM={lstm_gate:.2f}...", end="", flush=True)

    winrates, drawdowns, sharpes, pfs, pnls, trades_list = [], [], [], [], [], []

    for coin in ALL_COINS:
        try:
            rep = backtest_coin(
                symbol           = coin,
                feat_cols        = feat_cols,
                lgbm_model       = lgbm_model,
                lstm_model       = lstm_model,
                lstm_scaler      = lstm_scaler,
                guardian_model   = guardian_model,
                guardian_scaler  = guardian_scaler,
                guardian_enabled = GUARDIAN_ENABLED,
                guardian_exit_threshold = GUARDIAN_EXIT_THRESHOLD,
                trailing_stop_enabled   = TRAILING_STOP_ENABLED,
                trailing_stop_atr       = TRAILING_STOP_ATR,
                trailing_stop_min_bars  = TRAILING_STOP_MIN_BARS,
            )
            if rep is None:
                continue
            winrates.append(rep.get("winrate", 0.0))
            drawdowns.append(rep.get("max_drawdown_lev5x", 0.0))
            sharpes.append(rep.get("sharpe", 0.0))
            pfs.append(rep.get("profit_factor", 0.0))
            pnls.append(rep.get("total_pnl_lev5x", 0.0))
            trades_list.append(rep.get("total_trades", 0))
        except Exception as e:
            pass   # skip coin dengan error

    if not winrates:
        print(f"\r{lgbm_gate:>6.2f} {lstm_gate:>6.2f} | NO DATA")
        continue

    n_coins   = len(winrates)
    n_months  = 5.0
    total_t   = sum(trades_list)
    total_pnl = sum(pnls)
    row = {
        "lgbm_gate":     lgbm_gate,
        "lstm_gate":     lstm_gate,
        "mean_wr":       round(np.mean(winrates) * 100, 2),
        "mean_dd":       round(np.mean(drawdowns) * 100, 2),
        "mean_sharpe":   round(np.mean(sharpes), 2),
        "mean_pf":       round(np.mean(pfs), 2),
        "total_pnl":     round(total_pnl, 2),
        "total_trades":  total_t,
        "trade_per_month": round(total_t / n_months / n_coins, 1),
    }
    results.append(row)
    print(f"\r{lgbm_gate:>6.2f} {lstm_gate:>6.2f} | "
          f"{row['mean_wr']:>5.1f}% {row['mean_dd']:>6.1f}% "
          f"{row['mean_sharpe']:>7.2f} {row['mean_pf']:>6.2f} "
          f"${row['total_pnl']:>8,.0f} {row['trade_per_month']:>7.1f} {total_t:>6}")

print("=" * 70)

if results:
    best_pnl = max(results, key=lambda x: x["total_pnl"])
    best_wr  = max(results, key=lambda x: x["mean_wr"])
    best_sh  = max(results, key=lambda x: x["mean_sharpe"])
    print()
    print(f"Best PnL    : LGBM={best_pnl['lgbm_gate']:.2f} LSTM={best_pnl['lstm_gate']:.2f}"
          f" -> ${best_pnl['total_pnl']:,.0f}, WR={best_pnl['mean_wr']}%")
    print(f"Best WR     : LGBM={best_wr['lgbm_gate']:.2f}  LSTM={best_wr['lstm_gate']:.2f}"
          f" -> WR={best_wr['mean_wr']}%, PnL=${best_wr['total_pnl']:,.0f}")
    print(f"Best Sharpe : LGBM={best_sh['lgbm_gate']:.2f} LSTM={best_sh['lstm_gate']:.2f}"
          f" -> Sharpe={best_sh['mean_sharpe']}, PnL=${best_sh['total_pnl']:,.0f}")

    out = ROOT / "models" / "runs" / f"sweep_dualgate_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(out / "sweep_results.csv", index=False)
    print(f"\nSaved -> {out / 'sweep_results.csv'}")
