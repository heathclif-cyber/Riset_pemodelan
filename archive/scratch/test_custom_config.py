import os
import sys
import joblib
import pandas as pd
import numpy as np
import torch
from pathlib import Path

# Add root directory to python path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from config import (
    ALL_COINS, MODEL_DIR, HOLDOUT_DIR,
    LABEL_MAP, NUM_CLASSES,
    MODAL_PER_TRADE, LEVERAGE_SIM, FEE_PER_SIDE, SLIPPAGE_PER_SIDE,
    SWING_LABEL_MIN_RR, SWING_LABEL_MIN_TP, SWING_LABEL_MAX_SL,
    MAX_HOLDING_BARS, TP_SL_FALLBACK_TP, TP_SL_FALLBACK_SL
)
from core.models import load_lstm
from core.evaluator import full_trading_report
from pipeline.backtest_utils import hierarchical_predict
from core.utils import ensure_utc_index

HOLDOUT_LABEL_DIR = HOLDOUT_DIR / "labeled"
DEVICE = torch.device("cpu")

def run_simulation(lgbm_long, lgbm_short, gate_entry, pyramiding_enabled, pyramiding_max):
    # 0. Patch simulation function at runtime to inject custom pyramiding settings
    import core.evaluator
    original_sim = getattr(core.evaluator, "_original_simulate_trades_swing", core.evaluator.simulate_trades_swing)
    if not hasattr(core.evaluator, "_original_simulate_trades_swing"):
        core.evaluator._original_simulate_trades_swing = original_sim
        
    def patched_sim(*args, **kwargs):
        kwargs["pyramiding_enabled"] = pyramiding_enabled
        kwargs["pyramiding_max_per_coin"] = pyramiding_max
        return original_sim(*args, **kwargs)
        
    core.evaluator.simulate_trades_swing = patched_sim

    # 1. Load Models
    lgbm_model = joblib.load(MODEL_DIR / "lgbm_baseline.pkl")
    lstm_model = load_lstm(MODEL_DIR / "lstm_best.pt", device=DEVICE)
    lstm_scaler = joblib.load(MODEL_DIR / "lstm_scaler.pkl")
    
    # 2. Load feature columns list
    with open(MODEL_DIR / "feature_cols_v2.json") as f:
        import json
        feat_cols = json.load(f)
    
    # 3. Load Guardian
    from config import GUARDIAN_ENABLED
    guardian_model = joblib.load(MODEL_DIR / "guardian_best.pkl")
    guardian_scaler = joblib.load(MODEL_DIR / "guardian_scaler.pkl")
    with open(MODEL_DIR / "guardian_feature_cols.json") as f:
        g_feat_cols = json.load(f)
    g_static = [c for c in g_feat_cols if c not in [
        "bars_held_norm", "current_pnl_pct", "current_pnl_atr",
        "max_favorable_pnl_pct", "drawdown_from_peak_pct",
        "direction", "entry_price_ratio",
    ]]

    # 4. Load Data & Predict
    coin_data = {}
    for symbol in ALL_COINS:
        path = HOLDOUT_LABEL_DIR / f"{symbol}_features_v3.parquet"
        if not path.exists():
            continue
        df = pd.read_parquet(path)
        df = ensure_utc_index(df)
        df = df.sort_index()
        
        mask = df["label"].astype(str).isin(LABEL_MAP)
        df = df[mask].copy()
        y = df["label"].map(LABEL_MAP).values.astype(np.int64)
        
        # Align columns
        X = np.zeros((len(df), len(feat_cols)), dtype=np.float64)
        for idx, col in enumerate(feat_cols):
            if col in df.columns:
                X[:, idx] = df[col].ffill().fillna(0).values.astype(np.float64)
                
        # Patch config
        import config
        config.LGBM_THRESHOLD_LONG = lgbm_long
        config.LGBM_THRESHOLD_SHORT = lgbm_short
        
        y_pred, confidence = hierarchical_predict(
            None, lgbm_model, lstm_model, lstm_scaler,
            X, feat_cols, [], df,
        )
        
        # Filter confidence by entry gate
        below = (y_pred != 1) & (confidence < gate_entry)
        y_pred_filtered = y_pred.copy()
        y_pred_filtered[below] = 1
        
        # Prepare guardian
        from pipeline.backtest_utils import compute_guardian_static_array
        X_guardian = compute_guardian_static_array(df, g_static)
        
        coin_data[symbol] = {
            "y_pred_filtered": y_pred_filtered,
            "y": y,
            "df": df,
            "X_guardian": X_guardian,
            "confidence": confidence
        }

    # 5. Run evaluator for each coin
    reports = []
    for symbol, data in coin_data.items():
        df = data["df"]
        atr_arr   = df["atr_14_h1"].values if "atr_14_h1" in df.columns else np.ones(len(df))
        close_arr = df["close"].values    if "close"      in df.columns else np.ones(len(df))
        high_arr  = df["high"].values     if "high"       in df.columns else close_arr
        low_arr   = df["low"].values      if "low"        in df.columns else close_arr
        h4_sh_arr = df["h4_swing_high"].values if "h4_swing_high" in df.columns else None
        h4_sl_arr = df["h4_swing_low"].values  if "h4_swing_low"  in df.columns else None
        h4_trend_arr = df["h4_trend"].values if "h4_trend" in df.columns else None
        vol_ratio_arr = df["vol_ratio_20"].values if "vol_ratio_20" in df.columns else None

        rep = full_trading_report(
            y_pred         = data["y_pred_filtered"],
            y_actual       = data["y"],
            atr            = atr_arr,
            close          = close_arr,
            high           = high_arr,
            low            = low_arr,
            h4_swing_highs = h4_sh_arr,
            h4_swing_lows  = h4_sl_arr,
            index          = df.index,
            modal          = MODAL_PER_TRADE,
            leverages      = LEVERAGE_SIM,
            fee_per_side   = FEE_PER_SIDE,
            slippage       = SLIPPAGE_PER_SIDE,
            min_rr         = SWING_LABEL_MIN_RR,
            min_tp_atr     = SWING_LABEL_MIN_TP,
            max_sl_atr     = SWING_LABEL_MAX_SL,
            max_hold       = MAX_HOLDING_BARS,
            symbol         = symbol,
            confidence     = data["confidence"],
            guardian_model  = guardian_model,
            guardian_scaler = guardian_scaler,
            X_guardian      = data["X_guardian"],
            guardian_enabled = GUARDIAN_ENABLED,
            vol_ratio       = vol_ratio_arr,
            h4_trend        = h4_trend_arr
        )
        reports.append(rep)

    # 6. Aggregate results
    total_trades = sum(len(r.get("trades", [])) for r in reports)
    total_pnl = sum(sum(t["net_pnl"] for t in r.get("trades", [])) for r in reports)
    
    # Calculate average metrics across coins
    win_rates = []
    sharpes = []
    drawdowns = []
    
    for r in reports:
        trades = r.get("trades", [])
        if not trades:
            continue
        wins = sum(1 for t in trades if t["net_pnl"] > 0)
        win_rates.append(wins / len(trades))
        
        # Drawdown calculation
        eq = [t["equity"] for t in trades]
        if len(eq) > 1:
            peak = eq[0]
            dd_list = []
            for val in eq:
                if val > peak:
                    peak = val
                dd = (peak - val) / peak
                dd_list.append(dd)
            drawdowns.append(max(dd_list))
            
            # Sharpe calculation
            sharpes.append(r.get("sharpe_ratio", 0))
        else:
            drawdowns.append(0)
            sharpes.append(0)
            
    mean_winrate = np.mean(win_rates) if win_rates else 0.0
    mean_sharpe = np.mean(sharpes) if sharpes else 0.0
    mean_dd = np.mean(drawdowns) if drawdowns else 0.0
    
    print(f"| {lgbm_long:<6} | {lgbm_short:<6} | {gate_entry:<5} | {str(pyramiding_enabled):<5} (Max {pyramiding_max}) | ${total_pnl:<11.2f} | {total_trades:<6} | {mean_winrate*100:<7.2f}% | {mean_dd*100:<7.2f}% | {mean_sharpe:<6.2f} |")

if __name__ == "__main__":
    print("-" * 100)
    print(f"| LONG   | SHORT  | Gate  | Pyram (Limit)  | Net PnL ($) | Trades | WinRate | Mean DD | Sharpe |")
    print("-" * 100)
    # Test LGBM_L=0.65, LGBM_S=0.60
    # Gate 0.65, Pyramiding ON (Limit 3)
    run_simulation(0.65, 0.60, 0.65, True, 3)
    # Gate 0.60, Pyramiding ON (Limit 3)
    run_simulation(0.65, 0.60, 0.60, True, 3)
    # Gate 0.65, Pyramiding OFF (Limit 1 - Real Live Setup)
    run_simulation(0.65, 0.60, 0.65, True, 1)
    # Gate 0.60, Pyramiding OFF (Limit 1 - Real Live Setup)
    run_simulation(0.65, 0.60, 0.60, True, 1)
    print("-" * 100)
