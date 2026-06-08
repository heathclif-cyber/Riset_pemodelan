import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
import torch
import warnings
import logging
from datetime import datetime, timezone
from pathlib import Path

warnings.filterwarnings("ignore")
# Set up paths
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from config import (
    TRAINING_COINS, ALL_COINS, SYMBOL_MAP,
    MODEL_DIR, HOLDOUT_DIR, LABEL_MAP,
    MODAL_PER_TRADE, LEVERAGE_SIM, FEE_PER_SIDE, SLIPPAGE_PER_SIDE,
    MAX_HOLDING_BARS, SWING_LABEL_MIN_RR, SWING_LABEL_MIN_TP, SWING_LABEL_MAX_SL,
)
from core.models import load_lstm
from core.evaluator import full_trading_report
from core.utils import setup_logger, ensure_utc_index
import pipeline.backtest_utils as bu

logger = setup_logger("run_sweep")
logging.getLogger("evaluator").setLevel(logging.WARNING)
DEVICE = torch.device("cpu")
HOLDOUT_LABEL_DIR = HOLDOUT_DIR / "labeled"

def fast_predict(
    lgbm_proba,  # (N, 3)
    lstm_proba,  # (N, 3)
    h4_trend,    # (N,)
    long_thr,
    short_thr,
    conf_entry,
    opp_pen,
    trend_align
):
    n = len(lgbm_proba)
    y_pred = np.ones(n, dtype=np.int64)
    confidence = np.full(n, 1.0 / 3)
    
    lgbm_long_conf = lgbm_proba[:, 2]
    lgbm_short_conf = lgbm_proba[:, 0]
    
    # 1. Determine base LGBM direction and confidence
    is_long_candidate = lgbm_long_conf >= lgbm_short_conf
    lgbm_dir = np.where(is_long_candidate, 2, 0)
    lgbm_conf = np.where(is_long_candidate, lgbm_long_conf, lgbm_short_conf)
    lgbm_thr = np.where(is_long_candidate, long_thr, short_thr)
    
    # 2. Check if passes raw LGBM threshold
    passes_lgbm = (lgbm_long_conf >= long_thr) | (lgbm_short_conf >= short_thr)
    
    # 3. LSTM adjustment
    lstm_dir = np.argmax(lstm_proba, axis=1)
    
    is_agree = lstm_dir == lgbm_dir
    is_neutral = lstm_dir == 1
    
    adj = np.where(is_agree, 0.05, np.where(is_neutral, 0.0, -opp_pen))
    adj_conf = np.clip(lgbm_conf + adj, 0.0, 1.0)
    
    # 4. Trend alignment
    if trend_align and h4_trend is not None:
        is_h4_up = h4_trend > 0
        is_h4_down = h4_trend < 0
        
        is_with_trend = ((lgbm_dir == 2) & is_h4_up) | ((lgbm_dir == 0) & is_h4_down)
        is_counter = ((lgbm_dir == 2) & is_h4_down) | ((lgbm_dir == 0) & is_h4_up)
        
        valid_trend = ~np.isnan(h4_trend)
        
        trend_adj = np.where(is_with_trend & valid_trend, -0.10, np.where(is_counter & valid_trend, 0.05, 0.0))
        adj_conf = np.clip(adj_conf + trend_adj, 0.0, 1.0)
        
    # 5. Final decision & final gate
    final_thr = np.maximum(lgbm_thr, conf_entry)
    passes_final = passes_lgbm & (adj_conf >= final_thr)
    
    y_pred = np.where(passes_final, lgbm_dir, 1)
    confidence = np.where(passes_final, adj_conf, 1.0 / 3)
    
    return y_pred, confidence

def main():
    # 1. Load models
    logger.info("Loading models into memory...")
    lgbm_model = joblib.load(MODEL_DIR / "lgbm_baseline.pkl")
    lstm_model = load_lstm(MODEL_DIR / "lstm_best.pt", device=str(DEVICE)).to(DEVICE)
    lstm_scaler = joblib.load(MODEL_DIR / "lstm_scaler.pkl")
    
    with open(MODEL_DIR / "feature_cols_v2.json") as f:
        feat_cols = json.load(f)
        
    guardian_model = joblib.load(MODEL_DIR / "guardian_best.pkl")
    guardian_scaler = joblib.load(MODEL_DIR / "guardian_scaler.pkl")
    
    # 2. Load and pre-align data for all 21 coins, and pre-compute model probabilities
    logger.info("Loading and pre-computing predictions for all 21 coins...")
    coin_data = {}
    
    # Get feature list expected by LGBM
    gbm_feats = lgbm_model.feature_name_
    
    for symbol in TRAINING_COINS:
        path = HOLDOUT_LABEL_DIR / f"{symbol}_features_v3.parquet"
        if not path.exists():
            logger.warning(f"[{symbol}] File not found, skipping.")
            continue
            
        df = pd.read_parquet(path)
        df = ensure_utc_index(df)
        df = df.sort_index()
        
        mask = df["label"].astype(str).isin(LABEL_MAP)
        df = df[mask].copy()
        y = df["label"].map(LABEL_MAP).values.astype(np.int64)
        
        # Align columns for LSTM
        X = np.zeros((len(df), len(feat_cols)), dtype=np.float64)
        for idx, col in enumerate(feat_cols):
            if col in df.columns:
                X[:, idx] = df[col].ffill().fillna(0).values.astype(np.float64)
                
        # Align columns for LGBM
        X_pred = np.zeros((len(df), len(gbm_feats)), dtype=np.float64)
        for idx, col in enumerate(gbm_feats):
            if col in df.columns:
                X_pred[:, idx] = df[col].values.astype(np.float64)
                
        # Run predictions once
        lgbm_proba = lgbm_model.predict_proba(X_pred)
        lstm_proba = bu.get_lstm_proba(lstm_model, lstm_scaler, X, len(df))
        
        # Guardian static features
        guardian_feat_path = MODEL_DIR / "guardian_feature_cols.json"
        X_guardian = None
        if guardian_feat_path.exists():
            with open(guardian_feat_path) as f:
                g_feat_cols = json.load(f)
            g_static = [c for c in g_feat_cols if c not in [
                "bars_held_norm", "current_pnl_pct", "current_pnl_atr",
                "max_favorable_pnl_pct", "drawdown_from_peak_pct",
                "direction", "entry_price_ratio",
            ]]
            X_guardian = bu.compute_guardian_static_array(df, g_static)
            
        coin_data[symbol] = {
            "df": df,
            "y": y,
            "lgbm_proba": lgbm_proba,
            "lstm_proba": lstm_proba,
            "X_guardian": X_guardian,
            "atr": df["atr_14_h1"].values if "atr_14_h1" in df.columns else np.ones(len(df)),
            "close": df["close"].values if "close" in df.columns else np.ones(len(df)),
            "high": df["high"].values if "high" in df.columns else np.ones(len(df)),
            "low": df["low"].values if "low" in df.columns else np.ones(len(df)),
            "h4_swing_high": df["h4_swing_high"].values if "h4_swing_high" in df.columns else None,
            "h4_swing_low": df["h4_swing_low"].values if "h4_swing_low" in df.columns else None,
            "h4_trend": df["h4_trend"].values if "h4_trend" in df.columns else None,
            "vol_ratio_20": df["vol_ratio_20"].values if "vol_ratio_20" in df.columns else None,
        }
        
    logger.info(f"Pre-loaded {len(coin_data)} coins successfully.")

    # 3. Define sweep parameters
    # Kita hanya membandingkan 5 variasi parameter utama untuk Long & Short sesuai kebutuhan:
    valid_configs = [
        {"lgbm_long": 0.60, "lgbm_short": 0.60, "conf_entry": 0.60, "opp_pen": 0.99, "trend_align": True}, # Baseline (Gate 0.60)
        {"lgbm_long": 0.70, "lgbm_short": 0.60, "conf_entry": 0.60, "opp_pen": 0.99, "trend_align": True}, # Moderate
        {"lgbm_long": 0.75, "lgbm_short": 0.60, "conf_entry": 0.60, "opp_pen": 0.99, "trend_align": True}, # Optimal (Rank 5)
        {"lgbm_long": 0.80, "lgbm_short": 0.65, "conf_entry": 0.65, "opp_pen": 0.99, "trend_align": True}, # Strict
        {"lgbm_long": 0.55, "lgbm_short": 0.55, "conf_entry": 0.55, "opp_pen": 0.99, "trend_align": True}, # Aggressive
    ]
                        
    total_runs = len(valid_configs)
    logger.info(f"Filtered parameter combinations to test (non-redundant): {total_runs}")
    
    sweep_results = []
    
    # 4. Sweep loop
    for run_idx, cfg in enumerate(valid_configs):
        long_thr = cfg["lgbm_long"]
        short_thr = cfg["lgbm_short"]
        conf_entry = cfg["conf_entry"]
        opp_pen = cfg["opp_pen"]
        trend_align = cfg["trend_align"]
        
        if (run_idx + 1) % 10 == 0 or run_idx == 0:
            logger.info(f"Evaluating combination {run_idx + 1}/{total_runs}...")
            
        # Evaluate all loaded coins
        total_trades_count = 0
        portfolio_pnl = 0.0
        coin_winrates = []
        coin_drawdowns = []
        coin_sharpes = []
        
        for symbol, data in coin_data.items():
            # Fast vectorized prediction
            y_pred_filtered, confidence = fast_predict(
                lgbm_proba=data["lgbm_proba"],
                lstm_proba=data["lstm_proba"],
                h4_trend=data["h4_trend"],
                long_thr=long_thr,
                short_thr=short_thr,
                conf_entry=conf_entry,
                opp_pen=opp_pen,
                trend_align=trend_align
            )
            
            # Run trading simulation
            report = full_trading_report(
                y_pred=y_pred_filtered,
                y_actual=data["y"],
                atr=data["atr"],
                close=data["close"],
                high=data["high"],
                low=data["low"],
                h4_swing_highs=data["h4_swing_high"],
                h4_swing_lows=data["h4_swing_low"],
                index=data["df"].index,
                modal=MODAL_PER_TRADE,
                leverages=LEVERAGE_SIM,
                fee_per_side=FEE_PER_SIDE,
                slippage=SLIPPAGE_PER_SIDE,
                min_rr=SWING_LABEL_MIN_RR,
                min_tp_atr=SWING_LABEL_MIN_TP,
                max_sl_atr=SWING_LABEL_MAX_SL,
                max_hold=MAX_HOLDING_BARS,
                symbol=symbol,
                confidence=confidence,
                guardian_model=guardian_model,
                guardian_scaler=guardian_scaler,
                X_guardian=data["X_guardian"],
                guardian_enabled=True,
                guardian_exit_threshold=0.65,
                trailing_stop_enabled=False,
                vol_ratio=data["vol_ratio_20"],
                h4_trend=data["h4_trend"]
            )
            
            pnl = report.get("pnl_lev5x", 0.0)
            wr = report.get("winrate", 0.0)
            dd = report.get("max_drawdown_lev5x", 0.0)
            sh = report.get("sharpe_ratio", 0.0)
            tc = report.get("total_trades", 0)
            
            portfolio_pnl += pnl
            total_trades_count += tc
            coin_winrates.append(wr)
            coin_drawdowns.append(dd)
            coin_sharpes.append(sh)
            
        # Compute portfolio metrics
        mean_wr = np.mean(coin_winrates) if coin_winrates else 0.0
        mean_dd = np.mean(coin_drawdowns) if coin_drawdowns else 0.0
        mean_sh = np.mean(coin_sharpes) if coin_sharpes else 0.0
        
        sweep_results.append({
            "lgbm_long": long_thr,
            "lgbm_short": short_thr,
            "conf_entry": conf_entry,
            "opp_pen": opp_pen,
            "trend_align": trend_align,
            "portfolio_pnl": portfolio_pnl,
            "total_trades": total_trades_count,
            "mean_winrate": mean_wr,
            "mean_drawdown": mean_dd,
            "mean_sharpe": mean_sh,
        })
                        
    # 5. Sort results by Net PnL
    df_results = pd.DataFrame(sweep_results)
    df_results = df_results.sort_values(by="portfolio_pnl", ascending=False).reset_index(drop=True)
    
    # Save results to JSON
    out_dir = ROOT / "reports" / "experiments"
    out_dir.mkdir(parents=True, exist_ok=True)
    df_results.to_json(out_dir / "parameter_sweep_holdout.json", indent=2, orient="records")
    
    # Print top 15 results
    print("\n" + "=" * 100)
    print("  TOP 15 PARAMETER COMBINATIONS ON HOLDOUT DATA (Sorted by Net PnL)")
    print("=" * 100)
    print(f"{'Rank':<5} | {'LGBM_L':<6} | {'LGBM_S':<6} | {'Gate':<5} | {'Pen':<5} | {'Trend':<5} | {'Net PnL ($)':<12} | {'Trades':<6} | {'WinRate':<7} | {'Mean DD':<7} | {'Sharpe':<6}")
    print("-" * 100)
    
    for i in range(min(15, len(df_results))):
        r = df_results.iloc[i]
        print(f"{i+1:<5} | {r['lgbm_long']:<6.2f} | {r['lgbm_short']:<6.2f} | {r['conf_entry']:<5.2f} | {r['opp_pen']:<5.2f} | {str(r['trend_align']):<5} | {r['portfolio_pnl']:<12.2f} | {int(r['total_trades']):<6} | {r['mean_winrate']:<7.2%} | {r['mean_drawdown']:<7.2%} | {r['mean_sharpe']:<6.2f}")
        
    print("=" * 100)
    print(f"Full results saved to: {out_dir / 'parameter_sweep_holdout.json'}")
    
    # Create Markdown report
    md_lines = [
        "# 📊 Parameter Sweep Report: Holdout Optimization",
        f"Generated at: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC",
        "",
        "Berikut adalah hasil simulasi dari kombinasi parameter untuk menemukan setelan paling optimal (sweet spot) tanpa terjadi redundancy (mubazir) antar gate.",
        "",
        "| Rank | LGBM LONG | LGBM SHORT | Final Gate | Opp. Pen. | Trend Align | Net PnL ($) | Total Trades | Win Rate | Mean DD (5x) | Sharpe |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |"
    ]
    
    for i in range(min(20, len(df_results))):
        r = df_results.iloc[i]
        md_lines.append(
            f"| {i+1} | {r['lgbm_long']:.2f} | {r['lgbm_short']:.2f} | {r['conf_entry']:.2f} | {r['opp_pen']:.2f} | {r['trend_align']} | **`{r['portfolio_pnl']:.2f}`** | {int(r['total_trades'])} | {r['mean_winrate']:.2%} | {r['mean_drawdown']:.2%} | {r['mean_sharpe']:.2f} |"
        )
        
    with open(out_dir / "parameter_sweep_holdout_report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))
    logger.info(f"Markdown report saved to: {out_dir / 'parameter_sweep_holdout_report.md'}")

if __name__ == "__main__":
    main()
