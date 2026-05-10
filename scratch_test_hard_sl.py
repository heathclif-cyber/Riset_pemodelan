import sys
from pathlib import Path
import numpy as np
import pandas as pd
import warnings
import json

warnings.filterwarnings("ignore")
ROOT = Path("d:/Apps-Dev/Riset_pemodelan")
sys.path.insert(0, str(ROOT))

from pipeline.visualize_aspects import load_models, load_coin_data, get_signals, run_sim

def main():
    lgbm, lstm, scaler, feat_cols = load_models()
    holdout_dir = ROOT / "data" / "holdout" / "labeled"
    symbols = sorted([p.stem.replace("_features_v3", "") for p in holdout_dir.glob("*_features_v3.parquet")])
    
    spec_exact = {
        "hybrid_mode": True, "swing_freshness_check": True, "structural_filter": True,
        "min_rr": 0.5, "min_tp_atr": 1.2, "max_sl_atr": 3.0,
        "sl_fallback_atr": 1.5, "tp_fallback_atr": 2.0, "slippage_enabled": True,
        "sl_trigger_mode": "highlow", "sizing_mode": "fixed", "cooldown_enabled": False,
        "swing_sl_bumper_atr": 0.5, "structural_tolerance_pct": 0.0
    }
    spec_implementation = {
        "hybrid_mode": True, "swing_freshness_check": True, "structural_filter": True,
        "min_rr": 0.5, "min_tp_atr": 1.2, "max_sl_atr": 3.0,
        "sl_fallback_atr": 1.5, "tp_fallback_atr": 2.0, "slippage_enabled": True,
        "sl_trigger_mode": "highlow", "sizing_mode": "fixed", "cooldown_enabled": False,
        "swing_sl_bumper_atr": 0.5, "structural_tolerance_pct": 0.04
    }
    
    results = {
        "1. Spec Asli (Toleransi Breakout 0%)": [],
        "2. Implementasi Anda (Toleransi Breakout 4%)": []
    }
    
    for label, params in [
        ("1. Spec Asli (Toleransi Breakout 0%)", spec_exact),
        ("2. Implementasi Anda (Toleransi Breakout 4%)", spec_implementation),
    ]:
        print(f"Running {label}...")
        for sym in symbols:
            data = load_coin_data(sym, feat_cols)
            if data is None: continue
            df, X, vc = data
            y_pred, conf = get_signals(df, X, vc, lgbm, lstm, scaler)
            sim = run_sim(df, y_pred, conf, **params)
            if not sim.get("error"):
                results[label].append(sim)
                
    print("\n" + "="*50)
    print("HASIL PERBANDINGAN")
    print("="*50)
    
    for label, sims in results.items():
        avg_wr = np.mean([s["winrate"] for s in sims])
        total_trades = int(np.mean([s["total_trades"] for s in sims]))
        avg_pnl = np.mean([s["total_pnl"] for s in sims])
        avg_dd = np.mean([s.get("max_drawdown", 0) for s in sims])
        print(f"[{label}]")
        print(f"  Winrate: {avg_wr:.1%}")
        print(f"  Trades (avg/coin): {total_trades}")
        print(f"  PnL (avg/coin): ${avg_pnl:+.0f}")
        print(f"  Max Drawdown: {avg_dd:.1%}")
        print("-" * 50)

if __name__ == "__main__":
    main()
