"""
Export granular trade history CSV — coin, entry/exit timestamp, direction, PnL, outcome.
Output: reports/trade_history_holdout.csv
"""
import json, sys, warnings, numpy as np, pandas as pd
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
warnings.filterwarnings("ignore")

from config import *
from core.evaluator import simulate_trades_swing
from core.utils import ensure_utc_index
from pipeline.backtest_utils import hierarchical_predict
from core.models import load_lstm
import joblib

HOLDOUT = Path(__file__).parent.parent / "data" / "holdout" / "labeled"
MODEL_DIR = Path(__file__).parent.parent / "models"
OUT = Path(__file__).parent.parent / "reports" / "trade_history_holdout.csv"

lgbm = joblib.load(MODEL_DIR / "lgbm_baseline.pkl")
lstm = load_lstm(MODEL_DIR / "lstm_best.pt", device="cpu")
scaler = joblib.load(MODEL_DIR / "lstm_scaler.pkl")
with open(MODEL_DIR / "feature_cols_v2.json") as f:
    feat_cols = json.load(f)

coins = sorted([p.stem.replace("_features_v3", "") for p in HOLDOUT.glob("*_features_v3.parquet")])

all_rows = []
for idx, sym in enumerate(coins, 1):
    df = pd.read_parquet(HOLDOUT / f"{sym}_features_v3.parquet")
    df = ensure_utc_index(df).sort_index()
    mask = df["label"].astype(str).isin(LABEL_MAP)
    df = df[mask].copy()
    valid = [c for c in feat_cols if c in df.columns]
    df[valid] = df[valid].ffill().fillna(0)
    X = df[valid].values.astype(np.float64)

    y_pred, conf = hierarchical_predict(None, lgbm, lstm, scaler, X, valid, [], df, trend_alignment_enabled=False)
    below = (y_pred != 1) & (conf < CONFIDENCE_THRESHOLD_ENTRY)
    y_pred[below] = 1

    result = simulate_trades_swing(
        y_pred=y_pred, close=df["close"].values, high=df["high"].values,
        low=df["low"].values, atr=df["atr_14_h1"].values,
        h4_swing_highs=df["h4_swing_high"].values if "h4_swing_high" in df.columns else None,
        h4_swing_lows=df["h4_swing_low"].values if "h4_swing_low" in df.columns else None,
        modal=MODAL_PER_TRADE, leverage=LEVERAGE_SIM[0],
        fee_per_side=FEE_PER_SIDE, slippage=SLIPPAGE_PER_SIDE,
        max_hold=MAX_HOLDING_BARS,
        min_rr=SWING_LABEL_MIN_RR, min_tp_atr=SWING_LABEL_MIN_TP,
        max_sl_atr=SWING_LABEL_MAX_SL, confidence=conf,
        hybrid_mode=TP_SL_HYBRID_MODE,
        swing_freshness_check=TP_SL_SWING_FRESHNESS,
        structural_filter=TP_SL_STRUCTURAL_FILTER,
        structural_tolerance_pct=TP_SL_STRUCTURAL_TOLERANCE,
        slippage_enabled=TP_SL_SLIPPAGE_ENABLED,
        sizing_mode=TP_SL_SIZING_MODE,
        cooldown_enabled=TP_SL_COOLDOWN_ENABLED,
        swing_sl_bumper_atr=0.5,
    )

    timestamps = df.index
    for t in result.get("trades", []):
        all_rows.append({
            "coin": sym,
            "entry_time": timestamps[t["bar_in"]],
            "exit_time": timestamps[t["bar_out"]] if t["bar_out"] < len(timestamps) else timestamps[-1],
            "direction": t["direction"],
            "entry_price": t["entry"],
            "exit_price": t["exit"],
            "tp": t["tp"],
            "sl": t["sl"],
            "rr": t["rr"],
            "outcome": t["outcome"],
            "net_pnl": t["net_pnl"],
        })

    print(f"  [{idx:2d}/21] {sym:<14} {len(result.get('trades', [])):>4d} trades")

df_out = pd.DataFrame(all_rows)
df_out.to_csv(OUT, index=False)
print(f"\nExported {len(df_out):,} trades to {OUT}")
print(f"Columns: {list(df_out.columns)}")
print(f"\nSample (5 rows):")
print(df_out.head().to_string())
print(f"\nSummary by outcome:")
print(df_out.groupby("outcome").agg(trades=("net_pnl", "count"), total_pnl=("net_pnl", "sum"), avg_pnl=("net_pnl", "mean")).to_string())
