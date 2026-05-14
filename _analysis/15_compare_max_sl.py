"""
Quick comparison: max_sl=3.0 (before) vs max_sl=4.0 (after)
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

lgbm = joblib.load(MODEL_DIR / "lgbm_baseline.pkl")
lstm = load_lstm(MODEL_DIR / "lstm_best.pt", device="cpu")
scaler = joblib.load(MODEL_DIR / "lstm_scaler.pkl")
with open(MODEL_DIR / "feature_cols_v2.json") as f:
    feat_cols = json.load(f)

coins = sorted([p.stem.replace("_features_v3", "") for p in HOLDOUT.glob("*_features_v3.parquet")])

def run(symbol, max_sl):
    df = pd.read_parquet(HOLDOUT / f"{symbol}_features_v3.parquet")
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
        max_sl_atr=max_sl, confidence=conf,
        hybrid_mode=TP_SL_HYBRID_MODE,
        swing_freshness_check=TP_SL_SWING_FRESHNESS,
        structural_filter=TP_SL_STRUCTURAL_FILTER,
        structural_tolerance_pct=TP_SL_STRUCTURAL_TOLERANCE,
        slippage_enabled=TP_SL_SLIPPAGE_ENABLED,
        sizing_mode=TP_SL_SIZING_MODE,
        cooldown_enabled=TP_SL_COOLDOWN_ENABLED,
        swing_sl_bumper_atr=0.5,
    )

    trades = result.get("trades", [])
    wins = sum(1 for t in trades if t["outcome"] == "WIN")
    total = len(trades)
    n_months = max((len(df) / 24 / 30.44), 0.1)

    l_t = [t for t in trades if t["direction"] == "LONG"]
    s_t = [t for t in trades if t["direction"] == "SHORT"]
    l_w = sum(1 for t in l_t if t["outcome"] == "WIN")
    s_w = sum(1 for t in s_t if t["outcome"] == "WIN")

    wl = [t["net_pnl"] for t in trades if t["net_pnl"] > 0]
    ll = [t["net_pnl"] for t in trades if t["net_pnl"] < 0]
    avg_w = np.mean(wl) if wl else 0
    avg_l = np.mean(ll) if ll else 0
    pf = abs(sum(wl)) / abs(sum(ll)) if ll else 0

    return {
        "symbol": symbol, "total_trades": total, "wins": wins,
        "losses": result.get("losses", 0), "time_exits": result.get("time_exits", 0),
        "winrate": wins / total if total else 0,
        "winrate_long": l_w / len(l_t) if l_t else 0,
        "winrate_short": s_w / len(s_t) if s_t else 0,
        "pnl": result.get("net_pnl_total", 0),
        "max_dd": result.get("max_drawdown", 0),
        "profit_factor": pf, "avg_win": avg_w, "avg_loss": avg_l,
        "trade_per_month": round(total / n_months, 2),
        "trades_long": len(l_t), "trades_short": len(s_t),
    }

def agg(results):
    t = sum(r["total_trades"] for r in results.values())
    w = sum(r["wins"] for r in results.values())
    l_ = sum(r["losses"] for r in results.values())
    te = sum(r["time_exits"] for r in results.values())
    pnl = sum(r["pnl"] for r in results.values())
    lt = sum(r["trades_long"] for r in results.values())
    st = sum(r["trades_short"] for r in results.values())
    return {
        "total_trades": t, "wins": w, "losses": l_, "time_exits": te,
        "winrate": w / t if t else 0, "pnl": pnl,
        "mean_dd": np.mean([r["max_dd"] for r in results.values()]),
        "mean_pf": np.mean([r["profit_factor"] for r in results.values()]),
        "mean_tpm": np.mean([r["trade_per_month"] for r in results.values()]),
        "mean_aw": np.mean([r["avg_win"] for r in results.values()]),
        "mean_al": np.mean([r["avg_loss"] for r in results.values()]),
        "long_wr": sum(r["trades_long"] * r["winrate_long"] for r in results.values()) / max(lt, 1),
        "short_wr": sum(r["trades_short"] * r["winrate_short"] for r in results.values()) / max(st, 1),
    }

print("Running BEFORE (max_sl=3.0) vs AFTER (max_sl=4.0)...")
r3 = {}; r4 = {}
for i, sym in enumerate(coins, 1):
    r3[sym] = run(sym, 3.0)
    r4[sym] = run(sym, 4.0)
    d = r4[sym]["total_trades"] - r3[sym]["total_trades"]
    dp = r4[sym]["pnl"] - r3[sym]["pnl"]
    print(f"  [{i:2d}/21] {sym:<14} T={r3[sym]['total_trades']:>4d}->{r4[sym]['total_trades']:<4d} (+{d:>3d})  PnL=${r3[sym]['pnl']:>7,.0f}->${r4[sym]['pnl']:>7,.0f} ({dp:>+8,.0f})")

a3 = agg(r3); a4 = agg(r4)

print()
print("=" * 115)
print("  BEFORE (max_sl=3.0)  vs  AFTER (max_sl=4.0)  --  21 COINS HOLD-OUT 2025-05 -> 2026-04")
print("=" * 115)
print(f"  {'Metric':<35} {'BEFORE (3.0)':>22} {'AFTER (4.0)':>22} {'DELTA':>22}")
print(f"  {'-'*35} {'-'*22} {'-'*22} {'-'*22}")

for label, key, fmt in [
    ("Total Trades", "total_trades", "d"),
    ("Total Wins", "wins", "d"),
    ("Total Losses", "losses", "d"),
    ("Time Exits", "time_exits", "d"),
    ("Win Rate", "winrate", "%"),
    ("Win Rate LONG", "long_wr", "%"),
    ("Win Rate SHORT", "short_wr", "%"),
    ("Net PnL", "pnl", "$"),
    ("Avg Win per Trade", "mean_aw", "$"),
    ("Avg Loss per Trade", "mean_al", "$"),
    ("Mean Max Drawdown", "mean_dd", "%"),
    ("Mean Profit Factor", "mean_pf", "x"),
    ("Mean Trade per Month", "mean_tpm", "f"),
]:
    v3, v4 = a3[key], a4[key]
    d = v4 - v3
    if fmt == "$":
        print(f"  {label:<35}  ${v3:>20,.2f}  ${v4:>20,.2f}  ${d:>+20,.2f}")
    elif fmt == "%":
        print(f"  {label:<35}  {v3:>21.2%}  {v4:>21.2%}  {d:>+21.2%}")
    elif fmt == "x":
        print(f"  {label:<35}  {v3:>21.2f}x  {v4:>21.2f}x  {d:>+21.2f}x")
    elif fmt == "f":
        print(f"  {label:<35}  {v3:>21.2f}  {v4:>21.2f}  {d:>+21.2f}")
    else:
        print(f"  {label:<35}  {v3:>22,}  {v4:>22,}  {d:>+22,}")

print()
print("=" * 135)
print("  PER-COIN BREAKDOWN")
print("=" * 135)
hdr = f"  {'Coin':<16} {'T (3.0)':>8} {'T (4.0)':>8} {'+T':>6} {'WR 3.0':>8} {'WR 4.0':>8} {'+WR':>7} {'PnL 3.0':>10} {'PnL 4.0':>10} {'+PnL':>10} {'DD 3.0':>8} {'DD 4.0':>8} {'+DD':>7}"
print(hdr)
print(f"  {'-'*16} {'-'*8} {'-'*8} {'-'*6} {'-'*8} {'-'*8} {'-'*7} {'-'*10} {'-'*10} {'-'*10} {'-'*8} {'-'*8} {'-'*7}")

for sym in coins:
    v3, v4 = r3[sym], r4[sym]
    dt = v4["total_trades"] - v3["total_trades"]
    dw = v4["winrate"] - v3["winrate"]
    dp = v4["pnl"] - v3["pnl"]
    dd = v4["max_dd"] - v3["max_dd"]
    print(f"  {sym:<16} {v3['total_trades']:>8} {v4['total_trades']:>8} {dt:>+6} {v3['winrate']:>7.2%} {v4['winrate']:>7.2%} {dw:>+6.2%} ${v3['pnl']:>9,.0f} ${v4['pnl']:>9,.0f} ${dp:>+9,.0f} {v3['max_dd']:>7.2%} {v4['max_dd']:>7.2%} {dd:>+6.2%}")

# Group summary
print()
print("=" * 115)
train_coins = [c for c in coins if c in TRAINING_COINS]
holdout_coins = [c for c in coins if c not in TRAINING_COINS]
for grp_name, grp in [("5 Training Coins", train_coins), ("16 Holdout Coins", holdout_coins)]:
    g3 = agg({c: r3[c] for c in grp})
    g4 = agg({c: r4[c] for c in grp})
    dt = g4["total_trades"] - g3["total_trades"]
    dw = g4["winrate"] - g3["winrate"]
    dp = g4["pnl"] - g3["pnl"]
    print(f"  {grp_name}:")
    print(f"    Trades: {g3['total_trades']:>5,} -> {g4['total_trades']:>5,}  (+{dt:>4})  |  WR: {g3['winrate']:.2%} -> {g4['winrate']:.2%} ({dw:+.2%})  |  PnL: ${g3['pnl']:>9,.0f} -> ${g4['pnl']:>9,.0f} (${dp:>+9,.0f})  |  DD: {g3['mean_dd']:.2%} -> {g4['mean_dd']:.2%}")

# Low-ATR coins focus
print()
print(f"  {'Low-ATR Coins Focus':}")
low_atr = ["DOGEUSDT", "1000SHIBUSDT", "1000PEPEUSDT", "POLUSDT"]
for sym in low_atr:
    if sym in coins:
        v3, v4 = r3[sym], r4[sym]
        dt = v4["total_trades"] - v3["total_trades"]
        dp = v4["pnl"] - v3["pnl"]
        print(f"    {sym:<14}  T={v3['total_trades']}->{v4['total_trades']} (+{dt})  WR={v3['winrate']:.2%}->{v4['winrate']:.2%}  PnL=${v3['pnl']:,.0f}->${v4['pnl']:,.0f} (+${dp:,.0f})  DD={v3['max_dd']:.2%}->{v4['max_dd']:.2%}")

print("=" * 115)
