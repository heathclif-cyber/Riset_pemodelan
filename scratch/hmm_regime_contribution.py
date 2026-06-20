"""TB v3 per HMM regime analysis + regime-based gating optimization"""
import json, sys, warnings, numpy as np, pandas as pd
warnings.filterwarnings('ignore')
sys.path.insert(0, '.')
from config import *
import joblib
from core.evaluator import full_trading_report, simulate_trades_swing
from core.utils import ensure_utc_index

tb_model = joblib.load('models/runs/tb_lgbm_widyawardhana_v3/lgbm.pkl')
with open('models/runs/tb_lgbm_widyawardhana_v3/tb_lgbm_widyawardhana_v3_features.json') as f:
    tb_feats = json.load(f)

REGIME_NAMES = {0: "TRENDING_DOWN", 1: "RANGING_LOW_VOL", 2: "RANGING_HIGH_VOL", 3: "TRENDING_UP"}
LABEL_MAP_IC32 = {"SHORT": 0, "FLAT": 1, "LONG": 2}
THRESH = 0.48

available = [s for s in ALL_COINS if (HOLDOUT_DIR / "labeled" / f"{s}_features_v3.parquet").exists()]

# Collect all trades with regime info
all_trades = []
regime_stats = {r: {"trades": 0, "wins": 0, "pnl": 0.0, "bars": 0} for r in range(4)}
regime_predictions = {r: {"LONG": 0, "SHORT": 0, "FLAT": 0, "correct_long": 0, "correct_short": 0,
                          "total_long": 0, "total_short": 0} for r in range(4)}

for sym in available:
    df = pd.read_parquet(HOLDOUT_DIR / "labeled" / f"{sym}_features_v3.parquet")
    df = ensure_utc_index(df).sort_index()

    # Merge HMM
    reg_path = HOLDOUT_DIR / "labeled" / f"{sym}_regime_h1.parquet"
    hmm_col = np.full(len(df), 1, dtype=np.int32)  # default RANGING_LOW_VOL
    if reg_path.exists():
        reg = pd.read_parquet(reg_path)
        if "hmm_regime_enc" in reg.columns:
            hmm_col = reg["hmm_regime_enc"].reindex(df.index, fill_value=1).values.astype(np.int32)

    mask = df["label"].isin(LABEL_MAP_IC32)
    df = df[mask].copy()
    hmm_col = hmm_col[mask.values]
    y_true = df["label"].map(LABEL_MAP_IC32).values.astype(np.int32)

    atr_arr = df["atr_14_h1"].values; close_arr = df["close"].values
    high_arr = df["high"].values; low_arr = df["low"].values
    h4_sh = df["h4_swing_high"].values if "h4_swing_high" in df.columns else None
    h4_sl = df["h4_swing_low"].values if "h4_swing_low" in df.columns else None
    h4_trend_arr = df["h4_trend"].values if "h4_trend" in df.columns else None

    # TB inference
    X_tb = np.zeros((len(df), len(tb_feats)), dtype=np.float64)
    for idx, col in enumerate(tb_feats):
        if col in df.columns:
            X_tb[:, idx] = df[col].ffill().fillna(0).values.astype(np.float64)
    proba = tb_model.predict_proba(X_tb)
    conf = np.max(proba, axis=1)
    y_pred = np.argmax(proba, axis=1)
    y_pred[(y_pred != 1) & (conf < THRESH)] = 1

    # Count regime bars
    for r in range(4):
        regime_stats[r]["bars"] += (hmm_col == r).sum()

    # Simulate trades (no gating yet)
    if (y_pred != 1).sum() > 0:
        report = full_trading_report(
            y_pred=y_pred, y_actual=y_true, atr=atr_arr, close=close_arr,
            high=high_arr, low=low_arr,
            h4_swing_highs=h4_sh, h4_swing_lows=h4_sl,
            index=df.index, modal=MODAL_PER_TRADE, leverages=LEVERAGE_SIM,
            fee_per_side=FEE_PER_SIDE, slippage=SLIPPAGE_PER_SIDE,
            min_rr=SWING_LABEL_MIN_RR, min_tp_atr=SWING_LABEL_MIN_TP,
            max_sl_atr=SWING_LABEL_MAX_SL, tp_fallback_atr=TP_SL_FALLBACK_TP,
            sl_fallback_atr=TP_SL_FALLBACK_SL, max_hold=MAX_HOLDING_BARS,
            symbol=sym, confidence=conf,
            guardian_enabled=False, trailing_stop_enabled=False, h4_trend=h4_trend_arr,
        )
        lev5 = report.get("lev5x", report)
        for t in lev5.get("trades", []):
            bar_in = t["bar_in"]
            regime = int(hmm_col[bar_in]) if bar_in < len(hmm_col) else 1
            regime_stats[regime]["trades"] += 1
            if t.get("outcome") == "WIN":
                regime_stats[regime]["wins"] += 1
            regime_stats[regime]["pnl"] += t.get("net_pnl", 0)
            all_trades.append({**t, "regime": regime, "symbol": sym})

    # Per-bar prediction accuracy by regime
    for r in range(4):
        mask_r = hmm_col == r
        regime_predictions[r]["LONG"] += (y_pred[mask_r] == 2).sum()
        regime_predictions[r]["SHORT"] += (y_pred[mask_r] == 0).sum()
        regime_predictions[r]["FLAT"] += (y_pred[mask_r] == 1).sum()

# ── BASELINE: No gating ──────────────────────────────────────────────────
all_trades_df = pd.DataFrame(all_trades)
total_t = len(all_trades_df)
total_w = (all_trades_df["outcome"] == "WIN").sum() if "outcome" in all_trades_df.columns else 0
total_pnl = all_trades_df["net_pnl"].sum() if "net_pnl" in all_trades_df.columns else 0.0
base_wr = total_w / max(total_t, 1) * 100

print("=" * 80)
print(f"  TB v3 (thresh={THRESH}) — Per HMM Regime Breakdown")
print("=" * 80)
print(f"  BASELINE (no gating): {total_t} trades, WR={base_wr:.1f}%, PnL=${total_pnl:+.0f}")
print()
print(f"  {'Regime':<20} {'Bars':>8} {'Pred LONG':>9} {'Pred SHORT':>9} {'Pred FLAT':>9} {'Trades':>8} {'WR%':>8} {'PnL':>10}")
print("  " + "-" * 85)

for r in range(4):
    s = regime_stats[r]; p = regime_predictions[r]
    wr = s["wins"] / max(s["trades"], 1) * 100
    print(f"  {REGIME_NAMES[r]:<20} {s['bars']:>8,} {p['LONG']:>9,} {p['SHORT']:>9,} {p['FLAT']:>9,} {s['trades']:>8} {wr:>7.1f}% {s['pnl']:>+10.0f}")

# ── GATE TEST: block bad regimes ───────────────────────────────────────────
print(f"\n{'='*80}")
print(f"  REGIME GATING: Block trading in specific regimes")
print(f"{'='*80}")

regime_combos = [
    # (name, blocked_regimes_list)
    ("No gate (all open)", []),
    ("Block TRENDING_DOWN", [0]),
    ("Block RANGING_LOW_VOL", [1]),
    ("Block RANGING_HIGH_VOL", [2]),
    ("Block TRENDING_UP", [3]),
    ("Block both RANGING", [1, 2]),
    ("Block both TRENDING", [0, 3]),
    ("Only TRENDING (block 1,2)", [1, 2]),
    ("Only RANGING (block 0,3)", [0, 3]),
    ("Only TRENDING_UP+RANGING_LOW", [0, 2]),
    ("Only TRENDING_DOWN+RANGING_HIGH", [1, 3]),
]

# Precompute everything per coin for fast re-evaluation
coin_data = {}
for sym in available:
    df = pd.read_parquet(HOLDOUT_DIR / "labeled" / f"{sym}_features_v3.parquet")
    df = ensure_utc_index(df).sort_index()
    reg_path = HOLDOUT_DIR / "labeled" / f"{sym}_regime_h1.parquet"
    hmm_col = np.full(len(df), 1, dtype=np.int32)
    if reg_path.exists():
        reg = pd.read_parquet(reg_path)
        if "hmm_regime_enc" in reg.columns:
            hmm_col = reg["hmm_regime_enc"].reindex(df.index, fill_value=1).values.astype(np.int32)
    mask = df["label"].isin(LABEL_MAP_IC32)
    df = df[mask].copy()
    hmm_col = hmm_col[mask.values]
    X_tb = np.zeros((len(df), len(tb_feats)), dtype=np.float64)
    for idx, col in enumerate(tb_feats):
        if col in df.columns:
            X_tb[:, idx] = df[col].ffill().fillna(0).values.astype(np.float64)
    coin_data[sym] = {"df": df, "hmm": hmm_col, "X": X_tb,
                       "atr": df["atr_14_h1"].values, "close": df["close"].values,
                       "high": df["high"].values, "low": df["low"].values,
                       "h4_sh": df["h4_swing_high"].values if "h4_swing_high" in df.columns else None,
                       "h4_sl": df["h4_swing_low"].values if "h4_swing_low" in df.columns else None,
                       "h4_trend": df["h4_trend"].values if "h4_trend" in df.columns else None,
                       "y_true": df["label"].map(LABEL_MAP_IC32).values.astype(np.int32)}

print(f"\n{'Gate Strategy':<30} {'Trades':>7} {'WR%':>7} {'PnL':>10} {'PnL/t':>7} {'vs Base':>8}")
print("-" * 72)

best_pnl, best_name, best_blocked = -99999, "", []

for name, blocked in regime_combos:
    total_t, total_w, total_pnl = 0, 0, 0.0
    for sym in available:
        d = coin_data[sym]
        proba = tb_model.predict_proba(d["X"])
        conf = np.max(proba, axis=1)
        y_pred = np.argmax(proba, axis=1)
        y_pred[(y_pred != 1) & (conf < THRESH)] = 1

        # Gate: block trades in certain regimes
        for r in blocked:
            y_pred[d["hmm"] == r] = 1  # force FLAT

        if (y_pred != 1).sum() == 0:
            continue

        report = full_trading_report(
            y_pred=y_pred, y_actual=d["y_true"], atr=d["atr"],
            close=d["close"], high=d["high"], low=d["low"],
            h4_swing_highs=d["h4_sh"], h4_swing_lows=d["h4_sl"],
            index=d["df"].index, modal=MODAL_PER_TRADE, leverages=LEVERAGE_SIM,
            fee_per_side=FEE_PER_SIDE, slippage=SLIPPAGE_PER_SIDE,
            min_rr=SWING_LABEL_MIN_RR, min_tp_atr=SWING_LABEL_MIN_TP,
            max_sl_atr=SWING_LABEL_MAX_SL, tp_fallback_atr=TP_SL_FALLBACK_TP,
            sl_fallback_atr=TP_SL_FALLBACK_SL, max_hold=MAX_HOLDING_BARS,
            symbol=sym, confidence=conf,
            guardian_enabled=False, trailing_stop_enabled=False, h4_trend=d.get("h4_trend"),
        )
        lev5 = report.get("lev5x", report)
        trades = lev5.get("trades", [])
        wins = sum(1 for t in trades if t.get("outcome") == "WIN")
        pnl = sum(t.get("net_pnl", 0) for t in trades)
        total_t += len(trades); total_w += wins; total_pnl += pnl

    wr = total_w / max(total_t, 1) * 100
    pnl_t = total_pnl / max(total_t, 1)
    delta = total_pnl - base_pnl if 'base_pnl' in dir() else 0

    # Compute base first time
    if name == "No gate (all open)":
        base_t, base_wr, base_pnl = total_t, wr, total_pnl
        print(f"{name:<30} {total_t:>7} {wr:>6.1f}% {total_pnl:>+10.0f} {pnl_t:>+6.2f}  {'BASELINE':>8}")
    else:
        delta_pnl = total_pnl - base_pnl
        delta_wr = wr - base_wr
        print(f"{name:<30} {total_t:>7} {wr:>6.1f}% {total_pnl:>+10.0f} {pnl_t:>+6.2f}  {delta_pnl:>+7.0f}")

    if total_pnl > best_pnl:
        best_pnl, best_name, best_blocked = total_pnl, name, blocked

print(f"\n  BEST: {best_name} (PnL=${best_pnl:+.0f}, blocked regimes={[REGIME_NAMES[r] for r in best_blocked]})")

# Also test: regime-specific threshold (more aggressive in TRENDING, conservative in RANGING)
print(f"\n{'='*80}")
print(f"  REGIME-SPECIFIC THRESHOLD")
print(f"{'='*80}")
# From breakdown: TRENDING_UP has highest WR? RANGING_LOW_VOL has lowest?
# Try: THRESH=0.45 in TRENDING, THRESH=0.55 in RANGING
configs = [
    ("Uniform 0.48", {0: 0.48, 1: 0.48, 2: 0.48, 3: 0.48}),
    ("Aggressive TRENDING", {0: 0.45, 1: 0.50, 2: 0.50, 3: 0.45}),
    ("Conservative RANGING", {0: 0.48, 1: 0.53, 2: 0.53, 3: 0.48}),
    ("Both adjusted", {0: 0.45, 1: 0.55, 2: 0.53, 3: 0.45}),
    ("TRENDING 0.50, RANGING 0.55", {0: 0.50, 1: 0.55, 2: 0.55, 3: 0.50}),
]

print(f"\n{'Config':<30} {'Trades':>7} {'WR%':>7} {'PnL':>10} {'PnL/t':>7} {'vs Base':>8}")
print("-" * 72)
for cname, thresholds in configs:
    total_t, total_w, total_pnl = 0, 0, 0.0
    for sym in available:
        d = coin_data[sym]
        proba = tb_model.predict_proba(d["X"])
        conf = np.max(proba, axis=1)
        y_pred = np.argmax(proba, axis=1)

        # Apply regime-specific threshold
        for r, thresh in thresholds.items():
            mask_r = d["hmm"] == r
            y_pred[mask_r & (y_pred != 1) & (conf < thresh)] = 1

        if (y_pred != 1).sum() == 0:
            continue
        report = full_trading_report(
            y_pred=y_pred, y_actual=d["y_true"], atr=d["atr"],
            close=d["close"], high=d["high"], low=d["low"],
            h4_swing_highs=d["h4_sh"], h4_swing_lows=d["h4_sl"],
            index=d["df"].index, modal=MODAL_PER_TRADE, leverages=LEVERAGE_SIM,
            fee_per_side=FEE_PER_SIDE, slippage=SLIPPAGE_PER_SIDE,
            min_rr=SWING_LABEL_MIN_RR, min_tp_atr=SWING_LABEL_MIN_TP,
            max_sl_atr=SWING_LABEL_MAX_SL, tp_fallback_atr=TP_SL_FALLBACK_TP,
            sl_fallback_atr=TP_SL_FALLBACK_SL, max_hold=MAX_HOLDING_BARS,
            symbol=sym, confidence=conf,
            guardian_enabled=False, trailing_stop_enabled=False, h4_trend=d.get("h4_trend"),
        )
        lev5 = report.get("lev5x", report)
        trades = lev5.get("trades", [])
        wins = sum(1 for t in trades if t.get("outcome") == "WIN")
        pnl = sum(t.get("net_pnl", 0) for t in trades)
        total_t += len(trades); total_w += wins; total_pnl += pnl

    wr = total_w / max(total_t, 1) * 100
    pnl_t = total_pnl / max(total_t, 1)
    if cname == "Uniform 0.48":
        base_pnl2 = total_pnl
        print(f"{cname:<30} {total_t:>7} {wr:>6.1f}% {total_pnl:>+10.0f} {pnl_t:>+6.2f}  {'BASELINE':>8}")
    else:
        print(f"{cname:<30} {total_t:>7} {wr:>6.1f}% {total_pnl:>+10.0f} {pnl_t:>+6.2f}  {total_pnl - base_pnl2:>+7.0f}")
