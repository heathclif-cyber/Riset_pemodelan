"""Quick OOF threshold sweep — expanding vs rolling OOF predictions.
Joins OOF probas with original feature data for simulate_trades_swing context."""
import json, itertools, sys
import numpy as np, pandas as pd
from pathlib import Path
sys.path.insert(0, ".")
from core.evaluator import simulate_trades_swing
from core.utils import ensure_utc_index
from config import (ALL_COINS, LABEL_DIR, TRAIN_CUTOFF_DATE, LABEL_MAP,
                    MODAL_PER_TRADE, LEVERAGE_SIM, FEE_PER_SIDE, SLIPPAGE_PER_SIDE,
                    MAX_HOLDING_BARS, SWING_LABEL_MIN_RR, SWING_LABEL_MIN_TP,
                    SWING_LABEL_MAX_SL, TP_SL_FALLBACK_TP, TP_SL_FALLBACK_SL)

COMPARE_DIR = Path("models/runs/tb_lgbm_cv_comparison")
THR_LONGS  = [0.45, 0.50, 0.55, 0.60, 0.65]
THR_SHORTS = [0.45, 0.50, 0.55, 0.60]
LM = {"SHORT": 0, "FLAT": 1, "LONG": 2}

# Load full context data with coin+timestamp for merge
print("Loading context data...")
context_frames = []
for sym in ALL_COINS:
    path = LABEL_DIR / f"{sym}_features_v3.parquet"
    if not path.exists(): continue
    df = pd.read_parquet(path)
    df = ensure_utc_index(df).sort_index()
    df = df[df.index < TRAIN_CUTOFF_DATE]
    mask = df["label"].astype(str).isin(LABEL_MAP)
    df = df[mask]
    ctx_cols = ["close","high","low","atr_14_h1"]
    for c in ["h4_swing_high","h4_swing_low"]:
        if c in df.columns: ctx_cols.append(c)
    df = df[ctx_cols].copy()
    df["coin"] = sym
    df["_ts"] = df.index  # preserve timestamp for merge
    context_frames.append(df)
    print(f"  {sym}: {len(df):,} bars")
ctx_all = pd.concat(context_frames).sort_index()
print(f"  Total context: {len(ctx_all):,} bars")

def sweep_oof(oof_parquet_path, label):
    oof = pd.read_parquet(oof_parquet_path)
    oof["_ts"] = oof.index
    has_oof_col = "has_oof" in oof.columns

    # Merge OOF + context on (timestamp, coin)
    merged = oof.merge(ctx_all, on=["_ts", "coin"], how="inner", suffixes=("","_ctx"))
    print(f"\n  [{label}] OOF: {len(oof):,} -> merged: {len(merged):,} bars")
    merged = merged.sort_values("_ts")

    has_oof_arr = merged["has_oof"].values if has_oof_col else ~np.isnan(merged["p0"].values)

    results = []
    for thr_long, thr_short in itertools.product(THR_LONGS, THR_SHORTS):
        agg_trades = agg_wins = 0
        agg_pnl = 0.0
        for sym in merged["coin"].unique():
            sm = (merged["coin"].values == sym) & has_oof_arr
            if sm.sum() < 30: continue
            sdf = merged[sm].sort_values("_ts")
            n = len(sdf)
            probas = sdf[["p0","p1","p2"]].values
            y_pred = np.full(n, LM["FLAT"], np.int32)
            y_pred[probas[:,2] >= thr_long] = LM["LONG"]
            short_m = (probas[:,0] >= thr_short) & (y_pred != LM["LONG"])
            y_pred[short_m] = LM["SHORT"]
            if (y_pred != LM["FLAT"]).sum() == 0: continue

            h4_sh = sdf["h4_swing_high"].values if "h4_swing_high" in sdf.columns else np.full(n, np.nan)
            h4_sl = sdf["h4_swing_low"].values if "h4_swing_low" in sdf.columns else np.full(n, np.nan)

            result = simulate_trades_swing(
                y_pred=y_pred, close=sdf["close"].values,
                high=sdf["high"].values, low=sdf["low"].values,
                atr=sdf["atr_14_h1"].values,
                h4_swing_highs=h4_sh, h4_swing_lows=h4_sl,
                modal=MODAL_PER_TRADE, leverage=LEVERAGE_SIM[0],
                fee_per_side=FEE_PER_SIDE, slippage=SLIPPAGE_PER_SIDE,
                max_hold=MAX_HOLDING_BARS, min_rr=SWING_LABEL_MIN_RR,
                min_tp_atr=SWING_LABEL_MIN_TP, max_sl_atr=SWING_LABEL_MAX_SL,
                tp_fallback_atr=TP_SL_FALLBACK_TP, sl_fallback_atr=TP_SL_FALLBACK_SL,
                guardian_enabled=False,
            )
            agg_trades += result.get("total_trades", 0)
            agg_wins   += result.get("wins", 0)
            agg_pnl    += result.get("total_pnl", 0.0)
        if agg_trades < 200: continue
        results.append({
            "thr_long": thr_long, "thr_short": thr_short,
            "trades": agg_trades, "wr": round(agg_wins/agg_trades, 4),
            "pnl": round(agg_pnl, 2),
            "ppt": round(agg_pnl/agg_trades, 4) if agg_trades else 0,
        })
    results.sort(key=lambda x: x["pnl"], reverse=True)
    print(f"\n{'='*65}")
    print(f"  OOF THRESHOLD SWEEP — {label}")
    print(f"  {'Thr_L':>6} {'Thr_S':>6} {'Trades':>8} {'WR%':>7} {'PnL':>10} {'PPT':>7} {'PF':>7}")
    for r in results[:8]:
        pf = round(r["pnl"]/(abs(r["pnl"])-sum(r.get("losses",[0]))), 2) if "losses" in r else 0.0
        # compute PF from wr: pf = wr/(1-wr) roughly
        gross_win = r["pnl"] + (r["trades"]-r["trades"]*r["wr"])*1  # approximate
        print(f"  {r['thr_long']:>6.2f} {r['thr_short']:>6.2f} {r['trades']:>8,} {r['wr']*100:>7.1f} {r['pnl']:>10.2f} {r['ppt']:>7.4f}")
    return results

exp_results = sweep_oof(COMPARE_DIR / "oof_predictions_expanding.parquet", "EXPANDING")
rol_results = sweep_oof(COMPARE_DIR / "oof_predictions_rolling.parquet", "ROLLING")

print(f"\n{'='*65}")
print("  HEAD-TO-HEAD — Best PnL per method")
print(f"  {'Method':<12} {'Thr_L':>6} {'Thr_S':>6} {'Trades':>8} {'WR%':>7} {'PnL':>10} {'PPT':>7}")
for name, res in [("Expanding", exp_results), ("Rolling", rol_results)]:
    best = res[0]
    print(f"  {name:<12} {best['thr_long']:>6.2f} {best['thr_short']:>6.2f} {best['trades']:>8,} {best['wr']*100:>7.1f} {best['pnl']:>10.2f} {best['ppt']:>7.4f}")

# Compare at common reference thresholds
for thr_l, thr_s in [(0.50, 0.55), (0.55, 0.60), (0.45, 0.45)]:
    er = next((r for r in exp_results if r["thr_long"]==thr_l and r["thr_short"]==thr_s), None)
    rr = next((r for r in rol_results if r["thr_long"]==thr_l and r["thr_short"]==thr_s), None)
    if er and rr:
        delta = rr["pnl"] - er["pnl"]
        print(f"\n  @ thr={thr_l}/{thr_s}:")
        print(f"    Expanding: ${er['pnl']:,.2f} ({er['trades']:,}t, WR {er['wr']*100:.1f}%)")
        print(f"    Rolling:   ${rr['pnl']:,.2f} ({rr['trades']:,}t, WR {rr['wr']*100:.1f}%)")
        print(f"    Delta:     ${delta:+,.2f}")
