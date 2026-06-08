"""
pipeline/03b_guardian_ic_test.py — IC Test khusus Guardian

Dua hal yang ditest:
  1. Static features (33) vs exit_label (HOLD/EXIT) — konfirmasi
  2. Dynamic features (7) vs exit_label — ini yang novel dan penting

Dynamic features (berubah tiap bar selama trade aktif):
  bars_held_norm, current_pnl_pct, current_pnl_atr,
  max_favorable_pnl_pct, drawdown_from_peak_pct,
  direction, entry_price_ratio

Target: HOLD=0, EXIT=1 (gabungan PARTIAL+FULL)
IC positif: feature tinggi -> cenderung EXIT
IC negatif: feature tinggi -> cenderung HOLD

Usage:
    python pipeline/03b_guardian_ic_test.py
    python pipeline/03b_guardian_ic_test.py --run-id ic32_regime_v1
"""

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from config import (
    TRAINING_COINS, LABEL_DIR, MODEL_DIR,
    TRAIN_CUTOFF_DATE, GUARDIAN_DYNAMIC_FEATURES,
    MAX_HOLDING_BARS, MODAL_PER_TRADE, LEVERAGE_SIM,
    FEE_PER_SIDE, SLIPPAGE_PER_SIDE,
    SWING_LABEL_MIN_RR, SWING_LABEL_MIN_TP, SWING_LABEL_MAX_SL,
    TP_SL_FALLBACK_TP, TP_SL_FALLBACK_SL,
    LGBM_THRESHOLD_LONG, LGBM_THRESHOLD_SHORT, LABEL_MAP,
)
from core.utils import setup_logger
from core.models import load_lstm
from pipeline.backtest_utils import hierarchical_predict
from core.evaluator import simulate_trades_swing

logger = setup_logger("03b_guardian_ic_test")

AUTOCORR_FACTOR = 24
DYNAMIC_FEATS = list(GUARDIAN_DYNAMIC_FEATURES)


def ic(x: np.ndarray, y: np.ndarray) -> float:
    mask = ~(np.isnan(x) | np.isnan(y))
    if mask.sum() < 30:
        return float("nan")
    corr, _ = stats.spearmanr(x[mask], y[mask])
    return float(corr) if not np.isnan(corr) else float("nan")


def tstat(ic_val: float, n: int) -> float:
    n_eff = max(n // AUTOCORR_FACTOR, 5)
    return ic_val * np.sqrt(n_eff) / np.sqrt(max(1 - ic_val**2, 1e-10))


def load_models(run_id: str):
    def _r(rfn, fn):
        p = MODEL_DIR / "runs" / run_id / rfn
        return p if p.exists() else MODEL_DIR / fn

    lgbm  = joblib.load(_r("lgbm.pkl", "lgbm_baseline.pkl"))
    guardian = joblib.load(_r("guardian.pkl", "guardian_best.pkl"))
    g_scaler = joblib.load(_r("guardian_scaler.pkl", "guardian_scaler.pkl"))
    g_feats  = json.load(open(_r("guardian_feature_cols.json", "guardian_feature_cols.json")))
    feat_cols = json.load(open(_r("feature_cols_v2.json", "feature_cols_v2.json")))
    return lgbm, guardian, g_scaler, g_feats, feat_cols


def generate_in_trade_samples(
    coin: str, lgbm, feat_cols: list, guardian, g_scaler, g_feats: list,
) -> list[dict]:
    """Generate in-trade bar samples dengan static + dynamic features."""
    path = LABEL_DIR / f"{coin}_features_v3.parquet"
    if not path.exists():
        return []

    df = pd.read_parquet(path)
    df = df.sort_index()
    df = df[df.index < TRAIN_CUTOFF_DATE]

    # merge HMM regime jika ada
    reg_path = LABEL_DIR / f"{coin}_regime_h1.parquet"
    if reg_path.exists() and "hmm_regime_enc" in feat_cols:
        try:
            reg = pd.read_parquet(reg_path)
            if "hmm_regime_enc" in df.columns:
                df = df.drop(columns=["hmm_regime_enc"])
            df = df.join(reg[["hmm_regime_enc"]], how="left")
            df["hmm_regime_enc"] = df["hmm_regime_enc"].fillna(1).astype("int32")
        except Exception:
            df["hmm_regime_enc"] = 1

    mask = df["label"].astype(str).isin(LABEL_MAP)
    df = df[mask].copy()
    if len(df) < 100:
        return []

    n = len(df)
    X = np.zeros((n, len(feat_cols)), dtype=np.float64)
    for i, col in enumerate(feat_cols):
        if col in df.columns:
            X[:, i] = df[col].ffill().fillna(0).values

    y_pred, confidence = hierarchical_predict(
        None, lgbm, None, None, X, feat_cols, [], df,
        trend_alignment_enabled=False,
    )
    below = (y_pred != 1) & (confidence < LGBM_THRESHOLD_LONG)
    y_pred[below] = 1

    close = df["close"].values
    high  = df["high"].values if "high" in df.columns else close
    low   = df["low"].values  if "low"  in df.columns else close
    atr   = df["atr_14_h1"].values if "atr_14_h1" in df.columns else np.ones(n)
    sh    = df["h4_swing_high"].values if "h4_swing_high" in df.columns else np.full(n, np.nan)
    sl    = df["h4_swing_low"].values  if "h4_swing_low"  in df.columns else np.full(n, np.nan)

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
        guardian_enabled=False,
    )

    trades = result.get("trades", [])
    if not trades:
        return []

    # Static features per bar
    g_static_cols = [c for c in g_feats if c not in DYNAMIC_FEATS]
    X_static = np.zeros((n, len(g_static_cols)), dtype=np.float64)
    for i, col in enumerate(g_static_cols):
        if col in df.columns:
            X_static[:, i] = df[col].ffill().fillna(0).values

    samples = []
    for t in trades:
        bar_in   = t.get("bar_in", t.get("entry_bar"))
        bar_out  = t.get("bar_out", t.get("exit_bar"))
        direction = 2 if t.get("direction") == "LONG" else 0
        entry_p  = t.get("entry", close[bar_in] if bar_in < n else 1.0)
        atr_i    = atr[bar_in] if bar_in and bar_in < n else 1.0
        net_pnl  = t.get("net_pnl", t.get("pnl", 0))

        if bar_in is None or bar_out is None or bar_out <= bar_in + 1:
            continue

        # Best future PnL dari j+1 ke bar_out
        mfe = 0.0
        for k in range(bar_in + 1, min(bar_out, n)):
            if np.isnan(close[k]): continue
            p = (close[k] - entry_p) / entry_p if direction == 2 else (entry_p - close[k]) / entry_p
            mfe = max(mfe, p)

        for j in range(bar_in + 1, min(bar_out, n)):
            if np.isnan(close[j]): continue

            bars_held = j - bar_in
            cur_pnl = (close[j] - entry_p) / entry_p if direction == 2 else (entry_p - close[j]) / entry_p
            cur_atr = cur_pnl * entry_p / atr_i if atr_i > 0 else 0.0
            mfe_j = 0.0
            for k in range(bar_in + 1, j + 1):
                if not np.isnan(close[k]):
                    p = (close[k] - entry_p) / entry_p if direction == 2 else (entry_p - close[k]) / entry_p
                    mfe_j = max(mfe_j, p)
            dd = (mfe_j - cur_pnl) / mfe_j if mfe_j > 0.001 else 0.0

            # Guardian label: is EXITING at this bar better than holding?
            # Proxy: if exit here -> lock in cur_pnl; hold -> get net_pnl
            # EXIT better if cur_pnl > net_pnl (exiting early preserves more)
            exit_better = 1 if cur_pnl > net_pnl else 0

            sample = {
                # dynamic features
                "bars_held_norm":          bars_held / MAX_HOLDING_BARS,
                "current_pnl_pct":         cur_pnl,
                "current_pnl_atr":         cur_atr,
                "max_favorable_pnl_pct":   mfe_j,
                "drawdown_from_peak_pct":  dd,
                "direction":               float(direction == 2),  # 1=LONG, 0=SHORT
                "entry_price_ratio":       (close[j] - entry_p) / atr_i if atr_i > 0 else 0.0,
                # outcome
                "exit_better":             exit_better,
                "net_pnl":                 float(net_pnl),
            }
            # add static features
            for i, col in enumerate(g_static_cols):
                sample[col] = float(X_static[j, i])

            samples.append(sample)

    return samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default="ic32_regime_v1")
    parser.add_argument("--coins",  nargs="+", default=TRAINING_COINS[:10])  # 10 koin untuk kecepatan
    parser.add_argument("--all-coins", action="store_true")
    args = parser.parse_args()

    coins = TRAINING_COINS if args.all_coins else args.coins

    print(f"\n{'='*65}")
    print(f" GUARDIAN IC TEST | run_id={args.run_id}")
    print(f" Coins: {len(coins)}")
    print(f"{'='*65}\n")

    lgbm, guardian, g_scaler, g_feats, feat_cols = load_models(args.run_id)
    logger.info(f"Models loaded | Guardian feats: {len(g_feats)} | LGBM feats: {len(feat_cols)}")

    # Generate in-trade samples
    print("[1/3] Generating in-trade bar samples...")
    all_samples = []
    for coin in coins:
        samples = generate_in_trade_samples(coin, lgbm, feat_cols, guardian, g_scaler, g_feats)
        if samples:
            all_samples.extend(samples)
            logger.info(f"  {coin}: {len(samples)} in-trade bars")

    if not all_samples:
        print("ERROR: No samples generated")
        return

    df = pd.DataFrame(all_samples)
    n = len(df)
    print(f"  Total in-trade bars: {n:,}")
    print(f"  Exit better: {df['exit_better'].mean()*100:.1f}% of bars")

    # IC test dynamic features
    print("\n[2/3] IC test dynamic features vs exit_better...")
    y = df["exit_better"].values.astype(float)
    n_eff = n // AUTOCORR_FACTOR

    dynamic_results = []
    for feat in DYNAMIC_FEATS:
        if feat not in df.columns:
            continue
        x = df[feat].values.astype(float)
        ic_val = ic(x, y)
        ts = tstat(ic_val, n) if not np.isnan(ic_val) else float("nan")
        verdict = "KEEP" if abs(ic_val) >= 0.02 and abs(ts) >= 2.0 else "WEAK"
        dynamic_results.append({
            "feature": feat, "type": "dynamic",
            "ic": round(ic_val, 4), "tstat": round(ts, 2),
            "verdict": verdict,
        })

    # IC test static features (sample — full set already tested)
    print("[3/3] IC test static features vs exit_better (top subset)...")
    g_static_cols = [c for c in g_feats if c not in DYNAMIC_FEATS]
    static_results = []
    for feat in g_static_cols:
        if feat not in df.columns:
            continue
        x = df[feat].values.astype(float)
        ic_val = ic(x, y)
        ts = tstat(ic_val, n) if not np.isnan(ic_val) else float("nan")
        verdict = "KEEP" if abs(ic_val) >= 0.02 and abs(ts) >= 2.0 else "WEAK"
        static_results.append({
            "feature": feat, "type": "static",
            "ic": round(ic_val, 4), "tstat": round(ts, 2),
            "verdict": verdict,
        })

    # Print dynamic results (paling penting)
    print(f"\n{'='*65}")
    print(f" DYNAMIC FEATURES — IC vs Exit Decision")
    print(f" n={n:,} | n_eff={n_eff:,} | exit_better_rate={df['exit_better'].mean()*100:.1f}%")
    print(f" IC positif = feature tinggi -> cenderung EXIT sekarang")
    print(f"{'='*65}")
    print(f"  {'Feature':<30} {'IC':>8} {'t-stat':>8}  Verdict  Intuisi")
    print("-" * 70)
    for r in sorted(dynamic_results, key=lambda x: abs(x["ic"]), reverse=True):
        flag = "+" if r["verdict"] == "KEEP" else "-"
        # intuisi
        if "pnl" in r["feature"] or "favorable" in r["feature"]:
            intuisi = "profit -> exit"
        elif "drawdown" in r["feature"]:
            intuisi = "drawdown -> exit"
        elif "bars_held" in r["feature"]:
            intuisi = "time -> exit"
        elif "direction" in r["feature"]:
            intuisi = "trade direction"
        else:
            intuisi = ""
        print(f"  {flag} {r['feature']:<28} {r['ic']:>+8.4f} {r['tstat']:>8.2f}  {r['verdict']:<8} {intuisi}")

    # Static features IC vs exit (top 10 by |IC|)
    print(f"\n{'='*65}")
    print(f" STATIC FEATURES — Top 10 by |IC| vs Exit Decision")
    print(f"{'='*65}")
    static_sorted = sorted(static_results, key=lambda x: abs(x["ic"]), reverse=True)[:10]
    print(f"  {'Feature':<30} {'IC':>8} {'t-stat':>8}  Verdict")
    print("-" * 55)
    for r in static_sorted:
        flag = "+" if r["verdict"] == "KEEP" else "-"
        print(f"  {flag} {r['feature']:<28} {r['ic']:>+8.4f} {r['tstat']:>8.2f}  {r['verdict']}")

    # Summary
    dyn_keep  = sum(1 for r in dynamic_results if r["verdict"]=="KEEP")
    stat_keep = sum(1 for r in static_results  if r["verdict"]=="KEEP")
    print(f"\nSummary:")
    print(f"  Dynamic: {dyn_keep}/{len(dynamic_results)} KEEP")
    print(f"  Static : {stat_keep}/{len(static_results)} KEEP")

    # Save results
    import os, datetime
    os.makedirs("reports/experiments", exist_ok=True)
    ts_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = f"reports/experiments/guardian_ic_test_{ts_str}.json"
    import json as json_lib
    with open(out_path, "w") as f:
        json_lib.dump({
            "run_id": args.run_id,
            "n_bars": n, "n_eff": n_eff,
            "exit_better_rate": float(df["exit_better"].mean()),
            "dynamic": dynamic_results,
            "static_top10": static_sorted,
        }, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
