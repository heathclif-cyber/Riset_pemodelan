"""
pipeline/03g_rr_sweep.py — Optimal RR Search untuk Triple Barrier (Simon Methodology)

Sweep berbagai kombinasi TP/SL, pilih yang IC-nya paling stabil lintas regime.
Metrik utama: IC_IR = mean(|IC|) / std(|IC|) lintas 6 temporal windows.

IC diukur antara: ordinal(label) vs forward_return_Nbar
  Label LONG=+1, FLAT=0, SHORT=-1
  Forward return: close[t+N] / close[t] - 1 (untuk beberapa N)

Usage:
    python pipeline/03g_rr_sweep.py
    python pipeline/03g_rr_sweep.py --coins SOLUSDT ETHUSDT BTCUSDT
    python pipeline/03g_rr_sweep.py --tp-range 1.5 2.0 2.5 3.0 --sl-range 1.0 1.5 2.0
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from config import LABEL_DIR, TRAIN_CUTOFF_DATE, MAX_HOLDING_BARS
from core.features import triple_barrier_labeling
from core.utils import setup_logger

logger = setup_logger("03g_rr_sweep")

LABEL_ORDINAL = {"LONG": 1, "FLAT": 0, "SHORT": -1}
AUTOCORR_FACTOR = 24

WINDOWS = [
    ("2020", "2020-01-01", "2021-01-01"),
    ("2021", "2021-01-01", "2022-01-01"),
    ("2022", "2022-01-01", "2023-01-01"),
    ("2023", "2023-01-01", "2024-01-01"),
    ("2024", "2024-01-01", "2025-01-01"),
    ("2025", "2025-01-01", "2025-11-01"),
]

FORWARD_HORIZONS = [8, 16, 36]  # bar ke depan untuk ukur IC


def compute_ic(label_ord: np.ndarray, fwd_ret: np.ndarray) -> float:
    mask = ~(np.isnan(label_ord) | np.isnan(fwd_ret))
    if mask.sum() < 50:
        return float("nan")
    corr, _ = stats.spearmanr(label_ord[mask], fwd_ret[mask])
    return float(corr) if not np.isnan(corr) else float("nan")


def load_coin_data(coin: str) -> pd.DataFrame:
    path = LABEL_DIR / f"{coin}_features_v3.parquet"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path, columns=["close", "high", "low", "atr_14_h1", "label"])
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, utc=True)
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    df = df[df.index < TRAIN_CUTOFF_DATE].copy()
    df["coin"] = coin
    return df


def evaluate_combo(
    df_all: pd.DataFrame,
    tp: float,
    sl: float,
    max_hold: int,
) -> dict:
    """Evaluasi satu kombinasi TP/SL pada semua coins."""

    # Compute triple barrier labels per coin (tidak bisa dicampur karena ATR beda)
    dfs = []
    for coin, df_coin in df_all.groupby("coin"):
        if len(df_coin) < 200:
            continue
        new_labels = triple_barrier_labeling(
            close       = df_coin["close"],
            high        = df_coin["high"],
            low         = df_coin["low"],
            atr_base    = df_coin["atr_14_h1"],
            tp_atr_mult = tp,
            sl_atr_mult = sl,
            max_hold    = max_hold,
        )
        df_coin = df_coin.copy()
        df_coin["tb_label"] = new_labels
        df_coin["label_ord"] = df_coin["tb_label"].map(LABEL_ORDINAL)
        # forward returns
        for h in FORWARD_HORIZONS:
            df_coin[f"fwd_{h}"] = df_coin["close"].pct_change(h).shift(-h)
        dfs.append(df_coin)

    if not dfs:
        return {}

    combined = pd.concat(dfs)
    n_total = len(combined)

    # Label distribution
    dist = combined["tb_label"].value_counts(normalize=True).to_dict()
    flat_pct = dist.get("FLAT", 0)

    # IC per window per horizon
    window_ics = {h: [] for h in FORWARD_HORIZONS}

    for wlabel, wstart, wend in WINDOWS:
        t0 = pd.Timestamp(wstart, tz="UTC")
        t1 = pd.Timestamp(wend, tz="UTC")
        w = combined[(combined.index >= t0) & (combined.index < t1)]
        if len(w) < 100:
            continue
        y_ord = w["label_ord"].values.astype(float)
        for h in FORWARD_HORIZONS:
            ic = compute_ic(y_ord, w[f"fwd_{h}"].values)
            if not np.isnan(ic):
                window_ics[h].append(ic)

    # IC_IR per horizon
    results = {
        "tp": tp, "sl": sl, "rr": round(tp / sl, 2),
        "max_hold": max_hold,
        "flat_pct": round(flat_pct * 100, 1),
        "long_pct": round(dist.get("LONG", 0) * 100, 1),
        "short_pct": round(dist.get("SHORT", 0) * 100, 1),
        "n_rows": n_total,
    }

    ic_irs = []
    for h in FORWARD_HORIZONS:
        ics = window_ics[h]
        if len(ics) < 3:
            results[f"ic_mean_{h}"] = None
            results[f"ic_ir_{h}"]   = None
            continue
        ic_mean = float(np.mean(ics))
        ic_std  = float(np.std(ics, ddof=1))
        ic_ir   = abs(ic_mean) / ic_std if ic_std > 1e-6 else float("inf")
        results[f"ic_mean_{h}"] = round(ic_mean, 4)
        results[f"ic_ir_{h}"]   = round(ic_ir, 2)
        ic_irs.append(ic_ir)
        results[f"ic_windows_{h}"] = [round(x, 4) for x in ics]

    # Composite score: mean IC_IR across horizons, penalized if FLAT too low/high
    if ic_irs:
        mean_ic_ir = float(np.mean(ic_irs))
        # penalize se FLAT terlalu kecil (<20%) atau terlalu besar (>65%)
        flat_penalty = max(0, 20 - flat_pct) * 0.1 + max(0, flat_pct - 65) * 0.1
        results["mean_ic_ir"] = round(mean_ic_ir, 2)
        results["score"]      = round(mean_ic_ir - flat_penalty, 2)
    else:
        results["mean_ic_ir"] = None
        results["score"]      = None

    return results


def print_results(all_results: list):
    print(f"\n{'TP':>5} {'SL':>5} {'RR':>5} {'FLAT%':>7} {'LONG%':>7} {'SHORT%':>7} "
          f"{'IC_IR_8':>8} {'IC_IR_16':>9} {'IC_IR_36':>9} {'Score':>7}")
    print("-" * 85)

    sorted_results = sorted(all_results, key=lambda x: x.get("score") or 0, reverse=True)
    for r in sorted_results:
        ir8  = f"{r['ic_ir_8']:>8.2f}"  if r.get("ic_ir_8")  is not None else "     N/A"
        ir16 = f"{r['ic_ir_16']:>9.2f}" if r.get("ic_ir_16") is not None else "      N/A"
        ir36 = f"{r['ic_ir_36']:>9.2f}" if r.get("ic_ir_36") is not None else "      N/A"
        sc   = f"{r['score']:>7.2f}"    if r.get("score")     is not None else "    N/A"
        print(f"{r['tp']:>5.1f} {r['sl']:>5.1f} {r['rr']:>5.2f} "
              f"{r['flat_pct']:>6.1f}% {r['long_pct']:>6.1f}% {r['short_pct']:>6.1f}% "
              f"{ir8} {ir16} {ir36} {sc}")
    print("-" * 85)


def main():
    parser = argparse.ArgumentParser(description="Sweep RR untuk Triple Barrier optimal")
    parser.add_argument("--coins", nargs="+", default=["BTCUSDT", "SOLUSDT", "ETHUSDT",
                                                        "DOGEUSDT", "XRPUSDT", "ADAUSDT"])
    parser.add_argument("--tp-range", nargs="+", type=float,
                        default=[1.5, 2.0, 2.5, 3.0, 4.0])
    parser.add_argument("--sl-range", nargs="+", type=float,
                        default=[1.0, 1.5, 2.0, 2.5])
    parser.add_argument("--max-hold", type=int, default=MAX_HOLDING_BARS)
    parser.add_argument("--output", default="reports/experiments/rr_sweep_results.json")
    args = parser.parse_args()

    combos = [(tp, sl) for tp in args.tp_range for sl in args.sl_range if tp > sl]
    print(f"\n{'='*65}")
    print(f" RR SWEEP | {len(combos)} kombinasi | {len(args.coins)} koin | max_hold={args.max_hold}")
    print(f" Coins: {args.coins}")
    print(f" TP range: {args.tp_range}")
    print(f" SL range: {args.sl_range}")
    print(f"{'='*65}\n")

    # Load semua coin data sekali
    print("Loading data...")
    dfs = []
    for coin in args.coins:
        df = load_coin_data(coin)
        if not df.empty:
            dfs.append(df)
            print(f"  {coin}: {len(df):,} bars")
    if not dfs:
        print("ERROR: tidak ada data")
        return
    df_all = pd.concat(dfs).sort_index()
    print(f"  Total: {len(df_all):,} bars\n")

    # Sweep
    all_results = []
    for i, (tp, sl) in enumerate(combos, 1):
        print(f"[{i:2d}/{len(combos)}] TP={tp:.1f} SL={sl:.1f} RR={tp/sl:.2f}...", end=" ", flush=True)
        r = evaluate_combo(df_all, tp, sl, args.max_hold)
        if r:
            all_results.append(r)
            ir_str = f"IC_IR={r.get('mean_ic_ir','?')} FLAT={r.get('flat_pct','?')}% Score={r.get('score','?')}"
            print(ir_str)
        else:
            print("skip")

    print_results(all_results)

    # Best
    best = max(all_results, key=lambda x: x.get("score") or 0)
    print(f"\n=== REKOMENDASI OPTIMAL ===")
    print(f"  TP={best['tp']}×ATR | SL={best['sl']}×ATR | RR={best['rr']}")
    print(f"  Distribution: LONG={best['long_pct']}% | FLAT={best['flat_pct']}% | SHORT={best['short_pct']}%")
    print(f"  IC_IR (8/16/36): {best.get('ic_ir_8')} / {best.get('ic_ir_16')} / {best.get('ic_ir_36')}")
    print(f"  Score: {best['score']}")

    import os
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({"sweep": all_results, "best": best, "run_at": datetime.now().strftime("%Y-%m-%d %H:%M")}, f, indent=2)
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
