"""
pipeline/04_analyze_swing.py — Fase 04: Grid Search Parameter Swing Detection

Dijalankan SEBELUM 03_engineer.py untuk menemukan parameter labeling terbaik.
Tidak ada side-effect — hanya membaca cleaned parquet dari 02_clean.py.

Alur tuning:
  02_clean.py → 04_analyze_swing.py → [update config.py] → 03_engineer.py

Tujuan:
  Temukan kombinasi parameter terbaik untuk swing-based labeling:
    1. Label distribution (target: LONG 20-35%, SHORT 20-35%, FLAT 30-60%)
    2. Rata-rata RR actual dari setup yang menang
    3. Rata-rata bars to TP (kualitas setup)
    4. Imbalance LONG vs SHORT

Jalankan:
  python pipeline/04_analyze_swing.py
  python pipeline/04_analyze_swing.py --coins SOLUSDT ETHUSDT
  python pipeline/04_analyze_swing.py --all
  python pipeline/04_analyze_swing.py --coins SOLUSDT --top 20
"""

import argparse
import sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from config import TRAINING_COINS, PROC_DIR
from core.utils import setup_logger, ensure_utc_index
from core.features import (
    calc_atr,
    detect_h4_swing_points,
    get_nearest_swing_levels,
)

logger = setup_logger("04_analyze_swing")


# ─── Grid parameter ───────────────────────────────────────────────────────────

LOOKBACK_GRID  = [2, 3, 4, 5, 6]          # H4 swing confirmation bars
MIN_RR_GRID    = [1.0, 1.2, 1.5, 2.0]     # minimum risk:reward
MIN_TP_GRID    = [1.0, 1.2, 1.5, 2.0]     # minimum TP distance (× ATR)
MAX_SL_GRID    = [2.5, 3.0]               # maximum SL distance (× ATR)
MAX_HOLD_GRID  = [16, 24, 36, 48]         # maksimum hold (bar H1)


# ─── Labeling mini (tanpa import engineer_features agar cepat) ────────────────

def run_swing_labeling(
    close: np.ndarray,
    high:  np.ndarray,
    low:   np.ndarray,
    atr:   np.ndarray,
    h4_sh: np.ndarray,   # swing high H4, aligned ke H1
    h4_sl: np.ndarray,   # swing low  H4, aligned ke H1
    max_hold:    int,
    min_rr:      float,
    min_tp_atr:  float,
    max_sl_atr:  float,
) -> dict:
    """Jalankan swing labeling dan return statistik."""
    n = len(close)
    labels       = np.full(n, "FLAT", dtype=object)
    tp_hits      = []   # bars to TP
    sl_hits      = []   # bars to SL
    rr_actuals   = []

    for i in range(n - 1):
        c   = close[i]
        a   = atr[i]
        sh  = h4_sh[i]
        sl_ = h4_sl[i]

        if np.isnan(c) or np.isnan(a) or a == 0 or np.isnan(sh) or np.isnan(sl_):
            continue

        tp_long  = sh
        sl_long  = sl_
        tp_dist  = tp_long - c
        sl_dist  = c - sl_long

        long_valid = (
            tp_dist > 0 and sl_dist > 0
            and tp_dist >= min_tp_atr * a
            and sl_dist <= max_sl_atr * a
            and (tp_dist / sl_dist) >= min_rr
        )

        tp_short  = sl_
        sl_short  = sh
        tp_dist_s = c - tp_short
        sl_dist_s = sl_short - c

        short_valid = (
            tp_dist_s > 0 and sl_dist_s > 0
            and tp_dist_s >= min_tp_atr * a
            and sl_dist_s <= max_sl_atr * a
            and (tp_dist_s / sl_dist_s) >= min_rr
        )

        if not long_valid and not short_valid:
            continue

        end = min(i + max_hold, n)
        out_long, out_short = "FLAT", "FLAT"
        bars_long, bars_short = max_hold, max_hold

        for j in range(i + 1, end):
            if np.isnan(high[j]) or np.isnan(low[j]):
                continue
            if long_valid and out_long == "FLAT":
                if high[j] >= tp_long:
                    out_long  = "LONG"
                    bars_long = j - i
                elif low[j] <= sl_long:
                    out_long  = "MISS"
                    bars_long = j - i
            if short_valid and out_short == "FLAT":
                if low[j] <= tp_short:
                    out_short  = "SHORT"
                    bars_short = j - i
                elif high[j] >= sl_short:
                    out_short  = "MISS"
                    bars_short = j - i
            if (not long_valid or out_long != "FLAT") and \
               (not short_valid or out_short != "FLAT"):
                break

        # Assign label
        if long_valid and short_valid:
            if out_long == "LONG" and out_short != "SHORT":
                labels[i] = "LONG"
            elif out_short == "SHORT" and out_long != "LONG":
                labels[i] = "SHORT"
            elif out_long == "LONG" and out_short == "SHORT":
                rr_l = tp_dist  / sl_dist
                rr_s = tp_dist_s / sl_dist_s
                labels[i] = "LONG" if rr_l >= rr_s else "SHORT"
        elif long_valid:
            labels[i] = "LONG" if out_long == "LONG" else "FLAT"
        elif short_valid:
            labels[i] = "SHORT" if out_short == "SHORT" else "FLAT"

        # Kumpulkan statistik
        if labels[i] == "LONG":
            tp_hits.append(bars_long)
            rr_actuals.append(tp_dist / sl_dist if sl_dist > 0 else 0)
        elif labels[i] == "SHORT":
            tp_hits.append(bars_short)
            rr_actuals.append(tp_dist_s / sl_dist_s if sl_dist_s > 0 else 0)

    total  = n
    n_long  = int((labels == "LONG").sum())
    n_short = int((labels == "SHORT").sum())
    n_flat  = int((labels == "FLAT").sum())

    return {
        "n_long":       n_long,
        "n_short":      n_short,
        "n_flat":       n_flat,
        "long_pct":     round(n_long  / total, 4),
        "short_pct":    round(n_short / total, 4),
        "flat_pct":     round(n_flat  / total, 4),
        "imbalance":    round(abs(n_long - n_short) / (n_long + n_short + 1e-9), 4),
        "avg_bars_tp":  round(float(np.mean(tp_hits)),    2) if tp_hits else 0,
        "avg_rr":       round(float(np.mean(rr_actuals)), 3) if rr_actuals else 0,
        "n_signals":    n_long + n_short,
    }


def score_params(stats: dict) -> float:
    """
    Skor kualitas parameter (lebih tinggi = lebih baik).
    Kriteria:
      - Long% dan Short% mendekati 25-30%
      - Imbalance rendah (LONG/SHORT seimbang)
      - Avg RR actual tinggi
      - Bukan terlalu sedikit sinyal
    """
    long_pct  = stats["long_pct"]
    short_pct = stats["short_pct"]
    flat_pct  = stats["flat_pct"]
    imbalance = stats["imbalance"]
    avg_rr    = stats["avg_rr"]
    n_signals = stats["n_signals"]

    # Penalti jika terlalu sedikit atau terlalu banyak sinyal
    if n_signals < 50:
        return -999.0

    # Target: LONG+SHORT ~40-60% dari total bars
    signal_rate = long_pct + short_pct
    signal_score = -abs(signal_rate - 0.45) * 10   # target 45%

    # Imbalance penalty
    imbalance_score = -imbalance * 5

    # RR reward
    rr_score = avg_rr * 2

    # Flat penalty jika terlalu tinggi
    flat_penalty = -max(0, flat_pct - 0.65) * 10

    return signal_score + imbalance_score + rr_score + flat_penalty


# ─── Main ─────────────────────────────────────────────────────────────────────

def analyze_symbol(symbol: str, verbose: bool = False) -> list[dict]:
    path = PROC_DIR / f"{symbol}_clean.parquet"
    if not path.exists():
        logger.warning(f"[{symbol}] Clean parquet tidak ditemukan: {path}")
        return []

    df = pd.read_parquet(path)
    df = ensure_utc_index(df)

    # Extract H1 OHLCV
    h  = df.get("1h_high",  df.get("high",  pd.Series(np.nan, index=df.index)))
    l  = df.get("1h_low",   df.get("low",   pd.Series(np.nan, index=df.index)))
    c  = df.get("1h_close", df.get("close", pd.Series(np.nan, index=df.index)))
    atr_h1 = calc_atr(h, l, c, 14).values

    # Extract H4 OHLCV
    h4_h = df.get("4h_high",  h)
    h4_l = df.get("4h_low",   l)
    h4_c = df.get("4h_close", c)

    results = []

    total_combos = (len(LOOKBACK_GRID) * len(MIN_RR_GRID) *
                    len(MIN_TP_GRID) * len(MAX_SL_GRID) * len(MAX_HOLD_GRID))
    logger.info(f"[{symbol}] Grid search {total_combos} kombinasi...")

    for lookback, min_rr, min_tp, max_sl, max_hold in product(
        LOOKBACK_GRID, MIN_RR_GRID, MIN_TP_GRID, MAX_SL_GRID, MAX_HOLD_GRID
    ):
        # Deteksi swing H4
        h4_sh_raw, h4_sl_raw = detect_h4_swing_points(h4_h, h4_l, lookback=lookback)
        h4_sh, h4_sl = get_nearest_swing_levels(h4_sh_raw, h4_sl_raw, df.index)

        stats = run_swing_labeling(
            close    = c.values,
            high     = h.values,
            low      = l.values,
            atr      = atr_h1,
            h4_sh    = h4_sh.values,
            h4_sl    = h4_sl.values,
            max_hold = max_hold,
            min_rr   = min_rr,
            min_tp_atr  = min_tp,
            max_sl_atr  = max_sl,
        )

        stats.update({
            "symbol":   symbol,
            "lookback": lookback,
            "min_rr":   min_rr,
            "min_tp":   min_tp,
            "max_sl":   max_sl,
            "max_hold": max_hold,
            "score":    score_params(stats),
        })
        results.append(stats)

    return results


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fase 04: Grid Search Parameter Swing Detection"
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--all",   action="store_true",
                       help="Semua koin (training + new)")
    group.add_argument("--coins", nargs="+", metavar="SYMBOL",
                       help="Koin spesifik (default: 2 training coins pertama)")
    parser.add_argument("--top",  type=int, default=15,
                        help="Tampilkan N kombinasi terbaik (default: 15)")
    return parser.parse_args()


def main():
    args = parse_args()

    from config import ALL_COINS
    if args.all:
        coins = ALL_COINS
    elif args.coins:
        coins = [c.upper() for c in args.coins]
    else:
        coins = TRAINING_COINS[:2]

    all_results = []
    for symbol in coins:
        res = analyze_symbol(symbol.upper(), verbose=args.verbose)
        all_results.extend(res)

    if not all_results:
        logger.error("Tidak ada hasil — pastikan clean parquet tersedia.")
        return

    df = pd.DataFrame(all_results)

    # Agregasi jika multi-coin: rata-rata score per kombinasi parameter
    param_cols = ["lookback", "min_rr", "min_tp", "max_sl", "max_hold"]
    agg = df.groupby(param_cols).agg(
        avg_score    = ("score",     "mean"),
        avg_long_pct = ("long_pct",  "mean"),
        avg_short_pct= ("short_pct", "mean"),
        avg_flat_pct = ("flat_pct",  "mean"),
        avg_imbalance= ("imbalance", "mean"),
        avg_rr       = ("avg_rr",    "mean"),
        avg_bars_tp  = ("avg_bars_tp","mean"),
        avg_signals  = ("n_signals", "mean"),
    ).reset_index().sort_values("avg_score", ascending=False)

    sep = "=" * 90
    print(f"\n{sep}")
    print(f"  TOP-{args.top} KOMBINASI PARAMETER SWING")
    print(f"  Koin: {coins}")
    print(f"{sep}")
    print(f"  {'LB':>3}  {'MinRR':>6}  {'MinTP':>6}  {'MaxSL':>6}  {'Hold':>5}  "
          f"{'LONG%':>6}  {'SHORT%':>7}  {'FLAT%':>6}  {'Imbalnc':>8}  "
          f"{'AvgRR':>6}  {'BarsTP':>7}  {'Score':>7}")
    print(f"  {'-'*85}")

    for _, row in agg.head(args.top).iterrows():
        # Tandai baris yang memenuhi target distribusi
        ok_long  = 0.20 <= row["avg_long_pct"]  <= 0.35
        ok_short = 0.20 <= row["avg_short_pct"] <= 0.35
        ok_flat  = 0.30 <= row["avg_flat_pct"]  <= 0.65
        marker   = "✓" if (ok_long and ok_short and ok_flat) else " "

        print(f"  {marker}{int(row['lookback']):>3}  {row['min_rr']:>6.1f}  "
              f"{row['min_tp']:>6.1f}  {row['max_sl']:>6.1f}  {int(row['max_hold']):>5}  "
              f"{row['avg_long_pct']:>6.1%}  {row['avg_short_pct']:>7.1%}  "
              f"{row['avg_flat_pct']:>6.1%}  {row['avg_imbalance']:>8.3f}  "
              f"{row['avg_rr']:>6.3f}  {row['avg_bars_tp']:>7.1f}  "
              f"{row['avg_score']:>7.3f}")

    print(f"\n  Legenda: ✓ = distribusi masuk target (LONG 20-35%, SHORT 20-35%, FLAT 30-65%)")

    # Rekomendasi terbaik yang juga memenuhi distribusi target
    valid = agg[
        (agg["avg_long_pct"]  >= 0.20) & (agg["avg_long_pct"]  <= 0.35) &
        (agg["avg_short_pct"] >= 0.20) & (agg["avg_short_pct"] <= 0.35) &
        (agg["avg_flat_pct"]  >= 0.30) & (agg["avg_flat_pct"]  <= 0.65)
    ]

    print(f"\n  {'─'*50}")
    if not valid.empty:
        best = valid.iloc[0]
        print(f"  REKOMENDASI:")
        print(f"    SWING_H4_LOOKBACK        = {int(best['lookback'])}")
        print(f"    SWING_LABEL_MIN_RR       = {best['min_rr']}")
        print(f"    SWING_LABEL_MIN_TP       = {best['min_tp']}")
        print(f"    SWING_LABEL_MAX_SL       = {best['max_sl']}")
        print(f"    SWING_LABEL_MAX_HOLD     = {int(best['max_hold'])}")
        print(f"    → LONG={best['avg_long_pct']:.1%}  SHORT={best['avg_short_pct']:.1%}  "
              f"FLAT={best['avg_flat_pct']:.1%}  AvgRR={best['avg_rr']:.3f}")
    else:
        print("  Tidak ada kombinasi yang memenuhi semua target distribusi.")
        print("  Coba perluas grid atau turunkan SWING_LABEL_MIN_RR.")
    print(f"{sep}\n")


if __name__ == "__main__":
    main()
