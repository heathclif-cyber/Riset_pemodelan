"""
pipeline/05a_generate_momentum_labels.py — Generate Momentum Labels untuk LSTM H4

Tujuan:
  Menggantikan swing labels (LGBM-oriented) dengan momentum labels yang
  path-aware dan lebih sesuai untuk LSTM sebagai momentum detector.

Metode (Hybrid B + Magnitude Filter):
  Untuk setiap bar H1 ke-i:
    1. Hitung arah N bar H1 ke depan (majority direction)
    2. Hitung total return dari close[i] ke close[i+N]
    3. Label:
       - LONG  jika up_bars >= ceil(N*0.625) DAN total_ret > min_move
       - SHORT jika up_bars <= floor(N*0.375) DAN total_ret < -min_move
       - FLAT  selain itu (noise, konsolidasi, tidak decisive)

Referensi:
  - Lopez de Prado (2018) "Advances in Financial Machine Learning" — path-aware labeling
  - Fischer & Krauss (2018) LSTM for financial prediction — forward return labels

Output:
  LABEL_DIR/{COIN}_momentum_labels.parquet
  Kolom: momentum_label (LONG/FLAT/SHORT), plus close + atr untuk referensi

Jalankan:
  python pipeline/05a_generate_momentum_labels.py
  python pipeline/05a_generate_momentum_labels.py --n 8 --min-move 0.4
  python pipeline/05a_generate_momentum_labels.py --n 6 --min-move 0.3
"""

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from config import (
    TRAINING_COINS, ALL_COINS,
    LABEL_DIR, TRAIN_CUTOFF_DATE,
)
from core.utils import setup_logger

logger = setup_logger("05a_momentum_labels")


# ─── Core labeling function ───────────────────────────────────────────────────

def generate_momentum_labels(
    close:    pd.Series,
    high:     pd.Series,
    low:      pd.Series,
    atr:      pd.Series,
    n:        int   = 8,
    min_move: float = 0.4,
) -> pd.Series:
    """
    Hybrid momentum label: majority direction + minimum magnitude filter.

    Args:
        close:    H1 close price series
        high:     H1 high price series
        low:      H1 low price series
        atr:      H1 ATR-14 series
        n:        jumlah bar H1 ke depan (horizon)
        min_move: minimum gerak sebagai kelipatan ATR agar tidak dihitung noise

    Returns:
        Series dengan label LONG/FLAT/SHORT per bar H1
    """
    length    = len(close)
    labels    = np.full(length, "FLAT", dtype=object)
    close_arr = close.values
    atr_arr   = atr.values

    # Threshold majority:
    # LONG  jika >= 62.5% bar naik  (misal N=8: >= 5 dari 8)
    # SHORT jika <= 37.5% bar naik  (misal N=8: <= 3 dari 8)
    long_bar_thr  = math.ceil(n * 0.625)   # N=8 → 5
    short_bar_thr = math.floor(n * 0.375)  # N=8 → 3

    for i in range(length - n):
        atr_i = atr_arr[i]
        if np.isnan(atr_i) or atr_i <= 0:
            continue

        c0 = close_arr[i]
        if np.isnan(c0):
            continue

        # Hitung arah setiap bar ke depan
        bar_rets  = close_arr[i+1 : i+n+1] - close_arr[i : i+n]
        up_count  = int(np.sum(bar_rets > 0))

        # Total return dari entry ke N bar ke depan
        total_ret = close_arr[i+n] - c0
        min_move_abs = min_move * atr_i

        if up_count >= long_bar_thr and total_ret > min_move_abs:
            labels[i] = "LONG"
        elif up_count <= short_bar_thr and total_ret < -min_move_abs:
            labels[i] = "SHORT"
        # else: FLAT (noise, sideways, reversal di akhir)

    # Tail N bar tidak punya forward window → paksa FLAT
    labels[-n:] = "FLAT"

    return pd.Series(labels, index=close.index, name="momentum_label")


# ─── Per-coin processing ──────────────────────────────────────────────────────

def process_coin(sym: str, n: int, min_move: float) -> dict | None:
    path = LABEL_DIR / f"{sym}_features_v3.parquet"
    if not path.exists():
        logger.warning(f"File tidak ditemukan, skip: {path}")
        return None

    df = pd.read_parquet(path)
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, utc=True)
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")

    # Filter training only — sama dengan LGBM/LSTM training
    df = df[df.index < TRAIN_CUTOFF_DATE].copy()

    if len(df) < n + 100:
        logger.warning(f"{sym}: data terlalu sedikit ({len(df)} rows), skip")
        return None

    # Generate momentum labels
    mom_labels = generate_momentum_labels(
        close    = df["close"],
        high     = df["high"],
        low      = df["low"],
        atr      = df["atr_14_h1"],
        n        = n,
        min_move = min_move,
    )

    # Gabungkan label ke dataframe
    out = df.copy()
    out["momentum_label"] = mom_labels

    # Simpan
    out_path = LABEL_DIR / f"{sym}_momentum_labels.parquet"
    out.to_parquet(out_path)

    # Statistik distribusi
    dist  = mom_labels.value_counts()
    total = len(mom_labels)
    stats = {
        "symbol":  sym,
        "total":   total,
        "LONG":    int(dist.get("LONG",  0)),
        "FLAT":    int(dist.get("FLAT",  0)),
        "SHORT":   int(dist.get("SHORT", 0)),
        "pct_LONG":  round(dist.get("LONG",  0) / total * 100, 1),
        "pct_FLAT":  round(dist.get("FLAT",  0) / total * 100, 1),
        "pct_SHORT": round(dist.get("SHORT", 0) / total * 100, 1),
    }

    # Bandingkan dengan swing label lama
    if "label" in df.columns:
        old_dist  = df["label"].value_counts()
        old_total = len(df)
        stats["old_pct_FLAT"] = round(old_dist.get("FLAT", 0) / old_total * 100, 1)

    logger.info(
        f"{sym:16}: LONG={stats['pct_LONG']:5.1f}%  "
        f"FLAT={stats['pct_FLAT']:5.1f}%  "
        f"SHORT={stats['pct_SHORT']:5.1f}%  "
        f"(lama FLAT={stats.get('old_pct_FLAT', '?')}%)"
    )
    return stats


# ─── Main ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n",        type=int,   default=8,   help="Horizon bar H1 ke depan (default: 8)")
    p.add_argument("--min-move", type=float, default=0.4, help="Min move sebagai x ATR (default: 0.4)")
    p.add_argument("--all",      action="store_true",     help="Pakai semua koin (ALL_COINS)")
    return p.parse_args()


def main():
    args  = parse_args()
    coins = ALL_COINS if args.all else TRAINING_COINS
    n, mm = args.n, args.min_move

    logger.info(f"Generate momentum labels: N={n} bar, min_move={mm}x ATR")
    logger.info(f"Koin: {len(coins)} | TRAIN_CUTOFF: {TRAIN_CUTOFF_DATE.date()}")
    logger.info(f"Output: {LABEL_DIR}")
    logger.info("-" * 70)

    all_stats = []
    for sym in coins:
        stats = process_coin(sym, n=n, min_move=mm)
        if stats:
            all_stats.append(stats)

    if not all_stats:
        logger.error("Tidak ada file berhasil diproses.")
        return

    # Ringkasan agregat
    import statistics
    sep = "=" * 70
    print(f"\n{sep}")
    print(f"  MOMENTUM LABEL SUMMARY — N={n}, min_move={mm}x ATR")
    print(f"{sep}")
    print(f"  {'Coin':16} {'LONG%':>7} {'FLAT%':>7} {'SHORT%':>8} {'Old FLAT%':>10}")
    print(f"  {'-'*16} {'-'*7} {'-'*7} {'-'*8} {'-'*10}")
    for s in all_stats:
        print(
            f"  {s['symbol']:16} {s['pct_LONG']:>6.1f}% {s['pct_FLAT']:>6.1f}% "
            f"{s['pct_SHORT']:>7.1f}% {s.get('old_pct_FLAT','?'):>9}%"
        )

    pct_longs  = [s["pct_LONG"]  for s in all_stats]
    pct_flats  = [s["pct_FLAT"]  for s in all_stats]
    pct_shorts = [s["pct_SHORT"] for s in all_stats]
    print(f"  {'-'*16} {'-'*7} {'-'*7} {'-'*8} {'-'*10}")
    print(
        f"  {'MEAN':16} {statistics.mean(pct_longs):>6.1f}% "
        f"{statistics.mean(pct_flats):>6.1f}% "
        f"{statistics.mean(pct_shorts):>7.1f}%"
    )
    print(f"{sep}\n")
    print(f"  File disimpan di: {LABEL_DIR}")
    print(f"  Pattern: {{COIN}}_momentum_labels.parquet\n")


if __name__ == "__main__":
    main()
