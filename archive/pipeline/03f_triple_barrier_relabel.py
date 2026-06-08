"""
pipeline/03f_triple_barrier_relabel.py — Re-label dengan Triple Barrier

Mengganti kolom 'label' di parquet yang sudah ada dengan Triple Barrier labels.
Tidak perlu re-run feature engineering dari awal.

Triple Barrier:
  LONG:  TP = close + tp_atr × ATR,  SL = close - sl_atr × ATR
  SHORT: TP = close - tp_atr × ATR,  SL = close + sl_atr × ATR
  Time:  max_hold bars → FLAT

Default parameters selaras dengan sistem trading nyata:
  tp_atr_mult = 2.0  (= TP_SL_FALLBACK_TP)
  sl_atr_mult = 1.5  (= TP_SL_FALLBACK_SL)
  max_hold    = 36   (= MAX_HOLDING_BARS)

Usage:
    python pipeline/03f_triple_barrier_relabel.py
    python pipeline/03f_triple_barrier_relabel.py --holdout
    python pipeline/03f_triple_barrier_relabel.py --tp 2.0 --sl 1.5 --max-hold 36
    python pipeline/03f_triple_barrier_relabel.py --coins SOLUSDT ETHUSDT
"""

import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from config import (
    TRAINING_COINS, ALL_COINS,
    LABEL_DIR, HOLDOUT_DIR,
    TRAIN_CUTOFF_DATE,
    MAX_HOLDING_BARS, TP_SL_FALLBACK_TP, TP_SL_FALLBACK_SL,
)
from core.features import triple_barrier_labeling
from core.utils import setup_logger

logger = setup_logger("03f_triple_barrier_relabel")


def relabel_file(
    fpath: Path,
    tp_atr_mult: float,
    sl_atr_mult: float,
    max_hold: int,
    cutoff=None,
    dry_run: bool = False,
) -> dict:
    df = pd.read_parquet(fpath)
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, utc=True)
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")

    if cutoff is not None:
        df_work = df[df.index < cutoff].copy()
    else:
        df_work = df.copy()

    # Pastikan kolom yang dibutuhkan ada
    required = ["close", "high", "low", "atr_14_h1"]
    missing  = [c for c in required if c not in df_work.columns]
    if missing:
        logger.warning(f"  Missing columns: {missing} — skip")
        return {"status": "skip", "reason": f"missing {missing}"}

    # Hitung Triple Barrier labels
    new_labels = triple_barrier_labeling(
        close       = df_work["close"],
        high        = df_work["high"],
        low         = df_work["low"],
        atr_base    = df_work["atr_14_h1"],
        tp_atr_mult = tp_atr_mult,
        sl_atr_mult = sl_atr_mult,
        max_hold    = max_hold,
    )

    # Statistik perbandingan
    old_dist = df_work["label"].value_counts().to_dict() if "label" in df_work.columns else {}
    new_dist = new_labels.value_counts().to_dict()

    if dry_run:
        return {"status": "dry_run", "old": old_dist, "new": new_dist}

    # Update kolom label di parquet (seluruh file)
    df.loc[new_labels.index, "label"] = new_labels
    df.to_parquet(fpath)

    return {"status": "ok", "old": old_dist, "new": new_dist, "n": len(df_work)}


def main():
    parser = argparse.ArgumentParser(description="Re-label training/holdout data dengan Triple Barrier")
    parser.add_argument("--coins",   nargs="+", default=None)
    parser.add_argument("--all",     action="store_true")
    parser.add_argument("--holdout", action="store_true", help="Re-label holdout data juga")
    parser.add_argument("--tp",      type=float, default=TP_SL_FALLBACK_TP)
    parser.add_argument("--sl",      type=float, default=TP_SL_FALLBACK_SL)
    parser.add_argument("--max-hold", type=int,  default=MAX_HOLDING_BARS)
    parser.add_argument("--dry-run", action="store_true", help="Tampilkan perubahan tanpa menyimpan")
    args = parser.parse_args()

    coins = args.coins or (ALL_COINS if args.all else TRAINING_COINS)

    rr = args.tp / args.sl
    print(f"\n{'='*65}")
    print(f" TRIPLE BARRIER RE-LABEL")
    print(f" TP={args.tp}×ATR | SL={args.sl}×ATR | RR={rr:.2f} | MaxHold={args.max_hold}")
    print(f" Coins: {len(coins)} | Dry run: {args.dry_run}")
    print(f"{'='*65}\n")

    total_old = {"LONG": 0, "FLAT": 0, "SHORT": 0}
    total_new = {"LONG": 0, "FLAT": 0, "SHORT": 0}

    # === Training data ===
    print("--- Training data ---")
    for coin in coins:
        fpath = LABEL_DIR / f"{coin}_features_v3.parquet"
        if not fpath.exists():
            logger.warning(f"[{coin}] File tidak ada, skip")
            continue

        result = relabel_file(fpath, args.tp, args.sl, args.max_hold,
                              cutoff=TRAIN_CUTOFF_DATE, dry_run=args.dry_run)

        if result["status"] in ("ok", "dry_run"):
            old = result.get("old", {})
            new = result.get("new", {})
            for k in ["LONG", "FLAT", "SHORT"]:
                total_old[k] += old.get(k, 0)
                total_new[k] += new.get(k, 0)
            n = result.get("n", "?")
            print(f"  [{coin}] {n} bars | "
                  f"OLD: L={old.get('LONG',0)} F={old.get('FLAT',0)} S={old.get('SHORT',0)} | "
                  f"NEW: L={new.get('LONG',0)} F={new.get('FLAT',0)} S={new.get('SHORT',0)}")
        else:
            print(f"  [{coin}] {result}")

    # === Holdout data ===
    if args.holdout:
        print("\n--- Holdout data ---")
        holdout_label_dir = HOLDOUT_DIR / "labeled"
        for coin in coins:
            fpath = holdout_label_dir / f"{coin}_features_v3.parquet"
            if not fpath.exists():
                continue
            result = relabel_file(fpath, args.tp, args.sl, args.max_hold,
                                  cutoff=None, dry_run=args.dry_run)
            if result["status"] in ("ok", "dry_run"):
                old = result.get("old", {})
                new = result.get("new", {})
                n   = result.get("n", "?")
                print(f"  [{coin}] {n} bars | "
                      f"OLD: L={old.get('LONG',0)} F={old.get('FLAT',0)} S={old.get('SHORT',0)} | "
                      f"NEW: L={new.get('LONG',0)} F={new.get('FLAT',0)} S={new.get('SHORT',0)}")

    # Summary
    total_old_n = sum(total_old.values())
    total_new_n = sum(total_new.values())
    print(f"\n{'='*65}")
    print(f" SUMMARY — Training data")
    print(f"  OLD labels ({total_old_n:,}): "
          f"LONG={total_old['LONG']:,} ({total_old['LONG']/max(total_old_n,1)*100:.1f}%) | "
          f"FLAT={total_old['FLAT']:,} ({total_old['FLAT']/max(total_old_n,1)*100:.1f}%) | "
          f"SHORT={total_old['SHORT']:,} ({total_old['SHORT']/max(total_old_n,1)*100:.1f}%)")
    print(f"  NEW labels ({total_new_n:,}): "
          f"LONG={total_new['LONG']:,} ({total_new['LONG']/max(total_new_n,1)*100:.1f}%) | "
          f"FLAT={total_new['FLAT']:,} ({total_new['FLAT']/max(total_new_n,1)*100:.1f}%) | "
          f"SHORT={total_new['SHORT']:,} ({total_new['SHORT']/max(total_new_n,1)*100:.1f}%)")
    if args.dry_run:
        print("  [DRY RUN — tidak ada file yang diubah]")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()
