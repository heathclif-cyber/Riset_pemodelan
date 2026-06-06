"""
pipeline/03h_hybrid_relabel.py — Hybrid Label: Swing + Triple Barrier (Opsi C)

score = swing_weight × swing_ordinal + tb_weight × tb_ordinal

Label = LONG  jika score >= threshold
        SHORT jika score <= -threshold
        FLAT  lainnya

Default (swing_weight=0.6, tb_weight=0.4, threshold=0.4):
  Both LONG       (score=1.0) → LONG  ✓
  Swing LONG only (score=0.6) → LONG  ✓ (swing saja cukup)
  TB LONG only    (score=0.4) → LONG  ✓ (borderline)
  Conflicting     (score=0.2) → FLAT  ✓ (tidak masuk)

Usage:
    python pipeline/03h_hybrid_relabel.py
    python pipeline/03h_hybrid_relabel.py --sw 0.6 --tb 0.4 --thr 0.4
    python pipeline/03h_hybrid_relabel.py --holdout --dry-run
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from config import (
    TRAINING_COINS, ALL_COINS,
    LABEL_DIR, HOLDOUT_DIR,
    TRAIN_CUTOFF_DATE,
    MAX_HOLDING_BARS,
    SWING_LABEL_MIN_RR, SWING_LABEL_MIN_TP, SWING_LABEL_MAX_SL,
    TP_SL_FALLBACK_TP, TP_SL_FALLBACK_SL,
)
from core.features import hybrid_labeling
from core.utils import setup_logger

logger = setup_logger("03h_hybrid_relabel")


def relabel_file(
    fpath: Path,
    swing_weight: float,
    tb_weight: float,
    threshold: float,
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

    df_work = df[df.index < cutoff].copy() if cutoff else df.copy()

    required = ["close", "high", "low", "atr_14_h1", "h4_swing_high", "h4_swing_low"]
    missing  = [c for c in required if c not in df_work.columns]
    if missing:
        return {"status": "skip", "reason": f"missing {missing}"}

    new_labels = hybrid_labeling(
        close          = df_work["close"],
        high           = df_work["high"],
        low            = df_work["low"],
        atr_base       = df_work["atr_14_h1"],
        h4_swing_highs = df_work["h4_swing_high"],
        h4_swing_lows  = df_work["h4_swing_low"],
        swing_weight   = swing_weight,
        max_hold       = max_hold,
        min_rr         = SWING_LABEL_MIN_RR,
        min_tp_atr     = SWING_LABEL_MIN_TP,
        max_sl_atr     = SWING_LABEL_MAX_SL,
        tb_weight      = tb_weight,
        tp_atr_mult    = tp_atr_mult,
        sl_atr_mult    = sl_atr_mult,
        threshold      = threshold,
    )

    old_dist = df_work["label"].value_counts().to_dict() if "label" in df_work.columns else {}
    new_dist = new_labels.value_counts().to_dict()

    if dry_run:
        return {"status": "dry_run", "old": old_dist, "new": new_dist, "n": len(df_work)}

    df.loc[new_labels.index, "label"] = new_labels
    df.to_parquet(fpath)
    return {"status": "ok", "old": old_dist, "new": new_dist, "n": len(df_work)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--coins",   nargs="+", default=None)
    parser.add_argument("--all",     action="store_true")
    parser.add_argument("--holdout", action="store_true")
    parser.add_argument("--sw",  type=float, default=0.6,  dest="swing_weight")
    parser.add_argument("--tb",  type=float, default=0.4,  dest="tb_weight")
    parser.add_argument("--thr", type=float, default=0.4,  dest="threshold")
    parser.add_argument("--tp",  type=float, default=TP_SL_FALLBACK_TP)
    parser.add_argument("--sl",  type=float, default=TP_SL_FALLBACK_SL)
    parser.add_argument("--max-hold", type=int, default=MAX_HOLDING_BARS)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    coins = args.coins or (ALL_COINS if args.all else TRAINING_COINS)

    print(f"\n{'='*65}")
    print(f" HYBRID LABEL (Swing + Triple Barrier)")
    print(f" Score = {args.swing_weight}×swing + {args.tb_weight}×TB | Threshold=±{args.threshold}")
    print(f" TP={args.tp}×ATR | SL={args.sl}×ATR | MaxHold={args.max_hold}")
    print(f" Coins: {len(coins)} | Dry run: {args.dry_run}")
    print(f"{'='*65}\n")

    total_old = {"LONG": 0, "FLAT": 0, "SHORT": 0}
    total_new = {"LONG": 0, "FLAT": 0, "SHORT": 0}

    print("--- Training data ---")
    for coin in coins:
        fpath = LABEL_DIR / f"{coin}_features_v3.parquet"
        if not fpath.exists():
            continue
        r = relabel_file(fpath, args.swing_weight, args.tb_weight, args.threshold,
                         args.tp, args.sl, args.max_hold,
                         cutoff=TRAIN_CUTOFF_DATE, dry_run=args.dry_run)
        if r["status"] in ("ok", "dry_run"):
            for k in ["LONG", "FLAT", "SHORT"]:
                total_old[k] += r.get("old", {}).get(k, 0)
                total_new[k] += r.get("new", {}).get(k, 0)
            print(f"  [{coin}] {r['n']} bars | "
                  f"OLD L={r.get('old',{}).get('LONG',0)} F={r.get('old',{}).get('FLAT',0)} S={r.get('old',{}).get('SHORT',0)} | "
                  f"NEW L={r.get('new',{}).get('LONG',0)} F={r.get('new',{}).get('FLAT',0)} S={r.get('new',{}).get('SHORT',0)}")

    if args.holdout:
        print("\n--- Holdout data ---")
        holdout_label_dir = HOLDOUT_DIR / "labeled"
        for coin in coins:
            fpath = holdout_label_dir / f"{coin}_features_v3.parquet"
            if not fpath.exists():
                continue
            r = relabel_file(fpath, args.swing_weight, args.tb_weight, args.threshold,
                             args.tp, args.sl, args.max_hold,
                             cutoff=None, dry_run=args.dry_run)
            if r["status"] in ("ok", "dry_run"):
                print(f"  [{coin}] {r['n']} bars | "
                      f"NEW L={r.get('new',{}).get('LONG',0)} F={r.get('new',{}).get('FLAT',0)} S={r.get('new',{}).get('SHORT',0)}")

    n_old = sum(total_old.values())
    n_new = sum(total_new.values())
    print(f"\n{'='*65}")
    print(f" SUMMARY — Training")
    if n_old > 0:
        print(f"  OLD: LONG={total_old['LONG']:,} ({total_old['LONG']/n_old*100:.1f}%) | "
              f"FLAT={total_old['FLAT']:,} ({total_old['FLAT']/n_old*100:.1f}%) | "
              f"SHORT={total_old['SHORT']:,} ({total_old['SHORT']/n_old*100:.1f}%)")
    if n_new > 0:
        print(f"  NEW: LONG={total_new['LONG']:,} ({total_new['LONG']/n_new*100:.1f}%) | "
              f"FLAT={total_new['FLAT']:,} ({total_new['FLAT']/n_new*100:.1f}%) | "
              f"SHORT={total_new['SHORT']:,} ({total_new['SHORT']/n_new*100:.1f}%)")
    if args.dry_run:
        print("  [DRY RUN]")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()
