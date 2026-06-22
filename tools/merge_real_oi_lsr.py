"""
tools/merge_real_oi_lsr.py — Merge real OI/LSR (dari data/metrics/) ke clean parquet.

Overwrite kolom open_interest_open_interest & long_short_ratio_long_short_ratio di
clean parquet dengan nilai REAL hasil fetch metrics (H1). Bars sebelum metrics ada
(mis. BTC < Sep 2020) dibiarkan — engineer akan pakai synthetic untuk LSR, OI raw.

Setelah merge, JALANKAN ULANG 03_engineer untuk regen features_v3.

Usage:
  python tools/merge_real_oi_lsr.py --all
  python tools/merge_real_oi_lsr.py --all --holdout-test
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from config import TRAINING_COINS, ALL_COINS, PROC_DIR, HOLDOUT_DIR
from core.utils import setup_logger, ensure_utc_index

logger = setup_logger("merge_real_oi_lsr")
METRICS_DIR = ROOT / "data" / "metrics"

OI_COL  = "open_interest_open_interest"
LSR_COL = "long_short_ratio_long_short_ratio"


def merge_coin(coin: str, proc_dir: Path, suffix: str) -> bool:
    clean_p   = proc_dir / f"{coin}_clean.parquet"
    metrics_p = METRICS_DIR / f"{coin}_oi_lsr_h1{suffix}.parquet"
    if not clean_p.exists():
        logger.warning(f"[{coin}] clean parquet tidak ada — skip")
        return False
    if not metrics_p.exists():
        logger.warning(f"[{coin}] metrics tidak ada — skip")
        return False

    clean = ensure_utc_index(pd.read_parquet(clean_p)).sort_index()
    metr  = ensure_utc_index(pd.read_parquet(metrics_p)).sort_index()

    # align metrics ke index clean; ffill jeda kecil (max 6 jam)
    real_oi  = metr["open_interest"].reindex(clean.index, method="ffill", limit=6)
    real_lsr = metr["long_short_ratio"].reindex(clean.index, method="ffill", limit=6)

    # statistik coverage
    cov = real_lsr.notna().mean()

    # kolom existing bisa bertipe string -> coerce ke float dulu
    old_oi  = pd.to_numeric(clean[OI_COL],  errors="coerce") if OI_COL  in clean.columns else pd.Series(np.nan, index=clean.index)
    old_lsr = pd.to_numeric(clean[LSR_COL], errors="coerce") if LSR_COL in clean.columns else pd.Series(np.nan, index=clean.index)

    # overwrite hanya di mana real tersedia; sisanya pakai nilai lama (numeric)
    clean[OI_COL]  = real_oi.where(real_oi.notna(),  old_oi).astype(float)
    clean[LSR_COL] = real_lsr.where(real_lsr.notna(), old_lsr).astype(float)

    clean.to_parquet(clean_p)
    lsr_sample = float(real_lsr.dropna().iloc[-1]) if real_lsr.notna().any() else float("nan")
    logger.info(f"[{coin}] merged | LSR coverage {cov*100:.0f}% | LSR terakhir {lsr_sample:.3f} | "
                f"{len(clean)} bars -> {clean_p.name}")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coins", nargs="+", default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--holdout-test", action="store_true", dest="holdout")
    args = ap.parse_args()

    coins = args.coins if args.coins else (ALL_COINS if args.all else TRAINING_COINS)
    proc_dir = (HOLDOUT_DIR / "processed") if args.holdout else PROC_DIR
    suffix = "_holdout" if args.holdout else ""

    print(f"\n{'='*64}")
    print(f"  MERGE REAL OI/LSR -> clean | {len(coins)} koin | {'HOLDOUT' if args.holdout else 'TRAINING'}")
    print(f"{'='*64}\n")

    ok, fail = [], []
    for coin in coins:
        (ok if merge_coin(coin, proc_dir, suffix) else fail).append(coin)

    print(f"\n{'='*64}")
    print(f"  SELESAI: {len(ok)} merged | {len(fail)} skip")
    if fail:
        print(f"  Skip: {fail}")
    print(f"  LANGKAH BERIKUTNYA: python pipeline/03_engineer.py --all{' --holdout-test' if args.holdout else ''}")
    print(f"{'='*64}")


if __name__ == "__main__":
    main()
