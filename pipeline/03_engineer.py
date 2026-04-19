"""
pipeline/03_engineer.py — Fase 3: Feature Engineering + Triple Barrier Labeling

Jalankan:
  python pipeline/03_engineer.py                 # training coins
  python pipeline/03_engineer.py --new           # new coins
  python pipeline/03_engineer.py --all           # semua 20 koin
  python pipeline/03_engineer.py --coins SOLUSDT
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from config import (
    TRAINING_COINS, NEW_COINS, ALL_COINS,
    SYMBOL_MAP, PROC_DIR, LABEL_DIR,
    TP_ATR_MULT, SL_ATR_MULT, MAX_HOLDING_BARS,
    VP_WINDOW, VP_BINS, SWING_LOOKBACK,
    FVG_MIN_GAP_ATR, FEATURE_COLS,
)
from core.utils import setup_logger, load_df, save_df

logger = setup_logger("03_engineer")


def validate_features(df, symbol: str) -> bool:
    """Pastikan 58 kolom output sama persis dengan FEATURE_COLS + label."""
    expected = set(FEATURE_COLS) | {"label"}
    actual   = set(df.columns)
    missing  = expected - actual
    extra    = actual - expected - {"label"}

    if missing:
        logger.warning(f"[{symbol}] Kolom MISSING: {missing}")
    if extra:
        logger.warning(f"[{symbol}] Kolom EXTRA (akan di-drop): {extra}")
    if "OB_price" in actual:
        logger.error(f"[{symbol}] OB_price TIDAK BOLEH ada di output!")
        return False
    return len(missing) == 0


def engineer_symbol(symbol: str) -> bool:
    """Feature engineering satu koin. Return True jika berhasil."""
    from core.features import engineer_features

    logger.info(f"[{symbol}] Starting feature engineering...")

    clean_path = PROC_DIR / f"{symbol}_clean.parquet"
    df = load_df(clean_path, logger)
    if df is None:
        logger.error(f"[{symbol}] Clean parquet tidak ditemukan: {clean_path}")
        return False

    symbol_id = SYMBOL_MAP.get(symbol, -1)
    if symbol_id == -1:
        logger.warning(f"[{symbol}] Tidak ada di SYMBOL_MAP — pakai -1")

    try:
        feat_df = engineer_features(
            df           = df,
            symbol       = symbol,
            symbol_id    = symbol_id,
            tp_mult      = TP_ATR_MULT,
            sl_mult      = SL_ATR_MULT,
            max_hold     = MAX_HOLDING_BARS,
            vp_window    = VP_WINDOW,
            vp_bins      = VP_BINS,
            swing_lookback = SWING_LOOKBACK,
            fvg_min_gap  = FVG_MIN_GAP_ATR,
            add_label    = True,
        )
    except Exception as e:
        logger.exception(f"[{symbol}] Feature engineering error: {e}")
        return False

    # Drop OB_price jika masih ada
    if "OB_price" in feat_df.columns:
        feat_df = feat_df.drop(columns=["OB_price"])
        logger.info(f"[{symbol}] OB_price di-drop.")

    # Reorder kolom sesuai FEATURE_COLS + label
    ordered_cols = [c for c in FEATURE_COLS if c in feat_df.columns]
    if "label" in feat_df.columns:
        ordered_cols += ["label"]
    feat_df = feat_df[ordered_cols]

    # Validasi
    validate_features(feat_df, symbol)

    # Print ringkasan
    nan_pct = feat_df.isnull().mean().mean()
    label_dist = feat_df["label"].value_counts().to_dict() if "label" in feat_df.columns else {}
    logger.info(
        f"[{symbol}] Rows={len(feat_df):,} | Cols={len(feat_df.columns)} | "
        f"NaN={nan_pct:.1%} | Labels={label_dist}"
    )

    out_path = LABEL_DIR / f"{symbol}_features.parquet"
    LABEL_DIR.mkdir(parents=True, exist_ok=True)
    if save_df(feat_df, out_path, logger):
        logger.info(f"[{symbol}] Saved → {out_path}")
        return True
    return False


def parse_args():
    parser = argparse.ArgumentParser(description="Feature Engineering Pipeline")
    group  = parser.add_mutually_exclusive_group()
    group.add_argument("--training", action="store_true")
    group.add_argument("--new",  action="store_true")
    group.add_argument("--all",  action="store_true")
    group.add_argument("--coins", nargs="+", metavar="SYMBOL")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.new:
        coins = NEW_COINS
    elif args.all:
        coins = ALL_COINS
    elif args.coins:
        coins = [c.upper() for c in args.coins]
    else:
        coins = TRAINING_COINS

    logger.info(f"Feature engineering untuk: {coins}")
    success, failed = [], []

    for symbol in coins:
        try:
            ok = engineer_symbol(symbol)
            (success if ok else failed).append(symbol)
        except Exception as e:
            logger.exception(f"[{symbol}] Error: {e}")
            failed.append(symbol)

    sep = "=" * 55
    print(f"\n{sep}")
    print(f"  FEATURE ENGINEERING SELESAI")
    print(f"{sep}")
    print(f"  Berhasil : {len(success)} — {success}")
    print(f"  Gagal    : {len(failed)}  — {failed}")
    print(f"{sep}\n")


if __name__ == "__main__":
    main()
