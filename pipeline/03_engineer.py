"""
pipeline/03_engineer.py — Fase 3: Feature Engineering & Labeling

Jalankan:
  python pipeline/03_engineer.py                    # engineer training coins
  python pipeline/03_engineer.py --new              # engineer new coins
  python pipeline/03_engineer.py --all              # engineer semua koin
  python pipeline/03_engineer.py --coins SOLUSDT    # koin spesifik
"""

import argparse
import json
import sys
import traceback
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from config import (
    TRAINING_COINS, NEW_COINS, ALL_COINS, SYMBOL_MAP,
    PROC_DIR, LABEL_DIR, REPORT_DIR,
    SWING_LABEL_MAX_HOLD, SWING_LABEL_MIN_RR,
    SWING_LABEL_MIN_TP, SWING_LABEL_MAX_SL,
    FEATURE_COLS_V3,
    VP_WINDOW, VP_BINS, SWING_LOOKBACK, FVG_MIN_GAP_ATR,
    SWING_ROLLING_BARS
)
from core.utils import setup_logger, ensure_utc_index
from core.features import engineer_features

logger = setup_logger("03_engineer")

# Filter thresholds (disesuaikan jika belum ada di config)
LONG_MAX_PRICE_IN_RANGE = 0.8
SHORT_MIN_PRICE_IN_RANGE = 0.2


def validate_features(df: pd.DataFrame) -> list[str]:
    """Memastikan semua fitur V3 ada di DataFrame."""
    missing = [col for col in FEATURE_COLS_V3 if col not in df.columns]
    return missing


def _save(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(df, preserve_index=True)
    pq.write_table(table, str(path), compression="snappy")


def engineer_symbol(symbol: str) -> dict[str, Any]:
    report = {"symbol": symbol}
    in_path = PROC_DIR / f"{symbol}_clean.parquet"
    
    if not in_path.exists():
        logger.error(f"[{symbol}] File clean tidak ditemukan: {in_path}")
        report["status"] = "missing_input"
        return report

    logger.info(f"[{symbol}] Starting feature engineering v3...")
    df = pd.read_parquet(in_path)
    df = ensure_utc_index(df)

    symbol_id = SYMBOL_MAP.get(symbol, -1)

    try:
        # ★ Panggilan fungsi dengan parameter V3 (Base H1)
        feat_df = engineer_features(
            df                       = df,
            symbol                   = symbol,
            symbol_id                = symbol_id,
            max_hold                 = SWING_LABEL_MAX_HOLD,   # ★ BARU
            min_rr                   = SWING_LABEL_MIN_RR,     # ★ BARU
            min_tp_atr               = SWING_LABEL_MIN_TP,     # ★ BARU
            max_sl_atr               = SWING_LABEL_MAX_SL,     # ★ BARU
            vp_window                = VP_WINDOW,
            vp_bins                  = VP_BINS,
            swing_lookback           = SWING_LOOKBACK,
            fvg_min_gap              = FVG_MIN_GAP_ATR,
            long_max_price_in_range  = LONG_MAX_PRICE_IN_RANGE,
            short_min_price_in_range = SHORT_MIN_PRICE_IN_RANGE,
            add_label                = True,
        )
        
        # Validasi menggunakan FEATURE_COLS_V3
        missing_cols = validate_features(feat_df)
        if missing_cols:
            logger.warning(f"[{symbol}] Ada {len(missing_cols)} fitur V3 yang hilang: {missing_cols}")
            report["missing_features"] = missing_cols

        # Amankan kolom-kolom V3 yang valid + kolom label
        cols_to_keep = [c for c in FEATURE_COLS_V3 if c in feat_df.columns] + ["label"]
        feat_df = feat_df[cols_to_keep]

        # Buang NaN baris awal akibat rolling windows
        initial_len = len(feat_df)
        feat_df = feat_df.dropna(subset=[c for c in FEATURE_COLS_V3 if c in feat_df.columns])
        dropped = initial_len - len(feat_df)

        # Output ke _features_v3.parquet
        out_path = LABEL_DIR / f"{symbol}_features_v3.parquet"
        _save(feat_df, out_path)

        report.update({
            "status": "success",
            "rows_output": len(feat_df),
            "rows_dropped_nan": dropped,
            "columns": len(feat_df.columns),
            "output_file": str(out_path)
        })
        logger.info(f"[{symbol}] Saved {len(feat_df)} rows to {out_path.name}")

    except Exception as e:
        logger.error(f"[{symbol}] Error during engineering: {e}")
        logger.error(traceback.format_exc())
        report["status"] = "error"
        report["error"] = str(e)

    return report


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

    LABEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    full_report = {"symbols": {}}
    for symbol in coins:
        full_report["symbols"][symbol] = engineer_symbol(symbol)

    report_path = REPORT_DIR / "engineering_v3_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2, default=str)
    logger.info(f"Report saved → {report_path}")


if __name__ == "__main__":
    main()