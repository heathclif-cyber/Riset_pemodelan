"""
pipeline/03b_regime_hmm.py — Fase 03b: HMM Regime Label Generation

Fit GaussianHMM (walk-forward, leak-free) pada H4 data untuk setiap koin.
Output: data/labeled/{coin}_regime_h1.parquet — kolom hmm_regime (string)
                                                 kolom hmm_regime_enc (int)

Urutan pipeline:
  02_clean.py → 03b_regime_hmm.py → 04_engineer.py → ...

Jalankan:
  python pipeline/03b_regime_hmm.py               # training coins
  python pipeline/03b_regime_hmm.py --all         # semua 20 koin
  python pipeline/03b_regime_hmm.py --coins SOLUSDT XRPUSDT
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from config import (
    TRAINING_COINS, ALL_COINS,
    PROC_DIR, LABEL_DIR,
    HMM_N_STATES, HMM_N_FOLDS, HMM_PURGE_H4, HMM_N_ITER,
)
from core.regime import generate_oof_regime_labels, encode_regime
from core.utils import setup_logger

logger = setup_logger("03b_regime_hmm")


# ─── H4 Extractor ────────────────────────────────────────────────────────────

def extract_h4_ohlcv(df_h1: pd.DataFrame) -> pd.DataFrame:
    """
    Extract H4 OHLCV dari clean parquet.
    Clean parquet punya kolom 4h_open/high/low/close/volume (pre-resampled dari 02_clean).
    Ambil unique H4 candle per 4h window. Fallback ke resample dari 1h_* jika perlu.
    """
    h4_map = {"4h_open": "open", "4h_high": "high", "4h_low": "low",
               "4h_close": "close", "4h_volume": "volume"}
    present = {k: v for k, v in h4_map.items() if k in df_h1.columns}

    if len(present) >= 4:
        df_h4 = df_h1[list(present.keys())].copy()
        df_h4.columns = [present[c] for c in df_h4.columns]
        # De-duplicate: ambil nilai pertama per 4h window
        df_h4 = df_h4.resample("4h").first().dropna(subset=["close"])
        return df_h4

    # Fallback: resample dari 1h_* atau bare kolom
    h1_map = {"1h_open": "open", "1h_high": "high", "1h_low": "low",
               "1h_close": "close", "1h_volume": "volume"}
    present_1h = {k: v for k, v in h1_map.items() if k in df_h1.columns}
    if not present_1h:
        present_1h = {k: k for k in ["open", "high", "low", "close", "volume"]
                      if k in df_h1.columns}

    df_base = df_h1[list(present_1h.keys())].copy()
    df_base.columns = [present_1h[c] for c in df_base.columns]
    agg = {c: ("first" if c == "open" else "max" if c == "high" else
               "min" if c == "low" else "last" if c == "close" else "sum")
           for c in df_base.columns}
    return df_base.resample("4h").agg(agg).dropna(subset=["close"])


# ─── Per-Symbol Processing ────────────────────────────────────────────────────

def process_symbol(symbol: str) -> dict:
    in_path = PROC_DIR / f"{symbol}_clean.parquet"
    if not in_path.exists():
        logger.warning(f"[{symbol}] File tidak ditemukan: {in_path}")
        return {"symbol": symbol, "status": "missing"}

    # Load H1 data
    df_h1 = pd.read_parquet(in_path)
    if not isinstance(df_h1.index, pd.DatetimeIndex):
        df_h1.index = pd.to_datetime(df_h1.index, utc=True)
    if df_h1.index.tz is None:
        df_h1.index = df_h1.index.tz_localize("UTC")
    df_h1 = df_h1.sort_index()

    logger.info(f"[{symbol}] H1 rows: {len(df_h1):,}")

    # Extract H4 dari clean parquet (4h_* kolom atau fallback resample)
    df_h4 = extract_h4_ohlcv(df_h1)
    logger.info(f"[{symbol}] H4 rows: {len(df_h4):,}")

    if len(df_h4) < 200:
        logger.warning(f"[{symbol}] H4 terlalu pendek ({len(df_h4)} bars), skip")
        return {"symbol": symbol, "status": "too_short", "h4_bars": len(df_h4)}

    # Generate OOF regime labels (walk-forward, leak-free)
    logger.info(
        f"[{symbol}] Fitting HMM — n_states={HMM_N_STATES}, "
        f"n_folds={HMM_N_FOLDS}, purge={HMM_PURGE_H4} H4 bars"
    )
    try:
        regime_h4 = generate_oof_regime_labels(
            df_h4,
            n_states=HMM_N_STATES,
            n_folds=HMM_N_FOLDS,
            purge=HMM_PURGE_H4,
            n_iter=HMM_N_ITER,
        )
    except Exception as e:
        logger.error(f"[{symbol}] HMM gagal: {e}")
        return {"symbol": symbol, "status": "hmm_error", "error": str(e)}

    # Forward-fill ke H1
    regime_h1 = regime_h4.reindex(df_h1.index, method="ffill").ffill().bfill()
    regime_h1.name = "hmm_regime"

    regime_enc = encode_regime(regime_h1, n_states=HMM_N_STATES)
    regime_enc.name = "hmm_regime_enc"

    # Distribusi
    dist = regime_h1.value_counts(normalize=True).round(3).to_dict()
    logger.info(f"[{symbol}] Regime dist (H1 ffill): {dist}")

    # Simpan ke labeled/
    out_df   = pd.DataFrame({"hmm_regime": regime_h1, "hmm_regime_enc": regime_enc})
    out_path = LABEL_DIR / f"{symbol}_regime_h1.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    table = pa.Table.from_pandas(out_df, preserve_index=True)
    pq.write_table(table, str(out_path), compression="snappy")
    logger.info(f"[{symbol}] Saved {len(out_df)} rows -> {out_path.name}")

    return {
        "symbol":      symbol,
        "status":      "success",
        "h4_bars":     len(df_h4),
        "h1_bars":     len(df_h1),
        "regime_dist": dist,
    }


# ─── Main ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="HMM Regime Label Generation")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--all",   action="store_true", help="Semua 20 koin")
    g.add_argument("--coins", nargs="+", metavar="SYMBOL")
    return p.parse_args()


def main():
    args = parse_args()
    if args.all:
        coins = ALL_COINS
    elif args.coins:
        coins = [c.upper() for c in args.coins]
    else:
        coins = TRAINING_COINS

    logger.info(f"Processing {len(coins)} coin(s): {coins}")

    results = {}
    for sym in coins:
        results[sym] = process_symbol(sym)

    # Summary
    ok   = [s for s, r in results.items() if r.get("status") == "success"]
    fail = [s for s, r in results.items() if r.get("status") != "success"]

    print(f"\n{'='*50}")
    print(f"  HMM Regime Labels — SELESAI")
    print(f"  Sukses: {len(ok)} | Gagal: {len(fail)}")
    if fail:
        print(f"  Gagal: {fail}")
    print(f"\n  Output: data/labeled/{{coin}}_regime_h1.parquet")
    print(f"  Kolom : hmm_regime (str), hmm_regime_enc (int)")
    print(f"  Langkah selanjutnya: python pipeline/04_engineer.py")
    print(f"{'='*50}\n")

    # Simpan summary ke reports/
    from config import REPORT_DIR
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORT_DIR / "regime_hmm_report.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Report -> {out}")


if __name__ == "__main__":
    main()
