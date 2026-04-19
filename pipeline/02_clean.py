"""
pipeline/02_clean.py — Fase 2: Data Cleaning & Validation

Jalankan:
  python pipeline/02_clean.py                    # clean training coins
  python pipeline/02_clean.py --new              # clean new coins
  python pipeline/02_clean.py --all              # clean semua 20 koin
  python pipeline/02_clean.py --coins SOLUSDT    # koin spesifik
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from config import (
    TRAINING_COINS, NEW_COINS, ALL_COINS,
    RAW_DIR, PROC_DIR, REPORT_DIR,
)
from core.utils import setup_logger, ensure_utc_index

logger = setup_logger("02_clean")

INTERVALS      = ["15m", "1h", "4h", "1d"]
INTERVAL_FREQ  = {"15m": "15min", "1h": "1h", "4h": "4h", "1d": "1D"}
LEAKAGE_WORDS  = ("future", "_fwd", "next_", "lead_", "label",
                  "barrier", "exit_time", "pnl", "ret_fwd")


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _load(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        df = pd.read_parquet(path)
        if df.empty:
            return None
        return ensure_utc_index(df)
    except Exception as e:
        logger.warning(f"Gagal load {path}: {e}")
        return None


def _save(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(df, preserve_index=True)
    pq.write_table(table, str(path), compression="snappy")


def detect_gaps(df: pd.DataFrame, freq: str) -> list[dict]:
    if len(df) < 2:
        return []
    expected_ns = pd.tseries.frequencies.to_offset(freq).nanos
    diffs = df.index.to_series().diff().dropna()
    gaps  = []
    for ts, delta in diffs.items():
        if delta.total_seconds() * 1e9 > expected_ns * 1.5:
            missing = round(delta.total_seconds() * 1e9 / expected_ns) - 1
            gaps.append({
                "gap_start":    str(ts - delta),
                "gap_end":      str(ts),
                "missing_bars": int(missing),
            })
    return gaps


def fix_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    col_map = {c.lower(): c for c in df.columns}
    if not {"open","high","low","close"}.issubset(col_map):
        return df
    ocols = [col_map[k] for k in ("open","high","low","close")]
    mat   = df[ocols].values.astype(float)
    df    = df.copy()
    df[col_map["high"]] = np.nanmax(mat, axis=1)
    df[col_map["low"]]  = np.nanmin(mat, axis=1)
    return df


def ffill_macro(macro_df: pd.DataFrame, m15_index: pd.DatetimeIndex) -> pd.DataFrame:
    combined = macro_df.reindex(macro_df.index.union(m15_index)).sort_index().ffill()
    return combined.reindex(m15_index)


def audit_leakage(df: pd.DataFrame) -> list[str]:
    flagged = []
    for col in df.columns:
        cl = col.lower()
        if any(kw in cl for kw in LEAKAGE_WORDS):
            flagged.append(col)
    return flagged


# ─── Per-Symbol Pipeline ─────────────────────────────────────────────────────

def clean_symbol(symbol: str) -> dict[str, Any]:
    report: dict[str, Any] = {
        "symbol": symbol, "klines": {}, "auxiliary": {}, "macro": {}
    }
    logger.info(f"[{symbol}] Starting cleaning...")

    # ── Load klines ───────────────────────────────────────────────────────────
    klines = {}
    for tf in INTERVALS:
        path = RAW_DIR / "klines" / symbol / f"{tf}_all.parquet"
        df   = _load(path)
        klines[tf] = df
        if df is None:
            report["klines"][tf] = {"status": "missing"}
            continue
        df   = fix_ohlc(df)
        gaps = detect_gaps(df, INTERVAL_FREQ[tf])
        klines[tf] = df
        report["klines"][tf] = {
            "status": "ok", "rows": len(df),
            "start": str(df.index.min()), "end": str(df.index.max()),
            "gaps": len(gaps), "nan_pct": float(df.isnull().mean().mean()),
        }
        logger.info(f"[{symbol}] {tf}: {len(df):,} rows, {len(gaps)} gaps")

    # ── Load auxiliary data ───────────────────────────────────────────────────
    aux_sources = {
        "open_interest":    RAW_DIR / "open_interest"    / f"{symbol}_1h.parquet",
        "funding_rate":     RAW_DIR / "funding_rate"     / f"{symbol}_8h.parquet",
        "taker_ratio":      RAW_DIR / "taker_ratio"      / f"{symbol}_15m.parquet",
        "long_short_ratio": RAW_DIR / "long_short_ratio" / f"{symbol}_15m.parquet",
    }
    aux = {}
    for name, path in aux_sources.items():
        df = _load(path)
        aux[name] = df
        if df is None:
            report["auxiliary"][name] = {"status": "missing"}
        else:
            report["auxiliary"][name] = {
                "status": "ok", "rows": len(df),
                "start": str(df.index.min()), "end": str(df.index.max()),
            }

    # ── Load macro ────────────────────────────────────────────────────────────
    macro_paths = {
        "btc_dominance":    RAW_DIR / "macro" / "btc_dominance.parquet",
        "fear_greed_index": RAW_DIR / "macro" / "fear_greed_index.parquet",
    }
    macro = {}
    for name, path in macro_paths.items():
        df = _load(path)
        macro[name] = df
        report["macro"][name] = {
            "status": "ok" if df is not None else "missing",
            "rows": len(df) if df is not None else 0,
        }

    # ── Build M15 master frame ────────────────────────────────────────────────
    base_m15 = klines.get("15m")
    if base_m15 is None:
        logger.error(f"[{symbol}] Tidak ada M15 klines — skip.")
        report["output"] = "skipped"
        return report

    master = base_m15.copy()
    master.columns = [f"m15_{c}" for c in master.columns]

    # Join H1, H4, D1 dengan ffill ke M15 grid
    for tf in ("1h", "4h", "1d"):
        df_tf = klines.get(tf)
        if df_tf is None:
            continue
        df_tf = df_tf.rename(columns={c: f"{tf}_{c}" for c in df_tf.columns})
        df_tf_m15 = df_tf.reindex(df_tf.index.union(master.index)).sort_index().ffill()
        master = master.join(df_tf_m15.reindex(master.index), how="left")

    # Join auxiliary dengan ffill
    for name, df_aux in aux.items():
        if df_aux is None:
            continue
        df_aux = df_aux.rename(columns={c: f"{name}_{c}" for c in df_aux.columns})
        df_aux_m15 = df_aux.reindex(df_aux.index.union(master.index)).sort_index().ffill()
        master = master.join(df_aux_m15.reindex(master.index), how="left")

    # Forward-fill macro ke M15 grid
    for name, df_macro in macro.items():
        if df_macro is None:
            continue
        df_macro = df_macro.rename(columns={c: f"macro_{name}_{c}" for c in df_macro.columns})
        resampled = ffill_macro(df_macro, master.index)
        master = master.join(resampled, how="left")

    # Leakage audit
    flagged = audit_leakage(master)
    report["leakage_flagged"] = flagged
    if flagged:
        logger.warning(f"[{symbol}] Potential leakage: {flagged}")

    # Save
    out_path = PROC_DIR / f"{symbol}_clean.parquet"
    _save(master, out_path)
    report["output"]         = str(out_path)
    report["output_rows"]    = len(master)
    report["output_columns"] = len(master.columns)
    logger.info(f"[{symbol}] Saved → {out_path} ({len(master):,} rows × {len(master.columns)} cols)")

    return report


# ─── Main ─────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Data Cleaning Pipeline")
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

    PROC_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    full_report = {"symbols": {}}
    for symbol in coins:
        try:
            full_report["symbols"][symbol] = clean_symbol(symbol)
        except Exception as e:
            logger.exception(f"[{symbol}] Error: {e}")
            full_report["symbols"][symbol] = {"error": str(e)}

    report_path = REPORT_DIR / "cleaning_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2, default=str)
    logger.info(f"Report saved → {report_path}")


if __name__ == "__main__":
    main()
