"""
pipeline/05b_build_h4_sequences.py — Build H4 Sequence Dataset untuk LSTM Momentum

Tujuan:
  Untuk setiap bar H1 yang punya momentum_label, ekstrak 16 bar H4 sebelumnya
  sebagai input sequence LSTM. Hasilnya adalah dataset siap training.

Flow:
  1. Load {COIN}_momentum_labels.parquet  (H1 dengan momentum_label)
  2. Resample ke H4 → 8 fitur H4-native
  3. Untuk setiap H1 bar: cari 16 H4 bars yang berakhir sebelum bar tersebut
  4. Simpan sebagai numpy arrays per coin + satu combined file

Fitur H4 Sequence (8 fitur):
  h4_return       : return bar-over-bar H4 (bukan absolute price → stasioner)
  volume          : volume H4 (sum H1 volumes dalam 4h window)
  rsi_h4          : RSI di timeframe H4
  ema_21_slope_h4 : kemiringan EMA 21 H4 (arah trend)
  h4_trend        : arah trend H4 (-1/0/1)
  trend_strength  : kekuatan trend H4 (-1 sampai 1)
  cvd_slope_h4    : slope CVD di H4 (flow order)
  atr_percent_h4  : ATR% H4 (regime volatilitas)

Label encoding (sama dengan LABEL_MAP existing):
  SHORT=0, FLAT=1, LONG=2

Output:
  data/training/h4_sequences/{COIN}_seq.npz
    - X : (n_samples, 16, 8) float32
    - y : (n_samples,) int32
    - ts: (n_samples,) int64  ← unix timestamp H1, untuk purged CV

  data/training/h4_sequences/all_coins_seq.npz
    - X, y, ts, coin_id (combined semua koin)

Jalankan:
  python pipeline/05b_build_h4_sequences.py
  python pipeline/05b_build_h4_sequences.py --seq-len 16 --all
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from config import (
    TRAINING_COINS, ALL_COINS, SYMBOL_MAP,
    LABEL_DIR, TRAINING_DIR, TRAIN_CUTOFF_DATE,
)
from core.utils import setup_logger

logger = setup_logger("05b_h4_sequences")

# ─── Konstanta ────────────────────────────────────────────────────────────────

SEQ_DIR = TRAINING_DIR / "h4_sequences"

LABEL_MAP = {"SHORT": 0, "FLAT": 1, "LONG": 2}

# 8 fitur H4 sequence — semua tersedia di parquet existing
H4_RAW_COLS = [
    "rsi_h4",
    "ema_21_slope_h4",
    "h4_trend",
    "trend_strength",
    "cvd_slope_h4",
    "atr_percent_h4",
]
# close dan volume di-derive saat resample (h4_return + volume_norm)
# Total fitur per H4 bar = 8


# ─── Resample H1 → H4 ────────────────────────────────────────────────────────

def resample_to_h4(df: pd.DataFrame) -> pd.DataFrame:
    """
    Resample DataFrame H1 ke H4.
    - close  → last H1 close dalam 4h window
    - volume → sum H1 volumes dalam 4h window
    - H4-native features → last value (sudah forward-filled dari H4)
    """
    h4 = pd.DataFrame(index=df.resample("4h").last().index)
    h4["close"]  = df["close"].resample("4h").last()
    h4["volume"] = df["volume"].resample("4h").sum()
    for col in H4_RAW_COLS:
        if col in df.columns:
            h4[col] = df[col].resample("4h").last()
        else:
            h4[col] = np.nan

    # Derived: return bar-over-bar H4 (menggantikan absolute close)
    h4["h4_return"] = h4["close"].pct_change(fill_method=None)

    # Drop baris NaN (terutama baris pertama karena pct_change)
    h4 = h4.dropna(subset=["close", "h4_return"])
    return h4


def build_feature_matrix(h4: pd.DataFrame) -> np.ndarray:
    """
    Susun fitur H4 ke array [n_h4_bars, 8].
    Urutan: h4_return, volume, rsi_h4, ema_21_slope_h4,
            h4_trend, trend_strength, cvd_slope_h4, atr_percent_h4
    """
    feat_cols = ["h4_return", "volume"] + H4_RAW_COLS
    arr = h4[feat_cols].values.astype(np.float32)
    # Replace inf dan NaN dengan 0
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return arr


# ─── Build sequences per coin ─────────────────────────────────────────────────

def build_coin_sequences(
    sym:     str,
    seq_len: int = 16,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """
    Untuk satu coin:
    Returns:
        X  : (n_samples, seq_len, 8) float32
        y  : (n_samples,) int32
        ts : (n_samples,) int64 — unix timestamp H1 untuk purged CV
    """
    path = LABEL_DIR / f"{sym}_momentum_labels.parquet"
    if not path.exists():
        logger.warning(f"{sym}: momentum labels tidak ditemukan, skip")
        return None

    df = pd.read_parquet(path)
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, utc=True)
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")

    # Filter hanya training period
    df = df[df.index < TRAIN_CUTOFF_DATE]
    if len(df) < seq_len * 4 + 100:
        logger.warning(f"{sym}: data terlalu sedikit, skip")
        return None

    # Resample ke H4
    h4       = resample_to_h4(df)
    h4_arr   = build_feature_matrix(h4)          # [n_h4, 8]
    h4_times = h4.index.values                    # array datetime64

    # H1 bars dengan label valid (bukan NaN)
    h1_valid = df.dropna(subset=["momentum_label"])
    h1_valid = h1_valid[h1_valid["momentum_label"].isin(LABEL_MAP)]

    X_list, y_list, ts_list = [], [], []
    skipped = 0

    # Vectorized: untuk setiap H1 bar, cari posisi di H4 array
    h1_times = h1_valid.index.values  # array datetime64

    # FIX look-ahead: floor H1 ke batas 4h (awal H4 period yang sedang berjalan)
    # Contoh: H1=10:00 → floored=08:00 → searchsorted cari H4 bar >= 08:00
    # → idx_h4 menunjuk ke 08:00 bar → h4_arr[:idx_h4] hanya berisi bar sebelum 08:00
    # Ini menjamin H4 bar yang sedang berjalan (belum closed) tidak masuk sequence.
    h1_times_floored = h1_times.astype("datetime64[4h]").astype("datetime64[ns]")
    idx_h4_arr = np.searchsorted(h4_times, h1_times_floored, side="left")

    for i, (h1_time, idx_h4) in enumerate(zip(h1_times, idx_h4_arr)):
        # idx_h4 = posisi H4 pertama yang >= h1_time
        # → H4 bars yang valid: h4_arr[:idx_h4]
        if idx_h4 < seq_len:
            skipped += 1
            continue

        seq   = h4_arr[idx_h4 - seq_len : idx_h4]   # [seq_len, 8]
        label = LABEL_MAP[h1_valid["momentum_label"].iloc[i]]

        X_list.append(seq)
        y_list.append(label)
        # Simpan sebagai nanoseconds (int64) — pastikan unit konsisten
        ts_list.append(h1_time.astype("datetime64[ns]").astype(np.int64))

    if not X_list:
        logger.warning(f"{sym}: tidak ada sample valid")
        return None

    X  = np.stack(X_list,  axis=0).astype(np.float32)   # [n, seq_len, 8]
    y  = np.array(y_list,  dtype=np.int32)
    ts = np.array(ts_list, dtype=np.int64)

    # Distribusi label
    unique, counts = np.unique(y, return_counts=True)
    dist = {["SHORT","FLAT","LONG"][u]: int(c) for u, c in zip(unique, counts)}
    total = len(y)
    logger.info(
        f"{sym:16}: {total:6,} samples | "
        f"LONG={dist.get('LONG',0)/total*100:5.1f}%  "
        f"FLAT={dist.get('FLAT',0)/total*100:5.1f}%  "
        f"SHORT={dist.get('SHORT',0)/total*100:5.1f}%  "
        f"(skip={skipped})"
    )
    return X, y, ts


# ─── Main ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seq-len", type=int, default=16, help="Panjang H4 sequence (default: 16)")
    p.add_argument("--all",     action="store_true",  help="Pakai semua koin")
    return p.parse_args()


def main():
    args    = parse_args()
    coins   = ALL_COINS if args.all else TRAINING_COINS
    seq_len = args.seq_len

    SEQ_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"Build H4 sequences: seq_len={seq_len}, koin={len(coins)}")
    logger.info(f"Fitur per bar: 8 (h4_return, volume, rsi_h4, ema_21_slope_h4,")
    logger.info(f"                   h4_trend, trend_strength, cvd_slope_h4, atr_percent_h4)")
    logger.info(f"Output: {SEQ_DIR}")
    logger.info("-" * 70)

    all_X, all_y, all_ts, all_cid = [], [], [], []

    for cid, sym in enumerate(coins):
        result = build_coin_sequences(sym, seq_len=seq_len)
        if result is None:
            continue

        X, y, ts = result

        # Simpan per-coin
        out = SEQ_DIR / f"{sym}_seq.npz"
        np.savez_compressed(out, X=X, y=y, ts=ts, coin_id=np.array(cid))
        logger.info(f"  Saved: {out.name}")

        all_X.append(X)
        all_y.append(y)
        all_ts.append(ts)
        all_cid.append(np.full(len(y), cid, dtype=np.int32))

    if not all_X:
        logger.error("Tidak ada data berhasil diproses.")
        return

    # Combined file semua koin
    X_all   = np.concatenate(all_X,   axis=0)
    y_all   = np.concatenate(all_y,   axis=0)
    ts_all  = np.concatenate(all_ts,  axis=0)
    cid_all = np.concatenate(all_cid, axis=0)

    # Sort by timestamp untuk temporal consistency
    order  = np.argsort(ts_all)
    X_all  = X_all[order]
    y_all  = y_all[order]
    ts_all = ts_all[order]
    cid_all= cid_all[order]

    combined_path = SEQ_DIR / "all_coins_seq.npz"
    np.savez_compressed(combined_path, X=X_all, y=y_all, ts=ts_all, coin_id=cid_all)

    # Summary
    total   = len(y_all)
    n_long  = (y_all == 2).sum()
    n_flat  = (y_all == 1).sum()
    n_short = (y_all == 0).sum()
    sep = "=" * 70
    print(f"\n{sep}")
    print(f"  H4 SEQUENCE DATASET — seq_len={seq_len}, {len(coins)} koin")
    print(f"{sep}")
    print(f"  Total samples : {total:>10,}")
    print(f"  Shape X       : {X_all.shape}   (samples × seq × features)")
    print(f"  LONG          : {n_long:>10,}  ({n_long/total*100:.1f}%)")
    print(f"  FLAT          : {n_flat:>10,}  ({n_flat/total*100:.1f}%)")
    print(f"  SHORT         : {n_short:>10,}  ({n_short/total*100:.1f}%)")
    print(f"  File size est : ~{X_all.nbytes / 1e6:.0f} MB (uncompressed float32)")
    print(f"{sep}")
    print(f"  Combined: {combined_path}")
    print(f"  Per-coin: {SEQ_DIR}/{{COIN}}_seq.npz\n")


if __name__ == "__main__":
    main()
