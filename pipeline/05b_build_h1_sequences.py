"""
pipeline/05b_build_h1_sequences.py — Build H1 Sequence Dataset untuk LSTM Momentum

Fitur H1 Sequence (12 fitur) — v2: trajectory-focused, tidak duplikasi LGBM snapshot:

  PRICE TRAJECTORY (3):
  h1_return       : return bar-over-bar H1 — inti sinyal arah per bar
  log_ret_5       : 5-bar return — slope medium-term dalam sequence
  log_ret_20      : 20-bar return — arah tren lebih panjang dalam sequence

  ORDER FLOW TRAJECTORY (3):
  volume_delta    : taker buy-sell imbalance per bar — tekanan beli/jual
  ofi_raw         : order flow imbalance H1 — smart money flow
  ofi_acceleration: rate of change OFI — apakah smart money makin agresif?

  MOMENTUM OSCILLATORS (3 — LSTM lihat EVOLUSI, bukan snapshot):
  rsi_6           : RSI cepat — apakah momentum naik/turun konsisten?
  stochrsi_k      : stochastic RSI — posisi overbought/oversold dalam sequence
  vwdp_smooth     : VWAP delta pressure (smoothed) — tekanan akumulasi/distribusi

  VOLATILITY + STRUCTURE (3):
  atr_14_h1       : volatilitas H1 absolut — apakah market melebar/menyempit?
  vol_ratio_20    : volume vs 20-bar baseline — deteksi volume spike
  bars_since_BOS  : jarak dari BOS terakhir — konteks struktur temporal

Fitur yang DIHAPUS vs v1 (duplikat snapshot LGBM):
  volume, h4_trend, trend_strength, ema_21_slope_h4, MSB_BOS, atr_percent_h4

Output:
  data/training/h1_sequences/{COIN}_seq.npz
  data/training/h1_sequences/all_coins_seq.npz

Jalankan:
  python pipeline/05b_build_h1_sequences.py --all
  python pipeline/05b_build_h1_sequences.py --seq-len 32 --all
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

logger = setup_logger("05b_h1_sequences")

SEQ_DIR_H1 = TRAINING_DIR / "h1_sequences"
LABEL_MAP  = {"SHORT": 0, "FLAT": 1, "LONG": 2}

H1_SEQ_FEATURES = [
    # Price trajectory — LGBM tahu nilai saat ini, LSTM tahu urutan 32 nilai
    "h1_return",        # derived: pct_change dari close (bar-over-bar)
    "log_ret_5",        # 5-bar return: slope medium-term
    "log_ret_20",       # 20-bar return: arah tren lebih panjang
    # Order flow trajectory — evolusi tekanan beli/jual
    "volume_delta",     # taker buy-sell imbalance per bar
    "ofi_raw",          # order flow imbalance H1
    "ofi_acceleration", # rate of change OFI (smart money makin agresif?)
    # Momentum oscillators — LSTM lihat evolusi, bukan snapshot
    "rsi_6",            # fast RSI
    "stochrsi_k",       # stochastic RSI
    "vwdp_smooth",      # VWAP delta pressure smoothed
    # Volatility + structure
    "atr_14_h1",        # H1 volatility
    "vol_ratio_20",     # volume vs 20-bar baseline (spike detection)
    "bars_since_BOS",   # bars since last structure break
]
N_FEATURES = len(H1_SEQ_FEATURES)  # 12


def build_coin_sequences(
    sym:     str,
    seq_len: int = 32,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:

    path = LABEL_DIR / f"{sym}_momentum_labels.parquet"
    if not path.exists():
        logger.warning(f"{sym}: momentum labels tidak ditemukan, skip")
        return None

    df = pd.read_parquet(path)
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, utc=True)
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")

    df = df[df.index < TRAIN_CUTOFF_DATE].copy()
    if len(df) < seq_len + 100:
        logger.warning(f"{sym}: data terlalu sedikit, skip")
        return None

    # Derive h1_return dari close (bar-over-bar return, stasioner)
    df["h1_return"] = df["close"].pct_change(fill_method=None)

    # Pastikan semua fitur tersedia
    missing = [c for c in H1_SEQ_FEATURES if c not in df.columns]
    if missing:
        logger.warning(f"{sym}: kolom tidak ditemukan: {missing}, skip")
        return None

    # Bersihkan NaN
    df[H1_SEQ_FEATURES] = df[H1_SEQ_FEATURES].ffill().fillna(0)
    feat_arr   = df[H1_SEQ_FEATURES].values.astype(np.float32)
    feat_arr   = np.nan_to_num(feat_arr, nan=0.0, posinf=0.0, neginf=0.0)
    label_arr  = df["momentum_label"].values
    ts_arr     = df.index.values

    X_list, y_list, ts_list = [], [], []

    for i in range(seq_len, len(df)):
        lbl = label_arr[i]
        if not isinstance(lbl, str) or lbl not in LABEL_MAP:
            continue

        # Sliding window: seq_len bar H1 sebelum bar i (tidak include bar i)
        # Tidak ada look-ahead: kita ambil [i-seq_len : i]
        seq = feat_arr[i - seq_len : i]   # [seq_len, 12]
        X_list.append(seq)
        y_list.append(LABEL_MAP[lbl])
        ts_list.append(ts_arr[i].astype("datetime64[ns]").astype(np.int64))

    if not X_list:
        logger.warning(f"{sym}: tidak ada sample valid")
        return None

    X  = np.stack(X_list, axis=0).astype(np.float32)
    y  = np.array(y_list, dtype=np.int32)
    ts = np.array(ts_list, dtype=np.int64)

    dist  = {["SHORT","FLAT","LONG"][u]: int(c)
             for u, c in zip(*np.unique(y, return_counts=True))}
    total = len(y)
    logger.info(
        f"{sym:16}: {total:6,} samples | "
        f"LONG={dist.get('LONG',0)/total*100:5.1f}%  "
        f"FLAT={dist.get('FLAT',0)/total*100:5.1f}%  "
        f"SHORT={dist.get('SHORT',0)/total*100:5.1f}%"
    )
    return X, y, ts


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seq-len", type=int, default=32)
    p.add_argument("--all",     action="store_true")
    return p.parse_args()


def main():
    args    = parse_args()
    coins   = ALL_COINS if args.all else TRAINING_COINS
    seq_len = args.seq_len

    SEQ_DIR_H1.mkdir(parents=True, exist_ok=True)
    logger.info(f"Build H1 sequences: seq_len={seq_len}, koin={len(coins)}, features={N_FEATURES}")
    logger.info(f"Output: {SEQ_DIR_H1}")
    logger.info("-" * 65)

    all_X, all_y, all_ts, all_cid = [], [], [], []

    for cid, sym in enumerate(coins):
        result = build_coin_sequences(sym, seq_len=seq_len)
        if result is None:
            continue
        X, y, ts = result
        np.savez_compressed(SEQ_DIR_H1 / f"{sym}_seq.npz", X=X, y=y, ts=ts, coin_id=np.array(cid))
        all_X.append(X); all_y.append(y)
        all_ts.append(ts); all_cid.append(np.full(len(y), cid, dtype=np.int32))

    if not all_X:
        logger.error("Tidak ada data berhasil diproses.")
        return

    X_all   = np.concatenate(all_X,   axis=0)
    y_all   = np.concatenate(all_y,   axis=0)
    ts_all  = np.concatenate(all_ts,  axis=0)
    cid_all = np.concatenate(all_cid, axis=0)

    order   = np.argsort(ts_all)
    X_all, y_all, ts_all, cid_all = X_all[order], y_all[order], ts_all[order], cid_all[order]

    np.savez_compressed(SEQ_DIR_H1 / "all_coins_seq.npz",
                        X=X_all, y=y_all, ts=ts_all, coin_id=cid_all)

    total   = len(y_all)
    sep = "=" * 65
    print(f"\n{sep}")
    print(f"  H1 SEQUENCE DATASET — seq_len={seq_len}, {len(coins)} koin, {N_FEATURES} fitur")
    print(f"{sep}")
    print(f"  Total samples : {total:>10,}")
    print(f"  Shape X       : {X_all.shape}")
    print(f"  LONG          : {(y_all==2).sum():>10,}  ({(y_all==2).mean()*100:.1f}%)")
    print(f"  FLAT          : {(y_all==1).sum():>10,}  ({(y_all==1).mean()*100:.1f}%)")
    print(f"  SHORT         : {(y_all==0).sum():>10,}  ({(y_all==0).mean()*100:.1f}%)")
    print(f"  File size est : ~{X_all.nbytes/1e6:.0f} MB (uncompressed)")
    print(f"{sep}")
    print(f"  Combined: {SEQ_DIR_H1}/all_coins_seq.npz\n")


if __name__ == "__main__":
    main()
