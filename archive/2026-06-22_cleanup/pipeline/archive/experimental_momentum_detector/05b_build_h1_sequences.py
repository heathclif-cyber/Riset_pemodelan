"""
pipeline/05b_build_h1_sequences.py — Build H1 Sequence Dataset untuk LSTM

REKOMENDASI PIPELINE LSTM SAAT INI (2026-06):

**PENTING — Ada dua pendekatan:**

1. **Untuk cascade_v2.5_hybrid & revival v2 spirit**:
   - Gunakan fitur yang **sama dengan LGBM** (bukan jalur ini).

2. **Untuk eksperimen Advanced Momentum Detector**:
   - Pakai 05a → 05b → 05c (jalur ini).

Lihat `pipeline/archive/README.md` untuk detail lengkap.

JANGAN pakai script lama di folder archive/ kecuali untuk audit.

Dua mode utama:

**Mode Recommended (Alignment dengan LGBM - disarankan sekarang):**
  Gunakan label yang sama persis dengan LGBM entry model:
    --label-col label
  LSTM akan dilatih pada task yang sama (SHORT/FLAT/LONG dari swing labeling) tapi dengan
  keunggulan sequence + 18 fitur akumulasi/flow. Sangat cocok untuk peran "sequence confirmation"
  seperti di cascade_v3 (hard_consensus / LSTM sebagai confirmation vote).

**Mode Legacy (Custom Momentum Heads):**
  Untuk eksperimen LSTM sebagai residual corrector + booster (strong_up_label + exhaustion).
  Membutuhkan pipeline/05a_generate_momentum_labels.py terlebih dahulu.

Fitur: 18 fitur dari config.py (LSTM_MOMENTUM_FEATURES) — fokus akumulasi/flow.

Contoh jalankan (mode alignment dengan LGBM):
  python pipeline/05b_build_h1_sequences.py --all --seq-len 32 --label-col label

Contoh jalankan (mode custom momentum):
  python pipeline/05b_build_h1_sequences.py --all --seq-len 32 --label-col strong_up_label
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
    LSTM_MOMENTUM_FEATURES, LSTM_SEQ_FEATURES,
)
from core.utils import setup_logger

logger = setup_logger("05b_h1_sequences")

SEQ_DIR_H1 = TRAINING_DIR / "h1_sequences"
# Default mapping untuk label klasik
DEFAULT_LABEL_MAP = {"SHORT": 0, "FLAT": 1, "LONG": 2}

# Mapping untuk strong momentum quantile (3-class)
STRONG_LABEL_MAP = {"Strong_SHORT": 0, "Normal": 1, "Strong_LONG": 2}

# N_FEATURES dihitung dari definisi sentral di config.py
N_FEATURES = len(LSTM_MOMENTUM_FEATURES)


def build_coin_sequences(
    sym:     str,
    seq_len: int = 32,
    btc_df:    pd.DataFrame | None = None,
    label_col: str = "momentum_label",
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

    # Alpha: OI delta — % change open interest per H1 bar (robust version)
    # Clip lebih ketat dari v1 karena outlier ekstrem di beberapa coin
    if "open_interest" in df.columns and not df["open_interest"].isna().all():
        oi_pct = df["open_interest"].pct_change(fill_method=None)
        df["oi_delta_pct"] = oi_pct.clip(-0.5, 0.5).fillna(0.0)   # v2: lebih konservatif
    else:
        df["oi_delta_pct"] = 0.0

    # Alpha: BTC cross-momentum — market-wide directional bias
    # Untuk BTCUSDT sendiri, gunakan 0 (tidak ada self-reference)
    if sym == "BTCUSDT" or btc_df is None:
        df["btc_h1_return"] = 0.0
    else:
        btc_aligned = btc_df["btc_h1_return"].reindex(df.index, method="ffill").fillna(0.0)
        df["btc_h1_return"] = btc_aligned

    # Pastikan semua fitur tersedia (v2 robust set)
    missing = [c for c in LSTM_MOMENTUM_FEATURES if c not in df.columns]
    if missing:
        logger.warning(f"{sym}: kolom tidak ditemukan: {missing}, skip")
        return None

    # Bersihkan NaN
    df[LSTM_MOMENTUM_FEATURES] = df[LSTM_MOMENTUM_FEATURES].ffill().fillna(0)
    feat_arr   = df[LSTM_MOMENTUM_FEATURES].values.astype(np.float32)
    feat_arr   = np.nan_to_num(feat_arr, nan=0.0, posinf=0.0, neginf=0.0)
    label_arr  = df[label_col].values
    ts_arr     = df.index.values

    # Determine label handling mode
    is_binary = label_col in ["strong_up_label", "upward_exhaustion_label"]

    if label_col == "strong_momentum_label":
        label_map = STRONG_LABEL_MAP
    else:
        label_map = DEFAULT_LABEL_MAP

    X_list, y_list, ts_list = [], [], []

    for i in range(seq_len, len(df)):
        lbl = label_arr[i]

        if is_binary:
            # Binary 0/1 labels from Multi-Head
            if not isinstance(lbl, (int, np.integer)) or lbl not in [0, 1]:
                continue
            y_val = int(lbl)
        else:
            # String-based labels (classic or 3-class)
            if not isinstance(lbl, str) or lbl not in label_map:
                continue
            y_val = label_map[lbl]

        # Sliding window: seq_len bar H1 ending at bar i (inclusive).
        seq = feat_arr[i - seq_len + 1 : i + 1]
        X_list.append(seq)
        y_list.append(y_val)
        ts_list.append(ts_arr[i].astype("datetime64[ns]").astype(np.int64))

    if not X_list:
        logger.warning(f"{sym}: tidak ada sample valid")
        return None

    X  = np.stack(X_list, axis=0).astype(np.float32)
    y  = np.array(y_list, dtype=np.int32)
    ts = np.array(ts_list, dtype=np.int64)

    total = len(y)

    if is_binary:
        positive = int((y == 1).sum())
        logger.info(
            f"{sym:16}: {total:6,} samples | "
            f"Positive(1)={positive/total*100:5.1f}%  |  Negative(0)={(total-positive)/total*100:5.1f}%"
        )
    else:
        dist  = {["SHORT","FLAT","LONG"][u]: int(c)
                 for u, c in zip(*np.unique(y, return_counts=True))}
        logger.info(
            f"{sym:16}: {total:6,} samples | "
            f"LONG={dist.get('LONG',0)/total*100:5.1f}%  "
            f"FLAT={dist.get('FLAT',0)/total*100:5.1f}%  "
            f"SHORT={dist.get('SHORT',0)/total*100:5.1f}%"
        )
    return X, y, ts


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seq-len",   type=int, default=32, help="Sequence length (default 32)")
    p.add_argument("--label-col", type=str, default="label",
                   help="Label column. Recommended: 'label' (same 3-class swing labels as LGBM). "
                        "Other options: 'momentum_label', 'strong_up_label', 'upward_exhaustion_label'")
    p.add_argument("--all",       action="store_true")
    p.add_argument("--features-only", action="store_true",
                   help="Build and save ONLY the feature sequences (X, ts, coin_id). "
                        "No y labels are saved. Useful if you want to experiment with many label variants later.")
    return p.parse_args()


def load_btc_returns() -> pd.DataFrame | None:
    """Load BTC H1 returns untuk cross-momentum feature (dipanggil sekali di main)."""
    btc_path = LABEL_DIR / "BTCUSDT_momentum_labels.parquet"
    if not btc_path.exists():
        logger.warning("BTCUSDT momentum labels tidak ditemukan — btc_h1_return akan 0")
        return None
    btc = pd.read_parquet(btc_path)
    if not isinstance(btc.index, pd.DatetimeIndex):
        btc.index = pd.to_datetime(btc.index, utc=True)
    if btc.index.tz is None:
        btc.index = btc.index.tz_localize("UTC")
    btc = btc[btc.index < TRAIN_CUTOFF_DATE].copy()
    btc["btc_h1_return"] = btc["close"].pct_change(fill_method=None).fillna(0.0)
    logger.info(f"BTC cross-momentum loaded: {len(btc):,} bars")
    return btc[["btc_h1_return"]]


def main():
    args    = parse_args()
    coins   = ALL_COINS if args.all else TRAINING_COINS
    seq_len = args.seq_len

    SEQ_DIR_H1.mkdir(parents=True, exist_ok=True)
    logger.info(f"Build H1 sequences: seq_len={seq_len}, koin={len(coins)}, features={N_FEATURES}")
    logger.info(f"Output: {SEQ_DIR_H1}")
    logger.info("-" * 65)

    btc_df  = load_btc_returns()
    all_X, all_ts, all_cid = [], [], []
    all_y = [] if not args.features_only else None

    label_col = args.label_col
    if args.features_only:
        logger.info("Mode: --features-only → hanya menyimpan X (18 fitur), ts, coin_id. Tidak ada y.")
    else:
        logger.info(f"Using label column: {label_col}")

    for cid, sym in enumerate(coins):
        if args.features_only:
            # Build sequences without labels (y is ignored)
            result = build_coin_sequences(sym, seq_len=seq_len, btc_df=btc_df, label_col="momentum_label")
            if result is None:
                continue
            X, _, ts = result   # ignore y
            y = None
        else:
            result = build_coin_sequences(sym, seq_len=seq_len, btc_df=btc_df, label_col=label_col)
            if result is None:
                continue
            X, y, ts = result

        if y is not None:
            np.savez_compressed(SEQ_DIR_H1 / f"{sym}_seq.npz", X=X, y=y, ts=ts, coin_id=np.array(cid))
            all_y.append(y)
        else:
            np.savez_compressed(SEQ_DIR_H1 / f"{sym}_seq.npz", X=X, ts=ts, coin_id=np.array(cid))

        all_X.append(X)
        all_ts.append(ts)
        all_cid.append(np.full(len(X), cid, dtype=np.int32))

    if not all_X:
        logger.error("Tidak ada data berhasil diproses.")
        return

    X_all   = np.concatenate(all_X,   axis=0)
    ts_all  = np.concatenate(all_ts,  axis=0)
    cid_all = np.concatenate(all_cid, axis=0)

    order   = np.argsort(ts_all)
    X_all, ts_all, cid_all = X_all[order], ts_all[order], cid_all[order]

    save_dict = {"X": X_all, "ts": ts_all, "coin_id": cid_all}
    if all_y is not None and len(all_y) > 0:
        y_all = np.concatenate(all_y, axis=0)[order]
        save_dict["y"] = y_all

    np.savez_compressed(SEQ_DIR_H1 / "all_coins_seq.npz", **save_dict)

    total   = len(X_all)
    sep = "=" * 65
    print(f"\n{sep}")
    print(f"  H1 SEQUENCE DATASET — seq_len={seq_len}, {len(coins)} koin, {N_FEATURES} fitur")
    print(f"{sep}")
    print(f"  Total samples : {total:>10,}")
    print(f"  Shape X       : {X_all.shape}")

    if "y" in save_dict:
        y_all = save_dict["y"]
        if label_col in ["strong_up_label", "upward_exhaustion_label"]:
            pos = (y_all == 1).sum()
            print(f"  Positive (1)  : {pos:>10,}  ({pos/total*100:5.1f}%)")
            print(f"  Negative (0)  : {total-pos:>10,}  ({(total-pos)/total*100:5.1f}%)")
        elif label_col == "strong_momentum_label":
            print(f"  Strong_LONG   : {(y_all==2).sum():>10,}  ({(y_all==2).mean()*100:.1f}%)")
            print(f"  Normal        : {(y_all==1).sum():>10,}  ({(y_all==1).mean()*100:.1f}%)")
            print(f"  Strong_SHORT  : {(y_all==0).sum():>10,}  ({(y_all==0).mean()*100:.1f}%)")
        else:
            # Standard LGBM swing labels or other 3-class
            print(f"  LONG          : {(y_all==2).sum():>10,}  ({(y_all==2).mean()*100:.1f}%)")
            print(f"  FLAT          : {(y_all==1).sum():>10,}  ({(y_all==1).mean()*100:.1f}%)")
            print(f"  SHORT         : {(y_all==0).sum():>10,}  ({(y_all==0).mean()*100:.1f}%)")
    else:
        print("  Labels        : NOT SAVED (features-only mode)")

    print(f"  File size est : ~{X_all.nbytes/1e6:.0f} MB (uncompressed)")
    print(f"{sep}")
    print(f"  Combined: {SEQ_DIR_H1}/all_coins_seq.npz\n")


if __name__ == "__main__":
    main()
