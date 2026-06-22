"""
pipeline/03e_regime_hmm.py — HMM Regime Detection Generator (Simon Step)
Generate walk-forward OOF regime labels untuk semua koin.

Pipeline:
  1. Load H4 OHLCV dari processed parquet (resampled dari H1)
  2. Filter < TRAIN_CUTOFF_DATE
  3. GaussianHMM walk-forward OOF (leak-free)
  4. Forward-fill H4 regime ke H1 timestamps
  5. Save: data/training/labeled/{coin}_regime_h1.parquet

4 States (canonical, sorted by mean_return):
  0: TRENDING_DOWN  — return rendah, momentum turun
  1: RANGING_LOW_VOL  — sideways, volatilitas rendah
  2: RANGING_HIGH_VOL — sideways, volatilitas tinggi
  3: TRENDING_UP    — return tinggi, momentum naik

Usage:
    python pipeline/03e_regime_hmm.py
    python pipeline/03e_regime_hmm.py --coins SOLUSDT ETHUSDT
    python pipeline/03e_regime_hmm.py --n-states 3
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from config import (
    TRAINING_COINS, ALL_COINS,
    PROC_DIR, LABEL_DIR,
    TRAIN_CUTOFF_DATE,
    HMM_N_STATES, HMM_N_ITER, HMM_N_FOLDS, HMM_PURGE_H4,
)
from core.regime import generate_oof_regime_labels, encode_regime
from core.utils import setup_logger

logger = setup_logger("03e_regime_hmm")


def load_h4_from_processed(coin: str, cutoff=None) -> pd.DataFrame:
    """
    Load H4 bars dari processed parquet.
    Processed file berisi H1 index dengan kolom 4h_open/high/low/close/volume.
    Resample ke H4 proper dengan mengambil bar terakhir setiap 4 jam.
    """
    path = PROC_DIR / f"{coin}_clean.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Processed file tidak ada: {path}")

    df = pd.read_parquet(path)
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, utc=True)
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")

    if cutoff is not None:
        df = df[df.index < cutoff]

    # Extract H4 bars: ambil setiap baris dimana hour % 4 == 0
    # (H4 bar tutup di jam 0, 4, 8, 12, 16, 20)
    h4_mask = df.index.hour % 4 == 0
    df_h4 = df[h4_mask].copy()

    # rename ke kolom yang diexpect core/regime.py
    rename_map = {
        "4h_open":   "open",
        "4h_high":   "high",
        "4h_low":    "low",
        "4h_close":  "close",
        "4h_volume": "volume",
    }
    df_h4 = df_h4.rename(columns=rename_map)

    # pastikan kolom yang dibutuhkan ada
    needed = ["open", "high", "low", "close", "volume"]
    missing = [c for c in needed if c not in df_h4.columns]
    if missing:
        raise ValueError(f"Missing H4 columns: {missing}")

    df_h4 = df_h4[needed].dropna(subset=["close", "volume"])
    return df_h4


def h4_to_h1(regime_h4: pd.Series, h1_index: pd.DatetimeIndex) -> pd.Series:
    """
    Forward-fill H4 regime labels ke H1 grid.
    Setiap H4 bar 'mendominasi' 4 H1 bar berikutnya.
    """
    # reindex ke H1, lalu forward-fill
    regime_h1 = regime_h4.reindex(h1_index, method="ffill")
    # fill sisa NaN (sebelum data H4 dimulai) dengan RANGING_LOW_VOL
    regime_h1 = regime_h1.fillna("RANGING_LOW_VOL")
    return regime_h1


def process_coin(
    coin: str,
    n_states: int,
    n_iter: int,
    n_folds: int,
    purge_h4: int,
    cutoff=None,
    btc_h4: pd.DataFrame | None = None,
) -> bool:
    logger.info(f"[{coin}] Processing HMM regime...")
    try:
        # load H4 data
        df_h4 = load_h4_from_processed(coin, cutoff=cutoff)
        if len(df_h4) < 200:
            logger.warning(f"[{coin}] Terlalu sedikit H4 bars ({len(df_h4)}), skip")
            return False
        logger.info(f"[{coin}] H4 bars: {len(df_h4)} | {df_h4.index.min().date()} - {df_h4.index.max().date()}")

        # BTC cross-asset features (jika bukan BTC sendiri dan data tersedia)
        btc_ctx = btc_h4 if (coin != "BTCUSDT" and btc_h4 is not None) else None

        # generate OOF regime labels (walk-forward, leak-free)
        regime_h4 = generate_oof_regime_labels(
            df_h4,
            n_states=n_states,
            n_folds=n_folds,
            purge=purge_h4,
            n_iter=n_iter,
            btc_h4=btc_ctx,
        )

        # load H1 features untuk dapatkan H1 index
        feat_path = LABEL_DIR / f"{coin}_features_v3.parquet"
        if not feat_path.exists():
            logger.warning(f"[{coin}] Feature parquet tidak ada, skip H1 alignment")
            return False

        df_feat = pd.read_parquet(feat_path, columns=[])
        if cutoff is not None:
            df_feat = df_feat[df_feat.index < cutoff]
        h1_index = df_feat.index

        # forward-fill H4 regime ke H1
        regime_h1 = h4_to_h1(regime_h4, h1_index)

        # encode integer
        regime_enc = encode_regime(regime_h1, n_states=n_states)

        # build output DataFrame
        out = pd.DataFrame({
            "hmm_regime":     regime_h1,
            "hmm_regime_enc": regime_enc,
        }, index=h1_index)

        # simpan
        out_path = LABEL_DIR / f"{coin}_regime_h1.parquet"
        out.to_parquet(out_path)

        dist = regime_h1.value_counts(normalize=True).round(3).to_dict()
        logger.info(f"[{coin}] Saved -> {out_path.name} | dist={dist}")
        return True

    except Exception as e:
        logger.error(f"[{coin}] Error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def main():
    parser = argparse.ArgumentParser(description="HMM Regime Detection untuk semua koin")
    parser.add_argument("--coins", nargs="+", default=None,
                        help="Koin spesifik (default: TRAINING_COINS)")
    parser.add_argument("--all", action="store_true", help="Pakai ALL_COINS")
    parser.add_argument("--n-states", type=int, default=HMM_N_STATES)
    parser.add_argument("--n-iter",   type=int, default=HMM_N_ITER)
    parser.add_argument("--n-folds",  type=int, default=HMM_N_FOLDS)
    parser.add_argument("--purge-h4", type=int, default=HMM_PURGE_H4)
    args = parser.parse_args()

    if args.coins:
        coins = args.coins
    elif args.all:
        coins = ALL_COINS
    else:
        coins = TRAINING_COINS

    print(f"\n{'='*65}")
    print(f" HMM REGIME DETECTION | {len(coins)} koin | n_states={args.n_states}")
    print(f" Cutoff: {TRAIN_CUTOFF_DATE.date()}")
    print(f" Config: n_iter={args.n_iter}, n_folds={args.n_folds}, purge_h4={args.purge_h4}")
    print(f"{'='*65}\n")

    # Load BTC H4 sekali untuk semua koin (cross-asset context)
    btc_h4 = None
    try:
        btc_h4 = load_h4_from_processed("BTCUSDT", cutoff=TRAIN_CUTOFF_DATE)
        logger.info(f"BTC H4 loaded: {len(btc_h4)} bars (cross-asset context)")
    except Exception as e:
        logger.warning(f"Gagal load BTC H4 ({e}) — HMM tanpa BTC features")

    success, failed = [], []
    for coin in coins:
        ok = process_coin(
            coin,
            n_states=args.n_states,
            n_iter=args.n_iter,
            n_folds=args.n_folds,
            purge_h4=args.purge_h4,
            cutoff=TRAIN_CUTOFF_DATE,
            btc_h4=btc_h4,
        )
        (success if ok else failed).append(coin)

    print(f"\n{'='*65}")
    print(f" SELESAI: {len(success)} sukses | {len(failed)} gagal")
    if failed:
        print(f" Failed: {failed}")
    print(f"{'='*65}")

    # Simpan juga untuk holdout data jika ada
    holdout_proc = Path("data/holdout-test/processed")
    holdout_label = Path("data/holdout-test/labeled")
    if holdout_proc.exists():
        print(f"\nGenerating holdout regime labels...")
        for coin in coins:
            try:
                # untuk holdout: fit model pada semua training data, predict holdout
                df_h4_train = load_h4_from_processed(coin, cutoff=TRAIN_CUTOFF_DATE)
                from core.regime import fit_hmm, predict_hmm
                btc_ctx_hold = btc_h4 if (coin != "BTCUSDT" and btc_h4 is not None) else None
                model, _, state_map = fit_hmm(df_h4_train, n_states=args.n_states, n_iter=args.n_iter, btc_h4=btc_ctx_hold)

                # load holdout processed
                hold_proc_path = holdout_proc / f"{coin}_clean.parquet"
                if not hold_proc_path.exists():
                    continue
                df_h4_hold = pd.read_parquet(hold_proc_path)
                if not isinstance(df_h4_hold.index, pd.DatetimeIndex):
                    df_h4_hold.index = pd.to_datetime(df_h4_hold.index, utc=True)
                h4_mask = df_h4_hold.index.hour % 4 == 0
                df_h4_hold = df_h4_hold[h4_mask].rename(columns={
                    "4h_close": "close", "4h_volume": "volume",
                    "4h_open": "open", "4h_high": "high", "4h_low": "low",
                })
                if "close" not in df_h4_hold.columns:
                    continue

                regime_hold_h4 = pd.Series(
                    predict_hmm(model, df_h4_hold[["open","high","low","close","volume"]], state_map, btc_h4=btc_ctx_hold),
                    index=df_h4_hold.index,
                )

                hold_feat_path = holdout_label / f"{coin}_features_v3.parquet"
                if not hold_feat_path.exists():
                    continue
                df_hold_feat = pd.read_parquet(hold_feat_path, columns=[])
                regime_hold_h1 = h4_to_h1(regime_hold_h4, df_hold_feat.index)
                regime_hold_enc = encode_regime(regime_hold_h1, n_states=args.n_states)

                out_hold = pd.DataFrame({
                    "hmm_regime":     regime_hold_h1,
                    "hmm_regime_enc": regime_hold_enc,
                }, index=df_hold_feat.index)
                out_hold_path = holdout_label / f"{coin}_regime_h1.parquet"
                out_hold.to_parquet(out_hold_path)
                logger.info(f"[{coin}] Holdout regime saved")
            except Exception as e:
                logger.warning(f"[{coin}] Holdout regime failed: {e}")


if __name__ == "__main__":
    main()
