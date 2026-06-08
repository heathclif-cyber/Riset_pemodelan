"""
pipeline/05a_momentum_labels_v2.py — Momentum Flow Labels V2

Label berbasis FLOW (OFI, CVD, volume imbalance), BUKAN swing structure.
LSTM dengan label ini menjawab pertanyaan berbeda dari LGBM:

  LGBM  : "apakah STRUKTUR pasar mendukung arah ini?"  (swing high/low, liq level)
  LSTM  : "apakah MOMENTUM FLOW mendukung arah ini?"   (OFI, CVD, volume delta)

Pemisahan ini memberikan marginal IC yang nyata untuk LSTM sebagai konfirmasi.

Vote System (4 sinyal, butuh >= 2/4 setuju):
  Signal 1: OFI z-score mean N bar ke depan
  Signal 2: CVD momentum direction N bar ke depan
  Signal 3: Volume delta (per-coin z-scored) N bar ke depan
  Signal 4: Price return N bar ke depan (konfirmasi)

Label: 0=BEARISH  1=NEUTRAL  2=BULLISH
(Sengaja sama numeriknya dengan LABEL_MAP agar compatible dengan model & inference)

Kompatibilitas:
  - Numerik label sama: SHORT=0, FLAT=1, LONG=2 -> BEARISH=0, NEUTRAL=1, BULLISH=2
  - Model LSTM TETAP 3-class (tidak perlu ubah arsitektur)
  - Inference: LGBM=LONG + LSTM=BULLISH -> confirm
               LGBM=LONG + LSTM=BEARISH -> penalize (hard_consensus mode)
               LGBM=LONG + LSTM=NEUTRAL -> mild penalize

Usage:
  python pipeline/05a_momentum_labels_v2.py --all
  python pipeline/05a_momentum_labels_v2.py --coins BTCUSDT ETHUSDT SOLUSDT
  python pipeline/05a_momentum_labels_v2.py --all --n 8      # N=8 bar forward
  python pipeline/05a_momentum_labels_v2.py --all --holdout  # generate untuk holdout
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from config import (
    TRAINING_COINS, LABEL_DIR, TRAIN_CUTOFF_DATE,
    HOLDOUT_DIR,
)
from core.utils import setup_logger

logger = setup_logger("05a_momentum_labels_v2")

# ─── Hyperparameter Label ────────────────────────────────────────────────────
MOMENTUM_N          = 8      # bar ke depan untuk momentum window (8 jam)
OFI_Z_THRESHOLD     = 0.25   # OFI z-score threshold untuk bullish/bearish vote
VOL_DELTA_THRESHOLD = 0.20   # volume delta (z-scored per coin) threshold
MIN_PRICE_RET       = 0.0003 # 0.03% minimum price move untuk vote
CVD_MOM_THRESHOLD   = 0.0    # CVD momentum change threshold (positif = bullish)

LABEL_MAP_MOMENTUM = {0: "BEARISH", 1: "NEUTRAL", 2: "BULLISH"}


def compute_per_coin_z(series: np.ndarray, window: int = 500) -> np.ndarray:
    """
    Rolling z-score per coin untuk normalize fitur raw (volume_delta, cvd_momentum_adv).
    window=500 ≈ 3 minggu data H1 — cukup untuk capture regime terkini.
    Clip pada [-4, 4] untuk cegah outlier ekstrem.
    """
    s = pd.Series(series)
    roll_mean = s.rolling(window=window, min_periods=50).mean()
    roll_std  = s.rolling(window=window, min_periods=50).std().clip(lower=1e-8)
    z = ((s - roll_mean) / roll_std).clip(-4, 4).fillna(0).values
    return z


def compute_momentum_labels(df: pd.DataFrame, N: int = MOMENTUM_N) -> np.ndarray:
    """
    Compute momentum flow labels untuk setiap bar di df.

    Returns: np.ndarray int8 dengan nilai 0=BEARISH, 1=NEUTRAL, 2=BULLISH
    Last N bars selalu NEUTRAL (tidak bisa dihitung forward).

    Tidak ada look-ahead di features — hanya label yang pakai future data.
    Features di training tetap hanya dari window [t-seq_len..t].
    """
    n = len(df)
    labels = np.ones(n, dtype=np.int8)  # default: NEUTRAL

    close = df["close"].values

    # Signal 1: OFI z-score (sudah normalized di features_v3)
    ofi_z = df["ofi_z_score"].values if "ofi_z_score" in df.columns else np.zeros(n)

    # Signal 2: CVD momentum — per-coin z-score untuk fix scale mismatch
    if "cvd_momentum_adv" in df.columns:
        cvd_mom_raw = df["cvd_momentum_adv"].values
        cvd_mom = compute_per_coin_z(cvd_mom_raw)
    else:
        cvd_mom = np.zeros(n)

    # Signal 3: Volume delta — per-coin z-score (range BTC vs DOGE beda 50,000x!)
    if "volume_delta" in df.columns:
        vol_delta_raw = df["volume_delta"].values
        vol_delta = compute_per_coin_z(vol_delta_raw)
    else:
        vol_delta = np.zeros(n)

    # Hitung label untuk setiap bar kecuali N bar terakhir
    for t in range(n - N - 1):
        end = t + N + 1

        # Signal 1: OFI z-score mean ke depan
        ofi_future = float(np.nanmean(ofi_z[t + 1:end]))
        ofi_vote   = (1 if ofi_future >  OFI_Z_THRESHOLD
                      else -1 if ofi_future < -OFI_Z_THRESHOLD else 0)

        # Signal 2: CVD momentum direction ke depan
        cvd_change = float(cvd_mom[end - 1] - cvd_mom[t])
        cvd_vote   = (1 if cvd_change >  CVD_MOM_THRESHOLD
                      else -1 if cvd_change < -CVD_MOM_THRESHOLD else 0)

        # Signal 3: Volume delta persistence ke depan
        vd_future = float(np.nanmean(vol_delta[t + 1:end]))
        vd_vote   = (1 if vd_future >  VOL_DELTA_THRESHOLD
                     else -1 if vd_future < -VOL_DELTA_THRESHOLD else 0)

        # Signal 4: Price return confirmation
        if close[t] > 0 and close[end - 1] > 0 and not (
            np.isnan(close[t]) or np.isnan(close[end - 1])
        ):
            price_ret  = (close[end - 1] - close[t]) / close[t]
            price_vote = (1 if price_ret >  MIN_PRICE_RET
                          else -1 if price_ret < -MIN_PRICE_RET else 0)
        else:
            price_vote = 0

        vote = ofi_vote + cvd_vote + vd_vote + price_vote

        if vote >= 2:
            labels[t] = 2   # BULLISH
        elif vote <= -2:
            labels[t] = 0   # BEARISH
        # else: tetap NEUTRAL (1)

    return labels


def process_coin(coin: str, is_holdout: bool, N: int) -> dict:
    """Process satu koin dan simpan label ke parquet."""
    base_dir = HOLDOUT_DIR / "labeled" if is_holdout else LABEL_DIR
    path = base_dir / f"{coin}_features_v3.parquet"

    if not path.exists():
        logger.warning(f"{coin}: features_v3 tidak ditemukan di {path}, skip")
        return {"coin": coin, "status": "skip", "n": 0}

    df = pd.read_parquet(path)
    df = df.sort_index()

    if not is_holdout:
        df = df[df.index < TRAIN_CUTOFF_DATE]

    if len(df) < N + 50:
        logger.warning(f"{coin}: data terlalu pendek ({len(df)} bar), skip")
        return {"coin": coin, "status": "skip", "n": 0}

    if "close" not in df.columns:
        logger.warning(f"{coin}: kolom 'close' tidak ada, skip")
        return {"coin": coin, "status": "skip", "n": 0}

    labels = compute_momentum_labels(df, N=N)

    # Distribusi label
    n_bull = int((labels == 2).sum())
    n_neu  = int((labels == 1).sum())
    n_bear = int((labels == 0).sum())
    n_tot  = len(labels)

    # Simpan sebagai parquet
    label_df = pd.DataFrame(
        {"momentum_v2_label": labels},
        index=df.index,
    )
    out_path = base_dir / f"{coin}_momentum_v2_labels.parquet"
    label_df.to_parquet(out_path)

    logger.info(
        f"{coin}: {n_tot:,} bars | BULL={n_bull/n_tot*100:.1f}% "
        f"NEU={n_neu/n_tot*100:.1f}% BEAR={n_bear/n_tot*100:.1f}%"
    )
    return {
        "coin": coin, "status": "ok", "n": n_tot,
        "bull_pct": round(n_bull / n_tot * 100, 1),
        "neu_pct":  round(n_neu  / n_tot * 100, 1),
        "bear_pct": round(n_bear / n_tot * 100, 1),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--all",     action="store_true", help="Semua 21 training coins")
    parser.add_argument("--holdout", action="store_true", help="Generate untuk holdout data")
    parser.add_argument("--coins",   nargs="+", default=None)
    parser.add_argument("--n",       type=int, default=MOMENTUM_N,
                        help=f"N bar ke depan untuk momentum window (default {MOMENTUM_N})")
    args = parser.parse_args()

    if args.coins:
        coins = args.coins
    elif args.all:
        coins = TRAINING_COINS
    else:
        coins = TRAINING_COINS[:5]

    split_tag = "holdout" if args.holdout else "training"
    print(f"\n{'='*60}")
    print(f"  MOMENTUM LABELS V2 | N={args.n} bar | {split_tag}")
    print(f"  Coins: {len(coins)}")
    print(f"{'='*60}\n")
    print(f"  Vote system: OFI z-score + CVD momentum + vol_delta_z + price return")
    print(f"  BULLISH = 3-4/4 votes positif | BEARISH = 3-4/4 negatif | NEUTRAL = split\n")

    results = []
    for coin in coins:
        r = process_coin(coin, is_holdout=args.holdout, N=args.n)
        results.append(r)

    ok_results = [r for r in results if r["status"] == "ok"]
    if not ok_results:
        print("ERROR: tidak ada koin berhasil diproses")
        return

    # Agregat distribusi
    avg_bull = sum(r["bull_pct"] for r in ok_results) / len(ok_results)
    avg_neu  = sum(r["neu_pct"]  for r in ok_results) / len(ok_results)
    avg_bear = sum(r["bear_pct"] for r in ok_results) / len(ok_results)

    print(f"\n{'='*60}")
    print(f"  SELESAI: {len(ok_results)}/{len(results)} koin berhasil")
    print(f"  Rata-rata distribusi label:")
    print(f"    BULLISH (2): {avg_bull:.1f}%")
    print(f"    NEUTRAL (1): {avg_neu:.1f}%")
    print(f"    BEARISH (0): {avg_bear:.1f}%")
    print(f"{'='*60}")
    print(f"\n  Target: BULL~25%, NEU~50%, BEAR~25%")
    print(f"  Jika NEU >> 60% -> turunkan threshold (OFI_Z_THRESHOLD, VOL_DELTA_THRESHOLD)")
    print(f"  Jika NEU << 30% -> naikkan threshold\n")

    base_dir = HOLDOUT_DIR / "labeled" if args.holdout else LABEL_DIR
    print(f"  Labels disimpan di: {base_dir}/{{coin}}_momentum_v2_labels.parquet")
    print(f"\n  Next step:")
    print(f"    python pipeline/05b_train_lstm_momentum_v2.py --run-id ic32_lstm_momentum_v2")
    print(f"    python pipeline/05b_train_lstm_momentum_v2.py --run-id ic32_lstm_momentum_v2 --all")


if __name__ == "__main__":
    main()
