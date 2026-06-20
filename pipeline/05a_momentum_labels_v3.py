"""
pipeline/05a_momentum_labels_v3.py -- Flow-Price Divergence labels

Metode: momentum_v3
  flow_fwd  = z( mean forward delta of cvd_slope_h4, ofi_h4_delta, volume_delta )
  price_fwd = z( mean forward log_ret_1 )
  divergence = price_fwd - flow_fwd

Label:
  BULLISH (2): flow_fwd > FLOW_THR  AND divergence < -DIV_THR  (organic -- flow leads price)
  BEARISH (0): flow_fwd > FLOW_THR  AND divergence > +DIV_THR  (hype exhaust)
  BEARISH (0): flow_fwd < -FLOW_THR AND divergence > +DIV_THR  (distribution)
  NEUTRAL (1): else

Usage:
  python pipeline/05a_momentum_labels_v3.py --all
"""
import argparse, sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from config import TRAINING_COINS, LABEL_DIR, TRAIN_CUTOFF_DATE, HOLDOUT_DIR
from core.utils import setup_logger

logger = setup_logger("05a_momentum_labels_v3")

MOMENTUM_N  = 8
FLOW_THR    = 0.3
DIV_THR     = 0.5
Z_WINDOW    = 500


def per_coin_z(series: np.ndarray, window: int = Z_WINDOW) -> np.ndarray:
    s = pd.Series(series)
    m = s.rolling(window=window, min_periods=50).mean()
    sd = s.rolling(window=window, min_periods=50).std().clip(lower=1e-8)
    return ((s - m) / sd).clip(-4, 4).fillna(0).values


def compute_momentum_v3_labels(df: pd.DataFrame, n: int = MOMENTUM_N) -> np.ndarray:
    labels = np.ones(len(df), dtype=np.int8)
    close = df["close"].values

    cvd_slope = df["cvd_slope_h4"].values if "cvd_slope_h4" in df.columns else np.zeros(len(df))
    ofi_delta = df["ofi_h4_delta"].values if "ofi_h4_delta" in df.columns else np.zeros(len(df))
    vol_delta = df["volume_delta"].values if "volume_delta" in df.columns else np.zeros(len(df))
    vol_delta_z = per_coin_z(vol_delta.astype(np.float64))
    log_ret_1 = df["log_ret_1"].values if "log_ret_1" in df.columns else np.zeros(len(df))

    for t in range(len(df) - n - 1):
        end = t + n + 1
        fwd = slice(t + 1, end)

        dcvd = float(np.nanmean(np.diff(cvd_slope[t:end])))
        dofi = float(np.nanmean(np.diff(ofi_delta[t:end])))
        dvol = float(np.nanmean(vol_delta_z[fwd]))

        flow_raw = dcvd + dofi + dvol
        price_raw = float(np.nanmean(log_ret_1[fwd]))

        flow_fwd = np.clip(flow_raw, -4, 4)
        price_fwd = np.clip(price_raw * 100, -4, 4)
        divergence = price_fwd - flow_fwd

        if flow_fwd > FLOW_THR and divergence < -DIV_THR:
            labels[t] = 2
        elif flow_fwd > FLOW_THR and divergence > DIV_THR:
            labels[t] = 0
        elif flow_fwd < -FLOW_THR and divergence > DIV_THR:
            labels[t] = 0

    return labels


def process_coin(coin: str, is_holdout: bool) -> dict:
    base = HOLDOUT_DIR / "labeled" if is_holdout else LABEL_DIR
    path = base / f"{coin}_features_v3.parquet"
    if not path.exists():
        return {"coin": coin, "status": "skip"}

    df = pd.read_parquet(path).sort_index()
    if not is_holdout:
        df = df[df.index < TRAIN_CUTOFF_DATE]
    if len(df) < MOMENTUM_N + 50:
        return {"coin": coin, "status": "skip"}

    labels = compute_momentum_v3_labels(df)
    out = pd.DataFrame({"momentum_v3_label": labels}, index=df.index)
    out.to_parquet(base / f"{coin}_momentum_v3_labels.parquet")

    n = len(labels)
    bull = int((labels == 2).sum())
    bear = int((labels == 0).sum())
    neu  = int((labels == 1).sum())
    logger.info(
        f"  [{coin}] {n:,} | BULL={bull/n*100:.1f}% NEU={neu/n*100:.1f}% BEAR={bear/n*100:.1f}%"
    )
    return {"coin": coin, "status": "ok", "n": n}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--holdout", action="store_true")
    parser.add_argument("--coins", nargs="*", default=None)
    args = parser.parse_args()

    coins = args.coins or (TRAINING_COINS if args.all else TRAINING_COINS[:3])
    print(f"\n{'='*60}")
    print(f"  momentum_v3 labels | N={MOMENTUM_N} flow_thr={FLOW_THR} div_thr={DIV_THR}")
    print(f"  Coins: {len(coins)} | holdout={args.holdout}")
    print(f"{'='*60}\n")

    for coin in coins:
        process_coin(coin, args.holdout)


if __name__ == "__main__":
    main()