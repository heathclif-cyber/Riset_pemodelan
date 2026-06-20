"""
pipeline/05a_momentum_labels_v4.py -- Continuation labels on pump/dump bars only

Paradigma (beda dari v3 divergence):
  - Hanya bar pump/dump yang bisa directional; sisanya NEUTRAL
  - BULL: harga lanjut naik + flow masih positif (organic continuation)
  - BEAR: harga lanjut turun / dump market / flow mati saat pump
  - NEU: chop / belum jelas

Gate pump/dump bar:
  vol_spike_zscore >= 2.0  OR  range_expansion_h4 >= 1.5

Label (hanya jika gate=True, butuh N bar forward):
  fwd_ret = log(close[t+N] / close[t])
  flow_fwd = mean delta(cvd_slope_h4 + ofi_h4_delta) over [t+1..t+N]

  BULLISH (2): fwd_ret >= FWD_BULL_THR  AND flow_fwd > FLOW_CONT_THR
  BEARISH (0): fwd_ret <= FWD_BEAR_THR
               OR (market_stress AND fwd_ret < 0)
               OR (fwd_ret < 0 AND flow_fwd < -FLOW_CONT_THR)
  NEUTRAL (1): else

market_stress: median 4-bar return 21 koin < MARKET_STRESS_THR

Usage:
  python pipeline/05a_momentum_labels_v4.py --all
  python pipeline/05a_momentum_labels_v4.py --all --holdout
"""
import argparse, sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from config import TRAINING_COINS, LABEL_DIR, TRAIN_CUTOFF_DATE, HOLDOUT_DIR
from core.utils import setup_logger

logger = setup_logger("05a_momentum_labels_v4")

FWD_N              = 8
FWD_BULL_THR       = 0.010   # +1.0% dalam 8 jam (momentum kuat, opsi A)
FWD_BEAR_THR       = -0.010  # -1.0% (dump jelas)
FLOW_CONT_THR      = 0.0     # flow forward harus positif untuk BULL
VOL_SPIKE_THR      = 2.0
RANGE_EXP_THR      = 1.5
MARKET_STRESS_THR  = -0.01   # median 4-bar ret 21 koin < -1%
MARKET_RET_BARS    = 4


def pump_dump_gate(df: pd.DataFrame) -> np.ndarray:
    vs = df["vol_spike_zscore"].values if "vol_spike_zscore" in df.columns else np.zeros(len(df))
    re = df["range_expansion_h4"].values if "range_expansion_h4" in df.columns else np.zeros(len(df))
    return (vs >= VOL_SPIKE_THR) | (re >= RANGE_EXP_THR)


def build_market_stress_index(coins: list, base_dir: Path, is_holdout: bool) -> pd.Series:
    """Median 4-bar return across coins per timestamp."""
    ret_frames = []
    for coin in coins:
        path = base_dir / f"{coin}_features_v3.parquet"
        if not path.exists():
            continue
        df = pd.read_parquet(path).sort_index()
        if not is_holdout:
            df = df[df.index < TRAIN_CUTOFF_DATE]
        if "close" not in df.columns or len(df) < 10:
            continue
        ret4 = np.log(df["close"] / df["close"].shift(MARKET_RET_BARS).replace(0, np.nan))
        ret_frames.append(ret4.rename(coin))

    if not ret_frames:
        return pd.Series(dtype=float)

    panel = pd.concat(ret_frames, axis=1)
    median_ret = panel.median(axis=1)
    stress = median_ret < MARKET_STRESS_THR
    logger.info(
        f"  market_stress bars: {stress.sum():,} / {len(stress):,} "
        f"({stress.mean()*100:.1f}%)"
    )
    return stress


def compute_labels(df: pd.DataFrame, market_stress: pd.Series | None) -> np.ndarray:
    n = len(df)
    labels = np.ones(n, dtype=np.int8)
    gate = pump_dump_gate(df)

    close = df["close"].values.astype(np.float64)
    cvd = df["cvd_slope_h4"].values if "cvd_slope_h4" in df.columns else np.zeros(n)
    ofi = df["ofi_h4_delta"].values if "ofi_h4_delta" in df.columns else np.zeros(n)
    flow = cvd + ofi

    stress_arr = np.zeros(n, dtype=bool)
    if market_stress is not None and len(market_stress) > 0:
        joined = df.index.to_series().map(market_stress).fillna(False).values
        stress_arr = joined.astype(bool)

    for t in range(n - FWD_N):
        if not gate[t]:
            continue

        c0 = close[t]
        cN = close[t + FWD_N]
        if c0 <= 0 or np.isnan(c0) or np.isnan(cN):
            continue

        fwd_ret = float(np.log(cN / c0))
        flow_fwd = float(np.nanmean(np.diff(flow[t:t + FWD_N + 1])))

        if fwd_ret >= FWD_BULL_THR and flow_fwd > FLOW_CONT_THR:
            labels[t] = 2
        elif fwd_ret <= FWD_BEAR_THR or (stress_arr[t] and fwd_ret < 0):
            labels[t] = 0

    return labels


def process_coin(coin: str, base_dir: Path, market_stress: pd.Series, is_holdout: bool) -> dict:
    path = base_dir / f"{coin}_features_v3.parquet"
    if not path.exists():
        return {"coin": coin, "status": "skip"}

    df = pd.read_parquet(path).sort_index()
    if not is_holdout:
        df = df[df.index < TRAIN_CUTOFF_DATE]
    if len(df) < FWD_N + 50:
        return {"coin": coin, "status": "skip"}

    labels = compute_labels(df, market_stress)
    gate = pump_dump_gate(df)
    n = len(labels)

    out = pd.DataFrame({
        "momentum_v4_label": labels,
        "is_pump_dump_bar": gate.astype(np.int8),
    }, index=df.index)
    out.to_parquet(base_dir / f"{coin}_momentum_v4_labels.parquet")

    on_gate = gate.sum()
    sub = labels[gate]
    bull = int((sub == 2).sum())
    bear = int((sub == 0).sum())
    neu_g = int((sub == 1).sum())

    logger.info(
        f"  [{coin}] {n:,} bars | pump/dump={on_gate/n*100:.1f}% | "
        f"on_gate: BULL={bull/max(on_gate,1)*100:.0f}% "
        f"NEU={neu_g/max(on_gate,1)*100:.0f}% BEAR={bear/max(on_gate,1)*100:.0f}%"
    )
    return {"coin": coin, "status": "ok", "n": n, "pump_pct": on_gate / n}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--holdout", action="store_true")
    parser.add_argument("--coins", nargs="*", default=None)
    args = parser.parse_args()

    coins = args.coins or TRAINING_COINS
    is_holdout = args.holdout
    base_dir = (HOLDOUT_DIR / "labeled") if is_holdout else LABEL_DIR

    print(f"\n{'='*60}")
    print(f"  momentum_v4 labels | N={FWD_N} pump_gate vol>={VOL_SPIKE_THR}")
    print(f"  BULL: fwd>={FWD_BULL_THR*100:.1f}% + flow>0")
    print(f"  BEAR: fwd<={FWD_BEAR_THR*100:.1f}% OR market_stress+red")
    print(f"  Mode: {'holdout' if is_holdout else 'training'} | coins={len(coins)}")
    print(f"{'='*60}\n")

    logger.info("Building market stress index...")
    market_stress = build_market_stress_index(coins, base_dir, is_holdout)

    for coin in coins:
        process_coin(coin, base_dir, market_stress, is_holdout)


if __name__ == "__main__":
    main()