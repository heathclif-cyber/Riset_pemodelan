"""
pipeline/03e_regime_hmm_holdout.py
Generate HMM regime labels untuk holdout period (Apr-Jun 2026).

Strategy:
  1. Fit HMM on training H4 data (< TRAIN_CUTOFF_DATE = 2026-04-01)
  2. Predict regime for holdout H4 bars (Apr-Jun 2026)
  3. Forward-fill to H1, save to HOLDOUT_DIR/labeled/{coin}_regime_h1.parquet

OOS-clean: model tidak lihat holdout data saat fitting.
"""

import sys
import joblib
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from config import (
    ALL_COINS, PROC_DIR, HOLDOUT_DIR,
    TRAIN_CUTOFF_DATE, HMM_N_STATES, HMM_N_ITER,
)
from core.regime import fit_hmm, predict_hmm, encode_regime
from core.utils import setup_logger

logger = setup_logger("03e_regime_hmm_holdout")

OUT_DIR = HOLDOUT_DIR / "labeled"
HMM_MODEL_DIR = ROOT / "models" / "hmm"
HMM_MODEL_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_h4_from_processed(path: Path, start=None, end=None) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, utc=True)
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    if start is not None:
        df = df[df.index >= start]
    if end is not None:
        df = df[df.index < end]

    h4_mask = df.index.hour % 4 == 0
    df_h4 = df[h4_mask].copy()

    rename_map = {
        "4h_open": "open", "4h_high": "high",
        "4h_low": "low", "4h_close": "close", "4h_volume": "volume",
    }
    df_h4 = df_h4.rename(columns={k: v for k, v in rename_map.items() if k in df_h4.columns})

    needed = ["open", "high", "low", "close", "volume"]
    missing = [c for c in needed if c not in df_h4.columns]
    if missing:
        raise ValueError(f"Missing H4 columns: {missing}")
    return df_h4[needed].dropna(subset=["close", "volume"])


def h4_to_h1(regime_h4: pd.Series, h1_index: pd.DatetimeIndex) -> pd.Series:
    regime_h1 = regime_h4.reindex(h1_index, method="ffill")
    return regime_h1.fillna("RANGING_LOW_VOL")


def process_coin(coin: str) -> bool:
    try:
        # training H4 (< TRAIN_CUTOFF_DATE)
        train_path = PROC_DIR / f"{coin}_clean.parquet"
        if not train_path.exists():
            logger.warning(f"[{coin}] Training processed tidak ada: {train_path}")
            return False

        df_train = load_h4_from_processed(train_path, end=TRAIN_CUTOFF_DATE)
        if len(df_train) < 200:
            logger.warning(f"[{coin}] Training H4 terlalu sedikit ({len(df_train)}), skip")
            return False

        # Fit HMM pada training data
        model, _, state_map = fit_hmm(df_train, n_states=HMM_N_STATES, n_iter=HMM_N_ITER)
        logger.info(f"[{coin}] HMM fitted: {len(df_train)} H4 bars < {TRAIN_CUTOFF_DATE.date()}")

        # Simpan model untuk dipakai production real-time prediction
        model_path = HMM_MODEL_DIR / f"{coin}_hmm.pkl"
        joblib.dump({"model": model, "state_map": state_map, "n_states": HMM_N_STATES}, model_path)
        logger.info(f"[{coin}] Model saved -> {model_path.name}")

        # holdout H4 (>= TRAIN_CUTOFF_DATE)
        holdout_path = HOLDOUT_DIR / "processed" / f"{coin}_clean.parquet"
        if not holdout_path.exists():
            logger.warning(f"[{coin}] Holdout processed tidak ada: {holdout_path}")
            return False

        df_holdout = load_h4_from_processed(holdout_path, start=TRAIN_CUTOFF_DATE)
        if len(df_holdout) < 4:
            logger.warning(f"[{coin}] Holdout H4 terlalu sedikit ({len(df_holdout)}), skip")
            return False

        # Predict regime for holdout
        h4_labels = predict_hmm(model, df_holdout, state_map)
        regime_h4 = pd.Series(h4_labels, index=df_holdout.index)

        # Forward-fill to H1 holdout grid
        holdout_feat_path = HOLDOUT_DIR / "labeled" / f"{coin}_features_v3.parquet"
        if not holdout_feat_path.exists():
            logger.warning(f"[{coin}] Holdout features_v3 tidak ada — skip H1 alignment")
            return False

        df_feat = pd.read_parquet(holdout_feat_path, columns=[])
        h1_index = df_feat.index
        regime_h1 = h4_to_h1(regime_h4, h1_index)
        regime_enc = encode_regime(regime_h1, n_states=HMM_N_STATES)

        out = pd.DataFrame({
            "hmm_regime":     regime_h1,
            "hmm_regime_enc": regime_enc,
        }, index=h1_index)

        out_path = OUT_DIR / f"{coin}_regime_h1.parquet"
        out.to_parquet(out_path)

        dist = regime_h1.value_counts().to_dict()
        logger.info(f"[{coin}] Saved -> {out_path.name} | {len(out)} H1 bars | dist={dist}")
        return True

    except Exception as e:
        import traceback
        logger.error(f"[{coin}] Error: {e}\n{traceback.format_exc()}")
        return False


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--coins", nargs="+", default=None)
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    coins = args.coins if args.coins else (ALL_COINS if args.all else ALL_COINS)

    logger.info(f"Generating holdout HMM regime for {len(coins)} coins | cutoff={TRAIN_CUTOFF_DATE.date()}")
    ok, fail = [], []
    for coin in coins:
        if process_coin(coin):
            ok.append(coin)
        else:
            fail.append(coin)

    logger.info(f"Done: {len(ok)} ok, {len(fail)} fail")
    if fail:
        logger.warning(f"Failed: {fail}")
