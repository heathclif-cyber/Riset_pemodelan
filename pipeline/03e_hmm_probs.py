"""
pipeline/03e_hmm_probs.py — Generate HMM State Probabilities (bukan argmax)

GaussianHMM 4-state walk-forward OOF → 4 probability columns per H1 bar.
Simpan ke: {coin}_hmm_probs.parquet (training + holdout)

Usage:
  python pipeline/03e_hmm_probs.py --all
  python pipeline/03e_hmm_probs.py --all --holdout
"""

import argparse, sys, numpy as np, pandas as pd
from pathlib import Path

ROOT = Path(__file__).parent.parent; sys.path.insert(0, str(ROOT))

from config import (
    TRAINING_COINS, LABEL_DIR, HOLDOUT_DIR, TRAIN_CUTOFF_DATE,
    HMM_N_STATES, HMM_N_FOLDS, HMM_PURGE_H4, HMM_N_ITER, REGIME_NAMES,
)
from core.utils import setup_logger
from core.regime import _build_hmm_features, _align_states

logger = setup_logger("03e_hmm_probs")


def fit_hmm_probs(df_h4, n_states=4, n_iter=100, random_state=42):
    """Fit HMM and return ALIGNED state probabilities (not just argmax)."""
    from hmmlearn.hmm import GaussianHMM
    X = _build_hmm_features(df_h4)

    model = GaussianHMM(
        n_components=n_states, covariance_type="diag",
        n_iter=n_iter, random_state=random_state, tol=1e-3, verbose=False,
    )
    model.fit(X)

    # State alignment
    state_map = _align_states(model, n_states)
    # state_map: raw_idx → canonical_name
    # Reverse: canonical_name → raw_idx
    raw_to_canon = state_map  # {raw: name}
    canon_to_raw = {name: raw for raw, name in state_map.items()}

    # Get posterior probabilities: shape (n_samples, n_states)
    probs_raw = model.predict_proba(X)  # columns: raw state order

    # Reorder to canonical: 0=TRENDING_DOWN, 1=RANGING_LOW_VOL, 2=RANGING_HIGH_VOL, 3=TRENDING_UP
    probs_canon = np.zeros((len(probs_raw), n_states), dtype=np.float64)
    for name, raw_idx in canon_to_raw.items():
        canon_idx = REGIME_NAMES.index(name) if name in REGIME_NAMES else -1
        if canon_idx >= 0:
            probs_canon[:, canon_idx] = probs_raw[:, raw_idx]

    return probs_canon, df_h4.index


def h4_to_h1(probs_h4, h4_index, h1_index):
    """Forward-fill H4 probabilities to H1 grid."""
    df_probs = pd.DataFrame(probs_h4, index=h4_index,
                            columns=[f"hmm_prob_{i}" for i in range(probs_h4.shape[1])])
    # Reindex to H1, forward fill
    df_probs = df_probs.reindex(h1_index, method="ffill").fillna(0.25)  # fallback: uniform
    return df_probs


def process_coin_walkforward(coin, is_holdout):
    """Walk-forward HMM: fit per fold, generate OOF probabilities."""
    if is_holdout:
        proc_path = HOLDOUT_DIR / "processed" / f"{coin}_clean.parquet"
    else:
        from config import PROC_DIR
        proc_path = PROC_DIR / f"{coin}_clean.parquet"
    h1_label_path = (HOLDOUT_DIR / "labeled" / f"{coin}_features_v3.parquet"
                     if is_holdout else LABEL_DIR / f"{coin}_features_v3.parquet")

    if not proc_path.exists():
        logger.warning(f"{coin}: processed file not found at {proc_path}")
        return None
    if not h1_label_path.exists():
        logger.warning(f"{coin}: labeled H1 file not found")
        return None

    # Load H1 index
    h1_df = pd.read_parquet(h1_label_path).sort_index()
    h1_idx = h1_df.index

    # Build H4 from processed (which has H4 columns)
    proc = pd.read_parquet(proc_path).sort_index()
    # Extract H4 bars: 4h_open/high/low/close/volume
    h4_cols = [c for c in proc.columns if c.startswith("4h_")]
    if not h4_cols:
        logger.warning(f"{coin}: no H4 columns in processed")
        return None

    # Build H4 DataFrame
    h4_data = {}
    for col in ["open", "high", "low", "close", "volume"]:
        h4c = f"4h_{col}"
        if h4c in proc.columns:
            h4_data[col] = proc[h4c].dropna()
    if not h4_data:
        return None
    df_h4 = pd.DataFrame(h4_data).sort_index()
    if len(df_h4) < 50:
        return None

    # Cut training portion if needed
    if not is_holdout:
        df_h4 = df_h4[df_h4.index < TRAIN_CUTOFF_DATE]

    n_h4 = len(df_h4)
    fold_size = n_h4 // HMM_N_FOLDS

    # Allocate output
    all_probs = np.zeros((n_h4, HMM_N_STATES), dtype=np.float64)

    for fold in range(HMM_N_FOLDS):
        test_start = fold * fold_size
        test_end = (fold + 1) * fold_size if fold < HMM_N_FOLDS - 1 else n_h4

        if test_end - test_start < 10:
            continue

        # Train on all data up to test_start (with purge)
        train_end = max(0, test_start - HMM_PURGE_H4)
        if train_end < 20:
            continue

        train_df = df_h4.iloc[:train_end]
        test_df = df_h4.iloc[test_start:test_end]

        try:
            probs_test, _ = fit_hmm_probs(train_df, n_states=HMM_N_STATES, n_iter=HMM_N_ITER)
            # Predict on test set using the trained model...
            # Actually we need to use the fitted model on test data
            from hmmlearn.hmm import GaussianHMM
            X_train = _build_hmm_features(train_df)
            X_test = _build_hmm_features(test_df)

            model = GaussianHMM(
                n_components=HMM_N_STATES, covariance_type="diag",
                n_iter=HMM_N_ITER, random_state=42, tol=1e-3, verbose=False,
            )
            model.fit(X_train)

            # State alignment
            state_map = _align_states(model, HMM_N_STATES)
            canon_to_raw = {name: raw for raw, name in state_map.items()}

            # Predict probabilities on test
            probs_raw_test = model.predict_proba(X_test)
            probs_canon_test = np.zeros((len(probs_raw_test), HMM_N_STATES), dtype=np.float64)
            for name, raw_idx in canon_to_raw.items():
                canon_idx = REGIME_NAMES.index(name) if name in REGIME_NAMES else -1
                if canon_idx >= 0:
                    probs_canon_test[:, canon_idx] = probs_raw_test[:, raw_idx]

            all_probs[test_start:test_end] = probs_canon_test
        except Exception as e:
            logger.warning(f"{coin} fold {fold}: HMM fit failed: {e}")
            all_probs[test_start:test_end] = 0.25

    # Forward-fill to H1
    df_h1_probs = h4_to_h1(all_probs, df_h4.index, h1_idx)

    # Save
    out_dir = HOLDOUT_DIR / "labeled" if is_holdout else LABEL_DIR
    out_path = out_dir / f"{coin}_hmm_probs.parquet"
    df_h1_probs.to_parquet(out_path)

    means = df_h1_probs.mean()
    logger.info(f"{coin}: {len(df_h1_probs)} bars | "
                f"P0={means['hmm_prob_0']:.2f} P1={means['hmm_prob_1']:.2f} "
                f"P2={means['hmm_prob_2']:.2f} P3={means['hmm_prob_3']:.2f}")
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--holdout", action="store_true")
    parser.add_argument("--coins", nargs="+", default=None)
    args = parser.parse_args()

    coins = args.coins or (TRAINING_COINS if args.all else TRAINING_COINS[:5])
    tag = "holdout" if args.holdout else "training"

    print(f"\n{'='*60}")
    print(f"  HMM STATE PROBABILITIES | {len(coins)} coins | {tag}")
    print(f"  Output: 4 probability columns (TRENDING_DOWN/LOW_VOL/HIGH_VOL/UP)")
    print(f"  Walk-forward OOF, {HMM_N_FOLDS} folds, {HMM_PURGE_H4}-bar purge")
    print(f"{'='*60}\n")

    for coin in coins:
        process_coin_walkforward(coin, is_holdout=args.holdout)

    print(f"\nDone! Files saved to {'holdout' if args.holdout else 'training'}/labeled/{{coin}}_hmm_probs.parquet")


if __name__ == "__main__":
    main()
