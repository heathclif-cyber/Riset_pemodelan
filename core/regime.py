"""
core/regime.py — HMM Regime Detection (Walk-Forward Safe)

Fit GaussianHMM per fold, align states via return-sorting untuk
konsistensi lintas fold dan koin.

State canonical ordering (by mean_return ascending):
  0 → TRENDING_DOWN
  1 → RANGING_LOW_VOL
  2 → RANGING_HIGH_VOL
  3 → TRENDING_UP

Digunakan oleh:
  pipeline/11_regime_hmm.py  — generate OOF regime labels
  pipeline/12_train_per_regime.py — per-regime LGBM training
"""

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM

from core.utils import setup_logger

logger = setup_logger("regime")

# ─── Constants ────────────────────────────────────────────────────────────────

# Canonical regime names, ordered by mean_return ascending
REGIME_NAMES_4 = ["TRENDING_DOWN", "RANGING_LOW_VOL", "RANGING_HIGH_VOL", "TRENDING_UP"]
REGIME_NAMES_3 = ["TRENDING_DOWN", "RANGING", "TRENDING_UP"]
REGIME_NAMES_2 = ["BEAR", "BULL"]

# Integer encoding (untuk LGBM categorical)
REGIME_ENC_4 = {name: i for i, name in enumerate(REGIME_NAMES_4)}
REGIME_ENC_3 = {name: i for i, name in enumerate(REGIME_NAMES_3)}


# ─── Feature Builder ──────────────────────────────────────────────────────────

def _build_hmm_features(
    df_h4: pd.DataFrame,
    btc_h4: pd.DataFrame | None = None,
) -> np.ndarray:
    """
    Build feature matrix untuk HMM dari H4 OHLCV.

    Features (base):
      0: return_1bar     — pct_change close-to-close
      1: volatility_24   — rolling 24-bar std of returns
      2: momentum_48     — close / rolling_mean_48 - 1
      3: volume_ratio    — log(vol / rolling_mean_48)

    Features tambahan jika btc_h4 diberikan:
      4: btc_ret_h4      — BTC pct_change, di-align ke index df_h4
      5: btc_mom_48      — BTC momentum 48-bar

    Semua NaN di-fill 0 — aman untuk HMM.
    """
    ret = df_h4["close"].pct_change().fillna(0.0)
    vol = ret.rolling(24, min_periods=4).std().fillna(ret.std())
    mom = (df_h4["close"] / df_h4["close"].rolling(48, min_periods=8).mean() - 1).fillna(0.0)
    vr  = (df_h4["volume"] / df_h4["volume"].rolling(48, min_periods=8).mean()).clip(0.01, 20.0)
    lvr = np.log(vr).fillna(0.0)

    cols = [ret.values, vol.values, mom.values, lvr.values]

    if btc_h4 is not None:
        btc_ret = btc_h4["close"].pct_change().fillna(0.0)
        btc_mom = (btc_h4["close"] / btc_h4["close"].rolling(48, min_periods=8).mean() - 1).fillna(0.0)
        # align ke index coin — ffill karena BTC dan altcoin pakai index yang sama (H4 UTC)
        br = btc_ret.reindex(df_h4.index).ffill().fillna(0.0)
        bm = btc_mom.reindex(df_h4.index).ffill().fillna(0.0)
        cols.extend([br.values, bm.values])

    X = np.column_stack(cols)
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)


# ─── State Alignment ──────────────────────────────────────────────────────────

def _align_states(model: GaussianHMM, n_states: int) -> dict:
    """
    Map raw HMM state index → canonical regime name.

    Strategy: sort states by mean_return (feature index 0) ascending.
    Lowest return → TRENDING_DOWN, highest → TRENDING_UP.
    Middle states sorted by volatility (feature index 1).
    """
    means = model.means_          # shape (n_states, n_features)
    mean_ret = means[:, 0]        # return feature
    mean_vol = means[:, 1]        # volatility feature

    if n_states == 4:
        # Sort all by return first
        sorted_by_ret = np.argsort(mean_ret)  # ascending
        names = REGIME_NAMES_4
    elif n_states == 3:
        sorted_by_ret = np.argsort(mean_ret)
        names = REGIME_NAMES_3
    elif n_states == 2:
        sorted_by_ret = np.argsort(mean_ret)
        names = REGIME_NAMES_2
    else:
        sorted_by_ret = np.argsort(mean_ret)
        names = [f"STATE_{i}" for i in range(n_states)]

    mapping = {}
    for rank, state_idx in enumerate(sorted_by_ret):
        name = names[rank] if rank < len(names) else f"STATE_{rank}"
        mapping[int(state_idx)] = name

    return mapping


# ─── Core Fit/Predict ─────────────────────────────────────────────────────────

def fit_hmm(
    df_h4: pd.DataFrame,
    n_states: int = 4,
    n_iter: int = 100,
    random_state: int = 42,
    btc_h4: pd.DataFrame | None = None,
) -> tuple:
    """
    Fit GaussianHMM on df_h4.

    Returns:
      (model, labels_array, state_map)
      labels_array: np.ndarray of regime name strings (same len as df_h4)
      state_map: dict[int → str] — raw state idx → canonical name
    """
    X = _build_hmm_features(df_h4, btc_h4=btc_h4)

    model = GaussianHMM(
        n_components=n_states,
        covariance_type="diag",
        n_iter=n_iter,
        random_state=random_state,
        tol=1e-3,
        verbose=False,
    )
    model.fit(X)

    raw_labels = model.predict(X)
    state_map  = _align_states(model, n_states)
    labels     = np.array([state_map[int(s)] for s in raw_labels])

    return model, labels, state_map


def predict_hmm(
    model: GaussianHMM,
    df_h4: pd.DataFrame,
    state_map: dict,
    btc_h4: pd.DataFrame | None = None,
) -> np.ndarray:
    """Predict regime for new H4 bars using a previously fitted model."""
    X = _build_hmm_features(df_h4, btc_h4=btc_h4)
    raw_labels = model.predict(X)
    return np.array([state_map[int(s)] for s in raw_labels])


# ─── Walk-Forward OOF Generator ───────────────────────────────────────────────

def generate_oof_regime_labels(
    df_h4: pd.DataFrame,
    n_states: int = 4,
    n_folds: int = 8,
    purge: int = 6,       # H4 bars to purge between train/val (6 × 4h = 24h)
    n_iter: int = 100,
    random_state: int = 42,
    btc_h4: pd.DataFrame | None = None,
) -> pd.Series:
    """
    Generate walk-forward OOF regime labels — leak-free.

    Untuk setiap fold:
      1. Fit HMM hanya pada train portion
      2. Predict regime pada val portion saja
      3. Stitch val predictions menjadi final series

    Fold pertama train portion diprediksi oleh model yang di-fit
    pada fold 1 training data (satu-satunya cara yang leak-free
    untuk menutup "head" yang tidak punya train sebelumnya).

    Returns:
      pd.Series dengan index = df_h4.index, values = regime strings
    """
    n    = len(df_h4)
    fallback = REGIME_NAMES_4[1] if n_states == 4 else REGIME_NAMES_3[1]  # RANGING_*

    labels_out = np.full(n, "", dtype=object)

    # Walking folds — expanding window
    fold_size = max(n // (n_folds + 1), 50)

    for fold in range(n_folds):
        train_end  = (fold + 1) * fold_size
        val_start  = train_end + purge
        val_end    = min(val_start + fold_size, n)

        if val_start >= n or train_end < 100:
            continue
        if val_start >= val_end:
            continue

        df_train = df_h4.iloc[:train_end]
        df_val   = df_h4.iloc[val_start:val_end]

        try:
            model, _, state_map = fit_hmm(df_train, n_states, n_iter, random_state, btc_h4=btc_h4)
            val_labels = predict_hmm(model, df_val, state_map, btc_h4=btc_h4)
            labels_out[val_start:val_end] = val_labels

            dist = dict(pd.Series(val_labels).value_counts())
            logger.info(
                f"  Fold {fold+1}: train={train_end}, val=[{val_start}:{val_end}] "
                f"| dist={dist}"
            )
        except Exception as e:
            logger.warning(f"  Fold {fold+1}: HMM failed — {e}, using fallback")
            labels_out[val_start:val_end] = fallback

    # ── Fill "head" (bars before first val_start) ─────────────────────────────
    # Fit on first fold_size bars and predict them
    head_end = fold_size + purge  # end of first val start
    empty_mask = labels_out == ""
    if empty_mask[:head_end].any():
        try:
            df_head = df_h4.iloc[:fold_size]
            model_h, _, state_map_h = fit_hmm(df_head, n_states, n_iter, random_state, btc_h4=btc_h4)
            # Predict the full head range using this model
            head_pred = predict_hmm(model_h, df_h4.iloc[:head_end], state_map_h, btc_h4=btc_h4)
            for i in range(head_end):
                if labels_out[i] == "":
                    labels_out[i] = head_pred[i]
        except Exception as e:
            logger.warning(f"  Head fill failed — {e}")
            labels_out[:head_end][labels_out[:head_end] == ""] = fallback

    # ── Fill any remaining gaps (forward-fill then fallback) ──────────────────
    s = pd.Series(labels_out, index=df_h4.index)
    s = s.replace("", np.nan).ffill().fillna(fallback)   # no bfill: ffill-only prevents look-ahead

    # Distribution log
    dist_final = s.value_counts(normalize=True).round(3).to_dict()
    logger.info(f"  Final regime distribution (H4): {dist_final}")

    return s


# ─── Encoding Helper ─────────────────────────────────────────────────────────

def encode_regime(regime_series: pd.Series, n_states: int = 4) -> pd.Series:
    """
    Convert regime string labels → integer encoding (untuk LGBM).

    TRENDING_DOWN=0, RANGING_LOW_VOL=1, RANGING_HIGH_VOL=2, TRENDING_UP=3
    """
    enc = REGIME_ENC_4 if n_states == 4 else REGIME_ENC_3
    return regime_series.map(enc).fillna(1).astype(np.int32)
