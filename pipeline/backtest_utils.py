"""
pipeline/backtest_utils.py — Shared 2-Model Cascade Logic

Berisi hierarchical_predict() dan helper functions yang dipakai bersama oleh:
  - pipeline/08_backtest.py
  - pipeline/09_holdout_backtest.py

Cascade flow (arsitektur 2 model):
  STEP 1: LGBM  → entry signal + confidence (primary)
  STEP 2: LSTM  → soft confidence adjustment (confirmation)
  STEP 3: Decision layer → final signal

H4 LGBM dihapus dari cascade. Regime context (H4 trend, D1 alignment,
trend quality) sudah embedded langsung sebagai fitur di LGBM.
"""

import logging
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

logger = logging.getLogger("backtest_utils")

from config import (
    NUM_CLASSES,
    LGBM_THRESHOLD_LONG, LGBM_THRESHOLD_SHORT,
    LSTM_CONFIRMATION_ENABLED,
    LSTM_ADJUST_MODE,
    LSTM_ADJUST_AGREE_BOOST, LSTM_ADJUST_NEUTRAL_PEN, LSTM_ADJUST_OPPOSITE_PEN,
)
from pipeline.p05_utils import SequenceDataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─── Pass Rate Counter ─────────────────────────────────────────────────────────
_pass_rate = {"lgbm": 0, "lstm": 0, "total": 0}


def _lstm_adjustment(h1_conf: float, lstm_dir: int, bias: int) -> float:
    """
    Hitung LSTM confidence adjustment berdasarkan mode yang dikonfigurasi.

    Mode "relative" (original):  adj = {agree/neutral/opposite} × h1_conf
    Mode "absolute" (fixed):     adj = fixed value terlepas dari h1_conf
    Mode "tiered":               adj bervariasi berdasarkan margin di atas threshold

    Returns: adjustment value (float), applied as: adjusted = clip(h1_conf + adj, 0, 1)
    """
    if lstm_dir == bias:       # agree → boost
        base = LSTM_ADJUST_AGREE_BOOST
        return base * (1.0 - h1_conf) if LSTM_ADJUST_MODE == "relative" else base
    elif lstm_dir == 1:        # neutral (FLAT) → slight reduce
        pen  = LSTM_ADJUST_NEUTRAL_PEN
        return -pen * h1_conf if LSTM_ADJUST_MODE == "relative" else -pen
    else:                      # opposite → strong reduce
        pen  = LSTM_ADJUST_OPPOSITE_PEN
        if LSTM_ADJUST_MODE == "tiered":
            # Penalti lebih ringan jika margin besar (confident)
            margin = h1_conf - 0.62  # threshold reference
            if margin < 0.05:
                return -pen * 1.5        # borderline → heavy
            elif margin < 0.10:
                return -pen * 1.0        # moderate → medium
            else:
                return -pen * 0.5        # confident → light
        return -pen * h1_conf if LSTM_ADJUST_MODE == "relative" else -pen


def get_lstm_proba(
    lstm_model,
    lstm_scaler,
    X: np.ndarray,
    n_total: int,
) -> np.ndarray:
    """Run LSTM inference; pad head rows yang tidak punya full sequence."""
    X_sc   = lstm_scaler.transform(X)
    dummy  = np.zeros(len(X_sc), dtype=np.int64)
    ds     = SequenceDataset(X_sc, dummy)
    loader = DataLoader(ds, batch_size=1024, shuffle=False)

    lstm_list = []
    lstm_model.eval()
    with torch.no_grad():
        for xb, _ in loader:
            logits = lstm_model(xb.to(DEVICE))
            proba  = torch.softmax(logits, dim=1).cpu().numpy()
            lstm_list.append(proba)
    lstm_proba = np.vstack(lstm_list)  # shape (N - seq_len + 1, 3)

    if len(lstm_proba) < n_total:
        pad = np.ones((n_total - len(lstm_proba), NUM_CLASSES)) / NUM_CLASSES
        lstm_proba = np.vstack([pad, lstm_proba])
    return lstm_proba


def hierarchical_predict(
    h4_model,        # unused — dipertahankan untuk kompatibilitas signature
    lgbm_model,
    lstm_model,
    lstm_scaler,
    X: np.ndarray,
    feat_cols: list[str],
    h4_feat_cols: list[str],  # unused
    df_slice,
) -> tuple[np.ndarray, np.ndarray]:
    """
    2-Model cascade: LGBM (primary) → LSTM (soft confirmation).

    H4 LGBM dihapus dari decision layer. Regime context (trend acceleration,
    volume confirmation, HTF alignment, D1 trend) sudah embedded sebagai fitur
    di LGBM — model belajar interaksinya sendiri tanpa layer terpisah.

    LSTM soft adjustment (mode "tiered"):
      agree   : +LSTM_ADJUST_AGREE_BOOST
      neutral : -LSTM_ADJUST_NEUTRAL_PEN
      opposite: -LSTM_ADJUST_OPPOSITE_PEN × multiplier(margin)

    Returns:
      y_pred     : array int64 (0=SHORT, 1=FLAT, 2=LONG)
      confidence : array float64 (adjusted confidence of predicted class)
    """
    n = len(df_slice)

    # STEP 1: LGBM entry signal (primary)
    valid_cols  = [c for c in feat_cols if c in df_slice.columns]
    lgbm_proba  = lgbm_model.predict_proba(df_slice[valid_cols])
    # lgbm_proba: (N, 3) — col 0=SHORT, 1=FLAT, 2=LONG

    # STEP 2: LSTM soft adjustment
    if LSTM_CONFIRMATION_ENABLED and lstm_model is not None:
        lstm_proba = get_lstm_proba(lstm_model, lstm_scaler, X, n)
    else:
        lstm_proba = None

    _pass_rate["total"] = n
    _pass_rate["lgbm"]  = 0
    _pass_rate["lstm"]  = 0

    y_pred     = np.ones(n, dtype=np.int64)
    confidence = np.full(n, 1.0 / NUM_CLASSES)

    for i in range(n):
        lgbm_long_conf  = lgbm_proba[i, 2]
        lgbm_short_conf = lgbm_proba[i, 0]

        if lgbm_long_conf < LGBM_THRESHOLD_LONG and lgbm_short_conf < LGBM_THRESHOLD_SHORT:
            continue  # LGBM tidak yakin → FLAT

        if lgbm_long_conf >= lgbm_short_conf:
            lgbm_dir  = 2
            lgbm_conf = lgbm_long_conf
            lgbm_thr  = LGBM_THRESHOLD_LONG
        else:
            lgbm_dir  = 0
            lgbm_conf = lgbm_short_conf
            lgbm_thr  = LGBM_THRESHOLD_SHORT

        _pass_rate["lgbm"] += 1

        # STEP 3: LSTM soft adjustment
        adj_conf = lgbm_conf
        if lstm_proba is not None:
            lstm_dir = int(np.argmax(lstm_proba[i]))
            adj      = _lstm_adjustment(adj_conf, lstm_dir, lgbm_dir)
            adj_conf = float(np.clip(adj_conf + adj, 0.0, 1.0))

        if adj_conf >= lgbm_thr:
            _pass_rate["lstm"] += 1
            y_pred[i]     = lgbm_dir
            confidence[i] = adj_conf

    if n > 0:
        n_lgbm = _pass_rate["lgbm"]
        n_fin  = _pass_rate["lstm"]
        logger.info(
            f"[pass_rate] LGBM_pass={n_lgbm}/{n} ({n_lgbm/n:.1%}) → "
            f"FINAL={n_fin}/{n} ({n_fin/n:.1%})"
        )

    return y_pred, confidence
