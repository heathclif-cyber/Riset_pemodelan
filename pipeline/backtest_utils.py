"""
pipeline/backtest_utils.py — Shared Hierarchical Cascade Logic

Berisi hierarchical_predict() dan helper functions yang dipakai bersama oleh:
  - pipeline/08_backtest.py
  - pipeline/09_holdout_backtest.py

Cascade flow:
  STEP 1: H4 LGBM → bias direction (LONG/SHORT/FLAT)
  STEP 2: H1 LGBM → entry signal dengan confidence threshold
  STEP 3: LSTM    → confirmation vote
  STEP 4: Decision layer → final signal
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
    H4_BINARY_THRESHOLD_LONG, H4_BINARY_THRESHOLD_SHORT, H4_BINARY_MARGIN,
    H4_SOFT_FILTER_ENABLED, H4_SOFT_ALIGN_BOOST, H4_SOFT_MISALIGN_PENALTY,
    H1_THRESHOLD_LONG, H1_THRESHOLD_SHORT,
    LSTM_CONFIRMATION_ENABLED,
    LSTM_ADJUST_MODE,
    LSTM_ADJUST_AGREE_BOOST, LSTM_ADJUST_NEUTRAL_PEN, LSTM_ADJUST_OPPOSITE_PEN,
)
from pipeline.p05_utils import SequenceDataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─── Pass Rate Counter ─────────────────────────────────────────────────────────
# Global counter untuk tracking pass rate per layer selama backtest.
# Di-reset per panggilan hierarchical_predict().
_pass_rate = {"h4": 0, "h1": 0, "lstm": 0, "total": 0}


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


def get_h4_bias(
    h4_model,
    df_slice,
    h4_feat_cols: list[str],
) -> tuple[np.ndarray, "np.ndarray | None"]:
    """
    Hitung H4 bias per H1 bar.
    Jika h4_model None: return FLAT semua (fallback).
    Returns: (bias_dir, h4_proba) — bias_dir: 0=SHORT, 1=FLAT, 2=LONG.
    """
    n = len(df_slice)
    if h4_model is None or not h4_feat_cols:
        return np.ones(n, dtype=np.int64), None

    valid_h4_cols = [c for c in h4_feat_cols if c in df_slice.columns]
    if not valid_h4_cols:
        return np.ones(n, dtype=np.int64), None

    h4_proba = h4_model.predict_proba(df_slice[valid_h4_cols])
    # Binary model output: col 0 = prob_SHORT, col 1 = prob_LONG
    # Margin-based: bias hanya jika prob unggul >= margin atas lawan
    prob_long  = h4_proba[:, 1]
    prob_short = h4_proba[:, 0]
    
    bias_dir = np.full(n, 1, dtype=np.int64)  # default FLAT
    
    long_mask  = (prob_long  >= H4_BINARY_THRESHOLD_LONG)  & (prob_long  >= prob_short + H4_BINARY_MARGIN)
    short_mask = (prob_short >= H4_BINARY_THRESHOLD_SHORT) & (prob_short >= prob_long  + H4_BINARY_MARGIN)
    
    bias_dir[long_mask]  = 2  # LONG bias
    bias_dir[short_mask] = 0  # SHORT bias
    return bias_dir, h4_proba


def hierarchical_predict(
    h4_model,
    h1_model,
    lstm_model,
    lstm_scaler,
    X: np.ndarray,
    feat_cols: list[str],
    h4_feat_cols: list[str],
    df_slice,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Hierarchical cascade decision with H4 soft filter:
      1. H4 LGBM → bias (confidence modifier, NOT hard gate)
      2. H1 LGBM → entry probability (primary decision layer)
      3. H4 soft adjustment → boost/penalty h1_conf based on alignment
      4. LSTM     → soft proportional confidence adjustment
      5. Decision layer → final signal

    H4 soft filter (replaces hard gate):
      - H4 AUC ~0.55 (weak edge) → tidak cukup kuat untuk hard veto
      - If H4 bias aligns with H1 direction: h1_conf += H4_SOFT_ALIGN_BOOST
      - If H4 bias is FLAT or opposite:        h1_conf -= H4_SOFT_MISALIGN_PENALTY
      - H1 always gets a chance to decide (no hard reject)

    LSTM soft adjustment:
      - LSTM agree  : +0.05 × (1 - h1_conf)
      - LSTM neutral: -0.05 × h1_conf
      - LSTM oppose : -0.15 × h1_conf

    Returns:
      y_pred     : array int64 (0=SHORT, 1=FLAT, 2=LONG)
      confidence : array float64 (adjusted probability of predicted class)
    """
    n = len(df_slice)

    # STEP 1: H4 bias
    bias_dir, _ = get_h4_bias(h4_model, df_slice, h4_feat_cols)

    # STEP 2: H1 entry signal
    valid_h1_cols = [c for c in feat_cols if c in df_slice.columns]
    h1_proba = h1_model.predict_proba(df_slice[valid_h1_cols])
    # h1_proba: (N, 3) — col 0=SHORT, 1=FLAT, 2=LONG

    # STEP 3: LSTM soft adjustment
    if LSTM_CONFIRMATION_ENABLED and lstm_model is not None:
        lstm_proba = get_lstm_proba(lstm_model, lstm_scaler, X, n)
    else:
        lstm_proba = None

    # STEP 4: Decision layer with H4 soft filter + LSTM adjustment
    _pass_rate["total"] = n
    _pass_rate["h4"]    = 0
    _pass_rate["h1"]    = 0
    _pass_rate["lstm"]  = 0

    y_pred     = np.ones(n, dtype=np.int64)   # default FLAT
    confidence = np.full(n, 1.0 / NUM_CLASSES)

    for i in range(n):
        bias          = bias_dir[i]
        h1_long_conf  = h1_proba[i, 2]  # P(LONG)  dari H1 LGBM
        h1_short_conf = h1_proba[i, 0]  # P(SHORT) dari H1 LGBM

        # H1 is always the primary decision layer
        # Check both LONG and SHORT thresholds for H1
        h1_long_passed  = h1_long_conf  >= H1_THRESHOLD_LONG
        h1_short_passed = h1_short_conf >= H1_THRESHOLD_SHORT

        if not h1_long_passed and not h1_short_passed:
            continue  # H1 tidak yakin sama sekali → FLAT

        # Determine H1's preferred direction and confidence
        if h1_long_conf >= h1_short_conf:
            h1_best_dir = 2  # LONG
            h1_best_conf = h1_long_conf
            h1_thr = H1_THRESHOLD_LONG
        else:
            h1_best_dir = 0  # SHORT
            h1_best_conf = h1_short_conf
            h1_thr = H1_THRESHOLD_SHORT

        # STEP 4a: H4 soft filter — adjust H1 confidence based on H4 alignment
        h4_adjustment = 0.0
        if H4_SOFT_FILTER_ENABLED:
            if bias == h1_best_dir:
                # H4 aligns with H1 → slight boost
                h4_adjustment = H4_SOFT_ALIGN_BOOST
                _pass_rate["h4"] += 1
            elif bias == 1:
                # H4 is FLAT → slight penalty (but NOT hard reject)
                h4_adjustment = -H4_SOFT_MISALIGN_PENALTY
            else:
                # H4 opposite to H1 → slightly bigger penalty
                h4_adjustment = -H4_SOFT_MISALIGN_PENALTY

        adjusted_conf = np.clip(h1_best_conf + h4_adjustment, 0.0, 1.0)

        # STEP 4b: LSTM soft adjustment
        if lstm_proba is not None:
            lstm_dir = int(np.argmax(lstm_proba[i]))
            lstm_adj = _lstm_adjustment(adjusted_conf, lstm_dir, h1_best_dir)
            adjusted_conf = np.clip(adjusted_conf + lstm_adj, 0.0, 1.0)

        if adjusted_conf >= h1_thr:
            _pass_rate["lstm"] += 1
            y_pred[i]     = h1_best_dir
            confidence[i] = adjusted_conf
        # else: remain FLAT

    # Log pass rate summary
    if n > 0:
        n_h4_pass = _pass_rate['h4']
        n_lstm = _pass_rate['lstm']
        logger.info(
            f"[pass_rate] H4_align={n_h4_pass}/{n} ({n_h4_pass/n:.1%}) → "
            f"FINAL={n_lstm}/{n} ({n_lstm/n:.1%} total)"
        )

    return y_pred, confidence
