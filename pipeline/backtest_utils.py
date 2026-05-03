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
    H4_BINARY_THRESHOLD_LONG, H4_BINARY_THRESHOLD_SHORT,
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
    bias_dir  = np.full(n, 1, dtype=np.int64)  # default FLAT
    bias_dir[h4_proba[:, 1] >= H4_BINARY_THRESHOLD_LONG]  = 2  # LONG bias
    bias_dir[h4_proba[:, 0] >= H4_BINARY_THRESHOLD_SHORT] = 0  # SHORT bias
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
    Hierarchical cascade decision:
      1. H4 LGBM → bias
      2. H1 LGBM → entry probability
      3. LSTM     → soft proportional confidence adjustment
      4. Decision layer → final signal

    LSTM soft adjustment (replaces hard veto):
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

    # STEP 4: Decision layer with soft adjustment
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

        if bias == 1:
            continue  # H4 FLAT → skip

        _pass_rate["h4"] += 1  # H4 gate passed

        # Pilih threshold dan confidence sesuai bias
        if bias == 2:
            h1_conf = h1_long_conf
            h1_thr  = H1_THRESHOLD_LONG
        else:
            h1_conf = h1_short_conf
            h1_thr  = H1_THRESHOLD_SHORT

        if h1_conf < h1_thr:
            continue  # H1 threshold not met

        _pass_rate["h1"] += 1  # H1 gate passed

        # LSTM soft adjustment
        if lstm_proba is not None:
            lstm_dir = int(np.argmax(lstm_proba[i]))
            adj = _lstm_adjustment(h1_conf, lstm_dir, bias)
            adjusted_conf = np.clip(h1_conf + adj, 0.0, 1.0)
        else:
            adjusted_conf = h1_conf

        if adjusted_conf >= h1_thr:
            _pass_rate["lstm"] += 1  # LSTM gate passed
            y_pred[i]     = bias
            confidence[i] = adjusted_conf
        # else: remain FLAT

    # Log pass rate summary
    if n > 0:
        logger.info(
            f"[pass_rate] H4={_pass_rate['h4']}/{n} "
            f"({_pass_rate['h4']/n:.1%}) → "
            f"H1={_pass_rate['h1']}/{_pass_rate['h4']} "
            f"({_pass_rate['h1']/_pass_rate['h4']:.1%} of H4_pass) → "
            f"LSTM={_pass_rate['lstm']}/{_pass_rate['h1']} "
            f"({_pass_rate['lstm']/_pass_rate['h1']:.1%} of H1_pass) → "
            f"FINAL={_pass_rate['lstm']}/{n} "
            f"({_pass_rate['lstm']/n:.1%} total)"
        )

    return y_pred, confidence
