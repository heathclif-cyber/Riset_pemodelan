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

import joblib
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
    LGBM_FLAT_REVIEW_THRESHOLD,
    LSTM_CONFIRMATION_ENABLED,
    LSTM_FLAT_REVIEW_ENABLED,
    LSTM_ADJUST_MODE,
    LSTM_ADJUST_AGREE_BOOST, LSTM_ADJUST_NEUTRAL_PEN, LSTM_ADJUST_OPPOSITE_PEN,
    LSTM_TIERED_MULTIPLIERS,
    LSTM_OVERRIDE_THRESHOLD,
    REGIME_NAMES, MODEL_DIR,
    TREND_ALIGNMENT_ENABLED, WITH_TREND_PENALTY, COUNTER_TREND_BOOST,
    WITH_TREND_BLOCK_CONF,
)
from pipeline.shared import SequenceDataset
from core.utils import get_lstm_device

DEVICE = torch.device("cpu")  # LSTM inference via CPU (hindari DirectML device mismatch)

# ─── Regime Model Registry ────────────────────────────────────────────────────
# Cache per-regime LGBM models (loaded lazily on first use)
_regime_models: dict = {}   # regime_name → lgbm model | None
_regime_models_loaded = False


def load_regime_models(model_dir: Path = MODEL_DIR) -> dict:
    """
    Load semua per-regime LGBM models dari models/lgbm_regime_*.pkl.
    Return dict: {regime_name: model_or_None}.
    Dipanggil sekali saat pertama kali hierarchical_predict() dijalankan.
    """
    global _regime_models, _regime_models_loaded
    if _regime_models_loaded:
        return _regime_models

    for rname in REGIME_NAMES:
        safe = rname.replace(" ", "_")
        path = model_dir / f"lgbm_regime_{safe}.pkl"
        if path.exists():
            try:
                _regime_models[rname] = joblib.load(path)
                logger.info(f"[regime] Loaded: {path.name}")
            except Exception as e:
                logger.warning(f"[regime] Gagal load {path.name}: {e}")
                _regime_models[rname] = None
        else:
            _regime_models[rname] = None

    loaded = [k for k, v in _regime_models.items() if v is not None]
    logger.info(f"[regime] Per-regime models loaded: {loaded}")
    _regime_models_loaded = True
    return _regime_models

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
            # Multipliers dari config: [borderline, moderate, confident]
            margin = h1_conf - 0.62  # threshold reference
            mul = LSTM_TIERED_MULTIPLIERS
            if margin < 0.05:
                return -pen * mul[0]       # borderline
            elif margin < 0.10:
                return -pen * mul[1]       # moderate
            else:
                return -pen * mul[2]       # confident
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
    # ── Trend Alignment (Grup 2) ──────────────────────────────────────────────
    trend_alignment_enabled: bool = TREND_ALIGNMENT_ENABLED,
    with_trend_penalty:      float = WITH_TREND_PENALTY,   # 2a
    counter_trend_boost:     float = COUNTER_TREND_BOOST,  # 2b
    with_trend_block_conf:   float = WITH_TREND_BLOCK_CONF, # 2c (0 = disable)
) -> tuple[np.ndarray, np.ndarray]:
    """
    2-Model cascade: LGBM (primary) → LSTM (soft confirmation / FLAT review)
    → Trend Alignment (Grup 2).

    H4 LGBM dihapus dari decision layer. Regime context (trend acceleration,
    volume confirmation, HTF alignment, D1 trend) sudah embedded sebagai fitur
    di LGBM — model belajar interaksinya sendiri tanpa layer terpisah.

    Alur:
      LGBM predict
        ├─ LONG  & conf ≥ LGBM_THRESHOLD_LONG  → LSTM adjustment     (seperti semula)
        ├─ SHORT & conf ≥ LGBM_THRESHOLD_SHORT → LSTM adjustment     (seperti semula)
        ├─ FLAT  & max_conf < FLAT_REVIEW_THR  → LSTM review (BARU)  ← fix FLAT bias
        └─ FLAT  & max_conf ≥ FLAT_REVIEW_THR  → langsung FLAT       (hemat komputasi)

    LSTM soft adjustment (mode "tiered"):
      agree   : +LSTM_ADJUST_AGREE_BOOST
      neutral : -LSTM_ADJUST_NEUTRAL_PEN
      opposite: -LSTM_ADJUST_OPPOSITE_PEN × multiplier(margin)

    LSTM FLAT review:
      LSTM LONG  & conf ≥ LGBM_THRESHOLD_LONG  → override ke LONG
      LSTM SHORT & conf ≥ LGBM_THRESHOLD_SHORT → override ke SHORT
      LSTM FLAT  atau conf tidak cukup         → tetap FLAT

    Trend Alignment (Grup 2) — applied AFTER LSTM adjustment:
      with_trend    (signal searah H4 trend)   → -with_trend_penalty
      counter_trend (signal lawan H4 trend)    → +counter_trend_boost
      block if with_trend & conf < with_trend_block_conf (2c)

    Trend determined by h4_trend column in df_slice:
      h4_trend > 0 → UP, h4_trend < 0 → DOWN, else RANGING

    Returns:
      y_pred     : array int64 (0=SHORT, 1=FLAT, 2=LONG)
      confidence : array float64 (adjusted confidence of predicted class)
    """
    n = len(df_slice)

    # ── Load regime models (lazy, first call only) ────────────────────────────
    regime_models = load_regime_models()
    use_per_regime = any(v is not None for v in regime_models.values())
    if use_per_regime:
        regime_col = "hmm_regime" if "hmm_regime" in df_slice.columns else None
        logger.info(f"[cascade] Regime-aware mode: regime col={'found' if regime_col else 'MISSING'}")
    else:
        regime_col = None
        logger.info("[cascade] Standard mode (no per-regime models)")

    # STEP 1: LGBM entry signal (primary)
    valid_cols  = [c for c in feat_cols if c in df_slice.columns]

    # Jika per-regime: kita perlu proba per baris menggunakan model yang sesuai
    if use_per_regime and regime_col is not None:
        # Batch predict per regime
        lgbm_proba = np.full((n, NUM_CLASSES), 1.0 / NUM_CLASSES)
        regimes    = df_slice[regime_col].fillna("RANGING_LOW_VOL").values

        for rname, rmodel in regime_models.items():
            if rmodel is None:
                # Fallback ke lgbm_model global
                rmodel = lgbm_model
            mask = regimes == rname
            if mask.sum() == 0:
                continue
            gbm_feats = rmodel.feature_name_
            X_pred = np.zeros((n, len(gbm_feats)), dtype=np.float64)
            for idx, col in enumerate(gbm_feats):
                if col in df_slice.columns:
                    X_pred[:, idx] = df_slice[col].values.astype(np.float64)
            lgbm_proba[mask] = rmodel.predict_proba(X_pred[mask])
    else:
        # Standard: satu model untuk semua bar
        # Align feature order ke model, pad missing columns dengan nol
        gbm_feats = lgbm_model.feature_name_
        X_pred = np.zeros((n, len(gbm_feats)), dtype=np.float64)
        for idx, col in enumerate(gbm_feats):
            if col in df_slice.columns:
                X_pred[:, idx] = df_slice[col].values.astype(np.float64)
        lgbm_proba = lgbm_model.predict_proba(X_pred)
    # lgbm_proba: (N, 3) — col 0=SHORT, 1=FLAT, 2=LONG

    # STEP 2: LSTM probabilities (jika enabled)
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

        # ── LGBM output FLAT (tidak melewati threshold entry) ─────────────────
        if lgbm_long_conf < LGBM_THRESHOLD_LONG and lgbm_short_conf < LGBM_THRESHOLD_SHORT:
            # Hitung max confidence LGBM dari semua kelas
            lgbm_max_conf = float(np.max(lgbm_proba[i]))

            # FLAT review: jika LGBM ragu (max_conf < threshold), beri LSTM kesempatan
            # Dapat di-disable via LSTM_FLAT_REVIEW_ENABLED (default False — WR 78.8% vs 57.9%)
            if LSTM_FLAT_REVIEW_ENABLED and lstm_proba is not None and lgbm_max_conf < LGBM_FLAT_REVIEW_THRESHOLD:
                lstm_dir       = int(np.argmax(lstm_proba[i]))
                lstm_conf_long  = lstm_proba[i, 2]
                lstm_conf_short = lstm_proba[i, 0]

                if lstm_dir == 2 and lstm_conf_long >= LSTM_OVERRIDE_THRESHOLD:
                    # LSTM yakin LONG — override FLAT (pakai threshold khusus override)
                    override_conf = lstm_conf_long
                    # Trend alignment for FLAT-review path (lstm_dir == LONG)
                    if trend_alignment_enabled and "h4_trend" in df_slice.columns:
                        trend_val = df_slice["h4_trend"].iloc[i]
                        if not np.isnan(trend_val):
                            if trend_val > 0:  # H4 UP → LONG is with-trend
                                override_conf -= with_trend_penalty
                            elif trend_val < 0:  # H4 DOWN → LONG is counter-trend
                                override_conf += counter_trend_boost
                            override_conf = float(np.clip(override_conf, 0.0, 1.0))
                            if trend_val > 0 and with_trend_block_conf > 0 and override_conf < with_trend_block_conf:
                                continue
                    if override_conf >= LSTM_OVERRIDE_THRESHOLD:
                        _pass_rate["lgbm"] += 1
                        _pass_rate["lstm"] += 1
                        y_pred[i]     = 2
                        confidence[i] = override_conf
                elif lstm_dir == 0 and lstm_conf_short >= LSTM_OVERRIDE_THRESHOLD:
                    # LSTM yakin SHORT — override FLAT
                    override_conf = lstm_conf_short
                    if trend_alignment_enabled and "h4_trend" in df_slice.columns:
                        trend_val = df_slice["h4_trend"].iloc[i]
                        if not np.isnan(trend_val):
                            if trend_val < 0:  # H4 DOWN → SHORT is with-trend
                                override_conf -= with_trend_penalty
                            elif trend_val > 0:  # H4 UP → SHORT is counter-trend
                                override_conf += counter_trend_boost
                            override_conf = float(np.clip(override_conf, 0.0, 1.0))
                            if trend_val < 0 and with_trend_block_conf > 0 and override_conf < with_trend_block_conf:
                                continue
                    if override_conf >= LSTM_OVERRIDE_THRESHOLD:
                        _pass_rate["lgbm"] += 1
                        _pass_rate["lstm"] += 1
                        y_pred[i]     = 0
                        confidence[i] = override_conf
                # else: LSTM FLAT atau confidence tidak cukup → tetap FLAT (continue)

            continue  # FLAT (baik high-confidence maupun LSTM gagal confirm)

        # ── LGBM output LONG atau SHORT ───────────────────────────────────────
        if lgbm_long_conf >= lgbm_short_conf:
            lgbm_dir  = 2
            lgbm_conf = lgbm_long_conf
            lgbm_thr  = LGBM_THRESHOLD_LONG
        else:
            lgbm_dir  = 0
            lgbm_conf = lgbm_short_conf
            lgbm_thr  = LGBM_THRESHOLD_SHORT

        _pass_rate["lgbm"] += 1

        # STEP 3: LSTM soft adjustment (untuk LONG/SHORT path)
        adj_conf = lgbm_conf
        if lstm_proba is not None:
            lstm_dir = int(np.argmax(lstm_proba[i]))
            adj      = _lstm_adjustment(adj_conf, lstm_dir, lgbm_dir)
            adj_conf = float(np.clip(adj_conf + adj, 0.0, 1.0))

        # STEP 4: Trend Alignment (Grup 2) — after LSTM adjustment
        if trend_alignment_enabled:
            h4_trend_col = "h4_trend"
            if h4_trend_col in df_slice.columns:
                trend_val = df_slice[h4_trend_col].iloc[i]
                if not np.isnan(trend_val):
                    is_with_trend = (lgbm_dir == 2 and trend_val > 0) or (lgbm_dir == 0 and trend_val < 0)
                    is_counter    = (lgbm_dir == 2 and trend_val < 0) or (lgbm_dir == 0 and trend_val > 0)
                    if is_with_trend:
                        adj_conf -= with_trend_penalty
                    elif is_counter:
                        adj_conf += counter_trend_boost
                    adj_conf = float(np.clip(adj_conf, 0.0, 1.0))
                    # 2c: Block with-trend if conf below absolute threshold
                    if is_with_trend and with_trend_block_conf > 0 and adj_conf < with_trend_block_conf:
                        continue  # skip trade → FLAT

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


def compute_guardian_static_array(
    df: "pd.DataFrame",
    guardian_feat_cols: list[str],
) -> "np.ndarray":
    """
    Extract guardian static features from DataFrame as (N, n_features) array.
    Uses exactly the columns in guardian_feat_cols — missing columns are
    zero-filled to match the scaler's expected input dimensions.
    """
    n = len(df)
    out = np.zeros((n, len(guardian_feat_cols)), dtype=np.float64)
    for idx, col in enumerate(guardian_feat_cols):
        if col in df.columns:
            out[:, idx] = df[col].ffill().fillna(0).values.astype(np.float64)
    return out
