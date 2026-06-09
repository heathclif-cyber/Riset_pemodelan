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

# Module-level storage for position sizing (Simons: "change the bet size, not the signal")
_last_size_mult: np.ndarray | None = None

from config import (
    NUM_CLASSES,
    LGBM_THRESHOLD_LONG, LGBM_THRESHOLD_SHORT,
    LGBM_FLAT_REVIEW_THRESHOLD,
    LSTM_CONFIRMATION_ENABLED,
    LSTM_FLAT_REVIEW_ENABLED,
    LSTM_DIRECTIONAL_REVIEW_THRESHOLD,
    LSTM_ADJUST_MODE,
    LSTM_ADJUST_AGREE_BOOST, LSTM_ADJUST_NEUTRAL_PEN, LSTM_ADJUST_OPPOSITE_PEN,
    LSTM_TIERED_MULTIPLIERS,
    LSTM_OVERRIDE_THRESHOLD,
    REGIME_NAMES, MODEL_DIR,
    TREND_ALIGNMENT_ENABLED, WITH_TREND_PENALTY, COUNTER_TREND_BOOST,
    WITH_TREND_BLOCK_CONF, REGIME_AWARE_ALIGNMENT,
    POSITIONING_ENGINE_ENABLED, POSITIONING_SIZE_MULTIPLIER, POSITIONING_LS_EXTREME_THR,
    HMM_GATE_LSTM_ENABLED,
    CONFIDENCE_THRESHOLD_ENTRY,
    # New Momentum Gate parameters
    LSTM_FUSION_MODE,
    LSTM_CORRECTION_POWER,
    LSTM_BOOST_MULTIPLIER,
    LSTM_MIN_SCORE_TO_CORRECT,
    LSTM_MAX_PROB_SHIFT,
)
from pipeline.shared import SequenceDataset
from core.utils import get_lstm_device

DEVICE = torch.device("cpu")  # LSTM inference via CPU (hindari DirectML device mismatch)

# ─── Momentum Dynamic Threshold ──────────────────────────────────────────────
# Turunkan LGBM threshold saat vol_spike_zscore tinggi (pump/dump detection).
# Diaktifkan per-skenario via patching dari luar (default: off).
MOMENTUM_DYNAMIC_THRESHOLD_ENABLED = False
MOMENTUM_VOL_SPIKE_COL = "vol_spike_zscore"
MOMENTUM_SPIKE_L1      = 2.0    # vol spike level 1 → kurangi threshold 0.04
MOMENTUM_SPIKE_L2      = 3.0    # vol spike level 2 → kurangi threshold 0.07
MOMENTUM_REDUCE_L1     = 0.04
MOMENTUM_REDUCE_L2     = 0.07

# ─── Trend Score Dynamic Threshold ───────────────────────────────────────────
# Turunkan threshold SECARA ASIMETRIS berdasarkan trend koin.
# Saat uptrend kuat → hanya LONG threshold turun (SHORT tidak turun/naik).
# Saat downtrend kuat → hanya SHORT threshold turun.
# Logika: koin yang sedang trending kuat, entry searah trend lebih valid
# walau confidence LGBM sedikit di bawah normal threshold.
TREND_DYNAMIC_THRESHOLD_ENABLED  = False
TREND_STRENGTH_MIN                = 2.0    # |trend_strength| minimum untuk aktif
TREND_REDUCE_AMOUNT               = 0.05   # kurangi threshold 0.05 di arah trend

# ─── LSTM Standalone (Dual-Path) ─────────────────────────────────────────────
# LSTM bisa masuk mandiri saat LGBM di bawah threshold.
# Tidak ada filter tambahan — murni dari output model LSTM.
# LGBM tetap jalur utama (structural), LSTM jalur momentum.
LSTM_STANDALONE_ENABLED   = False
LSTM_STANDALONE_THRESHOLD = 0.65    # LSTM harus min 65% yakin ke satu arah

# ─── Smart Entry — Simultaneous LGBM+LSTM Fusion ─────────────────────────────
# Kedua model berkontribusi ke setiap bar secara bersamaan.
# LGBM tidak lagi gatekeeper tunggal — LSTM punya suara penuh.
#
# Modes:
#   "disabled"       : current hard_consensus (LGBM gates first)
#   "soft_vote"      : α×p_LGBM + β×p_LSTM
#   "geometric"      : √(p_LGBM × p_LSTM) normalize — strict consensus
#   "dynamic_weight" : model lebih yakin dapat bobot lebih besar
#
SMART_ENTRY_MODE      = "disabled"
SMART_ENTRY_LGBM_W    = 0.60    # weight LGBM di soft_vote (LSTM = 1 - ini)
SMART_ENTRY_THRESHOLD = 0.45    # threshold confidence combined untuk masuk
SMART_ENTRY_LSTM_MIN  = 0.38    # LSTM harus minimal agree (>random 0.333) untuk arah yang dipilih
                                 # Kalau LSTM flat (0.33), entry diblokir — LSTM punya veto!
LSTM_SOFT_GATE_OPP_MAX = 0.35   # soft_gate: LSTM opposite prob max (default 0.35)
RATIO_MULTIPLIER       = 2.0    # ratio_gate: direktional harus >= X * lawan (default 2.0)
RATIO_FLAT_MAX         = 0.70   # ratio_gate: FLAT harus < nilai ini agar tidak overwhelm sinyal
LSTM_DOMINANT_THRESHOLD = 0.35  # lstm_dominant: prob max(LONG,SHORT) LSTM harus >= nilai ini
SMART_ENTRY_LGBM_GATE = 0.55   # dual_gate: LGBM hard threshold (independen, lebih rendah dari 0.69)
SMART_ENTRY_LSTM_GATE = 0.42   # dual_gate: LSTM hard threshold (independen)

# ─── Regime Model Registry ────────────────────────────────────────────────────
# Cache per-regime LGBM models (loaded lazily on first use)
_regime_models: dict = {}   # regime_name → lgbm model | None
_last_loaded_model_dir: Path | None = None


def load_regime_models(model_dir: Path = MODEL_DIR) -> dict:
    """
    Load semua per-regime LGBM models dari models/lgbm_regime_*.pkl.
    Return dict: {regime_name: model_or_None}.
    Dipanggil sekali saat pertama kali hierarchical_predict() dijalankan.
    """
    global _regime_models, _last_loaded_model_dir
    if _last_loaded_model_dir == model_dir:
        return _regime_models

    _regime_models = {}
    _last_loaded_model_dir = model_dir

    for rname in REGIME_NAMES:
        safe = rname.replace(" ", "_")
        path = model_dir / f"lgbm_regime_{safe}.pkl"
        if path.exists():
            try:
                _regime_models[rname] = joblib.load(path)
                logger.info(f"[regime] Loaded: {path.name} from {model_dir}")
            except Exception as e:
                logger.warning(f"[regime] Gagal load {path.name} dari {model_dir}: {e}")
                _regime_models[rname] = None
        else:
            _regime_models[rname] = None

    loaded = [k for k, v in _regime_models.items() if v is not None]
    logger.info(f"[regime] Per-regime models loaded from {model_dir}: {loaded}")
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
    """Run LSTM inference; pad head rows yang tidak punya full sequence.
    Auto-handles feature mismatch: jika X punya lebih banyak kolom dari
    yang LSTM tahu (misalnya LGBM diretrain dengan fitur baru tapi LSTM tidak),
    kolom ekstra di akhir di-drop — aman selama fitur baru di-append di akhir.
    """
    lstm_n_feat = getattr(lstm_scaler, "n_features_in_", X.shape[1])
    if X.shape[1] > lstm_n_feat:
        X = X[:, :lstm_n_feat]
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
    # ── Trend Alignment / HMM Controller (Grup 2) ──────────────────────────────
    trend_alignment_enabled: bool = TREND_ALIGNMENT_ENABLED,
    with_trend_penalty:      float = WITH_TREND_PENALTY,   # 2a
    counter_trend_boost:     float = COUNTER_TREND_BOOST,  # 2b
    with_trend_block_conf:   float = WITH_TREND_BLOCK_CONF, # 2c (0 = disable)
    hmm_controller_enabled:  bool = False,  # HMM-based regime controller (replaces h4_trend)
    regime_aware_alignment:  bool = REGIME_AWARE_ALIGNMENT,  # FLIP from config
    model_dir:               Path = MODEL_DIR,              # Path to regime models
    lstm_feat_cols:          list = None,
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
    regime_models = load_regime_models(model_dir)
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
        if lstm_feat_cols is not None:
            # Align features dynamically from df_slice
            X_lstm = np.zeros((n, len(lstm_feat_cols)), dtype=np.float64)
            for idx, col in enumerate(lstm_feat_cols):
                if col in df_slice.columns:
                    X_lstm[:, idx] = df_slice[col].ffill().fillna(0).values.astype(np.float64)
        else:
            X_lstm = X
        lstm_proba = get_lstm_proba(lstm_model, lstm_scaler, X_lstm, n)
    else:
        lstm_proba = None

    _pass_rate["total"] = n
    _pass_rate["lgbm"]  = 0
    _pass_rate["lstm"]  = 0

    y_pred      = np.ones(n, dtype=np.int64)
    confidence  = np.full(n, 1.0 / NUM_CLASSES)
    size_mult   = np.ones(n, dtype=np.float64)  # positioning size multiplier per bar

    for i in range(n):
        lgbm_long_conf  = lgbm_proba[i, 2]
        lgbm_short_conf = lgbm_proba[i, 0]

        # ── Dynamic Threshold Adjustment (momentum pump/dump detection) ───────
        # Saat vol_spike_zscore tinggi, pasar sedang dalam momentum kuat.
        # Kurangi effective threshold agar LGBM tidak miss sinyal yang sedikit
        # di bawah normal threshold tapi valid secara momentum.
        if MOMENTUM_DYNAMIC_THRESHOLD_ENABLED and MOMENTUM_VOL_SPIKE_COL in df_slice.columns:
            vol_spike = float(df_slice[MOMENTUM_VOL_SPIKE_COL].iloc[i])
            if vol_spike >= MOMENTUM_SPIKE_L2:
                _momentum_reduce = MOMENTUM_REDUCE_L2
            elif vol_spike >= MOMENTUM_SPIKE_L1:
                _momentum_reduce = MOMENTUM_REDUCE_L1
            else:
                _momentum_reduce = 0.0
        else:
            _momentum_reduce = 0.0

        # Trend-based ASYMMETRIC threshold reduction
        # Hanya kurangi threshold di arah yang sama dengan trend — counter-trend tidak dibantu
        _trend_long_reduce  = 0.0
        _trend_short_reduce = 0.0
        if TREND_DYNAMIC_THRESHOLD_ENABLED:
            h4_t  = df_slice["h4_trend"].iloc[i]     if "h4_trend"       in df_slice.columns else 0.0
            t_str = df_slice["trend_strength"].iloc[i] if "trend_strength" in df_slice.columns else 0.0
            e_sl  = df_slice["ema_21_slope_h4"].iloc[i] if "ema_21_slope_h4" in df_slice.columns else 0.0

            if h4_t > 0 and t_str > TREND_STRENGTH_MIN and e_sl > 0:
                _trend_long_reduce  = TREND_REDUCE_AMOUNT   # uptrend: bantu LONG masuk
            elif h4_t < 0 and t_str < -TREND_STRENGTH_MIN and e_sl < 0:
                _trend_short_reduce = TREND_REDUCE_AMOUNT   # downtrend: bantu SHORT masuk

        eff_long_thr  = max(0.0, LGBM_THRESHOLD_LONG  - _momentum_reduce - _trend_long_reduce)
        eff_short_thr = max(0.0, LGBM_THRESHOLD_SHORT - _momentum_reduce - _trend_short_reduce)

        # ── Smart Entry: Simultaneous LGBM+LSTM Fusion ───────────────────────
        # Kedua model berkontribusi SETARA — LGBM bukan gatekeeper tunggal.
        # Saat aktif, hard_consensus path DINONAKTIFKAN sepenuhnya.
        # LGBM tidak bisa masuk sendiri tanpa kontribusi LSTM.
        if SMART_ENTRY_MODE != "disabled" and lstm_proba is not None:
            p_lgbm = lgbm_proba[i]                    # shape (3,)
            p_lstm = lstm_proba[i]                     # shape (3,)

            if SMART_ENTRY_MODE == "soft_vote":
                w_l = SMART_ENTRY_LGBM_W
                w_s = 1.0 - w_l
                p_combined = w_l * p_lgbm + w_s * p_lstm

            elif SMART_ENTRY_MODE == "mutual_agree":
                # Soft vote TAPI LSTM harus minimal agree untuk arah terpilih.
                # Jika LSTM flat/netral → entry diblokir, LSTM punya veto.
                w_l = SMART_ENTRY_LGBM_W
                w_s = 1.0 - w_l
                p_combined = w_l * p_lgbm + w_s * p_lstm

            elif SMART_ENTRY_MODE == "geometric":
                p_combined = np.sqrt(np.clip(p_lgbm * p_lstm, 1e-10, 1.0))
                p_combined = p_combined / (p_combined.sum() + 1e-10)

            elif SMART_ENTRY_MODE == "dynamic_weight":
                w_lgbm = float(p_lgbm.max())
                w_lstm = float(p_lstm.max())
                denom  = w_lgbm + w_lstm + 1e-10
                p_combined = (w_lgbm * p_lgbm + w_lstm * p_lstm) / denom

            elif SMART_ENTRY_MODE == "ratio_gate":
                # Ratio Gate: entry jika skor direktional salah satu model
                # >= RATIO_MULTIPLIER × skor arah berlawanan.
                # Berlaku KEDUANYA untuk LGBM dan LSTM, dan harus searah.
                #
                # Contoh (multiplier=2.0):
                #   LGBM LONG=0.40, SHORT=0.20 → ratio=2.0 → LOLOS
                #   LGBM LONG=0.40, SHORT=0.30 → ratio=1.33 → TIDAK
                #   LSTM harus menunjukkan dominasi yang sama di arah yang sama.
                lgbm_l = float(p_lgbm[2]); lgbm_s = float(p_lgbm[0]); lgbm_f = float(p_lgbm[1])
                lstm_l = float(p_lstm[2]); lstm_s = float(p_lstm[0]); lstm_f = float(p_lstm[1])
                mul    = RATIO_MULTIPLIER

                # LGBM: direktional dominan DAN FLAT tidak boleh sangat dominan
                # "FLAT harus dibawah salah satu posisi" = FLAT < RATIO_FLAT_MAX
                flat_max = RATIO_FLAT_MAX
                if lgbm_l >= mul * lgbm_s and lgbm_f < flat_max:
                    lgbm_dir_rg = 2   # LONG dominan, FLAT tidak overwhelm
                elif lgbm_s >= mul * lgbm_l and lgbm_f < flat_max:
                    lgbm_dir_rg = 0   # SHORT dominan, FLAT tidak overwhelm
                else:
                    continue          # ratio tidak cukup atau FLAT terlalu dominan

                # LSTM: searah DAN FLAT tidak overwhelm
                if lgbm_dir_rg == 2:
                    lstm_ok = lstm_l >= mul * lstm_s and lstm_f < flat_max
                else:
                    lstm_ok = lstm_s >= mul * lstm_l and lstm_f < flat_max

                if lstm_ok:
                    lgbm_conf_rg = lgbm_l if lgbm_dir_rg == 2 else lgbm_s
                    y_pred[i]     = lgbm_dir_rg
                    confidence[i] = lgbm_conf_rg
                    _pass_rate["lgbm"] += 1
                    _pass_rate["lstm"] += 1

                continue

            elif SMART_ENTRY_MODE == "dual_dominant":
                # Dual Dominant Gate:
                #   LGBM → argmax >= lgbm_gate (independen, bukan std threshold)
                #   LSTM → max(LONG, SHORT) >= LSTM_DOMINANT_THRESHOLD (abaikan FLAT)
                #   Keduanya harus lolos dan SEARAH — true dual gate
                #   Confidence final = (lgbm_conf + lstm_dom_prob) / 2
                lgbm_gate_dd = SMART_ENTRY_LGBM_GATE
                p_l = float(p_lgbm[2]); p_s = float(p_lgbm[0])

                # LGBM: argmax directional harus >= gate
                if p_l >= lgbm_gate_dd and p_l > p_s:
                    lgbm_dir_dd = 2; lgbm_conf_dd = p_l
                elif p_s >= lgbm_gate_dd and p_s > p_l:
                    lgbm_dir_dd = 0; lgbm_conf_dd = p_s
                else:
                    continue  # LGBM tidak lolos gate

                # LSTM: max(L,S) harus >= threshold dan searah LGBM
                lstm_l_dd = float(p_lstm[2]); lstm_s_dd = float(p_lstm[0])
                if lstm_l_dd > lstm_s_dd:
                    lstm_dir_dd = 2; lstm_prob_dd = lstm_l_dd
                elif lstm_s_dd > lstm_l_dd:
                    lstm_dir_dd = 0; lstm_prob_dd = lstm_s_dd
                else:
                    continue  # LSTM tie

                if lstm_dir_dd == lgbm_dir_dd and lstm_prob_dd >= LSTM_DOMINANT_THRESHOLD:
                    final_conf_dd = (lgbm_conf_dd + lstm_prob_dd) / 2.0
                    y_pred[i]     = lgbm_dir_dd
                    confidence[i] = final_conf_dd
                    _pass_rate["lgbm"] += 1
                    _pass_rate["lstm"] += 1

                continue

            elif SMART_ENTRY_MODE == "lstm_dominant":
                # LSTM Dominant Gate:
                #   LGBM → standard threshold (0.69/0.59) → menentukan arah
                #   LSTM → abaikan FLAT, bandingkan LONG vs SHORT saja
                #          ambil yang lebih tinggi sebagai "dominant direction"
                #          jika dominant prob >= LSTM_DOMINANT_THRESHOLD
                #          DAN arahnya sama dengan LGBM → ENTRY
                #
                # Contoh: S=27% F=37% L=36% → dominant=LONG (36%>27%)
                #          36% >= 35% threshold → ENTRY (tidak peduli F=37%)
                eff_lt_dm = max(0.0, LGBM_THRESHOLD_LONG  - _momentum_reduce - _trend_long_reduce)
                eff_st_dm = max(0.0, LGBM_THRESHOLD_SHORT - _momentum_reduce - _trend_short_reduce)

                lgbm_l_dm = float(p_lgbm[2]); lgbm_s_dm = float(p_lgbm[0])
                if lgbm_l_dm >= eff_lt_dm:
                    lgbm_dir_dm = 2; lgbm_conf_dm = lgbm_l_dm
                elif lgbm_s_dm >= eff_st_dm:
                    lgbm_dir_dm = 0; lgbm_conf_dm = lgbm_s_dm
                else:
                    continue  # LGBM FLAT

                # LSTM: dominant direction dari {LONG, SHORT} saja (FLAT diabaikan)
                lstm_l_dm = float(p_lstm[2]); lstm_s_dm = float(p_lstm[0])
                if lstm_l_dm > lstm_s_dm:
                    lstm_dom_dir = 2; lstm_dom_prob = lstm_l_dm
                elif lstm_s_dm > lstm_l_dm:
                    lstm_dom_dir = 0; lstm_dom_prob = lstm_s_dm
                else:
                    continue  # LSTM tie, tidak ada dominant

                # Masuk jika arah sama dengan LGBM dan prob >= threshold
                if lstm_dom_dir == lgbm_dir_dm and lstm_dom_prob >= LSTM_DOMINANT_THRESHOLD:
                    y_pred[i]     = lgbm_dir_dm
                    confidence[i] = lgbm_conf_dm
                    _pass_rate["lgbm"] += 1
                    _pass_rate["lstm"] += 1

                continue

            elif SMART_ENTRY_MODE == "lstm_ratio":
                # LSTM Ratio Gate:
                #   LGBM → standard threshold (0.69/0.59) → menentukan arah
                #   LSTM → ratio check: direktional >= RATIO_MULTIPLIER × lawan
                #          DAN FLAT < RATIO_FLAT_MAX
                #   LGBM tidak diubah sama sekali, LSTM hanya sebagai confirmer rasio.
                eff_lt = max(0.0, LGBM_THRESHOLD_LONG  - _momentum_reduce - _trend_long_reduce)
                eff_st = max(0.0, LGBM_THRESHOLD_SHORT - _momentum_reduce - _trend_short_reduce)

                lgbm_l_r = float(p_lgbm[2]); lgbm_s_r = float(p_lgbm[0])
                if lgbm_l_r >= eff_lt:
                    dir_r = 2; conf_r = lgbm_l_r
                elif lgbm_s_r >= eff_st:
                    dir_r = 0; conf_r = lgbm_s_r
                else:
                    continue  # LGBM FLAT — skip

                # LSTM ratio check (LGBM sudah lolos, tinggal LSTM konfirmasi)
                lstm_l_r = float(p_lstm[2]); lstm_s_r = float(p_lstm[0]); lstm_f_r = float(p_lstm[1])
                mul_r = RATIO_MULTIPLIER; flat_r = RATIO_FLAT_MAX

                if dir_r == 2:
                    lstm_ok_r = lstm_l_r >= mul_r * lstm_s_r and lstm_f_r < flat_r
                else:
                    lstm_ok_r = lstm_s_r >= mul_r * lstm_l_r and lstm_f_r < flat_r

                if lstm_ok_r:
                    y_pred[i]     = dir_r
                    confidence[i] = conf_r
                    _pass_rate["lgbm"] += 1
                    _pass_rate["lstm"] += 1

                continue

            elif SMART_ENTRY_MODE == "soft_gate":
                # Soft Gate: LGBM gates (tidak perlu argmax match),
                # LSTM hanya memblokir jika KUAT berlawanan (opp > threshold).
                # Filosofi: "LSTM boleh netral/diam, tapi tidak boleh aktif menolak."
                lgbm_dir_sg  = int(np.argmax(p_lgbm))
                lgbm_conf_sg = float(p_lgbm[lgbm_dir_sg])

                if lgbm_dir_sg != 1 and lgbm_conf_sg >= SMART_ENTRY_LGBM_GATE:
                    # Prob LSTM ke arah BERLAWANAN (bukan FLAT, bukan searah)
                    opp_dir = 2 if lgbm_dir_sg == 0 else 0   # SHORT→LONG, LONG→SHORT
                    lstm_opp_sg = float(p_lstm[opp_dir])

                    if lstm_opp_sg < LSTM_SOFT_GATE_OPP_MAX:
                        y_pred[i]     = lgbm_dir_sg
                        confidence[i] = lgbm_conf_sg
                        _pass_rate["lgbm"] += 1
                        _pass_rate["lstm"] += 1

                continue

            elif SMART_ENTRY_MODE == "dual_gate":
                # Hard AND gate — keduanya harus lolos threshold masing-masing
                # LGBM dan LSTM berjalan PARALEL, saling mengunci satu sama lain
                lgbm_dir_dg  = int(np.argmax(p_lgbm))
                lgbm_conf_dg = float(p_lgbm[lgbm_dir_dg])
                lstm_dir_dg  = int(np.argmax(p_lstm))
                lstm_conf_dg = float(p_lstm[lstm_dir_dg])

                # Kedua arah harus sama, bukan FLAT, dan masing-masing lolos gate
                both_agree      = (lgbm_dir_dg == lstm_dir_dg) and (lgbm_dir_dg != 1)
                lgbm_gate_pass  = lgbm_conf_dg >= SMART_ENTRY_LGBM_GATE
                lstm_gate_pass  = lstm_conf_dg >= SMART_ENTRY_LSTM_GATE

                if both_agree and lgbm_gate_pass and lstm_gate_pass:
                    # Confidence final = rata-rata kedua model (keduanya sudah lolos)
                    final_conf_dg = (lgbm_conf_dg + lstm_conf_dg) / 2.0
                    y_pred[i]     = lgbm_dir_dg
                    confidence[i] = final_conf_dg
                    _pass_rate["lgbm"] += 1
                    _pass_rate["lstm"] += 1

                # Skip sisa logic — saat dual_gate, hard_consensus tidak jalan
                continue

            else:
                p_combined = p_lgbm   # fallback soft_vote

            if SMART_ENTRY_MODE != "dual_gate":
                final_dir  = int(np.argmax(p_combined))
                final_conf = float(p_combined[final_dir])

                # Cek LSTM minimum agreement (untuk semua mode kecuali geometric)
                lstm_agree = float(p_lstm[final_dir])
                lstm_veto  = (SMART_ENTRY_MODE != "geometric") and (lstm_agree < SMART_ENTRY_LSTM_MIN)

                if final_dir != 1 and final_conf >= SMART_ENTRY_THRESHOLD and not lstm_veto:
                    y_pred[i]     = final_dir
                    confidence[i] = final_conf
                    _pass_rate["lgbm"] += 1
                    _pass_rate["lstm"] += 1

            # KRITIS: skip sisa logic (hard_consensus) REGARDLESS apakah smart entry berhasil.
            # LGBM tidak bisa masuk sendiri saat smart entry aktif.
            continue

        # ── LGBM output FLAT (tidak melewati threshold entry) ─────────────────
        if lgbm_long_conf < eff_long_thr and lgbm_short_conf < eff_short_thr:

            # ── LSTM Standalone / Dual-Path Entry ─────────────────────────────
            # LGBM di bawah threshold tapi LSTM mandiri bisa masuk.
            # Murni dari output model — tidak ada filter teknikal tambahan.
            # LSTM adalah jalur momentum: ia melihat sequence 16 bar dan
            # bisa mendeteksi akselerasi SEBELUM LGBM cukup yakin.
            if LSTM_STANDALONE_ENABLED and lstm_proba is not None:
                lstm_dir_sa  = int(np.argmax(lstm_proba[i]))
                lstm_conf_sa = float(lstm_proba[i, lstm_dir_sa])

                if lstm_dir_sa != 1 and lstm_conf_sa >= LSTM_STANDALONE_THRESHOLD:
                    # LSTM mandiri masuk — bypass LGBM threshold
                    y_pred[i]     = lstm_dir_sa
                    confidence[i] = lstm_conf_sa
                    _pass_rate["lgbm"] += 1
                    _pass_rate["lstm"] += 1
                    continue   # skip sisa loop untuk bar ini

            # Hitung max confidence LGBM dari semua kelas
            lgbm_max_conf = float(np.max(lgbm_proba[i]))

            # LSTM Review Activation (V2.5 Hybrid)
            # LSTM "hidup" dalam dua kasus:
            #   1. Jalur lama: LGBM FLAT + max_conf rendah (jika LSTM_FLAT_REVIEW_ENABLED)
            #   2. Jalur baru: LGBM LONG atau SHORT score > LSTM_DIRECTIONAL_REVIEW_THRESHOLD (0.35)
            #
            # Tujuan: Memberi LSTM kesempatan mengoreksi / menguatkan sinyal directional
            # yang sedang "sedang-sedang saja" (>0.35), bukan hanya menyelamatkan FLAT.
            lgbm_directional_score = max(lgbm_long_conf, lgbm_short_conf)
            lstm_should_review = (
                LSTM_FLAT_REVIEW_ENABLED and lstm_proba is not None and
                (lgbm_max_conf < LGBM_FLAT_REVIEW_THRESHOLD or
                 lgbm_directional_score > LSTM_DIRECTIONAL_REVIEW_THRESHOLD)
            )

            if lstm_should_review:
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
            lgbm_thr  = eff_long_thr
        else:
            lgbm_dir  = 0
            lgbm_conf = lgbm_short_conf
            lgbm_thr  = eff_short_thr

        _pass_rate["lgbm"] += 1

        # STEP 3: LSTM adjustment (HMM-gated: only in TRENDING)
        adj_conf = lgbm_conf

        # HMM Gate: LSTM only speaks in TRENDING regime (0=DOWN, 3=UP)
        _hmm_regime = int(df_slice["hmm_regime_enc"].iloc[i]) if "hmm_regime_enc" in df_slice.columns else 1
        _lstm_active = True
        if HMM_GATE_LSTM_ENABLED:
            _lstm_active = _hmm_regime in (0, 3)  # TRENDING only

        if lstm_proba is not None and _lstm_active:
            if LSTM_FUSION_MODE == "momentum_gate":
                # New Corrector + Booster logic
                # For now we map 3-class proba to approximate scores (temporary bridge)
                lstm_up_score = float(lstm_proba[i, 2])      # LONG as proxy for up momentum
                lstm_exh_score = float(lstm_proba[i, 0])     # SHORT as proxy for exhaustion

                final_dir, final_conf = lstm_momentum_fusion(
                    lgbm_long_conf=lgbm_long_conf,
                    lgbm_short_conf=lgbm_short_conf,
                    up_momentum_score=lstm_up_score,
                    exhaustion_score=lstm_exh_score,
                    base_threshold_long=LGBM_THRESHOLD_LONG,
                    base_threshold_short=LGBM_THRESHOLD_SHORT,
                    correction_power=LSTM_CORRECTION_POWER,
                    boost_multiplier=LSTM_BOOST_MULTIPLIER,
                    min_score_to_correct=LSTM_MIN_SCORE_TO_CORRECT,
                )
                adj_conf = final_conf
            elif LSTM_FUSION_MODE == "soft_multiplier":
                # LSTM sebagai confidence multiplier, bukan gate.
                # Ambil prob LSTM untuk arah yang dipilih LGBM.
                # Multiplier range: 0.65 (LSTM sangat kontra) s/d 1.35 (LSTM sangat setuju).
                # Tidak pernah blokir — hanya modulasi.
                if lgbm_dir == 2:   # LGBM LONG -> pakai BULLISH prob
                    lstm_support = float(lstm_proba[i, 2])
                else:               # LGBM SHORT -> pakai BEARISH prob
                    lstm_support = float(lstm_proba[i, 0])
                # lstm_support range ~0.15-0.55 untuk 3-class model (FLAT dominan)
                # Normalize: 0.15->mult=0.70, 0.55->mult=1.30
                lstm_support_norm = (lstm_support - 0.15) / 0.40  # normalize ke 0-1
                lstm_support_norm = max(0.0, min(1.0, lstm_support_norm))
                mult = 0.70 + 0.60 * lstm_support_norm  # range 0.70-1.30
                adj_conf = float(np.clip(adj_conf * mult, 0.0, 1.0))
            else:
                # Old hard_consensus logic
                lstm_dir = int(np.argmax(lstm_proba[i]))
                if lstm_dir == lgbm_dir:
                    adj = LSTM_ADJUST_AGREE_BOOST
                elif lstm_dir == 1:
                    adj = -LSTM_ADJUST_NEUTRAL_PEN
                else:
                    adj = -LSTM_ADJUST_OPPOSITE_PEN
                adj_conf = float(np.clip(adj_conf + adj, 0.0, 1.0))

        # STEP 4: HMM Controller / Trend Alignment — after LSTM adjustment
        if regime_aware_alignment:
            # ── Regime-Aware Alignment (FLIP) ──────────────────────────────────
            # RANGING (regime 1,2): counter-trend boost (swing mode)
            # TRENDING (regime 0,3): with-trend boost (momentum mode)
            # Ini kunci: SAMA model, DUA mode, tanpa retrain.
            regime_col = "hmm_regime_enc"
            h4_col = "h4_trend"
            if regime_col in df_slice.columns and h4_col in df_slice.columns:
                regime = int(df_slice[regime_col].iloc[i])
                h4_t = float(df_slice[h4_col].iloc[i])
                if not np.isnan(h4_t):
                    if regime in [1, 2]:  # RANGING
                        is_with = (lgbm_dir == 2 and h4_t > 0) or (lgbm_dir == 0 and h4_t < 0)
                        is_counter = (lgbm_dir == 2 and h4_t < 0) or (lgbm_dir == 0 and h4_t > 0)
                        if is_with: adj_conf -= 0.10
                        elif is_counter: adj_conf += 0.05
                    else:  # TRENDING (0,3)
                        is_with = (lgbm_dir == 2 and h4_t > 0) or (lgbm_dir == 0 and h4_t < 0)
                        is_counter = (lgbm_dir == 2 and h4_t < 0) or (lgbm_dir == 0 and h4_t > 0)
                        if is_with: adj_conf += 0.10    # FLIP: boost with-trend
                        elif is_counter: adj_conf -= 0.05  # FLIP: penalize counter-trend
                    adj_conf = float(np.clip(adj_conf, 0.0, 1.0))

        elif hmm_controller_enabled:
            # ── HMM Regime Controller ────────────────────────────────────────
            # Regime: 0=TRENDING_DOWN, 1=RANGING_LOW_VOL, 2=RANGING_HIGH_VOL, 3=TRENDING_UP
            # Soft adjustments (no hard block kecuali RANGING_HIGH_VOL)
            regime_col = "hmm_regime_enc"
            if regime_col in df_slice.columns:
                regime = int(df_slice[regime_col].iloc[i])

                # RANGING_HIGH_VOL → block only if confidence is borderline
                if regime == 2:  # RANGING_HIGH_VOL
                    if adj_conf < 0.72:  # only block if not very confident
                        continue
                    else:
                        adj_conf -= 0.03  # slight penalty for confident entries

                # TRENDING regime adjustments (softer than legacy ±0.10)
                elif regime == 0:  # TRENDING_DOWN
                    if lgbm_dir == 2:   # LONG counter-trend → boost
                        adj_conf += 0.07
                    elif lgbm_dir == 0: # SHORT with-trend → mild penalty
                        adj_conf -= 0.05
                elif regime == 3:  # TRENDING_UP
                    if lgbm_dir == 2:   # LONG with-trend → mild penalty
                        adj_conf -= 0.05
                    elif lgbm_dir == 0: # SHORT counter-trend → boost
                        adj_conf += 0.07
                # regime == 1: RANGING_LOW_VOL — no adjustment

                adj_conf = float(np.clip(adj_conf, 0.0, 1.0))

        elif trend_alignment_enabled:
            # ── Legacy h4_trend Alignment ────────────────────────────────────
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

        # STEP 5: Positioning Engine — SIZE multiplier (Simons: "change the bet, not the signal")
        pos_size_mult = 1.0
        if POSITIONING_ENGINE_ENABLED:
            pos_extreme = float(df_slice["pos_extreme"].iloc[i]) if "pos_extreme" in df_slice.columns else 0.0
            # LS extreme (>2σ) → size × 0.50 (|IC|=0.16, strongest positioning signal)
            # Predicts VOLATILITY ahead, not direction → reduce exposure
            if pos_extreme > 0:
                pos_size_mult = POSITIONING_SIZE_MULTIPLIER  # 0.50

        if LSTM_FUSION_MODE == "momentum_gate" and 'final_dir' in locals():
            effective_dir = final_dir
        else:
            effective_dir = lgbm_dir

        if adj_conf >= lgbm_thr:
            _pass_rate["lstm"] += 1
            y_pred[i]      = effective_dir
            confidence[i]  = adj_conf
            size_mult[i]   = pos_size_mult

    if n > 0:
        n_lgbm = _pass_rate["lgbm"]
        n_fin  = _pass_rate["lstm"]
        logger.info(
            f"[pass_rate] LGBM_pass={n_lgbm}/{n} ({n_lgbm/n:.1%}) → "
            f"FINAL={n_fin}/{n} ({n_fin/n:.1%})"
        )

    # Store size_mult globally for callers
    global _last_size_mult
    _last_size_mult = size_mult

    return y_pred, confidence


# ─── New Fusion Logic: LSTM as Corrector + Booster (2026-06 Design) ──────────

def lstm_momentum_fusion(
    lgbm_long_conf: float,
    lgbm_short_conf: float,
    up_momentum_score: float,      # from Head 1 (0-1)
    exhaustion_score: float,       # from Head 2 (0-1)
    base_threshold_long: float,
    base_threshold_short: float,
    correction_power: float = 0.35,   # seberapa kuat LSTM boleh mengoreksi (0.0 - 1.0)
    boost_multiplier: float = 0.8,    # seberapa kuat boost ketika aligned
    min_score_to_correct: float = 0.65,
) -> tuple[int, float]:
    """
    Fusion baru untuk peran LSTM sebagai Corrector + Booster.

    - LGBM memberikan base probability.
    - LSTM memberikan skor kekuatan momentum (bukan hard direction).
    - LSTM boleh mengoreksi meskipun LGBM yakin, tapi dengan batas kekuatan sedang.
    - Dirancang agar lebih sering memberikan pengaruh (sesuai keinginan user).

    Return: (final_direction, adjusted_confidence)
    """
    # Hitung adjusted probabilities
    adj_long = lgbm_long_conf
    adj_short = lgbm_short_conf

    # Boost ketika LSTM mendukung arah LGBM
    if lgbm_long_conf > lgbm_short_conf:
        # LGBM condong LONG
        if up_momentum_score >= min_score_to_correct:
            adj_long = min(1.0, adj_long + up_momentum_score * boost_multiplier)
        if exhaustion_score >= min_score_to_correct:
            # Koreksi: kurangi LONG karena exhaustion
            adj_long = max(0.0, adj_long - exhaustion_score * correction_power)
    else:
        # LGBM condong SHORT
        if exhaustion_score >= min_score_to_correct:
            adj_short = min(1.0, adj_short + exhaustion_score * boost_multiplier)
        if up_momentum_score >= min_score_to_correct:
            # Koreksi: kurangi SHORT karena ada momentum naik kuat
            adj_short = max(0.0, adj_short - up_momentum_score * correction_power)

    # Tentukan arah final
    if adj_long > adj_short:
        final_dir = 2  # LONG
        final_conf = adj_long
        threshold = base_threshold_long
    else:
        final_dir = 0  # SHORT
        final_conf = adj_short
        threshold = base_threshold_short

    # Jika confidence masih di bawah threshold, kita bisa biarkan LSTM "menyelamatkan"
    # dengan memberikan kesempatan lebih (sesuai semangat "lebih sering boost")
    if final_conf < threshold and max(up_momentum_score, exhaustion_score) > 0.70:
        final_conf = max(final_conf, 0.58)  # kasih minimum confidence jika LSTM sangat yakin

    final_conf = float(np.clip(final_conf, 0.0, 1.0))

    return final_dir, final_conf


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
