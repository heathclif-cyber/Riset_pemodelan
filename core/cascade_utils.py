"""
core/cascade_utils.py — Cascade Entry Logic untuk Production

Berisi implementasi berbagai mode cascade entry:
  - hard_consensus : LGBM gates, LSTM adjusts (default)
  - dual_gate      : LGBM dan LSTM masing-masing punya threshold independen,
                     keduanya harus lolos dan arah sama

Cara pakai di inference.py:
    from core.cascade_utils import evaluate_entry

    result = evaluate_entry(
        lgbm_proba   = lgbm_proba,        # shape (3,) — [SHORT, FLAT, LONG]
        lstm_proba   = lstm_proba,         # shape (3,) atau None
        cascade_cfg  = config["cascade"],  # dict dari inference_config.json
    )
    if result["entry"]:
        direction  = result["direction"]   # 0=SHORT, 2=LONG
        confidence = result["confidence"]
"""

import numpy as np
from typing import Optional

SHORT, FLAT, LONG = 0, 1, 2


def evaluate_entry(
    lgbm_proba: np.ndarray,
    lstm_proba: Optional[np.ndarray],
    cascade_cfg: dict,
) -> dict:
    """
    Evaluasi entry satu bar berdasarkan konfigurasi cascade.

    Parameters
    ----------
    lgbm_proba : shape (3,) — probabilitas [SHORT, FLAT, LONG] dari LGBM
    lstm_proba : shape (3,) atau None — probabilitas dari LSTM
    cascade_cfg : dict dari inference_config.json["cascade"]

    Returns
    -------
    dict:
        entry      : bool
        direction  : int  (0=SHORT, 1=FLAT, 2=LONG)
        confidence : float
        mode_used  : str
    """
    mode = cascade_cfg.get("mode", "hard_consensus")

    if mode == "dual_gate":
        return _dual_gate(lgbm_proba, lstm_proba, cascade_cfg)
    elif mode == "dual_dominant":
        return _dual_dominant(lgbm_proba, lstm_proba, cascade_cfg)
    elif mode == "lstm_dominant":
        return _lstm_dominant(lgbm_proba, lstm_proba, cascade_cfg)
    else:
        return _hard_consensus(lgbm_proba, lstm_proba, cascade_cfg)


# ─── Mode: Dual Hard Gate ─────────────────────────────────────────────────────

def _dual_gate(lgbm_proba, lstm_proba, cfg) -> dict:
    """
    Dual Hard Gate: LGBM dan LSTM berjalan paralel, masing-masing punya
    threshold independen. Keduanya harus lolos DAN arah sama.

    Tidak ada yang bisa bypass yang lain — true partnership.

    Config keys:
        lgbm_gate  : LGBM minimum confidence (default 0.60)
        lstm_gate  : LSTM minimum confidence (default 0.45)
    """
    lgbm_gate = float(cfg.get("lgbm_gate", 0.60))
    lstm_gate = float(cfg.get("lstm_gate", 0.45))

    # Arah terkuat masing-masing model
    lgbm_dir  = int(np.argmax(lgbm_proba))
    lgbm_conf = float(lgbm_proba[lgbm_dir])
    lstm_dir  = int(np.argmax(lstm_proba)) if lstm_proba is not None else FLAT
    lstm_conf = float(lstm_proba[lstm_dir]) if lstm_proba is not None else 0.0

    # Kedua syarat harus terpenuhi sekaligus
    both_agree     = (lgbm_dir == lstm_dir) and (lgbm_dir != FLAT)
    lgbm_passes    = lgbm_conf >= lgbm_gate
    lstm_passes    = lstm_conf >= lstm_gate

    if both_agree and lgbm_passes and lstm_passes:
        final_conf = (lgbm_conf + lstm_conf) / 2.0
        return {
            "entry": True, "direction": lgbm_dir,
            "confidence": final_conf, "mode_used": "dual_gate",
            "lgbm_conf": lgbm_conf, "lstm_conf": lstm_conf,
        }

    return {"entry": False, "direction": FLAT, "confidence": 0.0,
            "mode_used": "dual_gate", "lgbm_conf": lgbm_conf, "lstm_conf": lstm_conf}


# ─── Mode: Dual Dominant ─────────────────────────────────────────────────────

def _dual_dominant(lgbm_proba, lstm_proba, cfg) -> dict:
    """
    Dual Dominant Gate: LGBM dan LSTM masing-masing punya gate independen,
    tapi LSTM menggunakan logika dominant (max(LONG,SHORT) >= threshold, FLAT diabaikan).

    Keduanya harus lolos DAN searah — true dual gate dengan LSTM lebih fleksibel.

    Config keys:
        lgbm_gate                : LGBM argmax minimum confidence (default 0.65)
        lstm_dominant_threshold  : LSTM max(LONG,SHORT) minimum (default 0.35)
    """
    lgbm_gate  = float(cfg.get("lgbm_gate", 0.65))
    lstm_thr   = float(cfg.get("lstm_dominant_threshold", 0.35))

    lgbm_l = float(lgbm_proba[LONG]);  lgbm_s = float(lgbm_proba[SHORT])
    if lgbm_l >= lgbm_gate and lgbm_l > lgbm_s:
        lgbm_dir, lgbm_conf = LONG, lgbm_l
    elif lgbm_s >= lgbm_gate and lgbm_s > lgbm_l:
        lgbm_dir, lgbm_conf = SHORT, lgbm_s
    else:
        return {"entry": False, "direction": FLAT, "confidence": 0.0,
                "mode_used": "dual_dominant"}

    if lstm_proba is None:
        return {"entry": False, "direction": FLAT, "confidence": 0.0,
                "mode_used": "dual_dominant"}

    lstm_l = float(lstm_proba[LONG]); lstm_s = float(lstm_proba[SHORT])
    if lstm_l > lstm_s:
        lstm_dir, lstm_dom = LONG, lstm_l
    elif lstm_s > lstm_l:
        lstm_dir, lstm_dom = SHORT, lstm_s
    else:
        return {"entry": False, "direction": FLAT, "confidence": 0.0,
                "mode_used": "dual_dominant"}

    if lstm_dir == lgbm_dir and lstm_dom >= lstm_thr:
        final_conf = (lgbm_conf + lstm_dom) / 2.0
        return {"entry": True, "direction": lgbm_dir, "confidence": final_conf,
                "mode_used": "dual_dominant", "lgbm_conf": lgbm_conf, "lstm_conf": lstm_dom}

    return {"entry": False, "direction": FLAT, "confidence": 0.0,
            "mode_used": "dual_dominant", "lgbm_conf": lgbm_conf, "lstm_conf": lstm_dom}


# ─── Mode: LSTM Dominant ─────────────────────────────────────────────────────

def _lstm_dominant(lgbm_proba, lstm_proba, cfg) -> dict:
    """
    LSTM Dominant Gate: LGBM pakai standard threshold (lgbm_threshold_long/short).
    LSTM: abaikan FLAT, ambil dominant antara LONG vs SHORT,
    konfirmasi jika dominant prob >= threshold dan searah LGBM.

    Config keys:
        lgbm_threshold_long      : 0.69
        lgbm_threshold_short     : 0.59
        lstm_dominant_threshold  : 0.35
    """
    thr_long  = float(cfg.get("lgbm_threshold_long",  0.69))
    thr_short = float(cfg.get("lgbm_threshold_short", 0.59))
    lstm_thr  = float(cfg.get("lstm_dominant_threshold", 0.35))

    lgbm_l = float(lgbm_proba[LONG]); lgbm_s = float(lgbm_proba[SHORT])
    if lgbm_l >= thr_long:
        lgbm_dir, lgbm_conf = LONG, lgbm_l
    elif lgbm_s >= thr_short:
        lgbm_dir, lgbm_conf = SHORT, lgbm_s
    else:
        return {"entry": False, "direction": FLAT, "confidence": 0.0,
                "mode_used": "lstm_dominant"}

    if lstm_proba is None:
        return {"entry": True, "direction": lgbm_dir, "confidence": lgbm_conf,
                "mode_used": "lstm_dominant"}

    lstm_l = float(lstm_proba[LONG]); lstm_s = float(lstm_proba[SHORT])
    if lstm_l > lstm_s:
        lstm_dir, lstm_dom = LONG, lstm_l
    elif lstm_s > lstm_l:
        lstm_dir, lstm_dom = SHORT, lstm_s
    else:
        return {"entry": False, "direction": FLAT, "confidence": 0.0,
                "mode_used": "lstm_dominant"}

    if lstm_dir == lgbm_dir and lstm_dom >= lstm_thr:
        return {"entry": True, "direction": lgbm_dir, "confidence": lgbm_conf,
                "mode_used": "lstm_dominant", "lgbm_conf": lgbm_conf, "lstm_conf": lstm_dom}

    return {"entry": False, "direction": FLAT, "confidence": 0.0,
            "mode_used": "lstm_dominant", "lgbm_conf": lgbm_conf, "lstm_conf": lstm_dom}


# ─── Mode: Hard Consensus (default) ──────────────────────────────────────────

def _hard_consensus(lgbm_proba, lstm_proba, cfg) -> dict:
    """
    Hard Consensus: LGBM gates, LSTM adjusts confidence.

    Config keys:
        lgbm_threshold_long  : 0.69
        lgbm_threshold_short : 0.59
        confidence_threshold_entry : 0.59
        lstm_adjust_agree_boost    : 0.05
        lstm_adjust_neutral_pen    : 0.10
        lstm_adjust_opposite_pen   : 0.85
    """
    thr_long  = float(cfg.get("lgbm_threshold_long",  0.69))
    thr_short = float(cfg.get("lgbm_threshold_short", 0.59))
    thr_entry = float(cfg.get("confidence_threshold_entry", 0.59))
    boost     = float(cfg.get("lstm_adjust_agree_boost",   0.05))
    neu_pen   = float(cfg.get("lstm_adjust_neutral_pen",   0.10))
    opp_pen   = float(cfg.get("lstm_adjust_opposite_pen",  0.85))

    lgbm_long_conf  = float(lgbm_proba[LONG])
    lgbm_short_conf = float(lgbm_proba[SHORT])

    # Tentukan arah LGBM
    if lgbm_long_conf >= thr_long:
        lgbm_dir, lgbm_conf = LONG, lgbm_long_conf
    elif lgbm_short_conf >= thr_short:
        lgbm_dir, lgbm_conf = SHORT, lgbm_short_conf
    else:
        return {"entry": False, "direction": FLAT, "confidence": 0.0,
                "mode_used": "hard_consensus"}

    # LSTM adjustment
    adj_conf = lgbm_conf
    if lstm_proba is not None:
        lstm_dir = int(np.argmax(lstm_proba))
        if lstm_dir == lgbm_dir:
            adj_conf += boost
        elif lstm_dir == FLAT:
            adj_conf -= neu_pen
        else:
            adj_conf -= opp_pen
        adj_conf = float(np.clip(adj_conf, 0.0, 1.0))

    if adj_conf >= thr_entry:
        return {"entry": True, "direction": lgbm_dir, "confidence": adj_conf,
                "mode_used": "hard_consensus"}

    return {"entry": False, "direction": FLAT, "confidence": adj_conf,
            "mode_used": "hard_consensus"}
