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
    elif mode == "conditional_momentum":
        return _conditional_momentum(lgbm_proba, lstm_proba, cascade_cfg, hmm_cfg=None)
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


# ─── Vectorized OOF score fusion (LSTM boost / penalty on LGBM) ───────────────

def apply_lstm_proba_fusion_pre(
    p0: np.ndarray,
    p2: np.ndarray,
    lstm_proba: np.ndarray,
    *,
    agree_boost: float = 0.05,
    neutral_pen: float = 0.05,
    opposite_pen: float = 0.08,
    active_mask: Optional[np.ndarray] = None,
    proportional: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Adjust LGBM directional scores before HMM thresholding.

    LSTM classes: 0=SHORT/BEAR, 1=FLAT/NEUTRAL, 2=LONG/BULL (same as LGBM).
    - BULLISH LSTM: boost p2, penalize p0
    - BEARISH LSTM: boost p0, penalize p2
    - NEUTRAL LSTM: penalize both directional scores

    Bars without LSTM (active_mask=False or NaN proba) are unchanged.
    """
    p0_out = p0.astype(np.float32, copy=True)
    p2_out = p2.astype(np.float32, copy=True)
    n = len(p0_out)
    if lstm_proba is None or len(lstm_proba) != n:
        return p0_out, p2_out

    valid = np.isfinite(lstm_proba).all(axis=1)
    if active_mask is not None:
        valid &= active_mask

    if not valid.any():
        return p0_out, p2_out

    lstm_dir = np.argmax(lstm_proba, axis=1)
    bull = valid & (lstm_dir == LONG)
    bear = valid & (lstm_dir == SHORT)
    neu  = valid & (lstm_dir == FLAT)

    if proportional:
        b_boost = agree_boost * lstm_proba[bull, LONG]
        s_boost = agree_boost * lstm_proba[bear, SHORT]
    else:
        b_boost = agree_boost
        s_boost = agree_boost

    p2_out[bull] += b_boost
    p0_out[bull] -= opposite_pen
    p0_out[bear] += s_boost
    p2_out[bear] -= opposite_pen
    p2_out[neu]  -= neutral_pen
    p0_out[neu]  -= neutral_pen

    np.clip(p0_out, 0.0, 1.0, out=p0_out)
    np.clip(p2_out, 0.0, 1.0, out=p2_out)
    return p0_out, p2_out


def apply_lstm_boost_only_pre(
    p0: np.ndarray,
    p2: np.ndarray,
    lstm_proba: np.ndarray,
    *,
    agree_boost: float = 0.08,
    active_mask: Optional[np.ndarray] = None,
    boost_side: str = "both",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Momentum overlay: boost only, no penalty. For pump/dump gate bars.

    boost_side: "both" | "long" | "short"
      - long  : boost p2 when LSTM BULLISH
      - short : boost p0 when LSTM BEARISH
      - both  : both directions
    """
    p0_out = p0.astype(np.float32, copy=True)
    p2_out = p2.astype(np.float32, copy=True)
    n = len(p0_out)
    if lstm_proba is None or len(lstm_proba) != n:
        return p0_out, p2_out

    valid = np.isfinite(lstm_proba).all(axis=1)
    if active_mask is not None:
        valid &= active_mask
    if not valid.any():
        return p0_out, p2_out

    lstm_dir = np.argmax(lstm_proba, axis=1)
    if boost_side in ("both", "long"):
        bull = valid & (lstm_dir == LONG)
        p2_out[bull] += agree_boost
    if boost_side in ("both", "short"):
        bear = valid & (lstm_dir == SHORT)
        p0_out[bear] += agree_boost

    np.clip(p0_out, 0.0, 1.0, out=p0_out)
    np.clip(p2_out, 0.0, 1.0, out=p2_out)
    return p0_out, p2_out


def resolve_hmm_thresholds(hmm_enc: int, hmm_cfg: dict) -> tuple[float, float]:
    """Per-state [thr_long, thr_short] for one bar (Config B)."""
    raw = hmm_cfg.get("per_state_thresholds", {}) if hmm_cfg else {}
    if not raw:
        tl = float(hmm_cfg.get("trending_threshold", 0.45) if hmm_cfg else 0.45)
        ts = float(hmm_cfg.get("ranging_threshold", 0.55) if hmm_cfg else 0.55)
        return tl, ts
    default = raw.get("-1", raw.get(-1, [0.45, 0.45]))
    state_thr = raw.get(str(hmm_enc), raw.get(hmm_enc, default))
    return float(state_thr[0]), float(state_thr[1])


def apply_hmm_gate_single(p0: float, p2: float, tl: float, ts: float) -> tuple[int, float]:
    """Direction + confidence after HMM per-state thresholds (long wins tie)."""
    if p2 >= tl:
        return LONG, float(p2)
    if p0 >= ts:
        return SHORT, float(p0)
    return FLAT, 0.0


def proba_from_direction(direction: int, confidence: float) -> np.ndarray:
    """Build [SHORT, FLAT, LONG] proba for inference output."""
    proba = np.zeros(3, dtype=np.float32)
    if direction == FLAT or confidence <= 0.0:
        proba[FLAT] = 1.0
        return proba
    proba[direction] = float(np.clip(confidence, 0.0, 1.0))
    proba[FLAT] = float(np.clip(1.0 - proba[direction], 0.0, 1.0))
    return proba


def _conditional_momentum(
    lgbm_proba: np.ndarray,
    lstm_proba: Optional[np.ndarray],
    cascade_cfg: dict,
    hmm_cfg: Optional[dict] = None,
    *,
    vol_spike: float = 0.0,
    hmm_enc: int = -1,
) -> dict:
    """
    LGBM + HMM primary; LSTM adjusts p0/p2 on vol_spike bars only (05j winner).
    Fusion pre-HMM using per-state tl/ts, then re-gate.
    """
    mom = cascade_cfg.get("lstm_momentum", {})
    if hmm_cfg is None:
        hmm_cfg = {}
    tl, ts = resolve_hmm_thresholds(int(hmm_enc), hmm_cfg)

    p0 = float(lgbm_proba[SHORT])
    p2 = float(lgbm_proba[LONG])
    lstm_enabled = cascade_cfg.get("lstm_confirmation_enabled", True) and mom.get("enabled", True)

    if lstm_enabled and lstm_proba is not None and np.isfinite(lstm_proba).all():
        p0_arr = np.array([p0], dtype=np.float32)
        p2_arr = np.array([p2], dtype=np.float32)
        tl_arr = np.array([tl], dtype=np.float32)
        ts_arr = np.array([ts], dtype=np.float32)
        vol_arr = np.array([float(vol_spike)], dtype=np.float32)
        lstm_arr = np.asarray(lstm_proba, dtype=np.float32).reshape(1, 3)
        p0_arr, p2_arr = apply_conditional_momentum_fusion_pre(
            p0_arr, p2_arr, lstm_arr, tl_arr, ts_arr, vol_arr,
            vol_thr=float(mom.get("vol_thr", 2.0)),
            bull_thr=float(mom.get("bull_thr", 0.38)),
            bear_thr=float(mom.get("bear_thr", 0.50)),
            near_miss_gap=float(mom.get("near_miss_gap", 0.03)),
            boost=float(mom.get("boost", 0.10)),
            opposite_pen=float(mom.get("opposite_pen", 0.14)),
            enable_boost=bool(mom.get("enable_boost", True)),
            enable_penalty=bool(mom.get("enable_penalty", True)),
            proportional=bool(mom.get("proportional", True)),
        )
        p0, p2 = float(p0_arr[0]), float(p2_arr[0])

    direction, confidence = apply_hmm_gate_single(p0, p2, tl, ts)
    lstm_dom_dir, lstm_dom_val = None, 0.0
    if lstm_proba is not None and np.isfinite(lstm_proba).all():
        d_arr, dom_arr = _lstm_dominant_direction(
            np.asarray(lstm_proba, dtype=np.float32).reshape(1, 3)
        )
        lstm_dom_dir, lstm_dom_val = int(d_arr[0]), float(dom_arr[0])

    return {
        "entry": direction != FLAT,
        "direction": direction,
        "confidence": confidence,
        "mode_used": "conditional_momentum",
        "fused_p0": p0,
        "fused_p2": p2,
        "hmm_tl": tl,
        "hmm_ts": ts,
        "vol_spike": float(vol_spike),
        "lstm_dom_dir": lstm_dom_dir,
        "lstm_dom": lstm_dom_val,
    }


def evaluate_conditional_momentum_entry(
    lgbm_proba: np.ndarray,
    lstm_proba: Optional[np.ndarray],
    cascade_cfg: dict,
    hmm_cfg: dict,
    *,
    vol_spike: float,
    hmm_enc: int,
) -> dict:
    """Public wrapper for live inference (single bar)."""
    return _conditional_momentum(
        lgbm_proba, lstm_proba, cascade_cfg, hmm_cfg,
        vol_spike=vol_spike, hmm_enc=hmm_enc,
    )


def _lstm_dominant_direction(lstm_proba: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Ignore NEUTRAL — take max(SHORT, LONG) as momentum direction + strength."""
    p0 = lstm_proba[:, SHORT]
    p2 = lstm_proba[:, LONG]
    long_wins = p2 >= p0
    dom = np.where(long_wins, p2, p0)
    direction = np.where(long_wins, LONG, SHORT).astype(np.int32)
    return direction, dom.astype(np.float32)


def apply_conditional_momentum_fusion_pre(
    p0: np.ndarray,
    p2: np.ndarray,
    lstm_proba: np.ndarray,
    tl_arr: np.ndarray,
    ts_arr: np.ndarray,
    vol_spike: np.ndarray,
    *,
    vol_thr: float = 2.0,
    bull_thr: float = 0.40,
    bear_thr: float = 0.55,
    near_miss_gap: float = 0.05,
    boost: float = 0.08,
    opposite_pen: float = 0.12,
    enable_boost: bool = True,
    enable_penalty: bool = True,
    lstm_valid: Optional[np.ndarray] = None,
    proportional: bool = True,
    breadth_mask: Optional[np.ndarray] = None,
    breadth_gate_boost_side: str = "both",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Specialized momentum fusion (pump/dump):

    BOOST  — LGBM FLAT or near-miss on vol_spike bars, LSTM momentum confident:
             raise p2 (BULL) or p0 (BEAR). Targets cases LGBM rarely enters.

    PENALTY — LGBM would enter but LSTM dominant opposes on vol_spike bars:
             cut directional score (momentum reversal / wrong-way entry).
    """
    p0_out = p0.astype(np.float32, copy=True)
    p2_out = p2.astype(np.float32, copy=True)
    n = len(p0_out)
    if lstm_proba is None or len(lstm_proba) != n:
        return p0_out, p2_out

    valid = np.isfinite(lstm_proba).all(axis=1) & (vol_spike >= vol_thr)
    if lstm_valid is not None:
        valid &= lstm_valid
    if not valid.any():
        return p0_out, p2_out

    lstm_dir, lstm_dom = _lstm_dominant_direction(lstm_proba)
    would_long = p2_out >= tl_arr
    would_short = (p0_out >= ts_arr) & ~would_long
    lgbm_flat = ~would_long & ~would_short
    near_miss_long = ~would_long & (p2_out >= (tl_arr - near_miss_gap))
    near_miss_short = ~would_short & ~would_long & (p0_out >= (ts_arr - near_miss_gap))

    if enable_boost:
        boost_long_zone = valid & (lgbm_flat | near_miss_long)
        boost_short_zone = valid & (lgbm_flat | near_miss_short)
        if breadth_mask is not None:
            bm = breadth_mask.astype(bool)
            if breadth_gate_boost_side in ("both", "long"):
                boost_long_zone &= bm
            if breadth_gate_boost_side in ("both", "short"):
                boost_short_zone &= bm
        bull_ok = (lstm_dir == LONG) & (lstm_dom >= bull_thr)
        bear_ok = (lstm_dir == SHORT) & (lstm_dom >= bear_thr)

        bl = boost_long_zone & bull_ok
        bs = boost_short_zone & bear_ok
        if proportional:
            b_amt = boost * np.clip((lstm_dom - bull_thr) / max(1.0 - bull_thr, 1e-6), 0.0, 1.0)
            s_amt = boost * np.clip((lstm_dom - bear_thr) / max(1.0 - bear_thr, 1e-6), 0.0, 1.0)
            p2_out[bl] += b_amt[bl]
            p0_out[bs] += s_amt[bs]
        else:
            p2_out[bl] += boost
            p0_out[bs] += boost

    if enable_penalty:
        pen_long = valid & would_long & (lstm_dir == SHORT) & (lstm_dom >= bear_thr)
        pen_short = valid & would_short & (lstm_dir == LONG) & (lstm_dom >= bull_thr)
        if proportional:
            p2_out[pen_long] -= opposite_pen * np.clip(
                (lstm_dom[pen_long] - bear_thr) / max(1.0 - bear_thr, 1e-6), 0.0, 1.0
            )
            p0_out[pen_short] -= opposite_pen * np.clip(
                (lstm_dom[pen_short] - bull_thr) / max(1.0 - bull_thr, 1e-6), 0.0, 1.0
            )
        else:
            p2_out[pen_long] -= opposite_pen
            p0_out[pen_short] -= opposite_pen

    np.clip(p0_out, 0.0, 1.0, out=p0_out)
    np.clip(p2_out, 0.0, 1.0, out=p2_out)
    return p0_out, p2_out


def apply_lstm_conf_fusion_post(
    y_pred: np.ndarray,
    conf: np.ndarray,
    lstm_proba: np.ndarray,
    tl_arr: np.ndarray,
    ts_arr: np.ndarray,
    *,
    agree_boost: float = 0.05,
    neutral_pen: float = 0.05,
    opposite_pen: float = 0.08,
    active_mask: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Post-HMM hard_consensus: adjust directional confidence, then re-gate vs HMM thr.
    """
    y_out = y_pred.copy()
    c_out = conf.astype(np.float32, copy=True)
    n = len(y_out)

    valid = np.isfinite(lstm_proba).all(axis=1) if lstm_proba is not None else np.zeros(n, bool)
    if active_mask is not None:
        valid &= active_mask

    dir_mask = (y_out == LONG) | (y_out == SHORT)
    work = valid & dir_mask
    if not work.any():
        return y_out, c_out

    idx = np.where(work)[0]
    lstm_dir = np.argmax(lstm_proba[idx], axis=1)
    sig = y_out[idx].astype(np.int32)
    adj = c_out[idx].copy()

    agree = lstm_dir == sig
    neu   = lstm_dir == FLAT
    opp   = ~(agree | neu)

    adj[agree] += agree_boost
    adj[neu]   -= neutral_pen
    adj[opp]   -= opposite_pen
    np.clip(adj, 0.0, 1.0, out=adj)

    thr = np.where(sig == LONG, tl_arr[idx], ts_arr[idx])
    fail = adj < thr
    y_out[idx[fail]] = FLAT
    c_out[idx[fail]] = 0.0
    c_out[idx[~fail]] = adj[~fail]
    return y_out, c_out


# ─── Regime-Aware FLIP Alignment ─────────────────────────────────────────────

def compute_regime_flip_delta(
    direction: int,
    hmm_regime_enc: int,
    h4_trend: float,
    regime_cfg: dict,
) -> tuple[float, str]:
    """
    Hitung penyesuaian confidence dari HMM regime-aware FLIP alignment.

    RANGING (1,2): counter-trend boost, with-trend penalty (swing mode).
    TRENDING (0,3): with-trend boost, counter-trend penalty (momentum mode).
    """
    if not regime_cfg.get("enabled", False):
        return 0.0, "flip_off"

    if direction == FLAT:
        return 0.0, "flip_skip_flat"

    if h4_trend is None or (isinstance(h4_trend, float) and np.isnan(h4_trend)):
        return 0.0, "flip_skip_nan_h4"

    h4_t = float(h4_trend)
    ranging_cfg = regime_cfg.get("ranging", {})
    trending_cfg = regime_cfg.get("trending", {})

    is_with = (direction == LONG and h4_t > 0) or (direction == SHORT and h4_t < 0)
    is_counter = (direction == LONG and h4_t < 0) or (direction == SHORT and h4_t > 0)

    if hmm_regime_enc in (1, 2):
        if is_with:
            delta = -float(ranging_cfg.get("with_trend_penalty", 0.10))
            return delta, f"ranging_with-{abs(delta):.2f}"
        if is_counter:
            delta = float(ranging_cfg.get("counter_trend_boost", 0.05))
            return delta, f"ranging_counter+{delta:.2f}"
        return 0.0, "ranging_neutral"

    if hmm_regime_enc in (0, 3):
        if is_with:
            delta = float(trending_cfg.get("with_trend_boost", 0.10))
            return delta, f"trending_with+{delta:.2f}"
        if is_counter:
            delta = -float(trending_cfg.get("counter_trend_penalty", 0.05))
            return delta, f"trending_counter-{abs(delta):.2f}"
        return 0.0, "trending_neutral"

    return 0.0, "flip_skip_unknown_regime"


def apply_flip_to_proba(
    proba: np.ndarray,
    signal_idx: int,
    hmm_regime_enc: int,
    h4_trend: float,
    regime_cfg: dict,
) -> tuple[np.ndarray, float, str]:
    """Terapkan FLIP delta ke kelas sinyal dan renormalisasi (sama pola LSTM adj)."""
    delta, label = compute_regime_flip_delta(signal_idx, hmm_regime_enc, h4_trend, regime_cfg)
    if abs(delta) < 1e-9:
        return proba, 0.0, label

    out = proba.copy().astype(np.float32)
    new_signal = float(np.clip(out[signal_idx] + delta, 0.0, 1.0))
    d = new_signal - out[signal_idx]
    other_indices = [i for i in range(len(out)) if i != signal_idx]
    other_sum = sum(out[i] for i in other_indices)
    out[signal_idx] = new_signal
    if other_sum > 0:
        for oi in other_indices:
            out[oi] = float(np.clip(out[oi] - d * (out[oi] / other_sum), 0.0, 1.0))
    total = out.sum()
    if total > 0:
        out = out / total
    return out, delta, label
