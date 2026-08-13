"""
core/evaluator.py — Trading Metrics & PnL Simulation v3
Dipakai oleh pipeline/07_evaluate.py dan pipeline/08_backtest.py

Fungsi utama:
  simulate_trades()       — simulasi trade dari fixed ATR multiple (legacy v2)
  simulate_trades_swing() — simulasi trade dari H4 Swing Points (BARU v3)
  calc_drawdown()         — max drawdown dari equity curve
  calc_consecutive_loss() — streak loss terpanjang
  calc_trade_per_month()  — rata-rata trade per bulan
  full_trading_report()   — metrik PnL lengkap
"""

import numpy as np
import pandas as pd
from typing import Optional

try:
    from tools.model.guardian_k5mom_inference import pre_tp_emergency_exit, tp_proximity
except ImportError:
    pre_tp_emergency_exit = None  # type: ignore
    tp_proximity = None  # type: ignore

from core.lgbm_score_guardian import LgbmScoreGuardParams, lgbm_score_exit

from config import (
    TP_SL_HYBRID_MODE, TP_SL_SWING_FRESHNESS, TP_SL_STRUCTURAL_FILTER,
    TP_SL_RR_GATE_ENABLED, TP_SL_MIN_RR, TP_SL_MIN_TP, TP_SL_MAX_SL,
    TP_SL_FALLBACK_TP, TP_SL_FALLBACK_SL,
    TP_SL_SLIPPAGE_ENABLED, TP_SL_TRIGGER_MODE,
    TP_SL_SIZING_MODE, TP_SL_COOLDOWN_ENABLED,
    TP_SL_STRUCTURAL_TOLERANCE,
    TP_SL_VOLR_CONDITIONAL_ENABLED, TP_SL_VOLR_THRESHOLD,
    TP_SL_MAX_SL_VOLR_LOW, TP_SL_VOLR_DISABLE_MAX_SL,
    TP_SL_MAX_SL_PCT_ENABLED, TP_SL_MAX_SL_PCT,
    TP_SL_MAX_SWING_DEVIATION_PCT, TP_SL_INDIVIDUAL_SWING_FRESHNESS,
    TP_SL_SIZING_WITH_TREND_HALF,
    GUARDIAN_ENABLED, GUARDIAN_EXIT_THRESHOLD, GUARDIAN_SL_EXIT_THRESHOLD,
    GUARDIAN_SL_SAFETY_ATR, GUARDIAN_TP_ATR,
    GUARDIAN_MIN_HOLD_BARS, GUARDIAN_ACTIVATION_ATR,
    GUARDIAN_PARTIAL_EXIT_RATIO, GUARDIAN_MOMENTUM_FLOOR_FRAC,
    TRAILING_STOP_ENABLED, TRAILING_STOP_ATR, TRAILING_STOP_MIN_BARS,
)
from core.utils import setup_logger

logger = setup_logger("evaluator")


# ─── Simulasi Trade (ATR Fixed Multiple - Legacy v2) ─────────────────────────

def simulate_trades(
    y_pred:       np.ndarray,
    close:        np.ndarray,
    atr:          np.ndarray,
    modal:        float = 100.0,
    leverage:     float = 5.0,
    fee_per_side: float = 0.0004,
    slippage:     float = 0.0005,
    tp_mult:      float = 2.0,
    sl_mult:      float = 1.0,
    max_hold:     int   = 24,
    min_hold:     int   = 2,
) -> dict:
    y_pred = np.asarray(y_pred, dtype=np.int32)
    close  = np.asarray(close,  dtype=np.float64)
    atr    = np.asarray(atr,    dtype=np.float64)
    n      = len(y_pred)

    equity_curve  = np.zeros(n, dtype=np.float64)
    pnl_per_trade = []
    trade_log     = []
    cumulative    = 0.0
    total_fee     = 0.0
    wins = losses = time_exits = 0
    win_long = win_short = loss_long = loss_short = 0

    last_exit_bar = -1

    i = 0
    while i < n:
        pred = y_pred[i]
        if pred == 1 or (i - last_exit_bar) < min_hold:
            equity_curve[i] = cumulative
            i += 1
            continue

        raw_entry   = close[i]
        atr_i       = atr[i]

        if np.isnan(raw_entry) or np.isnan(atr_i) or atr_i == 0 or raw_entry == 0:
            equity_curve[i] = cumulative
            i += 1
            continue

        # Apply slippage on entry — LONG buy at ask, SHORT sell at bid
        if pred == 2:  # LONG
            entry_price = raw_entry * (1.0 + slippage)
        else:  # SHORT
            entry_price = raw_entry * (1.0 - slippage)

        if pred == 2:  # LONG
            tp_price = entry_price + tp_mult * atr_i
            sl_price = entry_price - sl_mult * atr_i
        else:          # SHORT
            tp_price = entry_price - tp_mult * atr_i
            sl_price = entry_price + sl_mult * atr_i

        fee = 2 * fee_per_side * modal
        outcome   = "time_exit"
        exit_bar  = min(i + max_hold, n - 1)
        exit_price = close[exit_bar]

        for j in range(i + 1, min(i + max_hold + 1, n)):
            if np.isnan(close[j]):
                continue

            est_high = close[j] + 0.5 * (atr[j] if not np.isnan(atr[j]) else atr_i)
            est_low  = close[j] - 0.5 * (atr[j] if not np.isnan(atr[j]) else atr_i)

            if pred == 2:  # LONG
                if est_high >= tp_price and est_low <= sl_price:
                    outcome  = "win" if close[j] >= entry_price else "loss"
                elif est_high >= tp_price:
                    outcome = "win"
                elif est_low <= sl_price:
                    outcome = "loss"
            else:  # SHORT
                if est_low <= tp_price and est_high >= sl_price:
                    outcome = "win" if close[j] <= entry_price else "loss"
                elif est_low <= tp_price:
                    outcome = "win"
                elif est_high >= sl_price:
                    outcome = "loss"

            if outcome in ("win", "loss"):
                exit_bar = j
                exit_price = close[j]
                break

        # Apply slippage on exit — LONG sell at bid, SHORT buy to cover at ask
        if pred == 2:  # LONG exit = sell
            exit_price = exit_price * (1.0 - slippage)
        else:  # SHORT exit = buy to cover
            exit_price = exit_price * (1.0 + slippage)

        tp_pct = (tp_mult * atr_i) / entry_price
        sl_pct = (sl_mult * atr_i) / entry_price

        if outcome == "win":
            trade_pnl = tp_pct * leverage * modal - fee
            wins += 1
            if pred == 2: win_long  += 1
            else:         win_short += 1
        elif outcome == "loss":
            trade_pnl = -(sl_pct * leverage * modal) - fee
            losses += 1
            if pred == 2: loss_long  += 1
            else:         loss_short += 1
        else:  # time_exit
            if pred == 2:
                actual_ret = (exit_price - entry_price) / entry_price
            else:
                actual_ret = (entry_price - exit_price) / entry_price
            trade_pnl = actual_ret * leverage * modal - fee
            time_exits += 1
            if trade_pnl >= 0:
                wins += 1
                if pred == 2: win_long  += 1
                else:         win_short += 1
            else:
                losses += 1
                if pred == 2: loss_long  += 1
                else:         loss_short += 1

        cumulative += trade_pnl
        total_fee  += fee
        pnl_per_trade.append(trade_pnl)

        trade_log.append({
            "entry_bar":   int(i),
            "exit_bar":    int(exit_bar),
            "pred":        int(pred),
            "outcome":     outcome,
            "entry_price": round(float(entry_price), 6),
            "exit_price":  round(float(exit_price), 6),
            "pnl":         round(float(trade_pnl), 4),
        })

        for k in range(i, min(exit_bar + 1, n)):
            equity_curve[k] = cumulative

        last_exit_bar = exit_bar
        i = exit_bar + 1

    if n > 0:
        last_val = equity_curve[last_exit_bar] if last_exit_bar >= 0 else 0.0
        for k in range(last_exit_bar + 1, n):
            equity_curve[k] = last_val

    total_trades = wins + losses
    winrate = round(wins / total_trades, 4) if total_trades > 0 else 0.0

    wl  = win_long  + loss_long
    ws  = win_short + loss_short
    win_by_class = {
        "LONG":  round(win_long  / wl, 4) if wl > 0 else 0.0,
        "SHORT": round(win_short / ws, 4) if ws > 0 else 0.0,
    }

    return {
        "equity_curve":   equity_curve.tolist(),
        "pnl_per_trade":  pnl_per_trade,
        "trade_log":      trade_log,
        "total_pnl":      round(float(cumulative), 4),
        "total_trades":   total_trades,
        "wins":           wins,
        "losses":         losses,
        "time_exits":     time_exits,
        "total_fee_paid": round(float(total_fee), 4),
        "winrate":        winrate,
        "win_by_class":   win_by_class,
    }


# ─── Guardian Helper ────────────────────────────────────────────────────────

def _compute_guardian_dynamic(
    bars_held: int, entry_price: float, current_price: float,
    direction: int, atr_val: float, max_favorable_pnl: float,
    g_static_current: np.ndarray | None = None,
    g_static_entry:   np.ndarray | None = None,
    delta_map: dict | None = None,
) -> np.ndarray:
    """Compute dynamic trade-context features for guardian per-bar check.

    7 base features + optional 5 delta features (IC-validated).
    Delta features computed as g_static_current[src_idx] - g_static_entry[src_idx].
    """
    pnl_pct = (current_price - entry_price) / entry_price
    if direction == 0:  # SHORT
        pnl_pct = -pnl_pct

    bars_held_norm = bars_held / 24.0
    current_pnl_atr = pnl_pct * entry_price / atr_val if atr_val > 0 else 0.0
    dd_from_peak = (
        (max_favorable_pnl - pnl_pct) / max_favorable_pnl
        if max_favorable_pnl > 0.001 else 0.0
    )
    entry_ratio = entry_price / current_price if current_price > 0 else 1.0

    base = np.array([
        bars_held_norm, pnl_pct, current_pnl_atr, max_favorable_pnl,
        dd_from_peak, 1.0 if direction == 2 else 0.0, entry_ratio,
    ], dtype=np.float64)

    # Delta features (IC-validated)
    if g_static_current is not None and g_static_entry is not None and delta_map:
        deltas = np.array([
            float(g_static_current[sidx] - g_static_entry[sidx])
            if sidx is not None and sidx < len(g_static_current) else 0.0
            for sidx in delta_map.values()
        ], dtype=np.float64)
        return np.concatenate([base, deltas])
    return base


def _assemble_guardian_row(
    feat_cols: list,
    static_names: list,
    g_static_cur: np.ndarray,
    g_static_ent: np.ndarray,
    bars_held: int,
    entry_price: float,
    current_price: float,
    direction: int,
    atr_val: float,
    max_favorable_pnl: float,
    flow_momentum_3bar: float = 0.0,
    lgbm_entry_conf: float = 0.0,
    lgbm_conf_now: float | None = None,
    tp_phase: float | None = None,
) -> np.ndarray:
    """Build guardian feature row in feat_cols order (static snapshot + dynamic)."""
    pnl_pct = (current_price - entry_price) / entry_price
    if direction == 0:
        pnl_pct = -pnl_pct

    bars_held_norm = bars_held / 24.0
    current_pnl_atr = pnl_pct * entry_price / atr_val if atr_val > 0 else 0.0
    dd_from_peak = (
        (max_favorable_pnl - pnl_pct) / max_favorable_pnl
        if max_favorable_pnl > 0.001 else 0.0
    )
    entry_ratio = entry_price / current_price if current_price > 0 else 1.0

    static_idx = {n: i for i, n in enumerate(static_names)}
    cvd_cur = g_static_cur[static_idx["cvd_slope_h4"]] if "cvd_slope_h4" in static_idx else 0.0
    cvd_ent = g_static_ent[static_idx["cvd_slope_h4"]] if "cvd_slope_h4" in static_idx else 0.0
    ofi_cur = g_static_cur[static_idx["ofi_h4_delta"]] if "ofi_h4_delta" in static_idx else 0.0
    ofi_ent = g_static_ent[static_idx["ofi_h4_delta"]] if "ofi_h4_delta" in static_idx else 0.0

    atr_pct = atr_val / entry_price if entry_price > 0 else 0.01
    mfe_atr_ratio = max_favorable_pnl / atr_pct if atr_pct > 0 else 0.0

    lec = float(lgbm_entry_conf)
    lcn = float(lgbm_conf_now) if lgbm_conf_now is not None else lec
    dynamic_vals = {
        "bars_held_norm": bars_held_norm,
        "current_pnl_pct": pnl_pct,
        "current_pnl_atr": current_pnl_atr,
        "max_favorable_pnl_pct": max_favorable_pnl,
        "drawdown_from_peak_pct": dd_from_peak,
        "direction": 1.0 if direction == 2 else 0.0,
        "entry_price_ratio": entry_ratio,
        "cvd_slope_h4_delta_entry": float(cvd_cur - cvd_ent),
        "ofi_h4_delta_entry": float(ofi_cur - ofi_ent),
        "flow_momentum_3bar": float(flow_momentum_3bar),
        "lgbm_entry_conf": lec,
        "lgbm_conf_now": lcn,
        "lgbm_conf_delta": lcn - lec,
        "lgbm_conf_ratio": lcn / lec if lec > 0.001 else 1.0,
        "mfe_atr_ratio": float(mfe_atr_ratio),
    }
    if tp_phase is not None:
        dynamic_vals["tp_phase"] = float(tp_phase)

    row = np.zeros(len(feat_cols), dtype=np.float64)
    for i, col in enumerate(feat_cols):
        if col in static_idx:
            row[i] = g_static_cur[static_idx[col]]
        elif col in dynamic_vals:
            row[i] = dynamic_vals[col]
        else:
            row[i] = 0.0
    return row


def _compute_guardian_delta_vector(
    bar: int, entry_bar: int,
    raw_features: dict,       # {feat_name: np.ndarray}
    gd_feats: list,           # ordered feature names model expects
    close_arr: np.ndarray,
    entry_price: float,
    direction: int,           # 1=LONG, -1=SHORT
    momentum_window: int = 3,
) -> np.ndarray:
    """Build Guardian Delta feature vector (delta + curr + rolling momentum + context)."""
    n = len(close_arr)
    features: dict = {}

    for f in gd_feats:
        # Skip context features — handled separately below
        if f in ("pnl_pct", "bars_held", "direction", "price_momentum_3bar"):
            continue
        arr = raw_features.get(f)  # exact match (e.g. ofi_z_score_delta)
        if arr is not None:
            v = arr[bar] if bar < len(arr) else 0.0
            features[f] = 0.0 if np.isnan(v) else float(v)
            continue
        # Derived: _delta  →  curr - entry
        if f.endswith("_delta"):
            base = f[:-6]
            src = raw_features.get(base)
            if src is not None:
                nb = len(src)
                ev = src[entry_bar] if entry_bar < nb else 0.0
                cv = src[bar]       if bar < nb else 0.0
                ev = 0.0 if np.isnan(ev) else ev
                cv = 0.0 if np.isnan(cv) else cv
                features[f] = cv - ev
            else:
                features[f] = 0.0
        # Derived: _curr  →  current value
        elif f.endswith("_curr"):
            base = f[:-5]
            src = raw_features.get(base)
            if src is not None:
                cv = src[bar] if bar < len(src) else 0.0
                features[f] = 0.0 if np.isnan(cv) else float(cv)
            else:
                features[f] = 0.0
        # Derived: _mean3 / _trend3 (or other window sizes)
        elif f"_mean{momentum_window}" in f or f"_trend{momentum_window}" in f:
            tag = f"_mean{momentum_window}" if f"_mean{momentum_window}" in f else f"_trend{momentum_window}"
            base = f[:f.index(tag)]
            src = raw_features.get(base)
            if src is not None:
                nb = len(src)
                start = max(entry_bar, bar - momentum_window + 1)
                vals = [src[b] for b in range(start, bar + 1)
                        if b < nb and not np.isnan(src[b])]
                if vals:
                    if "_mean" in tag:
                        features[f] = float(np.mean(vals))
                    else:
                        features[f] = float(vals[-1] - vals[0]) if len(vals) >= 2 else 0.0
                else:
                    features[f] = 0.0
            else:
                features[f] = 0.0
        else:
            features[f] = 0.0

    # Context features
    bars_held = bar - entry_bar
    pnl_pct   = (close_arr[bar] - entry_price) / entry_price * direction if entry_price > 0 else 0.0
    if bars_held >= 3:
        pch = [(close_arr[b] - close_arr[b-1]) / close_arr[b-1] * direction
               for b in range(entry_bar + 1, bar + 1) if b < n]
        price_mom = float(np.mean(pch[-3:])) if len(pch) >= 3 else 0.0
    else:
        price_mom = 0.0
    features["pnl_pct"]             = pnl_pct
    features["bars_held"]           = float(bars_held)
    features["direction"]           = float(direction)
    features["price_momentum_3bar"] = price_mom

    return np.array([features.get(f, 0.0) for f in gd_feats], dtype=np.float64)


# ─── ★ BARU v3: Simulasi Trade (Dinamis dari H4 Swing Points) ────────────────

def simulate_trades_swing(
    y_pred:          np.ndarray,
    close:           np.ndarray,
    high:            np.ndarray,
    low:             np.ndarray,
    atr:             np.ndarray,
    h4_swing_highs:  np.ndarray,   # swing high H4, aligned ke base tf
    h4_swing_lows:   np.ndarray,   # swing low  H4, aligned ke base tf
    modal:           float = 100.0,
    leverage:        float = 5.0,
    fee_per_side:    float = 0.0004,
    slippage:        float = 0.0005,
    min_rr:          float = 1.2,
    min_tp_atr:      float = 1.2,
    max_sl_atr:      float = 3.0,
    max_hold:        int   = 24,
    swing_lookback:  int   = 3,    # look-ahead bars di detect_h4_swing_points
    tp_fallback_atr: float = TP_SL_FALLBACK_TP,  # TP fallback (× ATR) jika swing NaN
    sl_fallback_atr: float = TP_SL_FALLBACK_SL,  # SL fallback (× ATR) jika swing NaN
    confidence               = None,  # np.ndarray — needed for sizing_mode="tiered"
    modal_arr                = None,  # np.ndarray — per-bar dynamic modal (overrides modal+sizing_mode)
    # ── Aspect toggles (from config.py) ────────────────────────────────────
    hybrid_mode:          bool = TP_SL_HYBRID_MODE,       # #1
    swing_freshness_check: bool = TP_SL_SWING_FRESHNESS,  # #2
    structural_filter:     bool = TP_SL_STRUCTURAL_FILTER, # #3
    slippage_enabled:      bool = TP_SL_SLIPPAGE_ENABLED,  # #7
    sl_trigger_mode:       str  = TP_SL_TRIGGER_MODE,      # #8
    sizing_mode:           str  = TP_SL_SIZING_MODE,       # #12
    cooldown_enabled:      bool = TP_SL_COOLDOWN_ENABLED,  # #15
    # Cooldown profit-only: setelah trade close dengan net_pnl>0, skip entry N bar (1 bar = 1h).
    # Kalau True, rule WIN/LOSS lama (2h/4h) diganti: hanya profit → +cooldown_profit_bars.
    cooldown_profit_only:  bool = False,
    cooldown_profit_bars:  int = 1,
    swing_sl_bumper_atr:   float = 0.5,                    # Bumper untuk mitigasi stop-hunt (0.5 ATR)
    structural_tolerance_pct: float = 0.04,                # Toleransi breakout filter struktural (4%)
    # ── NEW: Grup 1 — VolR Conditional & SL % Cap ──────────────────────────
    vol_ratio                = None,  # np.ndarray — vol_ratio_20 untuk conditional max_sl
    volr_conditional_enabled: bool = False,  # #16: enable VolR conditional max_sl
    volr_threshold:           float = 0.2,  # threshold VolR low-vol
    max_sl_volr_low:          float = 8.0,  # max_sl_atr saat low-vol (1b)
    volr_disable_max_sl:      bool = False,  # disable max_sl total di low-vol (1c)
    max_sl_pct_enabled:       bool = False,  # #17: enable SL % distance cap (1d)
    max_sl_pct:               float = 0.30,  # max SL = 30% dari entry
    min_sl_pct:               float = 0.0,   # #18: floor jarak SL (min % dari entry); 0 = off
    # sh>price & sl<price sblm pakai swing (match live paper_trading.py). Default False:
    # diuji 2026-07-12 (OOF 12,579 vs 5,415 trade; OOS 591 vs 303) -- kalau True, WR turun
    # ~6-7pp & OOS MaxDD 3x lebih buruk (trade tambahan yg lolos krn fallback ATR net NEGATIF
    # PnL). Toggle disimpan utk uji "match live literally", TAPI default TETAP perilaku lama
    # (reject saat swing basi, bukan fallback ATR) krn empirically lebih baik. Lihat EXPERIMENTS.md.
    swing_sidedness_check:  bool = False,
    # Opsional: batasi kapan swing_sidedness_check aktif berdasar regime HMM per-bar.
    # None = aktif semua bar (kalau swing_sidedness_check=True). Kalau diisi array kode
    # regime + swing_sidedness_active_states, check cuma aktif di state itu (mis. TRENDING
    # saja) -- di state lain, balik ke perilaku lama (tolak via RR gate, bukan fallback ATR).
    swing_sidedness_regime_arr:    np.ndarray = None,
    swing_sidedness_active_states: tuple = None,
    # ── NEW: Grup 3 — Swing Freshness ──────────────────────────────────────
    max_swing_deviation_pct:       float = 0.15,  # #19: max deviasi swing (3b)
    individual_swing_freshness:    bool = False,  # #20: cek freshness per swing (3c)
    # ── NEW: Grup 4 — Conditional Sizing ───────────────────────────────────
    h4_trend                   = None,  # np.ndarray — h4_trend untuk with-trend detection
    sizing_with_trend_half: bool = False,  # #21: half-size untuk with-trend (4b)
    # ── Exit Guardian (3rd Model) ─────────────────────────────────────────
    guardian_model            = None,
    guardian_scaler           = None,
    X_guardian                = None,
    guardian_exit_threshold   = GUARDIAN_EXIT_THRESHOLD,
    guardian_sl_exit_threshold = GUARDIAN_SL_EXIT_THRESHOLD,
    guardian_sl_safety_atr    = GUARDIAN_SL_SAFETY_ATR,
    guardian_tp_atr           = GUARDIAN_TP_ATR,
    guardian_min_hold_bars    = GUARDIAN_MIN_HOLD_BARS,
    guardian_activation_atr   = GUARDIAN_ACTIVATION_ATR,
    guardian_enabled          = GUARDIAN_ENABLED,
    guardian_momentum_floor_frac = GUARDIAN_MOMENTUM_FLOOR_FRAC,  # post-TP trailing floor
    guardian_momentum_floor_tp_frac = 0.0,  # opsional: floor tambahan = frac x TP_pnl (jaminan
    # minimum stabil, independen dari MFE). Floor final = max(floor_frac*mfe_pnl, tp_frac*tp_pnl)
    # -- KECUALI guardian_floor_replace_with_tp=True (lihat bawah).
    # Default 0.0 = OFF, perilaku identik sebelumnya. Lihat EXPERIMENTS.md 2026-07-12.
    guardian_floor_replace_with_tp = False,  # True: floor JADI tp_frac*TP_pnl SAJA (fixed,
    # tidak ikut MFE) -- guardian_momentum_floor_frac diabaikan total. Eksekusi live: exchange-side
    # STOP-LIMIT dipasang begitu TP tersentuh, bukan trailing. Lihat EXPERIMENTS.md 2026-07-12.
    guardian_floor_intrabar = False,  # PARITY EKSEKUSI FLOOR (2026-08-13). False (default lama):
    # floor dicek di CLOSE bar & exit di close[j] -- padahal live memasang STOP-LIMIT resting di
    # bursa (paper_trading.py::_place_floor_stop, trigger=limit=floor_price) yang ter-trigger
    # INTRABAR dan fill DI level floor. Model lama sistematis MEREMEHKAN floor: exit di close[j]
    # yang menurut definisi sudah DI BAWAH floor (itu syarat trigger-nya). Bandingkan SL yang
    # SUDAH dimodelkan benar (raw_exit = sl_price, bukan close). True = model floor seperti SL:
    # trigger saat wick menyentuh floor_price, exit DI floor_price. Lihat memory
    # `feedback-uji-resolusi-eksekusi-asli` -- mekanisme reaktif-waktu diuji di resolusi kasar
    # bisa MEMBALIK arah kesimpulan, bukan sekadar kurang presisi.
    guardian_feat_cols        = None,   # full feature order for custom guardian models
    guardian_static_names     = None,   # static column names matching X_guardian
    flow_momentum_arr         = None,   # optional per-bar flow_momentum_3bar array
    lgbm_conf_arr             = None,   # optional per-bar LGBM confidence at entry bar
    lgbm_p0_arr               = None,   # hourly SHORT conf — for lgbm_conf_now features
    lgbm_p2_arr               = None,   # hourly LONG conf — for lgbm_conf_now features
    # ── Guardian Delta mode (binary, delta features) ─────────────────────
    guardian_delta_raw        = None,   # dict {feat_name: np.ndarray}
    guardian_delta_feats      = None,   # list[str] — model's feature order
    guardian_mom_thresh       = 0.45,   # P(EXIT) to confirm exit AFTER TP
    guardian_def_thresh       = 0.70,   # P(EXIT) to cut early loss
    guardian_def_min_loss     = -0.010, # min pnl_pct before defense activates
    guardian_binary_peak      = False,  # binary HOLD/EXIT — post-TP escort only
    guardian_emergency_pre_tp = False,  # k5mom: pre-TP exit on LGBM conf collapse (darurat)
    guardian_emergency_conf_delta       = -0.10,
    guardian_emergency_conf_delta_near_tp = -0.16,
    guardian_near_tp_proximity          = 0.85,
    guardian_emergency_min_loss         = -0.004,
    # ── Hybrid dual Guardian (pre-TP 3-class + post-TP binary peak) ───
    guardian_hybrid_dual          = False,
    guardian_pre_model            = None,
    guardian_pre_scaler           = None,
    guardian_peak_model           = None,
    guardian_peak_scaler          = None,
    guardian_pre_feat_cols        = None,
    guardian_peak_feat_cols       = None,
    guardian_pre_exit_threshold   = None,
    guardian_pre_exit_min_pnl     = None,   # pre-TP EXIT only if cur_pnl >= this (L1 emergency bypass)
    guardian_peak_exit_threshold  = None,
    # ── LGBM score guardian (rule-based, no ML model) ─────────────────
    guardian_lgbm_score_enabled   = False,
    lgbm_score_min_hold_bars      = 2,
    lgbm_score_activation_atr     = 0.5,
    lgbm_score_wrong_entry_conf_delta = -0.12,
    lgbm_score_wrong_entry_min_loss = -0.002,
    lgbm_score_conf_dead_floor    = 0.42,
    lgbm_score_profit_take_min_pnl = 0.008,
    lgbm_score_profit_take_conf_delta = -0.08,
    lgbm_score_near_tp_proximity  = 0.85,
    lgbm_score_near_tp_conf_delta = -0.06,
    lgbm_score_near_tp_min_pnl    = 0.004,
    lgbm_score_profit_take_enabled = True,
    # ── Trailing Stop ─────────────────────────────────────────────────
    trailing_stop_enabled     = TRAILING_STOP_ENABLED,
    trailing_stop_atr         = TRAILING_STOP_ATR,
    trailing_stop_min_bars    = TRAILING_STOP_MIN_BARS,
    # ── VCB (Volatility Circuit Breaker) ── match production signal_filter.py
    vcb_enabled:          bool  = False,  # OFF — backtest benchmark
    vcb_atr_multiplier:   float = 3.0,
    vcb_lookback_bars:    int   = 24,
    # ── Pyramiding ── match production signal_filter.py
    pyramiding_enabled:     bool = False,
    pyramiding_max_per_coin: int = 1,
    pyramiding_same_dir:    bool = True,
    pyramiding_exit_mode:   str = "independent",  # independent | shared_sl_first | close_with_first | scale_in
    # Gate tambahan khusus mode "scale_in": leg baru cuma boleh nambah kalau harga sudah
    # bergerak MERUGIKAN posisi (dari VWAP saat ini) minimal sekian persen -- niatnya
    # koreksi entry yang wrong-timing, bukan nambah ke posisi yang sudah untung. 0.0 = off.
    pyramiding_min_adverse_pct: float = 0.0,
    # Gate waktu (mode "scale_in" saja): leg baru cuma boleh nambah kalau sudah minimal N bar
    # (1 bar = 1h) sejak leg TERAKHIR ditambahkan. 0 = off. Independen dari pyramiding_min_adverse_pct
    # (gate harga) -- bisa dipakai bareng atau sendiri-sendiri.
    pyramiding_min_bars_gap:  int = 0,
    entry_price_override     = None,  # np.ndarray — harga entry M15 per bar H1 (opsional)
) -> dict:
    """
    Simulasi trade dengan TP/SL dinamis — 2-tier priority:

    Tier 1: H4 Swing Points
      TP = swing high/low H4 terdekat, SL = swing low/high H4 terdekat

    Tier 2: Fallback ATR Fixed (swing NaN)
      TP = tp_fallback_atr × ATR, SL = sl_fallback_atr × ATR

    Aspect toggles (lihat config.py untuk default):
      hybrid_mode          — #1: max(swing,ATR) TP / min(swing,ATR) SL
      swing_freshness_check — #2: tolak trade jika deviasi swing > max_swing_deviation_pct
      structural_filter     — #3: entry harus dalam [H4 Low, H4 High]
      slippage_enabled      — #7: apply slippage entry/exit
      sl_trigger_mode       — #8: "close" = close candle, "highlow" = high/low candle
      sizing_mode           — #12: "fixed"=$100, "tiered"=confidence-based
      cooldown_enabled      — #15: cooldown setelah exit (tp=2h, sl=4h, time=2h)
      volr_conditional_enabled — #16: longgarkan max_sl saat VolR < threshold
      max_sl_pct_enabled    — #17: batas SL berbasis % dari entry
      max_swing_deviation_pct — #19: max deviasi swing H4
      individual_swing_freshness — #20: cek masing-masing swing level
      sizing_with_trend_half — #21: half-size untuk with-trend

    Semua tier melalui validasi R:R yang sama — skip trade jika gagal.
    """
    if pyramiding_enabled and pyramiding_exit_mode == "scale_in":
        # NOTE: cuma dukung jalur Guardian "plain" (feat_cols+static_names) -- lihat
        # docstring core/scale_in_sim.py. Kalau stack pakai hybrid/lgbm_score/binary_peak/
        # delta/trailing_stop, porting dulu sebelum pakai mode ini.
        from core.scale_in_sim import run_scale_in_simulation
        return run_scale_in_simulation(
            y_pred=y_pred, close=close, high=high, low=low, atr=atr,
            h4_swing_highs=h4_swing_highs, h4_swing_lows=h4_swing_lows,
            modal=modal, leverage=leverage, fee_per_side=fee_per_side, slippage=slippage,
            min_rr=min_rr, min_tp_atr=min_tp_atr, max_sl_atr=max_sl_atr, max_hold=max_hold,
            tp_fallback_atr=tp_fallback_atr, sl_fallback_atr=sl_fallback_atr,
            confidence=confidence, modal_arr=modal_arr,
            vol_ratio=vol_ratio, volr_conditional_enabled=volr_conditional_enabled,
            volr_threshold=volr_threshold, max_sl_volr_low=max_sl_volr_low,
            volr_disable_max_sl=volr_disable_max_sl,
            max_sl_pct_enabled=max_sl_pct_enabled, max_sl_pct=max_sl_pct,
            max_swing_deviation_pct=max_swing_deviation_pct,
            individual_swing_freshness=individual_swing_freshness,
            hybrid_mode=hybrid_mode, swing_freshness_check=swing_freshness_check,
            structural_filter=structural_filter,
            structural_tolerance_pct=structural_tolerance_pct,
            slippage_enabled=slippage_enabled, sl_trigger_mode=sl_trigger_mode,
            sizing_mode=sizing_mode, cooldown_enabled=cooldown_enabled,
            swing_sl_bumper_atr=swing_sl_bumper_atr,
            guardian_model=guardian_model, guardian_scaler=guardian_scaler,
            X_guardian=X_guardian, guardian_exit_threshold=guardian_exit_threshold,
            guardian_min_hold_bars=guardian_min_hold_bars,
            guardian_activation_atr=guardian_activation_atr, guardian_enabled=guardian_enabled,
            guardian_feat_cols=guardian_feat_cols, guardian_static_names=guardian_static_names,
            flow_momentum_arr=flow_momentum_arr, lgbm_conf_arr=lgbm_conf_arr,
            guardian_momentum_floor_frac=guardian_momentum_floor_frac,
            guardian_momentum_floor_tp_frac=guardian_momentum_floor_tp_frac,
            guardian_floor_replace_with_tp=guardian_floor_replace_with_tp,
            min_sl_pct=min_sl_pct,
            vcb_enabled=vcb_enabled, vcb_atr_multiplier=vcb_atr_multiplier,
            vcb_lookback_bars=vcb_lookback_bars,
            pyramiding_max_per_coin=pyramiding_max_per_coin,
            pyramiding_same_dir=pyramiding_same_dir,
            pyramiding_min_adverse_pct=pyramiding_min_adverse_pct,
            pyramiding_min_bars_gap=pyramiding_min_bars_gap,
            sizing_with_trend_half=sizing_with_trend_half, h4_trend=h4_trend,
            entry_price_override=entry_price_override,
        )

    n          = len(close)
    trades     = []
    equity     = modal
    equity_curve = [equity]
    n_vcb_blocked = 0

    LONG, SHORT, FLAT = 2, 0, 1   # sesuai LABEL_MAP
    cooldown_until = -1            # #15: bar index sampai kapan skip entry
    open_positions = []            # pyramiding: list of {exit_bar, sig, sl_price, entry_bar}

    for i in range(n - 1):
        sig = y_pred[i]

        # ── #15 Cooldown check ──────────────────────────────────────────
        if cooldown_enabled and i < cooldown_until:
            equity_curve.append(equity)
            continue

        if sig == FLAT:
            equity_curve.append(equity)
            continue

        if entry_price_override is not None and np.isfinite(entry_price_override[i]):
            raw_price = float(entry_price_override[i])
        else:
            raw_price = close[i]
        # Apply slippage on entry — LONG buy at ask, SHORT sell at bid
        if slippage_enabled:
            if sig == LONG:
                price = raw_price * (1.0 + slippage)
            else:
                price = raw_price * (1.0 - slippage)
        else:
            price = raw_price
        atr_i  = atr[i]

        # ── #12 Sizing ───────────────────────────────────────────────────
        if modal_arr is not None:
            trade_modal = float(modal_arr[i])
        elif sizing_mode == "tiered" and confidence is not None:
            conf_i = confidence[i]
            if conf_i > 0.75:
                trade_modal = modal
            elif conf_i > 0.60:
                trade_modal = modal * 0.5
            else:
                equity_curve.append(equity)
                continue
            # ── #21: Sizing with-trend half ──────────────────────────────
            if sizing_with_trend_half and h4_trend is not None:
                trend_i = h4_trend[i]
                if not np.isnan(trend_i):
                    is_with_trend = (sig == LONG and trend_i > 0) or (sig == SHORT and trend_i < 0)
                    if is_with_trend:
                        trade_modal = trade_modal * 0.5
        else:
            trade_modal = modal
        sh_i   = h4_swing_highs[i]
        sl_i   = h4_swing_lows[i]

        if np.isnan(price) or np.isnan(atr_i) or atr_i == 0:
            equity_curve.append(equity)
            continue

        # ── VCB (Volatility Circuit Breaker) ── match production signal_filter.py
        if vcb_enabled and atr_i > 0:
            vcb_start = max(0, i - vcb_lookback_bars)
            vcb_window = atr[vcb_start:i+1]
            vcb_valid = vcb_window[~np.isnan(vcb_window)]
            if len(vcb_valid) >= vcb_lookback_bars:
                vcb_mean = vcb_valid[-vcb_lookback_bars:].mean()
                if atr_i > vcb_atr_multiplier * vcb_mean:
                    n_vcb_blocked += 1
                    equity_curve.append(equity)
                    continue

        # Swing valid hanya kalau posisinya masih masuk akal thd harga entry —
        # swing high di ATAS & swing low di BAWAH price (kalau tidak, struktur H4
        # basi/sudah ditembus; match live paper_trading.py::_calculate_tp_sl yg
        # sudah validasi sh>entry & sl<entry sebelum pakai swing, fallback ke ATR
        # kalau gagal — evaluator ini sebelumnya TIDAK validasi sisi, pakai swing
        # basi apa adanya sehingga SL bisa jadi sangat lebar & RR gagal padahal
        # live lolos via ATR fallback (fix 2026-07-12).
        use_swing = not np.isnan(sh_i) and not np.isnan(sl_i)
        if use_swing and swing_sidedness_check:
            _check_active = True
            if swing_sidedness_regime_arr is not None and swing_sidedness_active_states is not None:
                _check_active = int(swing_sidedness_regime_arr[i]) in swing_sidedness_active_states
            if _check_active:
                use_swing = sh_i > price and sl_i < price

        # ── #2 Swing Freshness Check ────────────────────────────────────
        if swing_freshness_check and use_swing:
            if individual_swing_freshness:
                # #20: Cek masing-masing swing individually — cegah TONUSDT leak
                high_dev = abs(sh_i - price) / price
                low_dev  = abs(sl_i - price) / price
                if high_dev > max_swing_deviation_pct or low_dev > max_swing_deviation_pct:
                    equity_curve.append(equity)
                    continue
            else:
                swing_dev = abs(sh_i - price) / price if sig == LONG else abs(sl_i - price) / price
                if swing_dev > max_swing_deviation_pct:
                    equity_curve.append(equity)
                    continue

        # ── #3 Structural Filter ────────────────────────────────────────
        if structural_filter and use_swing:
            # Toleransi: entry diizinkan menembus swing max sebesar tolerance_pct
            upper_bound = sh_i * (1.0 + structural_tolerance_pct)
            lower_bound = sl_i * (1.0 - structural_tolerance_pct)
            if price > upper_bound or price < lower_bound:  # entry di luar [H4 Low, H4 High] + tolerance
                equity_curve.append(equity)
                continue

        # ── Pyramiding check ── match production (separate Trade per leg) ──
        open_positions = [p for p in open_positions if p["exit_bar"] > i]
        if pyramiding_enabled and open_positions:
            existing_dir = "LONG" if open_positions[0]["sig"] == LONG else "SHORT"
            new_dir = "LONG" if sig == LONG else "SHORT"
            if new_dir != existing_dir:
                equity_curve.append(equity)
                continue
            if len(open_positions) >= pyramiding_max_per_coin:
                equity_curve.append(equity)
                continue
        elif open_positions:
            equity_curve.append(equity)
            continue

        # ── Tentukan TP/SL — Gate: Swing/ATR ────────────────────────────

        if use_swing:
            if sig == LONG:
                swing_tp = sh_i
                swing_sl = sl_i - (swing_sl_bumper_atr * atr_i)
                atr_tp   = price + tp_fallback_atr * atr_i
                atr_sl   = price - sl_fallback_atr * atr_i
                if hybrid_mode:
                    tp_price = max(swing_tp, atr_tp)   # max: whichever is further above
                    sl_price = min(swing_sl, atr_sl)   # min: whichever is further below
                else:
                    tp_price = swing_tp
                    sl_price = swing_sl
                tp_dist  = tp_price - price
                sl_dist  = price    - sl_price
            else:
                swing_tp = sl_i
                swing_sl = sh_i + (swing_sl_bumper_atr * atr_i)
                atr_tp   = price - tp_fallback_atr * atr_i
                atr_sl   = price + sl_fallback_atr * atr_i
                if hybrid_mode:
                    tp_price = min(swing_tp, atr_tp)   # min: whichever is further below
                    sl_price = max(swing_sl, atr_sl)   # max: whichever is further above
                else:
                    tp_price = swing_tp
                    sl_price = swing_sl
                tp_dist  = price    - tp_price
                sl_dist  = sl_price - price
        else:
            # ── Tier 2: ATR Fallback (swing NaN) ─────────────────────────
            if sig == LONG:
                tp_price = price + tp_fallback_atr * atr_i
                sl_price = price - sl_fallback_atr * atr_i
                tp_dist  = tp_price - price
                sl_dist  = price    - sl_price
            else:
                tp_price = price - tp_fallback_atr * atr_i
                sl_price = price + sl_fallback_atr * atr_i
                tp_dist  = price    - tp_price
                sl_dist  = sl_price - price

        # ── #18 SL% Floor — lebarkan SL jika lebih dekat dari min_sl_pct dari entry ──
        # Koin low-vol (ATR rendah) menghasilkan band SL terlalu sempit → kena noise.
        # Floor ini independen dari ATR. Diterapkan SEBELUM RR gate (efek RR ikut terukur).
        if min_sl_pct > 0.0:
            floor_dist = price * min_sl_pct
            if sl_dist < floor_dist:
                sl_dist = floor_dist
                if sig == LONG:
                    sl_price = price - floor_dist
                else:
                    sl_price = price + floor_dist

        # ── Pyramiding exit schema: anchor SL / cap hold to first leg ──
        if pyramiding_enabled and open_positions and pyramiding_exit_mode == "shared_sl_first":
            sl_price = open_positions[0]["sl_price"]
            if sig == LONG:
                sl_dist = price - sl_price
            else:
                sl_dist = sl_price - price

        # ── Guardian active flag (tidak override TP/SL — pakai swing H4 / ATR fallback)
        guardian_hybrid_active = (
            guardian_hybrid_dual
            and guardian_pre_model is not None
            and guardian_peak_model is not None
            and guardian_pre_scaler is not None
            and guardian_peak_scaler is not None
            and X_guardian is not None
        )
        guardian_lgbm_score_active = (
            guardian_lgbm_score_enabled
            and lgbm_conf_arr is not None
            and lgbm_p0_arr is not None
            and lgbm_p2_arr is not None
        )
        guardian_active = (
            (guardian_enabled and guardian_model is not None and X_guardian is not None)
            or guardian_hybrid_active
            or guardian_lgbm_score_active
        )
        _lgbm_score_params = LgbmScoreGuardParams(
            min_hold_bars=lgbm_score_min_hold_bars,
            activation_atr=lgbm_score_activation_atr,
            wrong_entry_conf_delta=lgbm_score_wrong_entry_conf_delta,
            wrong_entry_min_loss=lgbm_score_wrong_entry_min_loss,
            conf_dead_floor=lgbm_score_conf_dead_floor,
            profit_take_min_pnl=lgbm_score_profit_take_min_pnl,
            profit_take_conf_delta=lgbm_score_profit_take_conf_delta,
            near_tp_proximity=lgbm_score_near_tp_proximity,
            near_tp_conf_delta=lgbm_score_near_tp_conf_delta,
            near_tp_min_pnl=lgbm_score_near_tp_min_pnl,
            profit_take_enabled=lgbm_score_profit_take_enabled,
        )
        guardian_delta_active = guardian_model is not None and guardian_delta_raw is not None and guardian_delta_feats is not None
        _pre_exit_thr = (
            guardian_pre_exit_threshold
            if guardian_pre_exit_threshold is not None
            else guardian_exit_threshold
        )
        _peak_exit_thr = (
            guardian_peak_exit_threshold
            if guardian_peak_exit_threshold is not None
            else guardian_exit_threshold
        )

        # Validasi R:R (GATE — gagal = trade di-skip)
        if TP_SL_RR_GATE_ENABLED:
            if tp_dist <= 0 or sl_dist <= 0:
                equity_curve.append(equity)
                continue
            if tp_dist < min_tp_atr * atr_i:
                equity_curve.append(equity)
                continue

            # ── #16: VolR Conditional max_sl ──────────────────────────────
            _effective_max_sl = max_sl_atr
            if volr_conditional_enabled and vol_ratio is not None:
                vr_i = vol_ratio[i]
                if not np.isnan(vr_i) and vr_i < volr_threshold:
                    if volr_disable_max_sl:
                        _effective_max_sl = float("inf")  # 1c: disable max_sl total
                    else:
                        _effective_max_sl = max_sl_volr_low  # 1b: longgarkan

            # ── #17: SL % Distance Cap (alternatif ATR-based) ─────────────
            _sl_cap = float("inf")
            if max_sl_pct_enabled:
                _sl_cap = price * max_sl_pct

            if sl_dist > min(_effective_max_sl * atr_i, _sl_cap):
                equity_curve.append(equity)
                continue
            rr = tp_dist / sl_dist
            if rr < min_rr:
                equity_curve.append(equity)
                continue
        else:
            if tp_dist <= 0 or sl_dist <= 0:
                equity_curve.append(equity)
                continue
            rr = tp_dist / sl_dist

        # ── Scan ke depan ─────────────────────────────────────────────────────
        outcome = "TIMEOUT"
        raw_exit = price
        mfe_pnl = 0.0  # max favorable excursion (PnL %)
        best_price_trail = price  # trailing stop reference price
        tp_touched = False  # TP momentum mode flag
        # Partial exit tracking
        partial_bar = None
        partial_price = None
        partial_pnl = 0.0
        position_remaining = 1.0  # 1.0 = full, 0.5 = half after partial

        end = min(i + max_hold, n)
        if pyramiding_enabled and open_positions and pyramiding_exit_mode == "close_with_first":
            end = min(end, open_positions[0]["exit_bar"] + 1)
        for j in range(i + 1, end):
            if np.isnan(close[j]):
                continue

            bars_held = j - i

            # Track max favorable excursion
            if sig == LONG:
                mfe_pnl = max(mfe_pnl, (high[j] - price) / price)
                tp_hit = high[j] >= tp_price
                sl_hit = (low[j] <= sl_price) if sl_trigger_mode == "highlow" else (close[j] <= sl_price)
            else:
                mfe_pnl = max(mfe_pnl, (price - low[j]) / price)
                tp_hit = low[j] <= tp_price
                sl_hit = (high[j] >= sl_price) if sl_trigger_mode == "highlow" else (close[j] >= sl_price)

            # ── Floor INTRABAR (parity STOP-LIMIT exchange-side) ─────────────────
            # Sengaja dicek SEBELUM SL: floor_price ada di sisi UNTUNG dari entry,
            # sl_price di sisi RUGI -- bar yang menembus SL PASTI melewati floor lebih
            # dulu, jadi order STOP-LIMIT yang resting di bursa ter-trigger duluan.
            # Model lama (cek di close, di blok bawah) memberi kemenangan ke SL pada bar
            # yang sama = bias KEDUA yang merugikan floor, di luar bias harga exit.
            # Dilewati pada bar TP pertama tersentuh (tp_touched masih False di titik
            # ini) -- realistis: order baru dipasang setelah TP terdeteksi.
            if (guardian_floor_intrabar and guardian_floor_replace_with_tp
                    and tp_touched and guardian_momentum_floor_tp_frac > 0.0):
                if sig == LONG:
                    _floor_lvl = price + guardian_momentum_floor_tp_frac * (tp_price - price)
                    _floor_hit = low[j] <= _floor_lvl
                else:
                    _floor_lvl = price - guardian_momentum_floor_tp_frac * (price - tp_price)
                    _floor_hit = high[j] >= _floor_lvl
                if _floor_hit:
                    outcome = "GUARDIAN_MOMENTUM_FLOOR"
                    raw_exit = _floor_lvl
                    break

            # ── SL hard exit ── trigger & exit price per sl_trigger_mode
            # "close"  : SL triggered saat close lewati SL → exit @ sl_price (stop order fill di SL)
            # "highlow": SL triggered saat wick sentuh SL → exit @ sl_price (bisa fill via stop order)
            if sl_hit:
                outcome = "LOSS"
                raw_exit = sl_price
                break

            # tp_phase for guardian v5 (0=pre, 1=at first hit, 2=post) — before tp_touched flip
            if tp_touched:
                _tp_phase = 2.0
            elif tp_hit:
                _tp_phase = 1.0
            else:
                _tp_phase = 0.0

            # ── TP → momentum mode (match production) ──────────────────
            # TP tidak hard-close — trigger Guardian momentum (bypass gates)
            if tp_hit and not tp_touched and guardian_active:
                tp_touched = True
            elif tp_hit and not guardian_active:
                # No Guardian → legacy hard TP close
                outcome = "WIN"; raw_exit = tp_price; break

            # ── Trailing floor (momentum mode) — kunci min frac×MFE ──────
            # Jaring pengaman: setelah lewat TP (momentum mode), bila give-back
            # menembus floor → exit walau Guardian belum memutuskan. Simetris
            # long/short (mfe_pnl & cur_pnl sudah per-arah).
            # Floor tambahan opsional (guardian_momentum_floor_tp_frac > 0): jaminan
            # minimum stabil = tp_frac x TP_pnl, independen dari seberapa jauh MFE.
            # Floor efektif = max(kedua patokan) -- MFE dekat TP -> patokan TP dominan
            # (cegah give-back di bawah TP asli); MFE jauh melebihi TP -> patokan MFE
            # tetap ambil alih (jangan capping untung di trade yang lari jauh).
            if (not guardian_floor_intrabar) and guardian_floor_replace_with_tp and tp_touched and guardian_momentum_floor_tp_frac > 0.0:
                # MODEL LAMA (guardian_floor_intrabar=False): cek di CLOSE, exit di close[j].
                # Dipertahankan supaya scorecard lama bisa direproduksi persis -- TAPI ini
                # TIDAK match eksekusi live (lihat blok intrabar di atas + catatan parameter).
                # Fixed floor = tp_frac x TP_pnl SAJA, tidak trailing ikut MFE (ganti total,
                # bukan tambahan max()) -- parity eksekusi live: STOP-LIMIT dipasang sekali
                # begitu TP tersentuh, level tetap sampai exit.
                cur_pnl = (close[j] - price) / price if sig == LONG else (price - close[j]) / price
                tp_pnl = (tp_price - price) / price if sig == LONG else (price - tp_price) / price
                floor_pnl = guardian_momentum_floor_tp_frac * tp_pnl
                if cur_pnl < floor_pnl:
                    outcome = "GUARDIAN_MOMENTUM_FLOOR"
                    raw_exit = close[j]
                    break
            elif tp_touched and guardian_momentum_floor_frac > 0.0 and mfe_pnl > 0.0:
                cur_pnl = (close[j] - price) / price if sig == LONG else (price - close[j]) / price
                floor_pnl = guardian_momentum_floor_frac * mfe_pnl
                if guardian_momentum_floor_tp_frac > 0.0:
                    tp_pnl = (tp_price - price) / price if sig == LONG else (price - tp_price) / price
                    floor_pnl = max(floor_pnl, guardian_momentum_floor_tp_frac * tp_pnl)
                if cur_pnl < floor_pnl:
                    outcome = "GUARDIAN_MOMENTUM_FLOOR"
                    raw_exit = close[j]
                    break

            # ── Guardian exit (multiclass 3-class OR binary peak escort OR hybrid) ──
            # momentum_mode = tp_touched — bypasses min_hold + activation gates
            guardian_momentum = tp_touched
            pre_tp_emergency_check = (
                guardian_binary_peak
                and guardian_emergency_pre_tp
                and not tp_touched
                and bars_held >= guardian_min_hold_bars
            )
            if guardian_active and position_remaining > 0.5:
                if guardian_lgbm_score_active:
                    lec = float(lgbm_conf_arr[i]) if i < len(lgbm_conf_arr) else 0.0
                    lcn = float(
                        lgbm_p2_arr[j] if sig == LONG else lgbm_p0_arr[j]
                    ) if j < len(lgbm_p0_arr) else None
                    cur_pnl = (
                        (close[j] - price) / price
                        if sig == LONG
                        else (price - close[j]) / price
                    )
                    price_moved_atr = abs(close[j] - price) / atr_i if atr_i > 0 else 0.0
                    score_reason = lgbm_score_exit(
                        direction_long=(sig == LONG),
                        entry_price=price,
                        current_price=close[j],
                        tp_price=tp_price,
                        entry_conf=lec,
                        conf_now=lcn,
                        cur_pnl_pct=cur_pnl,
                        mfe_pnl_pct=mfe_pnl,
                        bars_held=bars_held,
                        tp_touched=tp_touched,
                        price_moved_atr=price_moved_atr,
                        params=_lgbm_score_params,
                    )
                    if score_reason:
                        outcome = score_reason
                        raw_exit = close[j]
                        break
                elif guardian_hybrid_active:
                    should_check = guardian_momentum or bars_held >= guardian_min_hold_bars
                else:
                    should_check = guardian_momentum or pre_tp_emergency_check or (
                        not guardian_binary_peak and bars_held >= guardian_min_hold_bars
                    )
                if not guardian_lgbm_score_active and should_check:
                    price_moved_atr = abs(close[j] - price) / atr_i if atr_i > 0 else float("inf")
                    bypass_gates = guardian_momentum or pre_tp_emergency_check
                    if guardian_hybrid_active:
                        gate_ok = guardian_momentum or price_moved_atr >= guardian_activation_atr
                    else:
                        gate_ok = bypass_gates or (
                            not guardian_binary_peak and price_moved_atr >= guardian_activation_atr
                        )
                    if gate_ok:
                        g_static_cur = X_guardian[j, :]
                        g_static_ent = X_guardian[i, :]
                        fm_val = 0.0
                        if flow_momentum_arr is not None and j < len(flow_momentum_arr):
                            fm_val = float(flow_momentum_arr[j])
                        lec = float(lgbm_conf_arr[i]) if lgbm_conf_arr is not None and i < len(lgbm_conf_arr) else 0.0
                        lcn = None
                        if lgbm_p0_arr is not None and lgbm_p2_arr is not None and j < len(lgbm_p0_arr):
                            lcn = float(lgbm_p2_arr[j] if sig == LONG else lgbm_p0_arr[j])
                        def _scaled_guard_row(feat_cols, scaler):
                            if feat_cols and guardian_static_names:
                                row = _assemble_guardian_row(
                                    feat_cols, guardian_static_names,
                                    g_static_cur, g_static_ent,
                                    bars_held, price, close[j], sig, atr_i, mfe_pnl,
                                    flow_momentum_3bar=fm_val,
                                    lgbm_entry_conf=lec,
                                    lgbm_conf_now=lcn,
                                    tp_phase=_tp_phase,
                                )
                                feat = row.reshape(1, -1)
                            else:
                                try:
                                    from config import GUARDIAN_DELTA_MAP, GUARDIAN_EXTENDED_STATIC
                                    _dmap = {
                                        dname: GUARDIAN_EXTENDED_STATIC.index(src)
                                        if src in GUARDIAN_EXTENDED_STATIC else None
                                        for dname, src in GUARDIAN_DELTA_MAP.items()
                                    }
                                except Exception:
                                    _dmap = None
                                g_dynamic = _compute_guardian_dynamic(
                                    bars_held, price, close[j], sig, atr_i, mfe_pnl,
                                    g_static_cur, g_static_ent, _dmap,
                                )
                                feat = np.concatenate([g_static_cur, g_dynamic]).reshape(1, -1)
                            return (feat - scaler.mean_) / scaler.scale_

                        if guardian_hybrid_active:
                            if tp_touched:
                                peak_cols = guardian_peak_feat_cols or guardian_feat_cols
                                g_feat_s = _scaled_guard_row(peak_cols, guardian_peak_scaler)
                                p_exit = float(guardian_peak_model.predict_proba(g_feat_s)[0, 1])
                                if p_exit >= _peak_exit_thr:
                                    outcome = "GUARDIAN_MOMENTUM_EXIT"
                                    raw_exit = close[j]
                                    break
                            elif bars_held >= guardian_min_hold_bars:
                                pre_cols = guardian_pre_feat_cols or guardian_feat_cols
                                g_feat_s = _scaled_guard_row(pre_cols, guardian_pre_scaler)
                                g_proba = guardian_pre_model._Booster.predict(g_feat_s)[0]
                                g_pred = int(g_proba.argmax())
                                if g_pred == 2 and g_proba[2] >= _pre_exit_thr:
                                    outcome = "GUARDIAN_PRE_EXIT"
                                    raw_exit = close[j]
                                    break
                                if g_pred == 1 and g_proba[1] >= _pre_exit_thr and partial_bar is None:
                                    partial_bar = j
                                    partial_price = close[j]
                                    pct_partial = (partial_price - price) / price
                                    if sig == SHORT:
                                        pct_partial = -pct_partial
                                    gross_partial = trade_modal * leverage * pct_partial * GUARDIAN_PARTIAL_EXIT_RATIO
                                    fee_partial = trade_modal * leverage * fee_per_side * GUARDIAN_PARTIAL_EXIT_RATIO
                                    partial_pnl = gross_partial - fee_partial
                                    position_remaining = 1.0 - GUARDIAN_PARTIAL_EXIT_RATIO
                        elif guardian_feat_cols and guardian_static_names:
                            g_row = _assemble_guardian_row(
                                guardian_feat_cols, guardian_static_names,
                                g_static_cur, g_static_ent,
                                bars_held, price, close[j], sig, atr_i, mfe_pnl,
                                flow_momentum_3bar=fm_val,
                                lgbm_entry_conf=lec,
                                lgbm_conf_now=lcn,
                                tp_phase=_tp_phase,
                            )
                            g_feat = g_row.reshape(1, -1)
                            g_feat_s = (g_feat - guardian_scaler.mean_) / guardian_scaler.scale_
                        else:
                            try:
                                from config import GUARDIAN_DELTA_MAP, GUARDIAN_EXTENDED_STATIC
                                _dmap = {
                                    dname: GUARDIAN_EXTENDED_STATIC.index(src)
                                    if src in GUARDIAN_EXTENDED_STATIC else None
                                    for dname, src in GUARDIAN_DELTA_MAP.items()
                                }
                            except Exception:
                                _dmap = None
                            g_dynamic = _compute_guardian_dynamic(
                                bars_held, price, close[j], sig, atr_i, mfe_pnl,
                                g_static_cur, g_static_ent, _dmap,
                            )
                            g_feat = np.concatenate([g_static_cur, g_dynamic]).reshape(1, -1)
                            g_feat_s = (g_feat - guardian_scaler.mean_) / guardian_scaler.scale_

                        if guardian_hybrid_active:
                            pass
                        elif guardian_binary_peak:
                            p_exit = float(guardian_model.predict_proba(g_feat_s)[0, 1])
                            if tp_touched:
                                if p_exit >= guardian_exit_threshold:
                                    outcome = "GUARDIAN_MOMENTUM_EXIT"
                                    raw_exit = close[j]
                                    break
                            elif (
                                guardian_emergency_pre_tp
                                and bars_held >= guardian_min_hold_bars
                                and lcn is not None
                            ):
                                cur_pnl = (close[j] - price) / price if sig == LONG else (price - close[j]) / price
                                prox = tp_proximity(sig == LONG, price, close[j], tp_price)
                                cd = lcn - lec
                                if pre_tp_emergency_exit(
                                    conf_delta=cd,
                                    proximity=prox,
                                    p_exit=p_exit,
                                    cur_pnl_pct=cur_pnl,
                                    near_tp_proximity=guardian_near_tp_proximity,
                                    conf_delta_normal=guardian_emergency_conf_delta,
                                    conf_delta_near_tp=guardian_emergency_conf_delta_near_tp,
                                    exit_threshold=guardian_exit_threshold,
                                    min_loss_pct=guardian_emergency_min_loss,
                                ):
                                    outcome = "GUARDIAN_EMERGENCY_EXIT"
                                    raw_exit = close[j]
                                    break
                        else:
                            g_proba = guardian_model._Booster.predict(g_feat_s)[0]
                            g_pred = int(g_proba.argmax())
                            cur_pnl = (
                                (close[j] - price) / price
                                if sig == LONG
                                else (price - close[j]) / price
                            )
                            _thr = guardian_exit_threshold if guardian_momentum else _pre_exit_thr

                            def _pre_tp_exit_ok() -> bool:
                                if guardian_momentum or guardian_pre_exit_min_pnl is None:
                                    return True
                                if cur_pnl <= -0.006:
                                    return True
                                return cur_pnl >= guardian_pre_exit_min_pnl

                            if g_pred == 2 and g_proba[2] >= _thr and _pre_tp_exit_ok():
                                if guardian_momentum:
                                    outcome = "GUARDIAN_MOMENTUM_EXIT"
                                elif partial_bar is not None:
                                    outcome = "GUARDIAN_FULL"
                                else:
                                    outcome = "GUARDIAN_EXIT"
                                raw_exit = close[j]
                                break
                            elif g_pred == 1 and g_proba[1] >= _thr and partial_bar is None:
                                if guardian_momentum:
                                    outcome = "GUARDIAN_MOMENTUM_PARTIAL"
                                partial_bar = j
                                partial_price = close[j]
                                pct_partial = (partial_price - price) / price
                                if sig == SHORT:
                                    pct_partial = -pct_partial
                                gross_partial = trade_modal * leverage * pct_partial * GUARDIAN_PARTIAL_EXIT_RATIO
                                fee_partial = trade_modal * leverage * fee_per_side * GUARDIAN_PARTIAL_EXIT_RATIO
                                partial_pnl = gross_partial - fee_partial
                                position_remaining = 1.0 - GUARDIAN_PARTIAL_EXIT_RATIO

            # ── Guardian Delta (binary: HOLD=1 / EXIT=0) ─────────────────
            if guardian_delta_active and bars_held >= guardian_min_hold_bars and partial_bar is None:
                direction_num = 1 if sig == LONG else -1
                pnl_pct_now   = (close[j] - price) / price * direction_num if price > 0 else 0.0
                tp_hit_now    = (sig == LONG and high[j] >= tp_price) or (sig == SHORT and low[j] <= tp_price)

                run_delta = tp_hit_now or (pnl_pct_now <= guardian_def_min_loss)
                if run_delta:
                    gd_vec = _compute_guardian_delta_vector(
                        bar=j, entry_bar=i,
                        raw_features=guardian_delta_raw,
                        gd_feats=guardian_delta_feats,
                        close_arr=close,
                        entry_price=price,
                        direction=direction_num,
                        momentum_window=3,
                    ).reshape(1, -1)
                    p_exit = guardian_model.predict_proba(gd_vec)[0][0]  # P(EXIT)

                    gd_exit = False
                    if tp_hit_now and p_exit > guardian_mom_thresh:
                        gd_exit = True   # momentum mode: TP hit but Guardian says exit
                    elif pnl_pct_now <= guardian_def_min_loss and p_exit > guardian_def_thresh:
                        gd_exit = True   # defense mode: in loss, Guardian confident to cut

                    if gd_exit:
                        outcome  = "GUARDIAN_DELTA_EXIT"
                        raw_exit = close[j]
                        break

            # ── Trailing Stop ────────────────────────────────────────────
            if trailing_stop_enabled and bars_held >= trailing_stop_min_bars:
                if sig == LONG:
                    best_price_trail = max(best_price_trail, high[j])
                    trail_stop = best_price_trail - trailing_stop_atr * atr_i
                    if low[j] <= trail_stop:
                        outcome = "TRAILING_STOP"; raw_exit = trail_stop; break
                else:
                    best_price_trail = min(best_price_trail, low[j])
                    trail_stop = best_price_trail + trailing_stop_atr * atr_i
                    if high[j] >= trail_stop:
                        outcome = "TRAILING_STOP"; raw_exit = trail_stop; break

        # Apply slippage on exit — LONG sell at bid, SHORT buy to cover at ask
        if slippage_enabled:
            if sig == LONG:
                exit_price = raw_exit * (1.0 - slippage)
                partial_exit_price_adj = (partial_price * (1.0 - slippage)) if partial_price else None
            else:
                exit_price = raw_exit * (1.0 + slippage)
                partial_exit_price_adj = (partial_price * (1.0 + slippage)) if partial_price else None
        else:
            exit_price = raw_exit
            partial_exit_price_adj = partial_price

        # ── Hitung PnL (remaining position) ──────────────────────────────
        pct_move = (exit_price - price) / price
        if sig == SHORT:
            pct_move = -pct_move

        gross_pnl  = trade_modal * leverage * pct_move * position_remaining
        fee_total  = trade_modal * leverage * fee_per_side * (1.0 + position_remaining)  # entry + remaining exit
        net_pnl    = gross_pnl - fee_total + partial_pnl  # add partial exit PnL

        equity    += net_pnl
        equity_curve.append(equity)

        trade_record = {
            "bar_in":    i,
            "bar_out":   j if outcome != "TIMEOUT" else end,
            "direction": "LONG" if sig == LONG else "SHORT",
            "entry":     price,
            "exit":      exit_price,
            "tp":        tp_price,
            "sl":        sl_price,
            "rr":        round(rr, 2),
            "outcome":   ("TIMEOUT_MOMENTUM" if (outcome == "TIMEOUT" and tp_touched) else outcome),
            "net_pnl":   round(net_pnl, 4),
            "equity":    round(equity, 4),
            "modal_used": round(trade_modal, 2),
        }
        if partial_bar is not None:
            trade_record["partial_bar"] = partial_bar
            trade_record["partial_price"] = round(partial_exit_price_adj, 6) if partial_exit_price_adj else None
            trade_record["partial_pnl"] = round(partial_pnl, 4)
            trade_record["partial_ratio"] = GUARDIAN_PARTIAL_EXIT_RATIO
        else:
            trade_record["partial_bar"] = None
            trade_record["partial_pnl"] = 0.0

        trades.append(trade_record)

        exit_bar = j if outcome != "TIMEOUT" else end
        open_positions.append({
            "exit_bar": exit_bar, "sig": sig, "sl_price": sl_price, "entry_bar": i,
        })

        # ── #15 Cooldown ─────────────────────────────────────────────────
        if cooldown_enabled:
            exit_bar = j if outcome != "TIMEOUT" else end
            if cooldown_profit_only:
                # Hanya setelah close profit (net_pnl > 0) — covering WIN + guardian
                # momentum/exit yang untung. Loss/SL: boleh re-entry tanpa jeda.
                if net_pnl > 0:
                    cooldown_until = exit_bar + max(0, int(cooldown_profit_bars))
            else:
                if outcome == "WIN":
                    cooldown_until = exit_bar + 2   # tp_hit = 2h
                elif outcome == "LOSS":
                    cooldown_until = exit_bar + 4   # sl_hit = 4h
                else:
                    cooldown_until = exit_bar + 2   # time_exit / guardian / timeout = 2h

    # ── Summary & Compatibility Mapping ───────────────────────────────────────
    if not trades:
        return {
            "error": "no_trades", "total_trades": 0, "winrate": 0.0,
            "total_pnl": 0.0, "max_drawdown": 0.0, "max_drawdown_pct": 0.0,
            "equity_curve": equity_curve, "pnl_per_trade": [],
            "wins": 0, "losses": 0, "time_exits": 0,
            "win_by_class": {"LONG": 0.0, "SHORT": 0.0}
        }

    guardian_outcomes = ("TRAILING_STOP", "GUARDIAN_EXIT", "GUARDIAN_FULL",
                         "GUARDIAN_PRE_EXIT", "GUARDIAN_MOMENTUM_EXIT",
                         "GUARDIAN_MOMENTUM_PARTIAL", "GUARDIAN_EMERGENCY_EXIT",
                         "GUARDIAN_MOMENTUM_FLOOR", "GUARDIAN_DELTA_EXIT",
                         "LGBM_SCORE_CONF_DEAD", "LGBM_SCORE_WRONG_ENTRY",
                         "LGBM_SCORE_TAKE_PROFIT", "LGBM_SCORE_TP_PROXIMITY",
                         "LGBM_SCORE_MFE_GIVEBACK",
                         "TIMEOUT", "TIMEOUT_MOMENTUM")
    wins   = [t for t in trades if t["outcome"] == "WIN"
              or (t["outcome"] in guardian_outcomes and t["net_pnl"] > 0)]
    losses = [t for t in trades if t["outcome"] == "LOSS"
              or (t["outcome"] in guardian_outcomes and t["net_pnl"] <= 0)]
    time_e = [t for t in trades if t["outcome"] in ("TIMEOUT", "TIMEOUT_MOMENTUM")]

    winrate    = len(wins) / len(trades) if trades else 0.0
    avg_win    = np.mean([t["net_pnl"] for t in wins])   if wins   else 0.0
    avg_loss   = np.mean([t["net_pnl"] for t in losses]) if losses else 0.0

    profit_factor = 0.0
    sum_loss = abs(sum(t["net_pnl"] for t in losses))
    sum_win  = abs(sum(t["net_pnl"] for t in wins))
    if sum_loss > 0:
        profit_factor = sum_win / sum_loss

    equity_arr   = np.array([e for e in equity_curve if not np.isnan(e)])
    peak         = np.maximum.accumulate(equity_arr)
    drawdown     = (equity_arr - peak) / (peak + 1e-10)
    max_drawdown = float(drawdown.min())

    # Map class winrate
    lw = len([t for t in wins if t["direction"] == "LONG"])
    lt = len([t for t in trades if t["direction"] == "LONG"])
    sw = len([t for t in wins if t["direction"] == "SHORT"])
    st = len([t for t in trades if t["direction"] == "SHORT"])

    total_net_pnl = sum(t["net_pnl"] for t in trades)

    return {
        "total_trades":   len(trades),
        "winrate":        round(winrate, 4),
        "avg_win":        round(avg_win, 4),
        "avg_loss":       round(avg_loss, 4),
        "profit_factor":  round(profit_factor, 4),
        "net_pnl_total":  round(total_net_pnl, 4),
        "max_drawdown":   round(max_drawdown, 4),
        "avg_rr":         round(np.mean([t["rr"] for t in trades]), 4),
        "trades":         trades,
        
        # Compatibility keys untuk full_trading_report dan pipeline
        "equity_curve":   [e - modal for e in equity_curve], # convert equity ke PnL cumulative
        "pnl_per_trade":  [t["net_pnl"] for t in trades],
        "trade_log":      trades,
        "total_pnl":      round(total_net_pnl, 4),
        "wins":           len(wins),
        "losses":         len(losses),
        "n_vcb_blocked":  n_vcb_blocked,
        "time_exits":     len(time_e),
        "win_by_class": {
            "LONG":  round(lw / lt, 4) if lt > 0 else 0.0,
            "SHORT": round(sw / st, 4) if st > 0 else 0.0,
        }
    }


# ─── Drawdown ────────────────────────────────────────────────────────────────

def calc_drawdown(equity_curve: list, modal_per_trade: float = 100.0) -> dict:
    if not equity_curve:
        return {"max_drawdown": 0.0, "max_drawdown_pct": 0.0, "drawdown_curve": []}

    eq   = np.array(equity_curve, dtype=np.float64)
    peak = np.maximum.accumulate(eq)
    dd   = peak - eq

    dd_pct = dd / (modal_per_trade + 1e-9)

    return {
        "max_drawdown":     round(float(dd.max()), 4),
        "max_drawdown_pct": round(float(dd_pct.max()), 4),
        "drawdown_curve":   dd.tolist(),
    }


# ─── Consecutive Loss ────────────────────────────────────────────────────────

def calc_consecutive_loss(pnl_per_trade: list) -> int:
    if not pnl_per_trade:
        return 0
    max_streak = current = 0
    for pnl in pnl_per_trade:
        if pnl < 0:
            current   += 1
            max_streak = max(max_streak, current)
        else:
            current = 0
    return max_streak


# ─── Portfolio Execution Constraints (live parity) ───────────────────────────

def apply_portfolio_execution_limits(
    trades: list,
    max_open_positions: int = 10,
    daily_loss_limit: int = 8,
    wita_offset_hours: int = 8,
) -> tuple[list, list]:
    """Filter kandidat trade (tiap koin sudah disimulasikan independen) supaya
    patuh 2 constraint eksekusi live yang TIDAK PERNAH dimodelkan di
    simulate_trades_swing() -- gate asli ada di swint_tradev2/app/services/
    execution.py::place_entry():
      a. max_open_positions -- Trade.query.filter_by(status="open", is_live=True).count()
         >= max_pos -> entry ditolak (global lintas semua koin/arah, bukan per-koin).
      b. daily_loss_limit -- jumlah trade rugi (pnl_net<0) yang closed_at hari ini
         (WITA, midnight-to-midnight) >= limit -> entry ditolak sisa hari itu
         (_get_daily_loss_count(), scope is_live=True).

    simulate_trades_swing() men-simulasikan tiap koin SENDIRI-SENDIRI, tidak tahu
    apa yang terjadi di koin lain -- jadi constraint ini diterapkan POST-HOC setelah
    semua kandidat trade lintas-koin terkumpul: urutkan kronologis via entry_time,
    replay slot posisi terbuka (min-heap by exit_time) + hitung ulang rugi harian
    WITA persis query live. Kandidat yang ditolak TIDAK diganti kandidat lain (sama
    seperti live: sinyal yang ditolak hilang, bukan dicoba ulang di bar berikutnya).

    Return (accepted, rejected) -- trade dict butuh field "entry_time"/"exit_time"
    (tz-aware UTC) dan "net_pnl" terisi (dari holdout_oos.evaluate_coin atau
    run_oof_live_parity_check.py collect_trades).
    """
    import heapq

    candidates = [
        t for t in trades
        if t.get("entry_time") is not None and t.get("exit_time") is not None
    ]
    candidates.sort(key=lambda t: t["entry_time"])

    def _wita_date(ts):
        return (pd.Timestamp(ts) + pd.Timedelta(hours=wita_offset_hours)).date()

    open_heap: list = []   # [(exit_time, net_pnl), ...] min-heap by exit_time
    daily_losses: dict = {}  # {wita_date: n_losses_closed}
    accepted, rejected = [], []

    for t in candidates:
        entry_time = t["entry_time"]
        # Bebaskan slot yang sudah closed sebelum entry kandidat ini, catat rugi
        # harian dari CLOSED_AT-nya (persis _get_daily_loss_count, bukan opened_at).
        while open_heap and open_heap[0][0] <= entry_time:
            closed_exit_time, closed_pnl = heapq.heappop(open_heap)
            if closed_pnl < 0:
                d = _wita_date(closed_exit_time)
                daily_losses[d] = daily_losses.get(d, 0) + 1

        today = _wita_date(entry_time)
        if daily_losses.get(today, 0) >= daily_loss_limit:
            t["_reject_reason"] = "daily_loss_limit"
            rejected.append(t)
            continue
        if len(open_heap) >= max_open_positions:
            t["_reject_reason"] = "max_open_positions"
            rejected.append(t)
            continue

        heapq.heappush(open_heap, (t["exit_time"], t.get("net_pnl", 0.0)))
        accepted.append(t)

    return accepted, rejected


# ─── Trade Per Month ─────────────────────────────────────────────────────────

def calc_trade_per_month(total_trades: int, index: pd.DatetimeIndex) -> float:
    if total_trades == 0 or len(index) == 0:
        return 0.0
    n_months = (index[-1] - index[0]).days / 30.44
    if n_months < 0.1:
        return float(total_trades)
    return round(total_trades / n_months, 2)


# ─── Risk-Adjusted Metrics ───────────────────────────────────────────────────

def calc_risk_metrics(
    pnl_per_trade:   list,
    modal:           float,
    index:           pd.DatetimeIndex,
    max_drawdown_pct: float,
    equity_curve:    list = None,
    rfr:             float = 0.0,
) -> dict:
    """
    Hitung Sharpe, Sortino, Calmar, dan Profit Factor dari trade list.

    Metodologi (standard industry):
      - Sharpe/Sortino: dibangun dari daily equity curve → daily returns → annualized.
        Ini menggantikan metode per-trade IID lama yang meng-inflate metrik 2-4x.
        Annualization factor = sqrt(365) untuk crypto (trading 365 hari/tahun).
      - Calmar: annualized return / |max drawdown|.
      - Profit Factor: gross profit / gross loss (tetap — ini metrik trade-level).

    Jika equity_curve tidak diberikan, fallback ke metode per-trade lama
    (backward compatibility untuk pemanggil yang belum di-update).
    """
    # ── Profit Factor (trade-level, always computed the same way) ────────────
    wins_sum = sum(p for p in pnl_per_trade if p > 0)
    loss_sum = abs(sum(p for p in pnl_per_trade if p < 0))
    profit_factor = (wins_sum / loss_sum) if loss_sum > 0 else 0.0

    if len(pnl_per_trade) < 2:
        return {
            "sharpe_ratio":  0.0,
            "sortino_ratio": 0.0,
            "calmar_ratio":  0.0,
            "profit_factor": round(profit_factor, 4),
        }

    # ── Daily equity curve method (preferred) ───────────────────────────────
    if equity_curve is not None and len(equity_curve) > 1 and len(index) > 1:
        try:
            return _calc_risk_metrics_daily(
                pnl_per_trade, modal, index, max_drawdown_pct,
                equity_curve, rfr, profit_factor,
            )
        except Exception:
            # Fall through ke metode per-trade jika daily gagal
            pass

    # ── Fallback: per-trade IID method (backward compat) ────────────────────
    return _calc_risk_metrics_per_trade(
        pnl_per_trade, modal, index, max_drawdown_pct, rfr, profit_factor,
    )


def _calc_risk_metrics_daily(
    pnl_per_trade:   list,
    modal:           float,
    index:           pd.DatetimeIndex,
    max_drawdown_pct: float,
    equity_curve:    list,
    rfr:             float,
    profit_factor:   float,
) -> dict:
    """
    Hitung Sharpe/Sortino/Calmar dari daily equity curve.

    Alur:
      1. Build portfolio value = modal + equity_curve (per H1 bar)
      2. Resample ke daily: ambil nilai terakhir setiap hari
      3. Daily returns = pct_change portfolio value harian
      4. Annualize dengan sqrt(365) untuk crypto
    """
    ANNUAL_TRADING_DAYS = 365  # crypto — 24/7/365

    eq_arr   = np.asarray(equity_curve, dtype=np.float64)
    port_val = modal + eq_arr                                   # portfolio value per bar
    eq_series = pd.Series(port_val, index=index)

    # Resample ke daily — ambil nilai penutupan hari (bar terakhir setiap UTC day)
    daily_eq = eq_series.resample("D").last().dropna()
    if len(daily_eq) < 2:
        raise ValueError("Insufficient daily data points")

    daily_rets = daily_eq.pct_change().dropna()
    if len(daily_rets) < 2:
        raise ValueError("Insufficient daily returns")

    # ── Sharpe Ratio ───────────────────────────────────────────────────────
    mean_ret  = float(daily_rets.mean())
    std_ret   = float(daily_rets.std(ddof=1))
    ann_factor = np.sqrt(ANNUAL_TRADING_DAYS)
    sharpe = (mean_ret / std_ret * ann_factor) if std_ret > 1e-10 else 0.0

    # ── Sortino Ratio ──────────────────────────────────────────────────────
    downside = daily_rets[daily_rets < 0]
    if len(downside) > 1:
        std_down = float(downside.std(ddof=1))
        sortino = (mean_ret / std_down * ann_factor) if std_down > 1e-10 else 0.0
    else:
        sortino = 0.0

    # ── Calmar Ratio ───────────────────────────────────────────────────────
    # annualized return = CAGR dari daily equity
    total_return = (port_val[-1] - modal) / modal          # total period return
    n_days = max((index[-1] - index[0]).days, 1)
    ann_return = (1 + total_return) ** (ANNUAL_TRADING_DAYS / n_days) - 1
    calmar = (ann_return / abs(max_drawdown_pct)) if abs(max_drawdown_pct) > 1e-9 else 0.0

    return {
        "sharpe_ratio":  round(sharpe, 4),
        "sortino_ratio": round(sortino, 4),
        "calmar_ratio":  round(calmar, 4),
        "profit_factor": round(profit_factor, 4),
    }


def _calc_risk_metrics_per_trade(
    pnl_per_trade:   list,
    modal:           float,
    index:           pd.DatetimeIndex,
    max_drawdown_pct: float,
    rfr:             float,
    profit_factor:   float,
) -> dict:
    """
    Fallback: per-trade IID method (legacy).
    Hanya dipakai jika equity_curve tidak tersedia.
    """
    returns = np.array(pnl_per_trade, dtype=np.float64) / (modal + 1e-9)

    n_years = max((index[-1] - index[0]).days / 365.25, 1 / 52)
    trades_per_year = len(returns) / n_years
    ann_factor = np.sqrt(trades_per_year)

    rfr_per_trade = rfr / max(trades_per_year, 1)
    excess = returns - rfr_per_trade

    std_r = np.std(returns, ddof=1)
    sharpe = float(np.mean(excess) / std_r * ann_factor) if std_r > 0 else 0.0

    downside = excess[excess < 0]
    std_down = np.std(downside, ddof=1) if len(downside) > 1 else 0.0
    sortino  = float(np.mean(excess) / std_down * ann_factor) if std_down > 0 else 0.0

    ann_return = float(np.mean(returns)) * trades_per_year
    calmar = (ann_return / abs(max_drawdown_pct)) if abs(max_drawdown_pct) > 1e-9 else 0.0

    return {
        "sharpe_ratio":  round(sharpe, 4),
        "sortino_ratio": round(sortino, 4),
        "calmar_ratio":  round(calmar, 4),
        "profit_factor": round(profit_factor, 4),
    }


# ─── Full Report ─────────────────────────────────────────────────────────────

def full_trading_report(
    y_pred:       np.ndarray,
    y_actual:     np.ndarray,
    atr:          np.ndarray,
    close:        np.ndarray,
    index:        pd.DatetimeIndex,
    modal:        float = 100.0,
    leverages:    list  = [5.0],
    fee_per_side: float = 0.0004,
    tp_mult:      float = 2.0,
    sl_mult:      float = 1.0,
    max_hold:     int   = 24,
    min_hold:     int   = 2,
    symbol:       Optional[str] = None,
    slippage:     float = 0.0005,
    # Parameters for Swing V3 Option:
    high:         Optional[np.ndarray] = None,
    low:          Optional[np.ndarray] = None,
    h4_swing_highs: Optional[np.ndarray] = None,
    h4_swing_lows:  Optional[np.ndarray] = None,
    min_rr:       float = TP_SL_MIN_RR,
    min_tp_atr:   float = TP_SL_MIN_TP,
    max_sl_atr:   float = TP_SL_MAX_SL,
    tp_fallback_atr: float = TP_SL_FALLBACK_TP,
    sl_fallback_atr: float = TP_SL_FALLBACK_SL,
    confidence         = None,  # for sizing_mode="tiered"
    modal_arr          = None,  # per-bar dynamic modal (overrides modal+sizing_mode)
    # Parameters for Dynamic TP/SL Regressor (Priority 1):
    tp_regressor     = None,
    sl_regressor     = None,
    X_tp             = None,
    X_sl             = None,
    tp_reg_clip      = (0.5, 8.0),
    sl_reg_clip      = (0.3, 5.0),
    # ── NEW: Grup 1, 3, 4 parameters ──────────────────────────────────────
    vol_ratio                 = None,  # np.ndarray
    volr_conditional_enabled  = TP_SL_VOLR_CONDITIONAL_ENABLED,
    volr_threshold            = TP_SL_VOLR_THRESHOLD,
    max_sl_volr_low           = TP_SL_MAX_SL_VOLR_LOW,
    volr_disable_max_sl       = TP_SL_VOLR_DISABLE_MAX_SL,
    max_sl_pct_enabled        = TP_SL_MAX_SL_PCT_ENABLED,
    max_sl_pct                = TP_SL_MAX_SL_PCT,
    max_swing_deviation_pct   = TP_SL_MAX_SWING_DEVIATION_PCT,
    individual_swing_freshness = TP_SL_INDIVIDUAL_SWING_FRESHNESS,
    h4_trend                   = None,  # np.ndarray
    sizing_with_trend_half     = TP_SL_SIZING_WITH_TREND_HALF,
    structural_tolerance_pct   = TP_SL_STRUCTURAL_TOLERANCE,
    swing_sl_bumper_atr        = 0.5,
    hybrid_mode                = TP_SL_HYBRID_MODE,
    swing_freshness_check      = TP_SL_SWING_FRESHNESS,
    structural_filter          = TP_SL_STRUCTURAL_FILTER,
    slippage_enabled           = TP_SL_SLIPPAGE_ENABLED,
    sl_trigger_mode            = TP_SL_TRIGGER_MODE,
    sizing_mode                = TP_SL_SIZING_MODE,
    cooldown_enabled           = TP_SL_COOLDOWN_ENABLED,
    # ── Exit Guardian ─────────────────────────────────────────────────
    guardian_model            = None,
    guardian_scaler           = None,
    X_guardian                = None,
    guardian_exit_threshold   = 0.60,
    guardian_sl_exit_threshold = GUARDIAN_SL_EXIT_THRESHOLD,
    guardian_sl_safety_atr    = GUARDIAN_SL_SAFETY_ATR,
    guardian_tp_atr           = GUARDIAN_TP_ATR,
    guardian_min_hold_bars    = GUARDIAN_MIN_HOLD_BARS,
    guardian_activation_atr   = GUARDIAN_ACTIVATION_ATR,
    guardian_enabled          = GUARDIAN_ENABLED,
    guardian_feat_cols        = None,
    guardian_static_names     = None,
    flow_momentum_arr         = None,
    entry_price_override      = None,  # np.ndarray — M15 entry price per H1 bar (optional)
    # ── Guardian Delta mode ───────────────────────────────────────────────
    guardian_delta_raw        = None,
    guardian_delta_feats      = None,
    guardian_mom_thresh       = 0.45,
    guardian_def_thresh       = 0.70,
    guardian_def_min_loss     = -0.010,
    guardian_binary_peak      = False,
    guardian_emergency_pre_tp = False,
    guardian_emergency_conf_delta       = -0.10,
    guardian_emergency_conf_delta_near_tp = -0.16,
    guardian_near_tp_proximity          = 0.85,
    guardian_emergency_min_loss         = -0.004,
    trailing_stop_enabled     = TRAILING_STOP_ENABLED,
    trailing_stop_atr         = TRAILING_STOP_ATR,
    trailing_stop_min_bars    = TRAILING_STOP_MIN_BARS,
    pyramiding_enabled        = False,
    pyramiding_max_per_coin   = 1,
    pyramiding_same_dir       = True,
    pyramiding_exit_mode      = "independent",
) -> dict:
    """
    Jalankan full trading simulation dan return metrics lengkap.

    Semua parameter TP/SL mengikuti config.py (TP_SL_*).
    """
    label_prefix = f"[{symbol}] " if symbol else ""
    use_swing = h4_swing_highs is not None and h4_swing_lows is not None and high is not None

    def run_sim(lev):
        if use_swing:
            return simulate_trades_swing(
                y_pred=y_pred, close=close, high=high, low=low, atr=atr,
                h4_swing_highs=h4_swing_highs, h4_swing_lows=h4_swing_lows,
                modal=modal, leverage=lev, fee_per_side=fee_per_side,
                slippage=slippage,
                min_rr=min_rr, min_tp_atr=min_tp_atr, max_sl_atr=max_sl_atr,
                max_hold=max_hold,
                tp_fallback_atr=tp_fallback_atr, sl_fallback_atr=sl_fallback_atr,
                confidence=confidence, modal_arr=modal_arr,
                # New params Grup 1, 3, 4
                vol_ratio=vol_ratio,
                volr_conditional_enabled=volr_conditional_enabled,
                volr_threshold=volr_threshold,
                max_sl_volr_low=max_sl_volr_low,
                volr_disable_max_sl=volr_disable_max_sl,
                max_sl_pct_enabled=max_sl_pct_enabled,
                max_sl_pct=max_sl_pct,
                max_swing_deviation_pct=max_swing_deviation_pct,
                individual_swing_freshness=individual_swing_freshness,
                h4_trend=h4_trend,
                sizing_with_trend_half=sizing_with_trend_half,
                # Existing toggles
                hybrid_mode=hybrid_mode,
                swing_freshness_check=swing_freshness_check,
                structural_filter=structural_filter,
                structural_tolerance_pct=structural_tolerance_pct,
                slippage_enabled=slippage_enabled,
                sl_trigger_mode=sl_trigger_mode,
                sizing_mode=sizing_mode,
                cooldown_enabled=cooldown_enabled,
                swing_sl_bumper_atr=swing_sl_bumper_atr,
                guardian_model=guardian_model,
                guardian_scaler=guardian_scaler,
                X_guardian=X_guardian,
                guardian_exit_threshold=guardian_exit_threshold,
                guardian_sl_exit_threshold=guardian_sl_exit_threshold,
                guardian_sl_safety_atr=guardian_sl_safety_atr,
                guardian_tp_atr=guardian_tp_atr,
                guardian_min_hold_bars=guardian_min_hold_bars,
                guardian_activation_atr=guardian_activation_atr,
                guardian_enabled=guardian_enabled,
                guardian_feat_cols=guardian_feat_cols,
                guardian_static_names=guardian_static_names,
                flow_momentum_arr=flow_momentum_arr,
                guardian_delta_raw=guardian_delta_raw,
                guardian_delta_feats=guardian_delta_feats,
                guardian_mom_thresh=guardian_mom_thresh,
                guardian_def_thresh=guardian_def_thresh,
                guardian_def_min_loss=guardian_def_min_loss,
                guardian_binary_peak=guardian_binary_peak,
                guardian_emergency_pre_tp=guardian_emergency_pre_tp,
                guardian_emergency_conf_delta=guardian_emergency_conf_delta,
                guardian_emergency_conf_delta_near_tp=guardian_emergency_conf_delta_near_tp,
                guardian_near_tp_proximity=guardian_near_tp_proximity,
                guardian_emergency_min_loss=guardian_emergency_min_loss,
                trailing_stop_enabled=trailing_stop_enabled,
                trailing_stop_atr=trailing_stop_atr,
                trailing_stop_min_bars=trailing_stop_min_bars,
                entry_price_override=entry_price_override,
                pyramiding_enabled=pyramiding_enabled,
                pyramiding_max_per_coin=pyramiding_max_per_coin,
                pyramiding_same_dir=pyramiding_same_dir,
                pyramiding_exit_mode=pyramiding_exit_mode,
            )
        else:
            return simulate_trades(
                y_pred=y_pred, close=close, atr=atr,
                modal=modal, leverage=lev, fee_per_side=fee_per_side,
                slippage=slippage,
                tp_mult=tp_mult, sl_mult=sl_mult,
                max_hold=max_hold, min_hold=min_hold,
            )

    # Base simulation (leverage pertama) untuk winrate dan consecutive loss
    base = run_sim(leverages[0])

    tpm        = calc_trade_per_month(base.get("total_trades", 0), index)
    max_consec = calc_consecutive_loss(base.get("pnl_per_trade", []))

    # Drawdown base untuk Calmar Ratio
    base_dd  = calc_drawdown(base.get("equity_curve", []), modal_per_trade=modal)
    base_ddp = base_dd.get("max_drawdown_pct", 0)

    risk = calc_risk_metrics(
        pnl_per_trade    = base.get("pnl_per_trade", []),
        modal            = modal,
        index            = index,
        max_drawdown_pct = base_ddp,
        equity_curve     = base.get("equity_curve", None),
    )

    logger.info(
        f"{label_prefix}Winrate: {base.get('winrate', 0):.2%} "
        f"({base.get('wins', 0)}W / {base.get('losses', 0)}L / {base.get('total_trades', 0)} trades "
        f"| time_exit={base.get('time_exits', 0)}) "
        f"| Sharpe={risk['sharpe_ratio']:.2f} Sortino={risk['sortino_ratio']:.2f} "
        f"Calmar={risk['calmar_ratio']:.2f} PF={risk['profit_factor']:.2f}"
    )

    report = {
        "symbol":               symbol,
        "winrate":              base.get("winrate", 0),
        "total_trades":         base.get("total_trades", 0),
        "wins":                 base.get("wins", 0),
        "losses":               base.get("losses", 0),
        "time_exits":           base.get("time_exits", 0),
        "win_by_class":         base.get("win_by_class", {}),
        "trade_per_month":      tpm,
        "max_consecutive_loss": max_consec,
        # Risk-adjusted metrics
        "sharpe_ratio":         risk["sharpe_ratio"],
        "sortino_ratio":        risk["sortino_ratio"],
        "calmar_ratio":         risk["calmar_ratio"],
        "profit_factor":        risk["profit_factor"],
        "trades":               base.get("trades") or base.get("trade_log") or [],
    }

    # PnL & Drawdown per leverage
    for lev in leverages:
        sim = run_sim(lev)
        dd  = calc_drawdown(sim.get("equity_curve", []), modal_per_trade=modal)
        key = f"lev{int(lev)}x"

        report[f"pnl_{key}"]          = sim.get("total_pnl", 0)
        report[f"max_drawdown_{key}"] = dd.get("max_drawdown_pct", 0)
        report[f"total_fee_{key}"]    = sim.get("total_fee_paid", 0)

        logger.info(
            f"{label_prefix}Lev {lev}x -> "
            f"PnL: ${sim.get('total_pnl', 0):+.2f} | "
            f"DD: {dd.get('max_drawdown_pct', 0):.2%}"
        )

    return report