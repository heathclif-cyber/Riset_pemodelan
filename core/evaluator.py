"""
core/evaluator.py — Trading Metrics & PnL Simulation v2
Dipakai oleh pipeline/07_evaluate.py dan pipeline/08_backtest.py

Fungsi utama:
  simulate_trades()       — simulasi trade nyata dari price array (TP/SL dari harga)
  calc_drawdown()         — max drawdown dari equity curve
  calc_consecutive_loss() — streak loss terpanjang
  calc_trade_per_month()  — rata-rata trade per bulan
  full_trading_report()   — jalankan semua metrics sekaligus, return dict

PENTING — perbedaan v1 vs v2:
  v1: winrate dihitung dari y_pred == y_actual (label matching) → INFLATED
  v2: winrate dihitung dari simulasi TP/SL nyata dari close price array → REALISTIS
      Setiap trade di-simulasi: masuk di close[i], cek apakah close[i+1..i+max_hold]
      hit TP atau SL duluan. Tidak menggunakan label sama sekali.
"""

import numpy as np
import pandas as pd
from typing import Optional

from core.utils import setup_logger

logger = setup_logger("evaluator")


# ─── Simulasi Trade dari Price (Realistis) ────────────────────────────────────

def simulate_trades(
    y_pred:       np.ndarray,
    close:        np.ndarray,
    atr:          np.ndarray,
    modal:        float = 1000.0,
    leverage:     float = 3.0,
    fee_per_side: float = 0.0004,
    tp_mult:      float = 2.0,
    sl_mult:      float = 1.0,
    max_hold:     int   = 48,
    min_hold:     int   = 4,
) -> dict:
    """
    Simulasi trade nyata — masuk di close[i], cek TP/SL dari bar berikutnya.

    Mekanisme:
      1. Setiap bar yang diprediksi LONG/SHORT → masuk trade di close[i]
      2. Scan bar i+1 sampai i+max_hold:
         - LONG : cek apakah high >= TP dulu, atau low <= SL dulu
         - SHORT: cek apakah low <= TP dulu, atau high >= SL dulu
      3. Kalau tidak hit sampai max_hold → keluar di close[i+max_hold] (time exit)
      4. min_hold: setelah entry, minimal tunggu N bar sebelum entry baru
         (mencegah overtrading / signal spam)

    Args:
        y_pred       : prediksi model (int: 0=SHORT, 1=FLAT, 2=LONG)
        close        : array close price
        atr          : array ATR M15
        modal        : modal per trade USD (tetap, tidak compounding)
        leverage     : leverage
        fee_per_side : fee per sisi (0.0004 = 0.04%)
        tp_mult      : TP = entry ± tp_mult × ATR
        sl_mult      : SL = entry ∓ sl_mult × ATR
        max_hold     : maksimum bar hold sebelum time exit (default 48 = 12 jam)
        min_hold     : minimum bar antara dua trade (default 4 = 1 jam)

    Returns dict:
        equity_curve    : list float, kumulatif PnL per bar (panjang = len(y_pred))
        pnl_per_trade   : list float, PnL tiap trade yang dieksekusi
        trade_log       : list dict, detail tiap trade
        total_pnl       : float
        total_trades    : int
        wins            : int
        losses          : int
        time_exits      : int (keluar karena max_hold, bukan TP/SL)
        total_fee_paid  : float
        winrate         : float
        win_by_class    : dict
    """
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

    last_exit_bar = -1  # untuk min_hold filter

    i = 0
    while i < n:
        pred = y_pred[i]

        # Skip FLAT atau terlalu dekat dengan trade sebelumnya
        if pred == 1 or (i - last_exit_bar) < min_hold:
            equity_curve[i] = cumulative
            i += 1
            continue

        entry_price = close[i]
        atr_i       = atr[i]

        if np.isnan(entry_price) or np.isnan(atr_i) or atr_i == 0 or entry_price == 0:
            equity_curve[i] = cumulative
            i += 1
            continue

        # Hitung level TP dan SL
        if pred == 2:  # LONG
            tp_price = entry_price + tp_mult * atr_i
            sl_price = entry_price - sl_mult * atr_i
        else:          # SHORT
            tp_price = entry_price - tp_mult * atr_i
            sl_price = entry_price + sl_mult * atr_i

        fee = 2 * fee_per_side * modal

        # Scan bar ke depan untuk cek TP/SL
        outcome   = "time_exit"
        exit_bar  = min(i + max_hold, n - 1)
        exit_price = close[exit_bar]

        for j in range(i + 1, min(i + max_hold + 1, n)):
            if np.isnan(close[j]):
                continue

            # Gunakan high/low yang diestimasikan dari close + ATR
            # Karena backtest hanya punya close, estimasi:
            # high[j] ≈ close[j] + 0.5 * atr[j]
            # low[j]  ≈ close[j] - 0.5 * atr[j]
            est_high = close[j] + 0.5 * (atr[j] if not np.isnan(atr[j]) else atr_i)
            est_low  = close[j] - 0.5 * (atr[j] if not np.isnan(atr[j]) else atr_i)

            if pred == 2:  # LONG
                if est_high >= tp_price and est_low <= sl_price:
                    # Ambiguous — lihat close untuk tiebreak
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

        # Hitung PnL
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
            # Keluar di close[exit_bar], hitung actual return
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

        # Update equity curve untuk semua bar sampai exit
        for k in range(i, min(exit_bar + 1, n)):
            equity_curve[k] = cumulative

        last_exit_bar = exit_bar
        i = exit_bar + 1  # lanjut dari bar setelah exit

    # Fill sisa equity curve
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


# ─── Drawdown ────────────────────────────────────────────────────────────────

def calc_drawdown(equity_curve: list, modal_per_trade: float = 1000.0) -> dict:
    if not equity_curve:
        return {"max_drawdown": 0.0, "max_drawdown_pct": 0.0, "drawdown_curve": []}

    eq   = np.array(equity_curve, dtype=np.float64)
    peak = np.maximum.accumulate(eq)
    dd   = peak - eq

    # Normalisasi terhadap modal per trade, bukan peak equity
    # DD 1.5 = pernah rugi 1.5x modal dalam satu streak
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


# ─── Trade Per Month ─────────────────────────────────────────────────────────

def calc_trade_per_month(total_trades: int, index: pd.DatetimeIndex) -> float:
    if total_trades == 0 or len(index) == 0:
        return 0.0
    n_months = (index[-1] - index[0]).days / 30.44
    if n_months < 0.1:
        return float(total_trades)
    return round(total_trades / n_months, 2)


# ─── Full Report ─────────────────────────────────────────────────────────────

def full_trading_report(
    y_pred:       np.ndarray,
    y_actual:     np.ndarray,      # dipertahankan untuk kompatibilitas, tidak dipakai
    atr:          np.ndarray,
    close:        np.ndarray,
    index:        pd.DatetimeIndex,
    modal:        float = 1000.0,
    leverages:    list  = [3.0, 5.0],
    fee_per_side: float = 0.0004,
    tp_mult:      float = 2.0,
    sl_mult:      float = 1.0,
    max_hold:     int   = 48,
    min_hold:     int   = 4,
    symbol:       Optional[str] = None,
) -> dict:
    """
    Jalankan full trading simulation dan return metrics lengkap.

    y_actual dipertahankan di signature untuk backward compat dengan
    07_evaluate.py dan 08_backtest.py — tapi tidak dipakai di v2.
    Simulasi sepenuhnya berbasis price (close + ATR).
    """
    label_prefix = f"[{symbol}] " if symbol else ""

    # Base simulation (leverage pertama) untuk winrate dan consecutive loss
    base = simulate_trades(
        y_pred, close, atr,
        modal=modal, leverage=leverages[0],
        fee_per_side=fee_per_side,
        tp_mult=tp_mult, sl_mult=sl_mult,
        max_hold=max_hold, min_hold=min_hold,
    )

    tpm        = calc_trade_per_month(base["total_trades"], index)
    max_consec = calc_consecutive_loss(base["pnl_per_trade"])

    logger.info(
        f"{label_prefix}Winrate: {base['winrate']:.2%} "
        f"({base['wins']}W / {base['losses']}L / {base['total_trades']} trades "
        f"| time_exit={base['time_exits']})"
    )

    report = {
        "symbol":               symbol,
        "winrate":              base["winrate"],
        "total_trades":         base["total_trades"],
        "wins":                 base["wins"],
        "losses":               base["losses"],
        "time_exits":           base["time_exits"],
        "win_by_class":         base["win_by_class"],
        "trade_per_month":      tpm,
        "max_consecutive_loss": max_consec,
    }

    # PnL & Drawdown per leverage
    for lev in leverages:
        sim = simulate_trades(
            y_pred, close, atr,
            modal=modal, leverage=lev,
            fee_per_side=fee_per_side,
            tp_mult=tp_mult, sl_mult=sl_mult,
            max_hold=max_hold, min_hold=min_hold,
        )
        dd  = calc_drawdown(sim["equity_curve"], modal_per_trade=modal)
        key = f"lev{int(lev)}x"

        report[f"pnl_{key}"]          = sim["total_pnl"]
        report[f"max_drawdown_{key}"] = dd["max_drawdown_pct"]
        report[f"total_fee_{key}"]    = sim["total_fee_paid"]

        logger.info(
            f"{label_prefix}Lev {lev}x → "
            f"PnL: ${sim['total_pnl']:+.2f} | "
            f"DD: {dd['max_drawdown_pct']:.2%} | "
            f"Fee: ${sim['total_fee_paid']:.2f}"
        )

    return report