"""Single-position scale-in simulation (1 trade record, VWAP entry, unified exit)."""
from __future__ import annotations

import numpy as np

from config import GUARDIAN_PARTIAL_EXIT_RATIO, TP_SL_RR_GATE_ENABLED
from core.evaluator import (
    _assemble_guardian_row,
    _compute_guardian_delta_vector,
    _compute_guardian_dynamic,
)


def _trade_summary(trades: list, equity_curve: list, modal: float, n_vcb_blocked: int) -> dict:
    if not trades:
        return {
            "error": "no_trades", "total_trades": 0, "winrate": 0.0,
            "total_pnl": 0.0, "max_drawdown": 0.0, "max_drawdown_pct": 0.0,
            "equity_curve": equity_curve, "pnl_per_trade": [],
            "wins": 0, "losses": 0, "time_exits": 0,
            "win_by_class": {"LONG": 0.0, "SHORT": 0.0},
        }
    guardian_outcomes = (
        "TRAILING_STOP", "GUARDIAN_EXIT", "GUARDIAN_FULL",
        "GUARDIAN_MOMENTUM_EXIT", "GUARDIAN_MOMENTUM_PARTIAL",
        "GUARDIAN_DELTA_EXIT", "TIMEOUT", "TIMEOUT_MOMENTUM",
    )
    wins = [t for t in trades if t["outcome"] == "WIN"
            or (t["outcome"] in guardian_outcomes and t["net_pnl"] > 0)]
    losses = [t for t in trades if t["outcome"] == "LOSS"
              or (t["outcome"] in guardian_outcomes and t["net_pnl"] <= 0)]
    time_e = [t for t in trades if t["outcome"] in ("TIMEOUT", "TIMEOUT_MOMENTUM")]
    winrate = len(wins) / len(trades) if trades else 0.0
    sum_loss = abs(sum(t["net_pnl"] for t in losses))
    sum_win = abs(sum(t["net_pnl"] for t in wins))
    pf = sum_win / sum_loss if sum_loss > 0 else 0.0
    equity_arr = np.array([e for e in equity_curve if not np.isnan(e)])
    peak = np.maximum.accumulate(equity_arr)
    drawdown = (equity_arr - peak) / (peak + 1e-10)
    lw = len([t for t in wins if t["direction"] == "LONG"])
    lt = len([t for t in trades if t["direction"] == "LONG"])
    sw = len([t for t in wins if t["direction"] == "SHORT"])
    st = len([t for t in trades if t["direction"] == "SHORT"])
    total_net_pnl = sum(t["net_pnl"] for t in trades)
    return {
        "total_trades": len(trades),
        "winrate": round(winrate, 4),
        "avg_win": round(float(np.mean([t["net_pnl"] for t in wins])), 4) if wins else 0.0,
        "avg_loss": round(float(np.mean([t["net_pnl"] for t in losses])), 4) if losses else 0.0,
        "profit_factor": round(pf, 4),
        "net_pnl_total": round(total_net_pnl, 4),
        "max_drawdown": round(float(drawdown.min()), 4),
        "avg_rr": round(float(np.mean([t["rr"] for t in trades])), 4),
        "trades": trades,
        "equity_curve": [e - modal for e in equity_curve],
        "pnl_per_trade": [t["net_pnl"] for t in trades],
        "trade_log": trades,
        "total_pnl": round(total_net_pnl, 4),
        "wins": len(wins),
        "losses": len(losses),
        "n_vcb_blocked": n_vcb_blocked,
        "time_exits": len(time_e),
        "win_by_class": {
            "LONG": round(lw / lt, 4) if lt > 0 else 0.0,
            "SHORT": round(sw / st, 4) if st > 0 else 0.0,
        },
    }


def _calc_tp_sl(sig, price, atr_i, sh_i, sl_i, use_swing, hybrid_mode,
                tp_fallback_atr, sl_fallback_atr, swing_sl_bumper_atr,
                min_sl_pct, max_sl_atr, vol_ratio, volr_conditional_enabled,
                volr_threshold, volr_disable_max_sl, max_sl_volr_low,
                max_sl_pct_enabled, min_tp_atr, min_rr):
    if use_swing:
        if sig == 2:  # LONG
            swing_tp = sh_i
            swing_sl = sl_i - (swing_sl_bumper_atr * atr_i)
            atr_tp = price + tp_fallback_atr * atr_i
            atr_sl = price - sl_fallback_atr * atr_i
            if hybrid_mode:
                tp_price = max(swing_tp, atr_tp)
                sl_price = min(swing_sl, atr_sl)
            else:
                tp_price, sl_price = swing_tp, swing_sl
            tp_dist, sl_dist = tp_price - price, price - sl_price
        else:
            swing_tp = sl_i
            swing_sl = sh_i + (swing_sl_bumper_atr * atr_i)
            atr_tp = price - tp_fallback_atr * atr_i
            atr_sl = price + sl_fallback_atr * atr_i
            if hybrid_mode:
                tp_price = min(swing_tp, atr_tp)
                sl_price = max(swing_sl, atr_sl)
            else:
                tp_price, sl_price = swing_tp, swing_sl
            tp_dist, sl_dist = price - tp_price, sl_price - price
    else:
        if sig == 2:
            tp_price = price + tp_fallback_atr * atr_i
            sl_price = price - sl_fallback_atr * atr_i
            tp_dist, sl_dist = tp_price - price, price - sl_price
        else:
            tp_price = price - tp_fallback_atr * atr_i
            sl_price = price + sl_fallback_atr * atr_i
            tp_dist, sl_dist = price - tp_price, sl_price - price

    if min_sl_pct > 0.0:
        floor_dist = price * min_sl_pct
        if sl_dist < floor_dist:
            sl_dist = floor_dist
            sl_price = price - floor_dist if sig == 2 else price + floor_dist

    if not TP_SL_RR_GATE_ENABLED:
        if tp_dist <= 0 or sl_dist <= 0:
            return None
        return tp_price, sl_price, tp_dist / sl_dist

    if tp_dist <= 0 or sl_dist <= 0 or tp_dist < min_tp_atr * atr_i:
        return None
    eff_max_sl = max_sl_atr
    if volr_conditional_enabled and vol_ratio is not None and not np.isnan(vol_ratio):
        if vol_ratio < volr_threshold:
            eff_max_sl = float("inf") if volr_disable_max_sl else max_sl_volr_low
    sl_cap = price * max_sl_pct if max_sl_pct_enabled else float("inf")
    if sl_dist > min(eff_max_sl * atr_i, sl_cap):
        return None
    rr = tp_dist / sl_dist
    if rr < min_rr:
        return None
    return tp_price, sl_price, rr


def run_scale_in_simulation(
    y_pred, close, high, low, atr,
    h4_swing_highs, h4_swing_lows,
    modal=10.0, leverage=5.0, fee_per_side=0.0004, slippage=0.0005,
    min_rr=0.6, min_tp_atr=1.2, max_sl_atr=4.0, max_hold=36,
    tp_fallback_atr=1.2, sl_fallback_atr=1.5,
    confidence=None, modal_arr=None,
    vol_ratio=None, volr_conditional_enabled=False, volr_threshold=0.5,
    max_sl_volr_low=5.0, volr_disable_max_sl=False,
    max_sl_pct_enabled=False, max_sl_pct=0.05,
    max_swing_deviation_pct=0.15, individual_swing_freshness=False,
    hybrid_mode=False, swing_freshness_check=False, structural_filter=False,
    structural_tolerance_pct=0.03, slippage_enabled=True, sl_trigger_mode="close",
    sizing_mode="fixed", cooldown_enabled=False, swing_sl_bumper_atr=0.5,
    guardian_model=None, guardian_scaler=None, X_guardian=None,
    guardian_exit_threshold=0.65, guardian_min_hold_bars=4,
    guardian_activation_atr=0.0, guardian_enabled=True,
    guardian_feat_cols=None, guardian_static_names=None,
    flow_momentum_arr=None, guardian_delta_raw=None, guardian_delta_feats=None,
    guardian_mom_thresh=0.45, guardian_def_thresh=0.70, guardian_def_min_loss=-0.010,
    trailing_stop_enabled=False, trailing_stop_atr=1.5, trailing_stop_min_bars=3,
    min_sl_pct=0.0, vcb_enabled=False, vcb_atr_multiplier=3.0, vcb_lookback_bars=24,
    pyramiding_max_per_coin=2, pyramiding_same_dir=True,
    sizing_with_trend_half=False, h4_trend=None,
    entry_price_override=None,
) -> dict:
    n = len(close)
    trades = []
    equity = modal
    equity_curve = [equity]
    n_vcb_blocked = 0
    cooldown_until = -1
    LONG, SHORT, FLAT = 2, 0, 1
    use_swing = h4_swing_highs is not None and h4_swing_lows is not None and high is not None
    guardian_active = guardian_enabled and guardian_model is not None and X_guardian is not None
    guardian_delta_active = (
        guardian_model is not None and guardian_delta_raw is not None and guardian_delta_feats is not None
    )
    active = None

    def _entry_modal(i, sig):
        if modal_arr is not None:
            return float(modal_arr[i])
        if sizing_mode == "tiered" and confidence is not None:
            conf_i = confidence[i]
            if conf_i > 0.75:
                m = modal
            elif conf_i > 0.60:
                m = modal * 0.5
            else:
                return None
            if sizing_with_trend_half and h4_trend is not None:
                trend_i = h4_trend[i]
                if not np.isnan(trend_i):
                    wt = (sig == LONG and trend_i > 0) or (sig == SHORT and trend_i < 0)
                    if wt:
                        m *= 0.5
            return m
        return modal

    def _close_position(act, bar_out, outcome, raw_exit):
        nonlocal equity, cooldown_until
        sig = act["sig"]
        vwap = act["vwap"]
        total_modal = act["total_modal"]
        if slippage_enabled:
            if sig == LONG:
                exit_price = raw_exit * (1.0 - slippage)
                partial_adj = (act["partial_price"] * (1.0 - slippage)) if act["partial_price"] else None
            else:
                exit_price = raw_exit * (1.0 + slippage)
                partial_adj = (act["partial_price"] * (1.0 + slippage)) if act["partial_price"] else None
        else:
            exit_price = raw_exit
            partial_adj = act["partial_price"]

        pct_move = (exit_price - vwap) / vwap
        if sig == SHORT:
            pct_move = -pct_move
        rem = act["position_remaining"]
        gross_pnl = total_modal * leverage * pct_move * rem
        fee_total = total_modal * leverage * fee_per_side * (1.0 + rem) + act["extra_entry_fees"]
        net_pnl = gross_pnl - fee_total + act["partial_pnl"]
        equity += net_pnl
        trades.append({
            "bar_in": act["entry_bar"],
            "bar_out": bar_out,
            "direction": "LONG" if sig == LONG else "SHORT",
            "entry": round(vwap, 6),
            "exit": round(exit_price, 6),
            "tp": act["tp_price"],
            "sl": act["sl_price"],
            "rr": round(act["rr"], 2),
            "outcome": ("TIMEOUT_MOMENTUM" if outcome == "TIMEOUT" and act["tp_touched"] else outcome),
            "net_pnl": round(net_pnl, 4),
            "equity": round(equity, 4),
            "modal_used": round(total_modal, 2),
            "n_legs": act["legs"],
            "scale_in": True,
            "partial_bar": act["partial_bar"],
            "partial_pnl": round(act["partial_pnl"], 4),
        })
        if cooldown_enabled:
            if outcome == "WIN":
                cooldown_until = bar_out + 2
            elif outcome == "LOSS":
                cooldown_until = bar_out + 4
            else:
                cooldown_until = bar_out + 2

    def _check_exit(act, j):
        sig = act["sig"]
        vwap = act["vwap"]
        entry_bar = act["entry_bar"]
        bars_held = j - entry_bar
        atr_i = atr[entry_bar]
        tp_price, sl_price = act["tp_price"], act["sl_price"]
        mfe = act["mfe_pnl"]
        if sig == LONG:
            mfe = max(mfe, (high[j] - vwap) / vwap)
            tp_hit = high[j] >= tp_price
            sl_hit = (low[j] <= sl_price) if sl_trigger_mode == "highlow" else (close[j] <= sl_price)
        else:
            mfe = max(mfe, (vwap - low[j]) / vwap)
            tp_hit = low[j] <= tp_price
            sl_hit = (high[j] >= sl_price) if sl_trigger_mode == "highlow" else (close[j] >= sl_price)
        act["mfe_pnl"] = mfe

        if sl_hit:
            return "LOSS", sl_price, True
        guardian_momentum = act["tp_touched"]
        if tp_hit and not guardian_momentum and guardian_active:
            act["tp_touched"] = True
            guardian_momentum = True
        elif tp_hit and not guardian_active:
            return "WIN", tp_price, True

        if guardian_active and act["position_remaining"] > 0.5:
            should_check = guardian_momentum or bars_held >= guardian_min_hold_bars
            if should_check:
                price_moved_atr = abs(close[j] - vwap) / atr_i if atr_i > 0 else float("inf")
                if guardian_momentum or price_moved_atr >= guardian_activation_atr:
                    g_static_cur = X_guardian[j, :]
                    g_static_ent = X_guardian[entry_bar, :]
                    if guardian_feat_cols and guardian_static_names:
                        fm_val = float(flow_momentum_arr[j]) if flow_momentum_arr is not None else 0.0
                        g_row = _assemble_guardian_row(
                            guardian_feat_cols, guardian_static_names,
                            g_static_cur, g_static_ent,
                            bars_held, vwap, close[j], sig, atr_i, mfe,
                            flow_momentum_3bar=fm_val,
                        )
                        g_feat = g_row.reshape(1, -1)
                    else:
                        g_dynamic = _compute_guardian_dynamic(
                            bars_held, vwap, close[j], sig, atr_i, mfe,
                            g_static_cur, g_static_ent, None,
                        )
                        g_feat = np.concatenate([g_static_cur, g_dynamic]).reshape(1, -1)
                    g_feat_s = (g_feat - guardian_scaler.mean_) / guardian_scaler.scale_
                    g_proba = guardian_model._Booster.predict(g_feat_s)[0]
                    g_pred = int(g_proba.argmax())
                    if g_pred == 2 and g_proba[2] >= guardian_exit_threshold:
                        oc = "GUARDIAN_MOMENTUM_EXIT" if guardian_momentum else (
                            "GUARDIAN_FULL" if act["partial_bar"] is not None else "GUARDIAN_EXIT"
                        )
                        return oc, close[j], True
                    if g_pred == 1 and g_proba[1] >= guardian_exit_threshold and act["partial_bar"] is None:
                        act["partial_bar"] = j
                        act["partial_price"] = close[j]
                        pct_partial = (close[j] - vwap) / vwap
                        if sig == SHORT:
                            pct_partial = -pct_partial
                        gross_partial = act["total_modal"] * leverage * pct_partial * GUARDIAN_PARTIAL_EXIT_RATIO
                        fee_partial = act["total_modal"] * leverage * fee_per_side * GUARDIAN_PARTIAL_EXIT_RATIO
                        act["partial_pnl"] = gross_partial - fee_partial
                        act["position_remaining"] = 1.0 - GUARDIAN_PARTIAL_EXIT_RATIO

        if guardian_delta_active and bars_held >= guardian_min_hold_bars and act["partial_bar"] is None:
            direction_num = 1 if sig == LONG else -1
            pnl_pct_now = (close[j] - vwap) / vwap * direction_num if vwap > 0 else 0.0
            tp_hit_now = (sig == LONG and high[j] >= tp_price) or (sig == SHORT and low[j] <= tp_price)
            if tp_hit_now or pnl_pct_now <= guardian_def_min_loss:
                gd_vec = _compute_guardian_delta_vector(
                    bar=j, entry_bar=entry_bar, raw_features=guardian_delta_raw,
                    gd_feats=guardian_delta_feats, close_arr=close,
                    entry_price=vwap, direction=direction_num, momentum_window=3,
                ).reshape(1, -1)
                p_exit = guardian_model.predict_proba(gd_vec)[0][0]
                if (tp_hit_now and p_exit > guardian_mom_thresh) or (
                    pnl_pct_now <= guardian_def_min_loss and p_exit > guardian_def_thresh
                ):
                    return "GUARDIAN_DELTA_EXIT", close[j], True

        if trailing_stop_enabled and bars_held >= trailing_stop_min_bars:
            if sig == LONG:
                act["best_price_trail"] = max(act["best_price_trail"], high[j])
                trail_stop = act["best_price_trail"] - trailing_stop_atr * atr_i
                if low[j] <= trail_stop:
                    return "TRAILING_STOP", trail_stop, True
            else:
                act["best_price_trail"] = min(act["best_price_trail"], low[j])
                trail_stop = act["best_price_trail"] + trailing_stop_atr * atr_i
                if high[j] >= trail_stop:
                    return "TRAILING_STOP", trail_stop, True

        if bars_held >= max_hold:
            return "TIMEOUT", close[j], True
        return None, None, False

    for i in range(n - 1):
        if active is not None and i > active["entry_bar"]:
            outcome, raw_exit, done = _check_exit(active, i)
            if done:
                _close_position(active, i, outcome, raw_exit)
                active = None

        if cooldown_enabled and i < cooldown_until:
            equity_curve.append(equity)
            continue

        sig = y_pred[i]
        if sig == FLAT:
            equity_curve.append(equity)
            continue

        if entry_price_override is not None and np.isfinite(entry_price_override[i]):
            raw_price = float(entry_price_override[i])
        else:
            raw_price = close[i]
        if slippage_enabled:
            price = raw_price * (1.0 + slippage) if sig == LONG else raw_price * (1.0 - slippage)
        else:
            price = raw_price

        trade_modal = _entry_modal(i, sig)
        if trade_modal is None:
            equity_curve.append(equity)
            continue

        atr_i = atr[i]
        sh_i = h4_swing_highs[i] if use_swing else np.nan
        sl_i = h4_swing_lows[i] if use_swing else np.nan
        if np.isnan(price) or np.isnan(atr_i) or atr_i == 0:
            equity_curve.append(equity)
            continue

        if vcb_enabled and atr_i > 0:
            vcb_start = max(0, i - vcb_lookback_bars)
            vcb_window = atr[vcb_start:i + 1]
            vcb_valid = vcb_window[~np.isnan(vcb_window)]
            if len(vcb_valid) >= vcb_lookback_bars:
                vcb_mean = vcb_valid[-vcb_lookback_bars:].mean()
                if atr_i > vcb_atr_multiplier * vcb_mean:
                    n_vcb_blocked += 1
                    equity_curve.append(equity)
                    continue

        use_sw = use_swing and not np.isnan(sh_i) and not np.isnan(sl_i)
        if swing_freshness_check and use_sw:
            if individual_swing_freshness:
                if abs(sh_i - price) / price > max_swing_deviation_pct or abs(sl_i - price) / price > max_swing_deviation_pct:
                    equity_curve.append(equity)
                    continue
            else:
                dev = abs(sh_i - price) / price if sig == LONG else abs(sl_i - price) / price
                if dev > max_swing_deviation_pct:
                    equity_curve.append(equity)
                    continue

        if structural_filter and use_sw:
            upper_bound = sh_i * (1.0 + structural_tolerance_pct)
            lower_bound = sl_i * (1.0 - structural_tolerance_pct)
            if price > upper_bound or price < lower_bound:
                equity_curve.append(equity)
                continue

        vr_i = vol_ratio[i] if vol_ratio is not None else None

        if active is not None:
            if pyramiding_same_dir and sig != active["sig"]:
                equity_curve.append(equity)
                continue
            if active["legs"] >= pyramiding_max_per_coin:
                equity_curve.append(equity)
                continue
            old_m = active["total_modal"]
            active["total_modal"] += trade_modal
            active["vwap"] = (active["vwap"] * old_m + price * trade_modal) / active["total_modal"]
            active["legs"] += 1
            active["extra_entry_fees"] += trade_modal * leverage * fee_per_side
            equity_curve.append(equity)
            continue

        tpsl = _calc_tp_sl(
            sig, price, atr_i, sh_i, sl_i, use_sw, hybrid_mode,
            tp_fallback_atr, sl_fallback_atr, swing_sl_bumper_atr, min_sl_pct,
            max_sl_atr, vr_i, volr_conditional_enabled, volr_threshold,
            volr_disable_max_sl, max_sl_volr_low, max_sl_pct_enabled,
            min_tp_atr, min_rr,
        )
        if tpsl is None:
            equity_curve.append(equity)
            continue
        tp_price, sl_price, rr = tpsl
        active = {
            "entry_bar": i, "sig": sig, "vwap": price, "total_modal": trade_modal,
            "legs": 1, "tp_price": tp_price, "sl_price": sl_price, "rr": rr,
            "tp_touched": False, "partial_bar": None, "partial_price": None,
            "partial_pnl": 0.0, "position_remaining": 1.0,
            "mfe_pnl": 0.0, "best_price_trail": price,
            "extra_entry_fees": trade_modal * leverage * fee_per_side,
        }
        equity_curve.append(equity)

    if active is not None:
        end = min(active["entry_bar"] + max_hold, n - 1)
        outcome, raw_exit, done = _check_exit(active, end)
        if not done:
            outcome, raw_exit = "TIMEOUT", close[end]
        _close_position(active, end, outcome, raw_exit)

    return _trade_summary(trades, equity_curve, modal, n_vcb_blocked)