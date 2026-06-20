# -*- coding: utf-8 -*-
"""
Simulasi counterfactual: breadth gate + HMM controller + FLIP breadth-aware + LSTM macro veto
pada data live (signals + trades) dari VPS cache.

Usage:
  python tools/simulate_macro_gate_live.py
  python tools/simulate_macro_gate_live.py --pull   # refresh DB dari VPS dulu
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SWINT = Path("D:/Apps-Dev/swint_tradev2")
sys.path.insert(0, str(SWINT))

from tools.live_db_bridge import LOCAL_DB, load_signals, load_trades, pull_live_db
from core.cascade_utils import (
    SHORT,
    LONG,
    FLAT,
    check_macro_alignment_gate,
    check_hmm_controller_gate,
    compute_regime_flip_delta,
    apply_flip_to_proba,
)

INF_CFG_PATH = ROOT / "models/inference_config.json"
REPORT_PATH = ROOT / "reports/experiments/2026-06-18_macro_gate_live_simulation.json"

MODAL = 10.0
LEVERAGE = 5.0
FEE_PER_SIDE = 0.0004
MAX_HOLD_BARS = 48
SIM_START = "2026-05-01"  # live trading period


def _parse_snap(raw) -> dict:
    if not raw or (isinstance(raw, float) and np.isnan(raw)):
        return {}
    try:
        return json.loads(raw) if isinstance(raw, str) else dict(raw)
    except (json.JSONDecodeError, TypeError):
        return {}


def _h4_label(v) -> str:
    try:
        v = int(v)
        return {1: "UP", -1: "DOWN", 0: "FLAT"}.get(v, str(v))
    except (TypeError, ValueError):
        return "?"


def _load_proposed_cfg() -> dict:
    with open(INF_CFG_PATH, encoding="utf-8") as f:
        return json.load(f)


def _gate_reasons(row: pd.Series, cfg: dict) -> dict:
    """Evaluate all proposed gates on one signal row."""
    snap = _parse_snap(row.get("feature_snapshot"))
    direction = row["direction"]
    h4 = snap.get("h4_trend")
    try:
        h4_f = float(h4) if h4 is not None else float("nan")
    except (TypeError, ValueError):
        h4_f = float("nan")
    try:
        hmm_enc = int(snap.get("hmm_regime_enc", -1))
    except (TypeError, ValueError):
        hmm_enc = -1
    try:
        mb = float(snap.get("market_breadth", 0.5))
    except (TypeError, ValueError):
        mb = 0.5

    breadth_cfg = cfg.get("breadth_gate", {})
    hmm_ctrl = cfg.get("hmm", {}).get("controller", {})
    ra = cfg.get("regime_alignment", {})
    cascade = cfg.get("cascade", {})

    breadth_ok, breadth_lbl = check_macro_alignment_gate(direction, h4_f, mb, breadth_cfg)
    hmm_ok, hmm_lbl = check_hmm_controller_gate(direction, hmm_enc, h4_f, hmm_ctrl)

    # Recompute FLIP + LSTM adj on stored proba
    lgbm_p = snap.get("_lgbm_proba") or snap.get("lgbm_proba")
    lstm_p = snap.get("_lstm_proba") or snap.get("lstm_proba")
    conf_after = float(row.get("confidence") or 0)
    flip_lbl = snap.get("_flip_adj") or ""
    lstm_block = False
    lstm_lbl = ""

    if lgbm_p and len(lgbm_p) == 3:
        proba = np.array(lgbm_p, dtype=np.float32)
        dir_idx = LONG if direction == "LONG" else SHORT
        proba, flip_delta, flip_lbl = apply_flip_to_proba(
            proba, dir_idx, hmm_enc, h4_f, ra, market_breadth=mb,
        )
        conf_after = float(proba[dir_idx])

        if lstm_p and len(lstm_p) == 3:
            lstm_idx = int(np.argmax(lstm_p))
            opp_pen = float(cascade.get("lstm_adjust_opposite_pen", 0.65))
            bull_dom_thr = float(cascade.get("lstm_bull_veto_short_dom", 0.40))
            bear_dom_thr = float(cascade.get("lstm_bear_veto_long_dom", 0.40))
            thr_entry = float(cascade.get("confidence_threshold_entry", 0.59))
            no_veto = float(cascade.get("lstm_no_veto_threshold", 0.50))

            force_pen = (
                (dir_idx == SHORT and lstm_idx == LONG and float(lstm_p[LONG]) >= bull_dom_thr)
                or (dir_idx == LONG and lstm_idx == SHORT and float(lstm_p[SHORT]) >= bear_dom_thr)
            )
            if lstm_idx != dir_idx and lstm_idx != FLAT:
                if force_pen:
                    conf_after = max(0.0, conf_after - opp_pen)
                    lstm_lbl = f"macro_veto-{opp_pen:.2f}"
                    lstm_block = conf_after < thr_entry
                elif proba[dir_idx] <= no_veto:
                    conf_after = max(0.0, conf_after - opp_pen)
                    lstm_lbl = f"opp_pen-{opp_pen:.2f}"
                    lstm_block = conf_after < thr_entry

    short_offset = float(cfg.get("hmm", {}).get("short_offset", 0.0))
    thr_entry = float(cascade.get("confidence_threshold_entry", 0.59))
    eff_thr = thr_entry + (short_offset if direction == "SHORT" else 0.0)
    conf_block = conf_after < eff_thr - 1e-6

    blocked = not breadth_ok or not hmm_ok or lstm_block or conf_block
    blockers = []
    if not breadth_ok:
        blockers.append(breadth_lbl)
    if not hmm_ok:
        blockers.append(hmm_lbl)
    if lstm_block:
        blockers.append(lstm_lbl or "lstm_conf_block")
    if conf_block and not lstm_block:
        blockers.append(f"conf_{conf_after:.2f}<{eff_thr:.2f}")

    is_counter = (
        (direction == "SHORT" and h4_f > 0)
        or (direction == "LONG" and h4_f < 0)
    )

    return {
        "h4_trend": h4_f,
        "h4_label": _h4_label(h4_f),
        "hmm_enc": hmm_enc,
        "market_breadth": mb,
        "is_counter_trend": is_counter,
        "breadth_ok": breadth_ok,
        "hmm_ok": hmm_ok,
        "conf_after": conf_after,
        "eff_thr": eff_thr,
        "flip_lbl": flip_lbl,
        "blocked": blocked,
        "blockers": blockers,
        "block_reason": "|".join(blockers) if blockers else "pass",
    }


def _simulate_tpsl_path(
    direction: str,
    entry: float,
    tp: float,
    sl: float,
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
) -> tuple[str, float]:
    """Simplified TP/SL/time-exit (no guardian). Returns (exit_reason, pnl_pct_leveraged)."""
    for i in range(min(len(highs), MAX_HOLD_BARS)):
        h, l = float(highs[i]), float(lows[i])
        if direction == "LONG":
            if l <= sl:
                raw = (sl - entry) / entry
                return "SL", raw * LEVERAGE * 100
            if h >= tp:
                raw = (tp - entry) / entry
                return "TP", raw * LEVERAGE * 100
        else:
            if h >= sl:
                raw = (entry - sl) / entry
                return "SL", raw * LEVERAGE * 100
            if l <= tp:
                raw = (entry - tp) / entry
                return "TP", raw * LEVERAGE * 100
    # time exit
    c = float(closes[min(len(closes) - 1, MAX_HOLD_BARS - 1)])
    if direction == "LONG":
        raw = (c - entry) / entry
    else:
        raw = (entry - c) / entry
    return "TIME", raw * LEVERAGE * 100


def _pnl_usd(pnl_pct: float) -> float:
    """pnl_pct = leveraged return in percent (e.g. -5.6 = -5.6%)."""
    gross = (pnl_pct / 100.0) * MODAL
    fee = 2 * FEE_PER_SIDE * MODAL * LEVERAGE
    return gross - fee


def _fetch_klines_after(client, symbol: str, start_ts: pd.Timestamp) -> pd.DataFrame | None:
    start_ms = int(start_ts.timestamp() * 1000)
    end_ms = start_ms + (MAX_HOLD_BARS + 2) * 3_600_000
    try:
        raw = client.get_klines(symbol, "1h", start_ms, end_ms, limit=MAX_HOLD_BARS + 2)
    except Exception:
        return None
    if not raw:
        return None
    df = pd.DataFrame(raw, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "qav", "trades", "tbav", "tqav", "ignore",
    ])
    for c in ("high", "low", "close"):
        df[c] = df[c].astype(float)
    return df


def _simulate_signals_klines(enriched: pd.DataFrame) -> dict:
    """Forward-simulate semua directional signal (entry+TP/SL) via Binance H1 klines."""
    from core.binance_client import BinanceClient

    sim_start = pd.Timestamp(SIM_START, tz="UTC")
    sig = enriched.copy()
    if "signal_time" not in sig.columns:
        raise KeyError("signal_time missing — pass merged dir_signals+gates dataframe")
    sig["_st"] = pd.to_datetime(sig["signal_time"], utc=True)
    sig = sig[sig["_st"] >= sim_start]
    sig = sig[sig["tp_price"].notna() & sig["sl_price"].notna() & sig["entry_price"].notna()]
    if sig.empty:
        return {"error": "no_signals_with_tpsl"}

    client = BinanceClient()
    kline_cache: dict[str, pd.DataFrame] = {}
    results = []

    for _, row in sig.iterrows():
        sym = row["coin_symbol"]
        t0 = pd.Timestamp(row["_st"])

        cache_key = f"{sym}_{t0.floor('h')}"
        if cache_key not in kline_cache:
            kline_cache[cache_key] = _fetch_klines_after(client, sym, t0)

        kl = kline_cache.get(cache_key)
        if kl is None or len(kl) < 2:
            continue

        # Bar 0 = entry bar; forward from bar 1
        highs = kl["high"].values[1:]
        lows = kl["low"].values[1:]
        closes = kl["close"].values[1:]

        exit_r, pnl_pct = _simulate_tpsl_path(
            row["direction"], float(row["entry_price"]),
            float(row["tp_price"]), float(row["sl_price"]),
            highs, lows, closes,
        )
        results.append({
            "signal_id": row["signal_id"],
            "symbol": sym,
            "direction": row["direction"],
            "entry_price": row["entry_price"],
            "blocked": bool(row["blocked"]),
            "exit_reason": exit_r,
            "pnl_pct": pnl_pct,
            "pnl_usd": _pnl_usd(pnl_pct),
            "h4_label": row.get("h4_label"),
            "market_breadth": row.get("market_breadth"),
            "block_reason": row.get("block_reason"),
        })

    if not results:
        return {"error": "klines_fetch_failed"}

    rdf = pd.DataFrame(results)

    def _portfolio(df):
        n = len(df)
        if n == 0:
            return {"n": 0, "pnl": 0.0, "wr": 0.0, "pf": 0.0}
        wins = df[df["pnl_usd"] > 0]["pnl_usd"].sum()
        losses = abs(df[df["pnl_usd"] < 0]["pnl_usd"].sum())
        return {
            "n": n,
            "pnl": round(float(df["pnl_usd"].sum()), 2),
            "wr": round(float((df["pnl_usd"] > 0).mean()) * 100, 1),
            "pf": round(float(wins / losses), 2) if losses > 0 else 99.0,
            "long_n": int((df["direction"] == "LONG").sum()),
            "short_n": int((df["direction"] == "SHORT").sum()),
        }

    all_p = _portfolio(rdf)
    kept_p = _portfolio(rdf[~rdf["blocked"]])
    blocked_p = _portfolio(rdf[rdf["blocked"]])

    # Variant: breadth gate only
    breadth_only_block = rdf["block_reason"].str.contains("breadth_bull|breadth_bear", na=False)
    breadth_kept = _portfolio(rdf[~breadth_only_block])

    return {
        "simulated_signals": int(len(rdf)),
        "period_from": SIM_START,
        "note": "Simplified TP/SL/TIME exit, no guardian, $10/5x",
        "all_directional": all_p,
        "proposed_gates_kept": kept_p,
        "proposed_gates_blocked": blocked_p,
        "breadth_gate_only_kept": breadth_kept,
        "losses_in_blocked": int((rdf[rdf["blocked"]]["pnl_usd"] < 0).sum()),
        "wins_in_blocked": int((rdf[rdf["blocked"]]["pnl_usd"] > 0).sum()),
        "losses_in_kept": int((rdf[~rdf["blocked"]]["pnl_usd"] < 0).sum()),
        "mitigated_loss_usd": round(
            float(-rdf[rdf["blocked"] & (rdf["pnl_usd"] < 0)]["pnl_usd"].sum()), 2
        ),
        "sacrificed_win_usd": round(
            float(rdf[rdf["blocked"] & (rdf["pnl_usd"] > 0)]["pnl_usd"].sum()), 2
        ),
        "worst_blocked": rdf[rdf["blocked"]].nsmallest(5, "pnl_usd")[
            ["symbol", "direction", "pnl_usd", "exit_reason", "block_reason"]
        ].to_dict("records"),
        "best_kept": rdf[~rdf["blocked"]].nlargest(5, "pnl_usd")[
            ["symbol", "direction", "pnl_usd", "exit_reason"]
        ].to_dict("records"),
    }


def run_simulation(pull: bool = False) -> dict:
    if pull:
        pull_live_db(force=True)

    cfg = _load_proposed_cfg()
    signals = load_signals(LOCAL_DB)
    trades = load_trades(LOCAL_DB)

    # Filter live period (is_live trades)
    live_trades = trades[trades["is_live"] == 1].copy()
    closed_live = live_trades[live_trades["closed_at"].notna()].copy()

    dir_signals = signals[signals["direction"].isin(["LONG", "SHORT"])].copy()
    print(f"Directional signals: {len(dir_signals)}")
    print(f"Live closed trades: {len(closed_live)}")

    # Enrich signals with gate eval
    gate_rows = []
    for _, row in dir_signals.iterrows():
        g = _gate_reasons(row, cfg)
        g["signal_id"] = row["id"]
        g["symbol"] = row["coin_symbol"]
        g["direction"] = row["direction"]
        g["confidence_orig"] = row["confidence"]
        g["entry_price"] = row["entry_price"]
        g["signal_time"] = row["signal_time"]
        g["entry_reason"] = row.get("entry_reason", "")
        gate_rows.append(g)
    gates = pd.DataFrame(gate_rows)

    # Merge trades
    merged = closed_live.merge(
        gates,
        left_on="signal_id",
        right_on="signal_id",
        how="left",
        suffixes=("", "_gate"),
    )

    # Signals that opened live trades
    opened_sig_ids = set(closed_live["signal_id"].dropna().astype(int))
    gates["opened_live"] = gates["signal_id"].isin(opened_sig_ids)

    # --- Portfolio stats: ACTUAL live closed ---
    actual_pnl = closed_live["pnl_net"].sum()
    actual_wr = (closed_live["pnl_net"] > 0).mean()
    actual_n = len(closed_live)

    # --- Counterfactual: block trades that would fail new gates ---
    # For merged trades with gate info
    has_gate = merged["blocked"].notna()
    would_block = merged.loc[has_gate, "blocked"] == True
    kept = merged.loc[has_gate & ~would_block]
    blocked_trades = merged.loc[has_gate & would_block]

    cf_pnl = kept["pnl_net"].sum()
    cf_n = len(kept)
    cf_wr = (kept["pnl_net"] > 0).mean() if cf_n else 0.0

    losses_blocked = blocked_trades[blocked_trades["pnl_net"] < 0]
    wins_blocked = blocked_trades[blocked_trades["pnl_net"] > 0]
    loss_saved = -losses_blocked["pnl_net"].sum()
    win_lost = wins_blocked["pnl_net"].sum()

    # --- Signal-level analysis (all directional, incl not traded) ---
    n_dir = len(gates)
    n_blocked_sig = gates["blocked"].sum()
    n_pass_sig = n_dir - n_blocked_sig

    # Counter-trend SHORT in bull
    bull_short = gates[
        (gates["direction"] == "SHORT")
        & (gates["is_counter_trend"])
        & (gates["market_breadth"] >= 0.70)
    ]
    bull_short_blocked = bull_short["blocked"].sum()

    # By blocker type on trades
    blocker_stats = {}
    for _, t in blocked_trades.iterrows():
        for b in str(t.get("block_reason", "")).split("|"):
            if not b:
                continue
            if b not in blocker_stats:
                blocker_stats[b] = {"n": 0, "pnl": 0.0, "losses": 0}
            blocker_stats[b]["n"] += 1
            blocker_stats[b]["pnl"] += float(t["pnl_net"])
            if t["pnl_net"] < 0:
                blocker_stats[b]["losses"] += 1

    # Direction split on kept vs blocked
    def _dir_split(df, name):
        out = {}
        for d in ("LONG", "SHORT"):
            sub = df[df["direction"] == d]
            out[d] = {
                "n": len(sub),
                "pnl": float(sub["pnl_net"].sum()) if len(sub) else 0.0,
                "wr": float((sub["pnl_net"] > 0).mean()) if len(sub) else 0.0,
            }
        return out

    # Entry reason analysis: signals rejected by filter chain before trade
    traded_dir = dir_signals.merge(
        trades[["signal_id", "is_live", "pnl_net", "closed_at"]],
        left_on="id",
        right_on="signal_id",
        how="left",
    )
    traded_dir["trade_opened"] = traded_dir["signal_id"].notna() & (traded_dir["is_live"] == 1)
    rejected_signals = traded_dir[
        traded_dir["direction"].isin(["LONG", "SHORT"])
        & ~traded_dir["trade_opened"].fillna(False)
        & traded_dir["entry_reason"].astype(str).str.contains("DITOLAK", na=False)
    ]

    # Simulate: if we only took signals that PASS new gates AND opened before
    pass_and_opened = gates[gates["opened_live"] & ~gates["blocked"]]

    enriched = dir_signals.merge(
        gates, left_on="id", right_on="signal_id", how="left", suffixes=("", "_gate"),
    )
    print("\n[KLINES] Simulating signal outcomes (TP/SL forward)...")
    try:
        kline_sim = _simulate_signals_klines(enriched)
    except Exception as e:
        import traceback
        kline_sim = {"error": str(e)}
        print(f"  Klines sim failed: {e}")
        traceback.print_exc()

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "db_cache": str(LOCAL_DB),
        "config": {
            "breadth_gate": cfg.get("breadth_gate"),
            "hmm_controller": cfg.get("hmm", {}).get("controller"),
            "short_offset": cfg.get("hmm", {}).get("short_offset"),
            "regime_breadth_aware": cfg.get("regime_alignment", {}).get("breadth_aware"),
        },
        "signal_summary": {
            "directional_total": int(n_dir),
            "would_block": int(n_blocked_sig),
            "would_pass": int(n_pass_sig),
            "block_rate_pct": round(n_blocked_sig / max(n_dir, 1) * 100, 1),
            "bull_counter_short_signals": int(len(bull_short)),
            "bull_counter_short_blocked": int(bull_short_blocked),
        },
        "live_trades_actual": {
            "n_closed": int(actual_n),
            "pnl_net": round(float(actual_pnl), 2),
            "wr_pct": round(float(actual_wr) * 100, 1),
            "long_short": _dir_split(closed_live, "actual"),
        },
        "live_trades_counterfactual": {
            "n_kept": int(cf_n),
            "n_blocked": int(len(blocked_trades)),
            "pnl_net_kept": round(float(cf_pnl), 2),
            "wr_pct_kept": round(float(cf_wr) * 100, 1),
            "losses_mitigated_n": int(len(losses_blocked)),
            "losses_mitigated_usd": round(float(loss_saved), 2),
            "wins_sacrificed_n": int(len(wins_blocked)),
            "wins_sacrificed_usd": round(float(win_lost), 2),
            "net_improvement_usd": round(float(cf_pnl - actual_pnl), 2),
            "long_short_kept": _dir_split(kept, "kept"),
            "long_short_blocked": _dir_split(blocked_trades, "blocked"),
        },
        "blocker_breakdown_on_trades": blocker_stats,
        "blocked_loss_trades_detail": [
            {
                "symbol": r["coin_symbol"],
                "direction": r["direction"],
                "pnl_net": round(float(r["pnl_net"]), 2),
                "pnl_pct": round(float(r["pnl_pct"]), 2) if pd.notna(r.get("pnl_pct")) else None,
                "h4": r.get("h4_label"),
                "breadth": round(float(r.get("market_breadth", 0.5)), 2),
                "block_reason": r.get("block_reason"),
                "entry_price": r.get("entry_price"),
                "opened_at": str(r.get("opened_at", ""))[:16],
            }
            for _, r in losses_blocked.sort_values("pnl_net").iterrows()
        ],
        "rejected_by_filter_count": int(len(rejected_signals)),
        "signal_klines_simulation": kline_sim,
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    _print_report(report, blocked_trades, losses_blocked, bull_short, gates, kline_sim)
    return report


def _print_report(report, blocked_trades, losses_blocked, bull_short, gates, kline_sim):
    print("\n" + "=" * 70)
    print("SIMULASI MACRO GATE — DATA LIVE")
    print("=" * 70)

    ss = report["signal_summary"]
    print(f"\n[SIGNALS] Directional: {ss['directional_total']}")
    print(f"  Would BLOCK: {ss['would_block']} ({ss['block_rate_pct']}%)")
    print(f"  Would PASS:  {ss['would_pass']}")
    print(f"  Bull+counter SHORT (breadth>=0.7): {ss['bull_counter_short_signals']} "
          f"-> blocked {ss['bull_counter_short_blocked']}")

    act = report["live_trades_actual"]
    cf = report["live_trades_counterfactual"]
    print(f"\n[LIVE TRADES CLOSED] n={act['n_closed']}")
    print(f"  ACTUAL:  PnL ${act['pnl_net']:+.2f}  WR {act['wr_pct']}%")
    print(f"    LONG:  n={act['long_short']['LONG']['n']} PnL ${act['long_short']['LONG']['pnl']:+.2f} "
          f"WR {act['long_short']['LONG']['wr']*100:.1f}%")
    print(f"    SHORT: n={act['long_short']['SHORT']['n']} PnL ${act['long_short']['SHORT']['pnl']:+.2f} "
          f"WR {act['long_short']['SHORT']['wr']*100:.1f}%")

    print(f"\n[COUNTERFACTUAL — apply proposed gates]")
    print(f"  KEPT:    n={cf['n_kept']}  PnL ${cf['pnl_net_kept']:+.2f}  WR {cf['wr_pct_kept']}%")
    print(f"  BLOCKED: n={cf['n_blocked']}")
    print(f"  Losses mitigated: {cf['losses_mitigated_n']} trades, ${cf['losses_mitigated_usd']:.2f} saved")
    print(f"  Wins sacrificed:  {cf['wins_sacrificed_n']} trades, ${cf['wins_sacrificed_usd']:.2f} lost")
    print(f"  NET improvement: ${cf['net_improvement_usd']:+.2f} "
          f"(actual ${act['pnl_net']:+.2f} -> counterfactual ${cf['pnl_net_kept']:+.2f})")

    if report["blocker_breakdown_on_trades"]:
        print("\n[BLOCKER BREAKDOWN on blocked trades]")
        for b, st in sorted(report["blocker_breakdown_on_trades"].items(), key=lambda x: -x[1]["n"]):
            print(f"  {b}: n={st['n']} pnl=${st['pnl']:+.2f} losses={st['losses']}")

    if len(losses_blocked):
        print("\n[WORST LOSSES THAT WOULD BE BLOCKED]")
        for d in report["blocked_loss_trades_detail"][:10]:
            print(f"  {d['opened_at']} {d['symbol']:12} {d['direction']:5} "
                  f"PnL ${d['pnl_net']:+.2f} h4={d['h4']} mb={d['breadth']} "
                  f"entry={d['entry_price']} [{d['block_reason']}]")

    # Profitability check
    profitable = cf["pnl_net_kept"] > 0
    print(f"\n[VERDICT] Setup proposed {'PROFITABLE' if profitable else 'STILL LOSS'} "
          f"on live closed trades (${cf['pnl_net_kept']:+.2f})")
    if not profitable and cf["net_improvement_usd"] > 0:
        print(f"  But improves vs actual by ${cf['net_improvement_usd']:+.2f} "
              f"({cf['losses_mitigated_n']} losses avoided)")

    ks = report.get("signal_klines_simulation", {})
    if ks and "error" not in ks:
        print(f"\n[SIGNAL KLINE SIMULATION] desde {ks.get('period_from')} n={ks.get('simulated_signals')}")
        print(f"  {ks.get('note')}")
        a, k, b = ks["all_directional"], ks["proposed_gates_kept"], ks["proposed_gates_blocked"]
        print(f"  ALL signals traded:     n={a['n']} PnL ${a['pnl']:+.2f} WR {a['wr']}% PF {a['pf']}")
        print(f"  PROPOSED gates (kept):  n={k['n']} PnL ${k['pnl']:+.2f} WR {k['wr']}% PF {k['pf']}")
        print(f"  Blocked by gates:       n={b['n']} PnL ${b['pnl']:+.2f} (losses mitigated ${ks['mitigated_loss_usd']:.2f})")
        bk = ks.get("breadth_gate_only_kept", {})
        if bk:
            print(f"  Breadth gate ONLY kept: n={bk['n']} PnL ${bk['pnl']:+.2f} WR {bk['wr']}%")
        if k["pnl"] > 0:
            print(f"  >> KLINE SIM: setup PROFITABLE jika hanya trade signal yang lolos gate")
        elif k["pnl"] > a["pnl"]:
            print(f"  >> KLINE SIM: gate meningkatkan PnL ${k['pnl'] - a['pnl']:+.2f} vs trade semua signal")

    print(f"\nReport: {REPORT_PATH}")
    print("=" * 70)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pull", action="store_true", help="Pull fresh DB from VPS")
    args = ap.parse_args()
    run_simulation(pull=args.pull)


if __name__ == "__main__":
    main()