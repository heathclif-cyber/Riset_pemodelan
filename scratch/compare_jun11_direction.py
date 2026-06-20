# -*- coding: utf-8 -*-
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.live_db_bridge import load_trades, load_signals

TARGET = "2026-06-11"
HOLDOUT = Path(r"D:\Apps-Dev\Riset_pemodelan\reports\experiments\holdout_ic32_trades_apr_jun26.csv")


def wita_date(series):
    s = pd.to_datetime(series, utc=True, errors="coerce")
    return s.dt.tz_convert("Asia/Makassar").dt.strftime("%Y-%m-%d")


def main():
    live = load_trades()
    sigs = load_signals()

    live["open_d"] = wita_date(live["opened_at"])
    sigs["sig_d"] = wita_date(sigs["signal_time"])

    # --- LIVE ---
    day_t = live[(live["open_d"] == TARGET) & (live["is_live"] == 1)]
    print(f"LIVE trades is_live=1 on {TARGET}: n={len(day_t)}")
    if not day_t.empty:
        print(day_t["direction"].value_counts().to_string())
        print("\nPer coin:")
        print(day_t.groupby(["coin_symbol", "direction"]).size().unstack(fill_value=0).to_string())
        print("\nModel types:")
        print(day_t["model_type"].value_counts().to_string())

    day_sig = sigs[sigs["sig_d"] == TARGET]
    print(f"\nLIVE signals on {TARGET}: n={len(day_sig)}")
    if not day_sig.empty:
        print(day_sig["direction"].value_counts().to_string())
        print("\nPer coin (signals):")
        print(day_sig.groupby(["coin_symbol", "direction"]).size().unstack(fill_value=0).to_string())

    # trades linked to jun11 signals
    if not day_sig.empty:
        linked = day_t.merge(day_sig[["id", "coin_symbol", "direction", "confidence", "signal_time"]],
                             left_on="signal_id", right_on="id", how="left", suffixes=("", "_sig"))
        print(f"\nSignal vs trade direction mismatch: {(linked['direction'] != linked['direction_sig']).sum()}")

    # --- HOLDOUT ---
    h = pd.read_csv(HOLDOUT)
    h["d"] = wita_date(h["entry_time"])
    day_h = h[h["d"] == TARGET]
    print(f"\nHOLDOUT ic32_regime_v1 on {TARGET}: n={len(day_h)}")
    if not day_h.empty:
        print(day_h["direction"].value_counts().to_string())
        print("\nPer coin:")
        print(day_h.groupby(["coin", "direction"]).size().unstack(fill_value=0).to_string())
        print("\nH4 trend distribution:")
        if "h4_trend" in day_h.columns:
            print(day_h["h4_trend"].value_counts().to_string())
        if "trend_align" in day_h.columns:
            print("\nTrend align:")
            print(day_h["trend_align"].value_counts().to_string())

    # overlap coins
    if not day_t.empty and not day_h.empty:
        live_coins = set(day_t["coin_symbol"])
        hold_coins = set(day_h["coin"])
        both = live_coins & hold_coins
        print(f"\nCoins overlap: {len(both)}")
        rows = []
        for c in sorted(both):
            l = day_t[day_t["coin_symbol"] == c]["direction"].value_counts().to_dict()
            ho = day_h[day_h["coin"] == c]["direction"].value_counts().to_dict()
            rows.append({
                "coin": c,
                "live_L": l.get("LONG", 0),
                "live_S": l.get("SHORT", 0),
                "hold_L": ho.get("LONG", 0),
                "hold_S": ho.get("SHORT", 0),
            })
        cmp = pd.DataFrame(rows)
        print(cmp.to_string(index=False))
        mismatch = cmp[(cmp.live_L != cmp.hold_L) | (cmp.live_S != cmp.hold_S)]
        print(f"\nCoin direction count mismatch: {len(mismatch)}/{len(cmp)}")

    # hourly breakdown live
    if not day_t.empty:
        day_t = day_t.copy()
        day_t["hour_wita"] = pd.to_datetime(day_t["opened_at"], utc=True).dt.tz_convert("Asia/Makassar").dt.hour
        print("\nLIVE hourly direction:")
        print(day_t.groupby(["hour_wita", "direction"]).size().unstack(fill_value=0).to_string())

    if not day_h.empty:
        day_h = day_h.copy()
        day_h["hour_wita"] = pd.to_datetime(day_h["entry_time"], utc=True).dt.tz_convert("Asia/Makassar").dt.hour
        print("\nHOLDOUT hourly direction:")
        print(day_h.groupby(["hour_wita", "direction"]).size().unstack(fill_value=0).to_string())

    # detail table
    if not day_t.empty:
        print("\n--- LIVE detail ---")
        cols = ["coin_symbol", "direction", "signal_confidence", "opened_at", "pnl_net", "exit_reason", "model_type"]
        print(day_t[cols].sort_values("opened_at").to_string(index=False))


if __name__ == "__main__":
    main()