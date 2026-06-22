# -*- coding: utf-8 -*-
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.live_db_bridge import load_trades, load_signals

TARGET = "2026-06-11"
HOLDOUT = Path(r"D:\Apps-Dev\Riset_pemodelan\reports\experiments\holdout_ic32_trades_apr_jun26.csv")

COINS_FOCUS = ["SUIUSDT", "ARBUSDT", "TAOUSDT", "XRPUSDT", "SOLUSDT", "LINKUSDT"]


def wita_hour(ts):
    return pd.to_datetime(ts, utc=True).dt.tz_convert("Asia/Makassar").dt.strftime("%Y-%m-%d %H:00")


def main():
    sigs = load_signals()
    trades = load_trades()
    h = pd.read_csv(HOLDOUT)

    sigs["wita_h"] = wita_hour(sigs["signal_time"])
    h["wita_h"] = wita_hour(h["entry_time"])

    day_sig = sigs[sigs["wita_h"].str.startswith(TARGET)]
    day_h = h[h["wita_h"].str.startswith(TARGET)]

    print("=== LIVE signals per hour (non-FLAT only) ===")
    nf = day_sig[day_sig["direction"] != "FLAT"]
    print(nf.groupby(["wita_h", "direction"]).size().unstack(fill_value=0).to_string())

    print("\n=== HOLDOUT entries per hour ===")
    print(day_h.groupby(["wita_h", "direction"]).size().unstack(fill_value=0).to_string())

    for coin in COINS_FOCUS:
        print(f"\n{'='*60}\n{coin}")
        hs = day_h[day_h["coin"] == coin][["wita_h", "direction", "confidence", "hmm_state", "h4_trend"]]
        ls = day_sig[(day_sig["coin_symbol"] == coin) & (day_sig["direction"] != "FLAT")][
            ["wita_h", "direction", "confidence", "entry_reason"]
        ]
        lt = trades[(trades["coin_symbol"] == coin) & (trades["is_live"] == 1)].copy()
        lt["wita_h"] = wita_hour(lt["opened_at"])
        lt = lt[lt["wita_h"].str.startswith(TARGET)][["wita_h", "direction", "signal_confidence", "status"]]

        print("HOLDOUT:")
        print(hs.to_string(index=False) if not hs.empty else "  -")
        print("LIVE signals (directional):")
        print(ls.to_string(index=False) if not ls.empty else "  -")
        print("LIVE trades:")
        print(lt.to_string(index=False) if not lt.empty else "  -")

        # FLAT signals at holdout LONG hours
        hold_long_hours = set(hs[hs["direction"] == "LONG"]["wita_h"])
        if hold_long_hours:
            flats = day_sig[(day_sig["coin_symbol"] == coin) & (day_sig["wita_h"].isin(hold_long_hours))]
            print(f"Live ALL signals at holdout-LONG hours ({len(flats)} rows):")
            print(flats.groupby(["wita_h", "direction"]).size().to_string())

    # open positions blocking?
    print("\n=== Trades still OPEN entering Jun 11 (could block new entries) ===")
    trades["open_d"] = pd.to_datetime(trades["opened_at"], utc=True).dt.tz_convert("Asia/Makassar").dt.strftime("%Y-%m-%d")
    trades["close_d"] = pd.to_datetime(trades["closed_at"], utc=True).dt.tz_convert("Asia/Makassar").dt.strftime("%Y-%m-%d")
    jun11_start = pd.Timestamp("2026-06-11", tz="Asia/Makassar").tz_convert("UTC")
    blocking = trades[
        (trades["is_live"] == 1)
        & (pd.to_datetime(trades["opened_at"], utc=True) < jun11_start)
        & (
            trades["status"].eq("open")
            | (pd.to_datetime(trades["closed_at"], utc=True) >= jun11_start)
        )
    ]
    print(f"n={len(blocking)}")
    if not blocking.empty:
        print(blocking[["coin_symbol", "direction", "opened_at", "closed_at", "status"]].to_string(index=False))


if __name__ == "__main__":
    main()