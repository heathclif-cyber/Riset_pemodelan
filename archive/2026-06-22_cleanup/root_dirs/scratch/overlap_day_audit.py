# -*- coding: utf-8 -*-
"""Audit mendalam hari overlap holdout vs live (8-13 Jun 2026)."""
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

OUT = ROOT / "reports" / "experiments"
OVERLAP = ["2026-06-08", "2026-06-09", "2026-06-10", "2026-06-11", "2026-06-12", "2026-06-13"]
TZ = "Asia/Makassar"


def _hour_wita(ts):
    return pd.Timestamp(ts, tz="UTC").tz_convert(TZ).hour


def main():
    hold = pd.read_csv(OUT / "holdout_scale_in_trades_full.csv")
    live = pd.read_csv(OUT / "live_ic32_trades_is_live.csv")
    sigs = pd.read_csv(OUT / "live_ic32_signals.csv")

    hold = hold[hold["date_entry_wita"].isin(OVERLAP)].copy()
    live = live[live["date_entry_wita"].isin(OVERLAP)].copy()
    sigs = sigs[sigs["date_wita"].isin(OVERLAP)].copy()

    hold["hour_wita"] = hold["ts_in_wita"].str[11:13].astype(int)
    live["hour_wita"] = live["ts_in_wita"].str[11:13].astype(int)
    sigs["hour_wita"] = sigs["ts_wita"].str[11:13].astype(int)

    # Open positions tracker for holdout (approximate blocking)
    hold_sorted = hold.sort_values("ts_in_utc")
    open_pos = {}  # symbol -> {dir, out_wita}
    hold_blocked_notes = []

    day_reports = []
    mismatch_rows = []

    for day in OVERLAP:
        h = hold[hold["date_entry_wita"] == day]
        l = live[live["date_entry_wita"] == day]
        s = sigs[sigs["date_wita"] == day]

        # classify each holdout trade
        only_hold = []
        only_live = []
        matched = []
        dir_conflict = []

        live_by_sym = {}
        for _, row in l.iterrows():
            live_by_sym.setdefault(row["symbol"], []).append(row)

        hold_syms = set(h["symbol"])
        live_syms = set(l["symbol"])

        for sym in sorted(hold_syms | live_syms):
            ht = h[h["symbol"] == sym]
            lt = l[l["symbol"] == sym]
            if ht.empty and not lt.empty:
                for _, lr in lt.iterrows():
                    only_live.append({
                        "date": day, "symbol": sym, "live_dir": lr["direction"],
                        "live_hour": lr["hour_wita"], "live_pnl": lr["net_pnl"],
                        "live_conf": lr.get("signal_confidence"),
                        "reason": "only_live",
                    })
            elif lt.empty and not ht.empty:
                for _, hr in ht.iterrows():
                    # check live signal same hour bucket
                    sig_match = s[(s["symbol"] == sym) & (s["hour_wita"] == hr["hour_wita"])]
                    live_sig_dir = sig_match["direction"].iloc[0] if len(sig_match) else "NO_SIGNAL"
                    only_hold.append({
                        "date": day, "symbol": sym, "hold_dir": hr["direction"],
                        "hold_hour": hr["hour_wita"], "hold_pnl": hr["net_pnl"],
                        "live_signal_at_hour": live_sig_dir,
                        "reason": "only_holdout",
                    })
            else:
                # both have trades - check direction agreement per day
                h_dirs = ht["direction"].value_counts().to_dict()
                l_dirs = lt["direction"].value_counts().to_dict()
                if h_dirs == l_dirs and len(ht) == len(lt):
                    matched.append(sym)
                else:
                    dir_conflict.append({
                        "date": day, "symbol": sym,
                        "hold_dirs": h_dirs, "live_dirs": l_dirs,
                        "hold_n": len(ht), "live_n": len(lt),
                    })

        # signal stats
        sig_flat = int((s["direction"] == "FLAT").sum())
        sig_trade = int(s["direction"].isin(["LONG", "SHORT"]).sum())
        sig_long = int((s["direction"] == "LONG").sum())
        sig_short = int((s["direction"] == "SHORT").sum())

        day_reports.append({
            "date_wita": day,
            "hold_trades": len(h),
            "hold_long": int((h["direction"] == "LONG").sum()),
            "hold_short": int((h["direction"] == "SHORT").sum()),
            "hold_pnl": round(h["net_pnl"].sum(), 2),
            "live_trades": len(l),
            "live_long": int((l["direction"] == "LONG").sum()),
            "live_short": int((l["direction"] == "SHORT").sum()),
            "live_pnl": round(l["net_pnl"].sum(), 2),
            "live_signals_total": len(s),
            "live_signals_flat": sig_flat,
            "live_signals_long": sig_long,
            "live_signals_short": sig_short,
            "only_holdout_count": len(only_hold),
            "only_live_count": len(only_live),
            "dir_conflict_coins": len(dir_conflict),
            "matched_coins": len(matched),
        })

        mismatch_rows.extend(only_hold)
        mismatch_rows.extend(only_live)
        for dc in dir_conflict:
            dc["reason"] = "direction_conflict"
            mismatch_rows.append(dc)

    # Why only_holdout: breakdown by live signal at that hour
    only_h_df = pd.DataFrame([r for r in mismatch_rows if r.get("reason") == "only_holdout"])
    if not only_h_df.empty:
        only_h_df["live_signal_at_hour"] = only_h_df["live_signal_at_hour"].fillna("NO_SIGNAL")
        sig_breakdown = only_h_df.groupby("live_signal_at_hour").size().to_dict()
    else:
        sig_breakdown = {}

    # Position blocking analysis for holdout on Jun 11
    blocking_examples = []
    h11 = hold[hold["date_entry_wita"] == "2026-06-11"].sort_values("ts_in_utc")
    # trades still open from prior days
    prior = hold[hold["date_entry_wita"] < "2026-06-11"].sort_values("ts_in_utc")
    open_at_jun11 = {}
    for _, t in prior.iterrows():
        open_at_jun11[t["symbol"]] = {
            "dir": t["direction"],
            "out": t["ts_out_wita"],
            "pnl": t["net_pnl"],
        }
    # filter still open at Jun 11 08:00
    still_open = []
    for sym, info in open_at_jun11.items():
        if info["out"] >= "2026-06-11 08:00:00":
            still_open.append({"symbol": sym, "dir": info["dir"], "closes": info["out"]})

    # Live scale-in blocking: check if live had open pos when short signalled
    live_sorted = live.sort_values("ts_in_utc")
    live_open = {}
    live_block_candidates = []
    for _, t in live_sorted.iterrows():
        sym = t["symbol"]
        # close positions that ended before this entry
        to_del = [k for k, v in live_open.items() if v["out"] < t["ts_in_wita"]]
        for k in to_del:
            del live_open[k]
        if sym in live_open and live_open[sym]["dir"] != t["direction"]:
            live_block_candidates.append({
                "note": "would_be_blocked_by_scale_in",
                "symbol": sym,
                "open_dir": live_open[sym]["dir"],
                "attempt_dir": t["direction"],
                "ts": t["ts_in_wita"],
            })
        live_open[sym] = {"dir": t["direction"], "out": t["ts_out_wita"] or "9999"}

    report = {
        "overlap_days": OVERLAP,
        "daily": day_reports,
        "only_holdout_signal_breakdown": sig_breakdown,
        "holdout_positions_still_open_jun11_08": still_open,
        "live_scale_in_flip_attempts": live_block_candidates,
        "summary": {
            "hold_trades": len(hold),
            "live_trades": len(live),
            "only_holdout_entries": len([r for r in mismatch_rows if r.get("reason") == "only_holdout"]),
            "only_live_entries": len([r for r in mismatch_rows if r.get("reason") == "only_live"]),
            "direction_conflict_coins": len([r for r in mismatch_rows if r.get("reason") == "direction_conflict"]),
        },
    }

    pd.DataFrame(day_reports).to_csv(OUT / "overlap_daily_audit.csv", index=False)
    pd.DataFrame(mismatch_rows).to_csv(OUT / "overlap_mismatch_detail.csv", index=False)

    with open(OUT / "overlap_audit.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)

    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()