# -*- coding: utf-8 -*-
"""Forensics: kenapa banyak loss live."""
import json
import sqlite3
from pathlib import Path

import pandas as pd

DB = Path(__file__).resolve().parents[1] / "data" / "live_cache" / "app.db"
FIX_TS = pd.Timestamp("2026-06-18 16:00:00", tz="UTC")


def lsr_from_snap(s):
    try:
        return float(json.loads(s or "{}").get("long_short_ratio", float("nan")))
    except (json.JSONDecodeError, TypeError, ValueError):
        return float("nan")


def main():
    conn = sqlite3.connect(DB)
    trades = pd.read_sql(
        """
        SELECT t.id, c.symbol, t.direction, t.opened_at, t.closed_at, t.pnl_net,
               t.exit_reason, t.quantity, t.leverage, t.hold_bars,
               s.confidence, s.feature_snapshot
        FROM trade t
        JOIN coin c ON c.id = t.coin_id
        LEFT JOIN signal s ON s.id = t.signal_id
        WHERE t.is_live = 1 AND t.status = 'closed'
        ORDER BY t.closed_at
        """,
        conn,
    )
    conn.close()

    trades["opened_at"] = pd.to_datetime(trades["opened_at"], utc=True)
    trades["closed_at"] = pd.to_datetime(trades["closed_at"], utc=True)
    trades["win"] = trades["pnl_net"] > 0
    trades["lsr"] = trades["feature_snapshot"].map(lsr_from_snap)
    trades["lsr_bad"] = trades["lsr"].isna() | (trades["lsr"] <= 0.01)

    def block(df, title):
        n = len(df)
        print(f"\n=== {title} ===")
        if n == 0:
            print("  (kosong)")
            return
        w = int(df["win"].sum())
        pnl = df["pnl_net"].sum()
        print(f"  trades: {n}  |  WR: {100*w/n:.1f}%  |  PnL: ${pnl:+.2f}  |  avg: ${pnl/n:+.3f}")
        bad = int(df["lsr_bad"].sum())
        print(f"  LSR=0/invalid: {bad} ({100*bad/n:.0f}%)")
        losses = df[~df["win"]]
        if len(losses):
            print(f"  avg loss: ${losses.pnl_net.mean():.2f}  |  worst: ${losses.pnl_net.min():.2f}")
        er = (
            df.groupby("exit_reason")
            .agg(cnt=("id", "count"), pnl=("pnl_net", "sum"), wr=("win", "mean"))
            .sort_values("cnt", ascending=False)
        )
        print("  exit breakdown:")
        for reason, row in er.head(6).iterrows():
            print(f"    {reason}: {int(row.cnt)}x  pnl=${row.pnl:+.2f}  wr={100*row.wr:.0f}%")

    print("LIVE LOSS FORENSICS")
    print(f"Total closed live: {len(trades)}")
    block(trades, "SEMUA LIVE")
    block(trades[trades["lsr_bad"]], "PERIODE BUG (LSR=0 di snapshot)")
    block(trades[~trades["lsr_bad"]], "LSR OK (fitur tidak rusak)")
    block(trades[trades["opened_at"] < FIX_TS], "Opened SEBELUM fix 18 Jun 16:00 UTC")
    block(trades[trades["opened_at"] >= FIX_TS], "Opened SETELAH fix 18 Jun 16:00 UTC")

    # Loss streak
    recent = trades.sort_values("closed_at")
    streak = 0
    for pnl in recent["pnl_net"].iloc[::-1]:
        if pnl < 0:
            streak += 1
        else:
            break
    print(f"\n=== LOSS STREAK TERAKHIR: {streak} trade ===")

    print("\n=== 12 TRADE TERAKHIR ===")
    for _, r in recent.tail(12).iterrows():
        lsr = f"{r.lsr:.3f}" if pd.notna(r.lsr) else "?"
        flag = " [LSR BUG]" if r.lsr_bad else ""
        print(
            f"  {str(r.closed_at)[:16]}  {r.symbol:<14} {r.direction:<5}  "
            f"${r.pnl_net:+.2f}  {r.exit_reason or '?':<20}  conf={r.confidence:.2f}  "
            f"lsr={lsr}  modal=${r.quantity}{flag}"
        )

    # Worst days
    trades["day"] = trades["closed_at"].dt.date
    daily = trades.groupby("day").agg(n=("id", "count"), pnl=("pnl_net", "sum"), wins=("win", "sum"))
    daily["losses"] = daily["n"] - daily["wins"]
    print("\n=== 5 HARI TERBURUK ===")
    for day, row in daily.nsmallest(5, "pnl").iterrows():
        print(f"  {day}: {int(row.n)} trade, {int(row.losses)} loss, PnL ${row.pnl:+.2f}")

    # Worst coins
    by_coin = trades.groupby("symbol").agg(n=("id", "count"), wr=("win", "mean"), pnl=("pnl_net", "sum"))
    print("\n=== 5 KOIN TERBURUK (PnL) ===")
    for sym, row in by_coin.nsmallest(5, "pnl").iterrows():
        print(f"  {sym}: {int(row.n)} trade, WR {100*row.wr:.0f}%, PnL ${row.pnl:+.2f}")

    # Modal distribution
    print("\n=== MODAL PER TRADE (quantity) ===")
    for q, g in trades.groupby("quantity"):
        print(f"  ${q}: {len(g)} trade, WR {100*g.win.mean():.1f}%, PnL ${g.pnl_net.sum():+.2f}")


if __name__ == "__main__":
    main()