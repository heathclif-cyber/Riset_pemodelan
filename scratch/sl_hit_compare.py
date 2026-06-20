# -*- coding: utf-8 -*-
"""Bandingkan sl_hit holdout vs live."""
import json
import sqlite3
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
XLSX = Path(r"D:\Datatrade_ic32regime.xlsx")
DB = ROOT / "data" / "live_cache" / "app.db"


def snap_val(s, key):
    try:
        return json.loads(s or "{}").get(key)
    except (json.JSONDecodeError, TypeError):
        return None


def main():
    h = pd.read_excel(XLSX, sheet_name="Trades")
    h["exit_norm"] = h["outcome"].astype(str).str.upper()

    print("=== HOLDOUT by exit ===")
    for ex, g in h.groupby("exit_norm"):
        w = (g["net_pnl"] > 0).sum()
        print(
            f"  {ex:<22} n={len(g):3d} WR={100*w/len(g):5.1f}% "
            f"PnL={g.net_pnl.sum():+8.2f} avg_hold={g.hold_bars.mean():.1f}"
        )
    h_sl_rate = 100 * (h["exit_norm"] == "LOSS").mean()
    print(f"SL rate holdout: {h_sl_rate:.1f}%")

    conn = sqlite3.connect(DB)
    live = pd.read_sql(
        """
        SELECT t.pnl_net, t.exit_reason, t.hold_bars, t.direction, t.opened_at,
               t.quantity, c.symbol, s.confidence, s.feature_snapshot
        FROM trade t
        JOIN coin c ON c.id = t.coin_id
        LEFT JOIN signal s ON s.id = t.signal_id
        WHERE t.is_live = 1 AND t.status = 'closed'
        """,
        conn,
    )
    conn.close()
    live["win"] = live["pnl_net"] > 0
    live["opened_at"] = pd.to_datetime(live["opened_at"], utc=True)
    live["hmm"] = live["feature_snapshot"].map(lambda s: snap_val(s, "hmm_regime_enc"))
    live["lsr"] = live["feature_snapshot"].map(lambda s: snap_val(s, "long_short_ratio"))

    print("\n=== LIVE by exit ===")
    for ex, g in live.groupby("exit_reason"):
        w = g["win"].sum()
        print(
            f"  {str(ex):<22} n={len(g):3d} WR={100*w/len(g):5.1f}% "
            f"PnL={g.pnl_net.sum():+8.2f} avg_hold={g.hold_bars.mean():.1f}"
        )
    live_sl_rate = 100 * (live["exit_reason"] == "sl_hit").mean()
    print(f"SL rate live: {live_sl_rate:.1f}%")

    sl = live[live["exit_reason"] == "sl_hit"].copy()
    print("\n=== LIVE sl_hit (29 trade) ===")
    print(f"avg conf={sl.confidence.mean():.2f}  avg hold={sl.hold_bars.mean():.1f} bars")
    print(f"LONG={int((sl.direction=='LONG').sum())} SHORT={int((sl.direction=='SHORT').sum())}")
    print("hold_bars:", sl.hold_bars.value_counts().sort_index().to_dict())
    print("worst days:")
    sl["day"] = sl["opened_at"].dt.date
    for day, g in sl.groupby("day"):
        print(f"  {day}: {len(g)} sl_hit PnL ${g.pnl_net.sum():+.2f}")

    h_loss = h[h["exit_norm"] == "LOSS"]
    print("\n=== HOLDOUT LOSS (66 trade) ===")
    print(f"avg hold={h_loss.hold_bars.mean():.1f} bars")
    print("hold_bars sample:", h_loss.hold_bars.value_counts().sort_index().head(8).to_dict())

    # Guardian gap = main WR driver
    h_g = h[h["exit_norm"] == "GUARDIAN_EXIT"]
    l_g = live[live["exit_reason"] == "guardian_exit"]
    print("\n=== GUARDIAN_EXIT (bukan SL) ===")
    print(
        f"holdout: n={len(h_g)} WR={100*(h_g.net_pnl>0).mean():.1f}% "
        f"PnL=${h_g.net_pnl.sum():+.2f}"
    )
    print(
        f"live:    n={len(l_g)} WR={100*l_g.win.mean():.1f}% "
        f"PnL=${l_g.pnl_net.sum():+.2f}"
    )

    # sl_hit PnL per trade vs holdout LOSS
    print("\n=== avg loss per SL trade ===")
    print(f"live sl_hit avg: ${sl.pnl_net.mean():.3f}")
    print(f"holdout LOSS avg: ${h_loss.net_pnl.mean():.3f}")


if __name__ == "__main__":
    main()