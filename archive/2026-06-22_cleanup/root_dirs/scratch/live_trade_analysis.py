"""
scratch/live_trade_analysis.py
Analisis live trading — filter per model, bukan campur semua.
"""
import sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

LIVE_CSV = Path(r"D:\Apps-Dev\Riset_pemodelan\scratch\live_trades_update.csv")

def banner(t, w=65):
    print(f"\n{'='*w}\n  {t}\n{'='*w}")

df = pd.read_csv(LIVE_CSV)
df.columns = df.columns.str.strip()
df["Opened"] = pd.to_datetime(df["Opened"], errors="coerce")
df["Closed"] = pd.to_datetime(df["Closed"], errors="coerce")
df["win"]    = (df["PnL ($)"] > 0).astype(float)
df["is_gdn"] = df["Exit Reason"].str.lower().str.contains("guardian", na=False).astype(int)
df["is_sl"]  = df["Exit Reason"].str.lower().str.contains("sl_hit", na=False).astype(int)
df["pnl"]    = pd.to_numeric(df["PnL ($)"], errors="coerce")
df["conf"]   = pd.to_numeric(df["Conf"], errors="coerce")
df["open_date"] = df["Opened"].dt.date
df["is_short"]  = (df["Direction"] == "SHORT").astype(int)

# ── 1. Model breakdown ────────────────────────────────────────────────────────
banner("Semua Model — Breakdown")
model_grp = df[df["Status"] == "closed"].groupby("Model").agg(
    trades=("win", "count"),
    wr=("win", "mean"),
    pnl=("pnl", "sum"),
    sl_rate=("is_sl", "mean"),
    avg_hold=("Hold Bars", "mean"),
).sort_values("trades", ascending=False)

print(f"\n  {'Model':<42} {'Trades':>7} {'WR':>7} {'PnL':>9} {'SL%':>7} {'Hold':>6}")
for model, row in model_grp.iterrows():
    print(f"  {str(model):<42} {int(row['trades']):>7} {row['wr']:>7.1%} "
          f"{row['pnl']:>+9.2f} {row['sl_rate']:>7.1%} {row['avg_hold']:>6.1f}")

# ── 2. ic32_regime_v1 saja ────────────────────────────────────────────────────
ic32 = df[df["Model"] == "ic32_regime_v1"].copy()
ic32_closed = ic32[ic32["Status"] == "closed"].copy()
ic32_open   = ic32[ic32["Status"] == "open"].copy()

banner(f"ic32_regime_v1 Closed Trades (N={len(ic32_closed)})")
if len(ic32_closed) == 0:
    print("  Tidak ada closed trades dari ic32_regime_v1")
else:
    overall_wr = ic32_closed["win"].mean()
    overall_pnl = ic32_closed["pnl"].sum()
    gdn_exits = ic32_closed[ic32_closed["is_gdn"] == 1]
    sl_exits  = ic32_closed[ic32_closed["is_sl"] == 1]
    gdn_wr    = gdn_exits["win"].mean() if len(gdn_exits) > 0 else float("nan")

    print(f"\n  Total trades : {len(ic32_closed)}")
    print(f"  Win Rate     : {overall_wr:.1%}  {'!! ALERT' if overall_wr < 0.631 else 'OK'}")
    print(f"  Net PnL      : ${overall_pnl:+.2f}")
    print(f"  PnL/trade    : ${overall_pnl/len(ic32_closed):+.4f}")
    print(f"  Guardian WR  : {gdn_wr:.1%}  ({len(gdn_exits)} trades)  {'!! ALERT' if not np.isnan(gdn_wr) and gdn_wr < 0.752 else 'OK'}")
    print(f"  SL rate      : {ic32_closed['is_sl'].mean():.1%}  (baseline 17.3%)")
    print(f"  Avg hold     : {ic32_closed['Hold Bars'].mean():.1f} bar")
    print(f"  SHORT pct    : {ic32_closed['is_short'].mean():.1%}")
    print(f"  Avg conf     : {ic32_closed['conf'].mean():.4f}")

    # Daily breakdown
    print(f"\n  Daily breakdown:")
    print(f"  {'Date':<14} {'Trades':>7} {'WR':>7} {'PnL':>9} {'SL%':>6} {'Gdn WR':>9}")
    daily = ic32_closed.groupby("open_date")
    for d, grp in daily:
        gdn_sub = grp[grp["is_gdn"] == 1]
        g_wr    = gdn_sub["win"].mean() if len(gdn_sub) > 0 else float("nan")
        g_str   = f"{g_wr:.0%}" if not np.isnan(g_wr) else "N/A"
        flag    = " !!" if grp["win"].mean() < 0.50 else ""
        print(f"  {str(d):<14} {len(grp):>7} {grp['win'].mean():>7.1%}{flag} "
              f"{grp['pnl'].sum():>+9.2f} {grp['is_sl'].mean():>6.1%} {g_str:>9}")

    # ── 3. Direction split ────────────────────────────────────────────────────
    banner("ic32: LONG vs SHORT breakdown")
    for dir_label, mask in [("LONG", ic32_closed["Direction"] == "LONG"),
                             ("SHORT", ic32_closed["Direction"] == "SHORT")]:
        sub = ic32_closed[mask]
        if len(sub) == 0:
            continue
        print(f"\n  {dir_label} (n={len(sub)}): WR={sub['win'].mean():.1%}  "
              f"PnL={sub['pnl'].sum():+.2f}  SL={sub['is_sl'].mean():.1%}  "
              f"Avg hold={sub['Hold Bars'].mean():.1f} bar")

    # ── 4. Guardian exit analysis ─────────────────────────────────────────────
    banner("ic32: Guardian exit analysis")
    exit_map = {}
    for reason in ic32_closed["Exit Reason"].dropna().unique():
        grp = ic32_closed[ic32_closed["Exit Reason"] == reason]
        exit_map[reason] = {
            "n":   len(grp),
            "wr":  grp["win"].mean(),
            "pnl": grp["pnl"].sum(),
        }
    print(f"\n  {'Exit Reason':<30} {'N':>6} {'WR':>8} {'PnL':>9}")
    for reason, stats in sorted(exit_map.items(), key=lambda x: -x[1]["n"]):
        print(f"  {str(reason):<30} {stats['n']:>6} {stats['wr']:>8.1%} {stats['pnl']:>+9.2f}")

    # ── 5. Conf bucket analysis ───────────────────────────────────────────────
    banner("ic32: WR per confidence bucket (live)")
    bins  = [0.58, 0.62, 0.66, 0.70, 0.74, 0.78, 1.00]
    labs  = ["0.58-0.62","0.62-0.66","0.66-0.70","0.70-0.74","0.74-0.78","0.78+"]
    ic32_closed["conf_bkt"] = pd.cut(ic32_closed["conf"], bins=bins, labels=labs)
    conf_grp = ic32_closed.groupby("conf_bkt", observed=True)
    print(f"\n  {'Bucket':<14} {'N':>6} {'WR':>8} {'PnL':>9}")
    for bkt, grp in conf_grp:
        if len(grp) == 0:
            continue
        print(f"  {str(bkt):<14} {len(grp):>6} {grp['win'].mean():>8.1%} "
              f"{grp['pnl'].sum():>+9.2f}")

    # ── 6. H4 Trend alignment ─────────────────────────────────────────────────
    banner("ic32: H4 Trend vs Direction alignment")
    ic32_closed["h4_str"] = ic32_closed["H4 Trend"].astype(str).str.upper()
    for h4, h4_grp in ic32_closed.groupby("h4_str"):
        longs  = h4_grp[h4_grp["Direction"] == "LONG"]
        shorts = h4_grp[h4_grp["Direction"] == "SHORT"]
        print(f"\n  H4 Trend = {h4} (n={len(h4_grp)}):")
        if len(longs) > 0:
            print(f"    LONG  n={len(longs):>3}  WR={longs['win'].mean():.1%}  PnL={longs['pnl'].sum():+.2f}")
        if len(shorts) > 0:
            print(f"    SHORT n={len(shorts):>3}  WR={shorts['win'].mean():.1%}  PnL={shorts['pnl'].sum():+.2f}")

# ── 7. Open positions ─────────────────────────────────────────────────────────
banner(f"Open Positions sekarang (N={len(ic32_open)})")
if len(ic32_open) > 0:
    print(f"\n  {'Opened':<18} {'Coin':<16} {'Dir':<6} {'Conf':>6} {'Hold':>6}")
    for _, r in ic32_open.sort_values("Opened").iterrows():
        print(f"  {str(r['Opened']):<18} {r['Coin']:<16} {r['Direction']:<6} "
              f"{r['Conf']:>6.2f} {int(r['Hold Bars']):>4} bar")
