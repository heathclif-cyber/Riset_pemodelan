"""Tambah sheet Daily_Trades_v2 ke ic32regimev2.xlsx yang sudah ada."""
import pandas as pd
from pathlib import Path
from openpyxl import load_workbook

RUN = Path("d:/Apps-Dev/Riset_pemodelan/models/runs/ic32_regime_v2")
OUT = Path("D:/ic32regimev2.xlsx")

df_all = pd.read_csv(RUN / "holdout_v1_vs_v2_trades.csv")
v2 = df_all[df_all["model"] == "v2"].copy()
v2["ts"] = pd.to_datetime(v2["ts"], utc=True)
v2["date"] = v2["ts"].dt.date

# Bangun ringkasan per hari
rows = []
for date, day_df in sorted(v2.groupby("date")):
    n       = len(day_df)
    wins    = (day_df["pnl"] > 0).sum()
    losses  = (day_df["pnl"] < 0).sum()
    gp      = day_df[day_df["pnl"] > 0]["pnl"].sum()
    gl      = abs(day_df[day_df["pnl"] < 0]["pnl"].sum())
    pf      = round(gp / gl, 3) if gl > 0 else None
    net     = round(day_df["pnl"].sum(), 4)
    avg_w   = round(gp / wins, 4) if wins > 0 else 0
    avg_l   = round(-gl / losses, 4) if losses > 0 else 0
    coins   = ", ".join(sorted(day_df["coin"].unique()))
    longs   = (day_df["dir"] == "LONG").sum()
    shorts  = (day_df["dir"] == "SHORT").sum()

    exit_counts = day_df["outcome"].value_counts().to_dict()
    rows.append({
        "Date":               str(date),
        "Total Trades":       n,
        "Wins":               int(wins),
        "Losses":             int(losses),
        "Win Rate %":         round(wins / n * 100, 1) if n > 0 else 0,
        "Net PnL (ATR)":      net,
        "Gross Profit (ATR)": round(gp, 4),
        "Gross Loss (ATR)":   round(gl, 4),
        "Profit Factor":      pf,
        "Avg Win (ATR)":      avg_w,
        "Avg Loss (ATR)":     avg_l,
        "LONG":               int(longs),
        "SHORT":              int(shorts),
        "Guardian Momentum":  exit_counts.get("GUARDIAN_MOMENTUM_EXIT", 0),
        "Guardian Exit":      exit_counts.get("GUARDIAN_EXIT", 0),
        "Loss (SL)":          exit_counts.get("LOSS", 0),
        "Timeout":            exit_counts.get("TIMEOUT", 0) + exit_counts.get("TIMEOUT_MOMENTUM", 0),
        "Coins Traded":       coins,
    })

df_daily = pd.DataFrame(rows)

# Tambah kolom kumulatif
df_daily["Cum PnL (ATR)"] = df_daily["Net PnL (ATR)"].cumsum().round(4)
df_daily["Cum Trades"]    = range(1, len(df_daily) + 1)
df_daily["Cum Trades"]    = df_daily["Total Trades"].cumsum()

# Tulis ke sheet baru di file yang sudah ada
with pd.ExcelWriter(str(OUT), engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
    df_daily.to_excel(writer, sheet_name="Daily_v2", index=False)

print(f"Done: {len(df_daily)} hari ditambahkan ke {OUT}")
print(f"  Periode: {df_daily['Date'].iloc[0]} -> {df_daily['Date'].iloc[-1]}")
print(f"  Total trades: {df_daily['Total Trades'].sum()}")
print(f"  Total PnL: {df_daily['Net PnL (ATR)'].sum():.4f} ATR")
