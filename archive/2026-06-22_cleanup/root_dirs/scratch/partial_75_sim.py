# -*- coding: utf-8 -*-
"""Simulasi partial 75% reinvest - timeline bulanan."""
import pandas as pd

tr = pd.read_excel(r"D:\Datatrade_ic32regime.xlsx", sheet_name="Trades_scale_in")
tr["ts_in"] = pd.to_datetime(tr["ts_in"])
period_months = ((tr["ts_in"].max() - tr["ts_in"].min()).days + 1) / 30.44
monthly_oof = 5128.75 / 75
monthly_holdout = float(tr["net_pnl"].sum()) / period_months

BASE_MODAL = 10.0
MAX_SLOTS = 21
BASE_CAP = BASE_MODAL * MAX_SLOTS
REINVEST = 0.75
WITHDRAW = 1 - REINVEST
TARGET = 12000
START = 210


def simulate(rate_monthly_on_base, label, max_m=36):
    """rate = monthly $ at $210 deployed; scales linearly with deployed cap."""
    r = rate_monthly_on_base / BASE_CAP
    trading_bal = START
    withdrawn_total = 0.0
    rows = []
    for m in range(1, max_m + 1):
        modal = BASE_MODAL * (trading_bal / START)
        deployed = min(trading_bal, modal * MAX_SLOTS)
        gross_pnl = deployed * r
        reinvest = gross_pnl * REINVEST
        withdraw = gross_pnl * WITHDRAW
        trading_bal += reinvest
        withdrawn_total += withdraw
        rows.append({
            "month": m,
            "trading_bal": trading_bal,
            "modal": modal,
            "gross_pnl": gross_pnl,
            "reinvest": reinvest,
            "withdraw": withdraw,
            "withdrawn_cum": withdrawn_total,
            "net_worth": trading_bal + withdrawn_total,
        })
        if trading_bal >= TARGET:
            break
    return label, rows


def print_table(label, rows):
    print(f"\n{label}")
    print("-" * 95)
    print(f"{'Bln':>3} {'Saldo trading':>14} {'Modal/tr':>9} {'PnL bruto':>10} "
          f"{'Reinvest':>9} {'Tarik':>8} {'Tarik kum':>10} {'Net worth':>12}")
    show = {1, 2, 3, 6, 9, 12, 15, 18, 19} | {rows[-1]["month"]}
    for row in rows:
        if row["month"] in show or row["month"] <= 3:
            print(f"{row['month']:3d} ${row['trading_bal']:12,.0f} "
                  f"${row['modal']:7.1f} ${row['gross_pnl']:8.0f} "
                  f"${row['reinvest']:7.0f} ${row['withdraw']:6.0f} "
                  f"${row['withdrawn_cum']:8,.0f} ${row['net_worth']:10,.0f}")
    hit = next(r for r in rows if r["trading_bal"] >= TARGET)
    print(f"\n  Target ${TARGET:,} saldo trading: bulan ke-{hit['month']}")
    print(f"  Profit sudah ditarik kumulatif: ${hit['withdrawn_cum']:,.0f}")
    print(f"  Total net worth (trading + tarikan): ${hit['net_worth']:,.0f}")


_, oof_rows = simulate(monthly_oof, "OOF")
print_table("PARTIAL 75% REINVEST (rate OOF konservatif) | start $210", oof_rows)

print("\n" + "=" * 95)
print("CARA PRAKTIS BULANAN")
print("=" * 95)
print("""
1. Awal bulan: catat saldo trading (modal kerja) = $210
2. Selama bulan: trade dengan modal_per_trade sesuai saldo / 21 slot
3. Akhir bulan: hitung profit bulan ini (saldo akhir - saldo awal bulan, atau sum net_pnl)
4. Reinvest 75% -> tambahkan ke saldo trading -> hitung modal baru
5. Tarik 25% -> transfer ke rekening / wallet terpisah (JANGAN dipakai trade)
6. Update modal_per_trade di inference_config = floor(saldo_trading / 21)
""")

# milestone modal updates
print("MILESTONE UPDATE modal_per_trade (OOF 75%):")
_, rows = simulate(monthly_oof, "x")
prev_modal = 10
for row in rows:
    new_modal = round(row["modal"])
    if new_modal >= prev_modal + 2:
        print(f"  Bulan {row['month']:2d}: saldo ${row['trading_bal']:,.0f} -> set modal ${new_modal}/trade "
              f"(tarikan kum ${row['withdrawn_cum']:,.0f})")
        prev_modal = new_modal
    if row["trading_bal"] >= TARGET:
        break