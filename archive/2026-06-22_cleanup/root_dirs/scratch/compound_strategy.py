# -*- coding: utf-8 -*-
"""Simulasi strategi compounding modal untuk scale_in."""
import math
import pandas as pd

# --- rates dari backtest ---
tr = pd.read_excel(r"D:\Datatrade_ic32regime.xlsx", sheet_name="Trades_scale_in")
tr["ts_in"] = pd.to_datetime(tr["ts_in"])
period_months = ((tr["ts_in"].max() - tr["ts_in"].min()).days + 1) / 30.44
pnl_holdout = float(tr["net_pnl"].sum())
ppt_holdout = float(tr["net_pnl"].mean())  # per trade row (scale_in aggregated)
n_trades = len(tr)
trades_per_month_h = n_trades / period_months

monthly_holdout = pnl_holdout / period_months
monthly_oof = 5128.75 / 75
ppt_oof = monthly_oof / (905 / 2.4)  # approx trades/month from holdout density

BASE_MODAL = 10.0
MAX_SLOTS = 21
BASE_CAP = BASE_MODAL * MAX_SLOTS  # $210 fully deployed
LEVERAGE = 5

# monthly return on deployed capital (linear scaling assumption)
R_H = monthly_holdout / BASE_CAP
R_O = monthly_oof / BASE_CAP


def full_compound(start, r, target=12000, max_m=60):
    bal = start
    rows = []
    for m in range(max_m + 1):
        modal_per_trade = BASE_MODAL * (bal / start)
        peak_cap = min(bal, modal_per_trade * MAX_SLOTS)
        monthly_pnl = peak_cap * r
        rows.append({
            "month": m,
            "balance": bal,
            "modal_per_trade": modal_per_trade,
            "peak_deployed": peak_cap,
            "monthly_pnl_est": monthly_pnl,
        })
        if bal >= target:
            break
        bal += monthly_pnl
    return rows


def partial_compound(start, r, reinvest_frac, target=12000, max_m=60):
    bal = start
    rows = []
    for m in range(max_m + 1):
        modal_per_trade = BASE_MODAL * (bal / start)
        peak_cap = min(bal, modal_per_trade * MAX_SLOTS)
        monthly_pnl = peak_cap * r
        rows.append((m, bal, modal_per_trade, monthly_pnl))
        if bal >= target:
            break
        bal += monthly_pnl * reinvest_frac
    return rows


def step_monthly(start, r, step_every=1, target=12000, max_m=60):
    """Naikkan modal setiap N bulan dari profit yang dikumpulkan."""
    bal = start
    modal = BASE_MODAL
    reserve = 0.0
    for m in range(1, max_m + 1):
        peak_cap = min(bal, modal * MAX_SLOTS)
        pnl = peak_cap * r
        bal += pnl
        reserve += pnl
        if m % step_every == 0 and reserve >= modal * MAX_SLOTS * 0.5:
            # bump modal 10% jika reserve cukup
            old = modal
            modal = min(modal * 1.10, bal / MAX_SLOTS)
            reserve = 0
        if bal >= target:
            return m, bal, modal
    return max_m, bal, modal


def months_full(start, r, target=12000):
    return math.log(target / start) / math.log(1 + r)


TARGET = 12000
START = 210

print("=" * 60)
print("BASE ASSUMPTIONS (scale_in, 21 koin, leverage 5x)")
print("=" * 60)
print(f"  holdout: ${monthly_holdout:.2f}/bln pada ${BASE_CAP:.0f} deployed ({R_H*100:.1f}%/bln)")
print(f"  OOF:     ${monthly_oof:.2f}/bln pada ${BASE_CAP:.0f} deployed ({R_O*100:.1f}%/bln)")
print(f"  trades/bln holdout: {trades_per_month_h:.0f}")
print(f"  ppt holdout: ${ppt_holdout:.3f}/trade")
print()

print("STRATEGI 1: FULL COMPOUND (100% reinvest, scale modal proporsional)")
print("-" * 60)
for label, r in [("OOF (konservatif)", R_O), ("Holdout (optimis)", R_H)]:
    n = months_full(START, r, TARGET)
    path = full_compound(START, r, TARGET, int(n) + 2)
    print(f"\n  [{label}] start ${START} -> ${TARGET} dalam {n:.1f} bulan")
    for row in path[0:4] + path[-2:]:
        if row["month"] <= 3 or row["balance"] >= TARGET * 0.9:
            print(f"    bln {row['month']:2d}: saldo ${row['balance']:8,.0f} | "
                  f"modal/trade ${row['modal_per_trade']:6.1f} | "
                  f"PnL/bln ~${row['monthly_pnl_est']:6.0f}")

print()
print("STRATEGI 2: PARTIAL COMPOUND (% profit di-reinvest)")
print("-" * 60)
for frac in [1.0, 0.75, 0.50, 0.25]:
    for label, r in [("OOF", R_O)]:
        rows = partial_compound(START, r, frac, TARGET)
        m_hit = next((i for i, row in enumerate(rows) if row[1] >= TARGET), len(rows))
        print(f"  reinvest {frac*100:3.0f}% ({label}): ~{m_hit} bulan -> ${rows[min(m_hit, len(rows)-1)][1]:,.0f}")

print()
print("STRATEGI 3: STEP-UP BULANAN (naik modal tiap bulan dari profit)")
print("-" * 60)
for label, r in [("OOF", R_O), ("Holdout", R_H)]:
    m, bal, modal = step_monthly(START, r, step_every=1, target=TARGET)
    print(f"  {label}: {m} bulan -> ${bal:,.0f} | modal akhir ${modal:.1f}/trade")

print()
print("STRATEGI 4: FIXED MODAL + COMPOUND MANUAL (ubah inference_config tiap milestone)")
print("-" * 60)
milestones = [
    (10, "start"),
    (15, "saldo ~$400"),
    (20, "saldo ~$700"),
    (30, "saldo ~$1,500"),
    (50, "saldo ~$4,000"),
    (100, "saldo ~$12,000+"),
]
for modal, note in milestones:
    cap = modal * MAX_SLOTS
    pnl_h = cap * R_H
    pnl_o = cap * R_O
    print(f"  modal ${modal:3d}/trade ({note}): PnL/bln OOF ${pnl_o:6.0f} | holdout ${pnl_h:6.0f}")

print()
print("RISK CHECK @ $12,000 full compound")
print("-" * 60)
modal_12k = BASE_MODAL * (12000 / START)
peak_12k = modal_12k * MAX_SLOTS
print(f"  modal/trade: ${modal_12k:.0f}")
print(f"  peak margin (21 slot): ${peak_12k:,.0f}")
print(f"  exposure 5x: ${peak_12k * LEVERAGE:,.0f}")
print(f"  1 hari buruk holdout (avg -$1.12 @ $10 modal): ~${-1.12 * (modal_12k/BASE_MODAL):.0f}")