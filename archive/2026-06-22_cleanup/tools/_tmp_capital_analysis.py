"""Hitung modal optimal untuk target PnL $200/bulan."""
import sqlite3, json, numpy as np
from pathlib import Path
import pandas as pd

# ─── 1. Live trade data (jika tersedia) ───────────────────────────────────────
db = Path("data/live_cache/app.db")
live_trades = []
if db.exists():
    try:
        con = sqlite3.connect(db)
        rows = con.execute("""
            SELECT t.entry_price, t.exit_price, t.notional, t.pnl_usd,
                   t.direction, t.status, t.entry_time, c.symbol
            FROM trade t
            JOIN coin c ON t.coin_id = c.id
            WHERE t.status IN ('closed','CLOSED')
            AND strftime('%Y-%m', t.entry_time) = '2026-06'
        """).fetchall()
        live_trades = rows
        con.close()
        print(f"Live June trades: {len(rows)}")
        if rows:
            pnls = [r[3] for r in rows if r[3] is not None]
            nots = [r[2] for r in rows if r[2] is not None]
            wins = [p for p in pnls if p > 0]
            loss = [p for p in pnls if p < 0]
            avg_not = np.mean(nots) if nots else 25.0
            print(f"  Total PnL   : ${sum(pnls):.3f}")
            print(f"  Avg per trade: ${np.mean(pnls):.4f} (notional ${avg_not:.1f})")
            print(f"  WR: {len(wins)}/{len(pnls)} = {len(wins)/len(pnls)*100:.1f}%")
            if wins:  print(f"  AvgWin : ${np.mean(wins):.4f}")
            if loss:  print(f"  AvgLoss: ${np.mean(loss):.4f}")
    except Exception as e:
        print(f"DB error: {e}")
else:
    print("DB not found - skip live data")

print()

# ─── 2. Holdout-based projection ───────────────────────────────────────────────
print("=== HOLDOUT v2 PROJECTION ===")

csv = Path("models/runs/ic32_regime_v2/holdout_v1_vs_v2_trades.csv")
if not csv.exists():
    print("Holdout CSV not found"); exit()

df = pd.read_csv(csv)
v2 = df[df["model"] == "v2"].copy()

# Holdout period: 6 months (Nov 2025 - Apr 2026)
holdout_months = 6
n_coins = 21
notional_base = 25.0  # current live notional per trade
leverage = 5.0

n_trades_total = len(v2)
trades_per_month = n_trades_total / holdout_months
trades_per_coin_month = trades_per_month / n_coins

print(f"Total trades holdout: {n_trades_total} over {holdout_months} months")
print(f"Per month: {trades_per_month:.1f} trades | Per coin: {trades_per_coin_month:.2f} trades/month")

# PnL in ATR units. Convert to USD:
# 1 ATR ≈ ATR% of price. For $25 notional: $USD = pnl_atr × (ATR/price) × notional
# ATR/price estimasi per coin dari config (TP_ATR=2, SL_ATR=1.5)
# avg ATR/price untuk 21 altcoins ≈ 2.0-2.5%
atr_pct_estimates = {
    "conservative": 0.018,   # 1.8% (large cap BTC/ETH dominan)
    "mid":          0.022,   # 2.2% (campuran)
    "aggressive":   0.028,   # 2.8% (small cap altcoin)
}

wins = v2[v2["pnl"] > 0]
losses = v2[v2["pnl"] < 0]
wr = len(wins) / len(v2)
avg_win_atr = wins["pnl"].mean()
avg_loss_atr = losses["pnl"].mean()
net_atr_total = v2["pnl"].sum()
net_atr_monthly = net_atr_total / holdout_months

print(f"\nHoldout stats: WR={wr*100:.1f}%  AvgWin={avg_win_atr:.4f}ATR  AvgLoss={avg_loss_atr:.4f}ATR")
print(f"Net ATR/bulan: {net_atr_monthly:.2f} ATR")

print("\n=== PROYEKSI PnL/BULAN @ notional $25/trade ===")
TARGET = 200.0  # USD

for scenario, atr_pct in atr_pct_estimates.items():
    usd_per_atr = atr_pct * notional_base
    pnl_per_month_usd = net_atr_monthly * usd_per_atr
    pnl_per_trade_usd = (v2["pnl"].mean()) * usd_per_atr

    # Untuk $200/bulan
    scale_factor = TARGET / pnl_per_month_usd if pnl_per_month_usd > 0 else float("inf")
    notional_needed = notional_base * scale_factor
    margin_per_trade = notional_needed / leverage

    # Concurrent positions estimate
    median_hold_h = float(v2["bars"].median())  # 1 bar = 1 jam
    trade_freq_h = (30 * 24) / trades_per_month  # jam antar trade (seluruh 21 koin)
    avg_concurrent = n_coins * (median_hold_h / (30 * 24)) * trades_per_coin_month
    # = expected concurrent positions at any given time
    peak_concurrent = min(n_coins, max(3, int(avg_concurrent * 3)))  # safety 3x

    capital_peak = peak_concurrent * margin_per_trade
    capital_safe = capital_peak * 1.5  # +50% buffer drawdown

    print(f"\n[{scenario.upper()} ATR={atr_pct*100:.1f}%]")
    print(f"  PnL/bulan @ $25 notional : ${pnl_per_month_usd:.2f}")
    print(f"  Scale faktor             : {scale_factor:.1f}x")
    print(f"  Notional per trade needed: ${notional_needed:.0f}")
    print(f"  Margin per trade (5x)    : ${margin_per_trade:.0f}")
    print(f"  Avg concurrent positions : ~{avg_concurrent:.1f}")
    print(f"  Peak concurrent (est.)   : ~{peak_concurrent}")
    print(f"  Modal minimum (peak)     : ${capital_peak:.0f}")
    print(f"  Modal AMAN (+50% buffer) : ${capital_safe:.0f}")

print()
print("=== RINGKASAN REKOMENDASI ===")
# Pakai mid scenario
atr_pct = atr_pct_estimates["mid"]
usd_per_atr = atr_pct * notional_base
pnl_per_month_usd = net_atr_monthly * usd_per_atr
scale_factor = TARGET / pnl_per_month_usd
notional_needed = notional_base * scale_factor
margin_per_trade = notional_needed / leverage
avg_concurrent = n_coins * (v2["bars"].median() / (30 * 24)) * trades_per_coin_month
capital_safe = min(n_coins, max(3, int(avg_concurrent * 3))) * margin_per_trade * 1.5

print(f"Target $200/bulan dengan skenario medium (ATR={atr_pct*100:.1f}%):")
print(f"  Notional per trade : ${notional_needed:.0f}")
print(f"  Modal minimum aman : ~${capital_safe:.0f}")
print(f"  Dengan drawdown max holdout: -{abs(v2['pnl'].min()):.2f} ATR/trade")
