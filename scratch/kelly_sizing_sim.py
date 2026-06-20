"""
scratch/kelly_sizing_sim.py
Simulasi confidence-weighted sizing vs flat sizing.

Dari EV analysis:
  conf 0.58-0.62: Kelly=+0.013  (marginal)
  conf 0.62-0.66: Kelly=+0.068
  conf 0.66-0.70: Kelly=+0.122
  conf 0.70-0.74: Kelly=+0.166
  conf 0.74-0.78: Kelly=+0.220
  conf 0.78+:     Kelly=+0.284  (strong)

Tiga skenario sizing:
  A. Flat      : $10/trade semua
  B. Kelly raw : proporsional Kelly fraction * capital
  C. Kelly cap : capped 1x-3x flat ($10-$30), proporsional Kelly
"""
import sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

OOF_PATH = ROOT / "data" / "meta_labels" / "ic32_oof_trades.parquet"
df = pd.read_parquet(OOF_PATH)

df["conf"] = np.where(df["direction"] == 1, df["conf_long"], df["conf_short"])
df["hour"] = pd.to_datetime(df["timestamp"], utc=True).dt.hour

MODAL_BASE = 10.0  # USD per trade flat
CAPITAL    = 500.0  # modal total diasumsikan untuk Kelly fraction

N = len(df)
BASE_PROD_TRADES = 2434  # jumlah trades produksi 5 bulan
SCALE = BASE_PROD_TRADES / N

def banner(t, w=65):
    print(f"\n{'='*w}\n  {t}\n{'='*w}")

# ── Hitung Kelly per bucket ───────────────────────────────────────────────────
conf_bins  = [0.58, 0.62, 0.66, 0.70, 0.74, 0.78, 1.00]
conf_labs  = ["0.58-0.62","0.62-0.66","0.66-0.70","0.70-0.74","0.74-0.78","0.78+"]
df["bucket"] = pd.cut(df["conf"], bins=conf_bins, labels=conf_labs)

kelly_map = {}
for bkt, grp in df.groupby("bucket", observed=True):
    wins  = grp[grp["win"] == 1]["net_pnl"]
    loss  = grp[grp["win"] == 0]["net_pnl"]
    wr    = grp["win"].mean()
    avg_w = wins.mean() if len(wins) > 0 else 0.0
    avg_l = loss.mean() if len(loss) > 0 else 0.0
    payoff = abs(avg_w / avg_l) if avg_l != 0 else 0.0
    kelly = wr - (1 - wr) / payoff if payoff > 0 else 0.0
    kelly_map[bkt] = max(kelly, 0.0)  # floor 0 — jangan sizing negatif

df["kelly_f"] = df["bucket"].map(kelly_map).astype(float)

# ── Skenario A: Flat $10 ──────────────────────────────────────────────────────
df["size_A"]   = MODAL_BASE
df["pnl_A"]    = df["net_pnl"]  # sudah per-$10

# ── Skenario B: Kelly raw (fraksi dari CAPITAL) ───────────────────────────────
# size = Kelly_f * CAPITAL, normalized ke MODAL_BASE rata-rata
# normalisasi supaya perbandingan fair (total modal deployed sama)
raw_kelly_mean = df["kelly_f"].mean()
norm_factor    = MODAL_BASE / (raw_kelly_mean * CAPITAL) if raw_kelly_mean > 0 else 1.0
df["size_B"]   = (df["kelly_f"] * CAPITAL * norm_factor).clip(lower=1.0)
# scale PnL: net_pnl sudah dinormalisasi ke $10 flat, jadi multiply ratio
df["pnl_B"]    = df["net_pnl"] * (df["size_B"] / MODAL_BASE)

# ── Skenario C: Kelly 1x-3x capped ($10-$30) ─────────────────────────────────
# Pendekatan praktis: conf bucket -> multiplier 1x/1.5x/2x/2.5x/3x
kelly_vals = sorted(kelly_map.values())
kelly_min  = min(v for v in kelly_vals if v > 0)
kelly_max  = max(kelly_vals)

def kelly_to_mult(k, k_min=kelly_min, k_max=kelly_max, mult_min=1.0, mult_max=3.0):
    if k <= 0:
        return mult_min
    norm = (k - k_min) / (k_max - k_min + 1e-9)
    return mult_min + norm * (mult_max - mult_min)

df["mult_C"] = df["kelly_f"].apply(kelly_to_mult)
df["size_C"] = (df["mult_C"] * MODAL_BASE).clip(lower=MODAL_BASE, upper=3*MODAL_BASE)
df["pnl_C"]  = df["net_pnl"] * (df["size_C"] / MODAL_BASE)

# ── Print sizing table ────────────────────────────────────────────────────────
banner("Sizing per Confidence Bucket")
print(f"\n  {'Bucket':<14} {'N':>7} {'Kelly_f':>9} {'Flat':>8} {'Kelly_raw':>11} {'Kelly_cap':>11} {'mult_C':>7}")
for bkt in conf_labs:
    sub  = df[df["bucket"] == bkt]
    if len(sub) == 0:
        continue
    kf   = kelly_map.get(bkt, 0.0)
    s_b  = sub["size_B"].mean()
    s_c  = sub["size_C"].mean()
    mult = sub["mult_C"].mean()
    print(f"  {str(bkt):<14} {len(sub):>7,} {kf:>+9.4f} {MODAL_BASE:>8.2f} {s_b:>11.2f} {s_c:>11.2f} {mult:>7.2f}x")

# ── Perbandingan total PnL ────────────────────────────────────────────────────
banner("Hasil Simulasi OOF (Terukur)")
pnl_A = df["pnl_A"].sum()
pnl_B = df["pnl_B"].sum()
pnl_C = df["pnl_C"].sum()

avg_size_A = df["size_A"].mean()
avg_size_B = df["size_B"].mean()
avg_size_C = df["size_C"].mean()

print(f"\n  {'Skenario':<22} {'Total PnL':>12} {'Avg size':>10} {'PnL/trade':>11}")
for label, pnl, avg_s in [
    ("A. Flat $10",        pnl_A, avg_size_A),
    ("B. Kelly raw",       pnl_B, avg_size_B),
    ("C. Kelly cap 1x-3x", pnl_C, avg_size_C),
]:
    print(f"  {label:<22} {pnl:>+12.2f} {avg_s:>10.2f} {pnl/N:>+11.4f}")

print(f"\n  Kelly raw vs flat : {pnl_B-pnl_A:+.2f} ({(pnl_B/pnl_A-1)*100:+.1f}%)")
print(f"  Kelly cap vs flat : {pnl_C-pnl_A:+.2f} ({(pnl_C/pnl_A-1)*100:+.1f}%)")

# ── Produksi-scale estimate ───────────────────────────────────────────────────
banner("Produksi Estimate (scale ke 2,434 trades / 5 bulan)")
print(f"\n  A. Flat $10        : {pnl_A*SCALE:>+10.2f} USD/5mo  ({pnl_A*SCALE/5:>+8.2f}/bulan)")
print(f"  B. Kelly raw       : {pnl_B*SCALE:>+10.2f} USD/5mo  ({pnl_B*SCALE/5:>+8.2f}/bulan)")
print(f"  C. Kelly cap 1x-3x : {pnl_C*SCALE:>+10.2f} USD/5mo  ({pnl_C*SCALE/5:>+8.2f}/bulan)")
print(f"\n  Estimasi gain Kelly cap vs flat : {(pnl_C-pnl_A)*SCALE:>+.2f} USD/5mo")

# ── Breakdown per bucket: kontribusi PnL ─────────────────────────────────────
banner("Kontribusi PnL per Bucket")
print(f"\n  {'Bucket':<14} {'N':>7} {'PnL_A':>10} {'PnL_C':>10} {'Delta':>10} {'pnl/trade_A':>13} {'pnl/trade_C':>13}")
for bkt in conf_labs:
    sub = df[df["bucket"] == bkt]
    if len(sub) == 0:
        continue
    pA = sub["pnl_A"].sum()
    pC = sub["pnl_C"].sum()
    print(f"  {str(bkt):<14} {len(sub):>7,} {pA:>+10.2f} {pC:>+10.2f} {pC-pA:>+10.2f} "
          f"{pA/len(sub):>+13.4f} {pC/len(sub):>+13.4f}")

# ── Risk: max drawdown dan Sharpe per skenario ────────────────────────────────
banner("Risk Profile: Daily Equity Curve")
df_sorted = df.sort_values("timestamp").reset_index(drop=True)

def equity_stats(pnl_col):
    eq = df_sorted[pnl_col].cumsum()
    peak = eq.cummax()
    dd   = (eq - peak)
    max_dd = dd.min()
    # daily equity (approximate — group by date)
    df_sorted["_date"] = pd.to_datetime(df_sorted["timestamp"], utc=True).dt.date
    daily = df_sorted.groupby("_date")[pnl_col].sum()
    sharpe = (daily.mean() / (daily.std() + 1e-9)) * (365 ** 0.5)
    return max_dd, sharpe

dd_A, sh_A = equity_stats("pnl_A")
dd_C, sh_C = equity_stats("pnl_C")

print(f"\n  {'Skenario':<22} {'Max DD':>10} {'Sharpe':>10}")
print(f"  {'A. Flat $10':<22} {dd_A:>+10.2f} {sh_A:>10.2f}")
print(f"  {'C. Kelly cap 1x-3x':<22} {dd_C:>+10.2f} {sh_C:>10.2f}")
print(f"\n  Note: Kelly cap meningkatkan drawdown proporsional dengan sizing.")
print(f"  DD relatif ke modal: pastikan max_DD/capital masih acceptable.")

# ── Rekomendasi threshold ─────────────────────────────────────────────────────
banner("Rekomendasi Implementasi")
print(f"""
  Praktis tanpa ubah inference logic:
    conf < 0.62  -> size = 1.0x ($10)    Kelly={kelly_map.get('0.58-0.62',0):.3f}
    conf 0.62-0.66 -> size = 1.5x ($15) Kelly={kelly_map.get('0.62-0.66',0):.3f}
    conf 0.66-0.70 -> size = 2.0x ($20) Kelly={kelly_map.get('0.66-0.70',0):.3f}
    conf 0.70-0.74 -> size = 2.0x ($20) Kelly={kelly_map.get('0.70-0.74',0):.3f}
    conf 0.74-0.78 -> size = 2.5x ($25) Kelly={kelly_map.get('0.74-0.78',0):.3f}
    conf 0.78+     -> size = 3.0x ($30) Kelly={kelly_map.get('0.78+',0):.3f}

  Implementasi: ubah MODAL_PER_TRADE di inference menjadi fungsi conf,
  bukan konstanta. Tidak butuh retrain apapun.
""")
