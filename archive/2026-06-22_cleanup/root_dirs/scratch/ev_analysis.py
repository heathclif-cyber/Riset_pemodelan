"""
scratch/ev_analysis.py
Expected Value per trade — bukan WR.
EV = p(win) * avg_win_pnl + p(loss) * avg_loss_pnl

Pertanyaan: apakah gradient WR di confidence dan hour juga
konsisten di EV? Kalau conf 0.78+ WR=49% tapi avg_loss 2x avg_win,
edge-nya masih negatif meski WR tinggi.

Input: data/meta_labels/ic32_oof_trades.parquet
"""
import sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

OOF_PATH = ROOT / "data" / "meta_labels" / "ic32_oof_trades.parquet"
df = pd.read_parquet(OOF_PATH)
df["conf"]    = np.where(df["direction"] == 1, df["conf_long"], df["conf_short"])
df["hour"]    = pd.to_datetime(df["timestamp"], utc=True).dt.hour

N  = len(df)
WR = df["win"].mean() * 100
EV = df["net_pnl"].mean()

def ev_table(df, col, bins=None, labels=None):
    if bins is not None:
        df = df.copy()
        df["_bin"] = pd.cut(df[col], bins=bins, labels=labels)
        col = "_bin"
    rows = []
    for val, grp in df.groupby(col, observed=True):
        wins   = grp[grp["win"] == 1]["net_pnl"]
        losses = grp[grp["win"] == 0]["net_pnl"]
        n      = len(grp)
        wr     = grp["win"].mean()
        ev     = grp["net_pnl"].mean()
        avg_w  = wins.mean()  if len(wins)  > 0 else 0.0
        avg_l  = losses.mean() if len(losses) > 0 else 0.0
        payoff = abs(avg_w / avg_l) if avg_l != 0 else np.inf
        kelly  = wr - (1 - wr) / payoff if payoff > 0 else 0.0
        rows.append({"group": val, "n": n, "wr_pct": round(wr*100, 1),
                     "ev": round(ev, 4), "avg_win": round(avg_w, 4),
                     "avg_loss": round(avg_l, 4),
                     "payoff": round(payoff, 3), "kelly_f": round(kelly, 4)})
    return pd.DataFrame(rows).sort_values("group")


def banner(title, w=65):
    print(f"\n{'='*w}\n  {title}\n{'='*w}")


banner(f"EV ANALYSIS — OOF Trades (N={N:,})")
print(f"\n  Base: WR={WR:.1f}%  EV={EV:+.4f} USD/trade")
print(f"  avg_win  = {df[df['win']==1]['net_pnl'].mean():+.4f}")
print(f"  avg_loss = {df[df['win']==0]['net_pnl'].mean():+.4f}")
payoff_base = abs(df[df['win']==1]['net_pnl'].mean() / df[df['win']==0]['net_pnl'].mean())
kelly_base  = WR/100 - (1 - WR/100) / payoff_base
print(f"  payoff ratio (|win/loss|) = {payoff_base:.3f}")
print(f"  Kelly fraction = {kelly_base:+.4f}  "
      f"({'POSITIVE EDGE' if kelly_base > 0 else 'NEGATIVE EDGE'})")

# ── 1. EV by confidence bucket ────────────────────────────────────────────────
banner("1. EV by LGBM Confidence Bucket")
conf_bins = [0.58, 0.62, 0.66, 0.70, 0.74, 0.78, 1.00]
conf_labs = ["0.58-0.62","0.62-0.66","0.66-0.70","0.70-0.74","0.74-0.78","0.78+"]
t = ev_table(df, "conf", bins=conf_bins, labels=conf_labs)
print(f"\n  {'Conf range':<14} {'N':>7} {'WR%':>7} {'EV':>9} "
      f"{'avg_win':>9} {'avg_loss':>10} {'payoff':>8} {'kelly':>8}")
for _, r in t.iterrows():
    edge_flag = " <-- NEG" if r["kelly_f"] < 0 else (" <-- BEST" if r["kelly_f"] == t["kelly_f"].max() else "")
    print(f"  {str(r['group']):<14} {r['n']:>7,} {r['wr_pct']:>6.1f}% "
          f"{r['ev']:>+9.4f} {r['avg_win']:>+9.4f} {r['avg_loss']:>+10.4f} "
          f"{r['payoff']:>8.3f} {r['kelly_f']:>+8.4f}{edge_flag}")

# ── 2. EV by hour ─────────────────────────────────────────────────────────────
banner("2. EV by Hour of Day (UTC)")
t_h = ev_table(df, "hour")
print(f"\n  {'Hour':>6} {'N':>7} {'WR%':>7} {'EV':>9} "
      f"{'avg_win':>9} {'avg_loss':>10} {'payoff':>8} {'kelly':>8}")
best_ev_hour  = t_h.loc[t_h["ev"].idxmax(), "group"]
worst_ev_hour = t_h.loc[t_h["ev"].idxmin(), "group"]
for _, r in t_h.iterrows():
    flag = " <-- BEST"  if r["group"] == best_ev_hour  else (
           " <-- WORST" if r["group"] == worst_ev_hour else "")
    print(f"  {r['group']:>6} {r['n']:>7,} {r['wr_pct']:>6.1f}% "
          f"{r['ev']:>+9.4f} {r['avg_win']:>+9.4f} {r['avg_loss']:>+10.4f} "
          f"{r['payoff']:>8.3f} {r['kelly_f']:>+8.4f}{flag}")

# Hours with negative EV
neg_ev_hours = t_h[t_h["ev"] < 0]["group"].tolist()
pos_ev_hours = t_h[t_h["ev"] > 0.02]["group"].tolist()
print(f"\n  Negative EV hours: {sorted(neg_ev_hours)}")
print(f"  Positive EV hours (>0.02): {sorted(pos_ev_hours)}")

# ── 3. EV by direction ────────────────────────────────────────────────────────
banner("3. EV by Direction")
for dir_, label in [(1, "LONG"), (-1, "SHORT")]:
    sub = df[df["direction"] == dir_]
    wins   = sub[sub["win"] == 1]["net_pnl"]
    losses = sub[sub["win"] == 0]["net_pnl"]
    wr     = sub["win"].mean() * 100
    ev     = sub["net_pnl"].mean()
    avg_w  = wins.mean()
    avg_l  = losses.mean()
    payoff = abs(avg_w / avg_l) if avg_l != 0 else 0
    kelly  = wr/100 - (1 - wr/100) / payoff if payoff > 0 else 0
    print(f"\n  {label} (n={len(sub):,}):")
    print(f"    WR={wr:.1f}%  EV={ev:+.4f}  avg_win={avg_w:+.4f}  "
          f"avg_loss={avg_l:+.4f}  payoff={payoff:.3f}  Kelly={kelly:+.4f}")

# ── 4. EV by SHORT confidence sub-bucket (focus area) ────────────────────────
banner("4. SHORT trades: EV by confidence sub-bucket (0.59-0.70)")
short_df  = df[df["direction"] == -1].copy()
sb_bins   = [0.58, 0.60, 0.62, 0.64, 0.66, 0.68, 0.70, 0.75, 1.00]
sb_labs   = ["0.58-0.60","0.60-0.62","0.62-0.64","0.64-0.66",
             "0.66-0.68","0.68-0.70","0.70-0.75","0.75+"]
t_s = ev_table(short_df, "conf", bins=sb_bins, labels=sb_labs)
print(f"\n  {'Conf range':<14} {'N':>7} {'WR%':>7} {'EV':>9} "
      f"{'avg_win':>9} {'avg_loss':>10} {'payoff':>8} {'kelly':>8}")
for _, r in t_s.iterrows():
    flag = " <-- NEG EV" if r["ev"] < 0 else (" <-- cut here?" if r["kelly_f"] < 0.01 else "")
    print(f"  {str(r['group']):<14} {r['n']:>7,} {r['wr_pct']:>6.1f}% "
          f"{r['ev']:>+9.4f} {r['avg_win']:>+9.4f} {r['avg_loss']:>+10.4f} "
          f"{r['payoff']:>8.3f} {r['kelly_f']:>+8.4f}{flag}")

# ── 5. Summary: are bad-WR hours also bad-EV? ─────────────────────────────────
banner("5. WR vs EV alignment check (hours 3, 7, 11, 15, 19)")
bad_hours = [3, 7, 11, 15, 19]
good_hours = [0, 8, 12, 16, 20]
print(f"\n  Predicted bad hours (pre-4H close): {bad_hours}")
bad_sub  = df[df["hour"].isin(bad_hours)]
good_sub = df[df["hour"].isin(good_hours)]
print(f"  Bad hours  : n={len(bad_sub):,}  WR={bad_sub['win'].mean()*100:.1f}%  "
      f"EV={bad_sub['net_pnl'].mean():+.4f}")
print(f"  Good hours : n={len(good_sub):,}  WR={good_sub['win'].mean()*100:.1f}%  "
      f"EV={good_sub['net_pnl'].mean():+.4f}")

# Are bad-WR hours also bad-EV?
bad_ev  = bad_sub["net_pnl"].mean()
good_ev = good_sub["net_pnl"].mean()
if bad_ev < 0 and good_ev > 0:
    print(f"  CONFIRMED: bad hours have NEGATIVE EV. Pre-filter is valid.")
elif bad_ev < good_ev * 0.5:
    print(f"  PARTIAL: bad hours have significantly lower EV but still positive.")
    print(f"  Pre-filter still valid economically.")
else:
    print(f"  WR and EV diverge: bad-WR hours still positive EV.")
    print(f"  Pre-filter based on WR alone would be WRONG.")

# ── 6. Quantify: what's the EV gain from cutting low-conf SHORT? ──────────────
banner("6. Dollar impact of threshold changes (extrapolated to production)")
print(f"""
  OOF vs production scale factor:
    OOF: 16,093 trades over ~5 years (training period)
    Production: 487 trades/month = 2,434/5 months

  Adjusting to production scale (2,434 trades, 5 months):
""")
PROD_SCALE = 2434 / N  # scale OOF to production trade count

def estimate_impact(df_subset, label, df_base=df):
    n_before = len(df_base)
    n_after  = len(df_subset)
    ev_before = df_base["net_pnl"].mean()
    ev_after  = df_subset["net_pnl"].mean()
    n_removed  = n_before - n_after
    # In production: removing trades reduces count, potentially improves EV
    prod_trades_removed = n_removed * PROD_SCALE
    prod_ev_gain = (ev_after - ev_before) * (n_after * PROD_SCALE)
    print(f"  {label}:")
    print(f"    OOF trades removed: {n_removed:,} ({n_removed/n_before*100:.1f}%)")
    print(f"    EV change: {ev_before:+.4f} -> {ev_after:+.4f} (+{ev_after-ev_before:+.4f})")
    print(f"    Prod estimate: remove ~{prod_trades_removed:.0f} trades, "
          f"EV gain ~{prod_ev_gain:+.2f} USD/5mo")

# Remove bad hours
estimate_impact(df[~df["hour"].isin(bad_hours)], "Filter bad hours (3,7,11,15,19)")
# Tighten SHORT threshold
estimate_impact(
    df[(df["direction"] == 1) | ((df["direction"] == -1) & (df["conf"] >= 0.64))],
    "SHORT threshold 0.59->0.64"
)
# Both combined
estimate_impact(
    df[~df["hour"].isin(bad_hours) &
       ((df["direction"] == 1) | ((df["direction"] == -1) & (df["conf"] >= 0.64)))],
    "Both combined"
)
