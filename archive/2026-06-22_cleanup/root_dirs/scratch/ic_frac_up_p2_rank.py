"""IC walk-forward test for cross-sectional features frac_up and p2_rank."""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from config import ALL_COINS, LABEL_DIR, MODEL_DIR

MARKET_RET_BARS = 4
TIME_WINDOWS = [
    ("2020", pd.Timestamp("2020-01-01", tz="UTC"), pd.Timestamp("2021-01-01", tz="UTC")),
    ("2021", pd.Timestamp("2021-01-01", tz="UTC"), pd.Timestamp("2022-01-01", tz="UTC")),
    ("2022", pd.Timestamp("2022-01-01", tz="UTC"), pd.Timestamp("2023-01-01", tz="UTC")),
    ("2023", pd.Timestamp("2023-01-01", tz="UTC"), pd.Timestamp("2024-01-01", tz="UTC")),
    ("2024", pd.Timestamp("2024-01-01", tz="UTC"), pd.Timestamp("2025-01-01", tz="UTC")),
    ("2025", pd.Timestamp("2025-01-01", tz="UTC"), pd.Timestamp("2026-04-01", tz="UTC")),
]

oof = pd.read_parquet(MODEL_DIR / "runs/tb_lgbm_genuine_v2/oof_predictions.parquet")
oof = oof[oof["has_oof"] == True][["coin", "p0", "p2", "tb_label"]].copy()
oof["direction"] = oof["tb_label"].map({2: 1.0, 0: -1.0, 1: np.nan})

print("=" * 70)
print("  IC TEST: frac_up + p2_rank (cross-sectional features)")
print(f"  OOF bars: {len(oof):,}  |  Coins: {len(ALL_COINS)}")
print("=" * 70)

ret_frames = []
for coin in ALL_COINS:
    p = LABEL_DIR / f"{coin}_features_v3.parquet"
    if not p.exists():
        continue
    df = pd.read_parquet(p, columns=["close"]).sort_index()
    df = df[df.index < pd.Timestamp("2026-04-01", tz="UTC")]
    ret4 = np.log(df["close"] / df["close"].shift(MARKET_RET_BARS).replace(0, np.nan))
    ret_frames.append(ret4.rename(coin))

panel_ret = pd.concat(ret_frames, axis=1)
frac_up = (panel_ret > 0).mean(axis=1).rename("frac_up")
mkt_ret4 = panel_ret.mean(axis=1).rename("mkt_ret4")

oof_idx = oof.reset_index().rename(columns={"timestamp": "ts"})
oof_idx["p2_rank"] = oof_idx.groupby("ts")["p2"].rank(pct=True, method="average")

ret4_long = panel_ret.stack(future_stack=True).rename("ret4")
ret4_long.index.names = ["ts", "coin"]
oof_idx = oof_idx.set_index(["ts", "coin"])
oof_idx = oof_idx.join(ret4_long, how="left")
oof_idx = oof_idx.reset_index()
oof_idx["cs_ret4_rank"] = oof_idx.groupby("ts")["ret4"].rank(pct=True, method="average")
oof_idx = oof_idx.merge(
    frac_up.reset_index().rename(columns={"timestamp": "ts"}),
    on="ts",
    how="left",
)
oof_idx = oof_idx.merge(
    mkt_ret4.reset_index().rename(columns={"timestamp": "ts"}),
    on="ts",
    how="left",
)

records = []
for coin in ALL_COINS:
    merged = oof_idx[oof_idx["coin"] == coin].set_index("ts")
    merged = merged[merged["direction"].notna()]
    if len(merged) < 200:
        continue

    candidates = ["frac_up", "mkt_ret4", "p2_rank", "cs_ret4_rank", "p2"]
    for wname, wstart, wend in TIME_WINDOWS:
        window = merged[(merged.index >= wstart) & (merged.index < wend)]
        active = window.dropna(subset=["direction"])
        if len(active) < 100:
            continue
        for feat in candidates:
            if feat not in active.columns:
                continue
            sub = active[["direction", feat]].dropna()
            if len(sub) < 50:
                continue
            ic, pval = spearmanr(sub[feat], sub["direction"])
            if np.isnan(ic):
                continue
            records.append({
                "feature": feat,
                "coin": coin,
                "window": wname,
                "IC": ic,
                "pval": pval,
                "n": len(sub),
            })

df_rec = pd.DataFrame(records)
agg = (
    df_rec.groupby("feature")
    .agg(
        mean_IC=("IC", "mean"),
        std_IC=("IC", "std"),
        median_IC=("IC", "median"),
        n_obs=("IC", "count"),
        hit_rate=("IC", lambda x: (x > 0).mean()),
        sig_pct=("pval", lambda x: (x < 0.05).mean()),
    )
    .assign(
        t_stat=lambda d: d["mean_IC"] / (d["std_IC"] / d["n_obs"].pow(0.5)),
        ic_ir=lambda d: d["mean_IC"].abs() / d["std_IC"],
    )
    .sort_values("mean_IC", key=abs, ascending=False)
    .round(4)
)

sample = oof_idx.dropna(subset=["frac_up", "p2_rank", "cs_ret4_rank", "p2", "ret4"])
sample = sample.sample(min(100000, len(sample)), random_state=42)
corr = sample[["frac_up", "mkt_ret4", "p2_rank", "cs_ret4_rank", "p2", "ret4"]].corr(
    method="spearman"
).round(3)

print()
print("  HASIL WALK-FORWARD IC (vs tb_label direction)")
print("  Gate: |IC|>0.01 t>2 weak | |IC|>0.02 t>2 good | |IC|>0.05 t>3 strong")
print("  " + "-" * 72)
hdr = f"  {'Feature':<16} {'mean_IC':>8} {'t_stat':>8} {'IC_IR':>7} {'hit%':>6} {'sig%':>6}  gate"
print(hdr)
for feat, row in agg.iterrows():
    if abs(row["mean_IC"]) > 0.05 and abs(row["t_stat"]) > 3:
        qual = "STRONG"
    elif abs(row["mean_IC"]) > 0.02 and abs(row["t_stat"]) > 2:
        qual = "good"
    elif abs(row["mean_IC"]) > 0.01 and abs(row["t_stat"]) > 2:
        qual = "weak"
    else:
        qual = "FAIL"
    print(
        f"  {feat:<16} {row['mean_IC']:>+8.4f} {row['t_stat']:>8.2f} "
        f"{row['ic_ir']:>7.2f} {row['hit_rate']*100:>5.0f}% {row['sig_pct']*100:>5.0f}%  {qual}"
    )

print()
print("  Spearman corr antar fitur CS (sample 100k rows):")
print(corr.to_string())

print()
print("  Per-window mean IC (pooled across coins):")
for feat in ["frac_up", "p2_rank", "cs_ret4_rank", "p2", "mkt_ret4"]:
    sub = df_rec[df_rec["feature"] == feat]
    if sub.empty:
        continue
    pw = sub.groupby("window")["IC"].mean().round(4)
    print(f"  {feat}: {dict(pw)}")

import json

feat36 = json.load(open(MODEL_DIR / "feature_cols_v2.json"))
print()
print(f"  Current LGBM features: {len(feat36)}")
print(f"  frac_up in current set: {'frac_up' in feat36}")
print(f"  p2_rank in current set: {'p2_rank' in feat36}")