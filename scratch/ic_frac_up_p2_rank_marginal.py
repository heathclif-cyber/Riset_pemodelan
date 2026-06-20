"""Marginal IC: does CS rank add info beyond absolute p2?"""
import sys
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from config import ALL_COINS, LABEL_DIR, MODEL_DIR

oof = pd.read_parquet(MODEL_DIR / "runs/tb_lgbm_genuine_v2/oof_predictions.parquet")
oof = oof[oof["has_oof"] == True].copy()
oof["direction"] = oof["tb_label"].map({2: 1.0, 0: -1.0, 1: np.nan})
oof = oof[oof["direction"].notna()]

ret_frames = []
for coin in ALL_COINS:
    p = LABEL_DIR / f"{coin}_features_v3.parquet"
    if not p.exists():
        continue
    df = pd.read_parquet(p, columns=["close"]).sort_index()
    df = df[df.index < pd.Timestamp("2026-04-01", tz="UTC")]
    ret_frames.append(
        np.log(df["close"] / df["close"].shift(4).replace(0, np.nan)).rename(coin)
    )
panel = pd.concat(ret_frames, axis=1)
frac_up = (panel > 0).mean(axis=1)
frac_up.name = "frac_up"

df = oof.reset_index().rename(columns={"timestamp": "ts"})
df["p2_rank"] = df.groupby("ts")["p2"].rank(pct=True, method="average")
df = df.merge(
    frac_up.reset_index().rename(columns={"timestamp": "ts"}),
    on="ts",
    how="left",
)
df = df.dropna(subset=["p2", "p2_rank", "frac_up", "direction"])

x = df["p2"].values
y = df["p2_rank"].values
coef = np.polyfit(x, y, 1)
p2_rank_resid = y - (coef[0] * x + coef[1])

coef2 = np.polyfit(df["p2"].values, df["frac_up"].values, 1)
frac_resid = df["frac_up"].values - (coef2[0] * df["p2"].values + coef2[1])

ic_p2, _ = spearmanr(df["p2"], df["direction"])
ic_full, _ = spearmanr(df["p2_rank"], df["direction"])
ic_resid, _ = spearmanr(p2_rank_resid, df["direction"])
ic_frac, _ = spearmanr(df["frac_up"], df["direction"])
ic_frac_resid, _ = spearmanr(frac_resid, df["direction"])

print("=== MARGINAL IC (pooled OOF directional bars) ===")
print(f"N bars: {len(df):,}")
print(f"p2 standalone IC           : {ic_p2:+.4f}")
print(f"p2_rank standalone IC      : {ic_full:+.4f}")
print(f"p2_rank residual IC (vs p2): {ic_resid:+.4f}  <- CS rank beyond p2")
print(f"frac_up standalone IC      : {ic_frac:+.4f}")
print(f"frac_up residual IC (vs p2): {ic_frac_resid:+.4f}  <- breadth beyond p2")

feat36 = json.load(open(MODEL_DIR / "feature_cols_v2.json"))

rows = []
for coin in ALL_COINS:
    p = LABEL_DIR / f"{coin}_features_v3.parquet"
    if not p.exists():
        continue
    f = pd.read_parquet(p, columns=feat36).sort_index()
    cd = df[df["coin"] == coin].set_index("ts")
    m = cd.join(f, how="inner")
    if len(m) < 500:
        continue
    for feat in feat36 + ["frac_up", "p2_rank"]:
        if feat not in m.columns:
            continue
        sub = m[[feat, "direction"]].dropna()
        if len(sub) < 200:
            continue
        ic, _ = spearmanr(sub[feat], sub["direction"])
        rows.append({"coin": coin, "feat": feat, "ic": ic})

agg = pd.DataFrame(rows).groupby("feat")["ic"].agg(["mean", "std", "count"])
agg["t_stat"] = agg["mean"] / (agg["std"] / agg["count"].pow(0.5))
agg = agg.sort_values("mean", key=abs, ascending=False)
print()
print("=== Per-coin IC mean: top-10 existing + CS rank ===")
print(agg.head(12).round(4).to_string())
for f in ["frac_up", "p2_rank"]:
    if f in agg.index:
        rank = list(agg.index).index(f) + 1
        row = agg.loc[f]
        print(f"  {f}: rank #{rank}/{len(agg)}, IC={row['mean']:+.4f}, t={row['t_stat']:.1f}")