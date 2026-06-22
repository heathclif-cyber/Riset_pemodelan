"""Export holdout test results ic32_regime_v2 ke Excel."""
import pandas as pd
import json
import numpy as np
from pathlib import Path

RUN = Path("d:/Apps-Dev/Riset_pemodelan/models/runs/ic32_regime_v2")
OUT = Path("D:/ic32regimev2.xlsx")

df_all = pd.read_csv(RUN / "holdout_v1_vs_v2_trades.csv")
with open(RUN / "cv_results.json") as f:
    cv = json.load(f)


def scorecard(sub, label=""):
    n = len(sub)
    if n == 0:
        return {}
    wins   = sub[sub["pnl"] > 0]
    losses = sub[sub["pnl"] < 0]
    gp = wins["pnl"].sum()
    gl = abs(losses["pnl"].sum())
    pf = round(gp / gl, 3) if gl > 0 else None
    wr = round(len(wins) / n * 100, 2)
    avg_w = round(gp / len(wins), 4) if len(wins) > 0 else 0
    avg_l = round(-gl / len(losses), 4) if len(losses) > 0 else 0
    net   = round(sub["pnl"].sum(), 4)
    exp   = round((wr / 100) * avg_w - (1 - wr / 100) * abs(avg_l), 4)
    avg_bars = round(sub["bars"].mean(), 1)
    streak = max_streak = 0
    for p in sub.sort_values("ts")["pnl"]:
        if p < 0:
            streak += 1
            max_streak = max(max_streak, streak)
        else:
            streak = 0
    return {
        "Model / Coin / Dir": label,
        "Trades":              n,
        "Wins":                len(wins),
        "Losses":              len(losses),
        "Win Rate %":          wr,
        "Net PnL (ATR)":       net,
        "Gross Profit (ATR)":  round(gp, 4),
        "Gross Loss (ATR)":    round(gl, 4),
        "Profit Factor":       pf,
        "Avg Win (ATR)":       avg_w,
        "Avg Loss (ATR)":      avg_l,
        "Expectancy (ATR)":    exp,
        "Avg Hold (bars)":     avg_bars,
        "Max Consec Losses":   max_streak,
    }


# Sheet 1: Trades v2
v2 = df_all[df_all["model"] == "v2"].copy()
v2 = v2.sort_values("ts").reset_index(drop=True)
v2.index = v2.index + 1
v2.index.name = "No"
v2_out = v2[["coin", "ts", "dir", "entry", "exit", "outcome", "pnl", "bars", "rr"]].rename(columns={
    "coin":    "Coin",
    "ts":      "Open Time (UTC)",
    "dir":     "Direction",
    "entry":   "Entry Price",
    "exit":    "Exit Price",
    "outcome": "Exit Type",
    "pnl":     "PnL (ATR)",
    "bars":    "Hold Bars",
    "rr":      "R:R",
})
v2_out["Win/Loss"] = v2_out["PnL (ATR)"].apply(lambda x: "WIN" if x > 0 else "LOSS")
v2_out["Cum PnL (ATR)"] = v2_out["PnL (ATR)"].cumsum().round(4)

# Sheet 2: Model comparison scorecard
sc_rows = []
for m in ["v1", "v2", "v2+lstm1", "v2+lstm2"]:
    sub = df_all[df_all["model"] == m]
    sc_rows.append(scorecard(sub, m))
df_sc = pd.DataFrame(sc_rows)

# Sheet 3: Per-coin scorecard v2
coin_rows = [scorecard(v2[v2["coin"] == c], c) for c in sorted(v2["coin"].unique())]
df_coin = pd.DataFrame(coin_rows).sort_values("Net PnL (ATR)", ascending=False)

# Sheet 4: Exit type breakdown v2
exit_rows = []
for et in sorted(v2["outcome"].unique()):
    sub = v2[v2["outcome"] == et]
    wins = (sub["pnl"] > 0).sum()
    exit_rows.append({
        "Exit Type":       et,
        "Count":           len(sub),
        "Wins":            wins,
        "Losses":          len(sub) - wins,
        "Win Rate %":      round(wins / len(sub) * 100, 1),
        "Avg PnL (ATR)":   round(sub["pnl"].mean(), 4),
        "Total PnL (ATR)": round(sub["pnl"].sum(), 4),
        "Avg Hold (bars)": round(sub["bars"].mean(), 1),
    })
df_exit = pd.DataFrame(exit_rows).sort_values("Total PnL (ATR)", ascending=False)

# Sheet 5: Direction breakdown v2
dir_rows = [scorecard(v2[v2["dir"] == d], d) for d in ["LONG", "SHORT"]]
df_dir = pd.DataFrame(dir_rows)

# Sheet 6: CV folds
cv_rows = []
for fold in cv["folds"]:
    cv_rows.append({
        "Fold":          fold["fold"],
        "Train Samples": fold["n_train"],
        "Val Samples":   fold["n_val"],
        "Best Iter":     fold["best_iter"],
        "F1 Macro":      round(fold["f1_macro"], 4),
        "F1 SHORT":      round(fold["f1_SHORT"], 4),
        "F1 FLAT":       round(fold["f1_FLAT"], 4),
        "F1 LONG":       round(fold["f1_LONG"], 4),
    })
df_cv = pd.DataFrame(cv_rows)

# Sheet 7: OOF Summary
oof_rows = [
    {"Metric": "Run Name",         "Value": cv["run_name"]},
    {"Metric": "Train Cutoff",     "Value": cv["train_cutoff"]},
    {"Metric": "N Features",       "Value": cv["n_features"]},
    {"Metric": "N Folds",          "Value": cv["n_folds"]},
    {"Metric": "Purge Bars",       "Value": cv["purge_bars"]},
    {"Metric": "TP (ATR mult)",    "Value": cv["tp_atr"]},
    {"Metric": "SL (ATR mult)",    "Value": cv["sl_atr"]},
    {"Metric": "Max Hold (bars)",  "Value": cv["max_hold"]},
    {"Metric": "Mean F1 Macro",    "Value": round(cv["mean_f1_macro"], 4)},
    {"Metric": "Std F1 Macro",     "Value": round(cv["std_f1_macro"], 4)},
    {"Metric": "Avg Iterations",   "Value": cv["avg_iterations"]},
    {"Metric": "Best Thr LONG",    "Value": cv["best_thr_long"]},
    {"Metric": "Best Thr SHORT",   "Value": cv["best_thr_short"]},
    {"Metric": "OOF PnL (ATR)",    "Value": cv["oof_pnl"]},
    {"Metric": "OOF Win Rate %",   "Value": round(cv["oof_wr"] * 100, 2)},
    {"Metric": "OOF Trades",       "Value": cv["oof_trades"]},
    {"Metric": "Holdout Period",   "Value": "Apr 2026 - Jun 2026"},
    {"Metric": "Coins",            "Value": "21 (ic32 universe)"},
]
df_oof = pd.DataFrame(oof_rows)

# Sheet 8: Features
df_feat = pd.DataFrame({"No": range(1, len(cv["features"]) + 1), "Feature": cv["features"]})

# Write Excel
with pd.ExcelWriter(str(OUT), engine="openpyxl") as writer:
    v2_out.to_excel(writer, sheet_name="Trades_v2", index=True)
    df_sc.to_excel(writer, sheet_name="Scorecard_Model", index=False)
    df_coin.to_excel(writer, sheet_name="Per_Coin_v2", index=False)
    df_exit.to_excel(writer, sheet_name="Exit_Types_v2", index=False)
    df_dir.to_excel(writer, sheet_name="Direction_v2", index=False)
    df_cv.to_excel(writer, sheet_name="CV_Folds", index=False)
    df_oof.to_excel(writer, sheet_name="OOF_Summary", index=False)
    df_feat.to_excel(writer, sheet_name="Features", index=False)

print(f"Done: {OUT}")
print(f"  Trades_v2:       {len(v2_out)} trades")
print(f"  Scorecard_Model: {len(df_sc)} models")
print(f"  Per_Coin_v2:     {len(df_coin)} coins")
print(f"  Exit_Types_v2:   {len(df_exit)} exit types")
print(f"  CV_Folds:        {len(df_cv)} folds")
