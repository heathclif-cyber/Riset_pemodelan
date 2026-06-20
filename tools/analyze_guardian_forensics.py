"""
Guardian forensics: loser profile, dir_ok_12h cross, live vs holdout early-cut.

Outputs:
  reports/experiments/guardian_forensics_holdout.json
  reports/experiments/guardian_forensics_holdout.md
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from tools.analyze_post_signal_path import (
    _load_features, _fwd_ret, _mfe_mae, FWD_BARS, analyze_trades,
)
from tools.live_db_bridge import load_trades

HOLDOUT_CSV = ROOT / "reports/experiments/holdout_ic32_trades_apr_jun26.csv"
CONT_CSV = ROOT / "reports/experiments/holdout_ic32_cont_v1_trades_apr_jun26.csv"
OUT_JSON = ROOT / "reports/experiments/guardian_forensics_holdout.json"
OUT_MD = ROOT / "reports/experiments/guardian_forensics_holdout.md"


def _dir_sign(d: str) -> int:
    return 1 if str(d).upper() == "LONG" else -1


def enrich_trades(trades: pd.DataFrame, label: str) -> pd.DataFrame:
    rows = []
    feat_cache: dict[str, pd.DataFrame] = {}
    trades = trades.copy()
    trades["entry_time"] = pd.to_datetime(trades.get("entry_time", trades.get("Opened")), utc=True)

    for _, tr in trades.iterrows():
        coin = str(tr.get("coin", tr.get("Coin", ""))).replace("/", "")
        if not coin.endswith("USDT"):
            coin = f"{coin}USDT"
        et = tr["entry_time"]
        if coin not in feat_cache:
            feat_cache[coin] = _load_features(coin, et - pd.Timedelta(days=30), et + pd.Timedelta(days=120))
        df = feat_cache[coin]
        if df.empty:
            continue
        if et not in df.index:
            idx_pos = df.index.get_indexer([et], method="nearest")[0]
            if idx_pos < 0:
                continue
            et = df.index[idx_pos]
        else:
            idx_pos = df.index.get_loc(et)

        close = df["close"].values.astype(float)
        high = df["high"].values.astype(float)
        low = df["low"].values.astype(float)
        sign = _dir_sign(tr.get("direction", tr.get("Direction", "LONG")))
        fwd1 = _fwd_ret(close, idx_pos, 1, sign)
        fwd12 = _fwd_ret(close, idx_pos, 12, sign)
        mfe12, mae12 = _mfe_mae(high, low, close, idx_pos, sign, 12)

        is_win = bool(tr.get("is_win", tr.get("net_pnl", tr.get("PnL Net", 0)) > 0))
        hold = int(tr.get("hold_bars", tr.get("Hold Bars", 0)) or 0)
        exit_norm = str(tr.get("exit_norm", tr.get("outcome", tr.get("Exit Reason", "")))).lower()
        pnl = float(tr.get("net_pnl", tr.get("PnL Net", 0)) or 0)

        rows.append({
            "label": label,
            "coin": coin,
            "entry_time": str(et),
            "direction": tr.get("direction", tr.get("Direction")),
            "is_win": is_win,
            "net_pnl": pnl,
            "hold_bars": hold,
            "exit_norm": exit_norm,
            "outcome": tr.get("outcome", tr.get("Exit Reason", "")),
            "dir_ok_12h": bool(fwd12 > 0) if np.isfinite(fwd12) else False,
            "fwd_ret_1": fwd1,
            "fwd_ret_12": fwd12,
            "mfe_12": mfe12,
            "mae_12": mae12,
            "early_guardian": ("guardian" in exit_norm) and (not is_win) and hold <= 3,
            "dump_then_up": (fwd1 < 0 if np.isfinite(fwd1) else False)
            and (fwd12 > 0 if np.isfinite(fwd12) else False),
        })

    return pd.DataFrame(rows)


def _slice_stats(df: pd.DataFrame, mask, name: str) -> dict:
    sub = df[mask]
    if sub.empty:
        return {"name": name, "n": 0}
    gdn = sub[sub.exit_norm.str.contains("guardian", na=False)]
    gdn_los = gdn[~gdn.is_win]
    return {
        "name": name,
        "n": len(sub),
        "wr_pct": round(sub.is_win.mean() * 100, 1),
        "guardian_pct": round(gdn.shape[0] / len(sub) * 100, 1),
        "guardian_losers": len(gdn_los),
        "early_cut_losers": int(gdn_los["hold_bars"].le(3).sum()) if len(gdn_los) else 0,
        "dir_ok_12h_losers": int((~sub.is_win & sub.dir_ok_12h).sum()),
        "fwd1_pos_losers_pct": round(
            (sub.loc[~sub.is_win, "fwd_ret_1"] > 0).mean() * 100, 1
        ) if (~sub.is_win).any() else 0,
        "hold_median": round(float(sub.hold_bars.median()), 1),
        "mom_exit_pct": round(
            sub.exit_norm.str.contains("momentum", na=False).mean() * 100, 1
        ),
    }


def summarize_enriched(rdf: pd.DataFrame, label: str) -> dict:
    losers = ~rdf["is_win"]
    gdn_los = rdf[losers & rdf.exit_norm.str.contains("guardian", na=False)]
    return {
        "label": label,
        "n": len(rdf),
        "slices": [
            _slice_stats(rdf, pd.Series(True, index=rdf.index), "all"),
            _slice_stats(rdf, losers, "losers"),
            _slice_stats(rdf, losers & rdf.dir_ok_12h, "losers_dir_ok_12h"),
            _slice_stats(rdf, rdf["early_guardian"] & losers, "early_guardian_losers"),
            _slice_stats(
                rdf,
                losers & rdf.exit_norm.str.contains("guardian", na=False),
                "guardian_losers",
            ),
            _slice_stats(
                rdf,
                losers & (
                    rdf.exit_norm.eq("sl_hit")
                    | rdf.outcome.astype(str).str.upper().eq("LOSS")
                ),
                "sl_losers",
            ),
        ],
        "exit_breakdown": rdf.groupby("exit_norm").agg(
            n=("net_pnl", "count"),
            wr=("is_win", "mean"),
            mean_pnl=("net_pnl", "mean"),
            mean_hold=("hold_bars", "mean"),
        ).round(4).reset_index().to_dict(orient="records"),
        "guardian_loser_hold_dist": gdn_los.hold_bars.describe().to_dict() if len(gdn_los) else {},
    }


def live_summary() -> dict:
    live = load_trades()
    closed = live[live["is_live"] == 1] if "is_live" in live.columns else live
    gdn = closed[closed["exit_reason"].str.contains("guardian", case=False, na=False)]
    g_los = gdn[gdn["pnl_net"] <= 0]
    g_win = gdn[gdn["pnl_net"] > 0]
    return {
        "n_closed": len(closed),
        "guardian_exits": len(gdn),
        "guardian_wr_pct": round((gdn["pnl_net"] > 0).mean() * 100, 1) if len(gdn) else 0,
        "guardian_losers": len(g_los),
        "early_cut_losers_hold_le3": int((g_los["hold_bars"] <= 3).sum()) if len(g_los) else 0,
        "momentum_exit_n": int(closed["exit_reason"].str.contains("momentum", case=False, na=False).sum()),
        "guardian_exit_only_n": int((closed["exit_reason"] == "guardian_exit").sum()),
        "hold_median_all": round(float(closed["hold_bars"].median()), 1),
        "hold_median_guardian_los": round(float(g_los["hold_bars"].median()), 1) if len(g_los) else 0,
        "hold_median_guardian_win": round(float(g_win["hold_bars"].median()), 1) if len(g_win) else 0,
        "sl_n": int((closed["exit_reason"] == "sl_hit").sum()),
    }


def write_md(payload: dict):
    h = payload.get("holdout_clean_v2", {})
    c = payload.get("holdout_cont_v1", {})
    live = payload.get("live", {})
    lines = [
        "# Guardian Forensics — ic32",
        "",
        "## Holdout clean_v2 (backtest baseline)",
        "",
    ]
    for s in h.get("slices", []):
        if s.get("n", 0) == 0:
            continue
        lines.append(
            f"- **{s['name']}** (n={s['n']}): guardian {s.get('guardian_pct')}% | "
            f"early-cut losers {s.get('early_cut_losers')} | dir_ok_12h losers {s.get('dir_ok_12h_losers')}"
        )
    if c:
        lines.extend(["", "## Holdout continuation_v1 (production Guardian)", ""])
        for s in c.get("slices", []):
            if s.get("n", 0) == 0:
                continue
            lines.append(
                f"- **{s['name']}** (n={s['n']}): guardian {s.get('guardian_pct')}% | "
                f"mom_exit {s.get('mom_exit_pct')}% | hold med {s.get('hold_median')}"
            )
    lines.extend([
        "",
        "## Live VPS",
        "",
        f"- Closed: {live.get('n_closed')} | Guardian exits: {live.get('guardian_exits')} "
        f"(WR {live.get('guardian_wr_pct')}%)",
        f"- Early-cut guardian losers (hold<=3): {live.get('early_cut_losers_hold_le3')}",
        f"- Momentum exits: {live.get('momentum_exit_n')} vs guardian_exit only: {live.get('guardian_exit_only_n')}",
        f"- Hold median: all {live.get('hold_median_all')} | guardian los {live.get('hold_median_guardian_los')}",
    ])
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")


def main():
    payload: dict = {"live": live_summary()}

    if HOLDOUT_CSV.exists():
        hold = pd.read_csv(HOLDOUT_CSV)
        hold["exit_norm"] = hold.get("exit_norm", hold["outcome"]).astype(str).str.lower()
        enriched = enrich_trades(hold, "holdout_clean_v2")
        payload["holdout_clean_v2"] = summarize_enriched(enriched, "holdout_clean_v2")
        payload["holdout_clean_v2_path"] = analyze_trades(hold, "holdout_clean_v2")

    if CONT_CSV.exists():
        cont = pd.read_csv(CONT_CSV)
        cont["exit_norm"] = cont["outcome"].astype(str).str.lower().str.replace(
            "guardian_momentum_exit", "guardian_momentum_exit"
        ).str.replace("guardian_exit", "guardian_exit")
        for oc, norm in {
            "GUARDIAN_EXIT": "guardian_exit",
            "GUARDIAN_MOMENTUM_EXIT": "guardian_momentum_exit",
            "GUARDIAN_MOMENTUM_PARTIAL": "guardian_momentum_partial",
            "LOSS": "sl_hit",
        }.items():
            cont.loc[cont["outcome"] == oc, "exit_norm"] = norm
        enriched_c = enrich_trades(cont, "holdout_cont_v1")
        payload["holdout_cont_v1"] = summarize_enriched(enriched_c, "holdout_cont_v1")

    OUT_JSON.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    write_md(payload)
    print(json.dumps(payload, indent=2, default=str))
    print(f"\nSaved {OUT_JSON}\nSaved {OUT_MD}")


if __name__ == "__main__":
    main()