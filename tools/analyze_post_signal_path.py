"""
Analisis empiris: pergerakan harga setelah sinyal entry ic32.

Data: holdout ic32 trades (Apr-Jun 2026) + optional live VPS trades.
Metrik:
  - SL touch bar +1 / +2 / +3
  - Forward return directional bar +1..+12
  - "Arah benar tapi kalah": loss/SL tapi fwd_ret_N > 0
  - Cluster entry berturut (bar i dan i+1 same coin+direction)
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from config import LABEL_DIR, TRAIN_CUTOFF_DATE, HOLDOUT_DIR

HOLDOUT_LABEL_DIR = HOLDOUT_DIR / "labeled"

HOLDOUT_TRADES = ROOT / "reports/experiments/holdout_ic32_trades_apr_jun26.csv"
OUT_JSON = ROOT / "reports/experiments/post_signal_path_ic32_holdout.json"
OUT_MD = ROOT / "reports/experiments/post_signal_path_ic32_holdout.md"

FWD_BARS = (1, 2, 3, 6, 12, 24)


def _load_features(coin: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Load H1 features; merge training + holdout labeled parquets when both overlap range."""
    parts = []
    for p in (LABEL_DIR / f"{coin}_features_v3.parquet", HOLDOUT_LABEL_DIR / f"{coin}_features_v3.parquet"):
        if not p.exists():
            continue
        df = pd.read_parquet(p).sort_index()
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        sub = df.loc[(df.index >= start) & (df.index <= end)]
        if not sub.empty:
            parts.append(sub)
    if not parts:
        return pd.DataFrame()
    out = pd.concat(parts)
    out = out[~out.index.duplicated(keep="last")].sort_index()
    return out


def _dir_sign(d: str) -> int:
    return 1 if str(d).upper() == "LONG" else -1


def _fwd_ret(close: np.ndarray, i: int, h: int, sign: int) -> float:
    if i + h >= len(close):
        return np.nan
    return sign * (close[i + h] / close[i] - 1.0)


def _sl_touch_bar(high, low, sl: float, direction: str, start: int, max_bars: int = 24) -> int | None:
    """First bar offset (1-based) where SL touched, or None."""
    for off in range(1, max_bars + 1):
        j = start + off
        if j >= len(high):
            break
        if direction == "LONG" and low[j] <= sl:
            return off
        if direction == "SHORT" and high[j] >= sl:
            return off
    return None


def _mfe_mae(high, low, close, i: int, sign: int, horizon: int) -> tuple[float, float]:
    """Max favorable / adverse excursion % over horizon bars after entry bar i."""
    end = min(i + horizon, len(close) - 1)
    if end <= i:
        return np.nan, np.nan
    entry = close[i]
    seg_h = high[i + 1 : end + 1]
    seg_l = low[i + 1 : end + 1]
    if sign == 1:
        mfe = (seg_h.max() / entry - 1.0) if len(seg_h) else 0.0
        mae = (entry - seg_l.min()) / entry if len(seg_l) else 0.0
    else:
        mfe = (entry - seg_l.min()) / entry if len(seg_l) else 0.0
        mae = (seg_h.max() / entry - 1.0) if len(seg_h) else 0.0
    return float(mfe), float(mae)


def analyze_trades(trades: pd.DataFrame, label: str) -> dict:
    trades = trades.copy()
    trades["entry_time"] = pd.to_datetime(trades["entry_time"], utc=True)
    trades = trades.sort_values(["coin", "entry_time"])

    rows = []
    feat_cache: dict[str, pd.DataFrame] = {}

    for _, tr in trades.iterrows():
        coin = tr["coin"]
        et = tr["entry_time"]
        if coin not in feat_cache:
            feat_cache[coin] = _load_features(coin, et - pd.Timedelta(days=30), et + pd.Timedelta(days=120))
        df = feat_cache[coin]
        if df.empty or et not in df.index:
            # nearest index
            if df.empty:
                continue
            idx_pos = df.index.get_indexer([et], method="nearest")[0]
            if idx_pos < 0:
                continue
            et = df.index[idx_pos]
        else:
            idx_pos = df.index.get_loc(et)

        close = df["close"].values.astype(float)
        high = df["high"].values.astype(float)
        low = df["low"].values.astype(float)
        sign = _dir_sign(tr["direction"])
        sl = float(tr["sl"]) if pd.notna(tr.get("sl")) else np.nan
        entry_p = float(tr["entry_price"])

        vol_spike = float(df["vol_spike_zscore"].iloc[idx_pos]) if "vol_spike_zscore" in df.columns else np.nan
        atr_pct = float(df["atr_percent_h1"].iloc[idx_pos]) if "atr_percent_h1" in df.columns else np.nan

        sl_off = _sl_touch_bar(high, low, sl, tr["direction"], idx_pos) if np.isfinite(sl) else None
        sl_dist_pct = abs(entry_p - sl) / entry_p * 100 if np.isfinite(sl) and entry_p > 0 else np.nan

        rec = {
            "coin": coin,
            "entry_time": str(et),
            "direction": tr["direction"],
            "outcome": tr.get("outcome", ""),
            "exit_norm": tr.get("exit_norm", ""),
            "is_win": bool(tr.get("is_win", tr.get("net_pnl", 0) > 0)),
            "hold_bars": int(tr.get("hold_bars", 0)),
            "net_pnl": float(tr.get("net_pnl", 0)),
            "confidence": float(tr.get("confidence", np.nan)),
            "vol_ratio_20": float(tr.get("vol_ratio_20", np.nan)),
            "vol_spike": vol_spike,
            "atr_pct": atr_pct,
            "sl_touch_bar": sl_off,
            "sl_dist_pct": sl_dist_pct,
            "h4_trend": float(tr.get("h4_trend", np.nan)),
            "trend_align": tr.get("trend_align", ""),
        }

        for h in FWD_BARS:
            rec[f"fwd_ret_{h}"] = _fwd_ret(close, idx_pos, h, sign)

        mfe12, mae12 = _mfe_mae(high, low, close, idx_pos, sign, 12)
        mfe24, mae24 = _mfe_mae(high, low, close, idx_pos, sign, 24)
        rec["mfe_12"] = mfe12
        rec["mae_12"] = mae12
        rec["mfe_24"] = mfe24
        rec["mae_24"] = mae24

        # Arah benar: fwd_ret positif pada horizon
        rec["dir_ok_6"] = rec["fwd_ret_6"] > 0 if np.isfinite(rec["fwd_ret_6"]) else False
        rec["dir_ok_12"] = rec["fwd_ret_12"] > 0 if np.isfinite(rec["fwd_ret_12"]) else False
        rec["dir_ok_24"] = _fwd_ret(close, idx_pos, 24, sign) > 0 if idx_pos + 24 < len(close) else False

        # Kalah tapi arah benar nanti
        rec["loss_dir_ok_12"] = (not rec["is_win"]) and rec["dir_ok_12"]
        rec["sl_bar1"] = sl_off == 1
        rec["sl_bar2"] = sl_off == 2

        rows.append(rec)

    if not rows:
        return {"label": label, "n": 0}

    rdf = pd.DataFrame(rows)

    # consecutive same-direction entries within 1 bar on same coin
    rdf = rdf.sort_values(["coin", "entry_time"])
    rdf["prev_entry"] = rdf.groupby("coin")["entry_time"].shift(1)
    rdf["entry_gap_h"] = (
        pd.to_datetime(rdf["entry_time"], utc=True) - pd.to_datetime(rdf["prev_entry"], utc=True)
    ).dt.total_seconds() / 3600
    rdf["repeat_1h"] = rdf["entry_gap_h"] == 1.0

    def agg_slice(mask, name):
        sub = rdf[mask]
        if sub.empty:
            return {"name": name, "n": 0}
        return {
            "name": name,
            "n": len(sub),
            "wr": round(sub["is_win"].mean() * 100, 1),
            "sl_bar1_pct": round(sub["sl_bar1"].mean() * 100, 1),
            "sl_bar2_pct": round(sub["sl_bar2"].mean() * 100, 1),
            "dir_ok_12_pct": round(sub["dir_ok_12"].mean() * 100, 1),
            "loss_dir_ok_12_pct": round(sub["loss_dir_ok_12"].mean() * 100, 1),
            "mean_fwd_ret_1_pct": round(sub["fwd_ret_1"].mean() * 100, 3),
            "mean_fwd_ret_6_pct": round(sub["fwd_ret_6"].mean() * 100, 3),
            "mean_mae_12_pct": round(sub["mae_12"].mean() * 100, 3),
            "mean_mfe_12_pct": round(sub["mfe_12"].mean() * 100, 3),
        }

    losers = ~rdf["is_win"]
    sl_fast = rdf["sl_touch_bar"].notna() & (rdf["sl_touch_bar"] <= 2)
    high_vol = rdf["vol_spike"] >= 2.0

    slices = [
        agg_slice(pd.Series(True, index=rdf.index), "all"),
        agg_slice(losers, "losers"),
        agg_slice(rdf["is_win"], "winners"),
        agg_slice(sl_fast, "sl_touch_within_2bars"),
        agg_slice(losers & rdf["dir_ok_12"], "losers_but_dir_ok_12h"),
        agg_slice(losers & rdf["sl_bar1"], "losers_sl_bar1"),
        agg_slice(losers & rdf["sl_bar1"] & rdf["dir_ok_12"], "sl_bar1_dir_ok_12h"),
        agg_slice(high_vol, "vol_spike_ge_2"),
        agg_slice(high_vol & losers, "vol_spike_ge_2_losers"),
        agg_slice(rdf["repeat_1h"], "repeat_entry_1h_gap"),
        agg_slice(rdf["repeat_1h"] & losers, "repeat_1h_losers"),
    ]

    # hold_bars distribution for losers
    hold_los = rdf.loc[losers, "hold_bars"].describe().to_dict() if losers.any() else {}

    return {
        "label": label,
        "n_trades": len(rdf),
        "n_analyzed": len(rdf),
        "slices": slices,
        "losers_hold_bars": {k: round(v, 2) for k, v in hold_los.items() if isinstance(v, (int, float))},
        "pct_sl_touch_bar1": round(rdf["sl_bar1"].mean() * 100, 2),
        "pct_sl_touch_bar2": round(rdf["sl_bar2"].mean() * 100, 2),
        "pct_loss_dir_ok_12": round(rdf.loc[losers, "dir_ok_12"].mean() * 100, 2) if losers.any() else 0,
        "fwd_ret_by_bar": {
            str(h): round(rdf[f"fwd_ret_{h}"].mean() * 100, 4) for h in FWD_BARS
        },
    }


def try_live_trades() -> pd.DataFrame | None:
    try:
        from tools.live_db_bridge import pull_live_db, load_trades
        pull_live_db()
        live = load_trades()
        if live is None or live.empty:
            return None
        live = live[live.get("model_type", live.get("model", "")).astype(str).str.contains("ic32", case=False, na=False)]
        if live.empty:
            return None
        coin_col = "coin_symbol" if "coin_symbol" in live.columns else ("symbol" if "symbol" in live.columns else "coin")
        out = pd.DataFrame({
            "coin": live[coin_col],
            "entry_time": live["opened_at"] if "opened_at" in live.columns else live.get("entry_time"),
            "direction": live["direction"],
            "confidence": live.get("confidence", live.get("signal_confidence", np.nan)),
            "entry_price": live.get("entry_price", live.get("entry", np.nan)),
            "exit_price": live.get("exit_price", live.get("exit", np.nan)),
            "sl": live.get("sl_price", live.get("sl", np.nan)),
            "outcome": live.get("exit_reason", live.get("outcome", "")),
            "net_pnl": live.get("pnl_net", live.get("net_pnl", 0)),
            "hold_bars": live.get("hold_bars", np.nan),
            "vol_ratio_20": np.nan,
            "is_win": live.get("pnl_net", live.get("net_pnl", 0)) > 0,
            "exit_norm": live.get("exit_reason", ""),
        })
        return out.dropna(subset=["coin", "entry_time"])
    except Exception as exc:
        print(f"Live skip: {exc}")
        return None


def write_md(holdout: dict, live: dict | None):
    lines = [
        "# Post-Signal Path Analysis — ic32_regime_v1",
        "",
        "Empiris: pergerakan harga **setelah** bar entry (holdout Apr-Jun 2026).",
        "",
        f"## Holdout ({holdout.get('n_trades', 0)} trades)",
        "",
        f"- SL tersentuh **bar +1**: {holdout.get('pct_sl_touch_bar1', 0)}%",
        f"- SL tersentuh **bar +2**: {holdout.get('pct_sl_touch_bar2', 0)}%",
        f"- Losers yang arah benar di +12h: {holdout.get('pct_loss_dir_ok_12', 0)}%",
        "",
        "### Forward return rata-rata (semua trade, directional %)",
        "",
    ]
    for h, v in holdout.get("fwd_ret_by_bar", {}).items():
        lines.append(f"- +{h}h: {v:+.4f}%")
    lines.extend(["", "### Slice comparison", "", "| Slice | n | WR% | SL bar1% | dir_ok_12% | loss+dir_ok_12% | fwd+1% | mae_12% |", "|-------|--:|----:|---------:|-----------:|----------------:|-------:|--------:|"])
    for s in holdout.get("slices", []):
        if s.get("n", 0) == 0:
            continue
        lines.append(
            f"| {s['name']} | {s['n']} | {s.get('wr', '-')} | {s.get('sl_bar1_pct', '-')} | "
            f"{s.get('dir_ok_12_pct', '-')} | {s.get('loss_dir_ok_12_pct', '-')} | "
            f"{s.get('mean_fwd_ret_1_pct', '-')} | {s.get('mean_mae_12_pct', '-')} |"
        )
    if live:
        lines.extend(["", f"## Live ic32 ({live.get('n_trades', 0)} trades)", ""])
        lines.append(f"- SL bar +1: {live.get('pct_sl_touch_bar1', 0)}%")
        lines.append(f"- Losers dir ok +12h: {live.get('pct_loss_dir_ok_12', 0)}%")
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")


def main():
    if not HOLDOUT_TRADES.exists():
        raise FileNotFoundError(HOLDOUT_TRADES)

    hold = pd.read_csv(HOLDOUT_TRADES)
    holdout_res = analyze_trades(hold, "holdout_apr_jun26")

    live_res = None
    live_df = try_live_trades()
    if live_df is not None and len(live_df) >= 10:
        live_res = analyze_trades(live_df, "live_ic32")

    out = {"holdout": holdout_res, "live": live_res}
    OUT_JSON.write_text(json.dumps(out, indent=2), encoding="utf-8")
    write_md(holdout_res, live_res)

    print(json.dumps(holdout_res, indent=2))
    if live_res:
        print("\n--- LIVE ---\n")
        print(json.dumps(live_res, indent=2))
    print(f"\nSaved {OUT_JSON}\nSaved {OUT_MD}")


if __name__ == "__main__":
    main()