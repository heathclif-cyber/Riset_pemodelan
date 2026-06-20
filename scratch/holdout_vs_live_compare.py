# -*- coding: utf-8 -*-
"""
Export holdout scale_in lengkap + bandingkan dengan live DB (ic32_regime_v1).
Output: reports/experiments/holdout_vs_live_*
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from importlib import import_module
from core.models import load_lstm
from config import ALL_COINS, MODEL_DIR, OOS_START, MODAL_PER_TRADE
from pipeline.ic32_fusion_shared import load_b_dir_hmm_cfg
from tools.live_db_bridge import pull_live_db, load_trades, load_signals

OUT_DIR = ROOT / "reports" / "experiments"
TZ = "Asia/Makassar"
MODEL = "ic32_regime_v1"
VARIANT = {"label": "pyr2_scale_in", "enabled": True, "max_per_coin": 2, "exit_mode": "scale_in"}


def _pf(gw: float, gl: float):
    if gl <= 0:
        return None if gw <= 0 else 999.0
    return gw / gl


def _scorecard(df: pd.DataFrame, pnl_col: str = "net_pnl") -> dict:
    if df.empty:
        return {"trades": 0}
    wins = df[df[pnl_col] > 0]
    losses = df[df[pnl_col] <= 0]
    gw = float(wins[pnl_col].sum())
    gl = abs(float(losses[pnl_col].sum()))
    pnl = float(df[pnl_col].sum())
    n = len(df)
    long_n = int((df["direction"] == "LONG").sum()) if "direction" in df.columns else 0
    short_n = int((df["direction"] == "SHORT").sum()) if "direction" in df.columns else 0
    return {
        "trades": n,
        "wins": len(wins),
        "losses": len(losses),
        "win_rate_pct": round(len(wins) / n * 100, 2),
        "profit_factor": round(_pf(gw, gl), 3) if _pf(gw, gl) is not None else None,
        "net_pnl": round(pnl, 2),
        "pnl_per_trade": round(pnl / n, 4),
        "long_trades": long_n,
        "short_trades": short_n,
        "coins_traded": int(df["symbol"].nunique()) if "symbol" in df.columns else int(df["coin_symbol"].nunique()),
    }


def _daily(df: pd.DataFrame, date_col: str, pnl_col: str) -> pd.DataFrame:
    rows = []
    for date, sub in df.groupby(date_col):
        sc = _scorecard(sub.assign(direction=sub["direction"]), pnl_col)
        sc["date"] = str(date)
        rows.append(sc)
    out = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    if not out.empty:
        out["cum_pnl"] = out["net_pnl"].cumsum().round(2)
    return out


def _collect_holdout() -> pd.DataFrame:
    h07 = import_module("pipeline.07h_holdout_ic32_scale_in_diag")
    live_cfg = h07._apply_live_config()
    hmm_cfg = load_b_dir_hmm_cfg()
    gdn = h07._load_guardian_cont()
    with open(MODEL_DIR / "feature_cols_ic32_regime.json", encoding="utf-8") as f:
        feat_cols = json.load(f)
    with open(MODEL_DIR / "feature_cols_lstm_temporal.json", encoding="utf-8") as f:
        lstm_feats = json.load(f)[:11]
    lgbm = joblib.load(MODEL_DIR / "runs/ic32_regime_v1/lgbm.pkl")
    lstm = load_lstm(MODEL_DIR / "lstm_best.pt", device="cpu")
    lstm_scaler = joblib.load(MODEL_DIR / "lstm_scaler.pkl")

    trades = []
    for sym in ALL_COINS:
        trades.extend(
            h07._run_holdout(sym, hmm_cfg, live_cfg, gdn, feat_cols, lstm_feats,
                             lgbm, lstm, lstm_scaler, VARIANT)
        )

    rows = []
    for t in trades:
        ts_in = pd.Timestamp(t["ts_in"])
        ts_out = pd.Timestamp(t["ts_out"])
        if ts_in.tzinfo is None:
            ts_in = ts_in.tz_localize("UTC")
        else:
            ts_in = ts_in.tz_convert("UTC")
        if ts_out.tzinfo is None:
            ts_out = ts_out.tz_localize("UTC")
        else:
            ts_out = ts_out.tz_convert("UTC")
        rows.append({
            "source": "holdout_scale_in",
            "model": MODEL,
            "symbol": t["symbol"],
            "direction": t["direction"],
            "ts_in_utc": ts_in.isoformat(),
            "ts_out_utc": ts_out.isoformat(),
            "ts_in_wita": ts_in.tz_convert(TZ).strftime("%Y-%m-%d %H:%M:%S"),
            "ts_out_wita": ts_out.tz_convert(TZ).strftime("%Y-%m-%d %H:%M:%S"),
            "date_entry_utc": str(ts_in.date()),
            "date_entry_wita": ts_in.tz_convert(TZ).strftime("%Y-%m-%d"),
            "entry": t.get("entry"),
            "exit": t.get("exit"),
            "tp": t.get("tp"),
            "sl": t.get("sl"),
            "outcome": t.get("outcome"),
            "net_pnl": round(float(t["net_pnl"]), 4),
            "modal_used": t.get("modal_used", MODAL_PER_TRADE),
            "n_legs": t.get("n_legs", 1),
            "scale_in": t.get("scale_in", True),
            "hold_bars": int(t.get("bar_out", 0) - t.get("bar_in", 0)),
            "is_win": t["net_pnl"] > 0,
        })
    return pd.DataFrame(rows).sort_values(["ts_in_utc", "symbol"]).reset_index(drop=True)


def _load_live_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    pull_live_db()
    trades = load_trades()
    signals = load_signals()
    trades = trades[trades["model_type"] == MODEL].copy()
    signals = signals[signals["model_type"] == MODEL].copy()
    return trades, signals


def _normalize_live_trades(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for r in df.itertuples(index=False):
        d = r._asdict()
        opened = pd.Timestamp(d["opened_at"], tz="UTC")
        closed = pd.Timestamp(d["closed_at"], tz="UTC") if d.get("closed_at") else pd.NaT
        rows.append({
            "source": "live_db",
            "model": MODEL,
            "symbol": d["coin_symbol"],
            "direction": d["direction"],
            "is_live": int(d.get("is_live") or 0),
            "ts_in_utc": opened.isoformat(),
            "ts_out_utc": closed.isoformat() if pd.notna(closed) else None,
            "ts_in_wita": opened.tz_convert(TZ).strftime("%Y-%m-%d %H:%M:%S"),
            "ts_out_wita": closed.tz_convert(TZ).strftime("%Y-%m-%d %H:%M:%S") if pd.notna(closed) else None,
            "date_entry_utc": str(opened.date()),
            "date_entry_wita": opened.tz_convert(TZ).strftime("%Y-%m-%d"),
            "entry": d.get("entry_price"),
            "exit": d.get("exit_price"),
            "tp": d.get("tp_price"),
            "sl": d.get("sl_price"),
            "outcome": d.get("exit_reason"),
            "net_pnl": round(float(d["pnl_net"]), 4) if d.get("pnl_net") is not None else None,
            "modal_used": d.get("quantity"),
            "signal_confidence": d.get("signal_confidence"),
            "hold_bars": d.get("hold_bars"),
            "status": d.get("status"),
            "is_win": (d.get("pnl_net") or 0) > 0,
        })
    return pd.DataFrame(rows).sort_values(["ts_in_utc", "symbol"]).reset_index(drop=True)


def _normalize_live_signals(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for r in df.itertuples(index=False):
        d = r._asdict()
        st = pd.Timestamp(d["signal_time"], tz="UTC")
        rows.append({
            "source": "live_db_signal",
            "model": MODEL,
            "symbol": d["coin_symbol"],
            "direction": d["direction"],
            "confidence": d.get("confidence"),
            "ts_utc": st.isoformat(),
            "ts_wita": st.tz_convert(TZ).strftime("%Y-%m-%d %H:%M:%S"),
            "date_wita": st.tz_convert(TZ).strftime("%Y-%m-%d"),
            "hour_wita": int(st.tz_convert(TZ).hour),
            "entry_reason": d.get("entry_reason"),
            "lstm_direction": d.get("lstm_direction"),
            "lstm_confidence": d.get("lstm_confidence"),
        })
    return pd.DataFrame(rows)


def _coin_direction_compare(hold: pd.DataFrame, live: pd.DataFrame) -> pd.DataFrame:
    h = hold.groupby("symbol")["direction"].value_counts().unstack(fill_value=0).rename(
        columns=lambda c: f"hold_{c.lower()}"
    )
    l = live.groupby("symbol")["direction"].value_counts().unstack(fill_value=0).rename(
        columns=lambda c: f"live_{c.lower()}"
    )
    merged = h.join(l, how="outer").fillna(0).astype(int).reset_index()
    for col in ["hold_long", "hold_short", "live_long", "live_short"]:
        if col not in merged.columns:
            merged[col] = 0
    merged["dir_mismatch"] = (
        (merged["hold_long"] != merged["live_long"])
        | (merged["hold_short"] != merged["live_short"])
    )
    return merged.sort_values("dir_mismatch", ascending=False)


def _hourly_direction(df: pd.DataFrame, ts_col: str) -> pd.DataFrame:
    tmp = df.copy()
    tmp["hour_wita"] = pd.to_datetime(tmp[ts_col], utc=True).dt.tz_convert(TZ).dt.hour
    return tmp.groupby(["date_entry_wita", "hour_wita", "direction"]).size().reset_index(name="n")


def _overlap_analysis(hold: pd.DataFrame, live: pd.DataFrame, signals: pd.DataFrame) -> dict:
    """Analisis hari dengan perbedaan arah dominan."""
    findings = []
    hold_daily = _daily(hold.rename(columns={"symbol": "symbol"}), "date_entry_wita", "net_pnl")
    live_all = live.copy()
    live_live_only = live[live["is_live"] == 1].copy()

    common_dates = sorted(
        set(hold["date_entry_wita"]) & set(live_live_only["date_entry_wita"])
    )

    for d in common_dates:
        h_day = hold[hold["date_entry_wita"] == d]
        l_day = live_live_only[live_live_only["date_entry_wita"] == d]
        if h_day.empty and l_day.empty:
            continue
        h_long = int((h_day["direction"] == "LONG").sum())
        h_short = int((h_day["direction"] == "SHORT").sum())
        l_long = int((l_day["direction"] == "LONG").sum())
        l_short = int((l_day["direction"] == "SHORT").sum())
        if h_long == l_long and h_short == l_short:
            continue

        sig_day = signals[signals["date_wita"] == d]
        flat_n = int((sig_day["direction"] == "FLAT").sum()) if not sig_day.empty else 0
        trade_n = int((sig_day["direction"].isin(["LONG", "SHORT"])).sum()) if not sig_day.empty else 0

        findings.append({
            "date_wita": d,
            "hold_trades": len(h_day),
            "hold_long": h_long,
            "hold_short": h_short,
            "hold_pnl": round(float(h_day["net_pnl"].sum()), 2),
            "live_trades": len(l_day),
            "live_long": l_long,
            "live_short": l_short,
            "live_pnl": round(float(l_day["net_pnl"].sum()), 2),
            "live_signals_flat": flat_n,
            "live_signals_trade": trade_n,
            "hold_dominant": "LONG" if h_long > h_short else ("SHORT" if h_short > h_long else "MIXED"),
            "live_dominant": "LONG" if l_long > l_short else ("SHORT" if l_short > l_long else "MIXED"),
        })

    return {"mismatch_days": findings, "common_dates_count": len(common_dates)}


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Collecting holdout scale_in trades...")
    hold = _collect_holdout()

    print("Loading live DB...")
    live_raw, sig_raw = _load_live_frames()
    live = _normalize_live_trades(live_raw)
    signals = _normalize_live_signals(sig_raw)

    # Filter overlap period: holdout start -> max live date
    hold_start = str(OOS_START.date())
    live_max = live["date_entry_wita"].max() if not live.empty else hold_start
    live = live[live["date_entry_wita"] >= hold_start].copy()
    signals = signals[signals["date_wita"] >= hold_start].copy()
    hold = hold[hold["date_entry_wita"] <= live_max].copy()

    live_paper = live[live["is_live"] == 0].copy()
    live_real = live[live["is_live"] == 1].copy()

    # Save full exports
    paths = {
        "holdout_trades": OUT_DIR / "holdout_scale_in_trades_full.csv",
        "holdout_daily_utc": OUT_DIR / "holdout_scale_in_daily_utc.csv",
        "holdout_daily_wita": OUT_DIR / "holdout_scale_in_daily_wita.csv",
        "live_trades_all": OUT_DIR / "live_ic32_trades_all.csv",
        "live_trades_real": OUT_DIR / "live_ic32_trades_is_live.csv",
        "live_signals": OUT_DIR / "live_ic32_signals.csv",
        "coin_compare": OUT_DIR / "holdout_vs_live_coin_direction.csv",
        "daily_compare": OUT_DIR / "holdout_vs_live_daily_wita.csv",
    }
    hold.to_csv(paths["holdout_trades"], index=False)
    _daily(hold, "date_entry_utc", "net_pnl").to_csv(paths["holdout_daily_utc"], index=False)
    _daily(hold, "date_entry_wita", "net_pnl").to_csv(paths["holdout_daily_wita"], index=False)
    live.to_csv(paths["live_trades_all"], index=False)
    live_real.to_csv(paths["live_trades_real"], index=False)
    signals.to_csv(paths["live_signals"], index=False)

    coin_cmp = _coin_direction_compare(hold, live_real)
    coin_cmp.to_csv(paths["coin_compare"], index=False)

    # Daily side-by-side (WITA entry date)
    h_d = _daily(hold, "date_entry_wita", "net_pnl").rename(
        columns=lambda c: f"hold_{c}" if c != "date" else c
    )
    l_d = _daily(live_real, "date_entry_wita", "net_pnl").rename(
        columns=lambda c: f"live_{c}" if c != "date" else c
    )
    daily_cmp = h_d.merge(l_d, on="date", how="outer").sort_values("date")
    daily_cmp.to_csv(paths["daily_compare"], index=False)

    overlap = _overlap_analysis(hold, live, signals)

    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "model": MODEL,
        "holdout_variant": VARIANT["label"],
        "period": {
            "holdout_start": hold_start,
            "live_end": live_max,
            "holdout_trades": len(hold),
            "live_trades_all": len(live),
            "live_trades_paper": len(live_paper),
            "live_trades_real": len(live_real),
            "live_signals": len(signals),
        },
        "scorecard": {
            "holdout_scale_in": _scorecard(hold),
            "live_all": _scorecard(live),
            "live_paper": _scorecard(live_paper),
            "live_real": _scorecard(live_real),
        },
        "coin_direction_mismatch": {
            "coins_total": len(coin_cmp),
            "coins_mismatch": int(coin_cmp["dir_mismatch"].sum()),
            "top_mismatch": coin_cmp[coin_cmp["dir_mismatch"]].head(15).to_dict("records"),
        },
        "day_direction_mismatch": overlap,
        "explanation": {
            "why_different": [
                "Holdout pakai parquet batch (data/holdout-test); live pakai feature real-time dari cron fetch — nilai fitur bisa beda di bar yang sama.",
                "Holdout variant scale_in: 1 posisi/koin, tidak boleh flip arah saat posisi terbuka; live juga scale_in tapi timing entry beda karena sinyal beda.",
                "Live punya fase paper (is_live=0) sebelum Juni 2026; bandingkan live_real untuk apples-to-apples.",
                "Agregat harian holdout Excel pakai date_entry UTC; analisis ini juga sediakan date_entry WITA.",
                "LSTM flat-review + confidence threshold di live bisa ubah FLAT yang di holdout jadi LONG/SHORT (atau sebaliknya).",
                "Data holdout HMM regime di-precompute; live pakai regime dari inference bar-by-bar — bisa beda state.",
            ],
            "jun11_example": None,
        },
        "output_files": {k: str(v) for k, v in paths.items()},
    }

    # Jun 11 detail
    d11 = "2026-06-11"
    h11 = hold[hold["date_entry_wita"] == d11]
    l11 = live_real[live_real["date_entry_wita"] == d11]
    s11 = signals[signals["date_wita"] == d11]
    report["explanation"]["jun11_example"] = {
        "holdout_trades": h11[["symbol", "direction", "ts_in_wita", "net_pnl", "n_legs"]].to_dict("records"),
        "live_trades": l11[["symbol", "direction", "ts_in_wita", "net_pnl", "signal_confidence"]].to_dict("records"),
        "live_signal_counts": s11["direction"].value_counts().to_dict() if not s11.empty else {},
        "live_flat_hours_with_lgbm_long": [],
    }
    if not s11.empty:
        trade_sigs = s11[s11["direction"].isin(["LONG", "SHORT"])]
        report["explanation"]["jun11_example"]["live_trade_signals"] = len(trade_sigs)

    out_json = OUT_DIR / "holdout_vs_live_comparison.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)

    print(f"\nSaved {len(paths)} CSV files + {out_json.name}")
    print("\n=== SCORECARD ===")
    for k, v in report["scorecard"].items():
        print(f"{k}: {v}")
    print(f"\nCoin direction mismatch: {report['coin_direction_mismatch']['coins_mismatch']}/{report['coin_direction_mismatch']['coins_total']}")
    print(f"Day direction mismatch (live real): {len(overlap['mismatch_days'])} days")


if __name__ == "__main__":
    main()