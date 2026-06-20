"""Analyze daily trade distribution for B-dir-combined holdout."""
import json
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import pipeline.backtest_utils as btu
from pipeline.backtest_utils import hierarchical_predict, compute_guardian_static_array
from core.evaluator import full_trading_report
from core.models import load_lstm
from core.utils import ensure_utc_index
from config import (
    ALL_COINS, HOLDOUT_DIR, MODEL_DIR, LABEL_MAP,
    MODAL_PER_TRADE, LEVERAGE_SIM, FEE_PER_SIDE, SLIPPAGE_PER_SIDE,
    MAX_HOLDING_BARS, GUARDIAN_DYNAMIC_FEATURES,
    SWING_LABEL_MIN_RR, SWING_LABEL_MIN_TP, SWING_LABEL_MAX_SL,
    TP_SL_FALLBACK_TP, TP_SL_FALLBACK_SL,
    TRAILING_STOP_ENABLED, TRAILING_STOP_ATR, TRAILING_STOP_MIN_BARS,
)

HOLDOUT_LABEL_DIR = HOLDOUT_DIR / "labeled"
RUN_DIR = MODEL_DIR / "runs" / "ic32_regime_v1"


def _setup():
    with open(RUN_DIR / "b_dir_combined_frozen.json", encoding="utf-8") as f:
        raw = json.load(f)["per_state_thresholds"]
    hmm_cfg = {int(k): (float(v[0]), float(v[1])) for k, v in raw.items()}

    with open(MODEL_DIR / "inference_config.json", encoding="utf-8") as f:
        cfg = json.load(f)
    cascade = cfg.get("cascade", {})
    guardian = cfg.get("guardian", {})
    ra = cfg.get("regime_alignment", {})

    btu.CONFIDENCE_THRESHOLD_ENTRY = float(cascade.get("confidence_threshold_entry", 0.59))
    btu.REGIME_AWARE_ALIGNMENT = bool(ra.get("enabled", True))
    btu.SMART_ENTRY_MODE = "disabled"
    btu.MOMENTUM_DYNAMIC_THRESHOLD_ENABLED = False
    btu.TREND_DYNAMIC_THRESHOLD_ENABLED = False
    btu.LSTM_STANDALONE_ENABLED = False

    return hmm_cfg, float(cascade.get("confidence_threshold_entry", 0.59)), float(
        guardian.get("exit_threshold", 0.65)
    ), int(guardian.get("min_hold_bars", 2))


def _build_thr(hmm_enc, hmm_cfg):
    n = len(hmm_enc)
    dtl, dts = hmm_cfg[-1]
    tl = np.full(n, dtl, dtype=np.float64)
    ts = np.full(n, dts, dtype=np.float64)
    for st, (a, b) in hmm_cfg.items():
        if st == -1:
            continue
        m = hmm_enc == st
        tl[m], ts[m] = a, b
    return tl, ts


def collect_trades():
    hmm_cfg, conf_thr, gdn_exit, gdn_min = _setup()

    lgbm = joblib.load(RUN_DIR / "lgbm.pkl")
    lstm = load_lstm(MODEL_DIR / "lstm_best.pt", device="cpu")
    lstm_scaler = joblib.load(MODEL_DIR / "lstm_scaler.pkl")
    with open(MODEL_DIR / "feature_cols_ic32_regime.json", encoding="utf-8") as f:
        feat_cols = json.load(f)
    with open(MODEL_DIR / "feature_cols_lstm_temporal.json", encoding="utf-8") as f:
        lstm_feat = json.load(f)
    g_model = joblib.load(MODEL_DIR / "guardian_clean_v2.pkl")
    g_scaler = joblib.load(MODEL_DIR / "guardian_clean_v2_scaler.pkl")
    with open(MODEL_DIR / "guardian_clean_v2_feature_cols.json", encoding="utf-8") as f:
        g_feats = json.load(f)
    g_static = [c for c in g_feats if c not in set(GUARDIAN_DYNAMIC_FEATURES)]

    rows = []
    for sym in ALL_COINS:
        p = HOLDOUT_LABEL_DIR / f"{sym}_features_v3.parquet"
        if not p.exists():
            continue
        df = pd.read_parquet(p)
        df = ensure_utc_index(df).sort_index()
        rp = HOLDOUT_LABEL_DIR / f"{sym}_regime_h1.parquet"
        if rp.exists():
            reg = pd.read_parquet(rp)
            for col in ["hmm_regime_enc", "hmm_regime"]:
                if col in df.columns:
                    df = df.drop(columns=[col])
            cols = [c for c in ["hmm_regime_enc", "hmm_regime"] if c in reg.columns]
            df = df.join(reg[cols], how="left")
            df["hmm_regime_enc"] = df["hmm_regime_enc"].fillna(1).astype("int32")
        else:
            df["hmm_regime_enc"] = 1

        mask = df["label"].astype(str).isin(LABEL_MAP)
        df = df[mask].copy()
        n = len(df)
        if n < 30:
            continue

        X = np.zeros((n, len(feat_cols)))
        for i, col in enumerate(feat_cols):
            if col in df.columns:
                X[:, i] = df[col].ffill().fillna(0).values

        hmm = df["hmm_regime_enc"].values.astype(np.int32)
        thr_l, thr_s = _build_thr(hmm, hmm_cfg)
        y_pred, conf = hierarchical_predict(
            None, lgbm, lstm, lstm_scaler, X, feat_cols, [], df,
            model_dir=RUN_DIR, lstm_feat_cols=lstm_feat,
            per_bar_thr_long=thr_l, per_bar_thr_short=thr_s,
        )
        below = (y_pred != 1) & (conf < conf_thr)
        y_pred[below] = 1

        y = df["label"].map(LABEL_MAP).values.astype(np.int64)
        atr = df["atr_14_h1"].values if "atr_14_h1" in df.columns else np.ones(n)
        h4t = df["h4_trend"].values if "h4_trend" in df.columns else None
        volr = df["vol_ratio_20"].values if "vol_ratio_20" in df.columns else None
        Xg = compute_guardian_static_array(df, g_static)

        rep = full_trading_report(
            y_pred=y_pred, y_actual=y, atr=atr,
            close=df["close"].values, high=df["high"].values, low=df["low"].values,
            h4_swing_highs=df["h4_swing_high"].values if "h4_swing_high" in df.columns else None,
            h4_swing_lows=df["h4_swing_low"].values if "h4_swing_low" in df.columns else None,
            index=df.index, modal=MODAL_PER_TRADE, leverages=LEVERAGE_SIM,
            fee_per_side=FEE_PER_SIDE, slippage=SLIPPAGE_PER_SIDE,
            min_rr=SWING_LABEL_MIN_RR, min_tp_atr=SWING_LABEL_MIN_TP,
            max_sl_atr=SWING_LABEL_MAX_SL,
            tp_fallback_atr=TP_SL_FALLBACK_TP, sl_fallback_atr=TP_SL_FALLBACK_SL,
            max_hold=MAX_HOLDING_BARS, symbol=sym, confidence=conf,
            guardian_model=g_model, guardian_scaler=g_scaler, X_guardian=Xg,
            guardian_exit_threshold=gdn_exit, guardian_min_hold_bars=gdn_min,
            trailing_stop_enabled=TRAILING_STOP_ENABLED,
            trailing_stop_atr=TRAILING_STOP_ATR,
            trailing_stop_min_bars=TRAILING_STOP_MIN_BARS,
            h4_trend=h4t, vol_ratio=volr,
        )

        for t in rep.get("trades", []):
            bi = int(t["bar_in"])
            entry_dt = df.index[bi]
            pnl = float(t.get("net_pnl", 0))
            rows.append({
                "date": entry_dt.date().isoformat(),
                "coin": sym,
                "direction": t.get("direction"),
                "net_pnl": pnl,
                "outcome": t.get("outcome"),
                "win": pnl > 0,
            })
    return pd.DataFrame(rows)


def main():
    df = collect_trades()
    daily = df.groupby("date").agg(
        trades=("net_pnl", "count"),
        wins=("win", "sum"),
        pnl=("net_pnl", "sum"),
    ).reset_index()
    daily["losses"] = daily["trades"] - daily["wins"]
    daily["wr"] = daily["wins"] / daily["trades"] * 100

    n_days = len(daily)
    cal_days = (pd.to_datetime(daily["date"].max()) - pd.to_datetime(daily["date"].min())).days + 1
    total = len(df)

    best = daily.loc[daily["pnl"].idxmax()]
    worst = daily.loc[daily["pnl"].idxmin()]

    print("=== TRADE PER HARI (B-dir-combined holdout Apr-Jun 2026) ===")
    print(f"Total trades     : {total}")
    print(f"Hari ada trade   : {n_days}")
    print(f"Rentang kalender : {daily['date'].min()} s/d {daily['date'].max()} ({cal_days} hari)")
    print(f"Avg trade/hari (hari aktif)   : {total / n_days:.2f}")
    print(f"Avg trade/hari (kalender full): {total / cal_days:.2f}")
    print(f"Median trade/hari             : {daily['trades'].median():.0f}")
    print(f"Min - Max trade/hari          : {int(daily['trades'].min())} - {int(daily['trades'].max())}")
    print(f"Hari tanpa trade (kalender)   : {cal_days - n_days}")

    print("\n=== HARI WIN TERBAIK (PnL tertinggi) ===")
    print(f"Tanggal  : {best['date']}")
    print(f"Trades   : {int(best['trades'])} (W={int(best['wins'])} / L={int(best['losses'])})")
    print(f"WR       : {best['wr']:.1f}%")
    print(f"PnL hari : ${best['pnl']:+.2f}")
    for _, r in df[df["date"] == best["date"]].sort_values("net_pnl", ascending=False).iterrows():
        print(f"  {r['coin']:<16} {r['direction']:<5} ${r['net_pnl']:+.2f}  {r['outcome']}")

    print("\n=== HARI LOSS TERBURUK (PnL terendah) ===")
    print(f"Tanggal  : {worst['date']}")
    print(f"Trades   : {int(worst['trades'])} (W={int(worst['wins'])} / L={int(worst['losses'])})")
    print(f"WR       : {worst['wr']:.1f}%")
    print(f"PnL hari : ${worst['pnl']:+.2f}")
    for _, r in df[df["date"] == worst["date"]].sort_values("net_pnl").iterrows():
        print(f"  {r['coin']:<16} {r['direction']:<5} ${r['net_pnl']:+.2f}  {r['outcome']}")

    print("\n=== TOP 5 HARI TERBAIK ===")
    for _, r in daily.nlargest(5, "pnl").iterrows():
        print(f"  {r['date']}  {int(r['trades']):>3} trades  W{int(r['wins'])}/L{int(r['losses'])}  ${r['pnl']:+.2f}")

    print("\n=== TOP 5 HARI TERBURUK ===")
    for _, r in daily.nsmallest(5, "pnl").iterrows():
        print(f"  {r['date']}  {int(r['trades']):>3} trades  W{int(r['wins'])}/L{int(r['losses'])}  ${r['pnl']:+.2f}")

    out = RUN_DIR / "holdout_b_dir_daily_stats.json"
    summary = {
        "total_trades": int(total),
        "active_days": int(n_days),
        "calendar_days": int(cal_days),
        "avg_per_active_day": round(total / n_days, 2),
        "avg_per_calendar_day": round(total / cal_days, 2),
        "median_per_day": float(daily["trades"].median()),
        "min_max_per_day": [int(daily["trades"].min()), int(daily["trades"].max())],
        "best_day": {
            "date": best["date"],
            "trades": int(best["trades"]),
            "wins": int(best["wins"]),
            "losses": int(best["losses"]),
            "pnl": round(float(best["pnl"]), 2),
        },
        "worst_day": {
            "date": worst["date"],
            "trades": int(worst["trades"]),
            "wins": int(worst["wins"]),
            "losses": int(worst["losses"]),
            "pnl": round(float(worst["pnl"]), 2),
        },
        "top5_best": daily.nlargest(5, "pnl")[["date", "trades", "wins", "losses", "pnl"]].to_dict("records"),
        "top5_worst": daily.nsmallest(5, "pnl")[["date", "trades", "wins", "losses", "pnl"]].to_dict("records"),
    }
    with open(out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()