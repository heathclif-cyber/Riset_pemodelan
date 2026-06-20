"""
Risk gate sweep — max_open_positions x daily_loss_limit pada holdout Apr-Jun 2026.

Stack: ic32 B-dir + hard_consensus + continuation_v1 + min_hold=4 + sl_trigger=close.
Simulasi kronologis (WITA) meniru execution.py live.
Modal $5 (scale 0.5x dari backtest $10).
"""
import json
import sys
import warnings
from datetime import datetime, timezone, timedelta
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
from pipeline.ic32_fusion_shared import build_per_bar_thresholds, load_b_dir_hmm_cfg
from pipeline import ic32_fusion_shared as ifs
from config import (
    ALL_COINS, HOLDOUT_DIR, MODEL_DIR, LABEL_MAP,
    LEVERAGE_SIM, FEE_PER_SIDE, SLIPPAGE_PER_SIDE,
    MAX_HOLDING_BARS, GUARDIAN_DYNAMIC_FEATURES,
    SWING_LABEL_MIN_RR, SWING_LABEL_MIN_TP, SWING_LABEL_MAX_SL,
    TP_SL_FALLBACK_TP, TP_SL_FALLBACK_SL,
    TRAILING_STOP_ENABLED, TRAILING_STOP_ATR, TRAILING_STOP_MIN_BARS,
)

MODAL = 5.0
SCALE = MODAL / 10.0
WITA = timezone(timedelta(hours=8))
HOLDOUT_LABEL_DIR = HOLDOUT_DIR / "labeled"
RUN_DIR = MODEL_DIR / "runs" / "ic32_regime_v1"
GDN_RUN = MODEL_DIR / "runs" / "ic32_guardian_continuation_v1"
OUT_JSON = RUN_DIR / "risk_gate_sweep_modal5.json"
FLOW_MOM_WINDOW = 3
DYN_EXTRA = {"cvd_slope_h4_delta_entry", "ofi_h4_delta_entry", "flow_momentum_3bar"}

MAX_POS_GRID = [1, 2, 3, 4, 5, 6, 8, 10, 15, 21]
DAILY_LOSS_GRID = [2, 3, 4, 5, 6, 8, 10, 999]


def _wita_date(ts: pd.Timestamp) -> str:
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return ts.astimezone(WITA).date().isoformat()


def _apply_live():
    prod = ifs.load_production_defaults()
    import config as pc
    with open(MODEL_DIR / "inference_config.json", encoding="utf-8") as f:
        inf = json.load(f)
    guardian = inf.get("guardian", {})
    for m in (pc, btu):
        m.CONFIDENCE_THRESHOLD_ENTRY = prod["conf_entry"]
        m.LSTM_ADJUST_AGREE_BOOST = prod["agree_boost"]
        m.LSTM_ADJUST_NEUTRAL_PEN = prod["neutral_pen"]
        m.LSTM_ADJUST_OPPOSITE_PEN = prod["opposite_pen"]
        m.LSTM_DIRECTIONAL_REVIEW_THRESHOLD = prod["dir_review_thr"]
        m.LSTM_FLAT_REVIEW_ENABLED = prod["flat_review"]
        m.LSTM_CONFIRMATION_ENABLED = True
        m.REGIME_AWARE_ALIGNMENT = prod["flip"]
        m.HMM_GATE_LSTM_ENABLED = prod["hmm_gate_lstm"]
    btu.SMART_ENTRY_MODE = "disabled"
    btu.MOMENTUM_DYNAMIC_THRESHOLD_ENABLED = False
    btu.TREND_DYNAMIC_THRESHOLD_ENABLED = False
    btu.LSTM_STANDALONE_ENABLED = False
    return {
        "conf_entry": prod["conf_entry"],
        "gdn_exit": float(guardian.get("exit_threshold", 0.65)),
        "gdn_min_hold": int(guardian.get("min_hold_bars", 4)),
        "sl_trigger_mode": str(inf.get("rr_gate", {}).get("sl_trigger_mode", "close")),
    }


def _prep_coin(sym: str) -> pd.DataFrame | None:
    p = HOLDOUT_LABEL_DIR / f"{sym}_features_v3.parquet"
    if not p.exists():
        return None
    df = pd.read_parquet(p)
    df = ensure_utc_index(df).sort_index()
    rp = HOLDOUT_LABEL_DIR / f"{sym}_regime_h1.parquet"
    if rp.exists():
        reg = pd.read_parquet(rp)
        if "hmm_regime_enc" in df.columns:
            df = df.drop(columns=["hmm_regime_enc"])
        df = df.join(reg[["hmm_regime_enc"]], how="left")
        df["hmm_regime_enc"] = df["hmm_regime_enc"].fillna(1).astype("int32")
    else:
        df["hmm_regime_enc"] = 1
    if "cvd_slope_h4" in df.columns:
        df["cvd_slope_h4_delta"] = df["cvd_slope_h4"].diff(1)
    if "ofi_z_score" in df.columns:
        df["flow_momentum_3bar"] = df["ofi_z_score"].rolling(FLOW_MOM_WINDOW, min_periods=1).mean()
    df = df[df["label"].astype(str).isin(LABEL_MAP)].copy()
    return df if len(df) >= 30 else None


def collect_trades() -> pd.DataFrame:
    live_cfg = _apply_live()
    hmm_cfg = load_b_dir_hmm_cfg()

    with open(GDN_RUN / "guardian_feature_cols.json", encoding="utf-8") as f:
        g_feats = json.load(f)
    dyn = set(GUARDIAN_DYNAMIC_FEATURES) | DYN_EXTRA
    g_static = [c for c in g_feats if c not in dyn]
    g_model = joblib.load(GDN_RUN / "guardian.pkl")
    g_scaler = joblib.load(GDN_RUN / "guardian_scaler.pkl")

    lgbm = joblib.load(RUN_DIR / "lgbm.pkl")
    lstm = load_lstm(MODEL_DIR / "lstm_best.pt", device="cpu")
    lstm_scaler = joblib.load(MODEL_DIR / "lstm_scaler.pkl")
    with open(MODEL_DIR / "feature_cols_ic32_regime.json", encoding="utf-8") as f:
        feat_cols = json.load(f)
    with open(MODEL_DIR / "feature_cols_lstm_temporal.json", encoding="utf-8") as f:
        lstm_feat = json.load(f)[:11]

    rows = []
    tid = 0
    for sym in ALL_COINS:
        df = _prep_coin(sym)
        if df is None:
            continue
        n = len(df)
        X = np.zeros((n, len(feat_cols)))
        for i, col in enumerate(feat_cols):
            if col in df.columns:
                X[:, i] = df[col].ffill().fillna(0).values

        hmm = df["hmm_regime_enc"].values.astype(np.int32)
        thr_l, thr_s = build_per_bar_thresholds(hmm, hmm_cfg)
        y_pred, conf = hierarchical_predict(
            None, lgbm, lstm, lstm_scaler, X, feat_cols, [], df,
            model_dir=RUN_DIR, lstm_feat_cols=lstm_feat,
            per_bar_thr_long=thr_l, per_bar_thr_short=thr_s,
        )
        below = (y_pred != 1) & (conf < live_cfg["conf_entry"])
        y_pred[below] = 1

        flow = (
            df["flow_momentum_3bar"].ffill().fillna(0).values
            if "flow_momentum_3bar" in df.columns else np.zeros(n)
        )
        Xg = compute_guardian_static_array(df, g_static)

        rep = full_trading_report(
            y_pred=y_pred,
            y_actual=df["label"].map(LABEL_MAP).values.astype(np.int64),
            atr=df["atr_14_h1"].values if "atr_14_h1" in df.columns else np.ones(n),
            close=df["close"].values, high=df["high"].values, low=df["low"].values,
            h4_swing_highs=df["h4_swing_high"].values if "h4_swing_high" in df.columns else None,
            h4_swing_lows=df["h4_swing_low"].values if "h4_swing_low" in df.columns else None,
            index=df.index, modal=10.0, leverages=LEVERAGE_SIM,
            fee_per_side=FEE_PER_SIDE, slippage=SLIPPAGE_PER_SIDE,
            min_rr=SWING_LABEL_MIN_RR, min_tp_atr=SWING_LABEL_MIN_TP,
            max_sl_atr=SWING_LABEL_MAX_SL,
            tp_fallback_atr=TP_SL_FALLBACK_TP, sl_fallback_atr=TP_SL_FALLBACK_SL,
            max_hold=MAX_HOLDING_BARS, symbol=sym, confidence=conf,
            guardian_model=g_model, guardian_scaler=g_scaler, X_guardian=Xg,
            guardian_exit_threshold=live_cfg["gdn_exit"],
            guardian_min_hold_bars=live_cfg["gdn_min_hold"],
            guardian_feat_cols=g_feats, guardian_static_names=g_static,
            flow_momentum_arr=flow,
            trailing_stop_enabled=TRAILING_STOP_ENABLED,
            trailing_stop_atr=TRAILING_STOP_ATR,
            trailing_stop_min_bars=TRAILING_STOP_MIN_BARS,
            h4_trend=df["h4_trend"].values if "h4_trend" in df.columns else None,
            vol_ratio=df["vol_ratio_20"].values if "vol_ratio_20" in df.columns else None,
            sl_trigger_mode=live_cfg["sl_trigger_mode"],
        )

        for t in rep.get("lev5x", rep).get("trades", []):
            bi = int(t["bar_in"])
            bo = min(int(t["bar_out"]), n - 1)
            pnl = float(t.get("net_pnl", 0)) * SCALE
            rows.append({
                "trade_id": tid,
                "coin": sym,
                "direction": t.get("direction"),
                "ts_in": df.index[bi],
                "ts_out": df.index[bo],
                "net_pnl": pnl,
                "win": pnl > 0,
                "outcome": t.get("outcome"),
            })
            tid += 1

    df_tr = pd.DataFrame(rows)
    df_tr["ts_in"] = pd.to_datetime(df_tr["ts_in"], utc=True)
    df_tr["ts_out"] = pd.to_datetime(df_tr["ts_out"], utc=True)
    return df_tr.sort_values("ts_in").reset_index(drop=True)


def peak_concurrent(trades: pd.DataFrame) -> dict:
    events = []
    for _, r in trades.iterrows():
        events.append((r["ts_in"], 1))
        events.append((r["ts_out"], -1))
    events.sort(key=lambda x: (x[0], x[1]))
    peak = cur = 0
    for _, delta in events:
        cur += delta
        peak = max(peak, cur)
    return {"peak_open": peak, "peak_margin": peak * MODAL}


def simulate_gates(trades: pd.DataFrame, max_pos: int, daily_loss_lim: int) -> dict:
    lookup = trades.set_index("trade_id").to_dict("index")
    events = []
    for _, r in trades.iterrows():
        tid = int(r["trade_id"])
        events.append((r["ts_in"], "entry", tid))
        events.append((r["ts_out"], "exit", tid))
    events.sort(key=lambda x: (x[0], 0 if x[1] == "exit" else 1))

    open_ids = set()
    daily_losses = {}
    accepted_pnls = []
    rej_max = rej_daily = 0
    peak_open = 0
    days_halted = set()

    for ts, kind, tid in events:
        if kind == "exit":
            if tid in open_ids:
                open_ids.discard(tid)
                pnl = lookup[tid]["net_pnl"]
                day = _wita_date(ts)
                if pnl < 0:
                    daily_losses[day] = daily_losses.get(day, 0) + 1
        else:
            day = _wita_date(ts)
            if len(open_ids) >= max_pos:
                rej_max += 1
                continue
            if daily_loss_lim < 999 and daily_losses.get(day, 0) >= daily_loss_lim:
                rej_daily += 1
                days_halted.add(day)
                continue
            open_ids.add(tid)
            peak_open = max(peak_open, len(open_ids))
            accepted_pnls.append(lookup[tid]["net_pnl"])

    n = len(accepted_pnls)
    total = len(trades)
    if n == 0:
        return {
            "max_pos": max_pos, "daily_loss_limit": daily_loss_lim,
            "trades": 0, "wr": 0, "pnl": 0, "ppt": 0, "pf": 0,
            "peak_open": 0, "peak_margin": 0,
            "rejected_max_pos": rej_max, "rejected_daily_loss": rej_daily,
            "days_halted": len(days_halted), "capture_pct": 0,
        }

    pnls = np.array(accepted_pnls)
    wins = pnls[pnls > 0]
    losses = pnls[pnls <= 0]
    gross_win = wins.sum() if len(wins) else 0
    gross_loss = abs(losses.sum()) if len(losses) else 1e-9

    return {
        "max_pos": max_pos,
        "daily_loss_limit": daily_loss_lim if daily_loss_lim < 999 else None,
        "trades": n,
        "wr": round(100 * (pnls > 0).mean(), 2),
        "pnl": round(float(pnls.sum()), 2),
        "ppt": round(float(pnls.mean()), 4),
        "pf": round(float(gross_win / gross_loss), 3),
        "peak_open": peak_open,
        "peak_margin": peak_open * MODAL,
        "rejected_max_pos": rej_max,
        "rejected_daily_loss": rej_daily,
        "days_halted": len(days_halted),
        "capture_pct": round(100 * n / total, 1),
    }


def main():
    print("Collecting holdout trades (continuation_v1, min_hold=4, sl=close, modal $5)...")
    trades = collect_trades()
    print(f"  Raw trades: {len(trades)}")
    raw = {
        "trades": len(trades),
        "pnl": round(trades["net_pnl"].sum(), 2),
        "ppt": round(trades["net_pnl"].mean(), 4),
        "wr": round(100 * trades["win"].mean(), 2),
    }
    raw.update(peak_concurrent(trades))
    print(f"  No-gate: PnL=${raw['pnl']} PPT=${raw['ppt']} WR={raw['wr']}% peak_open={raw['peak_open']}")

    results = []
    for mp in MAX_POS_GRID:
        for dl in DAILY_LOSS_GRID:
            results.append(simulate_gates(trades, mp, dl))

    df = pd.DataFrame(results)

    print("\n=== TOP 15 by PnL (peak_margin <= $50) ===")
    sub50 = df[df["peak_margin"] <= 50].sort_values("pnl", ascending=False).head(15)
    print(sub50.to_string(index=False))

    print("\n=== TOP 15 by PnL (peak_margin <= $105) ===")
    sub105 = df[df["peak_margin"] <= 105].sort_values("pnl", ascending=False).head(15)
    print(sub105.to_string(index=False))

    print("\n=== VPS CURRENT (max_pos=10, daily_loss=5) ===")
    cur = df[(df["max_pos"] == 10) & (df["daily_loss_limit"] == 5)]
    if not cur.empty:
        print(cur.to_string(index=False))

    print("\n=== BASELINE no gate (max_pos=21, daily_loss=unlimited) ===")
    base = df[(df["max_pos"] == 21) & (df["daily_loss_limit"].isna())]
    if not base.empty:
        print(base.to_string(index=False))

    out = {
        "meta": {
            "modal": MODAL,
            "holdout": "Apr-Jun 2026",
            "guardian": "ic32_guardian_continuation_v1",
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        },
        "no_gate": raw,
        "grid": results,
        "recommended": {},
    }

    for cap, label in [(50, "saldo_50"), (105, "saldo_105")]:
        cand = df[(df["peak_margin"] <= cap) & (df["capture_pct"] >= 70)]
        if cand.empty:
            cand = df[df["peak_margin"] <= cap]
        if not cand.empty:
            best = cand.sort_values("pnl", ascending=False).iloc[0]
            out["recommended"][label] = best.to_dict()

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nSaved: {OUT_JSON}")


if __name__ == "__main__":
    main()