"""
Diagnostic holdout: pyramiding variants (sekali, konfirmasi OOF 08i).

Stack frozen: B-dir + hard_consensus + continuation_v1 + min_hold=4 + SL close.

Variants:
  no_pyr              — live saat ini
  pyr2_shared_sl_first — tradeoff terkecil OOF
  pyr2_independent    — exit default live bila pyramiding ON

Usage:
  python pipeline/07g_holdout_ic32_pyramiding_diag.py
"""
import json
import sys
import warnings
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import pipeline.backtest_utils as btu
from pipeline.backtest_utils import hierarchical_predict, compute_guardian_static_array, apply_training_parity
from core.evaluator import full_trading_report
from core.models import load_lstm
from core.utils import setup_logger, ensure_utc_index
from pipeline.ic32_fusion_shared import build_per_bar_thresholds, load_b_dir_hmm_cfg
from pipeline import ic32_fusion_shared as ifs
from config import (
    ALL_COINS, HOLDOUT_DIR, MODEL_DIR, LABEL_MAP, OOS_START,
    MODAL_PER_TRADE, LEVERAGE_SIM, FEE_PER_SIDE, SLIPPAGE_PER_SIDE,
    MAX_HOLDING_BARS, GUARDIAN_DYNAMIC_FEATURES, GUARDIAN_MIN_HOLD_BARS,
    SWING_LABEL_MIN_RR, SWING_LABEL_MIN_TP, SWING_LABEL_MAX_SL,
    TP_SL_FALLBACK_TP, TP_SL_FALLBACK_SL,
    TRAILING_STOP_ENABLED, TRAILING_STOP_ATR, TRAILING_STOP_MIN_BARS,
)

logger = setup_logger("07g_pyramiding_holdout")
HOLDOUT_LABEL_DIR = HOLDOUT_DIR / "labeled"
RUN_DIR = MODEL_DIR / "runs" / "ic32_regime_v1"
INF_CFG_PATH = MODEL_DIR / "inference_config.json"
OUT_JSON = RUN_DIR / "holdout_pyramiding_diag_apr_jun26.json"
FLOW_MOM_WINDOW = 3
DYN_EXTRA = {"cvd_slope_h4_delta_entry", "ofi_h4_delta_entry", "flow_momentum_3bar"}

VARIANTS = [
    {"label": "no_pyr", "enabled": False, "max_per_coin": 1, "exit_mode": "independent"},
    {"label": "pyr2_shared_sl_first", "enabled": True, "max_per_coin": 2, "exit_mode": "shared_sl_first"},
    {"label": "pyr2_independent", "enabled": True, "max_per_coin": 2, "exit_mode": "independent"},
]


def _apply_live_config() -> dict:
    prod = ifs.load_production_defaults()
    import config as project_config

    with open(INF_CFG_PATH, encoding="utf-8") as f:
        inf = json.load(f)
    rr = inf.get("rr_gate", {})

    for mod in (project_config, btu):
        mod.CONFIDENCE_THRESHOLD_ENTRY = prod["conf_entry"]
        mod.LSTM_ADJUST_AGREE_BOOST = prod["agree_boost"]
        mod.LSTM_ADJUST_NEUTRAL_PEN = prod["neutral_pen"]
        mod.LSTM_ADJUST_OPPOSITE_PEN = prod["opposite_pen"]
        mod.LSTM_DIRECTIONAL_REVIEW_THRESHOLD = prod["dir_review_thr"]
        mod.LSTM_FLAT_REVIEW_ENABLED = prod["flat_review"]
        mod.LSTM_CONFIRMATION_ENABLED = True
        mod.REGIME_AWARE_ALIGNMENT = prod["flip"]
        mod.HMM_GATE_LSTM_ENABLED = prod["hmm_gate_lstm"]
    project_config.GUARDIAN_EXIT_THRESHOLD = 0.65
    project_config.GUARDIAN_MIN_HOLD_BARS = GUARDIAN_MIN_HOLD_BARS
    btu.SMART_ENTRY_MODE = "disabled"
    btu.MOMENTUM_DYNAMIC_THRESHOLD_ENABLED = False
    btu.TREND_DYNAMIC_THRESHOLD_ENABLED = False
    btu.LSTM_STANDALONE_ENABLED = False

    return {
        "conf_entry": prod["conf_entry"],
        "gdn_min_hold": GUARDIAN_MIN_HOLD_BARS,
        "sl_trigger_mode": str(rr.get("sl_trigger_mode", "close")),
    }


def _add_momentum_feats(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "cvd_slope_h4" in df.columns:
        df["cvd_slope_h4_delta"] = df["cvd_slope_h4"].diff(1)
    if "ofi_z_score" in df.columns:
        df["flow_momentum_3bar"] = df["ofi_z_score"].rolling(FLOW_MOM_WINDOW, min_periods=1).mean()
    else:
        df["flow_momentum_3bar"] = 0.0
    return df


def _load_guardian_cont() -> dict:
    run = MODEL_DIR / "runs" / "ic32_guardian_continuation_v1"
    with open(run / "guardian_feature_cols.json", encoding="utf-8") as f:
        feats = json.load(f)
    dyn = set(GUARDIAN_DYNAMIC_FEATURES) | DYN_EXTRA
    static = [c for c in feats if c not in dyn]
    return {
        "model": joblib.load(run / "guardian.pkl"),
        "scaler": joblib.load(run / "guardian_scaler.pkl"),
        "feats": feats,
        "static": static,
    }


def _attach_ts(trades: list, sym: str, index: pd.DatetimeIndex) -> list:
    out = []
    for t in trades:
        bi = int(t.get("bar_in", 0))
        bo = int(t.get("bar_out", bi))
        rec = dict(t)
        rec["symbol"] = sym
        rec["ts_in"] = index[bi] if bi < len(index) else None
        rec["ts_out"] = index[min(bo, len(index) - 1)] if len(index) else None
        out.append(rec)
    return out


def _pyramid_stats(trades: list) -> dict:
    if not trades:
        return {"avg_legs_per_stack": 0, "pct_stacks_multi_leg": 0, "add_on_legs": 0, "total_stacks": 0}

    by_sym = defaultdict(list)
    for t in trades:
        by_sym[t["symbol"]].append(t)

    stacks_multi = 0
    total_stacks = 0
    leg_counts = []
    add_on_legs = 0

    for ts in by_sym.values():
        ts = sorted(ts, key=lambda x: x["bar_in"])
        i = 0
        while i < len(ts):
            stack = [ts[i]]
            j = i + 1
            while j < len(ts) and ts[j]["bar_in"] < stack[-1]["bar_out"]:
                stack.append(ts[j])
                j += 1
            total_stacks += 1
            leg_counts.append(len(stack))
            if len(stack) > 1:
                stacks_multi += 1
                add_on_legs += len(stack) - 1
            i = j if j > i + 1 else i + 1

    return {
        "avg_legs_per_stack": round(float(np.mean(leg_counts)), 2) if leg_counts else 0,
        "pct_stacks_multi_leg": round(stacks_multi / total_stacks * 100, 1) if total_stacks else 0,
        "add_on_legs": add_on_legs,
        "total_stacks": total_stacks,
    }


def _dir_scorecard(trades: list, direction: str) -> dict:
    sub = [t for t in trades if t.get("direction") == direction]
    if not sub:
        return {"trades": 0, "win_rate": None, "profit_factor": None, "pnl_per_trade": None, "total_pnl": 0}
    n = len(sub)
    wins = [t for t in sub if t.get("net_pnl", 0) > 0]
    losses = [t for t in sub if t.get("net_pnl", 0) <= 0]
    gpnl = sum(t["net_pnl"] for t in wins)
    gloss = abs(sum(t["net_pnl"] for t in losses))
    pf = gpnl / gloss if gloss > 0 else None
    tpnl = sum(t.get("net_pnl", 0) for t in sub)
    return {
        "trades": n,
        "pct_of_total": round(n / len(trades) * 100, 1) if trades else 0,
        "win_rate": round(len(wins) / n * 100, 2),
        "profit_factor": round(pf, 3) if pf is not None else None,
        "total_pnl": round(tpnl, 2),
        "pnl_per_trade": round(tpnl / n, 4),
    }


def _scorecard(trades: list, variant: dict) -> dict:
    if not trades:
        return {"label": variant["label"], "trades": 0}

    df = pd.DataFrame(trades)
    df["ts_in"] = pd.to_datetime(df["ts_in"], utc=True)
    df["date_in"] = df["ts_in"].dt.date
    df["is_win"] = df["net_pnl"] > 0

    n = len(df)
    wins = df[df["is_win"]]
    losses = df[~df["is_win"]]
    gpnl = float(wins["net_pnl"].sum())
    gloss = abs(float(losses["net_pnl"].sum()))
    pf = gpnl / gloss if gloss > 0 else None
    tpnl = float(df["net_pnl"].sum())

    period_start = df["ts_in"].min()
    period_end = df["ts_in"].max()
    calendar_days = max((period_end - period_start).days + 1, 1)
    trading_days = int(df["date_in"].nunique())
    hold_months = max(calendar_days / 30.44, 0.1)

    outcome_counts: dict[str, int] = {}
    hold_bars = []
    for t in trades:
        oc = t.get("outcome", "UNKNOWN")
        outcome_counts[oc] = outcome_counts.get(oc, 0) + 1
        hold_bars.append(t.get("bar_out", 0) - t.get("bar_in", 0))
    gd_n = sum(v for k, v in outcome_counts.items() if "GUARDIAN" in k)
    sl_n = outcome_counts.get("LOSS", 0)

    daily = df.groupby("date_in").agg(
        n_trades=("net_pnl", "count"),
        pnl=("net_pnl", "sum"),
    ).reset_index()

    pyr = _pyramid_stats(trades)
    long_sc = _dir_scorecard(trades, "LONG")
    short_sc = _dir_scorecard(trades, "SHORT")

    return {
        "label": variant["label"],
        "config": variant,
        "period": {
            "start": str(period_start.date()),
            "end": str(period_end.date()),
            "calendar_days": calendar_days,
            "trading_days": trading_days,
            "holdout_months": round(hold_months, 2),
        },
        "trades": n,
        "trades_per_day_active": round(n / trading_days, 2) if trading_days else 0,
        "trades_per_day_calendar": round(n / calendar_days, 2),
        "trades_per_month": round(n / hold_months, 1),
        "win_rate": round(len(wins) / n * 100, 2),
        "total_pnl": round(tpnl, 2),
        "pnl_per_trade": round(tpnl / n, 4),
        "pnl_per_trading_day": round(tpnl / trading_days, 2) if trading_days else 0,
        "profit_factor": round(pf, 3) if pf is not None else None,
        "sl_rate_pct": round(sl_n / n * 100, 2),
        "guardian_exit_pct": round(gd_n / n * 100, 2),
        "avg_hold_bars": round(float(np.mean(hold_bars)), 2),
        "median_hold_bars": round(float(np.median(hold_bars)), 2),
        "long": long_sc,
        "short": short_sc,
        "short_long_ratio": round(short_sc["trades"] / max(long_sc["trades"], 1), 3),
        "daily_stats": {
            "avg_trades_per_day": round(float(daily["n_trades"].mean()), 2),
            "median_trades_per_day": round(float(daily["n_trades"].median()), 1),
            "max_trades_per_day": int(daily["n_trades"].max()),
            "positive_pnl_days": int((daily["pnl"] > 0).sum()),
            "negative_pnl_days": int((daily["pnl"] <= 0).sum()),
            "best_day_pnl": round(float(daily["pnl"].max()), 2),
            "worst_day_pnl": round(float(daily["pnl"].min()), 2),
        },
        "outcome_counts": outcome_counts,
        **pyr,
    }


def _run_holdout(sym: str, hmm_cfg: dict, live_cfg: dict, gdn: dict,
                 feat_cols: list, lstm_feats: list, lgbm, lstm, lstm_scaler,
                 variant: dict) -> list:
    p = HOLDOUT_LABEL_DIR / f"{sym}_features_v3.parquet"
    if not p.exists():
        return []
    df = pd.read_parquet(p)
    df = ensure_utc_index(df).sort_index()
    df = _add_momentum_feats(df)
    rp = HOLDOUT_LABEL_DIR / f"{sym}_regime_h1.parquet"
    if rp.exists():
        try:
            reg = pd.read_parquet(rp)
            if "hmm_regime_enc" in df.columns:
                df = df.drop(columns=["hmm_regime_enc"])
            df = df.join(reg[["hmm_regime_enc"]], how="left")
            df["hmm_regime_enc"] = df["hmm_regime_enc"].fillna(1).astype("int32")
        except Exception:
            df["hmm_regime_enc"] = 1
    else:
        df["hmm_regime_enc"] = 1

    mask = df["label"].astype(str).isin(LABEL_MAP)
    df = df[mask].copy()
    n = len(df)
    if n < 30:
        return []

    # Apply same training parity overrides seperti live inference
    df = apply_training_parity(df)

    X = np.zeros((n, len(feat_cols)), dtype=np.float64)
    for idx, col in enumerate(feat_cols):
        if col in df.columns:
            X[:, idx] = df[col].ffill().fillna(0).values.astype(np.float64)

    hmm_enc = df["hmm_regime_enc"].values.astype(np.int32)
    thr_l, thr_s = build_per_bar_thresholds(hmm_enc, hmm_cfg)
    y_pred, confidence = hierarchical_predict(
        None, lgbm, lstm, lstm_scaler, X, feat_cols, [], df,
        model_dir=RUN_DIR, lstm_feat_cols=lstm_feats,
        per_bar_thr_long=thr_l, per_bar_thr_short=thr_s,
    )
    below = (y_pred != 1) & (confidence < live_cfg["conf_entry"])
    y_pred[below] = 1

    flow = df["flow_momentum_3bar"].ffill().fillna(0).values.astype(np.float64)
    X_gd = compute_guardian_static_array(df, gdn["static"])
    rep = full_trading_report(
        y_pred=y_pred, y_actual=df["label"].map(LABEL_MAP).values.astype(np.int64),
        atr=df["atr_14_h1"].values if "atr_14_h1" in df.columns else np.ones(n),
        close=df["close"].values, high=df["high"].values, low=df["low"].values,
        h4_swing_highs=df["h4_swing_high"].values if "h4_swing_high" in df.columns else None,
        h4_swing_lows=df["h4_swing_low"].values if "h4_swing_low" in df.columns else None,
        index=df.index, modal=MODAL_PER_TRADE, leverages=LEVERAGE_SIM,
        fee_per_side=FEE_PER_SIDE, slippage=SLIPPAGE_PER_SIDE,
        min_rr=SWING_LABEL_MIN_RR, min_tp_atr=SWING_LABEL_MIN_TP, max_sl_atr=SWING_LABEL_MAX_SL,
        tp_fallback_atr=TP_SL_FALLBACK_TP, sl_fallback_atr=TP_SL_FALLBACK_SL,
        max_hold=MAX_HOLDING_BARS, symbol=sym, confidence=confidence,
        guardian_model=gdn["model"], guardian_scaler=gdn["scaler"],
        X_guardian=X_gd, guardian_exit_threshold=0.65,
        guardian_min_hold_bars=live_cfg["gdn_min_hold"],
        guardian_feat_cols=gdn["feats"], guardian_static_names=gdn["static"],
        flow_momentum_arr=flow,
        trailing_stop_enabled=TRAILING_STOP_ENABLED,
        trailing_stop_atr=TRAILING_STOP_ATR, trailing_stop_min_bars=TRAILING_STOP_MIN_BARS,
        h4_trend=df["h4_trend"].values if "h4_trend" in df.columns else None,
        vol_ratio=df["vol_ratio_20"].values if "vol_ratio_20" in df.columns else None,
        sl_trigger_mode=live_cfg["sl_trigger_mode"],
        pyramiding_enabled=variant["enabled"],
        pyramiding_max_per_coin=variant["max_per_coin"],
        pyramiding_same_dir=True,
        pyramiding_exit_mode=variant["exit_mode"],
    )
    trades = rep.get("lev5x", rep).get("trades", [])
    return _attach_ts(trades, sym, df.index)


def _print_scorecard(sc: dict):
    if not sc.get("trades"):
        print(f"  {sc.get('label', '?')}: no trades")
        return
    p = sc["period"]
    print(f"\n  [{sc['label']}]")
    print(f"  Periode         : {p['start']} - {p['end']} ({p['calendar_days']} hari kalender)")
    print(f"  Hari trading    : {p['trading_days']} hari (ada >=1 trade)")
    print(f"  Total trades    : {sc['trades']:,}")
    print(f"  Trade/hari aktif: {sc['trades_per_day_active']:.1f}")
    print(f"  Trade/hari kalen: {sc['trades_per_day_calendar']:.2f}")
    print(f"  Trade/bulan     : {sc['trades_per_month']:.0f}")
    print(f"  WR overall      : {sc['win_rate']:.1f}%")
    print(f"  WR LONG         : {sc['long']['win_rate']:.1f}%  ({sc['long']['trades']} trades, {sc['long']['pct_of_total']:.1f}%)")
    print(f"  WR SHORT        : {sc['short']['win_rate']:.1f}%  ({sc['short']['trades']} trades, {sc['short']['pct_of_total']:.1f}%)")
    print(f"  PF overall      : {sc['profit_factor']:.3f}")
    print(f"  PF LONG         : {sc['long']['profit_factor']}")
    print(f"  PF SHORT        : {sc['short']['profit_factor']}")
    print(f"  PnL total       : ${sc['total_pnl']:+.2f}")
    print(f"  PPT             : ${sc['pnl_per_trade']:+.4f}")
    print(f"  PnL/hari aktif  : ${sc['pnl_per_trading_day']:+.2f}")
    print(f"  SL rate         : {sc['sl_rate_pct']:.1f}%")
    print(f"  Guardian exit   : {sc['guardian_exit_pct']:.1f}%")
    print(f"  Multi-leg stacks: {sc['pct_stacks_multi_leg']:.1f}% ({sc['add_on_legs']} add-on legs)")
    ds = sc["daily_stats"]
    print(f"  Hari +PnL / -PnL: {ds['positive_pnl_days']} / {ds['negative_pnl_days']}")
    print(f"  Best/worst day  : ${ds['best_day_pnl']:+.2f} / ${ds['worst_day_pnl']:+.2f}")


def main():
    live_cfg = _apply_live_config()
    hmm_cfg = load_b_dir_hmm_cfg()
    gdn = _load_guardian_cont()
    with open(MODEL_DIR / "feature_cols_ic32_regime.json", encoding="utf-8") as f:
        feat_cols = json.load(f)
    with open(MODEL_DIR / "feature_cols_lstm_temporal.json", encoding="utf-8") as f:
        lstm_feats = json.load(f)[:11]
    lgbm = joblib.load(RUN_DIR / "lgbm.pkl")
    lstm = load_lstm(MODEL_DIR / "lstm_best.pt", device="cpu")
    lstm_scaler = joblib.load(MODEL_DIR / "lstm_scaler.pkl")

    results = {}
    for variant in VARIANTS:
        logger.info(f"Holdout {variant['label']}...")
        trades = []
        for sym in ALL_COINS:
            trades.extend(_run_holdout(sym, hmm_cfg, live_cfg, gdn, feat_cols, lstm_feats,
                                       lgbm, lstm, lstm_scaler, variant))
        results[variant["label"]] = _scorecard(trades, variant)

    base = results["no_pyr"]
    for label, sc in results.items():
        if label != "no_pyr" and base.get("trades"):
            sc["delta_trades_pct"] = round((sc["trades"] - base["trades"]) / base["trades"] * 100, 1)
            sc["delta_ppt"] = round(sc.get("pnl_per_trade", 0) - base.get("pnl_per_trade", 0), 4)
            sc["delta_pnl"] = round(sc.get("total_pnl", 0) - base.get("total_pnl", 0), 2)
            sc["delta_wr"] = round(sc.get("win_rate", 0) - base.get("win_rate", 0), 2)
            sc["delta_pf"] = round((sc.get("profit_factor") or 0) - (base.get("profit_factor") or 0), 3)

    out = {
        "methodology": "holdout_diagnostic_once",
        "holdout_start": str(OOS_START.date()),
        "stack": "B-dir + continuation_v1 + min_hold=4 + SL close",
        "modal": MODAL_PER_TRADE,
        "leverage": LEVERAGE_SIM,
        "created": datetime.now().isoformat(),
        "baseline": "no_pyr",
        "results": results,
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=str)

    sep = "=" * 72
    print(f"\n{sep}")
    print("  HOLDOUT PYRAMIDING DIAGNOSTIC — Apr-Jun 2026")
    print(f"{sep}")
    for v in VARIANTS:
        _print_scorecard(results[v["label"]])

    print(f"\n{sep}")
    print("  DELTA vs no_pyr (live)")
    print(f"{sep}")
    print(f"  {'variant':<22} {'trades':>7} {'dTr%':>6} {'WR':>6} {'dWR':>6} {'PF':>6} {'dPF':>6} {'PPT':>8} {'dPPT':>8} {'PnL':>9}")
    for v in VARIANTS:
        sc = results[v["label"]]
        if not sc.get("trades"):
            continue
        print(
            f"  {sc['label']:<22} {sc['trades']:>7} "
            f"{sc.get('delta_trades_pct', 0):>+5.1f}% "
            f"{sc['win_rate']:>5.1f}% {sc.get('delta_wr', 0):>+5.2f} "
            f"{sc['profit_factor']:>6.3f} {sc.get('delta_pf', 0):>+6.3f} "
            f"${sc['pnl_per_trade']:>+7.4f} {sc.get('delta_ppt', 0):>+8.4f} "
            f"${sc['total_pnl']:>+8.2f}"
        )
    print(f"\nSaved {OUT_JSON}")


if __name__ == "__main__":
    main()