"""
pipeline/08b_oof_ic32_swing_complement_eval.py

OOF full-stack eval: ic32 swing complement LSTM vs baseline lstm_best.pt.

Variants (all genuine OOF, ic32 fixed thr 0.69/0.59):
  1. baseline_hard_consensus  — lstm_best.pt + hard_consensus (production)
  2. complement_hard_consensus — ic32_lstm_swing_complement_v1 + hard_consensus
  3. complement_momentum_boost — complement LSTM + conditional_momentum (best OOF params)
  4. complement_boost_only    — complement LSTM + boost_only on vol_spike bars

Prerequisite:
  python pipeline/05u_train_lstm_ic32_swing_complement_v1.py --all

Usage:
  python pipeline/08b_oof_ic32_swing_complement_eval.py
"""
import argparse
import json
import sys
import warnings
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import pipeline.backtest_utils as btu
from pipeline.backtest_utils import hierarchical_predict, compute_guardian_static_array
from core.evaluator import full_trading_report, simulate_trades_swing
from core.models import load_lstm
from core.cascade_utils import (
    apply_conditional_momentum_fusion_pre,
    apply_lstm_boost_only_pre,
    SHORT, FLAT, LONG,
)
from core.utils import setup_logger, ensure_utc_index
from config import (
    ALL_COINS, LABEL_DIR, MODEL_DIR, LABEL_MAP, TRAIN_CUTOFF_DATE,
    MODAL_PER_TRADE, LEVERAGE_SIM, FEE_PER_SIDE, SLIPPAGE_PER_SIDE,
    MAX_HOLDING_BARS, GUARDIAN_DYNAMIC_FEATURES,
    SWING_LABEL_MIN_RR, SWING_LABEL_MIN_TP, SWING_LABEL_MAX_SL,
    TP_SL_FALLBACK_TP, TP_SL_FALLBACK_SL,
    TRAILING_STOP_ENABLED, TRAILING_STOP_ATR, TRAILING_STOP_MIN_BARS,
)

logger = setup_logger("08b_ic32_swing_complement")
RUN_DIR = MODEL_DIR / "runs" / "ic32_regime_v1"
DEFAULT_COMPLEMENT_RUN = "ic32_lstm_swing_complement_v2"
LSTM_PROD_N_FEAT = 11
OOF_PATH = RUN_DIR / "oof_predictions.parquet"
INF_CFG_PATH = MODEL_DIR / "inference_config.json"

IC32_THR_LONG = 0.69
IC32_THR_SHORT = 0.59
CONF_ENTRY = 0.59

MOMENTUM_CFG = {
    "bull_thr": 0.38,
    "bear_thr": 0.50,
    "boost": 0.10,
    "opposite_pen": 0.14,
    "near_miss_gap": 0.05,
    "vol_thr": 2.0,
    "enable_boost": True,
    "enable_penalty": True,
    "proportional": True,
}

BOOST_ONLY_CFG = {
    "agree_boost": 0.08,
    "vol_thr": 2.0,
}


def _apply_live_config():
    with open(INF_CFG_PATH, encoding="utf-8") as f:
        cfg = json.load(f)
    cascade = cfg.get("cascade", {})
    guardian = cfg.get("guardian", {})

    import config as project_config
    project_config.LGBM_THRESHOLD_LONG = IC32_THR_LONG
    project_config.LGBM_THRESHOLD_SHORT = IC32_THR_SHORT
    project_config.CONFIDENCE_THRESHOLD_ENTRY = CONF_ENTRY
    project_config.LSTM_ADJUST_AGREE_BOOST = float(cascade.get("lstm_adjust_agree_boost", 0.05))
    project_config.LSTM_ADJUST_NEUTRAL_PEN = float(cascade.get("lstm_adjust_neutral_pen", 0.0))
    project_config.LSTM_ADJUST_OPPOSITE_PEN = float(cascade.get("lstm_adjust_opposite_pen", 0.65))
    project_config.LSTM_DIRECTIONAL_REVIEW_THRESHOLD = float(
        cascade.get("lstm_directional_review_threshold", 0.35)
    )
    project_config.LSTM_FLAT_REVIEW_ENABLED = bool(cascade.get("lstm_flat_review_enabled", True))
    project_config.LSTM_CONFIRMATION_ENABLED = bool(cascade.get("lstm_confirmation_enabled", True))
    project_config.REGIME_AWARE_ALIGNMENT = bool(cfg.get("regime_alignment", {}).get("enabled", True))
    project_config.GUARDIAN_EXIT_THRESHOLD = float(guardian.get("exit_threshold", 0.65))
    project_config.GUARDIAN_MIN_HOLD_BARS = int(guardian.get("min_hold_bars", 2))

    btu.LGBM_THRESHOLD_LONG = project_config.LGBM_THRESHOLD_LONG
    btu.LGBM_THRESHOLD_SHORT = project_config.LGBM_THRESHOLD_SHORT
    btu.CONFIDENCE_THRESHOLD_ENTRY = project_config.CONFIDENCE_THRESHOLD_ENTRY
    btu.LSTM_ADJUST_AGREE_BOOST = project_config.LSTM_ADJUST_AGREE_BOOST
    btu.LSTM_ADJUST_NEUTRAL_PEN = project_config.LSTM_ADJUST_NEUTRAL_PEN
    btu.LSTM_ADJUST_OPPOSITE_PEN = project_config.LSTM_ADJUST_OPPOSITE_PEN
    btu.LSTM_DIRECTIONAL_REVIEW_THRESHOLD = project_config.LSTM_DIRECTIONAL_REVIEW_THRESHOLD
    btu.LSTM_FLAT_REVIEW_ENABLED = project_config.LSTM_FLAT_REVIEW_ENABLED
    btu.LSTM_CONFIRMATION_ENABLED = project_config.LSTM_CONFIRMATION_ENABLED
    btu.REGIME_AWARE_ALIGNMENT = project_config.REGIME_AWARE_ALIGNMENT
    btu.SMART_ENTRY_MODE = "disabled"
    btu.MOMENTUM_DYNAMIC_THRESHOLD_ENABLED = False
    btu.TREND_DYNAMIC_THRESHOLD_ENABLED = False
    btu.LSTM_STANDALONE_ENABLED = False


def _load_guardian():
    for model_name, feat_name, scaler_name in (
        ("guardian_clean_v2.pkl", "guardian_clean_v2_feature_cols.json", "guardian_clean_v2_scaler.pkl"),
        ("guardian_best.pkl", "guardian_feature_cols.json", "guardian_scaler.pkl"),
    ):
        mp = MODEL_DIR / model_name
        fp = MODEL_DIR / feat_name
        sp = MODEL_DIR / scaler_name
        if mp.exists() and fp.exists() and sp.exists():
            with open(fp, encoding="utf-8") as f:
                feat_cols = json.load(f)
            return joblib.load(mp), joblib.load(sp), feat_cols, model_name
    raise FileNotFoundError("Guardian model files not found")


def _apply_ic32_thr(p0, p2):
    n = len(p0)
    tl = np.full(n, IC32_THR_LONG, dtype=np.float32)
    ts = np.full(n, IC32_THR_SHORT, dtype=np.float32)
    long_mask = p2 >= tl
    short_mask = (p0 >= ts) & ~long_mask
    y = np.ones(n, dtype=np.int32)
    y[long_mask] = LONG
    y[short_mask] = SHORT
    conf = np.where(long_mask, p2, np.where(short_mask, p0, 0.0)).astype(np.float32)
    return y, conf, tl, ts


def _aggregate_trades(all_trades: list) -> dict:
    if not all_trades:
        return {"total_trades": 0}
    n_total = len(all_trades)
    n_wins = sum(1 for t in all_trades if t.get("net_pnl", 0) > 0)
    total_pnl = sum(t.get("net_pnl", 0) for t in all_trades)
    wins_pnl = [t["net_pnl"] for t in all_trades if t.get("net_pnl", 0) > 0]
    losses_pnl = [t["net_pnl"] for t in all_trades if t.get("net_pnl", 0) <= 0]
    gross_profit = sum(wins_pnl)
    gross_loss = abs(sum(losses_pnl))
    pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")
    sl_hits = sum(
        1 for t in all_trades
        if "SL" in str(t.get("outcome", "")) or t.get("exit_reason") == "sl"
    )
    gd_exits = sum(1 for t in all_trades if "GUARDIAN" in str(t.get("outcome", "")))
    long_trades = [t for t in all_trades if t.get("direction") == "LONG"]
    short_trades = [t for t in all_trades if t.get("direction") == "SHORT"]
    return {
        "total_trades": n_total,
        "win_rate": round(n_wins / n_total * 100, 2),
        "total_pnl": round(total_pnl, 2),
        "pnl_per_trade": round(total_pnl / n_total, 4),
        "profit_factor": round(pf, 3),
        "sl_rate_pct": round(sl_hits / n_total * 100, 2),
        "guardian_exit_pct": round(gd_exits / n_total * 100, 2),
        "long_trades": len(long_trades),
        "short_trades": len(short_trades),
    }


def backtest_hard_consensus(
    sym: str, oof_all: pd.DataFrame, feat_cols: list,
    lstm_model, lstm_scaler, lstm_feat_cols: list,
    guardian_model, guardian_scaler, g_static: list,
) -> dict | None:
    path = LABEL_DIR / f"{sym}_features_v3.parquet"
    if not path.exists():
        return None

    df = pd.read_parquet(path)
    df = ensure_utc_index(df).sort_index()
    df = df[df.index < TRAIN_CUTOFF_DATE]
    mask = df["label"].astype(str).isin(LABEL_MAP)
    df = df[mask].copy()
    n = len(df)
    if n < 50:
        return None

    oof_sym = oof_all[oof_all["coin"] == sym]
    if oof_sym.empty:
        return None

    merged = df.join(oof_sym[["p0", "p1", "p2", "has_oof"]], how="left")
    has_oof = merged["has_oof"].fillna(False).values.astype(bool)
    if has_oof.sum() < 30:
        return None

    oof_proba = np.column_stack([
        merged["p0"].values, merged["p1"].values, merged["p2"].values,
    ]).astype(np.float64)

    X = np.zeros((n, len(feat_cols)), dtype=np.float64)
    for idx, col in enumerate(feat_cols):
        if col in df.columns:
            X[:, idx] = df[col].ffill().fillna(0).values.astype(np.float64)

    y_pred, confidence = hierarchical_predict(
        None, None, lstm_model, lstm_scaler,
        X, feat_cols, [], df,
        model_dir=RUN_DIR,
        lstm_feat_cols=lstm_feat_cols,
        lgbm_proba=oof_proba,
    )

    below = has_oof & (y_pred != FLAT) & (confidence < CONF_ENTRY)
    y_pred[below] = FLAT
    y_pred[~has_oof] = FLAT
    confidence[~has_oof] = 0.0

    return _run_full_report(sym, df, y_pred, confidence, guardian_model, guardian_scaler, g_static)


def _run_full_report(sym, df, y_pred, confidence, guardian_model, guardian_scaler, g_static):
    y = df["label"].map(LABEL_MAP).values.astype(np.int64)
    atr = df["atr_14_h1"].values if "atr_14_h1" in df.columns else np.ones(len(df))
    h4_sh = df["h4_swing_high"].values if "h4_swing_high" in df.columns else None
    h4_sl = df["h4_swing_low"].values if "h4_swing_low" in df.columns else None
    h4t = df["h4_trend"].values if "h4_trend" in df.columns else None
    volr = df["vol_ratio_20"].values if "vol_ratio_20" in df.columns else None
    X_guardian = compute_guardian_static_array(df, g_static)

    return full_trading_report(
        y_pred=y_pred, y_actual=y, atr=atr,
        close=df["close"].values, high=df["high"].values, low=df["low"].values,
        h4_swing_highs=h4_sh, h4_swing_lows=h4_sl, index=df.index,
        modal=MODAL_PER_TRADE, leverages=LEVERAGE_SIM,
        fee_per_side=FEE_PER_SIDE, slippage=SLIPPAGE_PER_SIDE,
        min_rr=SWING_LABEL_MIN_RR, min_tp_atr=SWING_LABEL_MIN_TP,
        max_sl_atr=SWING_LABEL_MAX_SL,
        tp_fallback_atr=TP_SL_FALLBACK_TP, sl_fallback_atr=TP_SL_FALLBACK_SL,
        max_hold=MAX_HOLDING_BARS, symbol=sym, confidence=confidence,
        guardian_model=guardian_model, guardian_scaler=guardian_scaler,
        X_guardian=X_guardian, guardian_exit_threshold=0.65, guardian_min_hold_bars=2,
        trailing_stop_enabled=TRAILING_STOP_ENABLED,
        trailing_stop_atr=TRAILING_STOP_ATR, trailing_stop_min_bars=TRAILING_STOP_MIN_BARS,
        h4_trend=h4t, vol_ratio=volr,
    )


def preload_fusion_coins(oof_all: pd.DataFrame, lstm_oof: pd.DataFrame, g_static: list) -> list:
    coins = []
    for sym in ALL_COINS:
        path = LABEL_DIR / f"{sym}_features_v3.parquet"
        if not path.exists():
            continue
        df = pd.read_parquet(path)
        df = ensure_utc_index(df).sort_index()
        df = df[df.index < TRAIN_CUTOFF_DATE]
        df = df[df["label"].astype(str).isin(LABEL_MAP)]
        if df.empty:
            continue

        sym_lgbm = oof_all[(oof_all["coin"] == sym) & (oof_all["has_oof"] == True)][["p0", "p2"]]
        proba = sym_lgbm.reindex(df.index)
        has_oof = proba["p0"].notna()
        if has_oof.sum() < 30:
            continue

        sym_lstm = lstm_oof[lstm_oof["coin"] == sym][["p0", "p1", "p2", "has_oof", "vol_spike"]]
        lstm_aligned = sym_lstm.reindex(df.index[has_oof])
        lstm_p = lstm_aligned[["p0", "p1", "p2"]].values.astype(np.float32)
        lstm_valid = lstm_aligned["has_oof"].fillna(False).values.astype(bool)
        vol_spike = (
            df["vol_spike_zscore"].reindex(df.index[has_oof]).fillna(-99).values.astype(np.float32)
            if "vol_spike_zscore" in df.columns
            else lstm_aligned["vol_spike"].fillna(-99).values.astype(np.float32)
        )

        df_oof = df[has_oof].copy()
        n = len(df_oof)
        X_grd = np.zeros((n, len(g_static)), dtype=np.float64)
        for idx, col in enumerate(g_static):
            if col in df_oof.columns:
                X_grd[:, idx] = df_oof[col].ffill().fillna(0).values.astype(np.float64)

        coins.append({
            "sym": sym,
            "p0": proba["p0"][has_oof].values.astype(np.float32),
            "p2": proba["p2"][has_oof].values.astype(np.float32),
            "lstm_p": lstm_p,
            "lstm_valid": lstm_valid,
            "vol_spike": vol_spike,
            "close": df_oof["close"].values.astype(np.float64),
            "high": df_oof["high"].values.astype(np.float64),
            "low": df_oof["low"].values.astype(np.float64),
            "atr": df_oof["atr_14_h1"].values.astype(np.float64),
            "h4_sh": df_oof["h4_swing_high"].values.astype(np.float64)
            if "h4_swing_high" in df_oof.columns else np.full(n, np.nan),
            "h4_sl": df_oof["h4_swing_low"].values.astype(np.float64)
            if "h4_swing_low" in df_oof.columns else np.full(n, np.nan),
            "h4t": df_oof["h4_trend"].values.astype(np.float64)
            if "h4_trend" in df_oof.columns else None,
            "volr": df_oof["vol_ratio_20"].values.astype(np.float64)
            if "vol_ratio_20" in df_oof.columns else None,
            "X_grd": X_grd,
            "index": df_oof.index,
        })
    return coins


def eval_fusion_variant(coins: list, fusion_mode: str,
                        guardian_model, guardian_scaler, g_params: dict) -> list:
    all_trades = []
    common = dict(
        modal=MODAL_PER_TRADE, leverage=LEVERAGE_SIM[0],
        fee_per_side=FEE_PER_SIDE, slippage=SLIPPAGE_PER_SIDE,
        max_hold=MAX_HOLDING_BARS,
        min_rr=SWING_LABEL_MIN_RR, min_tp_atr=SWING_LABEL_MIN_TP,
        max_sl_atr=SWING_LABEL_MAX_SL,
        tp_fallback_atr=TP_SL_FALLBACK_TP, sl_fallback_atr=TP_SL_FALLBACK_SL,
    )

    for c in coins:
        p0, p2 = c["p0"].copy(), c["p2"].copy()
        _, _, tl, ts = _apply_ic32_thr(p0, p2)

        if fusion_mode == "conditional_momentum":
            p0, p2 = apply_conditional_momentum_fusion_pre(
                p0, p2, c["lstm_p"], tl, ts, c["vol_spike"],
                vol_thr=MOMENTUM_CFG["vol_thr"],
                bull_thr=MOMENTUM_CFG["bull_thr"],
                bear_thr=MOMENTUM_CFG["bear_thr"],
                near_miss_gap=MOMENTUM_CFG["near_miss_gap"],
                boost=MOMENTUM_CFG["boost"],
                opposite_pen=MOMENTUM_CFG["opposite_pen"],
                enable_boost=MOMENTUM_CFG["enable_boost"],
                enable_penalty=MOMENTUM_CFG["enable_penalty"],
                lstm_valid=c["lstm_valid"],
                proportional=MOMENTUM_CFG["proportional"],
            )
        elif fusion_mode == "boost_only":
            active = c["lstm_valid"] & (c["vol_spike"] >= BOOST_ONLY_CFG["vol_thr"])
            p0, p2 = apply_lstm_boost_only_pre(
                p0, p2, c["lstm_p"],
                agree_boost=BOOST_ONLY_CFG["agree_boost"],
                active_mask=active,
            )

        y_pred, conf, _, _ = _apply_ic32_thr(p0, p2)
        below = (y_pred != FLAT) & (conf < CONF_ENTRY)
        y_pred[below] = FLAT

        rep = simulate_trades_swing(
            y_pred=y_pred, close=c["close"], high=c["high"], low=c["low"],
            atr=c["atr"], h4_swing_highs=c["h4_sh"], h4_swing_lows=c["h4_sl"],
            confidence=conf,
            guardian_enabled=True,
            guardian_model=guardian_model, guardian_scaler=guardian_scaler,
            X_guardian=c["X_grd"],
            guardian_exit_threshold=g_params["exit_threshold"],
            guardian_min_hold_bars=g_params["min_hold_bars"],
            h4_trend=c["h4t"], vol_ratio=c["volr"],
            **common,
        )
        for t in rep.get("trades", []):
            t2 = dict(t)
            t2["symbol"] = c["sym"]
            all_trades.append(t2)
    return all_trades


def _prod_lstm_feat_cols() -> list[str]:
    with open(MODEL_DIR / "feature_cols_lstm_temporal.json", encoding="utf-8") as f:
        return json.load(f)[:LSTM_PROD_N_FEAT]


def run_eval(complement_run: str) -> dict:
    complement_dir = MODEL_DIR / "runs" / complement_run
    if not OOF_PATH.exists():
        raise FileNotFoundError(f"{OOF_PATH} missing")
    if not (complement_dir / "oof_lstm_predictions.parquet").exists():
        raise FileNotFoundError(
            f"{complement_dir}/oof_lstm_predictions.parquet missing -- train {complement_run} first"
        )

    _apply_live_config()
    oof_all = pd.read_parquet(OOF_PATH)
    if not isinstance(oof_all.index, pd.DatetimeIndex):
        oof_all.index = pd.to_datetime(oof_all.index, utc=True)

    complement_oof = pd.read_parquet(complement_dir / "oof_lstm_predictions.parquet")
    if not isinstance(complement_oof.index, pd.DatetimeIndex):
        complement_oof.index = pd.to_datetime(complement_oof.index, utc=True)
    complement_oof = complement_oof.reset_index().rename(columns={"index": "ts"})

    with open(MODEL_DIR / "feature_cols_ic32_regime.json", encoding="utf-8") as f:
        feat_cols = json.load(f)

    guardian_model, guardian_scaler, guardian_feat_cols, gdn_name = _load_guardian()
    g_static = [c for c in guardian_feat_cols if c not in set(GUARDIAN_DYNAMIC_FEATURES)]
    g_params = {"exit_threshold": 0.65, "min_hold_bars": 2}

    sep = "=" * 80
    print(f"\n{sep}")
    print("  ic32 SWING COMPLEMENT LSTM -- OOF EVAL")
    print(f"  Complement run: {complement_run}")
    print(f"  LGBM: ic32_regime_v1 | thr {IC32_THR_LONG}/{IC32_THR_SHORT} | conf {CONF_ENTRY}")
    print(f"  Guardian: {gdn_name}")
    print(f"{sep}\n")

    results = {}

    # --- Variant 1: baseline lstm_best.pt ---
    lstm_feat_cols = _prod_lstm_feat_cols()
    baseline_lstm = load_lstm(MODEL_DIR / "lstm_best.pt", device="cpu")
    baseline_scaler = joblib.load(MODEL_DIR / "lstm_scaler.pkl")

    baseline_trades = []
    for sym in ALL_COINS:
        r = backtest_hard_consensus(
            sym, oof_all, feat_cols,
            baseline_lstm, baseline_scaler, lstm_feat_cols,
            guardian_model, guardian_scaler, g_static,
        )
        if r:
            baseline_trades.extend(r.get("trades", []))
    results["baseline_hard_consensus"] = _aggregate_trades(baseline_trades)
    logger.info(f"  baseline: {results['baseline_hard_consensus']['total_trades']} trades "
                f"WR={results['baseline_hard_consensus']['win_rate']}% "
                f"PF={results['baseline_hard_consensus']['profit_factor']}")

    # --- Variant 2: complement LSTM + hard_consensus ---
    feat_path = complement_dir / f"{complement_run}_features.json"
    with open(feat_path, encoding="utf-8") as f:
        comp_feat_cols = json.load(f)
    comp_lstm = load_lstm(complement_dir / "lstm_momentum.pt", device="cpu")
    comp_scaler = joblib.load(complement_dir / "lstm_momentum_scaler.pkl")

    complement_trades = []
    for sym in ALL_COINS:
        r = backtest_hard_consensus(
            sym, oof_all, feat_cols,
            comp_lstm, comp_scaler, comp_feat_cols,
            guardian_model, guardian_scaler, g_static,
        )
        if r:
            complement_trades.extend(r.get("trades", []))
    results["complement_hard_consensus"] = _aggregate_trades(complement_trades)
    logger.info(f"  complement+hard_consensus: {results['complement_hard_consensus']['total_trades']} trades "
                f"WR={results['complement_hard_consensus']['win_rate']}% "
                f"PF={results['complement_hard_consensus']['profit_factor']}")

    # --- Variants 3-4: fusion modes with complement OOF (sparse LSTM) ---
    fusion_coins = preload_fusion_coins(oof_all, complement_oof, g_static)

    for mode in ("conditional_momentum", "boost_only"):
        trades = eval_fusion_variant(fusion_coins, mode, guardian_model, guardian_scaler, g_params)
        key = f"complement_{mode}"
        results[key] = _aggregate_trades(trades)
        logger.info(f"  {key}: {results[key]['total_trades']} trades "
                    f"WR={results[key]['win_rate']}% PF={results[key]['profit_factor']}")

    # Load complement CV meta
    meta_path = complement_dir / f"{complement_run}_meta.json"
    complement_meta = {}
    if meta_path.exists():
        with open(meta_path, encoding="utf-8") as f:
            complement_meta = json.load(f)

    baseline_ref = {}
    ref_path = RUN_DIR / "oof_full_stack_scorecard.json"
    if ref_path.exists():
        with open(ref_path, encoding="utf-8") as f:
            baseline_ref = json.load(f).get("aggregate", {})

    print(f"\n{sep}")
    print("  OOF SCORECARD COMPARISON")
    print(f"{sep}")
    hdr = f"  {'Variant':<32} {'Trades':>8} {'WR%':>7} {'PF':>6} {'PnL':>10} {'ppt':>8}"
    print(hdr)
    print(f"  {'-'*72}")
    for key, agg in results.items():
        print(f"  {key:<32} {agg['total_trades']:>8,} {agg['win_rate']:>7.1f} "
              f"{agg['profit_factor']:>6.2f} ${agg['total_pnl']:>+9.2f} "
              f"${agg['pnl_per_trade']:>+7.4f}")
    if baseline_ref:
        print(f"\n  Archived baseline (08_oof): {baseline_ref.get('total_trades')} trades "
              f"WR={baseline_ref.get('win_rate')}% PF={baseline_ref.get('profit_factor')}")

    out = {
        "meta": {
            "eval": "ic32_swing_complement_oof",
            "complement_run": complement_run,
            "lstm_n_feat": len(comp_feat_cols),
            "period": f"2020-01-01 to {TRAIN_CUTOFF_DATE.date()}",
            "guardian": gdn_name,
            "ic32_thr": [IC32_THR_LONG, IC32_THR_SHORT],
            "conf_entry": CONF_ENTRY,
            "momentum_cfg": MOMENTUM_CFG,
            "boost_only_cfg": BOOST_ONLY_CFG,
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        },
        "complement_cv": {
            "mean_f1_macro": complement_meta.get("mean_f1_macro"),
            "complement_asymmetric_oof": complement_meta.get("complement_asymmetric_oof"),
        },
        "variants": results,
        "archived_baseline": baseline_ref,
    }
    out_path = complement_dir / "oof_stack_eval.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\n  Saved -> {out_path}")
    print(f"{sep}\n")
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--complement-run",
        default=DEFAULT_COMPLEMENT_RUN,
        help="Run folder under models/runs/ (default: ic32_lstm_swing_complement_v2)",
    )
    args = parser.parse_args()
    run_eval(args.complement_run)


if __name__ == "__main__":
    main()