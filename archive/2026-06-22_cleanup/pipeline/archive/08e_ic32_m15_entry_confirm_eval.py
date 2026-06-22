"""
Evaluasi Opsi A: H1 ic32 signal + M15 entry confirmation — PROTOKOL KETAT.

OOF (development):
  - Seluruh bar has_oof=True dari oof_predictions.parquet
  - Periode: 2020-08-26 -> 2025-10-31 (labeled training, BUKAN slice arbitrer)
  - LGBM proba = OOF (purged CV); bar non-OOF dipaksa FLAT (no entry)
  - Aturan M15 dipilih HANYA dari hasil OOF ini

Holdout (konfirmasi sekali):
  - HANYA dengan flag --confirm-holdout (default: TIDAK dijalankan)
  - Model/threshold frozen; tidak tuning di holdout

Prasyarat:
  python tools/fetch_m15_research.py --mode oof --force
  python tools/fetch_m15_research.py --mode holdout

Jalankan:
  python pipeline/08e_ic32_m15_entry_confirm_eval.py
  python pipeline/08e_ic32_m15_entry_confirm_eval.py --confirm-holdout
"""
from __future__ import annotations

import argparse
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
from core.m15_entry_confirm import CONFIRM_RULES, apply_m15_confirmation, load_m15
from core.models import load_lstm
from core.utils import setup_logger, ensure_utc_index
from config import (
    ALL_COINS, LABEL_DIR, MODEL_DIR, HOLDOUT_DIR, REPORT_DIR,
    LABEL_MAP, TRAIN_CUTOFF_DATE, OOS_START, OOS_END,
    MODAL_PER_TRADE, LEVERAGE_SIM, FEE_PER_SIDE, SLIPPAGE_PER_SIDE,
    MAX_HOLDING_BARS, GUARDIAN_DYNAMIC_FEATURES,
    SWING_LABEL_MIN_RR, SWING_LABEL_MIN_TP, SWING_LABEL_MAX_SL,
    TP_SL_FALLBACK_TP, TP_SL_FALLBACK_SL, TP_SL_TRIGGER_MODE,
    TRAILING_STOP_ENABLED, TRAILING_STOP_ATR, TRAILING_STOP_MIN_BARS,
)

logger = setup_logger("08e_m15_entry")
RUN_DIR = MODEL_DIR / "runs" / "ic32_regime_v1"
OOF_PATH = RUN_DIR / "oof_predictions.parquet"
INF_CFG_PATH = MODEL_DIR / "inference_config.json"
OUT_JSON = REPORT_DIR / "experiments" / "ic32_m15_entry_confirm_eval.json"
OUT_MD = REPORT_DIR / "experiments" / "ic32_m15_entry_confirm_eval.md"
HOLDOUT_GUARD = RUN_DIR / ".m15_entry_holdout_confirmed"

# Genuine OOF bounds (from oof_predictions has_oof + labeled training end)
OOF_START = pd.Timestamp("2020-08-26 11:00:00", tz="UTC")
LABELED_END = pd.Timestamp("2025-10-31 23:00:00", tz="UTC")
HOLDOUT_LABEL_DIR = HOLDOUT_DIR / "labeled"
MAX_WAIT_M15 = 4

FLAT = 1


def _oof_bounds() -> tuple[pd.Timestamp, pd.Timestamp]:
    if not OOF_PATH.exists():
        return OOF_START, LABELED_END
    oof = pd.read_parquet(OOF_PATH, columns=["has_oof"])
    h = oof[oof["has_oof"] == True]
    if h.empty:
        return OOF_START, LABELED_END
    idx = pd.to_datetime(h.index, utc=True)
    return idx.min(), min(idx.max(), LABELED_END)


def _apply_live_config():
    with open(INF_CFG_PATH, encoding="utf-8") as f:
        cfg = json.load(f)
    cascade = cfg.get("cascade", {})
    guardian = cfg.get("guardian", {})

    import config as project_config
    project_config.LGBM_THRESHOLD_LONG = float(cascade.get("lgbm_threshold_long", 0.69))
    project_config.LGBM_THRESHOLD_SHORT = float(cascade.get("lgbm_threshold_short", 0.59))
    project_config.CONFIDENCE_THRESHOLD_ENTRY = float(cascade.get("confidence_threshold_entry", 0.59))
    project_config.LSTM_ADJUST_AGREE_BOOST = float(cascade.get("lstm_adjust_agree_boost", 0.05))
    project_config.LSTM_ADJUST_NEUTRAL_PEN = float(cascade.get("lstm_adjust_neutral_pen", 0.0))
    project_config.LSTM_ADJUST_OPPOSITE_PEN = float(cascade.get("lstm_adjust_opposite_pen", 0.65))
    project_config.LSTM_DIRECTIONAL_REVIEW_THRESHOLD = float(
        cascade.get("lstm_directional_review_threshold", 0.35)
    )
    project_config.LSTM_FLAT_REVIEW_ENABLED = bool(cascade.get("lstm_flat_review_enabled", True))
    project_config.LSTM_CONFIRMATION_ENABLED = bool(cascade.get("lstm_confirmation_enabled", True))
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
    btu.SMART_ENTRY_MODE = "disabled"
    btu.MOMENTUM_DYNAMIC_THRESHOLD_ENABLED = False
    btu.TREND_DYNAMIC_THRESHOLD_ENABLED = False
    btu.LSTM_STANDALONE_ENABLED = False

    return {
        "thr_long": project_config.LGBM_THRESHOLD_LONG,
        "thr_short": project_config.LGBM_THRESHOLD_SHORT,
        "conf_entry": project_config.CONFIDENCE_THRESHOLD_ENTRY,
        "gdn_exit": project_config.GUARDIAN_EXIT_THRESHOLD,
        "gdn_min_hold": project_config.GUARDIAN_MIN_HOLD_BARS,
    }


def _load_models():
    lgbm = joblib.load(RUN_DIR / "lgbm.pkl")
    lstm = load_lstm(MODEL_DIR / "lstm_best.pt", device="cpu")
    lstm_scaler = joblib.load(MODEL_DIR / "lstm_scaler.pkl")
    with open(MODEL_DIR / "feature_cols_ic32_regime.json", encoding="utf-8") as f:
        feat_cols = json.load(f)
    with open(MODEL_DIR / "feature_cols_lstm_temporal.json", encoding="utf-8") as f:
        lstm_feat_cols = json.load(f)
    guardian = joblib.load(MODEL_DIR / "guardian_clean_v2.pkl")
    guardian_scaler = joblib.load(MODEL_DIR / "guardian_clean_v2_scaler.pkl")
    with open(MODEL_DIR / "guardian_clean_v2_feature_cols.json", encoding="utf-8") as f:
        guardian_feat_cols = json.load(f)
    g_static = [c for c in guardian_feat_cols if c not in GUARDIAN_DYNAMIC_FEATURES]
    return {
        "lgbm": lgbm, "lstm": lstm, "lstm_scaler": lstm_scaler,
        "feat_cols": feat_cols, "lstm_feat_cols": lstm_feat_cols,
        "guardian": guardian, "guardian_scaler": guardian_scaler,
        "g_static": g_static,
    }


def _load_coin_df(sym: str, label_dir: Path, t_start, t_end, regime_dir: Path | None = None):
    path = label_dir / f"{sym}_features_v3.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    df = ensure_utc_index(df).sort_index()
    df = df[(df.index >= t_start) & (df.index <= t_end)]
    rp = (regime_dir or label_dir) / f"{sym}_regime_h1.parquet"
    if rp.exists():
        try:
            reg = pd.read_parquet(rp)
            for col in ["hmm_regime_enc", "hmm_regime"]:
                if col in df.columns:
                    df = df.drop(columns=[col])
            cols = [c for c in ["hmm_regime_enc", "hmm_regime"] if c in reg.columns]
            df = df.join(reg[cols], how="left")
            df["hmm_regime_enc"] = df["hmm_regime_enc"].fillna(1).astype("int32")
        except Exception:
            pass
    if "hmm_regime_enc" not in df.columns:
        df["hmm_regime_enc"] = 1
    mask = df["label"].astype(str).isin(LABEL_MAP)
    df = df[mask].copy()
    return df if len(df) >= 50 else None


def _predict_stack(df, models, oof_proba=None):
    n = len(df)
    feat_cols = models["feat_cols"]
    X = np.zeros((n, len(feat_cols)), dtype=np.float64)
    for idx, col in enumerate(feat_cols):
        if col in df.columns:
            X[:, idx] = df[col].ffill().fillna(0).values.astype(np.float64)
    y_pred, confidence = hierarchical_predict(
        None, models["lgbm"], models["lstm"], models["lstm_scaler"],
        X, feat_cols, [], df, model_dir=RUN_DIR,
        lstm_feat_cols=models["lstm_feat_cols"],
        lgbm_proba=oof_proba,
    )
    return y_pred, confidence, X


def _apply_oof_mask(y_pred, confidence, has_oof: np.ndarray, conf_thr: float):
    """Genuine OOF: hanya bar has_oof boleh entry; sisanya FLAT."""
    y_out = y_pred.copy()
    conf_out = confidence.copy()
    below = has_oof & (y_out != FLAT) & (conf_out < conf_thr)
    y_out[below] = FLAT
    y_out[~has_oof] = FLAT
    conf_out[~has_oof] = 0.0
    return y_out, conf_out


def _run_report(df, y_pred, confidence, models, live_cfg, entry_price_override=None):
    n = len(df)
    y = df["label"].map(LABEL_MAP).values.astype(np.int64)
    atr = df["atr_14_h1"].values if "atr_14_h1" in df.columns else np.ones(n)
    close = df["close"].values.astype(float)
    high = df["high"].values.astype(float)
    low = df["low"].values.astype(float)
    h4_sh = df["h4_swing_high"].values if "h4_swing_high" in df.columns else None
    h4_sl = df["h4_swing_low"].values if "h4_swing_low" in df.columns else None
    h4t = df["h4_trend"].values if "h4_trend" in df.columns else None
    volr = df["vol_ratio_20"].values if "vol_ratio_20" in df.columns else None
    X_guardian = compute_guardian_static_array(df, models["g_static"])

    return full_trading_report(
        y_pred=y_pred,
        y_actual=y,
        atr=atr,
        close=close,
        high=high,
        low=low,
        h4_swing_highs=h4_sh,
        h4_swing_lows=h4_sl,
        index=df.index,
        modal=MODAL_PER_TRADE,
        leverages=LEVERAGE_SIM,
        fee_per_side=FEE_PER_SIDE,
        slippage=SLIPPAGE_PER_SIDE,
        min_rr=SWING_LABEL_MIN_RR,
        min_tp_atr=SWING_LABEL_MIN_TP,
        max_sl_atr=SWING_LABEL_MAX_SL,
        tp_fallback_atr=TP_SL_FALLBACK_TP,
        sl_fallback_atr=TP_SL_FALLBACK_SL,
        max_hold=MAX_HOLDING_BARS,
        symbol=df.attrs.get("coin", ""),
        confidence=confidence,
        guardian_model=models["guardian"],
        guardian_scaler=models["guardian_scaler"],
        X_guardian=X_guardian,
        guardian_exit_threshold=live_cfg["gdn_exit"],
        guardian_min_hold_bars=live_cfg["gdn_min_hold"],
        trailing_stop_enabled=TRAILING_STOP_ENABLED,
        trailing_stop_atr=TRAILING_STOP_ATR,
        trailing_stop_min_bars=TRAILING_STOP_MIN_BARS,
        h4_trend=h4t,
        vol_ratio=volr,
        sl_trigger_mode=TP_SL_TRIGGER_MODE,
        entry_price_override=entry_price_override,
    )


def _scorecard(report: dict) -> dict:
    trades = report.get("trades") or []
    n = report.get("total_trades", len(trades))
    if n == 0:
        return {"trades": 0, "wr": 0.0, "pf": 0.0, "pnl": 0.0, "ppt": 0.0}
    lev = int(LEVERAGE_SIM[0]) if LEVERAGE_SIM else 5
    pnl = float(report.get(f"pnl_lev{lev}x", 0.0))
    if not pnl and trades:
        pnl = sum(float(t.get("net_pnl", t.get("pnl", 0))) for t in trades)
    n_wins = int(report.get("wins", 0))
    if not n_wins and trades:
        n_wins = sum(1 for t in trades if float(t.get("net_pnl", t.get("pnl", 0))) > 0)
    wr = n_wins / n * 100 if n else 0.0
    pf = float(report.get("profit_factor", 0.0))
    return {
        "trades": n,
        "n_wins": n_wins,
        "wr": round(wr, 1),
        "pf": round(pf, 2),
        "pnl": round(pnl, 2),
        "ppt": round(pnl / n, 3) if n else 0.0,
    }


def eval_coin_oof(sym: str, df, models, live_cfg, rules: tuple[str, ...], oof_sym: pd.DataFrame) -> dict:
    m15 = load_m15(sym, df.index.min(), df.index.max() + pd.Timedelta(days=2), holdout=False)
    if m15.empty:
        logger.warning(f"[{sym}] M15 OOF kosong — skip (jalankan fetch --mode oof)")
        return {}

    merged = df.join(oof_sym[["p0", "p1", "p2", "has_oof"]], how="left")
    has_oof = merged["has_oof"].fillna(False).values.astype(bool)
    if has_oof.sum() < 20:
        return {}

    oof_proba = np.column_stack([
        merged["p0"].values, merged["p1"].values, merged["p2"].values,
    ]).astype(np.float64)
    y_pred, confidence, _ = _predict_stack(df, models, oof_proba=oof_proba)
    y_pred, confidence = _apply_oof_mask(y_pred, confidence, has_oof, live_cfg["conf_entry"])

    out: dict[str, dict] = {}
    h1_close = df["close"].values.astype(float)
    for rule in rules:
        y_c, price_ov, conf_c, stats = apply_m15_confirmation(
            df.index, y_pred, confidence, h1_close, m15, rule, max_wait=MAX_WAIT_M15,
        )
        rep = _run_report(df, y_c, conf_c, models, live_cfg, entry_price_override=price_ov)
        sc = _scorecard(rep)
        sc.update(stats)
        sc["oof_bars"] = int(has_oof.sum())
        out[rule] = sc
    return out


def eval_coin_holdout(sym: str, df, models, live_cfg, rules: tuple[str, ...]) -> dict:
    m15 = load_m15(sym, df.index.min(), df.index.max() + pd.Timedelta(days=2), holdout=True)
    if m15.empty:
        logger.warning(f"[{sym}] M15 holdout kosong — skip")
        return {}

    y_pred, confidence, _ = _predict_stack(df, models)
    below = (y_pred != FLAT) & (confidence < live_cfg["conf_entry"])
    y_pred[below] = FLAT

    out: dict[str, dict] = {}
    h1_close = df["close"].values.astype(float)
    for rule in rules:
        y_c, price_ov, conf_c, stats = apply_m15_confirmation(
            df.index, y_pred, confidence, h1_close, m15, rule, max_wait=MAX_WAIT_M15,
        )
        rep = _run_report(df, y_c, conf_c, models, live_cfg, entry_price_override=price_ov)
        sc = _scorecard(rep)
        sc.update(stats)
        out[rule] = sc
    return out


def aggregate(coin_results: dict) -> dict:
    rules = CONFIRM_RULES
    agg = {r: {"trades": 0, "wins": 0, "pnl": 0.0,
               "n_signal": 0, "n_confirmed": 0, "n_skipped": 0} for r in rules}
    for _sym, res in coin_results.items():
        for rule, sc in res.items():
            if not sc:
                continue
            agg[rule]["n_signal"] += sc.get("n_signal", 0)
            agg[rule]["n_confirmed"] += sc.get("n_confirmed", 0)
            agg[rule]["n_skipped"] += sc.get("n_skipped", 0)
            if sc.get("trades", 0) == 0:
                continue
            agg[rule]["trades"] += sc["trades"]
            agg[rule]["wins"] += sc.get("n_wins", 0)
            agg[rule]["pnl"] += sc["pnl"]
    final = {}
    base_ppt = None
    for rule in rules:
        a = agg[rule]
        t = a["trades"]
        if t == 0:
            final[rule] = {"trades": 0, "wr": 0, "pf": 0, "pnl": 0, "ppt": 0,
                           "confirm_rate_pct": 0, "delta_ppt_vs_h1": 0}
            continue
        wr = a["wins"] / t * 100
        ppt = a["pnl"] / t
        if rule == "h1_immediate":
            base_ppt = ppt
        final[rule] = {
            "trades": t,
            "wr": round(wr, 1),
            "pnl": round(a["pnl"], 2),
            "ppt": round(ppt, 3),
            "confirm_rate_pct": round(a["n_confirmed"] / a["n_signal"] * 100, 1) if a["n_signal"] else 0,
            "n_signal": a["n_signal"],
            "n_skipped": a["n_skipped"],
            "n_wins": a["wins"],
        }
    if base_ppt is not None:
        for rule in rules:
            if final[rule].get("trades"):
                final[rule]["delta_ppt_vs_h1"] = round(final[rule]["ppt"] - base_ppt, 3)
    return final


def write_md(results: dict):
    proto = results.get("protocol", {})
    lines = [
        "# ic32 M15 Entry Confirmation — Genuine OOF Eval",
        "",
        f"**Protocol**: {proto.get('name', 'genuine_oof_full')}",
        f"**OOF range**: {proto.get('oof_start')} -> {proto.get('oof_end')}",
        f"**LGBM**: OOF proba (has_oof only) | **Holdout**: flag --confirm-holdout only",
        "",
    ]
    for period in ("oof", "holdout"):
        if period not in results:
            continue
        lines.append(f"## {period.upper()}")
        lines.append("")
        lines.append("| Rule | Trades | WR% | PnL | PPT | vs H1 | Confirm% |")
        lines.append("|------|-------:|----:|----:|----:|------:|---------:|")
        for rule, sc in results[period].items():
            lines.append(
                f"| {rule} | {sc.get('trades', 0)} | {sc.get('wr', 0)} | "
                f"${sc.get('pnl', 0)} | ${sc.get('ppt', 0)} | "
                f"{sc.get('delta_ppt_vs_h1', 0):+.3f} | {sc.get('confirm_rate_pct', 0)}% |"
            )
        lines.append("")
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--confirm-holdout",
        action="store_true",
        help="Buka amplop holdout SEKALI (setelah OOF freeze). Default: OOF saja.",
    )
    args = parser.parse_args()

    if not OOF_PATH.exists():
        raise FileNotFoundError(f"OOF wajib ada: {OOF_PATH}")

    oof_start, oof_end = _oof_bounds()
    live_cfg = _apply_live_config()
    models = _load_models()
    rules = CONFIRM_RULES
    results: dict = {
        "protocol": {
            "name": "genuine_oof_full",
            "oof_start": str(oof_start),
            "oof_end": str(oof_end),
            "labeled_end": str(LABELED_END),
            "train_cutoff": str(TRAIN_CUTOFF_DATE),
            "lgbm_signal": "oof_proba_has_oof_only",
            "holdout_policy": "confirm_holdout_flag_only",
        },
    }

    logger.info("=== GENUINE OOF %s -> %s (has_oof only) ===", oof_start, oof_end)
    oof_all = pd.read_parquet(OOF_PATH)
    oof_coins = {}
    for sym in ALL_COINS:
        df = _load_coin_df(sym, LABEL_DIR, oof_start, oof_end)
        if df is None:
            continue
        oof_sym = oof_all[oof_all["coin"] == sym]
        if oof_sym.empty:
            continue
        df.attrs["coin"] = sym
        res = eval_coin_oof(sym, df, models, live_cfg, rules, oof_sym)
        if res:
            oof_coins[sym] = res
    results["oof"] = aggregate(oof_coins)
    results["oof_coins"] = len(oof_coins)
    results["oof_bars_total"] = int(oof_all["has_oof"].sum())

    if args.confirm_holdout:
        if HOLDOUT_GUARD.exists():
            logger.warning("Holdout sudah pernah dikonfirmasi — lanjut overwrite (manual guard)")
        logger.info("=== HOLDOUT CONFIRM %s -> %s ===", OOS_START, OOS_END)
        ho_coins = {}
        for sym in ALL_COINS:
            df = _load_coin_df(sym, HOLDOUT_LABEL_DIR, OOS_START, OOS_END, regime_dir=HOLDOUT_LABEL_DIR)
            if df is None:
                continue
            df.attrs["coin"] = sym
            res = eval_coin_holdout(sym, df, models, live_cfg, rules)
            if res:
                ho_coins[sym] = res
        results["holdout"] = aggregate(ho_coins)
        results["holdout_coins"] = len(ho_coins)
        HOLDOUT_GUARD.write_text(json.dumps({"confirmed_at": str(pd.Timestamp.now(tz="UTC"))}), encoding="utf-8")

    results["config"] = {
        "max_wait_m15": MAX_WAIT_M15,
        "rules": list(rules),
        "sl_trigger_mode": TP_SL_TRIGGER_MODE,
        "live_thr": live_cfg,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(results, indent=2), encoding="utf-8")
    write_md(results)
    print(json.dumps({k: v for k, v in results.items() if k not in ("oof_coins", "holdout_coins", "oof_bars_total")}, indent=2))
    print(f"\nSaved {OUT_JSON}\nSaved {OUT_MD}")


if __name__ == "__main__":
    main()