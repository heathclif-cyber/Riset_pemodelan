"""
pipeline/14_eval_meta_entry_fb_v2.py
Holdout meta-label test untuk tb_widyawardhana_v2 (flatboost_v2 stack).

Period: Apr 2026 – Jun 2026 (sama dengan scorecard produksi)
Baseline: HMM T50_R55 + LSTM soft veto (opposite_pen=0.08)
Arms: primary_hmm | stack_baseline | primary_meta_* | stack_meta_*

Prerequisites:
  python pipeline/08_generate_meta_labels_fb_v2.py
  python pipeline/09_train_meta_lgbm_fb_v2.py
"""
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

from config import (
    ALL_COINS, HOLDOUT_DIR, MODEL_DIR, MODAL_PER_TRADE, LEVERAGE_SIM,
    FEE_PER_SIDE, SLIPPAGE_PER_SIDE, SWING_LABEL_MIN_RR, SWING_LABEL_MIN_TP,
    SWING_LABEL_MAX_SL, TP_SL_FALLBACK_TP, TP_SL_FALLBACK_SL, MAX_HOLDING_BARS,
    LABEL_MAP,
)
from core.evaluator import full_trading_report
from core.meta_labeling import (
    apply_lstm_soft_veto, apply_meta_mask, build_meta_row,
    hmm_entry_from_proba, pass_fail_check, profit_factor,
)
from core.utils import setup_logger, ensure_utc_index
from pipeline.backtest_utils import get_lstm_proba

logger = setup_logger("14_meta_fb_v2")

RUN_NAME = "tb_meta_fb_v2"
RUN_DIR = MODEL_DIR / "runs" / RUN_NAME
FB_RUN = "tb_lgbm_flatboost_v2"
META_THRESHOLDS = [0.40, 0.45, 0.50, 0.55, 0.60]
META_CONTEXT = [
    "hmm_regime_enc", "atr_percentile_h1", "funding_rate",
    "vol_spike_zscore", "ofi_h4_delta", "cvd_slope_h4",
]
HOLDOUT_START = "2026-04-01"
MONTHS = 2.5
LM = LABEL_MAP if isinstance(LABEL_MAP, dict) else {"SHORT": 0, "FLAT": 1, "LONG": 2}


def build_meta_proba(df, proba_fb, meta_model, meta_feats):
    n = len(df)
    out = np.full(n, np.nan, dtype=np.float64)
    for i in range(n):
        sig = int(np.argmax(proba_fb[i]))
        if sig == 1:
            continue
        row = build_meta_row(proba_fb[i], sig, df.iloc[i], META_CONTEXT)
        X = np.array([[row.get(f, 0.0) for f in meta_feats]], dtype=np.float64)
        out[i] = float(meta_model.predict_proba(X)[0, 1])
    return out


def run_variant(yp, conf, base_kw):
    rep = full_trading_report(**base_kw, y_pred=yp, confidence=conf, guardian_enabled=False)
    trades = rep.get("trades") or []
    pnl = float(rep.get("pnl_lev5x", 0))
    wins = int(rep.get("wins", 0))
    total = int(rep.get("total_trades", len(trades)))
    pf = float(rep.get("profit_factor", profit_factor(trades)))
    pnl_list = [float(t.get("net_pnl", 0)) for t in trades]
    return {
        "trades": total,
        "wins": wins,
        "wr": round(wins / max(total, 1) * 100, 2),
        "pnl": round(pnl, 2),
        "pf": round(pf, 3) if pf != float("inf") else "inf",
        "pnl_per_trade": round(pnl / max(total, 1), 3),
        "pnl_per_month": round(pnl / MONTHS, 2),
        "pnl_list": pnl_list,
    }


def main():
    meta_path = RUN_DIR / "meta_lgbm.pkl"
    feat_path = RUN_DIR / f"{RUN_NAME}_features.json"
    if not meta_path.exists():
        raise FileNotFoundError("Run 09_train_meta_lgbm_fb_v2.py first")

    meta_model = joblib.load(meta_path)
    with open(feat_path, encoding="utf-8") as f:
        meta_feats = json.load(f)

    fb_model = joblib.load(MODEL_DIR / "runs" / FB_RUN / "lgbm.pkl")
    with open(MODEL_DIR / "runs" / FB_RUN / f"{FB_RUN}_features.json", encoding="utf-8") as f:
        fb_feats = json.load(f)

    from core.models import load_lstm as _load_lstm
    lstm_model = _load_lstm(MODEL_DIR / "lstm_best.pt")
    lstm_scaler = joblib.load(MODEL_DIR / "lstm_scaler.pkl")
    with open(MODEL_DIR / "feature_cols_lstm_temporal.json", encoding="utf-8") as f:
        lstm_feats = json.load(f)

    available = sorted(
        s for s in ALL_COINS
        if (HOLDOUT_DIR / "labeled" / f"{s}_features_v3.parquet").exists()
    )

    print(f"\n{'='*76}")
    print("  META-LABEL TEST — tb_widyawardhana_v2 (flatboost_v2)")
    print(f"  Holdout: {HOLDOUT_START} – Jun 2026 | {len(available)} coins")
    print(f"  Baseline: HMM T50_R55 + LSTM soft veto | meta thr: {META_THRESHOLDS}")
    print(f"{'='*76}\n")

    def new_arm():
        return {"trades": 0, "wins": 0, "pnl": 0.0, "pnl_list": []}

    arms = {"primary_hmm": new_arm(), "stack_baseline": new_arm()}
    for thr in META_THRESHOLDS:
        arms[f"primary_meta_{thr:.2f}"] = new_arm()
        arms[f"stack_meta_{thr:.2f}"] = new_arm()

    for sym in available:
        df = ensure_utc_index(pd.read_parquet(HOLDOUT_DIR / "labeled" / f"{sym}_features_v3.parquet"))
        df = df[df.index >= HOLDOUT_START].sort_index()
        if len(df) < 50:
            continue

        rp = HOLDOUT_DIR / "labeled" / f"{sym}_regime_h1.parquet"
        hmm = np.full(len(df), 1, np.int32)
        if rp.exists():
            reg = pd.read_parquet(rp)
            if "hmm_regime_enc" in reg.columns:
                hmm = reg["hmm_regime_enc"].reindex(df.index, fill_value=1).values.astype(np.int32)

        mask = df["label"].astype(str).isin(LM)
        df = df[mask].copy()
        hmm = hmm[mask.values]
        n = len(df)

        X_fb = np.zeros((n, len(fb_feats)), dtype=np.float64)
        for i, c in enumerate(fb_feats):
            if c in df.columns:
                X_fb[:, i] = df[c].ffill().fillna(0).values.astype(np.float64)
        proba_fb = fb_model.predict_proba(X_fb)

        X_lstm = np.zeros((n, len(lstm_feats)), dtype=np.float64)
        for i, c in enumerate(lstm_feats):
            if c in df.columns:
                X_lstm[:, i] = df[c].ffill().fillna(0).values.astype(np.float64)
        proba_lstm = get_lstm_proba(lstm_model, lstm_scaler, X_lstm, n)

        yp_hmm, conf_hmm = hmm_entry_from_proba(proba_fb, hmm)
        yp_stack, conf_stack = apply_lstm_soft_veto(yp_hmm, conf_hmm, proba_lstm)
        meta_proba = build_meta_proba(df, proba_fb, meta_model, meta_feats)

        base_kw = dict(
            y_actual=np.array([LM.get(str(v), 1) for v in df["label"].values], dtype=np.int32),
            atr=df["atr_14_h1"].values,
            close=df["close"].values,
            high=df["high"].values,
            low=df["low"].values,
            h4_swing_highs=df["h4_swing_high"].values if "h4_swing_high" in df.columns else np.full(n, np.nan),
            h4_swing_lows=df["h4_swing_low"].values if "h4_swing_low" in df.columns else np.full(n, np.nan),
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
            symbol=sym,
            h4_trend=df["h4_trend"].values if "h4_trend" in df.columns else None,
            trailing_stop_enabled=False,
        )

        def accumulate(key, yp, conf):
            res = run_variant(yp, conf, base_kw)
            arms[key]["trades"] += res["trades"]
            arms[key]["wins"] += res["wins"]
            arms[key]["pnl"] += res["pnl"]
            arms[key]["pnl_list"].extend(res.get("pnl_list", []))

        accumulate("primary_hmm", yp_hmm, conf_hmm)
        accumulate("stack_baseline", yp_stack, conf_stack)
        for thr in META_THRESHOLDS:
            yp_pm, cf_pm = apply_meta_mask(yp_hmm, conf_hmm, meta_proba, thr)
            yp_sm, cf_sm = apply_meta_mask(yp_stack, conf_stack, meta_proba, thr)
            accumulate(f"primary_meta_{thr:.2f}", yp_pm, cf_pm)
            accumulate(f"stack_meta_{thr:.2f}", yp_sm, cf_sm)

        logger.info(f"[{sym}] done")

    results = {}
    for k, a in arms.items():
        t, w, p = a["trades"], a["wins"], a["pnl"]
        pf = profit_factor([{"net_pnl": x} for x in a["pnl_list"]])
        results[k] = {
            "trades": t,
            "wins": w,
            "wr": round(w / max(t, 1) * 100, 2),
            "pnl": round(p, 2),
            "pf": round(pf, 2) if pf != float("inf") else "inf",
            "pnl_per_trade": round(p / max(t, 1), 3),
            "pnl_per_month": round(p / MONTHS, 2),
        }

    baseline = results["stack_baseline"]
    pass_fail = {}
    for k, v in results.items():
        if "meta" not in k:
            continue
        b_pf = baseline["pf"] if isinstance(baseline["pf"], (int, float)) else 0.0
        v_pf = v["pf"] if isinstance(v["pf"], (int, float)) else 0.0
        pass_fail[k] = pass_fail_check(
            {**baseline, "pf": b_pf},
            {**v, "pf": v_pf},
        )

    print(f"\n{'='*76}")
    print("  SCORECARD — flatboost_v2 meta entry (Guardian OFF, entry-only)")
    print(f"{'='*76}")
    print(f"  {'Arm':<26} {'Trades':>7} {'WR':>6} {'PF':>6} {'PnL':>9} {'$/tr':>7} {'$/mo':>7}")
    print("-" * 82)
    order = ["primary_hmm", "stack_baseline"]
    for thr in META_THRESHOLDS:
        order += [f"primary_meta_{thr:.2f}", f"stack_meta_{thr:.2f}"]
    for k in order:
        v = results[k]
        mark = " <-- baseline" if k == "stack_baseline" else ""
        pf_s = f"{v['pf']:.2f}" if isinstance(v["pf"], (int, float)) else str(v["pf"])
        print(
            f"  {k:<26} {v['trades']:>7,} {v['wr']:>5.1f}% {pf_s:>6} "
            f"${v['pnl']:>+8.0f} ${v['pnl_per_trade']:>+6.3f} "
            f"${v['pnl_per_month']:>+6.0f}{mark}"
        )

    print(f"\n  PASS/FAIL vs stack_baseline:")
    for k, chk in pass_fail.items():
        status = "PASS" if chk["passed"] else "FAIL"
        v = results[k]
        pf_s = f"{v['pf']:.2f}" if isinstance(v["pf"], (int, float)) else str(v["pf"])
        print(
            f"    [{status}] {k}: PnLΔ=${chk['pnl_delta']:+.0f} "
            f"PF={pf_s} drop={chk['trade_drop_pct']:.1f}%"
        )

    passed = [k for k, c in pass_fail.items() if c["passed"]]
    verdict = "GO" if passed else "NO-GO"
    best = max((k for k in results if "meta" in k), key=lambda k: results[k]["pnl"], default=None)
    print(f"\n  Verdict: {verdict}")
    if passed:
        print(f"  Passed: {passed}")
    else:
        print(f"  Best meta by PnL: {best} (${results[best]['pnl']:+.0f})" if best else "")
    print("  Note: Guardian continuation_v1 not included — fix holdout Guardian sim separately.")

    out = {
        "model": "tb_widyawardhana_v2",
        "base_lgbm": FB_RUN,
        "holdout_start": HOLDOUT_START,
        "results": results,
        "pass_fail": pass_fail,
        "verdict": verdict,
        "passed_arms": passed,
        "evaluated_at": datetime.now().isoformat(),
    }
    out_path = RUN_DIR / "ablation_fb_v2_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\n  Saved -> {out_path}")
    print(f"{'='*76}")


if __name__ == "__main__":
    main()