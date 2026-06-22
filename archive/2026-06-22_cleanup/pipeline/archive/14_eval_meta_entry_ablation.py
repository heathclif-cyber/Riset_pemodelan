"""
pipeline/14_eval_meta_entry_ablation.py
Holdout ablation: meta entry gate vs ic32 hard_consensus cascade.

Arms (spec: pipeline/meta_label_spec.json):
  1. primary_only     — ic32 LGBM production thresholds, no LSTM, no meta
  2. primary_meta     — ic32 LGBM + meta take/skip, no LSTM
  3. cascade_baseline — ic32 + LSTM hard_consensus + Guardian (production)
  4. cascade_meta     — cascade entry + meta take/skip + Guardian

Pass/fail vs cascade_baseline on PnL/PF with max 30% trade drop rule.

Prerequisites:
  python pipeline/08_generate_ic32_oof_trades.py
  python pipeline/09_train_meta_lgbm_ic32.py
"""
import json
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from config import (
    ALL_COINS, HOLDOUT_DIR, MODEL_DIR, MODAL_PER_TRADE, LEVERAGE_SIM,
    FEE_PER_SIDE, SLIPPAGE_PER_SIDE, SWING_LABEL_MIN_RR, SWING_LABEL_MIN_TP,
    SWING_LABEL_MAX_SL, TP_SL_FALLBACK_TP, TP_SL_FALLBACK_SL, MAX_HOLDING_BARS,
    LGBM_THRESHOLD_LONG, LGBM_THRESHOLD_SHORT,
)
from core.evaluator import full_trading_report
from core.meta_labeling import (
    build_meta_row, load_spec, pass_fail_check, profit_factor,
)
from core.models import TradingLSTM
from core.utils import setup_logger, ensure_utc_index

logger = setup_logger("14_meta_ablation")

PROD = Path("D:/Apps-Dev/swint_tradev2/models")
RUN_NAME = "ic32_meta_v1"
RUN_DIR = MODEL_DIR / "runs" / RUN_NAME
SPEC = load_spec()
META_THRESHOLDS = SPEC["meta"]["threshold_sweep"]
META_CONTEXT = [f for f in SPEC["meta"]["features"]
                if f not in ("p_short", "p_flat", "p_long", "confidence", "direction")]

AGREE_BOOST = 0.05
NEUTRAL_PEN = 0.0
OPPOSITE_PEN = 0.65
NO_VETO_THR = 0.50
DIR_REVIEW_THR = 0.35
LSTM_OVERRIDE_THR = 0.70
SEQ_LEN = 32
LM = {"SHORT": 0, "FLAT": 1, "LONG": 2}


def apply_hard_consensus(lgbm_proba, lstm_proba_list):
    n = len(lgbm_proba)
    yp = np.ones(n, dtype=np.int32)
    conf = np.zeros(n, dtype=np.float64)

    for i in range(n):
        lp = lgbm_proba[i].copy().astype(np.float64)
        lstm = lstm_proba_list[i]

        if lp[2] >= LGBM_THRESHOLD_LONG:
            sig = 2
        elif lp[0] >= LGBM_THRESHOLD_SHORT:
            sig = 0
        else:
            best_dir = max(lp[2], lp[0])
            if lstm is not None and best_dir >= DIR_REVIEW_THR:
                li = int(np.argmax(lstm))
                lc = float(lstm[li])
                if li != 1 and lc >= LSTM_OVERRIDE_THR:
                    sig = li
                    lp = lstm.copy().astype(np.float64)
                else:
                    continue
            else:
                continue

        if lstm is None:
            yp[i] = sig
            conf[i] = float(lp[sig])
            continue

        li = int(np.argmax(lstm))
        proba = lp.copy()
        if li == sig:
            adj = AGREE_BOOST
        elif li == 1:
            adj = -NEUTRAL_PEN
        else:
            if proba[sig] > NO_VETO_THR:
                other_idxs = [j for j in range(3) if j != sig]
                o = max(proba[j] for j in other_idxs)
                f_ = sum(proba[j] for j in other_idxs) - o
                tot = o + f_
                msp = max(0.0, (proba[sig] - o) * tot / (2 * o + f_) - 0.01) if tot > 0 and o < proba[sig] else 0.0
                adj = -min(OPPOSITE_PEN, msp)
            else:
                adj = -OPPOSITE_PEN

        new_conf = float(np.clip(proba[sig] + adj, 0.0, 1.0))
        thr = LGBM_THRESHOLD_LONG if sig == 2 else LGBM_THRESHOLD_SHORT
        if new_conf < thr:
            continue
        yp[i] = sig
        conf[i] = new_conf

    return yp, conf


def primary_only_signals(lgbm_proba):
    n = len(lgbm_proba)
    yp = np.ones(n, dtype=np.int32)
    conf = np.zeros(n, dtype=np.float64)
    for i in range(n):
        lp = lgbm_proba[i]
        if lp[2] >= LGBM_THRESHOLD_LONG and lp[2] > lp[0]:
            yp[i] = 2
            conf[i] = float(lp[2])
        elif lp[0] >= LGBM_THRESHOLD_SHORT and lp[0] > lp[2]:
            yp[i] = 0
            conf[i] = float(lp[0])
    return yp, conf


def build_meta_proba_series(df, lgbm_proba, meta_model, meta_feats):
    n = len(df)
    proba = np.full(n, np.nan, dtype=np.float64)
    for i in range(n):
        sig = int(np.argmax(lgbm_proba[i]))
        if sig == 1:
            continue
        row = build_meta_row(lgbm_proba[i], sig, df.iloc[i], META_CONTEXT)
        X = np.array([[row.get(f, 0.0) for f in meta_feats]], dtype=np.float64)
        proba[i] = float(meta_model.predict_proba(X)[0, 1])
    return proba


def apply_meta_filter(yp, conf, meta_proba, threshold):
    yp_f = yp.copy()
    conf_f = conf.copy()
    for i in range(len(yp_f)):
        if yp_f[i] == 1:
            continue
        if np.isnan(meta_proba[i]) or meta_proba[i] < threshold:
            yp_f[i] = 1
            conf_f[i] = 0.0
    return yp_f, conf_f


def run_variant(yp, conf, base_kwargs, gd_model, gd_scaler, X_gd, use_guardian):
    kwargs = dict(base_kwargs)
    if use_guardian:
        rep = full_trading_report(
            y_pred=yp, confidence=conf,
            guardian_model=gd_model, guardian_scaler=gd_scaler,
            X_guardian=X_gd, guardian_exit_threshold=0.65,
            guardian_min_hold_bars=2, guardian_enabled=True,
            **kwargs,
        )
    else:
        rep = full_trading_report(
            y_pred=yp, confidence=conf,
            guardian_enabled=False,
            **kwargs,
        )
    trades = rep.get("trades") or []
    pnl = float(rep.get("pnl_lev5x", rep.get("total_pnl", 0)))
    wins = int(rep.get("wins", 0))
    total = int(rep.get("total_trades", len(trades)))
    pnl_list = [float(t.get("net_pnl", 0)) for t in trades]
    pf = float(rep.get("profit_factor", profit_factor(trades)))
    return {
        "trades": total,
        "wins": wins,
        "wr": round(wins / max(total, 1) * 100, 2),
        "pnl": round(pnl, 2),
        "pf": round(pf, 3) if pf != float("inf") else "inf",
        "pnl_per_trade": round(pnl / max(total, 1), 3),
        "pnl_list": pnl_list,
    }


def main():
    meta_path = RUN_DIR / "meta_lgbm.pkl"
    feat_path = RUN_DIR / f"{RUN_NAME}_features.json"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"Meta model not found. Run 09_train_meta_lgbm_ic32.py first: {meta_path}"
        )

    meta_model = joblib.load(meta_path)
    with open(feat_path, encoding="utf-8") as f:
        meta_feats = json.load(f)

    ic32_model = joblib.load(MODEL_DIR / "runs" / "ic32_regime_v1" / "lgbm.pkl")
    ic32_feats = list(ic32_model.feature_name_)

    gd_model = joblib.load(PROD / "guardian_best.pkl")
    gd_scaler = joblib.load(PROD / "guardian_scaler.pkl")
    with open(PROD / "guardian_feature_cols.json", encoding="utf-8") as f:
        gd_all_feats = json.load(f)
    dynamic = {"bars_held_norm", "current_pnl_pct", "current_pnl_atr",
               "max_favorable_pnl_pct", "drawdown_from_peak_pct",
               "direction", "entry_price_ratio"}
    static_feats = [f for f in gd_all_feats if f not in dynamic]

    lstm_scaler = joblib.load(PROD / "lstm_scaler.pkl")
    with open(PROD / "feature_cols_lstm_temporal.json", encoding="utf-8") as f:
        lstm_feats = json.load(f)
    state = torch.load(PROD / "lstm_best.pt", map_location="cpu", weights_only=False)
    W_ih = state["lstm.cells.0.W_ih"]
    hidden = W_ih.shape[0] // 4
    n_inp = W_ih.shape[1]
    n_layers = sum(1 for k in state if k.startswith("lstm.cells") and k.endswith("W_ih"))
    lstm_model = TradingLSTM(
        n_features=n_inp, hidden_size=hidden,
        num_layers=n_layers, num_classes=3, dropout=0.0,
    )
    lstm_model.load_state_dict(state)
    lstm_model.eval()

    available = [
        s for s in ALL_COINS
        if (HOLDOUT_DIR / "labeled" / f"{s}_features_v3.parquet").exists()
    ]

    print(f"\n{'='*76}")
    print("  HOLDOUT ABLATION — Meta Entry vs Cascade Baseline")
    print(f"  Meta model: {RUN_NAME} | thresholds: {META_THRESHOLDS}")
    print(f"  Baseline  : ic32 + LSTM hard_consensus (Guardian OFF — entry-only ablation)")
    print(f"  Coins     : {len(available)} | Nov 2025 – Apr 2026")
    print(f"{'='*76}\n")

    def new_arm():
        return {"trades": 0, "wins": 0, "pnl": 0.0, "pnl_list": []}

    # Entry ablation — Guardian OFF (holdout Guardian sim currently broken: 0% WR)
    arms = {
        "primary_only": new_arm(),
        "cascade_lstm": new_arm(),
    }
    meta_arm_keys = {}
    for thr in META_THRESHOLDS:
        key_p = f"primary_meta_{thr:.2f}"
        key_c = f"cascade_meta_{thr:.2f}"
        arms[key_p] = new_arm()
        arms[key_c] = new_arm()
        meta_arm_keys[thr] = (key_p, key_c)

    for sym in available:
        df = pd.read_parquet(HOLDOUT_DIR / "labeled" / f"{sym}_features_v3.parquet")
        df = ensure_utc_index(df).sort_index()

        rp = HOLDOUT_DIR / "labeled" / f"{sym}_regime_h1.parquet"
        hmm = np.full(len(df), 1, np.int32)
        if rp.exists():
            reg = pd.read_parquet(rp)
            if "hmm_regime_enc" in reg.columns:
                hmm = reg["hmm_regime_enc"].reindex(df.index, fill_value=1).values.astype(np.int32)

        mask = df["label"].isin(LM)
        df = df[mask].copy()
        hmm = hmm[mask.values]
        n = len(df)

        close = df["close"].values
        high = df["high"].values
        low = df["low"].values
        atr = df["atr_14_h1"].values
        h4_sh = df["h4_swing_high"].values if "h4_swing_high" in df.columns else np.full(n, np.nan)
        h4_sl = df["h4_swing_low"].values if "h4_swing_low" in df.columns else np.full(n, np.nan)
        h4_tr = df["h4_trend"].values if "h4_trend" in df.columns else None

        X_ic = np.zeros((n, len(ic32_feats)), dtype=np.float64)
        for idx, c in enumerate(ic32_feats):
            if c in df.columns:
                X_ic[:, idx] = df[c].ffill().fillna(0).values.astype(np.float64)
            elif c == "hmm_regime_enc":
                X_ic[:, idx] = hmm.astype(np.float64)
        lgbm_proba = ic32_model.predict_proba(X_ic)

        X_lstm = np.zeros((n, len(lstm_feats)), dtype=np.float32)
        for idx, c in enumerate(lstm_feats):
            if c in df.columns:
                X_lstm[:, idx] = df[c].ffill().fillna(0).values.astype(np.float32)
        X_lstm_sc = lstm_scaler.transform(X_lstm)
        lstm_proba_list = [None] * n
        with torch.no_grad():
            for i in range(SEQ_LEN - 1, n):
                seq = torch.tensor(
                    X_lstm_sc[i - SEQ_LEN + 1: i + 1], dtype=torch.float32,
                ).unsqueeze(0)
                prob = torch.softmax(lstm_model(seq), dim=-1).squeeze(0).numpy()
                lstm_proba_list[i] = prob

        X_gd = np.zeros((n, len(static_feats)), dtype=np.float64)
        for idx, c in enumerate(static_feats):
            if c in df.columns:
                X_gd[:, idx] = df[c].ffill().fillna(0).values.astype(np.float64)
            elif c == "hmm_regime_enc":
                X_gd[:, idx] = hmm.astype(np.float64)

        base_kwargs = dict(
            y_actual=df["label"].map(LM).values.astype(np.int32),
            atr=atr, close=close, high=high, low=low,
            h4_swing_highs=h4_sh, h4_swing_lows=h4_sl,
            index=df.index, modal=MODAL_PER_TRADE, leverages=LEVERAGE_SIM,
            fee_per_side=FEE_PER_SIDE, slippage=SLIPPAGE_PER_SIDE,
            min_rr=SWING_LABEL_MIN_RR, min_tp_atr=SWING_LABEL_MIN_TP,
            max_sl_atr=SWING_LABEL_MAX_SL,
            tp_fallback_atr=TP_SL_FALLBACK_TP, sl_fallback_atr=TP_SL_FALLBACK_SL,
            max_hold=MAX_HOLDING_BARS, symbol=sym,
            trailing_stop_enabled=False, h4_trend=h4_tr,
        )

        yp_primary, conf_primary = primary_only_signals(lgbm_proba)
        yp_cascade, conf_cascade = apply_hard_consensus(lgbm_proba, lstm_proba_list)
        meta_proba = build_meta_proba_series(df, lgbm_proba, meta_model, meta_feats)

        def accumulate(arm_key, yp, conf, guardian):
            res = run_variant(yp, conf, base_kwargs, gd_model, gd_scaler, X_gd, guardian)
            arms[arm_key]["trades"] += res["trades"]
            arms[arm_key]["wins"] += res["wins"]
            arms[arm_key]["pnl"] += res["pnl"]
            arms[arm_key]["pnl_list"].extend(res.get("pnl_list", []))
            return res

        accumulate("primary_only", yp_primary, conf_primary, guardian=False)
        accumulate("cascade_lstm", yp_cascade, conf_cascade, guardian=False)

        for thr in META_THRESHOLDS:
            key_p, key_c = meta_arm_keys[thr]
            yp_pm, conf_pm = apply_meta_filter(yp_primary, conf_primary, meta_proba, thr)
            yp_cm, conf_cm = apply_meta_filter(yp_cascade, conf_cascade, meta_proba, thr)
            accumulate(key_p, yp_pm, conf_pm, guardian=False)
            accumulate(key_c, yp_cm, conf_cm, guardian=False)

        logger.info(f"[{sym}] done")

    def finalize(arm_key):
        a = arms[arm_key]
        t = a["trades"]
        w = a["wins"]
        p = a["pnl"]
        pf = profit_factor([{"net_pnl": x} for x in a["pnl_list"]])
        return {
            "trades": t,
            "wins": w,
            "wr": round(w / max(t, 1) * 100, 2),
            "pnl": round(p, 2),
            "pnl_per_trade": round(p / max(t, 1), 3),
            "pnl_per_month": round(p / 5, 2),
            "pf": round(pf, 3) if pf != float("inf") else "inf",
        }

    results = {k: finalize(k) for k in arms}
    baseline = results["cascade_lstm"]

    pass_fail = {}
    for thr in META_THRESHOLDS:
        key_p = f"primary_meta_{thr:.2f}"
        key_c = f"cascade_meta_{thr:.2f}"
        pass_fail[key_p] = pass_fail_check(
            baseline, results[key_p],
            max_trade_drop_pct=SPEC["pass_fail"]["max_trade_drop_pct"],
            min_pf_delta=SPEC["pass_fail"]["min_pf_delta_if_heavy_filter"],
        )
        pass_fail[key_c] = pass_fail_check(
            baseline, results[key_c],
            max_trade_drop_pct=SPEC["pass_fail"]["max_trade_drop_pct"],
            min_pf_delta=SPEC["pass_fail"]["min_pf_delta_if_heavy_filter"],
        )

    best_meta = None
    best_pnl = -1e18
    for k, v in results.items():
        if "meta" in k and v["pnl"] > best_pnl:
            best_pnl = v["pnl"]
            best_meta = k

    any_pass = any(v["passed"] for v in pass_fail.values())

    print(f"\n{'='*76}")
    print("  SCORECARD — Holdout Ablation (5 months)")
    print(f"{'='*76}")
    print(f"  {'Arm':<28} {'Trades':>7} {'WR':>7} {'PnL':>10} {'PF':>6} {'$/tr':>7}")
    print("-" * 76)
    order = ["primary_only", "cascade_lstm"]
    for thr in META_THRESHOLDS:
        order.extend([f"primary_meta_{thr:.2f}", f"cascade_meta_{thr:.2f}"])
    for k in order:
        v = results[k]
        marker = " <-- baseline" if k == "cascade_lstm" else ""
        pf_s = f"{v['pf']:.2f}" if v["pf"] != float("inf") else "inf"
        print(
            f"  {k:<28} {v['trades']:>7,} {v['wr']:>6.1f}% "
            f"${v['pnl']:>+8.0f} {pf_s:>6} ${v['pnl_per_trade']:>+6.3f}{marker}"
        )

    print(f"\n  PASS/FAIL vs cascade_lstm (entry baseline):")
    for k, chk in pass_fail.items():
        status = "PASS" if chk["passed"] else "FAIL"
        print(f"    [{status}] {k}: drop={chk['trade_drop_pct']:.1f}% "
              f"PnLΔ=${chk['pnl_delta']:+.0f} PFΔ={chk['pf_delta']:+.3f}")

    verdict = "NO-GO" if not any_pass else "CONDITIONAL-GO"
    print(f"\n  Verdict: {verdict}")
    if any_pass:
        passed_arms = [k for k, v in pass_fail.items() if v["passed"]]
        print(f"  Passed arms: {passed_arms}")
        print("  Next: pick simplest winner, re-run with Guardian once holdout exit sim fixed")
    else:
        print(f"  Best meta arm by PnL: {best_meta} (${best_pnl:+.0f})")
        print("  Meta does not beat cascade — keep hard_consensus, do not deploy meta gate")

    out = {
        "spec_version": SPEC["version"],
        "baseline_arm": "cascade_lstm",
        "results": results,
        "pass_fail": pass_fail,
        "verdict": verdict,
        "any_pass": any_pass,
        "best_meta_arm": best_meta,
        "note": (
            "Entry-only ablation: Guardian disabled because holdout Guardian sim "
            "currently yields 0% WR (see holdout_full_stack.json). "
            "Compare relative entry quality before stacking exit layers."
        ),
    }
    out_path = RUN_DIR / "ablation_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\n  Saved -> {out_path}")
    print(f"{'='*76}")


if __name__ == "__main__":
    main()