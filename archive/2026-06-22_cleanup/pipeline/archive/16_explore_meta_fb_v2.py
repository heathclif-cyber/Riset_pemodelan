"""
pipeline/16_explore_meta_fb_v2.py
Eksplorasi lanjutan meta-labeling flatboost_v2 setelah Gate #1 FAIL di holdout.

Varian:
  A) full        — fitur saat ini (p_short/p_long/conf + context)
  B) context_only — tanpa proba LGBM (cari sinyal orthogonal)
  C) orthogonal  — margin/entropy/gap + context (bukan raw proba)
  D) soft_mult   — multiplier confidence, bukan hard gate (arm terbaik dari A/B/C)

Usage:
  python pipeline/16_explore_meta_fb_v2.py
"""
import json
import sys
import warnings
from datetime import datetime
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from config import (
    ALL_COINS, HOLDOUT_DIR, MODEL_DIR, LABEL_MAP,
    MODAL_PER_TRADE, LEVERAGE_SIM, FEE_PER_SIDE, SLIPPAGE_PER_SIDE,
    SWING_LABEL_MIN_RR, SWING_LABEL_MIN_TP, SWING_LABEL_MAX_SL,
    TP_SL_FALLBACK_TP, TP_SL_FALLBACK_SL, MAX_HOLDING_BARS,
)
from core.evaluator import full_trading_report
from core.meta_labeling import (
    apply_lstm_soft_veto, apply_meta_mask, build_meta_row,
    hmm_entry_from_proba, profit_factor,
)
from core.utils import ensure_utc_index
from pipeline.backtest_utils import get_lstm_proba

FB_RUN = "tb_lgbm_flatboost_v2"
EXP_RUN = "tb_meta_fb_v2_explore"
OOF_PATH = ROOT / "data" / "meta_labels" / "fb_v2_oof_trades.parquet"
HOLDOUT_START = "2026-04-01"
MONTHS = 2.5
LM = LABEL_MAP if isinstance(LABEL_MAP, dict) else {"SHORT": 0, "FLAT": 1, "LONG": 2}

META_CONTEXT = [
    "hmm_regime_enc", "atr_percentile_h1", "funding_rate",
    "vol_spike_zscore", "ofi_h4_delta", "cvd_slope_h4",
]
META_THRESHOLDS = [0.45, 0.50, 0.55]
MULT_RANGE = (0.65, 1.40)
MARGINAL_IC_THR = 0.015
TSTAT_THR = 2.0

META_PARAMS = {
    "objective": "binary",
    "n_estimators": 400,
    "learning_rate": 0.03,
    "max_depth": 4,
    "num_leaves": 15,
    "min_child_samples": 50,
    "subsample": 0.8,
    "colsample_bytree": 0.7,
    "verbose": -1,
    "n_jobs": -1,
    "random_state": 42,
}

VARIANTS = {
    "full": [
        "p_short", "p_flat", "p_long", "confidence", "direction",
        *META_CONTEXT,
    ],
    "context_only": ["direction", *META_CONTEXT],
    "orthogonal": [
        "proba_margin", "proba_entropy", "conf_gap", "direction",
        *META_CONTEXT,
    ],
}


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    ps, pf, pl = out["p_short"].values, out["p_flat"].values, out["p_long"].values
    out["proba_margin"] = pl - ps
    probs = np.stack([ps, pf, pl], axis=1)
    probs = np.clip(probs, 1e-9, 1.0)
    out["proba_entropy"] = -np.sum(probs * np.log(probs), axis=1)
    winner = np.maximum(ps, pl)
    second = np.where(ps >= pl, pl, ps)
    out["conf_gap"] = winner - second
    return out


def rank_ic(x, y):
    mask = ~(np.isnan(x) | np.isnan(y))
    x, y = x[mask], y[mask]
    n = int(mask.sum())
    if n < 50:
        return {"ic": np.nan, "t": np.nan, "n": n}
    ic, _ = spearmanr(x, y)
    t = ic * np.sqrt((n - 2) / (1 - ic**2 + 1e-9))
    return {"ic": float(ic), "t": float(t), "n": n}


def marginal_ic_gs(x_new, x_base, y):
    mask = ~(np.isnan(x_new) | np.isnan(x_base) | np.isnan(y))
    xn, xb, ym = x_new[mask], x_base[mask], y[mask]
    n = int(mask.sum())
    if n < 50:
        return {"marginal_ic": np.nan, "t": np.nan, "corr_with_base": np.nan, "n": n}
    corr = float(np.corrcoef(xn, xb)[0, 1])
    xn_resid = xn - corr * xb
    ic, _ = spearmanr(xn_resid, ym)
    t = ic * np.sqrt((n - 2) / (1 - ic**2 + 1e-9))
    return {"marginal_ic": float(ic), "t": float(t), "corr_with_base": corr, "n": n}


def train_meta_variant(name: str, feats: list, oof_df: pd.DataFrame) -> tuple:
    avail = [c for c in feats if c in oof_df.columns]
    X = oof_df[avail].ffill().fillna(0)
    y = oof_df["win"].values.astype(np.int32)
    model = lgb.LGBMClassifier(**META_PARAMS)
    model.fit(X, y)
    p_all = model.predict_proba(X)[:, 1]
    auc = float(roc_auc_score(y, p_all))
    return model, avail, auc


def build_bar_proba(df, proba_fb, meta_model, meta_feats, variant: str):
    n = len(df)
    out = np.full(n, np.nan, dtype=np.float64)
    for i in range(n):
        sig = int(np.argmax(proba_fb[i]))
        if sig == 1:
            continue
        row = build_meta_row(proba_fb[i], sig, df.iloc[i], META_CONTEXT)
        if variant == "orthogonal":
            ps, pf, pl = row["p_short"], row["p_flat"], row["p_long"]
            row["proba_margin"] = pl - ps
            probs = np.clip([ps, pf, pl], 1e-9, 1.0)
            row["proba_entropy"] = float(-np.sum(probs * np.log(probs)))
            w, s = max(ps, pl), min(ps, pl) if ps != pl else ps
            row["conf_gap"] = w - s
        X = np.array([[row.get(f, 0.0) for f in meta_feats]], dtype=np.float64)
        out[i] = float(meta_model.predict_proba(X)[0, 1])
    return out


def apply_soft_multiplier(yp, conf, meta_proba, base_wr, hmm=None):
    """Scale confidence by meta p_win / base_wr, then re-gate with HMM thresholds."""
    from core.meta_labeling import hmm_thresholds
    yp_out = yp.copy()
    conf_out = conf.copy()
    for i in range(len(yp_out)):
        if yp_out[i] == 1 or np.isnan(meta_proba[i]):
            continue
        mult = float(np.clip(meta_proba[i] / base_wr, MULT_RANGE[0], MULT_RANGE[1]))
        adj = float(np.clip(conf_out[i] * mult, 0.0, 1.0))
        sig = int(yp_out[i])
        thr_l, thr_s = hmm_thresholds(int(hmm[i]) if hmm is not None else 1)
        thr = thr_l if sig == 2 else thr_s
        if adj < thr:
            yp_out[i] = 1
            conf_out[i] = 0.0
        else:
            conf_out[i] = adj
    return yp_out, conf_out


def holdout_marginal_ic(meta_model, meta_feats, variant: str, fb_model, fb_feats) -> dict:
    records = []
    for sym in ALL_COINS:
        path = HOLDOUT_DIR / "labeled" / f"{sym}_features_v3.parquet"
        if not path.exists():
            continue
        df = ensure_utc_index(pd.read_parquet(path))
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
        yp, conf_arr = hmm_entry_from_proba(proba_fb, hmm)

        base_kw = dict(
            y_pred=yp,
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
            confidence=conf_arr,
            guardian_enabled=False,
            trailing_stop_enabled=False,
            h4_trend=df["h4_trend"].values if "h4_trend" in df.columns else None,
        )
        rep = full_trading_report(**base_kw)
        trades = rep.get("trades") or []
        meta_bar = build_bar_proba(df, proba_fb, meta_model, meta_feats, variant)

        for t in trades:
            idx = t.get("bar_in")
            if idx is None or idx < 0 or idx >= n or yp[idx] == 1:
                continue
            records.append({
                "win": 1.0 if t.get("net_pnl", 0) > 0 else 0.0,
                "confidence": float(conf_arr[idx]),
                "p_meta": float(meta_bar[idx]),
            })

    if len(records) < 50:
        return {"verdict": "SKIP", "n": len(records)}
    hdf = pd.DataFrame(records)
    marg = marginal_ic_gs(
        hdf["p_meta"].values, hdf["confidence"].values, hdf["win"].values,
    )
    gate = (
        not np.isnan(marg["marginal_ic"])
        and abs(marg["marginal_ic"]) >= MARGINAL_IC_THR
        and abs(marg["t"]) >= TSTAT_THR
    )
    return {
        "n": marg["n"],
        "marginal_ic": round(marg["marginal_ic"], 4),
        "t_marginal": round(marg["t"], 2),
        "corr_meta_conf": round(marg["corr_with_base"], 4),
        "gate_marginal": gate,
        "verdict": "PASS" if gate else "FAIL",
    }


def holdout_pnl_eval(variant_models: dict, fb_model, fb_feats,
                     lstm_model, lstm_scaler, lstm_feats, base_wr: float) -> dict:
    """Evaluate primary_hmm + best hard gate + soft multiplier per variant."""
    arms = {"primary_hmm": {"trades": 0, "wins": 0, "pnl": 0.0, "pnl_list": []}}
    for vname in variant_models:
        for thr in META_THRESHOLDS:
            arms[f"{vname}_gate_{thr:.2f}"] = {"trades": 0, "wins": 0, "pnl": 0.0, "pnl_list": []}
        arms[f"{vname}_mult"] = {"trades": 0, "wins": 0, "pnl": 0.0, "pnl_list": []}

    for sym in ALL_COINS:
        path = HOLDOUT_DIR / "labeled" / f"{sym}_features_v3.parquet"
        if not path.exists():
            continue
        df = ensure_utc_index(pd.read_parquet(path))
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
            guardian_enabled=False,
        )

        def run_arm(key, yp, conf):
            rep = full_trading_report(**base_kw, y_pred=yp, confidence=conf)
            trades = rep.get("trades") or []
            arms[key]["trades"] += int(rep.get("total_trades", len(trades)))
            arms[key]["wins"] += int(rep.get("wins", 0))
            arms[key]["pnl"] += float(rep.get("pnl_lev5x", 0))
            arms[key]["pnl_list"].extend(float(t.get("net_pnl", 0)) for t in trades)

        run_arm("primary_hmm", yp_hmm, conf_hmm)

        for vname, (model, feats, _) in variant_models.items():
            meta_bar = build_bar_proba(df, proba_fb, model, feats, vname)
            for thr in META_THRESHOLDS:
                yp_g, cf_g = apply_meta_mask(yp_hmm, conf_hmm, meta_bar, thr)
                run_arm(f"{vname}_gate_{thr:.2f}", yp_g, cf_g)
            yp_m, cf_m = apply_soft_multiplier(yp_hmm, conf_hmm, meta_bar, base_wr, hmm)
            run_arm(f"{vname}_mult", yp_m, cf_m)

    results = {}
    for k, a in arms.items():
        t, w, p = a["trades"], a["wins"], a["pnl"]
        pf = profit_factor([{"net_pnl": x} for x in a["pnl_list"]])
        results[k] = {
            "trades": t,
            "wr": round(w / max(t, 1) * 100, 2),
            "pnl": round(p, 2),
            "pf": round(pf, 3) if pf != float("inf") else "inf",
            "pnl_per_month": round(p / MONTHS, 2),
        }
    return results


def main():
    if not OOF_PATH.exists():
        raise FileNotFoundError(f"Missing OOF labels: {OOF_PATH}")

    run_dir = MODEL_DIR / "runs" / EXP_RUN
    run_dir.mkdir(parents=True, exist_ok=True)

    oof_df = pd.read_parquet(OOF_PATH)
    if "timestamp" in oof_df.columns:
        oof_df = oof_df.set_index("timestamp")
    oof_df.index = pd.to_datetime(oof_df.index, utc=True)
    oof_df = add_derived_features(oof_df.sort_index())
    base_wr = float(oof_df["win"].mean())

    print(f"\n{'='*78}")
    print("  META EXPLORATION — flatboost_v2 (post Gate #1 FAIL)")
    print(f"  OOF: {len(oof_df):,} trades | base WR {base_wr*100:.1f}%")
    print(f"{'='*78}")

    # Train variants
    variant_models = {}
    train_report = {}
    for vname, feats in VARIANTS.items():
        model, avail, auc = train_meta_variant(vname, feats, oof_df)
        variant_models[vname] = (model, avail, auc)
        joblib.dump(model, run_dir / f"meta_{vname}.pkl")
        with open(run_dir / f"meta_{vname}_features.json", "w", encoding="utf-8") as f:
            json.dump(avail, f, indent=2)
        train_report[vname] = {"auc_insample": round(auc, 4), "features": avail, "n_features": len(avail)}
        print(f"\n  [{vname}] train AUC(in-sample)={auc:.4f} | {len(avail)} feats")

    fb_model = joblib.load(MODEL_DIR / "runs" / FB_RUN / "lgbm.pkl")
    with open(MODEL_DIR / "runs" / FB_RUN / f"{FB_RUN}_features.json", encoding="utf-8") as f:
        fb_feats = json.load(f)

    from core.models import load_lstm as _load_lstm
    lstm_model = _load_lstm(MODEL_DIR / "lstm_best.pt")
    lstm_scaler = joblib.load(MODEL_DIR / "lstm_scaler.pkl")
    with open(MODEL_DIR / "feature_cols_lstm_temporal.json", encoding="utf-8") as f:
        lstm_feats = json.load(f)

    # Marginal IC per variant
    print(f"\n  --- Marginal IC (holdout trade-level) ---")
    ic_report = {}
    for vname, (model, feats, _) in variant_models.items():
        ic = holdout_marginal_ic(model, feats, vname, fb_model, fb_feats)
        ic_report[vname] = ic
        if ic.get("verdict") == "SKIP":
            print(f"  {vname}: SKIPPED n={ic.get('n', 0)}")
        else:
            print(
                f"  {vname}: marginal_IC={ic['marginal_ic']:+.4f} t={ic['t_marginal']:+.1f} "
                f"corr={ic['corr_meta_conf']:+.3f} -> {ic['verdict']}"
            )

    # Holdout PnL
    print(f"\n  --- Holdout PnL (primary_hmm baseline, Guardian OFF) ---")
    pnl_results = holdout_pnl_eval(
        variant_models, fb_model, fb_feats, lstm_model, lstm_scaler, lstm_feats, base_wr,
    )
    baseline = pnl_results["primary_hmm"]
    print(f"  {'Arm':<28} {'Trades':>7} {'WR':>6} {'PF':>6} {'PnL':>9} {'$/mo':>7}")
    print("-" * 72)
    print(
        f"  {'primary_hmm':<28} {baseline['trades']:>7,} {baseline['wr']:>5.1f}% "
        f"{baseline['pf']:>6} ${baseline['pnl']:>+8.0f} ${baseline['pnl_per_month']:>+6.0f}"
    )
    ranked = sorted(
        ((k, v) for k, v in pnl_results.items() if k != "primary_hmm"),
        key=lambda x: x[1]["pnl"],
        reverse=True,
    )
    for k, v in ranked[:12]:
        delta = v["pnl"] - baseline["pnl"]
        print(
            f"  {k:<28} {v['trades']:>7,} {v['wr']:>5.1f}% "
            f"{v['pf']:>6} ${v['pnl']:>+8.0f} ${v['pnl_per_month']:>+6.0f} "
            f"(Δ${delta:+.0f})"
        )

    best_arm = max(pnl_results.items(), key=lambda x: x[1]["pnl"])
    best_meta = max(
        ((k, v) for k, v in pnl_results.items() if k != "primary_hmm"),
        key=lambda x: x[1]["pnl"],
    )
    any_ic_pass = any(r.get("gate_marginal") for r in ic_report.values())

    print(f"\n  Best overall arm : {best_arm[0]} (${best_arm[1]['pnl']:+.0f})")
    print(f"  Best meta arm    : {best_meta[0]} (${best_meta[1]['pnl']:+.0f}, Δ${best_meta[1]['pnl']-baseline['pnl']:+.0f})")
    print(f"  Any marginal IC pass: {'YES' if any_ic_pass else 'NO'}")

    if best_meta[1]["pnl"] > baseline["pnl"] and any_ic_pass:
        verdict = "PROMISING — lanjut Gate #2 (Guardian-OOF labels)"
    elif best_meta[1]["pnl"] > baseline["pnl"]:
        verdict = "PnL OK tapi IC lemah — coba soft multiplier only, jangan hard gate"
    else:
        verdict = "NO-GO — meta tidak beat primary_hmm di holdout"

    print(f"  Verdict: {verdict}")
    print(f"{'='*78}")

    out = {
        "evaluated_at": datetime.now().isoformat(),
        "oof_n": len(oof_df),
        "oof_base_wr": round(base_wr, 4),
        "variants": VARIANTS,
        "train": train_report,
        "marginal_ic_holdout": ic_report,
        "holdout_pnl": pnl_results,
        "baseline_pnl": baseline["pnl"],
        "best_meta_arm": best_meta[0],
        "verdict": verdict,
    }
    out_path = run_dir / "explore_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\n  Saved -> {out_path}")


if __name__ == "__main__":
    main()