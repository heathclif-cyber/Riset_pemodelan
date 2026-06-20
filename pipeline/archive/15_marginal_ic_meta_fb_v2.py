"""
pipeline/15_marginal_ic_meta_fb_v2.py
Simon Marginal IC Gate #1 — apakah meta menambah informasi di atas flatboost_v2?

Pertanyaan:
  IC(meta_p_win, WIN) standalone
  IC(meta_p_win, WIN | LGBM_conf) marginal  — Gram-Schmidt residual
  Pass: |marginal_IC| >= 0.015 AND |t| >= 2.0 (Simon threshold)

Data:
  OOF  : data/meta_labels/fb_v2_oof_trades.parquet + models/runs/tb_meta_fb_v2/meta_lgbm.pkl
  Holdout: sim entry Apr-Jun 2026 per coin (label WIN dari full_trading_report trades)

Usage:
  python pipeline/15_marginal_ic_meta_fb_v2.py
"""
import json
import sys
import warnings
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

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
from core.meta_labeling import build_meta_row, hmm_entry_from_proba
from core.utils import ensure_utc_index

META_CONTEXT = [
    "hmm_regime_enc", "atr_percentile_h1", "funding_rate",
    "vol_spike_zscore", "ofi_h4_delta", "cvd_slope_h4",
]
FB_RUN = "tb_lgbm_flatboost_v2"
META_RUN = "tb_meta_fb_v2"
OOF_PATH = ROOT / "data" / "meta_labels" / "fb_v2_oof_trades.parquet"
HOLDOUT_START = "2026-04-01"

IC_STANDALONE_THR = 0.02
MARGINAL_IC_THR = 0.015
TSTAT_THR = 2.0
LM = LABEL_MAP if isinstance(LABEL_MAP, dict) else {"SHORT": 0, "FLAT": 1, "LONG": 2}


def rank_ic(x: np.ndarray, y: np.ndarray) -> dict:
    mask = ~(np.isnan(x) | np.isnan(y))
    x, y = x[mask], y[mask]
    n = int(mask.sum())
    if n < 50:
        return {"ic": np.nan, "t": np.nan, "n": n}
    ic, _ = spearmanr(x, y)
    if np.isnan(ic):
        return {"ic": np.nan, "t": np.nan, "n": n}
    t = ic * np.sqrt((n - 2) / (1 - ic**2 + 1e-9))
    return {"ic": float(ic), "t": float(t), "n": n}


def marginal_ic_gs(x_new: np.ndarray, x_base: np.ndarray, y: np.ndarray) -> dict:
    """Gram-Schmidt: IC of residual(x_new | x_base) vs y."""
    mask = ~(np.isnan(x_new) | np.isnan(x_base) | np.isnan(y))
    xn, xb, ym = x_new[mask], x_base[mask], y[mask]
    n = int(mask.sum())
    if n < 50:
        return {"marginal_ic": np.nan, "t": np.nan, "corr_with_base": np.nan, "n": n}
    corr = float(np.corrcoef(xn, xb)[0, 1])
    xn_resid = xn - corr * xb
    ic, _ = spearmanr(xn_resid, ym)
    if np.isnan(ic):
        return {"marginal_ic": np.nan, "t": np.nan, "corr_with_base": corr, "n": n}
    t = ic * np.sqrt((n - 2) / (1 - ic**2 + 1e-9))
    return {"marginal_ic": float(ic), "t": float(t), "corr_with_base": corr, "n": n}


def pass_gate(ic: float, t: float, ic_thr: float) -> bool:
    return not (np.isnan(ic) or np.isnan(t)) and abs(ic) >= ic_thr and abs(t) >= TSTAT_THR


def load_meta_model():
    meta_path = MODEL_DIR / "runs" / META_RUN / "meta_lgbm.pkl"
    feat_path = MODEL_DIR / "runs" / META_RUN / f"{META_RUN}_features.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Run 09_train_meta_lgbm_fb_v2.py first: {meta_path}")
    model = joblib.load(meta_path)
    with open(feat_path, encoding="utf-8") as f:
        feats = json.load(f)
    return model, feats


def predict_meta_proba(model, feats, df: pd.DataFrame) -> np.ndarray:
    """Predict p_win for each row using stored meta features or rebuild."""
    if all(c in df.columns for c in feats):
        X = df[feats].ffill().fillna(0).values.astype(np.float64)
        return model.predict_proba(X)[:, 1].astype(np.float64)
    out = np.full(len(df), np.nan, dtype=np.float64)
    for i in range(len(df)):
        if "p_long" not in df.columns:
            continue
        sig = 2 if df.iloc[i].get("direction", 0) == 1 else 0
        row = df.iloc[i]
        proba = np.array([row["p_short"], row["p_flat"], row["p_long"]], dtype=np.float64)
        meta_row = build_meta_row(proba, sig, row, META_CONTEXT)
        X = np.array([[meta_row.get(f, 0.0) for f in feats]], dtype=np.float64)
        out[i] = float(model.predict_proba(X)[0, 1])
    return out


def analyze_dataset(name: str, y: np.ndarray, conf: np.ndarray, p_meta: np.ndarray) -> dict:
    ic_conf = rank_ic(conf, y)
    ic_meta = rank_ic(p_meta, y)
    marg = marginal_ic_gs(p_meta, conf, y)

    conf_pass = pass_gate(ic_conf["ic"], ic_conf["t"], IC_STANDALONE_THR)
    meta_sa_pass = pass_gate(ic_meta["ic"], ic_meta["t"], IC_STANDALONE_THR)
    marg_pass = pass_gate(marg["marginal_ic"], marg["t"], MARGINAL_IC_THR)

    verdict = "PASS" if marg_pass else "FAIL"
    return {
        "dataset": name,
        "n": ic_meta["n"],
        "ic_lgbm_conf": round(ic_conf["ic"], 4) if not np.isnan(ic_conf["ic"]) else None,
        "t_lgbm_conf": round(ic_conf["t"], 2) if not np.isnan(ic_conf["t"]) else None,
        "ic_meta_standalone": round(ic_meta["ic"], 4) if not np.isnan(ic_meta["ic"]) else None,
        "t_meta_standalone": round(ic_meta["t"], 2) if not np.isnan(ic_meta["t"]) else None,
        "corr_meta_conf": round(marg["corr_with_base"], 4) if not np.isnan(marg["corr_with_base"]) else None,
        "marginal_ic_meta_given_conf": round(marg["marginal_ic"], 4) if not np.isnan(marg["marginal_ic"]) else None,
        "t_marginal": round(marg["t"], 2) if not np.isnan(marg["t"]) else None,
        "gate_conf_standalone": conf_pass,
        "gate_meta_standalone": meta_sa_pass,
        "gate_marginal": marg_pass,
        "verdict": verdict,
    }


def run_oof(meta_model, meta_feats) -> dict:
    df = pd.read_parquet(OOF_PATH)
    if "timestamp" in df.columns:
        df = df.set_index("timestamp")
    df.index = pd.to_datetime(df.index, utc=True)

    y = df["win"].values.astype(np.float64)
    conf = df["confidence"].values.astype(np.float64)
    p_meta = predict_meta_proba(meta_model, meta_feats, df)
    return analyze_dataset("OOF_trades", y, conf, p_meta)


def run_holdout(meta_model, meta_feats) -> dict:
    fb_model = joblib.load(MODEL_DIR / "runs" / FB_RUN / "lgbm.pkl")
    with open(MODEL_DIR / "runs" / FB_RUN / f"{FB_RUN}_features.json", encoding="utf-8") as f:
        fb_feats = json.load(f)

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
        if not trades:
            continue

        for t in trades:
            idx = t.get("bar_in")
            if idx is None:
                idx = t.get("entry_bar")
            if idx is None:
                continue
            idx = int(idx)
            if idx < 0 or idx >= n:
                continue

            sig = int(yp[idx])
            if sig == 1:
                continue
            proba = proba_fb[idx]
            row = df.iloc[idx]
            meta_row = build_meta_row(proba, sig, row, META_CONTEXT)
            X = np.array([[meta_row.get(f, 0.0) for f in meta_feats]], dtype=np.float64)
            p_win = float(meta_model.predict_proba(X)[0, 1])
            records.append({
                "win": 1.0 if t.get("net_pnl", 0) > 0 else 0.0,
                "confidence": float(conf_arr[idx]),
                "p_meta": p_win,
            })

    if len(records) < 50:
        return {"dataset": "holdout_trades", "verdict": "SKIP", "n": len(records),
                "note": "insufficient holdout trade records"}

    hdf = pd.DataFrame(records)
    return analyze_dataset(
        "holdout_AprJun2026",
        hdf["win"].values,
        hdf["confidence"].values,
        hdf["p_meta"].values,
    )


def main():
    print(f"\n{'='*72}")
    print("  SIMON MARGINAL IC GATE #1 — tb_meta_fb_v2 vs flatboost_v2")
    print(f"  Pass: |marginal_IC| >= {MARGINAL_IC_THR} AND |t| >= {TSTAT_THR}")
    print(f"{'='*72}")

    meta_model, meta_feats = load_meta_model()
    oof = run_oof(meta_model, meta_feats)
    hold = run_holdout(meta_model, meta_feats)

    for r in (oof, hold):
        if r.get("verdict") == "SKIP":
            print(f"\n  [{r['dataset']}] SKIPPED — {r.get('note')}")
            continue
        print(f"\n  --- {r['dataset']} (n={r['n']:,}) ---")
        print(f"  IC(LGBM_conf, WIN)     : {r['ic_lgbm_conf']:+.4f}  t={r['t_lgbm_conf']:+.1f}  "
              f"{'PASS' if r['gate_conf_standalone'] else 'fail'}")
        print(f"  IC(meta_p_win, WIN)    : {r['ic_meta_standalone']:+.4f}  t={r['t_meta_standalone']:+.1f}  "
              f"{'PASS' if r['gate_meta_standalone'] else 'fail'}")
        print(f"  corr(meta, conf)       : {r['corr_meta_conf']:+.4f}")
        print(f"  Marginal IC(meta|conf) : {r['marginal_ic_meta_given_conf']:+.4f}  t={r['t_marginal']:+.1f}  "
              f"--> {r['verdict']}")

    overall = "PASS" if oof.get("gate_marginal") and hold.get("gate_marginal") else "FAIL"
    if hold.get("verdict") == "SKIP":
        overall = "PASS" if oof.get("gate_marginal") else "FAIL"
        overall_note = "OOF only (holdout skipped)"
    else:
        overall_note = "OOF + holdout both required"
        if not hold.get("gate_marginal"):
            overall = "FAIL"

    print(f"\n{'='*72}")
    print(f"  OVERALL GATE #1: {overall} ({overall_note})")
    if overall == "FAIL":
        print("  Meta tidak menambah informasi orthogonal cukup kuat.")
        print("  Rekomendasi: STOP hard-gate meta — explore soft multiplier atau fitur baru.")
    else:
        print("  Lanjut Gate #2: label meta dari sim Guardian-OOF.")
    print(f"{'='*72}")

    out = {
        "evaluated_at": datetime.now().isoformat(),
        "model": "tb_widyawardhana_v2",
        "meta_run": META_RUN,
        "thresholds": {
            "standalone_ic": IC_STANDALONE_THR,
            "marginal_ic": MARGINAL_IC_THR,
            "t_stat": TSTAT_THR,
        },
        "oof": oof,
        "holdout": hold,
        "overall_verdict": overall,
    }
    out_path = MODEL_DIR / "runs" / META_RUN / "marginal_ic_gate1.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\n  Saved -> {out_path}")


if __name__ == "__main__":
    main()