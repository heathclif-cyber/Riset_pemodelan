"""
pipeline/05h_filter_ablation_genuine.py

Genuine OOF ablation: structural filter, RR gate, VCB
pada stack frozen tb_genuine_v2_dynsize (LGBM 36f + HMM-B + Guardian + DynSize cm_0.60).

Holdout NOT touched.
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
    ALL_COINS, TRAIN_CUTOFF_DATE, LABEL_DIR, MODEL_DIR,
    MODAL_PER_TRADE, LEVERAGE_SIM, FEE_PER_SIDE, SLIPPAGE_PER_SIDE,
    SWING_LABEL_MIN_RR, SWING_LABEL_MIN_TP, SWING_LABEL_MAX_SL,
    TP_SL_FALLBACK_TP, TP_SL_FALLBACK_SL, MAX_HOLDING_BARS,
    GUARDIAN_ACTIVATION_ATR, TP_SL_STRUCTURAL_TOLERANCE,
)
from core.evaluator import simulate_trades_swing
from core.utils import ensure_utc_index

LGBM_RUN = "tb_lgbm_genuine_v2"
GUARDIAN_RUN = "tb_guardian_genuine_v2_hmm_v2"
LGBM_DIR = MODEL_DIR / "runs" / LGBM_RUN
GUARDIAN_DIR = MODEL_DIR / "runs" / GUARDIAN_RUN
INFERENCE_CFG = MODEL_DIR / "inference_config.json"

HMM_CONFIG_B = {
    0: (0.55, 0.55), 1: (0.55, 0.55), 2: (0.50, 0.50),
    3: (0.45, 0.50), -1: (0.45, 0.45),
}

DYNAMIC_FEATS = [
    "bars_held_norm", "current_pnl_pct", "current_pnl_atr",
    "max_favorable_pnl_pct", "drawdown_from_peak_pct",
    "direction", "entry_price_ratio",
]

DYNSIZE = {
    "conf_window": 0.10,
    "conf_max_mult": 0.60,
    "regime_mult": {0: 0.75, 1: 1.0, 2: 1.0, "3_long": 1.5, "3_short": 0.75, -1: 0.80},
    "clamp_min": 0.5,
    "clamp_max": 2.0,
}

VCB_CFG = {"atr_multiplier": 3.0, "lookback_bars": 24}

VARIANTS = {
    "frozen_baseline": {
        "label": "Frozen OOF scorecard (struct ON, RR ON, VCB OFF)",
        "structural_filter": True,
        "vcb_enabled": False,
        "rr_relaxed": False,
    },
    "prod_all_on": {
        "label": "Production intent (struct ON, RR ON, VCB ON)",
        "structural_filter": True,
        "vcb_enabled": True,
        "rr_relaxed": False,
    },
    "no_structural": {
        "label": "Structural filter OFF",
        "structural_filter": False,
        "vcb_enabled": False,
        "rr_relaxed": False,
    },
    "no_rr_gate": {
        "label": "RR gate OFF",
        "structural_filter": True,
        "vcb_enabled": False,
        "rr_relaxed": True,
    },
    "no_struct_no_rr": {
        "label": "Structural OFF + RR OFF",
        "structural_filter": False,
        "vcb_enabled": False,
        "rr_relaxed": True,
    },
    "vcb_only_delta": {
        "label": "VCB ON vs frozen (isolated)",
        "structural_filter": True,
        "vcb_enabled": True,
        "rr_relaxed": False,
    },
}


def _apply_hmm_thr(p0, p2, hmm_enc, hmm_cfg):
    n = len(p0)
    default_tl, default_ts = hmm_cfg[-1]
    tl_arr = np.full(n, default_tl, dtype=np.float32)
    ts_arr = np.full(n, default_ts, dtype=np.float32)
    for state, (tl, ts) in hmm_cfg.items():
        if state == -1:
            continue
        mask = hmm_enc == state
        tl_arr[mask] = tl
        ts_arr[mask] = ts
    long_mask = p2 >= tl_arr
    short_mask = (p0 >= ts_arr) & ~long_mask
    y = np.ones(n, dtype=np.int32)
    y[long_mask] = 2
    y[short_mask] = 0
    return y, tl_arr, ts_arr


def _compute_dynamic_modal(p0, p2, hmm_enc, y_pred, tl_arr, ts_arr):
    ds = DYNSIZE
    rm = ds["regime_mult"]
    n = len(p0)
    long_mask = y_pred == 2
    short_mask = y_pred == 0
    conf = np.where(long_mask, p2, np.where(short_mask, p0, 0.0)).astype(np.float32)
    thr = np.where(long_mask, tl_arr, ts_arr)
    c_mult = 1.0 + np.clip((conf - thr) / ds["conf_window"], 0.0, ds["conf_max_mult"])
    r_mult = np.full(n, rm[-1], dtype=np.float64)
    r_mult[hmm_enc == 0] = rm[0]
    r_mult[hmm_enc == 1] = rm[1]
    r_mult[hmm_enc == 2] = rm[2]
    r_mult[(hmm_enc == 3) & long_mask] = rm["3_long"]
    r_mult[(hmm_enc == 3) & short_mask] = rm["3_short"]
    total = np.clip(r_mult * c_mult, ds["clamp_min"], ds["clamp_max"])
    modal_arr = (MODAL_PER_TRADE * total).astype(np.float32)
    modal_arr[y_pred == 1] = MODAL_PER_TRADE
    return modal_arr


def _summarize(trades):
    if not trades:
        return {"n": 0, "wr": 0, "pnl": 0, "ppt": 0, "pf": 0,
                "sl_pct": 0, "avg_modal": MODAL_PER_TRADE, "ppt_norm": 0}
    n = len(trades)
    wins = sum(1 for t in trades if t["net_pnl"] > 0)
    sl_hit = sum(1 for t in trades if t["outcome"] == "LOSS")
    gpnl = sum(t["net_pnl"] for t in trades if t["net_pnl"] > 0)
    lloss = sum(abs(t["net_pnl"]) for t in trades if t["net_pnl"] < 0)
    tpnl = sum(t["net_pnl"] for t in trades)
    pf = gpnl / lloss if lloss > 0 else float("inf")
    modals = [t.get("modal_used", MODAL_PER_TRADE) for t in trades]
    avg_modal = float(np.mean(modals))
    ppt_norm = (tpnl / n) * (MODAL_PER_TRADE / avg_modal) if avg_modal > 0 else 0.0
    return {
        "n": n, "wr": wins / n * 100, "pnl": tpnl,
        "ppt": tpnl / n, "pf": pf, "sl_pct": sl_hit / n * 100,
        "avg_modal": avg_modal, "ppt_norm": ppt_norm,
    }


def load_guardian():
    g_model = joblib.load(GUARDIAN_DIR / "guardian.pkl")
    g_scaler = joblib.load(GUARDIAN_DIR / "guardian_scaler.pkl")
    with open(GUARDIAN_DIR / "guardian_features.json", encoding="utf-8") as f:
        g_all = json.load(f)
    g_static = [f for f in g_all if f not in DYNAMIC_FEATS]
    with open(INFERENCE_CFG, encoding="utf-8") as f:
        inf = json.load(f)
    g = inf.get("guardian", {})
    return g_model, g_scaler, g_static, float(g["exit_threshold"]), int(g["min_hold_bars"])


def preload_coins(oof_pred_df, g_static_feats):
    coins = []
    for sym in ALL_COINS:
        path = LABEL_DIR / f"{sym}_features_v3.parquet"
        if not path.exists():
            continue
        df = pd.read_parquet(path)
        df = ensure_utc_index(df).sort_index()
        df = df[df.index < TRAIN_CUTOFF_DATE]
        if df.empty:
            continue
        sym_oof = oof_pred_df[oof_pred_df["coin"] == sym]
        sym_oof = sym_oof[sym_oof["has_oof"] == True][["p0", "p2"]]
        proba = sym_oof.reindex(df.index)
        has_oof = proba["p0"].notna()
        df_oof = df[has_oof].copy()
        n = len(df_oof)
        if n < 30:
            continue
        X_grd = np.zeros((n, len(g_static_feats)), dtype=np.float64)
        for idx, col in enumerate(g_static_feats):
            if col in df_oof.columns:
                X_grd[:, idx] = df_oof[col].ffill().fillna(0).values.astype(np.float64)
        coins.append({
            "sym": sym, "n": n,
            "p0": proba["p0"][has_oof].values.astype(np.float32),
            "p2": proba["p2"][has_oof].values.astype(np.float32),
            "hmm": df_oof["hmm_regime_enc"].fillna(-1).values.astype(np.int8)
            if "hmm_regime_enc" in df_oof.columns else np.full(n, -1, np.int8),
            "close": df_oof["close"].values.astype(np.float64),
            "high": df_oof["high"].values.astype(np.float64),
            "low": df_oof["low"].values.astype(np.float64),
            "atr": df_oof["atr_14_h1"].values.astype(np.float64),
            "h4_sh": df_oof["h4_swing_high"].values.astype(np.float64)
            if "h4_swing_high" in df_oof.columns else np.full(n, np.nan),
            "h4_sl": df_oof["h4_swing_low"].values.astype(np.float64)
            if "h4_swing_low" in df_oof.columns else np.full(n, np.nan),
            "X_grd": X_grd,
        })
    return coins


def run_variant(coins, g_model, g_scaler, g_static, g_exit, g_hold, variant):
    all_trades = []
    vcb_blocked = 0
    rr_on = not variant["rr_relaxed"]
    min_rr = SWING_LABEL_MIN_RR if rr_on else 0.0
    min_tp = SWING_LABEL_MIN_TP if rr_on else 0.0
    max_sl = SWING_LABEL_MAX_SL if rr_on else 999.0

    common = dict(
        modal=MODAL_PER_TRADE, leverage=LEVERAGE_SIM[0],
        fee_per_side=FEE_PER_SIDE, slippage=SLIPPAGE_PER_SIDE,
        max_hold=MAX_HOLDING_BARS,
        min_rr=min_rr, min_tp_atr=min_tp, max_sl_atr=max_sl,
        tp_fallback_atr=TP_SL_FALLBACK_TP, sl_fallback_atr=TP_SL_FALLBACK_SL,
        structural_filter=variant["structural_filter"],
        structural_tolerance_pct=TP_SL_STRUCTURAL_TOLERANCE,
        vcb_enabled=variant["vcb_enabled"],
        vcb_atr_multiplier=VCB_CFG["atr_multiplier"],
        vcb_lookback_bars=VCB_CFG["lookback_bars"],
        guardian_enabled=True,
        guardian_model=g_model, guardian_scaler=g_scaler,
        guardian_exit_threshold=g_exit,
        guardian_min_hold_bars=g_hold,
        guardian_activation_atr=GUARDIAN_ACTIVATION_ATR,
    )

    for c in coins:
        y_hmm, tl_arr, ts_arr = _apply_hmm_thr(c["p0"], c["p2"], c["hmm"], HMM_CONFIG_B)
        modal_arr = _compute_dynamic_modal(c["p0"], c["p2"], c["hmm"], y_hmm, tl_arr, ts_arr)
        r = simulate_trades_swing(
            y_pred=y_hmm,
            close=c["close"], high=c["high"], low=c["low"], atr=c["atr"],
            h4_swing_highs=c["h4_sh"], h4_swing_lows=c["h4_sl"],
            X_guardian=c["X_grd"],
            modal_arr=modal_arr,
            **common,
        )
        all_trades.extend(r.get("trades", []))
        vcb_blocked += r.get("n_vcb_blocked", 0)

    summary = _summarize(all_trades)
    summary["vcb_blocked_bars"] = vcb_blocked
    return summary


def main():
    sep = "=" * 78
    print(f"\n{sep}")
    print("  05h: FILTER ABLATION — Genuine OOF Full Stack")
    print("  LGBM 36f + HMM Config B + Guardian v2 + DynSize cm_0.60")
    print(f"  holdout_used=False | TRAIN_CUTOFF enforced")
    print(sep)

    g_model, g_scaler, g_static, g_exit, g_hold = load_guardian()
    oof = pd.read_parquet(LGBM_DIR / "oof_predictions.parquet")
    print("  Preloading coins...")
    coins = preload_coins(oof, g_static)
    print(f"  Coins: {len(coins)}")

    results = {}
    for key, var in VARIANTS.items():
        if key == "vcb_only_delta":
            continue
        print(f"  Running {key}...", flush=True)
        results[key] = {**run_variant(coins, g_model, g_scaler, g_static, g_exit, g_hold, var),
                        "label": var["label"]}

    base = results["frozen_baseline"]
    prod = results["prod_all_on"]

    print(f"\n{sep}")
    print("  HASIL (full pipeline DYN, 21 koin)")
    print(sep)
    print(f"  {'Variant':<22} {'N':>7}  {'WR%':>5}  {'PPT_norm':>8}  {'PF':>5}  "
          f"{'Avg$':>6}  {'VCBblk':>7}")
    print("  " + "-" * 72)

    order = ["frozen_baseline", "prod_all_on", "no_structural", "no_rr_gate", "no_struct_no_rr"]
    for key in order:
        s = results[key]
        pf = f"{s['pf']:.3f}" if s["pf"] != float("inf") else "  INF"
        print(f"  {key:<22} {s['n']:>7,}  {s['wr']:>5.1f}%  {s['ppt_norm']:>+8.4f}  "
              f"{pf:>5}  ${s['avg_modal']:>5.1f}  {s['vcb_blocked_bars']:>7,}")

    print(f"\n  Delta vs frozen_baseline:")
    for key in order[1:]:
        s = results[key]
        print(f"    {key:<20}  N {s['n']-base['n']:>+6,}  "
              f"PPT_norm {s['ppt_norm']-base['ppt_norm']:>+.4f}  "
              f"PF {s['pf']-base['pf']:>+.3f}  WR {s['wr']-base['wr']:>+.1f}pp")

    # Recommendations
    rec = []
    if prod["n"] == base["n"] and abs(prod["ppt_norm"] - base["ppt_norm"]) < 0.0001:
        rec.append("VCB: negligible impact — safe to align OFF with OOF or keep ON as insurance")
    elif prod["ppt_norm"] < base["ppt_norm"]:
        rec.append("VCB: hurts PPT_norm — recommend OFF in inference_config")
    else:
        rec.append("VCB: improves PPT_norm — enable in OOF eval going forward")

    ns = results["no_structural"]
    if ns["ppt_norm"] < base["ppt_norm"]:
        rec.append("Structural filter: KEEP (OFF reduces PPT_norm)")
    else:
        rec.append("Structural filter: marginal — KEEP as safety net")

    nr = results["no_rr_gate"]
    if nr["ppt_norm"] < base["ppt_norm"] or nr["pf"] < base["pf"] * 0.98:
        rec.append("RR gate: KEEP")
    else:
        rec.append("RR gate: marginal benefit — KEEP for label consistency")

    print(f"\n  REKOMENDASI:")
    for r in rec:
        print(f"    - {r}")

    out = {
        "created": datetime.now().isoformat(),
        "methodology": "genuine_oof_filter_ablation",
        "holdout_used": False,
        "stack": "LGBM 36f + HMM Config B + Guardian v2 + DynSize cm_0.60",
        "variants": {k: results[k] for k in order},
        "recommendations": rec,
    }
    out_path = LGBM_DIR / "filter_ablation_genuine.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\n  Saved: {out_path}")
    print(f"{sep}\n")


if __name__ == "__main__":
    main()