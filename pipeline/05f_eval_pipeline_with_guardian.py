"""
pipeline/05f_eval_pipeline_with_guardian.py -- OOF Evaluation: LGBM + HMM + Guardian

Bandingkan 4 konfigurasi pipeline via OOF simulation:
  1. BASE    : LGBM flat threshold 0.45/0.45 (no Guardian)
  2. HMM-B   : LGBM + HMM Config B per-state threshold (no Guardian)
  3. FULL    : LGBM + HMM Config B + Guardian genuine_v2_hmm
  4. DYN     : LGBM + HMM Config B + Guardian + Dynamic Sizing (regime x confidence)

Semua via OOF predictions genuine_v2 -- tidak menyentuh holdout.

Output: ringkasan perbandingan + quarterly breakdown
"""
import json, sys, warnings
from pathlib import Path

import joblib, numpy as np, pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from config import (
    ALL_COINS, TRAIN_CUTOFF_DATE, LABEL_DIR, MODEL_DIR,
    MODAL_PER_TRADE, LEVERAGE_SIM, FEE_PER_SIDE, SLIPPAGE_PER_SIDE,
    SWING_LABEL_MIN_RR, SWING_LABEL_MIN_TP, SWING_LABEL_MAX_SL,
    TP_SL_FALLBACK_TP, TP_SL_FALLBACK_SL, MAX_HOLDING_BARS,
    GUARDIAN_ACTIVATION_ATR,
)
from core.evaluator import simulate_trades_swing
from core.utils import ensure_utc_index

LGBM_RUN      = "tb_lgbm_genuine_v2"
GUARDIAN_RUN  = "tb_guardian_genuine_v2_hmm_v2"
LGBM_DIR      = MODEL_DIR / "runs" / LGBM_RUN
GUARDIAN_DIR  = MODEL_DIR / "runs" / GUARDIAN_RUN
INFERENCE_CFG = MODEL_DIR / "inference_config.json"

# Frozen fallback (pre-sweep deploy); overridden by load_hmm_cfg() at runtime.
HMM_THR_CFG = {
    0:  (0.55, 0.55),
    1:  (0.55, 0.55),
    2:  (0.50, 0.50),
    3:  (0.45, 0.50),
    -1: (0.45, 0.45),
}

GUARDIAN_EXIT_THRESHOLD = 0.55
GUARDIAN_MIN_HOLD_BARS    = 2


def load_hmm_cfg() -> dict:
    """Load HMM Config B from latest OOF sweep artifact (genuine source of truth)."""
    path = LGBM_DIR / "hmm_threshold_best.json"
    if not path.exists():
        raise FileNotFoundError(
            f"HMM sweep belum dijalankan: {path}\n"
            "Jalankan: python pipeline/05e_hmm_threshold_sweep.py"
        )
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    cfg = data["best_config"]
    return {int(k): tuple(v) for k, v in cfg.items()}


def load_guardian_params() -> dict:
    """Deployed Guardian exit params from inference_config (OOF-tuned, not config.py default)."""
    with open(INFERENCE_CFG, encoding="utf-8") as f:
        inf = json.load(f)
    g = inf.get("guardian", {})
    return {
        "exit_threshold": float(g.get("exit_threshold", GUARDIAN_EXIT_THRESHOLD)),
        "min_hold_bars":  int(g.get("min_hold_bars", GUARDIAN_MIN_HOLD_BARS)),
    }

DYNAMIC_FEATS = [
    "bars_held_norm", "current_pnl_pct", "current_pnl_atr",
    "max_favorable_pnl_pct", "drawdown_from_peak_pct",
    "direction", "entry_price_ratio",
]


# Dynamic sizing: regime multiplier × confidence multiplier
# Regime multiplier based on historical WR profile per state
_REGIME_MULT = {
    0: 0.75,   # TRENDING_DOWN  — counter-trend risky both ways
    1: 1.0,    # RANGING_LOW    — post-filter quality standard
    2: 1.0,    # RANGING_HIGH   — standard
    # S3 handled direction-aware below
    -1: 0.80,  # unknown
}


def _compute_dynamic_modal(p0, p2, hmm_enc, y_pred, base_modal):
    """Per-bar modal = base_modal × regime_mult × confidence_mult.

    S3 LONG  = 1.5× (with-trend, highest WR)
    S3 SHORT = 0.75× (counter-trend in uptrend)
    Confidence mult: linear 1.0→1.5 over 10pp excess above HMM threshold.
    Total clamped to [0.5×, 2.0×].
    """
    n = len(p0)

    # Build per-bar threshold arrays (same logic as _apply_hmm_thr)
    tl_arr = np.full(n, HMM_THR_CFG[-1][0], dtype=np.float32)
    ts_arr = np.full(n, HMM_THR_CFG[-1][1], dtype=np.float32)
    for state, (tl, ts) in HMM_THR_CFG.items():
        if state == -1:
            continue
        mask = hmm_enc == state
        tl_arr[mask] = tl
        ts_arr[mask] = ts

    long_mask  = y_pred == 2
    short_mask = y_pred == 0

    conf = np.where(long_mask, p2, np.where(short_mask, p0, 0.0)).astype(np.float32)
    thr  = np.where(long_mask, tl_arr, ts_arr)

    # Confidence multiplier: 1.0 at threshold → 1.5 at threshold+0.10
    c_mult = 1.0 + np.clip((conf - thr) / 0.10, 0.0, 0.5)

    # Regime multiplier
    r_mult = np.full(n, 0.80, dtype=np.float64)  # default (unknown)
    r_mult[hmm_enc == 0] = 0.75
    r_mult[hmm_enc == 1] = 1.0
    r_mult[hmm_enc == 2] = 1.0
    r_mult[(hmm_enc == 3) & long_mask]  = 1.5   # S3 LONG  — with-trend
    r_mult[(hmm_enc == 3) & short_mask] = 0.75  # S3 SHORT — counter-trend

    total_mult = np.clip(r_mult * c_mult, 0.5, 2.0)
    modal_arr  = (base_modal * total_mult).astype(np.float32)
    modal_arr[y_pred == 1] = base_modal  # FLAT bars — irrelevant

    return modal_arr


def _apply_flat_thr(p0, p2, tl=0.45, ts=0.45):
    y = np.ones(len(p0), dtype=np.int32)
    y[p2 >= tl] = 2
    y[(p0 >= ts) & (y != 2)] = 0
    return y


def _apply_hmm_thr(p0, p2, hmm_enc):
    n = len(p0)
    default_tl, default_ts = HMM_THR_CFG[-1]
    tl_arr = np.full(n, default_tl, dtype=np.float32)
    ts_arr = np.full(n, default_ts, dtype=np.float32)
    for state, (tl, ts) in HMM_THR_CFG.items():
        if state == -1:
            continue
        mask = hmm_enc == state
        tl_arr[mask] = tl
        ts_arr[mask] = ts
    long_mask  = p2 >= tl_arr
    short_mask = (p0 >= ts_arr) & ~long_mask
    y = np.ones(n, dtype=np.int32)
    y[long_mask]  = 2
    y[short_mask] = 0
    return y


def _summarize(trades: list, base_modal: float = None) -> dict:
    if base_modal is None:
        base_modal = MODAL_PER_TRADE
    if not trades:
        return {"n": 0, "wr": 0, "pnl": 0, "ppt": 0, "pf": 0,
                "sl_pct": 0, "avg_hold": 0, "avg_modal": base_modal, "ppt_norm": 0}
    n      = len(trades)
    # WR = positive PnL trades (covers all outcome types incl. GUARDIAN_*)
    wins   = sum(1 for t in trades if t["net_pnl"] > 0)
    sl_hit = sum(1 for t in trades if t["outcome"] == "LOSS")
    gpnl   = sum(t["net_pnl"] for t in trades if t["net_pnl"] > 0)
    lloss  = sum(abs(t["net_pnl"]) for t in trades if t["net_pnl"] < 0)
    tpnl   = sum(t["net_pnl"] for t in trades)
    pf     = gpnl / lloss if lloss > 0 else float("inf")
    holds  = [t.get("bars_held", t.get("bar_out", 0) - t.get("bar_in", 0)) for t in trades]

    modals     = [t.get("modal_used", base_modal) for t in trades]
    avg_modal  = float(np.mean(modals))
    # PPT normalized to base_modal for apples-to-apples sizing comparison
    ppt_norm   = (tpnl / n) * (base_modal / avg_modal) if n > 0 and avg_modal > 0 else 0.0

    return {
        "n": n, "wr": wins / n * 100, "pnl": tpnl,
        "ppt": tpnl / n, "pf": pf, "sl_pct": sl_hit / n * 100,
        "avg_hold": float(np.mean(holds)) if holds else 0,
        "avg_modal": avg_modal,
        "ppt_norm": ppt_norm,
    }


def load_guardian():
    g_model  = joblib.load(GUARDIAN_DIR / "guardian.pkl")
    g_scaler = joblib.load(GUARDIAN_DIR / "guardian_scaler.pkl")
    with open(GUARDIAN_DIR / "guardian_features.json") as f:
        g_all_feats = json.load(f)
    g_static_feats = [f for f in g_all_feats if f not in DYNAMIC_FEATS]
    return g_model, g_scaler, g_static_feats


def run_coin(sym, oof_pred_df, g_model, g_scaler, g_static_feats):
    path = LABEL_DIR / f"{sym}_features_v3.parquet"
    if not path.exists():
        return None

    df = pd.read_parquet(path)
    df = ensure_utc_index(df).sort_index()
    df = df[df.index < TRAIN_CUTOFF_DATE]
    if df.empty:
        return None

    sym_oof = oof_pred_df[oof_pred_df["coin"] == sym]
    sym_oof = sym_oof[sym_oof["has_oof"] == True][["p0", "p2"]]
    proba   = sym_oof.reindex(df.index)
    has_oof = proba["p0"].notna()
    df_oof  = df[has_oof].copy()
    proba   = proba[has_oof]
    n       = len(df_oof)
    if n < 30:
        return None

    p0  = proba["p0"].values.astype(np.float32)
    p2  = proba["p2"].values.astype(np.float32)
    hmm = df_oof["hmm_regime_enc"].fillna(-1).values.astype(np.int8) \
          if "hmm_regime_enc" in df_oof.columns else np.full(n, -1, np.int8)
    ts  = df_oof.index

    close = df_oof["close"].values.astype(np.float64)
    high  = df_oof["high"].values.astype(np.float64)
    low   = df_oof["low"].values.astype(np.float64)
    atr   = df_oof["atr_14_h1"].values.astype(np.float64)
    h4_sh = df_oof["h4_swing_high"].values.astype(np.float64) \
            if "h4_swing_high" in df_oof.columns else np.full(n, np.nan)
    h4_sl = df_oof["h4_swing_low"].values.astype(np.float64) \
            if "h4_swing_low" in df_oof.columns else np.full(n, np.nan)

    # Guardian static matrix
    X_grd = np.zeros((n, len(g_static_feats)), dtype=np.float64)
    for idx, col in enumerate(g_static_feats):
        if col in df_oof.columns:
            X_grd[:, idx] = df_oof[col].ffill().fillna(0).values.astype(np.float64)

    common_kwargs = dict(
        close=close, high=high, low=low, atr=atr,
        h4_swing_highs=h4_sh, h4_swing_lows=h4_sl,
        modal=MODAL_PER_TRADE, leverage=LEVERAGE_SIM[0],
        fee_per_side=FEE_PER_SIDE, slippage=SLIPPAGE_PER_SIDE,
        max_hold=MAX_HOLDING_BARS,
        min_rr=SWING_LABEL_MIN_RR, min_tp_atr=SWING_LABEL_MIN_TP,
        max_sl_atr=SWING_LABEL_MAX_SL,
        tp_fallback_atr=TP_SL_FALLBACK_TP, sl_fallback_atr=TP_SL_FALLBACK_SL,
    )

    # (1) BASE: flat 0.45, no Guardian
    y_base = _apply_flat_thr(p0, p2)
    r_base = simulate_trades_swing(y_pred=y_base, guardian_enabled=False, **common_kwargs)

    # (2) HMM Config B, no Guardian
    y_hmm = _apply_hmm_thr(p0, p2, hmm)
    r_hmm  = simulate_trades_swing(y_pred=y_hmm, guardian_enabled=False, **common_kwargs)

    # (3) HMM Config B + Guardian
    r_full = simulate_trades_swing(
        y_pred=y_hmm,
        guardian_enabled=True,
        guardian_model=g_model,
        guardian_scaler=g_scaler,
        X_guardian=X_grd,
        guardian_exit_threshold=GUARDIAN_EXIT_THRESHOLD,
        guardian_min_hold_bars=GUARDIAN_MIN_HOLD_BARS,
        guardian_activation_atr=GUARDIAN_ACTIVATION_ATR,
        **common_kwargs,
    )

    # (4) HMM Config B + Guardian + Dynamic Sizing (regime × confidence)
    modal_arr_dyn = _compute_dynamic_modal(p0, p2, hmm, y_hmm, MODAL_PER_TRADE)
    r_dyn = simulate_trades_swing(
        y_pred=y_hmm,
        guardian_enabled=True,
        guardian_model=g_model,
        guardian_scaler=g_scaler,
        X_guardian=X_grd,
        guardian_exit_threshold=GUARDIAN_EXIT_THRESHOLD,
        guardian_min_hold_bars=GUARDIAN_MIN_HOLD_BARS,
        guardian_activation_atr=GUARDIAN_ACTIVATION_ATR,
        modal_arr=modal_arr_dyn,
        **common_kwargs,
    )

    # Attach timestamps to trades for quarterly analysis
    def tag_ts(trades, ts_arr):
        for t in trades:
            bi = t.get("bar_in", 0)
            t["ts"] = ts_arr[bi] if bi < len(ts_arr) else None
        return trades

    return {
        "base": tag_ts(r_base.get("trades", []), ts),
        "hmm":  tag_ts(r_hmm.get("trades",  []), ts),
        "full": tag_ts(r_full.get("trades", []), ts),
        "dyn":  tag_ts(r_dyn.get("trades",  []), ts),
    }


def quarterly_breakdown(trades: list) -> pd.DataFrame:
    rows = []
    for t in trades:
        ts = t.get("ts")
        if ts is None:
            continue
        qstr = f"{ts.year}Q{ts.quarter}"
        rows.append({
            "quarter": qstr,
            "pnl":     t["net_pnl"],
            "win":     1 if t["net_pnl"] > 0 else 0,
        })
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    g = df.groupby("quarter").agg(
        n=("pnl", "count"),
        pnl=("pnl", "sum"),
        wr=("win", "mean"),
    ).reset_index()
    g["ppt"]  = g["pnl"] / g["n"]
    g["wr"]   = g["wr"] * 100
    return g.sort_values("quarter")


def main():
    global HMM_THR_CFG, GUARDIAN_EXIT_THRESHOLD, GUARDIAN_MIN_HOLD_BARS

    SEP  = "=" * 78
    SEP2 = "-" * 78

    HMM_THR_CFG = load_hmm_cfg()
    g_params = load_guardian_params()
    GUARDIAN_EXIT_THRESHOLD = g_params["exit_threshold"]
    GUARDIAN_MIN_HOLD_BARS  = g_params["min_hold_bars"]

    print(f"\n{SEP}")
    print(f"  OOF PIPELINE EVAL: LGBM + HMM Config B + Guardian + Dynamic Sizing")
    print(f"  Genuine: OOF only | holdout NOT touched | TRAIN_CUTOFF enforced")
    print(f"{SEP}\n")
    print(f"  HMM Config B (from hmm_threshold_best.json):")
    for s in sorted(k for k in HMM_THR_CFG if k >= 0):
        tl, ts = HMM_THR_CFG[s]
        print(f"    S{s}: thr_long={tl:.2f}  thr_short={ts:.2f}")
    print(f"  Guardian: exit_thr={GUARDIAN_EXIT_THRESHOLD}  min_hold={GUARDIAN_MIN_HOLD_BARS}")
    print()

    oof_pred_df = pd.read_parquet(LGBM_DIR / "oof_predictions.parquet")
    g_model, g_scaler, g_static_feats = load_guardian()
    print(f"  Guardian model: {GUARDIAN_RUN} ({len(g_static_feats)} static + {len(DYNAMIC_FEATS)} dynamic)")

    all_base, all_hmm, all_full, all_dyn = [], [], [], []

    for sym in ALL_COINS:
        print(f"  {sym}...", end=" ", flush=True)
        res = run_coin(sym, oof_pred_df, g_model, g_scaler, g_static_feats)
        if res is None:
            print("skip")
            continue
        all_base.extend(res["base"])
        all_hmm.extend(res["hmm"])
        all_full.extend(res["full"])
        all_dyn.extend(res["dyn"])
        sb  = _summarize(res["base"])
        sh  = _summarize(res["hmm"])
        sf  = _summarize(res["full"])
        sd  = _summarize(res["dyn"])
        print(f"BASE PPT={sb['ppt']:>+.3f} | "
              f"HMM PPT={sh['ppt']:>+.3f} | "
              f"FULL PPT={sf['ppt']:>+.3f} | "
              f"DYN PPT={sd['ppt']:>+.3f} (norm={sd['ppt_norm']:>+.3f})")

    sb_all  = _summarize(all_base)
    sh_all  = _summarize(all_hmm)
    sf_all  = _summarize(all_full)
    sd_all  = _summarize(all_dyn)

    # ── Summary table ─────────────────────────────────────────────────────────
    print(f"\n{SEP}")
    print(f"  RINGKASAN PERBANDINGAN OOF (semua koin, 2020–2026-04-01)")
    print(SEP)
    print(f"\n  {'Config':<30} {'N':>7}  {'WR%':>5}  {'PnL':>10}  {'PPT':>7}  "
          f"{'PPT_norm':>8}  {'PF':>5}  {'SL%':>5}  {'AvgModal':>9}")
    print(f"  {SEP2}")

    for name, s in [("1. BASE (0.45/0.45)", sb_all),
                    ("2. HMM Config B", sh_all),
                    ("3. HMM + Guardian", sf_all),
                    ("4. HMM + Guardian + DynSize", sd_all)]:
        pf = f"{s['pf']:.3f}" if s["pf"] != float("inf") else "    INF"
        print(f"  {name:<30} {s['n']:>7,}  {s['wr']:>5.1f}%  "
              f"${s['pnl']:>9.1f}  {s['ppt']:>+7.4f}  "
              f"{s['ppt_norm']:>+8.4f}  {pf}  {s['sl_pct']:>5.1f}%  "
              f"${s['avg_modal']:>8.1f}")

    print(f"\n  Delta HMM vs BASE:       "
          f"N {sh_all['n']-sb_all['n']:>+7,}  "
          f"WR {sh_all['wr']-sb_all['wr']:>+5.1f}pp  "
          f"PPT ${sh_all['ppt']-sb_all['ppt']:>+.4f}  "
          f"PnL ${sh_all['pnl']-sb_all['pnl']:>+.1f}")
    print(f"  Delta FULL vs HMM:       "
          f"N {sf_all['n']-sh_all['n']:>+7,}  "
          f"WR {sf_all['wr']-sh_all['wr']:>+5.1f}pp  "
          f"PPT ${sf_all['ppt']-sh_all['ppt']:>+.4f}  "
          f"PnL ${sf_all['pnl']-sh_all['pnl']:>+.1f}")
    print(f"  Delta DYN vs FULL:       "
          f"N {sd_all['n']-sf_all['n']:>+7,}  "
          f"WR {sd_all['wr']-sf_all['wr']:>+5.1f}pp  "
          f"PPT_norm ${sd_all['ppt_norm']-sf_all['ppt_norm']:>+.4f}  "
          f"PnL ${sd_all['pnl']-sf_all['pnl']:>+.1f}  "
          f"AvgModal ${sd_all['avg_modal']:>.1f} (vs ${sf_all['avg_modal']:>.1f})")
    print(f"  Delta DYN vs BASE:       "
          f"N {sd_all['n']-sb_all['n']:>+7,}  "
          f"WR {sd_all['wr']-sb_all['wr']:>+5.1f}pp  "
          f"PPT_norm ${sd_all['ppt_norm']-sb_all['ppt']:>+.4f}  "
          f"PnL ${sd_all['pnl']-sb_all['pnl']:>+.1f}")

    # ── Quarterly breakdown ────────────────────────────────────────────────────
    print(f"\n{SEP}")
    print(f"  QUARTERLY BREAKDOWN — Config 4: DYN (HMM + Guardian + Dynamic Sizing)")
    print(SEP)
    qdf = quarterly_breakdown(all_dyn)
    if not qdf.empty:
        print(f"\n  {'Quarter':<8} {'N':>6}  {'WR%':>5}  {'PnL':>9}  {'PPT':>7}")
        print(f"  {'-'*45}")
        for _, row in qdf.iterrows():
            print(f"  {row['quarter']:<8} {int(row['n']):>6,}  "
                  f"{row['wr']:>5.1f}%  ${row['pnl']:>8.1f}  {row['ppt']:>+7.4f}")

    # ── Guardian impact breakdown ────────────────────────────────────────────
    print(f"\n{SEP}")
    print(f"  GUARDIAN IMPACT: Outcome distribution FULL vs DYN")
    print(SEP)

    def outcome_dist(trades):
        counts = {}
        for t in trades:
            o = t.get("outcome", "?")
            counts[o] = counts.get(o, 0) + 1
        return counts

    d_hmm  = outcome_dist(all_hmm)
    d_full = outcome_dist(all_full)
    d_dyn  = outcome_dist(all_dyn)

    print(f"\n  {'Outcome':<22} {'HMM':>9}  {'FULL':>9}  {'DYN':>9}")
    print(f"  {'-'*55}")
    all_outcomes = sorted(set(list(d_hmm.keys()) + list(d_full.keys()) + list(d_dyn.keys())))
    for o in all_outcomes:
        h = d_hmm.get(o, 0)
        f = d_full.get(o, 0)
        d = d_dyn.get(o, 0)
        print(f"  {o:<22} {h:>9,}  {f:>9,}  {d:>9,}")

    def pnl_by_outcome(trades):
        buckets = {}
        for t in trades:
            o = t.get("outcome", "?")
            buckets.setdefault(o, []).append(t["net_pnl"])
        return {o: (np.mean(v), len(v)) for o, v in buckets.items()}

    pbo_hmm  = pnl_by_outcome(all_hmm)
    pbo_full = pnl_by_outcome(all_full)
    pbo_dyn  = pnl_by_outcome(all_dyn)

    print(f"\n  Avg PnL per outcome (dynamic sizing: gross reflects modal used):")
    print(f"  {'Outcome':<22} {'HMM avg$':>10}  {'FULL avg$':>10}  {'DYN avg$':>10}")
    print(f"  {'-'*58}")
    for o in all_outcomes:
        h_avg = pbo_hmm.get(o,  (0, 0))[0]
        f_avg = pbo_full.get(o, (0, 0))[0]
        d_avg = pbo_dyn.get(o,  (0, 0))[0]
        print(f"  {o:<22} {h_avg:>+10.3f}  {f_avg:>+10.3f}  {d_avg:>+10.3f}")

    # ── Dynamic sizing distribution ────────────────────────────────────────
    print(f"\n{SEP}")
    print(f"  DYNAMIC SIZING DISTRIBUTION (Config 4)")
    print(SEP)
    modals_dyn = [t.get("modal_used", MODAL_PER_TRADE) for t in all_dyn]
    if modals_dyn:
        marr = np.array(modals_dyn)
        print(f"\n  Base modal    : ${MODAL_PER_TRADE:.0f}")
        print(f"  Mean modal    : ${marr.mean():.2f}")
        print(f"  Min / Max     : ${marr.min():.2f} / ${marr.max():.2f}")
        print(f"  Std           : ${marr.std():.2f}")
        # Distribution buckets (relative to base_modal)
        bm = MODAL_PER_TRADE
        buckets = [
            (0.0,    0.7*bm, "< 0.7x  (reduced)   "),
            (0.7*bm, 0.9*bm, "0.7-0.9x (below base)"),
            (0.9*bm, 1.1*bm, "0.9-1.1x (~base)     "),
            (1.1*bm, 1.5*bm, "1.1-1.5x (above base)"),
            (1.5*bm, 2.1*bm, "1.5-2.0x (max)       "),
        ]
        print(f"\n  Modal bucket               Count    Pct")
        for lo, hi, label in buckets:
            cnt = int(((marr >= lo) & (marr < hi)).sum())
            pct = cnt / len(marr) * 100
            print(f"  {label}  {cnt:>6,}  {pct:>5.1f}%")

    # ── Save audit artifact ─────────────────────────────────────────────────
    out = {
        "methodology": "oof_pipeline_eval_genuine",
        "holdout_used": False,
        "lgbm_run": LGBM_RUN,
        "guardian_run": GUARDIAN_RUN,
        "hmm_thresholds": {str(k): list(v) for k, v in HMM_THR_CFG.items()},
        "guardian_params": {
            "exit_threshold": GUARDIAN_EXIT_THRESHOLD,
            "min_hold_bars": GUARDIAN_MIN_HOLD_BARS,
        },
        "summary": {
            "base": sb_all,
            "hmm": sh_all,
            "full": sf_all,
            "dyn": sd_all,
        },
    }
    out_path = LGBM_DIR / "oof_pipeline_eval.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"  Saved: {out_path}")

    print(f"\n{SEP}\n")


if __name__ == "__main__":
    main()
