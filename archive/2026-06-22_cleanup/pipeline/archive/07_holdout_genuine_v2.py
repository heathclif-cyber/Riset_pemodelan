"""
pipeline/07_holdout_genuine_v2.py — Holdout Evaluation (SEKALI SAJA)

PENTING:
  - Jalankan HANYA setelah semua keputusan di-freeze
  - Set HOLDOUT_EVALUATED = True setelah dijalankan
  - Jangan tune parameter berdasarkan hasil ini

Stack yang dievaluasi:
  LGBM genuine_v2 (34 fitur) → HMM Config B → Guardian v2 (tight) → Dynamic Sizing

Period holdout : Apr 1 – Jun 30, 2026  (data tersedia s/d Jun 13)
Output         : reports/experiments/YYYY-MM-DD_genuine_v2_holdout.md
"""

# Dikunci setelah evaluasi 2026-06-15
HOLDOUT_EVALUATED = True

if HOLDOUT_EVALUATED:
    raise RuntimeError(
        "Holdout genuine_v2 sudah dievaluasi!\n"
        "Jangan jalankan ulang — melanggar aturan OOF (kontaminasi holdout).\n"
        "Jika perlu holdout baru: buat 07_holdout_genuine_v3.py dengan periode berbeda."
    )

import json, sys, warnings
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from config import (
    ALL_COINS, OOS_START, OOS_END,
    MODAL_PER_TRADE, LEVERAGE_SIM, FEE_PER_SIDE, SLIPPAGE_PER_SIDE,
    MAX_HOLDING_BARS, SWING_LABEL_MIN_RR, SWING_LABEL_MIN_TP,
    SWING_LABEL_MAX_SL, TP_SL_FALLBACK_TP, TP_SL_FALLBACK_SL,
    GUARDIAN_EXIT_THRESHOLD, GUARDIAN_MIN_HOLD_BARS, GUARDIAN_ACTIVATION_ATR,
    MODEL_DIR, HOLDOUT_DIR, REPORT_DIR,
)
from core.evaluator import simulate_trades_swing
from core.utils import ensure_utc_index

LGBM_RUN     = "tb_lgbm_genuine_v2"
GUARDIAN_RUN = "tb_guardian_genuine_v2_hmm_v2"
LGBM_DIR     = MODEL_DIR / "runs" / LGBM_RUN
GUARD_DIR    = MODEL_DIR / "runs" / GUARDIAN_RUN

# HMM Config B (frozen — jangan ubah)
HMM_THR_CFG = {
    0:  (0.55, 0.55),  # TRENDING_DOWN
    1:  (0.55, 0.55),  # RANGING_LOW
    2:  (0.50, 0.50),  # RANGING_HIGH
    3:  (0.45, 0.50),  # TRENDING_UP (L=0.45 with-trend, S=0.50 counter-trend)
    -1: (0.45, 0.45),  # unknown
}

DYNAMIC_FEATS = {
    "bars_held_norm", "current_pnl_pct", "current_pnl_atr",
    "max_favorable_pnl_pct", "drawdown_from_peak_pct",
    "direction", "entry_price_ratio",
}

LONG, SHORT, FLAT = 2, 0, 1


def _apply_hmm_thr(p0: np.ndarray, p2: np.ndarray, hmm_enc: np.ndarray) -> np.ndarray:
    """Terapkan HMM per-state threshold ke probability arrays."""
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
    y[long_mask]  = LONG
    y[short_mask] = SHORT
    return y


def _compute_dynamic_modal(
    p0: np.ndarray, p2: np.ndarray,
    hmm_enc: np.ndarray, y_pred: np.ndarray,
    base_modal: float,
) -> np.ndarray:
    """Dynamic sizing: base_modal × regime_mult × confidence_mult.

    Regime multipliers (berdasarkan WR per state dari OOF):
      S0 TRENDING_DOWN : 0.75x (risky)
      S1 RANGING_LOW   : 1.0x  (standard post-filter)
      S2 RANGING_HIGH  : 1.0x  (standard)
      S3 LONG          : 1.5x  (with-trend highest WR)
      S3 SHORT         : 0.75x (counter-trend in uptrend)
      unknown          : 0.8x

    Confidence multiplier: 1.0 at threshold → 1.5 at threshold+10pp (linear).
    Total clamped ke [0.5x, 2.0x].
    """
    n = len(p0)

    tl_arr = np.full(n, HMM_THR_CFG[-1][0], dtype=np.float32)
    ts_arr = np.full(n, HMM_THR_CFG[-1][1], dtype=np.float32)
    for state, (tl, ts) in HMM_THR_CFG.items():
        if state == -1:
            continue
        mask = hmm_enc == state
        tl_arr[mask] = tl
        ts_arr[mask] = ts

    long_mask  = y_pred == LONG
    short_mask = y_pred == SHORT

    conf = np.where(long_mask, p2, np.where(short_mask, p0, 0.0)).astype(np.float32)
    thr  = np.where(long_mask, tl_arr, ts_arr)

    c_mult = 1.0 + np.clip((conf - thr) / 0.10, 0.0, 0.5)

    r_mult = np.full(n, 0.80, dtype=np.float64)
    r_mult[hmm_enc == 0] = 0.75
    r_mult[hmm_enc == 1] = 1.00
    r_mult[hmm_enc == 2] = 1.00
    r_mult[(hmm_enc == 3) & long_mask]  = 1.50
    r_mult[(hmm_enc == 3) & short_mask] = 0.75

    total_mult = np.clip(r_mult * c_mult, 0.5, 2.0)
    modal_arr  = (base_modal * total_mult).astype(np.float32)
    modal_arr[y_pred == FLAT] = base_modal
    return modal_arr


def load_models():
    lgbm  = joblib.load(LGBM_DIR / "lgbm.pkl")
    with open(LGBM_DIR / "features.json") as f:
        lgbm_feats = json.load(f)

    guard  = joblib.load(GUARD_DIR / "guardian.pkl")
    scaler = joblib.load(GUARD_DIR / "guardian_scaler.pkl")
    with open(GUARD_DIR / "guardian_features.json") as f:
        guard_feats = json.load(f)

    return lgbm, lgbm_feats, guard, scaler, guard_feats


def evaluate_coin(sym, lgbm, lgbm_feats, guard, scaler, guard_feats):
    path = HOLDOUT_DIR / "labeled" / f"{sym}_features_v3.parquet"
    if not path.exists():
        return None

    df = pd.read_parquet(path)
    df = ensure_utc_index(df).sort_index()
    df = df[(df.index >= OOS_START) & (df.index < OOS_END)]
    if len(df) < 50:
        return None

    n = len(df)

    # LGBM live inference
    X_lgbm = np.zeros((n, len(lgbm_feats)), dtype=np.float64)
    for i, col in enumerate(lgbm_feats):
        if col in df.columns:
            X_lgbm[:, i] = df[col].ffill().fillna(0).values
    proba = lgbm.predict_proba(X_lgbm)
    p0 = proba[:, 0].astype(np.float32)   # P(SHORT)
    p2 = proba[:, 2].astype(np.float32)   # P(LONG)

    hmm = df["hmm_regime_enc"].fillna(-1).values.astype(np.int8) \
          if "hmm_regime_enc" in df.columns else np.full(n, -1, np.int8)

    y_hmm = _apply_hmm_thr(p0, p2, hmm)

    # Price data
    close = df["close"].values.astype(np.float64)
    high  = df["high"].values.astype(np.float64)
    low   = df["low"].values.astype(np.float64)
    atr   = df["atr_14_h1"].values.astype(np.float64)
    h4_sh = df["h4_swing_high"].values.astype(np.float64) \
            if "h4_swing_high" in df.columns else np.full(n, np.nan)
    h4_sl = df["h4_swing_low"].values.astype(np.float64) \
            if "h4_swing_low"  in df.columns else np.full(n, np.nan)

    # Guardian static matrix
    static_feats = [f for f in guard_feats if f not in DYNAMIC_FEATS]
    X_grd = np.zeros((n, len(static_feats)), dtype=np.float64)
    for i, col in enumerate(static_feats):
        if col in df.columns:
            X_grd[:, i] = df[col].ffill().fillna(0).values

    common = dict(
        close=close, high=high, low=low, atr=atr,
        h4_swing_highs=h4_sh, h4_swing_lows=h4_sl,
        modal=MODAL_PER_TRADE, leverage=LEVERAGE_SIM[0],
        fee_per_side=FEE_PER_SIDE, slippage=SLIPPAGE_PER_SIDE,
        max_hold=MAX_HOLDING_BARS,
        min_rr=SWING_LABEL_MIN_RR, min_tp_atr=SWING_LABEL_MIN_TP,
        max_sl_atr=SWING_LABEL_MAX_SL,
        tp_fallback_atr=TP_SL_FALLBACK_TP, sl_fallback_atr=TP_SL_FALLBACK_SL,
    )

    # Config 1: HMM only
    r_hmm = simulate_trades_swing(y_pred=y_hmm, guardian_enabled=False, **common)

    # Config 2: HMM + Guardian v2
    r_full = simulate_trades_swing(
        y_pred=y_hmm,
        guardian_enabled=True,
        guardian_model=guard,
        guardian_scaler=scaler,
        X_guardian=X_grd,
        guardian_exit_threshold=GUARDIAN_EXIT_THRESHOLD,
        guardian_min_hold_bars=GUARDIAN_MIN_HOLD_BARS,
        guardian_activation_atr=GUARDIAN_ACTIVATION_ATR,
        **common,
    )

    # Config 3: HMM + Guardian v2 + Dynamic Sizing
    modal_arr = _compute_dynamic_modal(p0, p2, hmm, y_hmm, MODAL_PER_TRADE)
    r_dyn = simulate_trades_swing(
        y_pred=y_hmm,
        guardian_enabled=True,
        guardian_model=guard,
        guardian_scaler=scaler,
        X_guardian=X_grd,
        guardian_exit_threshold=GUARDIAN_EXIT_THRESHOLD,
        guardian_min_hold_bars=GUARDIAN_MIN_HOLD_BARS,
        guardian_activation_atr=GUARDIAN_ACTIVATION_ATR,
        modal_arr=modal_arr,
        **common,
    )

    # Attach HMM state distribution for reporting
    state_dist = {int(s): int(c) for s, c in zip(*np.unique(hmm, return_counts=True))}

    return {
        "hmm":       r_hmm,
        "full":      r_full,
        "dyn":       r_dyn,
        "n_bars":    n,
        "state_dist": state_dist,
        "n_signals": int((y_hmm != FLAT).sum()),
    }


def _summarize(results_list: list) -> dict:
    """Aggregate list of per-coin simulate_trades_swing results."""
    all_trades = []
    for r in results_list:
        if r:
            all_trades.extend(r.get("trades", []))
    if not all_trades:
        return {"n": 0, "wr": 0, "pnl": 0, "ppt": 0, "pf": 0,
                "sl_pct": 0, "avg_modal": MODAL_PER_TRADE, "ppt_norm": 0}

    n     = len(all_trades)
    wins  = sum(1 for t in all_trades if t["net_pnl"] > 0)
    sl    = sum(1 for t in all_trades if t["outcome"] == "LOSS")
    gpnl  = sum(t["net_pnl"] for t in all_trades if t["net_pnl"] > 0)
    lloss = sum(abs(t["net_pnl"]) for t in all_trades if t["net_pnl"] < 0)
    tpnl  = sum(t["net_pnl"] for t in all_trades)
    pf    = gpnl / lloss if lloss > 0 else float("inf")

    modals   = [t.get("modal_used", MODAL_PER_TRADE) for t in all_trades]
    avg_m    = float(np.mean(modals))
    ppt_raw  = tpnl / n
    ppt_norm = ppt_raw * MODAL_PER_TRADE / avg_m if avg_m > 0 else 0.0

    return {
        "n": n, "wr": wins / n * 100, "pnl": round(tpnl, 2),
        "ppt": round(ppt_raw, 4), "ppt_norm": round(ppt_norm, 4),
        "pf": round(pf, 3), "sl_pct": sl / n * 100,
        "avg_modal": round(avg_m, 2),
    }


def _outcome_avg(results_list: list) -> dict:
    """Avg PnL per outcome type across all results."""
    buckets = {}
    for r in results_list:
        if not r:
            continue
        for t in r.get("trades", []):
            o = t.get("outcome", "?")
            buckets.setdefault(o, []).append(t["net_pnl"])
    return {o: (round(float(np.mean(v)), 3), len(v)) for o, v in buckets.items()}


def main():
    SEP  = "=" * 72
    SEP2 = "-" * 72

    print(f"\n{SEP}")
    print(f"  HOLDOUT GENUINE V2 — LGBM v2 + HMM Config B + Guardian v2 + DynSize")
    print(f"  Period : {OOS_START.date()} -> {OOS_END.date()}")
    print(f"{SEP}\n")

    lgbm, lgbm_feats, guard, scaler, guard_feats = load_models()
    static_feats = [f for f in guard_feats if f not in DYNAMIC_FEATS]
    print(f"  LGBM    : {len(lgbm_feats)} features")
    print(f"  Guardian: {len(static_feats)} static + {len(DYNAMIC_FEATS)} dynamic features")
    print(f"  HMM     : Config B per-state thresholds")
    print(f"  Sizing  : Dynamic (regime x confidence, clamped 0.5x-2.0x)\n")

    avail = [s for s in ALL_COINS
             if (HOLDOUT_DIR / "labeled" / f"{s}_features_v3.parquet").exists()]
    print(f"  Evaluating {len(avail)} coins...\n")

    res_hmm, res_full, res_dyn = [], [], []
    coin_rows = []

    for sym in avail:
        res = evaluate_coin(sym, lgbm, lgbm_feats, guard, scaler, guard_feats)
        if res is None:
            print(f"  {sym}: skip")
            continue

        rh = res["hmm"]; rf = res["full"]; rd = res["dyn"]
        sh = _summarize([rh]); sf = _summarize([rf]); sd = _summarize([rd])

        res_hmm.append(rh); res_full.append(rf); res_dyn.append(rd)

        print(f"  {sym:<16} "
              f"HMM={sh['n']:>4} PPT={sh['ppt']:>+.3f} | "
              f"FULL PPT={sf['ppt']:>+.3f} | "
              f"DYN PPT={sd['ppt']:>+.3f}(n={sd['ppt_norm']:>+.3f})")

        coin_rows.append({
            "sym": sym,
            "n_bars": res["n_bars"],
            "n_signals": res["n_signals"],
            "state_dist": res["state_dist"],
            "sh": sh, "sf": sf, "sd": sd,
        })

    # Global aggregates
    sh_all = _summarize(res_hmm)
    sf_all = _summarize(res_full)
    sd_all = _summarize(res_dyn)

    print(f"\n{SEP}")
    print(f"  RINGKASAN HOLDOUT — {OOS_START.date()} -> {OOS_END.date()}")
    print(SEP)
    print(f"\n  {'Config':<32} {'N':>6}  {'WR%':>5}  {'PnL':>9}  "
          f"{'PPT':>7}  {'PPT_n':>7}  {'PF':>5}  {'SL%':>5}  {'AvgM':>6}")
    print(f"  {'-'*78}")

    for name, s in [
        ("HMM Config B (no Guardian)", sh_all),
        ("HMM + Guardian v2", sf_all),
        ("HMM + Guardian v2 + DynSize", sd_all),
    ]:
        pf = f"{s['pf']:.3f}" if s["pf"] != float("inf") else "  INF"
        print(f"  {name:<32} {s['n']:>6,}  {s['wr']:>5.1f}%  "
              f"${s['pnl']:>8.2f}  {s['ppt']:>+7.4f}  {s['ppt_norm']:>+7.4f}  "
              f"{pf}  {s['sl_pct']:>5.1f}%  ${s['avg_modal']:>5.1f}")

    print(f"\n  Delta FULL vs HMM  : "
          f"WR {sf_all['wr']-sh_all['wr']:>+5.1f}pp  "
          f"PPT ${sf_all['ppt']-sh_all['ppt']:>+.4f}  "
          f"PF {sf_all['pf']-sh_all['pf']:>+.3f}  "
          f"PnL ${sf_all['pnl']-sh_all['pnl']:>+.2f}")
    print(f"  Delta DYN vs FULL  : "
          f"WR {sd_all['wr']-sf_all['wr']:>+5.1f}pp  "
          f"PPT_n ${sd_all['ppt_norm']-sf_all['ppt_norm']:>+.4f}  "
          f"PnL ${sd_all['pnl']-sf_all['pnl']:>+.2f}  "
          f"AvgModal ${sd_all['avg_modal']:>+.1f}")

    # OOF reference
    print(f"\n  OOF Reference (2020-2026-04-01):")
    print(f"    HMM Config B : N=32,727  PPT=+$0.4825  PF=2.305  WR=66.7%")
    print(f"    FULL         : N=32,727  PPT=+$0.5012  PF=2.521  WR=65.7%")
    print(f"    DYN          : N=32,727  PPT_n=+$0.5187  PF=2.521  WR=65.7%")

    # Outcome distribution
    print(f"\n{SEP}")
    print(f"  OUTCOME DISTRIBUTION")
    print(SEP)
    d_hmm  = _outcome_avg(res_hmm)
    d_full = _outcome_avg(res_full)
    d_dyn  = _outcome_avg(res_dyn)
    all_oc = sorted(set(list(d_hmm) + list(d_full) + list(d_dyn)))
    print(f"\n  {'Outcome':<24} {'HMM n':>7}  {'HMMavg':>7}  "
          f"{'FULLn':>7}  {'FULLavg':>7}  {'DYNn':>7}  {'DYNavg':>7}")
    print(f"  {'-'*75}")
    for o in all_oc:
        h_avg, h_n = d_hmm.get(o,  (0.0, 0))
        f_avg, f_n = d_full.get(o, (0.0, 0))
        d_avg, d_n = d_dyn.get(o,  (0.0, 0))
        print(f"  {o:<24} {h_n:>7,}  {h_avg:>+7.3f}  "
              f"{f_n:>7,}  {f_avg:>+7.3f}  {d_n:>7,}  {d_avg:>+7.3f}")

    # HMM state distribution
    print(f"\n{SEP}")
    print(f"  HMM STATE DISTRIBUTION (holdout period)")
    print(SEP)
    all_states = {}
    for row in coin_rows:
        for st, cnt in row["state_dist"].items():
            all_states[st] = all_states.get(st, 0) + cnt
    state_names = {0: "TRENDING_DOWN", 1: "RANGING_LOW", 2: "RANGING_HIGH",
                   3: "TRENDING_UP", -1: "unknown"}
    total_bars = sum(all_states.values())
    print(f"\n  {'State':<5} {'Name':<18} {'Count':>8}  {'Pct':>6}  {'Thr L/S':>10}")
    print(f"  {'-'*55}")
    for st in sorted(all_states.keys()):
        cnt  = all_states[st]
        pct  = cnt / total_bars * 100
        tl, ts = HMM_THR_CFG.get(int(st), (0.45, 0.45))
        print(f"  {st:<5} {state_names.get(st,'?'):<18} {cnt:>8,}  {pct:>6.1f}%  "
              f"{tl:.2f}/{ts:.2f}")

    # Dynamic sizing stats (DYN config)
    modals_dyn = []
    for r in res_dyn:
        if r:
            for t in r.get("trades", []):
                modals_dyn.append(t.get("modal_used", MODAL_PER_TRADE))
    if modals_dyn:
        marr = np.array(modals_dyn)
        bm   = MODAL_PER_TRADE
        print(f"\n  Dynamic sizing (DYN config, {len(marr)} trades):")
        print(f"    Mean modal: ${marr.mean():.2f}  Std: ${marr.std():.2f}  "
              f"Min: ${marr.min():.2f}  Max: ${marr.max():.2f}")
        buckets = [
            (0.0,    0.7*bm, "< 0.7x (reduced)"),
            (0.7*bm, 0.9*bm, "0.7-0.9x (below)"),
            (0.9*bm, 1.1*bm, "0.9-1.1x (~base)"),
            (1.1*bm, 1.5*bm, "1.1-1.5x (above)"),
            (1.5*bm, 2.1*bm, "1.5-2.0x (max)"),
        ]
        for lo, hi, label in buckets:
            cnt = int(((marr >= lo) & (marr < hi)).sum())
            print(f"    {label:<22} {cnt:>5,} ({cnt/len(marr)*100:.1f}%)")

    print(f"\n{SEP}\n")

    # Save report
    today = datetime.now().strftime("%Y-%m-%d")
    rdir  = REPORT_DIR / "experiments"
    rdir.mkdir(parents=True, exist_ok=True)
    md_path   = rdir / f"{today}_genuine_v2_holdout.md"
    json_path = rdir / f"{today}_genuine_v2_holdout.json"

    pf_full = sf_all["pf"] if sf_all["pf"] != float("inf") else "inf"
    pf_dyn  = sd_all["pf"] if sd_all["pf"] != float("inf") else "inf"

    md = f"""# Holdout Evaluation — Genuine v2
**Date**: {today}
**Stack**: {LGBM_RUN} → HMM Config B → {GUARDIAN_RUN} → Dynamic Sizing
**Period**: {OOS_START.date()} – {OOS_END.date()} ({len(coin_rows)} koin)

## Metodologi
- LGBM genuine_v2: 34 fitur, 8-fold purged CV, OOF threshold
- HMM Config B: per-state threshold (S1=0.55/0.55, S2=0.50/0.50, S3=0.45/0.50)
- Guardian v2: tight labeling (exit2_neg_pnl=0.0%), 25 fitur
- Dynamic sizing: regime × confidence multiplier, clamped [0.5x, 2.0x]
- **First look** — tidak ada tuning berdasarkan data ini

## Scorecard

| Config | N | WR% | PnL | PPT | PPT_norm | PF | SL% | AvgModal |
|--------|---|-----|-----|-----|----------|----|-----|---------|
| HMM only | {sh_all['n']:,} | {sh_all['wr']:.1f}% | ${sh_all['pnl']:.2f} | {sh_all['ppt']:+.4f} | {sh_all['ppt_norm']:+.4f} | {sh_all['pf']:.3f} | {sh_all['sl_pct']:.1f}% | ${sh_all['avg_modal']:.1f} |
| HMM + Guardian v2 | {sf_all['n']:,} | {sf_all['wr']:.1f}% | ${sf_all['pnl']:.2f} | {sf_all['ppt']:+.4f} | {sf_all['ppt_norm']:+.4f} | {pf_full} | {sf_all['sl_pct']:.1f}% | ${sf_all['avg_modal']:.1f} |
| **HMM + Guardian + DynSize** | **{sd_all['n']:,}** | **{sd_all['wr']:.1f}%** | **${sd_all['pnl']:.2f}** | **{sd_all['ppt']:+.4f}** | **{sd_all['ppt_norm']:+.4f}** | **{pf_dyn}** | **{sd_all['sl_pct']:.1f}%** | **${sd_all['avg_modal']:.1f}** |

## OOF Reference (2020–2026-04-01)

| Config | N | WR% | PPT | PPT_norm | PF |
|--------|---|-----|-----|----------|----|
| HMM only | 32,727 | 66.7% | +$0.4825 | +$0.4825 | 2.305 |
| FULL | 32,727 | 65.7% | +$0.5012 | +$0.5012 | 2.521 |
| DYN | 32,727 | 65.7% | +$0.6922 | +$0.5187 | 2.521 |

## HMM State Distribution (holdout period)
{chr(10).join(f'- S{st} {state_names.get(int(st),"?")}: {cnt:,} bars ({cnt/total_bars*100:.1f}%)' for st, cnt in sorted(all_states.items()))}

## Catatan
- PnL menggunakan ${MODAL_PER_TRADE}/trade, {LEVERAGE_SIM[0]}x leverage
- Holdout period pasar: seluruhnya HMM state 1 (RANGING_LOW) → threshold 0.55/0.55
- **Genuine OOF**: tidak ada parameter yang dipilih berdasarkan data Apr–Jun 2026 ini
- Jika tidak memuaskan: JANGAN tune → mulai riset baru dengan holdout baru (Jul–Sep 2026)

*Generated by 07_holdout_genuine_v2.py — SEKALI SAJA, tidak boleh diulang*
"""

    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md)

    with open(json_path, "w") as f:
        json.dump({
            "created":     datetime.now().isoformat(),
            "lgbm_run":    LGBM_RUN,
            "guardian_run": GUARDIAN_RUN,
            "period_start": str(OOS_START.date()),
            "period_end":   str(OOS_END.date()),
            "n_coins":     len(coin_rows),
            "scorecard_hmm":  sh_all,
            "scorecard_full": sf_all,
            "scorecard_dyn":  sd_all,
            "hmm_state_dist": all_states,
        }, f, indent=2)

    print(f"  Report : {md_path}")
    print(f"  JSON   : {json_path}")
    print(f"\n  INGAT: Set HOLDOUT_EVALUATED = True di baris 15 file ini setelah ini!\n")


if __name__ == "__main__":
    main()
