"""
pipeline/07_holdout_ic32_regime_v2.py — Holdout Evaluation (SEKALI SAJA)

Config freeze:
  LGBM       : ic32_regime_v2 (33 fitur, thr_long=0.75, thr_short=0.70)
  HMM filter : TREND_UP SHORT -> thr_short = 0.80 (naik dari 0.70)
  Guardian   : ic32_rv2_guard_oof v1 (exit_thr=0.90, min_hold=2)
  Period     : 2026-04-01 s.d. 2026-06-20 23:59 WITA (= 2026-06-20 15:59 UTC)

PENTING:
  - Set HOLDOUT_EVALUATED = True setelah pertama kali dijalankan
  - Jangan tune parameter apapun berdasarkan hasil ini
"""

# GUARD — ubah ke True SETELAH pertama kali dijalankan
HOLDOUT_EVALUATED = False

if HOLDOUT_EVALUATED:
    raise RuntimeError(
        "Holdout ic32_regime_v2 sudah dievaluasi!\n"
        "Jangan jalankan ulang — ini melanggar Aturan 1 (kontaminasi holdout).\n"
        "Buat script 07_holdout_ic32_regime_v2_v2.py jika perlu evaluasi baru."
    )

import json, sys, warnings, numpy as np, pandas as pd
from datetime import datetime, timezone
from pathlib import Path
warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import joblib
from core.evaluator import simulate_trades_swing
from core.utils import setup_logger, ensure_utc_index
from config import (
    ALL_COINS,
    MODAL_PER_TRADE, LEVERAGE_SIM, FEE_PER_SIDE, SLIPPAGE_PER_SIDE,
    MAX_HOLDING_BARS, SWING_LABEL_MIN_RR, SWING_LABEL_MIN_TP,
    SWING_LABEL_MAX_SL, TP_SL_FALLBACK_TP, TP_SL_FALLBACK_SL,
    MODEL_DIR, HOLDOUT_DIR, REPORT_DIR,
)

logger = setup_logger("07_holdout_ic32_regime_v2")

# === CONFIG (FREEZE — jangan ubah) ===
LGBM_RUN     = "ic32_regime_v2_parity"
GUARDIAN_RUN = "ic32_rv2_parity_guard_oof"  # parity: cvd-real + real-LSR + H4-causal

THR_LONG      = 0.75
THR_SHORT_BASE = 0.70
THR_SHORT_TREND_UP = 0.80   # HMM filter: TREND_UP SHORT diperketat

GUARDIAN_EXIT_THR  = 0.90
GUARDIAN_MIN_HOLD  = 2

# Holdout period: Apr 1 s.d. 22 Jun 2026 17:00 WITA = 22 Jun 2026 09:00 UTC
OOS_START = pd.Timestamp("2026-04-01", tz="UTC")
OOS_END   = pd.Timestamp("2026-06-22 09:00", tz="UTC")   # eksklusif, end of Jun 22 17:00 WITA

HMM_TREND_UP = 3  # state TRENDING_UP

DYNAMIC_FEATS = {
    "bars_held_norm", "current_pnl_pct", "current_pnl_atr",
    "max_favorable_pnl_pct", "drawdown_from_peak_pct",
    "direction", "entry_price_ratio",
    "lgbm_entry_conf", "mfe_atr_ratio",
}

LM = {"SHORT": 0, "FLAT": 1, "LONG": 2}


def load_models():
    lgbm_dir = MODEL_DIR / "runs" / LGBM_RUN
    gdir     = MODEL_DIR / "runs" / GUARDIAN_RUN

    lgbm_model = joblib.load(lgbm_dir / "lgbm.pkl")
    with open(lgbm_dir / "features.json") as f:
        lgbm_feats = json.load(f)

    g_model  = joblib.load(gdir / "guardian.pkl")
    g_scaler = joblib.load(gdir / "guardian_scaler.pkl")
    with open(gdir / "guardian_features.json") as f:
        g_feats = json.load(f)

    return lgbm_model, lgbm_feats, g_model, g_scaler, g_feats


def evaluate_coin(sym, lgbm_model, lgbm_feats, g_model, g_scaler, g_feats,
                  with_guardian=True):
    path = HOLDOUT_DIR / "labeled" / f"{sym}_features_v3.parquet"
    if not path.exists():
        return None

    df = pd.read_parquet(path)
    df = ensure_utc_index(df).sort_index()
    df = df[(df.index >= OOS_START) & (df.index < OOS_END)]
    if len(df) < 30:
        return None

    n = len(df)

    # LGBM inference
    X_lgbm = np.zeros((n, len(lgbm_feats)), dtype=np.float64)
    for i, col in enumerate(lgbm_feats):
        if col in df.columns:
            X_lgbm[:, i] = df[col].ffill().fillna(0).values
    proba = lgbm_model.predict_proba(X_lgbm)
    p0 = proba[:, 0]
    p2 = proba[:, 2]

    # HMM regime
    regime = df["hmm_regime_enc"].fillna(-1).astype(int).values if "hmm_regime_enc" in df.columns \
             else np.zeros(n, dtype=int)

    # Entry signal + HMM filter
    y_pred = np.full(n, LM["FLAT"], np.int32)
    y_pred[p2 >= THR_LONG] = LM["LONG"]
    for i in range(n):
        if y_pred[i] == LM["LONG"]:
            continue
        thr_s = THR_SHORT_TREND_UP if regime[i] == HMM_TREND_UP else THR_SHORT_BASE
        if p0[i] >= thr_s:
            y_pred[i] = LM["SHORT"]

    lgbm_conf = np.where(y_pred == LM["LONG"], p2,
                np.where(y_pred == LM["SHORT"], p0, 0.0))

    # H4 swing arrays
    h4_sh = df["h4_swing_high"].values if "h4_swing_high" in df.columns else np.full(n, np.nan)
    h4_sl = df["h4_swing_low"].values  if "h4_swing_low"  in df.columns else np.full(n, np.nan)

    # Guardian static features
    static_names = [f for f in g_feats if f not in DYNAMIC_FEATS]
    X_guard = np.zeros((n, len(static_names)), dtype=np.float64)
    for i, col in enumerate(static_names):
        if col in df.columns:
            X_guard[:, i] = df[col].ffill().fillna(0).values

    result = simulate_trades_swing(
        y_pred=y_pred,
        close=df["close"].values,
        high=df["high"].values,
        low=df["low"].values,
        atr=df["atr_14_h1"].values,
        h4_swing_highs=h4_sh,
        h4_swing_lows=h4_sl,
        modal=MODAL_PER_TRADE,
        leverage=LEVERAGE_SIM[0],
        fee_per_side=FEE_PER_SIDE,
        slippage=SLIPPAGE_PER_SIDE,
        max_hold=MAX_HOLDING_BARS,
        min_rr=SWING_LABEL_MIN_RR,
        min_tp_atr=SWING_LABEL_MIN_TP,
        max_sl_atr=SWING_LABEL_MAX_SL,
        tp_fallback_atr=TP_SL_FALLBACK_TP,
        sl_fallback_atr=TP_SL_FALLBACK_SL,
        guardian_enabled=with_guardian,
        guardian_model=g_model if with_guardian else None,
        guardian_scaler=g_scaler if with_guardian else None,
        X_guardian=X_guard if with_guardian else None,
        guardian_feat_cols=g_feats if with_guardian else [],
        guardian_static_names=static_names if with_guardian else [],
        lgbm_conf_arr=lgbm_conf,
        guardian_exit_threshold=GUARDIAN_EXIT_THR,
        guardian_min_hold_bars=GUARDIAN_MIN_HOLD,
    )
    return result


def aggregate(results):
    total_trades = total_wins = 0
    total_pnl = gross_profit = gross_loss = 0.0
    guardian_exits = sl_hits = time_exits = 0

    _GUARDIAN_OUTCOMES = {
        "GUARDIAN_EXIT", "GUARDIAN_FULL", "GUARDIAN_MOMENTUM_EXIT",
        "GUARDIAN_MOMENTUM_PARTIAL", "GUARDIAN_MOMENTUM_FLOOR",
        "GUARDIAN_DELTA_EXIT", "TRAILING_STOP",
    }
    _TIME_OUTCOMES = {"TIMEOUT", "TIMEOUT_MOMENTUM"}

    for res in results.values():
        if not res:
            continue
        total_trades += res.get("total_trades", 0)
        total_wins   += res.get("wins", 0)
        total_pnl    += res.get("total_pnl", 0.0)
        for t in res.get("trades", []):
            pnl = t.get("net_pnl", 0.0)
            out = t.get("outcome", "")
            if pnl > 0: gross_profit += pnl
            else:        gross_loss   += abs(pnl)
            if out == "LOSS":                  sl_hits += 1
            elif out in _GUARDIAN_OUTCOMES:    guardian_exits += 1
            elif out in _TIME_OUTCOMES:        time_exits += 1

    wr = total_wins / total_trades if total_trades else 0.0
    pf = gross_profit / gross_loss if gross_loss > 1e-9 else float("inf")
    ppt = total_pnl / total_trades if total_trades else 0.0
    period_months = (OOS_END - OOS_START).days / 30.44

    return {
        "total_trades":      total_trades,
        "wins":              total_wins,
        "wr":                round(wr, 4),
        "total_pnl":         round(total_pnl, 2),
        "pnl_per_trade":     round(ppt, 4),
        "pnl_per_month":     round(total_pnl / period_months, 2),
        "profit_factor":     round(pf, 3),
        "trades_per_month":  round(total_trades / period_months, 1),
        "sl_hits":           sl_hits,
        "sl_hit_rate":       round(sl_hits / total_trades, 4) if total_trades else 0,
        "guardian_exits":    guardian_exits,
        "guardian_exit_pct": round(guardian_exits / total_trades, 4) if total_trades else 0,
        "time_exits":        time_exits,
        "time_exit_pct":     round(time_exits / total_trades, 4) if total_trades else 0,
        "period_months":     round(period_months, 1),
    }


def main():
    sep = "=" * 70
    print(f"\n{sep}")
    print(f"  HOLDOUT — ic32_regime_v2 + HMM filter + Guardian v1")
    print(f"  Period : {OOS_START.date()} s.d. {OOS_END.date()} (~{(OOS_END-OOS_START).days} hari)")
    print(f"  LGBM   : {LGBM_RUN} | thr_long={THR_LONG} thr_short={THR_SHORT_BASE} (TU:{THR_SHORT_TREND_UP})")
    print(f"  Guard  : {GUARDIAN_RUN} | exit_thr={GUARDIAN_EXIT_THR} min_hold={GUARDIAN_MIN_HOLD}")
    print(f"{sep}\n")

    lgbm_model, lgbm_feats, g_model, g_scaler, g_feats = load_models()
    static_names = [f for f in g_feats if f not in DYNAMIC_FEATS]
    print(f"LGBM features  : {len(lgbm_feats)}")
    print(f"Guardian feats : {len(g_feats)} ({len(static_names)} static + {len(DYNAMIC_FEATS)} dynamic)")

    available = [s for s in ALL_COINS
                 if (HOLDOUT_DIR / "labeled" / f"{s}_features_v3.parquet").exists()]
    print(f"Coins          : {len(available)}\n")

    results_no_g = {}
    results_g    = {}

    for sym in available:
        r_no = evaluate_coin(sym, lgbm_model, lgbm_feats, g_model, g_scaler, g_feats, with_guardian=False)
        r_g  = evaluate_coin(sym, lgbm_model, lgbm_feats, g_model, g_scaler, g_feats, with_guardian=True)
        results_no_g[sym] = r_no
        results_g[sym]    = r_g
        if r_g:
            t = r_g.get("total_trades", 0)
            w = r_g.get("wins", 0)
            pnl = r_g.get("total_pnl", 0.0)
            wr_pct = (w / t * 100) if t else 0.0
            logger.info(f"[{sym}] {t} trades | WR={wr_pct:.1f}% | PnL=${pnl:.2f}")

    sc_no = aggregate(results_no_g)
    sc_g  = aggregate(results_g)

    print(f"\n{sep}")
    print(f"  ENTRY MODEL ONLY (no Guardian)")
    print(f"  Trades: {sc_no['total_trades']:,}  WR: {sc_no['wr']*100:.2f}%  PF: {sc_no['profit_factor']:.3f}  PnL: ${sc_no['total_pnl']:.2f}")
    print(f"\n  ENTRY + GUARDIAN")
    print(f"  Trades: {sc_g['total_trades']:,}  WR: {sc_g['wr']*100:.2f}%  PF: {sc_g['profit_factor']:.3f}  PnL: ${sc_g['total_pnl']:.2f}")
    print(f"  Guardian exit: {sc_g['guardian_exit_pct']*100:.1f}%  SL hit: {sc_g['sl_hit_rate']*100:.1f}%  Timeout: {sc_g['time_exit_pct']*100:.1f}%")
    print(f"{sep}\n")

    # Per-coin detail (sorted by PnL)
    print("Per-coin breakdown (sorted by PnL with Guardian):")
    print(f"  {'Coin':<16} Trades  WR%     PF     PnL")
    print("  " + "-" * 52)
    coin_rows = []
    for sym, res in results_g.items():
        if not res or res.get("total_trades", 0) == 0:
            continue
        t = res["total_trades"]; w = res["wins"]; pnl = res["total_pnl"]
        pf = res["profit_factor"]
        coin_rows.append((sym, t, w/t, pf, pnl))
    for sym, t, wr, pf, pnl in sorted(coin_rows, key=lambda x: -x[4]):
        print(f"  {sym:<16} {t:>5}  {wr*100:>5.1f}%  {pf:>6.3f}  ${pnl:>8.2f}")

    # Save report
    today_str   = datetime.now().strftime("%Y-%m-%d")
    report_path = REPORT_DIR / "experiments" / f"{today_str}_ic32_regime_v2_holdout.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)

    oof_ref = {"wr": 0.6732, "pf": 3.382, "trades": 12582, "pnl": 3720.0}

    md = f"""# Holdout Evaluation — ic32_regime_v2
**Date**   : {today_str}
**Models** : {LGBM_RUN} + {GUARDIAN_RUN} (v1)
**Period** : {OOS_START.date()} – {OOS_END.date()} ({sc_g['period_months']:.1f} bulan)
**Coins**  : {len(available)} koin

## Config (FREEZE)

| Parameter | Nilai |
|-----------|-------|
| thr_long | {THR_LONG} |
| thr_short (base) | {THR_SHORT_BASE} |
| thr_short TREND_UP | {THR_SHORT_TREND_UP} |
| Guardian exit_thr | {GUARDIAN_EXIT_THR} |
| Guardian min_hold | {GUARDIAN_MIN_HOLD} |

## Scorecard

### Entry Model Only (no Guardian)

| Metrik | Nilai |
|--------|-------|
| Total Trades | {sc_no['total_trades']:,} |
| Trades/bulan | {sc_no['trades_per_month']:.1f} |
| Win Rate | {sc_no['wr']*100:.2f}% |
| Profit Factor | {sc_no['profit_factor']:.3f} |
| Net PnL | ${sc_no['total_pnl']:.2f} |
| PnL/bulan | ${sc_no['pnl_per_month']:.2f} |
| PnL/trade | ${sc_no['pnl_per_trade']:.4f} |

### Entry + Guardian

| Metrik | Nilai | vs OOF |
|--------|-------|--------|
| Total Trades | {sc_g['total_trades']:,} | ref {oof_ref['trades']:,} |
| Trades/bulan | {sc_g['trades_per_month']:.1f} | |
| Win Rate | {sc_g['wr']*100:.2f}% | OOF {oof_ref['wr']*100:.2f}% |
| Profit Factor | {sc_g['profit_factor']:.3f} | OOF {oof_ref['pf']:.3f} |
| Net PnL | ${sc_g['total_pnl']:.2f} | OOF ${oof_ref['pnl']:.2f} (full period) |
| PnL/bulan | ${sc_g['pnl_per_month']:.2f} | |
| PnL/trade | ${sc_g['pnl_per_trade']:.4f} | |
| Guardian Exit % | {sc_g['guardian_exit_pct']*100:.1f}% | |
| SL Hit Rate | {sc_g['sl_hit_rate']*100:.1f}% | |
| Time Exit % | {sc_g['time_exit_pct']*100:.1f}% | |

## Metodologi

- Config FREEZE sebelum holdout — tidak ada parameter yang dituning berdasarkan data ini
- LGBM diinfer langsung (tidak retrain) pada data Apr–Jun 2026
- HMM regime dari `03e_regime_hmm_holdout.py` (out-of-sample walk-forward)
- Guardian v1 diinfer dengan exit_thr=0.90, min_hold=2 (dari OOF sweep)
- HMM filter: SHORT di TREND_UP (state 3) hanya jika p0 >= 0.80
- Data $10/trade, 5x leverage ($50 exposure per trade)

*Generated by 07_holdout_ic32_regime_v2.py — SEKALI, no tuning allowed*
"""
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"\nReport: {report_path}")

    json_path = report_path.with_suffix(".json")
    with open(json_path, "w") as f:
        json.dump({
            "created": datetime.now().isoformat(),
            "lgbm_run": LGBM_RUN, "guardian_run": GUARDIAN_RUN,
            "thr_long": THR_LONG, "thr_short_base": THR_SHORT_BASE,
            "thr_short_trend_up": THR_SHORT_TREND_UP,
            "guardian_exit_thr": GUARDIAN_EXIT_THR,
            "guardian_min_hold": GUARDIAN_MIN_HOLD,
            "period_start": str(OOS_START.date()),
            "period_end": str(OOS_END.date()),
            "scorecard_no_guardian": sc_no,
            "scorecard_with_guardian": sc_g,
        }, f, indent=2)
    print(f"JSON  : {json_path}")
    print("\nLangkah selanjutnya:")
    print("  1. Set HOLDOUT_EVALUATED = True di baris 14 file ini")
    print("  2. Jika result bagus: update widyawardhana_model.md + deploy")
    print("  3. Jika result buruk: mulai riset baru, jangan tune")


if __name__ == "__main__":
    main()
