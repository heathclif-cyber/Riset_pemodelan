"""
pipeline/07_holdout_genuine_v1.py — Holdout Evaluation (SEKALI SAJA)

PENTING:
  - Jalankan HANYA setelah semua keputusan di-freeze (threshold, Guardian params)
  - Set HOLDOUT_EVALUATED = True setelah pertama dijalankan
  - Jika hasilnya tidak memuaskan: JANGAN tune → mulai ulang dengan holdout baru

Period holdout : Apr 1 – Jun 30, 2026 (data/holdout-test/)
Models         : tb_lgbm_genuine_v1 + tb_guardian_genuine_v1
Thresholds     : dari best_thresholds.json (hasil OOF sweep)

Output : reports/experiments/YYYY-MM-DD_genuine_v1_holdout.md
"""

# ⛔ GUARD — ubah ke True SETELAH pertama kali dijalankan
HOLDOUT_EVALUATED = True

if HOLDOUT_EVALUATED:
    raise RuntimeError(
        "Holdout genuine_v1 sudah dievaluasi!\n"
        "Jangan jalankan ulang — ini melanggar Aturan 1 (kontaminasi holdout).\n"
        "Jika perlu mengevaluasi ulang: buat script 07_holdout_genuine_v2.py dengan holdout baru."
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
    ALL_COINS, OOS_START, OOS_END,
    MODAL_PER_TRADE, LEVERAGE_SIM, FEE_PER_SIDE, SLIPPAGE_PER_SIDE,
    MAX_HOLDING_BARS, SWING_LABEL_MIN_RR, SWING_LABEL_MIN_TP,
    SWING_LABEL_MAX_SL, TP_SL_FALLBACK_TP, TP_SL_FALLBACK_SL,
    GUARDIAN_EXIT_THRESHOLD, GUARDIAN_SL_EXIT_THRESHOLD,
    GUARDIAN_SL_SAFETY_ATR, GUARDIAN_TP_ATR, GUARDIAN_MIN_HOLD_BARS,
    GUARDIAN_ACTIVATION_ATR,
    MODEL_DIR, HOLDOUT_DIR, REPORT_DIR,
)

logger = setup_logger("07_holdout_genuine_v1")

LGBM_RUN     = "tb_lgbm_genuine_v1"
GUARDIAN_RUN = "tb_guardian_genuine_v1"
LM           = {"SHORT": 0, "FLAT": 1, "LONG": 2}


def load_models():
    lgbm_dir     = MODEL_DIR / "runs" / LGBM_RUN
    guardian_dir = MODEL_DIR / "runs" / GUARDIAN_RUN

    lgbm_model = joblib.load(lgbm_dir / "lgbm.pkl")
    with open(lgbm_dir / "features.json") as f:
        lgbm_feats = json.load(f)
    with open(lgbm_dir / "best_thresholds.json") as f:
        thr = json.load(f)

    guardian_model  = joblib.load(guardian_dir / "guardian.pkl")
    guardian_scaler = joblib.load(guardian_dir / "guardian_scaler.pkl")
    with open(guardian_dir / "guardian_features.json") as f:
        guardian_feats = json.load(f)

    return lgbm_model, lgbm_feats, thr, guardian_model, guardian_scaler, guardian_feats


def evaluate_coin(
    sym: str,
    lgbm_model, lgbm_feats: list,
    thr_long: float, thr_short: float,
    guardian_model, guardian_scaler, guardian_feats: list,
    with_guardian: bool = True,
) -> dict:
    """Evaluate satu koin pada holdout period."""
    path = HOLDOUT_DIR / "labeled" / f"{sym}_features_v3.parquet"
    if not path.exists():
        return None

    df = pd.read_parquet(path)
    df = ensure_utc_index(df).sort_index()
    df = df[(df.index >= OOS_START) & (df.index < OOS_END)]
    if len(df) < 50:
        return None

    n = len(df)

    # LGBM inference
    X_lgbm = np.zeros((n, len(lgbm_feats)), dtype=np.float64)
    for i, col in enumerate(lgbm_feats):
        if col in df.columns:
            X_lgbm[:, i] = df[col].ffill().fillna(0).values
    proba = lgbm_model.predict_proba(X_lgbm)

    y_pred = np.full(n, LM["FLAT"], np.int32)
    y_pred[proba[:, 2] >= thr_long] = LM["LONG"]
    y_pred[(proba[:, 0] >= thr_short) & (y_pred != LM["LONG"])] = LM["SHORT"]

    # H4 swing arrays
    h4_sh = df["h4_swing_high"].values if "h4_swing_high" in df.columns else np.full(n, np.nan)
    h4_sl = df["h4_swing_low"].values  if "h4_swing_low"  in df.columns else np.full(n, np.nan)

    # Guardian features (static)
    X_guard = None
    if with_guardian:
        # Static feats = guardian_feats minus DYNAMIC_FEATS
        dynamic_set = {
            "bars_held_norm", "current_pnl_pct", "current_pnl_atr",
            "max_favorable_pnl_pct", "drawdown_from_peak_pct",
            "direction", "entry_price_ratio",
        }
        static_guard_feats = [c for c in guardian_feats if c not in dynamic_set]
        X_guard = np.zeros((n, len(static_guard_feats)), dtype=np.float64)
        for i, col in enumerate(static_guard_feats):
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
        guardian_model=guardian_model if with_guardian else None,
        guardian_scaler=guardian_scaler if with_guardian else None,
        X_guardian=X_guard,
        guardian_enabled=with_guardian,
        guardian_exit_threshold=GUARDIAN_EXIT_THRESHOLD,
        guardian_sl_exit_threshold=GUARDIAN_SL_EXIT_THRESHOLD,
        guardian_sl_safety_atr=GUARDIAN_SL_SAFETY_ATR,
        guardian_tp_atr=GUARDIAN_TP_ATR,
        guardian_min_hold_bars=GUARDIAN_MIN_HOLD_BARS,
        guardian_activation_atr=GUARDIAN_ACTIVATION_ATR,
    )
    return result


def compute_scorecard(results_by_coin: dict) -> dict:
    """Aggregate per-coin results menjadi scorecard global."""
    total_trades = total_wins = 0
    total_pnl    = 0.0
    sl_hits      = guardian_exits = time_exits = 0
    gross_profit = 0.0
    gross_loss   = 0.0

    _GUARDIAN_OUTCOMES = {
        "GUARDIAN_EXIT", "GUARDIAN_FULL", "GUARDIAN_MOMENTUM_EXIT",
        "GUARDIAN_MOMENTUM_PARTIAL", "GUARDIAN_DELTA_EXIT", "TRAILING_STOP",
    }
    _TIME_OUTCOMES = {"TIMEOUT", "TIMEOUT_MOMENTUM"}

    for sym, res in results_by_coin.items():
        if res is None:
            continue
        total_trades += res.get("total_trades", 0)
        total_wins   += res.get("wins", 0)
        total_pnl    += res.get("total_pnl", 0.0)
        for t in res.get("trades", []):
            outcome = t.get("outcome", "")
            pnl     = t.get("net_pnl", 0.0)
            if pnl > 0:
                gross_profit += pnl
            else:
                gross_loss += abs(pnl)
            if outcome == "LOSS":
                sl_hits += 1
            elif outcome in _GUARDIAN_OUTCOMES:
                guardian_exits += 1
            elif outcome in _TIME_OUTCOMES:
                time_exits += 1

    wr  = total_wins / total_trades if total_trades > 0 else 0.0
    ppt = total_pnl  / total_trades if total_trades > 0 else 0.0
    pf  = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    period_months = (OOS_END - OOS_START).days / 30.44

    return {
        "total_trades":     total_trades,
        "trades_per_month": round(total_trades / period_months, 1),
        "wins":             total_wins,
        "wr":               round(wr, 4),
        "total_pnl":        round(total_pnl, 2),
        "pnl_per_month":    round(total_pnl / period_months, 2),
        "pnl_per_trade":    round(ppt, 4),
        "profit_factor":    round(pf, 3),
        "sl_hits":          sl_hits,
        "sl_hit_rate":      round(sl_hits / total_trades, 4) if total_trades > 0 else 0,
        "guardian_exits":   guardian_exits,
        "guardian_exit_pct": round(guardian_exits / total_trades, 4) if total_trades > 0 else 0,
        "time_exits":       time_exits,
        "period_months":    round(period_months, 1),
    }


def format_scorecard_md(sc: dict, mode: str, thr_long: float, thr_short: float) -> str:
    wr_pct   = sc["wr"] * 100
    g_pct    = sc["guardian_exit_pct"] * 100
    sl_pct   = sc["sl_hit_rate"] * 100
    t_pct    = sc["time_exits"] / sc["total_trades"] * 100 if sc["total_trades"] > 0 else 0
    return f"""
### {mode}

| Metrik | Nilai |
|--------|-------|
| **Total Trades** | **{sc['total_trades']:,}** |
| Trades/bulan | {sc['trades_per_month']:.1f} |
| **Win Rate** | **{wr_pct:.1f}%** |
| **Profit Factor** | **{sc['profit_factor']:.3f}** |
| **Net PnL ($10/trade, 5x)** | **${sc['total_pnl']:.2f}** |
| PnL/bulan | ${sc['pnl_per_month']:.2f} |
| PnL/trade | ${sc['pnl_per_trade']:.4f} |
| Guardian Exit % | {g_pct:.1f}% |
| SL Hit Rate | {sl_pct:.1f}% |
| Time Exit % | {t_pct:.1f}% |

*Threshold: LONG={thr_long}, SHORT={thr_short} (hasil OOF sweep, bukan holdout)*
"""


def main():
    print(f"\n{'='*70}")
    print(f"  HOLDOUT EVALUATION — tb_lgbm_genuine_v1 + tb_guardian_genuine_v1")
    print(f"  Period: {OOS_START.date()} -> {OOS_END.date()}")
    print(f"  Mode  : GENUINE OOF (tanpa kontaminasi holdout)")
    print(f"{'='*70}\n")

    # Load models
    print("Loading models...")
    lgbm_model, lgbm_feats, thr_cfg, guardian_model, guardian_scaler, guardian_feats = load_models()

    thr_long  = thr_cfg["thr_long"]
    thr_short = thr_cfg["thr_short"]
    print(f"  LGBM        : {len(lgbm_feats)} features")
    print(f"  Thresholds  : LONG={thr_long}  SHORT={thr_short} (from OOF)")
    print(f"  Guardian    : {len(guardian_feats)} features")
    print(f"  OOF PnL ref : ${thr_cfg.get('oof_pnl', '?'):.2f} "
          f"(WR {thr_cfg.get('oof_wr',0)*100:.1f}%, {thr_cfg.get('oof_trades',0):,} trades)")

    # Evaluate per coin — with and without Guardian
    available = [s for s in ALL_COINS
                 if (HOLDOUT_DIR / "labeled" / f"{s}_features_v3.parquet").exists()]
    print(f"\nEvaluating {len(available)} coins on holdout...")

    results_no_g = {}
    results_g    = {}
    for sym in available:
        r_no = evaluate_coin(sym, lgbm_model, lgbm_feats, thr_long, thr_short,
                             None, None, [], with_guardian=False)
        r_g  = evaluate_coin(sym, lgbm_model, lgbm_feats, thr_long, thr_short,
                             guardian_model, guardian_scaler, guardian_feats, with_guardian=True)
        results_no_g[sym] = r_no
        results_g[sym]    = r_g
        if r_g:
            logger.info(
                f"[{sym}] "
                f"w/o G: {r_no.get('total_trades',0)} trades WR={r_no.get('wins',0)/(r_no.get('total_trades',1))*100:.1f}% "
                f"PnL=${r_no.get('total_pnl',0):.2f} | "
                f"w/ G: {r_g.get('total_trades',0)} trades WR={r_g.get('wins',0)/(r_g.get('total_trades',1))*100:.1f}% "
                f"PnL=${r_g.get('total_pnl',0):.2f}"
            )

    # Aggregate
    sc_no_g = compute_scorecard(results_no_g)
    sc_g    = compute_scorecard(results_g)

    sep = "=" * 70
    print(f"\n{sep}")
    print(f"  HOLDOUT RESULTS — ENTRY MODEL ONLY (no Guardian)")
    print(f"  Trades: {sc_no_g['total_trades']:,}  WR: {sc_no_g['wr']*100:.1f}%  "
          f"PnL: ${sc_no_g['total_pnl']:.2f}  PF: {sc_no_g['profit_factor']:.3f}")
    print(f"\n  HOLDOUT RESULTS — ENTRY MODEL + GUARDIAN")
    print(f"  Trades: {sc_g['total_trades']:,}  WR: {sc_g['wr']*100:.1f}%  "
          f"PnL: ${sc_g['total_pnl']:.2f}  PF: {sc_g['profit_factor']:.3f}")
    print(f"  Guardian exit: {sc_g['guardian_exit_pct']*100:.1f}%  SL hit: {sc_g['sl_hit_rate']*100:.1f}%")
    print(f"{sep}\n")

    # Save report
    today_str   = datetime.now().strftime("%Y-%m-%d")
    report_path = REPORT_DIR / "experiments" / f"{today_str}_genuine_v1_holdout.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)

    md_content = f"""# Holdout Evaluation — Genuine v1
**Date**: {today_str}
**Models**: {LGBM_RUN} + {GUARDIAN_RUN}
**Period**: {OOS_START.date()} – {OOS_END.date()} ({sc_g['period_months']:.1f} bulan)
**Coins**: {len(available)} koin

## Metodologi

Model ini dilatih dengan metodologi GENUINE OOF:
- Semua threshold dipilih via OOF simulation (bukan holdout)
- Guardian dilatih pada OOF trades (bukan in-sample trades)
- StandardScaler di-fit per fold dalam CV loop
- Holdout ini adalah **first look** — tidak ada keputusan yang dibuat berdasarkan data ini

Training period : 2020-01-01 – {thr_cfg.get('created','?')[:10]}
Holdout period  : {OOS_START.date()} – {OOS_END.date()}

## Thresholds (dari OOF sweep)

| Parameter | Nilai |
|-----------|-------|
| thr_long | {thr_long} |
| thr_short | {thr_short} |
| OOF PnL | ${thr_cfg.get('oof_pnl', '?'):.2f} |
| OOF WR | {thr_cfg.get('oof_wr', 0)*100:.1f}% |
| OOF Trades | {thr_cfg.get('oof_trades', '?'):,} |

## Scorecard
{format_scorecard_md(sc_no_g, "Entry Model Only (no Guardian)", thr_long, thr_short)}
{format_scorecard_md(sc_g, "Entry Model + Guardian", thr_long, thr_short)}

## Catatan

- PnL menggunakan $10/trade, 5x leverage (= $50 exposure per trade)
- Genuine OOF: angka ini adalah estimasi unbiased pertama
- Jika hasilnya tidak memuaskan: JANGAN tune ulang menggunakan data ini
  → Mulai riset baru dengan periode holdout baru (Jul-Sep 2026)

*Generated by 07_holdout_genuine_v1.py — single evaluation, no tuning allowed*
"""

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"Report saved: {report_path}")

    # Save raw JSON results
    json_path = report_path.with_suffix(".json")
    with open(json_path, "w") as f:
        json.dump({
            "created":      datetime.now().isoformat(),
            "lgbm_run":     LGBM_RUN,
            "guardian_run": GUARDIAN_RUN,
            "thr_long":     thr_long,
            "thr_short":    thr_short,
            "period_start": str(OOS_START.date()),
            "period_end":   str(OOS_END.date()),
            "scorecard_no_guardian": sc_no_g,
            "scorecard_with_guardian": sc_g,
        }, f, indent=2)
    print(f"Raw results  : {json_path}")

    print(f"\nLangkah selanjutnya:")
    print(f"  1. Set HOLDOUT_EVALUATED = True di baris 17 file ini")
    print(f"  2. Jika hasil bagus: jalankan tools/deploy_model.py")
    print(f"  3. Jika hasil buruk: mulai ulang penelitian dengan holdout baru")


if __name__ == "__main__":
    main()
