"""
pipeline/05h_sweep_lgbm_lstm_combined.py — Combined LGBM + LSTM OOF Threshold Sweep

Sweep kombinasi threshold LGBM dan LSTM pada OOF predictions.
Tidak menyentuh holdout — murni OOF.

Logika filter:
  - LGBM LONG  → tetap LONG  hanya jika lstm_p2 >= lstm_thr
  - LGBM SHORT → tetap SHORT hanya jika lstm_p0 >= lstm_thr
  - Bar tanpa LSTM OOF (bukan candidate domain) → FLAT

Output:
  models/runs/ic32_rv2_lstm_14f_s36_c55/combined_sweep_results.json
  models/runs/ic32_rv2_lstm_14f_s36_c55/best_thresholds_combined.json

Jalankan: python pipeline/05h_sweep_lgbm_lstm_combined.py
"""
import json, sys, itertools, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from core.evaluator import simulate_trades_swing
from core.utils import setup_logger, ensure_utc_index
from config import (
    ALL_COINS, TRAIN_CUTOFF_DATE,
    MAX_HOLDING_BARS, MODAL_PER_TRADE, LEVERAGE_SIM, FEE_PER_SIDE,
    SLIPPAGE_PER_SIDE, SWING_LABEL_MIN_RR, SWING_LABEL_MIN_TP,
    SWING_LABEL_MAX_SL, TP_SL_FALLBACK_TP, TP_SL_FALLBACK_SL,
    MODEL_DIR, LABEL_DIR,
)

logger = setup_logger("05h_sweep_lgbm_lstm_combined")

# --- Path ---
LGBM_RUN  = MODEL_DIR / "runs" / "ic32_regime_v2"
LSTM_RUN  = MODEL_DIR / "runs" / "ic32_lstm_candidate_v3_seq36"
OUT_DIR   = LSTM_RUN   # simpan di run dir LSTM

# --- Sweep grid ---
# LGBM: fokus 3 pair terbaik dari sweep sebelumnya
THR_LGBM_LONGS  = [0.70, 0.75]
THR_LGBM_SHORTS = [0.65, 0.70]
# LSTM: padat di zona bermakna (>=median 0.394, di atas random 0.333)
THR_LSTM = [0.38, 0.40, 0.42, 0.44, 0.46, 0.48, 0.50, 0.52, 0.55, 0.58, 0.60]

LM = {"SHORT": 0, "FLAT": 1, "LONG": 2}
MAX_HOLD = MAX_HOLDING_BARS
TP_ATR   = TP_SL_FALLBACK_TP
SL_ATR   = TP_SL_FALLBACK_SL
MIN_TRADES = 200


def load_feature_data(coins):
    frames = []
    for sym in coins:
        path = LABEL_DIR / f"{sym}_features_v3.parquet"
        if not path.exists():
            continue
        df = pd.read_parquet(path)
        df = ensure_utc_index(df).sort_index()
        df = df[df.index < TRAIN_CUTOFF_DATE]
        if df.empty:
            continue
        df["coin"] = sym
        frames.append(df[["coin", "close", "high", "low", "atr_14_h1",
                           "h4_swing_high", "h4_swing_low"]])
    return pd.concat(frames).sort_index()


def run_sweep(feat_df, lgbm_oof, lstm_oof):
    coins = feat_df["coin"].unique()
    results = []
    total_combos = len(THR_LGBM_LONGS) * len(THR_LGBM_SHORTS) * len(THR_LSTM)
    done = 0

    for thr_ll, thr_ls, thr_lstm in itertools.product(THR_LGBM_LONGS, THR_LGBM_SHORTS, THR_LSTM):
        agg_trades = agg_wins = agg_losses = 0
        agg_pnl = agg_gross_win = agg_gross_loss = 0.0

        for sym in coins:
            # Feature data
            sm = feat_df["coin"] == sym
            sdf = feat_df[sm].copy()
            if sdf.empty:
                continue

            # LGBM OOF untuk koin ini (join by index)
            lg = lgbm_oof[lgbm_oof["coin"] == sym][["p0", "p1", "p2", "has_oof"]]
            sdf = sdf.join(lg, how="left", rsuffix="_lgbm")
            sdf.columns = [c.replace("_lgbm", "") if c.endswith("_lgbm") else c
                           for c in sdf.columns]

            # LSTM OOF untuk koin ini (join by index)
            lt = lstm_oof[lstm_oof["coin"] == sym][["p0", "p1", "p2", "has_oof"]]
            lt.columns = ["lstm_p0", "lstm_p1", "lstm_p2", "lstm_has_oof"]
            sdf = sdf.join(lt, how="left")

            # Hanya bar dengan LGBM OOF valid
            mask = sdf["has_oof"].fillna(False)
            sdf = sdf[mask]
            if len(sdf) < 30:
                continue

            n = len(sdf)
            lgbm_p0 = sdf["p0"].values
            lgbm_p2 = sdf["p2"].values
            lstm_p0 = sdf["lstm_p0"].fillna(0).values
            lstm_p2 = sdf["lstm_p2"].fillna(0).values
            lstm_ok  = sdf["lstm_has_oof"].fillna(False).values

            # Generate LGBM signal
            y_pred = np.full(n, LM["FLAT"], np.int32)
            y_pred[lgbm_p2 >= thr_ll] = LM["LONG"]
            short_m = (lgbm_p0 >= thr_ls) & (y_pred != LM["LONG"])
            y_pred[short_m] = LM["SHORT"]

            # LSTM filter: bar tanpa LSTM OOF → FLAT; bar dengan LSTM → filter
            for i in range(n):
                if y_pred[i] == LM["FLAT"]:
                    continue
                if not lstm_ok[i]:
                    y_pred[i] = LM["FLAT"]   # tidak ada LSTM OOF → drop
                    continue
                if y_pred[i] == LM["LONG"]  and lstm_p2[i] < thr_lstm:
                    y_pred[i] = LM["FLAT"]
                if y_pred[i] == LM["SHORT"] and lstm_p0[i] < thr_lstm:
                    y_pred[i] = LM["FLAT"]

            if (y_pred != LM["FLAT"]).sum() == 0:
                continue

            h4_sh = sdf["h4_swing_high"].values if "h4_swing_high" in sdf.columns else np.full(n, np.nan)
            h4_sl = sdf["h4_swing_low"].values  if "h4_swing_low"  in sdf.columns else np.full(n, np.nan)

            res = simulate_trades_swing(
                y_pred=y_pred,
                close=sdf["close"].values,
                high=sdf["high"].values,
                low=sdf["low"].values,
                atr=sdf["atr_14_h1"].values,
                h4_swing_highs=h4_sh,
                h4_swing_lows=h4_sl,
                modal=MODAL_PER_TRADE,
                leverage=LEVERAGE_SIM[0],
                fee_per_side=FEE_PER_SIDE,
                slippage=SLIPPAGE_PER_SIDE,
                max_hold=MAX_HOLD,
                min_rr=SWING_LABEL_MIN_RR,
                min_tp_atr=SWING_LABEL_MIN_TP,
                max_sl_atr=SWING_LABEL_MAX_SL,
                tp_fallback_atr=TP_ATR,
                sl_fallback_atr=SL_ATR,
                guardian_enabled=False,
            )
            t = res.get("total_trades", 0)
            w = res.get("wins", 0)
            l = res.get("losses", 0)
            pnl = res.get("total_pnl", 0.0)
            pf  = res.get("profit_factor", 0.0)

            # Reconstruct gross win/loss untuk aggregate PF
            if t > 0 and pf > 0 and l > 0:
                # pf = gross_win / gross_loss → gross_win = pf * gross_loss
                # pnl = gross_win - gross_loss → gross_loss = pnl / (pf - 1) jika pf != 1
                if abs(pf - 1.0) > 1e-6:
                    gl = abs(pnl / (pf - 1.0))
                    gw = pf * gl
                else:
                    gw = gl = abs(pnl) / 2
                agg_gross_win  += gw
                agg_gross_loss += gl

            agg_trades += t
            agg_wins   += w
            agg_losses += l
            agg_pnl    += pnl

        if agg_trades < MIN_TRADES:
            done += 1
            continue

        wr  = agg_wins / agg_trades
        ppt = agg_pnl / agg_trades
        pf_agg = (agg_gross_win / agg_gross_loss) if agg_gross_loss > 1e-9 else 0.0

        results.append({
            "lgbm_thr_long":  round(thr_ll, 2),
            "lgbm_thr_short": round(thr_ls, 2),
            "lstm_thr":       round(thr_lstm, 2),
            "trades":         agg_trades,
            "wr":             round(wr, 4),
            "pf":             round(pf_agg, 4),
            "pnl":            round(agg_pnl, 2),
            "pnl_per_trade":  round(ppt, 4),
        })
        done += 1
        if done % 20 == 0:
            logger.info(f"  Sweep {done}/{total_combos} ...")

    results.sort(key=lambda x: x["pnl_per_trade"], reverse=True)
    return results


def main():
    logger.info("="*70)
    logger.info("  LGBM + LSTM Combined OOF Threshold Sweep")
    logger.info(f"  LGBM run : {LGBM_RUN.name}")
    logger.info(f"  LSTM run : {LSTM_RUN.name}")
    logger.info(f"  Grid     : {len(THR_LGBM_LONGS)}x{len(THR_LGBM_SHORTS)} LGBM x {len(THR_LSTM)} LSTM = "
                f"{len(THR_LGBM_LONGS)*len(THR_LGBM_SHORTS)*len(THR_LSTM)} combinations")
    logger.info("="*70)

    # Load OOF
    logger.info("Loading OOF predictions...")
    lgbm_oof = pd.read_parquet(LGBM_RUN / "oof_predictions.parquet")
    lstm_oof = pd.read_parquet(LSTM_RUN / "oof_lstm_predictions.parquet")
    logger.info(f"  LGBM OOF: {len(lgbm_oof):,} bars ({lgbm_oof['has_oof'].sum():,} valid)")
    logger.info(f"  LSTM OOF: {len(lstm_oof):,} bars ({lstm_oof['has_oof'].sum():,} valid)")

    # Load feature data
    logger.info("Loading feature data...")
    feat_df = load_feature_data(ALL_COINS)
    logger.info(f"  Feature data: {len(feat_df):,} bars, {feat_df['coin'].nunique()} coins")

    # Baseline: LGBM only (best threshold dari sweep lama)
    with open(LGBM_RUN / "best_thresholds.json") as f:
        lgbm_best = json.load(f)
    logger.info(f"\n  BASELINE LGBM-only: thr_long={lgbm_best['thr_long']} "
                f"thr_short={lgbm_best['thr_short']} | "
                f"WR={lgbm_best['oof_wr']:.4f} trades={lgbm_best['oof_trades']:,} "
                f"PnL={lgbm_best['oof_pnl']:.2f}")

    # Run sweep
    logger.info("\nRunning combined sweep...")
    results = run_sweep(feat_df, lgbm_oof, lstm_oof)

    if not results:
        logger.error("No valid combinations found!")
        return

    # Display top 20
    logger.info(f"\n  Top 20 combinations (sorted by PnL/trade):")
    logger.info(f"  {'LG_L':>5} {'LG_S':>5} {'LS_T':>5} | {'Trades':>7} {'WR':>6} {'PF':>6} {'PnL/T':>7} {'PnL':>10}")
    logger.info("  " + "-"*65)
    for r in results[:20]:
        logger.info(
            f"  {r['lgbm_thr_long']:>5.2f} {r['lgbm_thr_short']:>5.2f} {r['lstm_thr']:>5.2f} | "
            f"{r['trades']:>7,} {r['wr']:>6.4f} {r['pf']:>6.3f} {r['pnl_per_trade']:>7.4f} {r['pnl']:>10.2f}"
        )

    best = results[0]
    logger.info(f"\n  BEST COMBINED: lgbm_long={best['lgbm_thr_long']} "
                f"lgbm_short={best['lgbm_thr_short']} lstm_thr={best['lstm_thr']}")
    logger.info(f"    Trades={best['trades']:,} WR={best['wr']:.4f} PF={best['pf']:.3f} "
                f"PnL/trade={best['pnl_per_trade']:.4f} PnL={best['pnl']:.2f}")

    # Save
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    sweep_path = OUT_DIR / "combined_sweep_results_v2.json"
    with open(sweep_path, "w") as f:
        json.dump({"created": datetime.now().isoformat(), "results": results}, f, indent=2)
    logger.info(f"\n  Saved: {sweep_path}")

    best_path = OUT_DIR / "best_thresholds_combined_v2.json"
    with open(best_path, "w") as f:
        json.dump({
            "lgbm_thr_long":  best["lgbm_thr_long"],
            "lgbm_thr_short": best["lgbm_thr_short"],
            "lstm_thr":       best["lstm_thr"],
            "oof_trades":     best["trades"],
            "oof_wr":         best["wr"],
            "oof_pf":         best["pf"],
            "oof_pnl":        best["pnl"],
            "pnl_per_trade":  best["pnl_per_trade"],
            "baseline_lgbm_only": {
                "thr_long":     lgbm_best["thr_long"],
                "thr_short":    lgbm_best["thr_short"],
                "oof_wr":       lgbm_best["oof_wr"],
                "oof_trades":   lgbm_best["oof_trades"],
                "oof_pnl":      lgbm_best["oof_pnl"],
                "pnl_per_trade": lgbm_best["pnl_per_trade"],
            },
            "created": datetime.now().isoformat(),
        }, f, indent=2)
    logger.info(f"  Saved: {best_path}")
    logger.info("\nDONE.")


if __name__ == "__main__":
    main()
