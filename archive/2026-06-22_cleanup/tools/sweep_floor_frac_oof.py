"""
tools/sweep_floor_frac_oof.py — OOF sweep GUARDIAN_MOMENTUM_FLOOR_FRAC

Mencari nilai optimal floor_frac berdasarkan OOF predictions (bukan holdout).
Tuning di OOF wajib per Aturan 1 — jangan gunakan holdout sebagai development set.

Penggunaan:
    python tools/sweep_floor_frac_oof.py

Output: tabel WR / PF / PnL / avg_win / avg_loss / floor_exit_pct per nilai frac.
"""

import json, sys, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

from pathlib import Path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import joblib
from core.evaluator import simulate_trades_swing
from core.utils import setup_logger, ensure_utc_index
from config import (
    ALL_COINS, MODEL_DIR, LABEL_DIR,
    MODAL_PER_TRADE, LEVERAGE_SIM, FEE_PER_SIDE, SLIPPAGE_PER_SIDE,
    MAX_HOLDING_BARS, SWING_LABEL_MIN_RR, SWING_LABEL_MIN_TP,
    SWING_LABEL_MAX_SL, TP_SL_FALLBACK_TP, TP_SL_FALLBACK_SL,
)

logger = setup_logger("sweep_floor_frac")

# === Sweep config ===
FLOOR_FRACS = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.70, 0.80, 0.90]

LGBM_RUN     = "ic32_regime_v2"
GUARDIAN_RUN = "ic32_rv2_guard_oof_v3mom"

THR_LONG           = 0.75
THR_SHORT_BASE     = 0.70
THR_SHORT_TREND_UP = 0.80
HMM_TREND_UP       = 3

GUARDIAN_EXIT_THR = 0.90
GUARDIAN_MIN_HOLD = 2

DYNAMIC_FEATS = {
    "bars_held_norm", "current_pnl_pct", "current_pnl_atr",
    "max_favorable_pnl_pct", "drawdown_from_peak_pct",
    "direction", "entry_price_ratio",
    "lgbm_entry_conf", "mfe_atr_ratio",
}

LM = {"SHORT": 0, "FLAT": 1, "LONG": 2}

_GUARDIAN_OUTCOMES = {
    "GUARDIAN_EXIT", "GUARDIAN_FULL", "GUARDIAN_MOMENTUM_EXIT",
    "GUARDIAN_MOMENTUM_PARTIAL", "GUARDIAN_MOMENTUM_FLOOR",
    "GUARDIAN_DELTA_EXIT", "TRAILING_STOP",
}
_TIME_OUTCOMES = {"TIMEOUT", "TIMEOUT_MOMENTUM"}


def load_models():
    lgbm_dir = MODEL_DIR / "runs" / LGBM_RUN
    gdir     = MODEL_DIR / "runs" / GUARDIAN_RUN

    g_model  = joblib.load(gdir / "guardian.pkl")
    g_scaler = joblib.load(gdir / "guardian_scaler.pkl")
    with open(gdir / "guardian_features.json") as f:
        g_feats = json.load(f)

    return g_model, g_scaler, g_feats


def load_oof_predictions():
    path = MODEL_DIR / "runs" / LGBM_RUN / "oof_predictions.parquet"
    df = pd.read_parquet(path)
    return df[df["has_oof"] == True].copy()


def simulate_coin(sym, df_feat, oof_coin, g_model, g_scaler, g_feats, floor_frac):
    """Jalankan simulasi satu koin dengan floor_frac tertentu (OOF bars saja)."""
    # Align ke bar yang punya OOF prediction
    oof_idx = oof_coin.index
    df_feat = df_feat[df_feat.index.isin(oof_idx)].copy()
    oof_coin = oof_coin[oof_coin.index.isin(df_feat.index)].copy()

    # Sort & align
    df_feat  = df_feat.sort_index()
    oof_coin = oof_coin.sort_index()

    if len(df_feat) < 30:
        return None

    n = len(df_feat)

    p0 = oof_coin["p0"].values
    p2 = oof_coin["p2"].values

    # Regime filter
    regime = df_feat["hmm_regime_enc"].fillna(-1).astype(int).values \
             if "hmm_regime_enc" in df_feat.columns else np.zeros(n, dtype=int)

    # Entry signals + HMM filter
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
    h4_sh = df_feat["h4_swing_high"].values if "h4_swing_high" in df_feat.columns \
            else np.full(n, np.nan)
    h4_sl = df_feat["h4_swing_low"].values  if "h4_swing_low"  in df_feat.columns \
            else np.full(n, np.nan)

    # Guardian static features
    static_names = [f for f in g_feats if f not in DYNAMIC_FEATS]
    X_guard = np.zeros((n, len(static_names)), dtype=np.float64)
    for i, col in enumerate(static_names):
        if col in df_feat.columns:
            X_guard[:, i] = df_feat[col].ffill().fillna(0).values

    result = simulate_trades_swing(
        y_pred=y_pred,
        close=df_feat["close"].values,
        high=df_feat["high"].values,
        low=df_feat["low"].values,
        atr=df_feat["atr_14_h1"].values,
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
        guardian_enabled=True,
        guardian_model=g_model,
        guardian_scaler=g_scaler,
        X_guardian=X_guard,
        guardian_feat_cols=g_feats,
        guardian_static_names=static_names,
        lgbm_conf_arr=lgbm_conf,
        guardian_exit_threshold=GUARDIAN_EXIT_THR,
        guardian_min_hold_bars=GUARDIAN_MIN_HOLD,
        guardian_momentum_floor_frac=floor_frac,
    )
    return result


def aggregate_results(all_results):
    total_trades = total_wins = 0
    total_pnl = gross_profit = gross_loss = 0.0
    floor_exits = guardian_exits = sl_hits = time_exits = 0
    wins_pnl = []
    losses_pnl = []

    for res in all_results:
        if not res:
            continue
        total_trades += res.get("total_trades", 0)
        total_wins   += res.get("wins", 0)
        total_pnl    += res.get("total_pnl", 0.0)

        for t in res.get("trades", []):
            pnl = t.get("net_pnl", 0.0)
            out = t.get("outcome", "")

            if pnl > 0:
                gross_profit += pnl
                wins_pnl.append(pnl)
            else:
                gross_loss += abs(pnl)
                losses_pnl.append(pnl)

            if out == "GUARDIAN_MOMENTUM_FLOOR":
                floor_exits += 1
            elif out == "LOSS":
                sl_hits += 1
            elif out in _GUARDIAN_OUTCOMES - {"GUARDIAN_MOMENTUM_FLOOR"}:
                guardian_exits += 1
            elif out in _TIME_OUTCOMES:
                time_exits += 1

    wr  = total_wins / total_trades if total_trades else 0.0
    pf  = gross_profit / gross_loss if gross_loss > 1e-9 else float("inf")
    ppt = total_pnl / total_trades  if total_trades else 0.0
    avg_win  = np.mean(wins_pnl)   if wins_pnl   else 0.0
    avg_loss = np.mean(losses_pnl) if losses_pnl else 0.0
    floor_pct = floor_exits / total_trades if total_trades else 0.0

    return {
        "trades":      total_trades,
        "wr":          round(wr * 100, 2),
        "pnl":         round(total_pnl, 2),
        "pf":          round(pf, 3),
        "ppt":         round(ppt, 4),
        "avg_win":     round(avg_win, 4),
        "avg_loss":    round(avg_loss, 4),
        "floor_pct":   round(floor_pct * 100, 1),
        "sl_hits":     sl_hits,
        "time_exits":  time_exits,
        "guard_exits": guardian_exits,
    }


def main():
    sep = "=" * 72
    print(f"\n{sep}")
    print("  OOF SWEEP — GUARDIAN_MOMENTUM_FLOOR_FRAC")
    print(f"  LGBM  : {LGBM_RUN}")
    print(f"  Guard : {GUARDIAN_RUN}")
    print(f"  Frac  : {FLOOR_FRACS}")
    print(f"{sep}\n")

    g_model, g_scaler, g_feats = load_models()
    oof_all = load_oof_predictions()
    print(f"OOF rows  : {len(oof_all):,} ({oof_all['coin'].nunique()} koin)")

    # Pre-load semua feature parquet (satu kali, dipakai ulang per sweep)
    coin_data = {}
    for sym in ALL_COINS:
        feat_path = LABEL_DIR / f"{sym}_features_v3.parquet"
        if not feat_path.exists():
            continue
        df = pd.read_parquet(feat_path)
        df = ensure_utc_index(df).sort_index()
        oof_coin = oof_all[oof_all["coin"] == sym]
        if len(oof_coin) < 30:
            continue
        coin_data[sym] = (df, oof_coin)

    print(f"Koin valid : {len(coin_data)}\n")

    # Sweep
    rows = []
    for frac in FLOOR_FRACS:
        all_results = []
        for sym, (df_feat, oof_coin) in coin_data.items():
            r = simulate_coin(sym, df_feat.copy(), oof_coin.copy(),
                              g_model, g_scaler, g_feats, frac)
            all_results.append(r)

        agg = aggregate_results(all_results)
        agg["floor_frac"] = frac
        rows.append(agg)
        print(f"  frac={frac:.2f}  trades={agg['trades']:5d}  "
              f"WR={agg['wr']:5.2f}%  PF={agg['pf']:.3f}  "
              f"PnL=${agg['pnl']:8.2f}  "
              f"avg_win=${agg['avg_win']:.4f}  avg_loss=${agg['avg_loss']:.4f}  "
              f"floor%={agg['floor_pct']:.1f}%")

    # Tabel ringkasan
    print(f"\n{sep}")
    print("  RINGKASAN SWEEP")
    print(f"{sep}")
    print(f"  {'frac':>6}  {'trades':>7}  {'WR%':>6}  {'PF':>6}  "
          f"{'PnL':>9}  {'PPT':>7}  {'avg_win':>8}  {'avg_loss':>9}  {'floor%':>7}")
    print("  " + "-" * 68)
    for r in rows:
        marker = " <-- BEST PF" if r["pf"] == max(x["pf"] for x in rows) else ""
        print(f"  {r['floor_frac']:>6.2f}  {r['trades']:>7d}  {r['wr']:>6.2f}  "
              f"{r['pf']:>6.3f}  {r['pnl']:>9.2f}  {r['ppt']:>7.4f}  "
              f"{r['avg_win']:>8.4f}  {r['avg_loss']:>9.4f}  "
              f"{r['floor_pct']:>6.1f}%{marker}")

    best_pf  = max(rows, key=lambda x: x["pf"])
    best_pnl = max(rows, key=lambda x: x["pnl"])
    best_wr  = max(rows, key=lambda x: x["wr"])
    print(f"\n  Best PF    : floor_frac={best_pf['pf_frac'] if 'pf_frac' in best_pf else best_pf['floor_frac']:.2f} "
          f"-> PF {best_pf['pf']:.3f}")
    print(f"  Best PnL   : floor_frac={best_pnl['floor_frac']:.2f} -> PnL ${best_pnl['pnl']:.2f}")
    print(f"  Best WR    : floor_frac={best_wr['floor_frac']:.2f} -> WR {best_wr['wr']:.2f}%")

    # Saran tuning
    print(f"\n  Catatan: frac lebih rendah = floor lebih loose = ride lebih jauh.")
    print(f"  floor_pct tinggi (>50%) = floor dominasi exit = avg_win tertekan.")
    print(f"  Pilih frac yang balance PF + avg_win tanpa korbankan WR drastis.")
    print(f"{sep}\n")

    # Simpan hasil CSV ke models/runs untuk referensi
    out_dir = MODEL_DIR / "runs" / GUARDIAN_RUN
    out_path = out_dir / "floor_frac_oof_sweep.csv"
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"  Hasil disimpan: {out_path}\n")


if __name__ == "__main__":
    main()
