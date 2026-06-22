"""
Sweep simulasi interaksi LGBM x LSTM gate domain (ic32_lstm_gate_v2).

Mode interaksi:
  A  — LGBM only (baseline)
  B  — LGBM fires + has LSTM OOF → LSTM agree required; else skip
       LGBM fires + no LSTM OOF  → trade (LGBM alone)
  C  — LGBM fires + has LSTM OOF + LSTM agree → trade
       LGBM fires + (no OOF OR disagree) → SKIP
"""
import json, sys
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    LABEL_DIR, TRAIN_CUTOFF_DATE,
    MODAL_PER_TRADE, LEVERAGE_SIM, FEE_PER_SIDE, SLIPPAGE_PER_SIDE,
    MAX_HOLDING_BARS, TP_SL_FALLBACK_TP, TP_SL_FALLBACK_SL,
    SWING_LABEL_MIN_RR, SWING_LABEL_MIN_TP, SWING_LABEL_MAX_SL,
)
from core.evaluator import simulate_trades_swing
from core.utils import ensure_utc_index

EXCLUDE_SIM = {"1000SHIBUSDT", "1000PEPEUSDT"}

# ── Load OOF ──────────────────────────────────────────────────────────────────
lgbm_oof = pd.read_parquet("models/runs/ic32_regime_v2/oof_predictions.parquet")
lgbm_oof = lgbm_oof[lgbm_oof["has_oof"]].copy()
lgbm_oof.index = pd.to_datetime(lgbm_oof.index, utc=True)

lstm_oof = pd.read_parquet("models/runs/ic32_lstm_gate_v2/oof_lstm_predictions.parquet")
lstm_oof = lstm_oof[lstm_oof["has_oof"]].copy()
lstm_oof.index = pd.to_datetime(lstm_oof.index, utc=True)
lstm_oof = lstm_oof.rename(columns={"p0": "p0_lstm", "p1": "p1_lstm", "p2": "p2_lstm"})


def run_sim(thr_long, thr_short, mode, lstm_thr_long=0.40, lstm_thr_short=0.40):
    agg = dict(trades=0, wins=0, pnl=0.0, sum_win=0.0, sum_loss=0.0,
               long_t=0, long_w=0, short_t=0, short_w=0)

    for sym in sorted(lgbm_oof["coin"].unique()):
        if sym in EXCLUDE_SIM:
            continue
        fp = LABEL_DIR / f"{sym}_features_v3.parquet"
        if not fp.exists():
            continue

        df = pd.read_parquet(fp)
        df = ensure_utc_index(df).sort_index()
        df = df[df.index < TRAIN_CUTOFF_DATE]

        sym_lgbm = lgbm_oof[lgbm_oof["coin"] == sym][["p0", "p2"]]
        df = df.join(sym_lgbm, how="inner").dropna(subset=["p0", "p2"])
        if len(df) < 30:
            continue

        # Join LSTM gate OOF
        sym_lstm = lstm_oof[lstm_oof["coin"] == sym][["p0_lstm", "p2_lstm"]]
        df = df.join(sym_lstm, how="left")  # NaN = bar tidak ada di gate OOF

        n = len(df)
        p2 = df["p2"].values
        p0 = df["p0"].values
        p0_lstm = df["p0_lstm"].values
        p2_lstm = df["p2_lstm"].values
        has_lstm = df["p0_lstm"].notna().values

        # LGBM signals
        y_pred = np.full(n, 1, np.int32)
        y_pred[p2 >= thr_long] = 2
        y_pred[(p0 >= thr_short) & (y_pred != 2)] = 0

        if mode == "A":
            pass  # LGBM only

        elif mode == "B":
            # has LSTM OOF → require agree; no LSTM OOF → keep LGBM
            for i in range(n):
                if y_pred[i] == 2 and has_lstm[i]:
                    if p2_lstm[i] < lstm_thr_long:
                        y_pred[i] = 1  # veto
                elif y_pred[i] == 0 and has_lstm[i]:
                    if p0_lstm[i] < lstm_thr_short:
                        y_pred[i] = 1  # veto

        elif mode == "C":
            # has LSTM OOF + agree → trade; else SKIP (even without LSTM)
            for i in range(n):
                if y_pred[i] == 2:
                    if not has_lstm[i] or p2_lstm[i] < lstm_thr_long:
                        y_pred[i] = 1
                elif y_pred[i] == 0:
                    if not has_lstm[i] or p0_lstm[i] < lstm_thr_short:
                        y_pred[i] = 1

        if (y_pred != 1).sum() == 0:
            continue

        h4_sh = df["h4_swing_high"].values if "h4_swing_high" in df.columns else np.full(n, np.nan)
        h4_sl = df["h4_swing_low"].values if "h4_swing_low" in df.columns else np.full(n, np.nan)

        r = simulate_trades_swing(
            y_pred=y_pred, close=df["close"].values, high=df["high"].values,
            low=df["low"].values, atr=df["atr_14_h1"].values,
            h4_swing_highs=h4_sh, h4_swing_lows=h4_sl,
            modal=MODAL_PER_TRADE, leverage=LEVERAGE_SIM[0],
            fee_per_side=FEE_PER_SIDE, slippage=SLIPPAGE_PER_SIDE,
            max_hold=MAX_HOLDING_BARS, min_rr=SWING_LABEL_MIN_RR,
            min_tp_atr=SWING_LABEL_MIN_TP, max_sl_atr=SWING_LABEL_MAX_SL,
            tp_fallback_atr=TP_SL_FALLBACK_TP, sl_fallback_atr=TP_SL_FALLBACK_SL,
            guardian_enabled=False,
        )
        t = r.get("total_trades", 0)
        if t == 0:
            continue

        w = r.get("wins", 0)
        tlog = r.get("trade_log", [])
        long_t  = sum(1 for x in tlog if x.get("direction") == "LONG")
        short_t = sum(1 for x in tlog if x.get("direction") == "SHORT")
        wr_l = r.get("win_by_class", {}).get("LONG", 0)
        wr_s = r.get("win_by_class", {}).get("SHORT", 0)

        ppl = r.get("pnl_per_trade", [])
        arr = np.array([float(x) for x in ppl]) if isinstance(ppl, (list, np.ndarray)) else np.array([])
        sw = float(arr[arr > 0].sum()) if (arr > 0).any() else 0.0
        sl_ = float(abs(arr[arr < 0].sum())) if (arr < 0).any() else 0.0

        agg["trades"]  += t
        agg["wins"]    += w
        agg["pnl"]     += r.get("net_pnl_total", 0.0)
        agg["sum_win"] += sw
        agg["sum_loss"]+= sl_
        agg["long_t"]  += long_t
        agg["long_w"]  += round(wr_l * long_t)
        agg["short_t"] += short_t
        agg["short_w"] += round(wr_s * short_t)

    return agg


def fmt(a):
    t = a["trades"]; w = a["wins"]
    lt = a["long_t"]; lw = a["long_w"]
    st = a["short_t"]; sw_ = a["short_w"]
    wr    = w / t if t else 0
    wr_l  = lw / lt if lt else 0
    wr_s  = sw_ / st if st else 0
    pnl   = a["pnl"]
    pf    = a["sum_win"] / a["sum_loss"] if a["sum_loss"] > 0 else float("inf")
    ppt   = pnl / t if t else 0
    return t, wr, wr_l, wr_s, pnl, pf, ppt


print("=" * 80)
print("  SWEEP LGBM x LSTM gate domain (ic32_lstm_gate_v2)")
print("  OOF sim | no Guardian | excl SHIB/PEPE | 19 koin")
print("=" * 80)

# ── Baseline LGBM only ────────────────────────────────────────────────────────
print("\n--- MODE A: LGBM ONLY (baseline) ---")
print(f"  {'Config':<18} {'T':>6} {'WR%':>6} {'WR_L%':>6} {'WR_S%':>6} {'PnL':>9} {'PF':>6} {'PPT':>7}")
for tl, ts in [(0.70, 0.65), (0.72, 0.70), (0.75, 0.70), (0.75, 0.75)]:
    a = run_sim(tl, ts, "A")
    t, wr, wrl, wrs, pnl, pf, ppt = fmt(a)
    print(f"  L={tl} S={ts}          {t:>6,} {wr*100:>6.1f} {wrl*100:>6.1f} {wrs*100:>6.1f} {pnl:>9.0f} {pf:>6.3f} {ppt:>7.4f}")

# ── Mode B: LSTM veto pada gate bars, sisanya trade ───────────────────────────
print("\n--- MODE B: LGBM + LSTM agree (gate only, LGBM-alone jika no gate) ---")
print(f"  {'Config':<28} {'T':>6} {'WR%':>6} {'WR_L%':>6} {'WR_S%':>6} {'PnL':>9} {'PF':>6} {'PPT':>7}")
for tl, ts in [(0.70, 0.65), (0.75, 0.70)]:
    for lstm_thr in [0.35, 0.40, 0.45, 0.50]:
        a = run_sim(tl, ts, "B", lstm_thr, lstm_thr)
        t, wr, wrl, wrs, pnl, pf, ppt = fmt(a)
        print(f"  LGBM {tl}/{ts} LSTM>={lstm_thr}     {t:>6,} {wr*100:>6.1f} {wrl*100:>6.1f} {wrs*100:>6.1f} {pnl:>9.0f} {pf:>6.3f} {ppt:>7.4f}")
    print()

# ── Mode C: LSTM strict — harus ada OOF dan agree ─────────────────────────────
print("\n--- MODE C: LGBM + LSTM strict (must agree, else skip) ---")
print(f"  {'Config':<28} {'T':>6} {'WR%':>6} {'WR_L%':>6} {'WR_S%':>6} {'PnL':>9} {'PF':>6} {'PPT':>7}")
for tl, ts in [(0.70, 0.65), (0.75, 0.70)]:
    for lstm_thr in [0.35, 0.40, 0.45, 0.50]:
        a = run_sim(tl, ts, "C", lstm_thr, lstm_thr)
        t, wr, wrl, wrs, pnl, pf, ppt = fmt(a)
        print(f"  LGBM {tl}/{ts} LSTM>={lstm_thr}     {t:>6,} {wr*100:>6.1f} {wrl*100:>6.1f} {wrs*100:>6.1f} {pnl:>9.0f} {pf:>6.3f} {ppt:>7.4f}")
    print()
