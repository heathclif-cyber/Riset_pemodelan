"""
Sweep LGBM x LSTM candidate domain (ic32_lstm_candidate_v2).
Coverage ~90% LGBM signal bars — hard consensus dan Mode C sekarang viable.

Mode:
  A  — LGBM only (baseline)
  HC — Hard Consensus v1 style: veto hanya jika p_opp > 0.50, boost +0.05 jika agree
  B  — LSTM veto jika has OOF + disagree; LGBM alone jika no OOF
  C  — LSTM strict: must have OOF + agree (p_dir >= thr), else skip
"""
import sys, numpy as np, pandas as pd
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

EXCLUDE = {"1000SHIBUSDT", "1000PEPEUSDT"}

# Hard Consensus params (ic32_v1 style)
CONF_ENTRY  = 0.59
AGREE_BOOST = 0.05
OPP_PEN     = 0.65
DIR_THR     = 0.35
NO_VETO_THR = 0.50

lgbm_oof = pd.read_parquet("models/runs/ic32_regime_v2/oof_predictions.parquet")
lgbm_oof = lgbm_oof[lgbm_oof["has_oof"]].copy()
lgbm_oof.index = pd.to_datetime(lgbm_oof.index, utc=True)

lstm_oof = pd.read_parquet("models/runs/ic32_lstm_candidate_v2/oof_lstm_predictions.parquet")
lstm_oof = lstm_oof[lstm_oof["has_oof"]].copy()
lstm_oof.index = pd.to_datetime(lstm_oof.index, utc=True)
lstm_oof = lstm_oof.rename(columns={"p0": "p0_lstm", "p2": "p2_lstm"})


def _sim(y_pred, df):
    n = len(df)
    h4sh = df["h4_swing_high"].values if "h4_swing_high" in df.columns else np.full(n, np.nan)
    h4sl = df["h4_swing_low"].values  if "h4_swing_low"  in df.columns else np.full(n, np.nan)
    return simulate_trades_swing(
        y_pred=y_pred, close=df["close"].values, high=df["high"].values,
        low=df["low"].values, atr=df["atr_14_h1"].values,
        h4_swing_highs=h4sh, h4_swing_lows=h4sl,
        modal=MODAL_PER_TRADE, leverage=LEVERAGE_SIM[0],
        fee_per_side=FEE_PER_SIDE, slippage=SLIPPAGE_PER_SIDE,
        max_hold=MAX_HOLDING_BARS, min_rr=SWING_LABEL_MIN_RR,
        min_tp_atr=SWING_LABEL_MIN_TP, max_sl_atr=SWING_LABEL_MAX_SL,
        tp_fallback_atr=TP_SL_FALLBACK_TP, sl_fallback_atr=TP_SL_FALLBACK_SL,
        guardian_enabled=False,
    )


def _accum(agg, r):
    t2 = r.get("total_trades", 0)
    if t2 == 0:
        return
    tlog = r.get("trade_log", [])
    lt2 = sum(1 for x in tlog if x.get("direction") == "LONG")
    st2 = sum(1 for x in tlog if x.get("direction") == "SHORT")
    wl  = r.get("win_by_class", {}).get("LONG", 0)
    ws  = r.get("win_by_class", {}).get("SHORT", 0)
    ppl = r.get("pnl_per_trade", [])
    arr = np.array([float(x) for x in ppl]) if isinstance(ppl, (list, np.ndarray)) else np.array([])
    agg["t"]   += t2;  agg["w"]   += r.get("wins", 0)
    agg["pnl"] += r.get("net_pnl_total", 0.0)
    agg["sw"]  += float(arr[arr > 0].sum()) if (arr > 0).any() else 0.0
    agg["sl"]  += float(abs(arr[arr < 0].sum())) if (arr < 0).any() else 0.0
    agg["lt"]  += lt2; agg["lw"] += round(wl * lt2)
    agg["st"]  += st2; agg["sw2"] += round(ws * st2)


def _blank():
    return dict(t=0, w=0, pnl=0.0, sw=0.0, sl=0.0, lt=0, lw=0, st=0, sw2=0,
                veto=0, boost=0, no_data=0, skipped=0)


def run(tl, ts, mode, lstm_thr=0.35):
    agg = _blank()
    for sym in sorted(lgbm_oof["coin"].unique()):
        if sym in EXCLUDE:
            continue
        fp = LABEL_DIR / f"{sym}_features_v3.parquet"
        if not fp.exists():
            continue
        df = pd.read_parquet(fp)
        df = ensure_utc_index(df).sort_index()
        df = df[df.index < TRAIN_CUTOFF_DATE]
        sl = lgbm_oof[lgbm_oof["coin"] == sym][["p0", "p2"]]
        df = df.join(sl, how="inner").dropna(subset=["p0", "p2"])
        if len(df) < 30:
            continue
        sl2 = lstm_oof[lstm_oof["coin"] == sym][["p0_lstm", "p2_lstm"]]
        df = df.join(sl2, how="left")

        n    = len(df)
        p2v  = df["p2"].values;       p0v  = df["p0"].values
        p2l  = df["p2_lstm"].values;  p0l  = df["p0_lstm"].values
        has  = df["p0_lstm"].notna().values

        y = np.full(n, 1, np.int32)

        for i in range(n):
            is_long  = p2v[i] >= tl
            is_short = p0v[i] >= ts and not is_long

            if not (is_long or is_short):
                continue

            if mode == "A":
                y[i] = 2 if is_long else 0

            elif mode == "HC":
                # Hard Consensus: veto hanya jika p_opp > NO_VETO_THR
                p_adj = p2v[i] if is_long else p0v[i]
                if has[i]:
                    agree = p2l[i] >= DIR_THR if is_long else p0l[i] >= DIR_THR
                    opp   = p0l[i] >= NO_VETO_THR if is_long else p2l[i] >= NO_VETO_THR
                    if agree:
                        p_adj += AGREE_BOOST
                        agg["boost"] += 1
                    elif opp:
                        p_adj -= OPP_PEN
                        agg["veto"] += 1
                else:
                    agg["no_data"] += 1
                if p_adj >= CONF_ENTRY:
                    y[i] = 2 if is_long else 0

            elif mode == "B":
                # LSTM veto jika has OOF + disagree; LGBM alone jika no OOF
                if has[i]:
                    agree = p2l[i] >= lstm_thr if is_long else p0l[i] >= lstm_thr
                    if agree:
                        y[i] = 2 if is_long else 0
                    else:
                        agg["veto"] += 1  # skip
                else:
                    y[i] = 2 if is_long else 0  # no OOF → trade
                    agg["no_data"] += 1

            elif mode == "C":
                # Strict: must have OOF + agree, else skip
                if has[i]:
                    agree = p2l[i] >= lstm_thr if is_long else p0l[i] >= lstm_thr
                    if agree:
                        y[i] = 2 if is_long else 0
                    else:
                        agg["skipped"] += 1
                else:
                    agg["skipped"] += 1  # no OOF → skip

        if (y != 1).sum() == 0:
            continue
        r = _sim(y, df)
        _accum(agg, r)
    return agg


def show(a, label, base=None):
    t = a["t"]; w = a["w"]; lt = a["lt"]; st = a["st"]
    wr  = w / t   if t  else 0
    wrl = a["lw"] / lt if lt else 0
    wrs = a["sw2"] / st if st else 0
    pf  = a["sw"] / a["sl"] if a["sl"] > 0 else float("inf")
    ppt = a["pnl"] / t if t else 0
    dt  = f"({(t - base['t']) / base['t'] * 100:+.1f}%)" if base else ""
    dp  = f"({(a['pnl'] - base['pnl']) / abs(base['pnl']) * 100:+.1f}%)" if base else ""
    print(f"  {label:<38} {t:>7,}{dt:>8}  {wr*100:>5.1f}  {wrl*100:>5.1f}  {wrs*100:>5.1f}  "
          f"{a['pnl']:>8.0f}{dp:>9}  {pf:>6.3f}  {ppt:>7.4f}")
    extra = []
    if a.get("veto"):   extra.append(f"veto={a['veto']:,}")
    if a.get("boost"):  extra.append(f"boost={a['boost']:,}")
    if a.get("no_data"):extra.append(f"no_data={a['no_data']:,}")
    if a.get("skipped"):extra.append(f"skipped={a['skipped']:,}")
    if extra:
        total = a["t"] + a.get("veto", 0) + a.get("skipped", 0)
        pct_filtered = 100 * (a.get("veto", 0) + a.get("skipped", 0)) / max(1, total)
        extra.append(f"filtered={pct_filtered:.1f}%")
        print(f"    >> {' | '.join(extra)}")


print("=" * 100)
print("  SWEEP LGBM x LSTM candidate domain (ic32_lstm_candidate_v2)")
print("  Coverage ~90% LGBM signal bars | OOF sim | 19 koin | no Guardian")
print("=" * 100)
hdr = (f"  {'Config':<38} {'Trades':>15}  {'WR%':>5}  {'WRL%':>5}  {'WRS%':>5}  "
       f"{'PnL':>17}  {'PF':>6}  {'PPT':>7}")
print(hdr)

configs = [(0.65, 0.60), (0.69, 0.59), (0.70, 0.65), (0.75, 0.70), (0.75, 0.75)]

print("\n--- MODE A: LGBM ONLY (baseline) ---")
baselines = {}
for tl, ts in configs:
    a = run(tl, ts, "A")
    baselines[(tl, ts)] = a
    show(a, f"LGBM {tl}/{ts} alone")

print("\n--- HARD CONSENSUS (veto jika p_opp > 0.50) ---")
for tl, ts in configs:
    a = run(tl, ts, "HC")
    show(a, f"HC  LGBM {tl}/{ts} + LSTM cand", base=baselines[(tl, ts)])

print("\n--- MODE B (LSTM veto jika disagree; LGBM alone jika no OOF) ---")
for tl, ts in [(0.69, 0.59), (0.70, 0.65), (0.75, 0.70)]:
    for lstm_thr in [0.35, 0.40, 0.45]:
        a = run(tl, ts, "B", lstm_thr)
        show(a, f"B   LGBM {tl}/{ts} LSTM>={lstm_thr}", base=baselines[(tl, ts)])
    print()

print("\n--- MODE C (strict: must have OOF + LSTM agree) ---")
for tl, ts in [(0.65, 0.60), (0.69, 0.59), (0.70, 0.65), (0.75, 0.70)]:
    for lstm_thr in [0.35, 0.40, 0.45]:
        a = run(tl, ts, "C", lstm_thr)
        show(a, f"C   LGBM {tl}/{ts} LSTM>={lstm_thr}", base=baselines[(tl, ts)])
    print()
