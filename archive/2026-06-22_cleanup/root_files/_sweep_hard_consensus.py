"""
Hard Consensus sweep — LGBM primary, LSTM gate sebagai survival filter.
Mereplikasi logic ic32_v1 cascade config:
  - confidence_threshold_entry = 0.59
  - lstm_adjust_agree_boost    = 0.05
  - lstm_adjust_opposite_pen   = 0.65  (kurang dari lgbm_conf)
  - lstm_directional_review_threshold = 0.35  (agree threshold)
  - lstm_no_veto_threshold     = 0.50  (veto hanya jika p_opp > 0.5)
  - no Guardian
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

CONF_ENTRY   = 0.59
AGREE_BOOST  = 0.05
OPP_PEN      = 0.65
DIR_THR      = 0.35
NO_VETO_THR  = 0.50

lgbm_oof = pd.read_parquet("models/runs/ic32_regime_v2/oof_predictions.parquet")
lgbm_oof = lgbm_oof[lgbm_oof["has_oof"]].copy()
lgbm_oof.index = pd.to_datetime(lgbm_oof.index, utc=True)

lstm_oof = pd.read_parquet("models/runs/ic32_lstm_candidate_v2/oof_lstm_predictions.parquet")
lstm_oof = lstm_oof[lstm_oof["has_oof"]].copy()
lstm_oof.index = pd.to_datetime(lstm_oof.index, utc=True)
lstm_oof = lstm_oof.rename(columns={"p0": "p0_lstm", "p2": "p2_lstm"})


def _simulate(y_pred, df):
    n = len(df)
    h4sh = df["h4_swing_high"].values if "h4_swing_high" in df.columns else np.full(n, np.nan)
    h4sl = df["h4_swing_low"].values  if "h4_swing_low"  in df.columns else np.full(n, np.nan)
    return simulate_trades_swing(
        y_pred=y_pred,
        close=df["close"].values, high=df["high"].values,
        low=df["low"].values, atr=df["atr_14_h1"].values,
        h4_swing_highs=h4sh, h4_swing_lows=h4sl,
        modal=MODAL_PER_TRADE, leverage=LEVERAGE_SIM[0],
        fee_per_side=FEE_PER_SIDE, slippage=SLIPPAGE_PER_SIDE,
        max_hold=MAX_HOLDING_BARS, min_rr=SWING_LABEL_MIN_RR,
        min_tp_atr=SWING_LABEL_MIN_TP, max_sl_atr=SWING_LABEL_MAX_SL,
        tp_fallback_atr=TP_SL_FALLBACK_TP, sl_fallback_atr=TP_SL_FALLBACK_SL,
        guardian_enabled=False,
    )


def run_lgbm_only(tl, ts):
    agg = dict(t=0, w=0, pnl=0.0, sw=0.0, sl=0.0, lt=0, lw=0, st=0, sw2=0)
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
        n = len(df)
        y = np.full(n, 1, np.int32)
        y[df["p2"].values >= tl] = 2
        y[(df["p0"].values >= ts) & (y != 2)] = 0
        if (y != 1).sum() == 0:
            continue
        r = _simulate(y, df)
        _accum(agg, r)
    return agg


def run_hard_consensus(tl, ts):
    agg = dict(t=0, w=0, pnl=0.0, sw=0.0, sl=0.0, lt=0, lw=0, st=0, sw2=0,
               veto_l=0, veto_s=0, boost_l=0, boost_s=0, no_data=0)
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
        n = len(df)
        p2v = df["p2"].values
        p0v = df["p0"].values
        p2l = df["p2_lstm"].values
        p0l = df["p0_lstm"].values
        has = df["p0_lstm"].notna().values

        y = np.full(n, 1, np.int32)
        for i in range(n):
            if p2v[i] >= tl:
                p_adj = p2v[i]
                if has[i]:
                    if p2l[i] >= DIR_THR:
                        p_adj += AGREE_BOOST
                        agg["boost_l"] += 1
                    elif p0l[i] >= NO_VETO_THR:
                        p_adj -= OPP_PEN
                        agg["veto_l"] += 1
                else:
                    agg["no_data"] += 1
                if p_adj >= CONF_ENTRY:
                    y[i] = 2
            elif p0v[i] >= ts:
                p_adj = p0v[i]
                if has[i]:
                    if p0l[i] >= DIR_THR:
                        p_adj += AGREE_BOOST
                        agg["boost_s"] += 1
                    elif p2l[i] >= NO_VETO_THR:
                        p_adj -= OPP_PEN
                        agg["veto_s"] += 1
                else:
                    agg["no_data"] += 1
                if p_adj >= CONF_ENTRY:
                    y[i] = 0

        if (y != 1).sum() == 0:
            continue
        r = _simulate(y, df)
        _accum(agg, r)
    return agg


def _accum(agg, r):
    t2 = r.get("total_trades", 0)
    if t2 == 0:
        return
    tlog = r.get("trade_log", [])
    lt2 = sum(1 for x in tlog if x.get("direction") == "LONG")
    st2 = sum(1 for x in tlog if x.get("direction") == "SHORT")
    wl = r.get("win_by_class", {}).get("LONG", 0)
    ws = r.get("win_by_class", {}).get("SHORT", 0)
    ppl = r.get("pnl_per_trade", [])
    arr = np.array([float(x) for x in ppl]) if isinstance(ppl, (list, np.ndarray)) else np.array([])
    agg["t"]   += t2
    agg["w"]   += r.get("wins", 0)
    agg["pnl"] += r.get("net_pnl_total", 0.0)
    agg["sw"]  += float(arr[arr > 0].sum()) if (arr > 0).any() else 0.0
    agg["sl"]  += float(abs(arr[arr < 0].sum())) if (arr < 0).any() else 0.0
    agg["lt"]  += lt2
    agg["lw"]  += round(wl * lt2)
    agg["st"]  += st2
    agg["sw2"] += round(ws * st2)


def show(a, label, base=None):
    t = a["t"]; w = a["w"]; lt = a["lt"]; st = a["st"]
    wr  = w / t  if t  else 0
    wrl = a["lw"] / lt if lt else 0
    wrs = a["sw2"] / st if st else 0
    pf  = a["sw"] / a["sl"] if a["sl"] > 0 else float("inf")
    ppt = a["pnl"] / t if t else 0
    dt = f"({(t - base['t']) / base['t'] * 100:+.1f}%)" if base else ""
    dp = f"({(a['pnl'] - base['pnl']) / abs(base['pnl']) * 100:+.1f}%)" if base else ""
    print(f"  {label:<32} {t:>7,}{dt:>8}  {wr*100:>5.1f}  {wrl*100:>5.1f}  {wrs*100:>5.1f}  "
          f"{a['pnl']:>8.0f}{dp:>9}  {pf:>6.3f}  {ppt:>7.4f}")
    if "veto_l" in a:
        vl = a["veto_l"]; vs = a["veto_s"]
        bl = a["boost_l"]; bs = a["boost_s"]
        nd = a["no_data"]
        total_lgbm = t + vl + vs
        print(f"    >> veto_L={vl:,}  veto_S={vs:,}  boost_L={bl:,}  boost_S={bs:,}  "
              f"no_data={nd:,}  total_lgbm~{total_lgbm:,}  veto_rate={100*(vl+vs)/max(1,total_lgbm):.1f}%")


print("=" * 95)
print("  HARD CONSENSUS — ic32_v1 style | LSTM: ic32_lstm_candidate_v2")
print("  conf_entry=0.59 | agree_boost=+0.05 | opp_pen=-0.65 | dir_thr=0.35 | no_veto=0.50")
print("  Tanpa Guardian | OOF sim | 19 koin")
print("=" * 95)
hdr = f"  {'Config':<36} {'Trades':>15}  {'WR%':>5}  {'WRL%':>5}  {'WRS%':>5}  {'PnL':>17}  {'PF':>6}  {'PPT':>7}"
print(hdr)
print()

configs = [(0.65, 0.60), (0.69, 0.59), (0.70, 0.65), (0.75, 0.70), (0.75, 0.75)]

print("--- BASELINE (LGBM only) ---")
baselines = {}
for tl, ts in configs:
    a = run_lgbm_only(tl, ts)
    baselines[(tl, ts)] = a
    show(a, f"LGBM {tl}/{ts} alone")

print()
print("--- HARD CONSENSUS (LSTM candidate survival filter) ---")
for tl, ts in configs:
    a = run_hard_consensus(tl, ts)
    show(a, f"HC  LGBM {tl}/{ts} + LSTM cand", base=baselines[(tl, ts)])
    print()

# Mode C: strict agree (harus ada OOF + LSTM agree)
print("--- MODE C: STRICT AGREE (must have OOF + LSTM p_dir >= 0.35) ---")
print("  (untuk referensi: coverage ~90% LGBM signal bars)")
for tl, ts in [(0.69, 0.59), (0.70, 0.65), (0.75, 0.70)]:
    bl = baselines[(tl, ts)]
    # Jalankan Mode C via run_hard_consensus dengan no_veto_thr=0.35 (LSTM harus agree, bukan hanya tidak veto)
    # Sebenarnya ini Mode C: if LGBM fires + has OOF + LSTM agree → trade, else skip
    pass  # akan dilakukan inline
print("  (gunakan sweep terpisah jika diperlukan)")
