"""
pipeline/07_holdout_genuine_v2_analysis.py — Analisis Detail Holdout Genuine v2

Menjalankan ulang evaluasi HANYA untuk analisis — bukan tuning.
Parameter identik dengan 07_holdout_genuine_v2.py (sudah terkunci).
"""

import json, sys, warnings
from pathlib import Path
from datetime import datetime

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
    MODEL_DIR, HOLDOUT_DIR,
)
from core.evaluator import simulate_trades_swing
from core.utils import ensure_utc_index

LGBM_RUN     = "tb_lgbm_genuine_v2"
GUARDIAN_RUN = "tb_guardian_genuine_v2_hmm_v2"
LGBM_DIR     = MODEL_DIR / "runs" / LGBM_RUN
GUARD_DIR    = MODEL_DIR / "runs" / GUARDIAN_RUN

HMM_THR_CFG = {
    0:  (0.55, 0.55),
    1:  (0.55, 0.55),
    2:  (0.50, 0.50),
    3:  (0.45, 0.50),
    -1: (0.45, 0.45),
}
DYNAMIC_FEATS = {
    "bars_held_norm", "current_pnl_pct", "current_pnl_atr",
    "max_favorable_pnl_pct", "drawdown_from_peak_pct",
    "direction", "entry_price_ratio",
}
LONG, SHORT, FLAT = 2, 0, 1


def _apply_hmm_thr(p0, p2, hmm):
    n = len(p0)
    tl_arr = np.full(n, 0.45, dtype=np.float32)
    ts_arr = np.full(n, 0.45, dtype=np.float32)
    for state, (tl, ts) in HMM_THR_CFG.items():
        if state == -1: continue
        mask = hmm == state
        tl_arr[mask] = tl; ts_arr[mask] = ts
    y = np.ones(n, dtype=np.int32)
    y[p2 >= tl_arr] = LONG
    y[(p0 >= ts_arr) & (y != LONG)] = SHORT
    return y


def _compute_dynamic_modal(p0, p2, hmm, y_pred, base):
    n = len(p0)
    tl_arr = np.full(n, 0.45, dtype=np.float32)
    ts_arr = np.full(n, 0.45, dtype=np.float32)
    for state, (tl, ts) in HMM_THR_CFG.items():
        if state == -1: continue
        mask = hmm == state
        tl_arr[mask] = tl; ts_arr[mask] = ts
    lm = y_pred == LONG; sm = y_pred == SHORT
    conf = np.where(lm, p2, np.where(sm, p0, 0.0)).astype(np.float32)
    thr  = np.where(lm, tl_arr, ts_arr)
    c_mult = 1.0 + np.clip((conf - thr) / 0.10, 0.0, 0.5)
    r_mult = np.full(n, 0.80)
    r_mult[hmm == 0] = 0.75; r_mult[hmm == 1] = 1.0; r_mult[hmm == 2] = 1.0
    r_mult[(hmm == 3) & lm] = 1.5; r_mult[(hmm == 3) & sm] = 0.75
    modal_arr = (base * np.clip(r_mult * c_mult, 0.5, 2.0)).astype(np.float32)
    modal_arr[y_pred == FLAT] = base
    return modal_arr


def collect_trades():
    lgbm  = joblib.load(LGBM_DIR / "lgbm.pkl")
    with open(LGBM_DIR / "features.json") as f: lgbm_feats = json.load(f)
    guard  = joblib.load(GUARD_DIR / "guardian.pkl")
    scaler = joblib.load(GUARD_DIR / "guardian_scaler.pkl")
    with open(GUARD_DIR / "guardian_features.json") as f: guard_feats = json.load(f)
    static_feats = [f for f in guard_feats if f not in DYNAMIC_FEATS]

    all_trades = []   # DYN config trade records enriched with metadata

    for sym in ALL_COINS:
        path = HOLDOUT_DIR / "labeled" / f"{sym}_features_v3.parquet"
        if not path.exists(): continue

        df = pd.read_parquet(path)
        df = ensure_utc_index(df).sort_index()
        df = df[(df.index >= OOS_START) & (df.index < OOS_END)]
        if len(df) < 50: continue
        n = len(df)

        X = np.zeros((n, len(lgbm_feats)))
        for i, c in enumerate(lgbm_feats):
            if c in df.columns: X[:, i] = df[c].ffill().fillna(0).values
        proba = lgbm.predict_proba(X)
        p0 = proba[:, 0].astype(np.float32)
        p2 = proba[:, 2].astype(np.float32)

        hmm = df["hmm_regime_enc"].fillna(-1).values.astype(np.int8) \
              if "hmm_regime_enc" in df.columns else np.full(n, -1, np.int8)
        y_hmm = _apply_hmm_thr(p0, p2, hmm)

        close = df["close"].values.astype(np.float64)
        high  = df["high"].values.astype(np.float64)
        low   = df["low"].values.astype(np.float64)
        atr   = df["atr_14_h1"].values.astype(np.float64)
        h4_sh = df["h4_swing_high"].values if "h4_swing_high" in df.columns else np.full(n, np.nan)
        h4_sl = df["h4_swing_low"].values  if "h4_swing_low"  in df.columns else np.full(n, np.nan)

        X_grd = np.zeros((n, len(static_feats)))
        for i, c in enumerate(static_feats):
            if c in df.columns: X_grd[:, i] = df[c].ffill().fillna(0).values

        modal_arr = _compute_dynamic_modal(p0, p2, hmm, y_hmm, MODAL_PER_TRADE)

        res = simulate_trades_swing(
            y_pred=y_hmm, close=close, high=high, low=low, atr=atr,
            h4_swing_highs=h4_sh, h4_swing_lows=h4_sl,
            modal=MODAL_PER_TRADE, leverage=LEVERAGE_SIM[0],
            fee_per_side=FEE_PER_SIDE, slippage=SLIPPAGE_PER_SIDE,
            max_hold=MAX_HOLDING_BARS,
            min_rr=SWING_LABEL_MIN_RR, min_tp_atr=SWING_LABEL_MIN_TP,
            max_sl_atr=SWING_LABEL_MAX_SL,
            tp_fallback_atr=TP_SL_FALLBACK_TP, sl_fallback_atr=TP_SL_FALLBACK_SL,
            guardian_enabled=True,
            guardian_model=guard, guardian_scaler=scaler, X_guardian=X_grd,
            guardian_exit_threshold=GUARDIAN_EXIT_THRESHOLD,
            guardian_min_hold_bars=GUARDIAN_MIN_HOLD_BARS,
            guardian_activation_atr=GUARDIAN_ACTIVATION_ATR,
            modal_arr=modal_arr,
        )

        ts_arr = df.index
        for t in res.get("trades", []):
            bi = t.get("bar_in", 0)
            bo = t.get("bar_out", bi)
            t["coin"]      = sym
            t["ts_in"]     = ts_arr[bi]  if bi < len(ts_arr) else None
            t["ts_out"]    = ts_arr[min(bo, len(ts_arr)-1)]
            t["bars_held"] = bo - bi
            t["conf"]      = float(p2[bi]) if t["direction"] == "LONG" else float(p0[bi])
            t["hmm_state"] = int(hmm[bi])
            all_trades.append(t)

        print(f"  {sym}: {len(res.get('trades',[]))} trades")

    return all_trades


def analyze(trades: list):
    SEP  = "=" * 72
    SEP2 = "-" * 72

    df = pd.DataFrame(trades)
    df["ts_in"]  = pd.to_datetime(df["ts_in"],  utc=True)
    df["ts_out"] = pd.to_datetime(df["ts_out"], utc=True)
    df["date_in"]  = df["ts_in"].dt.date
    df["month_in"] = df["ts_in"].dt.to_period("M")
    df["week_in"]  = df["ts_in"].dt.to_period("W")
    df["is_win"]   = df["net_pnl"] > 0
    df["is_sl"]    = df["outcome"] == "LOSS"
    df["is_long"]  = df["direction"] == "LONG"
    df["modal"]    = df["modal_used"].fillna(MODAL_PER_TRADE)

    n_total  = len(df)
    n_long   = df["is_long"].sum()
    n_short  = n_total - n_long
    n_win    = df["is_win"].sum()
    n_loss   = (~df["is_win"]).sum()
    n_sl     = df["is_sl"].sum()
    wr_total = n_win / n_total * 100

    total_pnl  = df["net_pnl"].sum()
    gross_win  = df.loc[df["is_win"],  "net_pnl"].sum()
    gross_loss = abs(df.loc[~df["is_win"], "net_pnl"].sum())
    pf         = gross_win / gross_loss if gross_loss > 0 else float("inf")
    avg_win    = df.loc[df["is_win"],  "net_pnl"].mean()
    avg_loss   = df.loc[~df["is_win"], "net_pnl"].mean()
    avg_modal  = df["modal"].mean()

    period_days = (df["ts_in"].max() - df["ts_in"].min()).days + 1
    trading_days = df["date_in"].nunique()
    trades_per_day_active = n_total / trading_days if trading_days > 0 else 0
    trades_per_cal_day    = n_total / period_days  if period_days > 0 else 0

    # Consecutive streaks
    pnl_seq = df["net_pnl"].values
    max_win_streak = max_loss_streak = cur_w = cur_l = 0
    for p in pnl_seq:
        if p > 0: cur_w += 1; cur_l = 0
        else:     cur_l += 1; cur_w = 0
        max_win_streak  = max(max_win_streak,  cur_w)
        max_loss_streak = max(max_loss_streak, cur_l)

    # Daily PnL
    daily = df.groupby("date_in").agg(
        n_trades=("net_pnl", "count"),
        pnl=("net_pnl", "sum"),
        wins=("is_win", "sum"),
    ).reset_index()
    daily["wr"]  = daily["wins"] / daily["n_trades"] * 100
    daily["ppt"] = daily["pnl"] / daily["n_trades"]

    best_day_row  = daily.loc[daily["pnl"].idxmax()]
    worst_day_row = daily.loc[daily["pnl"].idxmin()]
    pos_days = (daily["pnl"] > 0).sum()
    neg_days = (daily["pnl"] <= 0).sum()

    # Max drawdown on cumulative equity
    cum_pnl = df["net_pnl"].cumsum().values
    peak    = np.maximum.accumulate(cum_pnl)
    dd      = cum_pnl - peak
    max_dd  = float(dd.min())
    max_dd_idx = int(dd.argmin())

    # ─── PRINT ───────────────────────────────────────────────────────────────

    print(f"\n{SEP}")
    print(f"  ANALISIS DETAIL HOLDOUT — Genuine v2 (DYN Config)")
    print(f"  Stack: LGBM v2 + HMM Config B + Guardian v2 + Dynamic Sizing")
    print(f"  Period: {OOS_START.date()} – {df['ts_in'].max().date()}")
    print(SEP)

    # ── 1. Overview ──────────────────────────────────────────────────────────
    print(f"\n{'-'*72}")
    print(f"  1. OVERVIEW")
    print(f"{'-'*72}")
    print(f"  Periode aktif   : {df['ts_in'].min().date()} – {df['ts_in'].max().date()}")
    print(f"  Hari kalender   : {period_days} hari")
    print(f"  Hari trading    : {trading_days} hari (ada minimal 1 trade)")
    print(f"  Total trade     : {n_total:,}")
    print(f"  Trade/hari (aktif)   : {trades_per_day_active:.1f}")
    print(f"  Trade/hari (kalender): {trades_per_cal_day:.1f}")
    print(f"  Koin aktif      : {df['coin'].nunique()} dari 21")
    print(f"  Total PnL       : ${total_pnl:+.2f}")
    print(f"  PnL/hari (aktif): ${total_pnl/trading_days:+.2f}")
    print(f"  PnL/hari (kalen): ${total_pnl/period_days:+.2f}")
    print(f"  Modal rata-rata : ${avg_modal:.2f} (base ${MODAL_PER_TRADE})")

    # ── 2. Win/Loss ───────────────────────────────────────────────────────────
    print(f"\n{'-'*72}")
    print(f"  2. WIN / LOSS")
    print(f"{'-'*72}")
    print(f"  Total WIN       : {n_win:,}  ({wr_total:.1f}%)")
    print(f"  Total LOSS      : {n_loss:,}  ({n_loss/n_total*100:.1f}%)")
    print(f"  SL Hit          : {n_sl:,}  ({n_sl/n_total*100:.1f}%)")
    print(f"  Avg PnL WIN     : ${avg_win:+.4f}")
    print(f"  Avg PnL LOSS    : ${avg_loss:+.4f}")
    print(f"  Win/Loss Ratio  : {abs(avg_win/avg_loss):.2f}x")
    print(f"  Profit Factor   : {pf:.3f}")
    print(f"  Max win streak  : {max_win_streak} trade berturut")
    print(f"  Max loss streak : {max_loss_streak} trade berturut")

    # ── 3. Long vs Short ─────────────────────────────────────────────────────
    print(f"\n{'-'*72}")
    print(f"  3. LONG vs SHORT")
    print(f"{'-'*72}")
    for direc, mask in [("LONG", df["is_long"]), ("SHORT", ~df["is_long"])]:
        sub  = df[mask]
        if len(sub) == 0: continue
        wr_d = sub["is_win"].mean() * 100
        ppt_d= sub["net_pnl"].mean()
        pnl_d= sub["net_pnl"].sum()
        sl_d = sub["is_sl"].mean() * 100
        print(f"  {direc:<6}  N={len(sub):>5,}  ({len(sub)/n_total*100:.1f}%)  "
              f"WR={wr_d:.1f}%  PPT=${ppt_d:+.4f}  "
              f"PnL=${pnl_d:+.2f}  SL%={sl_d:.1f}%")

    # ── 4. Outcome distribution ───────────────────────────────────────────────
    print(f"\n{'-'*72}")
    print(f"  4. DISTRIBUSI OUTCOME")
    print(f"{'-'*72}")
    print(f"  {'Outcome':<28} {'N':>6}  {'Pct':>5}  {'AvgPnL':>8}  {'TotalPnL':>10}  {'WR%':>5}")
    print(f"  {'-'*65}")
    for o, grp in df.groupby("outcome"):
        n_o   = len(grp)
        pct_o = n_o / n_total * 100
        avg_o = grp["net_pnl"].mean()
        tot_o = grp["net_pnl"].sum()
        wr_o  = (grp["net_pnl"] > 0).mean() * 100
        print(f"  {o:<28} {n_o:>6,}  {pct_o:>5.1f}%  ${avg_o:>+7.3f}  "
              f"${tot_o:>+9.2f}  {wr_o:>5.1f}%")

    # ── 5. Monthly breakdown ──────────────────────────────────────────────────
    print(f"\n{'-'*72}")
    print(f"  5. MONTHLY BREAKDOWN")
    print(f"{'-'*72}")
    print(f"  {'Month':<10} {'N':>6}  {'WR%':>5}  {'PnL':>9}  {'PPT':>7}  "
          f"{'PF':>5}  {'SL%':>5}  {'Long':>5}  {'Short':>5}")
    print(f"  {'-'*65}")
    for m, grp in df.groupby("month_in"):
        n_m   = len(grp)
        wr_m  = grp["is_win"].mean() * 100
        pnl_m = grp["net_pnl"].sum()
        ppt_m = grp["net_pnl"].mean()
        gw    = grp.loc[grp["is_win"],  "net_pnl"].sum()
        gl    = grp.loc[~grp["is_win"], "net_pnl"].abs().sum()
        pf_m  = gw / gl if gl > 0 else float("inf")
        sl_m  = grp["is_sl"].mean() * 100
        nl_m  = grp["is_long"].sum()
        ns_m  = n_m - nl_m
        pf_s  = f"{pf_m:.2f}" if pf_m != float("inf") else " INF"
        print(f"  {str(m):<10} {n_m:>6,}  {wr_m:>5.1f}%  "
              f"${pnl_m:>8.2f}  {ppt_m:>+7.4f}  "
              f"{pf_s}  {sl_m:>5.1f}%  {nl_m:>5,}  {ns_m:>5,}")

    # ── 6. Weekly breakdown ───────────────────────────────────────────────────
    print(f"\n{'-'*72}")
    print(f"  6. WEEKLY BREAKDOWN")
    print(f"{'-'*72}")
    print(f"  {'Week':<14} {'N':>5}  {'WR%':>5}  {'PnL':>9}  {'PPT':>7}  {'SL%':>5}")
    print(f"  {'-'*50}")
    for w, grp in df.groupby("week_in"):
        n_w   = len(grp)
        wr_w  = grp["is_win"].mean() * 100
        pnl_w = grp["net_pnl"].sum()
        ppt_w = grp["net_pnl"].mean()
        sl_w  = grp["is_sl"].mean() * 100
        print(f"  {str(w):<14} {n_w:>5,}  {wr_w:>5.1f}%  "
              f"${pnl_w:>8.2f}  {ppt_w:>+7.4f}  {sl_w:>5.1f}%")

    # ── 7. Daily stats ────────────────────────────────────────────────────────
    print(f"\n{'-'*72}")
    print(f"  7. STATISTIK HARIAN")
    print(f"{'-'*72}")
    print(f"  Hari positif (PnL > 0) : {pos_days} hari ({pos_days/len(daily)*100:.1f}%)")
    print(f"  Hari negatif (PnL <= 0): {neg_days} hari ({neg_days/len(daily)*100:.1f}%)")
    print(f"  Avg trade/hari         : {daily['n_trades'].mean():.1f}  "
          f"(min={daily['n_trades'].min()}  max={daily['n_trades'].max()})")
    print(f"  Avg PnL/hari           : ${daily['pnl'].mean():+.3f}")
    print(f"  Median PnL/hari        : ${daily['pnl'].median():+.3f}")
    print(f"  Std PnL/hari           : ${daily['pnl'].std():.3f}")
    print()
    print(f"  HARI TERBAIK:")
    print(f"    Tanggal   : {best_day_row['date_in']}")
    print(f"    N trade   : {int(best_day_row['n_trades'])}")
    print(f"    PnL       : ${best_day_row['pnl']:+.3f}")
    print(f"    WR        : {best_day_row['wr']:.1f}%")
    best_day_trades = df[df["date_in"] == best_day_row["date_in"]][
        ["coin","direction","outcome","net_pnl","bars_held"]
    ]
    for _, r in best_day_trades.iterrows():
        print(f"      {r['coin']:<16} {r['direction']:<6} {r['outcome']:<28} "
              f"${r['net_pnl']:>+7.3f}  {int(r['bars_held'])}h")
    print()
    print(f"  HARI TERBURUK:")
    print(f"    Tanggal   : {worst_day_row['date_in']}")
    print(f"    N trade   : {int(worst_day_row['n_trades'])}")
    print(f"    PnL       : ${worst_day_row['pnl']:+.3f}")
    print(f"    WR        : {worst_day_row['wr']:.1f}%")
    worst_day_trades = df[df["date_in"] == worst_day_row["date_in"]][
        ["coin","direction","outcome","net_pnl","bars_held"]
    ]
    for _, r in worst_day_trades.iterrows():
        print(f"      {r['coin']:<16} {r['direction']:<6} {r['outcome']:<28} "
              f"${r['net_pnl']:>+7.3f}  {int(r['bars_held'])}h")

    # ── 8. Top 5 best & worst trades ─────────────────────────────────────────
    print(f"\n{'-'*72}")
    print(f"  8. TOP 5 TRADE TERBAIK & TERBURUK")
    print(f"{'-'*72}")
    top5  = df.nlargest(5,  "net_pnl")[["coin","ts_in","direction","outcome","net_pnl","bars_held","modal","conf"]]
    bot5  = df.nsmallest(5, "net_pnl")[["coin","ts_in","direction","outcome","net_pnl","bars_held","modal","conf"]]
    print(f"\n  Top 5 Terbaik:")
    print(f"  {'Coin':<16} {'Entry':<20} {'Dir':<6} {'Outcome':<24} {'PnL':>8}  {'Hold':>4}  {'Modal':>6}  {'Conf':>5}")
    for _, r in top5.iterrows():
        print(f"  {r['coin']:<16} {str(r['ts_in'])[:19]:<20} {r['direction']:<6} "
              f"{r['outcome']:<24} ${r['net_pnl']:>+7.3f}  {int(r['bars_held']):>4}h  "
              f"${r['modal']:>5.1f}  {r['conf']:.3f}")
    print(f"\n  Top 5 Terburuk:")
    for _, r in bot5.iterrows():
        print(f"  {r['coin']:<16} {str(r['ts_in'])[:19]:<20} {r['direction']:<6} "
              f"{r['outcome']:<24} ${r['net_pnl']:>+7.3f}  {int(r['bars_held']):>4}h  "
              f"${r['modal']:>5.1f}  {r['conf']:.3f}")

    # ── 9. Per-coin breakdown ─────────────────────────────────────────────────
    print(f"\n{'-'*72}")
    print(f"  9. PER-KOIN BREAKDOWN")
    print(f"{'-'*72}")
    print(f"  {'Coin':<16} {'N':>5}  {'WR%':>5}  {'PPT':>7}  {'PnL':>9}  "
          f"{'PF':>5}  {'SL%':>5}  {'Long':>5}  {'Short':>5}")
    print(f"  {'-'*72}")
    coin_stats = []
    for coin, grp in df.groupby("coin"):
        n_c  = len(grp)
        wr_c = grp["is_win"].mean() * 100
        ppt_c= grp["net_pnl"].mean()
        pnl_c= grp["net_pnl"].sum()
        gw_c = grp.loc[grp["is_win"],  "net_pnl"].sum()
        gl_c = grp.loc[~grp["is_win"], "net_pnl"].abs().sum()
        pf_c = gw_c / gl_c if gl_c > 0 else float("inf")
        sl_c = grp["is_sl"].mean() * 100
        nl_c = grp["is_long"].sum()
        ns_c = n_c - nl_c
        coin_stats.append((coin, n_c, wr_c, ppt_c, pnl_c, pf_c, sl_c, nl_c, ns_c))

    coin_stats.sort(key=lambda x: -x[4])  # sort by PnL desc
    for (coin, n_c, wr_c, ppt_c, pnl_c, pf_c, sl_c, nl_c, ns_c) in coin_stats:
        pf_s = f"{pf_c:.2f}" if pf_c != float("inf") else " INF"
        print(f"  {coin:<16} {n_c:>5,}  {wr_c:>5.1f}%  {ppt_c:>+7.4f}  "
              f"${pnl_c:>8.2f}  {pf_s}  {sl_c:>5.1f}%  {nl_c:>5,}  {ns_c:>5,}")

    # ── 10. Holding time analysis ─────────────────────────────────────────────
    print(f"\n{'-'*72}")
    print(f"  10. ANALISIS HOLDING TIME")
    print(f"{'-'*72}")
    print(f"  Avg hold (semua)  : {df['bars_held'].mean():.1f} bar (jam)")
    print(f"  Median hold       : {df['bars_held'].median():.1f} jam")
    print(f"  Min hold          : {df['bars_held'].min()} jam")
    print(f"  Max hold          : {df['bars_held'].max()} jam")
    print(f"  Avg hold WIN      : {df.loc[df['is_win'],  'bars_held'].mean():.1f} jam")
    print(f"  Avg hold LOSS     : {df.loc[~df['is_win'], 'bars_held'].mean():.1f} jam")
    print()
    print(f"  Hold distribution:")
    buckets_h = [(1,4), (4,8), (8,16), (16,24), (24,49)]
    for lo, hi in buckets_h:
        mask = (df["bars_held"] >= lo) & (df["bars_held"] < hi)
        cnt  = mask.sum()
        wr_h = df.loc[mask, "is_win"].mean() * 100 if cnt > 0 else 0
        ppt_h= df.loc[mask, "net_pnl"].mean() if cnt > 0 else 0
        print(f"    {lo:>2}-{hi:>2}h  {cnt:>5,} ({cnt/n_total*100:>5.1f}%)  "
              f"WR={wr_h:.1f}%  PPT=${ppt_h:>+.3f}")

    # ── 11. Confidence distribution ───────────────────────────────────────────
    print(f"\n{'-'*72}")
    print(f"  11. DISTRIBUSI CONFIDENCE (signal LGBM)")
    print(f"{'-'*72}")
    conf_buckets = [(0.55, 0.60), (0.60, 0.65), (0.65, 0.70), (0.70, 0.75), (0.75, 1.01)]
    print(f"  {'Conf range':<14} {'N':>5}  {'WR%':>5}  {'PPT':>7}  {'AvgModal':>9}")
    print(f"  {'-'*45}")
    for lo, hi in conf_buckets:
        mask = (df["conf"] >= lo) & (df["conf"] < hi)
        cnt  = mask.sum()
        if cnt == 0: continue
        wr_c  = df.loc[mask, "is_win"].mean() * 100
        ppt_c = df.loc[mask, "net_pnl"].mean()
        m_c   = df.loc[mask, "modal"].mean()
        print(f"  {lo:.2f}-{hi:.2f}       {cnt:>5,}  {wr_c:>5.1f}%  {ppt_c:>+7.4f}  ${m_c:>8.2f}")

    # ── 12. Dynamic sizing breakdown ──────────────────────────────────────────
    print(f"\n{'-'*72}")
    print(f"  12. DYNAMIC SIZING vs HASIL")
    print(f"{'-'*72}")
    bm = MODAL_PER_TRADE
    modal_buckets = [
        (0.0,    0.9*bm, "< 0.9x (reduced)   "),
        (0.9*bm, 1.1*bm, "~1.0x (base)       "),
        (1.1*bm, 1.3*bm, "1.1-1.3x           "),
        (1.3*bm, 1.5*bm, "1.3-1.5x           "),
        (1.5*bm, 2.1*bm, "1.5-2.0x (max)     "),
    ]
    print(f"  {'Tier':<22} {'N':>5}  {'WR%':>5}  {'PPT_raw':>8}  {'PPT_norm':>9}  {'PnL':>9}")
    print(f"  {'-'*62}")
    for lo, hi, label in modal_buckets:
        mask = (df["modal"] >= lo) & (df["modal"] < hi)
        cnt  = mask.sum()
        if cnt == 0: continue
        wr_m   = df.loc[mask, "is_win"].mean() * 100
        ppt_m  = df.loc[mask, "net_pnl"].mean()
        pnl_m  = df.loc[mask, "net_pnl"].sum()
        avgm_m = df.loc[mask, "modal"].mean()
        ppt_n  = ppt_m * bm / avgm_m
        print(f"  {label} {cnt:>5,}  {wr_m:>5.1f}%  ${ppt_m:>+7.3f}  ${ppt_n:>+8.3f}  ${pnl_m:>+8.2f}")

    # ── 13. Drawdown analysis ─────────────────────────────────────────────────
    print(f"\n{'-'*72}")
    print(f"  13. DRAWDOWN ANALYSIS")
    print(f"{'-'*72}")
    print(f"  Max drawdown (dari peak) : ${abs(max_dd):.2f}")
    print(f"  Terjadi pada trade ke    : {max_dd_idx+1}")
    if max_dd_idx < len(df):
        dd_trade = df.iloc[max_dd_idx]
        print(f"  Trade tsb               : {dd_trade['coin']} {dd_trade['direction']} "
              f"{dd_trade['outcome']} ${dd_trade['net_pnl']:+.3f} "
              f"@ {str(dd_trade['ts_in'])[:19]}")

    cum_pnl_series = df["net_pnl"].cumsum()
    final_pnl = cum_pnl_series.iloc[-1]
    print(f"  Equity akhir (vs awal)   : ${final_pnl:+.2f}")
    print(f"  Recovery ratio           : {final_pnl/abs(max_dd):.2f}x (PnL/MaxDD)")

    print(f"\n{SEP}\n")


def main():
    print("\nMengumpulkan trades dari holdout...")
    trades = collect_trades()
    print(f"\nTotal: {len(trades)} trades dikumpulkan\n")
    analyze(trades)


if __name__ == "__main__":
    main()
