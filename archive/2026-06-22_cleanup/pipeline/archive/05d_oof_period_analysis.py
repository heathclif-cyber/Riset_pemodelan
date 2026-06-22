"""
pipeline/05d_oof_period_analysis.py -- Analisis OOF genuine_v2 per Periode & Regime

Tujuan:
  Paparan komprehensif kelebihan/kekurangan model genuine_v2:
  - Performance per kuartal (2020 Q1 -- 2026 Q1)
  - Performance per HMM regime (TRENDING_DOWN/RANGING_LOW/RANGING_HIGH/TRENDING_UP)
  - Performance LONG vs SHORT
  - Performance per koin (top/bottom 5)
  - Statistik trade: trades/hari, avg hold, SL rate, timeout rate

Tidak ada model yang dilatih. Tidak menyentuh holdout.
Input: OOF predictions genuine_v2 + feature parquets (training data saja).
"""
import sys, warnings, json
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from config import (
    ALL_COINS, TRAIN_CUTOFF_DATE, LABEL_DIR, MODEL_DIR,
    MODAL_PER_TRADE, LEVERAGE_SIM, FEE_PER_SIDE, SLIPPAGE_PER_SIDE,
    SWING_LABEL_MIN_RR, SWING_LABEL_MIN_TP, SWING_LABEL_MAX_SL,
    TP_SL_FALLBACK_TP, TP_SL_FALLBACK_SL, MAX_HOLDING_BARS,
)
from core.evaluator import simulate_trades_swing
from core.utils import ensure_utc_index

LGBM_RUN = "tb_lgbm_genuine_v2"
RUN_DIR   = MODEL_DIR / "runs" / LGBM_RUN

# Threshold dari OOF sweep
with open(RUN_DIR / "best_thresholds.json") as f:
    _thr = json.load(f)
THR_LONG  = _thr["thr_long"]
THR_SHORT = _thr["thr_short"]

HMM_NAMES = {0: "TRENDING_DN", 1: "RANGING_LOW", 2: "RANGING_HIGH", 3: "TRENDING_UP"}
BAR_MIN   = 15   # base TF dalam menit

# ─── Helpers ─────────────────────────────────────────────────────────────────

def _is_sl(outcome): return outcome == "LOSS"
def _is_to(outcome): return outcome in {"TIMEOUT", "TIMEOUT_MOMENTUM"}
def _is_g(outcome):  return "GUARDIAN" in outcome or "TRAILING" in outcome
def _is_win(outcome): return outcome == "WIN"


def _stats(trades: list, trading_days: float) -> dict:
    if not trades:
        return {k: 0 for k in ["n", "wr", "long_wr", "short_wr", "pnl", "ppt",
                                "tpd", "sl_pct", "to_pct", "g_pct", "avg_hold_h"]}
    n = len(trades)
    wins        = sum(1 for t in trades if _is_win(t["outcome"]))
    sl_hits     = sum(1 for t in trades if _is_sl(t["outcome"]))
    to_hits     = sum(1 for t in trades if _is_to(t["outcome"]))
    g_hits      = sum(1 for t in trades if _is_g(t["outcome"]))
    total_pnl   = sum(t["net_pnl"] for t in trades)
    longs       = [t for t in trades if t["direction"] == "LONG"]
    shorts      = [t for t in trades if t["direction"] == "SHORT"]
    l_wr = sum(1 for t in longs  if _is_win(t["outcome"])) / len(longs)  * 100 if longs  else 0
    s_wr = sum(1 for t in shorts if _is_win(t["outcome"])) / len(shorts) * 100 if shorts else 0
    avg_hold  = np.mean([t["bar_out"] - t["bar_in"] for t in trades]) * BAR_MIN / 60
    tpd = n / trading_days if trading_days > 0 else 0
    return {
        "n":          n,
        "n_long":     len(longs),
        "n_short":    len(shorts),
        "wr":         wins / n * 100,
        "long_wr":    l_wr,
        "short_wr":   s_wr,
        "pnl":        total_pnl,
        "ppt":        total_pnl / n,
        "tpd":        tpd,
        "sl_pct":     sl_hits / n * 100,
        "to_pct":     to_hits / n * 100,
        "g_pct":      g_hits  / n * 100,
        "avg_hold_h": avg_hold,
    }


def _pf(trades: list) -> float:
    gross_profit = sum(t["net_pnl"] for t in trades if t["net_pnl"] > 0)
    gross_loss   = sum(abs(t["net_pnl"]) for t in trades if t["net_pnl"] < 0)
    return gross_profit / gross_loss if gross_loss > 0 else float("inf")


# ─── Load & Simulate ─────────────────────────────────────────────────────────

def load_coin_trades(sym: str, oof_pred_df: pd.DataFrame) -> list:
    path = LABEL_DIR / f"{sym}_features_v3.parquet"
    if not path.exists():
        return []

    df = pd.read_parquet(path)
    df = ensure_utc_index(df).sort_index()
    df = df[df.index < TRAIN_CUTOFF_DATE]
    if df.empty:
        return []

    # Computed features (same as 04c)
    for feat, shifts in [("ret_7d", 672), ("ret_14d", 1344), ("ret_30d", 2880)]:
        if feat not in df.columns:
            df[feat] = np.log(df["close"] / df["close"].shift(shifts).replace(0, np.nan))
    for feat, col in [("dist_pdh", "PDH"), ("dist_pdl", "PDL")]:
        if feat not in df.columns and col in df.columns:
            df[feat] = df["close"] / df[col].replace(0, np.nan) - 1

    # Join OOF predictions
    sym_oof = oof_pred_df[oof_pred_df["coin"] == sym]
    sym_oof = sym_oof[sym_oof["has_oof"] == True][["p0", "p2"]]
    proba   = sym_oof.reindex(df.index)
    has_oof = proba["p0"].notna()

    df_oof  = df[has_oof].copy()
    p0      = proba["p0"][has_oof].values
    p2      = proba["p2"][has_oof].values
    n       = len(df_oof)

    if n < 30:
        return []

    y_pred = np.full(n, 1, np.int32)
    y_pred[p2 >= THR_LONG] = 2
    y_pred[(p0 >= THR_SHORT) & (y_pred != 2)] = 0

    h4_sh = df_oof["h4_swing_high"].values if "h4_swing_high" in df_oof.columns \
            else np.full(n, np.nan)
    h4_sl = df_oof["h4_swing_low"].values if "h4_swing_low" in df_oof.columns \
            else np.full(n, np.nan)

    result = simulate_trades_swing(
        y_pred=y_pred,
        close=df_oof["close"].values,
        high=df_oof["high"].values,
        low=df_oof["low"].values,
        atr=df_oof["atr_14_h1"].values,
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
        guardian_enabled=False,
    )

    # Annotate trades with timestamp, HMM regime, coin
    hmm_arr = df_oof["hmm_regime_enc"].values if "hmm_regime_enc" in df_oof.columns \
              else np.full(n, -1)
    ts_arr  = df_oof.index

    annotated = []
    for t in result.get("trades", []):
        bi = t["bar_in"]
        if bi >= n:
            continue
        t["coin"]    = sym
        t["ts"]      = ts_arr[bi]
        t["quarter"] = ts_arr[bi].to_period("Q")
        t["month"]   = ts_arr[bi].to_period("M")
        t["year"]    = ts_arr[bi].year
        t["hmm"]     = int(hmm_arr[bi]) if not np.isnan(hmm_arr[bi]) else -1
        annotated.append(t)

    return annotated


# ─── Print helpers ────────────────────────────────────────────────────────────

SEP  = "=" * 78
SEP2 = "-" * 78

def _hdr(s): print(f"\n{SEP}\n  {s}\n{SEP}")
def _sub(s): print(f"\n{s}\n{SEP2}")

def _row(label, s, trading_days=None):
    pf_val = _pf(s["_raw"]) if "_raw" in s else None
    pf_str = f"{pf_val:.2f}" if pf_val and pf_val != float("inf") else ("INF" if pf_val else "  -")
    td_str = f"{s['tpd']:.1f}" if trading_days is not None else "   -"
    print(
        f"  {label:<16} {s['n']:>6} ({s['n_long']:>5}L/{s['n_short']:>5}S) "
        f"WR={s['wr']:>5.1f}% (L={s['long_wr']:>5.1f}%/S={s['short_wr']:>5.1f}%) "
        f"PnL={s['pnl']:>8.1f}  PPT={s['ppt']:>+6.3f}  "
        f"PF={pf_str:>5}  TPD={td_str}  "
        f"Hold={s['avg_hold_h']:.1f}h  "
        f"SL={s['sl_pct']:.0f}%/TO={s['to_pct']:.0f}%/G={s['g_pct']:.0f}%"
    )


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    # Load OOF
    oof_pred_df = pd.read_parquet(RUN_DIR / "oof_predictions.parquet")

    print(f"\n{SEP}")
    print(f"  OOF PERIOD ANALYSIS -- tb_lgbm_genuine_v2")
    print(f"  Threshold: LONG={THR_LONG}  SHORT={THR_SHORT}")
    print(f"  Training period: 2020-01-01 -> {TRAIN_CUTOFF_DATE.date()}")
    print(f"  Modal=${MODAL_PER_TRADE:.0f} x {LEVERAGE_SIM[0]:.0f}x leverage = ${MODAL_PER_TRADE*LEVERAGE_SIM[0]:.0f} exposure")
    print(f"{SEP}\n")

    print("Loading & simulating trades per coin...")
    all_trades = []
    for sym in ALL_COINS:
        trades = load_coin_trades(sym, oof_pred_df)
        all_trades.extend(trades)
        print(f"  [{sym}] {len(trades)} trades")

    print(f"\n  Total trades: {len(all_trades):,}")

    if not all_trades:
        print("Tidak ada trades! Cek OOF predictions.")
        return

    # ── 1. Overall Summary ────────────────────────────────────────────────────
    _hdr("1. RINGKASAN KESELURUHAN")
    total_days = (TRAIN_CUTOFF_DATE - pd.Timestamp("2020-01-01", tz="UTC")).days
    ov = _stats(all_trades, total_days)
    ov["_raw"] = all_trades
    print(f"  Total trades      : {ov['n']:,}  ({ov['n_long']:,} LONG / {ov['n_short']:,} SHORT)")
    print(f"  Win Rate          : {ov['wr']:.2f}%  (LONG {ov['long_wr']:.1f}% / SHORT {ov['short_wr']:.1f}%)")
    print(f"  Total PnL         : ${ov['pnl']:,.2f}")
    print(f"  PnL per trade     : ${ov['ppt']:+.4f}")
    print(f"  Profit Factor     : {_pf(all_trades):.3f}")
    print(f"  Trades per day    : {ov['tpd']:.2f}")
    print(f"  Avg hold          : {ov['avg_hold_h']:.2f} h")
    print(f"  SL hit rate       : {ov['sl_pct']:.1f}%")
    print(f"  Timeout rate      : {ov['to_pct']:.1f}%")
    print(f"  Guardian exit rate: {ov['g_pct']:.1f}%")

    # ── 2. Quarterly Breakdown ────────────────────────────────────────────────
    _hdr("2. PERFORMANCE PER KUARTAL")
    print(f"  {'Kuartal':<16} {'Trades':>6} (L/S)             "
          f"{'WR%':>5}  (L%/S%)              {'PnL':>8}  "
          f"{'PPT':>7}  {'PF':>5}  {'T/D':>4}  Hold  SL/TO/G%")
    print(SEP2)

    quarters = sorted(set(t["quarter"] for t in all_trades))
    q_summary = []
    for q in quarters:
        q_trades = [t for t in all_trades if t["quarter"] == q]
        q_days   = len(set(t["ts"].date() for t in q_trades))
        s        = _stats(q_trades, q_days)
        s["_raw"] = q_trades
        s["quarter"] = str(q)
        q_summary.append(s)
        _row(str(q), s, q_days)

    # Flag best / worst quarters
    sorted_q = sorted(q_summary, key=lambda x: x["pnl"], reverse=True)
    print(f"\n  Top-3 kuartal (by PnL)  : {', '.join(x['quarter'] for x in sorted_q[:3])}")
    print(f"  Bot-3 kuartal (by PnL)  : {', '.join(x['quarter'] for x in sorted_q[-3:])}")

    # ── 3. Yearly Breakdown ───────────────────────────────────────────────────
    _hdr("3. PERFORMANCE PER TAHUN")
    print(f"  {'Tahun':<16} {'Trades':>6} (L/S)             "
          f"{'WR%':>5}  (L%/S%)              {'PnL':>8}  "
          f"{'PPT':>7}  {'PF':>5}  {'T/D':>4}  Hold  SL/TO/G%")
    print(SEP2)

    years = sorted(set(t["year"] for t in all_trades))
    y_summary = []
    for y in years:
        y_trades = [t for t in all_trades if t["year"] == y]
        y_days   = len(set(t["ts"].date() for t in y_trades))
        s        = _stats(y_trades, y_days)
        s["_raw"] = y_trades
        s["year"] = y
        y_summary.append(s)
        _row(str(y), s, y_days)

    # ── 4. HMM Regime Breakdown ───────────────────────────────────────────────
    _hdr("4. PERFORMANCE PER HMM REGIME")
    print(f"  State 0 = TRENDING_DN | State 1 = RANGING_LOW | "
          f"State 2 = RANGING_HIGH | State 3 = TRENDING_UP")
    print(f"\n  {'Regime':<16} {'Trades':>6} (L/S)             "
          f"{'WR%':>5}  (L%/S%)              {'PnL':>8}  "
          f"{'PPT':>7}  {'PF':>5}  {'T/D':>4}  Hold  SL/TO/G%")
    print(SEP2)

    hmm_states = sorted(set(t["hmm"] for t in all_trades if t["hmm"] >= 0))
    hmm_summary = []
    for state in hmm_states:
        h_trades = [t for t in all_trades if t["hmm"] == state]
        h_days   = len(set(t["ts"].date() for t in h_trades))
        s        = _stats(h_trades, h_days)
        s["_raw"] = h_trades
        s["state"] = state
        hmm_summary.append(s)
        label = f"S{state} {HMM_NAMES.get(state, '???')}"
        _row(label, s, h_days)

    h_unknown = [t for t in all_trades if t["hmm"] < 0]
    if h_unknown:
        s = _stats(h_unknown, len(set(t["ts"].date() for t in h_unknown)))
        s["_raw"] = h_unknown
        _row("UNKNOWN", s, len(set(t["ts"].date() for t in h_unknown)))

    # ── 5. LONG vs SHORT Detail ───────────────────────────────────────────────
    _hdr("5. LONG vs SHORT — DETAIL")
    for direction in ["LONG", "SHORT"]:
        d_trades = [t for t in all_trades if t["direction"] == direction]
        d_days   = len(set(t["ts"].date() for t in d_trades))
        s        = _stats(d_trades, d_days)
        s["_raw"] = d_trades
        print(f"\n  [{direction}]")
        print(f"    Trades       : {s['n']:,}  ({s['n']/len(all_trades)*100:.1f}% dari total)")
        print(f"    Win Rate     : {s['wr']:.2f}%")
        print(f"    Total PnL    : ${s['pnl']:,.2f}  (${s['pnl']/s['n']:.4f}/trade)")
        print(f"    Profit Factor: {_pf(d_trades):.3f}")
        print(f"    Avg Hold     : {s['avg_hold_h']:.2f} h")
        print(f"    SL rate      : {s['sl_pct']:.1f}%  (Trade count: {sum(1 for t in d_trades if _is_sl(t['outcome'])):,})")
        print(f"    Timeout rate : {s['to_pct']:.1f}%  (Trade count: {sum(1 for t in d_trades if _is_to(t['outcome'])):,})")

        # LONG vs SHORT per year
        print(f"\n    Per tahun:")
        for y in years:
            yt = [t for t in d_trades if t["year"] == y]
            if not yt:
                continue
            yw  = sum(1 for t in yt if _is_win(t["outcome"]))
            ypnl = sum(t["net_pnl"] for t in yt)
            print(f"      {y}: {len(yt):>5} trades  WR={yw/len(yt)*100:.1f}%  PnL=${ypnl:>8.2f}  PPT=${ypnl/len(yt):>+.4f}")

    # ── 6. Coin Breakdown ─────────────────────────────────────────────────────
    _hdr("6. PERFORMANCE PER KOIN")
    coin_stats = []
    for sym in ALL_COINS:
        c_trades = [t for t in all_trades if t["coin"] == sym]
        if not c_trades:
            continue
        c_days = len(set(t["ts"].date() for t in c_trades))
        s      = _stats(c_trades, c_days)
        s["_raw"]  = c_trades
        s["coin"]  = sym
        s["pf"]    = _pf(c_trades)
        coin_stats.append(s)

    coin_stats.sort(key=lambda x: x["pnl"], reverse=True)

    print(f"\n  {'Koin':<16} {'N':>5} {'WR%':>6} {'PnL':>8} {'PPT':>7} {'PF':>5} "
          f"{'T/D':>4}  {'Hold':>5}  {'SL%':>4}")
    print(SEP2)
    for s in coin_stats:
        pf_str = f"{s['pf']:.2f}" if s['pf'] != float("inf") else " INF"
        print(
            f"  {s['coin']:<16} {s['n']:>5} {s['wr']:>6.1f}% "
            f"{s['pnl']:>8.2f} {s['ppt']:>+7.4f} {pf_str:>5} "
            f"{s['tpd']:>4.1f}  {s['avg_hold_h']:>4.1f}h  {s['sl_pct']:>4.0f}%"
        )

    # ── 7. Kekurangan & Kelebihan ─────────────────────────────────────────────
    _hdr("7. ANALISIS KELEBIHAN & KEKURANGAN")

    worst_quarters = sorted_q[-3:]
    best_quarters  = sorted_q[:3]

    print("\n  KELEBIHAN:")
    print(f"  [+] Kuartal terbaik : {', '.join(x['quarter'] for x in best_quarters)}")
    for x in best_quarters:
        print(f"       {x['quarter']}: {x['n']} trades, WR={x['wr']:.1f}%, PnL=${x['pnl']:.1f}, PF={_pf(x['_raw']):.2f}")

    best_hmm  = max(hmm_summary, key=lambda x: x["wr"])
    print(f"  [+] HMM regime terbaik: S{best_hmm['state']} {HMM_NAMES.get(best_hmm['state'],'?')} "
          f"(WR={best_hmm['wr']:.1f}%, PnL=${best_hmm['pnl']:.1f})")

    top_coin = coin_stats[0]
    print(f"  [+] Koin terbaik: {top_coin['coin']} "
          f"(WR={top_coin['wr']:.1f}%, PnL=${top_coin['pnl']:.1f}, PF={top_coin['pf']:.2f})")

    long_s  = _stats([t for t in all_trades if t["direction"] == "LONG"],  1)
    short_s = _stats([t for t in all_trades if t["direction"] == "SHORT"], 1)
    best_dir = "LONG" if long_s["wr"] > short_s["wr"] else "SHORT"
    print(f"  [+] Arah lebih kuat: {best_dir} "
          f"(LONG WR={long_s['wr']:.1f}% vs SHORT WR={short_s['wr']:.1f}%)")

    print(f"\n  KEKURANGAN:")
    print(f"  [-] Kuartal terburuk: {', '.join(x['quarter'] for x in worst_quarters)}")
    for x in worst_quarters:
        print(f"       {x['quarter']}: {x['n']} trades, WR={x['wr']:.1f}%, PnL=${x['pnl']:.1f}, PF={_pf(x['_raw']):.2f}")

    worst_hmm = min(hmm_summary, key=lambda x: x["wr"])
    print(f"  [-] HMM regime terlemah: S{worst_hmm['state']} {HMM_NAMES.get(worst_hmm['state'],'?')} "
          f"(WR={worst_hmm['wr']:.1f}%, PnL=${worst_hmm['pnl']:.1f})")

    bot_coin = coin_stats[-1]
    print(f"  [-] Koin terlemah: {bot_coin['coin']} "
          f"(WR={bot_coin['wr']:.1f}%, PnL=${bot_coin['pnl']:.1f}, PF={bot_coin['pf']:.2f})")

    # SL rate check
    if ov["sl_pct"] > 35:
        print(f"  [-] SL hit rate tinggi: {ov['sl_pct']:.1f}% -- threshold 0.45 terlalu agresif")
        print(f"       Threshold 0.50/0.45 atau 0.50/0.50 akan kurangi SL rate")

    if abs(long_s["wr"] - short_s["wr"]) > 3:
        weaker = "SHORT" if long_s["wr"] > short_s["wr"] else "LONG"
        print(f"  [-] Gap WR LONG vs SHORT = {abs(long_s['wr']-short_s['wr']):.1f}pp "
              f"({weaker} lebih lemah)")

    print(f"\n  TRADE CHARACTERISTICS:")
    print(f"  - Avg trades/hari  : {ov['tpd']:.2f} (total training period)")
    print(f"  - Avg hold         : {ov['avg_hold_h']:.2f} jam per trade")
    print(f"  - SL hit rate      : {ov['sl_pct']:.1f}%")
    print(f"  - Timeout rate     : {ov['to_pct']:.1f}% (closed at bar 36, no winner found)")
    print(f"  - WR formula       : WIN / (WIN + LOSS + TIMEOUT) = {ov['wr']:.1f}%")
    print(f"    Note: Win rate dihitung INCLUSIVE timeout -- timeout dihitung sebagai LOSS")

    # ── 8. Implikasi untuk HMM + LSTM Integration ─────────────────────────────
    _hdr("8. IMPLIKASI UNTUK HMM & LSTM INTEGRATION")

    print("\n  Dari analisis di atas, pertimbangkan:")

    trend_states  = [s for s in hmm_summary if s["state"] in [0, 3]]
    range_states  = [s for s in hmm_summary if s["state"] in [1, 2]]
    if trend_states and range_states:
        t_wr = np.mean([s["wr"] for s in trend_states])
        r_wr = np.mean([s["wr"] for s in range_states])
        t_pnl = sum(s["pnl"] for s in trend_states)
        r_pnl = sum(s["pnl"] for s in range_states)
        print(f"\n  HMM TRENDING (state 0+3): WR={t_wr:.1f}%, PnL=${t_pnl:.1f}")
        print(f"  HMM RANGING  (state 1+2): WR={r_wr:.1f}%, PnL=${r_pnl:.1f}")
        if t_wr > r_wr:
            print(f"  => Model lebih kuat di TRENDING. HMM bisa filter atau boost threshold di RANGING.")
        else:
            print(f"  => Model lebih kuat di RANGING. HMM alignment seperti ic32 mungkin menyakitkan.")

    # Quarterly trend: early periods vs late
    if len(q_summary) > 4:
        early = q_summary[:len(q_summary)//3]
        late  = q_summary[-len(q_summary)//3:]
        e_wr  = np.mean([s["wr"] for s in early])
        l_wr  = np.mean([s["wr"] for s in late])
        e_pnl = sum(s["pnl"] for s in early)
        l_pnl = sum(s["pnl"] for s in late)
        print(f"\n  Early OOF (kuartal pertama 1/3): WR={e_wr:.1f}%, PnL=${e_pnl:.1f}")
        print(f"  Late OOF  (kuartal terakhir 1/3): WR={l_wr:.1f}%, PnL=${l_pnl:.1f}")
        if l_wr < e_wr - 2:
            print(f"  => CONCERN: WR menurun di kuartal terbaru -- kemungkinan regime drift")
        else:
            print(f"  => WR stabil lintas waktu -- tidak ada drift signifikan")

    print(f"\n{SEP}")
    print(f"  SELESAI -- tb_lgbm_genuine_v2 OOF Analysis")
    print(f"  Gunakan hasil ini untuk desain HMM threshold + LSTM filter")
    print(SEP)


if __name__ == "__main__":
    main()
