"""
tools/trade_analyzer.py — Analisis performa trading dari CSV hasil live trading.

Usage:
    python tools/trade_analyzer.py
    python tools/trade_analyzer.py --file livetrade.csv
    python tools/trade_analyzer.py --file path/to/trades.csv --output path/to/report.md
    python tools/trade_analyzer.py --file trades.csv --p2 0.05 --p4 0.08
    python tools/trade_analyzer.py --file trades.csv --no-filter

Columns CSV yang diharapkan:
    Opened, Closed, Coin, Model, Direction, Conf, Entry, Exit,
    TP, SL, ATR, % to H4Hi, % to H4Lo, RR, H4 Trend, Vol Regime,
    H4 High, H4 Low, Qty, Leverage, PnL ($), PnL (%), Exit Reason, Hold Bars, Status
"""

import argparse
import csv
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean, stdev

# ── Defaults ──────────────────────────────────────────────────────────────────

# Deteksi otomatis berkas live trading.
# Prioritas: cache live dari VPS (live_db_bridge.py) > CSV swint lokal (basi) > fallback lokal.
# Jalankan `python tools/live_db_bridge.py` dulu untuk refresh cache live dari VPS.
LIVE_CACHE_CSV = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                              "data", "live_cache", "hasil_livetrading.csv")
PRODUCTION_CSV = r"D:\Apps-Dev\swint_tradev2\hasil_livetrading.csv"
DEFAULT_CSV    = (LIVE_CACHE_CSV if os.path.exists(LIVE_CACHE_CSV)
                  else PRODUCTION_CSV if os.path.exists(PRODUCTION_CSV)
                  else "livetrade.csv")
DEFAULT_REPORT = r"D:\Apps-Dev\Riset_pemodelan\reports\TRADE_ANALYSIS_REPORT.md"
DEFAULT_P2     = 0.05   # min Vol Regime
DEFAULT_P4     = 0.08   # max H4 swing deviation


# ── Helpers ───────────────────────────────────────────────────────────────────

def flt(row, col):
    try:
        return float(row.get(col, "").strip())
    except (ValueError, AttributeError):
        return None

def pnl(row):       return flt(row, "PnL ($)") or 0.0
def wins(lst):      return [r for r in lst if pnl(r) > 0]
def losses(lst):    return [r for r in lst if pnl(r) <= 0]
def gross_profit(lst): return sum(pnl(r) for r in lst if pnl(r) > 0)
def gross_loss(lst):   return abs(sum(pnl(r) for r in lst if pnl(r) <= 0))

def profit_factor(lst):
    gp, gl = gross_profit(lst), gross_loss(lst)
    return round(gp / gl, 2) if gl else 99.0

def win_rate(lst):  return round(len(wins(lst)) / len(lst) * 100, 1) if lst else 0.0
def net_pnl(lst):   return round(sum(pnl(r) for r in lst), 2)
def avg_pnl(lst):   return round(mean([pnl(r) for r in lst]), 2) if lst else 0.0

def max_consec_loss_streak(trades):
    sorted_t = sorted(trades, key=lambda r: r.get("Opened", ""))
    streak = max_streak = 0
    cur_loss = max_loss = 0.0
    for r in sorted_t:
        p = pnl(r)
        if p <= 0:
            streak += 1; cur_loss += p
            if streak > max_streak:
                max_streak = streak; max_loss = cur_loss
        else:
            streak = 0; cur_loss = 0.0
    return max_streak, round(max_loss, 2)

def sharpe_like(lst):
    pnls = [pnl(r) for r in lst]
    if len(pnls) < 2: return 0.0
    try:
        s = stdev(pnls)
        return round(mean(pnls) / s, 2) if s else 0.0
    except Exception:
        return 0.0

def swing_devs(row):
    entry = flt(row, "Entry") or 0
    h4hi  = flt(row, "H4 High") or 0
    h4lo  = flt(row, "H4 Low")  or 0
    if entry == 0: return 0.0, 0.0
    return abs(entry - h4hi) / entry * 100, abs(entry - h4lo) / entry * 100


# ── Filters ───────────────────────────────────────────────────────────────────

def passes_p4(row, threshold=DEFAULT_P4):
    entry = flt(row, "Entry") or 0
    h4hi  = flt(row, "H4 High")
    h4lo  = flt(row, "H4 Low")
    if not h4hi or not h4lo or entry == 0: return True
    return not (abs(entry - h4hi) / entry > threshold or
                abs(entry - h4lo) / entry > threshold)

def passes_p2(row, threshold=DEFAULT_P2):
    vol = flt(row, "Vol Regime")
    return True if vol is None else vol >= threshold


# ── Delta formatter ────────────────────────────────────────────────────────────

def delta_md(a, b, higher_better=True, fmt=".2f"):
    d = b - a
    sign = "+" if d > 0 else ""
    arrow = " ↑" if (d > 0) == higher_better and d != 0 else (" ↓" if d != 0 else "")
    return f"{sign}{d:{fmt}}{arrow}"


# ── Report sections ───────────────────────────────────────────────────────────

def section_overview(closed, open_trades):
    streak_n, streak_usd = max_consec_loss_streak(closed)
    w, l = wins(closed), losses(closed)
    ratio = f"{abs(avg_pnl(w)/avg_pnl(l)):.2f}x" if l else "N/A"
    return "\n".join([
        "## 1. Overview\n",
        "| Metrik | Nilai |", "|---|---|",
        f"| Total trades (closed) | {len(closed)} |",
        f"| Open positions | {len(open_trades)} |",
        f"| Win / Loss | {len(w)} / {len(l)} |",
        f"| **Win Rate** | **{win_rate(closed):.1f}%** |",
        f"| **Net PnL** | **${net_pnl(closed):+.2f}** |",
        f"| Gross Profit | ${gross_profit(closed):.2f} |",
        f"| Gross Loss | ${gross_loss(closed):.2f} |",
        f"| **Profit Factor** | **{profit_factor(closed):.2f}** |",
        f"| Sharpe-like | {sharpe_like(closed):.2f} |",
        f"| Avg Win | ${avg_pnl(w):+.2f} |",
        f"| Avg Loss | ${avg_pnl(l):+.2f} |",
        f"| Avg Win / Avg Loss | {ratio} |",
        f"| Max consec. losses | {streak_n} trades (${streak_usd:.2f}) |",
    ])


def section_per_model(closed):
    lines = ["## 2. Per Model\n",
             "| Model | N | WR | Net PnL | PF | Avg Conf | Avg Hold |",
             "|---|---|---|---|---|---|---|"]
    for m in sorted(set(r.get("Model","?").strip() for r in closed)):
        mt    = [r for r in closed if r.get("Model","").strip() == m]
        confs = [flt(r,"Conf") for r in mt if flt(r,"Conf")]
        holds = [flt(r,"Hold Bars") for r in mt if flt(r,"Hold Bars") is not None]
        ac = f"{mean(confs):.2f}" if confs else "-"
        ah = f"{mean(holds):.1f}h" if holds else "-"
        lines.append(f"| {m} | {len(mt)} | {win_rate(mt):.1f}% | ${net_pnl(mt):+.2f} | {profit_factor(mt):.2f} | {ac} | {ah} |")
    return "\n".join(lines)


def section_exit_reasons(closed):
    lines = ["## 3. Exit Reasons\n",
             "| Exit Reason | N | % | WR | Avg PnL | Total PnL |",
             "|---|---|---|---|---|---|"]
    for r in ["tp_hit","sl_hit","guardian_exit","guardian_momentum_exit","time_exit","manual_close"]:
        rt = [t for t in closed if t.get("Exit Reason","").strip() == r]
        if not rt: continue
        lines.append(f"| {r} | {len(rt)} | {len(rt)/len(closed)*100:.1f}% | {win_rate(rt):.1f}% | ${avg_pnl(rt):+.2f} | ${net_pnl(rt):+.2f} |")
    return "\n".join(lines)


def section_conf_band(closed):
    lines = ["## 4. Confidence Bands\n",
             "| Band | N | WR | Avg PnL | PF |", "|---|---|---|---|---|"]
    for label, lo, hi in [("0.60-0.70",0.60,0.70),("0.70-0.80",0.70,0.80),
                           ("0.80-0.90",0.80,0.90),("0.90-1.00",0.90,1.01)]:
        bt = [r for r in closed if lo <= (flt(r,"Conf") or 0) < hi]
        if bt: lines.append(f"| {label} | {len(bt)} | {win_rate(bt):.1f}% | ${avg_pnl(bt):+.2f} | {profit_factor(bt):.2f} |")
    return "\n".join(lines)


def section_vol_regime(closed):
    lines = ["## 5. Volume Regime\n",
             "| Vol Regime | N | WR | Avg PnL | PF |", "|---|---|---|---|---|"]
    for label, lo, hi in [("< 0.05",0,0.05),("0.05-0.20",0.05,0.20),
                           ("0.20-0.50",0.20,0.50),("0.50-1.00",0.50,1.00),("> 1.00",1.00,999)]:
        bt = [r for r in closed if lo <= (flt(r,"Vol Regime") or 0) < hi]
        if bt: lines.append(f"| {label} | {len(bt)} | {win_rate(bt):.1f}% | ${avg_pnl(bt):+.2f} | {profit_factor(bt):.2f} |")
    return "\n".join(lines)


def section_h4_trend(closed):
    lines = ["## 6. H4 Trend Alignment\n",
             "| Scenario | N | WR | Net PnL | PF |", "|---|---|---|---|---|"]
    scenarios = [
        ("LONG + UP  (with-trend)",   lambda r: r.get("Direction")=="LONG"  and r.get("H4 Trend")=="UP"),
        ("LONG + DOWN (counter)",     lambda r: r.get("Direction")=="LONG"  and r.get("H4 Trend")=="DOWN"),
        ("SHORT + DOWN (with-trend)", lambda r: r.get("Direction")=="SHORT" and r.get("H4 Trend")=="DOWN"),
        ("SHORT + UP  (counter)",     lambda r: r.get("Direction")=="SHORT" and r.get("H4 Trend")=="UP"),
        ("Trend = N/A",               lambda r: not r.get("H4 Trend","").strip()),
    ]
    for label, cond in scenarios:
        bt = [r for r in closed if cond(r)]
        if bt: lines.append(f"| {label} | {len(bt)} | {win_rate(bt):.1f}% | ${net_pnl(bt):+.2f} | {profit_factor(bt):.2f} |")
    return "\n".join(lines)


def section_hold_duration(closed):
    lines = ["## 7. Hold Duration\n",
             "| Hold | N | WR | Avg PnL | Top Exit |", "|---|---|---|---|---|"]
    for label, lo, hi in [("0-2h",0,3),("3-6h",3,7),("7-12h",7,13),("13-24h",13,25),("24h+",25,9999)]:
        bt = [r for r in closed if lo <= (flt(r,"Hold Bars") or 0) < hi]
        if not bt: continue
        top = Counter(r.get("Exit Reason","") for r in bt).most_common(1)[0][0]
        lines.append(f"| {label} | {len(bt)} | {win_rate(bt):.1f}% | ${avg_pnl(bt):+.2f} | {top} |")
    return "\n".join(lines)


def section_swing_dev_dist(closed):
    lines = ["## 8. H4 Swing Deviation Distribution\n",
             "_max(|entry-H4Hi|, |entry-H4Lo|) / entry_\n",
             "| Swing Dev | N | WR | Net PnL | PF | Note |", "|---|---|---|---|---|---|"]
    for label, lo, hi, note in [("<=5%",0,5,""),("5-8%",5,8,""),
                                  ("8-15%",8,15,"⚠️ stale"),("15-30%",15,30,"🔴 very stale"),(">30%",30,999,"🔴🔴 extreme")]:
        bt = [r for r in closed if lo <= max(swing_devs(r)) < hi]
        if bt: lines.append(f"| {label} | {len(bt)} | {win_rate(bt):.1f}% | ${net_pnl(bt):+.2f} | {profit_factor(bt):.2f} | {note} |")
    return "\n".join(lines)


def section_coin_concentration(closed):
    lines = ["## 9. Coin Concentration\n",
             "| Coin | N | WR | Net PnL | % Total |", "|---|---|---|---|---|"]
    coins_agg = defaultdict(list)
    for r in closed: coins_agg[r.get("Coin","?").strip()].append(r)
    total = net_pnl(closed)
    for coin, ct in sorted(coins_agg.items(), key=lambda x: net_pnl(x[1]), reverse=True):
        n = net_pnl(ct)
        pct = round(n/total*100) if total else 0
        lines.append(f"| {coin} | {len(ct)} | {win_rate(ct):.1f}% | ${n:+.2f} | {pct:+d}% |")
    return "\n".join(lines)


def section_filter_comparison(closed, p2_thresh, p4_thresh):
    after   = [r for r in closed if passes_p2(r, p2_thresh) and passes_p4(r, p4_thresh)]
    blocked = [r for r in closed if not (passes_p2(r, p2_thresh) and passes_p4(r, p4_thresh))]
    blk_w   = [r for r in blocked if pnl(r) > 0]
    blk_l   = [r for r in blocked if pnl(r) <= 0]
    sbn, sbu = max_consec_loss_streak(closed)
    san, sau = max_consec_loss_streak(after)
    no_ton_b = [r for r in closed if "TONUSDT" not in r.get("Coin","")]
    no_ton_a = [r for r in after  if "TONUSDT" not in r.get("Coin","")]

    lines = [
        f"## 10. Simulasi Filter P2 + P4\n",
        f"_P2: Vol Regime >= {p2_thresh} | P4: Swing dev <= {p4_thresh*100:.0f}%_\n",
        "| Metrik | Sebelum | Sesudah | Delta |", "|---|---|---|---|",
        f"| Jumlah trades | {len(closed)} | {len(after)} | {len(after)-len(closed):+d} |",
        f"| Win / Loss | {len(wins(closed))}/{len(losses(closed))} | {len(wins(after))}/{len(losses(after))} | — |",
        f"| **Win Rate** | **{win_rate(closed):.1f}%** | **{win_rate(after):.1f}%** | **{delta_md(win_rate(closed),win_rate(after),fmt='.1f')}pp** |",
        f"| **Net PnL** | **${net_pnl(closed):+.2f}** | **${net_pnl(after):+.2f}** | **{delta_md(net_pnl(closed),net_pnl(after))}** |",
        f"| Gross Profit | ${gross_profit(closed):.2f} | ${gross_profit(after):.2f} | {delta_md(gross_profit(closed),gross_profit(after))} |",
        f"| Gross Loss | ${gross_loss(closed):.2f} | ${gross_loss(after):.2f} | {delta_md(gross_loss(closed),gross_loss(after),False)} |",
        f"| **Profit Factor** | **{profit_factor(closed):.2f}** | **{profit_factor(after):.2f}** | **{delta_md(profit_factor(closed),profit_factor(after))}** |",
        f"| Avg Win | ${avg_pnl(wins(closed)):+.2f} | ${avg_pnl(wins(after)):+.2f} | {delta_md(avg_pnl(wins(closed)),avg_pnl(wins(after)))} |",
        f"| Avg Loss | ${avg_pnl(losses(closed)):+.2f} | ${avg_pnl(losses(after)):+.2f} | {delta_md(avg_pnl(losses(closed)),avg_pnl(losses(after)),False)} |",
        f"| Max consec. losses | {sbn} (${sbu:.2f}) | {san} (${sau:.2f}) | {san-sbn:+d} trades |",
        "",
        f"### Trade Diblokir: {len(blocked)}\n",
        "| | N | Total PnL |", "|---|---|---|",
        f"| Wins diblokir | {len(blk_w)} | ${sum(pnl(r) for r in blk_w):+.2f} |",
        f"| Losses dicegah | {len(blk_l)} | ${sum(pnl(r) for r in blk_l):+.2f} |",
        f"| **Net PnL diblokir** | **{len(blocked)}** | **${sum(pnl(r) for r in blocked):+.2f}** |",
        "",
        "### Detail Trade Diblokir\n",
        "| Tgl | Coin | Dir | Vol | Hi_dev | Lo_dev | PnL | Alasan Block |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in sorted(blocked, key=lambda x: x.get("Opened","")):
        hi_dev, lo_dev = swing_devs(r)
        vol = flt(r,"Vol Regime") or 0
        p   = pnl(r)
        flag = "✅ WIN" if p > 0 else "❌ LOSS"
        reasons = []
        if not passes_p2(r, p2_thresh): reasons.append(f"P2(vol={vol:.3f})")
        if not passes_p4(r, p4_thresh): reasons.append(f"P4(dev={max(hi_dev,lo_dev):.1f}%)")
        lines.append(
            f"| {r['Opened'][:10]} | {r.get('Coin','')} | {r.get('Direction','')} "
            f"| {vol:.3f} | {hi_dev:.1f}% | {lo_dev:.1f}% | ${p:+.2f} {flag} | {', '.join(reasons)} |"
        )

    lines += [
        "",
        "### Tanpa TONUSDT (apples-to-apples)\n",
        "| Metrik | Sebelum | Sesudah | Delta |", "|---|---|---|---|",
        f"| Win Rate | {win_rate(no_ton_b):.1f}% | {win_rate(no_ton_a):.1f}% | {delta_md(win_rate(no_ton_b),win_rate(no_ton_a),fmt='.1f')}pp |",
        f"| Net PnL | ${net_pnl(no_ton_b):+.2f} | ${net_pnl(no_ton_a):+.2f} | {delta_md(net_pnl(no_ton_b),net_pnl(no_ton_a))} |",
        f"| Profit Factor | {profit_factor(no_ton_b):.2f} | {profit_factor(no_ton_a):.2f} | {delta_md(profit_factor(no_ton_b),profit_factor(no_ton_a))} |",
    ]
    return "\n".join(lines)


def section_open_risk(open_trades):
    if not open_trades:
        return "## 11. Open Positions\n\n_Tidak ada posisi terbuka._"
    lines = ["## 11. Open Positions Risk\n",
             "| Coin | Dir | Conf | H4 | VolR | Hi_dev | Lo_dev | Hold | Flags |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in sorted(open_trades, key=lambda x: x.get("Opened","")):
        hi_dev, lo_dev = swing_devs(r)
        vol  = flt(r,"Vol Regime") or 0
        dirn = r.get("Direction","")
        h4t  = r.get("H4 Trend","?")
        flags = []
        if vol < 0.05:                           flags.append("vol_dead")
        if max(hi_dev, lo_dev) > 8:              flags.append("swing_stale")
        if dirn=="SHORT" and h4t=="UP":          flags.append("counter_trend")
        if dirn=="LONG"  and h4t=="DOWN":        flags.append("counter_trend")
        lines.append(
            f"| {r.get('Coin','')} | {dirn} | {flt(r,'Conf') or 0:.2f} | {h4t} "
            f"| {vol:.2f} | {hi_dev:.1f}% | {lo_dev:.1f}% | {r.get('Hold Bars','?')}h "
            f"| {', '.join(flags) or '—'} |"
        )
    return "\n".join(lines)


def section_recent(closed, n=20):
    recent = sorted(closed, key=lambda r: r.get("Opened",""))[-n:]
    sn, su = max_consec_loss_streak(recent)
    lines = [
        f"## 12. {n} Trade Terakhir\n",
        "| Metrik | Nilai |", "|---|---|",
        f"| Trades | {len(recent)} |",
        f"| Win Rate | {win_rate(recent):.1f}% |",
        f"| Net PnL | ${net_pnl(recent):+.2f} |",
        f"| Profit Factor | {profit_factor(recent):.2f} |",
        f"| Max consec. losses | {sn} (${su:.2f}) |",
        "",
        "| Tgl | Coin | Dir | PnL | Exit |", "|---|---|---|---|---|",
    ]
    for r in recent:
        p = pnl(r)
        lines.append(f"| {r['Opened'][:10]} | {r.get('Coin','')} | {r.get('Direction','')} | {'✅' if p>0 else '❌'} ${p:+.2f} | {r.get('Exit Reason','')} |")
    return "\n".join(lines)


# ── Build full report ─────────────────────────────────────────────────────────

def build_report(csv_path: str, p2: float, p4: float) -> str:
    with open(csv_path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    closed      = [r for r in rows if r.get("Status","").strip() == "closed"]
    open_trades = [r for r in rows if r.get("Status","").strip() == "open"]
    if not closed:
        return "# Trade Analysis Report\n\n_Tidak ada closed trade._"

    dates    = sorted(r.get("Opened","") for r in closed)
    header   = (
        f"# Trade Analysis Report\n\n"
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}  \n"
        f"**File:** `{csv_path}`  \n"
        f"**Period:** {dates[0][:10]} s/d {dates[-1][:10]}  \n"
        f"**Closed:** {len(closed)} | **Open:** {len(open_trades)}  \n"
        f"**Filter params:** P2 VolR >= {p2} | P4 swing_dev <= {p4*100:.0f}%\n\n---\n"
    )

    parts = [
        header,
        section_overview(closed, open_trades),      "",
        section_per_model(closed),                   "",
        section_exit_reasons(closed),                "",
        section_conf_band(closed),                   "",
        section_vol_regime(closed),                  "",
        section_h4_trend(closed),                    "",
        section_hold_duration(closed),               "",
        section_swing_dev_dist(closed),              "",
        section_coin_concentration(closed),          "",
        section_filter_comparison(closed, p2, p4),   "",
        section_open_risk(open_trades),              "",
        section_recent(closed, 20),                  "",
        "---\n_Generated by tools/trade_analyzer.py — SwingTrade v2_",
    ]
    return "\n".join(parts)


# ── Console summary ───────────────────────────────────────────────────────────

def print_summary(closed, p2, p4):
    after   = [r for r in closed if passes_p2(r,p2) and passes_p4(r,p4)]
    blocked = [r for r in closed if not (passes_p2(r,p2) and passes_p4(r,p4))]
    blk_w   = [r for r in blocked if pnl(r) > 0]
    blk_l   = [r for r in blocked if pnl(r) <= 0]
    sbn, sbu = max_consec_loss_streak(closed)
    san, sau = max_consec_loss_streak(after)
    no_ton_b = [r for r in closed if "TONUSDT" not in r.get("Coin","")]
    no_ton_a = [r for r in after  if "TONUSDT" not in r.get("Coin","")]

    def row(label, b, a, higher_better=True, fmt=".2f"):
        d = a - b
        arrow = "↑" if (d > 0) == higher_better and d != 0 else ("↓" if d != 0 else "=")
        sign  = "+" if d > 0 else ""
        print(f"  {label:<26} {b:>9{fmt}} {a:>9{fmt}}   {sign}{d:{fmt}} {arrow}")

    print("=" * 62)
    print("  TRADE ANALYSIS SUMMARY")
    print("=" * 62)
    print(f"  {'Metrik':<26} {'SEBELUM':>9} {'SESUDAH':>9}   {'DELTA':>9}")
    print(f"  {'-'*58}")
    row("Win Rate (%)",        win_rate(closed),       win_rate(after),        fmt=".1f")
    row("Net PnL ($)",         net_pnl(closed),        net_pnl(after))
    row("Profit Factor",       profit_factor(closed),  profit_factor(after))
    row("Avg Win ($)",         avg_pnl(wins(closed)),  avg_pnl(wins(after)))
    row("Avg Loss ($)",        avg_pnl(losses(closed)),avg_pnl(losses(after)), higher_better=False)
    row("Max streak (n)",      float(sbn),             float(san),             False, ".0f")
    row("Streak loss ($)",     sbu,                    sau,                    higher_better=False)
    print()
    print(f"  Trade diblokir  : {len(blocked)}  ({len(blk_w)} WIN + {len(blk_l)} LOSS)")
    print(f"  Wins diblokir   : ${sum(pnl(r) for r in blk_w):+.2f}")
    print(f"  Loss dicegah    : ${sum(pnl(r) for r in blk_l):+.2f}")
    print()
    print(f"  === Tanpa TONUSDT (apples-to-apples) ===")
    print(f"  WR  : {win_rate(no_ton_b):.1f}% -> {win_rate(no_ton_a):.1f}%")
    d_pnl = net_pnl(no_ton_a) - net_pnl(no_ton_b)
    print(f"  PnL : ${net_pnl(no_ton_b):+.2f} -> ${net_pnl(no_ton_a):+.2f}  (delta ${d_pnl:+.2f})")
    print(f"  PF  : {profit_factor(no_ton_b):.2f} -> {profit_factor(no_ton_a):.2f}")
    print("=" * 62)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    # Windows terminal UTF-8
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="Trade Analyzer — SwingTrade v2")
    parser.add_argument("--file",      default=DEFAULT_CSV,    help="Path ke CSV trades")
    parser.add_argument("--output",    default=DEFAULT_REPORT, help="Path output report .md")
    parser.add_argument("--p2",        type=float, default=DEFAULT_P2, help=f"Min Vol Regime (default {DEFAULT_P2})")
    parser.add_argument("--p4",        type=float, default=DEFAULT_P4, help=f"Max swing deviation (default {DEFAULT_P4})")
    parser.add_argument("--no-filter", action="store_true", help="Tampilkan baseline tanpa simulasi filter")
    parser.add_argument("--no-save",   action="store_true", help="Jangan simpan file, print ke stdout saja")
    args = parser.parse_args()

    csv_path = args.file
    if not os.path.isabs(csv_path):
        csv_path = str(Path(__file__).parent.parent / csv_path)

    if not os.path.exists(csv_path):
        print(f"ERROR: File tidak ditemukan: {csv_path}", file=sys.stderr)
        sys.exit(1)

    p2 = 0.0 if args.no_filter else args.p2
    p4 = 1.0 if args.no_filter else args.p4

    with open(csv_path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    closed = [r for r in rows if r.get("Status","").strip() == "closed"]

    print_summary(closed, p2, p4)

    if not args.no_save:
        report   = build_report(csv_path, p2, p4)
        out_path = args.output
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"\n  Report disimpan ke: {out_path}")


if __name__ == "__main__":
    main()
