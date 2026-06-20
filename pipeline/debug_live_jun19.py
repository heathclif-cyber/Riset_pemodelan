"""
Debug: analisis signals dan trades pada 19 Juni 2026 (WITA = UTC+8)
dari live DB VPS. Bandingkan dengan simulasi holdout.
"""
import json, sys, sqlite3
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from tools.live_db_bridge import LOCAL_DB, _connect

WITA = pd.Timedelta(hours=8)

# ── Load signals + trades ──────────────────────────────────────────────────────
con = _connect()

signals_df = pd.read_sql_query("""
    SELECT s.*, c.symbol AS coin_symbol, mm.model_type AS model_type
    FROM signal s
    JOIN coin c ON s.coin_id = c.id
    LEFT JOIN model_meta mm ON s.model_meta_id = mm.id
    ORDER BY s.signal_time
""", con)

trades_df = pd.read_sql_query("""
    SELECT t.*, c.symbol AS coin_symbol, mm.model_type AS model_type,
           s.confidence AS signal_conf, s.atr_at_signal, s.feature_snapshot
    FROM trade t
    JOIN coin c ON t.coin_id = c.id
    LEFT JOIN signal s ON t.signal_id = s.id
    LEFT JOIN model_meta mm ON s.model_meta_id = mm.id
    ORDER BY t.opened_at
""", con)
con.close()

# Parse timestamps
for df in [signals_df, trades_df]:
    for col in ['signal_time', 'opened_at', 'closed_at']:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], utc=True, errors='coerce')

# Filter Jun 19 WITA = Jun 18 16:00 UTC -> Jun 19 16:00 UTC
JUN19_START = pd.Timestamp('2026-06-18 16:00', tz='UTC')
JUN19_END   = pd.Timestamp('2026-06-19 16:00', tz='UTC')

sig_jun19   = signals_df[(signals_df['signal_time'] >= JUN19_START) &
                          (signals_df['signal_time'] < JUN19_END)].copy()
trade_jun19 = trades_df[(trades_df['opened_at'] >= JUN19_START) &
                         (trades_df['opened_at'] < JUN19_END)].copy()

sep = "=" * 95

print(f"\n{sep}")
print("  LIVE DB — Signals 19 Juni 2026 WITA")
print(f"  Window UTC: {JUN19_START} -> {JUN19_END}")
print(f"  Total signals: {len(sig_jun19)} | Trades opened: {len(trade_jun19)}")
print(f"{sep}\n")

# ── Semua signal June 19 ────────────────────────────────────────────────────
print(f"  {'Sym':<14} {'Time WITA':>8} {'Dir':>5} {'Conf':>5} {'Traded?':>8} {'Close':>10}  {'Reason not traded'}")
print("  " + "-" * 90)

# Buat lookup dari signal ke trade
sig_to_trade = {}
if 'signal_id' in trades_df.columns:
    for _, tr in trades_df.iterrows():
        sid = tr.get('signal_id')
        if sid:
            sig_to_trade[sid] = tr

sim_signals_jun19 = {
    'NEARUSDT':  '00:00',
    'NEARUSDT_01': '01:00',
    'LINKUSDT':  '05:00',
    'LINKUSDT_06': '06:00',
    'NEARUSDT_08': '08:00',
    'BTCUSDT':   '16:00',
    'ETHUSDT':   '16:00',
    'DOGEUSDT':  '18:00',
}

traded_syms = set()
for _, tr in trade_jun19.iterrows():
    traded_syms.add(tr['coin_symbol'])

for _, sig in sig_jun19.sort_values('signal_time').iterrows():
    sym       = sig['coin_symbol']
    ts_wita   = (sig['signal_time'] + WITA).strftime('%H:%M')
    direction = sig.get('direction', '?')
    conf      = sig.get('confidence', 0)
    traded    = sym in sig_to_trade.get(sig.get('id'), {}) or (sig.get('id') in sig_to_trade)
    model     = sig.get('model_type', '?')

    # Cek apakah signal ini punya trade
    sig_id    = sig.get('id')
    has_trade = sig_id in sig_to_trade
    trade_tag = "TRADED" if has_trade else "NO TRADE"

    # Ambil feature snapshot jika ada
    feat = {}
    if has_trade:
        tr = sig_to_trade[sig_id]
        fs = tr.get('feature_snapshot')
        if fs:
            try: feat = json.loads(fs)
            except: feat = {}

    close_price = sig.get('entry_price') or sig.get('close_price') or '?'

    print(f"  {sym:<14} {ts_wita:>8} {str(direction):>5} {float(conf) if conf else 0:>5.3f} {trade_tag:>8} {str(close_price):>10}")

# ── Trade yang benar-benar terjadi ───────────────────────────────────────────
print(f"\n{sep}")
print("  TRADES LIVE — 19 Juni 2026 WITA")
print(f"{sep}")

if trade_jun19.empty:
    print("  Tidak ada trade pada tanggal ini.")
else:
    for _, tr in trade_jun19.iterrows():
        sym        = tr['coin_symbol']
        opened_wita = (tr['opened_at'] + WITA).strftime('%H:%M') if pd.notna(tr['opened_at']) else '?'
        closed_wita = (tr['closed_at'] + WITA).strftime('%H:%M') if pd.notna(tr.get('closed_at')) else 'OPEN'
        direction   = tr.get('direction', '?')
        entry       = tr.get('entry_price', '?')
        exit_p      = tr.get('exit_price', '?')
        pnl         = tr.get('pnl_net', 0)
        conf_s      = tr.get('signal_conf', tr.get('confidence', '?'))
        exit_reason = tr.get('exit_reason', '?')
        bars_held   = tr.get('bars_held', '?')
        print(f"  {sym} | {opened_wita} -> {closed_wita} | {direction} | entry={entry} exit={exit_p}")
        print(f"    conf={conf_s} | PnL={pnl} | exit={exit_reason} | bars={bars_held}")
        # Feature snapshot
        fs = tr.get('feature_snapshot')
        if fs:
            try:
                feat = json.loads(fs)
                print(f"    Features @ signal: {json.dumps({k: round(float(v), 4) if isinstance(v, float) else v for k, v in feat.items()}, indent=6)[:500]}")
            except:
                pass
        print()

# ── Signals semua koin jam 08:00 UTC (16:00 WITA) ──────────────────────────
print(f"\n{sep}")
print("  DETAIL: Semua signal jam 08:00 UTC (16:00 WITA) — BTC vs ETH vs coin lain")
print(f"{sep}")

ts_target = pd.Timestamp('2026-06-19 08:00', tz='UTC')
sig_1600 = sig_jun19[
    (sig_jun19['signal_time'] >= ts_target) &
    (sig_jun19['signal_time'] < ts_target + pd.Timedelta(hours=1))
]

if sig_1600.empty:
    # Coba window lebih lebar ±30 menit
    sig_1600 = sig_jun19[
        (sig_jun19['signal_time'] >= ts_target - pd.Timedelta(minutes=30)) &
        (sig_jun19['signal_time'] < ts_target + pd.Timedelta(hours=1, minutes=30))
    ]

print(f"  Signals sekitar 16:00-17:00 WITA: {len(sig_1600)}")
for _, sig in sig_1600.sort_values('signal_time').iterrows():
    sym  = sig['coin_symbol']
    ts   = (sig['signal_time'] + WITA).strftime('%H:%M:%S')
    conf = float(sig.get('confidence', 0))
    dirn = sig.get('direction', '?')
    sid  = sig.get('id')
    has_trade = sid in sig_to_trade
    fs = {}
    if has_trade:
        tr_row = sig_to_trade[sid]
        fs_raw = tr_row.get('feature_snapshot') or ''
        try: fs = json.loads(fs_raw)
        except: fs = {}
    print(f"\n  [{sym}] {ts} WITA | {dirn} | conf={conf:.4f} | traded={'YES' if has_trade else 'NO'}")
    if fs:
        keys_of_interest = ['hmm_regime', 'hmm_regime_enc', 'h4_swing_high', 'h4_swing_low',
                            'cvd_slope_h4', 'ofi_h4_delta', 'atr_14_h1', 'vol_ratio_20',
                            'funding_rate', 'stochrsi_k', 'dist_liq_50x_long']
        for k in keys_of_interest:
            if k in fs:
                v = fs[k]
                print(f"    {k}: {round(float(v),6) if isinstance(v, (int, float)) else v}")

# ── Cek kolom feature_snapshot untuk signal yang TIDAK trade ───────────────
print(f"\n{sep}")
print("  FEATURE SNAPSHOT: Signal NON-TRADED di jam 16:00 WITA")
print(f"{sep}")

for _, sig in sig_1600.sort_values('signal_time').iterrows():
    sid = sig.get('id')
    if sid in sig_to_trade:
        continue  # skip yang traded
    sym  = sig['coin_symbol']
    ts   = (sig['signal_time'] + WITA).strftime('%H:%M:%S')
    conf = float(sig.get('confidence', 0))
    dirn = sig.get('direction', '?')
    # Signal tabel juga bisa punya feature_snapshot?
    fs_raw = sig.get('feature_snapshot') or ''
    if not fs_raw and 'feature_snapshot' in sig.index:
        fs_raw = sig['feature_snapshot'] or ''
    try:
        fs = json.loads(fs_raw)
    except:
        fs = {}
    print(f"\n  [{sym}] {ts} | {dirn} | conf={conf:.4f}")
    if fs:
        for k, v in fs.items():
            vdisp = round(float(v), 6) if isinstance(v, (int, float)) else v
            print(f"    {k}: {vdisp}")
    else:
        print("    (tidak ada feature_snapshot di signal table)")

print(f"\n{sep}\n")
