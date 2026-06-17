# -*- coding: utf-8 -*-
"""
tools/live_db_bridge.py — Jembatan baca DB live (signal + trade) dari Web App VPS.

Web App produksi (swint_tradev2) menyimpan signal & trade di SQLite pada VPS:
    /home/swint/swint_tradev2/instance/app.db
File lokal `D:\\Apps-Dev\\swint_tradev2\\instance\\app.db` adalah DB dev lama (BASI) —
JANGAN dipakai untuk analisis live. Bridge ini menarik DB live dari VPS via scp ke
cache lokal, lalu membacanya READ-ONLY menjadi DataFrame.

Alur:
    pull_live_db()  -> scp VPS:app.db -> data/live_cache/app.db
    load_trades()   -> DataFrame trade  (+ coin symbol, model_type, conf, atr, feature_snapshot)
    load_signals()  -> DataFrame signal (+ coin symbol, model_type)
    export_livetrading_csv() -> CSV format identik trades_export_csv() web app
                               (kolom: Opened..Status) supaya trade_analyzer.py jalan di data live.

CLI:
    python tools/live_db_bridge.py                 # pull + ringkasan + export CSV
    python tools/live_db_bridge.py --no-pull       # pakai cache lokal (tanpa scp)
    python tools/live_db_bridge.py --summary        # ringkasan saja
    python tools/live_db_bridge.py --export-csv P   # export ke path P

Konfigurasi via env (override default):
    SWINT_VPS_HOST     (default 139.180.157.176)
    SWINT_VPS_USER     (default root)
    SWINT_VPS_DB       (default /home/swint/swint_tradev2/instance/app.db)
    SWINT_VPS_SSH_KEY  (opsional, path private key untuk scp non-interaktif)

Catatan: scp non-interaktif butuh auth key-based ke VPS. Jika belum, set SWINT_VPS_SSH_KEY
atau tambahkan key VPS ke ssh-agent. DB live pakai journal_mode=delete (bukan WAL) dan
write jarang (tiap 5 menit) — scp file tunggal aman.
"""

import argparse
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

# ── Konfigurasi ───────────────────────────────────────────────────────────────

VPS_HOST = os.environ.get("SWINT_VPS_HOST", "139.180.157.176")
VPS_USER = os.environ.get("SWINT_VPS_USER", "root")
VPS_DB   = os.environ.get("SWINT_VPS_DB", "/home/swint/swint_tradev2/instance/app.db")
SSH_KEY  = os.environ.get("SWINT_VPS_SSH_KEY", "")

REPO_DIR    = Path(__file__).resolve().parents[1]
CACHE_DIR   = REPO_DIR / "data" / "live_cache"
LOCAL_DB    = CACHE_DIR / "app.db"
# Path CSV default yang dideteksi trade_analyzer.py (prioritas di atas DB swint basi)
LIVE_CSV    = CACHE_DIR / "hasil_livetrading.csv"


# ── Pull dari VPS ─────────────────────────────────────────────────────────────

def pull_live_db(force: bool = True, max_age_min: float = 10.0) -> Path:
    """Tarik app.db live dari VPS via scp ke cache lokal.

    force=False + cache lebih muda dari max_age_min → lewati scp (pakai cache).
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    if not force and LOCAL_DB.exists():
        age_min = (time.time() - LOCAL_DB.stat().st_mtime) / 60.0
        if age_min < max_age_min:
            print(f"[bridge] Cache masih segar ({age_min:.1f} mnt < {max_age_min}) — skip scp.")
            return LOCAL_DB

    remote = f"{VPS_USER}@{VPS_HOST}:{VPS_DB}"
    tmp = LOCAL_DB.with_suffix(".db.tmp")
    # BatchMode=yes: gagal cepat bila key-auth belum ada (jangan hang nunggu password)
    cmd = ["scp", "-o", "BatchMode=yes",
           "-o", "StrictHostKeyChecking=accept-new", "-o", "ConnectTimeout=15"]
    if SSH_KEY:
        cmd += ["-i", SSH_KEY]
    cmd += [remote, str(tmp)]

    print(f"[bridge] scp {remote} -> {LOCAL_DB} ...")
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        if tmp.exists():
            tmp.unlink()
        raise RuntimeError(
            f"scp gagal (exit {res.returncode}): {res.stderr.strip() or res.stdout.strip()}\n"
            f"  Pastikan auth SSH key-based ke {remote} aktif (set SWINT_VPS_SSH_KEY bila perlu)."
        )
    # atomic replace — cache tidak pernah setengah-tertulis
    os.replace(tmp, LOCAL_DB)
    size_kb = LOCAL_DB.stat().st_size / 1024
    print(f"[bridge] OK - {size_kb:.0f} KB tersalin ke {LOCAL_DB}")
    return LOCAL_DB


# ── Baca DB (read-only) ───────────────────────────────────────────────────────

def _connect(db_path: Path = LOCAL_DB) -> sqlite3.Connection:
    if not db_path.exists():
        raise FileNotFoundError(
            f"Cache DB tidak ada: {db_path}. Jalankan pull_live_db() / CLI tanpa --no-pull dulu."
        )
    # mode=ro: tidak akan pernah memodifikasi cache
    return sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)


def load_trades(db_path: Path = LOCAL_DB) -> pd.DataFrame:
    """DataFrame trade lengkap + symbol, model_type, conf, atr, feature_snapshot."""
    sql = """
        SELECT t.*,
               c.symbol           AS coin_symbol,
               mm.model_type      AS model_type,
               s.confidence       AS signal_confidence,
               s.atr_at_signal    AS atr_at_signal,
               s.feature_snapshot AS feature_snapshot
        FROM trade t
        JOIN coin c            ON t.coin_id = c.id
        LEFT JOIN signal s     ON t.signal_id = s.id
        LEFT JOIN model_meta mm ON s.model_meta_id = mm.id
        ORDER BY t.opened_at
    """
    with _connect(db_path) as con:
        return pd.read_sql_query(sql, con)


def load_signals(db_path: Path = LOCAL_DB) -> pd.DataFrame:
    """DataFrame signal lengkap + symbol, model_type."""
    sql = """
        SELECT s.*,
               c.symbol      AS coin_symbol,
               mm.model_type AS model_type
        FROM signal s
        JOIN coin c            ON s.coin_id = c.id
        LEFT JOIN model_meta mm ON s.model_meta_id = mm.id
        ORDER BY s.signal_time
    """
    with _connect(db_path) as con:
        return pd.read_sql_query(sql, con)


# ── Export CSV (mirror app/api/trades.py::trades_export_csv) ───────────────────

def _parse_feature(fs_json, key, default=""):
    if not fs_json:
        return default
    try:
        d = json.loads(fs_json) if isinstance(fs_json, str) else fs_json
        return d.get(key, default)
    except (json.JSONDecodeError, AttributeError, TypeError):
        return default


def _fmt_price(v):
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return ""
    # cukup presisi untuk dibaca ulang oleh trade_analyzer (float())
    return f"{float(v):.8g}"


def export_livetrading_csv(out_path: Path = LIVE_CSV, db_path: Path = LOCAL_DB) -> Path:
    """Tulis CSV 25-kolom identik web app dari data live (semua trade)."""
    df = load_trades(db_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cols = ["Opened", "Closed", "Coin", "Model", "Direction", "Conf", "Entry", "Exit",
            "TP", "SL", "ATR", "% to H4Hi", "% to H4Lo", "RR",
            "H4 Trend", "Vol Regime", "H4 High", "H4 Low", "Qty", "Leverage",
            "PnL ($)", "PnL (%)", "Exit Reason", "Hold Bars", "Status"]
    rows = []
    for r in df.itertuples(index=False):
        d = r._asdict()
        entry = d.get("entry_price") or 0
        h4_hi = d.get("h4_swing_high") or 0
        h4_lo = d.get("h4_swing_low") or 0
        pct_hi = f"{((h4_hi / entry) - 1) * 100:+.1f}" if h4_hi and entry else ""
        pct_lo = f"{(1 - (h4_lo / entry)) * 100:+.1f}" if h4_lo and entry else ""
        tp_dist = abs((d.get("tp_price") or entry) - entry)
        sl_dist = abs((d.get("sl_price") or entry) - entry)
        rr_val = f"{tp_dist / sl_dist:.2f}" if sl_dist > 0 and tp_dist > 0 else ""
        conf_v = d.get("signal_confidence")
        conf = f"{conf_v:.2f}" if conf_v else ""
        atr_v = d.get("atr_at_signal")
        atr = _fmt_price(atr_v) if atr_v else ""
        h4_trend = _parse_feature(d.get("feature_snapshot"), "h4_trend", "")
        vol_reg = _parse_feature(d.get("feature_snapshot"), "vol_regime", "")
        trend_str = "UP" if h4_trend == 1 else ("DOWN" if h4_trend == -1 else ("RANGE" if h4_trend == 0 else ""))
        vol_str = f"{vol_reg:.2f}" if isinstance(vol_reg, (int, float)) else ""

        def _dtfmt(v):
            if not v:
                return ""
            s = str(v)
            # opened_at/closed_at tersimpan ISO "YYYY-MM-DD HH:MM:SS[.ffffff]"
            return s[:16]  # -> "YYYY-MM-DD HH:MM"

        pnl_net = d.get("pnl_net")
        pnl_pct = d.get("pnl_pct")
        qty = d.get("quantity")
        lev = d.get("leverage")
        hold = d.get("hold_bars")
        rows.append([
            _dtfmt(d.get("opened_at")),
            _dtfmt(d.get("closed_at")),
            d.get("coin_symbol", ""),
            d.get("model_type") or "",
            d.get("direction", ""),
            conf,
            _fmt_price(d.get("entry_price")),
            _fmt_price(d.get("exit_price")) if d.get("exit_price") else "",
            _fmt_price(d.get("tp_price")) if d.get("tp_price") else "",
            _fmt_price(d.get("sl_price")) if d.get("sl_price") else "",
            atr, pct_hi, pct_lo, rr_val, trend_str, vol_str,
            _fmt_price(d.get("h4_swing_high")) if d.get("h4_swing_high") else "",
            _fmt_price(d.get("h4_swing_low")) if d.get("h4_swing_low") else "",
            f"{qty:.4f}" if qty else "",
            lev if lev else "",
            f"{pnl_net:+.2f}" if pnl_net is not None else "",
            f"{pnl_pct:+.1f}" if pnl_pct is not None else "",
            d.get("exit_reason") or "",
            hold if hold is not None else "",
            d.get("status", ""),
        ])
    out = pd.DataFrame(rows, columns=cols)
    out.to_csv(out_path, index=False)
    return out_path


# ── Ringkasan ─────────────────────────────────────────────────────────────────

def print_summary(db_path: Path = LOCAL_DB):
    sig = load_signals(db_path)
    trd = load_trades(db_path)
    print("\n================= RINGKASAN DB LIVE =================")
    print(f"Sumber cache : {db_path}")
    if db_path.exists():
        age = (time.time() - db_path.stat().st_mtime) / 60.0
        print(f"Umur cache   : {age:.1f} menit")
    print(f"Signals      : {len(sig)}")
    if len(sig):
        print(f"  rentang    : {sig['signal_time'].min()}  ->  {sig['signal_time'].max()}")
    print(f"Trades       : {len(trd)}")
    if len(trd):
        live_n = int((trd.get('is_live', pd.Series(dtype=int)) == 1).sum())
        open_n = int((trd['status'] == 'open').sum()) if 'status' in trd else 0
        print(f"  live / open: {live_n} live, {open_n} open")
        print(f"  rentang    : {trd['opened_at'].min()}  ->  {trd['opened_at'].max()}")
        if 'pnl_net' in trd:
            closed = trd[trd['status'] == 'closed']
            if len(closed):
                net = closed['pnl_net'].dropna().sum()
                wins = (closed['pnl_net'] > 0).sum()
                wr = wins / len(closed) * 100 if len(closed) else 0
                print(f"  closed PnL : {net:+.2f}  | WR {wr:.1f}% ({wins}/{len(closed)})")
    print("====================================================\n")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Jembatan baca DB live signal/trade dari VPS")
    ap.add_argument("--no-pull", action="store_true", help="Pakai cache lokal, jangan scp dari VPS")
    ap.add_argument("--summary", action="store_true", help="Cetak ringkasan saja (tanpa export CSV)")
    ap.add_argument("--export-csv", nargs="?", const=str(LIVE_CSV), default=None,
                    help=f"Export CSV format trade_analyzer (default: {LIVE_CSV})")
    ap.add_argument("--max-age-min", type=float, default=10.0,
                    help="Jika cache lebih muda dari ini (mnt), --no-pull aware lewati scp")
    args = ap.parse_args()

    try:
        if not args.no_pull:
            pull_live_db(force=True)
    except RuntimeError as e:
        print(f"[bridge] ERROR: {e}", file=sys.stderr)
        if not LOCAL_DB.exists():
            sys.exit(1)
        print("[bridge] Lanjut pakai cache lama yang ada.", file=sys.stderr)

    print_summary()

    # Default (tanpa flag spesifik): export CSV juga agar trade_analyzer siap pakai
    do_export = args.export_csv is not None or not args.summary
    if do_export:
        target = Path(args.export_csv) if args.export_csv else LIVE_CSV
        p = export_livetrading_csv(target)
        n = len(pd.read_csv(p))
        print(f"[bridge] CSV live diekspor: {p}  ({n} trade)")
        print(f"[bridge] -> trade_analyzer.py akan mendeteksinya otomatis bila path ini diprioritaskan.")


if __name__ == "__main__":
    main()
