# -*- coding: utf-8 -*-
"""
tools/live_db_bridge.py — Jembatan baca DB live (signal + trade) dari Web App VPS.

Web App produksi (swint_tradev2) menyimpan signal & trade di SQLite pada VPS:
    /home/swint/swint_tradev2/instance/app.db
File lokal `E:\\Widyawardhana_Capital\\swint_tradev2\\instance\\app.db` adalah DB dev lama (BASI) —
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

─── FILTER LIVE (jangan tertukar) ─────────────────────────────────────────────

UI scorecard "Selama Model Aktif" (swint: model_history_bp._scorecard_for_idx):
  - status == "closed"
  - closed_at >= inference_config._snapshot_time (deploy model aktif)
  - closed_at < now
  - BUKAN filter opened_at rentang kalender WITA

Halaman Trades (swint: trades.py _apply_trade_filters):
  - date_by=opened (default) atau closed
  - date_from/date_to = batas hari kalender WITA

Analisis parity holdout vs live → pakai filter_trades_model_active() agar angka
WR/PF/PnL sama dengan corecard UI. Jangan pakai opened_at 24–26 WITA untuk
memvalidasi scorecard (akan under-count & PnL salah).

CSV hasil_livetrading.csv: kolom Opened/Closed = UTC mentah (bukan WITA).
UI menampilkan WITA via wita_fmt — beda 8 jam dari CSV.
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

REPO_DIR       = Path(__file__).resolve().parents[2]
SWINT_LOCAL_DB = Path(os.environ.get(
    "SWINT_LOCAL_DB",
    r"E:\Widyawardhana_Capital\swint_tradev2\instance\app.db",
))
CACHE_DIRS = {
    "vps":   REPO_DIR / "data" / "live_cache",
    "local": REPO_DIR / "data" / "paper_cache",
}
# Back-compat aliases (default = VPS live)
CACHE_DIR = CACHE_DIRS["vps"]
LOCAL_DB  = CACHE_DIR / "app.db"
LIVE_CSV  = CACHE_DIR / "hasil_livetrading.csv"
PAPER_CSV = CACHE_DIRS["local"] / "hasil_papertrading.csv"


def resolve_cache(source: str = "vps") -> tuple[Path, Path, Path]:
    """Return (cache_dir, db_path, csv_path) untuk source vps|local."""
    if source not in CACHE_DIRS:
        raise ValueError(f"source harus vps|local, got {source!r}")
    cache = CACHE_DIRS[source]
    csv_name = "hasil_livetrading.csv" if source == "vps" else "hasil_papertrading.csv"
    return cache, cache / "app.db", cache / csv_name


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


def sync_local_db(
    src: Path | None = None,
    *,
    dest: Path | None = None,
    force: bool = True,
    max_age_min: float = 5.0,
) -> Path:
    """Salin app.db swint lokal ke paper_cache (untuk analisis paper soak)."""
    src = src or SWINT_LOCAL_DB
    cache_dir, db_path, _ = resolve_cache("local")
    dest = dest or db_path
    cache_dir.mkdir(parents=True, exist_ok=True)

    if not src.exists():
        raise FileNotFoundError(
            f"DB swint lokal tidak ada: {src}\n"
            f"  Jalankan swint lokal (TRADING_MODE=paper) dulu agar instance/app.db terisi."
        )

    if not force and dest.exists():
        age_min = (time.time() - dest.stat().st_mtime) / 60.0
        if age_min < max_age_min:
            print(f"[bridge] Paper cache masih segar ({age_min:.1f} mnt) — skip copy.")
            return dest

    tmp = dest.with_suffix(".db.tmp")
    shutil.copy2(src, tmp)
    os.replace(tmp, dest)
    size_kb = dest.stat().st_size / 1024
    print(f"[bridge] local {src} -> {dest} ({size_kb:.0f} KB)")
    return dest


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


WITA_TZ = "Asia/Makassar"
INFERENCE_CONFIG = REPO_DIR / "models" / "inference_config.json"
PAPER_INFERENCE_CONFIG = CACHE_DIRS["local"] / "inference_config_paper_candidate.json"


def load_inference_config(path: Path | None = None) -> dict:
    p = path or INFERENCE_CONFIG
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def load_model_snapshot_start(config_path: Path | None = None) -> pd.Timestamp:
    """Waktu deploy model aktif — awal periode scorecard UI (UTC)."""
    default = pd.Timestamp("2026-06-24 00:00:00", tz="UTC")
    cfg_path = config_path or INFERENCE_CONFIG
    data = load_inference_config(cfg_path)
    if not data:
        return default
    try:
        raw = data.get("_snapshot_time") or data.get("snapshot_at")
        if not raw:
            return default
        ts = pd.Timestamp(str(raw).replace("Z", "+00:00"))
        return ts.tz_convert("UTC") if ts.tzinfo else ts.tz_localize("UTC")
    except Exception:
        return default


def prepare_trades_df(tr: pd.DataFrame) -> pd.DataFrame:
    """Normalisasi kolom waktu + alias untuk script analisis."""
    out = tr.copy()
    out["opened_at"] = pd.to_datetime(out["opened_at"], utc=True)
    out["closed_at"] = pd.to_datetime(out["closed_at"], utc=True, errors="coerce")
    out["entry_date_wita"] = out["opened_at"].dt.tz_convert(WITA_TZ).dt.strftime("%Y-%m-%d")
    out["entry_time_wita"] = out["opened_at"].dt.tz_convert(WITA_TZ).dt.strftime("%Y-%m-%d %H:%M")
    out["exit_time_wita"] = out["closed_at"].dt.tz_convert(WITA_TZ).dt.strftime("%Y-%m-%d %H:%M")
    out["coin"] = out["coin_symbol"]
    out["conf"] = out["signal_confidence"]
    out["pnl"] = out["pnl_net"]
    out["outcome"] = out["exit_reason"]
    out.rename(columns={"entry_price": "entry", "exit_price": "exit"}, inplace=True)
    return out


def _apply_scope_mask(df: pd.DataFrame, scope: str) -> pd.Series:
    """scope: live (is_live=1), paper (is_live=0), all (tanpa filter)."""
    if scope == "all" or "is_live" not in df.columns:
        return pd.Series(True, index=df.index)
    if scope == "live":
        return df["is_live"] == 1
    if scope == "paper":
        return df["is_live"] == 0
    raise ValueError(f"scope harus live|paper|all, got {scope!r}")


def filter_trades_model_active(
    tr: pd.DataFrame,
    *,
    model_type: str | None = "ic32_regime_v4",
    live_only: bool | None = None,
    scope: str = "live",
    snapshot_start: pd.Timestamp | None = None,
    snapshot_end: pd.Timestamp | None = None,
    config_path: Path | None = None,
) -> pd.DataFrame:
    """
    Replikasi scorecard UI 'Selama Model Aktif' — filter closed_at, bukan opened_at.

    Ref: swint_tradev2/app/api/model_history_bp.py::_scorecard_for_idx
    """
    if live_only is not None:
        scope = "live" if live_only else "all"
    df = prepare_trades_df(tr)
    start = snapshot_start or load_model_snapshot_start(config_path)
    end = snapshot_end or pd.Timestamp.now(tz="UTC")
    mask = (df["status"] == "closed") & (df["closed_at"] >= start) & (df["closed_at"] < end)
    mask &= _apply_scope_mask(df, scope)
    if model_type and "model_type" in df.columns:
        mask &= df["model_type"] == model_type
    return df.loc[mask].copy()


def filter_trades_opened_wita(
    tr: pd.DataFrame,
    start_date: str,
    end_date: str,
    *,
    model_type: str | None = "ic32_regime_v4",
    live_only: bool | None = None,
    scope: str = "live",
) -> pd.DataFrame:
    """Filter opened_at [start_date, end_date] kalender WITA — untuk audit window entry."""
    if live_only is not None:
        scope = "live" if live_only else "all"
    df = prepare_trades_df(tr)
    entry_start = pd.Timestamp(start_date, tz=WITA_TZ).tz_convert("UTC")
    entry_end = (pd.Timestamp(end_date, tz=WITA_TZ) + pd.Timedelta(days=1)).tz_convert("UTC")
    mask = (df["opened_at"] >= entry_start) & (df["opened_at"] < entry_end)
    mask &= _apply_scope_mask(df, scope)
    if model_type and "model_type" in df.columns:
        mask &= df["model_type"] == model_type
    return df.loc[mask].copy()


def compute_trade_metrics(df: pd.DataFrame, pnl_col: str = "pnl_net") -> dict:
    """WR, PF, expectancy — selaras swint reporting.compute_trade_metrics (inti)."""
    if df.empty or pnl_col not in df.columns:
        return {
            "total": 0, "wins": 0, "losses": 0, "win_rate": 0.0,
            "profit_factor": 0.0, "net_pnl": 0.0, "expectancy": 0.0,
        }
    pnls = df[pnl_col].fillna(0).astype(float)
    wins = pnls[pnls > 0]
    losses = pnls[pnls < 0]
    total = len(pnls)
    n_wins, n_losses = len(wins), len(losses)
    net = float(pnls.sum())
    gp, gl = float(wins.sum()), abs(float(losses.sum()))
    pf = round(gp / gl, 2) if gl > 0 else (None if gp > 0 else 0.0)
    return {
        "total": total,
        "wins": n_wins,
        "losses": n_losses,
        "win_rate": round(n_wins / total * 100, 1) if total else 0.0,
        "profit_factor": pf,
        "net_pnl": round(net, 2),
        "expectancy": round(net / total, 3) if total else 0.0,
    }


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


def _parse_lgbm_lean(feature_snapshot: str | dict | None) -> dict:
    """Extract LGBM directional lean/proba from feature_snapshot JSON.
    This allows filtering 'condong kemana' (LONG/SHORT lean) even when
    final direction=FLAT or below threshold or no trade.
    """
    out = {
        "lgbm_p_short": None,
        "lgbm_p_flat": None,
        "lgbm_p_long": None,
        "lgbm_lean": None,          # SHORT | FLAT | LONG (from raw LGBM argmax)
        "lgbm_lean_conf": None,
        "lgbm_decision": None,
        "cascade_stage": None,
        "thr_lgbm_long": None,
        "thr_lgbm_short": None,
        "thr_entry": None,
        "hmm_enc": None,
    }
    if feature_snapshot is None or (isinstance(feature_snapshot, float) and pd.isna(feature_snapshot)):
        return out
    try:
        if isinstance(feature_snapshot, str):
            d = json.loads(feature_snapshot)
        elif isinstance(feature_snapshot, dict):
            d = feature_snapshot
        else:
            return out
        proba = d.get("_lgbm_proba") or d.get("lgbm_proba")
        if proba and len(proba) == 3:
            ps, pf, pl = float(proba[0]), float(proba[1]), float(proba[2])
            out["lgbm_p_short"] = ps
            out["lgbm_p_flat"] = pf
            out["lgbm_p_long"] = pl
            if pl > max(ps, pf):
                out["lgbm_lean"] = "LONG"
            elif ps > max(pf, pl):
                out["lgbm_lean"] = "SHORT"
            else:
                out["lgbm_lean"] = "FLAT"
            out["lgbm_lean_conf"] = max(ps, pf, pl)
        out["lgbm_decision"] = d.get("_lgbm_decision")
        out["cascade_stage"] = d.get("_cascade_stage")
        out["thr_lgbm_long"] = d.get("_thr_lgbm_long")
        out["thr_lgbm_short"] = d.get("_thr_lgbm_short")
        out["thr_entry"] = d.get("_thr_entry")
        out["hmm_enc"] = d.get("_hmm_enc") or d.get("hmm_regime_enc")
    except (json.JSONDecodeError, TypeError, ValueError, AttributeError):
        pass
    return out


def load_signals_with_lean(db_path: Path = LOCAL_DB) -> pd.DataFrame:
    """Signals + structured lean columns for easy filtering by bias/condong.

    Use this when you want to filter signals that were *leaning* LONG or SHORT
    (from LGBM proba), even if final direction=FLAT, below threshold,
    rejected by RR/structural, or did not result in a trade.

    Adds columns:
      lgbm_p_short, lgbm_p_flat, lgbm_p_long
      lgbm_lean (LONG/SHORT/FLAT from raw LGBM)
      lgbm_lean_conf
      cascade_stage, lgbm_decision, thr_* etc.

    Example usage:
      sig = load_signals_with_lean()
      long_lean = sig[sig['lgbm_lean'] == 'LONG']   # includes sub-thr and rejects
      print(long_lean[long_lean['direction'] == 'FLAT'][['symbol', 'lgbm_p_long', 'entry_reason']])
    """
    df = load_signals(db_path)
    lean_data = df["feature_snapshot"].apply(_parse_lgbm_lean).apply(pd.Series)
    # Fallback lean from entry_reason text if snapshot missing proba (e.g. older records)
    def _lean_from_reason(row):
        if row.get("lgbm_lean"):
            return row
        er = str(row.get("entry_reason") or "")
        import re
        m = re.search(r"LGBM=(LONG|SHORT)\(([\d.]+)\)", er)
        if m:
            side, conf = m.group(1), float(m.group(2))
            row["lgbm_lean"] = side
            row["lgbm_lean_conf"] = conf
            if side == "LONG":
                row["lgbm_p_long"] = conf
            else:
                row["lgbm_p_short"] = conf
        return row

    enriched = pd.concat([df.reset_index(drop=True), lean_data.reset_index(drop=True)], axis=1)
    enriched = enriched.apply(_lean_from_reason, axis=1)
    # convenient boolean filters
    enriched["is_long_lean"] = enriched["lgbm_lean"] == "LONG"
    enriched["is_short_lean"] = enriched["lgbm_lean"] == "SHORT"
    enriched["is_directional_lean"] = enriched["lgbm_lean"].isin(["LONG", "SHORT"])
    return enriched


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

def print_summary(db_path: Path = LOCAL_DB, *, label: str = "LIVE", scope: str = "live"):
    sig = load_signals(db_path)
    trd = load_trades(db_path)
    print(f"\n================= RINGKASAN DB {label} =================")
    print(f"Sumber cache : {db_path}")
    print(f"Scope filter : {scope}")
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
            active = filter_trades_model_active(trd, scope=scope)
            if len(active):
                m = compute_trade_metrics(active)
                snap = load_model_snapshot_start().strftime("%Y-%m-%d")
                pf_s = f"{m['profit_factor']:.2f}" if m['profit_factor'] is not None else "—"
                print(f"  scorecard  : closed_at>={snap} (model aktif)")
                print(f"    N={m['total']} WR={m['win_rate']}% PF={pf_s} PnL=${m['net_pnl']:+.2f}")
    print("====================================================\n")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Jembatan baca DB signal/trade (VPS live atau swint lokal paper)")
    ap.add_argument(
        "--source",
        choices=("vps", "local"),
        default="vps",
        help="vps=scp VPS live_cache (default); local=copy swint instance/app.db ke paper_cache",
    )
    ap.add_argument(
        "--scope",
        choices=("live", "paper", "all"),
        default=None,
        help="Filter is_live: live (default utk vps), paper (default utk local), all",
    )
    ap.add_argument("--no-pull", action="store_true", help="Pakai cache, jangan fetch ulang")
    ap.add_argument("--summary", action="store_true", help="Cetak ringkasan saja (tanpa export CSV)")
    ap.add_argument("--export-csv", nargs="?", const=None, default=None,
                    help="Export CSV format trade_analyzer (default: sesuai --source)")
    ap.add_argument("--max-age-min", type=float, default=10.0,
                    help="Jika cache lebih muda dari ini (mnt), --no-pull aware lewati fetch")
    args = ap.parse_args()

    scope = args.scope or ("paper" if args.source == "local" else "live")
    cache_dir, db_path, default_csv = resolve_cache(args.source)
    label = "PAPER (local)" if args.source == "local" else "LIVE (VPS)"

    if not db_path.exists() and args.no_pull:
        print("[bridge] Cache belum ada — fetch sekali meski --no-pull.", file=sys.stderr)
        args.no_pull = False

    try:
        if not args.no_pull:
            if args.source == "vps":
                pull_live_db(force=True)
            else:
                sync_local_db(force=True, max_age_min=args.max_age_min)
    except (RuntimeError, FileNotFoundError) as e:
        print(f"[bridge] ERROR: {e}", file=sys.stderr)
        if not db_path.exists():
            sys.exit(1)
        print("[bridge] Lanjut pakai cache lama yang ada.", file=sys.stderr)

    print_summary(db_path, label=label, scope=scope)

    do_export = args.export_csv is not None or not args.summary
    if do_export:
        target = Path(args.export_csv) if args.export_csv else default_csv
        p = export_livetrading_csv(target, db_path=db_path)
        n = len(pd.read_csv(p))
        print(f"[bridge] CSV diekspor: {p}  ({n} trade)")
        if args.source == "vps":
            print("[bridge] -> trade_analyzer.py mendeteksi hasil_livetrading.csv otomatis.")


if __name__ == "__main__":
    main()
