"""
tools/refresh_pbull.py — Daily p_bull refresh automation.

Alur:
  1. Fetch L/S terbaru dari Binance (01c append + backfill_binance_daily_ls --append)
  2. Hitung p_bull harian terkini (compute_daily_pbull --use-binance --lstm-run ...)
  3. SCP daily_pbull.parquet ke VPS
  4. VPS reload artefak (SIGHUP / endpoint)

Dijalankan setiap hari setelah daily bar tutup (disarankan 00:15 UTC = 07:15 WIB).

Usage:
  python tools/refresh_pbull.py                         # default model (coinank, v1)
  python tools/refresh_pbull.py --use-binance           # Binance L/S, v1 model
  python tools/refresh_pbull.py --use-binance --lstm-run ic32_daily_lstm_17f_s30_v2
  python tools/refresh_pbull.py --dry-run               # lihat langkah tanpa eksekusi

Registrasi Windows Task Scheduler (jalankan sekali sebagai admin):
  schtasks /Create /TN "RefreshPBull" /TR "python D:\\Apps-Dev\\Riset_pemodelan\\tools\\refresh_pbull.py --use-binance --lstm-run ic32_daily_lstm_17f_s30_v2" /SC DAILY /ST 07:20 /F

Atau pakai run_refresh_pbull.ps1 di folder tools.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT      = Path(__file__).resolve().parents[2]
TOOLS_DIR = ROOT / "tools" / "data"

VPS_HOST    = os.environ.get("SWINT_VPS_HOST", "139.180.157.176")
VPS_USER    = os.environ.get("SWINT_VPS_USER", "root")
SSH_KEY     = os.environ.get("SWINT_VPS_SSH_KEY", "")
VPS_APP_DIR = "/home/swint/swint_tradev2"

DEFAULT_LSTM_RUN = "ic32_daily_lstm_17f_s30"
PBULL_OUT        = ROOT / "models" / "daily_pbull.parquet"


def _ssh_opts() -> list[str]:
    opts = ["-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=accept-new",
            "-o", "ConnectTimeout=20"]
    if SSH_KEY:
        opts += ["-i", SSH_KEY]
    return opts


def _run(cmd: list[str], *, cwd: Path | None = None, dry_run: bool = False) -> int:
    label = " ".join(cmd)
    print(f"  $ {label}")
    if dry_run:
        return 0
    res = subprocess.run(cmd, cwd=str(cwd) if cwd else None)
    return res.returncode


def step_fetch_ls(dry_run: bool) -> bool:
    print("\n[1/4] Fetch L/S Binance terbaru (append)...")
    cmd = [sys.executable, str(TOOLS_DIR / "backfill_binance_daily_ls.py"), "--append"]
    return _run(cmd, cwd=ROOT, dry_run=dry_run) == 0


def step_compute_pbull(lstm_run: str, use_binance: bool, dry_run: bool) -> bool:
    print(f"\n[2/4] Hitung p_bull ({lstm_run}, binance={use_binance})...")
    cmd = [sys.executable, str(TOOLS_DIR / "compute_daily_pbull.py")]
    if use_binance:
        cmd.append("--use-binance")
    cmd += ["--lstm-run", lstm_run]
    return _run(cmd, cwd=ROOT, dry_run=dry_run) == 0


def step_scp(dry_run: bool) -> bool:
    print("\n[3/4] SCP daily_pbull.parquet ke VPS...")
    if not dry_run and not PBULL_OUT.exists():
        print(f"  [ERROR] {PBULL_OUT} tidak ditemukan — hitung dulu.")
        return False
    remote_dir = f"{VPS_APP_DIR}/models"
    remote     = f"{VPS_USER}@{VPS_HOST}:{remote_dir}/daily_pbull.parquet"
    # Pastikan direktori ada di VPS
    ssh_mkdir  = ["ssh", *_ssh_opts(), f"{VPS_USER}@{VPS_HOST}", f"mkdir -p '{remote_dir}'"]
    _run(ssh_mkdir, dry_run=dry_run)
    cmd = ["scp", *_ssh_opts(), str(PBULL_OUT), remote]
    return _run(cmd, dry_run=dry_run) == 0


def step_reload_vps(dry_run: bool) -> bool:
    print("\n[4/4] VPS: reload daily_pbull_service...")
    remote = f"{VPS_USER}@{VPS_HOST}"
    # Kirim SIGHUP ke proses gunicorn/uvicorn agar service reload artefak (lazy-load via mtime)
    # Tidak perlu restart penuh — daily_pbull_service.py sudah cek mtime setiap request
    cmd = ["ssh", *_ssh_opts(), remote,
           "kill -HUP $(cat /home/swint/swint_tradev2/swint.pid 2>/dev/null) 2>/dev/null || true"]
    rc = _run(cmd, dry_run=dry_run)
    print("  [OK] VPS notified (mtime-reload otomatis pada request berikutnya).")
    return True  # non-fatal jika SIGHUP gagal (lazy mtime reload tetap jalan)


def main() -> int:
    ap = argparse.ArgumentParser(description="Daily p_bull refresh: fetch -> compute -> scp -> VPS")
    ap.add_argument("--use-binance", action="store_true",
                    help="Pakai Binance-direct L/S (butuh backfill_binance_daily_ls data)")
    ap.add_argument("--lstm-run",    default=None,
                    help=f"Override LSTM run (default: {DEFAULT_LSTM_RUN})")
    ap.add_argument("--skip-fetch",  action="store_true",
                    help="Lewati langkah 1 (fetch L/S) — pakai data yang sudah ada")
    ap.add_argument("--local-only",  action="store_true",
                    help="Stop setelah compute (tanpa scp/VPS)")
    ap.add_argument("--dry-run",     action="store_true")
    args = ap.parse_args()

    lstm_run    = args.lstm_run or DEFAULT_LSTM_RUN
    use_binance = args.use_binance

    now = datetime.now(timezone.utc)
    print("=" * 60)
    print(f"  REFRESH PBULL — {now.strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"  LSTM: {lstm_run} | L/S: {'Binance-direct' if use_binance else 'coinank'}")
    print(f"  VPS : {VPS_USER}@{VPS_HOST}")
    if args.dry_run:
        print("  MODE: DRY-RUN")
    print("=" * 60)

    if not args.skip_fetch and use_binance:
        if not step_fetch_ls(args.dry_run):
            print("\n[WARN] Fetch L/S gagal — lanjut dengan data yang ada.")

    if not step_compute_pbull(lstm_run, use_binance, args.dry_run):
        print("\n[FAIL] compute_daily_pbull gagal.")
        return 1

    if args.local_only:
        print("\n[DONE] Local-only — parquet di models/daily_pbull.parquet")
        return 0

    if not step_scp(args.dry_run):
        print("\n[FAIL] SCP ke VPS gagal.")
        return 1

    step_reload_vps(args.dry_run)

    print("\n" + "=" * 60)
    print("  REFRESH SELESAI")
    print(f"  Cek: curl http://{VPS_HOST}:5000/api/health")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
