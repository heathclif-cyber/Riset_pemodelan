# -*- coding: utf-8 -*-
"""
deploy_production.py — Satu perintah: Riset -> swint lokal -> VPS live.

Alur otomatis:
  1. tools/deploy_model.py  (salin ke swint_tradev2 lokal + backup)
  2. git commit + push      (hanya file kode .py / .md)
  3. scp model & config     (binary + inference_config — tidak di git)
  4. VPS update.sh          (git pull + chown + restart)

Usage:
  python tools/deploy_production.py
  python tools/deploy_production.py --message "guardian continuation v1"
  python tools/deploy_production.py --code-only     # skip scp model (kode saja)
  python tools/deploy_production.py --models-only   # skip git push (model scp saja)
  python tools/deploy_production.py --local-only    # stop setelah salin ke swint lokal
  python tools/deploy_production.py --dry-run

Env (sama dengan live_db_bridge):
  SWINT_VPS_HOST, SWINT_VPS_USER, SWINT_VPS_SSH_KEY
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parents[1]
TOOLS_DIR = REPO_DIR / "tools"
SWINT_DIR = Path(r"D:\Apps-Dev\swint_tradev2")

VPS_HOST = os.environ.get("SWINT_VPS_HOST", "139.180.157.176")
VPS_USER = os.environ.get("SWINT_VPS_USER", "root")
SSH_KEY = os.environ.get("SWINT_VPS_SSH_KEY", "")
VPS_APP_DIR = "/home/swint/swint_tradev2"

# Import mapping dari deploy_model (single source of truth)
sys.path.insert(0, str(TOOLS_DIR))
from deploy_model import DEPLOY_MAPPING, TARGET_REPO_DIR  # noqa: E402

CODE_SUFFIXES = (".py", ".md")
MODEL_SUFFIXES = (".pkl", ".pt", ".json")


def _ssh_opts() -> list[str]:
    opts = ["-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=accept-new", "-o", "ConnectTimeout=20"]
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


def _classify_deploy_paths() -> tuple[list[str], list[str]]:
    code_paths: list[str] = []
    model_paths: list[str] = []
    seen: set[str] = set()
    for rel in DEPLOY_MAPPING.values():
        if rel in seen:
            continue
        seen.add(rel)
        if rel.endswith(CODE_SUFFIXES):
            code_paths.append(rel)
        elif rel.endswith(MODEL_SUFFIXES):
            model_paths.append(rel)
    return code_paths, model_paths


def step_local_deploy(dry_run: bool) -> bool:
    print("\n[1/4] Salin Riset -> swint_tradev2 lokal...")
    cmd = [sys.executable, str(TOOLS_DIR / "deploy_model.py")]
    if dry_run:
        print("  (dry-run: lewati deploy_model.py)")
        return True
    return _run(cmd, cwd=REPO_DIR) == 0


def step_git_push(
    message: str,
    code_paths: list[str],
    dry_run: bool,
    *,
    all_tracked: bool = False,
) -> bool:
    print("\n[2/4] Git push kode ke GitHub...")
    swint = Path(TARGET_REPO_DIR)
    if not swint.exists():
        print(f"  [ERROR] swint tidak ditemukan: {swint}")
        return False

    if all_tracked:
        if dry_run:
            print("  (dry-run) git add -u (semua perubahan tracked)")
            return True
        if _run(["git", "add", "-u"], cwd=swint) != 0:
            return False
    else:
        existing = [p for p in code_paths if (swint / p).exists()]
        if not existing:
            print("  [INFO] Tidak ada file kode deploy — lewati git.")
            return True
        if dry_run:
            print(f"  (dry-run) git add: {existing}")
            return True
        for rel in existing:
            if _run(["git", "add", rel], cwd=swint) != 0:
                return False

    status = subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=str(swint))
    if status.returncode == 0:
        print("  [INFO] Tidak ada perubahan kode — lewati commit.")
        return True

    if _run(["git", "commit", "-m", message], cwd=swint) != 0:
        return False
    if _run(["git", "push", "origin", "main"], cwd=swint) != 0:
        return False
    print("  [OK] Kode ter-push.")
    return True


def _remote_mkdir(remote_dir: str, dry_run: bool) -> None:
    remote = f"{VPS_USER}@{VPS_HOST}"
    cmd = ["ssh", *_ssh_opts(), remote, f"mkdir -p '{remote_dir}'"]
    _run(cmd, dry_run=dry_run)


def step_scp_models(model_paths: list[str], dry_run: bool) -> bool:
    print("\n[3/4] SCP model & config ke VPS...")
    swint = Path(TARGET_REPO_DIR)
    ok = True
    for rel in model_paths:
        local = swint / rel
        if not local.exists():
            print(f"  [SKIP] Tidak ada: {rel}")
            continue
        remote_dir = f"{VPS_APP_DIR}/{Path(rel).parent.as_posix()}"
        _remote_mkdir(remote_dir, dry_run)
        remote = f"{VPS_USER}@{VPS_HOST}:{VPS_APP_DIR}/{rel.replace(chr(92), '/')}"
        cmd = ["scp", *_ssh_opts(), str(local), remote]
        if _run(cmd, dry_run=dry_run) != 0:
            ok = False
    if ok:
        print("  [OK] Model & config ter-upload.")
    return ok


def step_vps_update(dry_run: bool) -> bool:
    print("\n[4/4] VPS: pull + chown + restart...")
    remote = f"{VPS_USER}@{VPS_HOST}"
    cmd = [
        "ssh", *_ssh_opts(), remote,
        f"bash {VPS_APP_DIR}/deploy/update.sh",
    ]
    return _run(cmd, dry_run=dry_run) == 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Deploy seamless Riset -> VPS")
    ap.add_argument("--message", "-m", default="", help="Git commit message")
    ap.add_argument("--code-only", action="store_true", help="Hanya git push + VPS restart (tanpa scp model)")
    ap.add_argument("--models-only", action="store_true", help="Hanya scp model + restart (tanpa git push)")
    ap.add_argument("--local-only", action="store_true", help="Stop setelah salin ke swint lokal")
    ap.add_argument("--dry-run", action="store_true", help="Tampilkan langkah tanpa eksekusi")
    args = ap.parse_args()

    msg = args.message.strip() or f"deploy: {datetime.now().strftime('%Y-%m-%d %H:%M')}"
    code_paths, model_paths = _classify_deploy_paths()

    print("=" * 60)
    print(" DEPLOY PRODUCTION — satu perintah")
    print(f" VPS: {VPS_USER}@{VPS_HOST}")
    print(f" Target lokal: {TARGET_REPO_DIR}")
    print("=" * 60)

    if args.code_only and args.models_only:
        print("[ERROR] --code-only dan --models-only tidak bisa bersamaan.")
        return 1

    # Salin Riset -> swint lokal (lewati jika hanya push kode VPS atau hanya scp model)
    if not args.code_only and not args.models_only:
        if not step_local_deploy(args.dry_run):
            print("\n[FAIL] deploy_model.py gagal.")
            return 1
        if args.local_only:
            print("\n[DONE] Local-only selesai.")
            return 0

    if not args.local_only:
        if not args.models_only:
            if not step_git_push(
                msg,
                code_paths,
                args.dry_run,
                all_tracked=args.code_only,
            ):
                print("\n[FAIL] git push gagal.")
                return 1
        if not args.code_only:
            if not step_scp_models(model_paths, args.dry_run):
                print("\n[FAIL] scp model gagal.")
                return 1
        if not step_vps_update(args.dry_run):
            print("\n[FAIL] VPS update gagal.")
            return 1

    print("\n" + "=" * 60)
    print(" DEPLOY SELESAI")
    print(f" Cek: http://{VPS_HOST}:5000/api/health")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())