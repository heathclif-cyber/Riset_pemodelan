# -*- coding: utf-8 -*-
"""Hapus LIVE_* risk override dari .env VPS — source of truth = inference_config.json."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

VPS = "root@139.180.157.176"
SCRIPT = Path(__file__).parent / "clean_vps_env_remote.py"
REMOTE = "/tmp/clean_vps_env.py"


def main() -> int:
    r = subprocess.run(
        ["scp", "-o", "BatchMode=yes", str(SCRIPT), f"{VPS}:{REMOTE}"],
        capture_output=True, text=True, timeout=40,
    )
    if r.returncode != 0:
        print(r.stderr or r.stdout)
        return 1
    r2 = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", VPS,
         f"python3 {REMOTE} ; systemctl restart swint-trade ; systemctl is-active swint-trade"],
        capture_output=True, text=True, timeout=60,
    )
    print(r2.stdout or r2.stderr)
    return 0 if r2.returncode == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())