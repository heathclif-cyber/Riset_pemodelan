"""Wait for Vision backfill, then run clean -> engineer -> macro -> LSTM v2."""
from __future__ import annotations

import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from pipeline._bootstrap import setup_path_from_file

ROOT = setup_path_from_file(__file__)
from config import TRAINING_COINS

HIST_DIR = ROOT / "data" / "positioning_hist"
TERMINAL_LOGS = [ROOT / "terminals" / "6.txt", ROOT / "terminals" / "5.txt"]
LOG_FILE = ROOT / "reports" / "post_vision_pipeline.log"
POLL_SEC = 60
# Backfill batch mulai ~14:18 UTC (22:18 WITA)
BACKFILL_AFTER = datetime(2026, 6, 27, 14, 0, tzinfo=timezone.utc)


def log(msg: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"{ts} | post_vision | {msg}"
    print(line, flush=True)
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def fresh_coins() -> list[str]:
    done = []
    for coin in TRAINING_COINS:
        p = HIST_DIR / f"{coin}_metrics.parquet"
        if not p.exists():
            continue
        mtime = datetime.fromtimestamp(p.stat().st_mtime, tz=timezone.utc)
        if mtime >= BACKFILL_AFTER:
            done.append(coin)
    return done


def metrics_ok(coin: str) -> bool:
    """File harus fresh (post-backfill) dan punya data daily."""
    import pandas as pd

    p = HIST_DIR / f"{coin}_metrics.parquet"
    if not p.exists():
        return False
    mtime = datetime.fromtimestamp(p.stat().st_mtime, tz=timezone.utc)
    if mtime < BACKFILL_AFTER:
        return False
    try:
        df = pd.read_parquet(p, columns=["oi_base"])
        return len(df) >= 100 and df["oi_base"].notna().sum() >= 50
    except Exception:
        return False


def backfill_done() -> bool:
    """Hanya True jika 21/21 koin punya metrics fresh + valid. SUMMARY saja tidak cukup."""
    ok = [c for c in TRAINING_COINS if metrics_ok(c)]
    missing = [c for c in TRAINING_COINS if c not in ok]
    if missing:
        return False
    return len(ok) == len(TRAINING_COINS)


def wait_backfill() -> None:
    log(f"Waiting for Vision backfill ({len(TRAINING_COINS)} coins)...")
    while True:
        try:
            done = fresh_coins()
            ok = [c for c in TRAINING_COINS if metrics_ok(c)]
            missing = [c for c in TRAINING_COINS if c not in ok]
            log(
                f"Progress: {len(ok)}/{len(TRAINING_COINS)} valid"
                f" | fresh_mtime={len(done)}"
                f" | missing={missing[:5]}{'...' if len(missing) > 5 else ''}"
            )
            if backfill_done():
                log("Backfill complete.")
                return
        except Exception as e:
            log(f"Poll error (retry): {e}")
        time.sleep(POLL_SEC)


def main() -> int:
    import os

    py = sys.executable
    venv_py = ROOT / ".venv" / "Scripts" / "python.exe"
    if venv_py.exists():
        py = str(venv_py)
    os.environ.setdefault("LSTM_DEVICE", "dml")
    wait_backfill()

    steps = [
        ([py, "pipeline/02_clean.py", "--all"], "clean"),
        ([py, "pipeline/03_engineer.py", "--all"], "engineer"),
        ([py, "pipeline/data/core/fetch_positioning.py", "--macro-only"], "macro"),
        ([py, "pipeline/model/experiments/train_lstm_daily_v2.py"], "lstm_v2"),
    ]
    env = {**os.environ, "PYTHONPATH": str(ROOT), "LSTM_DEVICE": os.environ.get("LSTM_DEVICE", "dml")}
    log(f"LSTM will use LSTM_DEVICE={env['LSTM_DEVICE']}")
    for cmd, name in steps:
        log(f"START {name}: {' '.join(cmd)}")
        proc = subprocess.run(cmd, cwd=str(ROOT), env=env)
        if proc.returncode != 0:
            raise SystemExit(f"{name} failed (exit {proc.returncode})")
        log(f"DONE {name}")

    log("Pipeline selesai.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())