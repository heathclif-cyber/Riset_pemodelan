"""
tools/backfill_binance_daily_ls.py — Backfill historical daily L/S dari Binance Futures API.

Fetch dua endpoint per koin (period=1d, paginasi backward):
  /futures/data/topLongShortPositionRatio  -> position_ls  (position-weighted, per-koin)
  /futures/data/topLongShortAccountRatio   -> account_ls   (account-count-weighted, per-koin)

Output: data/positioning/{coin}_ls_daily.parquet
  kolom: [position_ls, position_long_pct, position_short_pct,
          account_ls,  account_long_pct,  account_short_pct]
  index: timestamp (UTC, daily)

Binance limit: 500 bar per request. Script paginasi mundur sampai
  data habis atau mencapai START_DATE (2020-01-01).

Usage:
  python tools/backfill_binance_daily_ls.py              # semua koin
  python tools/backfill_binance_daily_ls.py BTCUSDT      # koin spesifik
  python tools/backfill_binance_daily_ls.py --append     # hanya fetch data baru (inkremensial)
"""
from __future__ import annotations

import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

from pipeline._bootstrap import setup_path_from_file
ROOT = setup_path_from_file(__file__)
from config import ALL_COINS

POSITIONING_DIR = ROOT / "data" / "positioning"
POSITIONING_DIR.mkdir(parents=True, exist_ok=True)

BINANCE_BASE = "https://fapi.binance.com"
LIMIT        = 500    # max per request
SLEEP        = 0.25   # detik antar request
START_DATE   = datetime(2020, 1, 1, tzinfo=timezone.utc)


# ─── Fetch satu chunk ────────────────────────────────────────────────────────

def _fetch_chunk(endpoint: str, symbol: str, end_ts_ms: int) -> pd.DataFrame | None:
    params = {
        "symbol":    symbol,
        "period":    "1d",
        "limit":     LIMIT,
        "endTime":   end_ts_ms,
    }
    try:
        r = requests.get(
            f"{BINANCE_BASE}{endpoint}",
            params=params, timeout=30, verify=False,
        )
        if r.status_code != 200:
            print(f"    HTTP {r.status_code}: {r.text[:120]}")
            return None
        data = r.json()
        if not isinstance(data, list) or len(data) == 0:
            return None
        df = pd.DataFrame(data)
        df["timestamp"] = pd.to_datetime(
            df["timestamp"].astype(np.int64), unit="ms", utc=True
        )
        return df.set_index("timestamp").sort_index()
    except Exception as e:
        print(f"    ERROR: {e}")
        return None


# ─── Fetch seluruh sejarah satu endpoint ─────────────────────────────────────

def _fetch_all(endpoint: str, symbol: str, start: datetime) -> pd.DataFrame | None:
    """Paginasi mundur dari sekarang sampai start."""
    end_ts = int(datetime.now(timezone.utc).timestamp() * 1000)
    start_ms = int(start.timestamp() * 1000)
    all_chunks: list[pd.DataFrame] = []

    while True:
        chunk = _fetch_chunk(endpoint, symbol, end_ts)
        if chunk is None or chunk.empty:
            break

        all_chunks.append(chunk)
        earliest_ts = chunk.index.min()

        if earliest_ts.timestamp() * 1000 <= start_ms:
            break

        # Geser end_ts ke sebelum data yang sudah diambil
        end_ts = int(earliest_ts.timestamp() * 1000) - 1
        time.sleep(SLEEP)

        if end_ts < start_ms:
            break

    if not all_chunks:
        return None

    df = pd.concat(all_chunks)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df = df[df.index >= pd.Timestamp(start)]
    return df


# ─── Fetch satu koin ─────────────────────────────────────────────────────────

def backfill_coin(symbol: str, append_only: bool = False) -> bool:
    out_path = POSITIONING_DIR / f"{symbol}_ls_daily.parquet"

    # Tentukan start date berdasarkan mode
    if append_only and out_path.exists():
        existing = pd.read_parquet(out_path)
        last_ts = existing.index.max()
        start = (last_ts + pd.Timedelta(days=1)).to_pydatetime()
        print(f"  {symbol}: append mode, last={last_ts.date()}, fetch from {start.date()}")
    else:
        start = START_DATE
        print(f"  {symbol}: full backfill from {start.date()}")

    # Fetch position L/S
    print(f"    Fetching topLongShortPositionRatio...")
    df_pos = _fetch_all("/futures/data/topLongShortPositionRatio", symbol, start)
    time.sleep(SLEEP)

    # Fetch account L/S
    print(f"    Fetching topLongShortAccountRatio...")
    df_acc = _fetch_all("/futures/data/topLongShortAccountRatio", symbol, start)
    time.sleep(SLEEP)

    if df_pos is None and df_acc is None:
        print(f"  {symbol}: tidak ada data — mungkin belum listing atau endpoint tidak support.")
        return False

    # Rename dan gabung
    rows: list[pd.DataFrame] = []

    if df_pos is not None:
        pos = df_pos.rename(columns={
            "longShortRatio": "position_ls",
            "longAccount":    "position_long_pct",
            "shortAccount":   "position_short_pct",
        })[["position_ls", "position_long_pct", "position_short_pct"]]
        pos = pos.apply(pd.to_numeric, errors="coerce")
        rows.append(pos)
    else:
        print(f"    WARNING: position L/S tidak tersedia untuk {symbol}")

    if df_acc is not None:
        acc = df_acc.rename(columns={
            "longShortRatio": "account_ls",
            "longAccount":    "account_long_pct",
            "shortAccount":   "account_short_pct",
        })[["account_ls", "account_long_pct", "account_short_pct"]]
        acc = acc.apply(pd.to_numeric, errors="coerce")
        rows.append(acc)
    else:
        print(f"    WARNING: account L/S tidak tersedia untuk {symbol}")

    if not rows:
        return False

    if len(rows) == 2:
        new_df = rows[0].join(rows[1], how="outer")
    else:
        new_df = rows[0]

    # Gabung dengan existing jika ada
    if out_path.exists() and not append_only:
        existing = pd.read_parquet(out_path)
        new_df = pd.concat([existing, new_df])
        new_df = new_df[~new_df.index.duplicated(keep="last")].sort_index()
    elif out_path.exists() and append_only:
        existing = pd.read_parquet(out_path)
        new_df = pd.concat([existing, new_df])
        new_df = new_df[~new_df.index.duplicated(keep="last")].sort_index()

    new_df.to_parquet(out_path)

    pos_col = "position_ls" if "position_ls" in new_df.columns else "-"
    acc_col  = "account_ls"  if "account_ls"  in new_df.columns else "-"
    pos_avail = new_df["position_ls"].notna().sum() if pos_col != "-" else 0
    acc_avail  = new_df["account_ls"].notna().sum()  if acc_col  != "-" else 0

    print(
        f"  {symbol}: {len(new_df)} hari total | "
        f"position_ls={pos_avail} bars | account_ls={acc_avail} bars | "
        f"{new_df.index.min().date()} -> {new_df.index.max().date()}"
    )
    return True


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    append = "--append" in sys.argv
    coins_arg = [a for a in sys.argv[1:] if not a.startswith("--")]
    coins = coins_arg if coins_arg else ALL_COINS

    mode = "APPEND (inkremensial)" if append else "FULL BACKFILL"
    print(f"\n{'='*60}")
    print(f"  Backfill Binance Daily L/S — {mode}")
    print(f"  Koin: {len(coins)} | Start: {START_DATE.date()}")
    print(f"  Output: {POSITIONING_DIR}")
    print(f"{'='*60}\n")

    ok, fail = 0, 0
    for i, coin in enumerate(coins, 1):
        print(f"[{i}/{len(coins)}]")
        try:
            if backfill_coin(coin, append_only=append):
                ok += 1
            else:
                fail += 1
        except Exception as e:
            print(f"  {coin} ERROR: {e}")
            fail += 1
        print()

    print(f"\n{'='*60}")
    print(f"  Selesai: {ok} OK, {fail} fail dari {len(coins)} koin")
    print(f"  Gunakan: python pipeline/11_train_lstm_daily_v2.py")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
