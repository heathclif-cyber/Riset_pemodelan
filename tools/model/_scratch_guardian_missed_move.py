"""Skrip sekali-pakai: seberapa sering Guardian exit disusul big-move yang terlewat
(vs seberapa sering Guardian exit menyelamatkan dari giveback). Dipicu insiden nyata
FILUSDT 2026-08-12 (Guardian exit 20:10 UTC, disusul crash besar 21:15 UTC).

Sumber: OOF varian E (portfolio-limits, base=0.65) -- 6 tahun data, jauh lebih
representatif drpd sampel live yang masih tipis.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from core.utils import ensure_utc_index

LABEL_DIR = Path("data/training/labeled_opt2")
TRADES_CSV = Path("data/live_cache/oof_hmmbase065_E_trades.csv")
GUARDIAN_EARLY_EXIT_OUTCOMES = {"GUARDIAN_EXIT", "GUARDIAN_MOMENTUM_EXIT", "GUARDIAN_MOMENTUM_FLOOR"}
WINDOWS_H = [6, 12, 24, 48]
BIG_MOVE_PCT = 0.03  # 3% pergerakan harga tambahan = "big move" (FILUSDT: ~4.5% tambahan)

_price_cache: dict[str, pd.DataFrame] = {}


def _load_price(coin: str) -> pd.DataFrame | None:
    if coin in _price_cache:
        return _price_cache[coin]
    fp = LABEL_DIR / f"{coin}_features_v3.parquet"
    if not fp.exists():
        _price_cache[coin] = None
        return None
    df = pd.read_parquet(fp, columns=["close", "high", "low"])
    df = ensure_utc_index(df).sort_index()
    _price_cache[coin] = df
    return df


def main() -> int:
    trades = pd.read_csv(TRADES_CSV)
    trades["exit_time"] = pd.to_datetime(trades["exit_time"], utc=True, errors="coerce")
    trades = trades[trades["outcome"].isin(GUARDIAN_EARLY_EXIT_OUTCOMES)].copy()
    trades = trades.dropna(subset=["exit_time"])
    print(f"Total trade Guardian-early-exit (GUARDIAN_EXIT/MOMENTUM_EXIT/MOMENTUM_FLOOR): {len(trades)}")

    rows = []
    for t in trades.itertuples(index=False):
        px = _load_price(t.coin)
        if px is None or px.empty:
            continue
        idx = px.index
        pos = idx.searchsorted(t.exit_time, side="right")
        if pos >= len(idx):
            continue
        # Harga exit riil: pakai close bar TERAKHIR sebelum/pada exit_time sbg proxy
        # (kolom 'exit' harga presisi tidak diekspor di CSV OOF varian E -- pakai
        # close bar exit_time sbg proksi, cukup akurat utk analisis arah pergerakan).
        exit_close = float(px["close"].iloc[max(pos - 1, 0)])

        row = {"coin": t.coin, "direction": t.direction, "outcome": t.outcome,
               "net_pnl": t.net_pnl, "exit_time": t.exit_time}
        for w in WINDOWS_H:
            fut = px.iloc[pos: pos + w]
            if fut.empty:
                row[f"fav_move_pct_{w}h"] = np.nan
                row[f"adv_move_pct_{w}h"] = np.nan
                continue
            if t.direction == "LONG":
                fav = (fut["high"].max() - exit_close) / exit_close
                adv = (exit_close - fut["low"].min()) / exit_close
            else:
                fav = (exit_close - fut["low"].min()) / exit_close
                adv = (fut["high"].max() - exit_close) / exit_close
            row[f"fav_move_pct_{w}h"] = fav * 100
            row[f"adv_move_pct_{w}h"] = adv * 100
        rows.append(row)

    res = pd.DataFrame(rows)
    print(f"Trade dgn data harga follow-up: {len(res)}\n")

    print(f"{'='*78}\n  Ringkasan per jendela follow-up (threshold big-move = {BIG_MOVE_PCT*100:.0f}%)\n{'-'*78}")
    for w in WINDOWS_H:
        favcol, advcol = f"fav_move_pct_{w}h", f"adv_move_pct_{w}h"
        sub = res.dropna(subset=[favcol, advcol])
        n = len(sub)
        if n == 0:
            continue
        n_big_missed = (sub[favcol] >= BIG_MOVE_PCT * 100).sum()
        n_big_saved = (sub[advcol] >= BIG_MOVE_PCT * 100).sum()
        print(f"  +{w:>2}h  | big-move TERLEWAT: {n_big_missed:>5}/{n} ({n_big_missed/n*100:5.1f}%)"
              f"  | big-move TERHINDAR (Guardian selamatkan): {n_big_saved:>5}/{n} ({n_big_saved/n*100:5.1f}%)"
              f"  | median pergerakan lanjut favorable: {sub[favcol].median():+.2f}%"
              f"  adverse: {sub[advcol].median():+.2f}%")

    print(f"\n{'='*78}\n  Breakdown per outcome (jendela +12h)\n{'-'*78}")
    for oc in sorted(GUARDIAN_EARLY_EXIT_OUTCOMES):
        sub = res[res["outcome"] == oc].dropna(subset=["fav_move_pct_12h", "adv_move_pct_12h"])
        n = len(sub)
        if n == 0:
            continue
        n_big_missed = (sub["fav_move_pct_12h"] >= BIG_MOVE_PCT * 100).sum()
        print(f"  {oc:28s} n={n:>5}  big-move terlewat: {n_big_missed:>4} ({n_big_missed/n*100:5.1f}%)"
              f"  median fav+12h={sub['fav_move_pct_12h'].median():+.2f}%")

    print(f"\n{'='*78}\n  Perbandingan LONG vs SHORT (jendela +12h)\n{'-'*78}")
    for d in ("LONG", "SHORT"):
        sub = res[res["direction"] == d].dropna(subset=["fav_move_pct_12h", "adv_move_pct_12h"])
        n = len(sub)
        if n == 0:
            continue
        n_big_missed = (sub["fav_move_pct_12h"] >= BIG_MOVE_PCT * 100).sum()
        n_big_saved = (sub["adv_move_pct_12h"] >= BIG_MOVE_PCT * 100).sum()
        print(f"  {d:6s} n={n:>5}  big-move terlewat: {n_big_missed:>4} ({n_big_missed/n*100:5.1f}%)"
              f"  big-move terhindar: {n_big_saved:>4} ({n_big_saved/n*100:5.1f}%)")

    out_path = "data/live_cache/guardian_missed_move_analysis.csv"
    res.to_csv(out_path, index=False)
    print(f"\nDetail lengkap -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
