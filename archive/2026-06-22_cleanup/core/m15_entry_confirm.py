"""
M15 entry confirmation layer on top of frozen H1 ic32 signals.

Signal fires at H1 bar close; entry waits for M15 confirmation within max_wait bars.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

FLAT, SHORT, LONG = 1, 0, 2

M15_DIR = Path(__file__).parent.parent / "data" / "research" / "m15" / "klines"

CONFIRM_RULES = (
    "h1_immediate",
    "m15_delay1",
    "m15_delay2",
    "m15_align",
    "m15_pullback_02",
    "m15_no_adverse_03",
)


@dataclass
class M15EntryResult:
    signal_idx: int
    entry_idx: int
    entry_time: pd.Timestamp
    entry_price: float
    direction: int
    m15_bars_waited: int
    skipped: bool = False


def load_m15(
    symbol: str,
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
    *,
    holdout: bool = False,
) -> pd.DataFrame:
    """Load M15 OHLCV; holdout=True reads separate holdout parquet (amplop tersegel)."""
    paths = [M15_DIR / (f"{symbol}_15m_holdout.parquet" if holdout else f"{symbol}_15m.parquet")]
    parts = []
    for path in paths:
        if not path.exists():
            continue
        df = pd.read_parquet(path).sort_index()
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        parts.append(df)
    if not parts:
        return pd.DataFrame()
    out = pd.concat(parts)
    out = out[~out.index.duplicated(keep="last")].sort_index()
    if start is not None:
        out = out.loc[out.index >= start]
    if end is not None:
        out = out.loc[out.index <= end]
    return out


def _slice_m15_after(m15: pd.DataFrame, signal_time: pd.Timestamp, max_wait: int) -> pd.DataFrame:
    sub = m15.loc[m15.index >= signal_time]
    if sub.empty:
        return sub
    return sub.iloc[:max_wait]


def _h1_idx_for_m15(h1_index: pd.DatetimeIndex, signal_idx: int, m15_time: pd.Timestamp) -> int:
    """Map M15 entry timestamp to H1 bar index (>= signal_idx)."""
    pos = int(h1_index.searchsorted(m15_time, side="right")) - 1
    pos = max(pos, signal_idx)
    return min(pos, len(h1_index) - 1)


def _find_delay(m15_win: pd.DataFrame, delay: int, direction: int) -> tuple[pd.Timestamp, float] | None:
    if len(m15_win) <= delay:
        return None
    row = m15_win.iloc[delay]
    return row.name, float(row["close"])


def _find_align(m15_win: pd.DataFrame, direction: int) -> tuple[pd.Timestamp, float] | None:
    prev_close = None
    for ts, row in m15_win.iterrows():
        c = float(row["close"])
        if prev_close is not None:
            if direction == LONG and c > prev_close:
                return ts, c
            if direction == SHORT and c < prev_close:
                return ts, c
        prev_close = c
    return None


def _find_pullback(m15_win: pd.DataFrame, direction: int, ref_price: float, pct: float = 0.002) -> tuple[pd.Timestamp, float] | None:
    if m15_win.empty:
        return None
    seen_pullback = False
    highs: list[float] = []
    for ts, row in m15_win.iterrows():
        c = float(row["close"])
        lo = float(row["low"])
        hi = float(row["high"])
        highs.append(c)
        if direction == LONG:
            if lo < ref_price * (1.0 - pct):
                seen_pullback = True
            if seen_pullback and len(highs) >= 2 and c > max(highs[-3:-1]):
                return ts, c
        else:
            if hi > ref_price * (1.0 + pct):
                seen_pullback = True
            if seen_pullback and len(highs) >= 2 and c < min(highs[-3:-1]):
                return ts, c
    return None


def _find_no_adverse(m15_win: pd.DataFrame, direction: int, ref_price: float, adverse_pct: float = 0.003) -> tuple[pd.Timestamp, float] | None:
    if m15_win.empty:
        return None
    first = m15_win.iloc[0]
    if direction == LONG:
        adv = (ref_price - float(first["low"])) / ref_price
        if adv > adverse_pct:
            return _find_align(m15_win.iloc[1:], direction)
    else:
        adv = (float(first["high"]) - ref_price) / ref_price
        if adv > adverse_pct:
            return _find_align(m15_win.iloc[1:], direction)
    return first.name, float(first["close"])


def find_m15_entry(
    rule: str,
    signal_idx: int,
    signal_time: pd.Timestamp,
    direction: int,
    h1_close: float,
    m15: pd.DataFrame,
    h1_index: pd.DatetimeIndex,
    max_wait: int = 4,
) -> M15EntryResult | None:
    if direction == FLAT:
        return None

    if rule == "h1_immediate":
        return M15EntryResult(
            signal_idx=signal_idx,
            entry_idx=signal_idx,
            entry_time=signal_time,
            entry_price=float(h1_close),
            direction=direction,
            m15_bars_waited=0,
        )

    win = _slice_m15_after(m15, signal_time, max_wait)
    if win.empty:
        return None

    hit: tuple[pd.Timestamp, float] | None = None
    waited = 0

    if rule == "m15_delay1":
        hit = _find_delay(win, 1, direction)
        waited = 1 if hit else 0
    elif rule == "m15_delay2":
        hit = _find_delay(win, 2, direction)
        waited = 2 if hit else 0
    elif rule == "m15_align":
        hit = _find_align(win, direction)
        if hit:
            waited = int(win.index.get_loc(hit[0])) + 1
    elif rule == "m15_pullback_02":
        hit = _find_pullback(win, direction, h1_close, pct=0.002)
        if hit:
            waited = int(win.index.get_loc(hit[0])) + 1
    elif rule == "m15_no_adverse_03":
        hit = _find_no_adverse(win, direction, h1_close, adverse_pct=0.003)
        if hit:
            waited = int(win.index.get_loc(hit[0])) + 1
    else:
        raise ValueError(f"Unknown rule: {rule}")

    if hit is None:
        return None

    entry_time, entry_price = hit
    entry_idx = _h1_idx_for_m15(h1_index, signal_idx, entry_time)
    return M15EntryResult(
        signal_idx=signal_idx,
        entry_idx=entry_idx,
        entry_time=entry_time,
        entry_price=entry_price,
        direction=direction,
        m15_bars_waited=waited,
    )


def apply_m15_confirmation(
    h1_index: pd.DatetimeIndex,
    y_pred: np.ndarray,
    confidence: np.ndarray | None,
    h1_close: np.ndarray,
    m15: pd.DataFrame,
    rule: str,
    max_wait: int = 4,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, dict]:
    """
    Build H1 y_pred with M15-confirmed entries.

    Returns y_pred_new, entry_price_override, confidence_new, stats.
    """
    n = len(h1_index)
    y_new = np.full(n, FLAT, dtype=np.int64)
    price_ov = np.full(n, np.nan, dtype=np.float64)
    conf_new = confidence.copy() if confidence is not None else None
    if conf_new is not None:
        conf_new = np.zeros(n, dtype=np.float64)

    n_signal = 0
    n_confirmed = 0
    n_skipped = 0
    wait_list: list[int] = []

    for i in range(n):
        sig = int(y_pred[i])
        if sig == FLAT:
            continue
        n_signal += 1
        res = find_m15_entry(
            rule=rule,
            signal_idx=i,
            signal_time=h1_index[i],
            direction=sig,
            h1_close=float(h1_close[i]),
            m15=m15,
            h1_index=h1_index,
            max_wait=max_wait,
        )
        if res is None:
            n_skipped += 1
            continue
        j = res.entry_idx
        if y_new[j] != FLAT:
            n_skipped += 1
            continue
        y_new[j] = sig
        price_ov[j] = res.entry_price
        if conf_new is not None and confidence is not None:
            conf_new[j] = float(confidence[i])
        n_confirmed += 1
        wait_list.append(res.m15_bars_waited)

    stats = {
        "rule": rule,
        "n_signal": n_signal,
        "n_confirmed": n_confirmed,
        "n_skipped": n_skipped,
        "confirm_rate_pct": round(n_confirmed / n_signal * 100, 2) if n_signal else 0.0,
        "mean_m15_wait": round(float(np.mean(wait_list)), 2) if wait_list else 0.0,
    }
    return y_new, price_ov, conf_new, stats