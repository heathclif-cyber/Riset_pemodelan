"""
core/positioning_engine.py — Positioning Data Engine

Integrates Binance-available positioning data (OI, LS Position, LS Account)
into the cascade as confidence multipliers and risk adjustments.

ALL features use only data AVAILABLE LIVE from Binance public API:
  - Open Interest: /futures/data/openInterestHist
  - Top Trader L/S: /futures/data/topLongShortPositionRatio
  - Global L/S: /futures/data/globalLongShortAccountRatio

IC-validated findings (19 coins, Jan-Oct 2025, |IC| >= 0.03):
  ls_pos_extreme → Sharp moves      (IC=+0.120)
  ls_pos_z20     → Volatility 3D    (IC=+0.161)
  oi_d1          → Return 1D        (IC=-0.092, reversal)
  oi_z20         → Direction 7D     (IC=-0.081, reversal)

Usage:
  from core.positioning_engine import PositioningEngine

  engine = PositioningEngine()

  # Daily update (call once per day)
  engine.update(coin, oi_value, ls_position, ls_account)

  # Per-bar query (at cascade time)
  score = engine.get_score(coin)  # {positioning_score, extreme_flag, size_multiplier}
"""
import numpy as np
from datetime import datetime, timezone
from collections import defaultdict
from dataclasses import dataclass, field


@dataclass
class PositioningState:
    """Per-coin positioning state, updated daily."""
    oi_value: float = 0.0
    oi_prev_1d: float = 0.0
    oi_prev_7d: float = 0.0
    oi_sma_20: float = 0.0
    oi_std_20: float = 0.0
    ls_position: float = 1.0
    ls_position_prev_7d: float = 1.0
    ls_position_sma_20: float = 1.0
    ls_position_std_20: float = 0.1
    ls_account: float = 1.0
    last_update: str = ""

    @property
    def oi_d1(self):
        return (self.oi_value - self.oi_prev_1d) / max(abs(self.oi_prev_1d), 1e-8)

    @property
    def oi_d7(self):
        return (self.oi_value - self.oi_prev_7d) / max(abs(self.oi_prev_7d), 1e-8)

    @property
    def oi_z20(self):
        if self.oi_std_20 < 1e-8:
            return 0.0
        return (self.oi_value - self.oi_sma_20) / self.oi_std_20

    @property
    def ls_pos_z20(self):
        if self.ls_position_std_20 < 1e-8:
            return 0.0
        return (self.ls_position - self.ls_position_sma_20) / self.ls_position_std_20

    @property
    def ls_pos_d7(self):
        return self.ls_position - self.ls_position_prev_7d

    @property
    def ls_extreme(self):
        return abs(self.ls_pos_z20) > 2.0

    @property
    def smart_vs_retail(self):
        return self.ls_position - self.ls_account


class PositioningEngine:
    """
    Positioning-aware confidence + risk adjustment engine.

    Updated DAILY from Binance public API data.
    Queried PER-BAR during cascade entry decisions.

    Rules (IC-validated):
      1. LS extreme (>2 std) → reduce size 30% (turbulence ahead, IC=+0.16)
      2. OI oversold (z < -2) → boost confidence +0.03 (bounce likely, IC=-0.09)
      3. OI overbought (z > +2) → penalize -0.05 (reversal risk, IC=-0.08)
      4. Smart money accumulating (ls_pos_d7 > 0) in TRENDING → boost +0.05
    """

    def __init__(self):
        self._states: dict[str, PositioningState] = defaultdict(PositioningState)

    # ── Daily Update ──────────────────────────────────────────────────────
    def update(self, coin: str, oi_value: float, ls_position: float,
               ls_account: float, oi_prev_1d: float = None, oi_prev_7d: float = None,
               ls_position_prev_7d: float = None,
               oi_sma_20: float = None, oi_std_20: float = None,
               ls_position_sma_20: float = None, ls_position_std_20: float = None):
        """Update daily positioning state for one coin."""
        state = self._states[coin]
        state.oi_prev_1d = oi_prev_1d if oi_prev_1d is not None else state.oi_value
        state.oi_prev_7d = oi_prev_7d if oi_prev_7d is not None else state.oi_prev_7d
        state.oi_value = oi_value
        state.ls_position = ls_position
        state.ls_position_prev_7d = ls_position_prev_7d if ls_position_prev_7d is not None else state.ls_position_prev_7d
        state.ls_account = ls_account
        state.oi_sma_20 = oi_sma_20 if oi_sma_20 is not None else state.oi_value
        state.oi_std_20 = oi_std_20 if oi_std_20 is not None else state.oi_std_20
        state.ls_position_sma_20 = ls_position_sma_20 if ls_position_sma_20 is not None else state.ls_position
        state.ls_position_std_20 = ls_position_std_20 if ls_position_std_20 is not None else state.ls_position_std_20
        state.last_update = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    def update_from_dict(self, coin: str, daily_row: dict):
        """Update from a dictionary of daily values."""
        self.update(
            coin=coin,
            oi_value=float(daily_row.get("oi_value", 0)),
            ls_position=float(daily_row.get("ls_position", 1.0)),
            ls_account=float(daily_row.get("ls_account", 1.0)),
            oi_prev_1d=float(daily_row.get("oi_prev_1d", 0)) if "oi_prev_1d" in daily_row else None,
            oi_prev_7d=float(daily_row.get("oi_prev_7d", 0)) if "oi_prev_7d" in daily_row else None,
            ls_position_prev_7d=float(daily_row.get("ls_position_prev_7d", 1.0)) if "ls_position_prev_7d" in daily_row else None,
            oi_sma_20=float(daily_row.get("oi_sma_20", 0)) if "oi_sma_20" in daily_row else None,
            oi_std_20=float(daily_row.get("oi_std_20", 0)) if "oi_std_20" in daily_row else None,
            ls_position_sma_20=float(daily_row.get("ls_position_sma_20", 1.0)) if "ls_position_sma_20" in daily_row else None,
            ls_position_std_20=float(daily_row.get("ls_position_std_20", 0.1)) if "ls_position_std_20" in daily_row else None,
        )

    # ── Per-Bar Query (cascade entry time) ────────────────────────────────
    def get_adjustment(self, coin: str, lgbm_direction: int, hmm_regime: int,
                       base_confidence: float) -> dict:
        """
        Get positioning-based confidence + size adjustments.

        Args:
            coin: coin symbol (e.g. BTCUSDT)
            lgbm_direction: 2=LONG, 0=SHORT, 1=FLAT
            hmm_regime: 0=TREND_DN, 1=RANGE_LO, 2=RANGE_HI, 3=TREND_UP
            base_confidence: LGBM confidence before positioning adjustment

        Returns:
            dict with: confidence_delta, size_multiplier, extreme_flag, notes
        """
        state = self._states.get(coin)
        if state is None or state.last_update == "":
            return {"confidence_delta": 0.0, "size_multiplier": 1.0,
                    "extreme_flag": False, "notes": "no positioning data"}

        conf_delta = 0.0
        size_mult = 1.0
        extreme = False
        notes = []

        # Rule 1: LS extreme → volatility ahead → reduce size
        if state.ls_extreme:
            size_mult = 0.70
            extreme = True
            notes.append("LS extreme (-30% size)")

        # Rule 2: OI oversold (z < -2) → potential bounce → slight boost
        if state.oi_z20 < -2.0:
            conf_delta += 0.03
            notes.append(f"OI oversold (+0.03) z={state.oi_z20:.1f}")

        # Rule 3: OI overbought (z > +2) → reversal risk → penalize
        if state.oi_z20 > 2.0:
            conf_delta -= 0.05
            notes.append(f"OI overbought (-0.05) z={state.oi_z20:.1f}")

        # Rule 4: Smart money accumulating in trending → boost
        is_trending = hmm_regime in (0, 3)
        if is_trending and state.ls_pos_d7 > 0.05:
            conf_delta += 0.05
            notes.append(f"Smart money accumulating (+0.05) d7={state.ls_pos_d7:+.3f}")

        # Rule 5: Smart/retail divergence → caution
        if abs(state.smart_vs_retail) > 0.5:
            conf_delta -= 0.03
            notes.append(f"Smart/retail diverge (-0.03)")

        # Rule 6: OI delta 1D extreme negative → short-term bounce
        if state.oi_d1 < -0.05:
            if lgbm_direction == 2:  # LONG
                conf_delta += 0.03
                notes.append(f"OI spike down, bounce (+0.03)")

        # Rule 7: OI delta 1D extreme positive → short-term mean-reversion risk
        if state.oi_d1 > 0.05:
            conf_delta -= 0.03
            notes.append(f"OI spike up, reversal risk (-0.03)")

        return {
            "confidence_delta": round(conf_delta, 4),
            "size_multiplier": round(size_mult, 2),
            "extreme_flag": extreme,
            "notes": "; ".join(notes) if notes else "OK",
        }

    def has_data(self, coin: str) -> bool:
        return self._states.get(coin) is not None and self._states[coin].last_update != ""


# Global singleton
_positioning_engine: PositioningEngine | None = None


def get_positioning_engine() -> PositioningEngine:
    global _positioning_engine
    if _positioning_engine is None:
        _positioning_engine = PositioningEngine()
    return _positioning_engine
