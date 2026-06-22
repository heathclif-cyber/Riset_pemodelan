"""
core/position_sizing.py — Kelly Criterion + Regime-Based Sizing

Simons: "Position sizing is MORE important than entry timing."

1. Kelly Criterion: f* = (p*b - q) / b
   p = win rate, q = loss rate, b = avg_win/avg_loss

2. Regime-Based: TRENDING → 50% size, RANGING → 100%

3. Drawdown Protection: DD > 15% → 50% size, DD > 25% → STOP

Usage:
  from core.position_sizing import PositionSizer
  sizer = PositionSizer()
  multiplier = sizer.get_multiplier(hmm_regime, current_drawdown)
"""
import numpy as np


class PositionSizer:
    """
    Kelly-optimal + regime-aware position sizing.

    Kelly determines the FRACTION of capital to risk per trade.
    Regime adjusts based on HMM state.
    Drawdown protection reduces size during losing streaks.
    """

    def __init__(self, win_rate=0.51, avg_win=1.5, avg_loss=1.0,
                 base_risk_pct=0.02, max_kelly_mult=2.0):
        """
        Args:
            win_rate: historical win rate (OOF purged CV)
            avg_win: average winning trade amount (ratio to avg loss)
            avg_loss: average losing trade amount
            base_risk_pct: max % of capital to risk per trade (2% = conservative)
            max_kelly_mult: cap on Kelly multiplier (2x = max double size)
        """
        self.win_rate = win_rate
        self.avg_win = avg_win
        self.avg_loss = avg_loss
        self.base_risk_pct = base_risk_pct
        self.max_mult = max_kelly_mult

        # Compute Kelly fraction
        b = avg_win / avg_loss if avg_loss > 0 else 1.0
        p = win_rate; q = 1 - p
        self.kelly_f = max(0, (p * b - q) / b) if b > 0 else 0.01

        # Half-Kelly (more conservative, recommended)
        self.half_kelly = self.kelly_f / 2.0

        # Normalize to 1.0 baseline
        # If Kelly says 2.5% risk and we use 2%, multiplier = 1.0
        if self.half_kelly > 0:
            self.kelly_mult = min(self.half_kelly / base_risk_pct, max_kelly_mult)
        else:
            self.kelly_mult = 0.5  # fallback: reduce size

        # Regime multipliers (from OOF analysis: model loses in TRENDING)
        self.regime_mult = {
            # TRENDING_DOWN (0): model weak → reduce
            0: 0.50,
            # RANGING_LOW_VOL (1): model strong → full size
            1: 1.00,
            # RANGING_HIGH_VOL (2): slight reduction
            2: 0.80,
            # TRENDING_UP (3): model weak → reduce
            3: 0.50,
        }

        self.peak_capital = None
        self.current_dd = 0.0

    def update_drawdown(self, current_equity):
        """Update drawdown from peak."""
        if self.peak_capital is None or current_equity > self.peak_capital:
            self.peak_capital = current_equity
        if self.peak_capital > 0:
            self.current_dd = (self.peak_capital - current_equity) / self.peak_capital

    def get_multiplier(self, hmm_regime, position_count=1):
        """
        Get final size multiplier.

        Args:
            hmm_regime: 0=TREND_DN, 1=RANGE_LO, 2=RANGE_HI, 3=TREND_UP
            position_count: current open positions (for diversification)

        Returns:
            float: multiplier on base trade size (MODAL_PER_TRADE)
        """
        mult = 1.0

        # 1. Kelly adjustment
        mult *= self.kelly_mult

        # 2. Regime adjustment
        if hmm_regime is not None:
            mult *= self.regime_mult.get(int(hmm_regime), 1.0)

        # 3. Drawdown protection
        if self.current_dd > 0.25:
            mult = 0.0  # STOP trading
        elif self.current_dd > 0.15:
            mult *= 0.50  # Half size during drawdown

        # 4. Position diversification (optional penalty for too many positions)
        if position_count > 15:
            mult *= 0.80

        return round(mult, 2)

    def __repr__(self):
        return (f"PositionSizer(Kelly={self.kelly_f:.3f}, Half={self.half_kelly:.3f}, "
                f"Mult={self.kelly_mult:.2f}x, DD={self.current_dd:.1%})")


# ─── Pre-computed values from OOF extended backtest ──────────────────────
# From purged CV 2020-2026: 29,369 trades, WR=51.4%
# Avg winning trade: ~$0.75, Avg losing trade: ~$0.60
# b = 0.75/0.60 = 1.25
# Kelly f = (0.514 * 1.25 - 0.486) / 1.25 = 0.125
# Half-Kelly = 0.0625
# Base risk = 2% → mult = 0.0625/0.02 = 3.125 → capped at 2.0

DEFAULT_SIZER = PositionSizer(
    win_rate=0.51,     # OOF purged CV (conservative)
    avg_win=1.09,      # avg_win/avg_loss from OOF
    avg_loss=1.0,
    base_risk_pct=0.02,
    max_kelly_mult=1.0,  # CAP at 1.0 — don't amplify a system with no edge
)
# Result: Kelly mult = ~1.0x (no amplification)
# Regime mult: TRENDING = 0.50x, RANGING = 1.00x
# Net: TRENDING → 0.50x, RANGING → 1.00x
