"""Smoke test: regime FLIP alignment matches backtest_utils logic."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from core.cascade_utils import (
    SHORT, LONG, FLAT,
    compute_regime_flip_delta,
    apply_flip_to_proba,
)

REGIME_CFG = {
    "enabled": True,
    "ranging": {"counter_trend_boost": 0.05, "with_trend_penalty": 0.1},
    "trending": {"counter_trend_penalty": 0.05, "with_trend_boost": 0.1},
}


def test_ranging_counter_trend_short():
    delta, label = compute_regime_flip_delta(SHORT, 1, 1.0, REGIME_CFG)
    assert abs(delta - 0.05) < 1e-9, delta
    assert "ranging_counter" in label


def test_trending_with_trend_long():
    delta, label = compute_regime_flip_delta(LONG, 3, 1.0, REGIME_CFG)
    assert abs(delta - 0.10) < 1e-9, delta
    assert "trending_with" in label


def test_flip_off():
    delta, label = compute_regime_flip_delta(LONG, 3, 1.0, {"enabled": False})
    assert delta == 0.0
    assert label == "flip_off"


def test_apply_renormalizes():
    proba = np.array([0.2, 0.3, 0.5], dtype=np.float32)
    out, delta, label = apply_flip_to_proba(proba, LONG, 3, 1.0, REGIME_CFG)
    assert abs(out.sum() - 1.0) < 1e-5
    assert out[LONG] > proba[LONG]
    assert abs(delta - 0.10) < 1e-9


if __name__ == "__main__":
    test_ranging_counter_trend_short()
    test_trending_with_trend_long()
    test_flip_off()
    test_apply_renormalizes()
    print("OK: all regime FLIP smoke tests passed")