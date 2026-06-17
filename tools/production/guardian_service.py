"""
app/services/guardian_service.py — Guardian dynamic exit (continuation_v1).

Multiclass LGBM: 0=HOLD, 1=PARTIAL_EXIT, 2=FULL_EXIT.
Supports tb_guardian_continuation_v1 feature set (29 feat):
  static market context + derived momentum + 10 dynamic (incl. entry-relative).
"""

import json
import logging
from datetime import timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent.parent

DYNAMIC_BASE = {
    "bars_held_norm",
    "current_pnl_pct",
    "current_pnl_atr",
    "max_favorable_pnl_pct",
    "drawdown_from_peak_pct",
    "direction",
    "entry_price_ratio",
    "cvd_slope_h4_delta_entry",
    "ofi_h4_delta_entry",
    "flow_momentum_3bar",
}

DERIVED_STATIC = {
    "cvd_slope_h4_delta",
    "ofi_h4_accel",
    "rsi_h4_slope",
    "flow_momentum_3bar",
}

FLOW_MOM_WINDOW = 3


class GuardianService:
    def __init__(self, config: dict):
        g = config.get("guardian", {})
        self._enabled = g.get("enabled", False)
        self._exit_threshold = g.get("exit_threshold", 0.60)
        self._partial_ratio = g.get("partial_exit_ratio", 0.5)
        self._min_hold_bars = g.get("min_hold_bars", 3)
        self._activation_atr = g.get("activation_atr", 1.0)
        self._max_hold_bars = config.get("inference", {}).get("max_hold_bars", 24)

        self._model = None
        self._scaler = None
        self._feature_cols = None
        self._static_cols = None

        if self._enabled:
            self._load(g)

    def _load(self, g: dict) -> None:
        models_dir = ROOT / "models"
        try:
            self._model = joblib.load(models_dir / g.get("model_file", "guardian_best.pkl"))
            self._scaler = joblib.load(models_dir / g.get("scaler_file", "guardian_scaler.pkl"))
            with open(models_dir / g.get("features_file", "guardian_feature_cols.json")) as f:
                self._feature_cols = json.load(f)
            self._static_cols = [c for c in self._feature_cols if c not in DYNAMIC_BASE]
            logger.info(
                f"[Guardian] Loaded: {len(self._feature_cols)} features "
                f"({len(self._static_cols)} static + "
                f"{len(self._feature_cols) - len(self._static_cols)} dynamic) "
                f"threshold={self._exit_threshold} min_hold={self._min_hold_bars}"
            )
        except Exception as e:
            logger.error(f"[Guardian] Load gagal: {e}", exc_info=True)
            self._enabled = False

    @property
    def enabled(self) -> bool:
        return self._enabled and self._model is not None

    def check_exit(
        self,
        trade,
        features_df,
        close_price: float,
        atr: float,
        momentum_mode: bool = False,
    ) -> str:
        if not self.enabled:
            return "hold"

        hold_bars = trade.hold_bars or 0
        entry = trade.entry_price or 0.0

        if not momentum_mode:
            if hold_bars < self._min_hold_bars:
                return "hold"
            if atr > 0:
                direction_sign = 1 if trade.direction == "LONG" else -1
                price_move = direction_sign * (close_price - entry)
                if abs(price_move) < self._activation_atr * atr:
                    return "hold"

        already_partial = bool(trade.partial_exit_done)

        try:
            row_scaled = self._build_feature_row(trade, features_df, close_price, atr)
            proba = self._model.predict_proba(
                pd.DataFrame(row_scaled, columns=self._feature_cols)
            )[0]
            pred = int(np.argmax(proba))

            mode_tag = "MOMENTUM" if momentum_mode else "EARLY"
            logger.debug(
                f"[Guardian/{mode_tag}] trade_id={trade.id} bars={hold_bars} "
                f"proba=[{proba[0]:.3f},{proba[1]:.3f},{proba[2]:.3f}] pred={pred}"
            )

            if pred == 2 and proba[2] >= self._exit_threshold:
                return "full_exit"
            if pred == 1 and proba[1] >= self._exit_threshold and not already_partial:
                return "partial_exit"
            return "hold"

        except Exception as e:
            logger.error(f"[Guardian] check_exit error trade_id={trade.id}: {e}", exc_info=True)
            return "hold"

    def _entry_bar_row(self, trade, features_df: pd.DataFrame | None) -> pd.Series | None:
        if features_df is None or features_df.empty or trade.opened_at is None:
            return None
        opened = trade.opened_at
        if opened.tzinfo is None:
            opened = opened.replace(tzinfo=timezone.utc)
        idx = features_df.index
        if getattr(idx, "tz", None) is None and opened.tzinfo is not None:
            sub = features_df[features_df.index <= opened.replace(tzinfo=None)]
        else:
            sub = features_df[idx <= opened]
        if sub.empty:
            return features_df.iloc[0]
        return sub.iloc[-1]

    def _compute_derived_static(self, features_df: pd.DataFrame | None) -> dict:
        out = {k: 0.0 for k in DERIVED_STATIC}
        if features_df is None or len(features_df) < 1:
            return out

        last = features_df.iloc[-1]
        if "cvd_slope_h4" in features_df.columns and len(features_df) >= 2:
            out["cvd_slope_h4_delta"] = float(
                last["cvd_slope_h4"] - features_df.iloc[-2]["cvd_slope_h4"]
            )
        if "ofi_h4_delta" in features_df.columns and len(features_df) >= 3:
            out["ofi_h4_accel"] = float(
                last["ofi_h4_delta"] - features_df.iloc[-3]["ofi_h4_delta"]
            )
        if "rsi_h4" in features_df.columns and len(features_df) >= 3:
            out["rsi_h4_slope"] = float(last["rsi_h4"] - features_df.iloc[-3]["rsi_h4"])
        if "ofi_z_score" in features_df.columns:
            out["flow_momentum_3bar"] = float(
                features_df["ofi_z_score"].tail(FLOW_MOM_WINDOW).mean()
            )
        return out

    def _build_feature_row(self, trade, features_df, close_price: float, atr: float) -> np.ndarray:
        entry = trade.entry_price or 1.0
        hold_bars = trade.hold_bars or 0
        is_long = trade.direction == "LONG"

        pnl_pct = (close_price - entry) / entry
        if not is_long:
            pnl_pct = -pnl_pct

        bars_held_norm = hold_bars / 24.0
        current_pnl_atr = pnl_pct * entry / atr if atr > 0 else 0.0

        if trade.max_favorable_price is None:
            trade.max_favorable_price = entry
        if is_long:
            if close_price > trade.max_favorable_price:
                trade.max_favorable_price = close_price
            mfe_pnl = (trade.max_favorable_price - entry) / entry
        else:
            if close_price < trade.max_favorable_price:
                trade.max_favorable_price = close_price
            mfe_pnl = (entry - trade.max_favorable_price) / entry
        mfe_pnl = max(mfe_pnl, 0.0)

        dd_from_peak = (mfe_pnl - pnl_pct) / mfe_pnl if mfe_pnl > 0.001 else 0.0
        entry_ratio = entry / close_price if close_price > 0 else 1.0
        direction_enc = 1.0 if is_long else 0.0

        entry_row = self._entry_bar_row(trade, features_df)
        cur_cvd = cur_ofi = 0.0
        ent_cvd = ent_ofi = 0.0
        if features_df is not None and len(features_df) > 0:
            last = features_df.iloc[-1]
            cur_cvd = float(last.get("cvd_slope_h4", 0) or 0)
            cur_ofi = float(last.get("ofi_h4_delta", 0) or 0)
        if entry_row is not None:
            ent_cvd = float(entry_row.get("cvd_slope_h4", 0) or 0)
            ent_ofi = float(entry_row.get("ofi_h4_delta", 0) or 0)

        derived = self._compute_derived_static(features_df)
        flow_mom = derived.get("flow_momentum_3bar", 0.0)

        dynamic = {
            "bars_held_norm": bars_held_norm,
            "current_pnl_pct": pnl_pct,
            "current_pnl_atr": current_pnl_atr,
            "max_favorable_pnl_pct": mfe_pnl,
            "drawdown_from_peak_pct": dd_from_peak,
            "direction": direction_enc,
            "entry_price_ratio": entry_ratio,
            "cvd_slope_h4_delta_entry": float(cur_cvd - ent_cvd),
            "ofi_h4_delta_entry": float(cur_ofi - ent_ofi),
            "flow_momentum_3bar": flow_mom,
        }

        static_vals = {}
        if features_df is not None and len(features_df) > 0:
            last = features_df.iloc[-1]
            for col in self._static_cols:
                if col in DERIVED_STATIC:
                    static_vals[col] = derived.get(col, 0.0)
                elif col in last.index and not pd.isna(last[col]):
                    static_vals[col] = float(last[col])
                else:
                    static_vals[col] = 0.0
        else:
            static_vals = {col: 0.0 for col in self._static_cols}

        combined = {**static_vals, **dynamic}
        row = np.array(
            [combined.get(col, 0.0) for col in self._feature_cols],
            dtype=np.float64,
        ).reshape(1, -1)
        return self._scaler.transform(row)