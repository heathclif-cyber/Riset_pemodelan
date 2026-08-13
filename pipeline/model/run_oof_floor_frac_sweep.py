"""Sweep guardian_momentum_floor_tp_frac (STOP-LIMIT floor pasca-TP) -- dipicu insiden
FILUSDT 2026-08-12 (Guardian exit dini, disusul crash besar yg terlewat) & analisis
_scratch_guardian_missed_move.py (floor_frac=0.7 saat ini: 28% trade terlewat big-move
+12h, tapi 30% terhindar dari reversal -- hampir seimbang).

Uji apakah floor_frac lebih longgar (kasih ruang gerak lebih) mengurangi rate
"terlewat" tanpa mengorbankan proteksi "terhindar" terlalu banyak.

Semua varian pakai config live SEKARANG: base=0.65/delta=0.10, cooldown ON,
portfolio_limits ON (max_open_positions=10, daily_loss_limit=8) -- SSOT malam ini.

Usage:
  python pipeline/model/run_oof_floor_frac_sweep.py
  python pipeline/model/run_oof_floor_frac_sweep.py --floor-fracs 0.5,0.6,0.7,0.8,0.9
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
warnings.filterwarnings("ignore")

from config import (
    FEE_PER_SIDE, GUARDIAN_EXIT_THRESHOLD, LIVE_DAILY_LOSS_LIMIT, LIVE_MAX_OPEN_POSITIONS,
    MAX_HOLDING_BARS, MODAL_PER_TRADE,
    SLIPPAGE_PER_SIDE, SWING_LABEL_MAX_SL, SWING_LABEL_MIN_RR, SWING_LABEL_MIN_TP,
    TP_SL_FALLBACK_SL, TP_SL_FALLBACK_TP, TRAIN_CUTOFF_DATE,
)
from core.evaluator import apply_portfolio_execution_limits, simulate_trades_swing
from core.utils import ensure_utc_index, setup_logger
from model.eval.constants import LM, LEV
from model.eval.oof_full_stack import load_coin_data, _predict
from model.eval.scorecard import compute_scorecard
from model.regime.thresholds import build_regime_thresholds
from model.stacks.load import load_stack

logger = setup_logger("oof_floor_frac_sweep")

GUARDIAN_EARLY_EXIT_OUTCOMES = {"GUARDIAN_EXIT", "GUARDIAN_MOMENTUM_EXIT", "GUARDIAN_MOMENTUM_FLOOR"}
MISS_WINDOW_H = 12
BIG_MOVE_PCT = 0.03
LABEL_DIR = Path("data/training/labeled_opt2")

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


def _missed_move_rate(trades: list[dict]) -> tuple[float, float, int]:
    """Return (pct_terlewat, pct_terhindar, n_dianalisis) utk trade Guardian-early-exit."""
    sub = [t for t in trades if t.get("outcome") in GUARDIAN_EARLY_EXIT_OUTCOMES and t.get("exit_time") is not None]
    n_missed = n_saved = n_ok = 0
    for t in sub:
        px = _load_price(t["coin"])
        if px is None or px.empty:
            continue
        pos = px.index.searchsorted(t["exit_time"], side="right")
        if pos >= len(px.index):
            continue
        exit_close = float(px["close"].iloc[max(pos - 1, 0)])
        fut = px.iloc[pos: pos + MISS_WINDOW_H]
        if fut.empty:
            continue
        if t["direction"] == "LONG":
            fav = (fut["high"].max() - exit_close) / exit_close
            adv = (exit_close - fut["low"].min()) / exit_close
        else:
            fav = (exit_close - fut["low"].min()) / exit_close
            adv = (fut["high"].max() - exit_close) / exit_close
        n_ok += 1
        if fav >= BIG_MOVE_PCT:
            n_missed += 1
        if adv >= BIG_MOVE_PCT:
            n_saved += 1
    if n_ok == 0:
        return float("nan"), float("nan"), 0
    return n_missed / n_ok * 100, n_saved / n_ok * 100, n_ok


def _fmt(sc: dict | None) -> str:
    if not sc:
        return "NO TRADES"
    return (
        f"trades={sc.get('trades', 0):,} WR={sc['wr']:.1%} PF={sc['pf']:.3f} "
        f"PnL=${sc['pnl']:,.2f} MaxDD=${sc['max_dd']:.2f}"
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="Sweep guardian_momentum_floor_tp_frac")
    ap.add_argument("--stack", default="fs37_18coin_polos")
    ap.add_argument("--hmm-base", type=float, default=0.65)
    ap.add_argument("--hmm-delta", type=float, default=0.10)
    ap.add_argument("--floor-fracs", default="0.5,0.6,0.7,0.8,0.9")
    ap.add_argument("--max-open-positions", type=int, default=None)
    ap.add_argument("--daily-loss-limit", type=int, default=None)
    ap.add_argument("--export-trades", default=None,
                    help="Simpan trade ACCEPTED (post portfolio_limits, dgn mae_pct) ke CSV ini. "
                         "Cuma masuk akal kalau --floor-fracs berisi SATU nilai.")
    args = ap.parse_args()

    max_pos = args.max_open_positions if args.max_open_positions is not None else LIVE_MAX_OPEN_POSITIONS
    daily_lim = args.daily_loss_limit if args.daily_loss_limit is not None else LIVE_DAILY_LOSS_LIMIT

    stack = load_stack(args.stack)
    regime_thr = build_regime_thresholds(args.hmm_base, args.hmm_delta)
    print(f"Loading OOF data stack={stack.name} HMM={args.hmm_base}/{args.hmm_delta} "
          f"window={stack.hmm_vol_window}/{stack.hmm_mom_window} ...")
    coin_data, g_model, g_scaler, g_feats, static_names = load_coin_data(stack, regime_thr)
    print(f"Coins loaded: {len(coin_data)}\n")

    full_start = pd.Timestamp("2020-01-01", tz="UTC")
    end = pd.Timestamp(TRAIN_CUTOFF_DATE).tz_convert("UTC")

    def run_window(floor_frac: float | None) -> tuple[dict, list]:
        """floor_frac=None -> floor DIMATIKAN total (guardian_floor_replace_with_tp=False +
        guardian_momentum_floor_frac=0.0) -- setelah TP tersentuh, trade jalan terus TANPA
        jaring pengaman, murni bergantung sinyal exit Guardian lain (probability/momentum)
        atau TIMEOUT. Beda dari floor_frac kecil (masih ada floor, cuma longgar)."""
        floor_kwargs = (
            dict(guardian_momentum_floor_tp_frac=0.0, guardian_floor_replace_with_tp=False,
                 guardian_momentum_floor_frac=0.0)
            if floor_frac is None else
            dict(guardian_momentum_floor_tp_frac=floor_frac, guardian_floor_replace_with_tp=True)
        )
        all_trades = []
        for coin, df in coin_data.items():
            sub = df[(df.index >= full_start) & (df.index < end)]
            n = len(sub)
            if n < 10:
                continue
            p0 = sub["p0"].values.astype(np.float64)
            p2 = sub["p2"].values.astype(np.float64)
            reg = sub["hmm_regime_enc"].fillna(1).astype(int).values
            y_pred, lgbm_conf = _predict(p0, p2, reg, regime_thr)
            if (y_pred != LM["FLAT"]).sum() == 0:
                continue
            X_g = np.zeros((n, len(static_names)), dtype=np.float64)
            for k, col in enumerate(static_names):
                if col in sub.columns:
                    X_g[:, k] = sub[col].ffill().fillna(0).values.astype(np.float64)
            h4_sh = sub["h4_swing_high"].values if "h4_swing_high" in sub.columns else np.full(n, np.nan)
            h4_sl = sub["h4_swing_low"].values if "h4_swing_low" in sub.columns else np.full(n, np.nan)
            res = simulate_trades_swing(
                y_pred=y_pred, close=sub["close"].values, high=sub["high"].values, low=sub["low"].values,
                atr=sub["atr_14_h1"].values, h4_swing_highs=h4_sh, h4_swing_lows=h4_sl,
                modal=MODAL_PER_TRADE, leverage=LEV, fee_per_side=FEE_PER_SIDE, slippage=SLIPPAGE_PER_SIDE,
                max_hold=MAX_HOLDING_BARS, min_rr=SWING_LABEL_MIN_RR, min_tp_atr=SWING_LABEL_MIN_TP,
                max_sl_atr=SWING_LABEL_MAX_SL, tp_fallback_atr=TP_SL_FALLBACK_TP, sl_fallback_atr=TP_SL_FALLBACK_SL,
                guardian_enabled=True, guardian_model=g_model, guardian_scaler=g_scaler, X_guardian=X_g,
                guardian_feat_cols=g_feats, guardian_static_names=static_names,
                guardian_exit_threshold=GUARDIAN_EXIT_THRESHOLD,
                **floor_kwargs,
                lgbm_conf_arr=lgbm_conf,
                cooldown_enabled=True, cooldown_profit_only=True, cooldown_profit_bars=1,
            )
            logs = res.get("trade_log", res.get("trades", []))
            _hi, _lo, _cl = sub["high"].values, sub["low"].values, sub["close"].values
            for t in logs:
                bi, bo = t.get("bar_in"), t.get("bar_out")
                t["coin"] = coin
                t["entry_time"] = (sub.index[bi] + pd.Timedelta(hours=1)) if bi is not None and bi < n else None
                t["exit_time"] = (sub.index[min(bo, n - 1)] + pd.Timedelta(hours=1)) if bo is not None else None
                if bi is not None and bo is not None:
                    from tools.model.leverage_mae_aware import compute_mae_pct
                    t["mae_pct"] = round(compute_mae_pct(_hi, _lo, _cl, bi, bo, t.get("direction")), 6)
                else:
                    t["mae_pct"] = None
            all_trades.extend(logs)

        accepted, rejected = apply_portfolio_execution_limits(
            all_trades, max_open_positions=max_pos, daily_loss_limit=daily_lim,
        )
        sc = compute_scorecard(accepted)
        return sc, accepted

    print(f"{'='*100}\n  Sweep guardian_momentum_floor_tp_frac (base HMM {args.hmm_base}, portfolio_limits ON: "
          f"max_pos={max_pos} daily_loss={daily_lim})\n{'-'*100}")
    rows = []
    for raw in [x.strip() for x in args.floor_fracs.split(",") if x.strip()]:
        ff = None if raw.lower() == "off" else float(raw)
        label = "OFF (no floor)" if ff is None else f"{ff}"
        print(f"  running floor_frac={label} ...", flush=True)
        sc, trades = run_window(ff)
        pct_missed, pct_saved, n_analyzed = _missed_move_rate(trades)
        rows.append({"floor_frac": label, **sc, "pct_missed_12h": pct_missed,
                      "pct_saved_12h": pct_saved, "n_guardian_analyzed": n_analyzed})
        print(f"    {_fmt(sc)}  | +12h big-move: terlewat={pct_missed:.1f}% terhindar={pct_saved:.1f}% (n={n_analyzed})")
        if args.export_trades:
            cols = ["coin", "direction", "net_pnl", "outcome", "entry_time", "exit_time", "mae_pct"]
            pd.DataFrame(trades)[cols].to_csv(args.export_trades, index=False)
            print(f"    Exported {len(trades)} trade -> {args.export_trades}")

    print(f"\n{'='*100}\n  RINGKASAN\n{'-'*100}")
    print(f"  {'floor_frac':14s} {'trades':>8s} {'WR':>7s} {'PF':>7s} {'PnL':>10s} {'MaxDD':>9s} "
          f"{'LongPF':>7s} {'ShortPF':>7s} {'%terlewat':>10s} {'%terhindar':>11s}")
    for r in rows:
        print(f"  {r['floor_frac']:<14s} {r['trades']:>8,} {r['wr']:>6.1%} {r['pf']:>7.3f} "
              f"${r['pnl']:>9,.2f} ${r['max_dd']:>8,.2f} {r.get('long_pf', float('nan')):>7.3f} "
              f"{r.get('short_pf', float('nan')):>7.3f} {r['pct_missed_12h']:>9.1f}% {r['pct_saved_12h']:>10.1f}%")

    out_df = pd.DataFrame(rows)
    out_path = "data/live_cache/floor_frac_sweep_result.csv"
    out_df.to_csv(out_path, index=False)
    print(f"\nSaved -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
