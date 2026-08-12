"""
OOF live-parity check — apakah scorecard.oof yang dilaporkan bisa direplikasi live?

Live sejak 12/13 Juli pakai 2 mekanisme exit yang TIDAK pernah dioper ke
simulate_trades_swing() di run_oof_scorecard.py (model/eval/oof_full_stack.py):
  1. Guardian floor = FIXED 0.7 x TP_pnl (STOP-LIMIT exchange-side begitu TP
     tersentuh) -- bukan trailing 0.7 x MFE (default GUARDIAN_MOMENTUM_FLOOR_FRAC).
  2. Cooldown profit-only 1 jam setelah net_pnl>0 close (coin+arah sama).

Script ini re-run OOF (predictions sudah genuine-OOF dari oof_predictions.parquet,
cuma re-simulasi layer eksekusi) utk 4 varian, di stack LIVE SEKARANG
(fs37_18coin_polos), window sama dgn scorecard.oof (2020-01-01 s.d. TRAIN_CUTOFF).

Varian D (true live parity) trade-level di-export dgn mae_pct utk overlay MAE-aware
@15x (tools/model/leverage_mae_aware.py) -- match metodologi model_registry.json.

Usage:
  python pipeline/model/run_oof_live_parity_check.py
  python pipeline/model/run_oof_live_parity_check.py --stack fs37_18coin_polos
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from datetime import datetime, timezone
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
from core.utils import setup_logger
from model.eval.constants import LM, LEV
from model.eval.oof_full_stack import load_coin_data, _predict
from model.eval.scorecard import compute_scorecard
from model.regime.thresholds import build_regime_thresholds
from model.stacks.load import load_stack
from pipeline.model.run_oof_cooldown_pyramiding import run_window as run_window_cd

logger = setup_logger("oof_live_parity")


def _fmt(sc: dict | None) -> str:
    if not sc:
        return "NO TRADES"
    return (
        f"trades={sc.get('trades', 0):,} WR={sc['wr']:.1%} PF={sc['pf']:.3f} "
        f"PnL=${sc['pnl']:,.2f} MaxDD=${sc['max_dd']:.2f} "
        f"LongPF={sc.get('long_pf', float('nan')):.3f} ShortPF={sc.get('short_pf', float('nan')):.3f}"
    )


def _delta(a: dict | None, b: dict | None, key: str) -> str:
    if not a or not b or key not in a or key not in b:
        return "n/a"
    va, vb = a[key], b[key]
    if not isinstance(va, (int, float)) or not isinstance(vb, (int, float)):
        return "n/a"
    d = va - vb
    if key == "wr":
        return f"{d:+.1%}"
    pct = (d / vb * 100) if vb else 0.0
    return f"{d:+.3f} ({pct:+.1f}%)" if key in ("pf",) else f"{d:+.2f} ({pct:+.1f}%)"


def main() -> int:
    ap = argparse.ArgumentParser(description="OOF live-parity check (floor+cooldown)")
    ap.add_argument("--stack", default="fs37_18coin_polos")
    ap.add_argument("--export-d-trades", default=None, help="CSV path to export variant D trades (with mae_pct)")
    ap.add_argument("--hmm-base", type=float, default=None, help="Override base threshold HMM (default: stack.hmm_base)")
    ap.add_argument("--hmm-delta", type=float, default=None, help="Override delta threshold HMM (default: stack.hmm_delta)")
    ap.add_argument("--portfolio-limits", action="store_true",
                    help="Tambah varian E = D + max_open_positions & daily_loss_limit PERSIS live "
                         "(execution.py::place_entry) -- D sendiri masih mengasumsikan modal tanpa "
                         "batas & tanpa circuit-breaker rugi harian.")
    ap.add_argument("--max-open-positions", type=int, default=None,
                    help="Override cap posisi konkuren (default: config.LIVE_MAX_OPEN_POSITIONS=10).")
    ap.add_argument("--daily-loss-limit", type=int, default=None,
                    help="Override circuit-breaker rugi harian WITA (default: config.LIVE_DAILY_LOSS_LIMIT=8).")
    args = ap.parse_args()

    stack = load_stack(args.stack)
    eff_base = args.hmm_base if args.hmm_base is not None else stack.hmm_base
    eff_delta = args.hmm_delta if args.hmm_delta is not None else stack.hmm_delta
    regime_thr = build_regime_thresholds(eff_base, eff_delta)
    print(f"Loading OOF data stack={stack.name} lgbm={stack.lgbm_run} guard={stack.guard_run} "
          f"HMM={eff_base}/{eff_delta}"
          f"{' (override dari stack default ' + str(stack.hmm_base) + '/' + str(stack.hmm_delta) + ')' if args.hmm_base is not None or args.hmm_delta is not None else ''} "
          f"window={stack.hmm_vol_window}/{stack.hmm_mom_window} ...")
    coin_data, g_model, g_scaler, g_feats, static_names = load_coin_data(stack, regime_thr)
    print(f"Coins loaded: {len(coin_data)}")

    full_start = pd.Timestamp("2020-01-01", tz="UTC")
    end = pd.Timestamp(TRAIN_CUTOFF_DATE).tz_convert("UTC")

    base_kw = dict(
        coin_data=coin_data, g_model=g_model, g_scaler=g_scaler, g_feats=g_feats,
        static_names=static_names, regime_thr=regime_thr, guardian=True,
        start=full_start, end=end,
    )

    variants = {
        "A_current_scorecard (trailing floor, no cooldown)": dict(
            cooldown_enabled=False, cooldown_profit_only=False,
        ),
        "B_plus_cooldown_live (trailing floor, +cooldown)": dict(
            cooldown_enabled=True, cooldown_profit_only=True, cooldown_profit_bars=1,
        ),
    }

    results = {}
    for vname, vkw in variants.items():
        print(f"  running {vname} ...", flush=True)
        sc = run_window_cd(**base_kw, **vkw)
        results[vname] = sc
        print(f"  {vname:48s}  {_fmt(sc)}")

    def run_window_floor(*, cooldown_enabled: bool, cooldown_profit_only: bool,
                          cooldown_profit_bars: int = 1, collect_trades: list | None = None):
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
                guardian_momentum_floor_tp_frac=0.7, guardian_floor_replace_with_tp=True,
                lgbm_conf_arr=lgbm_conf,
                cooldown_enabled=cooldown_enabled, cooldown_profit_only=cooldown_profit_only,
                cooldown_profit_bars=cooldown_profit_bars,
            )
            logs = res.get("trade_log", res.get("trades", []))
            for t in logs:
                t["coin"] = coin
            all_trades.extend(logs)
            if collect_trades is not None:
                from tools.model.leverage_mae_aware import compute_mae_pct
                _hi, _lo, _cl = sub["high"].values, sub["low"].values, sub["close"].values
                for t in logs:
                    bi, bo = t.get("bar_in"), t.get("bar_out")
                    # +1h: sub.index adalah waktu OPEN bar H1, keputusan baru dieksekusi
                    # SETELAH bar itu CLOSE (konvensi sama dgn holdout_oos.py) -- wajib
                    # supaya urutan kronologis & bucket hari WITA di apply_portfolio_
                    # execution_limits() konsisten dgn semantik live.
                    collect_trades.append({
                        "coin": coin, "direction": t.get("direction"), "net_pnl": t.get("net_pnl"),
                        "outcome": t.get("outcome"),
                        "entry_time": (sub.index[bi] + pd.Timedelta(hours=1)) if bi is not None and bi < n else None,
                        "exit_time": (sub.index[min(bo, n - 1)] + pd.Timedelta(hours=1)) if bo is not None else None,
                        "mae_pct": (round(compute_mae_pct(_hi, _lo, _cl, bi, bo, t.get("direction")), 6)
                                    if bi is not None and bo is not None else None),
                    })
        return compute_scorecard(all_trades)

    d_trades: list = []
    for vname, vkw in {
        "C_floor_live_only (FIXED 0.7xTP, no cooldown)": dict(cooldown_enabled=False, cooldown_profit_only=False),
        "D_TRUE_live_parity (FIXED floor + cooldown)": dict(
            cooldown_enabled=True, cooldown_profit_only=True, cooldown_profit_bars=1,
        ),
    }.items():
        print(f"  running {vname} ...", flush=True)
        collect = d_trades if vname.startswith("D_") else None
        sc = run_window_floor(**vkw, collect_trades=collect)
        results[vname] = sc
        print(f"  {vname:48s}  {_fmt(sc)}")

    e_max_pos = args.max_open_positions if args.max_open_positions is not None else LIVE_MAX_OPEN_POSITIONS
    e_daily_lim = args.daily_loss_limit if args.daily_loss_limit is not None else LIVE_DAILY_LOSS_LIMIT
    e_name = "E_plus_portfolio_limits (D + max_pos/daily_loss live)"
    e_rejected: list = []
    if args.portfolio_limits:
        print(f"  running {e_name} ...", flush=True)
        e_accepted, e_rejected = apply_portfolio_execution_limits(
            d_trades, max_open_positions=e_max_pos, daily_loss_limit=e_daily_lim,
        )
        sc_e = compute_scorecard(e_accepted)
        results[e_name] = sc_e
        n_max_pos = sum(1 for t in e_rejected if t.get("_reject_reason") == "max_open_positions")
        n_daily = sum(1 for t in e_rejected if t.get("_reject_reason") == "daily_loss_limit")
        print(f"  {e_name:48s}  {_fmt(sc_e)}")
        print(f"    -> {len(e_rejected)} kandidat trade D ditolak (max_open_positions={n_max_pos}, daily_loss_limit={n_daily})")

    base = results["A_current_scorecard (trailing floor, no cooldown)"]
    print(f"\n{'='*78}\n  vs A_current_scorecard (angka yang SEKARANG di model_registry.json)\n{'-'*78}")
    _delta_vnames = list(variants.keys())[1:] + ["C_floor_live_only (FIXED 0.7xTP, no cooldown)", "D_TRUE_live_parity (FIXED floor + cooldown)"]
    if args.portfolio_limits:
        _delta_vnames.append(e_name)
    for vname in _delta_vnames:
        sc = results[vname]
        print(
            f"  {vname:48s}  dtrades={_delta(sc, base, 'trades')}  dWR={_delta(sc, base, 'wr')}  "
            f"dPF={_delta(sc, base, 'pf')}  dPnL={_delta(sc, base, 'pnl')}  dMaxDD={_delta(sc, base, 'max_dd')}"
        )

    export_trades = e_accepted if args.portfolio_limits else d_trades
    export_tag = "variant-E (D + portfolio limits)" if args.portfolio_limits else "variant-D"
    if args.export_d_trades and export_trades:
        pd.DataFrame(export_trades).to_csv(args.export_d_trades, index=False)
        print(f"\nExported {len(export_trades)} {export_tag} trades -> {args.export_d_trades}")

    _hmm_tag = f"_hmmbase{eff_base}" if args.hmm_base is not None else ""
    out = stack.guard_run_dir / f"oof_live_parity_check{_hmm_tag}.json"
    payload = {
        "created": datetime.now(timezone.utc).isoformat(),
        "stack": stack.name, "lgbm_run": stack.lgbm_run, "guard_run": stack.guard_run,
        "hmm": f"{eff_base}/{eff_delta}", "window": f"{full_start.date()} .. {end.date()}",
        "note": (
            "A=persis run_oof_scorecard.py saat ini (sanity check vs model_registry.json.oof_scorecard). "
            "B=+cooldown profit-only 1h live. C=floor FIXED 0.7xTP live (STOP-LIMIT), no cooldown. "
            "D=TRUE live parity (floor FIXED + cooldown) -- exit mechanic PERSIS live sekarang. "
            "E=D + max_open_positions & daily_loss_limit PERSIS live (execution.py::place_entry) -- "
            "backtest A-D semua diam-diam asumsi modal tanpa batas & tanpa circuit-breaker rugi harian."
        ),
        "portfolio_limits": {
            "enabled": args.portfolio_limits,
            "max_open_positions": e_max_pos,
            "daily_loss_limit": e_daily_lim,
            "n_rejected": len(e_rejected),
        },
        "results": results,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"\nSaved -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
