"""Skrip glue sekali-pakai: funding overlay -> MAE-aware leverage overlay, replikasi
persis rantai yang menghasilkan angka SSOT (inference_config.json.scorecard.oof/
holdout_oos) sesi 2026-08-11/12. Dipakai malam ini utk basis TAMBAHAN
portfolio_limits (max_open_positions/daily_loss_limit) -- bukan skrip permanen,
boleh dihapus setelah dipakai.

Usage:
  python tools/model/_scratch_final_scorecard.py --trades <csv> --funding-dir <dir> --leverage 15 --label OOS
"""
import argparse
import json
from pathlib import Path

import pandas as pd

from tools.model.funding_cost_overlay import apply_funding_overlay
from tools.model.leverage_mae_aware import evaluate


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trades", required=True)
    ap.add_argument("--funding-dir", required=True)
    ap.add_argument("--leverage", type=float, default=15.0)
    ap.add_argument("--source-leverage", type=float, default=None)
    ap.add_argument("--label", default="")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    src_lev = args.source_leverage
    if src_lev is None:
        from model.eval.constants import LEV as _LEV
        src_lev = float(_LEV)

    trades = pd.read_csv(args.trades)
    missing = {"direction", "net_pnl", "mae_pct", "entry_time", "exit_time", "coin"} - set(trades.columns)
    if missing:
        raise SystemExit(f"kolom hilang: {sorted(missing)}")

    funded = apply_funding_overlay(trades, Path(args.funding_dir), args.leverage)
    print(f"[{args.label}] funding cost total: ${funded['funding_cost'].sum():+.2f} "
          f"({(funded['n_funding_settlements'] > 0).mean()*100:.1f}% trade kena settlement)")

    funded["net_pnl"] = funded["net_pnl_with_funding"]
    funded["net_pnl_10x"] = funded["net_pnl"] * (10.0 / src_lev)

    res = evaluate(funded, [args.leverage])
    key = f"{int(args.leverage)}x"
    sc = res[key]["scorecard"]
    print(f"[{args.label}] FINAL @ {args.leverage:g}x (MAE-aware + funding): "
          f"trades={sc['trades']} WR={sc['wr']:.1%} PF={sc['pf']:.3f} PnL=${sc['pnl']:.2f} "
          f"MaxDD=${sc['max_dd']:.2f} LongPF={sc['long_pf']:.3f} ShortPF={sc['short_pf']:.3f} "
          f"n_liquidated={res[key]['n_liquidated']}")
    if "monthly" in res[key]:
        print(f"[{args.label}] monthly:")
        for m in res[key]["monthly"]:
            print(f"    {m}")

    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=2), encoding="utf-8")
        print(f"Saved -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
