"""
Compare ic32 swing complement v1 (18 feat) vs v2 (11 feat, production-aligned).

Usage:
  python pipeline/08c_ic32_complement_v1_v2_compare.py
  python pipeline/08c_ic32_complement_v1_v2_compare.py --skip-v2-eval  # only print v1 cached
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from config import MODEL_DIR

V1 = "ic32_lstm_swing_complement_v1"
V2 = "ic32_lstm_swing_complement_v2"


def _load_eval(run: str) -> dict | None:
    path = MODEL_DIR / "runs" / run / "oof_stack_eval.json"
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _load_meta(run: str) -> dict:
    for name in (f"{run}_meta.json", "meta.json"):
        path = MODEL_DIR / "runs" / run / name
        if path.exists():
            with open(path, encoding="utf-8") as f:
                return json.load(f)
    return {}


def _row(label: str, ev: dict, key: str) -> str:
    v = ev.get("variants", {}).get(key, {})
    if not v or v.get("total_trades", 0) == 0:
        return f"  {label:<10} {'—':>8} {'—':>7} {'—':>6} {'—':>10}"
    return (
        f"  {label:<10} {v['total_trades']:>8,} {v['win_rate']:>7.1f} "
        f"{v['profit_factor']:>6.2f} ${v['total_pnl']:>+9.2f}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-v2-eval", action="store_true")
    args = parser.parse_args()

    if not args.skip_v2_eval:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "eval08b", ROOT / "pipeline" / "08b_oof_ic32_swing_complement_eval.py"
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        print("\n>>> Running OOF eval for v2 (11 feat)...\n")
        mod.run_eval(V2)

    ev1 = _load_eval(V1)
    ev2 = _load_eval(V2)
    m1, m2 = _load_meta(V1), _load_meta(V2)

    sep = "=" * 78
    print(f"\n{sep}")
    print("  ic32 SWING COMPLEMENT — v1 (18f) vs v2 (11f production)")
    print(f"{sep}")

    print("\n  CV / Complement signal (OOF)")
    print(f"  {'Run':<8} {'F1 macro':>10} {'n_feat':>7} {'prec_dir':>10} {'n_fire':>8}")
    print(f"  {'-'*48}")
    for label, meta in (("v1", m1), ("v2", m2)):
        ca = meta.get("complement_asymmetric_oof", {})
        print(
            f"  {label:<8} {meta.get('mean_f1_macro', 0):>10.4f} "
            f"{meta.get('n_features', '?'):>7} "
            f"{ca.get('mixed_precision_dir', 0):>10.3f} "
            f"{ca.get('n_complement', 0):>8,}"
        )

    key = "complement_conditional_momentum"
    print(f"\n  OOF trading — {key} (genuine LSTM OOF path)")
    print(f"  {'Run':<10} {'Trades':>8} {'WR%':>7} {'PF':>6} {'PnL':>10}")
    print(f"  {'-'*44}")
    if ev1:
        print(_row("v1 (18f)", ev1, key))
    else:
        print("  v1 (18f)  — no oof_stack_eval.json")
    if ev2:
        print(_row("v2 (11f)", ev2, key))
    else:
        print("  v2 (11f)  — no oof_stack_eval.json")

    if ev1 and ev2:
        b = ev1["variants"].get("baseline_hard_consensus", {})
        v1m = ev1["variants"].get(key, {})
        v2m = ev2["variants"].get(key, {})
        if v1m and v2m:
            print(f"\n  Delta v2 - v1 ({key}):")
            print(f"    Trades: {v2m['total_trades'] - v1m['total_trades']:+,}")
            print(f"    WR:     {v2m['win_rate'] - v1m['win_rate']:+.1f}pp")
            print(f"    PF:     {v2m['profit_factor'] - v1m['profit_factor']:+.3f}")
            print(f"    PnL:    ${v2m['total_pnl'] - v1m['total_pnl']:+.2f}")
        if b:
            print(f"\n  Baseline (lstm_best 11f): {b['total_trades']:,} trades "
                  f"WR {b['win_rate']}% PF {b['profit_factor']}")

    out = {
        "v1": {"meta": m1, "eval": ev1},
        "v2": {"meta": m2, "eval": ev2},
    }
    out_path = MODEL_DIR / "runs" / V2 / "v1_v2_compare.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\n  Saved -> {out_path}")
    print(f"{sep}\n")


if __name__ == "__main__":
    main()