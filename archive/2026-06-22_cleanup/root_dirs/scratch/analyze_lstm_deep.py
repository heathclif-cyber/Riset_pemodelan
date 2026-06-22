import json
from pathlib import Path

p = Path(__file__).parent.parent / "models/runs/tb_lgbm_flatboost_v2/lstm_deep_tune.json"
d = json.loads(p.read_text())
rows = [r for r in d["all_results"] if r.get("lstm_contributes") and r["lstm_mode"] != "lgbm_only"]
strict = [r for r in rows if 0 <= r["trade_drop_pct"] <= 20]
strict.sort(key=lambda x: (-x["pnl_delta"], -x["pnl"]))
print("=== STRICT (0<=drop<=20%, pnl_delta>0) ===", len(strict))
for r in strict[:25]:
    print(
        f"{r['hmm']} | {r['lstm_variant']} | {r['lstm_tag']} | "
        f"tr={r['trades']} dPnL={r['pnl_delta']:+.0f} drop={r['trade_drop_pct']:.1f}% "
        f"ppt={r['pnl_per_trade']:.3f}"
    )

prod = [r for r in d["all_results"] if r["hmm"] == "T50_R55_s5" and r["lstm_mode"] != "lgbm_only"]
prod.sort(key=lambda x: x["pnl_delta"], reverse=True)
print("\n=== T50_R55_s5 top 15 by dPnL ===")
for r in prod[:15]:
    print(
        f"{r['lstm_tag']} | dPnL={r['pnl_delta']:+.0f} drop={r['trade_drop_pct']:.1f}% "
        f"PnL={r['pnl']:+.0f} ppt={r['pnl_per_trade']:.3f}"
    )

wa = [r for r in d["all_results"] if r["lstm_variant"] == "widyawardhana" and r["pnl_delta"] > 0]
wa.sort(key=lambda x: x["pnl_delta"], reverse=True)
print("\n=== widyawardhana positive dPnL ===", len(wa))
for r in wa[:10]:
    print(f"{r['hmm']} {r['lstm_tag']} dPnL={r['pnl_delta']:+.0f} drop={r['trade_drop_pct']:.1f}%")