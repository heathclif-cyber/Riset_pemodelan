import json
from pathlib import Path

p = Path(__file__).parent.parent / "models/runs/tb_lgbm_flatboost_v2/lstm_deep_tune.json"
d = json.loads(p.read_text())
rows = d["all_results"]

for hmm in ["T48_R55_s5", "T50_R55_s5"]:
    sub = [r for r in rows if r["hmm"] == hmm and r["lstm_mode"] == "soft_pen_hmm"]
    sub.sort(key=lambda x: x["pnl_delta"], reverse=True)
    print(f"\n=== {hmm} soft_pen_hmm ===")
    for r in sub:
        print(f"  {r['lstm_tag']} dPnL={r['pnl_delta']:+.2f} drop={r['trade_drop_pct']:.1f}% ppt_d={r['ppt_delta']:+.4f}")

# prod_soft_veto best per HMM
for hmm in ["T48_R55_s5", "T50_R55_s5"]:
    sub = [r for r in rows if r["hmm"] == hmm and r["lstm_mode"] == "prod_soft_veto"]
    sub.sort(key=lambda x: x["pnl_delta"], reverse=True)
    print(f"\n=== {hmm} prod_soft_veto top 5 ===")
    for r in sub[:5]:
        print(f"  {r['lstm_tag']} dPnL={r['pnl_delta']:+.2f} drop={r['trade_drop_pct']:.1f}%")

# addon on T50 with moderate add
sub = [r for r in rows if r["hmm"] == "T50_R55_s5" and r["lstm_mode"] == "unlock_addon"
       and -20 <= r["trade_drop_pct"] <= 0]
sub.sort(key=lambda x: x["pnl_delta"], reverse=True)
print("\n=== T50 unlock_addon (adds <=20% trades) ===")
for r in sub[:8]:
    print(f"  {r['lstm_tag']} dPnL={r['pnl_delta']:+.0f} drop={r['trade_drop_pct']:.1f}% tr={r['trades']} ppt={r['pnl_per_trade']:.3f}")