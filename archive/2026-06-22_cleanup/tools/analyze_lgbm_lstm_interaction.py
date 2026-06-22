"""
Forensics: LGBM vs LSTM interaction on OOF trades from fusion stage2.

Analyzes agree/disagree/neutral LSTM states vs trade outcome.

Usage:
  python tools/analyze_lgbm_lstm_interaction.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from pipeline.ic32_fusion_shared import IC32_DIR, SHORT, FLAT, LONG

STAGE2_OUT = IC32_DIR / "ic32_lstm_fusion_stage2_pipeline.json"
LSTM_PATH = IC32_DIR / "oof_lstm_baseline_predictions.parquet"
OOF_PATH = IC32_DIR / "oof_predictions.parquet"
OUT_JSON = IC32_DIR / "lgbm_lstm_interaction_forensics.json"
OUT_MD = ROOT / "reports" / "experiments" / "lgbm_lstm_interaction_forensics.md"


def _lstm_relation(lgbm_dir: int, lstm_dir: int) -> str:
    if lstm_dir == lgbm_dir:
        return "agree"
    if lstm_dir == FLAT:
        return "neutral"
    return "opposite"


def main():
    if not STAGE2_OUT.exists():
        raise FileNotFoundError(f"Run stage2 first: {STAGE2_OUT}")

    with open(STAGE2_OUT, encoding="utf-8") as f:
        stage2 = json.load(f)

    lstm_oof = pd.read_parquet(LSTM_PATH).reset_index()
    if "index" in lstm_oof.columns:
        lstm_oof = lstm_oof.rename(columns={"index": "ts"})
    lgbm_oof = pd.read_parquet(OOF_PATH)
    if not isinstance(lgbm_oof.index, pd.DatetimeIndex):
        lgbm_oof.index = pd.to_datetime(lgbm_oof.index, utc=True)

    baseline_label = "baseline_production"
    trade_rows = []

    for sym in lgbm_oof["coin"].unique():
        sym_lgbm = lgbm_oof[lgbm_oof["coin"] == sym]
        sym_lstm = lstm_oof[lstm_oof["coin"] == sym]
        if sym_lstm.empty:
            continue
        for _, row in sym_lgbm.iterrows():
            ts = row.name if isinstance(row.name, pd.Timestamp) else row.get("ts")
            if not row.get("has_oof", True):
                continue
            p0, p2 = float(row["p0"]), float(row["p2"])
            lgbm_dir = LONG if p2 >= 0.69 else (SHORT if p0 >= 0.59 else FLAT)
            lstm_row = sym_lstm[sym_lstm["ts"] == ts] if "ts" in sym_lstm.columns else sym_lstm.loc[[ts]]
            if lstm_row.empty:
                continue
            lr = lstm_row.iloc[0]
            lstm_p = np.array([lr["p0"], lr["p1"], lr["p2"]], dtype=np.float32)
            lstm_dir = int(np.argmax(lstm_p))
            trade_rows.append({
                "coin": sym,
                "ts": str(ts),
                "lgbm_dir": lgbm_dir,
                "lstm_dir": lstm_dir,
                "relation": _lstm_relation(lgbm_dir, lstm_dir),
                "lgbm_p2": p2,
                "lgbm_p0": p0,
                "lstm_dom": float(lstm_p[lstm_dir]),
            })

    panel = pd.DataFrame(trade_rows)
    if panel.empty:
        raise RuntimeError("No interaction rows built")

    by_rel = panel.groupby("relation").agg(
        n=("relation", "count"),
        pct=("relation", lambda x: len(x) / len(panel) * 100),
        avg_lgbm_p2=("lgbm_p2", "mean"),
        avg_lstm_dom=("lstm_dom", "mean"),
    ).round(4)

    dir_panel = panel[panel["lgbm_dir"] != FLAT]
    by_rel_dir = dir_panel.groupby("relation").size().to_dict()

    forensics = {
        "source": str(STAGE2_OUT),
        "baseline_ppt": stage2.get("baseline", {}).get("ppt"),
        "best_candidate": stage2.get("best", {}).get("label"),
        "n_bars_analyzed": len(panel),
        "n_directional_bars": len(dir_panel),
        "relation_counts": panel["relation"].value_counts().to_dict(),
        "relation_pct": (panel["relation"].value_counts(normalize=True) * 100).round(2).to_dict(),
        "directional_relation_counts": by_rel_dir,
        "by_relation_stats": by_rel.to_dict(orient="index"),
        "stage2_decision": stage2.get("decision"),
        "winners": [w.get("label") for w in stage2.get("winners", [])],
    }

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(forensics, f, indent=2, default=str)

    lines = [
        "# LGBM+LSTM Interaction Forensics (OOF)",
        "",
        f"- Bars analyzed: {forensics['n_bars_analyzed']:,}",
        f"- Directional LGBM bars: {forensics['n_directional_bars']:,}",
        f"- Stage2 decision: {forensics['stage2_decision']}",
        f"- Best candidate: {forensics.get('best_candidate')}",
        "",
        "## LSTM relation to LGBM direction",
        "",
        "| Relation | Count | % |",
        "|----------|------:|--:|",
    ]
    for rel, cnt in forensics["relation_counts"].items():
        pct = forensics["relation_pct"].get(rel, 0)
        lines.append(f"| {rel} | {cnt:,} | {pct:.1f}% |")

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved: {OUT_JSON}")
    print(f"Saved: {OUT_MD}")
    print(json.dumps(forensics["relation_pct"], indent=2))


if __name__ == "__main__":
    main()