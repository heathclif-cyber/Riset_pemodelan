"""Cek detail sinyal ic32_regime_v2 dari live cache DB."""
import sqlite3, json
from pathlib import Path

DB = Path("data/live_cache/app.db")
con = sqlite3.connect(DB)
con.row_factory = sqlite3.Row

rows = con.execute("""
    SELECT s.id, s.direction, s.confidence, s.signal_time,
           s.feature_snapshot, s.entry_reason, c.symbol
    FROM signal s
    JOIN model_meta mm ON s.model_meta_id = mm.id
    JOIN coin c ON s.coin_id = c.id
    WHERE mm.model_type = 'ic32_regime_v2'
    ORDER BY s.signal_time DESC
    LIMIT 10
""").fetchall()

print("=== ic32_regime_v2 SINYAL TERBARU ===")
print(f"Total: {len(rows)} sinyal\n")

for r in rows:
    fs = json.loads(r["feature_snapshot"]) if r["feature_snapshot"] else {}
    proba = fs.get("_lgbm_proba")
    thr_l = fs.get("_thr_lgbm_long")
    thr_s = fs.get("_thr_lgbm_short")
    dec   = fs.get("_lgbm_decision", "N/A")
    hmm   = fs.get("hmm_regime_enc")
    stage = fs.get("_cascade_stage", "N/A")

    proba_str = f"S={proba[0]:.3f} F={proba[1]:.3f} L={proba[2]:.3f}" if proba and len(proba)==3 else str(proba)

    print(f"[{r['symbol']:16}] {r['signal_time']} | dir={r['direction']:5}")
    print(f"  proba: {proba_str}")
    print(f"  thr: LONG={thr_l} SHORT={thr_s}")
    print(f"  decision: {dec}  |  stage: {stage}  |  HMM: {hmm}")
    if r["entry_reason"]:
        reason = r["entry_reason"][:120].encode("ascii", "replace").decode()
        print(f"  reason: {reason}")
    print()

# Cek: apakah ada sinyal v2 yang LONG/SHORT tapi di-filter?
total_all = con.execute("""
    SELECT COUNT(*) FROM signal s
    JOIN model_meta mm ON s.model_meta_id = mm.id
    WHERE mm.model_type = 'ic32_regime_v2'
""").fetchone()[0]

total_action = con.execute("""
    SELECT COUNT(*) FROM signal s
    JOIN model_meta mm ON s.model_meta_id = mm.id
    WHERE mm.model_type = 'ic32_regime_v2' AND s.direction != 'FLAT'
""").fetchone()[0]

print(f"Total v2 signals: {total_all}")
print(f"Actionable (non-FLAT): {total_action}")
print(f"FLAT rate: {(total_all - total_action) / max(total_all,1) * 100:.1f}%")

# Waktu pertama & terakhir sinyal v2
bounds = con.execute("""
    SELECT MIN(s.signal_time), MAX(s.signal_time)
    FROM signal s
    JOIN model_meta mm ON s.model_meta_id = mm.id
    WHERE mm.model_type = 'ic32_regime_v2'
""").fetchone()
print(f"Rentang: {bounds[0]} -> {bounds[1]}")

con.close()
