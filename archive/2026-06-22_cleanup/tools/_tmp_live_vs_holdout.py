"""Audit: live feature_snapshot vs holdout parquet untuk bar yang sama.
Jalankan LGBM ic32_regime_v2 pada kedua set, bandingkan p0/p2 dan fitur."""
import sqlite3, json, sys, warnings, numpy as np, pandas as pd
from pathlib import Path
warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
import joblib
from core.utils import ensure_utc_index
from config import MODEL_DIR, HOLDOUT_DIR

lgbm = joblib.load(MODEL_DIR / "runs" / "ic32_regime_v2" / "lgbm.pkl")
feats = json.load(open(MODEL_DIR / "runs" / "ic32_regime_v2" / "features.json"))
THR_LONG, THR_SHORT = 0.75, 0.70

con = sqlite3.connect(str(ROOT / "data" / "live_cache" / "app.db"))

# Bar yang mau dicek: coin -> live signal_time (bar yang baru close)
# Backtest pakai bar index 01:00 UTC -> live signal di 01:05-01:07
CASES = {
    "BTCUSDT":  "2026-06-22 01:00",   # backtest SHORT, live FLAT
    "SOLUSDT":  "2026-06-22 01:00",   # backtest SHORT, live FLAT
    "BNBUSDT":  "2026-06-22 01:00",   # backtest SHORT, live FLAT
    "LINKUSDT": "2026-06-22 01:00",   # backtest SHORT, live FLAT
    "ETHUSDT":  "2026-06-22 01:00",   # backtest SHORT, live SHORT (match)
    "ARBUSDT":  "2026-06-22 01:00",   # backtest SHORT, live SHORT (match)
}

def get_live_snapshot(symbol, bar_hour):
    # cari signal dengan signal_time di jam bar_hour (signal_time = bar_hour + ~5-7 menit)
    q = """SELECT s.feature_snapshot, s.direction, s.confidence, s.signal_time, s.close_price
           FROM signal s JOIN coin c ON s.coin_id=c.id
           WHERE c.symbol=? AND s.signal_time >= ? AND s.signal_time < ?
           ORDER BY s.signal_time LIMIT 1"""
    start = pd.Timestamp(bar_hour) + pd.Timedelta(minutes=4)
    end   = pd.Timestamp(bar_hour) + pd.Timedelta(minutes=20)
    row = con.execute(q, (symbol, str(start), str(end))).fetchone()
    if not row or not row[0]:
        return None
    return {"snap": json.loads(row[0]), "dir": row[1], "conf": row[2],
            "time": row[3], "close": row[4]}

def predict(vec):
    p = lgbm.predict_proba(np.array(vec, dtype=np.float64).reshape(1, -1))[0]
    return p[0], p[2]  # p_short, p_long

print(f"{'coin':10s} {'src':8s} {'p_short':>8s} {'p_long':>8s} {'decision':>10s}")
print("=" * 50)

feat_diffs = {}
for coin, bar in CASES.items():
    live = get_live_snapshot(coin, bar)
    # holdout features
    hp = HOLDOUT_DIR / "labeled" / f"{coin}_features_v3.parquet"
    hdf = ensure_utc_index(pd.read_parquet(hp)).sort_index()
    ts = pd.Timestamp(bar, tz="UTC")
    if ts not in hdf.index:
        print(f"{coin}: bar {bar} tidak ada di holdout"); continue
    hrow = hdf.loc[ts]

    hvec = [float(hrow[f]) if f in hdf.columns and pd.notna(hrow[f]) else 0.0 for f in feats]
    ps_h, pl_h = predict(hvec)
    dec_h = "SHORT" if ps_h >= THR_SHORT else ("LONG" if pl_h >= THR_LONG else "FLAT")
    print(f"{coin:10s} {'holdout':8s} {ps_h:8.4f} {pl_h:8.4f} {dec_h:>10s}")

    if live:
        lvec = [float(live['snap'].get(f, 0.0)) for f in feats]
        ps_l, pl_l = predict(lvec)
        dec_l = "SHORT" if ps_l >= THR_SHORT else ("LONG" if pl_l >= THR_LONG else "FLAT")
        print(f"{coin:10s} {'live':8s} {ps_l:8.4f} {pl_l:8.4f} {dec_l:>10s}  (live actual: {live['dir']})")
        # simpan diff fitur
        diffs = []
        for f in feats:
            hv = float(hrow[f]) if f in hdf.columns and pd.notna(hrow[f]) else 0.0
            lv = float(live['snap'].get(f, 0.0))
            diffs.append((f, hv, lv, abs(hv - lv)))
        feat_diffs[coin] = sorted(diffs, key=lambda x: -x[3])
    else:
        print(f"{coin:10s} {'live':8s}  -- no live snapshot --")
    print("-" * 50)

# Tampilkan top fitur divergen untuk BTC (contoh paling jelas)
for coin in ["BTCUSDT", "SOLUSDT"]:
    if coin in feat_diffs:
        print(f"\n[{coin}] Top 12 fitur paling beda (holdout vs live):")
        print(f"  {'feature':28s} {'holdout':>14s} {'live':>14s} {'abs_diff':>12s}")
        for f, hv, lv, d in feat_diffs[coin][:12]:
            print(f"  {f:28s} {hv:14.6g} {lv:14.6g} {d:12.6g}")
con.close()
