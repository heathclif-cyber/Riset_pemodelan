"""
tools/audit_feature_parity.py — Harness parity fitur Riset <-> Live.

Pertanyaan inti: apakah nilai fitur yang dilihat model saat LIVE (dihitung dari
window pendek ~3000 bar yang di-fetch Binance) IDENTIK dengan nilai saat TRAINING
(dihitung dari histori penuh ~50k bar)? Jika tidak, arah/sinyal live menyimpang.

Metode: untuk tiap koin, jalankan engineer_features() pada
  (1) histori PENUH                -> proxy "training"
  (2) hanya N_LIVE bar terakhir     -> proxy "live fetch"
lalu bandingkan K baris terakhir (yang overlap) untuk SETIAP fitur model
(union LGBM + Guardian + LSTM). Fitur yang backward-looking window pendek -> diff ~0.
Fitur yang bergantung histori panjang (mis. cvd cumsum lama) -> diff besar = bocor parity.

Read-only. Tidak menyentuh model/training.

Usage:
  python tools/audit_feature_parity.py                 # default coins
  python tools/audit_feature_parity.py --coins BTCUSDT ETHUSDT --n-live 3000 --k 300
"""
import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from config import (
    PROC_DIR, MODEL_DIR,
    SWING_LABEL_MAX_HOLD, SWING_LABEL_MIN_RR, SWING_LABEL_MIN_TP, SWING_LABEL_MAX_SL,
    VP_WINDOW, VP_BINS, SWING_LOOKBACK, FVG_MIN_GAP_ATR, SYMBOL_MAP,
)
from core.features import engineer_features
from core.utils import ensure_utc_index

LONG_MAX_PRICE_IN_RANGE = 1.0
SHORT_MIN_PRICE_IN_RANGE = 0.0

# N_LIVE: jumlah bar yang di-fetch live (swint N_BARS=3000). K: bar terakhir yg dibandingkan.
DEFAULT_N_LIVE = 3000
DEFAULT_K = 300


def _run_engineer(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    return engineer_features(
        df=df, symbol=symbol, symbol_id=SYMBOL_MAP.get(symbol, -1),
        max_hold=SWING_LABEL_MAX_HOLD, min_rr=SWING_LABEL_MIN_RR,
        min_tp_atr=SWING_LABEL_MIN_TP, max_sl_atr=SWING_LABEL_MAX_SL,
        vp_window=VP_WINDOW, vp_bins=VP_BINS, swing_lookback=SWING_LOOKBACK,
        fvg_min_gap=FVG_MIN_GAP_ATR,
        long_max_price_in_range=LONG_MAX_PRICE_IN_RANGE,
        short_min_price_in_range=SHORT_MIN_PRICE_IN_RANGE,
        add_label=False,
    )


def _load_model_features() -> list[str]:
    feats: set[str] = set()
    for fp in [
        MODEL_DIR / "runs" / "ic32_regime_v2" / "features.json",
        MODEL_DIR / "runs" / "ic32_rv2_guard_oof" / "guardian_features.json",
        MODEL_DIR / "runs" / "ic32_rv2_lstm_momentum" / "lstm_v4_selected_features.json",
    ]:
        if fp.exists():
            feats.update(json.load(open(fp)))
    # Guardian punya fitur posisi runtime (current_pnl_pct dll) yg bukan dari engineer_features.
    # Hanya bandingkan yg memang diproduksi engineer_features (disaring saat compare).
    return sorted(feats)


def audit_coin(symbol: str, n_live: int, k: int, model_feats: list[str]) -> pd.DataFrame:
    in_path = PROC_DIR / f"{symbol}_clean.parquet"
    if not in_path.exists():
        print(f"  [{symbol}] clean parquet tidak ada — skip")
        return pd.DataFrame()

    df = ensure_utc_index(pd.read_parquet(in_path)).sort_index()
    if len(df) < n_live + k:
        print(f"  [{symbol}] histori terlalu pendek ({len(df)}) — skip")
        return pd.DataFrame()

    full_feat = _run_engineer(df, symbol)                 # training proxy
    live_feat = _run_engineer(df.iloc[-n_live:], symbol)  # live fetch proxy

    # Overlap K bar terakhir (live hanya punya last n_live; ambil last k yg pasti ada di dua-duanya)
    idx = live_feat.index[-k:]
    cols = [c for c in model_feats if c in full_feat.columns and c in live_feat.columns]

    rows = []
    for c in cols:
        a = pd.to_numeric(full_feat.loc[idx, c], errors="coerce").astype(float)
        b = pd.to_numeric(live_feat.loc[idx, c], errors="coerce").astype(float)
        d = (a - b).abs()
        denom = a.abs().clip(lower=1e-9)
        rel = (d / denom)
        rows.append({
            "feature": c,
            "max_abs_diff": float(d.max()),
            "mean_abs_diff": float(d.mean()),
            "max_rel_diff": float(rel.max()),
            "corr": float(a.corr(b)) if a.std() > 0 and b.std() > 0 else 1.0,
        })
    res = pd.DataFrame(rows).sort_values("max_abs_diff", ascending=False).reset_index(drop=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coins", nargs="+", default=["BTCUSDT", "ETHUSDT", "SOLUSDT"])
    ap.add_argument("--n-live", type=int, default=DEFAULT_N_LIVE)
    ap.add_argument("--k", type=int, default=DEFAULT_K)
    ap.add_argument("--tol", type=float, default=1e-4, help="ambang max_abs_diff dianggap PARITY")
    args = ap.parse_args()

    model_feats = _load_model_features()
    print(f"Model features dibandingkan: {len(model_feats)} | n_live={args.n_live} k={args.k} tol={args.tol}")
    print("=" * 78)

    all_res = []
    for coin in args.coins:
        print(f"\n[{coin}] full-history vs {args.n_live}-bar window (last {args.k} bars)")
        res = audit_coin(coin, args.n_live, args.k, model_feats)
        if res.empty:
            continue
        res["coin"] = coin
        all_res.append(res)
        # tampilkan 8 fitur dengan diff terbesar
        top = res.head(8)
        print(f"  {'feature':28s} {'max_absd':>12s} {'mean_absd':>12s} {'corr':>7s}")
        for _, r in top.iterrows():
            flag = "  OK" if r["max_abs_diff"] <= args.tol else "  <-- DIVERGEN"
            print(f"  {r['feature']:28s} {r['max_abs_diff']:12.6g} {r['mean_abs_diff']:12.6g} {r['corr']:7.4f}{flag}")
        n_div = int((res["max_abs_diff"] > args.tol).sum())
        print(f"  -> {n_div}/{len(res)} fitur DIVERGEN (max_absd>{args.tol})")

    if all_res:
        comb = pd.concat(all_res, ignore_index=True)
        worst = comb.sort_values("max_abs_diff", ascending=False).head(15)
        print("\n" + "=" * 78)
        print("RINGKASAN — 15 fitur paling divergen lintas koin:")
        print(f"  {'coin':12s} {'feature':28s} {'max_absd':>12s} {'corr':>7s}")
        for _, r in worst.iterrows():
            flag = "OK" if r["max_abs_diff"] <= args.tol else "DIVERGEN"
            print(f"  {r['coin']:12s} {r['feature']:28s} {r['max_abs_diff']:12.6g} {r['corr']:7.4f}  {flag}")
        n_div_total = int((comb["max_abs_diff"] > args.tol).sum())
        print(f"\nTotal pasangan (koin×fitur) divergen: {n_div_total}/{len(comb)}")
        out = MODEL_DIR / "runs" / "ic32_regime_v2" / "feature_parity_audit.csv"
        comb.to_csv(out, index=False)
        print(f"Saved: {out}")


if __name__ == "__main__":
    main()
