"""
tools/ops/compare_oos_live_signals.py

Bandingkan sinyal LIVE (VPS app.db) vs reproduksi OOS riset — per-bar, per-fitur.
Toggle spot-confirm/regime-routing/regime-disable di reproduksi ikut
models/inference_config.json (live SSOT) secara dinamis, BUKAN hardcode "stack
penuh" — kalau live "polos" (semua OFF, sejak 2026-07-14) reproduksi ikut polos.
Beda dari compare_holdout_live.py (agregat WR/PF trade-level, sekarang juga
rusak/stale): tool ini bandingkan NILAI tiap fitur satu-satu (bukan cuma
ada/tidak), supaya gap seperti market-panel corruption / spot-confirm
staleness / swing-sidedness yang beberapa kali ketemu manual sepanjang sesi
ini bisa kedeteksi otomatis.

Usage:
  python tools/ops/compare_oos_live_signals.py --start-date 2026-07-11 --end-date 2026-07-12
  python tools/ops/compare_oos_live_signals.py --start-date 2026-07-12 --coins BTCUSDT,ETHUSDT

Butuh SSH ke VPS (sama seperti live_db_bridge.py) -- env SWINT_VPS_HOST/USER/SSH_KEY
kalau beda dari default.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import joblib
import numpy as np
import pandas as pd

from config import (
    HOLDOUT_DIR, OOS_START,
)
from core.utils import ensure_utc_index
from core.cascade_utils import apply_spot_confirm_fusion_pre
from core.spottrade_confirm import get_spot_confirm_h1
from model.eval.constants import DYNAMIC_GUARDIAN_FEATS
from model.regime.thresholds import build_regime_thresholds
from model.stacks.load import load_stack

VPS_HOST = os.environ.get("SWINT_VPS_HOST", "217.15.160.127")
VPS_USER = "root"
VPS_APP_DIR = "/home/swint/swint_tradev2"
WITA = "Asia/Makassar"

COINS_18 = [
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT", "ADAUSDT",
    "DOGEUSDT", "AVAXUSDT", "LINKUSDT", "DOTUSDT", "LTCUSDT", "TRXUSDT",
    "ATOMUSDT", "UNIUSDT", "NEARUSDT", "FILUSDT", "ETCUSDT", "BCHUSDT",
]

RUN_DIR_OPT2 = REPO / "models" / "runs" / "opt2_plus_trend_18coin_iso37f"
RUN_DIR_TREND = REPO / "models" / "runs" / "lgbm37f_trend"
GUARD_DIR = REPO / "models" / "runs" / "guard_opt2_plus_trend_hmm_18coin"
LABEL_DIR = HOLDOUT_DIR / "labeled"

LONG, SHORT, FLAT = 2, 0, 1
STATE_NAMES = {0: "TRENDING_DOWN", 1: "RANGING_LOW_VOL", 2: "RANGING_HIGH_VOL", 3: "TRENDING_UP"}

# Threshold/spot-confirm dibaca dari model/stacks/fs38_28f/stack.json (SSOT) -- JANGAN
# hardcode di sini lagi. Insiden 2026-07-14: konstanta lama (BASE=0.65) basi vs live
# (0.70 sejak 2026-07-13) tapi kebetulan tidak kena kasus borderline saat ditemukan,
# jadi "100% match" yang dilaporkan waktu itu tidak sepenuhnya teruji.
_STACK = load_stack("fs38_28f")
BASE, DELTA = _STACK.hmm_base, _STACK.hmm_delta
THR = build_regime_thresholds(BASE, DELTA)
AGREE_BOOST = _STACK.spot_confirm_agree_boost
OPPOSITE_PEN = _STACK.spot_confirm_opposite_pen
SPOT_THR = _STACK.spot_confirm_threshold

# Toggle fusion/routing/disable HARUS ikut models/inference_config.json (live SSOT),
# BUKAN stack.json -- keduanya bisa beda (lihat model/stacks/fs38_28f/CLAUDE.md).
# Insiden 2026-07-19: tool ini sempat SELALU reproduksi "stack penuh" (spot-confirm+
# regime-routing+regime-disable ON) meski live sudah "polos" (semua OFF) sejak
# 2026-07-14 -- dampak empiris kecil (1/2467 sinyal beda arah) tapi tetap gap
# metodologi nyata, diperbaiki di sini biar otomatis ikut kalau toggle live berubah lagi.
_INFCFG = json.load(open(REPO / "models" / "inference_config.json", encoding="utf-8"))
SPOT_CONFIRM_ENABLED = _INFCFG["spot_confirm"]["enabled"]
REGIME_ROUTING_ENABLED = _INFCFG["regime_model_routing"]["enabled"]
REGIME_ROUTING_TREND_STATES = frozenset(_INFCFG["regime_model_routing"]["trend_states"])
REGIME_DISABLE_ENABLED = _INFCFG["regime_disable"]["enabled"]
BLOCK_LONG_IN = tuple(_INFCFG["regime_disable"]["block_long_in"]) if REGIME_DISABLE_ENABLED else ()

FEATURE_DIFF_REL_THRESHOLD = 0.05  # 5% relatif -- flag kalau lebih dari ini


def pull_live_signals(start_date: str, end_date: str, coins: list[str]) -> pd.DataFrame:
    """SSH ke VPS, tarik signal table utk rentang tanggal (WITA) -- query langsung,
    tidak sync seluruh DB (~126MB) spt live_db_bridge.py, lebih cepat utk cek sempit."""
    win_start = pd.Timestamp(start_date, tz=WITA).tz_convert("UTC")
    win_end = (pd.Timestamp(end_date, tz=WITA) + pd.Timedelta(days=1)).tz_convert("UTC")
    coin_list = ",".join(f"'{c}'" for c in coins)

    py_script = f"""
import sqlite3, json
con = sqlite3.connect('{VPS_APP_DIR}/instance/app.db')
con.row_factory = sqlite3.Row
rows = con.execute('''
    SELECT s.id, c.symbol, s.direction, s.confidence, s.entry_price, s.tp_price, s.sl_price,
           s.atr_at_signal, s.h4_swing_high, s.h4_swing_low, s.feature_snapshot,
           s.signal_time, s.created_at, s.entry_reason
    FROM signal s JOIN coin c ON s.coin_id = c.id
    WHERE c.symbol IN ({coin_list})
      AND s.created_at >= '{win_start.strftime('%Y-%m-%d %H:%M:%S')}'
      AND s.created_at <  '{win_end.strftime('%Y-%m-%d %H:%M:%S')}'
    ORDER BY s.created_at
''').fetchall()
out = [dict(r) for r in rows]
print(json.dumps(out, default=str))
"""
    cmd = ["ssh", f"{VPS_USER}@{VPS_HOST}", f"python3 -c \"{py_script}\""]
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    if res.returncode != 0:
        raise RuntimeError(f"SSH query gagal: {res.stderr}")
    data = json.loads(res.stdout)
    df = pd.DataFrame(data)
    if df.empty:
        return df
    df["created_at"] = pd.to_datetime(df["created_at"], utc=True)
    df["signal_time"] = pd.to_datetime(df["signal_time"], utc=True)
    return df


def build_research_stack():
    lgbm_opt2 = joblib.load(RUN_DIR_OPT2 / "lgbm.pkl")
    feats_opt2 = json.load(open(RUN_DIR_OPT2 / "features.json", encoding="utf-8"))
    lgbm_trend = joblib.load(RUN_DIR_TREND / "lgbm.pkl")
    feats_trend = json.load(open(RUN_DIR_TREND / "features.json", encoding="utf-8"))
    g_feats = json.load(open(GUARD_DIR / "guardian_features.json", encoding="utf-8"))
    static_names = [f for f in g_feats if f not in DYNAMIC_GUARDIAN_FEATS]
    return dict(
        lgbm_opt2=lgbm_opt2, feats_opt2=feats_opt2,
        lgbm_trend=lgbm_trend, feats_trend=feats_trend,
        g_feats=g_feats, static_names=static_names,
    )


def gate(p0, p2, state, block_long_regimes=BLOCK_LONG_IN):
    tl, ts = THR.get(int(state), (0.65, 0.65))
    regime = STATE_NAMES.get(int(state), "?")
    if p2 >= tl and regime not in block_long_regimes:
        return LONG
    if p0 >= ts:
        return SHORT
    return FLAT


def reproduce_research_bars(coins: list[str], stack: dict) -> dict:
    """Untuk tiap koin: load fitur holdout, hitung proba kedua model + regime +
    spot-confirm + decision per bar. Return dict[coin] -> DataFrame (index=waktu UTC)."""
    out = {}
    for coin in coins:
        feat_path = LABEL_DIR / f"{coin}_features_v3.parquet"
        regime_path = LABEL_DIR / f"{coin}_regime_h1.parquet"
        if not feat_path.exists():
            print(f"  [WARN] {coin}: fitur holdout tidak ada -- {feat_path}")
            continue
        df = ensure_utc_index(pd.read_parquet(feat_path)).sort_index()
        df = df[df.index >= OOS_START]
        if regime_path.exists():
            reg = ensure_utc_index(pd.read_parquet(regime_path))
            df["hmm_regime_enc"] = reg["hmm_regime_enc"].reindex(df.index).ffill().fillna(1)
        n = len(df)
        if n < 5:
            continue
        states = df["hmm_regime_enc"].fillna(1).astype(int).values

        X_opt2 = np.zeros((n, len(stack["feats_opt2"])), dtype=np.float64)
        for i, col in enumerate(stack["feats_opt2"]):
            if col in df.columns:
                X_opt2[:, i] = df[col].ffill().fillna(0).values
        proba_opt2 = stack["lgbm_opt2"].predict_proba(X_opt2)
        p0_opt2, p2_opt2 = proba_opt2[:, 0].astype(np.float32), proba_opt2[:, 2].astype(np.float32)

        if SPOT_CONFIRM_ENABLED:
            spot = get_spot_confirm_h1(coin, df.index)
            p0_opt2_fused, p2_opt2_fused = apply_spot_confirm_fusion_pre(
                p0_opt2, p2_opt2, spot["spot_proba_long"].values, spot["spot_proba_short"].values,
                spot_threshold=SPOT_THR, agree_boost=AGREE_BOOST, opposite_pen=OPPOSITE_PEN,
            )
        else:
            p0_opt2_fused, p2_opt2_fused = p0_opt2, p2_opt2

        if REGIME_ROUTING_ENABLED:
            X_trend = np.zeros((n, len(stack["feats_trend"])), dtype=np.float64)
            for i, col in enumerate(stack["feats_trend"]):
                if col in df.columns:
                    X_trend[:, i] = df[col].ffill().fillna(0).values
            proba_trend = stack["lgbm_trend"].predict_proba(X_trend)
            p0_trend, p2_trend = proba_trend[:, 0].astype(np.float32), proba_trend[:, 2].astype(np.float32)
            used_trend = np.isin(states, list(REGIME_ROUTING_TREND_STATES))
            p0_final = np.where(used_trend, p0_trend, p0_opt2_fused)
            p2_final = np.where(used_trend, p2_trend, p2_opt2_fused)
        else:
            used_trend = np.zeros(n, dtype=bool)
            p0_final, p2_final = p0_opt2_fused, p2_opt2_fused

        decisions = np.full(n, FLAT, dtype=np.int32)
        for i in range(n):
            decisions[i] = gate(float(p0_final[i]), float(p2_final[i]), states[i])

        meta = pd.DataFrame({
            "hmm_regime_enc": states,
            "used_trend_model": used_trend,
            "p0_final": p0_final, "p2_final": p2_final,
            "decision": decisions,
        }, index=df.index)
        out[coin] = df.join(meta.drop(columns=["hmm_regime_enc"]) if "hmm_regime_enc" in df.columns else meta)
        if "hmm_regime_enc" not in out[coin].columns:
            out[coin]["hmm_regime_enc"] = states
    return out


def _prov_summary(live_feats: dict) -> dict:
    """Fase 6 (kebijakan futures-only): ekstrak provenance dari `feature_snapshot`
    (`_prov`/`_prov_violations`, sudah ditulis `generate_signals.py` sejak Fase 1)
    supaya audit OOS-vs-live bisa LANGSUNG menyalahkan sumber data begitu ketahuan
    beda, bukan menebak lalu cross-check candle mentah Binance manual satu-satu
    spt insiden 27-28 Juli (baru ketahuan lewat audit berhari-hari, bukan lewat
    laporan tool ini)."""
    prov = live_feats.get("_prov") or {}
    violations = live_feats.get("_prov_violations") or []
    return {
        "prov_kline_1h_market_type": (prov.get("kline_1h") or {}).get("market_type"),
        "prov_kline_4h_market_type": (prov.get("kline_4h") or {}).get("market_type"),
        "prov_oi_source": prov.get("oi_source"),
        "prov_lsr_source": prov.get("lsr_source"),
        "prov_violations": ";".join(v.get("code", "?") for v in violations) or None,
    }


def match_and_diff(live_df: pd.DataFrame, research: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cocokkan tiap sinyal live ke bar riset TERDEKAT (<=1h), bandingkan fitur.
    Return (matched_diff_rows, unmatched_rows)."""
    matched_rows = []
    unmatched_rows = []
    DEC_NAME = {LONG: "LONG", SHORT: "SHORT", FLAT: "FLAT"}

    for _, sig in live_df.iterrows():
        coin = sig["symbol"]
        if coin not in research:
            unmatched_rows.append({
                "coin": coin, "live_time": sig["created_at"], "reason": "coin_missing_in_research",
                "live_direction": sig["direction"],
            })
            continue
        rdf = research[coin]
        t = sig["created_at"]
        # Bar diindeks by OPEN time; sinyal live dibuat SETELAH bar CLOSE (+1h) --
        # jadi bar riset yg relevan adalah yg CLOSE-nya <= t, bukan OPEN-nya <= t.
        # (bug off-by-one-hour yg sama persis spt spot-confirm indexing sebelumnya)
        idx = rdf.index[rdf.index <= (t - pd.Timedelta("1h"))]
        if len(idx) == 0:
            unmatched_rows.append({
                "coin": coin, "live_time": t, "reason": "no_research_bar_before",
                "live_direction": sig["direction"],
            })
            continue
        bar_time = idx[-1]
        if (t - pd.Timedelta("1h") - bar_time) > pd.Timedelta("70min"):
            unmatched_rows.append({
                "coin": coin, "live_time": t, "reason": f"nearest_bar_too_far ({bar_time})",
                "live_direction": sig["direction"],
            })
            continue

        row = rdf.loc[bar_time]
        research_dir = DEC_NAME.get(int(row["decision"]), "FLAT")
        live_dir = sig["direction"]

        # Parse feature_snapshot lebih awal (dulu di bawah) -- provenance (Fase 6)
        # dibutuhkan sblm cek direction_mismatch supaya baris itu ikut menyertakan
        # sumber data, bukan cuma baris feature diff.
        try:
            live_feats = json.loads(sig["feature_snapshot"]) if sig.get("feature_snapshot") else {}
        except (json.JSONDecodeError, TypeError):
            live_feats = {}
        prov_info = _prov_summary(live_feats)

        # Kedua arah dicek -- BUKAN cuma "live ambil trade tapi riset FLAT". Insiden
        # 2026-07-19: kasus sebaliknya (live FLAT krn borderline < threshold, riset
        # SHORT/LONG krn confidence dihitung ulang sedikit lebih tinggi) sempat lolos
        # tak terdeteksi krn cek lama cuma syarat live_dir in (LONG, SHORT).
        if research_dir != live_dir and (live_dir in ("LONG", "SHORT") or research_dir in ("LONG", "SHORT")):
            unmatched_rows.append({
                "coin": coin, "live_time": t, "bar_time_research": bar_time,
                "reason": "direction_mismatch",
                "live_direction": live_dir, "research_direction": research_dir,
                "research_p0": round(float(row["p0_final"]), 4),
                "research_p2": round(float(row["p2_final"]), 4),
                "used_trend_model": bool(row["used_trend_model"]),
                "hmm_regime": STATE_NAMES.get(int(row["hmm_regime_enc"]), "?"),
                **prov_info,
            })

        # ── Diff fitur nilai (bukan cuma presence) ──────────────────────────
        for col in rdf.columns:
            if col.startswith("_") or col in ("decision", "used_trend_model", "p0_final", "p2_final"):
                continue
            if col not in live_feats:
                continue
            live_val = live_feats[col]
            research_val = row[col]
            if live_val is None or (isinstance(research_val, float) and np.isnan(research_val)):
                continue
            try:
                live_f = float(live_val)
                research_f = float(research_val)
            except (TypeError, ValueError):
                continue
            denom = max(abs(live_f), abs(research_f), 1e-9)
            rel_diff = abs(live_f - research_f) / denom
            if rel_diff > FEATURE_DIFF_REL_THRESHOLD:
                matched_rows.append({
                    "coin": coin, "live_time": t, "bar_time_research": bar_time,
                    "feature": col, "live_value": live_f, "research_value": research_f,
                    "rel_diff": round(rel_diff, 4),
                    **prov_info,
                })

    return pd.DataFrame(matched_rows), pd.DataFrame(unmatched_rows)


def main():
    ap = argparse.ArgumentParser(description="Bandingkan sinyal live vs reproduksi OOS riset (per-fitur)")
    ap.add_argument("--start-date", required=True, help="WITA, YYYY-MM-DD")
    ap.add_argument("--end-date", default=None, help="WITA, YYYY-MM-DD (default = start-date)")
    ap.add_argument("--coins", default=None, help="Comma-separated, default 18-koin")
    ap.add_argument("--out-dir", default=str(REPO / "reports"))
    args = ap.parse_args()

    end_date = args.end_date or args.start_date
    coins = args.coins.split(",") if args.coins else COINS_18

    print(f"[1/4] Tarik sinyal live {args.start_date} s.d. {end_date} WITA, {len(coins)} koin...")
    live_df = pull_live_signals(args.start_date, end_date, coins)
    print(f"  -> {len(live_df)} sinyal live ditemukan")
    if live_df.empty:
        print("Tidak ada sinyal live di rentang ini -- berhenti.")
        return

    print("[2/4] Load model stack riset (opt2 + lgbm_trend + guardian feat list)...")
    stack = build_research_stack()

    print(f"[3/4] Reproduksi bar riset (holdout-test, {len(coins)} koin)...")
    research = reproduce_research_bars(list(live_df["symbol"].unique()), stack)
    print(f"  -> {len(research)} koin berhasil direproduksi")

    print("[4/4] Cocokkan & diff fitur...")
    matched_diff, unmatched = match_and_diff(live_df, research)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)
    tag = f"{args.start_date}_{end_date}"
    matched_path = out_dir / f"oos_live_feature_diff_{tag}.csv"
    unmatched_path = out_dir / f"oos_live_unmatched_{tag}.csv"
    matched_diff.to_csv(matched_path, index=False)
    unmatched.to_csv(unmatched_path, index=False)

    print(f"\n{'=' * 90}\nRINGKASAN\n{'=' * 90}")
    print(f"Total sinyal live: {len(live_df)}")
    print(f"Sinyal ter-flag beda arah (live vs riset): {(unmatched['reason'] == 'direction_mismatch').sum() if not unmatched.empty else 0}")
    print(f"Sinyal tak ter-match ke bar riset: {(unmatched['reason'] != 'direction_mismatch').sum() if not unmatched.empty else 0}")
    print(f"Baris fitur ter-flag beda nilai (>{FEATURE_DIFF_REL_THRESHOLD:.0%}): {len(matched_diff)}")

    if not matched_diff.empty:
        print("\n--- Fitur paling sering/paling besar bedanya (top 15) ---")
        summary = matched_diff.groupby("feature").agg(
            n_flag=("rel_diff", "size"), avg_rel_diff=("rel_diff", "mean"), max_rel_diff=("rel_diff", "max"),
        ).sort_values("n_flag", ascending=False).head(15)
        print(summary.to_string())

    if not unmatched.empty:
        print("\n--- Sinyal beda arah (live vs riset), max 15 ---")
        dm = unmatched[unmatched["reason"] == "direction_mismatch"]
        if not dm.empty:
            print(dm.head(15).to_string(index=False))

    print(f"\nSaved -> {matched_path}")
    print(f"Saved -> {unmatched_path}")


if __name__ == "__main__":
    main()
