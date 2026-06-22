"""
pipeline/live_holdout_compare.py
=================================
Bandingkan signal live (dari VPS DB) dengan research-pipeline inference
pada BAR YANG SAMA — verifikasi bahwa live dan simulasi menghasilkan output identik.

Cara kerja:
  1. Pull live DB dari VPS → ambil batch signal terbaru
  2. Deteksi bar_utc yang dipakai live (signal_time - floor ke jam - 1 jam)
  3. Fetch klines dari Binance public API untuk bar tersebut
  4. Jalankan research pipeline: engineer_features → apply_training_parity → inference
  5. Bandingkan: direction, confidence, semua fitur, tiap intermediate score

Usage:
    # Otomatis ambil signal terbaru dari live DB:
    python pipeline/live_holdout_compare.py

    # Paksa bar tertentu (UTC):
    python pipeline/live_holdout_compare.py --bar-utc "2026-06-20 08:00"

    # Hanya coin tertentu:
    python pipeline/live_holdout_compare.py --coins BTCUSDT ETHUSDT
"""

import argparse
import json
import sys
import time
import warnings
from pathlib import Path
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
import requests

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

# ─── Config & Models ─────────────────────────────────────────────────────────
from config import ALL_COINS as COINS
from core.features import engineer_features, compute_genuine_v2_derived_features
from core.regime import predict_hmm
from pipeline.backtest_utils import apply_training_parity, hierarchical_predict

MODEL_DIR      = ROOT / "models"
RUN_DIR        = MODEL_DIR / "runs" / "ic32_regime_v1"
HMM_DIR        = MODEL_DIR / "hmm"
FEAT_COLS_PATH = MODEL_DIR / "feature_cols_v2.json"
LIVE_CACHE     = ROOT / "data" / "live_cache"

FAPI_BASE = "https://fapi.binance.com/fapi/v1/klines"
N_BARS    = 600   # cukup untuk semua rolling window


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _fetch_binance(symbol: str, interval: str, limit: int = 600) -> pd.DataFrame:
    """Fetch klines dari Binance Futures public API. Tanpa auth."""
    resp = requests.get(
        FAPI_BASE,
        params={"symbol": symbol, "interval": interval, "limit": limit},
        timeout=20,
    )
    resp.raise_for_status()
    raw = resp.json()
    if not raw:
        return pd.DataFrame()

    cols = ["open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_vol", "n_trades",
            "taker_buy_base", "taker_buy_quote", "ignore"]
    df = pd.DataFrame(raw, columns=cols)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df = df.set_index("open_time")

    pfx = interval.replace("h", "h_").replace("4h_", "4h_")
    # Normalise prefix sama dengan data_service
    pfx_map = {"1h": "1h", "4h": "4h"}
    p = pfx_map.get(interval, interval)

    rename = {
        "open":           f"{p}_open",
        "high":           f"{p}_high",
        "low":            f"{p}_low",
        "close":          f"{p}_close",
        "volume":         f"{p}_volume",
        "taker_buy_base": f"{p}_taker_buy_volume",
        "n_trades":       f"{p}_num_trades",
        "quote_vol":      f"{p}_quote_volume",
    }
    df = df.rename(columns=rename)
    numeric_cols = [c for c in df.columns if c in rename.values()]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    return df[[c for c in rename.values() if c in df.columns]]


def _load_feat_cols() -> list[str]:
    data = json.loads(FEAT_COLS_PATH.read_text())
    if isinstance(data, list):
        return data
    return data.get("feature_cols", data.get("feat_cols", []))


def _load_models():
    # Baca paths dari inference_config (source of truth)
    inf_cfg_path = MODEL_DIR / "inference_config.json"
    inf_cfg = json.loads(inf_cfg_path.read_text()) if inf_cfg_path.exists() else {}
    models_cfg = inf_cfg.get("models", {})

    lgbm_file = models_cfg.get("lgbm", "lgbm_baseline.pkl")
    gdn_file  = models_cfg.get("guardian", "guardian_best.pkl")
    gs_file   = models_cfg.get("guardian_scaler", "guardian_scaler.pkl")

    # Coba RUN_DIR dulu, fallback ke MODEL_DIR
    def _load(fname):
        p = RUN_DIR / fname
        if p.exists():
            return joblib.load(p)
        return joblib.load(MODEL_DIR / fname)

    lgbm = _load(lgbm_file)
    gdn  = _load(gdn_file)
    gs   = _load(gs_file)
    gpath = RUN_DIR / "b_dir_combined_frozen.json"
    cfg   = json.loads(gpath.read_text()) if gpath.exists() else {}
    return lgbm, gdn, gs, cfg


def _load_lstm():
    import torch
    sys.path.insert(0, str(ROOT / "core"))
    from core.models import TradingLSTM
    lstm_dir = RUN_DIR
    weights  = lstm_dir / "lstm_best.pt"
    if not weights.exists():
        weights = MODEL_DIR / "lstm_best.pt"
    if not weights.exists():
        return None, None, []
    # Feature list: ic32 swing complement v2 (11 feats, matches lstm_best.pt checkpoint)
    for feat_path in [
        MODEL_DIR / "runs" / "ic32_lstm_swing_complement_v2" / "ic32_lstm_swing_complement_v2_features.json",
        lstm_dir / "lstm_v4_selected_features.json",
    ]:
        if feat_path.exists():
            break
    lstm_feats = json.loads(feat_path.read_text()) if feat_path.exists() else []
    ckpt = torch.load(str(weights), map_location="cpu", weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    # Baca arch dari checkpoint (W_ih shape: [4*hidden, n_feat])
    w_ih = state.get("lstm.cells.0.W_ih")
    if w_ih is not None:
        hidden_size = w_ih.shape[0] // 4
        n_feat_ckpt = w_ih.shape[1]
    else:
        hidden_size = 128
        n_feat_ckpt = None
    # Selalu pakai arch dari checkpoint (bukan panjang feature list)
    n_feat = n_feat_ckpt or ckpt.get("n_features", 16)
    model = TradingLSTM(n_features=n_feat, hidden_size=hidden_size, num_layers=2, num_classes=3)
    model.load_state_dict(state, strict=True)
    model.eval()
    # Cari scaler: RUN_DIR → MODEL_DIR → inference_config
    for sp in [lstm_dir / "lstm_scaler.pkl", MODEL_DIR / "lstm_scaler.pkl"]:
        if sp.exists():
            lstm_scaler = joblib.load(sp)
            break
    else:
        lstm_scaler = None
    return model, lstm_scaler, lstm_feats


def _load_hmm(symbol: str):
    pkl = HMM_DIR / f"{symbol}_hmm.pkl"
    if not pkl.exists():
        return None, None
    data = joblib.load(pkl)
    if isinstance(data, dict):
        return data.get("model"), data.get("state_map")
    if isinstance(data, tuple) and len(data) == 3:
        return data[0], data[2]
    return data, None


def _get_hmm_regime(symbol: str, df_4h: pd.DataFrame, df_1h_index: pd.Index) -> pd.Series:
    """Predict HMM regime dari df_4h, align ke df_1h index."""
    model, state_map = _load_hmm(symbol)
    from core.regime import REGIME_ENC_4 as REGIME_ENC  # TRENDING_DOWN=0,RLV=1,RHV=2,TUP=3
    default_enc = pd.Series(1, index=df_1h_index, name="hmm_regime_enc")
    if model is None:
        return default_enc
    h4_input = df_4h[["4h_close", "4h_volume"]].rename(
        columns={"4h_close": "close", "4h_volume": "volume"}
    )
    try:
        labels = predict_hmm(model, h4_input.tail(500), state_map)
        enc    = pd.Series(labels, index=h4_input.tail(500).index).map(REGIME_ENC).fillna(1).astype("int32")
        aligned = enc.reindex(df_1h_index, method="ffill").fillna(1).astype("int32")
        return aligned
    except Exception as e:
        print(f"  [WARN] HMM gagal untuk {symbol}: {e}")
        return default_enc


def _load_positioning(symbol: str, df_1h_index: pd.Index) -> dict[str, pd.Series]:
    """Load positioning parquets dari dev machine (atau live cache)."""
    pos_dir = ROOT / "data" / "positioning"
    result  = {}
    files   = {
        "long_short_ratio": f"{symbol}_global_ls.parquet",
        "oi_delta":         f"{symbol}_bybit_oi.parquet",
        "taker_ratio":      f"{symbol}_taker_ratio.parquet",
    }
    for key, fname in files.items():
        fp = pos_dir / fname
        if fp.exists():
            try:
                tmp = pd.read_parquet(fp)
                tmp.index = pd.to_datetime(tmp.index, utc=True)
                col = tmp.columns[0]
                s   = tmp[col].astype(float)
                result[key] = s.reindex(df_1h_index, method="ffill").fillna(method="bfill")
            except Exception:
                pass
    return result


def _get_hmm_config(cfg: dict) -> dict:
    """Extract per-state HMM threshold config."""
    ra = cfg.get("regime_alignment", {})
    hmm_raw = ra.get("hmm_thresholds", cfg.get("hmm_thresholds", {}))
    out = {}
    for k, v in hmm_raw.items():
        try:
            out[int(k)] = (float(v[0]), float(v[1]))
        except Exception:
            pass
    return out


# ─── Live DB Pull ─────────────────────────────────────────────────────────────

def pull_latest_signals(n_recent: int = 50) -> pd.DataFrame:
    """Pull signal terbaru dari live DB (auto SCP dari VPS jika perlu)."""
    try:
        from tools.live_db_bridge import pull_live_db, LOCAL_DB, _connect
        pull_live_db()
    except Exception as e:
        print(f"  [WARN] Gagal pull dari VPS: {e} — pakai cache lokal")

    from tools.live_db_bridge import LOCAL_DB, _connect
    con = _connect()
    df  = pd.read_sql_query(
        f"""
        SELECT s.id, s.signal_time, s.direction, s.confidence,
               s.feature_snapshot, s.entry_reason,
               c.symbol AS coin
        FROM signal s
        JOIN coin c ON s.coin_id = c.id
        ORDER BY s.signal_time DESC
        LIMIT {n_recent * 21}
        """, con
    )
    con.close()
    df["signal_time"] = pd.to_datetime(df["signal_time"], utc=True, errors="coerce")
    return df


def detect_bar_utc(signals_df: pd.DataFrame) -> pd.Timestamp | None:
    """
    Deteksi bar UTC yang dipakai dari batch signal terbaru.
    Live fires di HH:05 WITA dan drops bar HH:00 (5 mnt) → pakai bar HH-1:00.
    bar_utc = floor(signal_time_utc, hour) - 1h
    """
    if signals_df.empty:
        return None
    latest_ts = signals_df["signal_time"].max()
    floored   = latest_ts.floor("h")
    bar_utc   = floored - pd.Timedelta(hours=1)
    return bar_utc


# ─── Research Pipeline per Coin ───────────────────────────────────────────────

def run_research_inference(symbol: str, bar_utc: pd.Timestamp,
                           lgbm_model, lstm_model, lstm_scaler, lstm_feats,
                           feat_cols: list[str], hmm_cfg: dict) -> dict:
    """
    Jalankan full research pipeline untuk satu coin pada bar tertentu.
    Returns dict: features + model outputs.
    """
    result = {"symbol": symbol, "bar_utc": str(bar_utc), "error": None}

    # 1. Fetch klines + BTC close untuk relative_strength_z
    try:
        df_1h = _fetch_binance(symbol, "1h", N_BARS)
        df_4h = _fetch_binance(symbol, "4h", N_BARS // 4 + 20)
        if symbol != "BTCUSDT":
            df_btc = _fetch_binance("BTCUSDT", "1h", N_BARS)
            if not df_btc.empty and "1h_close" in df_btc.columns:
                df_1h["btc_close"] = df_btc["1h_close"].reindex(df_1h.index).ffill()
    except Exception as e:
        result["error"] = f"Fetch gagal: {e}"
        return result

    if df_1h.empty or len(df_1h) < 100:
        result["error"] = "Klines tidak cukup"
        return result

    # 2. Pastikan hanya sampai bar_utc (inklusif)
    bar_utc_ts = pd.Timestamp(bar_utc)
    if bar_utc_ts.tzinfo is None:
        bar_utc_ts = bar_utc_ts.tz_localize("UTC")
    else:
        bar_utc_ts = bar_utc_ts.tz_convert("UTC")
    df_1h = df_1h[df_1h.index <= bar_utc_ts]
    df_4h = df_4h[df_4h.index <= bar_utc_ts]

    if df_1h.empty or df_1h.index[-1] != bar_utc_ts:
        # Coba toleransi 1 menit
        close_idx = df_1h.index[df_1h.index <= bar_utc_ts + pd.Timedelta(minutes=1)]
        if close_idx.empty:
            result["error"] = f"Bar {bar_utc} tidak ditemukan di klines"
            return result
        df_1h = df_1h[df_1h.index <= close_idx[-1]]

    result["n_1h_bars"]  = len(df_1h)
    result["last_bar"]   = str(df_1h.index[-1])

    # 3. Taker sell volume
    if "1h_volume" in df_1h.columns and "1h_taker_buy_volume" in df_1h.columns:
        df_1h["1h_taker_sell_volume"] = (
            df_1h["1h_volume"].astype(float) - df_1h["1h_taker_buy_volume"].astype(float)
        ).clip(lower=0)

    # 4. Engineer features (research pipeline)
    # Merge 4h OHLCV ke df_1h — sama persis dengan live data_service.py
    # Tanpa ini, engineer_features fallback ke H1 data untuk h4_c_closed, atr_h4 → trend_accel_4h salah
    if not df_4h.empty:
        h4_aligned = df_4h.reindex(
            df_4h.index.union(df_1h.index)
        ).ffill().reindex(df_1h.index)
        for col in ["4h_open", "4h_high", "4h_low", "4h_close", "4h_volume"]:
            if col in h4_aligned.columns:
                df_1h[col] = h4_aligned[col]

    try:
        feat_df = engineer_features(df_1h, symbol=symbol, symbol_id=0, add_label=False)
    except Exception as e:
        result["error"] = f"engineer_features gagal: {e}"
        return result

    if feat_df.empty:
        result["error"] = "feat_df kosong setelah engineer_features"
        return result

    # 5. H4 regime
    hmm_enc = _get_hmm_regime(symbol, df_4h, feat_df.index)
    feat_df["hmm_regime_enc"] = hmm_enc.reindex(feat_df.index).fillna(1).astype(int)

    # 6. Compute genuine_v2 derived features (sama seperti live)
    feat_df = compute_genuine_v2_derived_features(feat_df)

    # 7. Training parity (LSR clip + CVD clamp — sama seperti live)
    feat_df = apply_training_parity(feat_df)

    # 8. Ambil baris terakhir (= bar_utc)
    last = feat_df.iloc[[-1]]
    result["features"] = {
        col: (float(last[col].iloc[0]) if not pd.isna(last[col].iloc[0]) else None)
        for col in last.columns if col in feat_cols or col.startswith("_")
    }
    result["hmm_regime_enc"] = int(last["hmm_regime_enc"].iloc[0]) if "hmm_regime_enc" in last.columns else 1

    # 9. Run inference — pakai slice panjang agar LSTM punya cukup sequence
    SEQ_LEN = 72  # sama dengan config LSTM_SEQ_LEN
    df_slice = feat_df.tail(SEQ_LEN + 10)  # sedikit margin
    n_slice  = len(df_slice)
    X = np.zeros((n_slice, len(feat_cols)), dtype=np.float64)
    for i, col in enumerate(feat_cols):
        if col in df_slice.columns:
            X[:, i] = df_slice[col].ffill().fillna(0).values.astype(np.float64)

    hmm_enc_val = result["hmm_regime_enc"]
    # Broadcast regime thresholds ke semua baris di slice
    default_tl, default_ts = hmm_cfg.get(-1, (0.50, 0.55))
    thr_l = np.full(n_slice, default_tl)
    thr_s = np.full(n_slice, default_ts)
    for state, (tl, ts) in hmm_cfg.items():
        if state == -1:
            continue
        if hmm_enc_val == state:
            thr_l[:] = tl
            thr_s[:] = ts

    try:
        y_pred, confidence = hierarchical_predict(
            None, lgbm_model, lstm_model, lstm_scaler,
            X, feat_cols, [], df_slice,
            model_dir=RUN_DIR,
            lstm_feat_cols=lstm_feats,
            per_bar_thr_long=thr_l,
            per_bar_thr_short=thr_s,
        )
        dir_map = {0: "SHORT", 1: "FLAT", 2: "LONG"}
        result["direction"]  = dir_map.get(int(y_pred[-1]), "FLAT")
        result["confidence"] = float(confidence[-1])
    except Exception as e:
        result["error_inference"] = str(e)
        result["direction"]  = "ERROR"
        result["confidence"] = 0.0

    return result


# ─── Comparison ───────────────────────────────────────────────────────────────

def compare(live_sig: dict, research: dict, feat_cols: list[str]) -> None:
    sym  = research["symbol"]
    sep  = "=" * 80

    print(f"\n{sep}")
    print(f"  {sym}  |  Bar: {research['bar_utc']}")
    print(sep)

    if research.get("error"):
        print(f"  [ERROR research] {research['error']}")
        return

    # Debug: signal time + bar info
    live_snap = {}
    try:
        import json as _json
        live_snap = _json.loads(live_sig.get("feature_snapshot") or "{}")
    except Exception:
        pass
    print(f"  Live signal_time  : {live_sig.get('signal_time', 'N/A')}")
    print(f"  Research last bar : {research.get('last_bar', 'N/A')}")
    print(f"  Research n_bars   : {research.get('n_1h_bars', 'N/A')}")
    if research.get("error_inference"):
        print(f"  [ERROR inference] {research['error_inference']}")

    # Direction + confidence
    live_dir  = live_sig.get("direction", "?")
    live_conf = live_sig.get("confidence", 0.0)
    res_dir   = research.get("direction", "?")
    res_conf  = research.get("confidence", 0.0)

    match_dir  = live_dir == res_dir
    match_conf = abs(live_conf - res_conf) < 0.02

    print(f"  {'':20s}  {'LIVE':>10}  {'RESEARCH':>10}  {'MATCH':>6}")
    print(f"  {'-'*55}")
    print(f"  {'direction':20s}  {live_dir:>10}  {res_dir:>10}  {'OK' if match_dir else '!! BEDA':>6}")
    print(f"  {'confidence':20s}  {live_conf:>10.4f}  {res_conf:>10.4f}  {'OK' if match_conf else '!! BEDA':>6}")

    # Model intermediates dari live feature_snapshot
    live_fs = live_sig.get("feature_snapshot", {})
    if isinstance(live_fs, str):
        try:
            live_fs = json.loads(live_fs)
        except Exception:
            live_fs = {}

    intermediates = [
        "_lgbm_conf", "_lstm_conf", "_lstm_adj",
        "_score_lgbm", "_score_after_lstm", "_score_final",
        "_thr_entry", "_thr_lgbm_long", "_thr_lgbm_short",
        "_hmm_enc", "_lgbm_decision", "_lstm_decision",
    ]
    print(f"\n  {'-- Intermediate Model (live only) --':}")
    for k in intermediates:
        v = live_fs.get(k)
        if v is not None:
            print(f"  {k:30s}: {v}")

    # Feature values comparison
    res_feats  = research.get("features", {})
    SKIP_COLS  = {"label", "symbol_id"}
    KEY_FEATS  = [
        "cvd_slope_h4", "cvd_momentum_adv", "cvd_div_h4",
        "long_short_ratio", "whale_retail_divergence",
        "vol_ratio_20", "vol_spike_zscore", "ultra_high_vol",
        "funding_rate", "ofi_h4_delta",
        "stochrsi_k", "stochrsi_d",
        "dist_liq_50x_long", "dist_liq_50x_short",
        "hmm_regime_enc",
    ]

    print(f"\n  {'-- Key Feature Values --':}")
    print(f"  {'Feature':35s}  {'Live':>12}  {'Research':>12}  {'Diff':>10}  Status")
    print(f"  {'-'*80}")

    all_feats = list(dict.fromkeys(KEY_FEATS + [c for c in feat_cols if c not in SKIP_COLS]))
    n_match = n_diff = 0

    for feat in all_feats:
        live_val = live_fs.get(feat)
        res_val  = res_feats.get(feat)
        if live_val is None and res_val is None:
            continue
        try:
            lv = float(live_val) if live_val is not None else float("nan")
            rv = float(res_val)  if res_val  is not None else float("nan")
            diff = abs(lv - rv) if not (np.isnan(lv) or np.isnan(rv)) else float("nan")
            tol  = 0.01 if abs(lv) > 1 else 0.001
            ok   = diff <= tol if not np.isnan(diff) else False
            status = "OK" if ok else "!!"
            if ok:
                n_match += 1
            else:
                n_diff  += 1
            # Hanya print baris yang berbeda atau key feats
            if not ok or feat in KEY_FEATS:
                print(f"  {feat:35s}  {lv:>12.5f}  {rv:>12.5f}  {diff:>10.5f}  {status}")
        except Exception:
            if feat in KEY_FEATS:
                print(f"  {feat:35s}  {str(live_val):>12}  {str(res_val):>12}")

    print(f"\n  Match: {n_match} fitur identik, {n_diff} berbeda")
    if n_diff == 0:
        print("  OK SEMUA FITUR IDENTIK — live dan research pipeline aligned!")
    else:
        print(f"  !! Ada {n_diff} fitur yang berbeda — perlu investigasi lebih lanjut")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bar-utc", type=str, default=None,
                        help="Bar UTC yang akan dianalisis, e.g. '2026-06-20 08:00'")
    parser.add_argument("--coins", nargs="*", default=None,
                        help="Subset coin, e.g. BTCUSDT ETHUSDT")
    args = parser.parse_args()

    print("\n=== LIVE vs RESEARCH PIPELINE COMPARISON ===")

    # 1. Pull live signals
    print("\n[1] Pull live DB dari VPS...")
    try:
        signals = pull_latest_signals(n_recent=2000)  # cukup untuk 24 jam × 22 coin
    except Exception as e:
        print(f"  ERROR: {e}")
        sys.exit(1)

    if signals.empty:
        print("  Tidak ada signal ditemukan")
        sys.exit(1)

    # 2. Deteksi bar_utc
    if args.bar_utc:
        bar_utc = pd.Timestamp(args.bar_utc, tz="UTC")
    else:
        bar_utc = detect_bar_utc(signals)

    print(f"  Bar UTC yang dianalisis: {bar_utc}")
    print(f"  (= {(bar_utc + pd.Timedelta(hours=8)).strftime('%Y-%m-%d %H:%M')} WITA)\n")

    # 3. Filter signal di batch ini
    # Bar bar_utc tutup di bar_utc+1h. Signal fire HH:05 → selesai ~HH:10.
    # Jendela: bar_utc+1h:00 sampai bar_utc+1h:29 (hanya satu batch)
    batch_start = bar_utc + pd.Timedelta(hours=1)  # bar close
    batch_end   = batch_start + pd.Timedelta(minutes=29)
    batch_sigs  = signals.sort_values("signal_time")[
        (signals["signal_time"] >= batch_start) &
        (signals["signal_time"] <= batch_end)
    ]
    if batch_sigs.empty:
        # fallback: ambil batch terbaru
        latest_hour = signals["signal_time"].max().floor("h")
        batch_sigs  = signals[signals["signal_time"] >= latest_hour - pd.Timedelta(minutes=30)]

    print(f"  Signal live ditemukan: {len(batch_sigs)} coin")

    # 4. Load models
    print("\n[2] Load models...")
    feat_cols = _load_feat_cols()
    lgbm_model, gdn, gs, run_cfg = _load_models()
    lstm_model, lstm_scaler, lstm_feats = _load_lstm()
    print(f"  LGBM: {RUN_DIR/'lgbm.pkl'}")
    print(f"  LSTM: {'OK' if lstm_model else 'tidak ditemukan'}")
    print(f"  Feat cols: {len(feat_cols)} fitur")

    # Buat hmm_cfg dari frozen config
    hmm_cfg_raw = run_cfg.get("regime_alignment", {}).get("hmm_thresholds", {})
    hmm_cfg = {}
    for k, v in hmm_cfg_raw.items():
        try:
            hmm_cfg[int(k)] = (float(v[0]), float(v[1]))
        except Exception:
            pass
    if not hmm_cfg:
        hmm_cfg = {
            -1: (0.50, 0.55),
            0:  (0.50, 0.55),
            1:  (0.55, 0.60),
            2:  (0.55, 0.60),
            3:  (0.50, 0.55),
        }

    # 5. Tentukan coin yang dianalisis
    target_coins = args.coins or batch_sigs["coin"].tolist()
    if not target_coins:
        target_coins = ["BTCUSDT", "ETHUSDT", "NEARUSDT", "LINKUSDT", "DOGEUSDT"]

    print(f"\n[3] Analisis {len(target_coins)} coin pada bar {bar_utc}...\n")

    # 6. Loop per coin
    summary = []
    for sym in target_coins:
        live_rows = batch_sigs[batch_sigs["coin"] == sym]
        live_sig  = live_rows.iloc[0].to_dict() if not live_rows.empty else {}

        print(f"  Fetching & computing {sym}...", end=" ", flush=True)
        try:
            res = run_research_inference(
                sym, bar_utc,
                lgbm_model, lstm_model, lstm_scaler, lstm_feats,
                feat_cols, hmm_cfg,
            )
            print("OK")
        except Exception as e:
            print(f"ERROR: {e}")
            res = {"symbol": sym, "error": str(e)}

        compare(live_sig, res, feat_cols)

        summary.append({
            "coin":         sym,
            "live_dir":     live_sig.get("direction", "?"),
            "res_dir":      res.get("direction", "?"),
            "live_conf":    live_sig.get("confidence", 0),
            "res_conf":     res.get("confidence", 0),
            "dir_match":    live_sig.get("direction") == res.get("direction"),
            "conf_diff":    abs((live_sig.get("confidence") or 0) - (res.get("confidence") or 0)),
        })
        time.sleep(0.3)  # rate limit

    # 7. Summary table
    print("\n\n" + "=" * 80)
    print("  SUMMARY — Live vs Research")
    print("=" * 80)
    print(f"  {'Coin':15s} {'LiveDir':>8} {'ResDir':>8} {'LiveConf':>9} {'ResConf':>9} {'DirOK':>6} {'ConfDiff':>9}")
    print(f"  {'-' * 65}")
    for s in summary:
        ok = "OK" if s["dir_match"] else "!!"
        print(
            f"  {s['coin']:15s} {s['live_dir']:>8} {s['res_dir']:>8} "
            f"{s['live_conf']:>9.4f} {s['res_conf']:>9.4f} {ok:>6} {s['conf_diff']:>9.4f}"
        )

    total = len(summary)
    matched = sum(1 for s in summary if s["dir_match"])
    print(f"\n  Direction match: {matched}/{total} coin")
    if matched == total:
        print("  OK SEMUA DIRECTION IDENTIK — fixes berhasil!")
    else:
        print(f"  !! {total - matched} coin masih berbeda direction")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
